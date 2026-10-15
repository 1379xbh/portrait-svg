"""Reconstruct continuous portrait structures from photo masks and face landmarks."""
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from line_geometry import resample, spline_path, crossing_points
from portrait import trace_edges, connect_chains
from portrait_regions import supported_jaw


def mask_runs(mask, minimum=15):
    """Trace measured boundaries, omitting artificial segments on the image crop."""
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    runs = []
    for contour in contours:
        points = contour[:, 0, ::-1].astype(float)
        good = ((points[:, 0] > 1) & (points[:, 0] < height - 2)
                & (points[:, 1] > 1) & (points[:, 1] < width - 2))
        if good.all():
            if cv2.arcLength(contour, True) >= minimum:
                runs.append(np.vstack((points, points[0])))
            continue
        # Rotate to a crop segment so an open run never breaks at contour index zero.
        start = np.flatnonzero(~good)[0]
        points, good = np.roll(points, -start, axis=0), np.roll(good, -start)
        boundaries = np.flatnonzero(np.diff(np.r_[False, good, False]))
        for a, b in zip(boundaries[::2], boundaries[1::2]):
            chain = points[a:b]
            if len(chain) >= minimum:
                runs.append(chain)
    return runs


def largest_components(mask, minimum, count=1):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    chosen = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:count]
    return (np.isin(labels, [i for i in chosen if stats[i, cv2.CC_STAT_AREA] >= minimum])
            .astype(np.uint8) * 255)


def subject_mask(rgb, face_points):
    """Only infer a foreground mask when the photograph has a uniform border color."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    height, width = rgb.shape[:2]
    border = np.concatenate((lab[0], lab[:height // 3, 0], lab[:height // 3, -1]))
    color = np.median(border, axis=0)
    if np.mean(np.linalg.norm(border - color, axis=1) < 18) < .85:
        return None
    mask = (np.linalg.norm(lab - color, axis=2) > 25).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = largest_components(mask, height * width * .03)
    center = np.rint(face_points[1]).astype(int)
    if not mask[np.clip(center[0], 0, height - 1), np.clip(center[1], 0, width - 1)]:
        return None
    return mask


def smooth_boundary(chain, scale):
    points = resample(chain)
    closed = np.linalg.norm(points[0] - points[-1]) < 1e-5
    if closed:
        core = gaussian_filter1d(points[:-1], max(.5, 1.5 * scale), axis=0, mode='wrap')
        return np.vstack((core, core[0]))
    smoothed = gaussian_filter1d(points, max(.5, 1.5 * scale), axis=0, mode='nearest')
    smoothed[0], smoothed[-1] = points[0], points[-1]
    return smoothed


def landmark_curve(points, closed):
    """Interpolate sparse anatomical guides smoothly, rejecting newly introduced loops."""
    points = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
    points = points[keep]
    if len(points) < 4:
        return points
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    parameter = np.r_[0, np.cumsum(np.sqrt(distance))]
    spline = CubicSpline(parameter, points, axis=0, bc_type='periodic' if closed else 'natural')
    curve = spline(np.linspace(0, parameter[-1], max(40, int(distance.sum() * 2))))
    if len(crossing_points(curve)) > len(crossing_points(points)):
        return points
    return curve


def structural_curve(points, closed=False):
    """Export interpolating cubics so silhouette junctions stay exactly on their anchors."""
    points = np.asarray(points, dtype=float)
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]]
    if len(points) < 3:
        path, xy = spline_path(points[:, ::-1], closed, .4, True)
        return path, xy[:, ::-1]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    u = np.r_[0., np.cumsum(np.sqrt(lengths))]
    curve = CubicSpline(u, points, axis=0, bc_type='periodic' if closed else 'natural')
    samples = curve(np.linspace(0, u[-1], max(40, int(lengths.sum() * 3))))
    if len(crossing_points(samples)) > len(crossing_points(points)):
        # Preserve exact structural junctions if interpolation creates a new loop.
        return 'M' + 'L'.join(f'{x:.3f} {y:.3f}' for y, x in points) + ('Z' if closed else ''), points
    derivatives = curve(u, 1)
    def xy(point):
        return f'{point[1]:.3f} {point[0]:.3f}'
    path = 'M' + xy(points[0])
    for i, interval in enumerate(np.diff(u)):
        a = points[i] + derivatives[i] * interval / 3
        b = points[i + 1] - derivatives[i + 1] * interval / 3
        path += f'C{xy(a)} {xy(b)} {xy(points[i + 1])}'
    return path + ('Z' if closed else ''), samples


def split_face_silhouette(outline, points, face_width, measured_jaw=None):
    """Remove duplicate cheek edges while retaining head/ear outline and both neck sides."""
    ids = [127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356]
    face = points[ids].copy() if measured_jaw is None else np.asarray(measured_jaw)
    # Keep anatomical face geometry intact. Neck/ear fragments attach to the face,
    # never pull the jaw outward toward the body's silhouette.
    _, face_samples = structural_curve(face)
    face_tree = cKDTree(face_samples)
    pieces, attachments = [], 0
    for chain in outline:
        keep = np.ones(len(chain), dtype=bool)
        targets = {}
        for top, bottom in ((132,136), (361,365)):
            distance, index = cKDTree(chain).query(points[[top,bottom]])
            if max(distance) > face_width * .12 or index[0] == index[1]:
                continue
            a, b = sorted(index)
            # Do not discard a long arc around the other side of the head.
            if np.linalg.norm(np.diff(chain[a:b+1],axis=0),axis=1).sum() > face_width * .65:
                continue
            keep[a+1:b] = False
            for anchor in index:
                _, nearest = face_tree.query(chain[anchor])
                targets[int(anchor)] = face_samples[nearest]
            attachments += 2
        cuts = np.flatnonzero(np.diff(np.r_[False, keep, False]))
        for a,b in zip(cuts[::2],cuts[1::2]):
            if b-a >= 2:
                piece=chain[a:b].copy()
                for index, reverse in ((a,False),(b-1,True)):
                    if index not in targets:
                        continue
                    oriented=piece[::-1] if reverse else piece
                    length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(oriented,axis=0),axis=1))]
                    t=np.clip(length/max(5,face_width*.08),0,1)
                    weight=1-3*t*t+2*t*t*t
                    oriented[:]+=(targets[index]-oriented[0])[None,:]*weight[:,None]
                pieces.append(piece)
    return pieces, face, attachments


def nostril_shapes(rgb, points, face_width):
    """Trace compact dark regions in landmark-localized nostril patches of the photo."""
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    result = []
    for ids in ([59,60,75,97,98], [289,290,305,326,327]):
        guide = points[ids]
        pad = max(2, round(face_width * .012))
        lo = np.maximum(0, np.floor(guide.min(axis=0)).astype(int)-pad)
        hi = np.minimum(luminance.shape, np.ceil(guide.max(axis=0)).astype(int)+pad+1)
        crop = luminance[lo[0]:hi[0],lo[1]:hi[1]]
        if not crop.size or np.percentile(crop,80)-float(crop.min()) < 12:
            result.append(None)
            continue
        threshold = float(crop.min()) + (np.percentile(crop,80)-float(crop.min())) * .32
        dark = (crop < threshold).astype(np.uint8)*255
        contours,_ = cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        candidates = [c for c in contours if cv2.contourArea(c) >= 1]
        if not candidates:
            result.append(None)
            continue
        contour = max(candidates,key=cv2.contourArea)[:,0,::-1].astype(float)+lo
        result.append(np.vstack((contour,contour[0])))
    return result


def ear_folds(rgb, foreground, points, face_width):
    """Use visible ear width and local image gradients to guide a conservative inner fold."""
    if foreground is None:
        return []
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(0,0),.8)
    gx = cv2.Sobel(gray,cv2.CV_32F,1,0)
    gy = cv2.Sobel(gray,cv2.CV_32F,0,1)
    strength = cv2.magnitude(gx,gy)
    folds = []
    height,width = foreground.shape
    for side,ids in enumerate(([127,234,93,132],[356,454,323,361])):
        guide = points[ids]
        top = guide[0,0]-face_width*.035
        bottom = guide[-1,0]-face_width*.045
        if bottom-top < face_width*.12:
            continue
        ys = top+(bottom-top)*np.array([.15,.07,.17,.4,.65,.83,.91])
        inner = np.interp(ys,guide[:,0],guide[:,1])
        outer=[]
        for y in ys:
            xs=np.flatnonzero(foreground[int(np.clip(round(y),0,height-1))])
            outer.append((xs[0] if side==0 else xs[-1]) if len(xs) else inner[len(outer)])
        outer=np.array(outer)
        if np.median(abs(inner-outer)) < face_width*.035:
            continue
        fraction=np.array([.85,.4,.22,.32,.5,.72,.82])
        xs=outer+(inner-outer)*fraction
        initial=landmark_curve(np.column_stack((ys,xs)),False)
        tangent=np.gradient(initial,axis=0)
        normal=np.column_stack((-tangent[:,1],tangent[:,0]))
        normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-8)
        offsets=np.linspace(-face_width*.01,face_width*.01,11)
        candidates=initial[:,None,:]+normal[:,None,:]*offsets[None,:,None]
        ij=np.rint(candidates).astype(int)
        y=np.clip(ij[:,:,0],0,height-1);x=np.clip(ij[:,:,1],0,width-1)
        score=strength[y,x]-abs(offsets)[None,:]*5
        chosen=score.argmax(axis=1)
        if np.mean(strength[y[np.arange(len(y)),chosen],x[np.arange(len(x)),chosen]]>10) < .5:
            continue
        shifts=gaussian_filter1d(offsets[chosen],3)
        shifts[[0,-1]]=0
        folds.append(initial+normal*shifts[:,None])
    return folds


def reconstruct_contours(gray, rgb, regions, destination, line_width=1.1, fit_error=.9, foreground_mask=None, detail_budget=24):
    if regions is None:
        raise ValueError('Contour reconstruction requires a detected face and the original portrait.')
    height, width = gray.shape
    scale = max(height, width) / 800
    p, face_width = regions['points'], regions['face_width']
    paths = []
    kinds = []
    fitted = []

    def add(chain, kind, closed=False, tolerance=None):
        chain = np.asarray(chain, dtype=float)
        if closed and np.linalg.norm(chain[0] - chain[-1]) > 1e-6:
            chain = np.vstack((chain, chain[0]))
        if kind in ('nose_side', 'nose_tip', 'brow', 'face_contour'):
            path, samples_yx = structural_curve(chain, closed)
            samples = samples_yx[:, ::-1]
        else:
            path, samples = spline_path(chain[:, ::-1], closed, tolerance or max(.4, fit_error * scale), True)
        paths.append(path)
        fitted.append(samples[:, ::-1])
        kinds.append(kind)

    foreground = subject_mask(rgb, p) if foreground_mask is None else foreground_mask
    outline = [] if foreground is None else mask_runs(foreground, face_width * .15)
    outline = [smooth_boundary(c, scale) for c in outline]
    measured_jaw = supported_jaw(regions, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    outline,face,face_attachments = split_face_silhouette(outline,p,face_width,measured_jaw)
    for chain in outline:
        add(chain, 'subject_outline', np.linalg.norm(chain[0] - chain[-1]) < 1e-5)
    outline = fitted.copy()
    add(face, 'face_contour')
    for fold in ear_folds(rgb,foreground,p,face_width):
        add(fold,'ear_inner',tolerance=.6*scale)

    eyes = [[33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
            [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]]
    iris_elements = []
    for index, (ids, iris_ids) in enumerate(zip(eyes, ([469, 470, 471, 472], [474, 475, 476, 477]))):
        eye = p[ids]
        add(eye, 'eye', True, max(.25, .35 * scale))
        eye_path = paths[-1]
        center = p[468 if index == 0 else 473]
        radius = np.mean(np.linalg.norm(p[iris_ids] - center, axis=1))
        # Clip the measured iris to the eyelid opening; do not draw through eyelids.
        iris_elements.append((eye_path, center, radius))
    brows = [[70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
             [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]]
    for ids in brows:
        add(p[ids], 'brow', True)
    lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
            375, 321, 405, 314, 17, 84, 181, 91, 146]
    add(p[lips], 'lip_outline', True, max(.3, .45 * scale))
    upper = p[[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]]
    lower = p[[78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]]
    if np.linalg.norm(p[13] - p[14]) < face_width * .025:
        add(np.vstack((p[61], (upper + lower)[1:-1] / 2, p[291])), 'mouth_seam', tolerance=.4 * scale)
    else:
        add(np.vstack((upper, lower[-2:0:-1])), 'mouth_opening', True, .4 * scale)
    nostrils = nostril_shapes(rgb,p,face_width)
    sides = [p[[196,3,51,220,49,129,64,98]].copy(),
             p[[419,248,281,440,279,358,294,327]].copy()]
    tip = p[[60,2,290]].copy()
    for index,shape in enumerate(nostrils):
        if shape is not None:
            add(smooth_boundary(shape,scale*.35),'nostril',True,tolerance=.25*scale)
            target=fitted[-1]
            tree=cKDTree(target)
            for source,i in ((sides[index],-1),(tip,0 if index==0 else -1)):
                distance,j=tree.query(source[i])
                if distance < face_width*.055:
                    source[i]=target[j]
    for side in sides:
        add(side,'nose_side')
    add(tip,'nose_tip')

    gray_photo = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    y_grid = np.indices(gray.shape)[0]
    brow_y = min(p[70, 0], p[300, 0])
    hair_limit = int(brow_y + face_width * .13)
    hair = ((gray_photo < 80) & (y_grid < hair_limit)).astype(np.uint8) * 255
    if foreground is not None:
        hair &= foreground
    hair = largest_components(cv2.morphologyEx(hair, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), face_width**2 * .08)
    if outline and hair.any():
        boundary_tree = cKDTree(np.concatenate(outline))
        for chain in mask_runs(hair, face_width * .1):
            far = (boundary_tree.query(chain)[0] > max(2., 3 * scale)) & (chain[:, 0] < hair_limit - 2)
            boundaries = np.flatnonzero(np.diff(np.r_[False, far, False]))
            for a, b in zip(boundaries[::2], boundaries[1::2]):
                if b - a > face_width * .2:
                    fringe = smooth_boundary(chain[a:b], scale * 3)
                    for end in (0, -1):
                        face_ends=face[[0,-1]]
                        distance,index=cKDTree(face_ends).query(fringe[end])
                        if distance < face_width*.13:
                            fringe[end]=face_ends[index]
                    add(fringe, 'hairline', tolerance=max(.8, 1.5 * scale))

    # White/neutral shirt boundaries supply continuous collar and tie borders on ID photos.
    spread = np.ptp(rgb.astype(float), axis=2)
    shirt = ((rgb.min(axis=2) > 150) & (spread < 30)
             & (y_grid > p[152, 0] - face_width * .12)).astype(np.uint8) * 255
    if foreground is not None:
        shirt &= foreground
    shirt = largest_components(cv2.morphologyEx(shirt, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)),
                               face_width**2 * .03, 2)
    for chain in mask_runs(shirt, face_width * .2):
        add(smooth_boundary(chain, scale), 'collar', np.linalg.norm(chain[0] - chain[-1]) < 1e-5)
    collar_curves = [c for c, kind in zip(fitted, kinds) if kind == 'collar']
    collar_tree = cKDTree(np.concatenate(collar_curves)) if collar_curves else None

    # Keep a small number of long model strokes for hair/ear/fabric structure.
    ink = (gray < 160).astype(np.uint8) * 255
    labels = regions['labels']
    ink[np.isin(labels, [3, 4, 10, 11, 12, 13, 14, 15])] = 0
    if shirt.any():
        ink[(y_grid > p[152, 0]) & (shirt == 0)] = 0
        for chain in collar_curves:
            cv2.polylines(ink, [chain[:, ::-1].astype(np.int32)], False, 0, max(3, round(5 * scale)))
    chains = trace_edges(cv2.ximgproc.thinning(ink))
    distance = cv2.distanceTransform((gray >= 180).astype(np.uint8), cv2.DIST_L2, 5)
    chains, joins = connect_chains(chains, 10 * scale, region_labels=labels, edge_distance=distance)
    if outline:
        edge_mask = np.zeros_like(ink)
        for chain in outline + mask_runs(hair):
            cv2.polylines(edge_mask, [chain[:, ::-1].astype(np.int32)], False, 255, max(3, round(7 * scale)))
        candidates = []
        for chain in chains:
            ij = np.rint(chain).astype(int)
            if len(chain) > face_width * .12 and np.mean(edge_mask[ij[:, 0], ij[:, 1]] > 0) < .15:
                candidates.append(chain)
        hair_chains = [c for c in candidates if np.mean(np.asarray(c)[:, 0]) < p[152, 0]]
        collar_chains = [c for c in candidates if np.mean(np.asarray(c)[:, 0]) >= p[152, 0]]
        chains = hair_chains + collar_chains
    chains = sorted(chains, key=len, reverse=True)[:detail_budget]
    for chain in chains:
        if len(chain) > face_width * .12:
            curve = smooth_boundary(chain, scale)
            collar = np.mean(curve[:, 0]) > p[152, 0] and collar_tree is not None
            if collar:
                for end in (0, -1):
                    distance, index = collar_tree.query(curve[end])
                    if distance < face_width * .07:
                        curve[end] = np.concatenate(collar_curves)[index]
            add(curve, 'collar_fold' if collar else 'model_detail', tolerance=max(.8, 1.3 * scale))

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<title>Landmark and photo contour reconstruction</title>', '<defs>']
    for index, (eye_path, _, _) in enumerate(iris_elements):
        svg.append(f'<clipPath id="eye-{index}"><path d="{eye_path}"/></clipPath>')
    svg += ['</defs>', f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g fill="none" stroke="#111" stroke-width="{line_width:g}" stroke-linejoin="round" stroke-linecap="round">']
    svg.extend(f'<path data-part="{kind}" '+('fill="#111" stroke="none" ' if kind=='nostril' else '')
               +f'd="{path}"/>' for path, kind in zip(paths, kinds))
    for index, (_, (y, x), radius) in enumerate(iris_elements):
        svg.append(f'<g clip-path="url(#eye-{index})"><circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.3f}"/>'
                   f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius * .32:.3f}" fill="#111" stroke="none"/></g>')
    svg += ['</g>', '</svg>']
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(svg), encoding='utf-8')
    return {'detail': 'contour', 'detail_budget': detail_budget, 'paths': len(paths), 'iris_circles': 4,
            'structure_parts': {kind: kinds.count(kind) for kind in sorted(set(kinds))},
            'jaw_edge_supported': measured_jaw is not None, 'photo_silhouette': foreground is not None, 'collar_reconstructed': bool(shirt.any()),
            'face_outline_junctions': face_attachments,
            'closed_face_contours': sum(path.endswith('Z') for path, kind in zip(paths, kinds)
                                       if kind in ('eye', 'brow', 'lip_outline')),
            'joined_gaps': joins, 'feature_protection': True,
            'reconstruction_basis': 'photo boundaries and landmark-guided curves; not literal tracing'}
