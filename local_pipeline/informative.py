"""Local Informative Drawings inference, followed by centerline SVG tracing."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from line_geometry import remove_duplicates, safe_spline_paths, spline_path
from portrait import trace_edges, connect_chains
from portrait_regions import detect_regions, region_of, FEATURES, glasses_evidence, protect_glasses
from line_cleanup import DETAIL_LEVELS, clean_feature_ink, select_clean_chains, budget_chains


def join_junctions(chains):
    """Follow straight continuations through existing skeleton junctions."""
    chains = [list(c) for c in chains]
    while True:
        at = {}
        for i, chain in enumerate(chains):
            if len(chain) < 2 or chain[0] == chain[-1]:
                continue
            for side in (0, 1):
                seq = chain if side == 0 else chain[::-1]
                direction = np.array(seq[min(5,len(seq)-1)],dtype=float) - seq[0]
                norm = np.linalg.norm(direction)
                if norm:
                    at.setdefault(tuple(seq[0]),[]).append((i,side,direction/norm))
        matches = []
        for endpoints in at.values():
            for index, (i,si,di) in enumerate(endpoints):
                for j,sj,dj in endpoints[index+1:]:
                    alignment = float(di @ dj)
                    if i != j and alignment < -.75:
                        matches.append((alignment,i,si,j,sj))
        if not matches:
            return chains
        used,removed = set(),set()
        for _,i,si,j,sj in sorted(matches):
            if i in used or j in used:
                continue
            first=chains[i] if si else chains[i][::-1]
            second=chains[j] if not sj else chains[j][::-1]
            chains[i]=first+second[1:]
            used.update((i,j))
            removed.add(j)
        chains=[c for i,c in enumerate(chains) if i not in removed]


def generate_drawing(source, destination, style='style1', size=800, device='auto'):
    import torch
    import torch.nn.functional as functional
    from third_party.informative_drawings.generator import Generator
    weights = Path(__file__).parent / 'models' / 'informative' / f'{style}.pth'
    if not weights.is_file():
        raise ValueError('Weights missing: run python setup_informative.py')
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im).convert('RGBA')
        background = Image.new('RGBA', im.size, 'white')
        background.alpha_composite(im)
        photo = background.convert('RGB')
    photo.thumbnail((size, size), Image.Resampling.LANCZOS)
    width, height = photo.size
    if min(width, height) < 16:
        raise ValueError('Input must be at least 16 pixels on both sides')
    selected_device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if selected_device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if selected_device == 'cpu':
        torch.set_num_threads(min(8, torch.get_num_threads()))
    model = Generator(3, 1, 3)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.to(selected_device).eval()
    tensor = torch.from_numpy(np.asarray(photo).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    # Network downsamples twice. Pad to multiples of four, then crop back exactly.
    tensor = functional.pad(tensor, (0, (-width) % 4, 0, (-height) % 4), mode='reflect')
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model(tensor.to(selected_device))[0, 0, :height, :width].clamp(0, 1).cpu().numpy()
    seconds = time.perf_counter() - started
    drawing = Image.fromarray(np.rint(prediction * 255).astype(np.uint8))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    drawing.save(destination)
    return {'device': selected_device, 'inference_seconds': round(seconds, 3),
            'width': width, 'height': height, 'style': style, 'weights': weights.name}


def vectorize_drawing(source, destination, threshold=180, min_length=12,
                      fit_error=0.9, gap=5, line_width=1.25, portrait_source=None,
                      detail='faithful', foreground_mask=None, detail_budget=24):
    if detail not in DETAIL_LEVELS:
        raise ValueError(f'Unknown detail level: {detail}')
    with Image.open(source) as image:
        gray = np.asarray(image.convert('L'))
    height, width = gray.shape
    if foreground_mask is not None:
        if foreground_mask.shape != gray.shape:
            raise ValueError("Person mask and drawing dimensions must match")
        gray = gray.copy()
        allowed = cv2.dilate((foreground_mask > 0).astype(np.uint8), np.ones((5,5), np.uint8))
        gray[allowed == 0] = 255
    labels = None
    regions = None
    if portrait_source is not None:
        with Image.open(portrait_source) as photo:
            photo = ImageOps.exif_transpose(photo).convert('RGB')
            photo = photo.resize((width,height),Image.Resampling.LANCZOS)
        regions = detect_regions(np.asarray(photo))
        if regions is not None:
            labels = regions['labels']
    possible_glasses = glasses_evidence(gray, regions)
    if possible_glasses:
        labels = protect_glasses(labels, regions)
    if detail == 'contour' and (regions is None or possible_glasses):
        report = vectorize_drawing(source, destination, threshold, min_length, fit_error,
            gap, line_width, portrait_source, 'clean', foreground_mask, detail_budget)
        report.update({'requested_detail': 'contour', 'fallback_reason': 'face_not_detected' if regions is None else 'possible_glasses'})
        return report
    if detail == 'contour':
        from contour_portrait import reconstruct_contours
        if portrait_source is None:
            raise ValueError('Contour mode requires the original portrait photo.')
        return reconstruct_contours(gray, np.asarray(photo), regions, destination, line_width, fit_error, foreground_mask=foreground_mask, detail_budget=detail_budget)
    ink = (gray < threshold).astype(np.uint8) * 255
    # Filled pupils / thick eyebrow marks make artificial branches when thinned.
    # Keep their original ink silhouettes as small editable vector shapes.
    feature_paths = []
    centerline_ink = ink.copy()
    if labels is not None:
        feature_ink = clean_feature_ink(ink, labels, detail)
        centerline_ink[np.isin(labels,[10,11,12,13])] = 0
        contours,hierarchy=cv2.findContours(feature_ink,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i,contour in enumerate(contours):
                if hierarchy[0,i,3] != -1 or cv2.contourArea(contour) < 1:
                    continue
                x, y = contour[0, 0]
                brow_outline = detail != 'faithful' and labels[y, x] in (12, 13)
                def outline(c):
                    points=c[:,0,:].astype(float)+.5
                    if len(points) < 3:
                        return ''
                    polygon='M'+'L'.join(f'{x:g} {y:g}' for x,y in points)+'Z'
                    if cv2.contourArea(c) < 4:
                        return polygon
                    try:
                        return spline_path(np.vstack([points,points[0]]),True,
                                           .22 if detail == 'faithful' else (.9 if brow_outline else .45))
                    except ValueError:
                        return polygon
                rings=[outline(contour)]
                child=hierarchy[0,i,2]
                while child != -1:
                    rings.append(outline(contours[child]))
                    child=hierarchy[0,child,0]
                feature_paths.append((''.join(rings), brow_outline))
    skeleton = cv2.ximgproc.thinning(centerline_ink, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    chains = join_junctions(trace_edges(skeleton))
    chains, duplicates = remove_duplicates(chains, 0.65)
    distance = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 5)
    chains, joins = connect_chains(chains, gap, region_labels=labels, edge_distance=distance)
    chains, cleanup = select_clean_chains(chains, labels, gray, min_length, detail, threshold)
    chains, budget_report = budget_chains(chains, labels, gray,
        None if detail == 'faithful' else detail_budget, threshold, foreground_mask)
    selected, tolerances = [], []
    for chain in chains:
        points = np.array([(x + .5, y + .5) for y, x in chain], dtype=np.float32)
        feature = region_of(chain,labels) in FEATURES
        cutoff = max(2,min_length*.20) if feature else min_length
        if len(points) >= 2 and cv2.arcLength(points, False) >= cutoff:
            selected.append(points)
            tolerances.append(fit_error*(.4 if feature else 1))
    paths, backoffs = safe_spline_paths(selected, tolerances)
    svg = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<title>Informative Drawings centerline portrait</title>',
           f'<path fill="white" d="M0 0H{width}V{height}H0Z"/>',
           f'<g fill="none" stroke="#111" stroke-width="{line_width:g}" stroke-linecap="round" stroke-linejoin="round">']
    svg += [f'<path d="{path}"/>' for path in paths]
    svg += ['</g>']
    for path, outline in feature_paths:
        if outline:
            svg.append(f'<path fill="none" stroke="#111" stroke-width="{line_width:g}" '
                       f'stroke-linecap="round" stroke-linejoin="round" d="{path}"/>')
        else:
            svg.append(f'<path fill="#111" fill-rule="evenodd" d="{path}"/>')
    svg += ['</svg>']
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(svg), encoding='utf-8')
    return {'paths': len(paths)+len(feature_paths), 'centerline_paths':len(paths),
            'filled_feature_paths':sum(not outline for _, outline in feature_paths),
            'outlined_feature_paths':sum(outline for _, outline in feature_paths),
            'joined_gaps': joins, 'removed_duplicates': duplicates,
            'fit_backoffs': backoffs, 'threshold': threshold, 'min_length': min_length,
            'fit_error': fit_error, 'gap': gap, 'line_width': line_width,
            'feature_protection': labels is not None, 'glasses_protection': possible_glasses, 'detail': detail, **cleanup, **budget_report}


def main():
    parser = argparse.ArgumentParser(description='照片 → Informative Drawings 线稿 → SVG')
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--remove-background', action='store_true', help='CPU person segmentation before drawing')
    parser.add_argument('--style', choices=('style1', 'style2'), default='style1')
    parser.add_argument('--size', type=int, default=800)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threshold', type=int, default=180)
    parser.add_argument('--min-length', type=float, default=12)
    parser.add_argument('--fit-error', type=float, default=.9)
    parser.add_argument('--detail-budget', type=int, default=24, help='Maximum optional strokes; protected structures are excluded')
    parser.add_argument('--gap', type=float, default=5)
    parser.add_argument('--line-width', type=float, default=1.25)
    parser.add_argument('--detail', choices=DETAIL_LEVELS, default='clean',
                        help='faithful: detail; clean: fewer strays; minimal: fewer lines; contour: reconstruct continuous structures')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.svg':
        parser.error('Output must end in .svg')
    for name, lower, upper in [('size',64,2048),('threshold',1,254),('min_length',0,1000),
                                ('detail_budget',0,200),('fit_error',.1,5),('gap',0,30),('line_width',.1,10)]:
        if not lower <= getattr(args,name) <= upper:
            parser.error(f'{name} must be between {lower} and {upper}')
    drawing = args.output.with_name(args.output.stem + '_model.png')
    report_path = args.output.with_suffix('.json')
    if args.input.resolve() in [p.resolve() for p in (args.output,drawing,report_path)]:
        parser.error('Output files must not overwrite the input')
    started = time.perf_counter()
    foreground_mask, background_report = None, {}
    model_source = args.input
    if args.remove_background:
        from background_removal import remove_background
        model_source = args.output.with_name(args.output.stem + '_white.png')
        mask_output = model_source.with_name(model_source.stem + '_mask.png')
        if args.input.resolve() in (model_source.resolve(), mask_output.resolve()):
            parser.error('Background outputs must not overwrite the input')
        foreground_mask, background_report = remove_background(args.input, model_source, args.size)
    report = generate_drawing(model_source, drawing, args.style, args.size, args.device)
    report.update(background_report)
    vector_started = time.perf_counter()
    report.update(vectorize_drawing(drawing,args.output,args.threshold,args.min_length,
                                    args.fit_error,args.gap,args.line_width,args.input,args.detail,foreground_mask,args.detail_budget))
    finished = time.perf_counter()
    report.update({'generation_seconds': round(vector_started - started, 3),
                   'vectorization_seconds': round(finished - vector_started, 3),
                   'pipeline_seconds': round(finished - started, 3)})
    report.update({'input': str(args.input.resolve()), 'drawing': str(drawing.resolve()),
                   'svg': str(args.output.resolve())})
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
