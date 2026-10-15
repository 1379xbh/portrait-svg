"""Convert a portrait photograph to a flat-color, path-only SVG."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps
from line_geometry import bridge_crosses, remove_duplicates, evidence_bridge, safe_spline_paths, SegmentIndex
from scipy.spatial import cKDTree
from portrait_regions import detect_regions, region_of, region_parameters, supported_jaw, FEATURES


def vectorize(source, destination, colors=16, size=1200, smooth=5,
              simplify=0.6, min_area=5):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must be different files")
    if destination.suffix.lower() != ".svg":
        raise ValueError("Output must have the .svg extension")
    with Image.open(source) as original:
        photo = ImageOps.exif_transpose(original).convert("RGBA")
        canvas = Image.new("RGBA", photo.size, "white")
        canvas.alpha_composite(photo)
        photo = canvas.convert("RGB")
    photo.thumbnail((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(photo)
    # Edge-preserving smoothing removes camera texture without blurring facial edges.
    if smooth:
        pixels = cv2.bilateralFilter(pixels, 9, smooth * 10, smooth * 10)
    indexed = Image.fromarray(pixels).quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    labels = np.asarray(indexed)
    palette = indexed.getpalette()
    height, width = labels.shape
    counts = np.bincount(labels.ravel(), minlength=colors)
    background = int(counts.argmax())

    def color(index):
        return "#" + "".join(f"{v:02x}" for v in palette[index * 3:index * 3 + 3])

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<title>Vector portrait</title>',
        f'<path fill="{color(background)}" d="M0 0H{width}V{height}H0Z"/>',
    ]
    path_count = 1
    for index in np.argsort(-counts):
        index = int(index)
        if index == background or not counts[index]:
            continue
        mask = (labels == index).astype(np.uint8) * 255
        # Double resolution represents even thin features as closed contours.
        mask = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
        mask = np.pad(mask, 1)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )
        if hierarchy is None:
            continue

        def ring(contour):
            polygon = cv2.approxPolyDP(contour, simplify * 2, True)
            if len(polygon) < 3:
                polygon = contour
            points = (polygon[:, 0, :].astype(float) - 0.5) / 2
            return "M" + "L".join(f"{x:g} {y:g}" for x, y in points) + "Z"

        for i, contour in enumerate(contours):
            if hierarchy[0, i, 3] != -1 or cv2.contourArea(contour) / 4 < min_area:
                continue
            rings = [ring(contour)]
            child = hierarchy[0, i, 2]
            while child != -1:
                rings.append(ring(contours[child]))
                child = hierarchy[0, child, 0]
            # Even-odd filling preserves holes such as spaces around hair strands.
            # A small matching stroke covers subpixel seams between adjacent colors.
            parts.append(
                f'<path fill="{color(index)}" stroke="{color(index)}" '
                f'stroke-width="1.2" stroke-linejoin="round" fill-rule="evenodd" '
                f'd="{"".join(rings)}"/>'
            )
            path_count += 1
    parts.append('</svg>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts), encoding="utf-8")
    return width, height, path_count


def connect_chains(chains, max_gap, region_labels=None, edge_distance=None):
    """Join nearby endpoints only when both tangents face the gap."""
    if max_gap <= 0:
        return chains, 0
    chains = [list(c) for c in chains if len(c) >= 3]
    segments = SegmentIndex(chains)
    joined = 0
    while True:
        endpoints = []
        for i, chain in enumerate(chains):
            if chain[0] == chain[-1]:
                continue
            for side in (0, 1):
                seq = chain if side == 0 else chain[::-1]
                point = np.asarray(seq[0], dtype=float)
                tangent = point - np.asarray(seq[min(6, len(seq) - 1)], dtype=float)
                norm = np.linalg.norm(tangent)
                if norm:
                    endpoints.append((i, side, point, tangent / norm))
        matches = []
        if len(endpoints) < 2:
            break
        coordinates = np.asarray([e[2] for e in endpoints])
        regions = [region_of([p], region_labels) for p in coordinates]
        pairs = cKDTree(coordinates).query_pairs(max_gap, output_type='ndarray')
        for a, b in pairs:
            i, side_i, p, tangent_i = endpoints[a]
            j, side_j, q, tangent_j = endpoints[b]
            if i == j:
                continue
            region_p, region_q = regions[a], regions[b]
            if region_labels is not None and region_p != region_q:
                continue
            delta = q - p
            distance_squared = np.sum(delta ** 2)
            if distance_squared <= 0.0001 or distance_squared > max_gap ** 2:
                continue
            distance = np.linalg.norm(delta)
            allowed_gap = max_gap * (0.55 if region_p in FEATURES else 1)
            if distance > allowed_gap:
                continue
            direction = delta / distance
            align_i = float(tangent_i @ direction)
            align_j = float(tangent_j @ -direction)
            if min(align_i, align_j) < 0.85:
                continue
            score = distance * (3 - align_i - align_j)
            matches.append((score, i, side_i, j, side_j))
        if not matches:
            break
        occupied, removed = set(), set()
        accepted = 0
        for _, i, side_i, j, side_j in sorted(matches):
            if i in occupied or j in occupied:
                continue
            first = chains[i] if side_i == 1 else chains[i][::-1]
            second = chains[j] if side_j == 0 else chains[j][::-1]
            p, q = np.asarray(first[-1], dtype=float), np.asarray(second[0], dtype=float)
            tangent_p = p - np.asarray(first[max(0, len(first) - 7)], dtype=float)
            tangent_q = q - np.asarray(second[min(6, len(second) - 1)], dtype=float)
            tangent_p /= np.linalg.norm(tangent_p)
            tangent_q /= np.linalg.norm(tangent_q)
            bridge = evidence_bridge(p, q, tangent_p, tangent_q, edge_distance)
            if bridge is None:
                continue
            if region_labels is not None:
                label = region_of([p], region_labels)
                if sum(region_of([point], region_labels) == label for point in bridge) < len(bridge) * 0.85:
                    continue
            if bridge_crosses(bridge, segments):
                continue
            chains[i] = first + [tuple(point) for point in bridge[1:-1]] + second
            # Existing segments remain present after merging. Include each new
            # bridge immediately so later joins cannot cross it in this round.
            segments.add(np.vstack((p, bridge[1:-1], q)))
            occupied.update((i, j))
            removed.add(j)
            joined += 1
            accepted += 1
        if not accepted:
            break
        chains = [c for i, c in enumerate(chains) if i not in removed]
    return chains, joined

def curve_path(points, closed):
    """Quadratic midpoint curves suppress pixel stair steps without overshoot."""
    def xy(p):
        return f'{p[0]:.2f} {p[1]:.2f}'
    if len(points) == 2:
        return f'M{xy(points[0])}L{xy(points[1])}' + ('Z' if closed else '')
    if closed:
        start = (points[-1] + points[0]) / 2
        path = 'M' + xy(start)
        for i, point in enumerate(points):
            end = (point + points[(i + 1) % len(points)]) / 2
            path += f'Q{xy(point)} {xy(end)}'
        return path + 'Z'
    path = 'M' + xy(points[0])
    for i in range(1, len(points) - 1):
        end = (points[i] + points[i + 1]) / 2
        path += f'Q{xy(points[i])} {xy(end)}'
    return path + 'L' + xy(points[-1])


def trace_edges(edges):
    # Trace the one-pixel edge graph, instead of outlining both sides of each line.
    pixels = set(zip(*np.nonzero(edges)))
    graph = {}
    for y, x in sorted(pixels):
        neighbors = []
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            other = (y + dy, x + dx)
            if other not in pixels:
                continue
            if dy and dx and ((y + dy, x) in pixels or (y, x + dx) in pixels):
                continue
            neighbors.append(other)
        graph[y, x] = neighbors
    used = set()
    chains = []

    def trace(start, neighbor):
        chain = [start, neighbor]
        used.add(frozenset((start, neighbor)))
        previous, current = start, neighbor
        while len(graph[current]) == 2:
            following = next(p for p in graph[current] if p != previous)
            edge = frozenset((current, following))
            if edge in used:
                break
            used.add(edge)
            chain.append(following)
            previous, current = current, following
        chains.append(chain)

    for start, neighbors in graph.items():
        if len(neighbors) != 2:
            for neighbor in neighbors:
                if frozenset((start, neighbor)) not in used:
                    trace(start, neighbor)
    for start, neighbors in graph.items():
        for neighbor in neighbors:
            if frozenset((start, neighbor)) not in used:
                trace(start, neighbor)
    return chains


def lineart(source, destination, size=1200, smooth=5, simplify=0.6,
            edge_low=35, edge_high=85, min_length=6, line_width=0.8,
            gap=12, blur=1.2, fit_error=1.0, dedup=1.25, regions=True):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must be different files")
    if destination.suffix.lower() != ".svg":
        raise ValueError("Output must have the .svg extension")
    with Image.open(source) as original:
        photo = ImageOps.exif_transpose(original).convert("RGBA")
        canvas = Image.new("RGBA", photo.size, "white")
        canvas.alpha_composite(photo)
        photo = canvas.convert("RGB")
    photo.thumbnail((size, size), Image.Resampling.LANCZOS)
    rgb = np.asarray(photo)
    original_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    regions = detect_regions(rgb) if regions else None
    labels = regions['labels'] if regions is not None else None
    gray = original_gray
    if smooth:
        gray = cv2.bilateralFilter(gray, 9, smooth * 10, smooth * 10)
    smoothed = cv2.GaussianBlur(gray, (0, 0), blur)
    edges = cv2.Canny(smoothed, edge_low, edge_high, L2gradient=True)
    if labels is not None:
        # Keep small eye/lip contours without retaining clothing microtexture.
        fine = cv2.GaussianBlur(gray, (0, 0), max(0.7, blur * 0.62))
        fine_edges = cv2.Canny(fine, edge_low, edge_high, L2gradient=True)
        feature_mask = np.isin(labels, list(FEATURES))
        edges[feature_mask] = fine_edges[feature_mask]
        # Face-interior shadows are not a useful anatomical outline.
        edges[labels == 4] = 0
    weak = cv2.Canny(cv2.GaussianBlur(original_gray, (0, 0), 0.9),
                     max(1, edge_low * 0.5), max(2, edge_high * 0.5), L2gradient=True)
    edge_distance = cv2.distanceTransform((weak == 0).astype(np.uint8), cv2.DIST_L2, 5)
    jaw = supported_jaw(regions, original_gray) if regions is not None else None
    if jaw is not None:
        # Replace only the narrow band around the supported jaw, not ears or hair.
        band = np.zeros_like(edges)
        cv2.polylines(band, [np.asarray(jaw)[:, ::-1].astype(np.int32)], False, 1,
                      max(5, round(regions['face_width'] * 0.065)))
        edges[band > 0] = 0
    chains = trace_edges(edges)
    # Clean duplicates within each anatomical region to avoid deleting nearby features.
    grouped = {}
    for chain in chains:
        grouped.setdefault(region_of(chain, labels), []).append(chain)
    chains, duplicates = [], 0
    for label, group in grouped.items():
        cleaned, removed = remove_duplicates(group, dedup * (0.45 if label in FEATURES else 1))
        chains.extend(cleaned)
        duplicates += removed
    chains, joined = connect_chains(chains, gap, labels, edge_distance)
    if jaw is not None:
        chains.insert(0, jaw)
    width, height = photo.size
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<title>Vector portrait line art</title>',
             f'<path fill="white" d="M0 0H{width}V{height}H0Z"/>',
             f'<g fill="none" stroke="#111111" stroke-width="{line_width:g}" stroke-linecap="round" stroke-linejoin="round">']
    selected, tolerances, region_counts = [], [], {}
    for chain in chains:
        points = np.array([(x + 0.5, y + 0.5) for y, x in chain], dtype=np.float32)
        label = region_of(chain, labels)
        length_factor, tolerance_factor = region_parameters(label) if labels is not None else (1, 1)
        if cv2.arcLength(points, False) < min_length * length_factor:
            continue
        if len(points) < 2:
            continue
        selected.append(points)
        tolerances.append(fit_error * tolerance_factor)
        region_counts[label] = region_counts.get(label, 0) + 1
    paths, backed_off = safe_spline_paths(selected, tolerances)
    for path in paths:
        parts.append(f'<path d="{path}"/>')
    count = len(paths)
    parts.extend(['</g>', '</svg>'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(parts), encoding='utf-8')
    print(f'Removed {duplicates} duplicates; connected {joined} supported gaps; retained {count} paths; landmarks={regions is not None}; jaw={jaw is not None}; fit backoffs={backed_off}')
    print(f'Region path counts: {region_counts}')
    return width, height, count


def main():
    parser = argparse.ArgumentParser(description="人像照片 → 纯路径 SVG 矢量肖像画（本地处理）")
    parser.add_argument("input", type=Path, help="输入照片，例如 photo.jpg")
    parser.add_argument("-o", "--output", type=Path, help="输出 SVG，默认与照片同目录")
    parser.add_argument("--mode", choices=("color", "line"), default="color", help="color 色块；line 黑白矢量线稿")
    parser.add_argument("--regions", choices=("auto", "off"), default="auto", help="auto 本地五官关键点分区；off 通用边缘模式")
    parser.add_argument("--edge-low", type=int, default=35, help="线稿弱边缘阈值 0–255")
    parser.add_argument("--edge-high", type=int, default=85, help="线稿强边缘阈值 0–255，降低可保留更多线条")
    parser.add_argument("--min-length", type=float, default=6, help="线稿最短线段长度，像素")
    parser.add_argument("--line-width", type=float, default=0.8, help="线稿线宽，像素")
    parser.add_argument("--gap", type=float, default=12, help="自动连接断线的最大距离，0 关闭，像素")
    parser.add_argument("--blur", type=float, default=1.2, help="线稿预平滑，越大细碎线越少")
    parser.add_argument("--fit-error", type=float, default=1.0, help="整条样条拟合容差，像素；五官区域自动减小")
    parser.add_argument("--dedup", type=float, default=1.25, help="重复路径判定距离，0 关闭；只删除至少95%%重合的路径")
    parser.add_argument("--colors", type=int, default=16, help="色数 2–64，默认 16")
    parser.add_argument("--size", type=int, default=1200, help="处理最长边 64–4096，默认 1200")
    parser.add_argument("--smooth", type=int, default=5, help="平滑强度 0–10，默认 5")
    parser.add_argument("--simplify", type=float, default=0.6, help="色块轮廓简化误差 0–5 像素；线稿用 --fit-error")
    parser.add_argument("--min-area", type=float, default=5, help="忽略小色块的面积 0–1000 像素")
    args = parser.parse_args()
    for name, low, high in [("colors", 2, 64), ("size", 64, 4096),
                            ("smooth", 0, 10), ("simplify", 0, 5), ("min_area", 0, 1000),
                            ("edge_low", 0, 255), ("edge_high", 0, 255),
                            ("min_length", 0, 10000), ("line_width", 0.1, 20),
                            ("gap", 0, 100), ("blur", 0.1, 10),
                            ("fit_error", 0.1, 5), ("dedup", 0, 5)]:
        if not low <= getattr(args, name) <= high:
            parser.error(f"--{name.replace('_', '-')} must be between {low} and {high}")
    if args.edge_low >= args.edge_high:
        parser.error("--edge-low must be less than --edge-high")
    suffix = "_line.svg" if args.mode == "line" else "_portrait.svg"
    output = args.output or args.input.with_name(args.input.stem + suffix)
    try:
        if args.mode == "line":
            width, height, paths = lineart(args.input, output, args.size, args.smooth,
                                           args.simplify, args.edge_low, args.edge_high,
                                           args.min_length, args.line_width, args.gap, args.blur,
                                           args.fit_error, args.dedup, args.regions == 'auto')
        else:
            width, height, paths = vectorize(args.input, output, args.colors, args.size,
                                         args.smooth, args.simplify, args.min_area)
    except (OSError, ValueError, Image.DecompressionBombError, cv2.error) as exc:
        print(f"转换失败: {exc}", file=sys.stderr)
        return 1
    print(f"已保存: {output.resolve()} ({width}×{height}, {paths} 个矢量路径)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
