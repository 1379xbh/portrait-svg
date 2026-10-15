"""Trace monochrome artwork into filled editable SVG curves, retaining holes."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from line_geometry import spline_path


def convert(source, output, simplify_details=False):
    gray = np.asarray(Image.open(source).convert('L'))
    if simplify_details:
        from simplify_portrait import simplify
        gray,report=simplify(gray)
        Image.fromarray(gray).save(Path(output).with_suffix('.simplified.png'))
        print(report)
    enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    enlarged = cv2.GaussianBlur(enlarged, (0, 0), .8)
    ink = (enlarged < 185).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    paths = []
    def ring(contour):
        points = (contour[:, 0, :].astype(float) + .5) / 2
        if len(points) < 3:
            return ''
        try:
            return spline_path(np.vstack([points, points[0]]), True, .7)
        except ValueError:
            return 'M' + 'L'.join(f'{x:g} {y:g}' for x, y in points) + 'Z'
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0, i, 3] != -1 or cv2.contourArea(contour) < 8:
                continue
            rings = [ring(contour)]
            child = hierarchy[0, i, 2]
            while child != -1:
                rings.append(ring(contours[child]))
                child = hierarchy[0, child, 0]
            paths.append(''.join(rings))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<title>Vector portrait traced from generated line artwork</title>',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           '<g fill="#111" fill-rule="evenodd">']
    svg += [f'<path id="ink-{i+1}" d="{p}"/>' for i, p in enumerate(paths)]
    svg += ['</g>', '</svg>']
    Path(output).write_text('\n'.join(svg), encoding='utf-8')
    print(f'{width}x{height}, {len(paths)} filled paths, no embedded raster')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--simplify',action='store_true')
    args = parser.parse_args()
    convert(args.input, args.output,args.simplify)
