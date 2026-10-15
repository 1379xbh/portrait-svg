"""Validate a deliberately small SVG subset and export its lossless element tree."""
import math
import re
import xml.etree.ElementTree as ET
from defusedxml.ElementTree import fromstring
from svgpathtools import parse_path

TAGS = {'svg','g','path','rect','circle','ellipse','line','polyline','polygon','title','desc','defs','clipPath'}
ATTRS = set('id viewBox width height d fill fill-rule stroke stroke-width stroke-linecap stroke-linejoin opacity fill-opacity stroke-opacity cx cy r rx ry x y x1 x2 y1 y2 points'.split())
ATTRS.update(('clip-path','data-part'))
NUMERIC = set('width height stroke-width opacity fill-opacity stroke-opacity cx cy r rx ry x y x1 x2 y1 y2'.split())


def svg_and_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:svg|xml)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    if len(text) > 1_000_000:
        raise ValueError('SVG exceeds 1 MB')
    try:
        root = fromstring(text, forbid_dtd=True, forbid_entities=True)
    except Exception as exc:
        raise ValueError('Invalid SVG XML') from exc
    nodes = list(root.iter())
    clip_ids={el.attrib.get('id') for el in nodes if el.tag.rsplit('}',1)[-1]=='clipPath'}
    ids=[el.attrib['id'] for el in nodes if 'id' in el.attrib]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate SVG IDs')
    if len(nodes) > 1000:
        raise ValueError('Too many SVG elements')
    for el in nodes:
        if el.tag.startswith('{') and not el.tag.startswith('{http://www.w3.org/2000/svg}'):
            raise ValueError('Unsupported namespace')
        el.tag = el.tag.rsplit('}',1)[-1]
        if el.tag not in TAGS or set(el.attrib)-ATTRS:
            raise ValueError('Unsupported SVG element or attribute')
        for key, value in el.attrib.items():
            if key == 'clip-path':
                match=re.fullmatch(r'url\(#([A-Za-z_][A-Za-z0-9_.-]*)\)',value)
                if not match or match[1] not in clip_ids:
                    raise ValueError('clip-path must reference an existing local clipPath')
            if key in NUMERIC and (not math.isfinite(float(value)) or abs(float(value)) > 100000):
                raise ValueError('Invalid numeric attribute')
            if key in ('fill','stroke') and not re.fullmatch(r'none|black|white|#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?', value):
                raise ValueError('Only plain colors allowed')
        if el.tag == 'path':
            try:
                path = parse_path(el.attrib.get('d',''))
                if not path or not all(math.isfinite(v) for v in path.bbox()):
                    raise ValueError()
            except Exception as exc:
                raise ValueError('Invalid path geometry') from exc
    if root.tag != 'svg' or not any(el.tag=='path' for el in nodes):
        raise ValueError('Expected SVG with vector paths')
    try:
        box = [float(v) for v in root.attrib['viewBox'].replace(',',' ').split()]
        if len(box)!=4 or not all(math.isfinite(v) for v in box) or min(box[2:])<=0:
            raise ValueError()
    except (KeyError, ValueError) as exc:
        raise ValueError('Valid viewBox required') from exc
    def encode(el):
        item = {'tag':el.tag, 'attributes':dict(el.attrib)}
        if el.text and el.text.strip(): item['text']=el.text.strip()
        if len(el): item['children']=[encode(c) for c in el]
        return item
    payload = {'schemaVersion':1, 'format':'svg-element-tree', 'svg':encode(root)}
    root.set('xmlns','http://www.w3.org/2000/svg')
    return ET.tostring(root,encoding='unicode'), payload


def svg_metrics(text):
    """Count visible geometry, excluding definitions, white background and hidden groups."""
    root = fromstring(text, forbid_dtd=True, forbid_entities=True)
    result = {'visible_paths': 0, 'visible_strokes': 0, 'curve_segments': 0,
              'segments': 0, 'path_length': 0.0, 'svg_bytes': len(text.encode('utf-8'))}
    def visit(el, inherited):
        tag = el.tag.rsplit('}', 1)[-1]
        if tag in ('defs', 'clipPath', 'title', 'desc'):
            return
        style = dict(inherited)
        style.update(el.attrib)
        if float(style.get('opacity', 1)) == 0:
            return
        stroke = style.get('stroke', 'none')
        fill = style.get('fill', 'black')
        visible = (stroke not in ('none', 'white', '#fff', '#ffffff') and float(style.get('stroke-opacity', 1)) > 0) or (fill not in ('none', 'white', '#fff', '#ffffff') and float(style.get('fill-opacity', 1)) > 0)
        if visible and tag == 'path':
            path = parse_path(el.attrib['d'])
            result['visible_paths'] += 1
            result['visible_strokes'] += len(re.findall(r'[Mm]', el.attrib['d']))
            result['segments'] += len(path)
            result['curve_segments'] += sum(type(s).__name__ != 'Line' for s in path)
            result['path_length'] += path.length(error=1e-4)
        elif visible and tag in ('circle', 'ellipse', 'line', 'polyline', 'polygon'):
            result['visible_strokes'] += 1
        for child in el:
            visit(child, style)
    visit(root, {})
    result['path_length'] = round(result['path_length'], 2)
    return result
