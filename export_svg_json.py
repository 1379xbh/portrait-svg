"""Offline conversion; no server or model key required."""
import argparse
import json
from pathlib import Path
from svg_io import svg_and_json

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('svg',type=Path)
    parser.add_argument('-o','--output',type=Path,required=True)
    args=parser.parse_args()
    if args.svg.resolve()==args.output.resolve(): parser.error('Do not overwrite the source SVG')
    _,document=svg_and_json(args.svg.read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.output.resolve())
