"""Typeset the rendered cover, keeping the README asset outside doc/public/."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-dir', type=Path, default=Path('temp/banner_render'))
    parser.add_argument('--output', type=Path, default=Path('assets/readme/banner.webp'))
    args = parser.parse_args()
    regular = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    def font(size, heavy=False):
        return ImageFont.truetype(bold if heavy else regular, size)

    background, ink, muted = '#f5f5f5', '#172c43', '#526578'
    blue, teal = '#2869b3', '#18857c'
    canvas = Image.new('RGB', (3000, 1000), background)
    crop = (380, 570, 3600, 2359)
    size, origin = (1800, 1000), (1200, 0)
    render = Image.open(args.render_dir/'view_0.png').convert('RGB')
    scene = render.crop(crop).resize(size, Image.Resampling.LANCZOS)
    # Blend only the empty edge of the scene into the title's background.
    mask = Image.new('L', size, 255)
    gradient = ImageDraw.Draw(mask)
    for x in range(150):
        gradient.line([(x, 0), (x, size[1])], fill=round(255*x/149))
    canvas.paste(scene, origin, mask)
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 365), 'Piper-X Calibration', font=font(108,True), fill=ink)
    draw.text((95, 505), 'Dual-arm workcell geometry',font=font(61),fill=muted)

    points = json.loads((args.render_dir/'views.json').read_text())['0']['points_px']
    def project(name):
        x,y = points[name]
        return (origin[0]+(x-crop[0])*size[0]/(crop[2]-crop[0]),
                origin[1]+(y-crop[1])*size[1]/(crop[3]-crop[1]))
    target = project('board')
    def dashed(start, end, color):
        length = math.dist(start, end)
        for offset in range(14, int(length)-15, 22):
            a, b = offset/length, min(offset+11, length-15)/length
            draw.line([(start[0]+(end[0]-start[0])*a, start[1]+(end[1]-start[1])*a),
                       (start[0]+(end[0]-start[0])*b, start[1]+(end[1]-start[1])*b)],
                      fill=color, width=2)
    for name in ('left_camera', 'right_camera', 'head_camera'):
        dashed(project(name), target, teal if name=='head_camera' else blue)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(args.output,quality=94,method=6)
    canvas.save(args.render_dir/'banner.png')
    print(f'{args.output}: {canvas.width} x {canvas.height}, {args.output.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
