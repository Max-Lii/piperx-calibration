"""Rebuild the corner-ID illustration from an unchanged simulation RGB sample.

Run from the repository root after installing the base calibration dependencies and Pillow:
    python doc/scripts/prepare_figures.py
This script does not load RoboTwin or modify calibration parameters.
"""
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "doc/source-images"


def font(size: int):
    # Pillow's bundled scalable default also works on machines without DejaVu.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def main():
    manifest = json.loads((ROOT / "assets/targets/BOARD_002_A4_Charuco_5x7_35mm_metric.json").read_text())
    source = Image.open(IMAGES / "left_wrist_sample.png").convert("RGB")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, manifest["dictionary"]))
    board = cv2.aruco.CharucoBoard(
        (manifest["squares_x"], manifest["squares_y"]),
        manifest["square_length_mm"] / 1000,
        manifest["marker_length_mm"] / 1000, dictionary,
    )
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(np.asarray(source))
    if ids is None or len(ids) < 4:
        raise RuntimeError("The source image has insufficient ChArUco corners")
    points = corners.reshape(-1, 2)
    extent = points.max(axis=0) - points.min(axis=0)
    margin = np.maximum(extent * 0.45, 60)
    low = np.maximum(points.min(axis=0) - margin, 0).astype(int)
    high = np.minimum(points.max(axis=0) + margin, source.size).astype(int)
    crop = source.crop((*low, *high))
    canvas = Image.new("RGB", (1800, 1050), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    ink, muted, accent = "#17253c", "#52637c", "#1669ba"
    draw.text((52, 32), "ChArUco: image points matched by ID", fill=ink, font=font(34))
    draw.text((52, 84), f"Left wrist RGB  |  {len(ids)} of 24 internal corners detected  |  BOARD_002", fill=muted, font=font(22))
    panels = [(52, 175, 710, 925), (758, 175, 1748, 925)]
    transforms = []
    for image, box in zip((source, crop), panels):
        x0, y0, x1, y1 = box
        scale = min((x1 - x0) / image.width, (y1 - y0) / image.height)
        size = (round(image.width * scale), round(image.height * scale))
        offset = (x0 + ((x1 - x0) - size[0]) // 2, y0 + ((y1 - y0) - size[1]) // 2)
        draw.rounded_rectangle(box, 12, fill="white", outline="#d7dfe9", width=2)
        canvas.paste(image.resize(size, Image.Resampling.LANCZOS), offset)
        transforms.append((scale, np.array(offset)))
    scale, offset = transforms[0]
    draw.rectangle((*tuple(low * scale + offset), *tuple(high * scale + offset)), outline=accent, width=3)
    scale, offset = transforms[1]
    for point, index in zip(points, ids.ravel()):
        x, y = (point - low) * scale + offset
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=accent, outline="white", width=2)
        label = str(int(index))
        label_font = font(21)
        bounds = draw.textbbox((0, 0), label, font=label_font)
        width = bounds[2] - bounds[0]
        draw.rounded_rectangle((x + 8, y - 30, x + width + 18, y - 2), 5, fill="white", outline="#c3d5e9")
        draw.text((x + 13, y - 30), label, fill=accent, font=label_font)
    draw.text((52, 955), "Original view / crop location", fill=ink, font=font(22))
    draw.text((758, 955), "Detected internal corners / actual detector IDs", fill=ink, font=font(22))
    draw.text((52, 1000), "Object points: board.getChessboardCorners()[ids]  |  Units: metres  |  Outer page IDs are not PnP corners", fill=muted, font=font(20))
    output = IMAGES / "charuco_detection.png"
    canvas.save(output)
    print(json.dumps({"output": str(output), "source_resolution": source.size, "detected_ids": ids.ravel().tolist()}))


if __name__ == "__main__":
    main()
