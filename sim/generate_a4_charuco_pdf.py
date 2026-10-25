"""Generate the canonical black-and-white A4 ChArUco calibration target."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


MM_PER_INCH = 25.4
A4_MM = (210.0, 297.0)
DPI = 600
SQUARE_MM = 35.0
MARKER_MM = 26.25
SQUARES = (5, 7)
BOARD_TOP_LEFT_MM = (8.0, 26.0)
X_CENTER_MM = (196.5, 148.5)
X_SIZE_MM = 14.0
X_STROKE_MM = 2.5
METRIC_MARKER_DICTIONARY = "DICT_4X4_50"
METRIC_MARKER_SIZE_MM = 12.0
METRIC_MARKER_BORDER_MM = 2.0
METRIC_MARKERS = {40: (2.0, 3.0), 41: (192.0, 3.0), 42: (2.0, 279.0), 43: (192.0, 279.0)}


def _px(mm: float) -> int:
    return round(mm / MM_PER_INCH * DPI)


def _font(size_px: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size_px)
    return ImageFont.load_default()


def render_page(
    *, metric_fiducials: bool = False, target_id: str = "BOARD_001",
    content_scale_x: float = 1.0, content_scale_y: float = 1.0,
) -> Image.Image:
    """Render the canonical page once for both PDF and RoboTwin texture use."""
    page_px = (_px(A4_MM[0]), _px(A4_MM[1]))
    board_mm = (SQUARES[0] * SQUARE_MM, SQUARES[1] * SQUARE_MM)
    board_px = (_px(board_mm[0]), _px(board_mm[1]))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(SQUARES, SQUARE_MM / 1000, MARKER_MM / 1000, dictionary)
    raster = board.generateImage(board_px, marginSize=0, borderBits=1)
    page = Image.new("L", page_px, 255)
    page.paste(Image.fromarray(raster), (_px(BOARD_TOP_LEFT_MM[0]), _px(BOARD_TOP_LEFT_MM[1])))

    if metric_fiducials:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, METRIC_MARKER_DICTIONARY))
        marker_size = _px(METRIC_MARKER_SIZE_MM)
        border = _px(METRIC_MARKER_BORDER_MM)
        canvas_size = marker_size + 2 * border
        for marker_id, (x_mm, y_mm) in METRIC_MARKERS.items():
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
            marker_canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
            marker_canvas[border:-border, border:-border] = marker
            page.paste(Image.fromarray(marker_canvas), (_px(x_mm), _px(y_mm)))

    draw = ImageDraw.Draw(page)
    cx, cy, half = _px(X_CENTER_MM[0]), _px(X_CENTER_MM[1]), _px(X_SIZE_MM / 2)
    thickness = _px(X_STROKE_MM)
    draw.line((cx - half, cy - half, cx + half, cy + half), fill=0, width=thickness)
    draw.line((cx - half, cy + half, cx + half, cy - half), fill=0, width=thickness)
    label_font = _font(_px(2.5))
    footer = f"{target_id}  5x7 Charuco  square=35.000mm  marker=26.250mm  PRINT ACTUAL SIZE / 100%"
    draw.text((_px(24), _px(288)), footer, fill=0, font=label_font)
    draw.text((_px(187), _px(158)), "ALIGN X", fill=0, font=label_font)
    if content_scale_x <= 0.0 or content_scale_y <= 0.0:
        raise ValueError("content scales must be positive")
    if content_scale_x != 1.0 or content_scale_y != 1.0:
        source = np.asarray(page)
        centre_x, centre_y = (source.shape[1] - 1) / 2.0, (source.shape[0] - 1) / 2.0
        matrix = np.asarray([
            [content_scale_x, 0.0, (1.0 - content_scale_x) * centre_x],
            [0.0, content_scale_y, (1.0 - content_scale_y) * centre_y],
        ], dtype=np.float32)
        scaled = cv2.warpAffine(source, matrix, (source.shape[1], source.shape[0]), borderValue=255)
        page = Image.fromarray(scaled)
    return page


def generate(
    output_pdf: Path, output_manifest: Path, *, target_id: str = "BOARD_001",
    metric_fiducials: bool = False, content_scale_x: float = 1.0,
    content_scale_y: float = 1.0,
) -> dict:
    page = render_page(
        metric_fiducials=metric_fiducials, target_id=target_id,
        content_scale_x=content_scale_x, content_scale_y=content_scale_y,
    )
    board_mm = (SQUARES[0] * SQUARE_MM, SQUARES[1] * SQUARE_MM)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_pdf, "PDF", resolution=float(DPI))
    digest = hashlib.sha256(output_pdf.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "target_id": target_id,
        "page_mm": list(A4_MM),
        "page_orientation": "portrait",
        "dpi": DPI,
        "dictionary": "DICT_4X4_50",
        "squares_x": SQUARES[0],
        "squares_y": SQUARES[1],
        "square_length_mm": SQUARE_MM,
        "marker_length_mm": MARKER_MM,
        "board_size_mm": list(board_mm),
        "board_top_left_from_page_top_left_mm": list(BOARD_TOP_LEFT_MM),
        "alignment_x_center_from_page_top_left_mm": list(X_CENTER_MM),
        "alignment_x_size_mm": X_SIZE_MM,
        "alignment_x_stroke_mm": X_STROKE_MM,
        "alignment_x_relative_to_board_top_left_mm": [
            X_CENTER_MM[0] - BOARD_TOP_LEFT_MM[0],
            X_CENTER_MM[1] - BOARD_TOP_LEFT_MM[1],
        ],
        "print_instruction": "A4 portrait; Actual size / 100%; disable Fit, Shrink, Scale to page, and borderless enlargement.",
        "pdf_sha256": digest,
        "content_scale_x": content_scale_x,
        "content_scale_y": content_scale_y,
    }
    if metric_fiducials:
        manifest["metric_fiducials"] = {
            "dictionary": METRIC_MARKER_DICTIONARY,
            "marker_size_mm": METRIC_MARKER_SIZE_MM,
            "marker_border_mm": METRIC_MARKER_BORDER_MM,
            "markers": {str(marker_id): list(position) for marker_id, position in METRIC_MARKERS.items()},
            "purpose": "page registration and print-warp diagnostics; absolute scale still uses measured A4 edges",
        }
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--target-id", default="BOARD_001")
    parser.add_argument("--metric-fiducials", action="store_true")
    parser.add_argument("--content-scale-x", type=float, default=1.0)
    parser.add_argument("--content-scale-y", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(generate(
        args.output_pdf, args.output_manifest, target_id=args.target_id,
        metric_fiducials=args.metric_fiducials,
        content_scale_x=args.content_scale_x, content_scale_y=args.content_scale_y,
    ), indent=2))


if __name__ == "__main__":
    main()
