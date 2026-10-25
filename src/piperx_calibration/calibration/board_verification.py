"""Verify the physical size of a metric A4 ChArUco board from phone/tablet photos.

The A4 page is the metric reference.  Perspective is removed with its four
outer corners, then detected ChArUco corners are compared with the PDF
manifest.  This catches both uniform ``Fit to page`` scaling and unequal X/Y
printer scaling; it does not pretend that a photograph can diagnose paper
curl, so several views are required by the workflow.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardVerificationPolicy:
    minimum_images: int = 3
    minimum_corners: int = 12
    maximum_scale_error_fraction: float = 0.005
    maximum_anisotropy_fraction: float = 0.003
    # Phone photos include lens residuals and sub-pixel uncertainty in the
    # outer-page contour.  The median still has a strict 0.5% absolute scale
    # gate; requiring three independent views to agree within 0.3% made valid
    # 1:1 A4 captures fail despite sub-millimetre rectified residuals.
    maximum_spread_fraction: float = 0.010


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    # TL, TR, BR, BL (the angular order can start by going down).
    if ordered[1, 0] < ordered[-1, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def detect_a4_page(image_bgr: np.ndarray, contained_points: np.ndarray | None = None) -> np.ndarray:
    """Return TL/TR/BR/BL page corners using contrast, geometry and containment."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    candidates: list[tuple[float, np.ndarray]] = []
    image_area = float(gray.shape[0] * gray.shape[1])
    for source_weight, binary in (
        (2.0, cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        (1.0, cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 5)),
    ):
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            if not 0.12 * image_area <= area <= 0.98 * image_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            poly = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(poly) != 4 or not cv2.isContourConvex(poly):
                continue
            quad = _order_quad(poly)
            if contained_points is not None and any(
                cv2.pointPolygonTest(quad, tuple(map(float, point)), False) < 0
                for point in contained_points.reshape(-1, 2)
            ):
                continue
            widths = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
            heights = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
            ratio = max(widths, heights) / max(min(widths, heights), 1e-6)
            ratio_error = abs(ratio - 297.0 / 210.0)
            candidates.append((source_weight * area / (1.0 + 8.0 * ratio_error), quad))
    if not candidates:
        raise ValueError("未检测到完整 A4 外边界；请放在有对比的平面上并让四角全部入镜")
    return max(candidates, key=lambda item: item[0])[1]


def detect_metric_fiducials(image_bgr: np.ndarray, manifest: dict) -> dict[int, np.ndarray]:
    config = manifest.get("metric_fiducials")
    if not config:
        return {}
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["dictionary"]))
    detected = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image_bgr)
    corners, ids = detected[:2]
    if ids is None:
        return {}
    wanted = {int(marker_id) for marker_id in config["markers"]}
    return {
        int(marker_id): np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
        for marker_corners, marker_id in zip(corners, ids.reshape(-1), strict=True)
        if int(marker_id) in wanted
    }


def decode_scale_qr(image_bgr: np.ndarray) -> dict | None:
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(image_bgr)
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return value if isinstance(value, dict) else {"raw": text}


def _fiducial_page_residual_mm(
    fiducials: dict[int, np.ndarray],
    manifest: dict,
    page_transform: np.ndarray,
    *,
    rotated_page: bool,
) -> float | None:
    if not fiducials:
        return None
    config = manifest["metric_fiducials"]
    offset = (float(config["marker_size_mm"]) + 2.0 * float(config["marker_border_mm"])) / 2.0
    observed = np.float32([np.mean(fiducials[marker_id], axis=0) for marker_id in sorted(fiducials)])
    rectified = cv2.perspectiveTransform(observed.reshape(-1, 1, 2), page_transform).reshape(-1, 2)
    if rotated_page:
        page_h = float(manifest["page_mm"][1])
        rectified = np.column_stack((rectified[:, 1], page_h - rectified[:, 0]))
    expected = np.asarray([
        [float(config["markers"][str(marker_id)][0]) + offset,
         float(config["markers"][str(marker_id)][1]) + offset]
        for marker_id in sorted(fiducials)
    ])
    return float(np.sqrt(np.mean(np.sum((rectified - expected) ** 2, axis=1))))


def _fiducial_content_scale(
    fiducials: dict[int, np.ndarray], manifest: dict, page_transform: np.ndarray,
    *, rotated_page: bool,
) -> tuple[float, float, float] | None:
    """Estimate printed-content X/Y scale from page-registered marker centers.

    The paper dimensions provide the absolute page frame. The marker centers
    then measure how the printed content moved within that fixed frame, so a
    uniform printer shrink is not normalized away.
    """
    if len(fiducials) < 4:
        return None
    config = manifest["metric_fiducials"]
    marker_offset = (float(config["marker_size_mm"]) + 2.0 * float(config["marker_border_mm"])) / 2.0
    observed = np.concatenate([fiducials[marker_id] for marker_id in sorted(fiducials)], axis=0).astype(np.float32)
    rectified = cv2.perspectiveTransform(observed.reshape(-1, 1, 2), page_transform).reshape(-1, 2)
    page_h = float(manifest["page_mm"][1])
    if rotated_page:
        rectified = np.column_stack((rectified[:, 1], page_h - rectified[:, 0]))
    nominal = np.concatenate([
        np.asarray([
            [float(config["markers"][str(marker_id)][0]) + marker_offset,
             float(config["markers"][str(marker_id)][1]) + marker_offset],
            [float(config["markers"][str(marker_id)][0]) + marker_offset + float(config["marker_size_mm"]),
             float(config["markers"][str(marker_id)][1]) + marker_offset],
            [float(config["markers"][str(marker_id)][0]) + marker_offset + float(config["marker_size_mm"]),
             float(config["markers"][str(marker_id)][1]) + marker_offset + float(config["marker_size_mm"])],
            [float(config["markers"][str(marker_id)][0]) + marker_offset,
             float(config["markers"][str(marker_id)][1]) + marker_offset + float(config["marker_size_mm"])],
        ], dtype=np.float64)
        for marker_id in sorted(fiducials)
    ], axis=0)
    design = np.column_stack((nominal, np.ones(len(nominal))))
    affine, *_ = np.linalg.lstsq(design, rectified, rcond=None)
    linear = affine[:2, :].T
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    residual = float(np.sqrt(np.mean(np.sum((design @ affine - rectified) ** 2, axis=1))))
    return scale_x, scale_y, residual


def analyze_board_photo(image_bytes: bytes, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码照片；请上传原始 JPEG/PNG/HEIC 转换后的 JPEG")
    qr_metadata = decode_scale_qr(image)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, manifest["dictionary"]))
    board = cv2.aruco.CharucoBoard(
        (manifest["squares_x"], manifest["squares_y"]),
        manifest["square_length_mm"], manifest["marker_length_mm"], dictionary,
    )
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
    if corners is None or ids is None or len(ids) < 8:
        raise ValueError("检测到的 ChArUco 角点不足；请靠近、对焦并避免反光")
    page_w, page_h = map(float, manifest["page_mm"])
    fiducials = detect_metric_fiducials(image, manifest)
    # Printed fiducials cannot establish absolute scale by themselves: a
    # printer's uniform shrink scales them with the ChArUco board. The A4
    # physical dimensions remain the metric reference, while the markers make
    # page registration and post-rectification warp diagnostics more robust.
    page = detect_a4_page(image, corners)
    horizontal = np.linalg.norm(page[1] - page[0])
    vertical = np.linalg.norm(page[3] - page[0])
    if horizontal <= vertical:
        dst = np.float32([[0, 0], [page_w, 0], [page_w, page_h], [0, page_h]])
    else:
        dst = np.float32([[0, 0], [page_h, 0], [page_h, page_w], [0, page_w]])
    page_transform = cv2.getPerspectiveTransform(page, dst)
    # The printable artifact is portrait.  The ordered quad's long direction
    # may be horizontal if the phone was rotated, so choose the destination
    # orientation which best matches its edge lengths.
    rectified = cv2.perspectiveTransform(corners.astype(np.float32), page_transform).reshape(-1, 2)
    nominal_local = board.getChessboardCorners()[ids.reshape(-1), :2].astype(np.float64)
    offset = np.asarray(manifest["board_top_left_from_page_top_left_mm"], dtype=np.float64)
    nominal = nominal_local + offset
    if horizontal > vertical:
        # Destination coordinates are rotated with the photograph.  Rotate
        # them back into the manifest's portrait page frame.
        rectified = np.column_stack((rectified[:, 1], page_h - rectified[:, 0]))
    fiducial_residual = _fiducial_page_residual_mm(
        fiducials, manifest, page_transform, rotated_page=horizontal > vertical,
    )
    fiducial_scale = _fiducial_content_scale(
        fiducials, manifest, page_transform, rotated_page=horizontal > vertical,
    )
    design = np.column_stack((nominal_local, np.ones(len(nominal_local))))
    affine, *_ = np.linalg.lstsq(design, rectified, rcond=None)
    linear = affine[:2, :].T
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    predicted = design @ affine
    residual = float(np.sqrt(np.mean(np.sum((predicted - rectified) ** 2, axis=1))))
    return {
        "corner_count": int(len(ids)),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "uniform_scale": (scale_x + scale_y) / 2.0,
        "anisotropy": abs(scale_x - scale_y),
        "rectified_rms_mm": residual,
        "page_corners_px": page.tolist(),
        "metric_fiducial_ids": sorted(fiducials),
        "metric_fiducial_count": len(fiducials),
        "metric_fiducial_page_residual_mm": fiducial_residual,
        "metric_fiducial_scale_x": None if fiducial_scale is None else fiducial_scale[0],
        "metric_fiducial_scale_y": None if fiducial_scale is None else fiducial_scale[1],
        "metric_fiducial_content_residual_mm": None if fiducial_scale is None else fiducial_scale[2],
        "absolute_scale_source": "physical_page_dimensions",
        "absolute_scale_warning": (
            "Printed fiducials diagnose registration and anisotropy but cannot independently reveal "
            "uniform printer scale; an external measured page/gauge dimension is required."
            if fiducials else None
        ),
        "image_size_px": [int(image.shape[1]), int(image.shape[0])],
        "qr_metadata": qr_metadata,
    }


def aggregate_board_photos(items: list[dict], policy: BoardVerificationPolicy = BoardVerificationPolicy()) -> dict:
    valid = [item for item in items if "error" not in item and item.get("corner_count", 0) >= policy.minimum_corners]
    if not valid:
        return {"accepted": False, "valid_images": 0, "total_images": len(items), "reasons": ["no_valid_images"], "images": items}
    use_fiducials = all(item.get("metric_fiducial_count", 0) >= 4 for item in valid)
    sx = np.asarray([
        item["metric_fiducial_scale_x"] if use_fiducials else item["scale_x"]
        for item in valid
    ])
    sy = np.asarray([
        item["metric_fiducial_scale_y"] if use_fiducials else item["scale_y"]
        for item in valid
    ])
    median_x, median_y = float(np.median(sx)), float(np.median(sy))
    spread = float(max(np.ptp(sx), np.ptp(sy))) if len(valid) > 1 else 0.0
    reasons = []
    if len(valid) < policy.minimum_images:
        reasons.append("need_at_least_three_valid_photos")
    if max(abs(median_x - 1.0), abs(median_y - 1.0)) > policy.maximum_scale_error_fraction:
        reasons.append("printer_scale_out_of_tolerance")
    if abs(median_x - median_y) > policy.maximum_anisotropy_fraction:
        reasons.append("printer_xy_scale_mismatch")
    if spread > policy.maximum_spread_fraction:
        reasons.append("photo_measurements_inconsistent")
    return {
        "accepted": not reasons,
        "valid_images": len(valid), "total_images": len(items),
        "scale_x": median_x, "scale_y": median_y,
        "scale_source": "metric_fiducials_registered_to_physical_page" if use_fiducials else "charuco_registered_to_physical_page",
        "uniform_scale": (median_x + median_y) / 2.0,
        "anisotropy": abs(median_x - median_y), "measurement_spread": spread,
        "reasons": reasons, "images": items,
    }
