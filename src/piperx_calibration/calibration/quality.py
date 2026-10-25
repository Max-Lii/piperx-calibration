"""Hardware-independent calibration-frame quality measurements."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from piperx_calibration.robotcontrol.camera import CalibrationFrame


@dataclass(frozen=True)
class FrameQuality:
    accepted: bool
    width_px: int
    height_px: int
    mean_brightness: float
    clipped_dark_fraction: float
    clipped_bright_fraction: float
    laplacian_variance: float
    valid_depth_fraction: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class FrameQualityPolicy:
    minimum_laplacian_variance: float = 8.0
    minimum_mean_brightness: float = 10.0
    maximum_mean_brightness: float = 245.0
    maximum_clipped_fraction: float = 0.40
    minimum_valid_depth_fraction: float = 0.50


@dataclass(frozen=True)
class CharucoTargetSpec:
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float

    @classmethod
    def from_manifest(cls, path: Path) -> "CharucoTargetSpec":
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            dictionary=value["dictionary"],
            squares_x=int(value["squares_x"]),
            squares_y=int(value["squares_y"]),
            square_length_m=float(value["square_length_mm"]) / 1000.0,
            marker_length_m=float(value["marker_length_mm"]) / 1000.0,
        )

    def board(self):
        dictionary_id = getattr(cv2.aruco, self.dictionary, None)
        if dictionary_id is None:
            raise ValueError(f"unsupported ArUco dictionary: {self.dictionary}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            dictionary,
        )


@dataclass(frozen=True)
class CharucoQualityPolicy:
    minimum_corners: int = 8
    minimum_hull_area_fraction: float = 0.005
    minimum_border_px: float = 6.0
    maximum_reprojection_error_px: float = 2.0


def evaluate_charuco(
    frame: CalibrationFrame,
    target: CharucoTargetSpec,
    policy: CharucoQualityPolicy = CharucoQualityPolicy(),
) -> dict:
    board = target.board()
    detector = cv2.aruco.CharucoDetector(board)
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _, _ = detector.detectBoard(gray)
    corner_count = 0 if ids is None else int(len(ids))
    result = {
        "accepted": False,
        "corner_count": corner_count,
        "hull_area_fraction": 0.0,
        "minimum_border_px": 0.0,
        "reprojection_error_px": None,
        "reasons": [],
    }
    if corners is None or ids is None or corner_count < policy.minimum_corners:
        result["reasons"].append("insufficient_charuco_corners")
        return result
    points = corners.reshape(-1, 2).astype(np.float64)
    height, width = gray.shape
    hull_area_fraction = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))) / (width * height))
    border = float(np.min(np.column_stack((points[:, 0], width - 1 - points[:, 0], points[:, 1], height - 1 - points[:, 1]))))
    result["hull_area_fraction"] = hull_area_fraction
    result["minimum_border_px"] = border
    if hull_area_fraction < policy.minimum_hull_area_fraction:
        result["reasons"].append("target_too_small")
    if border < policy.minimum_border_px:
        result["reasons"].append("target_near_border")
    object_points = board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        frame.color_intrinsics,
        frame.distortion,
        flags=cv2.SOLVEPNP_ITERATIVE if corner_count >= 6 else cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        result["reasons"].append("pnp_failed")
    else:
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, frame.color_intrinsics, frame.distortion,
        )
        reprojection = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - points) ** 2, axis=1))))
        result["reprojection_error_px"] = reprojection
        if reprojection > policy.maximum_reprojection_error_px:
            result["reasons"].append("high_reprojection_error")
    result["accepted"] = not result["reasons"]
    return result


def evaluate_frame(
    frame: CalibrationFrame,
    policy: FrameQualityPolicy = FrameQualityPolicy(),
) -> FrameQuality:
    rgb = np.asarray(frame.rgb)
    depth = np.asarray(frame.depth_raw)
    reasons: list[str] = []
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("calibration RGB must be uint8 HxWx3")
    if depth.shape != rgb.shape[:2] or depth.dtype != np.uint16:
        raise ValueError("calibration depth must be uint16 and aligned to RGB")
    if frame.color_intrinsics.shape != (3, 3) or not np.all(np.isfinite(frame.color_intrinsics)):
        raise ValueError("color intrinsics must be a finite 3x3 matrix")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    dark = float(np.mean(gray <= 2))
    bright = float(np.mean(gray >= 253))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    valid_depth = float(np.mean(depth > 0))
    if sharpness < policy.minimum_laplacian_variance:
        reasons.append("blurry")
    if brightness < policy.minimum_mean_brightness:
        reasons.append("too_dark")
    if brightness > policy.maximum_mean_brightness:
        reasons.append("too_bright")
    if dark > policy.maximum_clipped_fraction:
        reasons.append("dark_clipping")
    if bright > policy.maximum_clipped_fraction:
        reasons.append("bright_clipping")
    if valid_depth < policy.minimum_valid_depth_fraction:
        reasons.append("insufficient_depth")
    if not frame.depth_aligned_to_color:
        reasons.append("depth_not_aligned_to_color")
    return FrameQuality(
        accepted=not reasons,
        width_px=int(rgb.shape[1]),
        height_px=int(rgb.shape[0]),
        mean_brightness=brightness,
        clipped_dark_fraction=dark,
        clipped_bright_fraction=bright,
        laplacian_variance=sharpness,
        valid_depth_fraction=valid_depth,
        reasons=tuple(reasons),
    )
