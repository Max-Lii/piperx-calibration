from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from .board_verification import aggregate_board_photos, analyze_board_photo, detect_metric_fiducials
from piperx_calibration.robotcontrol.camera import CalibrationFrame
from .run_robotwin_e2e import (
    _accepted_frame_count, _average_transforms, _geometry_quality, _merge_gripper_records,
)


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "assets/targets/BOARD_001_A4_Charuco_5x7_35mm.json"


def _synthetic_page(scale_x: float = 1.0, scale_y: float = 1.0) -> bytes:
    manifest = json.loads(MANIFEST.read_text())
    pixels_per_mm = 4
    page = np.full((297 * pixels_per_mm, 210 * pixels_per_mm), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, manifest["dictionary"]))
    board = cv2.aruco.CharucoBoard(
        (manifest["squares_x"], manifest["squares_y"]), 35.0, 26.25, dictionary,
    )
    width = round(175 * pixels_per_mm * scale_x)
    height = round(245 * pixels_per_mm * scale_y)
    image = board.generateImage((width, height), marginSize=0, borderBits=1)
    x, y = round(8 * pixels_per_mm), round(26 * pixels_per_mm)
    page[y:y + height, x:x + width] = image
    source = np.float32([[0, 0], [page.shape[1] - 1, 0], [page.shape[1] - 1, page.shape[0] - 1], [0, page.shape[0] - 1]])
    destination = np.float32([[170, 90], [925, 170], [850, 1330], [80, 1240]])
    canvas = np.full((1420, 1040), 45, np.uint8)
    warped = cv2.warpPerspective(page, cv2.getPerspectiveTransform(source, destination), (canvas.shape[1], canvas.shape[0]), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    ok, encoded = cv2.imencode(".png", warped)
    assert ok
    return encoded.tobytes()


def _metric_fiducial_page() -> np.ndarray:
    """Build the small marker-only fixture needed by the detector test."""
    pixels_per_mm = 4
    page = np.full((297 * pixels_per_mm, 210 * pixels_per_mm), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size, border = 12 * pixels_per_mm, 2 * pixels_per_mm
    markers = {40: (2, 3), 41: (192, 3), 42: (2, 279), 43: (192, 279)}
    for marker_id, (x_mm, y_mm) in markers.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        canvas = np.full((marker_size + 2 * border, marker_size + 2 * border), 255, np.uint8)
        canvas[border:-border, border:-border] = marker
        x, y = round(x_mm * pixels_per_mm), round(y_mm * pixels_per_mm)
        page[y:y + canvas.shape[0], x:x + canvas.shape[1]] = canvas
    return page


def test_a4_photo_recovers_print_scale_and_aggregate_gate():
    analyses = [analyze_board_photo(_synthetic_page(), MANIFEST) for _ in range(3)]
    result = aggregate_board_photos(analyses)
    assert result["accepted"]
    assert abs(result["scale_x"] - 1.0) < 0.003
    assert abs(result["scale_y"] - 1.0) < 0.003

    shrunk = [analyze_board_photo(_synthetic_page(0.98, 0.98), MANIFEST) for _ in range(3)]
    rejected = aggregate_board_photos(shrunk)
    assert not rejected["accepted"]
    assert "printer_scale_out_of_tolerance" in rejected["reasons"]


def test_phone_view_spread_uses_realistic_gate_without_relaxing_scale_limit():
    # ±0.3% phone-view variation is plausible after page-edge detection; its
    # median is still checked against the stricter ±0.5% actual-print gate.
    passing = aggregate_board_photos([
        {"corner_count": 24, "scale_x": 1.0021, "scale_y": 0.9961},
        {"corner_count": 24, "scale_x": 0.9978, "scale_y": 0.9993},
        {"corner_count": 24, "scale_x": 0.9957, "scale_y": 0.9984},
    ])
    assert passing["accepted"]
    rejected = aggregate_board_photos([
        {"corner_count": 24, "scale_x": 0.990, "scale_y": 0.990},
        {"corner_count": 24, "scale_x": 0.990, "scale_y": 0.990},
        {"corner_count": 24, "scale_x": 0.990, "scale_y": 0.990},
    ])
    assert "printer_scale_out_of_tolerance" in rejected["reasons"]


def test_metric_fiducials_survive_perspective_render():
    image = _metric_fiducial_page()
    source = np.float32([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]])
    destination = np.float32([[80, 45], [930, 120], [860, 1370], [40, 1290]])
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, destination), (1000, 1420), borderValue=45)
    manifest = {
        "metric_fiducials": {
            "dictionary": "DICT_4X4_50",
            "markers": {"40": [2, 3], "41": [192, 3], "42": [2, 279], "43": [192, 279]},
        },
    }
    assert set(detect_metric_fiducials(cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR), manifest)) == {40, 41, 42, 43}


def test_closed_open_frames_are_fused_as_one_pose_observation():
    transforms = []
    for offset in (-0.001, 0.0, 0.001):
        value = np.eye(4)
        value[:3, :3], _ = cv2.Rodrigues(np.asarray([0.0, 0.0, offset]))
        value[0, 3] = offset
        transforms.append(value)
    fused = _average_transforms(transforms)
    assert fused[:3, 3] == pytest.approx([0.0, 0.0, 0.0])
    assert fused[:3, :3] == pytest.approx(np.eye(3), abs=1e-9)

    def record(state):
        return {
            "name": f"left_001_{state}", "metadata": {"gripper_state": state},
            "frames": {
                camera: [{"file": f"{camera}_{state}_{index}.npz"} for index in range(3)]
                for camera in ("left_wrist", "right_wrist", "third_view")
            },
        }
    merged = _merge_gripper_records("left_001", [record("closed"), record("open")])
    assert len(merged["frames"]["left_wrist"]) == 6
    assert [item["gripper_state"] for item in merged["frames"]["left_wrist"]] == ["closed"] * 3 + ["open"] * 3

    for entry in merged["frames"]["left_wrist"]:
        entry.update({"quality": {"accepted": False}, "target_quality": {"accepted": False}})
    merged["frames"]["left_wrist"][4].update({
        "quality": {"accepted": True}, "target_quality": {"accepted": True},
    })
    assert _accepted_frame_count(merged, "left_wrist") == 1


def test_geometry_quality_tolerates_redundant_rejected_poses():
    def arm(accepted):
        return {
            "accepted_poses": accepted,
            "reprojection_error_px_max": 0.2,
            "depth_consistency_median_mm": 0.3,
            "validation": {
                "held_out_translation_max_mm": 1.0,
                "held_out_rotation_max_deg": 0.1,
                "solver_agreement": {"base_translation_mm": 1.0, "handeye_translation_mm": 1.0},
                "bootstrap": {"base_translation_p95_mm": 1.0, "handeye_translation_p95_mm": 1.0},
            },
        }

    third = {"reprojection_error_px": 0.2, "depth_consistency_mm": 0.3}
    rejected = {"left": [{}] * 12, "right": [{}] * 12}
    assert _geometry_quality(arm(36), arm(36), third, rejected, 48)["passed"]
    warning = _geometry_quality(arm(5), arm(36), third, rejected, 48)
    assert warning["passed"]
    assert warning["reasons"] == []
    assert warning["warnings"] == ["left_insufficient_accepted_poses"]
