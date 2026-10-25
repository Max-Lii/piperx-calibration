"""SE(3) solvers operating only on robotcontrol capture artifacts."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from piperx_calibration.robotcontrol.camera import CalibrationFrame

from .quality import CharucoTargetSpec


def estimate_camera_target(frame: CalibrationFrame, target: CharucoTargetSpec) -> tuple[np.ndarray, dict]:
    board = target.board()
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(frame.rgb)
    # Four non-collinear points are sufficient for the planar PnP solve.
    # More points improve confidence, but a sparse ChArUco view must not
    # invalidate an otherwise useful calibration pose.
    if corners is None or ids is None or len(ids) < 4:
        raise ValueError(f"{frame.camera_id}: insufficient ChArUco corners")
    object_points = board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, corners, frame.color_intrinsics, frame.distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError(f"{frame.camera_id}: ChArUco PnP failed")
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, frame.color_intrinsics, frame.distortion,
    )
    error = float(np.sqrt(np.mean(np.sum(
        (projected.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1,
    ))))
    pixels = np.rint(corners.reshape(-1, 2)).astype(int)
    height, width = frame.depth_raw.shape
    valid_mask = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    valid_pixels = pixels[valid_mask]
    depth_values = frame.depth_raw[valid_pixels[:, 1], valid_pixels[:, 0]] if len(valid_pixels) else np.asarray([])
    nonzero = depth_values > 0
    depth_consistency_mm = None
    if np.any(nonzero):
        expected_depth_m = (object_points @ rotation.T + tvec.reshape(1, 3))[:, 2][valid_mask][nonzero]
        measured_depth_m = depth_values[nonzero].astype(np.float64) * frame.depth_scale_m
        depth_consistency_mm = float(np.median(np.abs(measured_depth_m - expected_depth_m)) * 1000.0)
    return transform, {
        "corner_count": int(len(ids)),
        "reprojection_error_px": error,
        "depth_consistency_mm": depth_consistency_mm,
    }


@dataclass(frozen=True)
class HandEyeObservation:
    camera_target: np.ndarray
    base_ee: np.ndarray


def solve_robot_world_hand_eye(
    observations: list[HandEyeObservation],
    *,
    method: int = cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
) -> dict:
    if len(observations) < 6:
        raise ValueError("robot-world/hand-eye requires at least six accepted poses")
    world_to_camera = [item.camera_target for item in observations]
    base_to_gripper = [np.linalg.inv(item.base_ee) for item in observations]
    r_world_base, t_world_base, r_camera_ee, t_camera_ee = cv2.calibrateRobotWorldHandEye(
        [value[:3, :3] for value in world_to_camera],
        [value[:3, 3:4] for value in world_to_camera],
        [value[:3, :3] for value in base_to_gripper],
        [value[:3, 3:4] for value in base_to_gripper],
        method=method,
    )
    world_base = np.eye(4)
    world_base[:3, :3] = r_world_base
    world_base[:3, 3] = np.asarray(t_world_base).reshape(3)
    camera_ee = np.eye(4)
    camera_ee[:3, :3] = r_camera_ee
    camera_ee[:3, 3] = np.asarray(t_camera_ee).reshape(3)
    return {
        "T_world_base": world_base,
        # OpenCV returns camera<-EE. The public calibration artifact stores
        # EE<-camera, i.e. the camera pose expressed in the EE frame.
        "T_ee_camera": np.linalg.inv(camera_ee),
    }


def hand_eye_validation(
    observations: list[HandEyeObservation],
    *,
    bootstrap_samples: int = 100,
    seed: int = 20260908,
) -> dict:
    """Validate a solve with held-out poses, an independent method and bootstrap."""
    if len(observations) < 12:
        raise ValueError("hand-eye validation requires at least twelve accepted poses")
    full = solve_robot_world_hand_eye(observations)

    held_out_indices = list(range(4, len(observations), 5))
    held_out = [observations[index] for index in held_out_indices]
    training = [item for index, item in enumerate(observations) if index not in held_out_indices]
    trained = solve_robot_world_hand_eye(training)

    def residual(solution: dict, item: HandEyeObservation) -> tuple[float, float]:
        measured_world_camera = np.linalg.inv(item.camera_target)
        predicted_world_camera = solution["T_world_base"] @ item.base_ee @ solution["T_ee_camera"]
        return transform_error(predicted_world_camera, measured_world_camera)

    held_out_errors = [residual(trained, item) for item in held_out]
    li = solve_robot_world_hand_eye(observations, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)
    base_agreement = transform_error(li["T_world_base"], full["T_world_base"])
    handeye_agreement = transform_error(li["T_ee_camera"], full["T_ee_camera"])

    rng = np.random.default_rng(seed)
    base_bootstrap: list[tuple[float, float]] = []
    handeye_bootstrap: list[tuple[float, float]] = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(observations), size=len(observations))
        if len(np.unique(indices)) < 6:
            continue
        try:
            solved = solve_robot_world_hand_eye([observations[int(index)] for index in indices])
        except cv2.error:
            continue
        base_bootstrap.append(transform_error(solved["T_world_base"], full["T_world_base"]))
        handeye_bootstrap.append(transform_error(solved["T_ee_camera"], full["T_ee_camera"]))
    if len(base_bootstrap) < max(20, bootstrap_samples // 2):
        raise RuntimeError("too few valid bootstrap hand-eye solves")

    def percentile(values: list[tuple[float, float]], axis: int) -> float:
        return float(np.percentile([value[axis] for value in values], 95))

    return {
        "held_out_poses": len(held_out),
        "held_out_translation_median_mm": float(np.median([value[0] for value in held_out_errors])),
        "held_out_translation_max_mm": float(np.max([value[0] for value in held_out_errors])),
        "held_out_rotation_median_deg": float(np.median([value[1] for value in held_out_errors])),
        "held_out_rotation_max_deg": float(np.max([value[1] for value in held_out_errors])),
        "solver_agreement": {
            "methods": ["SHAH", "LI"],
            "base_translation_mm": base_agreement[0],
            "base_rotation_deg": base_agreement[1],
            "handeye_translation_mm": handeye_agreement[0],
            "handeye_rotation_deg": handeye_agreement[1],
        },
        "bootstrap": {
            "requested_samples": bootstrap_samples,
            "valid_samples": len(base_bootstrap),
            "base_translation_p95_mm": percentile(base_bootstrap, 0),
            "base_rotation_p95_deg": percentile(base_bootstrap, 1),
            "handeye_translation_p95_mm": percentile(handeye_bootstrap, 0),
            "handeye_rotation_p95_deg": percentile(handeye_bootstrap, 1),
        },
    }


def transform_error(estimated: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(truth) @ estimated
    translation_mm = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return translation_mm, float(np.degrees(np.arccos(cosine)))
