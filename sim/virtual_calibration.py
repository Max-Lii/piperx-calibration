"""Rendered-image Robot-World/Hand-Eye regression, with hidden ground truth.

This is deliberately a small calibration-core proof before attaching the same
interfaces to a full RoboTwin scene.  The solver sees only rendered ChArUco
RGB, intrinsics, and robot FK observations.  Simulator truth is retained only
by the evaluator, so a passing result proves the coordinate, PnP and hand-eye
chain rather than merely echoing a configured transform.

Conventions: ``T_parent_child`` maps child-frame coordinates into parent-frame
coordinates; length is metres and angles are radians.  OpenCV camera axes are
used throughout this module (x right, y down, z forward).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from piperx_calibration.calibration.geometry import HandEyeObservation, solve_robot_world_hand_eye


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _inverse(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return _transform(rotation.T, -rotation.T @ translation)


def _pose_error(estimated: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    translation_m = float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
    delta = estimated[:3, :3] @ truth[:3, :3].T
    rotation_deg = float(np.degrees(Rotation.from_matrix(delta).magnitude()))
    return translation_m, rotation_deg


def _look_at_opencv(camera_position_w: np.ndarray, target_w: np.ndarray) -> np.ndarray:
    """Return ``T_world_camera`` for an OpenCV optical camera."""
    z_axis = target_w - camera_position_w
    z_axis /= np.linalg.norm(z_axis)
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(z_axis, world_up)
    if np.linalg.norm(x_axis) < 1e-7:
        world_up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, world_up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return _transform(np.column_stack((x_axis, y_axis, z_axis)), camera_position_w)


@dataclass(frozen=True)
class VirtualCalibrationResult:
    observations: int
    detections: int
    base_translation_error_mm: float
    base_rotation_error_deg: float
    handeye_translation_error_mm: float
    handeye_rotation_error_deg: float
    pnp_translation_median_mm: float
    pnp_rotation_median_deg: float
    passed: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "observations": self.observations,
            "detections": self.detections,
            "base_translation_error_mm": self.base_translation_error_mm,
            "base_rotation_error_deg": self.base_rotation_error_deg,
            "handeye_translation_error_mm": self.handeye_translation_error_mm,
            "handeye_rotation_error_deg": self.handeye_rotation_error_deg,
            "pnp_translation_median_mm": self.pnp_translation_median_mm,
            "pnp_rotation_median_deg": self.pnp_rotation_median_deg,
            "passed": self.passed,
        }


class RenderedCharucoRobotWorldHarness:
    """A hidden-truth rendered camera source for calibration-core regression.

    ``T_world_base`` and ``T_ee_camera`` are deliberately not passed to the
    calibration function.  The generated ``T_ee_base`` values have the same
    direction as a robot FK adapter should expose to OpenCV.
    """

    width = 640
    height = 480
    squares_x = 7
    squares_y = 9
    square_length_m = 0.024
    marker_length_m = 0.018

    def __init__(self, *, seed: int = 7, output_dir: Path | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.output_dir = output_dir
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_length_m, self.marker_length_m, dictionary,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.board_image = self.board.generateImage((self.squares_x * 160, self.squares_y * 160), marginSize=0)
        self.intrinsic = np.array(
            [[1050.0, 0.0, self.width / 2], [0.0, 1050.0, self.height / 2], [0.0, 0.0, 1.0]], dtype=np.float64,
        )
        self.distortion = np.zeros((5, 1), dtype=np.float64)

        # Ground truth: an intentionally nontrivial workcell placement and
        # camera mount.  These values are evaluator-only, never solver input.
        self._truth_world_base = _transform(
            Rotation.from_euler("xyz", [0.04, -0.03, 0.22]).as_matrix(), np.array([-0.29, 0.08, 0.735]),
        )
        self._truth_ee_camera = _transform(
            Rotation.from_euler("xyz", [-1.52, 0.18, 0.11]).as_matrix(), np.array([0.036, -0.051, 0.079]),
        )
        # The printed board's +Y is image-down, while the procedural scene's
        # physical plane uses +Y upward.  This is the explicit frame boundary
        # a real A4 target definition must carry; it is not hidden in PnP.
        self._physical_world_target = _transform(
            np.diag([1.0, -1.0, -1.0]), np.array([0.0, self.squares_y * self.square_length_m, 0.0]),
        )

    def _render(self, transform_camera_world: np.ndarray, *, index: int) -> np.ndarray:
        board_w = self.squares_x * self.square_length_m
        board_h = self.squares_y * self.square_length_m
        corners_w = np.array(
            [[0.0, 0.0, 0.0], [board_w, 0.0, 0.0], [board_w, board_h, 0.0], [0.0, board_h, 0.0]],
            dtype=np.float64,
        )
        rvec, _ = cv2.Rodrigues(transform_camera_world[:3, :3])
        projected, _ = cv2.projectPoints(corners_w, rvec, transform_camera_world[:3, 3], self.intrinsic, self.distortion)
        destination = projected.reshape(4, 2).astype(np.float32)
        source = np.array(
            [[0, 0], [self.board_image.shape[1] - 1, 0], [self.board_image.shape[1] - 1, self.board_image.shape[0] - 1], [0, self.board_image.shape[0] - 1]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(source, destination)
        image = np.full((self.height, self.width), 235, dtype=np.uint8)
        # The target's physical +Y convention is the image-down direction.
        # Flip once before projection so ArUco codes are not mirror-imaged.
        board_visual = cv2.flip(self.board_image, 0)
        warped = cv2.warpPerspective(
            board_visual, homography, (self.width, self.height), flags=cv2.INTER_NEAREST, borderValue=235,
        )
        mask = cv2.warpPerspective(
            np.full_like(self.board_image, 255), homography, (self.width, self.height), flags=cv2.INTER_NEAREST,
        )
        image[mask > 0] = warped[mask > 0]
        # Preserve marker edges for the core regression.  Pixel-noise sweeps
        # belong to a later robustness test, not the first correctness gate.
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.output_dir / f"render_{index:02d}.png"), rgb)
        return rgb

    def _sample_fk_pose(self, index: int, count: int) -> np.ndarray:
        """Generate diverse ``T_base_ee`` poses while keeping the board visible."""
        azimuth = -1.10 + 2.20 * index / max(count - 1, 1)
        elevation = 0.48 + 0.20 * np.sin(index * 1.7)
        radius = 0.31 + 0.06 * np.cos(index * 1.1)
        target_w = np.array([self.squares_x * self.square_length_m / 2, self.squares_y * self.square_length_m / 2, 0.0])
        position_w = target_w + np.array([radius * np.cos(azimuth), radius * np.sin(azimuth), elevation])
        world_camera = _look_at_opencv(position_w, target_w)
        return _inverse(self._truth_world_base) @ world_camera @ _inverse(self._truth_ee_camera)

    def run(self, *, observations: int = 14) -> VirtualCalibrationResult:
        if observations < 6:
            raise ValueError("at least six diverse poses are required for this regression")
        world2camera_rotations: list[np.ndarray] = []
        world2camera_translations: list[np.ndarray] = []
        base2gripper_rotations: list[np.ndarray] = []
        base2gripper_translations: list[np.ndarray] = []
        pnp_errors: list[tuple[float, float]] = []

        for index in range(observations):
            base_ee = self._sample_fk_pose(index, observations)
            world_camera = self._truth_world_base @ base_ee @ self._truth_ee_camera
            camera_world_truth = _inverse(world_camera)
            camera_target_truth = camera_world_truth @ self._physical_world_target
            image = self._render(camera_world_truth, index=index)
            corners, ids, _, _ = self.detector.detectBoard(image)
            if ids is None or corners is None or len(ids) < 8:
                continue
            object_points = self.board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
            ok, rvec, translation = cv2.solvePnP(object_points, corners, self.intrinsic, self.distortion, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            camera_world_pnp = _transform(rotation, translation.reshape(3))
            pnp_errors.append(_pose_error(camera_world_pnp, camera_target_truth))
            world2camera_rotations.append(rotation)
            world2camera_translations.append(translation.reshape(3, 1))
            # OpenCV requires T_gripper_base, whereas FK naturally exposes T_base_ee.
            ee_base = _inverse(base_ee)
            base2gripper_rotations.append(ee_base[:3, :3])
            base2gripper_translations.append(ee_base[:3, 3:4])

        if len(world2camera_rotations) < 6:
            raise RuntimeError(f"only {len(world2camera_rotations)}/{observations} rendered frames detected ChArUco")
        solver_observations = [
            HandEyeObservation(
                camera_target=_transform(camera_rotation, camera_translation.reshape(3)),
                base_ee=_inverse(_transform(ee_base_rotation, ee_base_translation.reshape(3))),
            )
            for camera_rotation, camera_translation, ee_base_rotation, ee_base_translation in zip(
                world2camera_rotations,
                world2camera_translations,
                base2gripper_rotations,
                base2gripper_translations,
                strict=True,
            )
        ]
        solved = solve_robot_world_hand_eye(solver_observations)
        estimated_world_base = solved["T_world_base"]
        estimated_ee_camera = solved["T_ee_camera"]
        truth_target_base = _inverse(self._physical_world_target) @ self._truth_world_base
        base_error = _pose_error(estimated_world_base, truth_target_base)
        handeye_error = _pose_error(estimated_ee_camera, self._truth_ee_camera)
        pnp_array = np.asarray(pnp_errors)
        result = VirtualCalibrationResult(
            observations=observations,
            detections=len(world2camera_rotations),
            base_translation_error_mm=base_error[0] * 1000.0,
            base_rotation_error_deg=base_error[1],
            handeye_translation_error_mm=handeye_error[0] * 1000.0,
            handeye_rotation_error_deg=handeye_error[1],
            pnp_translation_median_mm=float(np.median(pnp_array[:, 0]) * 1000.0),
            pnp_rotation_median_deg=float(np.median(pnp_array[:, 1])),
            passed=base_error[0] < 0.004 and base_error[1] < 0.5 and handeye_error[0] < 0.004 and handeye_error[1] < 0.5,
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = RenderedCharucoRobotWorldHarness(seed=args.seed, output_dir=args.output_dir).run(observations=args.observations)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    if not result.passed:
        raise SystemExit("virtual calibration did not meet recovery thresholds")


if __name__ == "__main__":
    main()
