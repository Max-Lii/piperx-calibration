"""Find a conservative, rendered-RoboTwin wrist-camera calibration pose library.

The resulting JSON is a *candidate library*, not an automatic permission to
move hardware. Every pose must still be checked on the physical Piper using
the slow, guarded ``move_to_joint_target`` path before it is enabled for a
calibration run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from sim.robotwin.robotwin_rendered import RoboTwinRenderedHandEyeHarness
from sim.robotwin.robotwin_dual_tabletop import BOARD_CENTER_W, make_printable_target_mesh


@dataclass(frozen=True)
class VisiblePose:
    qpos_rad: list[float]
    charuco_corners: int
    board_center_px: list[float]
    reprojection_error_px: float


def _halton(count: int, dimension: int) -> np.ndarray:
    """Deterministic low-discrepancy candidates; avoids an unsafe random scan."""
    primes = (2, 3, 5, 7, 11, 13)
    if dimension > len(primes):
        raise ValueError("Halton dimension exceeds configured prime bases")
    result = np.empty((count, dimension), dtype=np.float64)
    for axis, base in enumerate(primes[:dimension]):
        for index in range(count):
            value, denominator, integer = 0.0, 1.0, index + 1
            while integer:
                integer, remainder = divmod(integer, base)
                denominator *= base
                value += remainder / denominator
            result[index, axis] = value
    return result


class RoboTwinCalibrationPoseScanner:
    """Scan a bounded neighbourhood of the known observable home pose."""

    # This is the downward-facing table-observation anchor used by the
    # robotcontrol pose scanner.  The robot starts in a separate park/home
    # pose; it must move here before the wrist camera can see the tabletop.
    home_qpos = np.array([-.45, .25, -.55, 1.0, 0.0, 0.0], dtype=np.float64)
    offset_limit = np.array([.20, .22, .24, .42, .42, .55], dtype=np.float64)

    def __init__(self, output_dir: Path, *, wanted: int = 32, candidates: int = 96, arm: str = "left") -> None:
        if wanted < 6:
            raise ValueError("at least six poses are needed for hand-eye calibration")
        if candidates < wanted:
            raise ValueError("candidates must be at least wanted")
        if arm not in {"left", "right"}:
            raise ValueError("arm must be left or right")
        self.output_dir, self.wanted, self.candidates, self.arm = output_dir, wanted, candidates, arm
        if arm == "right":
            self.home_qpos = self.home_qpos.copy()
            self.home_qpos[0] *= -1.0
        self.harness = RoboTwinRenderedHandEyeHarness(output_dir=None, arm=arm)

    def _candidate_qpos(self) -> list[np.ndarray]:
        offsets = (2.0 * _halton(self.candidates - 1, 6) - 1.0) * self.offset_limit
        return [self.home_qpos.copy(), *(self.home_qpos + row for row in offsets)]

    @staticmethod
    def _quality(image: np.ndarray, detector, board, intrinsic: np.ndarray) -> tuple[VisiblePose, np.ndarray] | None:
        corners, ids, _, _ = detector.detectBoard(image)
        if corners is None or ids is None or len(ids) < 12:
            return None
        object_points = board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
        ok, rvec, translation = cv2.solvePnP(object_points, corners, intrinsic, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        projected, _ = cv2.projectPoints(object_points, rvec, translation, intrinsic, None)
        reprojection = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))
        centre = np.mean(corners.reshape(-1, 2), axis=0)
        if reprojection > 1.5:
            return None
        return VisiblePose([], int(len(ids)), [float(centre[0]), float(centre[1])], reprojection), translation.reshape(3)

    def _select_diverse(self, visible: list[VisiblePose]) -> list[VisiblePose]:
        """Farthest-point selection in normalised joint + image-centre space."""
        if len(visible) <= self.wanted:
            return visible
        qpos = np.asarray([item.qpos_rad for item in visible], dtype=np.float64) / self.offset_limit
        image = np.asarray([item.board_center_px for item in visible], dtype=np.float64) / np.array([640.0, 480.0])
        features = np.concatenate((qpos, 0.75 * image), axis=1)
        chosen = [0]
        min_distance = np.linalg.norm(features - features[0], axis=1)
        while len(chosen) < self.wanted:
            candidate = int(np.argmax(min_distance))
            chosen.append(candidate)
            min_distance = np.minimum(min_distance, np.linalg.norm(features - features[candidate], axis=1))
        return [visible[index] for index in chosen]

    @staticmethod
    def _draw_crosshair(image: np.ndarray, point: list[float]) -> np.ndarray:
        annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        x, y = (round(value) for value in point)
        cv2.drawMarker(annotated, (x, y), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=34, thickness=2)
        cv2.putText(annotated, "Place board centre here", (x + 15, max(24, y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        return annotated

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        env = self.harness._make_env()
        visible: list[VisiblePose] = []
        placement_image: np.ndarray | None = None
        try:
            env.reset(init_id=0)
            task = env._task
            assert task is not None
            mesh_path, _ = make_printable_target_mesh(self.output_dir)
            import sapien
            from scipy.spatial.transform import Rotation
            from sim.virtual_calibration import _inverse

            builder = task.scene.create_actor_builder(); builder.set_physx_body_type("static")
            builder.add_visual_from_file(filename=str(mesh_path)); actor = builder.build(name="dev_pose_scan_charuco")
            # The physical calibration board is fixed flat on the tabletop;
            # the arm moves into the downward-facing anchor before sampling.
            actor.set_pose(sapien.Pose(BOARD_CENTER_W, [1, 0, 0, 0]))
            for index, qpos in enumerate(self._candidate_qpos()):
                self.harness._drive_arm(task, qpos.tolist(), self.arm)
                image = np.asarray(task.get_obs()["observation"][f"{self.arm}_camera"]["rgb"])
                camera = task.cameras.get_config()[f"{self.arm}_camera"]
                measured = self._quality(image, self.harness.detector, self.harness.board, np.asarray(camera["intrinsic_cv"], dtype=np.float64))
                if measured is None:
                    continue
                quality, _ = measured
                quality = VisiblePose([float(value) for value in qpos], quality.charuco_corners, quality.board_center_px, quality.reprojection_error_px)
                visible.append(quality)
                if placement_image is None:
                    placement_image = image.copy()
            if not visible:
                raise RuntimeError("no sampled calibration pose can see the tabletop ChArUco board")
            selected = self._select_diverse(visible)
            if len(selected) < self.wanted:
                raise RuntimeError(f"only {len(selected)} visible poses; requested {self.wanted}. Increase candidates or revise envelope.")
            assert placement_image is not None
            cv2.imwrite(str(self.output_dir / "placement_preview.png"), self._draw_crosshair(placement_image, selected[0].board_center_px))
            payload = {
                "schema_version": 1, "arm": self.arm, "units": "radians",
                "selection": {"wanted": self.wanted, "candidates_scanned": self.candidates, "visible_before_selection": len(visible), "method": "ChArUco visibility+PnP gate, farthest-point diversity in joint/image space"},
                "placement_pose": {"qpos_rad": selected[0].qpos_rad, "board_center_target_px": selected[0].board_center_px, "preview": "placement_preview.png", "operator_instruction": "At this pose, slowly position the printed board so its printed ChArUco geometric centre aligns with the red X. Do not start an automatic scan until it is stable."},
                "calibration_poses": [asdict(item) for item in selected],
                "hardware_gate": "Simulation candidates only. Before hardware use, execute every pose individually with low speed, collision supervision, and observed-board quality checks; reject or replace any pose that is unsafe or not visible.",
            }
            (self.output_dir / f"{self.arm}_wrist_pose_library.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            return payload
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wanted", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    args = parser.parse_args()
    result = RoboTwinCalibrationPoseScanner(args.output_dir, wanted=args.wanted, candidates=args.candidates, arm=args.arm).run()
    print(json.dumps({"pose_file": str(args.output_dir / f"{args.arm}_wrist_pose_library.json"), **result["selection"]}, indent=2))


if __name__ == "__main__":
    main()
