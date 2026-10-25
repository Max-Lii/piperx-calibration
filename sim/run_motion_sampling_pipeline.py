"""Run the standalone motion-sampling and calibration regression.

This is the dependency-free simulation contract used by the project.  It
models the part RoboTwin supplies to calibration: a bounded joint-pose pool,
rendered ChArUco observations, and robot feedback for each accepted pose.  It
then selects a diverse library and feeds exactly that library to the formal
robot-world/hand-eye solver.

The real robotcontrol service remains the production backend.  This script is
only an offline acceptance test and never opens CAN, cameras, or RoboTwin.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sim.virtual_calibration import RenderedCharucoRobotWorldHarness


@dataclass(frozen=True)
class Candidate:
    index: int
    qpos_rad: list[float]
    base_ee: np.ndarray
    corners: int
    centre_px: tuple[float, float]
    reprojection_error_px: float

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "qpos_rad": self.qpos_rad,
            "charuco_corners": self.corners,
            "board_center_px": list(self.centre_px),
            "reprojection_error_px": self.reprojection_error_px,
        }


class _SelectedHarness(RenderedCharucoRobotWorldHarness):
    def __init__(self, poses: list[np.ndarray], **kwargs) -> None:
        super().__init__(**kwargs)
        self._poses = poses

    def _sample_fk_pose(self, index: int, count: int) -> np.ndarray:
        del count
        return self._poses[index]


def _candidate_qpos(index: int, count: int) -> list[float]:
    # A deterministic six-joint command envelope.  These values are metadata
    # for the motion library; the virtual renderer uses the resulting FK pose.
    phase = 2.0 * np.pi * index / max(count, 1)
    return [
        float(0.28 * np.sin(phase)),
        float(0.22 * np.cos(phase * 1.3)),
        float(-0.24 + 0.06 * np.sin(phase * 0.7)),
        float(0.18 * np.sin(phase * 1.7)),
        float(-0.14 * np.cos(phase * 1.1)),
        float(0.16 * np.sin(phase * 0.9)),
    ]


def _visible_candidates(harness: RenderedCharucoRobotWorldHarness, count: int) -> list[Candidate]:
    visible: list[Candidate] = []
    for index in range(count):
        base_ee = harness._sample_fk_pose(index, count)
        world_camera = harness._truth_world_base @ base_ee @ harness._truth_ee_camera
        image = harness._render(np.linalg.inv(world_camera), index=10_000 + index)
        corners, ids, _, _ = harness.detector.detectBoard(image)
        if corners is None or ids is None or len(ids) < 8:
            continue
        object_points = harness.board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
        ok, rvec, translation = cv2.solvePnP(
            object_points, corners, harness.intrinsic, harness.distortion, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, translation, harness.intrinsic, harness.distortion)
        error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))
        centre = np.mean(corners.reshape(-1, 2), axis=0)
        if error > 1.5:
            continue
        visible.append(Candidate(
            index=index,
            qpos_rad=_candidate_qpos(index, count),
            base_ee=base_ee,
            corners=int(len(ids)),
            centre_px=(float(centre[0]), float(centre[1])),
            reprojection_error_px=error,
        ))
    return visible


def _select_diverse(candidates: list[Candidate], wanted: int) -> list[Candidate]:
    if len(candidates) < wanted:
        raise RuntimeError(f"only {len(candidates)} visible candidates; need {wanted}")
    features = np.asarray([
        [*candidate.qpos_rad, candidate.centre_px[0] / 640.0, candidate.centre_px[1] / 480.0]
        for candidate in candidates
    ], dtype=np.float64)
    features /= np.maximum(np.linalg.norm(features, axis=0, keepdims=True), 1e-9)
    chosen = [int(np.argmin([candidate.reprojection_error_px for candidate in candidates]))]
    distances = np.linalg.norm(features - features[chosen[0]], axis=1)
    while len(chosen) < wanted:
        distances[chosen] = -np.inf
        chosen.append(int(np.argmax(distances)))
        distances = np.minimum(distances, np.linalg.norm(features - features[chosen[-1]], axis=1))
    return [candidates[index] for index in chosen]


def run(*, candidates: int, wanted: int, output_dir: Path) -> dict:
    if candidates < wanted or wanted < 6:
        raise ValueError("wanted must be >= 6 and candidates must be >= wanted")
    harness = RenderedCharucoRobotWorldHarness(output_dir=None)
    visible = _visible_candidates(harness, candidates)
    selected = _select_diverse(visible, wanted)
    selected_harness = _SelectedHarness([item.base_ee for item in selected], output_dir=output_dir / "renders")
    calibration = selected_harness.run(observations=len(selected)).as_dict()
    pose_library = {
        "schema_version": 1,
        "backend": "standalone_virtual",
        "selection": {
            "candidates_requested": candidates,
            "candidates_visible": len(visible),
            "poses_selected": len(selected),
            "method": "farthest_point_joint_and_image_center",
        },
        "calibration_poses": [item.as_dict() for item in selected],
    }
    report = {
        "schema_version": 1,
        "pipeline": "motion_sampling_then_geometry_calibration",
        "pose_library": pose_library["selection"],
        "calibration": calibration,
        "passed": bool(calibration["passed"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pose_library.json").write_text(json.dumps(pose_library, indent=2), encoding="utf-8")
    (output_dir / "simulation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--wanted", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(candidates=args.candidates, wanted=args.wanted, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("motion sampling pipeline did not meet calibration thresholds")


if __name__ == "__main__":
    main()
