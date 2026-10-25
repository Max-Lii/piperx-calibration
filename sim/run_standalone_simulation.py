"""Run the self-contained calibration simulation regression.

This backend deliberately does not import RoboTwin, SAPIEN, the parent VLA
tree, or a GPU runtime. It renders a metric ChArUco target from hidden workcell
truth, generates a deterministic dual-arm pose library, applies the same
formal calibration solver used by the real capture path, and writes a compact
regression artifact. It is the default simulation gate for the standalone
project; the optional RoboTwin scripts are development references only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from piperx_calibration.calibration.geometry import solve_robot_world_hand_eye
from sim.virtual_calibration import RenderedCharucoRobotWorldHarness


def make_pose_library(count: int) -> dict:
    if count < 6:
        raise ValueError("pose library needs at least six poses per arm")
    poses = []
    for index in range(count):
        phase = index / max(count - 1, 1)
        poses.append({
            "qpos_rad": [
                float(0.32 * np.sin(phase * 2.0 * np.pi)),
                float(0.18 + 0.14 * np.cos(phase * 3.0 * np.pi)),
                float(-0.22 + 0.10 * np.sin(phase * 4.0 * np.pi)),
                float(0.16 * np.sin(phase * 5.0 * np.pi)),
                float(0.12 * np.cos(phase * 2.0 * np.pi)),
                float(0.18 * np.sin(phase * 3.0 * np.pi)),
            ],
            "simulation_only": True,
        })
    return {
        "schema_version": 1,
        "backend": "standalone_virtual",
        "target_id": "synthetic_metric_charuco",
        "pose_selection": "deterministic coverage sweep",
        "left": poses,
        "right": poses,
    }


def run(output_dir: Path, observations: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_library = make_pose_library(observations)
    arms = {}
    for arm, seed in (("left", 7), ("right", 19)):
        result = RenderedCharucoRobotWorldHarness(seed=seed, output_dir=output_dir / arm).run(
            observations=observations,
        )
        arms[arm] = asdict(result)
    report = {
        "schema_version": 1,
        "backend": "standalone_virtual",
        "target": {
            "target_id": "synthetic_metric_charuco",
            "coordinate_convention": "T_parent_child maps child coordinates into parent coordinates",
            "length_unit": "m",
        },
        "pose_library": {
            "poses_per_arm": observations,
            "selection": pose_library["pose_selection"],
        },
        "arms": arms,
        "passed": all(item["passed"] for item in arms.values()),
        "external_runtime_required": False,
    }
    (output_dir / "pose_library.json").write_text(json.dumps(pose_library, indent=2) + "\n", encoding="utf-8")
    (output_dir / "simulation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observations", type=int, default=30)
    args = parser.parse_args()
    report = run(args.output_dir, args.observations)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("standalone simulation regression failed")


if __name__ == "__main__":
    main()
