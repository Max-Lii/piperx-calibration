from __future__ import annotations

import json

from sim.run_motion_sampling_pipeline import run


def test_motion_sampling_pipeline_selects_and_solves(tmp_path):
    report = run(candidates=24, wanted=12, output_dir=tmp_path)
    assert report["passed"]
    assert report["pose_library"]["candidates_visible"] >= 12
    assert report["pose_library"]["poses_selected"] == 12
    library = json.loads((tmp_path / "pose_library.json").read_text())
    assert len(library["calibration_poses"]) == 12
    assert all(item["charuco_corners"] >= 8 for item in library["calibration_poses"])
