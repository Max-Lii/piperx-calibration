from __future__ import annotations

from pathlib import Path

import pytest

from piperx_calibration.robotcontrol.motion_smoke import MAX_PROBE_DELTA_RAD, load_config, plan_waypoints


def _config_text(*, delta: float = 0.02, first_index: int = 0) -> str:
    return f"""service_url: http://127.0.0.1:8008
owner_id: test
expected_home: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
home_tolerance_rad: 0.02
timeout_s: 20
probes:
  - name: left
    action_index: {first_index}
    delta_rad: {delta}
  - name: right
    action_index: 7
    delta_rad: 0.02
grippers:
  left:
    open_action_value: 1.0
    closed_action_value: 0.0
  right:
    open_action_value: 1.0
    closed_action_value: 0.0
"""


def test_plan_has_one_joint_probe_then_explicit_return_for_each_arm(tmp_path: Path):
    path = tmp_path / "motion.yaml"
    path.write_text(_config_text())
    config = load_config(path)
    plan = plan_waypoints(config)
    assert [name for name, _ in plan] == ["left", "left_return", "right", "right_return"]
    assert plan[0][1][0] == pytest.approx(0.02)
    assert plan[1][1] == [0.0] * 14
    assert plan[2][1][7] == pytest.approx(0.02)


def test_gripper_cycle_is_opt_in_and_never_moves_both_grippers_together(tmp_path: Path):
    path = tmp_path / "motion.yaml"
    path.write_text(_config_text())
    config = load_config(path)
    without_grippers = plan_waypoints(config)
    with_grippers = plan_waypoints(config, include_gripper_cycle=True)
    assert len(without_grippers) == 4
    assert [name for name, _ in with_grippers[4:]] == [
        "left_gripper_open", "left_gripper_close", "left_gripper_return",
        "right_gripper_open", "right_gripper_close", "right_gripper_return",
    ]
    assert with_grippers[4][1][6] == pytest.approx(1.0)
    assert with_grippers[4][1][13] == pytest.approx(0.0)
    assert with_grippers[7][1][13] == pytest.approx(1.0)
    assert with_grippers[7][1][6] == pytest.approx(0.0)


def test_rejects_unsafe_or_cross_arm_probe_configuration(tmp_path: Path):
    path = tmp_path / "motion.yaml"
    path.write_text(_config_text(delta=MAX_PROBE_DELTA_RAD + 0.001))
    with pytest.raises(ValueError, match="delta_rad"):
        load_config(path)
    path.write_text(_config_text(first_index=7))
    with pytest.raises(ValueError, match="left joints"):
        load_config(path)
