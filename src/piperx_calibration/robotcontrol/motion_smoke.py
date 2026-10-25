"""Supervised, opt-in first-motion acceptance harness for a dual Piper workcell.

This module is deliberately a client of robotcontrol rather than a service
feature.  Its default mode is a dry run.  The host-local configuration names a
known-clear Home pose and two tiny, single-joint probes.  A fault always sends
HOLD and never attempts an automatic Go Home recovery.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .client import RobotControlClient
from .schemas import ACTION_DIM


MAX_PROBE_DELTA_RAD = 0.03
MAX_STATIONARY_DQ_RAD_S = 0.02


@dataclass(frozen=True)
class Probe:
    name: str
    action_index: int
    delta_rad: float


@dataclass(frozen=True)
class GripperCycle:
    name: str
    action_index: int
    open_action_value: float
    closed_action_value: float


@dataclass(frozen=True)
class MotionSmokeConfig:
    service_url: str
    owner_id: str
    expected_home: list[float]
    home_tolerance_rad: float
    probes: tuple[Probe, Probe]
    grippers: tuple[GripperCycle, GripperCycle]
    timeout_s: float


def _numbers(value: Any, *, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numbers") from exc
    if not all(abs(item) < float("inf") for item in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def load_config(path: Path) -> MotionSmokeConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("motion smoke config must be a mapping")
    home = _numbers(raw.get("expected_home"), name="expected_home", length=ACTION_DIM)
    raw_probes = raw.get("probes")
    if not isinstance(raw_probes, list) or len(raw_probes) != 2:
        raise ValueError("probes must contain exactly the left and right probes")
    probes = []
    for raw_probe in raw_probes:
        if not isinstance(raw_probe, dict):
            raise ValueError("each probe must be a mapping")
        name = str(raw_probe.get("name", ""))
        index = raw_probe.get("action_index")
        delta = raw_probe.get("delta_rad")
        if not name or not isinstance(index, int) or not 0 <= index < ACTION_DIM:
            raise ValueError("each probe needs a name and action_index in [0, 13]")
        try:
            delta = float(delta)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe delta_rad must be numeric") from exc
        if not 0 < abs(delta) <= MAX_PROBE_DELTA_RAD:
            raise ValueError(f"probe delta_rad must be within ±{MAX_PROBE_DELTA_RAD} rad and non-zero")
        probes.append(Probe(name=name, action_index=index, delta_rad=delta))
    if not (0 <= probes[0].action_index <= 5 and 7 <= probes[1].action_index <= 12):
        raise ValueError("first probe must target left joints [0,5]; second right joints [7,12]")
    raw_grippers = raw.get("grippers")
    if not isinstance(raw_grippers, dict):
        raise ValueError("grippers must define left and right action values")
    grippers = []
    for name, index in (("left", 6), ("right", 13)):
        raw_gripper = raw_grippers.get(name)
        if not isinstance(raw_gripper, dict):
            raise ValueError(f"grippers.{name} must be a mapping")
        try:
            open_value = float(raw_gripper["open_action_value"])
            closed_value = float(raw_gripper["closed_action_value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"grippers.{name} needs numeric open_action_value and closed_action_value") from exc
        if not (abs(open_value) < float("inf") and abs(closed_value) < float("inf")):
            raise ValueError(f"grippers.{name} values must be finite")
        if open_value == closed_value:
            raise ValueError(f"grippers.{name} open and closed values must differ")
        grippers.append(GripperCycle(name, index, open_value, closed_value))
    if not 0 < float(raw.get("home_tolerance_rad", 0.0)) <= 0.10:
        raise ValueError("home_tolerance_rad must be in (0, 0.10]")
    if not 1.0 <= float(raw.get("timeout_s", 0.0)) <= 60.0:
        raise ValueError("timeout_s must be in [1, 60]")
    return MotionSmokeConfig(
        service_url=str(raw.get("service_url", "http://127.0.0.1:8008")),
        owner_id=str(raw.get("owner_id", "supervised-first-motion")),
        expected_home=home,
        home_tolerance_rad=float(raw["home_tolerance_rad"]),
        probes=(probes[0], probes[1]),
        grippers=(grippers[0], grippers[1]),
        timeout_s=float(raw["timeout_s"]),
    )


def plan_waypoints(config: MotionSmokeConfig, *, include_gripper_cycle: bool = False) -> list[tuple[str, list[float]]]:
    """Return joint probes/returns plus an optional one-at-a-time gripper cycle."""
    plan: list[tuple[str, list[float]]] = []
    for probe in config.probes:
        target = list(config.expected_home)
        target[probe.action_index] += probe.delta_rad
        plan.extend(((probe.name, target), (f"{probe.name}_return", list(config.expected_home))))
    if include_gripper_cycle:
        for gripper in config.grippers:
            opened = list(config.expected_home)
            opened[gripper.action_index] = gripper.open_action_value
            closed = list(config.expected_home)
            closed[gripper.action_index] = gripper.closed_action_value
            plan.extend(((f"{gripper.name}_gripper_open", opened), (f"{gripper.name}_gripper_close", closed)))
            plan.append((f"{gripper.name}_gripper_return", list(config.expected_home)))
    return plan


def _require_ready_state(state: dict[str, Any], config: MotionSmokeConfig) -> None:
    qpos = _numbers(state.get("qpos"), name="state.qpos", length=ACTION_DIM)
    dq = _numbers(state.get("dq_rad_s"), name="state.dq_rad_s", length=ACTION_DIM)
    if state.get("lease_active"):
        raise RuntimeError("another controller already holds the robot lease")
    if state.get("holding") is not True:
        raise RuntimeError("robotcontrol is not in HOLD; inspect the workcell before testing")
    if max(abs(value) for value in dq) > MAX_STATIONARY_DQ_RAD_S:
        raise RuntimeError("robot is not stationary")
    error = max(abs(actual - expected) for actual, expected in zip(qpos, config.expected_home, strict=True))
    if error > config.home_tolerance_rad:
        raise RuntimeError(
            f"robot is not at the manually verified Home pose (max joint error {error:.4f} rad)"
        )


def _completed(client: RobotControlClient, execution_id: str, timeout_s: float) -> None:
    result = client.wait_execution(execution_id, timeout_s=timeout_s)
    if result["state"] != "completed":
        raise RuntimeError(f"execution {execution_id} ended as {result['state']}: {result.get('reason')}")


def execute(config: MotionSmokeConfig, *, include_gripper_cycle: bool) -> None:
    client = RobotControlClient(config.service_url)
    acquired = False
    try:
        _require_ready_state(client.state(), config)
        client.acquire(config.owner_id)
        acquired = True
        client.start_heartbeat()
        for name, target in plan_waypoints(config, include_gripper_cycle=include_gripper_cycle):
            print(f"starting supervised probe: {name}", flush=True)
            started = client.execute_joint_waypoints([target], profile="calibration_slow")
            _completed(client, started["execution_id"], config.timeout_s)
        print("starting explicit Go Home", flush=True)
        started = client.go_home()
        _completed(client, started["execution_id"], config.timeout_s)
        _require_ready_state(client.state(), config)
        print("MOTION_SMOKE_PASS", flush=True)
    except BaseException:
        # HOLD is the only automatic recovery: moving toward Home after an
        # unexpected fault would be an unsafe assumption about clearance.
        try:
            client.hold()
        except Exception:
            pass
        raise
    finally:
        if acquired:
            try:
                client.release()
            except Exception:
                pass
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised first-motion acceptance; dry-run by default")
    parser.add_argument("--config", type=Path, required=True, help="host-local motion smoke YAML")
    parser.add_argument("--execute", action="store_true", help="send motion only with the confirmation flag")
    parser.add_argument("--confirm-physical-clearance", action="store_true")
    parser.add_argument("--include-gripper-cycle", action="store_true", help="also open then close each empty gripper one at a time")
    args = parser.parse_args()
    config = load_config(args.config)
    plan = [
        {"name": name, "target": target}
        for name, target in plan_waypoints(config, include_gripper_cycle=args.include_gripper_cycle)
    ]
    print(json.dumps({"service_url": config.service_url, "plan": plan}, indent=2))
    if not args.execute:
        print("DRY_RUN: no robot request was made")
        return
    if not args.confirm_physical_clearance:
        raise SystemExit("refusing motion: pass --confirm-physical-clearance only after an operator verifies clearance")
    execute(config, include_gripper_cycle=args.include_gripper_cycle)


if __name__ == "__main__":
    main()
