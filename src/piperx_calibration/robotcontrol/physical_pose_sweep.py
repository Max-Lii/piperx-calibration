"""Operator-launched, supervised physical calibration pose sweep.

This tool does not start a service or open CAN itself. It talks only to an
already-running robotcontrol instance, keeps the lease alive, moves one arm at
a time, holds each feedback-settled target for the configured duration, and
returns Home only after every library pose succeeds. Ctrl-C or any error issues HOLD
at the measured pose and releases the lease; it never disables the robot.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .client import RobotControlClient


def _target(pose: list[float], arm: str, gripper: float = 0.0) -> list[float]:
    if len(pose) != 6:
        raise ValueError("each library pose must contain six joint angles")
    zeros = [0.0] * 6
    # Action dimensions are left joints, left gripper, right joints, right
    # gripper. Zero means closed when a physical gripper mapping is enabled.
    return ([*pose, gripper, *zeros, 0.0] if arm == "left" else [*zeros, 0.0, *pose, gripper])


def run(
    *, url: str, library: Path, hold_s: float, timeout_s: float,
    poses_per_arm: int | None = None, cycle_gripper: bool = False,
) -> None:
    raw = json.loads(library.read_text(encoding="utf-8"))
    left, right = raw.get("left"), raw.get("right")
    if not isinstance(left, list) or not isinstance(right, list) or not left or len(left) != len(right):
        raise ValueError("pose library must contain equal, non-empty left and right pose lists")
    if hold_s < 0:
        raise ValueError("hold_s must be non-negative")
    count = len(left) if poses_per_arm is None else poses_per_arm
    if not 1 <= count <= len(left):
        raise ValueError("poses_per_arm must be within the pose library")
    left, right = left[:count], right[:count]

    client = RobotControlClient(url)
    completed = False
    try:
        total = len(left) + len(right)
        client.acquire("operator-supervised-calibration-pose-sweep")
        client.start_heartbeat(interval_s=3.0)
        client.arm_motion()
        for arm, poses, offset in (("left", left, 0), ("right", right, len(left))):
            for index, pose in enumerate(poses, start=1):
                ordinal = offset + index
                values = (0.0, 1.0) if cycle_gripper else (0.0,)
                for gripper in values:
                    state = "open" if gripper else "closed"
                    print(f"START {ordinal}/{total} {arm}_{index:02d} gripper={state}", flush=True)
                    execution = client.execute_joint_waypoints(
                        [_target([float(v) for v in pose], arm, gripper)], "calibration_slow",
                    )
                    result = client.wait_execution(execution["execution_id"], timeout_s=timeout_s)
                    if result["state"] != "completed":
                        raise RuntimeError(f"{arm}_{index:02d} gripper={state} ended as {result}")
                    feedback = client.state()["qpos"][6 if arm == "left" else 13]
                    print(f"REACHED {arm}_{index:02d} gripper={state} feedback={feedback:.4f}", flush=True)
                print(f"HOLD {ordinal}/{total} {arm}_{index:02d} {hold_s:.1f}s", flush=True)
                time.sleep(hold_s)
        completed = True
    finally:
        try:
            if completed:
                print("ALL_POSES_COMPLETE: returning Home", flush=True)
                home = client.go_home()
                result = client.wait_execution(home["execution_id"], timeout_s=timeout_s)
                if result["state"] != "completed":
                    raise RuntimeError(f"Go Home ended as {result}")
                print("SWEEP_PASS: Home feedback settled", flush=True)
            else:
                client.hold()
                print("SWEEP_STOPPED: HOLD at measured pose; Home was not commanded", flush=True)
        finally:
            try:
                client.stop_heartbeat()
                client.release()
            finally:
                client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotcontrol-url", default="http://127.0.0.1:8008")
    parser.add_argument("--pose-library", type=Path, required=True)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=75.0)
    parser.add_argument("--poses-per-arm", type=int)
    parser.add_argument("--cycle-gripper", action="store_true")
    args = parser.parse_args()
    run(
        url=args.robotcontrol_url, library=args.pose_library, hold_s=args.hold_s,
        timeout_s=args.timeout_s, poses_per_arm=args.poses_per_arm,
        cycle_gripper=args.cycle_gripper,
    )


if __name__ == "__main__":
    main()
