"""Rebuild the simulator pose library through the robotcontrol protocol."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from piperx_calibration.robotcontrol.client import RobotControlClient

from .quality import CharucoTargetSpec, evaluate_charuco


def _candidates(arm: str) -> list[list[float]]:
    anchor = np.array([-.45, .25, -.55, 1., 0., 0.])
    if arm == "right":
        anchor[0] *= -1
    scale = np.array([.20, .22, .24, .42, .42, .55])
    seeds = (43, 23) if arm == "left" else (47, 29)
    values: list[list[float]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        values.extend([anchor.tolist(), *((anchor + rng.uniform(-1, 1, 6) * scale).tolist() for _ in range(95))])
    unique: dict[tuple[float, ...], list[float]] = {}
    for value in values:
        unique.setdefault(tuple(value), value)
    return list(unique.values())


def _wait_stable(client: RobotControlClient, target: list[float], timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    stable_since = None
    while time.monotonic() < deadline:
        state = client.state()
        q = state["qpos"]
        dq = state["dq_rad_s"]
        indices = tuple(i for i in range(14) if i not in (6, 13))
        stationary = (
            max(abs(q[i] - target[i]) for i in indices) <= 0.015
            and max(abs(dq[i]) for i in indices) <= 0.03
        )
        now = time.monotonic()
        stable_since = stable_since if stationary and stable_since is not None else (now if stationary else None)
        if stable_since is not None and now - stable_since >= 0.05:
            return state
        time.sleep(0.02)
    raise TimeoutError("simulator did not reach the candidate pose")


def _select(records: list[dict], wanted: int) -> list[list[float]]:
    eligible = [item for item in records if item.get("accepted")]
    if len(eligible) < wanted:
        raise RuntimeError(f"only {len(eligible)} candidates passed, need {wanted}")
    q = np.asarray([item["qpos_rad"] for item in eligible], dtype=np.float64)
    scale = np.maximum(np.ptp(q, axis=0), 1e-6)
    features = (q - np.mean(q, axis=0)) / scale
    quality = np.asarray([min(item["corner_count"], 24) / 24.0 for item in eligible])
    chosen = [int(np.argmax(quality))]
    minimum = np.linalg.norm(features - features[chosen[0]], axis=1)
    while len(chosen) < wanted:
        score = minimum * (0.65 + 0.35 * quality)
        score[chosen] = -1
        index = int(np.argmax(score))
        chosen.append(index)
        minimum = np.minimum(minimum, np.linalg.norm(features - features[index], axis=1))
    return [[float(value) for value in eligible[index]["qpos_rad"]] for index in chosen]


def run(args: argparse.Namespace) -> dict:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scan_path = args.output.with_name(args.output.stem + "_scan.json")
    scan = json.loads(scan_path.read_text(encoding="utf-8")) if scan_path.is_file() else {"left": [], "right": []}
    target = CharucoTargetSpec.from_manifest(args.target_manifest)
    client = RobotControlClient(args.robotcontrol_url)
    park = [0.0, 0.05, -0.05, 0.0, 0.0, 0.0]
    try:
        if client.state().get("backend") != "robotwin":
            raise RuntimeError("pose scan is allowed only against backend=robotwin")
        client.acquire("robotwin-pose-library-scan")
        client.start_heartbeat()
        for arm, camera in (("left", "left_wrist"), ("right", "right_wrist")):
            candidates = _candidates(arm)
            done = {tuple(item["qpos_rad"]) for item in scan[arm]}
            for index, candidate in enumerate(candidates, start=1):
                if tuple(candidate) in done:
                    continue
                left = candidate if arm == "left" else park
                right = candidate if arm == "right" else park
                waypoint = left + [1.0] + right + [1.0]
                try:
                    execution = client.execute_joint_waypoints([waypoint], "calibration_slow")
                    status = client.wait_execution(execution["execution_id"], timeout_s=30.0)
                    if status["state"] != "completed":
                        raise RuntimeError(f"candidate motion ended as {status['state']}")
                    _wait_stable(client, waypoint)
                    frame, _ = client.calibration_frame(camera)
                    metrics = evaluate_charuco(frame, target)
                    record = {
                        "qpos_rad": candidate,
                        "accepted": bool(metrics["accepted"]),
                        "corner_count": int(metrics["corner_count"]),
                        "hull_area_fraction": float(metrics["hull_area_fraction"]),
                        "minimum_border_px": float(metrics["minimum_border_px"]),
                        "reprojection_error_px": metrics["reprojection_error_px"],
                        "reasons": metrics["reasons"],
                    }
                except Exception as exc:
                    client.hold()
                    record = {
                        "qpos_rad": candidate, "accepted": False, "corner_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                scan[arm].append(record)
                scan_path.write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
                passed = sum(item.get("accepted", False) for item in scan[arm])
                print(f"[{arm}] {index:03d}/{len(candidates)} accepted={passed}", flush=True)
        library = {
            "schema_version": 3,
            "target": {
                "target_id": "BOARD_001", "square_length_mm": 35.0, "marker_length_mm": 26.25,
            },
            "source": "robotcontrol HTTP feedback-stable RoboTwin scan",
            "simulation_only_not_motion_approved": True,
            "left": _select(scan["left"], args.wanted),
            "right": _select(scan["right"], args.wanted),
        }
        args.output.write_text(json.dumps(library, indent=2) + "\n", encoding="utf-8")
        return {
            "output": str(args.output),
            "tested": {arm: len(scan[arm]) for arm in ("left", "right")},
            "accepted": {arm: sum(item.get("accepted", False) for item in scan[arm]) for arm in ("left", "right")},
            "selected": args.wanted,
        }
    finally:
        try:
            client.hold()
            client.release()
        except Exception:
            pass
        client.close()


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotcontrol-url", default="http://127.0.0.1:8009")
    parser.add_argument(
        "--target-manifest", type=Path,
        default=root / "assets/targets/BOARD_001_A4_Charuco_5x7_35mm.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wanted", type=int, default=48)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
