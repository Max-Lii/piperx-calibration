"""Select twelve additional high-view BOARD_002 poses per Piper arm.

The legacy 36-pose library remains the coverage baseline.  This tool renders
candidate poses in RoboTwin against the same flat BOARD_002 page, then appends
only high wrist-camera views which pass visibility, border and PnP gates.
The output is simulation-only: physical collision clearance is a separate,
supervised robotcontrol acceptance step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sim.generate_a4_charuco_pdf import MARKER_MM, SQUARE_MM, SQUARES
from sim.robotwin.robotwin_dual_tabletop import BOARD_CENTER_W, make_printable_target_mesh
from sim.robotwin.robotwin_rendered import RoboTwinRenderedHandEyeHarness
from sim.virtual_calibration import _inverse


def _load_candidates(paths: list[Path], arm: str) -> list[list[float]]:
    unique: dict[tuple[float, ...], list[float]] = {}
    for path in paths:
        for item in json.loads(path.read_text(encoding="utf-8"))[arm]:
            qpos = [float(value) for value in item["qpos_rad"]]
            unique.setdefault(tuple(qpos), qpos)
    return list(unique.values())


def _metrics(task, harness: RoboTwinRenderedHandEyeHarness, arm: str, qpos: list[float]) -> dict:
    for _ in range(120):
        task.robot.set_arm_joints(np.asarray(qpos), np.full(6, 0.18), arm)
        task.scene.step()
    task._update_render()
    image = np.asarray(task.get_obs()["observation"][f"{arm}_camera"]["rgb"])
    corners, ids, _, _ = harness.detector.detectBoard(image)
    if corners is None or ids is None or len(ids) < 8:
        return {"qpos_rad": qpos, "accepted": False, "corner_count": 0}
    camera = task.cameras.get_config()[f"{arm}_camera"]
    intrinsic = np.asarray(camera["intrinsic_cv"], dtype=np.float64)
    object_points = harness.board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {"qpos_rad": qpos, "accepted": False, "corner_count": int(len(ids)), "reason": "pnp_failed"}
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsic, None)
    points = corners.reshape(-1, 2)
    hull = cv2.convexHull(points.astype(np.float32))
    height, width = image.shape[:2]
    camera_world = _inverse(np.vstack((np.asarray(camera["extrinsic_cv"], dtype=np.float64), [0, 0, 0, 1])))
    reprojection = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - points, axis=1)))
    min_border = float(np.min(np.concatenate((points[:, 0], width - 1 - points[:, 0], points[:, 1], height - 1 - points[:, 1]))))
    return {
        "qpos_rad": qpos,
        "accepted": bool(reprojection <= 1.5 and min_border >= 10.0),
        "corner_count": int(len(ids)),
        "full_board": bool(len(ids) == (SQUARES[0] - 1) * (SQUARES[1] - 1)),
        "reprojection_error_px": reprojection,
        "minimum_border_px": min_border,
        "hull_area_fraction": float(cv2.contourArea(hull) / (width * height)),
        "board_center_px": np.mean(points, axis=0).tolist(),
        "target_distance_m": float(np.linalg.norm(tvec)),
        "camera_height_world_m": float(camera_world[2, 3]),
    }


def _choose_extensions(records: list[dict], base: list[list[float]], wanted: int) -> list[dict]:
    base_keys = {tuple(item) for item in base}
    eligible = [item for item in records if item.get("accepted") and tuple(item["qpos_rad"]) not in base_keys]
    if len(eligible) < wanted:
        raise RuntimeError(f"only {len(eligible)} new quality candidates, need {wanted}")
    complete = [item for item in eligible if item["full_board"]]
    if len(complete) < wanted:
        raise RuntimeError(f"only {len(complete)} new full-board candidates, need {wanted}")
    # Define the extension set by rank, not a fixed quantile. A narrow safe
    # envelope can legitimately have fewer than twelve candidates above its
    # 67th percentile; the top twelve quality-approved camera heights are
    # still the highest available views and meet the requested high-view role.
    high = sorted(complete, key=lambda item: item["camera_height_world_m"], reverse=True)[:max(wanted * 2, wanted)]
    # Prefer full-board views, then image footprint, then elevation. Greedy
    # joint-space spacing prevents twelve near-duplicate high poses.
    q = np.asarray([item["qpos_rad"] for item in high], dtype=np.float64)
    scale = np.maximum(np.ptp(q, axis=0), 1e-6)
    features = (q - np.mean(q, axis=0)) / scale
    quality = np.asarray([
        2.0 * float(item["full_board"]) + item["hull_area_fraction"] + 0.1 * item["camera_height_world_m"]
        for item in high
    ])
    chosen = [int(np.argmax(quality))]
    distance = np.linalg.norm(features - features[chosen[0]], axis=1)
    while len(chosen) < wanted:
        score = distance * (0.5 + quality)
        score[chosen] = -1.0
        index = int(np.argmax(score))
        chosen.append(index)
        distance = np.minimum(distance, np.linalg.norm(features - features[index], axis=1))
    return [high[index] for index in chosen]


def run(args: argparse.Namespace) -> dict:
    legacy = json.loads(args.base_library.read_text(encoding="utf-8"))
    result: dict = {"schema_version": 1, "target": {**legacy["target"], "target_id": "BOARD_002"}, "simulation_only_not_motion_approved": True}
    arms = (args.arm,) if args.arm else ("left", "right")
    harness = RoboTwinRenderedHandEyeHarness(output_dir=None)
    harness.square_length_m, harness.marker_length_m = SQUARE_MM / 1000.0, MARKER_MM / 1000.0
    harness.board = cv2.aruco.CharucoBoard(SQUARES, harness.square_length_m, harness.marker_length_m, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    harness.detector = cv2.aruco.CharucoDetector(harness.board)
    env = harness._make_env()
    try:
        env.reset(init_id=0)
        task = env._task
        assert task is not None
        if hasattr(task, "bell"):
            task.scene.remove_actor(task.bell.actor)
        import sapien
        assets = args.output.parent / f"{args.output.stem}_assets"
        mesh, _ = make_printable_target_mesh(assets)
        builder = task.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_visual_from_file(filename=str(mesh))
        board = builder.build(name="board002_flat_a4_charuco")
        board.set_pose(sapien.Pose(BOARD_CENTER_W, [1, 0, 0, 0]))
        for arm, paths in (("left", args.left_candidates), ("right", args.right_candidates)):
            if arm not in arms:
                result[arm] = legacy[arm]
                continue
            records = [_metrics(task, harness, arm, qpos) for qpos in _load_candidates(paths, arm)]
            # BOARD_002 changes the rendered page. Revalidate legacy poses
            # rather than assuming every BOARD_001-approved waypoint remains
            # acceptable; replace only failures with full-board candidates.
            if args.assume_base_accepted:
                retained = legacy[arm]
                replacements = 0
                base_records = []
            else:
                base_records = [_metrics(task, harness, arm, qpos) for qpos in legacy[arm]]
                retained = [item["qpos_rad"] for item in base_records if item.get("accepted")]
                replacements = len(legacy[arm]) - len(retained)
            extension = _choose_extensions(records, retained, args.additional + replacements)
            result[arm] = retained + [item["qpos_rad"] for item in extension]
            result[f"{arm}_extensions"] = extension
            result[f"{arm}_scan_summary"] = {
                "candidates": len(records), "accepted": sum(bool(item.get("accepted")) for item in records),
                "full_board": sum(bool(item.get("full_board")) for item in records),
                "legacy_total": len(legacy[arm]), "legacy_retained": len(retained), "legacy_replaced": replacements,
            }
    finally:
        env.close()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-library", type=Path, required=True)
    parser.add_argument("--left-candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--right-candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional", type=int, default=12)
    parser.add_argument("--arm", choices=("left", "right"))
    parser.add_argument("--assume-base-accepted", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
