"""Protocol-level workcell geometry calibration through robotcontrol HTTP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx
import numpy as np

from piperx_calibration.robotcontrol.camera import decode_calibration_frame
from piperx_calibration.robotcontrol.client import HeartbeatError, RobotControlClient

from .capture import CalibrationCaptureSession, CapturePolicy
from .geometry import (
    HandEyeObservation,
    estimate_camera_target,
    hand_eye_validation,
    solve_robot_world_hand_eye,
    transform_error,
)
from .quality import CharucoTargetSpec, FrameQualityPolicy


def _matrix(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError("robot state contains an invalid 4x4 kinematic transform")
    return result


def _camera_pitch_deg(world_camera: np.ndarray) -> float:
    forward = world_camera[:3, 2]
    return float(np.degrees(np.arctan2(-forward[2], np.linalg.norm(forward[:2]))))


def _frame(output: Path, record: dict, camera_id: str):
    entry = record["frames"][camera_id][0]
    return decode_calibration_frame((output / entry["file"]).read_bytes()), entry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _average_transforms(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("cannot average an empty transform sequence")
    rotation_sum = np.sum([value[:3, :3] for value in values], axis=0)
    u, _, vh = np.linalg.svd(rotation_sum)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.median([value[:3, 3] for value in values], axis=0)
    return result


def _solve_third_camera(output: Path, records: list[dict], target: CharucoTargetSpec) -> tuple[np.ndarray, dict, dict]:
    candidates = []
    failures = []
    for record in records:
        for entry in record["frames"].get("third_view", []):
            frame_quality = entry.get("quality", {})
            target_quality = entry.get("target_quality") or {}
            # The third-view pose is solved from the planar ChArUco target.
            # D435 depth can be sparse on a reflective board or at an oblique
            # angle, but that must not discard a sharp, fully detected target.
            # Keep all other frame-quality failures blocking.
            frame_reasons = set(frame_quality.get("reasons", []))
            depth_only_rejection = frame_reasons == {"insufficient_depth"}
            if not (frame_quality.get("accepted", False) or depth_only_rejection):
                continue
            if entry.get("target_quality") is not None and not target_quality.get("accepted", False):
                continue
            try:
                frame = decode_calibration_frame((output / entry["file"]).read_bytes())
                camera_target, metrics = estimate_camera_target(frame, target)
                candidates.append((metrics["reprojection_error_px"], camera_target, metrics, {
                    "pose": record["name"], "file": entry["file"], "frame_number": entry["frame_number"],
                }))
            except Exception as exc:
                failures.append(f"{record['name']}: {type(exc).__name__}: {exc}")
    if not candidates:
        suffix = f"; failures={failures[-3:]}" if failures else ""
        raise RuntimeError(f"no accepted third-view frame contains a solvable target{suffix}")
    _, camera_target, metrics, source = min(candidates, key=lambda item: item[0])
    source["candidate_count"] = len(candidates)
    return camera_target, metrics, source


def _merge_gripper_records(name: str, records: list[dict]) -> dict:
    def gripper_state(record: dict) -> str:
        state = record.get("metadata", {}).get("gripper_state")
        if state in {"closed", "open"}:
            return state
        for candidate in ("closed", "open"):
            if record["name"].endswith(f"_{candidate}"):
                return candidate
        raise RuntimeError(f"capture record {record['name']} is missing gripper state")

    merged = dict(records[-1])
    merged["name"] = name
    merged["gripper_captures"] = [
        {"state": gripper_state(record), "manifest_name": record["name"]}
        for record in records
    ]
    merged["frames"] = {
        camera_id: [
            {**frame, "gripper_state": gripper_state(record)}
            for record in records for frame in record["frames"][camera_id]
        ]
        for camera_id in ("left_wrist", "right_wrist", "third_view")
    }
    return merged


def _accepted_frame_count(record: dict, camera_id: str) -> int:
    return sum(
        entry.get("quality", {}).get("accepted", False)
        and (
            entry.get("target_quality") is None
            or entry["target_quality"].get("accepted", False)
        )
        for entry in record["frames"].get(camera_id, [])
    )


def _lease_was_lost(exc: Exception) -> bool:
    return isinstance(exc, HeartbeatError) or (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code == 403
    )


def _solve_arm(output: Path, records: list[dict], camera_id: str, arm: str, target: CharucoTargetSpec) -> dict:
    observations: list[HandEyeObservation] = []
    reprojection = []
    corners = []
    depth_consistency = []
    solve_rejected = []
    for record in records:
        pose_transforms = []
        pose_entries = []
        for entry in record["frames"][camera_id]:
            if not (entry.get("quality") or {}).get("accepted", False):
                continue
            try:
                frame = decode_calibration_frame((output / entry["file"]).read_bytes())
                camera_target, metrics = estimate_camera_target(frame, target)
            except (OSError, ValueError, RuntimeError) as exc:
                # The capture quality gate is intentionally permissive enough
                # to retain a pose, while the final detector may still fail
                # on one burst frame. Skip that frame and solve from the
                # remaining poses instead of invalidating all 96 captures.
                solve_rejected.append({
                    "pose": record["name"],
                    "file": entry.get("file"),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            pose_transforms.append(camera_target)
            pose_entries.append(entry)
            reprojection.append(metrics["reprojection_error_px"])
            corners.append(metrics["corner_count"])
            if metrics["depth_consistency_mm"] is not None:
                depth_consistency.append(metrics["depth_consistency_mm"])
        if not pose_transforms:
            solve_rejected.append({"pose": record["name"], "error": f"no solvable {camera_id} frame"})
            continue
        state = pose_entries[-1]["robot_state_after"]
        kinematics = state.get("kinematics") or {}
        observations.append(HandEyeObservation(
            camera_target=_average_transforms(pose_transforms),
            base_ee=_matrix(kinematics[f"{arm}_T_base_ee"]),
        ))
    if len(observations) < 6:
        raise RuntimeError(
            f"{camera_id} has only {len(observations)} solvable poses after frame rejection; "
            f"need at least 6; rejected={solve_rejected[-5:]}"
        )
    solved = solve_robot_world_hand_eye(observations)
    validation = (
        hand_eye_validation(observations)
        if len(observations) >= 12
        else {
            "validation_skipped": True,
            "reason": "less_than_12_solvable_poses",
            "held_out_translation_median_mm": None,
            "held_out_translation_max_mm": None,
            "held_out_rotation_median_deg": None,
            "held_out_rotation_max_deg": None,
            "solver_agreement": {"base_translation_mm": None, "handeye_translation_mm": None},
            "bootstrap": {"base_translation_p95_mm": None, "handeye_translation_p95_mm": None},
        }
    )
    return {
        "accepted_poses": len(observations),
        "corner_count_min": min(corners),
        "reprojection_error_px_median": float(np.median(reprojection)),
        "reprojection_error_px_max": float(np.max(reprojection)),
        "depth_consistency_median_mm": float(np.median(depth_consistency)) if depth_consistency else None,
        "validation": validation,
        "T_world_base": solved["T_world_base"],
        "T_ee_camera": solved["T_ee_camera"],
        "solve_rejected_frames": solve_rejected,
    }


def _geometry_quality(left: dict, right: dict, third: dict, rejected: dict, expected_poses: int) -> dict:
    # The 96-pose set is intentionally redundant. Do not make a partial but
    # solvable run unusable merely because one camera lost several views.
    minimum_accepted = 6
    thresholds = {
        "minimum_accepted_poses_per_arm": minimum_accepted,
        "maximum_reprojection_error_px": 2.0,
        "maximum_held_out_translation_mm": 10.0,
        "maximum_held_out_rotation_deg": 2.0,
        "maximum_solver_translation_disagreement_mm": 10.0,
        "maximum_bootstrap_translation_p95_mm": 10.0,
        "maximum_depth_consistency_median_mm": 30.0,
    }
    reasons = []
    warnings = []
    for arm, result in (("left", left), ("right", right)):
        if result["accepted_poses"] < minimum_accepted:
            warnings.append(f"{arm}_insufficient_accepted_poses")
        validation = result.get("validation") or {}
        agreement = validation.get("solver_agreement") or {}
        bootstrap = validation.get("bootstrap") or {}
        if result["reprojection_error_px_max"] > thresholds["maximum_reprojection_error_px"]:
            warnings.append(f"{arm}_reprojection")
        if validation.get("held_out_translation_max_mm") is not None and validation["held_out_translation_max_mm"] > thresholds["maximum_held_out_translation_mm"]:
            warnings.append(f"{arm}_held_out_translation")
        if validation.get("held_out_rotation_max_deg") is not None and validation["held_out_rotation_max_deg"] > thresholds["maximum_held_out_rotation_deg"]:
            warnings.append(f"{arm}_held_out_rotation")
        if agreement.get("base_translation_mm") is not None and max(agreement["base_translation_mm"], agreement["handeye_translation_mm"]) > thresholds["maximum_solver_translation_disagreement_mm"]:
            warnings.append(f"{arm}_solver_disagreement")
        if bootstrap.get("base_translation_p95_mm") is not None and max(bootstrap["base_translation_p95_mm"], bootstrap["handeye_translation_p95_mm"]) > thresholds["maximum_bootstrap_translation_p95_mm"]:
            warnings.append(f"{arm}_bootstrap_uncertainty")
        depth = result.get("depth_consistency_median_mm")
        if depth is not None and depth > thresholds["maximum_depth_consistency_median_mm"]:
            reasons.append(f"{arm}_depth_consistency")
    if third["reprojection_error_px"] > thresholds["maximum_reprojection_error_px"]:
        warnings.append("third_reprojection")
    if third.get("depth_consistency_mm") is not None and third["depth_consistency_mm"] > thresholds["maximum_depth_consistency_median_mm"]:
        warnings.append("third_depth_consistency")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "thresholds": thresholds,
        "rejected_pose_counts": {arm: len(rejected.get(arm, [])) for arm in ("left", "right")},
    }


def run(args: argparse.Namespace) -> dict:
    library = json.loads(args.pose_library.read_text(encoding="utf-8"))
    target_manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    target = CharucoTargetSpec.from_manifest(args.target_manifest)
    if not 12 <= args.poses <= min(len(library["left"]), len(library["right"])):
        raise ValueError("--poses must be between 12 and the available pose count")
    args.output.mkdir(parents=True, exist_ok=True)
    burst_frames = int(getattr(args, "burst_frames", 3))
    burst_interval_s = float(getattr(args, "burst_interval_s", 0.1))
    stable_for_s = float(getattr(args, "stable_for_s", 2.0))
    minimum_accepted_frames = int(getattr(args, "minimum_accepted_frames", 1))
    if not 1 <= minimum_accepted_frames <= burst_frames * 2:
        raise ValueError("minimum accepted frames must be between 1 and the six-frame logical pose burst")
    run_config = {
        "schema_version": 1,
        "pose_library_sha256": _sha256(args.pose_library),
        "target_manifest_sha256": _sha256(args.target_manifest),
        "target_id": library["target"]["target_id"],
        "poses_per_arm": args.poses,
        "capture": {
            "stable_for_s": stable_for_s,
            "burst_frames": burst_frames,
            "burst_interval_s": burst_interval_s,
            "minimum_accepted_frames_per_logical_pose": minimum_accepted_frames,
            "gripper_states": {"closed": 0.0, "open": 1.0},
        },
    }
    if target_manifest.get("target_id") != run_config["target_id"]:
        raise ValueError("pose library and target manifest target_id do not match")
    run_config_path = args.output / "run_config.json"
    if run_config_path.is_file():
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        if previous != run_config:
            raise RuntimeError("output directory belongs to a different pose library, target, or pose count")
    committed: dict[str, dict] = {}
    manifest_path = args.output / "manifest.jsonl"
    if manifest_path.is_file() and not run_config_path.is_file():
        raise RuntimeError("existing manifest.jsonl has no run_config.json and cannot be resumed safely")
    if not run_config_path.is_file():
        run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                committed[record["name"]] = record
    client = RobotControlClient(args.robotcontrol_url)
    left_records: list[dict] = []
    right_records: list[dict] = []
    rejected: dict[str, list[dict]] = {"left": [], "right": []}
    park = [0.0, 0.05, -0.05, 0.0, 0.0, 0.0]
    policy = CapturePolicy(
        burst_frames=burst_frames,
        burst_interval_s=burst_interval_s,
        stable_for_s=stable_for_s,
        stable_poll_s=0.02,
        stable_timeout_s=10.0,
        # Closed/open are acquired independently. Quality is evaluated after
        # merging both three-frame bursts into one logical pose.
        minimum_accepted_frames_per_camera=0,
        # The simulator's white studio background legitimately occupies more
        # than 40% of pixels. ChArUco reprojection remains strictly gated.
        quality=FrameQualityPolicy(maximum_clipped_fraction=0.75),
    )
    capture = CalibrationCaptureSession(client, args.output, policy, target)

    def raise_if_cancelled() -> None:
        should_cancel = getattr(args, "should_cancel", None)
        if callable(should_cancel) and should_cancel():
            raise InterruptedError("geometry capture cancelled by operator")

    try:
        backend = client.state().get("backend")
        expected_backend = getattr(args, "expected_backend", None)
        if expected_backend is not None and backend != expected_backend:
            raise RuntimeError(f"expected robotcontrol backend={expected_backend!r}, got {backend!r}")
        physical = backend != "robotwin"
        if physical:
            park = [0.0] * 6
        client.acquire("calibration-workcell-geometry")
        client.start_heartbeat(interval_s=3.0)
        if physical:
            client.arm_motion()
        progress = getattr(args, "progress", None)
        for arm, camera_id, destination in (
            ("left", "left_wrist", left_records),
            ("right", "right_wrist", right_records),
        ):
            for index, pose in enumerate(library[arm][:args.poses], start=1):
                raise_if_cancelled()
                name = f"{arm}_{index:03d}"
                variant_names = [f"{name}_closed", f"{name}_open"]
                if callable(progress):
                    reused = all(variant in committed for variant in variant_names)
                    progress(arm, index - 1, args.poses, "reuse" if reused else "capture")
                pose_records = []
                variant_errors: dict[str, str] = {}
                for gripper_state, gripper_value in (("closed", 0.0), ("open", 1.0)):
                    raise_if_cancelled()
                    left = list(pose) if arm == "left" else park
                    right = list(pose) if arm == "right" else park
                    left_gripper = gripper_value if arm == "left" else 0.0
                    right_gripper = gripper_value if arm == "right" else 0.0
                    waypoint = left + [left_gripper] + right + [right_gripper]
                    variant_name = f"{name}_{gripper_state}"
                    record = None
                    if variant_name in committed:
                        record = committed[variant_name]
                        saved_target = record.get("target_qpos")
                        if saved_target is None or not np.allclose(saved_target, waypoint, rtol=0.0, atol=1e-9):
                            raise RuntimeError(f"committed pose {variant_name} does not match the active pose library")
                        print(f"[{arm}] reuse {index:02d}/{args.poses} {gripper_state}", flush=True)
                    else:
                        for attempt in range(1, 3):
                            raise_if_cancelled()
                            print(f"[{arm}] capture {index:02d}/{args.poses} {gripper_state} attempt={attempt}", flush=True)
                            try:
                                record = capture.capture_pose(
                                    variant_name, waypoint, required_target_cameras=(camera_id,),
                                    metadata={"gripper_state": gripper_state},
                                )
                                break
                            except Exception as exc:
                                raise_if_cancelled()
                                print(f"[{arm}] rejected {index:02d} {gripper_state}: {type(exc).__name__}: {exc}", flush=True)
                                if _lease_was_lost(exc):
                                    raise RuntimeError(
                                        "robotcontrol lease was lost; aborting geometry capture immediately"
                                    ) from exc
                                variant_errors[gripper_state] = f"{type(exc).__name__}: {exc}"
                    if record is not None:
                        pose_records.append(record)
                raise_if_cancelled()
                merged = _merge_gripper_records(name, pose_records) if pose_records else None
                accepted = _accepted_frame_count(merged, camera_id) if merged is not None else 0
                if accepted < minimum_accepted_frames:
                    rejected[arm].append({
                        "index": index, "name": name,
                        "error": (
                            f"logical pose has {accepted}/{burst_frames * 2} accepted {camera_id} frames; "
                            f"required={minimum_accepted_frames}; variant_errors={variant_errors}"
                        ),
                    })
                else:
                    merged["accepted_target_frame_count"] = accepted
                    if variant_errors:
                        merged["gripper_capture_failures"] = variant_errors
                    destination.append(merged)
                    if callable(progress):
                        progress(arm, index, args.poses, "captured_closed3_open3")
        if len(left_records) < 6 or len(right_records) < 6:
            raise RuntimeError(
                f"insufficient accepted poses for solve: left={len(left_records)}, right={len(right_records)}"
            )
        left = _solve_arm(args.output, left_records, "left_wrist", "left", target)
        right = _solve_arm(args.output, right_records, "right_wrist", "right", target)
        camera_target, head_metrics, head_source = _solve_third_camera(
            args.output, left_records + right_records, target,
        )
        world_camera_third = np.linalg.inv(camera_target)
        base_distance = float(np.linalg.norm(
            left["T_world_base"][:3, 3] - right["T_world_base"][:3, 3],
        ))
        result: dict = {
            "schema_version": 1,
            "backend": f"{backend}-over-http",
            "poses_per_arm": args.poses,
            "rejected_poses": rejected,
            "target_id": library["target"]["target_id"],
            "left": left,
            "right": right,
            "third": {
                "corner_count": head_metrics["corner_count"],
                "reprojection_error_px": head_metrics["reprojection_error_px"],
                "depth_consistency_mm": head_metrics["depth_consistency_mm"],
                "source": head_source,
                "T_world_camera": world_camera_third,
                "height_from_target_plane_mm": float(abs(world_camera_third[2, 3]) * 1000.0),
                "pitch_deg": _camera_pitch_deg(world_camera_third),
            },
            "dual_base_distance_mm": base_distance * 1000.0,
            "base_height_from_target_plane_mm": {
                "left": float(abs(left["T_world_base"][2, 3]) * 1000.0),
                "right": float(abs(right["T_world_base"][2, 3]) * 1000.0),
            },
        }
        result["quality"] = _geometry_quality(left, right, result["third"], rejected, args.poses)
        if args.truth is not None:
            truth = json.loads(args.truth.read_text(encoding="utf-8"))
            errors = {}
            for arm, solved in (("left", left), ("right", right)):
                errors[f"{arm}_base"] = transform_error(
                    solved["T_world_base"], _matrix(truth[f"T_target_base_{arm}"]),
                )
                errors[f"{arm}_handeye"] = transform_error(
                    solved["T_ee_camera"], _matrix(truth[f"T_ee_camera_{arm}"]),
                )
            errors["third_camera"] = transform_error(
                world_camera_third, _matrix(truth["T_target_camera_third"]),
            )
            truth_distance = np.linalg.norm(
                _matrix(truth["T_target_base_left"])[:3, 3]
                - _matrix(truth["T_target_base_right"])[:3, 3]
            )
            result["truth_errors"] = {
                key: {"translation_mm": value[0], "rotation_deg": value[1]}
                for key, value in errors.items()
            }
            result["dual_base_distance_truth_mm"] = float(truth_distance * 1000.0)
            result["dual_base_distance_error_mm"] = float(abs(base_distance - truth_distance) * 1000.0)
            truth_third = _matrix(truth["T_target_camera_third"])
            result["third"]["height_truth_mm"] = float(abs(truth_third[2, 3]) * 1000.0)
            result["third"]["height_error_mm"] = float(
                abs(abs(world_camera_third[2, 3]) - abs(truth_third[2, 3])) * 1000.0
            )
            result["third"]["pitch_truth_deg"] = _camera_pitch_deg(truth_third)
            result["third"]["pitch_error_deg"] = float(abs(
                _camera_pitch_deg(world_camera_third) - _camera_pitch_deg(truth_third)
            ))
            result["base_height_truth_mm"] = {
                arm: float(abs(_matrix(truth[f"T_target_base_{arm}"])[2, 3]) * 1000.0)
                for arm in ("left", "right")
            }
            result["base_height_error_mm"] = {
                arm: float(abs(
                    result["base_height_from_target_plane_mm"][arm]
                    - result["base_height_truth_mm"][arm]
                ))
                for arm in ("left", "right")
            }
        serializable = _jsonable(result)
        (args.output / "workcell_calibration.json").write_text(
            json.dumps(serializable, indent=2) + "\n", encoding="utf-8",
        )
        return serializable
    finally:
        try:
            client.hold()
            client.release()
        except Exception:
            pass
        client.close()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotcontrol-url", default="http://127.0.0.1:8009")
    parser.add_argument(
        "--pose-library", type=Path,
        default=root / "assets/pose_libraries/robotwin_dual_tabletop_BOARD_002_48x2_pose_libraries.json",
    )
    parser.add_argument(
        "--target-manifest", type=Path,
        default=root / "assets/targets/BOARD_002_A4_Charuco_5x7_35mm_metric.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poses", type=int, default=48)
    parser.add_argument("--expected-backend")
    parser.add_argument("--stable-for-s", type=float, default=2.0)
    parser.add_argument("--burst-frames", type=int, default=3)
    parser.add_argument("--burst-interval-s", type=float, default=0.1)
    parser.add_argument(
        "--minimum-accepted-frames", type=int, default=1,
        help="Minimum valid active-wrist frames across the closed-3 plus open-3 logical pose.",
    )
    parser.add_argument("--truth", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
