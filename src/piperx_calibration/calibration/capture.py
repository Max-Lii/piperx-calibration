from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .quality import (
    CharucoQualityPolicy, CharucoTargetSpec, FrameQualityPolicy,
    evaluate_charuco, evaluate_frame,
)
from piperx_calibration.robotcontrol.client import RobotControlClient
from piperx_calibration.robotcontrol.schemas import CAMERA_IDS


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_JOINT_INDICES = tuple(i for i in range(14) if i not in (6, 13))


@dataclass(frozen=True)
class CapturePolicy:
    burst_frames: int = 12
    burst_interval_s: float = 0.04
    joint_tolerance_rad: float = 0.015
    velocity_tolerance_rad_s: float = 0.03
    gripper_tolerance: float = 0.03
    stable_for_s: float = 2.0
    stable_poll_s: float = 0.05
    stable_timeout_s: float = 10.0
    motion_timeout_s: float = 45.0
    minimum_accepted_frames_per_camera: int = 8
    quality: FrameQualityPolicy = field(default_factory=FrameQualityPolicy)
    charuco_quality: CharucoQualityPolicy = field(default_factory=CharucoQualityPolicy)

    def __post_init__(self) -> None:
        if self.burst_frames < 1:
            raise ValueError("burst_frames must be positive")
        if not 0 <= self.minimum_accepted_frames_per_camera <= self.burst_frames:
            raise ValueError("minimum accepted frames must be within the burst size")


@dataclass
class CalibrationCaptureSession:
    """Transactional, HTTP-only pose/burst capture for later SE(3) solving.

    A pose is committed to ``manifest.jsonl`` only after motion, feedback-based
    stationary gating, all camera bursts and quality gates succeed.  Every
    ``.npz`` is the original lossless payload returned by robotcontrol.
    """

    client: RobotControlClient
    output_dir: Path
    policy: CapturePolicy = field(default_factory=CapturePolicy)
    target_spec: CharucoTargetSpec | None = None

    def _state_errors(self, state: dict, target: list[float]) -> tuple[float, float, float]:
        qpos = state.get("qpos")
        dq = state.get("dq_rad_s")
        if not isinstance(qpos, list) or len(qpos) != 14:
            raise RuntimeError("robotcontrol state has no valid 14D qpos")
        if not isinstance(dq, list) or len(dq) != 14:
            raise RuntimeError("robotcontrol state has no verified 14D joint velocity")
        return (
            max(abs(qpos[i] - target[i]) for i in _JOINT_INDICES),
            max(abs(dq[i]) for i in _JOINT_INDICES),
            max(abs(qpos[i] - target[i]) for i in (6, 13)),
        )

    def _wait_stable(self, target: list[float]) -> dict:
        deadline = time.monotonic() + self.policy.stable_timeout_s
        stable_since: float | None = None
        last: dict | None = None
        while time.monotonic() < deadline:
            last = self.client.state()
            joint_error, joint_velocity, gripper_error = self._state_errors(last, target)
            stationary = (
                joint_error <= self.policy.joint_tolerance_rad
                and joint_velocity <= self.policy.velocity_tolerance_rad_s
                and gripper_error <= self.policy.gripper_tolerance
            )
            now = time.monotonic()
            stable_since = stable_since if stationary and stable_since is not None else (now if stationary else None)
            if stable_since is not None and now - stable_since >= self.policy.stable_for_s:
                return {
                    "state": last,
                    "stable_for_s": now - stable_since,
                    "max_joint_error_rad": joint_error,
                    "max_joint_velocity_rad_s": joint_velocity,
                    "max_gripper_error": gripper_error,
                }
            time.sleep(self.policy.stable_poll_s)
        raise TimeoutError(f"robot did not become stationary; last state={last}")

    def capture_pose(
        self,
        name: str,
        target: list[float],
        *,
        required_target_cameras: tuple[str, ...] = (),
        metadata: dict | None = None,
    ) -> dict:
        try:
            return self._capture_pose(
                name, target, required_target_cameras=required_target_cameras, metadata=metadata,
            )
        except Exception:
            # The control service owns the definitive hardware policy.  This
            # call only requests its safe HOLD and never disables or homes.
            try:
                self.client.hold()
            except Exception:
                pass
            raise

    def _capture_pose(
        self,
        name: str,
        target: list[float],
        *,
        required_target_cameras: tuple[str, ...],
        metadata: dict | None,
    ) -> dict:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("pose name must be a safe portable filename component")
        if len(target) != 14 or not all(math.isfinite(value) for value in target):
            raise ValueError("target must be a finite 14D waypoint")
        unknown = set(required_target_cameras) - set(CAMERA_IDS)
        if unknown:
            raise ValueError(f"unknown required target cameras: {sorted(unknown)}")
        if required_target_cameras and self.target_spec is None:
            raise ValueError("target_spec is required for ChArUco quality gating")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.output_dir / "manifest.jsonl"
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if line and json.loads(line).get("name") == name:
                    raise FileExistsError(f"pose {name!r} is already committed")

        execution = self.client.execute_joint_waypoints([target], profile="calibration_slow")
        status = self.client.wait_execution(
            execution["execution_id"], timeout_s=self.policy.motion_timeout_s,
        )
        if status["state"] != "completed":
            raise RuntimeError(f"calibration move did not complete: {status}")
        stable = self._wait_stable(target)
        attempt_id = str(time.time_ns())
        pose_dir = self.output_dir / "captures" / name / attempt_id
        pose_dir.mkdir(parents=True, exist_ok=False)
        frames: dict[str, list[dict]] = {camera_id: [] for camera_id in CAMERA_IDS}
        try:
            for burst_index in range(self.policy.burst_frames):
                for camera_id in CAMERA_IDS:
                    before = self.client.state()
                    sample, payload = self.client.calibration_frame(camera_id)
                    after = self.client.state()
                    for observed_state in (before, after):
                        joint_error, velocity, gripper_error = self._state_errors(observed_state, target)
                        if (
                            joint_error > self.policy.joint_tolerance_rad
                            or velocity > self.policy.velocity_tolerance_rad_s
                            or gripper_error > self.policy.gripper_tolerance
                        ):
                            raise RuntimeError("robot moved outside the stationary gate during burst capture")
                    quality = evaluate_frame(sample, self.policy.quality)
                    target_quality = (
                        evaluate_charuco(sample, self.target_spec, self.policy.charuco_quality)
                        if camera_id in required_target_cameras else None
                    )
                    filename = f"{camera_id}_{burst_index:03d}.npz"
                    (pose_dir / filename).write_bytes(payload)
                    frames[camera_id].append({
                        "file": str((pose_dir / filename).relative_to(self.output_dir)),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "frame_number": sample.frame_number,
                        "serial": sample.serial,
                        "host_timestamp_monotonic_s": sample.host_timestamp_monotonic_s,
                        "hardware_timestamp_ms": sample.hardware_timestamp_ms,
                        "camera_controls": sample.camera_controls,
                        "robot_state_before": before,
                        "robot_state_after": after,
                        "quality": quality.to_dict(),
                        "target_quality": target_quality,
                    })
                if burst_index + 1 < self.policy.burst_frames:
                    time.sleep(self.policy.burst_interval_s)
            accepted_counts = {
                camera_id: sum(
                    item["quality"]["accepted"]
                    and (item["target_quality"] is None or item["target_quality"]["accepted"])
                    for item in items
                )
                for camera_id, items in frames.items()
            }
            gated_cameras = required_target_cameras or CAMERA_IDS
            insufficient = {
                camera_id: accepted_counts[camera_id] for camera_id in gated_cameras
                if accepted_counts[camera_id] < self.policy.minimum_accepted_frames_per_camera
            }
            if insufficient:
                raise RuntimeError(f"burst quality gate failed: {insufficient}")
            record = {
                "schema_version": 2,
                "name": name,
                "attempt_id": attempt_id,
                "captured_at_unix_s": time.time(),
                "target_qpos": target,
                "execution": status,
                "stationary_gate": stable,
                "capture_policy": asdict(self.policy),
                "accepted_frame_counts": accepted_counts,
                "frames": frames,
            }
            if metadata:
                record["metadata"] = metadata
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            return record
        except Exception:
            # Partial files intentionally remain for diagnosis, but without a
            # manifest commit they can never enter a solver data set.
            raise
