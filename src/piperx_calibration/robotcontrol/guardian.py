from __future__ import annotations

import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

try:
    import toppra as _toppra
    from toppra import constraint as _toppra_constraint
except Exception:  # Keep legacy profiles usable on minimal read-only hosts.
    _toppra = None
    _toppra_constraint = None

from .driver import DualArmDriver
from .schemas import ACTION_DIM


class ControlError(RuntimeError):
    status_code = 400


class LeaseConflict(ControlError):
    status_code = 409


class LeaseDenied(ControlError):
    status_code = 403


class ExecutionNotFound(ControlError):
    status_code = 404


@dataclass(frozen=True)
class MotionProfile:
    max_joint_step_rad: float
    max_gripper_step: float
    sample_period_s: float
    settle_tolerance_rad: float
    settle_tolerance_gripper: float
    settle_timeout_s: float
    topp_enabled: bool = False
    max_velocity_rad_s: tuple[float, ...] | None = None
    max_acceleration_rad_s2: tuple[float, ...] | None = None
    # Keep the final target commanded for a short dwell so the physical
    # controller can settle after a fast terminal press/release.
    terminal_hold_s: float = 0.0


PROFILES = {
    # The physical first-motion controller is deliberately capped at 5%.
    # A 5-degree joint probe can therefore need longer than ten seconds even
    # though every streamed command was accepted.
    # Keep the per-command joint delta unchanged for safety, but send the
    # calibration trajectory 10% faster.  The physical Piper-X speed is
    # already capped at its configured 10%; this only removes idle cadence.
    "calibration_slow": MotionProfile(0.022, 0.02, 0.045, 0.02, 0.02, 30.0),
    "vla_conservative": MotionProfile(0.05, 0.04, 0.02, 0.03, 0.03, 10.0),
    # Real-time TOPP-RA retiming.  The limits are intentionally below the
    # existing fixed-step ceiling and can be calibrated per robot deployment.
    # The HTTP request and waypoint format remain unchanged; callers opt in by
    # selecting this additive profile name.
    "vla_topp_conservative": MotionProfile(
        0.05, 0.04, 0.02, 0.06, 0.10, 20.0,
        topp_enabled=True,
        max_velocity_rad_s=(0.8,) * 12,
        max_acceleration_rad_s2=(2.0,) * 12,
        terminal_hold_s=0.30,
    ),
    # Hardware at a deliberately low first-motion speed may still be moving
    # after the final streamed sample.  Do not report Go Home complete until
    # measured feedback reaches the goal.
    # Go Home is a recovery action, but 33 Hz was unnecessarily slow.  Keep
    # the same per-command deltas and tolerances while doubling command rate.
    "recovery_slow": MotionProfile(0.015, 0.015, 0.015, 0.015, 0.02, 20.0),
}


@dataclass
class _Execution:
    execution_id: str
    planned_samples: int
    state: Literal["running", "completed", "held", "failed"] = "running"
    reason: str | None = None
    completed_samples: int = 0


@dataclass
class MotionGuardian:
    """Owns the lease and runs one cancellable, time-based joint trajectory.

    This class intentionally has no hardware-specific policy.  It limits each
    consecutive command, sends commands at an explicit cadence, and turns a
    lost lease or error into HOLD (command the measured position), never into
    disable/reset/go-home.
    """

    driver: DualArmDriver
    safe_home: list[float] = field(default_factory=lambda: [0.0] * ACTION_DIM)
    lease_ttl_s: float = 15.0
    joint_limit_rad: float = 2.8
    gripper_min: float = 0.0
    gripper_max: float = 1.0
    cadence_scale: float = 1.0
    idle_disconnect_s: float = 60.0
    _token: str | None = None
    _owner_id: str | None = None
    _last_alive_s: float | None = None
    _holding: bool = True
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _cancel: threading.Event | None = field(default=None, init=False, repr=False)
    _worker: threading.Thread | None = field(default=None, init=False, repr=False)
    _execution: _Execution | None = field(default=None, init=False)
    _last_control_activity_s: float = field(default_factory=time.monotonic, init=False, repr=False)
    _idle_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _idle_worker: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be positive")
        if self.idle_disconnect_s <= 0:
            raise ValueError("idle_disconnect_s must be positive")
        if callable(getattr(self.driver, "release_connection", None)):
            self._idle_worker = threading.Thread(target=self._idle_loop, name="robotcontrol-can-idle", daemon=True)
            self._idle_worker.start()

    def _touch_control_locked(self) -> None:
        self._last_control_activity_s = self._now()

    def _idle_loop(self) -> None:
        while not self._idle_stop.wait(min(1.0, self.idle_disconnect_s / 4.0)):
            with self._lock:
                active_motion = self._worker is not None and self._worker.is_alive()
                if self._token is not None or active_motion or not self._holding:
                    continue
                if self._now() - self._last_control_activity_s < self.idle_disconnect_s:
                    continue
                try:
                    self.driver.release_connection()
                except Exception:
                    pass
                self._last_control_activity_s = self._now()

    def close(self) -> None:
        self.hold(reason="service_shutdown")
        self._idle_stop.set()
        if self._idle_worker is not None:
            self._idle_worker.join(timeout=1.0)

    def _now(self) -> float:
        return time.monotonic()

    def _expire_if_needed_locked(self) -> None:
        if self._token is not None and self._last_alive_s is not None and self._now() - self._last_alive_s > self.lease_ttl_s:
            self._hold_locked("lease_expired")
            self._token = self._owner_id = self._last_alive_s = None

    def acquire(self, owner_id: str) -> tuple[str, float]:
        with self._lock:
            self._expire_if_needed_locked()
            if self._token is not None:
                raise LeaseConflict(f"robot is already controlled by {self._owner_id}")
            ensure_connected = getattr(self.driver, "ensure_connected", None)
            if callable(ensure_connected):
                try:
                    ensure_connected()
                except Exception as exc:
                    raise ControlError(f"robot CAN reconnect failed: {type(exc).__name__}: {exc}") from exc
            if not self.driver.is_healthy():
                raise ControlError("driver is unhealthy after CAN reconnect")
            self._token = secrets.token_urlsafe(24)
            self._owner_id = owner_id
            self._last_alive_s = self._now()
            self._touch_control_locked()
            return self._token, self._last_alive_s + self.lease_ttl_s

    def _check_token_locked(self, token: str) -> None:
        self._expire_if_needed_locked()
        if self._token is None or not secrets.compare_digest(token, self._token):
            raise LeaseDenied("invalid or expired control token")

    def alive(self, token: str) -> float:
        with self._lock:
            self._check_token_locked(token)
            self._last_alive_s = self._now()
            self._touch_control_locked()
            return self._last_alive_s + self.lease_ttl_s

    def release(self, token: str) -> None:
        with self._lock:
            self._check_token_locked(token)
            self._hold_locked("control_released")
            self._token = self._owner_id = self._last_alive_s = None
            self._touch_control_locked()

    def arm_motion(self, token: str) -> None:
        """Explicitly enable physical motion without emitting a target pose."""
        with self._lock:
            self._check_token_locked(token)
            if self._worker is not None and self._worker.is_alive():
                raise LeaseConflict("cannot arm while a trajectory is running")
            arm = getattr(self.driver, "arm_motion", None)
            if not callable(arm):
                raise ControlError("this robotcontrol backend does not support physical motion")
            try:
                arm()
            except Exception as exc:
                self._hold_locked("arm_motion_failed")
                raise ControlError(f"physical motion arm failed: {type(exc).__name__}: {exc}") from exc
            self._holding = True
            self._touch_control_locked()

    def _normalize_target(self, target: list[float]) -> list[float]:
        normalized = list(target)
        for index, value in enumerate(normalized):
            if index in (6, 13):
                # OpenPI emits unconstrained normalized gripper values and the
                # simulator clips them to its [0, 1] actuator range. Mirror
                # that contract before TOPP and hardware command generation.
                normalized[index] = min(self.gripper_max, max(self.gripper_min, value))
            elif abs(value) > self.joint_limit_rad:
                raise ControlError(f"joint index {index} exceeds configured soft limit")
        return normalized

    @staticmethod
    def _segment_samples(start: list[float], end: list[float], profile: MotionProfile) -> list[list[float]]:
        steps = 1
        for index, (a, b) in enumerate(zip(start, end, strict=True)):
            limit = profile.max_gripper_step if index in (6, 13) else profile.max_joint_step_rad
            steps = max(steps, int(abs(b - a) / limit + 0.999999))
        return [[a + (b - a) * i / steps for a, b in zip(start, end, strict=True)] for i in range(1, steps + 1)]

    @staticmethod
    def _topp_samples(start: list[float], waypoints: list[list[float]], profile: MotionProfile) -> list[list[float]]:
        """Retiming that preserves every input waypoint.

        TOPP-RA computes a time parameterization over a clamped spline.  The
        exact times corresponding to the original path points are inserted
        into the fixed-period sample grid and those samples are replaced by
        the original points, so retiming never changes the policy targets.
        """
        if profile.max_velocity_rad_s is None or profile.max_acceleration_rad_s2 is None:
            raise ControlError("TOPP profile is missing velocity/acceleration limits")
        if len(profile.max_velocity_rad_s) != 12 or len(profile.max_acceleration_rad_s2) != 12:
            raise ControlError("TOPP limits must contain 12 arm-joint values")
        if _toppra is None or _toppra_constraint is None:
            raise ControlError("TOPP solver is unavailable; install toppra for vla_topp_conservative")

        points = np.asarray([start, *waypoints], dtype=np.float64)
        arm_points = points[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
        distances = np.linalg.norm(np.diff(arm_points, axis=0), axis=1)
        # SplineInterpolator requires strictly increasing path coordinates;
        # repeated policy points are still retained in ``points`` and receive
        # a negligible path interval.
        distances = np.maximum(distances, 1.0e-7)
        ss = np.concatenate(([0.0], np.cumsum(distances)))
        path = _toppra.SplineInterpolator(ss, arm_points, bc_type="clamped")
        algorithm = _toppra.algorithm.TOPPRA(
            [
                _toppra_constraint.JointVelocityConstraint(np.asarray(profile.max_velocity_rad_s)),
                _toppra_constraint.JointAccelerationConstraint(np.asarray(profile.max_acceleration_rad_s2)),
            ],
            path,
            parametrizer="ParametrizeConstAccel",
        )
        trajectory = algorithm.compute_trajectory(0.0, 0.0)
        if trajectory is None or not np.isfinite(float(trajectory.duration)):
            raise ControlError("TOPP-RA failed to produce a finite trajectory")
        duration = float(trajectory.duration)
        if duration > 60.0:
            raise ControlError(f"TOPP trajectory duration is unreasonable: {duration:.3f}s")

        # TOPP-RA exposes the path/time grid used by its parametrizer.  Use it
        # only to map original path coordinates to exact times; all output
        # positions are evaluated through the public trajectory callable.
        waypoint_times = np.interp(ss, trajectory._ss, trajectory._ts)
        sample_times = np.arange(0.0, duration, profile.sample_period_s, dtype=np.float64)
        sample_times = np.unique(np.concatenate((sample_times, waypoint_times, [duration])))
        arm_samples = np.asarray(trajectory(sample_times), dtype=np.float64)
        for index, timestamp in enumerate(waypoint_times):
            exact_index = int(np.argmin(np.abs(sample_times - timestamp)))
            arm_samples[exact_index] = arm_points[index]
        gripper_points = points[:, [6, 13]]
        gripper_samples = np.column_stack([
            np.interp(sample_times, waypoint_times, gripper_points[:, column])
            for column in range(2)
        ])
        full = np.zeros((len(sample_times), ACTION_DIM), dtype=np.float64)
        full[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] = arm_samples
        full[:, [6, 13]] = gripper_samples
        # Do not resend the measured starting point; every policy waypoint is
        # retained, including repeated points and the final target.
        result = full[1:].astype(np.float32).tolist()
        if profile.terminal_hold_s > 0.0 and result:
            hold_count = max(1, int(np.ceil(profile.terminal_hold_s / profile.sample_period_s)))
            result.extend([result[-1]] * hold_count)
        return result

    def execute(
        self, token: str, waypoints: list[list[float]], profile_name: str, *,
        allow_home_only: bool = False, speed_scale: float = 1.0,
    ) -> tuple[str, int]:
        with self._lock:
            if self.cadence_scale <= 0:
                raise ControlError("cadence_scale must be positive")
            if speed_scale <= 0 or speed_scale > 2.0:
                raise ControlError("speed_scale must be in (0, 2]")
            self._check_token_locked(token)
            if not bool(getattr(self.driver, "motion_enabled", True)):
                raise ControlError("motion commands are disabled for this robotcontrol backend")
            if bool(getattr(self.driver, "home_only", False)) and not allow_home_only:
                raise ControlError("this first-motion backend permits only explicit Go Home")
            if bool(getattr(self.driver, "requires_explicit_arm", False)) and not bool(getattr(self.driver, "is_motion_armed", False)):
                raise ControlError("physical motion is not armed; call /v1/motions/arm first")
            allowed_profiles = getattr(self.driver, "allowed_motion_profiles", None)
            if allowed_profiles is not None and profile_name not in allowed_profiles:
                raise ControlError(f"profile {profile_name!r} is not permitted by this physical backend")
            max_home_delta = getattr(self.driver, "first_motion_max_joint_delta_rad", None)
            if max_home_delta is not None and not allow_home_only:
                for target in waypoints:
                    left_moved = any(abs(target[index] - self.safe_home[index]) > 1e-9 for index in range(6))
                    right_moved = any(abs(target[index] - self.safe_home[index]) > 1e-9 for index in range(7, 13))
                    if left_moved and right_moved:
                        raise ControlError("first-motion backend permits one arm at a time")
                    moved_joints = sum(
                        abs(value - home) > 1e-9
                        for index, (value, home) in enumerate(zip(target, self.safe_home, strict=True))
                        if index not in (6, 13)
                    )
                    if moved_joints > 1:
                        raise ControlError("first-motion backend permits one joint at a time")
                    for index, (value, home) in enumerate(zip(target, self.safe_home, strict=True)):
                        if index in (6, 13):
                            continue
                        if abs(value - home) > max_home_delta:
                            raise ControlError(
                                f"joint index {index} exceeds first-motion limit of {max_home_delta:.3f} rad from Home"
                            )
            if self._worker is not None and self._worker.is_alive():
                raise LeaseConflict("another trajectory is still running; call hold before replacing it")
            if not self.driver.is_healthy():
                self._hold_locked("driver_unhealthy")
                raise ControlError("driver is unhealthy")
            profile = PROFILES[profile_name]
            if profile.topp_enabled and speed_scale != 1.0:
                profile = replace(
                    profile,
                    max_velocity_rad_s=tuple(value * speed_scale for value in profile.max_velocity_rad_s or ()),
                    max_acceleration_rad_s2=tuple(value * speed_scale * speed_scale for value in profile.max_acceleration_rad_s2 or ()),
                )
            current = self.driver.read_qpos()
            waypoints = [self._normalize_target(target) for target in waypoints]
            if profile.topp_enabled:
                samples = self._topp_samples(current, waypoints, profile)
            else:
                samples = []
                for target in waypoints:
                    samples.extend(self._segment_samples(current, target, profile))
                    current = target
            if not samples:
                raise ControlError("trajectory retiming produced no samples")
            execution = _Execution(str(uuid.uuid4()), len(samples))
            cancel = threading.Event()
            worker = threading.Thread(
                target=self._run_trajectory,
                args=(execution, cancel, samples, profile),
                name=f"robotcontrol-{execution.execution_id[:8]}", daemon=True,
            )
            self._execution, self._cancel, self._worker = execution, cancel, worker
            self._holding = False
            self._touch_control_locked()
            worker.start()
            return execution.execution_id, len(samples)

    @staticmethod
    def _at_target(actual: list[float], target: list[float], profile: MotionProfile) -> bool:
        if len(actual) != ACTION_DIM:
            return False
        for index, (value, goal) in enumerate(zip(actual, target, strict=True)):
            tolerance = profile.settle_tolerance_gripper if index in (6, 13) else profile.settle_tolerance_rad
            if abs(value - goal) > tolerance:
                return False
        return True

    def _run_trajectory(self, execution: _Execution, cancel: threading.Event, samples: list[list[float]], profile: MotionProfile) -> None:
        period_s = profile.sample_period_s * self.cadence_scale
        try:
            for target in samples:
                with self._lock:
                    self._expire_if_needed_locked()
                    if cancel.is_set():
                        return
                    if not self.driver.is_healthy():
                        self._hold_locked("driver_unhealthy")
                        execution.state, execution.reason = "failed", "driver_unhealthy"
                        return
                    try:
                        self.driver.command_qpos(target)
                    except Exception as exc:  # hardware exceptions must become HOLD, not process exit
                        self._hold_locked("driver_command_failed")
                        execution.state, execution.reason = "failed", f"driver_command_failed: {type(exc).__name__}"
                        return
                    execution.completed_samples += 1
                if cancel.wait(period_s):
                    return
            # Sending the final waypoint is not evidence that a physical arm
            # reached it.  In particular, low-speed controller motion keeps
            # running after the last streamed sample.  Keep the lease-owned
            # trajectory alive until feedback is actually within tolerance.
            deadline = time.monotonic() + profile.settle_timeout_s * self.cadence_scale
            while True:
                with self._lock:
                    self._expire_if_needed_locked()
                    if cancel.is_set() or execution.state != "running":
                        return
                    if not self.driver.is_healthy():
                        self._hold_locked("driver_unhealthy")
                        execution.state, execution.reason = "failed", "driver_unhealthy"
                        return
                    try:
                        actual = self.driver.read_qpos()
                    except Exception as exc:
                        self._hold_locked("driver_feedback_failed")
                        execution.state, execution.reason = "failed", f"driver_feedback_failed: {type(exc).__name__}"
                        return
                    if self._at_target(actual, samples[-1], profile):
                        execution.state, execution.reason = "completed", None
                        return
                    if time.monotonic() >= deadline:
                        self._hold_locked("target_settle_timeout")
                        errors = [abs(value - goal) for value, goal in zip(actual, samples[-1], strict=True)]
                        arm_index = max((i for i in range(ACTION_DIM) if i not in (6, 13)), key=errors.__getitem__)
                        gripper_index = max((6, 13), key=errors.__getitem__)
                        execution.state = "failed"
                        execution.reason = (
                            f"target_settle_timeout: arm_index={arm_index} arm_error_rad={errors[arm_index]:.4f} "
                            f"gripper_index={gripper_index} gripper_error={errors[gripper_index]:.4f}"
                        )
                        return
                if cancel.wait(min(0.05, period_s)):
                    return
        finally:
            with self._lock:
                if execution.state == "running":
                    execution.state, execution.reason = "held", "trajectory_cancelled"

    def _hold_locked(self, reason: str) -> None:
        if self._cancel is not None:
            self._cancel.set()
        if self.driver.is_healthy():
            try:
                self.driver.command_qpos(self.driver.read_qpos())
            except Exception:
                # The caller still receives the original control error; a real
                # host will record this as a hardware fault in a later phase.
                pass
        if self._execution is not None and self._execution.state == "running":
            self._execution.state, self._execution.reason = "held", reason
        self._holding = True

    def hold(self, *, reason: str) -> None:
        with self._lock:
            self._hold_locked(reason)
            self._touch_control_locked()
            worker = self._worker
        # A cancelled worker is waiting on an Event, so this should return
        # promptly.  Joining outside the lock prevents a replacement motion
        # racing the final command of the cancelled trajectory.
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)

    def go_home(self, token: str) -> tuple[str, int]:
        return self.execute(token, [self.safe_home], "recovery_slow", allow_home_only=True)

    def execution_status(self, execution_id: str) -> dict:
        with self._lock:
            self._expire_if_needed_locked()
            execution = self._execution
            if execution is None or execution.execution_id != execution_id:
                raise ExecutionNotFound(f"unknown execution_id: {execution_id}")
            return {
                "execution_id": execution.execution_id,
                "planned_samples": execution.planned_samples,
                "completed_samples": execution.completed_samples,
                "state": execution.state,
                "reason": execution.reason,
            }

    def state(self) -> dict:
        with self._lock:
            self._expire_if_needed_locked()
            execution = self._execution
            try:
                qpos = self.driver.read_qpos()
                state_error = None
            except Exception as exc:
                qpos = None
                state_error = f"{type(exc).__name__}: {exc}"
            read_dq = getattr(self.driver, "read_dq", None)
            try:
                dq_rad_s = read_dq() if callable(read_dq) else None
            except Exception:
                dq_rad_s = None
            read_kinematics = getattr(self.driver, "read_kinematics", None)
            try:
                kinematics = read_kinematics() if callable(read_kinematics) else None
            except Exception:
                kinematics = None
            return {
                "backend": str(getattr(self.driver, "backend_name", type(self.driver).__name__)),
                "timestamp_monotonic_s": self._now(),
                "qpos": qpos,
                "dq_rad_s": dq_rad_s,
                "kinematics": kinematics,
                "healthy": self.driver.is_healthy(),
                "holding": self._holding,
                "lease_owner": self._owner_id,
                "lease_active": self._token is not None,
                "execution_id": execution.execution_id if execution else None,
                "execution_state": execution.state if execution else "idle",
                "execution_reason": execution.reason if execution else None,
                "state_read_error": state_error,
                "can_connected": bool(getattr(self.driver, "is_connected", lambda: True)()),
            }
