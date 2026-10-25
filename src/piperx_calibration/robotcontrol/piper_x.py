"""Explicit-only pyAgxArm boundary for the robotcontrol service.

Safety invariant: this module must never call ``disable()``, SDK ``reset()``
or ``electronic_emergency_stop()``.  HOLD is an enabled position hold and
``close()`` only closes the host CAN connection.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .schemas import ACTION_DIM


@dataclass(frozen=True)
class PiperXConfig:
    left_can: str
    right_can: str
    firmware_profile: str
    interface: str = "socketcan"
    # Motion is deliberately a separate, opt-in backend.  The read-only
    # driver never creates this class.
    motion_speed_percent: int = 5
    joint_limits_rad: tuple[tuple[float, float], ...] | None = None
    first_motion_max_joint_delta_rad: float | None = None
    # Explicit lab transition after first-motion acceptance. It retains the
    # lease, HOLD, feedback and configured joint-limit protections.
    allow_calibration_pose_motion: bool = False
    gripper_open_width_m: float | None = None
    gripper_force_n: float = 1.0


class PiperXDualDriver:
    motion_enabled = False

    def __init__(self, config: PiperXConfig) -> None:
        self.config = config
        self._left = None
        self._right = None
        self._left_gripper = None
        self._right_gripper = None
        self._last_qpos_sample: tuple[float, list[float]] | None = None
        self._dq_rad_s: list[float] | None = None

    def is_connected(self) -> bool:
        return bool(self._left and self._right and self._left.is_connected() and self._right.is_connected())

    def ensure_connected(self) -> None:
        """The only place pyAgxArm is imported and CAN can be opened."""
        if self._left is not None and self._right is not None:
            self._left.connect()
            self._right.connect()
            return
        from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config

        def build(channel: str):
            kwargs = {}
            if self.config.joint_limits_rad is not None:
                kwargs["joint_limits"] = {
                    f"joint{index + 1}": list(limit)
                    for index, limit in enumerate(self.config.joint_limits_rad)
                }
            cfg = create_agx_arm_config(
                robot=ArmModel.PIPER_X,
                firmeware_version=self.config.firmware_profile,
                interface=self.config.interface,
                channel=channel,
                **kwargs,
            )
            arm = AgxArmFactory.create_arm(cfg)
            if self.config.joint_limits_rad is not None:
                # pyAgxArm keeps configured joint limits disabled by default.
                # Physical motion must enforce the configured envelope.
                arm.set_joint_limits_enabled(True)
            gripper = None
            if self.config.gripper_open_width_m is not None:
                gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            arm.connect()
            return arm, gripper

        left, left_gripper = build(self.config.left_can)
        try:
            right, right_gripper = build(self.config.right_can)
        except Exception:
            # Closing the host CAN connection is safe and sends no motor
            # enable/disable command. Avoid leaking the first connection if
            # the second interface cannot be opened.
            left.disconnect()
            raise
        self._left, self._right = left, right
        self._left_gripper, self._right_gripper = left_gripper, right_gripper

    connect = ensure_connected

    def release_connection(self) -> None:
        """Release only host-side CAN handles; motors remain enabled."""
        for arm in (self._left, self._right):
            if arm is not None and arm.is_connected():
                arm.disconnect()

    def close(self) -> None:
        """Close CAN resources without changing either arm's enable state."""
        errors: list[Exception] = []
        for arm in (self._left, self._right):
            if arm is not None:
                try:
                    arm.disconnect()
                except Exception as exc:
                    errors.append(exc)
        self._left = self._right = None
        self._left_gripper = self._right_gripper = None
        if errors:
            raise RuntimeError(f"failed to close {len(errors)} Piper-X CAN connection(s)") from errors[0]

    def readonly_diagnostics(self) -> dict:
        """Return cached feedback only; never enable, reset or command motion."""
        if self._left is None or self._right is None:
            raise RuntimeError("PiperXDualDriver is not connected")

        def arm_status(arm) -> dict:
            angles = arm.get_joint_angles()
            status = arm.get_arm_status()
            return {
                "connected": bool(arm.is_connected()),
                "ok": bool(arm.is_ok()),
                "joint_angles_rad": None if angles is None else list(angles.msg),
                "joint_feedback_hz": None if angles is None else float(angles.hz),
                "joint_enabled": list(arm.get_joints_enable_status_list()),
                "arm_status": None if status is None else str(status.msg),
                "can_channel": arm.get_channel(),
            }

        return {"left": arm_status(self._left), "right": arm_status(self._right)}

    def read_qpos(self) -> list[float]:
        if self._left is None or self._right is None:
            raise RuntimeError("PiperXDualDriver is not connected")
        left = self._left.get_joint_angles()
        right = self._right.get_joint_angles()
        if left is None or right is None:
            raise RuntimeError("missing Piper-X joint feedback")
        def gripper_position(gripper) -> float:
            if self.config.gripper_open_width_m is None:
                return 0.0
            status = gripper.get_gripper_status()
            if status is None:
                raise RuntimeError("missing Piper-X gripper feedback")
            return min(max(float(status.msg.value) / self.config.gripper_open_width_m, 0.0), 1.0)

        qpos = [*left.msg, gripper_position(self._left_gripper), *right.msg, gripper_position(self._right_gripper)]
        now = time.monotonic()
        if self._last_qpos_sample is not None:
            previous_at, previous = self._last_qpos_sample
            elapsed = now - previous_at
            if elapsed > 1e-4:
                self._dq_rad_s = [
                    (value - previous[index]) / elapsed
                    for index, value in enumerate(qpos)
                ]
        self._last_qpos_sample = (now, qpos.copy())
        return qpos

    def read_dq(self) -> list[float]:
        if self._dq_rad_s is None:
            # A second feedback sample is required before finite-difference
            # velocity has a physical meaning. Callers must retry state reads.
            raise RuntimeError("Piper-X joint velocity needs two feedback samples")
        return self._dq_rad_s.copy()

    @staticmethod
    def _pose6_matrix(pose: list[float]) -> list[list[float]]:
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            raise RuntimeError("Piper-X FK returned an invalid pose")
        x, y, z, roll, pitch, yaw = pose
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        # pyAgxArm documents flange orientation as ZYX RPY: Rz(yaw) Ry(pitch) Rx(roll).
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
            [-sp, cp * sr, cp * cr, z],
            [0.0, 0.0, 0.0, 1.0],
        ]

    @staticmethod
    def _pose_delta(first: list[list[float]], second: list[list[float]]) -> dict[str, float]:
        translation_m = math.sqrt(sum((first[row][3] - second[row][3]) ** 2 for row in range(3)))
        relative_trace = 0.0
        for row in range(3):
            for col in range(3):
                relative_trace += first[col][row] * second[col][row]
        cosine = max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0))
        return {
            "translation_mm": translation_m * 1000.0,
            "rotation_deg": math.degrees(math.acos(cosine)),
        }

    @classmethod
    def _direct_flange_pose(cls, arm) -> list[list[float]] | None:
        getter = getattr(arm, "get_flange_pose", None)
        if getter is None:
            return None
        feedback = getter()
        if feedback is None:
            return None
        return cls._pose6_matrix(list(feedback.msg))

    def read_kinematics(self) -> dict[str, object]:
        """Return FK plus optional SDK flange feedback for read-only comparison.

        The FK matrix remains the compatibility value used by calibration.  The
        SDK flange pose is the controller's measured CAN feedback and is exposed
        separately so a real-machine check can quantify the model discrepancy.
        """
        if self._left is None or self._right is None:
            raise RuntimeError("PiperXDualDriver is not connected")
        left = self._left.get_joint_angles()
        right = self._right.get_joint_angles()
        if left is None or right is None:
            raise RuntimeError("missing Piper-X joint feedback for FK")
        left_fk = self._pose6_matrix(list(self._left.fk(list(left.msg))))
        right_fk = self._pose6_matrix(list(self._right.fk(list(right.msg))))
        result: dict[str, object] = {
            "left_T_base_ee": left_fk,
            "right_T_base_ee": right_fk,
        }
        for name, arm, fk in (("left", self._left, left_fk), ("right", self._right, right_fk)):
            direct = self._direct_flange_pose(arm)
            if direct is None:
                continue
            result[f"{name}_T_base_ee_sdk"] = direct
            delta = self._pose_delta(fk, direct)
            result[f"{name}_fk_vs_sdk_translation_mm"] = delta["translation_mm"]
            result[f"{name}_fk_vs_sdk_rotation_deg"] = delta["rotation_deg"]
        return result

    def is_healthy(self) -> bool:
        return bool(self.is_connected() and self._left.is_ok() and self._right.is_ok())

    def command_qpos(self, target: list[float]) -> None:
        if len(target) != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}D target")
        raise RuntimeError(
            "real Piper-X command is intentionally disabled until gripper mapping, joint signs, "
            "hold behavior, and physical safety tests are approved"
        )


class PiperXMotionDriver(PiperXDualDriver):
    """Explicitly armed Piper-X motion backend.

    This is intentionally narrow for first motion acceptance:

    * the operator must call :meth:`arm_motion` after taking the HTTP lease;
    * six-joint ``move_j`` is always available; normalized gripper commands
      remain blocked unless a metric open width is explicitly configured;
    * a host-local, per-joint limit table is mandatory, so pyAgxArm's generic
      defaults cannot silently clamp a measured pose with a different sign
      convention.

    It never calls disable/reset/electronic_emergency_stop.  Closing this
    driver only releases host CAN handles.
    """

    motion_enabled = True
    requires_explicit_arm = True

    def __init__(self, config: PiperXConfig) -> None:
        if config.joint_limits_rad is None or len(config.joint_limits_rad) != 6:
            raise ValueError("PiperXMotionDriver requires six host-configured joint limit pairs")
        if not 1 <= config.motion_speed_percent <= 80:
            raise ValueError("first-motion speed must be in [1, 80] percent")
        if config.first_motion_max_joint_delta_rad is not None and not 0 < config.first_motion_max_joint_delta_rad <= 0.0873:
            raise ValueError("first-motion joint delta must be in (0, 0.0873] rad (5 degrees)")
        if config.gripper_open_width_m is not None and (
            not math.isfinite(config.gripper_open_width_m)
            or not 0.0 < config.gripper_open_width_m <= 0.1
        ):
            raise ValueError("gripper open width must be in (0, 0.1] meters")
        if not math.isfinite(config.gripper_force_n) or config.gripper_force_n <= 0.0:
            raise ValueError("gripper force must be positive")
        super().__init__(config)
        self._armed = False

    @property
    def home_only(self) -> bool:
        return self.config.first_motion_max_joint_delta_rad is None and not self.config.allow_calibration_pose_motion

    @property
    def first_motion_max_joint_delta_rad(self) -> float | None:
        return self.config.first_motion_max_joint_delta_rad

    @property
    def allowed_motion_profiles(self) -> frozenset[str]:
        return frozenset({"calibration_slow", "recovery_slow", "vla_topp_conservative"})

    @property
    def is_motion_armed(self) -> bool:
        return self._armed

    def arm_motion(self) -> None:
        """Enable both joint motor groups without issuing a position target."""
        self.ensure_connected()
        assert self._left is not None and self._right is not None
        for side, arm in (("left", self._left), ("right", self._right)):
            enabled = list(arm.get_joints_enable_status_list())
            if len(enabled) != 6:
                raise RuntimeError(f"{side} Piper-X returned invalid joint-enable feedback")
            if not all(enabled):
                # pyAgxArm's enable() sends the CAN frame and immediately
                # examines cached feedback.  On real hardware that feedback
                # can legitimately arrive one polling interval later, so do
                # not mistake a stale first read for an enable failure.
                arm.enable()
        deadline = time.monotonic() + 2.0
        while True:
            statuses = [list(arm.get_joints_enable_status_list()) for arm in (self._left, self._right)]
            if all(len(status) == 6 and all(status) for status in statuses):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Piper-X motor-enable feedback did not become ready within 2 seconds")
            time.sleep(0.05)
        for arm in (self._left, self._right):
            arm.set_speed_percent(self.config.motion_speed_percent)
        self._armed = True

    def command_qpos(self, target: list[float]) -> None:
        if len(target) != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}D target")
        if not self._armed:
            raise RuntimeError("physical motion is not armed; call /v1/motions/arm first")
        if self.config.gripper_open_width_m is None and (abs(target[6]) > 1e-9 or abs(target[13]) > 1e-9):
            raise RuntimeError("physical gripper mapping is not enabled")
        if self._left is None or self._right is None:
            raise RuntimeError("Piper-X motion driver is not connected")
        self._left.move_j(list(target[:6]))
        self._right.move_j(list(target[7:13]))
        if self.config.gripper_open_width_m is not None:
            width = self.config.gripper_open_width_m
            self._left_gripper.move_gripper_m(float(target[6]) * width, self.config.gripper_force_n)
            self._right_gripper.move_gripper_m(float(target[13]) * width, self.config.gripper_force_n)
