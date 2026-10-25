from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RobotControlSettings:
    host: str
    port: int
    lease_ttl_s: float
    idle_disconnect_s: float
    driver_kind: str
    left_can: str | None
    right_can: str | None
    firmware_profile: str | None
    can_interface: str
    motion_speed_percent: int | None
    joint_limits_rad: tuple[tuple[float, float], ...] | None
    first_motion_max_joint_delta_rad: float | None
    allow_calibration_pose_motion: bool
    gripper_open_width_m: float | None
    gripper_force_n: float
    camera_kind: str
    camera_serial_by_role: dict[str, str]
    camera_enabled_roles: tuple[str, ...]
    camera_depth_enabled: bool
    camera_width: int
    camera_height: int
    camera_fps: int
    camera_warmup_frames: int
    camera_lock_auto_controls_after_warmup: bool
    camera_manual_exposure: float | None
    image_processing_enabled: bool
    image_processing_backend: str
    image_processing_device: str
    image_processing_pinned_memory: bool
    image_processing_torch_compile: bool
    image_processing_lut_by_role: dict[str, str]
    image_processing_jpeg_quality: int
    joint_limit_rad: float
    safe_home: list[float]
    auth_mode: str
    api_key_env: str


def load_settings(path: Path | None = None) -> RobotControlSettings:
    path = path or Path(os.environ.get("ROBOTCONTROL_CONFIG", Path(__file__).with_name("config.yaml")))
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    image_processing = raw["camera"].get("image_processing", {})
    processing_backend = str(image_processing.get("backend", "cpu"))
    if processing_backend not in {"cpu", "cuda", "auto"}:
        raise ValueError("camera.image_processing.backend must be cpu, cuda or auto")
    jpeg_quality = int(image_processing.get("jpeg_quality", 90))
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("camera.image_processing.jpeg_quality must be in [1, 100]")
    safe_home = list(raw["safety"]["safe_home"])
    if len(safe_home) != 14:
        raise ValueError("safety.safe_home must contain 14 values")
    raw_limits = raw["driver"].get("joint_limits_rad")
    joint_limits_rad: tuple[tuple[float, float], ...] | None = None
    if raw_limits is not None:
        if not isinstance(raw_limits, list) or len(raw_limits) != 6:
            raise ValueError("driver.joint_limits_rad must contain six [min, max] pairs")
        parsed = []
        for index, pair in enumerate(raw_limits):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"driver.joint_limits_rad[{index}] must be [min, max]")
            low, high = float(pair[0]), float(pair[1])
            if not low < high:
                raise ValueError(f"driver.joint_limits_rad[{index}] must satisfy min < max")
            parsed.append((low, high))
        joint_limits_rad = tuple(parsed)
    enabled_roles = tuple(str(role) for role in raw["camera"].get("enabled_roles", ("left_wrist", "right_wrist", "third_view")))
    known_roles = {"left_wrist", "right_wrist", "third_view"}
    if not enabled_roles or not set(enabled_roles).issubset(known_roles) or len(set(enabled_roles)) != len(enabled_roles):
        raise ValueError("camera.enabled_roles must contain one or more unique known camera roles")
    return RobotControlSettings(
        host=str(raw["server"]["host"]), port=int(raw["server"]["port"]),
        lease_ttl_s=float(raw["lease"]["ttl_s"]), idle_disconnect_s=float(raw["connection"].get("idle_disconnect_s", 60.0)), driver_kind=str(raw["driver"]["kind"]),
        left_can=raw["driver"].get("left_can"), right_can=raw["driver"].get("right_can"),
        firmware_profile=raw["driver"].get("firmware_profile"), can_interface=str(raw["driver"].get("interface", "socketcan")),
        motion_speed_percent=(None if raw["driver"].get("motion_speed_percent") is None else int(raw["driver"]["motion_speed_percent"])),
        joint_limits_rad=joint_limits_rad,
        first_motion_max_joint_delta_rad=(None if raw["driver"].get("first_motion_max_joint_delta_rad") is None else float(raw["driver"]["first_motion_max_joint_delta_rad"])),
        allow_calibration_pose_motion=bool(raw["driver"].get("allow_calibration_pose_motion", False)),
        gripper_open_width_m=(None if raw["driver"].get("gripper_open_width_m") is None else float(raw["driver"]["gripper_open_width_m"])),
        gripper_force_n=float(raw["driver"].get("gripper_force_n", 1.0)),
        camera_kind=str(raw["camera"]["kind"]),
        camera_serial_by_role={str(key): str(value) for key, value in raw["camera"].get("serial_by_role", {}).items()},
        camera_enabled_roles=enabled_roles,
        camera_depth_enabled=bool(raw["camera"].get("depth_enabled", True)),
        camera_width=int(raw["camera"].get("width", 640)),
        camera_height=int(raw["camera"].get("height", 480)),
        camera_fps=int(raw["camera"].get("fps", 30)),
        camera_warmup_frames=int(raw["camera"].get("warmup_frames", 15)),
        camera_lock_auto_controls_after_warmup=bool(raw["camera"].get("lock_auto_controls_after_warmup", False)),
        camera_manual_exposure=(None if raw["camera"].get("manual_exposure") is None else float(raw["camera"]["manual_exposure"])),
        image_processing_enabled=bool(image_processing.get("enabled", False)),
        image_processing_backend=processing_backend,
        image_processing_device=str(image_processing.get("device", "cuda:0")),
        image_processing_pinned_memory=bool(image_processing.get("pinned_memory", False)),
        image_processing_torch_compile=bool(image_processing.get("torch_compile", True)),
        image_processing_lut_by_role={
            str(key): str(value) for key, value in image_processing.get("lut_by_role", {}).items()
        },
        image_processing_jpeg_quality=jpeg_quality,
        joint_limit_rad=float(raw["safety"]["joint_limit_rad"]),
        safe_home=safe_home, auth_mode=str(raw["auth"]["mode"]), api_key_env=str(raw["auth"]["api_key_env"]),
    )
