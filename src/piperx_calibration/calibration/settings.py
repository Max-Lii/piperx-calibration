from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CalibrationSettings:
    host: str
    port: int
    mode: str
    robotcontrol_url: str
    session_root: Path
    frontend_dist: Path
    preview_fps: float
    target_settle_s: float
    target_manifest: Path
    pose_library: Path | None = None
    poses_per_arm: int = 48
    preview_backend: str = "synthetic"


def load_settings(path: Path | None = None) -> CalibrationSettings:
    path = path or Path(os.environ.get("CALIBRATION_CONFIG", Path(__file__).with_name("config.yaml")))
    path = path.resolve()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (path.parent / candidate).resolve()

    mode = str(raw["workcell"]["mode"])
    if mode not in {"mock", "robotwin", "http"}:
        raise ValueError("workcell.mode must be mock, robotwin, or http")
    pose_library = raw.get("geometry", {}).get("pose_library")
    poses_per_arm = int(raw.get("geometry", {}).get("poses_per_arm", 48))
    if poses_per_arm < 6:
        raise ValueError("geometry.poses_per_arm must be at least 6")
    return CalibrationSettings(
        host=str(raw["server"]["host"]),
        port=int(raw["server"]["port"]),
        mode=mode,
        robotcontrol_url=str(raw["workcell"]["robotcontrol_url"]).rstrip("/"),
        session_root=resolve(str(raw["storage"]["session_root"])),
        frontend_dist=resolve(str(raw["frontend"]["dist"])),
        preview_fps=float(raw["camera"]["preview_fps"]),
        preview_backend=str(raw["camera"].get("preview_backend", "synthetic")),
        target_settle_s=float(raw["target"].get("settle_s", 0.6)),
        target_manifest=resolve(str(raw["target"]["manifest"])),
        pose_library=resolve(str(pose_library)) if pose_library else None,
        poses_per_arm=poses_per_arm,
    )
