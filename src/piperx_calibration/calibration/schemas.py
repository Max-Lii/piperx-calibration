from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Stage(StrEnum):
    IDLE = "IDLE"
    SYSTEM_CHECK = "SYSTEM_CHECK"
    TARGET_VERIFY = "TARGET_VERIFY"
    SAFETY_DRY_RUN = "SAFETY_DRY_RUN"
    A4_PLACEMENT = "A4_PLACEMENT"
    WORLD_REGISTRATION = "WORLD_REGISTRATION"
    CAPTURE_LEFT = "CAPTURE_LEFT"
    CAPTURE_RIGHT = "CAPTURE_RIGHT"
    SOLVE_GEOMETRY = "SOLVE_GEOMETRY"
    VALIDATE_GEOMETRY = "VALIDATE_GEOMETRY"
    REVIEW_AND_EXPORT = "REVIEW_AND_EXPORT"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


class RunStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    RECOVERY_REQUIRED = "recovery_required"
    COMPLETE = "complete"
    ABORTED = "aborted"


class DeviceStatus(BaseModel):
    device_id: str
    label: str
    kind: Literal["robot", "camera", "target"]
    connected: bool
    healthy: bool
    detail: str = ""


class ProgressState(BaseModel):
    current: int = 0
    total: int = 0
    label: str = ""


class SessionState(BaseModel):
    schema_version: int = 1
    session_id: str
    revision: int = 0
    mode: Literal["mock", "robotwin", "http"]
    stage: Stage
    status: RunStatus
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    instruction: str = ""
    active_camera: Literal["left_wrist", "right_wrist", "third_view"] = "third_view"
    devices: list[DeviceStatus] = Field(default_factory=list)
    progress: ProgressState = Field(default_factory=ProgressState)
    robot_holding: bool = True
    recovery_required: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    results: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    stage_ready: bool = False
    processed_command_ids: list[str] = Field(default_factory=list)


class StartSessionRequest(BaseModel):
    mode: Literal["mock", "robotwin", "http"] | None = None


class CommandRequest(BaseModel):
    command: Literal[
        "continue", "skip_pose_check", "pause", "retry", "abort", "terminate", "restart", "confirm_placement",
        "approve_pose", "reject_pose", "go_home",
    ]
    command_id: str = Field(min_length=8, max_length=128)
    expected_revision: int = Field(ge=0)
