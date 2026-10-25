from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


ACTION_DIM = 14
CAMERA_IDS = ("left_wrist", "right_wrist", "third_view")
EXECUTION_STATES = ("running", "completed", "held", "failed")


class AcquireRequest(BaseModel):
    owner_id: str = Field(min_length=1, max_length=128)


class TokenRequest(BaseModel):
    token: str = Field(min_length=16)


class CameraControlsRequest(BaseModel):
    exposure: float = Field(gt=0)
    gain: float = Field(ge=0)


class ExecuteRequest(TokenRequest):
    waypoints: list[list[float]] = Field(min_length=1, max_length=128)
    profile: Literal[
        "calibration_slow",
        "vla_conservative",
        "vla_topp_conservative",
        "recovery_slow",
    ] = "vla_conservative"
    speed_scale: float = Field(default=1.0, gt=0.0, le=2.0)

    @field_validator("waypoints")
    @classmethod
    def validate_waypoints(cls, value: list[list[float]]) -> list[list[float]]:
        import math

        for point in value:
            if len(point) != ACTION_DIM:
                raise ValueError(f"each waypoint must have {ACTION_DIM} values")
            if not all(math.isfinite(x) for x in point):
                raise ValueError("waypoints must contain only finite values")
        return value


class AcquireResponse(BaseModel):
    token: str
    expires_at_monotonic_s: float


class ExecutionResponse(BaseModel):
    execution_id: str
    planned_samples: int
    state: Literal["running"]


class ExecutionStatusResponse(BaseModel):
    execution_id: str
    planned_samples: int
    completed_samples: int
    state: Literal["running", "completed", "held", "failed"]
    reason: str | None = None
