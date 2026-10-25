from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class DualArmDriver(Protocol):
    """Hardware boundary. Only the future motion-guardian may implement this."""

    def read_qpos(self) -> list[float]: ...

    def read_dq(self) -> list[float]: ...

    def is_healthy(self) -> bool: ...

    def command_qpos(self, target: list[float]) -> None: ...

    def read_kinematics(self) -> dict[str, list[list[float]]]: ...


@dataclass
class MockDualPiperDriver:
    """Deterministic offline driver; it never imports pyAgxArm or opens CAN."""

    qpos: list[float] = field(default_factory=lambda: [0.0] * 14)
    healthy: bool = True
    command_history: list[list[float]] = field(default_factory=list)
    backend_name: str = "mock"

    def read_qpos(self) -> list[float]:
        return self.qpos.copy()

    def read_dq(self) -> list[float]:
        # The mock applies trajectory samples instantaneously.  A completed
        # execution is therefore stationary by construction.
        return [0.0] * len(self.qpos)

    def is_healthy(self) -> bool:
        return self.healthy

    def command_qpos(self, target: list[float]) -> None:
        if not self.healthy:
            raise RuntimeError("mock driver is unhealthy")
        self.qpos = target.copy()
        self.command_history.append(target.copy())

    def read_kinematics(self) -> dict[str, list[list[float]]]:
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return {"left_T_base_ee": identity, "right_T_base_ee": identity}
