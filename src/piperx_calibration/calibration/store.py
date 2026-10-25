from __future__ import annotations

import json
from pathlib import Path

from .schemas import RunStatus, SessionState, utc_now


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_pointer = self.root / "active_session.txt"

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def save(self, state: SessionState, *, event: str) -> None:
        directory = self.session_dir(state.session_id)
        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_json(directory / "session.json", state.model_dump(mode="json"))
        with (directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "at": utc_now(), "event": event, "revision": state.revision,
                "stage": state.stage, "status": state.status,
            }, ensure_ascii=False) + "\n")
        self.active_pointer.write_text(state.session_id + "\n", encoding="utf-8")

    def load_active(self) -> SessionState | None:
        if not self.active_pointer.is_file():
            return None
        session_id = self.active_pointer.read_text(encoding="utf-8").strip()
        path = self.session_dir(session_id) / "session.json"
        if not path.is_file():
            return None
        state = SessionState.model_validate_json(path.read_text(encoding="utf-8"))
        if state.status == RunStatus.RUNNING:
            state.status = RunStatus.RECOVERY_REQUIRED
            state.recovery_required = True
            state.robot_holding = True
            state.instruction = "服务曾在运行中退出。请检查现场，确认机械臂保持后再继续。"
            state.revision += 1
            state.updated_at = utc_now()
            self.save(state, event="server_recovery_required")
        return state
