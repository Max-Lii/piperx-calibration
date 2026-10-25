from __future__ import annotations

import json
import asyncio
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from .adapter import HttpWorkcellAdapter, MockWorkcellAdapter
from .app import create_app
from .schemas import CommandRequest, RunStatus, SessionState, Stage
from .settings import CalibrationSettings
from .store import SessionStore
from .workflow import CalibrationWorkflow
from .geometry import HandEyeObservation, hand_eye_validation, solve_robot_world_hand_eye, transform_error


REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_MANIFEST = REPO_ROOT / "assets/targets/BOARD_002_A4_Charuco_5x7_35mm_metric.json"


def _settings(tmp_path: Path) -> CalibrationSettings:
    return CalibrationSettings(
        host="127.0.0.1", port=8010, mode="mock",
        robotcontrol_url="http://127.0.0.1:8008",
        session_root=tmp_path / "sessions",
        frontend_dist=tmp_path / "missing-frontend",
        preview_fps=20.0,
        target_settle_s=0.0,
        target_manifest=TARGET_MANIFEST,
    )


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def call() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://calibrator",
        ) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(call())


def test_api_session_is_persistent_revisioned_and_idempotent(tmp_path):
    settings = _settings(tmp_path)
    app = create_app(settings=settings)
    assert _request(app, "GET", "/api/v1/session").json() is None
    state = _request(app, "POST", "/api/v1/session/start", json={}).json()
    assert state["stage"] == "SYSTEM_CHECK"
    request = {"command": "continue", "command_id": "command-0001", "expected_revision": 0}
    advanced = _request(app, "POST", "/api/v1/session/command", json=request)
    assert advanced.status_code == 200
    assert advanced.json()["stage"] == "TARGET_VERIFY"
    duplicate = _request(app, "POST", "/api/v1/session/command", json=request)
    assert duplicate.status_code == 200
    assert duplicate.json()["revision"] == advanced.json()["revision"]
    stale = _request(app, "POST", "/api/v1/session/command", json={
        "command": "continue", "command_id": "command-0002", "expected_revision": 0,
    })
    assert stale.status_code == 409
    saved = json.loads(next(settings.session_root.glob("calibration_*/session.json")).read_text())
    assert saved["stage"] == "TARGET_VERIFY"
    assert "command-0001" in saved["processed_command_ids"]


def test_caliper_measured_scale_can_skip_photo_verification(tmp_path):
    app = create_app(settings=_settings(tmp_path))
    state = _request(app, "POST", "/api/v1/session/start", json={}).json()
    state = _request(app, "POST", "/api/v1/session/command", json={
        "command": "continue", "command_id": "enter-target-verify",
        "expected_revision": state["revision"],
    }).json()
    assert state["stage"] == "TARGET_VERIFY" and state["stage_ready"]

    state = _request(app, "POST", "/api/v1/board-verification/use-measured-scale", json={}).json()
    verification = state["artifacts"]["board_verification"]
    assert state["stage_ready"]
    assert verification["verification_method"] == "operator_caliper_measurement"
    assert verification["square_length_mm"] == 35.0
    assert verification["scale_x"] == verification["scale_y"] == 1.0


def test_placement_confirmation_requires_all_three_complete_views(tmp_path):
    adapter = MockWorkcellAdapter()
    app = create_app(settings=_settings(tmp_path), adapter=adapter)
    state = _request(app, "POST", "/api/v1/session/start", json={}).json()
    for index in range(3):
        state = _request(app, "POST", "/api/v1/session/command", json={
            "command": "continue", "command_id": f"enter-placement-{index}",
            "expected_revision": state["revision"],
        }).json()
    assert state["stage"] == "A4_PLACEMENT"

    state = _request(app, "POST", "/api/v1/placement/confirm", json={
        "command": "confirm_placement", "command_id": "confirm-three-cameras",
        "expected_revision": state["revision"],
    }).json()
    detection = state["artifacts"]["placement_detection"]
    assert detection["all_cameras_accepted"]
    assert detection["virtual"]


def test_mock_workflow_reaches_complete_and_restores_interrupted_session(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    workflow = CalibrationWorkflow(store, MockWorkcellAdapter(), "mock")
    state = workflow.start()
    for index in range(10):
        state = workflow.command(CommandRequest(
            command="continue", command_id=f"complete-{index:03d}", expected_revision=state.revision,
        ))
    assert state.stage == Stage.COMPLETE
    assert state.status == RunStatus.COMPLETE
    assert state.robot_holding is True
    assert state.results["quality_status"] == "passed"

    second_store = SessionStore(tmp_path / "other")
    interrupted = CalibrationWorkflow(second_store, MockWorkcellAdapter(), "mock")
    interrupted.start()
    restored = CalibrationWorkflow(second_store, MockWorkcellAdapter(), "mock").snapshot()
    assert restored is not None
    assert restored.status == RunStatus.RECOVERY_REQUIRED
    assert restored.robot_holding is True


def test_workcell_probe_reads_all_lossless_cameras_without_motion(tmp_path):
    app = create_app(settings=_settings(tmp_path))
    response = _request(app, "GET", "/api/v1/workcell/probe")
    assert response.status_code == 200
    payload = response.json()
    assert payload["motion_requested"] is False
    assert set(payload["cameras"]) == {"left_wrist", "right_wrist", "third_view"}
    for camera in payload["cameras"].values():
        assert camera["rgb_shape"] == [480, 640, 3]
        assert camera["depth_shape"] == [480, 640]
        assert camera["depth_aligned_to_color"] is True
        assert camera["payload_bytes"] > 1000


def test_http_system_check_accepts_idle_physical_can_standby():
    class Client:
        def state(self):
            return {
                "backend": "PiperXMotionDriver", "healthy": False,
                "can_connected": False, "holding": True, "lease_active": False,
            }

        def frame_jpeg(self, _):
            return b"jpeg", 1.0

    adapter = HttpWorkcellAdapter.__new__(HttpWorkcellAdapter)
    adapter.client = Client()
    devices = adapter.system_check()
    assert all(device.connected and device.healthy for device in devices)
    assert devices[0].detail == "robotcontrol (CAN standby)"


def test_hold_is_always_separate_from_normal_transition(tmp_path):
    app = create_app(settings=_settings(tmp_path))
    _request(app, "POST", "/api/v1/session/start", json={})
    held = _request(app, "POST", "/api/v1/safety/hold").json()
    assert held["status"] == "paused"
    assert held["robot_holding"] is True
    assert held["stage"] == "SYSTEM_CHECK"


def test_robotwin_pose_check_and_placement_are_explicitly_gated(tmp_path):
    workflow = CalibrationWorkflow(SessionStore(tmp_path / "sessions"), MockWorkcellAdapter(), "robotwin")
    state = workflow.start()
    state = workflow.command(CommandRequest(command="continue", command_id="robotwin-1", expected_revision=state.revision))
    state = workflow.command(CommandRequest(command="continue", command_id="robotwin-2", expected_revision=state.revision))
    assert state.stage == Stage.SAFETY_DRY_RUN
    with pytest.raises(Exception, match="工作台动作"):
        workflow.command(CommandRequest(command="continue", command_id="robotwin-blocked", expected_revision=state.revision))
    started = workflow.begin_robotwin_operation("pose_check", total=96, label="准备")
    assert started.progress.total == 96 and not started.robot_holding
    progressed = workflow.update_robotwin_operation("pose_check", current=17, label="左臂 17/48")
    assert progressed.progress.current == 17
    checked = workflow.finish_robotwin_operation("pose_check", success=True, detail="完成")
    moved = workflow.command(CommandRequest(command="continue", command_id="robotwin-3", expected_revision=checked.revision))
    assert moved.stage == Stage.A4_PLACEMENT
    with pytest.raises(Exception, match="棋盘格放置姿态"):
        workflow.command(CommandRequest(
            command="confirm_placement",
            command_id="robotwin-stale-pose-check",
            expected_revision=moved.revision,
        ))
    with pytest.raises(Exception, match="工作台动作"):
        workflow.command(CommandRequest(command="continue", command_id="robotwin-placement-blocked", expected_revision=moved.revision))
    workflow.begin_robotwin_operation("placement", total=1, label="移动")
    placed = workflow.finish_robotwin_operation("placement", success=True, detail="到位")
    confirmed = workflow.command(CommandRequest(command="confirm_placement", command_id="robotwin-placement-confirm", expected_revision=placed.revision))
    assert confirmed.stage_ready
    registration = workflow.command(CommandRequest(
        command="continue", command_id="robotwin-registration", expected_revision=confirmed.revision,
    ))
    workflow.begin_geometry(total=96, session_id=registration.session_id)
    with pytest.raises(Exception, match="Session"):
        workflow.update_geometry(
            arm="left", index=1, poses_per_arm=48, phase="capture", session_id="stale-session",
        )


def test_terminate_requires_hold_goes_home_and_starts_fresh_paused_session(tmp_path):
    class RecordingAdapter(MockWorkcellAdapter):
        def __init__(self):
            super().__init__()
            self.hold_calls = 0
            self.go_home_calls = 0

        def hold(self) -> None:
            self.hold_calls += 1

        def go_home(self) -> None:
            self.go_home_calls += 1

    adapter = RecordingAdapter()
    workflow = CalibrationWorkflow(SessionStore(tmp_path / "sessions"), adapter, "mock")
    running = workflow.start()
    with pytest.raises(Exception, match="请先停止机器人执行"):
        workflow.command(CommandRequest(
            command="terminate", command_id="terminate-running", expected_revision=running.revision,
        ))

    held = workflow.hold()
    fresh = workflow.command(CommandRequest(
        command="terminate", command_id="terminate-held", expected_revision=held.revision,
    ))
    assert adapter.hold_calls == 1
    assert adapter.go_home_calls == 1
    assert fresh.session_id != held.session_id
    assert fresh.stage == Stage.SYSTEM_CHECK
    assert fresh.status == RunStatus.PAUSED
    assert fresh.robot_holding is True
    assert fresh.stage_ready is True

    old = SessionState.model_validate_json(
        (tmp_path / "sessions" / held.session_id / "session.json").read_text()
    )
    assert old.stage == Stage.ABORTED
    assert old.status == RunStatus.ABORTED


def test_failed_terminate_keeps_current_session_paused(tmp_path):
    class FailingAdapter(MockWorkcellAdapter):
        def go_home(self) -> None:
            raise RuntimeError("go-home failed")

    workflow = CalibrationWorkflow(
        SessionStore(tmp_path / "sessions"), FailingAdapter(), "mock",
    )
    workflow.start()
    held = workflow.hold()
    with pytest.raises(RuntimeError, match="go-home failed"):
        workflow.command(CommandRequest(
            command="terminate", command_id="terminate-failed", expected_revision=held.revision,
        ))
    current = workflow.snapshot()
    assert current is not None
    assert current.session_id == held.session_id
    assert current.stage == Stage.SYSTEM_CHECK
    assert current.status == RunStatus.PAUSED


def test_http_adapter_go_home_acquires_arms_and_releases():
    from .adapter import HttpWorkcellAdapter

    class RecordingClient:
        token = None

        def __init__(self):
            self.calls = []

        def acquire(self, owner_id):
            self.calls.append(("acquire", owner_id))
            self.token = "lease"

        def start_heartbeat(self, interval_s):
            self.calls.append(("start_heartbeat", interval_s))

        def arm_motion(self):
            self.calls.append(("arm_motion",))

        def go_home(self):
            self.calls.append(("go_home",))
            return {"execution_id": "home-1"}

        def wait_execution(self, execution_id, timeout_s):
            self.calls.append(("wait_execution", execution_id, timeout_s))
            return {"state": "completed"}

        def stop_heartbeat(self):
            self.calls.append(("stop_heartbeat",))

        def release(self):
            self.calls.append(("release",))
            self.token = None

    adapter = HttpWorkcellAdapter("http://unused")
    adapter.client.close()
    client = RecordingClient()
    adapter.client = client
    adapter.go_home()

    assert [call[0] for call in client.calls] == [
        "acquire", "start_heartbeat", "arm_motion", "go_home",
        "wait_execution", "stop_heartbeat", "release",
    ]


def test_complete_session_exports_and_can_restart_from_home(tmp_path):
    class RecordingAdapter(MockWorkcellAdapter):
        def __init__(self):
            super().__init__()
            self.go_home_calls = 0

        def go_home(self) -> None:
            self.go_home_calls += 1

    adapter = RecordingAdapter()
    workflow = CalibrationWorkflow(SessionStore(tmp_path / "sessions"), adapter, "mock")
    state = workflow.start()
    for index in range(10):
        state = workflow.command(CommandRequest(
            command="continue", command_id=f"export-{index:03d}", expected_revision=state.revision,
        ))
    artifact = workflow.export_payload()
    assert artifact["artifact"] == "piper_workcell_calibration"
    assert artifact["session"]["status"] == "complete"
    assert "T_world_third_camera" in artifact["results"]["transforms"]

    fresh = workflow.command(CommandRequest(
        command="restart", command_id="restart-complete", expected_revision=state.revision,
    ))
    assert adapter.go_home_calls == 1
    assert fresh.session_id != state.session_id
    assert fresh.stage == Stage.SYSTEM_CHECK
    assert fresh.status == RunStatus.PAUSED


def test_robot_world_hand_eye_matrix_directions():
    rng = np.random.default_rng(7)

    def transform(rotation_vector, translation):
        value = np.eye(4)
        value[:3, :3], _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))
        value[:3, 3] = translation
        return value

    world_base = transform([0.08, -0.04, 0.12], [-0.21, 0.39, 0.016])
    ee_camera = transform([-0.2, 0.12, 0.05], [0.02, -0.03, 0.08])
    observations = []
    for _ in range(24):
        base_ee = transform(rng.uniform(-0.7, 0.7, 3), rng.uniform(-0.25, 0.25, 3))
        world_camera = world_base @ base_ee @ ee_camera
        observations.append(HandEyeObservation(
            camera_target=np.linalg.inv(world_camera), base_ee=base_ee,
        ))
    solved = solve_robot_world_hand_eye(observations)
    assert transform_error(solved["T_world_base"], world_base)[0] < 1e-5
    assert transform_error(solved["T_ee_camera"], ee_camera)[0] < 1e-5
    validation = hand_eye_validation(observations, bootstrap_samples=30)
    assert validation["held_out_translation_max_mm"] < 1e-4
    assert validation["solver_agreement"]["base_translation_mm"] < 1e-4
    assert validation["bootstrap"]["valid_samples"] >= 20
