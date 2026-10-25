from __future__ import annotations

import time

import cv2
import httpx
import numpy as np
import pytest
from fastapi import HTTPException

from piperx_calibration.robotcontrol.app import create_app
from piperx_calibration.robotcontrol.client import HeartbeatError, RobotControlClient
from piperx_calibration.robotcontrol.driver import MockDualPiperDriver
from piperx_calibration.robotcontrol.guardian import LeaseConflict, LeaseDenied, MotionGuardian, PROFILES
from piperx_calibration.robotcontrol.schemas import AcquireRequest, ExecuteRequest, TokenRequest
from piperx_calibration.robotcontrol.settings import load_settings
from piperx_calibration.robotcontrol.camera import MockCameraManager, decode_calibration_frame
from piperx_calibration.robotcontrol.image_processing import CpuLutProcessor, ProcessedCameraManager
from piperx_calibration.robotcontrol.safety_audit import forbidden_sdk_calls
from piperx_calibration.robotcontrol.piper_x import PiperXConfig, PiperXDualDriver, PiperXMotionDriver
from piperx_calibration.robotcontrol.physical_pose_sweep import _target


def _guardian(*, ttl_s: float = 5.0) -> tuple[MotionGuardian, MockDualPiperDriver]:
    driver = MockDualPiperDriver()
    return MotionGuardian(driver, lease_ttl_s=ttl_s), driver


def _wait(guardian: MotionGuardian, execution_id: str) -> dict:
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        status = guardian.execution_status(execution_id)
        if status["state"] != "running":
            return status
        time.sleep(0.01)
    raise AssertionError("mock execution timed out")


def test_guardian_interpolates_executes_and_goes_home():
    guardian, driver = _guardian()
    token, _ = guardian.acquire("test-vla")
    action = [0.1] * 6 + [0.4] + [-0.1] * 6 + [0.6]
    execution, samples = guardian.execute(token, [action], "vla_conservative")
    assert samples > 1
    assert _wait(guardian, execution)["state"] == "completed"
    assert guardian.state()["qpos"] == action
    home, _ = guardian.go_home(token)
    assert _wait(guardian, home)["state"] == "completed"
    assert guardian.state()["qpos"] == [0.0] * 14
    assert len(driver.command_history) >= samples


def test_guardian_topp_profile_is_bounded_and_reaches_waypoints():
    profile = PROFILES["vla_topp_conservative"]
    start = [0.0] * 14
    waypoints = [
        [0.10] * 6 + [0.2] + [-0.08] * 6 + [0.3],
        [0.18] * 6 + [0.5] + [-0.14] * 6 + [0.7],
    ]
    samples = MotionGuardian._topp_samples(start, waypoints, profile)
    assert samples
    result = np.asarray(samples)
    assert np.isfinite(result).all()
    assert np.any(np.all(np.isclose(result, waypoints[0], atol=1e-6), axis=1))
    assert np.any(np.all(np.isclose(result, waypoints[1], atol=1e-6), axis=1))
    arm = result[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
    velocity = np.diff(np.vstack((np.zeros((1, 12)), arm)), axis=0) / profile.sample_period_s
    assert np.max(np.abs(velocity)) <= max(profile.max_velocity_rad_s) * 1.15


def test_pose_sweep_targets_only_the_active_gripper():
    pose = [0.1] * 6
    assert _target(pose, "left", 1.0) == [*pose, 1.0, *([0.0] * 6), 0.0]
    assert _target(pose, "right", 1.0) == [*([0.0] * 6), 0.0, *pose, 1.0]


def test_guardian_waits_for_measured_settle_before_completed():
    class DelayedFeedbackMock(MockDualPiperDriver):
        def __init__(self):
            super().__init__()
            self._reads_after_command = 0

        def command_qpos(self, target):
            super().command_qpos(target)
            self._reads_after_command = 0

        def read_qpos(self):
            self._reads_after_command += 1
            # Simulate a real low-speed controller whose last target is
            # received before the arm has physically arrived.
            if self.command_history and self._reads_after_command < 3:
                return [0.04] * 6 + [0.0] + [0.04] * 6 + [0.0]
            return super().read_qpos()

    driver = DelayedFeedbackMock()
    guardian = MotionGuardian(driver, cadence_scale=0.1)
    token, _ = guardian.acquire("settle-check")
    execution, _ = guardian.go_home(token)
    assert _wait(guardian, execution)["state"] == "completed"
    assert driver._reads_after_command >= 3


def test_guardian_serializes_motion_hold_and_lease_expiry():
    guardian, driver = _guardian(ttl_s=0.05)
    token, _ = guardian.acquire("one")
    target = [0.8] * 6 + [0.5] + [-0.8] * 6 + [0.5]
    execution, samples = guardian.execute(token, [target], "vla_conservative")
    try:
        guardian.execute(token, [target], "vla_conservative")
    except LeaseConflict:
        pass
    else:
        raise AssertionError("second trajectory should be rejected")
    time.sleep(0.12)  # the worker itself checks the silent-client lease expiry
    status = _wait(guardian, execution)
    assert (status["state"], status["reason"]) == ("held", "lease_expired")
    assert guardian.state()["lease_active"] is False
    assert len(driver.command_history) < samples + 1
    try:
        guardian.alive(token)
    except LeaseDenied:
        pass
    else:
        raise AssertionError("expired token should be rejected")


def test_guardian_hold_and_action_validation():
    guardian, _ = _guardian()
    token, _ = guardian.acquire("one")
    target = [0.6] * 6 + [0.8] + [-0.6] * 6 + [0.8]
    execution, _ = guardian.execute(token, [target], "vla_conservative")
    guardian.hold(reason="api_request")
    assert _wait(guardian, execution)["state"] == "held"
    assert guardian.state()["holding"] is True
    bad = [0.0] * 14
    bad[0] = 3.0
    try:
        guardian.execute(token, [bad], "vla_conservative")
    except Exception as exc:
        assert "soft limit" in str(exc)
    else:
        raise AssertionError("unsafe target should be rejected")


def test_guardian_clips_normalized_grippers_like_simulator():
    guardian, _ = _guardian()
    token, _ = guardian.acquire("gripper-clip")
    target = [0.0] * 14
    target[6], target[13] = -0.01, 1.02
    execution, _ = guardian.execute(token, [target], "vla_conservative")
    assert _wait(guardian, execution)["state"] == "completed"
    assert guardian.state()["qpos"] == [*([0.0] * 6), 0.0, *([0.0] * 6), 1.0]


def test_idle_can_release_after_one_minute_policy_and_lazy_reconnect():
    class ConnectionAwareMock(MockDualPiperDriver):
        def __init__(self):
            super().__init__()
            self.connected = True
            self.release_calls = 0
            self.connect_calls = 0

        def release_connection(self):
            self.release_calls += 1
            self.connected = False

        def ensure_connected(self):
            self.connect_calls += 1
            self.connected = True

        def is_healthy(self):
            return self.connected and self.healthy

        def is_connected(self):
            return self.connected

    driver = ConnectionAwareMock()
    guardian = MotionGuardian(driver, idle_disconnect_s=0.05)
    try:
        deadline = time.monotonic() + 1.0
        while not driver.release_calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert driver.release_calls == 1
        assert guardian.state()["can_connected"] is False
        token, _ = guardian.acquire("after-idle")
        assert token and driver.connect_calls == 1 and driver.connected
    finally:
        guardian.close()


def test_readonly_backend_can_connect_but_rejects_every_motion():
    class ReadonlyMock(MockDualPiperDriver):
        motion_enabled = False

        def ensure_connected(self): pass

    guardian = MotionGuardian(ReadonlyMock())
    try:
        token, _ = guardian.acquire("readonly-probe")
        with pytest.raises(Exception, match="motion commands are disabled"):
            guardian.execute(token, [[0.0] * 14], "calibration_slow")
    finally:
        guardian.close()


def test_fastapi_handlers_and_route_contract():
    guardian, _ = _guardian()
    app = create_app(guardian=guardian)
    routes = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
    required = {
        "/v1/control/acquire", "/v1/control/alive", "/v1/control/release",
        "/v1/motions/execute_joint_waypoints", "/v1/motions/arm", "/v1/motions/go_home", "/v1/motions/hold",
        "/v1/executions/{execution_id}", "/v1/state", "/v1/health",
        "/v1/cameras/{camera_id}/frame", "/v1/cameras/{camera_id}/stream",
        "/v1/cameras/{camera_id}/calibration-frame",
        "/v1/cameras/{camera_id}/ws",
        "/v1/cameras/frame-batch",
        "/v1/cameras/ws",
    }
    assert required <= set(routes)
    token = routes["/v1/control/acquire"](AcquireRequest(owner_id="api-test")).token
    target = [0.1] * 6 + [0.4] + [-0.1] * 6 + [0.6]
    started = routes["/v1/motions/execute_joint_waypoints"](ExecuteRequest(token=token, waypoints=[target]))
    assert started.state == "running"
    assert _wait(guardian, started.execution_id)["state"] == "completed"
    frame = routes["/v1/cameras/{camera_id}/frame"]("left_wrist")
    assert frame.media_type == "image/jpeg" and frame.body.startswith(b"\xff\xd8")
    calibration_frame = routes["/v1/cameras/{camera_id}/calibration-frame"]("left_wrist")
    assert calibration_frame.media_type == "application/x-npz"
    try:
        routes["/v1/cameras/{camera_id}/frame"]("not-a-camera")
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("unknown camera should return HTTP 404")
    routes["/v1/motions/hold"]()
    routes["/v1/control/release"](TokenRequest(token=token))


def test_default_http_client_ignores_host_proxy_environment(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    client = RobotControlClient("http://127.0.0.1:65534")
    try:
        assert client._client._trust_env is False
    finally:
        client.close()


def test_heartbeat_uses_independent_retry_loop_and_surfaces_sustained_failure(monkeypatch):
    client = RobotControlClient("http://127.0.0.1:65534")
    client.token = "x" * 24
    calls = []

    def transient_then_success(_):
        calls.append(time.monotonic())
        if len(calls) < 3:
            raise httpx.ConnectError("transient")

    monkeypatch.setattr(client, "_heartbeat_once", transient_then_success)
    client.start_heartbeat(interval_s=0.01, failure_timeout_s=0.08)
    time.sleep(0.07)
    client.raise_if_heartbeat_failed()
    assert len(calls) >= 3
    client.stop_heartbeat()

    def unavailable(_):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(client, "_heartbeat_once", unavailable)
    client.start_heartbeat(interval_s=0.01, failure_timeout_s=0.04)
    time.sleep(0.08)
    with pytest.raises(HeartbeatError, match="lease heartbeat failed"):
        client.raise_if_heartbeat_failed()
    client.close()


def test_root_config_is_valid_and_production_safe():
    settings = load_settings()
    assert (settings.host, settings.port) == ("127.0.0.1", 8008)
    assert settings.driver_kind == settings.camera_kind == "mock"
    assert settings.lease_ttl_s == 15.0
    assert len(settings.safe_home) == 14
    assert settings.image_processing_enabled is False
    assert settings.image_processing_torch_compile is True
    assert settings.camera_enabled_roles == ("left_wrist", "right_wrist", "third_view")
    assert settings.camera_depth_enabled is True


def test_lut_processing_changes_preview_but_preserves_raw_calibration_frame(tmp_path):
    lut = np.zeros((2, 2, 2, 3), dtype=np.float32)
    lut[..., 1] = 1.0
    paths = {}
    for camera_id in ("left_wrist", "right_wrist", "third_view"):
        path = tmp_path / f"{camera_id}.npz"
        np.savez_compressed(path, lut_srgb=lut)
        paths[camera_id] = str(path)
    raw = MockCameraManager()
    original = raw.calibration_frame("left_wrist").rgb.copy()
    processed = ProcessedCameraManager(raw, CpuLutProcessor(paths))
    jpeg, _ = processed.frame_jpeg("left_wrist")
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded[..., 1].mean() > 245
    assert np.array_equal(processed.calibration_frame("left_wrist").rgb, original)
    info = processed.processing_info("left_wrist")
    assert info.enabled and info.backend == "cpu" and info.processing_ms >= 0 and info.frame_number > 0


def test_processed_camera_http_headers_and_raw_payload(tmp_path):
    axis = np.linspace(0.0, 1.0, 2, dtype=np.float32)
    lut = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    paths = {}
    for camera_id in ("left_wrist", "right_wrist", "third_view"):
        path = tmp_path / f"{camera_id}.npz"
        np.savez_compressed(path, lut_srgb=lut)
        paths[camera_id] = str(path)
    cameras = ProcessedCameraManager(MockCameraManager(), CpuLutProcessor(paths))
    app = create_app(cameras=cameras)
    routes = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
    frame = routes["/v1/cameras/{camera_id}/frame"]("left_wrist")
    assert frame.headers["x-image-processing"] == "lut"
    assert frame.headers["x-image-processing-backend"] == "cpu"
    raw = routes["/v1/cameras/{camera_id}/calibration-frame"]("left_wrist")
    decoded = decode_calibration_frame(raw.body)
    assert decoded.rgb.shape == (480, 640, 3)


def test_processed_rgb_only_camera_does_not_require_depth(tmp_path):
    axis = np.linspace(0.0, 1.0, 2, dtype=np.float32)
    lut = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    paths = {}
    for camera_id in ("left_wrist", "right_wrist", "third_view"):
        path = tmp_path / f"{camera_id}.npz"
        np.savez_compressed(path, lut_srgb=lut)
        paths[camera_id] = str(path)

    class RgbOnlyCamera:
        fps = 15

        def rgb_frame(self, camera_id):
            return np.full((72, 128, 3), 127, dtype=np.uint8), 12.5, 7

        def calibration_frame(self, camera_id):
            raise RuntimeError("calibration-frame requires camera.depth_enabled=true")

        def set_color_controls(self, camera_id, *, exposure, gain):
            return {"exposure": exposure, "gain": gain}

    cameras = ProcessedCameraManager(RgbOnlyCamera(), CpuLutProcessor(paths))
    app = create_app(cameras=cameras)
    routes = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
    frame = routes["/v1/cameras/{camera_id}/frame"]("third_view")
    assert frame.status_code == 200 and frame.headers["x-frame-number"] == "7"
    with pytest.raises(HTTPException) as exc_info:
        routes["/v1/cameras/{camera_id}/calibration-frame"]("third_view")
    assert exc_info.value.status_code == 409


def test_physical_adapter_has_no_torque_releasing_or_controller_reset_calls():
    assert forbidden_sdk_calls() == []


def test_piper_readonly_connect_diagnostics_and_close_never_disable(monkeypatch):
    import pyAgxArm

    class Message:
        def __init__(self, msg):
            self.msg, self.hz = msg, 50.0

    class FakeArm:
        OPTIONS = type("Options", (), {})

        def __init__(self, channel):
            self.channel = channel
            self.calls = []

        def connect(self): self.calls.append("connect")
        def disconnect(self): self.calls.append("disconnect")
        def disable(self): raise AssertionError("disable must never be called")
        def reset(self): raise AssertionError("reset must never be called")
        def electronic_emergency_stop(self): raise AssertionError("SDK e-stop must never be called")
        def is_connected(self): return True
        def is_ok(self): return True
        def set_joint_limits_enabled(self, enabled): self.calls.append(("joint_limits", enabled))
        def get_channel(self): return self.channel
        def get_joint_angles(self): return Message([0.1] * 6)
        def get_joints_enable_status_list(self): return [True] * 6
        def get_arm_status(self): return Message("normal")

    arms = []

    def create_arm(config):
        arm = FakeArm(config["comm"]["can"]["channel"])
        arms.append(arm)
        return arm

    monkeypatch.setattr(pyAgxArm.AgxArmFactory, "create_arm", create_arm)
    driver = PiperXDualDriver(PiperXConfig("left-test", "right-test", "v188", interface="virtual"))
    driver.connect()
    diagnostics = driver.readonly_diagnostics()
    assert diagnostics["left"]["joint_enabled"] == [True] * 6
    assert diagnostics["right"]["joint_angles_rad"] == [0.1] * 6
    driver.close()
    assert [arm.calls for arm in arms] == [["connect", "disconnect"], ["connect", "disconnect"]]


def test_first_motion_backend_requires_explicit_arm_and_allows_only_home(monkeypatch):
    import pyAgxArm

    class Message:
        def __init__(self, msg):
            self.msg, self.hz = msg, 50.0

    class FakeArm:
        def __init__(self, channel):
            self.channel = channel
            self.calls = []
            self.enabled = False

        def connect(self): self.calls.append("connect")
        def disconnect(self): self.calls.append("disconnect")
        def is_connected(self): return True
        def is_ok(self): return True
        def set_joint_limits_enabled(self, enabled): self.calls.append(("joint_limits", enabled))
        def get_channel(self): return self.channel
        def get_joint_angles(self): return Message([0.0] * 6)
        def get_joints_enable_status_list(self): return [self.enabled] * 6
        def get_arm_status(self): return Message("normal")
        def enable(self): self.calls.append("enable"); self.enabled = True; return True
        def set_speed_percent(self, value): self.calls.append(("speed", value))
        def move_j(self, joints): self.calls.append(("move_j", joints))
        def disable(self): raise AssertionError("disable must never be called")
        def reset(self): raise AssertionError("reset must never be called")
        def electronic_emergency_stop(self): raise AssertionError("SDK e-stop must never be called")

    arms = []

    def create_arm(config):
        arm = FakeArm(config["comm"]["can"]["channel"])
        arms.append(arm)
        return arm

    monkeypatch.setattr(pyAgxArm.AgxArmFactory, "create_arm", create_arm)
    config = PiperXConfig(
        "left-test", "right-test", "v188", interface="virtual", motion_speed_percent=5,
        joint_limits_rad=((-1.0, 1.0),) * 6,
    )
    driver = PiperXMotionDriver(config)
    guardian = MotionGuardian(driver, safe_home=[0.0] * 14, cadence_scale=0.01)
    try:
        token, _ = guardian.acquire("first-motion")
        with pytest.raises(Exception, match="only explicit Go Home"):
            guardian.execute(token, [[0.01] * 6 + [0.0] + [0.0] * 7], "calibration_slow")
        with pytest.raises(Exception, match="not armed"):
            guardian.go_home(token)
        guardian.arm_motion(token)
        execution, _ = guardian.go_home(token)
        assert _wait(guardian, execution)["state"] == "completed"
        for arm in arms:
            assert "enable" in arm.calls
            assert ("speed", 5) in arm.calls
            assert ("move_j", [0.0] * 6) in arm.calls
    finally:
        guardian.close()


def test_piper_state_exposes_finite_difference_velocity_and_fk():
    class Message:
        def __init__(self, msg):
            self.msg = msg

    class FakeArm:
        def __init__(self, joints):
            self.joints = joints
        def is_connected(self): return True
        def is_ok(self): return True
        def get_joint_angles(self): return Message(self.joints.copy())
        def fk(self, joints): return [joints[0], joints[1], 0.25, 0.0, 0.0, 0.0]

    driver = PiperXDualDriver(PiperXConfig("left", "right", "v188"))
    driver._left = FakeArm([0.0] * 6)
    driver._right = FakeArm([0.0] * 6)
    driver.read_qpos()
    with pytest.raises(RuntimeError, match="two feedback samples"):
        driver.read_dq()
    time.sleep(0.002)
    driver._left.joints[0] = 0.001
    driver.read_qpos()
    assert driver.read_dq()[0] > 0.0
    kinematics = driver.read_kinematics()
    assert kinematics["left_T_base_ee"][0][3] == pytest.approx(0.001)
    assert kinematics["right_T_base_ee"][2][3] == pytest.approx(0.25)


def test_piper_gripper_normalized_mapping_uses_configured_metric_width():
    class Message:
        def __init__(self, value):
            self.msg = value

    class GripperMessage:
        def __init__(self, value):
            self.value = value

    class FakeGripper:
        def __init__(self, width):
            self.width = width
            self.commands = []
        def get_gripper_status(self): return Message(GripperMessage(self.width))
        def move_gripper_m(self, width, force): self.commands.append((width, force)); self.width = width

    class FakeArm:
        def get_joint_angles(self): return Message([0.0] * 6)
        def move_j(self, _): pass

    driver = PiperXMotionDriver(PiperXConfig(
        "left", "right", "v188", joint_limits_rad=((-1.0, 1.0),) * 6,
        gripper_open_width_m=0.07, gripper_force_n=1.5,
    ))
    driver._left = FakeArm()
    driver._right = FakeArm()
    driver._left_gripper = FakeGripper(0.035)
    driver._right_gripper = FakeGripper(0.0)
    driver._armed = True
    qpos = driver.read_qpos()
    assert qpos[6] == pytest.approx(0.5)
    assert qpos[13] == pytest.approx(0.0)
    target = [0.0] * 14
    target[6], target[13] = 1.0, 0.25
    driver.command_qpos(target)
    assert driver._left_gripper.commands[-1] == pytest.approx((0.07, 1.5))
    assert driver._right_gripper.commands[-1] == pytest.approx((0.0175, 1.5))


def test_first_motion_micro_mode_rejects_dual_arm_and_large_targets(monkeypatch):
    import pyAgxArm

    class Message:
        def __init__(self, msg):
            self.msg, self.hz = msg, 50.0

    class FakeArm:
        def __init__(self, channel): self.channel, self.enabled = channel, False
        def connect(self): pass
        def disconnect(self): pass
        def is_connected(self): return True
        def is_ok(self): return True
        def set_joint_limits_enabled(self, _enabled): pass
        def get_channel(self): return self.channel
        def get_joint_angles(self): return Message([0.0] * 6)
        def get_joints_enable_status_list(self): return [self.enabled] * 6
        def get_arm_status(self): return Message("normal")
        def enable(self): self.enabled = True; return True
        def set_speed_percent(self, _): pass
        def move_j(self, _): pass

    monkeypatch.setattr(pyAgxArm.AgxArmFactory, "create_arm", lambda config: FakeArm(config["comm"]["can"]["channel"]))
    driver = PiperXMotionDriver(PiperXConfig(
        "left-test", "right-test", "v188", interface="virtual", motion_speed_percent=5,
        joint_limits_rad=((-1.0, 1.0),) * 6, first_motion_max_joint_delta_rad=0.03,
    ))
    guardian = MotionGuardian(driver, safe_home=[0.0] * 14, cadence_scale=0.01)
    try:
        token, _ = guardian.acquire("micro-motion")
        guardian.arm_motion(token)
        good = [0.02] + [0.0] * 13
        execution, _ = guardian.execute(token, [good, [0.0] * 14], "calibration_slow")
        assert _wait(guardian, execution)["state"] == "completed"
        dual = [0.02] + [0.0] * 6 + [0.02] + [0.0] * 6
        with pytest.raises(Exception, match="one arm at a time"):
            guardian.execute(token, [dual], "calibration_slow")
        two_left_joints = [0.02, 0.02] + [0.0] * 12
        with pytest.raises(Exception, match="one joint at a time"):
            guardian.execute(token, [two_left_joints], "calibration_slow")
        too_far = [0.04] + [0.0] * 13
        with pytest.raises(Exception, match="first-motion limit"):
            guardian.execute(token, [too_far], "calibration_slow")
        with pytest.raises(Exception, match="not permitted"):
            guardian.execute(token, [good], "vla_conservative")
    finally:
        guardian.close()
