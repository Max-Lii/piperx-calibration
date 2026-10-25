from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol
from pathlib import Path

import cv2
import numpy as np

from piperx_calibration.robotcontrol.camera import CalibrationFrame, MockCameraManager, encode_calibration_frame
from piperx_calibration.robotcontrol.client import RobotControlClient

from .schemas import DeviceStatus


class WorkcellAdapter(Protocol):
    def system_check(self) -> list[DeviceStatus]: ...
    def frame_jpeg(self, camera_id: str) -> tuple[bytes, float]: ...
    def calibration_frame(self, camera_id: str) -> tuple[CalibrationFrame, bytes]: ...
    def set_camera_controls(self, camera_id: str, exposure: float, gain: float) -> dict: ...
    def hold(self) -> None: ...
    def go_home(self) -> None: ...
    def close(self) -> None: ...


class MockWorkcellAdapter:
    def __init__(self) -> None:
        self.cameras = MockCameraManager()
        self._capture_number = {camera_id: 0 for camera_id in ("left_wrist", "third_view", "right_wrist")}

    @staticmethod
    def _camera_matrix(camera_id: str) -> np.ndarray:
        # Slightly different synthetic factory models ensure the three camera
        # result cards are independently exercised in offline regression.
        focal = {"left_wrist": 603.0, "third_view": 600.0, "right_wrist": 597.0}[camera_id]
        return np.asarray([[focal, 0.0, 319.5], [0.0, focal + 1.5, 239.5], [0.0, 0.0, 1.0]])

    def _synthetic_frame(self, camera_id: str) -> CalibrationFrame:
        self._capture_number[camera_id] += 1
        number = self._capture_number[camera_id]
        rgb = self.cameras.calibration_frame(camera_id).rgb
        return CalibrationFrame(
            camera_id=camera_id, serial=f"MOCK-{camera_id}", frame_number=number,
            host_timestamp_monotonic_s=time.monotonic(), hardware_timestamp_ms=number * 33.333,
            rgb=rgb, depth_raw=np.full(rgb.shape[:2], 1000, dtype=np.uint16), depth_scale_m=0.001,
            color_intrinsics=self._camera_matrix(camera_id), distortion=np.zeros(5), distortion_model="none",
        )

    def system_check(self) -> list[DeviceStatus]:
        return [
            DeviceStatus(device_id="left_arm", label="左机械臂", kind="robot", connected=True, healthy=True, detail="MOCK / HOLD"),
            DeviceStatus(device_id="right_arm", label="右机械臂", kind="robot", connected=True, healthy=True, detail="MOCK / HOLD"),
            DeviceStatus(device_id="left_wrist", label="左腕相机", kind="camera", connected=True, healthy=True, detail="640×480"),
            DeviceStatus(device_id="third_view", label="第三视角相机", kind="camera", connected=True, healthy=True, detail="640×480"),
            DeviceStatus(device_id="right_wrist", label="右腕相机", kind="camera", connected=True, healthy=True, detail="640×480"),
        ]

    def frame_jpeg(self, camera_id: str) -> tuple[bytes, float]:
        frame = self._synthetic_frame(camera_id)
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("failed to encode mock frame")
        return encoded.tobytes(), frame.host_timestamp_monotonic_s

    def calibration_frame(self, camera_id: str) -> tuple[CalibrationFrame, bytes]:
        frame = self._synthetic_frame(camera_id)
        return frame, encode_calibration_frame(frame)

    def set_camera_controls(self, camera_id: str, exposure: float, gain: float) -> dict:
        return {"camera_id": camera_id, "exposure": exposure, "gain": gain, "auto_exposure": False}

    def hold(self) -> None:
        return None

    def go_home(self) -> None:
        return None

    def close(self) -> None:
        return None


class RobotwinPreviewAdapter(MockWorkcellAdapter):
    """Use the standalone RoboTwin renderer for browser preview frames.

    Calibration-frame metadata still comes from the deterministic mock path;
    this adapter is intended for UI inspection and does not claim to be a
    physical camera or a replacement for the robotcontrol adapter.
    """

    def __init__(self, preview_fps: float = 8.0) -> None:
        if not np.isfinite(preview_fps) or preview_fps <= 0:
            raise ValueError("preview_fps must be positive and finite")
        super().__init__()
        import sys
        project_root = str(Path(__file__).resolve().parents[3])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from sim.robotwin.adapter import Workcell
        # SAPIEN resources must be created, rendered and released on one thread.
        # Three MJPEG clients otherwise enter the same scene concurrently.
        self._renderer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robotwin-preview")
        self._lock = threading.Lock()
        self._frames: dict[str, bytes] = {}
        self._timestamp = 0.0
        self._interval = 1.0 / preview_fps
        self._closed = False
        try:
            self.workcell = self._renderer.submit(Workcell).result()
        except BaseException:
            self._renderer.shutdown(wait=True)
            raise

    def _render_frames(self) -> dict[str, bytes]:
        mapping = {"left_wrist": "left_camera", "right_wrist": "right_camera", "third_view": "head_camera"}
        observations = self.workcell.get_obs()["observation"]
        frames = {}
        for camera_id, name in mapping.items():
            rgb = observations[name]["rgb"]
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise RuntimeError("failed to encode RoboTwin preview frame")
            frames[camera_id] = encoded.tobytes()
        return frames

    def frame_jpeg(self, camera_id: str) -> tuple[bytes, float]:
        if camera_id not in {"left_wrist", "right_wrist", "third_view"}:
            raise KeyError(camera_id)
        with self._lock:
            if self._closed:
                raise RuntimeError("RoboTwin preview is closed")
            if not self._frames or time.monotonic() - self._timestamp >= self._interval:
                self._frames = self._renderer.submit(self._render_frames).result()
                self._timestamp = time.monotonic()
            return self._frames[camera_id], self._timestamp

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._renderer.submit(self.workcell.close).result()
            finally:
                self._renderer.shutdown(wait=True)


class HttpWorkcellAdapter:
    """Thin calibration-side adapter; robotcontrol remains the hardware owner."""

    def __init__(self, base_url: str) -> None:
        self.client = RobotControlClient(base_url)

    def system_check(self) -> list[DeviceStatus]:
        try:
            state = self.client.state()
            can_standby = (
                str(state.get("backend", "")).startswith("PiperX")
                and not state.get("can_connected")
                and bool(state.get("holding"))
                and not state.get("lease_active")
            )
            healthy = bool(state.get("healthy")) or can_standby
            robot_connected = True
            robot_detail = "robotcontrol (CAN standby)" if can_standby else "robotcontrol"
        except Exception as exc:
            healthy = False
            robot_connected = False
            robot_detail = f"robotcontrol offline: {type(exc).__name__}"
        devices = [
            DeviceStatus(device_id="left_arm", label="左机械臂", kind="robot", connected=robot_connected, healthy=healthy, detail=robot_detail),
            DeviceStatus(device_id="right_arm", label="右机械臂", kind="robot", connected=robot_connected, healthy=healthy, detail=robot_detail),
        ]
        for camera_id, label in (("left_wrist", "左腕相机"), ("third_view", "第三视角相机"), ("right_wrist", "右腕相机")):
            try:
                self.client.frame_jpeg(camera_id)
                devices.append(DeviceStatus(device_id=camera_id, label=label, kind="camera", connected=True, healthy=True))
            except Exception as exc:
                devices.append(DeviceStatus(device_id=camera_id, label=label, kind="camera", connected=False, healthy=False, detail=type(exc).__name__))
        return devices

    def frame_jpeg(self, camera_id: str) -> tuple[bytes, float]:
        payload, timestamp = self.client.frame_jpeg(camera_id)
        return payload, timestamp or 0.0

    def calibration_frame(self, camera_id: str) -> tuple[CalibrationFrame, bytes]:
        return self.client.calibration_frame(camera_id)

    def set_camera_controls(self, camera_id: str, exposure: float, gain: float) -> dict:
        return self.client.set_camera_controls(camera_id, exposure=exposure, gain=gain)

    def hold(self) -> None:
        self.client.hold()

    def go_home(self) -> None:
        # HOLD is deliberately token-free, but Go Home is a motion and must
        # participate in robotcontrol's exclusive-control lease.  Acquire the
        # lease only for this recovery operation; this also makes a competing
        # controller an explicit, safe failure instead of bypassing it.
        acquired_here = self.client.token is None
        if acquired_here:
            self.client.acquire("piper-workcell-calibrator-go-home")
            self.client.start_heartbeat(interval_s=3.0)
        try:
            # A freshly started physical robotcontrol backend deliberately
            # rejects every motion until the lease owner explicitly arms it.
            self.client.arm_motion()
            started = self.client.go_home()
            finished = self.client.wait_execution(started["execution_id"], timeout_s=60.0)
            if finished["state"] != "completed":
                raise RuntimeError(
                    f"Go Home 未完成：{finished['state']} ({finished.get('reason') or 'unknown'})"
                )
        finally:
            if acquired_here:
                self.client.stop_heartbeat()
                # Releasing a robotcontrol lease places the robot in HOLD.
                self.client.release()

    def close(self) -> None:
        self.client.close()
