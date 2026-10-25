from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from .schemas import CAMERA_IDS


@dataclass(frozen=True)
class CalibrationFrame:
    """One color-aligned RGB/depth sample in the color-camera pixel frame."""

    camera_id: str
    serial: str
    frame_number: int
    host_timestamp_monotonic_s: float
    hardware_timestamp_ms: float | None
    rgb: np.ndarray
    depth_raw: np.ndarray
    depth_scale_m: float
    color_intrinsics: np.ndarray
    distortion: np.ndarray
    distortion_model: str
    depth_aligned_to_color: bool = True
    camera_controls: dict[str, float | bool] = field(default_factory=dict)


def encode_calibration_frame(frame: CalibrationFrame) -> bytes:
    metadata = {
        "schema_version": 1,
        "camera_id": frame.camera_id,
        "serial": frame.serial,
        "frame_number": frame.frame_number,
        "host_timestamp_monotonic_s": frame.host_timestamp_monotonic_s,
        "hardware_timestamp_ms": frame.hardware_timestamp_ms,
        "depth_scale_m": frame.depth_scale_m,
        "distortion_model": frame.distortion_model,
        "depth_aligned_to_color": frame.depth_aligned_to_color,
        "camera_controls": frame.camera_controls,
        "rgb_color_order": "RGB",
    }
    output = io.BytesIO()
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata)),
        rgb=np.asarray(frame.rgb, dtype=np.uint8),
        depth_raw=np.asarray(frame.depth_raw, dtype=np.uint16),
        color_intrinsics=np.asarray(frame.color_intrinsics, dtype=np.float64),
        distortion=np.asarray(frame.distortion, dtype=np.float64),
    )
    return output.getvalue()


def decode_calibration_frame(payload: bytes) -> CalibrationFrame:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        return CalibrationFrame(
            camera_id=metadata["camera_id"],
            serial=metadata["serial"],
            frame_number=int(metadata["frame_number"]),
            host_timestamp_monotonic_s=float(metadata["host_timestamp_monotonic_s"]),
            hardware_timestamp_ms=metadata["hardware_timestamp_ms"],
            rgb=data["rgb"].copy(),
            depth_raw=data["depth_raw"].copy(),
            depth_scale_m=float(metadata["depth_scale_m"]),
            color_intrinsics=data["color_intrinsics"].copy(),
            distortion=data["distortion"].copy(),
            distortion_model=metadata["distortion_model"],
            depth_aligned_to_color=bool(metadata["depth_aligned_to_color"]),
            camera_controls=dict(metadata.get("camera_controls", {})),
        )


class CameraManager(Protocol):
    """Latest-frame interface shared by single-frame and MJPEG endpoints."""

    def rgb_frame(self, camera_id: str) -> tuple[np.ndarray, float, int]: ...

    def frame_jpeg(self, camera_id: str) -> tuple[bytes, float]: ...

    def frame_jpegs(self, camera_ids: tuple[str, ...] = CAMERA_IDS) -> dict[str, tuple[bytes, float]]: ...

    def calibration_frame(self, camera_id: str) -> CalibrationFrame: ...

    def set_color_controls(self, camera_id: str, *, exposure: float, gain: float) -> dict[str, float | bool]: ...


class MockCameraManager:
    """Offline RGB source; future RealSense workers keep this public contract."""

    def __init__(self) -> None:
        self._frame_number = {camera_id: 0 for camera_id in CAMERA_IDS}
        self._lock = threading.Lock()

    @staticmethod
    def _rgb(camera_id: str) -> np.ndarray:
        if camera_id not in CAMERA_IDS:
            raise KeyError(camera_id)
        image = np.empty((480, 640, 3), dtype=np.uint8)
        image[:] = {"left_wrist": (180, 80, 40), "right_wrist": (40, 80, 180), "third_view": (60, 160, 60)}[camera_id]
        cv2.putText(image, f"MOCK {camera_id}", (32, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image

    def calibration_frame(self, camera_id: str) -> CalibrationFrame:
        rgb = self._rgb(camera_id)
        with self._lock:
            self._frame_number[camera_id] += 1
            frame_number = self._frame_number[camera_id]
        return CalibrationFrame(
            camera_id=camera_id,
            serial=f"MOCK-{camera_id}",
            frame_number=frame_number,
            host_timestamp_monotonic_s=time.monotonic(),
            hardware_timestamp_ms=float(frame_number * 33.333),
            rgb=rgb,
            depth_raw=np.full(rgb.shape[:2], 1000, dtype=np.uint16),
            depth_scale_m=0.001,
            color_intrinsics=np.asarray([[600.0, 0.0, 319.5], [0.0, 600.0, 239.5], [0.0, 0.0, 1.0]]),
            distortion=np.zeros(5),
            distortion_model="none",
        )

    def frame_jpeg(self, camera_id: str) -> tuple[bytes, float]:
        rgb, timestamp, _ = self.rgb_frame(camera_id)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("failed to encode mock frame")
        return encoded.tobytes(), timestamp

    def frame_jpegs(self, camera_ids: tuple[str, ...] = CAMERA_IDS) -> dict[str, tuple[bytes, float]]:
        return {camera_id: self.frame_jpeg(camera_id) for camera_id in camera_ids}

    def rgb_frame(self, camera_id: str) -> tuple[np.ndarray, float, int]:
        frame = self.calibration_frame(camera_id)
        return frame.rgb, frame.host_timestamp_monotonic_s, frame.frame_number

    def set_color_controls(self, camera_id: str, *, exposure: float, gain: float) -> dict[str, float | bool]:
        if camera_id not in CAMERA_IDS:
            raise KeyError(camera_id)
        return {"exposure": exposure, "gain": gain, "auto_exposure": False}


class RealSenseCameraManager:
    """Own one background RealSense pipeline per configured camera role.

    The manager is intentionally the only place that imports ``pyrealsense2``.
    Pipelines are started when the robotcontrol service starts, and callers
    only see the latest complete color/depth sample under a per-camera lock.
    """

    def __init__(
        self,
        serial_by_camera: dict[str, str],
        *,
        enabled_camera_ids: tuple[str, ...] = CAMERA_IDS,
        depth_enabled: bool = True,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 15,
        lock_auto_controls_after_warmup: bool = False,
        manual_exposure: float | None = None,
    ):
        if not enabled_camera_ids or not set(enabled_camera_ids).issubset(set(CAMERA_IDS)):
            raise ValueError("enabled_camera_ids must contain known camera roles")
        if not set(enabled_camera_ids).issubset(serial_by_camera):
            raise ValueError("a D435 serial must be configured for every enabled camera role")
        if width <= 0 or height <= 0 or fps <= 0 or warmup_frames < 0:
            raise ValueError("RealSense dimensions, fps and warmup_frames must be positive")
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("camera.kind=realsense requires pyrealsense2") from exc

        self._rs = rs
        self.serial_by_camera = dict(serial_by_camera)
        self.enabled_camera_ids = tuple(enabled_camera_ids)
        self.depth_enabled = depth_enabled
        self.width = width
        self.height = height
        self.fps = fps
        if manual_exposure is not None and manual_exposure <= 0:
            raise ValueError("manual_exposure must be positive")
        self.manual_exposure = manual_exposure
        self._frames: dict[str, CalibrationFrame] = {}
        self._rgb_frames: dict[str, tuple[np.ndarray, float, int]] = {}
        self._locks = {camera_id: threading.Lock() for camera_id in CAMERA_IDS}
        self._pipelines: dict[str, object] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop = threading.Event()
        self._errors: dict[str, str] = {}
        self._camera_controls: dict[str, dict[str, float | bool]] = {}

        try:
            for camera_id in self.enabled_camera_ids:
                try:
                    pipeline = self._start_pipeline(camera_id)
                    self._pipelines[camera_id] = pipeline
                    if self.manual_exposure is not None:
                        self._apply_manual_exposure(pipeline)
                    for _ in range(warmup_frames):
                        pipeline.wait_for_frames()
                    self._camera_controls[camera_id] = self._configure_color_controls(
                        pipeline, lock=lock_auto_controls_after_warmup,
                    )
                except Exception as exc:
                    raise RuntimeError(f"failed to initialize RealSense camera {camera_id}: {exc}") from exc
                thread = threading.Thread(
                    target=self._capture_loop,
                    args=(camera_id, pipeline),
                    name=f"realsense-{camera_id}",
                    daemon=True,
                )
                self._threads[camera_id] = thread
                thread.start()
        except Exception:
            self.close()
            raise

    def _start_pipeline(self, camera_id: str):
        rs = self._rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_by_camera[camera_id])
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        if self.depth_enabled:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        pipeline.start(config)
        return pipeline

    def _apply_manual_exposure(self, pipeline: object) -> None:
        """Disable RGB auto exposure before warmup frames are captured."""
        rs = self._rs
        sensors = pipeline.get_active_profile().get_device().query_sensors()
        color = next((sensor for sensor in sensors if sensor.supports(rs.option.white_balance)), None)
        if color is None or not color.supports(rs.option.exposure):
            raise RuntimeError("RGB sensor does not support manual exposure")
        if color.supports(rs.option.enable_auto_exposure):
            color.set_option(rs.option.enable_auto_exposure, 0.0)
        color.set_option(rs.option.exposure, float(self.manual_exposure))

    def _configure_color_controls(self, pipeline: object, *, lock: bool) -> dict[str, float | bool]:
        rs = self._rs
        sensors = pipeline.get_active_profile().get_device().query_sensors()
        color = next((sensor for sensor in sensors if sensor.supports(rs.option.white_balance)), None)
        if color is None:
            raise RuntimeError("RealSense device has no RGB sensor with white-balance control")

        options = {
            "exposure": rs.option.exposure,
            "gain": rs.option.gain,
            "white_balance": rs.option.white_balance,
        }
        values = {
            name: float(color.get_option(option))
            for name, option in options.items()
            if color.supports(option)
        }
        if color.supports(rs.option.auto_exposure_priority):
            # RealSense otherwise lowers the actual frame rate in dim scenes.
            color.set_option(rs.option.auto_exposure_priority, 0.0)
        if lock or self.manual_exposure is not None:
            for option in (rs.option.enable_auto_exposure, rs.option.enable_auto_white_balance):
                if option == rs.option.enable_auto_white_balance and not lock:
                    continue
                if color.supports(option):
                    color.set_option(option, 0.0)
            if self.manual_exposure is not None:
                if not color.supports(rs.option.exposure):
                    raise RuntimeError("RGB sensor does not support manual exposure")
                color.set_option(rs.option.exposure, float(self.manual_exposure))
            for name, option in options.items():
                if name in values and name != "exposure":
                    color.set_option(option, values[name])
            if self.manual_exposure is not None:
                values["exposure"] = float(color.get_option(rs.option.exposure))
        values["auto_exposure"] = bool(
            color.get_option(rs.option.enable_auto_exposure)
        ) if color.supports(rs.option.enable_auto_exposure) else False
        values["auto_white_balance"] = bool(
            color.get_option(rs.option.enable_auto_white_balance)
        ) if color.supports(rs.option.enable_auto_white_balance) else False
        values["auto_exposure_priority"] = bool(
            color.get_option(rs.option.auto_exposure_priority)
        ) if color.supports(rs.option.auto_exposure_priority) else False
        values["locked_after_warmup"] = bool(lock or self.manual_exposure is not None)
        values["exposure_mode"] = "manual" if self.manual_exposure is not None else "warmup_locked" if lock else "auto"
        return values

    def set_color_controls(self, camera_id: str, *, exposure: float, gain: float) -> dict[str, float | bool]:
        if camera_id not in CAMERA_IDS:
            raise KeyError(camera_id)
        rs = self._rs
        pipeline = self._pipelines[camera_id]
        sensors = pipeline.get_active_profile().get_device().query_sensors()
        color = next((sensor for sensor in sensors if sensor.supports(rs.option.exposure)), None)
        if color is None:
            raise RuntimeError(f"{camera_id} has no RGB exposure control")
        if color.supports(rs.option.enable_auto_exposure):
            color.set_option(rs.option.enable_auto_exposure, 0.0)
        color.set_option(rs.option.exposure, float(exposure))
        if color.supports(rs.option.gain):
            color.set_option(rs.option.gain, float(gain))
        controls = dict(self._camera_controls[camera_id])
        controls.update({
            "exposure": float(color.get_option(rs.option.exposure)),
            "gain": float(color.get_option(rs.option.gain)),
            "auto_exposure": False,
            "locked_after_warmup": True,
        })
        self._camera_controls[camera_id] = controls
        return dict(controls)

    def _capture_loop(self, camera_id: str, pipeline: object) -> None:
        rs = self._rs
        align = rs.align(rs.stream.color) if self.depth_enabled else None
        frame_number = 0
        try:
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames()
                if align is not None:
                    frames = align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color or (self.depth_enabled and not depth):
                    continue
                profile = color.get_profile().as_video_stream_profile()
                intrinsics = profile.get_intrinsics()
                frame_number += 1
                if not self.depth_enabled:
                    with self._locks[camera_id]:
                        self._rgb_frames[camera_id] = (
                            np.asanyarray(color.get_data()).copy(),
                            time.monotonic(),
                            frame_number,
                        )
                        self._errors.pop(camera_id, None)
                    continue
                sample = CalibrationFrame(
                    camera_id=camera_id,
                    serial=self.serial_by_camera[camera_id],
                    frame_number=frame_number,
                    host_timestamp_monotonic_s=time.monotonic(),
                    hardware_timestamp_ms=float(color.get_timestamp()),
                    rgb=np.asanyarray(color.get_data()).copy(),
                    depth_raw=np.asanyarray(depth.get_data()).copy(),
                    depth_scale_m=float(pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()),
                    color_intrinsics=np.asarray(
                        [[intrinsics.fx, 0.0, intrinsics.ppx],
                         [0.0, intrinsics.fy, intrinsics.ppy],
                         [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    ),
                    distortion=np.asarray(intrinsics.coeffs, dtype=np.float64),
                    distortion_model=str(intrinsics.model),
                    camera_controls=dict(self._camera_controls[camera_id]),
                )
                with self._locks[camera_id]:
                    self._frames[camera_id] = sample
                    self._errors.pop(camera_id, None)
        except Exception as exc:
            self._errors[camera_id] = f"{type(exc).__name__}: {exc}"

    def _latest(self, camera_id: str) -> CalibrationFrame:
        if camera_id not in CAMERA_IDS:
            raise KeyError(camera_id)
        with self._locks[camera_id]:
            frame = self._frames.get(camera_id)
            error = self._errors.get(camera_id)
            if frame is not None:
                return frame
        detail = f": {error}" if error else ""
        raise RuntimeError(f"no RealSense frame available for {camera_id}{detail}")

    def frame_jpeg(self, camera_id: str) -> tuple[bytes, float]:
        rgb, timestamp, _ = self.rgb_frame(camera_id)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError(f"failed to encode RealSense frame for {camera_id}")
        return encoded.tobytes(), timestamp

    def frame_jpegs(self, camera_ids: tuple[str, ...] = CAMERA_IDS) -> dict[str, tuple[bytes, float]]:
        return {camera_id: self.frame_jpeg(camera_id) for camera_id in camera_ids}

    def rgb_frame(self, camera_id: str) -> tuple[np.ndarray, float, int]:
        if camera_id not in CAMERA_IDS:
            raise KeyError(camera_id)
        with self._locks[camera_id]:
            if self.depth_enabled:
                frame = self._frames.get(camera_id)
                rgb = None if frame is None else frame.rgb
                timestamp = None if frame is None else frame.host_timestamp_monotonic_s
                frame_number = 0 if frame is None else frame.frame_number
            else:
                rgb, timestamp, frame_number = self._rgb_frames.get(camera_id, (None, None, 0))
            error = self._errors.get(camera_id)
        if rgb is None or timestamp is None:
            detail = f": {error}" if error else ""
            raise RuntimeError(f"no RealSense RGB frame available for {camera_id}{detail}")
        return rgb, timestamp, frame_number

    def calibration_frame(self, camera_id: str) -> CalibrationFrame:
        if not self.depth_enabled:
            raise RuntimeError("calibration-frame requires camera.depth_enabled=true")
        return self._latest(camera_id)

    def close(self) -> None:
        self._stop.set()
        for pipeline in self._pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                pass
        for thread in self._threads.values():
            thread.join(timeout=2.0)
        self._threads.clear()
        self._pipelines.clear()
