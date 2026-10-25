from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .camera import CalibrationFrame, CameraManager
from .schemas import CAMERA_IDS


class RgbProcessor(Protocol):
    backend: str

    def apply(self, camera_id: str, rgb: np.ndarray) -> np.ndarray: ...

    def apply_batch(self, camera_ids: tuple[str, ...], rgbs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]: ...


def _load_luts(paths: dict[str, str]) -> dict[str, np.ndarray]:
    if set(paths) != set(CAMERA_IDS):
        raise ValueError("image_processing.lut_by_role must configure all three camera roles")
    luts: dict[str, np.ndarray] = {}
    for camera_id, raw_path in paths.items():
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise ValueError(f"LUT file does not exist for {camera_id}: {path}")
        with np.load(path, allow_pickle=False) as data:
            if "lut_srgb" not in data:
                raise ValueError(f"LUT file has no lut_srgb array: {path}")
            lut = np.asarray(data["lut_srgb"], dtype=np.float32)
        if lut.ndim != 4 or lut.shape[-1] != 3 or len(set(lut.shape[:3])) != 1:
            raise ValueError(f"expected cubic [N,N,N,3] LUT for {camera_id}, got {lut.shape}")
        if lut.shape[0] < 2 or not np.isfinite(lut).all():
            raise ValueError(f"invalid LUT values for {camera_id}: {path}")
        luts[camera_id] = np.clip(lut, 0.0, 1.0)
    return luts


def _trilinear_numpy(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32) * ((lut.shape[0] - 1) / 255.0)
    lower = np.floor(values).astype(np.intp)
    upper = np.minimum(lower + 1, lut.shape[0] - 1)
    fraction = values - lower
    r0, g0, b0 = (lower[..., index] for index in range(3))
    r1, g1, b1 = (upper[..., index] for index in range(3))
    fr, fg, fb = (fraction[..., index:index + 1] for index in range(3))
    c00 = lut[r0, g0, b0] * (1.0 - fr) + lut[r1, g0, b0] * fr
    c01 = lut[r0, g0, b1] * (1.0 - fr) + lut[r1, g0, b1] * fr
    c10 = lut[r0, g1, b0] * (1.0 - fr) + lut[r1, g1, b0] * fr
    c11 = lut[r0, g1, b1] * (1.0 - fr) + lut[r1, g1, b1] * fr
    c0 = c00 * (1.0 - fg) + c10 * fg
    c1 = c01 * (1.0 - fg) + c11 * fg
    output = c0 * (1.0 - fb) + c1 * fb
    return np.rint(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)


def _expand_lut_uint8(lut: np.ndarray, *, size: int = 65) -> tuple[np.ndarray, np.ndarray]:
    """Expand a small float LUT once so CPU frame processing is direct indexing."""
    axis = np.rint(np.linspace(0, 255, size)).astype(np.uint8)
    expanded = np.empty((size, size, size, 3), dtype=np.uint8)
    green, blue = np.meshgrid(axis, axis, indexing="ij")
    slab = np.empty((size, size, 3), dtype=np.uint8)
    slab[..., 1] = green
    slab[..., 2] = blue
    for index, red in enumerate(axis):
        slab[..., 0] = red
        expanded[index] = _trilinear_numpy(slab, lut)
    input_index = np.rint(np.arange(256, dtype=np.float32) * ((size - 1) / 255.0)).astype(np.uint8)
    return expanded, input_index


class CpuLutProcessor:
    backend = "cpu"

    def __init__(self, lut_by_role: dict[str, str]):
        self._luts = {
            camera_id: _expand_lut_uint8(lut)
            for camera_id, lut in _load_luts(lut_by_role).items()
        }
        self._executor = ThreadPoolExecutor(max_workers=len(self._luts), thread_name_prefix="lut-cpu")

    def apply(self, camera_id: str, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb, dtype=np.uint8)
        lut, input_index = self._luts[camera_id]
        return lut[
            input_index[image[..., 0]], input_index[image[..., 1]], input_index[image[..., 2]],
        ]

    def apply_batch(self, camera_ids, rgbs):
        return tuple(self._executor.map(self.apply, camera_ids, rgbs))

    def close(self) -> None:
        self._executor.shutdown(wait=True)


class TorchLutProcessor:
    backend = "cuda"

    def __init__(
        self,
        lut_by_role: dict[str, str],
        *,
        device: str = "cuda:0",
        pinned_memory: bool = False,
        torch_compile: bool = True,
        input_shape: tuple[int, int, int] | None = None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA LUT processing requires torch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA LUT processing requested but torch.cuda.is_available() is false")
        self._torch = torch
        self.device = torch.device(device)
        self._luts = {
            camera_id: torch.from_numpy(lut).to(self.device)
            for camera_id, lut in _load_luts(lut_by_role).items()
        }
        self.pinned_memory = pinned_memory
        self._parallel_streams = {
            camera_id: torch.cuda.Stream(device=self.device) for camera_id in self._luts
        }
        self._parallel_buffers: dict[str, tuple[object, object, object]] = {}
        self._lock = threading.Lock()
        self._compiled_kernel = None
        if torch_compile:
            if input_shape is None or len(input_shape) != 3 or input_shape[2] != 3:
                raise ValueError("torch_compile requires input_shape=(height, width, 3)")
            try:
                self._compiled_kernel = torch.compile(
                    self._apply_device_lut,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                warmup_input = torch.zeros(tuple(input_shape), dtype=torch.uint8, device=self.device)
                self._compiled_kernel(warmup_input, self._luts[next(iter(self._luts))])
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"torch.compile LUT warmup failed for {input_shape}: {type(exc).__name__}: {exc}"
                ) from exc

    def apply(self, camera_id: str, rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        if self.pinned_memory:
            return self.apply_batch((camera_id,), (rgb,))[0]
        with self._lock, torch.inference_mode():
            values = torch.as_tensor(np.ascontiguousarray(rgb), device=self.device, dtype=torch.uint8)
            kernel = self._compiled_kernel or self._apply_device_lut
            return kernel(values, self._luts[camera_id]).cpu().numpy()

    def apply_batch(self, camera_ids, rgbs):
        """Process independent camera frames on separate CUDA streams.

        A stacked 3D-LUT tensor was benchmarked slower for this workload. Each
        camera has its own staging buffers and stream, allowing transfers and
        LUT kernels to overlap without forcing a batch-indexed LUT gather.
        """
        torch = self._torch
        if not camera_ids or len(camera_ids) != len(rgbs):
            raise ValueError("camera_ids and rgbs must be non-empty and have equal length")
        images = tuple(np.ascontiguousarray(rgb, dtype=np.uint8) for rgb in rgbs)
        shape = images[0].shape
        if any(image.shape != shape for image in images):
            raise ValueError("batched RGB images must have identical shapes")
        if not self.pinned_memory:
            return tuple(self.apply(camera_id, image) for camera_id, image in zip(camera_ids, images))
        with self._lock, torch.inference_mode():
            for camera_id, image in zip(camera_ids, images):
                buffers = self._parallel_buffers.get(camera_id)
                if buffers is None or tuple(buffers[0].shape) != shape:
                    buffers = (
                        torch.empty(shape, dtype=torch.uint8, pin_memory=True),
                        torch.empty(shape, dtype=torch.uint8, pin_memory=True),
                        torch.empty(shape, dtype=torch.uint8, device=self.device),
                    )
                    self._parallel_buffers[camera_id] = buffers
                np.copyto(buffers[0].numpy(), image)
            for camera_id in camera_ids:
                host_input, host_output, device_input = self._parallel_buffers[camera_id]
                stream = self._parallel_streams[camera_id]
                with torch.cuda.stream(stream):
                    device_input.copy_(host_input, non_blocking=True)
                    lut = self._luts[camera_id]
                    kernel = self._compiled_kernel or self._apply_device_lut
                    output = kernel(device_input, lut)
                    host_output.copy_(output, non_blocking=True)
            for camera_id in camera_ids:
                stream = self._parallel_streams[camera_id]
                stream.synchronize()
            return tuple(self._parallel_buffers[camera_id][1].numpy() for camera_id in camera_ids)

    def _apply_device_lut(self, device_input, lut):
        torch = self._torch
        values = device_input.to(torch.float32)
        values.mul_((lut.shape[0] - 1) / 255.0)
        lower = torch.floor(values).to(torch.long)
        upper = torch.clamp(lower + 1, max=lut.shape[0] - 1)
        fraction = values - lower
        r0, g0, b0 = lower.unbind(dim=-1)
        r1, g1, b1 = upper.unbind(dim=-1)
        fr, fg, fb = (fraction[..., index].unsqueeze(-1) for index in range(3))
        c00 = lut[r0, g0, b0] * (1.0 - fr) + lut[r1, g0, b0] * fr
        c01 = lut[r0, g0, b1] * (1.0 - fr) + lut[r1, g0, b1] * fr
        c10 = lut[r0, g1, b0] * (1.0 - fr) + lut[r1, g1, b0] * fr
        c11 = lut[r0, g1, b1] * (1.0 - fr) + lut[r1, g1, b1] * fr
        return (((c00 * (1.0 - fg) + c10 * fg) * (1.0 - fb)
                 + (c01 * (1.0 - fg) + c11 * fg) * fb)
                .mul(255.0).round().clamp_(0, 255).to(torch.uint8))


@dataclass(frozen=True)
class ProcessingInfo:
    enabled: bool
    backend: str
    processing_ms: float
    frame_number: int


class ProcessedCameraManager:
    """Apply display/model RGB processing while preserving raw calibration frames."""

    def __init__(self, source: CameraManager, processor: RgbProcessor, *, jpeg_quality: int = 90):
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("image_processing.jpeg_quality must be in [1, 100]")
        self.source = source
        self.processor = processor
        self.jpeg_quality = jpeg_quality
        self.fps = int(getattr(source, "fps", 30))
        self._info = {
            camera_id: ProcessingInfo(True, processor.backend, 0.0, 0)
            for camera_id in CAMERA_IDS
        }
        self._info_lock = threading.Lock()
        self._frame_locks = {camera_id: threading.Lock() for camera_id in CAMERA_IDS}
        self._jpeg_cache: dict[str, tuple[int, bytes, float]] = {}
        self._jpeg_executor = ThreadPoolExecutor(max_workers=len(CAMERA_IDS), thread_name_prefix="jpeg")

    def frame_jpeg(self, camera_id: str) -> tuple[bytes, float]:
        return self.frame_jpegs((camera_id,))[camera_id]

    def frame_jpegs(self, camera_ids: tuple[str, ...] = CAMERA_IDS) -> dict[str, tuple[bytes, float]]:
        samples = {camera_id: self.source.rgb_frame(camera_id) for camera_id in camera_ids}
        uncached = []
        result = {}
        for camera_id, (rgb, timestamp, frame_number) in samples.items():
            with self._frame_locks[camera_id]:
                cached = self._jpeg_cache.get(camera_id)
                if cached is not None and cached[0] == frame_number:
                    result[camera_id] = (cached[1], cached[2])
                else:
                    uncached.append((camera_id, rgb, timestamp, frame_number))
        if not uncached:
            return result
        started = time.perf_counter()
        processed = self.processor.apply_batch(
            tuple(item[0] for item in uncached), tuple(item[1] for item in uncached),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        def encode(item):
            camera_id, timestamp, frame_number, image = item
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise RuntimeError(f"failed to encode processed frame for {camera_id}")
            return camera_id, encoded.tobytes(), timestamp, frame_number

        encoded_items = list(self._jpeg_executor.map(
            encode,
            ((item[0], item[2], item[3], image) for item, image in zip(uncached, processed)),
        ))
        for camera_id, jpeg, timestamp, frame_number in encoded_items:
            with self._frame_locks[camera_id]:
                self._jpeg_cache[camera_id] = (frame_number, jpeg, timestamp)
            with self._info_lock:
                self._info[camera_id] = ProcessingInfo(
                    True, self.processor.backend, elapsed_ms, frame_number,
                )
            result[camera_id] = (jpeg, timestamp)
        return result

    def calibration_frame(self, camera_id: str) -> CalibrationFrame:
        return self.source.calibration_frame(camera_id)

    def set_color_controls(self, camera_id: str, *, exposure: float, gain: float) -> dict[str, float | bool]:
        return self.source.set_color_controls(camera_id, exposure=exposure, gain=gain)

    def processing_info(self, camera_id: str) -> ProcessingInfo:
        with self._info_lock:
            return self._info[camera_id]

    def close(self) -> None:
        self._jpeg_executor.shutdown(wait=True)
        close_processor = getattr(self.processor, "close", None)
        if callable(close_processor):
            close_processor()
        close = getattr(self.source, "close", None)
        if callable(close):
            close()


def create_lut_processor(
    backend: str,
    lut_by_role: dict[str, str],
    *,
    device: str,
    pinned_memory: bool = False,
    torch_compile: bool = True,
    input_shape: tuple[int, int, int] | None = None,
) -> RgbProcessor:
    if backend == "cpu":
        return CpuLutProcessor(lut_by_role)
    if backend == "cuda":
        return TorchLutProcessor(
            lut_by_role, device=device, pinned_memory=pinned_memory,
            torch_compile=torch_compile, input_shape=input_shape,
        )
    if backend == "auto":
        try:
            return TorchLutProcessor(
                lut_by_role, device=device, pinned_memory=pinned_memory,
                torch_compile=torch_compile, input_shape=input_shape,
            )
        except RuntimeError:
            return CpuLutProcessor(lut_by_role)
    raise ValueError(f"unsupported image-processing backend: {backend}")
