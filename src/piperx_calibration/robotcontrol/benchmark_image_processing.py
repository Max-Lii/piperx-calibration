from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2

from .image_processing import create_lut_processor


def benchmark(
    backend: str,
    images: dict[str, Path],
    luts: dict[str, str],
    *,
    device: str,
    pinned_memory: bool,
    warmup: int,
    runs: int,
) -> dict:
    loaded = {}
    for camera_id, path in images.items():
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to read image: {path}")
        loaded[camera_id] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    processor = create_lut_processor(
        backend, luts, device=device, pinned_memory=pinned_memory,
        torch_compile=True,
        input_shape=next(iter(loaded.values())).shape,
    )
    for _ in range(warmup):
        for camera_id, image in loaded.items():
            processor.apply(camera_id, image)
    samples = {camera_id: [] for camera_id in loaded}
    pipeline_samples = {camera_id: [] for camera_id in loaded}
    jpeg_samples = {camera_id: [] for camera_id in loaded}
    for _ in range(runs):
        for camera_id, image in loaded.items():
            started = time.perf_counter()
            output = processor.apply(camera_id, image)
            processed = time.perf_counter()
            ok, _ = cv2.imencode(".jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            finished = time.perf_counter()
            if not ok:
                raise RuntimeError(f"JPEG encode failed for {camera_id}")
            samples[camera_id].append((processed - started) * 1000.0)
            jpeg_samples[camera_id].append((finished - processed) * 1000.0)
            pipeline_samples[camera_id].append((finished - started) * 1000.0)
    def summary(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": sorted(values)[max(0, int(0.95 * len(values)) - 1)],
            "max": max(values),
        }
    return {
        "requested_backend": backend,
        "actual_backend": processor.backend,
        "device": device if processor.backend == "cuda" else "cpu",
        "pinned_memory": pinned_memory if processor.backend == "cuda" else False,
        "resolution": {key: [int(value.shape[1]), int(value.shape[0])] for key, value in loaded.items()},
        "runs": runs,
        "lut_latency_ms": {key: summary(values) for key, values in samples.items()},
        "jpeg_latency_ms": {key: summary(values) for key, values in jpeg_samples.items()},
        "lut_plus_jpeg_latency_ms": {key: summary(values) for key, values in pipeline_samples.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark robotcontrol per-camera 3D LUT processing")
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs <= 0:
        parser.error("--warmup must be nonnegative and --runs must be positive")
    camera_ids = ("left_wrist", "right_wrist", "third_view")
    luts = {camera_id: str(args.asset_root / f"{camera_id}.npz") for camera_id in camera_ids}
    images = {camera_id: args.image_root / f"{camera_id}_1.png" for camera_id in camera_ids}
    print(json.dumps(benchmark(
        args.backend, images, luts, device=args.device, pinned_memory=args.pinned_memory,
        warmup=args.warmup, runs=args.runs,
    ), indent=2))


if __name__ == "__main__":
    main()
