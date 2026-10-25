from __future__ import annotations

import asyncio
import io
import json
import time
import zipfile
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse

from .camera import CameraManager, MockCameraManager, encode_calibration_frame
from .driver import MockDualPiperDriver
from .guardian import ControlError, MotionGuardian
from .schemas import (
    AcquireRequest, AcquireResponse, CAMERA_IDS, CameraControlsRequest, ExecuteRequest,
    ExecutionResponse, ExecutionStatusResponse, TokenRequest,
)


def create_app(*, guardian: MotionGuardian | None = None, cameras: CameraManager | None = None) -> FastAPI:
    guardian = guardian or MotionGuardian(MockDualPiperDriver())
    cameras = cameras or MockCameraManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            guardian.close()
            closed: set[int] = set()
            for resource in (cameras, guardian.driver):
                close = getattr(resource, "close", None)
                if callable(close) and id(resource) not in closed:
                    close()
                    closed.add(id(resource))

    app = FastAPI(title="robotcontrol-dev", version="0.1.0", lifespan=lifespan)

    def call(operation):
        try:
            return operation()
        except ControlError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/v1/control/acquire", response_model=AcquireResponse)
    def acquire(request: AcquireRequest):
        token, expires = call(lambda: guardian.acquire(request.owner_id))
        return AcquireResponse(token=token, expires_at_monotonic_s=expires)

    @app.post("/v1/control/alive", response_model=AcquireResponse)
    def alive(request: TokenRequest):
        expires = call(lambda: guardian.alive(request.token))
        return AcquireResponse(token=request.token, expires_at_monotonic_s=expires)

    @app.post("/v1/control/release", status_code=204)
    def release(request: TokenRequest):
        call(lambda: guardian.release(request.token))

    @app.post("/v1/motions/execute_joint_waypoints", response_model=ExecutionResponse)
    def execute(request: ExecuteRequest):
        execution_id, samples = call(lambda: guardian.execute(
            request.token, request.waypoints, request.profile, speed_scale=request.speed_scale,
        ))
        return ExecutionResponse(execution_id=execution_id, planned_samples=samples, state="running")

    @app.post("/v1/motions/arm")
    def arm_motion(request: TokenRequest):
        call(lambda: guardian.arm_motion(request.token))
        return {"armed": True}

    @app.post("/v1/motions/go_home", response_model=ExecutionResponse)
    def go_home(request: TokenRequest):
        execution_id, samples = call(lambda: guardian.go_home(request.token))
        return ExecutionResponse(execution_id=execution_id, planned_samples=samples, state="running")

    @app.get("/v1/executions/{execution_id}", response_model=ExecutionStatusResponse)
    def execution_status(execution_id: str):
        return call(lambda: guardian.execution_status(execution_id))

    @app.post("/v1/motions/hold", status_code=204)
    def hold():
        guardian.hold(reason="api_request")

    @app.get("/v1/state")
    def state():
        return guardian.state()

    @app.get("/v1/health")
    def health():
        state = guardian.state()
        return {"status": "ok" if state["healthy"] else "fault", **state}

    def get_frame(camera_id: str) -> tuple[bytes, float]:
        if camera_id not in CAMERA_IDS:
            raise HTTPException(status_code=404, detail="unknown camera_id")
        return cameras.frame_jpeg(camera_id)

    @app.get("/v1/cameras/{camera_id}/frame")
    def frame(camera_id: str):
        jpeg, timestamp = get_frame(camera_id)
        headers = {"X-Frame-Timestamp-Monotonic-S": str(timestamp)}
        processing_info = getattr(cameras, "processing_info", None)
        if callable(processing_info):
            info = processing_info(camera_id)
            headers.update({
                "X-Image-Processing": "lut" if info.enabled else "disabled",
                "X-Image-Processing-Backend": info.backend,
                "X-Image-Processing-Ms": f"{info.processing_ms:.3f}",
                "X-Frame-Number": str(info.frame_number),
            })
        return Response(jpeg, media_type="image/jpeg", headers=headers)

    @app.get("/v1/cameras/frame-batch")
    def frame_batch():
        batch = getattr(cameras, "frame_jpegs", None)
        if not callable(batch):
            raise HTTPException(status_code=501, detail="camera manager has no batch frame path")
        try:
            frames = batch(CAMERA_IDS)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        output = io.BytesIO()
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
            metadata = {}
            for camera_id in CAMERA_IDS:
                jpeg, timestamp = frames[camera_id]
                name = f"{camera_id}.jpg"
                archive.writestr(name, jpeg)
                metadata[camera_id] = {"file": name, "timestamp_monotonic_s": timestamp}
            archive.writestr("metadata.json", json.dumps({"encoding": "jpeg", "cameras": metadata}))
        return Response(output.getvalue(), media_type="application/zip")

    @app.get("/v1/cameras/{camera_id}/calibration-frame")
    def calibration_frame(camera_id: str):
        if camera_id not in CAMERA_IDS:
            raise HTTPException(status_code=404, detail="unknown camera_id")
        try:
            sample = cameras.calibration_frame(camera_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(
            encode_calibration_frame(sample),
            media_type="application/x-npz",
            headers={
                "X-Frame-Number": str(sample.frame_number),
                "X-Frame-Timestamp-Monotonic-S": str(sample.host_timestamp_monotonic_s),
            },
        )

    @app.post("/v1/cameras/{camera_id}/controls")
    def set_camera_controls(camera_id: str, request: CameraControlsRequest):
        if camera_id not in CAMERA_IDS:
            raise HTTPException(status_code=404, detail="unknown camera_id")
        try:
            return cameras.set_color_controls(camera_id, exposure=request.exposure, gain=request.gain)
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/cameras/{camera_id}/stream")
    def stream(camera_id: str):
        def frames() -> Iterator[bytes]:
            while True:
                jpeg, _ = get_frame(camera_id)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(1.0 / max(float(getattr(cameras, "fps", 30)), 1.0))
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/v1/cameras/{camera_id}/ws")
    async def websocket_stream(websocket: WebSocket, camera_id: str):
        if camera_id not in CAMERA_IDS:
            await websocket.close(code=1008, reason="unknown camera_id")
            return
        await websocket.accept()
        try:
            while True:
                jpeg, timestamp = await asyncio.to_thread(get_frame, camera_id)
                await websocket.send_json({
                    "camera_id": camera_id,
                    "timestamp_monotonic_s": timestamp,
                    "encoding": "jpeg",
                    "size_bytes": len(jpeg),
                })
                await websocket.send_bytes(jpeg)
                await asyncio.sleep(1.0 / max(float(getattr(cameras, "fps", 30)), 1.0))
        except WebSocketDisconnect:
            return

    @app.websocket("/v1/cameras/ws")
    async def websocket_all(websocket: WebSocket):
        batch = getattr(cameras, "frame_jpegs", None)
        if not callable(batch):
            await websocket.close(code=1011, reason="camera manager has no batch frame path")
            return
        await websocket.accept()
        try:
            while True:
                frames = await asyncio.to_thread(batch, CAMERA_IDS)
                await websocket.send_json({
                    "camera_ids": list(CAMERA_IDS),
                    "encoding": "jpeg",
                    "frames": {
                        camera_id: {"timestamp_monotonic_s": frames[camera_id][1], "size_bytes": len(frames[camera_id][0])}
                        for camera_id in CAMERA_IDS
                    },
                })
                for camera_id in CAMERA_IDS:
                    await websocket.send_bytes(frames[camera_id][0])
                await asyncio.sleep(1.0 / max(float(getattr(cameras, "fps", 30)), 1.0))
        except WebSocketDisconnect:
            return

    return app


# Uvicorn runs this sole long-lived robotcontrol process. It is the only
# process that will eventually own pyAgxArm/CAN; VLA and calibration stay HTTP
# clients and never access the driver directly.
app = create_app()
