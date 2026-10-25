from __future__ import annotations

import asyncio
import json
import re
import time
import threading
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .adapter import HttpWorkcellAdapter, MockWorkcellAdapter, RobotwinPreviewAdapter, WorkcellAdapter
from piperx_calibration.robotcontrol.client import RobotControlClient
from .board_verification import aggregate_board_photos, analyze_board_photo
from .quality import CharucoQualityPolicy, CharucoTargetSpec, evaluate_charuco, evaluate_frame
from .schemas import CommandRequest, SessionState, StartSessionRequest
from .settings import CalibrationSettings, load_settings
from .store import SessionStore
from .workflow import CalibrationWorkflow, WorkflowError


class ConnectionHub:
    def __init__(self) -> None:
        self.operator: set[WebSocket] = set()
        self.target: set[WebSocket] = set()

    async def broadcast(self, state: SessionState | None) -> None:
        payload = {"type": "session", "session": state.model_dump(mode="json") if state else None}
        stale: list[tuple[set[WebSocket], WebSocket]] = []
        for group in (self.operator, self.target):
            for socket in tuple(group):
                try:
                    await socket.send_json(payload)
                except Exception:
                    stale.append((group, socket))
        for group, socket in stale:
            group.discard(socket)


def create_app(
    *,
    settings: CalibrationSettings | None = None,
    adapter: WorkcellAdapter | None = None,
    store: SessionStore | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    if adapter is None:
        if settings.mode == "mock" and settings.preview_backend == "robotwin":
            try:
                adapter = RobotwinPreviewAdapter(preview_fps=settings.preview_fps)
            except Exception as exc:
                raise RuntimeError(f"RoboTwin preview requested but could not start: {exc}") from exc
        else:
            adapter = MockWorkcellAdapter() if settings.mode == "mock" else HttpWorkcellAdapter(settings.robotcontrol_url)
    store = store or SessionStore(settings.session_root)
    workflow = CalibrationWorkflow(store, adapter, settings.mode)
    hub = ConnectionHub()
    geometry_control = {"cancel": threading.Event(), "thread": None}
    target_spec = CharucoTargetSpec.from_manifest(settings.target_manifest)
    preview_fallback = MockWorkcellAdapter() if settings.mode == "robotwin" else None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            adapter.close()
            if preview_fallback is not None:
                preview_fallback.close()

    app = FastAPI(title="Piper Workcell Calibrator", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.adapter = adapter
    app.state.workflow = workflow
    app.state.hub = hub

    def _pose_library() -> dict:
        if settings.pose_library is None:
            raise WorkflowError("服务未配置 geometry.pose_library")
        library = json.loads(settings.pose_library.read_text(encoding="utf-8"))
        for arm in ("left", "right"):
            if len(library.get(arm, [])) < settings.poses_per_arm:
                raise WorkflowError(f"姿态库 {arm} 少于 {settings.poses_per_arm} 条")
        return library

    def _start_workcell_motion(operation: str, loop: asyncio.AbstractEventLoop) -> SessionState:
        if settings.mode not in {"robotwin", "http"}:
            raise WorkflowError("该接口只用于 robotcontrol 工作台")
        poses = settings.poses_per_arm
        total = poses * 2 if operation == "pose_check" else 1
        label = (
            f"准备连续检查左臂 1/{poses}" if operation == "pose_check"
            else "正在移动双臂至棋盘格放置姿态"
        )
        state = workflow.begin_robotwin_operation(operation, total=total, label=label)

        def emit(updated: SessionState) -> None:
            asyncio.run_coroutine_threadsafe(hub.broadcast(updated), loop)

        def worker() -> None:
            client = RobotControlClient(settings.robotcontrol_url)
            success = False
            detail = ""
            operation_metadata: dict = {}
            try:
                backend = client.state().get("backend")
                # The simulation profile may use either a real RoboTwin
                # robotcontrol backend or the protocol-compatible mock
                # driver.  The physical profile remains explicitly `http`
                # and never takes this branch.
                if settings.mode == "robotwin" and backend not in {"robotwin", "mock"}:
                    raise RuntimeError("robotcontrol 后端不是 RoboTwin 或其 mock 实现")
                client.acquire(f"calibration-robotwin-{operation}")
                client.start_heartbeat(interval_s=3.0)
                if settings.mode == "http":
                    client.arm_motion()
                park = [0.0] * 6 if settings.mode == "http" else [0.0, 0.05, -0.05, 0.0, 0.0, 0.0]
                gripper = 0.0 if settings.mode == "http" else 1.0
                profile = "calibration_slow" if settings.mode == "http" else "vla_conservative"
                if operation == "placement":
                    placement = _pose_library().get("placement_poses")
                    if not isinstance(placement, dict):
                        raise RuntimeError("姿态库缺少已验证的双臂 placement_poses")
                    placement_left = list(placement.get("left", []))
                    placement_right = list(placement.get("right", []))
                    if len(placement_left) != 6 or len(placement_right) != 6:
                        raise RuntimeError("placement_poses 必须包含左右各 6 个关节值")
                    waypoint = placement_left + [gripper] + placement_right + [gripper]
                    execution = client.execute_joint_waypoints([waypoint], profile)
                    status = client.wait_execution(execution["execution_id"], timeout_s=30.0)
                    if status["state"] != "completed":
                        raise RuntimeError(f"放置姿态被停止：{status['state']}")
                    operation_metadata["placement_marker"] = {
                        "x_fraction": 0.5, "y_fraction": 0.5,
                        "camera_id": "third_view",
                        "source": "third_view_image_center",
                    }
                    operation_metadata["validation_cameras"] = ["left_wrist", "third_view", "right_wrist"]
                    detail = "双臂已到达对称高位放置姿态；请在第三视角中居中棋盘格。"
                    success = True
                else:
                    library = _pose_library()
                    for arm, offset in (("left", 0), ("right", poses)):
                        for index, pose in enumerate(library[arm][:poses], start=1):
                            current = offset + index - 1
                            emit(workflow.update_robotwin_operation(
                                operation, current=current,
                                label=f"正在检查{'左' if arm == 'left' else '右'}臂姿态 {index}/{poses}（总计 {current}/{total}）",
                            ))
                            left = list(pose) if arm == "left" else park
                            right = list(pose) if arm == "right" else park
                            execution = client.execute_joint_waypoints([left + [gripper] + right + [gripper]], profile)
                            status = client.wait_execution(execution["execution_id"], timeout_s=30.0)
                            if status["state"] != "completed":
                                raise RuntimeError(f"姿态 {arm}_{index:02d} 被停止：{status['state']}")
                            emit(workflow.update_robotwin_operation(
                                operation, current=current + 1,
                                label=f"已完成{'左' if arm == 'left' else '右'}臂姿态 {index}/{poses}（总计 {current + 1}/{total}）",
                            ))
                    home = client.go_home()
                    home_status = client.wait_execution(home["execution_id"], timeout_s=75.0)
                    if home_status["state"] != "completed":
                        raise RuntimeError(f"姿态检查后回 Home 失败：{home_status['state']}")
                    success, detail = True, f"双臂 {total} 个姿态检查完成，已回 Home 并 HOLD。"
            except Exception as exc:
                detail = f"工作台动作停止：{type(exc).__name__}: {exc}"
            finally:
                try:
                    client.hold()
                    client.release()
                except Exception:
                    pass
                client.close()
                emit(workflow.finish_robotwin_operation(operation, success=success, detail=detail, metadata=operation_metadata))

        threading.Thread(target=worker, name=f"calibration-{operation}", daemon=True).start()
        return state

    @app.exception_handler(WorkflowError)
    async def workflow_error(_: Request, exc: WorkflowError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/api/v1/health")
    async def health():
        return {
            "status": "ok",
            "mode": settings.mode,
            "robotcontrol_url": settings.robotcontrol_url,
            "frontend_built": settings.frontend_dist.is_dir(),
        }

    @app.get("/api/v1/cameras/{camera_id}/stream")
    async def camera_stream(camera_id: str):
        """Expose the adapter preview as MJPEG for the operator UI.

        The mock adapter generates frames in-process; the HTTP adapter delegates
        frame ownership to robotcontrol. Both modes therefore use one browser
        endpoint without opening a camera twice.
        """
        if camera_id not in {"left_wrist", "third_view", "right_wrist"}:
            raise HTTPException(status_code=404, detail="unknown camera")

        async def frames():
            while True:
                payload, _ = await asyncio.to_thread(adapter.frame_jpeg, camera_id)
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                    + payload + b"\r\n"
                )
                await asyncio.sleep(1.0 / settings.preview_fps)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @app.get("/api/v1/workcell/probe")
    async def workcell_probe():
        """Read robot state and all lossless camera paths without acquiring motion control."""
        devices = adapter.system_check()
        cameras = {}
        for camera_id in ("left_wrist", "right_wrist", "third_view"):
            frame, raw = adapter.calibration_frame(camera_id)
            cameras[camera_id] = {
                "serial": frame.serial,
                "frame_number": frame.frame_number,
                "rgb_shape": list(frame.rgb.shape),
                "depth_shape": list(frame.depth_raw.shape),
                "depth_scale_m": frame.depth_scale_m,
                "depth_aligned_to_color": frame.depth_aligned_to_color,
                "intrinsics": frame.color_intrinsics.tolist(),
                "distortion_model": frame.distortion_model,
                "camera_controls": frame.camera_controls,
                "payload_bytes": len(raw),
                "quality": evaluate_frame(frame).to_dict(),
            }
        return {"motion_requested": False, "devices": [item.model_dump() for item in devices], "cameras": cameras}

    @app.get("/api/v1/session", response_model=SessionState | None)
    async def session():
        return workflow.snapshot()

    @app.post("/api/v1/session/start", response_model=SessionState)
    async def start(request: StartSessionRequest):
        state = workflow.start(request.mode)
        await hub.broadcast(state)
        return state

    @app.post("/api/v1/session/command", response_model=SessionState)
    async def command(request: CommandRequest):
        # Termination can wait for a slow physical Go Home.  Keep camera and
        # WebSocket traffic responsive while the synchronous adapter waits.
        previous = workflow.snapshot()
        if request.command in {"pause", "abort", "terminate"}:
            geometry_control["cancel"].set()
            await asyncio.to_thread(adapter.hold)
            thread = geometry_control.get("thread")
            if isinstance(thread, threading.Thread) and thread.is_alive():
                await asyncio.to_thread(thread.join, 15.0)
                if thread.is_alive():
                    raise WorkflowError("几何采集线程尚未停止；机器人已 HOLD，请稍后重试")
        if request.command in {"abort", "terminate", "restart", "go_home"}:
            state = await asyncio.to_thread(workflow.command, request)
        else:
            state = workflow.command(request)
        if (
            previous is not None
            and previous.stage.value == "SAFETY_DRY_RUN"
            and state.stage.value == "A4_PLACEMENT"
            and settings.mode in {"robotwin", "http"}
        ):
            state = _start_workcell_motion("placement", asyncio.get_running_loop())
        await hub.broadcast(state)
        return state

    @app.get("/api/v1/session/export")
    async def export_session():
        payload = workflow.export_payload()
        session_id = payload["session"]["session_id"]
        return Response(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{session_id}_workcell_calibration.json"',
                "Cache-Control": "no-store",
            },
        )

    @app.post("/api/v1/safety/hold", response_model=SessionState)
    async def hold():
        geometry_control["cancel"].set()
        # The mock adapter is an in-process no-op. Calling it directly keeps
        # short ASGI/unit-test requests from creating a thread-pool lifetime
        # dependency; the HTTP adapter remains isolated from the event loop.
        if isinstance(adapter, MockWorkcellAdapter):
            adapter.hold()
        else:
            await asyncio.to_thread(adapter.hold)
        thread = geometry_control.get("thread")
        if isinstance(thread, threading.Thread) and thread.is_alive():
            await asyncio.to_thread(thread.join, 15.0)
        state = workflow.hold()
        await hub.broadcast(state)
        return state

    @app.put("/api/v1/board-verification/photos/{photo_id}", response_model=SessionState)
    async def upload_board_photo(photo_id: str, request: Request):
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", photo_id):
            raise HTTPException(status_code=400, detail="invalid photo_id")
        state = workflow.snapshot()
        if state is None:
            raise HTTPException(status_code=409, detail="尚未创建校准 Session")
        payload = await request.body()
        if not 1_000 <= len(payload) <= 30 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="照片必须在 1KB 到 30MB 之间")
        directory = store.session_dir(state.session_id) / "board_verification"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{photo_id}.image").write_bytes(payload)
        analyses = []
        for path in sorted(directory.glob("*.image")):
            try:
                analyses.append({"photo_id": path.stem, **analyze_board_photo(path.read_bytes(), settings.target_manifest)})
            except Exception as exc:
                analyses.append({"photo_id": path.stem, "error": str(exc)})
        result = aggregate_board_photos(analyses)
        updated = workflow.record_board_verification(result)
        await hub.broadcast(updated)
        return updated

    @app.post("/api/v1/board-verification/use-measured-scale", response_model=SessionState)
    async def use_measured_board_scale():
        manifest = json.loads(settings.target_manifest.read_text(encoding="utf-8"))
        result = {
            "accepted": True,
            "verification_method": "operator_caliper_measurement",
            "target_id": manifest["target_id"],
            "square_length_mm": float(manifest["square_length_mm"]),
            "scale_x": 1.0,
            "scale_y": 1.0,
            "uniform_scale": 1.0,
            "measurement_spread": 0.0,
            "valid_images": 0,
            "total_images": 0,
            "operator_confirmed": True,
            "note": "棋盘格方格已用游标卡尺确认，纸张裁边不作为打印内容尺度。",
        }
        updated = workflow.record_board_verification(result)
        await hub.broadcast(updated)
        return updated

    @app.post("/api/v1/geometry/run", response_model=SessionState)
    async def run_geometry():
        state = workflow.snapshot()
        if settings.mode not in {"robotwin", "http"} or state is None:
            raise WorkflowError("该接口只用于已启动的 robotcontrol 校准 Session")
        if state.stage.value != "WORLD_REGISTRATION":
            raise WorkflowError("请先完成姿态检查并确认棋盘格放置")
        from .run_robotwin_e2e import run
        _pose_library()
        session_id = state.session_id
        geometry_control["cancel"].clear()
        started = workflow.begin_geometry(total=settings.poses_per_arm * 2, session_id=session_id)
        loop = asyncio.get_running_loop()

        def progress(arm: str, index: int, poses: int, phase: str) -> None:
            try:
                updated = workflow.update_geometry(
                    arm=arm, index=index, poses_per_arm=poses, phase=phase, session_id=session_id,
                )
            except WorkflowError:
                return
            asyncio.run_coroutine_threadsafe(hub.broadcast(updated), loop)

        args = Namespace(
            robotcontrol_url=settings.robotcontrol_url,
            pose_library=settings.pose_library,
            target_manifest=settings.target_manifest,
            output=store.session_dir(state.session_id) / "geometry",
            poses=settings.poses_per_arm, truth=None,
            expected_backend="robotwin" if settings.mode == "robotwin" else None,
            stable_for_s=2.0,
            burst_frames=3, burst_interval_s=0.1, minimum_accepted_frames=1,
            progress=progress,
            should_cancel=geometry_control["cancel"].is_set,
        )
        def worker() -> None:
            try:
                raw = run(args)
                updated = workflow.record_geometry_results(raw, session_id=session_id)
            except Exception as exc:
                try:
                    if geometry_control["cancel"].is_set():
                        updated = workflow.cancel_geometry(session_id=session_id)
                    else:
                        updated = workflow.fail_geometry(
                            f"几何采集失败：{type(exc).__name__}: {exc}", session_id=session_id,
                        )
                except WorkflowError:
                    return
            asyncio.run_coroutine_threadsafe(hub.broadcast(updated), loop)

        thread = threading.Thread(target=worker, name="calibration-geometry", daemon=True)
        geometry_control["thread"] = thread
        thread.start()
        await hub.broadcast(started)
        return started

    @app.post("/api/v1/robotwin/run-geometry", response_model=SessionState, include_in_schema=False)
    async def run_robotwin_geometry_compat():
        return await run_geometry()

    @app.post("/api/v1/motions/pose-check", response_model=SessionState)
    async def start_pose_check():
        state = _start_workcell_motion("pose_check", asyncio.get_running_loop())
        await hub.broadcast(state)
        return state

    @app.post("/api/v1/motions/placement", response_model=SessionState)
    async def start_placement():
        state = _start_workcell_motion("placement", asyncio.get_running_loop())
        await hub.broadcast(state)
        return state

    @app.post("/api/v1/robotwin/pose-check", response_model=SessionState, include_in_schema=False)
    async def start_robotwin_pose_check_compat():
        return await start_pose_check()

    @app.post("/api/v1/robotwin/placement", response_model=SessionState, include_in_schema=False)
    async def start_robotwin_placement_compat():
        return await start_placement()

    @app.post("/api/v1/placement/confirm", response_model=SessionState)
    async def confirm_placement(request: CommandRequest):
        if request.command != "confirm_placement":
            raise HTTPException(status_code=400, detail="command must be confirm_placement")
        if settings.mode == "mock":
            state = workflow.command(request)
            state = workflow.record_placement_detection({"all_cameras_accepted": True, "virtual": True})
            await hub.broadcast(state)
            return state
        camera_labels = {"left_wrist": "左腕", "third_view": "第三视角", "right_wrist": "右腕"}
        strict_policy = CharucoQualityPolicy(
            minimum_corners=24,
            minimum_hull_area_fraction=0.02,
            minimum_border_px=24.0,
            maximum_reprojection_error_px=2.0,
        )
        cameras = {}
        failures = []
        warnings = []
        for camera_id in ("left_wrist", "third_view", "right_wrist"):
            frame, _ = adapter.calibration_frame(camera_id)
            frame_quality = evaluate_frame(frame).to_dict()
            detected = evaluate_charuco(frame, target_spec, strict_policy)
            cameras[camera_id] = {"frame_quality": frame_quality, "target_quality": detected}
            blocking_frame_reasons = [
                reason for reason in frame_quality["reasons"] if reason != "insufficient_depth"
            ]
            if "insufficient_depth" in frame_quality["reasons"]:
                warnings.append(f"{camera_labels[camera_id]}深度覆盖不足")
            if blocking_frame_reasons:
                failures.append(f"{camera_labels[camera_id]}图像/深度：{', '.join(blocking_frame_reasons)}")
            if not detected["accepted"]:
                failures.append(f"{camera_labels[camera_id]}棋盘格：{', '.join(detected['reasons'])}")
        if failures:
            raise WorkflowError("三视角放置检查未通过；" + "；".join(failures))
        state = workflow.command(request)
        state = workflow.record_placement_detection({
            "all_cameras_accepted": True,
            "cameras": cameras,
            "warnings": warnings,
        })
        await hub.broadcast(state)
        return state

    @app.websocket("/ws/operator")
    async def operator_socket(socket: WebSocket):
        await socket.accept()
        hub.operator.add(socket)
        await socket.send_json({
            "type": "session",
            "session": workflow.snapshot().model_dump(mode="json") if workflow.snapshot() else None,
        })
        try:
            while True:
                message = await socket.receive_json()
                if message.get("type") == "ping":
                    await socket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            hub.operator.discard(socket)

    if settings.frontend_dist.is_dir():
        assets = settings.frontend_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

        @app.get("/operator", include_in_schema=False)
        async def frontend_page():
            # index.html points at content-hashed bundles and must revalidate;
            # otherwise a tablet can keep an old interaction bug after a
            # server-side UI rebuild.
            return FileResponse(
                settings.frontend_dist / "index.html",
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse("/operator")
    else:
        @app.get("/", include_in_schema=False)
        async def root_unbuilt():
            return {"message": "Frontend is not built; run npm install && npm run build in calibration/frontend."}

    return app
