from __future__ import annotations

import threading
import time
import uuid

from .adapter import WorkcellAdapter
from .schemas import (
    CommandRequest, ProgressState, RunStatus, SessionState, Stage,
    utc_now,
)
from .store import SessionStore


class WorkflowError(RuntimeError):
    pass


_NEXT_STAGE = {
    Stage.SYSTEM_CHECK: Stage.TARGET_VERIFY,
    Stage.TARGET_VERIFY: Stage.SAFETY_DRY_RUN,
    Stage.SAFETY_DRY_RUN: Stage.A4_PLACEMENT,
    Stage.A4_PLACEMENT: Stage.WORLD_REGISTRATION,
    Stage.WORLD_REGISTRATION: Stage.CAPTURE_LEFT,
    Stage.CAPTURE_LEFT: Stage.CAPTURE_RIGHT,
    Stage.CAPTURE_RIGHT: Stage.SOLVE_GEOMETRY,
    Stage.SOLVE_GEOMETRY: Stage.VALIDATE_GEOMETRY,
    Stage.VALIDATE_GEOMETRY: Stage.REVIEW_AND_EXPORT,
    Stage.REVIEW_AND_EXPORT: Stage.COMPLETE,
}

_INSTRUCTIONS = {
    Stage.SYSTEM_CHECK: "检查双臂、三台 D435 和机器人控制服务是否连接正常。",
    Stage.TARGET_VERIFY: "使用实测 35 mm 的默认 100% 尺度，或上传照片复核 BOARD_002 打印缩放。",
    Stage.SAFETY_DRY_RUN: "保持工作区清空，低速检查左右臂各 48 个采集姿态是否安全可达。",
    Stage.A4_PLACEMENT: "双臂进入对称高位姿态；根据第三视角定位十字居中并固定棋盘格，随后检查三路完整可见。",
    Stage.WORLD_REGISTRATION: "检测桌面棋盘格，建立统一的桌面世界坐标系。",
    Stage.CAPTURE_LEFT: "左臂依次到达 48 个标定姿态并采集；右臂保持停止。",
    Stage.CAPTURE_RIGHT: "右臂依次到达 48 个标定姿态并采集；左臂保持停止。",
    Stage.SOLVE_GEOMETRY: "解算双臂基座、双腕相机手眼关系和第三视角相机外参。",
    Stage.VALIDATE_GEOMETRY: "检查重投影误差、留出姿态、求解稳定性和深度一致性。",
    Stage.REVIEW_AND_EXPORT: "检查关键参数和质量门控，通过后导出结果；不会自动修改 RoboTwin。",
    Stage.COMPLETE: "校准完成。请检查结果并下载导出文件，或安全回到 Home 后开始新的标定。",
    Stage.ABORTED: "校准已终止，机械臂保持当前位置。",
}


class CalibrationWorkflow:
    def __init__(self, store: SessionStore, adapter: WorkcellAdapter, default_mode: str) -> None:
        self.store = store
        self.adapter = adapter
        self.default_mode = default_mode
        self._lock = threading.RLock()
        self._state = store.load_active()
        if self._state and self._state.stage == Stage.COMPLETE and not self._state.robot_holding:
            # Upgrade Sessions created before COMPLETE acquired explicit HOLD.
            self.adapter.hold()
            self._state.robot_holding = True
            self.store.save(self._state, event="complete_hold_reconciled")
        if (
            self._state and self._state.mode == "mock"
            and self._state.stage in {Stage.REVIEW_AND_EXPORT, Stage.COMPLETE}
            and not self._state.results
        ):
            self._state.results = self._mock_results()
            self.store.save(self._state, event="mock_results_reconciled")

    def snapshot(self) -> SessionState | None:
        with self._lock:
            return self._state.model_copy(deep=True) if self._state else None

    def start(self, mode: str | None = None) -> SessionState:
        with self._lock:
            if self._state and self._state.status not in {RunStatus.COMPLETE, RunStatus.ABORTED}:
                raise WorkflowError("已有未结束的校准 Session")
            selected_mode = mode or self.default_mode
            if selected_mode != self.default_mode:
                raise WorkflowError("请求模式与服务配置不一致")
            devices = self.adapter.system_check()
            state = SessionState(
                session_id=f"calibration_{utc_now().replace(':', '').replace('+00:00', 'Z')}_{uuid.uuid4().hex[:8]}",
                mode=selected_mode,
                stage=Stage.SYSTEM_CHECK,
                status=RunStatus.RUNNING,
                devices=devices,
                instruction=_INSTRUCTIONS[Stage.SYSTEM_CHECK],
                robot_holding=True,
                stage_ready=all(item.connected and item.healthy for item in devices),
            )
            self._state = state
            self.store.save(state, event="session_started")
            return state.model_copy(deep=True)

    def _require_state(self) -> SessionState:
        if self._state is None:
            raise WorkflowError("尚未创建校准 Session")
        return self._state

    def _commit(self, event: str, *, command_id: str | None = None) -> SessionState:
        state = self._require_state()
        if command_id is not None:
            state.processed_command_ids.append(command_id)
            state.processed_command_ids = state.processed_command_ids[-128:]
        state.revision += 1
        state.updated_at = utc_now()
        self.store.save(state, event=event)
        return state.model_copy(deep=True)

    def _fresh_paused_session(self, *, mode: str, instruction: str) -> SessionState:
        devices = self.adapter.system_check()
        fresh = SessionState(
            session_id=f"calibration_{utc_now().replace(':', '').replace('+00:00', 'Z')}_{uuid.uuid4().hex[:8]}",
            mode=mode,
            stage=Stage.SYSTEM_CHECK,
            status=RunStatus.PAUSED,
            devices=devices,
            instruction=instruction,
            robot_holding=True,
            stage_ready=all(item.connected and item.healthy for item in devices),
        )
        self._state = fresh
        self.store.save(fresh, event="session_reset_after_go_home")
        return fresh.model_copy(deep=True)

    @staticmethod
    def _mock_results() -> dict:
        """Representative output from the checked-in 48x2 RoboTwin regression."""
        return {
            "result_kind": "robotwin_simulation",
            "quality_status": "passed",
            "target_id": "BOARD_002",
            "poses_left": 48,
            "poses_right": 48,
            "dual_base_distance_mm": 599.374,
            "third_camera_height_mm": 515.892,
            "third_camera_pitch_deg": -59.964,
            "left_base_height_mm": 15.397,
            "right_base_height_mm": 15.216,
            "left_reprojection_median_px": 0.075,
            "right_reprojection_median_px": 0.079,
            "third_reprojection_px": 0.102,
            "transforms": {
                "T_world_left_base": [
                    [-0.0002118, -0.9999998, 0.0006060, -0.2028977],
                    [-0.9999999, 0.0002116, -0.0002925, 0.3923422],
                    [0.0002924, -0.0006061, -0.9999998, -0.0153968],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "T_world_right_base": [
                    [0.0004485, -0.9999998, 0.0003901, 0.3964761],
                    [-0.9999998, -0.0004483, 0.0004325, 0.3922800],
                    [-0.0004323, -0.0003903, -0.9999998, -0.0152160],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "T_left_ee_camera": [
                    [-0.0116192, -0.3309875, 0.9435636, -0.0159488],
                    [-0.9991230, 0.0418044, 0.0023609, 0.0427362],
                    [-0.0402265, -0.9427087, -0.3311829, 0.0838490],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "T_right_ee_camera": [
                    [-0.0110834, -0.3317656, 0.9432967, -0.0158828],
                    [-0.9991443, 0.0412664, 0.0027741, 0.0427418],
                    [-0.0398469, -0.9424588, -0.3319391, 0.0838083],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "T_world_third_camera": [
                    [1.0, -0.0001777, 0.0000719, 0.0649898],
                    [0.0001898, 0.8657123, -0.5005419, 0.3928398],
                    [0.0000267, 0.5005419, 0.8657123, -0.5158922],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "simulation_truth_errors": {
                "dual_base_distance_mm": 0.626,
                "third_camera_height_mm": 0.108,
                "third_camera_pitch_deg": 0.035,
            },
        }

    def record_board_verification(self, result: dict) -> SessionState:
        with self._lock:
            state = self._require_state()
            if state.stage != Stage.TARGET_VERIFY:
                raise WorkflowError("打印件照片只能在棋盘格打印验证阶段提交")
            state.artifacts["board_verification"] = result
            state.stage_ready = bool(result.get("accepted"))
            state.instruction = (
                "打印尺寸验证通过，可以继续。" if state.stage_ready
                else "尚未通过打印尺寸验证；请按提示补拍或按 100% 实际大小重新打印。"
            )
            return self._commit("board_verification_updated")

    def record_placement_detection(self, result: dict) -> SessionState:
        with self._lock:
            state = self._require_state()
            if state.stage != Stage.A4_PLACEMENT or not state.stage_ready:
                raise WorkflowError("棋盘格放置尚未确认")
            state.artifacts["placement_detection"] = result
            return self._commit("placement_detection_recorded")

    def _check_session(self, state: SessionState, session_id: str | None) -> None:
        if session_id is not None and state.session_id != session_id:
            raise WorkflowError("后台任务所属 Session 已结束，忽略迟到结果")

    def record_geometry_results(self, raw: dict, *, session_id: str | None = None) -> SessionState:
        with self._lock:
            state = self._require_state()
            self._check_session(state, session_id)
            expected = int(raw.get("poses_per_arm", 0))
            passed = (
                expected >= 12
                and bool(raw.get("quality", {}).get("passed"))
            )
            left, right, third = raw["left"], raw["right"], raw["third"]
            state.artifacts["geometry"] = raw
            state.artifacts["geometry_operation"] = {"state": "completed" if passed else "failed"}
            state.results = {
                "result_kind": "robotwin_simulation" if state.mode == "robotwin" else "physical_workcell",
                "quality_status": "passed" if passed else "failed",
                "target_id": raw.get("target_id", "BOARD_002"),
                "poses_left": left["accepted_poses"], "poses_right": right["accepted_poses"],
                "dual_base_distance_mm": raw["dual_base_distance_mm"],
                "third_camera_height_mm": third["height_from_target_plane_mm"],
                "third_camera_pitch_deg": third["pitch_deg"],
                "left_base_height_mm": raw["base_height_from_target_plane_mm"]["left"],
                "right_base_height_mm": raw["base_height_from_target_plane_mm"]["right"],
                "left_reprojection_median_px": left["reprojection_error_px_median"],
                "right_reprojection_median_px": right["reprojection_error_px_median"],
                "third_reprojection_px": third["reprojection_error_px"],
                "geometry_quality": raw.get("quality", {}),
                "transforms": {
                    "T_world_left_base": left["T_world_base"],
                    "T_world_right_base": right["T_world_base"],
                    "T_left_ee_camera": left["T_ee_camera"],
                    "T_right_ee_camera": right["T_ee_camera"],
                    "T_world_third_camera": third["T_world_camera"],
                },
            }
            if "dual_base_distance_error_mm" in raw:
                state.results["simulation_truth_errors"] = {
                    "dual_base_distance_mm": raw["dual_base_distance_error_mm"],
                    "third_camera_height_mm": third.get("height_error_mm"),
                    "third_camera_pitch_deg": third.get("pitch_error_deg"),
                }
            state.stage_ready = passed
            state.stage = Stage.VALIDATE_GEOMETRY
            state.progress = ProgressState()
            state.robot_holding = True
            state.instruction = (
                f"{expected} × 2 运动、采集与几何解算已完成；请检查质量结果。"
                if passed else "几何采集或求解未通过质量门控，请检查结果。"
            )
            return self._commit("geometry_results_recorded")

    def begin_geometry(self, *, total: int, session_id: str | None = None) -> SessionState:
        with self._lock:
            state = self._require_state()
            self._check_session(state, session_id)
            if state.stage != Stage.WORLD_REGISTRATION:
                raise WorkflowError("只能在桌面坐标系标定阶段开始几何采集")
            operation = state.artifacts.get("geometry_operation", {})
            if operation.get("state") == "running":
                raise WorkflowError("几何采集已经在运行")
            state.artifacts["geometry_operation"] = {"state": "running"}
            state.progress = ProgressState(current=0, total=total, label="准备几何采集")
            state.robot_holding = False
            state.stage_ready = False
            return self._commit("geometry_started")

    def update_geometry(self, *, arm: str, index: int, poses_per_arm: int, phase: str, session_id: str | None = None) -> SessionState:
        with self._lock:
            state = self._require_state()
            self._check_session(state, session_id)
            if state.artifacts.get("geometry_operation", {}).get("state") != "running":
                return state.model_copy(deep=True)
            offset = 0 if arm == "left" else poses_per_arm
            state.stage = Stage.CAPTURE_LEFT if arm == "left" else Stage.CAPTURE_RIGHT
            state.active_camera = "left_wrist" if arm == "left" else "right_wrist"
            state.progress.current = min(offset + index, state.progress.total)
            state.progress.label = f"{'左' if arm == 'left' else '右'}臂 {index}/{poses_per_arm} · {phase}"
            state.instruction = state.progress.label
            return self._commit("geometry_progress")

    def fail_geometry(self, detail: str, *, session_id: str | None = None) -> SessionState:
        with self._lock:
            state = self._require_state()
            self._check_session(state, session_id)
            state.artifacts["geometry_operation"] = {"state": "failed", "detail": detail}
            state.stage = Stage.WORLD_REGISTRATION
            state.progress = ProgressState()
            state.status = RunStatus.RECOVERY_REQUIRED
            state.recovery_required = True
            state.robot_holding = True
            state.errors.append(detail)
            state.instruction = detail
            return self._commit("geometry_failed")

    def cancel_geometry(self, *, session_id: str | None = None) -> SessionState:
        with self._lock:
            state = self._require_state()
            self._check_session(state, session_id)
            state.artifacts["geometry_operation"] = {
                "state": "cancelled", "detail": "几何采集已由操作者停止。",
            }
            state.robot_holding = True
            return self._commit("geometry_cancelled")

    def begin_robotwin_operation(self, operation: str, *, total: int, label: str) -> SessionState:
        with self._lock:
            state = self._require_state()
            if state.mode not in {"robotwin", "http"}:
                raise WorkflowError("该动作只允许连接 robotcontrol 的模式")
            expected_stage = (
                Stage.SAFETY_DRY_RUN if operation == "pose_check"
                else Stage.A4_PLACEMENT
            )
            if state.stage != expected_stage:
                raise WorkflowError("当前阶段不允许启动该工作台动作")
            active = state.artifacts.get("robotwin_operation", {})
            if active.get("state") == "running":
                raise WorkflowError("已有 RoboTwin 动作正在执行")
            state.artifacts["robotwin_operation"] = {"kind": operation, "state": "running"}
            state.progress = ProgressState(current=0, total=total, label=label)
            state.robot_holding = False
            state.stage_ready = False
            state.instruction = label
            return self._commit(f"robotwin_{operation}_started")

    def update_robotwin_operation(self, operation: str, *, current: int, label: str) -> SessionState:
        with self._lock:
            state = self._require_state()
            active = state.artifacts.get("robotwin_operation", {})
            if active.get("kind") != operation or active.get("state") != "running":
                return state.model_copy(deep=True)
            state.progress.current = min(max(current, 0), state.progress.total)
            state.progress.label = label
            state.instruction = label
            return self._commit(f"robotwin_{operation}_progress")

    def finish_robotwin_operation(self, operation: str, *, success: bool, detail: str, metadata: dict | None = None) -> SessionState:
        with self._lock:
            state = self._require_state()
            active = dict(state.artifacts.get("robotwin_operation", {}))
            if active.get("kind") != operation:
                return state.model_copy(deep=True)
            active.update({"state": "completed" if success else "failed", "detail": detail})
            if metadata:
                active.update(metadata)
            state.artifacts["robotwin_operation"] = active
            state.robot_holding = True
            if success:
                state.progress.current = state.progress.total
                state.stage_ready = operation == "pose_check"
                state.instruction = detail
            elif state.status == RunStatus.RUNNING:
                state.status = RunStatus.RECOVERY_REQUIRED
                state.recovery_required = True
                state.errors.append(detail)
                state.instruction = detail
            return self._commit(f"robotwin_{operation}_{'completed' if success else 'failed'}")

    def export_payload(self) -> dict:
        with self._lock:
            state = self._require_state()
            if state.stage not in {Stage.REVIEW_AND_EXPORT, Stage.COMPLETE}:
                raise WorkflowError("只有检查与导出阶段或已完成的 Session 可以导出")
            return {
                "schema_version": 1,
                "artifact": "piper_workcell_calibration",
                "session": state.model_dump(mode="json"),
                "results": state.results,
            }

    def command(self, request: CommandRequest) -> SessionState:
        with self._lock:
            state = self._require_state()
            if request.command_id in state.processed_command_ids:
                return state.model_copy(deep=True)
            if request.expected_revision != state.revision:
                raise WorkflowError(f"Session 已更新：expected={request.expected_revision}, current={state.revision}")
            if request.command == "pause":
                self.adapter.hold()
                state.status = RunStatus.PAUSED
                state.robot_holding = True
                state.instruction = "流程已暂停；检查现场后可继续。"
                return self._commit("paused", command_id=request.command_id)
            if request.command in {"abort", "terminate"}:
                if state.status not in {RunStatus.PAUSED, RunStatus.RECOVERY_REQUIRED} or not state.robot_holding:
                    raise WorkflowError("请先停止机器人执行，确认进入 HOLD 后再终止任务")
                # Keep the old Session until Go Home really finishes.  If any
                # call raises, the current paused Session remains the source of
                # truth and the operator can investigate/retry safely.
                self.adapter.go_home()
                state.stage = Stage.ABORTED
                state.status = RunStatus.ABORTED
                state.robot_holding = True
                state.instruction = "本次任务已终止，机械臂已回到 Home 并保持。"
                self._commit("terminated_after_go_home", command_id=request.command_id)

                return self._fresh_paused_session(
                    mode=state.mode,
                    instruction="上一任务已终止并安全回到 Home。确认现场安全后，恢复机器人执行并从系统检查重新开始。",
                )
            if request.command == "restart":
                if state.stage != Stage.COMPLETE or state.status != RunStatus.COMPLETE:
                    raise WorkflowError("只有已完成的 Session 可以开始新的标定")
                self.adapter.go_home()
                return self._fresh_paused_session(
                    mode=state.mode,
                    instruction="上一轮结果已保留，机械臂已安全回到 Home。恢复机器人执行以开始新的系统检查。",
                )
            if request.command == "go_home":
                self.adapter.go_home()
                state.robot_holding = False
                state.instruction = "正在执行显式低速 Go Home。"
                return self._commit("go_home_requested", command_id=request.command_id)
            if request.command == "skip_pose_check":
                if state.mode not in {"robotwin", "http"} or state.stage != Stage.SAFETY_DRY_RUN:
                    raise WorkflowError("只能在真机或 RoboTwin 的姿态安全检查阶段跳过")
                active = dict(state.artifacts.get("robotwin_operation", {}))
                if active.get("state") == "running":
                    raise WorkflowError("姿态安全检查正在执行，请先停止机器人执行")
                state.artifacts["robotwin_operation"] = {
                    "kind": "pose_check",
                    "state": "skipped",
                    "detail": "操作者确认已完成过安全验证，本轮跳过 96 姿态重复检查。",
                    "skipped_by_operator": True,
                }
                state.stage = Stage.A4_PLACEMENT
                state.stage_ready = False
                state.progress = ProgressState(current=0, total=1, label="等待双臂移动至棋盘格放置姿态")
                state.active_camera = "third_view"
                state.robot_holding = True
                state.instruction = _INSTRUCTIONS[Stage.A4_PLACEMENT]
                return self._commit("robotwin_pose_check_skipped", command_id=request.command_id)
            if request.command == "retry":
                state.status = RunStatus.RUNNING
                state.recovery_required = False
                state.errors.clear()
                state.instruction = _INSTRUCTIONS.get(state.stage, state.instruction)
                return self._commit("retry_requested", command_id=request.command_id)
            if request.command == "confirm_placement":
                if state.stage != Stage.A4_PLACEMENT:
                    raise WorkflowError("只能在放置棋盘格阶段确认")
                placement = state.artifacts.get("robotwin_operation", {})
                if state.mode in {"robotwin", "http"} and (
                    placement.get("kind") != "placement" or placement.get("state") != "completed"
                ):
                    raise WorkflowError("请先等待双臂到达棋盘格放置姿态")
                state.stage_ready = True
                state.instruction = "已确认棋盘格位置，准备建立桌面坐标系。"
                return self._commit("placement_confirmed", command_id=request.command_id)
            if request.command in {"approve_pose", "reject_pose"}:
                # Stage-specific semantics are intentionally serialized here;
                # physical motion handlers are added only after lab approval.
                if state.mode != "mock":
                    raise WorkflowError("真机阶段处理器尚未通过实验室认证")
            if request.command == "continue":
                if state.status in {RunStatus.PAUSED, RunStatus.RECOVERY_REQUIRED}:
                    state.status = RunStatus.RUNNING
                    state.recovery_required = False
                    state.instruction = _INSTRUCTIONS.get(state.stage, state.instruction)
                    return self._commit("resumed", command_id=request.command_id)
                if state.status != RunStatus.RUNNING:
                    raise WorkflowError("当前 Session 不能继续")
            # Mock mode is a deterministic UI walkthrough; hardware modes keep
            # their data gates.
            # RoboTwin/physical modes keep their actual data gates.
            if (
                state.mode == "http"
                and state.stage in {Stage.SYSTEM_CHECK, Stage.TARGET_VERIFY}
                and not state.stage_ready
            ):
                raise WorkflowError("当前阶段的数据或质量门控尚未完成")
            if (
                state.mode in {"robotwin", "http"}
                and state.stage in {Stage.SAFETY_DRY_RUN, Stage.A4_PLACEMENT}
                and not state.stage_ready
            ):
                raise WorkflowError("请先完成当前工作台动作与确认步骤")
            if (
                state.mode in {"robotwin", "http"}
                and state.stage in {
                    Stage.WORLD_REGISTRATION, Stage.CAPTURE_LEFT, Stage.CAPTURE_RIGHT,
                    Stage.SOLVE_GEOMETRY, Stage.VALIDATE_GEOMETRY,
                }
                and not state.stage_ready
                and not (state.stage == Stage.VALIDATE_GEOMETRY and bool(state.results))
            ):
                raise WorkflowError("请先完成几何采集、解算与质量门控")
            next_stage = _NEXT_STAGE.get(state.stage)
            if next_stage is None:
                raise WorkflowError("当前阶段没有下一步")
            state.stage = next_stage
            state.instruction = _INSTRUCTIONS[next_stage]
            state.active_camera = (
                "left_wrist" if next_stage == Stage.CAPTURE_LEFT else
                "right_wrist" if next_stage == Stage.CAPTURE_RIGHT else "third_view"
            )
            if next_stage in {Stage.CAPTURE_LEFT, Stage.CAPTURE_RIGHT}:
                state.progress = ProgressState(current=0, total=48, label="准备姿态")
            elif next_stage == Stage.SAFETY_DRY_RUN:
                state.progress = ProgressState(current=0, total=96, label="等待开始双臂姿态检查")
            else:
                state.progress = ProgressState()
            state.stage_ready = True
            if next_stage == Stage.TARGET_VERIFY:
                if state.mode in {"mock", "robotwin"}:
                    state.artifacts["board_verification"] = {
                        "accepted": True, "virtual": True, "valid_images": 3,
                        "scale_x": 1.0, "scale_y": 1.0, "measurement_spread": 0.0,
                    }
                else:
                    state.stage_ready = False
            elif next_stage == Stage.SAFETY_DRY_RUN and state.mode in {"robotwin", "http"}:
                state.stage_ready = False
            elif next_stage == Stage.A4_PLACEMENT and state.mode in {"robotwin", "http"}:
                state.stage_ready = False
                state.active_camera = "third_view"
                state.progress = ProgressState(current=0, total=1, label="等待双臂移动至棋盘格放置姿态")
            if next_stage == Stage.COMPLETE:
                self.adapter.hold()
                state.status = RunStatus.COMPLETE
                state.robot_holding = True
            if (
                state.mode in {"mock", "robotwin"}
                and next_stage in {Stage.REVIEW_AND_EXPORT, Stage.COMPLETE}
                and not state.results
            ):
                # UI walkthrough fallback. In robotwin mode an explicitly run
                # An explicitly run 48x2 regression overwrites this with session-local results.
                state.results = self._mock_results()
            return self._commit(f"entered_{next_stage.value.lower()}", command_id=request.command_id)

    def hold(self) -> SessionState:
        with self._lock:
            state = self._require_state()
            self.adapter.hold()
            state.status = RunStatus.PAUSED
            state.robot_holding = True
            state.instruction = "软件 HOLD 已触发。确认现场安全后再恢复。"
            return self._commit("safety_hold")
