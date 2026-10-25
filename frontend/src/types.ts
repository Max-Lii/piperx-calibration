export type Stage = "IDLE" | "SYSTEM_CHECK" | "TARGET_VERIFY" | "SAFETY_DRY_RUN" | "A4_PLACEMENT" | "WORLD_REGISTRATION" | "CAPTURE_LEFT" | "CAPTURE_RIGHT" | "SOLVE_GEOMETRY" | "VALIDATE_GEOMETRY" | "REVIEW_AND_EXPORT" | "COMPLETE" | "ABORTED"
export type RunStatus = "running" | "paused" | "recovery_required" | "complete" | "aborted"

export interface DeviceStatus { device_id: string; label: string; kind: "robot" | "camera" | "target"; connected: boolean; healthy: boolean; detail: string }
export interface SessionState {
  session_id: string; revision: number; mode: "mock" | "robotwin" | "http"; stage: Stage; status: RunStatus
  instruction: string; active_camera: "left_wrist" | "right_wrist" | "third_view"
  devices: DeviceStatus[]; progress: { current: number; total: number; label: string }
  robot_holding: boolean; recovery_required: boolean; warnings: string[]; errors: string[]
  results: Record<string, unknown>
  artifacts: Record<string, any>; stage_ready: boolean
}

export type SessionCommand = "continue" | "skip_pose_check" | "pause" | "retry" | "abort" | "terminate" | "restart" | "confirm_placement" | "approve_pose" | "reject_pose" | "go_home"
