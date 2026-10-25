import type { SessionCommand, SessionState } from "./types"
import { createCommandId } from "./command-id"

export function websocketUrl(path: string): string {
  const url = new URL(path, window.location.href)
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:"
  return url.href
}

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(payload.detail || response.statusText)
  }
  return response.json()
}

export const api = {
  session: () => jsonRequest<SessionState | null>("/api/v1/session"),
  start: () => jsonRequest<SessionState>("/api/v1/session/start", { method: "POST", body: JSON.stringify({}) }),
  command: (state: SessionState, command: SessionCommand) => jsonRequest<SessionState>("/api/v1/session/command", {
    method: "POST",
    body: JSON.stringify({ command, expected_revision: state.revision, command_id: createCommandId() }),
  }),
  hold: () => jsonRequest<SessionState>("/api/v1/safety/hold", { method: "POST", body: "{}" }),
  uploadBoardPhoto: async (photoId: string, file: File) => {
    const response = await fetch(`/api/v1/board-verification/photos/${photoId}`, {
      method: "PUT", headers: { "Content-Type": file.type || "application/octet-stream" }, body: file,
    })
    if (!response.ok) { const value = await response.json().catch(() => ({})); throw new Error(value.detail || response.statusText) }
    return response.json() as Promise<SessionState>
  },
  useMeasuredBoardScale: () => jsonRequest<SessionState>("/api/v1/board-verification/use-measured-scale", {
    method: "POST", body: "{}",
  }),
  runGeometry: () => jsonRequest<SessionState>("/api/v1/geometry/run", { method: "POST", body: "{}" }),
  startRobotwinPoseCheck: () => jsonRequest<SessionState>("/api/v1/motions/pose-check", { method: "POST", body: "{}" }),
  startRobotwinPlacement: () => jsonRequest<SessionState>("/api/v1/motions/placement", { method: "POST", body: "{}" }),
  confirmPlacement: (state: SessionState) => jsonRequest<SessionState>("/api/v1/placement/confirm", {
    method: "POST",
    body: JSON.stringify({ command: "confirm_placement", expected_revision: state.revision, command_id: createCommandId() }),
  }),
}
