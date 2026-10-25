import { useCallback, useEffect, useState } from "react"
import { api, websocketUrl } from "./api"
import type { SessionCommand, SessionState } from "./types"

export function useCalibrationSession() {
  const [session, setSession] = useState<SessionState | null>(null)
  const [connected, setConnected] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let stopped = false
    let socket: WebSocket | null = null
    let timer: number | undefined
    api.session().then((value) => { setSession(value); setConnected(true) }).catch((e) => setError(String(e)))
    const connect = () => {
      if (stopped) return
      socket = new WebSocket(websocketUrl("/ws/operator"))
      socket.onopen = () => setConnected(true)
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data)
        if (message.type === "session") setSession(message.session)
      }
      socket.onclose = () => {
        setConnected(false)
        if (!stopped) timer = window.setTimeout(connect, 1000)
      }
    }
    connect()
    return () => { stopped = true; if (timer) clearTimeout(timer); socket?.close() }
  }, [])

  // Execution progress is also polled. This keeps the UI live if a browser
  // reconnects while a workcell worker is in the middle of its pose sequence.
  useEffect(() => {
    const timer = window.setInterval(() => {
    api.session().then((value) => { setConnected(true); if (value) setSession(value) }).catch(() => setConnected(false))
    }, 750)
    return () => window.clearInterval(timer)
  }, [])

  const perform = useCallback(async (action: () => Promise<SessionState>) => {
    setBusy(true); setError(null)
    try { const value = await action(); setSession(value) }
    catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }, [])

  return {
    session, connected, busy, error,
    start: () => perform(api.start),
    command: (command: SessionCommand) => session && perform(() => api.command(session, command)),
    hold: () => perform(api.hold),
    uploadBoardPhotos: (files: File[]) => perform(async () => {
      let value = session!
      for (const file of files) value = await api.uploadBoardPhoto(createPhotoId(), file)
      return value
    }),
    useMeasuredBoardScale: () => perform(api.useMeasuredBoardScale),
    runGeometry: () => perform(api.runGeometry),
    startRobotwinPoseCheck: () => perform(api.startRobotwinPoseCheck),
    startRobotwinPlacement: () => perform(api.startRobotwinPlacement),
    confirmPlacement: () => session && perform(() => api.confirmPlacement(session)),
  }
}

function createPhotoId() {
  const random = Math.random().toString(36).slice(2)
  return `photo-${Date.now().toString(36)}-${random}`
}
