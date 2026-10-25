import { AlertTriangle, CheckCircle2, Download, House, Play, RotateCcw, ShieldAlert, Square } from "lucide-react"
import { Badge } from "./components/ui/badge"
import { Button } from "./components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card"
import { Progress } from "./components/ui/progress"
import { useEffect, useState } from "react"
import { useCalibrationSession } from "./use-session"
import type { SessionCommand } from "./types"
import type { SessionState, Stage } from "./types"

type Language = "zh" | "en"
const stageLabels: Record<Stage, [string, string]> = {
  IDLE: ["未开始", "Not started"], ABORTED: ["已终止", "Aborted"],
  SYSTEM_CHECK: ["设备连接检查", "Device check"], TARGET_VERIFY: ["棋盘格打印验证", "Target verification"],
  SAFETY_DRY_RUN: ["机械臂姿态检查", "Pose safety check"], A4_PLACEMENT: ["放置棋盘格", "Place target"],
  WORLD_REGISTRATION: ["桌面坐标系标定", "World registration"], CAPTURE_LEFT: ["左臂标定采集", "Capture left arm"],
  CAPTURE_RIGHT: ["右臂标定采集", "Capture right arm"], SOLVE_GEOMETRY: ["外参解算", "Solve geometry"],
  VALIDATE_GEOMETRY: ["标定精度验证", "Validate geometry"], REVIEW_AND_EXPORT: ["结果检查与导出", "Review and export"], COMPLETE: ["完成", "Complete"],
}
const cameraLabels: Record<"left_wrist" | "third_view" | "right_wrist", [string, string]> = { left_wrist: ["左腕相机", "Left wrist"], third_view: ["第三视角相机", "Third view"], right_wrist: ["右腕相机", "Right wrist"] }
const text = (language: Language, zh: string, en: string) => language === "zh" ? zh : en
const stageOrder: Stage[] = ["SYSTEM_CHECK", "TARGET_VERIFY", "SAFETY_DRY_RUN", "A4_PLACEMENT", "WORLD_REGISTRATION", "CAPTURE_LEFT", "CAPTURE_RIGHT", "SOLVE_GEOMETRY", "VALIDATE_GEOMETRY", "REVIEW_AND_EXPORT", "COMPLETE"]
const stages = (language: Language) => stageOrder.map(id => ({ id, label: stageLabels[id][language === "zh" ? 0 : 1] }))
const instructionLabels: Record<Stage, string> = {
  IDLE: "Start a calibration session.", ABORTED: "The calibration session was aborted.",
  SYSTEM_CHECK: "Check both arms, all three D435 cameras, and the control service.", TARGET_VERIFY: "Confirm the printed 35 mm scale or upload photos to verify BOARD_002.",
  SAFETY_DRY_RUN: "Keep the workspace clear and check the configured acquisition poses at low speed.", A4_PLACEMENT: "Move to the placement view, center the target in the third view, and keep it fixed.",
  WORLD_REGISTRATION: "Detect the tabletop target and establish the shared reference frame.", CAPTURE_LEFT: "Capture the left arm sequentially while the right arm remains parked.",
  CAPTURE_RIGHT: "Capture the right arm sequentially while the left arm remains parked.", SOLVE_GEOMETRY: "Solve both bases, both wrist hand–eye transforms, and the fixed camera.",
  VALIDATE_GEOMETRY: "Review reprojection, held-out poses, stability, and depth consistency.", REVIEW_AND_EXPORT: "Review the quality report and export the result for your simulation.", COMPLETE: "Calibration complete. Review and download the result.",
}
function useLanguage(): [Language, () => void] {
  const [language, setLanguage] = useState<Language>("en")
  useEffect(() => { const saved = window.localStorage.getItem("piper-calibration-language"); if (saved === "en" || saved === "zh") setLanguage(saved) }, [])
  const toggle = () => setLanguage(value => { const next = value === "zh" ? "en" : "zh"; window.localStorage.setItem("piper-calibration-language", next); return next })
  return [language, toggle]
}

function CameraView({ id, language, active, recorded, placementTarget, marker }: { id: keyof typeof cameraLabels; language: Language; active?: boolean; recorded?: boolean; placementTarget?: boolean; marker?: { x_fraction?: number; y_fraction?: number } }) {
  const markerStyle = { left: `${100 * (marker?.x_fraction ?? .5)}%`, top: `${100 * (marker?.y_fraction ?? .5)}%` }
  return <Card className={active ? "ring-2 ring-zinc-900" : ""}>
    <CardHeader className="flex-row items-center justify-between py-3"><CardTitle className="text-sm">{cameraLabels[id][language === "zh" ? 0 : 1]}</CardTitle><Badge ok>● {text(language, "在线", "Online")}</Badge></CardHeader>
    <CardContent><div className="relative overflow-hidden rounded-md bg-slate-900 aspect-[4/3]"><img className="h-full w-full object-cover" src={`/api/v1/cameras/${id}/stream`} /><span className="absolute bottom-2 left-2 rounded bg-black/65 px-2 py-1 text-xs text-white">640 × 480 · {recorded ? "RECORDED" : "LIVE"}</span>{placementTarget && <div className="absolute" style={markerStyle}><span className="absolute h-1 w-14 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-red-500 shadow"/><span className="absolute h-1 w-14 -translate-x-1/2 -translate-y-1/2 -rotate-45 bg-red-500 shadow"/><span className="absolute left-0 top-8 -translate-x-1/2 whitespace-nowrap rounded bg-black/70 px-2 py-1 text-xs font-medium text-white">棋盘格中心对准 X</span></div>}</div></CardContent>
  </Card>
}

function BoardVerificationPanel({ language, session, busy, upload, useMeasuredScale }: { language: Language; session: SessionState; busy: boolean; upload: (files: File[]) => void; useMeasuredScale: () => void }) {
  const result = session.artifacts.board_verification as any
  if (session.artifacts.offline_replay) return <Card><CardHeader><CardTitle>{text(language, "虚拟标定板参数", "Recorded virtual target")}</CardTitle></CardHeader><CardContent><p className="text-sm leading-6 text-zinc-600">{text(language, "BOARD_002 · 5×7 方格 · 35 mm · 标记 26.25 mm。参数来自本轮仿真 manifest；没有测量纸张，也没有验证实体打印。", "BOARD_002 · 5×7 squares · 35 mm · 26.25 mm markers. Dimensions come from this recording's simulation manifest; no paper was measured or physical print verified.")}</p></CardContent></Card>
  return <Card><CardHeader><CardTitle>{text(language, "确认棋盘格尺度", "Confirm target scale")}</CardTitle></CardHeader><CardContent>
    <Button disabled={busy || result?.accepted} onClick={useMeasuredScale}><CheckCircle2 className="mr-2 h-4 w-4"/>{text(language, "使用实测 35.00 mm（100%）", "Use measured 35.00 mm (100%)")}</Button>
    <div className="my-4 border-t border-zinc-200"/>
    <p className="mb-4 text-sm leading-6 text-zinc-500">{text(language, "用手机从正面、左斜、右斜各拍一张。A4 四角必须完整入镜，纸张放在有颜色对比的平面，上传原图。程序直接使用 210 × 297 mm 外边界检查打印缩放。", "Take one photo from the front and two oblique views. Keep all four A4 corners visible, place the sheet on a contrasting surface, and upload the original images. The service checks scale against the 210 × 297 mm outer border.")}</p>
    <input type="file" accept="image/jpeg,image/png,image/*" capture="environment" multiple disabled={busy} onChange={(event) => { const files = Array.from(event.currentTarget.files ?? []); if (files.length) upload(files) }} className="block w-full rounded-md border border-zinc-200 bg-zinc-50 p-3 text-sm file:mr-4 file:rounded-md file:border-0 file:bg-zinc-900 file:px-4 file:py-2 file:text-white"/>
    <div className={`mt-4 rounded-md border p-4 text-sm ${result?.accepted ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-zinc-200 bg-zinc-50 text-zinc-600"}`}>
      {result ? <><b>{result.accepted ? text(language, "验证通过", "Verified") : text(language, "尚未通过", "Not verified")}</b><div className="mt-2">{result.verification_method === "operator_caliper_measurement" ? text(language, `游标卡尺实测 ${Number(result.square_length_mm).toFixed(2)} mm · 采用 100%`, `Caliper measurement ${Number(result.square_length_mm).toFixed(2)} mm · using 100%`) : `${text(language, "有效照片", "Valid photos")} ${result.valid_images ?? 0} / ${result.total_images ?? 0} · X ${typeof result.scale_x === "number" ? (result.scale_x * 100).toFixed(2) + "%" : "—"} · Y ${typeof result.scale_y === "number" ? (result.scale_y * 100).toFixed(2) + "%" : "—"}`}</div>{result.reasons?.length > 0 && <div className="mt-2">{text(language, "原因：", "Reason: ")}{result.reasons.join(language === "zh" ? "、" : ", ")}</div>}</> : text(language, "选择实测 35.00 mm，或上传至少 3 张照片复核。", "Use the measured 35.00 mm scale, or upload at least three photos for verification.")}
    </div>
  </CardContent></Card>
}

function MainPanel({ language, session, busy, command, uploadBoardPhotos, useMeasuredBoardScale, runGeometry, startRobotwinPoseCheck, startRobotwinPlacement, confirmPlacement }: { language: Language; session: SessionState; busy: boolean; command: (command: SessionCommand) => void; uploadBoardPhotos: (files: File[]) => void; useMeasuredBoardScale: () => void; runGeometry: () => void; startRobotwinPoseCheck: () => void; startRobotwinPlacement: () => void; confirmPlacement: () => void }) {
  if (session.stage === "REVIEW_AND_EXPORT" || session.stage === "COMPLETE") {
    return <ResultPanel language={language} session={session} busy={busy}/>
  }
  const isCheck = session.stage === "SYSTEM_CHECK"
  const cameraOrder = ["left_wrist", "third_view", "right_wrist"] as const
  return <>
    <div className="mb-5"><p className="text-xs font-medium uppercase tracking-[0.18em] text-zinc-500">{text(language, "当前阶段", "Current stage")}</p><h1 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-950">{stageLabels[session.stage][language === "zh" ? 0 : 1]}</h1><p className="mt-2 text-zinc-500">{session.artifacts.offline_replay ? session.instruction : language === "en" ? instructionLabels[session.stage] : session.instruction}</p></div>
    {session.mode !== "mock" && session.stage === "SAFETY_DRY_RUN" && <Card className="mb-4"><CardHeader><CardTitle>{text(language, "双臂连续姿态检查", "Dual-arm pose check")}</CardTitle></CardHeader><CardContent><p className="mb-4 text-sm leading-6 text-zinc-500">{text(language, "连续执行左臂 48 个、右臂 48 个已验证姿态。执行中底部“停止机器人执行”始终可用；停止后保持 HOLD，不会自动进入下一阶段。", "Execute 48 validated poses for each arm. The stop control remains available throughout; after a stop the robot stays in HOLD and does not advance automatically.")}</p><div className="flex flex-wrap gap-2"><Button disabled={busy || session.artifacts.robotwin_operation?.state === "running" || session.stage_ready} onClick={startRobotwinPoseCheck}>{session.stage_ready ? text(language, "96 个姿态检查已通过", "96 poses passed") : session.artifacts.robotwin_operation?.state === "running" ? text(language, "正在执行…", "Running…") : text(language, "开始连续执行 96 个姿态", "Run 96 poses")}</Button><Button variant="outline" disabled={busy || session.artifacts.robotwin_operation?.state === "running" || session.stage_ready} onClick={() => void command("skip_pose_check")}>{text(language, "跳过本轮安全检查", "Skip this safety check")}</Button></div><p className="mt-3 text-xs text-amber-700">{text(language, "仅在你已确认当前姿态库和现场安全时使用；跳过会记录在 Session 中。", "Use only after confirming the pose library and workspace are safe; skipping is recorded in the Session.")}</p></CardContent></Card>}
    {session.mode !== "mock" && session.stage === "A4_PLACEMENT" && <Card className="mb-4"><CardHeader><CardTitle>棋盘格放置引导</CardTitle></CardHeader><CardContent><p className="mb-4 text-sm leading-6 text-zinc-500">双臂会同时移动到已验证的对称高位姿态。请在第三视角大画面中让棋盘格尽量居中，并保持 ALIGN X 方向正确；确认时服务器会检查三路无损 RGB-D 均完整识别 BOARD_002。</p>{session.artifacts.robotwin_operation?.state === "completed" ? <Button disabled={busy} onClick={confirmPlacement}>检查三路并确认棋盘格</Button> : <Button disabled={busy || session.artifacts.robotwin_operation?.state === "running"} onClick={startRobotwinPlacement}>{session.artifacts.robotwin_operation?.state === "running" ? "正在移动…" : "移动双臂到放置姿态"}</Button>}</CardContent></Card>}
    {session.mode !== "mock" && session.stage === "WORLD_REGISTRATION" && <Card className="mb-4"><CardHeader><CardTitle>开始几何标定</CardTitle></CardHeader><CardContent><p className="mb-4 text-sm leading-6 text-zinc-500">程序将独占 robotcontrol lease，逐姿态等待反馈稳定，采集三路无损 RGB-D，并自动解算双基座、双腕手眼与第三视角外参。</p><Button disabled={busy || session.artifacts.geometry_operation?.state === "running"} onClick={runGeometry}>{session.artifacts.geometry_operation?.state === "running" ? "采集正在运行" : "开始 96 姿态采集与解算"}</Button></CardContent></Card>}
    {session.stage === "TARGET_VERIFY" ? <BoardVerificationPanel language={language} session={session} busy={busy} upload={uploadBoardPhotos} useMeasuredScale={useMeasuredBoardScale}/> : isCheck ? <div className="grid grid-cols-1 gap-4 md:grid-cols-3">{cameraOrder.map((id) => <CameraView key={id} id={id} language={language} recorded={Boolean(session.artifacts.offline_replay)}/>)}</div> : <div className="grid min-w-0 grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(260px,1fr)]"><CameraView id={session.active_camera} language={language} recorded={Boolean(session.artifacts.offline_replay)} active placementTarget={session.stage === "A4_PLACEMENT" && session.active_camera === "third_view"} marker={session.artifacts.robotwin_operation?.placement_marker}/><div className="space-y-4"> <Card><CardHeader><CardTitle>{text(language, "阶段状态", "Stage status")}</CardTitle></CardHeader><CardContent className="space-y-4"><Badge ok={session.status === "running"}>{session.status === "running" ? text(language, "运行中", "Running") : session.status === "paused" ? "HOLD" : session.status}</Badge>{session.progress.total > 0 && <><div className="flex justify-between text-sm"><span>{session.artifacts.offline_replay ? session.progress.label : language === "en" ? "Progress" : session.progress.label}</span><span>{session.progress.current} / {session.progress.total}</span></div><Progress value={100 * session.progress.current / session.progress.total}/></>}</CardContent></Card>{cameraOrder.filter((x) => x !== session.active_camera).map((id) => <CameraView key={id} id={id} language={language} recorded={Boolean(session.artifacts.offline_replay)}/>)}</div></div>}
    <div className="mt-4 rounded-lg border border-zinc-200 bg-white p-4 text-sm text-zinc-600"><CheckCircle2 className="mr-2 inline h-5 w-5 text-emerald-600"/>{text(language, "所有判断以 Server 状态和质量门控为准，网页预览不参与解算。", "The server state and quality gates are authoritative; the browser preview is not used for solving.")}</div>
  </>
}

function metric(results: Record<string, unknown>, key: string, digits = 3) {
  const value = results[key]
  return typeof value === "number" ? value.toFixed(digits) : "—"
}

function ResultPanel({ language, session, busy }: { language: Language; session: SessionState; busy: boolean }) {
  const result = session.results
  const complete = session.stage === "COMPLETE"
  const simulated = result.result_kind === "robotwin_simulation" || result.result_kind === "offline_robotwin_replay"
  const expectedPoses = Number(result.poses_per_arm ?? session.artifacts.offline_replay?.poses_per_arm ?? 48)
  const rows = [
    [text(language, "左臂有效姿态", "Left accepted poses"), `${metric(result, "poses_left", 0)} / ${expectedPoses}`],
    [text(language, "右臂有效姿态", "Right accepted poses"), `${metric(result, "poses_right", 0)} / ${expectedPoses}`],
    [text(language, "左臂重投影中位误差", "Left median reprojection"), `${metric(result, "left_reprojection_median_px")} px`],
    [text(language, "右臂重投影中位误差", "Right median reprojection"), `${metric(result, "right_reprojection_median_px")} px`],
    [text(language, "第三视角重投影误差", "Third-view reprojection"), `${metric(result, "third_reprojection_px")} px`],
    [text(language, "标定板", "Target"), String(result.target_id ?? "—")],
  ]
  return <div className="mx-auto max-w-6xl">
    <div className="mb-6 flex items-start justify-between gap-4"><div><p className="text-xs font-medium uppercase tracking-[0.18em] text-zinc-500">{complete ? text(language, "校准完成", "Calibration complete") : text(language, "结果检查与导出", "Review and export")}</p><h1 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-950">{complete ? text(language, "校准完成", "Calibration complete") : text(language, "检查校准结果", "Review calibration result")}</h1><p className="mt-2 text-zinc-500">{session.artifacts.offline_replay ? session.instruction : language === "en" ? instructionLabels[session.stage] : session.instruction}</p></div><Badge ok={result.quality_status === "passed"}>● {result.quality_status === "passed" ? text(language, "质量门控通过", "Quality gates passed") : text(language, "等待结果", "Waiting for result")}</Badge></div>
    {simulated && <div className="mb-5 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800"><AlertTriangle className="mr-2 inline h-4 w-4"/>{text(language, "这是 RoboTwin 仿真回归结果，仅用于验证软件流程，不能作为真机外参使用。", "This is a RoboTwin simulation regression result. It validates the software flow and is not a physical calibration.")}</div>}
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      <Card><CardHeader><CardTitle className="text-sm text-zinc-500">{text(language, "双臂基座距离", "Dual-base distance")}</CardTitle></CardHeader><CardContent><div className="text-3xl font-semibold tracking-tight">{metric(result, "dual_base_distance_mm")} <span className="text-base font-normal text-zinc-400">mm</span></div></CardContent></Card>
      <Card><CardHeader><CardTitle className="text-sm text-zinc-500">{text(language, "第三视角相机高度", "Third-view camera height")}</CardTitle></CardHeader><CardContent><div className="text-3xl font-semibold tracking-tight">{metric(result, "third_camera_height_mm")} <span className="text-base font-normal text-zinc-400">mm</span></div></CardContent></Card>
      <Card><CardHeader><CardTitle className="text-sm text-zinc-500">{text(language, "第三视角俯仰角", "Third-view pitch")}</CardTitle></CardHeader><CardContent><div className="text-3xl font-semibold tracking-tight">{metric(result, "third_camera_pitch_deg")}<span className="text-base font-normal text-zinc-400">°</span></div></CardContent></Card>
    </div>
    <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[1.35fr_1fr]">
      <Card><CardHeader><CardTitle>{text(language, "质量摘要", "Quality summary")}</CardTitle></CardHeader><CardContent><div className="divide-y divide-zinc-100">{rows.map(([label, value]) => <div key={label} className="flex items-center justify-between py-3 text-sm"><span className="text-zinc-500">{label}</span><span className="font-medium text-zinc-900">{value}</span></div>)}</div></CardContent></Card>
      <Card><CardHeader><CardTitle>{text(language, "工作台几何", "Workcell geometry")}</CardTitle></CardHeader><CardContent className="space-y-4"><div className="rounded-lg bg-zinc-50 p-4"><div className="text-xs text-zinc-500">{text(language, "左基座距标定板平面高度", "Left base to target plane")}</div><div className="mt-1 text-xl font-semibold">{metric(result, "left_base_height_mm")} mm</div></div><div className="rounded-lg bg-zinc-50 p-4"><div className="text-xs text-zinc-500">{text(language, "右基座距标定板平面高度", "Right base to target plane")}</div><div className="mt-1 text-xl font-semibold">{metric(result, "right_base_height_mm")} mm</div></div><p className="text-xs leading-5 text-zinc-500">{text(language, "导出的 JSON 包含完整 4×4 外参矩阵、质量指标和 Session 元数据。", "The JSON export includes full 4×4 transforms, quality metrics and Session metadata.")}</p></CardContent></Card>
    </div>
  </div>
}

export function OperatorApp() {
  const [language, toggleLanguage] = useLanguage()
  const { session, connected, busy, error, start, command, hold, uploadBoardPhotos, useMeasuredBoardScale, runGeometry, startRobotwinPoseCheck, startRobotwinPlacement, confirmPlacement } = useCalibrationSession()
  const [confirmAction, setConfirmAction] = useState<"terminate" | "restart" | null>(null)
  const activeIndex = session ? stageOrder.indexOf(session.stage) : -1
  const isRunning = session?.status === "running"
  const isHolding = session?.status === "paused" || session?.status === "recovery_required"
  const isComplete = session?.stage === "COMPLETE" && session.status === "complete"
  const canExport = session?.stage === "REVIEW_AND_EXPORT" || isComplete
  const canRetry = Boolean(session && session.errors.length > 0)
  const confirm = () => {
    const action = confirmAction
    setConfirmAction(null)
    if (action) void command(action)
  }
  return <div className="min-h-screen overflow-x-hidden bg-zinc-50 text-zinc-900">
    <div className="flex min-h-screen">
      <aside className="relative w-64 shrink-0 border-r border-zinc-200 bg-white px-4 py-6 text-zinc-950"><div className="mb-8 flex items-center gap-2 text-lg font-semibold tracking-tight"><ShieldAlert className="h-5 w-5"/>{text(language, "Piper 工作站标定", "Piper Workcell Calibration")}</div><nav className="space-y-1">{stages(language).map((stage, index) => <div key={stage.id} className={`flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-colors ${session?.stage === stage.id ? "bg-zinc-100 font-medium text-zinc-950" : index < activeIndex ? "text-zinc-600" : "text-zinc-400"}`}><span className={`grid h-6 w-6 place-items-center rounded-full border text-xs ${session?.stage === stage.id ? "border-zinc-300 bg-white text-zinc-950" : "border-zinc-200 bg-zinc-50 text-zinc-500"}`}>{index + 1}</span>{stage.label}</div>)}</nav><div className="absolute bottom-5 text-xs text-zinc-400">v0.1.0 · development shadow</div></aside>
      <main className="flex min-w-0 flex-1 flex-col"><header className="flex h-16 items-center justify-end border-b border-zinc-200 bg-white/90 px-4 md:px-6"><div className="flex flex-wrap justify-end gap-2"><Button variant="outline" onClick={toggleLanguage}>{language === "zh" ? "English" : "中文"}</Button><Badge ok={connected}>● {text(language, "后端 ", "Backend ")}{connected ? text(language, "已连接", "Connected") : text(language, "断开", "Offline")}</Badge><Badge>● {text(language, "几何校准", "Calibration")}</Badge><Badge>{session?.artifacts.offline_replay ? "Offline replay" : session?.mode === "http" ? text(language, "真机", "Physical") : session?.mode === "robotwin" ? "RoboTwin" : "Mock"}</Badge></div></header>{session?.artifacts.offline_replay && <div className="border-b border-amber-200 bg-amber-50 px-6 py-2 text-sm text-amber-900">{text(language, "离线仿真回放：读取 BOARD_002 的 RGB 与 FK；不连接机器人或 D435。", "Offline simulation replay: recorded BOARD_002 RGB and FK; no robot or D435 is connected.")}</div>}<section className="min-w-0 flex-1 p-4 md:p-6">{!session ? <Card className="mx-auto mt-24 max-w-xl"><CardHeader><CardTitle className="text-xl">{text(language, "Piper 工作站标定", "Piper Workcell Calibration")}</CardTitle></CardHeader><CardContent><p className="mb-6 text-zinc-500">{text(language, "运行模式由校准服务配置决定。", "The operating mode is selected by the calibration service.")}</p><Button onClick={start} disabled={busy}>{text(language, "开始系统检查", "Start system check")}</Button></CardContent></Card> : <MainPanel language={language} session={session} busy={busy} command={(value) => void command(value)} uploadBoardPhotos={(files) => void uploadBoardPhotos(files)} useMeasuredBoardScale={() => void useMeasuredBoardScale()} runGeometry={() => void runGeometry()} startRobotwinPoseCheck={() => void startRobotwinPoseCheck()} startRobotwinPlacement={() => void startRobotwinPlacement()} confirmPlacement={() => void confirmPlacement()} />} {error && <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700"><AlertTriangle className="mr-2 inline h-4 w-4"/>{error}</div>}</section>
      <footer className="relative z-20 flex min-h-20 flex-wrap items-center justify-between gap-3 border-t border-zinc-200 bg-white/95 px-4 py-3 shadow-[0_-8px_30px_rgba(24,24,27,.08)] backdrop-blur md:px-6"><div className="flex gap-5 text-sm text-zinc-500"><span>{text(language, "机器人执行", "Robot motion")} <b className={(isHolding || isComplete) ? "text-amber-600" : "text-emerald-600"}>{(isHolding || isComplete) ? "HOLD" : text(language, "允许运行", "Ready")}</b></span><span>{text(language, "相机系统", "Cameras")} <b className="text-emerald-600">3 / 3</b></span></div><div className="flex flex-wrap gap-2">{canExport && <Button variant="outline" asChild><a href="/api/v1/session/export" download><Download className="mr-2 h-4 w-4"/>{text(language, "导出 JSON", "Export JSON")}</a></Button>}{isComplete ? <Button disabled={busy} onClick={() => setConfirmAction("restart")}><RotateCcw className="mr-2 h-4 w-4"/>{text(language, "重新校准", "Restart calibration")}</Button> : <>{canRetry && <Button variant="outline" disabled={busy} onClick={() => command("retry")}><RotateCcw className="mr-2 h-4 w-4"/>{text(language, "重试错误步骤", "Retry step")}</Button>}{isHolding ? <><Button disabled={busy} onClick={() => command("continue")}><Play className="mr-2 h-4 w-4"/>{text(language, "恢复机器人执行", "Resume robot")}</Button><Button variant="danger" disabled={busy} onClick={() => setConfirmAction("terminate")}><House className="mr-2 h-4 w-4"/>{text(language, "终止任务并回 Home", "Stop and go Home")}</Button></> : <><Button disabled={!session || busy || session.status === "aborted"} onClick={() => command("continue")}><Play className="mr-2 h-4 w-4"/>{text(language, "继续", "Continue")}</Button><Button variant="danger" disabled={!session || busy || !isRunning} onClick={() => void hold()}><Square className="mr-2 h-4 w-4"/>{text(language, "停止机器人执行", "Stop robot")}</Button></>}</>}</div></footer></main>
    </div>
    {confirmAction && <div className="fixed inset-0 z-50 grid place-items-center bg-black/45 p-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-labelledby="confirm-title"><Card className="w-full max-w-md shadow-2xl"><CardHeader><CardTitle id="confirm-title" className="text-lg">{confirmAction === "restart" ? "开始新的标定？" : "终止本次标定任务？"}</CardTitle></CardHeader><CardContent><p className="text-sm leading-6 text-zinc-500">{confirmAction === "restart" ? "当前结果会保留并仍可从 Session 目录读取。机械臂将先低速返回 Home，然后创建全新的标定 Session。" : "机械臂将低速返回 Home。确认执行完成后，本次进度才会关闭，并从系统检查重新开始。"}</p><div className="mt-6 flex justify-end gap-2"><Button variant="outline" onClick={() => setConfirmAction(null)}>取消</Button><Button variant={confirmAction === "restart" ? "default" : "danger"} onClick={confirm}><RotateCcw className="mr-2 h-4 w-4"/>{confirmAction === "restart" ? "确认重新校准" : "确认终止并回 Home"}</Button></div></CardContent></Card></div>}
  </div>
}
