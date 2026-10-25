"""Offline browser walkthrough driven by the recorded tutorial experiment.

This is a separate demo server: it creates no RobotControlClient, motion backend,
CAN interface or physical camera. Images and FK come from the same saved run;
ChArUco/PnP and hand-eye are recomputed when the operator starts replay.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, threading, time
from pathlib import Path
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from piperx_calibration.calibration.adapter import MockWorkcellAdapter
from piperx_calibration.calibration.workflow import CalibrationWorkflow, WorkflowError
from piperx_calibration.calibration.schemas import CommandRequest, Stage
from piperx_calibration.calibration.store import SessionStore
from piperx_calibration.calibration.geometry import estimate_camera_target
from piperx_calibration.calibration.quality import CharucoTargetSpec, evaluate_charuco
from piperx_calibration.robotcontrol.camera import CalibrationFrame
from capture_tutorial_experiment import ROOT, RUN_ID, solve_records

class ReplayAdapter(MockWorkcellAdapter):
    def __init__(self, directory, rows):
        super().__init__();self.directory=directory;self.current=rows[0];self.lock=threading.Lock()
    def select(self,row):
        with self.lock:self.current=row
    def frame_jpeg(self,camera_id):
        with self.lock:filename=self.current['images'][camera_id]
        image=cv2.imread(str(self.directory/filename))
        ok,encoded=cv2.imencode('.jpg',image)
        if not ok:raise RuntimeError('Cannot encode recorded preview')
        return encoded.tobytes(),time.monotonic()
    def system_check(self):
        devices=super().system_check()
        for d in devices:d.detail='OFFLINE: recorded simulation; no hardware connected'
        return devices

class ReplayWorkflow(CalibrationWorkflow):
    @staticmethod
    def _mock_results():
        raise WorkflowError('Offline replay must compute its own result; no representative fallback is allowed')
    def start(self,mode=None):
        state=super().start(mode)
        with self._lock:
            self._state.artifacts['offline_replay']={'run_id':RUN_ID,'poses_per_arm':30,'depth_recorded':False}
            return self._commit('offline_replay_identified')
    def command(self, request):
        super().command(request)
        with self._lock:
            state=self._state
            notes={
                Stage.SYSTEM_CHECK:'Read recorded camera frames; no physical devices are connected.',
                Stage.TARGET_VERIFY:'Review the BOARD_002 virtual target dimensions recorded in this experiment.',
                Stage.SAFETY_DRY_RUN:'Review the 60 recorded poses. This replay performs no motion safety check.',
                Stage.A4_PLACEMENT:'Inspect the fixed target in the recorded third view; no robot moves here.',
                Stage.WORLD_REGISTRATION:'Continue to replay recorded RGB and measured FK, then recompute the geometry.',
                Stage.VALIDATE_GEOMETRY:'Inspect the recomputed quality checks. Depth and hardware safety remain untested.',
                Stage.REVIEW_AND_EXPORT:'Review and export the recomputed offline result; reference frame is OpenCV board P.',
                Stage.COMPLETE:'Offline replay complete. This export is simulation evidence, not physical calibration.'}
            if state.stage in notes:state.instruction=notes[state.stage]
            if state.stage==Stage.TARGET_VERIFY:
                state.artifacts['board_verification']={'accepted':True,'virtual':True,'valid_images':0,'total_images':0,
                    'scale_x':1.,'scale_y':1.,'verification_method':'recorded_simulation_manifest'}
            if state.stage==Stage.SAFETY_DRY_RUN:
                state.progress.current=0;state.progress.total=60
                state.progress.label='60 recorded poses; no motion safety check in offline replay'
            return self._commit('offline_stage_explained')
    def solving(self):
        with self._lock:
            self._state.stage=Stage.SOLVE_GEOMETRY
            self._state.instruction='Recomputing hand-eye and held-out checks from recorded RGB and FK.'
            return self._commit('offline_solver_started')
    def record_geometry_results(self,raw,**kwargs):
        super().record_geometry_results(raw,**kwargs)
        with self._lock:
            self._state.results['result_kind']='offline_robotwin_replay'
            self._state.results['poses_per_arm']=30
            self._state.results['reference_frame']=raw['reference_frame']
            self._state.warnings=raw['quality']['warnings']
            return self._commit('offline_result_identified')

def make_app(directory:Path,session_root:Path,delay:float=1.0):
    rows=json.loads((directory/'observations.json').read_text())
    adapter=ReplayAdapter(directory,rows);workflow=ReplayWorkflow(SessionStore(session_root),adapter,'mock')
    target=CharucoTargetSpec.from_manifest(ROOT/'assets/targets/BOARD_002_A4_Charuco_5x7_35mm_metric.json')
    app=FastAPI(title='BOARD_002 offline tutorial replay')
    @app.exception_handler(WorkflowError)
    async def error(_,exc):return JSONResponse(status_code=409,content={'detail':str(exc)})
    @app.get('/api/v1/health')
    def health():return {'status':'ok','mode':'offline_replay','run_id':RUN_ID,'hardware_connected':False}
    @app.get('/api/v1/session')
    def session():return workflow.snapshot()
    @app.post('/api/v1/session/start')
    def start():return workflow.start()
    @app.post('/api/v1/session/command')
    def command(request:CommandRequest):
        current=workflow.snapshot()
        if current and current.stage==Stage.WORLD_REGISTRATION and request.command=='continue':
            if request.expected_revision!=current.revision:raise WorkflowError('Stale revision')
            started=workflow.begin_geometry(total=60)
            def worker():
                try:
                    measured=[]
                    for row in rows:
                        state=workflow.snapshot()
                        if state.status!='running':raise WorkflowError('Offline replay interrupted; restart to repeat the case')
                        adapter.select(row)
                        filename=directory/row['images'][row['arm']+'_wrist']
                        if hashlib.sha256(filename.read_bytes()).hexdigest()!=row['image_sha256']:
                            raise ValueError('Recorded RGB checksum mismatch')
                        rgb=cv2.cvtColor(cv2.imread(str(filename)),cv2.COLOR_BGR2RGB)
                        frame=CalibrationFrame(camera_id=row['arm']+'_wrist',serial='recorded-SAPIEN',frame_number=row['sequence_index'],
                            host_timestamp_monotonic_s=time.monotonic(),hardware_timestamp_ms=0,rgb=rgb,
                            depth_raw=np.zeros(rgb.shape[:2],np.uint16),depth_scale_m=.001,color_intrinsics=np.array(row['K']),
                            distortion=np.array(row['distortion']),distortion_model='none')
                        quality=evaluate_charuco(frame,target);updated=dict(row);updated['quality']=quality
                        if quality['accepted']:updated['T_C_P']=estimate_camera_target(frame,target)[0].tolist()
                        third_rgb=cv2.cvtColor(cv2.imread(str(directory/row['images']['third_view'])),cv2.COLOR_BGR2RGB)
                        third_frame=CalibrationFrame(camera_id='third_view',serial='recorded-SAPIEN',frame_number=row['sequence_index'],
                            host_timestamp_monotonic_s=time.monotonic(),hardware_timestamp_ms=0,rgb=third_rgb,
                            depth_raw=np.zeros(third_rgb.shape[:2],np.uint16),depth_scale_m=.001,
                            color_intrinsics=np.array(row['third']['K']),distortion=np.zeros(5),distortion_model='none')
                        try:
                            third_pose,third_metrics=estimate_camera_target(third_frame,target)
                            updated['third']={'K':row['third']['K'],'T_C_P':third_pose.tolist(),**third_metrics}
                        except (ValueError,cv2.error):
                            updated['third']={'K':row['third']['K'],'T_C_P':None}
                        measured.append(updated)
                        index=row['arm_pose_index']+1
                        workflow.update_geometry(arm=row['arm'],index=index,poses_per_arm=30,
                            phase=f"OFFLINE {row['pose_id']}: {quality['corner_count']} corners; {'accepted' if quality['accepted'] else 'rejected'}")
                        with workflow._lock:
                            workflow._state.progress.label=f"OFFLINE {row['arm']} {index}/30 · {row['pose_id']}: {quality['corner_count']} corners; {'accepted' if quality['accepted'] else 'rejected'}"
                            workflow._state.instruction=workflow._state.progress.label
                            workflow._commit('offline_observation_explained')
                        time.sleep(delay)
                    workflow.solving();result=solve_records(measured)
                    (session_root/'recomputed_result.json').write_text(json.dumps(result,indent=2))
                    workflow.record_geometry_results(result)
                except Exception as exc:workflow.fail_geometry(f'Offline replay failed: {exc}')
            threading.Thread(target=worker,name='offline-replay',daemon=True).start()
            return started
        if current and current.artifacts.get('geometry_operation',{}).get('state')=='running':
            raise WorkflowError('Wait for the recorded sequence and solver to finish')
        return workflow.command(request)
    @app.post('/api/v1/safety/hold')
    def hold():return workflow.hold()
    @app.post('/api/v1/board-verification/use-measured-scale')
    def board():
        return workflow.record_board_verification({'accepted':True,'virtual':True,'target_id':'BOARD_002',
            'square_length_mm':35.,'scale_x':1.,'scale_y':1.,'valid_images':0,'total_images':0,
            'verification_method':'recorded_simulation_manifest'})
    @app.get('/api/v1/session/export')
    def export():
        state=workflow.snapshot()
        if state is None or state.stage not in (Stage.REVIEW_AND_EXPORT,Stage.COMPLETE):raise HTTPException(409,'Not ready')
        return JSONResponse(state.model_dump(mode='json'),headers={'Content-Disposition':'attachment; filename="offline_replay_result.json"'})
    @app.get('/api/v1/cameras/{camera_id}/stream')
    async def stream(camera_id:str):
        if camera_id not in ('left_wrist','third_view','right_wrist'):raise HTTPException(404)
        async def frames():
            while True:
                payload,_=adapter.frame_jpeg(camera_id)
                yield b'--frame\r\nContent-Type: image/jpeg\r\n'+f'Content-Length: {len(payload)}\r\n\r\n'.encode()+payload+b'\r\n'
                await asyncio.sleep(.125)
        return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame')
    @app.websocket('/ws/operator')
    async def websocket(socket:WebSocket):
        await socket.accept()
        try:
            while True:
                state=workflow.snapshot()
                await socket.send_json({'type':'session','session':state.model_dump(mode='json') if state else None})
                await asyncio.sleep(.25)
        except (WebSocketDisconnect,RuntimeError):pass
    app.mount('/assets',StaticFiles(directory=ROOT/'frontend/dist/assets'))
    @app.get('/operator')
    def operator():return FileResponse(ROOT/'frontend/dist/index.html',headers={'Cache-Control':'no-cache'})
    return app

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8020);p.add_argument('--frame-delay',type=float,default=1.0)
    p.add_argument('--data',type=Path,default=ROOT/'doc/experiments'/RUN_ID)
    p.add_argument('--session-root',type=Path,default=ROOT/'temp/tutorial_replay_sessions')
    a=p.parse_args();uvicorn.run(make_app(a.data,a.session_root,delay=a.frame_delay),host='127.0.0.1',port=a.port,log_level='warning')
