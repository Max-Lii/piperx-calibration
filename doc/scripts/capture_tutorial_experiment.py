"""Record one reproducible 30+30 SAPIEN experiment for the guided tutorial.

Run from the project root. No robotcontrol service, CAN or RealSense is used.
Solver inputs contain only rendered images, camera intrinsics and measured FK.
The explicit texture-to-scene transform is used exclusively by the evaluator.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
from sim.robotwin.robotwin_dual_tabletop import DualTabletopHarness, _camera_matrix
from sim.robotwin.adapter import asset_root
from piperx_calibration.robotcontrol.camera import CalibrationFrame
from piperx_calibration.calibration.geometry import (
    estimate_camera_target, HandEyeObservation, solve_robot_world_hand_eye,
    hand_eye_validation, transform_error,
)
from piperx_calibration.calibration.quality import CharucoTargetSpec, evaluate_charuco
from piperx_calibration.calibration.run_robotwin_e2e import _geometry_quality

ROOT=Path(__file__).resolve().parents[2]
RUN_ID='BOARD_002_20261024_tutorial30_default'

def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    return x

def save_json(path,x): path.write_text(json.dumps(jsonable(x),indent=2,allow_nan=False)+'\n')
def save_rgb(path,rgb): cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def solve_records(records):
    """CPU-only solve reused by the offline UI; no scene truth is read here."""
    arms={}
    for arm in ('left','right'):
        rows=[r for r in records if r['arm']==arm and r['quality']['accepted']]
        obs=[HandEyeObservation(np.asarray(r['T_C_P']),np.asarray(r['T_B_E'])) for r in rows]
        solved=solve_robot_world_hand_eye(obs)
        validation=hand_eye_validation(obs)
        reproj=[r['quality']['reprojection_error_px'] for r in rows]
        arms[arm]={**solved,'accepted_poses':len(rows),
            'reprojection_error_px_median':float(np.median(reproj)),
            'reprojection_error_px_max':float(np.max(reproj)),
            'corner_count_min':min(r['corner_count'] for r in rows),
            'depth_consistency_median_mm':None,'validation':validation,
            'accepted_pose_ids':[r['pose_id'] for r in rows],
            'held_out_pose_ids':[r['pose_id'] for r in rows[4::5]]}
    third_record=next(r for r in records if r['third']['T_C_P'] is not None)
    tp=np.linalg.inv(np.asarray(third_record['third']['T_C_P']))
    # Height is a magnitude relative to the target plane. P +Z points down.
    forward=tp[:3,2]
    pitch=float(np.degrees(np.arctan2(forward[2],np.linalg.norm(forward[:2]))))
    result={'schema_version':1,'run_id':RUN_ID,'backend':'offline-sapien-rgb-fk',
        'target_id':'BOARD_002','poses_per_arm':30,'reference_frame':'P: OpenCV Charuco board',
        'left':arms['left'],'right':arms['right'],
        'third':{'T_world_camera':tp,'height_from_target_plane_mm':abs(float(tp[2,3]))*1000,
            'pitch_deg':pitch,'reprojection_error_px':third_record['third']['reprojection_error_px'],
            'depth_consistency_mm':None,'source_pose_id':third_record['pose_id']},
        'dual_base_distance_mm':float(np.linalg.norm(arms['left']['T_world_base'][:3,3]-arms['right']['T_world_base'][:3,3])*1000),
        'base_height_from_target_plane_mm':{a:abs(float(arms[a]['T_world_base'][2,3]))*1000 for a in arms},
        'rejected_poses':{a:[{'pose_id':r['pose_id'],'reasons':r['quality']['reasons']} for r in records if r['arm']==a and not r['quality']['accepted']] for a in arms}}
    result['quality']=_geometry_quality(arms['left'],arms['right'],result['third'],result['rejected_poses'],30)
    return jsonable(result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'doc/experiments'/RUN_ID);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'observations.json').exists():raise SystemExit('Refusing to overwrite a completed experiment; use a new --output directory')
    library=ROOT/'doc/data/BOARD_002_pose_libraries.json'
    h=DualTabletopHarness([],[],out,wanted=30,pose_library=library)
    target=CharucoTargetSpec.from_manifest(ROOT/'assets/targets/BOARD_002_A4_Charuco_5x7_35mm_metric.json')
    env=h.h._make_env();records=[];truth={};pairs=[]
    try:
        env.reset(init_id=0);task=env._task
        board,_=h._inject_board(task)
        # Explicit UV convention: texture rows point along scene -Y.
        t_s_p=np.diag([1.,-1.,-1.,1.]);t_s_p[:3,3]=[-.097,.0425,.742]
        for camera_id,data in task.get_obs()['observation'].items():save_rgb(out/('home_'+camera_id+'.png'),data['rgb'])
        for arm in ('left','right'):
            for index,q in enumerate(getattr(h,arm)):
                h._drive(task,q,arm)
                all_rgb=task.get_obs()['observation'];cfg=task.cameras.get_config()
                fk=h.h._list_pose_matrix(task.get_arm_pose(arm))
                base=h.h._pose_matrix(getattr(task.robot,arm+'_entity').get_pose())
                t_b_e=np.linalg.inv(base)@fk
                entity=getattr(task.robot,arm+'_entity');active=entity.get_active_joints()
                measured=[float(entity.get_qpos()[active.index(j)]) for j in getattr(task.robot,arm+'_arm_joints')]
                pose_id=f'{arm}_{index:02d}';rgb=all_rgb[arm+'_camera']['rgb'];k=np.asarray(cfg[arm+'_camera']['intrinsic_cv'])
                images={}
                for cid,cname in (('left_wrist','left_camera'),('right_wrist','right_camera'),('third_view','head_camera')):
                    filename=f'{pose_id}_{cid}.png';save_rgb(out/filename,all_rgb[cname]['rgb']);images[cid]=filename
                frame=CalibrationFrame(camera_id=arm+'_wrist',serial='SAPIEN-rendered-RGB',frame_number=len(records),
                    host_timestamp_monotonic_s=time.monotonic(),hardware_timestamp_ms=0,
                    rgb=rgb,depth_raw=np.zeros(rgb.shape[:2],np.uint16),depth_scale_m=.001,
                    color_intrinsics=k,distortion=np.zeros(5),distortion_model='none')
                quality=evaluate_charuco(frame,target)
                corners,ids,_,_=h.h.detector.detectBoard(rgb)
                t_c_p=None;metrics={}
                if ids is not None and len(ids)>=4:
                    try:t_c_p,metrics=estimate_camera_target(frame,target)
                    except (ValueError,cv2.error):pass
                if t_c_p is None:quality['accepted']=False
                headrgb=all_rgb['head_camera']['rgb'];hk=np.asarray(cfg['head_camera']['intrinsic_cv'])
                hf=CalibrationFrame(camera_id='third_view',serial='SAPIEN-rendered-RGB',frame_number=len(records),
                    host_timestamp_monotonic_s=time.monotonic(),hardware_timestamp_ms=0,rgb=headrgb,
                    depth_raw=np.zeros(headrgb.shape[:2],np.uint16),depth_scale_m=.001,color_intrinsics=hk,distortion=np.zeros(5),distortion_model='none')
                try:ht,hm=estimate_camera_target(hf,target)
                except (ValueError,cv2.error):ht=None;hm={}
                rec={'pose_id':pose_id,'sequence_index':len(records),'arm':arm,'arm_pose_index':index,
                    'command_qpos_rad':q,'measured_qpos_rad':measured,'T_B_E':t_b_e,'K':k,'distortion':[0]*5,
                    'image_size':[640,480],'images':images,'image_sha256':sha(out/images[arm+'_wrist']),
                    'charuco_ids':[] if ids is None else ids.reshape(-1).tolist(),
                    'corners_px':[] if corners is None else corners.reshape(-1,2).tolist(),
                    'corner_count':0 if ids is None else len(ids),'T_C_P':t_c_p,'quality':quality,
                    'depth_evidence':'not recorded; RGB/FK experiment only',
                    'third':{'K':hk,'T_C_P':ht,**hm}}
                records.append(jsonable(rec))
                cam_world=_camera_matrix(cfg[arm+'_camera'])
                truth[arm]={'T_P_base':np.linalg.inv(t_s_p)@base,
                    'T_ee_camera':np.linalg.inv(fk)@np.linalg.inv(cam_world)}
                if arm=='right' and index in (0,5,10,15,20,25):
                    observer=task.cameras.get_observer_rgb();save_rgb(out/(pose_id+'_observer.png'),observer)
                    # Image pair is captured without stepping physics between views.
                    pairs.append((pose_id,observer,rgb,rec))
                print(pose_id,quality['accepted'],rec['corner_count'],flush=True)
            h._drive(task,[0,.05,-.05,0,0,0],arm)
        truth['T_S_P']=t_s_p;truth['third_T_P_camera']=np.linalg.inv(t_s_p)@np.linalg.inv(_camera_matrix(task.cameras.get_config()['head_camera']))
        # Check known board orientation against actual image/PnP, not a guess.
        assert transform_error(np.asarray(records[-1]['third']['T_C_P']),np.linalg.inv(truth['third_T_P_camera']))[0]<5
        save_json(out/'observations.json',records)
        result=solve_records(records);errors={}
        for arm in ('left','right'):
            errors[arm+'_base']=transform_error(np.asarray(result[arm]['T_world_base']),truth[arm]['T_P_base'])
            errors[arm+'_handeye']=transform_error(np.asarray(result[arm]['T_ee_camera']),truth[arm]['T_ee_camera'])
        errors['third_camera']=transform_error(np.asarray(result['third']['T_world_camera']),truth['third_T_P_camera'])
        result['ground_truth_errors']={k:{'translation_mm':v[0],'rotation_deg':v[1]} for k,v in errors.items()}
        result['depth_validation']='not available in this RGB-only simulated experiment'
        save_json(out/'truth.json',truth);save_json(out/'result.json',result)
        trace=next(r for r in records if r['pose_id']=='right_05');save_json(out/'right_05.json',trace)
        accepted=[r for r in records if r['arm']=='right' and r['quality']['accepted']]
        train=[r for i,r in enumerate(accepted) if i%5!=4];held=accepted[4]
        sol=solve_robot_world_hand_eye([HandEyeObservation(np.array(r['T_C_P']),np.array(r['T_B_E'])) for r in train])
        predicted=np.linalg.inv(sol['T_world_base']@np.array(held['T_B_E'])@sol['T_ee_camera'])
        predicted_pixels,_=cv2.projectPoints(target.board().getChessboardCorners()[held['charuco_ids']],cv2.Rodrigues(predicted[:3,:3])[0],predicted[:3,3],np.array(held['K']),None)
        overlay=cv2.imread(str(out/held['images']['right_wrist']))
        for measured,pred in zip(held['corners_px'],predicted_pixels.reshape(-1,2),strict=True):
            cv2.circle(overlay,tuple(np.rint(measured).astype(int)),3,(0,180,0),1)
            cv2.drawMarker(overlay,tuple(np.rint(pred).astype(int)),(0,0,255),cv2.MARKER_CROSS,7,1)
        cv2.imwrite(str(out/'heldout_prediction.png'),overlay)
        save_json(out/'heldout_example.json',{'pose_id':held['pose_id'],'excluded_pose_ids':[r['pose_id'] for r in accepted[4::5]],'training_pose_ids':[r['pose_id'] for r in train],
            'T_C_P_measured':held['T_C_P'],'T_C_P_predicted':predicted,'pose_error':transform_error(predicted,np.array(held['T_C_P'])),
            'observed_pixels':held['corners_px'],'predicted_pixels':predicted_pixels.reshape(-1,2)})
        # Contact sheet: each row pairs observer and wrist from exactly one state.
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
        small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        canvas=Image.new('RGB',(1320,6*550),(248,249,251));draw=ImageDraw.Draw(canvas)
        for row,(pid,observer,wrist,rec) in enumerate(pairs):
            y=row*550; draw.text((20,y+8),pid+' | observer and wrist at the same recorded state',fill='#222222',font=font)
            canvas.paste(Image.fromarray(observer),(10,y+43));canvas.paste(Image.fromarray(wrist),(670,y+43))
            distance=np.linalg.norm(np.array(rec['T_C_P'])[:3,3]) if rec['T_C_P'] is not None else 0
            draw.text((20,y+522),f"{rec['corner_count']} corners | target-origin range {distance:.3f} m | border {rec['quality']['minimum_border_px']:.1f} px",fill='#444444',font=small)
        canvas.save(out/'paired_poses.png')
        # Annotate right_05 with actual detected corner IDs.
        img=cv2.imread(str(out/trace['images']['right_wrist']))
        cv2.aruco.drawDetectedCornersCharuco(img,np.array(trace['corners_px'],np.float32).reshape(-1,1,2),np.array(trace['charuco_ids'],np.int32).reshape(-1,1))
        cv2.imwrite(str(out/'right_05_detection.png'),img)
        save_json(out/'provenance.json',{'run_id':RUN_ID,'date':'2026-10-24','gpu':'CUDA_VISIBLE_DEVICES=0','renderer':'default','near_m':.01,
            'pose_library_sha256':sha(library),'poses_per_arm':30,'sequence':'left then right; non-acquiring arm parked; board fixed',
            'URDF_sha256':{a:sha(asset_root()/f'embodiments/piper_x_{a}/piper_x.urdf') for a in ('left','right')},
            'camera_config_sha256':{a:sha(asset_root()/f'embodiments/piper_x_{a}/config.yml') for a in ('left','right')},
            'source_sha256':sha(Path(__file__)),'depth_recorded':False,
            'historical_note':'New experiment. It does not overwrite the older 30+30 result in doc/data.'})
        print(json.dumps(result['ground_truth_errors'],indent=2))
    finally:env.close()
if __name__=='__main__':main()
