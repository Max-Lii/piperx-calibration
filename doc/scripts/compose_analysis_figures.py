"""Compose technical figures from archived RGB and corner measurements.

Uses the palette, typography and panel hierarchy of prepare_figures.py.
Writes derived assets only; no rendering, hardware access or detector edits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'doc'
RUN = PUBLIC / 'experiments/BOARD_002_20261024_tutorial30_default'
OUT = PUBLIC / 'images'
INK, MUTED, BLUE = '#17253c', '#52637c', '#1669ba'
ORANGE, BORDER, BG = '#c95b24', '#d7dfe9', '#f5f7fb'


def font(size, bold=False):
    return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plate(title, subtitle, height=1120):
    im = Image.new('RGB', (1800, height), BG)
    d = ImageDraw.Draw(im)
    d.text((52, 30), title, fill=INK, font=font(38, True))
    d.text((52, 90), subtitle, fill=MUTED, font=font(25))
    return im, d


def panel(im, box, title):
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(box, radius=12, fill='white', outline=BORDER, width=2)
    d.text((box[0]+20, box[1]+17), title, fill=INK, font=font(27, True))


def fit(im, source, box, nearest=False):
    """Return the exact affine pixel-center map for a displayed RGB crop."""
    x, y, right, bottom = box
    scale = min((right-x)/source.width, (bottom-y)/source.height)
    size = (round(source.width*scale), round(source.height*scale))
    offset = np.array([x+(right-x-size[0])//2, y+(bottom-y-size[1])//2])
    im.paste(source.resize(size, Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS), tuple(offset))
    return np.array(size)/source.size, offset


def mapped(points, low, scale, offset):
    # OpenCV integer coordinates identify pixel centers, PIL crop uses boundaries.
    return (np.asarray(points)-low+0.5)*scale-0.5+offset


def source_entry(path, crop=None):
    return {'path':str(path.relative_to(PUBLIC)), 'sha256':sha(path), 'crop_xyxy':crop}


def save(im, name, evidence):
    im.save(OUT/f'{name}.png')
    payload = {'run_id':RUN.name, 'generator':'doc/scripts/compose_analysis_figures.py',
               'canvas_px':list(im.size), 'reference_design':'doc/scripts/prepare_figures.py', **evidence}
    (OUT/f'{name}.provenance.json').write_text(json.dumps(payload,indent=2)+'\n')


def detection(rows):
    row = rows['right_05']
    path = RUN/row['images']['right_wrist']
    assert sha(path) == row['image_sha256']
    source = Image.open(path).convert('RGB')
    points = np.asarray(row['corners_px'])
    ids = row['charuco_ids']
    # A tight board-region crop permits nineteen compact labels without overlap.
    crop = (18, 195, 248, 440)
    im,d = plate('ChArUco: one observation, nineteen correspondences',
                 'right_05  |  BOARD_002  |  Recorded RGB: 640 × 480  |  19 / 24 internal corners')
    panel(im,(52,160,710,950),'(a)  Original wrist RGB')
    panel(im,(758,160,1748,950),'(b)  Detected internal corner IDs')
    scale,offset = fit(im,source,(66,230,696,925))
    low = np.array(crop[:2]); high = np.array(crop[2:])
    box = np.concatenate((low*scale+offset,high*scale+offset)).tolist()
    d.rectangle(box,outline=BLUE,width=3)
    scale,offset = fit(im,source.crop(crop),(780,230,1726,925))
    centers = mapped(points,low,scale,offset)
    label_boxes=[]
    for (x,y),index in zip(centers,ids):
        d.ellipse((x-6,y-6,x+6,y+6),fill=BLUE,outline='white',width=2)
        label=str(index); f=font(23,True); width=d.textlength(label,font=f)
        rect=(x+10,y-37,x+width+27,y-3)
        d.rounded_rectangle(rect,5,fill='white',outline='#b6cde5',width=2)
        d.text((x+18,y-35),label,fill=BLUE,font=f)
        label_boxes.append(rect)
    # Keep the geometry and label evidence machine-readable as well as visible.
    for i,a in enumerate(label_boxes):
        for b in label_boxes[i+1:]:
            assert a[2]<=b[0] or b[2]<=a[0] or a[3]<=b[1] or b[3]<=a[1], 'ID labels overlap'
    d.text((52,980),'ID k  →  board.getChessboardCorners()[k]  →  known 3D point in P',fill=INK,font=font(29))
    d.text((52,1035),f"Blue: actual detected corners   |   PnP reprojection RMS: {row['quality']['reprojection_error_px']:.3f} px",fill=MUTED,font=font(24))
    d.text((52,1075),'Missing IDs are not inferred. This right-wrist record is separate from the earlier 24-corner left-wrist example.',fill=MUTED,font=font(22))
    save(im,'board002_right05_correspondences',{'sources':[source_entry(path),source_entry(path,crop)],
         'pose_id':'right_05','detected_ids':ids,'corners_px':points.tolist(),
         'label_boxes_canvas':label_boxes,'reprojection_rms_px':row['quality']['reprojection_error_px']})


def heldout(rows):
    h=json.loads((RUN/'heldout_example.json').read_text())
    row=rows[h['pose_id']]
    assert h['pose_id'] not in h['training_pose_ids']
    observed=np.asarray(h['observed_pixels']); predicted=np.asarray(h['predicted_pixels'])
    assert np.array_equal(observed,np.asarray(row['corners_px']))
    # Confirm archived predictions actually come from the stored held-out pose.
    board=cv2.aruco.CharucoBoard((5,7),.035,.02625,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    T=np.asarray(h['T_C_P_predicted'])
    projection,_=cv2.projectPoints(board.getChessboardCorners()[row['charuco_ids']],
        cv2.Rodrigues(T[:3,:3])[0],T[:3,3],np.asarray(row['K']),np.asarray(row['distortion']))
    assert np.allclose(projection.reshape(-1,2),predicted,atol=1e-4,rtol=0)
    residual=np.linalg.norm(predicted-observed,axis=1)
    rms=float(np.sqrt(np.mean(residual**2))); maximum=float(residual.max())
    selected=np.argsort(residual)[[0,len(residual)//2,-1]].tolist()
    path=RUN/row['images']['right_wrist'];source=Image.open(path).convert('RGB')
    im,d=plate('Held-Out Pose Validation',
        'right_04  |  Excluded from fitting: 6 poses  |  Training set: 24 poses  |  Matched corners: 24',1200)
    panel(im,(52,160,1020,960),'(a)  New pose: all matched image points')
    scale,offset=fit(im,source,(72,230,1000,926))
    obs=mapped(observed,np.zeros(2),scale,offset);pred=mapped(predicted,np.zeros(2),scale,offset)
    for (x,y),(a,b) in zip(obs,pred):
        d.ellipse((x-6,y-6,x+6,y+6),outline=BLUE,width=2)
        d.line((a-5,b,a+5,b),fill=ORANGE,width=2);d.line((a,b-5,a,b+5),fill=ORANGE,width=2)
    details=[]
    for k,(idx,rank) in enumerate(zip(selected,('Smallest residual','Middle-ranked residual','Largest residual'))):
        top=160+k*270
        panel(im,(1056,top,1748,top+252),f'({chr(98+k)})  Corner ID {row["charuco_ids"][idx]}')
        center=np.floor(observed[idx]).astype(int)
        crop=(int(center[0]-6),int(center[1]-5),int(center[0]+6),int(center[1]+5))
        # 12×10 source pixels enlarged 20×, without interpolation or displacement.
        scale,offset=fit(im,source.crop(crop),(1076,top+51,1316,top+251),nearest=True)
        assert np.array_equal(scale,[20,20])
        for j in range(1,12):d.line((offset[0]+20*j-.5,offset[1],offset[0]+20*j-.5,offset[1]+199),fill='#d4dce6',width=1)
        for j in range(1,10):d.line((offset[0],offset[1]+20*j-.5,offset[0]+239,offset[1]+20*j-.5),fill='#d4dce6',width=1)
        (x,y),(a,b)=mapped([observed[idx],predicted[idx]],np.array(crop[:2]),scale,offset)
        d.ellipse((x-17,y-17,x+17,y+17),outline=BLUE,width=3)
        d.line((a-16,b,a+16,b),fill=ORANGE,width=3);d.line((a,b-16,a,b+16),fill=ORANGE,width=3)
        d.text((1340,top+74),rank,fill=MUTED,font=font(21))
        d.text((1340,top+116),f'{residual[idx]:.3f} px',fill=INK,font=font(32,True))
        d.text((1340,top+169),'12 × 10 px crop · 20×',fill=MUTED,font=font(22))
        # Exact crop box on the context view, no corner position is moved.
        scale0,offset0=fit_map(source,(72,230,1000,926))
        box=np.concatenate((np.array(crop[:2])*scale0+offset0,np.array(crop[2:])*scale0+offset0)).tolist()
        d.rectangle(box,outline=BLUE,width=2)
        d.text((box[2]+7,box[1]-8),chr(98+k),fill=BLUE,font=font(25,True))
        details.append({'corner_id':row['charuco_ids'][idx],'residual_rank':rank,'crop_xyxy':crop,
            'magnification':20,'observed_px':observed[idx].tolist(),'predicted_px':predicted[idx].tolist(),
            'residual_px':float(residual[idx])})
    d.ellipse((54,995,78,1019),outline=BLUE,width=3)
    d.text((90,990),'Detected',fill=INK,font=font(26))
    d.line((280,1007,308,1007),fill=ORANGE,width=3);d.line((294,993,294,1021),fill=ORANGE,width=3)
    d.text((322,990),'Predicted from training-only extrinsics + held-out FK',fill=INK,font=font(26))
    d.text((52,1042),f'Image residual: RMS {rms:.3f} px / max {maximum:.3f} px     |     Pose difference (T_C_P): {h["pose_error"][0]:.3f} mm / {h["pose_error"][1]:.3f}°',fill=INK,font=font(25))
    d.text((52,1095),'The zoom enlarges the source pixels and marker coordinates together. No residual is artificially separated.',fill=MUTED,font=font(23))
    d.text((52,1140),'Image residuals compare pixels; the pose difference compares two 3D transforms. Neither is a Ground Truth score.',fill=MUTED,font=font(23))
    save(im,'board002_right04_heldout',{'sources':[source_entry(path),source_entry(RUN/'heldout_example.json')],
        'pose_id':row['pose_id'],'training_pose_ids':h['training_pose_ids'],'excluded_pose_ids':h['excluded_pose_ids'],
        'matched_ids':row['charuco_ids'],'pixel_residuals':residual.tolist(),'pixel_rms':rms,'pixel_max':maximum,
        'pose_error_T_C_P_mm_deg':h['pose_error'],'zoom_details':details,
        'zoom_resampling':'nearest; identical affine transform for image and marker centers'})


def fit_map(source, box):
    x,y,right,bottom=box;scale=min((right-x)/source.width,(bottom-y)/source.height)
    size=np.round(np.array(source.size)*scale).astype(int)
    return size/np.array(source.size),np.array([x+(right-x-size[0])//2,y+(bottom-y-size[1])//2])


def ui_plates():
    """Record the sources for lossless WebP publication of original UI captures."""
    for name,filename in [('placement','04-placement'),('acquisition','06-capture-right'),('review','08-review')]:
        path=RUN/'ui'/f'{filename}.png'
        state=json.loads((RUN/'ui'/f'{filename}.json').read_text())
        evidence={'run_id':RUN.name,'generator':'doc/scripts/capture_replay_ui.py',
            'capture_stage':state['stage'],'canvas_px':list(Image.open(path).size),
            'source':source_entry(path),'state_path':str((RUN/'ui'/f'{filename}.json').relative_to(PUBLIC)),
            'browser_viewport_px':[1600,900],'full_page':True,'screenshot_state_edited':False}
        (OUT/f'board002_ui_{name}.provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')


def main():
    rows={r['pose_id']:r for r in json.loads((RUN/'observations.json').read_text())}
    detection(rows)
    heldout(rows)
    ui_plates()
    print('Wrote two PNG technical figures and provenance for three original UI screenshots. Run optimize_images.py to publish WebP.')


if __name__=='__main__':
    main()
