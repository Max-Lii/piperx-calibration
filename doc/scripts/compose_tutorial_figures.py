"""Re-layout four archived RGB pairs without changing any experiment record.

Writes only derived documentation assets under doc/source-images. The observer
uses the same crop in every panel; the measurement wrist image is uncropped.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'doc/experiments/BOARD_002_20261024_tutorial30_default'
OUT=ROOT/'doc/source-images'
FONT=Path('/usr/share/fonts/truetype/dejavu')
def font(size,bold=False):
    return ImageFont.truetype(str(FONT/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),size)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    rows={r['pose_id']:r for r in json.loads((RUN/'observations.json').read_text())}
    canvas=Image.new('RGB',(2600,1610),'#F5F7FB')
    d=ImageDraw.Draw(canvas)
    d.text((52,26),'Pose diversity: one fixed board, four different wrist views',font=font(48,True),fill='#17253C')
    d.text((52,94),'BOARD_002  |  Right-arm RGB / measured FK  |  Each observer view is paired with the same recorded wrist state',font=font(31),fill='#52637C')
    crop=(120,0,560,330)
    evidence=[]
    for i,pid in enumerate(('right_00','right_05','right_10','right_15')):
        row=rows[pid]
        x=44+(i%2)*1278;y=160+(i//2)*674
        w,h=1234,650
        d.rounded_rectangle((x,y,x+w,y+h),radius=14,fill='white',outline='#D7DFE9',width=2)
        d.text((x+24,y+12),f'({chr(97+i)})  {pid}',font=font(42,True),fill='#17253C')
        d.line((x+20,y+71,x+w-20,y+71),fill='#D5DEE9',width=2)
        d.text((x+25,y+84),'Observer · same crop',font=font(30),fill='#52637C')
        d.text((x+645,y+84),'Wrist RGB · full frame',font=font(30),fill='#52637C')
        observer=RUN/(pid+'_observer.png');wrist=RUN/row['images']['right_wrist']
        assert sha(wrist)==row['image_sha256']
        for path,is_observer,dx in ((observer,True,24),(wrist,False,644)):
            im=Image.open(path).convert('RGB')
            if is_observer:im=im.crop(crop)
            # Both sources are 4:3; preserve axes, orientation, and aspect ratio.
            im=im.resize((566,425),Image.Resampling.LANCZOS)
            canvas.paste(im,(x+dx,y+128))
            d.rectangle((x+dx,y+128,x+dx+566,y+128+425),outline='#D9E0E8',width=1)
        T=np.asarray(row['T_C_P'])
        distance=float(np.linalg.norm(T[:3,3]))
        tilt=float(np.degrees(np.arccos(np.clip(abs(T[2,2]),0,1))))
        d.text((x+25,y+582),f"{row['corner_count']} corners   |   range {distance:.3f} m   |   tilt {tilt:.1f}°",font=font(36),fill='#17253C')
        evidence.append({'pose_id':pid,'observer':str(observer.relative_to(ROOT/'doc')),
            'wrist':str(wrist.relative_to(ROOT/'doc')),'observer_sha256':sha(observer),
            'wrist_sha256':sha(wrist),'observer_crop_xyxy':crop,'wrist_crop':None,
            'range_to_P_origin_m':distance,'normal_tilt_deg':tilt,'corner_count':row['corner_count']})
    d.text((52,1538),'Compare end orientation, board tilt and image coverage. Rotational diversity supplies non-redundant hand–eye constraints.',font=font(31),fill='#52637C')
    canvas.save(OUT/'board002_pose_diversity.png')
    provenance={'run_id':RUN.name,'source_records_unchanged':True,'source_pose_ids':[e['pose_id'] for e in evidence],
        'layout':'2 columns by 2 rows; identical observer crop; full wrist frames',
        'canvas_px':list(canvas.size),'generator':'doc/scripts/compose_tutorial_figures.py',
        'reference_design':'doc/scripts/prepare_figures.py','pairs':evidence}
    (OUT/'board002_pose_diversity.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'image':str(OUT/'board002_pose_diversity.png'),'pose_ids':provenance['source_pose_ids']}))
if __name__=='__main__':main()
