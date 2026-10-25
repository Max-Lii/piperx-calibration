"""Check the two single-page tutorials and archived evidence after build.

Only explicitly marked runnable examples execute. No renderer, hardware client,
CAN device, camera, or motion API is opened by this check.
"""
from __future__ import annotations
import ast,hashlib,json,os,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'doc';RUN=DOC/'experiments/BOARD_002_20261024_tutorial30_default'

def snippets(s,lang):return re.findall(r'```'+lang+r'([^\n]*)\n(.*?)\n```',s,re.S)
def math(s):return [re.sub(r'\s+','',re.sub(r'\\text\{[^}]*\}',r'\\text{LABEL}',x)) for x in re.findall(r'\$\$(.*?)\$\$',s,re.S)]
def figures(s):return re.findall(r'<Figure\s+src="([^"]+)"',s)
def headings(s,depth):return re.findall(r'^'+'#'*depth+r' (.+)',s,re.M)
def tables(s):return [re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?',l) for l in s.splitlines() if l.startswith('|')]

def check_analysis_figures(rows):
    import cv2
    from PIL import Image
    from piperx_calibration.calibration.geometry import HandEyeObservation,solve_robot_world_hand_eye
    by_id={r['pose_id']:r for r in rows}
    manifests={}
    names=['board002_right05_correspondences','board002_right04_heldout',
           'board002_ui_placement','board002_ui_acquisition','board002_ui_review']
    for name in names:
        p=DOC/'source-images'/name
        m=json.loads(p.with_suffix('.provenance.json').read_text());manifests[name]=m
        assert m['run_id']==RUN.name
        assert list(Image.open(DOC/'public/images'/f'{name}.webp').size)==m['canvas_px']
        entries = m['sources'] if 'sources' in m else [m['source']]
        for entry in entries:
            source=DOC/entry['path']
            assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['sha256']
            if entry['crop_xyxy']:
                x,y,right,bottom=entry['crop_xyxy'];w,h=Image.open(source).size
                assert 0<=x<right<=w and 0<=y<bottom<=h
        if name.startswith('board002_ui_'):
            assert Image.open(DOC/'public/images'/f'{name}.webp').convert('RGBA').tobytes()==Image.open(source).convert('RGBA').tobytes()
            assert entry['crop_xyxy'] is None and not m['screenshot_state_edited']
            state=json.loads((DOC/m['state_path']).read_text())
            assert state['stage']==m['capture_stage']
            if name=='board002_ui_acquisition':
                assert state['progress']['current']==36 and state['progress']['total']==60
                assert 'right_05: 19 corners; accepted' in state['progress']['label']
    m=manifests['board002_right05_correspondences'];r=by_id['right_05']
    assert m['detected_ids']==r['charuco_ids'] and m['corners_px']==r['corners_px']
    assert m['reprojection_rms_px']==r['quality']['reprojection_error_px']
    # Independently redetect the nineteen IDs from the untouched wrist RGB.
    board=cv2.aruco.CharucoBoard((5,7),.035,.02625,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    rgb=cv2.cvtColor(cv2.imread(str(RUN/r['images']['right_wrist'])),cv2.COLOR_BGR2RGB)
    corners,ids,_,_=cv2.aruco.CharucoDetector(board).detectBoard(rgb)
    assert ids.ravel().tolist()==m['detected_ids']
    assert np.allclose(corners.reshape(-1,2),m['corners_px'],atol=1e-5,rtol=0)
    h=json.loads((RUN/'heldout_example.json').read_text());r=by_id['right_04']
    m=manifests['board002_right04_heldout']
    assert m['training_pose_ids']==h['training_pose_ids']
    assert not set(h['training_pose_ids']) & set(h['excluded_pose_ids'])
    assert len(h['training_pose_ids'])==24 and len(h['excluded_pose_ids'])==6
    training=[by_id[pid] for pid in h['training_pose_ids']]
    solution=solve_robot_world_hand_eye([HandEyeObservation(np.array(x['T_C_P']),np.array(x['T_B_E'])) for x in training])
    prediction=np.linalg.inv(solution['T_world_base']@np.array(r['T_B_E'])@solution['T_ee_camera'])
    assert np.allclose(prediction,h['T_C_P_predicted'],atol=1e-10,rtol=0)
    projected,_=cv2.projectPoints(board.getChessboardCorners()[r['charuco_ids']],
        cv2.Rodrigues(prediction[:3,:3])[0],prediction[:3,3],np.array(r['K']),np.array(r['distortion']))
    assert np.allclose(projected.reshape(-1,2),h['predicted_pixels'],atol=1e-4,rtol=0)
    residual=np.linalg.norm(np.array(h['predicted_pixels'])-h['observed_pixels'],axis=1)
    assert np.allclose(residual,m['pixel_residuals'],atol=1e-12,rtol=0)
    assert abs(float(np.sqrt(np.mean(residual**2)))-m['pixel_rms'])<1e-12
    assert m['pose_error_T_C_P_mm_deg']==h['pose_error']
    for z in m['zoom_details']:
        index=r['charuco_ids'].index(z['corner_id'])
        assert z['observed_px']==h['observed_pixels'][index]
        assert z['predicted_px']==h['predicted_pixels'][index]
        assert z['magnification']==20
    return {'derived_plates':len(names),'right05_redetected_ids':len(ids),
        'heldout_training_only_refit_matches':True,'heldout_pixel_rms':m['pixel_rms'],
        'heldout_pixel_max':m['pixel_max'],'heldout_pixel_and_pose_metrics_distinct':True}

def main():
    from optimize_images import check_assets
    optimized_assets=check_assets()
    en=(DOC/'content/docs/index.mdx').read_text();zh=(DOC/'content/docs/zh/index.mdx').read_text()
    assert len(list((DOC/'content/docs').rglob('*.mdx')))==2
    for depth in (2,3):assert len(headings(en,depth))==len(headings(zh,depth))
    assert len(headings(en,2))==13
    assert math(en)==math(zh)
    assert tables(en)==tables(zh)
    assert figures(en)==figures(zh)
    assert set(figures(en))==set(optimized_assets)
    for s in (en,zh):
        assert not re.search(r'^#{2,3} .*?[?？]',s,re.M)
        assert not re.search(r'\]\(/(?:zh/)?(?:setup|reference|technical|troubleshooting)/',s)
        assert 'AX=XB' in s
    examples=0;excerpts=0
    for locale,s in (('en',en),('zh',zh)):
        for meta,snippet in snippets(s,'python'):
            compile(snippet,locale+'.mdx','exec')
            if ' runnable' in meta:
                exec(compile(snippet,locale+'.mdx','exec'),{})
                examples+=1
            else:
                assert ' excerpt' in meta,meta
                excerpts+=1
    rows=json.loads((RUN/'observations.json').read_text());result=json.loads((RUN/'result.json').read_text())
    trace=json.loads((RUN/'right_05.json').read_text());assert trace==rows[35]
    assert len(rows)==60 and [r['arm'] for r in rows]==['left']*30+['right']*30
    for row in rows:
        p=RUN/row['images'][row['arm']+'_wrist']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['image_sha256']
    assert [result[a]['accepted_poses'] for a in ('left','right')]==[26,30]
    assert trace['corner_count']==19 and trace['quality']['accepted']
    analysis_check=check_analysis_figures(rows)
    for s in (en,zh):
        blocks=[]
        for _,block in snippets(s,'text'):
            if block.lstrip().startswith('[['):blocks.append(np.array(ast.literal_eval(block)))
        for found,actual in zip(blocks,[trace['T_B_E'],trace['T_C_P']],strict=True):
            assert np.allclose(found,actual,atol=5.1e-7,rtol=0)
    state=json.loads((RUN/'ui/export.json').read_text())
    assert state['results']['result_kind']=='offline_robotwin_replay'
    for component in ('left','right','third'):assert state['artifacts']['geometry'][component]==result[component]
    assert not json.loads((RUN/'ui/walkthrough.json').read_text())['browser_errors']
    provenance=json.loads((DOC/'source-images/board002_pose_diversity.provenance.json').read_text())
    assert provenance['source_pose_ids']==['right_00','right_05','right_10','right_15']
    for pair in provenance['pairs']:
        for key in ('observer','wrist'):
            assert hashlib.sha256((DOC/pair[key]).read_bytes()).hexdigest()==pair[key+'_sha256']
        assert pair['wrist_crop'] is None
        row=next(r for r in rows if r['pose_id']==pair['pose_id']);T=np.asarray(row['T_C_P'])
        assert pair['corner_count']==row['corner_count']
        assert abs(pair['range_to_P_origin_m']-np.linalg.norm(T[:3,3]))<1e-12
        assert abs(pair['normal_tilt_deg']-np.degrees(np.arccos(np.clip(abs(T[2,2]),0,1))))<1e-12
    baseline=json.loads((DOC/'data/figure_checksums.json').read_text())
    for entry in baseline['files']:
        assert hashlib.sha256((DOC/'source-images'/Path(entry['path']).name).read_bytes()).hexdigest()==entry['sha256']
    from check_static_export import check_export
    links=check_export(DOC/'out',os.environ.get('NEXT_PUBLIC_BASE_PATH'))
    failures=[]
    for route in ('setup','reference','technical','troubleshooting'):
        for locale in ('','zh'):assert not (DOC/'out'/locale/route/'index.html').exists()
    report={'single_page_locales':['en','zh'],'main_chapters':10,'appendices':3,'h2_per_language':13,
        'h3_per_language':len(headings(en,3)),'display_equations_per_language':len(math(en)),
        'images_per_language':len(figures(en)),'executed_python_examples':examples,'compiled_excerpts':excerpts,
        'exported_tutorial_pages':2,'local_links_and_assets_checked':links,'broken_links':failures,
        'recorded_wrist_checksums':60,'right_05_record_matches':True,'rounded_matrices_match':True,
        'replay_geometry_matches_capture':True,'bilingual_math_tables_figures_match':True,
        'mature_figures_unchanged':True,'new_figure_pairs_checked':4,'analysis_figures':analysis_check,
        'optimized_webp_images':len(optimized_assets),'webp_pixels_match_originals':True,
        'hardware_accessed':False}
    (DOC/'quality-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
