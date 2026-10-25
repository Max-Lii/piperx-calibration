"""Click the real offline server UI and save its actual API state alongside each shot."""
import json,os,time,urllib.request
from pathlib import Path
from PIL import Image, ImageChops, ImageStat
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'doc/experiments/BOARD_002_20261024_tutorial30_default/ui';OUT.mkdir(exist_ok=True)
BASE='http://127.0.0.1:18780'
client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def state():return json.load(client.open(BASE+'/api/v1/session'))
errors=[];stages=[];matches=[]
rows=json.loads((OUT.parent/"observations.json").read_text())
with sync_playwright() as pw:
 chrome=os.environ.get('PIPERX_CHROME')
 browser=pw.chromium.launch(**({'executable_path':chrome} if chrome else {}),headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1600,'height':900})
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(BASE+'/operator',wait_until='domcontentloaded')
 page.get_by_role('button',name='Start system check',exact=True).click()
 def shot(name):
  page.wait_for_timeout(400)
  page.screenshot(path=str(OUT/(name+'.png')),full_page=True)
  s=state();(OUT/(name+'.json')).write_text(json.dumps(s,indent=2))
  if s['stage'] in ('CAPTURE_LEFT','CAPTURE_RIGHT'):
   role=s['active_camera'];box=page.locator(f'img[src*="/{role}/stream"]').bounding_box()
   source=rows[s['progress']['current']-1]
   scroll=page.evaluate('window.scrollY')
   snap=Image.open(OUT/(name+'.png')).convert('RGB')
   crop=snap.crop((box['x'],box['y']+scroll,box['x']+box['width'],box['y']+scroll+box['height'])).resize((640,480))
   target=Image.open(OUT.parent/source['images'][role]).convert('RGB')
   # Exclude the recorded-source label, preserve actual captured content.
   region=(10,10,630,430)
   mae=sum(ImageStat.Stat(ImageChops.difference(crop.crop(region),target.crop(region))).mean)/3
   matches.append({'pose_id':source['pose_id'],'mean_absolute_rgb_difference':mae,'image':name+'.png'})
   assert mae<10, matches[-1]
  stages.append({'image':name+'.png','stage':s['stage'],'revision':s['revision'],'progress':s['progress']})
 def advance(expected):
  page.get_by_role('button',name='Continue',exact=True).click()
  for _ in range(60):
   if state()['stage']==expected:break
   page.wait_for_timeout(100)
  assert state()['stage']==expected,(expected,state())
  page.wait_for_timeout(400)
 page.wait_for_function('document.images.length===3 && Array.from(document.images).every(i=>i.naturalWidth===640)')
 shot('01-device')
 advance('TARGET_VERIFY');shot('02-target')
 advance('SAFETY_DRY_RUN');shot('03-pose-preparation')
 advance('A4_PLACEMENT');shot('04-placement')
 advance('WORLD_REGISTRATION')
 page.get_by_role('button',name='Continue',exact=True).click()
 seen=set();deadline=time.time()+90
 while time.time()<deadline:
  s=state();stage=s['stage']
  if stage in ('CAPTURE_LEFT','CAPTURE_RIGHT') and stage not in seen and s['progress']['current']%30==6:
   seen.add(stage);shot('05-capture-left' if stage=='CAPTURE_LEFT' else '06-capture-right')
  if stage=='VALIDATE_GEOMETRY':break
  if s['errors']:raise AssertionError(s['errors'])
  page.wait_for_timeout(150)
 assert state()['stage']=='VALIDATE_GEOMETRY',state()
 assert len(seen)==2,seen
 shot('07-validation')
 advance('REVIEW_AND_EXPORT');shot('08-review')
 with page.expect_download() as download:
  page.get_by_role('link',name='Export JSON',exact=True).click()
 download.value.save_as(str(OUT/'export.json'))
 advance('COMPLETE');shot('09-complete')
 assert not errors,errors
 browser.close()
(OUT/'walkthrough.json').write_text(json.dumps({'browser_errors':errors,'stages':stages,'preview_source_matches':matches,'actions':'browser clicks only; no response mocking or DOM state edits','viewport':[1600,900],'full_page':True,'result_source':'recomputed from this run recorded RGB and FK'},indent=2))
print(json.dumps(stages,indent=2))
