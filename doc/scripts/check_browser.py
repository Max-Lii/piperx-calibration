"""Inspect single-page Fumadocs in a real Chromium browser after export.

Start a loopback static server for doc/out on port 18781. No robot/GPU work.
"""
import json,os,re
from pathlib import Path
from urllib.parse import unquote
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'doc/validation/teaching-refinement';OUT.mkdir(exist_ok=True,parents=True)
BASE=os.environ.get('PIPERX_DOCS_URL','http://127.0.0.1:18781').rstrip('/')
CHROME=os.environ.get('PIPERX_CHROME')
errors=[];pages=[];toc_checks=[];detail_checks=[];figure_checks=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(**({'executable_path':CHROME} if CHROME else {}),headless=True,args=['--no-sandbox'])
 context=browser.new_context(viewport={'width':1600,'height':1000},permissions=['clipboard-read','clipboard-write'])
 page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for path in ('/','/zh/'):
  response=page.goto(BASE+path,wait_until='networkidle');assert response.status==200,path
  assert page.locator('h1').count()==1
  assert page.locator('article h2').count()==13
  assert page.locator('.tutorial-sections').count()==0
  assert page.locator('#nd-sidebar').count()==0 or not page.locator('#nd-sidebar').is_visible()
  images=page.locator('article img')
  for i in range(images.count()):
   im=images.nth(i);im.scroll_into_view_if_needed();im.evaluate('(im)=>im.decode()')
   assert im.evaluate('(im)=>im.naturalWidth>0'),im.get_attribute('src')
  assert page.locator('.katex-error').count()==0
  assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),path
  diagrams=page.locator('.mermaid-diagram svg').count();assert diagrams==3,(path,diagrams)
  page.evaluate('window.scrollTo(0,0)');page.wait_for_timeout(300)
  toc=page.locator('#nd-toc')
  assert toc.is_visible(),path
  entries=toc.locator('a[href*="#"]');assert entries.count()>=13
  # Native TOC click must change the hash and place the actual heading in view.
  href=entries.filter(has_text=re.compile('5\\.')).first.get_attribute('href')
  toc.locator('a').filter(has_text=re.compile('5\\.')).first.click()
  page.wait_for_timeout(650)
  assert unquote(page.evaluate('location.hash'))==unquote(href[href.index('#'):])
  box=page.locator('article h2').nth(4).bounding_box();assert box and -5<=box['y']<350,(path,box)
  toc_checks.append({'path':path,'entries':entries.count(),'clicked':href,'heading_top':round(box['y'],1)})
  page.screenshot(path=str(OUT/('zh-math.png' if path=='/zh/' else 'en-math.png')))
  details=page.locator('details.observation-data');assert details.count()==1
  assert not details.evaluate('(el)=>el.open')
  details.locator('summary').click();assert details.evaluate('(el)=>el.open')
  assert details.locator('pre').count()==3
  assert '0.798887' in details.inner_text() and '0.973822' in details.inner_text()
  page.screenshot(path=str(OUT/('zh-record-open.png' if path=='/zh/' else 'en-record-open.png')))
  details.locator('summary').click();assert not details.evaluate('(el)=>el.open')
  detail_checks.append({'path':path,'default_collapsed':True,'three_numeric_blocks_accessible':True})
  page.evaluate('window.scrollTo(0,0)');page.wait_for_timeout(250)
  page.screenshot(path=str(OUT/('zh-top.png' if path=='/zh/' else 'en-top.png')))
  pages.append({'path':path,'title':page.locator('h1').inner_text(),'loaded_images':images.count(),
      'mermaid_svg':diagrams,'main_heading_count':13,'horizontal_overflow':False})
 # Inspect the figure at actual reading width, then the native lightbox.
 page.goto(BASE+'/zh/',wait_until='networkidle')
 pair=page.locator('img[src$="board002_pose_diversity.webp"]');pair.scroll_into_view_if_needed()
 page.screenshot(path=str(OUT/'zh-pose-diversity.png'))
 page.locator('.image-trigger').filter(has=pair).click();assert page.get_by_role('dialog').is_visible()
 page.screenshot(path=str(OUT/'pose-diversity-lightbox.png'))
 page.get_by_role('button',name='Close image',exact=True).click()
 assert page.get_by_role('dialog').count()==0
 for key,src in [('device','01-device.webp'),('placement','board002_ui_placement.webp'),('capture','board002_ui_acquisition.webp'),('review','board002_ui_review.webp'),('right05','board002_right05_correspondences.webp'),('right04','board002_right04_heldout.webp')]:
  im=page.locator(f'img[src$="{src}"]');im.scroll_into_view_if_needed()
  page.screenshot(path=str(OUT/f'zh-ui-{key}.png'))
  if key in ('right05','right04','review'):
   before=page.url
   page.locator('.image-trigger').filter(has=im).click()
   assert page.get_by_role('dialog').is_visible() and page.url==before
   page.screenshot(path=str(OUT/f'{key}-lightbox.png'))
   page.get_by_role('button',name='Close image',exact=True).click()
   assert page.get_by_role('dialog').count()==0
   figure_checks.append({'image':src,'inline_lightbox_without_navigation':True,'display_width':round(im.bounding_box()['width'],1)})
 # Actual language, clipboard, and theme controls.
 page.evaluate('window.scrollTo(0,0)')
 page.locator('.language-switch').get_by_role('link',name='English',exact=True).click()
 page.wait_for_url(BASE+'/');assert page.locator('article h2').count()==13
 page.locator('.code-copy').first.scroll_into_view_if_needed();page.locator('.code-copy').first.click()
 page.wait_for_function("document.querySelector('.code-copy')?.textContent.includes('Copied')")
 assert 'piper_x_description.urdf' in page.evaluate('navigator.clipboard.readText()')
 page.evaluate('window.scrollTo(0,0)')
 page.get_by_role('button',name='Dark',exact=True).click()
 page.wait_for_function("document.documentElement.classList.contains('dark')")
 page.wait_for_timeout(500);page.screenshot(path=str(OUT/'en-dark.png'))
 page.get_by_role('button',name='Light',exact=True).click()
 page.wait_for_function("!document.documentElement.classList.contains('dark')")
 page.set_viewport_size({'width':390,'height':844})
 for path,name in [('/','en'),('/zh/','zh')]:
  page.goto(BASE+path,wait_until='networkidle')
  assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),path
  page.screenshot(path=str(OUT/f'{name}-mobile.png'))
 assert not errors,errors
 browser.close()
report={'pages':pages,'viewport':[1600,1000],'mobile_width':390,'toc_clicks':toc_checks,
    'page_errors':errors,'five_page_navigation_removed':True,'sidebar_absent':True,
    'lightbox':True,'code_clipboard':True,'theme_controls':True,'language_switch':True,'broken_images':0,
    'numeric_record_details':detail_checks,'refined_figures':figure_checks}
(OUT/'browser-check.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
