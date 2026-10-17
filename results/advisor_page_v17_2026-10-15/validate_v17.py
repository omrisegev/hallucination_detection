"""Validate the deliverable and interact with its exact JavaScript in local Chrome."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,re,subprocess,tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PAGE=HERE/'published_v17_2026-10-15.html'
CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

class Inspect(HTMLParser):
    def __init__(self):
        super().__init__();self.sections=[];self.assets=[];self.links=[];self.id=[];self.svg_depth=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='svg':self.svg_depth+=1
        if not self.svg_depth and 'id' in a:self.id.append(a['id'])
        if tag=='section':self.sections.append(a)
        if tag in ['img','script','link']:
            value=a.get('src') or (a.get('href') if tag=='link' else None)
            if value:self.assets.append(value)
        if tag=='a' and a.get('href','').startswith('#'):self.links.append(a['href'][1:])
    def handle_endtag(self,tag):
        if tag=='svg':self.svg_depth-=1

def main():
    page=PAGE.read_text();p=Inspect();p.feed(page)
    assert len(p.sections)==21 and len(set(p.id))==len(p.id)
    assert all(x.startswith('data:') for x in p.assets)
    assert set(p.links)<=set(p.id)
    assert 'Claims we withdraw' not in page
    assert len(re.findall(r'<aside class="speaker"',page))==21
    receipt=json.loads((HERE/'BUILD_RECEIPT.json').read_text())
    assert receipt['minutes']==31 and receipt['main_slides']==14 and receipt['appendix_slides']==7
    assert receipt['sha256']==hashlib.sha256(PAGE.read_bytes()).hexdigest()
    for r in json.loads((HERE/'SOURCE_MANIFEST.json').read_text())['sources']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    data=json.loads((HERE/'DATA.json').read_text())
    assert all(t['removed']==[] for t in data['transitions'])
    assert all(data['bank_metrics'][b]['LR']['auc']>data['bank_metrics'][a]['LR']['auc'] for a,b in zip(data['banks'],data['banks'][1:]))
    assert all(data['bank_metrics'][b]['LR']['prmscore']>data['bank_metrics'][a]['LR']['prmscore'] for a,b in zip(data['banks'],data['banks'][1:]))
    checks=r'''
<script>
(()=>{
const report={errors:window.__v17errors||[],checks:[],slides:[]};
function check(name,value){report.checks.push({name,pass:!!value})}
function go(id){document.querySelector('nav a[href="#'+id+'"]').click()}
try{
document.getElementById('present').click();check('presentation mode',document.body.classList.contains('presentation'));
document.getElementById('next').click();check('next slide',document.querySelector('.slide.active').id==='s02');
document.getElementById('prev').click();check('previous slide',document.querySelector('.slide.active').id==='s01');
document.getElementById('notes').click();check('Hebrew speaker notes',getComputedStyle(document.querySelector('.slide.active .speaker')).display==='block');document.getElementById('notes').click();
go('s07');document.querySelector('[data-panel="feature-profile"]').click();check('H profile view',!document.querySelector('[data-view="feature-profile"]').hidden&&document.querySelector('[data-view="heatmap"]').hidden);
document.querySelector('[data-panel="heatmap"]').click();document.querySelector('.zoom').click();check('heatmap enlargement',!!document.querySelector('.overlay'));window.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}));check('close enlargement',!document.querySelector('.overlay'));
go('a3');check('appendix navigation',document.querySelector('.slide.active').id==='a3');
document.querySelectorAll('.slide').forEach(s=>{go(s.id);const visible=document.querySelectorAll('.slide.active').length;const w=document.documentElement.clientWidth;const overflow=document.documentElement.scrollWidth>w+2;report.slides.push({id:s.id,oneActive:visible===1,horizontalOverflow:overflow,scrollWidth:document.documentElement.scrollWidth,width:w});});
go('s01');window.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight'}));check('keyboard navigation',document.querySelector('.slide.active').id==='s02');
go('__TARGET__');
}catch(error){report.errors.push(String(error))}
const output=document.createElement('pre');output.id='v17-validation-result';output.hidden=true;output.textContent=JSON.stringify(report);document.body.appendChild(output);
})();
</script>
'''
    (HERE/'validation').mkdir(exist_ok=True)
    browser=[]
    with tempfile.TemporaryDirectory(prefix='advisor-v17-',dir='/private/tmp') as td:
        td=Path(td)
        for size,target,name in [('1440,1000','s01','desktop'),('500,844','s01','mobile')]:
            injected=page.replace('<script>'+re.search(r'<script>(.*?)</script>',page,re.S)[1]+'</script>', '<script>window.__v17errors=[];window.addEventListener("error",e=>window.__v17errors.push(e.message));</script>'+re.search(r'<script>.*?</script>',page,re.S)[0],1)
            injected=injected.replace('</body>',checks.replace('__TARGET__',target)+'</body>')
            test_page=td/f'{name}.html';test_page.write_text(injected)
            command=[CHROME,'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',f'--user-data-dir={td/name}',f'--window-size={size}','--virtual-time-budget=1500','--dump-dom',f'--screenshot={HERE/"validation"/(name+".png")}',test_page.as_uri()]
            run=subprocess.run(command,capture_output=True,text=True,timeout=45)
            assert run.returncode==0,run.stderr[-1000:]
            dom=run.stdout
            from html import unescape
            match=re.search(r'<pre id="v17-validation-result"[^>]*>(.*?)</pre>',dom,re.S)
            assert match,'Browser did not produce validation output'
            result=json.loads(unescape(match[1]));result['viewport']=size
            assert not result['errors'],result['errors']
            assert all(c['pass'] for c in result['checks']),result['checks']
            assert all(r['oneActive'] and not r['horizontalOverflow'] for r in result['slides']),result['slides']
            browser.append(result)
        # Representative desktop screenshots of the exact deck with a selected slide.
        for target in ['s04','s07','s10','s11','s12']:
            screenshot_page=td/f'{target}.html'
            action='<script>document.getElementById("present").click();document.querySelector(\'nav a[href="#'+target+'"]\').click();</script>'
            screenshot_page.write_text(page.replace('</body>',action+'</body>'))
            command=[CHROME,'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',f'--user-data-dir={td/("shot-"+target)}','--window-size=1440,1000','--virtual-time-budget=1000',f'--screenshot={HERE/"validation"/(target+".png")}',screenshot_page.as_uri()]
            run=subprocess.run(command,capture_output=True,text=True,timeout=45)
            assert run.returncode==0,run.stderr[-1000:]
    report={'status':'PASS','html_sha256':receipt['sha256'],'standalone_assets':True,'section_count':21,'speaker_notes':21,'source_hashes_match':True,'nested_digit_banks':True,'browser':browser}
    (HERE/'VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: static checks, source hashes, desktop/mobile navigation, notes, heatmap views, keyboard, 21 sections without horizontal overflow.')

if __name__=='__main__':main()
