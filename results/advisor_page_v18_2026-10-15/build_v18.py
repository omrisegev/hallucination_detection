"""Alternative presentation: restore v16 taxonomy, paired examples and radar.

Reuses the unchanged v17 presentation and its exact charts/data; no numerical run.
"""
from pathlib import Path
import hashlib,html,json,re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V17=ROOT/'results/advisor_page_v17_2026-10-15'
SOURCE=ROOT/'results/advisor_page_v13_v14_2026-10-12/published_v16_2026-10-13.html'
if not SOURCE.exists():
    SOURCE=ROOT/'docs/reviews/advisor_v16_walkthrough_20261013/source_v16_2cf1209c.html'
source=SOURCE.read_text()
raw=SOURCE.read_bytes()
assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()=='9aaab92066600edb593b7050cd7f042acf7f1719'
v17=(V17/'published_v17_2026-10-15.html').read_text()
v17_hash=hashlib.sha256((V17/'published_v17_2026-10-15.html').read_bytes()).hexdigest()
assert v17_hash=='0117d91120c6e2c96f23dfd9f26c87fc1f5835d116318ebe79a9a7404f03c522'

def original_section(ident):
    return re.search(r'<section id="'+ident+r'">(.*?)</section>',source,re.S)[1]
def original_figure(ident,index=0):
    return re.findall(r'<figure>.*?</figure>',original_section(ident),re.S)[index]
def section(ident,page):
    return re.search(r'<section class="slide[^\"]*" id="'+re.escape(ident)+r'".*?</section>',page,re.S)[0]
def escape(text):return html.escape(text,quote=True)
def record(ident,title,old,notes,question,limit,sources):
    return {'id':ident,'number':0,'title':title,'minutes':1.5,'old':old,'notes':notes,'sources':sources,'question':question,'limit':limit,'appendix':False}
def new_section(r,body):
    return f'<section class="slide" id="{r["id"]}" aria-labelledby="{r["id"]}-title"><div class="eyebrow">ADVISOR MEETING · 15 OCTOBER 2026<span>1.5 MIN · {escape(r["old"])}</span></div><h1 id="{r["id"]}-title">{escape(r["title"])}</h1>{body}<aside class="speaker" lang="he" dir="rtl"><h3>הערות דובר</h3><p>{r["notes"]}</p><p><b>מה הגרף מודד ואיך קוראים אותו:</b> {r["question"]}</p><p><b>גבול המסקנה:</b> {r["limit"]}</p><p><b>מיפוי:</b> {escape(r["old"])}</p></aside><div class="source"><b>Evidence</b> {escape(r["sources"])}</div></section>'

taxonomy=original_figure('s2')
example_blocks=re.findall(r'<article class="example">.*?</article>',original_section('s2'),re.S)
assert len(example_blocks)==2
examples=''.join(example_blocks)
# Remove TeX delimiters only from step text; preserve currency in the question.
examples=examples.replace('$\\$1.24$','$1.24').replace('$\\$1.82$','$1.82')
examples=re.sub(r'<span>(.*?)</span>',lambda m:'<span>'+m[1].replace('$','')+'</span>',examples,flags=re.S)
radar=original_figure('s4')
# These auxiliary labels were outside the original cropped viewBox. Their full
# definitions are retained in the visible figure caption instead.
radar=re.sub(r'<text\b[^>]*>(?:Green:.*?|Radius:.*?|MS uses.*?)</text>','',radar,flags=re.S)
radar=re.sub(r'<figcaption>.*?</figcaption>','',radar,flags=re.S)
assert all(x in taxonomy for x in ['SIMPLICITY','SOUNDNESS','SENSITIVITY','Non-Redundancy','Non-Circular Logic','Empirical Soundness','Step Consistency','Domain Consistency','Confidence Invariance','Prerequisite Sensitivity','Deception Resistance'])
assert 'redundency_redundency_prm_test_p1_0' in examples and 'circular_circular_prm_test_p1_0' in examples
assert '<polygon' in radar

taxonomy_record=record('s01a','What kinds of reasoning errors does PRMBench test?','v16 §2 · original construction and taxonomy visual',
    'מחזירים את התרשים המקורי שהמשתמש אהב. עוברים לפי שלוש המשפחות: Simplicity היא איכות ההנמקה, Soundness היא תקפות הטענה והכלל, Sensitivity היא תגובה לתנאי חסר או טענה מטעה. לא מקריאים כל כרטיס; משתמשים בו כמפה לפענוח הגרפים בהמשך.',
    'זהו תרשים מושגי ולא מדד ביצועים. חץ הוא מעבר בהנמקה; אדום מסמן שגיאה, ירוק צעד תקין, וגבול מקווקו מסמן תנאי חסר. בצד שמאל מופיעה בניית ה־benchmark ואוכלוסיית ההערכה.',
    'השרטוטים הם המחשה, לא דוגמאות inference. MS מכיל מסלולים תקינים ומודד false flags, בניגוד לשמונה סוגי השגיאה.',
    'Pinned v16 §2 SVG; 6,211 noncontrol answers + 758 synthetic controls; no score or threshold in this concept diagram')
examples_record=record('s01b','NR and NCL: same correct answer, different reasoning faults','v16 §2 · two original worked examples',
    'שתי הדוגמאות המקוריות מופיעות זו לצד זו ובגלוי. בשתיהן מחיר העיפרון הוא 29 סנט. ב־NR הצעד חוזר על הנמקה שכבר בוצעה; ב־NCL מניחים את המחיר ואז מאשרים אותו. לכן מספרים נכונים ותשובה סופית נכונה אינם מבטיחים צעדי reasoning תקינים.',
    'אין כאן תחזיות של השיטה. הצבע והתווית ליד כל צעד הם האנומטציות האמיתיות. מציגים excerpts משתי תשובות על אותה שאלה, עם ה־source IDs המקוריים.',
    'הדוגמאות נבחרו בסדר קבוע של הנתונים, לא לפי הצלחה של מודל. מציגים רק את הצעדים המקוריים הרלוונטיים; זו אינה השוואת ביצועים על N=2.',
    'v16 §2: controls/DATASET_EXAMPLES.json; NR redundency_redundency_prm_test_p1_0; NCL circular_circular_prm_test_p1_0')
taxonomy_body='<p class="lead">Three families: unnecessary or circular reasoning, invalid claims, and sensitivity to changed premises.</p><div class="restored taxonomy-visual">'+taxonomy+'</div><p class="caption">Green: valid step. Red: erroneous step. Dashed: missing premise. The cards are conceptual illustrations; MS tests acceptance of multiple valid paths.</p>'
examples_body='<p class="lead">The final answer can remain 29 cents. The benchmark labels the reasoning steps.</p><div class="paired-examples">'+examples+'</div><div class="callout"><strong>Why these examples matter</strong><p>NR: a correct calculation can still be redundant. NCL: assuming the answer and checking it is circular reasoning.</p></div>'

page=v17.replace('for the example on slide 1.','for the NCL example on slide 3.')
page=page.replace('same example as slide 1','same NCL example as slide 3')
page=page.replace('on slide 11','on slide 13')
first=section('s01',page)
first_new=re.sub(r'<ol class="steps">.*?</ol>',
    '<div class="intro-task"><span class="tag">THE OBJECT WE SCORE</span><h2>A provided answer, split into steps</h2><p>Each step receives a validity label.</p><p>Our methods produce a risk ranking and a set of flagged steps.</p><p><b>Next:</b> the error taxonomy, then actual NR and NCL examples.</p></div>',first,flags=re.S)
first_new=first_new.replace('NCL example: assuming 29 cents and confirming it is circular reasoning. The final answer can still be correct.','The next two slides restore the original taxonomy visual and the paired NR/NCL examples.')
page=page.replace(first,first_new+new_section(taxonomy_record,taxonomy_body)+new_section(examples_record,examples_body),1)

old_profile=section('s06',page)
existing_profile=re.search(r'<figure class="chart[^\"]*">.*?</figure>',old_profile,re.S)[0]
restored_profile='<div class="tabs"><button class="view active" data-panel="type-radar">Category profile · original radar</button><button class="view" data-panel="type-lines">Matched profile · including LR</button></div><div data-view="type-radar" class="restored radar-visual">'+radar+'<p class="caption">N16; 6,211 noncontrol answers split by error type. L-SML / IU-PCR: Q80. PRM-7B: native reward &lt;0.5. Radius 0–100. MS uses valid-step F1 alone.</p></div><div data-view="type-lines" hidden>'+existing_profile+'</div>'
new_profile=old_profile.replace(existing_profile,restored_profile,1)
new_profile=new_profile.replace('The LR profile is computed under Q80 here. Its Top1 complementarity is a separate diagnostic in the appendix.','The alternate view includes LR under Q80. Its Top1 complementarity is a separate diagnostic in the appendix.')
new_profile=new_profile.replace('v16 §4a; LR saved OOF evaluation','v16 §4a · restored radar; LR in alternate view')
page=page.replace(old_profile,new_profile,1)

records=json.loads((V17/'SLIDE_MAP.json').read_text())
records[0]['notes']='פותחים במשימה, בשאלה אמיתית ובהגדרת PRM. פרטי התשובה עברו לשקופית דוגמאות NR/NCL אחרי מפת סוגי השגיאות, כדי להימנע מכפילות.'
records[0]['question']='מראים את השאלה האמיתית, מה נחשב לצעד ומהו פלט הדירוג והסימון. הדוגמאות עם תוויות האמת בשקופית השלישית.'
records=records[:1]+[taxonomy_record,examples_record]+records[1:]
for i,r in enumerate(records):
    if not r['appendix']:r['number']=i+1
    if r['id']=='s06':
        r['old']='v16 §4a · restored original radar; v17 profile as alternate view'
        r['notes']='מחזירים את גרף הפוליגונים המקורי כברירת המחדל. מתחילים מהחוזקה ב־NR/NCL ואז מראים את החולשה ב־CI/DR וב־MS. המסגרת של 0–100 נשמרת. LR נמצא בתצוגה הנוספת של הפרופיל, עם Q80 תואם, כדי לא להחליף את ה־radar המקורי שהמשתמש אהב.'
        r['question']='כל ציר הוא סוג שגיאה וכל פוליגון הוא שיטה. רדיוס גדול משמעו ציון גבוה. שמונה צירים הם macro-F1; MS הוא valid-step F1. אותן תוויות וכללי ההחלטה של v16.'
        r['sources']='Original pinned v16 §4a radar; unchanged v17 saved Q80 LR profile in alternate view'
    # Update visible notes from the final mapping, including the restored profile.
    r['sources']=r['sources'].replace('same example as slide 1','same NCL example as slide 3')
    r['notes']=r['notes'].replace('בשקופית השנייה','בשקופית המדדים')
    block=section(r['id'],page)
    speaker=f'<aside class="speaker" lang="he" dir="rtl"><h3>הערות דובר</h3><p>{r["notes"]}</p><p><b>מה הגרף מודד ואיך קוראים אותו:</b> {r["question"]}</p><p><b>גבול המסקנה:</b> {r["limit"]}</p><p><b>מיפוי:</b> {escape(r["old"])}</p></aside>'
    replacement=re.sub(r'<aside class="speaker".*?</aside>',lambda _:speaker,block,flags=re.S)
    page=page.replace(block,replacement,1)

nav=''.join(('<div class="appendix-label">Appendix</div>' if r['appendix'] and r['number']==1 else '')+f'<a href="#{r["id"]}"><span class="nav-num">{"A" if r["appendix"] else ""}{r["number"]:02}</span>{escape(r["title"])}</a>' for r in records)
page=re.sub(r'<nav aria-label="Slide navigation">.*?</nav>',lambda _:'<nav aria-label="Slide navigation">'+nav+'</nav>',page,flags=re.S)
page=page.replace('Advisor meeting · 15 Oct 2026 · v17','Advisor meeting · 15 Oct 2026 · v18 · restored visuals')
page=page.replace('Advisor meeting · v17','Advisor meeting · v18')
page=page.replace('31 min + 9 min discussion','34 min + 6 min discussion')
page=page.replace('Discussion · 9 minutes','Discussion · 6 minutes')
page=page.replace('Slide 1 / 14','Slide 1 / 16').replace('current<14','current<16').replace("' / 14'","' / 16'")
extra_css='''
.intro-task{background:#f6f8fb;border:1px solid var(--line);border-radius:8px;padding:24px;margin-top:16px}.paired-examples{display:grid;grid-template-columns:1fr 1fr;gap:22px}.example{border:1px solid var(--line);border-radius:8px;padding:16px}.example h3{margin:0 0 12px;font-size:19px;line-height:1.3}.example .small{font-size:13px}.step{padding:8px 10px;margin:6px 0;border-left:3px solid #7da69b;background:#f7faf9;font-size:13px;line-height:1.4}.step.bad{border-color:#d4543c;background:#fff3ef}.step b{display:block;font-size:10px;text-transform:uppercase;margin-bottom:3px}.example .source{font-size:10px}.restored figure svg{width:100%;height:auto;max-height:590px}.radar-visual figure svg{max-height:480px}.restored figure{margin:8px 0}.paired-examples .source{overflow-wrap:anywhere}@media(max-width:900px){.paired-examples{grid-template-columns:1fr}.restored figure svg{max-height:none}}@media print{.paired-examples{grid-template-columns:1fr 1fr}[data-view="type-lines"]{display:block!important}.restored figure svg{max-height:150mm}}
'''
page=page.replace('</style>',extra_css+'</style>',1)
destination=HERE/'published_v18_2026-10-15.html'
destination.write_text(page)
(HERE/'SLIDE_MAP.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
book='# v18 — הערות דובר ומיפוי תכנים\n\n34 דקות תוכן + 6 דקות דיון. v17 נשמרת כחלופה של 31 דקות.\n\n'
for r in records:
    book+=f'## {"נספח " if r["appendix"] else ""}{r["number"]} — {r["title"]}\n\nמקור: {r["old"]}. זמן: {r["minutes"]} דקות.\n\n{r["notes"]}\n\n**איך קוראים:** {r["question"]}\n\n**גבול המסקנה:** {r["limit"]}\n\nמקורות: {r["sources"]}\n\n'
(HERE/'SPEAKER_NOTES_HE.md').write_text(book.rstrip()+'\n')
(HERE/'restored_assets').mkdir(exist_ok=True)
for name,content in [('taxonomy.html',taxonomy),('nr_ncl_examples.html',examples),('category_radar.html',radar)]:
    (HERE/'restored_assets'/name).write_text(content)
receipt={'html':destination.name,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
    'v17_html_sha256':v17_hash,'source_v16_blob':'9aaab92066600edb593b7050cd7f042acf7f1719',
    'main_slides':16,'appendix_slides':7,'minutes':sum(r['minutes'] for r in records),
    'new_main_slides':['s01a','s01b'],'restored_radar_slide':'s06','data_changed':False}
assert receipt['minutes']==34
(HERE/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
sources=[]
for path in [V17/'published_v17_2026-10-15.html',V17/'SLIDE_MAP.json']:
    sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':path.stat().st_size})
sources.append({'path':'results/advisor_page_v13_v14_2026-10-12/published_v16_2026-10-13.html',
    'local_fallback':'docs/reviews/advisor_v16_walkthrough_20261013/source_v16_2cf1209c.html',
    'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)})
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps({'sources':sources,'frozen_data':'../advisor_page_v17_2026-10-15/DATA.json','no_new_metrics':True},indent=2)+'\n')
print('Built v18: 16 main slides, 7 appendices, 34 minutes; v17 preserved unchanged.')
