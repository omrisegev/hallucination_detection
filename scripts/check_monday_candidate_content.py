"""Independent standard-library comparison of the complete saved decision tables."""
import csv
import datetime
import hashlib
import html
import json
import math
from pathlib import Path
ROOT=Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT=ROOT/'results/monday_candidate_panel_v1/run_20261004'
OURS=OUT/'CONTENT_METRICS.csv'
CLAUDE=ROOT/'results/monday_fusion_review_claude_20261004_round2/decision_rules_precompute/DECISION_METRICS.csv'
KEY=['bank','cell','variant','recipe','decision_rule']
FIELDS=['within_answer_auc','prmscore','pb_first_error_accuracy','pb_decision_error_accuracy','pb_official_f1','clean_flag_rate','wrong_answer_auroc']
def read(path):
    with path.open(encoding='utf8',newline='') as stream: rows=list(csv.DictReader(stream))
    lookup={tuple(row[k] for k in KEY):row for row in rows}
    assert len(lookup)==len(rows)==5184
    return rows,lookup
rows,ours=read(OURS); _,reference=read(CLAUDE)
assert ours.keys()==reference.keys()
checked=0; maximum=0.
for key,row in ours.items():
    for field in FIELDS:
        a,b=row[field],reference[key][field]
        if not a or not b:
            assert a==b; continue
        av,bv=float(a),float(b)
        if math.isnan(av) or math.isnan(bv): assert math.isnan(av) and math.isnan(bv); continue
        delta=abs(av-bv); assert delta<=1e-12,(key,field,delta)
        maximum=max(maximum,delta); checked+=1
points=[]
for recipe in ['MEAN_MOM','HEM_MOM']:
    for rule in ['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80']:
        result={'recipe':recipe,'decision_rule':rule}
        for variant in ['groups_answer','channels_cell']:
            subset=[r for r in rows if r['recipe']==recipe and r['variant']==variant and r['decision_rule']==rule and r['cell'].startswith('pb_')]
            assert len(subset)==64
            result[variant]=sum(float(r['pb_official_f1']) for r in subset)/64
        result['delta_pb_official_f1']=result['channels_cell']-result['groups_answer']; points.append(result)
record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FULL_CONTENT_REPLAY_PASS',
    'n_rows':len(rows),'n_banks':8,'n_cells':9,'numeric_values_checked':checked,'max_difference':maximum,
    'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OURS,CLAUDE]},
    'pb_f1_full_population_point_checks':points,'intervals':'Pending completion of independent paired bootstrap; do not copy Claude intervals as ours',
    'scope':'Preselection development; no chosen recipe; no untouched confirmation; final position/LR/forensics report pending'}
(OUT/'CONTENT_REPLAY_STATUS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')

# The report is explicitly mutable. Give Omri the complete point metrics while
# the memory-bounded bootstrap/forensics worker finishes the full report.
def table(records,columns):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(c,'')))+'</td>' for c in columns)+'</tr>' for row in records)+'</tbody></table>'
panels={}
for benchmark in ['PRMBench','ProcessBench']:
    metric_fields=['within_answer_auc','prmscore','clean_flag_rate','wrong_answer_auroc'] if benchmark=='PRMBench' else ['pb_first_error_accuracy','pb_decision_error_accuracy','pb_official_f1','clean_flag_rate','wrong_answer_auroc']
    output=[]
    for variant in ['groups_answer','channels_cell']:
        for recipe in ['MEAN_MOM','HEM_MOM','DS_FILTER_EQ','ALL_EQ','MEAN_EQ','HEM_EQ']:
            for rule in ['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80','TOP_1','TOP_2','TOP_3']:
                subset=[r for r in rows if r['variant']==variant and r['recipe']==recipe and r['decision_rule']==rule and r['cell'].startswith('prm' if benchmark=='PRMBench' else 'pb_')]
                assert len(subset)==(8 if benchmark=='PRMBench' else 64)
                entry={'variant':variant,'recipe':recipe,'rule':rule,'top2_post_hoc':rule=='TOP_2'}
                for f in metric_fields: entry[f]=f'{sum(float(r[f]) for r in subset)/len(subset):.4f}' if all(r[f] for r in subset) else '—'
                counts={r['cell']:r for r in subset}
                for n in ['n_answers','n_auc_answers','n_prmscore_answers','n_erroneous','n_clean']: entry[n]=sum(int(float(r[n])) for r in counts.values())
                output.append(entry)
    panels[benchmark]=(output,metric_fields)
lr=[]; lr_checkpoints=0
for stamp in sorted((ROOT/'results/level_reduction_v1/run_20261004_resume/lr').glob('*__original__*.complete.json')):
    seal=json.loads(stamp.read_text(encoding='utf8')); stem=stamp.name.removesuffix('.complete.json')
    for suffix,expected in seal['artifacts'].items():
        assert hashlib.sha256((stamp.parent/(stem+suffix)).read_bytes()).hexdigest()==expected
    with (stamp.parent/(stem+'.metrics.csv')).open(encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['decision_rule']=='ANSWER_Z_Q80':
                row['source_file']=str(stamp.parent/(stem+'.metrics.csv')); lr.append(row)
    lr_checkpoints+=1
page='''<!doctype html><html dir="rtl" lang="he"><meta charset="utf-8"><title>ארבעת המועמדים — מדדי תוכן מלאים</title><style>body{font:17px Arial;background:white;color:#17212e;max-width:1550px;margin:32px auto;padding:20px}p{line-height:1.6}table{direction:ltr;border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}td,th{border:1px solid #8291a0;padding:7px}th{background:#e5ecf4}.note{background:#fff1be;padding:18px}a{color:#145a9b}</style><h1>ארבעת המועמדים: מדדי התוכן על כל האוכלוסייה</h1><p class="note">דוח ביניים: חישוב מדדי התוכן הושלם על כל 8 הבנקים ו־9 התאים. רווחי הסמך העצמאיים והשוואת המתחרים עדיין בהרצה, עם השהיות כשחסר זיכרון RAM. אין בחירה אוטומטית של מתכון או כלל החלטה. הדוח המלא יחליף את הדף הזה בסיום.</p>'''
page+='<p>השוואת פיתוח לפני בחירה; לא מתכון נעול ולא אימות על נתונים שלא נחשפו. כל 5,184 השורות ו־25,344 הערכים המספריים תואמים בדיוק לטבלת קלוד. מקור ובדיקות: <a href="CONTENT_REPLAY_STATUS.json">CONTENT_REPLAY_STATUS.json</a>; כל התאים והבנקים: <a href="CONTENT_METRICS.csv">CONTENT_METRICS.csv</a>.</p>'
page+='<p>groups_answer: הצבעות קבוצות ומכסה לכל תשובה. channels_cell: הצבעות ערוצים ומכסה לכל התא. MEAN או HEM הם המשקול בתוך הקבוצות; MoM הוא המשקול בין הקבוצות. כל המועמדים משתמשים בבנקים המקוריים, ללא הסרת LEVEL.</p>'
page+='<p>q80 מסמן צעדים מעל הסף לאחר תקנון בתוך תשובה. R2 קובע מספר סימונים לפי הבנק הגולמי ומסמן את הצעדים המדורגים גבוה ביותר. שער האנטרופיה סוגר תשובות לפי אנטרופיה, ומשאיר q80 בתוך תשובה פתוחה. החלטת ProcessBench משתמשת בסימון הראשון; איתור לפי המקסימום הוא מדד נפרד.</p>'
for benchmark,(entries,fields) in panels.items():
    page+='<h2>'+benchmark+'</h2><p>ממוצע שווה בין שמונת הבנקים; ב־ProcessBench גם בין שמונת התאים. N סופר תשובות מודל פעם אחת, ללא הכפלה במספר הבנקים. top-2 מסומן כבחירה היסטורית בדיעבד.</p>'
    page+=table([r for r in entries if not r['rule'].startswith('TOP_')],['variant','recipe','rule']+fields+['n_answers','n_erroneous','n_clean'])
    page+='<details><summary>top-1 / top-2 / top-3 לכל מועמד ולכל בקרה</summary>'+table([r for r in entries if r['rule'].startswith('TOP_')],['variant','recipe','rule','top2_post_hoc']+fields+['n_answers'])+'</details>'
page+=f'<h2>ייחוס לינארי מפוקח: LR</h2><p>נשמרו {lr_checkpoints} מתוך 288 יחידות LR של הבנקים המקוריים. מוצגים רק תאים חתומים; אין ממוצע של שמונה בנקים על סמך השמטת החסרים. מלא ושורדים, עם ובלי POSZ. זהו ייחוס מפוקח, לא תקרה מתמטית.</p>'
for scope,title in [('OOF','מחוץ לאימון — תוצאות ההערכה'),('IN_SAMPLE_DIAGNOSTIC','בתוך האימון — אבחון נפרד')]:
    page+='<h3>'+title+'</h3>'+table([r for r in lr if r['score_scope']==scope],['bank','cell','recipe','position_variant']+FIELDS+['n_answers','coverage','failures','source_file'])
page+='<p>רווחי הסמך, אבחוני המיקום וניתוח המתחרים יתווספו בדוח המלא. בשחזור Mind the Gap נשמר argmax ללא הימנעות; הוא אינו כלל החצייה הראשונה המתואר במאמר. מצב החישוב: <a href="MEMORY_STATUS.json">MEMORY_STATUS.json</a>.</p></html>'
(OUT/'REPORT_HE.html').write_text(page,encoding='utf8')
print(json.dumps({k:v for k,v in record.items() if k not in ['source_sha256','pb_f1_full_population_point_checks']},indent=2))
