"""Post-selection descriptive error ledger; frozen scores are never modified."""
from pathlib import Path
import sys,json,csv,html,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_temporal_research_baseline as base
from scripts.run_predictor_subset_study import OUT,write,sha
from spectral_utils.predictor_subset_fusion import METHODS

def run():
    data=json.loads((OUT/'METRICS.json').read_text());metrics=data['metrics']
    records,joined=base.load_contract(ROOT.parents[1]);offset=joined['offsets'];target=joined['target']
    cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_');error=pb&(target>=0)
    with np.load(OUT/'SCORES_FROZEN.npz') as f:scores={n:f[n] for n in f.files}
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:gate=f['gate_percentile']>=.33
    peaks={n:np.array([np.argmax(s[offset[i]:offset[i+1]]) for i in range(len(records))]) for n,s in scores.items()}
    pred={n:np.where(gate,p,-1) for n,p in peaks.items()};hits={n:error&(p==target) for n,p in pred.items()}
    summary={}
    position=target/np.maximum(1,np.diff(offset)-1)
    strata={'early':position<1/3,'middle':(position>=1/3)&(position<2/3),'late':position>=2/3}
    for n in METHODS+('ridge','tcn__real','bocpd','innovation5'):
        p=pred[n];raw=peaks[n]
        v=dict(errors=int(error.sum()),hits=int(hits[n].sum()),gate_closed=int((error&~gate).sum()),
            early=int((error&gate&(p<target)).sum()),late=int((error&gate&(p>target)).sum()),
            correct_raw_peak_suppressed=int((error&~gate&(raw==target)).sum()))
        assert v['hits']+v['gate_closed']+v['early']+v['late']==v['errors']
        assert int((error&(raw==target)).sum())==metrics[n]['pb_exact_count']
        assert v['correct_raw_peak_suppressed']==metrics[n]['pb_correct_peaks_suppressed']
        v['by_first_error_position']={k:dict(errors=int((error&mask).sum()),hits=int((hits[n]&mask).sum())) for k,mask in strata.items()}
        v['versus']={ref:dict(gained=int((hits[n]&~hits[ref]).sum()),lost=int((~hits[n]&hits[ref]).sum()),
            both=int((hits[n]&hits[ref]).sum())) for ref in ('ridge','tcn__real','bocpd')}
        summary[n]=v
    union=np.logical_or.reduce([hits[n] for n in METHODS]);rawunion=np.logical_or.reduce([error&(peaks[n]==target) for n in METHODS])
    allmiss=error&~union
    overlap=dict(error_answers=int(error.sum()),source_groups=len({records[i]['group_id'] for i in np.flatnonzero(error)}),
        hit_by_at_least_one_of32=int(union.sum()),missed_by_all32=int(allmiss.sum()),
        missed_all_gate_closed=int((allmiss&~gate).sum()),missed_all_gate_open=int((allmiss&gate).sum()),
        gate_closed_but_correct_raw_peak_exists=int((error&~gate&rawunion).sum()),
        gate_closed_no_correct_raw_peak=int((error&~gate&~rawunion).sum()),
        scope='Oracle selection among these existing peaks only; not an attainable detector or an upper bound on new fusion.')
    pb_lead=data['leaders']['pb_all8'];within_lead=data['leaders']['prm_within']
    roster=(pb_lead,within_lead,'equal__ridge+tcn+bocpd','iu__ridge+bocpd+noreset','tcn__real','ridge','bocpd')
    # Illustrative cases from the first PB cell, selected by category then UID,
    # not by narrative appeal. Full population counters are above.
    cell,path,kind,dataset=base.evaluator.source_specs()[0]
    rows=base.evaluator.old._source_row_map(base.evaluator.old.load_pickle(path),kind=kind,dataset=dataset)
    categories={'IU_loses_TCN_hit':hits['tcn__real']&~hits[pb_lead],
        'IU_gains_TCN_miss':hits[pb_lead]&~hits['tcn__real'],
        'all32_miss_gate_open':allmiss&gate,
        'gate_closes_correct_peak':error&~gate&rawunion,
        'within_leader_loses_TCN_hit':hits['tcn__real']&~hits[within_lead]}
    examples=[]
    for category,mask in categories.items():
        ix=sorted(np.flatnonzero(mask&(cells==cell)),key=lambda i:records[i]['uid'])
        if not ix:continue
        i=int(ix[0]);r=records[i];row=rows[r['row_id']]
        assert int(row['label'])==int(target[i]) and len(row['steps'])==r['steps']
        examples.append(dict(category=category,uid=r['uid'],row_id=r['row_id'],cell=r['cell'],group_id=r['group_id'],
            question=row['problem'],steps=row['steps'],first_error_step=int(target[i])+1,gate_open=bool(gate[i]),
            predictions={n:dict(step=None if pred[n][i]<0 else int(pred[n][i])+1,raw_peak=int(peaks[n][i])+1,
                scores=scores[n][offset[i]:offset[i+1]].tolist()) for n in roster}))
    with (OUT/'PB_ERROR_LEDGER.csv').open('w',encoding='utf8',newline='') as f:
        names=list(METHODS)+['tcn__real','ridge','bocpd','innovation5'];writer=csv.writer(f)
        writer.writerow(['uid','row_id','cell','source_group','first_error_step_1based','steps','gate_open','all32_miss']+names)
        for i in np.flatnonzero(error):
            r=records[i];writer.writerow([r['uid'],r['row_id'],r['cell'],r['group_id'],int(target[i])+1,r['steps'],bool(gate[i]),bool(allmiss[i])]+[int(pred[n][i])+1 if pred[n][i]>=0 else 'CLEAN' for n in names])
    write(OUT/'ERROR_ANALYSIS.json',dict(scope='Post-selection descriptive PB first-error analysis; no new fitting or significance tests. Rows are model-answer records, not independent source questions.',
        summary=summary,overlap=overlap,examples=examples,code_sha256=sha(Path(__file__)),scores_sha256=sha(OUT/'SCORES_FROZEN.npz'),
        ledger_sha256=sha(OUT/'PB_ERROR_LEDGER.csv')))
    text=['# אילו שגיאות הוחמצו — ניתוח תיאורי לאחר בחירת הצירופים',
        'הספירות הן תשובות ProcessBench עם שגיאה ידועה. אותה שאלת מקור יכולה להופיע תחת שני מודלים; אלה אינן ספירות של שאלות עצמאיות. הצעד הנכון הוא השגיאה הראשונה לפי התיוג המקורי. מספרי הצעדים מתחילים ב-1. CLEAN משמעו שה-gate סגר את התשובה.',
        '| תצורה | אותרו בדיוק | ה-gate סגר | נבחר מוקדם מדי | נבחר מאוחר מדי | נוספו מול TCN | אבדו מול TCN |\n|---|---:|---:|---:|---:|---:|---:|\n'+''.join('| '+n+' | '+' | '.join(str(x) for x in [summary[n]['hits'],summary[n]['gate_closed'],summary[n]['early'],summary[n]['late'],summary[n]['versus']['tcn__real']['gained'],summary[n]['versus']['tcn__real']['lost']])+' |\n' for n in roster),
        f"מתוך {overlap['error_answers']} תשובות עם שגיאה, כל 32 הצירופים מחמיצים {overlap['missed_by_all32']}: ב-{overlap['missed_all_gate_closed']} ה-gate סגור וב-{overlap['missed_all_gate_open']} הוא פתוח אך אף צירוף אינו בוחר את הצעד הנכון. בתוך קבוצת ה-gate הסגור, לפחות צירוף אחד הצביע נכון לפני הסגירה ב-{overlap['gate_closed_but_correct_raw_peak_exists']} תשובות. איחוד הפגיעות הוא אורקל בין הפסגות הקיימות, לא אלגוריתם זמין ולא חסם על מיזוג חדש.",
        'הדוגמאות הבאות נבחרו לפי הקטגוריה ואז מזהה UID, מתוך GSM8K/Qwen3-4B בלבד לצורך הצגת טקסט. הן ממחישות כשלים ואינן מדגם להערכת האיכות. כל הטקסט הוא מן התשובה המקורית; ייתכנו שגיאות נוספות אחרי השגיאה הראשונה.']
    for e in examples:
        text += ['## '+e['category']+' — '+e['row_id'],e['question'],
            'השגיאה הראשונה המתויגת: צעד '+str(e['first_error_step'])+'. gate '+('פתוח' if e['gate_open'] else 'סגור')+'.',
            '| שיטה | החלטה סופית | פסגה לפני gate |\n|---|---:|---:|\n'+''.join('| '+n+' | '+str(v['step'] or 'CLEAN')+' | '+str(v['raw_peak'])+' |\n' for n,v in e['predictions'].items()),
            '\n\n'.join(str(i+1)+'. '+s for i,s in enumerate(e['steps']))]
    text+=['[כל מזהי ההחמצות והתחזיות](PB_ERROR_LEDGER.csv) · [ניתוח וספירות מלאים](ERROR_ANALYSIS.json) · [דוח הצירופים](REPORT.html).']
    (OUT/'ERRORS.md').write_text('\n\n'.join(text)+'\n',encoding='utf8')
    body=[]
    for p in text:
        if p.startswith('|'):
            lines=p.strip().splitlines();row=lambda x,tag:'<tr>'+''.join('<'+tag+'>'+html.escape(c.strip())+'</'+tag+'>' for c in x.strip('|').split('|'))+'</tr>'
            body.append('<table>'+row(lines[0],'th')+''.join(row(x,'td') for x in lines[2:])+'</table>')
        elif p.startswith('# '):body.append('<h1>'+html.escape(p[2:])+'</h1>')
        elif p.startswith('## '):body.append('<h2>'+html.escape(p[3:])+'</h2>')
        else:body.append('<p style="white-space:pre-wrap">'+re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',html.escape(p))+'</p>')
    page='<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><title>Missed errors — Step389</title><style>body{font:17px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;padding:6px}h2{margin-top:45px}p{overflow-wrap:anywhere}</style>'+''.join(body)+'</html>'
    (OUT/'ERRORS.html').write_text(page,encoding='utf8')
    print(json.dumps(dict(overlap=overlap,summary={n:summary[n] for n in roster},examples=examples),ensure_ascii=False,indent=2))

if __name__=='__main__':run()
