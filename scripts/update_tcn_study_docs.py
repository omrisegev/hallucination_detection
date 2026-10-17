"""Publish the completed Step388 summary without normalizing old line endings."""
from pathlib import Path
import sys,json,re,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_tcn_aligned_study import OUT


def read(path):
    return path.read_bytes().decode('utf8')


def put(path,text):
    path.write_bytes(text.encode('utf8'))


def run():
    state=json.loads(read(OUT/'RUN_STATE.json'))
    if state['status']!='COMPLETE_REVIEWED':raise ValueError('Finish the full study first')
    data=json.loads(read(OUT/'METRICS.json'));m=data['metrics']
    review=json.loads(read(OUT/'REVIEW.json'))
    def point(k):
        v=m[k];return f"{100*v['pb_all8']:.4f}% / {v['prm_within']:.6f} / {v['prmscore_q08']:.6f}"
    def contrast(k):
        c=data['contrasts']['tcn__real_minus_'+k]
        return (f"PB {100*c['pb_delta']:+.4f}pp [{100*c['pb_ci'][0]:+.4f},{100*c['pb_ci'][1]:+.4f}]; "
                f"within {c['prm_within_delta_common']:+.6f} [{c['prm_within_ci'][0]:+.6f},{c['prm_within_ci'][1]:+.6f}]")
    lines=[f"- {k}: {point(k)}" for k in ['tcn__real','ridge','bocpd','noreset','innovation5','tcn__shuffled','tcn__zero']]
    evidence="\n".join(lines)
    inference="\n".join(f"- TCN minus {k}: {contrast(k)}" for k in ['ridge','bocpd','noreset','innovation5','tcn__shuffled','tcn__zero'])
    summary=(
        "LATEST — Step388 [Codex TCN] COMPLETE_REVIEWED, standalone predictors before fusion.\n"
        "All13769 answers/145597 steps/6968779 tokens;15 source-excluded fits, seed0 only.\n"
        "Existing TCN, innovation5, signed .25 residual correction, Top10 and tail15 gate\n"
        "unchanged. Reused old fold0; completed4 outer+10 pair-exclusion calibration fits.\n"
        "PB% / within / PRMScore:\n"+evidence+"\n\n"
        "6 primary pairs x2 endpoints;10000 source-group bootstrap draws,CI99.5833%:\n"+inference+"\n\n"+
        f"Prediction MSE: TCN {data['prediction']['tcn__real']['scalar_mse_answer_mean']:.6f}, "
        f"Ridge {data['reference_prediction']['ridge']['scalar_mse_answer_mean']:.6f}.\n"
        "TCN/Ridge signed residual median correlation .963236; after removing shared\n"
        "current scalar .902422. Better telemetry prediction is not a detection proof.\n"
        "Architecture audit:16 input slots but15 effective past tokens; unchanged.\n"
        "Shuffling16 slots can change which of the15 visible observations is omitted;\n"
        "not a pure order-only intervention. Zero-history is also an inference\n"
        "intervention; the current innovation target still contains past information.\n"
        "13 tests PASS; independent PB/within on16 methods; all13 reference headlines\n"
        "exact; signed readout independently replayed on all answers, maxdelta0.\n"
        "Source/data/checkpoint/group provenance verified; no fitting failures.\n"
        f"Decision: {review['decision']}.\n"
        "Development evidence only; bank/readout selected earlier on this population.\n"
        "No predictor fusion fitted. Original90-job FM/DiFlo queue remains paused4/90.\n"
        "Report: results/tcn_aligned_predictor_seed0_v1/REPORT.html.\n"
        "Full audit: results/tcn_aligned_predictor_seed0_v1/AUDIT.json.\n\n")
    p=ROOT/'PROGRESS.md';s=read(p)
    start=s.index('ACTIVE — Step388');end=s.index('LATEST — Step387',start)
    s=s[:start]+summary+s[end:].replace('LATEST — Step387','PREVIOUS — Step387',1)
    put(p,s)
    p=ROOT/'Research_Directions.md';s=read(p);end=s.index('## 2026-09-15 — predictors before fusion (Step387)')
    put(p,"## 2026-09-15 — complete TCN aligned predictor (Step388)\n\n"+summary+s[end:])
    old387="""## מנבאים מיושרים לפני Fusion — Step387

הושלמה השוואה על כל 13,769 התשובות, עם בסיס innovation5, תיקון חתום .25, Top10 ואותו gate. לא הותאם fusion בין מנבאים.

| מנבא | PB F1 | within-AUC | MSE |
|---|---:|---:|---:|
| Ridge | 40.8472% | 0.761620 | 0.688936 |
| BOCPD | 40.3676% | 0.763223 | 0.818527 |
| noreset | 39.8608% | 0.762839 | 1.015762 |
| mean16 | 39.8648% | 0.759979 | 0.957699 |
| innovation5 | 39.8314% | 0.760293 | — |

BOCPD משפר within מול innovation5: +.002930, CI99.5% [.000140,.005692]. ההשוואות מול Ridge ומול noreset כוללות אפס ברווחי הסמך הראשיים. לכן אין עדיין יתרון מוכח למנגנון שינוי-המצב. Ridge חוזה את הפיצ׳רים טוב יותר; MSE אינו בוחר את המאתר הטוב ביותר.

BOCPD מוסיף115 פגיעות PB מול Ridge ומאבד139. מתאם השאריות החציוני הוא .9103, ואחרי הסרה ליניארית של התצפית המשותפת .7258. האבחון אינו מוכיח את הנחות U-PCR או את תועלת ה-fusion. בקרת noreset נשארת נדרשת.

13 בדיקות עברו. חישוב עצמאי של PB ו-within לכל13 השיטות, שחזור9 כותרות הייחוס ושחזור readout סקלרי עם פער אפס. חמש השוואות ראשיות כפול שני מדדים,10,000 דגימות bootstrap לפי קבוצות מקור. זמן ניקוד800.84 שניות והערכה21.95 שניות. תוצאות פיתוח.

[הדוח המלא](../../results/aligned_context_predictors_v1/REPORT.html) · [הפרוטוקול](../experiments/ALIGNED_CONTEXT_PREDICTORS_20260915.md).

"""
    rows="\n".join(f"| {k} | {100*m[k]['pb_all8']:.4f}% | {m[k]['prm_within']:.6f} | {m[k]['prmscore_q08']:.6f} |"
        for k in ['tcn__real','ridge','bocpd','innovation5','tcn__shuffled','tcn__zero'])
    md388=("## השלמת TCN כמנבא לפני Fusion — Step388\n\n"
        "הושלמו15 התאמות, כולל הכיול המקונן, וכל13,769 התשובות נוקדו. seed0 בלבד.\n\n"+
        "\n\n".join(review['paragraphs_he'])+"\n\n"+
        "| שיטה | PB F1 | within-AUC | PRMScore |\n|---|---:|---:|---:|\n"+rows+"\n\n"+
        "13 בדיקות עברו; כל13 ציוני הייחוס שוחזרו. חישוב עצמאי של המדדים לכל16 השיטות ושחזור התיקון מתחזיות הטוקנים ללא פער. תוצאות פיתוח; לא הותאם fusion בין מנבאים.\n\n"+
        "[הדוח המלא](../../results/tcn_aligned_predictor_seed0_v1/REPORT.html) · [המדדים ורווחי הסמך](../../results/tcn_aligned_predictor_seed0_v1/METRICS.json) · [הפרוטוקול](../experiments/TCN_ALIGNED_PREDICTOR_20260915.md).\n\n")
    p=ROOT/'docs/reviews/temporal_research_execution_2026-09-15.md';s=read(p)
    first=s.index('## ');end=s.index('## עדכון אחרון: משקול לפי הקשר על הפיצ׳רים המקוריים',first)
    put(p,s[:first]+md388+old387+s[end:])
    p=ROOT/'docs/reviews/temporal_research_execution_2026-09-15.html';s=read(p)
    h388=('<section id="tcn-aligned-step388"><h2>השלמת TCN לפני Fusion — Step388</h2>'+
        ''.join('<p>'+html.escape(x)+'</p>' for x in review['paragraphs_he'])+
        '<p>כל13,769 התשובות;15 התאמות כולל כיול מקונן; seed0. 13 בדיקות ואימות עצמאי של המדדים עברו.</p>'+
        '<p><a href="../../results/tcn_aligned_predictor_seed0_v1/REPORT.html">דוח מלא</a> · '+
        '<a href="../../results/tcn_aligned_predictor_seed0_v1/METRICS.json">מדדים ורווחי סמך</a></p></section>')
    h387=('<section id="aligned-predictors-step387"><h2>מנבאים מיושרים לפני Fusion — Step387</h2>'+
        '<p>כל13,769 התשובות נוקדו. Ridge:40.8472% PB/.761620 within; BOCPD:40.3676%/.763223; noreset:39.8608%/.762839; mean16:39.8648%/.759979.</p>'+
        '<p>יתרון BOCPD ב-within מול innovation5: +.002930, CI99.5% [.000140,.005692]. ההשוואות מול Ridge ומול noreset כוללות אפס. חיזוי טוב יותר אינו מבטיח איתור טוב יותר; בקרת noreset נשארת נדרשת.</p>'+
        '<p>לא הותאם fusion בין מנבאים. כל13 השיטות עברו חישוב מדדים עצמאי. תוצאות פיתוח.</p>'+
        '<p><a href="../../results/aligned_context_predictors_v1/REPORT.html">הדוח המלא של Step387</a></p></section>')
    for ident,replacement in [('tcn-aligned-step388',h388),('aligned-predictors-step387',h387)]:
        s,n=re.subn(r'<section id="'+ident+r'">.*?</section>',lambda _:replacement,s,count=1,flags=re.S)
        if n!=1:raise ValueError('Missing section '+ident)
    put(p,s)
    p=ROOT/'HISTORY.md';s=read(p)
    if '### Step 388 ' in s:raise ValueError('Step388 already recorded')
    entry=(
        "\n### Step 388 — Complete aligned TCN prediction before predictor fusion [Codex]\n\n"
        "**What**: Complete the frozen TCN predictor with outer source-group exclusion\n"
        "and nested calibration. Preserve the old checkpoint, architecture, bank,\n"
        "signed residual readout and gate; score history interventions separately.\n\n"
        "**Why**: User asked to establish standalone predictor usefulness before\n"
        "attempting fusion between predictors or returning to U-PCR.\n\n"
        "**Result (PB% / within / PRMScore)**:\n"+evidence+"\n\n"
        "**Paired uncertainty**: Six primary pairs, two endpoints each;10000 source-\n"
        "group bootstrap draws,Bonferroni99.5833% intervals.\n"+inference+"\n\n"
        "**Mechanism and scope**: TCN feature MSE .647035 versus Ridge .688936;\n"
        "signed residual correlation median .963236, or .902422 after linear removal\n"
        "of the shared current scalar. This is not conditional error independence.\n"
        "The existing16-slot TCN has an effective15-token receptive field. Shuffling\n"
        "all visible slots can change the omitted observation; not pure order-only\n"
        "evidence. Zero intervention retains current innovation and relative position.\n"
        "Offline answer normalization, seed0 only, previous bank/readout selection\n"
        "on development data. No predictor fusion fitted; original flow queue paused.\n\n"
        "**Decision**: "+review['decision']+".\n\n"
        "**Validation**: All13769 answers/145597 steps/6968779 tokens,15 completed fits.\n"
        "13 tests PASS; independent PB and pairwise within for16 methods; all13\n"
        "reference headlines exact. Scalar readout replay maxdelta0 on all answers.\n"
        "Frozen data,source,checkpoint and exclusion-group audits PASS. Reused fold0\n"
        "unchanged. Transparent NTFS compression recovered1.25GiB without deleting\n"
        "data. Repaired question-mark-corrupted Hebrew in the central Step387\n"
        "summary from its intact report; numerical findings unchanged.\n\n"
        "**Files changed**:\n"
        "- scripts/{run,analyze,evaluate,report,rank,update}_tcn* — bounded execution, audits and reporting.\n"
        "- docs/experiments/TCN_ALIGNED_PREDICTOR_20260915.md — frozen contract.\n"
        "- tests/test_tcn_aligned_architecture.py and architecture audit — receptive field check.\n"
        "- results/tcn_aligned_predictor_seed0_v1/ — metrics,contrasts,reports and provenance.\n"
        "- PROGRESS,Research_Directions and central execution report — completion and interpretation.\n"
        "Large scores,SQLite token predictions and checkpoints remain local with hashes.\n\n---\n")
    put(p,s+entry)
    print('Updated Step388 documents; restored readable Step387 summaries.')


if __name__=='__main__':run()
