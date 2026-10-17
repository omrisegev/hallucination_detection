"""Preserve the reviewed digit experiment and measured limits in project logs."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_digit_fusion import OUT,write

SUMMARY="""Step393 COMPLETE_REVIEWED: independent digit-disagreement replay and a fixed12-arm full-data fusion/control experiment on all13769 answers/145597 steps/6968779 tokens. Verified ASCII digit IDs15..24 from both cached Qwen tokenizers, scalar event predicate on every token, sorted top-k, spans, baseline and Claude's auxiliary/scores. One teacher-forced pass reused; no model training, operator extension, gate change or flow restart.
PB%/within/PRMScore: base39.8314/.760293/.638830; digit02541.3300/.776036/.649780; digit1 41.1806/.779274/.649223; TCN40.9718/.761592/.641153; TCN+digit sum42.0781/.774945/.652284; amplitude-matched TCN+digit41.4627/.772607/.649612.
Controls: digit presence39.7521/.771779; disagreement-per-digit rate41.1532/.774964; disagreements permuted among provided-digit positions38.9219/.760928. Real disagreement beats permuted location on both primary endpoints; within also beats presence. Rate retains a gain in points. These controls do not prove removal of all length effects.
Six primary pairs x2 endpoints,10000 source-group bootstrap draws,CI99.5833%. Digit025-base PB+1.4986pp[-.1632,+3.0960],within+.015744[+.012740,+.018865]. Same points as Claude; its historical97.5% contrast remains recorded, but the broader primary correction includes0 for PB. Digit025-permuted PB+2.4080pp[+.8932,+3.8858],within+.015109[+.011878,+.018292]. Matched TCN+digit versusTCN: within+.011015[+.008581,+.013455],PB interval[-.7061,+1.7018]pp includes0.
The42.0781% sum is a prespecified SECONDARY arm: versusTCN PB+1.1062pp,exploratory95%CI[+.1269,+2.1044],within+.013353[+.011450,+.015226]. Keep as development candidate/Pareto tradeoff; no primary-confirmed or untouched winner claim. Higher gamma and auxiliary selection were already exposed to development outcomes.
Direct bank comparison, same token standardization/per-stream Top10/readout: bank5 equal39.3043/.755063;bank5 IU39.3825/.757218;bank6 equal41.3961/.773468;bank6 IU30.5284/.637115. IU6 loses to equal6 and IU5 with adjusted negative intervals in both endpoints. Digit coefficient negative in99.98% of10282 variable-digit answers,median-.27978;30 additional variable-digit canonical fits reproduce weights. Not convergence to equal: the new view is largely subtracted. No label-guided sign fix applied.
Digit is constant in3487 answers. Participation ratio on mean within-answer covariance: bank5 1.34024 -> bank6 1.71884; among variable-digit answers1.33970 ->1.83060. Claude's3.55 was for ALL seven new views, not digit alone. Low covariance and higher effective dimension do not imply a more accurate IU reliability estimate.
Final error hits: digit0251339(+310/-239 vsbase; common885 raw30/open707 final28); TCN+digit1385(+255/-204 vsTCN,net51; common885 raw16/open707 final14); matched1358(net24 vsTCN); equal6 1354(common70759). Standalone digit hits145/707,143 with positive digit evidence at gold; do not count zero ties as positive telemetry. Full4442-row ledger and length/digit-count strata saved.
All30 PB/within metric bundles independently audited;18 reference rows and all3 Claude rows replay exactly, including PRMScore. Nested PRMScore calibration reuses15 excluded-group TCN score sets; no cross-fold model exposure. Five new tests passed. Source, tokenizer, score and model-score provenance saved.
Decision: retain digit correction and TCN+digit as useful development fusion candidates; reject this native IU6 bank. Do not equate simple-combination gains with learned-IU gains. Next algorithmic question is how to respect a useful sparse view's direction while estimating reliability under the dependent old bank; no automatic post-result clipping or hyperparameter sweep in this stage. Report results/digit_fusion_v1/REPORT.html.
"""


def run():
    assert json.loads((OUT/'RUN_STATE.json').read_text())['status']=='COMPLETE_REVIEWED'
    heading='## Step393 - digit disagreement replay and fusion\n\n'
    for name in ('PROGRESS.md','Research_Directions.md'):
        p=ROOT/name;old=p.read_bytes();nl='\r\n' if b'\r\n' in old else '\n'
        if heading.split('\n')[0].encode() not in old:p.write_bytes((heading+SUMMARY+'\n').replace('\n',nl).encode()+old)
    p=ROOT/'HISTORY.md';old=p.read_bytes();title='### Step 393 - Replay digit disagreement and test its fusion contribution [Codex]'
    if title.encode() not in old:
        nl='\r\n' if b'\r\n' in old else '\n'
        entry='\n'+title+'\n\n**What**: Independently check Claude\'s new cached digit view and compare its additive/context and spectral fusion under fixed controls.\n\n**Why**: User asked whether genuinely complementary information can improve the algorithm and make fusion useful.\n\n**Result**:\n'+SUMMARY+'\n**Files changed**: spectral_utils/digit_fusion.py; scripts/run_digit_fusion.py, evaluate_digit_fusion.py, report_digit_fusion.py, update_digit_fusion_docs.py; tests/test_digit_fusion.py; docs/experiments/DIGIT_FUSION_20260915.md; results/digit_fusion_v1 summaries, ledger and audits; research logs and central HTML.\n\n---\n'
        p.write_bytes(old+entry.replace('\n',nl).encode())
    p=ROOT/'docs/reviews/temporal_research_execution_2026-09-15.html';s=p.read_bytes().decode('utf8')
    if 'id="step393-digit"' not in s:
        block='''<section id="step393-digit"><h2>Step393 — תצוגת ספרות מועילה לשילוב; IU הקנוני מדכא אותה</h2><p>שחזור עצמאי מלא מאשר את תוצאת Claude: בסיס + ספרות נותן 41.3300% PB ו־0.776036 within. הוספת תיקון הספרות ל־TCN מגיעה ל־42.0781% ו־0.774945, ו־PRMScore של 0.652284. זהו מועמד פיתוח משני, לא אישור ראשי או חיצוני; גרסת עוצמה תואמת נותנת 41.4627% ו־0.772607, עם יתרון within ראשי ו־PB שעדיין אינו מוכרע.</p><p>הניסוי נערך על כל 13,769 התשובות. בקרת מיקומי ספרות מעורבבים מפסידה לאי־ההסכמה האמיתית בשני המדדים ברווחי סמך ראשיים של 99.5833%. בקרת נוכחות הספרות מסבירה חלק מהשיפור בדירוג אך לא את כולו. תוספת הספרות לבנק מתוקנן נותנת בממוצע 41.3961%/.773468; IU-PCR באותו בנק יורד ל־30.5284%/.637115. מקדם הספרות שלילי ב־99.98% מהתשובות שבהן הזרם משתנה, ואומת מול הפונקציה הקנונית. זו מגבלת אומדן שנמדדה, לא היעדר מידע חדש.</p><p>TCN+ספרות מוסיף 255 פגיעות ומאבד 204 מול TCN, נטו 51; הוא מציל 14 מתוך 707 ההחמצות הפתוחות הקודמות. יחס ההשתתפות של הבנק עולה מ־1.34 ל־1.72, לא ל־3.55 — הנתון האחרון התייחס לכל שבע התצוגות החדשות. אין ניסוי gate חדש, הרחבת אופרטורים או חידוש תור flows.</p><p><a href="../../results/digit_fusion_v1/REPORT.html">הדוח המלא והבקרות</a> · <a href="../experiments/DIGIT_FUSION_20260915.md">הפרוטוקול</a></p></section>'''
        j=s.index('</h1>')+5;p.write_bytes((s[:j]+block+s[j:]).encode('utf8'))
    write(OUT/'DECISION.json',dict(status='KEEP_DIGIT_DEVELOPMENT_CANDIDATES_REJECT_NATIVE_IU6',
        summary=SUMMARY,primary_confirmed_pb_winner=False,untouched_confirmation=False,
        next_sweep_started=False))
    print('Step393 documented.')


if __name__=='__main__':run()
