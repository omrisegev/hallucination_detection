"""Record the completed bounded experiment without rewriting older results."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_step_evidence_fusion import OUT,write

SUMMARY="""Step392 COMPLETE_REVIEWED: one fixed three-view step-evidence bank on all13769 answers/145597 steps. Preserved innovation5 base ONCE; compared bounded .25 corrections from TCN signed Top10, raw last4, and per-stream best contiguous10. Answer-local step covariance/standardization; maintained canonical IU2PC versus equal, three singletons, two equal pair ablations, and shape-order shuffled controls. No predictor training, alpha/gate/subset sweep, or new first-error decoder.
PB% / within / PRMScore: TCN40.9718/.761592/.641153; end38.0611/.761946/.642517; sustained38.3463/.754040/.638746; equal three39.6301/.761989/.643185; IU three39.0007/.755534/.639643. Equal context+end39.5281/.765898/.644000 is a secondary Pareto tradeoff, not a replacement. Equal context+sustained39.8711/.759397/.640088.
Five primary pairs x2 endpoints;10000 source-group draws,CI99.5%. IU-TCN PB-1.9712pp[-3.1357,-.7801],within-.006058[-.009134,-.003069]. Equal-TCN PB-1.3418pp[-2.5859,-.1335],within+.000397[-.002084,+.002921]. IU-equal within-.006455[-.008535,-.004398],PB interval includes0. True shape order improves within over shuffle for both heads, with adjusted positive intervals, but PB intervals include0.
Secondary context+end versusTCN: within+.004306,exploratory95%CI[+.002627,+.005931]; PB-1.4437pp[-2.3416,-.5778]. Earlier bank and last4 motivation used development outcomes; no untouched confirmation or primary multiplicity claim for this ablation.
Final PB error hits: TCN1334; equal1256(gain143/lose221); IU1232(gain120/lose222); context+end1254(gain156/lose236). Among fixed885 prior raw misses, equal finds1 raw/0 final;IU1/1;context+end5/3. Direct last4 previously recovered119 raw, but a bounded last4 correction is a different detector; no contradiction and no demonstrated recovery of the broad missed cohort.
Native IU13710/13769, explicit equal fallback59 answers with fewer than3 steps. Median weights[.20871,.13505,.25262];59.19% native fits contain a negative coefficient;99.16% g2 at ceiling;42.58% have a Spearman pair>=.75. Three-pair identity is exactly identified, not an assumption test. Few steps remain a limitation. No correlated-view automatic bank search.
Reused15 source-excluded TCN score sets including10 pair exclusions for PRMScore calibration. All27 metric bundles independently audited for PB/within;18 reference headlines and context identity replayed. Five unit tests pass; all-answer weighted-score replay max1.78e-15, baseline max5.33e-15, canonical weight max4.58e-16. Scoring150.68s plus45.36s evaluation; existing FM/DiFlo queue unchanged.
Decision: do not adopt three-view equal/IU; retain context+end as a development tradeoff for within, and preserve current references. No automatic follow-up sweep. Report results/step_evidence_fusion_v1/REPORT.html; full table, cell metrics, corrected intervals, error ledger, input/score hashes and scope saved.
"""


def run():
    state=json.loads((OUT/'RUN_STATE.json').read_text())
    if state['status']!='COMPLETE_REVIEWED':raise ValueError('Incomplete study')
    header='## Step392 - complementary step evidence fusion\n\n'
    for name in ('PROGRESS.md','Research_Directions.md'):
        p=ROOT/name;old=p.read_bytes()
        if b'## Step392 - complementary step evidence fusion' not in old:
            nl='\r\n' if b'\r\n' in old else '\n'
            p.write_bytes((header+SUMMARY+'\n').replace('\n',nl).encode('utf8')+old)
    p=ROOT/'HISTORY.md';old=p.read_bytes()
    title='### Step 392 - Test complementary step-evidence fusion [Codex]'
    if title.encode() not in old:
        nl='\r\n' if b'\r\n' in old else '\n'
        text='\n'+title+'\n\n**What**: Try the user-authorized fusion of context and shape evidence, following the common-miss audit and a review of prior Top10/contiguous/order-statistic experiments.\n\n**Why**: Test whether complementary evidence improves the correction to the successful base, and isolate IU from equal aggregation.\n\n**Result**:\n'+SUMMARY+'\n**Files changed**: spectral_utils/step_evidence_fusion.py; scripts/run_step_evidence_fusion.py, evaluate_step_evidence_fusion.py, report_step_evidence_fusion.py, update_step_evidence_docs.py; tests/test_step_evidence_fusion.py; docs/experiments/STEP_EVIDENCE_FUSION_20260915.md; results/step_evidence_fusion_v1; central execution report and research logs.\n\n---\n'
        p.write_bytes(old+text.replace('\n',nl).encode('utf8'))
    p=ROOT/'docs/reviews/temporal_research_execution_2026-09-15.html'
    old=p.read_bytes();s=old.decode('utf8')
    if 'id="step392-evidence"' not in s:
        section='''<section id="step392-evidence"><h2>Step392 — שילוב ראיות לצעד: לא נמצא שיפור PB</h2><p>הושלם ניסוי מלא על 13,769 תשובות. הבסיס innovation5 נשמר פעם אחת, והתיקון משלב חריגה מתחזית TCN, ארבעת הטוקנים האחרונים וחריגה רציפה בחלון של עשרה טוקנים. כל רכיב נבדק לבד, בממוצע וב־IU-PCR, באותו gate ובעוצמת תיקון .25.</p><table><tr><th>תיקון</th><th>PB%</th><th>within</th><th>PRMScore</th></tr><tr><td>TCN הקיים</td><td>40.9718</td><td>.761592</td><td>.641153</td></tr><tr><td>ממוצע שלוש הראיות</td><td>39.6301</td><td>.761989</td><td>.643185</td></tr><tr><td>IU-PCR שלוש הראיות</td><td>39.0007</td><td>.755534</td><td>.639643</td></tr><tr><td>ממוצע הקשר + סוף הצעד</td><td>39.5281</td><td>.765898</td><td>.644000</td></tr></table><p>ירידות PB של השילובים המשולשים מול TCN מובחנות ברווחי סמך ראשיים של 99.5%; IU פוגע גם ב־within. השילוב הזוגי מציג פשרת פיתוח בדירוג בתוך תשובה, אך אינו מחליף את המוביל באיתור הראשון. מתוך 707 ההחמצות הקודמות עם gate פתוח, הממוצע המשולש מציל אפס ו־IU אחת. 59 תשובות קצרות עברו במפורש לממוצע במקום התאמת IU. כל 27 השורות נבדקו בנפרד; התור העצבי לא חודש.</p><p><a href="../../results/step_evidence_fusion_v1/REPORT.html">הדוח המלא, הפגיעות שנוספו ואבדו ורווחי הסמך</a> · <a href="../experiments/STEP_EVIDENCE_FUSION_20260915.md">הפרוטוקול</a></p></section>'''
        end=s.index('</h1>')+5;s=s[:end]+section+s[end:];p.write_bytes(s.encode('utf8'))
    write(OUT/'DECISION.json',dict(status='REJECT_THREE_VIEW_REPLACEMENT_KEEP_SECONDARY_TRADEOFF',
        summary=SUMMARY,development_only=True,new_method_promoted=False,automatic_followup=False))
    print('Step392 documented.')


if __name__=='__main__':run()
