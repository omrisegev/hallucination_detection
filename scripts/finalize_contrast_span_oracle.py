"""Preserve the reviewed run, add interpretation and append the project handoff."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/contrast_span_oracle_20261005/run_v3'
review = json.loads((OUT / 'REVIEW.json').read_text(encoding='utf8'))
assert review['status'] == 'PASS' and review['n_cells'] == 18
assert review['n_metric_rows_independently_replayed'] == 81
assert review['n_unit_tests'] == 7
gates = pd.read_json(OUT / 'GATES.json')
assert (gates.groupby('bank').n_answers.sum() == 13769).all()
summary = pd.read_csv(OUT / 'SUMMARY.csv')
assert set(summary[summary.panel == 'PRMB'].n_checked) == {6030}
assert set(summary[summary.panel == 'PB'].n_checked) == {4442}
assert len(pd.read_csv(OUT / 'PAIRED_CONTRASTS.csv')) == 48
report_path = OUT / 'REPORT_HE.html'
report = report_path.read_text(encoding='utf8')
marker = '<h2>פירוש התוצאה</h2>'
if marker not in report:
    interpretation = '''<h2>פירוש התוצאה</h2>
<p><strong>המכלול מועיל יותר מהניגוד היחיד באבחון המפוקח.</strong> ב־PRMBench B13 הניגוד היחיד מגיע ל־0.8046, מכלול הניגודים בתוך הקבוצות ל־0.8107, ו־LR על כל הפיצ׳רים ל־0.8125. הניגוד היחיד סוגר כ־76% מהפער מהממוצע ל־LR, ומכלול ניגודי הקבוצות כ־94%. ב־B32 המכלול סוגר כ־90% מהפער, ומגיע ל־0.8000 לעומת 0.8061 של LR המלא.</p>
<p>השיפור של המכלול לעומת הניגוד היחיד ב־B13 הוא 0.0061 AUC, עם רווח חקרני של 95% מ־0.0034 עד 0.0086. נותר פער קטן אך חיובי מול LR המלא בשני הבנקים; לכן לא נטענת שקילות סטטיסטית.</p>
<p>זהו מרחב רחב: ב־PRMBench B13 הגורם המשותף עם ניגודי הקבוצות תופסים 9 ממדים מתוך 11 פיצ׳רים ששרדו; ב־B32 הם תופסים 22 מתוך 27. התוצאה אינה הוכחה שרכיב יחיד נוסף מספיק.</p>
<p>ב־ProcessBench, על 4,442 תשובות שגויות, ניגודי הקבוצות עם הגורם המשותף נותנים 0.3812 ב־B13 לעומת 0.3698 לממוצע; רווח ההפרש כולל אפס. ב־B32 מתקבל 0.4086 לעומת 0.3236 לממוצע, עם רווח הפרש של 0.0633 עד 0.1071. הערך קרוב בנקודה ל־LR המלא, אך קרבה זו אינה מוכיחה שקילות.</p>
<p>תקנון הניגודים בתוך תשובה נותן ב־B13 AUC של 0.8131, אך אינו עדיף בבירור על LR המלא ורמת האיתור שלו ב־ProcessBench נמוכה יותר. אין כאן בחירה בגרסה הזו. מסקנת העבודה היא לשמור על מכלול הניגודים כיעד לפיתוח כלל משקלים ללא תוויות; הניסוי הנוכחי מדד מה ניתן לנצל כשיש תוויות.</p>
'''
    report = report.replace('<h2>מה הבדיקות מזהות</h2>', interpretation + '<h2>מה הבדיקות מזהות</h2>')
    report_path.write_text(report, encoding='utf8')
for target in re.findall(r'href="([^"]+)"', report):
    assert (OUT / target).resolve().exists(), target
assert report.count('ניגוד') > 5
implementation = OUT / 'implementation'
implementation.mkdir(exist_ok=True)
for relative in ['spectral_utils/contrast_span_oracle.py', 'scripts/run_contrast_span_oracle.py',
                 'scripts/review_contrast_span_oracle.py', 'scripts/finalize_contrast_span_oracle.py',
                 'tests/test_contrast_span_oracle.py']:
    destination = implementation / Path(relative).name
    if destination.exists():
        assert destination.read_bytes() == (ROOT / relative).read_bytes()
    else:
        shutil.copyfile(ROOT / relative, destination)

context = {'status': 'COMPLETE_REVIEWED', 'run': 'run_v3', 'n_unit_tests': 7,
           'command': 'C:/Users/DELL/AppData/Local/Programs/Python/Python313/python.exe -B C:/Users/omris/TAU/hallucination_detection/scripts/run_contrast_span_oracle.py',
           'review_command': 'python -B scripts/review_contrast_span_oracle.py',
           'population': '13769 unique answers per bank; 6030 PRMB and 4442 erroneous PB metric-eligible answers',
           'historical_reference_sha256': hashlib.sha256((ROOT / 'results/mom_filter_within_diagnostic_20261004/CONTRAST_STRUCT_RESULTS.csv').read_bytes()).hexdigest(),
           'notes': ['No GPU or inference', 'No historical scores modified', 'Incomplete attempts preserved',
                     'No commit or branch switch in the shared dirty checkout', 'No new unsupervised performance claim']}
(OUT / 'RUN_CONTEXT.json').write_text(json.dumps(context, indent=2), encoding='utf8')
note = '''

## 2026-10-05 - Contrast-span oracle diagnostic complete (Codex)

Omri requested the proposed oracle experiment and emphasized the whole contrast ensemble. Full reviewed artifacts: results/contrast_span_oracle_20261005/run_v3/REPORT_HE.html (SUMMARY.csv, PAIRED_CONTRASTS.csv, REVIEW.json, COMPLETE.json). Two banks x nine cells, 13,769 answers per bank; endpoints use 6,030 mixed-label PRMB answers and 4,442 erroneous PB answers. Supervised C=1 balanced LR readouts use source-question-disjoint OOF training; historical unlabeled per-cell DS survivors/groups remain transductive. No label-free method gain or untouched confirmation claimed.

PRMB: plain B13/B32 .780179/.746716; PC1 plus all within-group linear contrasts .810653/.800011; full LR .812462/.806071. The named single contrast in B13 gives .804586. Group contrasts recover 94%/90% of the plain-to-LR point gap versus 76% for the single contrast in B13. B32 has no realized_drv, so that named arm is explicitly unavailable. Complete pairwise contrasts plus mean and PC1 plus all residuals reproduce full LR by construction; group-restricted spans are 9/11 and 22/27 dimensions. PB group contrasts .381185/.408615 versus plain .369790/.323610; B13 interval includes zero, B32 gain interval [.063341,.107052].

Validation: seven tests, all 18 cells, 90 fold fits, eight historical aggregate replays, 81 independently recomputed metric rows, 5,000 paired source-group bootstrap draws per comparison, unchanged input/code hashes. First attempt stopped on B32 missing feature; second stopped at the unchanged numerical rotation gate. Both preserved. Final solver Newton-refines the same C=1 objective; legacy replay keeps its original solver. No historical score file changed. Next research question: learn a joint direction over the contrast space without correctness labels; no follow-on run authorized by this note.
'''
progress = ROOT / 'PROGRESS.md'
if '## 2026-10-05 - Contrast-span oracle diagnostic complete (Codex)' not in progress.read_text(encoding='utf8'):
    with progress.open('a', encoding='utf8') as f:
        f.write(note)
history = ROOT / 'HISTORY.md'
heading = '### Step 447 [Codex contrast-span oracle] - Joint contrasts recover most of the supervised gap, 2026-10-05'
if heading not in history.read_text(encoding='utf8'):
    with history.open('a', encoding='utf8') as f:
        f.write('\n\n' + heading + '\n\n**What**: User-requested supervised OOF diagnostic comparing one named contrast, all within-group contrasts, complete pairwise contrasts, and full residual/full-feature references on B13/B32, all nine cells.\n**Why**: Test whether a single contrast explains the missing signal or a broader joint representation is required.\n**Result**: Group-restricted linear spans recover 94%/90% of the PRMB plain-to-LR point gap (6,030 eligible answers); B13 single contrast recovers 76%. No named single contrast exists in B32. PB group contrast scores .381185/.408615 on 4,442 erroneous answers; B13 gain interval includes zero. All results are supervised development diagnostics, not a label-free improvement. Seven tests and 81 independent metric replays pass. Complete report/provenance: results/contrast_span_oracle_20261005/run_v3/. Initial failed attempts preserved; no historical score changes.\n')
lesson = '''

## 2026-10-05 - A solver stopping tolerance is not a coefficient-equivalence guarantee
What happened: the full-space rotation gate stopped the second contrast-span attempt at a score difference of 1.3667e-5, above the registered 1e-5 threshold, despite L-BFGS tol=1e-9. Earlier in the session, the named contrast was found absent from B32. Both incomplete attempts and their implementations were preserved.
Why: relative objective stopping can leave enough coefficient error to distinguish equivalent rotated representations; the initial feature-availability assumption was also too broad.
Rule: verify actual feature rosters and explicitly mark unavailable arms. For equivalence controls, check optimality of the same objective instead of relaxing the comparison gate.
Enforced by: code (training-gradient Newton refinement with unchanged balanced C=1 objective, missing-feature handling, seven tests including collinear rotation and fold isolation). Full run_v3 passed unchanged gates and independent metric replay.
'''
lessons = ROOT / 'LESSONS.md'
if '## 2026-10-05 - A solver stopping tolerance is not a coefficient-equivalence guarantee' not in lessons.read_text(encoding='utf8'):
    with lessons.open('a', encoding='utf8') as f:
        f.write(lesson)
delivery = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'DELIVERY_MANIFEST.json'}
(OUT / 'DELIVERY_MANIFEST.json').write_text(json.dumps(delivery, indent=2), encoding='utf8')
print('FINALIZED: report links and Hebrew encoding checked;', len(delivery), 'sealed artifacts; handoff appended', flush=True)
