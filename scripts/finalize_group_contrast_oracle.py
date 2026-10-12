"""Add the checked interpretation and project handoff without changing scores."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/group_contrast_oracle_20261005'
OUT = BASE / 'run_v1'
review = json.loads((OUT / 'REVIEW.json').read_text(encoding='utf8'))
assert review['status'] == 'PASS' and review['n_metric_rows'] == 234
complete = json.loads((OUT / 'COMPLETE.json').read_text(encoding='utf8'))
for name, digest in complete['artifacts'].items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
report_path = OUT / 'REPORT_HE.html'
report = report_path.read_text(encoding='utf8')
section = '''<h2>המסקנה מהניסוי</h2>
<p><strong>באבחון הלינארי הזה, ניגודים בין ציוני הקבוצות אינם מספיקים לסגירת הפער. רוב השיפור הנוסף מופיע כשמחזירים מידע מתוך הקבוצות.</strong></p>
<p>ב־PRMBench, LR על ציוני הקבוצות מגיע ל־0.783899 ב־B13 ול־0.761303 ב־B32. החזרת כל הניגודים הפנימיים מעלה זאת ל־0.812610 ול־0.804840. ההפרשים הם 0.028710 עם רווח חקרני [0.024840, 0.032613] ו־0.043538 עם רווח [0.038964, 0.048105], בהתאמה. המדד חושב על 6,030 תשובות בכל בנק.</p>
<p>LR על כל הפיצ׳רים נותן 0.812462 ו־0.806071. הקרבה הנקודתית אינה הוכחת שקילות: ב־B32 נותר פער של 0.001231 לטובת הייחוס המלא, ורווח ההפרש [0.000417, 0.002023] חיובי.</p>
<p>אין יתרון במתן משקלים שליליים לקבוצות: ב־PRMBench B13 לא נלמד אף משקל שלילי ב־15 המקדמים של חמשת הקיפולים, והתוצאה זהה לאילוץ משקלים לא שליליים. ב־B32 נלמדו חמישה משקלים שליליים מתוך 30, אך האילוץ הלא־שלילי נתן AUC מעט גבוה יותר: 0.761881. גם ב־ProcessBench השחרור לסימנים שליליים לא העלה את המדד הממוצע.</p>
<p>ב־ProcessBench, על 4,442 תשובות שגויות, החזרת מידע פנימי מעלה את B13 מ־0.361992 ל־0.387736 ואת B32 מ־0.358033 ל־0.412449. רווחי ההפרש הם [0.010715, 0.040589] ו־[0.036689, 0.072763]. אין הוכחת יתרון על LR המלא.</p>
<p>התקנון של ממוצעי הקבוצות אינו מסביר את רוב החסר: שימוש במרחב ממוצעי הקבוצות לפני התקנון נותן AUC של 0.784248 ו־0.762822 בלבד. המגבלה המרכזית בבדיקה היא דחיסת כל קבוצה לציון יחיד.</p>
<p>המשך אפשרי הוא לשמור לכל קבוצה ציון משותף וגם ייצוג של ההבדלים בתוכה. לא הוכח שניגוד אחד לכל קבוצה מספיק, ולא נלמד כאן כלל ללא תוויות לבחירת הניגודים או המשקלים. כל התוצאות חקרניות ומפוקחות.</p>
'''
if '<h2>המסקנה מהניסוי</h2>' not in report:
    report = report.replace('<h2>הגדרות ופירוש</h2>', section + '<h2>הגדרות ופירוש</h2>')
    report_path.write_text(report, encoding='utf8')
for target in re.findall(r'href="([^"]+)"', report):
    assert (OUT / target).resolve().exists(), target
assert 'המסקנה מהניסוי' in report_path.read_text(encoding='utf8')
dest = OUT / 'implementation' / Path(__file__).name
if dest.exists():
    assert dest.read_bytes() == Path(__file__).read_bytes()
else:
    shutil.copyfile(__file__, dest)
context = {'status': 'COMPLETE_REVIEWED', 'command': 'C:/Users/DELL/AppData/Local/Programs/Python/Python313/python.exe -B C:/Users/omris/TAU/hallucination_detection/scripts/run_group_contrast_oracle.py',
           'tests': '4/4 new tests PASS; shared previously tested solver and bootstrap reused',
           'population': '13769 answers per bank, 6030 PRMB eligible, 4442 erroneous PB',
           'scope': 'Supervised development diagnosis; no new label-free performance claim',
           'execution': 'CPU only, two numerical threads, no GPU/inference, existing score files unchanged',
           'checkout': 'Shared dirty codex/lsml-external-generalization-v1; new isolated filenames and result directory'}
(OUT / 'RUN_CONTEXT.json').write_text(json.dumps(context, indent=2), encoding='utf8')
note = '''

## 2026-10-05 - Group-level contrast oracle complete (Codex)

Omri authorized testing whether contrasts can act between existing groups instead of between features. Report: results/group_contrast_oracle_20261005/run_v1/REPORT_HE.html. All 18 B13/B32 cells completed, 13,769 answers per bank; 6,030 PRMB mixed-label and 4,442 erroneous PB answers define the endpoints. Same frozen group matrices, DS survivors and source folds; only diagnostic supervised OOF readouts changed. No GPU/inference or label-free candidate result.

PRMB B13/B32: existing MEAN_MOM .781659/.753675; LR on native group scores .783899/.761303; adding all within-group contrasts .812610/.804840; full-feature LR .812462/.806071. Within-group additions gain .028710 [.024840,.032613] and .043538 [.038964,.048105]. Native group signed LR does not beat its nonnegative counterpart: B13 equal; B32 nonnegative .761881. Using raw group means before answer standardization only reaches .784248/.762822. The scalar group compression, not merely between-group weighting or normalization, loses much of the linearly recoverable signal.

PB group LR -> groups plus within contrasts: .361992 -> .387736 (B13), .358033 -> .412449 (B32); paired intervals positive for both additions. Near full-LR point estimates are not equivalence evidence. Four new unit tests, 90 fold fits, 234 independent metric rows, 48 paired comparisons (5000 source-question draws each), exact historical controls, full-span equivalence checks and unchanged input/code hashes PASS. Follow-up question: preserve internal group structure while learning the final fusion without correctness labels; one internal direction per group has not been established sufficient.
'''
progress = ROOT / 'PROGRESS.md'
if '## 2026-10-05 - Group-level contrast oracle complete (Codex)' not in progress.read_text(encoding='utf8'):
    with progress.open('a', encoding='utf8') as f:
        f.write(note)
history = ROOT / 'HISTORY.md'
heading = '### Step 448 [Codex group-contrast oracle] - Scalar group scores leave most of the recoverable gap inside groups, 2026-10-05'
if heading not in history.read_text(encoding='utf8'):
    with history.open('a', encoding='utf8') as f:
        f.write('\n\n' + heading + '\n\n**What**: User-authorized supervised OOF comparison of existing group fusion, signed/nonnegative group LR, and group scores plus within-group contrasts, on all B13/B32 cells.\n**Why**: Test whether the contrast mechanism can be reduced to interactions between scalar group scores.\n**Result**: PRMB group LR .783899/.761303 rises to .812610/.804840 after internal contrasts (6,030 eligible answers); signed coefficients do not improve on nonnegative group LR. PB internal additions improve both banks (4,442 erroneous answers). Full results/protocol/model weights: results/group_contrast_oracle_20261005/run_v1/. Four tests and 234 independent metric rows pass. Supervised exploratory evidence only; historical data and scores preserved.\n')
(BASE / 'LATEST.json').write_text(json.dumps({'status': 'COMPLETE_REVIEWED', 'run': 'run_v1', 'report': 'run_v1/REPORT_HE.html'}, indent=2), encoding='utf8')
(OUT / 'DELIVERY_MANIFEST.json').write_text(json.dumps({str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'DELIVERY_MANIFEST.json'}, indent=2), encoding='utf8')
print('FINALIZED: interpretation, artifact hashes, links, Hebrew, progress and history', flush=True)
