"""Audit and report fixed orthogonal-family weighting without changing scores."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import html
import importlib.util
import json
import re
import shutil
import numpy as np
import pandas as pd
import run_contrast_span_oracle as infra

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/orthogonal_family_fusion_20261005'
OUT = BASE / 'run_v1'
COMMAND = 'C:/Users/DELL/AppData/Local/Programs/Python/Python313/python.exe -B C:/Users/omris/TAU/hallucination_detection/scripts/run_orthogonal_family_fusion.py'
complete = json.loads((OUT / 'COMPLETE.json').read_text(encoding='utf8'))
assert complete['status'] == 'PASS' and complete['n_cells'] == 18
for name, digest in complete['artifacts'].items():
    assert infra.sha(OUT / name) == digest, name
seal = json.loads((OUT / 'INPUT_CODE_SEAL.json').read_text(encoding='utf8'))
for section in ['inputs','code']:
    for path, digest in seal[section].items():
        assert infra.sha(path) == digest, path
summary = pd.read_csv(OUT / 'SUMMARY.csv')
paired = pd.read_csv(OUT / 'PAIRED_CONTRASTS.csv')
cells = pd.read_csv(OUT / 'PER_CELL.csv')
weights = pd.read_csv(OUT / 'WITHIN_WEIGHTS.csv')
gates = json.loads((OUT / 'GATES.json').read_text(encoding='utf8'))
review = json.loads((OUT / 'REVIEW.json').read_text(encoding='utf8'))
failures = json.loads((OUT / 'FAILURES.json').read_text(encoding='utf8'))
assert set(paired.valid_draws) == {5000}
assert (pd.DataFrame(gates).groupby('bank').n_answers.sum() == 13769).all()
for panel, expected in [('PRMB',6030),('PB',4442)]:
    assert set(summary[(summary.panel == panel) & summary.complete_population].n_checked) == {expected}
assert len(review['metric_checks']) + len(failures) == 18 * 24
assert all(max(g['replay_errors'].values()) <= 1e-9 for g in gates)

# Per-row provenance includes score hashes, complete eligible counts, exact command,
# and label_sanity flags. Localisation endpoints do not use answer-level AUROC gates.
sp = importlib.util.spec_from_file_location('orth_label_sanity', ROOT / 'spectral_utils/label_sanity.py')
ls = importlib.util.module_from_spec(sp)
sys.modules[sp.name] = ls
sp.loader.exec_module(ls)
pop = infra.CC.Population()
provenance = []
for bk in ['B13','B32']:
    for cell in sorted(set(pop.cells)):
        path = OUT / (bk + '__' + cell + '.scores.npz')
        pack = np.load(path)
        ids = pack['answers']
        is_prm = cell.startswith('prm')
        eligible = [int(i) for i in ids if (pop.eligible[i] if is_prm else pop.target[i] >= 0)]
        if is_prm:
            labels = np.concatenate([pop.labels[pop.off[i]:pop.off[i+1]] for i in eligible])
            sanity = ls.check_labels(labels)
            assert sanity.ok, sanity.summary()
            flag = sanity.flag_string()
        else:
            flag = 'NOT_APPLICABLE_FIRST_ERROR_LOCALIZATION'
        for row in cells[(cells.bank == bk) & (cells.cell == cell)].to_dict('records'):
            assert row['count'] == len(eligible)
            provenance.append(dict(bank=bk,cell=cell,method=row['method'],source_file=str(path),
                                   sha256=infra.sha(path),n_checked=row['count'],n_total=len(eligible),
                                   n_population=len(ids),flag=flag,coverage_flag=ls.feasibility_tag(row['count'],len(eligible)),command=COMMAND))
pd.DataFrame(provenance).to_csv(OUT / 'ROW_PROVENANCE.csv', index=False)

counts = weights.groupby(['bank','representation','within','source']).size().rename('n_groups').reset_index()
counts.to_csv(OUT / 'GUARD_COUNTS.csv', index=False)
contrast = weights[weights.representation == 'ORTH'].groupby(['bank','within']).agg(
    n_groups=('group','size'),mean_common_weight=('common_weight','mean'),
    mean_contrast_weight_share=('contrast_weight_share','mean'),mean_negative_share=('negative_weight_share','mean')).reset_index()
contrast.to_csv(OUT / 'CONTRAST_WEIGHT_SUMMARY.csv', index=False)
estimator_rows = []
for path in sorted(OUT.glob('*.fits.json')):
    bk, cell = path.name.removesuffix('.fits.json').split('__')
    fit = json.loads(path.read_text(encoding='utf8'))
    for record in fit['records']:
        if record['within'] == 'HEM':
            estimates = [(-1,record['details']['estimate'])]
        elif record['within'] == 'DS':
            estimates = [(g['group'],g['estimate']) for g in record['details']['groups'] if 'estimate' in g]
        else:
            continue
        for group, estimate in estimates:
            estimator_rows.append(dict(bank=bk,cell=cell,representation=record['representation'],within=record['within'],
                                       group=group,converged=estimate['converged'],boundary_emissions=estimate['boundary_emissions']))
estimator_frame = pd.DataFrame(estimator_rows)
estimator_frame.to_csv(OUT / 'ESTIMATOR_DIAGNOSTICS.csv',index=False)
nonconverged = int((~estimator_frame.converged).sum())
win_rows = []
for bk in ['B13','B32']:
    for within in ['EQ','SML','MOM','DS','HEM']:
        for between in ['EQ','MOM']:
            new, old = f'ORTH_{within}_{between}', f'RAW_{within}_{between}'
            a = cells[(cells.bank == bk) & (cells.method == new)].set_index('cell')['mean']
            b = cells[(cells.bank == bk) & (cells.method == old)].set_index('cell')['mean']
            delta = a.subtract(b).dropna()
            win_rows.append(dict(bank=bk,within=within,between=between,wins=int((delta>1e-12).sum()),
                                 losses=int((delta < -1e-12).sum()),ties=int((delta.abs()<=1e-12).sum()),n_cells=len(delta)))
wins = pd.DataFrame(win_rows)
wins.to_csv(OUT / 'CELL_WIN_LOSS.csv', index=False)

def value(bk,panel,method):
    s = summary[(summary.bank == bk) & (summary.panel == panel) & (summary.method == method)]
    return float(s.value.iloc[0]) if len(s) else float('nan')

def table(panel,between):
    rows = []
    for method in ['EQ','SML','MOM','DS','HEM']:
        row = {'שיטה בתוך קבוצה':method}
        for bk in ['B13','B32']:
            row[bk+' מקורי'] = value(bk,panel,f'RAW_{method}_{between}')
            row[bk+' אורתוגונלי'] = value(bk,panel,f'ORTH_{method}_{between}')
        rows.append(row)
    return pd.DataFrame(rows).to_html(index=False,float_format=lambda x:f'{x:.6f}',na_rep='לא הושלם',escape=True)

primary = paired[(paired.panel == 'PRMB') & paired.method.isin([f'ORTH_{m}_EQ' for m in ['SML','MOM','DS','HEM']]) & paired.reference.str.startswith('RAW_')]
all_lose = len(primary) == 8 and (primary.hi95 < 0).all()
n_losses = int((primary.hi95 < 0).sum())
n_gains = int((primary.lo95 > 0).sum())
headline = f'בהשוואה הראשית: {n_losses} מתוך 8 השוואות שיטה־בנק יורדות ב־PRMBench עם רווח הפרש שלילי; {n_gains} עולות עם רווח חיובי. הבסיס החדש אינו משחזר את השיפור שנמצא בניסוי המפוקח.'
report = '''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><title>משקול על רכיב משותף וניגודים</title>
<style>body{font-family:Arial,sans-serif;max-width:1160px;margin:36px auto;padding:0 20px;line-height:1.7;color:#192634}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #cad4de;padding:8px;text-align:right}th{background:#edf3f8}a{color:#17639b}</style>
<h1>SML / MoM / HEM / DS על בסיס אורתוגונלי בתוך משפחה</h1>'''
report += '<p><strong>'+headline+'</strong></p>'
report += '''<p>החריג הוא SML ב־B32: עלייה מ־0.731631 ל־0.737647, הפרש 0.006016 ורווח חקרני [0.002004, 0.009977]. זו עלייה מול SML המקורי, אבל התוצאה עדיין נמוכה מממוצע הקבוצות המקורי (0.744377). ב־B13 אותה החלפת ייצוג מורידה את SML מ־0.778598 ל־0.599104.</p>
<p>כאשר מוסיפים MoM בין הקבוצות, SML החדש ב־B32 נותן 0.754459 מול 0.757515 למקורי: הפרש ‎−0.003056 ורווח [‎−0.006914, 0.000759]. מול ממוצע פנימי ו־MoM בין הקבוצות (0.753675), ההפרש 0.000784 ורווח [‎−0.003168, 0.004772]. לכן אין כאן ראיה ליתרון על הביקורות האלה, אף שהוא עולה על הממוצע הפשוט של כל הפיצ׳רים.</p>'''
report += '''<p>5 באוקטובר 2026. ניסוי ללא שימוש בתוויות להתאמת המשקולות, על נתוני פיתוח שכבר נבחנו. שתי האוכלוסיות המלאות: 13,769 תשובות בכל בנק, תשעה תאים. ההתאמה משתמשת בכל הפיצ׳רים בתא (transductive), ואינה התאמה מתשובה יחידה. כל ציוני המועמדים נחתמו לפני חישוב מדדי האיכות בהרצה המלאה.</p>
<p>לאחר הסינון והקיבוץ ההיסטוריים, כל קבוצה עם m פיצ׳רים הוחלפה ב־m קואורדינטות: ממוצע מנורמל ועוד m−1 ניגודי Helmert. סדר שמות הפיצ׳רים קבוע אלפביתית, והסימנים קבועים מראש. אין הוספת עמודות כפולות או בחירת ניגודים לפי תוויות. אחרי ההטלה מבוצע תקנון לכל עמודה בתוך תשובה, ולאחר המיזוג תקנון לכל ציון קבוצה.</p>
<p>לדוגמה: (a+b+c)/√3, (a−b)/√2, (a+b−2c)/√6. הבסיס הפיך לפני התקנון, אבל אורתוגונליות של מקדמי הבסיס אינה אי־תלות סטטיסטית. התקנון מעניק גם לניגודים קטנים שונות יחידה; לא נבדקה כאן בנפרד השפעתו.</p>
<h2>השוואה ראשית: ממוצע שווה בין הקבוצות</h2><p>EQ הוא בקרת משקל שווה בתוך קבוצה; יתר השורות הן ארבע שיטות המשקול המבוקשות. השינוי היחיד בין שתי עמודות של אותו בנק הוא הייצוג והסימון שנגזר ממנו.</p>
<h3>PRMBench: AUC בתוך תשובה, 6,030 תשובות עם שתי מחלקות</h3>'''+table('PRMB','EQ')
report += '<h3>ProcessBench: דיוק איתור השגיאה הראשונה, 4,442 תשובות שגויות; ממוצע שווה של שמונה תאים</h3>'+table('PB','EQ')
report += '<h2>השוואה משנית: MoM בין הקבוצות</h2><h3>PRMBench</h3>'+table('PRMB','MOM')+'<h3>ProcessBench</h3>'+table('PB','MOM')
report += '<h2>הפרשי ההשוואה הראשית ורווחים חקרניים של 95%</h2>'+primary.to_html(index=False,float_format=lambda x:f'{x:.6f}')
report += '''<p>5,000 דגימות bootstrap מזווגות לפי שאלת מקור, עם דגימה משותפת של תאי PB. הרווחים מותנים בציונים שכבר הותאמו; אין התאמה חוזרת בכל דגימה, תיקון לריבוי השוואות או אישור על אוכלוסייה שלא נחשפה.</p>
<h2>מה המימוש הזה בודק</h2>
<p>SML מקבל ערכים רציפים כמו במימוש הקיים. MoM, DS ו־HEM מקבלים סימוני TAIL של 20% לכל קואורדינטה חדשה. HEM מותאם במשותף לכל הקבוצות הקבועות, ומשתמשים במשקולות הפליטה שלו בתוך קבוצה. MoM ו־DS מותאמים לכל קבוצה. כללי הסימן המקוריים נשמרו; MoM/DS/HEM משתמשים במשקולות לא שליליות עם קיטום, ו־SML יכול להפיק משקולות שליליות.</p>
<p>ניגוד אינו גלאי שגיאות מכוון מראש: סימן חיובי שלו יכול רק לומר שפיצ׳ר אחד גבוה מאחר. לכן החלפת הייצוג אינה מבטיחה שהנחות ההסכמה והסימן של השיטות המקוריות יתאימו לקואורדינטות החדשות. זו פרשנות אפשרית לתוצאה, לא מנגנון סיבתי שהוכח בניסוי.</p>
<p>בקבוצות קטנות נשמרו ההגנות המקוריות: MoM/DS/HEM משתמשים במשקל שווה כאשר יש פחות משלושה חברים; SML מפעיל הגנה בגודל שלוש. משקל שווה בקואורדינטות החדשות משלב גם ניגודים ואינו שקול לממוצע הפיצ׳רים המקוריים. הספירות והמשקולות נגישות להלן. תוצאה שלילית סוגרת את המתכון הקבוע הזה בלבד, ולא כל בסיס אפשרי או כל דרך להשתמש בניגודים.</p>
<h2>בקרות ואמינות</h2>'''
report += f'<p>ארבע בדיקות יחידה עברו. הושלמו 18 תאים; {review["n_metric_rows"]} שורות מדדים חושבו גם במימוש נפרד של ספירת זוגות. נרשמו {len(failures)} כישלונות אמידה צפויים. שוחזרו בקרות RAW EQ/HEM והרכיב המשותף בלבד, ונבדקו גיבובי המקורות לפני ואחרי. הבדיקה בוצעה בידי אותו סוכן, ללא סוקר חיצוני.</p>'
report += f'<p>מתוך {len(estimator_frame)} התאמות DS/HEM בתוך הקבוצות, {nonconverged} סומנו ללא התכנסות. נשמר כלל המימוש המקורי להפקת ציונים; פרטי ההתכנסות והפליטות בגבול נרשמו ב־ESTIMATOR_DIAGNOSTICS.csv. ב־HEM מדובר בהתאמה משותפת אחת לתא ולייצוג; ב־DS מדובר בהתאמה נפרדת לכל קבוצה שאינה קטנה מדי.</p>'
report += '<p>רכיב משותף בלבד משחזר את ממוצע הקבוצה הקודם. LR המפוקח מהמבחן הקודם מופיע ב־SUMMARY להשוואה בלבד; הוא לא שימש ליצירת הבסיס, לסימנים או למשקולות.</p>'
report += '<h3>מסה ממוצעת של משקולות הניגודים, ללא קשר לתוויות</h3>'+contrast.to_html(index=False,float_format=lambda x:f'{x:.4f}')
report += '<h3>ספירת כללי אמידה והגנות</h3>'+counts.to_html(index=False)
links = ['SUMMARY.csv','PER_CELL.csv','PAIRED_CONTRASTS.csv','CELL_WIN_LOSS.csv','WITHIN_WEIGHTS.csv','GUARD_COUNTS.csv','ESTIMATOR_DIAGNOSTICS.csv','FAILURES.json','ROW_PROVENANCE.csv','GATES.json','REVIEW.json','SCORES_SEALED_BEFORE_EVALUATION.json','../PROTOCOL.json']
report += '<p>'+' · '.join(f'<a href="{p}">{html.escape(p)}</a>' for p in links)+'</p></html>'
(OUT / 'REPORT_HE.html').write_text(report,encoding='utf8')
for p in re.findall(r'href="([^"]+)"',report):
    assert (OUT / p).resolve().exists(),p

if n_losses:
    lines = ['# Negative result — fixed orthogonal-family weighting', '',
             '**Hypothesis (one sentence):** Replacing each frozen family by its common direction and a complete orthogonal contrast basis improves existing label-free within-family weighting.',
             '**Date / step:** 2026-10-05, Step 449 [Codex]',
             '**Benchmark / population:** B13/B32, all nine cells, 13,769 answers per bank; PRMB v3 within-answer AUC on 6,030 eligible answers, PB first-error argmax accuracy on 4,442 erroneous answers, macro8. Candidate fit is label-free and transductive per cell; source folds remain unchanged but are not used for label-free fitting.',
             '', '## Arms compared',
             'Primary comparison holds between-group fusion at EQ. Every value below comes from SUMMARY.csv; PB is argmax accuracy, not F1.', '',
             '| Bank | Arm | PRMB within-AUC | PB macro argmax |', '|---|---|---:|---:|']
    for bk in ['B13','B32']:
        for method in ['EQ','SML','MOM','DS','HEM']:
            for rep in ['RAW','ORTH']:
                arm = f'{rep}_{method}_EQ'
                lines.append(f'| {bk} | {arm} | {value(bk,"PRMB",arm):.6f} | {value(bk,"PB",arm):.6f} |')
    lines += ['', '**Win/loss record:** Point-estimate cell tallies from CELL_WIN_LOSS.csv; paired PRMB differences and 95% intervals from PAIRED_CONTRASTS.csv (5,000 source-question draws; exploratory, unadjusted).', '',
              '| Bank | Within method | Wins–losses–ties / cells | PRMB difference [95% interval] | Contains zero |', '|---|---|---|---|---|']
    for r in primary.to_dict('records'):
        within = r['method'].split('_')[1]
        w = wins[(wins.bank == r['bank']) & (wins.within == within) & (wins.between == 'EQ')].iloc[0]
        contains_zero = 'Yes' if r['lo95'] <= 0 <= r['hi95'] else 'No'
        lines.append(f'| {r["bank"]} | {within} | {w.wins}–{w.losses}–{w.ties} / {w.n_cells} | {r["delta"]:+.6f} [{r["lo95"]:+.6f}, {r["hi95"]:+.6f}] | {contains_zero} |')
    lines += ['', 'There is one primary PRMB improvement: SML in B32. It remains below COMMON_EQ. With MoM between groups, the SML B32 difference against RAW_SML_MOM includes zero, as does its difference against COMMON_MOM. This is a mostly negative, mixed result, not a claim that every comparison loses.']
    lines += ['', '## Three plausible reasons it failed',
              'These are explanations consistent with the recipe, not separately established causal findings.',
              '1. Fixed contrast signs express relative feature values rather than known error direction. Existing latent orientation conventions do not resolve this semantic difference.',
              '2. Orthogonal coefficient vectors do not imply conditionally independent votes or one common latent error factor; pooling contrasts as ordinary detectors can violate the estimators\' assumptions.',
              '3. The existing nonnegative clipping in MoM/DS/HEM, equal-weight guards in small families, and per-answer rescaling of small contrasts can reduce the common component\'s influence. Actual weights and guard counts are in WITHIN_WEIGHTS.csv, CONTRAST_WEIGHT_SUMMARY.csv and GUARD_COUNTS.csv; their individual causal effects were not isolated.',
              '', '## Verdict',
              '- [ ] closes the DIRECTION (the idea cannot work under this access/contract)',
              '- [x] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open',
              'The fixed alphabetical Helmert recipe does not provide a robust recovery of the supervised gap: seven primary PRMB comparisons lose, one gains against a weaker same-estimator control. It does not test all possible contrast bases or sign/weight constraints. Passing replay and algebraic gates establishes implementation fidelity, not correctness of the modelling assumptions.',
              '', '## What would reopen it',
              'A separately registered label-free rule that retains the common score and assigns signed, constrained corrections to contrasts, followed by a matched evaluation; no additional search was run here.',
              '', '**Source files:** REPORT_HE.html, SUMMARY.csv, PAIRED_CONTRASTS.csv, CELL_WIN_LOSS.csv, ROW_PROVENANCE.csv, GATES.json, REVIEW.json. Full coverage: 6,030/6,030 PRMB and 4,442/4,442 erroneous PB per complete arm. Exact command: `' + COMMAND + '`.']
    (OUT / 'NEGATIVE_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

snapshot = OUT / 'implementation'
snapshot.mkdir(exist_ok=True)
for rel in ['spectral_utils/orthogonal_family_fusion.py','spectral_utils/contrast_span_oracle.py','spectral_utils/group_contrast_oracle.py','spectral_utils/label_sanity.py',
            'scripts/run_orthogonal_family_fusion.py','scripts/report_orthogonal_family_fusion.py','scripts/run_contrast_span_oracle.py','scripts/run_group_contrast_oracle.py',
            'tests/test_orthogonal_family_fusion.py']:
    target = snapshot / Path(rel).name
    if target.exists():
        assert target.read_bytes() == (ROOT / rel).read_bytes()
    else:
        shutil.copyfile(ROOT / rel,target)
cvf_root = ROOT / '.worktrees/cumulative-vote-fusion-v2/scripts/experiments/cvf_v2'
for source in sorted(cvf_root.rglob('*.py')):
    assert infra.sha(source) == seal['code'][str(source)]
    target = snapshot / 'cvf_v2' / source.relative_to(cvf_root)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert target.read_bytes() == source.read_bytes()
    else:
        shutil.copyfile(source,target)
infra.dump(OUT / 'RUN_CONTEXT.json',dict(command=COMMAND,scope='LABEL_FREE_TRANSDUCTIVE_DEVELOPMENT',
                                       population='13769 per bank; 6030 PRMB eligible; 4442 erroneous PB',
                                       label_contract='PRMB v3; unchanged official labels and source folds',
                                       review='Same agent, independent metric implementation; no external reviewer',
                                       execution='CPU only, 2 numerical threads; no inference/GPU/new data',
                                       failed_smoke='Initial empty smoke stopped on manifest string metadata; corrected explicit section selection; smoke_v2 passed'))
heading = '## 2026-10-05 - Fixed orthogonal-family label-free weighting complete (Codex)'
method_lines = []
for m in ['SML','MOM','HEM','DS']:
    method_lines.append(m + ': ' + '; '.join(f'{bk} {value(bk,"PRMB",f"RAW_{m}_EQ"):.6f} -> {value(bk,"PRMB",f"ORTH_{m}_EQ"):.6f}' for bk in ['B13','B32']))
finding = f'{n_losses}/8 primary PRMB comparisons have negative paired 95% intervals; {n_gains}/8 positive. SML B32 gains .006016 [.002004,.009977] over RAW SML but remains below COMMON_EQ. With MoM between groups, its differences against RAW SML+MoM and COMMON_MOM both include zero. No general recovery of the supervised gap.'
note = '\n\n' + heading + '\n\nOmri requested SML/MoM/HEM/DS on a common-plus-orthogonal-contrast representation inside frozen families. Completed all 18 B13/B32 cells (13769 answers per bank), with 6030 PRMB eligible and 4442 erroneous PB answers. Report: results/orthogonal_family_fusion_20261005/run_v1/REPORT_HE.html. Candidate fitting is label-free, transductive per cell. Fixed alphabetical Helmert coordinates replace original columns, followed by within-answer standardization and native weighting; no new filtering, grouping or sign search. RAW matched controls and common-only replay retained. EQ between groups is primary, MoM secondary.\n\nPRMB primary RAW -> ORTH: ' + ' | '.join(method_lines) + '. ' + finding + '\n\nFour tests, all common-only/raw EQ/raw HEM replay controls, ' + str(review['n_metric_rows']) + ' independent metric rows and unchanged input/code hashes PASS. All candidate scores sealed before full-run quality evaluation. Native small-group equal guards and clipping documented; expected failures: ' + str(len(failures)) + '. Orthogonality does not imply independent error votes. This result tests one fixed representation/normalization/weighting recipe, not the whole contrast direction. No supervised-label use to orient or weight contrasts, no GPU/inference, no historical score overwrite. Initial manifest-type smoke error preserved and corrected without changing a numerical gate.\n'
progress = ROOT / 'PROGRESS.md'
if heading not in progress.read_text(encoding='utf8'):
    with progress.open('a',encoding='utf8') as handle:
        handle.write(note)
history_heading = '### Step 449 [Codex orthogonal-family weighting] - Fixed common-plus-contrast coordinates with native label-free estimators, 2026-10-05'
history = ROOT / 'HISTORY.md'
if history_heading not in history.read_text(encoding='utf8'):
    assert '### Step 449 ' not in history.read_text(encoding='utf8'), 'History step already allocated'
    with history.open('a',encoding='utf8') as handle:
        handle.write('\n\n'+history_heading+'\n\n**What**: User-requested SML/MoM/HEM/DS within fixed Helmert family coordinates, matched RAW controls, EQ/MoM between groups, full development population.\n**Why**: Test whether an invertible common-plus-contrast basis lets existing label-free estimators recover the supervised contrast signal.\n**Result**: '+finding+' '+str(review['n_metric_rows'])+' independent metric rows and all replay/hash gates PASS. Complete report: results/orthogonal_family_fusion_20261005/run_v1/REPORT_HE.html. Fixed representation failure is not a general impossibility result.\n')
        if n_losses:
            handle.write('Negative result recorded: results/orthogonal_family_fusion_20261005/run_v1/NEGATIVE_RESULT.md\n')
infra.dump(BASE / 'LATEST.json',dict(status='COMPLETE_REVIEWED',run='run_v1',report='run_v1/REPORT_HE.html'))
infra.dump(OUT / 'DELIVERY_MANIFEST.json',{str(p.relative_to(OUT)):infra.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='DELIVERY_MANIFEST.json'})
print('REPORT PASS',headline,flush=True)
print(primary.to_string(index=False),flush=True)
print(contrast.to_string(index=False),flush=True)
print(wins.to_string(index=False),flush=True)
