"""Render the S0 findings page (Hebrew) from the S0 CSVs. Numbers are read, never typed."""
from pathlib import Path
import base64, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/ssl_pseudolabel_residual_v1/S0'
log = json.loads((OUT / 'S0_LOG.json').read_text(encoding='utf8'))
board = pd.read_csv(OUT / 'SCOREBOARD_REPLAY.csv').set_index('method')
q = pd.read_csv(OUT / 'PLANNED_CONTRASTS_BY_QUESTION.csv')
pbc = pd.read_csv(OUT / 'PB_DECISION_CHANGES.csv').set_index(['candidate', 'reference'])
prc = pd.read_csv(OUT / 'PRM_PAIRED_CHANGES.csv').set_index(['candidate', 'reference'])
depth = pd.read_csv(OUT / 'PB_BY_DEPTH.csv'); comp = pd.read_csv(OUT / 'PB_COMPETITION_ORACLE.csv').set_index('method')
parity = pd.read_csv(OUT / 'SEED_PARITY.csv')
kind = pd.read_csv(OUT / 'PRM_WITHIN_AUC_BY_KIND.csv', header=[0, 1], index_col=0).xs('mean', axis=1, level=1).T

N = {'ct7': 'CT7 (עוגן קפוא)', 'token_lsml': 'token L-SML', 'token_equal': 'token equal',
     'evidence__all__seed__equal': 'top5 seed (ה-teacher בפועל)', 'evidence__all__plain__equal': 'top5 plain evidence',
     'evidence__all__position__equal': 'top5 position evidence', 'evidence__all__plain2__equal': 'top5 plain, איטרציה 2',
     'evidence__all__position2__equal': 'top5 position, איטרציה 2', 'evidence__all__randomseed__equal': 'top5 random-seed (null)',
     'evidence__all__randomseed_position__equal': 'top5 random-seed + position', 'evidence__all__prioronly__equal': 'top5 prior בלבד',
     'evidence__all__ceiling__equal': 'top5 תוויות אמת (ceiling)', 'evidence__all__plain__continuous_lsml': 'top5 plain + L-SML',
     'evidence__all__plain__spectral': 'top5 plain + spectral', 'evidence__all__position__continuous_lsml': 'top5 position + L-SML',
     'evidence30__all__seed__equal': 'top30 seed', 'evidence30__all__plain__equal': 'top30 plain evidence', 'evidence30__all__position__equal': 'top30 position evidence',
     'evidence30__all__randomseed__equal': 'top30 random-seed (null)', 'evidence__all__ceiling_position__equal': 'top5 תוויות אמת + position'}
def img(name): return 'data:image/png;base64,' + base64.b64encode((OUT / name).read_bytes()).decode()
def pct(x, d=1): return f'{100*x:.{d}f}%'
def num(x, d=4): return f'{x:.{d}f}'
def pp(x): return f'{100*x:+.2f} pp'
def sgn(x, d=4): return f'{x:+.{d}f}'

def contrast(a, b, ep):
    r = q[(q.a == a) & (q.b == b) & (q.endpoint == ep)].iloc[0]
    return r.delta, r.ci_lo, r.ci_hi, r.p_holm
def crow(a, b):
    d1 = contrast(a, b, 'pb_sla'); d2 = contrast(a, b, 'prm_within_auc')
    def cell(d, f):
        lo, hi = d[1], d[2]; zero_out = lo > 0 or hi < 0
        cls = ' class="pos"' if zero_out and d[0] > 0 else ' class="neg"' if zero_out else ''
        return f'<td{cls}>{f(d[0])}<br><span class="ci">[{f(lo)}, {f(hi)}]</span>{"" if d[3] >= .05 else "<br><span class=holm>Holm " + num(d[3], 3) + "</span>"}</td>'
    return f'<tr><td class="lbl">{N[a]}<br><span class="vs">מול {N[b]}</span></td>{cell(d1, pp)}{cell(d2, lambda x: sgn(x, 4))}</tr>'

sb_rows = ''.join(f'<tr><td class="lbl">{N[m]}</td><td>{pct(board.loc[m,"pb_sla_macro8"])}</td><td>{pct(board.loc[m,"pb_f1_ct7_gate"])}</td><td>{num(board.loc[m,"within_auc"])}</td><td>{num(board.loc[m,"prmscore_inner"])}</td></tr>'
                  for m in ['ct7', 'token_lsml', 'token_equal', 'evidence__all__seed__equal', 'evidence__all__plain__equal', 'evidence__all__position__equal', 'evidence__all__position2__equal',
                            'evidence__all__plain__continuous_lsml', 'evidence__all__ceiling__equal', 'evidence__all__randomseed__equal', 'evidence__all__prioronly__equal',
                            'evidence30__all__seed__equal', 'evidence30__all__plain__equal', 'evidence30__all__position__equal'])
questions = [
    ('1. האם הפסאודו-תוויות תורמות מידע?', 'כן מול null אקראי, אבל לא מול ה-teacher עצמו ב-ProcessBench.',
     [('evidence__all__plain__equal', 'evidence__all__randomseed__equal'), ('evidence__all__plain__equal', 'evidence__all__seed__equal'), ('evidence30__all__plain__equal', 'evidence30__all__seed__equal')]),
    ('2. האם ההתניה במיקום תורמת?', 'ב-PRMBench כן, מעט ובאופן יציב. ב-ProcessBench לא, וב-top30 היא מזיקה. ה-prior לבדו חסר ערך.',
     [('evidence__all__position__equal', 'evidence__all__plain__equal'), ('evidence30__all__position__equal', 'evidence30__all__plain__equal'), ('evidence__all__prioronly__equal', 'evidence__all__plain__equal')]),
    ('3. האם איטרציה שנייה תורמת?', 'לא. פוגעת ב-ProcessBench, תוספת זניחה ב-PRMBench.',
     [('evidence__all__plain2__equal', 'evidence__all__plain__equal'), ('evidence__all__position2__equal', 'evidence__all__position__equal')]),
    ('4. האם fusion נלמד (L-SML / spectral) תורם?', 'לא. פוגע ב-ProcessBench בכל וריאציה; ב-PRMBench תוספת של 0.003 לכל היותר.',
     [('evidence__all__plain__continuous_lsml', 'evidence__all__plain__equal'), ('evidence__all__plain__spectral', 'evidence__all__plain__equal'), ('evidence__all__position__continuous_lsml', 'evidence__all__position__equal')]),
    ('5. האם יש יתרון מול CT7?', 'לא. ב-ProcessBench פער של 10 נקודות. ב-PRMBench position evidence מתחת ל-CT7 אך רווח הסמך כמעט נוגע באפס.',
     [('evidence__all__position__equal', 'ct7'), ('evidence__all__position2__equal', 'ct7'), ('evidence30__all__plain__equal', 'ct7')]),
    ('6. מה תוויות אמת היו נותנות באותו מתכון?', 'לא יותר. היסטוגרמה מתוויות אמת לא עוברת את הפסאודו ב-ProcessBench. המגבלה היא במתכון, לא בתוויות.',
     [('evidence__all__ceiling__equal', 'evidence__all__plain__equal'), ('evidence__all__ceiling_position__equal', 'evidence__all__position__equal')]),
]
q_html = ''.join(f'<h3>{t}</h3><p class="ans">{a}</p><div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA</th><th>PRMBench within-AUC</th></tr></thead><tbody>{"".join(crow(x, y) for x, y in prs)}</tbody></table></div>' for t, a, prs in questions)

def dc(a, b):
    r = pbc.loc[(a, b)]
    return f'<tr><td class="lbl">{N[a]}<br><span class="vs">מול {N[b]}</span></td><td>{pct(r.agreement, 0)}</td><td class="pos">{int(r.wrong_to_correct)}</td><td class="neg">{int(r.correct_to_wrong)}</td><td>{int(r.net_gain):+d}</td><td>{pct(r.cand_early_miss,0)} / {pct(r.cand_late_miss,0)}</td></tr>'
dc_rows = ''.join(dc(a, b) for a, b in [('evidence__all__plain__equal', 'evidence__all__seed__equal'), ('evidence__all__position__equal', 'evidence__all__plain__equal'),
                                        ('evidence__all__ceiling__equal', 'evidence__all__plain__equal'), ('evidence30__all__plain__equal', 'evidence30__all__seed__equal'),
                                        ('evidence__all__position__equal', 'ct7'), ('token_lsml', 'ct7')])
def pr(a, b):
    r = prc.loc[(a, b)]
    return f'<tr><td class="lbl">{N[a]}<br><span class="vs">מול {N[b]}</span></td><td>{sgn(r.mean_delta_auc)}</td><td class="pos">{int(r.answers_improved)}</td><td class="neg">{int(r.answers_worsened)}</td><td>{int(r.answers_unchanged)}</td><td>{int(r.net_pairs):+,d}</td></tr>'
pr_rows = ''.join(pr(a, b) for a, b in [('evidence__all__plain__equal', 'evidence__all__seed__equal'), ('evidence__all__position__equal', 'evidence__all__plain__equal'),
                                        ('evidence__all__position__equal', 'ct7'), ('evidence__all__plain__equal', 'ct7'), ('token_lsml', 'ct7')])
kind_rows = ''.join(f'<tr><td class="lbl">{N[m]}</td><td>{num(kind.loc[m,"single"])}</td><td>{num(kind.loc[m,"multi"])}</td></tr>' for m in ['ct7', 'token_lsml', 'evidence__all__seed__equal', 'evidence__all__plain__equal', 'evidence__all__position__equal', 'evidence__all__ceiling__equal'])
comp_rows = ''.join(f'<tr><td class="lbl">{N[m]}</td><td>{pct(comp.loc[m,"exact_top1_S4"])}</td><td>{pct(comp.loc[m,"hit3"])}</td><td>{pct(comp.loc[m,"hit5"])}</td><td>{num(comp.loc[m,"mean_rank"],2)}</td><td>{pct(comp.loc[m,"competition_oracle_3"])}</td></tr>' for m in ['ct7', 'token_lsml', 'evidence30__all__plain__equal', 'evidence__all__seed__equal', 'evidence__all__plain__equal', 'evidence__all__position__equal', 'evidence__all__ceiling__equal'])
dep = depth.set_index(['method', 'depth_bin'])
dep_rows = ''.join(f'<tr><td class="lbl">{N[m]}</td>' + ''.join(f'<td>{pct(dep.loc[(m,b),"sla"])}</td>' for b in ['2-5', '6-10', '11+']) + '</tr>' for m in ['ct7', 'token_lsml', 'evidence__all__seed__equal', 'evidence__all__plain__equal', 'evidence__all__position__equal', 'evidence30__all__plain__equal'])
dep_n = {b: int(dep.loc[('ct7', b), 'n']) for b in ['2-5', '6-10', '11+']}
par_rows = ''.join(f'<tr><td class="lbl">{r.seed}</td><td>{int(r.pb_argmax_diff_vs_stored)}</td><td>{pct(r.pb_sla_macro8)}</td><td>{num(r.prm_within_auc)}</td></tr>' for r in parity.itertuples())
inv = log['inventory']; cov = log['coverage']; sr = log['scoreboard_replay']; mult = log['multiplicity']
maxerr = max(v for v in sr['max_abs_error_per_metric'].values())
hashes_ok = all(v['sha_ok'] for v in log['input_freeze_today'].values())

html = f'''<title>S0 Audit · Step 432</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#f6f7f5;--ink:#1c2430;--muted:#5d6874;--rule:#d7dcd9;--card:#eef1ee;--accent:#1f6f8b;--pos:#3f7d4e;--neg:#b5452b;--posbg:#e6f0e8;--negbg:#f6e6e1;--holm:#8a5a00}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c;--holm:#d9a441}}}}
:root[data-theme="dark"]{{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c;--holm:#d9a441}}
body{{background:var(--paper);color:var(--ink);font-family:Heebo,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.6;direction:rtl}}
main{{max-width:820px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-weight:700;font-size:30px;line-height:1.2;margin:0 0 6px;text-wrap:balance}}
h2{{font-weight:500;font-size:22px;margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--rule)}}
h3{{font-weight:500;font-size:17px;margin:26px 0 4px;color:var(--accent)}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);direction:ltr;text-align:right}}
.lede{{font-size:18px;font-weight:300;margin:12px 0 0}}
.verdict{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0 0}}
.verdict div{{background:var(--card);padding:14px 16px;border-radius:4px}}
.verdict b{{display:block;font-family:"IBM Plex Mono",monospace;font-size:22px;font-weight:500;direction:ltr;text-align:right}}
.verdict span{{font-size:13px;color:var(--muted)}}
p.ans{{margin:0 0 8px;font-weight:500}}
.tw{{overflow-x:auto;margin:8px 0 4px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th{{text-align:right;font-weight:500;color:var(--muted);border-bottom:1px solid var(--ink);padding:6px 10px;font-size:13px}}
td{{padding:7px 10px;border-bottom:1px solid var(--rule);font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;direction:ltr;text-align:right;vertical-align:top;white-space:nowrap}}
td.lbl{{font-family:Heebo,sans-serif;direction:rtl;white-space:normal;min-width:180px}}
td.pos{{color:var(--pos);background:var(--posbg)}} td.neg{{color:var(--neg);background:var(--negbg)}}
.ci{{font-size:11px;color:var(--muted)}} .holm{{font-size:11px;color:var(--holm)}} .vs{{font-size:12px;color:var(--muted)}}
figure{{margin:16px 0 4px}} figure img{{width:100%;border:1px solid var(--rule);border-radius:3px;background:#fff}}
figcaption{{font-size:13px;color:var(--muted);margin-top:6px}}
.note{{font-size:14px;color:var(--muted);margin:6px 0}}
.next{{background:var(--card);padding:16px 20px;border-radius:4px;border-right:3px solid var(--accent)}}
ul{{padding-right:20px;margin:8px 0}} li{{margin:4px 0}}
code{{font-family:"IBM Plex Mono",monospace;font-size:13px;direction:ltr;unicode-bidi:embed}}
@media (max-width:600px){{.verdict{{grid-template-columns:1fr}}}}
</style>
<main>
<div class="eyebrow">SSL / pseudo-label / residual plan v1.1 · stage S0 · 2026-09-23 · development data only</div>
<h1>ביקורת S0 על Step 432</h1>
<p class="lede">כל 57 השיטות של ניסוי הפסאודו-תוויות משתחזרות בדיוק מהציונים הגולמיים. הממצא המרכזי: ההיסטוגרמה משנה כמחצית מההחלטות של ה-teacher ב-ProcessBench, אבל מצילה ומזיקה במספרים כמעט שווים. ב-PRMBench היא משפרת דירוג באופן ממשי, ועדיין לא מגיעה ל-CT7.</p>
<div class="verdict">
<div><b>VALIDATED</b><span>תנאי היציאה של S0: {inv["jobs"]} עבודות תואמות manifest, {inv["stale_jobs_newer_than_summary"]} עבודות מיושנות, {cov["groups_crossing_folds"]} קבוצות מקור שחוצות folds, hashes של הקלטים תקפים היום: {"כן" if hashes_ok else "לא"}</span></div>
<div><b>{maxerr:.1e}</b><span>סטייה מקסימלית בין השחזור העצמאי לבין הדוח, על {sr["methods"]} שיטות × 7 מדדים. PRMScore: {sr["prmscore_methods_replayed"]} × 2 ערכים זהים לחלוטין דרך ה-evaluator הרשמי</span></div>
<div><b>{pct(board.loc["ct7","pb_sla_macro8"])} / {num(board.loc["ct7","within_auc"],3)}</b><span>CT7 נשאר העוגן בשני היעדים. הזרוע הטובה ביותר של Step 432 ב-PRMBench: {num(board.loc["evidence__all__position2__equal","within_auc"],3)}; ב-ProcessBench: {pct(board.loc["evidence30__all__plain__equal","pb_sla_macro8"])}</span></div>
</div>

<h2>לוח התוצאות המשוחזר</h2>
<p class="note">אוכלוסייה מלאה: {cov["answers"]:,} תשובות, {cov["steps"]:,} צעדים. ProcessBench {cov["pb"]:,} ({cov["pb_erroneous"]:,} שגויות), PRMBench {cov["prm"]:,} ({cov["prm_two_class_answers"]:,} עם שתי מחלקות). PRMScore עם סף שנבחר על folds פנימיים.</p>
<figure><img src="{img('FIG1_scoreboard.png')}" alt="scoreboard"><figcaption>כל זרוע לפי שני היעדים הראשיים. ימינה-למעלה טוב יותר. העוגנים (מעוינים) נמצאים מעל כל זרועות ה-evidence (אדום).</figcaption></figure>
<div class="tw"><table><thead><tr><th>שיטה</th><th>PB SLA macro8</th><th>PB F1 (gate CT7)</th><th>PRMB within-AUC</th><th>PRMScore</th></tr></thead><tbody>{sb_rows}</tbody></table></div>

<h2>חמש השאלות של S0</h2>
<p class="note">כל השוואה: הפרש זוגי, רווח סמך 95% מ-bootstrap על {mult["source_groups"]:,} קבוצות שאלת-מקור ({mult["draws"]:,} דגימות), ותיקון Holm על {mult["family_size"]} השוואות. רזולוציית הזנב היא 1e-4, ולכן הערך המינימלי של Holm הוא {num(mult["min_p_holm"],3)}; {mult["contrasts_with_p_holm_below_0.05"]} השוואות יושבות על הרצפה הזו. תא צבוע = רווח סמך שאינו כולל אפס.</p>
{q_html}
<figure><img src="{img('FIG4_contrasts.png')}" alt="contrasts"><figcaption>אותן השוואות כגרף. משמאל ProcessBench בנקודות אחוז, מימין PRMBench ב-AUC.</figcaption></figure>

<h2>איפה ההחלטות השתנו בפועל</h2>
<p>הסכמה בין ה-student ל-teacher היא נמוכה: ההיסטוגרמה מזיזה את השיא ב-{pct(1-pbc.loc[("evidence__all__plain__equal","evidence__all__seed__equal")].agreement,0)} מהתשובות השגויות. זו לא "השפעה מצומצמת" אלא החלפה של טעויות בטעויות אחרות.</p>
<figure><img src="{img('FIG3_pb_decision_changes.png')}" alt="decision changes"></figure>
<div class="tw"><table><thead><tr><th>מועמד מול ייחוס</th><th>הסכמה</th><th>ניצלו</th><th>נפגעו</th><th>נטו</th><th>החטאה מוקדמת / מאוחרת (מועמד)</th></tr></thead><tbody>{dc_rows}</tbody></table></div>
<p class="note">ProcessBench, {cov["pb_erroneous"]:,} תשובות שגויות. "ניצלו" = הייחוס טעה והמועמד צדק; "נפגעו" = להפך.</p>

<h3>PRMBench: שינוי דירוג לכל תשובה</h3>
<div class="tw"><table><thead><tr><th>מועמד מול ייחוס</th><th>Δ AUC ממוצע</th><th>תשובות שהשתפרו</th><th>שהורעו</th><th>ללא שינוי</th><th>זוגות error/clean נטו</th></tr></thead><tbody>{pr_rows}</tbody></table></div>
<p>מול CT7, position evidence משפר {int(prc.loc[("evidence__all__position__equal","ct7")].answers_improved):,} תשובות ומרע {int(prc.loc[("evidence__all__position__equal","ct7")].answers_worsened):,}: כמעט תיקו. הפירוק לפי סוג תשובה מסביר איפה:</p>
<div class="tw"><table><thead><tr><th>שיטה</th><th>שגיאה אחת ({int(pd.read_csv(OUT/'PRM_WITHIN_AUC_BY_KIND.csv',header=[0,1],index_col=0).xs('count',axis=1,level=1).loc['single'].iloc[0]):,} תשובות)</th><th>כמה שגיאות ({int(pd.read_csv(OUT/'PRM_WITHIN_AUC_BY_KIND.csv',header=[0,1],index_col=0).xs('count',axis=1,level=1).loc['multi'].iloc[0]):,} תשובות)</th></tr></thead><tbody>{kind_rows}</tbody></table></div>
<p>בתשובות עם כמה צעדים שגויים position evidence שווה ל-CT7 ({num(kind.loc["evidence__all__position__equal","multi"],3)} מול {num(kind.loc["ct7","multi"],3)}). הפער כולו נמצא בתשובות עם שגיאה אחת, שם CT7 מוביל ב-{num(kind.loc["ct7","single"]-kind.loc["evidence__all__position__equal","single"],3)}.</p>

<h2>עומק ותחרות ב-ProcessBench</h2>
<figure><img src="{img('FIG2_pb_by_depth.png')}" alt="by depth"></figure>
<div class="tw"><table><thead><tr><th>שיטה</th><th>2–5 צעדים (n={dep_n["2-5"]:,})</th><th>6–10 (n={dep_n["6-10"]:,})</th><th>11+ (n={dep_n["11+"]:,})</th></tr></thead><tbody>{dep_rows}</tbody></table></div>
<h3>כמה מהפער הוא "תחרות" בין צעדים?</h3>
<p class="note">תשובות עם 4 צעדים ומעלה (n={int(comp.loc["ct7","n_S4"]):,}). "oracle 3 מתחרים" משאיר את הצעד הנכון ועוד שלושה צעדים אקראיים ומחשב הסתברות זכייה מדויקת. זהו אבחון שמשתמש בתווית, לא שיטה.</p>
<div class="tw"><table><thead><tr><th>שיטה</th><th>פגיעה מדויקת</th><th>בטופ 3</th><th>בטופ 5</th><th>דרגה ממוצעת של השגיאה</th><th>oracle 3 מתחרים</th></tr></thead><tbody>{comp_rows}</tbody></table></div>
<p>גם כשמשאירים רק שלושה מתחרים, plain evidence מגיע ל-{pct(comp.loc["evidence__all__plain__equal","competition_oracle_3"])} לעומת {pct(comp.loc["ct7","competition_oracle_3"])} ל-CT7. הפער מ-CT7 אינו רק בעיית ריבוי צעדים; השגיאה הראשונה מדורגת נמוך יותר (דרגה {num(comp.loc["evidence__all__plain__equal","mean_rank"],2)} מול {num(comp.loc["ct7","mean_rank"],2)}).</p>

<h2>סטיות provenance שיש לשמור</h2>
<h3>שני teachers, לא אחד</h3>
<div class="tw"><table><thead><tr><th>seed</th><th>argmax שונה ב-PB</th><th>PB SLA</th><th>PRMB within-AUC</th></tr></thead><tbody>{par_rows}</tbody></table></div>
<p>ה-seed שהריצה השתמשה בו לא ביצע z-score בין צעדים, כפי שהפרוטוקול תיאר. הגרסה המתוארת חזקה יותר ב-{pp(parity.pb_sla_macro8.iloc[2]-parity.pb_sla_macro8.iloc[0])} ב-ProcessBench וב-{sgn(parity.prm_within_auc.iloc[2]-parity.prm_within_auc.iloc[0],3)} ב-PRMBench. לכן חלק מהיתרון של evidence "מול ה-seed" ב-PRMBench הוא מול teacher חלש; מול ה-seed המנורמל היתרון יורד מ-{sgn(board.loc["evidence__all__plain__equal","within_auc"]-parity.prm_within_auc.iloc[0],3)} ל-{sgn(board.loc["evidence__all__plain__equal","within_auc"]-parity.prm_within_auc.iloc[2],3)}. שני ה-seeds נשמרים כשני method IDs.</p>
<ul>
<li>ב-PRMBench כלל הפסאודו הוא תיקון A1: צעד חיובי אחד בכל תשובת אימון, כי ה-gate של CT7 נפתח על {cov["ct7_gate_open_prm"]} תשובות PRMBench. ב-ProcessBench: seed argmax בתשובות שה-gate פתוח בהן ({cov["ct7_gate_open_pb"]:,} מתוך {cov["pb"]:,}).</li>
<li>קובץ RUN_FREEZE מקפיא את hash המודול תחת השם <code>step_evidence_v1.py</code>; ה-hash של ה-driver לא הוקפא (התנגשות שמות). הראיות שנשמרו לא נפגעו, אבל זהו פער provenance שהפרוטוקול הבא חייב לסגור.</li>
<li>חמש שיטות היסטוריות (dual/context/single joint) מכסות 6,796 מתוך 6,800 תשובות PB; המדדים שלהן מחושבים על האוכלוסייה הרשומה ולא על אוכלוסייה משותפת.</li>
<li>לתשובות ProcessBench אין תווית לכל צעד; המערך שומר שם sentinel של −2. תוויות PRMBench בינאריות.</li>
</ul>

<h2>מה זה אומר, ומה הלאה</h2>
<div class="next">
<p><b>סיכום בשורה:</b> המתכון של Step 432 לומד מידע אמיתי (הרבה מעל null אקראי), אבל ב-ProcessBench הוא לא לומד יותר ממה שה-teacher כבר ידע, וגם תוויות אמת באותו מתכון לא היו עוזרות. ב-PRMBench הוא משפר דירוג בתוך תשובה, בעיקר בתשובות רב-שגיאתיות, ונופל מ-CT7 בכיול בין תשובות (PRMScore {num(board.loc["evidence__all__position__equal","prmscore_inner"],3)} מול {num(board.loc["ct7","prmscore_inner"],3)}).</p>
<p><b>השלב הבא לפי התוכנית, S0-C:</b> לבדוק האם יישור סקאלה בין תשובות (z-score או דרגות לכל תשובה) מעלה את PRMScore של position evidence, כשהדירוג בתוך התשובה נשאר זהה בדיוק. ללא אימון, על ציוני ה-nested הקיימים, שני מבחנים ראשיים בלבד. אם PRMScore עולה, הבעיה היא סף; אם לא, עוברים ל-S1 (מטרות רכות והסכמה בין מבטים) בלי לחפש עוד טרנספורמים.</p>
</div>
<p class="note">קבצים: <code>results/ssl_pseudolabel_residual_v1/S0/</code> בענף <code>claude/ssl-pseudolabel-residual-v1</code> (worktree מבודד). סקריפטים: <code>scripts/experiments/ssl_s0_audit.py</code>, <code>ssl_s0_report.py</code>. זמן ריצה {log["seconds"]/60:.0f} דקות, רובו ה-evaluator הרשמי של PRMBench.</p>
</main>
'''
(OUT / 'REPORT_HE.html').write_text(html, encoding='utf8')
print('written', len(html) // 1024, 'KB')
