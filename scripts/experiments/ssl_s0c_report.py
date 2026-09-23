"""S0-C figures + Hebrew findings page from the S0-C outputs (no recomputation)."""
from pathlib import Path
import base64, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/ssl_pseudolabel_residual_v1/S0C'
res = pd.read_csv(OUT / 'S0C_CALIBRATION.csv'); boot = json.loads((OUT / 'BOOTSTRAP.json').read_text(encoding='utf8'))
thr = pd.read_csv(OUT / 'THRESHOLDS_PER_FOLD.csv'); ident = json.loads((OUT / 'WITHIN_RANK_IDENTITY.json').read_text(encoding='utf8'))
hyg = pd.read_csv(OUT / 'CALIBRATION_HYGIENE.csv'); P = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))
ct7 = boot['ct7_reference']; B = pd.DataFrame(boot['contrasts'])
N = {'evidence__all__plain__equal': 'top5 plain evidence', 'evidence__all__position__equal': 'top5 position evidence', 'evidence__all__position2__equal': 'top5 position, איטרציה 2'}
TN = {'C_RAW': 'RAW (ללא שינוי)', 'C_Z': 'Z (z-score לכל תשובה)', 'C_ECDF': 'ECDF (דרגות לכל תשובה)'}
R = res.set_index(['method', 'transform'])

# ---- Figure 1: PRMScore per arm (inner-selected and q80), CT7 line, original run marker
fig, ax = plt.subplots(figsize=(9, 4.8))
x = 0; ticks = []; labs = []
for m in P['methods']:
    for t in ['C_RAW', 'C_Z', 'C_ECDF']:
        r = R.loc[(m, t)]
        ax.bar(x - .2, r.prmscore_inner, .38, color='#1f6f8b', label='inner-selected threshold' if x == 0 else None)
        ax.bar(x + .2, r.prmscore_q80, .38, color='#8fbccb', label='fixed q80 (label-free)' if x == 0 else None)
        ticks.append(x); labs.append(f'{N[m]}\n{t}'); x += 1
    ax.plot([x - 3.5, x - .5], [R.loc[(m, "C_RAW")].original_run_prmscore_inner] * 2, ls=':', color='#333', lw=1.2, label='original run (inner)' if m == P['methods'][0] else None)
    ax.text(x - 2, .505, {'evidence__all__plain__equal': 'top5 plain evidence', 'evidence__all__position__equal': 'top5 position evidence', 'evidence__all__position2__equal': 'top5 position, iteration 2'}[m], ha='center', fontsize=8, color='#333')
    x += .6
ax.axhline(ct7['prmscore_inner'], color='#b5452b', lw=1.3, label=f'CT7 {ct7["prmscore_inner"]:.3f}')
ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(.5, .68); ax.set_ylabel('PRMScore (official, 6,969 answers)')
ax.set_title('S0-C: same within-answer ranking, three between-answer scales'); ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8, loc='upper left', ncol=2)
fig.tight_layout(); fig.savefig(OUT / 'FIG1_prmscore_arms.png', dpi=150); plt.close(fig)

# ---- Figure 2: paired contrasts with CIs
fig, ax = plt.subplots(figsize=(8.5, 4.2))
d = B[B.kind == 'inner'].reset_index(drop=True)
y = np.arange(len(d)); lo = d.ci95.map(lambda c: c[0]); hi = d.ci95.map(lambda c: c[1])
ax.errorbar(d.delta_prmscore, y, xerr=[d.delta_prmscore - lo, hi - d.delta_prmscore], fmt='o', color='#1f6f8b', capsize=3)
for i, r in d.iterrows():
    if r.primary: ax.text(r.ci95[1] + .002, i, 'primary', va='center', fontsize=7, color='#b5452b')
ax.set_yticks(y); ax.set_yticklabels([f'{N[m]}: {c}' for m, c in zip(d.method, d.contrast)], fontsize=7); ax.invert_yaxis(); ax.axvline(0, color='k', lw=.8)
ax.set_xlabel('Δ PRMScore (inner-selected thresholds), 95% CI from 100,000 paired source-group draws'); ax.grid(axis='x', alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG2_contrasts.png', dpi=150); plt.close(fig)

# ---- Figure 3: per-fold selected quantile and threshold stability
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for t, c in zip(['C_RAW', 'C_Z', 'C_ECDF'], ['#444', '#1f6f8b', '#3f7d4e']):
    s = thr[(thr.method == P['primary_family']['method']) & (thr.transform == t)]
    axes[0].plot(s.outer_fold, s.selected_quantile, marker='o', color=c, label=t); axes[1].plot(s.outer_fold, s.calibration_prmscore_at_selected, marker='o', color=c, label=t)
axes[0].set_title('quantile selected on calibration folds'); axes[0].set_xlabel('outer fold'); axes[0].set_ylim(.5, 1); axes[0].grid(alpha=.3); axes[0].legend(fontsize=8)
axes[1].set_title('calibration PRMScore at the selected quantile'); axes[1].set_xlabel('outer fold'); axes[1].grid(alpha=.3)
fig.suptitle(f'{N[P["primary_family"]["method"]]}: threshold selection per fold', fontsize=10); fig.tight_layout(); fig.savefig(OUT / 'FIG3_thresholds.png', dpi=150); plt.close(fig)

# ---- page
def img(name): return 'data:image/png;base64,' + base64.b64encode((OUT / name).read_bytes()).decode()
def num(x, d=4): return f'{x:.{d}f}'
def sgn(x, d=4): return f'{x:+.{d}f}'
def pct(x, d=1): return f'{100*x:.{d}f}%'
pm = P['primary_family']['method']
prim = {r['contrast']: r for _, r in B[B.primary].iterrows()}
diag = pd.read_csv(OUT / 'DIAGNOSTIC_COMPARATORS_TRANSFORMED.csv'); dboot = json.loads((OUT / 'DIAGNOSTIC_COMPARATORS_BOOTSTRAP.json').read_text(encoding='utf8'))['contrasts']
D = diag.set_index(['method', 'transform'])
DN = {'ct7': 'CT7', 'token_lsml': 'token L-SML', 'evidence__all__position__equal': 'top5 position evidence'}
drows = ''.join(f'<tr><td class="lbl">{DN[m]}</td>' + ''.join(f'<td>{num(D.loc[(m, t)].prmscore_inner)}</td>' for t in ['C_RAW', 'C_Z', 'C_ECDF']) + '</tr>' for m in ['ct7', 'token_lsml']) + f'<tr><td class="lbl">top5 position evidence (S0-C)</td>' + ''.join(f'<td>{num(R.loc[(pm, t)].prmscore_inner)}</td>' for t in ['C_RAW', 'C_Z', 'C_ECDF']) + '</tr>'
dbrows = ''.join(f'<tr><td class="lbl">{DN[c["method"]]}<br><span class="vs">{c["contrast"]}</span></td><td{" class=pos" if c["ci95"][0] > 0 else " class=neg" if c["ci95"][1] < 0 else ""}>{sgn(c["delta"])}</td><td>[{sgn(c["ci95"][0])}, {sgn(c["ci95"][1])}]</td></tr>' for c in dboot)
def ci(r): return f'[{sgn(r["ci95"][0])}, {sgn(r["ci95"][1])}]'
def cls(r): return ' class="pos"' if r["ci95"][0] > 0 else ' class="neg"' if r["ci95"][1] < 0 else ''
def bonf(r): b = r['ci_bonferroni_97.5']; return f'[{sgn(b[0])}, {sgn(b[1])}]' if isinstance(b, list) else '—'
rows = ''
for m in P['methods']:
    for t in ['C_RAW', 'C_Z', 'C_ECDF']:
        r = R.loc[(m, t)]
        rows += f'<tr><td class="lbl">{N[m]}<br><span class="vs">{TN[t]}</span></td><td>{num(r.within_auc)}</td><td>{num(r.prmscore_inner)}</td><td>{num(r.prmscore_q80)}</td><td>{num(r.original_run_prmscore_inner)}</td><td>{pct(r.clean_control_false_alarm_inner)}</td><td>{pct(r.flagged_fraction_eligible_inner)}</td><td>{num(r.pooled_step_auroc)}</td></tr>'
rows += f'<tr><td class="lbl">CT7 (עוגן, הריצה המקורית)</td><td>0.7724</td><td>{num(ct7["prmscore_inner"])}</td><td>{num(ct7["prmscore_q80"])}</td><td>{num(ct7["prmscore_inner"])}</td><td>—</td><td>—</td><td>—</td></tr>'
brows = ''.join(f'<tr><td class="lbl">{N[r["method"]]}<br><span class="vs">{r["contrast"]}{" · <b>ראשי</b>" if r["primary"] else ""}</span></td><td>{r["kind"]}</td><td{cls(r)}>{sgn(r["delta_prmscore"])}</td><td>{ci(r)}</td><td>{bonf(r)}</td><td>{num(r["p_two_sided"],4)}</td></tr>'
                for _, r in B.iterrows())
thr_p = thr[thr.method == pm].pivot(index='outer_fold', columns='transform', values='selected_quantile')
trows = ''.join(f'<tr><td class="lbl">fold {k}</td>' + ''.join(f'<td>{thr_p.loc[k, t]:.2f}</td>' for t in ['C_RAW', 'C_Z', 'C_ECDF']) + '</tr>' for k in thr_p.index)
dz = prim['C_Z - C_RAW']; de = prim['C_ECDF - C_RAW']
def vd(r): return 'משפר' if r['ci95'][0] > 0 else 'מזיק' if r['ci95'][1] < 0 else 'ללא שינוי מובהק'
verdict_z = vd(dz); verdict_e = vd(de); ct7z = D.loc[('ct7', 'C_Z')].prmscore_inner; gap_matched = [c for c in dboot if c['contrast'].startswith('C_Z position')][0]
gap_ct7 = R.loc[(pm, 'C_RAW')].prmscore_inner - ct7['prmscore_inner']
best_t = max(['C_RAW', 'C_Z', 'C_ECDF'], key=lambda t: R.loc[(pm, t)].prmscore_inner)
html = f'''<title>S0-C Calibration · Step 432</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#f6f7f5;--ink:#1c2430;--muted:#5d6874;--rule:#d7dcd9;--card:#eef1ee;--accent:#1f6f8b;--pos:#3f7d4e;--neg:#b5452b;--posbg:#e6f0e8;--negbg:#f6e6e1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}}}
:root[data-theme="dark"]{{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}
body{{background:var(--paper);color:var(--ink);font-family:Heebo,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.6;direction:rtl}}
main{{max-width:820px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-weight:700;font-size:30px;line-height:1.2;margin:0 0 6px;text-wrap:balance}}
h2{{font-weight:500;font-size:22px;margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--rule)}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);direction:ltr;text-align:right}}
.lede{{font-size:18px;font-weight:300;margin:12px 0 0}}
.verdict{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0 0}}
.verdict div{{background:var(--card);padding:14px 16px;border-radius:4px}}
.verdict b{{display:block;font-family:"IBM Plex Mono",monospace;font-size:22px;font-weight:500;direction:ltr;text-align:right}}
.verdict span{{font-size:13px;color:var(--muted)}}
.tw{{overflow-x:auto;margin:8px 0 4px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th{{text-align:right;font-weight:500;color:var(--muted);border-bottom:1px solid var(--ink);padding:6px 10px;font-size:13px}}
td{{padding:7px 10px;border-bottom:1px solid var(--rule);font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;direction:ltr;text-align:right;vertical-align:top;white-space:nowrap}}
td.lbl{{font-family:Heebo,sans-serif;direction:rtl;white-space:normal;min-width:180px}}
td.pos{{color:var(--pos);background:var(--posbg)}} td.neg{{color:var(--neg);background:var(--negbg)}}
.vs{{font-size:12px;color:var(--muted)}}
figure{{margin:16px 0 4px}} figure img{{width:100%;border:1px solid var(--rule);border-radius:3px;background:#fff}}
figcaption{{font-size:13px;color:var(--muted);margin-top:6px}}
.note{{font-size:14px;color:var(--muted);margin:6px 0}}
.next{{background:var(--card);padding:16px 20px;border-radius:4px;border-right:3px solid var(--accent)}}
ul{{padding-right:20px;margin:8px 0}} li{{margin:4px 0}}
code{{font-family:"IBM Plex Mono",monospace;font-size:13px;direction:ltr;unicode-bidi:embed}}
@media (max-width:600px){{.verdict{{grid-template-columns:1fr}}}}
</style>
<main>
<div class="eyebrow">SSL / pseudo-label / residual plan v1.1 · stage S0-C · 2026-09-23 · development data only · no fit</div>
<h1>S0-C: דירוג לעומת כיול</h1>
<p class="lede">אותו דירוג בתוך כל תשובה, שלוש סקאלות שונות בין תשובות. z-score לכל תשובה {verdict_z} את PRMScore של position evidence ({sgn(dz["delta_prmscore"])}), דרגות לכל תשובה {verdict_e} ({sgn(de["delta_prmscore"])}). הפער מ-CT7 יורד מ-{sgn(gap_ct7)} ל-{sgn(gap_matched["delta"])} כשגם CT7 מקבל את אותו טרנספורם. עדיין מתחת ל-CT7.</p>
<div class="verdict">
<div><b>{sgn(dz["delta_prmscore"])}</b><span>Z − RAW, ראשי. רווח סמך 95%: {ci(dz)}; Bonferroni 97.5%: {bonf(dz)}</span></div>
<div><b>{sgn(de["delta_prmscore"])}</b><span>ECDF − RAW, ראשי. רווח סמך 95%: {ci(de)}; Bonferroni 97.5%: {bonf(de)}</span></div>
<div><b>{ident["evaluator_fixture_max_error"]:.0e} / {max(ident["max_abs_within_auc_change"].values()):.0e}</b><span>בדיקות תקינות: ה-evaluator הווקטורי מול הרשמי על {ident["fixture_sets"]} סטים; שינוי מקסימלי ב-within-AUC אחרי הטרנספורם (חייב להיות אפס)</span></div>
</div>

<h2>התוצאה</h2>
<figure><img src="{img('FIG1_prmscore_arms.png')}" alt="PRMScore per arm"><figcaption>PRMScore לכל זרוע: סף שנבחר על folds הכיול (כחול כהה) וסף q80 קבוע (כחול בהיר). קו מקווקו = ה-PRMScore של הריצה המקורית לאותה שיטה. קו אדום = CT7.</figcaption></figure>
<div class="tw"><table><thead><tr><th>זרוע</th><th>within-AUC</th><th>PRMScore, סף נבחר</th><th>PRMScore, q80</th><th>הריצה המקורית</th><th>false alarm על control</th><th>צעדים מסומנים</th><th>pooled AUROC</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="note">within-AUC זהה לחלוטין בתוך כל שיטה: זו הבדיקה שהטרנספורם לא שינה דירוג. "false alarm על control" = חלק הצעדים שסומנו כשגויים בתשובות ה-control (מחלקה correct, {int((~hyg.held_all_in_fold_j).sum()) if False else 758} תשובות) שאינן נכנסות ל-PRMScore הרשמי.</p>

<h2>ההשוואות הזוגיות</h2>
<figure><img src="{img('FIG2_contrasts.png')}" alt="contrasts"></figure>
<div class="tw"><table><thead><tr><th>השוואה</th><th>סף</th><th>Δ PRMScore</th><th>CI 95%</th><th>Bonferroni 97.5%</th><th>p דו-צדדי</th></tr></thead><tbody>{brows}</tbody></table></div>
<p class="note">bootstrap זוגי על {boot["groups"]} קבוצות שאלת-מקור של PRMBench, {boot["draws"]:,} דגימות, seed {boot["seed"]}. משפחה ראשית: שני מבחנים על position evidence עם סף נבחר. השאר משני.</p>

<h2>יציבות הסף</h2>
<figure><img src="{img('FIG3_thresholds.png')}" alt="thresholds"></figure>
<div class="tw"><table><thead><tr><th>fold</th><th>RAW</th><th>Z</th><th>ECDF</th></tr></thead><tbody>{trows}</tbody></table></div>
<p class="note">quantile שנבחר לכל outer fold על ציוני הכיול של position evidence. כל ה-{len(hyg)} צירופי calibration עברו בדיקת היגיינה: התשובות המוחזקות כולן ב-fold j, ואפס קבוצות אימון חופפות ל-folds k או j.</p>

<h2>אבחון אחרי ההערכה: האם גם העוגנים מרוויחים מאותו טרנספורם?</h2>
<p class="note">מחוץ למשפחה הראשית שהוקפאה. הפרוטוקול השאיר את CT7 ו-token L-SML ללא כוונון; הבדיקה הזו קיימת כדי שהקריאה "הפער ל-CT7 כמעט נסגר" לא תישאר לא-מותאמת. אותו פרוטוקול סף מקונן, על ציוני ה-OOF של העוגנים.</p>
<div class="tw"><table><thead><tr><th>שיטה</th><th>RAW</th><th>Z</th><th>ECDF</th></tr></thead><tbody>{drows}</tbody></table></div>
<div class="tw"><table><thead><tr><th>השוואה</th><th>Δ PRMScore</th><th>CI 95%</th></tr></thead><tbody>{dbrows}</tbody></table></div>
<p>CT7 מרוויח מ-z-score רק {sgn(D.loc[("ct7","C_Z")].prmscore_inner - D.loc[("ct7","C_RAW")].prmscore_inner, 3)} ו-token L-SML מפסיד. הטרנספורם מתקן בעיית סקאלה ייחודית לציוני ה-evidence (סכומי log-ratio שגודלם משתנה מאוד בין תשובות: סטיית תקן של ממוצעי-התשובה {num(R.loc[(pm,"C_RAW")].answer_mean_sd_between_answers,2)}), ולא טריק כללי ל-PRMScore. בהשוואה מותאמת, Z מול Z, position evidence עדיין מתחת ל-CT7 ב-{sgn(gap_matched["delta"])} (CI [{sgn(gap_matched["ci95"][0])}, {sgn(gap_matched["ci95"][1])}]).</p>

<h2>איך לקרוא את זה</h2>
<div class="next">
<p><b>מה נבדק:</b> האם ההחלטה "צעד שגוי / תקין" של position evidence נכשלת בגלל סף גלובלי על ציונים שהסקאלה שלהם משתנה בין תשובות. אם כן, יישור לכל תשובה היה אמור להעלות PRMScore בלי לגעת בדירוג.</p>
<p><b>מה יצא:</b> Z {sgn(dz["delta_prmscore"])} ו-ECDF {sgn(de["delta_prmscore"])}, שניהם עם רווחי סמך רחוקים מאפס גם אחרי Bonferroni. כמעט כל הפער של 0.069 ל-CT7 ב-PRMScore היה בעיית סקאלה בין תשובות, לא בעיית דירוג. גם q80 ללא תוויות נותן את אותו רווח, כך שזה לא תוצר של בחירת סף מפוקחת. C_RAW המכויל מחדש משחזר את הריצה המקורית בתוך 0.0006.</p>
<p><b>לפי כלל הפירוש שהוקפא מראש:</b> שיפור PRMScore עם within-AUC זהה הוא שימוש טוב יותר באותם דירוגים לחוזה ההחלטה, לא מידע לוקליזציה חדש. position evidence עדיין מתחת ל-CT7 בשני המדדים: within-AUC {num(R.loc[(pm,"C_RAW")].within_auc,3)} מול 0.772, ו-PRMScore מותאם {num(R.loc[(pm,"C_Z")].prmscore_inner,3)} מול {num(ct7z,3)}. אין מועמד לקידום. המסקנה המעשית: כל ניסוי עתידי שמדווח PRMScore על ציוני evidence או על כל ציון שהוא סכום log-ratio חייב לנרמל לכל תשובה לפני הסף, אחרת PRMScore מודד סקאלה ולא איכות.</p>
</div>
<p class="note">קבצים: <code>results/ssl_pseudolabel_residual_v1/S0C/</code>: PROTOCOL.json (הוקפא לפני הריצה), S0C_CALIBRATION.csv, THRESHOLDS_PER_FOLD.csv, CALIBRATION_HYGIENE.csv, WITHIN_RANK_IDENTITY.json, BOOTSTRAP.json, DECISIONS.npz. סקריפט: <code>scripts/experiments/ssl_s0c_calibration.py</code>. זמן ריצה {boot["seconds"]/60:.0f} דקות.</p>
</main>
'''
(OUT / 'REPORT_HE.html').write_text(html, encoding='utf8')
print('written', len(html) // 1024, 'KB')
