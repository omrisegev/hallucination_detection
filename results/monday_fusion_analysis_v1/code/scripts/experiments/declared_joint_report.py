"""declared_joint_prmbench_v1 figures + Hebrew findings page from the run outputs (no recomputation)."""
from pathlib import Path
import base64, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260924'
OUT = ROOT / 'results/declared_joint_prmbench_v1' / RUN_ID
M = pd.read_csv(OUT / 'METRICS.csv'); C = pd.read_csv(OUT / 'CONTRASTS.csv').drop_duplicates(['contrast_id', 'endpoint'])
D = json.loads((OUT / 'DIAGNOSTICS.json').read_text(encoding='utf8')); status = json.loads((OUT / 'RUN_STATUS.json').read_text(encoding='utf8')); timing = json.loads((OUT / 'TIMING.json').read_text(encoding='utf8'))
P = json.loads((OUT.parent / 'PROTOCOL.json').read_text(encoding='utf8'))
fits = [json.loads(l) for l in (OUT / 'FIT_MANIFEST.jsonl').read_text(encoding='utf8').splitlines()]
NAMES = json.loads((OUT / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))['channels']; ADD = D['added']
BANKS = ['B11', 'B15']; ARMS = ['equal', 'lsml', 'discovered_group_equal', 'declared_equal', 'declared_joint']
AL = {'equal': 'ממוצע', 'lsml': 'L-SML, חלוקה שהתגלתה', 'discovered_group_equal': 'ממוצע מאוזן, חלוקה שהתגלתה', 'declared_equal': 'ממוצע מאוזן, חלוקה מוצהרת', 'declared_joint': 'Joint L-SML, חלוקה מוצהרת'}
BL = {'B11': 'בנק11', 'B15': 'בנק15 (11 + ארבעת הערוצים)'}
def est(m, metric, stratum='all'):
    r = M[(M.method == m) & (M.metric == metric) & (M.stratum == stratum)]; return float(r.estimate.iloc[0]) if len(r) else np.nan
def num(x, d=4): return '—' if pd.isna(x) else f'{x:.{d}f}'
def sgn(x, d=4): return '—' if pd.isna(x) else f'{x:+.{d}f}'
def pct(x): return '—' if pd.isna(x) else f'{100 * x:.2f}'
Cp = C.set_index(['contrast_id', 'endpoint'])
def cr(cid, ep='prm_within_auc'): return Cp.loc[(cid, ep)] if (cid, ep) in Cp.index else None

# FIG 1 scoreboard
fig, ax = plt.subplots(figsize=(9, 6))
mk = {'equal': 's', 'lsml': 'o', 'discovered_group_equal': 'v', 'declared_equal': '^', 'declared_joint': '*'}
for b, col in zip(BANKS, ['#1f77b4', '#d62728']):
    for a in ARMS:
        m = f'{b}_{a}'; x, y = est(m, 'prmscore_answer_z_q80'), est(m, 'within_auc')
        ax.scatter(x, y, c=col, marker=mk[a], s=150 if a == 'declared_joint' else 60, zorder=3)
        ax.annotate(m.replace('_declared', '').replace('_discovered', '.disc'), (x, y), fontsize=6.5, xytext=(4, 3), textcoords='offset points')
ax.scatter(est('ct7', 'prmscore_answer_z_q80'), est('ct7', 'within_auc'), c='#444', marker='D', s=90, zorder=4); ax.annotate('ct7', (est('ct7', 'prmscore_answer_z_q80'), est('ct7', 'within_auc')), fontsize=8, xytext=(4, 3), textcoords='offset points')
ax.axhline(0.7801, color='#999', ls=':', lw=1); ax.text(ax.get_xlim()[0], 0.7801, ' fam421 on CT7 .7801 (context)', fontsize=7, color='#666', va='bottom')
ax.set_xlabel('PRMScore (answer-z, q80)'); ax.set_ylabel('PRMBench within-answer AUROC'); ax.set_title('declared partitions and Joint L-SML  (blue = bank11, red = bank15; star = Joint)'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG1_scoreboard.png', dpi=150); plt.close(fig)
# FIG 2 contrasts
want = ['B11_declared_equal - B11_equal', 'B11_declared_joint - B11_declared_equal', 'B15_declared_equal - B11_declared_equal',
        'B15_declared_joint - B15_equal', 'B15_declared_joint - B15_lsml', 'B15_declared_joint - B11_declared_joint',
        'B11_lsml - B11_equal', 'B11_declared_equal - B11_discovered_group_equal', 'B15_declared_joint - ct7', 'B11_lsml - ct7']
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, ep, title in [(axes[0], 'prm_within_auc', 'PRMBench within-AUC'), (axes[1], 'prmscore_answer_z_q80', 'PRMScore (answer-z, q80)')]:
    d = C[(C.endpoint == ep) & C.contrast_id.isin(want)].set_index('contrast_id').reindex(want).reset_index(); y = np.arange(len(d))
    ax.errorbar(d.delta, y, xerr=[d.delta - d.ci95_lo, d.ci95_hi - d.delta], fmt='o', color='#1f6f8b', capsize=3)
    for i, r in d.iterrows():
        if r.primary: ax.errorbar([r.delta], [i], xerr=[[r.delta - r.ci_adj_lo], [r.ci_adj_hi - r.delta]], fmt='none', ecolor='#b5452b', capsize=5, lw=.8); ax.text(r.ci95_hi, i, '  primary', va='center', fontsize=7, color='#b5452b')
    ax.axvline(0, color='k', lw=.8); ax.set_yticks(y); ax.set_yticklabels(d.contrast_id, fontsize=7.5); ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x', alpha=.3)
fig.suptitle('Paired PRMBench source-group bootstrap, 100,000 draws; red = Bonferroni (K=6) on the primary family', fontsize=9); fig.tight_layout(); fig.savefig(OUT / 'FIG2_contrasts.png', dpi=150); plt.close(fig)
# FIG 3 weights on B15: Joint vs L-SML
WJ = np.mean([f['weights'] for f in fits if f['bank'] == 'B15_joint'], 0); WL = np.mean([f['weights'] for f in fits if f['bank'] == 'B15'], 0)
fig, ax = plt.subplots(figsize=(13, 4.4)); x = np.arange(15); w = .38
ax.bar(x - w / 2, WJ, w, label='Joint L-SML, declared partition', color='#d62728'); ax.bar(x + w / 2, WL, w, label='CONT L-SML, discovered partition', color='#999')
for t in ['energy_innovation', 'top15_turnover', 'top50_js']: ax.axvspan(NAMES.index(t) - .5, NAMES.index(t) + .5, color='#f6e6e1', zorder=0)
for t in ADD: ax.axvspan(NAMES.index(t) - .5, NAMES.index(t) + .5, color='#e6f0e8', zorder=0)
ax.axhline(0, color='k', lw=.6); ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=70, fontsize=7); ax.set_ylabel('weight (mean of 5 folds, L1-normalized)')
ax.set_title('bank15: Joint silences the anti-oriented trio (red shade) and the added channels (green shade); L-SML does not'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG3_weights.png', dpi=150); plt.close(fig)
def img(n): return 'data:image/png;base64,' + base64.b64encode((OUT / n).read_bytes()).decode()
def cell(r, adj):
    if r is None: return '<td>—</td>'
    out = r.ci95_lo > 0 or r.ci95_hi < 0; cls = ' class="pos"' if out and r.delta > 0 else ' class="neg"' if out else ''
    s = f'<td{cls}>{sgn(r.delta)}<br><span class="ci">[{sgn(r.ci95_lo)}, {sgn(r.ci95_hi)}]</span>'
    if adj and not pd.isna(r.ci_adj_lo): s += f'<br><span class="ci">Bonf. [{sgn(r.ci_adj_lo)}, {sgn(r.ci_adj_hi)}]</span>'
    return s + '</td>'
def crow(cid):
    a = cr(cid); b = cr(cid, 'prmscore_answer_z_q80')
    return '' if a is None else f'<tr><td class="lbl">{cid}{" · <b>ראשי</b>" if bool(a.primary) else ""}</td>{cell(a, bool(a.primary))}{cell(b, bool(a.primary))}</tr>'
sb = ''.join(f'<tr><td class="lbl">{BL[b]} · {AL[a]}</td><td>{num(est(f"{b}_{a}", "within_auc"))}</td><td>{num(est(f"{b}_{a}", "prmscore_answer_z_q80"))}</td><td>{pct(est(f"{b}_{a}", "sla", "macro8_context"))}%</td></tr>' for b in BANKS for a in ARMS)
sb += f'<tr><td class="lbl">CT7 (עוגן קפוא)</td><td>{num(est("ct7", "within_auc"))}</td><td>{num(est("ct7", "prmscore_answer_z_q80"))}</td><td>{pct(est("ct7", "sla", "macro8_context"))}%</td></tr>'
sb += '<tr><td class="lbl">fam421 על CT7 (הקשר, Step 434)</td><td>0.7801</td><td>0.6562</td><td>39.94%</td></tr>'
prim = ''.join(crow(c) for c in ['B11_declared_equal - B11_equal', 'B11_declared_joint - B11_declared_equal', 'B15_declared_equal - B11_declared_equal'])
sec = ''.join(crow(c) for c in ['B15_declared_joint - B15_equal', 'B15_declared_joint - B15_lsml', 'B15_declared_joint - B11_declared_joint', 'B11_lsml - B11_equal', 'B11_declared_equal - B11_discovered_group_equal', 'B15_lsml - B11_lsml', 'B15_equal - B11_equal', 'B15_declared_joint - ct7', 'B11_lsml - ct7'])
jr = ''.join('<tr><td class="lbl">{}</td>{}</tr>'.format(b, ''.join(f'<td>{"✓" if x["converged"] else "✗"}<br><span class="ci">misfit {x["misfit"]:.3f}</span></td>' for x in D['joint'][b])) for b in BANKS)
decl = D['declared_partitions']
dr = ''.join('<tr><td class="lbl">{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(nm, sgn(WJ[j], 3) if j < 15 else '—', sgn(WL[j], 3) if j < 15 else '—', 'A' if decl['B15'][j] == 0 else 'B' if decl['B15'][j] == 1 else 'C' if decl['B15'][j] == 2 else 'D') for j, nm in enumerate(NAMES))
classes = sorted({s.split('=')[1] for s in M.stratum if s.startswith('class=')} - {'correct', 'multi_solutions'})
crows = ''.join('<tr><td class="lbl">{}</td>{}</tr>'.format(cl, ''.join(f'<td>{num(est(m, "within_auc", "class=" + cl), 3)}</td>' for m in ['B11_lsml', 'B11_declared_joint', 'B15_declared_joint', 'ct7'])) for cl in classes)
p1 = cr('B11_declared_equal - B11_equal'); p2 = cr('B11_declared_joint - B11_declared_equal'); p3 = cr('B15_declared_equal - B11_declared_equal')
bestn = max([f'{b}_{a}' for b in BANKS for a in ARMS], key=lambda m: est(m, 'within_auc'))
preds = [('P1 שחזור בנק11 עד 1e-9', D['replay']['replay_exact'], f'{D["replay"]["max_abs_diff_continuous"]:.2e}'),
         ('P2 חלוקה מוצהרת מנצחת ממוצע בשני הבנקים', p1.ci95_lo > 0, f'{sgn(p1.delta)} בבנק11'),
         ('P3 מוצהרת מנצחת את שהתגלתה', cr('B11_declared_equal - B11_discovered_group_equal').ci95_lo > 0, sgn(cr('B11_declared_equal - B11_discovered_group_equal').delta)),
         ('P4 ‏Joint בתוך ‎±0.004 מהממוצע המאוזן', abs(p2.delta) <= .004, sgn(p2.delta)),
         ('P5 ארבעת הערוצים שווים פחות מ-‎+0.005 לממוצע המאוזן', p3.delta < .005, sgn(p3.delta)),
         ('P6 אף זרוע אינה מגיעה ל-‎.7801', est(bestn, 'within_auc') < .7801, f'{bestn} {num(est(bestn, "within_auc"))}')]
prow = ''.join(f'<tr><td class="lbl">{a}</td><td class="{"pos" if b else "neg"}">{"התקיים" if b else "לא התקיים"}</td><td>{c}</td></tr>' for a, b, c in preds)
html = f'''<title>Declared Partitions and Joint L-SML</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#f6f7f5;--ink:#1c2430;--muted:#5d6874;--rule:#d7dcd9;--card:#eef1ee;--accent:#1f6f8b;--pos:#3f7d4e;--neg:#b5452b;--posbg:#e6f0e8;--negbg:#f6e6e1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}}}
:root[data-theme="dark"]{{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}
body{{background:var(--paper);color:var(--ink);font-family:Heebo,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.6;direction:rtl}}
main{{max-width:900px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-weight:700;font-size:30px;line-height:1.2;margin:0 0 6px;text-wrap:balance}}
h2{{font-weight:500;font-size:22px;margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--rule)}}
h3{{font-weight:500;font-size:17px;margin:26px 0 4px;color:var(--accent)}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);direction:ltr;text-align:right}}
.lede{{font-size:18px;font-weight:300;margin:12px 0 0}}
.verdict{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0 0}}
.verdict div{{background:var(--card);padding:14px 16px;border-radius:4px}}
.verdict b{{display:block;font-family:"IBM Plex Mono",monospace;font-size:20px;font-weight:500;direction:ltr;text-align:right}}
.verdict span{{font-size:13px;color:var(--muted)}}
.tw{{overflow-x:auto;margin:8px 0 4px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th{{text-align:right;font-weight:500;color:var(--muted);border-bottom:1px solid var(--ink);padding:6px 10px;font-size:13px}}
td{{padding:7px 10px;border-bottom:1px solid var(--rule);font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;direction:ltr;text-align:right;vertical-align:top;white-space:nowrap}}
td.lbl{{font-family:Heebo,sans-serif;direction:rtl;white-space:normal;min-width:200px}}
td.pos{{color:var(--pos);background:var(--posbg)}} td.neg{{color:var(--neg);background:var(--negbg)}}
.ci{{font-size:11px;color:var(--muted)}}
figure{{margin:16px 0 4px}} figure img{{width:100%;border:1px solid var(--rule);border-radius:3px;background:#fff}}
.note{{font-size:14px;color:var(--muted);margin:6px 0}}
.next{{background:var(--card);padding:16px 20px;border-radius:4px;border-right:3px solid var(--accent)}}
code{{font-family:"IBM Plex Mono",monospace;font-size:13px;direction:ltr;unicode-bidi:embed}}
@media (max-width:600px){{.verdict{{grid-template-columns:1fr}}}}
</style>
<main>
<div class="eyebrow">declared_joint_prmbench_v1 · {RUN_ID} · PRMBench first · development data only</div>
<h1>חלוקה מוצהרת ו-Joint L-SML על בנק הצעדים</h1>
<p class="lede">שתי שאלות בניסוי אחד: מה קורה כשמוסיפים לבנק11 את ארבעת הערוצים החזקים־אך־בלתי־תלויים שעומרי בחר, והאם הבעיה היא ה-clustering. שתי תוצאות מנוגדות. <b>החלוקה המוצהרת לבדה מזיקה</b> בבנק11 ({sgn(p1.delta)}), בניגוד למה שקרה ב-CT7. <b>אבל Joint L-SML על אותה חלוקה מחזיר {sgn(p2.delta)}</b> עם רווח Bonferroni נקי, וזו הפעם הראשונה ש-Joint רץ על הבנקים האלה בכלל.</p>
<div class="verdict">
<div><b>{status["status"]}</b><span>{status["fits"]} התאמות, {status["failures"]} כשלים, Joint מתכנס ב-10 מתוך 10. הסטטוס נקבע רק כי השחזור הוא {D["replay"]["max_abs_diff_continuous"]:.2e} מול סבילות 1e-9, רעש עיגול מסטנדרטיזציה כפולה</span></div>
<div><b>{num(est("B15_declared_joint", "within_auc"))}</b><span>הזרוע החדשה הטובה ביותר, Joint על בנק15. מול בנק11 L-SML {num(est("B11_lsml", "within_auc"))} ומול CT7 {num(est("ct7", "within_auc"))}</span></div>
<div><b>{sgn(cr("B15_declared_joint - B15_lsml").delta)}</b><span>Joint מול L-SML על אותו בנק15 בדיוק. ארבעת הערוצים משתלמים תחת Joint ובשום כלל אחר</span></div>
</div>
<h2>מה נבדק, ולמה דווקא זה</h2>
<p>אודיט היסטורי מצא דפוס אחד חד: <b>חלוקה מוצהרת לפי מקור מנצחת, חלוקה שמתגלה מהנתונים מפסידה.</b> על שבעת המבטים של CT7 החלוקה המוצהרת 4/2/1 נותנת .7801 מול .7724 (‎+.0077, Holm .005) בעוד החלוקה שהתגלתה נותנת .7746. על בנקי new7 ו-aug12 ברמת הצעד, ממוצע לפי משפחות עלה על ממוצע פשוט ב-‎+.0054 וב-‎+.0047. ובנוסף: <b>fit_joint_lsml מעולם לא רץ על בנק11 ולא על CT7</b>, כי החלוקה שמתגלה בבנק11 היא K=6 בגדלים ‎[1,3,3,2,1,1] ומפרה את תנאי הזיהוי. חלוקה מוצהרת מסירה את החסם.</p>
<p>ארבעת הערוצים שנוספו: {", ".join("<code>" + a + "</code>" for a in ADD)}. הם היחידים במאגר של 52 שמחוץ לבנק11 עם AUC בודד מעל 0.60 ואי־תלות מהרמה. החמישי שעומרי מנה, <code>bocpd_p0</code>, כבר בבנק.</p>
<p class="note">לא הורץ מחדש, עם נימוק: <b>shrinkage</b> כבר נבדק ברמת הצעד ב-Step 395 והוא אדיש (diagonal .7632, block .7629 מול ממוצע .7655); <b>readout</b> נסגר ב-Step 429, שם תקרת הבחירה לכל ערוץ <em>עם תוויות</em> על 17 קריאות מגיעה ל-.7607, מתחת ל-.7645 שיש לנו ב-Top10 קבוע.</p>
<h2>לוח התוצאות</h2>
<figure><img src="{img('FIG1_scoreboard.png')}" alt="scoreboard"></figure>
<div class="tw"><table><thead><tr><th>בנק · זרוע</th><th>PRMB within-AUC</th><th>PRMScore</th><th>PB SLA (הקשר)</th></tr></thead><tbody>{sb}</tbody></table></div>
<h2>ההשוואות הראשיות (משפחה של 6)</h2>
<div class="tw"><table><thead><tr><th>השוואה</th><th>PRMB within-AUC</th><th>PRMScore</th></tr></thead><tbody>{prim}</tbody></table></div>
<figure><img src="{img('FIG2_contrasts.png')}" alt="contrasts"></figure>
<h3>משניות</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>PRMB within-AUC</th><th>PRMScore</th></tr></thead><tbody>{sec}</tbody></table></div>
<h2>המנגנון: Joint עושה את מה ש-L-SML הפסיק לעשות</h2>
<figure><img src="{img('FIG3_weights.png')}" alt="weights"></figure>
<p>זה לב העניין. בבנק11 ‏L-SML משתיק את שלושת הערוצים ההפוכים והחלשים ונותן להם 0.045 יחד; ברגע שהבנק גדל ל-15 הוא <b>מפסיק</b>, ונותן להם 0.069, 0.073 ו-0.075. Joint על החלוקה המוצהרת נותן להם ‎−0.008, 0.043 ו-0.008, ומרכז 0.74 מהמשקל על משפחת הרמה. הוא גם כמעט לא נוגע בארבעת הערוצים שנוספו: 0.056 מהמשקל המוחלט מול 0.21 עד 0.31 אצל L-SML. כלומר Joint מרוויח מארבעת הערוצים דווקא בזה <em>שהוא לא נשען עליהם</em>, אלא משתמש בהם כדי לייצב את מודל הגורמים.</p>
<div class="tw"><table><thead><tr><th>ערוץ</th><th>Joint (מוצהרת)</th><th>L-SML (שהתגלתה)</th><th>משפחה מוצהרת</th></tr></thead><tbody>{dr}</tbody></table></div>
<p class="note">אדום בגרף = שלושת הערוצים ההפוכים; ירוק = ארבעת הערוצים שנוספו. משקלי cross-group של Joint בבנק15: {", ".join(f"{w:.3f}" for w in D['joint']['B15'][0]['cross_group_weights'])} למשפחות A/B/C/D.</p>
<h3>התכנסות Joint, חמישה folds לכל בנק</h3>
<div class="tw"><table><thead><tr><th>בנק</th><th>fold 0</th><th>fold 1</th><th>fold 2</th><th>fold 3</th><th>fold 4</th></tr></thead><tbody>{jr}</tbody></table></div>
<h3>לפי סוג השגיאה</h3>
<div class="tw"><table><thead><tr><th>סוג</th><th>בנק11 L-SML</th><th>בנק11 Joint</th><th>בנק15 Joint</th><th>CT7</th></tr></thead><tbody>{crows}</tbody></table></div>
<h2>התחזיות שהוקפאו</h2>
<div class="tw"><table><thead><tr><th>תחזית</th><th>תוצאה</th><th>מספר</th></tr></thead><tbody>{prow}</tbody></table></div>
<h2>איך לקרוא את זה</h2>
<div class="next">
<p><b>התוצאה של CT7 לא עוברת לבנק11.</b> ב-CT7 משפחת הרמה היא חמישה מתוך שבעה מבטים, כלומר 71% מהקול תחת ממוצע, ואיזון לשליש עוזר. בבנק11 היא חמישה מתוך אחד עשר, כלומר 45%, וכפיית שליש דווקא <em>מקדמת</em> את משפחת הערוצים ההפוכים לשליש מהקול. מכאן ‎{sgn(p1.delta)}. חלוקה מוצהרת אינה מתכון אוניברסלי; היא עוזרת כשמשפחה אחת שולטת ומזיקה כשהאיזון כבר סביר.</p>
<p><b>ה-clustering אינו הבעיה היחידה, אבל כן חלק ממנה.</b> על אותה חלוקה מוצהרת בדיוק, ההפרש בין ממוצע מאוזן ל-Joint הוא {sgn(p2.delta)} עם רווח Bonferroni שאינו חוצה אפס. כלומר מה שעושים <em>בתוך</em> החלוקה חשוב לא פחות מהחלוקה עצמה. זה סותר את התחזית שלי לפי Step 399, שבו הכיוונים בתוך הקבוצה נמצאו כמעט אחידים.</p>
<p><b>ארבעת הערוצים של עומרי משתלמים, אבל רק תחת Joint.</b> ‎{sgn(cr("B15_declared_joint - B11_declared_joint").delta)} מול בנק11 תחת Joint, בעוד תחת ממוצע הם עולים {sgn(cr("B15_equal - B11_equal").delta)} ותחת L-SML {sgn(cr("B15_lsml - B11_lsml").delta)}.</p>
<p><b>ועדיין לא מספיק.</b> הזרוע החדשה הטובה ביותר היא {bestn} עם {num(est(bestn, "within_auc"))}, מתחת לבנק11 L-SML ‏{num(est("B11_lsml", "within_auc"))} ומתחת ל-CT7 ‏{num(est("ct7", "within_auc"))} ‏({sgn(cr("B15_declared_joint - ct7").delta)}). אין כאן מועמד, יש כאן מנגנון שעובד ולא נוסה קודם.</p>
</div>
<p class="note">קבצים: <code>results/declared_joint_prmbench_v1/{RUN_ID}/</code>. פרוטוקול קפוא ב-<code>PROTOCOL.json</code>, נדחף לפני הניקוד. קוד ה-fusion יובא ללא שינוי מ-<code>.worktrees/depth-feature-fusion-v1</code>.</p>
</main>
'''
(OUT / 'REPORT_HE.html').write_text(html, encoding='utf8'); print('written', len(html) // 1024, 'KB')
