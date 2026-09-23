"""S3 figures + Hebrew findings page from the run outputs (no recomputation)."""
from pathlib import Path
import base64, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260923'
OUT = ROOT / 'results/ssl_pseudolabel_residual_v1/S3' / RUN_ID
M = pd.read_csv(OUT / 'METRICS.csv'); C = pd.read_csv(OUT / 'CONTRASTS.csv').drop_duplicates(['contrast_id', 'endpoint']); D = pd.read_csv(OUT / 'RESIDUAL_DIAGNOSTICS.csv'); ST = pd.read_csv(OUT / 'DIRECTION_STABILITY.csv'); RM = pd.read_csv(OUT / 'RESCUE_MATRIX.csv')
status = json.loads((OUT / 'RUN_STATUS.json').read_text(encoding='utf8')); timing = json.loads((OUT / 'TIMING.json').read_text(encoding='utf8'))
fits = [json.loads(l) for l in (OUT / 'FIT_MANIFEST.jsonl').read_text(encoding='utf8').splitlines()]
CH = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0']
ARMS = ['R_CONTRIB', 'R_CONTRIB_ONLY', 'R_RANDOM_DIR', 'R_RANDOM_DIR_s0', 'R_RANDOM_DIR_s1', 'R_RANDOM_DIR_s2']
COMP = ['ct7', 'token_lsml', 'token_equal', 'evidence30__all__plain__equal', 'evidence__all__position__equal', 'evidence__all__position2__equal']
N = {'BASE': 'BASE (teacher)', 'R_CONTRIB': 'R_CONTRIB (כיוון ניטרלי)', 'R_CONTRIB_ONLY': 'R_CONTRIB_ONLY (aux בלבד, אבחון)', 'R_RANDOM_DIR': 'R_RANDOM_DIR (ממוצע 3 כיוונים אקראיים)', 'R_RANDOM_DIR_s0': 'כיוון אקראי seed 0', 'R_RANDOM_DIR_s1': 'כיוון אקראי seed 1', 'R_RANDOM_DIR_s2': 'כיוון אקראי seed 2',
     'ct7': 'CT7 (עוגן קפוא)', 'token_lsml': 'token L-SML', 'token_equal': 'token equal', 'evidence30__all__plain__equal': 'Step432 top30 plain', 'evidence__all__position__equal': 'Step432 top5 position', 'evidence__all__position2__equal': 'Step432 top5 position iter2'}
def est(m, bench, metric, stratum='all'):
    r = M[(M.method == m) & (M.benchmark == bench) & (M.metric == metric) & (M.stratum == stratum)]
    return float(r.estimate.iloc[0]) if len(r) else np.nan
def num(x, d=4): return '—' if pd.isna(x) else f'{x:.{d}f}'
def pct(x, d=1): return '—' if pd.isna(x) else f'{100*x:.{d}f}%'
def sgn(x, d=4): return f'{x:+.{d}f}'
def pp(x): return f'{100*x:+.2f}'

fig, ax = plt.subplots(figsize=(8.5, 6))
for m in ['BASE', 'R_CONTRIB', 'R_CONTRIB_ONLY', 'R_RANDOM_DIR'] + COMP:
    x, y = est(m, 'prm', 'within_auc'), 100 * est(m, 'pb', 'sla', 'macro8'); anchor = m in COMP
    ax.scatter(x, y, c='#444' if anchor else '#1f77b4' if m == 'BASE' else '#d62728' if m == 'R_CONTRIB' else '#999', marker='D' if anchor else 'o', s=80 if anchor else 60, zorder=3); ax.annotate(N[m], (x, y), fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.set_xlabel('PRMBench within-answer AUROC'); ax.set_ylabel('ProcessBench SLA macro8, %'); ax.set_title('S3: neutral contribution direction as a fixed-dose correction'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG1_scoreboard.png', dpi=150); plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
want = ['R_CONTRIB - BASE', 'R_CONTRIB - R_RANDOM_DIR', 'R_RANDOM_DIR - BASE', 'R_RANDOM_DIR_s0 - BASE', 'R_RANDOM_DIR_s1 - BASE', 'R_RANDOM_DIR_s2 - BASE', 'R_CONTRIB_ONLY - BASE', 'R_CONTRIB - ct7', 'BASE - ct7']
for ax, ep, scale, title in [(axes[0], 'pb_sla_macro8', 100, 'ProcessBench SLA, pp'), (axes[1], 'prm_within_auc', 1, 'PRMBench within-AUC')]:
    d = C[(C.endpoint == ep) & C.contrast_id.isin(want)].set_index('contrast_id').reindex(want).reset_index()
    y = np.arange(len(d)); ax.errorbar(scale * d.delta, y, xerr=[scale * (d.delta - d.ci95_lo), scale * (d.ci95_hi - d.delta)], fmt='o', color='#1f6f8b', capsize=3)
    for i, r in d.iterrows():
        if r.primary: ax.errorbar([scale * r.delta], [i], xerr=[[scale * (r.delta - r.ci_adj_lo)], [scale * (r.ci_adj_hi - r.delta)]], fmt='none', ecolor='#b5452b', capsize=5, lw=.8); ax.text(scale * r.ci95_hi, i, '  primary', va='center', fontsize=7, color='#b5452b')
    ax.axvline(0, color='k', lw=.8); ax.set_yticks(y); ax.set_yticklabels(d.contrast_id, fontsize=8); ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x', alpha=.3)
fig.suptitle('Paired source-group bootstrap, 100,000 draws; blue = 95% CI, red = Bonferroni (K=4) on primary contrasts', fontsize=9); fig.tight_layout(); fig.savefig(OUT / 'FIG2_contrasts.png', dpi=150); plt.close(fig)
# loadings per task/fold + eigenvalue spectrum
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for task, c in zip(['pb_q4', 'pb_q8', 'prm'], ['#1f77b4', '#2ca02c', '#d62728']):
    for k in range(5):
        r = D[(D.task == task) & (D.fold == k)].iloc[0]; axes[0].plot(range(11), [r[f'load_{ch}'] for ch in CH], color=c, alpha=.5, lw=1, label=task if k == 0 else None)
axes[0].set_xticks(range(11)); axes[0].set_xticklabels(CH, rotation=60, fontsize=7); axes[0].axhline(0, color='k', lw=.6); axes[0].set_title('v_neutral loadings per fold (sum oriented positive)'); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
for f in fits:
    if f['eigenvalues'] is not None: axes[1].plot(sorted(f['eigenvalues']), marker='.', alpha=.6, lw=1, label=f['model_id'] if f['model_id'].endswith('fold0/contrib') else None)
axes[1].axhline(1, color='#b5452b', lw=1, ls='--'); axes[1].set_title('eigenvalue spectrum of the cell-averaged covariance of U (per fit)'); axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG3_direction.png', dpi=150); plt.close(fig)

def img(name): return 'data:image/png;base64,' + base64.b64encode((OUT / name).read_bytes()).decode()
Cp = C.set_index(['contrast_id', 'endpoint'])
def crow(cid):
    if (cid, 'pb_sla_macro8') not in Cp.index: return ''
    a = Cp.loc[(cid, 'pb_sla_macro8')]; b = Cp.loc[(cid, 'prm_within_auc')]
    def cell(r, f, adj):
        out = r.ci95_lo > 0 or r.ci95_hi < 0; cls = ' class="pos"' if out and r.delta > 0 else ' class="neg"' if out else ''
        s = f'<td{cls}>{f(r.delta)}<br><span class="ci">[{f(r.ci95_lo)}, {f(r.ci95_hi)}]</span>'
        if adj and not pd.isna(r.ci_adj_lo): s += f'<br><span class="ci">Bonf. [{f(r.ci_adj_lo)}, {f(r.ci_adj_hi)}]</span>'
        return s + '</td>'
    return f'<tr><td class="lbl">{cid}{" · <b>ראשי</b>" if bool(a.primary) else ""}</td>{cell(a, pp, bool(a.primary))}{cell(b, lambda x: sgn(x, 4), bool(a.primary))}</tr>'
prim = ''.join(crow(c) for c in ['R_CONTRIB - BASE', 'R_CONTRIB - R_RANDOM_DIR']); ctrl = ''.join(crow(c) for c in ['R_RANDOM_DIR - BASE', 'R_RANDOM_DIR_s0 - BASE', 'R_RANDOM_DIR_s1 - BASE', 'R_RANDOM_DIR_s2 - BASE', 'R_CONTRIB_ONLY - BASE']); comp = ''.join(crow(c) for c in ['BASE - ct7', 'R_CONTRIB - ct7', 'R_CONTRIB - token_lsml'])
sb = ''.join(f'<tr><td class="lbl">{N[m]}</td><td>{pct(est(m,"pb","sla","macro8"))}</td><td>{pct(est(m,"pb","f1","macro8"))}</td><td>{num(est(m,"prm","within_auc"))}</td><td>{num(est(m,"prm","prmscore_inner_raw"))}</td><td>{num(est(m,"prm","prmscore_inner_answer_z"))}</td><td>{pct(est(m,"pb","early","macro8"))} / {pct(est(m,"pb","late","macro8"))}</td></tr>' for m in ['BASE'] + ARMS + COMP)
drows = ''.join(f'<tr><td class="lbl">{r.task} / fold {r.fold}</td><td>{r.status}</td><td>{num(r.chosen_eigenvalue,3)}</td><td>{num(r.eigengap,3)}</td><td>{num(r.component_sum,3)}</td><td>{num(r.corr_aux_base_A,3)}</td><td>{num(r.corr_aux_base_H,3)}</td><td>{int(r.active_channels)}</td></tr>' for r in D.itertuples())
load = D.groupby('task')[[f'load_{c}' for c in CH]].mean()
lrows = ''.join(f'<tr><td class="lbl">{c}</td>' + ''.join(f'<td>{sgn(load.loc[t, f"load_{c}"],3)}</td>' for t in ['pb_q4', 'pb_q8', 'prm']) + '</tr>' for c in CH)
stab = ST.groupby('task').cosine.agg(['mean', 'min']).round(3)
rm_pb = RM[RM.benchmark == 'pb'].set_index('arm'); rm_pr = RM[RM.benchmark == 'prm'].set_index('arm')
resc = ''.join(f'<tr><td class="lbl">{N[m]}</td><td>{pct(rm_pb.loc[m].agreement_with_base,0)}</td><td class="pos">{int(rm_pb.loc[m].arm_only)}</td><td class="neg">{int(rm_pb.loc[m].base_only)}</td><td>{int(rm_pb.loc[m].net):+d}</td><td>{int(rm_pb.loc[m].moved_earlier)} / {int(rm_pb.loc[m].moved_later)}</td><td>{sgn(rm_pr.loc[m].mean_delta_auc_vs_base)}</td><td>{int(rm_pr.loc[m].answers_improved)} / {int(rm_pr.loc[m].answers_worsened)}</td></tr>' for m in ['R_CONTRIB', 'R_RANDOM_DIR', 'R_CONTRIB_ONLY'])
cb = Cp.loc[('R_CONTRIB - BASE', 'pb_sla_macro8')]; cr = Cp.loc[('R_CONTRIB - R_RANDOM_DIR', 'pb_sla_macro8')]; cbp = Cp.loc[('R_CONTRIB - BASE', 'prm_within_auc')]; crp = Cp.loc[('R_CONTRIB - R_RANDOM_DIR', 'prm_within_auc')]
def vd(r): return 'תומך' if r.ci_adj_lo > 0 else 'שלילי' if r.ci_adj_hi < 0 else 'לא מכריע'
outcome = 'supported' if all(r.ci_adj_lo > 0 for r in [cb, cr, cbp, crp]) else 'unsupported' if (cb.ci_adj_hi < 0 or cbp.ci_adj_hi < 0) and (cr.ci_adj_hi < 0 or crp.ci_adj_hi < 0) else 'inconclusive / mixed'
html = f'''<title>S3 Contribution Residual · Step 436</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#f6f7f5;--ink:#1c2430;--muted:#5d6874;--rule:#d7dcd9;--card:#eef1ee;--accent:#1f6f8b;--pos:#3f7d4e;--neg:#b5452b;--posbg:#e6f0e8;--negbg:#f6e6e1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}}}
:root[data-theme="dark"]{{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}
body{{background:var(--paper);color:var(--ink);font-family:Heebo,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.6;direction:rtl}}
main{{max-width:840px;margin:0 auto;padding:40px 24px 80px}}
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
td.lbl{{font-family:Heebo,sans-serif;direction:rtl;white-space:normal;min-width:170px}}
td.pos{{color:var(--pos);background:var(--posbg)}} td.neg{{color:var(--neg);background:var(--negbg)}}
.ci{{font-size:11px;color:var(--muted)}}
figure{{margin:16px 0 4px}} figure img{{width:100%;border:1px solid var(--rule);border-radius:3px;background:#fff}}
.note{{font-size:14px;color:var(--muted);margin:6px 0}}
.next{{background:var(--card);padding:16px 20px;border-radius:4px;border-right:3px solid var(--accent)}}
code{{font-family:"IBM Plex Mono",monospace;font-size:13px;direction:ltr;unicode-bidi:embed}}
@media (max-width:600px){{.verdict{{grid-template-columns:1fr}}}}
</style>
<main>
<div class="eyebrow">SSL / pseudo-label / residual plan v1.1 · stage S3 · {RUN_ID} · development data only</div>
<h1>S3: שארית תרומה</h1>
<p class="lede">כל ערוץ תורם חלק לציון הבסיס; מסירים מכל תרומה את מה שכבר מוסבר על ידי הסכום, ומחפשים כיוון "ניטרלי" בשאריות (ערך עצמי הקרוב ביותר ל-1), בהשראת NRM. תוצאה: <b>{outcome}</b>. R_CONTRIB − BASE: {pp(cb.delta)} נקודות ב-ProcessBench ({vd(cb)}), {sgn(cbp.delta)} ב-PRMBench ({vd(cbp)}). מול כיוון אקראי מותאם: {pp(cr.delta)} ({vd(cr)}), {sgn(crp.delta)} ({vd(crp)}).</p>
<div class="verdict">
<div><b>{status["status"]}</b><span>{status["fits"]} התאמות (fold A בלבד), {status["unidentified_directions"]} כיוונים לא מזוהים, {status["fallback_H_answers"]} תשובות fallback; {timing["total_s"]/60:.0f} דקות</span></div>
<div><b>{num(D.chosen_eigenvalue.mean(),3)} / {num(D.eigengap.min(),3)}</b><span>הערך העצמי שנבחר בממוצע / ה-eigengap המינימלי על 15 ההתאמות; יציבות הכיוון בין folds (cosine ממוצע): {", ".join(f"{t} {stab.loc[t,'mean']}" for t in stab.index)}</span></div>
<div><b>{pct(est("R_CONTRIB","pb","sla","macro8"))} / {num(est("R_CONTRIB","prm","within_auc"),3)}</b><span>R_CONTRIB מול BASE {pct(est("BASE","pb","sla","macro8"))} / {num(est("BASE","prm","within_auc"),3)} ומול CT7 {pct(est("ct7","pb","sla","macro8"))} / {num(est("ct7","prm","within_auc"),3)}</span></div>
</div>
<h2>לוח התוצאות</h2>
<figure><img src="{img('FIG1_scoreboard.png')}" alt="scoreboard"></figure>
<div class="tw"><table><thead><tr><th>שיטה</th><th>PB SLA macro8</th><th>PB F1 (gate CT7)</th><th>PRMB within-AUC</th><th>PRMScore raw</th><th>PRMScore z-לכל-תשובה</th><th>החטאה מוקדמת / מאוחרת</th></tr></thead><tbody>{sb}</tbody></table></div>
<h2>ההשוואות הראשיות (משפחה של 4 מבחנים)</h2>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{prim}</tbody></table></div>
<figure><img src="{img('FIG2_contrasts.png')}" alt="contrasts"></figure>
<h3>בקרות ואבחונים</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{ctrl}</tbody></table></div>
<h3>מול העוגנים</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{comp}</tbody></table></div>
<h2>הכיוון שנבחר</h2>
<figure><img src="{img('FIG3_direction.png')}" alt="direction"></figure>
<div class="tw"><table><thead><tr><th>task / fold</th><th>סטטוס</th><th>ערך עצמי</th><th>eigengap</th><th>סכום רכיבים</th><th>corr(aux, BASE) על A</th><th>על H</th><th>ערוצים פעילים</th></tr></thead><tbody>{drows}</tbody></table></div>
<div class="tw"><table><thead><tr><th>ערוץ</th><th>pb_q4</th><th>pb_q8</th><th>prm</th></tr></thead><tbody>{lrows}</tbody></table></div>
<p class="note">טעינות v_neutral, ממוצע על חמישה folds. יציבות בין folds (cosine): {"; ".join(f"{t}: ממוצע {stab.loc[t,'mean']}, מינימום {stab.loc[t,'min']}" for t in stab.index)}. corr(aux, BASE) על A הוא אפס לפי בנייה רק אם הכיוון בתוך המרחב האורתוגונלי; על H הוא הבדיקה האמיתית.</p>
<h2>איפה ההחלטות השתנו</h2>
<div class="tw"><table><thead><tr><th>זרוע</th><th>הסכמה עם BASE</th><th>ניצלו</th><th>נפגעו</th><th>נטו</th><th>זז מוקדם / מאוחר</th><th>PRMB Δ AUC</th><th>השתפרו / הורעו</th></tr></thead><tbody>{resc}</tbody></table></div>
<h2>איך לקרוא את זה</h2>
<div class="next">
<p><b>כלל הפירוש שהוקפא:</b> רווח חייב לעבור גם את הכיוון האקראי המותאם (אותו מרחב, אותו כלל אוריינטציה, אותו מינון). אורתוגונליות על A אינה מבטיחה אורתוגונליות על H, עצמאות שגיאות או מידע סיבתי על reasoning.</p>
<p><b>לפי טבלת ההכרעה:</b> {"שיפור מול BASE וגם מול הכיוון האקראי: מועמד development להמשך, לא promotion." if outcome == "supported" else "לא נמצאה תועלת עקבית בלוקליזציה; הכיוון ה'ניטרלי' אינו טוב מכיוון אקראי, או שהתיקון עצמו מזיק." if outcome == "unsupported" else "תמונה מעורבת או לא מכריעה; אין winner ואין סגירת משפחה."}</p>
<p><b>הכיוון אינו יציב.</b> כלל "הערך העצמי הקרוב ביותר ל-1" בוחר ב-folds שונים וקטורים עצמיים שונים (ערך עצמי {num(D.chosen_eigenvalue.min(),2)} עד {num(D.chosen_eigenvalue.max(),2)}); ה-cosine בין הכיוונים של folds שונים הוא בממוצע {stab.loc["prm","mean"]} ב-PRMBench ו-{stab.loc["pb_q8","mean"]} ב-pb_q8, עם מינימום שלילי. הכיוון עובר את בדיקת הזיהוי הפורמלית (eigengap מעל 1e-6) אבל אין לו זהות עקבית בין folds. גם שלושת הכיוונים האקראיים נבדלים זה מזה בתוצאה ב-PRMBench (seed 1: {sgn(Cp.loc[("R_RANDOM_DIR_s1 - BASE","prm_within_auc")].delta)}; seed 2: {sgn(Cp.loc[("R_RANDOM_DIR_s2 - BASE","prm_within_auc")].delta)}), כלומר התיקון במינון 0.25 רגיש לכיוון שרירותי בסדר גודל דומה לזה של הכיוון ה"ניטרלי".</p>
<p><b>מול CT7:</b> R_CONTRIB {"מתחת" if est("R_CONTRIB","pb","sla","macro8") < est("ct7","pb","sla","macro8") else "מעל"} ל-CT7 ב-ProcessBench ו-{"מתחת" if est("R_CONTRIB","prm","within_auc") < est("ct7","prm","within_auc") else "מעל"} ב-PRMBench.</p>
</div>
<p class="note">קבצים: <code>results/ssl_pseudolabel_residual_v1/S3/{RUN_ID}/</code>. בדיקות: <code>tests/test_ssl_s3.py</code> (4 עוברות: סכום התרומות = BASE, residualizer מסיר את הסכום ומסמן עמודות לא פעילות, כלל בחירת הכיוון והאוריינטציה, מקרי קצה של התיקון).</p>
</main>
'''
(OUT / 'REPORT_HE.html').write_text(html, encoding='utf8')
print('written', len(html) // 1024, 'KB', outcome)
