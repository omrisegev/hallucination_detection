"""S2 figures + Hebrew findings page from the run outputs (no recomputation)."""
from pathlib import Path
import base64, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260923'
OUT = ROOT / 'results/ssl_pseudolabel_residual_v1/S2' / RUN_ID
M = pd.read_csv(OUT / 'METRICS.csv'); C = pd.read_csv(OUT / 'CONTRASTS.csv').drop_duplicates(['contrast_id', 'endpoint']); RD = pd.read_csv(OUT / 'RESIDUAL_DIAGNOSTICS.csv'); RM = pd.read_csv(OUT / 'RESCUE_MATRIX.csv')
oa = pd.read_csv(OUT / 'OOF_ANSWERS.csv'); status = json.loads((OUT / 'RUN_STATUS.json').read_text(encoding='utf8')); timing = json.loads((OUT / 'TIMING.json').read_text(encoding='utf8'))
fits = [json.loads(l) for l in (OUT / 'FIT_MANIFEST.jsonl').read_text(encoding='utf8').splitlines()]
ARMS = ['R_TEMP', 'R_ZERO', 'R_NORESET', 'R_ONLY', 'R_ABS']
COMP = ['ct7', 'token_lsml', 'token_equal', 'evidence30__all__plain__equal', 'evidence__all__position__equal', 'evidence__all__position2__equal']
N = {'BASE': 'BASE (teacher, ללא תיקון)', 'R_TEMP': 'R_TEMP (שארית Ridge)', 'R_ZERO': 'R_ZERO (חיזוי אפס, בקרה)', 'R_NORESET': 'R_NORESET (ממוצע רץ, בקרה)', 'R_ONLY': 'R_ONLY (שארית בלבד, אבחון)', 'R_ABS': 'R_ABS (|שארית|, אבחון)',
     'ct7': 'CT7 (עוגן קפוא)', 'token_lsml': 'token L-SML', 'token_equal': 'token equal', 'evidence30__all__plain__equal': 'Step432 top30 plain', 'evidence__all__position__equal': 'Step432 top5 position', 'evidence__all__position2__equal': 'Step432 top5 position iter2'}
def est(m, bench, metric, stratum='all'):
    r = M[(M.method == m) & (M.benchmark == bench) & (M.metric == metric) & (M.stratum == stratum)]
    return float(r.estimate.iloc[0]) if len(r) else np.nan
def num(x, d=4): return '—' if pd.isna(x) else f'{x:.{d}f}'
def pct(x, d=1): return '—' if pd.isna(x) else f'{100*x:.{d}f}%'
def sgn(x, d=4): return f'{x:+.{d}f}'
def pp(x): return f'{100*x:+.2f}'
pb_cells = sorted(M[(M.benchmark == 'pb') & M.stratum.str.startswith('pb_')].stratum.unique())

# Fig 1 scoreboard
fig, ax = plt.subplots(figsize=(8.5, 6))
for m in ['BASE'] + ARMS + COMP:
    x, y = est(m, 'prm', 'within_auc'), 100 * est(m, 'pb', 'sla', 'macro8'); anchor = m in COMP
    ax.scatter(x, y, c='#444' if anchor else '#1f77b4' if m == 'BASE' else '#d62728' if m == 'R_TEMP' else '#999', marker='D' if anchor else 'o', s=80 if anchor else 60, zorder=3); ax.annotate(N[m], (x, y), fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.set_xlabel('PRMBench within-answer AUROC'); ax.set_ylabel('ProcessBench SLA macro8, %'); ax.set_title('S2: prediction residual as a fixed-dose correction of the teacher'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG1_scoreboard.png', dpi=150); plt.close(fig)
# Fig 2 contrasts
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
want = ['R_TEMP - BASE', 'R_TEMP - R_ZERO', 'R_ZERO - BASE', 'R_NORESET - BASE', 'R_TEMP - R_NORESET', 'R_ABS - BASE', 'R_ONLY - BASE', 'R_TEMP - ct7', 'BASE - ct7']
for ax, ep, scale, title in [(axes[0], 'pb_sla_macro8', 100, 'ProcessBench SLA, pp'), (axes[1], 'prm_within_auc', 1, 'PRMBench within-AUC')]:
    d = C[(C.endpoint == ep) & C.contrast_id.isin(want)].set_index('contrast_id').reindex(want).reset_index()
    y = np.arange(len(d)); ax.errorbar(scale * d.delta, y, xerr=[scale * (d.delta - d.ci95_lo), scale * (d.ci95_hi - d.delta)], fmt='o', color='#1f6f8b', capsize=3)
    for i, r in d.iterrows():
        if r.primary: ax.errorbar([scale * r.delta], [i], xerr=[[scale * (r.delta - r.ci_adj_lo)], [scale * (r.ci_adj_hi - r.delta)]], fmt='none', ecolor='#b5452b', capsize=5, lw=.8); ax.text(scale * r.ci95_hi, i, '  primary', va='center', fontsize=7, color='#b5452b')
    ax.axvline(0, color='k', lw=.8); ax.set_yticks(y); ax.set_yticklabels(d.contrast_id, fontsize=8); ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x', alpha=.3)
fig.suptitle('Paired source-group bootstrap, 100,000 draws; blue = 95% CI, red = Bonferroni (K=4) on primary contrasts', fontsize=9); fig.tight_layout(); fig.savefig(OUT / 'FIG2_contrasts.png', dpi=150); plt.close(fig)
# Fig 3 residual MSE per channel (ridge vs zero vs noreset), t>=16, PB q4 task as representative + prm
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, task in zip(axes, ['pb_q4', 'pb_q8', 'prm']):
    d = RD[(RD.task == task) & (RD.tokens == 't>=16')].set_index('channel'); x = np.arange(len(d))
    ax.bar(x - .27, d.mse_zero, .27, label='zero predictor (= var of z)', color='#999'); ax.bar(x, d.mse_noreset, .27, label='running mean', color='#8fbccb'); ax.bar(x + .27, d.mse_ridge, .27, label='ridge (16 lags)', color='#1f6f8b')
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=60, fontsize=7); ax.set_title(f'{task}: token MSE, t>=16'); ax.grid(axis='y', alpha=.3)
axes[0].set_ylabel('MSE of z_token'); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / 'FIG3_residual_mse.png', dpi=150); plt.close(fig)
# Fig 4 per cell + depth
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(len(pb_cells)); w = .18
for j, m in enumerate(['R_TEMP', 'R_ZERO', 'R_NORESET', 'R_ABS']):
    axes[0].bar(x + (j - 1.5) * w, [100 * (est(m, 'pb', 'sla', c) - est('BASE', 'pb', 'sla', c)) for c in pb_cells], w, label=m, color=['#d62728', '#999', '#8fbccb', '#777'][j])
axes[0].axhline(0, color='k', lw=.8); axes[0].set_xticks(x); axes[0].set_xticklabels([c.replace('pb_', '') for c in pb_cells], fontsize=7); axes[0].set_ylabel('SLA change vs BASE, pp'); axes[0].set_title('ProcessBench per cell'); axes[0].legend(fontsize=7); axes[0].grid(axis='y', alpha=.3)
for m in ['ct7', 'BASE', 'R_TEMP', 'R_ZERO', 'R_NORESET']:
    axes[1].plot(['2-5', '6-10', '11+'], [100 * est(m, 'pb', 'sla', f'depth={b}') for b in ['2-5', '6-10', '11+']], marker='o', label=N[m])
axes[1].set_title('ProcessBench SLA by chain depth'); axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG4_cells_depth.png', dpi=150); plt.close(fig)

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
prim = ''.join(crow(c) for c in ['R_TEMP - BASE', 'R_TEMP - R_ZERO']); ctrl = ''.join(crow(c) for c in ['R_ZERO - BASE', 'R_NORESET - BASE', 'R_TEMP - R_NORESET', 'R_ABS - BASE', 'R_ABS - R_TEMP', 'R_ONLY - BASE']); comp = ''.join(crow(c) for c in ['BASE - ct7', 'R_TEMP - ct7', 'R_TEMP - token_lsml', 'R_TEMP - evidence30__all__plain__equal'])
sb = ''.join(f'<tr><td class="lbl">{N[m]}</td><td>{pct(est(m,"pb","sla","macro8"))}</td><td>{pct(est(m,"pb","f1","macro8"))}</td><td>{num(est(m,"prm","within_auc"))}</td><td>{num(est(m,"prm","within_auc","kind=single"))}</td><td>{num(est(m,"prm","within_auc","kind=multi"))}</td><td>{num(est(m,"prm","prmscore_inner_raw"))}</td><td>{num(est(m,"prm","prmscore_inner_answer_z"))}</td><td>{num(est(m,"prm","prmscore_q80_raw"))}</td></tr>' for m in ['BASE'] + ARMS + COMP)
mse = RD[RD.tokens == 't>=16'].groupby('task')[['mse_zero', 'mse_noreset', 'mse_ridge']].mean(); mse_early = RD[RD.tokens == 't<16'].groupby('task')[['mse_zero', 'mse_noreset', 'mse_ridge']].mean()
mse_rows = ''.join(f'<tr><td class="lbl">{t}</td><td>{num(mse_early.loc[t,"mse_zero"],3)} / {num(mse.loc[t,"mse_zero"],3)}</td><td>{num(mse_early.loc[t,"mse_noreset"],3)} / {num(mse.loc[t,"mse_noreset"],3)}</td><td>{num(mse_early.loc[t,"mse_ridge"],3)} / {num(mse.loc[t,"mse_ridge"],3)}</td><td>{pct(1 - mse.loc[t,"mse_ridge"]/mse.loc[t,"mse_zero"])}</td></tr>' for t in ['pb_q4', 'pb_q8', 'prm'])
ch = RD[(RD.tokens == 't>=16')].groupby('channel')[['mse_zero', 'mse_ridge', 'corr_pred_z', 'corr_resid_z']].mean().reindex([c for c in RD.channel.unique()])
ch_rows = ''.join(f'<tr><td class="lbl">{c}</td><td>{pct(1 - r.mse_ridge/r.mse_zero)}</td><td>{num(r.corr_pred_z,3)}</td><td>{num(r.corr_resid_z,3)}</td></tr>' for c, r in ch.iterrows())
rm_pb = RM[RM.benchmark == 'pb'].set_index('arm'); rm_pr = RM[RM.benchmark == 'prm'].set_index('arm')
resc = ''.join(f'<tr><td class="lbl">{N[m]}</td><td>{pct(rm_pb.loc[m].agreement_with_base,0)}</td><td class="pos">{int(rm_pb.loc[m].arm_only)}</td><td class="neg">{int(rm_pb.loc[m].base_only)}</td><td>{int(rm_pb.loc[m].net):+d}</td><td>{int(rm_pb.loc[m].moved_earlier)} / {int(rm_pb.loc[m].moved_later)}</td><td>{sgn(rm_pr.loc[m].mean_delta_auc_vs_base)}</td><td>{int(rm_pr.loc[m].answers_improved)} / {int(rm_pr.loc[m].answers_worsened)}</td></tr>' for m in ARMS)
tb = Cp.loc[('R_TEMP - BASE', 'pb_sla_macro8')]; tz = Cp.loc[('R_TEMP - R_ZERO', 'pb_sla_macro8')]; tbp = Cp.loc[('R_TEMP - BASE', 'prm_within_auc')]; tzp = Cp.loc[('R_TEMP - R_ZERO', 'prm_within_auc')]
def vd(r): return 'תומך' if r.ci_adj_lo > 0 else 'שלילי' if r.ci_adj_hi < 0 else 'לא מכריע'
outcome = 'supported' if tb.ci_adj_lo > 0 and tz.ci_adj_lo > 0 and tbp.ci_adj_lo > 0 and tzp.ci_adj_lo > 0 else 'unsupported' if (tb.ci_adj_hi < 0 or tbp.ci_adj_hi < 0) and (tz.ci_adj_hi < 0 or tzp.ci_adj_hi < 0) else 'inconclusive / mixed'
train_mse = np.mean([f['train_mse'] for f in fits]); zero_mse = np.mean([f['train_mse_zero_predictor'] for f in fits])
html = f'''<title>S2 Prediction Residual · Step 435</title>
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
figcaption{{font-size:13px;color:var(--muted);margin-top:6px}}
.note{{font-size:14px;color:var(--muted);margin:6px 0}}
.next{{background:var(--card);padding:16px 20px;border-radius:4px;border-right:3px solid var(--accent)}}
code{{font-family:"IBM Plex Mono",monospace;font-size:13px;direction:ltr;unicode-bidi:embed}}
@media (max-width:600px){{.verdict{{grid-template-columns:1fr}}}}
</style>
<main>
<div class="eyebrow">SSL / pseudo-label / residual plan v1.1 · stage S2 · {RUN_ID} · development data only</div>
<h1>S2: שארית חיזוי בזמן</h1>
<p class="lede">Ridge חוזה כל טוקן מ-16 הקודמים; ההפתעה (השארית) נהפכת לתיקון קבוע במינון 0.25 על ציון ה-teacher. תוצאה: <b>{outcome}</b>. R_TEMP − BASE: {pp(tb.delta)} נקודות ב-ProcessBench ({vd(tb)}), {sgn(tbp.delta)} ב-PRMBench ({vd(tbp)}). R_TEMP − R_ZERO (אותו תיקון עם חיזוי אפס): {pp(tz.delta)} ({vd(tz)}), {sgn(tzp.delta)} ({vd(tzp)}).</p>
<div class="verdict">
<div><b>{status["status"]}</b><span>{status["fits"]} התאמות Ridge (16,384 טוקנים כל אחת, מ-fold A בלבד), {status["failures"]} כשלים; {timing["total_s"]/60:.0f} דקות</span></div>
<div><b>{pct(1 - train_mse/zero_mse)}</b><span>כמה מהשונות של הטוקן הבא ה-Ridge מסביר בממוצע (MSE אימון {num(train_mse,3)} מול {num(zero_mse,3)} לחיזוי אפס). המשימה העצמית נלמדת; השאלה היא אם זה עוזר ללוקליזציה</span></div>
<div><b>{pct(est("R_TEMP","pb","sla","macro8"))} / {num(est("R_TEMP","prm","within_auc"),3)}</b><span>R_TEMP מול BASE {pct(est("BASE","pb","sla","macro8"))} / {num(est("BASE","prm","within_auc"),3)} ומול CT7 {pct(est("ct7","pb","sla","macro8"))} / {num(est("ct7","prm","within_auc"),3)}</span></div>
</div>

<h2>לוח התוצאות</h2>
<figure><img src="{img('FIG1_scoreboard.png')}" alt="scoreboard"></figure>
<div class="tw"><table><thead><tr><th>שיטה</th><th>PB SLA macro8</th><th>PB F1 (gate CT7)</th><th>PRMB within-AUC</th><th>שגיאה אחת</th><th>כמה שגיאות</th><th>PRMScore raw</th><th>PRMScore z-לכל-תשובה</th><th>PRMScore q80 raw</th></tr></thead><tbody>{sb}</tbody></table></div>

<h2>ההשוואות הראשיות (משפחה של 4 מבחנים)</h2>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{prim}</tbody></table></div>
<figure><img src="{img('FIG2_contrasts.png')}" alt="contrasts"></figure>
<h3>בקרות ואבחונים</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{ctrl}</tbody></table></div>
<h3>מול העוגנים</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{comp}</tbody></table></div>

<h2>האם המשימה העצמית נלמדה?</h2>
<figure><img src="{img('FIG3_residual_mse.png')}" alt="mse"></figure>
<div class="tw"><table><thead><tr><th>task</th><th>MSE חיזוי אפס (t&lt;16 / t≥16)</th><th>ממוצע רץ</th><th>Ridge</th><th>שונות מוסברת (t≥16)</th></tr></thead><tbody>{mse_rows}</tbody></table></div>
<div class="tw"><table><thead><tr><th>ערוץ</th><th>שונות מוסברת (t≥16)</th><th>corr(חיזוי, z)</th><th>corr(שארית, z)</th></tr></thead><tbody>{ch_rows}</tbody></table></div>
<p class="note">מדדי הטוקנים חושבו על טוקני H בלבד (מודל fold k על fold k), ממוצע על המשימות. corr(שארית, z) גבוה אומר שהשארית עדיין כמעט זהה לערך הנוכחי, כלומר התיקון קרוב ל-R_ZERO.</p>

<h3>שני ממצאים אבחוניים שמסבירים את התוצאה</h3>
<p><b>1. הנרמול הקפוא מפוצץ שלושה ערוצים.</b> חוק הנרמול של הבנק (חציון / IQR) נותן לערוצים עם זנב כבד ו-IQR זעיר ערכי z ענקיים: שונות של {RD[(RD.tokens=="t>=16")&(RD.channel=="top50_js")].var_z.mean():.1e} ל-top50_js, {RD[(RD.tokens=="t>=16")&(RD.channel=="chosen_surprisal")].var_z.mean():.1e} ל-chosen_surprisal, {RD[(RD.tokens=="t>=16")&(RD.channel=="true_tail50")].var_z.mean():.1e} ל-true_tail50, לעומת כ-1 לשאר. ה-readout של הפרוטוקול (ממוצע ערוצים של top5 השארית, ללא סטנדרטיזציה לכל ערוץ) נשלט לכן על ידי ערוץ אחד או שניים, וה-Ridge המשותף מתאים בעיקר אותם. על טוקני H, שגיאת ה-Ridge בערוצים האלה גרועה מחיזוי אפס (top50_js: {RD[(RD.tokens=="t>=16")&(RD.channel=="top50_js")].mse_ridge.mean():.1e} מול {RD[(RD.tokens=="t>=16")&(RD.channel=="top50_js")].var_z.mean():.1e}); רק dominant_freq16 נחזה היטב (corr {RD[(RD.tokens=="t>=16")&(RD.channel=="dominant_freq16")].corr_pred_z.mean():.2f}). המשימה העצמית נלמדה באימון אך לא הכלילה ברוב הערוצים. BASE אינו סובל מזה כי הוא מסטנדרט כל ערוץ לכל תשובה לפני הממוצע.</p>
<p><b>2. התיקון מזיז שיאים מוקדם.</b> R_TEMP מזיז את השיא מוקדם יותר ב-{int(rm_pb.loc["R_TEMP"].moved_earlier)} תשובות ומאוחר יותר ב-{int(rm_pb.loc["R_TEMP"].moved_later)} בלבד; שיעור ההחטאה המוקדמת עולה מ-{pct(est("BASE","pb","early","macro8"))} ל-{pct(est("R_TEMP","pb","early","macro8"))}. טוקנים בתחילת תשובה חסרים היסטוריה ולכן שאריתם גדולה: התיקון מוסיף עודף-התחלה, בדיוק ההטיה ש-Step 430 תיעד ב-CT7. R_ZERO, שאינו תלוי בהיסטוריה, כמעט זהה ל-BASE.</p>

<h2>איפה ההחלטות השתנו</h2>
<figure><img src="{img('FIG4_cells_depth.png')}" alt="cells"></figure>
<div class="tw"><table><thead><tr><th>זרוע</th><th>הסכמה עם BASE</th><th>ניצלו</th><th>נפגעו</th><th>נטו</th><th>זז מוקדם / מאוחר</th><th>PRMB Δ AUC</th><th>השתפרו / הורעו</th></tr></thead><tbody>{resc}</tbody></table></div>

<h2>איך לקרוא את זה</h2>
<div class="next">
<p><b>כלל הפירוש שהוקפא:</b> אין לייחס הצלחה לחיזוי בזמן אם R_TEMP − R_ZERO אינו נתמך. R_ZERO מקבל בדיוק את אותו readout ואותו תיקון, עם חיזוי אפס; ההבדל ביניהם הוא כל מה שהחיזוי מוסיף.</p>
<p><b>MSE מול לוקליזציה:</b> Ridge מסביר {pct(1 - train_mse/zero_mse)} מהשונות של הטוקן הבא. {"אבל" if outcome != "supported" else "ובנוסף"} ההשפעה על הלוקליזציה היא {pp(tb.delta)} נקודות ב-PB ו-{sgn(tbp.delta)} ב-PRMB. {"לפי טבלת ההכרעה: המשימה העצמית נלמדה; לא נמצאה תועלת עקבית בלוקליזציה." if outcome != "supported" else "לפי טבלת ההכרעה: מועמד development להמשך, לא promotion."}</p>
<p><b>ההחלטה האחת לשלב הבא:</b> הפרוטוקול הזה סגור כפי שנכתב: unsupported. הניגוד היחיד שיכול להכריע אם יש בכלל מידע בשארית הוא גרסה S2 v1.1 שבה top5 השארית מסטונדרט לכל ערוץ ולכל תשובה לפני ממוצע הערוצים (בדיוק כמו BASE), עם R_ZERO מותאם. זהו שינוי פרוטוקול שדורש הקפאה חדשה, לא ריצה חוזרת.</p>
<p><b>מול CT7:</b> R_TEMP {"מתחת" if est("R_TEMP","pb","sla","macro8") < est("ct7","pb","sla","macro8") else "מעל"} ל-CT7 ב-ProcessBench ו-{"מתחת" if est("R_TEMP","prm","within_auc") < est("ct7","prm","within_auc") else "מעל"} ב-PRMBench.</p>
</div>
<p class="note">קבצים: <code>results/ssl_pseudolabel_residual_v1/S2/{RUN_ID}/</code>: PROTOCOL.json (בתיקיית השלב, הוקפא לפני הריצה), INPUT_MANIFEST, CODE_MANIFEST + SOURCE_SNAPSHOT, SPLITS, TRAINING_SAMPLES.npz, FIT_MANIFEST.jsonl, OOF_STEP_SCORES.npz, OOF_ANSWERS.csv, EVAL_JOINED.csv, METRICS.csv, CONTRASTS.csv, BOOTSTRAP_DELTAS.npz, RESIDUAL_DIAGNOSTICS.csv, RESCUE_MATRIX.csv, FAILURES.csv, TIMING.json, RUN_STATUS.json. בדיקות: <code>tests/test_ssl_s2.py</code> (5 עוברות: סיבתיות ה-lag, Ridge סגור מול lstsq, מינון 0 = BASE, חיזוי אפס = ממוצע פרופיל top5 הקפוא, דגימה היררכית).</p>
</main>
'''
(OUT / 'REPORT_HE.html').write_text(html, encoding='utf8')
print('written', len(html) // 1024, 'KB', outcome)
