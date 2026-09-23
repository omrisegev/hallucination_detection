"""S1 figures + Hebrew findings page from the run outputs (no recomputation)."""
from pathlib import Path
import base64, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260923'
OUT = ROOT / 'results/ssl_pseudolabel_residual_v1/S1' / RUN_ID
M = pd.read_csv(OUT / 'METRICS.csv'); C = pd.read_csv(OUT / 'CONTRASTS.csv').drop_duplicates(['contrast_id', 'endpoint']); D = pd.read_csv(OUT / 'TEACHER_DIAGNOSTICS.csv'); Q = pd.read_csv(OUT / 'CONFIDENCE_QUINTILES.csv')
cov = pd.read_csv(OUT / 'TRAINING_COVERAGE.csv'); oa = pd.read_csv(OUT / 'OOF_ANSWERS.csv'); status = json.loads((OUT / 'RUN_STATUS.json').read_text(encoding='utf8')); timing = json.loads((OUT / 'TIMING.json').read_text(encoding='utf8'))
fits = [json.loads(l) for l in (OUT / 'FIT_MANIFEST.jsonl').read_text(encoding='utf8').splitlines()]
ARMS = ['P_HARD', 'P_SOFT', 'P_AGREE', 'P_RANDOM', 'P_POSITION_LENGTH', 'P_SOFT_COVERAGE_MATCH']
COMP = ['ct7', 'token_lsml', 'token_equal', 'evidence30__all__plain__equal', 'evidence__all__position__equal', 'evidence__all__position2__equal']
N = {'BASE': 'BASE (teacher, ללא fit)', 'P_HARD': 'P_HARD (one-hot)', 'P_SOFT': 'P_SOFT (התפלגות רכה)', 'P_AGREE': 'P_AGREE (הסכמה + הימנעות)', 'P_RANDOM': 'P_RANDOM (בקרה)', 'P_POSITION_LENGTH': 'P_POSITION_LENGTH (בקרה)', 'P_SOFT_COVERAGE_MATCH': 'P_SOFT coverage-matched (בקרה)',
     'ct7': 'CT7 (עוגן קפוא)', 'token_lsml': 'token L-SML', 'token_equal': 'token equal', 'evidence30__all__plain__equal': 'Step432 top30 plain', 'evidence__all__position__equal': 'Step432 top5 position', 'evidence__all__position2__equal': 'Step432 top5 position iter2'}
def est(m, bench, metric, stratum='all'):
    r = M[(M.method == m) & (M.benchmark == bench) & (M.metric == metric) & (M.stratum == stratum)]
    return float(r.estimate.iloc[0]) if len(r) else np.nan
def num(x, d=4): return '—' if pd.isna(x) else f'{x:.{d}f}'
def pct(x, d=1): return '—' if pd.isna(x) else f'{100*x:.{d}f}%'
def sgn(x, d=4): return f'{x:+.{d}f}'
def pp(x): return f'{100*x:+.2f}'
pb_cells = sorted(M[(M.benchmark == 'pb') & M.stratum.str.startswith('pb_')].stratum.unique())

# ---- Fig 1: scoreboard scatter
fig, ax = plt.subplots(figsize=(8.5, 6))
col = {'BASE': '#1f77b4', 'P_HARD': '#d62728', 'P_SOFT': '#d62728', 'P_AGREE': '#d62728', 'P_RANDOM': '#999', 'P_POSITION_LENGTH': '#999', 'P_SOFT_COVERAGE_MATCH': '#999'}
for m in ['BASE'] + ARMS + COMP:
    x, y = est(m, 'prm', 'within_auc'), 100 * est(m, 'pb', 'sla', 'macro8'); anchor = m in COMP
    ax.scatter(x, y, c=col.get(m, '#444'), marker='D' if anchor else 'o', s=80 if anchor else 60, zorder=3); ax.annotate(N[m], (x, y), fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.set_xlabel('PRMBench within-answer AUROC'); ax.set_ylabel('ProcessBench SLA macro8, %'); ax.set_title('S1: one linear student, six ways to build its target'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG1_scoreboard.png', dpi=150); plt.close(fig)

# ---- Fig 2: paired improvement per cell (arm - BASE), PB SLA
fig, ax = plt.subplots(figsize=(10, 4.5)); x = np.arange(len(pb_cells)); w = .13
for j, m in enumerate(ARMS):
    ax.bar(x + (j - 2.5) * w, [100 * (est(m, 'pb', 'sla', c) - est('BASE', 'pb', 'sla', c)) for c in pb_cells], w, label=m, color=['#d62728', '#e377c2', '#9467bd', '#bbb', '#999', '#777'][j])
ax.axhline(0, color='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([c.replace('pb_', '') for c in pb_cells], fontsize=8); ax.set_ylabel('SLA change vs teacher, pp'); ax.set_title('ProcessBench: student minus teacher, per cell'); ax.legend(fontsize=7, ncol=3); ax.grid(axis='y', alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG2_per_cell.png', dpi=150); plt.close(fig)

# ---- Fig 3: primary contrasts with CIs
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, ep, scale, title in [(axes[0], 'pb_sla_macro8', 100, 'ProcessBench SLA, pp'), (axes[1], 'prm_within_auc', 1, 'PRMBench within-AUC')]:
    d = C[(C.endpoint == ep) & (C.contrast_id.isin(['P_SOFT - P_HARD', 'P_AGREE - P_SOFT', 'P_HARD - BASE', 'P_SOFT - BASE', 'P_AGREE - BASE', 'P_SOFT - P_RANDOM', 'P_SOFT - P_POSITION_LENGTH', 'P_AGREE - P_SOFT_COVERAGE_MATCH', 'P_SOFT - ct7', 'BASE - ct7']))].reset_index(drop=True)
    y = np.arange(len(d)); ax.errorbar(scale * d.delta, y, xerr=[scale * (d.delta - d.ci95_lo), scale * (d.ci95_hi - d.delta)], fmt='o', color='#1f6f8b', capsize=3)
    for i, r in d.iterrows():
        if r.primary: ax.errorbar([scale * r.delta], [i], xerr=[[scale * (r.delta - r.ci_adj_lo)], [scale * (r.ci_adj_hi - r.delta)]], fmt='none', ecolor='#b5452b', capsize=5, lw=.8); ax.text(scale * r.ci95_hi, i, '  primary', va='center', fontsize=7, color='#b5452b')
    ax.axvline(0, color='k', lw=.8); ax.set_yticks(y); ax.set_yticklabels(d.contrast_id, fontsize=8); ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x', alpha=.3)
fig.suptitle('Paired source-group bootstrap, 100,000 draws; blue = 95% CI, red = Bonferroni (K=4) interval on primary contrasts', fontsize=9); fig.tight_layout(); fig.savefig(OUT / 'FIG3_contrasts.png', dpi=150); plt.close(fig)

# ---- Fig 4: SLA by depth + PRMB per-answer AUC delta histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for m in ['ct7', 'BASE', 'P_HARD', 'P_SOFT', 'P_AGREE', 'P_POSITION_LENGTH']:
    axes[0].plot(['2-5', '6-10', '11+'], [100 * est(m, 'pb', 'sla', f'depth={b}') for b in ['2-5', '6-10', '11+']], marker='o', label=N[m])
axes[0].set_title('ProcessBench SLA by chain depth'); axes[0].set_ylabel('SLA, %'); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
d = (oa['P_SOFT__within_auc'] - oa['BASE__within_auc']).dropna(); d2 = (oa['P_AGREE__within_auc'] - oa['BASE__within_auc']).dropna()
axes[1].hist(d, bins=60, alpha=.6, label=f'P_SOFT - teacher (mean {d.mean():+.4f})'); axes[1].hist(d2, bins=60, alpha=.6, label=f'P_AGREE - teacher (mean {d2.mean():+.4f})')
axes[1].axvline(0, color='k', lw=.8); axes[1].set_title('PRMBench per-answer within-AUC change vs teacher'); axes[1].legend(fontsize=8); axes[1].set_yscale('log')
fig.tight_layout(); fig.savefig(OUT / 'FIG4_depth_and_prm_delta.png', dpi=150); plt.close(fig)

# ---- Fig 5: confidence quintiles (coverage-accuracy) + rescue matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for c in ['teacher_top1', 'P_HARD_top1', 'P_SOFT_top1', 'P_AGREE_top1', 'P_POSITION_LENGTH_top1']:
    axes[0].plot(Q.index + 1 if 'conf_quintile' not in Q.columns else Q.conf_quintile, 100 * Q[c], marker='o', label=c.replace('_top1', ''))
axes[0].set_xlabel('teacher confidence quintile (edges fixed on B)'); axes[0].set_ylabel('exact first-error hit, %'); axes[0].set_title('Accuracy by teacher confidence (erroneous PB answers)'); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
dd = D[(D.benchmark == 'pb') & D.stratum.str.startswith('student=')].copy(); dd['arm'] = dd.stratum.str.replace('student=', '')
x = np.arange(len(dd)); axes[1].bar(x - .2, dd.student_only, .4, color='#3f7d4e', label='rescued (teacher wrong, student right)'); axes[1].bar(x + .2, -dd.teacher_only, .4, color='#b5452b', label='damaged (teacher right, student wrong)')
for i, r in enumerate(dd.itertuples()): axes[1].text(i, max(r.student_only, 0) + 10, f'net {int(r.net):+d}\nagree {100*r.agreement_with_teacher:.0f}%', ha='center', fontsize=7)
axes[1].set_xticks(x); axes[1].set_xticklabels(dd.arm, fontsize=7, rotation=20); axes[1].axhline(0, color='k', lw=.8); axes[1].set_title('Teacher/student rescue matrix (4,442 erroneous PB answers)'); axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / 'FIG5_confidence_and_rescue.png', dpi=150); plt.close(fig)

# ---- page
def img(name): return 'data:image/png;base64,' + base64.b64encode((OUT / name).read_bytes()).decode()
Cp = C.set_index(['contrast_id', 'endpoint'])
def crow(cid):
    a = Cp.loc[(cid, 'pb_sla_macro8')]; b = Cp.loc[(cid, 'prm_within_auc')]
    def cell(r, f, adj):
        out = lo > 0 or hi < 0 if (lo := r.ci95_lo) is not None and (hi := r.ci95_hi) is not None else False
        cls = ' class="pos"' if out and r.delta > 0 else ' class="neg"' if out else ''
        s = f'<td{cls}>{f(r.delta)}<br><span class="ci">[{f(r.ci95_lo)}, {f(r.ci95_hi)}]</span>'
        if adj and not pd.isna(r.ci_adj_lo): s += f'<br><span class="ci">Bonf. [{f(r.ci_adj_lo)}, {f(r.ci_adj_hi)}]</span>'
        return s + '</td>'
    return f'<tr><td class="lbl">{cid}{" · <b>ראשי</b>" if bool(a.primary) else ""}</td>{cell(a, pp, bool(a.primary))}{cell(b, lambda x: sgn(x, 4), bool(a.primary))}</tr>'
prim_rows = ''.join(crow(c) for c in ['P_SOFT - P_HARD', 'P_AGREE - P_SOFT'])
ctrl_rows = ''.join(crow(c) for c in ['P_HARD - BASE', 'P_SOFT - BASE', 'P_AGREE - BASE', 'P_SOFT - P_RANDOM', 'P_SOFT - P_POSITION_LENGTH', 'P_POSITION_LENGTH - BASE', 'P_AGREE - P_SOFT_COVERAGE_MATCH', 'P_AGREE - P_HARD'])
comp_rows = ''.join(crow(c) for c in ['BASE - ct7', 'P_SOFT - ct7', 'P_AGREE - ct7', 'P_SOFT - token_lsml', 'P_SOFT - evidence__all__position__equal', 'BASE - evidence30__all__plain__equal'])
sb = ''.join(f'<tr><td class="lbl">{N[m]}</td><td>{pct(est(m,"pb","sla","macro8"))}</td><td>{pct(est(m,"pb","f1","macro8"))}</td><td>{num(est(m,"prm","within_auc"))}</td><td>{num(est(m,"prm","within_auc","kind=single"))}</td><td>{num(est(m,"prm","within_auc","kind=multi"))}</td><td>{num(est(m,"prm","prmscore_inner_raw"))}</td><td>{num(est(m,"prm","prmscore_inner_answer_z"))}</td><td>{num(est(m,"prm","prmscore_q80_raw"))}</td></tr>' for m in ['BASE'] + ARMS + COMP)
tcov = cov.groupby('arm').agg(trainable=('trainable_fraction', 'mean'), steps=('selected_steps', 'mean')).reindex(ARMS)
cov_rows = ''.join(f'<tr><td class="lbl">{m}</td><td>{pct(cov[(cov.arm==m)&(cov.task!="prm")].trainable_fraction.mean())}</td><td>{pct(cov[(cov.arm==m)&(cov.task=="prm")].selected_steps.sum()/cov[(cov.arm==m)&(cov.task=="prm")].total_steps.sum()) if m in ("P_AGREE","P_SOFT_COVERAGE_MATCH") else "100%"}</td></tr>' for m in ARMS)
tp = D[(D.benchmark == 'pb') & (D.stratum == 'all_pb')].iloc[0]; tq = D[(D.benchmark == 'prm') & (D.stratum == 'all_prm')].iloc[0]
rows_res = ''.join(f'<tr><td class="lbl">{r.arm}</td><td>{pct(r.agreement_with_teacher,0)}</td><td class="pos">{int(r.student_only)}</td><td class="neg">{int(r.teacher_only)}</td><td>{int(r.net):+d}</td><td>{pct(r.student_top1_kept,1)} / {pct(r.teacher_top1_kept,1)}</td><td>{pct(r.student_top1_abstained,1)} / {pct(r.teacher_top1_abstained,1)}</td></tr>' for r in dd.itertuples())
prm_rows = ''.join(f'<tr><td class="lbl">{r.stratum.replace("student=","")}</td><td>{sgn(r.mean_delta_auc_vs_teacher)}</td><td class="pos">{int(r.answers_improved)}</td><td class="neg">{int(r.answers_worsened)}</td><td>{int(r.pairs_corrected - r.pairs_destroyed):+,d}</td></tr>' for r in D[(D.benchmark == 'prm') & D.stratum.str.startswith('student=')].itertuples())
nonconv = sum(not f['converged'] for f in fits)
soft_hard = Cp.loc[('P_SOFT - P_HARD', 'pb_sla_macro8')]; agree_soft = Cp.loc[('P_AGREE - P_SOFT', 'pb_sla_macro8')]; soft_hard_p = Cp.loc[('P_SOFT - P_HARD', 'prm_within_auc')]; agree_soft_p = Cp.loc[('P_AGREE - P_SOFT', 'prm_within_auc')]
def verdict(r): return 'תומך' if r.ci_adj_lo > 0 else 'שלילי' if r.ci_adj_hi < 0 else 'לא מכריע'
outcome = 'supported' if (soft_hard.ci_adj_lo > 0 and soft_hard_p.ci_adj_lo > 0) or (agree_soft.ci_adj_lo > 0 and agree_soft_p.ci_adj_lo > 0) else 'unsupported' if (soft_hard.ci_adj_hi < 0 or agree_soft.ci_adj_hi < 0) and (soft_hard_p.ci_adj_hi < 0 or agree_soft_p.ci_adj_hi < 0) else 'inconclusive / mixed'
html = f'''<title>S1 Pseudo-Label Targets · Step 434</title>
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
ul{{padding-right:20px;margin:8px 0}} li{{margin:4px 0}}
code{{font-family:"IBM Plex Mono",monospace;font-size:13px;direction:ltr;unicode-bidi:embed}}
@media (max-width:600px){{.verdict{{grid-template-columns:1fr}}}}
</style>
<main>
<div class="eyebrow">SSL / pseudo-label / residual plan v1.1 · stage S1 · {RUN_ID} · development data only</div>
<h1>S1: שלוש דרכים לבנות פסאודו-תווית</h1>
<p class="lede">אותו student ליניארי (11 משקולות על 11 ערוצים), אותה חלוקה, אותם קלטים; רק המטרה משתנה. תוצאה: <b>{outcome}</b>. P_SOFT − P_HARD: {pp(soft_hard.delta)} נקודות ב-ProcessBench ({verdict(soft_hard)}), {sgn(soft_hard_p.delta)} ב-PRMBench ({verdict(soft_hard_p)}). P_AGREE − P_SOFT: {pp(agree_soft.delta)} נקודות ({verdict(agree_soft)}), {sgn(agree_soft_p.delta)} ({verdict(agree_soft_p)}).</p>
<div class="verdict">
<div><b>{status["status"]}</b><span>{status["fits"]} התאמות, {status["failures"]} כשלים, {nonconv} לא התכנסו; שיעור דגימות bootstrap לא תקפות {status["invalid_bootstrap_draw_rate"]:.4f}; זמן כולל {timing["total_s"]/60:.0f} דקות</span></div>
<div><b>{pct(est("BASE","pb","sla","macro8"))} / {num(est("BASE","prm","within_auc"),3)}</b><span>ה-teacher החדש (step-z, softmax/sigmoid לכל ערוץ, ממוצע). זהו ה-seed שהפרוטוקול של Step 432 תיאר, לא זה שרץ שם</span></div>
<div><b>{pct(est("ct7","pb","sla","macro8"))} / {num(est("ct7","prm","within_auc"),3)}</b><span>CT7, העוגן הקפוא. הזרוע הטובה ביותר של S1: {max(ARMS, key=lambda m: est(m,"pb","sla","macro8"))} {pct(max(est(m,"pb","sla","macro8") for m in ARMS))} ב-PB, {max(ARMS, key=lambda m: est(m,"prm","within_auc"))} {num(max(est(m,"prm","within_auc") for m in ARMS),3)} ב-PRMB</span></div>
</div>

<h2>לוח התוצאות</h2>
<figure><img src="{img('FIG1_scoreboard.png')}" alt="scoreboard"></figure>
<div class="tw"><table><thead><tr><th>שיטה</th><th>PB SLA macro8</th><th>PB F1 (gate CT7)</th><th>PRMB within-AUC</th><th>שגיאה אחת</th><th>כמה שגיאות</th><th>PRMScore raw</th><th>PRMScore z-לכל-תשובה</th><th>PRMScore q80 raw</th></tr></thead><tbody>{sb}</tbody></table></div>
<p class="note">PRMScore לזרועות החדשות: סף quantile שנבחר על fold C (תוויות של C בלבד), מוקפא ומופעל על H; גם גרסת z-לכל-תשובה לפי כלל S0-C. לעוגנים מוצג q80 בלבד (ערכי הריצה המקורית שלהם משוחזרים ב-S0).</p>

<h2>ההשוואות הראשיות (משפחה של 4 מבחנים)</h2>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{prim_rows}</tbody></table></div>
<figure><img src="{img('FIG3_contrasts.png')}" alt="contrasts"></figure>
<h3>בקרות: האם הרווח הוא מהטלמטריה או מ-prior?</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{ctrl_rows}</tbody></table></div>
<h3>מול העוגנים</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{comp_rows}</tbody></table></div>

<h2>איפה ההחלטות השתנו</h2>
<figure><img src="{img('FIG2_per_cell.png')}" alt="per cell"></figure>
<figure><img src="{img('FIG5_confidence_and_rescue.png')}" alt="confidence and rescue"></figure>
<div class="tw"><table><thead><tr><th>student</th><th>הסכמה עם teacher</th><th>ניצלו</th><th>נפגעו</th><th>נטו</th><th>דיוק על תשובות ש-P_AGREE שמר (student / teacher)</th><th>על תשובות שנזנחו</th></tr></thead><tbody>{rows_res}</tbody></table></div>
<h3>PRMBench: שינוי לכל תשובה מול ה-teacher</h3>
<div class="tw"><table><thead><tr><th>student</th><th>Δ AUC ממוצע</th><th>השתפרו</th><th>הורעו</th><th>זוגות נטו</th></tr></thead><tbody>{prm_rows}</tbody></table></div>
<figure><img src="{img('FIG4_depth_and_prm_delta.png')}" alt="depth"></figure>

<h2>איכות הפסאודו-תוויות וכיסוי</h2>
<div class="tw"><table><thead><tr><th>מדד teacher</th><th>ערך</th></tr></thead><tbody>
<tr><td class="lbl">ProcessBench: דיוק top1 של ה-teacher על תשובות שגויות</td><td>{pct(tp.teacher_top1)}</td></tr>
<tr><td class="lbl">מסה על הצעד הנכון / דרגה ממוצעת של הצעד הנכון</td><td>{num(tp.true_step_mass,3)} / {num(tp.true_step_rank,2)}</td></tr>
<tr><td class="lbl">ביטחון c: תשובות שגויות / נקיות</td><td>{num(tp.conf_c_err,3)} / {num(tp.conf_c_clean,3)}</td></tr>
<tr><td class="lbl">P_AGREE שומר: תשובות שגויות / נקיות</td><td>{pct(tp.agree_keep_err)} / {pct(tp.agree_keep_clean)}</td></tr>
<tr><td class="lbl">דיוק teacher על מה ש-P_AGREE שומר</td><td>{pct(tp.teacher_top1_on_kept)}</td></tr>
<tr><td class="lbl">PRMBench: precision / recall של פסאודו קשיח (q ≥ 0.5) מול תוויות אמת</td><td>{num(tq.hard_pseudo_precision,3)} / {num(tq.hard_pseudo_recall,3)}</td></tr>
<tr><td class="lbl">חיוביים חזויים לתשובה / שגיאות אמת לתשובה</td><td>{num(tq.pseudo_positives_per_answer,2)} / {num(tq.true_errors_per_answer,2)}</td></tr>
<tr><td class="lbl">AUPRC של המטרה הרכה</td><td>{num(tq.soft_auprc,3)}</td></tr>
</tbody></table></div>
<div class="tw"><table><thead><tr><th>זרוע</th><th>PB: תשובות ניתנות לאימון ב-B</th><th>PRMB: צעדים בתוך ה-loss</th></tr></thead><tbody>{cov_rows}</tbody></table></div>

<h2>איך לקרוא את זה</h2>
<div class="next">
<p><b>מקור השינוי:</b> ההשוואות P_SOFT − P_HARD ו-P_AGREE − P_SOFT מבודדות את המטרה; P_SOFT − P_POSITION_LENGTH אומר אם ה-student למד מהטלמטריה או מ-prior של מיקום ואורך; P_SOFT − P_RANDOM אומר אם הקשר למיקום הכרחי; P_AGREE − coverage-matched אומר אם ההימנעות עצמה עוזרת או רק מספר הדוגמאות.</p>
<p><b>מול CT7:</b> הזרוע הטובה ביותר ב-S1 עדיין {"מתחת" if max(est(m,"pb","sla","macro8") for m in ARMS) < est("ct7","pb","sla","macro8") else "מעל"} ל-CT7 ב-ProcessBench ו-{"מתחת" if max(est(m,"prm","within_auc") for m in ARMS) < est("ct7","prm","within_auc") else "מעל"} ב-PRMBench. מועמד יכול להצליח במבחן מנגנון ולהישאר חלש מה-incumbent.</p>
<p><b>לפי כללי ההכרעה של התוכנית (14.3):</b> CI שכולל אפס אינו "אין הבדל"; שיפור ב-endpoint אחד ללא פגיעה מובהקת בשני אינו הוכחת no-harm; שיפור מול BASE שאינו עובר בקרה פשוטה הוא "שיפור במתכון, לא הוכחה למנגנון".</p>
</div>
<p class="note">קבצים: <code>results/ssl_pseudolabel_residual_v1/S1/{RUN_ID}/</code>: PROTOCOL.json (בתיקיית השלב, הוקפא לפני הריצה), INPUT_MANIFEST, CODE_MANIFEST + SOURCE_SNAPSHOT, SPLITS, PSEUDO_TARGETS.npz, FIT_MANIFEST.jsonl, OOF_STEP_SCORES.npz, OOF_ANSWERS.csv, EVAL_JOINED.csv, METRICS.csv, CONTRASTS.csv, BOOTSTRAP_DELTAS.npz, TEACHER_DIAGNOSTICS.csv, CONFIDENCE_QUINTILES.csv, TRAINING_COVERAGE.csv, EXAMPLES.csv, FAILURES.csv, TIMING.json, RUN_STATUS.json. בדיקות: <code>tests/test_ssl_s1.py</code> (9 עוברות).</p>
</main>
'''
(OUT / 'REPORT_HE.html').write_text(html, encoding='utf8')
print('written', len(html) // 1024, 'KB', outcome)
