"""S5-CPU figures + Hebrew findings page from the run outputs (no recomputation)."""
from pathlib import Path
import base64, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260923'
OUT = ROOT / 'results/ssl_pseudolabel_residual_v1/S5' / RUN_ID
M = pd.read_csv(OUT / 'METRICS.csv')
C = pd.read_csv(OUT / 'CONTRASTS.csv').drop_duplicates(['contrast_id', 'endpoint'])
A = pd.read_csv(OUT / 'ATTENTION_DIAGNOSTICS.csv')
W = pd.read_csv(OUT / 'TOKEN_RISK_WEIGHTS.csv')
RM = pd.read_csv(OUT / 'RESCUE_MATRIX.csv')
status = json.loads((OUT / 'RUN_STATUS.json').read_text(encoding='utf8'))
timing = json.loads((OUT / 'TIMING.json').read_text(encoding='utf8'))
traces = np.load(OUT / 'LOSS_TRACES.npz')
CH = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
      'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0']
ARMS = ['H_TOP5', 'H_MEAN', 'H_RANDATT', 'H_RAW']
COMP = ['ct7', 'token_lsml', 'token_equal', 'evidence30__all__plain__equal',
        'evidence__all__position__equal', 'evidence__all__position2__equal']
TASKS = ['pb_q4', 'pb_q8', 'prm']
N = {'BASE': 'BASE (teacher, top5 קבוע)', 'H_TOP5': 'H_TOP5 (top5 קבוע, משקלי ערוץ נלמדים)',
     'H_MEAN': 'H_MEAN (ממוצע טוקנים, נלמד)', 'H_RANDATT': 'H_RANDATT (attention אקראי קפוא)',
     'H_RAW': 'H_RAW (attention נלמד)', 'ct7': 'CT7 (עוגן קפוא)', 'token_lsml': 'token L-SML',
     'token_equal': 'token equal', 'evidence30__all__plain__equal': 'Step432 top30 plain',
     'evidence__all__position__equal': 'Step432 top5 position',
     'evidence__all__position2__equal': 'Step432 top5 position iter2'}


def est(m, bench, metric, stratum='all'):
    r = M[(M.method == m) & (M.benchmark == bench) & (M.metric == metric) & (M.stratum == stratum)]
    return float(r.estimate.iloc[0]) if len(r) else np.nan


def num(x, d=4): return '—' if pd.isna(x) else f'{x:.{d}f}'
def pct(x, d=1): return '—' if pd.isna(x) else f'{100 * x:.{d}f}%'
def sgn(x, d=4): return f'{x:+.{d}f}'
def pp(x): return f'{100 * x:+.2f}'


# ---------------------------------------------------------------- FIG 1 scoreboard
fig, ax = plt.subplots(figsize=(8.5, 6))
for m in ['BASE'] + ARMS + COMP:
    x, y = est(m, 'prm', 'within_auc'), 100 * est(m, 'pb', 'sla', 'macro8')
    anchor = m in COMP
    col = '#444' if anchor else '#1f77b4' if m == 'BASE' else '#d62728' if m == 'H_RAW' else '#999'
    ax.scatter(x, y, c=col, marker='D' if anchor else 'o', s=80 if anchor else 60, zorder=3)
    ax.annotate(m, (x, y), fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.set_xlabel('PRMBench within-answer AUROC'); ax.set_ylabel('ProcessBench SLA macro8, %')
ax.set_title('S5-CPU: learned attention pooling inside a step'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG1_scoreboard.png', dpi=150); plt.close(fig)

# ---------------------------------------------------------------- FIG 2 contrasts
want = ['H_RAW - BASE', 'H_RAW - H_MEAN', 'H_RAW - ct7', 'H_RAW - H_RANDATT', 'H_RAW - H_TOP5',
        'H_MEAN - BASE', 'H_TOP5 - BASE', 'H_RANDATT - BASE', 'BASE - ct7']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, ep, scale, title in [(axes[0], 'pb_sla_macro8', 100, 'ProcessBench SLA, pp'),
                             (axes[1], 'prm_within_auc', 1, 'PRMBench within-AUC')]:
    d = C[(C.endpoint == ep) & C.contrast_id.isin(want)].set_index('contrast_id').reindex(want).reset_index()
    y = np.arange(len(d))
    ax.errorbar(scale * d.delta, y, xerr=[scale * (d.delta - d.ci95_lo), scale * (d.ci95_hi - d.delta)],
                fmt='o', color='#1f6f8b', capsize=3)
    for i, r in d.iterrows():
        if r.primary:
            ax.errorbar([scale * r.delta], [i], xerr=[[scale * (r.delta - r.ci_adj_lo)], [scale * (r.ci_adj_hi - r.delta)]],
                        fmt='none', ecolor='#b5452b', capsize=5, lw=.8)
            ax.text(scale * r.ci95_hi, i, '  primary', va='center', fontsize=7, color='#b5452b')
    ax.axvline(0, color='k', lw=.8); ax.set_yticks(y); ax.set_yticklabels(d.contrast_id, fontsize=8)
    ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x', alpha=.3)
fig.suptitle('Paired source-group bootstrap, 100,000 draws; blue = 95% CI, red = Bonferroni (K=6) on primary contrasts', fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'FIG2_contrasts.png', dpi=150); plt.close(fig)

# ---------------------------------------------------------------- FIG 3 attention behaviour + training
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
w = .35
for j, arm in enumerate(['H_RANDATT', 'H_RAW']):
    d = A[A.arm == arm].set_index('task').reindex(TASKS)
    axes[0].bar(np.arange(3) + (j - .5) * w, d.attention_entropy_ratio, w, label=arm)
axes[0].axhline(1, color='#b5452b', ls='--', lw=1, label='uniform pooling')
axes[0].set_xticks(range(3)); axes[0].set_xticklabels(TASKS); axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel('attention entropy / log(tokens in step)')
axes[0].set_title('how concentrated the attention is'); axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=.3)
for task, c in zip(TASKS, ['#1f77b4', '#2ca02c', '#d62728']):
    r = A[(A.arm == 'H_RAW') & (A.task == task)]
    if len(r):
        axes[1].plot(range(11), [float(r['delta_' + ch].iloc[0]) for ch in CH], marker='o', color=c, label=task, lw=1.2)
axes[1].axhline(0, color='k', lw=.6); axes[1].set_xticks(range(11))
axes[1].set_xticklabels(CH, rotation=60, fontsize=7)
axes[1].set_ylabel('attention-weighted mean - uniform mean')
axes[1].set_title('which channels the learned attention leans on'); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
for arm, c in zip(['H_MEAN', 'H_RANDATT', 'H_RAW'], ['#999', '#2ca02c', '#d62728']):
    keys = [k for k in traces.files if k.endswith('__' + arm + '__seed0') and k.startswith('prm__')]
    if keys:
        tr = np.mean([traces[k] for k in keys], 0)
        axes[2].plot(np.convolve(tr, np.ones(25) / 25, 'valid'), color=c, label=arm, lw=1.2)
axes[2].set_xlabel('update'); axes[2].set_ylabel('training loss (25-update mean)')
axes[2].set_title('PRMBench training loss, seed 0, mean of 5 folds'); axes[2].legend(fontsize=8); axes[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG3_attention.png', dpi=150); plt.close(fig)


def img(name): return 'data:image/png;base64,' + base64.b64encode((OUT / name).read_bytes()).decode()


Cp = C.set_index(['contrast_id', 'endpoint'])


def crow(cid):
    if (cid, 'pb_sla_macro8') not in Cp.index: return ''
    a = Cp.loc[(cid, 'pb_sla_macro8')]; b = Cp.loc[(cid, 'prm_within_auc')]
    def cell(r, f, adj):
        out = r.ci95_lo > 0 or r.ci95_hi < 0
        cls = ' class="pos"' if out and r.delta > 0 else ' class="neg"' if out else ''
        s = f'<td{cls}>{f(r.delta)}<br><span class="ci">[{f(r.ci95_lo)}, {f(r.ci95_hi)}]</span>'
        if adj and not pd.isna(r.ci_adj_lo):
            s += f'<br><span class="ci">Bonf. [{f(r.ci_adj_lo)}, {f(r.ci_adj_hi)}]</span>'
        return s + '</td>'
    lbl = cid + (' · <b>ראשי</b>' if bool(a.primary) else '')
    return f'<tr><td class="lbl">{lbl}</td>{cell(a, pp, bool(a.primary))}{cell(b, lambda x: sgn(x, 4), bool(a.primary))}</tr>'


prim = ''.join(crow(c) for c in ['H_RAW - BASE', 'H_RAW - H_MEAN', 'H_RAW - ct7'])
ctrl = ''.join(crow(c) for c in ['H_RAW - H_RANDATT', 'H_RAW - H_TOP5', 'H_MEAN - BASE',
                                 'H_TOP5 - BASE', 'H_RANDATT - BASE', 'H_MEAN - H_TOP5'])
comp = ''.join(crow(c) for c in ['BASE - ct7', 'H_RAW - token_lsml', 'H_RAW - token_equal',
                                 'H_MEAN - ct7', 'H_TOP5 - ct7'])
sb = ''.join('<tr><td class="lbl">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{} / {}</td></tr>'.format(
    N[m], pct(est(m, 'pb', 'sla', 'macro8')), pct(est(m, 'pb', 'f1', 'macro8')),
    num(est(m, 'prm', 'within_auc')), num(est(m, 'prm', 'prmscore_inner_raw')),
    num(est(m, 'prm', 'prmscore_inner_answer_z')),
    pct(est(m, 'pb', 'early', 'macro8')), pct(est(m, 'pb', 'late', 'macro8'))) for m in ['BASE'] + ARMS + COMP)
arows = ''.join('<tr><td class="lbl">{} / {}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
    r.task, r.arm, int(r.multi_token_steps), num(r.attention_entropy_ratio, 3),
    num(r.max_token_mass_x_n, 2), num(r.first_token_mass_x_n, 2), num(r.last_token_mass_x_n, 2))
    for r in A.sort_values(['task', 'arm']).itertuples())
wm = W[W.arm == 'H_RAW'].groupby('task')[CH].mean()
ws = W[W.arm == 'H_RAW'].groupby('task')[CH].std()
wrows = ''.join('<tr><td class="lbl">{}</td>{}</tr>'.format(
    c, ''.join(f'<td>{sgn(wm.loc[t, c], 3)}<br><span class="ci">±{ws.loc[t, c]:.3f}</span></td>' for t in TASKS)) for c in CH)
rm_pb = RM[RM.benchmark == 'pb'].set_index('arm'); rm_pr = RM[RM.benchmark == 'prm'].set_index('arm')
resc = ''.join('<tr><td class="lbl">{}</td><td>{}</td><td class="pos">{}</td><td class="neg">{}</td><td>{:+d}</td><td>{} / {}</td><td>{}</td><td>{} / {}</td></tr>'.format(
    N[m], pct(rm_pb.loc[m].agreement_with_base, 0), int(rm_pb.loc[m].arm_only), int(rm_pb.loc[m].base_only),
    int(rm_pb.loc[m].net), int(rm_pb.loc[m].moved_earlier), int(rm_pb.loc[m].moved_later),
    sgn(rm_pr.loc[m].mean_delta_auc_vs_base), int(rm_pr.loc[m].answers_improved),
    int(rm_pr.loc[m].answers_worsened)) for m in ARMS)

rb = Cp.loc[('H_RAW - BASE', 'pb_sla_macro8')]; rbp = Cp.loc[('H_RAW - BASE', 'prm_within_auc')]
rm_ = Cp.loc[('H_RAW - H_MEAN', 'pb_sla_macro8')]; rmp = Cp.loc[('H_RAW - H_MEAN', 'prm_within_auc')]
rc = Cp.loc[('H_RAW - ct7', 'pb_sla_macro8')]; rcp = Cp.loc[('H_RAW - ct7', 'prm_within_auc')]


def vd(r): return 'תומך' if r.ci_adj_lo > 0 else 'שלילי' if r.ci_adj_hi < 0 else 'לא מכריע'


pos = [r.ci_adj_lo > 0 for r in [rb, rbp, rm_, rmp]]
neg_base = rb.ci_adj_hi < 0 or rbp.ci_adj_hi < 0
neg_mean = rm_.ci_adj_hi < 0 or rmp.ci_adj_hi < 0
outcome = 'supported' if all(pos) else 'unsupported' if (neg_base and neg_mean) else 'inconclusive / mixed'
verdict_txt = ('שיפור מול ה-teacher וגם מול ממוצע אחיד: מועמד development להמשך, לא promotion.' if outcome == 'supported'
               else 'pooling נלמד בתוך צעד אינו עדיף על ה-readout הקבוע; ה-attention לא מצא מה שהממוצע האחיד מפספס.' if outcome == 'unsupported'
               else 'תמונה מעורבת או לא מכריעה; אין winner ואין סגירת המשפחה.')
ent_raw = A[A.arm == 'H_RAW'].attention_entropy_ratio.mean()
ent_rnd = A[A.arm == 'H_RANDATT'].attention_entropy_ratio.mean()

html = f'''<title>S5 Attention Pooling · Step 437</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#f6f7f5;--ink:#1c2430;--muted:#5d6874;--rule:#d7dcd9;--card:#eef1ee;--accent:#1f6f8b;--pos:#3f7d4e;--neg:#b5452b;--posbg:#e6f0e8;--negbg:#f6e6e1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}}}
:root[data-theme="dark"]{{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}
body{{background:var(--paper);color:var(--ink);font-family:Heebo,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.6;direction:rtl}}
main{{max-width:860px;margin:0 auto;padding:40px 24px 80px}}
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
td.lbl{{font-family:Heebo,sans-serif;direction:rtl;white-space:normal;min-width:185px}}
td.pos{{color:var(--pos);background:var(--posbg)}} td.neg{{color:var(--neg);background:var(--negbg)}}
.ci{{font-size:11px;color:var(--muted)}}
figure{{margin:16px 0 4px}} figure img{{width:100%;border:1px solid var(--rule);border-radius:3px;background:#fff}}
.note{{font-size:14px;color:var(--muted);margin:6px 0}}
.next{{background:var(--card);padding:16px 20px;border-radius:4px;border-right:3px solid var(--accent)}}
code{{font-family:"IBM Plex Mono",monospace;font-size:13px;direction:ltr;unicode-bidi:embed}}
@media (max-width:600px){{.verdict{{grid-template-columns:1fr}}}}
</style>
<main>
<div class="eyebrow">SSL / pseudo-label / residual plan v1.1 · stage S5-CPU · {RUN_ID} · development data only</div>
<h1>S5: האם attention נלמד בתוך צעד עדיף על ממוצע קבוע</h1>
<p class="lede">ה-readout הקפוא לוקח מכל ערוץ את חמשת הטוקנים הגבוהים בצעד. כאן נותנים לראש קטן ללמוד בעצמו לאילו טוקנים להקשיב, מול אותו teacher בדיוק. תוצאה: <b>{outcome}</b>. H_RAW − BASE: {pp(rb.delta)} נקודות ב-ProcessBench ({vd(rb)}), {sgn(rbp.delta)} ב-PRMBench ({vd(rbp)}). מול ממוצע אחיד באותו אימון: {pp(rm_.delta)} ({vd(rm_)}), {sgn(rmp.delta)} ({vd(rmp)}).</p>
<div class="verdict">
<div><b>{status["status"]}</b><span>{status["fits"]} התאמות, {status["failures"]} כשלים, {sum(status["fallback_answers"].values())} תשובות fallback; {timing["total_s"] / 60:.0f} דקות CPU</span></div>
<div><b>{num(ent_raw, 3)} / {num(ent_rnd, 3)}</b><span>ריכוז ה-attention (אנטרופיה חלקי log של מספר הטוקנים בצעד) עבור H_RAW מול attention אקראי קפוא; 1.000 פירושו ממוצע אחיד</span></div>
<div><b>{pct(est("H_RAW", "pb", "sla", "macro8"))} / {num(est("H_RAW", "prm", "within_auc"), 3)}</b><span>H_RAW מול BASE {pct(est("BASE", "pb", "sla", "macro8"))} / {num(est("BASE", "prm", "within_auc"), 3)} ומול CT7 {pct(est("ct7", "pb", "sla", "macro8"))} / {num(est("ct7", "prm", "within_auc"), 3)}</span></div>
</div>
<h2>למה השלב הזה שונה ממה שכתוב בתוכנית</h2>
<p>סעיף 11 של התוכנית מרכיב לכל טוקן שלושה בלוקים: 11 ערוצי הטלמטריה, 11 שאריות של חיזוי ממוסך, ו-32 מספרים מ-encoder שנלמד ב-S4. שני הבלוקים האחרונים ושתיים מארבע הזרועות תלויים ב-S4, שהוא self-supervision על GPU. עומרי פסל אותו ב-23 בספטמבר, ולכן רץ כאן בדיוק מה ששורד בלעדיו: זרוע <code>H_RAW</code> של התוכנית עצמה, עם אותה מטרה (P_SOFT), אותו חוזה תפקידים, אותם פרמטרי אימון ואותן שתי נקודות מדידה. <b>מה שלא נענה כאן:</b> האם ייצוג נלמד מוסיף מידע. נענית רק השאלה הצרה של ה-pooling.</p>
<p>תוקן גם דבר אחד מ-S2: הקלט לכל טוקן מסטונדרט לכל ערוץ ולכל תשובה. הנרמול הקפוא median/IQR משאיר לשלושה ערוצים שונות של 1e6 עד 3e9, וכל pooling שמערבב ערוצים היה נשלט על ידם בלבד.</p>
<h2>סולם הזרועות</h2>
<div class="tw"><table><thead><tr><th>זרוע</th><th>readout</th><th>משקלי ערוץ</th><th>אימון</th></tr></thead><tbody>
<tr><td class="lbl">BASE</td><td>top5 קבוע לכל ערוץ</td><td>שווים</td><td>ללא</td></tr>
<tr><td class="lbl">H_TOP5</td><td>top5 קבוע לכל ערוץ</td><td>נלמדים</td><td>L-BFGS על fold B</td></tr>
<tr><td class="lbl">H_MEAN</td><td>ממוצע אחיד על טוקני הצעד</td><td>נלמדים</td><td>AdamW, 2000 עדכונים, 3 seeds</td></tr>
<tr><td class="lbl">H_RANDATT</td><td>attention בצורה זהה, קפוא באתחול אקראי</td><td>נלמדים</td><td>אותו אימון</td></tr>
<tr><td class="lbl">H_RAW</td><td>attention נלמד בתוך הצעד</td><td>נלמדים</td><td>אותו אימון</td></tr>
</tbody></table></div>
<h2>לוח התוצאות</h2>
<figure><img src="{img('FIG1_scoreboard.png')}" alt="scoreboard"></figure>
<div class="tw"><table><thead><tr><th>שיטה</th><th>PB SLA macro8</th><th>PB F1 (gate CT7)</th><th>PRMB within-AUC</th><th>PRMScore raw</th><th>PRMScore z-לכל-תשובה</th><th>החטאה מוקדמת / מאוחרת</th></tr></thead><tbody>{sb}</tbody></table></div>
<h2>ההשוואות הראשיות (משפחה של 6 מבחנים)</h2>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{prim}</tbody></table></div>
<figure><img src="{img('FIG2_contrasts.png')}" alt="contrasts"></figure>
<h3>בקרות ואבחונים</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{ctrl}</tbody></table></div>
<h3>מול העוגנים</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>ProcessBench SLA (pp)</th><th>PRMBench within-AUC</th></tr></thead><tbody>{comp}</tbody></table></div>
<h2>מה ה-attention למד לעשות</h2>
<figure><img src="{img('FIG3_attention.png')}" alt="attention diagnostics"></figure>
<div class="tw"><table><thead><tr><th>task / זרוע</th><th>צעדים עם יותר מטוקן אחד</th><th>אנטרופיה יחסית</th><th>מסת הטוקן המוביל ×n</th><th>טוקן ראשון ×n</th><th>טוקן אחרון ×n</th></tr></thead><tbody>{arows}</tbody></table></div>
<p class="note">אנטרופיה יחסית 1.000 = ממוצע אחיד. "מסה ×n" מנורמלת כך ש-1.00 הוא מה שממוצע אחיד היה נותן לאותו טוקן, כלומר ערך 3.0 פירושו פי שלושה מהמשקל האחיד. ה-attention אינו הסבר סיבתי.</p>
<h3>משקלי סיכון הטוקן שנלמדו (H_RAW, ממוצע ±SD על 5 folds × 3 seeds)</h3>
<div class="tw"><table><thead><tr><th>ערוץ</th><th>pb_q4</th><th>pb_q8</th><th>prm</th></tr></thead><tbody>{wrows}</tbody></table></div>
<h2>איפה ההחלטות השתנו</h2>
<div class="tw"><table><thead><tr><th>זרוע</th><th>הסכמה עם BASE</th><th>ניצלו</th><th>נפגעו</th><th>נטו</th><th>זז מוקדם / מאוחר</th><th>PRMB Δ AUC</th><th>השתפרו / הורעו</th></tr></thead><tbody>{resc}</tbody></table></div>
<h2>איך לקרוא את זה</h2>
<div class="next">
<p><b>כלל הפירוש שהוקפא:</b> רווח חייב לעבור גם את BASE וגם את H_MEAN, כלומר גם את ה-readout הקפוא וגם ממוצע אחיד שאומן באותו אופן ובאותו תקציב. ניצחון על BASE בלבד היה מערבב "pooling נלמד" עם "משקלי ערוץ נלמדים", ולכן H_TOP5 ו-H_MEAN נמצאים שם בדיוק בשביל להפריד את השניים.</p>
<p><b>לפי טבלת ההכרעה:</b> {verdict_txt}</p>
<p><b>מול CT7:</b> H_RAW {"מתחת" if est("H_RAW", "pb", "sla", "macro8") < est("ct7", "pb", "sla", "macro8") else "מעל"} ל-CT7 ב-ProcessBench ({pp(rc.delta)} נקודות) ו-{"מתחת" if est("H_RAW", "prm", "within_auc") < est("ct7", "prm", "within_auc") else "מעל"} ב-PRMBench ({sgn(rcp.delta)}).</p>
</div>
<p class="note">קבצים: <code>results/ssl_pseudolabel_residual_v1/S5/{RUN_ID}/</code>. פרוטוקול קפוא: <code>S5/PROTOCOL.json</code>. בדיקות: <code>tests/test_ssl_s5.py</code> (6 עוברות: סטנדרטיזציה לכל ערוץ ולכל תשובה, פעולות segment מול לולאת ייחוס, attention עם לוגיטים שטוחים שווה בדיוק לממוצע, שתי פונקציות ה-loss מול חישוב ידני ותשובה בת צעד אחד שעולה אפס, דחיית קלט דמוי-label והקפאת ה-attention האקראי, וכלל הסקאלה המשוקללת).</p>
</main>
'''
(OUT / 'REPORT_HE.html').write_text(html, encoding='utf8')
print('written', len(html) // 1024, 'KB', outcome)
