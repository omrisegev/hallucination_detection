"""bank20_lsml_prmbench_v1 figures + Hebrew findings page from the run outputs (no recomputation)."""
from pathlib import Path
import base64, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260924'
OUT = ROOT / 'results/bank20_lsml_prmbench_v1' / RUN_ID
M = pd.read_csv(OUT / 'METRICS.csv'); C = pd.read_csv(OUT / 'CONTRASTS.csv').drop_duplicates(['contrast_id', 'endpoint'])
D = json.loads((OUT / 'DIAGNOSTICS.json').read_text(encoding='utf8')); status = json.loads((OUT / 'RUN_STATUS.json').read_text(encoding='utf8')); timing = json.loads((OUT / 'TIMING.json').read_text(encoding='utf8'))
fits = [json.loads(l) for l in (OUT / 'FIT_MANIFEST.jsonl').read_text(encoding='utf8').splitlines()]
NAMES = json.loads((OUT / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))['channels']
BANKS = ['B11', 'B15', 'B19', 'B20']; ARMS = ['lsml', 'equal', 'group_equal']
LBL = {'lsml': 'L-SML', 'equal': 'ממוצע', 'group_equal': 'ממוצע מאוזן לפי הקבוצות שנלמדו'}
BANKLBL = {'B11': 'בנק11 (המקורי)', 'B15': 'בנק15 (+4 ערוצי CT7)', 'B19': 'בנק19 (+4 קריאות צורה של q15_H1)', 'B20': 'בנק20 (+שטף ראיות)'}
TRIO = ['energy_innovation', 'top15_turnover', 'top50_js']


def est(m, metric, stratum='all'):
    r = M[(M.method == m) & (M.metric == metric) & (M.stratum == stratum)]
    return float(r.estimate.iloc[0]) if len(r) else np.nan


def num(x, d=4): return '—' if pd.isna(x) else f'{x:.{d}f}'
def pct(x, d=2): return '—' if pd.isna(x) else f'{100 * x:.{d}f}'
def sgn(x, d=4): return '—' if pd.isna(x) else f'{x:+.{d}f}'
def pp(x): return '—' if pd.isna(x) else f'{100 * x:+.2f}'


Cp = C.set_index(['contrast_id', 'endpoint'])
def cr(cid, ep): return Cp.loc[(cid, ep)] if (cid, ep) in Cp.index else None

# ---------------------------------------------------------------- FIG 1 scoreboard
fig, ax = plt.subplots(figsize=(8.5, 6))
for b in BANKS:
    for a, mk in zip(ARMS, ['o', 's', '^']):
        m = f'{b}_{a}'; x, y = est(m, 'prmscore_answer_z_q80'), est(m, 'within_auc')
        col = {'B11': '#1f77b4', 'B15': '#2ca02c', 'B19': '#ff7f0e', 'B20': '#d62728'}[b]
        ax.scatter(x, y, c=col, marker=mk, s=70 if a == 'lsml' else 45, zorder=3, alpha=.9 if a == 'lsml' else .6)
        if a == 'lsml': ax.annotate(m, (x, y), fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.scatter(est('ct7', 'prmscore_answer_z_q80'), est('ct7', 'within_auc'), c='#444', marker='D', s=80, zorder=4); ax.annotate('ct7', (est('ct7', 'prmscore_answer_z_q80'), est('ct7', 'within_auc')), fontsize=8, xytext=(4, 3), textcoords='offset points')
ax.axhline(0.7801, color='#999', ls=':', lw=1); ax.text(ax.get_xlim()[0], 0.7801, ' fam421_answer .7801 (context)', fontsize=7, color='#666', va='bottom')
ax.set_xlabel('PRMScore (answer-z, q80, official evaluator)'); ax.set_ylabel('PRMBench within-answer AUROC'); ax.set_title('bank11 -> bank20 under the CONT L-SML recipe  (o L-SML, square equal, triangle group-equal)'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG1_scoreboard.png', dpi=150); plt.close(fig)

# ---------------------------------------------------------------- FIG 2 contrasts
want = ['B20_lsml - B20_equal', 'B20_lsml - B11_lsml', 'B20_lsml - ct7', 'B15_lsml - B11_lsml', 'B19_lsml - B15_lsml', 'B20_lsml - B19_lsml',
        'B11_lsml - B11_equal', 'B15_lsml - B15_equal', 'B19_lsml - B19_equal', 'B20_lsml - B20_group_equal', 'B15_equal - B11_equal', 'B20_equal - B11_equal']
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
for ax, ep, title in [(axes[0], 'prm_within_auc', 'PRMBench within-AUC'), (axes[1], 'prmscore_answer_z_q80', 'PRMScore (answer-z, q80)')]:
    d = C[(C.endpoint == ep) & C.contrast_id.isin(want)].set_index('contrast_id').reindex(want).reset_index(); y = np.arange(len(d))
    ax.errorbar(d.delta, y, xerr=[d.delta - d.ci95_lo, d.ci95_hi - d.delta], fmt='o', color='#1f6f8b', capsize=3)
    for i, r in d.iterrows():
        if r.primary: ax.errorbar([r.delta], [i], xerr=[[r.delta - r.ci_adj_lo], [r.ci_adj_hi - r.delta]], fmt='none', ecolor='#b5452b', capsize=5, lw=.8); ax.text(r.ci95_hi, i, '  primary', va='center', fontsize=7, color='#b5452b')
    ax.axvline(0, color='k', lw=.8); ax.set_yticks(y); ax.set_yticklabels(d.contrast_id, fontsize=8); ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x', alpha=.3)
fig.suptitle('Paired PRMBench source-group bootstrap, 100,000 draws; blue = 95% CI, red = Bonferroni (K=6) on primary contrasts', fontsize=9); fig.tight_layout(); fig.savefig(OUT / 'FIG2_contrasts.png', dpi=150); plt.close(fig)

# ---------------------------------------------------------------- FIG 3 weights + single-stream AUC
W = {b: np.mean([f['weights'] for f in fits if f['bank'] == b], 0) for b in BANKS}
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
x = np.arange(20); w = .2
for j, b in enumerate(BANKS):
    ww = np.full(20, np.nan); ww[:len(W[b])] = W[b]; axes[0].bar(x + (j - 1.5) * w, ww, w, label=b)
for t in TRIO: axes[0].axvspan(NAMES.index(t) - .5, NAMES.index(t) + .5, color='#f6e6e1', zorder=0)
axes[0].axhline(0, color='k', lw=.6); axes[0].set_xticks(x); axes[0].set_xticklabels(NAMES, rotation=70, fontsize=7); axes[0].set_ylabel('L-SML weight (mean of 5 folds, L1-normalized)'); axes[0].set_title('learned weights per bank; shaded = the anti-oriented trio'); axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=.3)
sa = D['single_stream_within_auc']; vals = [sa[nm] for nm in NAMES]
axes[1].bar(x, vals, color=['#1f77b4'] * 11 + ['#2ca02c'] * 4 + ['#ff7f0e'] * 4 + ['#d62728']); axes[1].axhline(.5, color='k', lw=.6, ls='--'); axes[1].axhline(est('ct7', 'within_auc'), color='#444', lw=.8, ls=':'); axes[1].text(0, est('ct7', 'within_auc'), ' ct7', fontsize=7, va='bottom')
axes[1].set_xticks(x); axes[1].set_xticklabels(NAMES, rotation=70, fontsize=7); axes[1].set_ylim(.35, .8); axes[1].set_ylabel('single-stream within-AUC (diagnostic)'); axes[1].set_title('what each channel carries alone under Top10 + answer-z'); axes[1].grid(axis='y', alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG3_weights.png', dpi=150); plt.close(fig)


def img(name): return 'data:image/png;base64,' + base64.b64encode((OUT / name).read_bytes()).decode()


def cell(r, f, adj):
    if r is None: return '<td>—</td>'
    out = r.ci95_lo > 0 or r.ci95_hi < 0; cls = ' class="pos"' if out and r.delta > 0 else ' class="neg"' if out else ''
    s = f'<td{cls}>{f(r.delta)}<br><span class="ci">[{f(r.ci95_lo)}, {f(r.ci95_hi)}]</span>'
    if adj and not pd.isna(r.ci_adj_lo): s += f'<br><span class="ci">Bonf. [{f(r.ci_adj_lo)}, {f(r.ci_adj_hi)}]</span>'
    return s + '</td>'


def crow(cid):
    a = cr(cid, 'prm_within_auc'); b = cr(cid, 'prmscore_answer_z_q80')
    if a is None: return ''
    prim = bool(a.primary); return f'<tr><td class="lbl">{cid}{" · <b>ראשי</b>" if prim else ""}</td>{cell(a, lambda v: sgn(v, 4), prim)}{cell(b, lambda v: sgn(v, 4), prim)}</tr>'


sb = ''.join(f'<tr><td class="lbl">{BANKLBL[b]} · {LBL[a]}</td><td>{num(est(f"{b}_{a}", "within_auc"))}</td><td>{num(est(f"{b}_{a}", "prmscore_answer_z_q80"))}</td><td>{pct(est(f"{b}_{a}", "sla", "macro8_context"))}%</td></tr>' for b in BANKS for a in ARMS)
sb += f'<tr><td class="lbl">בנק11 · L-SML, השחזור המקורי (התאמה על 4 folds)</td><td>{num(est("B11_lsml_replay4", "within_auc"))}</td><td>{num(est("B11_lsml_replay4", "prmscore_answer_z_q80"))}</td><td>{pct(est("B11_lsml_replay4", "sla", "macro8_context"))}%</td></tr>'
sb += f'<tr><td class="lbl">CT7 (עוגן קפוא)</td><td>{num(est("ct7", "within_auc"))}</td><td>{num(est("ct7", "prmscore_answer_z_q80"))}</td><td>{pct(est("ct7", "sla", "macro8_context"))}%</td></tr>'
sb += '<tr><td class="lbl">fam421_answer (הקשר בלבד, Step 434)</td><td>0.7801</td><td>0.6562</td><td>39.94%</td></tr>'
prim = ''.join(crow(c) for c in ['B20_lsml - B20_equal', 'B20_lsml - B11_lsml', 'B20_lsml - ct7'])
ladder = ''.join(crow(c) for c in ['B15_lsml - B11_lsml', 'B19_lsml - B15_lsml', 'B20_lsml - B19_lsml', 'B15_equal - B11_equal', 'B19_equal - B11_equal', 'B20_equal - B11_equal'])
edge = ''.join(crow(c) for c in ['B11_lsml - B11_equal', 'B15_lsml - B15_equal', 'B19_lsml - B19_equal', 'B11_lsml - B11_group_equal', 'B15_lsml - B15_group_equal', 'B19_lsml - B19_group_equal', 'B20_lsml - B20_group_equal', 'B11_lsml_replay4 - ct7', 'B11_lsml - ct7', 'B15_lsml - ct7'])
# per-fold K and the trio's weight per bank
kt = ''.join('<tr><td class="lbl">{}</td>{}</tr>'.format(BANKLBL[b], ''.join(f'<td>K={f["K"]}<br><span class="ci">trio {sum(f["weights"][NAMES.index(t)] for t in TRIO):+.3f}</span></td>' for f in sorted([f for f in fits if f['bank'] == b], key=lambda f: f['fold']))) for b in BANKS)
wrows = ''.join('<tr><td class="lbl">{}</td>{}<td>{}</td></tr>'.format(nm, ''.join(f'<td{" class=neg" if (j < len(W[b]) and W[b][j] < 0) else ""}>{sgn(W[b][j], 3) if j < len(W[b]) else "—"}</td>' for b in BANKS), num(D['single_stream_within_auc'][nm])) for j, nm in enumerate(NAMES))
classes = sorted({s.split('=')[1] for s in M.stratum if s.startswith('class=')})
crows = ''.join('<tr><td class="lbl">{}</td>{}</tr>'.format(cl, ''.join(f'<td>{num(est(m, "within_auc", "class=" + cl), 3)}</td>' for m in ['B11_lsml', 'B15_lsml', 'B20_lsml', 'ct7'])) for cl in classes if cl not in ('correct', 'multi_solutions'))
er = D['effective_rank_prm_steps']

# ---------------------------------------------------------------- verdict per the frozen rule
a1 = cr('B20_lsml - B20_equal', 'prm_within_auc'); a2 = cr('B20_lsml - B11_lsml', 'prm_within_auc'); a3 = cr('B20_lsml - ct7', 'prm_within_auc')
def vd(r): return 'תומך' if r.ci_adj_lo > 0 else 'שלילי' if r.ci_adj_hi < 0 else 'לא מכריע'
outcome = 'supported' if (a1.ci_adj_lo > 0 and a2.ci_adj_lo > 0) else 'unsupported' if (a2.ci_adj_hi < 0 or a1.ci_adj_hi < 0) else 'inconclusive / mixed'
preds = []
preds.append(('P1 שחזור מדויק של בנק11 על 4 folds', 'התקיים' if D['replay']['replay_exact'] else 'לא התקיים', f'הפרש מקסימלי {D["replay"]["max_abs_diff_continuous"]:.1e}'))
e_ok = all(cr(f'{b}_lsml - {b}_equal', 'prm_within_auc').ci95_lo > 0 for b in BANKS if cr(f'{b}_lsml - {b}_equal', 'prm_within_auc') is not None)
preds.append(('P2 ‏L-SML מנצח ממוצע בכל ארבעת הבנקים', 'התקיים' if e_ok else 'לא התקיים', '; '.join(f'{b}: {sgn(cr(f"{b}_lsml - {b}_equal", "prm_within_auc").delta)}' for b in BANKS if cr(f'{b}_lsml - {b}_equal', 'prm_within_auc') is not None)))
p3 = cr('B15_lsml - B11_lsml', 'prm_within_auc'); preds.append(('P3 ארבעת ערוצי CT7 מעלים את בנק11 L-SML ב-0.005 עד 0.010', 'התקיים' if (p3 is not None and .005 <= p3.delta <= .010 and p3.ci95_lo > 0) else 'לא התקיים', f'{sgn(p3.delta)} [{sgn(p3.ci95_lo)}, {sgn(p3.ci95_hi)}]' if p3 is not None else '—'))
p4 = cr('B19_lsml - B15_lsml', 'prm_within_auc'); preds.append(('P4 משפחת הצורה ניטרלית (|Δ| < 0.003)', 'התקיים' if (p4 is not None and abs(p4.delta) < .003) else 'לא התקיים', f'{sgn(p4.delta)} [{sgn(p4.ci95_lo)}, {sgn(p4.ci95_hi)}]' if p4 is not None else '—'))
preds.append(('P5 בנק20 L-SML נשאר מתחת ל-fam421_answer ‏0.7801', 'התקיים' if est('B20_lsml', 'within_auc') < .7801 else 'לא התקיים', num(est('B20_lsml', 'within_auc'))))
prow = ''.join(f'<tr><td class="lbl">{a}</td><td class="{"pos" if b == "התקיים" else "neg"}">{b}</td><td>{c}</td></tr>' for a, b, c in preds)
best = max(BANKS, key=lambda b: est(f'{b}_lsml', 'within_auc'))

html = f'''<title>Bank20 L-SML on PRMBench</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#f6f7f5;--ink:#1c2430;--muted:#5d6874;--rule:#d7dcd9;--card:#eef1ee;--accent:#1f6f8b;--pos:#3f7d4e;--neg:#b5452b;--posbg:#e6f0e8;--negbg:#f6e6e1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}}}
:root[data-theme="dark"]{{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}
body{{background:var(--paper);color:var(--ink);font-family:Heebo,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.6;direction:rtl}}
main{{max-width:880px;margin:0 auto;padding:40px 24px 80px}}
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
td.lbl{{font-family:Heebo,sans-serif;direction:rtl;white-space:normal;min-width:190px}}
td.pos{{color:var(--pos);background:var(--posbg)}} td.neg{{color:var(--neg);background:var(--negbg)}}
.ci{{font-size:11px;color:var(--muted)}}
figure{{margin:16px 0 4px}} figure img{{width:100%;border:1px solid var(--rule);border-radius:3px;background:#fff}}
.note{{font-size:14px;color:var(--muted);margin:6px 0}}
.next{{background:var(--card);padding:16px 20px;border-radius:4px;border-right:3px solid var(--accent)}}
code{{font-family:"IBM Plex Mono",monospace;font-size:13px;direction:ltr;unicode-bidi:embed}}
@media (max-width:600px){{.verdict{{grid-template-columns:1fr}}}}
</style>
<main>
<div class="eyebrow">bank20_lsml_prmbench_v1 · {RUN_ID} · PRMBench first · development data only</div>
<h1>מבנק11 לבנק20: האם עוד ערוצים משמרים את היתרון של L-SML ומעלים את PRMBench</h1>
<p class="lede">אותו מתכון בדיוק של Codex (Top10, נרמול לכל תשובה, CONT L-SML, חמישה source folds), על סולם מקונן של ארבעה בנקים. תוצאה: <b>{outcome}</b>. בנק20 L-SML מול ממוצע על אותו בנק: {sgn(a1.delta)} ({vd(a1)}). בנק20 L-SML מול בנק11 L-SML: {sgn(a2.delta)} ({vd(a2)}). מול CT7: {sgn(a3.delta)} ({vd(a3)}). הבנק הטוב ביותר תחת L-SML: {best} ‏({num(est(f"{best}_lsml", "within_auc"))}).</p>
<div class="verdict">
<div><b>{status["status"]}</b><span>{status["fits"]} התאמות, {status["failures"]} כשלים, שחזור מדויק {"כן" if D["replay"]["replay_exact"] else "לא"}; {timing["total_s"] / 60:.0f} דקות CPU</span></div>
<div><b>{num(est("B11_lsml", "within_auc"), 4)} → {num(est("B20_lsml", "within_auc"), 4)}</b><span>within-AUC של L-SML מבנק11 לבנק20 (פרוטוקול 3 folds להתאמה). CT7 ‏{num(est("ct7", "within_auc"), 4)}, fam421 ‏0.7801</span></div>
<div><b>{sgn(cr("B11_lsml - B11_equal", "prm_within_auc").delta)} → {sgn(a1.delta)}</b><span>היתרון של L-SML על ממוצע, בנק11 מול בנק20. דרגה אפקטיבית: {", ".join(f"{b} {er[b]:.2f}" for b in BANKS)}</span></div>
</div>
<h2>למה זה נבנה כך</h2>
<p>עומרי ביקש להרחיב את בנק 11 הערוצים לכיוון 20, כך שהיתרון של L-SML על מיצוע יישמר וגם התוצאה תשתפר, על PRMBench. לפני הריצה נרשם האבחון: היתרון של L-SML על בנק11 הוא <b>השתקה של ערוצים מזיקים</b>. המשקלים שנלמדו (זהים בכל חמשת ה-folds) נותנים ל-energy_innovation, top15_turnover ו-top50_js ‏0.015 כל אחד, ותחת קריאת Top10 שניים מהם הפוכים בכיוון (AUC ‏0.407 ו-0.458) והשלישי חלש (0.606). ממוצע לא יכול להשתיק אותם; L-SML כן. לכן כל ערוץ שנוסף חייב לשאת מידע וגם ליצור קבוצה שהמשקלים יודעים לנצל.</p>
<p>ההרכב: 11 המקוריים; ארבעה ערוצי טוקן מ-CT7 שאינם עותקים (ראיית הטוקן הנבחר, שארית BOCPD, innovation של H0lim, ve0); ארבע קריאות צורה של ערוץ העוגן q15_H1 מתוך Step 432 (טוקן ראשון, שיפוע, קפיצה, שיעור מעל סף), מוצהרות לפי בנייה ולא נבחרות; ושטף הראיות של Mind-the-Gap. הכול מחושב מראש, בלי GPU ובלי תוויות. הפרדת fit/calibration/evaluation לפי R0-B של Codex: 3 folds להתאמה, אחד לסף PRMScore, אחד להערכה; ובנוסף שחזור של בנק11 עם ההתאמה המקורית על 4 folds מול הציונים השמורים.</p>
<h2>לוח התוצאות</h2>
<figure><img src="{img('FIG1_scoreboard.png')}" alt="scoreboard"></figure>
<div class="tw"><table><thead><tr><th>בנק · זרוע</th><th>PRMB within-AUC</th><th>PRMScore (answer-z, q80)</th><th>PB SLA macro8 (הקשר)</th></tr></thead><tbody>{sb}</tbody></table></div>
<p class="note">6,030 תשובות PRMBench עם צעדים משני הסוגים ל-within-AUC; 6,211 תשובות ללא בקרות ל-PRMScore דרך ה-evaluator הרשמי. ProcessBench מדווח כהקשר בלבד ואינו וטו.</p>
<h2>ההשוואות הראשיות (משפחה של 6 מבחנים)</h2>
<div class="tw"><table><thead><tr><th>השוואה</th><th>PRMB within-AUC</th><th>PRMScore</th></tr></thead><tbody>{prim}</tbody></table></div>
<figure><img src="{img('FIG2_contrasts.png')}" alt="contrasts"></figure>
<h3>הסולם: איזו משפחה הזיזה מה</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>PRMB within-AUC</th><th>PRMScore</th></tr></thead><tbody>{ladder}</tbody></table></div>
<h3>האם היתרון של L-SML שורד את ההרחבה</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>PRMB within-AUC</th><th>PRMScore</th></tr></thead><tbody>{edge}</tbody></table></div>
<h2>המנגנון: מה קורה למשקלים</h2>
<figure><img src="{img('FIG3_weights.png')}" alt="weights"></figure>
<div class="tw"><table><thead><tr><th>בנק</th><th>fold 0</th><th>fold 1</th><th>fold 2</th><th>fold 3</th><th>fold 4</th></tr></thead><tbody>{kt}</tbody></table></div>
<p class="note">K = מספר הקבוצות שהחלוקה האוטומטית מצאה; "trio" = סכום המשקלים של שלושת הערוצים המזיקים באותו fold.</p>
<div class="tw"><table><thead><tr><th>ערוץ</th><th>בנק11</th><th>בנק15</th><th>בנק19</th><th>בנק20</th><th>AUC לבד</th></tr></thead><tbody>{wrows}</tbody></table></div>
<p class="note">משקלי L-SML, ממוצע על חמישה folds; אדום = משקל שלילי. העמודה האחרונה: within-AUC של הערוץ לבדו תחת Top10 ונרמול לכל תשובה (אבחון עם תוויות, לא נכנס לשום התאמה).</p>
<h3>לפי סוג השגיאה של PRMBench</h3>
<div class="tw"><table><thead><tr><th>סוג</th><th>בנק11 L-SML</th><th>בנק15 L-SML</th><th>בנק20 L-SML</th><th>CT7</th></tr></thead><tbody>{crows}</tbody></table></div>
<h2>התחזיות שהוקפאו</h2>
<div class="tw"><table><thead><tr><th>תחזית</th><th>תוצאה</th><th>מספר</th></tr></thead><tbody>{prow}</tbody></table></div>
<h2>איך לקרוא את זה</h2>
<div class="next">
<p><b>כלל הפירוש שהוקפא:</b> רווח חייב לעבור גם את בנק11 L-SML וגם לשמור את L-SML מעל ממוצע על אותו בנק. ניצחון על ממוצע בלבד משחזר את אפקט ההשתקה ואינו רווח בייצוג.</p>
<p><b>לפי הכלל:</b> {"בנק20 עולה על בנק11 ושומר את יתרון L-SML: מועמד development להמשך, לא promotion." if outcome == "supported" else "ההרחבה לא עוזרת: או שבנק20 אינו טוב מבנק11 תחת L-SML, או שהיתרון של L-SML על ממוצע נעלם כשמוסיפים ערוצים." if outcome == "unsupported" else "תמונה מעורבת; אין winner."}</p>
<p><b>מה שהסולם אומר על המנגנון:</b> בבנק11 החלוקה האוטומטית בודדה את שלושת הערוצים המזיקים לקבוצה משלהם וכמעט איפסה אותה. ברגע שמוסיפים ערוצים, החלוקה משתנה, וההשתקה הזו אינה מובטחת. זה מה שטבלת K והמשקלים מראה fold אחר fold.</p>
</div>
<p class="note">קבצים: <code>results/bank20_lsml_prmbench_v1/{RUN_ID}/</code>. פרוטוקול קפוא ב-<code>PROTOCOL.json</code> (commit לפני הניקוד). קוד ה-fusion יובא מ-<code>.worktrees/depth-feature-fusion-v1</code> ללא שינוי.</p>
</main>
'''
(OUT / 'REPORT_HE.html').write_text(html, encoding='utf8')
print('written', len(html) // 1024, 'KB', outcome)
