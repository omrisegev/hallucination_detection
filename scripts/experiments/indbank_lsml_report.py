"""indbank_lsml_prmbench_v1 figures + Hebrew findings page from the run outputs (no recomputation)."""
from pathlib import Path
import base64, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260924'
OUT = ROOT / 'results/indbank_lsml_prmbench_v1' / RUN_ID
M = pd.read_csv(OUT / 'METRICS.csv'); C = pd.read_csv(OUT / 'CONTRASTS.csv').drop_duplicates(['contrast_id', 'endpoint'])
D = json.loads((OUT / 'DIAGNOSTICS.json').read_text(encoding='utf8')); status = json.loads((OUT / 'RUN_STATUS.json').read_text(encoding='utf8')); timing = json.loads((OUT / 'TIMING.json').read_text(encoding='utf8'))
fits = [json.loads(l) for l in (OUT / 'FIT_MANIFEST.jsonl').read_text(encoding='utf8').splitlines()]
S = pd.read_csv(OUT.parent / 'POOL_STRUCTURE.csv').set_index('channel'); IND = D['IND']
def est(m, metric, stratum='all'):
    r = M[(M.method == m) & (M.metric == metric) & (M.stratum == stratum)]; return float(r.estimate.iloc[0]) if len(r) else np.nan
def num(x, d=4): return '—' if pd.isna(x) else f'{x:.{d}f}'
def sgn(x, d=4): return '—' if pd.isna(x) else f'{x:+.{d}f}'
def pct(x): return '—' if pd.isna(x) else f'{100 * x:.2f}'
Cp = C.set_index(['contrast_id', 'endpoint'])
def cr(cid, ep): return Cp.loc[(cid, ep)] if (cid, ep) in Cp.index else None
BANKS = ['B11', 'B11_IND_lf', 'B11_IND_or']; ARMS = ['lsml', 'equal', 'group_equal']
BL = {'B11': 'בנק11', 'B11_IND_lf': 'בנק11 + 21 בלתי תלויים, כיוון נטול תוויות', 'B11_IND_or': 'בנק11 + 21 בלתי תלויים, כיוון אורקל (אבחון עם תוויות)'}
AL = {'lsml': 'L-SML', 'equal': 'ממוצע', 'group_equal': 'ממוצע מאוזן לפי הקבוצות שנלמדו'}

# FIG 1: single-stream AUC vs independence for the whole pool, IND highlighted
fig, ax = plt.subplots(figsize=(9, 6))
for ch, r in S.iterrows():
    if not np.isfinite(r.r_level_marginal): continue
    sel = ch in IND; b11 = bool(r.in_bank11)
    ax.scatter(abs(r.r_level_marginal), r.auc_oriented, c='#d62728' if sel else '#1f77b4' if b11 else '#aaa', s=60 if sel or b11 else 30, zorder=3, alpha=.9)
    if sel or b11: ax.annotate(ch.replace('hist_entropy_rolling_', 'H.').replace('hist_', ''), (abs(r.r_level_marginal), r.auc_oriented), fontsize=6, xytext=(3, 2), textcoords='offset points')
ax.axvline(.35, color='#b5452b', ls='--', lw=1); ax.text(.36, .52, 'selection rule |r| < .35', fontsize=8, color='#b5452b')
ax.set_xlabel('|correlation with the level family|  (label-free)'); ax.set_ylabel('single-stream within-AUC, oriented  (diagnostic)'); ax.set_title('the pool: red = the 21 selected independent channels, blue = bank11'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG1_pool.png', dpi=150); plt.close(fig)
# FIG 2: contrasts
want = ['B11_IND_lf_lsml - B11_lsml', 'B11_IND_lf_lsml - B11_IND_lf_equal', 'B11_IND_or_lsml - B11_lsml', 'B11_IND_or_lsml - B11_IND_or_equal', 'B11_IND_lf_equal - B11_equal', 'B11_IND_or_equal - B11_equal', 'B11_lsml - B11_equal', 'B11_lsml - ct7']
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
for ax, ep, title in [(axes[0], 'prm_within_auc', 'PRMBench within-AUC'), (axes[1], 'prmscore_answer_z_q80', 'PRMScore (answer-z, q80)')]:
    d = C[(C.endpoint == ep) & C.contrast_id.isin(want)].set_index('contrast_id').reindex(want).reset_index(); y = np.arange(len(d))
    ax.errorbar(d.delta, y, xerr=[d.delta - d.ci95_lo, d.ci95_hi - d.delta], fmt='o', color='#1f6f8b', capsize=3)
    for i, r in d.iterrows():
        if r.primary: ax.errorbar([r.delta], [i], xerr=[[r.delta - r.ci_adj_lo], [r.ci_adj_hi - r.delta]], fmt='none', ecolor='#b5452b', capsize=5, lw=.8); ax.text(r.ci95_hi, i, '  primary', va='center', fontsize=7, color='#b5452b')
    ax.axvline(0, color='k', lw=.8); ax.set_yticks(y); ax.set_yticklabels(d.contrast_id, fontsize=8); ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x', alpha=.3)
fig.suptitle('Paired PRMBench source-group bootstrap, 100,000 draws; red = Bonferroni (K=4) on primary contrasts', fontsize=9); fig.tight_layout(); fig.savefig(OUT / 'FIG2_contrasts.png', dpi=150); plt.close(fig)
# FIG 3: weights on the enlarged bank
W = np.mean([f['weights'] for f in fits if f['bank'] == 'B11_IND_lf'], 0); names = [f for f in fits if f['bank'] == 'B11_IND_lf'][0]['members']
fig, ax = plt.subplots(figsize=(13, 4.4)); x = np.arange(len(W))
ax.bar(x, W, color=['#1f77b4'] * 11 + ['#d62728'] * 21); ax.axhline(0, color='k', lw=.6); ax.set_xticks(x); ax.set_xticklabels([n.replace('lf__hist_', '').replace('lf__', '').replace('entropy_rolling_', 'H.') for n in names], rotation=75, fontsize=7)
ax.set_ylabel('L-SML weight (mean of 5 folds)'); ax.set_title('where L-SML puts its weight on bank11 + 21 independent channels (blue = bank11, red = independent)'); ax.grid(axis='y', alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG3_weights.png', dpi=150); plt.close(fig)
def img(name): return 'data:image/png;base64,' + base64.b64encode((OUT / name).read_bytes()).decode()
def cell(r, adj):
    if r is None: return '<td>—</td>'
    out = r.ci95_lo > 0 or r.ci95_hi < 0; cls = ' class="pos"' if out and r.delta > 0 else ' class="neg"' if out else ''
    s = f'<td{cls}>{sgn(r.delta)}<br><span class="ci">[{sgn(r.ci95_lo)}, {sgn(r.ci95_hi)}]</span>'
    if adj and not pd.isna(r.ci_adj_lo): s += f'<br><span class="ci">Bonf. [{sgn(r.ci_adj_lo)}, {sgn(r.ci_adj_hi)}]</span>'
    return s + '</td>'
def crow(cid):
    a = cr(cid, 'prm_within_auc'); b = cr(cid, 'prmscore_answer_z_q80')
    return '' if a is None else f'<tr><td class="lbl">{cid}{" · <b>ראשי</b>" if bool(a.primary) else ""}</td>{cell(a, bool(a.primary))}{cell(b, bool(a.primary))}</tr>'
sb = ''.join(f'<tr><td class="lbl">{BL[b]} · {AL[a]}</td><td>{num(est(f"{b}_{a}", "within_auc"))}</td><td>{num(est(f"{b}_{a}", "prmscore_answer_z_q80"))}</td><td>{pct(est(f"{b}_{a}", "sla", "macro8_context"))}%</td></tr>' for b in BANKS for a in ARMS)
sb += f'<tr><td class="lbl">CT7 (עוגן קפוא)</td><td>{num(est("ct7", "within_auc"))}</td><td>{num(est("ct7", "prmscore_answer_z_q80"))}</td><td>{pct(est("ct7", "sla", "macro8_context"))}%</td></tr>'
prim = ''.join(crow(c) for c in ['B11_IND_lf_lsml - B11_lsml', 'B11_IND_lf_lsml - B11_IND_lf_equal'])
sec = ''.join(crow(c) for c in ['B11_IND_or_lsml - B11_lsml', 'B11_IND_or_lsml - B11_IND_or_equal', 'B11_IND_or_lsml - B11_IND_lf_lsml', 'B11_IND_lf_equal - B11_equal', 'B11_IND_or_equal - B11_equal', 'B11_lsml - B11_equal', 'B11_IND_lf_lsml - B11_IND_lf_group_equal', 'B11_lsml - ct7', 'B11_IND_lf_lsml - ct7'])
irows = ''.join(f'<tr><td class="lbl">{c}</td><td>{sgn(S.loc[c, "r_level_marginal"], 3)}</td><td>{num(S.loc[c, "within_auc_single"], 3)}</td><td>{"+" if D["lf_sign"][c] > 0 else "−"}</td><td>{"+" if D["oracle_sign"][c] > 0 else "−"}</td><td>{sgn(float(W[11 + IND.index(c)]), 3)}</td></tr>' for c in IND)
a1 = cr('B11_IND_lf_lsml - B11_lsml', 'prm_within_auc'); a2 = cr('B11_IND_lf_lsml - B11_IND_lf_equal', 'prm_within_auc'); o1 = cr('B11_IND_or_lsml - B11_lsml', 'prm_within_auc')
outcome = 'supported' if (a1.ci_adj_lo > 0 and a2.ci_adj_lo > 0) else 'unsupported' if (a1.ci_adj_hi < 0 or a2.ci_adj_hi < 0) else 'inconclusive / mixed'
blk = D['ind_block_weight_per_fold']['B11_IND_lf']
preds = [('P1 שחזור מדויק', D['replay']['replay_exact'], f'{D["replay"]["max_abs_diff_continuous"]:.1e}'),
         ('P2 ‏L-SML מנצח ממוצע בשני הבנקים המורחבים', a2.ci95_lo > 0 and cr('B11_IND_or_lsml - B11_IND_or_equal', 'prm_within_auc').ci95_lo > 0, f'{sgn(a2.delta)} / {sgn(cr("B11_IND_or_lsml - B11_IND_or_equal", "prm_within_auc").delta)}'),
         ('P3 כיוון נטול תוויות מסכים עם האורקל בפחות מ-15 מתוך 21', D['orientation_agreement'] < 15, f'{D["orientation_agreement"]}/21'),
         ('P4 האורקל מול בנק11 L-SML בתוך ±0.005', abs(o1.delta) <= .005, f'{sgn(o1.delta)} [{sgn(o1.ci95_lo)}, {sgn(o1.ci95_hi)}]'),
         ('P5 הבנק המורחב עם כיוון נטול תוויות אינו עולה על בנק11 L-SML', a1.delta <= 0, sgn(a1.delta))]
prow = ''.join(f'<tr><td class="lbl">{a}</td><td class="{"pos" if b else "neg"}">{"התקיים" if b else "לא התקיים"}</td><td>{c}</td></tr>' for a, b, c in preds)
html = f'''<title>Independent Weak Channels on PRMBench</title>
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
<div class="eyebrow">indbank_lsml_prmbench_v1 · {RUN_ID} · PRMBench first · development data only</div>
<h1>ערוצים חלשים ובלתי תלויים באנטרופיה: האם הם מרימים את L-SML</h1>
<p class="lede">ההשערה של עומרי: לא להוסיף ערוצים לפי משפחות אלא דווקא את החלשים שהטעויות שלהם בלתי תלויות באנטרופיה, כפי שתיאוריית SML דורשת. סינון נטול תוויות של 52 ערוצים מצא 21 כאלה. תוצאה: <b>{outcome}</b>. הבנק המורחב תחת L-SML מול בנק11 L-SML: {sgn(a1.delta)} ({"תומך" if a1.ci_adj_lo > 0 else "שלילי" if a1.ci_adj_hi < 0 else "לא מכריע"}); מול ממוצע על אותו בנק: {sgn(a2.delta)}. עם כיוון אורקל התוצאה של L-SML זהה בדיוק (המשקלים חתומים), והאורקל אפילו מזיק לממוצע.</p>
<div class="verdict">
<div><b>{status["status"]}</b><span>{status["fits"]} התאמות, {status["failures"]} כשלים, שחזור {"מדויק" if D["replay"]["replay_exact"] else "לא מדויק"}; {timing["total_s"] / 60:.0f} דקות CPU</span></div>
<div><b>{num(est("B11_lsml", "within_auc"))} → {num(est("B11_IND_lf_lsml", "within_auc"))}</b><span>within-AUC של L-SML, בנק11 מול בנק11+21. ממוצע: {num(est("B11_equal", "within_auc"))} → {num(est("B11_IND_lf_equal", "within_auc"))}</span></div>
<div><b>{np.mean(blk):.2f}</b><span>חלק המשקל ש-L-SML נותן ל-21 הערוצים הבלתי תלויים (ממוצע על folds); דרגה אפקטיבית {D["effective_rank_prm_steps"]["B11"]:.1f} → {D["effective_rank_prm_steps"]["B11_IND_lf"]:.1f}</span></div>
</div>
<h2>הסינון: מי בלתי תלוי באנטרופיה</h2>
<figure><img src="{img('FIG1_pool.png')}" alt="pool"></figure>
<p class="note">52 ערוצים ברמת הצעד (Top10, נרמול לכל תשובה): בנק11, שבעת זרמי הטוקן של CT7, ארבע קריאות צורה של q15_H1, שטף Mind-the-Gap, ו-29 הזרמים ההיסטוריים (ספקטרליים, STFT, אנטרופיית תמורות, Hurst, CUSUM, שונות בחלון על אנטרופיה, surprisal ואנרגיה). כלל הבחירה נטול תוויות: |קורלציה עם ממוצע משפחת הרמה| < 0.35 ולא עותק של ערוץ בבנק11. ה-AUC בציר האנכי הוא אבחון בלבד. כל הערוצים החזקים הם אנטרופיה במסווה; כל הבלתי תלויים חלשים.</p>
<div class="tw"><table><thead><tr><th>ערוץ</th><th>r עם הרמה</th><th>AUC לבד</th><th>כיוון נטול תוויות</th><th>כיוון אורקל</th><th>משקל L-SML</th></tr></thead><tbody>{irows}</tbody></table></div>
<p class="note">הכיוון נטול התוויות = סימן הקורלציה עם משפחת הרמה; לערוצים האלה הוא כמעט מטבע. האורקל מסכים איתו על {D["orientation_agreement"]} מתוך 21.</p>
<h2>לוח התוצאות</h2>
<div class="tw"><table><thead><tr><th>בנק · זרוע</th><th>PRMB within-AUC</th><th>PRMScore</th><th>PB SLA (הקשר)</th></tr></thead><tbody>{sb}</tbody></table></div>
<h2>ההשוואות הראשיות (משפחה של 4)</h2>
<div class="tw"><table><thead><tr><th>השוואה</th><th>PRMB within-AUC</th><th>PRMScore</th></tr></thead><tbody>{prim}</tbody></table></div>
<figure><img src="{img('FIG2_contrasts.png')}" alt="contrasts"></figure>
<h3>משניות</h3>
<div class="tw"><table><thead><tr><th>השוואה</th><th>PRMB within-AUC</th><th>PRMScore</th></tr></thead><tbody>{sec}</tbody></table></div>
<h2>איפה L-SML שם את המשקל</h2>
<figure><img src="{img('FIG3_weights.png')}" alt="weights"></figure>
<h2>התחזיות שהוקפאו</h2>
<div class="tw"><table><thead><tr><th>תחזית</th><th>תוצאה</th><th>מספר</th></tr></thead><tbody>{prow}</tbody></table></div>
<h2>איך לקרוא את זה</h2>
<div class="next">
<p><b>מה קרה בפועל.</b> L-SML על הבנק המורחב מעביר כרבע מהמשקל ל-21 הערוצים החלשים, ובאותה תנועה מוריד את המשקל של הערוצים החזקים. התוצאה יורדת ב-{abs(100 * a1.delta):.1f} נקודות מול בנק11, ו-L-SML נופל {abs(100 * a2.delta):.1f} נקודות מתחת לממוצע פשוט על אותו בנק. הממוצע עצמו כמעט לא זז ({sgn(cr("B11_IND_lf_equal - B11_equal", "prm_within_auc").delta)}): 21 ערוצים חלשים בממוצע של 32 הם רעש שמתקזז.</p>
<p><b>הכיוון אינו הצוואר.</b> עם כיוון אורקל L-SML נותן ציון זהה בדיוק (המשקלים הנלמדים חתומים, ולכן היפוך קלט מתקזז). כלומר ההשערה נופלת ללא קשר לבעיית הכיוון.</p>
<p><b>למה SML לא עובד כאן.</b> התנאי של SML הוא אי־תלות מותנית <em>ודיוק מעל מקרי</em> לכל מסווג. הערוצים הבלתי תלויים באנטרופיה עומדים בתנאי הראשון ונכשלים בשני: AUC של 0.52 עד 0.65 בודד, ועל רמת הצעד זה קרוב למטבע. אומדן האמינות של L-SML (מהקווריאנס) מזהה אותם כ"בלתי תלויים" ומתגמל אותם, אבל אין להם מספיק אות שיצדיק את המשקל. בכיוון ההפוך מהאפקט של בנק11: שם L-SML השתיק ערוצים מזיקים, כאן הוא מגביר ערוצים ריקים.</p>
</div>
<p class="note">קבצים: <code>results/indbank_lsml_prmbench_v1/{RUN_ID}/</code>, סינון ב-<code>POOL_STRUCTURE.csv</code>, פרוטוקול קפוא ב-<code>PROTOCOL.json</code>.</p>
</main>
'''
(OUT / 'REPORT_HE.html').write_text(html, encoding='utf8'); print('written', len(html) // 1024, 'KB', outcome)
