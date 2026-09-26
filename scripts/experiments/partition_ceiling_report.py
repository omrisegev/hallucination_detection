"""partition_ceiling_prmbench_v1 figure + Hebrew findings page (no recomputation)."""
from pathlib import Path
import base64, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260927'
OUT = ROOT / 'results/partition_ceiling_prmbench_v1' / RUN_ID
M = pd.read_csv(OUT / 'METRICS.csv'); C = pd.read_csv(OUT / 'CONTRASTS.csv').drop_duplicates(['contrast_id', 'endpoint'])
D = json.loads((OUT / 'DIAGNOSTICS.json').read_text(encoding='utf8'))
status = json.loads((OUT / 'RUN_STATUS.json').read_text(encoding='utf8')); timing = json.loads((OUT / 'TIMING.json').read_text(encoding='utf8'))
Z = np.load(OUT / 'EXHAUSTIVE_AUC.npz'); AUC = Z['auc']; NAMES = [str(x) for x in Z['names']]
sel = [json.loads(l) for l in (OUT / 'SELECTION_LOG.jsonl').read_text(encoding='utf8').splitlines() if json.loads(l)['arm'] == 'profile_selected']
def est(m, metric, stratum='all'):
    r = M[(M.method == m) & (M.metric == metric) & (M.stratum == stratum)]; return float(r.estimate.iloc[0]) if len(r) else np.nan
def num(x, d=4): return '—' if pd.isna(x) else f'{x:.{d}f}'
def sgn(x, d=4): return '—' if pd.isna(x) else f'{x:+.{d}f}'
Cp = C.set_index(['contrast_id', 'endpoint'])
def cr(c, ep='prm_within_auc'): return Cp.loc[(c, ep)] if (c, ep) in Cp.index else None

# ---------------- FIG: the whole distribution with the arms marked
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.hist(AUC, bins=400, color='#c9d3cf', edgecolor='none')
marks = [('worst partition', D['worst']['auc_all'], '#b5452b'),
         ('declared 5/3/3', est('declared_equal', 'within_auc'), '#b5452b'),
         ('random partition (median)', float(np.median(AUC)), '#8a8a8a'),
         ('equal (1/11 each)', est('equal', 'within_auc'), '#1f77b4'),
         ('discovered L-SML K=6', est('lsml', 'within_auc'), '#1f77b4'),
         ('energy_level alone', est('energy_level_alone', 'within_auc'), '#7a5ea8'),
         ('CT7 anchor', est('ct7', 'within_auc'), '#444'),
         ('label-selected profile (held out)', est('profile_selected', 'within_auc'), '#2e8b57'),
         ('exhaustive ceiling', D['ceiling']['auc_all'], '#2e8b57')]
ymax = ax.get_ylim()[1]
for i, (lbl, v, col) in enumerate(marks):
    ax.axvline(v, color=col, lw=1.6 if 'ceiling' in lbl or 'selected' in lbl else 1.1, ls='-' if col != '#8a8a8a' else '--')
    ax.text(v, ymax * (0.96 - .105 * i), f' {lbl}  {v:.4f}', color=col, fontsize=8, va='top', rotation=0)
ax.set_xlabel('PRMBench within-answer AUROC'); ax.set_ylabel('number of partitions')
ax.set_title(f"every block-equal partition of the 11-channel bank: {len(AUC):,} distinct weight vectors from 678,570 set partitions")
ax.set_xlim(0.55, 0.79); ax.grid(axis='y', alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'FIG1_distribution.png', dpi=150); plt.close(fig)


def img(nm): return 'data:image/png;base64,' + base64.b64encode((OUT / nm).read_bytes()).decode()
def cell(r, adj):
    if r is None: return '<td>—</td>'
    out = r.ci95_lo > 0 or r.ci95_hi < 0; cls = ' class="pos"' if out and r.delta > 0 else ' class="neg"' if out else ''
    s = f'<td{cls}>{sgn(r.delta)}<br><span class="ci">[{sgn(r.ci95_lo)}, {sgn(r.ci95_hi)}]</span>'
    if adj and not pd.isna(r.ci_adj_lo): s += f'<br><span class="ci">Bonf. [{sgn(r.ci_adj_lo)}, {sgn(r.ci_adj_hi)}]</span>'
    return s + '</td>'
def crow(c):
    a = cr(c); b = cr(c, 'prmscore_answer_z_q80')
    return '' if a is None else f'<tr><td class="lbl">{c}{" · <b>ראשי</b>" if bool(a.primary) else ""}</td>{cell(a, bool(a.primary))}{cell(b, bool(a.primary))}</tr>'

AL = {'equal': 'ממוצע, 1/11 לכל ערוץ', 'lsml': 'L-SML, החלוקה שהתגלתה', 'declared_equal': 'ממוצע מאוזן, החלוקה המוצהרת 5/3/3',
      'energy_level_alone': 'energy_level לבדו', 'profile_selected': 'פרופיל שנבחר לפי המדד (משתמש בתוויות)', 'ct7': 'CT7 (עוגן קפוא)'}
RANK = {'equal': D['rank_of_label_free_arms']['equal']['rank'], 'declared_equal': D['rank_of_label_free_arms']['declared_5_3_3']['rank']}
sb = ''.join('<tr><td class="lbl">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
    AL[a], num(est(a, 'within_auc')), num(est(a, 'prmscore_answer_z_q80')), f"{100*est(a,'sla','macro8_context'):.2f}%",
    f"{RANK[a]:,}" if a in RANK else '—') for a in ['equal', 'lsml', 'declared_equal', 'energy_level_alone', 'profile_selected', 'ct7'])
sb = ('<tr><td class="lbl"><b>התקרה הממצה</b></td><td><b>' + num(D['ceiling']['auc_all']) + '</b></td><td>—</td><td>—</td><td>1</td></tr>' + sb +
      f'<tr><td class="lbl">חלוקה אקראית (חציון)</td><td>{num(float(np.median(AUC)))}</td><td>—</td><td>—</td><td>{len(AUC)//2:,}</td></tr>'
      f'<tr><td class="lbl">החלוקה הגרועה ביותר</td><td>{num(D["worst"]["auc_all"])}</td><td>—</td><td>—</td><td>{len(AUC):,}</td></tr>')
prim = ''.join(crow(c) for c in ['profile_selected - equal', 'profile_selected - lsml', 'profile_selected - ct7', 'profile_selected - null_seed0'])
sec = ''.join(crow(c) for c in ['profile_selected - energy_level_alone', 'energy_level_alone - equal', 'lsml - equal', 'declared_equal - equal', 'lsml - ct7', 'declared_equal - ct7'])
cw = sorted(D['ceiling']['weights'].items(), key=lambda kv: -kv[1])
wr = ''.join(f'<tr><td class="lbl">{k}</td><td>{v:.4f}</td><td>{round(1/(3*v)) if v>0 else "—"}</td></tr>' for k, v in cw)
folds = ''.join(f'<tr><td class="lbl">fold {r["fold"]}</td><td>{r["rank_overall"]:,}</td><td>{r["auc_on_fit_folds"]:.4f}</td><td>{r["auc_on_eval_fold"]:.4f}</td></tr>' for r in sorted(sel, key=lambda r: r['fold']))
p = [('P1 אותו פרופיל נבחר בכל חמשת ה-folds', D['selected_identical_in_all_folds'], 'גדלים [8,2,1] בכולם'),
     ('P2 מנצח ממוצע, רווח Bonferroni ללא אפס', cr('profile_selected - equal').ci_adj_lo > 0, sgn(cr('profile_selected - equal').delta)),
     ('P3 מנצח את ניגוד התוויות המעורבלות', cr('profile_selected - null_seed0').ci_adj_lo > 0, sgn(cr('profile_selected - null_seed0').delta)),
     ('P4 מול CT7 חיובי אך הרווח כולל אפס', cr('profile_selected - ct7').delta > 0 and cr('profile_selected - ct7').ci_adj_lo < 0, sgn(cr('profile_selected - ct7').delta)),
     ('P5 מול energy_level לבדו בין +0.008 ל-+0.018', .008 <= cr('profile_selected - energy_level_alone').delta <= .018, sgn(cr('profile_selected - energy_level_alone').delta))]
prow = ''.join(f'<tr><td class="lbl">{a}</td><td class="{"pos" if b else "neg"}">{"התקיים" if b else "לא התקיים"}</td><td>{c}</td></tr>' for a, b, c in p)

html = f'''<title>Partition Ceiling on PRMBench</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#f6f7f5;--ink:#1c2430;--muted:#5d6874;--rule:#d7dcd9;--card:#eef1ee;--accent:#1f6f8b;--pos:#3f7d4e;--neg:#b5452b;--posbg:#e6f0e8;--negbg:#f6e6e1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}}}
:root[data-theme="dark"]{{--paper:#151a1f;--ink:#e6e9e6;--muted:#9aa5ae;--rule:#2d353c;--card:#1d242a;--accent:#6fb6cf;--pos:#7cc48c;--neg:#e58a70;--posbg:#1d2e22;--negbg:#33221c}}
body{{background:var(--paper);color:var(--ink);font-family:Heebo,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.6;direction:rtl}}
main{{max-width:900px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-weight:700;font-size:30px;line-height:1.2;margin:0 0 6px;text-wrap:balance}}
h2{{font-weight:500;font-size:22px;margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--rule)}}
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
td.lbl{{font-family:Heebo,sans-serif;direction:rtl;white-space:normal;min-width:210px}}
td.pos{{color:var(--pos);background:var(--posbg)}} td.neg{{color:var(--neg);background:var(--negbg)}}
.ci{{font-size:11px;color:var(--muted)}}
figure{{margin:16px 0 4px}} figure img{{width:100%;border:1px solid var(--rule);border-radius:3px;background:#fff}}
.note{{font-size:14px;color:var(--muted);margin:6px 0}}
.next{{background:var(--card);padding:16px 20px;border-radius:4px;border-right:3px solid var(--accent)}}
code{{font-family:"IBM Plex Mono",monospace;font-size:13px;direction:ltr;unicode-bidi:embed}}
@media (max-width:600px){{.verdict{{grid-template-columns:1fr}}}}
</style>
<main>
<div class="eyebrow">partition_ceiling_prmbench_v1 · {RUN_ID} · PRMBench · development, label-selected arm declared</div>
<h1>כל החלוקות האפשריות של בנק 11 הערוצים</h1>
<p class="lede">מדדתי את <b>כל</b> {len(AUC):,} וקטורי המשקל שחלוקה כלשהי של 11 הערוצים יכולה לייצר, ומצאתי איפה הכללים שלנו יושבים. התקרה היא {num(D['ceiling']['auc_all'])}, ממוצע פשוט הוא {num(est('equal','within_auc'))} במקום ה-{RANK['equal']:,}, ו<b>החלוקה המוצהרת לפי מקור יושבת ברבעון התחתון</b>, מקום {RANK['declared_equal']:,}.</p>
<div class="verdict">
<div><b>{num(D['ceiling']['auc_all'])}</b><span>התקרה הממצה. המבנה המנצח: energy_level לבדו עם שליש מהקול, chosen_surprisal ו-bocpd_p0 עם שישית כל אחד, שמונת האחרים חולקים את השליש האחרון</span></div>
<div><b>{num(est('profile_selected','within_auc'))}</b><span>הפרופיל שנבחר לפי המדד על שלושת folds האימון ונמדד על ה-fold שהוחזק. אותו פרופיל בדיוק ב-5 מתוך 5</span></div>
<div><b>{sgn(cr('profile_selected - ct7').delta)}</b><span>מול CT7, רווח [{sgn(cr('profile_selected - ct7').ci95_lo)}, {sgn(cr('profile_selected - ct7').ci95_hi)}] שכולל אפס. מול ממוצע {sgn(cr('profile_selected - equal').delta)}, מול L-SML {sgn(cr('profile_selected - lsml').delta)}</span></div>
</div>
<h2>העובדה שמאפשרת למצות את המרחב</h2>
<p>תחת ממוצע מאוזן המשקל של ערוץ הוא <code>1/(K · |הקבוצה שלו|)</code>. כלומר הציון תלוי <b>רק בגודל הקבוצה של כל ערוץ</b> ולא בזהות מי נמצא עם מי. שתי חלוקות שונות לגמרי עם אותה מפת גדלים הן אותה זרוע בדיוק, ולכן 678,570 החלוקות של 11 ערוצים מתמוטטות ל-{len(AUC):,} וקטורי משקל שאפשר למנות את כולם. המשמעות הרחבה יותר: ממוצע מאוזן אינו "מסנן ואז מאחד", הוא רק מוריד משקל לערוצים שנמצאים בקבוצה גדולה.</p>
<h2>איפה כל כלל יושב</h2>
<figure><img src="{img('FIG1_distribution.png')}" alt="distribution"></figure>
<div class="tw"><table><thead><tr><th>שיטה</th><th>PRMB within-AUC</th><th>PRMScore</th><th>PB SLA (הקשר)</th><th>דירוג מתוך {len(AUC):,}</th></tr></thead><tbody>{sb}</tbody></table></div>
<p class="note">{D['n_profiles_above_equal']:,} חלוקות מנצחות ממוצע פשוט, כלומר 34%. אבל חלוקה <b>אקראית</b> נותנת {num(float(np.median(AUC)))}, כלומר פחות מממוצע. הכלל האוטומטי של L-SML נמצא ב-20% העליונים; החלוקה המוצהרת נמצאת ברבעון התחתון, מתחת לאקראית.</p>
<h2>המבנה המנצח</h2>
<div class="tw"><table><thead><tr><th>ערוץ</th><th>משקל</th><th>גודל הקבוצה</th></tr></thead><tbody>{wr}</tbody></table></div>
<p class="note">בשמונת הפרופילים הטובים ביותר, בכולם, energy_level עומד לבדו עם שליש מהקול. זו בדיוק ההפרדה בין "מסננים בממוצע" (שמונת ערוצי משפחת האנטרופיה, נדחסים ל-1/24 כל אחד) לבין "מאחדים" (יחיד ועוד זוג). ו<b>זה בדיוק מה ש-Joint אוסר</b>: <code>fit_joint_lsml</code> דורשת K≥3 וכל קבוצה ≥3, ולכן אינה יכולה לבטא את הפתרון הטוב ביותר.</p>
<h2>האם אפשר למצוא אותו: כן</h2>
<div class="tw"><table><thead><tr><th>fold</th><th>דירוג הפרופיל שנבחר</th><th>AUC על folds האימון</th><th>AUC על ה-fold שהוחזק</th></tr></thead><tbody>{folds}</tbody></table></div>
<p class="note">הבחירה נעשתה על שלושה folds בלבד, והפרופיל הוערך על fold רביעי שלא השתתף. אותו פרופיל נבחר בכל חמשת המקרים, והיה מדורג ראשון בארבעה מהם ושני באחד.</p>
<h2>ההשוואות</h2>
<div class="tw"><table><thead><tr><th>השוואה</th><th>PRMB within-AUC</th><th>PRMScore</th></tr></thead><tbody>{prim}</tbody></table></div>
<div class="tw"><table><thead><tr><th>משניות</th><th>PRMB within-AUC</th><th>PRMScore</th></tr></thead><tbody>{sec}</tbody></table></div>
<h2>התחזיות שהוקפאו</h2>
<div class="tw"><table><thead><tr><th>תחזית</th><th>תוצאה</th><th>מספר</th></tr></thead><tbody>{prow}</tbody></table></div>
<h2>מה זה אומר, ומה זה לא</h2>
<div class="next">
<p><b>התשובה לשאלה "למה לא מצאנו את החלוקה האופטימלית": כי לא חיפשנו אותה.</b> כל כלל שיש לנו בוחר לפי קריטריון אחר מהמטרה. L-SML לפי התאמת מודל בלוקים לקווריאנס, Joint לפי יציבות, והמוצהרת לפי ידע על מקור הערוץ. כשבוחרים לפי המדד עצמו, הבחירה יציבה לחלוטין ועוברת בין folds.</p>
<p><b>אבל הרווח מוגבל.</b> התקרה היא {num(D['ceiling']['auc_all'])} ו-CT7 הוא {num(est('ct7','within_auc'))}. כלומר כל מרחב החלוקות של הבנק הזה מגיע בקושי ל-CT7 ולא עובר אותו: {sgn(cr('profile_selected - ct7').delta)} עם רווח שכולל אפס, ו-{sgn(cr('profile_selected - ct7','prmscore_answer_z_q80').delta)} ב-PRMScore. ב-ProcessBench הפרופיל אף גרוע יותר מממוצע.</p>
<p><b>זו זרוע שמשתמשת בתוויות.</b> הפרופיל נבחר לפי within-AUC על folds האימון. זו בחירה בתוך folds האימון, מותרת ומוצהרת, אבל אינה label-free.</p>
<p><b>פגם בניגוד שבניתי, ואני מדווח עליו.</b> ניגוד "התוויות המעורבלות" נחת בדיוק על המינימום של ההתפלגות ({num(D['worst']['auc_all'])}) בכל שלושת ה-seeds, כי עם קבוצת הזוגות קבועה ערבוב תוויות בתוך התשובה הופך את פונקציית המטרה לשלילת עצמה בקירוב. הוא מוכיח שהבחירה מגיבה חזק לתוויות, אבל הוא <b>אינו</b> ניגוד "בחירה אקראית". התפקיד הזה שייך לחציון ההתפלגות, {num(float(np.median(AUC)))}, שנמצא מתחת לממוצע פשוט.</p>
<p><b>וסייג חיצוני.</b> הקו המקביל הראה שדירוג על המקור אינו מנבא דירוג חיצוני: CT7 מוביל ב-PRMScore על המקור ונמצא אחרון חיצונית ב-Hard2Verify. פרופיל משקלים שנבחר לאופטימום על המקור הוא בדיוק סוג האובייקט שעלול לא לעבור. אין כאן מועמד.</p>
</div>
<p class="note">קבצים: <code>results/partition_ceiling_prmbench_v1/{RUN_ID}/</code>. פרוטוקול קפוא ונדחף לפני הניקוד. {status['fits'] if 'fits' in status else ''} זמן ריצה {timing['total_s']/60:.0f} דקות.</p>
</main>
'''
(OUT / 'REPORT_HE.html').write_text(html, encoding='utf8')
print('written', len(html) // 1024, 'KB')
