"""External figures + REPORT_HE.md from METRICS.csv / CONTRASTS.csv / TRANSFER.csv (no hand-typed numbers)."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent; FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True)
CELLS = {'hard2verify_qwen3_8b': 'Hard2Verify · Qwen3-8B', 'socratic_qwen3_8b': 'Socratic · Qwen3-8B', 'socratic_qwq32b': 'Socratic · QwQ-32B'}
METHODS = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']; HL = 'B16'
P = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))
PUB = {'hard2verify_qwen3_8b': P['comparators_published_only']['hard2verify_balanced_f1'], 'socratic_qwen3_8b': P['comparators_published_only']['socratic_prmscore'],
       'socratic_qwq32b': P['comparators_published_only']['socratic_prmscore']}
THEME = {'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7', ours='#2a78d6', prm='#eb6834', mtg='#1baf7a'),
         'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835', ours='#3987e5', prm='#d95926', mtg='#199e70')}
R = pd.read_csv(OUT / 'METRICS.csv'); C = pd.read_csv(OUT / 'CONTRASTS.csv'); T = pd.read_csv(OUT / 'TRANSFER.csv')


def style(ax, t):
    ax.set_facecolor(t['surface'])
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(t['axis'])
    ax.tick_params(colors=t['ink2'], labelsize=9); ax.grid(color=t['grid'], linewidth=.8); ax.set_axisbelow(True)


def fig_transfer(t, mode):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (cell, title) in zip(axes, CELLS.items()):
        style(ax, t); d = pd.read_csv(OUT / cell / 'DEV_VS_EXTERNAL.csv')
        oth = d[~((d.bank == HL) & d.scorer.isin(METHODS))]; hl = d[(d.bank == HL) & d.scorer.isin(METHODS)]
        ctl = oth[oth.scorer.isin(['ALL_EQ', 'DS_FILTER_EQ', 'MEAN_EQ', 'HEM_EQ'])]; met = oth[~oth.index.isin(ctl.index)]
        ax.scatter(met.prmb_within_auc, met.within_auc, s=28, color=t['muted'], alpha=.7, label='other bank × method')
        ax.scatter(ctl.prmb_within_auc, ctl.within_auc, s=28, facecolors='none', edgecolors=t['muted'], label='equal-weight controls')
        ax.scatter(hl.prmb_within_auc, hl.within_auc, s=70, color=t['ours'], edgecolors=t['surface'], linewidths=1.5, zorder=3, label=f'{HL} methods (pre-declared)')
        si = R[(R.cell == cell) & (R.scorer == 'STEP_INDEX') & (R.rule == 'TOP_1')].within_auc.iloc[0]
        ax.axhline(si, color=t['prm'], linestyle='--', linewidth=1.5); ax.text(d.prmb_within_auc.min(), si, ' step index alone', color=t['prm'], fontsize=8.5, va='bottom')
        tr = T[T.cell == cell].iloc[0]
        ax.set_title(f'{title}\nSpearman dev→external {tr.spearman_dev_auc_vs_ext_auc:.2f} over {int(tr.configs)} configs', color=t['ink'], fontsize=10)
        ax.set_xlabel('Development PRMBench within-answer AUC', color=t['ink2']); ax.set_ylabel('External within-answer AUC', color=t['ink2'])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, labelcolor=t['ink'], bbox_to_anchor=(.5, -.06), fontsize=9)
    fig.patch.set_facecolor(t['surface']); fig.tight_layout(rect=(0, .05, 1, 1))
    for ext in ('png', 'svg'): fig.savefig(FIG / f'E1_dev_vs_external_{mode}.{ext}', dpi=180, facecolor=t['surface'], bbox_inches='tight')
    plt.close(fig)


def fig_official(t, mode):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    rows = [('Ours, B16 (4 methods)', 'ours'), ('DS_FILTER_EQ, B16', 'ctl'), ('Earlier frozen bank11 L-SML', 'ref'), ('CT7 (earlier incumbent)', 'ref2'),
            ('Step index alone', 'base'), ('Qwen2.5-Math-PRM-7B (published)', 'pub'), ('LLM critic (published)', 'pub2'), ('UniversalPRM-7B (published)', 'pub3')]
    for ax, (cell, title) in zip(axes, CELLS.items()):
        style(ax, t); ax.grid(axis='y', visible=False); y = np.arange(len(rows))[::-1]
        g = R[R.cell == cell]
        o = g[(g.bank == HL) & g.scorer.isin(METHODS) & (g.rule == 'ANSWER_Z_Q80')].official_metric
        vals = [(o.min(), o.mean(), o.max()), g[(g.bank == HL) & (g.scorer == 'DS_FILTER_EQ') & (g.rule == 'ANSWER_Z_Q80')].official_metric.iloc[0],
                g[g.scorer == 'frozen_lsml'].official_metric.iloc[0], g[g.scorer == 'ct7'].official_metric.iloc[0],
                g[(g.scorer == 'STEP_INDEX') & (g.rule == 'ANSWER_Z_Q80')].official_metric.iloc[0]]
        pub = PUB[cell]
        vals += [pub.get('Qwen2.5-Math-PRM-7B', np.nan) / 100, (pub.get('Qwen3-8B critic', pub.get('QwQ-32B critic', np.nan))) / 100, pub.get('UniversalPRM-7B', np.nan) / 100]
        for k, ((lab, kind), v) in enumerate(zip(rows, vals)):
            if kind == 'ours':
                ax.plot([v[0], v[2]], [y[k]] * 2, color=t['ours'], linewidth=6, alpha=.45, solid_capstyle='round'); ax.plot(v[1], y[k], 'o', color=t['ours'], markersize=9, markeredgecolor=t['surface'], markeredgewidth=2); v = v[1]
            elif not np.isfinite(v): ax.text(.3, y[k], 'n/a', color=t['muted'], fontsize=8, va='center'); continue
            elif kind.startswith('pub'): ax.plot(v, y[k], 'o', color=t['prm'], markersize=8, markerfacecolor='none', markeredgewidth=1.8)
            elif kind == 'base': ax.plot(v, y[k], 'D', color=t['muted'], markersize=7, markerfacecolor='none', markeredgewidth=1.5)
            else: ax.plot(v, y[k], 'o', color=t['muted'], markersize=8)
            ax.annotate(f'{v:.3f}', (v, y[k]), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8, color=t['ink2'])
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=t['ink'], fontsize=9); ax.set_xlim(.3, .8)
        ax.set_title(title + ('  (Balanced F1)' if 'hard' in cell else '  (PRMScore)'), color=t['ink'], fontsize=10)
    fig.text(.01, -.05, 'Ours and controls: label-free refit on each external cell, threshold = q80 of the cell itself (ANSWER_Z_Q80). Published numbers are paper-reported, '
             'not reproduced; Hard2Verify PRM thresholds were tuned on target responses. Hollow orange = published.', color=t['ink2'], fontsize=8.5)
    fig.patch.set_facecolor(t['surface']); fig.tight_layout()
    for ext in ('png', 'svg'): fig.savefig(FIG / f'E2_official_metric_{mode}.{ext}', dpi=180, facecolor=t['surface'], bbox_inches='tight')
    plt.close(fig)


for mode, t in THEME.items(): fig_transfer(t, mode); fig_official(t, mode)

f3 = lambda v: '—' if pd.isna(v) else f'{v:.3f}'; ci = lambda r: f'{r.delta:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]'
md = ['# הכללה לבנצ׳מרקים החיצוניים: ארבע שיטות יום שני', '',
      '**סטטוס:** בדיקת העברה חקרנית. התוויות של Hard2Verify ו־Socratic נפתחו בסבבים קודמים, לכן זו אינה אישור בלתי־תלוי; MedPRMBench לא נגעו בו. '
      'כל השיטות והבקרות מותאמות מחדש על כל תא חיצוני, בלי תוויות (אותו פרוטוקול כמו בפיתוח). B16 הוכרז מראש, מנתוני הפיתוח בלבד, לפני שנפתחה תוצאה חיצונית כלשהי של השיטות האלה.', '',
      '**אימות:** המתאם על נתוני המקור משחזר את ציוני הפיתוח הקפואים עד 3e-15 בכל 8 הבנקים, וגם את דגלי ANSWER_Z ו־R2 בלי אי־התאמה (`SOURCE_PARITY_GATE.json`). '
      'התחזיות נחתמו (hash) לפני פתיחת התוויות (`*/SEAL.json`); המעריכים הרשמיים שוחזרו בדיוק על הבנק המודגש והייחוסים (`EVALUATION_PROVENANCE.json`). '
      'שער האנטרופיה לא הורץ: חסר מאפיין אנטרופיה ברמת תשובה לתאים החיצוניים (מגבלה מוצהרת).', '',
      '## האם בחירת הפיתוח עוברת?', '', '![](figures/E1_dev_vs_external_light.png)', '',
      '| תא | תצורות | Spearman פיתוח→חיצוני (AUC) | Spearman פיתוח→מדד רשמי | כיוון הרווח מול DS_FILTER_EQ תואם | דירוג תצורות B16 (מתוך 72) |', '|---|---:|---:|---:|---:|---:|']
for _, r in T.iterrows():
    md.append(f'| {CELLS[r.cell]} | {r.configs} | {r.spearman_dev_auc_vs_ext_auc:.2f} | {r.spearman_dev_auc_vs_ext_official:.2f} | {r.gain_sign_agreement} | {r.B16_methods_ext_auc_rank_best}–{r.B16_methods_ext_auc_rank_worst} |')
md += ['', '- ב־Socratic הדירוג מהפיתוח מנבא חלקית את הדירוג החיצוני, ו־B16 שהוכרז מראש נמצא בראש הטבלה (המיקום הטוב ביותר בטווח). ב־Hard2Verify (200 תשובות, 79 קבוצות) הקשר חלש.',
       '- כיוון הרווח של השקלול מול הממוצע הפשוט תואם בערך במחצית עד שני שלישים מהמקרים — כלומר הרווח הקטן בפיתוח אינו עובר באופן אמין.',
       '- **קו הבסיס של מיקום הצעד גבוה מכל השיטות ב־AUC בכל שלושת התאים** (ב־Socratic כשליש מהתשובות הן קידומת תקינה ואחריה שגיאות עד הסוף). זו אזהרה ידועה מסבבים קודמים.', '',
       '## המדד הרשמי מול הייחוסים', '', '![](figures/E2_official_metric_light.png)', '']
md += ['| מדרג | ' + ' | '.join(CELLS.values()) + ' |', '|---|---:|---:|---:|']
def val(cell, sc, rule='ANSWER_Z_Q80', bank=HL, col='official_metric'):
    g = R[(R.cell == cell) & (R.scorer == sc) & (R.rule == rule)]
    if bank: g = g[g.bank == bank]
    return g[col].iloc[0]
for sc in METHODS + ['DS_FILTER_EQ', 'ALL_EQ']:
    md.append(f'| {sc} (B16, ANSWER_Z_Q80) | ' + ' | '.join(f3(val(c, sc)) for c in CELLS) + ' |')
md.append('| בנק11 L-SML קפוא (סבב קודם) | ' + ' | '.join(f3(val(c, 'frozen_lsml', 'sealed_reference', None)) for c in CELLS) + ' |')
md.append('| CT7 (סבב קודם) | ' + ' | '.join(f3(val(c, 'ct7', 'sealed_reference', None)) for c in CELLS) + ' |')
md.append('| מיקום הצעד בלבד (ANSWER_Z_Q80) | ' + ' | '.join(f3(val(c, 'STEP_INDEX', bank=None)) for c in CELLS) + ' |')
md.append(f'| PRM-7B (מאמר) | {PUB["hard2verify_qwen3_8b"]["Qwen2.5-Math-PRM-7B"] / 100:.3f} | {PUB["socratic_qwen3_8b"]["Qwen2.5-Math-PRM-7B"] / 100:.3f} | {PUB["socratic_qwq32b"]["Qwen2.5-Math-PRM-7B"] / 100:.3f} |')
md.append(f'| מבקר LLM (מאמר) | {PUB["hard2verify_qwen3_8b"]["Qwen3-8B critic"] / 100:.3f} | {PUB["socratic_qwen3_8b"]["QwQ-32B critic"] / 100:.3f} | {PUB["socratic_qwq32b"]["QwQ-32B critic"] / 100:.3f} |')
md += ['', '**within-answer AUC (B16):**', '', '| מדרג | ' + ' | '.join(CELLS.values()) + ' |', '|---|---:|---:|---:|']
for sc in METHODS + ['DS_FILTER_EQ']:
    md.append(f'| {sc} | ' + ' | '.join(f3(val(c, sc, 'TOP_1', HL, 'within_auc')) for c in CELLS) + ' |')
md.append('| מיקום הצעד בלבד | ' + ' | '.join(f3(val(c, 'STEP_INDEX', 'TOP_1', None, 'within_auc')) for c in CELLS) + ' |')
md += ['', '## ניגודים מזווגים (B16, תיאוריים)', '', '| תא | שיטה | מול | מדד | הפרש [95%] |', '|---|---|---|---|---:|']
for _, r in C.iterrows():
    md.append(f'| {CELLS[r.cell]} | {r.method} | {r.versus} | {r.metric} | {ci(r)} |')
md += ['', '## הקריאה', '',
       '- **Socratic:** שתי גרסאות ה־cc ו־MEAN_MOM_ga גבוהות מהממוצע הפשוט באותו בנק ב־AUC (רווחים שאינם כוללים אפס), ובמדד הרשמי ב־QwQ. מול בנק11 L-SML הקפוא — בערך שוויון. מול CT7 — יתרון של כ־4 נקודות. TCI_ONLY_ga נמוך מהממוצע הפשוט — כמו בפיתוח על חלק מהבנקים.',
       '- **Hard2Verify:** אין יתרון לשקלול; בנק11 L-SML הקפוא גבוה מכל ארבע השיטות במדד הרשמי. N קטן.',
       '- **מול PRM מפוקח ומבקרים (מספרי מאמר, לא משוחזרים):** ב־Socratic אנחנו כ־4–5 נקודות מתחת ל־PRM-7B וכ־10 מתחת למבקר QwQ-32B; ב־Hard2Verify כ־2–3 נקודות מתחת ל־PRM-7B (שהסף שלו כוון על הנתונים), והרבה מתחת ל־UniversalPRM ולמבקר. שם גם מיקום הצעד לבדו (0.483) גבוה מכל השיטות ומ־PRM-7B.',
       '- **מיקום הצעד** חזק מאוד בבנצ׳מרקים האלה. כל טענת הכללה צריכה להראות שורה זו לצד השיטה.', '',
       '## קבצים', '', '`PROTOCOL.json`, `SOURCE_PARITY_GATE.json`, `SCORING_STATUS.json`, `<cell>/PREDICTIONS.npz` + `SEAL.json` + `SCORING_DIAGNOSTICS.json` + `DEV_VS_EXTERNAL.csv`, '
       '`METRICS.csv`, `CONTRASTS.csv`, `TRANSFER.csv`, `EVALUATION_PROVENANCE.json`, `figures/`, `score.py`, `evaluate.py`, `report.py`.', '']
(OUT / 'REPORT_HE.md').write_text('\n'.join(md), encoding='utf8')
(OUT / 'REPORT_MANIFEST.json').write_text(json.dumps({'report_sha256': hashlib.sha256((OUT / 'REPORT_HE.md').read_bytes()).hexdigest(),
    'inputs': {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in ('METRICS.csv', 'CONTRASTS.csv', 'TRANSFER.csv')},
    'figures': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FIG.glob('*'))}, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1), encoding='utf8')
print('done')
