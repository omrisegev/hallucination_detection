"""Derive base/build_base.py from the unchanged v17 builder (base/build_v17.py).

Only slides 12-13 change: the result renderers and their captions/notes, which in v17 described a pending
experiment. Every other slide, chart and number is produced by the v17 code unchanged. Each replacement is
asserted to match exactly once.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / 'base/build_v17.py').read_text(encoding='utf8')


def sub(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old[:80])
    src = src.replace(old, new)


# base/ sits one level deeper than the v17 directory.
sub("ROOT = HERE.parents[1]", "ROOT = HERE.parents[2]")

# --- slide 12 result: coloured bars per arm type + table with the rule, plus L-SML weights and paired CIs.
sub("""    rows=U['boosting']['rows'];fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,key in zip(axs,['prmscore','auc']):
        ax.barh(range(len(rows)),[r[key] for r in rows],color=COL['LSML_CONT']);ax.set_yticks(range(len(rows)),[r['label'] for r in rows],fontsize=8);ax.invert_yaxis();ax.set_xlabel('PRMScore ×100' if key=='prmscore' else 'Within-answer AUC');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();return chart('prm_lsml_update',fig,U['boosting']['source'])""",
"""    rows=U['boosting']['rows'];fig,axs=plt.subplots(1,2,figsize=(11,3.9))
    kind={'PRM_native':COL['PRM'],'PRM_Q80':'#e8907f','LSML3':'#8fb7d6','LSML_CONT':COL['LSML_CONT']}
    colors=[kind.get(r['arm'],'#aab4bf') for r in rows]
    for ax,key,lo in zip(axs,['prmscore','auc'],[50,.6]):
        vals=[r[key] for r in rows];ax.barh(range(len(rows)),vals,color=colors)
        for j,v in enumerate(vals):ax.text(v,j,f' {v:.2f}' if key=='prmscore' else f' {v:.3f}',va='center',fontsize=8)
        ax.set_yticks(range(len(rows)),[r['label'] for r in rows],fontsize=8);ax.invert_yaxis();ax.set_xlim(lo,max(vals)*1.06 if key=='prmscore' else .9)
        ax.set_xlabel('Official PRMScore ×100' if key=='prmscore' else 'Within-answer AUC');ax.grid(axis='x',alpha=.15)
    fig.tight_layout()
    d=U['boosting']['lsml_diagnostics'];w=d['LSML_CONT']['weights_normalized'];w3=d['LSML3']['weights_normalized'];c=U['boosting']['paired_ci_7b']
    def iv(x,s=2):return f'{x[0]:+.{s}f} [{x[1]:+.{s}f}, {x[2]:+.{s}f}]'
    tab=table(['Arm','Flag rule','PRMScore','AUC'],[[e(r['label']),e(r['rule']),f"{r['prmscore']:.2f}",f"{r['auc']:.3f}"] for r in rows])
    note=('<p class="caption">L-SML weights on PRM + 3 signals (K=2: PRM alone, the 3 signals together): PRM '+f"{w['prm_risk']:.2f}"+
          ', energy '+f"{w['energy_level']:.2f}"+', CUSUM '+f"{w['hist_entropy_cusum_abs_series']:.2f}"+', realized_drv '+f"{w['realized_drv']:.2f}"+
          '. Paired source-group 95% intervals (707 groups, 2,000 draws): PRM+3 minus native PRM '+iv(c['LSML_CONT - PRM_native']['prmscore'])+
          ' PRMScore, '+iv(c['LSML_CONT - PRM_native']['within_auc'],3)+' AUC; PRM+3 minus 3-signal L-SML '+iv(c['LSML_CONT - LSML3']['prmscore'])+
          ' PRMScore. The 3-signal-only fit is degenerate: with three views L-SML can only form K=2 and returns the same fixed weights ('+
          ', '.join(f'{v:.2f}' for v in w3.values())+') for every PRM, so it is shown as a weak control, not as a tuned fusion.</p>')
    return chart('prm_lsml_update',fig,U['boosting']['source'])+tab+note""")

sub("""'<p class="caption">The three features were selected using labels on Qwen-PRM-7B. L-SML fitting must exclude evaluation labels. Keep feature selection and evaluation separate. The PRM itself was trained with supervision.</p>',
    'זו השאלה החדשה: האם מידע שכבר קיים בתוך אותו PRM מוסיף על ה־head שלו באמצעות L-SML? להצביע על שלושה פיצ׳רים ועל פלט ה־head מאותו checkpoint. שומרים מקום לתוצאה של קלוד. 69.38 היה ממוצע עם משקלים קבועים; הוא אינו תוצאה של הניסוי המוצג כאן ולכן הוסר.',
    'PRM_UPDATE.json; PRM boosting branch small-bank selection; revised same-checkpoint L-SML result pending',
    'התרשים הוא מפרט ניסוי. הגרף העתידי צריך להראות baseline, כל פיצ׳ר, fusion של שלושת הפיצ׳רים ו־fusion של ארבעת הסיגנלים כולל PRM.',
    'אין כרגע תוצאה מאומתת של L-SML לניסוי הזה. בחירת הפיצ׳רים השתמשה בתוויות ולכן אין לתאר את כל התהליך כ־unsupervised.'))""",
"""'<p class="caption">Provenance: the three signals were chosen with labels on this same Qwen-PRM-7B (all five source-group folds picked the same three). The L-SML fit itself uses no labels. The PRM head was trained with supervision. PRMBench is previously inspected development data.</p>',
    'השאלה: האם מידע שכבר נמצא בתוך אותו PRM מוסיף על ה־head שלו?<br>תשובה: כן.<br>PRM לבד: 65.5 PRMScore.<br>L-SML של PRM ושלושת הסיגנלים: 69.5.<br>הפער: 4.0 נקודות. רווח סמך 95%: 3.3 עד 4.7.<br>שלושת הסיגנלים לבד, בלי PRM: 66.6. כלומר הרווח בא מהשילוב, לא מהסיגנלים לבד.<br>L-SML נתן ל־PRM משקל של כ־0.37, הכי גבוה מבין ארבעת הסיגנלים.<br>זהירות: שלושת הסיגנלים נבחרו עם תוויות על אותו מודל.',
    'PRM_UPDATE.json; sources/eval_slide12 (AIRCC job 284086, lsml_slide12.py = lsml_four.py of job 283967 + single-signal and 3-signal arms)',
    'פאנל שמאל: PRMScore רשמי. פאנל ימין: AUC בתוך תשובה.<br>אדום: PRM לבד. אפור: כל סיגנל לבד. תכלת: L-SML של שלושת הסיגנלים. כחול: L-SML של ארבעתם.<br>כלל ההחלטה כתוב בטבלה ליד כל שורה.',
    'בחירת הסיגנלים השתמשה בתוויות של PRM-7B, ולכן במודל הזה זה לא מבחן נקי.<br>ההתאמה של L-SML עצמה לא משתמשת בתוויות.<br>ה־PRM עצמו מפוקח.'))""")

# --- slide 13: notes for a completed result (renderer already handles status=complete).
sub("""'ההכללה הזו בין מודלי PRM שונה מההכללה בין datasets. אם קלוד יסיים עד הפגישה נציג תוצאות מתאימות. אם לא, נשאיר סטטוס מתוארך. הריצות שתוכננו קודם כללו ממוצע קבוע, ולכן יש לוודא יישור מתכון לפני שמייחסים להן הצלחת L-SML.',
    'docs/experiments/PRM_BOOSTING_GENERALIZATION_V1.md @ claude/prm-boosting-generalization-v1; PRM_UPDATE.json; status reported by Omri 14 Oct',
    'המקום השמור מיועד להשוואה של פלט ה־head מול fusion של סיגנלים פנימיים באותו מודל. יש להציג גם את כלל ההחלטה והמדד בכל מודל.',
    'הסטטוס נשען על דיווח המשתמש, לא על polling של AIRCC. מתכון L-SML עדיין מחכה לתוצאות; לא מציגים ממוצע קבוע כתוצאה שלו.'))""",
"""'כאן בודקים מודלי PRM אחרים, על אותו PRMBench.<br>לכל מודל: ה־head שלו ושלושת הסיגנלים מה־backbone שלו.<br>אותם שלושה סיגנלים ואותו L-SML, בלי בחירה מחדש.<br>L-SML עדיף על ה־PRM בכל חמשת המודלים, וכל הרווחים לא כוללים 0.<br>הרווח גדל ככל שה־PRM חלש יותר: 72B +1.7, PURE +1.4, RLHFlow +9.1, MathShepherd +16.7.<br>L-SML מוריד לבד את משקל ה־PRM כשהוא חלש: 0.37 במודלי Qwen, 0.23 ב־MathShepherd.',
    'PRM_UPDATE.json; sources/eval_slide12 (AIRCC job 284086); gates G1-G3 in results/prm_boosting_generalization_v1 @ claude/prm-boosting-generalization-results-v1',
    'נקודה אדומה: ה־PRM עם הכלל שלו. נקודה כחולה: L-SML ב־ANSWER_Z_Q80.<br>פאנל שמאל: PRMScore. פאנל ימין: AUC בתוך תשובה.',
    'הסיגנלים נבחרו על PRM-7B; לשאר המודלים זה מבחן העברה.<br>כל PRM מפוקח, כך שרק שלב ה־fusion ללא תוויות.<br>Skywork ו־ReasonEval לא הורצו.'))""")
sub("""'<p class="caption">These are cross-PRM tests on PRMBench, distinct from the external-benchmark tests on slide 11. Existing fixed-average runs do not establish L-SML transfer.</p>'""",
    """'<p class="caption">Cross-PRM tests on PRMBench (6,969 answers), distinct from the external-benchmark tests on slide 11. Every PRM reproduces its leaderboard score within 0.05 under its own rule (gate G1). Δ PRMScore L-SML − native, paired 95% interval: '+'; '.join(f"{r['checkpoint'].split(' (')[0]} {r['delta_prmscore_ci'][0]:+.2f} [{r['delta_prmscore_ci'][1]:+.2f}, {r['delta_prmscore_ci'][2]:+.2f}]" for r in U['generalization']['rows'])+'. Skywork and ReasonEval were not run.</p>'""")

# slide 13 chart: legend away from the first model's points.
sub("""ax.legend(fontsize=8)
    fig.tight_layout()
    return chart('prm_generalization_update'""", """ax.legend(fontsize=8,loc='lower right')
    fig.tight_layout()
    return chart('prm_generalization_update'""")

(HERE / 'base/build_base.py').write_text(src, encoding='utf8')
print('patched base/build_base.py')
