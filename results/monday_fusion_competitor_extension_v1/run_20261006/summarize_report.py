"""Build the presentation-facing report only from frozen analysis tables."""
from pathlib import Path
import hashlib, json
import pandas as pd

ROOT=Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT=ROOT/'results/monday_fusion_competitor_extension_v1/run_20261006'
METHODS=['HEM_MOM_cc','MEAN_MOM_cc','MEAN_MOM_ga','TCI_ONLY_ga']
RULES=['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80']
PRM='prmbench_qwen3_8b'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def fmt(v):
    return '—' if pd.isna(v) else f'{float(v):.4f}'

def main():
    metrics=pd.read_csv(OUT/'METRICS.csv')
    categories=pd.read_csv(OUT/'F11_PRM_CATEGORY_METRICS.csv')
    overlap=pd.read_csv(OUT/'F11_PRM_OVERLAP_SUMMARY.csv')
    pb=pd.read_csv(OUT/'F12_MTG_BREAKDOWN.csv')
    contrasts=pd.read_csv(OUT/'CONTRASTS.csv')
    status=json.loads((OUT/'RUN_STATUS.json').read_text(encoding='utf8'))
    prm=metrics[metrics.cell==PRM].groupby(['method','decision_rule']).agg(
        auc=('within_answer_auc','mean'),prmscore=('prmscore','mean')).reset_index()
    pbd=metrics[metrics.cell!=PRM].groupby(['method','decision_rule']).agg(
        pb_argmax=('pb_first_error_accuracy','mean'),pb_decision=('pb_decision_error_accuracy','mean'),
        pb_f1=('pb_official_f1','mean'),clean_flag=('clean_flag_rate','mean')).reset_index()
    results=prm.merge(pbd,on=['method','decision_rule'])
    md=['# F11–F12: השוואת ארבע השיטות מול PRM ו־Mind the Gap','',
        'כל המדדים להלן הם על אוכלוסיית הפיתוח שכבר נחשפה. הדוח אינו מציג הערכה חיצונית לא־חשופה. '
        'כל ערך מוחלט הוא ממוצע שווה־משקל על שמונת הבנקים; ב־ProcessBench נעשה ממוצע קודם על שמונת תאי הבדיקה בכל בנק.','',
        '## מדדי החלטה לפי שיטה','',
        '| שיטה | כלל | PRMB within-answer AUC | PRMScore | PB argmax: פגיעה בשגיאה ראשונה | PB: הצעד המסומן הראשון | PB official F1 | תשובות נקיות עם דגל |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for _,r in results.iterrows():
        md.append(f"| {r.method} | {r.decision_rule} | {fmt(r.auc)} | {fmt(r.prmscore)} | {fmt(r.pb_argmax)} | {fmt(r.pb_decision)} | {fmt(r.pb_f1)} | {fmt(r.clean_flag)} |")
    md += ['', 'AUC של PRMBench אינו תלוי בכלל ההחלטה ולכן חוזר בכל שלוש השורות של אותה שיטה. `clean_flag` הוא שיעור התשובות הנקיות שקיבלו לפחות דגל אחד. '
           'ה־F1 הרשמי ב־ProcessBench תלוי בכלל ההחלטה; לכן AUC, פגיעה ב־argmax, פגיעה בדגל הראשון ו־F1 נשארים מדדים נפרדים.','',
           '## F11 · PRMBench לפי קטגוריית שגיאה','',
           'הקטגוריות הן `sensitivity`, `simplicity`, `soundness`; `control` אינו קטגוריית שגיאה ולכן נשמר בטבלה אך אינו מצויר. הטבלה היא לכלל `ANSWER_Z_Q80`; יתר הכללים ו־top-k נמצאים ב־`F11_PRM_CATEGORY_METRICS.csv`.','',
           '| קטגוריה | שיטה | within-answer AUC | PRMScore | תשובות עם AUC |','|---|---|---:|---:|---:|']
    cat=categories[(categories.method.isin(METHODS))&(categories.rule=='ANSWER_Z_Q80')&(categories.category!='control')]
    for (category,method),g in cat.groupby(['category','method']):
        md.append(f'| {category} | {method} | {fmt(g.within_answer_auc.mean())} | {fmt(g.prmscore.mean())} | {int(g.n_auc_answers.mean())} |')
    md += ['', '## חפיפת זיהוי שגיאות ב־F11','',
           'ספירת צעדי השגיאה המסומנים גם על ידי השיטה וגם על ידי PRM, רק על ידי השיטה, רק על ידי PRM, או על ידי אף אחד מהם. הספירות כוללות את שמונת הבנקים; אין לפרש צעדים בתוך אותה תשובה כתצפיות בלתי־תלויות.','',
           '| שיטה | כלל | גם אנחנו וגם PRM | רק אנחנו | רק PRM | אף אחד |','|---|---|---:|---:|---:|---:|']
    ov=overlap[(overlap.method.isin(METHODS))&(overlap.rule=='ANSWER_Z_Q80')]
    for method,g in ov.groupby('method'):
        md.append(f"| {method} | ANSWER_Z_Q80 | {int(g['both'].sum())} | {int(g['only_ours'].sum())} | {int(g['only_PRM'].sum())} | {int(g['neither'].sum())} |")
    md += ['', '## F12 · Mind the Gap ב־ProcessBench','',
           'Mind the Gap הוא השחזור המקומי הקפוא: teacher-forced, paper EMA span 5, later-token attribution, top-5 drop mean ו־native argmax. הוא אינו תוצאת הכותרת מהמאמר. ל־MTG אין מנגנון abstention לתשובות נקיות, לכן השוואת שיעורי דגלים נקיים מוצגת רק עבור השיטות שלנו.','',
           '| שיטה | argmax שלנו פוגע בשגיאה ראשונה | argmax של MTG פוגע | הצעד הראשון שסומן פוגע (R2) | מוקדם | מאוחר | ללא דגל | פגיעה |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    pbe=pb[(pb.rule=='R2_COUNT')&(pb.stratum=='error_position_quartile')]
    for method,g in pbe.groupby('method'):
        n=g.n_answers.sum();
        weighted=lambda col: float((g[col]*g.n_answers).sum()/n) if n else float('nan')
        early=int(g.early.sum()); late=int(g.late.sum()); noflag=int(g.no_flag.sum()); hit=int(g.hit.sum())
        md.append(f'| {method} | {fmt(weighted("candidate_argmax_hit"))} | {fmt(weighted("MTG_native_argmax_hit"))} | {fmt(weighted("candidate_decision_hit"))} | {early} ({early/n:.1%}) | {late} ({late/n:.1%}) | {noflag} ({noflag/n:.1%}) | {hit} ({hit/n:.1%}) |')
    md += ['', 'הטבלה מאגדת את כל תשובות ProcessBench השגויות על פני הבנקים. הפירוט לפי רביעון מיקום השגיאה, אורך תשובה ותא נמצא ב־`F12_MTG_BREAKDOWN.csv`; רשומות התשובה עצמן ב־`F12_MTG_ANSWER_FORENSICS.csv.gz`.','',
           '## ניגודים מזווגים','',
           'הפרשים חיוביים למטה פירושם `TCI_ONLY_ga` גבוה יותר; רווחי סמך percentile של 95% מדגימת 5,000 קבוצות שאלת מקור. אלה רווחים תיאוריים ללא תיקון ריבוי השוואות.','',
           '| כלל | מול | מדד | TCI פחות השיטה הקודמת | 95% CI |','|---|---|---|---:|---:|']
    cm=contrasts[(contrasts.method_b=='TCI_ONLY_ga')&(contrasts.method_a.isin(METHODS[:-1]))&
                 (contrasts.metric.isin(['within_answer_auc','prmscore','pb_official_f1','pb_decision_error_accuracy']))]
    for _,r in cm.iterrows():
        delta=-float(r.delta_a_minus_b); lo=-float(r.hi95); hi=-float(r.lo95)
        md.append(f'| {r.rule} | {r.method_a} | {r.metric} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    md += ['', '## גרפים','',
           '- [F11 לפי קטגוריה, רקע בהיר](figures/F11_PRM_categories_revised_light.png) · [SVG](figures/F11_PRM_categories_revised_light.svg) · [רקע כהה](figures/F11_PRM_categories_revised_dark.png) · [SVG](figures/F11_PRM_categories_revised_dark.svg)',
           '- [F12 לפי מיקום ואורך, רקע בהיר](figures/F12_MTG_position_length_revised_light.png) · [SVG](figures/F12_MTG_position_length_revised_light.svg) · [רקע כהה](figures/F12_MTG_position_length_revised_dark.png) · [SVG](figures/F12_MTG_position_length_revised_dark.svg)','',
           '## כיסוי ומגבלות','',
           f"- כיסוי: {status['n_banks']} בנקים × {status['n_cells']} תאים × {status['n_methods']} שיטות; {status['metric_rows']:,} שורות מדדים, {status['overlap_rows']:,} רשומות חפיפה, {status['pb_forensic_rows']:,} רשומות ProcessBench.",
           f"- שחזור מדדים: {status['candidate_panel_parity_rows']} שורות legacy ו־{status['tci_stage2_parity_rows']} שורות TCI תואמות לשלב הקודם בסבילות 1e-12.",
           '- כללי החלטה: answer-z q80, R2, שער אנטרופיה; top-1/2/3 בנפרד. top-2 מסומן כבחירה בדיעבד.',
           '- בדיקה עצמאית: עדיין ממתינה ל־Claude. עד לסיום הבדיקה, אלה תוצאות פיתוח ולא מסקנות למצגת.',
           '', '## Claims ledger','',
           '| טענה | מספר/קובץ | N | מצב |','|---|---|---:|---|',
           f"| כיסוי מלא של הרשומות | RUN_STATUS.json; {status['metric_rows']} metric rows | 8×9×4×6 | שחזור פנימי עבר; ביקורת עצמאית ממתינה |",
           '| TCI-only יתרון PRM לפי קטגוריה | F11_PRM_CATEGORY_METRICS.csv | 3 קטגוריות × 8 בנקים × 4 שיטות | תיאור פיתוח; לא בחירת זוכה |',
           '| חפיפת PRM ו־ProcessBench/MTG | קובצי F11/F12 | 1.34M / 1.31M רשומות | תיאור פיתוח; לא ביקורת חיצונית |',
           '| הבדל בין מתכונים | CONTRASTS.csv | 5,000 bootstrap draws | רווחים תיאוריים; ממתין ל־Claude |','']
    text='\n'.join(md)
    text=text.replace('## ניגודים מזווגים', 'הבהרת משקול ל־F12: הטבלה מאגדת תשובות שגויות ולכן משוקללת לפי תשובות. בטבלת המדדים הראשית כל בנק מקבל משקל שווה (ולאחר מכן ממוצע שווה בין שמונת התאים שלו). אין להשוות את הערכים המצרפיים ב־F12 ישירות לממוצע התאים שם.\n\n## ניגודים מזווגים')
    tmp=OUT/'REPORT.md.tmp'; tmp.write_text(text,encoding='utf8'); tmp.replace(OUT/'REPORT.md')
    manifest={'status':'REPORT_BUILT_FROM_FROZEN_TABLES','inputs':{},'output':{'path':'REPORT.md','sha256':sha(OUT/'REPORT.md')},
              'script_sha256':sha(Path(__file__))}
    for name in ['METRICS.csv','F11_PRM_CATEGORY_METRICS.csv','F11_PRM_OVERLAP_SUMMARY.csv','F12_MTG_BREAKDOWN.csv','CONTRASTS.csv','RUN_STATUS.json']:
        manifest['inputs'][name]=sha(OUT/name)
    (OUT/'SUMMARY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print('REPORT.md updated from frozen tables; metrics unchanged.')

if __name__=='__main__': main()
