# חומרים למנחים: PRM boosting והמשך, 17–19.10.2026

המסמך הזה הוא אינדקס בלבד. מספרים מצוטטים מה-README של כל תיקייה, ולפני שמציגים מספר צריך לבדוק אותו שם.

**הסטטוס של כל התוצאות:**
- כל התוצאות על PRMBench הן development data: כבר הסתכלנו על הנתונים, כך שהן לא מבחן נקי.
- ה-fusion עצמו לא משתמש ב-labels. אבל ה-PRMs אומנו עם labels, והציון המשולב יורש את זה.
- החומר נמצא כולו בענף `claude/integration-2026-10-17`.

## 1. מה להציג

| נושא | קובץ | שורה תחתונה |
|---|---|---|
| שקף: מאמרי 2026 שמצטטים את PRMBench | `docs/meetings/prmbench_citing_papers_2026-10-18/SLIDE_HE.html` (שקופית 1: `?slide=1`), `SLIDE_HE.png` | PRISM, ‏CROP ו-SLIDER מול השיטות שלנו, כולל עמודת Socratic |
| שקף: RLVR והקשר ל-PRM boosting | אותו קובץ, `?slide=2`; ‏`SLIDE_HE_2.png` | נכתב בסשן נפרד |
| סיכום ספרות (כמה הנושא חם, Optimal Aggregation, היסטוריית ה-threshold) | `docs/meetings/prmbench_citing_papers_2026-10-18/LITERATURE_NOTES_HE.md`, `CITING_PAPERS_2026.csv` (64 מאמרים) | הנישה של fusion ללא labels עם PRM כמעט ריקה בכנסים המובילים |
| השוואת מטריקות מול המאמרים | `results/prmbench_paper_metric_parity_v1/README.md` + 4 גרפי `*_PARITY.png` | ב-PRISM המטריקה זהה; ה-PRM עם threshold מוזז מגיע לנקודת העבודה של PRISM (68.26) |

## 2. התוצאות, לפי סדר הסיפור

1. **שלב 1, Qwen-PRM-7B + telemetry:** `results/prm_boosting_v1/`.
2. **הכללה ל-5 PRMs על PRMBench:** `results/prm_boosting_generalization_v1/run_v1/eval_lsml4/README.md`.
   - ב-Qwen-7B, ה-PRMScore עולה מ-65.49 ל-69.48. זה עם 3 channels שנבחרו עם labels.
3. **בקרות CPU:** `results/prm_boosting_controls_v1/README.md`.
   - יש כאן בחירת channels בלי labels, null, וניכוי של מיקום ואורך הצעד.
4. **Bank sweep ו-CHAIN:** `results/prm_boosting_bank_sweep_v1/README.md`, ‏`TABLES.md`.
   - CHAIN_N16 בלי labels מגיע ל-68.25. ב-N16 יש 3 digit channels.
   - בלי digits (CHAIN_N13): 67.82.
5. **הכללה ל-Socratic, כפי שהוקפאה מראש:** `results/prm_boost_external_v1/README.md`.
   - ההכללה לא מחזיקה: השיפור הוא רק ב-RLHFlow, ‏1 מתוך 5 PRMs.
6. **CHAIN על Socratic:** `results/prm_boost_external_v1/chain_socratic/README.md`.
   - זה arm שנוסף אחרי ההקפאה.
   - ב-Qwen-7B התוצאה היא 65.5, לעומת 68.1 של ה-PRM לבד.
7. **אבחון Socratic, עם labels ולא שיטה:** `results/prm_boost_socratic_diagnosis_v1/README.md`.
   - LR מפוקח על PRM + N16 מגיע ל-0.836 within-AUC, מול 0.824 של ה-PRM.
   - energy_level חזק כמו ה-PRM על PRMBench, אבל חלש ממנו בהרבה על Socratic.
8. **PRM voters (fusion של כמה PRMs):** `results/prm_voters_prmbench_v1/README.md`.
   - no-go: התוצאה נמוכה מה-PRM הבודד הטוב ביותר.
9. **LR backup, ייצוגים לכל cluster:** `results/lr_backup_cluster_reps_v1/README.md`.
10. **ProcessBench, מוכן ל-AIRCC ועדיין לא הורץ:** `results/prm_boost_processbench_v1/HANDOFF.md`.
    - קודם רץ timing job. אחריו צריך החלטת תקציב לפני ההרצה המלאה.

**מפה מלאה של הענפים והקבצים הגולמיים:** `docs/handoffs/PRM_BOOSTING_RESULTS_MAP_2026-10-18.md`.
**תוכנית הראיות וסף ההצלחה שנקבע מראש:** `docs/post_advisor_2026-10-17/05_PRM_BOOSTING_EVIDENCE_PLAN_HE.md`.

## 2א. עדכונים מ-19–20.10 (ענף `claude/prm-boosting`)

11. **אבלציה של 5 זרועות, ותיקון ה-digits של Math-Shepherd:** `results/prm_boost_ablation5_v1/` ו-`digitfix_v1/README.md` (HISTORY Step 466).
    - ב-PRMBench, ‏PRM + טלמטריה עוקף את ה-PRM לבד ב-within-AUC בכל 5 ה-PRMs (‎+0.027 עד ‎+0.112). רוב הרווח מגיע מהטלמטריה עצמה. המשקול של L-SML מעל ממוצע שווה מוסיף ‎+0.013 עד ‎+0.024, ב-5 מתוך 5 על N16.
    - ב-Math-Shepherd ערוצי ה-digits היו אפס, כי מזהי הספרות של Qwen קבועים בקוד. התיקון מעלה את CHAIN_N16 מ-0.712 ל-0.778. אבל הרווח נעלם אחרי ניכוי מיקום ואורך, ולכן לא להציג אותו כאות מהתוכן.
12. **כלל הסימון ואומדן המשקלות של SML:** `results/sml_weight_estimator_v1/README.md` (HISTORY Step 467).
    - **באג בשיטה:** על ערוצים שכבר מכוונים, SML נותן משקל שלילי, וההיפוך לפי ה-PRM הופך את הטלמטריה. זה מה שהוריד את Math-Shepherd ב-Socratic מתחת למקריות (0.468). זה קרה ב-3 מתוך 20 שילובים ב-Socratic וב-0 מתוך 60 ב-PRMBench. המאמר של L-SML מניח שכל הערוצים טובים ממקריות, כך שמשקלות לא שליליים הם ההנחה שלו.
    - **אומדן המשקלות מוטה לכיוון משקלות שווים** ביחס למודל. אבל ההתאמה הלא מוטה **מפסידה** על בנקי הפיתוח: ב-N16, ‏ProcessBench יורד מ-41.61 ל-38.61 ו-PRMBench מ-0.7958 ל-0.7762. כלומר הכיווץ עובד כרגולריזציה. משקלות לא שליליים עוזרים רק בבנקים הגדולים שבהם היה משקל שלילי (N40, ‏N43, ‏N53). ‏N16 עם L-SML הנוכחי נשאר הטוב ביותר.
    - הבדיקה רצה על ה-template הישן, ולא על thinkfix_mask1.

## 3. מה לא נמצא ב-git

- **ציוני ה-fused לכל צעד על Socratic (npz):** לא ב-git, כמו `scores_socratic/*.npz`.
  - אפשר לשחזר אותם בדקה מה-banks שב-Drive (`gdrive:hallucination_detection/cluster_results/prm_boost_external_v1/`) עם הסקריפטים שב-`chain_socratic/` וב-`prm_boost_socratic_diagnosis_v1/`.
- **קבצי telemetry גולמיים:** נמצאים ב-AIRCC וב-Drive. הפירוט במפת התוצאות.
