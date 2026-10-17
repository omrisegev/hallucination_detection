# סקירת כיווני מחקר אחרי פגישת המנחים (2026-10-17)

כל מספר כאן מגיע מקובץ ב-`origin/claude/integration-2026-10-17`, והקובץ מצוין לידו. זה development על נתונים שכבר נבדקו, ואין כאן מבחן חיצוני בלתי נגוע. לא הרצתי שום ניסוי חדש בשביל הסקירה.

## סיכום ההמלצה

| כיוון | החלטה | למה, בשורה אחת |
|---|---|---|
| PRM boosting (PRM + telemetry מה-backbone שלו) | **להמשיך, כיוון ראשי** | השיפור עקבי ב-5 מתוך 5 PRMs, וכל ה-CIs לא כוללים 0 |
| PRM voters (כמה PRMs, judges ו-critic כמצביעים) | **להמשיך, מוגבל להרצה אחת** | ה-pilot על ProcessBench שלילי. PRMBench הוא המבחן האמיתי, וה-job מוכן |
| Label-free PRMBench (telemetry בלבד) | **להקפיא כרכיב, בלי חיפוש שיטות נוסף** | אחרי ה-think-fix נשאר בערך 0.78, וה-LR-gap סגור |
| אמידת מספר שגיאות (step-count) | **לעזוב כשיטה** | משפחת A–D נכשלה, ומבנה הספירה הוא artifact של הבנצ'מרק |
| Agent process rewards | **להשהות; לכל היותר בדיקת CPU אחת** | אין signal label-free שמחזיק מעבר לאורך ולמיקום |
| Jev | **לעזוב ככיוון** | API סגור בלי logprobs. לכל היותר מצביע נוסף אופציונלי |
| ProcessBench | נשאר panel משני | אין שם שיפור מעל הממוצע הפשוט |

## 1. PRM boosting: להמשיך (ראשי)

**ראיות.** `docs/reports/ADVISOR_MEETING_EXTERNAL_AND_PRM_BOOSTING_FINDINGS_20261015.md` מראה L-SML label-free של ה-reward של ה-PRM יחד עם 3 channels מה-backbone שלו (`energy_level`, `hist_entropy_cusum_abs_series`, `realized_drv`), במתכון קפוא:

| PRM | PRMScore native → boosted | Δ [95% CI] | within-AUC |
|---|---|---|---|
| Qwen2.5-Math-PRM-7B | 65.49 → 69.48 | +3.99 [3.31, 4.67] | 0.801 → 0.850 |
| Qwen2.5-Math-PRM-72B | 68.18 → 69.84 | +1.66 [1.01, 2.28] | 0.795 → 0.856 |
| PURE-PRM-7B | 65.31 → 66.69 | +1.38 [0.75, 2.08] | 0.763 → 0.811 |
| RLHFlow Llama-3.1-8B | 54.43 → 63.53 | +9.10 [8.42, 9.79] | 0.662 → 0.772 |
| MathShepherd Mistral-7B | 47.02 → 63.68 | +16.67 [15.13, 18.09] | 0.684 → 0.759 |

השיפור הגדול ביותר הוא בקטגוריות redundancy (49.0 → 67.7) ו-circular (54.9 → 71.8). זו בדיוק השאלה שהמנחים העלו.

**סייגים.**
- בחירת שלושת ה-channels נעשתה post hoc על PRM-7B. המעבר לארבעת ה-PRMs האחרים הוא ראיית generalization, אבל זה עדיין אותו בנצ'מרק.
- Step 463 מצא ש-69.48 עומד ליד Gemini-2-Thinking (68.8) ב-leaderboard המקורי של PRMBench. צריך לבדוק שהפרוטוקול זהה לפני שמציגים את זה כהשוואה.
- ה-think-fix לא רלוונטי כאן, כי אף backbone של PRM הוא לא Qwen3.

**הצעדים הבאים, לפי סדר:**
1. בדיקת comparability מול ה-leaderboard של PRMBench: אותו evaluator ואותו split. CPU בלבד.
2. מבחן חוץ-בנצ'מרק עם אותו מתכון קפוא: Socratic-PRMBench ו-Hard2Verify (קיים `cluster/run_external_prm.py`). נדרש GPU, ולכן צריך אישור.
3. ProcessBench כ-panel משני. ב-pilot של ה-voters, PRM-7B לבדו היה הכי חזק שם (answer AUC 0.944), אז זה מבחן קשה למתכון.

## 2. PRM voters: להמשיך, בהרצה אחת מוגבלת

**ראיות.** `results/prm_voters_processbench_pilot_2026-10-16/README.md`: על ProcessBench, PRM-7B לבדו מקבל answer AUC 0.944, L-SML על 4 מצביעים מקבל 0.913 (−3.1pp [−3.9, −2.2]), ו-MoM מעריך בחסר את שיעור התשובות השגויות (0.52 מול 0.65). אבל ב-ProcessBench יש רק first-error labels, ושני ה-judges חלשים מאוד שם (0.544 ו-0.730).

**החלטה.** להריץ את ה-job שמוכן (PRMBench, judges ו-critic). כבר עכשיו לקבוע את כלל ה-go/no-go: אם L-SML על כל המצביעים לא עובר את ה-PRM החזק ביותר לבדו ואת ה-PRM boosting מסעיף 1, הכיוון נסגר. אם הוא כן עובר, הוא מתאחד עם סעיף 1 לסיפור אחד: "label-free fusion של מאמתים ברמת הצעד".

## 3. Label-free PRMBench (telemetry בלבד): להקפיא כרכיב

**ראיות.**
- `results/thinkfix_rescore_v1/README.md`: אחרי תיקון ה-think block, within-AUC ירד מעט: ממוצע −0.0088 על 50 תאים, ו-N16 MEAN_MOM_ga ירד מ-0.7909 ל-0.7767.
- Supervised LR ירד מ-0.8224 ל-0.8015, והפער שלו מעל ה-label-free הצטמצם ל-0.0226.
- Step 464 סגר את כיוון ה-LR-gap (החלטה שלך מ-2026-10-17): ה-gain של LR מגיע מ-selection, orientation ו-weighting, ושני ה-diagnostics של pruning פגעו בתוצאה (0.712 / 0.727 מול 0.770).

**החלטה.** לא לפתוח חיפוש שיטות נוסף על telemetry בלבד. המשפחות שנבדקו (readout, tail weights, family, gate, banks) נסגרו כבר בספטמבר. ה-N16 bank עם L-SML או IU-PCR נשאר הרכיב הקפוא ושורת ה-baseline שמתחת ל-PRM boosting. מספרי ה-advisor צריכים להשתמש בגרסת ה-thinkfix_mask1.

## 4. Step-count: לעזוב כשיטה

**ראיות.**
- `results/label_free_error_count_v1_2026-10-12/FINDINGS_HE.md`: אף זרוע ב-A–D לא עברה את קריטריון ההצלחה. LOCAL_MOM מוריד PRMScore ב-0.48 מול Q80, ו-A נכשל ב-4 folds.
- `results/count_structure_check_v1/README.md`: מספר השגיאות האמיתי כמעט לא תלוי באורך (Spearman 0.092, ממוצע 2.12). k=2 קבוע נותן 67.90 מול Q80 65.36, אבל זה משקף את האופן שבו PRMBench מזריק שגיאות (בערך שתיים לתשובה). זו לא שיטה label-free.

**החלטה.** לסגור. לדווח את TOP_2 רק כ-diagnostic גלוי על מבנה הבנצ'מרק, ולא כשיטה. ה-continuous emission model נשאר השערה שלא בוצעה, ואין סיבה לממן אותה עכשיו.

## 5. Agent process rewards: להשהות

**ראיות.**
- HISTORY Step 453: אין signal label-free משכנע מעבר לאורך, למיקום ול-artifacts של הבנייה.
  - TPRM: length 0.625 עדיף על entropy 0.584.
  - APB: entropy 0.582, קרוב ל-length 0.567.
  - AFT: ה-signal מגיע רק משגיאות מוזרקות. ב-diagnosed, 3 קבוע מקבל 0.342 מול IU-PCR 0.233.
- Step 454: ב-APB, fusion של judge Qwen3-8B עם telemetry מעלה את first-error vs prior מ-0.592 ל-0.681 (+0.089 [0.055, 0.121]). זו התוצאה החיובית היחידה, ועדיין אין לה control של judge+length/position.

**החלטה.** CPU בלבד: להריץ את ה-control של judge+length ו-judge+position על APB. אם ה-+0.089 נעלם, לסגור. אם הוא נשאר, זה הופך לעוד מקרה של סעיף 1 (judge במקום PRM) ונבדק באותה מסגרת. לא להשקיע GPU ב-own-backbone TPRM לפני כן.

## 6. Jev: לעזוב

**ראיות.** `docs/research_notes/jev_2026-10-12/README.md`: מודל החלטות סגור, בלי weights ובלי logprobs, וההסתברויות מעוגלות ל-0.01 (הרבה ties). המספר של 66.38 על PRMBench מגיע מ-repo קהילתי עם פרוטוקול שלא פורסם. לא נעשה אף ניסוי.

**החלטה.** לא כיוון מחקר. אופציה זולה (בערך $1) היא להוסיף את Jev כמצביע נוסף ב-voters של סעיף 2. זה דורש לשלוח את PRMBench ל-API חיצוני, אז רק עם אישור מפורש ואחרי שהתשובה של סעיף 2 חיובית.

## סדר העבודה המוצע

1. להגיש את ה-voters job (ממתין לאישור התקציב).
2. בזמן שהוא רץ, על CPU: comparability מול ה-leaderboard (1.1) ו-control ל-APB (5).
3. אחרי שה-voters מסתיים: החלטת go/no-go לסעיף 2.
4. מבחן חוץ-בנצ'מרק ל-PRM boosting (1.2), עם אישור GPU.
