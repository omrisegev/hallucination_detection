# PRMBench: בדיקת הטענות של Claude ומיקום ביחס לספרות

2026-09-23. סקר ממוקד של מקורות ראשוניים וקוד benchmark, עם בדיקת תוצאות קיימות. אינו סקירה שיטתית ממצה של כל פרסום או הוכחת state of the art. ההשלכות היישומיות נמצאות ב־[תוכנית runtime fusion](../experiments/PRMBENCH_RUNTIME_FUSION_PLAN_HE.md).

## 1. מה Claude אמר ומה הנתונים תומכים בו

נקראה השיחה המקומית `2d14a8c9-8b3f-489b-b26f-812b4b84a8b3.jsonl`, הודעות 2026-09-23T16:14:27Z ו־16:32:19Z, תחת תיקיית Claude של הפרויקט. לא הועתקה השיחה הפרטית ל־repo.

הטענות המספריות על within-AUC, hit ו־threshold נכונות. "כמעט עצמאיים" אינו נובע מקורלציה נמוכה; "קורא את תהליך היצירה של המודל שכתב" אינו תיאור מדויק של teacher-forced scoring של תשובות נתונות; וערך לסקאלה המוחלטת אינו הוכחת probability calibration. יש פוטנציאל להשלמה בין האותות, אבל union oracle אינו fusion בר־ביצוע.

מקור מקומי עיקרי: `.worktrees/lsml-ct7-levers-run/results/prm_vs_ct7_prmbench_v1/MEASUREMENT.json`. מכיל 6,969 רשומות PRMB, ללא כשלי אורך/NaN של PRM; 6,030 eligible within-AUC.

| שיטה | within-AUC | PRMScore q80 | PRMScore inner-selected |
|---|---:|---:|---:|
| CT7-z | .772397 | .645689 | .650370 |
| Family421 לפי נרמול Step437 | .780120 | .654664 | .660091 |
| Family421 עם answer-z הסופי בניסוי המקורי | .780120 | .656227 | .662690 |
| Qwen2.5-Math-PRM-7B בסקאלה המקורית | **.801180** | **.680351** | **.682885** |
| אותו PRM אחרי answer-z | .801180 | .673272 | .674726 |

השורה השלישית באה מ־`results/ct7_family_equal_v1/SUMMARY.csv` באותו worktree. אין לאחד אותה עם השורה השנייה: normalization שונה משנה PRMScore, אף שאינו משנה דירוג בתוך תשובה. inner-selected משתמש בתוויות; q80 הוא כיול כמותי ללא תוויות, אך בחירתו ברמת המחקר כבר חשופה ל־development.

PRM עם סף היצרן 0.5 מקבל .654568. לכן אפשר לומר ש־family-equal מגיע לנקודה דומה ואף מעט גבוהה יותר מה־PRM עם ספו המקורי. **לא** אפשר להסיק מכך שהשיטה טובה יותר כאשר ל־PRM ניתן אותו כיול, או שההפרש הקטן מול native מובהק. Family-equal עצמו אינו מועמד סופי לפי דרישת Omri.

ה־fusion הנלמד `fam421_eigen` מגיע ל־.778634 within-AUC, ‏.654139 q80 ו־.659127 inner-selected; גם הוא טרם מנצח את native PRM בפאנל q80, והוא donor-fitted. אלה תוצאות מבטיחות לתכנון, לא ניצחון של answer-local learned fusion.

## 2. היתרון הקיים הוא במדד לוקליזציה מסוים

על **6,035 תשובות שגויות**:

| שיטה | argmax על אחד מצעדי השגיאה | שיעור |
|---|---:|---:|
| CT7 | 3,690 | 61.14% |
| Family421, בקרה | 3,865 | 64.04% |
| PRM מפוקח | 3,487 | 57.78% |

זה אינו first-error SLA ואינו PRMScore הרשמי. הוא מחזק כיוון של בחירת צעד לבדיקה. בתוך דירוג כל זוג correct/error, PRM דווקא טוב יותר: PRM−CT7 ‏+.028784, CI95 [.021415,.036008]; PRM−family ‏+.021061, CI95 [.013878,.028026]. אלה intervals קיימים למדד within-AUC בלבד.

ל־CT7 יתרונות תיאוריים ב־redundancy, circularity ו־domain inconsistency, ול־PRM יתרונות בסוגים אחרים. אין עדיין category CIs מתוקנים ואין להניח שהבדל זה מוכיח הפרדה בין "מבנה" ל"סמנטיקה". ‏79.90% איחוד הצלחות CT7/PRM הוא oracle diagnostic, לא הישג של מערכת.

## 3. המדד והאוכלוסייה הרשמיים

PRMScore הוא ממוצע F1 של המחלקה התקינה והשגויה, ולא AUROC או hit. חשוב לחשב אותו מאותה אוכלוסיית צעדים והחלטות. קוד המשימה הרשמי מסיר duplicate IDs, מוסיף correct controls ומוציא אותם מה־pooled confusion. לכן 6,969 רשומות מקומיות אינן סימן להערכה על benchmark חסר: 6,216 raw פחות 5 כפילויות ועוד 758 controls. יש גם ממשק validity/redundancy; השיטה שלנו משתמשת ב־validity fallback המתועד עבור סוגי redundancy. [קוד המשימה הרשמי](https://raw.githubusercontent.com/ssmisya/PRMBench/main/mr_eval/tasks/prmtest_classified/task.py).

המאמר המקורי מדווח, בספים המקוריים, Qwen PRM7B ‏65.5, ‏72B ‏68.2, Pure7B ‏65.3, Skywork7B ‏65.1 ו־ReasonEval34B ‏60.5. השחזור המקומי של Qwen7B, ‏65.4568, תואם בעיגול. זהו anchor משמעותי לתקינות, אך אינו מבטל הבדלי access, פיתוח על benchmark ואימון מוקדם. [המאמר הרשמי, טבלאות 3 ו־7](https://aclanthology.org/2025.acl-long.1230.pdf).

## 4. ספרות רלוונטית והשלכות על מגבלת החישוב

| עבודה ומקור ראשוני | מה נבדק | משמעות לפרויקט |
|---|---|---|
| [PRMBench, ACL2025](https://aclanthology.org/2025.acl-long.1230/) | benchmark לסוגי שגיאה, מדדי סיווג ולוקליזציה שונים | נשתמש במדד הרשמי לטענות השוואה למודלים; hit בנפרד |
| [The Lessons of Developing PRMs, 2025](https://arxiv.org/abs/2501.07301), [פרסום צוות Qwen](https://qwenlm.github.io/blog/qwen2.5-math-prm/) | PRM מפוקח, בניית אותות training וחשיבות step-error evaluation לצד BoN | cached Qwen7B הוא baseline מתאים; תוצאת BoN אינה עדות ללוקליזציה |
| [PathFinder-PRM, 2025](https://arxiv.org/html/2505.19706v1) | error-aware hierarchical supervision; מדווח PRMScore ‏67.7 ל־7B | יש להוסיף להקשר מעבר לטבלת Qwen הישנה; דורש אימון ומודל ייעודי, אין צורך להריצו כאן |
| [FreePRM, 2025](https://arxiv.org/html/2506.03570v1) | pseudo step labels מהצלחת התשובה הסופית, טיפול ברעש, אימון PRM; PB F1 ‏53.0 | דוגמה לכך שאין step labels אינו אומר אין supervision או אין training. ללמוד עקרון טיפול ברעש, לא לאמץ את האימון |
| [Unsupervised Process Reward Models, May2026](https://arxiv.org/html/2605.10158v1) | סימוני correctness בהקשר, LoRA/RL ל־PRM; אימון מדווח כ־5.5 שעות על 8 H200 | קשור ישירות לרעיון unsupervised, אך מחוץ לתקציב שלנו; אינו שקול ל־CPU fusion של trace קיים |
| [Efficient PRM via Contrastive Mutual Information, ACL2026](https://aclanthology.org/2026.acl-long.1744.pdf) | הסתברות תשובה נכונה מול hard negative, יצירת 80k labels ואימון reward head/backbone; PRMB ‏60.67 ל־CPMI_Merge | מקור לרעיון contribution; דורש gold/negative scoring ואימון ולכן אינו residual חינמי על cache רגיל |

סקר זה כולל עבודות חדשות יותר מטבלת 2025, אך אינו מצדיק "מוביל עולמי". ציונים מספרות הם context-only כל עוד אין per-example predictions, התאמת population/threshold ו־paired uncertainty. אין להשוות 60.67 של CPMI או 67.7 של PathFinder כאילו הופקו אצלנו באותו evaluator ובאותו תקציב כיול.

החידוש שנכון לבחון הוא **fusion נלמד מקומי וזול של טלמטריה זמינה**, עם תועלת מדידה מעל בקרות קבועות. עצם העדר labels אינו מספיק לחידוש, ובספרות SSL/unsupervised רבות עדיין נדרש אימון GPU משמעותי.

## 5. מה חסר לפני כותרת על תחרות במודלים מפוקחים

1. Replay ל־confusion counts ול־PRMScore הרשמי עבור saved decisions, עם hash של evaluator וגרסת dataset.
2. paired source-group CIs ל־PRMScore ול־any-error hit. ה־within-AUC CI אינו מכסה אותם.
3. ביקורת nesting: אף source group מוחזק בחוץ אינו נכנס ל־fusion fit או לכיול שיצר את תחזיתו. להבחין בין בחירה מחקרית על development לבין label-free fitting בזמן ריצה.
4. מדידת CPU runtime ומקור הטלמטריה. extra inference=0 מותנה בזמינות trace; אין להציג את teacher-forced pass הקיים כחינמי מחוץ לתנאי זה.
5. ניצחון מיוחס ללמידה מול same-bank equal/untrained. ה־family-equal הנוכחי אינו עונה לבדו על מטרת המחקר.
6. אחרי בחירה: benchmark חיצוני או אוכלוסייה שלא שימשה לפיתוח, עם בדיקת חפיפת source questions. אין להפוך את PRMBench שכבר נותח למבחן untouched באמצעות שינוי split בלבד.

**החלטה:** יש הצדקה למקד את הפיתוח ב־PRMBench ולבחון L-SML ובהמשך RBM קטן. הניסוח הנוכחי הנתמך הוא "יתרון במדד איתור צעד שגוי ותוצאות תחרותיות בחלק מההשוואות". טענה של עליונות כוללת על PRMs מפוקחים עדיין אינה נתמכת.
