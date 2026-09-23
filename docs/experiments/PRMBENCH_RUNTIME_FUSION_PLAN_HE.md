# תוכנית מעודכנת: fusion בזמן ריצה, עם PRMBench כיעד הראשי

תאריך: 2026-09-23. סטטוס: חוזה מחקר מעודכן; תיקוני דיווח מומשו והופקו מחדש. תיקוני רכיבי החישוב נבדקו בבדיקות יחידה. **טרם הורצה הערכת איכות מלאה של מועמד חדש.**

מסמך זה גובר על הנחיות סותרות בתוכנית SSL v1.1 ובתוכניות הישנות. הוא מיישם את הבהרת Omri: אין אימון GPU, אין encoder חדש ואין מעבר LLM/PRM נוסף. הלמידה המותרת היא חישוב CPU על הטלמטריה שכבר מופקת, בעיקר מתוך התשובה הנוכחית. מיצוע פשוט או משפחתי הוא בקרה בלבד; הוא אינו השיטה הסופית.

**עדכון לאחר סקירת החודש:** המוקד הוא L-SML שמייצר תרומה לתוצאה. בסיס ההמשך שנמצא הוא בנק11 ברמת step: within-AUC של 76.4531 ו־PRMScore רטרוספקטיבי של 64.1184, מול 74.9644 ו־63.3111 במיצוע אותו בנק. CT7 הוא עוגן בלבד. [הסקירה והאימות המלאים](../reviews/LSML_PRMBENCH_MONTH_REVIEW_20260923_HE.md) מפרידים בין התרומה הנלמדת, איזון הקבוצות וכיול הסף. המודל ההיסטורי donor-fitted; R0-B להלן מאמת אותו, ו־R1 בוחן בנפרד התאמה בזמן ריצה. חולשה ב־PB אינה פוסלת מועמד ל־PRMB.

## 1. ההחלטה והראיות

PRMBench יהיה יעד הפיתוח הראשי. ProcessBench יהיה פאנל משני, המדווח באותה גרסה קפואה אחרי בחירת המתכון; אין לדרוש שיפור בשניהם כדי להמשיך מועמד שטוב ב־PRMBench. אין להסתיר ירידה ב־PB או להשתמש בו שוב ושוב לבחירת פרמטרים ואז לקרוא לו מבחן העברה בלתי נגוע.

ההחלטה היא שינוי מוקד בעקבות תוצאות development שכבר נראו, ולא הכרזה על ניצחון. ראו [בדיקת הטענות והספרות](../reviews/PRMBENCH_LITERATURE_AND_CLAIM_AUDIT_20260923_HE.md).

ב־6,030 תשובות מתאימות לדירוג, PRM מפוקח מקבל within-AUC של 0.801180 לעומת CT7 עם 0.772397. לעומת זאת, מתוך 6,035 תשובות שגויות, CT7 פוגע באחד מצעדי השגיאה ב־3,690 תשובות, לעומת 3,487 של PRM. אלה מדדים שונים. Family421 משפר את התוצאות, אך משתמש במיצוע קבוע ולכן משמש בקרה בלבד במסגרת החדשה.

מקור: `.worktrees/lsml-ct7-levers-run/results/prm_vs_ct7_prmbench_v1/MEASUREMENT.json`. ההשוואה היא מול Qwen2.5-Math-PRM-7B שכבר נמצא ב־cache; אין כאן ריצה של מודל חדש.

## 2. מה ממשיך ומה יוצא מהתוכנית

| שלב ישן | החלטה | שינוי נדרש |
|---|---|---|
| S0 / S0-C, אימות וכיול | ממשיך כשלב ביקורת | להפריד איכות דירוג מכיול; לתת ל־PRM ולשיטה שלנו אותו תקציב כיול |
| S1, pseudo-labels | כלי אפשרי בתוך fusion מקומי; לא student גלובלי כברירת מחדל | להשתמש במשקל אמינות, בכיוון סימן או במטרות זוגיות; לבקר העתקת teacher |
| S2, prediction residual | ממשיך כרכיב משלים | לשמור ערוצים, לנרמל כל ערוץ לפני fusion, להשוות למנבא אפס; fitting מקומי הוא ניסוי חדש |
| S3, contribution residual | עדיפות נמוכה | לבדוק ערך נוסף בתוך fusion נלמד, מול random/leave-one-family controls; לא לכנות זאת HARP אמיתי |
| S4, masked SSL encoder | מחוץ להיקף הפעיל | אין להמתין לאימונו ואין להשיק אותו |
| S5, pooling נלמד מעל encoder | מחוץ להיקף הפעיל | גרסת CPU קודמת היא ראיה היסטורית; אין להרחיב אימון head כרגע |
| L-SML על בנק11 | בסיס ההמשך | תחילה אימות מתכון step קיים; התאמה מתוך התשובה עצמה היא ניסוי נפרד |
| RBM | מועמד חלופי תחום | H1 מדויק על CPU, עם בקרה לפני אימון; בלי grid של hidden units או CD |

## 3. חוזה נתונים וחישוב

יש להשתמש ב־v3 corrected labels וב־source folds v2 הקיימים. האוכלוסייה היא 13,769 רשומות: 6,969 PRMB ו־6,800 PB. ב־PRMB: 6,030 eligible within-AUC; 6,035 erroneous any-error-hit. יש לשמור רשומות ללא שגיאה ובקרות גם כשמדד מסוים אינו חל עליהן.

ב־PRMScore יש לשחזר את loader הרשמי: 6,216 רשומות מקור, הסרת חמישה IDs כפולים, הוספת 758 correct controls; הבקרות אינן נכנסות ל־pooled PRMScore. לא לבצע conversion נוסף לתוויות שכבר הומרו מ־one-based.

קלטים קיימים, ללא העתקת קבצי הענק:

- `results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz` וה־manifest שלו: שבעה ערוצים, validity masks, offsets ו־spans.
- `.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz`: 11 ערוצים, 6,968,779 טוקנים. SHA256: `ab4943bf72b3d27436721d7e2ce8d95dd1254e10d3d05b7c25b65706c1c729e5`.
- `results/localization_source_group_audit_v1/FOLDS_V2.json`.
- `.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/`: roster, labels, offsets ופרופילים קפואים.
- `dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl`: baseline מפוקח שמור בלבד.

במסלול ההתאמה המקומית R1–R3, כל preprocessing, לרבות covariance, קבוצות, residual predictor ומשקלי fusion, מתאים מתוך התשובה הנוכחית. לייבלים עוברים ל־evaluator בלבד. פאנל donor/calibrated ב־R0-B מאמת את המועמד ההיסטורי, עם שם מפורש; אין לערבבו בטענת answer-local. אין להסיק שכישלון התאמה מקומית מבטל את יתרון ה־fusion שנמדד במודל ההיסטורי.

"בזמן ריצה" כאן פירושו אחרי קבלת התשובה והטלמטריה שלה. נרמול על התשובה השלמה אינו causal streaming. בנוסף, הניסוי הקיים כולל teacher-forced scoring של תשובות benchmark נתונות. טענת אפס inference נוסף נכונה בתרחיש שבו הטלמטריה כבר זמינה; עדיין צריך למדוד בנפרד את עלות הפקתה במערכת אמיתית.

תקציב הנדסי התחלתי, לא נתון שנמדד: CPU בלבד, thread אחד לדיווח זמן, ללא CUDA imports מצד הקוד החדש, תוספת זיכרון שיא עד 1 GiB. עד 250ms ב־p95 לתשובה באורך עד 4,096 טוקנים; לדווח גם תשובות ארוכות בנפרד. חריגה גוררת סיווג כלא מתאים לתקציב, לא שינוי ספים בדיעבד. למדוד fit, transform, score, I/O ו־evaluation בנפרד. לריצה מלאה: תקרה של שעתיים; עצירה לפי הזמן, ללא הסקת איכות מהחלק שהספיק לרוץ. לוודא מקום לפלט מראש.

## 4. שלב R0 — תיקונים ואימות לפני שיפור

### מה תוקן כעת

1. `scripts/repair_runtime_fusion_reports.py` הפיק מחדש 228 שורות contrast ב־S1/S2/S3/S5 מתוך 13,769 רשומות OOF בכל שלב. `paired_N` הוא כעת 4,442 עבור PB SLA ו־6,030 עבור PRMB within-AUC. מספר bootstrap draws מופיע בנפרד. הקוד בודק שהאוכלוסיות זהות לפני שהוא משמר את ה־CI המקורי; הוא מסרב "לתקן metadata" כאשר נדרש חישוב סטטיסטי חדש.
2. `scripts/window_answer_local_fusion_v2.py` יוצר comparator עם אותה מסכת native בדיוק כמו כל learned arm. הוא משתמש בקוד המקור לקריאה בלבד ושומר scores/masks בגרסת תוצאה חדשה. התיקון נבדק ביחידה; הריצה המלאה טרם בוצעה.
3. `spectral_utils/runtime_fusion_protocol.py::residual_step_bank` שומר Top5 לכל ערוץ ומנרמל כל ערוץ על צעדי התשובה לפני fusion. אינו מחזיר מיצוע ערוצים. נבדקו invariance לשינוי קנה מידה, ערוצים קבועים, מנבא אפס ו־spans לא תקינים.

פלטים: `results/runtime_fusion_protocol_v2/`. אין דריסה של קוד או תוצאות Claude.

ההשוואה המתוקנת אלגברית בחלונות היא L-SML מינוס equal על native: PB ‏+0.3988 נקודות אחוז ו־PRMB ‏+0.003426 AUC. היא מבוססת על זהות fallback-to-equal ועל מוני התוצאה הישנה. **טרם חושב CI תקף לאוכלוסיית native זו; אין להסיק ממנה שיפור מובהק.**

### מה עוד נדרש ב־R0

להפיק עבור frozen learned candidates ו־cached PRM החלטות רשמיות, confusion counts ו־paired PRMScore CIs. לקבע את normalization המדויק: חישוב Step437 של family421 השמיט את answer-z הסופי שבניסוי המקורי. אין לאחד את שני ה־PRMScores תחת אותו שם.

לשחזר thresholds בשני פאנלים: native כפי שפורסם; q80 באותו donor partition ללא labels. פאנל inner-selected נפרד משתמש ב־labels ולכן אינו label-free. אם fusion למד מ־donors, calibration rows וה־held-out source groups חייבים להיות מחוץ לכל fit שיצר את ציוניהם. OOF בסיסי לבדו אינו מוכיח nested fitting תקין.

יש לבדוק גם `P_AGREE`: ‏812/6,969 תשובות, אך רק 857/94,203 צעדים, כלומר כ־0.91% מהצעדים. 11.65% הוא כיסוי תשובות. S5 finite final loss אינו הוכחת convergence. שני אלה תיקוני פרשנות; אינם הופכים כישלון לשיפור.

### R0-B — אימות בנק11 לפני שינוי השיטה

זהו השלב הבא לביצוע; טרם הורץ. הביקורת על ציונים שמורים כבר הושלמה ב־`results/prmbench_lsml_month_audit_v4/`. היא בדקה את כל האוכלוסייה, אך כיול OOF רטרוספקטיבי אינו תחליף להפרדת fit/calibration/evaluation.

1. לקבע את 11 הערוצים, סדרם, סימניהם וה־step readout מתוך `step_level_bank_baseline_v1` ואת `DERIVATIVE_CHANNELS.npz['level']` שב־token worktree. להעתיק לקונפיגורציה את פרמטרי המתכון המקורי, לא ברירות מחדל מגרסה אחרת. אין להוסיף depth/novelty. לשחזר את הציונים והמשקלים השמורים לפני fitting חדש, tolerance ‏1e-12.
2. לשמור את חמשת source folds הקיימים. עבור evaluation fold מספר k, calibration הוא fold הבא בסדר מזהי ה־folds הקפוא, במחזור; שלושת ה־folds האחרים בלבד משמשים fit. אותו מודל מנבא calibration ו־evaluation. לא ללמוד קבוצות, mean/scale או משקלים מאף אחד משני ה־folds המוחזקים. זו הקצאת שלושה folds לאימון, בשונה מארבעה בניסוי המקורי; לתעד זאת ולא לכנותה replay זהה.
3. ארבע זרועות: bank11 L-SML; equal באותו בנק ובאותו preprocessing; equal על אותה חלוקה שנלמדה ב־fit, במשקל `1/(K*group_size)` לכל ערוץ; CT7 הקפוא. אם הלומד מבצע donor-column standardization, להחיל אותו גם על שתי בקרות בנק11. לקבע את ה־partition בשלוש זרועות בנק11 כדי לבודד משקלים; אין fit של קבוצות לבקרת equal הרגילה ואין לבחור קבוצות באמצעות labels.
4. פלט ראשי: answer-z סופי ואז q80 מתוך calibration scores של כל זרוע בנפרד, ללא labels. לשמור גם raw+q80 כפאנל רגישות משני. להעתיק בדיוק את הגדרת quantile ואת כלל השוויון לסף מה־audit, ולשמור סף לכל fold. כל source group מקבל החלטה רק בהיותו ב־evaluation; לא לבחור סף לפי PRMScore.
5. שישה ניגודים ראשיים קפואים: L-SML מול כל אחת משלוש הבקרות, בכל אחד משני endpoints — within-AUC ו־official PRMScore אחרי answer-z. bootstrap לפי source group ותיקון Bonferroni על שישה ניגודים. לדווח effect ו־CI גם כאשר אינם חיוביים. raw, תתי־קבוצות ו־PB הם ניתוחים משניים, לא נתיב חלופי להכרזה על הצלחה.
6. לשמור per-fold groups, weights, כל סטטיסטיקות הנרמול, מזהי fit/calibration/evaluation, scores, decisions ו־confusion counts. למדוד CPU fit חד־פעמי בנפרד מעלות scoring בזמן ריצה. אין GPU, inference או שינוי cache. הצלחה כאן מאמתת donor-fitted fusion; אינה מוכיחה adaptation מתוך תשובה יחידה.

מיצוע הקבוצות הוא בקרה חשובה: הביקורת כבר מצאה יתרון דירוג מעבר אליו, אך יתרון PRMScore של המשקלים הנלמדים עדיין אינו מובהק לאחר תיקון. זהו פער הראיות שהניסוי נועד להכריע, לא תוצאה שיש להבטיח מראש.

## 5. שלב R1 — L-SML מקומי על בנק11, לאחר R0-B

שאלה: האם למידת תלות ומשקלים מתוך התשובה מפיקה ערך מעבר לאותו בנק עם משקלים קבועים?

ארבע שורות בלבד: CT7 הקפוא; equal token fusion על בנק11; equal על קבוצות ה־L-SML של אותה תשובה; continuous L-SML answer-local על בנק11. שתי שורות ה־equal הן בקרות, אינן מועמדות לקידום. זהו שינוי גם ברמת ה־fit וגם בהיקפו מול מועמד ה־step ההיסטורי; אין לייחס את ההפרש מול R0-B לגורם יחיד. לשמור את זרוע ה־step ההיסטורית כייחוס נפרד.

1. לשחזר את 11 הערוצים מתוך TOKEN_MATRICES וה־manifest של token-probability-fusion-v1. להחיל validity ו־step spans זהים על כל זרועות בנק11; לשמר preprocessing מקורי ולתעדו. אין להעתיק אוטומטית את despike_step0 של CT7 לבנק אחר ואין להכניס digit views.
2. לנרמל כל ערוץ על הטוקנים התקפים של אותה תשובה בלבד, עם mean/std; constant channel מסומן ונגרע לפני fit. לשמור את סטטיסטיקות הנרמול לשלב predict.
3. לבחור fit rows באופן דטרמיניסטי מכל שמונה טוקנים תקפים, החל מהראשון. זהו דילול לצמצום תלות, לא הוכחת עצמאות. להחיל את אותן rows על כל fit/baseline; scoring מתבצע על כל הטוקנים התקפים.
4. נדרשים לפחות `3*p_active` fit rows ולפחות שלושה ערוצים פעילים. continuous L-SML משתמש ב־`method='residual', groups=None, small_m_guard=True`; להקפיא בקובץ config את ברירות המחדל המדויקות של K_range מה־commit ששימש. אין לשנות את family partition לפי תוצאות labels. זו אינה C2 הישנה: שם ה־fit אסף donor answers.
5. להמיר את metadata למשקלים באמצעות `continuous_lsml_weight_vector`, ולוודא באמצעות replay שה־fit/predict משתמשים באותו scale. לכוון סימן לפי covariance עם סיכון equal מקומי; zero covariance או משקלים לא סופיים הם fit failure מפורש. ה־equal מכוון סימן בלבד, אינו הפלט הנלמד.
6. להפיק token score, masked Top10 לכל official step, ואז answer-z. לא להשוות למיצוע שנעשה בסדר פעולות אחר ולייחס את ההפרש רק ללמידה.
7. בכישלון fit: native מסומן invalid. לפאנל deployment מלא משתמשים בערוץ chosen-token הקיים בלבד, באותו readout, עם סימון fallback מפורש. **אין fallback שקט למיצוע.** בחוסר כיסוי טוקנים יש להפסיק ולתקן את חוזה הקלט, לא להמציא ציון.

הניגודים הראשיים: L-SML מול equal, מול learned-partition equal, ומול CT7. לדווח full-policy וגם native עם אותה מסכה בשני צדי כל השוואה. זיהוי groups ומשקלים אינו הישג בפני עצמו; למדוד שינוי דירוג ומקרי rescue/break.

## 6. שלב R2 — residual מקומי כתוספת לאותו fusion

להתחיל רק לאחר שיש replay יציב של R1; לא לשנות readout, fusion ו־predictor יחד. זהו ניסוי חדש ב־fit scope, לצד תיקון normalization של S2.

הקלט הוא בנק 11 הערוצים הקיים. נרמול token median/IQR נשמר לצורך התאמה למקור, אך הערוצים אינם מתמזגים לפני step normalization. לחשב residual prediction ממטריצת 16 lags, 16 presence bits ו־relative position; ridge alpha=1 ואינטרספט לא מענישים. אין target labels.

למנוע residuals מתוך אימון על אותן מטרות: לחלק את אינדקסי הטוקנים לחמישה בלוקים רציפים, `array_split`. עבור בלוק target מוחזק בחוץ, להסיר מהאימון כל row שה־target שלו או אחד מ־16 טוקני ההיסטוריה שלו שייך לבלוק זה. להתאים ridge על היתר ולנבא את הבלוק. normalization נשאר answer-local offline. נדרשות לפחות שתי שורות אימון; אחרת predictor=0 ומסמנים predictor fallback. לשמור את כל מסכות exclusions לבדיקה. אין לקרוא ל־cross-fit זה causal forecasting.

לכל step/channel: `top5(residual)` ואז mean/std על צעדי אותה תשובה בנפרד לכל ערוץ. מנבא האפס מקבל בדיוק אותו treatment: residual שלו הוא האות עצמו, **לא וקטור אפסים**.

שלוש מטריצות: level-only; level+zero-predictor residual; level+cross-fitted predictive residual. בכל אחת אותו learner שנקבע מראש ואותם hyperparameters. מכיוון שהבנק כאן ברמת step, נדרשת בקרה level-only **באותה רמת step**, ולא השוואה ישירה ל־R1 token learner כראיה לתרומת residual. נדרשים `3*p_active` צעדים ל־fit native. אם native coverage נמוך מ־80% ב־PRMB, לעצור את הפרוטוקול הזה כלא מתאים לרזולוציית step; אין לעבור בדיעבד ל־donor fitting או לשנות את guard כדי להציל אותו. התאמה ברמת token/window תהיה שלב חדש ומוגדר בנפרד.

ניגודים: predictive מול zero; predictive מול level-only; learned מול equal על אותו בנק. שני הראשונים נדרשים כדי לייחס תועלת לחיזוי ולא רק לשכפול ערוצים או לנרמול. רישום שונות כל ערוץ, משקל fusion, condition number, MSE מחוץ לבלוק ו־coverage. שיפור ב־MSE אינו קריטריון קידום.

## 7. שלב R3 — RBM קטן כחלופה ממוקדת

לשחזר את `spectral_utils/moment_rbm_fusion.py::fit_rbm` מתוך commit `72f1df685`, H1 exact Gaussian-Bernoulli, float64, L-BFGS-B, maxiter100 ו־fixed identity visible variance. לשמור את initialization וה־stopping tolerances בקונפיגורציה. אין לשגר את runner ההיסטורי בלי assets: checkpoints של moment banks אינם כולם קיימים מקומית.

הניסוי החדש משתמש **באותו בנק11 token ובאותו readout של R1**. אין לייחס השוואה ל־RBM12 ההיסטורי לבנק חדש בלבד. הניגודים: RBM fitted-logit מול אותו RBM initialized-untrained; מול L-SML; מול equal על אותו bank. כל ה־fit הוא answer-local CPU, עם אותה בחירת rows ומסכת coverage. סימן נקבע ללא labels; logits קבועים מראש, בלי בחירה בדיעבד בין posterior/logit. אי־התכנסות היא failure גם כש־loss סופי סופי.

ברשימת 130 התוצאות ההיסטורית, RBM12 shared-variance הגיע ל־0.747256 PRMB within-AUC, מתחת ל־CT7. זמני fit מצטברים: H1 בסיסי 146.63 שניות ל־13,769 תשובות; shared variance כ־384.56. אלה אינם זמני deployment כוללים. H4 לא התכנס בכל 13,769 התשובות; אין הצדקה לחזור כרגע להגדלת capacity או ל־CD.

## 8. Pseudo-labels ו־HARP במסגרת החדשה

Pseudo-labels יכולים לשמש בלי לאמן עוד LLM:

- **אמינות ערוצים:** לכל משפחה ליצור teacher משתי המשפחות האחרות; agreement מקומי מדרג משקל אמינות. אין להשתמש באותו ערוץ גם כ־target וגם כראיה עצמאית להצלחתו.
- **כיוון רכיב latent:** הסכמה עם pseudo-risk מקומי קובעת sign בלבד; אין לטעון ששינוי sign יצר מידע חדש.
- **מטרות זוגיות:** ranking רך בין צעדים, עם confidence mask; learner קטן על CPU ו־cross-fit מקומי, מול teacher עצמו ומול random/position controls.

אלה חלופות מותרות, לא שלוש זרועות להשקה אוטומטית. להתחיל באחת רק אם אבחון R1 מצביע על הבעיה המתאימה. לכל חלופה נדרשות בקרות teacher-only, ללא pseudo-label, random pseudo-labels ו־coverage-matched. הסכמה אינה אמת ואינה הוכחת עצמאות.

HARP/residual נשארים מקור למנגנון: לחלץ חלק מהאות שה־fusion לא מסביר ולהוסיף כערוץ. בנק 11 הסתברויות אינו hidden-state HARP. אם hidden states אינם כבר ב־cache, הרצת backbone כדי להשיגם אינה בהיקף. כל טענת HARP מחייבת מקור algorithm מדויק והתאמה לנתונים; כרגע השם הנכון הוא contribution residual על telemetry.

## 9. ניתוח, הצלחה ופלט מחייב

**מדד ראשי:** official pooled PRMScore בפאנל q80 label-free calibration קבוע. לצדו native published thresholds, ופאנל label-selected calibration נפרד. שיטה עם סף donor היא hybrid detector גם אם fusion answer-local. אין לשנות למדד שמנצח אחרי הריצה.

מדדי משנה: within-answer AUC על 6,030; any-error argmax על 6,035; first-error hit נפרד; negative-F1/positive-F1; pooled AUROC; PRMScore לפי קטגוריה; precision/recall של pseudo labels ו־coverage. PB: macro8 SLA, F1 עם gate קפוא ו־coverage. קטגוריות, עומק ואורך הם ניתוח heterogeneity, לא oracle selector למודל.

לכל contrast: identical IDs/masks; bootstrap מזווג לפי source question, 100,000 draws, seed20260923; לחשב PRMScore מחדש ממוני confusion בכל draw, לא ממוצע F1 של תשובות. corrected 95% intervals באמצעות Bonferroni על רשימת הניגודים הראשיים הקפואה של אותו שלב. לשמור B, valid_draws, paired_N, paired_groups ו־Monte Carlo resolution. אין למחזר CI שחושב על אוכלוסייה אחרת.

קריטריון להמשך פיתוח: תרומה נמדדת של L-SML מול equal המותאם, לצד איכות PRMB גבוהה; אין לדרוש ניצחון על CT7 כדי לחקור בנק שכבר הראה ערך. כדי לטעון שהמשקלים הנלמדים חיוניים ל־PRMScore נדרש CI מתוקן חיובי גם מול בקרת הקבוצות. CT7 הוא השוואה ראשית מדווחת, לא תנאי סף אוטומטי לפסילת המחקר. טענת עליונות עליו או על PRM מפוקח דורשת ניגוד ישיר מוצלח באותו calibration budget. קידום למאמר מחייב קיבוע המתכון ומבחן חיצוני שלא שימש לפיתוח. PB משני: לתעד מחיר ההעברה, לא לשנות את השיטה כדי למחוק אותו.

פלט מינימלי לכל agent:

| קובץ | שדות/תוכן מחייבים |
|---|---|
| CONFIG.json + CODE/INPUT_MANIFEST.json | commit, hashes, labels/folds, feature order, exact defaults, access scope, primary contrasts, CPU/disk budget |
| OOF_ANSWERS.csv | uid, source_group, fold, classification, n_steps, n_tokens, native/fallback, prediction, per-answer endpoints |
| STEP_SCORES.npz | scores לכל שיטה, uid order, offsets, labels-evaluator-only join, validity |
| FIT_DIAGNOSTICS.jsonl | fit rows, predictor exclusions, normalization, groups, weights, initial/final objective, optimizer success, failures |
| METRICS.csv / CONTRASTS.csv | benchmark, endpoint, population, paired_N/groups, estimate, delta, CI, adjustment, bootstrap draws |
| RUNTIME.csv | answer id, tokens, fit/score ms, CPU threads, RSS, CUDA_calls=0, extra_LLM_calls=0 |
| REPORT_HE.md | מה נלמד; האם הלמידה תרמה; מה עשה הנרמול; מה עשה הכיול; איפה יש נזק; האם התקציב נשמר; מה עדיין לא ידוע |

## 10. פקודות ומצב הביצוע

```powershell
python -B tests/test_runtime_fusion_protocol.py
python -B scripts/repair_runtime_fusion_reports.py
python -B scripts/window_answer_local_fusion_v2.py --source-root .worktrees/lsml-ct7-levers-run --output results/window_representation_b3_v2
```

פקודת תיקון הדוחות כבר בוצעה; היא מסרבת לדרוס את גרסת הפלט. שש בדיקות עברו. ריצת window v2 דורשת לפחות 256 MiB פנויים. חסימת האחסון טופלה בעקבות אישור Omri לניקוי worktrees: שני עותקים לא פעילים ומגובים הוסרו, וכ־3.94GB היו פנויים אחרי הניקוי. CLAUDE.md שוחזר לקובץ עצמאי. R1–R3 הם פרוטוקולים לביצוע עתידי תחום; אין תוצאה חדשה שלהם במסמך זה.
