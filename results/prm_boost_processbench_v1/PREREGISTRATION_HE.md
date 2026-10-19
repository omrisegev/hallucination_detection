# PRM boosting על ProcessBench: רישום מראש (אושר, 2026-10-19)

**סטטוס: אושר.** עמרי אישר את החלטות A–D ב-2026-10-19, כפי שהומלצו: "N16 with the digit fix, CHAIN with no new
anchoring, the NC rule with a one-point margin, and a 3-of-5 verdict with at least one strong PRM".
הרישום הוא ה-commit שבו הקובץ הזה נכנס ל-`claude/integration-2026-10-17`. מרגע זה אסור לשנות אף פרט כאן לפני שה-verdict
נכתב. שינוי אחרי האישור מקבל מזהה חדש ומדווח כזרוע שלא נרשמה מראש, כמו CHAIN על Socratic.

**מה הרישום קובע:** איזה מתכון, איזה כלל לזיהוי השגיאה הראשונה, ומה ייחשב הצלחה. הוא לא משנה את ה-inference.
ה-telemetry על ה-GPU (`HANDOFF.md`, שלבים 0–6) זהה לכל מתכון, ולכן אפשר להריץ timing ו-full run כבר אחרי האישור.
המתכון והכלל משפיעים רק על ה-downstream ב-CPU (שלב 7) ועל ההערכה המקומית (שלב 9). שלב 7 לא ירוץ לפני השינויים בסעיף 8.

## 1. מה כבר ידוע

- **ProcessBench לא "נקי" לגמרי.** קו ה-telemetry של Qwen3 כבר השתמש בו לפיתוח, וה-rewards של PRM-7B כבר נמדדו עליו
  (job 177776: F1 ממוצע 72.94; within-AUC ‏0.939 על 1,949 התשובות שבהן הוא מוגדר; `README.md`).
  ה-telemetry של ה-backbone של ה-PRMs לא נאסף עליו, ושום ציון fused לא חושב עליו.
  לכן זה מבחן held-out למתכון ה-boosting. זה לא מבחן publication שלא נגעו בו.
- **הציפייה המוקדמת לא טובה.** ב-Socratic, כשה-PRM חזק, ה-fusion פגע. CHAIN_N16 הפסיד ל-PRM ב-4 מתוך 5 PRMs, וניצח רק
  ב-RLHFlow (`../prm_boost_external_v1/chain_socratic/README.md`). ב-ProcessBench ה-PRM חזק עוד יותר (0.939),
  ולכן מקום השיפור קטן.
- **כלל Q80 לא מתאים ל-ProcessBench.** הוא מסמן צעדים כמעט בכל תשובה, גם בתשובות נכונות (1,179 מתוך 3,400 בלי שגיאה).
  ה-PRM לבד יורד איתו מ-72.94 ל-34.76 F1. לכן Q80 לא יכול להיות הכלל הראשי כאן.
- **שאלות חוזרות.** 988 מתוך 3,400 התשובות חולקות שאלה עם תשובה אחרת (2,842 שאלות שונות). ה-bootstrap צריך לדגום
  לפי שאלה ולא לפי תשובה.

## 2. המתכון הראשי (זרוע אחת)

**`CHAIN_N16`**, כמו ב-bank sweep וב-Socratic (`../prm_boosting_bank_sweep_v1/lsml_bank_sweep.py`), עם שני תיקונים
שנכנסו ב-`db054926` לפני שראינו נתונים של ProcessBench:
1. **מזהי digits לפי ה-tokenizer** (`spectral_utils/digit_token_ids.py`). ב-Qwen, ב-PURE וב-Llama זה זהה ביט-לביט
   ל-15..24. רק ב-Math-Shepherd (Mistral) ערוצי ה-digits מפסיקים להיות אפס.
2. **הגנה מעמודות קבועות לפני L-SML** (`lsml_continuous_guarded`). בלי עמודה קבועה היא זהה ל-`lsml_continuous`.

**איך ה-PRM מעוגן במתכון הזה (בלי שינוי מ-PRMBench):**
- ה-PRM נכנס כ-view נפרד, `z(logit(1-reward))` מנורמל בתוך התשובה, בשלב ה-cross של CHAIN, לצד ציוני הקבוצות של ה-telemetry.
- המשקל שלו נקבע רק על ידי L-SML. אין משקל קבוע ואין רצפה למשקל.
- הסימן של הציון המשולב מכוון כך שיתאם חיובית עם ה-PRM.

**החולשה הידועה:** ב-Socratic ו-ב-Math-Shepherd, L-SML נתן ל-PRM משקל קטן מדי (`HANDOFF_MS_DIGIT_FIX_2026-10-19.md`, סיבה 2).
המתכון הזה לא מתקן את זה. תיקון עיגון אחר הוא שיטה חדשה, וצריך לתכנן ולהקפיא אותה על PRMBench ו-Socratic **לפני**
ש-ProcessBench נפתח (החלטה B).

## 3. כלל השגיאה הראשונה: `NC` (native count, fused order)

לכל תשובה:
1. **שער:** אם אף צעד לא מקבל reward < 0.5 מה-PRM (כלל ה-tie של ProcessBench), החיזוי הוא ‎-1 (אין שגיאה).
2. **ספירה:** אחרת k = מספר הצעדים הלא-ריקים שה-PRM סימן.
3. **סדר:** מסמנים את k הצעדים עם הציון המשולב הגבוה ביותר. בשוויון, הצעד המוקדם קודם.
4. **חיזוי:** הצעד המסומן המוקדם ביותר.
5. **מקרה קצה:** אם ה-PRM סימן רק צעדים ריקים (k = 0), מחזירים את השגיאה הראשונה של ה-PRM. מספר המקרים האלה ידווח.

**למה הכלל הזה:**
- הוא לא משתמש ב-labels. סף 0.5 הוא ה-threshold המובנה של ה-PRM, ולא כוונן על ProcessBench.
- נקודת העבודה זהה ל-PRM. ההבדל מ-PRM_native נמצא רק במיקום השגיאה בתוך תשובות שה-PRM כבר סימן, ולכן ההשוואה
  בודקת בדיוק את מה שה-fusion אמור לתרום.
- הציון המשולב מנורמל בתוך התשובה, ולכן הוא לא יודע לבד אם יש בתשובה שגיאה. השער חייב לבוא מסקאלה שלא נורמלה.
- זה אותו כלל כמו NATIVECOUNT ב-`../prm_boosting_controls_v1/lsml_controls.py`, שהוגדר לפני ProcessBench.
  ב-PRMBench, עם LSML_CONT כציון, הוא עקף את PRM_native ב-4 מתוך 5 PRMs (Qwen-7B ‏67.11 מול 65.49) ונפל רק ב-Math-Shepherd
  (47.00 מול 47.02). הוא היה נמוך מ-Q80 (69.48), אבל שם המדד הוא PRMScore על כל הצעדים ולא השגיאה הראשונה
  (`../prm_boosting_controls_v1/README.md`).

Q80 ימשיך להיות מדווח לצורך רציפות, כזרוע משנית.

## 4. קריטריון ההצלחה

**לכל PRM**, CHAIN_N16 עם NC מול PRM_native, על אותן תשובות (לשני הצדדים אותו PRM ולכן אותו כיסוי):
- **E1 (דירוג):** within-AUC גבוה יותר, עם CI של 95% מעל 0.
- **E2 (החלטה):** F1 הרשמי הממוצע לא נמוך ביותר מנקודה אחת: הגבול התחתון של ה-CI של ההפרש גבוה מ-‎−1.0.
- ה-PRM מצליח רק אם E1 ו-E2 מתקיימים שניהם.

**Verdict כללי:** הצלחה ב-3 מתוך 5 ה-PRMs לפחות, ואחד מהם לפחות Qwen-7B או Qwen-72B. התנאי השני קיים כי ב-Socratic
הניצחון היחיד היה על ה-PRM החלש ביותר.

**טענה נפרדת: האם L-SML מוסיף מעבר לממוצע.** CHAIN_N16 מול HALF_N16 ומול EQ_N16 ב-within-AUC.
- זו לא דרישה ל-verdict של ה-boosting.
- היא דרישה לטענה ש-L-SML עצמו תורם (החלטת עמרי מ-2026-09-23).
- HALF ו-EQ הם בקרות בלבד, לא מתכונים.

**Bootstrap:** paired, ‏1,000 דגימות, seed 20261018, מרובד לפי subset ומקובץ לפי שאלה (טקסט מנורמל רווחים).
- תשובות שה-PRM לא הצליח לקרוא בגלל אורך נספרות ככישלון ב-F1. אין truncation.
- כמה PRMs ייבדקו במקביל: ה-CI לא מתוקן להשוואות מרובות. במקום זה הכלל דורש 3 מתוך 5 ולא אחד.

**המטריצה הקודמת** (`docs/post_advisor_2026-10-17/05_PRM_BOOSTING_EVIDENCE_PLAN_HE.md`, סעיף 4) דרשה הצלחה ב-2 מתוך 3
בנצ'מרקים חיצוניים. Socratic נכשל ו-Hard2Verify לא הורץ לפי התנאי של עמרי, ולכן הכלל הזה כבר לא יכול להתקיים כפי שנוסח.
ProcessBench מקבל verdict משלו, והמסגור הכללי נקבע מראש בסעיף 6.

## 5. זרועות משניות ובקרות (מדווחות, לא יכולות להחליף את הראשית)

- **telemetry:** CHAIN_N13 (בלי digits), PRMG_N16/N13, TELE_N16/N13, LSML_N16_PRMG, ‏LSML_CONT. ל-LSML_CONT
  ה-channels נבחרו עם labels של PRMBench, וזה יסומן בכל טבלה.
- **ממוצעים כבקרה:** HALF_N16, ‏EQ_N16, ‏MEAN_N16.
- **מיקום ואורך:** POS_ONLY ו-LEN_ONLY. ב-Socratic, מיקום בלבד היה טוב מה-telemetry בכל 5 ה-PRMs.
- **בקרת כלל:** PRM_Q80, כדי להפריד בין "הציון טוב יותר" לבין "הכלל שונה".
- **מדדים להשוואה לספרות:** PRISM (accuracies ברמת התשובה, Table 6) ו-CROP (pooled AUROC על היעד first_only).
- **תקרות עם labels:** מסומנות oracle ולא שיטה.

כל הזרועות מחושבות עם כלל NC ועם Q80. כל זרוע שלא מופיעה כאן ותתווסף אחרי פתיחת ProcessBench תסומן post-registration.

## 6. איך נקרא את התוצאה (נקבע מראש)

- **ProcessBench מצליח:** ה-boosting עוזר ב-PRMBench וב-ProcessBench ולא ב-Socratic. מציגים את זה ככה, כולל Socratic.
- **ProcessBench נכשל:** מציגים את ה-boosting כממצא של PRMBench בלבד, עם הסייגים. לא מחפשים כלל או זרוע אחרת על ProcessBench.
- **E1 מתקיים ו-E2 לא:** ה-fusion מדרג טוב יותר אבל לא מתורגם להחלטה. מדווחים ככה, בלי לכוונן את NC.

## 7. ההחלטות (אושרו ב-2026-10-19)

- **A. הבנק הראשי: N16 עם תיקון ה-digits (אושר).** זה תואם את ההחלטה מ-2026-09-30 להשאיר digits בקו
  label-free. N13 בלי digits מדווח כזרוע משנית.
- **B. עיגון ה-PRM: CHAIN כמו שהוא, בלי עיגון חדש (אושר).** זו הזרוע שנבחרה מראש; תיקון עכשיו היה מתוכנן על Socratic
  שכבר ראינו.
- **C. הכלל: NC כמו שהוא מוגדר בסעיף 3, ומרווח של נקודה אחת ב-E2 (אושר).**
- **D. ה-verdict: 3 מתוך 5, עם Qwen-7B או Qwen-72B ביניהם (אושר).**

## 8. מה צריך לשנות בקוד לפני ה-downstream (לא לפני ה-timing)

1. **Math-Shepherd:** לאמת על AIRCC את מזהי ה-digits של Mistral (`HANDOFF_MS_DIGIT_FIX_2026-10-19.md`, שלב 2).
2. **`aircc/pb_down.sh`:** כרגע בונה עם ה-builder הישן (`code_thinkfix_bank_553de79`) ובודק עם `cmp` שהוא זהה לגרסה של
   Socratic. צריך לעבור ל-`prm_boost_build_bank_external_v2.py --tokenizer`, ול-`lsml_continuous_guarded` ב-
   `score_external` וב-`score_chain_pb.py`.
3. **`aircc/score_chain_pb.py`:** להוסיף EQ_N16 ו-MEAN_N16, את HALF/EQ/MEAN על N13, ואת POS_ONLY ו-LEN_ONLY.
4. **`local/eval_processbench.py`:** להוסיף את כלל NC ואת ה-bootstrap המקובץ לפי שאלה, עם בדיקות על fixture.
5. **בדיקת parity:** בבנקים של Qwen/PURE/RLHFlow, ה-builder החדש צריך לתת בדיוק את מה שנתן הישן.

כל השינויים האלה נכנסים ל-commit לפני שה-downstream רץ. ה-hash שלו יירשם כאן.
