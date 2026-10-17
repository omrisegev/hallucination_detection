# PRM boosting: תוכנית לאיסוף ראיות מקיף (2026-10-18)

**המטרה:** לבנות מטריצת ראיות שתאפשר להציג את ה-boosting כממצא כללי ולא כתוצאה של בנצ'מרק אחד, מודל אחד או בחירה בדיעבד. זו תוכנית בלבד. שום ניסוי לא הותחל.

## 1. מה יש היום

| PRM | backbone | PRMBench native → L-SML (PRMScore) | within-AUC |
|---|---|---|---|
| Qwen2.5-Math-PRM-7B | Qwen2.5 | 65.49 → 69.48 (+3.99 [3.31, 4.67]) | 0.801 → 0.850 |
| Qwen2.5-Math-PRM-72B | Qwen2.5 | 68.18 → 69.84 (+1.66) | 0.795 → 0.856 |
| PURE-PRM-7B | Qwen2.5 + base head | 65.31 → 66.69 (+1.38) | 0.763 → 0.811 |
| RLHFlow Llama-3.1-8B | Llama-3.1 | 54.43 → 63.53 (+9.10) | 0.662 → 0.772 |
| MathShepherd Mistral-7B | Mistral | 47.02 → 63.68 (+16.67) | 0.684 → 0.759 |

המקור הוא `results/prm_boosting_generalization_v1/run_v1/eval_lsml4/`. כל ה-CIs לא כוללים 0.

החורים במטריצה:
- **בנצ'מרק אחד:** כל המספרים הם PRMBench בלבד.
- **channels שנבחרו עם תוויות** של PRM-7B.
- **שני PRMs חסומים:** Skywork (n+1 rewards, מחכה להחלטה שלך) ו-ReasonEval (head לא מאומת).
- **אין PRM גנרטיבי.**
- **אין השוואה מוסדרת מול ה-leaderboard.**

## 2. מטריצת הראיות המוצעת

### ציר א: בנצ'מרקים (אותו מתכון קפוא)
| בנצ'מרק | מה הוא מוסיף | מצב הנתונים | עלות |
|---|---|---|---|
| PRMBench | הבסיס | קיים, 5 PRMs | — |
| ProcessBench | ייחוס מוכר, first-error. ה-PRM לבד חזק מאוד שם (0.944 AUC), אז זה מבחן קשה | ה-dataset ב-cache. telemetry של ה-backbone של PRM לא קיים | GPU, בערך כמו PRMBench לכל PRM |
| Socratic-PRMBench | מתמטיקה מבנה אחר, ה-PRM מוביל שם | רק reward (12/10), בלי telemetry | GPU |
| Hard2Verify | הוכחות, PRM חלש, Balanced F1 | רק reward (12/10) | GPU, קטן (200 תשובות) |
| מחוץ למתמטיקה (MedPRMBench, BonaFide או LSR-Ben) | האם זה כללי מעבר לתחום | handoff ל-BonaFide מ-4/10, לא רץ. צריך לבדוק פורמט ורישיון | CPU לבדיקת היתכנות, אחר כך GPU |

### ציר ב: מודלים
- **קיימים:** 5 PRMs, בשלוש משפחות backbone (Qwen2.5, Llama-3.1, Mistral).
- **להוסיף:**
  - **Skywork-PRM.** צריך החלטה שלך על ה-protocol: מה לעשות עם ה-reward הנוסף בסוף.
  - **ReasonEval.** צריך validation של ה-head.
  - **PRM מבוסס Qwen3**, אם קיים checkpoint פתוח. צריך לבדוק זמינות.
  - **PRM גנרטיבי** (ThinkPRM או GenPRM): שאלת ה-"thinking". צריך לבדוק זמינות ורישיון.
- **בקרה חשובה:** "Instruct" בלי PRM, כלומר backbone שלא אומן כמאמת. ב-PRM-7B כבר יש נתון: Instruct 0.7885 / 65.93 (`results/prm_boosting_v1/README.md`).

### ציר ג: בקרות על המתכון (CPU, על הנתונים הקיימים)
1. **בחירת channels בלי תוויות:** PRM כקבוצה נפרדת, ולצידו N16 מלא כקבוצה, בלי לבחור 3 channels. יש כבר נקודת ייחוס: ½·PRM + ½·N16 = 0.8405 / 68.53.
2. **ablation לכל channel, ו-null של channels מעורבבים בין תשובות:** האם הרווח תלוי במידע ברמת הצעד.
3. **בקרות מיקום ואורך:** residualization לפי מיקום הצעד ואורכו, כמו בבדיקות NR/NCL.
4. **כלל החלטה:** כמות מה-PRM (native) עם סדר מה-fusion, לצד Q80. בכל השוואה מדווחים גם within-AUC, שלא תלוי בכלל.
5. **פירוק לפי סוג שגיאה ובדיקת מנגנון fluency:** האם הרווח מרוכז ב-NR/NCL בכל ה-PRMs.
6. **סטטיסטיקה:** paired source-group bootstrap ו-Bonferroni על מספר ה-PRMs × בנצ'מרקים.

### ציר ד: השוואות
- ה-leaderboard של PRMBench: אותו evaluator, אותו subset ותיקון התוויות של Step 313.
- LR מפוקח על אותם PRM + channels: תקרה מפוקחת, כמו ב-LR-gap.
- הקשר בספרות: FUSE, Weaver ו-ScalePRM, שעובדים ברמת התשובה או ב-majority vote, מול ה-step-level שלנו.

## 3. סדר מוצע

1. **CPU, עכשיו:** ציר ג כולו והשוואת ה-leaderboard, על 5 ה-PRMs הקיימים. זה לא דורש GPU, ואחרי זה מקפיאים מתכון "נקי" (בלי בחירה עם תוויות).
2. **GPU, אישור 1:** telemetry של ה-backbone ל-5 ה-PRMs על Socratic, Hard2Verify ו-ProcessBench, עם המתכון הקפוא. לפני ההרצה: timing job קטן (עד 12 דוגמאות ועד שעת GPU אחת לכל זוג), ואז הערכת תקציב.
3. **החלטות שלך:** Skywork protocol, ובחירת בנצ'מרק מחוץ למתמטיקה.
4. **GPU, אישור 2:** PRMs נוספים (Skywork, Qwen3-PRM, גנרטיבי) ובנצ'מרק מחוץ למתמטיקה.

## 4. מה ייחשב "ראיה מספקת" להצגה

הצעה שצריך לקבוע מראש:
- רווח PRMScore חיובי עם paired CI מעל 0.
- רווח within-AUC חיובי ב-**לפחות 2 מתוך 3 הבנצ'מרקים החיצוניים** לכל PRM.
- המתכון הנקי לא נופל ביותר מנקודה אחת מהמתכון עם 3 ה-channels ב-PRMBench.
- ההבדלים לא מוסברים בבקרות המיקום והאורך.

אם זה לא מתקיים, מציגים את ה-boosting כממצא חזק ב-PRMBench בלבד, עם הסייגים.
