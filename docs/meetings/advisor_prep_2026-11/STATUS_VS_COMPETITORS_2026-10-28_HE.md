# הערכת מצב מול המתחרים וכיוון התזה (2026-10-28)

**בסיס:** `claude/prm-boosting` @ `febbe005` (המחשב השני, עד Step 482 + `POSITIONING_UPDATE_2026-10-28.html`),
`claude/prm-new-features-2026-10-26` @ `e4f6d196`, ו-`claude/prm-boost-missing-benchmarks-2026-10-26` @ `109bf203`
(8 בנצ'מרקים, labeled eval, CROP protocol). כל המספרים הם development data.

## 0. בשורה אחת

- ב-PRMBench אנחנו "מעל PRISM" רק בנקודת עבודה (כלל GAP). ב-TPR קבוע, ה-PRM החזק לבד שווה ל-CHAIN.
- על הבנצ'מרקים הראשיים של CROP (CRV, ‏PRM800K, ‏GSM8K) ה-PRM לבד מנצח את ה-fusion בפער גדול (פי 2–4 ב-kept).
- הרווח האמיתי והעקבי: PRMs חלשים ב-PRMBench, קטגוריית redundancy, ו-DeltaBench (תשובות ארוכות).
- המלצה: לשנות את הטענה המרכזית מ-"מנצחים מתחרים" ל-"מתי fusion בלי תוויות עוזר, ולמה" (תנאי P2), ולהוסיף
  view חדש עם רעש נפרד (counterfactual PRM reward, דורש GPU) כניסיון אחד מוגדר מראש.

## 1. אימות העבודה של המחשב השני

| מה נבדק | תוצאה |
|---|---|
| מספרי טבלה 1 בדף (PRMScore, ‏FPR ב-TPR קבוע, ‏CROP, ‏ARES, ‏SLIDER) מול `stage2/COMPETITORS.csv` ו-`COMP_*.csv` | **תואם** בכל התאים שבדקתי (68.01 / 70.40 / 69.26 / 68.78; ‏47.70 / 47.78 / 50.63; ‏4.83 / 17.24 / 23.33 / 20.84) |
| CROP Direct PRM, ‏fixed grid | 4.83, תואם ל-4.8 של CROP Table 6 (parity) |
| טבלה 2 (Socratic, ‏Hard2Verify) | תואם ל-COMPETITORS / h2v_holdout |
| `prm-new-features` | ענף orphan (snapshot מלא של הריפו, אין merge base עם `prm-boosting`). צריך cherry-pick של 8 קבצים בלבד, לא merge |
| **חסר בדף** | (א) PRMScore ב-TPR קבוע (סעיף 2 כאן); (ב) תוצאות 8 הבנצ'מרקים החדשים (סעיף 3). שניהם משנים את המסקנה |

## 2. PRMBench: הרווח לפי סוג PRM

PRMScore. ‏GAP ו-Q80 הם כללי ספירה בלי תוויות. "TPR קבוע" = סף נבחר עם תוויות כך ש-TPR = 87.3 (כמו PRISM).

| PRM | PRM (GAP, ‏prob) | CHAIN (GAP) | CHAIN (Q80) | **PRM ב-TPR קבוע** | **CHAIN ב-TPR קבוע** | Δ ב-TPR קבוע |
|---|---|---|---|---|---|---|
| Qwen-7B | 68.01 | 70.40 | 68.25 | 68.26 | 68.20 | **−0.06** |
| Qwen-72B | 69.49 | 71.09 | 68.90 | 69.22 | 68.94 | **−0.28** |
| PURE | 66.24 | 69.51 | 67.29 | 66.41 | 67.43 | +1.02 |
| RLHFlow | 59.39 | 66.25 | 63.92 | 55.26 | 64.18 | **+8.92** |
| Math-Shepherd | 60.21 | 66.39 | 64.37 | 53.11 | 64.36 | **+11.25** |
| PRISM (אומן מחדש, Qwen-7B) | — | — | — | — | — | 68.0 |

- ב-PRMs החזקים, ה-+2.4 מעל PRISM הוא אפקט של נקודת עבודה. בסף זהה, CHAIN = PRM.
- PRISM עצמו (68.0) שווה ל-Qwen-7B עם הזזת סף בלבד (68.26). זה טיעון חזק נגד PRISM, לא בעדנו.
- ב-PRMs החלשים יש רווח ranking אמיתי (+9 עד +11), אבל הם עדיין לא מגיעים ל-Qwen-7B לבד (64 מול 68).

## 3. מעבר ל-PRMBench: 8 הבנצ'מרקים החדשים (Qwen-7B)

מקור: `results/prm_boost_crop_ares_benchmarks_v1/eval/SUMMARY_TABLES.md`.

| בנצ'מרק | PRM within-AUC | CHAIN | HALF | position only | CROP kept@5%: PRM | CROP kept: CHAIN | מי מנצח |
|---|---|---|---|---|---|---|---|
| CRV Arithmetic | .902 | .768 | .867 | **.957** | **97.4** | 24.0 | PRM (ו-position) |
| CRV GSM8K | .743 | .646 | .707 | **.769** | **79.8** | 32.5 | PRM |
| PRM800K | .929 | .835 | .913 | **.995** | **68.3** | 35.0 | PRM (ו-position) |
| Math-Shepherd | .445 | .435 | .448 | .355 | ~1–2 | 2.3 | אף אחד (<.5 לכולם, כולל CROP) |
| DeltaBench (~41% כיסוי) | .656 | .671 | **.683** | .555 | — | — | **fusion** (macro-F1 56.3 → 59.3) |
| Socratic | 70.10* | 66.89* | — | — | — | — | PRM |
| Hard2Verify (177) | 45.82* | 44.27* | — | — | — | — | SUB5 לבד 49.35 (מדגם קטן) |
| ProcessBench | — | — | — | — | — | — | נכשל בכלל שנרשם מראש (1/5) |

\* PRMScore, לא within-AUC.

- ב-CROP protocol ה-fusion מאבד את הכיול הגלובלי של ה-PRM (נרמול per-answer). זה ההסבר לפער הגדול ב-kept.
- position-only מנצח את ה-PRM ב-CRV ו-PRM800K: השגיאות שם בצעד האחרון. אלה בנצ'מרקים עם מבנה שגיאה פשוט.
- Math-Shepherd: הכול מתחת ל-.5, כולל position. כדאי לבדוק שוב את ה-polarity של התוויות לפני שמדווחים.
- טלמטריה standalone: within-AUC ‏.60–.77 ב-ProcessBench, ‏.62–.70 ב-Socratic, ‏~.5 ב-CRV / ‏PRM800K / ‏Math-Shepherd.
  L-SML ≈ MEAN כמעט בכל תא.

## 4. טבלת מצב מול כל מתחרה

| מתחרה | מה הם טוענים | איפה אנחנו | מצב |
|---|---|---|---|
| **PRISM** (2606.09078) | PRMBench 65.5 → 68.0 באימון מחדש | PRM + הזזת סף = 68.26; ‏CHAIN 70.40 רק ב-GAP | **שוויון.** הטענה הכנה: "אימון מחדש לא נדרש, סף מספיק". זה לא רווח של השיטה שלנו |
| **CROP** (2605.30085) | conformal clean prefix, ‏Trace+PRM מפוקח, ‏PRMBench 21.8 | PRMBench: ‏CHAIN 23.3 (quantile grid), אבל LR מפוקח שלנו 24.8 על fixed grid. ‏CRV / ‏PRM800K / ‏GSM8K: **מפסידים פי 2–4**. העברה ל-Socratic: risk 6.7% > 5% | **מפסידים** על הבנצ'מרקים הראשיים שלהם |
| **ARES** (2507.12948) | macro-F1, ‏DeltaBench | PRMBench ‏7B: ‏68.10 מול PRM 68.17 (שוויון). ‏DeltaBench: ‏+3 עד +9.5 macro-F1 | **יתרון ב-DeltaBench בלבד**, השוואה בקירוב (subset לא ידוע, כיסוי 41%) |
| **SLIDER** | redundancy, ‏Step-RRI ‏61.9 | TELE ‏75.7, ‏CHAIN 74.3 בלי קריאות LLM | **יתרון**, אבל הפרוטוקול שלהם לא מוגדר. זו הנישה הכי חזקה שלנו |
| **Mind the Gap** | entropy drops, טלמטריה בלי PRM | `evidence_drop_risk` קיים ב-N56, לא נמדד | **פתוח.** זה המתחרה הישיר של מסלול B, וחובה למדוד |
| **FUSE / Weaver** | ensemble בלי תוויות, response-level | לא step-level | אנלוגיה. FUSE עצמו מראה ש-spectral ensemble מפסיד ל-mean ב-7/10 |
| **ScalePRM / uPRM** | PRM מאומן בלי תוויות | לא נמדד אצלנו | related work |

## 5. מה זה אומר על הטענות

| טענה | מחזיקה? | הבסיס |
|---|---|---|
| "CHAIN מעל PRISM ב-PRMBench" | **לא בצורה הזו** | בסף זהה CHAIN = PRM ב-7B ו-72B |
| "fusion מחזק PRMs חלשים ב-PRMBench" | **כן** | +9 עד +11 ב-TPR קבוע (RLHFlow, ‏MS) |
| "fusion מחזק PRMs באופן כללי" | **לא** | CRV / ‏PRM800K / ‏GSM8K / ‏Socratic / ‏ProcessBench |
| "טלמטריה בלי PRM ברמת PRMs מאומנים" | **רק ב-PRMBench** | standalone ~.5 על CRV / ‏PRM800K; ‏L-SML ≈ mean |
| "ranking בתוך התשובה מעבר ל-position/length" | **כן, ב-PRMBench בלבד** | residualization 5/5 |
| "יש תנאי שחוזה מתי fusion עוזר" (P2) | **חלקית** | עם תוויות 13/15 תאים; ה-gate בלי תוויות נכשל |
| "redundancy / תשובות ארוכות" | **כן, בקירוב** | SLIDER 75.7 מול 61.9; ‏DeltaBench +3 |

## 6. המלצה לכיוון התזה

**הטענה המרכזית המומלצת:** אפיון של fusion בלי תוויות לזיהוי שגיאות ברמת הצעד, עם תנאי מתי הוא עוזר.

1. **לשנות את ה-headline.**
   - לא "מעל PRISM / ‏CROP". זה לא יחזיק מול reviewer שיריץ TPR קבוע או CRV.
   - כן: "fusion עוזר כשה-PRM חלש או כשסוג השגיאה לא בתחום האימון שלו (redundancy, תשובות ארוכות), ומזיק
     כשה-PRM חזק ומכויל. תנאי P2 מסביר את הסימן."
   - שני ממצאים צדדיים חזקים ושימושיים לקהילה: (א) PRISM ≈ הזזת סף; (ב) position-only מנצח PRMs ב-CRV ו-PRM800K.
2. **לסגור את תנאי P2 על 8 הבנצ'מרקים.**
   - יש כבר 40 תאים (5 PRMs × 8). זה מבחן אמיתי מחוץ ל-PRMBench, ב-CPU.
   - לבדוק אם P2 חוזה את הסימן גם שם. אם כן, זו התרומה המתמטית של התזה.
3. **ניסיון אחד מוגדר מראש לשיפור (GPU).**
   - counterfactual PRM reward (question-blind difference). זה ה-view היחיד שהרעש שלו לא משותף עם ה-PRM
     (ענף `prm-new-features`, §4).
   - עלות: בערך pass אחד של PRM ל-7B. pilot על PRMBench + ‏Socratic + ‏CRV GSM8K.
   - next-step PRM risk (חינם, r = .16 עם ה-PRM): להריץ קודם ב-CPU.
   - כלל הצלחה נרשם מראש, לפני ההרצה.
4. **להקפיא test set.** הצעה: CRV GSM8K או DeltaBench המלא (לפי מה שהמנחים יעדיפו). לא לגעת בו עד הגרסה הסופית.
5. **לסגור את החובות של מסלול B.** Mind the Gap על PRMBench, ‏CROP על fixed grid ל-CHAIN/TELE.

**מה לא לעשות:**
- עוד חיפוש בנקים / תת-סטים / אלגוריתמי SML. Steps 475–482 מראים שהשקלול הוא לא צוואר הבקבוק.
- להציג את SUB5 כתוצאה. הוא in-sample.

## 7. לפני הפגישה (CPU, לפי סדר)

| # | משימה | למה | איפה הנתונים |
|---|---|---|---|
| 1 | להוסיף לדף ה-positioning את PRMScore ב-TPR קבוע + טבלת 8 הבנצ'מרקים | בלי זה הדף מטעה | כבר ב-CSV |
| 2 | P2 על 40 התאים החדשים | התרומה המרכזית | Drive `prm_boost_crop_ares_benchmarks_v1/` |
| 3 | בדיקת polarity של התוויות ב-Math-Shepherd | כל השיטות < .5 | `make_inputs.py` |
| 4 | Mind the Gap + ‏CROP fixed grid | חובות מהדף | AIRCC / ‏Drive |
| 5 | next-step PRM risk כערוץ 17 | חינם, view כמעט בלתי תלוי | rewards שמורים |

## 8. שאלות למנחים (מעודכן)

1. האם טענת "אפיון + תנאי P2" מספיקה לתזה, או שנדרש שיפור מעל SOTA?
2. אישור GPU ל-counterfactual PRM reward (pilot אחד, כלל מראש).
3. איזה test set להקפיא?
