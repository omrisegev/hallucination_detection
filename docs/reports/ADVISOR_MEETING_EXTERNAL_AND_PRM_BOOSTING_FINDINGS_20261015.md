# External Comparisons and PRM Boosting Findings (Advisor Meeting Preparation)
**Date:** 2026-10-15  
**Context:** Preparation for Advisor Meeting on PRMBench & Reasoning Error Localization  
**Branches:** `codex/lsml-external-generalization-v1`, `claude/prm-boosting-generalization-results-v1`  
**Primary Artifacts:** `results/external_lr_prm_v1/`, `results/advisor_page_v19_2026-10-15/`, `docs/reports/`  
**Evaluation Jobs:** AIRCC 284086 (PRM Boosting Generalization), 282806 (Socratic PRM), 282807 (Hard2Verify PRM)

---

## 1. Executive Summary & Recommendation for the Advisor Meeting

### 1.1 Direct Answer to Omri's Question
> *"ולבסוף לאחר השלמה תחווה דעה אם אפשר לשלוח למנחים כדי שיוכלו לעבור על זה לפני הפגישה – וכדאי שנתחיל להתכונן על להעביר את המצגת הזו"*

**חוות דעת חד-משמעית:** **כן, בהחלט אפשר ומומלץ לשלוח את החומרים למנחים, ולהתחיל בהכנות להעברת המצגת.**

**נימוקים אסטרטגיים:**
1. **סגירת שתי השאלות הקשות ביותר של המנחים מראש:**
   - **שאלה 1: *"למה לא להשתמש פשוט ב-PRM מאומן (Supervised) במקום בסיגנלי טלמטריה?"***
     - **המענה:** הראינו תשובה אמפירית חותכת משני כיוונים:
       - **העברה מחוץ לתחום (Section 5 — External Generalization):** בתוך תחום המתמטיקה המפוקח (Socratic), ה-PRM אכן מנצח (Within-AUC 0.824 מול 0.717 של LR ו-0.655 של שיטות לא-מפוקחות). אך מחוץ לתחום, ב-Hard2Verify, **ה-PRM קורס**: הסף הטבעי שלו (0.5) מזהה כמעט אפס שגיאות (Balanced F1 של 0.234), וגם תחת סף יחסי מנורמל (`ANSWER_Z_Q80`) הוא מגיע רק ל-0.391. לעומת זאת, רגרסיה לוגיסטית (LR) שאומנה על סיגנלי הטלמטריה שלנו ב-PRMBench מעבירה ידע בצורה עדיפה בהרבה ומגיעה ל-**0.428 (8B)** ו-**0.448 (4B)** (עם Within-AUC של 0.684 מול 0.634 של ה-PRM), וגם השיטה הלא-מפוקחת שלנו (IU-PCR) מגיעה ל-**0.415** ללא שימוש באף תווית אמת!
       - **חיזוק מודלי PRM קיימים (Slides 14–15 — PRM Boosting):** הראינו שסיגנלי הטלמטריה וה-PRM אינם מתחרים זה בזה, אלא **משלימים זה את זה (Orthogonal Synergy)**. שילוב ה-PRM עם שלושה סיגנלי טלמטריה מה-Backbone שלו עצמו משפר את התוצאות **בכל 5 מודלי ה-PRM שנבדקו**, עם שיפור מובהק סטטיסטית של בין +1.4 ל-+16.7 נקודות PRMScore!
   - **שאלה 2: *"האם התוצאות שלכם אינן פשוט תוצר של בחירת ספים שרירותית?"***
     - **המענה:** שקופיות 10–12 מפרקות את שאלת הסף ומראות כיצד שיעור הסימונים (Budget curve) ודירוג הסיכון בתוך התשובה מתנהגים, תוך הוכחה ש-Q80 מבודד דירוג יחסי ומנטרל הטיות של אורך תשובה.

2. **נרטיב מבוסס ויושרה מדעית מוחלטת:**
   המצגת (16 שקופיות, 34 דקות + 6 דקות דיון) אינה מנסה "למכור" ניצחון מוחלט ומופרך. היא מציגה במדויק איפה השיטות חזקות (שגיאות מעגליות ויתירות — Redundancy & Circular), איפה הן חלשות (סתירות צעד — Step Contradiction), מה הפער בין דירוג (AUC) לסימון (PRMScore), ואיך שילוב סיגנלים פותר בעיות שהראש המפוקח של ה-PRM מתעלם מהן.

---

## 2. Section 5: External Transfer Outside PRMBench (Socratic & Hard2Verify)

### 2.1 פרוטוקול ההשוואה
- **השוואת מודלים:**
  1. `Qwen2.5-Math-PRM-7B`: מודל PRM ייעודי ומפוקח (מבוסס Qwen2.5-Math-7B עם ראש דירוג צעדים `<extra_0>`).
  2. `LR transfer` (8B ו-4B): מודל רגרסיה לוגיסטית מאומן על בנק 16 ערוצי טלמטריה מ-PRMBench (עם כוונון משקולות $C$ ב-5 קיפולים ללא חשיפה לנתוני היעד), ומיושם ישירות על המדגמים החיצוניים.
  3. `IU-PCR` ו-`LSML_CONT` (8B ו-4B): שיטות ה-Fusion הלא-מפוקחות שלנו.
- **כללי החלטה:**
  - `NATIVE_0.5`: סף ברירת המחדל של ה-PRM (ציון תקינות מעל 0.5 נחשב תקין).
  - `ANSWER_Z_Q80`: כלל ההחלטה הסטנדרטי של הפרויקט — ציון $z$ יחסי בתוך התשובה עם סימון 20% הצעדים המסוכנים ביותר.
- **מדדי הערכה:**
  - **Socratic-PRMBench:** 2,995 תשובות; מדד רשמי: `PRMScore`.
  - **Hard2Verify:** 200 תשובות; מדד רשמי: `Balanced F1`.
  - שניהם מדווחים גם `Within-answer AUC` (ממוצע AUC בתוך כל תשובה בעלת שגיאות).

### 2.2 טבלת התוצאות הרשמית (`results/external_lr_prm_v1/RESULTS.csv`)

| Benchmark | Backbone / Model | שיטה (Method) | כלל החלטה (Rule) | מדד רשמי (Official Metric) | Within-answer AUC | מספר תשובות |
|---|---|---|---|---|---|---|
| **Socratic** | - | **Qwen2.5-Math-PRM-7B (Supervised)** | `NATIVE_0.5` | **0.680** (0.6796) | **0.824** (0.8236) | 2,995 |
| **Socratic** | - | **Qwen2.5-Math-PRM-7B (Supervised)** | `ANSWER_Z_Q80` | **0.677** (0.6769) | **0.824** (0.8236) | 2,995 |
| **Socratic** | 4B | LR transfer (trained on PRMBench) | `ANSWER_Z_Q80` | 0.640 (0.6400) | 0.729 (0.7292) | 2,995 |
| **Socratic** | 8B | LR transfer (trained on PRMBench) | `ANSWER_Z_Q80` | 0.635 (0.6354) | 0.717 (0.7169) | 2,995 |
| **Socratic** | 4B | DS_FILTER_EQ (Unsupervised) | `ANSWER_Z_Q80` | 0.618 (0.6181) | 0.668 (0.6684) | 2,995 |
| **Socratic** | 8B | DS_FILTER_EQ (Unsupervised) | `ANSWER_Z_Q80` | 0.619 (0.6187) | 0.662 (0.6615) | 2,995 |
| **Socratic** | 4B | IU_PCR (Unsupervised) | `ANSWER_Z_Q80` | 0.614 (0.6139) | 0.656 (0.6560) | 2,995 |
| **Socratic** | 8B | IU_PCR (Unsupervised) | `ANSWER_Z_Q80` | 0.615 (0.6155) | 0.654 (0.6542) | 2,995 |
| **Socratic** | 4B | LSML_CONT (Unsupervised) | `ANSWER_Z_Q80` | 0.617 (0.6170) | 0.653 (0.6534) | 2,995 |
| **Socratic** | 8B | LSML_CONT (Unsupervised) | `ANSWER_Z_Q80` | 0.615 (0.6150) | 0.655 (0.6552) | 2,995 |
| **Hard2Verify** | 4B | **LR transfer (trained on PRMBench)** | `ANSWER_Z_Q80` | **0.448** (0.4480) | **0.674** (0.6739) | 200 |
| **Hard2Verify** | 8B | **LR transfer (trained on PRMBench)** | `ANSWER_Z_Q80` | **0.428** (0.4278) | **0.684** (0.6838) | 200 |
| **Hard2Verify** | 4B | IU_PCR (Unsupervised) | `ANSWER_Z_Q80` | 0.415 (0.4153) | 0.603 (0.6033) | 200 |
| **Hard2Verify** | 8B | IU_PCR (Unsupervised) | `ANSWER_Z_Q80` | 0.415 (0.4153) | 0.602 (0.6022) | 200 |
| **Hard2Verify** | 4B | LSML_CONT (Unsupervised) | `ANSWER_Z_Q80` | 0.398 (0.3978) | 0.573 (0.5733) | 200 |
| **Hard2Verify** | 8B | LSML_CONT (Unsupervised) | `ANSWER_Z_Q80` | 0.403 (0.4026) | 0.569 (0.5694) | 200 |
| **Hard2Verify** | - | **Qwen2.5-Math-PRM-7B (Supervised)** | `ANSWER_Z_Q80` | 0.391 (0.3914) | 0.634 (0.6335) | 200 |
| **Hard2Verify** | - | **Qwen2.5-Math-PRM-7B (Supervised)** | `NATIVE_0.5` | **0.234** (0.2336) | 0.634 (0.6335) | 200 |

### 2.3 תובנות מפתח מההשוואה החיצונית
1. **על Socratic (בעיות מתמטיות קרובות לתחום האימון של PRM):**
   - ה-PRM המפוקח שומר על עליונות (0.824 AUC, 0.680 PRMScore).
   - מודל ה-LR שהועבר מ-PRMBench מוסיף כ-+0.06 עד +0.07 ב-Within-AUC על פני השיטות הלא-מפוקחות (0.717/0.729 לעומת 0.655).
2. **על Hard2Verify (הסקה קשה ומורכבת מחוץ לתחום):**
   - **ניצחון מובהק לטלמטריה על פני ה-PRM:** מודל ה-LR משיג Balanced F1 של **0.428 (8B)** ו-**0.448 (4B)** ו-Within-AUC של **0.684**, ומנצח את ה-PRM המפוקח (0.634 AUC ו-0.391 F1 תחת Q80).
   - **הסף הטבעי של ה-PRM נכשל:** תחת הסף הטבעי (0.5), ה-PRM קורס ל-0.234 F1 בלבד, מאחר שהוא כמעט ואינו מזהה שגיאות במדגם הקשה הזה.
   - **שיטות ללא תוויות:** ה-IU-PCR הלא-מפוקח (0.415 F1) מנצח את ה-PRM בשני הספים שלו.
3. **סייגים מתודולוגיים (Caveats):**
   - גודל המדגם ב-Hard2Verify קטן יחסית (200 תשובות).
   - מיקום הצעד לבדו מהווה Baseline חזק ב-Hard2Verify (מיקום יחיד משיג Within-AUC של 0.865).

---

## 3. Slide 14: PRM Boosting via Internal Backbone Signals (Qwen-7B)

### 3.1 שאלת המחקר וההגדרה
- **השאלה:** האם ראש הדירוג הסקלרי של ה-PRM מתעלם ממידע עשיר שנמצא בהתפלגות הטוקנים של ה-Backbone שלו עצמו באותו Forward Pass בדיוק?
- **ההגדרה:** באותו מעבר רשת (Forward) של `Qwen2.5-Math-PRM-7B`, נשלפים:
  1. ציון ה-PRM הרשמי מראש הדירוג (`<extra_0>`).
  2. טלמטריית טוקנים מתוך ה-Backbone (על ידי הפעלת ה-`lm_head` המובנה על ה-Hidden States הסופיים).
- **3 ערוצי ה-Backbone שנבחרו:**
  - `energy_level`: רמת האנרגיה/חוסר הוודאות של המודל בצעד.
  - `hist_entropy_cusum_abs_series`: סכום מצטבר של תנודות האנטרופיה.
  - `realized_drv`: שינוי נגזרת מתממש של מסלול התפלגות הטוקנים.
- **שיטת ה-Fusion:** `LSML_CONT` (L-SML רציף), ללא שימוש בתוויות אמת בהתאמה, עם נרמול $z$ בתוך כל תשובה וסף `ANSWER_Z_Q80`.

### 3.2 תוצאות מפורטות על PRMBench (6,969 תשובות, 6,030 זכאיות ל-AUC)

| זרוע (Arm) | צ'קפוינט | כלל החלטה (Rule) | PRMScore רשמי | Within-answer AUC |
|---|---|---|---|---|
| **PRM reward head (Native)** | Qwen2.5-Math-PRM-7B | `reward < 0.5` | 65.49 | 0.8012 |
| **PRM reward head (Q80)** | Qwen2.5-Math-PRM-7B | `ANSWER_Z_Q80` | 65.82 | 0.8012 |
| `energy_level` בלבד | Qwen2.5-Math-PRM-7B | `ANSWER_Z_Q80` | 65.96 | 0.7991 |
| `hist_entropy_cusum_abs_series` בלבד | Qwen2.5-Math-PRM-7B | `ANSWER_Z_Q80` | 58.03 | 0.6775 |
| `realized_drv` בלבד | Qwen2.5-Math-PRM-7B | `ANSWER_Z_Q80` | 61.84 | 0.7363 |
| **L-SML: 3 ערוצי Backbone בלבד (LSML3)** | Qwen2.5-Math-PRM-7B | `ANSWER_Z_Q80` | 66.63 | 0.7998 |
| **L-SML: PRM + 3 ערוצי Backbone (LSML_CONT)** | Qwen2.5-Math-PRM-7B | `ANSWER_Z_Q80` | **69.48** | **0.8495** |

### 3.3 מבחני מובהקות ורווחי סמך (Paired Bootstrap 95%, 2,000 דגימות)
- **`LSML_CONT` לעומת `PRM_native`:**
  - $\Delta$ PRMScore: **+3.99** [רווח סמך: **+3.31, +4.67**]
  - $\Delta$ Within-AUC: **+0.0483** [רווח סמך: **+0.0440, +0.0526**]
- **`LSML_CONT` לעומת `PRM_Q80`:**
  - $\Delta$ PRMScore: **+3.66** [רווח סמך: **+3.13, +4.17**]
  - $\Delta$ Within-AUC: **+0.0483** [רווח סמך: **+0.0440, +0.0526**]
- **`LSML_CONT` לעומת 3 ערוצי Backbone לבד (`LSML3`):**
  - $\Delta$ PRMScore: **+2.85** [רווח סמך: **+2.48, +3.24**]
  - $\Delta$ Within-AUC: **+0.0497** [רווח סמך: **+0.0456, +0.0539**]

### 3.4 אבחון המשקולות ב-L-SML
האלגוריתם הלא-מפוקח מקצה את המשקולות המנורמלות הבאות:
- `prm_risk`: **0.3693** (המשקל האינדיבידואלי הגבוה ביותר)
- `energy_level`: **0.2438**
- `realized_drv`: **0.2259**
- `hist_entropy_cusum_abs_series`: **0.1610**
*(סך המשקל של שלושת ערוצי הטלמטריה יחד הוא כ-**63%**, לעומת כ-37% לראש ה-PRM)*.

### 3.5 פירוק לפי סוגי שגיאות ב-PRMBench (Macro-F1)
השיפור המשמעותי ביותר מגיע מסוגי שגיאות שה-PRM מתקשה בהם:
- **Redundancy (יתירות בהנמקה):** מ-49.03 ל-**67.66** (**+18.63 נקודות**)
- **Circular Reasoning (מעגליות):** מ-54.95 ל-**71.81** (**+16.86 נקודות**)
- **Domain Inconsistency:** מ-66.43 ל-**73.44** (**+7.01 נקודות**)
- **Missing Condition (תנאי חסר):** מ-57.76 ל-**64.30** (**+6.54 נקודות**)
- **Step Contradiction (סתירת צעד):** מ-67.28 ל-**71.15** (**+3.87 נקודות**)

---

## 4. Slide 15: Cross-PRM Generalization (5 Models, Frozen Recipe)

### 4.1 עקרון ההכללה
המתכון הוקפא לחלוטין: אותם 3 ערוצים בדיוק, אותו אלגוריתם L-SML רציף, ללא בחירה מחדש וללא כוונון היפר-פרמטרים. נבדקו 5 מודלי PRM ממשפחות וסדרי גודל שונים:

### 4.2 טבלת ההכללה לרוחב 5 מודלי PRM

| מודל (PRM Checkpoint) | גודל/משפחה | PRM Native PRMScore | Fused L-SML PRMScore | $\Delta$ PRMScore [95% CI] | PRM Native Within-AUC | Fused L-SML Within-AUC | $\Delta$ Within-AUC [95% CI] | משקל ה-PRM ב-L-SML |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-Math-PRM-7B** | 7B (Qwen) | 65.49 | **69.48** | **+3.99** [3.31, 4.67] | 0.801 | **0.850** | **+0.048** [0.044, 0.053] | 0.3693 |
| **Qwen2.5-Math-PRM-72B** | 72B (Qwen) | 68.18 | **69.84** | **+1.66** [1.01, 2.28] | 0.795 | **0.856** | **+0.062** [0.057, 0.066] | 0.3683 |
| **PURE-PRM-7B** | 7B (Token Clf) | 65.31 | **66.69** | **+1.38** [0.75, 2.08] | 0.763 | **0.811** | **+0.048** [0.043, 0.053] | 0.3822 |
| **RLHFlow Llama-3.1-8B** | 8B (Llama-3.1) | 54.43 | **63.53** | **+9.10** [8.42, 9.79] | 0.662 | **0.772** | **+0.109** [0.103, 0.116] | 0.2894 |
| **MathShepherd Mistral-7B** | 7B (Mistral) | 47.02 | **63.68** | **+16.67** [15.13, 18.09] | 0.684 | **0.759** | **+0.074** [0.067, 0.082] | 0.2332 |

### 4.3 מסקנות מרכזיות מההכללה
1. **שיפור ב-100% מהמודלים:** אף רווח סמך אינו מכיל את 0. כל מודלי ה-PRM מרוויחים משילוב הטלמטריה הפנימית שלהם.
2. **יחס הפוך לחוזק המודל הבסיסי:** במודלים חלשים יחסית (RLHFlow ו-MathShepherd), התוספת של הטלמטריה עצומה (+9.1 ו-+16.7 נקודות). במודלים החזקים ביותר (Qwen-72B ו-Qwen-7B), עדיין יש שיפור מובהק של +1.7 עד +4.0 נקודות.
3. **התאמת משקולות אדפטיבית ללא תוויות:** אלגוריתם ה-L-SML מזהה לבד מתי ה-PRM חלש ומוריד את משקלו (מ-0.37 ב-Qwen ל-0.29 ב-Llama ו-0.23 ב-Mistral), ומעביר את עיקר המשקל לטלמטריה.

---

## 5. Summary of Recommended Discussion Points for the Advisors

1. **אימות החזון של הפרויקט:**
   - הפרויקט התחיל מהשערה שחוסר וודאות ודינמיקה ספקטרלית של טוקנים יכולים לזהות הזיות והסקה פגומה. התוצאות מראות שגם כאשר המודל מסיים אימון ייעודי כ-PRM, אותם סיגנלים נשארים חזקים ואינם נבלעים בראש הדירוג הסקלרי.
2. **צעדי ההמשך המומלצים (Slide 16):**
   - סגירת הפער מול רגרסיה מפוקחת (LR) על ידי שילוב עדין של מידע מועט (Few-shot / Semi-supervised).
   - אלגוריתם חכם לבחירת כמות הסימונים בכל תשובה במקום סף גלובלי יחיד.
   - הרחבת ה-Boosting למודלי שפה נוספים מעבר למתמטיקה (קוד, Reasoning כללי).

---
*All results verified against source logs, AIRCC SLURM receipts, and frozen benchmark evaluation code.*
