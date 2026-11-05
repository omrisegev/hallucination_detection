# שלושת ה-features: הגדרה, PRMScore ותלות שגיאות

סדר ה-cells בכל הטבלאות: Qwen-7B / Qwen-72B / PURE / RLHFlow, קודם PRMBench ואחר כך Socratic. הבחירה של שלושת ה-features נעשתה לפי labels על נתונים שכבר נראו (development-level).

## 1. מה הם ואיך הם מחושבים

### Common definitions

```
Token series x_t, t = 1..T (after mask1: forced token 0 dropped).
Causal window ending at t:   w_t = (x_{t-W+1}, ..., x_t)
  (t < W: reuse the first full window; answer shorter than W: the whole answer)
Mean-centre:                 y = w_t - mean(w_t)
Periodogram, AC bins only:   P_k = |rfft(y)_k|^2 + 1e-12,   k = 1..W/2,   f_k = k/W
Normalised power:            p_k = P_k / sum_j P_j,         K = W/2 bins

Spectral flatness:  SF_t = exp( (1/K) sum_k log P_k ) / ( (1/K) sum_k P_k )     in (0, 1]
Spectral entropy:   SE_t = - sum_k p_k log p_k / log K                          in [0, 1]

Step value (step = tokens a..b-1):   v_step = stat_{t = b-1}      (value at the step's LAST token)
Risk orientation:   risk = - v_step          (low flatness / low spectral entropy = more suspect)
Before fusion:      answer-z of risk         (per answer: (r - mean) / std)
```

### Source series

```
JS_t  = JSD( p_{t-1} , p_t ),  p = renormalised top-50 distribution of token t,
        aligned by token id (ids missing on one side get probability 0), natural log
s_t   = - log p(chosen token_t)          (= token_spilled_energies, chosen-token surprisal)
```

### Three features

```
F1  JS_flat16  = - SF( JS series,        W = 16, K = 8 bins ) at the step's last token
F2  JS_sent32  = - SE( JS series,        W = 32, K = 16 bins) at the step's last token
F3  SUR_sent8  = - SE( surprisal series, W = 8,  K = 4 bins ) at the step's last token
```

**F1 ו-F2 (מבוססי JS):** ‏JS_t מודד בכמה ההתפלגות של המודל על ה-token הבא השתנתה בין שני tokens סמוכים. ה-feature לא מסתכל על גודל השינוי, אלא על הצורה של רצף השינויים ב-16 או 32 ה-tokens האחרונים.
- flatness או spectral entropy גבוהים: השינויים נראים כמו רעש לבן, כלומר תנודות אקראיות ללא מבנה.
- ערכים נמוכים: ההספק מרוכז בתדרים מעטים, בדרך כלל נמוכים. זה אומר שההתפלגות זזה במגמה, או בהתפרצות אחת גדולה.
- הנמוך הוא המסוכן (sign −1, יציב בכל 8 ה-cells). הפרשנות "מגמה או התפרצות" היא היסק מההגדרה, לא נמדדה ישירות.

**F3 (מבוסס surprisal):** אותו רעיון על ההפתעה של ה-token שנבחר, בחלון קצר מאוד של 8 tokens (4 bins). ערך נמוך פירושו שה-surprisal בחלון נשלט על ידי מבנה אחד, למשל קפיצה בודדת או מגמה, ולא על ידי תנודה אקראית.

כל שלושת ה-features רציפים: modal share <= 0.3%, חוץ מ-F3 ב-Socratic/PURE שם הוא 3.2%. יש 21k עד 71k ערכים שונים, לעומת כ-200 ב-dominant_freq16. אף אחד מהם לא מחושב מה-entropy.

## 2. PRMScore (Q80 rule, כמו ב-pipeline)

**Standalone:** ה-feature לבדו, answer-z ואז סימון 20% הצעדים העליונים.

| arm | PRMBench 7B | 72B | PURE | RLHFlow | Socratic 7B | 72B | PURE | RLHFlow |
|---|---|---|---|---|---|---|---|---|
| PRM alone | 65.82 | 66.25 | 63.28 | 56.48 | 67.36 | 66.86 | 64.07 | 49.12 |
| dominant_freq16 alone | 53.07 | 53.80 | 52.84 | 53.27 | 50.08 | 49.89 | 49.62 | 50.02 |
| F1 JS_flat16 alone | 52.39 | 52.08 | 51.57 | 52.07 | 53.37 | 51.51 | 51.87 | 51.41 |
| F2 JS_sent32 alone | 53.61 | 54.37 | 52.81 | 53.05 | 56.46 | 53.30 | 54.19 | 51.31 |
| F3 SUR_sent8 alone | 53.58 | 52.56 | 51.93 | 51.54 | 55.93 | 53.22 | 53.25 | 50.87 |

**בתוך ה-Z17 L-SML הנוכחי:** ‏[z(PRM), 16 ערוצים], automatic grouping, anchor z(PRM), ו-feature מחליף את dominant_freq16.

| arm | PRMBench 7B | 72B | PURE | RLHFlow | Socratic 7B | 72B | PURE | RLHFlow |
|---|---|---|---|---|---|---|---|---|
| Z17 current | 67.49 | 67.84 | 66.68 | 63.97 | 64.28 | 59.57 | 63.39 | 57.56 |
| Z17 minus dominant_freq16 | 67.83 | 68.11 | 66.10 | 63.88 | 64.47 | 59.57 | 63.50 | 57.47 |
| swap F1 JS_flat16 | 67.52 | 68.21 | 66.04 | 63.95 | 64.82 | 59.35 | 63.54 | 57.35 |
| swap F2 JS_sent32 | 67.56 | 68.17 | 65.94 | 63.96 | **65.20** | 59.67 | 63.59 | 57.07 |
| swap F3 SUR_sent8 | 67.54 | **69.06** | 66.32 | 63.92 | 64.63 | 59.60 | 63.52 | 57.69 |

**Equal-weight average** של 16 ערוצי answer-z, בלי PRM ובלי weights מותאמים.

| arm | PRMBench 7B | 72B | PURE | RLHFlow | Socratic 7B | 72B | PURE | RLHFlow |
|---|---|---|---|---|---|---|---|---|
| EQ16 current | 66.44 | 66.18 | 65.82 | 63.71 | 62.38 | 59.63 | 63.63 | 57.93 |
| swap F1 | 66.65 | 66.29 | 65.80 | 63.72 | 62.84 | 60.31 | 63.90 | 58.26 |
| swap F2 | 66.62 | 66.43 | 65.66 | 63.72 | 63.04 | 60.32 | 64.05 | 58.50 |
| swap F3 | 66.62 | 66.34 | 65.80 | 63.67 | 63.05 | 60.44 | 63.76 | 58.36 |

**מה רואים:**
- **לבד:** כל feature חלש (51-56), כמו dominant_freq16 ב-PRMBench. ב-Socratic, F2 ו-F3 עדיפים עליו (53-56 לעומת כ-50 שהוא chance).
- **בתוך ה-equal average:** כל swap משפר ב-Socratic בכל 4 ה-PRMs (+0.3 עד +0.8), וב-PRMBench השינוי קטן (−0.16 עד +0.25).
- **בתוך Z17 L-SML:** אין שיפור עקבי. הטווח הוא −0.74 עד +1.22. השיפור הבולט היחיד הוא F3 ב-PRMBench/72B (+1.22), וב-PURE כל ה-swaps מורידים (−0.36 עד −0.74).
- **Category breakdown:** ב-Z17 swap F3 ב-PRMBench, השינוי לכל קטגוריה הוא בתוך ±1.3, ועקבי רק ב-72B. הפירוט ב-`PRMSCORE_DEPENDENCE.json`.

**Parity ואזהרות:**
- Z17 current ב-PRMBench משחזר את ה-Z17 השמור (67.49/67.83/66.67/63.95) בהפרש של עד 0.02. ההפרש נובע מכך שצעדים ריקים (15-105 צעדים) לא מסומנים אצלי.
- את Z17 ב-Socratic בניתי בעצמי מ-REWARDS ו-FEATURES, ואין לו reference שמור. לכן ההפרשים ב-Socratic משווים רק בתוך אותה בנייה.
- PRMScore ב-Socratic חושב כ-50·(F1+negF1) על הצעדים שב-include. אין שם class בשם correct.
- **לא שוחזר:** תוצאת ה-Mac שהסרת dominant_freq16 עולה −4.89 ב-Socratic. אצלי ב-Z17 ההסרה משנה +0.19/0/+0.11/−0.09. כנראה מתכון ה-Mac (branch `claude/rigorous-recipe-2026-11-04`, שלא נדחף) שונה מ-Z17. זה היסק, לא נמדד.

## 3. האם השגיאות שלהם בלתי תלויות בשגיאות TELE16 (בהינתן ה-label)?

```
Class-conditional dependence (answer-z values, included steps, one cell at a time):
  rho_err(F, C)   = Spearman( z_F , z_C  |  error steps only )
  rho_valid(F, C) = Spearman( z_F , z_C  |  valid steps only )
Decision errors under each feature's own Q80 flags:
  miss_lift = P(F misses AND C misses | error step) / ( P(F misses) * P(C misses) )
  fa_lift   = P(F false-alarms AND C false-alarms | valid step) / ( P(F fa) * P(C fa) )
  lift = 1  <=>  independent errors;   lift > 1  <=>  errors co-occur
```

**סיכום לכל cell:** הערך העליון הוא max|rho| מול 16 הערוצים בפורמט error/valid. השורה התחתונה היא מספר הערוצים עם |rho| < 0.1 בשני ה-classes.

| feature | | PRMBench 7B | 72B | PURE | RLHFlow | Socratic 7B | 72B | PURE | RLHFlow |
|---|---|---|---|---|---|---|---|---|---|
| F1 JS_flat16 | max\|rho\| | .19/.19 | .13/.12 | .18/.15 | .19/.17 | .21/.23 | .10/.13 | .17/.13 | .16/.13 |
| | #ch < .1 | 8 | 11 | 6 | 7 | 6 | 11 | 9 | 9 |
| | max miss_lift | 1.05 | 1.03 | 1.04 | 1.05 | 1.04 | 1.02 | 1.03 | 1.02 |
| F2 JS_sent32 | max\|rho\| | .26/.23 | .22/.19 | .25/.21 | .26/.21 | .29/.31 | .13/.19 | .26/.21 | .20/.22 |
| | #ch < .1 | 6 | 8 | 6 | 6 | 4 | 9 | 6 | 7 |
| | max miss_lift | 1.09 | 1.08 | 1.07 | 1.07 | 1.07 | 1.03 | 1.07 | 1.04 |
| F3 SUR_sent8 | max\|rho\| | .25/.23 | .16/.18 | .22/.26 | .20/.18 | .26/.25 | .11/.18 | .26/.26 | .27/.25 |
| | #ch < .1 | 6 | 9 | 5 | 8 | 6 | 11 | 6 | 8 |
| | max miss_lift | 1.08 | 1.05 | 1.06 | 1.05 | 1.08 | 1.02 | 1.07 | 1.04 |

**פירוט לפי ערוץ ל-F1** (ממוצע על 8 ה-cells):

| TELE16 channel | rho_err | rho_valid | miss_lift | fa_lift |
|---|---|---|---|---|
| true_tail50 | +.16 | +.15 | 1.03 | 1.26 |
| q15_H1 | +.15 | +.12 | 1.03 | 1.23 |
| chosen_surprisal | +.13 | +.10 | 1.03 | 1.22 |
| energy_innovation | −.13 | −.13 | 0.99 | 0.88 |
| top50_js | −.11 | −.13 | 0.99 | 0.87 |
| q15_VE1, logprob_margin, energy_level | .05-.09 | .05-.08 | 1.02 | 1.09-1.16 |
| realized_z, realized_drv | .06-.07 | .01-.04 | 1.01 | 1.06-1.10 |
| dominant_freq16, bocpd_p0, top15_turnover, 3 digit | ≤ .04 | ≤ .08 | 1.00-1.01 | 0.91-1.12 |

ב-F2 וב-F3 הדפוס זהה אבל חזק יותר: מול q15_H1 ו-true_tail50, ‏|rho| הוא כ-.20-.23 ו-fa_lift כ-1.5.

**תשובה:** כן, ברובם.
- **תלות מותנית:** בהינתן ה-label, ה-features כמעט בלתי תלויים ברוב TELE16. ‏max|rho| המותנה הוא .10-.31, ו-4-11 מתוך 16 ערוצים נמצאים מתחת ל-.1 בשני ה-classes.
- **החטאות:** ההחטאות על צעדי שגיאה כמעט בלתי תלויות. ‏miss_lift הוא 1.02-1.09, ולכן ה-features מחטיאים צעדים שונים מאלה שהערוצים האחרים מחטיאים.
- **false alarms:** כאן יש תלות ממשית בקבוצה מסוימת. ‏fa_lift מגיע ל-1.2-1.5 מול q15_H1, ‏true_tail50 ו-chosen_surprisal: על צעדים תקינים, F מתריע יחד עם ערוצי ה-entropy וה-confidence.
- **מול dominant_freq16 עצמו:** כמעט אפס תלות (|rho| ≤ .08). לכן ה-features הם לא גרסה רציפה של אותו מידע, אלא מידע אחר.
- **הסתייגות:** חוסר מתאם (Spearman ו-lift) לא מוכיח אי-תלות מלאה.

## קבצים
- AIRCC `results/domfreq_variants_v1/`: ‏`code/prmscore_dep.py` ו-`code/run_prmscore.sbatch` (job 311889). הפלט ב-`PRMSCORE_DEPENDENCE.json`, כולל PRMScore לכל קטגוריה ותלות לכל ערוץ ולכל cell.
