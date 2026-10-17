# מקורות האומדים והאלגוריתם: מה אנחנו משתמשים בו, מאיזה מאמר, ומה נוסה

**תאריך:** 2026-09-29. **HISTORY:** Step 463. **ענף:** `claude/estimator-provenance-collection-2026-09-29`.
**מבוסס על:** הקוד והתוצאות של הקו המאוחד `claude/ssl-pseudolabel-residual-v1` עד Step 462 כולל.
**קהל יעד:** Omri, ובהמשך המנחים. כל מספר מצוטט יחד עם הקובץ שממנו נלקח.

## 1. המוטיבציה: מאיפה בא "אומדן DS"?

Omri שאל: "כשאתה משתמש במונח אומדן DS, מאיפה הוא מגיע? איזה מאמר הקוד מממש?"

הבירור העלה ארבעה דברים:

1. **"DS" = Dawid–Skene (1979).** בענפים העדכניים "אומדן DS" הוא EM של Dawid–Skene על marks בינאריים (top-20% בתוך כל תשובה). מקור הקוד: `cvf_v2/em.py`, מהענף `codex/cumulative-vote-fusion-v2`. הקריאה אליו היא דרך `er_stage_a.em_estimate(votes, 'ds')`. ה-EM מאותחל גם מציון ספקטרלי (SML).
2. **הוא לא מממש את Jaffe, Nadler & Kluger 2015** (המאמר שבועז הפנה אליו, `proceedings.mlr.press/v38/jaffe15.pdf`). זה המאמר של אומד ה-**tensor MoM**. עד Step 463 הוא הופיע ב-repo רק כהפניה.
3. **ב-`main` יש קוד אחר לגמרי.** ‏`hallucination_detection/core/lsml.py` הוא שלד מוקדם, ושם "Dawid-Skene style estimation" הוא בפועל הצבעת רוב. הוא לא רלוונטי לקו הנוכחי.
4. **נמצא ייחוס שגוי.** ‏`GLOSSARY.md` (ערך `a1_residual`) ייחס את ה-residual של Eq. 14 למאמר של 2015. אחרי בדיקה מול `papers/extracted/` התברר ש-Eq. 14 (residual) ו-Eq. 15 (score matrix) שייכות למאמר של **2016**, כפי ש-`spectral_utils/fusion_utils.py` כותב. הייחוס תוקן במקור, `spectral_utils/glossary.py`, והקובץ נבנה מחדש.

## 2. טבלת המקורות (מוכנה למנחים)

| שנה | מאמר | שם האומד | כותבים | עיקרון מתמטי | האם מומש בקוד שלנו |
|---|---|---|---|---|---|
| 1979 | *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm*, J. R. Stat. Soc. C 28(1) | **DS** (Dawid–Skene EM) | A. P. Dawid, A. M. Skene | Conditional independence (CI) של המסווגים בהינתן Y. ‏ψ, ‏η ו-prevalence נאמדים ב-EM (maximum likelihood עם Y לטנטי). | **כן.** `cvf_v2/em.py` עם `kind='ds'`, דרך `er_stage_a.em_estimate`. זה "אומדן DS" בכל הקו. |
| 2014 | *Ranking and Combining Multiple Predictors Without Labeled Data*, PNAS | **SML** (Spectral Meta-Learner) | F. Parisi, F. Strino, B. Nadler, Y. Kluger | תחת CI, הקווריאנס מחוץ לאלכסון היא rank-one: `q_ij = (1−b²)(2π_i−1)(2π_j−1)`. ה-leading eigenvector מדרג את המסווגים. | **כן.** `er_stage_a.sml_estimate` ו-`cvf_v2/core.spectral_weights`. את b̂ הוא **לוקח מ-DS-EM**. |
| 2015 | *Estimating the Accuracies of Multiple Classifiers Without Labeled Data*, AISTATS (arXiv:1407.7644) | **Tensor MoM** (השיטה השנייה במאמר, restricted likelihood, לא מומשה) | A. Jaffe, B. Nadler, Y. Kluger | Method of moments: ‏b מהטנזור מסדר שלישי (Lemma 4, Eq. 13–19), ואחריו ψ ו-η בנוסחה סגורה. בלי EM. | **כן, מ-Step 463.** `er_stage_a.tensor_mom_estimate`, בדיוק Algorithm 1 של המאמר. **עוד לא הורץ על PRMBench.** |
| 2016 | *Unsupervised Ensemble Learning with Dependent Classifiers*, AISTATS (arXiv:1510.05830) | **L-SML** (Latent SML). בקוד שלנו גם **HEM** | A. Jaffe, E. Fetaya, B. Nadler, T. Jiang, Y. Kluger | מסווגים תלויים מקובצים סביב משתנים לטנטיים α_k. ‏Score matrix מדטרמיננטות 2×2 (Eq. 15), ו-K לפי residual מינימלי (Eq. 14). | **חלקית.** ‏`discover_groups` ב-`cvf_v2/core.py` עם `fusion_utils._score_matrix_lsml` ו-`_residual_lsml`. ‏HEM ‏(`em.py`, `kind='hem'`) משתמש באותו מודל, אבל מאמד אותו ב-EM מלא. |
| 2026 | In-house, ‏Step 461 | **Position-prior DS** | הפרויקט | DS שבו ה-prevalence תלוי במיקום היחסי של הצעד: `P(Y=1 \| bin b) = π_b`. | **כן.** `scripts/experiments/position_prior_ds.py` (7 בדיקות). |

## 3. המודל המשותף ומשוואות המומנטים

**המודל:** ‏Y ∈ {±1} (‏‎+1 = צעד שגוי), והצבעות f_i ∈ {±1}. הפרמטרים:
```
p   = P(Y=+1)                  prevalence
b   = 2p − 1                   class imbalance
ψ_i = P(f_i=+1 | Y=+1)         sensitivity
η_i = P(f_i=−1 | Y=−1)         specificity
π_i = (ψ_i + η_i)/2            balanced accuracy
δ_i = 2π_i − 1                 "מעל רנדומלי"
```

**המומנטים.** בהינתן `E[f_i | Y] = μ_i + δ_i (Y − b)`:
```
Mean:          μ_i = E[f_i]
Covariance:    q_ij = (1 − b²) δ_i δ_j = t_i t_j,       t_i = √(1−b²) δ_i          (i ≠ j)
Third moment:  T_ijk = −2b(1 − b²) δ_i δ_j δ_k = α t_i t_j t_k,   α = −2b/√(1−b²)   (2015, Eq. 13, 16)
Inverse:       b = −α / √(4 + α²)                                                   (2015, Eq. 17)
Closed form:   ψ_i = (1 + μ_i + δ_i(1−b))/2,    η_i = (1 − μ_i + δ_i(1+b))/2
```

**מה כל אומד מזהה:**

| אומד | משתמש ב- | מזהה | חסר |
|---|---|---|---|
| SML | מומנט שני | ‏t_i, ומכאן הדירוג לפי π_i | את b לא אפשר להפריד מ-δ |
| Tensor MoM (2015) | מומנט שני ושלישי | ‏b, ‏δ_i, ‏ψ_i, ‏η_i | כלום. זה פתרון סגור |
| DS-EM (1979) | ה-likelihood המלא | ‏p, ‏ψ_i, ‏η_i | ‏EM מגיע ל-local optimum |
| HEM / L-SML (2016) | מודל עם קבוצות | גם את התלות בתוך קבוצה | — |

**משקל ה-ML** תחת DS: `w_i = ½ log(ψ_i η_i / ((1−ψ_i)(1−η_i)))`. בקירוב מסדר ראשון `w_i ≈ 2δ_i ∝ t_i`, כלומר משקלות SML הם הקירוב הליניארי של משקלות ה-ML.

## 4. האלגוריתם שלנו, שלב אחר שלב

אלא אם כתוב אחרת, המספרים הם PRMBench within-answer AUC על 6,030 תשובות. זו development evidence, כי הבחירה נעשתה על PRMBench.

### 4.1 הליבה: frozen candidate ‏(Step 457). ממוצע 8 banks: 0.7656 (`results/algorithm_decisions_v1/SUMMARY.md`)

| # | שלב | מה אנחנו משתמשים בו | מקור | משוואה | אלטרנטיבות | נוסתה? מה יצא |
|---|---|---|---|---|---|---|
| 1 | Normalization | ‏per-answer z-score ‏(`answer_standardize`) | In-house | `x̃_si = (x_si − mean_a x_i)/sd_a x_i` | ‏position adjustment ‏P1/P2 | ‏P1: לכל היותר +0.0015. ‏P2 הפסיד בכל ה-banks. **P0** (Step 457) |
| 2 | Binarization | ‏top-20% marks בתוך כל תשובה | In-house | `f_si = +1` אם s נמצא ב-⌈0.2 n_a⌉ העליונים | random-tie; median; continuous | random-tie הרס את ה-partition (amendment B1, Step 451) |
| 3 | אמידת ψ, η, π | DS-EM | Dawid–Skene 1979; אתחול בקו של Parisi 2014 | EM על `Σ_y P(y) Π_i P(f_i\|y)` | SML; HEM; **Tensor MoM (2015)** | ‏Stage A (Step 450): כולם נכשלו, prevalence ≈ 0.28 מול 0.14. ‏MoM: מומש ב-Step 463, **עוד לא הורץ** |
| 4 | Filter | ‏DS filter: ‏π̂_i > 0.5 | הכלל in-house; תחת DS משקל ה-ML חיובי אם ורק אם π > ½ | `keep i ⇔ ψ̂_i + η̂_i > 1` | band rule; correlation filter; בלי filter | ה-band rule לא ניצח (Step 456). ה-correlation filter מפספס ב-banks גדולים (Step 455). ה-filter נותן +0.0053 מול כל ה-13, ורוב הרווח הוא position (Step 451) |
| 5 | Fusion | plain average | Baseline | `score = mean_{i∈S} x̃_i` | SML; DS-ML; L-SML; HEM; קבוצות | הכלל הקפוא בחר ב-plain average (Step 457) |
| 6 | Orientation | הנחת "most beat random", או anchor | Parisi 2014 | `v ← −v` אם `Σv < 0` | flip של ערוצים הפוכים | ה-flip הזיק ב-B32 (−0.0157) וב-B51 (Step 456) |

### 4.2 הווריאנט עם קבוצות (אופציונלי). הכי טוב: 0.7670 (Step 457)

| # | שלב | מה | מקור | משוואה | אלטרנטיבות | נוסתה? |
|---|---|---|---|---|---|---|
| 7 | Partition | binary-mark partition, ‏L-SML Algorithm 1 | Jaffe et al. 2016, Eq. 14–15 | `s_ij = Σ_{k,l≠i,j} \|r_ij r_kl − r_il r_kj\|`; ‏K = argmin residual | continuous; declared; eigengap; random | continuous פחות טוב. ‏declared (Joint) לא התכנס על ה-partition שלו עצמו. טוב יותר מ-2,000 partitions אקראיים. ‏eigengap: לא ראיתי שנבדק |
| 8 | Merge step | ‏absorption ratio ‏ρ < 0.5 | In-house (Step 456) | `ρ = λ₂(R[A∪B]) / min(λ₁(R[A]), λ₁(R[B]))` | ‏Kaiser; מיזוג ידני; בלי מיזוג | מתקן את ה-partition, אבל L-SML עדיין מתחת ל-plain average ב-5/5 banks |
| 9 | Within weights | ‏EQ / SML / HEM | ‏Parisi 2014; המודל של 2016 | `w_i = max(0, log(e₁(1−e₀)/(e₀(1−e₁))))` | — | ‏HEM לא עקבי |
| 10 | Between weights | ‏DSM: ‏DS על marks של הקבוצות, ואז ML weights | ‏DS 1979 | `w_g = max(0, log(ψ_g η_g/((1−ψ_g)(1−η_g))))` | ‏EQ; SML; HEM; ceiling | הכי טוב בין הקבוצות, ושווה ל-ceiling. אבל ה-partition עולה יותר ממה שהמשקלות מחזירים |
| 11 | L-SML מלא | partition, ואחריו SML בתוך ובין הקבוצות | Jaffe et al. 2016 | `score = Σ_g c_g Σ_{i∈g} w_i x̃_i` | — | מתחת ל-plain average ב-8/8 banks (0.7532) |

### 4.3 Position (Steps 460–462, ‏2026-09-29)

| # | שלב | מה | מקור | משוואה | אלטרנטיבות | נוסתה? מה יצא |
|---|---|---|---|---|---|---|
| 12 | Position as channel | ‏POS = z-score של אינדקס הצעד בתוך התשובה. נכנס ל-DS filter ול-average | In-house; הנחה מוצהרת: error propagation | ‏POS = z_a(t) | ‏CUM (running mean) | **אומץ** ב-4/4 banks, הכי טוב 0.8002 על 13+d. ‏CUM: אין רווח. הכיוון לא דורש labels, המשקל כן (`1/(p−1)`) (`results/position_channel_v1/SUMMARY.md`) |
| 13 | Position as prior | ‏**Position-prior DS** | In-house, הרחבה של DS 1979 | `P(Y=1\|bin b)=π_b`; `score = S + logit(π_b)/a` | channel; slope cross-fitted | ה-grouped **אומץ**: 0.8056 על 13+d. מה שהוא מוסיף הוא משקל גדול יותר, לא הצורה, ועדיין קטן פי 2–5 מהאופטימום (`results/position_prior_v1/SUMMARY.md`) |
| 14 | Fit scope | per model per dataset (transductive) | דרישה של Omri | — | pooled | ב-PRMBench זה "חינם". ב-ProcessBench ה-fit מוריד את POS לבד (32/32). ה-prior בכל תא "נתפס" על artefact של הצעד הראשון (17/32) (`results/per_dataset_fit_v1/SUMMARY.md`) |
| 15 | ProcessBench readout | argmax | — | first-error: `P(first=t)=q_t Π_{s<t}(1−q_s)` | first-error readout | ה-first-error readout נמוך ב-10–20 נקודות. נשארים עם argmax |
| — | Cross-fitted slope | ‏slope מחצי אחד של הערוצים | In-house | — | — | 0.8089, השיא ב-PRMBench, **לא אומץ**: מפסיד על 23+d ופוגע ב-ProcessBench |

**הבעיה הפתוחה (Steps 460–462):** אין אומדן label-free ל**כמה** position צריך לספור. בנוסף, ProcessBench (first error) ו-PRMBench (step validity) מושכים לכיוונים הפוכים.

## 5. Tensor MoM: איפה הוא נכנס, ומה לצפות

**ארבעה מקומות אפשריים ב-flow:**
- **שלב 3–4** (π̂ ל-filter).
- **שלב 10** (ψ_g, η_g ל-DSM).
- **b̂ ל-SML,** במקום b̂ = 2p̂−1 מ-DS-EM.
- **אתחול ל-EM,** כולל ה-EM של position-prior DS בשלב 13.

**למה לא לצפות לשיפור ב-AUC:**
1. **ב-filter אין הבדל, מתמטית.** ‏`π̂_i − ½ = δ̂_i/2`, ו-`δ̂_i = t_i/√(1−b̂²)`, ולכן `keep i ⇔ t_i > 0`. ההחלטה לא תלויה ב-b̂, והיא זהה לסימן של SML.
2. **במשקלות יש תקרה נמוכה.** המשקלות **עם דיוקי הקבוצות האמיתיים** (label-using ceiling) נותנים 0.7667, מול 0.7656 ל-candidate ו-0.7670 לאומדנים המוטים של היום (Step 457). אפילו אומד מושלם לא היה משפר. צוואר הבקבוק הוא ה-partition והתלות, לא האמידה.
3. **אותה הנחה.** ‏MoM מניח CI כמו DS. תחת מודל שגוי שניהם מתכנסים ל"ערכים מדומים" שונים, ואין סיבה מראש ש-MoM יהיה קרוב יותר.

**בדיקה סינתטית (Step 463):** נתונים עם תלות בתוך קבוצות, באותו מבנה כמו `test_hem_recovers_marginal_rates_under_group_dependence`, עם prevalence אמיתי 0.15. ‏MoM אומד 0.258, 0.259 ו-0.274 (seeds 3, 11, 12). זה אותו דפוס של הערכת יתר ש-DS מראה על הנתונים האמיתיים. הסימולציה הורצה ב-session עצמו ולא נשמרה כקובץ.

**למה בכל זאת כדאי להריץ:**
- זו תוצאה נקייה ל-Stage A: האם האומד שהמנחה הציע קרוב יותר לאמת?
- אם כן, מריצים אותו גם בשלב 10.
- אם לא, השאלה נסגרת עם נימוק.

**הרצה (מה-worktree המקומי):**
```
python scripts/experiments/tensor_mom_stage_a_run.py
```
הפלט: `results/tensor_mom_v1/run_<date>/` ‏(`SUMMARY.json`, `CHANNELS.csv`). ה-runner עוצר בשני מקרים: אם ה-inputs לא זהים ל-`INPUT_MANIFEST` של Stage A, או אם ה-truth לא משחזר את `STAGE_A_CHANNELS.csv`.

## 6. מה נשאר פתוח
- **להריץ את `tensor_mom_stage_a_run.py`** (נדרשים הנתונים המקומיים).
- **‏MoM בשלב 10 ובאתחול ל-EM:** רק אם Stage A מראה שיפור.
- **השיטה השנייה של המאמר** (restricted likelihood, סעיף 4.2) לא מומשה.
- **בדיקה של eigengap** לבחירת K ב-partition לא נמצאה.
- **ה-CI מופרת גם בין קבוצות:** ‏|r| בין 0.14 ל-0.31, בזוגות הגרועים 0.72–0.86 (`results/lsml_merge_step_v1/SUMMARY.md`, addendum). זה ההסבר העיקרי לכך ש-L-SML לא מנצח, ולכך שאומדנים טובים יותר לא יעזרו לבדם.
