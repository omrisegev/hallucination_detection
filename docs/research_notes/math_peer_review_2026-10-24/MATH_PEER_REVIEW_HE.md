# Peer review מתמטי: PRM boosting (CHAIN / L-SML) — 2026-10-24

**מטרה.** ביקורת קשוחה על הבסיס המתמטי של השיטה, כמו שסוקר ב-NeurIPS/ICML היה כותב.
**בסיס.** קוד ב-`claude/prm-boosting` @ `976845d5`:
`results/prm_boosting_bank_sweep_v1/lsml_bank_sweep.py` (CHAIN), `spectral_utils/fusion_utils.py`
(`sml_fuse_signed`, `detect_dependent_groups`, `lsml_continuous`), `results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py`
(`az`, `flags_for` = Q80), `results/sml_weight_estimator_v1/README.md`, ותוצאות מ-project memory.
**לא חוזר על** ה-positioning review (`reviews/positioning_peer_review_2026-10-24/`): K2, CARE, CROP tie, PRISM.
**לא הורץ שום job.** חישוב אחד בלבד: דוגמת צעצוע Gaussian ב-numpy (סעיף 3).

---

## 0. Verdict של הסוקר

| # | חולשה | חומרה | האם הנתונים שלנו כבר מראים אותה? |
|---|---|---|---|
| W1 | ה-latent ש-SML מוצא אינו בהכרח "שגיאה" (identifiability) | **קריטית** | כן: position-only מנצח ב-Socratic/H2V |
| W2 | conditional independence בין קבוצות נשבר ע"י nuisance משותף (position/length) | **קריטית** | כן: PRM מקבל 0–8% משקל, sign flip ב-MS |
| W3 | אין תנאי שמבטיח שיפור על ה-PRM; אין הסבר מתי זה עוזר | **קריטית** | כן: עוזר ל-PRM חלש, פוגע ב-PRM חזק |
| W4 | continuous views מחוץ ל-Lemma 1; המשקלות אינם Bayes-optimal גם תחת המודל | גבוהה | חלקית |
| W5 | ה-estimator מוטה לכיוון equal weights, וה"הטיה עוזרת" | גבוהה | כן (README section 3) |
| W6 | answer-level z-scoring: זורק מידע, יוצר תלות, מפר iid | גבוהה | כן: PRM prob scale חזק יותר; MS Sensitivity |
| W7 | clustering heuristic בלי consistency/stability; ההנחה "PRM בקבוצה משלו" סותרת את AUTO | גבוהה | כן: AUTO בולע את ה-PRM ב-15/20 תאים |
| W8 | משקל ה-PRM ב-CHAIN תלוי ב-K, שנבחר בשרירות | בינונית-גבוהה | כן: share 0–21% |
| W9 | sign rule + anchor flip מחוץ להנחות המאמר | בינונית (תוקן חלקית) | כן: Socratic MS .468 |
| W10 | כללי ההחלטה (Q80/GAP) לא נגזרים מהמודל; אין posterior/calibration | בינונית-גבוהה | כן: GAP ≈ K2 |
| W11 | ה-bootstrap לא מתאים מחדש את ה-SML → CI צרים מדי | בינונית | מאומת בקוד |
| W12 | fit טרנסדוקטיבי על ה-test set + forking paths על PRMBench | גבוהה | כן: ~208 diffs, תיקון digit אחרי הכישלון |

**שורה תחתונה של הסוקר:** "השיטה היא heuristic שמשתמש ב-SML מחוץ לתנאי התקפות שלו. אין משפט שמסביר מתי היא
עוזרת, וה-failures החיצוניים מתאימים בדיוק למה שהתאוריה הייתה חוזה אם הייתה כתובה. Reject בצורתה הנוכחית;
major revision אם מוסיפים תנאי שיפור + אבחון label-free שחוזה מתי לא לאחד."

---

## 1. מה השיטה מניחה בפועל (formal)

CHAIN = `TELE fit` → group scores → `az` לכל group → concat עם `az(logit(1−reward))` → SML אחד על K+1 views → anchor flip → Q80/GAP.

| הנחה | איפה בקוד | נדרש ע"י | נבדק אצלנו? |
|---|---|---|---|
| A1. latent בינארי יחיד y (step שגוי / תקין) מסביר את כל הקורלציה בין קבוצות | `sml_fuse_signed` cross stage | Parisi 2014 / Jaffe 2016 | לא. position מתחרה בו |
| A2. conditional independence בין קבוצות בהינתן y | groups מ-`detect_dependent_groups` | Jaffe 2016 Lemma 1 | רק עם labels, רק ב-Socratic (`cov_decomp.py`) |
| A3. בתוך קבוצה: תלות דרך latent אחד (block structure) | Eq.14/15 | Jaffe 2016 | לא נבדק stability |
| A4. כל view טוב מאקראי (v_i > 0) | majority sign + anchor | Jaffe 2016 Lemma 4 | הופר ב-Socratic |
| A5. דגימות iid | `np.cov` על כל ה-steps המאוחדים | כל ה-moment estimators | מופר: steps מקובצים ב-answers |
| A6. views בינאריים ±1 | `lsml_continuous` docstring: "No theoretical guarantee" | Lemma 1 | לא |
| A7. PRM בלתי תלוי מכל קבוצות ה-telemetry בהינתן y | `groups=np.r_[cn.max()+1, cn]` | בחירת design | AUTO סותר (15/20) |

---

## 2. החולשות, אחת-אחת

### W1. Identifiability: מי ה-latent?
- **הטענה של הסוקר:** SML מוצא את ה-factor הדומיננטי של ה-off-diagonal covariance. אין שום ערובה שזה correctness.
  כל משתנה נסתר שמשפיע על כל ה-views (position בתשובה, step length, "קושי" של השאלה, סוף התשובה) הוא מועמד שווה ערך.
- **ראיות שלנו:**

| benchmark | position-only within-AUC | telemetry / CHAIN |
|---|---|---|
| PRMBench | 0.662 | 0.71–0.82 |
| Socratic | 0.731 | 0.562–0.712 (position מנצח בכל 5) |
| Hard2Verify | 0.866 | מנצח כל שיטה |

- ב-PRMBench שגיאות פזורות (synthetic perturbation), אז ה-factor הדומיננטי יוצא קרוב ל-y. ב-benchmarks טבעיים, שגיאות
  מרוכזות בסוף, ו-latent "position" ו-latent "error" מתערבבים. **זה ההסבר הסביר ביותר לכך שזה עובד דווקא על PRMBench.**
- **Fix (theory):** לנסח תנאי זיהוי: אם Cov(f|y) = D (diagonal) אז R_off = p(1−p)ΔΔᵀ ו-v ∝ Δ. אם יש nuisance u עם
  Cov(f|y) = D + ccᵀ, אז R_off = rank-2, וה-eigenvector הוא תערובת של Δ ו-c, עם משקל לפי ‖Δ‖²p(1−p) מול ‖c‖².
- **Fix (experiment, CPU):** לכל תא לדווח corr(latent המשוקלל, position) ו-corr(latent, length). אם |corr| > corr עם labels,
  ה-latent הוא position. להוסיף rank-2 model: לחלץ factor שני ולהראות איזה מהם מתאים ל-y.

### W2. Conditional independence בין קבוצות מופר
- **הטענה:** L-SML מטפל רק בתלות *בתוך* קבוצות. nuisance שמשותף ל*כל* הקבוצות (position, length, number formatting)
  הוא בדיוק המקרה שהמודל לא מכסה.
- **ראיות:** Socratic MS: Cov(PRM, TELE) = −.176 = −.186 (within class) + .010 (signal). Within-class corr של PRM עם position +.43,
  של TELE −.53..−.58. (`sml_weight_estimator_v1/README.md` §1).
- **Fix (experiment):** לחשב עם labels (רק כאבחון) את Cov(group scores | y) בכל 4 benchmarks × 5 PRMs, ולדווח את
  ‖R_off|y‖ / ‖R_off‖ ("fraction of off-diagonal covariance that is nuisance"). זה המספר שמסביר הצלחה/כישלון.
- **Fix (method):** להפוך residualization על position+length לחלק מהשיטה (pre-registered), לא רק control.
  לחלופין third-moment test label-free (כבר יש `violation4` ב-`monday_external_panel_v1/score.py`) כדי לזהות הפרה בלי labels.

### W3. אין תנאי שיפור על ה-PRM (השאלה "למה זה אמור לעבוד")
- **הטענה:** אין Proposition שאומר מתי fused AUC > PRM AUC. בלי זה, כל טבלת תוצאות היא אנקדוטה.
- **מה התאוריה הייתה אומרת (Gaussian, class-conditional):** עם separations d_i, משקלות w ו-within-class cov Σ_y:
  AUC(w) = Φ( wᵀd / √(2·wᵀΣ_y w) ). Fusion עוזר רק אם wᵀd/√(wᵀΣ_y w) > d_PRM.
- **מסקנה:** עם PRM חזק (d גדול) ו-telemetry חלש ומתואם (nuisance), כמעט כל משקל שלא קרוב ל-(1,0,…,0) פוגע. זה בדיוק הדפוס:
  RLHFlow/MS (PRM חלש) מרוויחים; Qwen-72B/PURE ב-Socratic/H2V/ProcessBench מפסידים.
- **Fix:** לכתוב את ה-Proposition, להעריך את הכמויות label-free (d_i מה-rank-1 loadings, Σ_y מהשארית), ולבנות **gate**:
  "אחד רק אם התנאי מתקיים". לבדוק על 20 תאים (4 benchmarks × 5 PRMs) אם ה-gate חוזה את הסימן של CHAIN−PRM.
  זה הופך את ה-failures לראיה *בעד* התאוריה.

### W4. Continuous views: המשקלות אינם Bayes-optimal גם תחת המודל
- `lsml_continuous` docstring: "No theoretical guarantee from Parisi-Nadler-Kluger Lemma 1".
- **החשבון:** תחת CI + views z-scored (Var=1): R_off = uuᵀ עם u_i = √(p(1−p))·Δ_i, ולכן σ²_{i|y} = 1 − u_i².
  SML נותן w ∝ u_i. ה-LDA/Bayes weight הוא Δ_i/σ²_{i|y} ∝ **u_i / (1 − u_i²)**.
- **תוצאה:** ה-view החזק ביותר (u גדול, σ_{i|y} קטן) מקבל משקל נמוך מדי באופן שיטתי. זה ה-PRM.
- **Fix (CPU, זול):** משקל מתוקן label-free u_i/(1−u_i²) עם loadings ב-`loading_scale='complete'` (כבר קיים ב-`_rank1_masked`).
  לבדוק על PRMBench + 3 החיצוניים. זה גם נותן לשיטה גזירה ולא רק heuristic.

### W5. ה-estimator מוטה, וה"הטיה עוזרת"
- `sml_fuse` docstring: "the zero-diagonal eigenvector ... shrinks the weights toward equal ... the unbiased fit scored worse".
  rank-1 fit נמוך מ-shipped ב-7/10 banks; oriented mean ≥ L-SML ב-7/10 banks.
- **הסוקר יכתוב:** "השיטה עובדת *כי* היא סוטה מהתאוריה שלה. אם equal weights ≥ SML, מה ה-SML מוסיף?"
- **Fix:** או להצהיר על shrinkage מפורש עם λ שנבחר label-free (למשל James-Stein לכיוון equal, או λ מה-SE של ה-loadings),
  או לדווח ש-"SML ≈ oriented mean + sign" ולמסגר כ-EQ. בכל מקרה: ablation עם λ ∈ [0,1] ולהראות שה-best λ לא נבחר על test.

### W6. Answer-level z-scoring (`az`)
| בעיה | מנגנון | ראיה |
|---|---|---|
| זורק את ה-level של התשובה | כל תשובה מקבלת mean 0, sd 1. מידע "התשובה כולה נכונה" נעלם | Q80 מסמן ~20% גם בתשובות נכונות → Sensitivity −3..−4 (MS 99.9→88) |
| זורק את ה-calibration של ה-PRM | ה-PRM מאומן לתת הסתברות מוחלטת; `az` הופך אותו ל-ranking בלבד | PRM prob scale + Q80 חזק מ-logit (7B 67.32 vs 65.82) — ה-scale משנה, אז `az` לא invariant |
| יוצר תלות בין views | mean התשובה תלוי במספר השגיאות בה → אותו term נכנס לכל view | הפרה ישירה של A2 |
| לא יציב ב-n קטן | n=2: z ∈ {±1} תמיד; n=3: 3 ערכים | לא דווח לפי אורך תשובה |
| שוקל תשובות ארוכות יותר | `np.cov` על steps מאוחדים | לא נבדק |
| בחירת scale (logit/prob) נבחרה עם labels | `STATUS_MAIN.csv` מסמן "scale picked with labels" | — |
- **Fix (theory):** מודל היררכי: x_{ij} = α_j (answer effect) + β·y_{ij} + ε. `az` הוא within-estimator; להראות מתי הוא consistent.
- **Fix (experiment):** global-z מול answer-z מול rank-within-answer; covariance עם משקל שווה לכל answer; תוצאות לפי bins של n steps.
  בנוסף arm שמשאיר את ה-PRM ב-scale מוחלט ומוסיף רק answer-z telemetry.

### W7. Grouping: heuristic בלי ערובות
- Eq.15 score + spectral clustering + K לפי Eq.14 residual עם `loading_scale='unit'`, שה-docstring עצמו אומר
  "does NOT satisfy Lemma 1 ... short by √λ₁". ה-README מציין ש-"the residual criterion itself is a poor guide" (Steps 203/204).
- **סתירה פנימית:** CHAIN מכריח את ה-PRM לקבוצה משלו (A7). אבל כש-AUTO מריץ את אותו אלגוריתם על PRM+telemetry, הוא שם
  את ה-PRM בתוך קבוצת telemetry ב-15/20 תאים. כלומר הנתונים אומרים שה-PRM **לא** CI מה-telemetry.
- **Fix:** (1) partition stability: bootstrap של answers, ARI בין partitions, ודיווח % runs שבהם K זהה. (2) sensitivity של
  CHAIN ל-K (להכריח K=2..8). (3) להצדיק את A7 או להחליף בבדיקה label-free של CI בין PRM לכל קבוצה (third-moment).

### W8. משקל ה-PRM תלוי ב-K
- ב-cross stage יש K+1 views, וה-eigenvector מוטה לכיוון שווה (W5) → share ה-PRM ≈ 1/(K+1) בערך, לא לפי איכות.
  נמדד: CHAIN share 0–21%, AUTO/FLAT 0–6% (`PRM_WEIGHT_SHARE_placement.json`).
- **Fix:** Proposition קצר: תחת shrinkage מלא share = 1/(K+1). ניסוי: K כפוי, גרף PRM share ו-PRMScore מול K.

### W9. Sign rule ו-anchor flip
- `sml_fuse_signed`: majority sign; אחר כך `if corrcoef(s, ref) < 0: s = -s`. ב-pre-oriented views, Lemma 4 אומר שכל v_i > 0,
  אז רכיב שלילי = **הפרת מודל**, לא "view הפוך".
- ראיה: Socratic MS CHAIN .468 (מתחת ל-chance); 4/10 banks עם cross weight שלילי.
- **Fix:** non-negative (abs/nnr1) כ-default pre-registered, ורכיב שלילי → abstain / fallback ל-PRM. זה גם gate טבעי ל-W3.
- **לבדוק:** מאיפה נקבעה ה-orientation של 16 ערוצי N16 (`+1` קבוע). אם היא נקבעה פעם עם labels על PRMBench, זה label use שצריך לדווח.

### W10. כללי ההחלטה לא נגזרים מהמודל
| כלל | מה הוא | בעיה |
|---|---|---|
| Q80 | `z > quantile(z, .8)` על כל ה-steps | 20% קבוע, בלי קשר ל-prevalence; מסמן תשובות נכונות |
| GAP | הפער הגדול ביותר, k ≤ ⌈n/2⌉ | ה-cap נבחר בידיעת מבנה הספירה של PRMBench; בלי cap 57.8 (−7.6); GAP − K2 = −0.35 |
- השיטה מתיימרת להיות מודל הסתברותי (Dawid–Skene) אבל לא מוציאה posterior.
- **Fix (theory):** SML/tensor methods מזהים גם את ה-prevalence p (Jaffe, Nadler, Kluger 2015, third-moment).
  מזה נגזר posterior P(y=1|f) וכלל Bayes (posterior > ½) או expected count Σ posterior לכל תשובה.
- **Fix (experiment):** calibration (ECE, reliability diagram) של ה-posterior; השוואה ל-Q80/GAP/K2.

### W11. Bootstrap לא מתאים מחדש את ה-SML
- ב-`lsml_bank_sweep.py` ה-bootstrap משקלל מחדש source groups על **scores קבועים**; ה-SML, ה-clustering, ה-K וה-sign לא מותאמים מחדש.
- לכן ה-CI מכסה רק evaluation noise, לא fitting variance (שעלולה להיות גדולה: K, partition, sign).
- **Fix (CPU):** bootstrap עם refit מלא (לפחות 200 draws) לניגודי הכותרת. לדווח גם variance של K ושל PRM share.

### W12. Transductive fit + forking paths
- ה-SML מותאם על אותם steps שעליהם מעריכים. label-free, אבל לא inductive. Inductive check קיים רק ב-CROP C4.
- כל הבחירות (bank N16, CHAIN מול PRMG, scale, Q80/GAP, cap, digit fix) נעשו על PRMBench. ~208 diffs/PRM.
  תיקון MS digit נמצא *אחרי* שראינו MS מפסיד (asymmetric bug hunting).
- CIs לא מתוקנים ל-multiplicity.
- **Fix:** split by source group (fit על חצי, apply על חצי) ב-PRMBench; holdout benchmark אחד pre-registered שלא נגעו בו
  (רשימת design frozen לפני הרצה); Holm על משפחת הניגודים.

---

## 3. דוגמת צעצוע: למה ה-PRM מקבל 5% משקל (חישוב numpy, לא נתונים אמיתיים)

Setup: 7 views z-scored, PRM d=1.5, 6 telemetry groups d=0.3 כל אחת, p=0.3.
"nuisance" = within-class corr 0.3 בין קבוצות ה-telemetry (כמו position).

| תרחיש | משקלות | PRM share | AUC |
|---|---|---|---|
| PRM לבד | (1,0,…) | 1.00 | **0.856** |
| CI, Bayes-optimal | w ∝ Σ_y⁻¹d | 0.45 | 0.881 |
| CI, equal weights | 1/7 | 0.14 | 0.811 |
| CI, SML eigenvector | מה-R_off | 0.25 | 0.858 |
| nuisance, Bayes-optimal | Σ_y⁻¹d | — | 0.867 |
| nuisance, equal weights | 1/7 | 0.14 | 0.720 |
| **nuisance, SML eigenvector** | מה-R_off | **0.055** | **0.664** |

- גם כשה-CI מתקיים, SML כמעט לא מוסיף על PRM חזק (0.858 מול 0.856), כי ה-shrinkage משאיר לו 25% במקום 45%.
- עם nuisance משותף, ה-eigenvector נמשך ל-telemetry, ה-PRM יורד ל-5.5% (כמו ה-0–8% שמדדנו), וה-AUC קורס.
- זה משחזר איכותית את Socratic/H2V/ProcessBench. **מומלץ להכניס את זה כ-Proposition + figure.**

---

## 4. מה אפשר לטעון היום ברמת ריגור

| טענה | מותר? | הסתייגות חובה |
|---|---|---|
| "label-free fusion משפר within-answer ranking על PRMBench, גם אחרי position/length residualization" | כן | dev data; transductive; CI בלי refit |
| "SML מזהה אוטומטית את משקלות ה-views" | **לא** | משקלות ≈ equal (W5), PRM share תלוי K (W8) |
| "מבוסס על Jaffe/Parisi עם הנחת CI" | רק עם disclaimer | views רציפים, CI מופר, sign מחוץ להנחות |
| "עובד על כל PRM / benchmark" | **לא** | Socratic, H2V, ProcessBench נכשלו |
| "יש הסבר מתי זה עוזר" | עדיין לא | צריך W3 |

---

## 5. תוכנית תיקון מתועדפת (הכל CPU, שום דבר לא הורץ)

| עדיפות | מה | סוג | עלות | פותר |
|---|---|---|---|---|
| 1 | Proposition תנאי שיפור (Gaussian) + gate label-free; בדיקה על 20 תאים | theory + CPU | 1–2 ימים | W3, W8, W1 |
| 2 | משקל מתוקן u/(1−u²) עם `loading_scale='complete'` | CPU | חצי יום | W4, W5 |
| 3 | אבחון nuisance: corr(latent, position/length); fraction of off-diag cov within class | CPU, labels לאבחון בלבד | חצי יום | W1, W2 |
| 4 | Bootstrap עם refit + partition stability (ARI, K variance) | CPU | יום | W7, W11 |
| 5 | Posterior מ-tensor prevalence + calibration, מול Q80/GAP/K2 | theory + CPU | 1–2 ימים | W10 |
| 6 | Split-half inductive fit על PRMBench; holdout benchmark pre-registered | CPU | יום | W12 |
| 7 | global-z / answer-z / rank; by-length bins; PRM ב-scale מוחלט | CPU | חצי יום | W6 |
| 8 | non-negative + abstain-on-negative כ-default; לתעד orientation provenance של N16 | CPU + doc | שעות | W9 |

## 6. מקורות בקוד
- `results/prm_boosting_bank_sweep_v1/lsml_bank_sweep.py` (CHAIN, anchor flip, bootstrap בלי refit)
- `spectral_utils/fusion_utils.py`: `sml_fuse` (bias docstring), `sml_fuse_signed` (sign rule), `_rank1_masked` ('unit' scale), `detect_dependent_groups`, `lsml_continuous` ("No theoretical guarantee")
- `results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py`: `az`, `flags_for` (Q80)
- `results/sml_weight_estimator_v1/README.md` (§1 Socratic flip, §3 estimator, §4 PRM share)
- `results/error_count_label_free_v1/README.md` (GAP cap, GAP vs K2)
