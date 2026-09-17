# כרטיס שיטה: Joint L-SML ללוקליזציה, כפי שרץ בפועל (Step 399, 2026-09-17)

כל פרט כאן אומת מול הקוד שהריץ את הזרוע `joint_hier_sml` ב-`results/joint_group_readouts_v1`:
`scripts/run_joint_redundancy_robustness_v1.py`, `spectral_utils/joint_lsml.py`,
`spectral_utils/joint_pair_extension.py`, `spectral_utils/joint_group_readouts.py`,
`spectral_utils/fusion_utils.py` (`lsml_continuous`, `sml_fuse_signed`, `detect_dependent_groups`),
`spectral_utils/lsml_gate_locator_research.py` (`answer_standardize`, `_orient`),
`scripts/run_lsml_gate_locator_research_v1.py` (`score_locator`), `spectral_utils/error_dependence.py` (gate).
מספרים: development בלבד, 13,769 תשובות, 5 source folds. ה-rosters כוללים digit, שנפסל ב-2026-09-17; המכניקה אינה תלויה בו.

---

## 0. תשובות קצרות לארבע השאלות

**1. האם בזמן ריצה משתמשים רק בתשובה אחת?** לא, לא בגרסה שרצה. הפיצ'רים והנרמול הם מתשובה אחת בלבד, אבל **המשקלים** (גילוי קבוצות, פיט Joint, readout) מותאמים על כל תשובות האימון של 4 מתוך 5 source folds יחד (מטריצה מאוחדת של ~116K צעדים), ואז מיושמים על תשובות ה-fold החמישי. זה מה שבמסמכי הפרויקט נקרא **hybrid**. הזרוע answer-only (פיט מהתשובה עצמה) קיימת במסלול ה-110 (`answer_localization_v2.py`), לא בניסוי הזה.

**2. האם שלב הקיבוץ שונה מזה של L-SML?** כן, בשלושה דברים: (א) מטריצת הדמיון: L-SML משתמש בציון Eq.15 של המאמר, s_ij = Σ|r_ij·r_kl − r_il·r_kj|; Joint משתמש ב-|C − v·vᵀ|, השארית אחרי הסרת גורם משותף rank-1. (ב) בחירת K: L-SML ממזער את שארית Eq.14 על K ∈ {2..8}; Joint בוחר K ∈ {3,4} לפי **יציבות** (ARI חציוני בין leave-one-source-out ל-consensus, בשוויון K קטן). (ג) גודל מינימלי 2 לקבוצה (מסלול הזוגות). התוצאה בפועל: Joint בחר K=3 ב-19 מתוך 20 (roster, fold), L-SML בחר 3/4/5/7 על L08/L11/L14x/L24.

**3. מה עושים בשלב 2 (הפיט)?** מתאימים לקווריאנס המאוחד את המודל C ≈ v·vᵀ + blockdiag(u·uᵀ) + D: v = כמה כל זרם נע עם הסיגנל המשותף לכולם; u = תנועה משותפת בתוך הקבוצה בלבד; D = אלכסון. **בזרוע המובילה, v ו-u אינם משמשים לחישוב המשקלים.** הפיט משמש רק כאודיט מבני (התכנסות, זיהוּת, תקינות זוגות), ונרשם בלבד; הוא לא חוסם את הזרוע. זו נקודה שצריך לומר במפורש: ההבדל של השיטה המובילה מ-Continuous L-SML הוא **בשלב הקיבוץ**, לא במודל הפקטורים.

**4. שלב 3, למה זה לא פשוט L-SML?** מבחינת ה-readout זה כמעט אותו דבר: SML בתוך כל קבוצה, ואז שילוב בין הקבוצות. ההבדל היחיד ב-readout: ב-K=3 השלב בין הקבוצות הוא לא eigen-solve אלא הכלל הרשום מ-Step 205 (משקל שווה לכל קבוצה אחרי סטנדרטיזציה של המסווג הווירטואלי שלה), וזה חל גם על Continuous L-SML כשהוא בוחר K=3. כלומר: **השיטה המובילה = הקיבוץ של Joint + ה-readout הדו-שלבי של L-SML**. ההישג (יציבות מ-8 ל-24 זרמים) נובע מכך ש-Joint בוחר K קטן ויציב ומבודד את המשפחה המשלימה, בעוד L-SML מפצל משפחות יתירות לקבוצות רבות ומדלל אותה.

**5. שלב 4 = אפליקציה.** נכון: argmax, gate, ומדדים הם השימוש ללוקליזציה, לא חלק מאלגוריתם ה-fusion.

---

## חלק 1: הפיצ'רים

### 1.1 מקור
- **דאטה:** ProcessBench (4 תת-מערכים: gsm8k, math, olympiadbench, omnimath) × 2 מודלים (Qwen3-4B, Qwen3-8B) = 8 תאי PB, ו-PRMBench (Qwen2.5-Math-7B, וטלמטריית Qwen3-8B). סה"כ 13,769 תשובות, 145,597 צעדים, 6,968,779 טוקנים. חלוקה ל-5 folds לפי **source group** (שאלת מקור), `FOLDS_V2`.
- **מעבר מודל אחד:** teacher forcing. הפתרון הנתון (של ה-benchmark) מוזן למודל הניקוד, ובכל מיקום נשמרות ההסתברויות ל-15 הטוקנים המובילים (`logprobs15`, [6,968,779 × 15]) וההסתברות של הטוקן שסופק. אין גינרוט, אין labels.
- **גבולות צעד:** spans של ה-benchmark ([145,597 × 2]).

### 1.2 זרמים ברמת הטוקן (מה שנכנס ל-rosters שנבדקו)
| שם בקוד | הגדרה (מאומתת מהרישום `fusion_signal_registry.py`) |
|---|---|
| `q15.VE0.75.prefix_mean_innovation` | varentropy של התפלגות escort בסדר 0.75 על top-15, פחות ממוצע כל הטוקנים הקודמים בתשובה (רקע סיבתי; טוקן 0 לא פעיל) |
| `q15.H0lim.prefix_mean_innovation` | אנטרופיית Renyi בגבול סדר 0 על top-15, פחות ממוצע העבר |
| `renyi_escort.a0.25`, `a8` | אנטרופיית Renyi סדר 0.25 / 8 על top-15 (level, בלי innovation) |
| `direct_probability.rank_k_risk` | סיכון מדרגת הסתברות k: 1−p₁ עבור k=1, p_k עבור k≥2 |
| `step395.logtail15` | log של מסת ההסתברות מחוץ ל-top-15 |
| `step395.mass_above` | מסת ההסתברות של הטוקנים המדורגים מעל הטוקן שסופק |
| `digit.disagreement` | אירוע: הטוקן שסופק הוא ספרה והטוקן המועדף על המודל הוא ספרה אחרת. **נפסל ב-2026-09-17** |
| `digit.token_clock_innovation` | אותו אירוע פחות ממוצע העבר. **נפסל** |

### 1.3 עיבוד לפני המודל
1. **טוקן → צעד (`::top10` / `::top8`):** ערך הצעד = ממוצע k הערכים הגדולים ביותר של הזרם בתוך הצעד (`np.partition(values, n−k)[−k:].mean()`, k=min(10,n)).
2. **סטנדרטיזציה בתוך התשובה** (`answer_standardize`): לכל זרם, z-score על צעדי אותה תשובה בלבד; זרם קבוע בתשובה → 0. זה נעשה **לפני** חלוקת folds, ולכן אינו מדליף בין תשובות.
3. **Rosters שנבדקו:** L08 (8 זרמים: digit top2, digit token-clock top1, VE0.75 innov top10, Renyi a0.25 top10, rank-1 top10, rank-9 top8, logtail15 top10, mass_above top10); L11 = L08 + digit top1, Renyi a8, logtail50; L14x = L11 + H0lim innov, rank-3, rank-10 top8; L24 = כל 24 זרמי הצעד הכשירים באטלס. L08 נבחר ב-Step 397 ב-nested selection **עם labels** (בתוך folds); L24 הוא "הכול", ללא בחירה.

### 1.4 מה נלמד מאיפה
| רכיב | מקור הנתונים |
|---|---|
| פיצ'רים, Top-k, z-score | התשובה עצמה בלבד |
| קבוצות, פיט Joint, כיוונים תוך-קבוצתיים, משקלי בין-קבוצות, סימן | **כל תשובות האימון** (4 folds), מטריצה מאוחדת של צעדים × זרמים |
| ניקוד | התשובה עצמה: z · w |
| labels | לא נכנסים לשום שלב של ה-fusion. משמשים רק להערכה (ולבחירת L08 ב-Step 397) |

---

## חלק 2: אלגוריתם ה-fusion

### 2.1 הנחות
- יש סיגנל משותף אחד (סיכון לשגיאה) שכל הזרמים נעים איתו במידה שונה, ובנוסף **יתירות**: משפחות של זרמים שנעות יחד גם בלי קשר לסיגנל (וריאנטים של אותה כמות).
- הזרמים בתוך משפחה תלויים; בין משפחות הם תלויים רק דרך הסיגנל (מודל L-SML). Joint מוסיף לזה מודל פקטורים מפורש.
- אין labels. הסימן הכולל של הציון נקבע לפי עוגן (סעיף 2.5).

### 2.2 מה זה משפר ביחס ל-IU-PCR ול-L-SML
- **IU-PCR** מניח שגיאות בלתי תלויות בין זרמים ופותר w ∝ Σ⁻¹ρ במרחב שני רכיבים ראשיים. עם יתירות חזקה הוא קורס (26.07% על L24, מול 43.22 לשיטה המובילה), ומדכא זרם משלים שאינו מתואם עם הרוב (חלק digit .04-.18).
- **Continuous L-SML** מטפל ביתירות בקבוצות, אבל בוחר K לפי שארית, ולכן ככל שמוסיפים וריאנטים הוא מפצל לעוד קבוצות (K=7 על L24) וכל קבוצה יתירה מקבלת קול; המשפחה המשלימה (digit) מדוללת (חלק .02) והביצוע יורד מ-43.74 ל-38.91.
- **השיטה המובילה** בוחרת את ה-K הקטן היציב (3), כך שהמשפחה המשלימה נשארת קבוצה אחת נקייה מול שתי קבוצות גדולות של יתירות, ונותנת לכל קבוצה קול: 43.48 / 42.37 / 43.02 / 43.22 על 8/11/14/24 זרמים.
- מה שהיא **לא** משפרת: על ה-roster הקטן והנקי (L08) היא שקולה ל-L-SML (−0.26pp, CI כולל 0).

### 2.3 פסאודו-קוד (בדיוק מה שרץ)
```
input  X_train : [N_train_steps × m]  z-scored within answer, pooled over training answers
       owner   : source-group id of every training step

# --- Stage A: grouping (Joint's rule)                    joint_lsml.discover_loao_consensus_groups
for K in {3, 4}:
    for each source group s (leave-one-source-out):
        C_s      = cov(X_train without s)                                 # from sufficient statistics
        v_s      = rank-1 fit to the off-diagonal of C_s (masked, 'complete' scale)   # fusion_utils._rank1_masked
        A_s      = |C_s − v_s v_sᵀ|, diagonal zeroed                          # residual affinity
        labels_s = spectral_clustering(A_s, K)
    consensus_K = spectral_clustering( mean_s coassignment(labels_s), K )
    admissible  = every group ≥ 2 members in consensus and in ≥95% of the held-out partitions
    stability_K = median_s ARI(labels_s, consensus_K)
labels = consensus of the admissible K with the highest stability (ties: mean ARI, min ARI, smaller K)

# --- Stage B: Joint factor fit (audit only in the leading arm)   joint_pair_jacobian.fit_joint_pairs_checked
C = cov(X_train)
fit  C_offdiag ≈ v vᵀ + M ⊙ (u uᵀ)     M = same-group mask, 5 random starts, coordinate descent + NNLS rescale
pairs (groups of size 2): only the product u_i·u_j is identified; parameterised as one product
audit: ≥4 converged starts agree (model 1e-5, cos(v) ≥ .999); profiled Jacobian full rank, cond ≤ 1e8
# v, u are NOT used below in the leading arm (they are used in the 'joint_global_v', 'joint_inverse', 'joint_hier(v)' arms)

# --- Stage C: two-stage readout                             joint_group_readouts.hierarchical_group_readout('within_sml')
for each group g:
    R_g  = cov(X_train[:, g]) with zero diagonal
    e_g  = leading eigenvector of R_g, sign so that most entries are positive     # fusion_utils.sml_fuse_signed
           (group of size 1: e_g = 1; size 2: entries ±0.707)
    z_g  = X_train[:, g] · e_g                                                    # group virtual classifier
cross = SML over (z_1 … z_K):
    if K == 3: c_g = (1/3) / std(z_g)          # registered small-m rule (Step 205): equal weight per standardised group
    else:      c   = leading eigenvector of cov(z) with zero diagonal, majority-positive sign
w[g] = e_g · c_g   for every g

# --- Stage D: orientation and scale                          lsml_gate_locator_research._orient
if Spearman(X_train · w, X_train[:, 0]) < 0: w = −w        # anchor = first roster column (a digit stream here)
w = w / Σ|w|
return w
```
**ההבדל מהקוד של Continuous L-SML (`lsml_continuous`) שרץ כזרוע ייחוס:** Stage A מוחלף ב-`detect_dependent_groups` (Eq.15 + Eq.14, K ∈ {2..8}); Stage C זהה, למעט שב-Continuous הכלל של K=3 חל גם **בתוך** קבוצה בגודל 3, ובזרוע המובילה בתוך קבוצה בגודל 3 רץ eigen-solve.

### 2.4 שתי החלופות שנבדקו ל-Stage C ונדחו
- `joint_hier` (המקורי, `hierarchical_joint_weights`): z_g = X_g · v_g. זרם זר עם v גדול בקבוצת digit משתלט עליה: חלק digit .12-.13 על L11/L14x, ביצוע 39.7/40.4.
- `joint_hier_u`: z_g = X_g · u_g. עבור משפחות יתירות u תופס תנועה טפלה; within-AUC יורד בכל roster רחב.

### 2.5 תלויות ב-digit שיש להסיר בגרסה הבאה
- העוגן לסימן (Stage D) הוא העמודה הראשונה ב-roster, שהיא זרם digit בכל ארבעת ה-rosters. זה משפיע רק על סימן כולל.
- ה-rosters עצמם מכילים 2-3 זרמי digit.
- ה-gate (חלק 3) משתמש ב-digit_rate.

---

## חלק 3: לוקליזציה יישומית

```
for each test answer a (fold held out from all fitting):
    Z_a      = z-scored step matrix of a                        [n_steps × m]
    score_a  = Z_a · w                                          # higher = more risk
    peak_a   = argmax(score_a)                                  # the predicted first-error step
    if gate(a) is open:  predict peak_a   else: predict "no error" (−1)
```
- **Gate** (קפוא, Step 396, `error_dependence._incumbent_gate`): לכל תשובת PB, שני ציוני תשובה: tail15 (ממוצע 10 הערכים הגדולים של מסת ההסתברות מחוץ ל-top-15) ו-digit_rate (אי-הסכמות ספרות / הזדמנויות ספרות). מדרגים כל אחד בתוך תא מודל×מערך (midrank), ממצעים את שני הדירוגים, ופותחים אם ≥ .33. **תלוי ב-digit; נפסל.** החלופה ללא digit (Codex, Step 399+): ממוצע tail15 Top10 בלבד, אותו סף.
- **מדד PB:** לכל אחד מ-8 התאים, F1 בין דיוק על תשובות נקיות (חיזוי "אין שגיאה") לדיוק מדויק על תשובות שגויות (peak = צעד השגיאה הראשון); macro על 8 התאים. תשובה שגויה עם gate סגור נספרת כהחמצה.
- **מדד PRMB:** לכל תשובה עם צעדים מסומנים, AUC של score_a כשצעדי label 1 (סיכון גבוה) הם החיוביים; ממוצע על התשובות. **ללא gate.**
- **אי-ודאות:** bootstrap זוגי לפי source group, 2,000 draws, אינטרוול 95%.

### תוצאות השיטה המובילה (PB / within)
| | L08 | L11 | L14x | L24 |
|---|---|---|---|---|
| Continuous L-SML | 43.74 / .778 | 41.19 / .766 | 39.47 / .756 | 38.91 / .751 |
| **Joint grouping + two-stage SML readout** | 43.48 / .777 | 42.37 / .770 | 43.02 / .777 | 43.22 / .775 |
| הפרש, 95% | −0.26 [−0.73, +0.22] | +1.18 [+0.14, +2.19] | +3.55 [+2.32, +4.72] | +4.31 [+3.00, +5.62] |

ייחוסים היסטוריים על אותו gate: incumbent digit025 43.25 / .776; ללא digit (gate tail15 בלבד): innovation5 39.83 / .760, BOCPD+innovation5 40.37 / .763.

---

## מה עוד לא נבדק, בסדר חשיבות
1. אותו מתכון על ה-bank ללא digit (50 זרמים של Codex), עם gate ללא digit ועוגן סימן ללא digit.
2. הזרוע answer-only: Stage A-D מהתשובה עצמה בלבד (מספר צעדים קטן, כמו במסלול ה-110).
3. confirmation על נתונים שלא שימשו לפיתוח.

---

## נספח (אותו יום, אחרי בדיקה): מה ה-readout מחשב בפועל

בדיקה על אותן חלוקות ואותם folds (`results/joint_group_readouts_v1`), OOF מלא, PB / within:

| readout על החלוקה של Joint (K=3) | L08 | L11 | L14x | L24 |
|---|---|---|---|---|
| SML בתוך + כלל K=3 בין (הזרוע המובילה) | 43.48 / .7767 | 42.47 / .7710 | 43.02 / .7773 | 43.22 / .7754 |
| **ממוצע בתוך + כלל K=3 בין** | 43.52 / .7769 | 42.08 / .7699 | 42.54 / .7752 | 43.24 / .7757 |
| SML בתוך + eigen-solve בין (בלי הכלל) | 40.43 / .7648 | 41.05 / .7678 | 40.84 / .7674 | 39.54 / .7567 |
| ממוצע בתוך + eigen-solve בין | 41.20 / .7691 | 40.69 / .7626 | 41.06 / .7711 | 40.35 / .7658 |

- הווקטורים העצמיים התוך-קבוצתיים כמעט אחידים: |cos(e_g, uniform)| ≥ .93 בכל קבוצה, ≥ .997 ברובן. לכן "SML בתוך קבוצה" שקול לממוצע.
- הכלל ב-K=3 (משקל שווה לכל קבוצה אחרי סטנדרטיזציה) הוא מה שעובד. החלפתו ב-eigen-solve אמיתי בין שלוש הקבוצות עולה כ-3 נקודות בכל roster.
- **המסקנה: הזרוע המובילה שקולה ל"חלוקה אוטומטית ל-3 קבוצות + ממוצע בתוך + ממוצע בין (אחרי סטנדרטיזציה)".** לא מודל הפקטורים של Joint ולא ה-SML נושאים את התוצאה. מה שנושא אותה הוא החלוקה.

הוכחה שהחלוקה היא הרכיב הפעיל (אותו כלל ממוצע, חלוקות שונות):

| כלל הממוצע על החלוקה של... | L08 | L14x | L24 |
|---|---|---|---|
| Joint, K=3 | 43.52 | 42.54 | 43.24 |
| L-SML (Eq.15/Eq.14; K=3 / 5 / 7) | 43.84 | 41.20 | 40.89 |
| בלי חלוקה (ממוצע על כל הזרמים) | 42.53 | 41.58 | 40.20 |

### בחירת K: מה קורה כשמרחיבים ל-3..8 (fold 0)
| roster | K נבחר | סיבה |
|---|---|---|
| L08 | 3 | K≥4 יוצרים קבוצה של 1 → לא קבילים |
| L14x | 3 | K=3,4,5 כולם יציבים לחלוטין (ARI 1.0) → שובר-השוויון "K קטן" בוחר 3 |
| L24 | 3 | K=3 ו-K=7 יציבים לחלוטין (K=7: 3/3/5/3/5/3/2, digit מבודד גם שם) → "K קטן" בוחר 3 |

כלומר K=3 אינו "מתגלה": הוא ה-K הקטן ביותר שמותר (זיהוּת דורשת K≥3) ושובר-השוויון מעדיף אותו בכל פעם שהחלוקה יציבה בכמה K. הרווח מול L-SML נובע מכך שעם 3 קבוצות המשפחה המשלימה מקבלת 1/3 מהמשקל, ועם 7 קבוצות 1/7.

### ניסוח נכון של השיטה הנוכחית
"קיבוץ ספקטרלי label-free של הזרמים על השארית |C − vvᵀ|, במספר הקבוצות הקטן ביותר שהוא קביל ויציב תחת leave-one-source-out, ואז שקלול שווה בתוך קבוצה ושווה בין קבוצות אחרי סטנדרטיזציה." ה-Joint factor model משמש רק להגדרת השארית ולאודיט. זו שיטה אלגוריתמית (החלוקה נלמדת מהנתונים ללא labels, והיא מה שמייצר את ההפרש +3 מול ממוצע רגיל ב-L24), אבל היא **לא** fusion משוקלל, וכל תיאור שלה כ-"Joint L-SML" או "SML" מטעה. Omri אישר ב-2026-09-17 שממוצע רגיל בין פיצ'רים אינו הפתרון המבוקש; ההבדל כאן הוא שהממוצע הוא בין בלוקים שנבחרו אוטומטית, לא בין פיצ'רים.
