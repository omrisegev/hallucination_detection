# Peer review מתמטי: telemetry + L-SML כ**מתחרה standalone** ל-PRM (לא כ-boost) — 2026-10-25

**השאלה של עומרי.** בלי הקושי של fusion עם ה-PRM output, האם השיטה כ-standalone מבוססת יותר מתמטית?

**בסיס.**
- קוד ב-`claude/prm-boosting` @ `00140d8d`: `results/prm_boost_ablation5_v1/ablation5.py` (arm `TELE_<bank>` = L-SML על ה-bank בלבד, sign לפי `V.mean(1)`), `spectral_utils/fusion_utils.py` (`lsml_continuous_guarded`).
- תוצאות: `results/prm_standalone_telemetry_v1/tables/STANDALONE_TABLES.md` (base model, PRMBench), `results/prm_boost_collection_v1/tables/STATUS_MAIN.csv` (backbone telemetry, PRMBench/Socratic/H2V), `results/prm_boost_ablation5_v1/ABL5_RESULTS_*.csv` (RES/PERM/POS controls), `results/telemetry_class_correlation_v1/README.md`.
- תאוריה: `docs/research_notes/math_rigor_2026-10-24/FRAMING_THEORY.md` (מודל M, P1–P6), `GAPS_STATUS_HE.md`.
- השוואה ל: `reviews/math_peer_review_2026-10-24/MATH_PEER_REVIEW_HE.md` (W1–W12).
- **לא הורץ שום job.** חישוב אחד: דוגמת צעצוע Gaussian ב-numpy (סעיף 4).

---

## 0. תשובה קצרה

**לא, בסך הכול לא יותר מבוסס. ה-estimator נקי יותר, אבל ה-detector פחות מזוהה.**

| ממד | boosting (CHAIN) | standalone (TELE) | מי מבוסס יותר |
|---|---|---|---|
| תנאי שיפור מעל input חזק (W3) | צריך P2 + gate; ה-gate נכשל (abstain 15/15) | לא רלוונטי: אין input חזק לשמר | **standalone** |
| איכות המשקלות (W4, W5, P3) | view חזק אחד + views חלשים → SML מוריד ל-PRM משקל | views הומוגניים → SML ≈ Fisher ≈ mean | **standalone** |
| חלק ה-PRM תלוי ב-K (W8), "PRM בקבוצה משלו" (A7) | בעיה מאושרת | נעלם | **standalone** |
| anchor flip (W9) | הפך את Socratic MS ל-.468 | sign גלובלי לפי ממוצע ה-bank; פחות שביר | **standalone** |
| רעש משותף PRM–telemetry (אותו backbone) | שבר את ה-gate | נעלם (ב-base model) | **standalone** |
| label-free באמת | PRM + backbone שאומנו עם labels | base model לא ראה step labels | **standalone** |
| **זיהוי ה-latent כ"שגיאה" (W1)** | P5: ה-PRM הוא עוגן מזהה (marker indicator) | **אין עוגן. P4: לא מזוהה מ-second moments** | **boosting** |
| CI בין views (W2) | מופר בין קבוצות | 16 ערוצים מאותו softmax; correlation תלוי-class | **boosting** (מעט) |
| חשיפה ל-artifact של PRMBench | ה-PRM מושך לכיוון שגיאה אמיתית | חשוף לגמרי | **boosting** |
| answer-z, GAP, transductive, forking paths (W6, W10, W12) | זהה | זהה | תיקו |

**שורה תחתונה של הסוקר:**
- ה-fusion כמעט נעלם כבעיה מתמטית: ב-standalone, L-SML ≈ oriented mean (+.001 עד +.012 within-AUC).
- אבל השאלה "למה ה-factor הדומיננטי הוא שגיאה" עוברת למרכז. ב-boosting ה-PRM ענה עליה. ב-standalone התשובה היחידה היא prior של orientation + המבנה של PRMBench.
- הראיות מתאימות לזה: standalone מנצח רק PRMs חלשים, ומחוץ ל-PRMBench ה-telemetry מפסיד ל-PRM ול-position-only.
- **Verdict:** reject כ-"label-free competitor to PRMs"; major revision אם מוסיפים תנאי זיהוי label-free (סעיף 5) ובדיקת artifact.

---

## 1. מה השיטה standalone מניחה (formal)

`TELE_N16` = 16 ערוצים גולמיים (`NESTED['N16']`) → `answer_z` → L-SML עם partition אוטומטי (Eq.14/15 + spectral) → sign גלובלי: `corr(s, V.mean(1)) > 0` → GAP.

לפי מודל (M) של `FRAMING_THEORY.md`, בלי view 0:

    x_ijk = a_jk + Δ_k·y_ij + β_k·z_ij + λ_k·η_ij + ε_ijk,   k = 1..16

| הנחה | נדרש ע"י | ב-standalone | נבדק? |
|---|---|---|---|
| A1. latent בינארי יחיד y מסביר את ה-off-diagonal | Parisi 2014 | **ללא עוגן** שמזהה ש-y = שגיאה | לא (סעיף 3, S1) |
| A2. CI בין קבוצות בהינתן y | Jaffe 2016 L1 | כל הערוצים פונקציות של אותה התפלגות next-token | מופר; R_err ≠ R_ok (frob .94–1.89 מול null .30–.50) |
| A3. block structure בתוך קבוצה | Jaffe 2016 | ARI חציוני 1.00 ב-PRMBench (Step 474, ל-CHAIN) | standalone לא נבדק |
| A4. Δ_k ≥ 0 לכל view | Jaffe 2016 L4 | orientation "by construction"; 2/16 הפוכים ב-PRMBench | חלקית (HISTORY_AUDIT W9) |
| A5. iid | moments | steps מקובצים ב-answers | מופר (כמו קודם) |
| A6. views בינאריים | Lemma 1 | רציפים | מחוץ לתאוריה (כמו קודם) |
| **A8 (חדש). sign: "ממוצע הערוצים הגולמיים עולה עם שגיאה"** | הגדרת הכיוון | prior ידני, לא נגזר | לא |

---

## 2. W1–W12: מה נעלם, מה נשאר, מה החמיר

| # | חולשה ב-boosting | ב-standalone | למה |
|---|---|---|---|
| W1 | latent ≠ שגיאה | **מחמיר → S1** | ב-boosting P5 נותן זיהוי דרך עוגן ה-PRM. כאן אין עוגן |
| W2 | CI בין קבוצות מופר | **נשאר, בצורה חזקה יותר → S2** | אין אפילו view אחד חיצוני; כל הערוצים מאותו forward pass |
| W3 | אין תנאי שיפור מעל PRM | **נעלם** | אין input חזק לשמר. השאלה הופכת להשוואה ישירה (סעיף 3, S4) |
| W4 | משקלות רציפים לא Bayes-optimal | **כמעט נעלם** | P3: עם loadings דומים, v ∝ u ≈ Fisher. זה הרגים ש-SML תוכנן אליו |
| W5 | shrinkage לכיוון equal | **נעלם כבעיה, נשאר כשאלת תרומה** | equal ≈ optimal ל-views הומוגניים. אבל אז SML ≈ mean: +.012/+.002/+.001/+.005/+.006 within-AUC (standalone, 5 PRMs) |
| W6 | answer-z | **נשאר** | P1 מצדיק centering; ה-scaling נשאר לא מוצדק. זהה |
| W7 | grouping heuristic; סתירת A7 | **A7 נעלם; stability לא נבדק ל-standalone** | אין PRM לשים בקבוצה. ה-partition של base model לא נבדק ב-ARI |
| W8 | PRM share תלוי K | **נעלם** | אין PRM |
| W9 | sign rule + anchor flip | **נחלש → A8** | אין flip לכיוון PRM. sign גלובלי לפי `V.mean(1)` = prior של orientation |
| W10 | Q80/GAP לא נגזרים; p לא מזוהה | **נשאר** | P6 זהה. GAP ≈ construction prior "~2 שגיאות" |
| W11 | bootstrap בלי refit | **נשאר ל-standalone** | Step 474 סגר רק ל-CHAIN. ה-CIs ב-`STANDALONE_TABLES.md` הם fixed-score |
| W12 | transductive + forking paths | **נשאר, וחשוב יותר → S5** | N16, answer-z, GAP נבחרו על PRMBench; ה-PRM לא ראה את PRMBench |

**ספירה:** נעלמים 4 (W3, W4, W8, A7-of-W7). נחלשים 2 (W5, W9). נשארים 4 (W6, W10, W11, W12). מחמירים 2 (W1, W2). חדשים 3 (S3, S4, S6; ראו סעיף 3).

---

## 3. חולשות חדשות / מחודדות ב-standalone

### S1. Identification בלי עוגן (קריטית; מחליפה את W1)
- **הטענה:** תחת (M) עם nuisance η, ה-off-diagonal של ה-telemetry הוא

      C_off = uuᵀ + λλᵀ   (off-diagonal),   u = √Var(y)·Δ

  זה rank-2. כל סיבוב אורתוגונלי של [u λ] נותן אותו C_off. **מ-second moments אין דרך לדעת איזה כיוון הוא שגיאה.**
- ב-boosting, P5 פותר את זה דרך עוגן: λ_0 = 0 על ה-PRM. **ב-standalone אין view עם λ = 0.**
- **מה SML עושה בפועל:** ה-eigenvector המוביל = הכיוון עם ה-energy הגדולה יותר. אם u ⟂ λ: switch חד ב-‖λ‖² ≈ ‖u‖² (toy, סעיף 4).
- **ראיה ישירה מהפרויקט:** AUTO-F (partition אוטומטי, בלי עוגן PRM) נכשל ב-5/5. `GAPS_STATUS_HE.md`: "בלי עוגן, הגורם המשותף הוא ה-nuisance". זה בדיוק המצב של standalone.
- **למה זה עובד ב-PRMBench בכל זאת:** שגיאות סינתטיות → ‖u‖ גדול; position-only רק .662. ב-Socratic position-only .731, ב-H2V .866 → λ דומיננטי.
- **מה מזהה בכל זאת (שלוש דרכים):**

| דרך | מה היא דורשת | מצב |
|---|---|---|
| prior של orientation (16 סימנים) | Δ_k ≥ 0 ידוע מראש; λ עם סימנים מעורבים מתבטל בממוצע | זה מה שקורה היום בפועל (`V.mean(1)`). לא מוצהר |
| third-order cumulants | κ₃(y) ≠ 0 (p ≠ ½) ו-κ₃(η) = 0 | **relative position אחיד → κ₃ = 0.** מזהה u בלי עוגן. לא נבדק. סיכון: heavy tails (P5) |
| הוצאת z ידוע (position, length) | η ⊂ span(z) | TELERES שורד ב-PRMBench (סעיף 6). לא מכסה η לא-נצפה |

### S2. CI בין ערוצים מאותו softmax (גבוהה; מחודד מ-W2)
- 16 הערוצים (surprisal, entropy, energy, tail, turnover, digit) הם פונקציות של אותה התפלגות next-token. מסגרת Dawid–Skene של "classifiers עצמאיים" לא מתאימה מהשורש. המסגרת הנכונה: **factor analysis** על features, לא unsupervised ensemble.
- **ראיה:** המבנה תלוי-class. Frobenius של R_err − R_ok = .94–1.89 מול null q95 .35–.55, שורד class-centering ו-position terciles (26/30 תאים). `telemetry_class_correlation_v1`.
- **השלכה:** (M) מניח Σ משותף לשני ה-classes. מופר. אבל ה-headroom הריבועי קטן (+.001..+.010), אז זה פוגע בטענת התאוריה יותר מאשר בביצועים.

### S3. Artifact של PRMBench (קריטית ל-standalone; חדשה ביחס ל-review הקודם)
- **הטענה:** teacher-forced telemetry מודד "כמה הטקסט מפתיע את ה-base model". שגיאות PRMBench הוכנסו ע"י LLM ששכתב צעד. latent אפשרי: "צעד ששוכתב" ולא "צעד שגוי".
- ב-boosting ה-PRM (אומן על שגיאות טבעיות) מושך את ה-latent לכיוון שגיאה. **ב-standalone אין שום דבר שמושך.**
- **ראיות עקיפות (לא הוכחה):**

| benchmark | סוג שגיאות | telemetry (backbone) within-AUC | PRM | position-only |
|---|---|---|---|---|
| PRMBench | perturbation סינתטי | .748–.823 | .662–.801 | .662 |
| Socratic | — | .582–.712 | .549–.824 | .731 |
| Hard2Verify | טבעיות | .480–.562 | .185–.763 | .866 |

- **בדיקה זולה שמפרידה:** telemetry על צעד מקורי מ-PRM800K מול הגרסה המשוכתבת שלו, כשהשכתוב **לא** משנה נכונות (paraphrase). אם ה-telemetry מסמן paraphrase נכון, זה artifact detector. דורש GPU (forward pass על ~1k צעדים, base model 7B, דקות).

### S4. הוגנות ההשוואה ל-PRM מאומן (גבוהה; חדשה)
- **העיצוב הטוב ביותר שיש לנו:** base model M מול PRM שעבר fine-tune מ-M. זה מודד "מה step-label fine-tuning קונה". זו טענה נקייה ומעניינת.

| PRM | base model | within-AUC: standalone − PRM [95% CI] | PRMScore GAP: standalone − PRM [95% CI] |
|---|---|---|---|
| Qwen-7B | Qwen2.5-Math-7B-Instruct | −.008 [−.014, −.001] | +0.26 [−0.36, +0.85] |
| Qwen-72B | Qwen2.5-Math-72B-Instruct | −.014 [−.021, −.006] | −1.69 [−2.36, −1.04] |
| PURE | Qwen2.5-Math-7B | +.008 [+.000, +.015] | +1.40 [+0.78, +1.99] |
| RLHFlow | Llama-3.1-8B-Instruct | +.125 [+.116, +.135] | +8.81 [+8.10, +9.55] |
| Math-Shepherd | Mistral-7B-v0.1 | +.076 [+.066, +.085] | +6.73 [+6.00, +7.53] |

- על ranking (המדד הראשי, E1): **standalone מפסיד לשני ה-PRMs החזקים**. מנצח בבירור רק PRMs עם MC labels חלשים, שאף מתרגל לא יבחר.
- **הטיות נגדנו (מחמירות את המספרים לטובתנו):** כל ה-design נבחר על PRMBench; ה-PRM zero-shot. ה-fit טרנסדוקטיבי משתמש ב-covariance של ה-test, ה-PRM לא.
- **הטיות לטובתנו (שמרניות):** אותו GAP rule ל-PRM; ה-scale של ה-PRM (prob) נבחר עם labels לטובת ה-PRM.
- **מחלקת המתחרים הנכונה:** ל-label-free detector, ה-baselines באותה רמת גישה הם single-channel uncertainty (mean surprisal, entropy), position-only, length-only. ה-PRM הוא reference עליון, לא היריב. **חסר: best single channel ו-`chosen_surprisal` לבד**, כדי להראות ש-16 ערוצים + SML מוסיפים על baseline UQ פשוט.
- **backbone telemetry אינו label-free:** hidden states של PRM שאומן עם labels. הגרסה ה-label-free באמת היא ה-base model, והיא חלשה מה-backbone ב-Qwen (−.016..−.042 within-AUC).

### S5. Forking paths: משקל גבוה יותר ב-standalone
- ב-boosting, ה-PRM הוא input שלא נבחר על PRMBench, וזה מגביל את מרחב הבחירות. ב-standalone **כל** רכיב נבחר על PRMBench: bank N16, orientation, answer-z, GAP, cap ⌈n/2⌉.
- Split-half (Step 474) סוגר את ה-fitting, לא את בחירות ה-design. אין holdout שלא נגעו בו.
- **Base-model telemetry מעולם לא הורץ מחוץ ל-PRMBench.** כל מספרי Socratic/H2V הם backbone.

### S6. SML לא תורם; ה-"identification" בפועל הוא ה-prior
- L-SML − mean (standalone, within-AUC): +.012, +.002, +.001, +.005, +.006. ב-PRMScore: +1.13, +0.42, +0.58, +0.75, +0.81.
- Step 468: oriented mean ≥ L-SML ב-7/10 banks (Qwen3 source).
- **מסקנה:** השיטה standalone היא בפועל "ממוצע מכוון של 16 ערוצי UQ, אחרי answer-centering". הבסיס המתמטי שלה הוא P1 (centering) + prior של orientation. ה-SML לא נושא את הטענה, אז גם לא את ה-"label-free theory".

---

## 4. דוגמת צעצוע: switch ה-identification בלי עוגן (numpy, לא נתונים אמיתיים)

Setup: 8 telemetry views, Δ = 0.5 לכולם, p = .3, N = 200k, z-scored. nuisance η עם loadings λ = s·dir.

**B. nuisance אורתוגונלי ל-signal (חצי ערוצים +, חצי −):**

| s | cos(SML, u) | AUC SML | AUC oriented mean | AUC Fisher (oracle) |
|---|---|---|---|---|
| 0 | 1.00 | .842 | .842 | .842 |
| 0.5 | 1.00 | .841 | .841 | .841 |
| **1.0** | **.001** | **.501** | .841 | .841 |
| 2.0 | .001 | .500 | .841 | .841 |

**C. nuisance חלקית באותו כיוון כמו ה-signal:**

| s | cos(SML, u) | AUC SML | AUC mean | AUC Fisher |
|---|---|---|---|---|
| 0 | 1.00 | .841 | .841 | .841 |
| 1 | .82 | .719 | .795 | .807 |
| 2 | .68 | .613 | .730 | .782 |
| 3 | .63 | .570 | .686 | .768 |

**D. nuisance מתואם עם y (כמו position בשגיאות טבעיות, ρ = .4), signal חלש (Δ = 0.2):**

| s | cos(SML, u) | AUC SML | AUC mean |
|---|---|---|---|
| 1 | .07 | .679 | .654 |
| 3 | .01 | .737 | .642 |

- **B:** בלי עוגן, SML עובר בבת אחת מ-error ל-nuisance כש-‖λ‖² עובר את ‖u‖². ה-oriented mean שורד רק כי ה-λ כאן מתבטל בסכום (prior של orientation).
- **C:** כש-λ לא מתבטל, גם ה-mean נפגע; Fisher (עם labels) נשאר גבוה. הפער = מחיר חוסר הזיהוי.
- **D:** SML "עובד" (.737) אבל cos עם u ≈ 0: הוא מדרג לפי ה-nuisance. **within-AUC טוב לא מוכיח שה-latent הוא שגיאה.** זה התרחיש המסוכן ל-H2V/Socratic.
- סקריפט: `toy_standalone_identification.py` בתיקייה זו.

---

## 5. מה יהפוך את standalone למבוסס (תוכנית, הכל CPU חוץ מ-#3)

| עדיפות | מה | label-free? | עלות | פותר |
|---|---|---|---|---|
| 1 | **אבחון label-free ל-identification:** corr(TELE score, position) ו-corr(TELE, log length); eigengap λ₁/λ₂ של C_off. ניבוי: ב-Socratic/H2V |corr| גבוה ו-eigengap קטן | כן | שעות | S1, W1 |
| 2 | **Third-order identification:** tensor של κ₃ על ה-group scores; position אחיד → κ₃(η)=0. להשוות את כיוון ה-tensor לכיוון SML | כן | 1–2 ימים | S1 (הדרך היחידה לזיהוי בלי עוגן) |
| 3 | **בדיקת artifact:** telemetry של base model על paraphrase נכון מול perturbation שגוי מאותו צעד PRM800K | כן (labels רק ל-eval) | GPU, שעה | S3 |
| 4 | **Baselines באותה רמת גישה:** best single channel, `chosen_surprisal`, mean entropy, POS_ONLY, LEN_ONLY — על standalone | כן | שעות | S4, S6 |
| 5 | **Base-model telemetry על Socratic + H2V** (לא backbone) עם design קפוא | כן | GPU, ~שעה ל-PRM | S5, S3 |
| 6 | refit bootstrap + ARI ל-TELE standalone (הקוד של Step 474) | כן | שעות | W11, W7 |
| 7 | להצהיר את ה-prior: "Δ_k ≥ 0 בכיוון הגולמי" ולבדוק את 2/16 ההפוכים | כן | שעות | A8, W9 |

---

## 6. Steelman: מה כן מחזיק

| טענה | ראיה | הסתייגות |
|---|---|---|
| telemetry נושא signal אמיתי ב-PRMBench, לא רק position/length | TELERES_N16 (residualized) .767–.807; PERM .49–.56 (backbone) | backbone, לא base; PRMBench בלבד |
| base model label-free ≈ PRM מאומן על Qwen-7B ב-PRMScore | +0.26 [−0.36, +0.85] | ranking −.008; dev data |
| step-label fine-tuning קונה מעט מעבר ל-telemetry של אותו base | Qwen-7B/72B: .008–.014 within-AUC | שאלה מדעית לגיטימית, לא method claim |
| ה-fusion פשוט ומובן (P1 + oriented mean) | L-SML ≈ mean | זה גם חיסרון: אין תרומה של SML |

## 7. מה מותר לטעון היום (standalone)

| טענה | מותר? |
|---|---|
| "label-free telemetry of the base model matches a step-label-trained PRM on PRMBench" | **רק ל-Qwen-7B ב-PRMScore**, ולא ב-ranking; dev data |
| "outperforms PRMs" | **לא** (מפסיד ל-72B; מנצח רק PRMs חלשים) |
| "מבוסס על SML עם ערובות" | **לא**. אין ערובת זיהוי; SML ≈ mean |
| "מזהה שגיאות ולא artifact של הבנצ'מרק" | **עדיין לא** (S3, #3) |
| "בשגיאות טבעיות" | **לא**. base model לא נבדק; backbone מפסיד ל-position |
| "fine-tuning על step labels מוסיף מעט מעבר ל-telemetry של אותו base, ב-PRMBench" | **כן**, כממצא ניתוח, עם 5 זוגות |

## 8. מקורות
- `results/prm_standalone_telemetry_v1/tables/STANDALONE_TABLES.md` (standalone vs PRM, L-SML vs mean)
- `results/prm_boost_collection_v1/tables/STATUS_MAIN.csv` (TELE_N16, PRM, PRMBench/Socratic/H2V)
- `results/prm_boost_ablation5_v1/ABL5_RESULTS_*.csv` (TELERES, TELEPERM, POS_ONLY, LEN_ONLY)
- `results/prm_boost_ablation5_v1/ablation5.py` שורה 111 (`TELE_<bk>`, anchor = `V.mean(1)`)
- `results/telemetry_class_correlation_v1/README.md` (class-dependent correlation)
- `docs/research_notes/math_rigor_2026-10-24/FRAMING_THEORY.md` (P1, P3, P4, P5), `GAPS_STATUS_HE.md` (AUTO-F, Step 474)
- `docs/handoffs/HONEST_PROJECT_REVIEW_HANDOFF_2026-10-25.md` §3.3 (perturbation artifact)
