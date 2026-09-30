# HANDOFF – הכנה לפגישת מנחים, 2026-09-17

סיכום של כל מה שנעשה בסשן הזה (Claude, 16-17.9) ושל עבודת Codex המקבילה על אותו ענף.
כל המספרים הם development evidence על 13,769 תשובות (8 תאי ProcessBench + PRMBench),
5 source folds, ללא confirmation נקי. ענף: `codex/fusion-independence-atlas-v1` (הכול נדחף ל-remote).

---

## 1. השורה התחתונה במשפט אחד לכל נושא

| נושא | תשובה |
|---|---|
| האם Joint L-SML חייב קבוצות ידניות? | לא. יש לו גילוי קבוצות משלו (LOAO consensus). מגבלת "3 בקבוצה" היא זיהוּת (identifiability) של גורם rank-1, ומסלול הזוגות (מכפלה אחת לקבוצה בגודל 2) מסיר אותה. |
| האם shrinkage כמו ב-IU-PCR עוזר ל-Joint? | ה-shrinkage ב-IU הוא בעצם "Joint בלי הפיט". בתוך Joint הוא כבר קיים ב-readout (ridge על ההיפוך) ולא נתן רווח. הפתרון הרלוונטי הוא readout, לא shrinkage. |
| מה הכשל של Joint מול Continuous L-SML? | ה-readout הגלובלי (Σ⁻¹v או v) מדכא זרמים משלימים (digit). ה-readout ההיררכי, שנותן קול לכל קבוצה, מחזיר אותם. |
| האם Joint עמיד ליתירות? | חלקית. כשקבוצת ה-digit נקייה (24 זרמים: 43.13% מול 38.91% ל-Continuous), כן. כשזרם זר נכנס לקבוצה (L08+3/+6), לא. המנגנון: בחירת K קטן ויציב + קול לקבוצה, לא הגורם המשותף. |
| החלטת Omri: digit | מנגנון אי-ההסכמה על ספרות **נפסל** (משווה לספרה שבפתרון הנתון, סותר את הרעיון המקורי). כל ה-incumbent-ים עם digit (43.25, L08 43.74, L24 43.13) הם היסטוריה. |
| מה הטוב ביותר ללא digit? | ייחוסים היסטוריים: innovation5 39.83% / .7603; BOCPD+innovation5 40.37% / .7632. כל השיטות החדשות ללא digit (Codex, Steps 399-412): 37.8-38.96%, אף אחת מעל 40. |

---

## 2. מה Claude עשה בסשן (Step 398)

### 2.1 ניתוח קוד
- `joint_lsml.py` (1,021 שורות, זהה בכל הענפים): המודל C = v·vᵀ + blockdiag(u·uᵀ) + D, פיט חילופי עם 5 starts,
  אודיט multistart + Jacobian, גילוי קבוצות `discover_loao_consensus_groups` (LOAO spectral clustering על |C − vvᵀ|,
  K ∈ {3,4,6,8} לפי median ARI).
- `shrinkage_iu.py`: C_α = (1−α)C + αT, target "joint" = איברים חוצי-קבוצה מוחלפים ב-rank-1. Step 332: +0.0028 within,
  +0.61pp PB, ובקרת diag כמעט זהה.
- מודולי הזוגות (`joint_pair_extension.py`, `joint_pair_jacobian.py`) לא היו מחויבים באף ענף → חויבו ונדחפו.

### 2.2 ניסוי: עמידות ליתירות (`results/joint_redundancy_robustness_v1`)
PB macro-F1 % / PRMB within-AUC. Gate קפוא (tail15 + digit_rate, midrank .33). כל עוגני ה-replay מדויקים.

| זרוע | L08 | L08+3 | L08+6 | L14 (Codex) | L24 |
|---|---|---|---|---|---|
| Continuous L-SML | 43.74 / .778 | 41.19 | 39.47 | 39.35 | 38.91 / .751 |
| Joint זוגות, היררכי, קבוצות משלו | 43.59 / .777 | 39.73 | 40.36 | 40.30 | **43.13 / .775** |
| Joint זוגות, היררכי, קבוצות L-SML | 43.81 / .778 | 38.96 | 39.19 | 39.21 | 38.54 |
| Joint global-v / model-inverse | 39.97 / 38.64 | 39.59 / 38.30 | 39.81 / 39.03 | 39.87 / 38.95 | 39.26 / 37.85 |
| IU-PCR | 38.57 | 39.52 | 38.71 | 38.77 | 26.07 |
| equal (ייחוס בלבד, לא שיטה) | 42.53 | 42.38 | 41.58 | 41.65 | 40.20 |

Bootstrap זוגי (2000 draws, קבוצות מקור, 95%):
- L08: Joint היררכי − Continuous = −0.15pp [−0.63, +0.34]; על קבוצות L-SML +0.07 [−0.10, +0.26]. שקולים.
- L24: Joint היררכי − Continuous = **+4.22pp [+2.91, +5.54]**, within +.025 [+.021, +.028].
- ירידה מ-L08 ל-L24: Continuous −4.83 [−6.25, −3.37]; Joint היררכי −0.46 [−0.85, −0.07].

מנגנון (מאודיטים לכל fold):
- זוגות: L08 → קבוצות 2/3/3, זוג אחד, כל fold PASS multistart/native/Jacobian (cond 8.4).
- חלק המשקל של digit: היררכי .34, global-v .03-.07, inverse .00-.01, IU .04-.18, Continuous .34 (L08) / .02 (L24, K=7).
- L24: Joint בחר K=3 = {digit×3 | 13 | 8}. Continuous בחר K=7. זה כל ההבדל.
- L08+3/+6: קבוצת ה-digit = {digit×3, **Renyi a0.25**}. `hierarchical_joint_weights` בונה את המסווג הווירטואלי
  של הקבוצה מ-v (שבו digit זעיר) → הזרם הזר משתלט → digit .13.

**הווריאנט הבא שהוצע (לא נבנה):** readout תוך-קבוצתי ב-Joint שלא תלוי ב-v (גורם הקבוצה u_g או SML תוך-קבוצתי).
**הערה:** כל זה נמדד עם digit, שנפסל אחר כך. המנגנון (K קטן + קול לקבוצה + טוהר קבוצה) תקף לכל זרם משלים.

### 2.3 תפעול
- ניקוי דיסק: 0.68 GB → 4.6 GB (6 worktrees ממוזגים ונקיים הוסרו). 5 נוספים ממוזגים עם תיקיית תוצאות אחת לא מחויבת
  (`answer-position-fusion-v1`, `deem-b3-probability-moments-v1`, `moment-rbm-fusion-v1`, `rbm-two-axis-fusion-v1`, `rbm-weight-shrinkage-v1`).
- נתוני האטלס נמשכו מ-Drive ב-rclone מקומי (remote `gdrive:`, אזהרה: client_id משותף נפסק ב-2026).
- sparse worktree ל-atlas: `.worktrees/fusion-independence-atlas-v1` (162 MB).
- קבצים untracked בעבודה הראשית שאף ענף לא מכיל: `answer_localization_v2.py`, `fusion_context_bank.py`,
  `fusion_reliability_regularization.py`, `shrinkage_iu.py` (המסלול answer-only של ה-110). כדאי לחייב.

---

## 3. מה Codex עשה במקביל (Steps 397-412, חויב היום כמו שהוא, לא כל חלקיו מאושרים)

### 3.1 Step 397 – Fusion Independence Atlas + L-SML locator
- 5,443 הגדרות סיגנל/readout, 114 נציגים מ-13 משפחות, 2,879 זוגות: **אף זוג לא עבר מבחן אי-תלות** בשגיאות (PB+PRMB).
  digit + tail15 הם "dependent-complementary".
- L08 Continuous L-SML נבחר 5/5 folds: 43.74 / .7781 (+0.49pp מול incumbent, CI חוצה אפס).
- Soft Joint = Hard (ARI 1.0); affinity ללא K: cond 1.5e12, נחסם.

### 3.2 Step 398 (Codex) – SLA
Raw SLA (לוקליזציה על traces שגויים בלבד, לפני gate): digit025 38.18% pooled; L08 38.97% pooled, 41.96/40.96 Qwen-4B/8B.
Mind-the-Gap Table 3 Shannon Drop: 39.14/39.39 → L08 +2.8/+1.6pp ברמת נקודה, **לא** טענת עליונות matched
(זהות traces לא מאומתת). החלטה: כל דוח לוקליזציה מציג 3 lanes: raw SLA, gated exact accuracy, end-to-end macro-F1.

### 3.3 Steps 399-400 – digit-free bank 50, אבחון gate/locator
- Bank 50 ללא ספרות (47 + top2/top1 ratio, top15 turnover, JS סמוך). Gate ללא ספרות: ממוצע Tail15 Top10 (תיקון ניסוח: לא "prominence").
- Broad50: Continuous 37.76, Joint 38.05, Joint מאוזן 37.99, equal 37.90, H1 36.35. ייחוס innovation5 39.83.
- אבחון: 2,428 החמצות locator בלבד, 293 gate בלבד, 528 שניהם, 935 אזעקות שווא על נקיות → הרגרסיה היחסית ב-locator, אבל ה-gate רחוק משלמות.
- בחירת פיצ'רים בתוך Joint (חילופין: פיט ↔ אלימינציה לפי מידע מותנה, 95% מידע): 33-38 זרמים, 38.27/.7480; עם BOCPD 38.60/.7491 (BOCPD נשמר 5/5). within מובהק, PB לא.
- Top8 מול Top10: PB −1.74, within +.0034 (tradeoff). מסלול N=8..50: מקסימום 38.41 (N=37).

### 3.4 Steps 401-412 – 12 מבחני עמידות (עותקים מדויקים / רעש iid / עותקים קרובים / nuisance מובנה)
| שלב | וריאנט | base | הישג | כשל |
|---|---|---|---|---|
| 401 | selector 95% | 38.60 | | עותקים −1.5pp, רעש −.0097 |
| 404-405 | sparse membership | 38.73 / .7501 | **עותקים מדויקים ורעש iid: זהות ביט-לביט** | |
| 406 | stress מובנה | | | near-copies −1.0pp, nuisance −0.8 |
| 407-411 | signal/staged/regroup/feasible/minimax | 38.07-38.73 | תיקונים נקודתיים | near או שימור baseline |
| 412 | mass-aware grouping (QP) | **38.96 / .7542** | exact/structured | iid/near: 40-60% fallback ל-H1 |

Codex עצמו: "אין להסיק עמידות כללית מהמבחנים הסינתטיים"; כל הפיטים hybrid (קווריאנס מאוחד על source folds), לא answer-only.

---

## 4. נקודות לדיון עם המנחים (הצעה)

1. **ההחלטה על digit.** הסיבה: הפיצ'ר משווה לספרה שבפתרון הנבדק, כלומר משתמש בתשובה. חלופה שנשארה פתוחה:
   פיצ'רי ספרות מההתפלגות בלבד (מסת ספרות, אנטרופיה בין ספרות, פער בין שתי הספרות המובילות) – לא נבנתה.
2. **Joint כעת:** הפער מול Continuous אינו במודל אלא ב-readout ובבחירת K. הממצא החיובי: על 24 זרמים Joint היררכי
   שומר 43 כש-Continuous נופל ל-39. הממצא השלילי: תלוי בטוהר קבוצה. תיקון readout תוך-קבוצתי הוא צעד אחד ומוגדר.
3. **ללא digit אף שיטה לא עוברת 40.** ייחוס BOCPD+innovation5 40.37. השאלה למנחים: האם המטרה היא לעבור 40 ללא ספרות,
   או לבסס עמידות/אוטומטיות של Joint כתרומה מתודולוגית גם בלי שיא?
4. **Gate.** 935 אזעקות שווא על תשובות נקיות באותו gate לכל השיטות. ה-gate כמעט ללא למידה; BOCPD/innovation
   כמועמד gate לא נבדק.
5. **SLA lane** (Codex): L08 מעל Mind-the-Gap ברמת נקודה, אך לא matched. לא לצטט כעליונות.
6. **מתודולוגיה:** סדרת 407-412 היא 6 וריאנטים ברצף על אותה שאלה – סותר "וריאנט אחד, דיון אחד".
   מומלץ לקבוע יחד אילו מהם נשמרים כהיסטוריה בלבד.

---

## 5. מצב הריפו
- Remote `codex/fusion-independence-atlas-v1` @ `236ac955f`: Step 398 (Claude) + Steps 397-412 (Codex) + merge.
- Ledger-ים: HISTORY Steps 398 [Claude] ו-398 [Codex] (התנגשות מספור – שניהם נשמרו עם תיוג), 399-412 (Codex).
- קבצים בינאריים (npz/npy/log, ~650 MB) נשארו מקומיים ומוחרגים ב-.gitignore.
- כלים: `scripts/run_joint_redundancy_robustness_v1.py` (rosters, folds, `--bootstrap-only`), מסמך
  `docs/experiments/JOINT_REDUNDANCY_ROBUSTNESS_V1.md`.
