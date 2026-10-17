# תוכנית refactor לריפו (2026-10-17)

המטרה: שהריפו ישקף את מה שאנחנו עושים בפועל, כלומר telemetry בזמן teacher forcing, fusion label-free (L-SML ו-IU-PCR) ו-PRM boosting על PRMBench. ההיסטוריה נשארת זמינה אבל לא מסתירה את הקו הפעיל.

## מצב נוכחי (נמדד על `origin/claude/integration-2026-10-17`)

| מה | כמות |
|---|---|
| קבצים במעקב | 16,231 |
| `results/` | 14,074 קבצים, 1.57 GB, 220 תיקיות ניסוי, ו-**1,169 קבצי `.py` בתוך `results/`** |
| קבצים בשורש | 89, מתוכם 64 מסמכי md (28 `HANDOFF_*`, 23 `SPEC_*`) ו-2 PDFs |
| `scripts/` | 488 קבצים בשכבה העליונה, מתוכם 109 `test_*` ו-55 `run_*` |
| `spectral_utils/` | 133 מודולים. 15 מהם מיובאים מקובץ אחד או פחות, ו-4 מאף קובץ |
| מסמכי מצב | `HISTORY.md` 1.4 MB, `PROGRESS.md` 0.6 MB, `Research_Directions.md` 0.3 MB, `CLAUDE.md` 78 KB |
| `papers/` | 281 MB |

הבעיות העיקריות:
1. קוד שיטה חי בתוך `results/<ניסוי>/run.py`. למשל ה-L-SML וה-IU-PCR שמאחורי מספרי PRMBench יושבים ב-`results/monday_cont_baselines_v1/run.py`, ו-`build_bank` ב-`results/monday_external_panel_v1/score.py`. הם מיובאים מתוך results.
2. אין נקודת כניסה אחת לקו הנוכחי, ויש כמה entry points לאותה שיטה (כבר מתועד ב-CLAUDE.md לגבי U-PCR).
3. `CLAUDE.md` ו-`PROGRESS.md` מכילים היסטוריה של חודשים, ו-agent חדש מתחיל מ-78 KB של הוראות שחלקן כבר הוחלפו.
4. ה-checkout מחזיק 40 GB של `.git/lfs` ו-worktrees ישנים, והדיסק בשימוש של 96%.

## עקרונות

- **לא לשנות מספרים.** כל הזזה של קוד חייבת לעבור replay parity מול התוצאות הקפואות. יש כבר כלי כזה: `REPLAY_PARITY.json` עם 410 שורות.
- **להזיז, לא למחוק.** תיקיות ישנות עוברות ל-`archive/`. ה-results הקפואים, ה-seals וה-HTML של המצגות לא משתנים.
- **שלב אחד = PR אחד**, עם בדיקות ירוקות, ונעשה ב-worktree ולא ב-checkout המשותף.

## השלבים

### שלב 0: בסיס (בלי לשנות קוד)
- למזג את שני ה-commits של Gemini ל-integration (ראה `01_BRANCH_CONSOLIDATION_HE.md`).
- להוסיף `scripts/verify_current_line.py`, שמשחזר את מספרי הליבה:
  - N16 L-SML ו-IU-PCR על PRMBench בגרסת thinkfix_mask1
  - PRM boosting ל-5 PRMs, כלומר 5 שורות PRMScore ו-within-AUC
  - Q80 PRMScore
  זה ה-test שכל שלב אחר צריך לעבור.

### שלב 1: חבילת הקו הפעיל
ליצור `spectral_utils/pipeline/` (או חבילה נפרדת בשם `stepfusion/`) עם ארבעה מודולים בלבד, שמועתקים מהמקור הקנוני בלי שינוי לוגי:

| מודול | מקור |
|---|---|
| `bank.py` (N16, `build_bank`) | `results/monday_external_panel_v1/score.py` |
| `fusion.py` (`lsml_continuous`, IU-PCR, MEAN_MOM) | `spectral_utils/fusion_utils.py`, `upcr.py`, `results/monday_cont_baselines_v1/run.py` |
| `decision.py` (Q80, flag rule, gate) | ה-advisor v11 scorer |
| `prmbench_eval.py` (official PRMScore, within-AUC, category breakdown) | `spectral_utils/prmbench.py` והעטיפה של ה-evaluator |

הקבצים הישנים ב-results נשארים ומייבאים מהחבילה, וה-replay parity מוודא שאין שינוי. ב-README של החבילה יהיה כתוב במפורש: "זה ה-entry point הקנוני".

### שלב 2: CLI אחד לקו הפעיל
`python -m stepfusion {extract,fuse,boost,evaluate} --bench prmbench --model ...`, עם configs ב-`configs/current/`, ו-cluster drivers ב-`cluster/current/` (`run_prmbench_prm_telemetry.py`, `run_prmbench_step_voters.py`, ה-teacher-forced driver). את `submit_*.sbatch` לאחד ל-template אחד עם הפרמטרים: wheelhouse, `--container-workdir` (הלקח מ-G7), account ו-QoS.

### שלב 3: ניקוי השורש והמסמכים
- `HANDOFF_*`, `SPEC_*`, `*_CONCLUSION.md`, `STATUS_*` ו-`REVIEW_*` עוברים ל-`docs/archive/2026-08/` עם אינדקס. ה-PDFs עוברים ל-`papers/`.
- `CLAUDE.md` עולה לגרסה קצרה (בערך 10 KB): הקו הפעיל, כללי AIRCC, הכללים הקבועים מ-LESSONS ו-pointers. כל העדכונים המתוארכים שהוחלפו עוברים ל-`docs/decisions/` כ-ledger, עם קובץ לכל החלטה.
- `PROGRESS.md`: להשאיר רק את 30 הימים האחרונים, ולהעביר את השאר ל-`docs/progress_archive/`. `HISTORY.md` נשאר append-only, אבל אפשר לפצל אותו לפי רבעון.
- `Research_Directions.md` מוחלף בטבלה קצרה של כיוונים פעילים וסגורים, לפי `02_RESEARCH_DIRECTIONS_REVIEW_HE.md`. הקובץ הישן עובר לארכיון.

### שלב 4: `scripts/` ו-`spectral_utils/`
- 109 קבצי `scripts/test_*.py` עוברים ל-`tests/` כדי ש-unittest discover ימצא אותם. קודם צריך לבדוק אילו מהם באמת tests ואילו סקריפטים שרק נקראים כך.
- ה-`run_*` של כיוונים סגורים (readout, tail, family, digit, depth, whitebox, RAG, A6 ועוד) עוברים ל-`scripts/archive/<direction>/`. ה-imports נשמרים דרך `sys.path` רק למי שצריך replay.
- `spectral_utils`: מודולים עם 0–1 imports עוברים ל-`spectral_utils/legacy/` עם shim של re-export, כדי שלא יישברו pickles ו-imports ישנים. גם `selectors/` ו-`upcr_pipeline` (legacy) עוברים לשם.

### שלב 5: אחסון
- `results/`: כל תיקייה עם יותר מ-50 MB או עם payload גולמי (npz/pkl) עוברת ל-manifest ו-Drive, בעזרת הכלי שכבר קיים ב-`docs/archives/integration_completion_20261017/`.
- `git lfs prune --verify-remote` (40 GB). **צריך את אישורך.** זו פעולה שלא ניתן להחזיר מקומית, אבל ה-remote מחזיק הכל.
- worktrees ישנים (`a6-s0b`, `ssl-pseudolabel-residual-v1`, `untracked-code-2026-10-12`): להסיר אחרי שמוודאים שהכל נמצא ב-remote או בארכיון ב-Drive. **צריך את אישורך.**

## סדר וגודל

| שלב | סיכון | הערה |
|---|---|---|
| 0 | נמוך | נדרש לפני כל השאר |
| 1 | בינוני | הכי חשוב. מאמתים ב-replay parity |
| 2 | נמוך | נבנה על 1 |
| 3 | נמוך | מסמכים בלבד, אפשר במקביל ל-1 |
| 4 | בינוני | imports ישנים. ה-shims מגינים |
| 5 | גבוה (מחיקה מקומית) | רק עם אישור מפורש |

שלבים 1 ו-3 לא נוגעים באותם קבצים ויכולים לרוץ במקביל ב-worktrees נפרדים.
