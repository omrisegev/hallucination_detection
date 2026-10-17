# בדיקת האינטגרציה של 17.10.2026

## היקף ומצב Git

- ענף: `claude/integration-2026-10-17`.
- Commit מקומי וב-GitHub: `65bcf88310d7ee98a2b4b314bd0602e7db064b9c` (בדיקת `git ls-remote --heads origin`).
- השוויתי את עצי הקבצים של 11 מקורות: v17/v18, v19/external, advisor v11, error count, PRM adapters, PRM generalization results, agent CPU delivery, agent evaluation, PRM voters, checkout Codex הישן, cluster-contrast.
- לא חסר ב-HEAD אף נתיב מהמקורות האלה. הבדיקה מתעדת גם קבצים שתוכנם השתנה; אין בכך אישור שכל שינוי סמנטי נכון. ראו SOURCE_COVERAGE.json.
- קוד, CSV/JSON, מצגות v17/v18/v19 וגרסת send, וניסויי S3/S5/S6 שביצעתי כלולים באינטגרציה. הגרפים הנפרדים מבקרת L-SML מוחרגים מ-Git.
- שני השינויים המקומיים ב-bank11 הם הבדלי סוף שורה בלבד לפי git diff --ignore-space-at-eol. לא שיניתי אותם.

## ממצאי review

### 1. [P1] המסקנה ש-LR אינו צריך חיסור אינה נובעת מהבקרה שבוצעה

`results/lr_gap_explanation_draft_2026-10-15/analysis/handoff_checks.py:32-34` הופך כל עמודה לפי ההבדל בין ממוצעי המחלקות בתוויות האימון, ורק אז כופה משקלים לא-שליליים. לכן המשקל בפיצ׳רים המקוריים הוא sg*w ויכול להיות שלילי. ב-feature_weights_N56.csv יש חמישה משקלי NNLR שליליים, עם 0.29575 מסה מוחלטת מתוך 1.0: hist_entropy_stft_high_series, hist_spilled_rolling_min, hist_energy_series, hist_energy_rolling_min, hist_energy_sw_var_series.

זה אינו מוכיח שה-AUC שדווח שגוי, ואין כאן ראיה לדליפת תוויות מבחן. אך ההשוואה אינה מבודדת את תרומת המשקלים השליליים ביחס לאותן כניסות של השיטות ללא תוויות. בנוסף NNLR משתמש ב-l2=1e-4 על הפסד ממוצע, בעוד decomp.lr משתמש ב-sklearn C=10; גם הרגולריזציה אינה מותאמת. יש לצמצם את הניסוח ב-EXPLANATION_HE.md:21, EXPERIMENTS_SUMMARY.md וב-PROGRESS.md:6: מדובר ב-positive weights לאחר אוריינטציה מפוקחת. בחירת הפיצ׳רים ב-NNLR עצמה מפוקחת, ולכן גם אין לכנותה label-free pruning.

### 2. [P1] הוראות ה-PRM voters מפנות לענף שאינו מכיל תלות נדרשת

`docs/experiments/PRMBENCH_STEP_VOTERS_HANDOFF.md:42` מנחה לפתוח Track A מהענף claude/prm-voters-processbench-pilot, שראשו 17975f92. באותו commit אין spectral_utils/error_count.py, בעוד scripts/run_prm_voters_prmbench.py מייבא ממנו fit_mom ו-posterior. לכן שלב הניתוח בסעיף 7 ייכשל ב-ModuleNotFoundError ב-checkout המתואר. בענף האינטגרציה התלות כן קיימת, אך ה-handoff לא משתמש בו. יש להפנות ל-commit שלם או לשמר במפורש את התלות ב-Track A.

### 3. [P1] שמות ה-adapters ב-handoff אינם תואמים לממשק שהוקפא

`docs/experiments/PRMBENCH_STEP_VOTERS_HANDOFF.md:94` (וגם פקודות ה-full) משתמש ב-rlhflow, pure, mathshepherd. הקוד ב-fba8e8d9 ובאינטגרציה דורש rlhflow_prm, pure_prm, math_shepherd. פקודת CPU בלבד עם --adapter pure יצאה 2 ב-argparse עם invalid choice, לפני יצירת תיקיית פלט ולפני טעינת מודל. יש לתקן גם את שמות תיקיות הפלט/ההפניות, או למפות את הכינויים במפורש. לא הורצה שום פקודת GPU.

### 4. [P2] גם לסקריפט הבקרה שלי נשארה תלות זמנית שמונעת שחזור נייד

`results/remote_claims_replay_2026-10-15/replay.py:190` קורא SLIDE12_RESULTS_qwen_prm_7b.csv מתוך /tmp/advisor_remote_review_20261015/... . אותו CSV כבר קיים ב-results/advisor_page_v19_2026-10-15/sources/eval_slide12 שב-Git. יש להפנות לנתיב מה-repository או לפרמטר קלט ולבדוק את כל הקלטים לפני כתיבת פלט. בנוסף הסקריפט הנוכחי כותב את CSV/JSON מחדש לתוך תיקיית התוצאות הקיימת. כדי לשחזר בלי לדרוס את העדות הקפואה צריך destination חדש. לא הרצתי אותו על החבילה הקיימת.

### 5. [P2] אימות חבילת ה-reading אינו מותאם ל-checkout המשולב

פקודת ה-README, `scripts/package_label_free_error_count_v1_for_git.py --verify-delivery ...`, נכשלת על Experiments_Report.md, שנחתם בגרסת הפרסום ואחר כך עודכן באינטגרציה. המניפסט של הפרסום שומר hash של מסמכי root חיים; לכן אינו יכול לשמש כבדיקת שלמות של ה-checkout המשולב ללא הפרדה בין קבצי החבילה לבין מסמכי הייחוס ב-commit המקורי. יש לתעד verification מול ה-commit המקורי או לספק מניפסט אינטגרציה נוסף, בלי לשנות את המניפסט המדעי הקפוא.

ה-verifier הישן scripts/verify_label_free_error_count_delivery.py נכשל גם הוא על README.md, אך זה כבר היה כך ב-6a0b8700: README הפרסום החליף את ה-README המקורי השמור ב-README_ANALYSIS_SNAPSHOT.md. זה אינו נזק חדש של המיזוג. validate_reference() של wrapper השחזור עבר בהצלחה.

## מה להשלים בענן

### GitHub

שישה קבצים בתיקייה results/remote_claims_replay_2026-10-15/plots (שלושה גרפים, SVG+PNG; 827,112 bytes) לא נכללו בגלל *.svg / *.png ב-.gitignore. אלה הגרפים שביקשת לראות על clustering מול spectral weights ועל ההרחבה ל-5/6 פיצ׳רים. plot_replay.py וה-CSV שלהם כן פורסמו. כדאי לשמר גם את הגרפים המדויקים ואת hashes שלהם.

### Drive

מצאתי 100 קובצי NPZ מחוץ ל-Git, סה״כ 136,533,227 bytes, בארבע חבילות שלי:

- clustered_iu_prmbench_v1_2026-10-12: 12 קבצים, 9,451,164 bytes.
- family_127_prmbench_v1_2026-10-12: 16 קבצים, 16,807,356 bytes.
- signal_gallery_extensions_2026-10-12: שני קבצים, 1,389,736 bytes.
- label_free_error_count_v1_2026-10-12: 70 קבצים, 108,884,971 bytes.

לא מצאתי ארכיון בשם החבילות האלה בגיבויי consolidated_results/local_backup_2026-10-12/13/14 או ברשימת cluster_results. זו אינה סריקת hashes של כל ארכיון היסטורי ב-Drive; נכון לסמן אותן כטעונות גיבוי/הוכחת כיסוי. CLOUD_CANDIDATES.json כולל נתיב, גודל ו-SHA256 של כל קובץ מוצע.

גם advisor_completion_2026-10-11 כולל 155 קבצים מוחרגים (2,133,718,789 bytes). יש להוכיח גיבוי של הבנקים, תחזיות OOF והתוצאות הנגזרות המדויקות; גיבוי הטלמטריה המקורית לבדו אינו שימור של כל החבילה הקפואה. בקו agent n16 של קלוד יש עוד שמונה קבצים מוחרגים, 18,477,688 bytes; הטלמטריה הגולמית ב-Drive, אך לא אומת כאן גיבוי של תוצרים נגזרים אלה.

שני ה-tar הלא-מנוהלים שנשארו ב-root audit, prm_boosting_v1.tar ו-thinkfix_rescore_v1.tar, כבר קיימים ב-Drive תחת local_backup_2026-10-12/results/*.tar.partx000. גודל ו-SHA256 מקומיים זהים ל-metadata של Drive שנקרא כעת. אין צורך לדחוף אותם שוב או להכניסם ל-Git.

## תוספת: שינויים מקומיים ב-worktrees ישנים

בדקתי גם את מצב 23 ה-worktrees הנוספים הרשומים. בחמישה מהם יש **21 קבצים מנוהלים עם שינוי תוכן מקומי** (מעבר ל-CRLF/LF). הם אינם זהים לקובץ המקביל באינטגרציה או בראש ענף המקור ב-GitHub, וגם תוספות מהותיות אינן מופיעות בראש אותו ענף. אין בכך הוכחה שלא נשמרו בתוך ארכיון Drive ישן או בענף אחר.

- `.worktrees/fusion-independence-atlas-v1`: שני מסמכי handoff עם תוצאות Innovation5/Step413 מספטמבר.
- `.worktrees/reconstruction-benchmark-v1`: 12 קבצים, ובהם עדכוני advisor מאוגוסט, HTML/SVG ומסמכי סטטוס.
- `.worktrees/su-pcr-24cell-audit-v1`: שינוי קוד ב-spectral_utils/dependency_fusion.py המוסיף offdiag_shrinkage ל-sparse_upcr_fit. הפרמטר אינו קיים בגרסת האינטגרציה.
- `local_cache/worktrees/ciw_multipop_benchmark_v1`: שלושה קובצי קוד/בדיקה עם consensus_response_gate וה-readout התואם. הפונקציה אינה קיימת בגרסת האינטגרציה.
- `local_cache/worktrees/token_local_fusion_optimization_v1`: שלושה מסמכי מחקר/handoff עם תוספות Phase1/Phase2 מאוגוסט.

כדי לשמר אין להעתיק את קובצי HISTORY/PROGRESS הישנים בשלמותם על המסמכים החדשים. שמרתי מחוץ לריפו **חמישה patches מלאים**, עם commit הבסיס ורשימת hashes ב-WORKTREE_DIRTY_INVENTORY.json. צריך לוודא כיסוי ארכיוני קיים או להעלות snapshot ממוספר של השינויים האלה. השימור אינו החלטה להחזיר את הכיוונים ההיסטוריים למחקר הפעיל.

### קבצים לא מנוהלים נוספים בקווים ההיסטוריים

סריקת קבצים לא מנוהלים (שאינם gitignored) ב-23 worktrees מצאה קבצים ב-14 מהם, סה״כ 7,964 קבצים / 2,227,076,187 bytes. בהשוואת תוכן לעצים שנבדקו: 3 זהים לתוכן באינטגרציה, 11 נוספים נמצאים בראשי ענפי מקור שנבדקו, 7,910 לא התאימו לאותם עצים, ו-40 קבצים מעל 5 MB לא חושב להם hash בסריקה זו. הרשימה המלאה ב-WORKTREE_UNTRACKED_GIT_COVERAGE.json.

זוהי **רשימת התאמה לארכיונים, לא רשימת העלאה אוטומטית**: חלק גדול שייך לאוגוסט/ספטמבר וייתכן שכבר נשמר בארכיוני Drive ישנים. לא בדקתי את תוכן כל ה-tar ההיסטוריים ולא סרקתי את כל ה-payloads המוחרגים בכל worktree. יש להשוות את הרשימה למניפסטי הגיבוי לפני ניקוי/מחיקה או הכרזה על שימור מלא.

ניסיון מעבר על כל היסטוריית refs נכשל על אובייקט commit מקומי חסר `3f048e67e57ea30afb5026b224e6e5fd888ff599` (אב של `f7e62c9a`). הריפו אינו shallow. לכן בדיקת היסטוריה מלאה אינה מאושרת; בדיקות העצים, ה-HEAD והבדיקות המקומיות המתוארות בדוח כן הושלמו. לא ביצעתי תיקון Git או שינויים ב-refs.

## בדיקות שבוצעו

- 110 בדיקות unittest עברו; בדיקת historical PRM replay אחת דולגה כי PRM_V1_METADATA לא הוגדר. הוגדר PRM_TOKENIZER_DIR=/private/tmp/prm-tokenizers לאחר שבדיקה ראשונית לא מצאה את מטמון הטוקנייזרים ברירת המחדל. לא נדרשה התקנה או הורדה.
- הפירוט: PRM 17 PASS + 1 SKIP; error-count 62 PASS; agent inputs 20 PASS; clustered IU 5 PASS; first-token mask 6 PASS.
- scripts/smoke_prmbench_step_voters.py עבר עם tiny random model ו-tokenizer מקומי על שלוש דוגמאות סינתטיות, כולל resume/sharding/SIGTERM. זו בדיקת מימוש, לא תוצאת benchmark.
- 616 קובצי Python שנוספו/השתנו ביחס ל-b7041de4 עברו compile בזיכרון ללא שגיאת תחביר. אין בכך אישור מספרי/לוגי לכל הקוד.
- בדיקת frozen error-count reference עברה; שני verifier checks מתועדים לעיל נכשלו, ואין להסיק מכך ש-CSV המדעי השתנה.
- לא בוצע inference חדש על benchmark או GPU/AIRCC, חישוב מחדש של כל המחקרים, או אימות עצמאי חדש של ארבעת ה-PRMs האחרים.
- לא שיניתי קוד או מצגות, לא עשיתי commit/push/upload. הראיות לדוח הזה נשמרו רק ב-/tmp/integration_review_2026-10-17.
