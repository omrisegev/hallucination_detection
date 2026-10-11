# Handoff לשיחה הבאה: הכנה לפגישת המנחים ביום חמישי 2026-10-15 + השלמת האיחוד

נכתב 2026-10-10 בסוף השיחה שבה הוכנה והוצגה פגישת 2026-10-08. שלושה נושאים פתוחים:
(א) ההכנה לפגישה ביום חמישי, (ב) השלמת התיעוד, (ג) האיחוד (merge) של ענפי הפיתוח, שעדיין לא הושלם.

## 0. מצב נוכחי

- **הענף בתיקייה הראשית:** `codex/lsml-external-generalization-v1`. ה-commit האחרון שלי: `52bf3f2ee`, שמתעד את הפגישה. אין שינויים ב-tracked files; יש כ-500 קבצים untracked ישנים שלא שייכים לעבודה הזו.
- **לא נדחף לשרת:** דחיפה חסומה בגלל LFS. ראו את הזיכרון `project_consolidation_merge_2026_10_03` ו-`project_git_push_blocked`.
- **דיסק:** 6.5GB פנויים, אחרי מחיקת 5 worktrees נקיים ב-2026-10-10 (Omri הריץ את הפקודה בעצמו; ה-hook `guard_git.py` חוסם מחיקת worktrees על ידי agent). הענפים שלהם נשמרו.
  - **נשארו 8 worktrees:**
    - נשמרים בכוונה: consolidation-fusion-2026-09-22, rescue-loose-files-2026-10-01.
    - יש בהם קבצים שלא נשמרו ב-commit: ssl-pseudolabel-residual-v1 (results 4.8GB, כנראה smoke), readout-quickest-detection-v1 (results 3GB), a6-s0b (master, 207 שינויים, מצב מוזר), decision-rule-v1, depth-feature-fusion-v1, self-generated-step-labels-v1.
  - **רוב המקום:** `.git/lfs` (40.5GB). `git lfs prune --verify-remote` מחכה לאישור של Omri.
- **סוכני רקע מהשיחה הקודמת שנעצרו בגלל מגבלת שימוש ולא הושלמו:**
  - מיקום ב-leaderboard של PRMBench וגרף radar לפי קטגוריות.
  - "Apply review fixes to full deck" (תיקוני המצגת; לא רלוונטי יותר, הפגישה מציגה HTML).

## א. ההכנה לפגישה ביום חמישי 2026-10-15

**התוכנית המלאה:** [docs/meetings/advisor_meeting_2026-10-08/README_HE.md](meetings/advisor_meeting_2026-10-08/README_HE.md). החלטת המנחים: מיקוד ב-PRMBench; ProcessBench בעדיפות נמוכה (אין שם יתרון משמעותי על הממוצע הפשוט, והפער מהמפוקח קטן). להציג דף HTML ממוקד וקטן יותר מהדף של 2026-10-08. לא מצגת.

**המיקוד: PRMBench בלבד.** ב-ProcessBench אין לנו יתרון על הממוצע הפשוט (N16: ‏0.414 מול 0.396–0.408 ל-pipeline), ו-LR המפוקח כמעט לא מוסיף שם (0.402), כך שאין הרבה לאן להתקדם. לא מכינים ניתוחים חדשים על ProcessBench. ב-PRMBench, LR מוסיף ‎+0.03 עד ‎+0.07 AUC, כלומר יש מרחב.

**שני הדברים הראשונים (לפני הסעיפים):**
- **0א. בדיקת בלוק ה-think ב-teacher forcing.** ייתכן שה-template לא מוגדר נכון: הקלט חסר את `<think></think>` הריק, ולכן המודל "מופתע" מהטוקן הראשון (34–36 nats; שיא בצעד 0 ב-90–93% מתשובות ProcessBench GSM8K/MATH). לבדוק את ה-template בקוד, למדוד את ההשפעה על PRMBench, לבדוק על CPU עם מיסוך הטוקנים הראשונים. הרצה מחדש על GPU רק באישור. אם ההשפעה גדולה, כל מספרי PRMBench מחושבים מחדש, ולכן זה ראשון. הפירוט בתוכנית, סעיף 0א.
- **0ב. ללמוד את PRMBench:** המאמר, האתר וה-GitHub; איך נבנה ה-dataset ואיך נשתלו השגיאות; מה בודקת כל קטגוריה; איך מוגדרות שגיאות redundancy ו-circular. **השאלה העיקרית:** האם המודלים החזקים נכשלים בהן בכוונה (כך נבנתה הקטגוריה)? זה קובע אם היתרון שלנו שם, שהוא כנראה לב הטענה, הוא ממצא אמיתי. הפירוט בתוכנית, סעיף 0ב.

**הסעיפים שצריך להכין:**
1. **ביצועים:** within-AUC ו-PRMScore רשמי. הפיזור בין שיטות, בנקים, כלל סימון (flags), gate, יחידת הצבעה ויחידת מכסה. פירוק PRMScore לפי סוג שגיאה ולפי Overall / Simplicity / Soundness / Sensitivity כמו ב-https://prmbench.github.io/.
2. **מבחני הכללה:** Socratic-PRMBench, Hard2Verify, תשובות עצמיות; בחירת בנק על חצי ובדיקה על חצי (source group); leave-one-error-type-out.
3. **קשרים סטטיסטיים בין הפיצ'רים:** קורלציות בתוך צעדים שגויים ובתוך צעדים תקינים בנפרד (בדיקה של הנחת אי-התלות המותנית); אילו ערוצים קורסים באילו סוגי שגיאות; אילו פיצ'רים טובים.
4. **teacher forcing:** (בלוק ה-think: ראו 0א) ההשוואה לתשובות עצמיות; בעיית בלוק ה-think הריק ([סקירת Codex](research_notes/QWEN3_TEACHER_FORCING_PREFIX_REVIEW_20261008_HE.md), HISTORY Step 450, הזיכרון `project_tf_missing_think_block_2026_10_08`). בדיקת CPU עם מיסוך מותרת; הרצה מחדש על GPU דורשת אישור.
5. **ניתוח מול LR המפוקח:** איפה כל אחד צודק או טועה (לפי סוג שגיאה, מיקום הצעד, אורך); המשקלות שלנו לעומת המשקלות של LR על אותו סולם.
6. **הצעות שלי שעוד לא אושרו:** AUC אחרי ניכוי מיקום הצעד, ושורת בסיס של המיקום בכל טבלה (מספר הצעד לבדו 0.662, בסוג confidence ‏0.868); רווחי סמך מזווגים לפי source group; redundancy ו-circular logic כמסר מרכזי אם זה מחזיק אחרי ניכוי המיקום; השגיאות ה"שוטפות" כגבול השיטה; N13 ליד N16 (מה ערוצי הספרות מוסיפים).

**נקודות פתיחה בנתונים (לבדוק לפני שמריצים משהו חדש):**
- `results/monday_cont_baselines_v1/ABSOLUTE.csv`: עמודה `prmb_prmscore_top1` לכל בנק N ולכל שיטה רציפה.
- `results/monday_candidate_panel_v1/run_20261004/PRM_CATEGORY_METRICS.csv`: within-AUC ו-PRMScore לפי קטגוריה, כלל החלטה ו-readout. שימו לב: כאן הבנקים נקראים B13 וכו', שם ישן. צריך לבדוק את ההתאמה ל-N13 לפני שמשווים.
- `results/prmscore_adapter_v1/METRICS.json`: ה-adapter של PRMScore הרשמי.
- `results/monday_ds_signed_v1/TYPES_SUMMARY_DS.csv`: סוגי השגיאות מול PRM-7B (within-AUC, 80 קונפיגורציות).
- `results/monday_lr_reference_v1`, `results/monday_lr_weights_v1`: LR והמשקלות שלו.
- HISTORY Steps 447–449 (Codex): oracles של ניגודים בתוך קבוצות. רוב הפער המפוקח נמצא בתוך הקבוצות.
- הקוד שמייצר את הדף הקודם: `docs/meetings/advisor_meeting_2026-10-08/findings_page/gen.py` (אפשר להשתמש בו כתבנית).

**כללים:** כל המספרים על PRMBench ו-ProcessBench הם development data. N16 נבחר בדיעבד. להפריד בין within-AUC, any-error hit ו-PRMScore רשמי; להשוות גישה לכיול סף באופן הוגן (ה-PRM משתמש בסף 0.5 משלו). לא להפוך הבדל לא מובהק לניצחון, ולא להפוך הבדל קטן לשקילות. ניסוי חדש: לשאול את Omri לפני שמריצים, בגבולות ההרשאות ב-CLAUDE.md.

## ב. התיעוד: מה מתועד ואיפה

| מה נעשה בשיחה | איפה מתועד |
|---|---|
| דף הממצאים: מבנה, שורות ועמודות ממוצע, הדגשת N16, שלושה דפי deep dive | HISTORY Step 451; README הפגישה; הקוד ב-`findings_page/` |
| מה הוצג, מה הוחלט, תוכנית לחמישי | README הפגישה; PROGRESS (2026-10-08, 2026-10-10); CLAUDE.md (החלטה 2026-10-08); Research_Directions (2026-10-08) |
| הסברים: `_cc`/`_ga`, סינון DS, MoM, HEM, SLA, הבנקים | GLOSSARY, הסעיף "Step-level fusion terms (2026-10-08 advisor meeting)" (נוצר מ-`spectral_utils/glossary.py` דרך `scripts/build_glossary.py --allow-gaps`; לא לערוך את GLOSSARY.md ידנית) |
| סיכום הממצאים מול המתחרים ומול הממוצע | README הפגישה, הסעיף "סיכום הממצאים" |
| ניקוי worktrees ומצב הדיסק | PROGRESS 2026-10-10; ה-handoff הזה |
| תיעוד ההכנה של Codex ושלי שעוד לא היה ב-commit (prep, handoffs, reviews, `results/monday_*`) | commit `52bf3f2ee` |

**לא תועד בכוונה:** שקופיות המצגת. הן היו ב-artifacts, והפגישה עברה ל-HTML.

## ג. האיחוד (consolidation merge): לא הושלם

**מה קרה עד עכשיו:**
- 2026-10-03: הענף המקומי `claude/consolidation-2026-10-01` (tip `54a2c2a8f`) איחד את הסדרה של Codex, את ה-snapshot של התיקייה הראשית ואת ענפי Claude. כל ענפי ה-claude (ssl-pseudolabel-residual-v1, decision-rule-v1, self-generated-step-labels-v1, estimator-provenance, readout-quickest-detection-v1, depth-feature-fusion-v1) וגם `codex/advisor-prep-2026-10-05` כבר כלולים בו (0 commits חסרים).
- בדיקת ההשלמה של Codex ([CONSOLIDATION_COMPLETION_REVIEW_20261003_HE.md](reviews/CONSOLIDATION_COMPLETION_REVIEW_20261003_HE.md)) מצאה שהאיחוד **לא שלם**:
  1. שמונה קטעי HISTORY חסרים (Steps 330, 332, 334–339 מהמקור `48f9ef291`). הטקסט המלא שלהם מוכן ב-`docs/reviews/consolidation_20261003/MISSING_HISTORY_BLOCKS_FROM_48f9ef291.md` יחד עם מניפסט.
  2. הטענה על שקילות בתשובות עצמיות עדיין לא תוקנה ב-handoff המחקרי (0.7566 לעומת 0.7630; "שקילות" לא הוכחה).
  3. השוואת הבדיקות לא מוכיחה שאין רגרסיות (1,654 עברו, 250 נכשלו, 57 שגיאות, 44 דילוגים).

**מה עוד לא נכנס לענף האיחוד (נבדק 2026-10-10):**

| ענף | commits שחסרים באיחוד | אחרון |
|---|---|---|
| `codex/monday-fusion-analysis-20261003` | 14 | 2026-10-04 |
| `codex/lsml-external-generalization-v1` (הענף הנוכחי; כולל את תיעוד הפגישה ואת ה-competitor outputs) | 5 | 2026-10-08 |
| `codex/consolidation-completion-review-2026-10-03` | 1 | 2026-10-03 |
| `origin/main` | 3 | — |

`claude/consolidation-2026-10-01` לא נדחף לשרת ולא נכנס ל-`origin/main`. אין `main` מקומי.

**הצעדים להשלמה (לאשר כל צעד עם Omri):**
1. למזג לענף האיחוד את ארבעת הענפים בטבלה.
2. לשלב את שמונת קטעי HISTORY החסרים, עם תגית הקו, בלי לשנות מספרים.
3. לתקן את טענת השקילות. לשמור את הפלט המקורי כראיה מתוארכת.
4. להשוות כישלונות בדיקות מול כל ענף מקור, לפי שם הבדיקה.
5. לאמת לפי מספר בלוקים ש-HISTORY ו-PROGRESS שלמים משני הצדדים (ראו commit `cd423ab` כדוגמה).
6. לפתור את חסימת ה-LFS בדחיפה, לדחוף, ולפתוח PR ל-main.
7. רק אחרי כל זה: לנקות את שאר ה-worktrees, ולהריץ `git lfs prune` באישור.

**כללי מיזוג מחייבים:**
- HISTORY ו-PROGRESS: מיזוג באיחוד (union), אף פעם לא `--ours` או `--theirs`.
- התנגשויות במספרי Steps: לתייג את הקו בכותרת, לא למספר מחדש.
- merge, פתרון וה-commit בפקודת Bash אחת (יש sessions במקביל; הזיכרון `project_concurrent_sessions`).
- worktree remove ו-prune רק בטרמינל של Omri.
- לשמור על סיומות השורות של כל קובץ: HISTORY רובו CRLF, PROGRESS ו-CLAUDE.md רובם LF. כתיבה ב-Python במצב טקסט על Windows הופכת הכול ל-CRLF.

## פרומפט מוכן להדבקה

```
המשך מ-docs/HANDOFF_prmbench_meeting_and_consolidation_2026-10-10.md.
קודם /session-start. אחר כך:
1) הכנה לפגישת המנחים ביום חמישי 2026-10-15. המיקוד הוא PRMBench בלבד, לא ProcessBench. התוכנית: docs/meetings/advisor_meeting_2026-10-08/README_HE.md.
   ראשון: סעיף 0א, בדיקת בלוק ה-think ב-teacher forcing (template, השפעה על PRMBench, בדיקת CPU; GPU רק באישור שלי).
   במקביל: סעיף 0ב, לימוד PRMBench (המאמר, ה-dataset, הקטגוריות, והאם redundancy/circular קשים למודלים חזקים בכוונה).
   אחר כך סעיפים 1–6: מיפוי הנתונים הקיימים (סעיף א, "נקודות פתיחה"), והצעה לסדר עבודה ומה דורש הרצה חדשה, לפני שמריצים משהו.
2) במקביל: תוכנית להשלמת האיחוד לפי סעיף ג, צעד אחר צעד, לאישור שלי.
התוצר לפגישה: דף HTML ממוקד, קטן מהדף של 2026-10-08.
```
