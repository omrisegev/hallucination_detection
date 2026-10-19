# ספרות סביב PRMBench ו-PRM boosting: סיכום (2026-10-18/19)

המסמך מרכז את הסקירות שנעשו בשיחה של 18–19.10, כדי שיהיה אפשר להציג אותן למנחים.
השקף עצמו: `SLIDE_HE.html`. ה-deck כולל 2 שקופיות; כדי לפתוח אחת מהן מוסיפים `?slide=1` או `?slide=2`.
רשימת המאמרים המלאה: `CITING_PAPERS_2026.csv`.

## 1. מאמרים מ-2026 שמצטטים את PRMBench

- **מקור הנתונים:** Google Scholar חסם את החיפוש עם CAPTCHA, ולכן הרשימה נלקחה מ-Semantic Scholar API, ב-18.10.2026.
  - נמצאו 134 מאמרים מצטטים, 64 מהם מ-2026.
  - ספירות הציטוטים של Semantic Scholar נמוכות בדרך כלל מאלה של Scholar.
- **המצוטטים ביותר** הם בעיקר בנצ'מרקים ל-agents, multimodal ו-RL, ולכן הקשר שלהם אלינו חלש:
  - AgentProcessBench: 36 ציטוטים.
  - ToolPRMBench: 14.
  - Verifiable PRMs: 14.
  - Save the Good Prefix (VPPO): 11.
- **הרלוונטיים ישירות להשוואה:**
  - **PRISM** (arXiv 2606.09078): fine-tuning של Qwen-PRM-7B על pairwise comparisons. ‏PRMScore עולה מ-65.5 ל-68.0.
  - **CROP** (arXiv 2605.30085): conformal certification של prefix נקי. ה-risk proxy הוא PRM + trace features, עם supervised LR. על PRMBench ה-pooled AUROC עולה מ-0.801 ל-0.834.
  - **SLIDER** (arXiv 2610.00571): Partial Information Decomposition על redundancy. ‏61.91% step accuracy.
- **רלוונטיים כרקע:**
  - **QD stress tests** (arXiv 2608.08008): ניצול של padding ב-Qwen-PRM-7B כשמצרפים את ציוני הצעדים בממוצע (mean pooling).
  - **MedPRMBench** (arXiv 2604.17282) ו-**LSR-Ben** (arXiv 2605.01203): בנצ'מרקים מחוץ למתמטיקה.
  - **BetaPRM** (arXiv 2605.15529) ו-**Distributional PRM** (arXiv 2605.06785): אמינות ו-calibration של PRM.
  - **LLaDA-PRM** (arXiv 2609.22700): מעריך דו-כיווני.
  - **CPMI** (arXiv 2604.10660): תוויות לצעדים מתוך ההסתברויות של המודל.

## 2. השוואה של המטריקות מול המאמרים

החישוב עצמו נמצא ב-`results/prmbench_paper_metric_parity_v1/README.md`, עם 4 גרפים, ושם הפרטים המלאים.
- **PRISM:**
  - המטריקה זהה לשלנו, וה-baseline שלהם משוחזר אצלנו.
  - אם רק מזיזים את ה-threshold של ה-PRM המקורי, הוא מגיע לנקודת העבודה של PRISM: ‏FPR ‏47.7 ו-PRMScore ‏68.26.
  - באותה נקודת עבודה, ‏CHAIN לא טוב יותר מה-PRM אחרי הזזת ה-threshold. רק LSML_CONT, שה-channels שלו נבחרו עם labels, גבוה יותר.
- **CROP:**
  - המטריקה שונה: AUROC על כל הצעדים יחד (pooled), וה-calibration נעשה עם labels.
  - על הפרוטוקול שלהם, LSML_CONT מגיע ל-0.802 ושומר 25.1% מהצעדים, מול 0.834 ו-21.8% של Trace+PRM.
- **SLIDER:** ההשוואה משוערת בלבד, כי הפרוטוקול שלהם לא מוגדר במלואו.
- **Optimal Aggregation** (arXiv 2510.13918):
  - אי אפשר לחשב את המטריקה שלהם על הנתונים שלנו, כי היא דורשת Best-of-N, כלומר כמה דגימות לכל שאלה.
  - בבדיקה חלופית, הציון המשולב שלנו מדרג גרוע תשובות שונות זו מול זו, כי הוא מנורמל בתוך כל תשובה.

## 3. Optimal Aggregation of LLM and PRM Signals: במה הוא שונה מאיתנו

| | הם (arXiv 2510.13918) | אנחנו |
|---|---|---|
| משימה | בחירת תשובה מתוך N דגימות (Best-of-N) | איתור צעד שגוי בתשובה אחת |
| אותות | תדירות התשובה (vote) + ציון PRM + אמינות ה-LLM | ציון PRM לכל צעד + telemetry מה-backbone |
| משקלים | נלמדים מ-labeled calibration set (KDE לפי נכון ולא נכון) | L-SML, ללא labels, לכל תשובה בנפרד |
| הערכה | MATH, ‏5 LLMs, ‏7 PRMs | PRMBench ו-Socratic |

- **הם לא unsupervised.** בפרק Limitations הם מציינים unsupervised calibration כעבודה עתידית.
- **ממצא משותף:** PRM לבד אינו האופטימום, והמשקלים האופטימליים תלויים בצירוף של LLM ו-PRM.
- **לא אומת:** קבלת המאמר ל-ICLR 2026. אתר OpenReview דרש challenge.

## 4. כמה PRMs הם נושא חם (דוח של subagent, 18.10)

**חשוב:** הממצאים בסעיף הזה לא אומתו ישירות. כל טענה מבוססת על המקור שמצוין לצידה.

- **הכרעה:** PRMs הם נושא פעיל בגודל בינוני, אבל הם כבר לא הכיוון המרכזי של אימון מודלי reasoning.
  - בכנסים הגדולים, מספר המאמרים המתקבלים עם PRM בכותרת גדל פי 3 בערך:
    - NeurIPS: מ-3 מתוך 4,538 ל-9 מתוך 5,858.
    - ICLR: מ-2 ל-10.
    - ICML: מ-3 ל-11.
  - זה עדיין רק כ-0.15% מכל המאמרים.
  - בתקופה הזו, כותרות RLVR עלו מ-0 לפי 2–3 ממספר כותרות ה-PRM.
- **מעבדות גדולות:**
  - DeepSeek-R1 (Nature 2025) כתבו במפורש ש-PRM לא עבד אצלם, בגלל reward hacking ו-compute.
  - לא נמצא פרסום של OpenAI, ‏DeepMind או Qwen על שימוש ב-PRM באימון.
- **כנסי ACL/EMNLP:** כאן מתפרסמים הכי הרבה מאמרי PRM, בערך 14–15 בכל כנס. Hard2Verify התקבל ל-ACL 2026.
- **רלוונטי במיוחד אלינו:**
  - *Optimal Aggregation* (ICLR 2026, poster): ‏majority voting מנצח לפעמים PRM, והפתרון הוא fusion משוקלל.
  - *Out of Distribution, Out of Luck* (EACL 2026, aclanthology.org/2026.eacl-short.31): ‏PRMs נכשלים על traces של reasoning models, ובערך 80% מה-features שלהם מגיבים ל-formatting.
  - *Reward Under Attack* (ICML 2026, openreview.net/forum?id=phHnze3Nfm): אפשר לעשות hacking ל-PRMs.
- **הנישה שלנו:** לא נמצא בכנסים המובילים מאמר שעושה label-free step-error detection משולב עם PRM.
- **מה reviewers יבקשו:**
  - baselines: majority voting, ‏thinking verifier או LLM-as-judge, ו-supervised readout.
  - traces של long-CoT.
- **כנסים מתאימים:** ACL/EMNLP (main או Findings), ‏ICLR/ICML אם יש תוצאה תיאורטית או של fusion, TMLR ו-COLM.

## 5. מה עשינו בעבר כשבדקנו threshold במקום Q80

- **הבדיקה:** "budget curves" מ-HISTORY Step 291, 11–12.10, על cell אחד בלבד: PRMBench עם Qwen-PRM-7B.
  - הקוד ב-`results/advisor_completion_2026-10-11/controls/run_controls.py`, והמספרים ב-`DIAGNOSTIC_CEILINGS.json`.
- **מה נבדק:**
  - sweep של threshold גלובלי אחד, בלי labels: אחוז הסימון נע בין 0% ל-50%, ב-501 ערכים.
  - "הנקודה הטובה ביותר", שנבחרה עם labels על אותם נתונים. זה oracle, ולא שיטה.
  - שתי תקרות: מספר השגיאות האמיתי לכל תשובה, ודירוג מושלם.
  - **לא נבדק** threshold חופשי לכל תשובה או לכל cell.
- **תוצאה:** הזזת ה-threshold משפרת את השיטה שלנו רק ב-+0.3 (מ-65.38 ל-65.69), ואת ה-PRM ב-+2.9 (מ-65.46 ל-68.31).

## 6. How Far Can Unsupervised RLVR Scale LLM Training? (arXiv 2603.08660, ICLR 2026, המשך של TTRL)

- **מה הם מראים:**
  - הם מחלקים את אותות ה-reward ללא labels לשני סוגים:
    - **intrinsic:** אותות certainty (entropy, ‏self-certainty, הסתברות) ו-majority vote. אלה בדיוק סוגי האותות שבבנק ה-telemetry שלנו.
    - **external:** אותות שנשענים על נתונים לא מתויגים, או על כך שבדיקה של תשובה קלה מיצירתה.
  - כל ה-intrinsic rewards רק "מחדדים" את מה שהמודל כבר מאמין בו.
    - כשה-confidence מתיישב עם ה-correctness, זה עובד.
    - כשלא, השיטה נכשלת: עלייה בהתחלה ואחר כך collapse.
    - לדבריהם, מועד ה-collapse נקבע לפי ה-prior של המודל, ולא לפי hyperparameters.
  - שיטות ה-entropy מגיעות ל-entropy נמוכה דרך חזרה על טקסט (repetition), ולא דרך תשובות נכונות.
  - הם מציעים **Model Collapse Step**: מדד שמנבא מראש אם RL ללא labels יעזור. אבל המדד דורש להריץ אימון RL.
- **איפה הם ביחס אלינו:**
  - **המטרה שונה:** הם מאמנים policy ב-RL. אנחנו לא מאמנים, אלא מאתרים צעד שגוי בתשובה קיימת.
  - **האותות דומים:** ה-telemetry שלנו הוא אותם אותות certainty. אבל אצלנו המודל שמודד אותם הוא ה-PRM, שקורא תשובה של מודל אחר (teacher-forced), ולא ה-policy שקורא את הפלט של עצמו.
  - **ה-PRM שלנו אומן עם labels.** לכן בטקסונומיה שלהם הוא לא intrinsic, וה-boosting שלנו הוא שילוב של verifier מפוקח עם אותות intrinsic.
- **איך זה משפיע על העבודה שלנו:**
  1. **זה מסביר את הכישלון ב-Socratic.**
     - ב-PRMBench, ‏energy_level מתיישב עם ה-correctness כמעט כמו ה-PRM: within-AUC ‏0.799 מול 0.801. שם ה-fusion ללא labels עוזר.
     - ב-Socratic הוא מתיישב פחות, 0.683 מול 0.824, וה-fusion פוגע.
     - מקור המספרים: `results/prm_boost_socratic_diagnosis_v1/README.md`.
     - **המשמעות:** זה כנראה לא באג במתכון, אלא אותה "תקרת confidence-correctness" שהם מתארים.
  2. **זה מחייב לשנות את המסגור.**
     - אי אפשר לטעון ש-boosting הוא שיפור כללי.
     - הטענה הנכונה: telemetry עוזר במידה שהוא מתיישב עם השגיאות. על PRMBench זה גבוה. ייתכן שזה נובע מהשגיאות המוזרקות שם, וזו השערה שלא נבדקה.
     - reviewer שמכיר את המאמר הזה ישאל בדיוק את זה.
  3. **זה תומך ב-PRM כעוגן, ובחשד לגבי redundancy.**
     - ה-external signal, ובמקרה שלנו ה-PRM, הוא החלק ש"ממשיך לעבוד", וה-telemetry צריך להיות מוכפף לו.
     - זה מתאים לממצא שלנו: ב-Socratic, ‏L-SML נתן ל-PRM משקל קטן מדי.
     - **השערה בלבד:** אם entropy יורדת דרך repetition, אולי זה מסביר למה הרווח שלנו ב-PRMBench מרוכז ב-redundancy וב-circular. אלה שגיאות שמשנות את דפוסי ה-confidence גם בלי קשר לנכונות.
  4. **רעיון שאפשר לקחת: מדד ללא labels שמנבא מראש אם ה-telemetry יעזור.**
     - Model Collapse Step עצמו לא רלוונטי, כי הוא דורש אימון.
     - אצלנו המקבילה יכולה להיות, למשל, הקורלציה בין ה-telemetry ל-PRM בכל cell. ב-Socratic היא הייתה 0.07–0.61.
     - **הקושי:** קורלציה נמוכה יכולה לסמן גם מידע משלים וגם רעש, ובלי labels אי אפשר להבחין ביניהם.
     - כל כלל כזה חייב להיקבע ולהיות מוקפא לפני הבדיקה. הבדיקה צריכה להיות על בנצ'מרק נקי, למשל ProcessBench, ולא על Socratic שכבר ראינו.
  5. **לחיבור ל-RLVR (שקופית 2):**
     - אם משתמשים ב-locator שלנו כ-reward, המאמר צופה שה-policy ילמד לנצל את אותות ה-certainty, למשל דרך חזרה על טקסט.
     - זה מחזק את ההמלצה לא לעבור לאימון לפני שה-fusion יציב.

## 7. איפה נמצאות שאר התוצאות

ראה `docs/meetings/ADVISOR_MATERIALS_INDEX_2026-10-19.md`.
