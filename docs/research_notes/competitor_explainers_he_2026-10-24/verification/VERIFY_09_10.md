# אימות עובדתי: 09_RPRM.md ו-10_SLIDER.md מול קובצי ה-PDF המקוריים

בדקתי כל טענה מול הטקסט המלא של קובצי ה-PDF (כולל הנספחים), שחולץ עם pymupdf. מספרי העמודים הם עמודי ה-PDF (1 עד N). ב-R-PRM מופיע בסוגריים גם מספר העמוד בכרך ה-Proceedings. לא ערכתי אף קובץ במאגר.

מקרא פסיקות: **נכון** / **לא מדויק** / **שגוי** / **לא נמצא במאמר** / **מחוץ להיקף** (טענות על השיטה שלנו או על מספרים שלנו).

---

## קובץ 1: 09_RPRM.md מול "R-PRM: Reasoning-Driven Process Reward Modeling"

| מס' | הטענה (בקיצור) | פסיקה | ראיה (עמוד, מיקום, ציטוט) | תיקון מוצע |
|---|---|---|---|---|
| 1 | She et al., EMNLP 2025 Main | נכון | עמ' 1, כותרת עליונה: "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 13438–13451". המילה "Main" לא כתובה, אבל זה כרך הכנס הראשי ולא Findings. | אפשר להוסיף: "(Proceedings, עמ' 13438–13451)". |
| 2 | 14 עמודים, כולל נספחים A–G | נכון | נספחים A–G מופיעים בעמ' 11–14 (13448–13451). נספח G הוא Table 6 בעמ' 14. | — |
| 3 | במאמר אין מספור למשוואות | נכון | עמ' 3–4: למשוואות של R_i, A_i/J_i, L_SFT, L_DPO ו-R_i (K דגימות) אין מספרים. | — |
| 4 | ערכי Figure 8 מופיעים רק בגרף | נכון | עמ' 12, נספח D: בטקסט יש רק "8.2 points"; Figure 8 מופיע ככיתוב בלבד, בלי ערכים מספריים בטקסט. | — |
| 5 | PRM רגיל: R_i=M(Q,s_1..s_i), ציון סקלרי (§3.1) | נכון | עמ' 3, §3.1: "reward model that outputs a scalar reward Ri for the step si". | — |
| 6 | בעיה (א): שיפוט ישיר קשה, ולכן הלמידה פחות יעילה | נכון | עמ' 1, §1: "constrains the model's learning process and reduces learning efficiency". עמ' 3: "direct prediction is relatively difficult". | — |
| 7 | בעיה (ב): אין הסבר מדוע הצעד שגוי | נכון | עמ' 1, §1: "fails to identify why specific steps are incorrect". | — |
| 8 | בעיה (ג): מחסור בתוויות; תיוג אנושי יקר; MC יקר ורועש (§1, §2.2) | נכון | עמ' 1, §1: "incurs substantial costs". עמ' 2, §2.2: "MC methods incur a large computational cost and inevitably introduce bias and noise". | — |
| 9 | תוכן הביקורת: צעדים קודמים, מקור נתונים, עקביות, חישוב; prompt בטבלה 6 | לא מדויק | עמ' 14, Table 6 (נספח G) מונה חמישה רכיבים: Previous Steps, **Now Step Analysis**, Data Source, Consistency, Calculation. עמ' 3, §3.1: "assessing the objective and data sources of the current step". | "ניתוח הצעדים הקודמים, ניתוח הצעד הנוכחי ומטרתו, מקור הנתונים, עקביות וחישוב (Table 6, נספח G)". |
| 10 | הציון הוא ההסתברות ל-"Yes" אחרי הנימוק | נכון | עמ' 4, §3.3: "average probability of 'Yes' judgments (using softmax with 'No' judgments)". | — |
| 11 | מיצוע כמה נימוקים נותן ציון יציב יותר | נכון | עמ' 4, §3.3: "helps mitigate potential reasoning inconsistencies and stochastic variations". | — |
| 12 | A_i=G(Q,s), J_i=G(Q,s,A_i), Y_i=A_i⊕J_i | נכון | עמ' 3, §3.1. | — |
| 13 | L_SFT הוא instruction tuning רגיל על טקסט שכתב המורה | נכון | עמ' 3: "This is equivalent to standard instruction tuning". הטקסט נוצר על ידי "a stronger LLM"; §4.1 מזהה אותו כ-LLaMA3.3-70B. | — |
| 14 | נוסחת L_DPO; π_ref הוא עותק מוקפא של R-PRM-SFT | נכון | עמ' 4: "We copy and freeze R-PRM-SFT as the reference policy πref". | — |
| 15 | §3.2 נקרא "Meta-Optimization" | נכון | עמ' 3: "3.2 Process Reward Modeling Meta-Optimization". | — |
| 16 | Y^w הוא מסלול שהשיפוט שלו תואם את התווית, ו-Y^l מסלול שלא תואם | נכון | עמ' 4: "categorized into two groups depending on whether the judgments align with the annotated labels". | אפשר לכתוב "התווית המתויגת (PRM800K)", בהתאם לנוסח המאמר. |
| 17 | "מה שמבחין בין המסלולים הוא נכונות התשובה הסופית, לא איכות הנימוק" | לא מדויק (ניסוח) | עמ' 4: ההבחנה היא בין שיפוטים (Yes/No) שתואמים לתווית ושיפוטים שלא. "התשובה הסופית" עלול להתפרש כתשובה הסופית של הפתרון. הקביעה "לא איכות הנימוק" היא הסקה נכונה: המאמר לא מעריך את איכות הנימוק בנפרד. | "מה שמבחין בין המסלולים הוא רק התאמת השיפוט הסופי (Yes/No) לתווית; איכות הנימוק עצמו לא נבדקת". |
| 18 | לא ברור אם מסלולי ה-DPO הראשונים נדגמו מהמורה או מ-R-PRM-SFT | נכון (המאמר אינו מכריע) | עמ' 5, §4.1: "generate four evaluation trajectories per PRM800K case, yielding approximately 289k SFT and 269k DPO samples". הניסוח הזה מרמז על המורה. לעומת זאת, עמ' 3, §3.2 מדבר על "self-evolve". | אפשר להוסיף: "הניסוח ב-§4.1 נוטה לכך שגם דגימות ה-DPO באו מהמורה". |
| 19 | נוסחת R_i בהיסק, softmax על Yes/No בלבד | נכון | עמ' 4, §3.3. | — |
| 20 | המורה LLaMA3.3-70B-Instruct מפיק 4 מסלולים לכל מקרה מ-PRM800K | נכון | עמ' 5, §4.1: "four evaluation trajectories per PRM800K case". | — |
| 21 | "נשמרים רק מסלולים תואמים. התוצאה: 289k SFT ו-269k DPO" | לא מדויק | עמ' 3, §3.1: הסינון ("We retain only those evaluation analyses...consistent with human label") חל על נתוני ה-SFT. עמ' 4, §3.2: ב-DPO גם המסלולים **הלא תואמים** משמשים, בתור Y^l. | "ל-SFT נשמרים רק מסלולים שהשיפוט שלהם תואם את התווית. ל-DPO משמשים גם המסלולים הלא תואמים, בתור הצד הנדחה (Y^l)". |
| 22 | כ-289k דגימות SFT וכ-269k דגימות DPO | נכון | עמ' 5, §4.1: "approximately 289k SFT and 269k DPO samples". | — |
| 23 | 285k מופיע ב-§5.2 ובאיור 3; הפער לא מוסבר | לא מדויק (חלקית) | עמ' 7, §5.2: "full 285k-sample dataset". באיור 3 עצמו אין תווית 285k, רק ציר "Training Data Amount (K)". זה שהפער לא מוסבר: נכון. | "ב-§5.2 (בטקסט שמתאר את איור 3) מופיע 285k; הפער מ-289k לא מוסבר". |
| 24 | המודל הבסיסי: Qwen2.5-Math-7B-Instruct | נכון | עמ' 1, §1, ועמ' 5, §4.1. | — |
| 25 | epoch אחד SFT (lr 5e-6), ואחריו epoch אחד DPO (lr 5e-7), batch 128 | נכון | עמ' 5, §4.1: "fine-tuned for one epoch with batch size 128 and learning rates of 5e-6 (SFT) and 5e-7 (DPO)". "epoch אחד לכל שלב" היא הקריאה הסבירה של הנוסח. | — |
| 26 | ה-checkpoint נבחר לפי loss על 20k דוגמאות ולידציה | נכון | עמ' 5: "We reserve 20k samples for validation and select the checkpoint with the lowest validation loss". | — |
| 27 | שיפור עצמי: R-PRM-DPO דוגם על PRM800K ונבנים 30k זוגות | נכון | עמ' 7, §5.1: "construct a new preference dataset of 30k examples". | המאמר כותב "examples", לא "pairs". |
| 28 | הזוגות עדיין מתויגים לפי תוויות PRM800K; "ללא נתונים מתויגים נוספים" אינו "ללא תוויות" | נכון (הסקה) | §5.1 לא מציין במפורש איך הזוגות מתויגים, אבל הדרך היחידה שהמאמר מגדיר לבניית זוגות (עמ' 4, §3.2) נשענת על "annotated labels". עמ' 7: "without requiring any additional labeled data". | אפשר לציין: "הסקה מ-§3.2. §5.1 לא אומר זאת במפורש". |
| 29 | ProcessBench: 70.4, ואחר כך 72.6 (Iter1) ו-74.1 (Iter2) | נכון | עמ' 7, Figure 2: Average 65.2 / 70.4 / 72.6 / 74.1. עמ' 6: "average F1 score of 74.1". | — |
| 30 | ל-PRMBench לא מדווח ציון אחרי האיטרציות | נכון | Figure 2 ו-Figure 7 מתייחסים רק ל-ProcessBench. Table 2 כולל רק SFT ו-DPO. | — |
| 31 | בכל הטבלאות K=10 (§4.1) | נכון | עמ' 5: "Unless stated otherwise, results are reported using ten evaluation trajectories per step". | אפשר להוסיף: גם LLM-as-judge ב-Table 1 דוגמים 10 ומכריעים ברוב קולות (כיתוב Table 1). |
| 32 | ההמלצה היא K=4 (§5.3) | נכון | עמ' 8: "We recommend using K = 4". | — |
| 33 | ממוצע ProcessBench עולה מ-61.3 ב-K=1 ל-71.7 ב-K=32 (איור 5) | נכון | עמ' 8, Figure 5: Average 61.3, 62.8, 67.6, 69.7, 70.8, 71.7. | — |
| 34 | 0.52s ב-K=4 ו-1.08s ב-K=10; H100 אחד, vLLM; נמדד על Iter2 | נכון | עמ' 11, נספח A: "single NVIDIA H100 GPU (vLLM 0.8.4)". Figure 7: "R-PRM-DPO-Iter2 Inference Time vs F1". | — |
| 35 | יצירת טקסט ארוך לכל צעד ולכל דגימה, לא מעבר forward אחד | נכון | עמ' 3, §3.1: "two phases within a single generation process". עמ' 4, §3.3: K מסלולים לכל צעד. | — |
| 36 | סף קבוע 0.5 ב-ProcessBench וב-PRMBench (§5.4) | נכון | עמ' 8: "we adopt a fixed threshold of 0.5 for binary classification". | — |
| 37 | נספח D: לבסיס PRM800K יש פער של 8.2 בין סף 0.5 לסף האופטימלי; R-PRM יציב יותר | נכון | עמ' 12: "performance gap of 8.2 points between the 0.5 threshold and its optimal". | — |
| 38 | Table 1: 70.4 מול 56.5 (פער 13.9), 57.4 (LLaMA-70B), 61.9 (GPT-4o) | נכון | עמ' 5, Table 1. עמ' 1, תקציר: "13.9 and 8.5 F1". | — |
| 39 | Table 2: 66.8 (DPO), 64.9 (SFT), 58.3 (PRM800K, פער 8.5), 66.8 (GPT-4o), 68.8 (o1-mini) | נכון | עמ' 5, Table 2. | — |
| 40 | פירוק לפי קטגוריות: Simplicity 55.2, Soundness 71.2, Sensitivity 76.6 | נכון | עמ' 5, Table 2, שורת R-PRM-7B-DPO. | — |
| 41 | DPO הוריד את Simplicity לעומת SFT: 58.7→55.2 | נכון | Table 2. גם NCL ירד, מ-64.7 ל-58.2. | אפשר להוסיף את הירידה ב-NCL. |
| 42 | NR (52.2) הוא הציון החלש ביותר | נכון | Table 2: הוא הנמוך מבין תשע תת-הקטגוריות של DPO. | — |
| 43 | שורות המסומנות * הועתקו מ-Zhang et al. ולא שוחזרו | נכון | כיתובי Table 1 ו-Table 2: "Results marked with * are from Zhang et al." | הערה: ב-Table 2 השורה Qwen2.5-Math-7B-PRM800K **לא** מסומנת *, כלומר הורצה על ידם, בעוד שב-Table 1 היא מסומנת. |
| 44 | §4: זו שיטת PRM של 7B שמדווחת על PRMBench ועל ProcessBench | נכון | עמ' 4–5, §4.1, Tables 1–2. | — |
| 45 | 66.8 > 65.5 שלנו; < 70.40 שלנו; "אותו פורמט פירוט שלנו" | מחוץ להיקף | — | — |
| 46 | השיטה תוקפת את חולשת "ציון סקלרי חסר הקשר" | לא מדויק | עמ' 1, תקציר: "existing PRMs typically output evaluation scores directly". המאמר מבקר את הציון **הישיר ללא נימוק וללא הסבר**, ולא טוען שהציון "חסר הקשר" (PRM רגיל מותנה בכל ה-prefix). | "תוקפת את חולשת 'ציון ישיר בלי נימוק'". |
| 47 | טבלת §5 (עמודת R-PRM): מודל גנרטיבי חדש (SFT+DPO); תוויות PRM800K לסינון ה-SFT ולזוגות ה-DPO; K=10 ביקורות לכל צעד; סף 0.5 על ממוצע ה-Yes | נכון | §3.1–3.3, §4.1, §5.4. | עמודת PRM boosting: מחוץ להיקף. |
| 48 | לא מאומת ש-"Overall" הוא ה-PRMScore הרשמי; המאמר קורא לו "F1" ו-"Overall score" ולא מגדיר אותו | נכון (המאמר אינו מכריע) | עמ' 1: "8.5 F1 scores". עמ' 5: "F1 scores of ... 64.9 on PRMBench". עמ' 12: "PRMBench Overall Score". הגדרה אין. | רמז חיצוני שלא נבדק מול PDF: ערכי השורות המסומנות * (למשל o1-mini 68.8, GPT-4o 66.8) נראים זהים למספרים שפורסמו ב-PRMBench. כדאי לאמת מול מאמר PRMBench. |
| 49 | "נספח D עצמו מראה שהסף מזיז ציונים בכ-8 נקודות" | לא מדויק | עמ' 12: הפער של 8.2 הוא של **הבסיס Qwen2.5-Math-7B-PRM800K**. על R-PRM המאמר טוען "significant robustness advantages". | "נספח D מראה שעבור PRM הבסיס הסף מזיז את הציון ב-8.2 נקודות. R-PRM מתואר כיציב (המספרים שלו נמצאים רק בגרף)". |
| 50 | Qwen2.5-Math-PRM-7B לא מופיע במאמר | נכון | אין לו מופע בטבלאות או בטקסט. Zhang et al. 2025b מצוטט, אבל המודל שלו לא מוערך. | — |
| 51 | "הבסיס שלהם הוא Qwen2.5-Math-7B-PRM800K" | לא מדויק | עמ' 1 ו-5: המודל שעליו מאמנים הוא Qwen2.5-Math-7B-Instruct. Qwen2.5-Math-7B-PRM800K הוא קו ההשוואה (עמ' 2: "strongest baseline trained on the same data"). | "קו הבסיס החזק ביותר שלהם, שאומן על אותם נתונים, הוא Qwen2.5-Math-7B-PRM800K. המודל שעליו מאמנים הוא Qwen2.5-Math-7B-Instruct". |
| 52 | הציון מבוסס על דגימה (K=10) ולא מדווחת שונות | נכון | בכל המאמר אין std, CI או מספר seeds. | — |
| 53 | סיכום: מאמת גנרטיבי שאומן בפיקוח, K ביקורות לכל צעד, 66.8 ב-PRMBench | נכון | §3–§4, Table 2. | ההשוואה ל-70.40 שלנו: מחוץ להיקף. |

### התיקונים החשובים ב-09_RPRM.md
1. **סינון ה-DPO (שורה 21):** רק נתוני ה-SFT מסוננים להתאמה לתווית. ה-DPO **משתמש** במסלולים הלא תואמים בתור Y^l. הנוסח הנוכחי מרמז שגם 269k דגימות ה-DPO הן מסלולים "תואמים" בלבד.
2. **נספח D (שורה 49):** הפער של 8.2 נקודות שייך ל-PRM הבסיס, לא ל-R-PRM. המאמר טוען ש-R-PRM יציב. צריך לתקן כדי לא לייחס ל-R-PRM רגישות לסף.
3. **"הבסיס שלהם" (שורה 51):** המודל שעליו מאמנים הוא Qwen2.5-Math-7B-Instruct. PRM800K-7B הוא קו ההשוואה.
4. **רכיבי הביקורת (שורה 9):** חסר "Now Step Analysis".
5. **285k (שורה 23):** המספר מופיע בטקסט של §5.2, לא כתווית באיור 3.
6. **"חסר הקשר" (שורה 46):** הביקורת של המאמר היא על ציון ישיר בלי נימוק.
7. **(רשות):** ב-Table 2 השורה PRM800K-7B לא מסומנת *, כלומר הורצה על ידי המחברים. המספר 58.3 הוא שלהם.

---

## קובץ 2: 10_SLIDER.md מול "Interpreting Reasoning of Large Language Models via Partial Information Decomposition"

| מס' | הטענה (בקיצור) | פסיקה | ראיה (עמוד, מיקום, ציטוט) | תיקון מוצע |
|---|---|---|---|---|
| 1 | Halder, Zhang, Dutta; arXiv 2610.00571v1; 30 בספטמבר 2026 | נכון | עמ' 1: "arXiv:2610.00571v1 [cs.IT] 30 Sep 2026". | — |
| 2 | מודלים ארוכים חוזרים על עצמם, כולל "double-check" שלא מוסיף מידע (§1, איור 1) | נכון | עמ' 1, §1: "frequently revisiting information that has already been established". עמ' 2, Figure 1: "Let's double-check". | — |
| 3 | "שלב כזה לא בהכרח שגוי... לכן אות שמודד 'האם השלב נכון' לא אמור לתפוס אותו" | לא נמצא במאמר (פרשנות) | המאמר אומר רק שחזרה מעלה את העלות "without necessarily improving the correctness" (עמ' 1). הוא לא דן באותות נכונות או ב-PRM. | לסמן כ"פרשנות שלי". |
| 4 | MI, CMI ו-InfoGain רק מודדים אם שלב אינפורמטיבי, בלי להבחין בין סוגי המידע (§1, §1.1) | נכון | עמ' 2: "only quantify whether the current step is informative or not". עמ' 3, §1.1: InfoGain/InfoBias של Yong et al. | — |
| 5 | המטרה: זיהוי ברמת השלב בלי reasoning ייחוס | נכון | עמ' 2: "without requiring reference reasoning traces". | — |
| 6 | פירוק לארבעה רכיבים אי-שליליים (Proposition 1) | נכון | עמ' 5, Proposition 1. | — |
| 7 | המשמעויות של Red, Uni(Si), Uni(S<i) ו-Syn | נכון | עמ' 4, §1.2: "Syn ... emerges only when X and Y are both present together". | — |
| 8 | שלב חוזר: Red שולט, Uni(Si) ו-Syn קרובים לאפס | נכון | עמ' 5: "we expect redundant information ... to be high and dominant". במקרה האידיאלי, Theorem 2 נותן אפס. | — |
| 9 | Definition 1 (Bertschinger): מינימום CMI על פני התפלגויות עם שוליים (A,X) ו-(A,Y) זהים | נכון | עמ' 4, Definition 1. | — |
| 10 | משוואה 1: I(A;X)=Uni+Red; משוואה 2: I(A;X\|Y)=Uni+Syn | נכון | עמ' 4, משוואות (1)–(2). | — |
| 11 | משוואה 4 / Definition 2, עם η≥1 | נכון | עמ' 5, Definition 2: "For some η ≥1 and i > 1". | אפשר להוסיף את התנאי i>1. חוסר עקביות במאמר: בעמ' 2 הוא נקרא "Proposition 2", ובעמ' 5 "next proposition", אבל הוא מסומן Definition 2. |
| 12 | Definition 3: Trajectory-RRI סופר שלבים עם RRI_i>0 | נכון | עמ' 6: "RRI(Q) = Σ_{i=2}^T I[RRI_i > 0]". | — |
| 13 | Theorem 2: Si=(SA,SN), SA⊆S<i, SN רעש בלתי תלוי, ולכן Uni=Syn=0 ו-Step-RRI>0 | נכון | עמ' 5, Theorem 2: "the relevant part SA ⊆S<i". נדרשים גם I(A;SA)>0 ו-SN⊥A,S<i. | אפשר להוסיף במפורש את ההנחה I(A;SA)>0. |
| 14 | ההוכחה בנספח B.2 נשענת על Lemma 2 (monotonicity תחת side information) | נכון | עמ' 15, B.2, צעד (b): "holds from Lemma 2 (Monotonicity under adversarial side information)". | — |
| 15 | Theorem 1: רעש בלתי תלוי לא מגדיל unique information (B.1, data processing) | נכון | עמ' 5: "Uni(A:Si\|Si−1) ≥Uni(A:S′i\|Si−1)". עמ' 15: "conditional form of the data processing inequality". | דיוק: המשפט מותנה ב-S_{i−1} (השלב הקודם בלבד), לא ב-S<i. הרעש חיבורי ומניחים N⊥(A,Si,Si−1). |
| 16 | זה הקשר היחיד לשגיאות שאינן חזרה, ונבדק רק על "בעיות סינתטיות" (איור 5, איור 10) | לא מדויק | איור 5 (עמ' 7) הוא בעיה אריתמטית סינתטית. איור 10 (עמ' 19, D.1) הוא בעיית מילים **אמיתית** מ-GSM8K, עם מסלול של DeepSeek-R1-Distill, 1,000 מופעים פרמטריים ו**רעש מוזרק**. | "נבדק רק עם רעש שמוזרק באופן סינתטי (איור 5: אריתמטיקה סינתטית; איור 10: בעיית GSM8K פרמטרית), לא על שגיאות טבעיות". |
| 17 | GPT-4o-mini מייצר עד 50 וריאנטים, רק המספרים משתנים, מבנה `<step i>` נשמר, ואותו מספר שלבים; עצירה אחרי 5 אי-התאמות רצופות | נכון | עמ' 6: "We generate up to 50 variants per problem and terminate generation after five consecutive structural mismatches". | אפשר להוסיף שכל וריאנט מתקבל משמש כבסיס לוריאנט הבא (שרשרת), ושבכל כישלון היצירה חוזרת למקור. חוסר עקביות: נספח C.1 (עמ' 16) אומר "we generate 50" ולא "up to 50". |
| 18 | prefix, שלב ותשובה מקודדים ב-all-MiniLM-L6-v2 | נכון | עמ' 7, §2. | — |
| 19 | k-means עם 10 clusters; Table 2: ערכי PID עולים עם מספר ה-clusters והמגמות נשמרות | נכון | עמ' 7, ועמ' 18, C.2: "increasing the number of clusters generally yields higher PID values". | דיוק: ניתוח הרגישות נעשה על בעיה אחת בלבד (Problem 3 באיור 6), עם 10, 15 ו-20 clusters. |
| 20 | התפלגות משותפת על פני הווריאנטים; PID עם CVX estimator של Liang et al. 2023 | נכון | עמ' 7: "we utilize the CVX estimator proposed by Liang et al. (2023)". | — |
| 21 | [לא אומת] עד 50 דגימות על 1000 תאים (10×10×10); המדידה היא של שונות מהחלפת מספרים, לא של הטקסט המקורי | נכון (נגזר מהמאמר; המאמר לא דן בכך) | עמ' 16, C.1: "50 numerical variants" ל-PRMBench. עמ' 7: 10 clusters לכל אחד משלושת המשתנים. Figure 4 (עמ' 6): ההתפלגות נאמדת על פני הווריאנטים. המאמר לא דן בדלילות הטבלה. | אפשר להסיר את התג [לא אומת]: המרכיבים מופיעים במאמר, והמסקנה היא חשבון. כדאי לציין שבניסויי GSM8K באיורים 6 ו-11 נוצרו 1,000 דגימות פרמטריות (עמ' 20), לא 50. |
| 22 | קלט: שאלה, שלבים, A ו-50 וריאנטים מ-LLM חיצוני | נכון | עמ' 6 ועמ' 16. | — |
| 23 | כ-38 שניות לדוגמה ב-PRMBench על RTX 6000 (D.5) | נכון | עמ' 23: "For PRMBench, the average runtime of SLIDER is approximately 38 seconds per sample". | המאמר לא אומר אם הזמן כולל את קריאות ה-API ל-GPT-4o-mini. |
| 24 | "עוד עד 50 קריאות ל-GPT-4o-mini לכל בעיה" | לא מדויק | עמ' 16–17, prompt: "Generate exactly 1 numerical variation" (קריאה לכל וריאנט). עמ' 6: וריאנט שנכשל נזרק. מספר הקריאות הוא מספר הווריאנטים שהתקבלו ועוד הניסיונות שנפסלו, ולכן יכול לעבור את 50. | "קריאה אחת ל-GPT-4o-mini לכל וריאנט, כולל ניסיונות שנפסלו: כ-50 קריאות ויותר לבעיה". |
| 25 | ציטוט: "we calibrate the decision threshold on a held-out set by training a linear classifier" (§3) | נכון | עמ' 8, ציטוט מדויק. | — |
| 26 | "כלומר הסף מכויל בעזרת תוויות" | נכון (הסקה מבוססת) | עמ' 8: אימון מסווג לינארי לזיהוי "step-level ground-truth redundancy" מחייב תוויות. המילה "labels" לא מופיעה. | אפשר לכתוב: "מחייב תוויות (המאמר לא אומר זאת במפורש)". |
| 27 | גודל ה-held-out וגודל ה-test לא מדווחים | נכון | עמ' 8 ועמ' 19: אין מספרים, ואין תיאור של אופן החלוקה. | — |
| 28 | η בניסוי PRMBench לא מדווח; 1.25 באיור 6; 1 באיור 11 | נכון | עמ' 8: "with η = 1.25". עמ' 20: "(with η = 1)". ל-PRMBench אין ערך. | — |
| 29 | מחלקת ה-redundancy של PRMBench, "around 750 problems" (נספח D) | נכון | עמ' 19: "This class has around 750 problems". | — |
| 30 | שלב שסומן redundant הוא החיובי | נכון | עמ' 19: "We treat steps annotated as redundant as the positive class". | — |
| 31 | [לא אומת] האם שלב 1 (i>1) נכלל בהערכה | נכון (המאמר אינו מכריע) | Definition 2 מגדיר את המדד רק ל-i>1 (עמ' 5). נספח D לא מתייחס לשלב 1. | — |
| 32 | המדד הוא "Accuracy (%)" (איור 7); אין balanced accuracy, F1 או AUC | נכון | עמ' 8, Figure 7, ציר: "Accuracy (%)". בכל המאמר אין F1, AUC או balanced. | — |
| 33 | תוצאות: 61.91 / 51.6 / 48.4 "(איור 7)" | נכון (עם הערה) | עמ' 8: האיור מציג 61.9 / 51.6 / 48.4. בטקסט: "highest accuracy of 61.91%". | "61.91 (טקסט §3; באיור 7: 61.9)". |
| 34 | הפערים +10.31 ו-+13.53 מופיעים ב-§3 | נכון | עמ' 8: "improving over InfoGain and Cosine Similarity by 13.53 and 10.31 percentage points". | לשייך במפורש: +13.53 מול InfoGain, +10.31 מול Cosine. (61.91−48.4=13.51; ההפרש הקטן הוא עניין של עיגול במאמר.) |
| 35 | InfoGain: entropy של top-10 tokens מ-QwQ-32B (D.2, משוואות 7–9) | נכון | עמ' 21–22: "we use QwQ-32B to obtain the top-K next-token probabilities"; "We use K = 10". | — |
| 36 | אין baseline של PRM | נכון | עמ' 8: רק InfoGain ו-Cosine Similarity. | — |
| 37 | Table 1: ρ=0.95 ו-0.91 ב-GSM8K | נכון | עמ' 9, Table 1: DeepSeek 0.95, QwQ 0.91, GPT-4.1 0.73. | דיוק חשוב: המתאם מחושב על ממוצעי bins לפי אורך (50 bins ב-GSM8K, 7 ב-AIME; עמ' 22), לא ברמת המסלול הבודד. |
| 38 | בחירת נתונים ל-SFT (איור 9, Table 3) | נכון | עמ' 9–10 ועמ' 22. | — |
| 39 | §4: אותה משימה ואותו benchmark (זיהוי שלבים מיותרים ב-PRMBench) | נכון (בחלק שנוגע למאמר) | עמ' 8 ועמ' 19. | 71.2 ו-54.7 שלנו: מחוץ להיקף. |
| 40 | "אותה טענה עקרונית: אות נכונות רגיל מפספס redundancy, וצריך אות שאינו PRM מאומן" | לא נמצא במאמר | המאמר לא דן ב-PRM או באותות נכונות. הטענה שלו היא על מדדי מידע קלאסיים (עמ' 2–3) ועל הצורך בשיטות "without human supervision" (עמ' 1). | "אצלנו זו טענה עקרונית. המאמר טוען טענה מקבילה, אבל על מדדי מידע קלאסיים, לא על PRM". |
| 41 | InfoGain הוא למעשה ערוץ telemetry מבוסס entropy | נכון (בתיאור); הדמיון לשלנו מחוץ להיקף | עמ' 21–22, משוואה 9: "∆Ii = Hi−1 −Hi". | דיוק: זה ערך אחד לשלב, ההפרש בין entropy של הטוקן הבא בגבולות השלבים, שמחושב ב-QwQ-32B. |
| 42 | §5.1: Step-RRI מוגדר רק ל-redundancy; אין מחלקות PRMBench אחרות; אין PRMScore | נכון | עמ' 8 ועמ' 19: הערכה על מחלקת ה-redundancy בלבד. Theorem 1 עוסק בשגיאות, אבל רק על רעש סינתטי. | החלק "אנחנו מכסים...": מחוץ להיקף. |
| 43 | §5.2: וריאנטים, encoder ו-PID; כ-38 שניות | נכון (עם הסתייגויות 23–24) | עמ' 6–7 ועמ' 23. | לתקן "עד 50 קריאות" כמו בשורה 24. השוואת העלות לשלנו: מחוץ להיקף. |
| 44 | §5.3: הסף מכויל במסווג לינארי על held-out מתויג | נכון (הסקה) | עמ' 8. | החלק על L-SML: מחוץ להיקף. |
| 45 | §5.4: "61.91 הוא accuracy רגיל" | לא מדויק | עמ' 8: "Accuracy (%)" בלי הגדרה. המאמר לא אומר אם מדובר ב-accuracy רגיל או מאוזן, ולא אם ה-test מאוזן. | "61.91 מדווח כ-'Accuracy' בלי הגדרה. המאמר לא מציין אם הוא רגיל או מאוזן, ולא את הרכב ה-test". |
| 46 | 86.8% של המחלקה הדומיננטית | מחוץ להיקף | — | — |
| 47 | [לא אומת] 61.91 נמוך מהמסווג הטריוויאלי, או שחושב על אוכלוסייה מאוזנת | נכון (המאמר אינו מכריע) | עמ' 19 מגדיר רק חיובי ושלילי. אין תיאור של איזון, דגימה או חלוקה. | — |
| 48 | §5.5: אין CI, אין גודל test, אין השוואה ל-PRM | נכון | בכל המאמר אין CI או std לתוצאת PRMBench. ל-p-values יש דיווח רק ב-Table 1 ובאיור 9. | — |
| 49 | §6: מוכיח (Theorem 2) שהמדד חיובי כשהשלב חוזר; בהערכה על PRMBench יש מספר accuracy אחד, סף מתויג, ואין baseline של PRM | נכון | עמ' 5, עמ' 8 ועמ' 15. | דיוק: ההוכחה תקפה בהנחות אידיאליות (SA⊆S<i, רעש בלתי תלוי, PID אמיתי), לא על האומדן הבדיד בפועל. "לא בר-השוואה לשלנו": מחוץ להיקף. |

### התיקונים החשובים ב-10_SLIDER.md
1. **"accuracy רגיל" (שורה 45):** המאמר כותב רק "Accuracy" ולא מגדיר אותו. אי אפשר לקבוע שזה accuracy רגיל. מכאן שגם הטיעון על "נמוך מהמסווג הטריוויאלי" תלוי בהנחה שהמאמר לא מאשר. הנוסח של [לא אומת] כבר מכיר בזה, אבל המשפט הקודם קובע זאת כעובדה.
2. **טענה עקרונית על PRM (שורה 40):** המאמר לא טוען שאותות נכונות או PRM מפספסים redundancy. זו טענה שלנו, וצריך לסמן אותה ככזו.
3. **Theorem 1 ו"בעיות סינתטיות" (שורה 16):** איור 10 הוא בעיית GSM8K אמיתית שהוזרק בה רעש סינתטי. "רק רעש סינתטי" מדויק יותר מ"רק בעיות סינתטיות". בנוסף, המשפט מותנה ב-S_{i−1} ולא ב-S<i.
4. **מספר הקריאות ל-GPT-4o-mini (שורות 24 ו-43):** כל וריאנט הוא קריאה, ויש גם ניסיונות שנפסלו, ולכן מדובר בכ-50 קריאות ויותר, לא "עד 50". המאמר לא אומר אם 38 השניות כוללות את הקריאות האלה.
5. **פרשנות "אות נכונות לא אמור לתפוס" (שורה 3):** לסמן כפרשנות.
6. **התג [לא אומת] בשורה 21 מיותר:** 50 וריאנטים, 10 clusters ושלושה משתנים מופיעים במאמר, ו-1000 תאים הוא חשבון.
7. **דיוקים קלים:** לשייך את +13.53 ל-InfoGain ואת +10.31 ל-Cosine. באיור מופיע 61.9 ובטקסט 61.91. המתאמים ב-Table 1 מחושבים על ממוצעי bins. ניתוח הרגישות ל-clusters נעשה על בעיה אחת.
8. **חוסר עקביות במאמר עצמו (לא נזכר בהסבר):** Step-RRI נקרא "Proposition 2" בעמ' 2 אבל מסומן Definition 2. בעמ' 6 כתוב "up to 50" ובנספח C.1 "we generate 50".
