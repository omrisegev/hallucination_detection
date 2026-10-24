# אימות מול המאמרים המקוריים: 01_CROP.md ו-02_FUSE.md

שיטה: חילצתי את הטקסט המלא של שני ה-PDF עם pymupdf (CROP: 39 עמודים, FUSE: 24 עמודים), קראתי את כל הטקסט כולל הנספחים, ובדקתי כל טענה מול הקטע שבו היא מופיעה. חיפשתי בטקסט המלא גם מונחים שהמסבירים אומרים שאינם במאמר (PRMScore, category, ICML, PRMBench, eigenvector). לא שיניתי אף קובץ במאגר.
מספרי העמודים הם מספרי עמודי ה-PDF. הציטוטים באנגלית מועתקים מהמאמר.

---

## קובץ 1: 01_CROP.md מול 2605.30085v2_CROP.pdf

| מס' | הטענה (בקיצור) | פסיקה | ראיה (עמוד, מיקום, ציטוט) | תיקון מוצע |
|---|---|---|---|---|
| 1 | מחברים: Cheung, Veeraraghavan, Chen, Balakrishnan, מ-Rice. arXiv 2605.30085v2 | נכון | עמ' 1, כותרת: "Department of Electrical & Computer Engineering, Rice University". בשוליים: "arXiv:2605.30085v2 [cs.AI] 10 Oct 2026" | — |
| 2 | פתרון שגוי כמעט אף פעם לא שגוי מתחילתו. יש כמה צעדים נכונים ואחריהם טעות שמתגלגלת (Sec 1, Figure 1) | נכון | עמ' 1, Sec 1: "its trace is rarely entirely wrong; valid early reasoning is typically followed". עמ' 2, Fig. 1: "First errors frequently occur after multiple correct steps" | — |
| 3 | UQ קיים נותן ציון אחד לתשובה (קבלה/דחייה). ציוני צעד כמו PRM אינם מכוילים ואין להם ערובה (Sec 1) | נכון | עמ' 1, Sec 1: "binary accept/reject decision"; "produce uncalibrated values that lack formal statistical guarantees" | — |
| 4 | השאלה: האורך המקסימלי של רישא שאפשר לשמור עם ערובה שהיא "נקייה בהסתברות גבוהה". השאר עובר לתיקון | נכון, עם הסתייגות | עמ' 1–2: "selecting a contiguous prefix ... while controlling the marginal probability that it contains an annotated error" | "בהסתברות גבוהה" היא הסתברות שולית (על פני כיול ומבחן), לא לכל רישא בנפרד. מוטב: "הרישא הארוכה ביותר שהסתברות הזיהום השולית שלה מוגבלת ב-α" |
| 5 | זריקת הכול מאבדת עבודה טובה, ושמירת הכול משאירה המשך מטעה שפוגע בתיקון (Table 3) | נכון | עמ' 1: "certifying an entire trace can discard valid early work". עמ' 12, Table 3: מול full trace ‏+21.9 כש-CROP עוצר לפני הגבול | — |
| 6 | אינטואיציה: עוצרים בחריגה הראשונה מ-λ. סף גבוה שומר יותר צעדים, ולכן ההפסד מונוטוני ואפשר לכייל כמו split conformal | נכון | עמ' 5, Sec 4: "raising the threshold can only increase the number of retained steps" | — |
| 7 | M_λ = max{m : R_t ≤ λ לכל t ≤ m} (Eq. 3) | נכון | עמ' 5, Eq. (3) | — |
| 8 | L_λ(Z) = 1{∃t ≤ M_λ : Y_t = 1} (Eq. 4) | נכון | עמ' 6, Eq. (4) | — |
| 9 | λ̂ = max{λ ∈ Λ : (1+Σ L_λ(Z_i))/(n+1) ≤ α} (Eq. 5, ובצורה כללית Eq. 1) | נכון | עמ' 5, Eq. (1); עמ' 6, Eq. (5) | — |
| 10 | אם אין סף אפשרי, מחזירים רישא ריקה (Sec 4.2) | נכון | עמ' 6: "otherwise, set λ̂ = ⊥, where M⊥(R1:T) = 0" | אפשר להוסיף: ה-fallback לא הופעל באף ריצה, אבל ב-GSM8K הוחזרה רישא ריקה ל-6.4% מתשובות המבחן (עמ' 12, "Empty prefixes") |
| 11 | מקרה פרטי של conformal risk control עבור הפסד בינארי מונוטוני (Lemma 1, Sec 4.1) | נכון | עמ' 5: "a specialized consequence of CRC [35]" | — |
| 12 | בהוכחה (Appendix A, Eq. 7) מוגדר B_i, "הסף הראשון שנכשל" | נכון | עמ' 18, Eq. (7): "define its first failing threshold as Bi = min{λ ∈ Λ : Lλ(Wi) = 1}" | — |
| 13 | (גזירה שלי) B_i הוא הסיכון המקסימלי על הצעדים עד הטעות הראשונה, כולל אותה | נכון, עם דיוק | הגזירה תקפה מ-Eq. 3–4: L_λ=1 בדיוק כאשר λ ≥ max_{t≤t*} R_t. המאמר עצמו לא כותב זאת | דיוק: על רשת סופית, B_i הוא הערך הקטן ביותר ב-Λ שאינו קטן מהמקסימום הזה. B_i = ∞ בתשובה נקייה או כשהמקסימום גבוה מכל הרשת (עמ' 18: "with Bi = ∞ if no threshold in Λ produces a failure") |
| 14 | λ̂ הוא בפועל קוונטיל נמוך של B_i, והערובה נובעת מטיעון דירוג תחת exchangeability | נכון (פרשנות עקבית עם ההוכחה) | עמ' 18, Eq. (11): לכל היותר k−1 ערכי כיול עם B_i ≤ λ̂, כאשר k=⌊α(n+1)⌋. עמ' 19: "the rank of (Bn+1, Un+1) is uniform" | המונח "קוונטיל" אינו במאמר. מדויק יותר: "λ̂ נמצא מתחת ל-B_i ה-k-י בגודלו בסט הכיול" |
| 15 | הערובה: P(הרישא מכילה צעד מסומן כשגוי) ≤ α (Proposition 1, Eq. 6) | נכון | עמ' 6, Eq. (6) | — |
| 16 | ההסתברות שולית, ביחד על סט הכיול ועל דוגמת המבחן | נכון | עמ' 6: "The probability is joint over calibration instances Z1:n and the future" | — |
| 17 | exchangeability ברמת התשובה השלמה, וצעדים בתוך תשובה רשאים להיות תלויים (Assumption 1, Sec 3) | נכון, עם השלמה | עמ' 4: "Steps within a trace may be arbitrarily dependent" | השלמה: ההנחה מותנית בנתוני אימון אות הסיכון ובכל בחירות העיבוד והתכנון שנקבעו לפני הכיול (עמ' 4, Assumption 1) |
| 18 | אין ערובה מותנית לכל רישא (סוף Sec 4.2, Sec 7) | נכון | עמ' 6: "This is not a conditional guarantee for each realized prefix." עמ' 13, Sec 7 | — |
| 19 | אין ערובה לפי קבוצות. בתשובות של 10 צעדים ומעלה הסיכון הנמדד 9.7% (Appendix H, Figure 5) | נכון, עם הסתייגות | עמ' 27–28: "For traces with at least 10 steps, risk increased to 9.7%" | המספר הוא על Arithmetic ו-GSM8K מאוחדים בלבד, כ-60 תשובות מבחן לפיצול, עם שימור 81.8%. כדאי לציין זאת כדי שלא ייקרא כמספר על PRMBench |
| 20 | אין ערובה לאמת סמנטית, רק ביחס לפרוטוקול התיוג (Sec 7) | נכון | עמ' 13: "CROP controls agreement with the fixed annotation protocol rather than latent semantic correctness" | — |
| 21 | אין ערובה שהחיתוך קרוב לטעות הראשונה. זו שאלת יעילות שתלויה באיכות האות (Sec 4.3) | נכון | עמ' 7, Sec 4.3: "More informative risk proxy functions can improve efficiency"; עמ' 13: "weak ... risk proxies may satisfy the risk constraint only by returning overly conservative, short prefixes" | — |
| 22 | פיצול 60/20/20 ברמת התשובה (Sec 5, Appendix B) | נכון | עמ' 9: "disjoint reasoning-instance-level 60/20/20 splits"; עמ' 19 | — |
| 23 | Λ: ‏201 קוונטילים של סיכוני האימון בדאטה-סטים העיקריים, ורשת קבועה {0,0.01,…,1} בחיצוניים, כולל PRMBench (Appendix B) | נכון | עמ' 19: "Λs contains 201 empirical quantiles"; "fixed numeric grid Λ = {0, 0.01, . . . , 1.00} with 101 points" | — |
| 24 | α = 0.05 | נכון | עמ' 9: "For a target risk α = 0.05". נבדקו גם 0.025 עד 0.10 (Appendix E) | — |
| 25 | חמישה אותות (Sec 5, Appendix B) | נכון | עמ' 8–9, רשימה 1–5 | — |
| 26 | Token/format: אורך, ספרות, אופרטורים ומיקום הצעד | נכון | עמ' 20: "character length, whitespace-token count, digit count ... normalized step position" | — |
| 27 | Trace-feature מוסיף token confidence, sequence likelihood, prediction entropy, calibrated confidence, energy וסיכומי hidden-state | נכון | עמ' 20: "token confidence, sequence likelihood, prediction entropy, calibrated confidence, and energy-style uncertainty"; וגם "latent-trajectory summaries" | — |
| 28 | Direct PRM: ‏R_t = 1 − softmax(h_t)_1 של Qwen2.5-Math-PRM-7B קפוא, במעבר אחד עם `<extra_0>` | נכון | עמ' 20: "with the Qwen PRM step separator <extra_0> after every step"; "scores the completed trace in one pass" | אפשר להוסיף: הקלט נחתך ב-8192 טוקנים (עמ' 20) |
| 29 | Trace+PRM: רגרסיה לוגיסטית על אותות 3 ו-4 יחד | לא מדויק | עמ' 9: "logistic regression combining the trace-feature inputs with the Direct PRM score"; עמ' 21: "appends the precomputed PRM error-risk proxy as an additional feature" | "רגרסיה לוגיסטית על קלטי ה-trace features (לא על הפלט של אות 3), עם ציון הסיכון של ה-PRM כמאפיין נוסף" |
| 30 | שלושת המודלים הלוגיסטיים מאומנים על Y_t (L2, C=1, class_weight=balanced) | נכון | עמ' 21: "C = 1.0, max_iter=500, class_weight=balanced" | — |
| 31 | ה-PRM הקפוא, "שלא קיבל תוויות", דורש סט כיול מתויג | לא מדויק (קל) | עמ' 9: "The Direct PRM remained frozen and received no CROP labels." | "שלא קיבל תוויות של CROP". ה-PRM עצמו הוא מודל שאומן בפיקוח, והמאמר אינו דן באימון שלו. החלק השני (צריך כיול מתויג) נכון |
| 32 | בזמן מבחן מחשבים s(X) בלי תוויות וחותכים בחריגה הראשונה מ-λ̂ | נכון | עמ' 4: "CROP computes s(X) from the unlabeled instance X and requires no step-level error labels" | — |
| 33 | "מעבירים את הסיפא למודל תיקון" | לא מדויק | עמ' 24–25, Appendix F: "The CROP-prefix mode supplies only the retained prefix." | בניסוי מודל התיקון מקבל את השאלה ואת הרישא בלבד, והסיפא מוסתרת ממנו. "ניתוב הסיפא לתיקון" הוא רק הניסוח הרעיוני (Abstract) |
| 34 | בניסוי התיקון זה עזר ב-5 מתוך 8 הגדרות (Table 2, Sec 5.4) | נכון | עמ' 11: "CROP improved over the best deployable non-CROP input in five of eight settings" | אפשר להוסיף: ההשוואה היא מול הקלט הטוב ביותר שאינו CROP. ההפרשים שליליים ל-Llama בשני הדומיינים ול-Gemma ב-GSM8K (עמ' 10) |
| 35 | אותו PRM ואותה קריאה: Qwen2.5-Math-PRM-7B, מעבר אחד, הסתברות לכל צעד | נכון (החלק על המאמר) | עמ' 20, "PRM risk proxy construction" | ההשוואה לשיטה שלנו מחוץ להיקף |
| 36 | אותם סוגי ערוצים: אנטרופיה, likelihood ו-confidence ברמת הצעד | נכון (החלק על המאמר) | עמ' 20: "For each completed step, we score the already-written step text under the model" | — |
| 37 | Trace+PRM משלב PRM עם ערוצי ודאות, "וזה בדיוק התפקיד של L-SML אצלנו" | נכון / מחוץ להיקף | עמ' 9, פריט 5 | ההשוואה ל-L-SML מחוץ להיקף |
| 38 | PRMBench: ‏5,000 תשובות, 15.9% צעדים שגויים (Table 4), וגם ProcessBench | נכון | עמ' 21, Table 4: "PRMBench 5,000 67,445 15.9" | — |
| 39 | "אותו סוג החלטה: הכרעה לכל תשובה אילו צעדים לסמן" | לא מדויק | עמ' 6: "The retained prefix is τ1:Mλ, and the uncertified suffix is τMλ+1:T" | CROP מכריע על נקודת חיתוך אחת לכל תשובה (רישא רציפה). אין בו סימון של צעדים בודדים, והסיפא אינה מוכרזת כשגויה (הקובץ עצמו אומר זאת בסעיף 5) |
| 40 | PRMBench, ‏Trace+PRM: ‏AUROC ‏.8336, שימור 21.8%, סיכון 5.0% (Table 6) | נכון | עמ' 32, Table 6: "Trace+PRM 0.8336 ± 0.0027 21.8 ± 0.9 5.0 ± 0.6" | — |
| 41 | PRMBench, ‏Direct PRM: ‏.8010, ‏4.8%, ‏0.6% | נכון | עמ' 32, Table 6: "Direct PRM 0.8010 ± 0.0049 4.8 ± 0.2 0.6 ± 0.1" | — |
| 42 | אין במאמר הסבר לסיכון 0.6%. ההשערה (לא נבדקה): הרשת של 0.01 גסה מדי | נכון: אין הסבר במאמר. ההשערה לא מוכרעת | חיפוש בטקסט המלא: אין דיון. כיתוב Table 6, עמ' 32: "prefix risk is not ranked because it is the controlled constraint" | המאמר אינו מכריע. תצפית שעולה בקנה אחד עם ההשערה בלי להוכיח אותה: גם ב-Math-Shepherd, על אותה רשת קבועה, ‏Direct PRM מקבל 0.9% סיכון ו-1.0% שימור, ואילו Trace+PRM מגיע לכ-5% (Table 6) |
| 43 | ב-96.4% מתשובות המבחן של PRMBench יש טעות מסומנת (Appendix H) | נכון | עמ' 28: "PRMBench contains an annotated error in 96.4% of held-out traces" | — |
| 44 | אצלם (LR על Y_t) ו(סט כיול מתויג) הם חובה, גם לערובה וגם לשילוב | נכון, עם דיוק | עמ' 4: "calibration labels select λ̂"; עמ' 21: אימון LR על Y_t | דיוק: כיול מתויג נדרש לכל אות. תוויות ל-LR נדרשות רק לאותות הנלמדים (Direct PRM נשאר בלי LR). החלק "אצלנו" מחוץ להיקף |
| 45 | ערוצי הוודאות מחושבים מ"המודל שיצר את הפתרון" (App B), שבניסויים העיקריים הוא Llama-3.1-8B-Instruct (Sec 5) | נכון בעיקרו | עמ' 9: "hidden-state summaries from Llama-3.1-8B-Instruct"; עמ' 20: "from the trace-generating language model"; עמ' 11: "generated by a Llama-family model" | המאמר מציין במפורש רק שהמאפיינים חושבו מ-Llama-3.1-8B-Instruct, ושהמחולל הוא "Llama-family". הוא לא אומר במפורש שהמחולל הוא Llama-3.1-8B-Instruct |
| 46 | (לא אומת) איזה מודל חישב את ה-trace features על PRMBench | המאמר אינו מכריע | עמ' 9 (Llama-3.1-8B-Instruct, בלי הבחנה בין דאטה-סטים) מול עמ' 20 ("trace-generating language model") | שני הניסוחים מתנגשים לגבי הדאטה-סטים החיצוניים, שאת המחולל שלהם המאמר אינו מתאר. נשאר פתוח, וההערה "לא אומת" מוצדקת |
| 47 | סף גלובלי אחד שמכוילים על פני תשובות רבות. כל מה שאחרי החריגה הראשונה "מסומן", בלי סימון צעדים בודדים | לא מדויק (קל) | עמ' 3, Fig. 2: "selects a domain-specific, conformally calibrated score threshold"; עמ' 19: "calibrate CROP separately within each dataset" | "סף אחד לכל דאטה-סט או דומיין". הסיפא "לא מאושרת" (uncertified) ומוסתרת, ואינה מוכרזת כשגויה |
| 48 | מדדים: שימור רישא, סיכון זיהום, step AUROC, ‏boundary deviation ודיוק תיקון (Sec 5) | נכון | עמ' 9: ארבעה מדדים. step AUROC מופיע ב-Sec 5.2, ב-Fig. 4 וב-Table 6 (עמ' 10) | — |
| 49 | אין PRMScore ואין פירוק לפי קטגוריות של PRMBench | נכון | חיפוש בטקסט המלא: אין "PRMScore" ואין "categor" | — |
| 50 | הערובה שולית ברמת התשובה | נכון | עמ' 13: "The reasoning instance is the exchangeable unit" | החלק "אצלנו" מחוץ להיקף |
| 51 | סיכום: CROP "חותך כל תשובה ברישא שהסיכוי שלה להכיל טעות מסומנת מוגבל ב-α" | לא מדויק | עמ' 6: "This is not a conditional guarantee for each realized prefix."; עמ' 13, Sec 8: אזהרה מפני פרשנות יתר | "חותך כל תשובה בסף אחד שמכויל כך שההסתברות השולית (על פני סט הכיול ודוגמה חדשה) שהרישא תכיל טעות מסומנת היא לכל היותר α" |
| 52 | CROP הוא שכבת כיול מפוקחת מעל כל ציון צעד | נכון | עמ' 1: "a verifier-agnostic calibration procedure"; הכיול דורש תוויות (עמ' 4) | — |
| 53 | Trace+PRM הוא שילוב לוגיסטי מפוקח של PRM ותלמטריה | נכון | עמ' 9 ועמ' 21 | — |
| 54 | (פרשנות שלי, לא נבדקה) ציון L-SML יכול לשמש אות סיכון בתוך CROP אם יש סט כיול מתויג | המאמר מכריע עקרונית, לא אמפירית | עמ' 7: "accepts any fixed step-level risk proxy function s"; עמ' 4: "it must be fixed before conformal calibration" | המסגרת מתירה זאת אם הציון מחושב מ-X בלבד, מכוון כך שערך גבוה פירושו סיכון גבוה, ונקבע לפני הכיול. המאמר אינו בודק אם זה יעיל. ההשוואה לשיטה שלנו מחוץ להיקף |
| 55 | ה-PDF הוא 39 עמודים | נכון | ספירת עמודים ב-pymupdf: 39 | — |

### התיקונים החשובים ב-01_CROP.md
1. **סעיף 6, משפט הסיכום (שורה 51):** הניסוח "הרישא שהסיכוי שלה להכיל טעות מוגבל ב-α" נשמע כמו ערובה לכל תשובה, ואת זה המאמר שולל במפורש (עמ' 6, Sec 7, Sec 8). צריך לכתוב "הסתברות שולית".
2. **Trace+PRM (שורה 29):** זו רגרסיה לוגיסטית על המאפיינים הגולמיים של trace features, עם ציון ה-PRM כמאפיין נוסף. היא לא בנויה על הפלט של אות 3.
3. **"מעבירים את הסיפא למודל תיקון" (שורה 33):** בניסוי מודל התיקון מקבל רק את השאלה ואת הרישא. הסיפא מוסתרת ממנו.
4. **"אותו סוג החלטה" (שורה 39) ו"כל מה שאחרי החריגה מסומן" (שורה 47):** ‏CROP בוחר נקודת חיתוך אחת. הסיפא "לא מאושרת", ולא מוכרזת כשגויה. הסף נקבע לכל דאטה-סט או דומיין, ולא אחד גלובלי.
5. **ה-PRM "שלא קיבל תוויות" (שורה 31):** לדייק ל"לא קיבל תוויות של CROP".
6. **9.7% (שורה 19):** לציין שהמספר הוא על Arithmetic ו-GSM8K מאוחדים, ולא על PRMBench.
7. **הסיכון 0.6% (שורה 42):** המאמר באמת אינו מסביר אותו. ההשערה על הרשת הגסה נשארת לא מוכרעת, אבל יש לה תמיכה עקיפה (Math-Shepherd ‏Direct PRM, ‏0.9%).

---

## קובץ 2: 02_FUSE.md מול "FUSE - Ensembling Verifiers with Zero Labeled Data.pdf"

| מס' | הטענה (בקיצור) | פסיקה | ראיה (עמוד, מיקום, ציטוט) | תיקון מוצע |
|---|---|---|---|---|
| 1 | מחברים: Lee, Ma, Zhao, Nair, Spector, Cohen, Candès, מ-Stanford ו-Google | נכון | עמ' 1: "1Stanford University 2Google" | — |
| 2 | arXiv 2604.18547v1, ‏20 באפריל 2026 | נכון | עמ' 1, שוליים: "arXiv:2604.18547v1 [stat.ML] 20 Apr 2026" | — |
| 3 | (לא אומת) שיוך ל-ICML 2026. ב-PDF כתוב רק "Preprint. April 21, 2026" | נכון. המאמר אינו מכריע | עמ' 1: "Preprint. April 21, 2026." חיפוש: "ICML" מופיע רק בהפניה ל-workshop של 2024 | ההערה מוצדקת. רמז חלש בלבד: יש במאמר "Impact Statement", שאופייני לתבנית של ICML. זו לא הוכחה |
| 4 | ב-Best-of-N דוגמים N תשובות ו-verifier (‏LLM judge או reward model) בוחר אחת | נכון | עמ' 1, §1.1: "N responses are sampled independently ... scores from a verifier are used" | — |
| 5 | איכות ה-verifiers משתנה בין verifiers ואפילו בין שאלות | נכון | עמ' 1: "verifier strength can be query-dependent" | — |
| 6 | ממוצע או majority vote נותנים לחלש אותו משקל כמו לחזק, ותוויות יקרות | נכון | עמ' 1: "treat scores from unreliable verifiers on par with those given by their stronger" | — |
| 7 | המטרה: לבחור תשובה נכונה מתוך V (‏N×m) בלבד (§1.1, §1.3) | נכון | עמ' 2, §1.3: "Our goal is to select, using the score matrix V alone" | — |
| 8 | אינטואיציה: משתנה סמוי y כמו ב-Dawid-Skene. תחת CI ההסכמה נובעת מ-y, ולכן המומנטים חושפים את הדיוק | נכון (פרשנות עקבית) | עמ' 2–3: Dawid & Skene מצוטטים; "ensure that the problem is identified" | — |
| 9 | שיטת המומנטים של Jaffe 2015 (Theorem 2.3) | נכון | עמ' 3: "Theorem 2.3 (Jaffe et al. (2015, Lemmas 1, 2, and 4))" | — |
| 10 | Assumption 2.1: רוב ה-verifiers טובים מאקראי | נכון | עמ' 2: "more than m/2 of the values {πj} are larger than 1/2" | — |
| 11 | TCI: כל שלשה בלתי תלויה בהינתן y (Assumption 2.2, eq. 10) | נכון | עמ' 2, Assumption 2.2; עמ' 11, App A.1, Eq. (10) | — |
| 12 | (i) האיברים מחוץ לאלכסון של Σ שווים ל-uuᵀ, עם u=√(1−b²)(2π−1) (eq. 1) | נכון | עמ' 3, Eq. (1) | — |
| 13 | π הוא balanced accuracy ו-b הוא חוסר האיזון בין המחלקות | נכון | עמ' 2: "denote the balanced accuracy"; עמ' 3: "b := P(y = 1) − P(y = −1) denote the class imbalance" | — |
| 14 | (ii) האיברים מחוץ לאלכסון של T שווים ל-w⊗w⊗w, עם w=(−2b(1−b²))^{1/3}(2π−1) (eq. 2) | נכון | עמ' 3, Eq. (2) | — |
| 15 | (iii) מהיחס בין u ל-w מחלצים את b, ומשם ψ ו-η (eq. 3) | נכון | עמ' 3: "these vectors differ only by a scale factor depending on b" | — |
| 16 | חלק (i) הוא בדיוק ליבת SML של Parisi 2014: וקטור עצמי מוביל של הקווריאנס מחוץ לאלכסון | לא נמצא במאמר (במפורש) | המאמר מציין את Parisi 2014 כבסיס (עמ' 2) ומציין שההוכחות של Parisi ו-Jaffe תקפות (עמ' 11). המילים "eigenvector" ו-"SML" אינן מופיעות | לסמן כ"פרשנות שלי מהספרות". הטענה נכונה בספרות, אבל המאמר אינו אומר אותה |
| 17 | Jaffe 2015 מוסיף מומנט שלישי כדי לזהות את b ולהפריד בין ψ ל-η | נכון | עמ' 3, הפסקה שאחרי Theorem 2.3 | — |
| 18 | ה-ensemble של Jaffe הוא MLE תחת Joint CI, עם משקלי log-odds (eq. 12, Appendix B) | נכון | עמ' 12, App B: "they show that the maximum likelihood estimate (MLE) of y(q, r) is given by" | — |
| 19 | ‏L-SML (‏Jaffe et al. 2016) מטפל בתלות על ידי מידול מבנה של קבוצות תלויות | נכון בעיקרו | עמ' 21, E.8: "An extension of Jaffe et al. (2015) to settings with structured conditional dependence." | השם "L-SML" אינו מופיע במאמר. המאמר אומר רק "structured conditional dependence" |
| 20 | סטטיסטי TCI (Proposition 2.4, eq. 4) | נכון | עמ' 4, Eq. (4) | — |
| 21 | (גזירה שלי) תחת TCI היחס T/Σ תלוי רק ב-j3 ופרופורציונלי ל-(2π_{j3}−1). לכן השונות על פני הזוגות היא אפס | נכון (הגזירה תקפה) | מ-(1)–(2): T/Σ = (β³/γ²)(2π_{j3}−1) = −2b(2π_{j3}−1), בתנאי ש-Σ_{j1j2}≠0 | אפשר לכתוב את הקבוע במפורש: היחס שווה ל-−2b(2π_{j3}−1) |
| 22 | הסכום Ŝ מודד הפרה של TCI בלי תוויות | נכון, עם הסתייגות | עמ' 4: "a large value of the left-hand side of (4) certifies that TCI does not hold" | ערך גבוה מעיד על הפרה, אבל ערך אפס אינו מוכיח TCI: זה תנאי הכרחי בלבד. בפועל המכנה נחתך מלמטה (עמ' 4, הערת שוליים 1) |
| 23 | בינאריזציה sign(v_j−τ_j), ספים ב-coordinate descent שממזער את Ŝ(τ), ואז MoM על Ṽ (§2.2) | נכון | עמ' 4: "find τ⋆ which (approximately) minimizes Ŝ(τ) ... via coordinate descent" | — |
| 24 | ב-Appendix D מוזכרת Box-Cox כחלופה | נכון | עמ' 13, "Alternative Transformations": "use Box-Cox transformations" | — |
| 25 | posterior לכל שלשה (Prop C.1, eq. 13), ממוצע על כל השלשות (eq. 9), וכך עוקפים את Joint CI | נכון | עמ' 12, Eq. (13); עמ' 5, Eq. (9); עמ' 2: "allows us to avoid assuming joint conditional independence" | — |
| 26 | LR על ה-V המקורי, שממקסם את Âcc(θ)=Σ(2p̂−1)f̂_θ(V_i•) (eq. 7) | נכון | עמ' 5: "logistic regression classifiers operating on the design matrix V"; Eq. (7) | — |
| 27 | verifiers שה-BA המשוער שלהם נמוך מ-½ מושמטים (Appendix D) | נכון, עם דיוק | עמ' 13: "drop verifiers with estimated balanced accuracy less than 1/2 before proceeding with posterior estimation" | ההשמטה קורית אחרי הטרנספורמציה ולפני חישוב ה-posteriors, ולא אחרי אימון ה-LR. זו היוריסטיקה של ברירת מחדל ("informally motivated") |
| 28 | N=100 ועד 33 verifiers (‏Saad-Falcon); ‏N=50 ו-7 ב-HLE; ‏N=50 ו-9 ב-IMO (§3) | נכון | עמ' 6: "100 generations per question"; "up to 33"; "50 generations ... seven"; "50 generations ... 9 open-source" | — |
| 29 | הציונים מנורמלים ל-[−1,1] (Appendix A.2, Appendix E.1 footnote 5) | לא מדויק (בהפניה) | עמ' 11, App A.2, הערה 3: "we re-scale verifier scores to lie in [−1, 1] before applying FUSE". עמ' 14, הערה 5: נרמול min-max מוגדר ל-naive ensemble | ההפניה ל-FUSE היא הערת שוליים 3 ב-A.2. הערה 5 מתארת min-max ל-naive ensemble, והמאמר אינו אומר במפורש ש-FUSE משתמש באותו min-max |
| 30 | מצבים: query-conditional (Table 5) או batched, שבו המטריצות משורשרות אנכית (Appendix D) | נכון | עמ' 2, §1.3 (הגדרה); עמ' 15, Table 5, עמודה "Question-Conditional"; עמ' 13: "we vertically concatenate score matrices" | ההגדרה של query-conditional נמצאת ב-§1.3. Table 5 רק מסמן את FUSE כ-"Yes" |
| 31 | (לא מצוין במפורש) באיזה מצב הורצו Tables 1, 3 ו-4 | נכון חלקית | אין הצהרה לכל טבלה. אבל: Table 5 מסמן את FUSE "Question-Conditional: Yes"; עמ' 2: "we restrict our attention to the single-query case"; עמ' 8: "the question-conditional version of FUSE is immune" | הניסוח מדויק מילולית, אבל אין במאמר אף תוצאה במצב batched, וכל הסימנים מצביעים על query-conditional. מוטב: "לא מוצהר לכל טבלה, אבל כל הסימנים מצביעים על query-conditional" |
| 32 | הפלט: התשובה עם f_θ* הגבוה ביותר (Algorithm 1) | נכון | עמ' 6, Algorithm 1, שורה 7 | — |
| 33 | FUSE אינו משתמש בתוויות. הן משמשות רק להערכה ולבסיסי ההשוואה ה-semi-supervised, עם 5% מהשאלות (Table 1 caption) | נכון, עם השלמה | עמ' 7, Table 1: "labels for all 100 responses to 5% of questions" | השלמה: גם בסיסי ה-oracle (‏Pass@k, ‏OBV) משתמשים בתוויות. ב-Table 2, בהגדרת GPQA-only, ניתנות תוויות ל-25 שאלות GPQA |
| 34 | מול baselines מפוקחים FUSE מנצח ב-27 מתוך 40 (§3.1) | נכון | עמ' 6: "FUSE wins 27 out of 40 comparisons against supervised baselines" | — |
| 35 | ‏GPQA Diamond 70B: ‏64.4 מול 64.1 של WEAVER (Table 1) | נכון | עמ' 6: "64.4% on GPQA Diamond for FUSE vs 64.1% for WEAVER" | — |
| 36 | ב-HLE ‏FUSE מגיע ל-54.3, מעל כל השאר (Table 3) | נכון | עמ' 7, Table 3: "FUSE 54.3%"; הבא אחריו OBV עם 53.5% | — |
| 37 | אותה משפחה: fusion ללא תוויות, משתנה סמוי בינארי, CI ומבנה rank-one מחוץ לאלכסון. התלות מקלקלת את ההערכה | נכון (החלק על המאמר) | עמ' 3–4; עמ' 4: "suggesting that its assumptions are untenable in realistic LLM verification" | ההשוואה ל-L-SML מחוץ להיקף |
| 38 | ב-Fig. 4 רואים קורלציות מותנות של עד כ-0.7 בין reward models | נכון | עמ' 16, Fig. 4: מטריצות עם ערכים של 0.7 (למשל בזוג EurusPRMStage2 min/max בהינתן y=−1) | לציין שאלה ממוצעי קורלציה מותנית על נתוני MMLU-Pro בלבד ("Average conditional correlations in MMLU-Pro data") |
| 39 | ‏QwenPRM ו-EurusPRM, כל אחד ב-min/max/avg, נמצאים במאגר (Table 6) | לא מדויק | עמ' 23, Table 6: ‏EurusPRMStage1 ו-EurusPRMStage2 בנפרד. "QwenPRM max reward model No Yes"; "EurusPRMStage1 avg reward model Yes No" | "QwenPRM ב-min/max/avg, ו-EurusPRM בשני שלבים (Stage1, Stage2), כל אחד ב-min/max/avg. לא כל הגרסאות נמצאות גם בהגדרת 8B וגם בהגדרת 70B" |
| 40 | ה-ensemble של Jaffe 2015 נופל מ-naive ensemble ב-7 מתוך 10 הגדרות (§2.1, Fig. 3) | נכון | עמ' 4: "under-performs a simple naive ensemble in 7 out of 10 settings" | — |
| 41 | FUSE משפר על Jaffe ב-+1.7 עד +12.3 נקודות (Fig. 3) | נכון | עמ' 5, Fig. 3: הערכים +6.5, +6.3, +12.3, +2.7, +1.7, +6.4, +8.1, +2.3, +2.6, +5.1 | — |
| 42 | ב-Table 9, כל 10 הערכים של "DCL" נמוכים מ-naive ensemble שב-Table 1 | נכון | עמ' 24, Table 9, ‏DCL: 8B ‏.2832/.3279/.6804/.7910/.6199; 70B ‏.5601/.5606/.8693/.9013/.8220. עמ' 7, ‏Naive: 8B ‏37.8/38.4/75.2/81.9/67.2; 70B ‏60.1/57.6/92.4/92.4/87.0 | — |
| 43 | (לא אומת) ש-DCL הוא Jaffe et al. 2016. "מוסק מסדר הרשימה ב-Appendix E.8" | לא מדויק בנימוק. המאמר אינו מכריע | עמ' 21, E.8: הסדר הוא DS, ‏Jaffe 2016, ‏GMM. עמ' 24, Table 9: הסדר הוא DS, ‏GMM, ‏DCL | הסדרים אינם תואמים. ההסקה נעשית בדרך האלימינציה: DCL היא השיטה היחידה שלא נקראת בשמה. כותרת ההפניה "Unsupervised ensemble learning with dependent classifiers" (עמ' 10) מתאימה לראשי התיבות. המאמר אינו מגדיר את הקיצור |
| 44 | ב-IMO Shortlist ‏FUSE משתווה ל-naive ensemble (63.8 מול 63.8, Table 4) | נכון | עמ' 8, Table 4: "Naive Ensemble 63.8%" ו-"FUSE 63.8%" | — |
| 45 | ה-verifiers ב-IMO הומוגניים וכמעט בלתי תלויים (§3.3, Fig. 6–7) | נכון | עמ' 7: "substantial homogeneity in verifier strength and near-conditional independence"; עמ' 19: "uniformly mild and positive" | אפשר להוסיף: לפי Fig. 7 ה-BA "only slightly better than random", חוץ מ-gpt-oss-20b |
| 46 | "גם אצלם הממוצע הוא תקרה כשה-views הומוגניים" | נכון (במונחי המאמר) | עמ' 8: "the naive ensemble baseline is effectively an oracle baseline" | — |
| 47 | יחידת ההחלטה היא תשובה שלמה (argmax אחד מתוך N), ואין החלטת "אין שגיאה" | נכון | עמ' 5: "i⋆ := arg max fθ⋆(Vi•)" | החלק "אצלנו" מחוץ להיקף |
| 48 | ה-views הם m מודלים שונים בהרצות נפרדות, והעלות היא המגבלה העיקרית שהם מציינים (§5) | נכון | עמ' 8: "The primary limitation of our method is its computational and logistical efficiency" | — |
| 49 | המומנטים מחושבים על N תשובות i.i.d. לאותה שאלה (§1.3) | נכון | עמ' 2: "(r1, . . . , rN) are sampled i.i.d. from G(q)" | — |
| 50 | ה-aggregator: ‏LR שמאומן על pseudo-labels | נכון בקירוב | עמ' 5, Eq. (7); עמ' 13: "once pseudo-labels have been acquired" | דיוק: θ נבחר כך שימקסם את Âcc, שבו כל תחזית ±1 משוקללת ב-(2p̂−1). זה לא fit רגיל של LR על תוויות. סעיף 2 בקובץ מתאר זאת נכון |
| 51 | הנתונים: GPQA, ‏MATH500, ‏MMLU(-Pro), ‏HLE ו-IMO, בלי PRMBench ובלי תוויות ברמת הצעד | נכון | עמ' 6, §3. חיפוש: "PRMBench" אינו מופיע | להוסיף GPQA Diamond. ה-PRMs משמשים רק כ-verifiers ברמת התשובה (min/max/avg) |
| 52 | סיכום: FUSE מדגים שאפשר לעקוף את כישלון ה-SML/MoM הקלאסי מול הממוצע | נכון בעיקרו | עמ' 4 (Jaffe < naive ב-7/10); עמ' 6: "always outperforms the natural unsupervised baseline of naive ensemble" | המאמר מדגים את הכישלון על ה-ensemble של Jaffe 2015 (‏MoM ו-JCI), ולא על SML של Parisi. FUSE עדיף על naive בכל 10 ההגדרות של Table 1 וב-HLE, ומשתווה לו ב-IMO |
| 53 | אי-עקביות: ב-caption של Fig. 3 כתוב "Llama 3.3 8B", וב-§3 כתוב "Llama 3.1 8B" | נכון | עמ' 5: "generator models are Llama 3.3 8B Instruct and Llama 3.3 70B"; עמ' 6–7, §3 ו-Table 1: "Llama 3.1 8B Instruct"; עמ' 15, E.2: גם "Llama 3.1 8B" | אפשר להוסיף אי-עקביות נוספת: אותם נתונים מיוחסים לפעמים ל-Saad-Falcon 2025a ולפעמים ל-2025b, ו-WEAVER מיוחס ל-2025a ב-§3 ול-2025b ב-Fig. 1, ב-E.1 וב-Table 5 |

### התיקונים החשובים ב-02_FUSE.md
1. **DCL (שורה 43):** הנימוק "מסדר הרשימה" שגוי, כי סדר הרשימה ב-E.8 שונה מסדר העמודות ב-Table 9. הזיהוי כ-Jaffe 2016 נעשה בדרך האלימינציה ובעזרת כותרת ההפניה. המאמר אינו מגדיר את הקיצור, ולכן "לא אומת" נשאר.
2. **PRMs במאגר (שורה 39):** ‏EurusPRM מופיע בשני שלבים, ולא כל גרסה נמצאת גם ב-8B וגם ב-70B.
3. **SML ווקטור עצמי (שורה 16):** הקשר הזה אינו במאמר. לסמן אותו כפרשנות מהספרות.
4. **נרמול ל-[−1,1] (שורה 29):** ההפניה הנכונה היא A.2, הערה 3. הערה 5 מתארת min-max ל-naive ensemble.
5. **מצב ההרצה (שורה 31):** לא מוצהר לכל טבלה, אבל אין במאמר תוצאות batched ו-Table 5 מסמן את FUSE כ-question-conditional. כדאי לרכך ל"כנראה query-conditional".
6. **סדר ההשמטה (שורה 27):** ה-verifiers החלשים מושמטים לפני חישוב ה-posteriors, ולא אחרי ה-LR.
7. **הסיכום (שורה 52):** המאמר מדגים כישלון של Jaffe 2015 (‏MoM עם JCI). הוא לא בודק SML.
8. **אי-עקביות (שורה 53):** אומתה. יש עוד אחת, בהפניות ל-Saad-Falcon 2025a ול-2025b.

---

## סיכום כללי
- **01_CROP.md:** מתוך 55 טענות, 4 "לא מדויק" מהותיות (שורות 29, 33, 39, 51) ו-3 "לא מדויק" קלות (שורות 31, 47, והדיוק בשורה 19). לא מצאתי טענה שגויה לגמרי. כל המספרים מ-Table 4, ‏Table 6 ו-Appendix H נכונים.
- **02_FUSE.md:** מתוך 53 טענות, 3 "לא מדויק" (שורות 29, 39, 43), טענה אחת "לא נמצא במאמר" (שורה 16), ועוד כמה דיוקים. כל המספרים מ-Tables 1, 3, 4 ו-9 ומ-Fig. 3 נכונים.
- שתי ההערות "לא אומת" ב-CROP (המודל שחישב את ה-trace features על PRMBench, וההסבר ל-0.6%) נשארות פתוחות: המאמר אינו מכריע בהן. מתוך שתי ההערות ב-FUSE, ‏ICML נשארת פתוחה, וב-DCL התשובה המסתברת היא Jaffe 2016, אבל בנימוק שונה מזה שבקובץ.
