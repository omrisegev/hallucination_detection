# פגישת מנחים -- 18 במאי, 2026
## הערות דובר (Speaker Notes) - מתורגם

---

## שקף 1 -- כותרת (Title)

**מה להגיד:**
> "בוקר טוב. היום אני רוצה להציג לכם את כל מה שעשינו מאז הפגישה האחרונה. התזה התקדמה משמעותית: כעת יש לנו תוצאות בחמישה domains -- math reasoning, science MCQ, factual QA (תוצאה שלילית שאני רוצה להסביר כראוי), retrieval-augmented generation (RAG), ועכשיו גם multi-step agent loops. אני אציג plots והשוואות למתחרים (competitors) עבור כל domain. בסוף אתאר שני פיילוטים (pilots) חדשים שאנחנו מריצים ומה נותר לעשות לפני שהתזה תושלם."

**שאלה צפויה מעופר:** "איזה כיוון הוא בעל העניין התיאורטי הרב ביותר?"
> RAG ו-agentic -- אלו התחומים שבהם ה-signal mechanism שונה ממה ש-EPR חוזה. ה-meta-analysis מראה שדירוג ה-features משתנה: ב-math, ה-EPR שולט; ב-RAG, ה-rpdi (עליית entropy בסוף ה-trace) וה-spectral_entropy תופסים פיקוד. ההסטה הזו מרמזת שתהליכים generative שונים פועלים כאן.

---

## שקף 2 -- מה אנחנו מזהים? (H(n) traces)

**מה להגיד:**
> "הרעיון המרכזי: במהלך text generation, לכל token יש probability distribution על פני כל ה-vocabulary. ה-entropy של ההתפלגות הזו היא H(n) = -sum_v p_v log p_v. אנחנו אוספים את המספר הזה עבור כל token שנוצר -- זהו ה-entropy trajectory. בשקף הזה תוכלו לראות 2x3 דוגמאות: תשובות נכונות למעלה, ותשובות שגויות (hallucinating) למטה. תשובות נכונות מראות trajectories חלקות עם variance נמוך יותר. תשובות שגויות מראות יותר instability -- קפיצות פתאומיות ו-variance bursts.
>
> השיטה היא gray-box: אנחנו צריכים token probabilities מ-forward pass בודד. אנחנו לא מסתכלים על ה-output text בכלל. אנחנו קוראים את ה-uncertainty signal הפנימי של המודל."

**נקודה מרכזית לאמיר:** "אנחנו קוראים signal שהמודל תמיד מייצר, עם אפס חישוב נוסף מעבר ל-inference."

---

## שקף 3 -- PSD מראה Spectral Signature

**מה להגיד:**
> "הנה הסיבה ש-'spectral' חשוב. זהו ה-Average Power Spectral Density של H(n) על פני דוגמאות רבות מ-MATH-500. קו כחול: תשובות נכונות. קו כתום: שגויות. ניתן לראות בבירור שתשובות שגויות מרכזות יותר כוח בתדרים גבוהים (HIGH frequencies) -- כלומר ל-entropy signal יש יותר bursts קצרי-טווח ו-instability. תשובות נכונות מרכזות כוח בתדרים נמוכים (LOW frequencies) -- ה-reasoning הוא חלק ומובנה.
>
> EPR, ה-baseline מהמאמר המקורי, משתמש רק בממוצע של H(n) -- זהו רכיב ה-DC של ה-PSD. אנחנו משתמשים בכל ה-spectrum: band powers, spectral entropy, dominant frequency, STFT. אנחנו משחזרים מידע ש-EPR זורק."

---

## שקף 4 -- Feature Library (16 features)

**מה להגיד:**
> "יש לנו 16 spectral features בארבע משפחות. תרשים העמודות מראה AUROC אישי ב-MATH-500 / Qwen-7B. כמה נקודות בולטות: sw_var_peak -- שיא ה-variance בחלון זז (sliding-window) -- תופס instability מקומית. cusum_max -- סטטיסטיקת change-point של סכום מצטבר -- מזהה שינויי regime ב-entropy signal. שני אלו הם הכי אוניברסליים: הם מופיעים ב-top 3 עבור כל domain שבדקנו.
>
> ה-Nadler fusion המלא של ה-subset הטוב ביותר משיג 90.0% AUROC ב-MATH-500. אלו 16 candidate features; Nadler בוחר את ה-best conditionally-independent subset באופן אוטומטי."

---

## שקף 5 -- Feature Correlation Topology

**מה להגיד:**
> "זוהי מטריצת Spearman correlation על פני כל 7,001 הדגימות מ-5 domains. שימו לב למבנה ה-cluster: epr, hurst, ו-pe_mean כולם ב-correlation גבוה -- אלו דרכים שונות למדוד את אותו דבר, ה-global average entropy level. ה-spectral_entropy וה-stft_spectral_entropy גם הם ב-correlation -- אותו signal, חלוקה שונה לחלונות (windowing).
>
> התצפית הקריטית: sw_var_peak ו-cusum_max הם ORTHOGONAL ל-EPR cluster. זו בדיוק הסיבה ש-Nadler fusion מרוויח signal -- הוא מוצא linear combination של orthogonal views שמסכימים על האם התשובה נכונה, אך חלוקים באופן שבו הם מבטאים זאת. pe_min מבודד לחלוטין מכל cluster, מה שמסביר מדוע הוא דורג אחרון (17/17) בכל domain והוסר."

**לעופר:** "מבנה ה-orthogonality כאן הוא ההצדקה התיאורטית לגישת ה-multi-view -- אתה בעצם עושה matrix factorization שבו רכיבים עצמאיים (independent components) כל אחד תורם signal ייחודי על ה-hidden binary variable (נכון/שגוי)."

---

## שקף 6 -- Feature Importance Per Domain

**מה להגיד:**
> "אלו ציוני Random Forest Gini importance, תרשים עמודות אחד לכל domain, מתוך ה-meta-analysis dataset (7,001 דגימות). הפאנל הימני התחתון מסכם את ה-cross-domain verdict. ה-cusum_max וה-sw_var_peak הם ב-top-3 בכל חמשת ה-domains -- אלו ה-universally robust features. אבל הדירוגים משתנים משמעותית: ב-math וב-GSM8K, ה-epr הוא ה-feature הטוב ביותר. ב-Factual QA וב-RAG, ה-rpdi (עליית entropy בסוף ה-trace) הופך לדומיננטי. ההסטה הזו מעניינת מהותית: rpdi מזהה עליית entropy בסוף ה-generation, וזה מה שהיית מצפה כשמודל עומד לייצר ungrounded text. ב-math, הדפוס הזה לא קיים -- ה-uncertainty נמצא באמצע ה-reasoning."

---

## שקף 7 -- איך זה עובד: Math Domain (דוגמה)

**מה להגיד:**
> "בואו נהפוך את השיטה לקונקרטית. משמאל: המבנה של דוגמת MATH-500. המודל מקבל שאלה -- למשל, פתור x^2+3x+2=0 -- ומייצר chain-of-thought (CoT) מלא של ה-reasoning trace, ואז נותן את התשובה הסופית. H(n) מכסה כל token שנוצר, בדרך כלל 500 עד 2,000 tokens סך הכל.
>
> מימין רואים איך H(n) נראה בפועל עבור תשובה נכונה. האזור הכתום הוא ה-chain-of-thought; האזור הירוק הוא התשובה. שימו לב: במהלך ה-reasoning יש מדי פעם entropy spikes (המודל לא בטוח בנקודות פיצול), ואז מקטע התשובה הוא ב-low-entropy ויציב מאוד. cusum_max מזהה את ה-regime shifts באמצע ה-reasoning. ה-sw_var_peak תופס את אותם local variance spikes.
>
> מודל שעושה hallucination יראה דפוס שונה: spikes תכופים וגדולים יותר, פחות מבנה, ו-final entropy גבוה יותר בתשובה."

---

## שקף 8 -- תוצאות Math + טבלת השוואה

**מה להגיד:**
> "המספרים המרכזיים: MATH-500 עם Qwen-7B ב-T=1.0 נותן 90.0% AUROC עם Nadler fusion. ב-GSM8K עם Llama-3.1-8B, אנחנו מקבלים 76.0%. המתחרה הכי קרוב שפורסם ומשתמש בגישה דומה -- LapEigvals, שגם משתמש ב-spectral features של hidden states -- מקבל 72.0% על אותו מודל ומשימה. זהו שיפור של +4 pp (unsupervised vs unsupervised).
>
> טבלת ההשוואה מראה מה הופך את זה לרלוונטי: שתי השיטות לא משתמשות ב-labels ודורשות רק forward pass בודד. ההבדל הוא במה שאנחנו מחשבים: LapEigvals משתמש ב-hidden-state Laplacian eigenvalues; אנחנו משתמשים ב-entropy trajectory spectrum, שדורש פחות גישה -- רק token probabilities, לא hidden states.
>
> הסתייגות אחת: T=1.5 נותן 96.6% -- זהו ceiling artifact. בטמפרטורה גבוהה יותר המודל עושה יותר טעויות, אז ה-class ratio משתנה וה-AUC מתנפח. ההשוואה הכנה היא T=1.0 = 90.0%."

---

## שקף 9 -- תוצאות GPQA Diamond

**מה להגיד:**
> "GPQA Diamond הוא 198 שאלות מדעיות ברמת דוקטורט עם 4 אפשרויות (multiple-choice). זו המשימה הקשה ביותר שבדקנו. עם מודלים של 7B, הדיוק במשימה הוא רק 30-40% -- בקושי מעל אקראי לשאלה עם 4 אפשרויות. כשהמודל צודק רק באחת מכל שלוש שאלות, פשוט אין מספיק דוגמאות נכונות כדי שהגלאי (detector) יוכל להבחין ביניהן.
>
> מעבר ל-Qwen2.5-72B-AWQ שיפר את הדיוק ל-40.4% ואת ה-AUC ל-69.0%, רווח של +3.6 pp. המגמה ברורה: ככל שאיכות המודל משתפרת, הזיהוי (detection) משתפר. השיטה מוגבלת על ידי המודל שבבסיסה, לא על ידי הגלאי שלנו.
>
> עדיין לא עברנו את סף ה-70%. הכיוון הוא לנסות מודלים גדולים עוד יותר או ייעודיים יותר. אין מתחרה spectral שפורסם על GPQA, כך שאנחנו לא יכולים להשוות ישירות -- למרות שאם מישהו ינסה את אותה גישה עם מודל של 70B+, התחזית שלנו היא שהם יראו את אותה מגמה."

---

## שקף 10 -- תוצאה שלילית: Factual QA

**מה להגיד:**
> "זוהי תוצאה שלילית חשובה ואני רוצה להסביר אותה כראוי. TriviaQA ו-WebQuestions הן משימות factual recall. המודל נשאל 'מהי בירת צרפת?' -- התשובה היא פשוט 'Paris'. ה-Direct-answer traces הם באורך 20-50 tokens. זה לא מספיק לחלוטין ל-spectral analysis; ל-FFT על 20 tokens אין frequency resolution.
>
> הוספנו chain-of-thought (CoT) prompting כדי להאריך את ה-traces ל-200-500 tokens. זה לא עזר: TriviaQA CoT נותן 53.6% (קרוב לאקראי), WebQ נותן 61.9%. חשוב לציין, ה-EPR baseline על תשובות ישירות -- רק ה-mean entropy -- נותן 79.1% ו-71.8%. כך שהגישה המורחבת שלנו גרועה משמעותית מה-baseline הפשוט.
>
> הסיבה: chain-of-thought מחליק (SMOOTHS) את ה-entropy trajectory במשימות factual recall. המודל בעצם חוזר על אותו טקסט בביטחון גבוה בניסוחים שונים, כך שאין מבנה תדרים סיסטמטי. זהו הבדל מבני ממשימות reasoning. ה-Factual recall הוא retrieval, לא generation.
>
> זה לא כישלון -- זהו boundary condition שמחדד את היקף (scope) התזה. עכשיו אנחנו יודעים: spectral features מזהים generative uncertainty במהלך REASONING, לא factual uncertainty במהלך recall. אלו תהליכים קוגניטיביים שונים."

**לברכה:** "ה-boundary condition הזה בעצם שימושי ל-LTT -- אנחנו יכולים עכשיו לתכנן את ה-calibration guarantee במיוחד עבור משימות מסוג reasoning."

---

## שקף 11 -- RAG: איך זה עובד

**מה להגיד:**
> "עכשיו נדבר על RAG -- retrieval-augmented generation. המשימה היא: בהינתן שאלה, המודל שולף מסמכים רלוונטיים ומייצר תשובה עם citation markers [1], [2] וכו'. השאלה שלנו היא: עבור כל משפט ב-output של המודל, האם הוא באמת מבוסס (grounded) במסמכים שנשלפו, או שהמודל פשוט המציא אותו?
>
> החדשנות המרכזית עבור RAG: אנחנו לא מנתחים את ה-trace המלא. אנחנו חותכים (SLICE) את H(n) בגבולות ה-citation. כל הצהרה בין citation markers מקבלת entropy subsequence משלה, spectral feature vector משלה, וציון Nadler fusion משלה. כל הצהרה מתויגת כ-GROUNDED אם הפסקה המצוטטת מופיעה ב-gold supporting facts, ו-UNGROUNDED אחרת.
>
> תסתכלו על הדוגמה: לשתי ההצהרות המבוססות (grounded) יש slices חלקים עם entropy נמוך יותר. ההצהרה הלא מבוססת (ungrounded) -- 'לכן, הם לא הוקמו באותה מדינה' -- היא מסקנה מומצאת שלא מופיעה באף אחד מהמסמכים שנשלפו. ה-spectral features שלה שונים: rpdi גבוה יותר (ה-entropy עולה לקראת סוף ההצהרה), ו-spectral_entropy גבוה יותר באופן כללי.
>
> זו הסיבה ש-rpdi דומיננטי ב-RAG אבל לא ב-math: ב-RAG, הגבול (BOUNDARY) בין טקסט מבוסס ללא מבוסס הוא המקום שבו נמצא ה-signal. ב-math, זה המבנה של כל ה-reasoning chain."

---

## שקף 12 -- תוצאות RAG

**מה להגיד:**
> "תוצאות על פני 16 תאים -- ארבעה מודלים מול ארבעה datasets. ה-heatmap מציג AUC לכל תא. מספרים מרכזיים: התא הטוב ביותר הוא llama8b / HotpotQA עם 87.7%, שזו התוצאה הכוללת הטובה ביותר בתזה עד כה. Qwen-7B / 2WikiMultiHopQA הוא 80.5%. החציון על פני כל 16 התאים הוא בערך 72.8%; 12 מתוך 16 תאים עוברים את סף ה-70%.
>
> השוואת המתחרים הקריטית: LOS-Net (arXiv:2503.14043) הוא גלאי hallucination עם supervision עבור RAG, שאומן עם labeled data. ב-HotpotQA / Mistral-7B הוא משיג 72.92%. ה-Qwen-7B / HotpotQA שלנו ללא שום labels בזמן הבדיקה נותן 79.5% -- זהו שיפור של +6.6 pp. התא הטוב ביותר שלנו (llama8b) נותן 87.7% -- שיפור של +14.8 pp. ניצחנו שיטה מבוססת supervision בלי לראות שום labels.
>
> ה-sanity check שחשוב: trace_length לבדו נותן 50.8% -- כמו אקראי. כך שה-signal הוא באמת spectral, ולא רק 'משפטים ארוכים יותר נוטים להיות נכונים יותר'."

---

## שקף 13 -- RAG Sanity Check

**מה להגיד:**
> "השקף הזה מציג את ההשוואה המבוקרת. ה-spectral features ללא אורך (length) נותנים את ה-AUC המרכזי. Trace length לבדו נותן תוצאה אקראית (50.8%). הוספת trace length ל-spectral features לא עוזרת באופן משמעותי. זה פוסל את החשש שאנחנו רק מזהים verbosity -- ה-signal מגיע כולו מה-spectral SHAPE של H(n), לא ממספר ה-tokens שיש בהצהרה."

---

## שקף 14 -- RAG Score Distributions

**מה להגיד:**
> "אלו ה-histograms של ציוני ה-Nadler-fused, מחולקים לפי label (grounded vs ungrounded), עבור כל שילוב של מודל ו-dataset. ציון Nadler גבוה יותר אומר 'סביר יותר שנכון / מבוסס'. ניתן לראות הפרדה ברורה בתאים עם AUC גבוה -- ההתפלגויות מופרדות היטב. תאים עם AUC נמוך יותר מראים יותר חפיפה. זה מאפשר לנו לאשר חזותית שהשיטה עושה משהו עקרוני, ולא רק מנצלת artifact התפלגותי."

---

## שקף 15 -- איך זה עובד: Agentic Domain (דוגמה)

**מה להגיד:**
> "עכשיו הכיוון החדש ביותר: multi-step agents. כאן אנחנו משתמשים ב-ReAct framework -- Reasoning + Acting. הסוכן (agent) מריץ loop: Thought, Action, Observation. בכל שלב, ה-Thought הוא מקטע reasoning של טקסט חופשי -- 'אני צריך למצוא מאיפה ה-Chainsmokers...' -- ואחריו Action כמו שאילתת חיפוש או פעולת סיום.
>
> אנחנו מחלצים H(n) רק מה-Thought tokens, עבור כל שלב. כל שלב מקבל spectral feature vector משלו וציון Nadler fusion משלו. לאחר מכן אנחנו עושים aggregation על פני השלבים בשלוש דרכים: Phi_min (החוליה החלשה -- הציון הנמוך ביותר מכל השלבים), Phi_avg (ממוצע), ו-Phi_last (שלב ה-reasoning האחרון).
>
> מימין תוכלו לראות איך זה נראה: שלב 1 מקבל 0.74, שלב 2 מקבל 0.61 (לא בטוח), שלב 3 מקבל 0.79. Phi_min = 0.61, מונע על ידי שלב 2. האינטואיציה: אם שלב reasoning כלשהו אינו אמין, הסוכן סביר שיגיע למסקנה שגויה. השרשרת חזקה רק כחוזק החוליה החלשה ביותר שלה."

---

## שקף 16 -- תוצאות Agentic + השוואה

**מה להגיד:**
> "המתחרה כאן הוא AUQ מ-Zhang et al. 2026. AUQ מבקש מהמודל להביע את הביטחון שלו במילים (verbalize confidence) -- 'עד כמה אתה בטוח?' -- ומשתמש בזה כהערכת ה-uncertainty. התוצאה הטובה ביותר שלהם, תוך שימוש ב-Phi_min aggregation ב-ALFWorld, היא 79.1%. הערה: ALFWorld היא סביבה שונה מה-multi-hop QA datasets שלנו, כך שזו השוואה של יכולת כללית, לא השוואה של setup זהה.
>
> Signal מאמצע הריצה (Mid-run signal) מ-DeepSeek-R1-7B על 2WikiMultiHopQA: ה-Phi_min שלנו נותן 85.0% -- זהו שיפור של +5.9 pp מעל ה-AUQ baseline. זה עדיין לא רשמי; הריצה עדיין זקוקה לתאים של Mistral-24B ו-Qwen-72B.
>
> טבלת ההשוואה מדגישה את ההבדלים המבניים המרכזיים: AUQ הוא white-box -- הוא זקוק למודל כדי להביע ביטחון ב-OUTPUT שלו, מה שאומר שמודלים שעברו RLHF-alignment עשויים לייצר טקסט עם overconfident באופן סיסטמטי. השיטה שלנו היא gray-box -- token probabilities ממקטע ה-Thought, שיותר קשה 'לשקר' דרכם באמצעות alignment. כמו כן, אנחנו לא דורשים prompting נוסף -- רק את ה-ReAct Thought output הסטנדרטי."

**לעופר:** "ההבחנה בין white-box ל-gray-box היא למעשה די עמוקה. ביטחון מילולי (Verbalized confidence) עובר דרך ה-RLHF distribution shift; ה-entropy ברמת ה-token במהלך ה-generation לא. לכן gray-box יכול להציג ביצועים טובים יותר מ-white-box אפילו עם פחות נתונים."

---

## שקף 17 -- מה הלאה (What's Next)

**מה להגיד:**
> "שלושה מסלולים קדימה. ראשית, השלמת Phase 11a: הרצת שני המודלים הנותרים, Mistral-24B ו-Qwen-72B, וביצוע הניתוח המלא על פני כל 8 התאים. זה אמור לתת לנו מספרי AUC רשמיים ולאפשר לנו לנצח את ה-AUQ baseline באופן רשמי.
>
> שנית, שני פיילוטים של הרחבת ה-domain -- זה חדש. Pilot A הוא HumanEval, יצירת קוד: N=20 בעיות, Qwen-7B, 3 ניסיונות לכל בעיה, מתויגים לפי האם ה-unit tests עוברים. השאלה המרכזית: האם spectral entropy במהלך יצירת קוד חוזה האם הקוד יתבצע בצורה נכונה? המתחרה הוא DSDE, שיטת disagreement מבוססת הרצה עם AUROC של 0.82-0.84. Pilot B הוא ALFWorld, ניווט מגולם (embodied navigation): N=5 משימות, המודל מנווט בסביבת בית מדומה. ה-label הוא הצלחה במשימה. המתחרה הוא AUQ על אותו benchmark -- 0.791. אלו פיילוטים -- אנחנו צריכים לעבור את ה-go/no-go gates לפני שנתחייב לריצות מלאות.
>
> לסיום התזה: ברכה, ה-LTT calibration הוא בערך 50 שורות קוד. הנתונים כבר קיימים מניסויי הטמפרטורה וה-behavioral ensemble. זה ממיר את מספר ה-AUROC שלנו לערבות פורמלית (formal guarantee): hallucination recall >= 90% בביטחון של 95%. זה מה שלדעתי סוגר את התזה פורמלית.
>
> עופר, שאלת ה-manifold: האם entropy trajectories מאותו מודל נמצאים על low-dimensional manifold? האם hallucination מתאימה לבריחה (escape) מאותו manifold? זה מתחבר ל-LOCA ול-IMM -- הערכת intrinsic dimensionality והערכת מצב של regime-switching. זהו ה-'למה' התיאורטי שיעלה את התזה מתרומה אמפירית לתרומה עקרונית."

---

## דף עזר (Cheat Sheet) -- כל מספרי הכותרות

| Domain | Dataset | Model | Our AUROC | Competitor | Their AUC | Delta | Labels | Access | Passes |
|--------|---------|-------|-----------|-----------|-----------|-------|--------|--------|--------|
| Math | MATH-500 | Qwen-7B T=1.0 | **90.0%** | -- | -- | -- | None | Gray-box | 1 |
| Math | GSM8K | Llama-3.1-8B | **76.0%** | LapEigvals | 72.0% | +4.0 pp | None | Gray-box | 1 |
| Science MCQ | GPQA Diamond | Qwen-72B-AWQ | **69.0%** | -- | -- | -- | None | Gray-box | 1 |
| Factual QA | TriviaQA CoT | Falcon-3-10B | 53.6% | EPR direct | 79.1% | -25.5 pp | None | Gray-box | 1 |
| Factual QA | WebQ CoT | Falcon-3-10B | 61.9% | EPR direct | 71.8% | -9.9 pp | None | Gray-box | 1 |
| RAG | HotpotQA | Llama-8B | **87.7%** | LOS-Net | 72.92% | +14.8 pp | LOS-Net needs labels | Gray-box | 1 |
| RAG | 2Wiki | Qwen-7B | **80.5%** | -- | -- | -- | None | Gray-box | 1 |
| Agentic (mid-run) | 2Wiki | DeepSeek-7B | **85.0%*** | AUQ | 79.1% | +5.9 pp | None | Gray-box vs White-box | 1/step |

---

## סיכום מתחרים (Competitor Summary)

| Competitor | Paper | Their Setting | Their Supervision | LLM Access |
|-----------|-------|--------------|-------------------|-----------|
| LapEigvals | arXiv:2502.17598 | GSM8K / Llama-8B | None | Gray-box (hidden states) |
| LOS-Net | arXiv:2503.14043 | HotpotQA RAG / Mistral-7B | **Required labels** | Gray-box |
| AUQ | Zhang et al. 2026 | ALFWorld agentic | None | **White-box (verbalized)** |
| DSDE | 2026 | HumanEval code | None | Black-box (exec. disagreement) |

---

## נקודות שיחה ספציפיות למנחים

**עופר (spectral methods, manifold):**
- ה-cross-domain feature orthogonality (שקף 5) הוא עמוד השדרה התיאורטי: sw_var_peak ו-cusum_max הם orthogonal ל-EPR על פני 7,001 דגימות. זה לא מקרי -- הם מודדים spectral modes שונים של H(n).
- הסטת ה-feature importance בין domains (rpdi דומיננטי ב-RAG, epr דומיננטי ב-math) מרמזת שתהליכים generative שונים מייצרים spectral fingerprints שונים. מודל manifold עשוי להסביר זאת כחלקים שונים של ה-manifold שפעילים למשימות שונות.
- השאלה התיאורטית הבאה: האם ה-entropy trajectory חי על low-D manifold, והאם hallucination היא חריגה ממנו?

**ברכה (conformal, calibration):**
- LTT calibration: נתוני ה-temperature + behavioral ensemble מ-Phase 6 (TriviaQA/WebQ) נותנים לנו מספר views עם AUROCs ידועים. זהו בדיוק הקלט ש-LTT צריך. כ-50 שורות קוד.
- ה-deployment guarantee שאנחנו רוצים: "בהינתן שאילתה זו, הגלאי מסמן אותה כ-hallucination עם hallucination_recall >= 90% בביטחון של 95%." זוהי risk-controlling prediction set.
- התוצאה השלילית ב-Factual QA למעשה עוזרת ל-calibration: אנחנו יכולים לתכנן את ה-guarantee כך שיחול רק על קלטים מסוג reasoning (ניתנים לסיווג לפי trace length > 200 tokens).

**אמיר:**
- התוצאה השלילית הנקייה ב-Factual QA היא חשובה: היא מגדירה את היקף התזה בצורה כנה ומונעת הבטחות יתר. אנחנו לא טוענים לזיהוי hallucination כללי -- אנחנו טוענים לזיהוי במשימות מסוג REASONING.
- תוצאת ה-RAG של 87.7% המנצחת את LOS-Net המבוססת supervision היא התרומה האמפירית החזקה ביותר עד כה. Unsupervised מול supervised, אותה רמת גישה, שיפור של +14.8 pp.
- הפיילוטים של Phase 11b הם קטנים במכוון (N=5-20) כדי להוות שער (gate) להחלטה לפני ריצות מלאות.
