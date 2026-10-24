# אימות מול המאמרים המקוריים: 07_Ising.md ו-08_ARES.md

שיטה: חילצתי את הטקסט המלא של שני קובצי ה-PDF עם pymupdf, כולל כל הנספחים (Ising: 46 עמודים, ARES: 20 עמודים). כל טענה בשני ההסברים נבדקה מול המקור. מספרי העמודים הם עמודי ה-PDF. ב-ARES מצוין בסוגריים גם מספר העמוד המודפס בכרך של EMNLP. את הממוצעים, הפערים וחסמי הדגימה חישבתי מחדש מהטבלאות שבמאמר.

---

## קובץ 1: 07_Ising.md
מאמר: "Dependence-Aware Label Aggregation for LLM-as-a-Judge via Ising Models" (arXiv 2601.22336v1)

| מס' | הטענה (בקיצור) | פסיקה | ראיה (עמוד, מיקום, ציטוט) | תיקון מוצע |
|---|---|---|---|---|
| 1 | המחברים Balasubramanian, Podkopaev, Kasiviswanathan, מ-UC Davis / AWS | נכון | עמ' 1, כותרת: "Department of Statistics, University of California, Davis" ו-"Amazon Web Services". רק המחבר הראשון שייך לשני המוסדות | — |
| 2 | arXiv 2601.22336v1, 29.1.2026 | נכון | עמ' 1, חותמת arXiv: "arXiv:2601.22336v1 [stat.ML] 29 Jan 2026". בעמוד השער עצמו מופיע התאריך "February 2, 2026" | אפשר להוסיף: "(בעמוד השער: 2 בפברואר 2026)" |
| 3 | ה-PDF לא מזכיר ICML 2026, ולכן השיוך לכנס לא אומת | נכון | אין במאמר שום שיוך לכנס. "ICML" מופיע רק בהפניות למאמרים אחרים (למשל [18], [24], [38]) | — |
| 4 | K שופטים נותנים הצבעות בינאריות, ו-Y לא ידוע. DS, majority vote ו-WMV מניחים CI | נכון | עמ' 1, סעיף 1: "built around the conditional independence (CI) assumption" | — |
| 5 | אצל שופטי LLM הנחת ה-CI נשברת בגלל נתונים, ארכיטקטורות, prompts ומצבי כשל משותפים (סעיף 1) | נכון | עמ' 1: "shared pretraining corpora, architectures, prompts, and failure modes" | — |
| 6 | מי שמניח CI סופר הסכמה בין שופטים מועתקים כראיה עצמאית שוב ושוב | נכון | עמ' 1: "can over-count agreement and yield systematically miscalibrated predictions" | — |
| 7 | בנספח B יש שלושה שופטים, ושופטים 1 ו-3 דומים | נכון | עמ' 17, נספח B: "Annotators 1 and 3 share substantial training data and prompt templates" | — |
| 8 | ה-CI מקבל את ההתפלגויות השוליות הנכונות, ועל (0,1,1) נותן P(Y=1)≈0.968 מול Bayes≈0.038 | נכון | עמ' 18: "≈0.038. Thus the Bayes-optimal prediction is Y = 0" וגם "PCI(Y = 1 \| J) … ≈0.968" | כדאי לציין שהדוגמה בנויה על Ising **בלתי תלוי-מחלקה** (W משותף, משוואה 9). דוגמה שנייה, תלוית-מחלקה, נותנת 0.031 מול 0.957 על (1,1,0) |
| 9 | ה-CI הופך את התווית ועוד בביטחון גבוה | נכון | עמ' 18: "the CI-predictor generates Y = 1 with high confidence" | — |
| 10 | משוואה 1: P(J\|Y=y) ∝ exp(Σh·J + ½Σ W·J·J) | נכון | עמ' 4, משוואה (1). הסכום הריבועי הוא על j≠k, ו-W סימטרית עם אלכסון אפס | — |
| 11 | ה-field h_j מתאר את נטיית שופט j להצביע 1 | נכון | עמ' 2: "controls how likely judge j is to output 1 under class y" | — |
| 12 | W>0 פירושו שהזוג נוטה להצביע 1 יחד, ו-W<0 שהוא נמנע מכך | נכון | עמ' 2: "W(y)jk > 0 encourages judges j and k to co-vote 1" | — |
| 13 | כש-W⁽⁰⁾≠W⁽¹⁾ ה-log-odds ריבועי, עם b_jk=ΔW_jk (משוואה 2) | נכון | עמ' 4, משוואה (2): "aj = ∆hj, bjk = ∆Wjk" | — |
| 14 | המאמר משווה את המקרה הריבועי ל-QDA (נספח C.2) | נכון | עמ' 20, C.2: "otherwise QDA is optimal and the decision boundary is quadratic" | — |
| 15 | Ising בלתי תלוי-מחלקה (משוואות 3–5): W משותף, האיברים הריבועיים מתבטלים והכלל ליניארי Σc_jJ_j+b₀ | נכון | עמ' 5, משוואות (3)–(5): "The quadratic terms cancel since the coupling matrix is shared" | — |
| 16 | התלות עדיין משפיעה על ה-partition function, כלומר על החותך, ועל אמידת הפרמטרים | נכון | עמ' 5: "partition functions (hence the intercept b0), and parameter estimation from finite data" | אפשר להוסיף: המבוא אומר גם שהתלות נספגת ב-"correlation-corrected weights" (עמ' 2) |
| 17 | המקבילה הליניארית היא LDA | נכון | עמ' 20: "vanishes exactly when the class covariances coincide … yielding LDA" | — |
| 18 | ההיררכיה CI ⊂ class-indep ⊂ class-dep (איור 2) | נכון | עמ' 3, איור 2: "Conditional Independence (CI) ⊂ Class-independent Ising ⊂ Class-dependent Ising" | — |
| 19 | משפט 1: כל שופט מצביע 1 בהסתברות ½ בשתי המחלקות, וההבדל היחיד הוא β₀<1 מול β₁>1 | נכון | עמ' 6–7, משפט 1 ומשוואה (6). עמ' 7: "P(Jj = 1 \| Y = y) = 1/2" | — |
| 20 | כלל CI קורס לפריור, עם סיכון min{π,1−π} לכל K | נכון | עמ' 6, משפט 1(1): "R(gind K) = min{π, 1 −π}, ∀K" | — |
| 21 | המבחן הריבועי 1{M_K²≥t} מגיע לסיכון שמתכנס ל-0 | נכון | עמ' 7, משפט 1(2): "satisfies R(˜gK) →0 as K →∞" | — |
| 22 | משפט 2 (משוואות 7–8): q₀<½<q₁, ועדיין הסיכון של CI מתכנס ל-π(1−p)>0 | נכון | עמ' 8–9, משפט 2(1),(3): "R(gind K) →π(1 −p)" | — |
| 23 | הסיבה היא מצב קולקטיבי של "הסכמה על הסימן הלא נכון" | נכון | עמ' 8: "the judges collectively enter a 'wrong-sign' agreement mode" | — |
| 24 | משפט 3 (נספח C.3): פקטור לטנטי משותף Z משאיר את CI תת-אופטימלי גם כש-K→∞ | לא מדויק | עמ' 25–26, נספח **C.3.1**, משפט 3. הגבול של הפער ניתן כביטוי מפורש, והוא חיובי רק בתנאי: "if P(gind ∞ ≠ g⋆ ∞) > 0, then the limit is strictly positive". המבוא (עמ' 3) מנסח "can remain strictly suboptimal" | "משפט 3 (נספח C.3.1): תחת מודל פקטור לוגיסטי-נורמלי **סקלרי וחליף**, הפער בין CI ל-Bayes מתכנס כש-K→∞ לביטוי מפורש. הפער חיובי ממש כאשר שני כללי הגבול נבדלים בהסתברות חיובית. כלומר CI **יכול** להישאר תת-אופטימלי" |
| 25 | Proposition 1 (משוואות 13–15): בצימוד חלש, מודל פקטור לטנטי מקורב עד סדר שני על ידי Ising עם צימודים בדרגה נמוכה, W_jk≈λ_jᵀλ_k | נכון | עמ' 21: "Latent-factor ⇒low-rank class-independent Ising couplings to second order". השארית חסומה ב-O(ε³), משוואה (15) | אפשר להוסיף שהצימודים לא תלויים במחלקה. רק ה-fields תלויים במחלקה |
| 26 | הנחת ה-CI של SML (כוואריאנס מחוץ לאלכסון מדרגה 1) היא המקרה W=0 | חלקי: "W=0 ⇔ CI" נכון. החלק על SML לא נמצא במאמר | עמ' 5: "If W ≡0, then the judges are conditionally independent given Y". המאמר לא מזכיר את SML ולא מבנה כוואריאנס מדרגה 1 | לסמן שהקישור ל-SML הוא פרשנות של כותב ההסבר. המאמר מכסה רק את W=0 ⇔ CI |
| 27 | L-SML ו-Jaffe et al. 2016 מתירים תלות לפי קבוצות | L-SML מחוץ להיקף. החלק על Jaffe נכון | עמ' 16, נספח A: "these works study the case of hierarchically dependent judges/annotators" | המאמר מתאר את Jaffe ו-Shaham כתלות **היררכית** ("tree-structured or nested"). כדאי להשתמש במונח הזה |
| 28 | Proposition 1 הוא מקבילה רכה לתלות לפי קבוצות: דרגה נמוכה ולא חלוקה קשיחה | פרשנות (לא נמצא במאמר) | המאמר לא מקשר את Prop. 1 ל-Jaffe או לחלוקה לקבוצות. התוכן (צימודים בדרגה ≤r) נכון, עמ' 21 | לסמן כפרשנות של הכותב |
| 29 | המאמר מצטט את Jaffe 2016 ו-Shaham 2016 וטוען שתלות היררכית "poorly aligned" עם שופטי LLM, שבהם התלות צפופה (נספח A) | נכון | עמ' 16: "this assumption is poorly aligned with LLM-as-a-judge settings" וגם "dense, non-hierarchical correlations" | — |
| 30 | אין השוואה ניסויית ל-Jaffe או ל-Shaham | נכון | טבלה 1 ואיור 3 כוללים רק Class-Dep, Class-Indep, CI-WMV ו-CI-UMV | — |
| 31 | המאמר לא מוכיח identifiability למודלי Ising, ומזכיר זיהוי רק לגבי DS (סעיף 1) | נכון | המופע היחיד של "identifiab" נמצא בעמ' 1: "subject to appropriate identifiability conditions" (בהקשר של DS ו-EM) | — |
| 32 | המאמר לא דן ב-label switching | נכון | המונח לא מופיע. קרוב לנושא: בעמ' 12 מומלץ לבדוק "stability across random initializations" כאבחון | אפשר להוסיף את האבחון הזה |
| 33 | התאוריה עוסקת בכלל Bayes ברמת האוכלוסייה | נכון | עמ' 11, סעיף 5: "stated in terms of the Bayes-optimal posterior under the true generative model" | — |
| 34 | המאמר מציין כצעד הבא הוכחה שהאלגוריתם המאומן מגיע לכלל הזה (סעיף 5) | נכון | עמ' 11: "A key next step is to extend the theory … to these algorithms" | — |
| 35 | ההתאמה ללא תוויות היא generalized EM (נספח D, Algorithm 1) | נכון | עמ' 30–31, Algorithm 1. עמ' 30: "We emphasize that the procedure is purely unsupervised" | — |
| 36 | ב-E-step מחשבים γ_i=P(Y=1\|J_i) | נכון | עמ' 31, משוואה (25) | אפשר להוסיף: אצל Ising גם ה-E-step משתמש בציון מקורב, pseudo-likelihood ("fully consistent surrogate EM"), עמ' 31–32 |
| 37 | ב-M-step משתמשים ב-pseudo-likelihood מרוגלר במקום partition function, וזה מתפרק לרגרסיה לוגיסטית לכל שופט עם משקלות γ ו-1−γ | נכון | עמ' 31: "reduces to a set of weighted logistic regressions (one per node)" | — |
| 38 | ההיפר-פרמטרים כוונו "manually … for best performance" (D.1) | נכון | עמ' 32: "we manually tune the hyperparameters (if any) for best performance" | — |
| 39 | לא אומת שהכיוונון נעשה בלי תוויות | נכון: המאמר לא מכריע | המאמר לא מציין קריטריון לכיוונון. בעמ' 32 הוא מציע "cross-validation", ובעמ' 12 מזכיר "predictive calibration on small labeled validation sets" ככלי אבחון | להשאיר "לא אומת", ולהוסיף שהמאמר עצמו מזכיר סט ולידציה קטן עם תוויות ככלי אבחון |
| 40 | הקלט הוא n×6 הצבעות מ-6 מודלים בשמותיהם, כולם בטמפרטורה 0 (סעיף 4.1) | נכון | עמ' 9: "For all models, the temperature parameter is set to zero" | — |
| 41 | הפלט הוא posterior לכל פריט ותווית | נכון | עמ' 31, Algorithm 1: "Output: parameters bΘ and posteriors bγi; predicted labels" | — |
| 42 | Relevance: WikiQA, כ-3000 פריטים | נכון | עמ' 10: "resulting in approximately 3000 evaluation instances" | — |
| 43 | Toxicity: Jigsaw, 1000 פריטים מכל אחד מ-4 דליים | נכון | עמ' 10: "We randomly sample 1000 comments from each of the four toxicity score buckets". סך הכול 4000 | — |
| 44 | Summarization: CNN/DailyMail-Phoenix, 1100 פריטים | נכון | עמ' 10: "The dataset contains 1100 instances" | — |
| 45 | טבלה 1: accuracy על test, ממוצע של 20 חזרות. סדר העמודות והמספרים (0.90/0.88/0.82/0.80 וכו') | נכון | עמ' 9, טבלה 1: "Test accuracy … Numbers are averages over 20 trials". כל תשעה המספרים תואמים | — |
| 46 | גודל האימון: 2500, 3000 ו-1000 פריטים | נכון | עמ' 10: "2500 samples for relevance, 3000 … toxicity, and 1000 … summarization" | חישוב שלי: נשארים כ-500, כ-1000 וכ-100 פריטי test בהתאמה. ב-Summarization ה-test קטן מאוד |
| 47 | איור 3: מודלי Ising עוקפים את CI **רק** מעל סף מסוים של פריטים ושופטים | לא מדויק | עמ' 10: "once the number … exceed a modest threshold (and in some regimes even with fewer training samples)" | "לפי איור 3, מודלי Ising עוקפים את CI מעל סף צנוע של מספר פריטים ושופטים, ובחלק מהמשטרים גם עם פחות פריטים" |
| 48 | טבלה 1 לא מציגה סטיות תקן | נכון | עמ' 9, טבלה 1 מציגה ממוצעים בלבד | להוסיף: באיור 3 יש standard error. עמ' 10: "report the average test accuracy (and standard error)" |
| 49 | אין השוואה ל-SML, ל-Jaffe או ל-Ai et al. | נכון | Ai et al. [2] נדון רק בנספח A (עמ' 16), וגם Jaffe ו-Shaham. אף אחד מהם לא נמדד | — |
| 50 | תלות משותפת שלא תלויה במחלקה משנה את המשקלות, אבל הכלל נשאר ליניארי | נכון | עמ' 2: "correlations are absorbed into correlation-corrected weights and intercepts" | — |
| 51 | כשהתלות שונה בין המחלקות, **שום כלל ליניארי**, כולל L-SML, לא יכול לנצל זאת (משפטים 1–2) | לא מדויק | משפטים 1–2 עוסקים ב-**population CI-predictor**, כלומר מכפלה של ההתפלגויות השוליות האמיתיות, במבנים ספציפיים של Curie-Weiss. הם לא עוסקים בכל כלל ליניארי. עמ' 3: "there exist regimes where each judge is better than random". ב-C.2 (עמ' 20) הניסוח רך: "linear/CI rules can be fundamentally misspecified" | "משפטים 1–2 מראים שבמבנים מסוימים (Curie-Weiss), כלל CI עם ההתפלגויות השוליות האמיתיות לא מנצל מידע שיושב במבנה התלות. הרחבה לכל כלל ליניארי, וממנה ל-L-SML, היא הסקה שלנו" |
| 52 | אם בטלמטריה שלנו מבנה הקורלציה שונה בין שלבים, זה מסביר את התקרה (השערה) | מחוץ להיקף | — | — |
| 53 | המקבילה הגאוסית היא LDA/QDA (נספח C.2) | נכון | עמ' 20, C.2 | — |
| 54 | ההתאמה הישירה אצלנו היא תערובת גאוסית בת שני רכיבים ולא Ising (לא נבדק) | השערה. המאמר לא מכריע | C.2 תומך באנלוגיה הגאוסית, אבל עוסק בסיווג מונחה (LDA/QDA) ולא בתערובת ללא תוויות | להשאיר כהשערה |
| 55 | אצלם יחידת הניקוד היא פריט שלם, והמאמר מניח פריטים בלתי תלויים (סעיף 2) | נכון | עמ' 3–4: "Suppose that we observe n independent items" וגם "We assume items are independent" | — |
| 56 | הם מתאימים על **אלפי** פריטים משותפים | לא מדויק (קל) | ב-Summarization יש 1100 פריטים בסך הכול ו-1000 באימון, עמ' 10 | "על כאלף עד כ-3000 פריטים בכל משימה" |
| 57 | class-dep עם K=6 כולל לפחות 2·(6+15) פרמטרים (חישוב שלי) | נכון (החישוב) | לכל מחלקה יש 6 fields ו-15 צימודים, ומשוואה (1) נותנת 42. עם π יוצא 43. המאמר לא מציין את המספר | — |
| 58 | ציטוט: "large number of training samples relative to the number of model parameters" (סעיף 4.1) | נכון, עם השמטה | עמ' 10: "large number of training (unsupervised) samples relative to the number of model parameters". המשפט מסביר למה class-dep עדיף על class-indep | לשחזר את המילה "(unsupervised)" בציטוט |
| 59 | ההתאמה שלנו עם 17 ערוצים לא סבירה עבור class-dep Ising, ודורשת איגום | מחוץ להיקף | חישוב עזר: 2·(17+136)=306 פרמטרים | — |
| 60 | הם מדווחים accuracy בלבד, בלי AUC, PRMScore או לוקליזציה | נכון | טבלה 1 ואיור 3 מציגים test accuracy. בסימולציות (נספח F) מוצג "Empirical Risk" | — |
| 61 | בסיכום: המאמר מוכיח שכאשר המידע יושב במבנה התלות, **כל** כלל CI, כולל SML ליניארי, נשאר תת-אופטימלי גם עם אינסוף שופטים | לא מדויק | אלה תוצאות קיום ("there exist regimes", עמ' 3) עבור ה-CI-predictor במודלים מסוימים. המאמר לא מזכיר את SML | "המאמר מוכיח שיש מודלי תלות (Curie-Weiss, פקטור לטנטי) שבהם כלל CI נשאר תת-אופטימלי באופן מובנה גם כש-K→∞" |
| 62 | EM עם pseudo-likelihood מראה רווח של 5–8 נקודות accuracy על WMV (טבלה 1) | לא מדויק | מטבלה 1: class-dep מוסיף 8, 7 ו-7 נקודות. class-indep מוסיף 6, 5 ו-3 | "רווח של 7–8 נקודות ל-class-dep ו-3–6 נקודות ל-class-indep על פני WMV (טבלה 1)" |
| 63 | ההעברה לציוני שלבים רציפים לא נבדקה | מחוץ להיקף (נכון, אין כזה ניסוי במאמר) | — | — |

### התיקונים החשובים ב-07_Ising.md
1. **טענה 51 וטענה 61 (החשובות ביותר):** משפטים 1–2 עוסקים ב-CI-predictor בלבד, במבני Curie-Weiss ספציפיים, ומוכיחים קיום ("there exist regimes"). הם לא מוכיחים ש"שום כלל ליניארי" יכול לנצל תלות תלוית-מחלקה, ולא עוסקים ב-SML או ב-L-SML. החלת המשפטים על L-SML היא הסקה של כותב ההסבר, וצריך לסמן אותה ככזאת.
2. **טענה 62:** הרווח על WMV הוא 7–8 נקודות ל-class-dep ו-3–6 ל-class-indep, לא "5–8".
3. **טענה 24:** משפט 3 נמצא ב-C.3.1. הוא נותן גבול מפורש, והגבול חיובי רק בתנאי שהכללים נבדלים בהסתברות חיובית. ניסוח נכון: CI "יכול" להישאר תת-אופטימלי.
4. **טענה 47:** לפי איור 3 העקיפה מתחילה מעל סף צנוע, ובחלק מהמשטרים גם מתחתיו. לא "רק" מעל סף.
5. **טענות 26 ו-28:** הקישור ל-SML (כוואריאנס מדרגה 1) ופרשנות Prop. 1 כ"גרסה רכה של חלוקה לקבוצות" לא נמצאים במאמר. צריך לסמן אותם כפרשנות.
6. **טענה 39:** להוסיף שהמאמר מזכיר "small labeled validation sets" ככלי אבחון לבחירת רמת המודל (עמ' 12). השאלה אם הכיוונון היה ללא תוויות נשארת פתוחה.
7. תיקונים קטנים: בציטוט בטענה 58 חסרה המילה "(unsupervised)". "אלפי פריטים" צריך להיות "כאלף עד 3000" (טענה 56). באיור 3 יש standard errors (טענה 48).

---

## קובץ 2: 08_ARES.md
מאמר: "Probabilistic Soundness Guarantees in LLM Reasoning Chains" (EMNLP 2025, עמ' 7506–7525)

| מס' | הטענה (בקיצור) | פסיקה | ראיה (עמוד, מיקום, ציטוט) | תיקון מוצע |
|---|---|---|---|---|
| 1 | פורסם ב-EMNLP 2025 Main | נכון (עם הערה) | עמ' 1: "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7506–7525". המילה "Main" לא מופיעה, אבל זה כרך ה-Proceedings ולא Findings | — |
| 2 | המחברים You, Xue, Havaldar, Rao, Jin, Callison-Burch, Wong, מ-UPenn / UT Austin | נכון | עמ' 1. רק Anton Xue משויך ל-UT Austin | — |
| 3 | שופט (LLM-Judge או PRM) שקורא את כל השרשרת "נגרר" אחרי טעויות מוקדמות | נכון | עמ' 1 (7506): "they are often distracted by the presence of propagated errors" | — |
| 4 | באיור 1, צעד 4 שגוי (5x=9x−20). צעד 5 נגזר ממנו נכון, ולכן השופט מסמן אותו כתקין | נכון | עמ' 2 (7507): "an LLM may incorrectly mark step 5 as sound by incorrectly relying on step 4" | — |
| 5 | צעד אינו sound אם הוא ungrounded (**סותר** את ההקשר), invalid, או נשען על טעות קודמת (כיתוב איור 1) | לא מדויק (קל) | עמ' 1, כיתוב איור 1: "ungrounded (incorrect with respect to the context), invalid (logically incorrectly derived), or contains propagated errors" | "ungrounded (שגוי ביחס להקשר)", לא "סותר" |
| 6 | הפתרון: לבדוק כל צעד רק מול צעדים קודמים שנמצאו sound, כמו שאדם מתעלם מטענה מפוקפקת (§1, איור 2) | נכון | עמ' 2, איור 2: "using only claims already identified to be sound as the premise" | — |
| 7 | אינטואיציה: צעד קודם נכלל כהנחה בהסתברות **ששווה לציון ה-soundness שלו** | לא מדויק | Algorithm 1, שורות 5–6 (עמ' 5): בכל דגימה i ההסתברות היא p⁽ⁱ⁾=E(C(α⁽ⁱ⁾),C), כלומר ה-entailment **בהינתן ההנחות שנבחרו באותה דגימה**. היא לא τ הממוצע. כיתוב איור 3: "with probability equal to its entailment rate" | "בכל 'עולם' צעד קודם נכלל בהסתברות השווה לציון ה-entailment שקיבל באותו עולם (בהינתן ההנחות שנבחרו בו). τ הוא הממוצע של הציונים האלה על פני העולמות" |
| 8 | τ של הצעד הנוכחי הוא ממוצע ההסתברות שההנחות שנבחרו גוררות אותו | נכון | עמ' 5, משוואות (6)–(7) | — |
| 9 | השרשרת היא C₁..Cₙ (base claims, מניחים שהם נכונים) ואחריהם derived claims (משוואה 1) | נכון | עמ' 3 (7508): "their factual accuracy is given and assumed to be validated by external mechanisms" | — |
| 10 | E: C⋆×C→[0,1] (§2.2) | נכון | עמ' 3, §2.2: "a probabilistic entailment model E : C⋆× C →[0, 1]" | — |
| 11 | Hard soundness דורש E=1 לכל צעד (הגדרה 2.1, משוואה 2) | נכון | עמ' 3, הגדרה 2.1 ומשוואה (2) | — |
| 12 | משוואה 3: τ(E,P,H)=Σ_α E(P(α),H)·Pr[α] על 2ᵏ תת-קבוצות | נכון | עמ' 4 (7509), משוואה (3): "we consider all 2k configurations of inclusion and exclusion" | — |
| 13 | משוואות 4–5: base claim נכלל בהסתברות pᵢ, צעד derived בהסתברות E(הנחות, צעד), וההסתברות נבנית רקורסיבית | נכון | עמ' 4–5: "a claim is added to our set of 'sound' premises based on how strongly the current set entails it" | — |
| 14 | שגיאת דפוס במשוואה 4: המעריך של (1−pᵢ) הוא αᵢ. במשוואה 5 מופיע רק מקרה ההכללה | נכון | עמ' 4, משוואה (4) כפי שהודפסה: "pαi i (1 −pi)αi". במשוואה (5) יש רק הכפל ב-E, בלי (1−E) למקרה ההשמטה | להוסיף: גם צד שמאל של משוואה (5) מודפס כ-"Pr[α1:n]" ולא Pr[α1:n+k]. זו שגיאת דפוס נוספת |
| 15 | משוואה 6: τₖ הוא שוליים על 2ⁿ⁺ᵏ⁻¹ צירופים | נכון | עמ' 5: "it requires summing over 2n+k−1 possible combinations" | — |
| 16 | משוואה 7 / Algorithm 1: τ̂ₖ=(1/N)ΣE. כל דגימה מקבלת p⁽ⁱ⁾ ואחר כך Bernoulli | נכון | עמ' 5, Algorithm 1, שורות 3–10 | — |
| 17 | משפט 3.1: אם N≥log(2m/δ)/(2ε²), אז בהסתברות ≥1−δ מתקיים \|τ̂ₖ−τₖ\|≤ε לכל k בו-זמנית | נכון | עמ' 5, משפט 3.1: "with probability at least 1 −δ, this estimate has error \|ˆτk −τk\| ≤ε for all k" | אפשר להוסיף: המשפט אומר "define τ1..τm as in Equation (7)", אבל τ מוגדר במשוואה (6). זו מעידה נוספת |
| 18 | ההוכחה היא Hoeffding לכל צעד ואחריו union bound על m צעדים (נספח A, משוואות 8–11) | נכון | עמ' 13 (7518): "According to Boole's inequality and Hoeffding's inequality", משוואות (8)–(11) | — |
| 19 | הערובה היא רק על שגיאת הדגימה של τ̂ ביחס ל-τ, ו-τ מוגדר דרך E. אין ערובה שהצעד נכון | נכון (נובע מנוסח המשפט) | עמ' 5: "For any entailment model E …". המשפט חוסם רק את \|τ̂−τ\| | — |
| 20 | הביצועים תלויים בכיול של מודל ה-entailment (Limitations) | נכון | עמ' 10 (7515): "poor calibration can lead to unreliable scores" | — |
| 21 | ציטוט: "can only improve upon entailment models that can already do correct entailment" (§4.6) | נכון | עמ' 9 (7514), §4.6 | — |
| 22 | התקציר כותב "certified statistical guarantees of its soundness", ניסוח רחב יותר ממה שהמשפט נותן | הציטוט נכון. ההערכה נתמכת | עמ' 1: "provides certified statistical guarantees of its soundness" | — |
| 23 | השופט הוא GPT-4o-mini או Qwen3-4B עם Likert בן 7 דרגות, הממופה ל-{1,.8,.6,.5,.4,.2,0} (C.1) | לא מדויק (חלקי) | עמ' 13, C.1: "We use two output modes: (i) binary YES/NO … (ii) a 7-point Likert". המיפוי נכון. במצב הבינארי משתמשים ב-ARES-bin (טבלה A11) | להוסיף: "יש גם מצב בינארי YES/NO→{1,0}, ובו נבדק ARES-bin באבלציה" |
| 24 | בנספח C.9 / טבלה A7 גם Qwen2.5-Math-PRM-7B משמש כ-backbone של ה-entailment | נכון | עמ' 16, C.9: "using a SOTA PRMs, Qwen2.5-Math-PRM-7B, as the base entailment model". נזכר גם ב-§4, עמ' 6 | — |
| 25 | לא מפורט איך ה-PRM מקבל תת-קבוצה של הנחות | נכון: המאמר לא מפרט | ב-C.9 אין שום פירוט על ממשק ה-PRM | — |
| 26 | δ=ε=0.1 ו-p=0.95 עבור base claims (C.2) | נכון | עמ' 14, C.2: "We use p = 0.95 for the inclusion rate for base claims" | להוסיף: האבלציה ממליצה דווקא על p=1 (עמ' 16: "we can effectively choose p = 1") |
| 27 | עבור m=10 יוצא 265 דגימות לצעד (כיתוב טבלה 3) | נכון | עמ' 8: "a sequence of length m = 10 needs 265, 67, 30, 17 samples". חישוב שלי: ln(200)/0.02=264.9 | — |
| 28 | בפועל, בזכות מטמון של זוגות חוזרים, משתמשים רק ב-0.03x–0.31x מהחסם (איור 5) | נכון | עמ' 7, כיתוב איור 5: "only uses 0.03x to 0.31x the number of samples". ובטקסט: "avoid redundant LLM calls for the same premise-hypothesis pairs" | להוסיף: המאמר מייחס את החיסכון גם לכך שפלטים של 0/1 דורשים דגימה אחת בלבד (עמ' 8) |
| 29 | העלות הכוללת ב-GPT-4o-mini הייתה כ-600$ (C.6) | נכון | עמ' 15, C.6: "approximately 600 USD in total for prototyping and experiments" | "כ-600$ כולל אב-טיפוס". Qwen3-4B רץ על A100 80GB |
| 30 | הסף: עוברים על כל הערכים ובוחרים את הממקסם Macro-F1, ב-5-fold CV (§4, C.3) | נכון | עמ' 6, §4: "sweep over all the values … select the one that maximizes Macro-F1" | — |
| 31 | "one fold for validation and four folds for testing" ב-§4 לא עקבי עם C.3 | לא מדויק | ב-C.3 (עמ' 14) נאמר רק "thresholds are picked … on the validation split … numbers are on the test split", בלי גדלים, ולכן אין סתירה מול C.3. הסתירה נמצאת **בתוך §4**: הסף נבחר "in the training split", ובאותו משפט fold אחד הוא "validation" | "ב-§4 הסף נבחר על 'training split', ובאותו משפט נאמר שבכל fold אחד משמש לוולידציה וארבעה לבדיקה, חלוקה הפוכה מהמקובל. C.3 אומר שהסף נבחר על validation ונמדד על test. המאמר לא מכריע מה בדיוק נעשה" |
| 32 | הסף נבחר בעזרת תוויות | נכון | מקסום Macro-F1 על split מתויג (§4, C.3) | — |
| 33 | ה-benchmarks: PRMBench, DeltaBench, ClaimTrees ו-CaptainCookRecipes, שני האחרונים סינתטיים (§4) | נכון | עמ' 6: "ClaimTrees (our synthetic data), and CaptainCookRecipes". עמ' 8: "synthetic benchmarks ClaimTrees and CaptainCookRecipes" | — |
| 34 | לפי C.3 הם השתמשו ב-"a subset of examples", וגודל ה-subset לא מדווח | נכון | עמ' 14, C.3: "We use a subset of examples for each experiment". אין מספר בשום מקום במאמר | — |
| 35 | המדדים: Macro-Recall, Macro-Precision ו-Macro-F1. אין PRMScore | נכון | עמ' 6: "We evaluate using Macro-recall, Macro-precision and Macro-F1". המונח PRMScore לא מופיע | — |
| 36 | בתקציר: 72.1% Macro-F1 (+8.2) | נכון | עמ' 1: "(72.1% Macro-F1, +8.2 points)" | — |
| 37 | 72.1 הוא ממוצע ARES/GPT-4o-mini על ארבעת הסטים בטבלה 1 (שחזור) | נכון (החישוב) | (0.640+0.708+0.903+0.633)/4 = 0.721. המאמר לא מפרט את החישוב | — |
| 38 | ‎+8.2 ≈ 8.1 מול ה-baseline הטוב ביותר בכל סט (שחזור) | נכון (החישוב). המאמר לא מכריע | ה-baselines הטובים לכל סט: 0.643, 0.699, 0.628, 0.589. הממוצע 0.640, והפער 8.125. ה-baseline הטוב ביותר כשיטה אחת (Entail-Base, 0.5715) נותן 14.9 | — |
| 39 | ClaimTrees: 0.903 (טבלה 1) | נכון | עמ' 6, טבלה 1 | להוסיף: התקציר מדווח "+27.6", אבל מהטבלה 0.903−0.628 (LLM-Judge) = 27.5 |
| 40 | בשרשראות של עד 50 צעדים: ≥89% (§4.2, איור 4) | נכון כציטוט. יש אי-התאמה פנימית | עמ' 7: "sustains a Macro-F1 score of at least 89% on chains up to 50 steps". אבל בטבלה A11 (עמ' 20), בתצורה הראשית ARES-0.95: 0.854 באורך 5 ו-0.867 באורך 50. ב-ARES-1: 0.873 באורך 5 | להוסיף: "בטבלה A11 בנספח, באותה תצורה, יש ערכים מתחת ל-89%" |
| 41 | PRMBench בטבלה 1: ARES 0.640 מול LLM-Judge 0.643 (GPT-4o-mini). ב-Qwen3-4B: ARES 0.636, Entail-Prev 0.641, LLM-Judge 0.675 | נכון | עמ' 6, טבלה 1. כל המספרים תואמים | — |
| 42 | על PRMBench ARES לא ראשון, למרות "best Macro-F1 on both datasets" ב-§4.1 | נכון | עמ' 6, §4.1: "ARES achieves the best Macro-F1 scores on both datasets" | להוסיף: גם בכיתוב טבלה 1 ("top-performing in majority of settings (5/8)") יש בעיה. בספירה שלי ARES ראשון רק ב-4 מתוך 8 תאים: GPT על DeltaBench, ClaimTrees ו-CaptainCook, ו-Qwen על ClaimTrees |
| 43 | טבלה A7: על PRMBench ה-PRM מקבל F1 0.749 ו-ARES 0.736 | נכון | עמ' 16, טבלה A7 | להוסיף: ARES ו-Entail-Prev זהים שם (0.736). כיתוב A7, "ARES performs the best across various datasets", סותר את שורת ה-PRM |
| 44 | ARES מנצח רק על ClaimTrees-10 (0.733 מול 0.594) | נכון | עמ' 16, טבלה A7 | — |
| 45 | על CaptainCook4D הם כמעט שווים (0.543 מול 0.552) | נכון | עמ' 16. C.9: "ARES achieves on par performance with PRM" | — |
| 46 | Best-of-2 על PRMBench (טבלה 4): 0.660 ל-ARES עם ציון הצעד האחרון | נכון | עמ' 8, טבלה 4 (ועם סטיות תקן בטבלה A6) | להוסיף: בממוצע על כל הצעדים Entail-Prev (0.790) עדיף על ARES (0.730) |
| 47 | אותו PRM קפוא (Qwen2.5-Math-PRM-7B), עם הכותרת "ARES Also Improves PRMs" (C.9) | נכון (על המאמר). ההשוואה אלינו מחוץ להיקף | עמ' 16: "C.9 ARES Also Improves PRMs" | — |
| 48 | ARES הוא wrapper ללא אימון סביב שופט קפוא | נכון | עמ' 10: "our model-agnostic approach". אין במאמר שלב אימון | להוסיף: ה-prompts כוונו ידנית לכל סט (C.2: "tuned by manually examining the results"), והסף נבחר על תוויות |
| 49 | עלות: "עשרות עד מאות קריאות לכל צעד" (איור 5) | לא מדויק | איור 5 (עמ' 7) מציג "Average Samples per Derived Claim" בפועל, כ-0.03x–0.31x מהחסם. לפי החישוב שלי החסם הוא 231 עד 345 עבור m=5..50, כך שבפועל מדובר בכ-7 עד כ-100 דגימות | "מכמה דגימות ועד כמאה לכל צעד בפועל (0.03x–0.31x מחסם של כ-230–345), לעומת forward pass אחד" |
| 50 | מכל קריאה מתקבל ציון entailment יחיד, והציונים ממוצעים על תת-קבוצות של הנחות | נכון | משוואה (7) | — |
| 51 | הסף נבחר על תוויות (CV, מקסימום Macro-F1) | נכון | §4, C.3 | — |
| 52 | המדד הוא Macro-F1 על subset לא מוגדר | נכון | C.3 | — |
| 53 | אי אפשר להשוות ישירות: המדד שונה, ה-subset לא ידוע, והסף מפוקח | נכון (על המאמר). ההשוואה עצמה מחוץ להיקף | — | — |
| 54 | ה-PRM לבד טוב מ-ARES על PRMBench, ולכן הנחת ההתפשטות לא נותנת רווח בתחום המתמטי | פרשנות. נתמכת בחלקה | C.9 (עמ' 16): "the specialized PRM is a strong baseline on its in-domain dataset (PRMBench)". ARES=Entail-Prev=0.736 בטבלה A7. המאמר לא קובע את המסקנה הכללית | לסמן כפרשנות |
| 55 | ציטוט: "error propagation is limited and not always annotated" (§4.6) | נכון | עמ' 9 (7514): "In naturally occurring datasets, error propagation is limited and not always annotated" | — |
| 56 | בשורה אחת: מגריל תת-קבוצות של צעדים קודמים **לפי ה-soundness שלהם**, הערובה רק על שגיאת הדגימה, ועל PRMBench לא עולה על ה-PRM (0.736 מול 0.749) | נכון ברובו | ראו טענות 7, 19 ו-43 | "לפי ציון ה-entailment שקיבלו בכל דגימה" (כמו בטענה 7) |

### התיקונים החשובים ב-08_ARES.md
1. **טענה 31:** הסתירה שמיוחסת ל-C.3 לא קיימת שם. C.3 לא מציין גדלים. הסתירה נמצאת בתוך §4 עצמו: הסף נבחר על "training split", ובאותו משפט נאמר שבכל fold אחד משמש לוולידציה וארבעה לבדיקה. בכל מקרה הסף נבחר בעזרת תוויות.
2. **טענות 7 ו-56:** ההסתברות שצעד קודם נכלל היא ציון ה-entailment **באותה דגימה** (בהינתן ההנחות שנבחרו בה), ולא ציון ה-soundness הממוצע שלו (τ).
3. **טענה 49:** "עשרות עד מאות קריאות לכל צעד" מגזים בקצה התחתון. לפי איור 5, בפועל מדובר בכ-7 עד כ-100 דגימות לכל צעד.
4. **טענה 23:** C.1 מגדיר גם מצב בינארי YES/NO, ולא רק Likert.
5. **אי-התאמות נוספות במאמר שכדאי להוסיף לרשימה:**
   (א) הכיתוב של טבלה 1 טוען "5/8", אבל בספירה ARES ראשון רק ב-4 מתוך 8 תאים.
   (ב) הכיתוב של טבלה A7 טוען ש-ARES "performs the best", אבל ה-PRM לבד עדיף על PRMBench ועל CaptainCook4D.
   (ג) הטענה "at least 89%" עד 50 צעדים לא מתיישבת עם טבלה A11: 0.854 ו-0.867 ב-ARES-0.95.
   (ד) צד שמאל של משוואה (5) מודפס כ-Pr[α1:n].
   (ה) משפט 3.1 מפנה למשוואה (7) כהגדרה של τ, אבל τ מוגדר במשוואה (6).
   (ו) ‎+27.6 בתקציר מול 27.5 לפי טבלה 1.
6. **פרטים קטנים:** ungrounded פירושו "שגוי ביחס להקשר", לא "סותר" (טענה 5). העלות של 600$ כוללת אב-טיפוס (טענה 29). ב-C.2 נבחר p=0.95, אבל האבלציה ממליצה על p=1 (טענה 26). ה-prompts כוונו ידנית לכל סט (טענה 48).
7. **מה מחזיק היטב:** כל המספרים המרכזיים נכונים. זה כולל את טבלה 1, טבלה A7 (0.749 מול 0.736), טבלה 4, 265 הדגימות והשחזור של 72.1. הטענה ש-ARES לא ראשון על PRMBench, למרות §4.1, מאומתת.
