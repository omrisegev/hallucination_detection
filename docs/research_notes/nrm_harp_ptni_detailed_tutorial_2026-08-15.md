# מ־HARP אל NRM, ומ־NRM אל PTNI

## מדריך מפורט להבנת השיטות, ההשראה, המגבלות והשלב המחקרי הנוכחי

**תאריך:** 2026-08-15  
**מטרת המסמך:** מסמך לימוד עצמאי לקראת הצגת המחקר למנחי התזה  
**סטטוס מדעי:** המסמך מפריד בקפדנות בין שיטות שכבר נוסו ונמדדו, שיטות שנדחו, ושיטות שעדיין נמצאות בשלב פרוספקטיבי.

---

## 1. התשובה הקצרה ביותר

הסיפור המחקרי מורכב מארבע שכבות:

1. **IU-PCR** הוא העוגן הלא־מפוקח: הוא ממזג כ־30 מדדי טלמטריה של אי־ודאות לציון ביטחון יחיד, בלי תוויות נכונות/שגויות.
2. **HARP** סיפק השראה מבנית: לפני שמסווגים hallucination, כדאי להפריד תת־מרחב דומיננטי אך לא־ממוקד־מטרה מתת־מרחב שבו אות ה־reasoning עשוי להיות נקי יותר.
3. **Family-NRM** תרגם את הרעיון הזה אל תוך המידע המותר לנו: במקום hidden states ו־unembedding, הוא מפרק את ציון IU לשש תרומות של משפחות פיצ'רים, מסיר מכל תרומה את החלק שכבר מוסבר על ידי IU, ובוחר כיוון ספקטרלי “ניטרלי” שאינו נראה כמו תלות משותפת חזקה או כפילות דטרמיניסטית.
4. **PTNI** נועד לפתור את החיסרון היסודי שנשאר: covariance לבדו אינו יודע איזה כיוון קשור לנכונות. PTNI יוצר התערבויות מכניות שבהן ידוע בדיוק מתי אותו response תקין ומתי הוא שגוי ביחס ל־prompt, ובמקביל מודד שינויי nuisance שאינם משנים את המשמעות. כך מתקבל כיוון target בעל פולריות ידועה, בלי להשתמש בתוויות hallucination טבעיות.

הצעת **PTNI-guided NRM** משלבת את שני האחרונים:

- PTNI אומר **לאיזה כיוון לחפש**;
- NRM אומר **באיזה תת־מרחב ספקטרלי כדאי לחפש או מאילו modes להתרחק**;
- IU-PCR נשאר העוגן וה־fallback המדויק כאשר אין ראיה אמינה לתיקון.

המשפט החשוב ביותר להצגה למנחים הוא:

> NRM מנסה לזהות גאומטריה שאריתית “לא־תלויה ולא־כפולה”, אבל אינו יכול לזהות נכונות מתוך covariance בלבד; PTNI מוסיף זיהוי מכני של target לעומת nuisance, וההיבריד העתידי בודק אם ההגבלה הספקטרלית של NRM מוסיפה ערך מעבר ל־PTNI עצמו.

---

## 2. מפת מושגים: מה כבר קיים ומה עדיין מוצע

| רכיב | סוג למידה | מה נכנס ל־fit | מה כבר ידוע אמפירית | סטטוס |
|---|---|---|---|---|
| IU-PCR | לא־מפוקח | מטריצת פיצ'רים בלבד | baseline פעיל ומוגן | ממומש ונמדד |
| HARP המקורי | מפוקח, white-box | hidden states, unembedding, תוויות hallucination | תוצאות מאמר, לא שיטה שלנו | מקור השראה |
| HARP-inspired contribution teacher | מפוקח | תרומות משפחות IU + תוויות | הוכיח שיש correction שימושי במרחב המשפחות | הוכחת היתכנות בלבד |
| Family-NRM | לא־מפוקח, trans-environment | תרומות משפחות IU ללא תוויות | העברה חיובית וכשל/הצלחה מעורבים באישורים | שיטה קפואה ומאושרת בתחום מוגבל |
| Atomic-NRM | לא־מפוקח | תרומות אטומיות ללא משפחות | נכשל מול IU ו־Family-NRM | נדחה |
| PTNI-IU | mechanically/self-supervised | quartets עם אמת מכנית, בלי תוויות benchmark טבעיות | S0a המכני עבר; אין עדיין תוצאת detector | הניסוי הפעיל A6 |
| PTNI-guided NRM | mechanically supervised hybrid | כיוון PTNI + projector ספקטרלי אטומי | טרם נוסה | הצעה מותנית לאחר A6 |

המונח “self-supervised” דורש דיוק. Family-NRM הוא קרוב יותר ל־**unsupervised**: הוא משתמש רק בגאומטריה של batches בלתי מתויגים. PTNI אינו unsupervised במובן המחמיר, כי הוא מקבל פולריות target מהתערבות מכנית; הוא כן **self-supervised / mechanically supervised**, משום שהפיקוח נוצר אוטומטית מחוקי המשימה ולא מתוויות אנושיות או מתוויות correctness טבעיות.

---

## 3. נקודת המוצא: מה IU-PCR עושה

### 3.1 בעיית המיזוג

בכל response אנו מחשבים אוסף של מדדי טלמטריה: entropy, varentropy, top-k margin, אנרגיית token שנדגם, דינמיקה ספקטרלית לאורך ה־trace, CUSUM, אורך trace ועוד. נסמן את וקטור הפיצ'רים המכוון כך ש”גדול = יותר ביטחון” ב־

\[
f(x)=(f_1(x),\ldots,f_m(x))^T.
\]

אנו רוצים ציון ליניארי

\[
s(x)=w^T f(x),
\]

שידרג responses תקינים מעל hallucinations. אילו היו תוויות \(Y\), משקל הרגרסיה האופטימלי היה קשור ל־

\[
Cw=\rho,
\qquad
C=\operatorname{Cov}(f),
\qquad
\rho=\operatorname{Cov}(f,Y).
\]

הבעיה היא ש־\(C\) ניתן לאמידה בלי labels, אך \(\rho\) אינו נצפה.

הבסיס התיאורטי מגיע מ־Tenzer et al., *Crowdsourcing Regression: A Spectral Approach*. המאמר מנסח U-PCR כשיטת ensemble regression ללא ground truth ומראה שהמשקלים האופטימליים מקיימים \(\rho=Cw^*\). ראו [דף המאמר ב־PMLR](https://proceedings.mlr.press/v151/tenzer22a.html) ו־[ה־PDF](https://proceedings.mlr.press/v151/tenzer22a/tenzer22a.pdf).

### 3.2 איך IU-PCR מעריך את \(\rho\) בלי labels

תחת מודל additive-error של experts, ה־off-diagonal covariance בין שני experts מאפשר מערכת משוואות אדיטיבית שממנה ניתן לאמוד את הקורלציה של כל expert עם היעד. במימוש הפרויקט:

1. מחשבים \(C=FF^T/n\), כאשר שורות \(F\) הן הפיצ'רים והעמודות הן דוגמאות.
2. פותרים מערכת אדיטיבית על ה־off-diagonal entries של \(C\).
3. מזיזים את אומדן \(\rho\) באמצעות פרמטר variance סמוי \(g^2\), הנבחר לפי התאמה לתת־המרחב הספקטרלי הדומיננטי.
4. מקרינים את פתרון הרגרסיה על שתי רכיבי covariance מובילים.

בקונפיגורציה המוגנת של הפרויקט אין feature exclusion, אין difficulty gate, ויש תמיד שני רכיבים. הקבועים המדויקים מופיעים ב־[`IU_FIT_DEFAULTS`](../../spectral_utils/laplacian_upcr.py), והמימוש האלגברי ב־[`upcr_fit`](../../spectral_utils/upcr.py).

אם \(U_K\) מכיל את \(K=2\) eigenvectors המובילים של \(C\), אז צורת ה־PCR היא בקירוב:

\[
w_{IU}
=
U_K\left(U_K^T C U_K\right)^{-1}U_K^T\widehat\rho.
\]

### 3.3 למה IU הוא עוגן ולא “האמת”

IU-PCR מספק baseline חזק, יציב וללא תוויות, אבל הוא אינו פותר לבדו את בעיית הזיהוי:

- covariance משותף יכול להגיע מנכונות, אך גם מאורך תשובה, סגנון, קושי, tokenizer או דומיין;
- הכיוון בעל הווריאנס הגבוה ביותר אינו בהכרח הכיוון של hallucination;
- אומדן \(\rho\) נשען על הנחות ensemble שאינן מבטיחות הפרדה מלאה בין target ל־nuisance.

לכן כל השיטות המאוחרות אינן מחליפות את IU מיד. הן בונות correction סביבו, כאשר “אפס correction” מחזיר את IU בדיוק.

---

## 4. HARP המקורי: מה השיטה במאמר באמת עושה

### 4.1 ההנחה המבנית

HARP — *Hallucination Detection via Reasoning Subspace Projection* — הוא מאמר arXiv מאת Hu et al. המציע לפרק את מרחב ה־hidden states לסכום ישר אורתוגונלי:

\[
\mathcal H_l
=
\mathcal S_{Semantic}
\oplus
\mathcal S_{Reasoning}.
\]

הטענה היא שה־unembedding ממפה את החלק הסמנטי אל logits, בעוד שחלק מן המידע הדרוש ל־reasoning נמצא בכיוונים שכמעט אינם משפיעים ישירות על ה־logits.

המאמר מתאר את הרעיון בקיצור כך: “the former encodes linguistic expression and the latter captures internal reasoning processes.” זהו ציטוט קצר מן [התקציר הרשמי ב־arXiv](https://arxiv.org/abs/2509.11536).

### 4.2 SVD של מטריצת ה־unembedding

נסמן את מטריצת ה־unembedding ב־\(W_{unemb}\). HARP מחשב

\[
W_{unemb}=U\Sigma V^T.
\]

במקרה האידאלי:

- right singular vectors בעלי singular values לא־אפסיים פורשים את תת־המרחב הסמנטי;
- null-space directions של \(W_{unemb}\) פורשים את תת־המרחב ה־reasoning.

במודל ממשי singular values אינם אפס בדיוק. לכן HARP משתמש ב־rank-\(k\) approximation, שומר בערך 95% מן הכיוונים כ־semantic, ומתייחס לכ־5% trailing directions כתת־מרחב reasoning. עבור hidden state \(h_l\), הייצוג הוא

\[
\operatorname{proj}_R(h_l)=V_R^T h_l.
\]

הנוסחאות וההצדקה מופיעות בסעיפים 4.2–4.3 של [המאמר המקומי המחולץ](../../papers/extracted/harp-hallucination-detection-via-reasoning-subspace-projecti.md) וב־[מקור arXiv](https://arxiv.org/abs/2509.11536).

### 4.3 HARP אינו detector לא־מפוקח

לאחר ההקרנה, HARP מאמן detector פרמטרי \(g_\theta\) על תוויות hallucination באמצעות binary cross-entropy. הוא מחשב score לכל token ולוקח maximum לאורך התשובה:

\[
g_\theta(y\mid x)
=
\max_i g_\theta\!\left(\operatorname{proj}_R(h_l^{(i)})\right).
\]

המאמר מדווח, בין היתר, AUROC של 92.8% על TriviaQA עם Qwen-2.5-7B-Instruct, שיפור של 7.5 נקודות אחוז מעל ה־baseline הטוב הבא. זו תוצאת המאמר, לא תוצאה של הפרויקט שלנו. המקור הוא טבלה 1 והתקציר של [HARP ב־arXiv](https://arxiv.org/abs/2509.11536).

חשוב לומר למנחים במפורש:

- HARP הוא **white-box**: צריך hidden states ו־unembedding weights;
- HARP הוא **supervised** בשלב detector;
- HARP אינו baseline הוגן ישיר מול IU/NRM, שפועלים מטלמטריית הסתברויות ואינם מקבלים labels ב־fit;
- HARP הוא כרגע arXiv preprint; לא נמצאה בקבצי הפרויקט ראיה לפרסום peer-reviewed.

### 4.4 מה לקחנו מ־HARP — ומה לא

לא לקחנו את hidden states, את \(W_{unemb}\), את חלוקת 95%/5%, או את ה־BCE classifier.

לקחנו עיקרון ארכיטקטוני:

> לפני שמבקשים מהמסווג לזהות hallucination, כדאי להסיר או לבודד תת־מרחב דומיננטי שמייצג מידע משותף אך עלול להיות nuisance.

בניסוח של מסמך המקור המקומי: “Separate a known nuisance or semantic subspace before asking the fusion method to identify hallucination-related variation.” ראו [הערת ההשראה של HARP](harp_subspace_inspiration_2026-08-12.md).

---

## 5. שלב הביניים החשוב: HARP-inspired Contribution Space

NRM לא הופיע ישירות. תחילה היה צריך להראות שיש בכלל correction שימושי בתוך מנגנון IU.

### 5.1 פירוק ציון IU לתרומות משפחה

הפיצ'רים חולקו מראש לשש משפחות provenance:

1. `entropy_level`
2. `entropy_dynamics`
3. `sampled_token_energy`
4. `partition_energy`
5. `topk_distribution`
6. `structural`

הרישום המלא נמצא ב־[`FEATURE_TO_VIEW`](../../spectral_utils/specrage_views.py). החלוקה מבוססת על המקור החישובי של הפיצ'ר, לא על AUROC שלו ולא על correctness labels.

אם \(I_g\) היא קבוצת האינדקסים של משפחה \(g\), מגדירים:

\[
h_g(x)=\sum_{i\in I_g}w_i f_i(x).
\]

סכום התרומות משחזר בדיוק את IU:

\[
s_{IU}(x)=\sum_g h_g(x)=w^Tf(x).
\]

המימוש ב־[`iu_family_contributions`](../../spectral_utils/contribution_subspace.py) בודק את השחזור נומרית ונכשל אם הוא אינו מתקיים.

### 5.2 למה מסירים מכל תרומה את IU

אם ננתח ישירות את \(h_g\), רוב הווריאציה עשויה להיות פשוט “אותו signal שכבר נמצא ב־IU”. לכן:

1. מתקננים את ציון IU ל־\(b\);
2. מתקננים כל תרומת משפחה \(h_g\);
3. מרגרסים כל תרומה על \(b\);
4. לוקחים את השארית ומתקננים אותה שוב.

פורמלית, אם \(\widetilde h_g\) היא התרומה המתוקננת:

\[
\ell_g
=
\frac{b^T\widetilde h_g}{b^Tb},
\qquad
r_g
=
\frac{\widetilde h_g-\ell_gb-\mu_{res,g}}{\sigma_{res,g}}.
\]

מטריצת \(R=[r_1,\ldots,r_G]\) מתארת **מי מן המשפחות מסכימה או חולקת על IU, מעבר לציון IU עצמו**.

זהו analogue מוגבל ל־HARP:

- HARP מסיר את תת־המרחב הסמנטי של hidden states;
- אנחנו מסירים מכל תרומת משפחה את תת־המרחב החד־ממדי של ציון IU המשותף.

אין כאן טענה ששני המרחבים זהים. הדמיון הוא במבנה “shared component + residual component”.

### 5.3 המורה המפוקח: הוכחת היתכנות בלבד

בשלב הראשון אומן head מפוקח:

\[
s_\delta(x)=b(x)+R(x)\delta.
\]

המקדם של \(b\) נשאר 1. כאשר \(\delta=0\), מקבלים בדיוק את דירוג IU. \(\delta\) נלמד באמצעות class-balanced logistic loss עם L2 prior סביב אפס.

התוצאה הראשית הייתה:

- IU cell-macro AUROC: 0.7698;
- anchored contribution head: 0.7778;
- שיפור: **+0.800pp**;
- 21 wins ו־2 losses;
- equal-family bootstrap interval: **[+0.309,+1.108]pp**.

ראו [`SPEC_HARP_CONTRIBUTION_SUBSPACE_IU_V1.md`](../../SPEC_HARP_CONTRIBUTION_SUBSPACE_IU_V1.md) ו־[דוח ה־PoC](../../results/harp_contribution_subspace_poc_v1/REPORT.md).

המשמעות אינה “מצאנו שיטה לא־מפוקחת”. המשמעות הצרה היא:

> במרחב שש תרומות המשפחה קיים target correction קטן, יציב ומועיל, אם נותנים למודל תוויות כדי למצוא אותו.

מכאן נולדה השאלה של NRM: האם ניתן לבחור כיוון correction דומה **בלי labels**?

---

## 6. Family-NRM: האלגוריתם המדויק

NRM בפרויקט פירושו **Neutral Residual Mode Contribution-Subspace IU**. זהו שם מקומי, לא שם של אלגוריתם סטנדרטי בספרות.

### 6.1 קלט

לכל source cell \(c\):

- מטריצת mixed-v2 של הפיצ'רים;
- משקלי IU-PCR \(w_c\);
- תרומות משפחה \(H_c\);
- residual matrix \(R_c\) לאחר הסרת IU ותקנון.

אין correctness labels ב־API של calibration.

### 6.2 covariance שאריתי בין משפחות

בכל cell מחשבים:

\[
C_c=\frac{1}{n_c}R_c^TR_c.
\]

לאחר מכן ממוצעים כל entry על פני source cells שבהם שתי המשפחות קיימות:

\[
C_R[g,h]
=
\operatorname{mean}_{c:\,g,h\in c} C_c[g,h].
\]

לבסוף מסמטרים:

\[
C_R\leftarrow\frac{C_R+C_R^T}{2}.
\]

הסיבה ל־pairwise averaging היא שמשפחה חסרה אינה אמורה להיחשב כ־residual אפס. הקוד נמצא ב־[`fit_neutral_residual_mode_calibration`](../../spectral_utils/contribution_subspace.py), והמפרט ב־[`SPEC_NEUTRAL_RESIDUAL_MODE_CS_IU_V1.md`](../../SPEC_NEUTRAL_RESIDUAL_MODE_CS_IU_V1.md).

### 6.3 למה eigenvalue קרוב ל־1 נקרא “neutral”

כל residual column מתוקנן ל־variance 1. לכן, אילו המשפחות היו בלתי תלויות לחלוטין לאחר הסרת IU, היינו מצפים בקירוב ל־

\[
C_R\approx I.
\]

מכאן נובעת ההיוריסטיקה:

- \(\lambda\gg1\): כמה משפחות נעות יחד בכיוון משותף חזק. זה יכול להיות target, אך יכול באותה מידה להיות nuisance כגון אורך, סגנון או קושי;
- \(\lambda\approx0\): redundancy או תלות כמעט דטרמיניסטית; הכיוון אינו מוסיף מידע עצמאי;
- \(\lambda\approx1\): mode בעל variance דומה ל־unit-variance null, שאינו spike משותף חזק ואינו redundancy.

NRM מבצע eigendecomposition:

\[
C_Rv_j=\lambda_jv_j,
\]

ובוחר:

\[
j^*=\arg\min_j|\lambda_j-1|.
\]

זהו **כלל מבני**, לא theorem של target identification. המפרט עצמו אומר במפורש שהכלל “encodes a structural assumption, not an identifiability theorem”.

### 6.4 בעיית הסימן וה־anchor

Eigenvector מוגדר רק עד כדי סימן: \(v\) ו־\(-v\) הם אותו mode. בלי target labels אי אפשר להסיק מן ה־eigendecomposition איזה סימן הוא “correctness”.

Family-NRM פותר זאת באמצעות equal-family confidence anchor:

\[
\operatorname{sign}(v)
\leftarrow
\operatorname{sign}(\mathbf 1^Tv).
\]

אם המכפלה בדיוק אפס, נבחר הסימן שבו הרכיב בעל הערך המוחלט הגדול ביותר חיובי.

ה־anchor אפשרי מפני שכל הפיצ'רים כבר כוונו מראש ל־confidence polarity. אך חשוב להבין: זהו prior סימטרי, לא ראיה שה־mode הוא hallucination direction.

### 6.5 יישום על target cell

ב־target חדש:

1. מתאימים IU-PCR ללא labels;
2. מפרקים למשפחות;
3. מתאימים את transform של \(b\) ו־\(R\) על target batch ללא labels;
4. מצמצמים את \(v\) למשפחות הקיימות;
5. מחשבים raw correction:

\[
q=Rv.
\]

6. מתקננים את גודל התיקון ל־standard deviation קבוע \(1/G\):

\[
\delta=\frac{1}{G}\frac{v}{\operatorname{sd}(Rv)},
\qquad
s_{NRM}=b+R\delta.
\]

בכתיבה מקוצרת:

\[
s_{NRM}
=
b+rac{Rv}{G\operatorname{sd}(Rv)}.
\]

כאשר ה־mode דגנרטיבי או בעל variance אפס, התיקון אפס ומוחזר IU בדיוק.

### 6.6 מדוע התיקון אורתוגונלי ל־IU

כל עמודת \(R\) נוצרה לאחר regression על \(b\), ולכן על fit rows:

\[
\operatorname{Cov}(b,R_g)\approx0.
\]

מכאן גם:

\[
\operatorname{Cov}(b,R\delta)\approx0.
\]

כלומר NRM אינו “מגדיל שוב את IU”; הוא מוסיף rank correction מתוך דפוסי disagreement בין המשפחות שאינם ליניארית מוסברים על ידי IU.

### 6.7 למה NRM עדיין head אפיני יחיד

לכאורה ביצענו pipeline מורכב: family sums, standardization, residualization ו־mode projection. אך לאחר שה־transform והכיוון קפואים, כל הפעולות ליניאריות ביחס ל־mixed-v2 coordinates.

המימוש מחשב במפורש effective weight vector ו־intercept כך ש־

\[
s_{NRM}(z)=b_{eff}+w_{eff}^Tz.
\]

אין detector שני, אין pass נוסף דרך המודל, ואין nonlinear fitted head אחרי mixed-v2. האפיניות היא ביחס ל־**mixed-v2 transformed coordinates**, לא ביחס לטלמטריה הגולמית; mixed-v2 עצמו כולל transforms לא־ליניאריים קפואים.

---

## 7. אינטואיציה עמוקה: מה NRM מנסה לסנן

נניח שלאחר הסרת IU מתקבלים שישה residuals. אפשר לחשוב על שלושה סוגי כיוונים:

### 7.1 כיוון shared nuisance

לדוגמה, תשובות ארוכות גורמות יחד לעלייה ב־entropy dynamics, sampled-token energy ו־structural length. residuals רבים ינועו יחד. כיוון כזה יקבל eigenvalue גדול.

NRM אומר: “אינני סומך אוטומטית על הכיוון החזק ביותר, כי תלות חזקה יכולה להיות nuisance.”

### 7.2 כיוון redundancy

אם שני summary features כמעט זהים, קיים contrast כגון \(r_1-r_2\) בעל variance כמעט אפס. זהו eigenvalue קטן מאוד.

NRM אומר: “כיוון שכמעט מתאפס בגלל שכפול מדידות אינו correction עצמאי.”

### 7.3 כיוון neutral

כיוון שה־variance שלו דומה ל־1 לאחר תקנון אינו מוסבר על ידי common spike חזק ואינו cancellation כפול. ההימור של NRM הוא שבמרחב שש המשפחות, כיוון כזה עשוי לשמר disagreement שימושי הקשור ל־correctness.

הנקודה העדינה היא שההיוריסטיקה אינה אומרת שכל target חייב להיות neutral. target אמיתי עשוי להיות spike, ו־neutral mode עשוי להיות noise. ההצלחה של Family-NRM היא אמפירית ומותנית בייצוג המשפחתי.

---

## 8. מה Family-NRM השיג בפועל

### 8.1 calibration קפוא

ה־eigenvalue שנבחר היה:

\[
\lambda^*=1.035378.
\]

כיוון המשפחות היה:

| משפחה | coefficient |
|---|---:|
| entropy_level | +0.093928 |
| entropy_dynamics | -0.113808 |
| sampled_token_energy | -0.673995 |
| partition_energy | +0.714635 |
| topk_distribution | +0.112033 |
| structural | +0.026490 |

הסימנים האלה תאמו את ששת הסימנים של ה־global supervised teacher, אף שה־NRM calibration לא קיבל labels. זהו ממצא מעניין, אך אין לפרשו כהוכחת identifiability.

### 8.2 retrospective transfer

מול IU-PCR:

| תחום | שינוי AUROC | 95% interval | W/L |
|---|---:|---:|---:|
| original 23, LOFO | +0.277pp | [+0.016,+0.533] | 15/8 |
| Qwen ProcessBench | +0.557pp | [+0.236,+0.828] | 7/1 |
| Llama ProcessBench | +1.580pp | [+0.918,+2.346] | 4/0 |
| SemGrad | +1.310pp | [+0.205,+2.415] | 2/0 |

ראו [דוח NRM המקורי](../../results/neutral_residual_mode_cs_iu_v1/REPORT.md).

### 8.3 confirmation חיובי ומוגבל: PRMBench

ב־PRMBench/Qwen3-8B, תחת פרוטוקול response-level:

- IU: 0.720602 AUROC;
- NRM: 0.725206;
- שינוי: **+0.460pp**;
- paired grouped interval: **[+0.068,+0.841]pp**;
- כל חמשת ה־gates עברו.

ראו [דוח PRMBench](../../results/neutral_residual_mode_prmbench_v1/REPORT.md).

אין להציג זאת כ־PRMBench step-level SOTA. הדוח מבהיר שזהו response-level correct-vs-error adaptation.

### 8.4 confirmation שלא עבר: HLE

ב־HLE/Qwen2.5-72B:

- IU: 0.516775;
- NRM: 0.520229;
- שינוי נקודתי: +0.345pp;
- interval: [-0.898,+1.628]pp;
- gate של lower bound חיובי נכשל.

לכן זהו **FAIL של confirmation**, לא success. המדגם כלל רק 68 תשובות שסומנו נכונות, תחת judge ביניים שאינו פרוטוקול HLE המקורי. ראו [דוח HLE](../../results/neutral_residual_mode_hle_v1/REPORT.md).

### 8.5 הסיכום המדעי הנכון

Family-NRM הוא bounded positive result:

- יש evidence שתיקון family-residual label-free יכול לשפר IU;
- יש confirmation חיובי אחד בתחום response-level;
- ההשפעה קטנה אך עקבית בכמה transfer surfaces;
- אין בסיס לטעון שהוא detector אוניברסלי;
- אין identifiability theorem;
- הצלחתו תלויה ב־manual provenance grouping.

---

## 9. מדוע Atomic-NRM נכשל — ולמה הכישלון חשוב

### 9.1 המוטיבציה ל־Atomic-NRM

החלוקה לשש משפחות היא prior ידני. אם רוצים שיטה group-free, טבעי לנסות לעבוד ישירות עם כל פיצ'ר אטומי.

ב־Atomic-NRM נשמרו 17 פיצ'רים שהיו קיימים ולא־דגנרטיביים בכל 23 source cells. במקום לבחור eigenvector בודד קרוב ל־1, נבנה permutation-null simultaneous interval:

\[
[0.934489,1.070026],
\]

ונשמר כל תת־המרחב הניטרלי; במקרה הקפוא dimension היה 2 עם eigenvalues 0.960685 ו־1.025557. בכך תוקנה מראש בעיית rotation של eigenvectors כמעט־שווי־ערך.

### 9.2 התוצאה

למרות stability מבני גבוה, התוצאה הייתה שלילית:

| תחום | Atomic projector מול IU | Family-NRM מול IU |
|---|---:|---:|
| original 23 | -0.667pp | +0.277pp |
| Llama ProcessBench | -1.106pp | +1.580pp |
| Qwen ProcessBench | -1.305pp | +0.557pp |
| SemGrad | -4.216pp | +1.310pp |

ראו [ה־structural audit](../../results/atomic_nrm_structural_audit_v1/report.md) ו־[ה־retrospective controls](../../results/atomic_nrm_retrospective_controls_v1/REPORT.md).

### 9.3 מה לא היה הבעיה

הכישלון אינו מוכיח שאין target signal בפיצ'רים האטומיים. להפך, supervised atomic head הראה יותר headroom מן family head. כלומר:

- המידע קיים;
- geometry סביב eigenvalue 1 אינה יודעת לבחור אותו;
- feature-order stability אינה target validity;
- null-like spectrum אינו semantic label.

זהו אחד הלקחים המרכזיים של התזה:

> Unsupervised stability, repeatability או “neutrality” יכולים לזהות מבנה אמיתי אך לא בהכרח מבנה של נכונות.

### 9.4 האבחנה שהובילה ל־PTNI

הניסוח המדויק בהצעת ההיבריד הוא:

> “NRM can reject strong dependence and redundancy, but it lacks a defensible group-free steering signal.”

כלומר NRM יודע לומר **ממה להיזהר**, אך לא **מהו target**.

---

## 10. PTNI: מהו השינוי העקרוני

PTNI פירושו **Paired Target/Nuisance Intervention IU**. במקום לנסות להסיק target מתוך covariance של responses טבעיים, הוא בונה מערך ניסויי שבו target ונ nuisance ידועים מכנית.

### 10.1 הבעיה ש־PTNI פותר

בנתונים טבעיים, אם feature עולה בתשובה שגויה, איננו יודעים אם הסיבה היא:

- השגיאה עצמה;
- prompt קשה יותר;
- response ארוך יותר;
- tokenizer;
- סגנון;
- domain;
- scorer/model family.

אפילו labels טבעיים אינם פותרים causal confounding בלי עיצוב ניסוי. PTNI בונה contrasts שבהם אותו prompt ואותו response מופיעים בשתי הפולריוֹת, ולכן prompt-only או response-only shortcut אינו יכול להסביר את target effect.

### 10.2 reciprocal 2×2 crossover

לכל source group בונים שני worlds תקינים ושווי־קושי:

- \(P_A\) עם תשובה ייחודית \(a\);
- \(P_B\) עם תשובה ייחודית \(b\neq a\).

בונים שתי responses:

- \(R_A\) שמבטאת \(a\);
- \(R_B\) שמבטאת \(b\).

ואז סורקים את כל ארבעת הצירופים:

| | response \(R_A\) | response \(R_B\) |
|---|---|---|
| prompt \(P_A\) | valid | invalid |
| prompt \(P_B\) | invalid | valid |

אותו \(R_A\) הוא byte-identical כאשר הוא valid וכאשר הוא invalid; גם \(R_B\). כל prompt וכל response מופיעים בדיוק 50/50 בשתי הפולריוֹת. לכן target הוא **יחס ההתאמה בין prompt ל־response**, לא זהות prompt או response.

### 10.3 renderings כ־nuisance interventions

כל quartet מופיע ב־canonical rendering ובשלוש וריאציות semantics-preserving כגון paraphrase, layout ו־notation. ה־task AST והאמת נשארים זהים, אך פני השטח משתנים.

כך PTNI מקבל שני סוגי שינוי:

- target intervention: valid ↔ invalid כאשר response קבוע;
- nuisance intervention: canonical ↔ alternate rendering כאשר המשמעות קבועה.

### 10.4 target contrast \(\tau\)

נסמן את וקטור mixed-v2 המתוקנן ב־\(z(P,R,r)\), כאשר \(r\) הוא rendering.

ה־prompt-balanced invalid-minus-valid target effect הוא:

\[
\tau_r
=
\frac12\Big[
z(P_B,R_A,r)-z(P_A,R_A,r)
+z(P_A,R_B,r)-z(P_B,R_B,r)
\Big].
\]

המחצית הראשונה מחזיקה את \(R_A\) קבוע ומשנה valid→invalid. המחצית השנייה מחזיקה את \(R_B\) קבוע ועושה אותו דבר בכיוון הסימטרי. הממוצע מבטל prompt/world marginal.

### 10.5 nuisance contrast \(\nu\)

לכל אחד מארבעת התאים \(c\):

\[
\nu_{c,r}
=
z(c,r)-z(c,canonical).
\]

זהו השינוי הנגרם רק מן ה־rendering.

### 10.6 target-by-render interaction \(\iota\)

\[
\iota_r
=
\tau_r-\tau_{canonical}.
\]

אם feature נראה target-sensitive רק תחת punctuation מסוים, interaction יהיה גדול. PTNI אינו מסתפק בכיוון שמפריד valid/invalid ב־canonical; הוא דורש שהאפקט ישרוד nuisance renderings.

### 10.7 nuisance-whitened target direction

מחשבים:

\[
\mu_T=E[\tau],
\]

\[
S_T=\operatorname{Cov}_{pop}(\tau),
\quad
S_N=E[\nu\nu^T],
\quad
S_I=E[\iota\iota^T],
\]

\[
S=S_T+S_N+S_I.
\]

עבור ridge \(\lambda\):

\[
r_0(\lambda)
=
(S_{scaled}+\lambda I)^{-1}\mu_T.
\]

זו נוסחה דמוית LDA/whitened mean direction:

- המונה \(\mu_T\) מבקש target effect עקבי;
- המכנה מעניש variance של target בין קבוצות, nuisance drift ו־target×render interaction;
- ridge מונע inversion לא־יציב.

הסימן אינו שרירותי: \(\tau\) מוגדר invalid-minus-valid, ולכן \(r_0^Tz>0\) פירושו “יותר intervention-identified error”. זו בדיוק ראיית ה־steering שחסרה ל־NRM.

### 10.8 IU-orthogonal trust path

ב־target model/batch מתאים IU-PCR ללא labels ומקבלים confidence weight \(u\). לאחר מכן מסירים מ־\(r_0\) את הרכיב המקביל ל־IU תחת covariance metric \(C\):

\[
r_\perp
=
r_0-u\frac{u^TCr_0}{u^TCu}.
\]

מכאן:

\[
u^TCr_\perp=0.
\]

אם השארית קטנה מדי, דגנרטיבית או לא יציבה, structural evidence מוגדר כאפס. אחרת מנרמלים את \(r_\perp\) לנורמת IU ובוחרים trust \(\alpha\) מתוך grid קפוא:

\[
s_\alpha(z)
=
u^Tz-\alpha r_\perp^Tz,
\qquad
\alpha\in\{0,0.0625,0.125,0.25,0.5,1\}.
\]

הסימן מינוס קיים מפני ש־\(r_\perp\) הוא risk direction בעוד \(u\) הוא confidence direction.

כאשר \(\alpha=0\):

\[
s_0(z)=u^Tz=s_{IU}(z).
\]

זו אינה “fallback דומה ל־IU”; זו זהות נומרית מתוכננת.

### 10.9 כיצד נבחרים \(\lambda\) ו־\(\alpha\)

הבחירה נעשית ב־nested source-group folds על ה־mechanical quartet objective, לא על natural hallucination labels. לכל score \(f\), contrast הביטחון הוא:

\[
\Delta_f
=
\frac12\Big[
f(P_A,R_A)-f(P_B,R_A)
+f(P_B,R_B)-f(P_A,R_B)
\Big].
\]

אם \(\Delta_f>0\), valid responses קיבלו confidence גבוה יותר. objective \(J_f\) הוא macro-average של wins עם חצי נקודה ל־tie. Arms עם nuisance או interaction ratios גבוהים אינם feasible; one-SE rule מעדיף alpha קטן יותר ו־ridge גדול יותר.

המפרט המלא נמצא ב־[`AUTOMATIC_GROUP_FREE_IU_PHASE_A6_V1.md`](../experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_V1.md), במיוחד סעיף 5.

---

## 11. מה PTNI מוסיף לעומת NRM

### 11.1 target identification

**NRM:** בוחר mode בגלל eigenvalue, לא בגלל שנצפה שינוי correctness.  
**PTNI:** מגדיר invalid-minus-valid contrast שידוע מכנית.

זהו השיפור החשוב ביותר.

### 11.2 sign identification

**NRM:** חייב equal-ones anchor כדי לבחור בין \(v\) ל־\(-v\).  
**PTNI:** הסימן מגיע מהגדרת ההתערבות. אין label switching.

### 11.3 nuisance measurement

**NRM:** מניח ש־large shared modes הם חשודים כ־nuisance.  
**PTNI:** מודד nuisance ישירות באמצעות semantics-preserving renders ומכניס אותו ל־\(S_N\).

### 11.4 interaction robustness

**NRM:** אינו יודע אם target effect תלוי בפורמט.  
**PTNI:** מודד \(\iota_r\) ומעניש כיוון שאינו יציב בין renderings.

### 11.5 group-free atomic coordinates

**Family-NRM:** תלוי בשש משפחות provenance ידניות.  
**PTNI:** פועל על ה־atomic mixed-v2 roster ואינו דורש חלוקה ידנית למשפחות כדי לזהות target.

### 11.6 held-family and held-scorer tests

PTNI preregisters leave-one-target-family-out, leave-one-nuisance-family-out והעברה חד־כיוונית מ־Qwen ל־Llama. NRM calibration המקורי למד כיוון trans-environment, אך לא נבנה מלכתחילה סביב התערבויות מסוג held family.

### 11.7 מה PTNI עדיין אינו פותר

PTNI יכול להיכשל אם:

- הוא לומד forced prompt-response incompatibility שאינה דומה ל־natural hallucination;
- directions אינם עוברים בין scorer families;
- mechanical construction מכיל shortcut;
- target effect חלש מדי לאחר nuisance whitening;
- הכיוון אינו מוסיף מעבר ל־IU;
- natural-response veto או PopQA confirmation נכשלים.

לכן PTNI אינו “תווית קסם”. הוא משפר את identifiability של הניסוי, אך עדיין חייב להוכיח transfer לטעויות טבעיות.

---

## 12. מהו PTNI-guided NRM המוצע

### 12.1 הבעיה שההיבריד מנסה לפתור

PTNI לבדו נותן כיוון target+nuisance-whitening, אך הכיוון עלול:

- לכלול unstable atomic residual directions;
- להיגרר על ידי near-redundant features;
- להשתנות בין folds כאשר covariance לא מותנה היטב;
- להתאים למכניקת ההתערבות ולא למבנה transferable.

NRM לבדו מספק regularization ספקטרלי, אך אינו יודע מהו target.

לכן ההיבריד שואל:

> האם ניתן להשתמש ב־PTNI כ־steering signal בתוך תת־מרחב neutral יציב, במקום לבקש מן הספקטרום לזהות target לבדו?

### 12.2 Option C — ההצעה המועדפת

הצעת המחקר ממליצה על השלבים הבאים:

1. fit של target-local IU-PCR ומקבלים \(u\);
2. בניית atomic IU-residual coordinates;
3. אמידת residual covariance;
4. בניית permutation-null spectrum;
5. שמירת כל neutral eigenvalue cluster, לא eigenvector יחיד;
6. יצירת projector יציב \(P_N\);
7. fit של PTNI steering direction \(r_{PTNI}\);
8. projection:

\[
r_{hybrid,0}=P_Nr_{PTNI};
\]

9. הסרת רכיב IU מקביל:

\[
r_{hybrid,\perp}
=
r_{hybrid,0}
-u\frac{u^TCr_{hybrid,0}}{u^TCu};
\]

10. normalization ו־trust path:

\[
w_{hybrid}(\alpha)
=
u+\alpha r_{hybrid,\perp},
\]

עם התאמת סימן confidence/risk לפי convention המדויק של המימוש, ועם \(\alpha=0\) כ־IU זהה.

### 12.3 למה projector של subspace ולא eigenvector יחיד

כאשר כמה eigenvalues קרובים, eigenvectors יכולים להסתובב באופן חד תחת perturbation קטן, אף שה־span שלהם יציב. לכן עדיף:

\[
P_N=V_NV_N^T
\]

עבור כל eigenvectors שנכנסו ל־neutral band קפוא.

אז PTNI בוחר את הכיוון **בתוך ה־span**, במקום ש־NRM יבחר בסיס שרירותי מתוך אותו span.

זהו חיבור מתמטי טבעי:

- הספקטרום קובע את feasible geometric region;
- PTNI קובע את target-oriented vector;
- projection משמר רק את החלק של PTNI שנמצא באזור הניטרלי.

### 12.4 מה ההיבריד אמור לפתור

אם יצליח, ההיבריד עשוי:

1. להסיר את התלות בשש משפחות provenance;
2. למנוע selection שרירותי של neutral eigenvector;
3. לדכא redundancy ו־shared nuisance לפני deployment;
4. לייצב את כיוון PTNI בין folds/scorers;
5. להקטין nuisance and interaction drift;
6. לשמור exact IU fallback.

### 12.5 מה ההיבריד אינו רשאי לטעון מראש

- neutral subspace אינו בהכרח המקום שבו target נמצא;
- projection יכול למחוק target signal;
- improvement יכול להיות רק shrinkage אל IU;
- PTNI יכול להיות טוב מספיק לבדו, ואז NRM מיותר;
- אם PTNI נכשל, NRM אינו רשאי “להציל” אותו.

הצעת המחקר אומרת במפורש: אם PTNI לא מבסס target premise, אין לפתוח את ההיבריד.

---

## 13. ההבדל בין שלוש טענות שקל לבלבל

### טענה א: “קיים signal אטומי של correctness”

יש לכך evidence מפוקח: supervised atomic head מצא headroom.

### טענה ב: “אפשר לזהות את ה־signal הזה ללא labels מתוך covariance”

Atomic-NRM דחה את הטענה בגרסה שנבדקה. neutral geometry לבדה לא הספיקה.

### טענה ג: “אפשר לזהות אותו בעזרת supervision מכני של PTNI”

זו הטענה שניסוי A6 אמור לבדוק. טרם התקבלה תוצאת detector.

PTNI-guided NRM מוסיף טענה רביעית:

### טענה ד: “לאחר ש־PTNI מזהה target, neutral projector מוסיף regularization מועיל מעבר ל־PTNI”

זו שאלה עתידית נפרדת. אפילו אם PTNI מצליח, אין להסיק שההיבריד יצליח.

---

## 14. טבלת השוואה מלאה

| מאפיין | HARP | Family-NRM | Atomic-NRM | PTNI-IU | PTNI-guided NRM |
|---|---|---|---|---|---|
| קלט | hidden states + unembedding | mixed-v2 + IU contributions | atomic mixed-v2 contributions | intervention mixed-v2 + natural IU anchor | PTNI direction + atomic neutral projector |
| white-box | כן | לא | לא | לא | לא |
| natural labels ב־fit | כן | לא | לא | לא | לא |
| mechanical supervision | לא | לא | לא | כן | כן |
| target sign | labels/BCE | equal-family anchor | symmetric anchor | invalid-minus-valid | PTNI |
| nuisance definition | semantic subspace | shared residual spikes | permutation-neutral spectrum | render nuisance + interaction | PTNI nuisance + spectral filter |
| ידני groups | לא | כן, 6 families | לא | לא לזיהוי target | לא |
| exact IU fallback | לא רלוונטי | כן | כן | כן | כן |
| affine אחרי mixed-v2 | לא | כן | כן | כן | כן |
| נבדק אמפירית בפרויקט | לא, רק מאמר | כן | כן ונכשל | עדיין לא כ־detector | לא |

---

## 15. מהו הסטטוס המדויק כיום

לפי [`PROGRESS.md`](../../PROGRESS.md), A6-S0a עבר עם verdict `PASS_S0A`:

- 1,800 reciprocal quartet groups נבנו ואומתו;
- 6,000 prompt-only natural manifest rows נבנו;
- 7,200 inner-fold assignments ו־36 null cells נקבעו;
- 7,800 checkpoints אומתו;
- tokenizer restore וה־boundary אומתו קריפטוגרפית.

אבל S0a הוא **mechanical construction result**, לא תוצאת detection. נכון למועד המסמך:

- אין response telemetry של A6;
- אין תוצאת S1 simulator;
- אין natural correctness sidecar פתוח;
- אין PTNI AUROC;
- אין השוואת PTNI מול IU או Family-NRM;
- PTNI-guided NRM טרם נוסה כלל.

השלב הבא הוא A6-S0b, שבודק shortcuts ומכין matching/null schedules. רק אם הוא יעבור ייפתח S1 simulator, ואחריו — אם יעבור — שלבי telemetry והעברה.

לכן התשובה לשאלה “האם PTNI כבר שיפר את NRM?” היא **לא ידוע**. השאלה טרם נבדקה.

---

## 16. איך להסביר את המחקר למנחים בשתי דקות

אפשר לומר:

> אנחנו מתחילים מ־IU-PCR, שממזג ללא תוויות כ־30 מדדי אי־ודאות מתוך התפלגות ה־token probabilities. הבעיה היא שקו־וריאנס חזק יכול לייצג נכונות, אבל גם אורך, קושי או סגנון.
>
> HARP נתן לנו השראה מבנית: הוא מפריד hidden states לתת־מרחב סמנטי ולתת־מרחב reasoning באמצעות SVD של ה־unembedding, ואז מאמן detector מפוקח על ההקרנה. אנחנו לא יכולים ולא רוצים להעתיק את ה־white-box detector, אבל אימצנו את הרעיון של הפרדת shared structure לפני זיהוי hallucination.
>
> ב־NRM פירקנו את ציון IU לשש משפחות טלמטריה, הסרנו מכל תרומה את מה שכבר מוסבר על ידי IU, ובחרנו mode שאריתי שה־eigenvalue שלו קרוב ל־unit-variance null. זה נתן שיפור קטן אך מאומת ב־PRMBench, אבל Atomic-NRM הראה ש־covariance לבדו אינו יודע למצוא target בלי חלוקת המשפחות הידנית.
>
> PTNI הוא הפתרון שאנחנו בודקים כעת: אנחנו יוצרים reciprocal prompt-response quartets שבהם כל prompt וכל response מופיעים חצי מהזמן כנכונים וחצי כשגויים, ובנפרד משנים רק rendering בלי לשנות משמעות. כך אפשר ללמוד כיוון שמגיב לשגיאה אך לא לפורמט, בלי להשתמש בתוויות hallucination טבעיות. אם PTNI יצליח, נבדוק בהמשך האם הקרנת הכיוון שלו לתוך neutral residual subspace מוסיפה יציבות מעבר ל־PTNI לבדו.

---

## 17. איך להסביר בעשר דקות עם לוח

### שלב 1 — כתבו את העוגן

\[
s_{IU}(z)=u^Tz.
\]

הסבירו ש־\(u\) נלמד מ־feature covariance ללא labels.

### שלב 2 — כתבו את Family-NRM

\[
h_g=\sum_{i\in g}u_i z_i,
\qquad
\sum_gh_g=s_{IU}.
\]

\[
R_g=\operatorname{standardize}\big(\widetilde h_g-\operatorname{proj}_b\widetilde h_g\big).
\]

\[
C_R=E[R^TR/n],
\quad
v^*=\arg\min_v|\lambda(v)-1|.
\]

\[
s_{NRM}=b+\frac{Rv^*}{G\,sd(Rv^*)}.
\]

הסבירו: spikes חשודים כתלות משותפת; near-zero חשוד כ־redundancy; unit-like הוא residual neutral.

### שלב 3 — הציגו את הכשל

כתבו:

\[
\text{geometry}\neq\text{semantics}.
\]

Atomic-NRM היה יציב ספקטרלית אך הפסיד בביצועים. לכן stability אינה target identification.

### שלב 4 — כתבו את PTNI quartet

\[
\begin{array}{c|cc}
&R_A&R_B\\\hline
P_A&valid&invalid\\
P_B&invalid&valid
\end{array}
\]

ואז:

\[
\tau_r=\frac12[(B,A)-(A,A)+(A,B)-(B,B)].
\]

הסבירו שהתגובה קבועה בכל pair ולכן length/style של response אינם משתנים עם הפולריות.

### שלב 5 — nuisance whitening

\[
r_0=(S_T+S_N+S_I+\lambda I)^{-1}\mu_T.
\]

הסבירו שהשיטה מחפשת target mean גדול אך מענישה target heterogeneity, rendering nuisance ו־interaction.

### שלב 6 — IU trust

\[
r_\perp=r_0-u\frac{u^TCr_0}{u^TCu},
\]

\[
s_\alpha=u^Tz-\alpha r_\perp^Tz.
\]

הדגישו: \(\alpha=0\) הוא exact IU.

### שלב 7 — ההיבריד העתידי

\[
r_{hybrid}=P_Nr_{PTNI}.
\]

הסבירו: NRM מספק subspace; PTNI מספק steering.

---

## 18. שאלות קשות שמנחים עשויים לשאול

### “למה eigenvalue קרוב ל־1 אמור להיות hallucination?”

הוא לא אמור בהכרח. זו היוריסטיקה שמסננת shared dependence ו־redundancy. הצלחת Family-NRM הייתה אמפירית; Atomic-NRM הראה שאין identifiability כללית.

### “אם Family-NRM הצליח, למה לא לעצור?”

כי הוא תלוי ב־manual provenance partition, השיפור קטן, HLE לא אישר, והשאיפה היא שיטה כללית יותר שאינה נשענת על grouping ידני.

### “האם PTNI הוא supervised?”

לא באמצעות natural labels, אבל כן באמצעות אמת מכנית שנוצרת מן ההתערבות. המינוח ההוגן הוא mechanically supervised או self-supervised. אם הוא יצליח, אין לטעון שהוא covariance-only unsupervised.

### “למה לא פשוט לאמן logistic regression על ה־quartets?”

זהו control חשוב, אך הוא עלול לנצל capacity ו־intervention shortcuts. PTNI מגביל את הכיוון באמצעות moments, nuisance penalties, IU orthogonality, small trust grid, held-family tests ו־exact fallback.

### “איך יודעים שה־quartets קשורים ל־natural hallucination?”

לא יודעים מראש. לכן הפרוטוקול דורש held-scorer audit, unmodified on-policy mechanical errors, one-way natural-response veto ולבסוף PopQA confirmation. מעבר מ־forced incompatibility ל־natural error הוא hypothesis מרכזי, לא הנחה סמויה.

### “מדוע projection של PTNI לתוך neutral subspace לא ימחק את ה־target?”

הוא בהחלט עלול. לכן ההיבריד מותנה ב־retained target norm/margin gates ונבדק מול PTNI ללא projection, random projectors ו־norm-matched shrinkage. כשל כזה סוגר את ההיבריד.

### “האם NRM ו־PTNI באמת שונים או ששניהם whitening?”

הם שונים:

- NRM עושה eigendecomposition של residual covariance ובוחר לפי null geometry;
- PTNI מחשב target mean מכני ומלבין אותו נגד target/nuisance/interaction moments;
- בהיבריד, NRM הוא projector ו־PTNI הוא steerer.

### “מהי הטענה החדשנית האפשרית?”

הטענה הזהירה היא לא “המצאנו spectral hallucination detection”. היא:

> mechanically identified target/nuisance interventions can steer a one-pass group-free atomic correction around an unsupervised IU anchor; a neutral residual projector may add stable spectral regularization beyond that steering.

החלק השני עדיין לא נוסה.

---

## 19. גבולות ניסוח לתזה

### מותר לומר

- Family-NRM הוא label-free calibration על multiple unlabeled source environments.
- הוא נתן positive transfer בכמה surfaces ו־PRMBench response-level confirmation.
- Atomic-NRM falsified the claim that neutral covariance geometry alone identifies target at atomic resolution.
- PTNI is designed to supply a mechanically identified target direction and explicit nuisance measurements.
- PTNI-guided NRM is a prospective, conditional successor.

### אסור לומר בשלב זה

- “PTNI improves hallucination detection” — טרם קיימת תוצאת detector.
- “PTNI-guided NRM beats NRM” — טרם נוסה.
- “NRM identifies the hallucination eigenvector” — אין theorem כזה.
- “HARP is unsupervised” — detector שלו מאומן עם BCE labels.
- “NRM is HARP on probability features” — זו השראה מבנית, לא העתקה מתמטית.
- “PRMBench step-level confirmed NRM” — האישור היה response-level adaptation.
- “S0a PASS means the method works” — S0a מאשר construction/provenance בלבד.

---

## 20. מילון מונחים

**Atomic feature** — אחד מכ־30 מדדי mixed-v2.  
**Provenance family** — קבוצה ידנית של פיצ'רים מאותו מקור חישובי.  
**IU-PCR** — unsupervised PCR ensemble המעריך weights ללא labels.  
**Contribution** — החלק של ציון IU שמגיע ממשפחה או מפיצ'ר.  
**Residual contribution** — contribution לאחר הסרת projection על IU.  
**Neutral mode** — eigenmode של residual covariance בעל eigenvalue קרוב ל־unit null.  
**Target identification** — היכולת לקבוע איזה כיוון קשור לנכונות ובאיזה סימן.  
**Nuisance** — גורם שמשנה telemetry אך אינו משנה את אמת היעד.  
**Reciprocal quartet** — 2 prompts × 2 responses עם diagonal valid ו־off-diagonal invalid.  
**Mechanical supervision** — target polarity שמופקת מחוק פורמלי, לא מתוויות benchmark.  
**Steering direction** — כיוון target-oriented שמכוון את החיפוש במרחב.  
**Projector** — אופרטור \(P\) ששומר רכיב בתוך subspace ומסיר את המשלים.  
**IU orthogonality** — correction שאינו משכפל ליניארית את ציון IU במטריקת covariance.  
**Trust \(\alpha\)** — גודל התיקון מעבר ל־IU.  
**Exact fallback** — כאשר \(\alpha=0\), מתקבל IU נומרית בדיוק.  
**Trans-environment unsupervised** — calibration ללא labels אך באמצעות כמה batches/environments.  
**Held scorer** — model family שלא שימשה לבחירת הכיוון או hyperparameters.

---

## 21. מקורות ראשיים

### ספרות חיצונית

1. Hu, J. et al. *HARP: Hallucination Detection via Reasoning Subspace Projection*. arXiv:2509.11536. [Abstract and paper](https://arxiv.org/abs/2509.11536).
2. Tenzer, Y., Dror, O., Nadler, B., Bilal, E., & Kluger, Y. *Crowdsourcing Regression: A Spectral Approach*. AISTATS 2022, PMLR 151:5225–5242. [PMLR page](https://proceedings.mlr.press/v151/tenzer22a.html), [PDF](https://proceedings.mlr.press/v151/tenzer22a/tenzer22a.pdf).

### מפרטים ומסמכי מחקר מקומיים

1. [HARP as Future Inspiration for U-PCR](harp_subspace_inspiration_2026-08-12.md).
2. [HARP-inspired Contribution-Subspace IU specification](../../SPEC_HARP_CONTRIBUTION_SUBSPACE_IU_V1.md).
3. [Neutral Residual Mode specification](../../SPEC_NEUTRAL_RESIDUAL_MODE_CS_IU_V1.md).
4. [Atomic Neutral Residual Projector specification](../../SPEC_ATOMIC_NEUTRAL_RESIDUAL_PROJECTOR_CS_IU_CANDIDATE_V1.md).
5. [A6 PTNI-IU protocol](../experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_V1.md).
6. [PTNI-Guided Neutral Residual Mode proposal](ptni_guided_nrm_research_proposal_2026-08-14.md).
7. [Project progress and current boundary](../../PROGRESS.md).
8. [Research roadmap](../../Research_Directions.md).

### מימושים

1. [`upcr_fit`](../../spectral_utils/upcr.py) — IU/U-PCR core.
2. [`IU_FIT_DEFAULTS`](../../spectral_utils/laplacian_upcr.py) — הקונפיגורציה המוגנת.
3. [`FEATURE_TO_VIEW` and `VIEW_ORDER`](../../spectral_utils/specrage_views.py) — שש משפחות provenance.
4. [`contribution_subspace.py`](../../spectral_utils/contribution_subspace.py) — contribution decomposition, residualization, supervised teacher ו־Family-NRM.
5. [`atomic_neutral_residual.py`](../../spectral_utils/atomic_neutral_residual.py) — Atomic-NRM וה־permutation neutral subspace.

### דוחות תוצאה

1. [HARP-inspired supervised contribution PoC](../../results/harp_contribution_subspace_poc_v1/REPORT.md).
2. [Global contribution teacher](../../results/harp_global_contribution_teacher_v1/REPORT.md).
3. [Family-NRM retrospective transfer](../../results/neutral_residual_mode_cs_iu_v1/REPORT.md).
4. [Family-NRM PRMBench confirmation](../../results/neutral_residual_mode_prmbench_v1/REPORT.md).
5. [Family-NRM HLE confirmation failure](../../results/neutral_residual_mode_hle_v1/REPORT.md).
6. [Atomic-NRM structural audit](../../results/atomic_nrm_structural_audit_v1/report.md).
7. [Atomic-NRM retrospective controls](../../results/atomic_nrm_retrospective_controls_v1/REPORT.md).

---

## 22. מסקנה

ההתפתחות המחקרית איננה רצף של “עוד detector”, אלא רצף של שאלות identifiability:

1. **IU-PCR:** האם אפשר למזג מדדי confidence בלי labels? כן, תחת הנחות ensemble.
2. **HARP-inspired teacher:** האם קיים correction שימושי מעבר ל־IU במרחב תרומות נמוך־ממד? כן, עם labels.
3. **Family-NRM:** האם כלל ספקטרלי ללא labels יכול למצוא correction מועיל? במרחב שש המשפחות — evidence חיובי ומוגבל.
4. **Atomic-NRM:** האם neutral geometry לבדה מספיקה בלי manual groups? לא בגרסה שנבדקה.
5. **PTNI:** האם התערבות מכנית יכולה לספק target steering ו־nuisance separation ללא natural labels? זו השאלה הפעילה.
6. **PTNI-guided NRM:** אם PTNI מצליח, האם neutral projector מוסיף regularization אמיתי מעבר ל־PTNI? זו השאלה העתידית.

במילים פשוטות:

> HARP לימד אותנו להפריד מרחבים; NRM ניסה לבצע את ההפרדה בתוך תרומות IU ללא labels; Atomic-NRM הראה שהפרדה ספקטרלית לבדה אינה יודעת מהי נכונות; PTNI מוסיף ניסוי מכני שמגדיר נכונות לעומת nuisance; וההיבריד העתידי יבדוק האם ניתן להשתמש בגאומטריית NRM כמסנן יציב לכיוון ש־PTNI כבר זיהה.
