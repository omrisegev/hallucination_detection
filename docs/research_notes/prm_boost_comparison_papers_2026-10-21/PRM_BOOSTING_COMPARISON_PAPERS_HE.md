# מאמרים להשוואה ל-PRM boosting (2026-10-21)

**מקורות** (כולם בענף `claude/integration-2026-10-17` @ 46b3a63):
- `docs/meetings/prmbench_citing_papers_2026-10-18/SLIDE_HE.html` + `LITERATURE_NOTES_HE.md` + `CITING_PAPERS_2026.csv` (64 מאמרים)
- `results/prmbench_paper_metric_parity_v1/README.md` (חישוב המטריקה של כל מאמר על הנתונים שלנו)
- `docs/research_notes/literature_survey_localization_prm_2026-10-16.md` (Part B, Part C)
- `results/prm_boost_processbench_v1/README.md` (PRISM/CROP על ProcessBench)

**הסתייגות:** רוב המספרים של המאמרים נקראו מהתקצירים ומהטבלאות הרלוונטיות, לא מהמאמר המלא. כל המספרים שלנו על PRMBench הם development data.

---

## עדיפות 1: השוואה ישירה (אותו PRM, אותו benchmark)

### PRISM (arXiv 2606.09078) ★ הכי חשוב
- **מה עושים:** fine-tuning ל-Qwen2.5-Math-PRM-7B על pairwise step comparisons, עם hard negatives ו-curriculum. בלי human labels חדשים, אבל זה אימון.
- **למה הוגן:** אותו PRM ואותו metric. ה-baseline שלהם משוחזר אצלנו: PRMScore 65.5 מול 65.49, ‏FPR 69.34 מול 69.42.
- **למה לא לגמרי הוגן:** הם משנים את המודל, אנחנו מוסיפים שכבה אחריו. אפשר לשלב: boosting על ה-checkpoint שלהם.
- **PRMBench:** PRMScore ‏65.5 → 68.0. ‏NR ‏49.1 → 53.5, ‏NCL ‏55.0 → 66.0, ‏FPR ‏69.3% → 47.1%.
- **שלנו:** CHAIN_N16 ‏68.25, ‏CHAIN_N13 ‏67.82, ‏LSML_CONT ‏69.48 (channels עם labels).
- **הסתייגות חשובה:** הזזת threshold בלבד של ה-PRM המקורי מגיעה ל-68.26 / FPR 47.7, כלומר לנקודה של PRISM. באותו TPR, ‏CHAIN לא טוב מה-PRM עם threshold מוזז. רק LSML_CONT טוב יותר.
- **ProcessBench (Table 6):** ה-baseline שלהם זהה ל-job 177776 שלנו (GSM8K 81.8, ‏MATH 77.2). התאים של OlympiadBench ו-Omni-MATH נראים מוחלפים. ה-boosting שלנו על ProcessBench נכשל (PRM-7B: ‏72.94 → 67.23), כך שאין לנו שם מה להציג מולם.

### CROP (arXiv 2605.30085) ★ הכי קרוב רעיונית
- **מה עושים:** conformal certification של prefix נקי. ה-risk proxy הוא Qwen-PRM-7B + position/length/format + trace features מ-Llama-3.1-8B, עם fusion ב-supervised LR.
- **למה רלוונטי:** זה הרעיון הכי קרוב אלינו: PRM + telemetry. ההבדל הוא שאצלם ה-fusion וה-calibration עם labels, ואצלנו ה-fusion ללא labels.
- **מטריקה שונה:** pooled step AUROC ו-retained prefix ב-α=0.05 (שלנו: within-AUC).
- **PRMBench (5,000 traces):** position/length/format ‏0.775, trace features ‏0.781, ‏PRM ‏0.801, ‏trace+PRM ‏0.834. ‏retained prefix ‏4.8% → 21.8%.
- **שלנו בפרוטוקול שלהם (שחזור משוער):** LSML_CONT ‏0.802 / kept 25.1%, ‏CHAIN_N16 ‏0.794 / 23.3%, ‏CHAIN_N13 ‏0.782 / 23.0%. כל ה-fused arms ללא labels שומרים prefix גדול יותר מכל שורה שלהם, אבל ה-AUROC שלנו נמוך מ-trace+PRM.
- **הסתייגות:** ה-4.8% של Direct PRM הוא בעיקר תוצאה של ה-grid שלהם. אותם ציונים על grid עדין שומרים 17.2%.
- **ProcessBench:** הם מדווחים Direct PRM ‏0.8801 pooled AUROC. אצלנו `first_only` ‏0.883, אבל ה-CHAIN_N16 יורד ל-0.801. כלומר גם כאן אין יתרון.

### SLIDER (arXiv 2610.00571)
- **מה עושים:** Partial Information Decomposition בין הצעד לצעדים הקודמים (Step-RRI), ל-redundancy בלבד. ‏50 קריאות GPT-4o-mini לבעיה.
- **PRMBench (redundancy, ~750 בעיות):** step accuracy ‏61.91%. הפרוטוקול לא מוגדר במלואו.
- **שלנו (קריאה משוערת):** CHAIN_N16 ‏72.8–74.3, ‏9–12 נקודות מעליהם, גם עם Q80 ללא labels. ההשוואה חלשה כי המטריקה לא ברורה.
- **חשיבות:** בינונית. מחזק את הסיפור של Simplicity, אבל לא baseline מרכזי.

## עדיפות 2: הכי קרובים במהות (fusion / label-free), לא אותה משימה

### Optimal Aggregation of LLM and PRM Signals (arXiv 2510.13918, ICLR 2026 poster, קבלה לא אומתה) ★ חשוב למסגור
- **מה עושים:** fusion משוקלל של majority vote + ציון PRM + אמינות ה-LLM, ל-Best-of-N על MATH.
- **למה לא הוגן:** משימה אחרת (בחירת תשובה, לא איתור צעד), ומשקלים מ-labeled calibration set.
- **מה משותף:** PRM לבד אינו האופטימום, והמשקלים תלויים בצירוף LLM ו-PRM. הם מציינים unsupervised calibration כ-future work, וזה בדיוק המקום שלנו.
- **המספרים שלהם:** אי אפשר לחשב על PRMBench. ב-proxy שלנו (בחירת trace נקי בתוך קבוצה) ה-fused scores גרועים מה-PRM (CHAIN_N16 ‏20.5% מול PRM ‏51.6%), כי הם מנורמלים בתוך תשובה.

### FUSE (arXiv 2604.18547, Candès) ו-Weaver (arXiv 2506.18203)
- **מה עושים:** unsupervised ensembling של verifiers, ברמת התשובה בלבד.
- **למה רלוונטי:** זו הספרות הקרובה ביותר ל-"label-free fusion". ההבדל שלנו: ברמת הצעד, ועם telemetry מאותו forward pass.
- **מספרים על PRMBench/ProcessBench:** אין.

### ScalePRM (arXiv 2512.03244) ו-uPRM (arXiv 2605.10158)
- **מה עושים:** PRMs שאומנו ללא ground-truth labels. ScalePRM: הרבה אימותים לכל צעד → pseudo-labels → generative PRM. ‏uPRM: ‏LoRA ללא labels מ-P(+/−).
- **ProcessBench:** ScalePRM ‏67.5 (מעל הגרסה עם ground truth, 66.4). ‏uPRM ‏~48 (58.3/52.6/42.7/39.8). אין PRMBench.
- **למה לא הוגן:** הם מאמנים PRM; אנחנו מתחילים מ-PRM שאומן עם labels. ההשוואה טובה ל-"מה אפשר בלי labels", לא כ-baseline ישיר.

### How Far Can Unsupervised RLVR Scale (arXiv 2603.08660, ICLR 2026)
- לא baseline, אלא הסבר: אותות certainty עוזרים רק כשהם מתיישבים עם correctness. זה מתאים לכישלון שלנו ב-Socratic, ב-ProcessBench וב-Hard2Verify. reviewer שמכיר אותו ישאל את זה.

## עדיפות 3: רקע / baselines ש-reviewers יבקשו

| מאמר | למה רלוונטי | מספרים |
|---|---|---|
| TIPS (2609.36641) | generative thinking PRM; ה-SOTA הפתוח | ProcessBench ‏85.2 |
| Hidden Error Awareness (2605.09502) | probe מפוקח על hidden states, תקרה מפוקחת ל-telemetry | AUROC ‏0.95, ‏0.79 מהצעד הראשון |
| HSRM (2608.30841, EMNLP 2026) | reader קטן של hidden states עם outcome labels | ≥ text verifier ב-15/16 |
| Reward Under Attack (2603.06621) | PRMs כ-"fluency detectors"; מסביר למה הרווח שלנו ב-NR/NCL | — |
| Out of Distribution, Out of Luck (EACL 2026) | PRMs נכשלים על traces של reasoning models | — |
| QD stress tests (2608.08008) | padding exploit ב-Qwen-PRM-7B ב-mean pooling | — |
| BetaPRM (2605.15529), Distributional PRM (2605.06785) | calibration ו-reliability של PRM | — |
| PRMBench leaderboard | Gemini-2-Thinking ‏68.8, ‏o1-mini ‏68.8, ‏PRM-7B ‏65.5 | שלנו: 72B CHAIN ‏68.9, ‏72B LSML_CONT ‏69.8 |

---

## שורה תחתונה

1. **PRISM ו-CROP הם שני ה-baselines שחייבים להופיע.** PRISM כי זה אותו PRM ואותו metric. CROP כי זה אותו רעיון (PRM + telemetry) עם labels.
2. **החולשה מול PRISM:** הזזת threshold לבד מגיעה לנקודה שלהם. לכן כדאי להשוות ב-matched TPR, ולהוביל עם within-AUC.
3. **היתרון מול CROP:** label-free ושומר prefix גדול יותר. החיסרון: pooled AUROC נמוך מ-trace+PRM.
4. **ב-ProcessBench אין לנו מה להציג מול אף מאמר:** ה-boosting נכשל על 4/5 PRMs.
5. **המסגור:** Optimal Aggregation, ‏FUSE ו-Weaver מראים שהנישה של label-free step-level fusion עם PRM ריקה. ‏Unsupervised RLVR מסביר מתי היא לא עובדת.
