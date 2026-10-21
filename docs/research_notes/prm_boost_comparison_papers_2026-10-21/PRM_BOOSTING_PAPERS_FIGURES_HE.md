# מאמרי ההשוואה: benchmarks, ‏metrics, גרפים וטבלאות, ומה אפשר לשחזר אצלנו (2026-10-21)

המשך של `PRM_BOOSTING_COMPARISON_PAPERS_HE.md`.
- **המקור לכל מאמר:** קראתי את ה-PDF או את ה-HTML ב-arXiv היום (קריאה אוטומטית, לא קריאה מלאה ידנית). מספרים שלא הופיעו בטקסט מסומנים כך.
- **"אפשר לייצר?"** מתייחס לנתונים שכבר יש לנו, בלי inference חדש:
  - **PRMBench:** ‏5 PRMs, ציון PRM ו-telemetry לכל צעד, labels וקטגוריות.
  - **ProcessBench:** ‏5 PRMs, ‏3,400 תשובות, job 289297. הנתונים ב-Drive.
  - **Socratic, ‏Hard2Verify.**
  - **אין לנו:** דגימות מרובות לכל שאלה (Best-of-N), אימון RL, ‏PRM800K, ‏Math-Shepherd dataset ו-CRV Arithmetic/GSM8K.

**מקרא:** ✅ אפשר עכשיו · ✅* כבר עשינו (`results/prmbench_paper_metric_parity_v1/` או ה-talking points) · 🟡 חלקית או אנלוגית · ❌ דורש generation, אימון או dataset שאין לנו

---

## 1. PRISM (arXiv 2606.09078)

**Benchmarks:**
- **הערכת PRM:** PRMBench ו-ProcessBench (GSM8K, ‏MATH500, ‏OmniMath, ‏OlympiadBench).
- **שימוש downstream:**
  - Best-of-N ו-guided beam search: MATH-500 (100 בעיות), ‏AIME24, ‏LiveCodeBench.
  - GRPO: AIME, ‏AMC, ‏GSM8K, ‏MATH-500, ‏Minerva, ‏OlympiadBench.
- **אימון:** PRM800K.

**Metrics:**
- **PRMBench:** FPR, ‏FNR, ‏Precision, ‏TPR, ‏TNR, ‏Pos-Acc, ‏Neg-Acc, ‏PRMScore, ‏FPR reduction %.
- **ProcessBench:** TPR/TNR/F1 לכל subset.
- **downstream:** accuracy.

**PRMs:** Qwen2.5-Math-PRM-7B (הראשי), ‏ReasonEval-7B, ‏ThinkPRM-7B, ‏GenPRM-7B, ‏GPT-4o.

| פריט | מה מוצג | אפשר לייצר? |
|---|---|---|
| Table 1 | Pos-Acc / Neg-Acc / PRMScore ל-PRMs שונים (Qwen-7B ‏95.36/30.66/65.5, ‏ThinkPRM 83.3/50.9/64.3, ‏GPT-4o 82.9/58.2/66.8) | ✅ אפשר להוסיף שורות לכל 5 ה-PRMs שלנו ולכל fused arm |
| Table 2 | ablation: FPR/FNR/Prec/TPR/TNR/PRMScore/FPR-reduction | ✅* ל-PRM-7B (‏`PRISM_RESULTS.json`); ✅ להרחיב ל-5 PRMs |
| Table 4 | מספר ה-FPs (9,109 ב-baseline) | ✅ טריוויאלי |
| Table 6 | ProcessBench: TPR/TNR/F1 לכל subset | ✅ עכשיו אפשר, כי ProcessBench הורץ. ה-baseline משוחזר עד 0.1. **אבל ה-fused arms מפסידים** (PRM-7B ‏72.94 → 67.23 avg F1) |
| Table 7 | PRMBench לפי קטגוריה ו-Simplicity/Soundness/Sensitivity | ✅* (טבלת S/S/S תחת שקופית 7 ב-talking points) |
| Tables 5, 8–10 | curriculum bins | ❌ (אימון) |
| Fig 2 | label skew ב-PRM800K | 🟡 אפשר לעשות את אותו גרף על PRMBench, אבל אין בזה ערך להשוואה |
| Fig 3 | סימולציה של BoN עם FP/FN | ❌ (תאורטי, דורש BoN) |
| Fig 4 | pairwise accuracy לפי score margin | ✅ גרסה within-answer (pairs של צעד שגוי ותקין באותה תשובה) לפי margin. זה בעצם פירוק של within-AUC |
| Fig 5a | FPR ו-precision לאורך curriculum | ❌ |
| **Fig 5b** | **trade-off בין FPR ל-FNR לאורך thresholds, ‏PRISM מול baseline** | ✅ **הכי חשוב לייצר.** זה הגרף שמראה ש-threshold shift לבד מגיע לנקודה שלהם. שתי עקומות: PRM ו-CHAIN, עם הנקודה של PRISM |
| Fig 6, 8, 9; Table 3 | BoN, ‏beam search, ‏GRPO | ❌ |
| Fig 7 | התפלגות ה-reward על צעדים שגויים | ✅ histogram של PRM מול fused score על error steps |

## 2. CROP (arXiv 2605.30085)

**Benchmarks:**
- **ראשי:** CRV, עם Arithmetic (1,500 traces) ו-GSM8K (1,319).
- **נוספים (Table 4):**
  - ProcessBench: ‏3,400 / 25,697 צעדים / 6.6% שגיאות.
  - Math-Shepherd: ‏10,000 traces.
  - PRMBench: ‏5,000 / 67,445 / 15.9%.
  - PRM800K: ‏8,000 traces.

**Metrics:**
- pooled step AUROC.
- retained-prefix % ו-prefix-contamination risk, ב-α=0.05.
- boundary deviation: over-withholding ו-unsafe overshoot.
- repair accuracy.
- risk ו-retention לפי קבוצות.

**מודלים:** Qwen2.5-Math-PRM-7B קפוא, ‏trace features מ-Llama-3.1-8B-Instruct, ו-fusion ב-LR.

**ה-proxies:** Random, ‏Token/format, ‏Trace features, ‏Direct PRM, ‏Trace+PRM.

| פריט | מה מוצג | אפשר לייצר? |
|---|---|---|
| **Table 6** | AUROC / kept% / risk% לכל proxy × dataset. **PRMBench:** ‏Token/format 0.775/16.2, ‏Trace 0.781/15.9, ‏PRM 0.801/4.8, ‏Trace+PRM 0.834/21.8. **ProcessBench:** ‏Token/format 0.706/10.6, ‏Trace 0.706/11.3, ‏PRM 0.880/46.7, ‏Trace+PRM 0.894/42.4 | ✅* PRMBench (‏`CROP_RESULTS.json`). ✅ **ProcessBench אפשר עכשיו** (הציונים קיימים), אבל קיים פער: target של 6.6% אצלם מול 8.6% אצלנו. צפוי שנפסיד: CHAIN first_only ‏0.801 מול PRM ‏0.883. ‏Math-Shepherd, ‏PRM800K ו-CRV: ❌ |
| Fig 1 | CDF של מיקום השגיאה הראשונה | ✅ (labels בלבד) |
| Fig 3 | kept prefix מול α | ✅ (‏`crop_metrics.py` עם sweep על α) |
| **Fig 4** | scatter של AUROC מול kept%, נקודה לכל proxy × dataset | ✅ **שימושי.** אצלנו בדיוק "AUROC נמוך יותר, prefix ארוך יותר", כך שהגרף מראה את זה ישירות |
| Fig 5 | risk/retention לפי אורך trace | ✅ |
| Fig 6 | risk לפי רמת קושי (Arithmetic) | 🟡 ב-PRMBench אין רמת קושי. ב-ProcessBench אפשר לחלק לפי subset |
| Fig 7 | risk/retention לכל dataset | 🟡 PRMBench + ProcessBench בלבד |
| Fig 8 | contamination לפי מיקום השגיאה הראשונה (early/mid/late) | ✅ |
| Table 1, 7, 8 | CROP מול whole-trace abstention, ‏boundary deviation, רגישות ל-α | ✅ (חישוב מהציונים בלבד) |
| Table 2, 3, 10 | repair accuracy (LLM ממשיך מה-prefix) | ❌ (generation) |
| Table 16 | artifact-matched, ‏label-shuffle ו-order controls | 🟡 עשינו אנלוגים: null/permutation ו-residualization על position/length |
| Table 17 | cross-domain: calibrate על dataset אחד ובודקים על אחר | ✅ calibrate על PRMBench, ‏test על ProcessBench או Socratic |

## 3. SLIDER (arXiv 2610.00571)

**Benchmarks:**
- PRMBench, רק מחלקת redundancy (~750 בעיות).
- GSM8K, ‏AIME2025, ‏synthetic arithmetic.

**Metrics:**
- Step-RRI, ‏Trajectory-RRI ורכיבי PID.
- redundancy detection accuracy.
- Spearman ρ מול אורך ה-reasoning.

**מודלים:** QwQ-32B, ‏DeepSeek-R1-Distill-Qwen-32B, ‏GPT-4.1. בנוסף GPT-4o-mini ליצירת variants, ו-MiniLM ל-embeddings.

| פריט | מה מוצג | אפשר לייצר? |
|---|---|---|
| **Fig 7** | accuracy של redundancy detection: Step-RRI ‏61.91, ‏cosine ~51.6, ‏InfoGain ~48.4 | ✅* (קריאה משוערת: CHAIN_N16 ‏72.8–74.3). ההגדרה המדויקת של accuracy לא ידועה |
| Table 1, Fig 8 | Trajectory-RRI מול אורך | ❌ (תלוי ב-PID שלהם) |
| Fig 5, 6, 10, 11; Table 2 | פירוקי PID לבעיות בודדות | ❌ (לא רלוונטי לנו) |
| Fig 9, Table 3 | fine-tuning על נתונים עם RRI שונה | ❌ |

**שורה תחתונה:** רק פריט אחד בר השוואה. מאמר משני.

## 4. Optimal Aggregation of LLM and PRM Signals (arXiv 2510.13918)

**Benchmarks:**
- MATH: ‏32 דגימות לשאלה, ‏1k ל-calibration.
- MATH500: ‏112 דגימות לשאלה, ‏100 ל-calibration.
- PRMBench ו-ProcessBench **לא בשימוש**. הם מוזכרים רק ב-related work.

**Metrics:**
- accuracy של התשובה שנבחרה, ‏Pass@N ו-oracle "Optimal".
- compute efficiency.
- MAE של אמינות ה-LLM.

**מודלים:** ‏5 LLMs (Mistral-7B, ‏Qwen2.5-1.5B/7B, ‏DeepSeek-1.5B/7B) × ‏7 PRMs (כולל Qwen-PRM-7B ו-Skywork-1.5B).

| פריט | מה מוצג | אפשר לייצר? |
|---|---|---|
| Table 1, 2 | accuracy לכל שיטת aggregation (BoN, ‏MV, ‏Vanilla/KDE/Linear/Logit WV) × LLM × PRM | ❌ (דורש N דגימות עם תשובות מדורגות) |
| Fig 1 | התפלגות ציוני PRM + המשקלים האופטימליים w*(p) לכל זוג LLM-PRM | 🟡 התפלגויות כן, המשקלים לא (הם מוגדרים לבחירת תשובה) |
| Fig 2 | accuracy מול N, ו-compute savings | ❌ |
| Fig 3 | grid של offset b, ו-accuracy מול גודל ה-calibration set | ❌ |
| Fig 4 | פונקציית משקל לכל שאלה, ו-MAE של אמינות ה-LLM | ❌ |

**הערה:** ב-v1 ב-arXiv, הממוצע של Logit WV עם Qwen-PRM-7B על MATH הוא 62.1. ב-README של ה-parity שלנו כתוב 63.3, כנראה מגרסה אחרת של המאמר. צריך לוודא את הגרסה לפני שמצטטים.

**שורה תחתונה:** אין אף גרף שאפשר לשחזר בלי generation חדש. רלוונטי למסגור בלבד.

## 5. מאמרים label-free על ProcessBench

### uPRM (arXiv 2605.10158)
- **Benchmarks:** ProcessBench, ‏MATH-500, ‏Minerva ו-OlympiadBench (TTS ו-RL).
- **Metrics:** Err. / Corr. / F1.
- **מודלים:** Qwen2.5-14B-Instruct.
- **Table 1 / D1** (ProcessBench F1, ‏LLM-judge → uPRM): GSM8K ‏49.8→58.3, ‏MATH ‏42.8→52.6, ‏Olympiad ‏29.4→42.7, ‏Omni ‏26.6→39.8. ‏**✅ אפשר להוסיף שורות שלנו.** עם זאת, PRM-7B לבד כבר ב-72.9, כי הוא אומן עם labels, כך שההשוואה לא הוגנת.
- **Fig 1, Table 2/3, Fig D:** ‏TTS, ‏RL ו-reward hacking. ❌

### ScalePRM (arXiv 2512.03244)
- **Benchmarks:** ProcessBench, ו-6 math benchmarks ל-RL.
- **Metrics:** F1 ו-pass@1.
- **מודלים:** generator Qwen2.5-14B, ‏verifier Qwen3-32B.
- **Fig 2** (avg F1 לפי מקור ה-labels; step-level consistency הוא 67.5): ✅ אפשר להציב את המספרים שלנו לידם.
- **Table 1 (RL):** ❌

### FUSE (arXiv 2604.18547) ו-Weaver
- **Benchmarks:** GPQA Diamond, ‏HLE, ‏IMO Shortlist.
- **רמה:** תשובה, לא צעד.
- **אפשר לשחזר?** ❌ אין טבלה שאפשר לשחזר. רלוונטיים כ-related work בלבד.

---

## מה הייתי מייצר קודם (כולו CPU, מנתונים קיימים)

1. **PRISM Fig 5b:** עקומת FPR–FNR של PRM ו-CHAIN, עם נקודת PRISM. הגרף עונה ישירות על השאלה "האם זה רק threshold".
2. **CROP Table 6, שורות ProcessBench:** כדי שתהיה טבלה מלאה של 2 benchmarks. התוצאה צפויה להיות שלילית, אבל זה עדיף מ-GAP.
3. **CROP Fig 4:** ‏scatter של AUROC מול kept%. מראה את ה-trade-off של השיטה שלנו.
4. **PRISM Table 1/2 לכל 5 ה-PRMs.**
5. **CROP Table 17:** calibrate על PRMBench, ‏test על ProcessBench.

כל אלה הם ניסויים חדשים (CPU בלבד). לא הרצתי אותם.
