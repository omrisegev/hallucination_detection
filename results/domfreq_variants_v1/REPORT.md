# חיפוש תחליף רציף ל-dominant_freq16: domfreq_variants_v1

**שורה תחתונה:** נמצאו variants רציפים שאינם מבוססי entropy, שה-sign שלהם יציב בכל 8 ה-cells: within-AUC מינימלי .55-.56 ו-R^2 של .05-.23 מול 16 הערוצים. אבל הם מוסיפים רק +0.1 עד +1.0pp ל-z-average השווה, ורובם מתואמים חלקית עם מיקום הצעד. כל הבחירה כאן היא development-level על נתונים שכבר נראו.

## נתונים ו-Step 0 (נמדד)
- PRMBench: `prm_boost_v1/full_prm/prmbench_telemetry.pkl` + `bank_prm/FEATURES.npz` (Qwen-7B), ו-`prm_boost_gen_v1/full_<prm>/` + `bank_<prm>` (72B/PURE/RLHFlow). התוויות וה-raw17 נלקחו מ-`auto_raw_plus_z_v1/bundle/inputs/INPUTS_<prm>.npz`.
- Socratic: `prm_boost_external_v1/full_*_socratic/external_telemetry.pkl` + `bank_v2/FEATURES.npz`. התוויות הן evaluator-only `socratic.json`, שהועלה מהמחשב המקומי ל-`domfreq_variants_v1/labels/`.
- Mask: mask1 (`code_prm_boost_v1/mask_fixed`). ה-token matrix נבנה עם הקוד הקפוא (`code_thinkfix_bank_553de79`, _bank11).
- **Alignment:** התאמה של 100% מהתשובות, max abs error 0.0, בכל 8 ה-cells (94,188/94,098 צעדים ב-PRMBench, 26,052 ב-Socratic).
- **ממצא:** הצבירה השמורה היא **top10** (ממוצע 10 ערכי הסיכון הגבוהים בצעד), ולא ממוצע פשוט. ממוצע פשוט מתאים רק ב-20-22% מהצעדים ב-PRMBench וב-9% ב-Socratic.
- ה-modal share של העמודה השמורה הוא .47-.57 ב-PRMBench ו-**.78-.85** ב-Socratic, עם 164-224 ערכים שונים. within-AUC של הערוץ השמור: .589/.590/.567/.578 ב-PRMBench ו-.512/.494/.506/.508 ב-Socratic. זה משחזר את Steps 504-505.

## Variants (718)
- 6 source series: surprisal, margin, logsumexp energy, top15 turnover, top50 JS, ו-entropy כ-REF בלבד.
- Statistics: slope, low10, low20, spectral entropy, flatness, centroid; lag1, lag2; DFA.
- Cross-series: MSC/phase low/high, לזוגות surprisal~energy ו-margin~turnover.
- Windows: 8/16/32 causal, כל token (stride 1), וגם step-aligned. בנוסף mc/lin ו-mean/last.
- **Pruned:** power-weighted mean frequency זהה ל-centroid לפי ההגדרה, אז נשמר פעם אחת. low10 ב-W=8 הושמט כי אין bin <= 0.1. DFA ו-coherence רק ב-W>=16.
- **Short steps:** ב-rolling windows, tokens לפני החלון המלא הראשון משתמשים בחלון הראשון (backfill, כמו `_causal`), ותשובה קצרה מ-W משתמשת בכולה. ב-step-aligned: צעד קצר מ-4 tokens (DFA ו-coherence: מ-16) מקבל את ממוצע העמודה בתשובה, לפי כלל fill_missing של ה-bank. שיעור ה-short/backfill נמצא בטבלה.
- **Filter (הוגדר מראש):** sign זהה בכל 4 ה-PRMs, Socratic באותו צד כמו PRMBench, ו-modal share < 5% בכל cell. 322/718 עברו. הדירוג לפי within-AUC מינימלי על 8 ה-cells, best config לכל (series, stat).

## Top variants
הערכים לכל cell בסדר 7B/72B/PURE/RLHFlow, קודם PRMBench | אחר כך Socratic. ה-sign נקבע לפי PRMBench (בתוך המדגם); ב-Socratic משתמשים ב-sign של אותו PRM מ-PRMBench.

| variant (sign) | within-AUC | max\|rho\| entropy-family | R^2 16 / ent | resid within | z-avg gain pp | rho step idx | short |
|---|---|---|---|---|---|---|---|
| js·spectral-entropy·w32·mc·last (−) | .588 .607 .566 .571 \| .657 .601 .609 .559 | .29 .23 .28 .27 \| .37 .22 .27 .19 | .10-.23 / .06-.16 | .510-.541 | +.16 +.35 −.04 +.19 \| +.92 +.61 +.38 +.34 | −.13 −.11 −.01 +.02 \| −.34 −.17 −.18 −.04 | 7% \| 0.3% |
| js·flatness·w16·mc·last (−) | .584 .574 .556 .565 \| .623 .578 .584 .555 | .21 .15 .21 .21 \| .27 .14 .18 .14 | .05-.13 / .03-.08 | .511-.535 | +.17 +.10 +.05 +.15 \| +.55 +.44 +.21 +.44 | −.12 −.07 −.02 −.01 \| −.28 −.15 −.16 −.04 | 2% \| 0% |
| surprisal·spectral-entropy·w8·mc·last (−) | .599 .562 .555 .550 \| .663 .602 .608 .551 | .31 .19 .32 .22 \| .38 .18 .35 .29 | .09-.22 / .06-.19 | .504-.562 | +.22 +.20 +.07 +.08 \| +1.03 +.79 +.34 +.21 | −.10 −.01 +.03 +.02 \| −.25 −.17 −.08 +.06 | 0.2% \| 0% |
| margin·slope·step·lin (−) | .558 .569 .566 .551 \| .581 .561 .574 .537 | .29-.38 | .12-.25 / .09-.19 | .515-.543 | −.13 −.21 −.05 +.02 \| −.08 +.17 +.09 .00 | ~0 (\|.06\|) | 1% \| 3-4% |
| margin·low10·step·lin (+) | .594 .608 .598 .587 \| .592 .570 .578 .551 | .42-.48 | .30-.49 / .22-.35 | .511-.548 | −.34 −.40 −.18 −.22 \| −.01 +.05 +.02 +.07 | −.01 to −.12 | 1% \| 3-4% |
| surprisal·flatness·step·mc (−) | .642 .623 .551 .687 \| .641 .629 .584 .578 | .31-**.62** | .14-.49 / .12-.45 | .477-.559 | −.10 −.14 −.26 +.58 \| +.37 +.34 −.17 +.44 | −.06 to +.21 | 1% \| 3-4% |
| REF entropy·centroid·step·lin (−) | .620 .633 .587 .592 \| .641 .597 .617 .589 | .41-.50 | .35-.40 / .24-.36 | .502-.558 | −.20 −.33 −.17 −.22 \| +.27 +.06 +.27 +.54 | ~0 | 1% \| 3-4% |

- Degeneracy: לכל השורות modal share <= .003 (חוץ מ-surprisal·w8 ב-Socratic/PURE: .032), ובין 21k ל-73k ערכים שונים.
- Top-3 Spearman: לשלושת ה-variants של JS ו-surprisal: true_tail50, top50_js, q15_H1, כולם |rho| ≈ .2-.4. ל-variants ה-step-aligned של margin: dominant_freq16, top50_js, top15_turnover.

## מה לא redundant עם entropy
- **JS / surprisal spectral-entropy/flatness בחלון rolling קצר (last token):** max|rho| מול entropy-family .14-.38, R^2_ent .03-.19, R^2 מול כל ה-16 .05-.23. ה-residual within-AUC נשאר מעל .5 בכל 8 ה-cells (.50-.56). זה המידע הלא-redundant היחיד שנמצא, והוא קטן.
- **Step-aligned margin/energy:** redundancy נמוכה עד בינונית, אבל **מתואמים חזק עם אורך הצעד** (margin·low10: rho .60-.67 ב-PRMBench), ואורך הצעד לבדו נותן within-AUC של .62. זה בעיקר proxy לאורך. ה-gain שלהם שלילי ב-PRMBench.
- **Entropy copies:** surprisal·flatness·step (rho .62 מול q15_H1 ב-Socratic, R^2 עד .49) ו-entropy·centroid·step (REF). הם חזקים, אבל זה עוד ערוץ entropy.

## המלצות לבדיקה בתוך ה-pipeline
1. **js·spectral-entropy·w32·mc·last:** ה-within-AUC המינימלי הגבוה ביותר מבין ה-non-entropy (.559), רציף, ו-gain חיובי ב-7/8 cells. **אזהרה:** rho עם מיקום עד −.34 ב-Socratic/7B. ה-risk עולה עם המיקום, וה-step index לבדו נותן .66/.73.
2. **js·flatness·w16·mc·last:** הכי פחות redundant (R^2 .05-.13), ו-gain חיובי ב-**8/8** cells (+.05 עד +.55pp). מתאם עם מיקום חלש יותר.
3. **surprisal·spectral-entropy·w8·mc·last:** gain חיובי ב-8/8 (+.07 עד +1.03pp) ו-R^2 .09-.22. ה-modal share בגבול (.032 ב-Socratic/PURE).
- הימנעו מ-margin/energy step-aligned (proxy לאורך) ומ-surprisal·flatness·step (העתק entropy).
- **Inferred, לא נמדד:** ההשפעה על PRMScore ועל ה-L-SML grouping. זה מצריך הרצה בתוך ה-pipeline.

## קבצים ופקודות (AIRCC, `results/domfreq_variants_v1/`)
- `code/compute_variants.py`, `code/run_compute.sbatch` (job 310406, 22 דקות): `sbatch -A cycle4_tau_averbuch_spot_prj -p cpu code/run_compute.sbatch`. הפלט: `cache/CACHE_*.npz` (גדולים, מחוץ ל-git).
- `code/metrics.py`, `code/run_metrics.sbatch` (job 311719): ALIGNMENT.json, RESULTS.json, CANDIDATES.json, TOP_TABLE.json.
- `code/plots.py` רץ מקומית (אין matplotlib בסביבת ה-CPU) ויצר `auc_vs_redundancy.png` ו-`top6_histograms.png`. הלוגים ב-`logs/`.
- Account: CPU עובד רק עם `cycle4_tau_averbuch_spot_prj`; cycle3 נדחה. לא שונו קבצים קיימים ולא נעשה commit.
