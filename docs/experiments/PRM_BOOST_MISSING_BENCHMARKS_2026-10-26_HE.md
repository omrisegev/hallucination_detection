# PRM boosting: איסוף הבנצ'מרקים החסרים של PRISM, ‏CROP, ‏SLIDER ו-ARES (2026-10-26)

**בקשה:** Omri אישר את התוכנית ואת ריצות ה-GPU על AIRCC. לכל בנצ'מרק חסר אוספים:
1. טלמטריה לכל צעד מה-backbone של כל PRM יחד עם ציוני ה-PRM (בשביל fusion בסגנון CHAIN), דרך `cluster/run_prmbench_prm_telemetry.py --mode prm --answers`.
2. טלמטריה ממודל הבסיס העצמאי של כל PRM (בשביל L-SML על פיצ'רים של הסתברויות טוקנים בלבד), דרך `--mode instruct` עם אותם מודלים ו-prompt styles כמו ב-`prm_standalone_telemetry_v1`.

ענף: `claude/prm-boost-missing-benchmarks-2026-10-26`, נחתך מ-`claude/prm-boosting` @ 78e40fb. לא ממזגים.

## 1. מה כל מאמר מריץ, ומה כבר יש לנו

מקורות: `docs/research_notes/competitor_registry_2026-10-21/REGISTRY.md`, ‏`papers/digests/crop-conformal-clean-prefix.md`, ‏הקוד של CROP (github.com/matthewyccheung/crop, `crop/scripts/import_external_process.py` ו-`crop/cheap_baselines.py`, נקראו היום) ו-ARES (arXiv 2507.12948v2, HTML).

| בנצ'מרק | מי מריץ | step labels? | ציוני PRM + טלמטריית backbone (5 PRMs) | טלמטריית מודל בסיס עצמאי (5 PRMs) | פעולה |
|---|---|---|---|---|---|
| PRMBench | PRISM, ‏CROP, ‏SLIDER, ‏ARES | כן | יש | יש (`prm_standalone_telemetry_v1`) | parity בלבד |
| ProcessBench | PRISM, ‏CROP | כן (שגיאה ראשונה) | יש (job 289297) | **חסר** | standalone בלבד |
| Socratic-PRMBench | (שלנו) | כן | יש | **חסר** (יש רק Qwen3) | standalone בלבד |
| Hard2Verify | (שלנו) | כן | יש | **חסר** (יש רק Qwen3) | standalone בלבד |
| CRV Arithmetic | CROP (ראשי) | כן | **חסר** | **חסר** | מלא |
| CRV GSM8K | CROP (ראשי) | כן | **חסר** | **חסר** | מלא |
| Math-Shepherd (10,000) | CROP | כן (אוטומטי, MC) | **חסר** | **חסר** | מלא |
| PRM800K (8,000) | CROP; ‏PRISM מתאמן עליו | כן (אנושי) | **חסר** | **חסר** | מלא |
| DeltaBench | ARES | כן, ברמת section | **חסר** | **חסר** | החלטה (ראו 4) |
| ClaimTrees, ‏CaptainCookRecipes | ARES | כן, סינתטי | — | — | לא מתוכנן: לא מתמטיקה, נבנים בקוד של ARES |
| GSM8K, ‏AIME2025, synthetic arithmetic | SLIDER | לא (מדד RRI מול אורך) | — | — | לא רלוונטי: אין תוויות שגיאה |
| BoN / beam / GRPO (MATH-500, ‏AIME24, ‏LiveCodeBench וכו') | PRISM | — | — | — | לא בתחום: דורש generation או אימון |

## 2. הגדרות הנתונים (משוחזרות מהקוד של CROP)

| בנצ'מרק | מקור | כלל בחירה | CROP Table 4 | שלנו (נבדק מקומית) |
|---|---|---|---|---|
| CRV Arithmetic | `facebook/crv`, ‏`arith.nt{3,5,7,10}.annotated.json` | ‏375 לכל קובץ ואז 1,500, ‏`np.random.default_rng(2806)`, בסדר glob ממוין | ‏1,500 / 10,316 / 3.4% | ייבדק בבנאי |
| CRV GSM8K | `facebook/crv`, ‏`gsm8k.annotated.json` | כל השורות | ‏1,319 / 8,995 / 7.3% | **1,319 / 8,995 / 7.3%** |
| Math-Shepherd | `peiyi9979/Math-Shepherd` train, streaming | ‏10,000 השורות הראשונות; חיתוך `Step k:`; שגיאה = `*` או ` - ` במקטע התווית | ‏10,000 / 37,514 / 47.6% | ייבדק על AIRCC |
| PRM800K | `openai/prm800k` `phase2_test.jsonl` ואז `phase2_train.jsonl` | ‏8,000 הראשונות; ה-completion הנבחר, rating ≥ 0 = תקין | ‏8,000 / 70,698 / 4.2% | ייבדק על AIRCC |
| DeltaBench | `OpenStellarTeam/DeltaBench` `Deltabench_v1.csv` | section = צעד; שגיאה = `reason_error_section_numbers` | ‏ARES לא מציין גודל | ‏1,236 (math 562 / 11,185 sections / 6.9%) |

את הקוד של CROP לא מריצים: הבנאי שלנו מממש מחדש את אותו כלל, ובודק את הספירות מול Table 4.

## 3. תוכנית ריצה

- **שלב A (CPU, מקומי/login node):** בנאי קלט `results/prm_boost_crop_ares_benchmarks_v1/make_inputs.py`. כותב `answers.json` ללא תוויות, ו-gold בתיקייה נפרדת. בודק את הספירות מול Table 4 ועוצר אם יש פער.
- **שלב B (CPU, AIRCC):** length census לכל tokenizer. תשובה שחורגת מה-context נרשמת ככישלון ולא נחתכת.
- **שלב C (GPU, timing):** ‏12 שורות לכל (PRM × בנצ'מרק × mode), שעת wall אחת. אם הכול נקי, ממשיכים ישר לריצה המלאה.
- **שלב D (GPU, מלא):** שרשראות ליניאריות.
  - mode prm: ‏5 PRMs × ‏{CRV-A, ‏CRV-G, ‏Math-Shepherd, ‏PRM800K}.
  - mode instruct (standalone): ‏5 מודלי בסיס × אותם 4 בנצ'מרקים, ועוד ProcessBench, ‏Socratic ו-Hard2Verify.
- **שלב E (CPU, AIRCC):** bank לכל תא (`prm_boost_build_bank_external.py`, ‏mask1, ‏digit ids מה-tokenizer של המודל), ואחריו scoring ללא תוויות (`ablation5_socratic.py`).
- **parity:** ריצת prm-mode ו-standalone על 40 השורות הראשונות של PRMBench (`--meta-pkl`) מול ה-pkl הקפואים. נדרש פער 0 ב-rewards ובטלמטריה.
- **העלאה:** rclone ל-`gdrive:hallucination_detection/cluster_results/prm_boost_crop_ares_benchmarks_v1/`.

**הערכת עלות (לפני timing):**
- mode prm לא ממוקבץ: 0.6–0.9 s/row ל-7B, ‏6–7 s/row ל-72B. ‏~20,800 שורות, ולכן ~4–5 GPU-h לכל 7B ו-~38 GPU-h ל-72B.
- standalone ממוקבץ: PRMBench לקח 6–14 דקות למודל, ולכן כמה GPU-h בסך הכול.

## 4. הסתייגויות שצריך להחליט עליהן או לסמן

- **DeltaBench ארוך:** החציון ב-math הוא ~16k תווים, ורק ~26% מהתשובות נכנסות ב-4,096 הטוקנים של משפחת Qwen2.5-Math (4 מתוך 5 ה-PRMs). Llama-3.1 ו-Mistral קוראים הכול. ההמלצה: להריץ את כולם ולדווח כיסוי ו-common set, בלי חיתוך. ההשוואה מול ARES ממילא משוערת, כי הם לא מציינים subset.
- **זיהום אימון:**
  - Math-Shepherd train הוא נתוני האימון של Math-Shepherd-PRM, ‏RLHFlow מאומן על נתונים מאותו סוג, ו-PURE-PRM מאומן על PRM800K.
  - השורות האלה יסומנו "in-training" ולא ייחשבו העברה.
- **CRV GSM8K נוצר על ידי Llama-3.1-8B-Instruct:** הטלמטריה העצמאית של RLHFlow (שהבסיס שלו הוא אותו מודל) היא טלמטריה של המחולל עצמו. אצל השאר זה teacher forcing על טקסט זר.
- כל הנתונים הם development data.
