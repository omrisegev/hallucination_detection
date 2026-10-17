# הכללה לבנצ׳מרקים החיצוניים: ארבע שיטות יום שני

**סטטוס:** בדיקת העברה חקרנית. התוויות של Hard2Verify ו־Socratic נפתחו בסבבים קודמים, לכן זו אינה אישור בלתי־תלוי; MedPRMBench לא נגעו בו. כל השיטות והבקרות מותאמות מחדש על כל תא חיצוני, בלי תוויות (אותו פרוטוקול כמו בפיתוח). B16 הוכרז מראש, מנתוני הפיתוח בלבד, לפני שנפתחה תוצאה חיצונית כלשהי של השיטות האלה.

**אימות:** המתאם על נתוני המקור משחזר את ציוני הפיתוח הקפואים עד 3e-15 בכל 8 הבנקים, וגם את דגלי ANSWER_Z ו־R2 בלי אי־התאמה (`SOURCE_PARITY_GATE.json`). התחזיות נחתמו (hash) לפני פתיחת התוויות (`*/SEAL.json`); המעריכים הרשמיים שוחזרו בדיוק על הבנק המודגש והייחוסים (`EVALUATION_PROVENANCE.json`). שער האנטרופיה לא הורץ: חסר מאפיין אנטרופיה ברמת תשובה לתאים החיצוניים (מגבלה מוצהרת).

## האם בחירת הפיתוח עוברת?

![](figures/E1_dev_vs_external_light.png)

| תא | תצורות | Spearman פיתוח→חיצוני (AUC) | Spearman פיתוח→מדד רשמי | כיוון הרווח מול DS_FILTER_EQ תואם | דירוג תצורות B16 (מתוך 72) |
|---|---:|---:|---:|---:|---:|
| Hard2Verify · Qwen3-8B | 72 | 0.32 | 0.19 | 21/40 | 6–42 |
| Socratic · Qwen3-8B | 72 | 0.67 | 0.62 | 25/40 | 1–12 |
| Socratic · QwQ-32B | 72 | 0.51 | 0.53 | 28/40 | 2–34 |

- ב־Socratic הדירוג מהפיתוח מנבא חלקית את הדירוג החיצוני, ו־B16 שהוכרז מראש נמצא בראש הטבלה (המיקום הטוב ביותר בטווח). ב־Hard2Verify (200 תשובות, 79 קבוצות) הקשר חלש.
- כיוון הרווח של השקלול מול הממוצע הפשוט תואם בערך במחצית עד שני שלישים מהמקרים — כלומר הרווח הקטן בפיתוח אינו עובר באופן אמין.
- **קו הבסיס של מיקום הצעד גבוה מכל השיטות ב־AUC בכל שלושת התאים** (ב־Socratic כשליש מהתשובות הן קידומת תקינה ואחריה שגיאות עד הסוף). זו אזהרה ידועה מסבבים קודמים.

## המדד הרשמי מול הייחוסים

![](figures/E2_official_metric_light.png)

| מדרג | Hard2Verify · Qwen3-8B | Socratic · Qwen3-8B | Socratic · QwQ-32B |
|---|---:|---:|---:|
| HEM_MOM_cc (B16, ANSWER_Z_Q80) | 0.398 | 0.630 | 0.645 |
| MEAN_MOM_cc (B16, ANSWER_Z_Q80) | 0.393 | 0.631 | 0.645 |
| MEAN_MOM_ga (B16, ANSWER_Z_Q80) | 0.404 | 0.633 | 0.643 |
| TCI_ONLY_ga (B16, ANSWER_Z_Q80) | 0.391 | 0.623 | 0.628 |
| DS_FILTER_EQ (B16, ANSWER_Z_Q80) | 0.401 | 0.629 | 0.636 |
| ALL_EQ (B16, ANSWER_Z_Q80) | 0.401 | 0.627 | 0.636 |
| בנק11 L-SML קפוא (סבב קודם) | 0.437 | 0.632 | 0.642 |
| CT7 (סבב קודם) | 0.378 | 0.588 | 0.602 |
| מיקום הצעד בלבד (ANSWER_Z_Q80) | 0.483 | 0.580 | 0.580 |
| PRM-7B (מאמר) | 0.424 | 0.680 | 0.680 |
| מבקר LLM (מאמר) | 0.535 | 0.738 | 0.738 |

**within-answer AUC (B16):**

| מדרג | Hard2Verify · Qwen3-8B | Socratic · Qwen3-8B | Socratic · QwQ-32B |
|---|---:|---:|---:|
| HEM_MOM_cc | 0.617 | 0.685 | 0.714 |
| MEAN_MOM_cc | 0.606 | 0.682 | 0.710 |
| MEAN_MOM_ga | 0.589 | 0.678 | 0.701 |
| TCI_ONLY_ga | 0.565 | 0.671 | 0.674 |
| DS_FILTER_EQ | 0.598 | 0.676 | 0.682 |
| מיקום הצעד בלבד | 0.865 | 0.731 | 0.731 |

## ניגודים מזווגים (B16, תיאוריים)

| תא | שיטה | מול | מדד | הפרש [95%] |
|---|---|---|---|---:|
| Hard2Verify · Qwen3-8B | HEM_MOM_cc | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | -0.003 [-0.023, +0.018] |
| Hard2Verify · Qwen3-8B | HEM_MOM_cc | frozen_lsml | official (ANSWER_Z_Q80) | -0.039 [-0.069, -0.008] |
| Hard2Verify · Qwen3-8B | HEM_MOM_cc | ct7 | official (ANSWER_Z_Q80) | +0.020 [-0.009, +0.049] |
| Hard2Verify · Qwen3-8B | HEM_MOM_cc | B16__DS_FILTER_EQ | within_auc | +0.018 [+0.007, +0.031] |
| Hard2Verify · Qwen3-8B | HEM_MOM_cc | STEP_INDEX | within_auc | -0.248 [-0.297, -0.198] |
| Hard2Verify · Qwen3-8B | HEM_MOM_cc | frozen_lsml | within_auc | -0.018 [-0.038, +0.004] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_cc | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | -0.008 [-0.029, +0.012] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_cc | frozen_lsml | official (ANSWER_Z_Q80) | -0.044 [-0.075, -0.010] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_cc | ct7 | official (ANSWER_Z_Q80) | +0.015 [-0.015, +0.046] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_cc | B16__DS_FILTER_EQ | within_auc | +0.008 [-0.002, +0.019] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_cc | STEP_INDEX | within_auc | -0.259 [-0.307, -0.209] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_cc | frozen_lsml | within_auc | -0.028 [-0.048, -0.007] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_ga | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | +0.003 [-0.012, +0.019] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_ga | frozen_lsml | official (ANSWER_Z_Q80) | -0.033 [-0.054, -0.011] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_ga | ct7 | official (ANSWER_Z_Q80) | +0.027 [+0.002, +0.052] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_ga | B16__DS_FILTER_EQ | within_auc | -0.009 [-0.019, -0.001] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_ga | STEP_INDEX | within_auc | -0.276 [-0.326, -0.225] |
| Hard2Verify · Qwen3-8B | MEAN_MOM_ga | frozen_lsml | within_auc | -0.045 [-0.067, -0.023] |
| Hard2Verify · Qwen3-8B | TCI_ONLY_ga | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | -0.010 [-0.031, +0.010] |
| Hard2Verify · Qwen3-8B | TCI_ONLY_ga | frozen_lsml | official (ANSWER_Z_Q80) | -0.045 [-0.073, -0.018] |
| Hard2Verify · Qwen3-8B | TCI_ONLY_ga | ct7 | official (ANSWER_Z_Q80) | +0.014 [-0.012, +0.041] |
| Hard2Verify · Qwen3-8B | TCI_ONLY_ga | B16__DS_FILTER_EQ | within_auc | -0.033 [-0.046, -0.021] |
| Hard2Verify · Qwen3-8B | TCI_ONLY_ga | STEP_INDEX | within_auc | -0.300 [-0.352, -0.247] |
| Hard2Verify · Qwen3-8B | TCI_ONLY_ga | frozen_lsml | within_auc | -0.069 [-0.094, -0.043] |
| Socratic · Qwen3-8B | HEM_MOM_cc | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | +0.001 [-0.001, +0.003] |
| Socratic · Qwen3-8B | HEM_MOM_cc | frozen_lsml | official (ANSWER_Z_Q80) | -0.002 [-0.007, +0.004] |
| Socratic · Qwen3-8B | HEM_MOM_cc | ct7 | official (ANSWER_Z_Q80) | +0.043 [+0.037, +0.049] |
| Socratic · Qwen3-8B | HEM_MOM_cc | B16__DS_FILTER_EQ | within_auc | +0.009 [+0.007, +0.012] |
| Socratic · Qwen3-8B | HEM_MOM_cc | STEP_INDEX | within_auc | -0.046 [-0.060, -0.031] |
| Socratic · Qwen3-8B | HEM_MOM_cc | frozen_lsml | within_auc | +0.001 [-0.003, +0.006] |
| Socratic · Qwen3-8B | MEAN_MOM_cc | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | +0.001 [-0.001, +0.004] |
| Socratic · Qwen3-8B | MEAN_MOM_cc | frozen_lsml | official (ANSWER_Z_Q80) | -0.001 [-0.007, +0.004] |
| Socratic · Qwen3-8B | MEAN_MOM_cc | ct7 | official (ANSWER_Z_Q80) | +0.043 [+0.038, +0.049] |
| Socratic · Qwen3-8B | MEAN_MOM_cc | B16__DS_FILTER_EQ | within_auc | +0.006 [+0.004, +0.008] |
| Socratic · Qwen3-8B | MEAN_MOM_cc | STEP_INDEX | within_auc | -0.049 [-0.064, -0.035] |
| Socratic · Qwen3-8B | MEAN_MOM_cc | frozen_lsml | within_auc | -0.002 [-0.006, +0.002] |
| Socratic · Qwen3-8B | MEAN_MOM_ga | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | +0.004 [+0.001, +0.006] |
| Socratic · Qwen3-8B | MEAN_MOM_ga | frozen_lsml | official (ANSWER_Z_Q80) | +0.001 [-0.004, +0.005] |
| Socratic · Qwen3-8B | MEAN_MOM_ga | ct7 | official (ANSWER_Z_Q80) | +0.046 [+0.040, +0.051] |
| Socratic · Qwen3-8B | MEAN_MOM_ga | B16__DS_FILTER_EQ | within_auc | +0.002 [+0.000, +0.004] |
| Socratic · Qwen3-8B | MEAN_MOM_ga | STEP_INDEX | within_auc | -0.053 [-0.068, -0.038] |
| Socratic · Qwen3-8B | MEAN_MOM_ga | frozen_lsml | within_auc | -0.006 [-0.009, -0.002] |
| Socratic · Qwen3-8B | TCI_ONLY_ga | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | -0.006 [-0.009, -0.003] |
| Socratic · Qwen3-8B | TCI_ONLY_ga | frozen_lsml | official (ANSWER_Z_Q80) | -0.009 [-0.014, -0.004] |
| Socratic · Qwen3-8B | TCI_ONLY_ga | ct7 | official (ANSWER_Z_Q80) | +0.036 [+0.031, +0.041] |
| Socratic · Qwen3-8B | TCI_ONLY_ga | B16__DS_FILTER_EQ | within_auc | -0.005 [-0.007, -0.003] |
| Socratic · Qwen3-8B | TCI_ONLY_ga | STEP_INDEX | within_auc | -0.060 [-0.075, -0.045] |
| Socratic · Qwen3-8B | TCI_ONLY_ga | frozen_lsml | within_auc | -0.013 [-0.017, -0.009] |
| Socratic · QwQ-32B | HEM_MOM_cc | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | +0.010 [+0.006, +0.013] |
| Socratic · QwQ-32B | HEM_MOM_cc | frozen_lsml | official (ANSWER_Z_Q80) | +0.003 [-0.002, +0.008] |
| Socratic · QwQ-32B | HEM_MOM_cc | ct7 | official (ANSWER_Z_Q80) | +0.044 [+0.038, +0.050] |
| Socratic · QwQ-32B | HEM_MOM_cc | B16__DS_FILTER_EQ | within_auc | +0.031 [+0.028, +0.035] |
| Socratic · QwQ-32B | HEM_MOM_cc | STEP_INDEX | within_auc | -0.017 [-0.031, -0.003] |
| Socratic · QwQ-32B | HEM_MOM_cc | frozen_lsml | within_auc | +0.013 [+0.009, +0.017] |
| Socratic · QwQ-32B | MEAN_MOM_cc | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | +0.009 [+0.006, +0.013] |
| Socratic · QwQ-32B | MEAN_MOM_cc | frozen_lsml | official (ANSWER_Z_Q80) | +0.003 [-0.002, +0.008] |
| Socratic · QwQ-32B | MEAN_MOM_cc | ct7 | official (ANSWER_Z_Q80) | +0.044 [+0.038, +0.050] |
| Socratic · QwQ-32B | MEAN_MOM_cc | B16__DS_FILTER_EQ | within_auc | +0.027 [+0.024, +0.031] |
| Socratic · QwQ-32B | MEAN_MOM_cc | STEP_INDEX | within_auc | -0.021 [-0.035, -0.007] |
| Socratic · QwQ-32B | MEAN_MOM_cc | frozen_lsml | within_auc | +0.009 [+0.005, +0.013] |
| Socratic · QwQ-32B | MEAN_MOM_ga | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | +0.007 [+0.002, +0.011] |
| Socratic · QwQ-32B | MEAN_MOM_ga | frozen_lsml | official (ANSWER_Z_Q80) | +0.000 [-0.003, +0.004] |
| Socratic · QwQ-32B | MEAN_MOM_ga | ct7 | official (ANSWER_Z_Q80) | +0.041 [+0.036, +0.046] |
| Socratic · QwQ-32B | MEAN_MOM_ga | B16__DS_FILTER_EQ | within_auc | +0.019 [+0.015, +0.023] |
| Socratic · QwQ-32B | MEAN_MOM_ga | STEP_INDEX | within_auc | -0.030 [-0.044, -0.016] |
| Socratic · QwQ-32B | MEAN_MOM_ga | frozen_lsml | within_auc | -0.000 [-0.003, +0.003] |
| Socratic · QwQ-32B | TCI_ONLY_ga | B16__DS_FILTER_EQ | official (ANSWER_Z_Q80) | -0.008 [-0.011, -0.004] |
| Socratic · QwQ-32B | TCI_ONLY_ga | frozen_lsml | official (ANSWER_Z_Q80) | -0.014 [-0.019, -0.009] |
| Socratic · QwQ-32B | TCI_ONLY_ga | ct7 | official (ANSWER_Z_Q80) | +0.027 [+0.021, +0.032] |
| Socratic · QwQ-32B | TCI_ONLY_ga | B16__DS_FILTER_EQ | within_auc | -0.008 [-0.011, -0.006] |
| Socratic · QwQ-32B | TCI_ONLY_ga | STEP_INDEX | within_auc | -0.057 [-0.072, -0.042] |
| Socratic · QwQ-32B | TCI_ONLY_ga | frozen_lsml | within_auc | -0.027 [-0.031, -0.023] |

## הקריאה

- **Socratic:** שתי גרסאות ה־cc ו־MEAN_MOM_ga גבוהות מהממוצע הפשוט באותו בנק ב־AUC (רווחים שאינם כוללים אפס), ובמדד הרשמי ב־QwQ. מול בנק11 L-SML הקפוא — בערך שוויון. מול CT7 — יתרון של כ־4 נקודות. TCI_ONLY_ga נמוך מהממוצע הפשוט — כמו בפיתוח על חלק מהבנקים.
- **Hard2Verify:** אין יתרון לשקלול; בנק11 L-SML הקפוא גבוה מכל ארבע השיטות במדד הרשמי. N קטן.
- **מול PRM מפוקח ומבקרים (מספרי מאמר, לא משוחזרים):** ב־Socratic אנחנו כ־4–5 נקודות מתחת ל־PRM-7B וכ־10 מתחת למבקר QwQ-32B; ב־Hard2Verify כ־2–3 נקודות מתחת ל־PRM-7B (שהסף שלו כוון על הנתונים), והרבה מתחת ל־UniversalPRM ולמבקר. שם גם מיקום הצעד לבדו (0.483) גבוה מכל השיטות ומ־PRM-7B.
- **מיקום הצעד** חזק מאוד בבנצ׳מרקים האלה. כל טענת הכללה צריכה להראות שורה זו לצד השיטה.

## קבצים

`PROTOCOL.json`, `SOURCE_PARITY_GATE.json`, `SCORING_STATUS.json`, `<cell>/PREDICTIONS.npz` + `SEAL.json` + `SCORING_DIAGNOSTICS.json` + `DEV_VS_EXTERNAL.csv`, `METRICS.csv`, `CONTRASTS.csv`, `TRANSFER.csv`, `EVALUATION_PROVENANCE.json`, `figures/`, `score.py`, `evaluate.py`, `report.py`.
