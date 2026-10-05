# F11–F12: השוואת ארבע השיטות מול PRM ו־Mind the Gap

כל המדדים להלן הם על אוכלוסיית הפיתוח שכבר נחשפה. הדוח אינו מציג הערכה חיצונית לא־חשופה. כל ערך מוחלט הוא ממוצע שווה־משקל על שמונת הבנקים; ב־ProcessBench נעשה ממוצע קודם על שמונת תאי הבדיקה בכל בנק.

## מדדי החלטה לפי שיטה

| שיטה | כלל | PRMB within-answer AUC | PRMScore | PB argmax: פגיעה בשגיאה ראשונה | PB: הצעד המסומן הראשון | PB official F1 | תשובות נקיות עם דגל |
|---|---|---:|---:|---:|---:|---:|---:|
| HEM_MOM_cc | ANSWER_Z_Q80 | 0.7690 | 0.6472 | 0.3658 | 0.3269 | 0.1022 | 0.9384 |
| HEM_MOM_cc | ENTROPY_GATE_Q80 | 0.7690 | 0.6447 | 0.3658 | 0.2346 | 0.3462 | 0.3117 |
| HEM_MOM_cc | R2_COUNT | 0.7690 | 0.6369 | 0.3658 | 0.2581 | 0.3325 | 0.5243 |
| HEM_MOM_cc | TOP_1 | 0.7690 | 0.6389 | 0.3658 | 0.3658 | 0.0000 | 1.0000 |
| HEM_MOM_cc | TOP_2 | 0.7690 | 0.6721 | 0.3658 | 0.3041 | 0.0000 | 1.0000 |
| HEM_MOM_cc | TOP_3 | 0.7690 | 0.6637 | 0.3658 | 0.2349 | 0.0000 | 1.0000 |
| MEAN_MOM_cc | ANSWER_Z_Q80 | 0.7675 | 0.6468 | 0.3703 | 0.3364 | 0.1045 | 0.9369 |
| MEAN_MOM_cc | ENTROPY_GATE_Q80 | 0.7675 | 0.6439 | 0.3703 | 0.2396 | 0.3523 | 0.3080 |
| MEAN_MOM_cc | R2_COUNT | 0.7675 | 0.6363 | 0.3703 | 0.2629 | 0.3365 | 0.5243 |
| MEAN_MOM_cc | TOP_1 | 0.7675 | 0.6381 | 0.3703 | 0.3703 | 0.0000 | 1.0000 |
| MEAN_MOM_cc | TOP_2 | 0.7675 | 0.6717 | 0.3703 | 0.3133 | 0.0000 | 1.0000 |
| MEAN_MOM_cc | TOP_3 | 0.7675 | 0.6628 | 0.3703 | 0.2439 | 0.0000 | 1.0000 |
| MEAN_MOM_ga | ANSWER_Z_Q80 | 0.7674 | 0.6464 | 0.3656 | 0.3223 | 0.1112 | 0.9317 |
| MEAN_MOM_ga | ENTROPY_GATE_Q80 | 0.7674 | 0.6427 | 0.3656 | 0.2273 | 0.3404 | 0.2974 |
| MEAN_MOM_ga | R2_COUNT | 0.7674 | 0.6363 | 0.3656 | 0.2523 | 0.3275 | 0.5243 |
| MEAN_MOM_ga | TOP_1 | 0.7674 | 0.6393 | 0.3656 | 0.3656 | 0.0000 | 1.0000 |
| MEAN_MOM_ga | TOP_2 | 0.7674 | 0.6721 | 0.3656 | 0.2991 | 0.0000 | 1.0000 |
| MEAN_MOM_ga | TOP_3 | 0.7674 | 0.6625 | 0.3656 | 0.2336 | 0.0000 | 1.0000 |
| TCI_ONLY_ga | ANSWER_Z_Q80 | 0.7700 | 0.6472 | 0.3562 | 0.3123 | 0.1088 | 0.9331 |
| TCI_ONLY_ga | ENTROPY_GATE_Q80 | 0.7700 | 0.6433 | 0.3562 | 0.2213 | 0.3322 | 0.3049 |
| TCI_ONLY_ga | R2_COUNT | 0.7700 | 0.6371 | 0.3562 | 0.2462 | 0.3218 | 0.5243 |
| TCI_ONLY_ga | TOP_1 | 0.7700 | 0.6409 | 0.3562 | 0.3562 | 0.0000 | 1.0000 |
| TCI_ONLY_ga | TOP_2 | 0.7700 | 0.6728 | 0.3562 | 0.2874 | 0.0000 | 1.0000 |
| TCI_ONLY_ga | TOP_3 | 0.7700 | 0.6636 | 0.3562 | 0.2260 | 0.0000 | 1.0000 |

AUC של PRMBench אינו תלוי בכלל ההחלטה ולכן חוזר בכל שלוש השורות של אותה שיטה. `clean_flag` הוא שיעור התשובות הנקיות שקיבלו לפחות דגל אחד. ה־F1 הרשמי ב־ProcessBench תלוי בכלל ההחלטה; לכן AUC, פגיעה ב־argmax, פגיעה בדגל הראשון ו־F1 נשארים מדדים נפרדים.

## F11 · PRMBench לפי קטגוריית שגיאה

הקטגוריות הן `sensitivity`, `simplicity`, `soundness`; `control` אינו קטגוריית שגיאה ולכן נשמר בטבלה אך אינו מצויר. הטבלה היא לכלל `ANSWER_Z_Q80`; יתר הכללים ו־top-k נמצאים ב־`F11_PRM_CATEGORY_METRICS.csv`.

| קטגוריה | שיטה | within-answer AUC | PRMScore | תשובות עם AUC |
|---|---|---:|---:|---:|
| sensitivity | HEM_MOM_cc | 0.6725 | 0.5858 | 1492 |
| sensitivity | MEAN_MOM_cc | 0.6728 | 0.5866 | 1492 |
| sensitivity | MEAN_MOM_ga | 0.6730 | 0.5868 | 1492 |
| sensitivity | TCI_ONLY_ga | 0.6764 | 0.5883 | 1492 |
| simplicity | HEM_MOM_cc | 0.8230 | 0.6910 | 1516 |
| simplicity | MEAN_MOM_cc | 0.8209 | 0.6886 | 1516 |
| simplicity | MEAN_MOM_ga | 0.8226 | 0.6894 | 1516 |
| simplicity | TCI_ONLY_ga | 0.8257 | 0.6913 | 1516 |
| soundness | HEM_MOM_cc | 0.7895 | 0.6580 | 3022 |
| soundness | MEAN_MOM_cc | 0.7875 | 0.6578 | 3022 |
| soundness | MEAN_MOM_ga | 0.7862 | 0.6565 | 3022 |
| soundness | TCI_ONLY_ga | 0.7882 | 0.6563 | 3022 |

## חפיפת זיהוי שגיאות ב־F11

ספירת צעדי השגיאה המסומנים גם על ידי השיטה וגם על ידי PRM, רק על ידי השיטה, רק על ידי PRM, או על ידי אף אחד מהם. הספירות כוללות את שמונת הבנקים; אין לפרש צעדים בתוך אותה תשובה כתצפיות בלתי־תלויות.

| שיטה | כלל | גם אנחנו וגם PRM | רק אנחנו | רק PRM | אף אחד |
|---|---|---:|---:|---:|---:|
| HEM_MOM_cc | ANSWER_Z_Q80 | 18340 | 31750 | 13708 | 41394 |
| MEAN_MOM_cc | ANSWER_Z_Q80 | 18430 | 31580 | 13618 | 41564 |
| MEAN_MOM_ga | ANSWER_Z_Q80 | 18156 | 31789 | 13892 | 41355 |
| TCI_ONLY_ga | ANSWER_Z_Q80 | 18269 | 31851 | 13779 | 41293 |

## F12 · Mind the Gap ב־ProcessBench

Mind the Gap הוא השחזור המקומי הקפוא: teacher-forced, paper EMA span 5, later-token attribution, top-5 drop mean ו־native argmax. הוא אינו תוצאת הכותרת מהמאמר. ל־MTG אין מנגנון abstention לתשובות נקיות, לכן השוואת שיעורי דגלים נקיים מוצגת רק עבור השיטות שלנו.

| שיטה | argmax שלנו פוגע בשגיאה ראשונה | argmax של MTG פוגע | הצעד הראשון שסומן פוגע (R2) | מוקדם | מאוחר | ללא דגל | פגיעה |
|---|---:|---:|---:|---:|---:|---:|---:|
| HEM_MOM_cc | 0.3433 | 0.3197 | 0.2444 | 15636 (44.0%) | 3867 (10.9%) | 7347 (20.7%) | 8686 (24.4%) |
| MEAN_MOM_cc | 0.3480 | 0.3197 | 0.2507 | 15125 (42.6%) | 4155 (11.7%) | 7347 (20.7%) | 8909 (25.1%) |
| MEAN_MOM_ga | 0.3441 | 0.3197 | 0.2379 | 16275 (45.8%) | 3459 (9.7%) | 7347 (20.7%) | 8455 (23.8%) |
| TCI_ONLY_ga | 0.3342 | 0.3197 | 0.2304 | 16862 (47.5%) | 3138 (8.8%) | 7347 (20.7%) | 8189 (23.0%) |

הטבלה מאגדת את כל תשובות ProcessBench השגויות על פני הבנקים. הפירוט לפי רביעון מיקום השגיאה, אורך תשובה ותא נמצא ב־`F12_MTG_BREAKDOWN.csv`; רשומות התשובה עצמן ב־`F12_MTG_ANSWER_FORENSICS.csv.gz`.

הבהרת משקול ל־F12: הטבלה מאגדת תשובות שגויות ולכן משוקללת לפי תשובות. בטבלת המדדים הראשית כל בנק מקבל משקל שווה (ולאחר מכן ממוצע שווה בין שמונת התאים שלו). אין להשוות את הערכים המצרפיים ב־F12 ישירות לממוצע התאים שם.

## ניגודים מזווגים

הפרשים חיוביים למטה פירושם `TCI_ONLY_ga` גבוה יותר; רווחי סמך percentile של 95% מדגימת 5,000 קבוצות שאלת מקור. אלה רווחים תיאוריים ללא תיקון ריבוי השוואות.

| כלל | מול | מדד | TCI פחות השיטה הקודמת | 95% CI |
|---|---|---|---:|---:|
| ANSWER_Z_Q80 | HEM_MOM_cc | within_answer_auc | +0.0010 | [-0.0000, +0.0021] |
| ANSWER_Z_Q80 | HEM_MOM_cc | prmscore | -0.0000 | [-0.0011, +0.0010] |
| ANSWER_Z_Q80 | HEM_MOM_cc | pb_official_f1 | +0.0046 | [-0.0013, +0.0105] |
| ANSWER_Z_Q80 | HEM_MOM_cc | pb_decision_error_accuracy | -0.0170 | [-0.0226, -0.0113] |
| ANSWER_Z_Q80 | MEAN_MOM_cc | within_answer_auc | +0.0024 | [+0.0016, +0.0033] |
| ANSWER_Z_Q80 | MEAN_MOM_cc | prmscore | +0.0004 | [-0.0005, +0.0014] |
| ANSWER_Z_Q80 | MEAN_MOM_cc | pb_official_f1 | +0.0020 | [-0.0039, +0.0083] |
| ANSWER_Z_Q80 | MEAN_MOM_cc | pb_decision_error_accuracy | -0.0274 | [-0.0340, -0.0206] |
| ANSWER_Z_Q80 | MEAN_MOM_ga | within_answer_auc | +0.0026 | [+0.0019, +0.0033] |
| ANSWER_Z_Q80 | MEAN_MOM_ga | prmscore | +0.0008 | [-0.0000, +0.0016] |
| ANSWER_Z_Q80 | MEAN_MOM_ga | pb_official_f1 | -0.0035 | [-0.0084, +0.0014] |
| ANSWER_Z_Q80 | MEAN_MOM_ga | pb_decision_error_accuracy | -0.0109 | [-0.0149, -0.0070] |
| R2_COUNT | HEM_MOM_cc | within_answer_auc | +0.0010 | [-0.0000, +0.0021] |
| R2_COUNT | HEM_MOM_cc | prmscore | +0.0001 | [-0.0006, +0.0008] |
| R2_COUNT | HEM_MOM_cc | pb_official_f1 | -0.0124 | [-0.0157, -0.0091] |
| R2_COUNT | HEM_MOM_cc | pb_decision_error_accuracy | -0.0140 | [-0.0177, -0.0102] |
| R2_COUNT | MEAN_MOM_cc | within_answer_auc | +0.0024 | [+0.0016, +0.0033] |
| R2_COUNT | MEAN_MOM_cc | prmscore | +0.0008 | [+0.0002, +0.0014] |
| R2_COUNT | MEAN_MOM_cc | pb_official_f1 | -0.0178 | [-0.0217, -0.0139] |
| R2_COUNT | MEAN_MOM_cc | pb_decision_error_accuracy | -0.0203 | [-0.0245, -0.0159] |
| R2_COUNT | MEAN_MOM_ga | within_answer_auc | +0.0026 | [+0.0019, +0.0033] |
| R2_COUNT | MEAN_MOM_ga | prmscore | +0.0007 | [+0.0002, +0.0013] |
| R2_COUNT | MEAN_MOM_ga | pb_official_f1 | -0.0066 | [-0.0089, -0.0044] |
| R2_COUNT | MEAN_MOM_ga | pb_decision_error_accuracy | -0.0075 | [-0.0100, -0.0050] |
| ENTROPY_GATE_Q80 | HEM_MOM_cc | within_answer_auc | +0.0010 | [-0.0000, +0.0021] |
| ENTROPY_GATE_Q80 | HEM_MOM_cc | prmscore | -0.0014 | [-0.0024, -0.0004] |
| ENTROPY_GATE_Q80 | HEM_MOM_cc | pb_official_f1 | -0.0152 | [-0.0209, -0.0093] |
| ENTROPY_GATE_Q80 | HEM_MOM_cc | pb_decision_error_accuracy | -0.0142 | [-0.0192, -0.0091] |
| ENTROPY_GATE_Q80 | MEAN_MOM_cc | within_answer_auc | +0.0024 | [+0.0016, +0.0033] |
| ENTROPY_GATE_Q80 | MEAN_MOM_cc | prmscore | -0.0006 | [-0.0015, +0.0003] |
| ENTROPY_GATE_Q80 | MEAN_MOM_cc | pb_official_f1 | -0.0222 | [-0.0289, -0.0151] |
| ENTROPY_GATE_Q80 | MEAN_MOM_cc | pb_decision_error_accuracy | -0.0199 | [-0.0258, -0.0137] |
| ENTROPY_GATE_Q80 | MEAN_MOM_ga | within_answer_auc | +0.0026 | [+0.0019, +0.0033] |
| ENTROPY_GATE_Q80 | MEAN_MOM_ga | prmscore | +0.0006 | [-0.0003, +0.0014] |
| ENTROPY_GATE_Q80 | MEAN_MOM_ga | pb_official_f1 | -0.0088 | [-0.0129, -0.0046] |
| ENTROPY_GATE_Q80 | MEAN_MOM_ga | pb_decision_error_accuracy | -0.0067 | [-0.0102, -0.0032] |

## גרפים

- [F11 לפי קטגוריה, רקע בהיר](figures/F11_PRM_categories_revised_light.png) · [SVG](figures/F11_PRM_categories_revised_light.svg) · [רקע כהה](figures/F11_PRM_categories_revised_dark.png) · [SVG](figures/F11_PRM_categories_revised_dark.svg)
- [F12 לפי מיקום ואורך, רקע בהיר](figures/F12_MTG_position_length_revised_light.png) · [SVG](figures/F12_MTG_position_length_revised_light.svg) · [רקע כהה](figures/F12_MTG_position_length_revised_dark.png) · [SVG](figures/F12_MTG_position_length_revised_dark.svg)

## כיסוי ומגבלות

- כיסוי: 8 בנקים × 9 תאים × 4 שיטות; 1,728 שורות מדדים, 1,338,048 רשומות חפיפה, 1,305,600 רשומות ProcessBench.
- שחזור מדדים: 648 שורות legacy ו־216 שורות TCI תואמות לשלב הקודם בסבילות 1e-12.
- כללי החלטה: answer-z q80, R2, שער אנטרופיה; top-1/2/3 בנפרד. top-2 מסומן כבחירה בדיעבד.
- בדיקה עצמאית: עדיין ממתינה ל־Claude. עד לסיום הבדיקה, אלה תוצאות פיתוח ולא מסקנות למצגת.

## Claims ledger

| טענה | מספר/קובץ | N | מצב |
|---|---|---:|---|
| כיסוי מלא של הרשומות | RUN_STATUS.json; 1728 metric rows | 8×9×4×6 | שחזור פנימי עבר; ביקורת עצמאית ממתינה |
| TCI-only יתרון PRM לפי קטגוריה | F11_PRM_CATEGORY_METRICS.csv | 3 קטגוריות × 8 בנקים × 4 שיטות | תיאור פיתוח; לא בחירת זוכה |
| חפיפת PRM ו־ProcessBench/MTG | קובצי F11/F12 | 1.34M / 1.31M רשומות | תיאור פיתוח; לא ביקורת חיצונית |
| הבדל בין מתכונים | CONTRASTS.csv | 5,000 bootstrap draws | רווחים תיאוריים; ממתין ל־Claude |
