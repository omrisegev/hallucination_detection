# חיפוש גרסת L-SML עם תרומה ממשית ואיכות PRMBench גבוהה

2026-09-23. בקשת Omri: לבחור גרסה שבה הלמידה ב־L-SML מועילה, גם אם בנק הפיצ'רים שונה מ־CT7 וגם אם ProcessBench חלש יותר. הביקורת מכוונת לניסויי 24 באוגוסט–23 בספטמבר; אינה טענה שכל ניסוי וכל קובץ בחודש נבדקו.

## 1. ממצא מוביל: בנק 11 הערוצים ברמת step

נמצא ניסוי מתאים להמשך: `step_level_bank_baseline_v1`, זרוע `continuous` (אותה זרוע נקראת `bank11_continuous` בניסוי depth המשלים). זהו continuous L-SML על 11 ערוצי telemetry, אחרי normalization מקומי לתשובה; משקלי fusion נלמדו בחמישה donor source-fold fits. אין encoder, אימון GPU או מעבר LLM חדש. זו **אינה** הוכחה ל־fit מתוך התשובה הנוכחית בלבד.

מקור הניסוי הוא worktree `depth-feature-fusion-v1`: קובצי `RESULTS.json` ו־`STEP_SCORES.npz` תחת `results/step_level_bank_baseline_v1` ו־`results/bank_plus_depth_fusion_v1`. אלה כוללים חומר מקומי לא מגובה ב־Git, ולכן ה־worktree נשמר בניקוי.

שחזור עצמאי: התאמת 145,597 תוויות ל־metadata המתוקן; בדיקת כל 6,969 רשומות PRMB; התאמת 6,030 תשובות mixed-label; שחזור משקלי חמשת ה־folds על בנק המקור ושחזור equal עד tolerance ‏1e-12; שחזור CT7 within-AUC. ה־PRMScore חושב מחדש מהחלטות validity באמצעות evaluator הרשמי המקומי, ונבדק גם באמצעות confusion counts עצמאיים על 83,371 צעדים ב־6,211 הרשומות שאינן correct controls.

| אותה אוכלוסייה ואותו q80 audit | within-AUC | PRMScore אחרי answer-z סופי |
|---|---:|---:|
| בנק 11 — מיצוע רגיל | 74.9644 | 63.3111 |
| בנק 11 — קבוצות L-SML, משקל שווה לקבוצות | 75.3084 | 63.7245 |
| **בנק 11 — L-SML מלא** | **76.4531** | **64.1184** |
| CT7, ייחוס בלבד | 77.2397 | 64.5689 |

כל המספרים בטבלה הם מדדים כפול 100. ה־PRMScore החדש הוא **כיול q80 רטרוספקטיבי על ציוני OOF קיימים**, ולא שחזור של training/calibration מקונן. בחינת שיפור הדירוג אינה דורשת סף; טענת deployment מאומתת ל־PRMScore תדרוש שחזור nesting נפרד.

לעומת equal, תרומת L-SML היא +1.4887 נקודות within-AUC ו־+0.8073 נקודות PRMScore. שניהם חיוביים גם לאחר תיקון לריבוי ההשוואות בתוך הביקורת החדשה. מול מיצוע הקבוצות, יתרון הדירוג נשאר; היתרון ב־PRMScore אחרי answer-z אינו עובר את התיקון. לכן הלמידה אינה מיותרת, אך עדיין צריך לשפר/לאמת את התועלת הרשמית מעבר ל־group balancing בלבד.

ההשוואה ל־PB מסבירה למה הכיוון עלול היה להידחות: בנק11 L-SML מקבל שם SLA ‏34.7755 לעומת CT7 ‏39.8862 — פער 5.11 נקודות. ב־PRMScore עם אותו כיול/נרמול, הפער מה־CT7 הוא רק 0.4506 נקודות. אלה אינם אותם מדדים ואין להשוות את יחידות הקושי שלהם, אבל אין הצדקה להסיק מכישלון ב־PB שהבנק אינו תחרותי גם ב־PRMBench.

11 הערוצים: `q15_H1`, `q15_VE1`, `chosen_surprisal`, `logprob_margin`, `true_tail50`, `energy_level`, `energy_innovation`, `top15_turnover`, `top50_js`, `dominant_freq16`, `bocpd_p0`. אין digit channel. הוספת ארבעה רכיבי depth או novelty שנבדקה באותו worktree **פגעה** בשני מדדי PRMB, ולכן אינה חלק מהמועמד המוביל.

## 2. שילובים נוספים שבהם fusion תרם

נוצר [אינדקס מכונה](../../results/prmbench_lsml_month_inventory_v1/INVENTORY.json) עם 67 שורות L-SML משישה bundles. הספירה כוללת שכפולים היסטוריים, soft/pmf ושורות shuffle; אינה 67 ניסויים בלתי תלויים. נשמרו שמות מדויקים, comparators, metric files ו־hashes. הטבלה הבאה מציגה ניגודים שימושיים, ללא סינון לפי PB.

| גרסה קיימת | within-AUC, equal → learned | PRMScore q80, equal → learned | פירוש |
|---|---:|---:|---|
| token L-SML על 11 ערוצים | 73.1105 → 75.3164 | 61.6949 → 63.0372 | תרומה נקודתית בשניהם; בחבילות שנבדקו לא נמצא paired PRMB CI למתכון הזה |
| cumulative Top5 hard continuous L-SML | 73.4951 → 75.8484 | 60.2963 → 62.4742 | within CI חיובי ו־Holm=.0328; יש לפרק בנפרד תרומת standardization מול משקלים |
| cumulative Top5 soft continuous L-SML | 75.3154 → 75.5038 | 62.9741 → 63.3305 | נקודות חיוביות, אך within CI כולל אפס |
| plain evidence +L-SML | 75.8912 → 76.1901 | raw כ־58.31 → 58.47; answer-z ‏63.54 → 63.66 בביקורת החדשה | אות דירוג טוב; PRMScore חלש בחלקו בגלל scale; יתרון PRMScore של הלמידה עדיין לא מובהק |
| selected soft continuous L-SML | 76.0480 → 76.1480 | 64.1277 → 64.1371 | אין תרומה מבוססת ללמידה; בחירת readouts משתמשת ב־training truth labels |

הטבלה מציגה פרוטוקולים קיימים שונים, ולכן אינה leaderboard מאוחד. בפרט, q80 המקורי של cumulative משתמש בגישת donor של אותו harness; q80 בביקורת החדשה משתמש בציוני OOF שמורים. השוואה סיבתית רק בתוך זוג תואם. `inner-selected` הוא label-tuned ומופיע בנפרד באינדקס, לא כ־label-free.

תיקון חשוב לקריאת Step432: השם `equal_std` מטעה. בפועל הקוד מחזיר mean של עמודות הקלט ללא donor standardization; Continuous L-SML כן משתמש ב־fit mean/scale. `plain_equal = 11 * plain_equal_std` בדיוק נומרי. לפיכך יתרון ה־pipeline כאן אינו מבודד בהכרח משקל נלמד מול preprocessing. הביקורת של bank11 לעיל מוסיפה גם שחזור מהמשקלים ובקרת קבוצות כדי להעמיק את הייחוס.

## 3. מדוע חלק מה־PRMScores נראו נמוכים מדי

S0-C של Claude בדק normalization מחדש לזרועות **equal**, ולא ל־plain evidence L-SML. הביקורת החדשה השלימה את החסר מציוני L-SML השמורים:

- `evidence__all__plain__continuous_lsml`: ‏58.4707 raw → **63.6571 answer-z**.
- אותה בקרה `equal_std`: ‏58.3129 → **63.5396**.
- within-AUC לא השתנה: L-SML ‏76.1901 מול equal ‏75.8912.

שיפור של כ־5.19 נקודות PRMScore כאן נובע משינוי scale לפני threshold; הוא אינו שיפור חדש בלוקליזציה. זה תומך בחשד של Omri שהפסד במדד אחד אינו מוכיח שכל המידע בפיצ'רים נחות. מנגד, ה־CI המזווג של יתרון ה־PRMScore הקטן של L-SML על equal כולל אפס, ולכן איננו מציגים אותו כהוכחה לעליונות רשמית.

## 4. ניסויי 20 הפיצ'רים מספטמבר 17

הסיכום ההיסטורי "no fusion rule beats block averaging" היה רחב מדי אם מפרשים אותו גם ביחס ל־PRMB. ב־RUN.json קיימים יתרונות דירוג ל־cross-group SML ול־Joint על אותה חלוקה, לצד חולשה ב־PB. יש להבחין בין `csml` — SML בין ממוצעי קבוצות, `lsml` — continuous L-SML בשתי הרמות, ו־`jrel` — משקלי אמינות של מודל Joint. אין לכנות כל אחד מהם L-SML מלא.

ציוני OOF לא היו זמינים מקומית. נמצאו ב־Drive והורדו רק שני קבצים (כ־72MB):

```text
gdrive:hallucination_detection/consolidated_results/local_backup_2026-09-17/atlas/digitfree20_ladder_v1/OOF.npz
SHA256 a1e449e974b2975d0dd05c1c42797b6f9d73c34b748ab181d40d588adc436e32
gdrive:hallucination_detection/consolidated_results/local_backup_2026-09-17/atlas/fusion_independence_atlas_v1/dependence/EVALUATION.npz
SHA256 9c4589ccf4150403ab10dd61d77e93c07ef5e1330e83df1449eadfe17a2549cb
```

ה־hashes אומתו, וה־metadata קושר את סדר הרשומות ל־JOINED.json הקנוני. הושלם חישוב PRMScore לכל 64 הזרועות השמורות, ללא fit חדש; כל מדדי within-AUC שוחזרו מול RUN.json עד 1e-10. Affinity בחר K3 אוטומטית בכל fold; Eq15 בחר K6, ולכן K4 הוא בחירה רטרוספקטיבית ולא מועמד שנרשם מראש.

| זרוע היסטורית | within-AUC | PRMScore, answer-z + q80 |
|---|---:|---:|
| equal20 | 75.5810 | 63.7212 |
| continuous_auto | 75.4963 | 63.7084 |
| affinity K3, מיצוע קבוצות | 74.9382 | 63.1373 |
| affinity K3, cross-group SML | 75.3569 | 63.5411 |
| affinity K3, L-SML מלא | 74.9228 | 63.0887 |
| eq15 K4, מיצוע קבוצות | 74.8408 | 63.3557 |
| eq15 K4, Joint reliability | 75.7410 | 63.5237 |

Cross-group SML ב־Affinity K3 משפר PRMScore על אותה חלוקה ב־0.4039 נקודות, עם CI מתוקן [0.1602, 0.6479]. זו ראיה לתרומה בתוך המתכון, אך התוצאה המוחלטת נמוכה מ־equal20 ומבנק11 L-SML. יתרון הדירוג של Joint K4 אינו מתורגם ליתרון PRMScore על equal20. זרוע ה־ladder הנלמדת הטובה ביותר לפי PRMScore בביקורת היא affinity K6 csml, עם 63.7855 ו־within-AUC של 75.7401; גם היא נמוכה מבנק11 בשני המדדים. אין בכך הוכחה שכל שימוש עתידי ב־20 ערוצים נכשל.

הפלט הקנוני הוא [AUDIT v4](../../results/prmbench_lsml_month_audit_v4/AUDIT.json): 74 שיטות, raw ו־answer-z לכל אחת, כלומר 148 שורות מדדים. [טבלת המדדים](../../results/prmbench_lsml_month_audit_v4/DEPTH_AND_BANK_PRMSCORE.csv) שומרת שם היסטורי אך כוללת גם את כל זרועות ה־ladder. נשמרו החלטות validity, ספים, hashes ו־[עותק הקוד שהורץ](../../results/prmbench_lsml_month_audit_v4/CODE_SNAPSHOT.py). גרסאות v1–v3 הן תיעוד ביניים; אין לערבב את תיקוני ריבוי ההשוואות שלהן עם v4.

ב־v4 תוקנו 30 ניגודים: עשרה זוגות כפול שלושה endpoints. לבנק11 L-SML מול equal מתקבלים CI מתוקנים של [0.9896, 1.9759] נקודות within-AUC ושל [0.3340, 1.2742] נקודות PRMScore אחרי answer-z. מול מיצוע הקבוצות, CI ה־PRMScore הוא [-0.0715, 0.8706] ולכן אינו מבסס יתרון רשמי מעבר לאיזון הקבוצות.

## 5. היקף הכיסוי ומגבלות ההשוואה

בנוסף לששת bundles האחרונים, נסקרו 10 משפחות מספריות רלוונטיות: digitfree20 ladder, broad50, broad50 Top8, mass-membership, historical Joint v3, Renyi v2, step_evidence_fusion הישן, cumulative vote, Step432 evidence ו־token probability. יש חפיפה בין קבוצות הסקירה. עבודות ספטמבר המוקדמות עם תוויות קודמות, pilots קטנים ומדדי AUROC ברמת תשובה אינן ראיה בת־השוואה ל־official PRMScore הנוכחי; לא תויגו כ"כישלון L-SML".

אין הוכחה כאן ליתרון על PRM מפוקח בכיול תואם, או לתוצאה על מבחן שלא נחשף לפיתוח. תיקון Bonferroni בביקורת החדשה מכסה את רשימת הניגודים המקומית בלבד, לא את כל הבחירות שבוצעו במהלך החודש. ה־bootstrap מזווג על 707 source groups, ‏100,000 draws, ומותנה במשקלים ובספים שכבר חושבו. הוא אינו refit bootstrap.

בנק11 ו־evidence הם donor-fitted; נרמול מקומי אינו הופך אותם ל־answer-only. העברה ל־fit מקומי בלבד היא שינוי מחקרי נפרד. מבחינת מגבלת החישוב, אלו מודלים קטנים על CPU ולא אימון נוסף של LLM.

## 6. המשך תחום

1. בסיס ההמשך הוא bank11 step continuous L-SML. CT7 נשאר עוגן חיצוני; אין להכריח את הבנק החדש לחזור למתכון CT7.
2. לשחזר את המתכון על כל האוכלוסייה עם separation מלא בין fit/calibration/evaluation. להשוות raw ו־answer-z כניגודים נפרדים ומוצהרים, לתת לכל baseline אותו calibration budget, ולשמר ordinary equal וגם learned-partition equal.
3. כדי להוכיח שהחלק הנלמד עצמו חיוני: לשמור partition קבוע, לבצע weights learned מול group-balanced; להוסיף אותה donor-column standardization לבקרה אם היא נכנסת ל־L-SML. אין לערבב feature search, readout ו־fit scope באותו ניגוד.
4. רק לאחר replay תקין, לבדוק שינוי יחיד בייצוג או ב־readout. residual מנורמל לערוץ נשאר תוספת אפשרית, לא הנחת עבודה. depth/novelty הגרסאות שנמדדו אינן מתווספות אוטומטית.
5. לדווח PB לצד PRMB, בלי להשתמש בירידה ב־PB לביטול היתרון ב־PRMB. לא להשיק אימון GPU או inference חדש.

תוכנית הביצוע המפורטת ומגבלות החישוב: [PRMBENCH_RUNTIME_FUSION_PLAN_HE.md](../experiments/PRMBENCH_RUNTIME_FUSION_PLAN_HE.md). תיעוד פינוי המקום: [WORKTREE_CLEANUP_20260923.md](WORKTREE_CLEANUP_20260923.md).
