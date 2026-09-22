# בדיקת היתכנות מבודדת: CCA ו־IU מותנה בהקשר

**כל התוצאות במסמך סינתטיות. לא בוצע ניסוי איכות חדש ב־PB או ב־PRMB.**

## מה נבדק

שוחזר S0 ההיסטורי בשלושה עולמות וב־20 seeds לכל עולם. לאחר מכן נבדקו אותם נתוני הווה עם covariance אוכלוסייה ידוע, אומדני שכנים והקשר נתון או נלמד. היסטוריות העבר הן הרחבה סינתטית חדשה בת 16 תצפיות באותו משטר; אינן חלק מ־S0 ההיסטורי.

המשקלים החדשים נפתרים על covariance מלא תחת simplex, עם τ=1 ו־η=.25. native IU בשני רכיבים, גבולות τ ו־ρ אמיתי מופיעים כאבחונים נפרדים. oracle פירושו מידע סינתטי ידוע על המשטר או המטרה; אינו אלגוריתם לפריסה.

## החלטה

`STOP_CCA_FULL_RUN_CONTEXT_WEIGHTING_FEASIBLE`

זרועות שעברו את כל תנאי השיפור והבטיחות שהוגדרו: population_context_full, oracle_full, dsp_full, energy_full. שלוש זרועות CCA לא עברו את שער השיפור. לא נפתח ניסוי איכות מלא של CCA, ולא החלפנו אותו אוטומטית במועמד חדש על נתונים אמיתיים.

## מה למדנו מהמנגנון

**יש היתכנות למשקול מותנה; אין כרגע הצדקה למורכבות של CCA בהגדרה שנבדקה.** בעולם האינפורמטיבי, covariance ידוע עם QP נותן .877768 לעומת .849700 סטטי. עם שכנים והקשר המשטר הנתון מתקבל .879375: אין כאן שחזור של ההפסד ההיסטורי. ההפרש בין השניים אינו מבודד רק רעש דגימה, משום שהאמידה המקומית כוללת shrinkage שנבחר ללא תוויות.

ב־native IU עם covariance אוכלוסייה מתקבל .973425 בעולם האינפורמטיבי, אך .505853 בעולם עם nuisance. לכן כשל תחת nuisance יכול להישאר גם ללא רעש covariance. ה־QP השמרני מצמצם את הפגיעה אך אינו פותר זיהוי סמנטי: .736094 לעומת .739602 סטטי. PASS בבטיחות משמעו שהנזק בתוך המרווח המותר, ולא שהנזק אפס.

ידיעת Cov(X,target) נותנת ל־QP .911706 ו־.819271, בהתאמה. הפער מאומדן IU ממקם מגבלה באומדן רגעי המטרה/סולמו; אין לייחס את כולו לרכיב יחיד או לטעון שהאורקל ניתן ללמידה ללא תוויות. הוא משתמש במידע סינתטי שאינו זמין במשימה.

CCA ליניארי כמעט אינו משנה את הייחוס. ריבועים בעבר בלבד אינם מצילים אותו. CCA לרגעים שניים לומד תלות מוחזקת מסוימת, אבל מגיע רק ל־.852372. סיכום אנרגיה פשוט של העבר נותן .874187. העובדה ש־CCA נלמד אינה מספיקה להצדיק שימוש בו לצורך fusion. אין מכאן שלילה של כל CCA או של רגעים שניים.

בקרת ערבוב שתי הקבוצות נותנת .873368 עם הקשר האנרגיה, קרוב ל־.874187 של QP מלא. הפער הקטן ויתרונו הסינתטי מפורטים בהשוואות; אין להציגו כשוויון מוכח או כשיפור משמעותי בנתונים אמיתיים. הכיוון הפשוט הוא מועמד הגיוני יותר לדיון הבא.

השלב הבא המומלץ הוא אבחון ללא תוויות של יציבות C, rho והמשקלים בבנק האמיתי, בהקשר האנרגיה הפשוט מול מיקום ושכנים אקראיים. הוא טרם בוצע, משום שהשער הנוכחי אינו מאשר את מועמד CCA המקורי. אין לפתוח סריקת CCA כדי להתאים ל־20 seeds אלה.

## שחזור המקור

כל 60 הרשומות שוחזרו ותואמות עד דיוק נומרי; גם בדיקות המכניקה תואמות. מקורות S0 ותוצריו לא שונו. [אימות השחזור](LEGACY_REPLAY.json).

## תוצאות ממוצעות — AUC סינתטי

| שיטה | informative | null | coherent nuisance |
|---|---:|---:|---:|
| dsp_full | 0.879522 | 0.965723 | 0.736512 |
| dsp_group | 0.878172 | 0.965737 | 0.735787 |
| energy_full | 0.874187 | 0.965674 | 0.737247 |
| energy_group | 0.873368 | 0.965722 | 0.737299 |
| equal | 0.849638 | 0.965816 | 0.739697 |
| history_square_only_full | 0.849591 | 0.965723 | 0.739528 |
| history_square_only_group | 0.849780 | 0.965737 | 0.739618 |
| linear_full | 0.849622 | 0.965723 | 0.739528 |
| linear_group | 0.849823 | 0.965737 | 0.739618 |
| oracle_full | 0.879375 | 0.965707 | 0.736520 |
| oracle_group | 0.878232 | 0.965749 | 0.735925 |
| oracle_rho_context | 0.911706 | 0.965816 | 0.819271 |
| oracle_rho_static | 0.849638 | 0.965816 | 0.739697 |
| population_context_full | 0.877768 | 0.965735 | 0.736094 |
| population_context_group | 0.877778 | 0.965789 | 0.736088 |
| population_context_minvar | 0.835265 | 0.965816 | 0.772854 |
| population_context_vertex | 0.912254 | 0.957395 | 0.658724 |
| population_native_context | 0.973425 | 0.965439 | 0.505853 |
| population_native_static | 0.847336 | 0.965439 | 0.738449 |
| population_static_full | 0.849700 | 0.965735 | 0.739602 |
| population_static_group | 0.849706 | 0.965789 | 0.739616 |
| population_static_minvar | 0.849638 | 0.965816 | 0.739697 |
| population_static_vertex | 0.821453 | 0.957395 | 0.734607 |
| random_full | 0.849583 | 0.965698 | 0.739457 |
| random_group | 0.849805 | 0.965728 | 0.739627 |
| second_moment_full | 0.852372 | 0.965723 | 0.739411 |
| second_moment_group | 0.852785 | 0.965737 | 0.739430 |
| static_full | 0.849640 | 0.965723 | 0.739528 |
| static_group | 0.849812 | 0.965737 | 0.739618 |
| static_minvar | 0.849633 | 0.965324 | 0.739473 |
| static_vertex | 0.822257 | 0.958301 | 0.735640 |

## השוואות מבודדות

הפרשים מוחלטים ב־AUC, 95% bootstrap מזווג של 20 seeds עם 10,000 דגימות. הרווחים אבחוניים, ללא תיקון לכל ריבוי ההשוואות; אין לפרשם כאישור שיפור על משימות reasoning.

| מועמד פחות ביקורת | עולם | הפרש | CI 95% |
|---|---|---:|---|
| population_context_full − oracle_rho_context | informative | -0.033938 | [-0.035323, -0.032598] |
| population_context_full − oracle_rho_context | null | -0.000082 | [-0.000154, -0.000017] |
| population_context_full − oracle_rho_context | coherent_nuisance | -0.083178 | [-0.085685, -0.080667] |
| oracle_full − population_context_full | informative | +0.001607 | [-0.000668, +0.003887] |
| oracle_full − population_context_full | null | -0.000028 | [-0.000114, +0.000058] |
| oracle_full − population_context_full | coherent_nuisance | +0.000426 | [-0.000344, +0.001209] |
| dsp_full − random_full | informative | +0.029940 | [+0.027426, +0.032214] |
| dsp_full − random_full | null | +0.000025 | [+0.000000, +0.000056] |
| dsp_full − random_full | coherent_nuisance | -0.002944 | [-0.003494, -0.002397] |
| second_moment_full − linear_full | informative | +0.002750 | [+0.001792, +0.003693] |
| second_moment_full − linear_full | null | +0.000000 | [+0.000000, +0.000000] |
| second_moment_full − linear_full | coherent_nuisance | -0.000118 | [-0.000265, +0.000028] |
| second_moment_full − second_moment_group | informative | -0.000413 | [-0.000673, -0.000146] |
| second_moment_full − second_moment_group | null | -0.000014 | [-0.000128, +0.000103] |
| second_moment_full − second_moment_group | coherent_nuisance | -0.000019 | [-0.000258, +0.000206] |
| second_moment_full − energy_full | informative | -0.021815 | [-0.023956, -0.019610] |
| second_moment_full − energy_full | null | +0.000049 | [+0.000007, +0.000099] |
| second_moment_full − energy_full | coherent_nuisance | +0.002164 | [+0.001795, +0.002540] |
| history_square_only_full − linear_full | informative | -0.000030 | [-0.000190, +0.000100] |
| history_square_only_full − linear_full | null | +0.000000 | [+0.000000, +0.000000] |
| history_square_only_full − linear_full | coherent_nuisance | +0.000000 | [+0.000000, +0.000000] |
| energy_full − energy_group | informative | +0.000819 | [+0.000209, +0.001394] |
| energy_full − energy_group | null | -0.000048 | [-0.000157, +0.000065] |
| energy_full − energy_group | coherent_nuisance | -0.000052 | [-0.000469, +0.000352] |
| dsp_full − dsp_group | informative | +0.001350 | [+0.000848, +0.001858] |
| dsp_full − dsp_group | null | -0.000014 | [-0.000128, +0.000103] |
| dsp_full − dsp_group | coherent_nuisance | +0.000725 | [+0.000217, +0.001304] |
| population_context_full − population_context_group | informative | -0.000011 | [-0.000137, +0.000110] |
| population_context_full − population_context_group | null | -0.000054 | [-0.000127, +0.000013] |
| population_context_full − population_context_group | coherent_nuisance | +0.000005 | [-0.000104, +0.000096] |

## שערי קבלה

מול ייחוס סטטי תואם: שיפור informative של לפחות .005 וב־18/20 seeds; שינוי מוחלט ב־null עד .005; ירידת nuisance ממוצעת לכל היותר .005 וב־seed הגרוע לכל היותר .020.

| זרוע | שיפור | ניצחונות | null | nuisance ממוצע | nuisance קצה |
|---|---|---|---|---|---|
| population_context_full | PASS | PASS | PASS | PASS | PASS |
| oracle_full | PASS | PASS | PASS | PASS | PASS |
| dsp_full | PASS | PASS | PASS | PASS | PASS |
| linear_full | FAIL | FAIL | PASS | PASS | PASS |
| history_square_only_full | FAIL | FAIL | PASS | PASS | PASS |
| second_moment_full | FAIL | FAIL | PASS | PASS | PASS |
| energy_full | PASS | PASS | PASS | PASS | PASS |

## אבחוני התאמה

| עולם | שיטה | alpha ממוצע | שיעור g2 בתקרה | מתאמי CCA מוחזקים ממוצעים |
|---|---|---:|---:|---|
| informative | oracle_full | 0.000 | 0.247 | [] |
| informative | dsp_full | 0.025 | 0.339 | [] |
| informative | random_full | 0.975 | 0.058 | [] |
| informative | linear_full | 0.950 | 0.054 | [-0.011995707796314588, -0.011772120194824974] |
| informative | history_square_only_full | 0.975 | 0.054 | [0.009135720401202047, -0.030002988237441353] |
| informative | second_moment_full | 0.650 | 0.285 | [0.13040139348997995, 0.021371286816617673] |
| informative | energy_full | 0.025 | 0.331 | [] |
| null | oracle_full | 0.900 | 1.000 | [] |
| null | dsp_full | 1.000 | 1.000 | [] |
| null | random_full | 0.925 | 1.000 | [] |
| null | linear_full | 1.000 | 1.000 | [0.01054470334217738, -0.004287451289037695] |
| null | history_square_only_full | 1.000 | 1.000 | [0.00297816781748153, 0.008993771408297789] |
| null | second_moment_full | 1.000 | 1.000 | [0.011676854275727025, -0.007823621548607258] |
| null | energy_full | 0.900 | 1.000 | [] |
| coherent_nuisance | oracle_full | 0.000 | 0.000 | [] |
| coherent_nuisance | dsp_full | 0.000 | 0.034 | [] |
| coherent_nuisance | random_full | 0.925 | 0.109 | [] |
| coherent_nuisance | linear_full | 1.000 | 0.150 | [0.0114656124473815, 2.9988341545921643e-05] |
| coherent_nuisance | history_square_only_full | 1.000 | 0.150 | [-0.007306986929319197, -0.0029149199550252023] |
| coherent_nuisance | second_moment_full | 0.725 | 0.061 | [0.07070116099938635, 0.03178454296959378] |
| coherent_nuisance | energy_full | 0.000 | 0.035 | [] |

## מגבלות וייחוס

אין טענה ש־S0 מקיים את הנחות הרגעים האדיטיביים. גם ידיעת המשטר אינה מבטיחה זיהוי אמינות ללא תוויות. ירידה ב־loss, מתאם CCA חיובי ושיפור סינתטי הם שלושה ממצאים שונים. η=.25 מגביל את עוצמת השינוי ולכן כישלון בהגדרה זו אינו הוכחה לכישלון בכל η. הגדלת תקציב, שינוי בנק או סריקת פרמטרים לא בוצעו.

## אימות ותוצרים

אימות עצמאי של 1860 חבילות ציונים עבר: AUC מחושב בזוגות חיובי–שלילי, וציוני fusion משוחזרים מהמשקלים ומהתצפיות. בדיקות QP מול SLSQP, התאמה למימוש IU הקנוני, יחידות, זהות, בידוד ההווה מ־CCA ושכנות אקראית מופיעות בבדיקות הקוד.

[פרוטוקול קפוא](../../docs/experiments/CCA_IU_ISOLATION_GATE_20260915.md) · [טבלה מלאה](SUMMARY.csv) · [השוואות](CONTRASTS.json) · [אימות](AUDIT.json) · [החלטה](DECISION.json) · [חתימות המקור](PROVENANCE.json)
