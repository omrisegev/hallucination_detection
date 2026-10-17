# עשרת הימים האחרונים (7–17/10): מה עבד, איך זה מתחבר למצגת, ואיך להמשיך

נכתב ב-2026-10-17. המקורות הם קבצים ב-`claude/integration-2026-10-17` (וב-branch של ספטמבר, בתהליך מיזוג). זה development על נתונים שכבר נבדקו. לא הרצתי ניסוי חדש בשביל הסקירה. מה שמסומן "השערה" או "לא נבדק" הוא באמת כזה.

## 1. השורה התחתונה

מכל הכיוונים, אחד בלט: **telemetry לא עובד טוב כמאמת עצמאי, אבל הוא משלים חזק למאמת מפוקח.** הדפוס חזר בשלושה מקומות בלתי תלויים:
- **PRM boosting על PRMBench:** +1.4 עד +16.7 PRMScore בחמישה PRMs.
- **Agents:** telemetry לבד לא עובר את בקרות האורך והמיקום, אבל judge+telemetry ב-APB מוסיף 0.089.
- **LR-gap:** הפער מול LR נובע מ-selection, orientation ו-weighting, ולא מחוסר מידע ב-telemetry.

ההמלצה היא לנסח את התזה סביב הממצא הזה: "label-free fusion של מאמת עם ה-telemetry של ה-backbone שלו". להמשך יש שני פערים מרכזיים:
1. אין עדיין מבחן חיצוני ל-boosting.
2. שלושת ה-channels נבחרו עם תוויות.

## 2. מיפוי: המצגת ששלחנו (v19, 15/10) מול קווי העבודה

| שקופית | מה הוצג | קו העבודה שמאחוריו | מצב עכשיו |
|---|---|---|---|
| 5–6 | השוואת שיטות ובנקים, הוספת פיצ'רים | nested banks, LR reference | LR-gap **נסגר** (Step 464, 17/10) |
| 7–8 | מיקום ב-PRMBench, radar לפי סוג שגיאה | official evaluator, thinkfix | חוזק ב-NR/NCL, חולשה ב-CI/DR/MS |
| 9 | אילו סיגנלים מסבירים את הפרופיל | channel×type | תיאורי בלבד |
| 10–12 | למה הציונים "כמעט שווים", ספים, אורך | decision ceilings, count study | משפחת A–D **נכשלה**; כמות קבועה k=2 משקפת את מבנה הבנצ'מרק |
| 13 | הכללה ל-Socratic / Hard2Verify | external transfer | מעורב; boosting עוד לא נבדק בחוץ |
| 14–15 | PRM boosting ו-5 PRMs | prm_boost_v1 / gen_v1 | **הממצא החזק ביותר** |
| 16 | שלוש שאלות להמשך | LR-gap, כמות סימונים, PRM + סיגנלים פנימיים | (1) נסגרה, (2) נכשלה בגרסתה הנוכחית, (3) בתהליך |
| נספח 2 | תיקון ה-think block | thinkfix_rescore_v1 | נקלט. המספרים במצגת הם אחרי התיקון |

דברים שנעשו ולא היו במצגת: agent telemetry (Steps 452–454), PRM voters על ProcessBench (16/10), REWIND וסקר ספרות (Step 463), Jev (סקר בלבד), וניתוח cluster-contrast של Codex (15/10).

## 3. מה עבד, ומה לא

### עבד
- **PRM boosting.** ב-Qwen2.5-Math-PRM-7B: מ-65.49 ל-69.48 PRMScore, +3.99 [3.31, 4.67]. within-AUC עלה מ-0.801 ל-0.850. המתכון עבר קפוא לארבעה PRMs נוספים, וכל ה-CIs לא כוללים 0 (`results/prm_boosting_generalization_v1/run_v1/eval_lsml4/README.md`).
  - L-SML נותן ל-PRM משקל נמוך יותר כשהוא חלש: 0.37 במודלי Qwen, 0.23 ב-MathShepherd. כלומר שלב ה-weighting הלא-מפוקח מתפקד.
  - שלושת הסיגנלים לבד, בלי PRM, מגיעים ל-66.6. הרווח מגיע מהשילוב.
- **יתרון NR.** הוא נשאר אחרי בקרות מיקום, אורך וחזרות: +19.7 נקודות AUC [15.9, 23.6]. ב-NCL ה-CI חוצה 0 (advisor v11 completion).
- **Hard2Verify:** LR שאומן על PRMBench הגיע ל-within-AUC 0.684 ו-Balanced F1 0.428 מול PRM 0.634 / 0.391. ב-Socratic ה-PRM מוביל: 0.824 / 0.680.

### לא עבד
- **Telemetry לבד כ-PRM ב-PRMBench.** אחרי התיקון N16 נשאר בערך 0.78 within-AUC, ו-PRMScore בערך 65.4, כמו PRM-7B. אין שיטה label-free שמשפרת את N16 באופן מובהק.
- **הרחבת בנקים.** LR עולה מ-0.785 ל-0.836, ו-L-SML קורס ב-N40 (0.724). זה הוסבר וה-LR-gap נסגר.
- **אמידת כמות ללא תוויות (A–D).** אף זרוע לא עברה. global cut נותן רק +0.3, כמות אמיתית לכל תשובה (oracle) נותנת 71.7, ו-k=2 קבוע (post hoc) נותן 67.9.
- **PRM voters על ProcessBench.** PRM-7B לבד (0.944) עדיף על L-SML של 4 voters (0.913).
- **Agent telemetry לבד.** ה-signal ש-telemetry נותן לבד מוסבר באורך ובמיקום.

## 4. איך התוצאות משליכות זו על זו

1. **LR-gap ו-boosting.** שאלת "איך מתקרבים ל-LR בלי תוויות" מתחלפת בשאלה טובה יותר: המידע המפוקח מגיע מה-PRM, ו-telemetry מוסיף מידע שאין לו. אבל `results/prm_boosting_v1/README.md` מראה שכש-PRM נכנס כ-channel אחד מתוך N16 ב-MEAN_MOM_ga, הוא "נבלע". רק כשהוא הוחזק כקבוצה נפרדת (½·PRM + ½·N16) התקבל 68.53. **מסקנה:** המבנה (PRM כקבוצה משלו) חשוב יותר מבחירת ה-channels. זה גם הדרך לנקות את הבעיה שה-channels נבחרו עם תוויות.
2. **Decision ceilings ו-boosting.** אצלנו global cut נותן +0.3, אבל ל-PRM הוא נותן +2.9. כלומר ל-PRM יש מידע כמותי (calibrated head) שאין ל-telemetry. **השערה שלא נבדקה:** לקחת את הכמות לכל תשובה מה-PRM (מספר הצעדים עם reward < 0.5, כולל 0) ואת הסדר מה-fusion. זה label-free, מכוון בדיוק לפער של per-answer count שבו ה-oracle מגיע ל-71.7, ולא דורש מודל כמות חדש.
3. **מבנה הספירה ופרשנות ה-PRMScore.** PRMBench מזריק בערך 2 שגיאות לתשובה (ממוצע 2.12, ובלי קשר לאורך). חלק מכל הבדל ב-PRMScore תלוי בכלל ההחלטה ובמבנה הזה. לכן כל טענת boosting צריכה לבוא גם עם within-AUC (הוא עלה בכל חמשת ה-PRMs) וגם עם בנצ'מרק חיצוני.
4. **פרופיל השגיאות.** telemetry חזק ב-NR/NCL (Simplicity) וה-PRM חזק בקטגוריות אחרות. הרווח הגדול של boosting הוא בדיוק ב-redundancy (49 → 67.7) וב-circular (55 → 71.8). זה מתיישב עם Reward Under Attack (2603.06621), שמראה ש-PRMs טקסטואליים מתנהגים כגלאי fluency. יש כאן סיפור מנגנון שאפשר לבדוק: telemetry מתקן הטיית fluency של ה-PRM.
5. **Voters.** ב-ProcessBench ה-voters החלשים פגעו. ב-boosting, L-SML הוריד את המשקל של PRM חלש בצורה נכונה. התחזית ל-run ב-PRMBench (שה-judges שלו כבר הסתיימו): fusion של PRMs בלבד צפוי להחזיק, ו-judges חלשים צפויים לפגוע. כדאי לקבוע את זה מראש כמבחן.
6. **Agents.** אותו דפוס בדיוק: telemetry לבד נכשל, ו-judge+telemetry עובד (+0.089). זו עוד דוגמה ל"מאמת + telemetry שלו", ולא כיוון נפרד. ה-control של judge+length/position חייב לבוא לפני כל טענה.
7. **Think-fix.** הוא פגע מעט ב-telemetry של Qwen3 (−0.009 AUC בממוצע) וב-LR. ה-boosting לא מושפע, כי ה-backbones של ה-PRMs הם Qwen2.5, Llama ו-Mistral, ואין בהם think block.

## 5. כיוונים רלוונטיים שלא עבדנו עליהם בכלל

| כיוון | למה הוא רלוונטי | עלות |
|---|---|---|
| **Boosting מחוץ ל-PRMBench** (Socratic, Hard2Verify, ובשורה משנית ProcessBench) | הפער הכי גדול בטענה. ה-runs החיצוניים של 12/10 שמרו רק את ה-reward של ה-PRM, בלי telemetry של ה-backbone | GPU, הרצת telemetry חיצונית |
| **בחירת channels בלי תוויות** | שלושת ה-channels נבחרו עם תוויות של PRM-7B. אפשר להחליף ב-PRM + כל N16 כקבוצה אחת, או ב-L-SML עם ה-PRM כקבוצה קבועה | CPU |
| **כמות מה-PRM, סדר מה-fusion** (סעיף 4.2) | מכוון ישירות לפער per-answer count (71.7 ב-oracle) | CPU |
| **השוואה מול ה-leaderboard של PRMBench** | 69.48 עומד ליד Gemini-2-Thinking (68.8). צריך לוודא אותו evaluator, אותו subset ואת תיקון התוויות של Step 313 | CPU |
| **PRMs גנרטיביים ("thinking") כבסיס או כ-voter** (ThinkPRM, GenPRM) | זה הטרנד המוביל ב-2026 (TIPS 85.2 ב-ProcessBench). שאלה פתוחה: האם boosting עובד גם על PRM חזק מאוד | GPU |
| **Force/Remove sanity check** (2605.08346) על ה-features | בדיקת שפיות לפני טענה ברמת הצעד | CPU |
| **בדיקת מנגנון fluency** (סעיף 4.4) | ממיר את פרופיל הקטגוריות לסיפור שאפשר לבדוק | CPU |
| **Skywork PRM** (n+1 rewards) | חסום עד החלטה שלך לגבי ה-protocol | החלטה |
| **AgentProcessBench** | בנצ'מרק חיצוני לקו ה-agents, אם ממשיכים בו | GPU |

## 6. איך להתקדם (סדר מוצע)

1. **CPU, מיד:**
   - ניקוי הבחירה עם התוויות: PRM כקבוצה עם N16 מלא, בלי בחירת channels.
   - הכלל "כמות מה-PRM, סדר מה-fusion".
   - השוואה מול ה-leaderboard.
   שלושתם על הנתונים הקיימים של 5 ה-PRMs.
2. **כשה-critic מסתיים:** ניתוח voters ב-PRMBench, עם כלל go/no-go שנקבע מראש לפי סעיף 4.5.
3. **GPU, דורש אישור:** telemetry של ה-backbone של PRM על Socratic ועל Hard2Verify, כדי לבחון boosting בחוץ.
4. **CPU, אחר כך:** control ל-APB של agents ובדיקת מנגנון fluency.
5. **לא להשקיע:** חיפוש שיטות על telemetry בלבד, מודלי כמות מסוג A–D, Jev, והמשך עבודה על ProcessBench.

## מקורות

- `results/advisor_page_v19_2026-10-15/SPEAKER_NOTES_HE.md` (המצגת)
- `results/prm_boosting_generalization_v1/run_v1/eval_lsml4/README.md`
- `docs/reports/ADVISOR_MEETING_EXTERNAL_AND_PRM_BOOSTING_FINDINGS_20261015.md`
- `results/thinkfix_rescore_v1/README.md`
- `results/count_structure_check_v1/README.md`
- `results/label_free_error_count_v1_2026-10-12/FINDINGS_HE.md`
- `results/prm_voters_processbench_pilot_2026-10-16/README.md`
- `results/agent_step_telemetry_v1_competitor_fusion_pilot_2026-10-17/README.md`
- `docs/research_notes/literature_survey_localization_prm_2026-10-16.md` (Part C)
- `docs/research_notes/LSML_VS_SUPERVISED_LR_CONTRAST_DISCOVERY_20261015_HE.md`
- HISTORY Steps 453, 454, 463, 464

## תיקון (2026-10-17, ערב)

- **ספירת שגיאות:** "לעזוב" מתייחס רק למשפחת A–D, שמשערכת ספירה מ-telemetry לבד. נשארים פתוחים:
  - MoM על כמה מאמתים בלתי תלויים. זו מטרה 1 של ניתוח ה-voters ב-PRMBench.
  - ספירה מה-PRM עם סדר מה-fusion.
  - מודל emission רציף עם תלות, שעדיין לא בוצע.

  זה ה-headroom הגדול ביותר: מ-65.4 ל-71.7 עם ספירה מושלמת (oracle).
- **מחוץ למתמטיקה:** רק agents רץ (Steps 452–454). BonaFide קיבל אישור ונכתב לו handoff ב-4/10 (`docs/HANDOFF_BONAFIDE_AND_THINKING_TO_RECALL_2026-10-04.md`), אבל לא רץ. MedPRMBench נדחה ב-24/9. LSR-Ben ו-PRMs Meet Planning מופיעים רק בסקר. זה גם המועמד הטבעי למבחן boosting חיצוני.
- **Thinking PRMs:** המתכון זהה (מאמת ועוד ה-telemetry שלו). החידוש הוא telemetry על ה-reasoning של המאמת עצמו, והשאלה אם יש רווח מעל מאמת חזק מאוד. זה יקר (generation), ולכן אחרי מבחן boosting חיצוני.
