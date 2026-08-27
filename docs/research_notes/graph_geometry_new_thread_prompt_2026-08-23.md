# Prompt for a new graph-geometry research thread

Copy the text below into a new Codex conversation opened at the repository
root.

---
אני רוצה לפתוח מחקר צדדי ומבודד שמטרתו לבדוק האם ניתן לזהות ולבחור גיאומטריית
גרף טובה יותר עבור מנגנון ה־pooled family-residual graph roughness, והאם אפשר
לעשות זאת ללא correctness labels.

לפני כל פעולה קרא במלואם את `CLAUDE.md`, את `PROGRESS.md`, את `HISTORY.md`,
ואת מפרט ההשקה:

`docs/experiments/GRAPH_GEOMETRY_SELECTION_RESEARCH_V1.md`

המפרט הזה הוא מקור האמת למשימה. קרא גם את כל הארטיפקטים הקנוניים שאליהם הוא
מפנה, ובמיוחד:

- `docs/experiments/POOLED_GRAPH_ROUGHNESS_DIRECTION_V1.md`
- `results/pooled_graph_roughness_direction_v2/REPORT.md`
- `results/pooled_graph_roughness_direction_v2/controls/REPORT.md`
- `docs/experiments/SU_POOLED_GRAPH_ADAPTATION_SIDECAR_V1.md`
- `docs/experiments/SU_POOLED_GRAPH_ADAPTATION_CONSERVATIVE_V2.md`
- `docs/research_notes/su_pooled_graph_adaptation_conclusion_2026-08-23.md`
- `results/family_residual_graph_liu_v3/SYNTHESIS.md`
- `results/direct_dufs_conditional_graph_topology_audit_v1/REPORT.md`

העובדה הקריטית שממנה מתחילים היא:

- הגרסה הקנונית, עם residual union-kNN קבוע ב־`k=7` ו־one-SE/tail guard,
  נתנה ‎+0.251pp מול IU-PCR.
- באותה lineage, sensitivity של max-mean עם הגרף הקבוע נתנה ‎+0.450pp.
- sidecar שחיפש גם בין ארבעה graphs, השתמש ב־max-mean ושינה את trust grid נתן
  ‎+0.452pp.

לכן אסור להניח שהפער בין ‎+0.251 ל־‎+0.452 הוא graph-selection oracle gap.
כמעט כל הפער עשוי להגיע מה־selector עצמו. המשימה הראשונה שלך היא לפרק באופן
מדויק ומבוקר את התרומות של:

1. fixed graph מול graph search;
2. one-SE/tail guard מול max-mean;
3. trust grid קנוני מול ה־grid של ה־sidecar.

בצע זאת באמצעות score bank label-free משותף, strict nested
leave-dataset-family-out, והשוואות paired שבהן רק גורם אחד משתנה בכל פעם.
אל תפרש דבר לפני שה־+0.251 וה־+0.450 משוחזרים כ־anchors מספריים.

לאחר הפירוק, בנה candidate bank קומפקטי ולא־רדונדנטי של גיאומטריות סבירות,
בהתבסס על המימושים והניסויים הקיימים. בדוק לכל הפחות residual,
unresidualized-contribution ו־DUFS coordinate controls; union/mutual/adaptive
kNN; מספר קטן של ערכי k; ורק metrics/edge weights שיש להם היפותזה מנגנונית
ברורה. אל תריץ Cartesian product בלתי מוגבל. בצע deduplication לפי edge
overlap או operator similarity ודווח את גודל ה־hypothesis class האפקטיבי.

שמור שלושה selectors נפרדים לחלוטין:

1. selector label-free קבוע מראש, המבוסס רק על graph health, perturbation
   stability, leave-family-out direction stability, consistency של `(A_e,c_e)`,
   predicted roughness descent ו־nuisance/length guards;
2. supervised donor-label meta-selector שמשתמש רק ב־training families בתוך
   כל outer fold;
3. held-family oracle לצורכי diagnostic בלבד.

אם משקולות של criterion נלמדות מ־AUROC, סווג אותן כ־supervised ואל תכנה את
השיטה unsupervised. מדוד selection agreement, rank correlation ו־regret מול
ה־oracle, אך היעד המרכזי נשאר AUROC מול exact IU-PCR והגרף הקנוני.

אל תשלב SU-rho או SU covariance cleaning: הכיוון הזה נסגר לאחר שלא הוסיף ערך
אינקרמנטלי. אל תשנה baseline קפוא ואל תדרוס תוצאות קיימות. עבוד בתיקיות
experiment/result חדשות, עם fit/report separation, hashes לפני label access,
בדיקות מכניות ו־manifests.

המחקר צריך להבחין במפורש בין ארבע מסקנות אפשריות:

- רק selector אגרסיבי יותר עוזר;
- קיימת geometry headroom שאפשר לנצל רק עם labels;
- criterion label-free אכן מזהה geometry מועילה;
- הרחבת מרחב הגרפים רק מייצרת selection optimism.

הרץ את הניסוי בפועל ואל תעצור לאחר כתיבת plan. נצל קודם caches, score banks
ו־moments קיימים; בצע inference חדש רק אם הוא באמת נדרש. אם זמינים סוכנים
מקבילים, השתמש בשני audits עצמאיים: אחד לבדיקת בידוד המנגנונים והפרוטוקול,
ואחד לבדיקת leakage, provenance ושלמות הארטיפקטים. הסוכן הראשי חייב לקרוא
בעצמו את כל הוראות הפרויקט והמפרט.

הפק plots שמראים לפחות:

- factorial effects של selector/geometry/trust;
- תוצאות paired לכל held family;
- heatmap של geometry מול family;
- קשר בין diagnostics label-free לביצועי geometry;
- oracle gap ו־selector regret;
- diversity/edge overlap של הגרפים;
- frozen retrospective transfer מול IU-PCR, הגרף הקנוני ו־Family-NRM.

ProcessBench, SemGrad, PRMBench ו־HLE כבר נפתחו היסטורית ולכן הם stress tests
רטרוספקטיביים בלבד. אל תכנה אותם confirmation. אם נמצא winner, הקפא אותו
לפני scoring חיצוני והסבר איזה dataset/model family חדש ונעול יידרש לאימות
אמיתי.

בסיום שמור report קנוני, figures, machine-readable results, config/hash
manifests, tests ו־research note עם החלטה תחומה. הצג לי את התוצאה באמצעות
ה־plots, והסבר האם מצאנו גיאומטריה טובה יותר, selector טוב יותר, או רק פער
oracle שאיננו יודעים לזהות ללא labels.

---
