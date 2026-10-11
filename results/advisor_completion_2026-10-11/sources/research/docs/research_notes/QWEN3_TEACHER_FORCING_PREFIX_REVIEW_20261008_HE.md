# Qwen3: teacher forcing ו-prefix של `/no_think` — נקודת המשך לאחר הפגישה

**תאריך:** 2026-10-08

**מצב:** בדיקת קוד ותיעוד בלבד. לא הורץ מודל, לא נפתחו תוויות ולא שונו ציונים או פרוטוקולים.

## מה בדקנו

### 1. התאמת teacher forcing לטלמטריית יצירה

`cluster/run_teacher_forced.py::run_gate_b` טוען תשובות שנוצרו, משחזר את הודעת המשתמש ואת `gen_token_ids`, מחשב מחדש אנטרופיה ב-teacher forcing ומשווה אותה לאנטרופיה השמורה. זו הבדיקה הישירה להתאמת prompt/template ונוסחת האנטרופיה.

יש תיעוד נוסף של בדיקת 40 תשובות GSM8K: התאמת top-50, אנטרופיה בהפרש מרבי של `2.1e-7`, surprisal של הטוקן שנבחר, ובדיקת `validate_telemetry`. הקשר הנתונים כולל Qwen3-4B ו-Qwen3-8B. עם זאת, התיעוד אינו מפרט את שיוך 40 הדוגמאות לכל מודל ואינו מצרף פלט Gate B שמוכיח השוואת teacher-forced מחדש לטלמטריית יצירה לכל דוגמה. יתרה מזו, `spectral_utils/external_generalization/fusion.py::validate_telemetry` בודקת יישור, סדר ותקינות פנימית של שדות שמורים; היא אינה מריצה את המודל מחדש.

לכן יש בריפו **בדיקת Qwen3 מתועדת**, אבל היא לבדה אינה מוכיחה Gate B מלא עבור פרוטוקול Qwen3/no-think. לגבי הריצה החיצונית, ה-runbook קובע שה-cache שנמצא אינו תואם no-think, ומזהיר שבדיקת serial-prefix מול batched teacher forcing אינה תחליף ל-Gate B. ה-Gate B המפורש במניפסט הרלוונטי הוא עבור Llama-3.1-8B. לכן הניסוח המדויק הוא: חסרה הוכחה מתועדת ל-Gate B על cache Qwen3 שנוצר תחת אותו פרוטוקול no-think; לא נכון לומר שלא נעשתה שום בדיקת Qwen3.

### 2. בלוק `<think></think>` ריק

ב-snapshot הרשמי והמאומת של tokenizer Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`, תבנית הצ'אט מוסיפה אחרי כותרת ה-assistant את `<think>\n\n</think>\n\n` כאשר קוראים ל-`apply_chat_template` עם `enable_thinking=False` ו-`add_generation_prompt=True`. פרוטוקול A6 דורש במפורש את הדגל הזה ובודק prefix, טוקניזציה ו-offsets.

המסלול הוותיק של `/no_think` שונה: הוא מוסיף את המחרוזת ` /no_think` להודעת המשתמש ומשתמש בתבנית ברירת המחדל בלי `enable_thinking=False`. לכן ה-prefix ההיסטורי מסתיים בכותרת assistant ואינו כולל את הבלוק הריק בתוך הפרומפט. במקביל, `spectral_utils/data_loaders.py::strip_think` מתעד ש-Qwen3 עשוי לפלוט בלוק think ריק גם תחת `/no_think`, ומסיר אותו בחילוץ התשובה.

מכאן שהחשש לגבי הפתעה מנופחת בטוקנים הראשונים **סביר אך לא הוכח**: ייתכן שטקסט תשובה שעבר הסרת think מקבל ניקוד מיד אחרי כותרת assistant, במקום אחרי הבלוק המבני שהמודל היה פולט או שהתבנית המפורשת הייתה מכניסה. צריך לבדוק את ה-prefix ואת הטוקנים בפועל של הריצה הרלוונטית, ולא להסיק זאת מה-template בלבד.

## מה לעשות כשחוזרים לנושא

1. לזהות את ה-cache והדוגמאות ששימשו לבדיקת 40 התשובות; לרשום לכל דוגמה model/tokenizer revision, prompt text, `prompt_ids`, `gen_token_ids` והאם ה-think block נשמר בתחילת הפלט.
2. עבור Qwen3-4B ו-Qwen3-8B בנפרד, לשחזר את ה-prefix ההיסטורי (`/no_think` במשתמש + template ברירת מחדל), ולהשוות את ה-token IDs שלו ל-prefix השמור בזמן היצירה.
3. רק כבדיקת אבחון נפרדת, לבנות גם prefix עם `enable_thinking=False`. להשוות את surprisal/entropy של 1, 3 ו-5 טוקני התוכן הראשונים ושל הצעד הראשון בשתי התצורות; לשמור בנפרד את הטוקנים המבניים ולא לערבב אותם עם טקסט התשובה.
4. אם קיים cache שנוצר מאותו Qwen3, revision ופרוטוקול no-think, להריץ Gate B לכל מודל ולהשוות אנטרופיות token-aligned עם ספי הפרוטוקול. אם אין cache מתאים, לסמן Gate B כפתוח; בדיקת serial-vs-batched בודקת עקביות חישובית בלבד.
5. לפני מסקנה או שינוי scorer, להחליט ולתעד אם הפרוטוקול הנבחר מייצג את הבלוק הריק בתוך ה-prefix או כטוקנים שנוצרים. לא לשנות ציונים קפואים או לטעון שהפער הוא סיבתי בלי בדיקת התאמה זו.

## מקורות בריפו

- `cluster/run_teacher_forced.py` — מימוש Gate B והשוואת entropy.
- `cluster/run_external_telemetry.py` ו-`spectral_utils/model_utils.py` — בניית prefix עם template ברירת מחדל.
- `docs/experiments/LSML_EXTERNAL_GENERALIZATION_V1_RUNBOOK.md` — cache no-think חסר וההבחנה בין alignment gate ל-Gate B.
- `docs/meetings/advisor_prep_2026-10-05/sources/own_protocol.md` — בדיקת 40 דוגמאות GSM8K.
- `spectral_utils/external_generalization/fusion.py` — מה `validate_telemetry` בודקת בפועל.
- `spectral_utils/data_loaders.py` — הסרת בלוק think ריק מתשובות Qwen3.
- `docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_S0_S1_EXECUTION_V1.md` וה-snapshot ב-`results/automatic_group_free_phase_a6_tokenizers_v2/evidence/official/qwen3-8b.json` — שימוש מפורש ב-`enable_thinking=False` ותבנית tokenizer.
