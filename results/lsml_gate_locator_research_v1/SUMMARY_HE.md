# סיכום מחקר L-SML ל-gate ול-locator

המחקר בוצע על מלוא 13,769 התשובות, עם source-fold cross-fitting. משקולות
וקבוצות ה-fusion חושבו ללא labels; labels שימשו רק לבחירת roster בתוך nested
folds ולהערכת התוצאות. כל הממצאים הם development evidence ולא confirmation.

## התוצאה המעשית

ה-gate הקיים Tail15 + digit-rate נשאר עדיף. המועמד הטוב ל-locator הוא roster
בן שמונה streams:

1. digit disagreement Top2
2. digit token-clock innovation Top1
3. VE0.75 prefix innovation Top10
4. Renyi a0.25 Top10
5. direct-probability rank-1 Top10
6. direct-probability rank-9 Top8
7. logtail15 Top10
8. mass-above Top10

Continuous L-SML על roster זה נבחר ב-5/5 folds והגיע ל-43.7402% PB ול-0.778143
within, לעומת 43.2546% ו-0.776036 באלגוריתם הנוכחי. השיפור הוא +0.4856pp PB
ו-+0.002107 within, אך רווחי הסמך של bootstrap מקור-מקובץ עדיין חוצים אפס:
PB [-0.5679,+1.5266]pp; within [-0.000555,+0.004851]. לכן זה research candidate,
לא winner מאושר.

## תובנת הפישוט

L-SML גילה בדיוק אותן שלוש קבוצות בכל חמשת ה-folds:

- שני ערוצי digit;
- VE0.75 innovation יחד עם mass-above;
- Renyi/direct-probability/logtail.

המשקל הכולל של כל קבוצה כמעט שליש. החלפה post-hoc במשקולות family-equal קבועות,
ללא fit וללא גילוי קבוצות, נתנה 43.7745% PB ו-0.778222 within. זו כרגע הצעת
הפישוט הטובה ביותר, אבל היא נולדה לאחר צפייה במבנה L-SML ולכן דורשת ריצת
cross-fitted/confirmation נפרדת לפני קידום.

## מה לא עבד

- bank6 ב-source-excluded OOF: 40.9854% / 0.764108. תוצאת 43.1586% מהבדיקה
  הטרנסדוקטיבית המהירה לא שרדה את חוזה ה-cross-fitting המחמיר.
- TCN-1 ו-TCN-4 בתוך הבנק הגיעו רק ל-42.4688% ו-42.0340% PB.
- Joint-14 הגיע ל-39.2785%; Continuous עם אותן קבוצות ל-39.7865%.
- all-24 הגיע ל-38.9074%. הוספת streams גורפת פוגעת.
- gate Joint+digit הגיע ל-43.0586% עם locator החדש, פחות מה-gate הקיים; בנוסף
  multistart של רכיב Joint היה BLOCKED בכל חמשת ה-folds.
- שילוב finalist gate + finalist locator הציג interaction שלילי של -0.5700pp.

## Ablation

הסרת Renyi a0.25 נתנה 44.0238%; הסרת rank-1 נתנה 43.9959%; הסרת rank-9 נתנה
43.9392%. לעומת זאת הסרת VE0.75 innovation הורידה ל-42.5910%, הסרת digit Top2
ל-42.8828%, והסרת mass-above ל-43.1384%. בבחירה nested לא התקבלה יציבות 4/5:
שלושה folds בחרו להסיר Renyi ושניים להסיר rank-9. לכן אין הצדקה לבחור בדיעבד
איזה stream למחוק, אבל ברור שמשפחת probability/Renyi מכילה redundancy מדללת.

## החלטה

לשמור את האלגוריתם הנוכחי כ-incumbent. לשמור שני research candidates:

1. L08 Continuous L-SML + ה-gate הקיים — המועמד המבוסס יותר.
2. אותו roster עם שלוש קבוצות קבועות ו-family-equal — מועמד הפישוט המועדף.

הניסוי הבא צריך להיות צר: השוואה קפואה בין שני אלה לבין incumbent, בלי roster
search נוסף, ורצוי על confirmation שלא שימש לבניית ה-Atlas.
