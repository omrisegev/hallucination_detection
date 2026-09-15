# אילו שגיאות הוחמצו — ניתוח תיאורי לאחר בחירת הצירופים

הספירות הן תשובות ProcessBench עם שגיאה ידועה. אותה שאלת מקור יכולה להופיע תחת שני מודלים; אלה אינן ספירות של שאלות עצמאיות. הצעד הנכון הוא השגיאה הראשונה לפי התיוג המקורי. מספרי הצעדים מתחילים ב-1. CLEAN משמעו שה-gate סגר את התשובה.

| תצורה | אותרו בדיוק | ה-gate סגר | נבחר מוקדם מדי | נבחר מאוחר מדי | נוספו מול TCN | אבדו מול TCN |
|---|---:|---:|---:|---:|---:|---:|
| iu__ridge+tcn+noreset | 1340 | 821 | 962 | 1319 | 45 | 39 |
| equal__ridge+bocpd+noreset | 1315 | 821 | 1035 | 1271 | 103 | 122 |
| equal__ridge+tcn+bocpd | 1335 | 821 | 987 | 1299 | 54 | 53 |
| iu__ridge+bocpd+noreset | 1325 | 821 | 1018 | 1278 | 87 | 96 |
| tcn__real | 1334 | 821 | 982 | 1305 | 0 | 0 |
| ridge | 1327 | 821 | 964 | 1330 | 63 | 70 |
| bocpd | 1303 | 821 | 1030 | 1288 | 119 | 150 |


מתוך 4442 תשובות עם שגיאה, כל 32 הצירופים מחמיצים 2853: ב-821 ה-gate סגור וב-2032 הוא פתוח אך אף צירוף אינו בוחר את הצעד הנכון. בתוך קבוצת ה-gate הסגור, לפחות צירוף אחד הצביע נכון לפני הסגירה ב-361 תשובות. איחוד הפגיעות הוא אורקל בין הפסגות הקיימות, לא אלגוריתם זמין ולא חסם על מיזוג חדש.

הדוגמאות הבאות נבחרו לפי הקטגוריה ואז מזהה UID, מתוך GSM8K/Qwen3-4B בלבד לצורך הצגת טקסט. הן ממחישות כשלים ואינן מדגם להערכת האיכות. כל הטקסט הוא מן התשובה המקורית; ייתכנו שגיאות נוספות אחרי השגיאה הראשונה.

## IU_loses_TCN_hit — gsm8k::gsm8k-142

A family of 6 (2 adults and 4 kids) are to divide a watermelon such that each adult gets a slice that is twice as big as that of each kid. What percentage of the watermelon does each adult get?

השגיאה הראשונה המתויגת: צעד 2. gate פתוח.

| שיטה | החלטה סופית | פסגה לפני gate |
|---|---:|---:|
| iu__ridge+tcn+noreset | 4 | 4 |
| equal__ridge+bocpd+noreset | 2 | 2 |
| equal__ridge+tcn+bocpd | 2 | 2 |
| iu__ridge+bocpd+noreset | 2 | 2 |
| tcn__real | 2 | 2 |
| ridge | 2 | 2 |
| bocpd | 2 | 2 |


1. Let's break down the problem step by step. First, let the size of each kid's slice be 'x'. Since each adult's slice is twice as big as that of each kid, the size of each adult's slice will be '2x'.

2. There are 2 adults and 4 kids, so the total number of slices is 2 + 4 = 6. The total watermelon can be represented as 6x (since we're considering the sum of all slices).

3. Each adult gets two slices of size '2x', which is 2 * 2x = 4x. To find the percentage of the watermelon that each adult gets, we need to calculate (4x / 6x) * 100.

4. Now, let's simplify the expression: (4x / 6x) * 100 = (4/6) * 100 = (2/3) * 100 = 66.67%. So, each adult gets approximately 66.67% of the watermelon. \boxed{66.67%}

## all32_miss_gate_open — gsm8k::gsm8k-121

Bill starts on the 3rd floor. He rides the elevator up to the floor that's equal to 4 times his starting floor plus 6. What floor is Bill on now?

השגיאה הראשונה המתויגת: צעד 6. gate פתוח.

| שיטה | החלטה סופית | פסגה לפני gate |
|---|---:|---:|
| iu__ridge+tcn+noreset | 1 | 1 |
| equal__ridge+bocpd+noreset | 1 | 1 |
| equal__ridge+tcn+bocpd | 1 | 1 |
| iu__ridge+bocpd+noreset | 1 | 1 |
| tcn__real | 1 | 1 |
| ridge | 1 | 1 |
| bocpd | 1 | 1 |


1. To determine the floor Bill is on after riding the elevator, we need to follow these steps: First, identify Bill's starting floor. Second, calculate the floor he goes to using the given formula. Third, add this floor to his starting floor to find the final floor.

2. Let's start with Bill's starting floor, which is the 3rd floor. According to the problem, he rides the elevator up to the floor that's equal to 4 times his starting floor plus 6. We can express this mathematically as: \[ 4 \times \text{starting floor} + 6 \]

3. Substituting the starting floor (which is 3) into the formula, we get: \[ 4 \times 3 + 6 \]

4. Now, perform the multiplication: \[ 4 \times 3 = 12 \]

5. Then add 6 to the result: \[ 12 + 6 = 18 \]

6. So, Bill rides the elevator up to the 18th floor. Since he started on the 3rd floor, we add the floor he goes up to his starting floor to find the final floor: \[ 3 + 18 = 21 \]

7. Therefore, Bill is on the \(\boxed{21}\)st floor now.

## gate_closes_correct_peak — gsm8k::gsm8k-290

Zack's locker is half as big as Timothy's locker. Peter's locker is 1/4 as big as Zack's locker. If Peter's locker is 5 cubic inches, how big is Timothy's locker in cubic inches?

השגיאה הראשונה המתויגת: צעד 1. gate סגור.

| שיטה | החלטה סופית | פסגה לפני gate |
|---|---:|---:|
| iu__ridge+tcn+noreset | CLEAN | 1 |
| equal__ridge+bocpd+noreset | CLEAN | 1 |
| equal__ridge+tcn+bocpd | CLEAN | 1 |
| iu__ridge+bocpd+noreset | CLEAN | 1 |
| tcn__real | CLEAN | 1 |
| ridge | CLEAN | 1 |
| bocpd | CLEAN | 1 |


1. If Peter's locker is 1/4 as big as Zack's locker, and Peter's locker is 5 cubic inches, then we can find out how big Zack's locker is by dividing the volume of Peter's locker by 1/4. So, if Peter's locker is 5 cubic inches, then Zack's locker (which is twice as big) would be: \[ \text{Volume of Zack's locker} = \frac{5}{1/4} = 5 \times 4 = 20 \text{ cubic inches} \]

2. Since Zack's locker is half the size of Timothy's locker, we can find the volume of Timothy's locker by multiplying the volume of Zack's locker by 2. Therefore, Timothy's locker is: \[ \text{Volume of Timothy's locker} = 20 \times 2 = 40 \text{ cubic inches} \]

## within_leader_loses_TCN_hit — gsm8k::gsm8k-75

Each person in a certain household consumes 0.2 kg of rice every meal. Supposing 5 members of the household eat rice every lunch and dinner, how many weeks will a 42 kg bag of rice last?

השגיאה הראשונה המתויגת: צעד 2. gate פתוח.

| שיטה | החלטה סופית | פסגה לפני gate |
|---|---:|---:|
| iu__ridge+tcn+noreset | 2 | 2 |
| equal__ridge+bocpd+noreset | 1 | 1 |
| equal__ridge+tcn+bocpd | 2 | 2 |
| iu__ridge+bocpd+noreset | 2 | 2 |
| tcn__real | 2 | 2 |
| ridge | 2 | 2 |
| bocpd | 1 | 1 |


1. To determine how many weeks a 42 kg bag of rice will last for the family of 5 that consumes 0.2 kg of rice per meal, we need to calculate how much rice is consumed in a week and then divide the total amount of rice by the weekly consumption.

2. First, calculate the daily consumption. Each person consumes 0.2 kg of rice per meal, and there are 5 people in the household. So, the daily consumption is: \[0.2 \text{ kg/person} \times 5 \text{ persons} = 1 \text{ kg}\]

3. Second, determine the weekly requirement. Since the household eats rice for lunch and dinner, they consume rice twice a day. Therefore, the weekly requirement is: \[1 \text{ kg/day} \times 2 \text{ days/week} = 2 \text{ kg/week}\]

4. Finally, calculate how many weeks the bag will last. Now, we divide the total amount of rice by the weekly consumption to find out how long it will last: \[42 \text{ kg} \div 2 \text{ kg/week} = 21 \text{ weeks}\]

5. Therefore, a 42 kg bag of rice will last 21 weeks for the family of 5 that consumes 0.2 kg of rice per meal.

[כל מזהי ההחמצות והתחזיות](PB_ERROR_LEDGER.csv) · [ניתוח וספירות מלאים](ERROR_ANALYSIS.json) · [דוח הצירופים](REPORT.html).
