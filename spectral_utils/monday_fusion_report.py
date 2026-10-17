"""Internal experiment status and claims ledger; no automatic deck promotion."""
import json
from pathlib import Path
import pandas as pd
from spectral_utils.monday_fusion_protocol import WORK,RUNS,dump,sha

def report(experiment):
    out=RUNS[experiment]
    def read(p): return json.loads(p.read_text(encoding='utf8')) if p.exists() else {}
    audit=read(WORK/'INPUT_AUDIT.json'); replay=read(WORK/'REPLAY_GATE.json')
    status=read(out/'RUN_STATUS.json'); figures=read(out/'FIGURE_STATUS.json')
    lines=['# דוח ניסוי — '+experiment,'',
           '**מצב:** '+str(status.get('status','NOT_STARTED'))+'. כל הממצאים דורשים ביקורת עצמאית של קלוד לפני המצגת.','',
           'זהו ניתוח פיתוח על נתונים שכבר נחשפו. כלל ההחלטה הסופי ממתין לבחירת עמרי; לא בוצעה הערכה חיצונית נוספת.','',
           '| טענה | מספר | קובץ מקור | N שנבדק | מצב ביקורת |',
           '|---|---:|---|---:|---|']
    claims=[]
    if audit.get('status')=='PASS':
        claims.append({'claim':'Input identities and source folds passed','number':audit['folds_checked'],'source':str(WORK/'INPUT_AUDIT.json'),
                       'N':audit['population']['answers'],'review':'AUTOMATED_PASS; CLAUDE_PENDING'})
    for key in ['step457','step462']:
        checks=replay.get(key,{})
        if checks:
            claims.append({'claim':key+' maximum replay difference','number':max(d['max_abs_difference'] for d in checks.values()),
                           'source':str(WORK/'REPLAY_GATE.json'),'N':len(checks),'review':'AUTOMATED_PASS; CLAUDE_PENDING'})
    if replay.get('status')=='PASS':
        claims.append({'claim':'Exact zero-removal input comparisons','number':len(replay['zero_removal']),
                       'source':str(WORK/'REPLAY_GATE.json'),'N':len(replay['zero_removal']),'review':'AUTOMATED_PASS; CLAUDE_PENDING'})
    for c in claims: lines.append(f"| {c['claim']} | {c['number']} | `{c['source']}` | {c['N']} | {c['review']} |")
    lines+=['','הסבילות בשחזור הציונים היא `1e-9`; שער אפס ההסרות דורש שוויון מדויק של הקלטים.','',
            '## תוצאות וכיסוי','']
    if(out/'METRICS.csv').exists():
        m=pd.read_csv(out/'METRICS.csv')
        lines += [f"נשמרו {len(m):,} שורות מדדים ב־[METRICS.csv](METRICS.csv). שורות כשל: {int((m.coverage==0).sum()):,}.",
                  'הטבלה מפרידה PRMScore, AUC בתוך תשובה, דיוק השגיאה הראשונה, F1 רשמי ושיעור דגלים בתשובות תקינות.',
                  'ב־ProcessBench אין תוויות מלאות לכל הצעדים; דיוק האומדנים והתפלגות מספר השגיאות המלאה אינם מחושבים שם.','']
    else: lines+=['ההרצה טרם יצרה טבלת אוכלוסייה מלאה. אין מסקנת איכות.','']
    selection=read(out/'SELECTION.json')
    winner=selection.get('winner')
    if winner:
        claim={'claim':'Selected development content recipe: '+winner['variant']+'/'+winner['recipe']+'; eight-bank mean advantage over matched EQ',
               'number':winner['mean_advantage'],'source':str(out/'SELECTION.json'),'N':8,'review':'CLAUDE_PENDING; NOT_DECK_APPROVED'}
        claims.append(claim)
        lines += [f"Development selection: `{winner['variant']}/{winner['recipe']}`; equal-weight eight-bank matched advantage = {winner['mean_advantage']:.6f}. Full cell coverage is required; this is a development comparison awaiting review.",'']
        candidates=selection.get('candidates',[])
        loss=sum(int(v.get('matched_eq_significant_losses',0)) for v in candidates)
        excluded=sum(not v.get('complete_8_banks_9_cells',False) for v in candidates)
        lines += [f"Across all registered weighted candidates: {loss} bank-level significant-loss findings (unadjusted paired 95%); {excluded} candidates excluded from winning because of incomplete coverage. See every absolute cell score in METRICS.csv and all intervals in CONTRASTS.csv.",'']
        lines += ['| Development claim | Number | Source | N | Review |','|---|---:|---|---:|---|',
                  f"| {claim['claim']} | {claim['number']} | SELECTION.json | 8 banks | CLAUDE_PENDING |",'']
    if selection:
        lines+=['בחירת הפיתוח הזמנית נמצאת ב־[SELECTION.json](SELECTION.json). היא אינה בחירה מאושרת למצגת.','',
                'לצד כלל הממוצע על שמונת הבנקים נשמר גם כלל Step 457: אין בנק עם הפסד מובהק, ולפחות חמישה ניצחונות מובהקים; אחרת חוזרים לממוצע המסונן.','']
    direct=out/'DIRECT_INTERVENTION_CONTRASTS_REVIEW.csv'
    if direct.exists():
        lines+=['## השוואות ישירות ותיקון הביקורת','',
                'התוצאות המקוריות נשמרו בתיקיית run_20261003. הבקרות האקראיות אינן מועמדות לבחירה; הממוצע הפשוט אינו מוכנס עם יתרון אפס מול נקודת ייחוס אחרת. ראו REVIEW_CORRECTIONS_V2.json.',
                'הטבלה DIRECT_INTERVENTION_CONTRASTS_REVIEW.csv משווה הסרה למקור ושינוי קלט לקלט המקורי באותו מתכון. זהו ניתוח חקרני שהושלם לאחר הביקורת, ולא רישום מראש בדיעבד.','']
        if experiment=='estimator_input_v1':
            contrasts=pd.read_csv(direct)
            q=contrasts[(contrasts.experiment=='input')&(contrasts.recipe=='HEM_MOM')&
                        (contrasts.comparison=='both_changes')&(contrasts.benchmark=='pb')].iloc[0]
            claim={'claim':'Original-bank HEM_MOM: channels/cell minus groups/answer, PB argmax first-error accuracy',
                   'number':float(q.delta),'source':str(direct),'N':int(q.n_answers),
                   'review':'POST_REVIEW_EXPLORATORY; OMRI_REQUESTED_HIGHLIGHT; CLAUDE_PENDING'}
            claims.append(claim)
            lines += [f"עמרי ביקש להבליט: HEM בתוך הקבוצות ו־MoM ביניהן, ללא הסרת LEVEL — שינוי קלט האומדנים משפר את איתור השגיאה הראשונה ב־ProcessBench ב־{q.delta*100:.3f} נקודות אחוז. רווח 95%: [{q.lo95*100:.3f}, {q.hi95*100:.3f}]; רווח מתוקן ל־40 השוואות באותה משפחה ומדד: [{q.lo_bonferroni*100:.3f}, {q.hi_bonferroni*100:.3f}]. N={int(q.n_answers)} תשובות שגויות, ממוצע שווה ל־8 בנקים ול־8 תאים.",
                      'אין להסיק מכך שיפור כולל: בכלל q80 המקורי F1 ירד בכ־0.529 נקודות אחוז ושיעור הדגלים בתשובות תקינות עלה בכ־0.441. ההבחנה היא בין צעד בעל ציון מרבי לבין הצעד הראשון שעבר סף. זו השוואת אותו מתכון מול קלטו המקורי, לא הוכחת עדיפות על מתחרים.','']
    lines+=['## תרשימים','']
    for f in sorted((out/'figures').glob('*.png')):
        lines += [f'![{f.stem}](figures/{f.name})','',f'[SVG](figures/{f.with_suffix(".svg").name})','']
    for f in figures.get('referenced_figures',[]):
        lines += [f'![Same locked final recipe: {Path(f).stem}]({f})','',f'[SVG]({str(Path(f).with_suffix(".svg")).replace(chr(92),"/")})','']
    lines+=['## פתוח','']
    for f,reason in figures.get('open',{}).items(): lines.append(f'- {f}: {reason}')
    lines+=['- ביקורת קלוד: שחזור עצמאי של מספרים, בדיקת אוכלוסייה וערבוב תוויות.',
            '- בחירת כלל ההחלטה בידי עמרי לאחר השוואת שלושת הכללים.',
            '- מתכון משולב ובקרותיו על האוכלוסייה המלאה; ניתוח מתחרים על אותו מתכון לאחר סגירתו.','',
            '## פירוש זהיר','',
            'LR הוא ייחוס לינארי מפוקח, לא תקרה מתמטית. תוצאות האימון הן אבחון נפרד. top-2 ההיסטורי נבחר בדיעבד; top-1 ו־top-3 נשמרים לצדו.',
            'גרסת השיפוע המוצלב בקבוצות היא הרחבה חדשה ומשנה גם את משקול התוכן. הבקרות המיקומיות אינן פירוק סיבתי.',
            'SML מספק דירוג אמינות; אין כאן אומדני רגישות וסגוליות שלו. בשלוש קבוצות מתועדת הגנת המשקלים הקיימת.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    pd.DataFrame(claims).to_csv(out/'CLAIMS.csv',index=False)
    evidence=[p for p in [out/'PROTOCOL.json',out/'INPUT_MANIFEST.json',out/'METRICS.csv',out/'CONTRASTS.csv',out/'DIAGNOSTICS.json',out/'REVIEW_CORRECTIONS_V2.json',direct,WORK/'INPUT_AUDIT.json',WORK/'REPLAY_GATE.json'] if p.exists()]
    dump(out/'REVIEW_PACKET.json',{'review_status':'CLAUDE_PENDING','claims':claims,'inputs':{str(p):sha(p) for p in evidence},
                                  'requests':['Recompute headline metrics from scores/fit parameters','Check every registered bank, cell and failure denominator','Independently run shuffled-label and same-length nulls'],
                                  'deck_eligible':False})
