"""Build a standalone advisor presentation from saved, audited display data."""
from pathlib import Path
import os
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mpl_cache'))
import base64, csv, hashlib, html, io, json, re, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

D = json.loads((HERE / 'DATA.json').read_text())
U = json.loads((HERE / 'PRM_UPDATE.json').read_text())
SOURCE = ROOT / 'results/advisor_page_v13_v14_2026-10-12/published_v16_2026-10-13.html'
if not SOURCE.exists():
    SOURCE = ROOT / 'docs/reviews/advisor_v16_walkthrough_20261013/source_v16_2cf1209c.html'
source_html = SOURCE.read_text()
assert hashlib.sha1(b'blob '+str(SOURCE.stat().st_size).encode()+b'\0'+SOURCE.read_bytes()).hexdigest() == '9aaab92066600edb593b7050cd7f042acf7f1719'
for part in ['boosting','generalization']:
    assert U[part]['fusion']=='LSML_CONT' and U[part]['same_checkpoint'] is True
    assert U[part]['status'] in ['pending','complete','failed']
    if U[part]['status']=='complete':
        assert U[part]['rows'] and U[part]['source']
        for row in U[part]['rows']:
            assert row['checkpoint'] and row['rule'] and row['source']
            for key in ['prmscore','auc'] if part=='boosting' else ['prm_prmscore','lsml_prmscore','prm_auc','lsml_auc']:
                assert isinstance(row[key],(int,float)) and np.isfinite(row[key])
            if part=='generalization':
                assert row['same_checkpoint'] is True and row['fixed_features'] is True

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':'#203044','text.color':'#203044',
    'axes.edgecolor':'#cbd5df','xtick.color':'#526173','ytick.color':'#526173',
    'svg.fonttype':'none','svg.hashsalt':'advisor-v17','figure.facecolor':'white',
    'axes.facecolor':'white','savefig.facecolor':'white'})
COL={'ALL_EQ':'#8d99a5','DS_FILTER_EQ':'#3b8c88','LSML_CONT':'#1766a1','IU_PCR':'#6f58a5','LR':'#c77b21','PRM':'#d4543c'}
NAMES={'ALL_EQ':'Plain average','DS_FILTER_EQ':'DS-filtered average','LSML_CONT':'L-SML','IU_PCR':'IU-PCR','LR':'LR · supervised OOF','PRM':'PRM-7B · native 0.5'}
FIGS={}

def e(v): return html.escape(str(v),quote=True)
def csv_rows(path): return list(csv.DictReader(Path(path).open()))
def original_section(s):
    return re.search(r'<section id="'+re.escape(s)+r'">(.*?)</section>',source_html,re.S)[1]
def original_figure(s,i=0):
    return re.findall(r'<figure>.*?</figure>',original_section(s),re.S)[i]
def chart(name,fig,caption,alt=None,tall=False):
    stream=io.StringIO();fig.savefig(stream,format='svg',bbox_inches='tight',metadata={'Date':None})
    svg=stream.getvalue();svg=svg[svg.index('<svg'):]
    svg=re.sub(r'<svg\b', '<svg role="img" aria-label="'+e(alt or caption)+'"',svg,count=1)
    (HERE/'figures').mkdir(exist_ok=True)
    (HERE/'figures'/f'{name}.svg').write_text(svg)
    fig.savefig(HERE/'figures'/f'{name}.png',dpi=150,bbox_inches='tight')
    plt.close(fig);FIGS[name]=caption
    return f'<figure class="chart {"tall" if tall else ""}">{svg}<figcaption>{e(caption)}</figcaption></figure>'
def table(headers,rows,classes=''):
    return '<div class="table-wrap"><table class="'+classes+'"><thead><tr>'+''.join('<th>'+e(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def colored(method,label=None): return f'<span class="dot" style="background:{COL[method]}"></span>{e(label or NAMES[method])}'
def callout(title,text,kind=''):
    return f'<div class="callout {kind}"><strong>{title}</strong><p>{text}</p></div>'
def bullets(items): return '<ul class="points">'+''.join('<li>'+x+'</li>' for x in items)+'</ul>'
def source_footer(paths):
    return '<div class="source"><b>Evidence</b> '+e(paths)+'</div>'

slides=[]
def slide(number,title,minutes,old,body,notes,sources,question,limit,appendix=False):
    ident=f'a{number}' if appendix else f's{number:02}'
    record={'id':ident,'number':number,'title':title,'minutes':minutes,'old':old,'notes':notes,'sources':sources,'question':question,'limit':limit,'appendix':appendix}
    slides.append(record)
    notehtml=f'<aside class="speaker" lang="he" dir="rtl"><h3>הערות דובר</h3><p>{notes}</p><p><b>מה הגרף מודד ואיך קוראים אותו:</b> {question}</p><p><b>גבול המסקנה:</b> {limit}</p><p><b>מיפוי:</b> {e(old)}</p></aside>'
    return f'<section class="slide {"appendix-slide" if appendix else ""}" id="{ident}" aria-labelledby="{ident}-title"><div class="eyebrow">{"APPENDIX" if appendix else "ADVISOR MEETING · 15 OCTOBER 2026"}<span>{"" if appendix else str(minutes)+" MIN · "}{e(old)}</span></div><h1 id="{ident}-title">{e(title)}</h1>{body}{notehtml}{source_footer(sources)}</section>'

M=D['bank_metrics']; S=M['N16']; P=D['prm_native']; ex=D['example']; parts=[]
example_steps=''
for i,text in enumerate(ex['steps'],1):
    content=e(re.sub(r'^\d+\.\s*','',text).replace('$',''))
    example_steps+=f'<li class="{"error" if i in ex["error_steps"] else "valid"}"><span class="step-label">Step {i} · {"annotated error" if i in ex["error_steps"] else "valid"}</span>{content}</li>'
prompt=callout('Actual PRMBench question',e(ex['question']))
pipeline='<div class="pipeline">'+''.join('<span>'+v+'</span>' for v in ['Token distributions','Step signals','L-SML / IU-PCR','Risk ranking','Error flags'])+'</div>'
parts.append(slide(1,'What are we trying to detect?',2,'v16 §2',
    '<p class="lead">Find erroneous reasoning steps in a given answer.</p><div class="columns"><div>'+prompt+callout('What is a PRM?','A process reward model scores the validity of each reasoning step. A supervised reward head reads the model’s backbone states to produce that score.')+pipeline+'</div><div><ol class="steps">'+example_steps+'</ol></div></div><p class="caption">NCL example: assuming 29 cents and confirming it is circular reasoning. The final answer can still be correct.</p>',
    'פותחים בדוגמה אמיתית, לפני שמציגים אלגוריתמים. השגיאה היא בהנמקה, לא בהכרח בתשובה הסופית. להסביר בשני משפטים PRM וההפרדה בין head ל־backbone. אצלנו קוראים סיגנלים מתוך מעבר על התשובה הנתונה ומחברים אותם ללא תוויות.',
    'controls/DATASET_EXAMPLES.json; v16 §2; example '+ex['idx'],
    'זהו prompt ותוכן תשובה אמיתיים מה־benchmark. הצבע מציין את תוויות הצעדים, לא תחזית של השיטה שלנו.',
    'הדוגמה נבחרה לפי סדר קבוע של הנתונים ולא לפי הצלחה של שיטה. זיהוי צעד שגוי אינו זהה לבדיקה אם התשובה הסופית נכונה.'))

fig,ax=plt.subplots(figsize=(10,2.35))
x=np.arange(1,len(ex['steps'])+1);ax.bar(x,ex['lsml_risk'],color=[COL['PRM'] if i in ex['error_steps'] else '#9db5c7' for i in x])
ax.axhline(ex['q80_cut'],color=COL['LSML_CONT'],ls='--',label=f'N16 global Q80 cut = {ex["q80_cut"]:.3f}')
ax.axhline(0,color='#bbc4ce',lw=.7);ax.set(xticks=x,xlabel='Step number in the same real example',ylabel='L-SML risk\n(answer z-score)');ax.legend(loc='upper right',fontsize=9)
example_chart=chart('example_risk',fig,'Actual saved L-SML scores for the example on slide 1. Red bars: annotated errors (3–5). This is one example, not a benchmark summary.')
parts.append(slide(2,'Two questions, two metrics',3,'v16 §§2–3',
    '<div class="metric-cards"><div><span class="tag">RANKING</span><h2>Within-answer AUC</h2><p>Does an erroneous step rank above a valid step <b>inside the same answer</b>?</p><p>No threshold. Average over 6,030 eligible answers.</p></div><div><span class="tag">OFFICIAL DECISION METRIC</span><h2>PRMScore</h2><p class="formula">½ (F1<sub>valid</sub> + F1<sub>error</sub>)</p><p>Score the binary step flags on 6,211 noncontrol answers. Includes MS; excludes 758 synthetic clean controls.</p></div></div>'+example_chart+callout('Top1-on-error','Is the highest-risk step annotated as erroneous? Evaluated on 6,035 answers with at least one in-range error.')+'<p class="caption">Overall PRMScore pools steps before computing the two F1 scores. It is not the average of the nine error-type scores. MS uses valid-step F1 alone.</p>',
    'כאן מבססים את שתי השאלות שילוו את המצגת: האם הדירוג נכון, ואילו צעדים בחרנו לסמן. משתמשים שוב באותה תשובה. מראים שהסף הוא קו נפרד מהדירוג. Top1 היא הצלחה של הבחירה בצעד אחד, לא דיוק התשובה כולה ולא first-error accuracy.',
    'Official evaluator: spectral_utils/prmbench.py; DATA.json example; N16 saved S__LSML_CONT; population.npz',
    'גובה כל עמודה הוא סיכון מנורמל של צעד. האדום מסמן תווית אמת; הקו מראה את הסף הגלובלי שנבחר על כלל הציונים המנורמלים.',
    'מספר התשובות שונה בין המדדים בשל תנאי זכאות שונים. הגרף אינו סימולציה ואינו טענה שהשיטה מזהה היטב כל דוגמה.'))

rows=[[colored(m),f'{S[m]["prmscore"]:.2f}',f'{S[m]["auc"]:.4f}',f'{S[m]["top1"]:.2f}','Q80'] for m in D['methods']]
ci=next(r for r in D['paired_contrasts'] if r['subset']=='ALL_NONCONTROL' and r['left']=='LSML_CONT' and r['right']=='ALL_EQ' and r['metric']=='PRMScore')
parts.append(slide(3,'Back to last week’s comparison',2,'v16 §0',
    '<p class="lead">Same feature comparison, now showing both the official score and ranking.</p>'+table(['N16 · Qwen3-8B features','PRMScore ×100','Within-answer AUC','Top1-on-error %','Flag rule'],rows)+
    callout('L-SML vs plain average',f'+{ci["difference"]*100:.2f} PRMScore points · paired 95% CI [{ci["ci_low"]*100:.2f}, {ci["ci_high"]*100:.2f}]')+
    '<p class="caption">All inputs use the corrected prompt + mask1. LR is the fresh source-group OOF diagnostic; both LR metrics use that same saved prediction. The historical LR TOP1 score is not used as a Q80 score.</p><a class="jump" href="#a1">Full methods × banks comparison → appendix</a>',
    'מתחילים בטבלה המוכרת משבוע שעבר, אחרי שהמנחים כבר מבינים את המדדים. אומרים רק משפט על התיקון. התוצאה של LR כאן היא ה־OOF החדש; לא מערבבים את ה־AUC ההיסטורי עם PRMScore מהתאמה אחרת. להצביע על המרווח הקטן בין שיטות ה־fusion ועל השאלה מה נותנת הרחבת הבנק.',
    'controls/ALL_BANK_TOP1_AUC.csv; saved bank_N16.npz; lr/OOF_N16.npz + lr/METRICS.csv; PAIRED_DECISION_ANALYSIS.json',
    'כל שורה היא שיטה על אותם פיצ׳רים ואותה אוכלוסייה. PRMScore משתמש ב־Q80 בכל השורות; עמודות הדירוג אינן משתמשות בסף.',
    'רווח הסמך הוא paired source-group bootstrap פנימי על נתוני פיתוח. LR משתמש בתוויות להתאמה, בשונה מה־fusion ללא תוויות.'))

fig,axs=plt.subplots(1,2,figsize=(11,3.6))
for m in D['methods']:
    for ax,key in zip(axs,['prmscore','auc']):
        ax.plot(range(5),[M[b][m][key] for b in D['banks']],'-o',color=COL[m],label=NAMES[m],lw=2,ms=5)
for ax in axs:
    ax.set_xticks(range(5),['N14','N16\n+2','N25\n+9','N43\n+18','N56\n+13']);ax.set_xlabel('Strictly nested digit banks · channels only added');ax.grid(axis='y',alpha=.18)
axs[0].set_ylabel('Official PRMScore ×100 · Q80');axs[1].set_ylabel('Within-answer AUC');axs[0].set_ylim(61.5,72.5);axs[1].set_ylim(.735,.835)
axs[0].set_title('Binary error decisions');axs[1].set_title('Ranking inside each answer');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=9,bbox_to_anchor=(.5,1.05));fig.tight_layout(rect=(0,0,1,.85))
bank_chart=chart('nested_banks',fig,'N14 ⊂ N16 ⊂ N25 ⊂ N43 ⊂ N56; all three digit channels retained. AUC: 6,030 eligible answers. PRMScore: 6,211 noncontrol answers, Q80. Both metrics use the same saved prediction; LR is fresh source-group OOF. Point estimates, no CI.')
lr_delta=M['N43']['LR']['auc']-M['N25']['LR']['auc']
parts.append(slide(4,'Adding features: what improves, and what does not?',3,'v16 §§0, 6a',bank_chart+
    bullets(['Label-free fusion does not improve monotonically as features are added.','LR improves in both AUC and Q80 PRMScore across this digit-bank chain.','L-SML often retains a larger gain over the plain average than IU-PCR; the supervised ranking gap is larger.'])+
    callout('The N25 → N43 question',f'18 channels added. LR AUC: +{lr_delta*100:.2f} points; L-SML: {(M["N43"]["LSML_CONT"]["auc"]-M["N25"]["LSML_CONT"]["auc"])*100:.2f}; IU-PCR: {(M["N43"]["IU_PCR"]["auc"]-M["N25"]["IU_PCR"]["auc"])*100:.2f}. Which additions help supervised LR and hurt label-free fusion?','research'),
    'מראים רק בנקים עם digits ורק הרחבה על ידי הוספה. הציפייה למונוטוניות נעשית סבירה יותר, אך אינה מובטחת. מבהירים שהמרווח הגדל מעל הממוצע יכול לשקף את הירידה שלו, ולא שיפור של L-SML. מתמקדים בקפיצה מ־25 ל־43 ומשאירים שאלת מחקר מפורשת על הפיצ׳רים שנוספו.',
    'BANKS.json; controls/ALL_BANK_TOP1_AUC.csv; bank_N14/16/25/43/56.npz; lr/OOF_N*.npz; DATA.json transitions',
    'משמאל PRMScore תחת אותו כלל Q80; מימין AUC. כל צבע הוא אותה שיטה בשני הפאנלים. התוספות במספרים על ציר X הן מספר הערוצים שנוספו.',
    'הבדלי הבנקים אינם ablation של פיצ׳ר יחיד, ולא מאפשרים לייחס סיבתיות לערוץ מסוים. שני מדדי LR מונוטוניים בשרשרת הזו, לא הבטחה לכל הרחבת בנק עתידית.'))

fourb=D['external_sources']['qwen3_4b_vs_8b__official_prmscore_4b__OFFICIAL_PRMSCORE_N16.csv']
rank4=D['external_sources']['qwen3_4b_vs_8b__prmbench_cont.csv']
scores4={m:float(next(r for r in fourb if r['method']==m and r['rule']=='R0')['overall'])*100 for m in ['LSML_CONT','IU_PCR']}
auc4={m:float(next(r for r in rank4 if r['bank']=='N16' and r['scorer']==m)['prmb_within_auc_4b']) for m in ['LSML_CONT','IU_PCR']}
lb=[(r['model']+' · published',float(r['overall']),'#a9b3bc') for r in D['paper'] if r['model'] in ['o1-mini','Gemini-2.0-thinking-exp-1219','GPT-4o','Gemini-2.0-flash-exp','ReasonEval-34B']]
lb += [(NAMES[m]+' · Qwen3-8B',S[m]['prmscore'],COL[m]) for m in ['LSML_CONT','IU_PCR','LR']]
lb += [('L-SML · Qwen3-4B',scores4['LSML_CONT'],'#8ab3d3'),('IU-PCR · Qwen3-4B',scores4['IU_PCR'],'#b5a6d0'),('PRM-7B · our run, native 0.5',P['prmscore'],COL['PRM'])]
lb.sort(key=lambda x:x[1],reverse=True)
fig,ax=plt.subplots(figsize=(10,4));ax.barh(range(len(lb)),[r[1] for r in lb],color=[r[2] for r in lb]);ax.set_yticks(range(len(lb)),[r[0] for r in lb],fontsize=9);ax.invert_yaxis();ax.set(xlim=(0,76),xlabel='Overall PRMScore ×100');ax.grid(axis='x',alpha=.13)
for j,r in enumerate(lb):ax.text(r[1]+.4,j,f'{r[1]:.2f}',va='center',fontsize=10)
fig.tight_layout();leaderboard=chart('leaderboard',fig,'Our reruns: same 6,211 noncontrol answers and evaluator port. Our methods/LR: Q80. PRM: native reward <0.5. Published paper rows retain their authors’ protocols; o1-mini uses N=394. This is a selected comparison, not a current live leaderboard.')
parts.append(slide(5,'Where are we on PRMBench?',2,'v16 §§3, 6c',leaderboard+
    '<div class="stat-strip"><div><b>65.38 vs 65.46</b><span>L-SML 8B vs native PRM</span></div><div><b>65.00 / 65.01</b><span>L-SML / IU-PCR with 4B features</span></div><div><b>Close point estimates</b><span>4B–8B equivalence was not established</span></div></div>'+callout('Question','How can the official scores be so close when AUC and Top1 differ?'),
    'חוזרים למטריקה הרשמית כדי למקם את העבודה. להבחין בין ההרצות שלנו לבין שורות המאמר. אפשר לראות כאן גם את LR תחת Q80. 4B ו־8B קרובים בתוצאות השיטות שלנו; אין כאן בדיקה שמוכיחה שקילות. ההמשך מפרק את הפרופיל לפי שגיאות ואחר כך את כלל ההחלטה.',
    'report/paper_table3.csv; v16 §3; DATA.json bank_metrics; official_prmscore_4b N16 CSV; prmbench_cont CSV',
    'אורך העמודה הוא Overall PRMScore. שורות המאמר באפור. הצבעים מזהים את ההרצות הישירות והשיטות שלנו. הציר מתחיל באפס.',
    'פרוטוקולי המאמר אינם reruns תואמים. הניסוח הוא השוואה לטבלת המאמר, לא טענה למיקום עדכני בלוח ציבורי.'))

axes=D['gallery']['axes'];keys=[a['key'] for a in axes];codes=[a['code'] for a in axes]
fig,axs=plt.subplots(1,2,figsize=(11,3.3),gridspec_kw={'width_ratios':[5,1]})
for m in ['LSML_CONT','IU_PCR','LR','PRM']:
    r=P if m=='PRM' else S[m]
    axs[0].plot(range(8),[r['by_type'][k] for k in keys],'-o',label=NAMES[m],color=COL[m],lw=2)
    axs[1].scatter([0],[r['by_type']['multi_solutions']],color=COL[m],s=55,zorder=3)
axs[0].set(xticks=range(8),xticklabels=codes,ylim=(35,100),ylabel='Per-type PRMScore ×100');axs[0].grid(axis='y',alpha=.2);axs[0].legend(fontsize=8,ncol=2,loc='upper right')
axs[1].set(xticks=[0],xticklabels=['MS'],ylim=(35,100),title='Valid-step F1');axs[1].grid(axis='y',alpha=.2);fig.tight_layout()
profile=chart('error_profile',fig,'N16; 6,211 noncontrol answers split by type (MS: 160). Our methods and LR: Q80. PRM: native 0.5. Same evaluator and rows. Eight error types use macro-F1; MS is shown separately with valid-step F1 only.')
legend='<div class="taxonomy">'+''.join(f'<span><b>{a["code"]}</b> {e(a["name"])}</span>' for a in axes)+'<span><b>MS</b> Multiple valid solutions</span></div>'
parts.append(slide(6,'Which errors do we detect?',3,'v16 §4a; LR saved OOF evaluation',profile+legend+
    bullets(['L-SML leads native PRM on NR and NCL; PRM leads on CI and DR.','Q80 also flags steps in valid MS answers; false flags reduce MS valid-step F1.','The LR profile is computed under Q80 here. Its Top1 complementarity is a separate diagnostic in the appendix.'])+'<a class="jump" href="#a3">Four-way Top1 counts → appendix</a>',
    'לא להסתפק בממוצע: להראות היכן אנחנו חזקים והיכן PRM חזק. ניתוח LR כאן מחושב באותה מטריקה ובאותו Q80 כמו שלנו. הטענה הישנה על סוגי השגיאות שבהם LR ואנחנו שונים הייתה Top1; לא להעתיק אותה אוטומטית לגרף PRMScore. MS הוא דיון ב־false positives על מסלולים תקינים.',
    'Official evaluator; DATA.json by_type for saved N16 fusion, fresh LR OOF and native PRM; v16 §4a',
    'כל נקודה היא ציון pooled בתוך סוג השגיאה. הפאנל הקטן מבודד את MS, משום שהמדד שם הוא F1 של צעדים תקינים בלבד. אין CI בגרף הזה.',
    'הבדלי הפרופיל כוללים גם את השפעת כללי ההחלטה: Q80 מול סף PRM המקורי. לא מסיקים מנגנון מתוך הפרופיל בלבד.'))

g=D['gallery']; features=sorted(g['features'],key=lambda f:f['family']); names=[f['id'] for f in features]+['STEP_INDEX']
vals=np.array([[g['metrics']['raw'][k][n]['value'] for k in keys] for n in names])
fig,ax=plt.subplots(figsize=(10,12.8));im=ax.imshow(vals,vmin=.35,vmax=.9,cmap='YlGnBu',aspect='auto')
labels=[f'{f["family"]}  {f["id"]}' for f in features]+['BASELINE  STEP_INDEX']
ax.set_yticks(range(57),labels,fontsize=7.5);ax.set_xticks(range(8),codes);ax.xaxis.tick_top();ax.tick_params(length=0)
for j in range(1,56):
    if features[j]['family']!=features[j-1]['family']:ax.axhline(j-.5,color='white',lw=2.5)
ax.axhline(55.5,color='#c77b21',lw=3)
for j,n in enumerate(names):
    for k in range(8):ax.text(k,j,f'{vals[j,k]:.2f}',ha='center',va='center',fontsize=6.5,color='white' if vals[j,k]>.72 else '#203044')
for n,cols in [('ct7_ve0',[0,1]),('digit_spread',[2]),('STEP_INDEX',[4])]:
    j=names.index(n)
    for k in cols:ax.add_patch(Rectangle((k-.49,j-.49),.98,.98,fill=False,edgecolor='#ff9f1c',lw=2))
fig.colorbar(im,ax=ax,fraction=.022,pad=.02,label='Within-answer AUC');fig.tight_layout()
heatmap=chart('signal_heatmap',fig,'All 56 signals + STEP_INDEX. Within-type error/valid eligible answers; 6,030 in total. Raw within-answer AUC with label-free signs; no threshold. Orange outlines guide the three findings. MS has no error/valid pairs and is omitted.',tall=True)
fig,ax=plt.subplots(figsize=(10,3.4));feature_cols=['#1766a1','#d4543c','#3b8c88','#6f58a5','#c77b21']
for n,c in zip(D['selected_features'],feature_cols):
    ax.plot(range(8),[D['feature_prmscore'][n][k] for k in keys],'-o',label=n,color=c,lw=2)
ax.set(xticks=range(8),xticklabels=codes,ylim=(35,85),ylabel='Per-feature PRMScore ×100');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout()
h_chart=chart('selected_feature_prmscore',fig,'Gallery H: individual-feature PRMScore across eight types (6,051 noncontrol answers; MS excluded). STEP_INDEX is newly derived with the same donor/source-fold Q80 (≥). This differs from pooled development Q80 for the main fused scores. No family fusion or CI is shown.')
families='<p class="caption">'+ ' · '.join(f'{k}: {e(v["en"])}' for k,v in g['familyNames'].items())+'</p>'
parts.append(slide(7,'Which signals explain this profile?',2.5,'v16 §4c; gallery H',
    '<div class="insight-grid"><div><b>NR / NCL</b><p>Concentration and distribution signals stand out.</p></div><div><b>DR</b><p>Digit alternatives help; step position also carries signal.</p></div><div><b>CI</b><p>STEP_INDEX beats every individual channel. Interpret with care.</p></div></div><p class="caption">The full heatmap is intentionally tall. Scroll to inspect every channel; select the PRMScore view for the one additional plot.</p><div class="tabs"><button class="view active" data-panel="heatmap">All 56 channels · AUC</button><button class="view" data-panel="feature-profile">Selected signals · PRMScore (H)</button><button class="zoom" data-zoom="heatmap-panel">Enlarge heatmap</button></div><div id="heatmap-panel" data-view="heatmap">'+heatmap+families+'</div><div data-view="feature-profile" hidden>'+h_chart+legend+'</div>',
    'לא נותנים למנחים לחפש את הסיפור בעצמם. מתחילים משלוש התובנות המסומנות בכתום, ואז מאפשרים לראות את כל 56 הערוצים. מראים רק עוד גרף אחד, H, כדי לחבר את הסיגנלים ל־PRMScore. F ו־G אינם חלק מההצגה. הדגשה נוספת: DC אינו נשען על אותו baseline של מיקום חזק כמו CI.',
    'gallery DATA.json @ codex/advisor-v11-completion; statistics/CHANNEL_BY_TYPE.csv; FEATURE_PRMSCORE.csv (gallery H)',
    'כל תא ב־heatmap הוא AUC של ערוץ יחיד בסוג שגיאה. השורה האחרונה היא מיקום הצעד. ב־H כל עקומה היא PRMScore של ערוץ יחיד תחת threshold שנלמד ללא תוויות מקיפולי donor.',
    'אלו ביצועי ערוצים יחידים, לא תרומה סיבתית ל־fusion. ממוצעי משפחות אינם מוצגים כתוצאת fusion. פרוטוקול הספים ב־H שונה מפרוטוקול ה־Q80 הראשי.'))

curves=D['budget_curves']
def draw_budget(ax):
    for m in ['LSML_CONT','IU_PCR','PRM_NATIVE']:
        r=[r for r in curves if r['method']==m and float(r['actual_budget'])<=.45]
        ax.plot([float(x['actual_budget'])*100 for x in r],[float(x['prm_score'])*100 for x in r],color=COL['PRM'] if m=='PRM_NATIVE' else COL[m],label='PRM-7B threshold sweep' if m=='PRM_NATIVE' else NAMES[m],lw=2)
    ax.scatter([S['LSML_CONT']['flag_rate_all']*100],[S['LSML_CONT']['prmscore']],color=COL['LSML_CONT'],s=70,zorder=4)
    ax.annotate('L-SML Q80',(20,S['LSML_CONT']['prmscore']),xytext=(26,62),arrowprops={'arrowstyle':'->','color':'#526173'},fontsize=9)
    ax.scatter([P['flag_rate_all']*100],[P['prmscore']],color=COL['PRM'],s=70,zorder=4)
    ax.annotate('PRM native 0.5',(P['flag_rate_all']*100,P['prmscore']),xytext=(1,72),arrowprops={'arrowstyle':'->','color':'#526173'},fontsize=9)
    ax.set(xlabel='All steps flagged (%) · includes synthetic controls',ylabel='Official PRMScore ×100 · noncontrol rows',xlim=(0,45),ylim=(43,76));ax.grid(alpha=.15);ax.legend(fontsize=9,loc='lower right')
fig,ax=plt.subplots(figsize=(10,3.7));draw_budget(ax);fig.tight_layout();budget_chart=chart('threshold_budget',fig,'Sweep of one scalar threshold for each fixed score. X uses all 94,203 steps; Y uses 6,211 noncontrol answers. Markers are the actual operating points. Sweep maxima use evaluation labels and are diagnostics, not deployable selected rules.')
metric_rows=[[colored(m),f'{r["prmscore"]:.2f}',f'{r["auc"]:.4f}',f'{r["top1"]:.2f}',f'{r["flag_rate_all"]*100:.2f}%'] for m,r in [('LSML_CONT',S['LSML_CONT']),('IU_PCR',S['IU_PCR']),('PRM',P)]]
parts.append(slide(8,'Why do the official scores look almost tied?',2,'v16 §§3, 7',
    table(['Method','PRMScore ×100','AUC','Top1 %','All steps flagged'],metric_rows)+budget_chart+
    callout('The decision matters','PRM has higher within-answer AUC; L-SML has higher Top1-on-error. Their native operating points produce similar PRMScore. Moving the global cut improves L-SML only slightly.'),
    'מתעכבים על הצירים. ההשוואה בין נקודות ההפעלה מסבירה מדוע PRMScore קרוב גם כשהמדדים האחרים שונים. לא אומרים שכל המדדים שוויוניים. סף PRM המקורי מסמן שיעור נמוך יותר; שינוי הסף שלו נותן יותר מרווח מאשר שינוי הסף אצלנו.',
    'controls/BUDGET_CURVES.csv; DIAGNOSTIC_CEILINGS.json; DATA.json N16 metric table; PAIRED_DECISION_ANALYSIS.json',
    'העקומה עוברת על ספים של אותו ציון קבוע. X הוא שיעור כל הצעדים שסומנו, כולל controls; Y הוא PRMScore הרשמי ללא controls. הנקודות הן Q80 שלנו וסף 0.5 המקורי של PRM.',
    'שיעור סימון גלובלי שווה אינו מבודד את דירוג הצעדים בתוך תשובה: גם חלוקת הסימונים בין תשובות יכולה להשתנות. מקסימום העקומה נבחר בעזרת תוויות.'))

count_labels=['Our ranking + current Q80','Our ranking + best global cut\n(label-selected diagnostic)','Our ranking + up to 2 flags / answer\n(post hoc count choice)','Our ranking + count from true error type\n(uses type labels)','Our ranking + true error count / answer\n(uses error labels)','Perfect ranking + our Q80 counts\n(uses error labels)']
fig,ax=plt.subplots(figsize=(10,4.2));colors=[COL['LSML_CONT'],'#9fabc1','#c77b21','#baa3cc','#baa3cc','#baa3cc'];ax.barh(range(6),D['ceilings'],color=colors);ax.set_yticks(range(6),count_labels,fontsize=9);ax.invert_yaxis();ax.set(xlim=(0,90),xlabel='Official PRMScore ×100 · same N16 population');ax.grid(axis='x',alpha=.15)
for j,v in enumerate(D['ceilings']):ax.text(v+.6,j,f'{v:.1f}',va='center',fontsize=11)
fig.tight_layout();ceil_chart=chart('count_and_ranking',fig,'N16 L-SML; 6,211 noncontrol answers. Six independent interventions, not additive gains. Rows 1–5 retain our ranking. Row 6 retains Q80 counts and replaces ranking with truth. Type-count 68.9 is rounded from v16; remaining values use saved diagnostics.')
parts.append(slide(9,'What is left to gain?',2.5,'v16 §8a',ceil_chart+
    callout('One-answer illustration','For the pencil example there are 3 annotated errors. “True count” keeps our ordering and flags its top 3 steps. “Perfect ranking” moves steps 3–5 above all valid steps and keeps however many flags Q80 originally produced.')+
    '<p class="caption">The count intervention asks what correct allocation could buy. The ranking intervention asks what perfect ordering could buy. Neither is a usable predictor.</p>',
    'שומרים את כל שש העמודות אבל מגדירים בדיוק מה מוחלף. בדוגמה שלנו true count קובע שלושה סימונים בלי לשנות את הדירוג; perfect ranking מחליף את הסדר בלי לשנות את הכמות. הנתון של שני צעדים הוא בחירה בדיעבד. הנתון לפי סוג שגיאה משתמש בסוג האמיתי.',
    'controls/DIAGNOSTIC_CEILINGS.json; EXACT_DECISION_RULES.csv; v16 §8a / count_structure_check_v1 @17aa1e4 (type-count); same example as slide 1',
    'אורך כל עמודה הוא PRMScore אחרי התערבות אחת. הציר מתחיל באפס. התיאור לצד כל עמודה מציין איזה מידע אמת ניתן לה ומי מחזיק בדירוג.',
    'הבדיקות הן diagnostics עם מידע אמת. השיפורים אינם מצטברים ומספר השגיאות האמיתי אינו חסם עליון מתמטי לכל כלל החלטה אפשרי.'))

ld=D['length_data'];fig,ax=plt.subplots(figsize=(10,3.1))
for (name,r),c in zip(ld['series'].items(),['#203044',COL['IU_PCR'],'#3b8c88',COL['PRM']]):
    ax.plot(range(7),r['values'],'-o',label=f'{name} · ρ={r["rho"]:.2f}',color=c,lw=2)
ax.set_xticks(range(7),[f'{a}\nn={b:,}' for a,b in zip(ld['labels'],ld['n'])]);ax.set(xlabel='Number of steps in the answer',ylabel='Mean errors or flags / answer',ylim=(0,max(max(r['values']) for r in ld['series'].values())+1));ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout()
length_chart=chart('threshold_and_length',fig,'Exactly the same IU-PCR fixed-z>0.5 prediction supplies the threshold table and this count curve. Noncontrol N=6,211; identical length bins. ρ is answer-level Spearman correlation with length. PRM native uses reward<0.5, a different score scale.')
fixed_rows=[]
for m in ['LSML_CONT','IU_PCR']:
    best=D['global_cut_diagnostics'][m]
    fixed_rows.append([colored(m),f'{S[m]["threshold"]:.3f}',f'{S[m]["prmscore"]:.2f}',f'{D["fixed_05"][m]["prmscore"]:.2f}',f'{best["cut"]:.3f} / {best["prmscore"]:.2f}'])
parts.append(slide(10,'Thresholds and answer length',2.5,'v16 §§7, 8b–8c',
    callout('Q80 is already a single global threshold','Normalize each answer’s score, pool all step scores, then apply one scalar 80th-percentile cut. A fixed 0.5 here means z-score >0.5; it is not a probability threshold.')+
    table(['Method','Q80 cut (z)','PRMScore · Q80','PRMScore · z>0.5','Best z cut / score · label-selected'],fixed_rows)+length_chart+
    '<p class="caption">The true error count varies little with length. Both Q80 and the fixed numeric cut assign more flags to longer answers. The documented count-estimator arms did not pass the joint research success criteria.</p><a class="jump" href="#a4">Count-estimator arms and partial failures → appendix</a>',
    'עונים במפורש לשאלה על סף קבוע. Q80 כבר גלובלי על ציון מנורמל. 0.5 של הציון שלנו אינו 0.5 של הסתברות PRM. הטבלה והעקומה משתמשות באותה תחזית בדיוק, שנגזרה מהציונים השמורים. מציגים גם את הסף המיטבי כאבחון עם תוויות ומסבירים שהוא לא פתר את בעיית ספירת השגיאות.',
    'Saved bank_N16.npz; derive_data.py exact z>0.5; controls/BUDGET_CURVES.csv; PER_ANSWER_COUNTS.csv; label_free_error_count_v1 reports',
    'הטבלה מציגה את ציון PRMScore תחת שלושה ספים. בגרף X הוא אורך התשובה ו־Y ממוצע מספר השגיאות האמיתיות או הסימונים בכל bin, לא שיעור הצלחה.',
    'הבדיקה היא אחרי normalization בתוך תשובה. היא אינה בדיקה של סף על סקאלת הסיכון הגולמית. הקשר לאורך אינו מסביר לבדו את כל הכישלונות.'))

external=D['external_sources']['qwen3_4b_vs_8b__external_cont.csv'];relay=D['external_sources']['RESULTS_relayed.csv']
external_panels=[]
for bench,title,metric,n in [('socratic','Socratic-PRMBench','PRMScore',2995),('hard2verify','Hard2Verify','Balanced F1',200)]:
    extrows=[]
    for m in ['LSML_CONT','IU_PCR']:
        r=next(r for r in external if r['cell'].startswith(bench) and r['bank']=='N16' and r['scorer']==m)
        extrows.append([colored(m),'Q80',f'{float(r["official_metric_8b"])*100:.2f} / {float(r["official_metric_4b"])*100:.2f}',f'{float(r["within_auc_8b"]):.3f} / {float(r["within_auc_4b"]):.3f}'])
    lr8=next(r for r in relay if r['benchmark']==bench and r['method']=='LR' and r['model']=='8b');lr4=next(r for r in relay if r['benchmark']==bench and r['method']=='LR' and r['model']=='4b')
    extrows.append([colored('LR','LR · supervised'),'Q80',f'{float(lr8["official_metric"])*100:.1f} / {float(lr4["official_metric"])*100:.1f}',f'{float(lr8["within_auc"]):.3f} / {float(lr4["within_auc"]):.3f}'])
    for rule,label in [('NATIVE_0_5','native 0.5'),('ANSWER_Z_Q80','Q80')]:
        p=next(r for r in relay if r['benchmark']==bench and r['method']=='PRM' and r['rule']==rule)
        extrows.append([colored('PRM','PRM-7B'),label,f'{float(p["official_metric"])*100:.1f}',f'{float(p["within_auc"]):.3f}'])
    external_panels.append('<div class="panel"><h2>'+title+'</h2><p class="caption">'+metric+' ×100 · '+str(n)+' answers</p>'+table(['Method','Rule','Official · 8B / 4B','AUC · 8B / 4B'],extrows,'compact')+'</div>')
parts.append(slide(11,'Generalization beyond PRMBench',2.5,'v16 §5',
    '<div class="columns">'+''.join(external_panels)+'</div>'+bullets(['Socratic: native PRM remains stronger; label-free fusion does not win everywhere.','Hard2Verify: changing PRM’s threshold matters greatly; the small N=200 result needs caution.','LR transfers from PRMBench labels. Label-free fusion is fitted on each target without labels; this is not frozen-weight transfer.'])+
    '<p class="caption">8B / 4B refers to the model producing our features. PRM is the same independent 7B baseline. Relayed LR/PRM values retain their published CSV precision; no paired intervals were supplied.</p>',
    'מראים את שני ה־benchmarks גם כשהתוצאות מעורבות. Socratic משתמש ב־PRMScore ו־Hard2Verify ב־Balanced F1. השיטות ללא תוויות מותאמות על כל יעד בלי תוויות; LR מאומן על PRMBench בלבד. ב־Hard2Verify ההשוואה תלויה מאוד בסף PRM, והאוכלוסייה קטנה.',
    'Pinned v16 external_cont.csv; external_lr_prm_v1/RESULTS_relayed.csv; v16 §5',
    'בכל פאנל עמודת המדד הרשמי שייכת ל־benchmark שלו. בזוגות המספרים ראשון הוא 8B והשני 4B. PRM מוצג פעמיים עם אותו AUC וכללי החלטה שונים.',
    'אין ממוצע משותף בין PRMScore ל־Balanced F1. התוצאות אינן הצלחת הכללה אחידה. frozen-weight results שייכות לבדיקה אחרת בנספח.'))

fusion_pipeline='<div class="fusion-diagram"><div class="model-box"><b>ONE PRM CHECKPOINT · ONE FORWARD</b><div class="columns"><div>Reward head<br><strong>PRM risk</strong></div><div>Backbone token distributions<br><code>energy_level</code><br><code>hist_entropy_cusum_abs_series</code><br><code>realized_drv</code></div></div></div><div class="fusion-arrow">↓</div><div class="fuser">Continuous L-SML → combined step risk</div></div>'
def boosting_result():
    if U['boosting']['status']=='failed':
        return callout('L-SML fit failed',e(U['boosting']['source']),'research')
    if U['boosting']['status']!='complete':
        return '<div class="placeholder"><span class="tag">RESULT + GRAPH RESERVED FOR CLAUDE</span><h2>L-SML results pending</h2><p>Compare PRM alone, each backbone signal, L-SML of the three backbone signals, and L-SML of PRM risk plus the three backbone signals.</p><p><b>Required graph:</b> aligned panels for official PRMScore and within-answer AUC on matched rows, with the flag rule beside every arm.</p><p>Show a negative result or fit failure explicitly. No fixed-weight average is shown as L-SML.</p></div>'
    rows=U['boosting']['rows'];fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,key in zip(axs,['prmscore','auc']):
        ax.barh(range(len(rows)),[r[key] for r in rows],color=COL['LSML_CONT']);ax.set_yticks(range(len(rows)),[r['label'] for r in rows],fontsize=8);ax.invert_yaxis();ax.set_xlabel('PRMScore ×100' if key=='prmscore' else 'Within-answer AUC');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();return chart('prm_lsml_update',fig,U['boosting']['source'])
parts.append(slide(12,'Can the PRM benefit from its own internal signals?',2,'v16 §9 → revised experiment',fusion_pipeline+boosting_result()+
    '<p class="caption">The three features were selected using labels on Qwen-PRM-7B. L-SML fitting must exclude evaluation labels. Keep feature selection and evaluation separate. The PRM itself was trained with supervision.</p>',
    'זו השאלה החדשה: האם מידע שכבר קיים בתוך אותו PRM מוסיף על ה־head שלו באמצעות L-SML? להצביע על שלושה פיצ׳רים ועל פלט ה־head מאותו checkpoint. שומרים מקום לתוצאה של קלוד. 69.38 היה ממוצע עם משקלים קבועים; הוא אינו תוצאה של הניסוי המוצג כאן ולכן הוסר.',
    'PRM_UPDATE.json; PRM boosting branch small-bank selection; revised same-checkpoint L-SML result pending',
    'התרשים הוא מפרט ניסוי. הגרף העתידי צריך להראות baseline, כל פיצ׳ר, fusion של שלושת הפיצ׳רים ו־fusion של ארבעת הסיגנלים כולל PRM.',
    'אין כרגע תוצאה מאומתת של L-SML לניסוי הזה. בחירת הפיצ׳רים השתמשה בתוויות ולכן אין לתאר את כל התהליך כ־unsupervised.'))

def generalization_result():
    if U['generalization']['status']=='failed':
        return callout('Cross-PRM evaluation failed or incomplete',e(U['generalization']['source']),'research')
    if U['generalization']['status']!='complete':
        return '<div class="placeholder"><span class="tag">STATUS · '+e(U['updated'])+'</span><h2>PRM model runs in progress</h2><p>Same-checkpoint L-SML results and recipe alignment are pending.</p><p>Targets in the documented generalization plan: Qwen PRM-72B, Skywork PRM-7B, RLHFlow Llama-3.1-8B PRM.</p><p><b>Reserved graph:</b> for each PRM, native head vs L-SML of its head + its own three backbone signals; PRMScore and within-answer AUC side by side.</p></div>'
    rows=U['generalization']['rows'];fig,axs=plt.subplots(1,2,figsize=(11,3.7))
    for ax,pk,lk,label in [(axs[0],'prm_prmscore','lsml_prmscore','Official PRMScore ×100'),(axs[1],'prm_auc','lsml_auc','Within-answer AUC')]:
        for j,r in enumerate(rows):
            ax.plot([r[pk],r[lk]],[j,j],color='#aebdca',lw=2)
        ax.scatter([r[pk] for r in rows],range(len(rows)),color=COL['PRM'],label='Native PRM',zorder=3)
        ax.scatter([r[lk] for r in rows],range(len(rows)),color=COL['LSML_CONT'],label='Same-checkpoint L-SML',zorder=3)
        ax.set_yticks(range(len(rows)),[r['checkpoint'] for r in rows],fontsize=8);ax.invert_yaxis();ax.set_xlabel(label);ax.grid(axis='x',alpha=.15);ax.legend(fontsize=8)
    fig.tight_layout()
    return chart('prm_generalization_update',fig,U['generalization']['source'])+table(['PRM checkpoint','Rule','PRMScore · native / L-SML','AUC · native / L-SML'],[[e(r['checkpoint']),e(r['rule']),f'{r["prm_prmscore"]:.2f} / {r["lsml_prmscore"]:.2f}',f'{r["prm_auc"]:.3f} / {r["lsml_auc"]:.3f}'] for r in rows])
parts.append(slide(13,'Does the same fusion transfer to other PRMs?',1,'PRM generalization specification',generalization_result()+
    bullets(['Keep the three feature definitions and the L-SML procedure fixed.','Extract head and backbone signals from each PRM’s own checkpoint.','No feature re-selection using the new model’s evaluation labels.'])+'<p class="caption">These are cross-PRM tests on PRMBench, distinct from the external-benchmark tests on slide 11. Existing fixed-average runs do not establish L-SML transfer.</p>',
    'ההכללה הזו בין מודלי PRM שונה מההכללה בין datasets. אם קלוד יסיים עד הפגישה נציג תוצאות מתאימות. אם לא, נשאיר סטטוס מתוארך. הריצות שתוכננו קודם כללו ממוצע קבוע, ולכן יש לוודא יישור מתכון לפני שמייחסים להן הצלחת L-SML.',
    'docs/experiments/PRM_BOOSTING_GENERALIZATION_V1.md @ claude/prm-boosting-generalization-v1; PRM_UPDATE.json; status reported by Omri 14 Oct',
    'המקום השמור מיועד להשוואה של פלט ה־head מול fusion של סיגנלים פנימיים באותו מודל. יש להציג גם את כלל ההחלטה והמדד בכל מודל.',
    'הסטטוס נשען על דיווח המשתמש, לא על polling של AIRCC. מתכון L-SML עדיין מחכה לתוצאות; לא מציגים ממוצע קבוע כתוצאה שלו.'))

parts.append(slide(14,'What should we investigate next?',1,'v16 §10; meeting discussion',
    '<div class="questions"><div><span>01</span><h2>Which additions help LR and hurt label-free fusion?</h2><p>Start with the 18 channels added from N25 to N43. Separate feature value, redundancy and score weighting.</p></div><div><span>02</span><h2>How many steps should we flag?</h2><p>Test whether score structure predicts error count beyond answer length.</p></div><div><span>03</span><h2>Can internal PRM signals improve its own head?</h2><p>Evaluate same-checkpoint L-SML, then keep the feature definitions fixed across PRMs.</p></div></div><p class="discussion">Discussion · 9 minutes</p>',
    'מסיימים בשלוש שאלות מחקר, לא ברשימת ניסויים ארוכה. מבקשים מהמנחים להתייחס למה ראוי להשקיע בו: הפער בין LR ל־fusion, כמות הסימונים, והכיוון של שילוב סיגנלים מתוך אותו PRM. להשאיר זמן לתגובה.',
    'v16 §10; agreed conversation; slides 4, 9–10, 12–13',
    'אין כאן תוצאות חדשות. אלו שלוש שאלות שמבוססות על הממצאים שהוצגו.',
    'הצעות להמשך אינן מסקנות סיבתיות ואינן ניסויים שהושלמו.'))

# Appendices retain evidence, but remove superseded claim language from the deck.
all_methods=csv_rows(HERE/'sources/ALL_BANK_Q80_PRMSCORE.csv')
matrix_methods=[m for m in dict.fromkeys(r['method'] for r in all_methods) if m!='PRM_NATIVE' and not m.endswith('_LEVELSIGN')]
matrix_rows=[]
for m in matrix_methods:
    values=[]
    for b in D['banks']:
        r=next((r for r in all_methods if r['method']==m and r['bank']==b and r['classification']=='ALL_NONCONTROL'),None)
        values.append(f'{float(r["category_score"])*100:.2f}' if r else '—')
    matrix_rows.append([e(NAMES.get(m,m))]+values)
parts.append(slide(1,'Full methods and feature-bank comparisons',0,'v16 §0; appendix D',
    '<h2>Original corrected AUC overview</h2>'+original_figure('s0')+'<h2>Official PRMScore · nested digit banks · Q80</h2>'+table(['Method']+D['banks'],matrix_rows)+'<p class="caption">The original AUC overview uses historical LR. Main slides use fresh LR OOF predictions for both metrics. The all-bank family contains parallel digit-free and digit-containing chains, not one universal inclusion chain.</p>',
    'טבלת AUC המלאה המקורית נשמרה למענה לשאלות. LR בה הוא היסטורי. בטבלה הנוספת מציגים PRMScore לכל השיטות השמורות על בנקי digits. ההבדל במקור LR מפורש כדי לא ליצור ערבוב.',
    'v16 §0; ALL_BANK_Q80_PRMSCORE.csv; historical LR PRMBENCH_COMPARISON.csv',
    'מטריצת AUC ושורת PRMScore משתמשות במדדים שונים. בנקים הם אוספי פיצ׳רים תלויים, לא חזרות עצמאיות.',
    'הטבלה המקורית אינה מקור PRMScore של LR בגרסה החדשה.',True))

bug_draft=HERE/'sources/correction_example.html'
draft=bug_draft.read_text()
bug_box=re.search(r'<div class="box"><span class="kt">What the bug was.*?</details></div>',draft,re.S)
bug_chart_path=HERE/'sources/prefix_token_diagnostic.png'
bug_chart='<figure><img src="data:image/png;base64,'+base64.b64encode(bug_chart_path.read_bytes()).decode()+'" alt="Aggregate first-token diagnostics before and after empty-think-prefix fix"><figcaption>Aggregate over 6,969 rows, not measurements of the example. Token-0 median surprisal 34.6 → 24.1 nats; first-position top prediction &lt;think&gt; 100% → 0%. mask1 is a separate decision.</figcaption></figure>'
selfgen=re.search(r'<div class="box"><span class="kt">Without teacher forcing</span>.*?</div>',original_section('s1'),re.S)[0]
parts.append(slide(2,'Input correction and separate generation checks',0,'v16 §§1, 2; walkthrough diagnostic',
    (bug_box[0] if bug_box else '')+bug_chart+original_figure('s1')+'<h2>A separate question: generated answers vs supplied answers</h2>'+selfgen+
    callout('Two different questions','The prompt correction checks whether we conditioned correctly when scoring an existing answer. Self-generation checks signal transfer to answers produced by the model; different datasets prevent a causal teacher-forcing comparison.'),
    'הבאג נשאר בנספח כדי לא לקטוע את הסיפור. אם נשאלים מה היה הבאג מציגים prompt ותשובה, ההבדל בגבול assistant, והנתון לפני/אחרי בטוקן הראשון. שאלת self-generation נפרדת מהתיקון. ההשוואה בין 4B ל־8B בגוף היא מודל המייצר פיצ׳רים, לא הוכחה לגבי teacher forcing.',
    'v16 §1; walkthrough concrete reconstructed prompt; THINKFIX_STATS.json aggregate; selfgen/B16_SELFGEN_VS_PB_TF.csv',
    'הנתונים הראשונים הם אבחון aggregate של מיקום הטוקן הראשון. הטקסט מדגים prompt משוחזר מה־formatter, לא capture של inference בודד.',
    'הבדל self-generation כולל שינוי dataset ואוכלוסיית שגיאות. רווח סמך החוצה אפס אינו מוכיח שקילות.',True))

four=[r for r in D['fourway'] if r['bank']=='N16' and r['method']=='LSML' and r['axis']=='classification']
rows=[]
for k in keys:
    r=next(r for r in four if r['value']==k)
    rows.append([codes[keys.index(k)],r['n'],r['both'],r['ours_only'],r['lr_only'],r['neither']])
totals={c:sum(int(r[c]) for r in four) for c in ['n','both','ours_only','lr_only','neither']}
rows.append(['Total']+[totals[c] for c in ['n','both','ours_only','lr_only','neither']])
parts.append(slide(3,'Which method’s highest-risk step is erroneous?',0,'v16 §6b',
    '<p class="lead">Four-way Top1-on-error counts · L-SML vs fresh OOF LR · N16</p>'+table(['Error type','Answers','Both choose error','Only L-SML','Only LR','Neither'],rows)+
    callout('Precisely what “right” means','The method’s highest-risk step belongs to the annotated error set. This does not mean that it found every error, the first error, or a correct final answer.')+
    '<p class="caption">Only L-SML: 485 answers. Only LR: 366. No threshold is used. The LR/PRM similarity previously discussed belongs to this Top1 diagnostic, not automatically to PRMScore.</p>',
    'מגדירים הצלחה לפני שמציגים מספרים. כל תשובה עם שגיאה נכנסת בדיוק לתא אחד. הקשר ל־LR כאן הוא complementarity של בחירת צעד אחד, לא של הסיווג הרשמי. קל יותר לקרוא טבלת ארבעת המצבים מאשר גרף שדורש ניחוש של המדד.',
    'lr/FOURWAY_SUMMARY.csv; v16 §6b; N16 LSML classification rows',
    'כל מספר הוא כמות תשובות. בשתי עמודות האמצע מופיעות תשובות שבהן רק אחת מהשיטות בחרה צעד שגוי.',
    'אין PRMScore בטבלה זו ואין להסיק ממנה מי מסווג טוב יותר את כל הצעדים.',True))

parts.append(slide(4,'Count-estimator attempts and incomplete arms',0,'v16 §8b',original_figure('s8',1)+
    callout('Coverage is part of the result','Arm A failed in 4 of 5 N16 folds: only 1,269 / 6,211 noncontrol answers were covered. It is not filled with zero counts. Codex OOF arms use their own OOF Q80 comparator.')+
    callout('Bounded conclusion','No tested candidate passed the joint MAE / PRMScore / recall criteria. Some rules reduced false flags on clean answers, a distinct benefit from improving overall PRMScore.'),
    'שומרים את תוצאות כל הזרועות ואת חלקיות הכיסוי. אין לומר שנבדקה משפחה במלואה אם חלק מהזרועות נכשלו בהפקת תחזיות. זו מסקנה מוגבלת לניסויים שבוצעו, לא הוכחה שספירת שגיאות ללא תוויות אינה אפשרית.',
    'v16 §8b; label_free_error_count_v1_2026-10-12/report; evaluation/SUCCESS.json; TASK_STATUS',
    'כל עמודה היא שינוי PRMScore מול comparator Q80 של אותו פרוטוקול. הזרועות אינן כולן חולקות בדיוק אותו fitted score.',
    'כיסוי חלקי וכישלון התאמה אינם תחזית של אפס. שיפור ב־clean-answer decision אינו בהכרח שיפור בציון הכולל.',True))

parts.append(slide(5,'Position, length and repetition controls',0,'v16 §4b',original_figure('s4',1)+
    callout('Read each eligible population separately','Raw AUC uses 6,030 answers. Position matching and position+length matching retain smaller, different populations. Compare methods within the same matched set; the raw-to-matched change is not a causal effect.')+
    bullets(['NR: L-SML − PRM after position+length matching is +19.7 AUC points [15.9, 23.6].','NCL: +4.3 points; its interval crosses zero.','Repetition matching is a supplementary nuisance diagnostic, not evidence of conditional independence.']),
    'זה הגרף שהמשתמש לא רצה להציג בלי להבין. הוא בנספח, עם הסבר מפורש: matching מצמצם אוכלוסייה, ולכן משווים שיטות בתוך אותה אוכלוסייה. NR נשאר חזק, NCL לא מקבל מסקנה עם CI חיובי.',
    'controls/NUISANCE_CONTROLS.csv; v16 §4b; matched source-group contrasts',
    'הגרף מציג AUC על זוגות valid/error שנשארו אחרי matching. N ומספר זוגות משתנים בין הבקרות.',
    'matching אינו התערבות סיבתית. חסימות של מיקום ואורך אינן מסירות כל confounding.',True))

locked=D['external_sources']['qwen3_4b_vs_8b__external_lsml_locked.csv']
locked_rows=[[e(r['cell'].replace('_qwen3_8b','')),e(r['arm']),f'{float(r["metric_8b"])*100:.2f} / {float(r["metric_4b"])*100:.2f}',f'{float(r["within_auc_8b"]):.3f} / {float(r["within_auc_4b"]):.3f}'] for r in locked]
parts.append(slide(6,'Model size and additional transfer evidence',0,'v16 §§5, 6c; appendix A',
    original_figure('s6',2)+callout('A separate robustness question','Changing the feature-generating model from Qwen3-8B to Qwen3-4B asks whether the signal depends strongly on model size. Close point estimates do not establish statistical equivalence.')+
    callout('Source-disjoint internal validation','The saved half-source split gives N16 IU-PCR AUC 0.786 and L-SML 0.784. These data were previously inspected; this is internal validation, not untouched confirmation.')+
    '<h2>Separate external locked-weight experiment</h2>'+table(['Benchmark','Arm','Official metric ×100 · 8B / 4B','Within-answer AUC · 8B / 4B'],locked_rows,'compact')+'<p class="caption">Socratic uses PRMScore; Hard2Verify uses Balanced F1. These older locked/local arms are a different experiment from the N16 target-fitted label-free pipeline on slide 11. Do not splice their higher scores into that comparison.</p>',
    'להפריד בדיקת 4B/8B, בדיקת source-disjoint ובדיקת frozen weights. לא לתת לכולן שם כללי של generalization בלי להגדיר מה נשאר קבוע ומה השתנה.',
    'v16 §6c; validation/HALF_SUMMARY.json; pinned qwen3_4b_vs_8b__external_lsml_locked.csv',
    'כל נקודה בגרף היא הפרש AUC של 4B פחות 8B עבור אותה שיטה ובנק. אלו אומדנים ללא paired CI.',
    'PASS בדוחות חצי־מקורות מציין שהתאמה הושלמה, לא שניסוי אישר יתרון.',True))

parts.append(slide(7,'Paper reference rows and evaluator conventions',0,'v16 §3; paper Table 3',
    table(['Published model','Overall PRMScore ×100','Simplicity','Soundness','Sensitivity'],[[e(r['model']),r['overall'],r['simplicity'],r['soundness'],r['sensitivity']] for r in sorted(D['paper'],key=lambda r:float(r['overall']),reverse=True)])+
    bullets(['Paper values retain published protocols; they are not matched reruns.','Overall pools all noncontrol steps. Category summaries average their member subtype scores.','NR/NCL use the official evaluator’s validity fallback because our scorer has no separate redundancy head.','Published out-of-range error indices retain the evaluator convention; they are not silently clipped.']),
    'לשאלות על ה־leaderboard מציגים את כל שורות המאמר. לציין את adaptation של evaluator ב־NR/NCL ואת ההבדל בין Overall לבין ממוצעי קטגוריות.',
    'report/paper_table3.csv; spectral_utils/prmbench.py evaluator port; v16 appendix D',
    'כל שורה היא תוצאה שפורסמה. אין כאן מספרים חדשים מהרצות שלנו.',
    'אין טענה להשוואה תואמת בין כל פרוטוקולי המאמר לבין שלנו.',True))

CSS=r'''
:root{--ink:#203044;--muted:#667587;--accent:#1766a1;--line:#dde4eb;--paper:#fff;--warm:#f4f6f8}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--warm);color:var(--ink);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}button,a{touch-action:manipulation}button{font:inherit;cursor:pointer}a{color:var(--accent)}.sidebar{position:fixed;width:232px;left:0;top:0;bottom:0;background:#14293b;color:#e9f0f5;padding:22px 16px;overflow:auto;z-index:15}.brand{font-size:21px;font-weight:750;letter-spacing:-.5px}.version{font-size:12px;color:#a8bccd;margin:7px 0 22px}.sidebar nav a{display:block;color:#c7d4df;text-decoration:none;font-size:12px;padding:8px 9px;border-left:3px solid transparent;border-radius:3px;line-height:1.35}.sidebar nav a.current{background:#234259;border-left-color:#f0ac54;color:white}.sidebar .nav-num{display:inline-block;width:24px;color:#96adbe}.appendix-label{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#91a8ba;margin:20px 8px 5px}.topbar{position:sticky;top:0;margin-left:232px;height:60px;display:flex;align-items:center;justify-content:space-between;padding:10px 28px;background:#ffffffee;border-bottom:1px solid var(--line);z-index:12;gap:10px;backdrop-filter:blur(10px)}.topbar button,.tabs button{border:1px solid var(--line);border-radius:7px;padding:6px 11px;background:white;color:var(--ink)}.topbar .controls{display:flex;gap:7px;align-items:center}.progress{font-size:13px;color:var(--muted)}main{margin-left:232px;padding:24px 28px 70px}.slide{background:white;max-width:1240px;margin:0 auto 28px;padding:34px 40px 28px;border:1px solid var(--line);border-radius:12px;scroll-margin-top:80px;box-shadow:0 4px 18px #172d4010;min-height:calc(100vh - 132px)}.eyebrow{display:flex;justify-content:space-between;font-size:10px;font-weight:700;letter-spacing:1.5px;color:var(--muted);gap:15px;margin-bottom:13px}.eyebrow span{letter-spacing:.3px;text-align:right}h1{font-size:clamp(26px,2.5vw,38px);line-height:1.12;letter-spacing:-.8px;margin:0 0 20px}h2{font-size:21px;line-height:1.25;margin:10px 0}p{margin:10px 0}.lead{font-size:21px;line-height:1.4;margin:0 0 20px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}.callout{border-left:4px solid var(--accent);padding:13px 16px;background:#f1f6fa;border-radius:0 7px 7px 0;margin:16px 0}.callout strong{font-size:17px}.callout p{margin:5px 0 0}.callout.research{border-color:#c77b21;background:#fff7e9}.caption,figcaption{font-size:12px;color:var(--muted);line-height:1.45}.source{font-size:10px;color:#758292;border-top:1px solid var(--line);margin-top:20px;padding-top:10px;overflow-wrap:anywhere}.steps{list-style:none;padding:0;margin:0}.steps li{border-left:3px solid #7da69b;background:#f7faf9;padding:7px 10px;margin:5px 0;font-size:13px;line-height:1.35}.steps li.error{border-color:#d4543c;background:#fff3ef}.step-label{display:block;font-size:10px;font-weight:700;text-transform:uppercase;color:#667587;margin-bottom:3px}.pipeline{display:flex;flex-wrap:wrap;gap:7px;margin-top:20px}.pipeline span{font-size:12px;background:#e9f1f7;padding:8px;border-radius:5px}.pipeline span+span:before{content:'→ ';font-weight:700}.metric-cards{display:grid;grid-template-columns:1fr 1fr;gap:24px}.metric-cards>div{background:#f6f8fb;border:1px solid var(--line);padding:18px;border-radius:8px}.tag{font-size:10px;font-weight:800;letter-spacing:1px;color:var(--accent)}.formula{font-size:27px;letter-spacing:.5px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;font-variant-numeric:tabular-nums}th{background:#edf3f8;font-size:11px;text-align:left;padding:12px 10px;white-space:nowrap}td{padding:11px 10px;border-bottom:1px solid var(--line)}tbody tr:nth-child(even){background:#f8fafb}.compact{font-size:12px}.compact th,.compact td{padding:10px 7px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}.chart{margin:14px 0 8px}.chart svg{display:block;width:100%;height:auto;max-height:410px}.chart.tall svg{max-height:none}.chart img,figure img{max-width:100%;height:auto}.points{padding-left:24px;margin:18px 0}.points li{margin:7px 0}.stat-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}.stat-strip>div{background:#f1f6fa;padding:14px;border-radius:7px}.stat-strip b{display:block;font-size:22px}.stat-strip span{font-size:12px;color:var(--muted)}.taxonomy{display:flex;gap:9px 18px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin:10px 0}.taxonomy b{color:var(--ink)}.insight-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.insight-grid>div{background:#fff6e9;border-top:3px solid #e8a44a;padding:12px}.insight-grid p{font-size:13px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:15px 0}.tabs button.active{background:var(--accent);color:white;border-color:var(--accent)}.panel{border:1px solid var(--line);padding:16px;border-radius:8px}.fusion-diagram{margin:20px auto;text-align:center;max-width:760px}.model-box{border:2px solid var(--accent);border-radius:10px;padding:16px;background:#f6faff}.model-box>.columns{margin-top:16px}.model-box code{font-size:12px;overflow-wrap:anywhere}.fusion-arrow{font-size:30px;color:var(--accent)}.fuser{padding:14px;background:#203e57;color:white;border-radius:8px;font-weight:700}.placeholder{border:2px dashed #c77b21;background:#fffbf3;padding:22px;border-radius:10px;margin:20px 0}.placeholder h2{color:#966018}.placeholder p{font-size:14px}.questions>div{border-bottom:1px solid var(--line);padding:18px 0}.questions span{font-size:12px;color:#c77b21;font-weight:800}.questions h2{font-size:25px}.discussion{font-size:24px;color:var(--accent);margin-top:28px}.jump{font-size:12px}.speaker{display:none;background:#fff8e9;border:1px solid #ebd5a7;padding:18px;border-radius:8px;font-size:14px;margin-top:24px}.notes-visible .speaker{display:block}.speaker h3{margin:0 0 9px}.appendix-slide{min-height:auto;border-top:4px solid #a0adba}.box{border:1px solid var(--line);padding:15px;border-radius:8px}.box .kt{font-weight:700}.box pre,pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid var(--line);padding:12px;margin:15px 0;border-radius:7px}summary{cursor:pointer;font-weight:650}.presentation .sidebar{display:none}.presentation .topbar,.presentation main{margin-left:0}.presentation .slide{display:none;max-width:1440px}.presentation .slide.active{display:block}.presentation main{padding-top:16px}.presentation .slide{min-height:calc(100vh - 112px)}.overlay{position:fixed;inset:0;background:#f8fafc;z-index:50;padding:55px 30px 25px;overflow:auto}.overlay button{position:fixed;top:12px;right:24px;z-index:51;padding:8px 16px;background:#203e57;color:white;border:0;border-radius:7px}.overlay .chart{max-width:1500px;margin:auto}.overlay svg{max-height:none!important;width:100%}[hidden]{display:none!important}
@media(max-width:1000px){.sidebar{width:188px}.topbar,main{margin-left:188px}.slide{padding:25px}.columns{grid-template-columns:1fr}.metric-cards{gap:12px}.eyebrow{font-size:9px}.compact{font-size:13px}}@media(max-width:700px){.sidebar{display:none}.topbar,main{margin-left:0}.topbar{padding:8px 12px;font-size:11px;flex-wrap:wrap;height:auto;min-height:60px}main{padding:12px}.slide{padding:20px 16px}.metric-cards,.stat-strip,.insight-grid{grid-template-columns:1fr}h1{font-size:28px}.progress{font-size:11px}.topbar button{font-size:11px}.chart svg{max-height:none}.eyebrow span{display:none}}
@media print{body{background:white;font-size:12px}.sidebar,.topbar,.tabs,.jump{display:none!important}main,.presentation main{margin:0;padding:0}.slide,.presentation .slide{display:block!important;page-break-after:always;border:0;box-shadow:none;margin:0;padding:12mm;min-height:0;max-width:none}.chart svg{max-height:150mm}.chart.tall svg{max-height:245mm}.speaker{display:none!important}.notes-visible .speaker{display:block!important}.source{font-size:8px}.placeholder{background:white}[data-view="feature-profile"]{display:block!important}details{display:block}h1{font-size:26px}.columns{grid-template-columns:1fr 1fr}}
'''
CSS+='\nfigure{max-width:100%;margin:16px 0}figure svg{display:block;width:100%;height:auto}svg text{font-family:Arial,sans-serif!important}.slide{overflow-wrap:break-word}\n'
JS=r'''
const sections=[...document.querySelectorAll('.slide')];let current=0;
const links=[...document.querySelectorAll('nav a')];
function select(index,scroll=true){current=Math.max(0,Math.min(index,sections.length-1));sections.forEach((s,i)=>s.classList.toggle('active',i===current));links.forEach((a,i)=>a.classList.toggle('current',i===current));document.querySelector('#progress').textContent=(sections[current].classList.contains('appendix-slide')?'Appendix':'Slide')+' '+(sections[current].classList.contains('appendix-slide')?sections[current].id.slice(1):current+1)+(current<14?' / 14':'');if(scroll){sections[current].scrollIntoView({block:'start'});history.replaceState(null,'','#'+sections[current].id)}document.querySelector('#prev').disabled=current===0;document.querySelector('#next').disabled=current===sections.length-1;}
document.querySelector('#prev').onclick=()=>select(current-1);document.querySelector('#next').onclick=()=>select(current+1);
document.querySelector('#notes').onclick=()=>{document.body.classList.toggle('notes-visible');document.querySelector('#notes').setAttribute('aria-pressed',document.body.classList.contains('notes-visible'));};
document.querySelector('#present').onclick=()=>{document.body.classList.toggle('presentation');document.querySelector('#present').textContent=document.body.classList.contains('presentation')?'Reading mode':'Present';select(current);};
document.querySelector('#print').onclick=()=>window.print();
document.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener('click',ev=>{let i=sections.findIndex(s=>'#'+s.id===a.getAttribute('href'));if(i>=0){ev.preventDefault();select(i)}}));
document.querySelectorAll('.view').forEach(b=>b.onclick=()=>{const s=b.closest('.slide');s.querySelectorAll('[data-view]').forEach(p=>p.hidden=p.dataset.view!==b.dataset.panel);s.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x===b));});
function closeZoom(){document.querySelector('.overlay')?.remove();document.body.style.overflow='';}
document.querySelectorAll('.zoom').forEach(b=>b.onclick=()=>{const o=document.createElement('div');o.className='overlay';o.setAttribute('role','dialog');o.setAttribute('aria-label','Full signal heatmap');o.innerHTML='<button aria-label="Close enlarged heatmap">Close · Esc</button>'+document.getElementById(b.dataset.zoom).innerHTML;o.querySelector('button').onclick=closeZoom;document.body.appendChild(o);document.body.style.overflow='hidden';o.querySelector('button').focus()});
window.addEventListener('keydown',ev=>{if(['INPUT','TEXTAREA','SELECT'].includes(ev.target.tagName))return;if(ev.key==='Escape'){closeZoom();return}if(ev.key==='ArrowRight'||ev.key==='PageDown'){ev.preventDefault();select(current+1)}if(ev.key==='ArrowLeft'||ev.key==='PageUp'){ev.preventDefault();select(current-1)}if(ev.key.toLowerCase()==='n')document.querySelector('#notes').click();if(ev.key.toLowerCase()==='p')document.querySelector('#present').click();});
const observer=new IntersectionObserver(entries=>{if(document.body.classList.contains('presentation'))return;const hit=entries.filter(x=>x.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0];if(hit)select(sections.indexOf(hit.target),false)},{rootMargin:'-65px 0px -45% 0px',threshold:[0,.1,.4,.6]});sections.forEach(s=>observer.observe(s));
const initial=sections.findIndex(s=>'#'+s.id===location.hash);select(initial>=0?initial:0,initial>=0);
'''
nav=''.join(('<div class="appendix-label">Appendix</div>' if r['appendix'] and r['number']==1 else '')+f'<a href="#{r["id"]}"><span class="nav-num">{"A" if r["appendix"] else ""}{r["number"]:02}</span>{e(r["title"])}</a>' for r in slides)
document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Advisor meeting · 15 Oct 2026 · v17</title><style>'+CSS+'</style></head><body><aside class="sidebar"><div class="brand">Signals → decisions</div><div class="version">Advisor meeting · v17<br>15 October 2026<br>31 min + 9 min discussion</div><nav aria-label="Slide navigation">'+nav+'</nav></aside><header class="topbar"><span class="progress" id="progress">Slide 1 / 14</span><div class="controls"><button id="prev" aria-label="Previous slide">←</button><button id="next" aria-label="Next slide">→</button><button id="present">Present</button><button id="notes" aria-pressed="false">הערות דובר</button><button id="print">Print / PDF</button></div></header><main>'+''.join(parts)+'</main><script>'+JS+'</script></body></html>'
destination=HERE/'published_v17_2026-10-15.html'
destination.write_text(document)
(HERE/'SLIDE_MAP.json').write_text(json.dumps(slides,ensure_ascii=False,indent=2)+'\n')
book='# v17 — הערות דובר ומיפוי תכנים\n\n31 דקות תוכן + 9 דקות דיון.\n\n'
for r in slides:
    book+=f'## {"נספח " if r["appendix"] else ""}{r["number"]} — {r["title"]}\n\nמקור: {r["old"]}. זמן: {r["minutes"]} דקות.\n\n{r["notes"]}\n\n**איך קוראים:** {r["question"]}\n\n**גבול המסקנה:** {r["limit"]}\n\nמקורות: {r["sources"]}\n\n'
(HERE/'SPEAKER_NOTES_HE.md').write_text(book.rstrip()+'\n')
(HERE/'BUILD_RECEIPT.json').write_text(json.dumps({'html':destination.name,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'main_slides':14,'appendix_slides':7,'minutes':sum(r['minutes'] for r in slides),'figures':FIGS,'source_v16_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'source_v16_blob':'9aaab92066600edb593b7050cd7f042acf7f1719'},indent=2)+'\n')
print(f'Built {destination}: 14 slides, 7 appendices, {sum(r["minutes"] for r in slides):g} minutes; standalone HTML.')
