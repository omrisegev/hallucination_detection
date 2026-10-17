"""Build the standalone v11 advisor page from persisted evidence, without HTML patches.

Optional report_data.json supplies newly generated sections and the final task inventory.
Every figure is embedded; relative links lead to the raw evidence in this package.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import html
import json
import math
import re
import shutil
import io
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
REPO = PACKAGE.parent.parent
PAPER = 'https://arxiv.org/html/2501.03124v1'
HISTORICAL = 'https://github.com/omrisegev/hallucination_detection/blob/8996379/docs/meetings/advisor_meeting_2026-10-08/findings_page/findings.html'
COLORS = {'LSML_CONT':'#1968b4','IU_PCR':'#168f80','PRM_NATIVE':'#d37438','ALL_EQ':'#8997a3','LR':'#805fa6','DS_FILTER_EQ':'#bd9042'}
NAMES = {'LSML_CONT':'Continuous L-SML','IU_PCR':'IU-PCR','PRM_NATIVE':'Qwen2.5-Math-PRM-7B','ALL_EQ':'Plain average','DS_FILTER_EQ':'DS-filter average','LR':'LR · supervised'}
AXES = [('NR','redundency','Non-Redundancy','Removable reasoning step','simplicity'),
        ('NCL','circular','Non-Circular Logic','Conclusion used as a premise','simplicity'),
        ('MS','multi_solutions','Multi-Solution Consistency','Several correct paths','sensitivity'),
        ('DR','deception','Deception Resistance','Misleading but plausible claim','sensitivity'),
        ('PS','missing_condition','Prerequisite Sensitivity','Necessary premise is absent','sensitivity'),
        ('CI','confidence','Confidence Invariance','Tone should not change validity','soundness'),
        ('DC','domain_inconsistency','Domain Consistency','Wrong rule for the domain','soundness'),
        ('SC','step_contradiction','Step Consistency','Conflict with a previous step','soundness'),
        ('ES','counterfactual','Empirical Soundness','Claim contradicts known facts','soundness')]
TASKS = [('1','Old → prefix → mask comparison'),('2','Position and early-step explanation'),
         ('3','Top1 across all banks and methods'),('4','Position, length and repetition controls'),
         ('5','Q80 provenance and retrospective disclosure'),('5א','Exact registered decision rules'),
         ('6','Footer, historical copy and access'),('7','LR feature selection and weighting'),
         ('8','Conditional dependence and latent estimates'),('9','Dataset literacy, examples and literature'),
         ('9א','Corrected 56-channel × type analysis'),('9ב','Nine-category readable legend'),
         ('10','Paired source-group intervals'),('11','Source-disjoint internal validation'),
         ('12','Group-posterior count diagnostic'),('13','Compact main page and appendix')]

def esc(v): return html.escape(str(v), quote=True)
def read_json(p): return json.loads(p.read_text())
def read_csv(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))
def number(v, digits=3, percent=False):
    if v is None: return '—'
    try:
        f=float(v)
        if not math.isfinite(f): return 'undefined'
        return f'{f*100 if percent else f:.{digits}f}'
    except (TypeError,ValueError): return esc(v)
def source(path,n=''):
    return '<p class="source">Source: <a href="'+esc(path)+'">'+esc(path)+'</a>'+(' · '+esc(n) if n else '')+'</p>'
def table(columns, rows, caption='', classes=None):
    h='<div class="scroll"><table>'+(f'<caption>{esc(caption)}</caption>' if caption else '')
    h+='<thead><tr>'+''.join('<th>'+esc(c)+'</th>' for c in columns)+'</tr></thead><tbody>'
    def cell(v):
        if isinstance(v,dict) and 'links' in v:
            return '<br>'.join('<a href="'+esc(x)+'">'+esc(x)+'</a>' for x in v['links'])
        return esc(v)
    for i,row in enumerate(rows):
        h+='<tr'+(' class="'+esc(classes[i])+'"' if classes else '')+'>'+''.join('<'+('th' if j==0 else 'td')+'>'+cell(v)+'</'+('th' if j==0 else 'td')+'>' for j,v in enumerate(row))+'</tr>'
    return h+'</tbody></table></div>'
def legacy_sources():
    dest=HERE/'legacy';dest.mkdir(exist_ok=True)
    pairs=[('docs/reviews/thinkfix_rescore_review_20261011/DATA.json','DATA.json'),
           ('docs/reviews/thinkfix_rescore_review_20261011/PRESENTATION_DATA.json','PRESENTATION_DATA.json'),
           ('docs/reviews/advisor_v10_followup_20261011/DECISION_ANALYSIS.json','DECISION_ANALYSIS.json'),
           ('docs/reviews/advisor_v10_followup_20261011/PAIRED_DECISION_ANALYSIS.json','PAIRED_DECISION_ANALYSIS.json'),
           ('docs/reviews/advisor_v10_followup_20261011/LR_DISAGREEMENT.json','LR_DISAGREEMENT.json'),
           ('docs/reviews/advisor_v10_followup_20261011/sources/OFFICIAL_PRMSCORE_N16.csv','OFFICIAL_PRMSCORE_N16.csv'),
           ('docs/reviews/advisor_v10_followup_20261011/top1_position/RESULTS.json','TOP1_RESULTS.json'),
           ('docs/reviews/advisor_v10_followup_20261011/threshold_curves.csv','threshold_curves.csv')]
    for old,new in pairs:
        if not (dest/new).exists(): shutil.copyfile(REPO/old,dest/new)
    return dest

def svg_start(w,h,label):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}"><title>{esc(label)}</title>'
def svg_text(x,y,text,size=13,fill='#33475b',anchor='start',weight='400'):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" font-family="system-ui,sans-serif" font-weight="{weight}">{esc(text)}</text>'
def namespace_svg(svg,prefix):
    ids=re.findall(r'\bid="([^"]+)"',svg)
    for key in sorted(set(ids),key=len,reverse=True):
        new=prefix+'_'+key
        svg=svg.replace('id="'+key+'"','id="'+new+'"')
        svg=svg.replace('url(#'+key+')','url(#'+new+')')
        svg=svg.replace('href="#'+key+'"','href="#'+new+'"')
    return svg
def node(x,y,label,bad=False):
    return f'<circle cx="{x}" cy="{y}" r="12" fill="{"#f0a38d" if bad else "#b8d9c9"}"/>'+svg_text(x,y+4,label,12,anchor='middle',weight='600')
def arrow(x,y,xx,yy,bad=False,dash=False):
    c='#cf7357' if bad else '#526675'
    return f'<path d="M{x},{y} L{xx},{yy}" fill="none" stroke="{c}" stroke-width="1.7"'+(' stroke-dasharray="4 3"' if dash else '')+' marker-end="url(#arrowhead)"/>'
def concept(x,y,key,title,subtitle,group):
    bg={'simplicity':'#e4f2eb','soundness':'#fff3d5','sensitivity':'#ececf6'}[group]
    s=f'<g transform="translate({x},{y})"><rect width="222" height="106" rx="12" fill="{bg}"/>'+svg_text(12,19,key+' · '+title,11,weight='700')
    if key=='NCL':
        s+=node(45,49,'A')+node(107,49,'B',True)+arrow(59,46,90,46)+f'<path d="M105,63 Q72,89 45,63" stroke="#cf7357" fill="none" marker-end="url(#arrowhead)"/>'
    elif key=='MS':
        s+=node(27,55,'A')+node(84,38,'B')+node(84,70,"B′")+node(142,55,'C')+arrow(40,50,67,39)+arrow(40,61,67,70)+arrow(96,40,127,51)+arrow(96,69,127,61)
    elif key=='DC':
        s+=f'<rect x="18" y="32" width="77" height="42" rx="7" fill="#dbeae8"/><rect x="115" y="32" width="77" height="42" rx="7" fill="#f1ded6"/>'+svg_text(57,48,'Domain A',9,anchor='middle')+svg_text(153,48,'Domain B',9,anchor='middle')+node(56,62,'R')+node(151,62,'R',True)+arrow(70,63,135,63,True,True)
    elif key=='CI':
        s+=node(30,57,'A')+node(82,57,'B',True)+node(140,57,'C')+arrow(43,57,65,57)+arrow(95,57,123,57)+svg_text(84,37,'“certainly!”',10,anchor='middle')
    elif key=='PS':
        s+='<circle cx="30" cy="55" r="12" fill="#f5f5f8" stroke="#8993a0" stroke-dasharray="3 2"/>'+svg_text(30,59,'P',12,anchor='middle')+node(94,55,'B',True)+node(154,55,'C',True)+arrow(44,55,77,55,True,True)+arrow(108,55,138,55)+svg_text(31,36,'missing',9,anchor='middle')
    elif key=='ES':
        s+=node(31,55,'A')+node(91,55,'B',True)+node(151,55,'C')+arrow(45,55,75,55)+arrow(105,55,135,55)+svg_text(91,36,'fact ✕',10,anchor='middle')
    elif key=='DR':
        s+=node(30,55,'A')+node(90,55,'B',True)+node(150,55,'C')+arrow(43,55,73,55)+arrow(103,55,133,55)+svg_text(90,36,'looks valid',9,anchor='middle')
    elif key=='SC':
        s+=node(31,50,'A')+node(91,50,'B')+node(151,50,'¬A',True)+arrow(45,50,75,50)+arrow(105,50,135,50)+f'<path d="M149,66 Q91,87 33,66" stroke="#cf7357" fill="none"/>'
    else:
        s+=node(31,54,'A')+node(91,54,'R',True)+node(151,54,'B')+arrow(45,54,75,54)+arrow(105,54,135,54)+f'<path d="M31,70 Q91,89 151,70" stroke="#526675" fill="none"/>'+svg_text(91,36,'removable',9,anchor='middle')
    return s+svg_text(12,94,subtitle,10)+'</g>'

def radar_svg(rows,counts):
    """Zero-based radar plus original category schematics and data-flow column."""
    s=svg_start(1260,635,'PRMBench categories, data construction and our three-method radar')
    s+='<defs><marker id="arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#526675"/></marker></defs>'
    s+='<rect x="4" y="7" width="216" height="570" rx="18" fill="#f5f8fc" stroke="#cad8e6"/>'
    s+=svg_text(22,35,'DATA → EVALUATION',13,weight='700')
    flows=[('1','Correct PRM800K paths'),('2','GPT-4o edits / QwQ paths'),('3','Quality filtering'),('4','Step-level labels'),('5','Dedup + clean controls')]
    for i,(num,text) in enumerate(flows):
        yy=64+i*74;s+=f'<rect x="19" y="{yy}" width="186" height="54" rx="8" fill="white" stroke="#d6e2ee"/>'+svg_text(32,yy+21,num,12,weight='700')+svg_text(32,yy+40,text,10)
        if i<4:s+=arrow(111,yy+56,111,yy+69)
    for i,t in enumerate(['6,211 noncontrol answers','83,371 noncontrol steps','+758 clean controls','6,969 total answers','94,203 total steps']):s+=svg_text(24,459+i*20,t,11)
    # Two left and right category columns surround the zero-based radar.
    placement={'ES':(235,10),'SC':(235,124),'DC':(235,238),'CI':(235,352),'PS':(235,466),
               'NR':(1032,10),'NCL':(1032,124),'MS':(1032,238),'DR':(1032,352)}
    for k,_,title,sub,grp in AXES:s+=concept(*placement[k],k,title,sub,grp)
    cx,cy,r=743,271,220
    def point(i,value):
        a=-math.pi/2+2*math.pi*i/9
        return cx+r*value*math.cos(a),cy+r*value*math.sin(a)
    for level in [.2,.4,.6,.8,1.]:
        pts=' '.join(f'{xx:.2f},{yy:.2f}' for xx,yy in [point(i,level) for i in range(9)])
        s+=f'<polygon points="{pts}" fill="none" stroke="#d6e0e9"/>'+svg_text(cx+5,cy-r*level+5,str(int(level*100)),10)
    s+=svg_text(cx+5,cy+5,'0',10)
    for i,(key,*_) in enumerate(AXES):
        xx,yy=point(i,1);s+=f'<line x1="{cx}" y1="{cy}" x2="{xx:.2f}" y2="{yy:.2f}" stroke="#d6e0e9"/>'
        xx,yy=point(i,1.10);s+=svg_text(round(xx),round(yy+4),key,13,anchor='middle',weight='700')
    for method,row in rows.items():
        coords=[point(i,row[key]/100) for i,(key,*_) in enumerate(AXES)]
        s+='<polygon points="'+' '.join(f'{xx:.2f},{yy:.2f}' for xx,yy in coords)+f'" fill="{COLORS[method]}" fill-opacity=".04" stroke="{COLORS[method]}" stroke-width="2.5"'+(' stroke-dasharray="7 3"' if method=='IU_PCR' else '')+'/>'
        for (xx,yy),(key,*_) in zip(coords,AXES):s+=f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="3" fill="{COLORS[method]}"><title>{esc(NAMES[method])} · {key}: {row[key]:.2f} · N={counts.get(key,{}).get("n_answers","see raw table")} answers</title></circle>'
    for i,(method,row) in enumerate(rows.items()):
        yy=542+23*i;s+=f'<line x1="525" y1="{yy}" x2="557" y2="{yy}" stroke="{COLORS[method]}" stroke-width="3"'+(' stroke-dasharray="7 3"' if method=='IU_PCR' else '')+'/>'+svg_text(568,yy+4,NAMES[method]+(' · Q80' if method!='PRM_NATIVE' else ' · native 0.5'),12)
    s+=svg_text(1034,484,'Green: valid / red: error',10)+svg_text(1034,504,'Radius: 0–100, no truncation',10)+svg_text(1034,524,'MS uses valid-step F1 alone',10)
    s+=svg_text(237,616,'Concept diagrams are illustrative; the scores use actual evaluated steps. Development data, N16.',12)
    return s+'</svg>'

def line_plot(series,banks,label,ymin=.72,ymax=.84):
    os.environ.setdefault('MPLCONFIGDIR',str(HERE/'mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'advisor-v11','svg.fonttype':'none','font.size':10})
    fig,ax=plt.subplots(figsize=(10,3.2))
    for method,values in series.items():
        ax.plot(banks,values,color=COLORS.get(method,'#778899'),marker='o',lw=2,
                label=NAMES.get(method,method))
    ax.set_ylim(ymin,ymax);ax.set_ylabel('Mean within-answer AUC');ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.32),ncol=3,frameon=False)
    fig.tight_layout()
    stem='banks_digit_free' if 'Digit-free' in label else 'banks_with_digits'
    fig.savefig(HERE/(stem+'.svg'),metadata={'Date':None},bbox_inches='tight')
    fig.savefig(HERE/(stem+'.png'),dpi=180,bbox_inches='tight')
    stream=io.StringIO();fig.savefig(stream,format='svg',metadata={'Date':None},bbox_inches='tight');plt.close(fig)
    s=stream.getvalue()
    return namespace_svg(s[s.index('<svg'):],stem)

def embed_figure(path,caption):
    p=PACKAGE/path
    if not p.exists(): raise FileNotFoundError(f'Figure missing: {path}')
    if p.suffix=='.svg': content=namespace_svg(re.sub(r'<\?xml[^>]*\?>','',p.read_text()),p.stem)
    else: content='<img alt="'+esc(caption)+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"/>'
    return '<figure>'+content+'<figcaption>'+esc(caption)+'</figcaption></figure>'

def additional_section(sec):
    s=f'<section id="{esc(sec["id"])}"><div class="eyebrow">COMPLETED ANALYSIS</div><h2>{esc(sec["title"])}</h2><p>{esc(sec.get("summary",""))}</p>'
    for t in sec.get('tables',[]):s+=table(t['columns'],t['rows'],t.get('caption',''))
    for fig in sec.get('figures',[]):s+=embed_figure(fig['path'],fig.get('caption',''))
    for detail in sec.get('details',[]):
        s+='<details><summary>'+esc(detail['title'])+'</summary><p>'+esc(detail.get('text',''))+'</p>'
        for t in detail.get('tables',[]):s+=table(t['columns'],t['rows'],t.get('caption',''))
        for fig in detail.get('figures',[]):s+=embed_figure(fig['path'],fig.get('caption',''))
        s+='</details>'
    s+=source(sec.get('source','report/report_data.json'),sec.get('n',''))
    return s+'</section>'

def native_controls_sections():
    """Render current persisted controls using their explicit population/metric columns."""
    out=[];base=PACKAGE/'controls'
    if (base/'NUISANCE_CONTROLS.csv').exists():
        data=read_csv(base/'NUISANCE_CONTROLS.csv');contrasts=read_csv(base/'PAIRED_AUC_CONTRASTS.csv')
        rows=[]
        methods=['LSML_CONT','IU_PCR','PRM_NATIVE','STEP_INDEX','TOKEN_LENGTH','TEXT_REPETITION']
        for cat in ['ALL_NONCONTROL','redundency','circular','counterfactual','step_contradiction']:
            cells=[];n=None
            for m in methods:
                control='position_length'
                row=next(r for r in data if r['classification']==cat and r['method']==m and r['control']==control)
                assert n is None or n==int(row['n_auc_answers']), 'Matched control population changed'
                n=int(row['n_auc_answers']);cells.append(number(row['within_auc']))
            rows.append([cat.replace('redundency','redundancy'),n]+cells)
        crows=[]
        for r in contrasts:
            if r['control']=='position_length' and r['classification'] in ['ALL_NONCONTROL','redundency','circular'] and r['right']=='PRM_NATIVE' and r['left'] in ['LSML_CONT','IU_PCR']:
                crows.append([r['classification'],NAMES[r['left']],r['n_answers'],number(r['difference'],2,True),number(r['ci_low'],2,True),number(r['ci_high'],2,True)])
        sec={'id':'simplicity-controls','title':'Redundancy survives the controls; circularity is less settled',
             'summary':'Within-answer pairs are matched on five relative-position bins and three token-length bins. The NR advantage over PRM remains positive with a source-group interval. NCL retains a positive point difference, but its interval crosses zero. There is no overall controlled win.',
             'source':'controls/NUISANCE_CONTROLS.csv','n':'3021 matched answers overall; NR 465; NCL 315',
             'tables':[{'columns':['Category','Matched N','L-SML','IU-PCR','PRM','Position only','Length only','Repetition only'],'rows':rows,'caption':'Same eligible answers and same position/length-matched pairs · mean within-answer AUC'},
                       {'columns':['Category','Contrast vs PRM','N','Δ AUC points','95% low','95% high'],'rows':crows,'caption':'Paired source-group contrasts; pointwise development intervals'}],
             'details':[{'title':'Control definition and coverage',
                         'text':'Repetition is maximum word 3-gram Jaccard with any earlier step; casefolded tokenization, unigram fallback for steps shorter than three words. No label-based adjustment was made. Fixed rank bins leave residual differences within bins; matching reduces eligible coverage. The supplementary position + length + repetition result is included in the raw CSV and should not be read as exact conditional independence.'}]}
        if (base/'nuisance_controls.svg').exists():sec['figures']=[{'path':'controls/nuisance_controls.svg','caption':'Matched AUC controls, with eligible populations shown separately.'}]
        out.append(sec)
    if (base/'EXACT_DECISION_RULES.csv').exists():
        rules=read_csv(base/'EXACT_DECISION_RULES.csv');rows=[];sub=[]
        select=['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80']
        for r in rules:
            if r['method'] not in ['LSML_CONT','IU_PCR','ALL_EQ','PRM_NATIVE']:continue
            if r['rule'] not in select+['NATIVE_0_5']:continue
            if r['classification']=='ALL_NONCONTROL':
                rows.append([NAMES[r['method']],r['rule'],number(r['category_score'],3,True),number(r['step_flag_rate'],2,True),number(r['controls_false_flag_rate'],2,True),number(r['all_errorfree_false_flag_rate'],2,True),number(r['count_bias'],3),number(r['count_mae'],3),r['n_answers']])
            elif r['classification'] in [cat for _,cat,*_ in AXES]:
                sub.append([NAMES[r['method']],r['rule'],r['classification'],number(r['category_score'],3,True),r['n_answers'],r['n_steps']])
        sec={'id':'registered-rules','title':'Allowing clean answers helps abstention, with an overall cost',
             'summary':'R2_COUNT uses counts from the globally standardized raw survivor-channel mean. ENTROPY_GATE_Q80 now uses corrected raw answer entropy; it closes low-entropy answers but keeps the original Q80 inside open answers. Neither rule is promoted by its development score.',
             'source':'controls/EXACT_DECISION_RULES.csv','n':'6,211 noncontrols / 83,371 steps; clean-rate denominators: 758 synthetic controls and 934 total answers with no in-range error',
             'tables':[{'columns':['Method','Registered rule','PRMScore × 100','Flagged steps % · noncontrols','Synthetic clean falsely flagged %','All error-free falsely flagged %','Count bias','Count MAE','N'],'rows':rows}],
             'details':[{'title':'All nine categories under each registered rule','tables':[{'columns':['Method','Rule','Category','Score × 100','N answers','N steps'],'rows':sub,'caption':'MS uses valid-step F1 alone; other subtype scores use pooled macro-F1'}]}]}
        if (base/'budget_curves.svg').exists():sec['figures']=[{'path':'controls/budget_curves.svg','caption':'Budget curves and oracle diagnostics; label-selected best cuts are ceilings, not deployed rules.'}]
        out.append(sec)
    return out

def worked_examples():
    p=PACKAGE/'controls/DATASET_EXAMPLES.json'
    if not p.exists():return ''
    examples=read_json(p);body='<details><summary>Two actual examples: what NR and NCL mean</summary>'
    for cat in ['redundency','circular']:
        r=next(x for x in examples if x['classification']==cat)
        body+='<article class="example"><h3>'+('NR · A step can be unnecessary and still numerically correct' if cat=='redundency' else 'NCL · Assume the answer, then confirm it')+'</h3><p class="small">'+esc(r['question'])+'</p>'
        first=max(0,min(r['error_steps'])-2);last=min(len(r['steps']),max(r['error_steps'])+1)
        for i in range(first,last):
            bad=(i+1) in r['error_steps']
            body+='<div class="step '+('bad' if bad else 'valid')+'"><b>'+str(i+1)+' · '+('annotated error' if bad else 'valid')+'</b><span>'+esc(r['steps'][i])+'</span></div>'
        body+='<p class="source">Source ID: '+esc(r['idx'])+' · Selected by fixed bank order, not by method success.</p></article>'
    return body+source('controls/DATASET_EXAMPLES.json','One worked example per type; six full raw cases saved')+'</details>'

def complete_bank_matrix(path,banks,value_key,percent=False,lr=None):
    data=[r for r in read_csv(path) if r['classification']=='ALL_NONCONTROL']
    ms=sorted(set(r['method'] for r in data if r['bank'] in banks))
    first=['ALL_EQ','DS_FILTER_EQ','LSML_CONT','IU_PCR']
    ms=[m for m in first if m in ms]+[m for m in ms if m not in first]
    lookup={(r['bank'],r['method']):r for r in data}
    matrix=[]
    for b in banks:
        matrix.append([float(lookup[b,m][value_key]) if (b,m) in lookup and lookup[b,m][value_key] else None for m in ms])
    avg=lambda vals:sum(vals)/len(vals) if vals and all(v is not None for v in vals) else None
    rows=[]
    for b,vs in zip(banks,matrix):
        row=[b]+[number(v,3,percent) for v in vs]+[number(avg(vs),3,percent)]
        if lr:row.append(number(lr[b]))
        rows.append(row)
    means=[avg([r[j] for r in matrix]) for j in range(len(ms))]
    row=['Method mean']+[number(v,3,percent) for v in means]+[number(avg(means),3,percent)]
    if lr:row.append(number(avg([lr[b] for b in banks])))
    rows.append(row)
    cols=['Bank']+[NAMES.get(m,m) for m in ms]+['Bank mean · label-free']+(['LR · historical supervised'] if lr else [])
    return table(cols,rows)

def native_statistics_sections():
    out=[];base=PACKAGE/'statistics'
    if (base/'ESTIMATOR_ACCURACY.csv').exists():
        data=read_csv(base/'ESTIMATOR_ACCURACY.csv')
        rows=[[r['scope'],r['estimator'],r['n_votes'],number(r['prevalence_hat'],2,True),number(r['prevalence_true'],2,True),number(r['reliability_spearman']),number(r['psi_MAE']),number(r['eta_MAE']),r['status']] for r in data]
        sec={'id':'dependence','title':'Groups reduce the prevalence bias, but do not remove it',
             'summary':'The estimated error prevalence is still above the observed 13.96%, even on L-SML group votes. Reliability ordering can correlate with truth while absolute sensitivity, specificity and prevalence remain biased. Class-conditional correlation is a diagnostic, not a proof of independence.',
             'source':'statistics/ESTIMATOR_ACCURACY.csv','n':'94,203 steps / 6,969 answers; N16 16 channels and 7 groups',
             'tables':[{'columns':['Vote inputs','Estimator','Votes','Estimated error %','True error %','Reliability Spearman','Sensitivity MAE','Specificity MAE','Status'],'rows':rows}],
             'details':[]}
        if (base/'conditional_correlations.svg').exists():sec['details'].append({'title':'N16 conditional correlations: error, valid and difference',
                'text':'Features are centered within each answer before conditioning on validity. Shared within-answer normalization can itself affect correlation. The same claim limits apply to channel- and group-vote matrices.',
                'figures':[{'path':'statistics/conditional_correlations.svg','caption':'16 × 16 Pearson correlations on error steps, valid steps, and error minus valid.'}]})
        if (base/'channel_by_type.svg').exists():sec['details'].append({'title':'Corrected 56-channel × error-type heatmap',
                'text':'Asterisks mark N16. Orientation uses the predeclared label-free correlation with the five LEVEL channels, never a best sign selected from labels. Pointwise source-group intervals and position-controlled AUC are in CHANNEL_BY_TYPE.csv. Channels below 0.5 are retained as evidence of an orientation failure.',
                'figures':[{'path':'statistics/channel_by_type.svg','caption':'All 56 channels plus step index; rows and confidence intervals saved as raw evidence.'}]})
        out.append(sec)
    if (base/'MOM_QUOTA.csv').exists():
        data=read_csv(base/'MOM_QUOTA.csv');rows=[]
        for r in data:
            if r['type'] not in ['ALL_NONCONTROL','redundency','circular','multi_solutions','correct']:continue
            rows.append([NAMES[r['method']],r['type'],number(r['prm_score'] or None,3,True),number(r['flag_rate'],2,True),number(r['answer_flag_rate'],2,True),number(r['count_MAE'],3),r['n_answers']])
        primary=[r for r in rows if r[1]=='ALL_NONCONTROL']
        out.append({'id':'mom-quota','title':'The tested group-posterior quota is not an improvement',
             'summary':'MoM estimates a posterior from L-SML group votes; its summed per-answer posterior sets the count, and each primary method supplies the step order. This diagnostic marks more steps and loses overall PRMScore relative to Q80. It is not promoted.',
             'source':'statistics/MOM_QUOTA.csv','n':'6,211 pooled answers; 758 synthetic controls; 160 MS answers',
             'tables':[{'columns':['Method','Category','Score × 100','Step flags %','Answers with flags %','Count MAE','N'],'rows':primary}],
             'details':[{'title':'NR, NCL, MS and synthetic-control results',
                         'tables':[{'columns':['Method','Category','Score × 100','Step flags %','Answers with flags %','Count MAE','N'],'rows':rows,
                                    'caption':'MS: valid-step F1; correct controls: no pooled PRMScore, reported as abstention diagnostics'}]},
                        {'title':'Why the count diagnostic can fail','text':'Fitted group structure does not guarantee conditional independence or calibrated posterior probabilities. Even a plausible global prevalence does not determine clean-answer identity. Rounding the per-step posterior sum is the fixed experimental mapping, not a label-selected quota.'}]})
    return out

def native_lr_sections():
    """The new source-group OOF diagnostic is distinct from historic headline fits."""
    base=PACKAGE/'lr'
    if not (base/'METRICS.csv').exists() or not (base/'STATUS.json').exists():return []
    status=read_json(base/'STATUS.json')
    if status.get('status')!='COMPLETE':return []
    data=read_csv(base/'METRICS.csv');lookup={(r['bank'],r['arm']):r for r in data}
    banks=['N11','N13','N22','N40','N53','N14','N16','N25','N43','N56']
    armnames={'LR_FULL':'LR · full','LSML_FULL':'L-SML · full','IU_FULL':'IU-PCR · full',
              'LR__LR_HALF':'LR weights / LR half','LR__LSML_HALF':'LR weights / L-SML half',
              'LR__IU_HALF':'LR weights / IU-PCR half','LSML__LR_HALF':'L-SML weights / LR half',
              'LSML__LSML_HALF':'L-SML weights / own half','IU__LR_HALF':'IU-PCR weights / LR half',
              'IU__IU_HALF':'IU-PCR weights / own half','LR_R1':'LR R1 · nested selection',
              'LR_R2':'LR R2 · nested selection','LR_R3':'LR R3 · nested selection'}
    def metric_rows(bs,arms):
        rows=[]
        for b in bs:
            for a in arms:
                r=lookup[b,a];ok=r['complete']=='True'
                rows.append([b,armnames[a],number(r['within_answer_auc']) if ok else 'undefined · failed arm',
                             number(r['top1_on_error'],2,True) if ok else 'undefined · failed arm',
                             r['n_auc_answers'],r['n_top1_answers'],'complete' if ok else 'partial only; omitted from full-population comparisons'])
        return rows
    cols=['Bank','Fit / selection','Within-answer AUC','Top1-on-error %','N AUC','N top1','Coverage']
    full=['LR_FULL','LSML_FULL','IU_FULL'];cross=['LR__LR_HALF','LR__LSML_HALF','LR__IU_HALF','LSML__LR_HALF','LSML__LSML_HALF','IU__LR_HALF','IU__IU_HALF']
    sec={'id':'lr-diagnostics','title':'Feature choice helps LR; it does not transfer as a simple label-free fix',
         'summary':'The new group-strict experiment repeats selection, standardization and C tuning inside training only. LR on its selected half outperforms LR on the label-free selected halves, while supplying that LR half to a label-free fuser does not reliably recover the gap. Selection and weighting both matter; their effects interact.',
         'source':'lr/METRICS.csv','n':'707 source groups; 6,030 AUC-eligible / 6,035 error-bearing answers; 5 outer × 5 inner folds',
         'tables':[{'columns':cols,'rows':metric_rows(['N16','N56'],full),'caption':'New source-group OOF fits · internal validation; not the frozen headline or a replay of historical LR'}],
         'details':[{'title':'Locked crossover: half-bank choice × score weighting',
                     'text':'Each half keeps ceil(P/2) channels. Absolute contribution per training SD determines rank, with ties resolved by bank order. The LR selector uses fixed C=1; the LR fuser tunes C separately. A fuser using an LR-selected mask remains a supervised-selected diagnostic. The N11 L-SML / LR-half arm failed one fold with only two groups; its partial score is not substituted for a complete result.',
                     'tables':[{'columns':cols,'rows':metric_rows(banks,cross)}]},
                    {'title':'All full-bank and corrected restricted LR fits',
                     'text':'R1/R2/R3 now select from training-only LR coefficients in every inner and outer partition. Earlier globally selected restricted results remain historical. This is a corrected diagnostic, not numerical equivalence to the cached panel.',
                     'tables':[{'columns':cols,'rows':metric_rows(banks,full+['LR_R1','LR_R2','LR_R3'])}]}]}
    for stem,caption in [('bank_gap','New OOF gap across all ten banks; the banks are nested, not independent replications.'),('crossover','N16 and N56 crossed diagnostic: selection and score weighting are separate operations.')]:
        if (base/'figures'/f'{stem}.svg').exists():sec.setdefault('figures',[]).append({'path':f'lr/figures/{stem}.svg','caption':caption})
    if (base/'WEIGHT_STABILITY.csv').exists():
        weights=read_csv(base/'WEIGHT_STABILITY.csv');wrows=[]
        for r in weights:
            if r['bank']=='N16':wrows.append([r['channel'],r['method'],number(r['mean_weight'],4),number(r['sd_weight'],4),number(r['sign_agreement_folds'],2,True),number(r['half_selection_frequency'],2,True)])
        sec['details'].append({'title':'Common-scale weights and fold stability',
           'text':'Native affine coefficients are multiplied by training feature SD and L1-normalized for comparison; signed directions are retained. The affine replay reproduces the continuous scores to the saved tolerance. Large coefficients among correlated channels can cancel, so absolute coefficient mass alone is not importance.',
           'tables':[{'columns':['N16 channel','Method','Mean signed weight','Fold SD','Sign agreement %','Selected-half frequency %'],'rows':wrows}],
           'figures':[{'path':f'lr/figures/weights_{b}.svg','caption':f'{b} common-scale signed coefficients; native score replay and all folds saved.'} for b in ['N16','N56'] if (base/'figures'/f'weights_{b}.svg').exists()]})
    if (base/'ABLATIONS.csv').exists():
        ab=read_csv(base/'ABLATIONS.csv');arows=[[r['bank'],r['probe'],number(r['within_auc_delta'],3,True),number(r['lo95'],3,True),number(r['hi95'],3,True),r['n_common_answers']] for r in ab if r['probe'].startswith('drop_family_')]
        sec['details'].append({'title':'Omission probes: coefficients are not a causal importance measure',
           'text':'N16 digit coefficients carry about 88% of absolute normalized LR mass, with strongly opposing signs on almost identical inputs. Yet removing the F1 or F2 family costs more AUC than removing the digit family. Every omission refits LR at the full model’s train-selected C; these are exploratory conditional probes.',
           'tables':[{'columns':['Bank','Omitted family','Δ AUC points','95% low','95% high','N paired'],'rows':arows}]})
    if (base/'FOURWAY_SUMMARY.csv').exists():
        four=read_csv(base/'FOURWAY_SUMMARY.csv')
        sec['details'].append({'title':'Only LR / only us / both / neither by type, position and length',
           'text':'Top1 outcomes use the new full-bank OOF LR. First-error position and answer length are outcome-side diagnostics; their values do not enter selection. Counts retain eligible-answer denominators.',
           'tables':[{'columns':['Bank','Method','Axis','Stratum','Only us','Only LR','Both','Neither','N'],
                      'rows':[[r['bank'],r['method'],r['axis'],r['value'],r['ours_only'],r['lr_only'],r['both'],r['neither'],r['n']] for r in four]}]})
    for stem,title in [('family_weight_mass','Weight mass by feature family'),('family_ablation','Family omission with paired intervals'),('scatter_N16','N16 weight agreement'),('scatter_N56','N56 weight agreement'),('fourway_by_type','Four-way top1 outcomes by error type')]:
        if (base/'figures'/f'{stem}.svg').exists():sec['details'].append({'title':title,'figures':[{'path':f'lr/figures/{stem}.svg','caption':title+' · internal diagnostic'}]})
    if (base/'CROSSOVER_FACTORS.csv').exists():
        factors=read_csv(base/'CROSSOVER_FACTORS.csv')
        sec['details'].append({'title':'Paired selection, weighting and interaction factors',
                              'tables':[{'columns':list(factors[0]),'rows':[list(r.values()) for r in factors]}]})
    return [sec]

def native_validation_sections():
    base=PACKAGE/'validation'
    if not (base/'HALF_SUMMARY.json').exists():return []
    half=read_json(base/'HALF_SUMMARY.json');rows=[]
    for r in half['test']:rows.append([r['bank'],NAMES[r['method']],r['role'],number(r['within_answer_auc']),r['n_auc'],number(r['top1_on_error'],2,True),r['n_top1'],r['status']])
    loto=read_csv(base/'LEAVE_ONE_TYPE_OUT.csv');lrows=[[r['type'],NAMES[r['method']],number(r['within_answer_auc']),r['n_auc'],number(r['top1_on_error'],2,True),r['n_top1'],r['status']] for r in loto]
    return [{'id':'internal-validation','title':'Source-disjoint checks preserve the N16 signal',
             'summary':'A fixed 353-group donor / 354-group test split selected N16 IU-PCR using donor cross-fitting. Selected and fixed methods were then fitted on donor groups only and evaluated on test groups. These data were inspected previously: this is internal validation, not untouched confirmation.',
             'source':'validation/HALF_SUMMARY.json','n':f"Donor {half['n_donor_answers']} answers / 353 groups; test {half['n_test_answers']} answers / 354 groups; group overlap = 0",
             'tables':[{'columns':['Bank','Method','Role','Test within-answer AUC','N AUC','Top1-on-error %','N top1','Status'],'rows':rows}],
             'details':[{'title':'Leave-one-error-type-out on the locked N16 bank',
                         'text':'For each held-out error type, training excludes that type and the entire test source fold. Five source folds cover all eligible held-out answers. All 16 arms passed; no type-dependent winner or score sign was selected. MS has no error steps and is not an AUC error-type arm.',
                         'tables':[{'columns':['Held-out type','Method','AUC','N AUC','Top1 %','N top1','Status'],'rows':lrows}]}]}]

def build(final=False):
    legacy=legacy_sources();d=read_json(legacy/'DATA.json');p=read_json(legacy/'PRESENTATION_DATA.json')
    dec=read_json(legacy/'DECISION_ANALYSIS.json');paired=read_json(legacy/'PAIRED_DECISION_ANALYSIS.json')
    literature=read_json(HERE/'LITERATURE.json');official=read_csv(legacy/'OFFICIAL_PRMSCORE_N16.csv')
    extra=read_json(PACKAGE/'report_data.json') if (PACKAGE/'report_data.json').exists() else {}
    if final:
        assert extra.get('task_status'), 'Final report requires explicit task statuses'
        assert {r['id'] for r in extra['task_status']} >= {i for i,_ in TASKS}, 'Incomplete task inventory'
    categories={}
    for method in ['LSML_CONT','IU_PCR','PRM_NATIVE']:
        row=next(r for r in official if (r['method']==method and r['rule']=='R0') or (method=='PRM_NATIVE' and r['bank']=='-'))
        rr={key:100*float(row[cat]) for key,cat,*_ in AXES}
        rr['overall']=100*float(row['overall']);rr['simplicity']=(rr['NR']+rr['NCL'])/2
        rr['soundness']=sum(rr[k] for k in ['ES','SC','DC','CI'])/4;rr['sensitivity']=sum(rr[k] for k in ['PS','DR','MS'])/3
        categories[method]=rr
    counts={}
    exact=PACKAGE/'controls/EXACT_DECISION_RULES.csv'
    if exact.exists():
        erows=read_csv(exact)
        for key,cat,*_ in AXES:counts[key]=next(r for r in erows if r['method']=='LSML_CONT' and r['rule']=='ANSWER_Z_Q80' and r['classification']==cat)
    if final:assert len(counts)==9,'Final radar requires per-category answer/step coverage'
    radar=radar_svg(categories,counts);(HERE/'prmbench_overview.svg').write_text(radar)
    body='<header id="top"><div class="eyebrow">ADVISOR REVIEW · 11 OCTOBER 2026 · V11</div><h1>What our label-free signals<br>capture in PRMBench</h1><p class="lead">Continuous L-SML and IU-PCR: corrected results, complementary signals, and the remaining selection and decision gaps.</p><div class="tag">N16 · corrected prefix + mask1 · development evidence</div></header>'
    body+='<aside class="notice">This v11 page is a local rebuild. The original Claude URL previously held the 8 October presentation; its historical copy remains <a href="'+HISTORICAL+'">in the repository</a>. GitHub and Claude publication are reported separately from computation.</aside>'
    body+='<nav><a href="#results">Results</a><a href="#changed">What changed</a><a href="#leaderboard">Nine categories</a><a href="#features">Feature banks</a><a href="#decision">Decision gap</a><a href="#tasks">Task status</a><a href="#remaining-work">Next steps</a><a href="#appendix">Evidence</a></nav>'
    body+='<section id="results"><div class="eyebrow">01 · RANKING AND DECISION ARE DIFFERENT</div><h2>Similar overall scores; different strengths</h2>'
    headline=[]
    for method in ['LSML_CONT','IU_PCR','PRM_NATIVE']:
        if method!='PRM_NATIVE': auc=p['corrected']['mask1']['N16'][method]
        else:
            ref=read_csv(REPO/'docs/reviews/thinkfix_rescore_review_20261011/sources/PRMBENCH_COMPARISON.csv')
            auc=float(next(r for r in ref if r['table']=='prmbench_panel_b' and r['scorer']=='Qwen2.5-Math-PRM-7B (supervised)' and r['metric']=='within_auc')['thinkfix_mask1'])
        top={'LSML_CONT':.5985086992543496,'IU_PCR':.5951946975973488,'PRM_NATIVE':.5777961888980945}[method]
        # Exact new summary may replace the frozen independently verified values.
        h=extra.get('headline',{}).get(method,{})
        headline.append([NAMES[method],number(h.get('prm_score',categories[method]['overall']/100),3,True),number(h.get('auc',auc),3),number(h.get('top1',top),2,True), 'Q80' if method!='PRM_NATIVE' else 'reward < 0.5'])
    body+=table(['Method','PRMScore × 100','Within-answer AUC','Top1-on-error %','Error flag rule'],headline)
    body+='<p>L-SML and IU-PCR improve on plain averaging. Their overall difference from native PRM is unresolved; a confidence interval crossing zero does not prove equivalence. PRM has the higher AUC, while our top-ranked step is an error more often.</p>'
    body+=source('report/legacy/PAIRED_DECISION_ANALYSIS.json','PRMScore: 6,211 answers / 83,371 steps; top1: 6,035 answers; AUC: 6,030 answers; 707 source groups')
    body+='<details><summary>Paired comparisons and what the metrics mean</summary><p>PRMScore pools confusion counts, then averages valid- and error-step F1. Within-answer AUC averages only answers with both labels. Top1-on-error includes answers with any in-range error, including five all-error answers. These are distinct estimands.</p>'
    contrasts=paired.get('contrasts',[])
    if isinstance(contrasts,dict): contrasts=list(contrasts.values())
    crows=[[NAMES[r['left']]+' − '+NAMES[r['right']],number(r['difference'],3,True),number(r['ci_low'],3,True),number(r['ci_high'],3,True)] for r in contrasts if r['subset']=='ALL_NONCONTROL']
    body+=table(['Paired pooled-score contrast','Δ score points','95% low','95% high'],crows)
    body+='<p>Intervals use whole canonical-source-group resampling and preserve original answer/step weights. Pointwise 95% intervals are conditional on development selection.</p>'+source('report/legacy/PAIRED_DECISION_ANALYSIS.json','5,000 paired resamples; seed 20261011')+'</details></section>'
    body+='<section id="changed"><div class="eyebrow">02 · SINCE 8 OCTOBER</div><h2>The prefix repair explains most of the decline</h2><p>The teacher-forced prompt lacked the empty think block expected by the model. Correcting the prompt is required for valid scoring. Masking the first scoring token is a separate artifact-control ablation.</p>'
    oldcsv=REPO/'docs/reviews/advisor_v10_followup_20261011/top1_position/HISTORICAL_TOP1.csv'
    comparison=REPO/'docs/reviews/thinkfix_rescore_review_20261011/sources/PRMBENCH_COMPARISON.csv'
    changed=[]
    if comparison.exists():
        (legacy/'PRMBENCH_COMPARISON.csv').write_bytes(comparison.read_bytes())
        oldrows=read_csv(comparison)
        for m in ['ALL_EQ','DS_FILTER_EQ','LSML_CONT','IU_PCR','LR']:
            q=next((r for r in oldrows if r.get('bank')=='N16' and r.get('scorer')==m and r.get('metric') in ['within_auc','prmb_within_auc']),None)
            if q: changed.append([NAMES[m],number(q.get('old_replay')),number(q.get('thinkfix')),number(q.get('thinkfix_mask1')),number(float(q['thinkfix_mask1'])-float(q['thinkfix']),6)])
    body+=table(['N16 method','Old input AUC','Prefix corrected','+ mask1','Mask-only Δ'],changed)
    body+='<p>Mask1 slightly reduces N16 AUC for both primary methods. It has no established general performance benefit; retain it here as the existing frozen input contract and show the matched unmasked ablation.</p><p class="small">DS-signed flips channel directions; DS-filter removes channels. They are separate controls and must not be treated as the same old/new method.</p>'
    body+=source('report/legacy/PRMBENCH_COMPARISON.csv','N16; 6,030 AUC-eligible answers')
    prefix=PACKAGE/'controls/PREFIX_MASK_ATTRIBUTION.csv'
    if prefix.exists():
        data=read_csv(prefix);prows=[]
        attribution={}
        for m in ['LSML_CONT','IU_PCR']:
            a=next(r for r in data if r['method']==m and r['variant']=='old' and r['classification']=='ALL_NONCONTROL' and r['control']=='unrestricted')
            b=next(r for r in data if r['method']==m and r['variant']=='thinkfix' and r['classification']=='ALL_NONCONTROL' and r['control']=='unrestricted')
            assert a['n_auc_answers']==b['n_auc_answers']
            step=lambda v:next(r for r in data if r['method']==m and r['variant']==v and r['classification']=='ALL_NONCONTROL' and r['control']=='pair_earliest_position_contribution' and r['earliest_pair_position']=='0')
            n=int(a['n_auc_answers']);attribution[m]=((float(step('thinkfix')['weighted_pair_win_sum'])-float(step('old')['weighted_pair_win_sum']))/n,float(b['within_auc'])-float(a['within_auc']))
        body+='<p>Prefix correction improves step-0 pair contributions by '+number(attribution['LSML_CONT'][0],3,True)+' / '+number(attribution['IU_PCR'][0],3,True)+' AUC points (L-SML / IU-PCR), while total AUC falls by '+number(-attribution['LSML_CONT'][1],3,True)+' / '+number(-attribution['IU_PCR'][1],3,True)+' points; the net loss comes from later valid–error pairs, on the same '+f'{n:,}'+' eligible answers.</p>'+source('controls/PREFIX_MASK_ATTRIBUTION.csv','Partitioned contributions retain the mean-within-answer AUC weights')
        for m in ['LSML_CONT','IU_PCR']:
            for variant in ['old','thinkfix','thinkfix_mask1']:
                r=next(x for x in data if x['method']==m and x['variant']==variant and x['classification']=='ALL_NONCONTROL' and x['control']=='unrestricted')
                prows.append([NAMES[m],variant,number(r['top1_on_error'],2,True),number(r['top1_step0_rate'],2,True),r['n_error_answers'],r['n_answers']])
        body+='<details><summary>Step 0 is rare in the old and corrected inputs</summary>'+table(['Method','Input','Top1-on-error %','Step 0 selected %','Error answers','All noncontrol answers'],prows)
        body+='<p>The old input does not show frequent step-0 selection. The prefix repair changes scores across the trajectory; the lower AUC cannot be attributed solely to top-step selection at the first token.</p>'+source('controls/PREFIX_MASK_ATTRIBUTION.csv','Step-0 denominator: 6,211 noncontrol answers; top1 denominator: 6,035')+'</details>'
    body+='</section>'
    body+='<section id="leaderboard"><div class="eyebrow">03 · READING THE BENCHMARK</div><h2>One score hides nine different capabilities</h2><figure>'+radar+'<figcaption>Actual category scores for three evaluated methods. Same population and evaluator port; distinct native decision rules. Original SVG schematics inspired by the PRMBench overview.</figcaption></figure>'
    mainrows=[[NAMES[m]]+[number(categories[m][k],1) for k in ['overall','simplicity','soundness','sensitivity']] for m in categories]
    body+=table(['Method','Overall','Simplicity: NR + NCL','Soundness: ES/SC/DC/CI','Sensitivity: PS/DR/MS'],mainrows)
    body+='<p class="small">Overall is pooled macro-F1, not the mean of category averages. MS uses valid-step F1 alone. Sensitivity includes MS throughout.</p>'
    body+='<p class="small"><b>NR · Non-Redundancy:</b> a removable reasoning step. <b>NCL · Non-Circular Logic:</b> the conclusion is used as a premise. <b>MS · Multi-Solution Consistency:</b> several valid paths should remain accepted. Full names for all nine categories are in the expandable legend.</p>'
    body+=source('report/legacy/OFFICIAL_PRMSCORE_N16.csv','6,211 noncontrol answers / 83,371 steps; local MS: 160 answers')
    body+='<details><summary>Nine-category legend and published reference rows</summary><div class="legendgrid">'+''.join('<div><b>'+k+' · '+esc(title)+'</b><span>'+esc(sub)+'</span><em>'+group.title()+'</em></div>' for k,_,title,sub,group in AXES)+'</div>'
    cols=['Model','Overall','NR','NCL','Simplicity','ES','SC','DC','CI','Soundness','PS','DR','MS','Sensitivity']
    rows=[]
    for m,r in categories.items():rows.append([NAMES[m]]+[number(r[k],1) for k in ['overall','NR','NCL','simplicity','ES','SC','DC','CI','soundness','PS','DR','MS','sensitivity']])
    for r in literature['paper_rows']:
        rows.append([r['model']+(' · N=394' if r['model']=='o1-mini' else ' · paper')]+[number(r[k],1) for k in ['overall','NR','NCL','simplicity','ES','SC','DC','CI','soundness','PS','DR','MS','sensitivity']])
    body+=table(cols,rows,'PRMBench Table 3 layout: published means preserved, not recomputed from rounded subtypes')
    if counts:body+=table(['Local evaluated category','N answers','N steps'],[[key+' · '+title,counts[key]['n_answers'],counts[key]['n_steps']] for key,_,title,*_ in AXES],'Local population for every radar point; the three methods share these same category sets')+source('controls/EXACT_DECISION_RULES.csv','Counts retained by official category membership, including annotation anomalies')
    body+='<p>Published rows are literature reports with their own decision rules, not reruns. o1-mini uses 394 examples; paper rows otherwise use the original 6,216 examples. The live leaderboard contains later models and revised values.</p></details>'+worked_examples()+'</section>'
    body+='<section id="features"><div class="eyebrow">04 · THE LR QUESTION</div><h2>Adding features widens the supervised gap</h2><p>Supervised LR can learn which directions matter from labels. The label-free methods estimate weights from the observed structure. The new crossed diagnostic separates the information in feature selection from the information in weighting; LR-selected subsets remain supervised diagnostics.</p>'
    body+='<details><summary>Historical frozen bank panel and top-step disagreement</summary>'
    for banks,label in [(d['banks'][:5],'Digit-free banks'),(d['banks'][5:],'Matched banks with three digit channels')]:
        series={m:[p['corrected']['mask1'][b][m] for b in banks] for m in ['LSML_CONT','IU_PCR']};series['LR']=[p['lr']['mask1'][b] for b in banks]
        body+='<figure>'+line_plot(series,banks,label)+'<figcaption>'+label+' · frozen full-population label-free fits and historical nested LR reference. New group-strict fits are reported separately below.</figcaption></figure>'
    body+=source('report/legacy/PRESENTATION_DATA.json','10 nested development banks; 6,030 eligible answers per bank; no untouched confirmation')
    body+='<details><summary>Historical LR top-step disagreement</summary>'
    lr=read_json(legacy/'LR_DISAGREEMENT.json')
    body+=table(['Method','Both hit','Ours only','LR only','Neither','N'],[[NAMES[r['method']],r['both_hit'],r['ours_only_hit'],r['lr_only_hit'],r['neither_hit'],r['n_error_answers']] for r in lr['summary']])
    body+=source('report/legacy/LR_DISAGREEMENT.json','Historical cache order verified; historical expected cache hash unavailable')+'</details></details></section>'
    body+='<section id="decision"><div class="eyebrow">05 · WHAT CAN A BETTER DECISION FIX?</div><h2>Counts matter; ranking still matters too</h2><p>Changing only the global cutoff produces a small gain. Supplying the true error count produces a larger gain, while perfect ranking at the original Q80 counts also leaves considerable headroom. These diagnostic ceilings change different information and cannot be added.</p>'
    rows=[]
    rule_names={'ANSWER_Z_Q80':'Frozen rule · Q80 / PRM native','GLOBAL_THRESHOLD_ORACLE':'Best global cutoff · label oracle','TRUE_COUNT_ORACLE':'True answer counts · label oracle','PERFECT_RANKING_AT_FIXED_COUNTS':'Perfect ordering at own frozen counts · label oracle'}
    for rule,label in rule_names.items():
        vals=[]
        for m in ['LSML_CONT','IU_PCR','PRM_NATIVE']:
            if rule=='ANSWER_Z_Q80': r=paired['rules'][m]
            elif m=='PRM_NATIVE' or rule=='PERFECT_RANKING_AT_FIXED_COUNTS':r=paired['oracle_diagnostics'][m].get(rule)
            else:r=dec['methods'][m].get(rule)
            vals.append(number(r.get('prm_score') if r else None,3,True))
        rows.append([label]+vals)
    body+=table(['Operating point','L-SML PRMScore × 100','IU-PCR PRMScore × 100','PRM PRMScore × 100'],rows)
    body+='<p class="small">A fair label-oracle global cutoff improves PRM by 2.857 score points, versus 0.309 / 0.191 for L-SML / IU-PCR. Perfect-ranking ceilings use each method’s own frozen counts and are not directly comparable common-budget ceilings.</p>'
    body+='<p>Q80 applies one global quantile after answer-level standardization; it does not impose 20% on each answer. It marks at least one step in 99.47% / 99.34% of the 758 synthetic clean controls. Estimated prevalence alone cannot identify which answers are clean.</p>'
    body+=source('report/legacy/PAIRED_DECISION_ANALYSIS.json','6,969 answers including controls; PRMScore excludes controls')+'<details><summary>Speaker note · עברית</summary><p lang="he" dir="rtl">ההחלטה כמה צעדים לסמן חשובה, אבל שינוי הסף הגלובלי לבדו כמעט לא משפר את השיטות שלנו. גם הדירוג וגם הזיהוי של תשובות נקיות דורשים שיפור. תקרות האורקל עוזרות להבין את הפער; הן אינן שיטות שנוכל להפעיל בלי תוויות.</p></details></section>'
    new_sections=native_lr_sections()+native_controls_sections()+native_statistics_sections()+native_validation_sections()+extra.get('sections',[])
    # Root-provided sections intentionally supersede a built-in adapter with the same ID.
    section_map={sec['id']:sec for sec in new_sections}
    for sec in section_map.values(): body+=additional_section(sec)
    provenance=read_json(HERE/'Q80_PROVENANCE.json')
    body+='<section id="q80-provenance"><div class="eyebrow">PROTOCOL AND HISTORY</div><h2>Q80 predates corrected exports in Git; historical selection remains retrospective</h2><p>The protocol directory names 3 October. The earliest currently public protocol commit is dated 8 October, before the corrected-result export commit dated 11 October. These committer timestamps establish versioned chronology; original public push time, private registration and absence of selection on older results are unverified. TOP_2 is explicitly post hoc, and the protocol leaves final rule choice to Omri after comparison.</p><p class="small">The registered Q80 comparison uses ≥; the frozen headline export uses &gt;. There are no threshold-boundary ties for these N16 methods, so both reproduce the same numbers here.</p>'+source('report/Q80_PROVENANCE.json','Pinned Git history; exact rule-boundary checks in controls/EXACT_RULE_DIAGNOSTICS.json')+'</section>'
    body+='<section id="tasks"><div class="eyebrow">06 · ACCOUNTABILITY</div><h2>Every Claude assignment, with its evidence</h2><p>All '+str(len(TASKS))+' assigned rows are accounted for. The CPU analyses and local presentation build are complete. One N11 experimental arm remains undefined and is reported as a failed arm. Task 6 is complete locally: GitHub and Claude publication were not executed; Claude Artifact access remains unverified.</p><details><summary>Show all '+str(len(TASKS))+' assignments: previous state, action, finding and evidence</summary>'
    tasks={r['id']:r for r in extra.get('task_status',[])}
    local={'9':('completed','Literature claim withdrawn: paper 10/15 and current leaderboard 23/26 have Simplicity below both other category means.','report/LITERATURE.json'),
           '9ב':('completed','Readable nine-category legend, original diagrams and correct MS convention.','report/prmbench_overview.svg'),
           '13':('completed','Compact main narrative; all-bank matrices and full evidence in appendix.','report/build_v11.py')}
    trows=[]
    for id,req in TASKS:
        row=tasks.get(id)
        if row is None:
            stat,find,evidence=local.get(id,('not done','Awaiting the current persisted analysis output.',''))
            row={'status':stat,'finding':find,'evidence':[evidence] if evidence else []}
        trows.append([id,row.get('requirement',req),row.get('previous_state','Documented v10 assignment; current verification pending.'),row.get('action','See the linked saved analysis.'),row['status'],row.get('finding',''),{'links':row.get('evidence',[])}])
    body+=table(['Task','Requirement','Previous state','Action completed','Status','Finding / remaining limit','Evidence file'],trows)
    body+='</details><p class="small">New splits are internal validation on already inspected data. CPU-only scope excludes PRM-model feature extraction. The repository is public, so advisor links require no private-repository membership.</p></section>'
    if extra.get('remaining_project_work'):
        body+='<section id="remaining-work"><div class="eyebrow">07 · NEXT PROJECT STEPS</div><h2>What remains beyond this CPU completion</h2><ul class="remaining">'+''.join('<li>'+esc(r)+'</li>' for r in extra['remaining_project_work'])+'</ul>'+source('report_data.json','Remaining project work, distinct from the completed CPU analysis assignments')+'</section>'
    body+='<section id="appendix"><div class="eyebrow">APPENDIX · AUDITABLE EVIDENCE</div><h2>Full matrices and source record</h2><details><summary>All banks × methods: AUC, including row and column means</summary>'
    methods=d['roster'];mat=[[p['corrected']['mask1'][b][m] for m in methods] for b in d['banks']]
    rows=[[b]+[number(v) for v in vs]+[number(sum(vs)/len(vs)),number(p['lr']['mask1'][b])] for b,vs in zip(d['banks'],mat)]
    rows.append(['Method mean']+[number(sum(row[j] for row in mat)/len(mat)) for j in range(len(methods))]+[number(sum(sum(row) for row in mat)/(len(mat)*len(methods))),number(sum(p['lr']['mask1'].values())/10)])
    fullauc=PACKAGE/'controls/ALL_BANK_TOP1_AUC.csv'
    if fullauc.exists():
        body+=complete_bank_matrix(fullauc,d['banks'],'within_auc',lr=p['lr']['mask1'])+source('controls/ALL_BANK_TOP1_AUC.csv','14 defined label-free methods × 10 banks; 6,030 eligible answers. LR: historical reference; bank means exclude LR.')
    else:body+=table(['Bank']+[NAMES.get(m,m) for m in methods]+['Bank mean · label-free','LR · supervised'],rows)+source('report/legacy/PRESENTATION_DATA.json','Bank/method means are descriptive; nested banks are not independent replicates')
    body+='</details>'
    allbank=PACKAGE/'controls/ALL_BANK_TOP1_AUC.csv'
    if allbank.exists():
        bankdata=[r for r in read_csv(allbank) if r['classification']=='ALL_NONCONTROL']
        allmethods=sorted(set(r['method'] for r in bankdata if r['bank']!='REFERENCE'))
        lookup={(r['bank'],r['method']):r for r in bankdata}
        body+='<details><summary>Top1-on-error · all defined methods × all ten banks</summary>'+complete_bank_matrix(allbank,d['banks'],'top1_on_error',percent=True)
        refs=[r for r in bankdata if r['bank']=='REFERENCE']
        body+=table(['Reference','Top1-on-error %','N error answers'],[[r['method'],number(r['top1_on_error'],2,True),r['n_error_answers']] for r in refs])
        body+=source('controls/ALL_BANK_TOP1_AUC.csv','6,035 error-containing answers; subtype tables also saved')+'</details>'
    q80=PACKAGE/'controls/ALL_BANK_Q80_PRMSCORE.csv'
    if q80.exists():
        body+='<details><summary>PRMScore · all defined methods × all ten banks · frozen Q80</summary>'+complete_bank_matrix(q80,d['banks'],'category_score',percent=True)
        body+=source('controls/ALL_BANK_Q80_PRMSCORE.csv','6,211 noncontrols / 83,371 steps; row and column means descriptive, not independent replicates')+'</details>'
    body+='<details><summary>Dataset construction and literature-claim audit</summary><p>Correct PRM800K solution paths were perturbed by GPT-4o and filtered; QwQ supplied correct alternative paths for MS. Redundancy and circular premises can preserve the final answer. The original paper has 6,216 answers; our deduplicated population has 6,211 noncontrols plus 758 synthetic clean controls.</p>'
    pp=literature['paper_v1'];ll=literature['live_snapshot']
    body+=table(['Source snapshot','Model rows','Simplicity < both other means','Both NR/NCL < own overall'],[['Paper v1 Table 3',pp['n_rows'],pp['simplicity_strictly_below_both_other_category_means'],pp['both_NR_and_NCL_strictly_below_own_overall']],['Official leaderboard · 11 Oct',ll['n_rows'],ll['simplicity_strictly_below_both_other_category_means'],ll['both_NR_and_NCL_strictly_below_own_overall']]])
    body+='<p>The universal “all ~40 literature rows” claim is withdrawn. Simplicity is frequently weaker, but counterexamples exist. Correctness-focused training is a possible mechanism; these results do not establish it causally. Models and duplicated tables are not independent replications.</p><p><a href="'+PAPER+'">Primary paper</a> · <a href="https://github.com/ssmisya/PRMBench">Official code</a> · <a href="https://prmbench.github.io/">Official leaderboard</a></p>'+source('report/LITERATURE.json','15 paper models; 26 nonhuman leaderboard rows, 27 including human')+'</details>'
    body+='<details><summary>External context, scope and provenance limits</summary><p>Hard2Verify: step index alone has Balanced F1 0.483, above the best panel 0.433 and locked L-SML 0.421 (200 answers). Socratic differs: index 0.580 versus locked L-SML 0.628. These task-specific metrics are not PRMBench PRMScore or AUC.</p><p>The original self-generation equivalence claim is withdrawn. A small point difference and an interval containing zero do not establish equivalence. New source-disjoint splits remain internal validation.</p>'+source('report/legacy/DATA.json','External frozen reports; not newly acquired datasets')+'</details>'
    body+='<details><summary>Raw tables, saved results and reproduction</summary><ul>'
    for f in sorted(PACKAGE.rglob('*')):
        if f.is_file() and f.suffix in ('.csv','.json','.md','.py') and not ('sources' in f.parts and f.suffix=='.py'):
            rel=f.relative_to(PACKAGE)
            body+='<li><a href="'+esc(rel)+'">'+esc(rel)+'</a></li>'
    body+='</ul><pre>.venv/bin/python results/advisor_completion_2026-10-11/report/audit_literature.py\n.venv/bin/python results/advisor_completion_2026-10-11/report/build_v11.py --final</pre></details></section>'
    body+='<footer><b>Evidence state:</b> local v11 artifact; no Git commit, push or Claude update asserted. V10 source: 53ef62e3; research exports: 080877eb. Reproducible builder integrates all views directly; historical files remain preserved. Generated from saved CPU analyses; no new GPU inference.</footer>'
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f7f9fc;color:#223548;font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1260px;margin:auto;padding:44px 32px}header{padding:18px 0 30px}h1{font-size:clamp(34px,5vw,59px);line-height:1.12;letter-spacing:-2px;margin:12px 0 22px}h2{font-size:30px;line-height:1.2;letter-spacing:-.6px;margin:10px 0 20px}p{max-width:1000px}a{color:#1768ac;text-decoration:none}a:hover{text-decoration:underline}.eyebrow{color:#657a91;font-size:12px;letter-spacing:1.5px;font-weight:750}.lead{font-size:20px;max-width:870px;color:#5a7087}.tag{display:inline-block;background:#e5f0fb;border-radius:24px;padding:6px 14px;font-size:13px;color:#275b8b}.notice{border-left:3px solid #c99a52;background:#fff7e8;padding:14px 20px;font-size:13px;max-width:none}nav{position:sticky;top:0;z-index:2;background:#f7f9fcf4;border-bottom:1px solid #dde5ee;padding:17px 0;display:flex;flex-wrap:wrap;gap:24px;font-size:13px}section{background:#fff;border:1px solid #e1e8f0;border-radius:16px;margin:24px 0;padding:28px;scroll-margin-top:75px}.scroll{overflow:auto;margin:20px 0}table{border-collapse:collapse;min-width:100%;font-size:13px;font-variant-numeric:tabular-nums}caption{caption-side:top;text-align:left;color:#566e85;font-size:12px;margin-bottom:9px}th,td{border-bottom:1px solid #e3eaf1;padding:10px 12px;text-align:right;vertical-align:top;white-space:nowrap}th:first-child{text-align:left}td:first-child{text-align:left}thead th{background:#f1f5f9;color:#3e596f;font-size:12px}tbody tr:hover{background:#f7fafc}#tasks table{min-width:1100px}#tasks td:nth-child(4),#tasks td:nth-child(5){white-space:normal;text-align:left;min-width:280px}.source,.small,figcaption{font-size:12px;color:#60778d;line-height:1.5}.source{max-width:none;overflow-wrap:anywhere}figure{margin:22px 0}figure svg,figure img{display:block;width:100%;height:auto}figcaption{padding:8px 3px}details{border-top:1px solid #e1e8ef;padding:14px 0;margin-top:15px}summary{cursor:pointer;font-weight:650;font-size:14px;color:#3d607b}.legendgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin:20px 0}.legendgrid div{padding:14px;background:#f2f6fa;border-radius:10px}.legendgrid b,.legendgrid span,.legendgrid em{display:block;font-size:12px}.legendgrid em{font-style:normal;color:#8494a3;margin-top:5px}pre{white-space:pre-wrap;font-size:12px;padding:16px;background:#f2f5f8}footer{padding:20px;font-size:12px;color:#698095}ul{font-size:12px} @media(max-width:720px){main{padding:20px 12px}section{padding:18px 14px}h1{letter-spacing:-1px}nav{gap:12px}.legendgrid{grid-template-columns:1fr 1fr}} @media print{body{background:white}main{padding:10px;max-width:none}nav{position:static}section{break-inside:avoid}details{display:block}details>*{display:block}}'''
    css+='''.example{margin:22px 0;padding:18px;background:#f8fafc;border-radius:12px}.example h3{font-size:17px;margin:0 0 12px}.step{display:grid;grid-template-columns:145px 1fr;gap:12px;padding:10px 14px;margin:6px 0;border-left:3px solid #92bba9;background:#edf6f1;font-size:12px}.step.bad{border-color:#d88769;background:#fff0e8}.step b{font-size:11px}.step span{overflow-wrap:anywhere}@media(max-width:600px){.step{grid-template-columns:1fr}}'''
    css+='''#tasks td:nth-child(3),#tasks td:nth-child(4),#tasks td:nth-child(6),#tasks td:nth-child(7){white-space:normal;text-align:left;min-width:245px}#tasks td:nth-child(5){min-width:110px}'''
    css+='''#tasks td:nth-child(2){white-space:normal;text-align:left;min-width:230px}.remaining{font-size:15px;max-width:1000px;padding-left:22px}.remaining li{margin:12px 0}'''
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Advisor review v11 · PRMBench</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>'
    assert '\u200e' not in page and '\u200f' not in page
    (PACKAGE/'advisor_review_v11.html').write_text(page)
    inputs=[f for f in (HERE/'legacy').rglob('*') if f.is_file()]+[HERE/'LITERATURE.json',HERE/'Q80_PROVENANCE.json']
    for folder in ['controls','statistics','validation','lr']:
        base=PACKAGE/folder
        if base.exists():inputs.extend(f for f in base.iterdir() if f.is_file() and f.suffix in ['.csv','.json'])
    inputs.extend(f for base in [PACKAGE/'controls',PACKAGE/'statistics',PACKAGE/'lr/figures'] if base.exists() for f in base.glob('*.svg'))
    if (PACKAGE/'report_data.json').exists(): inputs.append(PACKAGE/'report_data.json')
    inputs=sorted(set(inputs))
    receipt={'status':'FINAL' if final else 'DRAFT','build_date':'2026-10-11','builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'inputs':[{'path':str(f.relative_to(PACKAGE)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in inputs],
             'html_sha256':hashlib.sha256(page.encode()).hexdigest(),'bytes':len(page.encode()),'external_runtime_dependencies':False,
             'publication':'local only','task_ids':[i for i,_ in TASKS]}
    (HERE/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'output':str(PACKAGE/'advisor_review_v11.html'),'bytes':receipt['bytes'],'status':receipt['status']}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');args=ap.parse_args();build(args.final)
