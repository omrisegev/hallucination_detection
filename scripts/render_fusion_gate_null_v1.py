"""Render and check the controlled gate study without creating benchmark arms."""
import argparse
import ast
from html import escape
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module(ROOT/'scripts/run_fusion_gate_null_v1.py','null_report_driver');OUT=d.OUT
LABELS=dict(raw='Unfiltered',kalman_cold='Kalman: cold',imm_cold='IMM: cold',kalman_warm='Kalman: warm',imm_warm='IMM: warm')
def pct(x):return 'n/a' if x is None else f'{100*x:.2f}%'
def ci(x):return 'n/a' if x is None else '['+', '.join(f'{100*v:.2f}%' for v in x)+']'
def table(headers,rows,identity):
    return '<div class="scroll"><table id="'+identity+'"><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def sources():
    m=d.verify();r=d.load(OUT/'REVIEW.json');e=d.load(OUT/'RESULTS.json');f=d.load(OUT/'FROZEN.json');assert r['status']=='PASS'
    for group in (r['hashes'],r['dependencies']):
        for p,h in group.items():assert d.sha(p)==h,p
    return m,r,e,f


def render():
    m,r,e,f=sources();rows=e['summaries'];index={(x['n'],x['rho'],x['jump'],x['readout']):x for x in rows}
    fig,axes=plt.subplots(1,3,figsize=(13,5.2),sharey=True,layout='constrained')
    colors=['#31566e','#779ba9','#cc642e','#1d846a','#9d3958'];styles=['-','--','--','-', '-']
    for ax,n in zip(axes,(16,64,256)):
        for name,color,style in zip(m['readouts'],colors,styles):
            items=[index[n,rho,False,name] for rho in (0.,.6,.9)];rate=np.array([x['rate'] for x in items])*100
            ax.plot([0,.6,.9],rate,marker='o',color=color,ls=style,label=LABELS[name]);ax.set_xticks([0,.6,.9]);ax.grid(alpha=.2)
        ax.set_title(str(n)+' windows');ax.set_xlabel('Input AR(1) correlation');ax.set_ylim(0,100);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Gate opens (%)');fig.suptitle('One stationary Gaussian input regime —64 trials per condition\nConnected descriptive points; exact intervals are in the table',fontsize=12)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False)
    for ext in ('svg','png'):fig.savefig(OUT/('null_gate_rates.'+ext),dpi=175,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for j,(name,color) in enumerate(zip(('raw','kalman_warm','imm_warm'),('#31566e','#1d846a','#9d3958'))):
        vals=[index[n,0.,True,name]['rate']*100 for n in (16,64,256)];ax.bar(np.arange(3)+(j-1)*.23,vals,width=.22,color=color,label=LABELS[name])
    ax.set_xticks(range(3),['16 windows','64 windows','256 windows']);ax.set_ylim(0,105);ax.set_ylabel('Gate opens (%)')
    ax.set_title('Fixed positive control: a +3 SD mean jump halfway through\nDistribution-change sensitivity, not hallucination detection')
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.12),ncol=3,frameon=False);ax.spines[['top','right']].set_visible(False)
    for ext in ('svg','png'):fig.savefig(OUT/('jump_gate_rates.'+ext),dpi=175,bbox_inches='tight')
    plt.close(fig)
    summary=[[x['n'],x['rho'],'jump' if x['jump'] else 'stationary',x['readout'],x['valid'],x['failures'],x['open'],pct(x['rate']),ci(x['interval']),x['two_components']] for x in rows]
    paired=[[x['n'],x['rho'],'jump' if x['jump'] else 'stationary',x['left']+' minus '+x['right'],x['common'],x['open_gained'],x['open_lost']] for x in e['pairs']]
    parent=d.load(e['parent_benchmark']);names=['dual__iu','dual__cond100_graph010','sample_risk_top__equal_graph_perm','traj_iu_joint_graph__gls','traj_iu_joint_graph__imm']
    anchors=[[name,f'{parent["metrics"][name]["prm"]["auroc"]:.5f}',pct(parent['metrics'][name]['pb']['macro_f1'])] for name in names]
    links=[('Frozen protocol','../../docs/experiments/FUSION_GATE_NULL_V1.md'),('Simulation component','../../spectral_utils/fusion_gate_null.py'),('Driver','../../scripts/run_fusion_gate_null_v1.py'),
        ('Review code','../../scripts/review_fusion_gate_null_v1.py'),('Review evidence','REVIEW.json'),('All simulation summaries','RESULTS.json'),('Frozen trial manifest','MANIFEST.json'),
        ('Latest real benchmark report','../fusion_trajectory_imm_v1/REPORT.html'),('Earlier gate-interface audit','../fusion_gate_interface_audit_v1/REPORT.html')]
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fusion gate mechanism check — Step319</title><style>body{margin:0;background:#f4f7f9;color:#1c2d39;font:17px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:30px 24px 80px}h1{line-height:1.2}h2{margin-top:2rem}.note{padding:18px;background:white;border-left:5px solid #b56029}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:10px;border-bottom:1px solid #dce4eb;text-align:left}th{background:#e8f0f4}a{color:#145c80}img{width:100%;height:auto}figure{margin:24px 0}.muted{color:#516877}li{margin:7px 0}</style></head><body><main>
<p class="muted">7 September 2026 · Step319 · Controlled simulation · Review PASS</p>
<h1>The mixture gate can open after filtering a source with no change in regime.</h1>
<p>This study tests a possible weakness in the no-error decision supporting IU-PCR / Joint L-SML. It does not add a new detector or change any benchmark prediction. The source is explicitly synthetic: one stationary Gaussian AR(1) process. Its correlation and the filtering procedure vary under a frozen design.</p>
<div class="note"><strong>Concrete example:</strong> for256 independent input windows, the unfiltered gate opens in0/64 trials, versus26/64 (40.6%) after IMM. Processing256 preceding synthetic observations before scoring still gives26/64. For64 independent windows, the respective counts are0,11 and12. The behavior therefore survives this startup control. It is not evidence that40.6% of correct real answers would be flagged.</div>
<h2>Why this matters for our fusion</h2><p>In Step318 the primary IMM increased clean false alarms from18 to22 and reduced correct first-error decisions from12 to7. Fixed peak/gate exchanges showed both components contributed to the observed regression. Here, even with no change in the generating source, filtering can change the score distribution enough for the same GMM decision to open.</p>
<p>Lag dependence was already examined in Step302; this is a controlled extension, not a new discovery that adjacent observations can correlate. The current result establishes a mechanism under the declared simulation. It does not establish that this mechanism explains every real-data error.</p>
<h2>Design and interpretation</h2><p>Three lengths16/64/256, correlations0/.6/.9,64 independent replicates per cell:576 stationary-source trials. Three additional cells add a fixed+3 SD mean jump halfway through, for192 positive-control trials. All768 trials use five readouts: raw, ordinary Kalman cold/warm, IMM cold/warm. All3840 outputs are valid; no failed fits were removed.</p>
<p>Cold filters start at the scored segment. Warm filters process256 extra synthetic observations first. Warm is a startup diagnostic with additional data, not a deployable answer-only candidate. The same within-segment normalization and original noise heuristic/GMM settings are retained. Each readout is normalized again before the mixture fit.</p>
<p>A BIC preference for two components is a model-selection result. It does not provide a nominal5% alarm guarantee or semantic labels. IMM is nonlinear, so Gaussian input does not imply Gaussian output; ordinary filtering also creates dependent observations. We do not call these fitted mixture preferences a numerical bug. <a href="https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html">Official GMM model-selection example</a>.</p>
<figure><img src="null_gate_rates.png" alt="Gate-opening frequency for one stationary source, by length, correlation and cold or warm filtering"><figcaption>All nine stationary-source conditions. Rates are simulation frequencies; none is a hallucination false-positive guarantee.</figcaption></figure>
<figure><img src="jump_gate_rates.png" alt="Gate-opening rates for the fixed halfway mean-jump positive control"><figcaption>Sensitivity changes too: with64 windows, raw detects the declared jump in21/64 trials, versus64/64 for IMM. The next design must preserve sensitivity while checking false alarms.</figcaption></figure>
<h2>Every condition, including cold/warm and positive controls</h2><p>Intervals are exact binomial95% intervals over the64 independent replicates within a cell. Zero observations does not imply zero population probability:0/64 has interval[0,5.60%]. Methods are paired on the same generated sources; the paired table follows. These are descriptive intervals across many conditions, not a single confirmatory test.</p>'''
    html+=table(['Windows','Input rho','Source','Readout','Valid /64','Failures','Gate open','Open rate','95% interval','Two components'],summary,'summary')
    html+='''<h2>Paired opening changes</h2><p>An opening gained is a new alarm on a stationary source, but a newly detected distribution change in a jump control. Do not interpret the two as the same success metric. Common counts disclose fit coverage.</p>'''
    html+=table(['Windows','Input rho','Source','Comparison','Common /64','Open gained','Open lost'],paired,'pairs')
    html+='''<h2>The actual benchmark remains unchanged</h2><p>These are Step318's corrected current110 development results, shown only to preserve context. No simulation row is a new localization candidate; the full176-entry comparison stays in its own report.</p>'''
    html+=table(['Method','PRMB AUROC','ProcessBench'],anchors,'anchors')
    html+=f'''<h2>What to test next</h2><p>A calibration model should simulate the same score-processing procedure before applying the gate, rather than assume the filtered output is an independent Gaussian sample. A stationary Gaussian/AR source would remain an explicit approximation, not a model proven to describe correct answers. A fitted null also introduces parameter-estimation uncertainty.</p>
<p>First verify a fixed calibration rule on separate simulated replicates, including the mean-jump controls; keep the calibration distribution and evaluation replicates separate. Then freeze one gate-only real-benchmark comparison with unchanged fusion curves and peaks. If it loses sensitivity or fails under its own simulated assumptions, do not promote it. This is a bounded next question, not a promised solution or permission to tune on benchmark labels. <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.monte_carlo_test.html">Monte Carlo testing requires an explicit null generator</a>.</p>
<p>Keep both IU/Joint, matched simple fusion controls, real/permuted graph references, all176 historical anchors, within-answer PRMB evidence and both benchmarks. Better no-error handling alone cannot repair all wrong peak locations. The wider Joint/feature, named supporting-track, corrected multi-answer, comparator, untouched-confirmation and historical24 requirements remain open.</p>
<h2>Review and execution</h2><p>Three tests pass. Independent review replays768 source paths,1536 scalar Kalman trajectories,3840 normalizations and mixture-likelihood/BIC calculations,48 direct vector IMM trajectories and120 actual GMM refits. No failures. Maximum scalar-Kalman difference{r['maximum_kalman_difference']:.3g}. Simulation{f['seconds']:.2f}s; review{r['seconds']:.2f}s. No model inference or benchmark fitting was run.</p>
<p>{escape(r['scope'])}</p><h2>Evidence and code</h2><ul>'''+''.join('<li><a href="'+escape(path)+'">'+escape(label)+'</a></li>' for label,path in links)+'''</ul></main></body></html>'''
    (OUT/'REPORT.html').write_text(html,encoding='utf-8')
    md=['# Controlled fusion-gate check - Step319','','Review PASS. No new benchmark candidate or model inference.','',
        '768 synthetic source trials, five readouts,3840 valid outputs. All12 conditions retained. Warm uses256 additional synthetic observations as a startup diagnostic.',
        'At N256/rho0/no jump: raw opens0/64, IMM cold26/64, IMM warm26/64. At N64:0/64,11/64,12/64. Input has one stationary Gaussian regime; output distribution need not be Gaussian.',
        'At N64/rho0/+3SD jump: raw21/64, IMM cold/warm64/64. Sensitivity must be preserved in any future calibration.',
        'BIC is model selection, not a5% semantic-error test. These are simulation frequencies, not real-answer false-positive rates.',
        f'Simulation{f["seconds"]:.2f}s; review{r["seconds"]:.2f}s. Three tests;768 source/1536 Kalman/3840 GMM algebra/48 vector IMM/120 GMM refits pass.',
        'Next verify one procedure-matched calibration rule on independent simulations before a gate-only benchmark comparison. Preserve all176 anchors and full research scope.','',
        '| N | rho | Source | Readout | Valid | Failures | Open | Rate | 95% CI | Two components |','|---|---|---|---|---|---|---|---|---|---|']
    md+=['| '+' | '.join(map(str,row))+' |' for row in summary];(OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('Rendered null-gate report and two SVG/PNG figures.',flush=True)


def validate():
    m,r,e,f=sources();Page=module(ROOT/'scripts/validate_fusion_sampling_report_v1.py','null_static_page_parser').Page
    page=Page();page.feed((OUT/'REPORT.html').read_text(encoding='utf-8'));assert len(page.ids)==len(set(page.ids))
    links=0
    for link in page.links:
        if link.startswith('https://'):continue
        assert (OUT/link).resolve().is_file(),link;links+=1
    assert len(page.tables['summary'])==61 and len(page.tables['pairs'])==73
    for row,x in zip(page.tables['summary'][1:],e['summaries']):
        assert row==[str(x['n']),str(x['rho']),'jump' if x['jump'] else 'stationary',x['readout'],str(x['valid']),str(x['failures']),str(x['open']),pct(x['rate']),ci(x['interval']),str(x['two_components'])]
    for row,x in zip(page.tables['pairs'][1:],e['pairs']):
        assert row==[str(x['n']),str(x['rho']),'jump' if x['jump'] else 'stationary',x['left']+' minus '+x['right'],str(x['common']),str(x['open_gained']),str(x['open_lost'])]
    parent=d.load(e['parent_benchmark'])
    for row in page.tables['anchors'][1:]:
        x=parent['metrics'][row[0]];assert row[1:]==[f'{x["prm"]["auroc"]:.5f}',pct(x['pb']['macro_f1'])]
    files=[ROOT/'spectral_utils/fusion_gate_null.py',ROOT/'tests/test_fusion_gate_null.py',ROOT/'scripts/run_fusion_gate_null_v1.py',ROOT/'scripts/review_fusion_gate_null_v1.py',Path(__file__)]
    for p in files:ast.parse(p.read_text(encoding='utf-8'))
    for name in ('null_gate_rates','jump_gate_rates'):
        assert '<svg' in (OUT/(name+'.svg')).read_text(encoding='utf-8');assert (OUT/(name+'.png')).stat().st_size>10000
        files.extend([OUT/(name+'.svg'),OUT/(name+'.png')])
    hashes=dict(m['hashes'])
    for group in (f['files'],r['hashes'],r['dependencies']):
        for p,h in group.items():assert d.sha(p)==h,p;hashes[p]=h
    result=dict(status='PASS',report_sha256=d.sha(OUT/'REPORT.html'),verified_hashes=len(hashes),local_links_images=links,static_numeric_rows=137,python_ast_files=5,
        files={str(p):d.sha(p) for p in files+[OUT/'REPORT.html',OUT/'REPORT.md']},visual_scope='Static HTML/number/link checks and exported SVG/PNG assets; PNG visual inspection is separate. No browser rendering claim.')
    d.save(OUT/'ARTIFACT_VALIDATION.json',result);print({k:v for k,v in result.items() if k!='files'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['render','validate'],default='render');args=p.parse_args();render() if args.phase=='render' else validate()
