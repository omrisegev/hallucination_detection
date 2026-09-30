"""Independent oracle accounting, AUC decomposition and raw-level review."""
import os
for option in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[option]='1'
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fusion_gate_interface_audit_v1'
PARENT=ROOT/'results/answer_localization_representation_pilot_v1'
LATEST=ROOT/'results/fusion_reliability_regularization_v1'
ARMS=('equal_parent','iu_parent','joint_parent','joint_graph010_parent',
      'joint_graph_permuted_parent','entropy_parent','iu__dufs_graph')
NAMES=dict(zip(ARMS,('Equal fusion','IU-PCR','Joint L-SML, lambda 0',
           'Joint graph, lambda 0.1','Joint permuted graph','Entropy control','IU + graph correction')))


def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def review():
    manifest,frozen,evaluation=[load(OUT/n) for n in ('MANIFEST.json','DIAGNOSTICS_FROZEN.json','EVALUATION.json')]
    for p,h in {**manifest['hashes'],**frozen['files']}.items(): assert sha(p)==h,p
    assert frozen['manifest_sha256']==evaluation['manifest_sha256']==sha(OUT/'MANIFEST.json')
    assert evaluation['diagnostics_sha256']==sha(OUT/'DIAGNOSTICS_FROZEN.json')
    rows=evaluation['rows'];assert len(rows)==58 and {r['uid'] for r in rows}=={r['uid'] for r in manifest['selected']}
    old=load(LATEST/'EVALUATION.json'); parents={r['uid']:r for r in old['rows']}
    counts=Counter(); release=load(PARENT/'RELEASE.json')
    for cell in sorted({r['cell'] for r in rows}):
        info=release['cells'][cell];assert sha(info['label_path'])==info['label_opaque_sha256']
        with np.load(info['label_path'],allow_pickle=False) as label:
            ids={str(x):i for i,x in enumerate(label['row_ids'])}; assert len(ids)==len(label['row_ids'])
            for r in (r for r in rows if r['cell']==cell):
                pos=ids[r['row_id']]
                if cell.startswith('prm'):
                    a,b=label['step_flag_offsets'][pos:pos+2]; truth=label['step_error_flags'][a:b]
                else: truth=int(label['first_error'][pos])
                np.testing.assert_array_equal(r['target'],truth);counts['independent_label_joins']+=1
                with np.load(PARENT/'inputs'/f"{r['uid']}.npz",allow_pickle=False) as raw:
                    n=(len(raw['raw'])//8)*8
                    assert abs(raw['raw'][:n,1].mean()-r['raw_entropy_mean'])<1e-12
                    assert abs(raw['raw'][:n,15].mean()-r['raw_spilled_mean'])<1e-12
                counts['raw_level_reconstructions']+=2
                for arm,d in r['methods'].items():
                    assert d['valid']==parents[r['uid']]['decision_valid'][arm]
                    if not d['valid']: continue
                    np.testing.assert_array_equal(d['risk'],parents[r['uid']]['scores'][arm])
                    assert d['peak']==int(np.argmax(d['risk']))
                    assert d['prediction']==parents[r['uid']]['predictions'][arm]
                    np.testing.assert_allclose(np.asarray(d['risk'])+d['origin_offset'],d['origin_projection'],atol=1e-12)
                    a,b=d['gate']['log_likelihood'];n=r['tokens']//8
                    assert abs(d['gate']['bic_gain']-(2*(b-a)-3*np.log(n)))<1e-9
                    assert abs(d['gate']['duplicated_fixed_parameter_bic_gain']-(4*(b-a)-3*np.log(2*n)))<1e-9
                    counts['frozen_score_and_gate_checks']+=1
    for arm in ARMS:
        s=evaluation['summaries'][arm]
        for key in ('actual','perfect_gate','perfect_locator','both_perfect','fixed_parent_gate'):
            f1s=[]
            for cell in sorted({r['cell'] for r in rows if r['cell'].startswith('pb_')}):
                correct={True:[],False:[]}
                for r in (r for r in rows if r['cell']==cell):
                    d=r['methods'][arm];clean=r['target']==-1;hit=False
                    if d['valid']:
                        opened=d['gate_open'];exact=d['peak']==r['target']
                        if key=='actual': hit=(not opened) if clean else opened and exact
                        elif key=='perfect_gate': hit=clean or exact
                        elif key=='perfect_locator': hit=(not opened) if clean else opened
                        elif key=='both_perfect': hit=True
                        else: hit=d['fixed_parent_prediction']==r['target']
                    correct[clean].append(int(hit));counts['independent_oracle_outcomes']+=1
                ca,ea=np.mean(correct[True]),np.mean(correct[False]);f1=2*ca*ea/(ca+ea) if ca+ea else 0.
                expected=s['pb'][key]['cells'][cell]
                assert sum(correct[True])==expected['clean_successes'] and sum(correct[False])==expected['error_exact_successes']
                assert abs(f1-expected['f1'])<1e-12;f1s.append(f1)
            assert abs(np.mean(f1s)-s['pb'][key]['macro_f1'])<1e-12;counts['oracle_macro_checks']+=1
        assert abs(s['pb']['actual']['macro_f1']-old['metrics'][arm]['pb']['macro_f1'])<1e-12
        available=[r for r in rows if r['cell'].startswith('prm') and r['methods'][arm]['valid']]
        y=np.concatenate([r['target'] for r in available])
        groups=np.concatenate([np.repeat(i,len(r['target'])) for i,r in enumerate(available)])
        for key in ('risk','origin_projection'):
            x=np.concatenate([r['methods'][arm][key] for r in available]); expected=s['prm'][key]
            assert abs(roc_auc_score(y,x)-expected['pooled_auc'])<1e-12
            pi,ni=np.flatnonzero(y==1),np.flatnonzero(y==0)
            same=groups[pi,None]==groups[ni]
            wins=(x[pi,None]>x[ni]).astype(float)+.5*(x[pi,None]==x[ni])
            assert int(same.sum())==expected['within_pairs'] and int((~same).sum())==expected['cross_pairs']
            assert abs(wins[same].mean()-expected['pair_weighted_within_auc'])<1e-12
            assert abs(wins[~same].mean()-expected['cross_answer_auc'])<1e-12
            individual=[roc_auc_score(r['target'],r['methods'][arm][key]) for r in available if len(set(r['target']))==2]
            assert abs(np.mean(individual)-expected['mean_within_answer_auc'])<1e-12
            counts['auc_decompositions']+=1
        for cell,probe in s['pb_probe_auc'].items():
            subset=[r for r in rows if r['cell']==cell and r['methods'][arm]['valid']]
            yy=[int(r['target']!=-1) for r in subset]
            for key,xx in [('bic_gain_auc',[r['methods'][arm]['gate']['bic_gain'] for r in subset]),
                           ('raw_entropy_mean_auc',[r['raw_entropy_mean'] for r in subset]),
                           ('raw_spilled_mean_auc',[r['raw_spilled_mean'] for r in subset])]:
                assert abs(roc_auc_score(yy,xx)-probe[key])<1e-12;counts['probe_auc_checks']+=1
    result={'status':'PASS','counts':dict(counts),'evaluation_sha256':sha(OUT/'EVALUATION.json'),
        'review_script_sha256':sha(__file__),'max_affine_error':max(r['affine_matrix_error'] for r in rows),
        'max_translation_bic_error':max(d.get('translation_bic_error',0.) for r in rows for d in r['methods'].values()),
        'max_score_replay_error':max(d.get('score_replay_error',0.) for r in rows for d in r['methods'].values())}
    save(OUT/'REVIEW.json',result);print(json.dumps(result,indent=2),flush=True)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+
                     ['| '+' | '.join(str(c) for c in row)+' |' for row in rows])


def render():
    e,a=load(OUT/'EVALUATION.json'),load(OUT/'REVIEW.json')
    assert a['status']=='PASS' and a['evaluation_sha256']==sha(OUT/'EVALUATION.json') and a['review_script_sha256']==sha(__file__)
    s=e['summaries']
    lines=['# Our fusion needs both a gate and a useful locator','',
        'Completed diagnostic, 2026-09-07. Same 58 development answers: 12 PRMBench and 46 ProcessBench. IU-PCR / Joint L-SML remain the method. This audit changes no deployed prediction and promotes no candidate.','',
        'The next stage should address fusion representation and the location readout as well as the no-error decision. Fixing only the gate cannot solve the observed failures. Restoring a constant score offset cannot repair the current free-mean GMM gate.','',
        '## 1. What the current pipeline keeps and discards','',
        'Within each answer: form N windows by P features; subtract each feature mean and divide by its standard deviation; fit IU or Joint; normalize the fused score to unit standard deviation; use a GMM to decide whether two levels exist; return the highest-risk step if the gate opens. The feature signs have the declared negative-entropy anchor.','',
        'The normalization removes per-feature level and scale from the matrix used for fusion. All 58 coordinate-counterfactual checks reproduce the normalized matrix. All 361 valid core/answer checks have score mean zero and standard deviation one. Shifting the score and its step values by +10 leaves every GMM decision unchanged.','',
        'This is a verified invariance of this pipeline, not a proof that absolute telemetry is useless, that every gray-box method is impossible, or that centering caused all failures. The counterfactual transforms feature coordinates; it is not an alternate generated answer.','',
        '## 2. A perfect gate would still leave localization errors','',
        'PB macro-F1 (%), always on all 46 answers. Every oracle column uses labels and is diagnostic only. Invalid fits remain failures, including in the last column. These are ceilings conditional on the component held fixed, not attainable forecasts.','',
        table(['Core','Actual','Perfect binary gate; same peak','Same gate; perfect locator','Both perfect on valid fits'],[
            [NAMES[arm]]+[f"{100*s[arm]['pb'][k]['macro_f1']:.2f}" for k in ('actual','perfect_gate','perfect_locator','both_perfect')] for arm in ARMS]),'',
        'The raw IU peak hits the first error in 7 of 25 erroneous answers; Joint lambda-zero in 9, graph 0.1 in 8, equal fusion in 8. Joint has four invalid erroneous answers. These are small development counts, not evidence of a reliable Joint win.','',
        table(['Core','Peak before first error','Peak exact','Peak after first error','Invalid error fits','Exact peaks hidden by gate'],[
            [NAMES[arm]]+[s[arm]['peak_location'].get(k,0) for k in ('before','exact','after','invalid')]+[s[arm]['gate_counts'].get('exact_peak_gated_out',0)] for arm in ARMS]),'',
        '## 3. Mixture states are not correctness labels','',
        'A two-component GMM models two distributions of score values. The code treats that as evidence of an error, but a clean answer can have two uncertainty levels and a wrong answer can have one. The observations below measure the mismatch; they do not invalidate mixture modelling as a possible supporting tool.','',
        table(['Core','Clean: gate closed / 21','Clean: gate open','Clean: invalid','Error: gate open / 25','Error: gate closed','Error: invalid'],[
            [NAMES[arm]]+[s[arm]['gate_counts'].get(k,0) for k in ('clean_gate_closed','clean_gate_open','clean_invalid','error_gate_open','error_gate_closed','error_invalid')] for arm in ARMS]),'',
        'The IU graph correction illustrates component interaction: its own gate gives PB 14.45%; reusing the original IU gate gives 24.64%. That frozen diagnostic was already in the preceding experiment. It does not establish a graph gain or improvement on both tasks.','',
        '## 4. A higher pooled PRMB AUC can leave every local ordering unchanged','',
        'For IU, 4,109 of 4,600 positive-negative step pairs (89.33%) compare different answers. The pooled benchmark score therefore measures both within-answer ranking and alignment of score levels across answers. Both matter for that metric, but they are different achievements.','',
        'The diagnostic origin projection removes the centering offset while retaining the answer-fitted scales and weights. For entropy, it retains entropy divided by its own standard deviation. The offset has no established calibration meaning, and feature units/origins matter. It changes cross-answer ordering only. This is not a new selected candidate.','',
        table(['Core','Valid PRMB answers','Original pooled AUC','Origin projection pooled AUC','Mean within-answer AUC, both','Cross-answer pair share'],[
            [NAMES[arm],s[arm]['prm_valid_answers'],f"{s[arm]['prm']['risk']['pooled_auc']:.5f}",
             f"{s[arm]['prm']['origin_projection']['pooled_auc']:.5f}",f"{s[arm]['prm']['risk']['mean_within_answer_auc']:.5f}",
             f"{100*s[arm]['prm']['risk']['cross_pair_fraction']:.2f}%"] for arm in ARMS]),'',
        'Keep the registered pooled endpoint unchanged for continuity; this is our project evaluation contract, not a claim about every metric in the PRMBench paper. Also report within-answer ranking and exact first-error localization. Different PRMB coverage prevents reading this table as a matched ranking between Joint and IU. The report preserves both the unweighted mean per-answer AUC and the pair-weighted decomposition in EVALUATION.json.','',
        '## 5. Neither absolute uncertainty nor more rows is an automatic fix','',
        'Descriptive binary-error AUROCs for fixed-direction probes, on the IU-valid population (all 46 PB answers). No threshold or sign is fitted from these labels. The small length-stratified development sample does not establish significance or transfer.','',
        table(['PB subset','Clean / error','GMM BIC gain','Raw entropy mean','Raw spilled-energy mean'],[
            [cell.removeprefix('pb_').removesuffix('_q8'),f"{p['clean']} / {p['erroneous']}",
             f"{p['bic_gain_auc']:.3f}",f"{p['raw_entropy_mean_auc']:.3f}",f"{p['raw_spilled_mean_auc']:.3f}"]
             for cell,p in s['iu_parent']['pb_probe_auc'].items()]),'',
        'Absolute entropy has a strong observed separation only in Omnimath here; it does not supply a demonstrated common gate. BIC gain has some observed binary ranking in each subset, but the current zero cutoff and model-fit assumptions are not calibrated correctness evidence.','',
        'BIC also uses the nominal row count. An analytic stress test duplicates every observation while keeping the fitted parameters fixed. It adds no information, yet the number of IU PB answers favoring two components rises from 27 to 35. This is not a refit result or a proposed correction. Non-overlapping windows can still be temporally dependent; the saved lag-one correlations are diagnostics, not validated effective sample sizes.','',
        '## 6. Continue development of our fusion','',
        'Next bounded stage: revisit the window feature bank and its Joint grouping using an explicitly local, multiscale representation, with IU and equal fusion controls. First audit the existing fast/slow, innovation and persistence mechanisms in spectral_utils/unified_causal_iu.py and unified_causal_subset_search.py. Their existing fitted pipeline and subset search explicitly use supervised development, so reusing fitted signs, rosters or references would require a borrowed/hybrid label. A new answer-only adaptation must fit its quantities from the current answer and disclose fixed engineering choices. These mechanisms are not newly invented here.','',
        'Evaluate the same frozen no-error gate as a diagnostic alongside each native gate so a feature/ranking change can be separated from a gate change. Check fit coverage and incremental information; correlated extra coordinates must not be counted as independent views. Freeze one bounded feature-bank comparison before looking at its new scores.','',
        'Preserve absolute feature summaries for a separate future gate experiment, rather than assuming that adding an arbitrary offset fixes GMM. Any pooled unlabeled normalization or gate is explicitly hybrid; retain the strict one-answer reference. Freeze that experiment before evaluation. Neither the current GMM zero threshold nor a label-picked replacement is a final correctness rule.','',
        'Joint feature/group discovery, learned temporal support, task-aware sampling, the full comparator registry/replay, untouched confirmation on both tasks and historical 24-cell transfer remain open. No method is declared a publication winner.','',
        '## Verification and historical bridge','',
        'Five scientific identity tests passed. Independent review rejoined 58 labels, reconstructed 116 raw absolute levels, checked 361 frozen score/gate records, independently counted 1,610 oracle outcomes, verified 35 PB macro values, 14 AUC decompositions and 84 descriptive probe AUROCs. Fourteen original task endpoints replay exactly.','',
        f"Unlabeled audit time: {load(OUT/'RUN_STATE.json')['seconds']:.1f} seconds on one CPU process; evaluation and review are additional. Maximum affine-normalization discrepancy: {a['max_affine_error']:.3g}; shifted-GMM BIC discrepancy: {a['max_translation_bic_error']:.3g}. All bound source and result hashes match.",'',
        'The historical 30-long-answer IU AUC 0.70070 and Claude pooled-fit experiments have different populations/fitting contracts. They remain context; this diagnostic replays the current 58-answer release exactly. No model inference or Claude worktree edit was needed.','']
    md='\n'.join(lines);(OUT/'REPORT.md').write_text(md,encoding='utf-8')
    blocks=[];in_table=False
    for line in lines:
        if line.startswith('|'):
            if not in_table: blocks.append('<div class="scroll"><table>');in_table=True;header=True
            if set(line.replace('|','').replace('-','').strip())==set(): continue
            tag='th' if header else 'td';blocks.append('<tr>'+''.join(f'<{tag}>{html.escape(x.strip())}</{tag}>' for x in line.strip('|').split('|'))+'</tr>');header=False
            continue
        if in_table: blocks.append('</table></div>');in_table=False
        if line.startswith('# '): blocks.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '): blocks.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line: blocks.append('<p>'+html.escape(line)+'</p>')
    if in_table: blocks.append('</table></div>')
    interaction='''<section class="toy"><h2>Try it: change a feature's overall level</h2>
<p>Illustrative numbers, not experimental data. Moving the whole feature up changes its absolute level. Normalization removes that shift before fusion.</p>
<label for="shift">Added level: <output id="amount">0.0</output></label>
<input id="shift" type="range" min="0" max="4" step="0.1" value="0" />
<div class="charts"><div><h3>Feature before normalization</h3><svg viewBox="0 0 400 190" role="img" aria-label="Raw feature changes with added level"><path d="M20 10 V170 H390" fill="none" stroke="#aab9ce"/><polyline id="rawcurve" fill="none" stroke="#1775b6" stroke-width="3"/><text x="22" y="185">Window order</text></svg></div>
<div><h3>Feature entering fusion</h3><svg viewBox="0 0 400 190" role="img" aria-label="Normalized feature unchanged by added level"><path d="M20 10 V170 H390" fill="none" stroke="#aab9ce"/><polyline id="normcurve" fill="none" stroke="#188369" stroke-width="3"/><text x="22" y="185">Window order</text></svg></div></div>
<p id="readout" aria-live="polite"></p></section>
<script>const base=[.3,.5,.4,1.3,.9,.45,.7,.55];const slider=document.getElementById('shift');
function redraw(){let shift=Number(slider.value),raw=base.map(x=>x+shift),mean=raw.reduce((a,b)=>a+b)/raw.length,sd=Math.sqrt(raw.reduce((a,b)=>a+(b-mean)**2,0)/raw.length),norm=raw.map(x=>(x-mean)/sd);document.getElementById('amount').value=shift.toFixed(1);document.getElementById('rawcurve').setAttribute('points',raw.map((x,i)=>`${25+i*50},${165-x*27}`).join(' '));document.getElementById('normcurve').setAttribute('points',norm.map((x,i)=>`${25+i*50},${110-x*32}`).join(' '));document.getElementById('readout').textContent=`Raw mean: ${mean.toFixed(2)}. Normalized mean: 0; standard deviation: 1. The normalized pattern stays the same.`;}slider.addEventListener('input',redraw);redraw();</script>'''
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Fusion normalization and no-error audit</title>
<style>body{font:17px/1.6 system-ui,sans-serif;background:#f4f7fb;color:#172b45;margin:0}main{max-width:1180px;margin:auto;padding:32px 24px 70px}h1{line-height:1.15;font-size:2.3rem}h2{margin-top:2em;line-height:1.3}p{max-width:95ch}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:.89rem;background:white}th,td{padding:11px;border:1px solid #d6dfeb;text-align:left}th{background:#e4edf7}tr:nth-child(even){background:#f8fafc}.toy{padding:25px;border:2px solid #96b6d7;border-radius:12px;background:white}.charts{display:grid;grid-template-columns:1fr 1fr;gap:22px}svg{width:100%;max-width:470px}input{display:block;width:min(600px,95%);margin:18px 0}a{color:#1163a0}@media(max-width:660px){.charts{grid-template-columns:1fr}main{padding:18px}h1{font-size:1.9rem}}@media print{body{background:white}.toy{break-inside:avoid}input{display:none}}</style></head><body><main>'''
    # The toy follows the introduction, before the detailed evidence tables.
    intro_end=next(i for i,b in enumerate(blocks) if b.startswith('<h2>2.'))
    blocks.insert(intro_end,interaction)
    document+='\n'.join(blocks)+'<p><a href="../fusion_reliability_regularization_v1/REPORT.html">Previous regularization report</a> | <a href="../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html">Joint L-SML visual guide</a></p></main></body></html>'
    (OUT/'REPORT.html').write_text(document,encoding='utf-8')
    save(OUT/'REPORT_PROVENANCE.json',{'evaluation_sha256':sha(OUT/'EVALUATION.json'),'review_sha256':sha(OUT/'REVIEW.json'),
        'report_script_sha256':sha(__file__),'markdown_sha256':sha(OUT/'REPORT.md'),'html_sha256':sha(OUT/'REPORT.html')})
    print('HTML and Markdown reports written with provenance.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('review','render'),required=True)
    args=parser.parse_args();{'review':review,'render':render}[args.phase]()
