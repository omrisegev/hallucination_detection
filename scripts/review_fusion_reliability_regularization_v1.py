"""Independent raw-token sensitivity, weight, metric and provenance review."""
import os
for option in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[option]='1'
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fusion_reliability_regularization_v1'
PARENT=ROOT/'results/answer_localization_representation_pilot_v1'
REP='moments27_local8'
PARENTS={'equal_parent':REP+'__equal','iu_parent':REP+'__iu','joint_parent':REP+'__joint_lambda0',
    'joint_graph010_parent':REP+'__joint_graph010','joint_graph_permuted_parent':REP+'__joint_graph_permuted','entropy_parent':'entropy_mean_w8'}
LAMBDAS=(0.,.1,1.,10.)


def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d): Path(p).write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')


def raw_sensitivity(raw, normal, identity):
    """Vectorized reconstruction, independent of production perturb/moment APIs."""
    stream_columns=[1,15,19,23,24,25,26,27,28]
    stream_names=('entropy_series','spilled_series','energy_series','top1_logprob_series','logprob_margin_series',
                  'topk_entropy_series','topk_varentropy_series','topk_renyi2_series','topk_tail_mass_series')
    names=[s+'__'+op for s in stream_names for op in ('level','sd','slope')]
    columns=[names.index(s) for s in normal['active_features']]
    n=len(raw)//8; starts=np.arange(n)*8; position=np.linspace(-.5,.5,8)
    def measure(chunks):
        mean=chunks.mean(axis=1); sd=chunks.std(axis=1)
        slope=np.einsum('t,ntp->np',position,chunks-mean[:,None,:])/(position@position)
        return np.stack((mean,sd,slope),axis=-1).reshape(n,27)[:,columns]
    clean=measure(raw[starts[:,None]+np.arange(8)][:,:,stream_columns]); moments=[]
    for replicate in range(32):
        key=identity+'/reliability-v1/block-perturb/'+str(replicate)
        seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
        blocks=np.random.default_rng(seed).integers(0,4,size=(n,4))
        offsets=(2*blocks[:,:,None]+np.arange(2)).reshape(n,8)
        changed=measure(raw[starts[:,None]+offsets][:,:,stream_columns])
        delta=(changed-clean)/normal['sd']*normal['feature_signs']
        moments.append(np.einsum('ni,nj->ij',delta,delta)/n)
    return np.asarray(moments)


def auc(rows,arm):
    valid=[r for r in rows if r['cell'].startswith('prm') and r['valid'][arm]]
    if not valid: return None
    y=np.concatenate([r['target'] for r in valid]); x=np.concatenate([r['scores'][arm] for r in valid])
    positive,negative=x[y==1],x[y==0]
    if not len(positive) or not len(negative): return None
    return float(np.mean(positive[:,None]>negative[None,:])+.5*np.mean(positive[:,None]==negative[None,:]))


def pb(rows,arm,fixed=False):
    f1=[]
    for cell in sorted({r['cell'] for r in rows if r['cell'].startswith('pb_')}):
        hits={True:[],False:[]}
        for r in (r for r in rows if r['cell']==cell):
            pred=r['fixed_parent_predictions'][arm] if fixed else r['predictions'][arm]
            hits[r['target']==-1].append(bool(r['decision_valid'][arm] and pred==r['target']))
        if not hits[True] or not hits[False]: return None
        ca,ea=np.mean(hits[True]),np.mean(hits[False]); f1.append(2*ca*ea/(ca+ea) if ca+ea else 0.)
    return float(np.mean(f1))


def review():
    manifest,frozen,evaluation=[load(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json')]
    for p,h in {**manifest['hashes'],**frozen['files']}.items(): assert sha(p)==h,p
    assert frozen['manifest_sha256']==sha(OUT/'MANIFEST.json')
    assert evaluation['scores_sha256']==sha(OUT/'SCORES_FROZEN.json')
    rows=evaluation['rows']; assert len(rows)==58 and {r['uid'] for r in rows}=={r['uid'] for r in manifest['selected']}
    counts=Counter(); coverage=Counter(); failures=Counter(); lambdas={}; stability={}; fresh_stability={}; correlations={}
    max_q_error=max_mapping_error=max_weight_error=0.
    release=load(PARENT/'RELEASE.json')
    for cell in {r['cell'] for r in rows}:
        info=release['cells'][cell]; assert sha(info['label_path'])==info['label_opaque_sha256']
        with np.load(info['label_path'],allow_pickle=False) as labels:
            for row in (r for r in rows if r['cell']==cell):
                where=np.flatnonzero(labels['row_ids']==row['row_id']); assert len(where)==1;i=int(where[0])
                if cell.startswith('prm'):
                    a,b=labels['step_flag_offsets'][i:i+2]; target=labels['step_error_flags'][a:b]
                else: target=labels['first_error'][i]
                np.testing.assert_array_equal(target,row['target']); counts['direct_label_joins']+=1
    for row in rows:
        uid=row['uid']; meta=load(OUT/'scores'/f'{uid}.json'); parent=load(PARENT/'scores'/f'{uid}.json')
        detail=meta['regularization']; normal=detail['normalization']
        if 'joint' in detail:
            joint=detail['joint']; jac=joint['jacobian']
            assert joint['converged'] and joint['multistart']['status']=='PASS'
            assert jac['full_global_rank'] and jac['condition_number']<=1e8
            counts['joint_validity_checks']+=1
        with np.load(OUT/'scores'/f'{uid}.npz') as z, np.load(PARENT/'scores'/f'{uid}.npz') as old, np.load(PARENT/'inputs'/f'{uid}.npz') as data:
            all_replicates=raw_sensitivity(data['raw'],normal,row['cell']+'/'+row['row_id'])
            recreated=all_replicates[:24]; q_audit=all_replicates[24:].mean(axis=0)
            counts['fresh_audit_perturbations']+=8
            max_q_error=max(max_q_error,float(np.max(np.abs(recreated-z['Q_per_replicate']))))
            np.testing.assert_allclose(recreated,z['Q_per_replicate'],atol=1e-9,rtol=1e-9); counts['raw_token_replicates']+=24
            np.testing.assert_allclose(z['Q_train'],recreated[:16].mean(axis=0),atol=1e-9,rtol=1e-9)
            np.testing.assert_allclose(z['Q_validation'],recreated[16:].mean(axis=0),atol=1e-9,rtol=1e-9)
            for q in ('Q_train','Q_validation'):
                assert np.linalg.eigvalsh(z[q]).min() >= -1e-9; counts['sensitivity_psd_checks']+=1
            np.testing.assert_array_equal(z['block_full__penalty'],z['Q_train'])
            np.testing.assert_array_equal(np.diag(z['block_diag__penalty']),np.diag(z['Q_train']))
            np.testing.assert_array_equal(np.diag(z['block_diag_permuted__penalty']),np.diag(z['Q_train'])[detail['diagonal_permutation']])
            names=[s+'__'+op for s in ('entropy_series','spilled_series','energy_series','top1_logprob_series','logprob_margin_series',
                'topk_entropy_series','topk_varentropy_series','topk_renyi2_series','topk_tail_mass_series') for op in ('level','sd','slope')]
            columns=[names.index(s) for s in normal['active_features']]; indices=old[REP+'__fit_indices']
            features=(old[REP+'__features'][:,columns]-normal['mean'])/normal['sd']
            features-=features[indices].mean(axis=0);features*=normal['feature_signs']
            fit=features[indices];np.testing.assert_allclose(fit,z['normalization_fit'],atol=1e-12)
            for arm,grid in detail['families'].items():
                core,family=arm.split('__'); weights=z[arm+'__weights_grid']; scaled=z[arm+'__penalty_scaled']
                matrix=z['joint_model_covariance'] if core=='joint' else np.eye(fit.shape[1])
                rhs=z['joint_global_loading'] if core=='joint' else np.asarray(parent['report']['methods'][PARENTS[core+'_parent']]['standardized_weights'])
                np.testing.assert_allclose(np.trace(scaled),np.trace(matrix),atol=1e-11)
                losses=np.array([float(w@z['Q_validation']@w) for w in weights])
                recorded=np.array([np.inf if v is None else v for v in grid['validation_losses']])
                np.testing.assert_allclose(losses,recorded,atol=1e-10,rtol=1e-10)
                tolerance=.01*max(recorded[0],1e-12); selected=next(i for i,v in enumerate(recorded) if v<=min(recorded)+tolerance)
                assert selected==grid['selected_index'] and grid['selected_lambda']==LAMBDAS[selected]
                lambdas.setdefault(arm,Counter())[str(LAMBDAS[selected])]+=1
                stability.setdefault(arm,[]).append(float(recorded[selected]/max(recorded[0],1e-12)))
                fresh_ratio=float((weights[selected]@q_audit@weights[selected])/max(weights[0]@q_audit@weights[0],1e-12))
                fresh_stability.setdefault(arm,[]).append({'uid':uid,'ratio':fresh_ratio})
                for i,lam in enumerate(LAMBDAS):
                    eigen,basis=np.linalg.eigh(matrix+lam*scaled)
                    psd=(basis*np.maximum(eigen,0.))@basis.T;psd=(psd+psd.T)/2
                    raw=np.linalg.solve(psd+grid['grid'][i]['inverse']['ridge']*np.eye(len(rhs)),rhs)
                    normalized=raw/np.std(fit@raw)
                    if grid['grid'][i]['boundary']['flipped']: normalized=-normalized
                    max_weight_error=max(max_weight_error,float(np.max(np.abs(normalized-weights[i]))))
                    np.testing.assert_allclose(normalized,weights[i],atol=1e-8,rtol=1e-8); counts['independent_weight_solves']+=1
                np.testing.assert_allclose(-(features@weights[0]),old[PARENTS[core+'_parent']+'__window'],atol=1e-10,rtol=1e-10)
                counts['lambda_zero_replays']+=1
            starts,ends=old[REP+'__starts'],old[REP+'__ends']
            for arm,method in meta['methods'].items():
                assert method['valid']==row['valid'][arm]==row['decision_valid'][arm]
                if not method['valid']:
                    failures[method['reason']]+=1;assert row['predictions'][arm] is None;continue
                coverage[arm]+=1;window=z[arm+'__window'];steps=z[arm+'__risk']
                if method.get('parent_replay'):
                    np.testing.assert_array_equal(window,old[PARENTS[arm]+'__window'])
                    np.testing.assert_array_equal(steps,old[PARENTS[arm]+'__step']);counts['parent_replays']+=1
                else:
                    source=method.get('grid_source',arm);w=z[source+'__weights_grid'][method['selected_index']]
                    np.testing.assert_allclose(window,-(features@w),atol=1e-11)
                    parent_arm=PARENTS[source.split('__')[0]+'_parent']
                    correlation=float(np.corrcoef(window,old[parent_arm+'__window'])[0,1])
                    correlations.setdefault(arm,[]).append(correlation)
                    x=window[indices,None]
                    mixture=[GaussianMixture(n_components=k,n_init=3,max_iter=300,
                        reg_covar=1e-4,random_state=2026090705).fit(x) for k in (1,2)]
                    assert all(g.converged_ for g in mixture)
                    bic=[float(g.bic(x)) for g in mixture]
                    np.testing.assert_allclose(bic,method['gate_readout']['bic'],atol=1e-9)
                    candidates=np.flatnonzero(steps>mixture[1].means_.mean()) if bic[1]<bic[0] else []
                    gate_prediction=int(candidates[0]) if len(candidates) else -1
                    assert gate_prediction==method['gate_readout']['prediction'];counts['mixture_gate_replays']+=1
                total=np.zeros(row['tokens']);support=np.zeros(row['tokens'])
                for i in range(len(starts)): total[starts[i]:ends[i]]+=window[i];support[starts[i]:ends[i]]+=1
                assert np.all(support>0)
                mapped=np.array([np.max((total/support)[a:b]) for a,b in zip(z['step_starts'],z['step_ends'])])
                max_mapping_error=max(max_mapping_error,float(np.max(np.abs(mapped-steps))))
                np.testing.assert_allclose(mapped,steps,atol=1e-12,rtol=1e-12);counts['span_maps']+=1
                prediction=int(np.argmax(steps)) if method['gate_readout']['prediction']!=-1 else -1
                assert prediction==method['prediction']==row['predictions'][arm]
                np.testing.assert_array_equal(steps,row['scores'][arm])
    fixed={}
    for arm,expected in evaluation['metrics'].items():
        observed=auc(rows,arm)
        assert observed is None and expected['prm']['auroc'] is None or abs(observed-expected['prm']['auroc'])<1e-12
        assert abs(pb(rows,arm)-expected['pb']['macro_f1'])<1e-12;counts['endpoint_checks']+=1
        fixed[arm]=pb(rows,arm,fixed=True)
    parent_peak=load(ROOT/'results/fused_trajectory_readout_pilot_v1/EVALUATION.json')
    for arm,source in PARENTS.items():
        assert evaluation['metrics'][arm]==parent_peak['metrics'][source+'@@parent_peak'];counts['historical_endpoint_replays']+=1
    result={'status':'PASS','counts':dict(counts),'coverage':dict(coverage),'failures':dict(failures),
        'lambda_counts':{a:dict(c) for a,c in lambdas.items()},
        'mean_validation_loss_ratio':{a:float(np.mean(v)) for a,v in stability.items()},
        'fresh_perturbation_audit':{'status':'POST_FREEZE_UNLABELED_DIAGNOSTIC','replicates':[24,31],
            'mean_loss_ratio':{a:float(np.mean([r['ratio'] for r in v])) for a,v in fresh_stability.items()},
            'answer_ratios':fresh_stability,'changed_selection_or_scores':False},
        'mean_parent_score_correlation':{a:float(np.mean(v)) for a,v in correlations.items()},
        'fixed_parent_gate_pb':fixed,'max_raw_sensitivity_error':max_q_error,'max_weight_error':max_weight_error,
        'max_mapping_error':max_mapping_error,'evaluation_sha256':sha(OUT/'EVALUATION.json'),'review_script_sha256':sha(__file__)}
    save(OUT/'REVIEW.json',result)
    print(json.dumps({k:result[k] for k in ('status','counts','failures','max_raw_sensitivity_error','max_weight_error')},indent=2),flush=True)


def render():
    e,a,c,m=[load(OUT/n) for n in ('EVALUATION.json','REVIEW.json','CONTRASTS.json','MANIFEST.json')]
    assert c['state']=='COMPLETE' and len(c['pairs'])==38
    assert a['evaluation_sha256']==c['evaluation_sha256']==sha(OUT/'EVALUATION.json')
    cores,families=m['cores'],m['families']
    lines=['# Stability regularization supports our fusion','',
        'Completed development pilot, 2026-09-07. The core remains IU-PCR / Joint L-SML. This is not untouched confirmation.','',
        'The penalties reduce sensitivity to block perturbations but establish no consistent improvement on both localization tasks. Larger Joint graph lambdas change the scores substantially; they do not yield a clear winner. The next priority is the interface between fused scores and the no-error decision.','',
        '## What this experiment tests','',
        'Keep all fitting windows and the same moment features, groups and score orientation. Resample two-token blocks inside each eight-token window. Sixteen perturbations estimate sensitivity penalties; eight separate perturbations choose lambda from 0, 0.1, 1 and 10. No error label enters the choice. Each candidate has unit original-score standard deviation; choose the smallest lambda within 1% of the baseline instability of the best validation loss.','',
        'Joint uses its native fitted covariance and global loading in the regularized inverse solve. IU and equal aggregation retain their fitted weights and receive an explicitly different identity-head correction. Each component is compared with all three cores. These corrections support fusion; they do not replace it with a new detector.','',
        'The five penalties are isotropic conditioning, existing DUFS graph roughness, diagonal block sensitivity, permuted diagonal sensitivity and the full sensitivity matrix. The matrix includes perturbation bias; it is not established measurement-noise covariance or a LOCA burst model. The full trace and the declared negative-entropy sign anchor are retained. The method is not anchor-free.','',
        '## ProcessBench macro F1 (%) - same 46 answers','',
        '| Core | Parent peak | Isotropic | DUFS graph | Block diagonal | Permuted diagonal | Full block matrix |','|---|---:|---:|---:|---:|---:|---:|']
    for core in cores:
        arms=[core+'_parent']+[core+'__'+f for f in families]
        lines.append('| '+core+' | '+' | '.join(f"{100*e['metrics'][arm]['pb']['macro_f1']:.2f}" for arm in arms)+' |')
    lines+=['','## PRMBench AUROC (valid answers) - coverage matters','','| Core | Parent peak | Isotropic | DUFS graph | Block diagonal | Permuted diagonal | Full block matrix |','|---|---:|---:|---:|---:|---:|---:|']
    for core in cores:
        metrics=[e['metrics'][arm]['prm'] for arm in [core+'_parent']+[core+'__'+f for f in families]]
        lines.append('| '+core+' | '+' | '.join(f"{x['auroc']:.5f} ({x['answers']})" for x in metrics)+' |')
    lines+=['','Do not compare AUROCs with different available answers as a matched leaderboard. Paired PRMB comparisons use the common valid IDs; PB always penalizes failure on the full population.','',
        '## Direct answer to the larger-lambda suggestion','','| Joint graph recipe | PRMB AUROC | Valid PRMB | PB F1 (%) | Mean correlation with lambda-zero score |','|---|---:|---:|---:|---:|']
    for arm in ('joint_parent','joint_graph010_parent','joint_graph_fixed1','joint_graph_fixed10','joint__dufs_graph'):
        metric=e['metrics'][arm];corr=a['mean_parent_score_correlation'].get(arm)
        lines.append(f"| {arm} | {metric['prm']['auroc']:.5f} | {metric['prm']['answers']} | {100*metric['pb']['macro_f1']:.2f} | {corr if corr is not None else 'parent replay'} |")
    lines+=['','## What the label-free selection chose','','| Core / penalty | Lambda counts | Selection sensitivity / baseline | Fresh-perturbation sensitivity / baseline | Mean correlation with parent |','|---|---|---:|---:|---:|']
    for arm,count in a['lambda_counts'].items():
        lines.append(f"| {arm} | {count} | {a['mean_validation_loss_ratio'][arm]:.3f} | {a['fresh_perturbation_audit']['mean_loss_ratio'][arm]:.3f} | {a['mean_parent_score_correlation'][arm]:.3f} |")
    lines+=['','Lower sensitivity on the selection set is partly built into the selection rule. The independent review adds eight unused perturbations (replicates 24-31) after score freezing, retaining every chosen lambda and prediction. This is a post-freeze unlabeled robustness diagnostic, not a new correctness test. Robustness to this block perturbation does not establish better error ranking, factual correctness or robustness to every plausible trace change. Isotropic correction of IU/equal is an invariance control: its scalar scaling cancels after score normalization.','',
        '## Paired comparisons','','Exploratory, unadjusted 95% source-group bootstrap intervals, 1,000 draws. Some draws omit a class in a PB subset and remain undefined; valid-draw counts are retained in CONTRASTS.json. All 38 registered comparisons are saved.','',
        '| Left minus right | Common PRMB N | PRMB delta [95% CI] | PB delta in percentage points [95% CI] |','|---|---:|---|---|']
    for name,pair in c['pairs'].items():
        if not (pair['right'].endswith('_parent') or (pair['left'].endswith('block_full') and pair['right'].endswith('block_full'))): continue
        u=pair['uncertainty']; delta=pair['left_prm']['auroc']-pair['right_prm']['auroc']
        dpb=pair['left_pb']['macro_f1']-pair['right_pb']['macro_f1']
        def formatted(v,ci,scale): return f'{scale*v:+.4f} ['+', '.join(f'{scale*x:+.4f}' for x in ci)+']' if ci is not None else 'undefined'
        lines.append(f"| {name} | {pair['left_prm']['answers']} | {formatted(delta,u['prm_common_valid_ci95'],1)} | {formatted(dpb,u['pb_all_population_ci95'],100)} |")
    lines+=['','## Isolating the error gate','','All candidates use the same GMM rule on all original fused windows, but regularization changes the inputs and can change the gate. The diagnostic below instead fixes each core to its lambda-zero parent decision; it cannot credit unavailable fits.','','| Candidate | PB F1 (%) | PB with fixed parent gate (%) | Valid answers / 58 |','|---|---:|---:|---:|']
    for core in ('iu','joint'):
        for family in families:
            arm=core+'__'+family
            lines.append(f"| {arm} | {100*e['metrics'][arm]['pb']['macro_f1']:.2f} | {100*a['fixed_parent_gate_pb'][arm]:.2f} | {a['coverage'][arm]} |")
    lines+=['','## Review and historical continuity','',
        f"Six scientific tests passed. The independent review reconstructs {a['counts']['raw_token_replicates']} perturbation matrices directly from raw token arrays, verifies {a['counts']['independent_weight_solves']} weight solves, {a['counts']['lambda_zero_replays']} lambda-zero replays, {a['counts']['parent_replays']} exact parent score replays, all 23 endpoints and 58 direct label joins. Selection rules, PSD, normalization, spans and hashes pass. Numerical discrepancies are recorded in REVIEW.json.",'',
        f"Scoring took {load(OUT/'SCORES_FROZEN.json')['seconds']:.1f} seconds on three CPU workers, including perturbations and grid evaluation. Bootstrap time is additional. Six parent endpoints exactly reproduce the previous peak report. Historical 30-long-answer IU 0.70070 and Claude pooled-fit values use different populations/access contracts; they remain context, not matched gains. The 58 answers have already been used for development. No candidate is a publication winner from this pilot alone.",'']
    interpretation=OUT/'INTERPRETATION.md'
    if interpretation.exists(): lines+=interpretation.read_text(encoding='utf-8').splitlines()
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    body=[];table=False
    for line in lines:
        if line.startswith('|---'): continue
        if line.startswith('|'):
            tag='td' if table else 'th'
            if not table: body.append('<div class="scroll"><table>');table=True
            body.append('<tr>'+''.join(f'<{tag}>{html.escape(cell.strip())}</{tag}>' for cell in line.strip('|').split('|'))+'</tr>');continue
        if table:body.append('</table></div>');table=False
        if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line:body.append('<p>'+html.escape(line)+'</p>')
    flow='<div class="flow"><span>One answer, all windows</span><b>→</b><span>Measure feature sensitivity</span><b>→</b><strong>Regularize our fusion weights</strong><b>→</b><span>Localize errors</span></div>'
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stability supports fusion</title><style>body{font:17px/1.6 system-ui;background:#f4f7f8;color:#173d48;margin:0}main{max-width:1160px;margin:auto;padding:32px}h1{font-size:38px;line-height:1.2}h2{margin-top:40px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{border:1px solid #cbdedc;padding:10px;text-align:left}th{background:#dcebe7}.flow{display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:20px;background:#e1edec}.flow strong{padding:12px;background:#175f59;color:white}</style><main>'+flow+''.join(body)+'</main></html>'
    (OUT/'REPORT.html').write_text(page,encoding='utf-8')
    save(OUT/'REPORT_PROVENANCE.json',{'evaluation_sha256':sha(OUT/'EVALUATION.json'),'review_sha256':sha(OUT/'REVIEW.json'),
        'contrasts_sha256':sha(OUT/'CONTRASTS.json'),'report_script_sha256':sha(__file__),
        'interpretation_sha256':sha(interpretation) if interpretation.exists() else None})
    print('REPORT.md and REPORT.html written.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('review','report'),required=True)
    {'review':review,'report':render}[parser.parse_args().phase]()
