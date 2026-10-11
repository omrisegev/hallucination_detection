#!/usr/bin/env python
"""Source-group nested diagnostic: feature choice versus fusion weighting.

Uses prepared v11 inputs. Saves one checkpoint per bank/outer fold, so a CPU
run can resume. All feature masks and normalization are fitted in the relevant
training partition, including inner validation; LR selector C=1 was fixed
before the run. New OOF baselines are distinct from historical cached LR.
"""
from __future__ import annotations
import os
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_env] = '1'
import argparse, csv, hashlib, json, math, sys, time, traceback
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.advisor_lr import (C_GRID, SELECTOR_C, fit_lr, fit_labelfree,
    balanced_log_loss, rank_half, restricted_sets, as_dict)

BANKS = ['N11', 'N13', 'N22', 'N40', 'N53', 'N14', 'N16', 'N25', 'N43', 'N56']
LF = ('LSML_CONT', 'IU_PCR')
ARM_SELECTORS = {'LR_FULL': 'ALL', 'LR__LR_HALF': 'LR_HALF',
                 'LR__LSML_HALF': 'LSML_HALF', 'LR__IU_HALF': 'IU_HALF',
                 'LR_R1': 'R1', 'LR_R2': 'R2', 'LR_R3': 'R3'}
LF_ARMS = {'LSML_FULL': ('LSML_CONT', 'ALL'), 'IU_FULL': ('IU_PCR', 'ALL'),
           'LSML__LR_HALF': ('LSML_CONT', 'LR_HALF'),
           'LSML__LSML_HALF': ('LSML_CONT', 'LSML_HALF'),
           'IU__LR_HALF': ('IU_PCR', 'LR_HALF'), 'IU__IU_HALF': ('IU_PCR', 'IU_HALF')}
ARMS = list(ARM_SELECTORS) + list(LF_ARMS)
FAMILIES = {
 'F1': ['q15_H1','q15_VE1','logprob_margin','true_tail50','energy_level','ct7_H0lim','ct7_ve0','ct7_ve0.75','ct7_ve1','hist_entropy_series','hist_topk_entropy_series','hist_topk_varentropy_series','hist_topk_renyi2_series','hist_topk_tail_mass_series','hist_energy_series','hist_energy_rolling_min','hist_top1_logprob_series','hist_logprob_margin_series'],
 'F2': ['chosen_surprisal','realized_z','realized_drv','ct7_chosen_std_excess','hist_spilled_series','hist_spilled_rolling_min'],
 'F3': ['energy_innovation','top15_turnover','top50_js','bocpd_p0','ct7_H0lim_prefix_innovation','ct7_bocpd_residual','evidence_drop_risk'],
 'F4': ['H1_first_token','H1_slope','H1_jump','H1_frac_above_z'],
 'F5': ['dominant_freq16','hist_entropy_rolling_spectral_entropy','hist_entropy_rolling_low_band_power','hist_entropy_rolling_high_band_power','hist_entropy_rolling_hl_ratio','hist_entropy_rolling_dominant_freq','hist_entropy_rolling_spectral_centroid','hist_entropy_stft_high_series','hist_entropy_stft_frame_entropy','hist_entropy_rolling_tail_ratio','hist_entropy_pe_series','hist_entropy_rolling_rs_hurst'],
 'F6': ['hist_entropy_sw_var_series','hist_entropy_cusum_abs_series','hist_spilled_sw_var_series','hist_spilled_cusum_abs_series','hist_energy_sw_var_series','hist_energy_cusum_abs_series'],
 'F7': ['digit_alternative','digit_spread','digit_alternative_innovation']}
FAMILY = {c:f for f,cols in FAMILIES.items() for c in cols}


def dump(path, value):
    def default(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=default, allow_nan=False))
    tmp.replace(path)


def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def load_inputs(inp, bank):
    p = np.load(inp/'population.npz', allow_pickle=False)
    b = np.load(inp/f'bank_V_{bank}.npz', allow_pickle=False)
    X, off = b['V'].astype(float), p['offsets'].astype(int)
    y = p['y'].astype(int); groups = p['group'].astype(str); folds = p['fold'].astype(int)
    names = b['columns'].astype(str).tolist()
    assert len(X) == len(y) == off[-1]
    assert len(groups) == len(folds) == len(off)-1
    assert np.array_equal(off, b['offsets'])
    assert np.isfinite(X).all()
    assert len(names) == X.shape[1]
    if 'idx' in b.files: assert np.array_equal(p['idx'], b['idx'])
    split = json.loads((inp/'FOLDS_V2.json').read_text())['prmbench']
    for g, f in zip(groups, folds): assert split['outer'][g] == int(f)
    for f in range(5):
        assert not set(groups[folds==f]) & set(groups[folds!=f])
    return dict(X=X,y=y,off=off,names=names,groups=groups,folds=folds,
                step_folds=np.repeat(folds,np.diff(off)),
                classification=p['classification'].astype(str),idx=p['idx'].astype(str),
                token_lengths=p['token_lengths'] if 'token_lengths' in p.files else None,
                split=split)


def selector_partition(X, y):
    n = X.shape[1]; masks = {'ALL': np.arange(n)}; models = {}; failures = {}
    rank = fit_lr(X,y,SELECTOR_C)
    masks['LR_HALF'] = rank_half(rank.weight_std)
    masks.update(restricted_sets(rank.weight_std,X))
    models['LR_SELECTOR_C1'] = rank
    for method, label in [('LSML_CONT','LSML_HALF'),('IU_PCR','IU_HALF')]:
        try:
            m = fit_labelfree(X,method)
            masks[label] = rank_half(m.weight_std); models[method] = m
        except Exception as e:
            failures[label] = f'{type(e).__name__}: {e}'
    return masks, models, failures


def tune_lr(X, y, outer_train, inner_step, mask_cache, selector, log):
    losses = []; errors = []
    for C in C_GRID:
        fold_loss = []
        try:
            for inner in range(5):
                tr = outer_train & (inner_step!=inner); va = outer_train & (inner_step==inner)
                if not tr.any() or not va.any(): raise ValueError('Empty inner partition')
                cols = mask_cache[inner][0].get(selector)
                if cols is None: raise RuntimeError(mask_cache[inner][2].get(selector,'Missing mask'))
                model = fit_lr(X[tr][:,cols],y[tr],C)
                fold_loss.append(balanced_log_loss(y[va],model.predict(X[va][:,cols])))
            losses.append(float(np.mean(fold_loss)))
        except Exception as e:
            losses.append(None); errors.append({'C':C,'reason':f'{type(e).__name__}: {e}'})
    eligible = [i for i,v in enumerate(losses) if v is not None and np.isfinite(v)]
    if not eligible: raise RuntimeError(f'No valid C for {selector}: {errors}')
    best = min(eligible,key=lambda i:(losses[i],C_GRID[i]))
    log.update({'C':C_GRID[best],'inner_balanced_log_loss':losses,'C_errors':errors,
                'inner_selected_columns':{str(k):v[0][selector].tolist() for k,v in mask_cache.items()}})
    return C_GRID[best]


def outer_stage(inp,out,bank,outer):
    dest = out/'checkpoints'/bank/f'outer_{outer}'
    if (dest/'DONE.json').exists(): return
    dest.mkdir(parents=True,exist_ok=True)
    t = time.time(); d=load_inputs(inp,bank)
    X,y,folds = d['X'],d['y'],d['step_folds']
    train, test = folds!=outer, folds==outer
    imap=d['split']['inner'][str(outer)]
    inner_ans=np.array([imap[g] if f!=outer else -1 for g,f in zip(d['groups'],d['folds'])])
    inner_step=np.repeat(inner_ans,np.diff(d['off']))
    for k in range(5):
        gs_train=set(d['groups'][(d['folds']!=outer)&(inner_ans!=k)])
        gs_valid=set(d['groups'][(d['folds']!=outer)&(inner_ans==k)])
        assert not gs_train & gs_valid
    print(bank,outer,'selectors',round(time.time()-t,1),flush=True)
    cache={}
    for k in range(5):
        inner_tr=train&(inner_step!=k)
        cache[k]=selector_partition(X[inner_tr],y[inner_tr])
    masks, models, selector_fail = selector_partition(X[train],y[train])
    diag={'bank':bank,'outer_fold':outer,'names':d['names'],
          'n_train_steps':int(train.sum()),'n_test_steps':int(test.sum()),
          'train_groups':sorted(set(d['groups'][d['folds']!=outer])),
          'test_groups':sorted(set(d['groups'][d['folds']==outer])),
          'inner_valid_groups':{str(k):sorted(set(d['groups'][(d['folds']!=outer)&(inner_ans==k)])) for k in range(5)},
          'selector_models':{k:as_dict(v) for k,v in models.items()},
          'outer_masks':{k:v.tolist() for k,v in masks.items()},'selector_failures':selector_fail,
          'arms':{}}
    scores={}; failures={}
    for arm,selector in ARM_SELECTORS.items():
        info={'selector':selector}
        try:
            if selector not in masks: raise RuntimeError(selector_fail.get(selector,'Missing mask'))
            C=tune_lr(X,y,train,inner_step,cache,selector,info)
            cols=masks[selector]; m=fit_lr(X[train][:,cols],y[train],C)
            scores[arm]=m.predict(X[test][:,cols]); info['model']=as_dict(m)
            info['columns']=cols.tolist()
            models[arm]=m
        except Exception as e:
            failures[arm]=f'{type(e).__name__}: {e}'; info['failed']=failures[arm]
        diag['arms'][arm]=info
        print(bank,outer,arm,round(time.time()-t,1),failures.get(arm,''),flush=True)
    for arm,(method,selector) in LF_ARMS.items():
        try:
            if selector not in masks: raise RuntimeError(selector_fail.get(selector,'Missing mask'))
            cols=masks[selector]
            m=models[method] if selector=='ALL' else fit_labelfree(X[train][:,cols],method)
            scores[arm]=m.predict(X[test][:,cols])
            diag['arms'][arm]={'method':method,'selector':selector,'columns':cols.tolist(),'model':as_dict(m)}
        except Exception as e:
            failures[arm]=f'{type(e).__name__}: {e}'; diag['arms'][arm]={'failed':failures[arm]}
        print(bank,outer,arm,round(time.time()-t,1),failures.get(arm,''),flush=True)
    # Drop-one/family probes: fixed C selected by full nested training baseline.
    # Their masks are predeclared, so there is no feature-selecting test access.
    if bank in ('N16','N56') and 'LR_FULL' in scores:
        C=diag['arms']['LR_FULL']['C']
        probes=[('drop_'+name,np.array([k for k in range(X.shape[1]) if k!=j]))
                for j,name in enumerate(d['names'])]
        for family in sorted(set(FAMILY.get(n,'unmapped') for n in d['names'])):
            cols=np.array([j for j,n in enumerate(d['names']) if FAMILY.get(n,'unmapped')!=family])
            if len(cols): probes.append(('drop_family_'+family,cols))
        for arm,cols in probes:
            try:
                m=fit_lr(X[train][:,cols],y[train],C)
                scores[arm]=m.predict(X[test][:,cols])
                diag['arms'][arm]={'columns':cols.tolist(),'model':as_dict(m),'C_fixed_from':'LR_FULL'}
            except Exception as e:
                failures[arm]=f'{type(e).__name__}: {e}'; diag['arms'][arm]={'failed':failures[arm]}
    np.savez_compressed(dest/'SCORES.npz',rows=np.flatnonzero(test),**scores)
    dump(dest/'MODELS.json',diag)
    dump(dest/'DONE.json',{'status':'COMPLETE_WITH_EXPLICIT_FAILURES' if failures else 'COMPLETE',
         'failures':failures,'runtime_s':round(time.time()-t,2),
         'source_sha256':sha(Path(__file__)),'helper_sha256':sha(ROOT/'spectral_utils/advisor_lr.py'),
         'bank_input_sha256':sha(inp/f'bank_V_{bank}.npz'),'population_sha256':sha(inp/'population.npz')})


def within(score, off, y):
    n=len(off)-1; a=np.full(n,np.nan); top=np.full(n,np.nan); am=np.full(n,-1,int)
    for i,(b,e) in enumerate(zip(off[:-1],off[1:])):
        ss=score[b:e]; yy=y[b:e]; n1=int(yy.sum()); n0=len(yy)-n1
        if not np.isfinite(ss).all(): continue
        # Canonical tie tolerance, first step.
        at=int(np.flatnonzero(ss >= ss.max()-8*np.finfo(float).eps)[0]); am[i]=at
        if n1:
            top[i]=float(yy[at])
            if n0: a[i]=(rankdata(ss)[yy.astype(bool)].sum()-n1*(n1+1)/2)/(n1*n0)
    return a,top,am


def cosine(a,b):
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>0 else None


def correlation(a,b):
    return float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None


def bootstrap_ci(values,groups,weights):
    good=np.isfinite(values); unique,inv=np.unique(groups,return_inverse=True)
    sums=np.bincount(inv,weights=np.where(good,values,0.),minlength=len(unique))
    counts=np.bincount(inv,weights=good,minlength=len(unique))
    denom=weights@counts; sample=(weights@sums)/np.where(denom>0,denom,np.nan)
    finite=sample[np.isfinite(sample)]
    return [float(np.quantile(finite,.025)),float(np.quantile(finite,.975))] if len(finite) else [None,None]


def aggregate(inp,out):
    summaries=[]; weight_rows=[]; agreement=[]; ablations=[]; fourway=[]; ci_rows=[]; all_status=[]
    t=time.time(); reference=None
    for bank in BANKS:
        d=load_inputs(inp,bank); n=len(d['y']); score={}; folds=[]
        if not all((out/'checkpoints'/bank/f'outer_{k}'/'DONE.json').exists() for k in range(5)):
            all_status.append({'bank':bank,'status':'PENDING'}); continue
        for k in range(5):
            dest=out/'checkpoints'/bank/f'outer_{k}'
            models=json.loads((dest/'MODELS.json').read_text()); folds.append(models)
            z=np.load(dest/'SCORES.npz',allow_pickle=False)
            for arm in z.files:
                if arm=='rows': continue
                score.setdefault(arm,np.full(n,np.nan))[z['rows']]=z[arm]
        for arm in ARMS: score.setdefault(arm,np.full(n,np.nan))
        arrays={}; metrics={}; bankrows=[]
        for arm,s in score.items():
            a,top,am=within(s,d['off'],d['y']); arrays[arm+'__auc']=a; arrays[arm+'__top1']=top; arrays[arm+'__argmax']=am
            metrics[arm]=(a,top,am)
            complete=bool(np.isfinite(s).all())
            row={'bank':bank,'n_channels':len(d['names']),'arm':arm,'complete':complete,
                 'n_scored_steps':int(np.isfinite(s).sum()),'n_auc_answers':int(np.isfinite(a).sum()),
                 'n_top1_answers':int(np.isfinite(top).sum()),
                 'within_answer_auc':float(np.nanmean(a)) if np.isfinite(a).any() else None,
                 'top1_on_error':float(np.nanmean(top)) if np.isfinite(top).any() else None}
            row['partial_only']=not complete
            summaries.append(row); bankrows.append(row)
        np.savez_compressed(out/f'OOF_{bank}.npz',**score)
        np.savez_compressed(out/f'PER_ANSWER_{bank}.npz',**arrays)
        dump(out/f'SUMMARY_{bank}.json',bankrows)
        # Source-cluster paired intervals on answer AUC (not global OOF ranks).
        unique=np.unique(d['groups']); rng=np.random.default_rng(20261011)
        W=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=5000).astype(float)
        comparisons=[('LR_FULL','LSML_FULL'),('LR_FULL','IU_FULL'),
                     ('LR__LR_HALF','LR_FULL'),('LR__LSML_HALF','LR_FULL'),('LR__IU_HALF','LR_FULL'),
                     ('LSML__LR_HALF','LSML__LSML_HALF'),('IU__LR_HALF','IU__IU_HALF'),
                     ('LR_R1','LR_FULL'),('LR_R2','LR_FULL'),('LR_R3','LR_FULL')]
        for left,right in comparisons:
            delta=metrics[left][0]-metrics[right][0]; good=np.isfinite(delta)
            lo,hi=bootstrap_ci(delta,d['groups'],W)
            ci_rows.append({'bank':bank,'left':left,'right':right,'estimand':'paired within-answer AUC',
               'delta':float(np.nanmean(delta)) if good.any() else None,'lo95':lo,'hi95':hi,
               'n_common_answers':int(good.sum()),'both_complete':bool(np.isfinite(score[left]).all() and np.isfinite(score[right]).all()),
               'inference':'descriptive pointwise; exploratory selected/multiple contrasts'})
        for outer,m in enumerate(folds):
            common_sd=np.asarray(m['selector_models']['LR_SELECTOR_C1']['training_scale'])
            ws={}
            for label,arm in [('LR','LR_FULL'),('LSML','LSML_FULL'),('IU','IU_FULL')]:
                model=m['arms'].get(arm,{}).get('model')
                if model is None: continue
                weight=np.asarray(model['weight']); common=weight*common_sd
                norm=np.sum(np.abs(common)); normal=common/norm if norm else common
                ws[label]=normal
                for j,name in enumerate(d['names']):
                    weight_rows.append({'bank':bank,'outer_fold':outer,'method':label,'channel':name,
                        'family':FAMILY.get(name,'unmapped'),'effective_weight_on_V':weight[j],
                        'common_training_std_weight':common[j],'l1_normalized_common_weight':normal[j],
                        'selected_in_half':int(j in m['outer_masks'].get({'LR':'LR_HALF','LSML':'LSML_HALF','IU':'IU_HALF'}[label],[])),
                        'affine_max_abs':model['diagnostic']['affine_max_abs']})
            for left,right in combinations(ws,2):
                agreement.append({'bank':bank,'outer_fold':outer,'left':left,'right':right,
                    'signed_spearman':correlation(ws[left],ws[right]),'abs_spearman':correlation(np.abs(ws[left]),np.abs(ws[right])),
                    'cosine':cosine(ws[left],ws[right]),'sign_agreement':float(np.mean(np.sign(ws[left])==np.sign(ws[right])))})
        if bank in ('N16','N56'):
            for arm,(a,top,am) in metrics.items():
                if not arm.startswith('drop_'): continue
                ref=metrics['LR_FULL']; delta=a-ref[0]; good=np.isfinite(delta)
                lo,hi=bootstrap_ci(delta,d['groups'],W)
                ablations.append({'bank':bank,'probe':arm,'within_auc_delta':float(np.nanmean(delta)) if good.any() else None,
                    'lo95':lo,'hi95':hi,'top1_delta':float(np.nanmean(top-ref[1])) if np.isfinite(top-ref[1]).any() else None,
                    'n_common_answers':int(good.sum()),'complete':bool(np.isfinite(score[arm]).all()),
                    'C':'fixed at full LR train-selected C','inference':'exploratory; correlated-feature omission, not causal importance'})
            top_lr=metrics['LR_FULL'][1]
            lengths=np.diff(d['off']); error=[i for i,(b,e) in enumerate(zip(d['off'][:-1],d['off'][1:])) if d['y'][b:e].any()]
            for method,arm in [('LSML','LSML_FULL'),('IU','IU_FULL')]:
                top_u=metrics[arm][1]
                for i in error:
                    if not np.isfinite(top_u[i]+top_lr[i]): continue
                    nstep=int(lengths[i]); errorat=int(np.flatnonzero(d['y'][d['off'][i]:d['off'][i+1]])[0])
                    total_tokens=int(np.sum(d['token_lengths'][d['off'][i]:d['off'][i+1]])) if d['token_lengths'] is not None else None
                    fourway.append({'bank':bank,'method':method,'idx':d['idx'][i], 'group':d['groups'][i],
                         'classification':d['classification'][i], 'n_steps':nstep,'total_tokens':total_tokens,
                         'length_bin':'<=10' if nstep<=10 else ('11-20' if nstep<=20 else '>20'),
                         'first_error_relative_position':errorat/max(nstep-1,1),
                         'position_bin':min(4,int(5*errorat/max(nstep-1,1))),
                         'ours_correct':int(top_u[i]),'lr_correct':int(top_lr[i]),
                         'outcome':'both' if top_u[i] and top_lr[i] else ('ours_only' if top_u[i] else ('lr_only' if top_lr[i] else 'neither')),
                         'ours_argmax':int(metrics[arm][2][i]),'lr_argmax':int(metrics['LR_FULL'][2][i])})
        all_status.append({'bank':bank,'status':'COMPLETE','failed_arms':[r['arm'] for r in bankrows if not r['complete']]})
        print('aggregate',bank,round(time.time()-t,1),flush=True)
    write_csv(out/'METRICS.csv',summaries); write_csv(out/'WEIGHTS.csv',weight_rows)
    write_csv(out/'WEIGHT_AGREEMENT.csv',agreement); write_csv(out/'PAIRED_CONTRASTS.csv',ci_rows)
    write_csv(out/'ABLATIONS.csv',ablations); write_csv(out/'FOURWAY_PER_ANSWER.csv',fourway)
    # Grouped descriptive frequencies; no outcome-dependent bins.
    four_summ=[]
    for axis in ('classification','position_bin','length_bin'):
        keys=sorted(set((r['bank'],r['method'],str(r[axis])) for r in fourway))
        for bank,method,value in keys:
            rows=[r for r in fourway if r['bank']==bank and r['method']==method and str(r[axis])==value]
            counts={k:sum(r['outcome']==k for r in rows) for k in ('ours_only','lr_only','both','neither')}
            four_summ.append({'bank':bank,'method':method,'axis':axis,'value':value,'n':len(rows),**counts})
    write_csv(out/'FOURWAY_SUMMARY.csv',four_summ)
    # Across-outer model stability on normalized common-scale weights.
    stability=[]; family_mass=[]
    for bank in BANKS:
        for method in ('LR','LSML','IU'):
            rs=[r for r in weight_rows if r['bank']==bank and r['method']==method]
            by_fold={k:[r for r in rs if r['outer_fold']==k] for k in range(5)}
            available=[k for k,r in by_fold.items() if r]
            W=np.array([[r['l1_normalized_common_weight'] for r in by_fold[k]] for k in available])
            if not len(W): continue
            names=[r['channel'] for r in by_fold[available[0]]]
            co=[cosine(W[i],W[j]) for i,j in combinations(range(len(W)),2)]
            for j,name in enumerate(names):
                stability.append({'bank':bank,'method':method,'channel':name,'n_folds':len(W),
                    'mean_weight':float(W[:,j].mean()),'sd_weight':float(W[:,j].std()),
                    'sign_agreement_folds':float(abs(np.sign(W[:,j]).mean())),
                    'half_selection_frequency':float(np.mean([by_fold[k][j]['selected_in_half'] for k in available])),
                    'mean_fold_cosine':float(np.mean(co)) if co else None})
            for fam in sorted(set(FAMILY.get(n,'unmapped') for n in names)):
                ids=[j for j,n in enumerate(names) if FAMILY.get(n,'unmapped')==fam]
                family_mass.append({'bank':bank,'method':method,'family':fam,'n_channels':len(ids),
                    'mean_abs_weight_mass':float(np.abs(W[:,ids]).sum(1).mean())})
    write_csv(out/'WEIGHT_STABILITY.csv',stability);write_csv(out/'FAMILY_MASS.csv',family_mass)
    dump(out/'STATUS.json',{'status':'COMPLETE' if len(summaries) and all(s['status']=='COMPLETE' for s in all_status) else 'PARTIAL',
         'banks':all_status,'max_affine_error':max([r['affine_max_abs'] for r in weight_rows] or [0]),
         'n_arms':len(ARMS),'n_answer_groups':707,'inference':'internal exploratory source-group OOF',
         'historical_LR_equivalence':'not asserted: historic split provenance unavailable',
         'summary':summaries,'source_sha256':sha(Path(__file__))})
    figures(out,summaries,weight_rows,ablations,family_mass)


def figures(out,metrics,weights,ablations,family_mass):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    colors={'LR_FULL':'#e2632c','LSML_FULL':'#206b9c','IU_FULL':'#28977c'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    for ax,banks in zip(axes,[BANKS[:5],BANKS[5:]]):
        for arm,color in colors.items():
            values=[next((r['within_answer_auc'] for r in metrics if r['bank']==b and r['arm']==arm and r['complete']),np.nan) for b in banks]
            ax.plot(range(5),values,'o-',label=arm.replace('_FULL',''),color=color)
        ax.set_xticks(range(5),banks);ax.set_xlabel('Feature bank');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Mean within-answer AUC')
    fig.suptitle('New source-group OOF reference: more features versus fusion method')
    fig.tight_layout();save(fig,figdir/'bank_gap');plt.close(fig)
    for bank in ('N16','N56'):
        fig,ax=plt.subplots(figsize=(13,4.5 if bank=='N16' else 7))
        rs=[r for r in weights if r['bank']==bank];names=list(dict.fromkeys(r['channel'] for r in rs));x=np.arange(len(names))
        for k,method in enumerate(('LR','LSML','IU')):
            vals=[np.mean([r['l1_normalized_common_weight'] for r in rs if r['method']==method and r['channel']==n]) for n in names]
            ax.bar(x+(k-1)*.25,vals,.25,label=method)
        ax.set_xticks(x,names,rotation=70 if bank=='N16' else 90,ha='right',fontsize=8)
        ax.set_ylabel('Signed contribution per training SD / sum |weights|')
        ax.set_title(f'{bank}: common-scale weights; means across source-group folds')
        ax.legend();ax.axhline(0,color='#777',lw=.8);fig.tight_layout();save(fig,figdir/f'weights_{bank}');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,bank in zip(axes,('N16','N56')):
        arms=['LR_FULL','LR__LR_HALF','LR__LSML_HALF','LR__IU_HALF','LSML_FULL','LSML__LR_HALF','LSML__LSML_HALF','IU_FULL','IU__LR_HALF','IU__IU_HALF']
        rs=[next((r for r in metrics if r['bank']==bank and r['arm']==arm),{}) for arm in arms]
        vals=[r.get('within_answer_auc') if r.get('complete') else np.nan for r in rs]
        ax.barh(np.arange(len(arms)),vals,color=['#e2632c']*4+['#206b9c']*3+['#28977c']*3)
        ax.set_yticks(np.arange(len(arms)),arms,fontsize=8);ax.set_xlim(.5,.9);ax.set_title(bank)
        ax.set_xlabel('Within-answer AUC (incomplete arm omitted)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Matched half-bank crossover: selection versus weighting')
    fig.tight_layout();save(fig,figdir/'crossover');plt.close(fig)


def save(fig,path):
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=170,bbox_inches='tight')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',type=Path,default=ROOT/'results/advisor_completion_2026-10-11/inputs')
    ap.add_argument('--out',type=Path,default=ROOT/'results/advisor_completion_2026-10-11/lr')
    ap.add_argument('--bank',choices=BANKS);ap.add_argument('--outer',type=int,choices=range(5));ap.add_argument('--aggregate',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    protocol={'frozen':True,'purpose':'single diagnostic selection-versus-weighting experiment, not a new label-free method',
      'bank_order':BANKS,'outer_split':'FOLDS_V2 prmbench.outer','inner_split':'FOLDS_V2 prmbench.inner[outer]',
      'C_GRID':C_GRID,'selector_C':SELECTOR_C,'selector_C_note':'fixed before run; fitted fresh in every inner/outer training partition',
      'half_count':'ceil(n_channels/2); stable tie by bank column order',
      'selection_scale':'effective weights on V multiplied by common training SD',
      'ranking_metric':'per-answer within-answer AUC averaged; no concatenated-global OOF AUC',
      'R1_R2_R3':'historical definitions, training-only C1 selector inside each validation partition',
      'drop_one_family':'N16/N56 only, fixed full-LR inner-selected C per outer',
      'LSML_failure':'K<3 explicit failed arm, no fallback; grouping degeneracy recorded',
      'bootstrap':{'draws':5000,'seed':20261011,'unit':'canonical source group'},
      'interpretation':'internal validation on previously inspected data; supervised-selected LF arms are supervised diagnostics',
      'script_sha256':sha(Path(__file__)),'helper_sha256':sha(ROOT/'spectral_utils/advisor_lr.py')}
    p=args.out/'PROTOCOL.json'
    if p.exists():
        old=json.loads(p.read_text())
        for key in ('C_GRID','selector_C','half_count','outer_split','inner_split'):
            assert old[key]==json.loads(json.dumps(protocol[key])),f'Protocol changed: {key}'
    else: dump(p,protocol)
    if args.aggregate: aggregate(args.inputs,args.out);return
    for bank in ([args.bank] if args.bank else BANKS):
        for outer in ([args.outer] if args.outer is not None else range(5)):
            outer_stage(args.inputs,args.out,bank,outer)
    if args.bank is None: aggregate(args.inputs,args.out)


if __name__=='__main__': main()
