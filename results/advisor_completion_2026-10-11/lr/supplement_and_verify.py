"""Independent arithmetic/replay acceptance checks on saved LR diagnostics."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv,json,hashlib
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
INP=OUT.parent/'inputs'

def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)

def main():
    pop=np.load(INP/'population.npz');classification=pop['classification'].astype(str)
    off=pop['offsets'].astype(int);y=pop['y'];step_fold=np.repeat(pop['fold'],np.diff(off))
    groups=pop['group'].astype(str);F=json.loads((INP/'FOLDS_V2.json').read_text())['prmbench']
    maximum=0.;partition_count=0;failures=[];contrast_pairs=[]
    population_hash=hashlib.sha256((INP/'population.npz').read_bytes()).hexdigest()
    for bank in ['N11','N13','N22','N40','N53','N14','N16','N25','N43','N56']:
        X=np.load(INP/f'bank_V_{bank}.npz')['V']
        bank_hash=hashlib.sha256((INP/f'bank_V_{bank}.npz').read_bytes()).hexdigest()
        for outer in range(5):
            dest=OUT/'checkpoints'/bank/f'outer_{outer}'
            done=json.loads((dest/'DONE.json').read_text())
            assert done['population_sha256']==population_hash
            assert done['bank_input_sha256']==bank_hash
            model=json.loads((dest/'MODELS.json').read_text());score=np.load(dest/'SCORES.npz')
            rows=np.flatnonzero(step_fold==outer)
            assert np.array_equal(rows,score['rows'])
            train=set(model['train_groups']);test=set(model['test_groups'])
            assert not train&test
            assert test==set(groups[pop['fold']==outer])
            for inner,valid in model['inner_valid_groups'].items():
                assert set(valid)<=train
                assert all(F['inner'][str(outer)][g]==int(inner) for g in valid)
            for arm,info in model['arms'].items():
                if 'failed' in info:
                    failures.append({'bank':bank,'outer':outer,'arm':arm,'reason':info['failed']});continue
                cols=np.array(info['columns']);m=info['model']
                replay=X[rows][:,cols]@np.array(m['weight'])+m['intercept']
                error=float(np.max(np.abs(replay-score[arm])))
                maximum=max(maximum,error)
                assert error<=1e-9,(bank,outer,arm,error)
            # Check masks belong to training selector: no averaged global coefs.
            select=model['selector_models']['LR_SELECTOR_C1']
            coef=np.array(select['weight_std']);k=int(np.ceil(len(coef)/2))
            expected=np.sort(np.lexsort((np.arange(len(coef)),-np.abs(coef)))[:k])
            assert np.array_equal(expected,model['outer_masks']['LR_HALF'])
            if bank in ('N16','N56'):
                lr=model['arms']['LR_FULL']['model'];coef=np.array(lr['weight_std'])
                corr=np.corrcoef(X[step_fold!=outer],rowvar=False);median=np.median(np.abs(coef))
                for left in range(len(coef)):
                    for right in range(left+1,len(coef)):
                        if (abs(corr[left,right])>=.8 and
                            np.sign(coef[left])*np.sign(coef[right])*np.sign(corr[left,right])<0 and
                            abs(coef[left])>median and abs(coef[right])>median):
                            contrast_pairs.append({'bank':bank,'outer':outer,
                               'left':model['names'][left],'right':model['names'][right],
                               'training_correlation':float(corr[left,right]),
                               'LR_standardized_weight_left':float(coef[left]),
                               'LR_standardized_weight_right':float(coef[right]),
                               'interpretation':'correlated contrast pair; coefficient magnitude alone is not independent contribution'})
            partition_count+=1
    categories=[]
    for bank in ('N16','N56'):
        pa=np.load(OUT/f'PER_ANSWER_{bank}.npz')
        probes=[k.removesuffix('__auc') for k in pa.files if k.startswith('drop_') and k.endswith('__auc')]
        for probe in probes:
            a=pa[probe+'__auc'];top=pa[probe+'__top1']
            for category in sorted(set(classification)):
                subset=classification==category
                diff=a-pa['LR_FULL__auc'];good=subset&np.isfinite(diff)
                td=top-pa['LR_FULL__top1'];tg=subset&np.isfinite(td)
                categories.append({'bank':bank,'probe':probe,'classification':category,
                    'n_auc_answers':int(good.sum()),'within_auc_delta':float(diff[good].mean()) if good.any() else None,
                    'n_top1_answers':int(tg.sum()),'top1_delta':float(td[tg].mean()) if tg.any() else None,
                    'inference':'descriptive exploratory subgroup; no multiplicity-adjusted significance claim'})
    write('ABLATIONS_BY_TYPE.csv',categories)
    if contrast_pairs:write('CORRELATED_CONTRAST_PAIRS.csv',contrast_pairs)
    # Matched-mask contrasts directly separate the requested factors. These
    # are metric differences, not an additive causal attribution of the gap.
    unique,inv=np.unique(groups,return_inverse=True)
    rng=np.random.default_rng(20261011)
    W=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=5000).astype(float)
    factor_rows=[];top_ci=[]
    for bank in ['N11','N13','N22','N40','N53','N14','N16','N25','N43','N56']:
        pa=np.load(OUT/f'PER_ANSWER_{bank}.npz')
        for method,tag in [('LSML','LSML'),('IU','IU')]:
            for metric in ['auc','top1']:
                keys={'L_L':f'LR__LR_HALF__{metric}','L_U':f'LR__{tag}_HALF__{metric}',
                      'U_L':f'{method}__LR_HALF__{metric}','U_U':f'{method}__{tag}_HALF__{metric}'}
                values={k:pa[v] for k,v in keys.items()}
                contrast={'selection_at_LR':values['L_L']-values['L_U'],
                          'selection_at_LF':values['U_L']-values['U_U'],
                          'weighting_at_LR_mask':values['L_L']-values['U_L'],
                          'weighting_at_LF_mask':values['L_U']-values['U_U'],
                          'interaction':values['L_L']-values['L_U']-values['U_L']+values['U_U']}
                for name,diff in contrast.items():
                    good=np.isfinite(diff)
                    sums=np.bincount(inv,weights=np.where(good,diff,0),minlength=len(unique))
                    counts=np.bincount(inv,weights=good,minlength=len(unique))
                    denom=W@counts;boot=(W@sums)/np.where(denom>0,denom,np.nan)
                    finite=boot[np.isfinite(boot)]
                    factor_rows.append({'bank':bank,'method':method,'metric':metric,'factor':name,
                       'delta':float(diff[good].mean()) if good.any() else None,
                       'lo95':float(np.quantile(finite,.025)) if len(finite) else None,
                       'hi95':float(np.quantile(finite,.975)) if len(finite) else None,
                       'n_common_answers':int(good.sum()),
                       'complete_four_arms':all(int(np.isfinite(v).sum())==int(np.isfinite(pa['LR_FULL__'+metric]).sum()) for v in values.values()),
                       'inference':'exploratory pointwise source-group interval; no multiple-comparison promotion'})
            # Top1 comparison of full-bank methods against the strict LR.
            diff=pa[method+'_FULL__top1']-pa['LR_FULL__top1'];good=np.isfinite(diff)
            sums=np.bincount(inv,weights=np.where(good,diff,0),minlength=len(unique))
            counts=np.bincount(inv,weights=good,minlength=len(unique))
            denom=W@counts;boot=(W@sums)/np.where(denom>0,denom,np.nan);finite=boot[np.isfinite(boot)]
            top_ci.append({'bank':bank,'left':method+'_FULL','right':'LR_FULL',
               'top1_delta':float(diff[good].mean()) if good.any() else None,
               'lo95':float(np.quantile(finite,.025)) if len(finite) else None,
               'hi95':float(np.quantile(finite,.975)) if len(finite) else None,
               'n_common_answers':int(good.sum()),'inference':'internal exploratory source-group interval'})
    write('CROSSOVER_FACTORS.csv',factor_rows);write('TOP1_PAIRED_CONTRASTS.csv',top_ci)
    (OUT/'ACCEPTANCE.json').write_text(json.dumps({'status':'PASS','n_outer_partitions':partition_count,
        'max_saved_affine_test_score_replay_difference':maximum,
        'checks':['population and bank input SHA256 match every checkpoint','source-group outer and inner membership','disjoint outer train/test groups',
          'all saved fitted-model affine test scores','training-selector LR half mask',
          'no concatenated-global OOF ranking used'],
        'explicit_method_failures':failures,
        'not_certified':['historical LR split identity','causal feature importance','fresh external generalization']},indent=2))
    print('PASS partitions',partition_count,'affine replay max',maximum,'method failures',len(failures),flush=True)

if __name__=='__main__':main()
