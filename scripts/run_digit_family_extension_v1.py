"""Two-feature numeric-family extension, full source population, no GPU."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.digit_feature_family import token_family,step_family,answer_standardize,NAMES
from spectral_utils.external_generalization.fusion import fit_weights,standardize,answer_z,FEATURE_NAMES
from spectral_utils.external_generalization.evaluation import confusion,metric,paired_bootstrap
from spectral_utils.prmbench import prmbench_evaluate
from spectral_utils.label_sanity import check_labels
from scripts.run_digit_alternative_probability_v1 import sha,rows,auc

OUT=ROOT/'results/digit_family_extension_v1'
OLD=ROOT/'results/digit_alternative_probability_v1'
CONTRASTS=('bank11_lsml','plus1_lsml','copies3_lsml','plus3_equal')


def write(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def final_z(x,off):
    z=np.empty_like(x,dtype=float)
    for a,b in zip(off[:-1],off[1:]):z[a:b]=answer_z(x[a:b])
    return z


def extract(source):
    t=time.perf_counter();rec=json.loads((OLD/'RECORDS.json').read_text());old=np.load(OLD/'SCORES.npz')
    off=old['offsets'];expected=old['scores'][:,0]
    prior=json.loads((OLD/'EXTRACTION_AUDIT.json').read_text());spana=json.loads((OLD/'SPAN_AUDIT.json').read_text())
    overlaps={r['row_id']:r['spans'] for r in spana['overlap_rows']}
    x=np.full((off[-1],3),np.nan);active=np.zeros(x.shape,bool);audits=[];total=0;max_error=0.
    for cell in sorted({r['cell'] for r in rec}):
        selected=[i for i,r in enumerate(rec) if r['cell']==cell]
        if cell.startswith('pb_'):
            _,ds,model=cell.split('_');size='4b' if model=='q4' else '8b'
            path=source/f'dataset_cache/repgrid/pb_qwen3_{size}/processbench_{ds}.pkl'
        else:
            ds=None;path=source/'dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl'
        print('extract',cell,flush=True);h=sha(path);assert h==prior['source_inputs'][str(path)]
        raw=rows(path);byid={(f"{ds}::{r['id']}" if ds else str(r['idx'])):r for r in raw};assert len(byid)==len(selected)
        for i in selected:
            r=byid[rec[i]['row_id']];top=r['top_k_logprobs'];ids=np.asarray(top['ids']);lp=np.asarray(top['logprobs'],float)
            spans=np.asarray(r['step_token_spans'],int);assert len(ids)==rec[i]['tokens'] and len(spans)==rec[i]['steps']
            if np.any(spans[1:,0]<spans[:-1,1]):np.testing.assert_array_equal(spans,overlaps[rec[i]['row_id']])
            v,ok,diag=token_family(ids,lp,range(15,25));sl=slice(off[i],off[i+1]);x[sl],active[sl]=step_family(v,ok,spans)
            np.testing.assert_allclose(x[sl,0],expected[sl],atol=1e-14,rtol=0)
            # Separate scalar entropy formula over ALL saved token positions.
            independent=[]
            for tt,ll in zip(ids,lp):
                probs=[math.exp(float(l)) for token,l in zip(tt,ll) if 15<=int(token)<25]
                mass=math.fsum(probs)
                independent.append(-math.fsum(p*math.log(p/mass) for p in probs)/math.log(10) if mass else 0.)
            err=float(np.max(np.abs(np.asarray(independent)-v[:,1])));max_error=max(max_error,err);assert err<1e-12
            total+=len(ids)
        audits.append({'cell':cell,'answers':len(selected),'source':str(path),'sha256':h})
        del raw,byid
    assert np.isfinite(x).all() and total==6968779
    np.savez_compressed(OUT/'FEATURES.npz',values=x,active=active,offsets=off)
    write(OUT/'EXTRACTION.json',{'status':'PASS','answers':len(rec),'tokens':total,'steps':int(off[-1]),'sources':audits,
        'feature1_replay':'all steps exact to 1e-14','scalar_spread_max_error':max_error,'seconds':time.perf_counter()-t,
        'sha256':sha(OUT/'FEATURES.npz'),'inactive_steps':(~active).sum(axis=0).tolist()})


def fit(source):
    t=time.perf_counter();rec=json.loads((OLD/'RECORDS.json').read_text());feat=np.load(OUT/'FEATURES.npz');off=feat['offsets']
    numeric=answer_standardize(feat['values'],off,feat['active'])
    path=source/'.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz'
    source_manifest=json.loads((source/'results/lsml_external_generalization_v1/evaluation/source/INPUTS.json').read_text())
    assert sha(path)==source_manifest['paths']['level']['sha256']
    raw=np.load(path);base=answer_standardize(raw['level'],off);assert base.shape==(145597,11)
    np.testing.assert_array_equal(raw['channels'],FEATURE_NAMES)
    foldpath=source/'results/localization_source_group_audit_v1/FOLDS_V2.json'
    fm=json.loads(foldpath.read_text())['outer'];fold=np.array([fm[r['group_id']] for r in rec]);sf=np.repeat(fold,np.diff(off))
    fits_path=source/'results/lsml_external_generalization_v1/evaluation/source/VALIDATION_FITS.json'
    oldfits=json.loads(fits_path.read_text());oldscore=np.load(source/'results/lsml_external_generalization_v1/evaluation/source/VALIDATION.npz')
    np.testing.assert_array_equal(oldscore['folds'],fold);np.testing.assert_array_equal(oldscore['offsets'],off)
    group_sets=[]
    for oldfit in oldfits:
        gr=np.array(oldfit['fit']['groups']);group_sets.append(sorted(tuple(np.flatnonzero(gr==g).tolist()) for g in np.unique(gr)))
    assert all(g==group_sets[0] for g in group_sets)
    families=group_sets[0];assert len(families)==6
    banks={'bank11':base,'plus1':np.column_stack((base,numeric[:,0])),
           'plus3':np.column_stack((base,numeric)), 'copies3':np.column_stack((base,np.tile(numeric[:,0,None],(1,3))))}
    fixed={name:numeric[:,j] for j,name in enumerate(NAMES)}
    fixed['numeric_family_mean']=numeric.mean(axis=1)
    fixed.update({f'general_family_{i}':base[:,list(g)].mean(axis=1) for i,g in enumerate(families)})
    names=[b+'_lsml' for b in banks]+[b+'_equal' for b in ('bank11','plus1','plus3')]+list(fixed)
    scores={n:np.full(off[-1],np.nan) for n in names};pred={n:np.full(off[-1],-1,np.int8) for n in names};fits=[];writes=np.zeros(off[-1],int)
    write(OUT/'FIT_INPUTS.json',{'level':{'path':str(path),'sha256':sha(path)},'folds_sha256':sha(foldpath),'source_fits_sha256':sha(fits_path),
          'feature_sha256':sha(OUT/'FEATURES.npz'),'families':[[FEATURE_NAMES[j] for j in g] for g in families],
          'names':names,'answer_normalization':'same answer only, offline','fit_scope':'3 source folds, 1 unlabeled calibration, 1 held-out fold'})
    for test in range(5):
        cal=(test+1)%5;train=(sf!=test)&(sf!=cal);hold=sf==test;calmask=sf==cal
        entry={'test':test,'calibration':cal,'fits':{},'thresholds':{}};current=dict(fixed)
        for name,bank in banks.items():
            f=next(f['fit'] for f in oldfits if f['test']==test) if name=='bank11' else fit_weights(bank[train])
            entry['fits'][name]=f;current[name+'_lsml']=bank@np.array(f['weights'])
            if name!='copies3':current[name+'_equal']=bank.mean(axis=1)
        for name,s in current.items():
            z=final_z(s,off);tau=float(np.quantile(z[calmask],.8));entry['thresholds'][name]=tau
            scores[name][hold]=z[hold];pred[name][hold]=(z[hold]<tau)
        np.testing.assert_allclose(scores['bank11_lsml'][hold],oldscore['frozen_lsml_score'][hold],atol=1e-6,rtol=0)
        np.testing.assert_array_equal(pred['bank11_lsml'][hold],oldscore['frozen_lsml_pred'][hold])
        np.testing.assert_allclose(scores['bank11_equal'][hold],oldscore['frozen_equal_score'][hold],atol=1e-6,rtol=0)
        np.testing.assert_array_equal(pred['bank11_equal'][hold],oldscore['frozen_equal_pred'][hold])
        writes[hold]+=1;fits.append(entry);write(OUT/'FITS.json',fits)
        print('fit fold',test,'numeric groups',entry['fits']['plus3']['groups'][11:],'numeric weights',entry['fits']['plus3']['weights'][11:],flush=True)
        if time.perf_counter()-t>1800:raise TimeoutError('Fit exceeds 30-minute cap')
    assert np.all(writes==1) and all(np.isfinite(v).all() for v in scores.values())
    np.savez_compressed(OUT/'PREDICTIONS.npz',offsets=off,folds=fold,**{n+'_score':v for n,v in scores.items()},**{n+'_pred':v for n,v in pred.items()})
    write(OUT/'SEAL.json',{'status':'PASS','prediction_sha256':sha(OUT/'PREDICTIONS.npz'),'fits_sha256':sha(OUT/'FITS.json'),
         'baseline_replay':'all answers, both scores and decisions','fit_seconds':time.perf_counter()-t,'names':names,'n':len(rec),'steps':int(off[-1])})


def dependence(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);joint=float(np.mean(a*b));ma=float(a.mean());mb=float(b.mean())
    denom=math.sqrt(ma*(1-ma)*mb*(1-mb))
    return {'n':len(a),'error_rate_a':ma,'error_rate_b':mb,'joint_error':joint,'independent_product':ma*mb,
            'excess_joint_error':joint-ma*mb,'phi':(joint-ma*mb)/denom if denom>0 else None}


def evaluate(source):
    t=time.perf_counter();seal=json.loads((OUT/'SEAL.json').read_text());assert sha(OUT/'PREDICTIONS.npz')==seal['prediction_sha256']
    rec=json.loads((OLD/'RECORDS.json').read_text());n=len(rec);f=np.load(OUT/'PREDICTIONS.npz');off=f['offsets'];names=seal['names']
    z={k:f[k] for k in f.files};old=np.load(OLD/'SCORES.npz');labels=old['labels'];target=old['target'];cells=np.array([r['cell'] for r in rec]);pb=np.char.startswith(cells,'pb_')
    meta={str(r['idx']):r for r in rows(source/'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl')}
    noncontrol=np.array([not pb[i] and meta[rec[i]['row_id']]['classification']!='correct' for i in range(n)])
    mixed=np.array([not pb[i] and (labels[a:b]==0).any() and (labels[a:b]==1).any() for i,(a,b) in enumerate(zip(off[:-1],off[1:]))])
    bad=pb&(target>=0);assert (noncontrol.sum(),mixed.sum(),bad.sum())==(6211,6030,4442)
    owner=np.repeat(np.arange(n),np.diff(off));validsteps=noncontrol[owner]
    sanity=check_labels(labels[validsteps]);assert sanity.ok,sanity.summary()
    metrics={};counts={};hits={};withins={};pbcell=sorted(set(cells[pb]))
    for name in names:
        s=z[name+'_score'];p=z[name+'_pred'];c=np.zeros((n,4),int);hit=np.zeros(n);wa=np.full(n,np.nan)
        for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
            if noncontrol[i]:c[i]=confusion(labels[a:b]==0,p[a:b])
            if mixed[i]:wa[i]=auc(labels[a:b]==1,s[a:b])
            if bad[i]:hit[i]=int(np.flatnonzero(s[a:b]>=s[a:b].max()-8*np.finfo(float).eps)[0])==target[i]
        official=prmbench_evaluate([{'idx':rec[i]['row_id'],'labels':p[off[i]:off[i+1]].astype(int).tolist()} for i in np.flatnonzero(~pb)],
                                  [meta[rec[i]['row_id']] for i in np.flatnonzero(~pb)])['total']
        ps=float(metric(c.sum(axis=0),'socratic'));assert abs(ps-(official['f1']+official['negative_f1'])/2)<1e-12
        metrics[name]={'prmscore':ps,'prmscore_n':int(noncontrol.sum()),'within_auc':float(wa[mixed].mean()),'within_n':int(mixed.sum()),
                       'pb_macro_exact':float(np.mean([hit[bad&(cells==cc)].mean() for cc in pbcell])), 'pb_n':int(bad.sum()),
                       'pb_cells':{cc:float(hit[bad&(cells==cc)].mean()) for cc in pbcell},'flag':sanity.flag_string()}
        counts[name]=c;hits[name]=hit;withins[name]=wa
    groups=np.array([r['group_id'] for r in rec]);contrasts={}
    for control in CONTRASTS:
        contrasts[control]=paired_bootstrap(counts['plus3_lsml'][~pb],counts[control][~pb],groups[~pb],'socratic',4,draws=5000,seed=20260927)
    family_names=[f'general_family_{i}' for i in range(6)];comparisons=family_names+list(NAMES)
    errors={name:(z[name+'_pred']!=(labels==0)) for name in names};dep={};numeric='numeric_family_mean'
    for name in comparisons:
        dep[name]={}
        for truth in (0,1):
            mask=validsteps&(labels==truth);dep[name]['true_'+('correct' if truth==0 else 'error')]=dependence(errors[numeric][mask],errors[name][mask])
        dep[name]['pb_miss_by_cell']={cc:dependence(1-hits[numeric][bad&(cells==cc)],1-hits[name][bad&(cells==cc)]) for cc in pbcell}
    # Explicit within-family redundancy and dependence, not just mean-versus-members.
    internal={}
    for i in range(3):
        for j in range(i+1,3):
            key=NAMES[i]+'__'+NAMES[j]
            internal[key]={str(truth):dependence(errors[NAMES[i]][validsteps&(labels==truth)],errors[NAMES[j]][validsteps&(labels==truth)]) for truth in (0,1)}
    correlations={name:float(np.corrcoef(z[numeric+'_score'][validsteps],z[name+'_score'][validsteps])[0,1]) for name in comparisons}
    x=np.load(OUT/'FEATURES.npz');std=answer_standardize(x['values'],off,x['active']);internal_corr=np.corrcoef(std.T)
    np.savez_compressed(OUT/'EVALUATION.npz',**{k+'_counts':v for k,v in counts.items()},**{k+'_within':v for k,v in withins.items()},**{k+'_hit':v for k,v in hits.items()})
    result={'status':'COMPLETE','n_checked':n,'n_total':13769,'steps':int(off[-1]),'methods':metrics,'primary_prmscore_contrasts':contrasts,
            'dependence':dep,'within_digit_errors':internal,'numeric_score_correlations':correlations,'within_digit_score_correlations':internal_corr.tolist(),
            'official_replays':len(names),'evaluation_seconds':time.perf_counter()-t,'prediction_sha256':seal['prediction_sha256'],
            'caveat':'diagnostic pairwise errors, not proof of latent conditional independence; development source data'}
    write(OUT/'METRICS.json',result)
    print(json.dumps({'methods':{k:v for k,v in metrics.items() if not k.startswith('general_family')},'contrasts':contrasts,'digit_correlations':internal_corr.tolist()},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extract','fit','evaluate']);ap.add_argument('--source-root',type=Path,required=True);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.stage=='extract':
        if (OUT/'FEATURES.npz').exists():raise RuntimeError('Refuse to overwrite frozen features')
        paths=[Path(__file__),ROOT/'spectral_utils/digit_feature_family.py',ROOT/'docs/experiments/DIGIT_FAMILY_EXTENSION_V1.md']
        write(OUT/'FREEZE.json',{'command':sys.argv,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                               'hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},'names':NAMES})
    if a.stage=='fit' and (OUT/'SEAL.json').exists():raise RuntimeError('Refuse to overwrite frozen predictions')
    {'extract':extract,'fit':fit,'evaluate':evaluate}[a.stage](a.source_root)
