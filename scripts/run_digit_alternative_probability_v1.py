"""Full source-only, CPU digit-alternative experiment; see frozen protocol."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
import gc
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.digit_alternative_probability import (
    second_digit_probability, disagreement, prefix_innovation, step_top_mean, digit_innovation_step_max)
from spectral_utils.label_sanity import check_labels

METHODS = ('digit_alternative_top2','old_digit_top2','old_digit_innovation_top1',
           'entropy_top2','entropy_top10','censor_upper_diagnostic')
SEED = 20260927


def write(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def rows(path):
    with path.open('rb') as f: p=pickle.load(f)
    return list(p.values()) if isinstance(p,dict) else p


def auc(y,s):
    y=np.asarray(y,bool); n=int(y.sum()); m=len(y)-n
    return float((rankdata(s)[y].sum()-n*(n+1)/2)/(n*m))


def independent_auc(y,s):
    pos=s[y==1];neg=s[y==0]
    return float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))


def extract(source,out):
    start=time.perf_counter()
    bench=source/'results/localization_full_benchmark_v3/evaluation'
    meta=json.loads((bench/'JOINED.json').read_text())
    records=meta['records']; frozen=np.load(bench/'JOINED.npz')
    offsets=frozen['offsets']; labels=frozen['labels']; target=frozen['target']
    assert len(records)==13769 and int(offsets[-1])==145597
    assert sha(bench/'JOINED.npz')==meta['arrays_sha256']
    scores=np.full((int(offsets[-1]),len(METHODS)),np.nan)
    tokfiles=sorted((source/'results/automatic_group_free_phase_a6_s0a_v1/inputs').glob('qwen*/tokenizer.json'))
    assert len(tokfiles)==2
    tokenizers={}
    for p in tokfiles:
        v=json.loads(p.read_text(encoding='utf-8'))['model']['vocab']
        ids=[v[str(i)] for i in range(10)]
        assert ids==list(range(15,25))
        tokenizers[p.parent.name]={'digit_ids':ids,'sha256':sha(p)}
    digits=np.arange(15,25);digitset=set(digits.tolist())
    lpfile=source/'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl'
    labelmap={str(r['idx']):r for r in rows(lpfile)}
    specs=[(f'pb_{ds}_{model}',source/f'dataset_cache/repgrid/pb_qwen3_{size}/processbench_{ds}.pkl',ds)
           for model,size in [('q4','4b'),('q8','8b')]
           for ds in ('gsm8k','math','olympiadbench','omnimath')]
    specs.append(('prmbench_qwen3_8b',source/'dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl',None))
    inputs={str(bench/'JOINED.json'):sha(bench/'JOINED.json'),str(bench/'JOINED.npz'):sha(bench/'JOINED.npz'),str(lpfile):sha(lpfile)}
    per_answer=[];schema=[];total=Counter();max_scalar_error=0.
    span_audit=json.loads((out/'SPAN_AUDIT.json').read_text(encoding='utf-8'))
    assert span_audit['status']=='PASS'
    known_overlap={r['row_id']:r['spans'] for r in span_audit['overlap_rows']}
    inputs[str(out/'SPAN_AUDIT.json')]=sha(out/'SPAN_AUDIT.json')
    for cell,path,ds in specs:
        print('AUDIT/EXTRACT',cell,flush=True)
        inputs[str(path)]=sha(path)
        payload=rows(path)
        mapping={(f"{ds}::{r['id']}" if ds else str(r['idx'])):r for r in payload}
        assert len(mapping)==len(payload)
        selected=[i for i,r in enumerate(records) if r['cell']==cell]
        assert len(selected)==len(payload)
        checks=Counter(); widths=set(); lens=[]
        for i in selected:
            rec=records[i];r=mapping[rec['row_id']]
            top=r.get('top_k_logprobs')
            if not isinstance(top,dict): top=r['top_k_logprobs_raw']
            ids=np.asarray(top['ids']); lp=np.asarray(top['logprobs'],dtype=float)
            gen=np.asarray(r['gen_token_ids']); ent=np.asarray(r['token_entropies'],float)
            spans=np.asarray(r['step_token_spans'],int); sl=slice(offsets[i],offsets[i+1])
            assert ids.shape==lp.shape and ids.shape[1]==50
            assert len(ids)==len(gen)==len(ent)==rec['tokens']
            assert spans.shape==(rec['steps'],2) and len(labels[sl])==len(spans)
            assert np.isfinite(ent).all() and np.all(ent>=-1e-8)
            assert np.all(spans[1:,0]>=spans[:-1,0])
            if np.any(spans[1:,0]<spans[:-1,1]):
                assert ds is None and rec['row_id'] in known_overlap
                np.testing.assert_array_equal(spans,known_overlap[rec['row_id']])
            if ds:
                assert int(r['label'])==int(target[i])
            else:
                lab=labelmap[rec['row_id']]
                assert lab['n_steps']==len(spans)
                expected=np.zeros(len(spans),dtype=int)
                for s in lab['error_steps']:
                    if 1<=s<=len(spans):expected[s-1]=1
                np.testing.assert_array_equal(labels[sl],expected)
            lo,hi,seen=second_digit_probability(ids,lp,digits)
            old=disagreement(gen,ids[:,0],digits)
            # Separate scalar extraction over every token; no vector helper reuse.
            scalar=np.fromiter((sorted((float(np.exp(v)) for token,v in zip(tt,ll) if int(token) in digitset),reverse=True)[1]
                               if sum(int(token) in digitset for token in tt)>=2 else 0.
                               for tt,ll in zip(ids,lp)),float,count=len(ids))
            err=float(np.max(np.abs(scalar-lo))); max_scalar_error=max(max_scalar_error,err)
            assert err<1e-14
            oldscalar=np.fromiter((float(int(g) in digitset and int(p) in digitset and g!=p)
                                  for g,p in zip(gen,ids[:,0])),float,count=len(gen))
            np.testing.assert_array_equal(old,oldscalar)
            streams=(lo,old,prefix_innovation(old),ent,ent,hi)
            for j,(stream,k) in enumerate(zip(streams,(2,2,1,2,10,2))):
                scores[sl,j]=step_top_mean(stream,spans,k)
            scores[sl,2], innovation_available = digit_innovation_step_max(old,gen,digits,spans)
            # Independent sorted readout, every step of primary and historical digit.
            for j,stream in [(0,lo),(1,old)]:
                ref=np.array([sum(sorted(stream[a:b],reverse=True)[:min(2,b-a)])/min(2,b-a) for a,b in spans])
                np.testing.assert_allclose(scores[sl,j],ref,rtol=0,atol=1e-14)
            peak=int(np.argmax(scores[sl,0])); other=scores[sl,5].copy();other[peak]=-np.inf
            certified=bool(scores[sl,0][peak]>np.max(other))
            d={'row':i,'uid':rec['uid'],'cell':cell,'tokens':len(ids),
               'censored_tokens':int((seen<2).sum()),'old_disagreements':int(old.sum()),
               'positive_alternative_tokens':int((lo>0).sum()),
               'old_innovation_missing_steps':int((~innovation_available).sum()),
               'max_token_upper_gap':float(np.max(hi-lo)),
               'max_step_upper_gap':float(np.max(scores[sl,5]-scores[sl,0])),
               'peak_certified_under_censoring':certified,
               'lower_upper_same_peak':bool(peak==int(np.argmax(scores[sl,5])))}
            per_answer.append(d)
            total.update({k:d[k] for k in ('tokens','censored_tokens','old_disagreements','positive_alternative_tokens')})
            total['answers']+=1;total['steps']+=len(spans);widths.add(ids.shape[1]);lens.append(len(ids));checks['validated_rows']+=1
        schema.append({'cell':cell,'bytes':path.stat().st_size,'sha256':inputs[str(path)],
                       'rows':len(payload),'checks':dict(checks),'saved_k':sorted(widths),
                       'min_tokens':min(lens),'max_tokens':max(lens),'status':'PASS'})
        del payload,mapping;gc.collect()
    assert np.isfinite(scores).all() and total['tokens']==6968779
    per_answer.sort(key=lambda d:d['row'])
    assert [d['uid'] for d in per_answer]==[r['uid'] for r in records]
    np.savez_compressed(out/'SCORES.npz',scores=scores,offsets=offsets,labels=labels,target=target)
    write(out/'RECORDS.json',records);write(out/'COVERAGE.json',per_answer)
    write(out/'EXTRACTION_AUDIT.json',{'status':'PASS','counts':dict(total),'source_inputs':inputs,
          'tokenizers':tokenizers,'schema':schema,'independent_scalar_tokens_checked':total['tokens'],
          'scalar_max_error':max_scalar_error,'seconds':time.perf_counter()-start,
          'scores_sha256':sha(out/'SCORES.npz'),'label_contract':'corrected v3; PRMB one-based errors verified for all rows',
          'alignment':'saved per-token/per-step layout verified; no new forward pass'})
    print('EXTRACTION PASS',dict(total),flush=True)


def evaluate(out,draws):
    start=time.perf_counter();f=np.load(out/'SCORES.npz');s=f['scores'];offsets=f['offsets'];labels=f['labels'];target=f['target']
    records=json.loads((out/'RECORDS.json').read_text(encoding='utf-8'));cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_');bad=pb&(target>=0)
    groups,gi=np.unique([r['group_id'] for r in records],return_inverse=True);n=len(records);m=len(METHODS)
    peaks=np.zeros((n,m),int);within=np.full((n,m),np.nan);ties=np.zeros((n,m),int);uniform=np.zeros((n,m));constant=np.zeros((n,m),bool)
    for i in range(n):
        x=s[offsets[i]:offsets[i+1]];y=labels[offsets[i]:offsets[i+1]]
        peaks[i]=np.argmax(x,axis=0);mx=x.max(axis=0);ties[i]=(x==mx).sum(axis=0);constant[i]=np.ptp(x,axis=0)==0
        if bad[i]:uniform[i]=(x[target[i]]==mx)/ties[i]
        if not pb[i] and (y==0).any() and (y==1).any():
            for j in range(m):
                within[i,j]=auc(y==1,x[:,j]);assert abs(within[i,j]-independent_auc(y,x[:,j]))<1e-12
    sanity={}
    for c in sorted(set(cells)):
        ix=cells==c
        ys=(target[ix]>=0) if c.startswith('pb_') else labels[np.repeat(ix,np.diff(offsets))]
        check=check_labels(ys);assert check.ok,check.summary()
        sanity[c]={'n':check.n,'n_pos':check.n_pos,'n_neg':check.n_neg,'flag':check.flag_string()}
    hits=(peaks==target[:,None]);mixed=np.isfinite(within[:,0]);pb_cells=sorted(set(cells[pb]));metric={}
    for j,name in enumerate(METHODS):
        per={}
        for c in sorted(set(cells)):
            mask=(cells==c);eligible=mask&bad if c.startswith('pb_') else mask&mixed
            per[c]={'n_total':int(mask.sum()),'n_scored':int(eligible.sum()),
                    'score':float(hits[eligible,j].mean() if c.startswith('pb_') else within[eligible,j].mean()),
                    'constant_answers':int(constant[mask,j].sum()),'tied_peak_answers':int((ties[mask,j]>1).sum()),'flag':sanity[c]['flag']}
        metric[name]={'pb_macro_exact':float(np.mean([per[c]['score'] for c in pb_cells])),
                      'pb_erroneous_n':int(bad.sum()),'prmb_within_auc':float(within[mixed,j].mean()),'prmb_mixed_n':int(mixed.sum()),
                      'pb_tie_uniform_macro':float(np.mean([uniform[bad&(cells==c),j].mean() for c in pb_cells])),
                      'cells':per}
    # Cluster sufficient statistics preserve every occurrence of a source question.
    denom=np.zeros((len(groups),9));numer=np.zeros((len(groups),9,m))
    for k,c in enumerate(pb_cells+['PRMB_MIXED']):
        ix=bad&(cells==c) if k<8 else mixed
        np.add.at(denom[:,k],gi[ix],1.)
        for j in range(m):np.add.at(numer[:,k,j],gi[ix],hits[ix,j] if k<8 else within[ix,j])
    rng=np.random.default_rng(SEED);boot=[]
    for startdraw in range(0,draws,100):
        count=min(100,draws-startdraw)
        w=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=count)
        den=w@denom;assert np.all(den>0)
        val=(w@numer.reshape(len(groups),-1)).reshape(count,9,m)/den[:,:,None]
        boot.append(np.stack((val[:,:8].mean(axis=1),val[:,8]),axis=1))
    boot=np.concatenate(boot);contrasts={}
    for control in ('old_digit_top2','entropy_top2','entropy_top10','old_digit_innovation_top1'):
        j=METHODS.index(control);delta=boot[:,:,0]-boot[:,:,j]
        contrasts[control]={}
        for k,key in enumerate(('pb_macro_exact','prmb_within_auc')):
            contrasts[control][key]={'delta':metric[METHODS[0]][key]-metric[control][key],
                'ci98_75':np.quantile(delta[:,k],[.00625,.99375]).tolist(),
                'primary':control in ('old_digit_top2','entropy_top2')}
    # Fixed randomization checks localization chance and AUC polarity/tie handling.
    rng=np.random.default_rng(SEED+1);null=[]
    for rep in range(20):
        nh=np.zeros(n);na=np.full(n,np.nan)
        for i in np.flatnonzero(bad|mixed):
            x=s[offsets[i]:offsets[i+1],0];z=rng.permutation(x)
            if bad[i]:nh[i]=float(z[target[i]]==z.max())/int((z==z.max()).sum())
            if mixed[i]:na[i]=auc(labels[offsets[i]:offsets[i+1]]==1,z)
        null.append([np.mean([nh[bad&(cells==c)].mean() for c in pb_cells]),np.mean(na[mixed])])
    coverage=json.loads((out/'COVERAGE.json').read_text(encoding='utf-8'))
    overlap_ids={r['row_id'] for r in json.loads((out/'SPAN_AUDIT.json').read_text(encoding='utf-8'))['overlap_rows']}
    no_overlap=mixed&np.array([r['row_id'] not in overlap_ids for r in records])
    result={'status':'COMPLETE','scope':'full source development; standalone localization, no gate/fusion',
            'n_checked':n,'n_total':13769,'steps':len(s),'methods':metric,'contrasts':contrasts,'label_sanity':sanity,
            'bootstrap':{'draws':draws,'seed':SEED,'unit':'source_question','simultaneous_primary_intervals':'98.75%, family of four'},
            'null':{'repeats':20,'pb_tie_uniform_macro_mean':float(np.mean(null,axis=0)[0]),'prmb_auc_mean':float(np.mean(null,axis=0)[1]),
                    'pb_random_step_macro':float(np.mean([np.mean(1/np.diff(offsets)[bad&(cells==c)]) for c in pb_cells]))},
            'censoring':{'certified_peak_answers':sum(d['peak_certified_under_censoring'] for d in coverage),
                         'same_lower_upper_peak_answers':sum(d['lower_upper_same_peak'] for d in coverage),
                         'max_step_gap':max(d['max_step_upper_gap'] for d in coverage)},
            'evaluation_seconds':time.perf_counter()-start,'scores_sha256':sha(out/'SCORES.npz'),
            'inherited_overlap_sensitivity':{'excluded_answers':len(overlap_ids),'mixed_n':int(no_overlap.sum()),
                'prmb_within_auc':{name:float(within[no_overlap,j].mean()) for j,name in enumerate(METHODS)}},
            'independent_pairwise_auc_checked':int(mixed.sum())*m}
    np.savez_compressed(out/'ANSWER_METRICS.npz',peaks=peaks,within=within,ties=ties,constant=constant,bootstrap=boot)
    write(out/'METRICS.json',result)
    with (out/'PER_CELL.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['method','cell','n_total','n_scored','score','constant_answers','tied_peak_answers','flag']);w.writeheader()
        for name,v in metric.items():
            for cell,row in v['cells'].items():w.writerow({'method':name,'cell':cell,**row})
    print(json.dumps({'methods':{k:{x:v[x] for x in ('pb_macro_exact','prmb_within_auc')} for k,v in metric.items()},'contrasts':contrasts,'null':result['null'],'censoring':result['censoring']},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--evaluate-only',action='store_true');ap.add_argument('--bootstrap',type=int,default=5000);args=ap.parse_args()
    out=ROOT/'results/digit_alternative_probability_v1';out.mkdir(parents=True,exist_ok=True)
    if not args.evaluate_only:
        if (out/'SCORES.npz').exists():raise RuntimeError('Refuse to overwrite frozen scores')
        write(out/'RUN.json',{'command':sys.argv,'methods':METHODS,'protocol_sha256':sha(ROOT/'docs/experiments/DIGIT_ALTERNATIVE_PROBABILITY_V1.md'),
                              'code_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'spectral_utils/digit_alternative_probability.py']},'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        extract(args.source_root,out)
    evaluate(out,args.bootstrap)


if __name__=='__main__':main()
