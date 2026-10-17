"""Top-step agreement diagnostic using an order-validated corrected LR cache.

Observed SHA pins this download; no historical expected cache SHA was found.
No LR reconstruction, feature refit, new head, or example-based selection occurs.
"""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from decision_analysis import OUT, INPUT, write_csv

def main():
    path=OUT/'sources/LR_N16_thinkfix_mask1.pkl'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest=='e73260d33db3a289766bcce44f681d0428bbdb5088d90eeb41e408ad3a7ac083'
    cache=pickle.loads(path.read_bytes()); z=np.load(INPUT/'bank_N16.npz',allow_pickle=False)
    ids=z['idx'].astype(str); off=z['offsets']; n=np.diff(off)
    meta={r['idx']:r for r in json.loads((INPUT/'prmbench_meta.json').read_text())['rows']}
    groups={r['row_id']:r['canonical_group_id'] for r in json.loads((OUT/'sources/CANONICAL_GROUPS.json').read_text())['rows']}
    _,group=np.unique([groups[u] for u in ids],return_inverse=True)
    y=np.zeros(int(off[-1]),bool)
    for i,u in enumerate(ids):y[off[i]:off[i+1]]=np.isin(np.arange(1,n[i]+1),meta[u]['error_steps'])
    eligible=np.add.reduceat(y.astype(int),off[:-1])>0
    assert int(eligible.sum())==6035 and cache['n_answers']==len(ids) and cache['n_steps']==len(y)
    positions={}
    for m in ['ALL_EQ','DS_FILTER_EQ','MEAN_MOM_ga','LSML_CONT','IU_PCR']:
        pos=np.array([np.argmax(z['S__'+m][a:b]) for a,b in zip(off[:-1],off[1:])])
        positions[m]=pos
        if m in cache['methods']:
            assert np.array_equal(pos,cache['methods'][m]['argmax'])
            assert np.array_equal(y[off[:-1]+pos][eligible],np.asarray(cache['methods'][m]['top1err'])[eligible])
    lrpos=np.asarray(cache['lr']['argmax'],int)
    assert lrpos.shape==n.shape and np.all((lrpos>=0)&(lrpos<n))
    lrhit=y[off[:-1]+lrpos]
    assert np.array_equal(lrhit[eligible],np.asarray(cache['lr']['top1err'])[eligible])
    weights=np.random.default_rng(20261011).multinomial(707,np.full(707,1/707),size=5000)
    denom=weights@np.bincount(group[eligible],minlength=707)
    results=[]; answers=[]
    for m in ['LSML_CONT','IU_PCR']:
        pos=positions[m]; hit=y[off[:-1]+pos]
        four={'both_hit':eligible&hit&lrhit,'ours_only_hit':eligible&hit&~lrhit,
              'lr_only_hit':eligible&~hit&lrhit,'neither_hit':eligible&~hit&~lrhit}
        counts={k:int(v.sum()) for k,v in four.items()}
        assert sum(counts.values())==eligible.sum()
        delta=np.bincount(group[eligible],weights=(hit.astype(int)-lrhit.astype(int))[eligible],minlength=707)
        boot=(weights@delta)/denom
        lo,hi=np.quantile(boot,[.025,.975])
        results.append(dict(method=m,n_error_answers=int(eligible.sum()),**counts,
            same_top_step=int((eligible&(pos==lrpos)).sum()),ours_top1=float(hit[eligible].mean()),
            lr_top1=float(lrhit[eligible].mean()),difference=float((hit.astype(int)-lrhit.astype(int))[eligible].mean()),
            ci_low=float(lo),ci_high=float(hi)))
        for i,u in enumerate(ids):
            if not eligible[i]:continue
            outcome=next(k for k,v in four.items() if v[i])
            answers.append(dict(method=m,idx=u,source_group=groups[u],classification=meta[u]['classification'],
                ours_step_1based=int(pos[i]+1),lr_step_1based=int(lrpos[i]+1),outcome=outcome))
    report={'observed_cache_sha256':digest,'historical_cache_hash_verified':False,
        'order_gate':'All argmax positions and eligible top1 labels match frozen NPZ for ALL_EQ,DS_FILTER_EQ,MEAN_MOM_ga.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'n_canonical_groups':707,'bootstrap':'5000 whole-source-group resamples seed20261011; pointwise percentile95%, development data',
        'limits':['No full LR step scores reconstructed.','No label-driven choice of examples or new combination fitted.',
                  'Paired CI conditions on this cached nested OOF fit; not a fresh fitted-system generalization estimate.'],
        'summary':results}
    (OUT/'LR_DISAGREEMENT.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(OUT/'lr_top1_disagreement.csv',answers)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
