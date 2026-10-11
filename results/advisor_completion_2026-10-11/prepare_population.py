"""Exact identity joins and the canonical answer-standardized bank adapter."""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
OUT=Path(__file__).resolve().parent; I=OUT/'inputs'
BANKS=json.loads((OUT/'sources/research/results/monday_nested_banks_v2/BANKS.json').read_text())
DIG=['digit_alternative','digit_spread','digit_alternative_innovation']
def answer_std(x,off):
    out=np.zeros_like(x,dtype=float)
    for a,b in zip(off[:-1],off[1:]):
        v=x[a:b];sd=v.std(0);out[a:b]=np.divide(v-v.mean(0),sd,out=np.zeros_like(v),where=sd>1e-12)
    return out
def main():
    z=np.load(I/'bank_N16.npz');idx=z['idx'];off=z['offsets'];F=np.load(I/'features_thinkfix_mask1.npz')
    meta={r['idx']:r for r in json.loads((I/'prmbench_meta.json').read_text())['rows']}
    group={r['row_id']:r['canonical_group_id'] for r in json.loads((I/'CANONICAL_GROUPS.json').read_text())['rows']}
    folds=json.loads((I/'FOLDS_V2.json').read_text())['prmbench']['outer']
    fa={u:i for i,u in enumerate(F['row_ids'])};ans=np.array([fa[u] for u in idx]);fo=F['offsets']
    rows=np.concatenate([np.arange(fo[a],fo[a+1]) for a in ans]);assert np.array_equal(np.diff(off),np.diff(fo)[ans])
    assert np.all(F['cells'][ans]=='prmbench_qwen3_8b') and len(set(idx))==len(meta)==6969
    y=np.zeros(off[-1],bool)
    for i,u in enumerate(idx):y[off[i]:off[i+1]]=np.isin(np.arange(1,off[i+1]-off[i]+1),meta[u]['error_steps'])
    cls=np.array([meta[u]['classification'] for u in idx]);groups=np.array([group[u] for u in idx]);fold=np.array([folds[g] for g in groups])
    extra={}
    sp=I/'step_spans_thinkfix_mask1.npz'
    if sp.exists():
        s=np.load(sp);spans=s['step_spans'];assert len(spans)==len(F['values']);extra['token_lengths']=np.diff(spans,axis=1).ravel()[rows]
        extra['answer_token_lengths']=np.add.reduceat(extra['token_lengths'],off[:-1])
    np.savez_compressed(I/'population.npz',idx=idx,offsets=off,y=y,classification=cls,group=groups,fold=fold,feature_rows=rows,feature_answers=ans,source_uid=F['uids'][ans],**extra)
    nm=list(map(str,F['names']));X=F['values'];Vall=answer_std(X,fo);da=F['digit_active']
    vd=np.zeros((len(X),3))
    for a,b in zip(fo[:-1],fo[1:]):
        for j,c in enumerate(DIG):
            act=da[a:b,j]; v=X[a:b,nm.index(c)][act]
            if len(v) and v.std()>1e-12:vd[a:b,j][act]=(v-v.mean())/v.std()
    checks=[]
    for bk,cols in BANKS.items():
        assert not any(c.startswith('lf__') for c in cols),'Unexpected LF orientation: extend adapter explicitly'
        raw=X[rows][:,[nm.index(c) for c in cols]]
        V=np.column_stack([vd[rows,DIG.index(c)] if c in DIG else Vall[rows,nm.index(c)] for c in cols])
        np.savez_compressed(I/f'bank_V_{bk}.npz',V=V,raw=raw,columns=np.array(cols),rows=rows,offsets=off)
        export=I/f'bank_{bk}.npz'
        if export.exists():
            b=np.load(export);assert np.array_equal(b['idx'],idx) and np.array_equal(b['offsets'],off)
            score=answer_std(V.mean(1)[:,None],off).ravel();err=float(np.max(np.abs(score-b['S__ALL_EQ'])))
            assert err<1e-9,(bk,err);checks.append({'bank':bk,'ALL_EQ_max_abs':err})
        print('prepared',bk,V.shape,flush=True)
    (OUT/'POPULATION_GATE.json').write_text(json.dumps({'status':'PASS','n_answers':len(idx),'n_steps':len(y),'n_groups':len(set(groups)),'fold_counts':np.bincount(fold).tolist(),'n_controls':int((cls=='correct').sum()),'canonical_annotation_membership':True,'bank_parity':checks,'lengths_available':bool(extra)},indent=2)+'\n')
if __name__=='__main__':main()
