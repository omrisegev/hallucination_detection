import numpy as np, json
U='/mnt/user-data/uploads/hd_collection_data/'
META='/home/user/hallucination_detection/results/advisor_completion_2026-10-11/inputs/prmbench_meta.json'
def pb():
    F=np.load(U+'c4u_r1_inputs/crop_ares_all/full_sa_qwen_prm_processbench/bank_v3/FEATURES.npz')
    G={g['uid']:g for g in json.load(open(U+'crop_ares_v1/gold/processbench.json'))}
    names=list(F['names']); V=F['values']; off=F['offsets']; uids=F['uids']
    assert F['nonempty'].all()
    err=np.concatenate([~np.array(G[u]['correct']) for u in uids]); inc=np.concatenate([np.array(G[u]['include']) for u in uids])
    cat=np.array([G[u]['category'] for u in uids])
    assert len(err)==len(V)
    return dict(names=names,V=V,off=off,err=err,keep=inc,ids=uids,cat=cat)
def prmb():
    F=np.load(U+'prm_standalone_banks/bank_qwen_prm_7b/FEATURES.npz')
    names=list(F['names']); off=F['offsets']; sel=np.flatnonzero(F['cells']=='prmbench_qwen3_8b')
    ids=F['row_ids'][sel]; meta={r['idx']:r for r in json.load(open(META))['rows']}
    rows=np.concatenate([np.arange(off[j],off[j+1]) for j in sel]); V=F['values'][rows]
    lens=np.diff(off)[sel]; o=np.r_[0,np.cumsum(lens)]
    err=np.zeros(len(rows),bool)
    for k,i in enumerate(ids):
        assert lens[k]==meta[i]['n_steps']; err[o[k]:o[k+1]]=np.isin(np.arange(1,lens[k]+1),meta[i]['error_steps'])
    cls=np.array([meta[i]['classification'] for i in ids])
    keep=np.repeat(cls!='correct',lens)
    return dict(names=names,V=V,off=o,err=err,keep=keep,ids=ids,cat=cls)
