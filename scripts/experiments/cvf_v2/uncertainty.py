"""Common source-question draws, primary endpoint intervals and planned contrasts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from .data import dump
from .core import ARMS

LEGACY_ROSTERS=[('top5','all'),('selected','all'),('shuffle','all'),('top5','errors')]

def planned(names,rosters=None,arms=None,extra=None):
    """Planned contrasts.  With the defaults this is the frozen v2 list; the extended run adds
    the pmf-versus-cumulative, roster-versus-top5, consensus-versus-selected and
    shuffle-consensus-versus-consensus contrasts."""
    rosters=LEGACY_ROSTERS if rosters is None else list(rosters);arms=ARMS if arms is None else list(arms)
    pairs=[]
    def add(a,b,why):
        if a in names and b in names and a!=b and (a,b,why) not in pairs:pairs.append((a,b,why))
    for roster,pop in rosters:
        if roster.startswith('shuffle'):continue
        pre=f'{roster}__{pop}__'
        for kind in ['equal','spectral','continuous_lsml']:add(pre+'soft__'+kind,pre+'hard__'+kind,'soft_minus_binary')
        add(pre+'hard__continuous_lsml',pre+'hard__binary_lsml','continuous_core_bridge')
        for enc,kind in arms:
            name=pre+enc+'__'+kind
            if kind!='equal':add(name,pre+enc+'__equal','learned_minus_equal')
            for base in ['ct7','mindgap']:add(name,base,'new_minus_'+base)
            if roster!='top5' and pop=='all':add(name,'top5__all__'+enc+'__'+kind,f'{roster}_minus_top5')
            if pop=='errors':add(name,'top5__all__'+enc+'__'+kind,'error_only_minus_all_training')
            if roster=='consensus':add(name,'selected__all__'+enc+'__'+kind,'consensus_minus_selected')
            if roster=='selected_all':add(name,'selected__all__'+enc+'__'+kind,'selected_all_minus_selected')
            if roster=='consensus10':add(name,'consensus__all__'+enc+'__'+kind,'ten_channel_minus_eleven')
        for kind in ['equal','spectral','continuous_lsml']:add(pre+'pmf__'+kind,pre+'soft__'+kind,'pmf_minus_cumulative')
        for kind in ['ds','hem']:add(pre+'hard__'+kind,pre+'hard__spectral','em_minus_spectral_initialization')
        add(pre+'hard__hem',pre+'hard__ds','hierarchical_minus_ds')
    for enc,kind in arms:
        add('shuffle__all__'+enc+'__'+kind,'top5__all__'+enc+'__'+kind,'token_shuffle_minus_original')
        add('shuffle_consensus__all__'+enc+'__'+kind,'consensus__all__'+enc+'__'+kind,'token_shuffle_minus_original')
    # Experiment-specific contrasts declared in the config (Step 432); absent by default.
    for a,b,why in (extra or []):add(a,b,why)
    return pairs

def bootstrap(d,methods,within):
    h=hashlib.sha256(Path(__file__).read_bytes())
    for x in [d.groups,d.cells,d.target,d.gate]:h.update(x.tobytes())
    h.update(str((d.c['seed'],d.c['bootstrap_draws'])).encode())
    for name,m in methods.items():
        h.update(name.encode())
        for key in ['pred','valid']:h.update(m[key].tobytes())
        if name in within:h.update(within[name].tobytes())
    fingerprint=h.hexdigest();cache=d.out/'UNCERTAINTY.json'
    if cache.exists():
        previous=json.loads(cache.read_text(encoding='utf8'))
        if previous.get('input_fingerprint')==fingerprint:return previous
    names=list(methods);M=len(names);_,group=np.unique(d.groups,return_inverse=True);G=group.max()+1
    incidence=sparse.csr_matrix((np.ones(d.n),(group,np.arange(d.n))),shape=(G,d.n))
    valid=np.column_stack([methods[n]['valid'] for n in names]);pred=np.column_stack([methods[n]['pred'] for n in names])
    hit=(pred==d.target[:,None])&valid
    blocks=[];cells=sorted(set(d.cells[d.pb]))
    for cell in cells:
        error=(d.cells==cell)&(d.target>=0);clean=(d.cells==cell)&(d.target<0)
        for values in [valid&error[:,None],hit&error[:,None],hit&(error&d.gate)[:,None],valid&clean[:,None],valid&(clean&~d.gate)[:,None]]:
            blocks.append(incidence@sparse.csr_matrix(values.astype(float)))
    wa=np.column_stack([within.get(n,np.full(d.n,np.nan)) for n in names]);eligible=np.isfinite(wa)
    blocks += [incidence@sparse.csr_matrix(np.nan_to_num(wa)),incidence@sparse.csr_matrix(eligible.astype(float))]
    packed=sparse.hstack(blocks,format='csr')
    def metrics(weights):
        raw=(packed.T@weights.T).T.reshape(len(weights),42,M)
        with np.errstate(divide='ignore',invalid='ignore'):
            sla=[];f1=[]
            for c in range(8):
                ed,eh,eg,cd,ch=(raw[:,5*c+k] for k in range(5));sla.append(eh/ed);a=ch/cd;b=eg/ed
                f1.append(np.divide(2*a*b,a+b,out=np.zeros_like(a),where=(a+b)>0))
            return np.mean(sla,axis=0),np.mean(f1,axis=0),raw[:,40]/raw[:,41]
    point=metrics(np.ones((1,G)));B=d.c['bootstrap_draws'];arrays=[np.empty((B,M)) for _ in range(3)]
    rng=np.random.default_rng(d.c['seed']);prob=np.full(G,1/G)
    for start in range(0,B,128):
        count=min(128,B-start);weights=rng.multinomial(G,prob,size=count).astype(float)
        out=metrics(weights)
        for a,v in zip(arrays,out):a[start:start+count]=v
        if start%2048==0:print(f'paired source bootstrap {start}/{B}',flush=True)
    intervals={};contrasts=[];endpoints=['pb_sla','pb_common_gate_f1','prm_within_auc']
    rosters=getattr(d,'rosters',None);arms=getattr(d,'fusion_arms',None)
    if rosters is not None:rosters=list(rosters)+list(getattr(d,'pb_extra_rosters',[]))
    for endpoint,pts,draws in zip(endpoints,point,arrays):
        table={}
        for j,name in enumerate(names):
            if not np.isfinite(pts[0,j]):continue
            table[name]={'point':float(pts[0,j]),'ci95':np.nanpercentile(draws[:,j],[2.5,97.5]).tolist()}
        intervals[endpoint]=table
        for a,b,why in planned(table,rosters,arms,d.c.get('planned_contrasts_extra')):
            ia=names.index(a);ib=names.index(b);delta=float(pts[0,ia]-pts[0,ib]);boot=draws[:,ia]-draws[:,ib]
            boot=boot[np.isfinite(boot)]
            # Centered nonparametric bootstrap test, finite Monte Carlo correction.
            p=(1+np.sum(abs(boot-delta)>=abs(delta)))/(len(boot)+1)
            contrasts.append({'endpoint':endpoint,'a':a,'b':b,'contrast':why,'delta':delta,'ci95':np.percentile(boot,[2.5,97.5]).tolist(),
              'p_bootstrap':float(p),'draws':len(boot)})
    order=np.argsort([r['p_bootstrap'] for r in contrasts],kind='stable');running=0.;n=len(order)
    for rank,i in enumerate(order):
        running=max(running,min(1.,(n-rank)*contrasts[i]['p_bootstrap']));contrasts[i]['p_holm']=running
    result={'input_fingerprint':fingerprint,'draws':B,'seed':d.c['seed'],'unit':'source_question_shared_across_scorers','source_groups':int(G),
      'intervals':intervals,'planned_comparisons':len(contrasts),'holm_family':'all listed primary-endpoint contrasts jointly',
      'minimum_raw_p':1/(B+1),'contrasts':contrasts,'secondary_metrics':'descriptive; fold AUROC/AP not pooled across fitted models'}
    dump(d.out/'UNCERTAINTY.json',result)
    np.savez_compressed(d.out/'BOOTSTRAP_PRIMARY_DRAWS.npz',names=np.array(names),**dict(zip(endpoints,arrays)))
    return result
