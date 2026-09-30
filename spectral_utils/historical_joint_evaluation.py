"""Fixed mechanism contrasts for the historical Joint extension."""
import numpy as np

PAIRS = (
    ('internal_joint_liu010','internal_joint_modelinv_lam0'),
    ('internal_joint_liu050','internal_joint_modelinv_lam0'),
    ('internal_joint_liu010','permctl_graph_internal_joint_liu010'),
    ('internal_joint_diag010','internal_joint_modelinv_lam0'),
    ('internal_joint_diag050','internal_joint_modelinv_lam0'),
    ('internal_joint_gate050','internal_joint'),
    ('internal_joint_gate100','internal_joint'),
    ('internal_joint','internal_cont'),
)


def contrasts(records,arms,a,outer,metric_core,*,draws=1000):
    names=sorted({r['group_id'] for r in records});lookup={g:i for i,g in enumerate(names)}
    gi=np.array([lookup[r['group_id']] for r in records]);ng=len(names)
    weights=np.random.default_rng(20260907328).multinomial(ng,np.full(ng,1/ng),size=draws)
    cells=np.array([r['cell'] for r in records]);prm=np.char.startswith(cells,'prm')
    owner=np.repeat(np.arange(len(records)),np.diff(a['offsets']))
    all_results=[]
    for left,right in PAIRS:
        js=[arms.index(left),arms.index(right)]
        common=prm & a['valid'][:,js[0]] & a['valid'][:,js[1]]
        mixed=common & np.isfinite(a['within'][:,js[0]]) & np.isfinite(a['within'][:,js[1]])
        plans=[]
        for j in js:
            current=[]
            for fold in range(5):
                mask=(common & (outer==fold))[owner]
                current.append(metric_core.auc_plan(a['labels'][mask],a['scores'][mask,j],gi[owner][mask]) if mask.any() else None)
            plans.append(current)
        distribution=[];point=None
        for draw,w in enumerate([np.ones(ng)]+list(weights)):
            rw=w[gi];den=rw[mixed].sum();values=[]
            for j,current in zip(js,plans):
                foldauc=np.mean([metric_core.weighted_auc(plan,w) if plan is not None else np.nan for plan in current])
                within=np.dot(rw[mixed],a['within'][mixed,j])/den if den else np.nan
                pb=metric_core.pb_metrics(a['target'],a['predictions'][:,j],a['decision'][:,j],cells,rw)['macros']
                values.append(np.array([foldauc,within,pb['q4'],pb['q8'],pb['all']],float))
            difference=values[0]-values[1]
            if draw==0:point=difference
            else:distribution.append(difference)
        distribution=np.array(distribution)
        endpoints=('prm_fold_mean_auc','prm_within_auc','pb_q4','pb_q8','pb_all')
        all_results.append(dict(candidate=left,control=right,prm_common_answers=int(common.sum()),
            prm_common_mixed_answers=int(mixed.sum()),endpoints={name:dict(difference=float(point[j]),
            low=float(np.nanquantile(distribution[:,j],.025)),high=float(np.nanquantile(distribution[:,j],.975)),
            finite_draws=int(np.isfinite(distribution[:,j]).sum())) for j,name in enumerate(endpoints)}))
        print('Historical mechanism contrast',left,'vs',right,'complete',flush=True)
    return dict(draws=draws,groups=ng,scope='Fixed predictions/thresholds, common-valid PRMB; all PB denominators',
        calibration_refit_uncertainty=False,selection_uncertainty=False,contrasts=all_results)
