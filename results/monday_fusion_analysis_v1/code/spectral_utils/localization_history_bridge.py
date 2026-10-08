"""Read-only historical score adapters and optional-gate grouped intervals."""
from copy import deepcopy
import numpy as np
from .fusion_benchmark_bootstrap import group_strata,sample_multiplicity,auc_counts,auc_from_counts,pb_counts,pb_from_counts


def corrected_rows(rows,label_index,group_index):
    out=[]
    for row in rows:
        rec=deepcopy(row);key=(row['cell'],row['row_id']);target=label_index[key]
        assert len(target['flags'])==row['steps'] if row['cell'].startswith('prm') else isinstance(target['label'],int)
        if row['cell'].startswith('prm'):
            assert row['target'] in (target['previous_flags'],target['flags'])
            rec['target']=list(target['flags'])
        else:assert row['target']==target['label']
        rec['group_id']=group_index[key];rec['bridge_previous_group_id']=row['group_id'];out.append(rec)
    assert len({r['uid'] for r in out})==len(out)
    return out


def intervals(rows,left,right,include_fixed=False,draws=1000):
    """Same source-group draws; no synthetic fixed-gate fields are created."""
    pp=[r for r in rows if r['cell'].startswith('prm') and r['valid'][left] and r['valid'][right]]
    bb=[r for r in rows if r['cell'].startswith('pb')]
    ps,bs=group_strata(pp),group_strata(bb);pc=[auc_counts(pp,a) for a in (left,right)]
    bc=[pb_counts(bb,a) for a in (left,right)]
    fc=[pb_counts(bb,a,True) for a in (left,right)] if include_fixed else None
    values={k:[] for k in ('prm_common_valid','prm_within_answer_common_valid','pb_all_population')}
    if include_fixed:values['pb_common_iu_gate_all_population']=[]
    rng=np.random.default_rng(2026090706)
    for _ in range(draws):
        if pp:
            weight=sample_multiplicity(ps,len(pp),rng);a,b=[auc_from_counts(x,weight) for x in pc]
            for i,k in enumerate(('prm_common_valid','prm_within_answer_common_valid')):
                if a[i] is not None and b[i] is not None:values[k].append(a[i]-b[i])
        weight=sample_multiplicity(bs,len(bb),rng)
        for counts,k in [(bc,'pb_all_population')]+([(fc,'pb_common_iu_gate_all_population')] if include_fixed else []):
            a,b=[pb_from_counts(x,weight) for x in counts]
            if a is not None and b is not None:values[k].append(a-b)
    out={'unit':'source group, stratified by cell','seed':2026090706,'draws':draws,
        'implementation':'cached_pair_counts_optional_fixed_gate','fixed_gate_included':include_fixed}
    for k,v in values.items():
        out[k+'_ci95']=np.quantile(v,[.025,.975]).tolist() if v else None
        out[k+'_valid_draws']=len(v)
    return out
