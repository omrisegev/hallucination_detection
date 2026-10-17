"""Task-linked contrasts within the approved fixed diagnostic strata.

Both features use step means. This resolves the feature/fusion aggregation
difference in the descriptive table without training a regime-dependent model.
"""
from pathlib import Path
import json
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_rbm_data_diagnostics import mean_intervals
from scripts.run_rbm_data_diagnostics import base,stat
OUT=ROOT/'results/rbm_data_diagnostics_v1'


def main():
    contrasts=[];data=[]
    with np.load(OUT/'BANK6_DIAGNOSTICS.npz') as z:
        rel=z['rel'];groups=z['group_id']
    # Same six coordinates in both banks; duplicated evidence is not two replications.
    for lo,hi,regime in [(1,2,'high_minus_low_entropy'),(5,6,'late_minus_early')]:
        for j,reference in [(0,'entropy15'),(2,'moment3_15')]:
            a=rel[:,lo,3]-rel[:,lo,j];b=rel[:,hi,3]-rel[:,hi,j]
            good=np.isfinite(a)&np.isfinite(b)
            for name,v in [('low_or_early',a),('high_or_late',b),('interaction',b-a)]:
                value=np.where(good,v,np.nan);data.append(value)
                contrasts.append(dict(regime=regime,reference=reference,contrast=name,
                    selected_feature='selected_surprisal',**stat(value)))
    with threadpool_limits(limits=1):ci=mean_intervals(np.column_stack(data),groups)
    for item,interval in zip(contrasts,ci):item.update(ci95=interval.tolist())
    base.atomic_json(OUT/'RELIABILITY_REGIMES.json',dict(contrasts=contrasts,
        scope='Full-data diagnostic on fixed regimes, same-answer matched eligibility, identical step-mean aggregation. '
              'No adaptive weights trained or chosen. Conditional descriptive 95% group bootstrap, 10000 draws. '
              'Shared feature bank coordinates are the same evidence for RBM6 and RBM12.'))
    print(base.dumps(contrasts))


if __name__=='__main__':main()
