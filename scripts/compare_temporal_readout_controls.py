"""Development-only paired PB comparisons against historical step-selection rules."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import run_temporal_research_baseline as base
from spectral_utils.temporal_research_features import BASELINE


def run(source):
    records,joined=base.load_contract(source);out=ROOT/'results/temporal_feature_diagnostics_v1'
    root=ROOT/'results/temporal_research_baseline_v1'
    methods=['append_innovation__H0lim','readout__earlier_VE0_VE075_peak','readout__first_near_max_025',BASELINE]
    with np.load(root/'PREDICTIONS.npz',allow_pickle=False) as f:
        per={n:{k:f[k+'__'+n] for k in ['prediction','decision_valid','within']} for n in methods}
    pairs=[(methods[0],methods[1]),(methods[0],methods[2]),(methods[1],BASELINE),(methods[2],BASELINE)]
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set())
    metrics=base.read_json(root/'METRICS.json')['metrics']
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    base.common.atomic_json(out/'READOUT_CONTRASTS.json',dict(contrasts=contrasts,development_only=True,
        scope='PB step-selection rules have no corresponding PRMB score change; within comparisons are intentionally unavailable.',
        multiplicity='Exploratory 95% intervals, not confirmatory familywise claims.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source_root)
