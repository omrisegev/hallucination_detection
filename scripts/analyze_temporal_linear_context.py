"""Independent PB audit and explicitly secondary linear-context contrasts."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import run_temporal_research_baseline as base
from scripts.analyze_temporal_research_baseline import independent_pb


def run(source):
    records,joined=base.load_contract(source);out=ROOT/'results/temporal_linear_context_v1'
    data=base.read_json(out/'METRICS.json');metrics=data['metrics'];cells=np.array([r['cell'] for r in records])
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz',allow_pickle=False) as f:opened=f['gate_percentile']>=.33
    with np.load(out/'SCORES_FROZEN.npz',allow_pickle=False) as f:scores={k[7:]:f[k] for k in f.files}
    audit={}
    for name,values in scores.items():
        peaks=np.array([int(np.argmax(values[joined['offsets'][i]:joined['offsets'][i+1]])) for i in range(len(records))])
        prediction=np.where(opened,peaks,-1);table,macro=independent_pb(joined['target'],cells,prediction,np.ones(len(records),bool))
        np.testing.assert_allclose(macro,metrics[name]['pb_all8'],rtol=0,atol=1e-14)
        audit[name]=dict(macro=macro,cells=table)
    base.common.atomic_json(out/'INDEPENDENT_PB_AUDIT.json',dict(status='PASS',methods=len(audit),results=audit))
    pairs=[]
    for bank in ('original4','innovation5'):
        signed=bank+'__real__signed_residual_0.25'
        pairs.extend([(signed,bank+'__base'),(signed,bank+'__shuffled__signed_residual_0.25'),
                      (bank+'__real__squared_residual_0.25',bank+'__shuffled__squared_residual_0.25')])
    names={n for pair in pairs for n in pair}
    # Only paired PB and within metrics are used here. PRMScore's correctly nested
    # thresholds remain the immutable values in METRICS.json, not this recalculation.
    _,per=base.evaluator.evaluate_arrays(records,joined,{n:scores[n] for n in names},fold_auc=True,pb_gate_open=opened)
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set())
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    base.common.atomic_json(out/'SECONDARY_CONTRASTS.json',dict(contrasts=contrasts,development_only=True,
        analysis_timing='Contrasts added after inspecting registered diagnostic arms; exploratory 95% intervals. Primary squared-error results unchanged.'))
    names=list(metrics);frontier=[n for n in names if not any(metrics[m]['pb_all8']>=metrics[n]['pb_all8'] and metrics[m]['prm_within']>=metrics[n]['prm_within']
        and (metrics[m]['pb_all8']>metrics[n]['pb_all8'] or metrics[m]['prm_within']>metrics[n]['prm_within']) for m in names)]
    base.common.atomic_json(out/'PARETO.json',dict(frontier=frontier,development_only=True))
    lines=['# Full linear chronological-context reference','','13,769 development answers; source-excluded fitting and nested PRM calibration.',
           'Independent scalar PB audit: PASS. Primary squared-error and secondary signed-innovation readouts are distinct.','',
           '| Method | PB % | within | PRMScore |','|---|---:|---:|---:|']
    for n,v in metrics.items():lines.append(f"| {n} | {100*v['pb_all8']:.4f} | {v['prm_within']:.6f} | {v['prmscore_q08']:.6f} |")
    lines.extend(['','Secondary intervals were added after inspecting diagnostic arms; do not promote them to preregistered primary results.',
                  'Shuffled/zero controls perturb inference history in the same model; refitted null-model controls remain distinct work.'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    status=base.read_json(out/'RUN_STATE.json');status.update(status='COMPLETE_REVIEWED',independent_pb_audit='PASS')
    base.common.atomic_json(out/'RUN_STATE.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source_root)
