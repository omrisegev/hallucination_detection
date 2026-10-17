"""Exhaustive frozen-score failure attribution; oracle rows are diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.historical_fusion_evaluation import pb_metrics
from scripts.run_lsml_gate_locator_research_v1 import dump

OUT=ROOT/'results/broad50_gate_locator_audit_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(ROOT/'results/fusion_independence_atlas_v1/dependence/EVALUATION.npz') as z:
        data={k:z[k] for k in ('target','offsets','cells','groups','folds')}
    with np.load(ROOT/'results/digitfree_broad50_v1/SCORES.npz') as z:scores={k:z[k] for k in z.files}
    gate=scores.pop('gate');target=data['target'];cells=data['cells'].astype(str)
    pb=np.char.startswith(cells,'pb_');error=pb&(target>=0);clean=pb&(target==-1)
    records=json.loads((ROOT.parents[1]/'results/localization_full_benchmark_v3/evaluation/JOINED.json').read_text())['records']
    output={};ledger=[];peaks={}
    valid=np.ones(len(target),bool)
    def metric(pred):return pb_metrics(target,pred,valid,cells)['macros']['all']
    for arm,score in scores.items():
        peak=np.asarray([np.argmax(score[a:b]) for a,b in zip(data['offsets'][:-1],data['offsets'][1:])]);peaks[arm]=peak
        rawhit=peak==target;pred=np.where(gate,peak,-1)
        masks={'clean_correct':clean&~gate,'clean_false_alarm':clean&gate,
            'error_hit':error&gate&rawhit,'gate_only_miss':error&~gate&rawhit,
            'locator_only_miss':error&gate&~rawhit,'both_miss':error&~gate&~rawhit}
        assert sum(int(m.sum()) for m in masks.values())==int(pb.sum())
        ideal_gate=np.where(target>=0,peak,-1)
        ideal_locator=np.where(gate,np.maximum(target,0),-1)
        # Conditional gate oracle also suppresses open clean false alarms.
        row=dict(counts={k:int(m.sum()) for k,m in masks.items()},pb=metric(pred),
            raw_error_exact=float(rawhit[error].mean()),pb_with_label_oracle_gate=metric(ideal_gate),
            pb_with_label_oracle_locator=metric(ideal_locator),per_cell={})
        for c in sorted(set(cells[pb])):
            cm=cells==c;row['per_cell'][c]={k:int((m&cm).sum()) for k,m in masks.items()}
        output[arm]=row
        for i in np.flatnonzero(pb):
            ledger.append(dict(uid=records[i]['uid'],cell=cells[i],group=data['groups'][i],fold=int(data['folds'][i]),
                method=arm,target=int(target[i]),peak=int(peak[i]),gate_open=bool(gate[i]),
                category=next(k for k,m in masks.items() if m[i])))
    reference=peaks['innovation5'];comparisons={}
    for arm,peak in peaks.items():
        gain=error&(peak==target)&(reference!=target);loss=error&(peak!=target)&(reference==target)
        comparisons[arm]=dict(raw_gained=int(gain.sum()),raw_lost=int(loss.sum()),
            final_gained=int((gain&gate).sum()),final_lost=int((loss&gate).sum()),
            shared_raw_miss=int((error&(peak!=target)&(reference!=target)).sum()))
    report=dict(status='COMPLETE',scope='full frozen development diagnostic; no fitting or selection',
        answers=int(pb.sum()),erroneous=int(error.sum()),clean=int(clean.sum()),methods=output,
        versus_innovation5=comparisons,oracle_warning='Label-using bottleneck diagnostics, not deployable performance.',
        attribution='The gate is identical across methods. Any PB difference between these methods is caused by locator decisions on gate-open answers. PRMB within-AUC is ungated.')
    dump(OUT/'RESULTS.json',report)
    with (OUT/'ANSWER_FAILURES.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    lines=['# Gate / locator attribution of the frozen broad50 run','',report['attribution'],'',
        '| Method | Error hit | Gate only | Locator only | Both | Clean false alarm | Raw error exact % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in output.items():
        c=r['counts'];lines.append(f"| {arm} | {c['error_hit']} | {c['gate_only_miss']} | {c['locator_only_miss']} | {c['both_miss']} | {c['clean_false_alarm']} | {100*r['raw_error_exact']:.2f} |")
    lines+=['','Gate only: the locator peak is right, but gate is closed. Locator only: gate open, wrong peak. Both: gate closed and wrong peak.',
        '', 'The JSON contains label-oracle diagnostics and every cell. These ceilings are not achievable methods. PRMB ranking regressions cannot be explained by this PB gate.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
