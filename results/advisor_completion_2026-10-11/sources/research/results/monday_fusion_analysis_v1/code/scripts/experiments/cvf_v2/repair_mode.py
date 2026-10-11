"""Readout-only migration: preserve fitted models and scores, repair roundoff ties.

This implements the already specified earliest tie rule, before final evaluation.
The old indices and code freeze are archived. No data-dependent tuning occurs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import config,Dataset,dump,digest
from .readout import earliest_mode
from .runner import code_freeze

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();d=Dataset(config(args.config))
    if (d.out/'READOUT_TIE_REPAIR.json').exists():
        code_freeze(d);print('Readout migration already complete; original audit preserved.');return
    original=json.loads((d.out/'RUN_FREEZE.json').read_text())
    for name in ['core.py','em.py','data.py']:
        assert digest(Path(__file__).with_name(name))==original[name],f'not a readout-only migration: {name}'
    old={};changed={}
    for path in sorted((d.out/'jobs').glob('pb_*.npz')):
        z=dict(np.load(path));count=0
        for key in list(z):
            if not key.endswith('__mode'):continue
            scores=z[key[:-6]+'__scores'];off=z['step_offsets'];prior=z[key].copy()
            corrected=np.array([earliest_mode(scores[a:b]) for a,b in zip(off[:-1],off[1:])])
            old[path.stem+'___'+key]=prior;count+=int(np.sum(prior!=corrected));z[key]=corrected
        np.savez_compressed(path,**z);changed[path.stem]=count
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf8'));meta['readout_revision']='earliest_mode_arithmetic_ties_v2';dump(path.with_suffix('.json'),meta)
    np.savez_compressed(d.out/'PRE_REPAIR_MODE_INDICES.npz',**old)
    dump(d.out/'PRE_REPAIR_RUN_FREEZE.json',original)
    # Retain original source snapshot in its own directory before re-freezing.
    archive=d.out/'source_snapshot_before_tie_repair';archive.mkdir(exist_ok=True)
    for path in (d.out/'source_snapshot').iterdir():(archive/path.name).write_bytes(path.read_bytes())
    current=dict(original);current['runner.py']=digest(Path(__file__).with_name('runner.py'));current['readout.py']=digest(Path(__file__).with_name('readout.py'))
    dump(d.out/'RUN_FREEZE.json',current)
    for name in ['runner.py','readout.py']:(d.out/'source_snapshot'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    code_freeze(d)
    dump(d.out/'READOUT_TIE_REPAIR.json',{'reason':'earliest ties prescribed before fitting; differencing equal rational masses creates ulp differences',
       'tolerance':'8 * float64 epsilon * max(1, max(abs(mass)))','changed_predictions_by_job':changed,
       'total_changed':sum(changed.values()),'models_and_scores_unchanged':True,'archived_old_predictions':'PRE_REPAIR_MODE_INDICES.npz'})
    print('Repaired arithmetic mode ties:',sum(changed.values()))

if __name__=='__main__':main()
