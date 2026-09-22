"""Verify batched zero-update readout against the frozen original scores."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.matched_rbm_coefficient_update import Top10
from scripts.run_rbm_data_diagnostics import base


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args();source=a.source_root
    out=ROOT/'results/rbm_supervision_matched_v1';n=0;changed=0;diff=0.
    j=np.load(source/'results/localization_full_benchmark_v3/evaluation/JOINED.npz')
    z=np.load(source/'.worktrees/rbm-position-fusion-v1/results/rbm_position_fusion_v1_overlap_fix/SCORES.npz');ref=z['steps__rbm12__logit_old']
    for path in sorted(out.glob('cache_*.npz')):
        with np.load(path) as c:
            replay,_=Top10(c['x'],c['spans']).evaluate(c['base'])
            for local,i in enumerate(c['ids']):
                a,b=c['step_offsets'][local:local+2];old=ref[j['offsets'][i]:j['offsets'][i+1]];new=replay[a:b]
                np.testing.assert_allclose(new,old,atol=1e-11,rtol=1e-12)
                diff=max(diff,float(np.max(np.abs(new-old))));changed+=int(np.argmax(new)!=np.argmax(old));n+=1
    assert n==13769 and changed==0,(n,changed)
    base.atomic_json(out/'BATCHED_ZERO_REVIEW.json',dict(status='PASS',answers=n,changed_peaks=changed,max_score_error=diff,
        note='Stable-sort versus partition summation can differ at floating-point roundoff; all original peaks must agree.'))
    print('PASS zero update:',n,'answers;',changed,'changed peaks; max error',diff)


if __name__=='__main__':main()
