"""Reproduce the post-ranking label-free weight-age diagnostic on all answers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from spectral_utils.context_training import FeatureBundle
from spectral_utils.context_weighted_levels import selected_tokens
from scripts.run_context_weighted_levels import OUT,SOURCE,write

def run():
    b=FeatureBundle(ROOT/'results/temporal_context_data_v1','innovation5')
    with np.load(SOURCE/'LANDMARKS.npz') as a:answer=a['answer'];pos=a['local']
    unique,first,counts=np.unique(answer,return_index=True,return_counts=True)
    ages=[];per=[];warm=0;total=0;gaps=[]
    for i,begin,count in zip(unique,first,counts):
        m=b.metadata[i];x=np.asarray(b.features[m['offset']:m['offset']+m['tokens']]);pp=pos[begin:begin+count]
        spans=np.asarray(b.spans[m['step_start']:m['step_stop']])-m['offset']
        selected=np.concatenate([v.ravel() for v in selected_tokens(x,spans)])
        idx=np.searchsorted(pp,selected,side='right')-1;valid=idx>=0
        age=selected[valid]-pp[idx[valid]]
        ages.extend(age);warm+=int((~valid).sum());total+=len(selected)
        per.append(float(np.mean(age>=16)) if len(age) else 0.)
        gaps.append(int(np.diff(np.r_[0,pp,m['tokens']]).max()))
    result=dict(status='DESCRIPTIVE_AFTER_FULL_RANKING_NO_LABELS',answers=len(unique),selected_feature_token_contributions=total,
        warmup_fraction=warm/total,age_ge16_fraction=float(np.mean(np.array(ages)>=16)),
        answer_mean_age_ge16_fraction=float(np.mean(per)),age_q50_q90_q99=np.quantile(ages,[.5,.9,.99]).tolist(),
        max_gap_q50_q90_q99=np.quantile(gaps,[.5,.9,.99]).tolist(),max_gap=max(gaps),
        interpretation='Temporal hold can apply context older than one16-token window; exact per-token context fitting remains untested.')
    write(OUT/'SCHEDULE_DIAGNOSTICS.json',result)
    print(result)

if __name__=='__main__':run()
