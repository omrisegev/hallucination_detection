"""Artifact-level validation: saved models, complete coverage and split isolation."""
import argparse
import json
import pickle
import time
import numpy as np
from .data import config,Dataset,dump
from .core import ARMS,READOUTS,encode,location
from .readout import earliest_mode
from .runner import code_freeze,answer_profiles

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    d=Dataset(config(args.config));code_freeze(d);started=time.perf_counter()
    profiles=np.load(d.out/'profiles.npy',mmap_mode='r');shuffle=np.load(d.out/'shuffled_top5.npy',mmap_mode='r')
    files=list((d.out/'jobs').glob('*.json'));outer=inner=trajectories=predictions=0;nonconverged=[];bad_range=[]
    for path in files:
        j=json.loads(path.read_text(encoding='utf8'));z=dict(np.load(path.with_suffix('.npz')));test=z['indices'];off=z['step_offsets']
        assert set(j['train_source_groups']).isdisjoint(j['test_source_groups'])
        held=set(d.groups[d.fold==j['fold']]);assert set(j['train_source_groups']).isdisjoint(held)
        if j['inner_fold'] is None:outer+=1;assert np.all(d.fold[test]==j['fold'])
        else:inner+=1;assert np.all(d.fold[test]==j['inner_fold']) and np.all(d.fold[test]!=j['fold'])
        expected=[enc+'__'+kind for enc,kind in ARMS if (kind in ['ds','hem'])==(j['stage']=='em')]
        assert sorted(k[:-8] for k in z if k.endswith('__scores'))==sorted(expected)
        for arm in expected:
            s=z[arm+'__scores'];assert np.isfinite(s).all()
            if j['task']!='prm':
                for k,(a,b) in enumerate(zip(off[:-1],off[1:])):
                    assert (s[a:b]>=0).all() and abs(s[a:b].sum()-1)<1e-12
                    assert z[arm+'__mode'][k]==earliest_mode(s[a:b])
            info=j['models'][arm]
            if info['status']!='ok':bad_range.append(path.stem+'__'+arm)
            for start in info['diagnostics'].get('starts',[]):
                ll=np.asarray(start['log_likelihood']);assert np.all(np.diff(ll)>=-1e-9*np.maximum(1,abs(ll[:-1])))
                trajectories+=1
            if info['diagnostics'].get('converged') is False:nonconverged.append(path.stem+'__'+arm)
        with open(path.with_suffix('.pkl'),'rb') as f:models=pickle.load(f)
        ps=answer_profiles(d,profiles,shuffle,np.array([READOUTS.index(r) for r in j['readouts']]),j['roster'])
        task='prm' if j['task']=='prm' else 'pb'
        for pos in sorted(set([0,len(test)//2,len(test)-1])):
            i=test[pos];a,b=off[pos:pos+2]
            for (enc,kind),model in models.items():
                if task=='pb':s,failed=location(model,ps[i],enc)
                else:
                    x=encode(ps[i],enc,task);s=model.predict(x);failed=model.status!='ok' or not np.isfinite(s).all()
                    if failed:s=x.mean(1)
                np.testing.assert_allclose(s,z[enc+'__'+kind+'__scores'][a:b],rtol=1e-13,atol=1e-13)
                assert failed==z[enc+'__'+kind+'__fallback'][pos];predictions+=1
    assert (outer,inner)==(110,120),(outer,inner)
    raw=json.loads((d.out/'RAW_CACHE_AUDIT.json').read_text(encoding='utf8'));assert raw['answers']==13769
    llama=json.loads((d.out/'LLAMA_APPENDIX.json').read_text(encoding='utf8'));assert llama['all_archived_metrics_match']
    dump(d.out/'VALIDATION.json',{'outer_jobs':outer,'inner_jobs':inner,'likelihood_trajectories_checked':trajectories,
      'saved_model_holdout_predictions_reapplied':predictions,'native_invalid_range_models':bad_range,
      'selected_em_nonconverged':nonconverged,'all_job_splits_and_scores_checked':True,'raw_answer_joins':13769,
      'seconds':time.perf_counter()-started})
    print('Artifact validation passed:',outer,inner,predictions,'reapplied predictions',flush=True)

if __name__=='__main__':main()
