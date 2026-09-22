"""Outer source-fold fits and genuine inner-fold PRM threshold predictions."""
import json
import pickle
import time
from pathlib import Path
import numpy as np
from .core import ARMS, READOUTS, encode, training_matrix, fit_spectral, location
from .em import fit_em
from .readout import earliest_mode
from .data import digest,dump,within_auc

def code_freeze(d):
    paths=[Path(__file__).with_name(n) for n in ['core.py','em.py','data.py','runner.py','readout.py']]
    paths += [Path(__file__).resolve().parents[3]/'spectral_utils/fusion_utils.py']
    state={p.name:digest(p) for p in paths};state['inputs']=digest(d.out/'INPUT_FREEZE.json')
    f=d.out/'RUN_FREEZE.json'
    if f.exists() and json.loads(f.read_text())!=state:raise ValueError('fit code changed; cannot resume mixed versions')
    if not f.exists():
        dump(f,state)
        for p in paths:
            dst=d.out/'source_snapshot'/p.name;dst.parent.mkdir(exist_ok=True);dst.write_bytes(p.read_bytes())

def merits(d,p):
    path=d.out/'channel_merits.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    result=np.full((d.n,11,7),np.nan)
    for i,(a,b) in enumerate(zip(d.off[:-1],d.off[1:])):
        if d.pb[i] and d.target[i]>=0:result[i]=(p[a:b].argmax(0)==d.target[i])
        elif d.prm[i]:
            for j in range(11):
                for r in range(6):result[i,j,r]=within_auc(d.labels[a:b],p[a:b,j,r])
        if i%3000==0:print(f'channel merits {i}/{d.n}',flush=True)
    np.save(path,result);return result

def choose_readouts(d,train,merit,task,roster):
    if roster!='selected':return np.zeros(11,int)
    if task=='pb':
        values=np.mean([np.nanmean(merit[train[d.cells[train]==c]],axis=0) for c in sorted(set(d.cells[train]))],axis=0)
    else:values=np.nanmean(merit[train,:,:6],axis=0)
    if not np.isfinite(values).all():raise ValueError('unidentified readout selection')
    return np.argmax(values,axis=1) # listed order wins exact ties

def answer_profiles(d,p,shuffle,readouts,roster):
    matrix=shuffle if roster=='shuffle' else p[:,np.arange(11),readouts]
    return [matrix[a:b] for a,b in zip(d.off[:-1],d.off[1:])]

def fit_models(d,ps,train,task,which):
    models={};matrices={}
    need=[(enc,kind) for enc,kind in ARMS if (kind in ['ds','hem'])==(which=='em')]
    for enc in set(enc for enc,_ in need):matrices[enc]=training_matrix(ps,train,d.cells,enc,task)
    if which=='em':
        x,w=matrices['hard'];initial=fit_spectral(x,w,'spectral');hierarchy=fit_spectral(x,w,'binary_lsml')
        for kind in ['ds','hem']:
            models[('hard',kind)]=fit_em(x,w,kind,initial,hierarchy.groups,seed=d.c['seed'],max_iter=d.c['em_max_iter'],tol=d.c['em_relative_tolerance'])
    else:
        for enc,kind in need:
            x,w=matrices[enc];models[(enc,kind)]=fit_spectral(x,w,kind)
    return models,matrices

def predictions(d,ps,test,task,models,matrices,inner=False):
    result={'indices':np.asarray(test),'step_offsets':np.r_[0,np.cumsum([len(ps[i]) for i in test])]}
    qgrid=np.linspace(*d.c['inner_threshold_quantiles'])
    for (enc,kind),model in models.items():
        name=enc+'__'+kind;values=[];fallback=[];modes=[];medians=[]
        for i in test:
            if task=='pb':
                score,fail=location(model,ps[i],enc);modes.append(earliest_mode(score));medians.append(int(np.searchsorted(np.cumsum(score),.5)))
            else:
                x=encode(ps[i],enc,task);score=model.predict(x);fail=model.status!='ok' or not np.isfinite(score).all()
                if fail:score=x.mean(1)
            values.append(score);fallback.append(fail)
        values=np.concatenate(values);result[name+'__scores']=values;result[name+'__fallback']=np.array(fallback)
        if task=='pb':result[name+'__mode']=np.array(modes);result[name+'__median']=np.array(medians)
        else:
            x,_=matrices[enc];train_scores=model.predict(x) if model.status=='ok' else x.mean(1)
            thresholds=np.quantile(train_scores,qgrid);result[name+'__thresholds']=thresholds
            result[name+'__threshold_q80']=np.array(np.quantile(train_scores,.8))
            if inner:result[name+'__grid_valid']=(values[None,:]<thresholds[:,None])
    return result

def one_job(d,p,shuffle,merit,task_name,fold,roster,population,stage,inner_fold=None):
    task='prm' if task_name=='prm' else 'pb'
    selected=d.prm if task=='prm' else (d.pb&np.char.endswith(d.cells,task_name[-2:]))
    train=np.flatnonzero(selected&(d.fold!=fold))
    test=np.flatnonzero(selected&(d.fold==fold))
    if inner_fold is not None:
        test=train[d.fold[train]==inner_fold];train=train[d.fold[train]!=inner_fold]
    if population=='errors':train=train[d.target[train]>=0]
    assert not set(d.groups[train])&set(d.groups[test])
    assert not (set(d.groups[train])&set(d.groups[selected&(d.fold==fold)]))
    suffix='' if inner_fold is None else f'__inner{inner_fold}'
    stem=f'{task_name}__fold{fold}__{roster}__{population}__{stage}{suffix}'
    dest=d.out/'jobs'/stem;dest.parent.mkdir(exist_ok=True)
    if dest.with_suffix('.json').exists():return
    started=time.perf_counter();readouts=choose_readouts(d,train,merit,task,roster)
    ps=answer_profiles(d,p,shuffle,readouts,roster)
    models,matrices=fit_models(d,ps,train,task,stage)
    result=predictions(d,ps,test,task,models,matrices,inner_fold is not None)
    np.savez_compressed(dest.with_suffix('.npz'),**result)
    with open(dest.with_suffix('.pkl'),'wb') as f:pickle.dump(models,f,protocol=5)
    info={'task':task_name,'fold':fold,'inner_fold':inner_fold,'roster':roster,'population':population,'stage':stage,
          'readouts':[READOUTS[r] for r in readouts],'label_selected_readout':roster=='selected','train_answers':len(train),'test_answers':len(test),
          'train_source_groups':sorted(set(d.groups[train])),'test_source_groups':sorted(set(d.groups[test])),
          'seconds':time.perf_counter()-started,'models':{enc+'__'+kind:m for (enc,kind),m in models.items()}}
    dump(dest.with_suffix('.json'),info)
    print(stem+f' done in {info["seconds"]:.1f}s',flush=True)

def run(d,stage,task_filter='all',fold_filter=None):
    if not (d.out/'PROFILES_COMPLETE.json').exists():raise ValueError('run prepare first')
    code_freeze(d)
    p=np.load(d.out/'profiles.npy',mmap_mode='r');shuffle=np.load(d.out/'shuffled_top5.npy',mmap_mode='r')
    merit=merits(d,p)
    tasks=['prm'] if stage=='inner' else ['pb_q4','pb_q8','prm']
    for task in tasks:
        if task_filter not in ['all',task]:continue
        for fold in range(5):
            if fold_filter is not None and fold_filter!=fold:continue
            specs=[('top5','all'),('selected','all'),('shuffle','all')]
            if task!='prm':specs.append(('top5','errors'))
            for roster,pop in specs:
                if stage=='inner':
                    for inner in range(5):
                        if inner==fold:continue
                        for substage in ['spectral','em']:one_job(d,p,shuffle,merit,task,fold,roster,pop,substage,inner)
                else:one_job(d,p,shuffle,merit,task,fold,roster,pop,stage)
