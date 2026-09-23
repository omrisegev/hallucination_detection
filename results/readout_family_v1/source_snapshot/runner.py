"""Outer source-fold fits and genuine inner-fold PRM threshold predictions.

Extended for the readout-family experiment (Step 429): readouts, rosters, fusion arms and a
fold-specific first-crossing column come from the config.  With the frozen v2 config every
path below reduces to the original behaviour: seven readouts, the rosters top5 / selected /
shuffle (+ top5 on erroneous answers for ProcessBench), nine arms, no page column.
"""
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from .core import READOUTS, CHANNELS, encode, training_matrix, fit_spectral, location
from .em import fit_em
from .readout import earliest_mode
from .data import digest,dump,within_auc
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from spectral_utils.step_readouts_v1 import answer_max_by_offsets,page_threshold,page_first_crossing,consensus_readout_choice,tied_argmax

NONFINITE={'onset80','page_cross'}   # suffix-masked profiles; PRMBench ranking cannot use them
PAGE='page_cross'
LEGACY_SELECT=7                      # 'selected' = the frozen label-selected roster over the seven readouts

def code_freeze(d):
    names=['core.py','em.py','data.py','runner.py','readout.py']
    paths=[Path(__file__).with_name(n) for n in names]
    paths += [Path(__file__).resolve().parents[3]/'spectral_utils/fusion_utils.py']
    if d.extended:paths += [Path(__file__).resolve().parents[3]/'spectral_utils/step_readouts_v1.py']
    state={p.name:digest(p) for p in paths};state['inputs']=digest(d.out/'INPUT_FREEZE.json')
    f=d.out/'RUN_FREEZE.json'
    if f.exists() and json.loads(f.read_text())!=state:raise ValueError('fit code changed; cannot resume mixed versions')
    if not f.exists():
        dump(f,state)
        for p in paths:
            dst=d.out/'source_snapshot'/p.name;dst.parent.mkdir(exist_ok=True);dst.write_bytes(p.read_bytes())

def finite_readouts(d):
    return np.array([r not in NONFINITE for r in d.readouts])

def merits(d,p):
    path=d.out/'channel_merits.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    K=len(d.readouts);finite=finite_readouts(d);result=np.full((d.n,11,K),np.nan)
    for i,(a,b) in enumerate(zip(d.off[:-1],d.off[1:])):
        if d.pb[i] and d.target[i]>=0:result[i]=(p[a:b].argmax(0)==d.target[i])
        elif d.prm[i]:
            for j in range(11):
                for r in np.flatnonzero(finite):result[i,j,r]=within_auc(d.labels[a:b],p[a:b,j,r])
        if i%3000==0:print(f'channel merits {i}/{d.n}',flush=True)
    np.save(path,result);return result

def roster_channels(d,roster):
    drop=set(d.c.get('channel_drop',{}).get(roster,[]))
    return np.array([j for j,ch in enumerate(CHANNELS) if ch not in drop])

def page_column(d,p,train,channels):
    """First-crossing column of the per-step Page maximum; h = training-fold quantile of the
    per-answer maximum over ALL training answers (labels unused)."""
    w=np.ascontiguousarray(p[:,channels,d.readouts.index('page_wmax')])
    per_answer=answer_max_by_offsets(w,d.off)
    h=page_threshold(per_answer[train],d.c['page_h_quantile'])
    col=np.empty_like(w);crossed=np.zeros((d.n,len(channels)),bool)
    for i,(a,b) in enumerate(zip(d.off[:-1],d.off[1:])):col[a:b],crossed[i]=page_first_crossing(w[a:b],h)
    return col,h,crossed

def choose_readouts(d,train,merit,task,roster,p=None,shuffle=None,channels=None,page=None):
    """Per-channel readout index; index len(d.readouts) means the fold-specific page column."""
    K=len(d.readouts);C=11 if channels is None else len(channels);info={}
    if roster in ('top5','shuffle'):return np.zeros(C,int),info
    if roster in d.readouts:return np.full(C,d.readouts.index(roster)),info
    if roster==PAGE:return np.full(C,K),info
    if roster in ('selected','selected_all'):
        use=np.arange(LEGACY_SELECT) if roster=='selected' else np.arange(K)
        if task!='pb':use=use[finite_readouts(d)[use]]
        sub=merit[:,channels][:,:,use] if channels is not None else merit[:,:,use]
        if task=='pb':
            values=np.mean([np.nanmean(sub[train[d.cells[train]==c]],axis=0) for c in sorted(set(d.cells[train]))],axis=0)
        else:values=np.nanmean(sub[train],axis=0)
        if not np.isfinite(values).all():raise ValueError('unidentified readout selection')
        return use[np.argmax(values,axis=1)],info # listed order wins exact ties
    if roster.startswith(('consensus','shuffle_consensus')):   # consensus, consensus10, shuffle_consensus
        base=shuffle if roster.startswith('shuffle') else p
        ch=np.arange(11) if channels is None else channels
        answers=[]
        for i in train:
            a,b=d.off[i:i+2];block=np.asarray(base[a:b][:,ch,:],float)
            if task=='pb' and page is not None:block=np.concatenate([block,page[a:b][:,:,None]],axis=2)
            answers.append(block)
        mask=np.ones(answers[0].shape[2],bool)
        if task!='pb':mask[:K]=finite_readouts(d)
        choice,agreement=consensus_readout_choice(answers,base=0,sweeps=d.c.get('consensus_sweeps',2),
            cells=d.cells[train] if task=='pb' else None,candidate_mask=mask)
        info={'agreement':agreement.tolist(),'candidates':[*d.readouts]+([PAGE] if task=='pb' and page is not None else [])}
        return choice,info
    raise ValueError('unknown roster '+roster)

def answer_profiles(d,p,shuffle,readouts,roster,channels=None,page=None):
    if not d.extended:
        matrix=shuffle if roster=='shuffle' else p[:,np.arange(11),readouts]
        return [matrix[a:b] for a,b in zip(d.off[:-1],d.off[1:])]
    base=shuffle if roster.startswith('shuffle') else p
    ch=np.arange(11) if channels is None else channels;K=len(d.readouts)
    matrix=np.empty((int(d.off[-1]),len(ch)))
    for k,(c,r) in enumerate(zip(ch,readouts)):
        matrix[:,k]=page[:,k] if r==K else base[:,c,r]
    return [matrix[a:b] for a,b in zip(d.off[:-1],d.off[1:])]

def fit_models(d,ps,train,task,which):
    models={};matrices={}
    need=[(enc,kind) for enc,kind in d.fusion_arms if (kind in ['ds','hem'])==(which=='em')]
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
    started=time.perf_counter()
    channels=None;page=None;page_h=None;crossed=None
    if d.extended:
        channels=roster_channels(d,roster)
        if task=='pb' and 'page_wmax' in d.readouts:
            base=shuffle if roster.startswith('shuffle') else p
            page,page_h,crossed=page_column(d,base,train,channels)
    readouts,extra=choose_readouts(d,train,merit,task,roster,p,shuffle,channels,page)
    ps=answer_profiles(d,p,shuffle,readouts,roster,channels,page)
    models,matrices=fit_models(d,ps,train,task,stage)
    result=predictions(d,ps,test,task,models,matrices,inner_fold is not None)
    np.savez_compressed(dest.with_suffix('.npz'),**result)
    with open(dest.with_suffix('.pkl'),'wb') as f:pickle.dump(models,f,protocol=5)
    names=[*d.readouts,PAGE]
    info={'task':task_name,'fold':fold,'inner_fold':inner_fold,'roster':roster,'population':population,'stage':stage,
          'readouts':[names[r] for r in readouts],'label_selected_readout':roster in ('selected','selected_all'),'train_answers':len(train),'test_answers':len(test),
          'train_source_groups':sorted(set(d.groups[train])),'test_source_groups':sorted(set(d.groups[test])),
          'seconds':time.perf_counter()-started,'models':{enc+'__'+kind:m for (enc,kind),m in models.items()}}
    if d.extended:
        info['channels']=[CHANNELS[c] for c in channels]
        info['tied_argmax_rate_test']=[float(np.mean([tied_argmax(ps[i][:,k:k+1])[0] for i in test if len(ps[i])>1])) for k in range(len(channels))]
        if page_h is not None:
            uses=np.flatnonzero(readouts==len(d.readouts))
            info['page_h']=page_h.tolist();info['page_h_quantile']=d.c['page_h_quantile'];info['page_k']=d.c.get('page_k')
            info['page_crossed_rate_train']=crossed[train].mean(0).tolist();info['page_crossed_rate_test']=crossed[test].mean(0).tolist()
            info['page_column_used_by']=[CHANNELS[channels[k]] for k in uses]
        info.update(extra)
    dump(dest.with_suffix('.json'),info)
    print(stem+f' done in {info["seconds"]:.1f}s',flush=True)

def specs_for(d,task):
    specs=list(d.rosters)
    if task!='prm':specs += list(d.pb_extra_rosters)
    else:specs=[(r,pop) for r,pop in specs if r not in NONFINITE]
    return specs

def run(d,stage,task_filter='all',fold_filter=None):
    if not (d.out/'PROFILES_COMPLETE.json').exists():raise ValueError('run prepare first')
    if d.extended and not (d.out/'PROFILES_EXT_COMPLETE.json').exists():raise ValueError('run prepare first (extended profiles)')
    code_freeze(d)
    p=np.load(d.out/d.profile_file,mmap_mode='r');shuffle=np.load(d.out/d.shuffle_file,mmap_mode='r')
    merit=merits(d,p)
    tasks=['prm'] if stage=='inner' else ['pb_q4','pb_q8','prm']
    for task in tasks:
        if task_filter not in ['all',task]:continue
        for fold in range(5):
            if fold_filter is not None and fold_filter!=fold:continue
            for roster,pop in specs_for(d,task):
                if stage=='inner':
                    for inner in range(5):
                        if inner==fold:continue
                        for substage in ['spectral','em']:one_job(d,p,shuffle,merit,task,fold,roster,pop,substage,inner)
                else:one_job(d,p,shuffle,merit,task,fold,roster,pop,stage)

def expected_outer_jobs(d):
    return 2*5*sum(len(specs_for(d,task)) for task in ['pb_q4','pb_q8','prm'])
