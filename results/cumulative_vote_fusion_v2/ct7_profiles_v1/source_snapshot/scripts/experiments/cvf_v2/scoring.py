"""Common-gate PB scoring, fold-wise PRMB ranking and honest inner thresholds."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from .core import CHANNELS,standardize
from .data import within_auc,dump

spec=importlib.util.spec_from_file_location('cvf_prm_evaluator',Path(__file__).resolve().parents[3]/'spectral_utils/prmbench.py')
prm=importlib.util.module_from_spec(spec);spec.loader.exec_module(prm)

def collect(d):
    methods={};jobs=[]
    def create(name):
        if name not in methods:methods[name]={'scores':np.full(int(d.off[-1]),np.nan),'pred':np.full(d.n,-999,int),
          'median':np.full(d.n,-999,int),'fallback':np.zeros(d.n,bool),'valid':np.zeros(d.n,bool)}
        return methods[name]
    for path in sorted((d.out/'jobs').glob('*.json')):
        if '__inner' in path.stem:continue
        info=json.loads(path.read_text(encoding='utf8'));jobs.append(info)
        z=np.load(path.with_suffix('.npz'));idx=z['indices'];off=z['step_offsets']
        for key in z.files:
            if not key.endswith('__scores'):continue
            arm=key[:-8];name=f'{info["roster"]}__{info["population"]}__{arm}';m=create(name)
            if m['valid'][idx].any():raise ValueError('duplicate predictions '+name)
            saved_scores=z[key] # decompress once per arm, not once per answer
            for ii,i in enumerate(idx):m['scores'][d.off[i]:d.off[i+1]]=saved_scores[off[ii]:off[ii+1]]
            m['valid'][idx]=True;m['fallback'][idx]=z[arm+'__fallback']
            if info['task']!='prm':m['pred'][idx]=z[arm+'__mode'];m['median'][idx]=z[arm+'__median']
    if len(jobs)!=110:raise ValueError(f'need all 110 outer jobs, found {len(jobs)}')
    for name,m in methods.items():
        want=d.pb if '__errors__' in name else np.ones(d.n,bool)
        assert np.array_equal(m['valid'],want),name
    for name,s in d.references.items():
        m=create(name);m['scores']=s.copy();m['valid']=d.reference_valid[name].copy();m['pred']=d.peaks(s)
    profiles=np.load(d.out/'profiles.npy',mmap_mode='r')
    for j,channel in enumerate(CHANNELS):
        m=create('single__'+channel);s=profiles[:,j,0].copy()
        for i in np.flatnonzero(d.prm):a,b=d.off[i:i+2];s[a:b]=standardize(s[a:b])[0]
        m['scores']=s;m['pred']=d.peaks(s);m['valid'][:]=True
    lengths=np.load(d.out/'step_lengths.npy')
    m=create('control__longest_step');m['scores']=lengths.astype(float);m['pred']=d.peaks(lengths);m['valid'][:]=True
    for position in [0.,.5,1.]:
        m=create(f'control__relative_position_{position}')
        for a,b in zip(d.off[:-1],d.off[1:]):m['scores'][a:b]=-(np.linspace(0,1,b-a)-position)**2
        m['pred']=d.peaks(m['scores']);m['valid'][:]=True
    return methods,jobs

def pb_metrics(d,m):
    rows={}
    for cell in sorted(set(d.cells[d.pb])):
        full=d.cells==cell;take=full&m['valid'];err=take&(d.target>=0);clean=take&(d.target<0)
        delta=m['pred'][err]-d.target[err];ca=float((~d.gate[clean]).mean());ea=float(((delta==0)&d.gate[err]).mean())
        rows[cell]={'n':int(take.sum()),'total_n':int(full.sum()),'errors':int(err.sum()),'sla':float((delta==0).mean()),
          'tolerance_one':float((abs(delta)<=1).mean()),'early':float((delta<0).mean()),'late':float((delta>0).mean()),'mae':float(abs(delta).mean()),
          'clean_accuracy':ca,'gated_error_accuracy':ea,'f1':2*ca*ea/(ca+ea) if ca+ea else 0.,'fallbacks':int(m['fallback'][take].sum()),
          'native_sla':float((delta[~m['fallback'][err]]==0).mean()) if (~m['fallback'][err]).any() else None}
        if (m['median'][err]>=0).all():rows[cell]['median_sla']=float((m['median'][err]==d.target[err]).mean())
    result={'cells':rows}
    for scope,suffix in [('macro8',''),('q4','q4'),('q8','q8')]:
        values=[v for k,v in rows.items() if k.endswith(suffix)]
        result[scope]={k:float(np.mean([v[k] for v in values])) for k in ['sla','f1','tolerance_one','early','late','mae']}
    return result

def prm_metrics(d,m):
    auc=np.full(d.n,np.nan);mask=d.prm&m['valid'];folds=[]
    for i in np.flatnonzero(mask):a,b=d.off[i:i+2];auc[i]=within_auc(d.labels[a:b],m['scores'][a:b])
    for fold in range(5):
        steps=np.repeat(mask&(d.fold==fold),np.diff(d.off));y=d.labels[steps];s=m['scores'][steps]
        if not len(y):continue
        folds.append({'fold':fold,'answers':int((mask&(d.fold==fold)).sum()),'steps':len(y),'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s))})
    return {'answers':int(mask.sum()),'eligible':int(np.isfinite(auc).sum()),'within_auc':float(np.nanmean(auc)),
        'step_auroc_mean_folds':float(np.mean([f['auroc'] for f in folds])),'step_auprc_mean_folds':float(np.mean([f['auprc'] for f in folds])),
        'folds':folds},auc

def grid_prmscore(valid,y,eligible):
    """Vectorized exact total PRMScore, including evaluator's undefined=-1 rule."""
    v=np.asarray(valid,bool)[:,eligible];good=(np.asarray(y)[eligible]==0)[None,:]
    tp=np.sum(v&good,1);fp=np.sum(v&~good,1);tn=np.sum(~v&~good,1);fn=np.sum(~v&good,1)
    def ratio(a,b):return np.divide(a,b,out=np.full(np.shape(a),-1.,float),where=b!=0)
    p=ratio(tp,tp+fp);r=ratio(tp,tp+fn);f=ratio(2*p*r,p+r)
    p=ratio(tn,tn+fn);r=ratio(tn,tn+fp);nf=ratio(2*p*r,p+r)
    return (f+nf)/2

def official(d,valid,indices):
    preds=[];metadata=[]
    for i in indices:
        a,b=d.off[i:i+2];preds.append({'idx':d.ids[i],'labels':valid[a:b].astype(int).tolist()});metadata.append(d.meta_by_id[d.ids[i]])
    result=prm.prmbench_evaluate(preds,metadata);t=result['total']
    return {'prmscore':.5*(t['f1']+t['negative_f1']),**result}

def prmscores(d,methods):
    qgrid=np.linspace(*d.c['inner_threshold_quantiles']);tables={};predictions={};selection={}
    noncontrol=np.zeros(d.n,bool)
    for i in np.flatnonzero(d.prm):noncontrol[i]=d.meta_by_id[d.ids[i]]['classification']!='correct'
    for name,m in methods.items():
        if not (d.prm&m['valid']).any():continue
        fixed=np.zeros(int(d.off[-1]),bool);tuned=fixed.copy();choices=[]
        for outer in range(5):
            train=d.prm&m['valid']&(d.fold!=outer);test=d.prm&m['valid']&(d.fold==outer)
            trainsteps=np.flatnonzero(np.repeat(train,np.diff(d.off)));teststeps=np.flatnonzero(np.repeat(test,np.diff(d.off)))
            # Inner OOF label decisions, never outer held-out labels.
            inner_valid=np.zeros((len(qgrid),len(trainsteps)),bool);global_to_train=np.full(int(d.off[-1]),-1,int);global_to_train[trainsteps]=np.arange(len(trainsteps))
            is_new=name.startswith(('top5__','selected__','shuffle__'))
            if is_new:
                roster,pop,enc,kind=name.split('__');stage='em' if kind in ['ds','hem'] else 'spectral';arm=enc+'__'+kind
                stem=f'prm__fold{outer}__{roster}__all__{stage}'
                z=np.load(d.out/'jobs'/f'{stem}.npz');tau80=float(z[arm+'__threshold_q80']);outer_grid=z[arm+'__thresholds']
                filled=np.zeros(len(trainsteps),bool)
                for inner in range(5):
                    if inner==outer:continue
                    zi=np.load(d.out/'jobs'/f'{stem}__inner{inner}.npz');idx=zi['indices']
                    steps=np.concatenate([np.arange(d.off[i],d.off[i+1]) for i in idx]);dest=global_to_train[steps]
                    assert (dest>=0).all() and not filled[dest].any()
                    inner_valid[:,dest]=zi[arm+'__grid_valid'];filled[dest]=True
                assert filled.all()
            else:
                outer_grid=np.quantile(m['scores'][trainsteps],qgrid);tau80=float(np.quantile(m['scores'][trainsteps],.8))
                for inner in range(5):
                    if inner==outer:continue
                    fitting=np.repeat(train&(d.fold!=inner),np.diff(d.off));held=np.repeat(train&(d.fold==inner),np.diff(d.off))
                    thresholds=np.quantile(m['scores'][fitting],qgrid)
                    inner_valid[:,global_to_train[np.flatnonzero(held)]]=m['scores'][held][None,:]<thresholds[:,None]
            eligible=np.repeat(noncontrol,np.diff(d.off))[trainsteps]
            grid=grid_prmscore(inner_valid,d.labels[trainsteps],eligible);best=int(np.argmax(grid))
            fixed[teststeps]=m['scores'][teststeps]<tau80;tuned[teststeps]=m['scores'][teststeps]<outer_grid[best]
            choices.append({'fold':outer,'selected_quantile':float(qgrid[best]),'threshold':float(outer_grid[best]),'q80_threshold':tau80,'inner_prmscores':grid.tolist()})
        idx=np.flatnonzero(d.prm&m['valid'])
        tables[name]={'quantile_0.8':official(d,fixed,idx),'inner_selected':official(d,tuned,idx)}
        predictions[name+'__q80']=fixed;predictions[name+'__inner']=tuned;selection[name]=choices
        print(f'PRMScore {name} complete',flush=True)
    # Supervised model has a distinct access regime and is not included in fusion panels.
    supervised=np.zeros(int(d.off[-1]),bool)
    for i in np.flatnonzero(d.prm):
        rewards=d.meta_by_id[d.ids[i]]['rewards'];rewards=json.loads(rewards) if isinstance(rewards,str) else rewards
        a,b=d.off[i:i+2];supervised[a:b]=np.asarray(rewards)>=.5
    tables['supervised_qwen25math_prm7b']={'native_threshold_0.5':official(d,supervised,np.flatnonzero(d.prm))}
    np.savez_compressed(d.out/'PRM_OOF_DECISIONS.npz',**predictions)
    dump(d.out/'PRMSCORE.json',tables);dump(d.out/'PRMSCORE_THRESHOLD_SELECTION.json',selection)
    return tables
