"""Exact roster joins, input freeze, anchor replay and reusable profile extraction."""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from .core import CHANNELS, READOUTS, profiles

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if hasattr(x,'__dict__'):return x.__dict__
    raise TypeError(type(x).__name__)

def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=jsonable),encoding='utf8')

def config(path):
    path=Path(path).resolve();c=json.loads(path.read_text(encoding='utf8'))
    c['paths']={k:str((path.parent/v).resolve()) for k,v in c['paths'].items()}
    if c['temperature']!=1 or c['em_stable_iterations']!=3 or c['em_probability_clip']!=1e-6:raise ValueError('unsupported protocol settings')
    return c

class Dataset:
    def __init__(self,c):
        self.c=c;self.out=Path(c['paths']['output']);self.out.mkdir(parents=True,exist_ok=True)
        j=json.loads(Path(c['paths']['roster']).read_text(encoding='utf8'));self.records=j['records'];self.arms=j['arms']
        self.z=np.load(c['paths']['joined']);self.off=self.z['offsets'];self.target=self.z['target'];self.labels=self.z['labels']
        self.cells=np.array([r['cell'] for r in self.records]);self.groups=np.array([r['group_id'] for r in self.records])
        self.ids=np.array([r['row_id'] for r in self.records]);self.n=len(self.records)
        outer=json.loads(Path(c['paths']['folds']).read_text(encoding='utf8'))['outer']
        self.fold=np.array([outer[g] for g in self.groups],int)
        self.pb=np.char.startswith(self.cells,'pb_');self.prm=~self.pb
        assert (self.n,self.pb.sum(),self.prm.sum(),(self.pb&(self.target>=0)).sum())==(13769,6800,6969,4442)
        assert set(self.fold)==set(range(5)) and len(self.off)==self.n+1
        assert len(set(r['uid'] for r in self.records))==self.n
        assert all(self.off[i+1]-self.off[i]==r['steps'] for i,r in enumerate(self.records))
        assert all(-1<=self.target[i]<self.off[i+1]-self.off[i] for i in np.flatnonzero(self.pb))
        ct=np.load(c['paths']['ct7']);self.gate=ct['gate'];self.references={'ct7':ct['step_scores']}
        tok=np.load(c['paths']['token_fusion']);self.references.update(token_lsml=tok['l_sml'],token_equal=tok['equal'])
        self.token_gate=tok['gate_open']
        ed=np.load(c['paths']['mindgap']);self.references['mindgap_previous_unadjusted']=ed['step_worst']
        # Replay the actual existing adapter: adjusted EMA and negative flux only.
        adapter_dir=Path(__file__).resolve().parents[2]/'localization'
        sys.path.insert(0,str(adapter_dir))
        from localization_metrics import step_drop_scores
        replay_path=self.out/'MINDGAP_ADAPTER_REPLAY.npz'
        if replay_path.exists():self.references['mindgap']=np.load(replay_path)['scores']
        else:
            bank=np.load(c['paths']['tokens']);spans=bank['step_spans'];toff=ed['token_offsets'];evidence=ed['evidence']
            scores=np.empty(int(self.off[-1]))
            for i,(a,b) in enumerate(zip(self.off[:-1],self.off[1:])):
                ta,tb=toff[i:i+2];scores[a:b]=step_drop_scores(evidence[ta:tb],spans[a:b],ema_span=5)
            assert np.isfinite(scores).all()
            np.savez(replay_path,scores=scores);self.references['mindgap']=scores
            dump(self.out/'MINDGAP_REPLAY_MANIFEST.json',{'evidence_source_sha256':digest(c['paths']['mindgap']),
                 'adapter_sha256':digest(adapter_dir/'localization_metrics.py'),'ema_sha256':digest(adapter_dir/'evidence_drop.py'),
                 'scores_sha256':digest(replay_path),'answers':self.n,'source_precision':'cached float32 top20 evidence',
                 'aggregation':'worst negative flux in step; adjusted EMA span5; argmax locator; no gate',
                 'not_paper_replication':'Qwen scorers, teacher-forced caches, registered token-to-step adapter'})
        self.reference_valid={k:np.ones(self.n,bool) for k in self.references}
        for name in ['dual__iu','dual__joint0','dual__equal','context__iu','single__joint0']:
            col=self.arms.index(name)
            self.references[name]=self.z['scores'][:,col]
            self.reference_valid[name]=self.z['valid'][:,col]
        for name,s in self.references.items():
            assert np.shape(s)==(self.off[-1],)
            valid_steps=np.repeat(self.reference_valid[name],np.diff(self.off))
            assert np.isfinite(s[valid_steps]).all()
        self.meta_by_id={m['idx']:m for m in pickle.load(open(c['paths']['prm_metadata'],'rb')).values()}
        assert set(self.ids[self.prm])==set(self.meta_by_id)
        for i in np.flatnonzero(self.prm):
            m=self.meta_by_id[self.ids[i]];a,b=self.off[i:i+2]
            assert m['n_steps']==b-a
            err=m['error_steps'];err=json.loads(err) if isinstance(err,str) else err
            # PRMBench's documented metadata uses 1-based step indices.
            expected=np.isin(np.arange(1,b-a+1),err).astype(int)
            assert np.array_equal(expected,self.labels[a:b]),f'PRM label mismatch {self.ids[i]}'
        self.profiles=None
    def peaks(self,s):return np.array([np.argmax(s[a:b]) for a,b in zip(self.off[:-1],self.off[1:])])
    def pb_metrics(self,p,gate=None):
        rows={}
        for cell in sorted(set(self.cells[self.pb])):
            take=self.cells==cell;err=take&(self.target>=0);clean=take&(self.target<0);d=p[err]-self.target[err]
            ca=float((~self.gate[clean]).mean()) if gate is None else float((~gate[clean]).mean())
            opened=self.gate if gate is None else gate
            ea=float(((p[err]==self.target[err])&opened[err]).mean())
            rows[cell]={'n':int(take.sum()),'erroneous':int(err.sum()),'sla':float((d==0).mean()),'tolerance_one':float((abs(d)<=1).mean()),
                        'early':float((d<0).mean()),'late':float((d>0).mean()),'mae':float(abs(d).mean()),'clean_accuracy':ca,
                        'gated_error_accuracy':ea,'macro_f1':2*ca*ea/(ca+ea) if ca+ea else 0.}
        return {'cells':rows,'sla':float(np.mean([x['sla'] for x in rows.values()])),
                'macro_f1':float(np.mean([x['macro_f1'] for x in rows.values()]))}

def within_auc(y,s):
    y=np.asarray(y,bool);n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return np.nan
    return float((rankdata(s)[y].sum()-n1*(n1+1)/2)/(n1*n0))

def prepare(d):
    paths=d.c['paths'];freeze={k:{'path':v,'bytes':Path(v).stat().st_size,'sha256':digest(v)} for k,v in paths.items() if k!='output'}
    freeze['protocol']=d.c
    dest=d.out/'INPUT_FREEZE.json'
    if dest.exists() and json.loads(dest.read_text(encoding='utf8'))!=freeze:raise ValueError('input/config changed; use a new output directory')
    dump(dest,freeze)
    anchors={k:d.pb_metrics(d.peaks(v)) for k,v in d.references.items() if d.reference_valid[k].all()}
    anchors['historical_coverage']={k:{'valid':int(v.sum()),'missing_ids':d.ids[~v].tolist()} for k,v in d.reference_valid.items()}
    expected={'token_lsml':.3592,'token_equal':.3259}
    for k,want in expected.items():
        assert abs(anchors[k]['sla']-want)<.000051,(k,anchors[k]['sla'],want)
    assert abs(anchors['ct7']['macro_f1']-.41188745848863717)<1e-12
    assert freeze['ct7']['sha256']=='9d10d2ff04402f56d6b04c643bfc87a55413786e67d721ad4a28b1578cd3b430'
    auc=np.array([within_auc(d.labels[a:b],d.references['ct7'][a:b]) for i,(a,b) in enumerate(zip(d.off[:-1],d.off[1:])) if d.prm[i]])
    assert np.isfinite(auc).sum()==6030 and abs(np.nanmean(auc)-.7723966352864217)<1e-12
    anchors['ct7_prm']={'within_auc':float(np.nanmean(auc)),'eligible':int(np.isfinite(auc).sum())}
    anchors['historical_native_gates']={k:d.pb_metrics(d.peaks(d.references[k]),d.token_gate) for k in ['token_lsml','token_equal']}
    dump(d.out/'ANCHOR_REPLAY.json',anchors)
    print('Anchors passed: '+str({k:anchors[k]['sla'] for k in expected}),flush=True)
    if (d.out/'PROFILES_COMPLETE.json').exists():return
    z=np.load(paths['tokens']);tokens=z['tokens'];toff=z['token_offsets'];spans=z['step_spans']
    assert list(z['channels'])==CHANNELS and tokens.shape==(toff[-1],11)
    assert spans.shape==(d.off[-1],2) and np.isfinite(tokens).all()
    assert np.array_equal(np.diff(toff),[r['tokens'] for r in d.records])
    ed=np.load(paths['mindgap']);assert np.array_equal(ed['token_offsets'],toff)
    shape=(int(d.off[-1]),11,len(READOUTS))
    out=np.lib.format.open_memmap(d.out/'profiles.npy',mode='w+',dtype='float64',shape=shape)
    shuffled=np.lib.format.open_memmap(d.out/'shuffled_top5.npy',mode='w+',dtype='float64',shape=shape[:2])
    lengths=np.diff(spans,axis=1).ravel();np.save(d.out/'step_lengths.npy',lengths)
    started=time.perf_counter()
    for i in range(d.n):
        a,b=d.off[i:i+2];ta,tb=toff[i:i+2];x=tokens[ta:tb].astype(float)
        out[a:b]=profiles(x,spans[a:b])
        rng=np.random.default_rng(np.random.SeedSequence([d.c['seed'],i]))
        perm=np.column_stack([rng.permutation(x[:,j]) for j in range(11)])
        shuffled[a:b]=profiles(perm,spans[a:b])[:,:,0]
        if i%1000==0:print(f'profiles {i}/{d.n}, {time.perf_counter()-started:.1f}s',flush=True)
    out.flush();shuffled.flush()
    dump(d.out/'PROFILES_COMPLETE.json',{'seconds':time.perf_counter()-started,'profiles_sha256':digest(d.out/'profiles.npy'),
         'shuffle_sha256':digest(d.out/'shuffled_top5.npy'),'shuffle':'independent permutations per channel within answer; fixed seed; models refitted'})
