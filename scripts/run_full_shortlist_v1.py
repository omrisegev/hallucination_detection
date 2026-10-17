"""Full-population fixed shortlist after the immutable 19-anchor pass.

This run adds condition-100 Joint/graph and equal-graph controls using the
same-answer feature matrices and the original answer's Joint groups. It is an
answer-only, one-pass, unsupervised gray-box adapter. Labels are read only in
the later evaluation stage.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib, json, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/localization_full_benchmark_v3'
OUT=RUN/'shortlist_v1'
CAP=ROOT/'local_cache/short_cycle01_code'
import sys
sys.path.insert(0,str(CAP));import spectral_utils;spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.full_shortlist import NEW_ARMS, ROUTED_ARMS, score_shortlist

PARENT_ARMS=('moment__equal','moment__iu','moment__joint0','moment__graph010','moment__graph_perm',
 'context__equal','context__iu','context__joint0','context__graph010','context__graph_perm',
 'entropy_parent','single__joint0','single__graph010','single__graph_perm','dual__joint0',
 'dual__graph010','dual__graph_perm','dual__equal','dual__iu')
ARMS=PARENT_ARMS+ROUTED_ARMS
RELEASE=ROOT/'results/localization_prm_label_audit_v1/RELEASE_V3.json'
PARENT_EVAL=RUN/'evaluation'
PROTOCOL=ROOT/'docs/experiments/LOCALIZATION_FULL_SHORTLIST_V1.md'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [safe(v) for v in x]
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(safe(value),indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def state(phase,**kw):save(OUT/'RUN_STATE.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**kw))


def preflight():
    """Replay representative 110 graph outputs before freezing full outputs."""
    old=ROOT/'results/fusion_graph_conditioning_v1';m=load(old/'MANIFEST.json');tested=[]
    for rec in m['selected'][:6]:
        uid=rec['uid'];orig=load(RUN/'scores'/(uid+'.json'))
        with np.load(RUN/'scores'/(uid+'.npz'),allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
        features={'moment':arrays['moment__features'],'context':arrays['context__features']}
        names={b:orig['diagnostics']['banks'][b]['names'] for b in ('moment','context')}
        identity='localization-cached-v1-20260907/'+orig['cell']+'/'+orig['row_id']+'/moments27_local8'
        got,methods,_=score_shortlist(features,names,arrays,orig,orig['tokens'],arrays['step_starts'],arrays['step_ends'],identity)
        expected=load(old/'scores'/(uid+'.json'))
        with np.load(old/'scores'/(uid+'.npz'),allow_pickle=False) as ez:
            for arm in ROUTED_ARMS:
                if not expected['methods'][arm]['valid'] or not methods[arm]['valid']:continue
                np.testing.assert_allclose(got[arm+'__window'],ez[arm+'__window'],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(got[arm+'__risk'],ez[arm+'__risk'],atol=1e-10,rtol=1e-10)
                assert methods[arm]['peak']==expected['methods'][arm]['peak']
        tested.append(uid)
    save(OUT/'PREFLIGHT.json',dict(status='PASS',tested=tested,arms=ROUTED_ARMS,
       labels_used=False,scope='Six current110 exact replays of fixed condition100 and equal-graph outputs.'))
    print('SHORTLIST PREFLIGHT PASS',len(tested),flush=True)


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'MANIFEST.json').exists():
        m=load(OUT/'MANIFEST.json');assert m['status']=='FROZEN_FULL_SHORTLIST';print('Manifest already frozen');return
    preflight();full=load(RUN/'MANIFEST.json');frozen=load(RUN/'SCORES_FROZEN.json')
    assert frozen['status']=='COMPLETE_ANCHOR_PASS' and len(full['selected'])==13769
    hashes={str(Path(__file__)):sha(Path(__file__)),str(PROTOCOL):sha(PROTOCOL),str(RELEASE):sha(RELEASE),
            str(RUN/'MANIFEST.json'):sha(RUN/'MANIFEST.json'),str(RUN/'SCORES_FROZEN.json'):sha(RUN/'SCORES_FROZEN.json'),
            str(ROOT/'spectral_utils/full_shortlist.py'):sha(ROOT/'spectral_utils/full_shortlist.py'),
            str(OUT/'PREFLIGHT.json'):sha(OUT/'PREFLIGHT.json')}
    save(OUT/'MANIFEST.json',dict(status='FROZEN_FULL_SHORTLIST',release_id=full['release_id'],
       scoring_namespace=full['scoring_namespace'],selected=full['selected'],arms=ROUTED_ARMS,
       parent_arms=PARENT_ARMS,hashes=hashes,workers=3,seconds_cap=28800,labels_used=False,created_unix=time.time()))
    print('SHORTLIST MANIFEST FROZEN',flush=True)


def verify():
    m=load(OUT/'MANIFEST.json')
    for path,h in m['hashes'].items():assert sha(path)==h,path
    return m


def process_one(rec,digest):
    uid=rec['uid'];out=OUT/'scores'/(uid+'.json');npz=out.with_suffix('.npz')
    if out.exists():
        d=load(out);assert d['manifest_sha256']==digest and sha(npz)==d['array_sha256'];return
    started=time.monotonic();source=RUN/'scores'/(uid+'.json');source_npz=source.with_suffix('.npz')
    orig=load(source)
    with np.load(source_npz,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    features={'moment':arrays['moment__features'],'context':arrays['context__features']}
    names={b:orig['diagnostics']['banks'][b]['names'] for b in ('moment','context')}
    identity='localization-cached-v1-20260907/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8'
    new_arrays,methods,diagnostics=score_shortlist(features,names,arrays,orig,rec['tokens'],arrays['step_starts'],arrays['step_ends'],identity)
    npz.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(npz)+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**new_arrays)
    tmp.replace(npz)
    save(out,{**rec,'methods':methods,'diagnostics':diagnostics,'manifest_sha256':digest,
       'source_array_sha256':orig['array_sha256'],'array_sha256':sha(npz),'labels_used':False,
       'seconds':time.monotonic()-started})


def scores():
    m=verify();digest=sha(OUT/'MANIFEST.json');records=m['selected'];remaining=[];done=0
    if (OUT/'SCORES_FROZEN.json').exists():
        f=load(OUT/'SCORES_FROZEN.json');assert f['manifest_sha256']==digest
        for p,h in f['files'].items():assert sha(p)==h
        print('Existing shortlist scores verified');return
    for rec in records:
        p=OUT/'scores'/(rec['uid']+'.json')
        if p.exists():
            d=load(p);assert d['manifest_sha256']==digest and sha(p.with_suffix('.npz'))==d['array_sha256'];done+=1
        else:remaining.append(rec)
    started=time.monotonic();state('RUNNING',completed=done,total=len(records),seconds=0.)
    idx=0
    with ProcessPoolExecutor(max_workers=m['workers']) as pool:
        active={}
        while active or idx<len(remaining):
            while len(active)<m['workers'] and idx<len(remaining) and time.monotonic()-started<m['seconds_cap']:
                rec=remaining[idx];idx+=1;active[pool.submit(process_one,rec,digest)]=rec['uid']
            if not active:break
            ready,_=wait(active,return_when=FIRST_COMPLETED)
            for fut in ready:fut.result();del active[fut];done+=1
            state('RUNNING',completed=done,total=len(records),seconds=time.monotonic()-started)
            if done%25==0 or done==len(records):print('Shortlist',done,'/',len(records),flush=True)
    if done!=len(records):state('PAUSED_AT_CAP',completed=done,total=len(records));return
    files=sorted((OUT/'scores').glob('*'));assert len(files)==2*len(records)
    save(OUT/'SCORES_FROZEN.json',dict(status='COMPLETE',manifest_sha256=digest,labels_used=False,
       files={str(p):sha(p) for p in files},seconds=time.monotonic()-started))
    state('SCORES_COMPLETE',completed=done,total=len(records));print('SHORTLIST SCORES COMPLETE',flush=True)


def evaluate():
    """Assemble new score arrays beside pass-1 JOINED arrays; labels enter here."""
    from importlib.util import spec_from_file_location,module_from_spec
    q=ROOT/'scripts/evaluate_localization_full_anchors_v3.py';spec=spec_from_file_location('full_anchor_metrics',q);mod=module_from_spec(spec);spec.loader.exec_module(mod)
    m=verify();f=load(OUT/'SCORES_FROZEN.json');assert f['manifest_sha256']==sha(OUT/'MANIFEST.json')
    for p,h in f['files'].items():assert sha(p)==h
    parent=load(PARENT_EVAL/'JOINED.json');
    with np.load(PARENT_EVAL/'JOINED.npz',allow_pickle=False) as z: old={k:z[k] for k in z.files}
    n=len(m['selected']);assert parent['records']==m['selected'];assert parent['arms']==PARENT_ARMS
    k=len(ARMS);offsets=old['offsets'];total=int(offsets[-1]);scores=np.full((total,k),np.nan);scores[:,:len(PARENT_ARMS)]=old['scores'];valid=np.zeros((n,k),bool);decision=np.zeros((n,k),bool);pred=np.full((n,k),-2,np.int32);peaks=np.full((n,k),-2,np.int32);pred[:,:len(PARENT_ARMS)]=old['predictions'];peaks[:,:len(PARENT_ARMS)]=old['peaks'];valid[:,:len(PARENT_ARMS)]=old['valid'];decision[:,:len(PARENT_ARMS)]=old['decision']
    records=m['selected'];by_arm={a:i for i,a in enumerate(ARMS)}
    for i,rec in enumerate(records):
        d=load(OUT/'scores'/(rec['uid']+'.json'));lo,hi=offsets[i:i+2]
        with np.load(OUT/'scores'/(rec['uid']+'.npz'),allow_pickle=False) as z:
            for arm,j in by_arm.items():
                if arm in PARENT_ARMS:continue
                md=d['methods'][arm];valid[i,j]=md['valid'];decision[i,j]=md['decision_valid'];pred[i,j]=md.get('prediction',-2) if md.get('prediction') is not None else -2;peaks[i,j]=md.get('peak',-2) if md.get('peak') is not None else -2
                if md['valid']:
                    scores[lo:hi,j]=z[arm+'__risk'];assert np.isfinite(scores[lo:hi,j]).all()
    release=load(RELEASE);target=np.full(n,-2,np.int32);labels=np.full(total,-2,np.int8)
    for cell in sorted({r['cell'] for r in records}):
        info=release['cells'][cell]
        with np.load(info['label_path'],allow_pickle=False) as q:
            pos={str(v):i for i,v in enumerate(q['row_ids'])}
            for i,rec in enumerate(records):
                if rec['cell']!=cell:continue
                z=pos[rec['row_id']];lo,hi=offsets[i:i+2]
                if cell.startswith('prm'):
                    a,b=q['step_flag_offsets'][z:z+2];labels[lo:hi]=q['step_error_flags'][a:b];target[i]=-2
                else:target[i]=int(q['first_error'][z])
    joined=dict(records=records,arms=ARMS,scores_freeze_sha256=sha(OUT/'SCORES_FROZEN.json'))
    tmp=OUT/'COMBINED.npz.tmp'
    with tmp.open('wb') as fh:np.savez_compressed(fh,offsets=offsets,scores=scores,valid=valid,decision=decision,predictions=pred,peaks=peaks,target=target,labels=labels)
    tmp.replace(OUT/'COMBINED.npz');joined['arrays_sha256']=sha(OUT/'COMBINED.npz');save(OUT/'COMBINED.json',joined)
    # Convert the compact matrix to the anchor evaluator's in-memory contract.
    a=dict(offsets=offsets,scores=scores,valid=valid,decision=decision,predictions=pred,peaks=peaks,target=target,labels=labels,within=np.full((n,k),np.nan))
    owner=np.repeat(np.arange(n),np.diff(offsets));
    for i,rec in enumerate(records):
        if not rec['cell'].startswith('prm'):continue
        lo,hi=offsets[i:i+2]
        for j in range(k):
            if valid[i,j]:
                x=mod.rank_auc(labels[lo:hi],scores[lo:hi,j]);a['within'][i,j]=np.nan if x is None else x
    metrics={arm:mod.metric(records,a,j) for j,arm in enumerate(ARMS)}
    # Pass-1 point values must remain bit-identical.
    old_metrics=load(PARENT_EVAL/'METRICS.json')['metrics']
    for arm in PARENT_ARMS:
        assert metrics[arm]==old_metrics[arm],arm
    save(OUT/'METRICS.json',dict(status='POINT_COMPLETE',metrics=metrics,records=len(records),arms=ARMS,labels_used=True))
    save(OUT/'STATE.json',dict(phase='POINT_COMPLETE',rows=n,methods=k))
    print('SHORTLIST POINT METRICS COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('prepare','preflight','scores','evaluate'),required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True)
    {'prepare':prepare,'preflight':preflight,'scores':scores,'evaluate':evaluate}[a.phase]()


if __name__=='__main__':main()
