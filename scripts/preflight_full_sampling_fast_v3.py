"""Execution-only ARI acceleration, checked against every plain bridge output."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
import numpy as np
import preflight_full_sampling_v3 as plain
from spectral_utils.historical_joint_acceleration import compatible_ari, preflight as ari_preflight, small_partition_ari
from sklearn.metrics import adjusted_rand_score

ROOT,OUT=plain.ROOT,plain.OUT
sha,load,save=plain.sha,plain.load,plain.save


def scientific(value):
    """Ignore only elapsed-time fields when comparing diagnostic dictionaries."""
    if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if not k.endswith('seconds')}
    if isinstance(value,list):return [scientific(v) for v in value]
    return value


def compute(rec,target,digest):
    reference={'joint_lsml':sys.modules['spectral_utils.joint_lsml']}
    with compatible_ari(reference) as acceleration:
        result=plain.compute(rec,target,digest)
    target=Path(target);meta=load(target.with_suffix('.json'))
    meta['execution_acceleration']=acceleration
    save(target.with_suffix('.json'),meta)
    result['ari_calls']=acceleration['calls']
    return result


def check_record(rec,digest):
    target=OUT/'preflight_fast'/rec['uid'];oldpath=OUT/'preflight'/rec['uid']
    if target.with_suffix('.json').exists():
        meta=load(target.with_suffix('.json'));assert meta['manifest_sha256']==digest
        assert sha(target.with_suffix('.npz'))==meta['array_sha256']
    else:
        compute(rec,target,digest);meta=load(target.with_suffix('.json'))
    old=load(oldpath.with_suffix('.json'))
    assert old['manifest_sha256']==sha(OUT/'PREFLIGHT_MANIFEST.json')
    assert sha(oldpath.with_suffix('.npz'))==old['array_sha256']
    with np.load(target.with_suffix('.npz'),allow_pickle=False) as z, np.load(oldpath.with_suffix('.npz'),allow_pickle=False) as previous:
        assert set(z.files)==set(previous.files)
        for key in z.files:np.testing.assert_array_equal(z[key],previous[key],err_msg=rec['uid']+'/'+key)
        arrays=len(z.files)
    for key in ('methods','diagnostics','routing'):
        assert scientific(meta[key])==scientific(old[key]),(rec['uid'],key)
    return dict(uid=rec['uid'],arrays=arrays,ari_calls=meta['execution_acceleration']['calls'],
                fast_seconds=meta['seconds'],plain_seconds=old['seconds'])


def main():
    assert load(OUT/'PREFLIGHT.json')['status']=='PASS'
    original=load(OUT/'PREFLIGHT_MANIFEST.json');selected=original['selected']
    paths={Path(__file__),OUT/'PREFLIGHT.json',OUT/'PREFLIGHT_MANIFEST.json'}
    paths|={Path(mod.__file__).resolve() for name,mod in list(sys.modules.items())
            if name.startswith('spectral_utils') and getattr(mod,'__file__',None)}
    hashes={**original['hashes'],**{str(path):sha(path) for path in paths}}
    for path,h in hashes.items():assert sha(path)==h,path
    manifest=OUT/'FAST_MANIFEST.json'
    if manifest.exists():assert load(manifest)['hashes']==hashes
    else:save(manifest,dict(selected=selected,hashes=hashes,workers=2,
                           purpose='exact execution equivalence; no performance selection'))
    digest=sha(manifest);arithmetic=ari_preflight();started=time.time();done=[]
    rng=np.random.default_rng(20260908330)
    for _ in range(200):
        a,b=rng.integers(0,8,(2,27))
        assert small_partition_ari(a,b)==adjusted_rand_score(a,b)
    arithmetic['additional_P27_cases']=200
    save(OUT/'FAST_STATE.json',dict(phase='RUNNING',pid=os.getpid(),completed=0,total=len(selected)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(check_record,rec,digest) for rec in selected]
        for future in as_completed(futures):
            done.append(future.result())
            save(OUT/'FAST_STATE.json',dict(phase='RUNNING',pid=os.getpid(),completed=len(done),total=len(selected),seconds=time.time()-started))
            if len(done)%10==0:print('Exact accelerated sampling replay',len(done),'/',len(selected),flush=True)
    for path,h in hashes.items():assert sha(path)==h,path
    save(OUT/'PREFLIGHT_FAST.json',dict(status='PASS',arithmetic=arithmetic,records=len(done),
        exact_array_checks=sum(d['arrays'] for d in done),ari_calls=sum(d['ari_calls'] for d in done),
        plain_record_seconds=sum(d['plain_seconds'] for d in done),fast_record_seconds=sum(d['fast_seconds'] for d in done),
        machine_contention_held_fixed=False,manifest_sha256=digest,hashes=hashes,rows=done,
        seconds=time.time()-started,scope='execution-only exact replay, no new scientific result'))
    save(OUT/'FAST_STATE.json',dict(phase='PASS',pid=os.getpid(),completed=len(done),total=len(selected),seconds=time.time()-started))
    print('Exact accelerated sampling bridge PASS',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(OUT/'FAST_STATE.json',dict(phase='FAILED',pid=os.getpid(),error=repr(error)));raise
