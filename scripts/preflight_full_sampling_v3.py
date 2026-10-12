"""Replay every prior sampling output; feasibility/fidelity only, no metrics."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/localization_full_benchmark_v3'
SHORTLIST=ROOT/'results/localization_full_shortlist_v3'
OUT=ROOT/'results/localization_full_sampling_v3'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.answer_localization_v2 import json_safe
from spectral_utils.fusion_full_sampling import ARMS,score_full_sampling


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(json_safe(value),indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def load_source(rec):
    path=PARENT/'scores'/(rec['uid']+'.json');meta=load(path)
    assert sha(path.with_suffix('.npz'))==meta['array_sha256']
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    for key in ('uid','row_id','group_id','tokens','steps','cell'):assert meta[key]==rec[key]
    cell=PARENT/'inputs'/rec['cell'];offsets=np.load(cell/'token_offsets.npy',mmap_mode='r')
    raw=np.load(cell/'raw.npy',mmap_mode='r');lo,hi=offsets[rec['row']:rec['row']+2]
    raw=raw[lo:hi]
    return raw,arrays,meta


def compute(rec,target,manifest_sha=None):
    started=time.time();raw,original,meta=load_source(rec)
    identity='localization-cached-v1-20260907/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8'
    # The exact frozen kernel can reconstruct full anchors before pass2a finishes.
    arrays,methods,diagnostics=score_full_sampling(raw,original,meta,identity)
    checks=0
    if rec.get('reuse_original110'):
        for directory,selectors in [('fusion_sampling_replication_v1',('full','uniform','risk_top','dufs_transposed','dufs_permuted','window_diffusion')),
                                     ('fusion_entropy_sampling_v1',('entropy_tails','entropy_quantiles'))]:
            path=ROOT/'results'/directory/'scores'/(rec['uid']+'.json');old=load(path)
            assert sha(path.with_suffix('.npz'))==old['array_sha256']
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                for selector in selectors:
                    if selector+'__selected' in z:np.testing.assert_array_equal(arrays[selector+'__selected'],z[selector+'__selected'])
                for arm in ARMS:
                    if arm not in old['methods']:continue
                    d=methods[arm];reference=old['methods'][arm]
                    for key in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):
                        assert d.get(key)==reference.get(key),(rec['uid'],arm,key)
                    if d['valid']:
                        for suffix in ('window','risk'):
                            np.testing.assert_allclose(arrays[arm+'__'+suffix],z[arm+'__'+suffix],atol=1e-10,rtol=1e-10)
                    checks+=1
        assert checks==56
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(target.with_suffix('.npz'))
    save(target.with_suffix('.json'),dict(rec,methods=methods,diagnostics=diagnostics,routing=meta['routing'],
        array_sha256=sha(target.with_suffix('.npz')),source_array_sha256=meta['array_sha256'],
        source_metadata_sha256=sha(PARENT/'scores'/(rec['uid']+'.json')),
        manifest_sha256=manifest_sha,seconds=time.time()-started,labels_used=False,replayed_prior_arms=checks))
    return dict(uid=rec['uid'],replayed=checks,seconds=time.time()-started)


def main():
    manifest=load(PARENT/'MANIFEST.json');selected=[r for r in manifest['selected'] if r.get('reuse_original110')]
    assert len(selected)==110
    selected+=[min(manifest['selected'],key=lambda r:r['tokens']),max(manifest['selected'],key=lambda r:r['tokens'])]
    files=[Path(__file__),PARENT/'MANIFEST.json',PARENT/'SCORES_FROZEN.json',SHORTLIST/'MANIFEST.json']
    files += [Path(module.__file__).resolve() for name,module in list(sys.modules.items()) if name.startswith('spectral_utils') and getattr(module,'__file__',None)]
    hashes={str(path):sha(path) for path in files}
    existing=OUT/'PREFLIGHT_MANIFEST.json'
    if existing.exists():assert load(existing)['hashes']==hashes,'Preflight source changed; keep prior artifacts and use a new version'
    else:save(existing,dict(selected=selected,hashes=hashes,workers=2,labels_used=False))
    digest=sha(existing);started=time.time();done=[];pending=[]
    for rec in selected:
        path=OUT/'preflight'/rec['uid']
        if path.with_suffix('.json').exists():
            d=load(path.with_suffix('.json'));assert d['manifest_sha256']==digest and sha(path.with_suffix('.npz'))==d['array_sha256']
            done.append(dict(uid=rec['uid'],replayed=d['replayed_prior_arms'],seconds=d['seconds']))
        else:pending.append(rec)
    save(OUT/'PREFLIGHT_STATE.json',dict(phase='RUNNING',pid=os.getpid(),completed=len(done),total=len(selected)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(compute,rec,OUT/'preflight'/rec['uid'],digest) for rec in pending]
        for future in as_completed(futures):
            done.append(future.result())
            save(OUT/'PREFLIGHT_STATE.json',dict(phase='RUNNING',pid=os.getpid(),completed=len(done),total=len(selected),seconds=time.time()-started))
            if len(done)%10==0:print('Sampling fidelity',len(done),'/',len(selected),flush=True)
    for path,h in hashes.items():assert sha(path)==h,path
    assert len(done)==112 and sum(d['replayed'] for d in done)==110*56
    save(OUT/'PREFLIGHT.json',dict(status='PASS',purpose='fidelity/feasibility only; no pilot performance selection',
        records=112,prior_records=110,short_and_long=2,replayed_method_bundles=110*56,hashes=hashes,
        preflight_manifest_sha256=digest,seconds=time.time()-started))
    save(OUT/'PREFLIGHT_STATE.json',dict(phase='PASS',pid=os.getpid(),completed=112,total=112,seconds=time.time()-started))
    print('Full sampling bridge preflight PASS: 6160 method bundles and short/long support',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(OUT/'PREFLIGHT_STATE.json',dict(phase='FAILED',pid=os.getpid(),error=type(error).__name__+': '+str(error)));raise
