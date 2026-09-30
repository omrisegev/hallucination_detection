"""Checkpointed full sampling benchmark; at most two active answer jobs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys
import time
import preflight_full_sampling_v3 as preflight
import preflight_full_sampling_fast_v3 as accelerated
from spectral_utils.full_sampling_evaluation import contrasts

ROOT=preflight.ROOT;OUT=preflight.OUT;PARENT=preflight.PARENT;SHORTLIST=preflight.SHORTLIST
HISTORICAL=ROOT/'results/historical_fusion_refit_v3'
PROTOCOL=ROOT/'docs/experiments/FULL_LOCALIZATION_SAMPLING_V3.md'
sha,load,save=preflight.sha,preflight.load,preflight.save


def state(phase,**extra):
    save(OUT/'RUN_STATE.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**extra))


@contextmanager
def exclusive_run():
    """OS releases the byte lock even if this invocation dies."""
    import msvcrt
    path=OUT/'RUN.lock';handle=path.open('a+b')
    if path.stat().st_size==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        handle.close();raise RuntimeError('Another full sampling invocation owns RUN.lock')
    try:yield
    finally:
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1);handle.close()


def verify():
    manifest=load(OUT/'MANIFEST.json')
    for path,h in manifest['hashes'].items():assert sha(path)==h,path
    return manifest


def prepare():
    if (OUT/'MANIFEST.json').exists():return verify()
    if not (OUT/'PREFLIGHT.json').exists():
        state('WAITING_FOR_PREFLIGHT');return None
    fidelity=load(OUT/'PREFLIGHT.json');assert fidelity['status']=='PASS'
    if not (OUT/'PREFLIGHT_FAST.json').exists():
        state('WAITING_FOR_EXACT_ACCELERATION_REPLAY');return None
    fast=load(OUT/'PREFLIGHT_FAST.json');assert fast['status']=='PASS' and fast['records']==112
    assert fast['manifest_sha256']==sha(OUT/'FAST_MANIFEST.json')
    assert fidelity['replayed_method_bundles']==6160
    assert fidelity['preflight_manifest_sha256']==sha(OUT/'PREFLIGHT_MANIFEST.json')
    parent=load(PARENT/'MANIFEST.json');shortlist=load(SHORTLIST/'MANIFEST.json')
    assert parent['selected']==shortlist['selected']
    assert load(HISTORICAL/'REVIEW_SUPPLEMENT.json')['status']=='PASS'
    history=load(HISTORICAL/'JOINED.json');historical_arms=history['arms'][19:]
    assert len(historical_arms)==5 and history['records']==parent['selected']
    paths={Path(__file__),Path(preflight.__file__),Path(accelerated.__file__),PROTOCOL,
        ROOT/'scripts/evaluate_full_sampling_v3.py',ROOT/'scripts/evaluate_localization_full_anchors_v3.py',
        OUT/'PREFLIGHT.json',OUT/'PREFLIGHT_MANIFEST.json',OUT/'PREFLIGHT_FAST.json',OUT/'FAST_MANIFEST.json',HISTORICAL/'JOINED.json',
        HISTORICAL/'JOINED.npz',HISTORICAL/'METRICS.json',HISTORICAL/'REVIEW_SUPPLEMENT.json',
        ROOT/'results/localization_source_group_audit_v1/FOLDS_V2.json'}
    paths|={Path(mod.__file__).resolve() for name,mod in list(sys.modules.items())
            if name.startswith('spectral_utils') and getattr(mod,'__file__',None)}
    hashes=deepcopy(parent['hashes']);hashes.update(fidelity['hashes']);hashes.update(fast['hashes'])
    hashes.update({str(path):sha(path) for path in paths})
    for path,h in hashes.items():assert sha(path)==h,path
    manifest=dict(status='FROZEN_FULL_DEVELOPMENT_SAMPLING',selected=parent['selected'],
        release_id=parent['release_id'],arms=list(preflight.ARMS),shortlist_arms=shortlist['arms'],
        historical_arms=historical_arms,contrasts=contrasts(),hashes=hashes,workers=2,
        submission_seconds_cap=28800,labels_used_for_scoring=False,created_unix=time.time(),
        preflight_purpose='fidelity/feasibility only; no performance selection')
    assert len(manifest['selected'])==13769 and len(manifest['arms'])==56
    save(OUT/'MANIFEST.json',manifest);state('PREPARED_FULL_SAMPLING',records=13769,methods=56)
    return manifest


def adopt_preflight(rec,digest):
    source=OUT/'preflight_fast'/rec['uid'];path=source.with_suffix('.json')
    if not path.exists():return False
    meta=load(path)
    assert meta['manifest_sha256']==sha(OUT/'FAST_MANIFEST.json')
    assert sha(source.with_suffix('.npz'))==meta['array_sha256']
    for key in ('uid','row_id','group_id','tokens','steps','cell'):assert meta[key]==rec[key]
    target=OUT/'scores'/rec['uid'];target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix('.npz.tmp');shutil.copyfile(source.with_suffix('.npz'),tmp);tmp.replace(target.with_suffix('.npz'))
    meta.update(manifest_sha256=digest,preflight_metadata_sha256=sha(path),reused_preflight=True)
    save(target.with_suffix('.json'),meta)
    return True


def score(manifest):
    digest=sha(OUT/'MANIFEST.json');pending=[];completed=0
    for rec in manifest['selected']:
        path=OUT/'scores'/(rec['uid']+'.json')
        if not path.exists():adopt_preflight(rec,digest)
        if path.exists():
            d=load(path);assert d['manifest_sha256']==digest
            assert sha(path.with_suffix('.npz'))==d['array_sha256']
            assert set(d['methods'])==set(manifest['arms']);completed+=1
        else:pending.append(rec)
    started=time.time();n=len(manifest['selected']);cap=manifest['submission_seconds_cap']
    state('SCORING',completed=completed,total=n,workers=manifest['workers'])
    # Only active work is submitted: no 13k-future queue and no lost cap state.
    if pending:
        iterator=iter(pending)
        with ProcessPoolExecutor(max_workers=manifest['workers']) as pool:
            active={}
            def submit_next():
                if time.time()-started>=cap:return False
                rec=next(iterator,None)
                if rec is None:return False
                active[pool.submit(accelerated.compute,rec,OUT/'scores'/rec['uid'],digest)]=rec
                return True
            for _ in range(manifest['workers']):submit_next()
            while active:
                ready,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
                for future in ready:
                    active.pop(future);future.result();completed+=1;submit_next()
                state('SCORING' if time.time()-started<cap else 'DRAINING_INVOCATION_CAP',
                    completed=completed,total=n,active=len(active),seconds=time.time()-started)
                if ready and completed%25==0:print('Full sampling',completed,'/',n,flush=True)
    if completed<n:
        state('CHECKPOINTED_INVOCATION_CAP',completed=completed,total=n);return False
    verify()
    files={str(path):sha(path) for rec in manifest['selected'] for path in
           (OUT/'scores'/(rec['uid']+'.json'),OUT/'scores'/(rec['uid']+'.npz'))}
    frozen=OUT/'SCORES_FROZEN.json'
    if frozen.exists():assert load(frozen)['files']==files and load(frozen)['manifest_sha256']==digest
    else:save(frozen,dict(status='COMPLETE_FULL_SAMPLING',files=files,rows=n,
                         arms=manifest['arms'],manifest_sha256=digest,labels_used=False))
    return True


def evaluate():
    review=SHORTLIST/'evaluation/REVIEW.json'
    if not review.exists() or load(review)['status']!='PASS':
        state('SCORES_COMPLETE_WAITING_FOR_SHORTLIST_REVIEW',completed=13769,total=13769)
        return
    module_path=ROOT/'scripts/evaluate_full_sampling_v3.py'
    spec=importlib.util.spec_from_file_location('full_sampling_evaluator',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()
    state('COMPLETE_REVIEWED_FULL_SAMPLING',completed=13769,total=13769,report='evaluation/REPORT.html')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','run','evaluate'),default='run')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    with exclusive_run():
        try:
            manifest=prepare()
            if manifest is None or args.phase=='prepare':return
            if args.phase=='evaluate':evaluate()
            elif score(manifest):evaluate()
        except BaseException as error:
            state('FAILED_CHECKPOINTS_PRESERVED',error=repr(error));raise


if __name__=='__main__':main()
