"""Extend the full corrected historical panel with the original Joint roster."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import gc
import importlib.util
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    output=importlib.util.module_from_spec(spec);spec.loader.exec_module(output)
    return output


base=module('historical_base_driver','scripts/run_historical_fusion_refit_v3.py')
joint=module('historical_joint_core','spectral_utils/historical_joint_refit.py')
fast=module('historical_joint_acceleration','spectral_utils/historical_joint_acceleration.py')
BASE=ROOT/'results/historical_fusion_refit_v3'
OUT=ROOT/'results/historical_joint_refit_v3'
PROTOCOL=ROOT/'docs/experiments/HISTORICAL_JOINT_REFIT_V3.md'


def state(phase,**extra):
    base.save(OUT/'RUN_STATE.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**extra))


def verify():
    manifest=base.load(OUT/'MANIFEST.json')
    for path,h in manifest['hashes'].items():assert base.sha(path)==h,path
    return manifest


def prepare_run():
    reference,files=base.core.reference_modules(base.SOURCE)
    if (OUT/'MANIFEST.json').exists():return verify(),reference
    assert base.load(OUT/'PREFLIGHT.json')['status']=='PASS'
    assert base.load(OUT/'PREFLIGHT_FAST.json')['status']=='PASS'
    assert base.load(BASE/'REVIEW.json')['status']=='PASS','First historical panel must finish review before extension'
    parent=base.verify();freeze=base.load(BASE/'SCORES_FROZEN.json')
    preflight=base.load(OUT/'PREFLIGHT_FAST.json')
    for path,h in preflight['source_hashes'].items():assert base.sha(path)==h,path
    hashes=dict(parent['hashes']);hashes.update(freeze['files']);hashes.update(preflight['source_hashes'])
    files=[Path(x) for x in files]+[Path(__file__),ROOT/'spectral_utils/historical_joint_refit.py',
        ROOT/'scripts/evaluate_historical_joint_refit_v3.py',ROOT/'spectral_utils/historical_joint_evaluation.py',PROTOCOL,BASE/'MANIFEST.json',BASE/'REVIEW.json',
        BASE/'SCORES_FROZEN.json',OUT/'PREFLIGHT.json',OUT/'PREFLIGHT_FAST.json',
        ROOT/'spectral_utils/historical_joint_acceleration.py']
    for path in files:hashes[str(path)]=base.sha(path)
    manifest=dict(status='FROZEN_FULL_DEVELOPMENT_HISTORICAL_JOINT_EXTENSION',selected=parent['selected'],
        release_id=parent['release_id'],arms=parent['arms']+list(joint.ARMS),new_arms=list(joint.ARMS),
        jobs=parent['jobs'],hashes=hashes,reference_checkout=parent['reference_checkout'],
        versions={**parent['versions'],'sklearn':__import__('sklearn').__version__},
        execution_acceleration='small-partition ARI via exact dense integer pair-confusion counts; full historical replay PASS',
        label_access=parent['label_access'],workers=1,
        invocation_seconds_cap=28800,created_unix=time.time())
    base.save(OUT/'MANIFEST.json',manifest);return manifest,reference


def run():
    manifest,reference=prepare_run();mh=base.sha(OUT/'MANIFEST.json');folds=base.load(base.FOLDS)
    current_cell=None;started=time.time();done=0
    for job in manifest['jobs']:
        name=base.job_name(job);target=OUT/'fits'/name
        if target.with_suffix('.json').exists():
            previous=base.load(target.with_suffix('.json'))
            assert previous['manifest_sha256']==mh and base.sha(target.with_suffix('.npz'))==previous['array_sha256']
            done+=1;continue
        state('FITTING',completed=done,total=len(manifest['jobs']),job=job)
        if current_cell!=job['cell']:
            cell=base.cell_data(job['cell']);current_cell=job['cell']
            records=sorted([r for r in manifest['selected'] if r['cell']==current_cell],key=lambda r:r['row'])
            assert [r['row_id'] for r in records]==list(map(str,cell['row_ids']))
            groups=[r['group_id'] for r in records]
        train,evaluate=base.core.fold_masks(groups,folds['outer'],folds['inner'][str(job['outer'])],job['outer'],job['inner'])
        source=BASE/'fits'/name;sm=base.load(source.with_suffix('.json'))
        assert base.sha(source.with_suffix('.npz'))==sm['array_sha256']
        with np.load(source.with_suffix('.npz'),allow_pickle=False) as saved:arrays={key:saved[key] for key in saved.files}
        np.testing.assert_array_equal(arrays['rows'],np.flatnonzero(evaluate))
        np.testing.assert_array_equal(arrays['train_rows'],np.flatnonzero(train))
        tick=time.time();prep=base.core.prepare(cell,train,reference)
        for key in ('mean','std','medians','fit_indices'):
            np.testing.assert_array_equal(getattr(prep,key),arrays[key])
        with fast.compatible_ari(reference) as acceleration:
            weights,details,failures,audit=joint.fit(prep,reference,cell=job['cell'],outer=job['outer'],inner=job['inner'])
        audit['runtime_acceleration']=acceleration
        added=base.core.score(prep,cell,weights,np.flatnonzero(evaluate))
        for key in ('rows','steps'):np.testing.assert_array_equal(added.pop(key),arrays[key])
        assert set(added).isdisjoint(arrays);arrays.update(added);del prep
        target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.npz.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        tmp.replace(target.with_suffix('.npz'))
        elapsed=time.time()-tick
        meta=dict(sm,manifest_sha256=mh,array_sha256=base.sha(target.with_suffix('.npz')),
            methods={**sm['methods'],**details},failures={**sm['failures'],**failures},
            joint_audit=audit,joint_fit_seconds=elapsed,source_control_seconds=sm['seconds'],seconds=sm['seconds']+elapsed,
            source_control_array_sha256=sm['array_sha256'],source_control_metadata_sha256=base.sha(source.with_suffix('.json')))
        base.save(target.with_suffix('.json'),meta);done+=1;gc.collect()
        print('Historical Joint',done,'/',len(manifest['jobs']),name,'seconds',round(meta['joint_fit_seconds'],1),flush=True)
        if time.time()-started>manifest['invocation_seconds_cap']:
            state('CHECKPOINTED_INVOCATION_CAP',completed=done,total=len(manifest['jobs']));return
    state('SCORING_COMPLETE',completed=done,total=len(manifest['jobs']),seconds=time.time()-started);verify()
    evaluator=module('joint_extension_evaluator','scripts/evaluate_historical_joint_refit_v3.py');evaluator.main()
    state('COMPLETE_REVIEWED_JOINT_EXTENSION',completed=done,total=len(manifest['jobs']))


if __name__=='__main__':
    try:run()
    except BaseException as error:
        state('FAILED',error=type(error).__name__+': '+str(error));raise
