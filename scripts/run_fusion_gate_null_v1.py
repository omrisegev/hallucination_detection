"""Frozen, resumable synthetic gate mechanism experiment; no benchmark labels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_gate_null_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils import fusion_gate_null as core


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')
    for attempt in range(8):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(min(.1*2**attempt,2))
def identity(r):return f'n{r["n"]}_rho{int(r["rho"]*100):02d}_jump{int(r["jump"])}_rep{r["replicate"]:02d}'
def verify():
    m=load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items():assert sha(p)==h,p
    return m


def prepare():
    if (OUT/'MANIFEST.json').exists():verify();print('Existing frozen manifest verified.');return
    spec=importlib.util.spec_from_file_location('null_tests',ROOT/'tests/test_fusion_gate_null.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(t))
    save(OUT/'TESTS.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,output=stream.getvalue()))
    if not result.wasSuccessful():raise RuntimeError(stream.getvalue())
    selected=[dict(n=n,rho=rho,jump=jump,replicate=rep) for n in (16,64,256) for rho,jump in [(0.,False),(.6,False),(.9,False),(0.,True)] for rep in range(64)]
    sources=[Path(__file__),Path(core.__file__),ROOT/'tests/test_fusion_gate_null.py',ROOT/'docs/experiments/FUSION_GATE_NULL_V1.md',
        ROOT/'spectral_utils/fused_trajectory_readouts.py',ROOT/'spectral_utils/fusion_gate_interface_audit.py',OUT/'TESTS.json',
        ROOT/'results/fusion_trajectory_imm_v1/EVALUATION.json',ROOT/'results/fusion_trajectory_imm_v1/REVIEW.json',ROOT/'results/fusion_trajectory_imm_v1/GATE_AUDIT.json']
    save(OUT/'MANIFEST.json',dict(status='FROZEN_BEFORE_SIMULATION',selected=selected,readouts=list(core.READOUTS),warm=core.WARM,
        labels_used=False,hashes={str(p):sha(p) for p in sources},python=sys.version))
    print('Tests PASS; frozen768 trials, five readouts. No benchmark predictions changed.',flush=True)


def worker(rec):
    uid=identity(rec);meta=OUT/'trials'/(uid+'.json');array=meta.with_suffix('.npz');h=sha(OUT/'MANIFEST.json')
    if meta.exists():
        old=load(meta);assert old['manifest_sha256']==h and old['array_sha256']==sha(array);return uid
    arrays,record=core.trial(**rec);array.parent.mkdir(parents=True,exist_ok=True)
    temporary=array.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
    temporary.replace(array);record.update(uid=uid,manifest_sha256=h,array_sha256=sha(array));save(meta,record);return uid


def run():
    m=verify();started=time.monotonic();futures=[];complete=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for rec in m['selected']:
            if time.monotonic()-started>600:raise TimeoutError('Submission cap; existing trial checkpoints retained')
            futures.append(pool.submit(worker,rec))
        for future in as_completed(futures):
            complete.append(future.result())
            if len(complete)%128==0:print('Complete',len(complete),'/768',flush=True)
    files={str(p):sha(p) for rec in m['selected'] for p in [OUT/'trials'/(identity(rec)+'.json'),OUT/'trials'/(identity(rec)+'.npz')]}
    save(OUT/'FROZEN.json',dict(status='COMPLETE',trials=len(complete),seconds=time.monotonic()-started,manifest_sha256=sha(OUT/'MANIFEST.json'),files=files))
    print('All768 trials complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','run'],required=True);args=parser.parse_args()
    prepare() if args.phase=='prepare' else run()
