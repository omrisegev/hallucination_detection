"""Run the frozen independent audit with two immutable NPZ files cached in RAM.

Only I/O changes: the original audit repeatedly decompresses full saved scores
inside its per-answer loop. No arithmetic, test or tolerance is changed.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import verify_rbm_data_diagnostics as audit

original_load=np.load
cache={}


def cached_load(file,*args,**kwargs):
    if isinstance(file,(str,Path)) and Path(file).name in ('SCORES.npz','JOINED.npz'):
        key=str(Path(file).resolve())
        if key not in cache:
            with original_load(file,*args,**kwargs) as z:
                cache[key]={name:z[name] for name in z.files}
        return cache[key]
    return original_load(file,*args,**kwargs)


if __name__=='__main__':
    np.load=cached_load
    try:audit.main()
    finally:np.load=original_load
    p=ROOT/'results/rbm_data_diagnostics_v1/DIAGNOSTIC_REVIEW.json'
    d=json.loads(p.read_text())
    d['io_cache_wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    d['io_cache_note']='Immutable full score/joined arrays loaded once; frozen audit arithmetic unchanged.'
    p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
