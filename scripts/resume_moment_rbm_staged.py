"""Windows JSON-sharing retry wrapper; frozen scorer/checkpoint manifests unchanged."""
import json,os,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_moment_rbm_staged as run


def atomic_json_retry(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(run.base.dumps(payload)+'\n',encoding='utf8')
    for attempt in range(40):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)


if __name__=='__main__':
    run.base.atomic_json=atomic_json_retry
    atomic_json_retry(run.STAGED_OUT/'IO_RESUME.json',dict(wrapper=str(Path(__file__)),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        change='Retry transient Windows sharing violations on JSON replace; no fit/evaluation changes'))
    try:run.main()
    except BaseException as e:
        atomic_json_retry(run.STAGED_OUT/'RUN_STATE.json',dict(status='FAILED',error=f'{type(e).__name__}: {e}'));raise
