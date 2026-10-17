"""Replay seven locked methods on collected telemetry; accepts no label path."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT'): os.environ[k]='1'
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectral_utils.external_generalization.artifacts import RecordStore,atomic_json,file_hash
from spectral_utils.external_generalization.scoring import score_answer,ALL_ARMS
from spectral_utils.external_generalization.contracts import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--telemetry',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    files=sorted((a.telemetry/'records').glob('*.record.json'))
    if not files:raise ValueError('no telemetry records')
    files=files[a.shard::a.shards]
    if a.limit:files=files[:a.limit]
    bundle=json.loads(a.bundle.read_text());started=time.perf_counter()
    code=Path(__file__).resolve().parents[1]/'spectral_utils/external_generalization'
    identity={'bundle':file_hash(a.bundle),'telemetry':str(a.telemetry),'shard':a.shard,'shards':a.shards,
              'code':{str(f.relative_to(code)):file_hash(f) for f in sorted(code.rglob('*.py'))}}
    directory=a.out/('shard_%03d'%a.shard)
    with RecordStore(directory,identity) as store:
        for j,f in enumerate(files):
            original=json.loads(f.read_text());uid=original['uid']
            previous=store.get(uid)
            if previous is None:
                result=score_answer(original['payload']['telemetry'],bundle)
                result.update(telemetry_sha256=file_hash(f),arms=list(ALL_ARMS))
                store.put(uid,result)
            if j%25==0:print(j+1,'/',len(files),flush=True)
    atomic_json(directory/'DONE.json',{'answers':len(files),'elapsed':time.perf_counter()-started,
                'identity':digest(identity),'shard':a.shard,'limit':a.limit})

if __name__=='__main__':main()
