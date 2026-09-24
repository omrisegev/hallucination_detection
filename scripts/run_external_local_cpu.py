"""Local CPU fallback runner, identical per-answer scorer and immutable records."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT'):os.environ[k]='1'
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.external_generalization.artifacts import atomic_json,file_hash,RecordStore
from spectral_utils.external_generalization.scoring import score_answer,ALL_ARMS


def one(args):
    cell,path,bundle=args
    d=json.loads(path.read_text());v=score_answer(d['payload']['telemetry'],bundle)
    v.update(telemetry_sha256=file_hash(path),arms=list(ALL_ARMS))
    return cell,d['uid'],v


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    out=ROOT/'results/lsml_external_generalization_v1/evaluation'
    source=ROOT/'scratch/external_generalization_private/evaluation_archives'
    bundle=json.loads((out/'source/BUNDLE.json').read_text());code=ROOT/'spectral_utils/external_generalization'
    identity={'bundle':file_hash(out/'source/BUNDLE.json'),'code':{str(f.relative_to(code)):file_hash(f) for f in sorted(code.rglob('*.py'))},'executor':'local_cpu'}
    cells=('hard2verify_qwen3_8b','socratic_qwen3_8b','socratic_qwq32b')
    stores={};jobs=[];started=time.perf_counter()
    try:
        for cell in cells:
            store=RecordStore(out/cell/'shard_000',identity);store.__enter__();stores[cell]=store
            files=sorted((source/cell/'records').glob('*.record.json'))
            if len(files)!=(200 if cell.startswith('hard') else 2995):raise ValueError('incomplete telemetry '+cell)
            if a.limit:files=files[:a.limit]
            for f in files:
                # Record filename is digest(uid), shared by input and output stores.
                if not (store.directory/f.name).exists():jobs.append((cell,f,bundle))
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for n,(cell,uid,payload) in enumerate(pool.map(one,jobs,chunksize=1)):
                stores[cell].put(uid,payload)
                if n%25==0:print(n+1,'/',len(jobs),cell,'elapsed',round(time.perf_counter()-started),flush=True)
        atomic_json(out/'CPU_EXECUTION.json',{'workers':a.workers,'answers_this_run':len(jobs),'elapsed':time.perf_counter()-started,'limit':a.limit,'identity':identity,'location':'local CPU fallback after SSH timeouts'})
    finally:
        for store in stores.values():store.__exit__(None,None,None)

if __name__=='__main__':main()
