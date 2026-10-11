"""Size-inspect, acquire and hash only explicitly enumerated Drive inputs."""
from pathlib import Path
import concurrent.futures, hashlib, json, subprocess, time
OUT=Path(__file__).resolve().parent
DEST=OUT/'inputs'
BASE='gdrive:hallucination_detection/cluster_results/thinkfix_rescore_v1'
BANKS=['N11','N13','N14','N16','N22','N25','N40','N43','N53','N56']
FILES=[]
for variant, folder in [('old','source_bank_old_rebuild'),('thinkfix','source_bank_thinkfix'),('thinkfix_mask1','source_bank_thinkfix_mask1')]:
    FILES.extend([(f'{BASE}/{folder}/FEATURES.npz',f'features_{variant}.npz'),(f'{BASE}/{folder}/MANIFEST.json',f'features_{variant}_MANIFEST.json')])
for b in BANKS: FILES.append((f'{BASE}/thinkfix_mask1/prmscore_inputs/bank_{b}.npz',f'bank_{b}.npz'))
for v in ['old','thinkfix']: FILES.append((f'{BASE}/{v}/prmscore_inputs/bank_N16.npz',f'bank_N16_{v}.npz'))
for v in ['OLD','THINKFIX','THINKFIX_MASK1']: FILES.append((f'{BASE}/STEP_TOKEN_SPANS_{v}.npz',f'step_spans_{v.lower()}.npz'))
TB='gdrive:hallucination_detection/cluster_results/prmbench_qwen3_8b_telemetry_thinkfix_mask1'
for f in ['prmbench_telemetry.pkl','SOURCE_manifest.json','MASK.json']: FILES.append((f'{TB}/{f}',f))
def call(args): return subprocess.check_output(['rclone',*args],text=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def get(item):
    src,name=item; p=DEST/name
    listing=json.loads(call(['lsjson',src,'--stat']))
    if isinstance(listing,list): listing=listing[0]
    if listing['IsDir']:raise ValueError(src)
    if not p.exists() or p.stat().st_size!=listing['Size']:
        subprocess.run(['rclone','copyto',src,str(p),'--retries','3','--stats','30s'],check=True)
    record={'source':src,'destination':name,'size':p.stat().st_size,'mod_time':listing['ModTime'],'sha256':sha(p),'inspected_size':listing['Size']}
    assert record['size']==listing['Size']
    (DEST/(name+'.receipt.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return record
if __name__=='__main__':
    DEST.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: receipts=list(pool.map(get,FILES))
    (OUT/'INPUT_MANIFEST.json').write_text(json.dumps({'files':receipts,'created_unix':time.time()},indent=2)+'\n')
    expected={'features_thinkfix_mask1.npz':'f14ab5d68360cd30819077051329678944bc631b94f2510bc67c9b697bf0948f','prmbench_telemetry.pkl':'684ce4c2a72d38e59a8841c8d0c69076f69a5adf707d5ed997b7c48b4bfba30f'}
    for r in receipts:
        if r['destination'] in expected: assert r['sha256']==expected[r['destination']],r
    print('ALL_INPUTS_ACQUIRED_AND_PRIMARY_HASHES_VERIFIED',flush=True)
