"""One user-requested matched Top8 readout alternative; no label API."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.digitfree_broad50 import token_bank,NAMES
from scripts.run_lsml_gate_locator_research_v1 import dump
from scripts.run_digitfree_broad50_v1 import sha
from scripts import run_direct_probability_temporal as e
from scripts.run_fusion_independence_atlas_v1 import RAW_SOURCE_HASHES
SOURCE=ROOT.parents[1];OUT=ROOT/'results/broad50_top8_v1'


def main():
    OUT.mkdir(exist_ok=True);folder=OUT/'extracted';folder.mkdir(exist_ok=True)
    contract=dict(readout='top8',features=list(NAMES),labels_used=False,
        source='same full13769 benchmark',hashes={str(p):sha(p) for p in
        (Path(__file__),ROOT/'spectral_utils/digitfree_broad50.py',ROOT/'results/digitfree_broad50_v1/MANIFEST.json')})
    mp=OUT/'EXTRACTION_MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=contract:raise ValueError('manifest drift')
    dump(mp,contract);e.old.configure_source_root(SOURCE)
    records=json.loads((SOURCE/'results/localization_full_benchmark_v3/evaluation/JOINED.json').read_text())['records']
    completed=0
    for cell,path,kind,dataset in e.source_specs():
        dest=folder/f'{cell}.npz';indexes=[i for i,r in enumerate(records) if r['cell']==cell]
        if dest.exists():completed+=len(indexes);continue
        digest=sha(path);assert digest==RAW_SOURCE_HASHES[path.relative_to(SOURCE).as_posix()]
        source=e.old._source_row_map(e.old.load_pickle(path),kind=kind,dataset=dataset);blocks=[];masks=[]
        for j,i in enumerate(indexes):
            row=source[records[i]['row_id']];payload=e.old._topk_payload(row)
            x,active,_=token_bank(payload['logprobs'],payload['ids'],row['gen_token_ids'],row['token_spilled_energies'])
            assert len(x)==records[i]['tokens']
            spans=np.asarray(row['step_token_spans'],int);assert spans.shape==(records[i]['steps'],2)
            block=np.full((len(spans),50),np.nan);mask=np.zeros(block.shape,bool)
            for s,(a,b) in enumerate(spans):
                for f in range(50):
                    v=x[a:b,f][active[a:b,f]];k=min(8,len(v))
                    if k:block[s,f]=np.partition(v,len(v)-k)[-k:].mean();mask[s,f]=True
            blocks.append(block);masks.append(mask)
            if (j+1)%500==0:
                dump(OUT/'RUN_STATE.json',dict(status='EXTRACTING',cell=cell,completed=completed+j+1,expected=13769))
                print('top8',completed+j+1,13769,flush=True)
        values=np.vstack(blocks).astype(np.float32);available=np.vstack(masks)
        with np.load(ROOT/f'results/digitfree_broad50_v1/extracted/{cell}.npz') as z:
            np.testing.assert_array_equal(indexes,z['indexes']);np.testing.assert_array_equal(available,z['available'])
            assert np.all(values[available]>=z['values'][available]-2e-5)
        np.savez_compressed(dest,indexes=indexes,values=values,available=available,source_sha256=digest)
        completed+=len(indexes);del source,blocks,masks
        print('top8 cell complete',cell,completed,flush=True)
    dump(OUT/'RUN_STATE.json',dict(status='EXTRACTION_COMPLETE',completed=completed,expected=13769))


if __name__=='__main__':main()
