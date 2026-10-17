"""Reproduce inherited PRMB boundary overlaps without changing frozen spans."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer


def audit(source,out):
    bench=source/'results/localization_full_benchmark_v3/evaluation'
    rec=json.loads((bench/'JOINED.json').read_text())['records']
    a=np.load(bench/'JOINED.npz');offsets=a['offsets'];labels=a['labels']
    with (source/'dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl').open('rb') as f:p=pickle.load(f)
    rows=list(p.values()) if isinstance(p,dict) else p
    with (source/'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl').open('rb') as f:p=pickle.load(f)
    lab={str(r['idx']):r for r in (list(p.values()) if isinstance(p,dict) else p)}
    byid={r['row_id']:i for i,r in enumerate(rec) if r['cell']=='prmbench_qwen3_8b'}
    tokpath=next((source/'results/automatic_group_free_phase_a6_s0a_v1/inputs').glob('qwen3-8b*/tokenizer.json'))
    tok=Tokenizer.from_file(str(tokpath));overlaps=[]
    for r in rows:
        i=byid[str(r['idx'])];sp=np.asarray(r['step_token_spans'],int)
        expected=np.zeros(len(sp),int)
        for k in lab[str(r['idx'])]['error_steps']:
            if 1<=k<=len(sp):expected[k-1]=1
        np.testing.assert_array_equal(labels[offsets[i]:offsets[i+1]],expected)
        if np.any(sp[1:,0]<sp[:-1,1]):
            text='\n\n'.join(r['steps']);cs=[];pos=0
            for step in r['steps']:
                cs.append((pos,pos+len(step)));pos+=len(step)+2
            enc=tok.encode(text,add_special_tokens=False)
            np.testing.assert_array_equal(enc.ids,r['gen_token_ids'])
            rebuilt=[]
            for lo,hi in cs:
                ix=[k for k,(x,y) in enumerate(enc.offsets) if x<hi and y>lo and y>x]
                assert ix;rebuilt.append([ix[0],ix[-1]+1])
            np.testing.assert_array_equal(rebuilt,sp)
            overlaps.append({'row_id':str(r['idx']),'spans':sp.tolist(),'shared_boundaries':int((sp[1:,0]<sp[:-1,1]).sum()),
                             'tokenizer_replay':'EXACT','producer_diagnostics':r['align_diag']['problems']})
    assert len(overlaps)==3 and sum(r['shared_boundaries'] for r in overlaps)==10
    result={'status':'PASS','prmb_rows_checked':len(rows),'labels_match_corrected_v3':True,
            'overlap_rows':overlaps,'reason':'producer uses character intersection and assert_alignment(strict=False); frozen spans replay exactly',
            'original_assertion':'spans[1:,0] >= spans[:-1,1] was not a valid invariant of this existing benchmark'}
    out.mkdir(parents=True,exist_ok=True);(out/'SPAN_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='overlap_rows'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);a=ap.parse_args()
    audit(a.source_root,Path(__file__).resolve().parents[1]/'results/digit_alternative_probability_v1')
