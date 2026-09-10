"""Post-hoc score orientation diagnostic; never changes benchmark predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();out=ROOT/'results/surprisal_power_fusion_v1'
    bench=args.source_root/'results/localization_full_benchmark_v3/evaluation'
    z=np.load(out/'SCORES.npz');off=np.load(bench/'JOINED.npz')['offsets']
    rec=json.loads((bench/'JOINED.json').read_text(encoding='utf8'))['records']
    entropy=z['steps__ref__entropy'];result={}
    for method in ('d1__equal','d2__equal','d3__equal'):
        flat=z['steps__'+method]
        for task in ('pb','prmbench'):
            values=[]
            for i,row in enumerate(rec):
                if not row['cell'].startswith(task):continue
                s=flat[off[i]:off[i+1]];e=entropy[off[i]:off[i+1]]
                if len(s)<3 or s.std()<1e-12 or e.std()<1e-12:continue
                values.append(float(np.corrcoef(s,e)[0,1]))
            result[method+'__'+task]=dict(n=len(values),median_correlation=float(np.median(values)),
                negative_fraction=float(np.mean(np.array(values)<0)))
    report=dict(scope='Post-hoc orientation diagnostic against frozen entropy step scores; no label-based sign choice or changed predictions.',rows=result)
    (out/'ORIENTATION_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2),encoding='utf8')


if __name__=='__main__':main()
