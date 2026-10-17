"""Export all fitted thresholds/weights and diagnose vote/step resolution."""
import argparse
import collections
import gzip
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_direct_probability_temporal import atomic_json
from spectral_utils.binary_moment_fusion import METHODS


def describe(values):
    a=np.asarray(values,float)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),
                mean=float(a.mean()),max=float(a.max())) if len(a) else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    out=ROOT/'results/binary_moment_fusion_v1'
    records=json.loads((a.source_root/'results/localization_full_benchmark_v3/evaluation/JOINED.json').read_text())['records']
    joined=np.load(a.source_root/'results/localization_full_benchmark_v3/evaluation/JOINED.npz')
    offsets, targets = joined['offsets'], joined['target']
    saved=np.load(out/'SCORES.npz');metrics=json.loads((out/'METRICS.json').read_text())
    con=sqlite3.connect((out/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    if con.execute('SELECT count(*) FROM answers').fetchone()[0]!=13769:raise ValueError('incomplete checkpoint')
    stats={m:dict(distinct=[],active=[],flips=[],groups=[],degenerate=0,constant_virtual=0,constant_score=0,
                  thresholds=collections.defaultdict(list),negative_share=[],failure=0) for m in METHODS}
    W=[];uids=[];methods=list(METHODS)
    with gzip.open(out/'ANSWER_FITS.jsonl.gz','wt',encoding='utf8',compresslevel=1) as f:
        for i,blob,info_json in con.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
            info=json.loads(info_json);assert info['uid']==records[i]['uid'];f.write(info_json+'\n')
            assert info['raw50_max_error']==0
            with np.load(io.BytesIO(blob)) as z:
                w=z['weights'].copy();W.append(w);uids.append(info['uid'])
                for j,m in enumerate(methods):
                    st=stats[m]
                    if m in info['failures']:st['failure']+=1;continue
                    d=info['diagnostics'][m];st['distinct'].append(d['distinct_votes']);st['active'].append(d['active_columns'])
                    st['flips'].append(sum(v<0 for v in d['signs']))
                    st['constant_score']+=d['constant_score']
                    for col,v in zip(d['threshold_columns'],d['signed_thresholds']):st['thresholds'][str(col)].append(v)
                    if m.endswith('__lsml'):
                        st['groups'].append(d['group_count']);st['degenerate']+=d['grouping']['degenerate']
                        st['constant_virtual']+=d['constant_virtual']>0
                    else:
                        vals=w[j][np.isfinite(w[j])];total=np.abs(vals).sum()
                        st['negative_share'].append(float(-np.minimum(vals,0).sum()/total) if total else 0)
    diagnostics={}
    for m,st in stats.items():
        result={k:describe(st[k]) for k in ('distinct','active','flips','groups','negative_share')}
        result.update({k:st[k] for k in ('failure','degenerate','constant_virtual','constant_score')})
        result['signed_threshold_summary']={k:describe(v) for k,v in st['thresholds'].items()}
        result['group_count_histogram']=dict(collections.Counter(st['groups']))
        result['step_resolution']={}
        for panel in ('pb','prm'):
            ties=[];first=0;n=0;all_tied=0;raw_exact=0;true_step_in_tie=0;unique_scores=[]
            flat=saved['steps__'+m]
            for i,r in enumerate(records):
                if (panel=='pb') != r['cell'].startswith('pb_'):continue
                s=flat[offsets[i]:offsets[i+1]]
                if not len(s) or not np.isfinite(s).all():continue
                peak=int(np.argmax(s));nt=int(np.sum(s==s[peak]));ties.append(nt)
                unique_scores.append(len(np.unique(s)))
                first+=peak==0;n+=1;all_tied+=nt==len(s)
                if panel=='pb' and targets[i]>=0:
                    raw_exact+=peak==targets[i]
                    true_step_in_tie+=s[int(targets[i])]==s[peak] and peak!=targets[i]
            result['step_resolution'][panel]=dict(valid=n,first_step_peaks=int(first),all_steps_tied=int(all_tied),
                max_tie_count=describe(ties),answers_with_peak_tie=sum(x>1 for x in ties),raw_exact=int(raw_exact),
                unique_step_scores=describe(unique_scores),missed_true_step_tied_at_max=int(true_step_in_tie))
        diagnostics[m]=result
    np.savez_compressed(out/'COEFFICIENTS.npz',uids=np.array(uids),methods=np.array(methods),weights=np.stack(W))
    atomic_json(out/'DIAGNOSTICS.json',dict(methods=diagnostics,
        note='Thresholds differ by answer, no correctness-label tuning. Continuous weights apply to standardized raw columns; binary weights to oriented votes. L-SML has within/cross weights in ANSWER_FITS.jsonl.gz, not a single linear vector. Ties counted with exact frozen equality.'))
    print('Exported13769 fitted-answer records, thresholds, group weights and step-resolution diagnostics.')


if __name__=='__main__':main()
