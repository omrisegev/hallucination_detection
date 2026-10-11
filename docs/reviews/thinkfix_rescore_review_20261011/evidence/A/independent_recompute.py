"""Independent PRMBench metric audit: no project imports, summary files or reported metrics.

The exported scores are risk-oriented. Raw error_steps are 1-based. Per-answer AUC
is computed directly from every error/correct pair, with half credit for ties.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROSTER = ['ALL_EQ','DS_FILTER_EQ','LSML_CONT','IU_PCR','HEM_MOM_cc',
          'MEAN_MOM_cc','MEAN_MOM_ga','TCI_ONLY_ga']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def matrix_metrics(count):
    # Valid/correct is the positive class. Error/rejected is the negative class.
    tp, fp, tn, fn = map(int, count)
    fc = 2 * tp / (2 * tp + fp + fn)
    fe = 2 * tn / (2 * tn + fp + fn)
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,f1_correct=fc,f1_error=fe,
                pooled_macro_f1=(fc+fe)/2)

def confusion(error, flagged):
    return np.array([np.count_nonzero(~error & ~flagged),
                     np.count_nonzero(error & ~flagged),
                     np.count_nonzero(error & flagged),
                     np.count_nonzero(~error & flagged)], dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    args=ap.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=True)
    bp=args.input_root/'bank_N16.npz'; mp=args.input_root/'prmbench_meta.json'
    bank=np.load(bp,allow_pickle=False)
    metadata=json.loads(mp.read_text())['rows']
    meta={row['idx']:row for row in metadata}
    offsets=bank['offsets']; ids=bank['idx'].astype(str)
    assert len(meta)==len(metadata), 'duplicate metadata IDs'
    assert len(set(ids))==len(ids), 'duplicate bank IDs'
    assert offsets[0]==0 and np.all(np.diff(offsets)>0)
    assert len(offsets)==len(ids)+1
    assert set(ids)==set(meta), 'bank/metadata ID mismatch'
    counts=Counter(); error=np.zeros(offsets[-1],dtype=bool)
    control=np.zeros(len(ids),dtype=bool)
    single_class=[]; outofbounds=[]
    for i,(idx,start,end) in enumerate(zip(ids,offsets[:-1],offsets[1:])):
        row=meta[idx]; n=int(end-start)
        assert n==row['n_steps'],(idx,n,row['n_steps'])
        counts[row['classification']]+=1
        control[i]=row['classification']=='correct'
        e=np.isin(np.arange(1,n+1),row['error_steps'])
        error[start:end]=e
        if any(x<1 or x>n for x in row['error_steps']): outofbounds.append(idx)
        if not e.any() or e.all(): single_class.append(dict(idx=idx,n_steps=n,n_error=int(e.sum())))
    info=dict(input_bank_sha256=sha(bp),input_metadata_sha256=sha(mp),
              script_sha256=sha(Path(__file__)),n_answers=len(ids),n_steps=int(offsets[-1]),
              n_control_answers=int(control.sum()),n_control_steps=int(sum(np.diff(offsets)[control])),
              n_noncontrol_answers=int((~control).sum()),
              n_noncontrol_steps=int(sum(np.diff(offsets)[~control])),n_error_steps=int(error.sum()),
              n_correct_steps=int((~error).sum()),classes=dict(counts),
              n_rows_oob_error_annotation=len(outofbounds),outofbounds_rows=outofbounds,
              single_class_answers=single_class,roster=ROSTER,
              metric_contract='within-answer pairwise risk AUC, half credit for ties; valid class positive; synthetic correct controls excluded from pooled macro-F1')
    metrics={}; per_answer=[]; flatrows=[]
    for key in bank.files:
        if not key.startswith('S__'):continue
        name=key[3:]; s=bank[key]
        assert s.shape==error.shape and np.isfinite(s).all()
        q80=float(np.quantile(s,.8))
        auc_values=[]; top1_hits=[]; tied_answers=0; constant_answers=0
        totals={r:np.zeros(4,dtype=np.int64) for r in ['top1','top2','q80_gt','q80_ge']}
        classes_conf={r:defaultdict(lambda:np.zeros(4,dtype=np.int64)) for r in totals}
        clean_stats={r:dict(n_answers=0,n_false_flagged_answers=0,n_steps=0,n_flagged_steps=0) for r in totals}
        strict_control_stats={r:dict(n_answers=0,n_false_flagged_answers=0,n_steps=0,n_flagged_steps=0) for r in totals}
        for i,(idx,start,end) in enumerate(zip(ids,offsets[:-1],offsets[1:])):
            v=s[start:end]; e=error[start:end]; n=len(v)
            bad=v[e]; good=v[~e]
            av=None
            if len(bad) and len(good):
                pair=bad[:,None]-good[None,:]
                av=float((np.count_nonzero(pair>0)+.5*np.count_nonzero(pair==0))/pair.size)
                auc_values.append(av)
            order=np.argsort(-v,kind='stable')
            tied_answers+=int(np.count_nonzero(v==v.max())>1)
            constant_answers+=int(v.min()==v.max())
            if e.any():top1_hits.append(int(e[order[0]]))
            flags={'top1':np.isin(np.arange(n),order[:1]),'top2':np.isin(np.arange(n),order[:2]),
                   'q80_gt':v>q80,'q80_ge':v>=q80}
            for rule,f in flags.items():
                c=confusion(e,f)
                if not control[i]:
                    totals[rule]+=c;classes_conf[rule][meta[idx]['classification']]+=c
                for clean_condition, destination in [(not e.any(),clean_stats),(control[i],strict_control_stats)]:
                    if clean_condition:
                        dest=destination[rule]
                        dest['n_answers']+=1;dest['n_false_flagged_answers']+=int(f.any())
                        dest['n_steps']+=n;dest['n_flagged_steps']+=int(f.sum())
            per_answer.append(dict(scorer=name,idx=idx,n_steps=n,n_error=int(e.sum()),
                                   auc=av,top1_position_0based=int(order[0]),
                                   top1_hit=int(e[order[0]]) if e.any() else None,
                                   control=int(control[i])))
        rules={r:matrix_metrics(c) for r,c in totals.items()}
        for rule in totals:
            for dest in [clean_stats[rule],strict_control_stats[rule]]:
                dest['answer_false_flag_rate']=dest['n_false_flagged_answers']/dest['n_answers']
                dest['step_false_flag_rate']=dest['n_flagged_steps']/dest['n_steps']
        metrics[name]=dict(within_answer_auc=float(np.mean(auc_values)),auc_n_answers=len(auc_values),
                           top1_on_error=float(np.mean(top1_hits)),top1_error_answers=len(top1_hits),
                           top1_hits=int(sum(top1_hits)),q80_threshold=q80,
                           n_tied_maxima_answers=tied_answers,n_constant_score_answers=constant_answers,
                           rules=rules,all_error_free_answers=clean_stats,synthetic_controls=strict_control_stats,
                           by_class={r:{c:matrix_metrics(v) for c,v in cs.items()} for r,cs in classes_conf.items()})
        flatrows.append(dict(scorer=name,within_auc=metrics[name]['within_answer_auc'],
                             auc_answers=len(auc_values),top1_hits=sum(top1_hits),error_answers=len(top1_hits),
                             top1_on_error=metrics[name]['top1_on_error'],
                             top1_macro_f1=rules['top1']['pooled_macro_f1'],
                             q80_gt_macro_f1=rules['q80_gt']['pooled_macro_f1'],
                             q80_ge_macro_f1=rules['q80_ge']['pooled_macro_f1'],
                             q80_clean_flag_rate=clean_stats['q80_gt']['answer_false_flag_rate'],
                             q80_control_flag_rate=strict_control_stats['q80_gt']['answer_false_flag_rate']))
    info['eight_scorer_mean_within_auc']=float(np.mean([metrics[x]['within_answer_auc'] for x in ROSTER]))
    info['eight_scorer_mean_top1_macro_f1']=float(np.mean([metrics[x]['rules']['top1']['pooled_macro_f1'] for x in ROSTER]))
    info['eight_scorer_mean_q80_gt_macro_f1']=float(np.mean([metrics[x]['rules']['q80_gt']['pooled_macro_f1'] for x in ROSTER]))
    info['eight_scorer_mean_q80_gt_clean_false_flag_rate']=float(np.mean([metrics[x]['all_error_free_answers']['q80_gt']['answer_false_flag_rate'] for x in ROSTER]))
    info['eight_scorer_mean_q80_gt_synthetic_control_false_flag_rate']=float(np.mean([metrics[x]['synthetic_controls']['q80_gt']['answer_false_flag_rate'] for x in ROSTER]))
    (args.output_root/'audit.json').write_text(json.dumps(dict(integrity=info,metrics=metrics),indent=2))
    for fname,rows in [('metrics.csv',flatrows),('per_answer.csv',per_answer)]:
        with (args.output_root/fname).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in info.items() if k not in ['single_class_answers','outofbounds_rows']},indent=2))
    print((args.output_root/'metrics.csv').read_text())

if __name__=='__main__':main()
