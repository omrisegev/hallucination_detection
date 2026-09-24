"""Seal all cells before joining evaluator-only labels; source-group uncertainty."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.external_generalization.artifacts import atomic_json,file_hash,seal_predictions,require_seal
from spectral_utils.external_generalization.contracts import digest,Answer,overlap_manifest,question_key
from spectral_utils.external_generalization.scoring import ALL_ARMS
from spectral_utils.external_generalization.evaluation import confusion,metric,within_auc,paired_bootstrap
CELLS={'hard2verify_qwen3_8b':'hard2verify','socratic_qwen3_8b':'socratic','socratic_qwq32b':'socratic'}
CONTRASTS=[(v+'_lsml',v+'_'+c) for v in ('frozen','local') for c in ('equal','partition_equal')]+[(v+'_lsml','ct7') for v in ('frozen','local')]


def load(p):return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--seal-only',action='store_true')
    p.add_argument('--draws',type=int,default=100000);a=p.parse_args()
    bundle=a.root/'source/BUNDLE.json';lock_hash=file_hash(bundle)
    answers={b:[Answer(**{**r,'steps':tuple(r['steps'])}) for r in load(a.inputs/b/'answers.json')] for b in set(CELLS.values())}
    overlap=overlap_manifest(answers)
    devgroups={}
    for filename in ('QUESTION_METADATA.json','PB_QUESTION_METADATA.json'):
        for row in load(ROOT/'results/localization_source_group_audit_v1'/filename)['rows']:
            devgroups[row['question_whitespace_sha256']]=filename
    excluded=set()
    for rows in answers.values():
        for row in rows:
            if any(question_key(t) in devgroups for t in (row.question,row.original_question)):
                excluded.add(overlap['groups'][row.uid])
    overlap['development_overlapping_groups']=sorted(excluded)
    overlap['development_check']='exact whitespace-normalized saved source question hashes; paraphrases not excluded'
    atomic_json(a.root/'OVERLAP.json',overlap)
    execution=load(a.root/'CPU_EXECUTION.json')
    identity=execution['identity']
    if identity['bundle']!=lock_hash:raise ValueError('execution bundle mismatch')
    code=ROOT/'spectral_utils/external_generalization'
    if identity['code']!={str(f.relative_to(code)):file_hash(f) for f in sorted(code.rglob('*.py'))}:raise ValueError('scoring code changed since execution')
    expected_identity=digest(identity)
    all_predictions={}
    for cell,bench in CELLS.items():
        rows={}
        for f in sorted((a.root/cell).glob('shard_*/*.record.json')):
            record=load(f);uid=record['uid']
            if record['run_identity']!=expected_identity or load(f.parent/'RUN.json')['identity']!=expected_identity:raise ValueError('mixed/stale prediction run identity')
            if uid in rows:raise ValueError('duplicate output '+uid)
            rows[uid]=record['payload']
        expected=[r.uid for r in answers[bench]]
        for answer in answers[bench]:
            row=rows[answer.uid]
            for arm in ALL_ARMS:
                if len(row['predictions'][arm])!=len(answer.steps):raise ValueError('unaligned predictions')
                if not set(row['predictions'][arm])<={0,1}:raise ValueError('nonbinary predictions')
        seal=seal_predictions(rows,expected,ALL_ARMS,a.root/cell/'SEAL.json',lock_hash)
        atomic_json(a.root/cell/'PREDICTIONS.json',rows)
        all_predictions[cell]=rows
    atomic_json(a.root/'ALL_CELLS_SEALED.json',{'bundle_sha256':lock_hash,'seals':{c:load(a.root/c/'SEAL.json') for c in CELLS}})
    if a.seal_only:return
    # Only now open external annotations. The scorer has no API accepting them.
    gold={b:{g['uid']:g for g in load(a.inputs/'evaluator_only'/(b+'.json'))} for b in set(CELLS.values())}
    all_metrics={};all_contrasts=[]
    for cell,bench in CELLS.items():
        rows=all_predictions[cell];require_seal(rows,load(a.root/cell/'SEAL.json'),lock_hash)
        uids=[v.uid for v in answers[bench]];groups=[overlap['groups'][u] for u in uids]
        assert set(gold[bench])==set(uids)
        counts={arm:[] for arm in ALL_ARMS};native_counts={arm:[] for arm in ALL_ARMS}
        auc={arm:[] for arm in ALL_ARMS};cats={};tokens=0;seconds=[];cpu=[];native=0;empty=0;degenerate=0;guarded=0;flagged=0
        disjoint=np.array([g not in excluded for g in groups])
        for uid in uids:
            g=gold[bench][uid];r=rows[uid];valid=np.asarray(r['nonempty'],bool)&np.asarray(g['include'],bool)
            native+=r['local']['native'];empty+=int(np.sum(~np.asarray(r['nonempty'],bool)))
            degenerate+=bool(r['local'].get('grouping_diag',{}).get('degenerate',False))
            guarded+=bool(r['local'].get('small_m_guarded',[]));flagged+=bool(r['local'].get('small_m_flags',[]))
            tokens+=r['tokens'];seconds.append(r['cpu_seconds']);cpu.append(r['process_cpu_seconds']);category=g['category']
            cats.setdefault(category,{arm:[] for arm in ALL_ARMS})
            for arm in ALL_ARMS:
                c=confusion(g['correct'],r['predictions'][arm],g['include']);counts[arm].append(c);cats[category][arm].append(c)
                nc=confusion(g['correct'],r['predictions'][arm],valid & bool(r['local']['native']))
                native_counts[arm].append(nc)
                s=np.asarray([np.nan if v is None else v for v in r['scores'][arm]])
                v=within_auc(g['correct'],s,valid)
                if v is not None:auc[arm].append(v)
        result={'benchmark':bench,'answers':len(uids),'groups':len(set(groups)),
          'steps':int(np.asarray(counts['ct7']).sum()),'tokens':tokens,'empty_steps':empty,'local_native_answers':native,'local_degenerate_answers':degenerate,'local_guarded_answers':guarded,'local_small_m_flagged_answers':flagged,
          'per_answer_elapsed_seconds_sum':sum(seconds),'process_cpu_seconds_sum':sum(cpu),'cpu_seconds_p50_p95':np.quantile(seconds,[.5,.95]).tolist(),
          'disjoint_answers':int(disjoint.sum()),'arms':{},'categories':{}}
        for arm in ALL_ARMS:
            c=np.asarray(counts[arm]);tp,fp,tn,fn=c.sum(0)
            result['arms'][arm]={'metric':float(metric(c.sum(0),bench)), 'confusion':c.sum(0).tolist(),
              'correct_recall':float(tp/(tp+fn)), 'error_recall':float(tn/(tn+fp)),
              'within_auc':float(np.mean(auc[arm])) if auc[arm] else None,'within_auc_answers':len(auc[arm]),
              'matched_native_metric':float(metric(np.asarray(native_counts[arm]).sum(0),bench)),
              'disjoint_metric':float(metric(c[disjoint].sum(0),bench)) if disjoint.any() else None}
        for category,cr in cats.items():result['categories'][category]={'answers':len(cr['ct7']),'arms':{k:float(metric(np.asarray(v).sum(0),bench)) for k,v in cr.items()}}
        np.savez_compressed(a.root/cell/'COUNTS.npz',**{k:np.asarray(v) for k,v in counts.items()},uids=np.array(uids),groups=np.array(groups))
        atomic_json(a.root/cell/'METRICS.json',result);all_metrics[cell]=result
        for left,right in CONTRASTS:
            comparison=paired_bootstrap(counts[left],counts[right],groups,bench,18,draws=a.draws)
            comparison.update(cell=cell,left=left,right=right);all_contrasts.append(comparison)
            print(cell,left,right,comparison['delta'],comparison['ci_bonferroni'],flush=True)
        atomic_json(a.root/'CONTRASTS.json',all_contrasts)
    atomic_json(a.root/'METRICS.json',all_metrics)
    atomic_json(a.root/'EVALUATION_PROVENANCE.json',{'bundle':lock_hash,'script':file_hash(__file__),
      'draws':a.draws,'seed':20260924,'family':18,'complete':len(all_contrasts)==18,
      'labels':{b:file_hash(a.inputs/'evaluator_only'/(b+'.json')) for b in gold}})

if __name__=='__main__':main()
