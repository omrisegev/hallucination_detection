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
from spectral_utils.external_generalization.evaluation import confusion,metric,within_auc
from spectral_utils.external_bootstrap import bootstrap_all,contrast_from_draws
from spectral_utils.label_sanity import check_labels,feasibility_tag
CELLS={'hard2verify_qwen3_8b':'hard2verify','socratic_qwen3_8b':'socratic','socratic_qwq32b':'socratic'}
CONTRASTS=[(v+'_lsml',v+'_'+c) for v in ('frozen','local') for c in ('equal','partition_equal')]+[(v+'_lsml','ct7') for v in ('frozen','local')]


def load(p):return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--seal-only',action='store_true')
    p.add_argument('--draws',type=int,default=100000);a=p.parse_args()
    analysis_freeze=load(a.root/'ANALYSIS_FREEZE.json')
    for name,expected in analysis_freeze['files'].items():
        if file_hash(ROOT/name)!=expected:raise ValueError('analysis changed after pre-evaluation freeze: '+name)
    if a.draws!=analysis_freeze['draws']:raise ValueError('unregistered bootstrap draw count')
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
    freeze=load(a.root/'METHOD_FREEZE.json')
    frozen_code={str(Path(k).relative_to('spectral_utils/external_generalization')):v for k,v in freeze['code'].items()}
    if identity['code']!=frozen_code:raise ValueError('execution differs from frozen method code')
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
    atomic_json(a.root/'ALL_CELLS_SEALED.json',{'bundle_sha256':lock_hash,'analysis_freeze_sha256':file_hash(a.root/'ANALYSIS_FREEZE.json'),'seals':{c:load(a.root/c/'SEAL.json') for c in CELLS}})
    if a.seal_only:return
    # Only now open external annotations. The scorer has no API accepting them.
    gold={b:{g['uid']:g for g in load(a.inputs/'evaluator_only'/(b+'.json'))} for b in set(CELLS.values())}
    all_metrics={};all_contrasts=[];disjoint_contrasts=[]
    for cell,bench in CELLS.items():
        rows=all_predictions[cell];require_seal(rows,load(a.root/cell/'SEAL.json'),lock_hash)
        uids=[v.uid for v in answers[bench]];groups=[overlap['groups'][u] for u in uids]
        assert set(gold[bench])==set(uids)
        sanity=check_labels([y for uid in uids for y,keep in zip(gold[bench][uid]['correct'],gold[bench][uid]['include']) if keep])
        if not sanity.ok:raise ValueError(sanity.summary())
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
        changes={}
        for left,right in CONTRASTS:
            changed_steps=changed_answers=fallback_changed_steps=0
            for uid in uids:
                row=rows[uid];different=np.array(row['predictions'][left])!=np.array(row['predictions'][right])
                changed_steps+=int(different.sum());changed_answers+=bool(different.any())
                if left.startswith('local_') and not row['local']['native']:fallback_changed_steps+=int(different.sum())
            changes[left+'__vs__'+right]={'changed_steps':changed_steps,'changed_answers':changed_answers,'local_fallback_changed_steps':fallback_changed_steps}
        result={'n_checked':len(uids),'n_total':len(answers[bench]),'flag':sanity.flag_string(),'coverage_flag':feasibility_tag(len(uids),len(answers[bench])),'label_sanity':vars(sanity),'decision_changes':changes,'benchmark':bench,'answers':len(uids),'groups':len(set(groups)),
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
        names,boot=bootstrap_all(counts,groups,bench,draws=a.draws)
        np.savez_compressed(a.root/cell/"BOOTSTRAP.npz",arms=np.array(names),metrics=boot)
        for arm in ALL_ARMS:
            sample=boot[:,names.index(arm)];sample=sample[np.isfinite(sample)]
            result['arms'][arm]['ci95_descriptive']=np.quantile(sample,[.025,.975]).tolist()
            tp,fp,tn,fn=result['arms'][arm]['confusion']
            result['arms'][arm]['gold_error_fraction']=(tn+fp)/(tp+fp+tn+fn)
            result['arms'][arm]['predicted_error_fraction']=(tn+fn)/(tp+fp+tn+fn)
        atomic_json(a.root/cell/'METRICS.json',result)
        for left,right in CONTRASTS:
            comparison=contrast_from_draws(counts,names,boot,left,right,groups,bench,18)
            comparison.update(cell=cell,left=left,right=right);all_contrasts.append(comparison)
            print(cell,left,right,comparison['delta'],comparison['ci_bonferroni'],flush=True)
        # Pre-evaluation sensitivity panel; full-set eighteen-contrast family stays primary.
        disjoint_counts={k:np.asarray(v)[disjoint] for k,v in counts.items()}
        disjoint_groups=np.asarray(groups)[disjoint].tolist()
        if disjoint.all():dnames,dboot=names,boot
        else:dnames,dboot=bootstrap_all(disjoint_counts,disjoint_groups,bench,draws=a.draws)
        for left,right in CONTRASTS:
            comparison=contrast_from_draws(disjoint_counts,dnames,dboot,left,right,disjoint_groups,bench,18)
            comparison.update(cell=cell,left=left,right=right,panel='observed_disjoint_sensitivity')
            disjoint_contrasts.append(comparison)
        atomic_json(a.root/'DISJOINT_CONTRASTS.json',disjoint_contrasts)
        atomic_json(a.root/'CONTRASTS.json',all_contrasts)
    atomic_json(a.root/'METRICS.json',all_metrics)
    atomic_json(a.root/'EVALUATION_PROVENANCE.json',{'bundle':lock_hash,'script':file_hash(__file__),'bootstrap_helper':file_hash(ROOT/'spectral_utils/external_bootstrap.py'),
      'draws':a.draws,'seed':20260924,'family':18,'complete':len(all_contrasts)==18,
      'label_sanity_code':file_hash(ROOT/'spectral_utils/label_sanity.py'),'labels':{b:file_hash(a.inputs/'evaluator_only'/(b+'.json')) for b in gold}})

if __name__=='__main__':main()
