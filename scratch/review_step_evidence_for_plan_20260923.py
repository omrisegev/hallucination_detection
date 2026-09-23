"""Read-only replay of Claude Step432 A1, for the research plan. No fits."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import softmax

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
OUT=ROOT/'scratch/step_evidence_plan_review_20260923'
OUT.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

summary=pd.read_csv(R/'SUMMARY.csv').set_index('method')
ans=pd.read_csv(R/'OOF_ANSWERS.csv')
pb=ans.cell.str.startswith('pb_').to_numpy()
target=ans.target.to_numpy()
gate=ans.ct7_gate.to_numpy().astype(bool)
scores=np.load(R/'OOF_STEP_SCORES.npz')
off=scores['offsets']; labels=scores['labels']
results={'scope':'read-only replay, no model refit or raw inference audit',
         'population':{'answers':len(ans),'pb':int(pb.sum()),'prm':int((~pb).sum()),
         'steps':int(off[-1]),'groups_crossing_folds':int((ans.groupby('source_group').fold.nunique()>1).sum())}}
pb_replay=[]
for name,row in summary.iterrows():
    pred=ans[name+'__mode'].to_numpy()
    slas=[];f1s=[]
    for cell in sorted(ans.cell[pb].unique()):
        mask=(ans.cell==cell).to_numpy()&np.isfinite(pred)
        e=mask&(target>=0);c=mask&(target<0)
        sla=(pred[e]==target[e]).mean()
        ca=(~gate[c]).mean();ea=((pred[e]==target[e])&gate[e]).mean()
        slas.append(sla);f1s.append(2*ca*ea/(ca+ea) if ca+ea else 0.)
    pb_replay.append({'method':name,'sla_error':abs(float(np.mean(slas))-row.pb_sla_macro8),
                      'f1_error':abs(float(np.mean(f1s))-row.pb_f1_ct7_gate),
                      'pb_covered':int((pb&np.isfinite(pred)).sum())})
results['pb_replay']={'methods':len(pb_replay),'max_sla_error':max(x['sla_error'] for x in pb_replay),
                      'max_f1_error':max(x['f1_error'] for x in pb_replay)}
results['pb_partial_historical_references']=[x for x in pb_replay if x['pb_covered']<6800]
selected=['evidence__all__seed__equal','evidence__all__plain__equal','evidence__all__position__equal',
          'evidence__all__plain2__equal','evidence__all__position2__equal',
          'evidence__all__ceiling__equal','evidence__all__plain__continuous_lsml',
          'evidence30__all__plain__equal','ct7','token_lsml']
prm_replay=[]
for name in selected:
    s=scores[name];aucs=[]
    for i in np.flatnonzero(~pb):
        a,b=off[i:i+2];y=labels[a:b].astype(bool)
        n1=int(y.sum());n0=len(y)-n1
        if n1 and n0:
            aucs.append(float((rankdata(s[a:b])[y].sum()-n1*(n1+1)/2)/(n1*n0)))
    prm_replay.append({'method':name,'eligible':len(aucs),'within':float(np.mean(aucs)),
                       'error':abs(float(np.mean(aucs))-summary.loc[name,'within_auc'])})
results['prm_replay']=prm_replay
jobs=[read(p) for p in (R/'jobs').glob('*.json')]
outer_overlap=0
for j in jobs:
    taskmask=(~pb) if j['task']=='prm' else (pb&ans.cell.str.endswith(j['task'][-2:]).to_numpy())
    heldgroups=set(ans.loc[taskmask&(ans.fold.to_numpy()==j['fold']),'source_group'])
    outer_overlap+=int(bool(heldgroups&set(j['train_source_groups'])))
results['jobs']={'outer':sum(j['inner_fold'] is None for j in jobs),
 'inner':sum(j['inner_fold'] is not None for j in jobs),
 'train_test_overlap':sum(bool(set(j['train_source_groups'])&set(j['test_source_groups'])) for j in jobs),
 'train_outer_held_overlap':outer_overlap,
 'prm_pseudo_rules':sorted(set(j.get('pseudo_positive_rule','missing') for j in jobs if j['task']=='prm')),
 'prm_min_positive_mass':min(j['pseudo_positive_total'] for j in jobs if j['task']=='prm')}
manifest=read(R/'REPORT_MANIFEST.json')
bad=[]
for section in ['jobs','outputs']:
    for rel,entry in manifest[section].items():
        p=R/rel.replace('\\','/')
        if not p.exists() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:
            detail={'section':section,'path':rel}
            if p.exists() and p.suffix=='.log' and p.stat().st_size>entry['bytes']:
                data=p.read_bytes()
                detail['recorded_prefix_hash_matches']=hashlib.sha256(data[:entry['bytes']]).hexdigest()==entry['sha256']
                detail['appended_text']=data[entry['bytes']:].decode('utf8',errors='replace')
            bad.append(detail)
results['manifest_replay']={'checked':sum(len(manifest[s]) for s in ['jobs','outputs']),'mismatches':bad,
 'summary_sha256':sha(R/'SUMMARY.csv'),'manifest_sha256':sha(R/'REPORT_MANIFEST.json')}
results['anchor_parity']=read(R/'ANCHOR_PARITY.json')
results['anchor_parity'].pop('rows',None)
profiles=np.load(R/'profiles_full.npy',mmap_mode='r')
raw_mismatch=0;normalized_changes=0;raw_scores_diff=0.
stored_seed_scores=scores['evidence__all__seed__equal']
stored_seed_modes=ans['evidence__all__seed__equal__mode'].to_numpy()
raw_pred=[];norm_pred=[]
for i in range(len(ans)):
    a,b=off[i:i+2];x=np.asarray(profiles[a:b,:,0])
    raw=softmax(x,axis=0).mean(1)
    sd=x.std(0);z=(x-x.mean(0))/np.where(sd>1e-8,sd,1.)
    normalized=softmax(z,axis=0).mean(1)
    def first(v):return int(np.flatnonzero(v>=v.max()-8*np.finfo(float).eps)[0])
    pr=first(raw);pn=first(normalized)
    raw_pred.append(pr);norm_pred.append(pn)
    raw_scores_diff=max(raw_scores_diff,float(np.max(np.abs(raw-stored_seed_scores[a:b]))))
    if pb[i]:
        raw_mismatch+=int(pr!=int(stored_seed_modes[i]))
        normalized_changes+=int(pr!=pn)
raw_pred=np.asarray(raw_pred);norm_pred=np.asarray(norm_pred)
def macro(pred):
    return float(np.mean([(pred[m]==target[m]).mean() for cell in sorted(ans.cell[pb].unique())
               for m in [(ans.cell==cell).to_numpy()&(target>=0)]]))
results['seed_scale_audit']={'raw_replay_pb_mode_mismatches':raw_mismatch,
 'raw_replay_max_score_error':raw_scores_diff,'pb_modes_changed_by_step_z':normalized_changes,
 'raw_pb_sla':macro(raw_pred),'step_z_pb_sla':macro(norm_pred),
 'interpretation':'step-z seed is a descriptive alternative, not a fitted new experiment'}
freeze=read(R/'RUN_FREEZE.json')
wt=R.parents[1]
paths=[wt/'scripts/experiments/step_evidence_v1.py',wt/'spectral_utils/step_evidence_v1.py']
results['source_name_collision']={'actual_hashes':{str(p.relative_to(wt)):sha(p) for p in paths},
 'stored_basename_hash':freeze.get('step_evidence_v1.py'),
 'snapshot_basename_hash':sha(R/'source_snapshot/step_evidence_v1.py')}
results['headline_rows']=summary.loc[selected,['pb_sla_macro8','within_auc','prmscore_q80','prmscore_inner']].reset_index().to_dict('records')
contrasts=pd.read_csv(R/'PAIRED_CONTRASTS.csv')
wanted=[('evidence__all__position__equal','evidence__all__plain__equal'),
 ('evidence__all__plain__continuous_lsml','evidence__all__plain__equal'),
 ('evidence__all__position2__equal','ct7'),
 ('evidence30__all__plain__equal','evidence30__all__seed__equal')]
results['selected_reported_contrasts']=contrasts[[((a,b) in wanted) for a,b in zip(contrasts.a,contrasts.b)]].to_dict('records')
results['multiplicity']={'tests':len(contrasts),'draws':10000,'minimum_reported_holm':float(contrasts.p_holm.min())}
(OUT/'AUDIT.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in results.items() if k not in ['anchor_parity','headline_rows','selected_reported_contrasts']},indent=2))
