"""Version source groups/folds and bridge existing frozen answer-only scores."""
import os
for option in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[option]='1'
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/localization_source_group_audit_v1'
PARENT=ROOT/'results/fusion_explicit_fallback_pilot_v1'
RELEASE_ID='localization-cached-v2-sourcegroups-20260907'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def assign(groups,strata,namespace):
    by_stratum={}
    for g in groups:by_stratum.setdefault(strata[g],[]).append(g)
    out={}
    for s in sorted(by_stratum):
        ordered=sorted(by_stratum[s],key=lambda g:hashlib.sha256((namespace+'\0'+g).encode()).hexdigest())
        out.update({g:i%5 for i,g in enumerate(ordered)})
    return out


def bridge():
    started=time.monotonic();audit=load(OUT/'AUDIT.json');mapping=load(OUT/'CANONICAL_GROUPS.json')
    assert audit['audit_script_sha256']==sha(ROOT/'scripts/audit_localization_source_groups.py')
    assert audit['canonical_map_sha256']==sha(OUT/'CANONICAL_GROUPS.json')
    assert audit['pb_cross_subset_question_groups']==0
    old_release=ROOT/'results/answer_localization_representation_pilot_v1/RELEASE.json'
    source_files=[OUT/n for n in ('AUDIT.json','CANONICAL_GROUPS.json','QUESTION_METADATA.json','PB_QUESTION_METADATA.json')]+[
        PARENT/n for n in ('EVALUATION.json','CONTRASTS.json','ADDITIONAL_COMPARISONS.json','REVIEW.json','SCORES_FROZEN.json')]+[
        old_release,Path(__file__),ROOT/'scripts/audit_localization_source_groups.py',
        ROOT/'spectral_utils/fusion_benchmark_bootstrap.py',ROOT/'scripts/run_answer_localization_v2.py',
        ROOT/'docs/experiments/LOCALIZATION_SOURCE_GROUP_REPAIR_V2.md',ROOT/'tests/test_source_group_audit.py']
    hashes={str(p):sha(p) for p in source_files}
    freeze=OUT/'BRIDGE_MANIFEST.json'
    if freeze.exists():assert load(freeze)['hashes']==hashes,'FROZEN_BRIDGE_SOURCE_DRIFT'
    else:save(freeze,{'release_id':RELEASE_ID,'hashes':hashes,'kind':'RETROSPECTIVE_GROUPING_CORRECTION','created_unix':time.time()})
    by_id={r['row_id']:r['canonical_group_id'] for r in mapping['rows']+mapping['pb_rows']}
    assert len(by_id)==len(mapping['rows'])+len(mapping['pb_rows'])
    release=deepcopy(load(old_release));release.update(release_id=RELEASE_ID,predecessor_release_id=release['release_id'],
        predecessor_manifest_sha256=sha(old_release),grouping_version=mapping['grouping_version'],
        source_group_map_sha256=sha(OUT/'CANONICAL_GROUPS.json'),exposure='development_previously_evaluated_by_v2_not_untouched')
    memberships={}
    for cell,info in release['cells'].items():
        stratum='prmbench' if cell.startswith('prm') else cell.rsplit('_',1)[0]
        for row in info['rows']:
            row['legacy_group_id']=row['group_id'];row['group_id']=by_id[row['row_id']]
            memberships.setdefault(row['group_id'],set()).add(stratum)
    save(OUT/'RELEASE_V2.json',release)
    strata={g:'|'.join(sorted(v)) for g,v in memberships.items()}
    outer=assign(sorted(memberships),strata,RELEASE_ID)
    inner={str(k):assign([g for g in outer if outer[g]!=k],strata,RELEASE_ID+'/outer'+str(k)+'/inner') for k in range(5)}
    folds={'release_id':RELEASE_ID,'namespace':RELEASE_ID,'outer':outer,'inner':inner,
           'group_strata':strata,'source_release_sha256':sha(OUT/'RELEASE_V2.json'),
           'status':'FOLDS_PREPARED_TRAINED_METHODS_NOT_RERUN'}
    for task in ('prmbench','processbench'):
        groups={r['group_id'] for c,info in release['cells'].items() if (c.startswith('prm'))==(task=='prmbench') for r in info['rows']}
        folds[task]={'outer':{g:outer[g] for g in sorted(groups)},
                     'inner':{str(k):{g:inner[str(k)][g] for g in sorted(groups) if g in inner[str(k)]} for k in range(5)}}
    save(OUT/'FOLDS_V2.json',folds)
    previous=load(PARENT/'EVALUATION.json');corrected=deepcopy(previous)
    for row in corrected['rows']:
        row['legacy_group_id']=row['group_id'];row['group_id']=by_id[row['row_id']]
    spec=importlib.util.spec_from_file_location('bridge_metrics',ROOT/'scripts/run_answer_localization_v2.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    fixed=[{**r,'decision_valid':r['fixed_iu_valid'],'predictions':r['fixed_iu_predictions']} for r in corrected['rows']]
    recomputed={arm:{'prm':metrics.prm_metric(corrected['rows'],arm),'pb':metrics.pb_metric(corrected['rows'],arm),
        'pb_common_iu_gate':metrics.pb_metric(fixed,arm)} for arm in previous['metrics']}
    assert recomputed==previous['metrics'];corrected['metrics']=recomputed
    corrected.update(release_id=RELEASE_ID,status='RETROSPECTIVE_GROUPING_CORRECTION',
        previous_evaluation_sha256=sha(PARENT/'EVALUATION.json'),unchanged_metric_bundles=len(recomputed),
        grouping_manifest_sha256=sha(OUT/'RELEASE_V2.json'))
    save(OUT/'EVALUATION_V2.json',corrected)
    old_pairs={**load(PARENT/'CONTRASTS.json')['pairs'],**load(PARENT/'ADDITIONAL_COMPARISONS.json')['pairs']}
    target=OUT/'CONTRASTS_V2.json';state=load(target) if target.exists() else {'evaluation_sha256':sha(OUT/'EVALUATION_V2.json'),'pairs':{}}
    assert state['evaluation_sha256']==sha(OUT/'EVALUATION_V2.json')
    for key,pair in old_pairs.items():
        if key in state['pairs']:continue
        left,right=pair['left'],pair['right'];common=[r for r in corrected['rows'] if r['valid'][left] and r['valid'][right] and r['cell'].startswith('prm')]
        state['pairs'][key]={'left':left,'right':right,'prm_common_answers':len(common),
            'prm_common_groups':len({r['group_id'] for r in common}),
            'old_uncertainty':pair['uncertainty'],'corrected_uncertainty':paired_source_group_intervals(corrected['rows'],left,right)}
        save(target,state)
    assert len(state['pairs'])==32;state.update(status='COMPLETE',seconds=time.monotonic()-started);save(target,state)
    print('Corrected release and folds saved; 25 point-metric bundles unchanged; 32 corrected interval comparisons complete.',flush=True)


if __name__=='__main__':bridge()
