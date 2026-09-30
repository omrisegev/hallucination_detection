"""Preserve historical predictions; repair all declared early score lanes."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.localization_history_bridge import corrected_rows,intervals
from spectral_utils.fusion_gate_interface_audit import comparison_parts

def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
io=module(ROOT/'scripts/audit_fusion_localization_forensics_v1.py','history_bridge_io')
load,save,sha=io.load,io.save,io.sha
OUT=ROOT/'results/localization_history_bridge_v3'
LABELS=ROOT/'results/localization_prm_label_audit_v1'
GROUPS=ROOT/'results/localization_source_group_audit_v1'
LANES=dict(representation='answer_localization_representation_pilot_v1',readout='fused_trajectory_readout_pilot_v1',
    sampling='fusion_window_sampling_pilot_v1',regularization='fusion_reliability_regularization_v1',context='fusion_context_bank_pilot_v1')
GATE=ROOT/'results/fusion_gate_interface_audit_v1'


def pairs(lane,path):
    e=load(path/'EVALUATION.json')
    if lane=='representation':return [dict(left=k.split('_minus_')[0],right=k.split('_minus_')[1],scope='all',original_key=k) for k in e['paired']]
    return [dict(left=p['left'],right=p['right'],scope='all',original_key=k) for k,p in load(path/'CONTRASTS.json')['pairs'].items()]


def prepare():
    assert not (OUT/'MANIFEST.json').exists()
    release=load(LABELS/'RELEASE_V3.json');paths=[Path(__file__),ROOT/'spectral_utils/localization_history_bridge.py',
        ROOT/'spectral_utils/fusion_benchmark_bootstrap.py',ROOT/'spectral_utils/fusion_gate_interface_audit.py',
        ROOT/'scripts/run_answer_localization_v2.py',ROOT/'scripts/audit_fusion_localization_forensics_v1.py',
        ROOT/'scripts/audit_localization_source_groups.py',ROOT/'spectral_utils/prmbench.py',
        ROOT/'docs/experiments/LOCALIZATION_HISTORY_BRIDGE_V3.md',LABELS/'RELEASE_V3.json',LABELS/'LABEL_AUDIT.json',LABELS/'REVIEW.json',
        LABELS/'original58_EVALUATION_V3.json',LABELS/'original58_CONTRASTS_V3.json',GROUPS/'CANONICAL_GROUPS.json',GROUPS/'FOLDS_V2.json',
        ROOT/'results/fusion_token_gap_v1/EVALUATION.json',ROOT/'results/fusion_token_gap_v1/REVIEW.json']
    cohort=set();lanes={}
    for lane,directory in LANES.items():
        p=ROOT/'results'/directory;e=load(p/'EVALUATION.json');ids={r['uid'] for r in e['rows']}
        if cohort:assert ids==cohort
        cohort=ids;assert len(ids)==58
        freeze=load(p/'SCORES_FROZEN.json');paths+=[p/'EVALUATION.json',p/'REVIEW.json',p/'SCORES_FROZEN.json']
        # Pin original manifests and verify their frozen scientific dependencies.
        mf=p/('PREPARED.json' if lane=='representation' else 'MANIFEST.json');paths.append(mf);source=load(mf)
        for file,digest in source.get('source_hashes',source.get('hashes',{})).items():
            assert sha(file)==digest,file;paths.append(Path(file))
        for file,digest in freeze['files'].items():assert sha(file)==digest,file;paths.append(Path(file))
        if (p/'CONTRASTS.json').exists():paths.append(p/'CONTRASTS.json')
        lanes[lane]={'directory':str(p),'arms':list(e['metrics']),'contrasts':pairs(lane,p)}
    for name in ('MANIFEST.json','EVALUATION.json','DIAGNOSTICS_FROZEN.json','REVIEW.json'):paths.append(GATE/name)
    for file,digest in load(GATE/'DIAGNOSTICS_FROZEN.json')['files'].items():assert sha(file)==digest;paths.append(Path(file))
    paths+=list(io.RAW_FILES.values())
    paths+=[Path(v['label_path']) for cell,v in release['cells'].items() if cell in {r['cell'] for r in e['rows']}]
    assert sum(len(x['arms']) for x in lanes.values())==131
    assert sum(len(x['contrasts']) for x in lanes.values())==199
    save(OUT/'MANIFEST.json',{'status':'HISTORICAL_SCORE_ONLY_LABEL_GROUP_BRIDGE','release_id':release['release_id'],
        'lanes':lanes,'selected_ids':sorted(cohort),'hashes':{str(p):sha(p) for p in paths},
        'source_scope':'Five early primary lanes plus gate diagnostics, original58 fallback anchors and current110 context; not all project history.',
        'new_model_fits':0,'new_predictions':0,'labels_previously_seen':True,'created_unix':time.time()})
    print('Frozen131 early entries,25 repaired fallback references,199 comparisons and7 gate diagnostics.',flush=True)


def verify():
    m=load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items():assert sha(p)==h,p
    return m


def indexes():
    audit=load(LABELS/'LABEL_AUDIT.json');labels={('prmbench_qwen3_8b',r['row_id']):{'flags':r['corrected_flags'],'previous_flags':r['previous_flags']} for r in audit['rows']}
    release=load(LABELS/'RELEASE_V3.json')
    for cell,info in release['cells'].items():
        if cell.startswith('pb') and cell in io.RAW_FILES:
            with np.load(info['label_path'],allow_pickle=False) as a:
                for rid,label in zip(a['row_ids'],a['first_error']):labels[(cell,str(rid))]={'label':int(label)}
    g=load(GROUPS/'CANONICAL_GROUPS.json')
    groups={('prmbench_qwen3_8b',r['row_id']):r['canonical_group_id'] for r in g['rows']}
    groups.update({('pb_'+r['subset']+'_q8',r['row_id']):r['canonical_group_id'] for r in g['pb_rows']})
    return labels,groups


def mm():return module(ROOT/'scripts/run_answer_localization_v2.py','history_metrics')
def fixed(rows):return [{**r,'decision_valid':r['fixed_iu_valid'],'predictions':r['fixed_iu_predictions']} for r in rows]


def bridge():
    m=verify();start=time.monotonic();labels,groups=indexes();metric=mm();registry=[];summaries={}
    for lane,info in m['lanes'].items():
        source=Path(info['directory'])/'EVALUATION.json';previous=load(source);rows=corrected_rows(previous['rows'],labels,groups);stats={}
        for arm,old in previous['metrics'].items():
            bundle={'prm':metric.prm_metric(rows,arm),'pb':metric.pb_metric(rows,arm)}
            # Original PB labels/predictions and complete point metric remain exact.
            assert bundle['pb']==old['pb'],(lane,arm)
            if 'pb_common_iu_gate' in old:
                bundle['pb_common_iu_gate']=metric.pb_metric(fixed(rows),arm);assert bundle['pb_common_iu_gate']==old['pb_common_iu_gate']
            stats[arm]=bundle;registry.append({'lane':lane,'arm':arm,'metric':bundle,'previous_metric':old,'cohort':'original58','diagnostic_only':False})
        changed=[r['uid'] for r,p in zip(rows,previous['rows']) if r['target']!=p['target']]
        save(OUT/(lane+'_EVALUATION_V3.json'),{'status':'SCORES_PREDICTIONS_FITS_UNCHANGED','release_id':m['release_id'],
            'source_sha256':sha(source),'manifest_sha256':sha(OUT/'MANIFEST.json'),'rows':rows,'metrics':stats,
            'previous_metrics':previous['metrics'],'changed_target_ids':changed,'group_map_sha256':sha(GROUPS/'CANONICAL_GROUPS.json')})
        summaries[lane]={'answers':len(rows),'arms':len(stats),'changed_prm_targets':len(changed),'changed_group_ids':sum(r['group_id']!=r['bridge_previous_group_id'] for r in rows)}
        print('Bridged',lane,summaries[lane],flush=True)
    anchor=load(LABELS/'original58_EVALUATION_V3.json')
    for arm,bundle in anchor['metrics'].items():registry.append({'lane':'fallback_references','arm':arm,'metric':bundle,'previous_metric':bundle,'cohort':'original58','already_corrected_reference':True,'diagnostic_only':False})
    assert len(registry)==156 and {r['uid'] for r in anchor['rows']}==set(m['selected_ids'])
    previous=load(GATE/'EVALUATION.json');rows=corrected_rows(previous['rows'],labels,groups);gs=deepcopy(previous['summaries'])
    for arm,summary in gs.items():
        selected=[r for r in rows if r['cell'].startswith('prm') and r['methods'][arm]['valid']]
        summary['prm']={kind:comparison_parts([r['target'] for r in selected],[r['methods'][arm][kind] for r in selected]) for kind in ('risk','origin_projection')}
        assert abs(summary['prm']['risk']['mean_within_answer_auc']-summary['prm']['origin_projection']['mean_within_answer_auc'])<1e-12
    save(OUT/'gate_DIAGNOSTICS_V3.json',{'status':'DIAGNOSTICS_NOT_NEW_METHODS','manifest_sha256':sha(OUT/'MANIFEST.json'),
        'source_sha256':sha(GATE/'EVALUATION.json'),'rows':rows,'summaries':gs,'previous_summaries':previous['summaries']})
    save(OUT/'REGISTRY.json',{'status':'CORRECTED_ORIGINAL58_EVIDENCE','entries':registry,'lane_counts':summaries,
        'current110_external_evaluation':str(ROOT/'results/fusion_token_gap_v1/EVALUATION.json'),
        'separate_current110_entries':107,'seconds':time.monotonic()-start})


def contrasts():
    m=verify();metric=mm();start=time.monotonic()
    for lane,info in m['lanes'].items():
        path=OUT/(lane+'_EVALUATION_V3.json');e=load(path);destination=OUT/(lane+'_CONTRASTS_V3.json');digest=sha(path)
        state=load(destination) if destination.exists() else {'evaluation_sha256':digest,'pairs':{}}
        assert state['evaluation_sha256']==digest
        for pair in info['contrasts']:
            key=pair['original_key']
            if key in state['pairs']:continue
            left,right=pair['left'],pair['right'];rows=e['rows'];common=[r for r in rows if r['valid'][left] and r['valid'][right]]
            point={**pair,'selected_ids':[r['uid'] for r in rows],'common_valid_ids':[r['uid'] for r in common],
                'left_prm':metric.prm_metric(common,left),'right_prm':metric.prm_metric(common,right),
                'left_pb':metric.pb_metric(rows,left),'right_pb':metric.pb_metric(rows,right),
                'uncertainty':intervals(rows,left,right,include_fixed=lane=='context')}
            if lane=='context':
                point.update(left_pb_common_iu_gate=metric.pb_metric(fixed(rows),left),right_pb_common_iu_gate=metric.pb_metric(fixed(rows),right))
            state['pairs'][key]=point;save(destination,state)
        assert len(state['pairs'])==len(info['contrasts']);state['status']='COMPLETE';save(destination,state)
        print('Paired bridge complete:',lane,len(state['pairs']),flush=True)
    save(OUT/'CONTRAST_EXECUTION.json',{'status':'COMPLETE','comparisons':199,'seconds':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['prepare','bridge','contrasts']);globals()[p.parse_args().phase]()
