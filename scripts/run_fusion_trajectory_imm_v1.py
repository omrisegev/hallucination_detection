"""Bounded same-answer trajectory combination and supporting IMM."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import sys
import time
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_trajectory_imm import NEW_ARMS,PAIRS,SINGLES,READOUTS,SOURCES,score_trajectories
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals
OUT=ROOT/'results/fusion_trajectory_imm_v1';PARENT=ROOT/'results/fusion_sampling_replication_v1'
ORIGINAL=ROOT/'results/fusion_replication_v1';ANCHOR=ROOT/'results/fusion_graph_conditioning_v1';LABELS=ROOT/'results/localization_prm_label_audit_v1'
EVALUATION=PARENT/'EVALUATION.json'


def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


io_module=module(ROOT/'scripts/audit_fusion_localization_forensics_v1.py','trajectory_imm_io')
sha,load=io_module.sha,io_module.load


def retry(operation):
    for attempt in range(8):
        try:return operation()
        except PermissionError:
            if attempt==7:raise
            time.sleep(min(.1*2**attempt,2.))


def save(path,value):
    path=Path(path).resolve();assert path.is_relative_to(OUT.resolve())
    return retry(lambda:io_module.save(path,value))


def key(p):return p['left']+' minus '+p['right']+' ['+p['scope']+']'
def arm(family,readout):return f'traj_{family}__{readout}'


def pairs():
    out=[]
    def add(left,right,scope='all'):
        p=dict(left=left,right=right,scope=scope)
        if key(p) not in {key(q) for q in out}:out.append(p)
    for family in PAIRS:
        for a,b in [('imm','hold'),('gls','mean'),('hold','gls')]:add(arm(family,a),arm(family,b))
    primary='iu_joint_graph'
    for readout in READOUTS:
        for reference in ('dual__iu','dual__cond100_graph010','sample_risk_top__equal_graph_perm'):add(arm(primary,readout),reference)
        for family in ('iu_joint0','iu_joint_perm','equal_graph'):add(arm(primary,readout),arm(family,readout))
    for family in ('iu','joint_graph'):add(arm(primary,'imm'),arm(family,'imm'))
    for family in SINGLES:add(arm(family,'imm'),arm(family,'hold'))
    add(arm(primary,'imm'),arm(primary,'imm_permuted'));add(arm('equal_graph','imm'),arm('equal_perm','imm'))
    for readout in ('mean','imm'):
        for reference in ('dual__iu','dual__cond100_graph010'):add(arm(primary,readout),reference,'native_parent_joint')
    assert len(out)==50
    return out


def tests():
    suite=unittest.defaultTestLoader.loadTestsFromModule(module(ROOT/'tests/test_fusion_trajectory_imm.py','trajectory_imm_tests'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    save(OUT/'TESTS.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,output=stream.getvalue()))
    print(stream.getvalue());assert result.wasSuccessful()


def prepare():
    assert not (OUT/'MANIFEST.json').exists() and load(OUT/'TESTS.json')['passed']
    original=load(ORIGINAL/'MANIFEST.json');parent=load(PARENT/'MANIFEST.json');release=load(LABELS/'RELEASE_V3.json')
    assert len(parent['arms'])==149 and len(NEW_ARMS)==27
    paths=[Path(__file__),ROOT/'spectral_utils/fusion_trajectory_imm.py',ROOT/'tests/test_fusion_trajectory_imm.py',
        ROOT/'docs/experiments/FUSION_TRAJECTORY_IMM_V1.md',ROOT/'docs/reviews/trajectory_fusion_history_audit_2026-09-07.md',
        ROOT/'scripts/run_answer_localization_v2.py',ROOT/'scripts/audit_fusion_localization_forensics_v1.py',
        ORIGINAL/'MANIFEST.json',PARENT/'MANIFEST.json',PARENT/'REVIEW.json',EVALUATION,LABELS/'RELEASE_V3.json',LABELS/'REVIEW.json']
    for name,mod in list(sys.modules.items()):
        if name.startswith('spectral_utils') and getattr(mod,'__file__',None):paths.append(Path(mod.__file__).resolve())
    for rec in original['selected']:
        uid=rec['uid'];paths.extend([ORIGINAL/'inputs'/(uid+'.npz'),ORIGINAL/'scores'/(uid+'.json'),ANCHOR/'scores'/(uid+'.json'),ANCHOR/'scores'/(uid+'.npz')])
    paths += [Path(info['label_path']) for cell,info in release['cells'].items() if cell in {r['cell'] for r in original['selected']}]
    paths += list(io_module.RAW_FILES.values())
    save(OUT/'MANIFEST.json',dict(status='DEVELOPMENT_FUSED_TRAJECTORY_IMM',release_id=release['release_id'],
        scoring_namespace=original['scoring_namespace'],selected=original['selected'],arms=parent['arms']+list(NEW_ARMS),
        new_arms=list(NEW_ARMS),external_arms=parent['arms'],contrasts=pairs(),hashes={str(p):sha(p) for p in paths},
        labels_decoded_for_scoring=False,created_unix=time.time(),workers=3,score_seconds_cap=600))
    print('Frozen176 entries,27 new outputs and50 paired comparisons.',flush=True)


def verify():
    m=load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items():assert sha(p)==h,p
    return m


def one(rec,digest,namespace):
    start=time.monotonic();uid=rec['uid'];jp=OUT/'scores'/(uid+'.json');npz=OUT/'scores'/(uid+'.npz')
    if jp.exists():
        old=load(jp);assert old['manifest_sha256']==digest and old['array_sha256']==sha(npz);return uid
    with np.load(ORIGINAL/'inputs'/(uid+'.npz'),allow_pickle=False) as a:ss,ee=a['step_starts'],a['step_ends']
    with np.load(ANCHOR/'scores'/(uid+'.npz'),allow_pickle=False) as a:source={k:a[k] for k in a.files if any(k==name+'__window' for name in SOURCES.values())}
    meta=load(ANCHOR/'scores'/(uid+'.json'));original=load(ORIGINAL/'scores'/(uid+'.json'))
    identity=namespace+'/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8'
    arrays,methods,diagnostics=score_trajectories(rec['tokens'],ss,ee,source,meta,original,identity)
    npz.parent.mkdir(parents=True,exist_ok=True);tmp=npz.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    retry(lambda:tmp.replace(npz))
    save(jp,{**rec,'methods':methods,'diagnostics':diagnostics,'routing':original['routing'],
        'original_joint_valid':original['routing']['routes']['dual']!='moment_iu','labels_used':False,
        'array_sha256':sha(npz),'manifest_sha256':digest,'seconds':time.monotonic()-start})
    return uid


def scores():
    m=verify();started=time.monotonic();digest=sha(OUT/'MANIFEST.json');remaining=iter(m['selected']);completed=[]
    with ProcessPoolExecutor(max_workers=m['workers']) as executor:
        pending={}
        def submit():
            try:rec=next(remaining)
            except StopIteration:return
            pending[executor.submit(one,rec,digest,m['scoring_namespace'])]=rec['uid']
        for _ in range(m['workers']):submit()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                completed.append(future.result());pending.pop(future)
                if time.monotonic()-started<m['score_seconds_cap']:submit()
            if len(completed)%10==0:print('Completed',len(completed),'/110',flush=True)
    assert len(completed)==110,'Submission cap reached; verified checkpoints retained.'
    paths=[OUT/'scores'/(r['uid']+ext) for r in m['selected'] for ext in ('.json','.npz')]
    save(OUT/'SCORES_FROZEN.json',dict(status='COMPLETE',files={str(p):sha(p) for p in paths},manifest_sha256=digest,labels_decoded=False,seconds=time.monotonic()-started))
    print('All27 outputs frozen for110 answers.',flush=True)


def metric_module():return module(ROOT/'scripts/run_answer_localization_v2.py','trajectory_metric_reference')
def fixed(rows):return [{**r,'predictions':r['fixed_iu_predictions'],'decision_valid':r['fixed_iu_valid']} for r in rows]


def evaluate():
    m=verify();freeze=load(OUT/'SCORES_FROZEN.json');assert freeze['status']=='COMPLETE' and not freeze['labels_decoded']
    assert freeze['manifest_sha256']==sha(OUT/'MANIFEST.json')
    for p,h in freeze['files'].items():assert sha(p)==h,p
    previous=load(EVALUATION);rows=deepcopy(previous['rows']);release=load(LABELS/'RELEASE_V3.json')
    for cell in {r['cell'] for r in rows}:
        with np.load(release['cells'][cell]['label_path'],allow_pickle=False) as labels:
            index={str(v):i for i,v in enumerate(labels['row_ids'])}
            for row in (r for r in rows if r['cell']==cell):
                i=index[row['row_id']]
                if cell.startswith('prm'):
                    a,b=labels['step_flag_offsets'][i:i+2];np.testing.assert_array_equal(row['target'],labels['step_error_flags'][a:b])
                else:assert row['target']==int(labels['first_error'][i])
                meta=load(OUT/'scores'/(row['uid']+'.json'));assert meta['routing']==row['routing']
                row['trajectory_imm']={'original_joint_valid':meta['original_joint_valid']}
                with np.load(OUT/'scores'/(row['uid']+'.npz'),allow_pickle=False) as a:
                    for name,detail in meta['methods'].items():
                        for k in ('valid','decision_valid','fixed_iu_valid'):row[k][name]=detail[k]
                        for dst,src in [('predictions','prediction'),('fixed_iu_predictions','fixed_iu_prediction'),('peaks','peak')]:row[dst][name]=detail.get(src)
                        row['sources'][name]=detail.get('source_arm')
                        if detail['valid']:row['scores'][name]=a[name+'__risk'].tolist()
    mm=metric_module();fx=fixed(rows);metrics={name:dict(prm=mm.prm_metric(rows,name),pb=mm.pb_metric(rows,name),pb_common_iu_gate=mm.pb_metric(fx,name)) for name in m['arms']}
    for name in m['external_arms']:assert metrics[name]==previous['metrics'][name],name
    save(OUT/'EVALUATION.json',dict(status='DEVELOPMENT_TRAJECTORY_IMM',release_id=m['release_id'],scores_sha256=sha(OUT/'SCORES_FROZEN.json'),rows=rows,metrics=metrics))
    for name in NEW_ARMS:print(name,metrics[name]['prm']['auroc'],metrics[name]['pb']['macro_f1'],flush=True)


def select(rows,scope):
    if scope=='all':return rows
    assert scope=='native_parent_joint';return [r for r in rows if r['trajectory_imm']['original_joint_valid']]


def contrasts():
    m=verify();e=load(OUT/'EVALUATION.json');path=OUT/'CONTRASTS.json';digest=sha(OUT/'EVALUATION.json');start=time.monotonic();mm=metric_module()
    state=load(path) if path.exists() else dict(evaluation_sha256=digest,pairs={});assert state['evaluation_sha256']==digest
    for p in m['contrasts']:
        k=key(p)
        if k in state['pairs']:continue
        rows=select(e['rows'],p['scope']);common=[r for r in rows if r['valid'][p['left']] and r['valid'][p['right']]]
        state['pairs'][k]={**p,'selected_ids':[r['uid'] for r in rows],'left_prm':mm.prm_metric(common,p['left']),'right_prm':mm.prm_metric(common,p['right']),
            'left_pb':mm.pb_metric(rows,p['left']),'right_pb':mm.pb_metric(rows,p['right']),
            'uncertainty':paired_source_group_intervals(rows,p['left'],p['right'])};save(path,state)
    assert len(state['pairs'])==50;state.update(status='COMPLETE',seconds=time.monotonic()-start);save(path,state)
    print('All50 comparisons complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['tests','prepare','scores','evaluate','contrasts']);globals()[parser.parse_args().phase]()
