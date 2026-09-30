"""Bounded fusion-quality comparison with frozen external historical anchors."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fusion_prediction_quality_v1'
PARENT=ROOT/'results/fusion_graph_conditioning_v1'
AUDIT=ROOT/'results/fusion_prediction_view_audit_v1'
ORIGINAL=ROOT/'results/fusion_replication_v1'
SOURCE=ROOT/'results/localization_source_group_audit_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_prediction_quality import ARMS,PARENT_ARMS,NEW_ARMS,KINDS,CORES,score_augmented
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals


def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


io=module(ROOT/'scripts/audit_fusion_prediction_view_v1.py','prediction_quality_io')
sha,load,save,safe=io.sha,io.load,io.save,io.safe
ANCHORS=dict(equal='dual__equal',iu='dual__iu',joint0='dual__cond100',graph010='dual__cond100_graph010',
    graph_perm='dual__cond100_graph_perm',equal_graph010='dual__equal_graph010',equal_graph_perm='dual__equal_graph_perm')


def registered_pairs():
    pairs=[]
    def add(l,r,scope='all'):pairs.append({'left':l,'right':r,'scope':scope})
    for kind in KINDS:
        for core in CORES:add(kind+'__'+core,ANCHORS[core])
        for left,right in [('iu','equal'),('joint0','iu'),('joint0','equal'),('graph010','joint0'),
            ('graph010','graph_perm'),('graph010','iu'),('graph010','equal_graph010'),
            ('equal_graph010','equal'),('equal_graph010','equal_graph_perm')]:add(kind+'__'+left,kind+'__'+right)
    for kind in ('last','ema32'):
        for core in CORES:add('ar1__'+core,kind+'__'+core)
    for kind in KINDS:
        add(kind+'__graph010',kind+'__iu','native:'+kind)
        add(kind+'__graph010',kind+'__equal_graph010','native:'+kind)
        add(kind+'__graph010',ANCHORS['graph010'],'native_and_original:'+kind)
        add(kind+'__joint0',ANCHORS['joint0'],'native_and_original:'+kind)
    assert len(pairs)==len({pair_key(p) for p in pairs})==74
    assert all(p['left'] in ARMS and p['right'] in ARMS for p in pairs)
    return pairs


def pair_key(p):return p['left']+' minus '+p['right']+' ['+p['scope']+']'


def select_rows(rows,scope):
    if scope=='all':return rows
    mode,kind=scope.split(':');assert mode in ('native','native_and_original') and kind in KINDS
    return [r for r in rows if r['augmentation']['native_joint_valid'][kind]
            and (mode!='native_and_original' or r['augmentation']['original_joint_valid'])]


def tests():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    command=[sys.executable,str(ROOT/'tests/test_fusion_prediction_quality.py')]
    result=subprocess.run(command,capture_output=True,timeout=180)
    p=OUT/'TESTS.txt';p.write_bytes(result.stdout+result.stderr)
    save(OUT/'TEST_EXECUTION.json',{'command':command,'exit_code':result.returncode,'test_count':3,
        'seconds':time.monotonic()-start,'output_sha256':sha(p),
        'source_hashes':{str(x):sha(x) for x in (ROOT/'tests/test_fusion_prediction_quality.py',ROOT/'spectral_utils/fusion_prediction_quality.py')}})
    print(p.read_text(encoding='utf-8'),flush=True)
    if result.returncode:raise RuntimeError('TEST_FAILURE')


def prepare():
    if (OUT/'MANIFEST.json').exists():verify();print('Existing manifest verified.');return
    io.verify();audit_review=load(AUDIT/'REVIEW.json');prior_review=load(PARENT/'REVIEW.json')
    assert audit_review['status']==prior_review['status']=='PASS'
    for path,h in {**audit_review['hashes'],**prior_review['hashes'],**prior_review['review_dependencies']}.items():assert sha(path)==h,path
    assert sha(ROOT/'scripts/review_fusion_graph_conditioning_v1.py')==prior_review['review_script_sha256']
    test=load(OUT/'TEST_EXECUTION.json');assert test['exit_code']==0 and test['test_count']==3
    for p,h in test['source_hashes'].items():assert sha(p)==h,p
    assert sha(OUT/'TESTS.txt')==test['output_sha256']
    parent=load(PARENT/'MANIFEST.json');audit=load(AUDIT/'MANIFEST.json')
    assert parent['selected']==audit['selected'] and len(ARMS)==98
    paths=[Path(__file__),ROOT/'scripts/audit_fusion_prediction_view_v1.py',ROOT/'scripts/run_answer_localization_v2.py',
        ROOT/'docs/experiments/FUSION_PREDICTION_QUALITY_V1.md',ROOT/'tests/test_fusion_prediction_quality.py',
        OUT/'TESTS.txt',OUT/'TEST_EXECUTION.json',SOURCE/'RELEASE_V2.json',
        PARENT/'MANIFEST.json',PARENT/'SCORES_FROZEN.json',PARENT/'EVALUATION.json',PARENT/'REVIEW.json',
        AUDIT/'MANIFEST.json',AUDIT/'AUDIT_FROZEN.json',AUDIT/'REVIEW.json',
        ORIGINAL/'MANIFEST.json',ORIGINAL/'SCORES_FROZEN.json']
    paths += [Path(obj.__file__).resolve() for name,obj in sys.modules.copy().items()
              if name.startswith('spectral_utils') and getattr(obj,'__file__',None)]
    hashes={str(p):sha(p) for p in paths}
    for directory,freeze in [(PARENT,'SCORES_FROZEN.json'),(ORIGINAL,'SCORES_FROZEN.json'),(AUDIT,'AUDIT_FROZEN.json')]:
        for path,h in load(directory/freeze)['files'].items():assert sha(path)==h,path;hashes[path]=h
    release=load(SOURCE/'RELEASE_V2.json')
    for cell in {r['cell'] for r in parent['selected']}:
        info=release['cells'][cell];assert sha(info['label_path'])==info['label_opaque_sha256'];hashes[info['label_path']]=info['label_opaque_sha256']
    save(OUT/'MANIFEST.json',{'release_id':parent['release_id'],'scoring_namespace':parent['scoring_namespace'],
        'selected':parent['selected'],'arms':ARMS,'new_arms':NEW_ARMS,'external_parent_arms':PARENT_ARMS,
        'contrasts':registered_pairs(),'hashes':hashes,'condition':100,'worker_cap':3,'seconds_cap':1200,
        'labels_decoded':False,'status':'DEVELOPMENT_PREDICTION_VIEW_FUSION_COMPARISON','created_unix':time.time()})
    print('Frozen 98 arms (77 exact references + 21 new), 74 comparisons.',flush=True)


def verify():
    m=load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items():assert sha(p)==h,p
    return m


def process_one(rec,digest,namespace):
    start=time.monotonic();uid=rec['uid'];mp=OUT/'scores'/(uid+'.json');ap=mp.with_suffix('.npz')
    if mp.exists():
        r=load(mp);assert r['manifest_sha256']==digest and r['array_sha256']==sha(ap);return
    original=load(ORIGINAL/'scores'/(uid+'.json'));audit=load(AUDIT/'answers'/(uid+'.json'))
    with np.load(ORIGINAL/'scores'/(uid+'.npz'),allow_pickle=False) as a:
        original['official_step_starts']=a['step_starts'];original['official_step_ends']=a['step_ends']
    with np.load(AUDIT/'answers'/(uid+'.npz'),allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    values,methods,diagnostics=score_augmented(arrays,audit,original,rec['tokens'],
        namespace+'/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8')
    ap.parent.mkdir(exist_ok=True,parents=True)
    with ap.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,**values)
    ap.with_suffix('.npz.tmp').replace(ap)
    save(mp,{**rec,'methods':methods,'diagnostics':diagnostics,'routing':original['routing'],
        'labels_decoded':False,'manifest_sha256':digest,'array_sha256':sha(ap),'seconds':time.monotonic()-start})


def scores():
    m=verify();digest=sha(OUT/'MANIFEST.json');start=time.monotonic();done=0;remaining=[]
    if (OUT/'SCORES_FROZEN.json').exists():
        f=load(OUT/'SCORES_FROZEN.json');assert f['manifest_sha256']==digest
        for p,h in f['files'].items():assert sha(p)==h,p
        print('Existing scores verified.');return
    for rec in m['selected']:
        p=OUT/'scores'/(rec['uid']+'.json')
        if p.exists():
            row=load(p);assert row['manifest_sha256']==digest and row['array_sha256']==sha(p.with_suffix('.npz'));done+=1
        else:remaining.append(rec)
    def state(status):save(OUT/'RUN_STATE.json',{'state':status,'pid':os.getpid(),'completed':done,
        'total':len(m['selected']),'seconds_this_invocation':time.monotonic()-start})
    state('RUNNING');i=0
    with ProcessPoolExecutor(max_workers=m['worker_cap']) as pool:
        active={}
        while active or i<len(remaining):
            while len(active)<m['worker_cap'] and i<len(remaining) and time.monotonic()-start<m['seconds_cap']:
                rec=remaining[i];i+=1;active[pool.submit(process_one,rec,digest,m['scoring_namespace'])]=rec['uid']
            if not active:break
            ready,_=wait(active,return_when=FIRST_COMPLETED)
            for future in ready:
                future.result();del active[future];done+=1
                if done%10==0 or done==len(m['selected']):print(done,'/',len(m['selected']),flush=True)
            state('RUNNING')
    if done!=len(m['selected']):state('PAUSED_AT_CAP');return
    verify();files=sorted((OUT/'scores').glob('*.json'))+sorted((OUT/'scores').glob('*.npz'));assert len(files)==220
    save(OUT/'SCORES_FROZEN.json',{'manifest_sha256':digest,'files':{str(p):sha(p) for p in files},
        'external_parent_freeze_sha256':sha(PARENT/'SCORES_FROZEN.json'),'labels_decoded':False,
        'seconds_this_invocation':time.monotonic()-start,'workers':m['worker_cap']})
    state('COMPLETE');print('All new scores frozen before evaluation.',flush=True)


def metric_module():return module(ROOT/'scripts/run_answer_localization_v2.py','prediction_quality_metrics')
def fixed_rows(rows):return [{**r,'decision_valid':r['fixed_iu_valid'],'predictions':r['fixed_iu_predictions']} for r in rows]


def evaluate():
    m=verify();f=load(OUT/'SCORES_FROZEN.json');assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and not f['labels_decoded']
    for p,h in f['files'].items():assert sha(p)==h,p
    previous=load(PARENT/'EVALUATION.json');rows=deepcopy(previous['rows']);by_uid={r['uid']:r for r in rows}
    release=load(SOURCE/'RELEASE_V2.json')
    for cell in sorted({r['cell'] for r in m['selected']}):
        with np.load(release['cells'][cell]['label_path'],allow_pickle=False) as lab:
            index={str(v):i for i,v in enumerate(lab['row_ids'])};assert len(index)==len(lab['row_ids'])
            for rec in (r for r in m['selected'] if r['cell']==cell):
                uid=rec['uid'];row=by_uid[uid];i=index[rec['row_id']]
                if cell.startswith('prm'):
                    a,b=lab['step_flag_offsets'][i:i+2];target=lab['step_error_flags'][a:b]
                else:target=int(lab['first_error'][i])
                np.testing.assert_array_equal(row['target'],target);assert row['group_id']==rec['group_id']
                meta=load(OUT/'scores'/(uid+'.json'));assert meta['routing']==row['routing']
                d=meta['diagnostics'];row['augmentation']={'bank':d['bank'],'original_joint_valid':d['original_joint_valid'],
                    'native_joint_valid':{kind:d['variants'][kind]['joint_valid'] for kind in KINDS}}
                with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as a:
                    for arm,detail in meta['methods'].items():
                        for key in ('valid','decision_valid','fixed_iu_valid'):row[key][arm]=detail[key]
                        for key,source in [('predictions','prediction'),('fixed_iu_predictions','fixed_iu_prediction'),('peaks','peak')]:row[key][arm]=detail.get(source)
                        row['sources'][arm]=detail['source_arm']
                        if detail['valid']:
                            risk=a[arm+'__risk'];assert risk.shape==(rec['steps'],) and np.isfinite(risk).all()
                            row['scores'][arm]=risk
    mm=metric_module();fixed=fixed_rows(rows)
    metrics={arm:{'prm':mm.prm_metric(rows,arm),'pb':mm.pb_metric(rows,arm),'pb_common_iu_gate':mm.pb_metric(fixed,arm)} for arm in ARMS}
    for arm in PARENT_ARMS:assert metrics[arm]==previous['metrics'][arm],arm
    save(OUT/'EVALUATION.json',{'release_id':m['release_id'],'status':'DEVELOPMENT_PREDICTION_VIEW_FUSION_COMPARISON',
        'scores_sha256':sha(OUT/'SCORES_FROZEN.json'),'rows':rows,'metrics':metrics,'parent_metrics':previous['metrics'],
        'older_58_answer_metrics':previous['older_58_answer_metrics'],'labels_decoded':True})
    for arm in NEW_ARMS:print(arm,'PRM',metrics[arm]['prm']['auroc'],'PB',metrics[arm]['pb']['macro_f1'],flush=True)


def contrasts():
    m=verify();e=load(OUT/'EVALUATION.json');p=OUT/'CONTRASTS.json';digest=sha(OUT/'EVALUATION.json');start=time.monotonic()
    state=load(p) if p.exists() else {'evaluation_sha256':digest,'pairs':{}};assert state['evaluation_sha256']==digest
    mm=metric_module()
    for pair in m['contrasts']:
        key=pair_key(pair)
        if key in state['pairs']:continue
        rows=select_rows(e['rows'],pair['scope']);fixed=fixed_rows(rows);left,right=pair['left'],pair['right']
        common=[r for r in rows if r['valid'][left] and r['valid'][right]]
        state['pairs'][key]={**pair,'selected_answers':len(rows),'selected_ids':[r['uid'] for r in rows],
            'left_prm':mm.prm_metric(common,left),'right_prm':mm.prm_metric(common,right),
            'left_pb':mm.pb_metric(rows,left),'right_pb':mm.pb_metric(rows,right),
            'left_pb_common_iu_gate':mm.pb_metric(fixed,left),'right_pb_common_iu_gate':mm.pb_metric(fixed,right),
            'uncertainty':paired_source_group_intervals(rows,left,right)}
        save(p,state)
    assert len(state['pairs'])==74;state.update(status='COMPLETE',seconds_this_invocation=time.monotonic()-start);save(p,state)
    print('All 74 comparisons complete.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=('tests','prepare','scores','evaluate','contrasts'));a=p.parse_args()
    {'tests':tests,'prepare':prepare,'scores':scores,'evaluate':evaluate,'contrasts':contrasts}[a.phase]()
