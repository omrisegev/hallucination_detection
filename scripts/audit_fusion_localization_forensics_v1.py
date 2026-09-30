"""Step313: inspect frozen localization failures against original text/telemetry."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np
from scipy.stats import spearmanr
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fusion_localization_forensics_v1'
PARENT=ROOT/'results/fusion_prediction_quality_v1'
GRAPH=ROOT/'results/fusion_graph_conditioning_v1'
ORIGINAL=ROOT/'results/fusion_replication_v1'
SOURCE=ROOT/'results/localization_source_group_audit_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.localization_forensics import character_alignment,peak_geometry,pb_outcome

OLD_ARMS=['dual__equal','dual__iu','dual__cond100','dual__cond100_graph010','dual__cond100_graph_perm',
          'dual__equal_graph010','dual__equal_graph_perm','moment__iu','context__iu','context__equal']
NEW_ARMS=[kind+'__'+core for kind in ('ar1','last','ema32') for core in ('iu','graph010')]
ARMS=OLD_ARMS+NEW_ARMS
COMPARISONS=[(kind+'__'+core,'dual__iu' if core=='iu' else 'dual__cond100_graph010')
             for kind in ('ar1','last','ema32') for core in ('iu','graph010')]
RAW_FILES={'prmbench_qwen3_8b':ROOT/'dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl'}
RAW_FILES.update({'pb_'+subset+'_q8':ROOT/'dataset_cache/repgrid/pb_qwen3_8b'/('processbench_'+subset+'.pkl')
                  for subset in ('gsm8k','math','olympiadbench','omnimath')})


def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


metadata=module(ROOT/'scripts/audit_localization_source_groups.py','forensics_metadata_reader')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(v):
    if isinstance(v,np.ndarray):return safe(v.tolist())
    if isinstance(v,np.generic):return safe(v.item())
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [safe(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(safe(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def tests():
    import io
    suite=unittest.defaultTestLoader.loadTestsFromModule(module(ROOT/'tests/test_localization_forensics.py','forensics_tests'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    save(OUT/'TESTS.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'output':stream.getvalue()})
    print(stream.getvalue(),flush=True)
    assert result.wasSuccessful()


def prepare():
    assert not (OUT/'MANIFEST.json').exists(),'Use existing freeze; do not silently overwrite.'
    assert load(OUT/'TESTS.json')['passed']
    parent=load(PARENT/'MANIFEST.json')
    tp=Path(r'C:\Users\DELL\.cache\huggingface\hub\models--Qwen--Qwen3-8B\snapshots\b968826d9c46dd6066d109eabc6255188de91218\tokenizer.json')
    paths=[Path(__file__),ROOT/'spectral_utils/localization_forensics.py',ROOT/'tests/test_localization_forensics.py',
           ROOT/'docs/experiments/FUSION_LOCALIZATION_FORENSICS_V1.md',ROOT/'scripts/audit_localization_source_groups.py',
           ROOT/'spectral_utils/processbench.py',ROOT/'scripts/run_answer_localization_v2.py',
           SOURCE/'RELEASE_V2.json',SOURCE/'CANONICAL_GROUPS.json',PARENT/'EVALUATION.json',
           PARENT/'MANIFEST.json',PARENT/'SCORES_FROZEN.json',PARENT/'REVIEW.json',tp]+list(RAW_FILES.values())
    release=load(SOURCE/'RELEASE_V2.json')
    paths += [Path(v['label_path']) for cell,v in release['cells'].items() if cell in RAW_FILES]
    for rec in parent['selected']:
        uid=rec['uid'];paths.append(ORIGINAL/'inputs'/(uid+'.npz'))
        for directory in (GRAPH,PARENT):paths.extend(directory/'scores'/(uid+ext) for ext in ('.npz','.json'))
    hashes={str(p):sha(p) for p in paths}
    save(OUT/'MANIFEST.json',{'status':'POST_EVALUATION_DIAGNOSTIC','labels_previously_seen':True,
        'new_predictions':False,'selected':parent['selected'],'release_id':parent['release_id'],'arms':ARMS,
        'comparisons':COMPARISONS,'tokenizer_path':str(tp),'hashes':hashes,'created_unix':time.time(),
        'raw_files':{k:{'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime} for k,p in RAW_FILES.items()}})
    print('Frozen',len(hashes),'sources/inputs, 110 answers, 16 diagnostic trajectories.',flush=True)


def verify():
    m=load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items():assert sha(p)==h,p
    return m


def extract():
    m=verify();started=time.monotonic();rows=[]
    for cell,path in RAW_FILES.items():
        with path.open('rb') as f:data=metadata.MetadataUnpickler(f).load()
        def row_id(raw):return raw['idx'] if cell.startswith('prm') else cell.split('_')[1]+'::'+str(raw['id'])
        index={row_id(v):v for v in data.values()};assert len(index)==len(data)
        for rec in (r for r in m['selected'] if r['cell']==cell):
            raw=index[rec['row_id']]
            rows.append({**rec,'question':raw.get('question',raw.get('problem')),'text_steps':raw['steps'],
                'raw_spans':raw['step_token_spans'],'align_diag':raw['align_diag'],'gen_token_ids':raw['gen_token_ids'],
                'raw_target':raw['error_steps'] if cell.startswith('prm') else raw['label'],
                'classification':raw.get('classification'),'category':raw.get('category'),
                'token_entropies':raw['token_entropies'],'token_spilled_energies':raw['token_spilled_energies'],
                'token_logsumexp':raw['token_logsumexp']})
        del index,data;gc.collect();print('Extracted',cell,flush=True)
    assert len(rows)==110
    save(OUT/'RAW_METADATA.json',{'manifest_sha256':sha(OUT/'MANIFEST.json'),'rows':rows,
        'numpy_payloads_discarded':True,'labels_used':True,'seconds':time.monotonic()-started})


def finite_median(values):
    v=[x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def aggregate(records):
    result={}
    for arm in ARMS:
        pb=[r for r in records if r['cell'].startswith('pb')];err=[r for r in pb if r['target']!=-1]
        clean=[r for r in pb if r['target']==-1];ds=[r['methods'][arm] for r in pb]
        result[arm]={'native_categories':dict(Counter(d['outcome']['category'] for d in ds)),
            'raw_error_peak_exact':sum(r['methods'][arm]['outcome']['raw_peak_exact'] for r in err),
            'raw_error_peak_before':sum(r['methods'][arm]['outcome']['raw_peak_before'] for r in err),
            'raw_error_peak_after':sum(r['methods'][arm]['outcome']['raw_peak_after'] for r in err),
            'error_top_tie_contains_target':sum(r['methods'][arm]['outcome']['target_in_top_tie'] for r in err),
            'exact_peak_hidden_by_gate':sum(r['methods'][arm]['outcome']['exact_peak_hidden'] for r in err),
            'all_pb_multiple_exact_top_steps':sum(len(d['exact_top_steps'])>1 for d in ds),
            'all_pb_multiple_numerical_top_steps':sum(len(d['numerical_top_steps'])>1 for d in ds),
            'all_pb_shared_top_plateaus':sum(bool(d['shared_peak_plateaus']) for d in ds),
            'all_pb_small_margin':sum(d['margin_le_0_1'] for d in ds),
            'median_error_peak_minus_target':finite_median([r['methods'][arm]['peak']-r['target'] for r in err]),
            'median_error_true_step_rank':finite_median([r['methods'][arm]['true_step_rank_best'] for r in err]),
            'median_true_step_margin':finite_median([r['methods'][arm]['peak_minus_true_score'] for r in err]),
            'median_length_score_spearman':finite_median([r['methods'][arm]['length_score_spearman'] for r in records]),
            'median_clean_bic_advantage_two':finite_median([r['methods'][arm]['bic_advantage_two'] for r in clean]),
            'median_error_bic_advantage_two':finite_median([r['methods'][arm]['bic_advantage_two'] for r in err])}
    transitions={}
    for new,old in COMPARISONS:
        pb=[r for r in records if r['cell'].startswith('pb')]
        def ids(test):return [r['uid'] for r in pb if test(r,r['methods'][new],r['methods'][old])]
        transitions[new+' minus '+old]={
            'native_lost_correct':ids(lambda r,n,o:o['outcome']['correct'] and not n['outcome']['correct']),
            'native_gained_correct':ids(lambda r,n,o:n['outcome']['correct'] and not o['outcome']['correct']),
            'raw_peak_lost':ids(lambda r,n,o:r['target']!=-1 and o['peak']==r['target'] and n['peak']!=r['target']),
            'raw_peak_gained':ids(lambda r,n,o:r['target']!=-1 and n['peak']==r['target'] and o['peak']!=r['target']),
            'gate_changed':ids(lambda r,n,o:(n['prediction']==-1)!=(o['prediction']==-1)),
            'peak_changed':ids(lambda r,n,o:n['peak']!=o['peak']),
            'old_margin_small_when_peak_changed':ids(lambda r,n,o:n['peak']!=o['peak'] and o['margin_le_0_1'])}
    return result,transitions


def analyze():
    m=verify();started=time.monotonic();raw=load(OUT/'RAW_METADATA.json');assert raw['manifest_sha256']==sha(OUT/'MANIFEST.json')
    evaluation=load(PARENT/'EVALUATION.json');by_uid={r['uid']:r for r in evaluation['rows']}
    tok=Tokenizer.from_file(m['tokenizer_path']);records=[];counts=Counter();largest=0.
    for rec in sorted(raw['rows'],key=lambda r:r['uid']):
        uid=rec['uid'];row=by_uid[uid];text='\n\n'.join(rec['text_steps']);enc=tok.encode(text,add_special_tokens=False)
        assert enc.ids==rec['gen_token_ids'] and len(enc.ids)==rec['tokens'],uid
        chars,spans=character_alignment(rec['text_steps'],enc.offsets)
        assert spans==rec['raw_spans'],uid
        with np.load(ORIGINAL/'inputs'/(uid+'.npz'),allow_pickle=False) as inp:
            ss,ee=inp['step_starts'],inp['step_ends'];assert np.column_stack((ss,ee)).tolist()==spans,uid
            for name,col in [('token_entropies',1),('token_spilled_energies',15),('token_logsumexp',19)]:
                np.testing.assert_array_equal(inp['raw'][:,col],rec[name]);counts['primitive_stream_exact_matches']+=1
        assert len(ss)==len(rec['text_steps'])==rec['steps']
        if rec['cell'].startswith('prm'):
            flags=[int(i+1 in rec['raw_target']) for i in range(rec['steps'])];assert flags==row['target']
        else:assert rec['raw_target']==row['target']
        used=np.zeros(rec['tokens'],int)
        for a,b in spans:used[a:b]+=1
        assert not np.any(used>1);assert not rec['align_diag']['problems']
        record={k:rec[k] for k in ('uid','row_id','cell','group_id','tokens','steps','question','text_steps','classification','category')}
        record.update(target=row['target'],spans=spans,char_spans=chars,token_offsets=enc.offsets,
            separator_tokens=int((used==0).sum()),methods={})
        graph=load(GRAPH/'scores'/(uid+'.json'));aug=load(PARENT/'scores'/(uid+'.json'))
        with np.load(GRAPH/'scores'/(uid+'.npz'),allow_pickle=False) as ga,np.load(PARENT/'scores'/(uid+'.npz'),allow_pickle=False) as aa:
            for arm in ARMS:
                a,meta=(ga,graph) if arm in OLD_ARMS else (aa,aug);detail=meta['methods'][arm]
                assert row['valid'][arm] and row['decision_valid'][arm] and detail['valid'] and detail['decision_valid']
                np.testing.assert_array_equal(a[arm+'__risk'],row['scores'][arm])
                d=peak_geometry(a['window_starts'],a['window_ends'],a[arm+'__window'],ss,ee,rec['tokens'],row['scores'][arm])
                assert d['peak']==row['peaks'][arm]==detail['peak'];assert detail['prediction']==row['predictions'][arm]
                d.update(prediction=detail['prediction'],bic=detail['gate']['bic'],
                    bic_advantage_two=float(detail['gate']['bic'][0]-detail['gate']['bic'][1]),
                    two_components_selected=detail['gate']['two_components_selected'],
                    length_score_spearman=float(spearmanr(ee-ss,row['scores'][arm]).statistic) if len(ss)>1 and np.ptp(ee-ss)>0 and np.ptp(row['scores'][arm])>0 else None,
                    window_starts=a['window_starts'].tolist(),window_ends=a['window_ends'].tolist(),
                    window_risk=a[arm+'__window'].tolist())
                if rec['cell'].startswith('pb'):
                    target=row['target'];d['outcome']=pb_outcome(target,d['prediction'],d['peak'],d['numerical_top_steps'])
                    if target!=-1:
                        scores=np.array(row['scores'][arm]);d.update(true_step_rank_best=int(1+np.sum(scores>scores[target])),
                            peak_minus_true_score=float(scores.max()-scores[target]),true_step_length=int(ee[target]-ss[target]),
                            peak_step_length=int(ee[d['peak']]-ss[d['peak']]))
                largest=max(largest,d['max_replay_discrepancy']);record['methods'][arm]=d;counts['trajectory_replays']+=1
        records.append(record);counts['raw_label_matches']+=1;counts['token_id_replays']+=1;counts['span_replays']+=1
    summary,transitions=aggregate(records)
    mm=module(ROOT/'scripts/run_answer_localization_v2.py','forensics_historical_metrics')
    for arm,bundle in evaluation['metrics'].items():
        assert mm.prm_metric(evaluation['rows'],arm)==bundle['prm'];assert mm.pb_metric(evaluation['rows'],arm)==bundle['pb']
        counts['historical_metric_bundles']+=1
    cases={}
    conditions={
        'shared_boundary_peak_with_gold_in_tie':lambda r:r['target']!=-1 and r['methods']['dual__iu']['peak']!=r['target'] and r['methods']['dual__iu']['outcome']['target_in_top_tie'] and r['methods']['dual__iu']['shared_peak_plateaus'],
        'AR_graph_loses_correct_peak':lambda r:r['target']!=-1 and r['methods']['dual__cond100_graph010']['peak']==r['target'] and r['methods']['ar1__graph010']['peak']!=r['target'],
        'AR_graph_new_clean_false_alarm':lambda r:r['target']==-1 and r['methods']['dual__cond100_graph010']['prediction']==-1 and r['methods']['ar1__graph010']['prediction']!=-1,
        'both_originals_miss_outside_top_tie':lambda r:r['target']!=-1 and not r['methods']['dual__iu']['outcome']['target_in_top_tie'] and not r['methods']['dual__cond100_graph010']['outcome']['target_in_top_tie']}
    for name,test in conditions.items():
        eligible=[r['uid'] for r in records if r['cell'].startswith('pb') and test(r)];cases[name]={'eligible_ids':eligible,'illustrated_id':eligible[0] if eligible else None}
    save(OUT/'AUDIT.json',{'status':'COMPLETE_POST_EVALUATION_DIAGNOSTIC','manifest_sha256':sha(OUT/'MANIFEST.json'),
        'metadata_sha256':sha(OUT/'RAW_METADATA.json'),'records':records,'summary':summary,'transitions':transitions,
        'counts':dict(counts),'max_step_replay_discrepancy':largest,'historical_metrics':evaluation['metrics'],
        'illustration_rule':'First lexicographic UID in each explicit diagnostic category','cases':cases,
        'seconds':time.monotonic()-started})
    print('Completed audit',dict(counts),'max discrepancy',largest,flush=True)
    for arm in ('dual__iu','dual__cond100_graph010','ar1__iu','ar1__graph010'):print(arm,summary[arm],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['tests','prepare','extract','analyze'])
    globals()[parser.parse_args().phase]()
