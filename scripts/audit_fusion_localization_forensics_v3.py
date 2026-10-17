"""Resume the frozen geometry audit against the corrected label bridge."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]


def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


old=module(ROOT/'scripts/audit_fusion_localization_forensics_v1.py','forensics_original_helpers')
OUT=ROOT/'results/fusion_localization_forensics_v3'
CORRECTED=ROOT/'results/localization_prm_label_audit_v1'
ORIGINAL=old.ORIGINAL
ARMS=old.ARMS
load,save,sha=old.load,old.save,old.sha
from spectral_utils.prm_label_contract import prm_error_flags


def prepare():
    assert not (OUT/'MANIFEST.json').exists()
    prior=old.verify();corrected=load(CORRECTED/'RELEASE_V3.json')
    paths=[Path(__file__),ROOT/'docs/experiments/FUSION_LOCALIZATION_FORENSICS_V3.md',
           ROOT/'spectral_utils/prm_label_contract.py',ROOT/'spectral_utils/localization_forensics.py',
           old.OUT/'MANIFEST.json',old.OUT/'RAW_METADATA.json',old.OUT/'FAILURE.json',old.OUT/'TESTS.json',
           CORRECTED/'RELEASE_V3.json',CORRECTED/'current110_EVALUATION_V3.json',CORRECTED/'REVIEW.json',
           Path(corrected['cells']['prmbench_qwen3_8b']['label_path'])]
    hashes=prior['hashes'].copy();hashes.update({str(p):sha(p) for p in paths})
    save(OUT/'MANIFEST.json',{'status':'CORRECTED_LABEL_FROZEN_SCORE_FORENSICS','release_id':corrected['release_id'],
        'selected':prior['selected'],'arms':ARMS,'hashes':hashes,'tokenizer_path':prior['tokenizer_path'],
        'raw_metadata_path':str(old.OUT/'RAW_METADATA.json'),'corrected_evaluation_path':str(CORRECTED/'current110_EVALUATION_V3.json'),
        'labels_used':True,'new_predictions':False,'created_unix':time.time()})
    print('Frozen v3 geometry audit; original raw metadata and all scores retained.',flush=True)


def verify():
    m=load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items():assert sha(p)==h,p
    return m


def analyze():
    m=verify();started=time.monotonic();raw=load(m['raw_metadata_path'])
    assert raw['manifest_sha256']==sha(old.OUT/'MANIFEST.json')
    evaluation=load(m['corrected_evaluation_path']);by_uid={r['uid']:r for r in evaluation['rows']}
    tok=Tokenizer.from_file(m['tokenizer_path']);records=[];counts=Counter();largest=0.
    for rec in sorted(raw['rows'],key=lambda r:r['uid']):
        uid=rec['uid'];row=by_uid[uid];text='\n\n'.join(rec['text_steps']);enc=tok.encode(text,add_special_tokens=False)
        assert enc.ids==rec['gen_token_ids'] and len(enc.ids)==rec['tokens'],uid
        chars,spans=old.character_alignment(rec['text_steps'],enc.offsets)
        assert spans==rec['raw_spans'],uid
        with np.load(ORIGINAL/'inputs'/(uid+'.npz'),allow_pickle=False) as inp:
            ss,ee=inp['step_starts'],inp['step_ends'];assert np.column_stack((ss,ee)).tolist()==spans,uid
            for name,col in [('token_entropies',1),('token_spilled_energies',15),('token_logsumexp',19)]:
                np.testing.assert_array_equal(inp['raw'][:,col],rec[name]);counts['primitive_stream_exact_matches']+=1
        assert len(ss)==len(rec['text_steps'])==rec['steps']
        if rec['cell'].startswith('prm'):
            np.testing.assert_array_equal(prm_error_flags(rec['raw_target'],rec['steps']),row['target'])
        else:assert rec['raw_target']==row['target']
        used=np.zeros(rec['tokens'],int)
        for a,b in spans:used[a:b]+=1
        assert not np.any(used>1);assert not rec['align_diag']['problems']
        record={k:rec[k] for k in ('uid','row_id','cell','group_id','tokens','steps','question','text_steps','classification','category')}
        record.update(target=row['target'],spans=spans,char_spans=chars,token_offsets=enc.offsets,
            separator_tokens=int((used==0).sum()),methods={})
        graph=load(old.GRAPH/'scores'/(uid+'.json'));aug=load(old.PARENT/'scores'/(uid+'.json'))
        with np.load(old.GRAPH/'scores'/(uid+'.npz'),allow_pickle=False) as ga,np.load(old.PARENT/'scores'/(uid+'.npz'),allow_pickle=False) as aa:
            for arm in ARMS:
                a,meta=(ga,graph) if arm in old.OLD_ARMS else (aa,aug);detail=meta['methods'][arm]
                assert row['valid'][arm] and row['decision_valid'][arm] and detail['valid'] and detail['decision_valid']
                np.testing.assert_array_equal(a[arm+'__risk'],row['scores'][arm])
                diag=old.peak_geometry(a['window_starts'],a['window_ends'],a[arm+'__window'],ss,ee,rec['tokens'],row['scores'][arm])
                assert diag['peak']==row['peaks'][arm]==detail['peak'];assert detail['prediction']==row['predictions'][arm]
                diag.update(prediction=detail['prediction'],bic=detail['gate']['bic'],
                    bic_advantage_two=float(detail['gate']['bic'][0]-detail['gate']['bic'][1]),
                    two_components_selected=detail['gate']['two_components_selected'],
                    length_score_spearman=float(spearmanr(ee-ss,row['scores'][arm]).statistic) if len(ss)>1 and np.ptp(ee-ss)>0 and np.ptp(row['scores'][arm])>0 else None,
                    window_starts=a['window_starts'].tolist(),window_ends=a['window_ends'].tolist(),window_risk=a[arm+'__window'].tolist())
                if rec['cell'].startswith('pb'):
                    target=row['target'];diag['outcome']=old.pb_outcome(target,diag['prediction'],diag['peak'],diag['numerical_top_steps'])
                    if target!=-1:
                        scores=np.array(row['scores'][arm]);diag.update(true_step_rank_best=int(1+np.sum(scores>scores[target])),
                            peak_minus_true_score=float(scores.max()-scores[target]),true_step_length=int(ee[target]-ss[target]),
                            peak_step_length=int(ee[diag['peak']]-ss[diag['peak']]))
                largest=max(largest,diag['max_replay_discrepancy']);record['methods'][arm]=diag;counts['trajectory_replays']+=1
        records.append(record);counts['raw_label_matches']+=1;counts['token_id_replays']+=1;counts['span_replays']+=1
    summary,transitions=old.aggregate(records)
    mm=module(ROOT/'scripts/run_answer_localization_v2.py','v3_geometry_historical_metrics')
    for arm,bundle in evaluation['metrics'].items():
        assert mm.prm_metric(evaluation['rows'],arm)==bundle['prm'];assert mm.pb_metric(evaluation['rows'],arm)==bundle['pb']
        counts['historical_metric_bundles']+=1
    for arm in ARMS:
        oracle={}
        for kind in ('perfect_binary_gate','perfect_locator','top_tie_locator'):
            rows=deepcopy(evaluation['rows'])
            for row in rows:
                if not row['cell'].startswith('pb'):continue
                target=row['target'];diag=next(r for r in records if r['uid']==row['uid'])['methods'][arm]
                if kind=='perfect_binary_gate':prediction=-1 if target==-1 else diag['peak']
                elif kind=='perfect_locator':prediction=-1 if diag['prediction']==-1 else target if target!=-1 else diag['peak']
                else:prediction=target if target!=-1 and diag['prediction']!=-1 and target in diag['numerical_top_steps'] else diag['prediction']
                row['predictions'][arm]=prediction
            oracle[kind]=mm.pb_metric(rows,arm)
        summary[arm]['oracle_diagnostics']=oracle
    conditions={
        'shared_boundary_peak_with_gold_in_tie':lambda r:r['target']!=-1 and r['methods']['dual__iu']['peak']!=r['target'] and r['methods']['dual__iu']['outcome']['target_in_top_tie'] and r['methods']['dual__iu']['shared_peak_plateaus'],
        'AR_graph_loses_correct_peak':lambda r:r['target']!=-1 and r['methods']['dual__cond100_graph010']['peak']==r['target'] and r['methods']['ar1__graph010']['peak']!=r['target'],
        'AR_graph_new_clean_false_alarm':lambda r:r['target']==-1 and r['methods']['dual__cond100_graph010']['prediction']==-1 and r['methods']['ar1__graph010']['prediction']!=-1,
        'both_originals_miss_outside_top_tie':lambda r:r['target']!=-1 and not r['methods']['dual__iu']['outcome']['target_in_top_tie'] and not r['methods']['dual__cond100_graph010']['outcome']['target_in_top_tie']}
    cases={}
    for name,test in conditions.items():
        eligible=[r['uid'] for r in records if r['cell'].startswith('pb') and test(r)];cases[name]={'eligible_ids':eligible,'illustrated_id':eligible[0] if eligible else None}
    save(OUT/'AUDIT.json',{'status':'COMPLETE_POST_EVALUATION_DIAGNOSTIC','manifest_sha256':sha(OUT/'MANIFEST.json'),
        'metadata_sha256':sha(Path(m['raw_metadata_path'])),'corrected_evaluation_sha256':sha(Path(m['corrected_evaluation_path'])),
        'records':records,'summary':summary,'transitions':transitions,'counts':dict(counts),
        'max_step_replay_discrepancy':largest,'historical_metrics':evaluation['metrics'],
        'illustration_rule':'First lexicographic UID in each declared category','cases':cases,'seconds':time.monotonic()-started})
    print('Audit complete',dict(counts),'maximum discrepancy',largest,flush=True)
    for arm in ('dual__iu','dual__cond100_graph010','ar1__iu','ar1__graph010'):print(arm,summary[arm],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['prepare','analyze']);globals()[p.parse_args().phase]()
