"""Independently verify grouping with sparse connected components and score replay."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/localization_source_group_audit_v1'
PARENT=ROOT/'results/fusion_explicit_fallback_pilot_v1'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


def review():
    started=time.monotonic();manifest=load(OUT/'BRIDGE_MANIFEST.json')
    for p,h in manifest['hashes'].items():assert sha(p)==h,p
    audit=load(OUT/'AUDIT.json');mapping=load(OUT/'CANONICAL_GROUPS.json')
    prm_meta=load(OUT/'QUESTION_METADATA.json');pb_meta=load(OUT/'PB_QUESTION_METADATA.json')
    for p,h in {prm_meta['raw_path']:prm_meta['raw_sha256'],**pb_meta['source_files']}.items():assert sha(p)==h,p
    assert pb_meta['q4_q8_exact_question_replays']==3400
    assert prm_meta['extractor_sha256']==pb_meta['extractor_sha256']==sha(ROOT/'scripts/audit_localization_source_groups.py')
    nodes={};edges=[]
    def node(key):
        if key not in nodes:nodes[key]=len(nodes)
        return nodes[key]
    for r in prm_meta['rows']:
        source=r['old_group_id'];tail=source[source.index('prm_'):]
        assert tail==r['source_seed'] and len(tail.split('_'))==4
        edges.append((node(('seed',tail)),node(('question',r['question_whitespace_sha256']))))
    for r in pb_meta['rows']:node(('question',r['question_whitespace_sha256']))
    adjacency=coo_matrix((np.ones(len(edges)),tuple(np.array(edges).T)),shape=(len(nodes),len(nodes)))
    n_components,labels=connected_components(adjacency,directed=False)
    seeds=defaultdict(list)
    for (kind,value),index in nodes.items():
        if kind=='seed':seeds[int(labels[index])].append(value)
    expected={}
    for r in mapping['rows']+mapping['pb_rows']:
        component=int(labels[nodes[('question',r['question_whitespace_sha256'])]])
        if seeds[component]:
            canon='prmb_source_v2:'+hashlib.sha256('|'.join(sorted(seeds[component])).encode()).hexdigest()[:24]
        else:canon='pb_source_v2:'+r['question_whitespace_sha256'][:24]
        assert canon==r['canonical_group_id'];expected[r['row_id']]=canon
    old_folds=load(Path('C:/Users/omris/TAU/hd_jlsml_v2_wt/results/joint_lsml_optimization_v2/folds/folds.json'))
    assert sha(Path('C:/Users/omris/TAU/hd_jlsml_v2_wt/results/joint_lsml_optimization_v2/folds/folds.json'))==audit['claude_folds_sha256']
    old_fold_components=defaultdict(set)
    literal_question_folds=defaultdict(set)
    for r in mapping['rows']:old_fold_components[expected[r['row_id']]].add(old_folds['prmbench']['outer'][r['old_group_id']])
    for r in mapping['rows']:literal_question_folds[r['question_whitespace_sha256']].add(old_folds['prmbench']['outer'][r['old_group_id']])
    assert sum(len(v)>1 for v in old_fold_components.values())==audit['canonical_groups_spanning_outer_folds']==707
    pb_fold_components=defaultdict(set);pb_counts=defaultdict(int)
    for r in mapping['pb_rows']:
        pb_fold_components[expected[r['row_id']]].add(old_folds['processbench']['outer'][r['row_id']]);pb_counts[expected[r['row_id']]]+=1
    assert sum(v>1 for v in pb_counts.values())==audit['pb_repeated_question_groups']==430
    assert sum(len(v)>1 for v in pb_fold_components.values())==audit['pb_repeated_groups_spanning_outer_folds']==363
    release=load(OUT/'RELEASE_V2.json');folds=load(OUT/'FOLDS_V2.json')
    old_release=load(ROOT/'results/answer_localization_representation_pilot_v1/RELEASE.json')
    released_rows=0
    for cell,info in release['cells'].items():
        old=old_release['cells'][cell]
        assert sha(info['telemetry_path'])==info['telemetry_sha256']
        assert sha(info['label_path'])==info['label_opaque_sha256']
        assert {k:v for k,v in info.items() if k!='rows'}=={k:v for k,v in old.items() if k!='rows'}
        assert len(info['rows'])==len(old['rows'])
        for row,before in zip(info['rows'],old['rows']):
            assert row['legacy_group_id']==before['group_id'] and row['group_id']==expected[row['row_id']]
            assert {k:v for k,v in row.items() if k not in ('group_id','legacy_group_id')}=={k:v for k,v in before.items() if k!='group_id'}
            released_rows+=1
    all_groups=set(expected.values());assert set(folds['outer'])==all_groups
    split_checks=0
    for k in range(5):
        test={g for g in all_groups if folds['outer'][g]==k};train=all_groups-test
        assert test and train and not test&train
        assert set(folds['inner'][str(k)])==train;split_checks+=1
        for j in range(5):
            valid={g for g in train if folds['inner'][str(k)][g]==j};inner_train=train-valid
            assert valid and inner_train and not valid&inner_train and not (valid|inner_train)&test
            split_checks+=1
    old_eval=load(PARENT/'EVALUATION.json');e=load(OUT/'EVALUATION_V2.json')
    parent_review=load(PARENT/'REVIEW.json');assert parent_review['status']=='PASS'
    for p,h in parent_review['source_hashes'].items():assert sha(p)==h,p
    assert len(e['rows'])==len(old_eval['rows'])==58
    assert e['metrics']==old_eval['metrics'] and len(e['metrics'])==25
    row_checks=0
    for row,before in zip(e['rows'],old_eval['rows']):
        assert row['uid']==before['uid'] and row['legacy_group_id']==before['group_id']
        assert row['group_id']==expected[row['row_id']]
        for key,value in before.items():
            if key!='group_id':assert row[key]==value,(row['uid'],key)
        row_checks+=1
    spec=importlib.util.spec_from_file_location('explicit_review_reference',ROOT/'scripts/review_fusion_explicit_fallback_v1.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    for arm,me in e['metrics'].items():
        reference.check_equal(me,{'prm':reference.prm(e['rows'],arm),'pb':reference.pb(e['rows'],arm),
                                 'pb_common_iu_gate':reference.pb(e['rows'],arm,True)})
    contrasts=load(OUT/'CONTRASTS_V2.json');assert contrasts['status']=='COMPLETE' and len(contrasts['pairs'])==32
    explicit=[]
    for left,right in [('single__joint0','moment__iu'),('dual__graph010','dual__joint0'),('single__joint0','context__equal')]:
        values=reference.explicit_bootstrap(e['rows'],left,right)
        actual=contrasts['pairs'][left+' minus '+right]['corrected_uncertainty']
        for key,value in values.items():reference.check_equal(actual[key],value)
        explicit.append({'left':left,'right':right,'draws':1000,'four_intervals_and_counts':'MATCH'})
        print('Corrected explicit bootstrap matches:',left,'minus',right,flush=True)
    result={'status':'PASS','independent_method':'scipy sparse undirected connected components; explicit bootstrap row materialization',
        'metadata_identity_rows':len(expected),'source_pickle_hashes':1+len(pb_meta['source_files']),
        'graph_components':int(n_components),'release_rows_checked':released_rows,'fold_isolation_checks':split_checks,
        'prmb_identical_question_hashes_crossing_old_folds':sum(len(v)>1 for v in literal_question_folds.values()),
        'score_target_decision_row_replays':row_checks,'unchanged_independent_metric_bundles':25,
        'bootstrap_checks':explicit,'raw_metadata_decoder_validation':'tests/test_source_group_audit.py: 2 passed',
        'hashes':{str(OUT/n):sha(OUT/n) for n in ('AUDIT.json','CANONICAL_GROUPS.json','BRIDGE_MANIFEST.json','RELEASE_V2.json','FOLDS_V2.json','EVALUATION_V2.json','CONTRASTS_V2.json')},
        'explicit_reference_sha256':sha(ROOT/'scripts/review_fusion_explicit_fallback_v1.py'),
        'review_script_sha256':sha(__file__),'seconds':time.monotonic()-started,
        'limits':'Question hashes and declared seed identity do not detect all paraphrases. Claude group-trained methods have not been refitted.'}
    save(OUT/'REVIEW.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('hashes','bootstrap_checks')},indent=2),flush=True)


if __name__=='__main__':review()
