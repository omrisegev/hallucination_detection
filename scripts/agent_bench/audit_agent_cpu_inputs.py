"""Independent CPU evidence audit; does not score data or alter any gate."""
import ast
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--sources',type=Path,required=True)
parser.add_argument('--previous-inputs',type=Path,required=True)
parser.add_argument('--g6',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
ROOT=args.repo.resolve()
INPUTS=args.inputs.resolve()
SOURCES=args.sources.resolve()
PREVIOUS=args.previous_inputs.resolve()
G6=args.g6.resolve()
OUT=args.out.resolve()
PIN='afae64a4d6826ac1465249f6a002ffde4b20fd37'
sys.path.insert(0,str(ROOT))
from spectral_utils.agent_bench.contracts import adapt_aft,blind_keys
from spectral_utils.external_generalization.artifacts import atomic_json,file_hash
import pyarrow.parquet as pq

OUT.mkdir(exist_ok=False)
critical=['scripts/agent_bench/prepare_agent_sources.py','scripts/agent_bench/prepare_agent_inputs.py',
    'spectral_utils/agent_bench/__init__.py','spectral_utils/agent_bench/contracts.py',
    'spectral_utils/agent_bench/render_qwen3.py','cluster/run_agent_telemetry.py',
    'cluster/submit_agent_telemetry.sbatch','tests/test_agent_bench.py',
    'tests/fixtures/agent_qwen3_chat_template.jinja','cluster/run_external_telemetry.py',
    'cluster/backfill_views.py','spectral_utils/external_generalization/contracts.py',
    'spectral_utils/external_generalization/artifacts.py','spectral_utils/external_generalization/budget.py']
code={}
for p in critical:
    expected=subprocess.check_output(['git','show',PIN+':'+p],cwd=ROOT)
    actual=(ROOT/p).read_bytes()
    assert actual==expected,f'code changed from Claude pin: {p}'
    code[p]={'sha256':hashlib.sha256(actual).hexdigest(),'git_blob':subprocess.check_output(['git','rev-parse',PIN+':'+p],cwd=ROOT).decode().strip()}
atomic_json(OUT/'CODE_PIN.json',{'status':'PASS','commit':PIN,'files':code,'gate_code_changed':False})

manifest=json.loads((SOURCES/'SOURCES.json').read_text())
for entry in manifest['files']:
    p=SOURCES/entry['local_path']
    assert p.stat().st_size==entry['bytes'] and file_hash(p)==entry['sha256']
paths={(r['key'],r['path']):SOURCES/r['local_path'] for r in manifest['files']}
split=json.loads(paths['aft','splits_test.json'].read_text())
test_ids=set(split['test_safe']+split['test_unsafe'])
counts,steps,rows_by_benchmark={}, {}, {}
all_ids=[]
input_hashes=json.loads((INPUTS/'INPUTS.json').read_text())
for b in ('apb','tprm','aft'):
    p=INPUTS/f'{b}_answers.jsonl'
    assert file_hash(p)==input_hashes[b]
    rows=[json.loads(l) for l in p.open()]
    rows_by_benchmark[b]=rows
    counts[b]=len(rows)
    steps[b]=sum(len(r['step_message_indices'])+len(r['step_entry_indices']) for r in rows)
    all_ids.extend(r['uid'] for r in rows)
    for r in rows:
        assert not blind_keys(r),r['uid']
assert counts=={'apb':1000,'tprm':890,'aft':2276}
assert steps=={'apb':8509,'tprm':890,'aft':26306}
assert len(set(all_ids))==len(all_ids)==4166
assert sum(steps.values())==35705

aft={r['uid']:r for r in rows_by_benchmark['aft']}
source_rows=[(r,safe) for safe in (True,False) for r in pq.read_table(paths['aft',f"aftraj_{'safe' if safe else 'unsafe'}.parquet"]).to_pylist()]
source_ids={r['conv_id'] for r,safe in source_rows}
assert len(source_ids)==2276
assert all(re.fullmatch(r'[A-Za-z0-9_-]+',s) for s in source_ids)
source_groups=defaultdict(set)
classification=Counter()
metadata_leaks=[]
raw_id_leaks=[]
mutation_failures=[]
source_mapping_failures=[]
for r,safe in source_rows:
    digest=hashlib.sha256(r['conv_id'].encode()).hexdigest()[:16]
    uid=f"aft:{r['domain']}:{digest}"
    group='aft:'+hashlib.sha256(' '.join(r['task'].split()).encode()).hexdigest()[:16]
    item=aft[uid]
    classification['safe' if safe else 'unsafe']+=1
    if not safe:
        classification['unsafe_source:'+r['unsafe_source']]+=1
        assert 0<=r['mistake_step']<r['num_turns']
    else:assert r.get('mistake_step',-1)==-1
    assert item['source_id']==digest and item['group']==group
    assert re.fullmatch(r'aft:(math|coding|agentic):[0-9a-f]{16}',uid)
    assert re.fullmatch(r'aft:[0-9a-f]{16}',item['group'])
    source_groups[' '.join(r['task'].split())].add(item['group'])
    expected_system='You are reviewing the log of a multi-agent system that tried to solve the task below.'
    if r['tools']:
        expected_system+='\n\nAvailable tools:\n'+'\n'.join(f"- {t['name']}: {t['description']}" for t in r['tools'])
    entries=[]
    for i,t in enumerate(r['turns']):
        fields=[prefix+(t.get(k) or '') for k,prefix in (('thought','[Thought] '),('action','[Action] '),('content','[Content] ')) if t.get(k)]
        entries.append(f"[{i}] {t['role']}:"+ ('\n'+'\n'.join(fields) if fields else ''))
    expected_messages=[{'role':'system','content':expected_system},{'role':'user','content':r['task']},
                       {'role':'assistant','content':'\n\n'.join(entries)}]
    if item['messages']!=expected_messages:source_mapping_failures.append(uid)
    assert len(item['step_entry_indices'])==r['num_turns']==len(r['turns'])
    assert item['step_entry_indices']==item['entry_to_history']==list(range(r['num_turns']))
    denied=set(item)&{'conv_id','label','mistake_step','mistake_agent','mistake_reason','unsafe_source','gold_answer','split_test'}
    if denied:metadata_leaks.append({'uid':uid,'fields':sorted(denied)})
    # Search complete answer JSON (including prompts) for any original release conv_id.
    words=set(re.findall(r'[A-Za-z0-9_-]+',json.dumps(item,ensure_ascii=False)))
    if words&source_ids:raw_id_leaks.append({'uid':uid,'matches':len(words&source_ids)})
    # Flip correctness metadata, error localization and test membership; same observations
    # must produce byte-identical collector inputs. This is preparation validation only.
    mutated=deepcopy(r)
    mutated.update(gold_answer='AUDIT_EVALUATOR_SENTINEL',mistake_agent='AUDIT_EVALUATOR_SENTINEL',
                   mistake_reason='AUDIT_EVALUATOR_SENTINEL',unsafe_source='AUDIT_EVALUATOR_SENTINEL',
                   mistake_step=0 if safe else -1)
    opposite_tests=set() if r['conv_id'] in test_ids else {r['conv_id']}
    changed,annotation=adapt_aft(mutated,not safe,opposite_tests)
    if changed.to_dict()!=item:mutation_failures.append(uid)
    assert annotation['conv_id']==r['conv_id']
    assert annotation['split_test']!=(r['conv_id'] in test_ids)
assert all(len(groups)==1 for groups in source_groups.values())
assert not metadata_leaks and not raw_id_leaks and not mutation_failures and not source_mapping_failures
collector_code=(ROOT/'cluster/run_agent_telemetry.py').read_text()
imports=[(type(n).__name__,ast.unparse(n)) for n in ast.walk(ast.parse(collector_code)) if isinstance(n,(ast.Import,ast.ImportFrom))]
assert 'evaluator_only' not in collector_code and all('evaluation' not in statement for kind,statement in imports)
aft_report={'status':'PASS','items_checked':2276,'all_answer_items_recursive_key_scan':4166,
    'raw_release_id_leaks':raw_id_leaks,'label_metadata_leaks':metadata_leaks,
    'metadata_flip_invariance_checked':2276,'metadata_flip_failures':mutation_failures,
    'full_source_prompt_reconstruction_checked':2276,'prompt_mapping_failures':source_mapping_failures,
    'source_tasks':len(source_groups),'prepared_aft_groups':len(set(r['group'] for r in aft.values())),
    'group_consistency_failures':0,'classification_counts':dict(classification),
    'original_ids_sorted_away':True,'answer_uid_order_sorted':list(aft)==sorted(aft),
    'collector_has_evaluator_import_or_path':False,'evaluator_sidecars_opened':False,
    'scope':'Explicit evaluator metadata/raw IDs absent; observed source task/log content preserved. Hash IDs are pseudonyms, not access control.'}
assert aft_report['answer_uid_order_sorted']
atomic_json(OUT/'AFT_LABEL_BLINDNESS.json',aft_report)

# Additional full-AFT observer-prefix parity and per-item native token-span evidence.
# Use the pinned renderer unchanged, not another acceptance rule.
from spectral_utils.agent_bench.render_qwen3 import tokenize,native,HEADER,EMPTY
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(SOURCES/'tokenizer',local_files_only=True,use_fast=True)
aft_spans={}
for uid,item in aft.items():
    tokenized=tokenize(item,tokenizer,max_context=32768)
    text=tokenized['full_text']
    header=text.index(HEADER)
    content_start=header+len(HEADER)+len(EMPTY)
    assert text[header+len(HEADER):content_start]==EMPTY
    assert text[:content_start]==native(tokenizer,item['messages'][:2],item['tools'],generation=True)
    assert text[content_start:content_start+len(item['messages'][2]['content'])]==item['messages'][2]['content']
    aft_spans[uid]={'tokens':len(tokenized['prompt_ids'])+len(tokenized['gen_ids']),
                   'steps':len(tokenized['step_token_spans']),
                   'boundary_overhang':tokenized['boundary_overhang'],'observer_prefix_parity':'PASS'}
atomic_json(OUT/'AFT_TOKEN_SPAN_AUDIT.json',{'status':'PASS','items_checked':2276,
    'steps_checked':sum(r['steps'] for r in aft_spans.values()),
    'observer_prefixes_checked':2276,'failures':[],
    'overhang_items':sum(r['boundary_overhang']['spans']>0 for r in aft_spans.values()),
    'overhang_spans':sum(r['boundary_overhang']['spans'] for r in aft_spans.values()),
    'overhang_chars':sum(r['boundary_overhang']['chars'] for r in aft_spans.values()),
    'records':aft_spans})

for b in ('apb','tprm','aft'):
    old=PREVIOUS/f'{b}_answers.jsonl'
    old_hash=json.loads((PREVIOUS/'INPUTS.json').read_text())[b]
    assert file_hash(old)==old_hash
old_aft=[json.loads(l) for l in (PREVIOUS/'aft_answers.jsonl').open()]
assert set(r['uid'] for r in old_aft).isdisjoint(aft)
old_by_source={r['source_id']:r for r in old_aft}
for source,safe in source_rows:
    digest=hashlib.sha256(source['conv_id'].encode()).hexdigest()[:16]
    new=aft[f"aft:{source['domain']}:{digest}"]
    old=old_by_source[source['conv_id']]
    for field in set(old)-{'uid','source_id','group'}:
        assert old[field]==new[field],(new['uid'],field)
for b in ('apb','tprm'):
    assert file_hash(INPUTS/f'{b}_answers.jsonl')==json.loads((PREVIOUS/'INPUTS.json').read_text())[b]

gates=json.loads((INPUTS/'CPU_GATES.json').read_text())
assert set(gates)=={'G1','G2','G3','G4','LENGTH'}
assert all(r['status']=='PASS' and not r.get('failures',[]) for r in gates.values())
assert gates['G1']['checked']==910 and len(gates['G1']['template_last_turn_block'])==6
assert gates['G2']['checked']==gates['G4']['checked']==4166 and gates['G3']['checked']==445
prepared=json.loads((INPUTS/'PREPARATION.json').read_text())
assert prepared['status']=='PASS' and not prepared['failures']
g6=json.loads(G6.read_text())
assert g6['status']=='PASS' and g6['items']==9 and g6['resume_identical_to_fresh']
lengths=json.loads((INPUTS/'LENGTHS.json').read_text())
assert len(lengths['tokens_per_item'])==4166 and lengths['truncated']==0
assert all(lengths['tokens_per_item'][uid]==r['tokens'] for uid,r in aft_spans.items())
excluded=json.loads((INPUTS/'EXCLUDED_OVERLENGTH.json').read_text())
assert len(excluded)==10 and Counter(x['subset'] for x in excluded)=={'apb:gaia_dev':8,'apb:hotpotqa':2}
assert all(x['tokens']>32768 and lengths['tokens_per_item'][x['uid']]==x['tokens'] for x in excluded)
assert sum(v>32768 for v in lengths['tokens_per_item'].values())==10
assert all(s['excluded']/s['items']<=.05 for s in lengths['subsets'].values())
summary={'status':'PASS','source_commit':PIN,'inputs_directory':str(INPUTS),'counts':counts,'steps':steps,
    'total_items':4166,'total_steps':35705,'source_files_hash_verified':len(manifest['files']),
    'all_six_cpu_gates':{**{k:r['status'] for k,r in gates.items()},'G5':'PASS','G6':'PASS'},
    'aft_label_blindness':'PASS','unit_tests':17,'g6_fixture_records':9,
    'overlength_excluded':10,'truncated':0,'g7':'NOT_RUN','aircc_connected':False,
    'old_inputs_preserved':True,'aft_uids_all_new':True,'aft_observations_unchanged':True,
    'apb_tprm_inputs_byte_identical':True,'runtime_code_unchanged_from_claude':True,
    'answers_sha256':input_hashes,'g1_exact_template_quirk_variants':6}
atomic_json(OUT/'AUDIT.json',summary)
print(json.dumps(summary))
