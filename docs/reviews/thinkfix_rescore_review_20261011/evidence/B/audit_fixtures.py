import ast, copy, importlib.util, json, sys
sys.dont_write_bytecode=True
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path('/private/tmp/thinkfix-review-080877eb')
OUT=Path('/private/tmp/thinkfix-audit-B')
spec=importlib.util.spec_from_file_location('mask', ROOT/'spectral_utils/first_token_mask.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
entry={'gen_token_ids':[11,12,13,14], 'token_entropies':[1.,2.,3.,4.],
       'token_spilled_energies':np.arange(4.), 'top_k_logprobs':{'ids':np.arange(8).reshape(4,2),'logprobs':np.arange(8).reshape(4,2)},
       'step_token_spans':[(0,2),(2,2),(2,4)],'align_diag':{'n_tokens':4,'token_coverage':1.,'step_token_lengths':[2,0,2]}}
before=copy.deepcopy(entry); masked=m.mask_first_token(entry)
assert entry['gen_token_ids']==before['gen_token_ids']
assert np.array_equal(entry['token_spilled_energies'],before['token_spilled_energies'])
assert masked['gen_token_ids']==[12,13,14]
assert masked['step_token_spans']==[(0,1),(1,1),(1,3)]
assert masked['align_diag']['step_token_lengths']==[1,0,2]
assert np.array_equal(masked['top_k_logprobs']['ids'],before['top_k_logprobs']['ids'][1:])
try:m.shift_spans([(0,1)])
except m.MaskError:pass
else:raise AssertionError('new empty step silently accepted')
assert m.shift_spans([(387,387),(631,631),(230,230)])==[(386,386),(630,630),(229,229)]

# Execute the production score_rows function with a preexisting old cache and changed prompt,
# stubbing GPU work. This checks whether resume detects prompt mismatch without GPU/imports.
resume=[]
for rel in ['cluster/run_teacher_forced.py','cluster/run_prmbench_teacher_forced.py']:
    tree=ast.parse((ROOT/rel).read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('score_rows','batched')]
    env={'STOP':{'flag':False},'save_cache_atomic':lambda *args:None,'batched':lambda *args:[],
         'time':SimpleNamespace(time=lambda:0)}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),rel,'exec'),env)
    cache={0:{'token_entropies':[35.],'gen_token_ids':[11],'old_prompt':True}}
    items=[{'idx':0,'prompt_ids':[999],'gen_ids':[11]}]
    ok=env['score_rows'](None,items,cache,OUT/'unused',SimpleNamespace(max_batch_tokens=16,max_batch=1,checkpoint_every=1))
    assert ok and cache[0]['old_prompt']
    resume.append({'path':rel,'returned_complete':ok,'stale_payload_kept':True})

p=ROOT/'results/thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz'
z=dict(np.load(p,allow_pickle=False));meta=json.loads((p.parent/'prmbench_meta.json').read_text())
md={r['idx']:r for r in meta['rows']};idx=z['idx'].astype(str);off=z['offsets']
assert len(idx)==6969 and len(set(idx))==6969
assert set(idx)==set(md)
assert len(off)==6970 and off[0]==0 and off[-1]==94203
assert np.all(np.diff(off)>0)
assert [md[i]['n_steps'] for i in idx]==np.diff(off).tolist()
cols=[k for k in z if k.startswith('S__') or k=='G']
assert all(z[k].shape==(94203,) and np.isfinite(z[k]).all() for k in cols)
zmeans={k:float(max(abs(z[k][a:b].mean()) for a,b in zip(off[:-1],off[1:]))) for k in cols if k.startswith('S__')}
assert max(zmeans.values())<1e-9
output={'mask_fixtures':'PASS','already_empty_source_steps':'PASS','newly_empty_steps_rejected':'PASS',
        'mask_n_tokens_metadata_stale':masked['align_diag']['n_tokens']!=len(masked['gen_token_ids']),
        'resume_fixtures':resume,
        'independently_checked_PRMBench_export':{'answers':len(idx),'steps':int(off[-1]),'unique_ids':len(set(idx)),
        'metadata_id_and_stepcount_match':True,'finite_score_columns':len(cols),'max_abs_answer_z_means':zmeans},
        'limitations':['Only bank_N16 legacy export is tracked, not variant per-answer arrays or raw GPU caches.',
        'PB13769-total and external3195 counts currently verified only from manifests/status/seals, not raw population.']}
(OUT/'fixture_results.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
