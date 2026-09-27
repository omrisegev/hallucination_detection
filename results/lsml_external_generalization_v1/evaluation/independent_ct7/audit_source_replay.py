"""Read-only source replay; no external labels or quality metrics are accessed."""
import sys, json, pickle, hashlib, types, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / '.worktrees/lsml-ct7-levers-run'
OUT = Path(__file__).resolve().parent
for name, path in [('spectral_utils', ROOT/'spectral_utils'), ('original_ct7', OLD/'spectral_utils')]:
    pkg = types.ModuleType(name); pkg.__path__ = [str(path)]; sys.modules[name] = pkg
from spectral_utils.external_generalization import ct7, fusion
from original_ct7.ct7_token_streams import bocpd_residual_temporal_recipe
from original_ct7.digitfree_broad50 import step_bank, masked_answer_standardize, NAMES
from original_ct7.chosen_token_calibration import token_calibration, step_sufficient_stats
from original_ct7.frozen_locator_ct7 import despiked_chosen_token_z
from original_ct7.length_calibrated_readout import step_topk_and_calibrated

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    started = time.perf_counter()
    paths = {
        'raw': ROOT/'dataset_cache/repgrid/pb_qwen3_4b/processbench_gsm8k.pkl',
        'roster': ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json',
        'ct7': ROOT/'.worktrees/token-probability-fusion-v1/results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz',
        'tokens': ROOT/'.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz',
        'profiles': ROOT/'results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy',
        'port': ROOT/'spectral_utils/external_generalization/ct7.py',
    }
    with paths['raw'].open('rb') as f: cache = pickle.load(f)
    source = {'gsm8k::'+str(r['id']): r for r in cache.values()}
    records = json.loads(paths['roster'].read_text(encoding='utf8'))['records']
    offsets = np.r_[0, np.cumsum([r['steps'] for r in records])]
    frozen = np.load(paths['ct7'])['step_scores']
    profiles = np.load(paths['profiles'], mmap_mode='r')
    with np.load(paths['tokens']) as saved:
        tokens = saved['tokens']; token_offsets = saved['token_offsets']; saved_spans = saved['step_spans']
    cell_indexes = [i for i,r in enumerate(records) if r['cell']=='pb_gsm8k_q4']
    ordered = sorted(cell_indexes, key=lambda i: (records[i]['tokens'], records[i]['uid']))
    chosen = sorted({ordered[int(j)] for j in np.linspace(0,len(ordered)-1,12)})
    rows = []
    cols = [NAMES.index(n) for n in ('H0lim','ve0','ve0.75','ve1','H0lim_prefix_innovation')]
    for i in chosen:
        rec = records[i]; row = source[rec['row_id']]
        payload = row.get('top_k_logprobs') or row.get('top_k_logprobs_raw')
        safe = {k: row[k] for k in ('gen_token_ids','token_entropies','token_spilled_energies','token_logsumexp')}
        safe['top_k_logprobs'] = payload
        spans = np.asarray(row['step_token_spans'], int)
        a,b = offsets[i:i+2]; ta,tb = token_offsets[i:i+2]
        assert np.array_equal(spans, saved_spans[a:b])
        assert len(safe['gen_token_ids']) == tb-ta == rec['tokens']
        got = ct7.step_scores(safe, spans)
        raw, avail, _ = step_bank(payload['logprobs'],payload['ids'],safe['gen_token_ids'],safe['token_spilled_energies'],spans)
        old_bank = masked_answer_standardize(raw.astype(np.float32).astype(float), avail, np.array([0,len(spans)]))[:,cols]
        residual = bocpd_residual_temporal_recipe(payload['logprobs'],safe['token_entropies'])
        residual_top = step_topk_and_calibrated(residual,np.ones(len(residual),bool),spans)[0]
        residual_view = masked_answer_standardize(residual_top[:,None],np.ones((len(spans),1),bool),np.array([0,len(spans)]))[:,0]
        x,c,d = token_calibration(payload['logprobs'],payload['ids'],safe['gen_token_ids'],safe['token_spilled_energies'])
        chosen_view = despiked_chosen_token_z(step_sufficient_stats(x,c,d,spans),np.array([0,len(spans)]))
        old_views = np.column_stack([old_bank,residual_view,chosen_view])
        independent = old_views.mean(axis=1)
        replay11 = fusion.validate_telemetry(safe)
        if isinstance(replay11,tuple): replay11 = replay11[0]
        e = {'roster_index':i,'uid':rec['uid'],'row_id':rec['row_id'],'tokens':rec['tokens'],'steps':rec['steps'],
             'frozen_ct7_raw_error':float(np.max(np.abs(independent-frozen[a:b]))),
             'original_profiles_error':float(np.max(np.abs(old_views-profiles[a:b]))),
             'port_vs_frozen_answer_z_error':float(np.max(np.abs(got-fusion.answer_z(frozen[a:b])))),
             'port_vs_original_functions_error':float(np.max(np.abs(got-fusion.answer_z(independent)))),
             'bank11_saved_dtype':str(tokens.dtype),'bank11_error_float64':float(np.max(np.abs(replay11-tokens[ta:tb]))),
             'bank11_equal_after_saved_cast':bool(np.array_equal(np.asarray(replay11).astype(tokens.dtype),tokens[ta:tb])),
             'step_spans_exact':True,'bocpd_top10_dtype':str(residual_top.dtype)}
        rows.append(e)
        print(json.dumps(e),flush=True)
    result = {'scope':'FEASIBILITY implementation replay only; no external quality or labels read',
              'n_checked':len(rows),'source_cell_total':len(cell_indexes),'benchmark_total':len(records),
              'selection':'12 deterministic quantiles of token length within pb_gsm8k_q4; identity matched by row_id',
              'files':{k:{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for k,p in paths.items()},
              'rows':rows,'seconds':time.perf_counter()-started}
    result['pass'] = all(r['port_vs_frozen_answer_z_error']<1e-8 and r['port_vs_original_functions_error']<1e-8 and r['bank11_equal_after_saved_cast'] for r in rows)
    (OUT/'SOURCE_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('PASS',result['pass'], 'N',len(rows),flush=True)
    if not result['pass']: raise SystemExit(1)

if __name__=='__main__': main()
