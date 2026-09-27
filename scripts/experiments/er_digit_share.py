"""expectation_realization_v1, digit-share diagnostic (PRMBench only; protocol 'controls.digit_share').

How much of the realization block's signal sits on tokens that contain a digit?  Diagnostic only:
nothing here is an arm, and digit flags never enter a fused score (Omri's 2026-09-17 decision).
Provided token ids come from the raw PRMBench telemetry pickle; tokens are decoded with the local
Qwen3-8B tokenizer (the telemetry model).  Alignment is proven by reproducing the stored
chosen_surprisal token column and the stored derivative channel before anything is counted.
"""
from pathlib import Path
import importlib.util, json, sys, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
TPFW = MAIN / '.worktrees/token-probability-fusion-v1'; LEV = MAIN / '.worktrees/lsml-ct7-levers-run'
RUN = ROOT / 'results/expectation_realization_v1' / (sys.argv[1] if len(sys.argv) > 1 else 'run_20260927')
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
DRV = load('drv_v1', TPFW / 'spectral_utils/derivative_step_channel_v1.py')
CTC = load('ctc_v1', LEV / 'spectral_utils/chosen_token_calibration.py')
sys.path.insert(0, str(TPFW)); import scripts.run_claude_feature_bank_v1 as runner  # noqa: E402
runner.ROOT = MAIN
t0 = time.time(); out = {'status': 'RUNNING'}

R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Zs = np.load(R / 'OOF_STEP_SCORES.npz'); off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans)
prm = ~ans.cell.str.startswith('pb_').to_numpy()
eligible = np.array([prm[i] and labels[off[i]:off[i+1]].any() and (~labels[off[i]:off[i+1]]).any() for i in range(n)])
TM = np.load(TPFW / 'results/token_probability_fusion_v1/TOKEN_MATRICES.npz'); toff = TM['token_offsets']; spans = TM['step_spans']
tok_cs = TM['tokens'][:, list(TM['channels'].astype(str)).index('chosen_surprisal')].astype(float)
drv_stored = np.load(TPFW / 'results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz')['derivative'][:, 2].astype(float)
recs = json.load(open(TPFW / 'results/localization_full_benchmark_v3/evaluation/JOINED.json', encoding='utf8'))['records']
assert [r['uid'] for r in recs] == ans.uid.astype(str).tolist(), 'JOINED record order differs from OOF answer order'
cell, path, kind, dataset = [s for s in runner.source_specs() if s[0] == 'prmbench_qwen3_8b'][0]
src = runner.source_row_map(runner.load_pickle(path), kind=kind, dataset=dataset)
print(f'pickle loaded {time.time()-t0:.0f}s', flush=True)

from transformers import AutoTokenizer  # noqa: E402
tk = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', local_files_only=True)
digit_cache = {}
def is_digit_token(i):
    if i not in digit_cache: digit_cache[i] = any(ch.isdigit() for ch in tk.decode([int(i)]))
    return digit_cache[i]

surpr_diff = []; drv_diff = []; top_digit = top_cnt = 0; base_tok_digit = base_tok = 0; uniform_expect = []
exc_pos_digit = exc_pos_all = 0.0; drv_nodigit = np.zeros(int(off[-1])); n_ans = 0
for i in np.flatnonzero(prm):
    row = src[str(recs[i]['row_id'])]; ta, tb = int(toff[i]), int(toff[i+1]); sp = spans[off[i]:off[i+1]]
    gid = np.asarray(row['gen_token_ids'], np.int64); se = np.asarray(row['token_spilled_energies'], float)
    assert len(gid) == tb - ta and len(se) == tb - ta, (i, len(gid), tb - ta)
    surpr_diff.append(float(np.abs(se - tok_cs[ta:tb]).max()))
    x = tok_cs[ta:tb, None]; d = DRV.derivative_step_readout(x, sp)[:, 0]; drv_diff.append(float(np.abs(d - drv_stored[off[i]:off[i+1]]).max()))
    dig = np.array([is_digit_token(v) for v in gid], bool)
    sm = DRV.ema(x[:, 0]); rises = np.maximum(np.diff(sm, prepend=sm[:1]), 0.0)
    lp = np.asarray(row['top_k_logprobs']['logprobs'], float); ids = np.asarray(row['top_k_logprobs']['ids'], np.int64)
    cal, _, _ = CTC.token_calibration(lp, ids, gid, se); exc = np.maximum(cal[:, 2], 0.0)
    exc_pos_digit += float(exc[dig].sum()); exc_pos_all += float(exc.sum())
    for s, (a, b) in enumerate(sp):
        seg = rises[a:b]; k = min(3, len(seg))
        top = np.argpartition(seg, len(seg) - k)[-k:]; top_digit += int(dig[a:b][top].sum()); top_cnt += k
        uniform_expect.append(k * dig[a:b].mean())
        nd = seg[~dig[a:b]]; kk = min(3, len(nd)); drv_nodigit[off[i] + s] = np.partition(nd, len(nd) - kk)[-kk:].mean() if kk else 0.0
    base_tok_digit += int(dig.sum()); base_tok += len(dig); n_ans += 1
out['alignment'] = {'answers': n_ans, 'max_abs_chosen_surprisal_vs_token_matrix': max(surpr_diff), 'max_abs_derivative_recomputed_vs_stored': max(drv_diff)}
assert out['alignment']['max_abs_chosen_surprisal_vs_token_matrix'] < 1e-4 and out['alignment']['max_abs_derivative_recomputed_vs_stored'] < 1e-9, out['alignment']
def within_auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1; return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def mean_within(s):
    return float(np.mean([within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]]) for i in np.flatnonzero(eligible)]))
out.update({
    'digit_token_base_rate': base_tok_digit / base_tok,
    'derivative_top3_rises_on_digit_tokens': top_digit / top_cnt,
    'derivative_top3_expected_if_uniform_within_step': float(np.sum(uniform_expect) / top_cnt),
    'zscore_positive_excess_mass_on_digit_tokens': exc_pos_digit / exc_pos_all,
    'derivative_within_auc_stored': mean_within(drv_stored),
    'derivative_within_auc_digit_rises_excluded': mean_within(drv_nodigit),
    'tokenizer': 'Qwen/Qwen3-8B (local HF cache)', 'distinct_token_ids_decoded': len(digit_cache), 'seconds': round(time.time() - t0, 1), 'status': 'COMPLETE'})
RUN.mkdir(parents=True, exist_ok=True); (RUN / 'DIGIT_SHARE.json').write_text(json.dumps(out, indent=1), encoding='utf8')
print(json.dumps(out, indent=1))
