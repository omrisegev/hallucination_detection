"""Align the 29-stream token bank (localization_full_benchmark_v3/inputs) to the roster and reduce to Top10 step profiles."""
import sys, time, json
import numpy as np, pandas as pd
from pathlib import Path
MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = Path(sys.argv[1])
STREAM_NAMES = ("trace_length_series", "entropy_series", "entropy_rolling_spectral_entropy", "entropy_rolling_low_band_power", "entropy_rolling_high_band_power", "entropy_rolling_hl_ratio", "entropy_rolling_dominant_freq", "entropy_rolling_spectral_centroid", "entropy_stft_high_series", "entropy_stft_frame_entropy", "entropy_rolling_tail_ratio", "entropy_sw_var_series", "entropy_pe_series", "entropy_rolling_rs_hurst", "entropy_cusum_abs_series", "spilled_series", "spilled_sw_var_series", "spilled_cusum_abs_series", "spilled_rolling_min", "energy_series", "energy_rolling_min", "energy_sw_var_series", "energy_cusum_abs_series", "top1_logprob_series", "logprob_margin_series", "topk_entropy_series", "topk_varentropy_series", "topk_renyi2_series", "topk_tail_mass_series")  # spectral_utils/answer_localization_v2.py L25-37
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
A = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Z = np.load(R / 'OOF_STEP_SCORES.npz'); off = Z['offsets']; n = len(A)
tk = np.load(MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz'); toff = tk['token_offsets']; spans = tk['step_spans']
key = {(c, i): k for k, (c, i) in enumerate(zip(A.cell, A.id))}
prof = np.full((int(off[-1]), 29), np.nan); seen = np.zeros(n, bool); t0 = time.time()
def top10(v):
    v = v[np.isfinite(v)]
    if len(v) == 0: return np.nan
    k = min(10, len(v)); return np.partition(v, len(v) - k)[-k:].mean()
for cell in sorted(set(A.cell)):
    d = MAIN / 'results/localization_full_benchmark_v3/inputs' / cell
    raw = np.load(d / 'raw.npy', mmap_mode='r'); rid = np.load(d / 'row_ids.npy', allow_pickle=True); to = np.load(d / 'token_offsets.npy'); ss = np.load(d / 'step_starts.npy'); se = np.load(d / 'step_ends.npy'); sro = np.load(d / 'step_row_offsets.npy')
    for r, rowid in enumerate(rid):
        k = key[(cell, str(rowid))]; seen[k] = True
        assert to[r + 1] - to[r] == toff[k + 1] - toff[k], (cell, rowid, 'token count')
        st = ss[sro[r]:sro[r + 1]] - to[r]; en = se[sro[r]:sro[r + 1]] - to[r]; sp = spans[off[k]:off[k + 1]]   # bank spans are absolute
        assert len(st) == len(sp) and np.array_equal(st, sp[:, 0]) and np.array_equal(en, sp[:, 1]), (cell, rowid, 'spans')
        x = np.asarray(raw[to[r]:to[r + 1]], float)
        for j, (a, b) in enumerate(zip(st, en)):
            seg = x[a:b]; prof[off[k] + j] = [top10(seg[:, c]) for c in range(29)]
    print(cell, len(rid), f'{time.time()-t0:.0f}s', flush=True)
assert seen.all() and np.isfinite(prof).mean() > .99
np.save(OUT / 'hist29_top10_profiles.npy', prof); json.dump(list(STREAM_NAMES), open(OUT / 'hist29_names.json', 'w'))
print('done', prof.shape, 'nan frac', float(np.isnan(prof).mean()))
