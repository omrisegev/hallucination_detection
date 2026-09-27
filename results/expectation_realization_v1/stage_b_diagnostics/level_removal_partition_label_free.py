# LABEL-FREE diagnostic (no labels read): after the stage-B filter (11 survivors), remove 2 or 3 of the 5 level channels
# and re-cluster with the stable B1 recipe (tie-aware centred top-20% marks, all fit-fold steps, tail_calib_common.lsml_fit_scaled).
# Question (Omri 2026-09-27): do the remaining level channels form ONE group, giving three separate families?
import sys, json, itertools, numpy as np, pandas as pd
from pathlib import Path
W = Path(r'C:\Users\omris\TAU\hallucination_detection\.worktrees')
sys.path.insert(0, str(W / 'depth-feature-fusion-v1')); sys.path.insert(0, str(W / 'ssl-pseudolabel-residual-v1/scripts/experiments'))
from spectral_utils.lsml_gate_locator_research import answer_standardize
import tail_calib_common as TC
from calfix_common import tail_marks
R = W / 'readout-quickest-detection-v1/results/step_evidence_v1'; TPF = W / 'token-probability-fusion-v1/results/token_probability_fusion_v1'
CT7P = W / 'cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1'
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); off = np.load(R / 'OOF_STEP_SCORES.npz')['offsets']; ns = np.diff(off)
fold = ans.fold.to_numpy()
lv = np.load(TPF / 'DERIVATIVE_CHANNELS.npz'); names11 = list(map(str, lv['channels']))
prof = np.load(CT7P / 'profiles.npy').astype(float); pn = json.loads((CT7P / 'PROFILE_VALIDATION.json').read_text(encoding='utf8'))['channels']
raw = np.column_stack([lv['level'].astype(float), prof[:, pn.index('chosen_token_z_despiked')], lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)])
names = names11 + ['realized_z', 'realized_drv']; V = answer_standardize(raw, off)
T = tail_marks(V, off, .2, tie_aware=True, centred=True)[0]
def rows_of(mask): return np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(mask)])
LEVEL = ['q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level']
SURV = [c for c in names if c not in ('energy_innovation', 'top50_js')]
FIT = {}
for k in range(5):
    cal = (k + 1) % 5; FIT[k] = rows_of(np.isin(fold, [f for f in range(5) if f not in (k, cal)]))
def partition(keep, k):
    idx = [names.index(c) for c in keep]; rr = FIT[k]
    g = TC.lsml_fit_scaled(T[rr][:, idx], 0, V[rr][:, idx], standardize=True, loading_scale='unit')['groups']
    groups = {}
    for c, gi in zip(keep, g): groups.setdefault(int(gi), []).append(c)
    return sorted([sorted(v) for v in groups.values()], key=lambda v: keep.index(v[0]) if v[0] in keep else 0)
out = []
for n_remove in (0, 2, 3):
    for rem in itertools.combinations(LEVEL, n_remove):
        keep = [c for c in SURV if c not in rem]; lv_keep = [c for c in LEVEL if c not in rem]
        parts = [partition(keep, k) for k in range(5)]
        canon = {json.dumps(p) for p in parts}
        p0 = parts[0]; lvl_groups = [g for g in p0 if any(c in LEVEL for c in g)]
        one_level_group = len(lvl_groups) == 1 and set(lvl_groups[0]) == set(lv_keep)
        three_families = one_level_group and len(p0) == 3
        row = {'removed': '+'.join(rem) or '(none)', 'K_per_fold': [len(p) for p in parts], 'identical_5_folds': len(canon) == 1,
               'level_one_group_fold0': one_level_group, 'three_families_fold0': three_families,
               'fold0': ' | '.join('+'.join(g) for g in p0)}
        out.append(row); print(json.dumps(row), flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=1), encoding='utf8')
