# LABEL-FREE diagnostic (no labels read): why does L-SML's grouping split the level family, and what would an
# eigenvalue-based merge step do?  Omri 2026-09-28: "it makes no sense that the clustering does not work for us".
# (1) Residual test: is the L-SML Eq.14 residual of the chosen partition lower than that of the same partition with the
#     two level-containing groups merged (-> the model itself prefers the split: the level cartel "explains itself" through
#     the off-group factor), or higher (-> the candidate generator never proposes the merge)?  The K curve shows which
#     partition is proposed for each K.
# (2) Merge-rule behaviour: for every pair of discovered groups, lambda_2 and lambda_1/p of the union's correlation
#     matrix; the Kaiser/VARCLUS rule (merge the pair with the smallest lambda_2 while lambda_2 < 1) is replayed.
# Banks: B13, B20, B32, B51 (as er_generality_v1) and B13 + the three decoding-independent digit features (B16).
# Survivors of the label-free DS filter are used (the chain's input to grouping). Continuous and binary (tie-aware) inputs.
import sys, json, numpy as np, pandas as pd
from pathlib import Path
W = Path(r'C:\Users\omris\TAU\hallucination_detection\.worktrees')
ROOT = W / 'ssl-pseudolabel-residual-v1'
sys.path.insert(0, str(W / 'depth-feature-fusion-v1')); sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from spectral_utils.lsml_gate_locator_research import answer_standardize, fit_fusion_weights, FusionRecipe
import tail_calib_common as TC, er_stage_a as SA, er_stage_b as SB
from calfix_common import tail_marks
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location('digit_std_src', ROOT / 'spectral_utils/digit_feature_family.py')
DNAMES = ('digit_alternative', 'digit_spread', 'digit_alternative_innovation')   # spectral_utils/digit_feature_family.NAMES
def digit_std(x, offsets, active):   # = spectral_utils.digit_feature_family.answer_standardize (masked per-answer z; inactive -> 0)
    x = np.asarray(x, float); out = np.zeros(x.shape)
    for a, b in zip(offsets[:-1], offsets[1:]):
        for j in range(x.shape[1]):
            v = active[a:b, j]; y = x[a:b, j][v]
            if len(y) and y.std() > 1e-12: out[a:b, j][v] = (y - y.mean()) / y.std()
    return out
F = TC.fu()
R = W / 'readout-quickest-detection-v1/results/step_evidence_v1'; TPF = W / 'token-probability-fusion-v1/results/token_probability_fusion_v1'
CT7P = W / 'cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1'
SCR = Path(r'C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\2d14a8c9-8b3f-489b-b26f-812b4b84a8b3\scratchpad')
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); off = np.load(R / 'OOF_STEP_SCORES.npz')['offsets']; ns = np.diff(off); S = int(off[-1])
prm = (~ans.cell.str.startswith('pb_')).to_numpy(); fold = ans.fold.to_numpy(); prm_steps = np.repeat(prm, ns)
lv = np.load(TPF / 'DERIVATIVE_CHANNELS.npz'); names11 = list(map(str, lv['channels']))
prof = np.load(CT7P / 'profiles.npy').astype(float); pn = json.loads((CT7P / 'PROFILE_VALIDATION.json').read_text(encoding='utf8'))['channels']
raw13 = np.column_stack([lv['level'].astype(float), prof[:, pn.index('chosen_token_z_despiked')], lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)])
names13 = names11 + ['realized_z', 'realized_drv']; V13 = answer_standardize(raw13, off)
POOL = np.load(SCR / 'pool_z.npy'); PN = json.load(open(SCR / 'pool_names.json')); PS = pd.read_csv(ROOT / 'results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv').set_index('channel')
IND = json.loads((ROOT / 'results/indbank_lsml_prmbench_v1/PROTOCOL.json').read_text(encoding='utf8'))['selected_IND_21']
lf = np.array([float(np.sign(PS.loc[c, 'r_level_marginal'])) or 1.0 for c in IND])
ADD20 = ['ct7_chosen_std_excess', 'ct7_bocpd_residual', 'ct7_H0lim_prefix_innovation', 'ct7_ve0', 'H1_first_token', 'H1_slope', 'H1_jump', 'H1_frac_above_z', 'evidence_drop_risk']
LIVE = [j for j in range(POOL.shape[1]) if POOL[:, j].std() > 1e-12]
DF = np.load(ROOT / 'results/digit_family_extension_v1/FEATURES.npz'); assert np.array_equal(DF['offsets'], off)
D3 = digit_std(DF['values'], off, DF['active'])
BANKS = {'B13': (V13, names13), 'B16_digits': (np.column_stack([V13, D3]), names13 + list(DNAMES)),
         'B20': (answer_standardize(POOL[:, [PN.index(c) for c in names11 + ADD20]], off), names11 + ADD20),
         'B32': (answer_standardize(np.column_stack([POOL[:, :11], POOL[:, [PN.index(c) for c in IND]] * lf]), off), names11 + ['lf__' + c for c in IND]),
         'B51': (answer_standardize(POOL[:, LIVE], off), [PN[j] for j in LIVE])}
LEVEL = {'q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level'}   # used ONLY to name the level-containing groups in the residual test
def rows_of(mask): return np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(mask)])
def canon(g): return np.unique(np.asarray(g, int), return_inverse=True)[1]
def eig_stats(C):
    w = np.sort(np.linalg.eigvalsh(C))[::-1]; return float(w[0]), float(w[1]) if len(w) > 1 else 0.0
def union_table(Rm, g):
    G = g.max() + 1; out = []
    for a in range(G):
        for b in range(a + 1, G):
            idx = np.flatnonzero((g == a) | (g == b)); l1, l2 = eig_stats(Rm[np.ix_(idx, idx)])
            out.append({'a': a, 'b': b, 'p': len(idx), 'lambda2': l2, 'lambda1_over_p': l1 / len(idx)})
    return out
def kaiser_merge(Rm, g):
    g = g.copy(); seq = []
    while g.max() > 0:
        t = [r for r in union_table(Rm, g) if r['lambda2'] < 1]
        if not t: break
        r = min(t, key=lambda r: r['lambda2']); seq.append((r['a'], r['b'], round(r['lambda2'], 3), r['p'])); g[g == r['b']] = r['a']; g = canon(g)
    return g, seq
def absorb_merge(Rm, g, thr=0.5):
    """Size-free alternative: rho = lambda_2(union) / min(lambda_1(A), lambda_1(B)); rho = 1 for independent groups (the
    weaker group keeps its own direction), rho -> 0 when the weaker group is absorbed into one common direction.
    Merge the pair with the smallest rho while rho < thr."""
    g = g.copy(); seq = []
    while g.max() > 0:
        G = g.max() + 1; lam1 = [eig_stats(Rm[np.ix_(np.flatnonzero(g == a), np.flatnonzero(g == a))])[0] for a in range(G)]; best = None
        for a in range(G):
            for b in range(a + 1, G):
                idx = np.flatnonzero((g == a) | (g == b)); rho = eig_stats(Rm[np.ix_(idx, idx)])[1] / min(lam1[a], lam1[b])
                if best is None or rho < best[0]: best = (rho, a, b, len(idx))
        if best[0] >= thr: seq.append(('stop_at_rho', round(best[0], 3))); break
        seq.append((best[1], best[2], round(best[0], 3), best[3])); g[g == best[2]] = best[1]; g = canon(g)
    return g, seq
def show(g, nm): return ' | '.join('+'.join(nm[j] for j in np.flatnonzero(g == k)) for k in range(g.max() + 1))
key = lambda m: np.random.default_rng(20260928).random((S, m))
report = {}
for bk, (V, nm) in BANKS.items():
    m = V.shape[1]; votes = SB.random_tie_marks(V, off, .2, key(m)); Tta = tail_marks(V, off, .2, tie_aware=True, centred=True)[0]
    for k in [0, 2, 4]:                                   # three folds are enough to see the pattern; label-free
        cal = (k + 1) % 5; fr = rows_of(np.isin(fold, [f for f in range(5) if f not in (k, cal)])); pf = fr[prm_steps[fr]]
        est = SA.em_estimate(votes[pf], 'ds'); surv = np.flatnonzero(est['pi'] > 0.5); sn = [nm[j] for j in surv]
        for kind, X in (('continuous', V[fr][:, surv]), ('binary_tie_aware', Tta[fr][:, surv])):
            Xs = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1); Rm = np.corrcoef(Xs, rowvar=False)
            K, c, res, curve = TC.groups_from_R(Rm) if len(surv) > 4 else (None, None, None, [])
            g = canon(c)
            lvl_groups = sorted({int(g[j]) for j, x in enumerate(sn) if x in LEVEL})
            gm = g.copy()
            for h in lvl_groups[1:]: gm[gm == h] = lvl_groups[0]
            gm = canon(gm)
            res_merged = float(F._residual_lsml(Rm, gm, loading_scale='unit'))
            gk, seq = kaiser_merge(Rm, g)
            ga, seqa = absorb_merge(Rm, g)
            rec = {'K': int(K), 'residual_chosen': res, 'residual_level_merged': res_merged, 'level_groups': len(lvl_groups),
                   'curve': [(int(k_), round(float(r_), 4)) for k_, r_, _ in curve], 'merged_partition_proposed_at_some_K': any(np.array_equal(canon(c_), gm) or SB.canonical(c_) == SB.canonical(gm) for _, _, c_ in curve),
                   'chosen': show(g, sn), 'kaiser_merges': seq, 'after_kaiser': show(gk, sn), 'absorb_merges': seqa, 'after_absorb': show(ga, sn), 'union_table': union_table(Rm, g)}
            report[f'{bk}|fold{k}|{kind}'] = rec
            print(f"{bk:10s} fold{k} {kind:16s} K={K} level_groups={len(lvl_groups)} res chosen={res:.3f} merged={res_merged:.3f} "
                  f"merged@K={rec['merged_partition_proposed_at_some_K']}; kaiser={seq}; absorb={seqa}", flush=True)

Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=1, default=float), encoding='utf8')
