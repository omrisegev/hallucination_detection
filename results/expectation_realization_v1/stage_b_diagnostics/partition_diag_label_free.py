# LABEL-FREE diagnostic: which input choice (rows, tie handling) makes the mark partition equal the families?
import sys, json, numpy as np, pandas as pd
from pathlib import Path
W = Path(r'C:\Users\omris\TAU\hallucination_detection\.worktrees')
sys.path.insert(0, str(W / 'depth-feature-fusion-v1')); sys.path.insert(0, str(W / 'ssl-pseudolabel-residual-v1/scripts/experiments'))
from spectral_utils.lsml_gate_locator_research import answer_standardize
import tail_calib_common as TC, er_stage_b as SB
from calfix_common import tail_marks
R = W / 'readout-quickest-detection-v1/results/step_evidence_v1'; TPF = W / 'token-probability-fusion-v1/results/token_probability_fusion_v1'
CT7P = W / 'cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1'
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Zs = np.load(R / 'OOF_STEP_SCORES.npz'); off = Zs['offsets']; ns = np.diff(off); S = int(off[-1])
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); prm_steps = np.repeat(prm, ns)
lv = np.load(TPF / 'DERIVATIVE_CHANNELS.npz'); names11 = list(map(str, lv['channels']))
prof = np.load(CT7P / 'profiles.npy').astype(float); pn = json.loads((CT7P / 'PROFILE_VALIDATION.json').read_text(encoding='utf8'))['channels']
raw = np.column_stack([lv['level'].astype(float), prof[:, pn.index('chosen_token_z_despiked')], lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)])
names = names11 + ['realized_z', 'realized_drv']; V = answer_standardize(raw, off)
def rows_of(mask): return np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(mask)])
key = np.random.default_rng(20260928).random((S, 13))
M = {'random_tie': SB.answer_center((SB.random_tie_marks(V, off, .2, key) > 0).astype(float), off),
     'tie_aware': tail_marks(V, off, .2, tie_aware=True, centred=True)[0],
     'position_tie': tail_marks(V, off, .2, tie_aware=False, centred=True)[0]}
fam = [0,0,2,0,0,0,1,1,1,1,1,2,2]
def show(g): return ' | '.join('+'.join(names[j] for j in range(13) if g[j] == c) for c in sorted(set(g), key=lambda c: min(np.flatnonzero(np.asarray(g) == c))))
drop = [names.index('energy_innovation'), names.index('top50_js')]; keep = [j for j in range(13) if j not in drop]
for k in range(5):
    cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]; fr = rows_of(np.isin(fold, fitf)); pf = fr[prm_steps[fr]]
    for mk, T in M.items():
        for rn, rr in [('pooled', fr), ('prm', pf)]:
            g = TC.lsml_fit_scaled(T[rr], 0, V[rr], standardize=True, loading_scale='unit')['groups']
            gk = TC.lsml_fit_scaled(T[rr][:, keep], 0, V[rr][:, keep], standardize=True, loading_scale='unit')['groups']
            same_fam = SB.canonical(g) == SB.canonical(fam); stable = SB.canonical(np.asarray(g)[keep]) == SB.canonical(gk)
            print(f"fold {k} {mk:12s} {rn:6s} K11={len(set(gk))} :: " + " | ".join("+".join(names[keep[j]] for j in range(11) if gk[j] == c) for c in sorted(set(gk), key=lambda c: min(np.flatnonzero(np.asarray(gk) == c)))), flush=True)
