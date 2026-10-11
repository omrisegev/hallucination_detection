"""tensor_mom_v1: does the third-moment method of moments (Jaffe, Nadler & Kluger, AISTATS 2015) estimate the channels'
sensitivity, specificity and prevalence better than the label-free estimators of expectation_realization_v1 stage A?

Same population, channels, top-20% within-answer marks, folds and fit rows as stage A (expectation_realization_run.py,
HISTORY Step 450): for eval fold k, cal (k+1)%5, the estimates are fitted on the PRMBench steps of the other three folds
and compared with the truth on those same steps.  Stage A's DS / HEM / SML numbers are read from its saved outputs (not
refitted); this run adds MoM and SML scaled by the MoM imbalance.  Hard stops: the inputs must hash equal to stage A's
INPUT_MANIFEST, and the per-fold truth must reproduce stage A's STAGE_A_CHANNELS.csv.  Labels enter only `truth`.

Run from the ssl worktree:  python scripts/experiments/tensor_mom_stage_a_run.py [run_id]
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; sys.path.insert(0, str(DEPTH))
from spectral_utils.lsml_gate_locator_research import answer_standardize  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import er_stage_a as SA  # noqa: E402

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_' + datetime.now().strftime('%Y%m%d')
STAGE_A = ROOT / 'results/expectation_realization_v1/run_20260927'
OUT = ROOT / 'results/tensor_mom_v1' / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
P = json.loads((ROOT / 'results/expectation_realization_v1/PROTOCOL.json').read_text(encoding='utf8'))
TPF = MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1'
CT7P = MAIN / '.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1'
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
INPUTS = {'level_bank': TPF / 'DERIVATIVE_CHANNELS.npz', 'ct7_profiles': CT7P / 'profiles.npy', 'ct7_profile_validation': CT7P / 'PROFILE_VALIDATION.json',
          'oof_answers': R / 'OOF_ANSWERS.csv', 'oof_step_scores': R / 'OOF_STEP_SCORES.npz'}


def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def hard_stop(reason):
    dump(OUT / 'RUN_STATUS.json', {'status': 'STOPPED', 'reason': reason, 'finished': datetime.now().isoformat(timespec='seconds')})
    raise SystemExit('HARD STOP: ' + reason)


manA = json.loads((STAGE_A / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))
hashes = {k: sha(v) for k, v in INPUTS.items()}
if any(hashes[k] != manA[k]['sha256'] for k in INPUTS):
    hard_stop('inputs differ from stage A: ' + ', '.join(k for k in INPUTS if hashes[k] != manA[k]['sha256']))

# ------------------------------------------------------------------ population and channels, exactly as stage A
ans = pd.read_csv(INPUTS['oof_answers'], encoding='utf-8-sig'); Zs = np.load(INPUTS['oof_step_scores'])
off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off); S = int(off[-1])
prm = ~ans.cell.str.startswith('pb_').to_numpy(); fold = ans.fold.to_numpy(); prm_steps = np.repeat(prm, ns)
lv = np.load(INPUTS['level_bank']); level = lv['level'].astype(float); names11 = list(map(str, lv['channels'])); assert level.shape == (S, 11)
drv = lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)
prof = np.load(INPUTS['ct7_profiles']).astype(float); pnames = json.loads(INPUTS['ct7_profile_validation'].read_text(encoding='utf8'))['channels']
if float(np.abs(prof.mean(1) - Zs['ct7']).max()) >= 1e-12: hard_stop('ct7 profile order')
names = names11 + ['realized_z', 'realized_drv']
if names != manA['channels']: hard_stop('channel list differs from stage A')
values = answer_standardize(np.column_stack([level, prof[:, pnames.index('chosen_token_z_despiked')], drv]), off); assert np.isfinite(values).all()
lab = np.array(manA['block_labels']); blocks = list(manA['declared_blocks'])
A0 = names.index('q15_H1')
votes, vote_info = SA.binary_votes(values, off, .2)


def rows_of(mask):
    return np.concatenate([np.arange(off[i], off[i + 1]) for i in np.flatnonzero(mask)])


# ------------------------------------------------------------------ estimates per fold
old = pd.read_csv(STAGE_A / 'STAGE_A_CHANNELS.csv'); stA = json.loads((STAGE_A / 'STAGE_A.json').read_text(encoding='utf8'))
rows, per_fold = [], {}
for k in range(5):
    cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]
    fit_rows = rows_of(np.isin(fold, fitf)); pf = fit_rows[prm_steps[fit_rows]]
    V = votes[pf]; tru = SA.truth(V, labels[pf])
    o = old[old.fold == k].set_index('channel').loc[names]
    if not (np.allclose(o.psi_true, tru['psi'], atol=1e-12) and np.allclose(o.eta_true, tru['eta'], atol=1e-12)):
        hard_stop(f'fold {k}: truth does not reproduce stage A')
    mom = SA.tensor_mom_estimate(V, anchor=A0)
    sml_mom = SA.sml_estimate(V, anchor=A0, b_hat=mom['b'])
    per_fold[k] = {'rows': int(len(pf)), 'prevalence_true': tru['prevalence'], 'MoM': {kk: mom[kk] for kk in ('prevalence', 'b', 'alpha', 'lambda', 'out_of_range')},
                   'bar_MoM': SA.bar(mom, tru), 'bar_SML_with_MoM_b': SA.bar(sml_mom, tru, prev_tol=None),
                   'kept_MoM': [names[j] for j in np.flatnonzero(mom['pi'] > 0.5)],
                   'kept_DS': [names[j] for j in np.flatnonzero(o.pi_DS.to_numpy() > 0.5)]}
    for j, c in enumerate(names):
        rows.append({'fold': k, 'channel': c, 'block': blocks[lab[j]], 'psi_true': tru['psi'][j], 'eta_true': tru['eta'][j], 'pi_true': tru['pi'][j],
                     'psi_MoM': mom['psi'][j], 'eta_MoM': mom['eta'][j], 'pi_MoM': mom['pi'][j], 't_MoM': mom['t'][j], 'pi_SML_MoM_b': sml_mom['pi'][j],
                     **{f'{q}_{e}': o[f'{q}_{e}'].iloc[j] for e in ('DS', 'HEM') for q in ('psi', 'eta', 'pi')}, 'pi_SML_DS_b': o.pi_SML.iloc[j]})
    print(f"fold {k}: prevalence true {tru['prevalence']:.3f} MoM {mom['prevalence']:.3f}; MoM bar passes={per_fold[k]['bar_MoM']['passes']}; "
          f"filter MoM == DS: {per_fold[k]['kept_MoM'] == per_fold[k]['kept_DS']}", flush=True)
pd.DataFrame(rows).to_csv(OUT / 'CHANNELS.csv', index=False)

# ------------------------------------------------------------------ side by side with stage A (same folds, same rows)
cmp = {}
for e, src in [('MoM', None), ('DS', 'DS'), ('HEM', 'HEM')]:
    bars = [per_fold[k]['bar_MoM'] for k in range(5)] if src is None else [b for b in stA[src]['per_fold'] if b]
    cmp[e] = {m: float(np.mean([b[m] for b in bars if m in b])) for m in ('prevalence_error', 'mae_psi', 'mae_eta', 'mae_pi', 'spearman_pi')} | {'passes_folds': int(sum(b['passes'] for b in bars))}
dump(OUT / 'SUMMARY.json', {'status': 'COMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'vote_info': vote_info,
                            'input_sha256': hashes, 'compare_mean_over_folds': cmp, 'per_fold': per_fold, 'channels': names})
print(pd.DataFrame(cmp).T.round(4).to_string())
