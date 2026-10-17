"""family_tail_calfix_v1: calibration-corrected replay of the named-family, tail-weighted and bank
runs (handoff 2026-09-24, section 2 + 8).  Protocol: results/family_tail_calfix_v1/PROTOCOL.json.

Every arm is refitted per outer fold on the three fit folds; one fitted model scores its calibration
fold and its evaluation fold into write-once arrays (calfix_common.ScoreBundle); calfix_evaluate
calibrates thresholds on the model's own calibration scores only.

    python -B scripts/experiments/family_tail_calfix_run.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from calfix_common import MAIN, Population, ScoreBundle, dump, extgen_hashes, load_extgen, lsml_fit, partition_equal, roles_of, sha, tail_marks  # noqa: E402
import calfix_evaluate as EV  # noqa: E402

DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; sys.path.insert(0, str(DEPTH))
from spectral_utils.fusion_utils import sml_fuse_signed  # noqa: E402
from spectral_utils.lsml_gate_locator_research import _orient  # noqa: E402

STAGE = ROOT / 'results/family_tail_calfix_v1'; OUT = STAGE / f'run_{datetime.now():%Y%m%d_%H%M}'; OUT.mkdir(parents=True, exist_ok=False)
P = json.loads((STAGE / 'PROTOCOL.json').read_text(encoding='utf8'))
SCR = Path('C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad')
POOL_SHA = 'd9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16'
SPLITS = json.loads((ROOT / 'results/named_group_fusion_v1/PROTOCOL.json').read_text(encoding='utf8'))['splits']
OLD_DESIGN = json.loads((ROOT / 'results/named_group_fusion_v1/run_20260924/DESIGN.json').read_text(encoding='utf8'))
DROP = ['ct7_ve1', 'hist_entropy_series', 'hist_spilled_series', 'hist_trace_length_series']
CT7C = ['ct7_H0lim', 'ct7_ve0', 'ct7_ve0.75', 'ct7_ve1', 'ct7_H0lim_prefix_innovation', 'ct7_bocpd_residual', 'ct7_chosen_std_excess']
T0 = time.perf_counter()
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds')}; dump(OUT / 'RUN_STATUS.json', status)

pop = Population(); off = pop.off
assert sha(SCR / 'pool_z.npy') == POOL_SHA
POOL = np.load(SCR / 'pool_z.npy'); PN = json.loads((SCR / 'pool_names.json').read_text(encoding='utf8')); assert POOL.shape == (pop.total, len(PN))
names = [c for c in PN if c not in DROP]
Z = pop.answer_standardize(POOL[:, [PN.index(c) for c in names]])
B11N = PN[:11]

# ------------------------------------------------------------------ retrospective label-using design (all folds), identical to the superseded runs
elig = np.flatnonzero(pop.eligible)
AUCM = np.array([[EV.within_auc(pop.labels[off[i]:off[i + 1]], Z[off[i]:off[i + 1], j]) for j in range(len(names))] for i in elig])
def design(answer_mask):
    a = AUCM[answer_mask[elig]].mean(0); s = np.where(a >= .5, 1.0, -1.0)
    return s, [c for c, x in zip(names, a) if max(x, 1 - x) >= .60], a
sign, kept, chan_auc = design(np.ones(pop.n, bool))
assert {c: int(v) for c, v in zip(names, sign)} == OLD_DESIGN['orientation'] and kept == OLD_DESIGN['kept'], 'retrospective design differs from the superseded run'
for G in SPLITS.values():
    assert sorted(sum(G.values(), [])) == sorted(kept)
Zo = Z * sign; ix = {c: j for j, c in enumerate(names)}

def fam_features(G, s):
    """equal mean of the (s-oriented) members of each family, answer-standardized; members absent from s are skipped."""
    cols, used = [], []
    for g, mem in G.items():
        mem = [c for c in mem if c in s]
        if mem:
            cols.append(np.column_stack([Z[:, ix[c]] * s[c] for c in mem]).mean(1)); used.append((g, mem))
    return pop.answer_standardize(np.column_stack(cols)), used

S3, S4 = SPLITS['S3_M15'], SPLITS['S4_M16']; sg = dict(zip(names, sign))
F15, _ = fam_features(S3, sg); F16, _ = fam_features(S4, sg)
F15_anchor = list(S3).index('level_entropy'); F16_anchor = list(S4).index('entropy')
X = {'B11': Z[:, [ix[c] for c in B11N]], 'B11o': Zo[:, [ix[c] for c in B11N]], 'CT7s': pop.answer_standardize(POOL[:, [PN.index(c) for c in CT7C]]),
     'A48n': Z, 'A48o': Zo, 'K28': Zo[:, [ix[c] for c in kept]], 'F15': F15, 'F16': F16}
ANCHOR = {'B11': 0, 'B11o': 0, 'CT7s': 0, 'A48n': ix['q15_H1'], 'A48o': ix['q15_H1'], 'K28': kept.index('q15_H1'), 'F15': F15_anchor, 'F16': F16_anchor}
MARKS, DEGEN = {}, {}
for b in ('F15', 'F16', 'B11', 'B11o', 'CT7s'):
    for tk, frac in (('tail20', .2), ('tail1', None)):
        MARKS[b, tk], DEGEN[b, tk] = tail_marks(X[b], off, frac, tie_aware=False)
for b in ('F15', 'K28', 'A48o', 'B11'):
    MARKS[b, 'tailtie'], DEGEN[b, 'tailtie'] = tail_marks(X[b], off, .2, tie_aware=True)
MARKS['F15', 'tailtie_votes'], _ = tail_marks(X['F15'], off, .2, tie_aware=True, centred=False)
print('design, family features and marks built', f'{time.perf_counter()-T0:.0f}s', flush=True)
dump(OUT / 'INPUT_MANIFEST.json', {'pool_z': {'path': str(SCR / 'pool_z.npy'), 'sha256': POOL_SHA}, 'pool_names': sha(SCR / 'pool_names.json'),
      **{k: {'path': str(v), 'sha256': sha(v)} for k, v in pop.inputs.items()}, 'extgen': extgen_hashes(),
      'script_sha256': sha(Path(__file__)), 'common_sha256': sha(HERE / 'calfix_common.py'), 'evaluator_sha256': sha(HERE / 'calfix_evaluate.py'),
      'protocol_sha256': sha(STAGE / 'PROTOCOL.json'), 'families_protocol_sha256': sha(ROOT / 'results/named_group_fusion_v1/PROTOCOL.json')})
dump(OUT / 'DESIGN.json', {'retrospective': {'orientation': dict(zip(names, sign.astype(int))), 'prm_auc': dict(zip(names, chan_auc)), 'kept': kept},
                           'mark_degeneracy': {f'{b}_{t}': v for (b, t), v in DEGEN.items()}})

# ------------------------------------------------------------------ fits
B = ScoreBundle(pop); failures = []; bridge = {'extgen_fit_weights_max_abs_diff': 0.0}
codex = json.loads((MAIN / 'results/lsml_external_generalization_v1/evaluation/source/VALIDATION_FITS.json').read_text())
w421 = (lambda blk: 1.0 / (3 * np.bincount(blk)[blk]))(np.array([0, 0, 0, 0, 1, 1, 2]))

def emit(method, k, raw, record):
    B.put_full(method, k, pop.answer_z(raw), record)

def lsml_rec(fit, cols):
    return {'weights': dict(zip(cols, np.round(fit['weights'], 10))), 'groups': dict(zip(cols, fit['groups'].tolist())), 'K': fit['K'],
            'anchor_spearman': fit['anchor_spearman'], 'anchor_flipped': fit['anchor_flipped'], 'residual': fit['residual'], 'small_m_guarded': fit['small_m_guarded']}

def safe_lsml(method, k, Xfit, anchor, orient=None):
    try:
        return lsml_fit(Xfit, anchor, orient), None
    except ValueError as e:                                   # visible, never silent: recorded and the arm is marked
        failures.append({'method': method, 'outer_fold': k, 'reason': str(e)}); return None, str(e)

def sml_w(Vfit, anchor):
    _, v = sml_fuse_signed(*Vfit.T, small_m_guard=True); w, o = _orient(Vfit, np.asarray(v, float), anchor); return w

def run_bank(b, k, fit_rows, Xb, anchor, cols, rules, marks=None, prefix=None, extra=None):
    """rules subset of equal, cov_lsml, tail20_lsml, tail1_lsml, tailtie_lsml, cov_sml, tail20_sml, tail1_sml, tailtie_partition_equal, tailtie_votes, partition_equal"""
    pre = prefix or b; fits = {}; extra = extra or {}
    def emit(method, k, raw, record):
        B.put_full(method, k, pop.answer_z(raw), {**record, **extra})
    for r in rules:
        if r == 'equal':
            emit(f'{pre}_equal', k, Xb.mean(1), {'rule': 'equal', 'columns': cols})
        elif r.endswith('_lsml'):
            mk = r[:-5]
            Tfit = Xb[fit_rows] if mk == 'cov' else marks[mk][fit_rows]
            fit, err = safe_lsml(f'{pre}_{r}', k, Tfit, anchor, None if mk == 'cov' else Xb[fit_rows])
            if fit is None:
                continue
            fits[mk] = fit; emit(f'{pre}_{r}', k, Xb @ fit['weights'], {'rule': r, **lsml_rec(fit, cols)})
        elif r.endswith('_sml'):
            mk = r[:-4]; Tfit = Xb[fit_rows] if mk == 'cov' else marks[mk][fit_rows]
            _, v = sml_fuse_signed(*Tfit.T, small_m_guard=True); w, o = _orient(Xb[fit_rows], np.asarray(v, float), anchor)
            emit(f'{pre}_{r}', k, Xb @ w, {'rule': r, 'weights': dict(zip(cols, np.round(w, 10))), **o})
    if 'partition_equal' in rules and 'cov' in fits:
        w = partition_equal(fits['cov']['groups']); emit(f'{pre}_partition_equal', k, Xb @ w, {'rule': 'partition_equal of cov_lsml groups', 'weights': dict(zip(cols, w))})
    if 'tailtie_partition_equal' in rules and 'tailtie' in fits:
        w = partition_equal(fits['tailtie']['groups']); emit(f'{pre}_tailtie_partition_equal', k, Xb @ w, {'rule': 'partition_equal of tailtie_lsml groups', 'weights': dict(zip(cols, w))})
    if 'tailtie_votes' in rules and 'tailtie' in fits:
        emit(f'{pre}_tailtie_votes', k, marks['tailtie_votes'] @ fits['tailtie']['weights'], {'rule': 'tailtie_lsml weights on uncentred tie-aware marks', 'weights': dict(zip(cols, fits['tailtie']['weights']))})
    return fits

fz = load_extgen()
for k in range(5):
    t = time.perf_counter(); fit_f, cal, ev = roles_of(k); fit_rows = pop.rows(np.isin(pop.fold, fit_f))
    mk = lambda b: {m: MARKS[bb, m] for (bb, m) in MARKS if bb == b}
    # --- transfer list, decomposition, leading-candidate controls
    f = run_bank('B11', k, fit_rows, X['B11'], 0, B11N, ['equal', 'cov_lsml', 'partition_equal', 'tail20_lsml', 'tail1_lsml', 'tailtie_lsml'], mk('B11'))
    ref = fz.fit_weights(X['B11'][fit_rows])                    # frozen external implementation, same rows
    bridge['extgen_fit_weights_max_abs_diff'] = max(bridge['extgen_fit_weights_max_abs_diff'], float(np.abs(np.array(ref['weights']) - f['cov']['weights']).max()))
    cx = [r for r in codex if r['test'] == k][0]
    bridge[f'codex_fold{k}_weights_max_abs_diff'] = float(np.abs(np.array(cx['fit']['weights']) - f['cov']['weights']).max())
    bridge[f'codex_fold{k}_groups_equal'] = cx['fit']['groups'] == f['cov']['groups'].tolist()
    run_bank('F15', k, fit_rows, X['F15'], F15_anchor, list(S3), ['equal', 'cov_lsml', 'tailtie_lsml', 'tailtie_partition_equal', 'tailtie_votes', 'tail20_lsml', 'tail1_lsml', 'cov_sml', 'tail20_sml', 'tail1_sml'], mk('F15'))
    run_bank('K28', k, fit_rows, X['K28'], ANCHOR['K28'], kept, ['equal', 'cov_lsml', 'tailtie_lsml'], mk('K28'))
    run_bank('A48o', k, fit_rows, X['A48o'], ANCHOR['A48o'], names, ['equal', 'cov_lsml', 'tailtie_lsml'], mk('A48o'))
    run_bank('A48n', k, fit_rows, X['A48n'], ANCHOR['A48n'], names, ['equal', 'cov_lsml'])
    # --- historical reproduction (other banks and within-family SML)
    run_bank('F16', k, fit_rows, X['F16'], F16_anchor, list(S4), ['equal', 'cov_lsml', 'tail20_lsml', 'tail1_lsml', 'cov_sml', 'tail20_sml', 'tail1_sml'], mk('F16'))
    run_bank('B11o', k, fit_rows, X['B11o'], 0, B11N, ['equal', 'cov_lsml', 'tail20_lsml', 'tail1_lsml'], mk('B11o'))
    run_bank('CT7s', k, fit_rows, X['CT7s'], 0, CT7C, ['equal', 'cov_lsml', 'tail20_lsml', 'tail1_lsml'], mk('CT7s'))
    emit('CT7s_block421', k, X['CT7s'] @ w421, {'rule': 'declared 4/2/1 block-equal', 'weights': dict(zip(CT7C, w421))})
    for nm, G, anc in (('F15w', S3, F15_anchor), ('F16w', S4, F16_anchor)):  # SML within families (>=3 members), then between
        cols, wlog = [], {}
        for g, mem in G.items():
            Xm = np.column_stack([Zo[:, ix[c]] for c in mem])
            if len(mem) >= 3:
                _, w = sml_fuse_signed(*Xm[fit_rows].T, small_m_guard=True); w = np.asarray(w, float); w = -w if w.sum() < 0 else w
            else:
                w = np.ones(len(mem)) / len(mem)
            wlog[g] = dict(zip(mem, np.round(w / np.abs(w).sum(), 6))); cols.append(Xm @ w)
        V = pop.answer_standardize(np.column_stack(cols))
        emit(f'{nm}_sml_equal', k, V.mean(1), {'rule': 'within SML, between equal', 'within': wlog})
        emit(f'{nm}_sml_sml', k, V @ sml_w(V[fit_rows], anc), {'rule': 'within SML, between SML', 'within': wlog})
        fit, err = safe_lsml(f'{nm}_sml_lsml', k, V[fit_rows], anc)
        if fit is not None:
            emit(f'{nm}_sml_lsml', k, V @ fit['weights'], {'rule': 'within SML, between L-SML', 'within': wlog, **lsml_rec(fit, list(G))})
    # --- in-fold selection sensitivity: orientation and AUC filter from the fit folds' PRMBench answers only
    s_if, kept_if, a_if = design(np.isin(pop.fold, fit_f)); sgi = dict(zip(names, s_if))
    rec_if = {'orientation_flips_vs_retrospective': [c for c in names if sgi[c] != sg[c]], 'kept': kept_if,
              'kept_added_vs_retrospective': sorted(set(kept_if) - set(kept)), 'kept_dropped_vs_retrospective': sorted(set(kept) - set(kept_if))}
    emit('A48if_equal', k, (Z * s_if).mean(1), {'rule': 'equal, in-fold orientation', **rec_if})
    if 'q15_H1' in kept_if:
        Xk = (Z * s_if)[:, [ix[c] for c in kept_if]]; Tk, dk = tail_marks(Xk, off, .2, tie_aware=True)
        run_bank('K28if', k, fit_rows, Xk, kept_if.index('q15_H1'), kept_if, ['equal', 'cov_lsml', 'tailtie_lsml'], {'tailtie': Tk}, extra={**rec_if, 'mark_degeneracy': dk})
        Fi, used = fam_features(S3, {c: sgi[c] for c in kept_if}); anc_if = [g for g, _ in used].index('level_entropy')
        Ti, di = tail_marks(Fi, off, .2, tie_aware=True)
        run_bank('F15if', k, fit_rows, Fi, anc_if, [g for g, _ in used], ['equal', 'cov_lsml', 'tailtie_lsml'], {'tailtie': Ti}, extra={**rec_if, 'in_fold_families': used, 'mark_degeneracy': di})
    else:
        failures.append({'method': 'K28if/F15if', 'outer_fold': k, 'reason': 'anchor q15_H1 not kept in fold'})
    # --- references (fixed scores; the same frozen vector for both roles)
    emit('ct7', k, pop.ct7, {'rule': 'frozen CT7 step scores, answer-z', 'sha256_source': sha(pop.inputs['oof_step_scores'])})
    B.put_full('ct7_raw', k, pop.ct7, {'rule': 'frozen CT7 step scores, historical scale (no answer-z)'})
    print(f'fold {k} done ({time.perf_counter()-t:.0f}s); failures so far {len(failures)}', flush=True)

bridge['status'] = 'PASS' if bridge['extgen_fit_weights_max_abs_diff'] < 1e-12 and all(v < 1e-9 for kk, v in bridge.items() if kk.endswith('weights_max_abs_diff')) and all(v for kk, v in bridge.items() if kk.endswith('groups_equal')) else 'FAIL'
dump(OUT / 'BRIDGE_B11.json', bridge); assert bridge['status'] == 'PASS', bridge
checks = B.finalize(OUT); dump(OUT / 'FAILURES.json', failures)
print('bundle', checks['status'], 'methods', checks['methods'], f'{time.perf_counter()-T0:.0f}s', flush=True)

# ------------------------------------------------------------------ evaluation (labels enter here only)
import family_tail_calfix_eval as FE  # noqa: E402
FE.evaluate_run(OUT, pop, P, status, T0, n_failures=len(failures))
