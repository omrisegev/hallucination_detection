"""Step 4: rebuild the teacher-forced ProcessBench B16 reference of the self-generated experiment on the think-fix banks.

The self-generated side (b16_fit on the GENERATED evdrop answers) does not use teacher forcing and is unchanged.
Its reference, per_dataset_fit_v1 run_20260929 STEP_SCORES (Step 462: B16 BASE / GRP fitted per cell), came from
the old teacher-forced cells. Its stored inputs (DERIVATIVE_CHANNELS level bank, ct7 profiles) no longer exist;
the same 13 raw channels + 3 digit channels are taken from the rebuilt external_banks_v4 source bank instead
(old rebuild == frozen bank to 6.1e-7): BANK11 level channels, realized_z (= ct7 chosen_token_z_despiked) and
realized_drv (= derivative of chosen_surprisal).
Uses b16_fit.build_bank / fit_cell / keys unchanged (claude/self-generated-step-labels-v1).
  1. replay: old bank -> must reproduce Step 462 B16 BASE/GRP on every cell (reported max abs diff);
  2. thinkfix and thinkfix_mask1 banks -> STEP_SCORES_<variant>.npz with the same keys/offsets.
    python step4_pb_reference.py <selfgen worktree> <banks root> <step462 STEP_SCORES.npz> <out dir>
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

SG = Path(sys.argv[1]); BANKS = Path(sys.argv[2]); REF = Path(sys.argv[3]); OUT = Path(sys.argv[4])
sys.path.insert(0, str(SG / 'scripts/self_generated_labels'))
import b16_fit as B  # noqa: E402  (HD_MAIN_CHECKOUT must point at a tree with .worktrees/ssl-pseudolabel-residual-v1)
from spectral_utils.external_generalization._bank11.claude_feature_bank_v1 import FEATURE_NAMES as BANK11  # noqa: E402

OUT.mkdir(parents=True, exist_ok=False)
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']


def load_bank(variant):
    z = np.load(BANKS / f'source_bank_{variant}' / 'FEATURES.npz'); nm = [str(x) for x in z['names']]
    raw13 = z['values'][:, [nm.index(c) for c in list(BANK11) + ['realized_z', 'realized_drv']]].astype(float)
    dv = z['values'][:, [nm.index(c) for c in DIG]].astype(float)
    return raw13, dv, z['digit_active'].astype(bool), z['offsets'].astype(np.int64), z['cells'].astype(str)


def fit_all(raw13, dv, da, off, cells):
    V = B.build_bank(raw13, dv, da, off); kb, kg = B.keys(int(off[-1]))
    step_cell = np.repeat(cells, np.diff(off)); base = np.full(len(V), np.nan); grp = np.full(len(V), np.nan); fits = {}
    for ce in sorted(set(cells)):
        rc = np.flatnonzero(step_cell == ce)
        b, g, d = B.fit_cell(V, off, rc, kb, kg); base[rc], grp[rc] = b[rc], g[rc]
        fits[ce] = {'survivors': len(d['survivors']), 'K': d['K']}
        print(ce, fits[ce], flush=True)
    return base, grp, fits


ref = np.load(REF); report = {'inputs': {'banks': str(BANKS), 'step462': str(REF)}, 'bank11_names': list(BANK11)}
t0 = time.time()
for variant in ('old_rebuild', 'thinkfix', 'thinkfix_mask1'):
    raw13, dv, da, off, cells = load_bank(variant)
    assert np.array_equal(off, ref['offsets']) if 'offsets' in ref.files else True
    base, grp, fits = fit_all(raw13, dv, da, off, cells)
    r = {'fits': fits}
    if variant == 'old_rebuild':
        step_cell = np.repeat(cells, np.diff(off))
        r['replay_max_abs_diff_vs_step462'] = {ce: {'BASE': float(np.nanmax(np.abs(base[step_cell == ce] - ref['B16__BASE'][step_cell == ce]))),
                                                    'GRP': float(np.nanmax(np.abs(grp[step_cell == ce] - ref['B16__GRP'][step_cell == ce])))}
                                               for ce in sorted(set(cells))}
        print('replay', json.dumps(r['replay_max_abs_diff_vs_step462']), flush=True)
    np.savez_compressed(OUT / f'STEP_SCORES_{variant}.npz', B16__BASE=base, B16__GRP=grp, offsets=off)
    report[variant] = r
report['elapsed_s'] = round(time.time() - t0)
(OUT / 'STEP4_PB_REFERENCE.json').write_text(json.dumps(report, indent=1))
print('STEP4_DONE', report['elapsed_s'], flush=True)
