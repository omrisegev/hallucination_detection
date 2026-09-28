import sys, time, json, warnings; sys.path.insert(0, '.')
from pathlib import Path
import numpy as np, pandas as pd
from harness import drr, make_env
R = Path(r'C:/Users/omris/TAU/hallucination_detection/.worktrees/readout-quickest-detection-v1/results/step_evidence_v1')
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); off = np.load(R / 'OOF_STEP_SCORES.npz')['offsets']
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); S_ = int(off[-1])
env = make_env(off, fold, prm)
rng3 = np.random.default_rng(20261004)
for b in range(3):
    Xr = rng3.standard_normal((S_, 11)); Sr = rng3.standard_normal(S_); t = time.perf_counter(); rec = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')      # surface any RuntimeWarning (NaN/div0) as an error
            fr, post, G = env['build_rules'](Xr, Sr, None, drr.DS_SEED, record=rec)
        print('draw', b, 'OK', round(time.perf_counter() - t, 1), 's; posterior finite', bool(np.isfinite(post).all()),
              '; DS prevalence by fold', [round(r['ds_prevalence'], 3) for r in rec.values()], '; converged', [r['ds_converged'] for r in rec.values()],
              '; r4 cutoffs', [round(r['r4_cutoff'], 3) for r in rec.values()], '; closed-form gate', max(r['closed_form_vs_model_posterior_max_abs'] for r in rec.values()))
    except Exception as e:
        print('draw', b, 'EXCEPTION', repr(e)[:300])
