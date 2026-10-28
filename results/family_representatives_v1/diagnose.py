"""Post-evaluation diagnostic (uses labels; explanation only, not a selection input).

Q: do the new banks lose information, or lose weighting? Per-column within-AUC, and the
weight each original channel gets inside plain averaging / L-SML, mapped back to N56.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from spectral_utils.advisor_statistics import answer_auc  # noqa: E402
from spectral_utils.advisor_lr import fit_labelfree  # noqa: E402
import transforms as T  # noqa: E402
from evaluate import load_inputs  # noqa: E402

X, names, off, P, _ = load_inputs(); y = P['y'].astype(bool); cov = T.step_covariates(off, P['token_lengths'])
out = {}
for arm, res in (('N56', None), ('A_copies_pc1', False), ('B_copies_pc1_lenpos', True)):
    if res is None:
        Z, nm = X, names
    else:
        tf = T.FrozenT(res).fit(X, names, off, cov); Z, nm = tf.transform(X, off, cov), tf.names_out
    auc = [float(np.nanmean(answer_auc(y, Z[:, j], off)[0])) for j in range(Z.shape[1])]
    m = fit_labelfree(Z, 'LSML_CONT'); w = m.weight * Z.std(0); share = np.abs(w) / np.abs(w).sum()
    out[arm] = dict(columns=[dict(name=n, within_auc=round(a, 4), lsml_weight=round(float(ww), 4), lsml_share=round(float(s), 4))
                             for n, a, ww, s in zip(nm, auc, w, share)])
# weight of the uncertainty-level channels (the 9-member entropy/varentropy block + energy_level) under each fusion
lev_block = next(b['members'] for b in T.FrozenT(False).fit(X, names, off).describe()['blocks'] if len(b['members']) == 9)
for arm in out:
    cols = out[arm]['columns']
    is_lev = [any(mb in c['name'] for mb in lev_block) for c in cols]
    out[arm]['uncertainty_block_eq_share'] = float(np.mean(is_lev))
    out[arm]['uncertainty_block_lsml_share'] = float(sum(c['lsml_share'] for c, l in zip(cols, is_lev) if l))
    out[arm]['energy_level_lsml_share'] = float(sum(c['lsml_share'] for c in cols if c['name'].startswith('energy_level')))
out['uncertainty_block_members'] = lev_block
(HERE / 'evaluation' / 'DIAGNOSE.json').write_text(json.dumps(out, indent=1))
for arm in ('N56', 'A_copies_pc1', 'B_copies_pc1_lenpos'):
    d = out[arm]; top = sorted(d['columns'], key=lambda c: -abs(c['within_auc'] - .5))[:6]
    print(arm, 'EQ share unc-block', round(d['uncertainty_block_eq_share'], 3), 'LSML share unc-block', round(d['uncertainty_block_lsml_share'], 3),
          'energy_level', round(d['energy_level_lsml_share'], 3))
    print('   top AUC:', [(c['name'][:38], c['within_auc'], c['lsml_share']) for c in top])
