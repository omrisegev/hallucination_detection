"""Freeze the two candidates before any label is read (spec section 12, step 3)."""
import hashlib, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parents[1]
import sys; sys.path.insert(0, str(HERE))
import transforms as T  # noqa: E402

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
D = ROOT / 'results/advisor_completion_2026-10-11/inputs'
Zf = np.load(D / 'bank_V_N56.npz'); P = np.load(D / 'population.npz'); off = Zf['offsets']; names = [str(c) for c in Zf['columns']]
cov = T.step_covariates(off, P['token_lengths'])
full = {k: T.FrozenT(r).fit(Zf['V'], names, off, cov).describe() for k, r in (('A_copies_pc1', False), ('B_copies_pc1_lenpos', True))}
freeze = dict(
    status='FROZEN_BEFORE_LABEL_EVALUATION', date='2026-10-20', labels_read=False,
    candidates={'A_copies_pc1': 'FrozenT(residualize=False)', 'B_copies_pc1_lenpos': 'FrozenT(residualize=True)'},
    selection='chosen from label-free structure metrics only (METRICS_A_STEPS.json); Omri asked to evaluate both, once',
    evaluation_protocol='5 source-group outer folds (population.npz fold = FOLDS_V2 prmbench.outer); T and every fusion fitted on training steps only; Q80 flag threshold from training answer-z scores; official PRMBench evaluator',
    decision_rule='Z beats N56 if L-SML within-AUC paired source-group 95% CI > 0 and PRMScore does not fall beyond its CI; S1/S3 must move the same way (spec section 11)',
    not_included='L-SML+HEM (tail-vote pipeline needs its own data environment; not run here)',
    code={p: sha(HERE / p) for p in ('transforms.py', 'metrics_a.py', 'freeze.py')},
    inputs={str(p.relative_to(ROOT)): sha(p) for p in (D / 'bank_V_N56.npz', D / 'population.npz', D / 'prmbench_meta.json', D / 'prmbench_prm.pkl')},
    full_data_fit_description=full)
(HERE / 'FREEZE.json').write_text(json.dumps(freeze, indent=1))
for k, d in full.items(): print(k, d['n_out'], 'copies', len(d['copies_dropped']), 'blocks', [len(b['members']) for b in d['blocks']])
