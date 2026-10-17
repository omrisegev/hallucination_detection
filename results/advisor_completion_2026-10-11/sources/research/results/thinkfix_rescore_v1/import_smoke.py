"""Import smoke for the downstream mirror: loads the frozen dev_run and cont scripts' module-level
code (their inputs, adapter, CODE modules) without running main()."""
import importlib.util, sys
from pathlib import Path
M = Path(__file__).resolve().parents[2]
for name, rel in (('dev_run', 'results/monday_nested_banks_v2/dev_run.py'), ('cont', 'results/monday_cont_baselines_v1/run.py')):
    spec = importlib.util.spec_from_file_location(name, M / rel); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    print('loaded', rel, 'FE =', getattr(getattr(mod, 'ext', None) or getattr(mod, 'dr').ext, 'FE'))
fr = __import__('json').loads((M / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
import pickle, numpy as np
print('prm meta rows', len(pickle.load(open(fr['prm_metadata']['path'], 'rb'))))
print('IMPORT SMOKE OK')
