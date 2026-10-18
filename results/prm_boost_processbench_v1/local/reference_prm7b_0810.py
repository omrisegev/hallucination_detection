"""Evaluator check on real data: score file from the 2026-08-10 Qwen2.5-Math-PRM-7B ProcessBench run (job 177776).

Builds SCORES_prm7b_0810.npz in the score_external.py layout (risk = logit(1 - reward), native = reward > 0.5,
every step non-empty) from dataset_cache/four_localization/pb_prm_qwen25math7b_full, runs eval_processbench.py
without bootstrap, and checks the official per-subset F1 against that run's manifest. Also gives the PRM-only
reference row (pooled step AUROC etc.) and, after the new run, the reward parity anchor for qwen_prm.

    .venv/bin/python results/prm_boost_processbench_v1/local/reference_prm7b_0810.py <out dir>
"""
import json, pickle, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path('/Users/osegev/Desktop/hallucination_detection/dataset_cache/four_localization/pb_prm_qwen25math7b_full')
out = Path(sys.argv[1]); (out / 'scores').mkdir(parents=True, exist_ok=True)
rec = {}
for s in ('gsm8k', 'math', 'olympiadbench', 'omnimath'):
    for r in pickle.load(open(SRC / f'pb_prm_{s}.pkl', 'rb')).values():
        rec[r['id']] = r
uids = sorted(rec)
rew = np.concatenate([np.asarray(rec[u]['rewards'], float) for u in uids])
lens = np.array([len(rec[u]['rewards']) for u in uids])
p = np.clip(rew, 1e-6, 1 - 1e-6)
np.savez(out / 'scores/SCORES_prm7b_0810.npz', uids=np.array(uids), nonempty=np.ones(len(rew), bool),
         nonempty_lengths=lens, native_valid=(rew > .5).astype(int), score__PRM=np.log((1 - p) / p))
subprocess.run([sys.executable, str(HERE / 'eval_processbench.py'), str(out / 'scores'), str(out / 'eval'),
                '--no-bootstrap'], check=True)
man = json.loads((SRC / 'manifest.json').read_text())['cells']
import csv
rows = list(csv.DictReader(open(out / 'eval/SUBSETS_processbench.csv')))
for arm in ('PRM_native', 'PRM_native_adapter'):
    sub = {r['subset']: r for r in rows if r['arm'] == arm}
    gaps = {s: round(abs(float(sub[s]['f1']) - man[s]['f1']), 12) for s in man}
    print(arm, 'official F1 gap vs job 177776 manifest:', gaps)
    if arm == 'PRM_native':
        assert max(gaps.values()) < 1e-9, gaps
print('PASS: PRM_native reproduces the recorded official F1 on all 4 subsets')
