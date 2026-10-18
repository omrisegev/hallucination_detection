"""Parity anchor: qwen_prm step rewards of the new run vs the 2026-08-10 ProcessBench run (job 177776).

Both use Qwen2.5-Math-PRM-7B, the PRM_SYSTEM_PROMPT template and raw steps joined by <extra_0>; the 08-10 run came
from cluster/run_processbench_prm.py, the new one from cluster/run_prmbench_prm_telemetry.py --adapter qwen_prm.
Reports max |reward gap|, the number of answers whose ProcessBench first-error prediction (reward < 0.5) changes,
and the per-subset official F1 of both. No labels are needed for the reward gap; F1 uses the 08-10 pkls' labels.

    python parity_qwen_prm_rewards.py <full_qwen_prm/bank_v2/REWARDS.npz> [--pkl-dir <pb_prm_qwen25math7b_full>]
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('rewards', type=Path)
ap.add_argument('--pkl-dir', type=Path, default=Path(
    '/Users/osegev/Desktop/hallucination_detection/dataset_cache/four_localization/pb_prm_qwen25math7b_full'))
a = ap.parse_args()
R = np.load(a.rewards); uids = [str(u) for u in R['uids']]; lens = R['nonempty_lengths']
starts = np.r_[0, np.cumsum(lens)]
new = {u: R['rewards'][starts[i]:starts[i + 1]] for i, u in enumerate(uids)}
old = {}
for s in ('gsm8k', 'math', 'olympiadbench', 'omnimath'):
    for r in pickle.load(open(a.pkl_dir / f'pb_prm_{s}.pkl', 'rb')).values():
        old[r['id']] = r


def first(rw):
    f = np.flatnonzero(np.asarray(rw) < .5)
    return int(f[0]) if len(f) else -1


gaps, changed, missing = [], [], sorted(set(old) - set(new))
for u in uids:
    o = np.asarray(old[u]['rewards'], float)
    if len(o) != len(new[u]): raise SystemExit(f'{u}: step count {len(new[u])} vs {len(o)}')
    gaps.append(float(np.max(np.abs(o - new[u]))))
    if first(o) != first(new[u]): changed.append(u)
f1 = {}
for tag, src in (('0810', {u: old[u]['rewards'] for u in uids}), ('new', new)):
    for s in ('gsm8k', 'math', 'olympiadbench', 'omnimath'):
        us = [u for u in uids if u.startswith(s + '-')]
        e = [u for u in us if old[u]['label'] != -1]; c = [u for u in us if old[u]['label'] == -1]
        ea = 100 * np.mean([first(src[u]) == old[u]['label'] for u in e]); ca = 100 * np.mean([first(src[u]) == -1 for u in c])
        f1.setdefault(s, {})[tag] = round(2 * ea * ca / (ea + ca), 3)
print(json.dumps({'n_answers_compared': len(uids), 'n_unscored_in_new_run': len(missing),
                  'max_abs_reward_gap': max(gaps), 'median_answer_max_gap': float(np.median(gaps)),
                  'n_answers_first_error_prediction_changed': len(changed), 'changed_uids': changed[:20],
                  'official_f1_by_subset': f1}, indent=1))
