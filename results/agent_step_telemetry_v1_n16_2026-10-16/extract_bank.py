"""Build the 56-channel external_banks_v4 step bank (N16 is a 16-column subset) from the
agent_step_telemetry_v1 Qwen3-8B teacher-forced records. LABEL-BLIND: never reads evaluator files.

Mirrors scripts/rebuild_source_bank_v4.py:one() at commit 080877eb (the code_thinkfix_bank_553de79
feature code; the six feature modules are taken from results/monday_fusion_analysis_v1/code, whose
SHA256 equal the code_sha256 recorded in thinkfix_rescore_v1/source_bank_thinkfix_mask1/MANIFEST.json):
  step_features (48) + external_banks_v4.new_step_features (5, fill_missing per answer) +
  external_banks_v4.digit_step_features (3 + active mask).

Adaptations required by this telemetry (documented in README):
  1. "Answer trace" = concatenation of the item's NONEMPTY step-token spans (the scored assistant
     tokens). top_k_logprobs rows exist only for those tokens (top_k_token_indices); the frozen code
     requires a [T, 50] top-k matrix aligned with every trace token. Non-step tokens (user turns, tool
     outputs, chat-template tokens) are dropped; spans are remapped to the concatenated coordinates.
  2. mask1 rule (spectral_utils/first_token_mask.RULE) applied to that trace: drop token 0, spans
     [lo,hi) -> [max(lo-1,0), hi-1). A step that becomes empty is recorded as a mask failure and the
     item keeps its remaining steps only if step 0 was not its only step (see MASK_FAILURES).

    python -I extract_bank.py <records_root> <out_dir> <bench> [workers]
"""
import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import glob, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

W = Path('/Users/osegev/.codex/worktrees/agent-step-telemetry-v1/hallucination_detection')
CODE = W / 'results/monday_fusion_analysis_v1/code'
sys.path.insert(0, str(CODE))
import spectral_utils.family_tail_transfer as FTT  # noqa: E402
FTT.LOCK_PATH = W / 'results/family_tail_transfer_v1/TRANSFER_LOCK_V1.json'   # hash-checked by load_lock()
from spectral_utils import external_banks_v4 as E  # noqa: E402
from spectral_utils.family_external_features import step_features  # noqa: E402

NAMES = list(FTT.load_lock()['recipe']['channels_48']) + list(E.NEW_NAMES) + list(E.DIGIT_NAMES)


def fill_missing(matrix):   # verbatim scripts/rebuild_source_bank_v4.py
    x = np.array(matrix, float); filled = (~np.isfinite(x)).sum(axis=0)
    if np.isinf(x).any():
        raise ValueError('infinite feature values')
    for j in range(x.shape[1]):
        good = np.isfinite(x[:, j])
        x[~good, j] = x[good, j].mean() if good.any() else 0.
    return x, filled


def step_keys(t, n):
    if t['benchmark'] == 'apb':
        return list(t['step_message_indices'])
    if t['benchmark'] == 'aft':
        return [t['entry_to_history'][j] for j in t['step_entry_indices']]
    return list(range(n))


def one(path):
    r = json.load(open(path)); t = r['payload']['telemetry']
    spans = [tuple(map(int, s)) for s in t['step_token_spans']]
    keys = step_keys(t, len(spans))
    keep = [k for k, (s, e) in enumerate(spans) if e > s]
    out = {'uid': r['uid'], 'subset': t['subset'], 'group': t['group'], 'n_steps_raw': len(spans)}
    if not keep:
        return {**out, 'status': 'no_nonempty_steps'}
    idx = np.concatenate([np.arange(spans[k][0], spans[k][1]) for k in keep])
    tki = np.asarray(t['top_k_token_indices'], int)
    if not np.array_equal(idx, tki):
        return {**out, 'status': 'topk_index_mismatch'}
    lens = np.array([spans[k][1] - spans[k][0] for k in keep])
    # mask1: drop trace token 0 -> step 0 loses one token
    lens[0] -= 1; sel = idx[1:]; topk_rows = slice(1, None)
    mask_fail = []
    if lens[0] == 0:
        mask_fail.append(int(keys[keep[0]]) if keys else 0)
        keep = keep[1:]; lens = lens[1:]
        if not keep:
            return {**out, 'status': 'mask1_emptied_only_step'}
    ends = np.cumsum(lens); new_spans = np.column_stack([ends - lens, ends]).astype(int)
    ids_all = np.asarray(t['gen_token_ids'])
    row = {'gen_token_ids': ids_all[sel],
           'token_entropies': np.asarray(t['token_entropies'], float)[sel],
           'token_spilled_energies': np.asarray(t['token_spilled_energies'], float)[sel],
           'token_logsumexp': np.asarray(t['token_logsumexp'], float)[sel],
           'top_k_logprobs': {'ids': np.asarray(t['top_k_logprobs']['ids'])[topk_rows],
                              'logprobs': np.asarray(t['top_k_logprobs']['logprobs'], float)[topk_rows]},
           'step_token_spans': new_spans}
    try:
        b48, _ = step_features(row)
        b5, _ = E.new_step_features(row)
        d3, act = E.digit_step_features(row)
    except Exception as exc:
        return {**out, 'status': f'feature_error: {type(exc).__name__}: {exc}'[:300]}
    b5, filled = fill_missing(b5)
    v = np.column_stack((b48, b5, d3))
    if not np.isfinite(v).all():
        return {**out, 'status': 'nonfinite'}
    return {**out, 'status': 'ok', 'values': v, 'active': np.asarray(act, bool),
            'keys': [keys[k] for k in keep], 'lens': lens.tolist(), 'mask1_emptied_steps': mask_fail,
            'filled': filled.tolist()}


def main():
    root, outd, bench = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    files = sorted(glob.glob(str(root / bench / '*.record.json')))
    t0 = time.time(); res = []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap(one, files, chunksize=4)):
            res.append(r)
            if (i + 1) % 200 == 0: print(bench, i + 1, len(files), round(time.time() - t0), flush=True)
    ok = [r for r in res if r['status'] == 'ok']
    off = np.r_[0, np.cumsum([len(r['keys']) for r in ok])]
    np.savez_compressed(outd / f'FEATURES_{bench}.npz', values=np.vstack([r['values'] for r in ok]),
                        digit_active=np.vstack([r['active'] for r in ok]), names=np.asarray(NAMES), offsets=off,
                        uids=np.asarray([r['uid'] for r in ok]), subsets=np.asarray([r['subset'] for r in ok]),
                        groups=np.asarray([r['group'] for r in ok]),
                        step_keys=np.concatenate([np.asarray(r['keys'], int) for r in ok]),
                        step_lens=np.concatenate([np.asarray(r['lens'], int) for r in ok]))
    status = {}
    for r in res: status[r['status']] = status.get(r['status'], 0) + 1
    man = {'bench': bench, 'n_files': len(files), 'n_ok': len(ok), 'n_steps': int(off[-1]), 'status_counts': status,
           'not_ok': [{k: r[k] for k in ('uid', 'subset', 'status')} for r in res if r['status'] != 'ok'],
           'mask1_emptied_steps': [{'uid': r['uid'], 'keys': r['mask1_emptied_steps']} for r in ok if r['mask1_emptied_steps']],
           'new_channel_filled_values': dict(zip(E.NEW_NAMES, np.sum([r['filled'] for r in ok], 0).tolist())),
           'labels_read': False, 'seconds': round(time.time() - t0, 1)}
    (outd / f'MANIFEST_{bench}.json').write_text(json.dumps(man, indent=1))
    print('DONE', bench, status, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
