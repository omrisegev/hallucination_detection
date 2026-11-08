"""Merge a DeltaBench long-context run into the original cell (AIRCC CPU, no labels).

    python merge_longctx.py OLD_DIR NEW_DIR OUT_DIR MODE

OLD_DIR = full_<mode>_<a>_deltabench (native 4,096 context), NEW_DIR = full_<mode>_<a>_deltabench_ropedyn3 (same answers,
dynamic NTK x3). Rows the original run scored are kept from OLD_DIR unchanged; rows it counted as context_overflow
are taken from NEW_DIR and tagged `rope_extension`. Every row the original scored is compared with its new twin:
prm mode must match exactly (same code path, unbatched); standalone mode is reported (the original run was batched,
the new one is not, and bf16 batch composition moves ~2% of tokens, see the README). Writes OUT_DIR/external_telemetry.pkl,
OUT_DIR/manifest.json (same status fields downstream.sbatch reads) and OUT_DIR/PARITY_longctx.json. Exit 3 on a prm-mode
parity failure; nothing is written then.
"""
import json
import os
import pickle
import sys

import numpy as np

KEYS = ('rewards', 'token_entropies', 'token_spilled_energies', 'token_logsumexp', 'gen_token_ids', 'step_token_spans')


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def gap(a, b):
    try:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        return 0.0 if a == b else float('inf')
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a - b), initial=0.0))


def main():
    old_dir, new_dir, out_dir, mode = sys.argv[1:5]
    old, new = load(f'{old_dir}/external_telemetry.pkl'), load(f'{new_dir}/external_telemetry.pkl')
    new_manifest = json.load(open(f'{new_dir}/manifest.json'))
    if new_manifest['status'] not in ('COMPLETE', 'COMPLETE_WITH_FAILURES'):
        raise SystemExit(f'new run not complete: {new_manifest["status"]}')
    if sorted(old) != sorted(new):
        raise SystemExit(f'row keys differ: {len(old)} vs {len(new)}')
    merged, gaps, n_old, n_new, failed = {}, {k: 0.0 for k in KEYS + ('top_k_ids', 'top_k_logprobs')}, 0, 0, {}
    n_rows_diff = 0
    for i in sorted(old):
        o, n = old[i], new[i]
        if o['idx'] != n['idx'] or o['question'] != n['question'] or list(o['steps']) != list(n['steps']):
            raise SystemExit(f'row {i}: idx/question/steps differ')
        if 'failure' not in o:
            if 'failure' in n:
                raise SystemExit(f'row {i}: scored natively but failed in the extended run: {n["failure"]}')
            row_gap = 0.0
            for k in KEYS:
                if k in o or k in n:
                    g = gap(o.get(k), n.get(k)); gaps[k] = max(gaps[k], g); row_gap = max(row_gap, g)
            for k, kk in (('ids', 'top_k_ids'), ('logprobs', 'top_k_logprobs')):
                g = gap(o['top_k_logprobs'][k], n['top_k_logprobs'][k]); gaps[kk] = max(gaps[kk], g); row_gap = max(row_gap, g)
            n_rows_diff += row_gap > 0
            merged[i] = o; n_old += 1
        elif o['failure'] == 'context_overflow' and 'failure' not in n:
            r = dict(n); r['rope_extension'] = new_manifest.get('rope_extension'); merged[i] = r; n_new += 1
        else:
            r = n if 'failure' in n else o
            merged[i] = r; failed[str(i)] = {'idx': r['idx'], 'failure': r['failure'], **{k: r[k] for k in
                                             ('n_input_tokens', 'context_limit') if k in r}}
    exact = all(v == 0.0 for v in gaps.values())
    parity = {'mode': mode, 'old_dir': old_dir, 'new_dir': new_dir, 'n_rows': len(merged), 'n_native_rows_kept': n_old,
              'n_rows_from_extension': n_new, 'n_failed': len(failed), 'max_abs_gap_native_rows': gaps,
              'n_native_rows_with_any_gap': int(n_rows_diff), 'exact': exact,
              'status': 'PASS' if exact else ('FAIL' if mode == 'prm' else 'REPORTED_BATCHING_DRIFT')}
    print(json.dumps(parity, indent=1))
    if mode == 'prm' and not exact:
        sys.exit(3)
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/external_telemetry.pkl.tmp', 'wb') as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(f'{out_dir}/external_telemetry.pkl.tmp', f'{out_dir}/external_telemetry.pkl')
    manifest = {'driver': 'merge_longctx.py', 'mode': mode, 'model': new_manifest.get('model'),
                'adaptation': 'rows over the native 4,096-token context scored with dynamic NTK RoPE (rope_extension)',
                'rope_extension': new_manifest.get('rope_extension'), 'sources': [old_dir, new_dir],
                'status': 'COMPLETE_WITH_FAILURES' if failed else 'COMPLETE', 'n_rows': len(merged),
                'n_failed_rows': len(failed), 'failed_rows': failed,
                'n_rows_context_overflow': sum(v['failure'] == 'context_overflow' for v in failed.values()),
                'n_rows_from_extension': n_new, 'parity': parity}
    json.dump(manifest, open(f'{out_dir}/manifest.json', 'w'), indent=1)
    json.dump(parity, open(f'{out_dir}/PARITY_longctx.json', 'w'), indent=1)
    print(f'MERGED rows={len(merged)} native={n_old} extension={n_new} failed={len(failed)} -> {out_dir}')


if __name__ == '__main__':
    main()
