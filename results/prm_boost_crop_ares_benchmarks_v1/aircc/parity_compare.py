"""Parity of a new telemetry pkl against a frozen one on their first N shared rows (AIRCC CPU).

Same tests as cluster/run_prmbench_prm_telemetry.py:compare_caches (T1), but `rewards` is compared only when both
pkls carry it, so it also covers --mode instruct (standalone) runs. Exit 0 = PASS, 1 = FAIL. Writes a JSON report.

    python parity_compare.py NEW.pkl FROZEN.pkl --n 40 --out PARITY.json
"""
import argparse, json, pickle, sys
import numpy as np


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('new'); ap.add_argument('frozen')
    ap.add_argument('--n', type=int, default=40); ap.add_argument('--atol', type=float, default=1e-5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    new, ref = load(a.new), load(a.frozen)
    ref_by = {str(e['idx']): e for e in ref.values()}
    rows = [e for _, e in sorted(new.items())][:a.n]
    order_new = [str(e['idx']) for e in rows]
    order_ref = [str(e['idx']) for _, e in sorted(ref.items())][:a.n]
    rep = {'new': a.new, 'frozen': a.frozen, 'n_requested': a.n, 'order_match': order_new == order_ref,
           'exact_mismatch': [], 'max_abs_gaps': {}, 'n_checked': 0}
    for e in rows:
        if 'failure' in e:
            rep['exact_mismatch'].append(f'{e["idx"]}: failure {e["failure"]}'); continue
        o = ref_by.get(str(e['idx']))
        if o is None:
            rep['exact_mismatch'].append(f'{e["idx"]}: not in frozen'); continue
        keys = ['gen_token_ids', 'step_token_spans'] + (['rewards'] if 'rewards' in e and 'rewards' in o else [])
        for k in keys:
            if not np.array_equal(np.asarray(e[k]), np.asarray(o[k])):
                rep['exact_mismatch'].append(f'{e["idx"]}: {k}')
        if not np.array_equal(e['top_k_logprobs']['ids'], o['top_k_logprobs']['ids']):
            rep['exact_mismatch'].append(f'{e["idx"]}: top-k ids')
        pairs = [(k, e[k], o[k]) for k in ('token_entropies', 'token_spilled_energies', 'token_logsumexp') if k in e and k in o]
        pairs.append(('top_k_logprobs', e['top_k_logprobs']['logprobs'], o['top_k_logprobs']['logprobs']))
        for k, x, y in pairs:
            x, y = np.asarray(x, float), np.asarray(y, float)
            if x.shape != y.shape:
                rep['exact_mismatch'].append(f'{e["idx"]}: {k} shape'); continue
            rep['max_abs_gaps'][k] = max(rep['max_abs_gaps'].get(k, 0.0), float(np.max(np.abs(x - y), initial=0)))
        rep['n_checked'] += 1
    ok = (rep['order_match'] and not rep['exact_mismatch'] and rep['n_checked'] == a.n
          and all(v <= a.atol for v in rep['max_abs_gaps'].values()))
    rep['status'] = 'PASS' if ok else 'FAIL'
    with open(a.out, 'w') as f:
        json.dump(rep, f, indent=1)
    print(rep['status'], {k: rep[k] for k in ('n_checked', 'order_match', 'max_abs_gaps')}, rep['exact_mismatch'][:5])
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
