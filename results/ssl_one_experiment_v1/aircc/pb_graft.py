"""ProcessBench bank input for the one SSL experiment (RC_HANDOFF D.3). No labels are read.

The external builder (prm_boost_build_bank_external_v2.py) needs `rewards` and `native_labels` per row, but uses them
only for REWARDS.npz. A standalone run has neither, and the V1/V2 answers have no PRM run. So this writes DUMMY zero
rewards / zero native labels (one per step). With --rewards-from PRM.pkl it instead copies the PRM's values for the same
uids (rows must match on question and steps, as in graft_rewards.py); used only to check that FEATURES.npz does not
depend on the rewards (zero-reward bank == PRM-reward bank on the C preflight rows).
    python pb_graft.py RAW.pkl OUT.pkl [--rewards-from PRM.pkl]
"""
import argparse, pickle


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('raw'); ap.add_argument('out'); ap.add_argument('--rewards-from')
    a = ap.parse_args()
    raw = pickle.load(open(a.raw, 'rb'))
    prm = None
    if a.rewards_from:
        prm = {str(e['idx']): e for e in pickle.load(open(a.rewards_from, 'rb')).values()}
    out, n = {}, 0
    for k, e in raw.items():
        r = dict(e)
        if 'failure' not in r:
            ns = len(r['step_token_spans'])
            if prm is None:
                r['rewards'], r['native_labels'] = [0.0] * ns, [0] * ns
            else:
                p = prm[str(e['idx'])]
                if p['question'] != e['question'] or list(p['steps']) != list(e['steps']):
                    raise SystemExit(f'{e["idx"]}: question/steps differ between the runs')
                r['rewards'], r['native_labels'] = p['rewards'], p['native_labels']
            n += 1
        out[k] = r
    with open(a.out + '.tmp', 'wb') as f:
        pickle.dump(out, f)
    import os; os.replace(a.out + '.tmp', a.out)
    print('GRAFT', 'prm' if prm else 'zero', n, 'rows of', len(out), flush=True)


if __name__ == '__main__':
    main()
