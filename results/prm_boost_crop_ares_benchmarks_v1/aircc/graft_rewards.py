"""Attach a PRM run's step rewards to a standalone (--mode instruct) telemetry pkl of the SAME answers (AIRCC CPU).

The external bank builder (prm_boost_build_bank_external_v2.py) and ablation5_socratic.py need `rewards` and
`native_labels` per row. A standalone run has neither. This copies them from the PRM-mode pkl of the same benchmark
and adapter, row by row, after checking that uid, question and the step list are identical. Telemetry keys of the
standalone row are never touched. Rows that failed in either run (e.g. context_overflow), or that the PRM run excluded, are counted failures. No labels.

    python graft_rewards.py STANDALONE.pkl PRM.pkl OUT.pkl
"""
import pickle, sys


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def main():
    sa_path, prm_path, out_path = sys.argv[1:4]
    sa, prm = load(sa_path), load(prm_path)
    by = {str(e['idx']): e for e in prm.values()}
    out, n_graft, n_fail = {}, 0, 0
    for k, e in sa.items():
        p = by.get(str(e['idx']))
        r = dict(e)
        if p is None:   # the PRM could not read this answer (e.g. its step tag inside the text): counted failure
            r.setdefault('failure', 'prm_unscored'); out[k] = r; n_fail += 1; continue
        if e['question'] != p['question'] or list(e['steps']) != list(p['steps']):
            raise SystemExit(f'{e["idx"]}: question/steps differ between the runs')
        if 'failure' in p and 'failure' not in r:
            r['failure'] = 'prm_' + str(p['failure'])
        if 'failure' in r:
            n_fail += 1
        else:
            r['rewards'], r['native_labels'] = p['rewards'], p['native_labels']; n_graft += 1
        out[k] = r
    extra = set(by) - {str(e['idx']) for e in sa.values()}
    if extra:
        raise SystemExit(f'{len(extra)} PRM rows are not in the standalone pkl')
    with open(out_path + '.tmp', 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    import os
    os.replace(out_path + '.tmp', out_path)
    print(f'GRAFT rows={len(out)} grafted={n_graft} failed={n_fail} -> {out_path}')


if __name__ == '__main__':
    main()
