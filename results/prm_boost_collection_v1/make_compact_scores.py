"""Compact, label-free copy of the stage-1 scores for git (plan 1.6): float32 key arms + ids/offsets/covariates.

    python make_compact_scores.py   ->  scores/<bench>_<prm>.npz  + scores/MANIFEST.json (sha256 of every source file)

Key arms: PRM logit, PRM prob, TELE_N16, MEAN_N16, EQ_N16, HALF_N16, PRMG_N16, CHAIN_N16, CHAIN_N13, LSML_CONT,
CHAIN_N16 sign-abs, POS_ONLY, LEN_ONLY (where the benchmark has them). No labels are written.
External files keep the original layout (uids, nonempty, nonempty_lengths, scores on non-empty steps).
"""
import hashlib, json
import numpy as np
import evaluate_collection as E

OUT = E.HERE / 'scores'
PB_KEYS = {'PRM_logit': 'PRM_Q80', 'PRM_prob': 'PRM_Q80_PROB', 'TELE_N16': 'TELE_N16', 'MEAN_N16': 'MEAN_N16', 'EQ_N16': 'EQ_N16',
           'HALF_N16': 'HALF_N16', 'PRMG_N16': 'PRMG_N16', 'CHAIN_N16': 'CHAIN_N16', 'CHAIN_N13': 'CHAIN_N13',
           'LSML_CONT': 'LSML_CONT', 'POS_ONLY': 'POS_ONLY', 'LEN_ONLY': 'LEN_ONLY'}
EXT_KEYS = {'PRM_logit': 'PRM', 'PRM_prob': 'PRMPROB', 'TELE_N16': 'TELE_N16', 'MEAN_N16': 'MEAN_N16', 'EQ_N16': 'EQ_N16',
            'HALF_N16': 'HALF_N16', 'PRMG_N16': 'PRMG_N16', 'CHAIN_N16': 'CHAIN_N16', 'CHAIN_N13': 'CHAIN_N13', 'POS_ONLY': 'POS_ONLY'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True); man = {'labels_written': False, 'files': {}}
    for prm in E.PRMS:
        src = E.DATA / 'prmbench' / f'ABL5_SCORES_{prm}.npz'; sab = E.DATA / 'prmbench_signabs' / f'ABL5_SCORES_{prm}.npz'
        S = np.load(src); A = np.load(sab)
        d = dict(row_ids=S['row_ids'].astype(str), offsets=S['offsets'].astype(np.int64), source_group=S['source_group'].astype(str),
                 position=S['position'].astype(np.float32), log_step_tokens=S['log_step_tokens'].astype(np.float32),
                 reward=S['reward'].astype(np.float32), native_error_flag=S['native_error_flag'].astype(bool),
                 **{f'score__{k}': S[f'score__{v}'].astype(np.float32) for k, v in PB_KEYS.items()},
                 score__CHAIN_N16_signabs=A['score__CHAIN_N16'].astype(np.float32))
        assert np.array_equal(A['offsets'], S['offsets'])
        p = OUT / f'prmbench_{prm}.npz'; np.savez_compressed(p, **d)
        man['files'][p.name] = dict(benchmark='prmbench', prm=prm, sources={str(src.name): sha(src), 'signabs/' + sab.name: sha(sab)},
                                    aircc=f'$R/prm_boost_gen_v1/ablation5_export_v1/{src.name} (+ ablation5_signabs_v1)')
        for bench, d1, d2 in (('socratic', 'socratic', 'socratic_signabs'), ('hard2verify', 'h2v', 'h2v_signabs')):
            n = E.EXT_NAME.get(prm, prm); src = E.DATA / d1 / f'SCORES_{n}.npz'; sab = E.DATA / d2 / f'SCORES_{n}.npz'
            S = np.load(src); A = np.load(sab)
            d = dict(uids=S['uids'].astype(str), nonempty=S['nonempty'].astype(bool), nonempty_lengths=S['nonempty_lengths'].astype(np.int64),
                     native_valid=S['native_valid'].astype(np.int8),
                     **{f'score__{k}': S[f'score__{v}'].astype(np.float32) for k, v in EXT_KEYS.items() if f'score__{v}' in S},
                     score__CHAIN_N16_signabs=A['score__CHAIN_N16'].astype(np.float32))
            fz = E.DATA / {'socratic': 'frozen_socratic', 'hard2verify': 'frozen_h2v'}[bench] / f'SCORES_{n}.npz'
            if fz.exists():
                F = np.load(fz)
                if 'score__LSML_CONT' in F and np.array_equal(F['uids'].astype(str), S['uids'].astype(str)):
                    d['score__LSML_CONT'] = F['score__LSML_CONT'].astype(np.float32)
            p = OUT / f'{bench}_{prm}.npz'; np.savez_compressed(p, **d)
            man['files'][p.name] = dict(benchmark=bench, prm=prm, sources={src.name: sha(src), 'signabs/' + sab.name: sha(sab),
                                                                         **({'frozen/' + fz.name: sha(fz)} if fz.exists() else {})},
                                        aircc=f"$X/{'collect_socratic_v1' if bench == 'socratic' else 'collect_h2v_v1'}/{src.name}")
    tot = sum((OUT / k).stat().st_size for k in man['files'])
    man['total_bytes'] = tot
    (OUT / 'MANIFEST.json').write_text(json.dumps(man, indent=1))
    print('total MB', tot / 1e6)


if __name__ == '__main__':
    main()
