"""C4U and TELE16 on every competitor benchmark, in both modes (2026-11-08).

Modes:
  prm = PRM backbone telemetry (the PRM's own hidden states, Instruct vocabulary head) + the PRM's rewards.
  sa  = standalone base model telemetry (Qwen2.5-Math-7B-Instruct / -72B-Instruct / Qwen2.5-Math-7B / Llama-3.1-8B-Instruct),
        with the SAME PRM's rewards grafted onto the same answers.
Arms: TELE16 = stored TELE_N16 (label-free L-SML of the 16 channels, no PRM); C4U = frozen recipe
C4U-parisi-within-jaffe-cross-v1 (c4u_new.c4u, unchanged). In sa mode C4U's inputs are built exactly as build_R1, with the
16 raw channels taken from the standalone bank instead of the PRM backbone bank (PRM columns unchanged). The R1 gate checks
that the raw bank reproduces the stored answer-z channels of the same cell (< 1e-4).

Protocols (functions reused unchanged from telemetry_only_vs_competitors_v1/compute.py and c4u_vs_competitors_v1):
  PRMBench      PRISM T2 / T7 / matched TPR, SLIDER, CROP T6 (prmbench_core).
  ProcessBench  PRISM T6 F1 per subset (own flags), Mind the Gap T3 SLA (first flagged step), CROP T6.
  CRV-Arith, CRV-GSM8K, PRM800K, Math-Shepherd   CROP T6 (alpha .05, 10 seeds, q201 grid; PRM also on fixed101).
  DeltaBench    ARES-style: step macro-F1 at each arm's own label-free flags (PRM: native threshold), within-AUC,
                pooled AUROC, on the common answers of the cells passed (see --deltabench-bench).
PRM-backbone PRMBench/ProcessBench/CROP rows for C4U reproduce c4u_vs_competitors_v1/RESULTS.csv (parity gate).

    COLLECTION_DATA=C:/Users/omris/TAU/hd_collection_data python run_all.py [prm|sa ...]
"""
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'results/c4u_vs_competitors_v1'))
import c4u_competitors as T  # noqa: E402  (frozen recipe + protocol code; patches E.Cell._flags for 'C4U')
N, E, C, A = T.N, T.E, T.C, T.A
DATA = Path(E.DATA)
ALL = DATA / 'c4u_r1_inputs' / 'crop_ares_all'           # every CROP/ARES bank_v3 + scores, copied from AIRCC 2026-11-08
if not N.BUNDLE.exists():                                   # the PC keeps the bundle under c4u_r1_inputs/
    N.BUNDLE = DATA / 'c4u_r1_inputs' / 'auto_raw_plus_z_bundle' / 'inputs'
MODE = 'prm'
CROP_BENCHES = ('crv_arith', 'crv_gsm8k', 'prm800k', 'math_shepherd')
DELTAS = os.environ.get('DELTABENCH_BENCH', 'deltabench,deltabench_ext').split(',')
# deltabench = 447 answers common to every cell with a 4,096-token context; deltabench_ext = all 1,236 answers
# (long answers of Qwen-PRM-7B/72B and PURE scored with dynamic-NTK RoPE x3, see crop_ares_v1/DELTABENCH_EXT_PROVENANCE.md)


def bank_path(prm, bench):
    p = T.PFILE.get(prm, prm)
    if MODE == 'prm':
        if bench == 'processbench':
            return DATA / 'sub5_all_v1' / 'pb_prm' / f'bank_{p}' / 'FEATURES.npz'
        return ALL / f'full_prm_{p}_{bench}' / 'bank_v3' / 'FEATURES.npz'
    if bench == 'prmbench':
        return DATA / 'prm_standalone_banks' / f'bank_{prm}' / 'FEATURES.npz'
    return ALL / f'full_sa_{p}_{bench}' / 'bank_v3' / 'FEATURES.npz'


T.bank_path = bank_path


def raw16(c, prm, bench):
    """c4u_competitors.raw16, except that a multi-cell bank (prm_standalone_banks: 8 ProcessBench grid cells + the
    PRMBench cell) is keyed by the PRMBench cell's row_ids, which are the PRMBench uids of E.load."""
    p = bank_path(prm, bench)
    F = np.load(p); names = list(F['names'].astype(str)); V = F['values']; act = F['digit_active'].astype(bool)
    off = F['offsets'].astype(np.int64)
    if 'cells' in F.files:
        assert bench == 'prmbench'
        sel = np.flatnonzero(F['cells'].astype(str) == 'prmbench_qwen3_8b'); uids = F['row_ids'].astype(str)[sel]
    else:
        sel = np.arange(len(F['uids'])); uids = F['uids'].astype(str)
    pos = {u: j for j, u in zip(sel, uids)}
    assert len(pos) == len(sel)
    rows = np.concatenate([np.arange(off[pos[u]], off[pos[u] + 1]) for u in c.ids])
    assert len(rows) == c.off[-1], (p, len(rows), c.off[-1])
    raw = np.column_stack([V[rows, names.index(n)] for n in T.N16]).astype(float)
    return raw, act[rows]


T.raw16 = raw16


def need(prm, bench):
    p = bank_path(prm, bench)
    if not p.exists():
        raise SystemExit(f'missing raw bank {p}')


def tag(n0, **kw):
    for r in C.ROWS[n0:]:
        r.setdefault('mode', MODE); r.update({k: v for k, v in kw.items() if k not in r})


# ----------------------------------------------------------------------------------------------------- PRMBench
def prmbench(prm):
    n0 = len(C.ROWS)
    if MODE == 'prm':
        T.prmbench(prm)                                  # frozen R1, gated against expected_results.json
        c, *_ = N.load('prmbench', prm, 'R1')
    else:
        need(prm, 'prmbench')
        c = E.load('prmbench', f'{prm}__standalone')
        c.arms['PRMPROB'] = 1. - c.extra['reward']
        c.arms['PRM'] = c.arms[N.K.prm_arm(c)]
        X, ab, rep, g = T.build_X(c, prm, 'prmbench')
        T.FIT[f'sa/prmbench/{prm}/gate'] = g
        T.run_c4u(c, X, ab, f'sa/prmbench/{prm}/{rep}')
        C.prmbench_core(c, prm)
    tag(n0, input='R1')
    del c


# ----------------------------------------------------------------------------------------------------- ProcessBench
def processbench(prm):
    n0 = len(C.ROWS)
    need(prm, 'processbench')
    c = C.cab_cell('processbench', prm)
    X, ab, rep, g = T.build_X(c, prm, 'processbench')
    T.FIT[f'{MODE}/processbench/{prm}/gate'] = g
    T.run_c4u(c, X, ab, f'{MODE}/processbench/{prm}/{rep}')
    preds = [('PRM native', T.pb_predict(c, c.arms['PRM'] > 0), 'native'),
             ('C4U', T.pb_predict(c, c.c4u_flags), 'C4U own count'),
             ('TELE_N16', T.pb_predict(c, c.flags('TELE_N16', 'GAP')), 'GAP')]
    for label, pred, rule in preds:
        m = T.pb_metrics(c, pred)
        for s in T.SUBSETS:
            for k in ('TPR', 'TNR', 'F1'):
                C.add(prm, 'processbench', 'PRISM PB T6', label, f'{s} {k}', m[s][k], rule=rule)
            C.add(prm, 'processbench', 'MtG T3 SLA', label, f'{s} SLA', m[s]['SLA'], rule=rule)
        C.add(prm, 'processbench', 'PRISM PB T6', label, 'average F1', m['average F1'], rule=rule)
        for k in ('macro SLA', 'overall SLA', 'macro SLA lo', 'macro SLA hi'):
            C.add(prm, 'processbench', 'MtG T3 SLA', label, k, m[k], rule=rule)
    C.crop_table6(c, 'processbench', prm)
    tag(n0, input=rep)
    del c


# ----------------------------------------------------------------------------------------------------- CROP sets
def hook(c, bench, prm):
    need(prm, bench)
    X, ab, rep, g = T.build_X(c, prm, bench)
    T.FIT[f'{MODE}/{bench}/{prm}/gate'] = g; c.rep = rep
    T.run_c4u(c, X, ab, f'{MODE}/{bench}/{prm}/{rep}')


def crop(prm):
    for bench in CROP_BENCHES:
        n0 = len(C.ROWS)
        C.CELL_HOOK = hook
        c = C.cab_cell(bench, prm)
        C.crop_table6(c, bench, prm)
        tag(n0, input=c.rep)
        del c
    C.CELL_HOOK = None


# ----------------------------------------------------------------------------------------------------- DeltaBench
def macro_f1(err, flag):
    def f1(y, p):
        tp = (y & p).sum(); return 0. if tp == 0 else 2 * tp / (y.sum() + p.sum())
    return 100 * .5 * (f1(err, flag) + f1(~err, ~flag))


def deltabench(prm):
    for d in DELTAS:
        deltabench_one(prm, d)


def deltabench_one(prm, DELTA):
    n0 = len(C.ROWS)
    C.CELL_HOOK = hook
    c = C.cab_cell(DELTA, prm)
    C.CELL_HOOK = None
    err = c.err.astype(bool)
    for label, score, flag in (('PRM native', c.arms['PRM'], c.arms['PRM'] > 0),
                               ('TELE_N16', c.arms['TELE_N16'], c.flags('TELE_N16', 'GAP')),
                               ('C4U', c.arms['C4U'], c.c4u_flags)):
        wa = E.per_answer_auc(err, score, c.off, None)
        C.add(prm, DELTA, 'ARES-style', label, 'macro-F1 (own flags)', macro_f1(err, flag), rule='label-free' if label != 'PRM native' else 'native')
        C.add(prm, DELTA, 'ARES-style', label, 'within-AUC', float(np.nanmean(wa)))
        C.add(prm, DELTA, 'ARES-style', label, 'pooled AUROC', float(roc_auc_score(err, score)))
        C.add(prm, DELTA, 'ARES-style', label, 'flag rate', 100 * float(np.mean(flag)))
    C.add(prm, DELTA, 'ARES-style', 'population', 'answers', len(c.ids)); C.add(prm, DELTA, 'ARES-style', 'population', 'steps', len(err))
    tag(n0, input=c.rep)
    del c


def parity_backbone():
    """C4U prm-mode rows vs c4u_vs_competitors_v1/RESULTS.csv (same code, same cells)."""
    ref = {}
    for r in E.read_csv(ROOT / 'results/c4u_vs_competitors_v1/RESULTS.csv'):
        if r['arm'] == 'C4U':
            ref[(r['prm'], r['benchmark'], r['protocol'], r['metric'], r.get('pool', ''), r.get('rule', ''), r.get('scope', ''), r.get('grid', ''))] = float(r['value'])
    n = 0
    for r in C.ROWS:
        if r.get('mode') != 'prm' or r['arm'] != 'C4U': continue
        k = (r['prm'], r['benchmark'], r['protocol'], r['metric'], r.get('pool', ''), r.get('rule', ''), r.get('scope', ''), r.get('grid', ''))
        if k in ref:
            C.gate(f"backbone parity {' '.join(str(x) for x in k if x)}", r['value'], ref[k], 1e-6); n += 1
    print('backbone parity rows', n)
    return n


def main():
    global MODE
    modes = [m for m in sys.argv[1:] if m in ('prm', 'sa')] or ['prm', 'sa']
    C.TELE = ['TELE_N16', 'C4U']
    for MODE in modes:
        C.MODE = MODE
        for prm in N.PRMS:
            prmbench(prm); processbench(prm); crop(prm); deltabench(prm)
            print(MODE, prm, 'done', flush=True)
    for r in C.ROWS:
        r.pop('standalone_model', None)
    if 'prm' in modes and parity_backbone() == 0:
        raise SystemExit('no backbone parity rows matched')
    sfx = '' if modes == ['prm', 'sa'] else '_' + '_'.join(modes)
    E.write_csv(HERE / f'RESULTS{sfx}.csv', C.ROWS)
    fails = [p['check'] for p in C.PAR if not p['ok']]
    json.dump(dict(n_checks=len(C.PAR), failed=fails, checks=C.PAR), open(HERE / f'PARITY{sfx}.json', 'w'), indent=1, default=float)
    json.dump(T.FIT, open(HERE / f'FIT{sfx}.json', 'w'), indent=1, default=float)
    print('rows', len(C.ROWS), 'parity', len(C.PAR), 'failed', fails)
    if fails: raise SystemExit(3)


if __name__ == '__main__':
    main()
