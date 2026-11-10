"""C4U with telemetry only: no PRM score and no PRM anchor (2026-11-11).

Omri asked for a version of the frozen C4U (C4U-parisi-within-jaffe-cross-v1) that uses telemetry only, in two modes:
  prm = telemetry from the PRM network (its backbone hidden states, Instruct vocabulary head), reward output NOT used;
  sa  = telemetry from the standalone base model (Qwen2.5-Math-7B-Instruct / -72B-Instruct / Qwen2.5-Math-7B /
        Llama-3.1-8B-Instruct for Qwen-PRM-7B / Qwen-PRM-72B / PURE / RLHFlow).
Benchmarks: PRMBench and ProcessBench, the 4 PRMs. CPU only, from the stored raw banks. Development data.

C4U-T (this adaptation) = c4u_new.c4u with the three places where the PRM enters replaced by telemetry:
  1. inputs   X = the 16 R1 telemetry columns only (answer-z of the global rank->normal of the 16 raw channels, exactly
                the columns 2..17 of build_R1 / c4u_competitors.build_X). Columns 0 (answer-z PRM risk) and 1 (raw PRM
                probability) are removed, and so is the fixed PRM vote (raw p > .5).
  2. L-SML anchor (sign of the fused score s): the plain mean of the 16 columns, the convention S8.chain already uses
                for its telemetry stage. Every column is then oriented by corr with s and gets the GAP vote (in C4U only
                the PRM column is exempt from both).
  3. anchor group (sign of the binary spectral fit): the group whose fused column correlates most with s. In C4U it
                is the group holding the PRM column.
Everything else (L-SML groups, Parisi rank-one within-group weights, group votes, Jaffe spectral psi/eta/prevalence,
posterior count per answer, rank = XG @ log-odds weights, top-k flags) is the frozen code, unchanged.
The adaptation was fixed before any quality number was computed; it has no tuned parameter.

Reference arms in the same cells and through the same metric code: PRM native (its own threshold) and full C4U
(frozen, PRM + the same mode's telemetry). The full-C4U and PRM rows are gated against
results/c4u_tele16_all_benchmarks_v1/RESULTS_{prm,sa}.csv. TELE16 rows are copied from there (no new run).

Metrics: PRMBench PRMScore (own flags, PRISM T2), PRISM T7 categories, within-answer AUC; ProcessBench PRISM T6 F1,
Mind the Gap T3 SLA. No within-AUC on ProcessBench.

    COLLECTION_DATA=C:/Users/omris/TAU/hd_collection_data python run.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'results/c4u_tele16_all_benchmarks_v1'))
import run_all as RA  # noqa: E402  (bank paths per mode, multi-cell standalone bank reader)
T, N, E, C = RA.T, RA.N, RA.E, RA.C
S8, K = N.S8, N.K
from spectral_utils.binary_view_estimation import rank_one_offdiag, spectral_binary  # noqa: E402


class RecipeFailure(RuntimeError):
    pass


def c4u_t(c, X):
    """Telemetry-only C4U. X: n_steps x 16 telemetry columns. Same body as c4u_new.c4u except the 3 marked lines."""
    off = c.off
    s, g, _ = N.lsml(X, X.mean(1))                                                     # (2) anchor = channel mean
    sign = np.array([1. if np.corrcoef(X[:, j], s)[0, 1] >= 0 else -1. for j in range(X.shape[1])])   # (2) all columns
    V = np.column_stack([np.where(S8.gap_flags(c, X[:, j] * sign[j]), 1., -1.) for j in range(X.shape[1])])
    Xo = X * sign
    Xo = (Xo - Xo.mean(0)) / Xo.std(0)
    VG, XG, ws, gid = [], [], [], []
    for k in np.unique(g[g >= 0]):
        idx = np.flatnonzero(g == k); wt = np.ones(len(idx))
        if len(idx) >= 3:
            wt = np.abs(rank_one_offdiag(np.cov(Xo[:, idx].T)))
        wt = wt / wt.sum()
        VG.append(np.where(V[:, idx] @ wt > 0, 1., -1.))
        xs = Xo[:, idx] @ wt
        XG.append((xs - xs.mean()) / xs.std())
        ws.append(wt); gid.append(idx)
    if len(VG) < 3:
        raise RecipeFailure(f'{len(VG)} groups')
    VG, XG = np.column_stack(VG), np.column_stack(XG)
    ag = int(np.argmax([np.corrcoef(XG[:, k], s)[0, 1] for k in range(XG.shape[1])]))  # (3) anchor group
    sp = spectral_binary(VG, anchor=ag)
    P = N.posterior(VG, sp['psi'], sp['eta'], float(np.clip(sp['prevalence'], .01, .99)))
    rank = XG @ N.lw(sp['psi'], sp['eta'])
    k = np.rint(np.add.reduceat(P, off[:-1])).astype(int)
    F = E.topk_flags(rank, off, k)
    return dict(rank=rank, flags=F, k=k, prevalence=float(sp['prevalence']), groups=gid, weights=ws, anchor_group=ag,
                unassigned=int((g < 0).sum()), psi=sp['psi'], eta=sp['eta'], s=s)


ROWS, FIT, PAR, SAVE = [], {}, [], {}


def add(mode, prm, bench, arm, metric, value, **kw):
    ROWS.append(dict(mode=mode, prm=prm, benchmark=bench, arm=arm, metric=metric, value=float(value), **kw))


def fitinfo(r):
    return dict(K=len(r['groups']), group_sizes=[len(g) for g in r['groups']],
                groups=[[T.N16[j] for j in g] for g in r['groups']], prevalence=r['prevalence'],
                flag_rate=float(r['flags'].mean()), answers_zero_flags=int((r['k'] == 0).sum()),
                unassigned=r['unassigned'], anchor_group=r['anchor_group'],
                psi=[float(x) for x in r['psi']], eta=[float(x) for x in r['eta']])


def prmbench(mode, prm):
    RA.MODE = mode
    if mode == 'prm':
        c, ans_eval, X, ab, _ = N.load('prmbench', prm, 'R1')
        Xt = X[:, 2:]
    else:
        RA.need(prm, 'prmbench')
        c = E.load('prmbench', f'{prm}__standalone')
        c.arms['PRMPROB'] = 1. - c.extra['reward']; c.arms['PRM'] = c.arms[K.prm_arm(c)]
        X, ab, rep, g = T.build_X(c, prm, 'prmbench'); assert rep == 'R1'
        _, ans_eval = K.eval_rows(c, 'prmbench')
        Xt = X[:, 2:]
    full = N.c4u(c, X, ab)
    tel = c4u_t(c, Xt)
    W, inv, _ = E.boot_weights(c); SAVE[f'{mode}|prmbench|{prm}|groups'] = inv
    FIT[f'{mode}/prmbench/{prm}'] = fitinfo(tel)
    native = c.flags('PRM_native', 'native')
    for arm, rank, flags, is_native in (('PRM native', c.arms['PRM_native'], native, True),
                                        ('C4U (PRM + telemetry)', full['rank'], full['flags'], False),
                                        ('C4U-T (telemetry only)', tel['rank'], tel['flags'], False)):
        ps, _ = E.official(c, flags, is_native)
        add(mode, prm, 'prmbench', arm, 'PRMScore', ps)
        SAVE[f'{mode}|prmbench|{prm}|{arm}|counts'] = E.counts_by_answer(c, flags)
        for kk, v in E.category_scores(c, flags).items():
            add(mode, prm, 'prmbench', arm, f'cat {kk}', v)
        wa = E.per_answer_auc(c.err, rank, c.off, None); wa[~ans_eval] = np.nan
        add(mode, prm, 'prmbench', arm, 'within-AUC', float(np.nanmean(wa)))
        add(mode, prm, 'prmbench', arm, 'flag rate', 100 * float(flags.mean()))
    del c


def processbench(mode, prm):
    RA.MODE = mode
    RA.need(prm, 'processbench')
    c = C.cab_cell('processbench', prm)
    X, ab, rep, g = T.build_X(c, prm, 'processbench'); assert rep == 'R1'
    full = N.c4u(c, X, ab)
    tel = c4u_t(c, X[:, 2:])
    SAVE[f'{mode}|processbench|{prm}|label'] = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids])
    SAVE[f'{mode}|processbench|{prm}|subset'] = np.array([T.A.PB_GOLD[u]['subset'] for u in c.ids])
    FIT[f'{mode}/processbench/{prm}'] = fitinfo(tel)
    for arm, flags in (('PRM native', c.arms['PRM'] > 0), ('C4U (PRM + telemetry)', full['flags']),
                       ('C4U-T (telemetry only)', tel['flags'])):
        m = T.pb_metrics(c, T.pb_predict(c, flags))
        SAVE[f'{mode}|processbench|{prm}|{arm}|pred'] = T.pb_predict(c, flags)
        for s in T.SUBSETS:
            for kk in ('TPR', 'TNR', 'F1'):
                add(mode, prm, 'processbench', arm, f'{s} {kk}', m[s][kk])
        for kk in ('average F1', 'macro SLA', 'overall SLA', 'macro SLA lo', 'macro SLA hi'):
            add(mode, prm, 'processbench', arm, kk, m[kk])
        add(mode, prm, 'processbench', arm, 'flag rate', 100 * float(np.mean(flags)))
        add(mode, prm, 'processbench', arm, 'answers with no flag %', 100 * float(np.mean(T.pb_predict(c, flags) == -1)))
    del c


def parity():
    """PRM native and full C4U rows vs results/c4u_tele16_all_benchmarks_v1/RESULTS_{mode}.csv; TELE16 rows copied."""
    src = {('PRM native', 'prmbench', 'PRMScore'): ('PRISM T2', 'PRM native', 'PRMScore', dict(pool='non-control')),
           ('C4U (PRM + telemetry)', 'prmbench', 'PRMScore'): ('PRISM T2', 'C4U', 'PRMScore', dict(pool='non-control')),
           ('PRM native', 'processbench', 'average F1'): ('PRISM PB T6', 'PRM native', 'average F1', {}),
           ('C4U (PRM + telemetry)', 'processbench', 'average F1'): ('PRISM PB T6', 'C4U', 'average F1', {}),
           ('PRM native', 'processbench', 'macro SLA'): ('MtG T3 SLA', 'PRM native', 'macro SLA', {}),
           ('C4U (PRM + telemetry)', 'processbench', 'macro SLA'): ('MtG T3 SLA', 'C4U', 'macro SLA', {})}
    for mode in ('prm', 'sa'):
        ref = E.read_csv(ROOT / f'results/c4u_tele16_all_benchmarks_v1/RESULTS_{mode}.csv')
        def look(bench, prm, proto, arm, met, kw):
            hit = [float(r['value']) for r in ref if r['prm'] == prm and r['benchmark'] == bench and r['protocol'] == proto
                   and r['arm'] == arm and r['metric'] == met and all(r.get(k) == v for k, v in kw.items())]
            assert len(hit) == 1, (mode, bench, prm, proto, arm, met, len(hit)); return hit[0]
        for r in [r for r in ROWS if r['mode'] == mode]:
            key = (r['arm'], r['benchmark'], r['metric'])
            if key in src:
                proto, arm, met, kw = src[key]; v = look(r['benchmark'], r['prm'], proto, arm, met, kw)
                PAR.append(dict(check=f"{mode} {r['prm']} {key}", got=r['value'], ref=v, ok=abs(r['value'] - v) < 1e-6))
        for prm in N.PRMS:   # TELE16 reference rows, copied (not re-run)
            add(mode, prm, 'prmbench', 'TELE16 (copied)', 'PRMScore', look('prmbench', prm, 'PRISM T2', 'TELE_N16', 'PRMScore', dict(pool='non-control')))
            add(mode, prm, 'processbench', 'TELE16 (copied)', 'average F1', look('processbench', prm, 'PRISM PB T6', 'TELE_N16', 'average F1', {}))
            add(mode, prm, 'processbench', 'TELE16 (copied)', 'macro SLA', look('processbench', prm, 'MtG T3 SLA', 'TELE_N16', 'macro SLA', {}))


def main():
    for mode in ('sa', 'prm'):
        C.MODE = mode
        for prm in N.PRMS:
            prmbench(mode, prm); processbench(mode, prm)
            print(mode, prm, 'done', flush=True)
    parity()
    E.write_csv(HERE / 'RESULTS.csv', ROWS)
    json.dump(FIT, open(HERE / 'FIT.json', 'w'), indent=1, default=float)
    np.savez_compressed(HERE / 'PER_ANSWER.npz', **SAVE)   # inputs of boot.py (gitignored .npz)
    fails = [p['check'] for p in PAR if not p['ok']]
    json.dump(dict(n_checks=len(PAR), failed=fails, checks=PAR), open(HERE / 'PARITY.json', 'w'), indent=1, default=float)
    print('rows', len(ROWS), 'parity', len(PAR), 'failed', fails)
    if fails: raise SystemExit(3)


if __name__ == '__main__':
    main()
