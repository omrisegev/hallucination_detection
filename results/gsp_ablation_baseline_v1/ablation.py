"""GSP ablation baseline on the standalone base-model telemetry (2026-11-12).

Question: where does GSP (= frozen C4U recipe C4U-parisi-within-jaffe-cross-v1) stand now when it runs on TELE16 from
the standalone base model, before any of the advisors' new experiments? One table per benchmark, one change per row.

Cells: PRMBench, Socratic, Hard2Verify, ProcessBench x 4 PRM pairings (Qwen-PRM-7B <- Qwen2.5-Math-7B-Instruct,
Qwen-PRM-72B <- Qwen2.5-Math-72B-Instruct, PURE <- Qwen2.5-Math-7B, RLHFlow <- Llama-3.1-8B-Instruct). The telemetry
is always the standalone base model (sa). The PRM enters only the reference arms.

Arms (rank, flags) per cell:
  A  GSP-T own count        c4u_telemetry_only_v1.run.c4u_t on the 16 sa TELE16 columns (no PRM), k = round(sum q)
  B  GSP-T + GAP            same rank, GAP count (largest sorted-score gap) instead of the posterior count
  C  GSP full own count     frozen c4u_new.c4u on [PRM risk, PRM p, 16 sa columns]
     GSP full + GAP         same rank, GAP count
     PRM native             PRM own threshold (p > .5 / native flag)
     PRM + GAP              PRM risk, GAP count
  D  GSP-T hybXX own / GAP  A and B with the hybrid step readout of the 11 token channels (hybrid.py, added later)
  E  ProcessBench only: change-point first-error rule (pb_first_error_rule_v1/cp_rule.py, 3-state chain, MAP) on
     the GSP-T rank (PRM-free EM, as 'FREE CP' there), the GSP full rank (PRM-anchored) and the PRM (anchored).

Metrics: PRMBench / Socratic official PRMScore, Hard2Verify official balanced F1 (harmonic mean of TPR and TNR),
within-answer AUC of the rank (not on ProcessBench: invalid there), ProcessBench official average F1 (+ Mind the Gap
macro SLA for context). Output: CELLS/<bench>__<prm>.npz (ranks and flags), RESULTS.csv, PARITY.json.

    COLLECTION_DATA=<hd_collection_data> python ablation.py
"""
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'results/c4u_telemetry_only_v1'))
import run as Rn  # noqa: E402
import bracha_panel as BP  # noqa: E402
E, N, T, RA = Rn.E, Rn.N, Rn.T, Rn.RA
S8 = N.S8
CELLS = HERE / 'cells'; CELLS.mkdir(exist_ok=True)
PRMS = list(N.PRMS)
BENCHES = ('prmbench', 'socratic', 'hard2verify', 'processbench')


_common = Rn.C.A.common_uids


def _common_uids(bench):
    """ProcessBench: the 3,400 answers. The cloud copy has no crop_ares_v1/scores (the 5 standalone CAB score files that
    common_uids intersects); every ProcessBench cell holds all 3,400 answers, so the intersection of the 4 PRM score
    files in sub5_all_v1/scores is the same set (checked: 3,400 each, intersection 3,400)."""
    d = Path(E.DATA) / 'crop_ares_v1' / 'scores'
    if bench == 'processbench' and not d.exists():
        S = [set(np.load(f)['uids'].astype(str)) for f in sorted((Path(E.DATA) / 'sub5_all_v1' / 'scores').glob('SCORES_prm_*_processbench.npz'))]
        u = set.intersection(*S); assert len(S) == 4 and len(u) == 3400
        return u
    return _common(bench)


Rn.C.A.common_uids = _common_uids


def gap(c, s):
    return S8.gap_flags(c, s)


def load_cell(bench, prm):
    """Cell c, the 18-column sa matrix X (PRM risk, PRM p, 16 sa TELE16 columns) and the ab vote dict."""
    RA.MODE = 'sa'
    if bench == 'prmbench':
        RA.need(prm, 'prmbench')
        c = E.load('prmbench', f'{prm}__standalone')
        c.arms['PRMPROB'] = 1. - c.extra['reward']; c.arms['PRM'] = c.arms[N.K.prm_arm(c)]
        X, ab, rep, g = T.build_X(c, prm, 'prmbench'); assert rep == 'R1'
    elif bench == 'processbench':
        RA.need(prm, 'processbench')
        c = Rn.C.cab_cell('processbench', prm)
        X, ab, rep, g = T.build_X(c, prm, 'processbench'); assert rep == 'R1'
    else:
        c = E.load(bench, prm)
        Xs = BP.sa_X(c, prm, bench)
        zp = E.answer_z(c.arms['PRM'], c.off); raw = c.arms['PRMPROB']
        X = np.column_stack([zp, (raw - raw.mean()) / raw.std(), Xs]); ab = {1: raw > .5}
    return c, X, ab


def prm_native(c, bench):
    if bench == 'prmbench':
        return c.flags('PRM_native', 'native'), c.arms['PRM_native']
    if bench == 'processbench':
        return c.arms['PRM'] > 0, c.arms['PRM']
    return c.arms['PRMPROB'] > .5, c.arms['PRM']


def prm_gap_arm(c, bench, prm):
    """The PRM arm of Omri's chart (Bracha LR panel): PRM reference scale, GAP rule."""
    if bench == 'prmbench':
        scale = 'PRM_Q80_PROB' if Rn.C.A.META[f'prmbench/{prm}']['prm_reference_scale'] == '1-reward' else 'PRM_Q80'
        return c.flags(scale, 'GAP'), c.z(scale)
    return c.flags('PRM', 'GAP'), c.arms['PRM']


def arms(c, X, ab, bench, prm, Xt=None):
    """Every arm of the ablation on one cell. Xt (optional) replaces the 16 telemetry columns for GSP-T."""
    out = {}
    nf, nr = prm_native(c, bench); out['PRM native'] = (nr, nf)
    gf, gr = prm_gap_arm(c, bench, prm); out['PRM + GAP'] = (gr, gf)
    full = N.c4u(c, X, ab)
    out['GSP full (PRM + tele) own count'] = (full['rank'], full['flags'])
    out['GSP full (PRM + tele) + GAP'] = (full['rank'], gap(c, full['rank']))
    tel = Rn.c4u_t(c, X[:, 2:] if Xt is None else Xt)
    out['GSP-T own count'] = (tel['rank'], tel['flags'])
    out['GSP-T + GAP'] = (tel['rank'], gap(c, tel['rank']))
    fit = dict(gspt_prevalence=tel['prevalence'], gspt_K=len(tel['groups']), gspt_anchor_group=tel['anchor_group'],
               gspt_zero_flag_answers=int((tel['k'] == 0).sum()), full_prevalence=full['prevalence'])
    return out, fit


def score(c, bench, rank, flags):
    if bench == 'processbench':
        m = T.pb_metrics(c, T.pb_predict(c, flags))
        return {'avg F1': m['average F1'], 'macro SLA': m['macro SLA'], 'flag rate': 100 * float(flags.mean())}
    v = float(E.official_from_counts(E.counts_by_answer(c, flags).sum(0), bench))
    keep = None if bench == 'prmbench' else c.keep
    wa = E.per_answer_auc(c.err, rank, c.off, keep)
    name = 'balanced F1' if bench == 'hard2verify' else 'PRMScore'
    return {name: v if bench == 'prmbench' else 100 * v, 'within-AUC': float(np.nanmean(wa)),
            'flag rate': 100 * float(flags.mean())}


def main(benches=BENCHES):
    rows, fits = [], {}
    for bench in benches:
        for prm in PRMS:
            c, X, ab = load_cell(bench, prm)
            A, fit = arms(c, X, ab, bench, prm)
            fits[f'{bench}/{prm}'] = fit
            save = dict(ids=np.asarray(c.ids).astype(str), off=np.asarray(c.off), X=X.astype(np.float32))
            for k, (r, f) in A.items():
                for mk, mv in score(c, bench, r, f).items():
                    rows.append(dict(bench=bench, prm=prm, arm=k, metric=mk, value=mv))
                save[f'rank|{k}'] = np.asarray(r, float); save[f'flags|{k}'] = np.asarray(f, bool)
            np.savez_compressed(CELLS / f'{bench}__{prm}.npz', **save)
            print(bench, prm, {k: round(score(c, bench, *v)[next(iter(score(c, bench, *v)))], 2) for k, v in A.items()},
                  flush=True)
            del c
    E.write_csv(HERE / f'RESULTS_{"_".join(benches)}.csv', rows)
    json.dump(fits, open(HERE / f'FIT_{"_".join(benches)}.json', 'w'), indent=1)


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or BENCHES)
