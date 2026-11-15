"""Simes answer gate inside the standalone pipeline (2026-11-15).

Pipeline per cell (benchmark x base model), no PRM, no labels in any fitted quantity:
  1. TELE16 telemetry of the standalone base model (16 channels per step).
  2a. Step score for ranking = TELE16 L-SML on R0 columns (per-answer z of each raw channel)  = "F2 L-SML | R0" of
      gsp_status_review_v1 (the recommended paper method). GAP count on this score -> step flags (baseline).
  2b. Step score for the gate = L-SML on global rank-normal columns, no per-answer z (cross_model_gate_check_v1/steps).
  3. Step p-value p_i = share of all steps of the cell with gate score >= s_i (upper tail, ties counted).
     Simes(answer) = min_i n * p_(i) / i over the n steps of the answer.
     Gate rule, fixed before the run: Simes > ALPHA = .05 -> answer called clean -> no flag in that answer.
  4. Gated flags = baseline GAP flags AND (Simes <= ALPHA). ProcessBench prediction = first flagged step, else -1.
Labels are used only for scoring.

    COLLECTION_DATA=<hd_collection_data> python -B run.py
"""
import gzip, json, os, sys, time
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'results/gsp_ablation_baseline_v1'))
import ablation as AB  # noqa: E402
import r0 as R0  # noqa: E402
E, T, Rn = AB.E, AB.T, AB.Rn

ALPHA = .05                                   # fixed before the run; one value for all cells
BENCHES = ('prmbench', 'processbench')
STEPS = ROOT / 'results/cross_model_gate_check_v1/steps'
SWEEP = np.array([.001, .002, .005, .01, .02, .05, .1, .2, .3, .5, .7, 1.0])   # diagnostic only
DRAWS = 2000
MODEL = {'qwen_prm_7b': 'Qwen2.5-Math-7B-Instruct', 'qwen_prm_72b': 'Qwen2.5-Math-72B-Instruct',
         'pure_prm': 'Qwen2.5-Math-7B', 'rlhflow_prm': 'Llama-3.1-8B-Instruct'}


def read_steps(bench, prm):
    with gzip.open(STEPS / f'{bench}__{prm}.csv.gz', 'rt') as f:
        head = f.readline().strip().split(',')
        rows = [l.rstrip('\n').split(',') for l in f]
    i = {k: head.index(k) for k in ('uid', 'step', 'y', 's')}
    d = {}
    for r in rows:
        d.setdefault(r[i['uid']], []).append((int(r[i['step']]), int(float(r[i['y']])), float(r[i['s']])))
    return d


def align_gate_score(c, d):
    """Global gate score s_g in the step order of cell c. Checks: same answers, same step count, same labels."""
    sg, yg = np.empty(c.off[-1]), np.empty(c.off[-1], bool)
    for j, u in enumerate(c.ids):
        v = sorted(d[u]); a, b = c.off[j], c.off[j + 1]
        assert len(v) == b - a, (u, len(v), b - a)
        sg[a:b] = [x[2] for x in v]; yg[a:b] = [bool(x[1]) for x in v]
    return sg, yg


def simes(c, sg):
    """Simes statistic per answer from step p-values against all steps of the cell."""
    srt = np.sort(sg); n_all = len(sg)
    p = (n_all - np.searchsorted(srt, sg, side='left')) / n_all          # share with score >= s_i
    out = np.ones(len(c.ids))
    for j, (a, b) in enumerate(zip(c.off[:-1], c.off[1:])):
        n = b - a
        if n == 0:
            continue
        q = np.sort(p[a:b]); out[j] = float(np.min(n * q / np.arange(1, n + 1)))
    return np.minimum(out, 1.)


def expand(c, ans_bool):
    return np.repeat(ans_bool, np.diff(c.off))


def pb_parts(c, pred):
    lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids]); sub = np.array([T.A.PB_GOLD[u]['subset'] for u in c.ids])
    return lab, sub, (pred == lab)


def pb_f1_from_hits(hit, lab, sub, idx=None):
    f = []
    for s in T.SUBSETS:
        m = sub == s
        if idx is not None:
            m = idx[s]
            e = m[lab[m] != -1]; k = m[lab[m] == -1]
        else:
            e = np.flatnonzero(m & (lab != -1)); k = np.flatnonzero(m & (lab == -1))
        ea, ca = 100 * hit[e].mean(), 100 * hit[k].mean()
        f.append(2 * ea * ca / (ea + ca) if ea + ca else 0.)
    return float(np.mean(f))


def pb_boot(c, preds):
    """Paired bootstrap of ProcessBench avg F1: resample answers within each subset (2,000 draws)."""
    lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids]); sub = np.array([T.A.PB_GOLD[u]['subset'] for u in c.ids])
    rng = np.random.default_rng(20261115)
    ix = {s: np.flatnonzero(sub == s) for s in T.SUBSETS}
    draws = [{s: rng.choice(v, len(v)) for s, v in ix.items()} for _ in range(DRAWS)]
    out = {}
    for k, pr in preds.items():
        hit = pr == lab
        out[k] = np.array([pb_f1_from_hits(hit, lab, sub)] + [pb_f1_from_hits(hit, lab, sub, d) for d in draws])
    return out


def main():
    t0 = time.time(); rows, boots, gates, sweep = [], [], {}, []
    for bench in BENCHES:
        for prm in AB.PRMS:
            tc = time.time()
            c, X, ab = AB.load_cell(bench, prm)
            rv, act = R0.raw_rows(c, prm, bench); X0 = R0.r0_cols(rv, act, c.off)
            g = {'R0_vs_stored_N16': float(np.abs(X0 - E.channels(c, 'N16')).max())}
            assert g['R0_vs_stored_N16'] < 1e-4, g
            tel = Rn.c4u_t(c, X0); s = tel['s']                         # TELE16 L-SML R0 step score
            f0 = AB.gap(c, s)
            sg, yg = align_gate_score(c, read_steps(bench, prm))
            g['labels_equal'] = bool((yg == c.err).all()); assert g['labels_equal']
            g['n_answers'] = len(c.ids); g['n_steps'] = int(c.off[-1])
            S = simes(c, sg)
            y_ans = np.add.reduceat(c.err.astype(int), c.off[:-1]) > 0
            if bench == 'processbench':
                lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids]); y_ans = lab != -1
            open_ = S <= ALPHA
            f1 = f0 & expand(c, open_)
            gates[f'{bench}/{prm}'] = g
            # answer-level gate description (labels only to describe)
            clean = ~y_ans
            desc = dict(true_clean_share=100 * clean.mean(), gate_closed_share=100 * (~open_).mean(),
                        closed_among_clean=100 * (~open_[clean]).mean(), closed_among_wrong=100 * (~open_[y_ans]).mean(),
                        clean_share_of_closed=100 * clean[~open_].mean() if (~open_).any() else float('nan'))
            for k, v in desc.items():
                rows.append(dict(bench=bench, prm=prm, arm='gate', metric=k, value=float(v)))
            F = {'TELE16 + GAP (no gate)': f0, f'TELE16 + GAP + Simes gate (alpha={ALPHA})': f1}
            for arm, f in F.items():
                sc = AB.score(c, bench, s, f)
                if bench == 'processbench':
                    m = T.pb_metrics(c, T.pb_predict(c, f))
                    sc['error acc (macro)'] = float(np.mean([m[x]['TPR'] for x in T.SUBSETS]))
                    sc['correct acc (macro)'] = float(np.mean([m[x]['TNR'] for x in T.SUBSETS]))
                    for x in T.SUBSETS:
                        sc[f'{x} F1'] = m[x]['F1']
                sc['answers with >=1 flag %'] = 100 * float((np.add.reduceat(f.astype(int), c.off[:-1]) > 0).mean())
                for mk, mv in sc.items():
                    rows.append(dict(bench=bench, prm=prm, arm=arm, metric=mk, value=float(mv)))
            # paired bootstrap of the main metric
            if bench == 'processbench':
                D = pb_boot(c, {k: T.pb_predict(c, f) for k, f in F.items()})
            else:
                W, inv, ng = E.boot_weights(c); wa = np.zeros(len(c.ids))
                D = {k: E.boot_stats(c, E.counts_by_answer(c, f), wa, W, inv, ng)[0] for k, f in F.items()}
            ka, kb = list(F)
            for k in F:
                v = D[k]; boots.append(dict(bench=bench, prm=prm, arm=k, kind='abs', value=v[0],
                                            lo=np.percentile(v[1:], 2.5), hi=np.percentile(v[1:], 97.5)))
            d = D[kb] - D[ka]
            boots.append(dict(bench=bench, prm=prm, arm='gate minus no gate', kind='diff', value=d[0],
                              lo=np.percentile(d[1:], 2.5), hi=np.percentile(d[1:], 97.5)))
            # diagnostic sweep over alpha (not the main result)
            for a in SWEEP:
                o = S <= a; f = f0 & expand(c, o)
                if bench == 'processbench':
                    m = T.pb_metrics(c, T.pb_predict(c, f)); val = m['average F1']
                else:
                    val = float(E.official_from_counts(E.counts_by_answer(c, f).sum(0), bench))
                sweep.append(dict(bench=bench, prm=prm, alpha=float(a), metric=val, closed=100 * float((~o).mean()),
                                  closed_clean=100 * float((~o[clean]).mean()), closed_wrong=100 * float((~o[y_ans]).mean())))
            # answer AUROC of Simes (sanity vs cross-model check)
            from sklearn.metrics import roc_auc_score
            g['simes_answer_auroc'] = float(roc_auc_score(y_ans, -np.log10(np.maximum(S, 1e-300))))
            np.savez_compressed(HERE / 'cells' / f'{bench}__{prm}.npz', ids=np.asarray(c.ids).astype(str), off=c.off,
                                simes=S, y_ans=y_ans)
            print(bench, prm, f'{time.time() - tc:.0f}s', json.dumps(g), json.dumps({k: round(v, 1) for k, v in desc.items()}), flush=True)
            del c
    E.write_csv(HERE / 'RESULTS.csv', rows); E.write_csv(HERE / 'BOOT.csv', boots); E.write_csv(HERE / 'SWEEP.csv', sweep)
    json.dump(dict(alpha=ALPHA, gates=gates, runtime_s=time.time() - t0), open(HERE / 'GATES.json', 'w'), indent=1)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    (HERE / 'cells').mkdir(exist_ok=True)
    main()
