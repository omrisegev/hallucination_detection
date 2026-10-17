"""Readout-oracle and token-level ceilings on the common-miss cohort (PB, development, labels used only to define cohorts).

Question: for the 885 answers no archived method localizes, is the labelled step recoverable from the SAME
frozen token streams by ANY step readout (readout hypothesis), or is there no extreme token there at all
(feature hypothesis)? Streams = innovation5 bank: H0lim, VE0, VE0.75, VE1 (oriented, natural units) + prefix innovation.
"""
import sys, io, json, sqlite3, csv
from pathlib import Path
ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection\.worktrees\temporal-research-20260915'); sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_temporal_research_baseline as base
from spectral_utils.temporal_research_features import prefix_innovation
SRC = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = ROOT / 'results/claude_real_checks_v1'

READOUTS = ['top10', 'max', 'mean', 'median', 'tophalf', 'first4', 'last4', 'top3']

def step_readouts(x, spans):
    """x: tokens of one stream; returns dict readout -> per-step value."""
    out = {k: np.empty(len(spans)) for k in READOUTS}
    for j, (a, b) in enumerate(spans):
        r = np.sort(x[a:b])[::-1]; n = len(r)
        out['top10'][j] = r[:min(10, n)].mean(); out['max'][j] = r[0]; out['mean'][j] = r.mean(); out['median'][j] = np.median(r)
        out['tophalf'][j] = r[:max(1, int(np.ceil(n / 2)))].mean(); out['top3'][j] = r[:min(3, n)].mean()
        out['first4'][j] = x[a:min(b, a + 4)].mean(); out['last4'][j] = x[max(a, b - 4):b].mean()
    return out

def rank_of(values, t):
    """1 = highest; ties counted pessimistically (earlier-step tie-break of the evaluator is argmax-first)."""
    v = values; return int(1 + np.sum(v > v[t]) + np.sum((v == v[t]) & (np.arange(len(v)) < t)))

def main():
    records, joined = base.load_contract(SRC); target = joined['target']
    uid_to_idx = {r['uid']: i for i, r in enumerate(records)}
    cm = list(csv.DictReader(open(ROOT / 'results/predictor_error_profiles_v1/COMMON_MISSES.csv', encoding='utf8')))
    cohort = {}
    for r in cm:
        i = uid_to_idx[r['uid']]
        if r['no_archive_peak_correct_even_without_gate'] == 'True': cohort[i] = 'loc_miss_open' if r['gate_open'] == 'True' else 'loc_miss_closed'
        else: cohort[i] = 'gate_blocked'
    pb_err = [i for i, r in enumerate(records) if r['cell'].startswith('pb_') and target[i] >= 0]
    for i in pb_err: cohort.setdefault(i, 'caught')
    con = sqlite3.connect('file:' + str(ROOT / 'results/temporal_research_baseline_v1/CHECKPOINT.sqlite') + '?mode=ro', uri=True)
    rows = []
    for i in pb_err:
        blob, = con.execute('SELECT payload FROM answers WHERE idx=?', (i,)).fetchone()
        with np.load(io.BytesIO(blob), allow_pickle=False) as f: M = f['features'].astype(float); spans = f['spans']
        streams = np.column_stack((M[:, :4], prefix_innovation(M[:, 0])[0]))
        t = int(target[i]); T, S = streams.shape; a, b = spans[t]
        per = [step_readouts(streams[:, s], spans) for s in range(S)]
        fused = {k: np.mean([per[s][k] for s in range(S)], axis=0) for k in READOUTS}
        ranks = {k: rank_of(fused[k], t) for k in READOUTS}
        # token-level: does the labelled step contain the answer-wide max token of any stream? top-3 token of any stream?
        z = (streams - streams.mean(0)) / np.maximum(streams.std(0), 1e-12)
        contains_max = any(a <= int(np.argmax(z[:, s])) < b for s in range(S))
        contains_top3 = any(np.any((np.argsort(-z[:, s])[:3] >= a) & (np.argsort(-z[:, s])[:3] < b)) for s in range(S))
        # best z inside the step vs best outside, per stream, then max over streams
        inside = z[a:b].max(0); outside = np.concatenate((z[:a], z[b:])).max(0) if T > (b - a) else np.full(S, -np.inf)
        rows.append(dict(idx=i, cohort=cohort[i], cell=records[i]['cell'], steps=len(spans), tokens=T, step_share=(b - a) / T,
                         rank_oracle=min(ranks.values()), best_readout=min(ranks, key=ranks.get),
                         contains_max=contains_max, contains_top3=contains_top3, inside_minus_outside=float((inside - outside).max()),
                         **{'rank_' + k: v for k, v in ranks.items()}))
    con.close()
    def summarize(sub):
        n = len(sub); f = lambda key, fn: float(np.mean([fn(r) for r in sub]))
        return dict(answers=n, median_steps=float(np.median([r['steps'] for r in sub])), median_step_share=float(np.median([r['step_share'] for r in sub])),
                    top10_rank1=f('', lambda r: r['rank_top10'] == 1), top10_rank_le2=f('', lambda r: r['rank_top10'] <= 2), top10_median_rank=float(np.median([r['rank_top10'] for r in sub])),
                    oracle_rank1=f('', lambda r: r['rank_oracle'] == 1), oracle_rank_le2=f('', lambda r: r['rank_oracle'] <= 2), oracle_median_rank=float(np.median([r['rank_oracle'] for r in sub])),
                    **{f'readout_{k}_rank1': f('', lambda r, k=k: r['rank_' + k] == 1) for k in READOUTS},
                    contains_answer_max_token_any_stream=f('', lambda r: r['contains_max']), chance_contains_max=float(np.mean([1 - (1 - r['step_share']) ** 5 for r in sub])),
                    contains_top3_token_any_stream=f('', lambda r: r['contains_top3']),
                    inside_minus_outside_max_z_median=float(np.median([r['inside_minus_outside'] for r in sub])),
                    inside_beats_outside=f('', lambda r: r['inside_minus_outside'] > 0))
    summary = {c: summarize([r for r in rows if r['cohort'] == c]) for c in ('caught', 'gate_blocked', 'loc_miss_open', 'loc_miss_closed')}
    (OUT / 'MISS_READOUT_CEILING.json').write_text(json.dumps(summary, indent=1), encoding='utf8')
    with open(OUT / 'MISS_READOUT_CEILING_ROWS.csv', 'w', newline='', encoding='utf8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    keys = list(next(iter(summary.values())))
    print(f"{'metric':44s}" + ''.join(f'{c:>16s}' for c in summary))
    for k in keys: print(f'{k:44s}' + ''.join(f"{summary[c][k]:16.4f}" if isinstance(summary[c][k], float) else f"{summary[c][k]:16d}" for c in summary))

if __name__ == '__main__': main()
