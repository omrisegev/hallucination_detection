"""Cheap candidate views already in the raw cache, tested on the PB error cohorts (development; labels only define cohorts).

Candidate token streams (teacher-forced PB answers, Qwen3-4B/8B scorers):
  surprisal_provided : -log p(provided token)            (token_spilled_energies)
  rank_provided      : rank of the provided token in the saved top-50 (50 = absent)
  mass_above         : probability mass the scorer put on tokens it ranked above the provided one
  gap_top1           : logp(top-1) - logp(provided)
  tail15 / tail50    : log mass outside the top-15 / top-50 (via token_logsumexp when present)
  digit_disagree     : provided token is a digit and the scorer's top-1 is a different digit (needs digit ids)
Existing reference streams: the innovation5 bank (H0lim, VE0, VE0.75, VE1, prefix innovation) from the frozen checkpoint.

For each stream: token-level ceiling (labelled step contains the answer-wide max / a top-3 token), single-stream
Top10 step readout rank of the labelled step, within-answer correlation with H0lim and with the fused innovation5
token score, and the participation ratio of the augmented bank.
"""
import sys, io, csv, json, pickle, sqlite3, pathlib
import numpy as np
ROOT = pathlib.Path(r'C:\Users\omris\TAU\hallucination_detection\.worktrees\temporal-research-20260915'); sys.path.insert(0, str(ROOT))
SRC = pathlib.Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = ROOT / 'results/claude_real_checks_v1'
from scripts import run_temporal_research_baseline as base
from scripts import run_direct_probability_temporal as ev
from spectral_utils.temporal_research_features import prefix_innovation
from spectral_utils.direct_probability_fusion import step_top_mean

def main():
    records, joined = base.load_contract(SRC); target = joined['target']
    uid_to_idx = {r['uid']: i for i, r in enumerate(records)}
    cm = list(csv.DictReader(open(ROOT / 'results/predictor_error_profiles_v1/COMMON_MISSES.csv', encoding='utf8')))
    cohort = {}
    for r in cm:
        i = uid_to_idx[r['uid']]
        cohort[i] = ('loc_miss_open' if r['gate_open'] == 'True' else 'loc_miss_closed') if r['no_archive_peak_correct_even_without_gate'] == 'True' else 'gate_blocked'
    pb_err = [i for i, r in enumerate(records) if r['cell'].startswith('pb_') and target[i] >= 0]
    for i in pb_err: cohort.setdefault(i, 'caught')
    digit_ids = None
    p = OUT / 'digit_token_ids.json'
    if p.exists(): digit_ids = set(json.load(open(p)))
    con = sqlite3.connect('file:' + str(ROOT / 'results/temporal_research_baseline_v1/CHECKPOINT.sqlite') + '?mode=ro', uri=True)
    NEW = ['surprisal_provided', 'rank_provided', 'mass_above', 'gap_top1', 'tail15', 'tail50'] + (['digit_disagree'] if digit_ids else [])
    REF = ['H0lim', 'VE0', 'VE075', 'VE1', 'innovation', 'fused_innovation5']
    rows = []; second = np.zeros((5 + len(NEW), 5 + len(NEW))); n_second = 0; absent_total = 0; tokens_total = 0
    for cell, path, kind, dataset in ev.source_specs():
        if kind != 'pb': continue
        idx = [i for i in pb_err if records[i]['cell'] == cell]
        if not idx: continue
        print('[load]', cell, len(idx), flush=True)
        srcrows = ev.old._source_row_map(ev.old.load_pickle(path), kind=kind, dataset=dataset)
        for i in idx:
            row = srcrows[records[i]['row_id']]
            tk = ev.old._topk_payload(row); ids = np.asarray(tk['ids']); lp = np.asarray(tk['logprobs'], dtype=np.float64)
            gen = np.asarray(row['gen_token_ids']); T = len(gen)
            blob, = con.execute('SELECT payload FROM answers WHERE idx=?', (i,)).fetchone()
            with np.load(io.BytesIO(blob), allow_pickle=False) as f: M = f['features'].astype(float); spans = f['spans']
            assert len(M) == T and spans.shape[0] == records[i]['steps']
            bank = np.column_stack((M[:, :4], prefix_innovation(M[:, 0])[0]))
            fused = bank.mean(1)  # natural-unit equal fusion of the innovation5 token streams
            hit = ids == gen[:, None]; present = hit.any(1); pos = np.where(present, hit.argmax(1), 50)
            absent_total += int((~present).sum()); tokens_total += T
            spill = np.asarray(row['token_spilled_energies'], dtype=np.float64)
            surpr = np.where(present, -lp[np.arange(T), np.minimum(pos, 49)], spill)
            probs = np.exp(lp); cum = np.cumsum(probs, axis=1)
            mass_above = np.where(present, np.where(pos > 0, cum[np.arange(T), np.maximum(pos - 1, 0)], 0.), cum[:, -1])
            gap = lp[:, 0] - (-surpr)
            lse = row.get('token_logsumexp'); lse = np.asarray(lse, dtype=np.float64) if lse is not None else np.zeros(T)
            head15 = np.log(np.maximum(1 - np.exp(lp[:, :15] - lse[:, None]).sum(1), 1e-12)); head50 = np.log(np.maximum(1 - np.exp(lp - lse[:, None]).sum(1), 1e-12))
            new = {'surprisal_provided': surpr, 'rank_provided': pos.astype(float), 'mass_above': mass_above, 'gap_top1': gap, 'tail15': head15, 'tail50': head50}
            if digit_ids:
                isdig = np.isin(gen, list(digit_ids)); top1dig = np.isin(ids[:, 0], list(digit_ids))
                new['digit_disagree'] = (isdig & top1dig & (ids[:, 0] != gen)).astype(float)
            t = int(target[i]); a, b = spans[t]
            all_streams = {**{k: bank[:, j] for j, k in enumerate(REF[:5])}, 'fused_innovation5': fused, **new}
            rec = dict(idx=i, cohort=cohort[i], cell=cell, steps=len(spans), share=(b - a) / T)
            for k, x in all_streams.items():
                z = (x - x.mean()) / max(x.std(), 1e-12)
                rec[f'{k}__contains_max'] = bool(a <= int(np.argmax(z)) < b)
                top3 = np.argsort(-z)[:3]; rec[f'{k}__contains_top3'] = bool(np.any((top3 >= a) & (top3 < b)))
                s = step_top_mean(x, spans[:, 0], spans[:, 1], 10); rk = 1 + int(np.sum(s > s[t]) + np.sum((s == s[t]) & (np.arange(len(s)) < t)))
                rec[f'{k}__rank'] = rk
                if k in new: rec[f'{k}__corr_H0lim'] = float(np.corrcoef(x, bank[:, 0])[0, 1]) if x.std() > 0 else 0.; rec[f'{k}__corr_fused'] = float(np.corrcoef(x, fused)[0, 1]) if x.std() > 0 else 0.
            Z = np.column_stack([all_streams[k] for k in REF[:5] + NEW]); Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-12)
            second += Z.T @ Z / T; n_second += 1
            rows.append(rec)
    con.close()
    C = second / n_second; ev_ = np.linalg.eigvalsh(C)[::-1]
    pr_all = float(ev_.sum() ** 2 / (ev_ ** 2).sum()); e5 = np.linalg.eigvalsh(C[:5, :5])[::-1]; pr5 = float(e5.sum() ** 2 / (e5 ** 2).sum())
    cohorts = ('caught', 'gate_blocked', 'loc_miss_open', 'loc_miss_closed')
    summary = {'participation_ratio_bank5': pr5, 'participation_ratio_bank5_plus_new': pr_all, 'absent_from_top50_fraction': absent_total / tokens_total,
               'mean_within_answer_corr_matrix': {'names': REF[:5] + NEW, 'matrix': np.round(C, 3).tolist()}, 'cohorts': {}}
    for c in cohorts:
        sub = [r for r in rows if r['cohort'] == c]; s = {'answers': len(sub), 'chance_contains_max_1stream': float(np.mean([r['share'] for r in sub]))}
        for k in REF + NEW:
            s[k] = dict(contains_max=float(np.mean([r[f'{k}__contains_max'] for r in sub])), contains_top3=float(np.mean([r[f'{k}__contains_top3'] for r in sub])),
                        top10_rank1=float(np.mean([r[f'{k}__rank'] == 1 for r in sub])), top10_rank_le2=float(np.mean([r[f'{k}__rank'] <= 2 for r in sub])))
            if k in NEW: s[k].update(corr_H0lim=float(np.median([r[f'{k}__corr_H0lim'] for r in sub])), corr_fused=float(np.median([r[f'{k}__corr_fused'] for r in sub])))
        s['union_new_contains_max'] = float(np.mean([any(r[f'{k}__contains_max'] for k in NEW) for r in sub]))
        s['union_bank5_contains_max'] = float(np.mean([any(r[f'{k}__contains_max'] for k in REF[:5]) for r in sub]))
        summary['cohorts'][c] = s
    (OUT / 'NEW_VIEW_CEILING.json').write_text(json.dumps(summary, indent=1), encoding='utf8')
    with open(OUT / 'NEW_VIEW_CEILING_ROWS.csv', 'w', newline='', encoding='utf8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('participation ratio bank5 %.3f -> bank5+new %.3f ; provided token absent from top50: %.4f' % (pr5, pr_all, absent_total / tokens_total))
    print('median within-answer corr with fused innovation5 (caught cohort):', {k: round(summary['cohorts']['caught'][k]['corr_fused'], 3) for k in NEW})
    print(f"\n{'stream':20s}" + ''.join(f'{c:>34s}' for c in cohorts)); print(' ' * 20 + ''.join(f"{'max/top3/rank1/rank<=2':>34s}" for _ in cohorts))
    for k in REF + NEW:
        print(f'{k:20s}' + ''.join(f"{summary['cohorts'][c][k]['contains_max']:8.3f}{summary['cohorts'][c][k]['contains_top3']:8.3f}{summary['cohorts'][c][k]['top10_rank1']:9.3f}{summary['cohorts'][c][k]['top10_rank_le2']:9.3f}" for c in cohorts))
    print(f"{'chance (1 stream)':20s}" + ''.join(f"{summary['cohorts'][c]['chance_contains_max_1stream']:8.3f}{'':26s}" for c in cohorts))
    print(f"{'union bank5':20s}" + ''.join(f"{summary['cohorts'][c]['union_bank5_contains_max']:8.3f}{'':26s}" for c in cohorts))
    print(f"{'union new':20s}" + ''.join(f"{summary['cohorts'][c]['union_new_contains_max']:8.3f}{'':26s}" for c in cohorts))

if __name__ == '__main__': main()
