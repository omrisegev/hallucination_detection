"""Smoke checks of the one SSL experiment on the 12-row runs (2026-11-18). CPU, seconds. No labels are read.

    python smoke_check.py --runs DIR --variants VARDIR --stored-readouts DIR_readout_steps_sa_v1 --out SMOKE.json
DIR holds <bench>_<C|V1|V2>/ with the driver's manifest.json and external_telemetry.pkl or prmbench_telemetry.pkl.
Checks:
  S1 every run: manifest status COMPLETE, 0 failed rows; s/row = elapsed_sec / n_rows (includes model load).
  S2 equal prefix (stop rule, ANSWERS 20): V2 vs its parent C on the tokens of steps 1..k-1, V1 vs C on step 1:
     gen_token_ids equal, and token_entropies / token_spilled_energies / token_logsumexp max abs diff <= 1e-4.
     Also reported (no stop): max abs diff of the 11 risk-oriented token channels on the same tokens.
  S3 C vs stored standalone 7B (ANSWERS 14, 20): exact per-step top10 of the 11 token channels vs
     data_export/readout_steps_sa_v1/<bench>__qwen_prm_7b.npz: Spearman per channel (>= .99) and max abs diff.
Exit 0 only if S1 and S2 pass; S3 is reported (Spearman >= .99 expected on CUDA).
"""
import argparse, json, os, pickle, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(os.environ.get('HD_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'results/bench_gap_diagnosis_v1/readout_alternatives'))
from extract_readouts import mask_like_builders, safe_row  # noqa: E402
from spectral_utils.external_generalization.fusion import validate_telemetry  # noqa: E402

RAWQ = ('token_entropies', 'token_spilled_energies', 'token_logsumexp')


def load_run(d):
    d = Path(d); man = json.load(open(d / 'manifest.json'))
    pk = d / ('external_telemetry.pkl' if (d / 'external_telemetry.pkl').exists() else 'prmbench_telemetry.pkl')
    raw = pickle.load(open(pk, 'rb'))
    return man, {str(e['idx']): e for e in raw.values() if 'failure' not in e}


def tok_values(e):
    m, _, _ = mask_like_builders({0: dict(e)})
    r = m[0]
    return validate_telemetry(safe_row(r)).astype(float), list(r['step_token_spans'])


def prefix_cmp(c, v, n_steps_equal):
    """Compare the unmasked token arrays of steps 1..n_steps_equal (spans of the parent C row)."""
    sc, sv = c['step_token_spans'], v['step_token_spans']
    if n_steps_equal < 1: return None
    end_c = sc[n_steps_equal - 1][1]; end_v = sv[n_steps_equal - 1][1]
    out = dict(tokens=int(end_c), spans_equal=bool([tuple(x) for x in sc[:n_steps_equal]] == [tuple(x) for x in sv[:n_steps_equal]]),
               ids_equal=bool(list(c['gen_token_ids'][:end_c]) == list(v['gen_token_ids'][:end_v])))
    for q in RAWQ:
        a = np.asarray(c[q][:end_c], float); b = np.asarray(v[q][:end_v], float)
        out[q] = float(np.abs(a - b).max()) if len(a) == len(b) and len(a) else float('inf')
    try:
        Vc, spc = tok_values(c); Vv, spv = tok_values(v)
        e1 = spc[n_steps_equal - 1][1]; e2 = spv[n_steps_equal - 1][1]
        out['channels_max_abs_diff'] = float(np.abs(Vc[:e1] - Vv[:e2]).max()) if e1 == e2 and e1 else None
    except Exception as ex:  # report only
        out['channels_error'] = repr(ex)[:200]
    out['pass'] = bool(out['spans_equal'] and out['ids_equal'] and all(out[q] <= 1e-4 for q in RAWQ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True); ap.add_argument('--variants', required=True)
    ap.add_argument('--stored-readouts', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); R = {}; ok = True
    for bench in ('processbench', 'prmbench'):
        runs = {}
        for V in ('C', 'V1', 'V2'):
            d = Path(a.runs) / f'{bench}_{V}'
            if not d.exists(): continue
            man, rows = load_run(d); runs[V] = rows
            r = dict(status=man.get('status'), n_rows=man.get('n_rows'), failed=man.get('n_failed_rows'),
                     s_per_row=round(man['elapsed_sec'] / max(1, man['n_rows']), 2), max_input_tokens=man.get('max_input_tokens'))
            r['pass'] = r['status'] == 'COMPLETE' and r['failed'] == 0; ok &= r['pass']; R[f'S1 {bench} {V}'] = r
        side = {}
        for V in ('V1', 'V2'):
            p = Path(a.variants) / f'SIDECAR_{bench}_{V}.jsonl'
            side[V] = {s['uid']: s for s in map(json.loads, open(p)) if 'skip' not in s}
        for V in ('V1', 'V2'):
            if V not in runs: continue          # this job has no V run (a C job, or the other variant)
            res = []
            for u, e in runs.get(V, {}).items():
                s = side[V][u]; c = runs['C'].get(s['parent_uid'])
                if c is None: continue
                n_eq = s['k'] - 1 if V == 'V2' else 1
                x = prefix_cmp(c, e, n_eq)
                if x: res.append(dict(uid=u, **x))
            p = bool(res) and all(x['pass'] for x in res); ok &= p
            R[f'S2 {bench} {V} equal prefix'] = dict(n_pairs=len(res), pass_=p,
                                                     worst={q: max((x[q] for x in res), default=None) for q in RAWQ},
                                                     channels_worst=max((x.get('channels_max_abs_diff') or 0 for x in res), default=None),
                                                     failures=[x for x in res if not x['pass']][:3])
        st = Path(a.stored_readouts) / f'{bench}__qwen_prm_7b.npz'
        if st.exists() and 'C' in runs:
            D = np.load(st); pos = {u: i for i, u in enumerate(D['uids'].astype(str))}; off = D['offsets']
            mine, ref = [], []
            for u, e in runs['C'].items():
                if u not in pos: continue
                V, sp = tok_values(e); i = pos[u]; r10 = D['readouts'][off[i]:off[i + 1], :, 0]
                t = np.array([np.sort(V[x:y], axis=0)[-min(10, y - x):].mean(0) if y > x else np.full(V.shape[1], np.nan) for x, y in sp])
                if len(t) != len(r10): t = t[~np.isnan(t[:, 0])]
                if len(t) != len(r10): continue
                mine.append(t); ref.append(r10)
            if mine:
                m = np.concatenate(mine); r = np.concatenate(ref)
                rho = [float(spearmanr(m[:, j], r[:, j])[0]) for j in range(m.shape[1])]
                R[f'S3 {bench} C vs stored top10'] = dict(steps=len(m), spearman_min=min(rho), spearman=rho,
                                                          max_abs_diff=float(np.nanmax(np.abs(m - r))), pass_=min(rho) >= .99)
    R['ALL_STOP_RULES_PASS'] = bool(ok)
    json.dump(R, open(a.out, 'w'), indent=1, default=str)
    print(json.dumps(R, indent=1, default=str)[:4000])
    sys.exit(0 if ok else 4)


if __name__ == '__main__':
    main()
