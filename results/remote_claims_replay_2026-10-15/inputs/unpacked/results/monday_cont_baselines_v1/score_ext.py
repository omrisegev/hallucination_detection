"""Stage 1 (sealed): continuous L-SML and IU-PCR on the three external cells, label-free. See AMENDMENT_EXT.json.
Reads no evaluator file. Writes ext/<cell>/PREDICTIONS.npz and ext/SEAL.json."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]; V2 = MAIN / 'results/monday_nested_banks_v2'; EXT = MAIN / 'results/monday_external_panel_v1'
sys.path.insert(0, str(MAIN))
import spectral_utils.fusion_utils, spectral_utils.upcr, spectral_utils.laplacian_upcr  # noqa: E402,F401  same package order as run.py (main copy first)
spec = importlib.util.spec_from_file_location('ext_score', EXT / 'score.py'); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
spec = importlib.util.spec_from_file_location('base_run', OUT / 'run.py'); br = importlib.util.module_from_spec(spec); spec.loader.exec_module(br)
NB = json.loads((V2 / 'BANKS.json').read_text(encoding='utf8')); ext.BN = NB


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    t0 = time.time(); D = OUT / 'ext'
    if (D / 'SEAL.json').exists(): raise SystemExit('already sealed')
    json.loads((OUT / 'AMENDMENT_EXT.json').read_text(encoding='utf8'))
    seal, diag, failures = {}, {}, []
    for cell in ext.CELLS:
        F = np.load(ext.FE / cell / 'FEATURES.npz'); uids = [str(u) for u in F['uids']]; lens = F['nonempty_lengths']; ne = F['nonempty'].astype(bool)
        pop = ext.Population(F, list(NB)); off = pop.off; arrays = {'uids': np.array(uids), 'nonempty': ne, 'lengths': lens}
        for bk in NB:
            bank = pop.cell_bank(bk, np.arange(len(uids))); V = bank['V']; lev = pop.level[bank['rows']]
            for name, fn in (('LSML_CONT', br.fit_lsml), ('IU_PCR', br.fit_iu)):
                try:
                    s, d = fn(V)
                    if not np.all(np.isfinite(s)): raise RuntimeError('non-finite score')
                    d['corr_with_level'] = float(np.corrcoef(s, lev)[0, 1]); diag[f'{cell}/{bk}/{name}'] = d
                except Exception as exc:
                    failures.append({'cell': cell, 'bank': bk, 'scorer': name, 'reason': f'{type(exc).__name__}: {exc}'[:300]}); continue
                az = np.zeros_like(s)
                for a, b in zip(off[:-1], off[1:]): v = s[a:b]; az[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
                flag = az >= np.quantile(az, .8)          # ANSWER_Z_Q80, same formula as v2 external_oriented.py
                full_s = np.full(int(lens.sum()), np.nan); full_s[np.flatnonzero(ne)] = s
                full_p = np.zeros(int(lens.sum()), int); full_p[np.flatnonzero(ne)] = (~flag).astype(int)
                arrays[f'{bk}__{name}__score'] = full_s; arrays[f'{bk}__{name}__pred_correct'] = full_p
        (D / cell).mkdir(parents=True, exist_ok=True); p = D / cell / 'PREDICTIONS.npz'; np.savez_compressed(p, **arrays)
        seal[cell] = {'path': str(p.relative_to(OUT)), 'sha256': sha(p)}
        print(cell, 'scored', round(time.time() - t0), flush=True)
    (D / 'SCORING_DIAGNOSTICS.json').write_text(json.dumps({'diag': diag, 'failures': failures}, indent=1), encoding='utf8')
    (D / 'SEAL.json').write_text(json.dumps({'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'cells': seal, 'n_failures': len(failures),
        'score_ext_sha256': sha(__file__), 'amendment_sha256': sha(OUT / 'AMENDMENT_EXT.json'), 'evaluator_files_read': False}, indent=1), encoding='utf8')
    print('sealed', {c: v['sha256'][:12] for c, v in seal.items()}, 'failures', len(failures))


if __name__ == '__main__':
    main()
