"""S3 orchestration: contribution residual (plan v1.1, section 9).

Frozen protocol: results/ssl_pseudolabel_residual_v1/S3/PROTOCOL.json.  Outputs to S3/<run_id>/.
Labels enter only spectral_utils.ssl_eval; the residualizer and direction never see them.
"""
from pathlib import Path
import json, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from spectral_utils import ssl_s1 as S1, ssl_s3 as S3
from spectral_utils.ssl_eval import Frame, evaluate, bootstrap, rescue_rows, dump, sha, R, MAIN

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260923'
STAGE = ROOT / 'results/ssl_pseudolabel_residual_v1/S3'; OUT = STAGE / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
ARMS = ['R_CONTRIB', 'R_CONTRIB_ONLY', 'R_RANDOM_DIR', 'R_RANDOM_DIR_s0', 'R_RANDOM_DIR_s1', 'R_RANDOM_DIR_s2']; TASKS = ['pb_q4', 'pb_q8', 'prm']
COMPARATORS = ['ct7', 'token_lsml', 'token_equal', 'evidence30__all__plain__equal', 'evidence__all__position__equal', 'evidence__all__position2__equal']
T0 = time.perf_counter(); timing = {}
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'run_id': RUN_ID}; dump(OUT / 'RUN_STATUS.json', status)
F = Frame(); off, n, nsteps = F.off, F.n, F.nsteps
prof = np.load(R / 'profiles_full.npy', mmap_mode='r'); Pmat = np.ascontiguousarray(prof[:, :, 0])
inputs = {'profiles_full.npy': R / 'profiles_full.npy', 'OOF_ANSWERS.csv': R / 'OOF_ANSWERS.csv', 'OOF_STEP_SCORES.npz': R / 'OOF_STEP_SCORES.npz', 'prmbench_prm.pkl': F.prm_meta_path, 'FOLDS_V2.json': MAIN / 'results/localization_source_group_audit_v1/FOLDS_V2.json'}
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': sha(v)} for k, v in inputs.items()} | {'population': {'answers': n, 'pb': int(F.pb.sum()), 'prm': int(F.prm.sum()), 'steps': int(off[-1])}, 'labels': 'v3', 'folds': 'source_groups_v2'})
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel in ['spectral_utils/ssl_s1.py', 'spectral_utils/ssl_s3.py', 'spectral_utils/ssl_eval.py', 'scripts/experiments/ssl_s3_run.py', 'tests/test_ssl_s3.py']:
    dst = snap / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(ROOT / rel, dst); code[rel] = sha(ROOT / rel)
git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head': git('rev-parse', 'HEAD'), 'dirty': git('status', '--porcelain', '--', *code), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json')})
dump(OUT / 'SPLITS.json', F.splits_json(TASKS))

# ------------------------------------------------------------------ Z, BASE, contributions for every answer (label-free)
Zall = np.empty_like(Pmat); base = np.empty(int(off[-1])); Hc = np.empty_like(Pmat)
for i in range(n):
    a, b = off[i:i+2]; Z = S1.answer_z(Pmat[a:b]); Zall[a:b] = Z; task = 'pb' if F.pb[i] else 'prm'
    Hc[a:b] = S3.contributions(Z, task); base[a:b] = Hc[a:b].sum(1)
    assert np.allclose(base[a:b], S1.teacher_pb(Z) if F.pb[i] else S1.teacher_prm(Z), atol=1e-12)
scores = {'BASE': base} | {m: np.full(int(off[-1]), np.nan) for m in ARMS}
cal_scores = {}; fit_log = []; failures = []; diag = []; fallback_answers = []; dirs = {}

def step_weights(idx, task):
    """plan 5.3 answer mass spread over the answer's steps."""
    w_ans = S1.answer_weights(F.cells[idx], F.groups[idx], task)
    return np.concatenate([np.full(nsteps[i], w_ans[j] / nsteps[i]) for j, i in enumerate(idx)])

for task in TASKS:
    for k in range(5):
        t = time.perf_counter(); rm = F.roles(task, k); A = np.flatnonzero(rm['A']); H = np.flatnonzero(rm['H']); C = np.flatnonzero(rm['C'])
        rowsA = np.concatenate([np.arange(off[i], off[i+1]) for i in A]); wA = step_weights(A, task); cellsA = np.repeat(F.cells[A], nsteps[A])
        HA = Hc[rowsA]; bA = base[rowsA]
        rz = S3.fit_residualizer(HA, bA, wA); RA, UA = S3.residualize(HA, bA, rz)
        cov, ncell = S3.cell_averaged_cov(UA, wA, cellsA); d = S3.neutral_direction(cov, rz['active'])
        rnd = {s: S3.random_direction(rz['active'], s) for s in (0, 1, 2)}
        wn = wA / wA.sum()
        def wcorr(x, y):
            mx, my = wn @ x, wn @ y; return float((wn @ ((x - mx) * (y - my))) / np.sqrt(max((wn @ (x - mx) ** 2) * (wn @ (y - my) ** 2), 1e-300)))
        auxA = UA @ d['v']
        rec = {'model_id': f'{task}/fold{k}/contrib', 'role': 'A', 'A_answers': len(A), 'A_steps': len(rowsA), 'cells_in_A': ncell, 'active_channels': int(rz['active'].sum()), 'inactive': [S1.CHANNELS[c] for c in np.flatnonzero(~rz['active'])],
               'status': d['status'], 'reason': d.get('reason', ''), 'chosen_eigenvalue': d.get('eigenvalue'), 'eigengap': d.get('eigengap'), 'eigenvalues': d.get('eigenvalues'), 'component_sum': d.get('component_sum'), 'v_neutral': d['v'],
               'beta': rz['beta'], 'alpha': rz['alpha'], 'sd_raw_residual': rz['sd'], 'corr_aux_base_A': wcorr(auxA, bA) if d['status'] == 'OK' else None,
               'corr_h_base_A_before': [wcorr(HA[:, c], bA) for c in range(11)], 'corr_U_base_A_after': [wcorr(UA[:, c], bA) for c in range(11)],
               'variance_along_eigendirections': d.get('eigenvalues'), 'random_dirs': {s: r['v'] for s, r in rnd.items()}, 'elapsed_s': time.perf_counter() - t}
        fit_log.append(rec); dirs[(task, k)] = d['v']
        for role, idx in [('H', H), ('C', C)]:
            for i in idx:
                a, b = off[i:i+2]; _, U = S3.residualize(Hc[a:b], base[a:b], rz); out = {}
                if d['status'] == 'OK': aux = U @ d['v']; out['R_CONTRIB'] = S3.corrected(base[a:b], aux); out['R_CONTRIB_ONLY'] = aux.copy()
                else:
                    out['R_CONTRIB'] = base[a:b].copy(); out['R_CONTRIB_ONLY'] = np.zeros(b - a)
                    if role == 'H': fallback_answers.append({'task': task, 'fold': k, 'answer': int(i), 'reason': d.get('reason', '')})
                rs = []
                for s in (0, 1, 2):
                    v = rnd[s]['v']; sc = S3.corrected(base[a:b], U @ v) if rnd[s]['status'] == 'OK' else base[a:b].copy(); out[f'R_RANDOM_DIR_s{s}'] = sc; rs.append(sc)
                out['R_RANDOM_DIR'] = np.mean(rs, 0)
                if role == 'H':
                    for m in ARMS: scores[m][a:b] = out[m]
                elif F.prm[i]:
                    for m in ARMS: cal_scores.setdefault((m, k), {})[int(i)] = out[m]
        if task == 'prm': cal_scores[('BASE', k)] = {int(i): base[off[i]:off[i+1]] for i in C}
        if d['status'] != 'OK': failures.append({'stage': 'S3', 'task': task, 'fold': k, 'arm': 'R_CONTRIB', 'reason': d.get('reason', ''), 'retry': 0, 'fallback': 'baseline-preserving (BASE kept), reported separately'})
        # H-side diagnostics
        rowsH = np.concatenate([np.arange(off[i], off[i+1]) for i in H]); UH = S3.residualize(Hc[rowsH], base[rowsH], rz)[1]
        diag.append({'task': task, 'fold': k, 'status': d['status'], 'chosen_eigenvalue': d.get('eigenvalue'), 'eigengap': d.get('eigengap'), 'component_sum': d.get('component_sum'), 'active_channels': int(rz['active'].sum()),
                     'corr_aux_base_A': rec['corr_aux_base_A'], 'corr_aux_base_H': float(np.corrcoef(UH @ d['v'], base[rowsH])[0, 1]) if d['status'] == 'OK' else None,
                     'var_aux_H': float((UH @ d['v']).var()) if d['status'] == 'OK' else None, 'mean_abs_corr_U_base_H': float(np.mean([abs(np.corrcoef(UH[:, c], base[rowsH])[0, 1]) for c in np.flatnonzero(rz['active'])])),
                     **{f'load_{S1.CHANNELS[c]}': float(d['v'][c]) for c in range(11)}})
        print(f'{task} fold{k}: {d["status"]} eigenvalue {d.get("eigenvalue")} gap {d.get("eigengap")} corr(aux,base|A) {rec["corr_aux_base_A"]}, {time.perf_counter()-t:.0f}s', flush=True)
timing['fit_and_score_s'] = time.perf_counter() - T0
with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for r in fit_log: f.write(json.dumps(r, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else str(x)) + '\n')
D = pd.DataFrame(diag)
# direction stability across folds (cosine), per task
stab = []
for task in TASKS:
    vs = [dirs[(task, k)] for k in range(5)]
    for i in range(5):
        for j in range(i + 1, 5): stab.append({'task': task, 'fold_a': i, 'fold_b': j, 'cosine': float(vs[i] @ vs[j])})
D.to_csv(OUT / 'RESIDUAL_DIAGNOSTICS.csv', index=False); pd.DataFrame(stab).to_csv(OUT / 'DIRECTION_STABILITY.csv', index=False)
pd.DataFrame(failures or [{'stage': 'S3', 'task': '', 'fold': '', 'arm': '', 'reason': 'none', 'retry': 0, 'fallback': ''}]).to_csv(OUT / 'FAILURES.csv', index=False)
pd.DataFrame(fallback_answers or [{'task': '', 'fold': '', 'answer': '', 'reason': 'none'}]).to_csv(OUT / 'FALLBACK_ANSWERS.csv', index=False)
for c in COMPARATORS: scores[c] = F.Zs[c]
METHODS = ['BASE'] + ARMS + COMPARATORS
np.savez_compressed(OUT / 'OOF_STEP_SCORES.npz', offsets=off, uid=F.uid, **{m: scores[m] for m in METHODS})

t = time.perf_counter(); pred, auc, metrics, covered = evaluate(F, scores, ['BASE'] + ARMS, COMPARATORS, cal_scores, OUT); timing['eval_s'] = time.perf_counter() - t; t = time.perf_counter()
contrasts = [('R_CONTRIB', 'BASE', True), ('R_CONTRIB', 'R_RANDOM_DIR', True), ('R_CONTRIB_ONLY', 'BASE', False), ('R_RANDOM_DIR', 'BASE', False)] + [(f'R_RANDOM_DIR_s{s}', 'BASE', False) for s in (0, 1, 2)] + [(a, c, False) for a in ['BASE', 'R_CONTRIB'] for c in COMPARATORS]
crow, inv_rate = bootstrap(F, pred, auc, covered, METHODS, contrasts, K=4, out=OUT); timing['bootstrap_s'] = time.perf_counter() - t
pd.DataFrame(rescue_rows(F, pred, auc, scores, ARMS)).to_csv(OUT / 'RESCUE_MATRIX.csv', index=False)
timing['total_s'] = time.perf_counter() - T0; timing['output_bytes'] = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()); dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE' if not failures else 'INCOMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'fits': len(fit_log), 'failures': len(failures), 'unidentified_directions': int((D.status != 'OK').sum()), 'fallback_H_answers': len(fallback_answers), 'methods': METHODS, 'invalid_bootstrap_draw_rate': inv_rate}); dump(OUT / 'RUN_STATUS.json', status)
print(pd.DataFrame(crow)[['contrast_id', 'primary', 'endpoint', 'delta', 'ci95_lo', 'ci95_hi']].to_string()); print(json.dumps(status, indent=1, default=str)); print(json.dumps(timing, indent=1))
