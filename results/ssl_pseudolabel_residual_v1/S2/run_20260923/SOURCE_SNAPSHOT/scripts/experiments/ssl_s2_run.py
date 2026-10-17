"""S2 orchestration: prediction residual with a linear control (plan v1.1, section 8).

Frozen protocol: results/ssl_pseudolabel_residual_v1/S2/PROTOCOL.json.  Outputs to S2/<run_id>/.
No label reaches fit_ridge; labels enter only spectral_utils.ssl_eval (evaluation) and the diagnostics.
"""
from pathlib import Path
import json, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from spectral_utils import ssl_s1 as S1, ssl_s2 as S2
from spectral_utils.ssl_eval import Frame, evaluate, bootstrap, rescue_rows, read, dump, sha, R, MAIN

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260923'
STAGE = ROOT / 'results/ssl_pseudolabel_residual_v1/S2'; OUT = STAGE / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
P = json.loads((STAGE / 'PROTOCOL.json').read_text(encoding='utf8'))
ARMS = ['R_TEMP', 'R_ZERO', 'R_NORESET', 'R_ONLY', 'R_ABS']; TASKS = ['pb_q4', 'pb_q8', 'prm']
COMPARATORS = ['ct7', 'token_lsml', 'token_equal', 'evidence30__all__plain__equal', 'evidence__all__position__equal', 'evidence__all__position2__equal']
SEED = 20260923; N_SAMPLE = 16384
T0 = time.perf_counter(); timing = {}
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'run_id': RUN_ID}; dump(OUT / 'RUN_STATUS.json', status)

F = Frame(); off, n, nsteps = F.off, F.n, F.nsteps
TOK = MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz'
tk = np.load(TOK); tokens = tk['tokens']; toff = tk['token_offsets']; spans = tk['step_spans']; assert list(tk['channels']) == S1.CHANNELS and spans.shape[0] == off[-1]
prof = np.load(R / 'profiles_full.npy', mmap_mode='r'); Pmat = np.ascontiguousarray(prof[:, :, 0])
inputs = {'TOKEN_MATRICES.npz': TOK, 'profiles_full.npy': R / 'profiles_full.npy', 'OOF_ANSWERS.csv': R / 'OOF_ANSWERS.csv', 'OOF_STEP_SCORES.npz': R / 'OOF_STEP_SCORES.npz', 'prmbench_prm.pkl': F.prm_meta_path, 'FOLDS_V2.json': MAIN / 'results/localization_source_group_audit_v1/FOLDS_V2.json'}
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': sha(v)} for k, v in inputs.items()} | {'population': {'answers': n, 'pb': int(F.pb.sum()), 'prm': int(F.prm.sum()), 'steps': int(off[-1]), 'tokens': int(toff[-1])}, 'labels': 'v3', 'folds': 'source_groups_v2'})
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel in ['spectral_utils/ssl_s1.py', 'spectral_utils/ssl_s2.py', 'spectral_utils/ssl_eval.py', 'scripts/experiments/ssl_s2_run.py', 'tests/test_ssl_s2.py']:
    dst = snap / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(ROOT / rel, dst); code[rel] = sha(ROOT / rel)
git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head': git('rev-parse', 'HEAD'), 'dirty': git('status', '--porcelain', '--', *code), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json')})
dump(OUT / 'SPLITS.json', F.splits_json(TASKS))

# ------------------------------------------------------------------ BASE (plan 5.2 teacher, as in S1)
base = np.empty(int(off[-1]))
for i in range(n):
    a, b = off[i:i+2]; Z = S1.answer_z(Pmat[a:b]); base[a:b] = S1.teacher_pb(Z) if F.pb[i] else S1.teacher_prm(Z)
scores = {'BASE': base} | {m: np.full(int(off[-1]), np.nan) for m in ARMS}
cal_scores = {}; fit_log = []; failures = []; samples = {}; diag_acc = {}
def zt_of(i):
    ta, tb = toff[i:i+2]; return S2.robust_standardize_tokens(tokens[ta:tb].astype(float))

# ------------------------------------------------------------------ per task / fold: fit ridge on A, score H and C
for task in TASKS:
    for k in range(5):
        t = time.perf_counter(); rm = F.roles(task, k); A = np.flatnonzero(rm['A']); H = np.flatnonzero(rm['H']); C = np.flatnonzero(rm['C'])
        rng = np.random.default_rng(np.random.SeedSequence([SEED, TASKS.index(task), k]))
        ai, ti = S2.hierarchical_token_sample(rng, F.cells[A] if task != 'prm' else ['prm'] * len(A), F.groups[A], A, np.diff(toff)[A], N_SAMPLE)
        samples[f'{task}__fold{k}__answer'] = ai; samples[f'{task}__fold{k}__token'] = ti
        X = np.empty((N_SAMPLE, 16 * 11 + 17)); Y = np.empty((N_SAMPLE, 11)); order = np.argsort(ai, kind='stable'); pos = 0
        for i in np.unique(ai):
            z = zt_of(i); Xi = S2.lag_features(z); rows = ti[ai == i]; cnt = len(rows)
            X[order[pos:pos + cnt]] = Xi[rows]; Y[order[pos:pos + cnt]] = z[rows]; pos += cnt
        assert pos == N_SAMPLE
        try: model = S2.fit_ridge(X, Y)
        except Exception as e:
            failures.append({'stage': 'S2', 'task': task, 'fold': k, 'arm': 'R_TEMP', 'reason': str(e), 'retry': 0, 'fallback': 'none'}); continue
        train_mse = float(((X @ model['W'] + model['b'] - Y) ** 2).mean()); zero_mse = float((Y ** 2).mean())
        fit_log.append({'model_id': f'{task}/fold{k}/ridge', 'role': 'A', 'train_answers_distinct': int(len(np.unique(ai))), 'train_tokens': N_SAMPLE, 'alpha': model['alpha'], 'd': model['d'], 'train_mse': train_mse, 'train_mse_zero_predictor': zero_mse, 'W_fro': float(np.linalg.norm(model['W'])), 'seed': SEED, 'elapsed_s': time.perf_counter() - t, 'device': 'cpu'})
        acc = diag_acc.setdefault(task, {key: np.zeros((2, 11)) for key in ['n', 'sz', 'szz', 'sp', 'spp', 'szp', 'se', 'see', 'sez', 'sq', 'sqq', 'sqz', 'snr', 'snrr']})
        for role, idx in [('H', H), ('C', C)]:
            for i in idx:
                a, b = off[i:i+2]; z = zt_of(i); sp = spans[a:b]; pr = S2.predict_ridge(z, model); nr = S2.noreset_prediction(z)
                e = z - pr; aux = {'R_TEMP': S2.step_aux(e, sp), 'R_ZERO': S2.step_aux(z, sp), 'R_NORESET': S2.step_aux(z - nr, sp), 'R_ABS': S2.step_aux(np.abs(e), sp)}
                out = {m: S2.corrected(base[a:b], aux[m]) for m in ['R_TEMP', 'R_ZERO', 'R_NORESET', 'R_ABS']}; out['R_ONLY'] = aux['R_TEMP'].copy()
                if role == 'H':
                    for m in ARMS: scores[m][a:b] = out[m]
                    T = len(z); late = (np.arange(T) >= 16).astype(int)
                    for g in (0, 1):
                        sel = late == g
                        if not sel.any(): continue
                        zz, pp, ee, nn = z[sel], pr[sel], e[sel], (z - nr)[sel]
                        acc['n'][g] += sel.sum(); acc['sz'][g] += zz.sum(0); acc['szz'][g] += (zz ** 2).sum(0); acc['sp'][g] += pp.sum(0); acc['spp'][g] += (pp ** 2).sum(0); acc['szp'][g] += (zz * pp).sum(0)
                        acc['se'][g] += ee.sum(0); acc['see'][g] += (ee ** 2).sum(0); acc['sez'][g] += (ee * zz).sum(0); acc['snr'][g] += nn.sum(0); acc['snrr'][g] += (nn ** 2).sum(0)
                elif F.prm[i]:
                    for m in ARMS: cal_scores.setdefault((m, k), {})[int(i)] = out[m]
        if task == 'prm': cal_scores[('BASE', k)] = {int(i): base[off[i]:off[i+1]] for i in C}
        print(f'{task} fold{k}: ridge train MSE {train_mse:.4f} (zero {zero_mse:.4f}), {time.perf_counter()-t:.0f}s', flush=True)
timing['fit_and_score_s'] = time.perf_counter() - T0
with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for r in fit_log: f.write(json.dumps(r) + '\n')
np.savez_compressed(OUT / 'TRAINING_SAMPLES.npz', **samples, seed=SEED, n=N_SAMPLE)
pd.DataFrame(failures or [{'stage': 'S2', 'task': '', 'fold': '', 'arm': '', 'reason': 'none', 'retry': 0, 'fallback': ''}]).to_csv(OUT / 'FAILURES.csv', index=False)
for c in COMPARATORS: scores[c] = F.Zs[c]
METHODS = ['BASE'] + ARMS + COMPARATORS
np.savez_compressed(OUT / 'OOF_STEP_SCORES.npz', offsets=off, uid=F.uid, **{m: scores[m] for m in METHODS})

# ------------------------------------------------------------------ residual diagnostics (label-free)
rows = []
for task, acc in diag_acc.items():
    for g, name in [(0, 't<16'), (1, 't>=16')]:
        N = acc['n'][g]; mz = acc['sz'][g] / N; vz = acc['szz'][g] / N - mz ** 2; mp = acc['sp'][g] / N; vp = acc['spp'][g] / N - mp ** 2; cov_zp = acc['szp'][g] / N - mz * mp
        me = acc['se'][g] / N; ve = acc['see'][g] / N - me ** 2; cov_ez = acc['sez'][g] / N - me * mz
        for c, ch in enumerate(S1.CHANNELS):
            rows.append({'task': task, 'tokens': name, 'channel': ch, 'n_tokens': int(N[c]), 'mse_ridge': acc['see'][g][c] / N[c], 'mse_zero': acc['szz'][g][c] / N[c], 'mse_noreset': acc['snrr'][g][c] / N[c],
                         'corr_pred_z': cov_zp[c] / np.sqrt(max(vp[c] * vz[c], 1e-300)), 'corr_resid_z': cov_ez[c] / np.sqrt(max(ve[c] * vz[c], 1e-300)), 'var_resid': ve[c], 'var_z': vz[c],
                         'resid_var_after_regressing_on_z_posthoc': ve[c] - cov_ez[c] ** 2 / max(vz[c], 1e-300)})
pd.DataFrame(rows).to_csv(OUT / 'RESIDUAL_DIAGNOSTICS.csv', index=False)

# ------------------------------------------------------------------ evaluation + bootstrap + rescue
t = time.perf_counter()
pred, auc, metrics, covered = evaluate(F, scores, ['BASE'] + ARMS, COMPARATORS, cal_scores, OUT)
timing['eval_s'] = time.perf_counter() - t; t = time.perf_counter()
contrasts = [('R_TEMP', 'BASE', True), ('R_TEMP', 'R_ZERO', True), ('R_ZERO', 'BASE', False), ('R_NORESET', 'BASE', False), ('R_TEMP', 'R_NORESET', False), ('R_ABS', 'BASE', False), ('R_ONLY', 'BASE', False), ('R_ABS', 'R_TEMP', False)] + [(a, c, False) for a in ['BASE', 'R_TEMP', 'R_ZERO'] for c in COMPARATORS]
crow, inv_rate = bootstrap(F, pred, auc, covered, METHODS, contrasts, K=4, out=OUT)
timing['bootstrap_s'] = time.perf_counter() - t
pd.DataFrame(rescue_rows(F, pred, auc, scores, ARMS)).to_csv(OUT / 'RESCUE_MATRIX.csv', index=False)
timing['total_s'] = time.perf_counter() - T0; timing['output_bytes'] = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()); dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE' if not failures else 'INCOMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'fits': len(fit_log), 'failures': len(failures), 'methods': METHODS, 'invalid_bootstrap_draw_rate': inv_rate}); dump(OUT / 'RUN_STATUS.json', status)
print(pd.DataFrame(crow)[['contrast_id', 'primary', 'endpoint', 'delta', 'ci95_lo', 'ci95_hi']].to_string()); print(json.dumps(status, indent=1, default=str)); print(json.dumps(timing, indent=1))
