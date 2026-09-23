"""S5-CPU orchestration: learned attention pooling inside a step (plan v1.1 section 11, without S4).

Frozen protocol: results/ssl_pseudolabel_residual_v1/S5/PROTOCOL.json.  Outputs to S5/<run_id>/.
No label reaches any fit; labels enter only spectral_utils.ssl_eval (evaluation) and the diagnostics.
"""
from pathlib import Path
import json, os, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from spectral_utils import ssl_s1 as S1, ssl_s5 as S5
from spectral_utils.ssl_eval import Frame, evaluate, bootstrap, rescue_rows, read, dump, sha, R, MAIN

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260923'
STAGE = ROOT / 'results/ssl_pseudolabel_residual_v1/S5'; OUT = STAGE / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
P = json.loads((STAGE / 'PROTOCOL.json').read_text(encoding='utf8'))
NEURAL = [('H_MEAN', 'mean', True), ('H_RANDATT', 'attention', False), ('H_RAW', 'attention', True)]
ARMS = ['H_TOP5'] + [a for a, _, _ in NEURAL]
TASKS = ['pb_q4', 'pb_q8', 'prm']
COMPARATORS = ['ct7', 'token_lsml', 'token_equal', 'evidence30__all__plain__equal',
               'evidence__all__position__equal', 'evidence__all__position2__equal']
SEED = 20260923; SEEDS = list(S5.SEEDS); UPDATES = S5.UPDATES; BATCH = S5.BATCH_ANSWERS
# smoke overrides: any value here makes the run self-labelling as a feasibility check, never a result
SMOKE = {k: os.environ[k] for k in ['S5_UPDATES', 'S5_FOLDS', 'S5_TASKS', 'S5_SEEDS', 'S5_MAXB'] if k in os.environ}
UPDATES = int(SMOKE.get('S5_UPDATES', UPDATES))
FOLDS = [int(x) for x in SMOKE['S5_FOLDS'].split(',')] if 'S5_FOLDS' in SMOKE else list(range(5))
if 'S5_TASKS' in SMOKE: TASKS = SMOKE['S5_TASKS'].split(',')
if 'S5_SEEDS' in SMOKE: SEEDS = [int(x) for x in SMOKE['S5_SEEDS'].split(',')]
MAXB = int(SMOKE.get('S5_MAXB', 0))
T0 = time.perf_counter(); timing = {}
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'run_id': RUN_ID, 'smoke_overrides': SMOKE}
dump(OUT / 'RUN_STATUS.json', status)
torch.set_num_threads(4)

# ------------------------------------------------------------------ inputs
F = Frame(); off, n, nsteps = F.off, F.n, F.nsteps
TOK = MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz'
tk = np.load(TOK); tokens = tk['tokens']; toff = tk['token_offsets']; spans = tk['step_spans']
assert list(tk['channels']) == S1.CHANNELS and spans.shape[0] == int(off[-1])
prof = np.load(R / 'profiles_full.npy', mmap_mode='r'); Pmat = np.ascontiguousarray(prof[:, :, 0])
inputs = {'TOKEN_MATRICES.npz': TOK, 'profiles_full.npy': R / 'profiles_full.npy',
          'OOF_ANSWERS.csv': R / 'OOF_ANSWERS.csv', 'OOF_STEP_SCORES.npz': R / 'OOF_STEP_SCORES.npz',
          'prmbench_prm.pkl': F.prm_meta_path,
          'FOLDS_V2.json': MAIN / 'results/localization_source_group_audit_v1/FOLDS_V2.json'}
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel in ['spectral_utils/ssl_s1.py', 'spectral_utils/ssl_s5.py', 'spectral_utils/ssl_eval.py',
            'scripts/experiments/ssl_s5_run.py', 'tests/test_ssl_s5.py']:
    dst = snap / rel; dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / rel, dst); code[rel] = sha(ROOT / rel)
git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head': git('rev-parse', 'HEAD'),
                                  'dirty': git('status', '--porcelain', '--', *code),
                                  'protocol_sha256': sha(STAGE / 'PROTOCOL.json'),
                                  'torch': torch.__version__, 'threads': torch.get_num_threads()})
dump(OUT / 'SPLITS.json', F.splits_json(TASKS))

# ------------------------------------------------------------------ representation
# per-answer, per-channel standardized tokens; only tokens that lie inside a step are kept
t = time.perf_counter()
cov_n = np.array([(spans[off[i]:off[i + 1], 1] - spans[off[i]:off[i + 1], 0]).sum() for i in range(n)], dtype=np.int64)
cov_off = np.concatenate([[0], np.cumsum(cov_n)])
ZT = np.empty((int(cov_off[-1]), 11), dtype=np.float32)
SEGL = np.empty(int(cov_off[-1]), dtype=np.int32)
for i in range(n):
    z = S5.standardize_tokens_answer(tokens[toff[i]:toff[i + 1]].astype(np.float64))
    sp = spans[off[i]:off[i + 1]]
    rows = np.concatenate([np.arange(a, b) for a, b in sp])
    ZT[cov_off[i]:cov_off[i + 1]] = z[rows].astype(np.float32)
    SEGL[cov_off[i]:cov_off[i + 1]] = np.repeat(np.arange(len(sp)), sp[:, 1] - sp[:, 0])
del tokens, tk
ZTt = torch.from_numpy(ZT)
dump(OUT / 'INPUT_MANIFEST.json',
     {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': sha(v)} for k, v in inputs.items()} |
     {'population': {'answers': n, 'pb': int(F.pb.sum()), 'prm': int(F.prm.sum()),
                     'steps': int(off[-1]), 'tokens_total': int(toff[-1]),
                     'tokens_inside_a_step': int(cov_off[-1])},
      'normalization': 'per answer, per channel over ALL of the answer tokens; only tokens inside a step are kept',
      'labels': 'v3', 'folds': 'source_groups_v2'})
timing['representation_s'] = time.perf_counter() - t
print('representation: {:,} tokens in {:.0f}s'.format(int(cov_off[-1]), timing['representation_s']), flush=True)

# ------------------------------------------------------------------ BASE teacher + P_SOFT targets (identical arrays)
base = np.empty(int(off[-1])); Zstep = np.empty((int(off[-1]), 11))
for i in range(n):
    a, b = off[i:i + 2]; Z = S1.answer_z(Pmat[a:b]); Zstep[a:b] = Z
    base[a:b] = S1.teacher_pb(Z) if F.pb[i] else S1.teacher_prm(Z)
scores = {'BASE': base}
for m in ARMS:
    scores[m] = np.full(int(off[-1]), np.nan)
cal_scores = {}; fit_log = []; failures = []; diag = []; att_acc = {}; loss_traces = {}; risk_w = []


def batch_tensors(idx):
    """(f, seg, nseg, q, step_answer, nans, w_ans) for a list of answer indices."""
    idx = np.asarray(idx)
    f = torch.cat([ZTt[cov_off[i]:cov_off[i + 1]] for i in idx])
    bases = np.concatenate([[0], np.cumsum(nsteps[idx])[:-1]])
    seg = np.concatenate([SEGL[cov_off[i]:cov_off[i + 1]].astype(np.int64) + s for i, s in zip(idx, bases)])
    ns = int(nsteps[idx].sum())
    q = torch.from_numpy(np.concatenate([base[off[i]:off[i + 1]] for i in idx]).astype(np.float32))
    sa = torch.from_numpy(np.repeat(np.arange(len(idx)), nsteps[idx]))
    return f, torch.from_numpy(seg), ns, q, sa, len(idx), torch.full((len(idx),), 1. / len(idx))


def step_weights(idx, w):
    return np.concatenate([np.full(nsteps[i], w[j] / nsteps[i]) for j, i in enumerate(idx)])


# ------------------------------------------------------------------ per task / fold
for ti, task in enumerate(TASKS):
    for k in FOLDS:
        tf = time.perf_counter(); rm = F.roles(task, k)
        B = np.flatnonzero(rm['B']); H = np.flatnonzero(rm['H']); C = np.flatnonzero(rm['C'])
        Hset = set(int(i) for i in H)
        if MAXB: B = B[:MAXB]; H = H[:MAXB]; C = C[:MAXB]; Hset = set(int(i) for i in H)
        wB = S1.answer_weights(F.cells[B], F.groups[B], 'prm' if task == 'prm' else 'pb')
        wstep = step_weights(B, wB)
        score_idx = np.concatenate([H, C]) if task == 'prm' else H

        # --- H_TOP5: the S1 P_SOFT student on the frozen top5 step profile (L-BFGS, no seeds)
        try:
            X = [Zstep[off[i]:off[i + 1]] for i in B]; T = [base[off[i]:off[i + 1]] for i in B]
            m5 = S1.fit_head(X, T, wB, 'prm' if task == 'prm' else 'pb')
            if not m5['converged']:
                failures.append({'stage': 'S5', 'task': task, 'fold': k, 'arm': 'H_TOP5', 'seed': -1,
                                 'reason': 'L-BFGS not converged: ' + m5['message'], 'retry': 0, 'fallback': 'BASE'})
            uB = np.concatenate([S1.predict_head(Zstep[off[i]:off[i + 1]], m5) for i in B])
            mu, sd = S5.weighted_mean_sd(uB, wstep)
            for i in score_idx:
                a, b = off[i:i + 2]
                s = S5.standardize_scores(S1.predict_head(Zstep[a:b], m5), mu, sd)
                if int(i) in Hset:
                    scores['H_TOP5'][a:b] = s
                elif F.prm[i]:
                    cal_scores.setdefault(('H_TOP5', k), {})[int(i)] = s
            fit_log.append({'model_id': task + '/fold' + str(k) + '/H_TOP5', 'arm': 'H_TOP5', 'role': 'B', 'seed': -1,
                            'train_answers': len(B), 'converged': bool(m5['converged']), 'iterations': m5['iterations'],
                            'loss': m5['loss'], 'grad_inf_norm': m5['grad_inf_norm'], 'mu_B': mu, 'sd_B': sd,
                            'elapsed_s': time.perf_counter() - tf, 'device': 'cpu'})
        except Exception as e:
            failures.append({'stage': 'S5', 'task': task, 'fold': k, 'arm': 'H_TOP5', 'seed': -1,
                             'reason': str(e), 'retry': 0, 'fallback': 'BASE'})

        # --- neural arms
        for ai, (arm, pooling, train_att) in enumerate(NEURAL):
            u_seed = {int(i): [] for i in score_idx}
            for sdv in SEEDS:
                ts = time.perf_counter()
                rng = np.random.default_rng(np.random.SeedSequence([SEED, ti, k, ai, sdv]))

                def batches(_it, _rng=rng, _B=B, _w=wB):
                    return batch_tensors(_B[_rng.choice(len(_B), BATCH, p=_w)])

                try:
                    model, trace = S5.train_head(batches, task, 11, pooling=pooling,
                                                 train_attention=train_att, seed=sdv, updates=UPDATES)
                except Exception as e:
                    failures.append({'stage': 'S5', 'task': task, 'fold': k, 'arm': arm, 'seed': sdv,
                                     'reason': str(e), 'retry': 0, 'fallback': 'BASE'})
                    continue
                key = task + '__fold' + str(k) + '__' + arm + '__seed' + str(sdv)
                loss_traces[key] = trace.astype(np.float32)
                rw = model.risk.weight.detach().numpy().ravel()
                row = {'task': task, 'fold': k, 'arm': arm, 'seed': sdv, 'bias': float(model.risk.bias)}
                row.update({c: float(v) for c, v in zip(S1.CHANNELS, rw)})
                risk_w.append(row)
                uB = np.concatenate([S5.score_answer(model, *batch_tensors([i])[:3])[0] for i in B])
                mu, sd = S5.weighted_mean_sd(uB, wstep)
                fit_log.append({'model_id': task + '/fold' + str(k) + '/' + arm, 'arm': arm, 'role': 'B', 'seed': sdv,
                                'train_answers': len(B), 'converged': bool(np.isfinite(trace[-1])), 'iterations': UPDATES,
                                'loss': float(trace[-50:].mean()), 'loss_first50': float(trace[:50].mean()),
                                'grad_inf_norm': None, 'mu_B': mu, 'sd_B': sd,
                                'elapsed_s': time.perf_counter() - ts, 'device': 'cpu'})
                acc = att_acc.setdefault((task, arm), {'steps': 0, 'ent': 0., 'first': 0., 'last': 0.,
                                                       'top': 0., 'dz': np.zeros(11), 'answers': 0})
                for i in score_idx:
                    f, seg, ns = batch_tensors([i])[:3]
                    u, alpha, _ = S5.score_answer(model, f, seg, ns)
                    if not np.isfinite(u).all():
                        failures.append({'stage': 'S5', 'task': task, 'fold': k, 'arm': arm, 'seed': sdv,
                                         'reason': 'non-finite u on answer ' + str(int(i)), 'retry': 0, 'fallback': 'BASE'})
                        continue
                    u_seed[int(i)].append(S5.standardize_scores(u, mu, sd))
                    if sdv == SEEDS[0] and int(i) in Hset:
                        sl = SEGL[cov_off[i]:cov_off[i + 1]].astype(np.int64)
                        z = ZT[cov_off[i]:cov_off[i + 1]].astype(np.float64)
                        cnt = np.bincount(sl, minlength=ns).astype(float)
                        ent = -np.bincount(sl, weights=alpha * np.log(np.maximum(alpha, 1e-300)), minlength=ns)
                        multi = cnt > 1
                        if not multi.any():
                            continue
                        bnd = np.concatenate([[0], np.cumsum(cnt).astype(int)])
                        acc['steps'] += int(multi.sum()); acc['answers'] += 1
                        acc['ent'] += float((ent[multi] / np.log(cnt[multi])).sum())
                        acc['first'] += float((alpha[bnd[:-1]][multi] * cnt[multi]).sum())
                        acc['last'] += float((alpha[bnd[1:] - 1][multi] * cnt[multi]).sum())
                        acc['top'] += float((np.maximum.reduceat(alpha, bnd[:-1])[multi] * cnt[multi]).sum())
                        wm = np.stack([np.bincount(sl, weights=alpha * z[:, c], minlength=ns) for c in range(11)], 1)
                        um = np.stack([np.bincount(sl, weights=z[:, c], minlength=ns) / cnt for c in range(11)], 1)
                        acc['dz'] += (wm[multi] - um[multi]).sum(0)
            for i in score_idx:
                a, b = off[i:i + 2]
                if len(u_seed[int(i)]) != len(SEEDS):
                    continue
                s = np.mean(u_seed[int(i)], axis=0)
                if int(i) in Hset:
                    scores[arm][a:b] = s
                elif F.prm[i]:
                    cal_scores.setdefault((arm, k), {})[int(i)] = s
        print('{} fold{}: B={} H={} done in {:.0f}s'.format(task, k, len(B), len(H), time.perf_counter() - tf), flush=True)
timing['fit_and_score_s'] = time.perf_counter() - T0

# ------------------------------------------------------------------ baseline-preserving fallback
fallback = {}
for m in ARMS:
    miss = np.array([not np.isfinite(scores[m][off[i]:off[i + 1]]).all() for i in range(n)])
    fallback[m] = int(miss.sum())
    for i in np.flatnonzero(miss):
        scores[m][off[i]:off[i + 1]] = base[off[i]:off[i + 1]]
for k in range(5):
    cal_scores[('BASE', k)] = {int(i): base[off[i]:off[i + 1]] for i in np.flatnonzero(F.prm & (F.fold == (k + 1) % 5))}

with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for r in fit_log:
        f.write(json.dumps(r, default=str) + '\n')
pd.DataFrame(failures or [{'stage': 'S5', 'task': '', 'fold': '', 'arm': '', 'seed': '',
                           'reason': 'none', 'retry': 0, 'fallback': ''}]).to_csv(OUT / 'FAILURES.csv', index=False)
pd.DataFrame(risk_w).to_csv(OUT / 'TOKEN_RISK_WEIGHTS.csv', index=False)
np.savez_compressed(OUT / 'LOSS_TRACES.npz', **loss_traces)
for (task, arm), a in att_acc.items():
    s = max(a['steps'], 1)
    row = {'task': task, 'arm': arm, 'answers': a['answers'], 'multi_token_steps': a['steps'],
           'attention_entropy_ratio': a['ent'] / s, 'first_token_mass_x_n': a['first'] / s,
           'last_token_mass_x_n': a['last'] / s, 'max_token_mass_x_n': a['top'] / s}
    row.update({'delta_' + c: a['dz'][ci] / s for ci, c in enumerate(S1.CHANNELS)})
    diag.append(row)
pd.DataFrame(diag).to_csv(OUT / 'ATTENTION_DIAGNOSTICS.csv', index=False)
for c in COMPARATORS:
    scores[c] = F.Zs[c]
METHODS = ['BASE'] + ARMS + COMPARATORS
np.savez_compressed(OUT / 'OOF_STEP_SCORES.npz', offsets=off, uid=F.uid, **{m: scores[m] for m in METHODS})

# ------------------------------------------------------------------ evaluation + bootstrap + rescue
t = time.perf_counter()
pred, auc, metrics, covered = evaluate(F, scores, ['BASE'] + ARMS, COMPARATORS, cal_scores, OUT)
timing['eval_s'] = time.perf_counter() - t; t = time.perf_counter()
contrasts = [('H_RAW', 'BASE', True), ('H_RAW', 'H_MEAN', True), ('H_RAW', 'ct7', True),
             ('H_RAW', 'H_RANDATT', False), ('H_RAW', 'H_TOP5', False), ('H_MEAN', 'BASE', False),
             ('H_TOP5', 'BASE', False), ('H_RANDATT', 'BASE', False), ('H_MEAN', 'H_TOP5', False)]
contrasts += [(a, c, False) for a in ['BASE'] + ARMS for c in COMPARATORS]
crow, inv_rate = bootstrap(F, pred, auc, covered, METHODS, contrasts, K=6, out=OUT)
timing['bootstrap_s'] = time.perf_counter() - t
pd.DataFrame(rescue_rows(F, pred, auc, scores, ARMS)).to_csv(OUT / 'RESCUE_MATRIX.csv', index=False)
timing['total_s'] = time.perf_counter() - T0
timing['output_bytes'] = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE' if not failures else 'INCOMPLETE',
               'finished': datetime.now().isoformat(timespec='seconds'), 'fits': len(fit_log),
               'failures': len(failures), 'fallback_answers': fallback, 'methods': METHODS,
               'invalid_bootstrap_draw_rate': inv_rate})
dump(OUT / 'RUN_STATUS.json', status)
print(pd.DataFrame(crow)[['contrast_id', 'primary', 'endpoint', 'delta', 'ci95_lo', 'ci95_hi']].head(30).to_string())
print(json.dumps(status, indent=1, default=str))
print(json.dumps(timing, indent=1))
