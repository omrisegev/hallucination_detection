"""S0 of the SSL / pseudo-label / residual localization plan (v1.1, section 6).

Read-only audit of Claude Step 432 (A1).  No model is fitted, no inference runs.
Every number is recomputed from the frozen OOF arrays with independent code; the
frozen PRMBench evaluator (spectral_utils/prmbench.py) is used only to replay the
official PRMScore from the saved decision masks.

Outputs -> results/ssl_pseudolabel_residual_v1/S0/
"""
from pathlib import Path
import hashlib, json, math, pickle, sys, time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[2]          # this worktree
MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection')   # frozen inputs live in the main checkout tree
W = MAIN / '.worktrees/readout-quickest-detection-v1'
R = W / 'results/step_evidence_v1'
OUT = ROOT / 'results/ssl_pseudolabel_residual_v1/S0'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MAIN))
from spectral_utils.prmbench import prmbench_evaluate   # frozen official evaluator port

TOL = 8 * np.finfo(float).eps
KEY = ['evidence__all__seed__equal', 'evidence__all__plain__equal', 'evidence__all__position__equal',
       'evidence__all__plain2__equal', 'evidence__all__position2__equal', 'evidence__all__randomseed__equal',
       'evidence__all__randomseed_position__equal', 'evidence__all__prioronly__equal',
       'evidence__all__ceiling__equal', 'evidence__all__plain__continuous_lsml',
       'evidence30__all__seed__equal', 'evidence30__all__plain__equal', 'evidence30__all__position__equal',
       'ct7', 'token_lsml', 'token_equal']


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
def first_argmax(v):
    return int(np.flatnonzero(v >= v.max() - TOL)[0])
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

t0 = time.perf_counter()
log = {}
# ---------------------------------------------------------------- inputs
summary = pd.read_csv(R / 'SUMMARY.csv', encoding='utf-8-sig').set_index('method')
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
Z = np.load(R / 'OOF_STEP_SCORES.npz')
off = Z['offsets']; labels = Z['labels']; n = len(ans); nsteps = np.diff(off)
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb
target = ans.target.to_numpy(); gate = ans.ct7_gate.to_numpy().astype(bool)
fold = ans.fold.to_numpy(); cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy()
step_len = np.load(R / 'step_lengths.npy')
methods = [m for m in Z.files if m not in ('offsets', 'labels')]
assert list(summary.index) == methods, 'SUMMARY/NPZ method order differs'

# ---------------------------------------------------------------- 1. RUN_INVENTORY
freeze_in = read(R / 'INPUT_FREEZE.json'); run_freeze = read(R / 'RUN_FREEZE.json'); manifest = read(R / 'REPORT_MANIFEST.json')
summary_mtime = (R / 'SUMMARY.csv').stat().st_mtime
inv = []
for p in sorted((R / 'jobs').glob('*.json')):
    j = read(p); npz = p.with_suffix('.npz')
    inv.append({'job': p.stem, 'task': j['task'], 'outer_fold': j['fold'], 'inner_fold': j['inner_fold'],
                'roster': j['roster'], 'stage': j['stage'], 'train_answers': j['train_answers'], 'test_answers': j['test_answers'],
                'gate_open_train': j['gate_open_train_answers'], 'pseudo_positive_total': j['pseudo_positive_total'],
                'pseudo_negative_train_answers': j['train_answers'] - int(j['pseudo_positive_total']) if j['task'] != 'prm' else 0,
                'pseudo_rule': j.get('pseudo_positive_rule', 'seed argmax of gate-open training answers (original protocol)'),
                'iteration2_changed_frac': j['iteration2_changed_pseudo_labels'], 'fit_seconds': j['seconds'],
                'json_mtime': datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds'),
                'npz_mtime': datetime.fromtimestamp(npz.stat().st_mtime).isoformat(timespec='seconds'),
                'newer_than_summary': p.stat().st_mtime > summary_mtime or npz.stat().st_mtime > summary_mtime,
                'lineage': 'A1_rerun' if j['task'] == 'prm' else 'original_2026-09-22',
                'manifest_sha_ok': sha(npz) == manifest['jobs'].get(str(Path('jobs') / npz.name).replace('/', '\\'), manifest['jobs'].get('jobs/' + npz.name, {})).get('sha256')})
inv = pd.DataFrame(inv); inv.to_csv(OUT / 'RUN_INVENTORY.csv', index=False)
log['inventory'] = {'jobs': len(inv), 'outer': int(inv.inner_fold.isna().sum()), 'inner': int(inv.inner_fold.notna().sum()),
                    'stale_jobs_newer_than_summary': int(inv.newer_than_summary.sum()), 'manifest_sha_ok_all': bool(inv.manifest_sha_ok.all()),
                    'summary_mtime': datetime.fromtimestamp(summary_mtime).isoformat(timespec='seconds'),
                    'prm_pseudo_rules': sorted(inv[inv.task == 'prm'].pseudo_rule.unique().tolist()),
                    'prm_min_pseudo_positive': float(inv[inv.task == 'prm'].pseudo_positive_total.min()),
                    'pb_min_pseudo_positive': float(inv[inv.task != 'prm'].pseudo_positive_total.min())}
# input freeze: do the frozen inputs still hash the same today?
inputs = {}
for k, v in freeze_in.items():
    if k == 'protocol' or not isinstance(v, dict): continue
    p = Path(v['path']); ok = p.exists() and p.stat().st_size == v['bytes']
    inputs[k] = {'exists': p.exists(), 'bytes_ok': ok, 'sha_ok': ok and sha(p) == v['sha256']}
log['input_freeze_today'] = inputs
snap = R / 'source_snapshot'
log['source_freeze'] = {f: {'snapshot_sha': sha(snap / f), 'run_freeze_sha': run_freeze.get(f)} for f in ['core.py', 'data.py', 'scoring.py', 'uncertainty.py', 'report.py', 'readout.py', 'step_evidence_v1.py']}
log['source_freeze']['driver_vs_module'] = {'scripts/experiments/step_evidence_v1.py': sha(W / 'scripts/experiments/step_evidence_v1.py'),
                                            'spectral_utils/step_evidence_v1.py': sha(W / 'spectral_utils/step_evidence_v1.py'),
                                            'note': 'RUN_FREEZE key step_evidence_v1.py is the MODULE hash (basename collision); the driver hash is not frozen.'}

# ---------------------------------------------------------------- 2. coverage / contract
folds_v2 = read(MAIN / 'results/localization_source_group_audit_v1/FOLDS_V2.json')['outer']
cov = {'answers': n, 'pb': int(pb.sum()), 'prm': int(prm.sum()), 'steps': int(off[-1]), 'pb_erroneous': int((pb & (target >= 0)).sum()),
       'pb_clean': int((pb & (target < 0)).sum()), 'prm_two_class_answers': int(sum(1 for i in np.flatnonzero(prm) if 0 < labels[off[i]:off[i+1]].sum() < nsteps[i])),
       'uid_unique': bool(ans.uid.is_unique), 'groups_crossing_folds': int((ans.groupby('source_group').fold.nunique() > 1).sum()),
       'fold_matches_FOLDS_V2': bool(all(folds_v2[g] == f for g, f in zip(groups, fold))),
       'pb_target_in_range': bool(all(-1 <= target[i] < nsteps[i] for i in np.flatnonzero(pb))),
       'step_lengths_positive': bool((step_len > 0).all()), 'labels_binary_prm_steps': bool(set(np.unique(labels[np.repeat(prm, nsteps)])) <= {0, 1}), 'pb_step_label_sentinel': sorted(set(np.unique(labels[np.repeat(pb, nsteps)])).tolist()),
       'ct7_gate_open_pb': int((gate & pb).sum()), 'ct7_gate_open_prm': int((gate & prm).sum())}
all_finite = {m: bool(np.isfinite(Z[m]).all()) for m in methods}
cov['methods_with_nonfinite_scores'] = [m for m, ok in all_finite.items() if not ok]
cov['pb_predictions_missing'] = {m: int((pb & ~np.isfinite(ans[m + '__mode'].to_numpy())).sum()) for m in methods if (pb & ~np.isfinite(ans[m + '__mode'].to_numpy())).any()}
log['coverage'] = cov

# ---------------------------------------------------------------- 3. SEED_PARITY (actual seed vs protocol-described seed)
profiles = np.load(R / 'profiles_full.npy', mmap_mode='r')
seed_stored = Z['evidence__all__seed__equal']; seed30_stored = Z['evidence30__all__seed__equal']
raw = np.empty(int(off[-1])); zed = np.empty(int(off[-1]))
for i in range(n):
    a, b = off[i:i+2]; x = np.asarray(profiles[a:b, :, 0])
    raw[a:b] = softmax(x, axis=0).mean(1)
    sd = x.std(0); zz = (x - x.mean(0)) / np.where(sd > 1e-8, sd, 1.)
    zed[a:b] = softmax(zz, axis=0).mean(1)
def pb_macro(pred):
    return float(np.mean([(pred[m] == target[m]).mean() for c in sorted(set(cells[pb])) for m in [(cells == c) & (target >= 0)]]))
def peaks(s): return np.array([first_argmax(s[a:b]) for a, b in zip(off[:-1], off[1:])])
def tie_rate(s):
    return float(np.mean([(s[a:b] >= s[a:b].max() - TOL).sum() > 1 for a, b in zip(off[:-1], off[1:])]))
def prm_within(s): return float(np.nanmean([within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]]) for i in np.flatnonzero(prm)]))
p_raw, p_z, p_stored = peaks(raw), peaks(zed), ans['evidence__all__seed__equal__mode'].to_numpy()
parity = pd.DataFrame([
    {'seed': 'stored evidence__all__seed__equal (top5, actual run)', 'max_abs_score_diff_vs_stored': 0., 'pb_argmax_diff_vs_stored': 0,
     'pb_sla_macro8': pb_macro(p_stored), 'prm_within_auc': prm_within(seed_stored), 'top_tie_rate': tie_rate(seed_stored)},
    {'seed': 'replay: softmax over steps of raw top5 profile, mean over channels (no step-z)', 'max_abs_score_diff_vs_stored': float(np.abs(raw - seed_stored).max()),
     'pb_argmax_diff_vs_stored': int((p_raw[pb] != p_stored[pb]).sum()), 'pb_sla_macro8': pb_macro(p_raw), 'prm_within_auc': prm_within(raw), 'top_tie_rate': tie_rate(raw)},
    {'seed': 'protocol-described: step-z per channel, then softmax, mean over channels', 'max_abs_score_diff_vs_stored': float(np.abs(zed - seed_stored).max()),
     'pb_argmax_diff_vs_stored': int((p_z[pb] != p_stored[pb]).sum()), 'pb_sla_macro8': pb_macro(p_z), 'prm_within_auc': prm_within(zed), 'top_tie_rate': tie_rate(zed)},
])
parity.to_csv(OUT / 'SEED_PARITY.csv', index=False)
log['seed_parity'] = {'raw_replay_reproduces_stored': bool(np.abs(raw - seed_stored).max() < 1e-9),
                      'step_z_changes_pb_argmax': int((p_z[pb] != p_raw[pb]).sum()), 'deviation': 'two seeds = two method IDs: seed_raw (run) and seed_stepz (protocol text / Step429 equal)'}

# ---------------------------------------------------------------- 4. scoreboard replay (all methods)
rows = []; prm_auc_per_answer = {}
for m in methods:
    s = Z[m]; pred = ans[m + '__mode'].to_numpy(); ok = np.isfinite(pred)
    slas, f1s, q4, q8 = [], [], [], []
    for c in sorted(set(cells[pb])):
        take = (cells == c) & ok; e = take & (target >= 0); cl = take & (target < 0)
        sla = float((pred[e] == target[e]).mean()); ca = float((~gate[cl]).mean()); ea = float(((pred[e] == target[e]) & gate[e]).mean())
        slas.append(sla); f1s.append(2 * ca * ea / (ca + ea) if ca + ea else 0.); (q4 if c.endswith('q4') else q8).append(sla)
    valid_prm = prm & ~np.isin(np.arange(n), [i for i in np.flatnonzero(prm) if not np.isfinite(s[off[i]:off[i+1]]).all()])
    auc = np.full(n, np.nan)
    for i in np.flatnonzero(valid_prm): auc[i] = within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]])
    prm_auc_per_answer[m] = auc
    rep = {'method': m, 'pb_sla_macro8': float(np.mean(slas)), 'pb_f1_ct7_gate': float(np.mean(f1s)), 'pb_sla_q4': float(np.mean(q4)), 'pb_sla_q8': float(np.mean(q8)),
           'within_auc': float(np.nanmean(auc)), 'eligible': int(np.isfinite(auc).sum()), 'pb_covered': int((pb & ok).sum()), 'prm_covered': int(valid_prm.sum())}
    for k in ['pb_sla_macro8', 'pb_f1_ct7_gate', 'pb_sla_q4', 'pb_sla_q8', 'within_auc']: rep['err_' + k] = abs(rep[k] - summary.loc[m, k])
    rep['err_eligible'] = int(abs(rep['eligible'] - summary.loc[m, 'eligible']))
    rows.append(rep)
board = pd.DataFrame(rows).set_index('method')
# PRMScore replay from saved decision masks through the official evaluator
dec = np.load(R / 'PRM_OOF_DECISIONS.npz'); prmscore_json = read(R / 'PRMSCORE.json')
meta = {m['idx']: m for m in pickle.load(open(freeze_in['prm_metadata']['path'], 'rb')).values()}
ids = ans.id.to_numpy(); prm_idx = np.flatnonzero(prm)
def official(valid, idx):
    preds = [{'idx': ids[i], 'labels': valid[off[i]:off[i+1]].astype(int).tolist()} for i in idx]
    r = prmbench_evaluate(preds, [meta[ids[i]] for i in idx]); t = r['total']
    return .5 * (t['f1'] + t['negative_f1'])
board['prmscore_q80'] = np.nan; board['prmscore_inner'] = np.nan; board['err_prmscore_q80'] = np.nan; board['err_prmscore_inner'] = np.nan
for m in methods:
    if m + '__inner' not in dec.files: continue
    idx = prm_idx[np.isfinite(prm_auc_per_answer[m][prm_idx]) | True]  # official uses all valid prm answers
    idx = np.array([i for i in prm_idx if np.isfinite(Z[m][off[i]:off[i+1]]).all()])
    for kind, key in [('q80', 'quantile_0.8'), ('inner', 'inner_selected')]:
        v = official(dec[m + '__' + kind], idx); board.loc[m, 'prmscore_' + kind] = v
        board.loc[m, 'err_prmscore_' + kind] = abs(v - prmscore_json[m][key]['prmscore'])
board.to_csv(OUT / 'SCOREBOARD_REPLAY.csv')
errcols = [c for c in board.columns if c.startswith('err_')]
log['scoreboard_replay'] = {'methods': len(board), 'max_abs_error_per_metric': board[errcols].max().to_dict(),
                            'pb_partial_coverage': board[board.pb_covered < 6800].pb_covered.to_dict(),
                            'prmscore_methods_replayed': int(board.prmscore_inner.notna().sum())}

# ---------------------------------------------------------------- 5. decision changes vs teacher (PB) and paired per-answer AUROC (PRMB)
def pb_changes(cand, ref):
    pc, pr = ans[cand + '__mode'].to_numpy(), ans[ref + '__mode'].to_numpy()
    e = pb & (target >= 0) & np.isfinite(pc) & np.isfinite(pr)
    hc, hr = pc[e] == target[e], pr[e] == target[e]
    d = {'candidate': cand, 'reference': ref, 'erroneous_answers': int(e.sum()), 'agreement': float((pc[e] == pr[e]).mean()),
         'both_correct': int((hc & hr).sum()), 'wrong_to_correct': int((hc & ~hr).sum()), 'correct_to_wrong': int((~hc & hr).sum()), 'both_wrong': int((~hc & ~hr).sum()),
         'net_gain': int((hc & ~hr).sum() - (~hc & hr).sum()),
         'moved_earlier': int((pc[e] < pr[e]).sum()), 'moved_later': int((pc[e] > pr[e]).sum()),
         'cand_early_miss': float((pc[e] < target[e]).mean()), 'cand_late_miss': float((pc[e] > target[e]).mean()),
         'ref_early_miss': float((pr[e] < target[e]).mean()), 'ref_late_miss': float((pr[e] > target[e]).mean())}
    # long-chain slice (11+ steps)
    L = e & (nsteps >= 11); hcL, hrL = pc[L] == target[L], pr[L] == target[L]
    d.update({'long11_n': int(L.sum()), 'long11_wrong_to_correct': int((hcL & ~hrL).sum()), 'long11_correct_to_wrong': int((~hcL & hrL).sum())})
    return d
pairs = [('evidence__all__plain__equal', 'evidence__all__seed__equal'), ('evidence__all__position__equal', 'evidence__all__seed__equal'),
         ('evidence__all__position__equal', 'evidence__all__plain__equal'), ('evidence__all__plain2__equal', 'evidence__all__plain__equal'),
         ('evidence__all__position2__equal', 'evidence__all__position__equal'), ('evidence__all__ceiling__equal', 'evidence__all__plain__equal'),
         ('evidence__all__plain__continuous_lsml', 'evidence__all__plain__equal'), ('evidence__all__randomseed__equal', 'evidence__all__plain__equal'),
         ('evidence30__all__plain__equal', 'evidence30__all__seed__equal'), ('evidence30__all__position__equal', 'evidence30__all__plain__equal'),
         ('evidence__all__plain__equal', 'ct7'), ('evidence__all__position__equal', 'ct7'), ('evidence30__all__plain__equal', 'ct7'), ('token_lsml', 'ct7')]
pd.DataFrame([pb_changes(a, b) for a, b in pairs]).to_csv(OUT / 'PB_DECISION_CHANGES.csv', index=False)
def prm_pairs(cand, ref):
    ac, ar = prm_auc_per_answer[cand], prm_auc_per_answer[ref]; ok = np.isfinite(ac) & np.isfinite(ar); d = ac[ok] - ar[ok]
    # pairs corrected / destroyed, counted over error/clean step pairs within each answer
    fixed = broken = 0
    for i in np.flatnonzero(ok):
        a, b = off[i:i+2]; y = labels[a:b].astype(bool); sc, sr = Z[cand][a:b], Z[ref][a:b]
        wc = sc[y][:, None] > sc[~y][None, :]; wr = sr[y][:, None] > sr[~y][None, :]
        fixed += int((wc & ~wr).sum()); broken += int((~wc & wr).sum())
    return {'candidate': cand, 'reference': ref, 'answers': int(ok.sum()), 'mean_delta_auc': float(d.mean()), 'median_delta': float(np.median(d)),
            'answers_improved': int((d > 1e-12).sum()), 'answers_worsened': int((d < -1e-12).sum()), 'answers_unchanged': int((np.abs(d) <= 1e-12).sum()),
            'pairs_corrected': fixed, 'pairs_destroyed': broken, 'net_pairs': fixed - broken}
pd.DataFrame([prm_pairs(a, b) for a, b in pairs]).to_csv(OUT / 'PRM_PAIRED_CHANGES.csv', index=False)

# ---------------------------------------------------------------- 6. planned contrasts: pseudo vs random, position vs plain vs prior, iteration 2, learned fusion, vs CT7
con = pd.read_csv(R / 'PAIRED_CONTRASTS.csv', encoding='utf-8-sig')
unc = read(R / 'UNCERTAINTY.json')
questions = {
    'Q1 labels contribute? (primary - random-seed null)': [('evidence__all__plain__equal', 'evidence__all__randomseed__equal'), ('evidence__all__position__equal', 'evidence__all__randomseed_position__equal'), ('evidence30__all__plain__equal', 'evidence30__all__randomseed__equal')],
    'Q1b evidence vs its own seed': [('evidence__all__plain__equal', 'evidence__all__seed__equal'), ('evidence30__all__plain__equal', 'evidence30__all__seed__equal')],
    'Q2 position conditioning? (position - plain; prior-only - plain)': [('evidence__all__position__equal', 'evidence__all__plain__equal'), ('evidence30__all__position__equal', 'evidence30__all__plain__equal'), ('evidence__all__prioronly__equal', 'evidence__all__plain__equal')],
    'Q3 iteration 2?': [('evidence__all__plain2__equal', 'evidence__all__plain__equal'), ('evidence__all__position2__equal', 'evidence__all__position__equal')],
    'Q4 learned fusion?': [('evidence__all__plain__continuous_lsml', 'evidence__all__plain__equal'), ('evidence__all__plain__spectral', 'evidence__all__plain__equal'), ('evidence__all__position__continuous_lsml', 'evidence__all__position__equal')],
    'Q5 vs CT7': [('evidence__all__plain__equal', 'ct7'), ('evidence__all__position__equal', 'ct7'), ('evidence__all__position2__equal', 'ct7'), ('evidence30__all__plain__equal', 'ct7')],
    'Q6 label ceiling - primary': [('evidence__all__ceiling__equal', 'evidence__all__plain__equal'), ('evidence__all__ceiling_position__equal', 'evidence__all__position__equal')],
}
qrows = []
for q, prs in questions.items():
    for a, b in prs:
        for ep in ['pb_sla', 'prm_within_auc', 'within_auc', 'prm_within']:
            hit = con[(con.a == a) & (con.b == b) & (con.endpoint == ep)]
            for _, r in hit.iterrows():
                lo, hi = json.loads(r.ci95)
                qrows.append({'question': q, 'a': a, 'b': b, 'endpoint': r.endpoint, 'contrast': r.contrast, 'delta': r.delta, 'ci_lo': lo, 'ci_hi': hi, 'p_bootstrap': r.p_bootstrap, 'p_holm': r.p_holm})
qdf = pd.DataFrame(qrows)
if qdf.empty:  # column names differ; fall back to raw structure
    qdf = con.copy()
qdf.to_csv(OUT / 'PLANNED_CONTRASTS_BY_QUESTION.csv', index=False)
log['contrast_columns'] = list(con.columns); log['contrast_endpoints'] = sorted(con.endpoint.unique().tolist())
log['multiplicity'] = {'family_size': len(con), 'draws': unc['draws'], 'seed': unc['seed'], 'unit': unc['unit'], 'source_groups': unc['source_groups'],
                       'minimum_raw_p_resolution': unc['minimum_raw_p'], 'min_p_holm': float(con.p_holm.min()), 'contrasts_with_p_holm_below_0.05': int((con.p_holm < .05).sum()),
                       'note': 'Holm over all 312 planned primary-endpoint contrasts jointly; tail resolution 1e-4 so p_holm floor = 312e-4 = 0.0312'}

# ---------------------------------------------------------------- 7. section 13.2 competition / position / depth breakdowns
def analyse(m):
    s = Z[m]; pred = ans[m + '__mode'].to_numpy(); out = []
    for i in np.flatnonzero(pb & (target >= 0)):
        if not np.isfinite(pred[i]): continue
        a, b = off[i:i+2]; v = s[a:b]; t = int(target[i]); S = b - a
        order = np.argsort(-v, kind='stable'); rank = int(np.flatnonzero(order == t)[0]) + 1   # stable: earlier index wins ties -> earliest-tie rank
        others = np.delete(v, t); margin = float(v[t] - others.max()) if S > 1 else np.nan
        A = int((v[t] > others).sum() + ((v[t] == others) & (np.delete(np.arange(S), t) > t)).sum())   # steps the truth beats (ties: earlier wins)
        comp = math.comb(A, 3) / math.comb(S - 1, 3) if S >= 4 else np.nan
        out.append({'method': m, 'answer': i, 'cell': cells[i], 'S': S, 'depth_bin': '1' if S == 1 else '2-5' if S <= 5 else '6-10' if S <= 10 else '11+',
                    'rel_pos': t / max(S - 1, 1), 'rel_bin': ['[0,.2)', '[.2,.4)', '[.4,.6)', '[.6,.8)', '[.8,1]'][min(int(5 * t / max(S - 1, 1)), 4)] if S > 1 else '[0,.2)',
                    'err_len': int(step_len[a + t]), 'len_bin': '1-16' if step_len[a + t] <= 16 else '17-32' if step_len[a + t] <= 32 else '33-64' if step_len[a + t] <= 64 else '65+',
                    'hit1': rank == 1, 'hit2': rank <= 2, 'hit3': rank <= 3, 'hit5': rank <= 5, 'rank': rank, 'margin': margin,
                    'signed': int(pred[i]) - t, 'tie_at_top': bool((v >= v.max() - TOL).sum() > 1), 'competition_oracle': comp, 'top1_S4': (rank == 1) if S >= 4 else np.nan})
    return pd.DataFrame(out)
per = pd.concat([analyse(m) for m in KEY], ignore_index=True)
per.to_csv(OUT / 'PB_PER_ANSWER_COMPETITION.csv', index=False)
def summarise(df, by):
    g = df.groupby(['method', by])
    return g.agg(n=('hit1', 'size'), sla=('hit1', 'mean'), hit3=('hit3', 'mean'), hit5=('hit5', 'mean'), mean_rank=('rank', 'mean'), median_margin=('margin', 'median'),
                 early=('signed', lambda x: float((x < 0).mean())), late=('signed', lambda x: float((x > 0).mean())), tie_at_top=('tie_at_top', 'mean')).reset_index()
summarise(per, 'depth_bin').to_csv(OUT / 'PB_BY_DEPTH.csv', index=False)
summarise(per, 'rel_bin').to_csv(OUT / 'PB_BY_RELATIVE_POSITION.csv', index=False)
summarise(per, 'len_bin').to_csv(OUT / 'PB_BY_ERROR_STEP_LENGTH.csv', index=False)
comp = per[per.S >= 4].groupby('method').agg(n_S4=('S', 'size'), exact_top1_S4=('top1_S4', 'mean'), competition_oracle_3=('competition_oracle', 'mean'),
                                              hit1=('hit1', 'mean'), hit3=('hit3', 'mean'), hit5=('hit5', 'mean'), mean_rank=('rank', 'mean'), median_margin=('margin', 'median')).reset_index()
comp.to_csv(OUT / 'PB_COMPETITION_ORACLE.csv', index=False)

# ---------------------------------------------------------------- 8. PRMB per-answer AUROC by category (multi-error / single-error), by depth
cat = []
for i in np.flatnonzero(prm):
    a, b = off[i:i+2]; y = labels[a:b]; ne = int(y.sum())
    cat.append({'answer': i, 'S': b - a, 'n_err': ne, 'kind': 'clean' if ne == 0 else 'all_error' if ne == b - a else 'single' if ne == 1 else 'multi',
                'classification': meta[ids[i]]['classification'], **{m: prm_auc_per_answer[m][i] for m in KEY}})
cat = pd.DataFrame(cat)
cat.groupby('kind')[KEY].agg(['mean', 'count']).to_csv(OUT / 'PRM_WITHIN_AUC_BY_KIND.csv')
cat.groupby('classification')[KEY].mean().to_csv(OUT / 'PRM_WITHIN_AUC_BY_CLASSIFICATION.csv')
cat['depth_bin'] = pd.cut(cat.S, [0, 1, 5, 10, 10 ** 6], labels=['1', '2-5', '6-10', '11+'])
cat.groupby('depth_bin', observed=True)[KEY].mean().to_csv(OUT / 'PRM_WITHIN_AUC_BY_DEPTH.csv')

log['seconds'] = time.perf_counter() - t0
dump(OUT / 'S0_LOG.json', log)
print(json.dumps(log, indent=1, default=str)[:6000])
