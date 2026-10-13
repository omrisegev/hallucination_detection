#!/usr/bin/env python
"""Frozen cross-PRM transfer evaluation. Labels enter metrics only, never fitting.

All production arms are withheld unless G1-G3 pass on the complete 6,969 rows.
--replay-v1 reconstructs historical S3/EQ2 from saved arrays for CPU test T4;
it is a regression test, not a new-model result or a gate bypass.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spectral_utils.prmbench import prmbench_evaluate
from spectral_utils.prm_adapters import ADAPTERS

S3 = ('energy_level', 'hist_entropy_cusum_abs_series', 'realized_drv')
S6 = ('bb:energy_level', 'bb:hist_entropy_cusum_abs_series', 'bb:realized_drv',
      'q3:energy_level', 'q3:hist_energy_cusum_abs_series', 'bb:hist_topk_varentropy_series')
DRAWS, SEED, N_ROWS = 2000, 20261013, 6969
CELL = 'prmbench_qwen3_8b'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def standardizer():
    """Execute the frozen Monday dependency's exact function and EPS assignment.

    Avoid importing the fitter (and pandas) for saved-array metric regression.
    No function body is rewritten and no substitute z-score is introduced.
    """
    p = ROOT / 'results/monday_fusion_analysis_v1/code/spectral_utils/lsml_gate_locator_research.py'
    tree = ast.parse(p.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'answer_standardize'
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EPS' for t in n.targets)]
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), namespace)
    return namespace['answer_standardize']


answer_standardize = standardizer()


def az(v, offsets):
    return answer_standardize(np.asarray(v, float)[:, None], offsets)[:, 0]


def risk(rewards):
    v = np.asarray(rewards, float)
    if not np.isfinite(v).all() or np.any((v < 0) | (v > 1)):
        raise ValueError('invalid validity probabilities')
    p = np.clip(1-v, 1e-6, 1-1e-6)
    return np.log(p/(1-p))


def frozen_arms(prm, n16, own_channels, q3_channels, offsets):
    """Inputs are fixed +1 answer-z channels, built via Population.cell_bank."""
    zp = az(prm, offsets)
    return {'PRM_native': prm, 'PRM_Q80': prm, 'N16_own': n16,
            'EQ2_own': .5*zp + .5*az(n16, offsets),
            'S3_own': .5*zp + .5*np.column_stack([own_channels[c] for c in S3]).mean(1),
            'S3_q3': .5*zp + .5*np.column_stack([q3_channels[c] for c in S3]).mean(1),
            'S6_own': .5*zp + .5*np.column_stack([(own_channels if p == 'bb' else q3_channels)[c]
                                                for p, c in (x.split(':', 1) for x in S6)]).mean(1)}


def auc(y, score):
    n1 = int(y.sum()); n0 = len(y)-n1
    return np.nan if not n1 or not n0 else (rankdata(score)[y].sum()-n1*(n1+1)/2)/(n1*n0)


def measure(scores, flags, labels, offsets, ids, metadata):
    values = np.array([auc(labels[a:b], scores[a:b]) for a, b in zip(offsets[:-1], offsets[1:])])
    predictions = [{'idx': i, 'labels': (~flags[a:b]).astype(int).tolist()}
                   for i, a, b in zip(ids, offsets[:-1], offsets[1:])]
    official = prmbench_evaluate(predictions, metadata)
    t = official['total']; bc = official['by_classification']
    row = {'within_auc': float(np.nanmean(values)), 'n_auc': int(np.isfinite(values).sum()),
           'prmscore': 50*(t['f1']+t['negative_f1']), 'n_answers': len(ids)}
    for c in sorted({m['classification'] for m in metadata}-{'correct'}):
        row['prmscore__'+c] = 50*(bc['f1'][c]+bc['negative_f1'][c])
    return row, values


def flags_for(score, offsets):
    z = az(score, offsets)
    return z > np.quantile(z, .8)


def bootstrap(arms, flags, labels, offsets, ids, metadata, groups):
    """Same grouped sufficient statistics and shared multinomial draws as small_bank_bootstrap.py."""
    unique, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(SEED)
    W = np.vstack([np.ones(len(unique)), rng.multinomial(len(unique),
                   np.full(len(unique), 1/len(unique)), size=DRAWS)])
    official_error = np.zeros(offsets[-1], bool)
    for j, m in enumerate(metadata):
        for e in m['error_steps']:
            if 1 <= e <= offsets[j+1]-offsets[j]: official_error[offsets[j]+e-1] = True
    B = {}
    for name, score in arms.items():
        row, au = measure(score, flags[name], labels, offsets, ids, metadata)
        c4 = np.zeros((len(ids), 4))
        for j, m in enumerate(metadata):
            if m['classification'] == 'correct': continue
            a, b = offsets[j:j+2]; y, f = official_error[a:b], flags[name][a:b]
            c4[j] = [(~y & ~f).sum(), (y & ~f).sum(), (y & f).sum(), (~y & f).sum()]
        G = np.zeros((len(unique), 6))
        np.add.at(G, inv, np.column_stack([np.where(np.isfinite(au), au, 0), np.isfinite(au), c4]))
        totals = W @ G; tp, fp, tn, fn = totals[:, 2:].T
        B[name] = (totals[:, 0]/totals[:, 1],
                   50*(tp/(tp+.5*(fp+fn))+tn/(tn+.5*(fp+fn))))
        if not all(np.isfinite(v).all() for v in B[name]): raise ValueError('nonfinite source bootstrap '+name)
        if abs(B[name][1][0]-row['prmscore']) > 1e-6: raise ValueError('bootstrap point parity '+name)
    rows = []
    for a, b in [('S3_own', 'PRM_native'), ('S3_own', 'PRM_Q80'),
                 ('S3_own', 'N16_own'), ('EQ2_own', 'PRM_native')]:
        for j, metric in enumerate(('within_auc', 'prmscore')):
            delta = B[a][j]-B[b][j]
            rows.append({'contrast': a+' - '+b, 'metric': metric, 'delta': delta[0],
                         'ci_low': np.quantile(delta[1:], .025), 'ci_high': np.quantile(delta[1:], .975),
                         'draws': DRAWS, 'seed': SEED, 'n_source_groups': len(unique)})
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)


def metadata_from(path):
    with open(path, 'rb') as f: rows = pickle.load(f)
    result = {}
    for r in rows.values():
        key = str(r['idx'])
        if key in result: raise ValueError('duplicate telemetry idx '+key)
        result[key] = r
    return result


def check_gates(native_score, manifest, parity, n_rows, expected):
    g1 = expected is not None and abs(native_score-expected) <= 1.0
    median = manifest.get('token_surprisal_median')
    mismatch = manifest.get('n_rows_step_text_mismatch')
    g2 = median is not None and median < 1 and mismatch == 0 and manifest.get('n_step_text_mismatch', 0) == 0
    g3 = parity.get('max_abs_error') == 0.0 and parity.get('digit_active_mismatch') == 0
    complete = (n_rows == N_ROWS and manifest.get('n_rows') == N_ROWS and
                manifest.get('n_failed_rows', 0) == 0 and manifest.get('status', 'COMPLETE') == 'COMPLETE')
    return dict(G1={'pass': bool(g1), 'native_prmscore': native_score, 'leaderboard': expected, 'tolerance': 1.0},
                G2={'pass': bool(g2), 'token_surprisal_median': median, 'n_rows_step_text_mismatch': mismatch},
                G3={'pass': bool(g3), **parity}, full_population={'pass': complete, 'n_rows': n_rows},
                publish_fused=bool(g1 and g2 and g3 and complete))


def replay_v1(prm_path):
    """Reconstruct, rather than read back, both T4 metrics on all 6,969 rows."""
    directory = ROOT / 'results/prm_boosting_v1'
    with np.load(directory/'CHANNEL_CONDCORR_prm_backbone.npz', allow_pickle=True) as z:
        own = {str(n): z['S'][:, k].astype(float) for k, n in enumerate(z['names'])}
        labels, offsets, ids = z['y'].astype(bool), z['offsets'], z['idx'].astype(str)
    with np.load(directory/'CHANNEL_CONDCORR_qwen3_8b_mask1.npz', allow_pickle=True) as z:
        if not np.array_equal(z['idx'].astype(str), ids) or not np.array_equal(z['offsets'], offsets):
            raise ValueError('historical Q3 order')
        q3 = {str(n): z['S'][:, k].astype(float) for k, n in enumerate(z['names'])}
    with np.load(directory/'SCORES_prm_backbone.npz', allow_pickle=True) as z:
        if not np.array_equal(z['idx'].astype(str), ids) or not np.array_equal(z['offsets'], offsets):
            raise ValueError('historical score order')
        n16 = z['N16__MEAN_MOM_ga']; prm64 = z['prm_risk']
    # Stage-2 small_bank.py used the stored float32 channel atlas (including PRM).
    arms = frozen_arms(own['prm_risk'], n16, own, q3, offsets)
    arms['EQ2_own'] = .5*az(prm64, offsets)+.5*az(n16, offsets)
    meta = metadata_from(prm_path); metadata = [dict(meta[i], idx=i) for i in ids]
    out = {}
    for name, csv_name, row_name in [('S3_own', 'SMALL_BANK.csv', 'greedy k=3, all-data selection (in-sample)'),
                                    ('EQ2_own', 'COMPOSITIONS.csv', 'EQ2(PRM,BB)')]:
        row, _ = measure(arms[name], flags_for(arms[name], offsets), labels, offsets, ids, metadata)
        with open(directory/csv_name, newline='', encoding='utf-8-sig') as f:
            expected = next(r for r in csv.DictReader(f) if r['arm'] == row_name)
        for metric in ('within_auc', 'prmscore'):
            if abs(row[metric]-float(expected[metric])) > 1e-6:
                raise ValueError(f'T4 {name} {metric}: {row[metric]} != {expected[metric]}')
        out[name] = row
    return {'status': 'PASS', 'tag': 'qwen_prm_7b', 'n_checked': len(ids), 'metrics': out,
            'prm_metadata_sha256': sha(prm_path),
            'inputs': {p.name: sha(p) for p in directory.glob('*.npz') if p.name.startswith(('CHANNEL_CONDCORR','SCORES_prm_backbone'))}}


def run(args):
    if getattr(ADAPTERS[args.adapter], 'blocked_reason', None):
        raise SystemExit(ADAPTERS[args.adapter].blocked_reason)
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    for prefix, suffix in [('GATES', 'json'), ('RESULTS', 'csv'), ('BOOTSTRAP', 'csv'), ('SCORES', 'npz')]:
        if (out/f'{prefix}_{args.tag}.{suffix}').exists():
            raise ValueError('refusing to overwrite an existing run; choose a fresh --out')
    telemetry = metadata_from(args.prm_pkl)
    manifest = json.loads(args.manifest.read_text()); parity = json.loads(args.q3_parity.read_text())
    parity_manifest_path = args.q3_parity.parent/'MANIFEST.json'
    parity_manifest = json.loads(parity_manifest_path.read_text())
    q3_hash = sha(args.q3_features)
    if q3_hash not in (parity_manifest['base_sha256'], parity_manifest['features_sha256']):
        raise ValueError('G3 parity artifact is not bound to the supplied Qwen3 feature bank')
    with open(args.answers, newline='', encoding='utf-8-sig') as f: A = list(csv.DictReader(f))
    F = dict(np.load(args.features)); Q = dict(np.load(args.q3_features))
    if list(map(str, F['uids'])) != [r['uid'] for r in A]: raise ValueError('feature/roster UID order')
    if list(F['row_ids'].astype(str)) != [r['id'] for r in A] or list(F['cells'].astype(str)) != [r['cell'] for r in A]:
        raise ValueError('feature/roster ID or cell order')
    for key in ('uids', 'offsets', 'row_ids', 'cells'):
        if not np.array_equal(F[key], Q[key]): raise ValueError('own/Q3 order '+key)
    offsets_all = F['offsets'].astype(np.int64)
    ans = np.flatnonzero(F['cells'].astype(str) == CELL)
    ids = F['row_ids'][ans].astype(str); offsets = np.r_[0, np.cumsum(np.diff(offsets_all)[ans])]
    if set(telemetry) != set(ids): raise ValueError('telemetry must contain exactly the registered PRMBench rows')
    meta = [dict(telemetry[i], idx=i) for i in ids]
    if any('failure' in r for r in meta): raise ValueError('counted inference failures block full-population evaluation')
    if manifest.get('adapter') != args.adapter: raise ValueError('manifest adapter mismatch')
    if manifest.get('model') != ADAPTERS[args.adapter].model_id: raise ValueError('manifest model does not match adapter')
    for j, r in enumerate(meta):
        n_steps = offsets[j+1]-offsets[j]
        if any(len(r[k]) != n_steps for k in ('rewards', 'native_labels', 'steps')):
            raise ValueError('reward/native/text step count '+ids[j])
        if not set(r['native_labels']) <= {0, 1}: raise ValueError('invalid native labels '+ids[j])
    prm = np.concatenate([risk(r['rewards']) for r in meta])
    native = np.concatenate([np.asarray(r['native_labels'], int) == 0 for r in meta])
    with np.load(args.labels) as z:
        if not np.array_equal(z['offsets'], offsets_all): raise ValueError('v3 labels must be in full roster order')
        lab_all = z['labels'].astype(bool)
        if len(lab_all) != offsets_all[-1]: raise ValueError('v3 label count')
    selected_rows = np.concatenate([np.arange(offsets_all[i], offsets_all[i+1]) for i in ans])
    labels = lab_all[selected_rows]
    base, _ = measure(prm, native, labels, offsets, ids, meta)
    gates = check_gates(base['prmscore'], manifest, parity, len(ids), ADAPTERS[args.adapter].leaderboard)
    sources = dict(features=args.features, q3_features=args.q3_features, telemetry=args.prm_pkl,
                   manifest=args.manifest, q3_parity=args.q3_parity, q3_parity_manifest=parity_manifest_path,
                   answers=args.answers, labels=args.labels)
    gates['sources'] = {k: {'path': str(p), 'sha256': sha(p)} for k, p in sources.items()}
    gates['frozen_code_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in [
        ROOT/'results/prm_boosting_v1/build_bank.py', ROOT/'results/monday_external_panel_v1/score.py']}
    gates['claim'] = 'Cross-model development test; supervised PRM, Qwen-7B label-selected channels, label-free fusion.'
    gate_path = out/f'GATES_{args.tag}.json'; gate_path.write_text(json.dumps(gates, indent=2))
    write_csv(out/f'RESULTS_{args.tag}.csv', [{'arm': 'PRM_native', **base}])
    if not gates['publish_fused']: raise SystemExit('G1/G2/G3/full-population gate failed; fused arms withheld')
    spec = importlib.util.spec_from_file_location('prm_fuse', ROOT/'results/prm_boosting_v1/fuse.py')
    FU = importlib.util.module_from_spec(spec); spec.loader.exec_module(FU); ext = FU.ext
    # Use the unchanged Monday population on the full original roster, preserving random tie keys.
    ext.BN = dict(ext.BN, N16=FU.N16, TRANSFER=list(S3)+['hist_topk_varentropy_series'],
                  Q3_TRANSFER=list(S3)+['hist_energy_cusum_abs_series'])
    pop = ext.Population(F, ['N16', 'TRANSFER']); bank = pop.cell_bank('N16', ans)
    sc, _, diag = ext.fit_cell(pop, bank, 0)
    own_bank = pop.cell_bank('TRANSFER', ans)
    qpop = ext.Population(Q, ['Q3_TRANSFER']); qb = qpop.cell_bank('Q3_TRANSFER', ans)
    own = {n: own_bank['V'][:, k] for k, n in enumerate(own_bank['names'])}
    q3 = {n: qb['V'][:, k] for k, n in enumerate(qb['names'])}
    arms = frozen_arms(prm, sc['MEAN_MOM_ga'], own, q3, offsets)
    flags = {n: native if n == 'PRM_native' else flags_for(s, offsets) for n, s in arms.items()}
    rows = [{'arm': n, **measure(s, flags[n], labels, offsets, ids, meta)[0]} for n, s in arms.items()]
    write_csv(out/f'RESULTS_{args.tag}.csv', rows)
    groups = np.array([A[i]['source_group'] for i in ans])
    write_csv(out/f'BOOTSTRAP_{args.tag}.csv', bootstrap(arms, flags, labels, offsets, ids, meta, groups))
    np.savez_compressed(out/f'SCORES_{args.tag}.npz', idx=ids, offsets=offsets,
                        **{'score__'+n: s for n, s in arms.items()},
                        **{'flag__'+n: f for n, f in flags.items()})
    gates['fit_diagnostics'] = diag; gate_path.write_text(json.dumps(gates, indent=2, default=str))
    summary = out/'SUMMARY.csv'; old = []
    if summary.exists():
        with open(summary, newline='') as f: old = [r for r in csv.DictReader(f) if r['model'] != args.tag]
    row = {'model': args.tag}
    for r in rows:
        for metric in ('within_auc', 'prmscore'): row[r['arm']+'__'+metric] = r[metric]
    write_csv(summary, old+[row])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay-v1', action='store_true')
    ap.add_argument('--prm-pkl', type=Path, required=True)
    ap.add_argument('--adapter', choices=list(ADAPTERS), default='qwen_prm')
    ap.add_argument('--tag', default='qwen_prm_7b')
    ap.add_argument('--out', type=Path, required=True)
    for name in ('features', 'q3-features', 'manifest', 'q3-parity', 'answers', 'labels'):
        ap.add_argument('--'+name, type=Path)
    args = ap.parse_args()
    if args.replay_v1:
        args.out.mkdir(parents=True, exist_ok=True)
        report = replay_v1(args.prm_pkl)
        (args.out/'REPLAY_V1.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))
        return
    for name in ('features', 'q3_features', 'manifest', 'q3_parity', 'answers', 'labels'):
        if getattr(args, name) is None: ap.error('--'+name.replace('_','-')+' is required')
    run(args)


if __name__ == '__main__':
    main()
