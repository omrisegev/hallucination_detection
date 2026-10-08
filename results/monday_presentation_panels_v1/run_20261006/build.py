"""Presentation panels A (algorithm vs simple fusion) and B (detector vs competitors).

Development, descriptive, post hoc (see PROTOCOL.json exposure_disclosure). Reads
frozen scores only; writes only into this run directory.
"""
from __future__ import annotations

import ast, hashlib, importlib.util, json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT = ROOT / 'results/monday_presentation_panels_v1/run_20261006'
COMP = ROOT / 'results/monday_fusion_competitor_extension_v1/run_20261006'
PANEL = ROOT / 'results/monday_candidate_panel_v1/run_20261004'
TCI = ROOT / 'results/mom_filter_within_diagnostic_20261004/tci_wide_20261005'
INPUT_RUN = ROOT / 'results/estimator_input_v1/run_20261004_resume'
SE = ROOT / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
FL = ROOT / 'dataset_cache/four_localization'
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
HIGHLIGHT = 'B16'
PRM = 'prmbench_qwen3_8b'
METHODS = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
EXTRA = ['HEM_MOM_ga']
CONTROLS = ['ALL_EQ', 'DS_FILTER_EQ', 'MEAN_EQ', 'HEM_EQ']
RULES = ['ANSWER_Z_Q80', 'R2_COUNT', 'ENTROPY_GATE_Q80']
SUBSETS = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
DRAWS, SEED = 5000, 20261006


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def dump(p, v):
    p = Path(p); tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n', encoding='utf8')
    tmp.replace(p)


def inputs():
    files = {'competitor_metrics': COMP / 'METRICS.csv', 'competitor_run': COMP / 'run.py',
             'panel_metrics': PANEL / 'CONTENT_METRICS.csv', 'oof_answers': SE / 'OOF_ANSWERS.csv',
             'oof_step_scores': SE / 'OOF_STEP_SCORES.npz', 'input_freeze': SE / 'INPUT_FREEZE.json',
             'monday_candidate_panel_code': ROOT / 'spectral_utils/monday_candidate_panel.py'}
    for p in sorted((PANEL / 'candidate_scores').glob('*.npz')): files['panel_score/' + p.name] = p
    for p in sorted((TCI / 'scores').glob('*.npz')): files['tci_score/' + p.name] = p
    for p in sorted((INPUT_RUN / 'chunks').glob('*.diagnostics.json')): files['diagnostics/' + p.name] = p
    for s in SUBSETS:
        files[f'pb_prm/{s}'] = FL / f'pb_prm_qwen25math7b_full/pb_prm_{s}.pkl'
        files[f'pb_critic/{s}'] = FL / f'pb_critic_qwen72b_full/pb_critic_{s}.pkl'
    fr = json.loads((SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
    for k in ('prm_metadata', 'mindgap', 'tokens'): files[k] = Path(fr[k]['path'])
    return files


def freeze_inputs():
    path = OUT / 'INPUT_MANIFEST.json'
    files = inputs()
    if path.exists():
        man = json.loads(path.read_text(encoding='utf8'))
        for k, v in man.items():
            if sha(v['path']) != v['sha256']: raise RuntimeError('Input changed: ' + k)
        return man
    man = {k: {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for k, p in files.items()}
    dump(path, man); return man


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def topk(s, k):
    f = np.zeros(len(s), bool); f[np.lexsort((np.arange(len(s)), -s))[:k]] = True; return f


def conf(f, y):
    v, g = ~f, ~y
    return [(v & g).sum(), (v & ~g).sum(), (~v & ~g).sum(), (~v & g).sum()]


def main():
    t0 = time.time()
    if (OUT / 'PER_ANSWER.npz').exists(): raise RuntimeError('Outputs exist; refusing to overwrite')
    prot = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))
    assert prot['status'] == 'FROZEN_BEFORE_FINAL_TABLES'
    man = freeze_inputs()
    spec = importlib.util.spec_from_file_location('codex_run', COMP / 'run.py'); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    pop, data, evidence, rules, *_ = run.load_runtime()
    spec = importlib.util.spec_from_file_location('cpanel', ROOT / 'spectral_utils/monday_candidate_panel.py'); cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
    fr = json.loads((SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
    meta = {m['idx']: m for m in pickle.load(open(fr['prm_metadata']['path'], 'rb')).values()}
    with np.load(fr['mindgap']['path']) as z: MTG = z['step_m5'].astype(float)
    with np.load(fr['tokens']['path']) as z: spans = z['step_spans']
    TLEN = (spans[:, 1] - spans[:, 0]).astype(float)
    OFF, LAB, NS = pop.off, pop.labels, pop.ns
    N = pop.n; cells = pop.cells; ids = pop.ids; target = pop.target
    assert np.array_equal(MTG, np.asarray(data.mtg, float))
    print('loaded', round(time.time() - t0, 1), flush=True)

    # ---------------- per-answer containers
    R = {}  # scorer -> dict of arrays length N

    def new():
        return {'auc': np.full(N, np.nan), 'top1err': np.full(N, np.nan), 'c1': np.zeros((N, 4), int), 'c2': np.zeros((N, 4), int),
                'cown': np.zeros((N, 4), int), 'argmax': np.full(N, -9, int), 'first_R2': np.full(N, -9, int),
                'first_ENT': np.full(N, -9, int), 'first_AZ': np.full(N, -9, int), 'own_flagged': np.zeros(N, int)}

    def fill(store, i, s, f_own=None, firsts=None):
        y = LAB[OFF[i]:OFF[i + 1]]; s = np.asarray(s, float)
        # frozen metric convention (monday_fusion_metrics.evaluate): first step within 8 eps of the maximum
        store['argmax'][i] = int(np.flatnonzero(s >= s.max() - 8 * np.finfo(float).eps)[0])
        if not pop.pb[i]:
            store['auc'][i] = auc(y, s)
            if y.any(): store['top1err'][i] = float(y[int(np.argmax(s))])
            if pop.noncontrol[i]:
                store['c1'][i] = conf(topk(s, 1), y); store['c2'][i] = conf(topk(s, 2), y)
                if f_own is not None: store['cown'][i] = conf(f_own, y)
            if f_own is not None: store['own_flagged'][i] = int(f_own.any())
        if firsts is not None:
            for k, v in firsts.items(): store['first_' + k][i] = v

    # bank-independent scorers
    for name in ['PRM', 'MTG', 'STEP_INDEX', 'STEP_LENGTH']: R[name] = new()
    for i in range(N):
        sl = slice(OFF[i], OFF[i + 1])
        fill(R['MTG'], i, MTG[sl]); fill(R['STEP_INDEX'], i, np.arange(NS[i]) + 1e-9 * 0); fill(R['STEP_LENGTH'], i, TLEN[sl])
        if not pop.pb[i]:
            m = meta[ids[i]]
            fill(R['PRM'], i, -np.asarray(m['rewards'], float), f_own=np.asarray(m['labels']) == 0)
    # STEP_INDEX argmax should be the last step: np.argmax of arange gives last index -- correct orientation
    # ProcessBench PRM / critic
    R['PRM']['pb_native_first'] = np.full(N, -9, int); R['CRITIC'] = {'pb_native_first': np.full(N, -9, int)}
    crit_missing = 0
    by_id = {}
    for s in SUBSETS:
        dp = {x['id']: x for x in pickle.load(open(FL / f'pb_prm_qwen25math7b_full/pb_prm_{s}.pkl', 'rb')).values()}
        dc = {x['id']: x for x in pickle.load(open(FL / f'pb_critic_qwen72b_full/pb_critic_{s}.pkl', 'rb')).values()}
        by_id[s] = (dp, dc)
    for i in np.flatnonzero(pop.pb):
        s = cells[i].split('_')[1]; key = ids[i].split('::')[1]; dp, dc = by_id[s]; x = dp[key]
        rw = np.asarray(x['rewards'] if isinstance(x['rewards'], list) else ast.literal_eval(x['rewards']), float)
        st = x['steps'] if isinstance(x['steps'], list) else ast.literal_eval(x['steps'])
        if len(rw) != NS[i] or len(st) != NS[i] or int(x['label']) != target[i]: raise RuntimeError('PB PRM cache misaligned ' + ids[i])
        R['PRM']['argmax'][i] = int(np.flatnonzero(-rw >= (-rw).max() - 8 * np.finfo(float).eps)[0]); R['PRM']['pb_native_first'][i] = int(x['prediction'])
        cpred = dc[key]['prediction']
        if cpred is None: crit_missing += 1
        R['CRITIC']['pb_native_first'][i] = -9 if cpred is None else int(cpred)
    print('competitors done', round(time.time() - t0, 1), 'critic missing predictions', crit_missing, flush=True)

    # bank-dependent scorers
    M = pd.read_csv(COMP / 'METRICS.csv'); C = pd.read_csv(PANEL / 'CONTENT_METRICS.csv', low_memory=False)
    checks = {'score_replay_max': 0., 'metric_replay_max': 0., 'control_replay_max': 0., 'flag_replays': 0}
    for bk in BANKS:
        for name in METHODS + EXTRA + CONTROLS: R[(bk, name)] = new()
        for cell in sorted(set(cells)):
            ans = np.flatnonzero(cells == cell); rows = pop.rows(cells == cell); off = np.r_[0, np.cumsum(NS[ans])]
            rec_ga = json.loads((INPUT_RUN / 'chunks' / f'{cell}__{bk}__groups_answer.diagnostics.json').read_text(encoding='utf8'))
            rec_cc = json.loads((INPUT_RUN / 'chunks' / f'{cell}__{bk}__channels_cell.diagnostics.json').read_text(encoding='utf8'))
            bank = data.bank(bk, cell, rec_ga)
            assert np.array_equal(bank['answers'], ans) and np.array_equal(bank['rows'], rows)
            sga = cp.reconstruct(bank, rec_ga); scc = cp.reconstruct(bank, rec_cc)
            for c in CONTROLS:
                checks['score_replay_max'] = max(checks['score_replay_max'], float(np.max(np.abs(sga[c] - scc[c]))))
            scores = {'HEM_MOM_cc': scc['HEM_MOM'], 'MEAN_MOM_cc': scc['MEAN_MOM'], 'MEAN_MOM_ga': sga['MEAN_MOM'], 'HEM_MOM_ga': sga['HEM_MOM'],
                      **{c: sga[c] for c in CONTROLS}}
            stored_flags = {}
            for variant, sc in (('groups_answer', sga), ('channels_cell', scc)):
                with np.load(PANEL / 'candidate_scores' / f'{cell}__{bk}__{variant}.npz') as z:
                    for r in ('MEAN_MOM', 'HEM_MOM'):
                        checks['score_replay_max'] = max(checks['score_replay_max'], float(np.max(np.abs(z[f'{r}__score'] - sc[r]))))
                        stored_flags[(r + ('_cc' if variant == 'channels_cell' else '_ga'))] = {u: z[f'{r}__{u}__flags'].astype(bool) for u in RULES}
            with np.load(TCI / 'scores' / f'{bk}__{cell}.npz') as z:
                assert np.array_equal(z['answers'], ans)
                scores['TCI_ONLY_ga'] = z['TCI_ONLY_ga'].astype(float); surv = z['surv'].astype(int)
            raw_surv = bank['raw'][:, surv]
            for name, s in scores.items():
                own = None
                if name in METHODS + EXTRA:
                    own, _, _ = rules(s, bank, raw_surv, data.entropy[ans], pop.uids[ans])
                    if name in stored_flags:
                        for u in RULES:
                            if not np.array_equal(own[u], stored_flags[name][u]): raise RuntimeError(f'flag replay {bk}/{cell}/{name}/{u}')
                            checks['flag_replays'] += 1
                for j, i in enumerate(ans):
                    a, b = off[j], off[j + 1]
                    firsts = None; f_own = None
                    if own is not None:
                        firsts = {}
                        for u, tag in (('R2_COUNT', 'R2'), ('ENTROPY_GATE_Q80', 'ENT'), ('ANSWER_Z_Q80', 'AZ')):
                            ff = own[u][a:b]; firsts[tag] = int(np.flatnonzero(ff)[0]) if ff.any() else -1
                        f_own = own['ANSWER_Z_Q80'][a:b]
                    fill(R[(bk, name)], i, s[a:b], f_own=f_own, firsts=firsts)
        print(bk, 'done', round(time.time() - t0, 1), flush=True)

    # ---------------- aggregation helpers
    grp = pop.groups
    prm_idx = np.flatnonzero(~pop.pb); pb_idx = np.flatnonzero(pop.pb)
    pg, pinv = np.unique(grp[prm_idx], return_inverse=True); bg, binv = np.unique(grp[pb_idx], return_inverse=True)
    rng = np.random.default_rng(SEED)
    Wp = rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=DRAWS).astype(float)
    Wb = rng.multinomial(len(bg), np.full(len(bg), 1 / len(bg)), size=DRAWS).astype(float)
    Wp = np.vstack([np.ones(len(pg)), Wp]); Wb = np.vstack([np.ones(len(bg)), Wb])  # row 0 = point estimate
    pb_cells = sorted(set(cells[pb_idx])); cell_of = np.array([pb_cells.index(c) for c in cells[pb_idx]])
    err_pb = target[pb_idx] >= 0
    longest = np.array([int(np.argmax(TLEN[OFF[i]:OFF[i + 1]])) for i in pb_idx])  # integer lengths: first longest step
    content_only = err_pb & (target[pb_idx] != 0) & (longest != target[pb_idx])
    cls = np.array([meta[ids[i]]['classification'] for i in prm_idx]); cat = np.array([meta[ids[i]]['category'] for i in prm_idx])

    def gsum(X, inv, G):
        out = np.zeros((G, X.shape[1])); np.add.at(out, inv, X); return out

    def prm_stats(store, mask=None):
        """Return dict of arrays (DRAWS+1,) for PRMBench metrics."""
        m = np.ones(len(prm_idx), bool) if mask is None else mask
        a = store['auc'][prm_idx]; t1 = store['top1err'][prm_idx]
        X = np.column_stack([np.where(m & np.isfinite(a), a, 0), m & np.isfinite(a), np.where(m & np.isfinite(t1), t1, 0), m & np.isfinite(t1)])
        S = Wp @ gsum(X.astype(float), pinv, len(pg))
        out = {'within_auc': S[:, 0] / S[:, 1], 'top1_on_error': S[:, 2] / S[:, 3]}
        for key, cname in (('prmscore_top1', 'c1'), ('prmscore_top2_posthoc', 'c2'), ('prmscore_own', 'cown')):
            c = Wp @ gsum(store[cname][prm_idx] * m[:, None], pinv, len(pg))
            tp, fp, tn, fn = c.T
            with np.errstate(invalid='ignore', divide='ignore'):
                f1c = 2 * tp / (2 * tp + fp + fn); f1e = 2 * tn / (2 * tn + fp + fn)
            out[key] = .5 * (f1c + f1e)
            if key == 'prmscore_top1': out['f1_correct_top1'] = f1c; out['f1_error_top1'] = f1e
            if key == 'prmscore_own': out['flag_rate_own'] = (tn + fn) / c.sum(1)
        return out

    def pb_stats(store, field='argmax', mask=None, f1=False):
        """Cell-equal mean of hit rate on erroneous answers (and F1 for decision rules)."""
        m = np.ones(len(pb_idx), bool) if mask is None else mask
        pred = store[field][pb_idx]; t = target[pb_idx]; hit = pred == t
        cols = []
        for k in range(len(pb_cells)):
            ck = (cell_of == k) & m
            cols += [hit & err_pb & ck, err_pb & ck, hit & ~err_pb & ck, ~err_pb & ck]
        S = Wb @ gsum(np.column_stack(cols).astype(float), binv, len(bg))
        S = S.reshape(len(Wb), len(pb_cells), 4)
        with np.errstate(invalid='ignore', divide='ignore'):
            ae = S[:, :, 0] / S[:, :, 1]; ac = S[:, :, 2] / S[:, :, 3]
            out = {'error_hit': np.nanmean(ae, 1)}
            if f1:
                f = np.where(ae + ac > 0, 2 * ae * ac / np.where(ae + ac > 0, ae + ac, 1), 0.)
                out['official_f1'] = np.nanmean(f, 1); out['clean_correct'] = np.nanmean(ac, 1)
        return out

    def ci(x):
        return float(x[0]), float(np.nanquantile(x[1:], .025)), float(np.nanquantile(x[1:], .975))

    # ---------------- validation against frozen metrics
    for bk in BANKS:
        for name in METHODS:
            st = prm_stats(R[(bk, name)])
            ref = M[(M.bank == bk) & (M.cell == PRM) & (M.method == name)].set_index('decision_rule')
            for got, refv in ((st['within_auc'][0], ref.loc['TOP_1', 'within_answer_auc']), (st['prmscore_top1'][0], ref.loc['TOP_1', 'prmscore']),
                              (st['prmscore_top2_posthoc'][0], ref.loc['TOP_2', 'prmscore']), (st['prmscore_own'][0], ref.loc['ANSWER_Z_Q80', 'prmscore'])):
                checks['metric_replay_max'] = max(checks['metric_replay_max'], abs(got - refv))
            refpb = M[(M.bank == bk) & (M.cell != PRM) & (M.method == name)]
            got = pb_stats(R[(bk, name)])['error_hit'][0]
            checks['metric_replay_max'] = max(checks['metric_replay_max'], abs(got - refpb[refpb.decision_rule == 'TOP_1'].pb_first_error_accuracy.mean()))
            for tag, rule in (('R2', 'R2_COUNT'), ('ENT', 'ENTROPY_GATE_Q80'), ('AZ', 'ANSWER_Z_Q80')):
                g = pb_stats(R[(bk, name)], 'first_' + tag, f1=True)
                checks['metric_replay_max'] = max(checks['metric_replay_max'], abs(g['official_f1'][0] - refpb[refpb.decision_rule == rule].pb_official_f1.mean()))
        for name in CONTROLS + EXTRA:
            st = prm_stats(R[(bk, name)])
            variant, recipe = ('groups_answer', name) if name in CONTROLS else ('groups_answer', 'HEM_MOM')
            ref = C[(C.bank == bk) & (C.variant == variant) & (C.recipe == recipe) & (C.decision_rule == 'TOP_1')]
            checks['control_replay_max'] = max(checks['control_replay_max'], abs(st['within_auc'][0] - float(ref[ref.cell == PRM].within_answer_auc.iloc[0])),
                                               abs(st['prmscore_top1'][0] - float(ref[ref.cell == PRM].prmscore.iloc[0])),
                                               abs(pb_stats(R[(bk, name)])['error_hit'][0] - ref[ref.cell != PRM].pb_first_error_accuracy.mean()))
    if checks['score_replay_max'] > 1e-9 or checks['metric_replay_max'] > 1e-12 or checks['control_replay_max'] > 1e-12:
        dump(OUT / 'VALIDATION.json', {**checks, 'status': 'FAIL'}); raise RuntimeError(checks)
    print('validation', checks, flush=True)

    # ---------------- Panel A
    rowsA, conA = [], []
    for bk in BANKS:
        cache = {}
        for name in METHODS + EXTRA + CONTROLS:
            p = prm_stats(R[(bk, name)]); b = pb_stats(R[(bk, name)]); cache[name] = (p['within_auc'], b['error_hit'])
            rowsA.append({'bank': bk, 'scorer': name, 'role': 'control' if name in CONTROLS else 'method', 'n_channels': int(bk[1:]),
                          'digit_channels': bk in ('B16', 'B23', 'B35', 'B54'),
                          'prmb_within_auc': p['within_auc'][0], 'prmb_prmscore_top1': p['prmscore_top1'][0], 'pb_argmax_hit': b['error_hit'][0]})
        for name in METHODS + EXTRA:
            for ref in ('DS_FILTER_EQ', 'ALL_EQ'):
                for k, metric in enumerate(('prmb_within_auc', 'pb_argmax_hit')):
                    d = cache[name][k] - cache[ref][k]; pt, lo, hi = ci(d)
                    conA.append({'bank': bk, 'method': name, 'reference': ref, 'metric': metric, 'delta': pt, 'lo95': lo, 'hi95': hi})
    A = pd.DataFrame(rowsA); A.to_csv(OUT / 'PANEL_A_ABSOLUTE.csv', index=False); pd.DataFrame(conA).to_csv(OUT / 'PANEL_A_CONTRASTS.csv', index=False)

    # ---------------- Panel B
    def prm_row(label, store, bank, mask=None, group='ALL'):
        p = prm_stats(store, mask)
        return {'group': group, 'scorer': label, 'bank': bank, **{k: float(v[0]) for k, v in p.items()}}
    rowsB, typesB = [], []
    comp_stores = {'Qwen2.5-Math-PRM-7B (supervised)': R['PRM'], 'Mind the Gap (local repro.)': R['MTG'],
                   'Step index (later = riskier)': R['STEP_INDEX'], 'Step token length': R['STEP_LENGTH']}
    for bk in BANKS:
        for name in METHODS:
            rowsB.append(prm_row(name, R[(bk, name)], bk))
    for label, store in comp_stores.items(): rowsB.append(prm_row(label, store, '-'))
    groups_def = [('cls:' + c, cls == c) for c in sorted(set(cls) - {'correct'})] + [('cat:' + c, cat == c) for c in sorted(set(cat) - {'control'})]
    for gname, mask in groups_def:
        for name in METHODS:
            vals = [prm_row(name, R[(bk, name)], bk, mask, gname) for bk in BANKS]
            for v in vals: typesB.append(v)
        for label, store in comp_stores.items(): typesB.append(prm_row(label, store, '-', mask, gname))
    pd.DataFrame(rowsB).to_csv(OUT / 'PANEL_B_PRMBENCH.csv', index=False); pd.DataFrame(typesB).to_csv(OUT / 'PANEL_B_PRMBENCH_TYPES.csv', index=False)

    # ProcessBench strata
    rel = np.where(err_pb, target[pb_idx] / np.maximum(NS[pb_idx] - 1, 1), np.nan)
    strata = [('ALL', np.ones(len(pb_idx), bool)), ('content-only (not step 0, not longest)', content_only),
              ('error at step 0', err_pb & (target[pb_idx] == 0)), ('error in longest step', err_pb & (longest == target[pb_idx]) & (target[pb_idx] != 0))]
    for s in SUBSETS: strata.append(('subset:' + s, np.array([c.split('_')[1] == s for c in cells[pb_idx]])))
    for lab, lo, hi in (('pos:early', 0, .34), ('pos:middle', .34, .67), ('pos:late', .67, 1.01)):
        strata.append((lab, (rel >= lo) & (rel < hi) & (target[pb_idx] != 0)))
    for lab, lo, hi in (('len:1-5', 1, 5), ('len:6-10', 6, 10), ('len:11+', 11, 10 ** 6)):
        strata.append((lab, (NS[pb_idx] >= lo) & (NS[pb_idx] <= hi)))
    pb_ref = {'Mind the Gap (local repro.)': (R['MTG'], 'argmax'), 'Qwen2.5-Math-PRM-7B argmax': (R['PRM'], 'argmax'),
              'Qwen2.5-Math-PRM-7B native (0.5)': (R['PRM'], 'pb_native_first'), 'Qwen2.5-72B critic native': (R['CRITIC'], 'pb_native_first')}
    rowsP = []
    for sname, mask in strata:
        n_err = int((mask & err_pb).sum())
        for bk in BANKS:
            for name in METHODS:
                rowsP.append({'stratum': sname, 'scorer': name, 'bank': bk, 'n_error_rows': n_err, 'argmax_hit': float(pb_stats(R[(bk, name)], 'argmax', mask)['error_hit'][0])})
        for label, (store, field) in pb_ref.items():
            rowsP.append({'stratum': sname, 'scorer': label, 'bank': '-', 'n_error_rows': n_err, 'argmax_hit': float(pb_stats(store, field, mask)['error_hit'][0])})
        stub = {'argmax': np.full(N, -9, int)}
        for label, pred in (('Always step 0', np.zeros(len(pb_idx), int)), ('Longest step', longest)):
            stub['argmax'][pb_idx] = pred
            rowsP.append({'stratum': sname, 'scorer': label, 'bank': '-', 'n_error_rows': n_err, 'argmax_hit': float(pb_stats(stub, 'argmax', mask)['error_hit'][0])})
    pd.DataFrame(rowsP).to_csv(OUT / 'PANEL_B_PROCESSBENCH.csv', index=False)
    # own-rule F1 (different access, labelled)
    rowsF = []
    for bk in BANKS:
        for name in METHODS:
            for tag, rule in (('R2', 'R2_COUNT'), ('ENT', 'ENTROPY_GATE_Q80'), ('AZ', 'ANSWER_Z_Q80')):
                g = pb_stats(R[(bk, name)], 'first_' + tag, f1=True)
                rowsF.append({'scorer': name, 'bank': bk, 'rule': rule, 'official_f1': g['official_f1'][0], 'error_hit': g['error_hit'][0], 'clean_correct': g['clean_correct'][0]})
    for label, (store, field) in list(pb_ref.items())[2:]:
        g = pb_stats(store, field, f1=True)
        rowsF.append({'scorer': label, 'bank': '-', 'rule': 'native', 'official_f1': g['official_f1'][0], 'error_hit': g['error_hit'][0], 'clean_correct': g['clean_correct'][0]})
    pd.DataFrame(rowsF).to_csv(OUT / 'PANEL_B_PROCESSBENCH_F1.csv', index=False)

    # contrasts at the highlighted bank
    conB = []
    for name in METHODS:
        mine = prm_stats(R[(HIGHLIGHT, name)])
        for label, store in (('Mind the Gap (local repro.)', R['MTG']), ('Qwen2.5-Math-PRM-7B (supervised)', R['PRM'])):
            other = prm_stats(store)
            for metric in ('within_auc', 'top1_on_error', 'prmscore_top1'):
                pt, lo, hi = ci(mine[metric] - other[metric])
                conB.append({'benchmark': 'PRMBench', 'method': name, 'bank': HIGHLIGHT, 'versus': label, 'metric': metric, 'stratum': 'ALL', 'delta': pt, 'lo95': lo, 'hi95': hi})
        for label, (store, field) in list(pb_ref.items())[:2]:
            for sname, mask in strata[:2]:
                d = pb_stats(R[(HIGHLIGHT, name)], 'argmax', mask)['error_hit'] - pb_stats(store, field, mask)['error_hit']
                pt, lo, hi = ci(d)
                conB.append({'benchmark': 'ProcessBench', 'method': name, 'bank': HIGHLIGHT, 'versus': label, 'metric': 'argmax_hit', 'stratum': sname, 'delta': pt, 'lo95': lo, 'hi95': hi})
    pd.DataFrame(conB).to_csv(OUT / 'PANEL_B_CONTRASTS.csv', index=False)

    # ---------------- per-answer archive
    arrays = {'answer_ids': ids.astype(str), 'cells': cells.astype(str), 'target': target}
    for key, store in R.items():
        nm = key if isinstance(key, str) else f'{key[0]}__{key[1]}'
        for f in ('auc', 'top1err', 'argmax', 'first_R2', 'first_ENT', 'first_AZ', 'pb_native_first'):
            if f in store: arrays[f'{nm}__{f}'] = store[f]
    np.savez_compressed(OUT / 'PER_ANSWER.npz', **arrays)
    status = {'status': 'BUILT', **checks, 'critic_missing_predictions_per_cell_pair': crit_missing,
              'n_prmbench_answers': int(len(prm_idx)), 'n_processbench_rows': int(len(pb_idx)), 'n_pb_error_rows': int(err_pb.sum()),
              'n_content_only_error_rows': int(content_only.sum()), 'runtime_s': round(time.time() - t0, 1),
              'build_sha256': sha(Path(__file__)), 'protocol_sha256': sha(OUT / 'PROTOCOL.json')}
    dump(OUT / 'VALIDATION.json', status); print(json.dumps(status, indent=1))


if __name__ == '__main__':
    main()
