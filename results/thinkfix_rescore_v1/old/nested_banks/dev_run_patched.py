# PATCHED COPY of results/monday_nested_banks_v2/dev_run.py for thinkfix_rescore_v1 (variant=old).
# Changes: OUT = this directory; ext.FE set by the wrapper; recheck enforced only for variant=old;
# export hook after each fit; Mind the Gap reference not recomputed (input gone); step spans from
# STEP_TOKEN_SPANS_OLD.npz. Everything else is identical to the frozen script (see DIFF.txt).
"""monday_nested_banks_v1 development run: Monday methods on nested banks over the 13,769 source answers.

Fits label-free per cell with the validated adapter (monday_external_panel_v1/score.py), then builds
Panel A (methods vs ALL_EQ, decomposition vs DS_FILTER_EQ) and Panel B (highlight bank vs competitors).
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import ast, hashlib, importlib.util, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = Path(__file__).resolve().parent
EXT = MAIN / 'results/monday_external_panel_v1'
SE = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'; FL = MAIN / 'dataset_cache/four_localization'
PANEL = MAIN / 'results/monday_candidate_panel_v1/run_20261004'
spec = importlib.util.spec_from_file_location('ext_score', EXT / 'score.py'); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
MAN = json.loads((MAIN / 'results/estimator_input_v1/run_20261004_resume/INPUT_MANIFEST.json').read_text(encoding='utf8'))
OB = MAN['banks']; CORE = OB['B13'][:11]; REAL = OB['B13'][11:]; DIG = OB['B16'][-3:]
E20 = [c for c in OB['B20'] if c not in CORE]; E51 = [c for c in OB['B51'] if c not in OB['B20']]
HIST18 = [c[4:] for c in OB['B32'] if c.startswith('lf__') and c[4:] not in E20]   # the old 32/35 bank's history family, raw
NB = {'N11': CORE, 'N13': CORE + REAL, 'N22': CORE + REAL + E20, 'N40': CORE + REAL + E20 + HIST18,
      'N53': CORE + REAL + E20 + E51}   # same order as v1 (tie keys depend on column order)
NB.update({{'N11': 'N14', 'N13': 'N16', 'N22': 'N25', 'N40': 'N43', 'N53': 'N56'}[k]: v + DIG for k, v in list(NB.items())})
BANKS = ['N11', 'N14', 'N13', 'N16', 'N22', 'N25', 'N53', 'N56', 'N40', 'N43']
ext.BN = dict(NB, B13=OB['B13'])                       # adapter reads its module-level bank dictionary
PRM = 'prmbench_qwen3_8b'; METHODS4 = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
METHODS = METHODS4 + ['HEM_MOM_ga']; CONTROLS = ['ALL_EQ', 'DS_FILTER_EQ', 'MEAN_EQ', 'HEM_EQ']
SUBSETS = ['gsm8k', 'math', 'olympiadbench', 'omnimath']; DRAWS, SEED = 5000, 20261006


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


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
    if (OUT / 'PER_ANSWER.npz').exists(): raise SystemExit('outputs exist')
    P = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8')); assert P['status'] == 'FROZEN'
    # nesting check (hard)
    for fam in (['N11', 'N13', 'N22', 'N40', 'N53'], ['N14', 'N16', 'N25', 'N43', 'N56']):
        for a, b in zip(fam[:-1], fam[1:]):
            if not set(NB[a]) < set(NB[b]): raise SystemExit(f'not nested {a} {b}')
    for a, b in zip(['N11', 'N13', 'N22', 'N40', 'N53'], ['N14', 'N16', 'N25', 'N43', 'N56']):
        if not set(NB[a]) < set(NB[b]): raise SystemExit('digit twin not nested')
    json.dump({k: NB[k] for k in BANKS}, open(OUT / 'BANKS.json', 'w'), indent=1)
    A = pd.read_csv(SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
    with np.load(SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
    NS = np.diff(OFF); N = len(A); cells = A.cell.to_numpy(); ids = A.id.astype(str).to_numpy(); target = A.target.to_numpy().astype(int)
    uids = A.uid.astype(str).to_numpy(); groups = A.source_group.to_numpy(); pb = np.array([c.startswith('pb_') for c in cells])
    fr = json.loads((SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
    meta = {m['idx']: m for m in pickle.load(open(fr['prm_metadata']['path'], 'rb')).values()}
    noncontrol = np.array([(not pb[i]) and meta[ids[i]]['classification'] != 'correct' for i in range(N)])
    MTG = None  # thinkfix_rescore_v1: EVIDENCE_DROP.npz is gone; Mind the Gap rows are not recomputed
    with np.load(_SPANS) as z: spans = z['step_spans']
    TLEN = (spans[:, 1] - spans[:, 0]).astype(float)
    with np.load(MAN['inputs']['answer_features']['path']) as z:
        if not np.array_equal(z['ids'].astype(str), ids): raise SystemExit('entropy identity')
        ENT = z['X'][:, list(map(str, z['names'])).index('epr')].astype(float)
    F = np.load(ext.FE / 'source/FEATURES.npz')
    if list(map(str, F['uids'])) != A.uid.astype(str).tolist() or not np.array_equal(F['offsets'], OFF): raise SystemExit('source order')
    pop = ext.Population(F, BANKS + ['B13']); print('population built', round(time.time() - t0), flush=True)
    sorted_cells = sorted(set(cells))
    # re-check of the validated path on original B13 / PRMBench
    ans_prm = np.flatnonzero(cells == PRM); bank = pop.cell_bank('B13', ans_prm)
    sc, _, _ = ext.fit_cell(pop, bank, 1000 * 0 + sorted_cells.index(PRM))
    with np.load(PANEL / 'candidate_scores' / f'{PRM}__B13__groups_answer.npz') as z: recheck = float(np.max(np.abs(sc['MEAN_MOM_ga'] - z['MEAN_MOM__score'])))
    if recheck > 1e-9 and _ENFORCE_RECHECK: raise SystemExit(f'recheck failed {recheck}')
    (OUT / 'SOURCE_PARITY_GATE.json').write_text(json.dumps({'status': ('PASS' if recheck <= 1e-9 else 'RECORDED_NOT_ENFORCED'), 'variant': _VARIANT, 'recheck_B13_PRMBench_MEAN_MOM_ga_max_abs_diff': recheck,
        'full_gate': str(EXT / 'SOURCE_PARITY_GATE.json'), 'full_gate_sha256': sha(EXT / 'SOURCE_PARITY_GATE.json')}, indent=1), encoding='utf8')
    print('recheck', recheck, flush=True)

    def new():
        return {'auc': np.full(N, np.nan), 'top1err': np.full(N, np.nan), 'c1': np.zeros((N, 4), int), 'c2': np.zeros((N, 4), int),
                'cown': np.zeros((N, 4), int), 'argmax': np.full(N, -9, int), 'first_R2': np.full(N, -9, int), 'first_ENT': np.full(N, -9, int), 'first_AZ': np.full(N, -9, int),
                'failed': np.zeros(N, bool)}

    def fill(store, i, s, f_own=None, firsts=None):
        y = LAB[OFF[i]:OFF[i + 1]]; s = np.asarray(s, float)
        store['argmax'][i] = int(np.flatnonzero(s >= s.max() - 8 * np.finfo(float).eps)[0])
        if not pb[i]:
            store['auc'][i] = auc(y, s)
            if y.any(): store['top1err'][i] = float(y[store['argmax'][i]])
            if noncontrol[i]:
                store['c1'][i] = conf(topk(s, 1), y); store['c2'][i] = conf(topk(s, 2), y)
                if f_own is not None: store['cown'][i] = conf(f_own, y)
        if firsts:
            for k, v in firsts.items(): store['first_' + k][i] = v

    R = {}
    for name in ['PRM', 'MTG', 'STEP_INDEX', 'STEP_LENGTH']: R[name] = new()
    for i in range(N):
        sl = slice(OFF[i], OFF[i + 1]); fill(R['STEP_INDEX'], i, np.arange(NS[i]).astype(float)); fill(R['STEP_LENGTH'], i, TLEN[sl])
        if not pb[i]: m = meta[ids[i]]; fill(R['PRM'], i, -np.asarray(m['rewards'], float), f_own=np.asarray(m['labels']) == 0)
    R['PRM']['pb_native_first'] = np.full(N, -9, int); R['CRITIC'] = {'pb_native_first': np.full(N, -9, int)}
    by = {s: ({x['id']: x for x in pickle.load(open(FL / f'pb_prm_qwen25math7b_full/pb_prm_{s}.pkl', 'rb')).values()},
              {x['id']: x for x in pickle.load(open(FL / f'pb_critic_qwen72b_full/pb_critic_{s}.pkl', 'rb')).values()}) for s in SUBSETS}
    for i in np.flatnonzero(pb):
        dp, dc = by[cells[i].split('_')[1]]; key = ids[i].split('::')[1]; x = dp[key]
        rw = np.asarray(x['rewards'] if isinstance(x['rewards'], list) else ast.literal_eval(x['rewards']), float)
        if len(rw) != NS[i] or int(x['label']) != target[i]: raise SystemExit('PB cache alignment')
        R['PRM']['argmax'][i] = int(np.flatnonzero(-rw >= (-rw).max() - 8 * np.finfo(float).eps)[0]); R['PRM']['pb_native_first'][i] = int(x['prediction'])
        cp = dc[key]['prediction']; R['CRITIC']['pb_native_first'][i] = -9 if cp is None else int(cp)
    diag = {}; failures = []
    for bi, bk in enumerate(BANKS):
        for name in METHODS + CONTROLS: R[(bk, name)] = new()
        for ci, ce in enumerate(sorted_cells):
            ans = np.flatnonzero(cells == ce); bank = pop.cell_bank(bk, ans); off = bank['off']
            try:
                sc, surv, d = ext.fit_cell(pop, bank, 1000 * bi + ci); diag[f'{bk}/{ce}'] = d
                _HOOK(bk, ce, bank, sc, surv)
            except RuntimeError as exc:                      # Amendment A1: record, no fallback
                st0 = ext.fit_content(pop, bank, recipes=[]); sc = {k: st0['scores'][k] for k in ('ALL_EQ', 'DS_FILTER_EQ') if k in st0['scores']}
                surv = st0['surv']; failures.append({'bank': bk, 'cell': ce, 'reason': str(exc)[:300], 'computed': sorted(sc)})
                diag[f'{bk}/{ce}'] = {'FAILED': str(exc)[:300]}
                for name in METHODS + CONTROLS:
                    if name not in sc: R[(bk, name)]['failed'][ans] = True
            for name in METHODS + CONTROLS:
                if name not in sc: continue
                s = sc[name]; own = None
                if name in METHODS:
                    own, _, _ = ext.rules(s, bank, bank['raw'][:, surv], ENT[ans], uids[ans])
                for j, i in enumerate(ans):
                    a, b = off[j], off[j + 1]; firsts = None; f_own = None
                    if own is not None:
                        firsts = {}
                        for u, tag in (('R2_COUNT', 'R2'), ('ENTROPY_GATE_Q80', 'ENT'), ('ANSWER_Z_Q80', 'AZ')):
                            ff = own[u][a:b]; firsts[tag] = int(np.flatnonzero(ff)[0]) if ff.any() else -1
                        f_own = own['ANSWER_Z_Q80'][a:b]
                    fill(R[(bk, name)], i, s[a:b], f_own, firsts)
            print(bk, ce, round(time.time() - t0), flush=True)
    incomplete = {(bk, n) for bk in BANKS for n in METHODS + CONTROLS if R[(bk, n)]['failed'].any()}
    (OUT / 'COVERAGE.json').write_text(json.dumps({'amendment': 'A1', 'failures': failures,
        'incomplete': sorted(f'{b}/{n}' for b, n in incomplete)}, indent=1), encoding='utf8')
    (OUT / 'FIT_DIAGNOSTICS.json').write_text(json.dumps(diag, indent=1, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf8')

    # ---------------- aggregation (same definitions as monday_presentation_panels_v1/build.py)
    prm_idx = np.flatnonzero(~pb); pb_idx = np.flatnonzero(pb)
    pg, pinv = np.unique(groups[prm_idx], return_inverse=True); bg, binv = np.unique(groups[pb_idx], return_inverse=True)
    rng = np.random.default_rng(SEED)
    Wp = np.vstack([np.ones(len(pg)), rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=DRAWS)]).astype(float)
    Wb = np.vstack([np.ones(len(bg)), rng.multinomial(len(bg), np.full(len(bg), 1 / len(bg)), size=DRAWS)]).astype(float)
    pb_cells = sorted(set(cells[pb_idx])); cell_of = np.array([pb_cells.index(c) for c in cells[pb_idx]]); err_pb = target[pb_idx] >= 0
    longest = np.array([int(np.argmax(TLEN[OFF[i]:OFF[i + 1]])) for i in pb_idx])
    content_only = err_pb & (target[pb_idx] != 0) & (longest != target[pb_idx])
    cls = np.array([meta[ids[i]]['classification'] for i in prm_idx])

    def gsum(X, inv, G):
        out = np.zeros((G, X.shape[1])); np.add.at(out, inv, X); return out

    def prm_stats(store, mask=None):
        m = np.ones(len(prm_idx), bool) if mask is None else mask
        a = store['auc'][prm_idx]; t1 = store['top1err'][prm_idx]
        X = np.column_stack([np.where(m & np.isfinite(a), a, 0), m & np.isfinite(a), np.where(m & np.isfinite(t1), t1, 0), m & np.isfinite(t1)])
        S = Wp @ gsum(X.astype(float), pinv, len(pg)); out = {'within_auc': S[:, 0] / S[:, 1], 'top1_on_error': S[:, 2] / S[:, 3]}
        for key, cn in (('prmscore_top1', 'c1'), ('prmscore_top2_posthoc', 'c2'), ('prmscore_own', 'cown')):
            c = Wp @ gsum(store[cn][prm_idx] * m[:, None], pinv, len(pg)); tp, fp, tn, fn = c.T
            with np.errstate(invalid='ignore', divide='ignore'):
                f1c = 2 * tp / (2 * tp + fp + fn); f1e = 2 * tn / (2 * tn + fp + fn)
            out[key] = .5 * (f1c + f1e)
            if key == 'prmscore_top1': out['f1_correct_top1'] = f1c; out['f1_error_top1'] = f1e
        return out

    def pb_stats(store, field='argmax', mask=None, f1=False):
        m = np.ones(len(pb_idx), bool) if mask is None else mask
        hit = store[field][pb_idx] == target[pb_idx]; cols = []
        for k in range(len(pb_cells)):
            ck = (cell_of == k) & m; cols += [hit & err_pb & ck, err_pb & ck, hit & ~err_pb & ck, ~err_pb & ck]
        S = (Wb @ gsum(np.column_stack(cols).astype(float), binv, len(bg))).reshape(len(Wb), len(pb_cells), 4)
        with np.errstate(invalid='ignore', divide='ignore'):
            ae = S[:, :, 0] / S[:, :, 1]; ac = S[:, :, 2] / S[:, :, 3]; out = {'error_hit': np.nanmean(ae, 1)}
            if f1:
                f = np.where(ae + ac > 0, 2 * ae * ac / np.where(ae + ac > 0, ae + ac, 1), 0.)
                out['official_f1'] = np.nanmean(f, 1); out['clean_correct'] = np.nanmean(ac, 1)
        return out

    ci = lambda x: (float(x[0]), float(np.nanquantile(x[1:], .025)), float(np.nanquantile(x[1:], .975)))
    rowsA, conA, cache = [], [], {}
    for bk in BANKS:
        for name in METHODS + CONTROLS:
            if (bk, name) in incomplete:
                rowsA.append({'bank': bk, 'n_channels': len(NB[bk]), 'digits': bk in ('N14', 'N16', 'N25', 'N43', 'N56'), 'scorer': name, 'role': 'control' if name in CONTROLS else 'method',
                              'prmb_within_auc': np.nan, 'prmb_top1_on_error': np.nan, 'prmb_prmscore_top1': np.nan, 'pb_argmax_hit': np.nan, 'status': 'INCOMPLETE'}); continue
            p = prm_stats(R[(bk, name)]); b = pb_stats(R[(bk, name)]); cache[(bk, name)] = (p, b)
            rowsA.append({'bank': bk, 'n_channels': len(NB[bk]), 'digits': bk in ('N14', 'N16', 'N25', 'N43', 'N56'), 'scorer': name,
                          'role': 'control' if name in CONTROLS else 'method', 'prmb_within_auc': p['within_auc'][0], 'prmb_top1_on_error': p['top1_on_error'][0],
                          'prmb_prmscore_top1': p['prmscore_top1'][0], 'pb_argmax_hit': b['error_hit'][0], 'status': 'COMPLETE'})
        for name in METHODS + ['DS_FILTER_EQ']:
            for ref in ('ALL_EQ', 'DS_FILTER_EQ'):
                if name == ref or (bk, name) in incomplete or (bk, ref) in incomplete: continue
                for metric, (k1, k2) in (('prmb_within_auc', (0, 'within_auc')), ('pb_argmax_hit', (1, 'error_hit'))):
                    d = cache[(bk, name)][k1][k2] - cache[(bk, ref)][k1][k2]; pt, lo, hi = ci(d)
                    conA.append({'bank': bk, 'method': name, 'reference': ref, 'metric': metric, 'delta': pt, 'lo95': lo, 'hi95': hi})
    A_ = pd.DataFrame(rowsA); A_.to_csv(OUT / 'PANEL_A_ABSOLUTE.csv', index=False); pd.DataFrame(conA).to_csv(OUT / 'PANEL_A_CONTRASTS.csv', index=False)
    # bank-size contrasts within each digit family (nesting): next bank minus previous, per method
    conN = []
    for fam in (['N11', 'N13', 'N22', 'N40', 'N53'], ['N14', 'N16', 'N25', 'N43', 'N56']):
        for a, b in zip(fam[:-1], fam[1:]):
            for name in METHODS4 + ['ALL_EQ']:
                if (a, name) in incomplete or (b, name) in incomplete: continue
                for metric, (k1, k2) in (('prmb_within_auc', (0, 'within_auc')), ('pb_argmax_hit', (1, 'error_hit'))):
                    pt, lo, hi = ci(cache[(b, name)][k1][k2] - cache[(a, name)][k1][k2])
                    conN.append({'from': a, 'to': b, 'added': len(NB[b]) - len(NB[a]), 'scorer': name, 'metric': metric, 'delta': pt, 'lo95': lo, 'hi95': hi})
    pd.DataFrame(conN).to_csv(OUT / 'NESTING_CONTRASTS.csv', index=False)
    # highlight rule (pre-registered): highest mean PRMB AUC over the 4 registered methods
    eligible = [bk for bk in BANKS if not any((bk, m) in incomplete for m in METHODS4)]
    mean4 = A_[A_.scorer.isin(METHODS4) & A_.bank.isin(eligible)].groupby('bank').prmb_within_auc.mean().sort_values(ascending=False)
    HL = str(mean4.index[0])
    (OUT / 'HIGHLIGHT.json').write_text(json.dumps({'rule': P['development']['highlight_rule'], 'highlight': HL, 'mean_auc_by_bank': mean4.to_dict(),
                                                   'decided_before_external': True}, indent=1), encoding='utf8')
    # Panel B
    comp = {'Qwen2.5-Math-PRM-7B (supervised)': R['PRM'], 'Mind the Gap (local repro.)': R['MTG'], 'Step index (later = riskier)': R['STEP_INDEX'], 'Step token length': R['STEP_LENGTH']}
    rowsB, typesB = [], []
    for bk in BANKS:
        for name in METHODS + ['ALL_EQ', 'DS_FILTER_EQ']:
            if (bk, name) in incomplete: continue
            rowsB.append({'scorer': name, 'bank': bk, **{k: float(v[0]) for k, v in cache[(bk, name)][0].items()}})
    for lab, st in comp.items(): rowsB.append({'scorer': lab, 'bank': '-', **{k: float(v[0]) for k, v in prm_stats(st).items()}})
    for c_ in sorted(set(cls) - {'correct', 'multi_solutions'}):
        mask = cls == c_
        for bk in BANKS:
            for name in METHODS4 + ['ALL_EQ']:
                if (bk, name) in incomplete: continue
                typesB.append({'group': c_, 'scorer': name, 'bank': bk, **{k: float(v[0]) for k, v in prm_stats(R[(bk, name)], mask).items()}})
        for lab, st in comp.items(): typesB.append({'group': c_, 'scorer': lab, 'bank': '-', **{k: float(v[0]) for k, v in prm_stats(st, mask).items()}})
    pd.DataFrame(rowsB).to_csv(OUT / 'PANEL_B_PRMBENCH.csv', index=False); pd.DataFrame(typesB).to_csv(OUT / 'PANEL_B_PRMBENCH_TYPES.csv', index=False)
    strata = [('ALL', np.ones(len(pb_idx), bool)), ('content-only (not step 0, not longest)', content_only),
              ('error at step 0', err_pb & (target[pb_idx] == 0)), ('error in longest step', err_pb & (longest == target[pb_idx]) & (target[pb_idx] != 0))]
    strata += [('subset:' + s, np.array([c.split('_')[1] == s for c in cells[pb_idx]])) for s in SUBSETS]
    pb_ref = {'Mind the Gap (local repro.)': (R['MTG'], 'argmax'), 'Qwen2.5-Math-PRM-7B argmax': (R['PRM'], 'argmax'),
              'Qwen2.5-Math-PRM-7B native (0.5)': (R['PRM'], 'pb_native_first'), 'Qwen2.5-72B critic native': (R['CRITIC'], 'pb_native_first')}
    rowsP = []
    for sname, mask in strata:
        n_err = int((mask & err_pb).sum())
        for bk in BANKS:
            for name in METHODS + ['ALL_EQ']:
                if (bk, name) in incomplete: continue
                rowsP.append({'stratum': sname, 'scorer': name, 'bank': bk, 'n_error_rows': n_err, 'argmax_hit': float(pb_stats(R[(bk, name)], 'argmax', mask)['error_hit'][0])})
        for lab, (st, f) in pb_ref.items(): rowsP.append({'stratum': sname, 'scorer': lab, 'bank': '-', 'n_error_rows': n_err, 'argmax_hit': float(pb_stats(st, f, mask)['error_hit'][0])})
        stub = {'argmax': np.full(N, -9, int)}
        for lab, pred in (('Always step 0', np.zeros(len(pb_idx), int)), ('Longest step', longest)):
            stub['argmax'][pb_idx] = pred
            rowsP.append({'stratum': sname, 'scorer': lab, 'bank': '-', 'n_error_rows': n_err, 'argmax_hit': float(pb_stats(stub, 'argmax', mask)['error_hit'][0])})
    pd.DataFrame(rowsP).to_csv(OUT / 'PANEL_B_PROCESSBENCH.csv', index=False)
    rowsF = []
    for bk in BANKS:
        for name in METHODS:
            if (bk, name) in incomplete: continue
            for tag, rule in (('R2', 'R2_COUNT'), ('ENT', 'ENTROPY_GATE_Q80'), ('AZ', 'ANSWER_Z_Q80')):
                g = pb_stats(R[(bk, name)], 'first_' + tag, f1=True)
                rowsF.append({'scorer': name, 'bank': bk, 'rule': rule, 'official_f1': g['official_f1'][0], 'error_hit': g['error_hit'][0], 'clean_correct': g['clean_correct'][0]})
    for lab, (st, f) in list(pb_ref.items())[2:]:
        g = pb_stats(st, f, f1=True); rowsF.append({'scorer': lab, 'bank': '-', 'rule': 'native', 'official_f1': g['official_f1'][0], 'error_hit': g['error_hit'][0], 'clean_correct': g['clean_correct'][0]})
    pd.DataFrame(rowsF).to_csv(OUT / 'PANEL_B_PROCESSBENCH_F1.csv', index=False)
    conB = []
    for name in METHODS4 + ['ALL_EQ']:
        mine = cache[(HL, name)][0]
        for lab in ('Mind the Gap (local repro.)', 'Qwen2.5-Math-PRM-7B (supervised)'):
            other = prm_stats(comp[lab])
            for metric in ('within_auc', 'top1_on_error', 'prmscore_top1'):
                pt, lo, hi = ci(mine[metric] - other[metric])
                conB.append({'benchmark': 'PRMBench', 'method': name, 'bank': HL, 'versus': lab, 'metric': metric, 'stratum': 'ALL', 'delta': pt, 'lo95': lo, 'hi95': hi})
        for lab in ('Mind the Gap (local repro.)', 'Qwen2.5-Math-PRM-7B argmax'):
            st, f = pb_ref[lab]
            for sname, mask in strata[:2]:
                pt, lo, hi = ci(pb_stats(R[(HL, name)], 'argmax', mask)['error_hit'] - pb_stats(st, f, mask)['error_hit'])
                conB.append({'benchmark': 'ProcessBench', 'method': name, 'bank': HL, 'versus': lab, 'metric': 'argmax_hit', 'stratum': sname, 'delta': pt, 'lo95': lo, 'hi95': hi})
    pd.DataFrame(conB).to_csv(OUT / 'PANEL_B_CONTRASTS.csv', index=False)
    # clean-answer flag rates of ANSWER_Z_Q80 per benchmark (reconciles the 99% / 94% figures)
    cf = []
    for bk in BANKS:
        for name in METHODS4:
            if (bk, name) in incomplete: continue
            st = R[(bk, name)]
            g = pb_stats(st, 'first_AZ', f1=True); cf.append({'bank': bk, 'scorer': name, 'pb_clean_flagged_ANSWER_Z': 1 - g['clean_correct'][0], 'n_pb_clean_rows': int((~err_pb).sum())})
    pd.DataFrame(cf).to_csv(OUT / 'PB_CLEAN_FLAG_RATES.csv', index=False)
    arrays = {'answer_ids': ids, 'cells': cells.astype(str), 'target': target}
    for key, st in R.items():
        nm = key if isinstance(key, str) else f'{key[0]}__{key[1]}'
        for f in ('auc', 'top1err', 'argmax', 'first_R2', 'first_ENT', 'first_AZ', 'pb_native_first', 'failed'):
            if f in st: arrays[f'{nm}__{f}'] = st[f]
    np.savez_compressed(OUT / 'PER_ANSWER.npz', **arrays)
    (OUT / 'DEV_STATUS.json').write_text(json.dumps({'status': 'BUILT', 'highlight': HL, 'runtime_s': round(time.time() - t0), 'dev_run_sha256': sha(Path(__file__)),
        'protocol_sha256': sha(OUT / 'PROTOCOL.json'), 'adapter_sha256': sha(EXT / 'score.py')}, indent=1), encoding='utf8')
    print('done', HL, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
