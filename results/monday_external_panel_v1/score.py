"""monday_external_panel_v1 scoring: label-free refit of the Monday methods on each external cell.

Never opens external labels (evaluator_only). Stage 1 = source parity gate on the development PRMBench cell;
stage 2 = external scoring -> <cell>/PREDICTIONS.npz + SCORING_STATUS.json. Protocol: PROTOCOL.json.
Usage: python score.py [--gate-only]
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import argparse, hashlib, itertools, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = Path(__file__).resolve().parent
CODE = MAIN / 'results/monday_fusion_analysis_v1/code'; SSL = MAIN / '.worktrees/ssl-pseudolabel-residual-v1'
FE = SSL / 'results/external_banks_v4'; OLD = MAIN / 'results/lsml_external_generalization_v1/evaluation'
INPUTS = MAIN / 'scratch/external_generalization_private/inputs'
PANEL = MAIN / 'results/monday_candidate_panel_v1/run_20261004'; TCIR = MAIN / 'results/mom_filter_within_diagnostic_20261004/tci_wide_20261005'
SE = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'scripts/experiments'))
import spectral_utils  # noqa: E402
spectral_utils.__path__.append(str(CODE / 'spectral_utils'))
from spectral_utils.lsml_gate_locator_research import answer_standardize  # noqa: E402
from spectral_utils.monday_fusion_core import fit_content, quota_marks, between_weights, LEVEL  # noqa: E402
from spectral_utils.monday_fusion_decision import rules  # noqa: E402
from spectral_utils.monday_fusion_metrics import top_flags  # noqa: E402
import er_stage_a as SA, er_stage_b as SB, ds_group_weights as GW  # noqa: E402
from calfix_common import tail_marks  # noqa: E402

CELLS = {'hard2verify_qwen3_8b': 'hard2verify', 'socratic_qwen3_8b': 'socratic', 'socratic_qwq32b': 'socratic'}
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
SCORERS = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'HEM_MOM_ga', 'TCI_ONLY_ga', 'ALL_EQ', 'DS_FILTER_EQ', 'MEAN_EQ', 'HEM_EQ']
RULES = ['ANSWER_Z_Q80', 'R2_COUNT', 'TOP_1']; REFS = ['ct7', 'frozen_lsml', 'frozen_equal']
BN = json.loads((MAIN / 'results/estimator_input_v1/run_20261004_resume/INPUT_MANIFEST.json').read_text(encoding='utf8'))['banks']
PS = pd.read_csv(SSL / 'results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv').set_index('channel')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def stop(msg): raise SystemExit('HARD STOP: ' + msg)


def masked_std(x, offsets, active):
    out = np.zeros(x.shape)
    for a, b in zip(offsets[:-1], offsets[1:]):
        for j in range(x.shape[1]):
            v = active[a:b, j]; y = x[a:b, j][v]
            if len(y) and y.std() > 1e-12: out[a:b, j][v] = (y - y.mean()) / y.std()
    return out


def build_bank(cols, Xr, nm, dact, offsets):
    """algorithm_external_v1 build_bank (verified equal to the development bank)."""
    plain = [c for c in cols if c not in DIG]; src = [c[4:] if c.startswith('lf__') else c for c in plain]
    Vp = answer_standardize(Xr[:, [nm.index(c) for c in src]], offsets)
    for q, c in enumerate(plain):
        if c.startswith('lf__'): Vp[:, q] *= float(np.sign(PS.loc[c[4:], 'r_level_marginal'])) or 1.0
    Vd = masked_std(Xr[:, [nm.index(c) for c in DIG]], offsets, dact)
    out = np.zeros((len(Xr), len(cols)))
    for j, c in enumerate(cols): out[:, j] = Vd[:, DIG.index(c)] if c in DIG else Vp[:, plain.index(c)]
    return out


class Population:
    """Full feature population (the development source or one external cell) with Monday-style marks and keys."""

    def __init__(self, F, banks):
        self.nm = [str(x) for x in F['names']]; self.X = F['values'].astype(float); self.DA = F['digit_active'].astype(bool)
        self.off = F['offsets'].astype(np.int64); self.S = int(self.off[-1])
        self.POOLZ = answer_standardize(self.X, self.off)                      # answer z of every raw channel (pool_z analogue)
        self.level = self.POOLZ[:, [self.nm.index(c) for c in LEVEL]].mean(1)
        self.VB, self.MARKS, self.TTA, self.KEYB = {}, {}, {}, {}
        for bk in banks:
            V = build_bank(BN[bk], self.X, self.nm, self.DA, self.off); self.VB[bk] = V
            self.KEYB[bk] = np.random.default_rng(20260928).random(V.shape)
            self.MARKS[bk] = SB.random_tie_marks(V, self.off, .2, self.KEYB[bk])
            self.TTA[bk] = tail_marks(V, self.off, .2, tie_aware=True, centred=True)[0]
        self.KEYG = np.random.default_rng(20260929).random((self.S, 13))
        self.env = {'KEYG': self.KEYG}

    def cell_bank(self, bk, answers):
        """monday_fusion_core.variant_bank('original') on a subset of answers; returns bank dict + raw values."""
        off_all = self.off; ns = np.diff(off_all)
        rows = np.concatenate([np.arange(off_all[i], off_all[i + 1]) for i in answers]); off = np.r_[0, np.cumsum(ns[answers])]
        names = BN[bk]; v = self.VB[bk][rows].copy(); signs = {}; flips = []
        level = self.level[rows]
        raw = np.column_stack([self.X[rows, self.nm.index(c[4:] if c.startswith('lf__') else c)] for c in names])
        for j, c in enumerate(names):
            if c.startswith('lf__'):
                r = float(np.corrcoef(self.POOLZ[rows, self.nm.index(c[4:])], level)[0, 1])
                if not np.isfinite(r): stop('undefined orientation ' + c)
                old = float(np.sign(PS.loc[c[4:], 'r_level_marginal'])) or 1.; sign = float(np.sign(r)) or 1.
                v[:, j] *= sign / old; raw[:, j] *= sign
                if sign != old: flips.append(j)
                signs[c] = {'correlation': r, 'sign': sign}
        key = self.KEYB[bk][rows]; marks = self.MARKS[bk][rows].copy(); tt = self.TTA[bk][rows].copy()
        if flips:
            marks[:, flips] = SB.random_tie_marks(v[:, flips], off, .2, key[:, flips])
            tt[:, flips] = tail_marks(v[:, flips], off, .2, tie_aware=True, centred=True)[0]
        return {'V': v, 'marks': marks, 'TT': tt, 'key': key, 'off': off, 'rows': rows, 'answers': np.asarray(answers),
                'names': list(names), 'raw': raw, 'diagnostics': {'removed': [], 'lf_signs': signs, 'sign_flips': [names[j] for j in flips]}}


def violation4(M):
    z = M - M.mean(0); n = len(z); C = z.T @ z / n; out = []
    for s in range(4):
        rest = [q for q in range(4) if q != s]; prs = list(itertools.combinations(rest, 2))
        t = np.array([np.mean(z[:, p] * z[:, q] * z[:, s]) for p, q in prs]); c = np.array([C[p, q] for p, q in prs])
        den = (t @ t) * (c @ c); out.append(1. if den <= 0 else 1 - (t @ c) ** 2 / den)
    return float(np.mean(out))


def fit_cell(pop, bank, tci_seed):
    """All scorers for one bank x cell; mirrors fit_content + estimator_state('channels','cell') + tci_wide_stage1."""
    need = ['MEAN_EQ', 'MEAN_MOM', 'HEM_EQ', 'HEM_MOM']
    st = fit_content(pop, bank, recipes=need)
    missing = [r for r in need + ['DS_FILTER_EQ'] if r not in st['scores']]
    if missing: raise RuntimeError(f'fit failures {missing}: ' + json.dumps(st['failures'], default=str))
    off = bank['off']; X = st['X']; surv = st['surv']; ai = int(st['anchor'])
    groups_raw = np.asarray(st['groups'], int); groups = GW.canon(groups_raw)   # raw labels index the fitted Z; canon for the TCI rebuild (as tci_wide_stage1)
    ganchor = int(groups[ai]); G = int(groups.max()) + 1; keyG = pop.KEYG[bank['rows'], :G]
    sc = {k: st['scores'][k] for k in ('ALL_EQ', 'DS_FILTER_EQ', 'MEAN_EQ', 'HEM_EQ')}
    sc['MEAN_MOM_ga'] = st['scores']['MEAN_MOM']; sc['HEM_MOM_ga'] = st['scores']['HEM_MOM']
    channel = quota_marks(bank['V'][:, surv], off, bank['key'][:, surv], True)
    for within in ('MEAN', 'HEM'):
        z = st['recipes'][within + '_EQ']['Z']; votes = quota_marks(z, off, keyG, False)
        w, _ = between_weights('MOM', z, votes, X[:, ai], int(groups_raw[ai]), channel_votes=channel, groups=groups_raw, channel_anchor=ai)
        sc[within + '_MOM_cc'] = GW.weighted_group_score(z, w)
    # TCI_ONLY_ga (Prop 2.4 triplets, MoM between on group votes)
    def fuse_ga(within):
        z = GW.group_matrix(X, off, groups, within, answer_standardize); votes = quota_marks(z, off, keyG, False)
        w, _ = between_weights('MOM', z, votes, X[:, ai], ganchor); return GW.weighted_group_score(z, w)
    gate = float(np.max(np.abs(fuse_ga(None) - sc['MEAN_MOM_ga'])))
    if gate > 1e-12: raise RuntimeError(f'all-members replay gate {gate}')
    keyX = np.asarray(bank['key'], float)[:, surv]; keyA = np.random.default_rng(tci_seed).random((len(X), 1))
    Mx = np.column_stack([SB.random_tie_marks(X[:, [j]], off, .2, keyX[:, [j]]) for j in range(X.shape[1])]).astype(np.int8)
    zfull = GW.group_matrix(X, off, groups, None, answer_standardize); within = [None] * G; kept = {}
    for h in range(G):
        mem = np.flatnonzero(groups == h)
        if len(mem) < 4: continue
        anchor = np.delete(zfull, h, axis=1).mean(1)
        Ma = SB.random_tie_marks(anchor[:, None], off, .2, keyA).astype(np.int8); best = None
        for tri in itertools.combinations(range(len(mem)), 3):
            Vv = violation4(np.column_stack([Mx[:, mem[list(tri)]], Ma]))
            if best is None or Vv < best[1]: best = (tri, Vv)
        w = np.zeros(len(mem)); w[list(best[0])] = 1.; within[h] = w; kept[h] = [bank['names'][surv[mem[q]]] for q in best[0]]
    sc['TCI_ONLY_ga'] = fuse_ga(within)
    diag = {'survivors': [bank['names'][i] for i in surv], 'n_channels': len(bank['names']), 'groups': groups.tolist(), 'anchor': bank['names'][surv[ai]],
            'prevalence_hat': float(st['diagnostics']['filter_estimator']['prevalence']), 'tci_kept': kept, 'lf_sign_flips': bank['diagnostics']['sign_flips'],
            'all_members_gate': gate}
    return sc, surv, diag


def decisions(score, bank, surv):
    flags, _, d = rules(score, bank, bank['raw'][:, surv], np.zeros(len(bank['answers'])), np.array([str(i) for i in bank['answers']]))
    return {'ANSWER_Z_Q80': flags['ANSWER_Z_Q80'], 'R2_COUNT': flags['R2_COUNT'], 'TOP_1': top_flags(score, bank['off'], np.ones(len(bank['answers']), int))}, d


def parity_gate():
    """Adapter + fitting on the development source must reproduce the frozen development scores and flags."""
    F = np.load(FE / 'source/FEATURES.npz'); ans = pd.read_csv(SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
    if list(map(str, F['uids'])) != ans.uid.astype(str).tolist(): stop('source order')
    pop = Population(F, BANKS); cells = sorted(set(ans.cell)); ce = 'prmbench_qwen3_8b'; ci = cells.index(ce)
    answers = np.flatnonzero(ans.cell.to_numpy() == ce); res = {}
    for bi, bk in enumerate(BANKS):
        t = time.perf_counter(); bank = pop.cell_bank(bk, answers)
        sc, surv, diag = fit_cell(pop, bank, 1000 * bi + ci)
        ref = {}
        for variant, tag in (('groups_answer', 'ga'), ('channels_cell', 'cc')):
            with np.load(PANEL / 'candidate_scores' / f'{ce}__{bk}__{variant}.npz') as z:
                for r in ('MEAN_MOM', 'HEM_MOM'):
                    ref[f'{r}_{tag}'] = z[f'{r}__score']; ref[f'{r}_{tag}__flags'] = {u: z[f'{r}__{u}__flags'] for u in ('ANSWER_Z_Q80', 'R2_COUNT')}
        with np.load(TCIR / 'scores' / f'{bk}__{ce}.npz') as z: ref['TCI_ONLY_ga'] = z['TCI_ONLY_ga']; ref['DS_FILTER_EQ'] = z['plain']
        out = {k: float(np.max(np.abs(sc[k] - ref[k]))) for k in ('MEAN_MOM_ga', 'HEM_MOM_ga', 'MEAN_MOM_cc', 'HEM_MOM_cc', 'TCI_ONLY_ga', 'DS_FILTER_EQ')}
        fl = {}
        for k in ('MEAN_MOM_ga', 'HEM_MOM_cc'):
            dd, _ = decisions(sc[k], bank, surv)
            for u in ('ANSWER_Z_Q80', 'R2_COUNT'): fl[f'{k}/{u}'] = int((dd[u] != ref[k + '__flags'][u]).sum())
        res[bk] = {'max_abs_score_diff': out, 'flag_mismatches': fl, 'seconds': round(time.perf_counter() - t, 1)}
        print('parity', bk, {k: f'{v:.1e}' for k, v in out.items()}, fl, flush=True)
    worst = max(v for r in res.values() for v in r['max_abs_score_diff'].values())
    flags_bad = sum(v for r in res.values() for v in r['flag_mismatches'].values())
    status = {'status': 'PASS' if worst <= 1e-6 and flags_bad == 0 else 'FAIL', 'tolerance': 1e-6, 'worst_score_diff': worst, 'flag_mismatches': flags_bad, 'banks': res}
    (OUT / 'SOURCE_PARITY_GATE.json').write_text(json.dumps(status, indent=1), encoding='utf8')
    if status['status'] != 'PASS': stop(f'source parity gate failed: worst {worst}, flags {flags_bad}')
    return status


def score_external():
    if (OUT / 'SCORING_STATUS.json').exists(): stop('already scored')
    P = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))
    if P['status'] != 'FROZEN': stop('protocol not frozen')
    status = {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'protocol_sha256': sha(OUT / 'PROTOCOL.json'), 'score_script_sha256': sha(Path(__file__)),
              'external_labels_opened': False, 'source_parity_gate_sha256': sha(OUT / 'SOURCE_PARITY_GATE.json'), 'cells': {}}
    for ci, (cell, bench) in enumerate(CELLS.items()):
        tc = time.perf_counter(); F = np.load(FE / cell / 'FEATURES.npz'); uids = [str(u) for u in F['uids']]
        lens = F['nonempty_lengths']; ne = F['nonempty'].astype(bool); starts = np.r_[0, np.cumsum(lens)]
        public = {r['uid']: r for r in json.loads((INPUTS / bench / 'answers.json').read_text(encoding='utf8'))}   # text/steps only, no labels
        if set(public) != set(uids): stop(cell + ': population differs')
        pop = Population(F, BANKS); off = pop.off
        for i, u in enumerate(uids):
            if len(public[u]['steps']) != lens[i] or int(ne[starts[i]:starts[i + 1]].sum()) != off[i + 1] - off[i]: stop(cell + ': step alignment ' + u)
        answers = np.arange(len(uids)); names_sc, names_pr, SC, PR, diag = [], [], [], [], {}
        for bi, bk in enumerate(BANKS):
            bank = pop.cell_bank(bk, answers)
            sc, surv, d = fit_cell(pop, bank, 1000 * bi + ci); diag[bk] = d
            for s in SCORERS:
                if not np.isfinite(sc[s]).all(): stop(f'{cell} {bk} {s} nonfinite')
                names_sc.append(f'{bk}__{s}'); SC.append(sc[s])
                dd, info = decisions(sc[s], bank, surv)
                for r in RULES: names_pr.append(f'{bk}__{s}__{r}'); PR.append(dd[r])
            print(f'  {cell} {bk}: survivors {len(surv)}/{len(BN[bk])} groups {max(d["groups"]) + 1} prevalence_hat {d["prevalence_hat"]:.3f}', flush=True)
        # step-index reference (later = riskier) over non-empty steps
        pos = (np.arange(pop.S) - np.repeat(off[:-1], np.diff(off))).astype(float)
        names_sc.append('STEP_INDEX'); SC.append(pos)
        az = np.zeros(pop.S)
        for a, b in zip(off[:-1], off[1:]): v = pos[a:b]; az[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
        names_pr.append('STEP_INDEX__ANSWER_Z_Q80'); PR.append(az >= np.quantile(az, .8))
        names_pr.append('STEP_INDEX__TOP_1'); PR.append(top_flags(pos, off, np.ones(len(uids), int)))
        # full-length arrays (empty steps: score NaN, prediction 'error')
        full_n = int(lens.sum()); idx = np.flatnonzero(ne)
        SCf = np.full((full_n, len(SC)), np.nan); SCf[idx] = np.column_stack(SC)
        PRf = np.zeros((full_n, len(PR)), np.int8); PRf[idx] = (~np.column_stack(PR)).astype(np.int8)   # 1 = predicted correct
        # earlier sealed references (continuity rows), copied by uid with telemetry identity check
        oldmap = {json.loads(p.read_text(encoding='utf8'))['uid']: p for p in (OLD / cell / 'shard_000').glob('*.record.json')}
        RS = np.full((full_n, len(REFS)), np.nan); RP = np.zeros((full_n, len(REFS)), np.int8)
        tele = [str(t) for t in F['telemetry_sha256']]
        for i, u in enumerate(uids):
            o = json.loads(oldmap[u].read_text(encoding='utf8'))
            if o['payload']['telemetry_sha256'] != tele[i] or o['payload']['nonempty'] != ne[starts[i]:starts[i + 1]].tolist(): stop(cell + ': reference identity ' + u)
            for k, r in enumerate(REFS):
                RS[starts[i]:starts[i + 1], k] = [np.nan if v is None else v for v in o['payload']['scores'][r]]
                RP[starts[i]:starts[i + 1], k] = o['payload']['predictions'][r]
        (OUT / cell).mkdir(exist_ok=True)
        np.savez_compressed(OUT / cell / 'PREDICTIONS.npz', uids=np.array(uids), starts=starts, nonempty=ne, scores=SCf, score_names=np.array(names_sc),
                            predictions=PRf, prediction_names=np.array(names_pr), ref_scores=RS, ref_predictions=RP, ref_names=np.array(REFS))
        (OUT / cell / 'SCORING_DIAGNOSTICS.json').write_text(json.dumps(diag, indent=1, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf8')
        status['cells'][cell] = {'answers': len(uids), 'steps': full_n, 'nonempty_steps': int(ne.sum()), 'scores': len(names_sc), 'predictions': len(names_pr),
                                 'predictions_sha256': sha(OUT / cell / 'PREDICTIONS.npz'), 'features_sha256': sha(FE / cell / 'FEATURES.npz'),
                                 'seconds': round(time.perf_counter() - tc, 1)}
        print(cell, 'scored', status['cells'][cell]['seconds'], 's', flush=True)
    status['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (OUT / 'SCORING_STATUS.json').write_text(json.dumps(status, indent=1), encoding='utf8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--gate-only', action='store_true'); a = ap.parse_args()
    gate = parity_gate()
    if not a.gate_only: score_external()
