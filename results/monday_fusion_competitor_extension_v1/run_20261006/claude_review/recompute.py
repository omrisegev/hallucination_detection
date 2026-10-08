"""Independent recomputation of F11/F12 per-answer records (Claude review, 2026-10-06).

Only the TCI decision flags come from the frozen rule code (they are not stored);
every metric, aggregation and stratum below is reimplemented here.
"""
import importlib.util, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT = ROOT / 'results/monday_fusion_competitor_extension_v1/run_20261006'
SCR = Path(__file__).parent
SE = ROOT / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
PANEL = ROOT / 'results/monday_candidate_panel_v1/run_20261004'
TCI = ROOT / 'results/mom_filter_within_diagnostic_20261004/tci_wide_20261005'
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
RULES = ['ANSWER_Z_Q80', 'R2_COUNT', 'ENTROPY_GATE_Q80']
ACTIONS = RULES + ['TOP_1', 'TOP_2', 'TOP_3']
METHODS = {'HEM_MOM_cc': ('channels_cell', 'HEM_MOM'), 'MEAN_MOM_cc': ('channels_cell', 'MEAN_MOM'),
           'MEAN_MOM_ga': ('groups_answer', 'MEAN_MOM'), 'TCI_ONLY_ga': ('tci', 'TCI_ONLY_ga')}
PRM = 'prmbench_qwen3_8b'
t0 = time.time()

# ---- independent population read
A = pd.read_csv(SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
with np.load(SE / 'OOF_STEP_SCORES.npz') as z:
    OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
NS = np.diff(OFF)
freeze = json.loads((SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
meta = {m['idx']: m for m in pickle.load(open(freeze['prm_metadata']['path'], 'rb')).values()}
ids = A.id.astype(str).to_numpy(); cells_all = A.cell.to_numpy(); target = A.target.to_numpy().astype(int)
groups = A.source_group.to_numpy()
with np.load(freeze['mindgap']['path']) as z:
    MTG = z['step_m5'].astype(float)
assert MTG.shape == (OFF[-1],)

# ---- frozen rule code only for TCI flags (and replay of stored panel flags)
spec = importlib.util.spec_from_file_location('codex_run', OUT / 'run.py'); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
pop, data, evidence, rules, *_ = run.load_runtime()
assert np.array_equal(pop.off, OFF) and np.array_equal(pop.ids, ids) and np.array_equal(pop.target, target)
assert np.array_equal(np.asarray(data.mtg, float), MTG), 'MTG array differs from EVIDENCE_DROP step_m5'
print('loaded', round(time.time() - t0, 1), 's', flush=True)


def my_top(score, off, k):
    out = np.zeros(len(score), bool)
    for a, b in zip(off[:-1], off[1:]):
        idx = np.lexsort((np.arange(b - a), -score[a:b]))[:k]   # descending score, earlier step on ties
        out[a + idx] = True
    return out


def first_true(flags, off):
    n = len(off) - 1; loc = np.arange(len(flags)) - np.repeat(off[:-1], np.diff(off))
    big = 10 ** 9; v = np.where(flags, loc, big)
    f = np.minimum.reduceat(v, off[:-1]); f[f == big] = -1
    return f


def first_argmax(score, off):
    loc = np.arange(len(score)) - np.repeat(off[:-1], np.diff(off))
    mx = np.maximum.reduceat(score, off[:-1])
    return np.minimum.reduceat(np.where(score == np.repeat(mx, np.diff(off)), loc, 10 ** 9), off[:-1])


def auc_one(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


cells = sorted(set(cells_all))
rec = {}            # (bank, cell, method, action) -> dict of per-answer arrays
argmax_tol_diff = 0
for bk in BANKS:
    for cell in cells:
        ans = np.flatnonzero(cells_all == cell)
        rows = np.concatenate([np.arange(OFF[i], OFF[i + 1]) for i in ans]); off = np.r_[0, np.cumsum(NS[ans])]
        is_prm = cell == PRM
        with np.load(TCI / 'scores' / f'{bk}__{cell}.npz') as z:
            assert np.array_equal(z['answers'], ans) and np.array_equal(z['rows'], rows)
            surv = z['surv'].astype(int); tci_score = z['TCI_ONLY_ga'].astype(float)
            tci_copy = {m: z[m].astype(float) for m in ('HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga')}
        record = json.loads((run.INPUT_RUN / 'chunks' / f'{cell}__{bk}__groups_answer.diagnostics.json').read_text(encoding='utf8'))
        fb = data.bank(bk, cell, record); raw_surv = fb['raw'][:, surv]
        y = LAB[rows] if is_prm else None
        for method, (variant, recipe) in METHODS.items():
            if variant == 'tci':
                score = tci_score; fl, _, _ = rules(score, fb, raw_surv, data.entropy[ans], pop.uids[ans])
            else:
                with np.load(PANEL / 'candidate_scores' / f'{cell}__{bk}__{variant}.npz') as z:
                    assert np.array_equal(z['answer_indices'], ans) and np.array_equal(z['offsets'], off)
                    score = z[f'{recipe}__score'].astype(float)
                    fl = {r: z[f'{recipe}__{r}__flags'].astype(bool) for r in RULES}
                # cross-file consistency: TCI run's copy of the same recipe
                rec[('_copydiff', bk, cell, method)] = float(np.max(np.abs(tci_copy[method] - score)))
            am = first_argmax(score, off)
            # tolerance-based argmax used by Codex's evaluate
            for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
                s = score[a:b]
                if int(np.flatnonzero(s >= s.max() - 8 * np.finfo(float).eps)[0]) != am[j]: argmax_tol_diff += 1
            auc = np.array([auc_one(y[a:b], score[a:b]) for a, b in zip(off[:-1], off[1:])]) if is_prm else None
            for act in ACTIONS:
                f = fl[act] if act in RULES else my_top(score, off, int(act[-1]))
                d = {'ans': ans, 'first': first_true(f, off), 'argmax': am,
                     'nflag': np.add.reduceat(f.astype(int), off[:-1])}
                if is_prm:
                    valid = ~f; gold = ~y; nc = np.repeat(np.array([meta[ids[i]]['classification'] != 'correct' for i in ans]), NS[ans])
                    conf = np.stack([np.add.reduceat((valid & gold & nc).astype(int), off[:-1]),
                                     np.add.reduceat((valid & ~gold & nc).astype(int), off[:-1]),
                                     np.add.reduceat((~valid & ~gold & nc).astype(int), off[:-1]),
                                     np.add.reduceat((~valid & gold & nc).astype(int), off[:-1])], 1)
                    prmflag = np.concatenate([np.asarray(meta[ids[i]]['labels']) == 0 for i in ans])
                    assert len(prmflag) == len(y)
                    ov = np.stack([np.add.reduceat((y & f & prmflag).astype(int), off[:-1]),
                                   np.add.reduceat((y & f & ~prmflag).astype(int), off[:-1]),
                                   np.add.reduceat((y & ~f & prmflag).astype(int), off[:-1]),
                                   np.add.reduceat((y & ~f & ~prmflag).astype(int), off[:-1])], 1)
                    d.update(auc=auc, conf=conf, ov=ov)
                rec[(bk, cell, method, act)] = d
    print(bk, 'done', round(time.time() - t0, 1), 's', flush=True)

# MTG per PB answer (exact first argmax)
pb_ans = np.flatnonzero(cells_all != PRM)
mtg_arg = np.array([int(np.argmax(MTG[OFF[i]:OFF[i + 1]])) for i in pb_ans])
pickle.dump({'rec': rec, 'mtg_arg': mtg_arg, 'pb_ans': pb_ans, 'argmax_tol_diff': argmax_tol_diff,
             'target': target, 'NS': NS, 'groups': groups, 'ids': ids, 'cells': cells_all,
             'category': {i: meta[ids[i]].get('category') for i in np.flatnonzero(cells_all == PRM)},
             'classification': {i: meta[ids[i]].get('classification') for i in np.flatnonzero(cells_all == PRM)},
             'n_errors': {i: int(LAB[OFF[i]:OFF[i + 1]].sum()) for i in np.flatnonzero(cells_all == PRM)},
             'meta_error_steps_ok': all(np.array_equal(LAB[OFF[i]:OFF[i + 1]], [j + 1 in set(meta[ids[i]]['error_steps']) for j in range(NS[i])]) for i in np.flatnonzero(cells_all == PRM))},
            open(SCR / 'records.pkl', 'wb'))
print('saved; argmax tolerance disagreements:', argmax_tol_diff, 'total', round(time.time() - t0, 1), 's')
