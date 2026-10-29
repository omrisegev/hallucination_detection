"""Check F (PROTOCOL.json): CROP-style token/format counts per step as a possible extra DS group.

Counts from the step text, log1p, length- and position-cleaned (pooled, label-free), oriented by the sign of their pooled
correlation with the plain average of the stored answer-z N16 channels. Labels only for evaluation (R1-R4).

    python format_features.py          # -> R1_AUC.csv, R2_CONDCORR.csv, R3_CLUSTER.csv, R4_PRMBENCH_CLASS.csv
    from format_features import format_block; s = format_block('processbench', 'qwen_prm_7b')
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
import ast, json, re, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'bracha_lr_transfer_v1'))
import lr_transfer as L  # noqa: E402
E = L.E
from spectral_utils import prm_anchored_lsml as A  # noqa: E402

DATA = Path(r'C:\Users\omris\TAU\hd_collection_data')
BENCHES = ['prmbench', 'processbench', 'crv_arith', 'crv_gsm8k']
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
LENGTH_COUNTS = ['char_len', 'n_words']
COUNTS = LENGTH_COUNTS + ['n_digits', 'n_letters', 'n_eq', 'n_ops', 'n_newlines', 'n_colons', 'n_parens', 'n_angle']


# ------------------------------------------------------------------ text
@lru_cache(None)
def step_texts(bench):
    """uid -> list of step strings."""
    if bench == 'prmbench':
        sys.path.insert(0, r'C:\Users\omris\TAU\hallucination_detection')
        from spectral_utils.prmbench import load_prmbench
        meta, _ = load_prmbench()
        return {m['idx']: list(m['steps']) for m in meta}
    if bench == 'processbench':
        from datasets import load_dataset
        ds = load_dataset('Qwen/ProcessBench')
        out = {}
        for split in ds:
            for r in ds[split]:
                st = r['steps']
                out[r['id']] = list(st) if not isinstance(st, str) else ast.literal_eval(st)
        return out
    rows = json.load(open(DATA / 'crop_ares_v1/inputs_text' / bench / 'answers.json', encoding='utf8'))
    return {r['uid']: list(r['steps']) for r in rows}


def counts_of(t):
    return [len(t), len(t.split()), sum(ch.isdigit() for ch in t), sum(ch.isalpha() for ch in t), t.count('='),
            sum(t.count(o) for o in '+-*/^'), t.count('\n'), t.count(':'), t.count('(') + t.count(')'), t.count('<') + t.count('>')]


def raw_counts(bench, c):
    T = step_texts(bench); n = np.diff(c.off)
    rows = []
    for u, k in zip(c.ids, n):
        st = T[u]
        assert len(st) == k, f'{bench} {u}: {len(st)} text steps vs {k} cell steps'
        rows += [counts_of(s) for s in st]
    return np.log1p(np.asarray(rows, float))


# ------------------------------------------------------------------ cleaning
def rel_pos(off):
    return np.concatenate([(np.arange(k) + .5) / k for k in np.diff(off)])


def residualize(Y, B, pool):
    beta = np.linalg.lstsq(B[pool], Y[pool], rcond=None)[0]
    return Y - B @ beta


def clean_counts(R, off):
    n = np.diff(off); pool = np.repeat(n, n) > 1; p = rel_pos(off)
    P = np.column_stack([np.ones_like(p), p, p ** 2, p ** 3])
    out = np.empty_like(R)
    out[:, :2] = residualize(R[:, :2], P, pool)                                   # length counts: position only
    out[:, 2:] = residualize(R[:, 2:], np.column_stack([P, R[:, 0]]), pool)        # + log1p(char length)
    sd = out.std(0); return (out - out.mean(0)) / np.where(sd > 1e-12, sd, 1.)


def position_clean_az(Z, off):
    """Pooled cubic in relative position on answer-z columns, then answer-z again (Step 481)."""
    n = np.diff(off); pool = np.repeat(n, n) > 1; p = rel_pos(off)
    P = np.column_stack([np.ones_like(p), p, p ** 2, p ** 3])
    Zr = residualize(Z, P, pool)
    return np.column_stack([E.answer_z(Zr[:, k], off) for k in range(Zr.shape[1])])


def load_cell(bench, prm):
    c = L.load(bench, prm, 'prm')
    V = E.channels(c, 'N16')
    prm_risk = np.asarray(c.arms['PRM_Q80' if bench == 'prmbench' else 'PRM'], float)
    return c, V, prm_risk


def oriented_counts(bench, c, V):
    C = clean_counts(raw_counts(bench, c), c.off)
    mean16 = V.mean(1)
    sign = np.sign(np.nan_to_num([np.corrcoef(C[:, k], mean16)[0, 1] if C[:, k].std() > 1e-12 else 0. for k in range(C.shape[1])]))
    sign[sign == 0] = 1                                               # a constant count (e.g. no '<>' in a benchmark) adds 0 (amendment A1)
    return C * sign, sign


def format_block(bench, prm):
    """Cleaned, label-free oriented per-step format score (plain average of the 10 counts), cell order of lr_transfer.load."""
    c, V, _ = load_cell(bench, prm)
    C, _ = oriented_counts(bench, c, V)
    return C.mean(1)


# ------------------------------------------------------------------ evaluation (labels from here on)
def eval_keep(bench, c):
    n = np.diff(c.off)
    if bench == 'processbench': return c.keep_gold.astype(bool)
    k = np.asarray(c.keep, bool)
    if bench == 'prmbench': k = k & np.repeat(c.cls != 'correct', n)
    return k


def wauc(c, s, keep):
    return float(np.nanmean(E.per_answer_auc(c.err, s, c.off, keep)))


def gap_votes(Z, off):
    return np.column_stack([E.topk_flags(Z[:, k], off, E.gap_counts(Z[:, k], off)) for k in range(Z.shape[1])]).astype(float)


def run_cell(bench, prm):
    c, V, prm_risk = load_cell(bench, prm)
    C, sign = oriented_counts(bench, c, V)
    fb = C.mean(1); off = c.off; keep = eval_keep(bench, c); y = c.err.astype(bool)
    r1 = dict(benchmark=bench, prm=prm, **{f'auc_{nm}': wauc(c, C[:, k], keep) for k, nm in enumerate(COUNTS)},
              auc_format_block=wauc(c, fb, keep), auc_mean16=wauc(c, V.mean(1), keep),
              signs=' '.join(f'{nm}:{int(s):+d}' for nm, s in zip(COUNTS, sign)))
    # position-cleaned answer-z: PRM, 16 channels, format block
    M = position_clean_az(np.column_stack([E.answer_z(prm_risk, off), V, E.answer_z(fb, off)]), off)
    zp, Zc, zf = M[:, 0], M[:, 1:17], M[:, 17]
    n = np.diff(off); pool = np.repeat(n, n) > 1
    K16, g16 = A.groups_from_cov(np.cov(Zc[pool].T))
    r2 = []
    for g in range(K16):
        gs = Zc[:, g16 == g].mean(1)
        for cls_name, m in (('error_steps', keep & y), ('correct_steps', keep & ~y)):
            r2.append(dict(benchmark=bench, prm=prm, partner=f'group{g}', channels=int((g16 == g).sum()), class_=cls_name,
                           abs_r=abs(float(np.corrcoef(zf[m], gs[m])[0, 1]))))
    for cls_name, m in (('error_steps', keep & y), ('correct_steps', keep & ~y)):
        r2.append(dict(benchmark=bench, prm=prm, partner='PRM', channels=1, class_=cls_name, abs_r=abs(float(np.corrcoef(zf[m], zp[m])[0, 1]))))
    Kc, gc = A.groups_from_cov(np.cov(M[pool].T))
    S = np.where(gap_votes(M, off) > 0, 1., -1.)
    Kv, gv = A.groups_from_cov(np.cov(S[pool].T))
    r3 = dict(benchmark=bench, prm=prm, K_continuous=int(Kc), format_singleton_continuous=bool((gc == gc[17]).sum() == 1),
              format_group_continuous=' '.join(str(j) for j in np.flatnonzero(gc == gc[17])),
              K_votes=int(Kv), format_singleton_votes=bool((gv == gv[17]).sum() == 1),
              format_group_votes=' '.join(str(j) for j in np.flatnonzero(gv == gv[17])))
    r4 = []
    if bench == 'prmbench':
        cls_step = np.repeat(c.cls, n)
        for cl in sorted(set(c.cls) - {'correct'}):
            kk = keep & (cls_step == cl)
            r4.append(dict(prm=prm, classification=cl, n_answers=int((c.cls == cl).sum()), auc_format_block=wauc(c, fb, kk),
                           auc_digits=wauc(c, C[:, 2], kk), auc_char_len=wauc(c, C[:, 0], kk), auc_mean16=wauc(c, V.mean(1), kk)))
    return r1, r2, r3, r4


def main():
    R1, R2, R3, R4 = [], [], [], []
    for bench in BENCHES:
        for prm in PRMS:
            r1, r2, r3, r4 = run_cell(bench, prm); R1.append(r1); R2 += r2; R3.append(r3); R4 += r4
            mx = max(r['abs_r'] for r in r2)
            print(f"{bench:12s} {prm:13s} format AUC {r1['auc_format_block']:.3f} (MEAN16 {r1['auc_mean16']:.3f}) "
                  f"max cond|r| {mx:.3f} singleton cont/votes {r3['format_singleton_continuous']}/{r3['format_singleton_votes']}", flush=True)
    for nm, rows in (('R1_AUC', R1), ('R2_CONDCORR', R2), ('R3_CLUSTER', R3), ('R4_PRMBENCH_CLASS', R4)):
        E.write_csv(HERE / f'{nm}.csv', rows)


if __name__ == '__main__':
    main()
