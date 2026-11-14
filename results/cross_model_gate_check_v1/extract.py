"""Cross-model answer gate, go/no-go check, stage 1 (PC, CPU only, 2026-11-14).

For each benchmark and each standalone base model, write the label-free TELE16 step score WITHOUT per-answer z:
  1. raw16  = the 16 raw TELE16 channels of the standalone bank (same source and step alignment as the stored R1).
  2. G16    = global rank -> N(0,1) of each raw channel over the whole cell (digit channels: active steps only, else 0).
              This is the R1 construction WITHOUT the final per-answer z step.
  3. s      = L-SML fused score of the 16 G16 columns, anchor = channel mean (c4u_new.lsml; same convention as GSP-T).
Gate: answer_z(G16) must equal the stored R1 columns X[:, 2:] of the same cell (max abs diff < 1e-6).

Base model per PRM key (the standalone pairing of the project):
  qwen_prm_7b  -> Qwen2.5-Math-7B-Instruct      qwen_prm_72b -> Qwen2.5-Math-72B-Instruct
  pure_prm     -> Qwen2.5-Math-7B (base)        rlhflow_prm  -> Llama-3.1-8B-Instruct
The PRM score is NOT used. Labels are written only for the diagnostic of stage 2.

Output: steps/<bench>__<prm>.csv.gz  (uid, step, y, s)  and  EXTRACT_GATE.json.

    COLLECTION_DATA=C:/Users/omris/TAU/hd_collection_data [HD_ROOT=<checkout>] [OUT_DIR=<dir>] python extract.py [bench ...]
"""
import gzip, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('HD_ROOT', HERE.parents[1]))   # a checkout that holds results/c4u_telemetry_only_v1
sys.path.insert(0, str(ROOT / 'results/c4u_telemetry_only_v1'))
import run as Rn  # noqa: E402  (c4u_telemetry_only_v1: brings RA = run_all, T = c4u_competitors, E, N, C)
E, T, RA, N = Rn.E, Rn.T, Rn.RA, Rn.N
try:
    import bracha_panel as BP  # noqa: E402  (only for the Socratic / Hard2Verify R1 gate)
except ImportError:
    BP = None
OUT_DIR = Path(os.environ.get('OUT_DIR', HERE)); OUT = OUT_DIR / 'steps'; OUT.mkdir(parents=True, exist_ok=True)
_common = Rn.C.A.common_uids


def _common_uids(bench):
    """Copied from gsp_ablation_baseline_v1/ablation.py: ProcessBench = the 3,400 answers if crop_ares_v1/scores is absent."""
    d = Path(E.DATA) / 'crop_ares_v1' / 'scores'
    if bench == 'processbench' and not d.exists():
        S = [set(np.load(f)['uids'].astype(str)) for f in sorted((Path(E.DATA) / 'sub5_all_v1' / 'scores').glob('SCORES_prm_*_processbench.npz'))]
        u = set.intersection(*S); assert len(S) == 4 and len(u) == 3400
        return u
    return _common(bench)


Rn.C.A.common_uids = _common_uids


def load_cell(bench, prm):
    """gsp_ablation_baseline_v1/ablation.load_cell (copied, PRM columns dropped): cell c and the stored R1 telemetry
    columns (None for Socratic / Hard2Verify when bracha_panel is not in HD_ROOT)."""
    if bench == 'prmbench':
        RA.MODE = 'sa'; RA.need(prm, 'prmbench')
        c = E.load('prmbench', f'{prm}__standalone')
        c.arms['PRMPROB'] = 1. - c.extra['reward']; c.arms['PRM'] = c.arms[N.K.prm_arm(c)]
        X, ab, rep, g = T.build_X(c, prm, 'prmbench'); assert rep == 'R1'
        return c, X[:, 2:]
    if bench == 'processbench':
        RA.MODE = 'sa'; RA.need(prm, 'processbench')
        c = Rn.C.cab_cell('processbench', prm)
        X, ab, rep, g = T.build_X(c, prm, 'processbench'); assert rep == 'R1'
        return c, X[:, 2:]
    c = E.load(bench, prm)
    return c, (BP.sa_X(c, prm, bench) if BP is not None else None)
BENCHES = ('prmbench', 'processbench', 'socratic', 'hard2verify', 'crv_arith', 'crv_gsm8k')


def g16_sa_external(c, prm, bench):
    """bracha_panel.sa_X without the last answer-z line (copied; same alignment)."""
    p = RA.ALL / f'full_sa_{T.PFILE.get(prm, prm)}_{bench}' / 'bank_v3' / 'FEATURES.npz'
    F = np.load(p); names = list(F['names'].astype(str)); V = F['values']; act = F['digit_active'].astype(bool)
    S = np.load(Path(E.DATA) / {'socratic': 'socratic', 'hard2verify': 'h2v'}[bench] / f'SCORES_{T.PFILE.get(prm, prm)}.npz')
    rows = []
    for lay in (F, S):
        ne = lay['nonempty'].astype(bool); L = lay['nonempty_lengths'].astype(np.int64)
        st = np.concatenate([[0], np.cumsum(L)]); rid = np.cumsum(ne) - 1
        pos = {u: j for j, u in enumerate(lay['uids'].astype(str))}
        rows.append((ne, st, rid, pos))
    (bne, bst, brid, bpos), (pne, pst, _, ppos) = rows
    rows = []
    for u in c.ids:
        a, b = bpos[u], ppos[u]; assert bst[a + 1] - bst[a] == pst[b + 1] - pst[b], u
        for k in range(pst[b + 1] - pst[b]):
            if not pne[pst[b] + k]: continue
            q = bst[a] + k; assert bne[q], (u, k); rows.append(brid[q])
    rows = np.array(rows); assert len(rows) == c.off[-1]
    rv = np.column_stack([V[rows, names.index(n)] for n in T.N16]).astype(float); act = act[rows]
    return np.column_stack([T.rank_gauss(rv[:, j], act[:, T.DIG.index(n)] if n in T.DIG else None)
                            for j, n in enumerate(T.N16)])


def g16_bank(c, prm, bench):
    rv, act = T.raw16(c, prm, bench)
    assert rv is not None, (prm, bench)
    return np.column_stack([T.rank_gauss(rv[:, j], act[:, T.DIG.index(n)] if n in T.DIG else None)
                            for j, n in enumerate(T.N16)])


def load(bench, prm):
    """-> cell c, G16 (no answer-z), gate value (max |answer_z(G16) - stored R1|, None if no stored R1 here)."""
    if bench in ('prmbench', 'processbench', 'socratic', 'hard2verify'):
        c, Xr = load_cell(bench, prm)
        G = g16_sa_external(c, prm, bench) if bench in ('socratic', 'hard2verify') else g16_bank(c, prm, bench)
        if Xr is None:
            return c, G, None
        Z = np.column_stack([E.answer_z(G[:, j], c.off) for j in range(16)])
        return c, G, float(np.abs(Z - Xr).max())
    RA.MODE = 'sa'; Rn.C.MODE = 'sa'; Rn.C.CELL_HOOK = None
    RA.need(prm, bench)
    c = Rn.C.cab_cell(bench, prm)
    return c, g16_bank(c, prm, bench), None


def main(benches):
    gate = {}
    for bench in benches:
        for prm in N.PRMS:
            try:
                c, G, g = load(bench, prm)
            except (SystemExit, FileNotFoundError, KeyError, AssertionError) as ex:
                gate[f'{bench}/{prm}'] = dict(status='SKIP', reason=repr(ex)[:300]); print('skip', bench, prm, ex, flush=True)
                continue
            if g is not None and g >= 1e-6:
                gate[f'{bench}/{prm}'] = dict(status='FAIL', max_abs_diff=g); print('GATE FAIL', bench, prm, g, flush=True)
                continue
            s = np.asarray(N.lsml(G, G.mean(1))[0], float)
            err = np.asarray(c.err, bool)
            ids = np.asarray(c.ids).astype(str); off = np.asarray(c.off, np.int64)
            with gzip.open(OUT / f'{bench}__{prm}.csv.gz', 'wt', encoding='utf8') as f:
                f.write('uid,step,y,s\n')
                for u, a, b in zip(ids, off[:-1], off[1:]):
                    for k in range(b - a):
                        f.write(f'{u},{k},{int(err[a + k])},{s[a + k]:.6g}\n')
            gate[f'{bench}/{prm}'] = dict(status='PASS' if g is not None else 'NO_R1_GATE', max_abs_diff=g,
                                          answers=len(ids), steps=int(off[-1]),
                                          answer_error_share=float(np.mean([err[a:b].any() for a, b in zip(off[:-1], off[1:])])))
            print(bench, prm, gate[f'{bench}/{prm}'], flush=True)
            del c, G
    p = OUT_DIR / 'EXTRACT_GATE.json'
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(gate); p.write_text(json.dumps(old, indent=1))


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or BENCHES)
