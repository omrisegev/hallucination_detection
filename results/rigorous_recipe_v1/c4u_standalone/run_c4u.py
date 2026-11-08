"""Run the frozen C4.U on the 8 cells and check it against expected_results.json.
python run_c4u.py            -> prints per cell and PASS/FAIL (exit 1 on any mismatch)"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c4u_new as N  # noqa: E402

E, S8 = N.E, N.S8
EXP = json.loads((HERE / 'expected_results.json').read_text())['cells']


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a); h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


bad = 0
for bench, rep in (('prmbench', 'R1'), ('socratic', 'R0')):
    for prm in N.PRMS:
        key = f'{bench}|{prm}|{rep}'; x = EXP[key]
        c, ans_eval, X, ab, wk = N.load(bench, prm, rep)
        r = N.c4u(c, X, ab)
        wa = E.per_answer_auc(c.err, r['rank'], c.off, wk); wa[~ans_eval] = np.nan
        pt = S8.point(c, (wa, E.counts_by_answer(c, r['flags']) * ans_eval[:, None]))
        got = dict(input=sha(X, ab[1], c.off, ans_eval, np.zeros(0) if wk is None else wk),
                   flags=sha(r['flags'].astype(np.uint8)), rank_r9=sha(np.round(r['rank'], 9)))
        ok = (got['input'] == x['input_sha256'] and got['flags'] == x['flags_sha256'] and got['rank_r9'] == x['rank_sha256_r9']
              and abs(pt['gap_prmscore'] - x['prmscore']) < 1e-9 and abs(pt['within_auc'] - x['within_auc']) < 1e-9)
        bad += not ok
        print(f"{key}: PRMScore {pt['gap_prmscore']:.4f} within-AUC {pt['within_auc']:.4f} "
              f"rank bitwise {'yes' if sha(r['rank']) == x['rank_sha256_exact'] else 'no'}  {'PASS' if ok else 'FAIL'}", flush=True)
print('ALL PASS' if not bad else f'{bad} FAIL'); sys.exit(1 if bad else 0)
