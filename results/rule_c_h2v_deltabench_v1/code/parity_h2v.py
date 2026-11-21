"""Parity gate for the new H2V C step table (CPU). Method of record = hard top10 readout + 5 step channels (16),
R0 per-answer z, frozen L-SML (ssl_lib.lsml_linear), GAP. Compared with the stored TELE_N16 standalone score
(hd_collection_data/h2v/SCORES_qwen_prm.npz) and the stored official H2V step Balanced F1 (GAP) 47.2247 on the
177-answer common set (results/prm_sub5_all_benchmarks_v1/SUB5_ALL.csv). Labels: evaluation only (last block).
    HD_ROOT=$WS/hd COLLECTION_DATA=<hd_collection_data> python parity_h2v.py <STEPS_hard2verify_C.npz> <GROUPS json> <out json>"""
import json, os, sys
from pathlib import Path
import numpy as np
HD = Path(os.environ['HD_ROOT'])
sys.path.insert(0, str(HD / 'results/ssl_one_experiment_v1')); sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(HD / 'results/prm_boost_collection_v1'))
import ssl_data as D, ssl_lib as L, bench_adapters as A  # noqa
import evaluate_collection as E  # noqa
from collection_parity import h2v_common_uids  # noqa
from scipy.stats import spearmanr

steps, gpath, outp = sys.argv[1:4]
P = A.load_c_pool(steps, 'hard2verify', groups=json.load(open(gpath)))
raw = D.readout(P); X = L.r0_cols(raw, P.act, P.off); s, W, meta = L.lsml_linear(X); f = L.gap_flags(s, P.off)
S = np.load(Path(os.environ['COLLECTION_DATA']) / 'h2v/SCORES_qwen_prm.npz')
assert (S['uids'].astype(str) == P.uids).all() and S['nonempty'].all()
ref = S['score__TELE_N16'].astype(float)
out = dict(n_answers=int(P.n), n_steps=int(P.off[-1]), lsml=meta,
           corr_vs_stored_TELE_N16=float(np.corrcoef(s, ref)[0, 1]), spearman=float(spearmanr(s, ref)[0]),
           max_abs_diff=float(np.abs(s - ref).max()),
           gap_flags_equal_stored=bool((L.gap_flags(ref, P.off) == f).all()))
# ---- evaluation (labels) on the 177 common answers, official evaluate_collection path
common = h2v_common_uids()
c = E.load_external('hard2verify', 'qwen_prm_7b', uids_keep=common)
pos = {u: j for j, u in enumerate(P.uids)}
rows = np.concatenate([np.arange(P.off[pos[u]], P.off[pos[u] + 1]) for u in c.ids])
assert len(rows) == c.off[-1]
for name, sc in (('new_table_MoR', s[rows]), ('stored_TELE_N16', ref[rows])):
    fl = E.topk_flags(E.answer_z(sc, c.off), c.off, E.gap_counts(E.answer_z(sc, c.off), c.off))
    out[f'balanced_f1_GAP_common177_{name}'] = 100 * E.official(c, fl)[0]
out['reference_SUB5_ALL'] = 47.22467768355034
out['gate_abs_diff'] = abs(out['balanced_f1_GAP_common177_new_table_MoR'] - out['reference_SUB5_ALL'])
out['gate'] = 'PASS' if out['gate_abs_diff'] <= 0.1 else 'FAIL'
json.dump(out, open(outp, 'w'), indent=1); print(json.dumps(out, indent=1))
