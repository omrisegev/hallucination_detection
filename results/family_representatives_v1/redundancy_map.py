"""Label-free redundancy map of the corrected N56 bank (PRMBench, thinkfix mask1).

Reads only the answer-z-scored channel matrix V; no labels are loaded.
Output: REDUNDANCY.json (copies, near-copies, per-family spectra, clusters).
"""
import ast, json, sys
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / 'results/advisor_completion_2026-10-11/inputs/bank_V_N56.npz'
OUT = Path(__file__).with_name('REDUNDANCY.json')

def families():
    src = (ROOT / 'scripts/run_advisor_v11_lr.py').read_text()
    return next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign)
                and any(getattr(t, 'id', None) == 'FAMILIES' for t in n.targets))

z = np.load(BANK); V = z['V']; cols = [str(c) for c in z['columns']]
fam = families(); fam_of = {c: f for f, cs in fam.items() for c in cs}
assert set(fam_of) == set(cols), set(cols) ^ set(fam_of)
R = np.corrcoef(V, rowvar=False)
p = len(cols)

pairs = sorted(((abs(R[i, j]), cols[i], cols[j], R[i, j]) for i in range(p) for j in range(i + 1, p)), reverse=True)
copies = [dict(a=a, b=b, r=round(r, 5)) for s, a, b, r in pairs if s >= .999]
near = [dict(a=a, b=b, r=round(r, 4), fa=fam_of[a], fb=fam_of[b]) for s, a, b, r in pairs if .9 <= s < .999]

def spectrum(idx):
    w = np.linalg.eigvalsh(R[np.ix_(idx, idx)])[::-1]
    return dict(n=len(idx), top=[round(x, 3) for x in w[:4]], pc1_share=round(w[0] / w.sum(), 3),
                eff_rank=round(float(np.exp(-(w / w.sum() * np.log(np.clip(w / w.sum(), 1e-12, None))).sum())), 2))
fam_spec = {f: spectrum([cols.index(c) for c in cs]) for f, cs in fam.items()}

D = 1 - np.abs(R); np.fill_diagonal(D, 0)
L = linkage(squareform(D, checks=False), 'average')
clusters = {}
for thr in (.1, .2, .3, .5):
    lab = fcluster(L, t=thr, criterion='distance')
    groups = [sorted((cols[i] for i in np.flatnonzero(lab == k)), key=cols.index) for k in np.unique(lab)]
    clusters[f'|r|>={1-thr:.1f}'] = dict(n_clusters=len(groups), multi=[dict(members=g, families=sorted({fam_of[c] for c in g})) for g in groups if len(g) > 1])

# cross-family mean |r|
F = list(fam)
cross = {a: {b: round(float(np.mean([abs(R[cols.index(x), cols.index(y)]) for x in fam[a] for y in fam[b] if x != y])), 3) for b in F} for a in F}
res = dict(bank=str(BANK.relative_to(ROOT)), steps=int(V.shape[0]), channels=p, labels_read=False,
           exact_copies=copies, near_copies_ge_0_9=near, family_spectra=fam_spec, clusters=clusters, mean_abs_r_family=cross,
           whole_bank=spectrum(list(range(p))))
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(dict(copies=copies, n_near=len(near), whole=res['whole_bank'], fam=fam_spec,
                      n_clusters={k: v['n_clusters'] for k, v in clusters.items()}), indent=1))
