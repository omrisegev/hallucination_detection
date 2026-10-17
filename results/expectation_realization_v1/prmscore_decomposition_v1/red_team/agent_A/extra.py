exec(open('recompute.py').read().split('# --- within-answer AUROC')[0].replace('print(f"SHA256','pass #print(f"SHA256'))
import collections
# folds vs source groups
g2f = collections.defaultdict(set)
for i in P: g2f[rows[i]["source_group"]].add(rows[i]["fold"])
print("source groups spanning >1 fold:", sum(len(v) > 1 for v in g2f.values()), "of", len(g2f))
# ties at tau for PRM_raw_q80 (strict < vs <=)
t = q80_taus(prm_risk); tau = np.array([t[f] for f in step_fold])
print("PRMBench steps with risk exactly == tau:", int((step_is_P & (prm_risk == tau)).sum()), "of", int(step_is_P.sum()))
inv2 = ~(prm_risk <= tau)
preds = [{"idx": rows[i]["id"], "labels": [0 if x else 1 for x in inv2[off[i]:off[i + 1]]]} for i in P]
o = pbm.prmbench_evaluate(preds, meta)["total"]; print("PRM_raw_q80 with <= instead of <:", round(0.5*(o["f1"]+o["negative_f1"]),4))
# argmax tie sensitivity for PRM (latest vs earliest max)
E = [i for i in P if lab[off[i]:off[i + 1]].any()]
ntie = sum(int((prm_risk[off[i]:off[i+1]] == prm_risk[off[i]:off[i+1]].max()).sum() > 1) for i in E)
late = np.array([lab[off[i] + (off[i+1]-off[i]-1) - int(np.argmax(prm_risk[off[i]:off[i+1]][::-1]))] for i in E])
early = np.array([lab[off[i] + int(np.argmax(prm_risk[off[i]:off[i+1]]))] for i in E])
print(f"PRM answers with tied max: {ntie}/{len(E)}; hit earliest={early.mean():.4f} latest={late.mean():.4f}")
# bootstrap Monte-Carlo spread over seeds
NC = [i for i in P if cls[i] != "correct"]
def counts(inv):
    M = np.zeros((len(rows), 4))
    for i in NC:
        v = ~inv[off[i]:off[i + 1]]; e = lab[off[i]:off[i + 1]]
        M[i] = [(v & ~e).sum(), (v & e).sum(), (~v & e).sum(), (~v & ~e).sum()]
    return M
def ps(c):
    tp, fp, tn, fn = c; return 0.5 * (2*tp/(2*tp+fp+fn) + 2*tn/(2*tn+fn+fp))
CA, CB = counts(invalid["S_equal"]), counts(invalid["realized_drv"])
groups = sorted({rows[i]["source_group"] for i in NC}); gi = {g: j for j, g in enumerate(groups)}
GA = np.zeros((len(groups), 4)); GB = np.zeros((len(groups), 4))
for i in NC: GA[gi[rows[i]["source_group"]]] += CA[i]; GB[gi[rows[i]["source_group"]]] += CB[i]
for seed in (1, 2, 3, 4, 5):
    rng = np.random.default_rng(seed); d = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)); d.append(ps(w@GA)-ps(w@GB))
    d = np.array(d); print(f"seed {seed}: 95% [{np.quantile(d,.025):+.4f}, {np.quantile(d,.975):+.4f}]  P(d>0)={np.mean(d>0):.3f}")
# ddof sensitivity: std ddof=1 for S_equal answer-z (thresholds recomputed by same rule)
