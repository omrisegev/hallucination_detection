"""SLIDER bridge: their cosine-similarity baseline (App. D.2, Eq. 6) on our redundancy-class split.

CosSim(S_i, S_<i) = e_Si . e_S<i, all-MiniLM-L6-v2 embeddings (mean pooling, L2-normalized, 256-token truncation,
as sentence-transformers does for this model). Higher = more redundant. Same protocol as
prm_boost_collection_v1/collection_competitors.py:slider (10 trace-level 50/50 held-out/test splits, seeds 0..9,
1-D logistic-regression threshold, unweighted and class-balanced). Step 1 has no prefix: it is excluded in the
steps_ge2 scope and scored 0 in the all_steps scope. Parity: CHAIN_N16 is recomputed and must equal COMP_SLIDER.csv.

    python cosine_bridge.py <minilm dir> <prmbench_telemetry.pkl with question/steps>
"""
import csv, pickle, sys
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'prm_boost_collection_v1'))
import evaluate_collection as E  # noqa: E402

model_dir, pkl = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir); mdl = AutoModel.from_pretrained(model_dir).eval()

def embed(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad(): h = mdl(**b).last_hidden_state
        m = b['attention_mask'].unsqueeze(-1).float(); v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(v, dim=1).numpy())
    return np.concatenate(out)

c = E.load_prmbench('qwen_prm_7b')
R = np.flatnonzero(c.cls == 'redundency'); sl = [slice(c.off[j], c.off[j + 1]) for j in R]
raw = pickle.load(open(pkl, 'rb')); steps = {str(e['idx']): e['steps'] for e in raw.values()}; del raw
cos = np.full(c.off[-1], np.nan)
pairs_a, pairs_b, where = [], [], []
for j, x in zip(R, sl):
    st = steps[c.ids[j]]; assert len(st) == x.stop - x.start, c.ids[j]
    for k in range(1, len(st)):
        pairs_a.append(st[k]); pairs_b.append('\n\n'.join(st[:k])); where.append(x.start + k)
ea, eb = embed(pairs_a), embed(pairs_b)
cos[np.array(where)] = (ea * eb).sum(1)
for x in sl: cos[x.start] = 0.0          # step 1: no prefix (all_steps scope only)

def protocol(s):
    res = {}
    for scope in ('all_steps', 'steps_ge2'):
        keep = [np.arange(x.start, x.stop)[(slice(None) if scope == 'all_steps' else slice(1, None))] for x in sl]
        yy = np.concatenate([c.err[k] for k in keep]); res[f'{scope}__majority_acc'] = 100 * float((~yy).mean())
        res[f'{scope}__n_steps'] = int(len(yy)); res[f'{scope}__pos_rate'] = 100 * float(yy.mean())
        for cw in ('unweighted', 'balanced'):
            acc, bal = [], []
            for sd in range(10):
                perm = np.random.default_rng(sd).permutation(len(keep)); h = len(keep) // 2
                xh = np.concatenate([s[keep[k]] for k in perm[:h]])[:, None]; yh = np.concatenate([c.err[keep[k]] for k in perm[:h]])
                xt = np.concatenate([s[keep[k]] for k in perm[h:]])[:, None]; yt = np.concatenate([c.err[keep[k]] for k in perm[h:]])
                p = LogisticRegression(class_weight=None if cw == 'unweighted' else 'balanced').fit(xh, yh).predict(xt).astype(bool)
                acc.append(100 * (p == yt).mean()); bal.append(50 * (p[yt].mean() + (~p[~yt]).mean()))
            res[f'{scope}__LR_{cw}__acc'] = float(np.mean(acc)); res[f'{scope}__LR_{cw}__acc_sd'] = float(np.std(acc))
            res[f'{scope}__LR_{cw}__bal_acc'] = float(np.mean(bal)); res[f'{scope}__LR_{cw}__bal_acc_sd'] = float(np.std(bal))
        # threshold-free reference: AUROC of the score itself on this scope
        sc = np.concatenate([s[k] for k in keep]); res[f'{scope}__auroc'] = float(E.auc(yy, sc))
    return res

rows = [dict(arm='Cosine similarity (SLIDER App. D.2, ours)', **protocol(cos)),
        dict(arm='CHAIN_N16 (parity with COMP_SLIDER)', **protocol(c.arms['CHAIN_N16']))]
ref = [r for r in E.read_csv(HERE.parent / 'prm_boost_collection_v1/tables/COMP_SLIDER.csv') if r['prm'] == 'qwen_prm_7b' and r['row'] == 'CHAIN_N16 (GAP)'][0]
for k in ('all_steps__LR_balanced__bal_acc', 'all_steps__LR_unweighted__acc', 'steps_ge2__LR_balanced__bal_acc'):
    d = abs(rows[1][k] - float(ref[k])); print('PARITY', k, rows[1][k], ref[k], 'PASS' if d < 1e-9 else 'FAIL'); assert d < 1e-9
with open(HERE / 'COSINE_BRIDGE.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r['arm'], {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items() if k != 'arm'})
