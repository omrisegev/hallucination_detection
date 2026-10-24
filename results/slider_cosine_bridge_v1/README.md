# SLIDER bridge: their cosine baseline on our redundancy split (2026-10-24)

SLIDER (arXiv 2610.00571) reports one PRMBench number, Figure 7: "accuracy" on the redundancy class, with a threshold
from a linear classifier trained on a labelled held-out set (Step-RRI 61.9, cosine similarity 51.6, InfoGain 48.4).
The accuracy is not defined. To find out which accuracy it is, we reproduced their cosine-similarity baseline
(App. D.2, Eq. 6: all-MiniLM-L6-v2, current step against the preceding steps) under the split and threshold protocol
of `../prm_boost_collection_v1/collection_competitors.py:slider` (758 problems, 10 random 50/50 splits by problem,
1-D logistic-regression threshold). CPU only; `cosine_bridge.py`, output `COSINE_BRIDGE.csv`.

Parity: CHAIN_N16 recomputed by the same script equals `COMP_SLIDER.csv` exactly on three checked quantities.

| cosine similarity, ours | all steps (step 1 scored 0) | steps 2 onward |
|---|---|---|
| accuracy, class-balanced threshold | **51.9 ± 0.9** | 50.5 ± 1.1 |
| balanced accuracy, class-balanced threshold | 55.4 ± 0.8 | 53.5 ± 0.9 |
| accuracy, unweighted threshold | 86.8 (= always "not redundant") | 85.9 |
| AUROC (no threshold) | .58 | .55 |

Reading: their 51.6 matches plain accuracy with a class-balanced threshold (51.9, within one SD), not balanced
accuracy (55.4). Under that metric on Qwen-7B: L-SML 74.1, backbone telemetry 76.2, the PRM 78.8, Step-RRI 61.9.
Their split sizes and step-1 handling are not stated, and InfoGain (QwQ-32B) was not reproduced.
