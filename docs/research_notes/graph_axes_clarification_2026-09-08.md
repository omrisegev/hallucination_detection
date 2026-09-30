# Which graph acts on which axis?

This corrects ambiguous wording in older explanations, including calling
Joint-LIU's graph a "feature graph". Its induced penalty acts on feature
weights, but its graph nodes are fitting observations. Frozen experiment
code, protocols and results are unchanged by this clarification.

Let `Z` have N rows and P columns. In the current single-answer method,
N means fitting windows from that answer and P means feature definitions.
In the historical pooled token scorer, N means training-token observations
from other answers. Some low-level functions call `F = Z.T`, so their
documented input shape is P by N. The transpose does not change the data.

| Component | Graph nodes or gated coordinates | Output | Role in fusion |
|---|---|---|---|
| Internal grouping | P feature coordinates | Feature groups | Model which features share residual dependence |
| Joint-LIU graph | N fitting observations | N by N Laplacian, then P by P penalty | Change the fusion weights while retaining the fitting observations |
| Ordinary DUFS feature gates | Gates for P features; graph over N observations | P gate values | Change the geometry used by a graph or the feature weights, depending on the variant |
| Transposed DUFS sampling | Gates for N windows; internal graph over retained feature coordinates | N window priorities | Select fitting windows before the same IU/Joint fusion |
| Window diffusion sampling | N windows | A subset of fitting windows | Cover different parts of the window geometry |

The Joint-LIU path is:

```text
Z: N windows x P features
  -> feature-gated distances between windows
  -> W and L: N x N
  -> R = Z.T @ L @ Z / N: P x P
  -> regularized feature weights w: P
  -> score every window, then map scores to steps
```

For example, with N=80 windows and P=27 feature coordinates, this graph has
80 nodes and R is 27 by 27. The graph encourages nearby observations to have
similar fused scores: `w.T @ R @ w = (Z @ w).T @ L @ (Z @ w) / N`.
Nearby means similar telemetry under the chosen geometry. It does not mean
adjacent in time. Chronological readouts are a separate supporting component.

For transposed DUFS, the current code normalizes each window's feature
profile and feeds an N by P array to a function expecting features by
samples. Its trainable gates now correspond to windows. The current
`dufs_permuted` control shuffles those learned window probabilities; it
does not train a second selector on a new shuffled graph.

Omri's token/window selection idea therefore has a distinct role from
Joint-LIU regularization, even though both use observation-related geometry.
The present selector changes which rows fit the fusion. It still computes
features and scores on the dense grid, so it does not yet demonstrate an
end-to-end sparse-sampling speedup or retained short-error recall.

Code inspected:

- `local_cache/short_cycle01_code/spectral_utils/laplacian_upcr.py`,
  `build_graph_from_features`: transpose P by N into N by P before kNN.
- `local_cache/short_cycle01_code/spectral_utils/joint_lsml.py`,
  `regularized_joint_map_weights`: construct the observation Laplacian and induced R.
- `spectral_utils/fusion_window_sampling.py`, `transposed_gates`,
  `choose_all`, `diffusion_indices`: window gates, shuffled priorities,
  and direct window-graph selection respectively.

These are local implementation facts, not claims that any graph has improved
accuracy. The full matched benchmark must establish the latter.
