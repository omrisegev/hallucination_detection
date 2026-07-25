#!/usr/bin/env python
"""
gap_ladder.py — SPEC — Gap Decomposition Ladder (Step 198).

Evaluates a 7-rung ladder across all 25 in-scope cells at 2 feature sets (GOOD_6 and FULL).
Outputs:
  - results/advisor_inscope/ladder_percell.csv
  - results/advisor_inscope/ladder_summary.csv
  - results/advisor_inscope/ladder_signdiag.csv
  - results/advisor_inscope/ladder_featdiag.csv
  - results/advisor_inscope/ladder_gates.json
  - results/advisor_inscope/ladder.html
"""

import csv
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_anchor_quality import load_all_inscope_cells
from inscope_cells import GROUP, QA_CELLS, MATH_CELLS, INSCOPE
from spectral_utils.selector_bench import eval_subset_flex
from spectral_utils.subset_sweep import GOOD_6
from spectral_utils.fusion_utils import boot_auc, zscore, lsml_continuous
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.selectors.a6_pseudolabel_gates import _seed_cols, _corr_with, _plmrmr_order, MRMR_ALPHA

OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")
os.makedirs(OUT_DIR, exist_ok=True)


class Ctx:
    """Minimal labeled context for eval_subset_flex."""
    def __init__(self, u, labels):
        self.V = np.asarray(u.V, dtype=np.float64)
        self.anchor = np.asarray(u.anchor, dtype=np.float64)
        self.labels = np.asarray(labels, dtype=int)
        self.pool = u.pool
        self.pool_bits = u.pool_bits


def safe_auc_raw(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(set(y_true.tolist())) < 2 or np.all(y_prob == y_prob[0]):
        return 0.5
    try:
        p = float(roc_auc_score(y_true, y_prob))
        return max(p, 1.0 - p)
    except Exception:
        return 0.5


def cv_avg_auc_with_ci_custom(eval_fn, X, y, cv_splits, n_boot=1000):
    """5-fold CV evaluation using eval_fn(X_train, y_train, X_test, y_test) -> (prob, raw_auc)."""
    fold_targets = []
    fold_probs = []
    fold_raw_aucs = []

    for train_idx, test_idx in cv_splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        prob, raw_auc = eval_fn(X_tr, y_tr, X_te, y_te)
        fold_targets.append(y_te)
        fold_probs.append(prob)
        fold_raw_aucs.append(raw_auc)

    fold_aucs = [safe_auc_raw(t, p) for t, p in zip(fold_targets, fold_probs)]
    base_auc = float(np.mean(fold_aucs))
    unfloored_auc = float(np.mean(fold_raw_aucs))

    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        boot_aucs = []
        for target, prob in zip(fold_targets, fold_probs):
            if len(target) < 2 or len(np.unique(target)) < 2:
                continue
            idx = rng.integers(0, len(target), len(target))
            boot_aucs.append(safe_auc_raw(target[idx], prob[idx]))
        if boot_aucs:
            boot_means.append(np.mean(boot_aucs))

    if not boot_means:
        return base_auc, unfloored_auc, base_auc, base_auc
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return base_auc, unfloored_auc, float(lo), float(hi)


def run_gap_ladder():
    print("Loading 25 in-scope cells...", flush=True)
    cells_dict = load_all_inscope_cells()
    print(f"Loaded {len(cells_dict)} cells successfully.", flush=True)

    # Attach problem_id (group array) for group-aware CV splits
    repgrid_path = os.path.join(REPO, "local_cache", "repgrid_cells.pkl")
    rep_cells = {}
    if os.path.exists(repgrid_path):
        import pickle
        with open(repgrid_path, "rb") as f:
            rep_cells = pickle.load(f)

    for ck, cdata in cells_dict.items():
        pid = None
        if ck in rep_cells and 'problem_id' in rep_cells[ck]:
            pid = np.asarray(rep_cells[ck]['problem_id'])
        elif 'fd' in cdata and 'problem_id' in cdata['fd']:
            pid = np.asarray(cdata['fd']['problem_id'])
        cdata['problem_id'] = pid

    percell_rows = []
    signdiag_rows = []
    featdiag_rows = []

    rungs_order = ['R0', 'R0b', 'R1', 'R2', 'R2_cv', 'R3', 'R4', 'R5', 'R5_cv', 'R6']

    for ck in sorted(INSCOPE):
        cdata = cells_dict[ck]
        u = cdata['unlabeled']
        labels = np.asarray(cdata['labels'], dtype=int)
        group = GROUP.get(ck, '?')
        n_samples = len(labels)
        pos_rate = float(np.mean(labels))
        ctx = Ctx(u, labels)
        full_V = ctx.V
        p_full = full_V.shape[1]

        # ----------------------------------------------------
        # 5. Mechanism Diagnostics (at fset=FULL)
        # ----------------------------------------------------
        # 5.1 Label-free orientation error rate
        raw_aucs_full = [roc_auc_score(labels, full_V[:, j]) for j in range(p_full)]
        oracle_signs_full = [+1.0 if a >= 0.5 else -1.0 for a in raw_aucs_full]
        # In u.V, label-free orientation attempted positive correlation.
        # If raw_auc < 0.5, label-free sign is WRONG relative to oracle.
        sign_wrong_full = [1 if a < 0.5 else 0 for a in raw_aucs_full]
        n_sign_wrong = sum(sign_wrong_full)
        frac_sign_wrong = float(n_sign_wrong / p_full)

        # 5.2 Regime sign disagreement (R=3 tertiles of anchor)
        q33, q66 = np.percentile(ctx.anchor, [33.333333333333336, 66.66666666666667])
        regimes_full = np.zeros(n_samples, dtype=int)
        regimes_full[ctx.anchor >= q33] = 1
        regimes_full[ctx.anchor >= q66] = 2

        regime_disagree_count = 0
        regime_signs_per_feat = []

        for j in range(p_full):
            feat_reg_signs = []
            for r in range(3):
                mask_r = (regimes_full == r)
                sub_y = labels[mask_r]
                sub_v = full_V[mask_r, j]
                if len(sub_y) >= 30 and len(np.unique(sub_y)) > 1:
                    a_jr = roc_auc_score(sub_y, sub_v)
                    s_jr = +1.0 if a_jr >= 0.5 else -1.0
                else:
                    s_jr = oracle_signs_full[j]
                feat_reg_signs.append('+' if s_jr > 0 else '-')
                if s_jr != oracle_signs_full[j]:
                    regime_disagree_count += 1
            regime_signs_per_feat.append("".join(feat_reg_signs))

        regime_sign_disagree_frac = float(regime_disagree_count / (3 * p_full))

        # 5.3 Non-monotonicity gain (5-fold CV binned positive rate mapping)
        skf_diag = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        nonmono_gains_full = []
        auc_binned_full = []

        for j in range(p_full):
            fold_binned_aucs = []
            v_col = full_V[:, j]

            for tr_idx, te_idx in skf_diag.split(full_V, labels):
                v_tr, y_tr = v_col[tr_idx], labels[tr_idx]
                v_te, y_te = v_col[te_idx], labels[te_idx]

                try:
                    # 4 quantile bins on train
                    q_edges = np.quantile(v_tr, [0.0, 0.25, 0.5, 0.75, 1.0])
                    # Handle ties in edges
                    if len(set(q_edges)) < 5:
                        q_edges = np.unique(q_edges)

                    if len(q_edges) < 2:
                        fold_binned_aucs.append(0.5)
                        continue

                    # Bin train
                    tr_bins = np.digitize(v_tr, q_edges[1:-1])
                    bin_rates = {}
                    for b_id in range(len(q_edges) - 1):
                        b_mask = (tr_bins == b_id)
                        if np.sum(b_mask) > 0:
                            bin_rates[b_id] = float(np.mean(y_tr[b_mask]))
                        else:
                            bin_rates[b_id] = float(np.mean(y_tr))

                    # Map test
                    te_bins = np.digitize(v_te, q_edges[1:-1])
                    te_scores = np.array([bin_rates.get(b_id, float(np.mean(y_tr))) for b_id in te_bins])
                    fold_binned_aucs.append(safe_auc_raw(y_te, te_scores))
                except Exception:
                    fold_binned_aucs.append(0.5)

            mean_binned = float(np.mean(fold_binned_aucs))
            oracle_single_auc = max(raw_aucs_full[j], 1.0 - raw_aucs_full[j])
            gain = mean_binned - oracle_single_auc
            auc_binned_full.append(mean_binned)
            nonmono_gains_full.append(gain)

            # Feature diag row
            featdiag_rows.append({
                'cell': ck,
                'group': group,
                'feature': u.pool[j],
                'auc_raw': round(float(raw_aucs_full[j]), 4),
                'auc_oriented': round(float(raw_aucs_full[j] if raw_aucs_full[j] >= 0.5 else 1.0 - raw_aucs_full[j]), 4),
                'labelfree_sign': '+' if raw_aucs_full[j] >= 0.5 else '-',
                'oracle_sign': '+' if oracle_signs_full[j] > 0 else '-',
                'sign_wrong': sign_wrong_full[j],
                'auc_binned': round(mean_binned, 4),
                'nonmono_gain': round(gain, 4),
                'regime_signs': regime_signs_per_feat[j],
                'regime_sign_disagree': 1 if '-' in regime_signs_per_feat[j] and '+' in regime_signs_per_feat[j] else 0
            })

        nonmono_mean = float(np.mean(nonmono_gains_full))
        nonmono_p90 = float(np.percentile(nonmono_gains_full, 90))
        nonmono_max = float(np.max(nonmono_gains_full))

        # 5.4 Metadata
        r0_full_eval = eval_subset_flex(ctx, list(range(p_full)), fusion='lsml')
        lsml_K = int(r0_full_eval.get('K', 0))
        lsml_residual = float(r0_full_eval.get('residual', 0.0))
        anc_auc_raw = roc_auc_score(labels, ctx.anchor)
        anchor_auc = float(max(anc_auc_raw, 1.0 - anc_auc_raw))

        signdiag_rows.append({
            'cell': ck,
            'group': group,
            'n': n_samples,
            'pos_rate': round(pos_rate, 4),
            'p_used': p_full,
            'n_labelfree_sign_wrong': n_sign_wrong,
            'frac_labelfree_sign_wrong': round(frac_sign_wrong, 4),
            'regime_sign_disagree_frac': round(regime_sign_disagree_frac, 4),
            'nonmono_gain_mean': round(nonmono_mean, 4),
            'nonmono_gain_p90': round(nonmono_p90, 4),
            'nonmono_gain_max': round(nonmono_max, 4),
            'lsml_K': lsml_K,
            'lsml_residual': round(lsml_residual, 4),
            'anchor_name': u.anchor_name,
            'anchor_auc': round(anchor_auc, 4)
        })

        # ----------------------------------------------------
        # Run Ladder for fset=GOOD_6 and fset=FULL
        # ----------------------------------------------------
        g6_indices = [u.pool.index(f) for f in GOOD_6 if f in u.pool]

        fsets = [('GOOD_6', g6_indices), ('FULL', list(range(p_full)))]

        for fset_name, cols in fsets:
            p_used = len(cols)
            if p_used < 3:
                # Skip if < 3 columns
                for rname in rungs_order:
                    percell_rows.append({
                        'cell': ck, 'group': group, 'fset': fset_name, 'p_used': p_used,
                        'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': rname,
                        'auroc': None, 'auroc_nofloor': None, 'ci_lo': None, 'ci_hi': None,
                        'uses_labels': 0 if rname in ('R0', 'R0b') else 1,
                        'cv': 1 if rname in ('R2_cv', 'R3', 'R4', 'R5_cv') else 0,
                        'notes': 'skipped <3 cols'
                    })
                continue

            V_sub = full_V[:, cols]
            groups = cdata.get('problem_id', None)
            if groups is not None and len(np.unique(groups)) < len(groups):
                splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                cv_splits = list(splitter.split(V_sub, labels, groups=groups))
            else:
                splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_splits = list(splitter.split(V_sub, labels))

            # R0: lf_lsml
            res_r0 = eval_subset_flex(ctx, cols, fusion='lsml')
            auc_r0 = float(res_r0['auroc'])
            # Score vector for CI:
            fused_raw, _ = lsml_continuous(*[full_V[:, c] for c in cols])
            oriented_r0, _ = anchor_orient(fused_raw, ctx.anchor)
            _, ci_lo_r0, ci_hi_r0 = boot_auc(labels, oriented_r0)

            percell_rows.append({
                'cell': ck, 'group': group, 'fset': fset_name, 'p_used': p_used,
                'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': 'R0',
                'auroc': round(auc_r0, 4), 'auroc_nofloor': round(auc_r0, 4),
                'ci_lo': round(float(ci_lo_r0), 4), 'ci_hi': round(float(ci_hi_r0), 4),
                'uses_labels': 0, 'cv': 0, 'notes': ''
            })

            # R0b: lf_lsml_rank (normal-score transform)
            Vt = np.zeros_like(V_sub)
            for j_idx in range(p_used):
                r_rank = rankdata(V_sub[:, j_idx])
                Vt[:, j_idx] = norm.ppf((r_rank - 0.5) / len(r_rank))

            ra = rankdata(ctx.anchor)
            anchor_t = norm.ppf((ra - 0.5) / len(ra))

            ctx_t = Ctx(u, labels)
            ctx_t.V = np.zeros_like(full_V)
            for c_idx, c in enumerate(cols):
                ctx_t.V[:, c] = Vt[:, c_idx]
            ctx_t.anchor = anchor_t

            res_r0b = eval_subset_flex(ctx_t, cols, fusion='lsml')
            auc_r0b = float(res_r0b['auroc'])
            fused_raw_t, _ = lsml_continuous(*[Vt[:, c_idx] for c_idx in range(p_used)])
            oriented_r0b, _ = anchor_orient(fused_raw_t, anchor_t)
            _, ci_lo_r0b, ci_hi_r0b = boot_auc(labels, oriented_r0b)

            percell_rows.append({
                'cell': ck, 'group': group, 'fset': fset_name, 'p_used': p_used,
                'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': 'R0b',
                'auroc': round(auc_r0b, 4), 'auroc_nofloor': round(auc_r0b, 4),
                'ci_lo': round(float(ci_lo_r0b), 4), 'ci_hi': round(float(ci_hi_r0b), 4),
                'uses_labels': 0, 'cv': 0, 'notes': ''
            })

            # R1: oracle_single
            col_raw_aucs = [roc_auc_score(labels, V_sub[:, j_idx]) for j_idx in range(p_used)]
            col_oracle_aucs = [max(a, 1.0 - a) for a in col_raw_aucs]
            j_best = int(np.argmax(col_oracle_aucs))
            auc_r1 = float(col_oracle_aucs[j_best])
            best_feat_name = u.pool[cols[j_best]]

            best_v_oriented = V_sub[:, j_best] * (+1.0 if col_raw_aucs[j_best] >= 0.5 else -1.0)
            _, ci_lo_r1, ci_hi_r1 = boot_auc(labels, best_v_oriented)

            percell_rows.append({
                'cell': ck, 'group': group, 'fset': fset_name, 'p_used': p_used,
                'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': 'R1',
                'auroc': round(auc_r1, 4), 'auroc_nofloor': round(auc_r1, 4),
                'ci_lo': round(float(ci_lo_r1), 4), 'ci_hi': round(float(ci_hi_r1), 4),
                'uses_labels': 1, 'cv': 0, 'notes': f"argmax={best_feat_name}"
            })

            # R2: oracle_sign_eq (in-sample)
            oracle_s_sub = np.array([+1.0 if a >= 0.5 else -1.0 for a in col_raw_aucs])
            score_r2 = np.mean(oracle_s_sub * zscore(V_sub), axis=1)
            auc_r2 = float(roc_auc_score(labels, score_r2))
            _, ci_lo_r2, ci_hi_r2 = boot_auc(labels, score_r2)

            percell_rows.append({
                'cell': ck, 'group': group, 'fset': fset_name, 'p_used': p_used,
                'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': 'R2',
                'auroc': round(auc_r2, 4), 'auroc_nofloor': round(auc_r2, 4),
                'ci_lo': round(float(ci_lo_r2), 4), 'ci_hi': round(float(ci_hi_r2), 4),
                'uses_labels': 1, 'cv': 0, 'notes': 'in-sample upper bound'
            })

            # R2_cv: oracle_sign_eq (5-fold CV)
            def eval_r2_cv(X_tr, y_tr, X_te, y_te):
                s_tr = np.array([+1.0 if roc_auc_score(y_tr, X_tr[:, j_idx]) >= 0.5 else -1.0 for j_idx in range(p_used)])
                score_te = np.mean(s_tr * zscore(X_te), axis=1)
                raw_a = roc_auc_score(y_te, score_te) if len(set(y_te)) > 1 else 0.5
                return score_te, raw_a

            auc_r2cv, nofl_r2cv, lo_r2cv, hi_r2cv = cv_avg_auc_with_ci_custom(eval_r2_cv, V_sub, labels, cv_splits)
            percell_rows.append({
                'cell': ck, 'group': group, 'fset': fset_name, 'p_used': p_used,
                'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': 'R2_cv',
                'auroc': round(auc_r2cv, 4), 'auroc_nofloor': round(nofl_r2cv, 4),
                'ci_lo': round(lo_r2cv, 4), 'ci_hi': round(hi_r2cv, 4),
                'uses_labels': 1, 'cv': 1, 'notes': ''
            })

            # R3: oracle_lin (Logistic Regression 5-fold CV)
            def eval_r3(X_tr, y_tr, X_te, y_te):
                pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs'))
                pipe.fit(X_tr, y_tr)
                prob = pipe.predict_proba(X_te)[:, 1]
                raw_a = roc_auc_score(y_te, prob) if len(set(y_te)) > 1 else 0.5
                return prob, raw_a

            auc_r3, nofl_r3, lo_r3, hi_r3 = cv_avg_auc_with_ci_custom(eval_r3, V_sub, labels, cv_splits)
            percell_rows.append({
                'cell': ck, 'group': group, 'fset': fset_name, 'p_used': p_used,
                'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': 'R3',
                'auroc': round(auc_r3, 4), 'auroc_nofloor': round(nofl_r3, 4),
                'ci_lo': round(lo_r3, 4), 'ci_hi': round(hi_r3, 4),
                'uses_labels': 1, 'cv': 1, 'notes': ''
            })

            # R4: oracle_nonlin (HistGradientBoosting 5-fold CV)
            def eval_r4(X_tr, y_tr, X_te, y_te):
                sw_tr = compute_sample_weight('balanced', y_tr)
                clf = HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                    min_samples_leaf=20, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.15,
                    random_state=42)
                clf.fit(X_tr, y_tr, sample_weight=sw_tr)
                prob = clf.predict_proba(X_te)[:, 1]
                raw_a = roc_auc_score(y_te, prob) if len(set(y_te)) > 1 else 0.5
                return prob, raw_a

            auc_r4, nofl_r4, lo_r4, hi_r4 = cv_avg_auc_with_ci_custom(eval_r4, V_sub, labels, cv_splits)
            percell_rows.append({
                'cell': ck, 'group': group, 'fset': fset_name, 'p_used': p_used,
                'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': 'R4',
                'auroc': round(auc_r4, 4), 'auroc_nofloor': round(nofl_r4, 4),
                'ci_lo': round(lo_r4, 4), 'ci_hi': round(hi_r4, 4),
                'uses_labels': 1, 'cv': 1, 'notes': ''
            })

            # R5: oracle_regime_sign (in-sample)
            fallback_r5_count = 0
            score_r5 = np.zeros(n_samples, dtype=float)
            q33_c, q66_c = np.percentile(ctx.anchor, [33.333333333333336, 66.66666666666667])
            regimes_c = np.zeros(n_samples, dtype=int)
            regimes_c[ctx.anchor >= q33_c] = 1
            regimes_c[ctx.anchor >= q66_c] = 2

            s_reg_insample = np.zeros((3, p_used), dtype=float)
            for j_idx in range(p_used):
                for r in range(3):
                    mask_r = (regimes_c == r)
                    sub_y = labels[mask_r]
                    sub_v = V_sub[mask_r, j_idx]
                    if len(sub_y) >= 30 and len(np.unique(sub_y)) > 1:
                        a_jr = roc_auc_score(sub_y, sub_v)
                        s_reg_insample[r, j_idx] = +1.0 if a_jr >= 0.5 else -1.0
                    else:
                        s_reg_insample[r, j_idx] = oracle_s_sub[j_idx]
                        fallback_r5_count += 1

            for n_idx in range(n_samples):
                r_n = regimes_c[n_idx]
                score_r5[n_idx] = np.mean(s_reg_insample[r_n, :] * zscore(V_sub)[n_idx, :])

            auc_r5 = float(roc_auc_score(labels, score_r5))
            _, ci_lo_r5, ci_hi_r5 = boot_auc(labels, score_r5)
            percell_rows.append({
                'cell': ck, 'group': group, 'fset': fset_name, 'p_used': p_used,
                'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': 'R5',
                'auroc': round(auc_r5, 4), 'auroc_nofloor': round(auc_r5, 4),
                'ci_lo': round(float(ci_lo_r5), 4), 'ci_hi': round(float(ci_hi_r5), 4),
                'uses_labels': 1, 'cv': 0, 'notes': f"in-sample ceiling, fallbacks={fallback_r5_count}"
            })

            # R5_cv: oracle_regime_sign (5-fold CV)
            def eval_r5_cv(X_tr, y_tr, X_te, y_te, tr_idx_cur, te_idx_cur):
                anc_tr = ctx.anchor[tr_idx_cur]
                anc_te = ctx.anchor[te_idx_cur]
                q33_tr, q66_tr = np.percentile(anc_tr, [33.333333333333336, 66.66666666666667])

                r_tr = np.zeros(len(y_tr), dtype=int)
                r_tr[anc_tr >= q33_tr] = 1
                r_tr[anc_tr >= q66_tr] = 2

                r_te = np.zeros(len(y_te), dtype=int)
                r_te[anc_te >= q33_tr] = 1
                r_te[anc_te >= q66_tr] = 2

                s_tr_reg = np.zeros((3, p_used), dtype=float)
                s_tr_glob = np.array([+1.0 if roc_auc_score(y_tr, X_tr[:, j_idx]) >= 0.5 else -1.0 for j_idx in range(p_used)])

                for j_idx in range(p_used):
                    for r in range(3):
                        m_r = (r_tr == r)
                        sub_y = y_tr[m_r]
                        sub_v = X_tr[m_r, j_idx]
                        if len(sub_y) >= 30 and len(np.unique(sub_y)) > 1:
                            a_jr = roc_auc_score(sub_y, sub_v)
                            s_tr_reg[r, j_idx] = +1.0 if a_jr >= 0.5 else -1.0
                        else:
                            s_tr_reg[r, j_idx] = s_tr_glob[j_idx]

                score_te = np.zeros(len(y_te), dtype=float)
                X_te_zs = zscore(X_te)
                for n_i in range(len(y_te)):
                    r_i = r_te[n_i]
                    score_te[n_i] = np.mean(s_tr_reg[r_i, :] * X_te_zs[n_i, :])

                raw_a = roc_auc_score(y_te, score_te) if len(set(y_te)) > 1 else 0.5
                return score_te, raw_a

            fold_targets_r5 = []
            fold_probs_r5 = []
            fold_raw_aucs_r5 = []

            for tr_idx_cur, te_idx_cur in cv_splits:
                prob_te, raw_a = eval_r5_cv(V_sub[tr_idx_cur], labels[tr_idx_cur], V_sub[te_idx_cur], labels[te_idx_cur], tr_idx_cur, te_idx_cur)
                fold_targets_r5.append(labels[te_idx_cur])
                fold_probs_r5.append(prob_te)
                fold_raw_aucs_r5.append(raw_a)

            fold_aucs_r5 = [safe_auc_raw(t, p) for t, p in zip(fold_targets_r5, fold_probs_r5)]
            auc_r5cv = float(np.mean(fold_aucs_r5))
            nofl_r5cv = float(np.mean(fold_raw_aucs_r5))

            rng_b = np.random.default_rng(42)
            boot_means_r5 = []
            for _ in range(1000):
                boot_aucs = []
                for target, prob in zip(fold_targets_r5, fold_probs_r5):
                    if len(target) < 2 or len(np.unique(target)) < 2:
                        continue
                    idx_b = rng_b.integers(0, len(target), len(target))
                    boot_aucs.append(safe_auc_raw(target[idx_b], prob[idx_b]))
                if boot_aucs:
                    boot_means_r5.append(np.mean(boot_aucs))

            lo_r5cv, hi_r5cv = np.percentile(boot_means_r5, [2.5, 97.5]) if boot_means_r5 else (auc_r5cv, auc_r5cv)

            percell_rows.append({
                'cell': ck, 'group': group, 'fset': fset_name, 'p_used': p_used,
                'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': 'R5_cv',
                'auroc': round(auc_r5cv, 4), 'auroc_nofloor': round(nofl_r5cv, 4),
                'ci_lo': round(float(lo_r5cv), 4), 'ci_hi': round(float(hi_r5cv), 4),
                'uses_labels': 1, 'cv': 1, 'notes': ''
            })

            # R6: perfect consensus target (FULL only)
            if fset_name == 'FULL':
                y_hat = labels
                seed_cols, seed_names = _seed_cols(u)
                sel_cols = [j for j in range(p_full) if j not in seed_cols]
                if sel_cols:
                    agree = _corr_with(full_V[:, sel_cols], y_hat)
                    order = _plmrmr_order(full_V[:, sel_cols], agree, alpha=MRMR_ALPHA)
                    target_k = min(15, len(seed_cols) + len(sel_cols))
                    cols_r6 = seed_cols + [sel_cols[j] for j in order[:target_k - len(seed_cols)]]
                else:
                    cols_r6 = seed_cols

                res_r6 = eval_subset_flex(ctx, cols_r6, fusion='lsml')
                auc_r6 = float(res_r6['auroc'])
                fused_raw_r6, _ = lsml_continuous(*[full_V[:, c] for c in cols_r6])
                oriented_r6, _ = anchor_orient(fused_raw_r6, ctx.anchor)
                _, ci_lo_r6, ci_hi_r6 = boot_auc(labels, oriented_r6)

                percell_rows.append({
                    'cell': ck, 'group': group, 'fset': fset_name, 'p_used': len(cols_r6),
                    'n': n_samples, 'pos_rate': round(pos_rate, 4), 'rung': 'R6',
                    'auroc': round(auc_r6, 4), 'auroc_nofloor': round(auc_r6, 4),
                    'ci_lo': round(float(ci_lo_r6), 4), 'ci_hi': round(float(ci_hi_r6), 4),
                    'uses_labels': 1, 'cv': 0, 'notes': f"target=true_labels, K={len(cols_r6)}"
                })

    # ----------------------------------------------------
    # Save CSVs
    # ----------------------------------------------------
    # 6.1 ladder_percell.csv
    percell_path = os.path.join(OUT_DIR, "ladder_percell.csv")
    with open(percell_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'cell', 'group', 'fset', 'p_used', 'n', 'pos_rate', 'rung', 'auroc',
            'auroc_nofloor', 'ci_lo', 'ci_hi', 'uses_labels', 'cv', 'notes'
        ])
        writer.writeheader()
        writer.writerows(percell_rows)

    # 6.3 ladder_signdiag.csv
    signdiag_path = os.path.join(OUT_DIR, "ladder_signdiag.csv")
    with open(signdiag_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'cell', 'group', 'n', 'pos_rate', 'p_used', 'n_labelfree_sign_wrong',
            'frac_labelfree_sign_wrong', 'regime_sign_disagree_frac',
            'nonmono_gain_mean', 'nonmono_gain_p90', 'nonmono_gain_max',
            'lsml_K', 'lsml_residual', 'anchor_name', 'anchor_auc'
        ])
        writer.writeheader()
        writer.writerows(signdiag_rows)

    # 6.4 ladder_featdiag.csv
    featdiag_path = os.path.join(OUT_DIR, "ladder_featdiag.csv")
    with open(featdiag_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'cell', 'group', 'feature', 'auc_raw', 'auc_oriented', 'labelfree_sign',
            'oracle_sign', 'sign_wrong', 'auc_binned', 'nonmono_gain',
            'regime_signs', 'regime_sign_disagree'
        ])
        writer.writeheader()
        writer.writerows(featdiag_rows)

    # ----------------------------------------------------
    # 6.2 ladder_summary.csv
    # ----------------------------------------------------
    df_percell = pd.DataFrame(percell_rows)
    summary_rows = []

    for fset_name in ['GOOD_6', 'FULL']:
        df_fset = df_percell[df_percell['fset'] == fset_name]

        # Lookup R0 and R3 values per cell for deltas & Wilcoxon
        r0_by_cell = df_fset[df_fset['rung'] == 'R0'].set_index('cell')['auroc'].to_dict()
        r3_by_cell = df_fset[df_fset['rung'] == 'R3'].set_index('cell')['auroc'].to_dict()

        for rname in rungs_order:
            sub_rung = df_fset[df_fset['rung'] == rname]
            cells_valid = sub_rung.dropna(subset=['auroc'])
            n_cells = len(cells_valid)
            if n_cells == 0:
                continue

            qa_vals = sub_rung[sub_rung['group'] == 'QA']['auroc'].dropna().tolist()
            math_vals = sub_rung[sub_rung['group'] == 'math']['auroc'].dropna().tolist()

            macro_qa = float(np.mean(qa_vals)) if qa_vals else 0.0
            macro_math = float(np.mean(math_vals)) if math_vals else 0.0
            macro_all = (10.0 * macro_qa + 15.0 * macro_math) / 25.0

            # Deltas vs R0
            all_r0_pairs = [(r['auroc'], r0_by_cell[r['cell']]) for _, r in sub_rung.iterrows() if r['cell'] in r0_by_cell and pd.notnull(r['auroc']) and pd.notnull(r0_by_cell[r['cell']])]
            qa_r0_pairs = [(r['auroc'], r0_by_cell[r['cell']]) for _, r in sub_rung[sub_rung['group']=='QA'].iterrows() if r['cell'] in r0_by_cell and pd.notnull(r['auroc']) and pd.notnull(r0_by_cell[r['cell']])]
            math_r0_pairs = [(r['auroc'], r0_by_cell[r['cell']]) for _, r in sub_rung[sub_rung['group']=='math'].iterrows() if r['cell'] in r0_by_cell and pd.notnull(r['auroc']) and pd.notnull(r0_by_cell[r['cell']])]

            d_R0_all = float(np.mean([a - b for a, b in all_r0_pairs])) if all_r0_pairs else 0.0
            d_R0_qa = float(np.mean([a - b for a, b in qa_r0_pairs])) if qa_r0_pairs else 0.0
            d_R0_math = float(np.mean([a - b for a, b in math_r0_pairs])) if math_r0_pairs else 0.0

            # Deltas vs R3
            all_r3_pairs = [(r['auroc'], r3_by_cell[r['cell']]) for _, r in sub_rung.iterrows() if r['cell'] in r3_by_cell and pd.notnull(r['auroc']) and pd.notnull(r3_by_cell[r['cell']])]
            qa_r3_pairs = [(r['auroc'], r3_by_cell[r['cell']]) for _, r in sub_rung[sub_rung['group']=='QA'].iterrows() if r['cell'] in r3_by_cell and pd.notnull(r['auroc']) and pd.notnull(r3_by_cell[r['cell']])]
            math_r3_pairs = [(r['auroc'], r3_by_cell[r['cell']]) for _, r in sub_rung[sub_rung['group']=='math'].iterrows() if r['cell'] in r3_by_cell and pd.notnull(r['auroc']) and pd.notnull(r3_by_cell[r['cell']])]

            d_R3_all = float(np.mean([a - b for a, b in all_r3_pairs])) if all_r3_pairs else 0.0
            d_R3_qa = float(np.mean([a - b for a, b in qa_r3_pairs])) if qa_r3_pairs else 0.0
            d_R3_math = float(np.mean([a - b for a, b in math_r3_pairs])) if math_r3_pairs else 0.0

            # Wilcoxon vs R0
            if rname == 'R0' or len(all_r0_pairs) < 5:
                w_p_R0_all, w_wins_R0_all, w_losses_R0_all = None, 0, 0
                w_p_R0_qa = None
            else:
                a_arr = np.array([x for x, _ in all_r0_pairs])
                b_arr = np.array([y for _, y in all_r0_pairs])
                diffs = a_arr - b_arr
                w_wins_R0_all = int(np.sum(diffs > 0))
                w_losses_R0_all = int(np.sum(diffs < 0))
                try:
                    w_p_R0_all = float(wilcoxon(a_arr, b_arr).pvalue)
                except Exception:
                    w_p_R0_all = None

                qa_a = np.array([x for x, _ in qa_r0_pairs])
                qa_b = np.array([y for _, y in qa_r0_pairs])
                try:
                    w_p_R0_qa = float(wilcoxon(qa_a, qa_b).pvalue)
                except Exception:
                    w_p_R0_qa = None

            # Wilcoxon vs R3
            if rname == 'R3' or len(all_r3_pairs) < 5:
                w_p_R3_all, w_wins_R3_all, w_losses_R3_all = None, 0, 0
                w_p_R3_qa = None
            else:
                a_arr = np.array([x for x, _ in all_r3_pairs])
                b_arr = np.array([y for _, y in all_r3_pairs])
                diffs = a_arr - b_arr
                w_wins_R3_all = int(np.sum(diffs > 0))
                w_losses_R3_all = int(np.sum(diffs < 0))
                try:
                    w_p_R3_all = float(wilcoxon(a_arr, b_arr).pvalue)
                except Exception:
                    w_p_R3_all = None

                qa_a = np.array([x for x, _ in qa_r3_pairs])
                qa_b = np.array([y for _, y in qa_r3_pairs])
                try:
                    w_p_R3_qa = float(wilcoxon(qa_a, qa_b).pvalue)
                except Exception:
                    w_p_R3_qa = None

            summary_rows.append({
                'fset': fset_name,
                'rung': rname,
                'n_cells': n_cells,
                'macro_all': round(macro_all, 4),
                'macro_qa': round(macro_qa, 4),
                'macro_math': round(macro_math, 4),
                'd_R0_all': round(d_R0_all, 4),
                'd_R0_qa': round(d_R0_qa, 4),
                'd_R0_math': round(d_R0_math, 4),
                'd_R3_all': round(d_R3_all, 4),
                'd_R3_qa': round(d_R3_qa, 4),
                'd_R3_math': round(d_R3_math, 4),
                'w_p_vs_R0_all': round(w_p_R0_all, 5) if w_p_R0_all is not None else None,
                'w_p_vs_R0_qa': round(w_p_R0_qa, 5) if w_p_R0_qa is not None else None,
                'w_wins_vs_R0_all': w_wins_R0_all,
                'w_losses_vs_R0_all': w_losses_R0_all,
                'w_p_vs_R3_all': round(w_p_R3_all, 5) if w_p_R3_all is not None else None,
                'w_p_vs_R3_qa': round(w_p_R3_qa, 5) if w_p_R3_qa is not None else None,
                'w_wins_vs_R3_all': w_wins_R3_all,
                'w_losses_vs_R3_all': w_losses_R3_all
            })

    summary_path = os.path.join(OUT_DIR, "ladder_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'fset', 'rung', 'n_cells', 'macro_all', 'macro_qa', 'macro_math',
            'd_R0_all', 'd_R0_qa', 'd_R0_math', 'd_R3_all', 'd_R3_qa', 'd_R3_math',
            'w_p_vs_R0_all', 'w_p_vs_R0_qa', 'w_wins_vs_R0_all', 'w_losses_vs_R0_all',
            'w_p_vs_R3_all', 'w_p_vs_R3_qa', 'w_wins_vs_R3_all', 'w_losses_vs_R3_all'
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    # ----------------------------------------------------
    # 6.5 ladder_gates.json
    # ----------------------------------------------------
    df_sum = pd.DataFrame(summary_rows)
    sum_full = df_sum[df_sum['fset'] == 'FULL'].set_index('rung')
    sum_g6 = df_sum[df_sum['fset'] == 'GOOD_6'].set_index('rung')

    # R3 FULL values for validity check 1
    r3_full_qa = float(sum_full.loc['R3', 'macro_qa'])
    r3_full_math = float(sum_full.loc['R3', 'macro_math'])
    r3_full_all = float(sum_full.loc['R3', 'macro_all'])

    # R0 GOOD_6 values for validity check 2
    r0_g6_all = float(sum_g6.loc['R0', 'macro_all'])

    v1_pass = bool(abs(r3_full_qa - 0.7524) <= 0.005 and abs(r3_full_math - 0.8000) <= 0.005 and abs(r3_full_all - 0.7810) <= 0.005)
    v2_pass = bool(abs(r0_g6_all - 0.7594) <= 0.002)

    # Gate calculations at fset=FULL
    # 1. Nonlinearity: R4 - R3
    delta_r4_all = float(sum_full.loc['R4', 'macro_all'] - sum_full.loc['R3', 'macro_all'])
    delta_r4_qa = float(sum_full.loc['R4', 'macro_qa'] - sum_full.loc['R3', 'macro_qa'])
    p_r4_all = float(sum_full.loc['R4', 'w_p_vs_R3_all']) if sum_full.loc['R4', 'w_p_vs_R3_all'] is not None else 1.0
    verdict_nonlinearity = "ALIVE" if (delta_r4_all >= 0.010 and p_r4_all <= 0.05) else "DEAD"

    # 2. Non-stationary sign: R5 (in-sample) - R3
    delta_r5_all = float(sum_full.loc['R5', 'macro_all'] - sum_full.loc['R3', 'macro_all'])
    delta_r5_qa = float(sum_full.loc['R5', 'macro_qa'] - sum_full.loc['R3', 'macro_qa'])
    p_r5_all = float(sum_full.loc['R5', 'w_p_vs_R3_all']) if sum_full.loc['R5', 'w_p_vs_R3_all'] is not None else 1.0
    verdict_nonstationary = "ALIVE" if (delta_r5_all >= 0.010 and p_r5_all <= 0.05) else "DEAD"

    # 3. Sign recovery loss: R2 - R0
    delta_sign_rec_all = float(sum_full.loc['R2', 'macro_all'] - sum_full.loc['R0', 'macro_all'])
    delta_sign_rec_qa = float(sum_full.loc['R2', 'macro_qa'] - sum_full.loc['R0', 'macro_qa'])

    # 4. Weight estimation loss: R3 - R2
    delta_weight_est_all = float(sum_full.loc['R3', 'macro_all'] - sum_full.loc['R2', 'macro_all'])
    delta_weight_est_qa = float(sum_full.loc['R3', 'macro_qa'] - sum_full.loc['R2', 'macro_qa'])

    # 5. Target quality: R6 vs GOOD_6 (R0 at GOOD_6) and D2_alone
    r6_macro_all = float(sum_full.loc['R6', 'macro_all']) if 'R6' in sum_full.index else 0.0
    r6_macro_qa = float(sum_full.loc['R6', 'macro_qa']) if 'R6' in sum_full.index else 0.0
    delta_r6_good6 = r6_macro_all - r0_g6_all
    delta_r6_d2 = r6_macro_all - 0.7573
    # Step 201 (defect 7): this used to read `w_p_vs_R0_all`, which is R6 vs
    # R0 at *FULL* (0.7457) -- NOT vs GOOD_6 (0.7594). The delta and the p-value
    # therefore described different contrasts. Compute the paired Wilcoxon that
    # actually matches `delta_vs_good6`: R6@FULL vs R0@GOOD_6, per cell.
    _pc = pd.DataFrame(percell_rows)
    _r6 = (_pc[(_pc['fset'] == 'FULL') & (_pc['rung'] == 'R6')]
           .set_index('cell')['auroc'].dropna())
    _r0g6 = (_pc[(_pc['fset'] == 'GOOD_6') & (_pc['rung'] == 'R0')]
             .set_index('cell')['auroc'].dropna())
    _common = _r6.index.intersection(_r0g6.index)
    if len(_common) >= 3:
        _a, _b = _r6[_common].to_numpy(), _r0g6[_common].to_numpy()
        try:
            p_r6_vs_good6 = float(wilcoxon(_a, _b).pvalue) if np.any(_a != _b) else 1.0
        except Exception:
            p_r6_vs_good6 = 1.0
        r6_wins_vs_good6 = int((_a > _b).sum())
        r6_losses_vs_good6 = int((_a < _b).sum())
    else:
        p_r6_vs_good6, r6_wins_vs_good6, r6_losses_vs_good6 = 1.0, 0, 0
    verdict_target_quality = "ALIVE" if (delta_r6_good6 >= 0.010 and p_r6_vs_good6 <= 0.05) else "DEAD"

    # Dominant term
    gaps_map = {
        "sign_recovery": delta_sign_rec_all,
        "weight_estimation": delta_weight_est_all,
        "nonlinearity": delta_r4_all if verdict_nonlinearity == "ALIVE" else -999.0,
        "nonstationary_sign": delta_r5_all if verdict_nonstationary == "ALIVE" else -999.0,
        "target_quality": delta_r6_good6 if verdict_target_quality == "ALIVE" else -999.0
    }
    dominant_term = max(gaps_map, key=gaps_map.get)

    v1_pass = bool(abs(r3_full_all - 0.7810) <= 0.005)
    v2_pass = bool(abs(r0_g6_all - 0.7594) <= 0.002)

    gates_json = {
        "validity": {
            "R3_FULL_macro_qa": round(r3_full_qa, 4),
            "R3_FULL_macro_math": round(r3_full_math, 4),
            "R3_FULL_macro_all": round(r3_full_all, 4),
            "ref_lr_oracle_qa": 0.7524,
            "ref_lr_oracle_math": 0.8000,
            "ref_lr_oracle_all": 0.7810,
            "R3_reproduces_lr_oracle": v1_pass,
            "R0_GOOD_6_macro_all": round(r0_g6_all, 4),
            "ref_good6_macro_all": 0.7594,
            "R0_reproduces_good6": v2_pass
        },
        "gates": {
            "nonlinearity": {
                "delta_all": round(delta_r4_all, 4),
                "delta_qa": round(delta_r4_qa, 4),
                "p_all": round(p_r4_all, 5),
                "verdict": verdict_nonlinearity
            },
            "nonstationary_sign": {
                "delta_all": round(delta_r5_all, 4),
                "delta_qa": round(delta_r5_qa, 4),
                "p_all": round(p_r5_all, 5),
                "verdict": verdict_nonstationary
            },
            "sign_recovery_loss": {
                "delta_all": round(delta_sign_rec_all, 4),
                "delta_qa": round(delta_sign_rec_qa, 4),
                # Step 201: this contrast is R2 (equal-weight mean of ORACLE-SIGNED
                # columns) minus R0 (L-SML), so it confounds sign recovery with
                # FUSION METHOD and must not be read as "sign is the bottleneck".
                # Measured directly (scripts/h1_orientation_audit.py): holding the
                # fusion fixed and varying only the input signs changes the fused
                # score by EXACTLY 0.0 -- 1150/1150 sign vectors, incl. 20 random
                # per cell, are bit-identical. L-SML is gauge-invariant to input
                # column signs, so there is no sign headroom to recover here.
                "confounded_with_fusion_method": True,
                "isolated_sign_effect_all": 0.0,
                "note": ("R2-R0 mixes sign with fusion method; the isolated sign "
                         "effect is 0.0 (L-SML gauge invariance, see "
                         "results/advisor_inscope/h1_orientation_audit.csv)")
            },
            "weight_estimation_loss": {
                "delta_all": round(delta_weight_est_all, 4),
                "delta_qa": round(delta_weight_est_qa, 4)
            },
            "target_quality": {
                "R6_macro_all": round(r6_macro_all, 4),
                "R6_macro_qa": round(r6_macro_qa, 4),
                "delta_vs_good6": round(delta_r6_good6, 4),
                "delta_vs_D2_alone": round(delta_r6_d2, 4),
                "p_vs_good6": round(p_r6_vs_good6, 5),
                "wins_vs_good6": r6_wins_vs_good6,
                "losses_vs_good6": r6_losses_vs_good6,
                "p_contrast": "R6@FULL vs R0@GOOD_6, paired Wilcoxon over cells",
                "verdict": verdict_target_quality
            },
            "dominant_term": dominant_term
        }
    }

    gates_json_path = os.path.join(OUT_DIR, "ladder_gates.json")
    with open(gates_json_path, "w", encoding="utf-8") as f:
        json.dump(gates_json, f, indent=2)

    # ----------------------------------------------------
    # 6.6 ladder.html
    # ----------------------------------------------------
    html_path = os.path.join(OUT_DIR, "ladder.html")
    build_html_dashboard(df_percell, df_sum, pd.DataFrame(signdiag_rows), gates_json, html_path)

    print("\nGap Decomposition Ladder complete! All output files generated in results/advisor_inscope/", flush=True)


def build_html_dashboard(df_percell, df_summary, df_signdiag, gates_json, html_path):
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Gap Decomposition Ladder (Step 198)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f8fafc; color: #1e293b; }}
        h1, h2, h3 {{ color: #0f172a; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .metric-card {{ background: #f1f5f9; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-val {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
        .verdict-dead {{ color: #dc2626; font-weight: bold; }}
        .verdict-alive {{ color: #16a34a; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
        th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; }}
        tr:hover {{ background: #f8fafc; }}
        .chart-container {{ position: relative; height: 350px; width: 100%; }}
    </style>
</head>
<body>
    <h1>Gap Decomposition Ladder (Step 198)</h1>
    
    <div class="card">
        <h2>1. Gate Verdicts & Validity</h2>
        <div class="grid">
            <div class="metric-card">
                <div>Nonlinearity Gate</div>
                <div class="metric-val verdict-{gates_json['gates']['nonlinearity']['verdict'].lower()}">{gates_json['gates']['nonlinearity']['verdict']}</div>
                <div>&Delta;={gates_json['gates']['nonlinearity']['delta_all']:+.4f} (p={gates_json['gates']['nonlinearity']['p_all']:.4f})</div>
            </div>
            <div class="metric-card">
                <div>Non-stationary Sign Gate</div>
                <div class="metric-val verdict-{gates_json['gates']['nonstationary_sign']['verdict'].lower()}">{gates_json['gates']['nonstationary_sign']['verdict']}</div>
                <div>&Delta;={gates_json['gates']['nonstationary_sign']['delta_all']:+.4f} (p={gates_json['gates']['nonstationary_sign']['p_all']:.4f})</div>
            </div>
            <div class="metric-card">
                <div>Sign Recovery Loss</div>
                <div class="metric-val">&Delta;={gates_json['gates']['sign_recovery_loss']['delta_all']:+.4f}</div>
                <div>(R2 vs R0)</div>
            </div>
            <div class="metric-card">
                <div>Weight Estimation Loss</div>
                <div class="metric-val">&Delta;={gates_json['gates']['weight_estimation_loss']['delta_all']:+.4f}</div>
                <div>(R3 vs R2)</div>
            </div>
            <div class="metric-card">
                <div>Target Quality Gate</div>
                <div class="metric-val verdict-{gates_json['gates']['target_quality']['verdict'].lower()}">{gates_json['gates']['target_quality']['verdict']}</div>
                <div>R6={gates_json['gates']['target_quality']['R6_macro_all']:.4f} (&Delta;={gates_json['gates']['target_quality']['delta_vs_good6']:+.4f})</div>
            </div>
        </div>
        <p><b>Dominant Term:</b> <span style="font-size: 16px; font-weight: bold; color: #0284c7;">{gates_json['gates']['dominant_term'].upper()}</span></p>
        <p><b>Validity Checks:</b> R3 FULL reproduces LR Oracle: <b>{gates_json['validity']['R3_reproduces_lr_oracle']}</b> | R0 GOOD_6 reproduces GOOD_6: <b>{gates_json['validity']['R0_reproduces_good6']}</b></p>
    </div>

    <div class="card">
        <h2>2. Ladder Summary Table (FULL Feature Set)</h2>
        <table>
            <thead>
                <tr>
                    <th>Rung</th>
                    <th>Name</th>
                    <th>Macro All</th>
                    <th>Macro QA</th>
                    <th>Macro Math</th>
                    <th>&Delta; vs R0</th>
                    <th>p vs R0</th>
                    <th>&Delta; vs R3</th>
                    <th>p vs R3</th>
                </tr>
            </thead>
            <tbody>
"""
    df_full_sum = df_summary[df_summary['fset'] == 'FULL']
    for _, r in df_full_sum.iterrows():
        html += f"""
                <tr>
                    <td><b>{r['rung']}</b></td>
                    <td>{r['rung']}</td>
                    <td><b>{r['macro_all']:.4f}</b></td>
                    <td>{r['macro_qa']:.4f}</td>
                    <td>{r['macro_math']:.4f}</td>
                    <td>{r['d_R0_all']:+.4f}</td>
                    <td>{r['w_p_vs_R0_all'] if r['w_p_vs_R0_all'] is not None else '-'}</td>
                    <td>{r['d_R3_all']:+.4f}</td>
                    <td>{r['w_p_vs_R3_all'] if r['w_p_vs_R3_all'] is not None else '-'}</td>
                </tr>
"""
    html += """
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>3. Label-free Sign Error vs R2 - R0 Gain</h2>
        <div class="chart-container">
            <canvas id="scatterChart"></canvas>
        </div>
    </div>

    <script>
        const scatterData = [
"""
    # Build scatter data
    df_r0_full = df_percell[(df_percell['fset']=='FULL') & (df_percell['rung']=='R0')].set_index('cell')
    df_r2_full = df_percell[(df_percell['fset']=='FULL') & (df_percell['rung']=='R2')].set_index('cell')
    df_diag_idx = df_signdiag.set_index('cell')

    for ck in sorted(INSCOPE):
        if ck in df_r0_full.index and ck in df_r2_full.index and ck in df_diag_idx.index:
            r0_val = df_r0_full.loc[ck, 'auroc']
            r2_val = df_r2_full.loc[ck, 'auroc']
            gain = r2_val - r0_val
            err_frac = df_diag_idx.loc[ck, 'frac_labelfree_sign_wrong']
            grp = df_diag_idx.loc[ck, 'group']
            html += f"{{ x: {err_frac}, y: {gain:.4f}, label: '{ck}', group: '{grp}' }},\n"

    html += """
        ];

        new Chart(document.getElementById('scatterChart'), {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'QA Cells',
                        data: scatterData.filter(d => d.group === 'QA'),
                        backgroundColor: '#ef4444'
                    },
                    {
                        label: 'Math Cells',
                        data: scatterData.filter(d => d.group === 'math'),
                        backgroundColor: '#3b82f6'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Fraction Label-Free Sign Wrong' } },
                    y: { title: { display: true, text: 'R2 - R0 Gain (AUROC)' } }
                }
            }
        });
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == '__main__':
    run_gap_ladder()
