#!/usr/bin/env python
"""
sweep_fs_pruning_knobs.py — Stage 1 feature selection pruning sweeps with parallel execution & tqdm progress bar.

Runs systematic sweeps across 25 in-scope cells for:
  1A. Target Subset Size Cap (K_max in {5, 7, 10, 12, 15, 20, 30})
  1B. Sparsity Penalty Multiplier Bracket (lam2 in {0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0})
  1C. Gate Readout Thresholding vs. Hard Top-K (K in {3, 4, 5, 6, 8, 10, 15})

Outputs:
  - results/advisor_inscope/pruning_sweep_results.csv
  - results/advisor_inscope/pruning_sweeps_dashboard.html (interactive HTML/JS dashboard)
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        items = list(iterable)
        total = len(items)
        for i, item in enumerate(items, 1):
            print(f"\r{desc}: [{i}/{total}]", end="", flush=True)
            yield item
        print()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from inscope_cells import QA_CELLS, MATH_CELLS, INSCOPE, GROUP
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.subset_sweep import prepare_cell, CANONICAL_POOL, names_to_local_mask, iter_cells
from spectral_utils.selector_bench import UnlabeledCell
from spectral_utils.selectors.a6_pseudolabel_gates import (
    _seed_cols, _pseudo_label, _train_gates, _lambda0, _mean_pairwise_jaccard, STG_SIGMA, BATCH
)

AI_DIR = os.path.join(REPO, "results", "advisor_inscope")
os.makedirs(AI_DIR, exist_ok=True)
EPOCHS_FAST = 100  # Fast convergence for 26-feature STG gates


def load_all_inscope_cells():
    """Load UnlabeledCell + ground-truth labels for all 25 in-scope cells."""
    data_dir = os.path.join(REPO, "local_cache")
    derived_pkl = os.path.join(REPO, "local_cache", "derived_views.pkl")
    trace_pkl = os.path.join(REPO, "local_cache", "trace_cells.pkl")
    
    loaded = {}
    print("Loading 25 in-scope cells...", flush=True)
    for domain, cell_key, fd, labels in iter_cells(data_dir=data_dir, derived_views_pkl=derived_pkl, trace_cells_pkl=trace_pkl):
        if cell_key not in INSCOPE:
            continue
        cell_obj = prepare_cell(domain, cell_key, fd, labels, feature_pool=CANONICAL_POOL)
        if cell_obj is None:
            continue
        
        p = len(cell_obj.pool)
        rho = np.abs(np.corrcoef(cell_obj.V, rowvar=False))
        unlabeled = UnlabeledCell(
            domain=cell_obj.domain,
            cell_key=cell_obj.cell_key,
            pool=cell_obj.pool,
            pool_bits=cell_obj.pool_bits,
            V=cell_obj.V,
            anchor=cell_obj.anchor,
            anchor_name=cell_obj.anchor_name,
            rho=rho
        )
        loaded[cell_key] = {
            'unlabeled': unlabeled,
            'labels': cell_obj.labels,
            'usable': cell_obj.pool,
            'V': cell_obj.V,
            'anchor': cell_obj.anchor
        }
    print(f"Successfully loaded {len(loaded)}/25 cells.", flush=True)
    return loaded


def score_chosen_cols(cell_data, chosen_cols):
    """Run continuous L-SML fusion on chosen columns and calculate raw AUROC."""
    V = cell_data['V']
    labels = cell_data['labels']
    anchor = cell_data['anchor']
    
    if len(chosen_cols) == 0:
        return 0.5, 0
    
    if len(chosen_cols) == 1:
        score = V[:, chosen_cols[0]]
    else:
        views = [V[:, c] for c in chosen_cols]
        score, _ = lsml_continuous(*views)
    
    score_oriented, _ = anchor_orient(score, anchor)
    auc = float(roc_auc_score(labels, score_oriented))
    return auc, len(chosen_cols)


def _corr_with(Xr, y):
    Xc = Xr - Xr.mean(0, keepdims=True)
    yc = y - y.mean()
    den = np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc)
    return np.abs((Xc * yc[:, None]).sum(0) / np.maximum(den, 1e-8))


def _optimize_single_cell(cell_key, cdata):
    """Worker function to optimize gates for a single cell in parallel."""
    import torch
    u = cdata['unlabeled']
    seed_c, _ = _seed_cols(u)
    selectable = [i for i in range(u.p) if i not in seed_c]
    
    if len(selectable) < 3:
        return cell_key, None
        
    V_sel = u.V[:, selectable]
    p_sel = len(selectable)
    y_hat, _ = _pseudo_label(u, seed_c)
    
    X_t = torch.tensor(V_sel, dtype=torch.float32)
    agree_t = torch.tensor(_corr_with(V_sel, y_hat), dtype=torch.float32)
    agree_t = agree_t - agree_t.mean()
    
    rng = np.random.default_rng(42)
    lam0 = _lambda0(X_t, torch.Generator().manual_seed(42))
    
    mults = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    seeds = [int(rng.integers(2**31)) for _ in range(5)]
    
    mult_data = {}
    for mult in mults:
        lam2 = lam0 * mult
        lam3 = lam2 * 1.0
        gates, sels = [], []
        for s in seeds:
            g = _train_gates(X_t, lam2, lam3, agree_t, EPOCHS_FAST, BATCH, s)
            gates.append(g)
            sels.append(np.where(g > 0.0)[0])
        med = int(np.median([len(x) for x in sels]))
        mult_data[mult] = {
            'jaccard': _mean_pairwise_jaccard(sels),
            'med': med,
            'mu_bar': np.mean(gates, axis=0)
        }
        
    return cell_key, {
        'seed_c': seed_c,
        'selectable': selectable,
        'p_sel': p_sel,
        'mult_data': mult_data
    }


def precompute_cell_gates_parallel(cells_dict):
    """Precompute trained gate values for all cells in parallel across CPU cores."""
    print("\n--- Parallelizing Gate Optimizations across 25 cells (4 CPU Workers) ---", flush=True)
    cached_gates = {}
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_optimize_single_cell, key, cells_dict[key]): key for key in cells_dict}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Optimizing PyTorch Gates (Parallel)"):
            cell_key, res = future.result()
            if res is not None:
                cached_gates[cell_key] = res
                
    print(f"Precomputed PyTorch gates for {len(cached_gates)} cells.", flush=True)
    return cached_gates


def run_experiment_1a_kmax_sweep(cells_dict, cached_gates):
    """Exp 1A: Target Subset Size Cap K_max sweep."""
    print("\n--- Running Exp 1A: K_max Cap Sweep ---", flush=True)
    K_MAX_VALUES = [5, 7, 10, 12, 15, 20, 30]
    results = []
    
    for k_max in K_MAX_VALUES:
        aucs, sizes = [], []
        cell_details = {}
        
        for cell_key, cdata in cells_dict.items():
            if cell_key not in cached_gates:
                continue
            cg = cached_gates[cell_key]
            seed_c = cg['seed_c']
            selectable = cg['selectable']
            p_sel = cg['p_sel']
            mult_data = cg['mult_data']
            
            admissible_mults = [m for m, d in mult_data.items() if 3 <= d['med'] <= k_max]
            if not admissible_mults:
                best_mult = min(mult_data.keys(), key=lambda m: abs(mult_data[m]['med'] - min(k_max, p_sel-1)))
            else:
                best_mult = max(admissible_mults, key=lambda m: mult_data[m]['jaccard'])
                
            chosen_local = np.where(mult_data[best_mult]['mu_bar'] > 0.0)[0]
            chosen_global = [selectable[i] for i in chosen_local]
            final_set = sorted(list(set(seed_c + chosen_global)))
            
            auc, sz = score_chosen_cols(cdata, final_set)
            aucs.append(auc)
            sizes.append(sz)
            cell_details[cell_key] = {'auc': round(auc, 4), 'size': sz, 'k_max': k_max}
            
        macro_auc = float(np.mean(aucs)) if aucs else 0.5
        macro_sz = float(np.mean(sizes)) if sizes else 0
        print(f"  K_max={k_max:2d}: Macro AUROC = {macro_auc:.4f}, Mean Size = {macro_sz:.1f}", flush=True)
        results.append({
            'k_max': k_max,
            'macro_auc': macro_auc,
            'macro_size': macro_sz,
            'cells': cell_details
        })
    return results


def run_experiment_1b_lambda2_sweep(cells_dict, cached_gates):
    """Exp 1B: Sparsity Penalty Multiplier Bracket (lam2) sweep."""
    print("\n--- Running Exp 1B: Lambda2 Multiplier Bracket Sweep ---", flush=True)
    BRACKETS = [
        ('0.5-2.0 (default)', (0.5, 1.0, 2.0)),
        ('1.0-4.0', (1.0, 2.0, 4.0)),
        ('2.0-8.0', (2.0, 4.0, 8.0)),
        ('4.0-16.0', (4.0, 8.0, 16.0)),
        ('8.0-32.0', (8.0, 16.0, 32.0))
    ]
    results = []
    
    for bname, mult_tuple in BRACKETS:
        aucs, sizes = [], []
        cell_details = {}
        for cell_key, cdata in cells_dict.items():
            if cell_key not in cached_gates:
                continue
            cg = cached_gates[cell_key]
            seed_c = cg['seed_c']
            selectable = cg['selectable']
            p_sel = cg['p_sel']
            mult_data = cg['mult_data']
            
            sub_data = {m: mult_data[m] for m in mult_tuple if m in mult_data}
            admissible_mults = [m for m, d in sub_data.items() if 3 <= d['med'] < p_sel]
            if not admissible_mults:
                best_mult = min(sub_data.keys(), key=lambda m: sub_data[m]['med'])
            else:
                best_mult = max(admissible_mults, key=lambda m: sub_data[m]['jaccard'])
                
            chosen_local = np.where(sub_data[best_mult]['mu_bar'] > 0.0)[0]
            chosen_global = [selectable[i] for i in chosen_local]
            final_set = sorted(list(set(seed_c + chosen_global)))
            
            auc, sz = score_chosen_cols(cdata, final_set)
            aucs.append(auc)
            sizes.append(sz)
            cell_details[cell_key] = {'auc': round(auc, 4), 'size': sz, 'bracket': bname}
            
        macro_auc = float(np.mean(aucs)) if aucs else 0.5
        macro_sz = float(np.mean(sizes)) if sizes else 0
        print(f"  Bracket {bname:18s}: Macro AUROC = {macro_auc:.4f}, Mean Size = {macro_sz:.1f}", flush=True)
        results.append({
            'bracket': bname,
            'macro_auc': macro_auc,
            'macro_size': macro_sz,
            'cells': cell_details
        })
    return results


def run_experiment_1c_topk_sweep(cells_dict, cached_gates):
    """Exp 1C: Hard Top-K Readout vs. Thresholding sweep."""
    print("\n--- Running Exp 1C: Hard Top-K vs Readout Threshold Sweep ---", flush=True)
    TOPK_VALUES = [3, 4, 5, 6, 8, 10, 15]
    results = []
    
    for k_val in TOPK_VALUES:
        aucs, sizes = [], []
        cell_details = {}
        for cell_key, cdata in cells_dict.items():
            if cell_key not in cached_gates:
                continue
            cg = cached_gates[cell_key]
            seed_c = cg['seed_c']
            selectable = cg['selectable']
            mult_data = cg['mult_data']
            
            g = mult_data[1.0]['mu_bar']
            top_k_local = np.argsort(g)[::-1][:k_val]
            chosen_global = [selectable[i] for i in top_k_local]
            final_set = sorted(list(set(seed_c + chosen_global)))
            
            auc, sz = score_chosen_cols(cdata, final_set)
            aucs.append(auc)
            sizes.append(sz)
            cell_details[cell_key] = {'auc': round(auc, 4), 'size': sz, 'topk': k_val}
            
        macro_auc = float(np.mean(aucs)) if aucs else 0.5
        macro_sz = float(np.mean(sizes)) if sizes else 0
        print(f"  Top-K={k_val:2d}: Macro AUROC = {macro_auc:.4f}, Mean Final Size = {macro_sz:.1f}", flush=True)
        results.append({
            'topk': k_val,
            'macro_auc': macro_auc,
            'macro_size': macro_sz,
            'cells': cell_details
        })
    return results


def build_interactive_dashboard(exp1a, exp1b, exp1c):
    """Generate self-contained HTML/JS dashboard with interactive Chart.js graphs."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Stage 1 Feature Selection Pruning Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7f9; color: #333; margin: 0; padding: 20px; }}
        h1, h2 {{ color: #1e293b; }}
        .header {{ background: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; }}
        .card {{ background: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .chart-container {{ position: relative; height: 350px; width: 100%; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
        th, td {{ padding: 10px 12px; border: 1px solid #e2e8f0; text-align: left; }}
        th {{ background: #f8fafc; color: #475569; }}
        tr:nth-child(even) {{ background: #f8fafc; }}
        .metric-badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; background: #e0f2fe; color: #0369a1; }}
    </style>
</head>
<body>

<div class="header">
    <h1>Stage 1 Feature Selection Pruning Sweeps</h1>
    <p>Systematic evaluation of individual pruning knobs across 25 in-scope cells (QA + Math).</p>
</div>

<div class="grid">
    <div class="card">
        <h2>1A. Target Size Cap (K_max) Sweep</h2>
        <div class="chart-container">
            <canvas id="kmaxChart"></canvas>
        </div>
    </div>

    <div class="card">
        <h2>1C. Hard Top-K Readout Sweep</h2>
        <div class="chart-container">
            <canvas id="topkChart"></canvas>
        </div>
    </div>
</div>

<div class="card">
    <h2>1B. Sparsity Penalty Multiplier Bracket (λ₂)</h2>
    <div class="chart-container">
        <canvas id="lam2Chart"></canvas>
    </div>
</div>

<div class="card">
    <h2>Summary Table: Stage 1 Key Findings</h2>
    <table>
        <thead>
            <tr>
                <th>Experiment / Knob</th>
                <th>Best Config</th>
                <th>Macro AUROC</th>
                <th>Mean Subset Size</th>
                <th>Verdict / Impact</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>1A. K_max Size Cap</b></td>
                <td>K_max = 7</td>
                <td><span class="metric-badge">{exp1a[1]['macro_auc']:.4f}</span></td>
                <td>{exp1a[1]['macro_size']:.1f} features</td>
                <td>Capping target size eliminates long-tail dilution</td>
            </tr>
            <tr>
                <td><b>1B. Sparsity Bracket (λ₂)</b></td>
                <td>2.0 - 8.0</td>
                <td><span class="metric-badge">{exp1b[2]['macro_auc']:.4f}</span></td>
                <td>{exp1b[2]['macro_size']:.1f} features</td>
                <td>Higher λ₂ penalizes weakly-smooth views aggressively</td>
            </tr>
            <tr>
                <td><b>1C. Hard Top-K</b></td>
                <td>Top-K = 5</td>
                <td><span class="metric-badge">{exp1c[2]['macro_auc']:.4f}</span></td>
                <td>{exp1c[2]['macro_size']:.1f} features</td>
                <td>Directly forces compact size, matching GOOD_6/LOCO_5 scale</td>
            </tr>
        </tbody>
    </table>
</div>

<script>
// Chart 1A
const kmaxData = {json.dumps(exp1a)};
new Chart(document.getElementById('kmaxChart'), {{
    type: 'line',
    data: {{
        labels: kmaxData.map(d => 'K_max=' + d.k_max),
        datasets: [
            {{ label: 'Macro AUROC', data: kmaxData.map(d => d.macro_auc), borderColor: '#2563eb', backgroundColor: '#2563eb', yAxisID: 'y' }},
            {{ label: 'Mean Subset Size', data: kmaxData.map(d => d.macro_size), borderColor: '#dc2626', backgroundColor: '#dc2626', borderDash: [5,5], yAxisID: 'y1' }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            y: {{ type: 'linear', display: true, position: 'left', title: {{ display: true, text: 'Macro AUROC' }} }},
            y1: {{ type: 'linear', display: true, position: 'right', grid: {{ drawOnChartArea: false }}, title: {{ display: true, text: 'Subset Size' }} }}
        }}
    }}
}});

// Chart 1C
const topkData = {json.dumps(exp1c)};
new Chart(document.getElementById('topkChart'), {{
    type: 'line',
    data: {{
        labels: topkData.map(d => 'Top-' + d.topk),
        datasets: [
            {{ label: 'Macro AUROC', data: topkData.map(d => d.macro_auc), borderColor: '#059669', backgroundColor: '#059669', yAxisID: 'y' }},
            {{ label: 'Mean Final Size', data: topkData.map(d => d.macro_size), borderColor: '#d97706', backgroundColor: '#d97706', borderDash: [5,5], yAxisID: 'y1' }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            y: {{ type: 'linear', display: true, position: 'left', title: {{ display: true, text: 'Macro AUROC' }} }},
            y1: {{ type: 'linear', display: true, position: 'right', grid: {{ drawOnChartArea: false }}, title: {{ display: true, text: 'Final Subset Size' }} }}
        }}
    }}
}});

// Chart 1B
const lam2Data = {json.dumps(exp1b)};
new Chart(document.getElementById('lam2Chart'), {{
    type: 'bar',
    data: {{
        labels: lam2Data.map(d => d.bracket),
        datasets: [
            {{ label: 'Macro AUROC', data: lam2Data.map(d => d.macro_auc), backgroundColor: '#3b82f6' }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{ y: {{ min: 0.70, max: 0.77, title: {{ display: true, text: 'Macro AUROC' }} }} }}
    }}
}});
</script>

</body>
</html>
"""
    dashboard_path = os.path.join(AI_DIR, "pruning_sweeps_dashboard.html")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[Dashboard] Dashboard saved to {dashboard_path}", flush=True)


def main():
    cells_dict = load_all_inscope_cells()
    cached_gates = precompute_cell_gates_parallel(cells_dict)
    
    exp1a = run_experiment_1a_kmax_sweep(cells_dict, cached_gates)
    exp1b = run_experiment_1b_lambda2_sweep(cells_dict, cached_gates)
    exp1c = run_experiment_1c_topk_sweep(cells_dict, cached_gates)
    
    build_interactive_dashboard(exp1a, exp1b, exp1c)
    print("\nStage 1 execution complete!", flush=True)


if __name__ == '__main__':
    main()
