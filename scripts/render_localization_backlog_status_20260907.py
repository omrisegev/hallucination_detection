"""Evidence-linked status snapshot; never presents pilot numbers as full results."""
import hashlib
import html
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/reviews/localization_backlog_and_full_benchmark_2026-09-07.html'
SOURCE=ROOT/'results/fusion_trajectory_imm_v1/EVALUATION.json'
E=json.loads(SOURCE.read_text(encoding='utf-8'))


def link(path,label):
    import os
    return '<a href="'+html.escape(os.path.relpath(ROOT/path,OUT.parent).replace('\\','/'))+'">'+html.escape(label)+'</a>'


methods=[
 ('dual__iu','IU reference','FEATURES'),
 ('dual__equal','Equal-weight reference','FEATURES'),
 ('dual__joint0','Original Joint, condition1000, lambda0','FEATURES'),
 ('dual__cond100','Joint, condition100, lambda0','Conditioning'),
 ('dual__cond100_graph010','Joint, condition100, graph0.1','Graph'),
 ('dual__equal_graph_perm','Equal + permuted graph','Simple graph control'),
 ('sample_risk_top__iu','Risk-selected windows + IU','TOKENS / fitting rows'),
 ('sample_risk_top__graph010','Risk-selected windows + Joint graph','TOKENS + graph'),
 ('sample_risk_top__equal_graph_perm','Risk-selected windows + equal + permuted graph','Strong simple control'),
 ('sample_dufs_transposed__iu','Transposed DUFS selection + IU','TOKENS / fitting rows'),
 ('traj_iu_joint_graph__mean','IU + Joint graph: static mean','Fusion of two curves'),
 ('traj_iu_joint_graph__gls','IU + Joint graph: GLS','Fusion of two curves'),
 ('traj_iu_joint_graph__imm','IU + Joint graph: IMM','Chronological state'),
 ('gap__iu','Token-gap representation + IU','Feature reparameterization'),
 ('ar1__iu','AR residual features + IU','Additional features'),
]
rows=[]
for key,label,axis in methods:
    m=E['metrics'][key]
    rows.append('<tr data-method="'+key+'"><td>'+html.escape(label)+'</td><td>'+axis+'</td>'+
        f'<td>{m["prm"]["auroc"]:.4f}</td><td>{m["prm"]["within_answer_auc"]:.4f}</td><td>{100*m["pb"]["macro_f1"]:.2f}%</td></tr>')
full=ROOT/'results/localization_full_benchmark_v3'
state=json.loads((full/'RUN_STATE.json').read_text()) if (full/'RUN_STATE.json').exists() else {'state':'PREPARING_NUMERICAL_REPLAY','completed':0,'total':13769}
snapshot=datetime.now(timezone.utc).isoformat(timespec='seconds')
calibration=ROOT/'results/fusion_gate_calibration_v1'
gate_state='The 384-trial calibration simulation has completed; review is pending.'
if (calibration/'REVIEW.json').exists():
    review=json.loads((calibration/'REVIEW.json').read_text())
    if review['status']=='PASS':
        result=json.loads((calibration/'RESULTS.json').read_text())
        passed=[k for k,v in result['advancement_screen'].items() if v['pass_all_cells']]
        gate_state='The 384-trial calibration simulation and its review are complete. '+('Screen passed for '+', '.join(passed)+'.' if passed else 'Neither raw nor IMM passed the fixed advancement screen.')
body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Localization: results, backlog and full benchmark</title>
<style>body{font:17px/1.6 system-ui,sans-serif;background:#f4f7fa;color:#172b40;margin:0}main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:34px;line-height:1.2}h2{margin-top:36px;font-size:24px}.card{padding:20px 24px;background:white;border:1px solid #d9e2ed;border-radius:12px;margin:18px 0}.notice{border-left:5px solid #b66714}.good{border-left:5px solid #227b6b}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:11px 12px;text-align:left;border-bottom:1px solid #dde5ed;vertical-align:top}th{background:#e8eef5}td:nth-last-child(-n+3){font-variant-numeric:tabular-nums}a{color:#176b9a}.scroll{overflow-x:auto}.muted{color:#506477;font-size:14px}.axis{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.axis .card{margin:0}code{font-size:14px}button{font:inherit;padding:8px 12px;cursor:pointer}@media print{body{background:white}main{padding:0}.card{break-inside:avoid}button{display:none}}</style>
<main><h1>What improved, what remains, and why the full benchmark comes next</h1>
<p class="muted">Snapshot: SNAPSHOT. A project status report, not a publication claim.</p>
<div class="card notice"><strong>No method has yet demonstrated a reproducible advantage on both localization benchmarks.</strong>
The recent development comparison uses only 24 PRMBench answers and 86 ProcessBench answers, all scored with Qwen3-8B.
The earlier 58-answer cohort is separate. Repeated controls, label bridges and diagnostic outputs are not independent algorithm discoveries.</div>
<h2>Actual changes inside the method</h2><div class="axis">
<div class="card"><strong>New features</strong><p>Added prediction-residual views (AR1, last value, EMA) and tested a provided-token confidence gap.
The gap replaces existing surprisal coordinates; it does not add an independent observation. No tested addition improved both tasks.
These use existing telemetry, with no new model inference.</p></div>
<div class="card"><strong>FEATURES axis</strong><p>Made the answer-only N-window by P-feature pipeline runnable, with moment/context banks,
entropy-based orientation and explicit IU fallback. Safer grouping improved numerical coverage; inverse conditioning helped Joint's own baseline.
These are useful implementation gains, but Joint has not overtaken IU on both tasks.</p></div>
<div class="card"><strong>TOKENS axis</strong><p>Tested uniform, entropy-risk, transposed DUFS, shuffled DUFS and diffusion selection.
Risk selection is the strongest lead for pooled PRMBench AUROC. It has not improved the IU or Joint first-error score on the replication.
Selection changes fitting rows; every window is still computed and scored.</p></div>
<div class="card"><strong>Graphs</strong><p>Tested real and node-permuted graphs, lambda and inverse-conditioning changes, with equal-weight controls.
Condition100 + graph raises Joint's own point estimates, but the permuted equal-weight control remains competitive.
A graph-specific benefit is not established.</p></div>
<div class="card"><strong>Chronological trajectory</strong><p>Combined IU and Joint curves using mean, GLS and IMM. GLS has a small positive point change on both tasks,
with uncertainty intervals crossing zero. IMM worsens exact first-error/no-error performance. HMM, ordinary Kalman and BOCPD were also tested earlier.
Ordinary Kalman is not KalmanNet.</p></div>
<div class="card"><strong>Evaluation and no-error decision</strong><p>Corrected PRMBench's one-based step-label mapping and source-group folds.
Separated wrong peak selection from incorrect no-error gating. Smoothing can open the GMM gate on a stationary synthetic source.
These repairs and diagnostics improve trust in the evidence; they are not new accuracy gains.</p></div></div>
<h2>Matched results we actually have</h2>
<p>Same corrected labels, same 110 answers and declared fallbacks. PRMBench pooled AUROC compares steps across answers;
within-answer AUROC measures ranking inside each mixed-label answer. ProcessBench is the four-cell Qwen3-8B macro of the harmonic
mean of clean-answer accuracy and exact first-error accuracy. Higher is better, but these columns measure different tasks.</p>
<div class="scroll"><table><thead><tr><th>Method</th><th>Change</th><th>PRMB pooled AUROC</th><th>PRMB within-answer AUROC</th><th>PB exact/no-error score</th></tr></thead><tbody>ROWS</tbody></table></div>
<div class="card notice"><strong>Read the risk-sampling result carefully.</strong>
IU pooled AUROC increases from .6813 to .7584, while within-answer AUROC changes from .7688 to .7789 and PB falls from30.16% to28.35%.
An affine diagnostic reproduced much of the pooled gain by changing answer-specific location/scale while preserving local ranking.
The strongest simple control reaches .7684 /33.92%, but its PB improvement over its own full-grid control has a95% interval of
[-3.29,+8.03] percentage points. This is a full-benchmark contender, not a confirmed winner.</div>
<p>SCORES_LINK · SAMPLING_LINK · TRAJECTORY_LINK · LABEL_LINK · GROUP_LINK</p>
<h2>The backlog, in priority order</h2><div class="scroll"><table>
<thead><tr><th>Priority</th><th>Work</th><th>What is still missing</th></tr></thead><tbody>
<tr><td>Now</td><td>Full matched benchmark</td><td>Run the existing anchors, fixed recent shortlist and historical leaders on shared corrected populations. Publish coverage, per-cell results, paired uncertainty and runtime. This requirement is not yet fulfilled.</td></tr>
<tr><td>Now</td><td>Historical methods and Claude refits</td><td>IU/U-PCR, LIU/DUFS-LIU, CONT/L-SML, Joint model-inverse/graph controls; dedicated localizers family6, GL-LIU, token-IU29, Unified28 and entropy/top5. Refit learned/calibrated quantities using corrected groups and labels. Keep extra-access PRMs/critics visible in their own panel.</td></tr>
<tr><td>After the full diagnosis</td><td>Keep or repair the strongest fusion candidate</td><td>Use per-dataset and per-answer failures to choose one change. Distinguish feature weights, local ranking, answer scale and no-error gate. Do not continue widening a graph/lambda grid based on one pooled score.</td></tr>
<tr><td>Open</td><td>Gate calibration</td><td>GATE_STATE It is not a real-answer method result. No real-data gate change is being launched from this status request; full benchmarking now takes priority.</td></tr>
<tr><td>Open</td><td>Sampling and N versus P</td><td>Jointly useful window width/row count, stability with longer traces, sparse end-to-end computation and retention of short errors. The sampling-eligible PB pilot had zero first errors of32 tokens or fewer, so it cannot answer that question.</td></tr>
<tr><td>Open</td><td>Supporting ideas</td><td>Actual LOCA, Diverging Flows, KalmanNet and Shlezinger-inspired designs tailored to support IU/Joint fusion. Prior temporal baselines do not close these mechanisms, and no new win from them has been demonstrated here.</td></tr>
<tr><td>Later</td><td>Untouched confirmation</td><td>Lock the method and decision rule before new, genuinely uninspected data. The full current cache is development data even if an answer was absent from the110 pilot.</td></tr>
<tr><td>Separate follow-up</td><td>Historical24 final-answer transfer</td><td>Evaluate the localization-selected fusion recipe under the old24-cell final-answer contract. Do not compare those response-level AUROCs directly with first-error localization.</td></tr>
</tbody></table></div>
<h2>Full-dataset measurement: concrete execution state</h2>
<div class="card good"><strong>13,769 model-answer rows are registered.</strong>
6,969 retained PRMBench answers plus3,400 PB answers scored with each of Qwen3-4B and Qwen3-8B.
These are not13,769 independent questions. The three existing PRMBench alignment exclusions remain declared.
All21 too-short rows stay in the population as explicit unsupported outputs;11 long PRMBench traces beyond the old pilot range are included.</div>
<p><strong>Snapshot state:</strong> STATE. COMPLETED /13,769 anchor records completed.
The first pass scores the unchanged19 original outputs from two bank fits and reuses110 frozen outputs after exact input checks.
This is the anchor pass, not completion of the newer-shortlist or historical-refit passes.
No full-population performance number is claimed here.</p>
<p>Local telemetry is already available; no new inference is needed. Three CPU workers use memory-mapped inputs and per-answer resumable checkpoints.
The run reads Claude's worktree and writes into a new result folder. It preserves the earlier scores.</p>
<p>PROTOCOL_LINK · REGISTRY_LINK · RUN_LINK · DRIVER_LINK</p>
<h2>Why old headline numbers cannot be the comparison column</h2>
<p>Claude's completed full experiment fitted across training answers, used label-selected tuning and calibrated thresholds, and averaged eight PB model/subset cells.
Our110 pilot fits each answer alone and uses four Qwen3-8B cells and a local GMM. The old PRMBench labels and source folds were subsequently corrected.
The historical family6/GL-LIU leaders also include a Llama scoring contract. They belong in the benchmark, but require a matched adapter/refit;
printing their old headlines next to this pilot would not measure improvement.</p>
<p>The practical optimization is to stop adding small sweeps for now: reuse telemetry and frozen scores, finish the fixed full comparison,
then choose the next short experiment from its failures. Fusion remains the method being developed.</p>
<p class="muted">Source evaluation SHA256: SOURCE_HASH. Snapshot content is generated from frozen metric fields. No external review or browser rendering is claimed.</p>
<button onclick="window.print()">Print / save PDF</button></main></html>'''
replacements={'SNAPSHOT':snapshot,'ROWS':''.join(rows),'GATE_STATE':gate_state,'STATE':html.escape(state['state']),'COMPLETED':str(state['completed']),
 'SOURCE_HASH':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'SCORES_LINK':link('results/fusion_trajectory_imm_v1/EVALUATION.json','Frozen metrics'),
 'SAMPLING_LINK':link('results/fusion_sampling_replication_v1/REPORT.html','Sampling review'),
 'TRAJECTORY_LINK':link('results/fusion_trajectory_imm_v1/REPORT.html','Trajectory review'),
 'LABEL_LINK':link('results/localization_prm_label_audit_v1/REPORT.html','Label repair'),
 'GROUP_LINK':link('results/localization_source_group_audit_v1/REPORT.html','Source-group repair'),
 'PROTOCOL_LINK':link('docs/experiments/LOCALIZATION_FULL_BENCHMARK_V3.md','Full benchmark protocol'),
 'REGISTRY_LINK':link('results/localization_full_benchmark_v3/METHOD_REGISTRY.json','Method registry with pending entries'),
 'RUN_LINK':link('results/localization_full_benchmark_v3/RUN_STATE.json','Live run-state file'),
 'DRIVER_LINK':link('scripts/run_localization_full_benchmark_v3.py','Full anchor driver')}
for key,value in replacements.items():body=body.replace(key,value)
OUT.write_text(body,encoding='utf-8')
print(OUT)
