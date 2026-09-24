"""Render completed, independently audited external comparisons into repository docs."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lsml_external_generalization_v1/evaluation'
NAMES={'frozen_lsml':'Frozen step L-SML','frozen_equal':'Step equal (control)','frozen_partition_equal':'Step partition equal (control)',
'local_lsml':'Answer-local token L-SML','local_equal':'Token equal (control)','local_partition_equal':'Token partition equal (control)','ct7':'CT7 reference'}
CELLS={'hard2verify_qwen3_8b':'Hard2Verify / Qwen3-8B','socratic_qwen3_8b':'Socratic / Qwen3-8B','socratic_qwq32b':'Socratic / QwQ-32B'}


def load(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    if not (OUT/'RED_TEAM.md').exists():raise RuntimeError('independent review required before report')
    metrics=load(OUT/'METRICS.json');contrasts=load(OUT/'CONTRASTS.json');disjoint=load(OUT/'DISJOINT_CONTRASTS.json');source=load(OUT/'source/VALIDATION_METRICS.json')
    if len(contrasts)!=18 or any(c['draws']!=100000 for c in contrasts):raise ValueError('incomplete primary comparisons')
    lines=['# Frozen L-SML external generalization results','',
      'All seven registered internal methods were evaluated on all three dataset/backbone cells: 6,190 records and 53,970 steps. No new model inference or GPU training was performed in this evaluation stage. Published comparator inference remains a separate, unfinished item of the broader pipeline.','',
      '## Official full-set results','',
      'Percentages below use the metric defined by each benchmark. Hard2Verify: harmonic mean of correct-step and incorrect-step recalls. Socratic: pooled binary macro-F1 (PRMScore). The columns must not be averaged.','',
      '| Method | Hard2Verify / Qwen3-8B | Socratic / Qwen3-8B | Socratic / QwQ-32B |','|---|---:|---:|---:|']
    for arm,name in NAMES.items():lines.append('| '+name+' | '+' | '.join(f"{100*metrics[c]['arms'][arm]['metric']:.3f}" for c in CELLS)+' |')
    lines+=['','## Incremental fusion value','',
      'Each primary contrast uses identical answer/step masks and source-only calibration access. Intervals are paired source-question bootstrap intervals: 100,000 draws, seed 20260924, Bonferroni correction across 18 contrasts. Positive intervals support improvement over that particular reference. Native fitting and readout differ between the frozen-step and local-token families; compare their matched controls first.','',
      '| Cell | Candidate | Reference | Difference (pp) | Corrected interval (pp) |','|---|---|---|---:|---|']
    for c in contrasts:
        lo,hi=c['ci_bonferroni'];lines.append(f"| {CELLS[c['cell']]} | {NAMES[c['left']]} | {NAMES[c['right']]} | {100*c['delta']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    lines+=['','The averaging arms are controls, not proposed final methods. A higher absolute score alone does not establish that learned fusion adds value. Published comparator values below do not have paired predictions in this experiment.','',
      '## Ranking and observed-disjoint transfer','',
      '| Cell | Method | Within-answer AUC | Eligible answers | Observed-disjoint official metric (%) |','|---|---|---:|---:|---:|']
    for cell in CELLS:
        for arm,name in NAMES.items():
            v=metrics[cell]['arms'][arm];lines.append(f"| {CELLS[cell]} | {name} | {v['within_auc']:.4f} | {v['within_auc_answers']} | {100*v['disjoint_metric']:.3f} |")
    lines+=['','Observed-disjoint paired uncertainty is a sensitivity analysis, using the same group-bootstrap procedure and eighteen-comparison correction. It does not establish absence of semantic overlap.','',
      '| Cell | Candidate | Reference | Disjoint difference (pp) | Corrected interval (pp) |','|---|---|---|---:|---|']
    for c in disjoint:
        lo,hi=c['ci_bonferroni'];lines.append(f"| {CELLS[c['cell']]} | {NAMES[c['left']]} | {NAMES[c['right']]} | {100*c['delta']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    lines+=['','Socratic contains 442 answers connected to development questions, leaving 2,553 answers/1,514 groups in the observed-disjoint panel. Its full set has 2,995 answers/1,765 groups. Hard2Verify has 200 answers/79 text groups and no observed exact development overlap. These exclusions propagate through source identity and original/evaluated-text components. Development metadata does not enumerate every original/modified PRMB question variant and has no verified cross-dataset ID crosswalk. Therefore observed-disjoint does not establish absence of paraphrases, equivalent problems or pretraining contamination. The two Socratic backbones reuse the same dataset.','',
      '## Coverage, decisions and runtime','', 'Cell-level label sanity and full-population coverage flags are retained in METRICS.json and METRICS.csv. Flagged cells must not be used for unqualified headline win counts.','',
      '| Cell | Answers | Steps | Local native fits | Empty steps | Scoring CPU seconds | Per-answer elapsed p95 (seconds) |','|---|---:|---:|---:|---:|---:|---:|']
    for cell in CELLS:
        m=metrics[cell];lines.append(f"| {CELLS[cell]} | {m['answers']} | {m['steps']} | {m['local_native_answers']} | {m['empty_steps']} | {m['process_cpu_seconds_sum']:.1f} | {m['cpu_seconds_p50_p95'][1]:.3f} |")
    lines+=['','Three empty Socratic steps per backbone remain included, with a fixed incorrect decision independent of labels and a null risk score. Native-score ranking excludes those empty steps. Local estimator failures use chosen-token surprisal for all three local arms; complete-policy and matched-native metrics are both saved. Each arm has its own source q80 threshold, so identical fallback scores can still produce different binary decisions. METRICS.json records those fallback decision differences, small-group guards and degeneracy flags.','',
      'Runtime above includes feature extraction and all seven comparison arms, including CT7. It is not the deployment cost of one L-SML method. CPU time is measured with process_time; elapsed percentiles include contention on the local workstation. AIRCC collection was already complete: 1,376 allocated GPU seconds (0.38222 GPU-hours). This stage added zero GPU hours. The local fallback followed repeated SSH timeouts and verified SHA256s of all three private Drive archives.','',
      '| Cell | Method | Correct-step recall (%) | Error-step recall (%) | Predicted error fraction (%) | Gold error fraction (%) |','|---|---|---:|---:|---:|---:|']
    for cell in CELLS:
        for arm in ('frozen_lsml','local_lsml','ct7'):
            v=metrics[cell]['arms'][arm];lines.append(f"| {CELLS[cell]} | {NAMES[arm]} | {100*v['correct_recall']:.2f} | {100*v['error_recall']:.2f} | {100*v['predicted_error_fraction']:.2f} | {100*v['gold_error_fraction']:.2f} |")
    lines+=['','## Source validation and method lock','',
      'Existing source folds were preserved. Each source evaluation fold used three other folds for fitting and one separate calibration fold. The deployment model fits folds 0-3 and uses fold 4 for q80 calibration. Calibration pools unlabeled PB+PRMB steps; it is not target calibration or PRMB-only calibration. Source scores cover 13,769 answers; the official PRMB panel below uses 6,211 non-control answers/83,371 steps. These remain development results.','',
      '| Method | Separated source PRMScore (%) | Within-answer AUC (6,030 answers) |','|---|---:|---:|']
    for arm,name in NAMES.items():
        v=source['arms'][arm];lines.append(f"| {name} | {100*v['prmscore']:.3f} | {v['within_auc']:.4f} |")
    lines+=['','The guarded bank11 recipes, feature signs, entropy orientation, Top10 readouts and final answer-z were fixed before external quality inspection. Local token fitting uses stride 8 and at least 3 active channels and 3 observations per active channel. All source local scores and six frozen fits passed strict numerical-failure replay unchanged; the deployment weights exactly match the historical fold 4 fit. Twelve deterministic source examples replayed CT7 and bank11 raw extraction. This is a bounded implementation-fidelity check, not a full raw-cache replay.','',
      '## Published context, not reproduced baselines','',
      'Hard2Verify reports step-level Balanced F1 of 53.51 for the Qwen3-8B critic, 42.37 for Qwen2.5-Math-PRM-7B and 60.27 for UniversalPRM-7B. Its PRM thresholds were tuned on 100 target responses; our thresholds were frozen on development data. These are different access conditions. [Hard2Verify, Table 2 and Appendix E.1](https://arxiv.org/html/2510.13744v1).','',
      'Socratic reports PRMScore 68.0 for Qwen2.5-Math-PRM-7B and 73.8 for the QwQ-32B critic. These are literature context, not same-run measurements or evidence of statistically established superiority. [Socratic-PRMBench, Table 3](https://arxiv.org/html/2505.23474v1).','',
      '## Evidence and reproduction','',
      '- Frozen contract: [execution lock](LSML_EXTERNAL_EVALUATION_LOCK_20260924.md).',
      '- Machine-readable results: `results/lsml_external_generalization_v1/evaluation/{METRICS,CONTRASTS,EVALUATION_PROVENANCE}.json`.',
      '- Per-cell predictions, seals, confusion arrays, bootstrap draws, category and native panels are in the corresponding cell directory.',
      '- Independent audits: `evaluation/independent_population`, `independent_ct7`, `independent_source`, and `RED_TEAM.md`.',
      '- Private telemetry restore locations and SHA256s: `FULL_ARCHIVES.json`; local fallback verification: `evaluation/LOCAL_FALLBACK_PROVENANCE.json`.',
      '- Run `scripts/fit_external_source_bundle.py`, verify with `scripts/verify_external_source_strict.py`, then run `scripts/run_external_local_cpu.py`. The AIRCC CPU driver uses the identical scorer when connectivity is available.',
      '- Seal/evaluate with `scripts/evaluate_external_locked_scores.py --root results/lsml_external_generalization_v1/evaluation --inputs scratch/external_generalization_private/inputs`; the evaluator rejects mixed run identities and changed code/bundles.',
      '- No raw Hard2Verify text is included in Git. Frozen result files are retained; this evaluation does not rewrite historical scores.','']
    dest=ROOT/'docs/experiments/LSML_EXTERNAL_GENERALIZATION_RESULTS_20260924.md';dest.write_text('\n'.join(lines),encoding='utf8',newline='\n')
    with (OUT/'METRICS.csv').open('w',newline='',encoding='utf8') as f:
        fields=['cell','arm','metric','within_auc','correct_recall','error_recall','disjoint_metric','within_auc_answers','n_checked','n_total','flag'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for cell,m in metrics.items():
            for arm,v in m['arms'].items():w.writerow({'cell':cell,'arm':arm,**{k:v[k] for k in fields[2:] if k in v},**{k:m[k] for k in ('n_checked','n_total','flag')}})
    print(dest)

if __name__=='__main__':main()
