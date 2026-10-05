"""F11/F12 development comparison for the four frozen Monday candidates.

--freeze writes only protocol and input/code manifests. --run evaluates cached
step scores; it does not fit models or modify earlier run directories.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT = ROOT / 'results/monday_fusion_competitor_extension_v1/run_20261006'
CODE = ROOT / 'results/monday_fusion_analysis_v1/code'
PANEL = ROOT / 'results/monday_candidate_panel_v1/run_20261004'
TCI = ROOT / 'results/mom_filter_within_diagnostic_20261004/tci_wide_20261005'
TCI_ROOT = ROOT / 'results/mom_filter_within_diagnostic_20261004'
INPUT_RUN = ROOT / 'results/estimator_input_v1/run_20261004_resume'
STEP_EVIDENCE = ROOT / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
RULES = ['ANSWER_Z_Q80', 'R2_COUNT', 'ENTROPY_GATE_Q80']
TOPS = ['TOP_1', 'TOP_2', 'TOP_3']
METHODS = {
    'HEM_MOM_cc': ('channels_cell', 'HEM_MOM'),
    'MEAN_MOM_cc': ('channels_cell', 'MEAN_MOM'),
    'MEAN_MOM_ga': ('groups_answer', 'MEAN_MOM'),
    'TCI_ONLY_ga': ('tci', 'TCI_ONLY_ga'),
}
METRIC_FIELDS = ['within_answer_auc', 'prmscore', 'pb_first_error_accuracy',
                 'pb_decision_error_accuracy', 'pb_official_f1', 'clean_flag_rate',
                 'wrong_answer_auroc']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n', encoding='utf8')
    tmp.replace(path)


def collect_inputs() -> dict[str, Path]:
    """Pin every score and frozen source used by the extension."""
    paths: dict[str, Path] = {}
    for p in sorted((PANEL / 'candidate_scores').glob('*.npz')):
        paths['candidate_score/' + p.name] = p
    for p in sorted((TCI / 'scores').glob('*.npz')):
        paths['tci_score/' + p.name] = p
    for name, p in {
        'candidate_panel_metrics': PANEL / 'CONTENT_METRICS.csv',
        'candidate_panel_validation': PANEL / 'AUTOMATED_VALIDATION.json',
        'tci_stage2_metrics': TCI / 'STAGE2_CELL_METRICS.csv',
        'tci_stage2_gates': TCI / 'STAGE2_GATES.csv',
        'tci_triplet_choices': TCI_ROOT / 'TCI_TRIPLET_SELECTION.csv',
        'tci_calibration': TCI_ROOT / 'TCI_CALIBRATION.json',
        'tci_research_note': TCI_ROOT / 'RESEARCH_NOTE_HE.md',
        'input_manifest': INPUT_RUN / 'INPUT_MANIFEST.json',
        'raw_input_manifest': ROOT / 'results/monday_fusion_analysis_v1/RAW_INPUT_MANIFEST_V3.json',
        'population_answers': STEP_EVIDENCE / 'OOF_ANSWERS.csv',
        'population_step_scores': STEP_EVIDENCE / 'OOF_STEP_SCORES.npz',
        'population_freeze': STEP_EVIDENCE / 'INPUT_FREEZE.json',
        'mindgap_freeze': ROOT / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/INPUT_FREEZE.json',
    }.items():
        paths[name] = p
    # The panel manifest lists PRM metadata and answer features; the input manifests
    # list all normalized banks and the raw sources used for the R2 rule.
    for label, manifest_path in [('model_input', INPUT_RUN / 'INPUT_MANIFEST.json'),
                                 ('raw_input', ROOT / 'results/monday_fusion_analysis_v1/RAW_INPUT_MANIFEST_V3.json')]:
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        sections = manifest.get('inputs', {}) if label == 'model_input' else manifest.get('additional_inputs', {})
        for key, spec in sections.items():
            if isinstance(spec, dict) and spec.get('path'):
                paths[f'{label}/{key}'] = Path(spec['path'])
    freeze = json.loads((STEP_EVIDENCE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
    for key in ('oof_answers', 'oof_step_scores', 'prm_metadata'):
        spec = freeze.get(key)
        if isinstance(spec, dict) and spec.get('path'):
            paths['population/' + key] = Path(spec['path'])
    for key in ('mindgap',):
        spec = freeze.get(key)
        if isinstance(spec, dict) and spec.get('path'):
            paths['comparator/' + key] = Path(spec['path'])
    # Ensure the PRM metadata referenced by the freeze is explicit.
    spec = freeze.get('prm_metadata', {})
    if spec.get('path'):
        paths['population/prm_metadata'] = Path(spec['path'])
    return paths


def freeze() -> None:
    protocol_path = OUT / 'PROTOCOL.json'
    if protocol_path.exists():
        generated = ['METRICS.csv', 'F11_PRM_CATEGORY_METRICS.csv', 'F11_PRM_OVERLAP.csv.gz',
                     'F11_PRM_OVERLAP_SUMMARY.csv', 'F11_ERROR_COUNTS.csv',
                     'F12_MTG_ANSWER_FORENSICS.csv.gz', 'F12_MTG_BREAKDOWN.csv', 'CONTRASTS.csv',
                     'REPORT.md', 'RUN_STATUS.json']
        if any((OUT / name).exists() for name in generated) or (OUT / 'figures').exists():
            raise RuntimeError('Results already exist; refusing to replace a frozen protocol')
    required = collect_inputs()
    missing = [f'{k}: {p}' for k, p in required.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))
    branch = os.popen(f'git -C "{ROOT}" branch --show-current').read().strip()
    head = os.popen(f'git -C "{ROOT}" rev-parse HEAD').read().strip()
    protocol = {
        'schema': 'monday-fusion-competitor-extension-v1',
        'status': 'FROZEN_BEFORE_EVALUATION',
        'frozen_local_date': '2026-10-06',
        'branch': branch,
        'code_base_commit': head,
        'scope': 'Development comparison on already exposed PRMBench/ProcessBench population; not untouched confirmation.',
        'question': 'Do the four registered candidate methods localize annotated PRM errors and ProcessBench first errors differently from the supervised PRM and the frozen Mind the Gap reproduction?',
        'candidates': [
            {'id': 'HEM_MOM_cc', 'source': 'Monday candidate panel; HEM within-group weights, MoM between groups, channels_cell input.'},
            {'id': 'MEAN_MOM_cc', 'source': 'Monday candidate panel; mean within groups, MoM between groups, channels_cell input.'},
            {'id': 'MEAN_MOM_ga', 'source': 'Monday candidate panel; mean within groups, MoM between groups, groups_answer input.'},
            {'id': 'TCI_ONLY_ga', 'source': 'Prop 2.4 only plus MoM; groups_answer input, saved score arrays from 2026-10-05 TCI run.'},
        ],
        'population': 'All 8 banks (B13/B16, B20/B23, B32/B35, B51/B54), PRMBench and all 8 ProcessBench cells; source-question groups retained for paired bootstrap.',
        'decision_rules': RULES,
        'top_k': {'values': [1, 2, 3], 'status': 'Fixed diagnostic rows; top-2 is post hoc and will be labelled.'},
        'f11_prm': {
            'categories': 'Use the frozen PRM metadata error category field; retain the broader classification field as an additional label.',
            'candidate_metrics': 'Per category and bank, report within-answer AUC and PRMScore for each decision rule and top-1/2/3.',
            'supervised_prm': 'Use frozen labels==0 as PRM flags and negative reward as its continuous error score; report native and top-1/2/3.',
            'overlap': 'On gold error steps, report both, only candidate, only supervised PRM, neither; candidate rules and top-k are separate rows.',
            'error_distribution': 'Report answers by count of full PRMBench annotated errors; no analogous full-error count for ProcessBench.'},
        'f12_processbench': {
            'mind_the_gap': 'Frozen teacher-forced Mind the Gap reproduction, Shannon / paper EMA span5 / later-token attribution / top5 drop mean / native argmax; not paper headline scores.',
            'metrics': 'By first-error-position quartile and answer-length bin, report candidate argmax hit, first flagged step hit, and MTG native argmax hit; report early/late/no-flag misses and clean-answer flags.',
            'mtg_clean': 'MTG argmax always selects a step; it has no abstention/clean-answer rule.'},
        'primary_comparisons': 'All six paired candidate-vs-candidate contrasts under the same decision rule, plus candidate argmax vs MTG native argmax. Top-1/2/3 rows are descriptive diagnostics without separate inferential selection.',
        'statistics': {'bootstrap_draws': 5000, 'seed': 20261006, 'unit': 'source question',
                       'pairing': 'Shared source-group resampling across banks, candidates, decision rules, and comparators within benchmark.',
                       'intervals': 'Two-sided percentile 95% intervals; descriptive for this competitor breakdown, no winner selection or multiplicity-based claim.'},
        'validation': ['Population IDs/offsets equal for saved scores and frozen population.',
                       'Recompute HEM_MOM_cc, MEAN_MOM_cc, MEAN_MOM_ga metrics and require parity with candidate panel <=1e-12.',
                       'Recompute TCI_ONLY_ga metrics and require parity with saved Stage 2 metrics <=1e-12.',
                       'Mind the Gap hash, shape, offsets and PRM metadata label alignment must pass.',
                       'Any incomplete cell, hash mismatch or metric mismatch stops the run; no fallback or partial winner.'],
        'inputs': 'INPUT_MANIFEST.json pins all score arrays, frozen population, PRM metadata, source manifests, answer features, and Mind the Gap comparator.',
        'outputs': ['METRICS.csv', 'F11_PRM_CATEGORY_METRICS.csv', 'F11_PRM_OVERLAP.csv.gz',
                    'F11_PRM_OVERLAP_SUMMARY.csv', 'F11_ERROR_COUNTS.csv', 'F12_MTG_ANSWER_FORENSICS.csv.gz',
                    'F12_MTG_BREAKDOWN.csv', 'CONTRASTS.csv', 'REPORT.md', 'figures/F11_*', 'figures/F12_*'],
        'storage': 'Use existing scores. Expected outputs below 200 MB; stop if measured projection exceeds 200 MB.'}
    dump(protocol_path, protocol)
    input_manifest = {key: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
                      for key, path in sorted(required.items())}
    dump(OUT / 'INPUT_MANIFEST.json', {'schema': 'monday-fusion-competitor-inputs-v1', 'files': input_manifest})
    source_files = [Path(__file__), ROOT / 'spectral_utils/monday_candidate_panel.py',
                    CODE / 'spectral_utils/monday_fusion_decision.py', CODE / 'spectral_utils/monday_fusion_metrics.py',
                    CODE / 'spectral_utils/monday_fusion_protocol.py',
                    CODE / 'spectral_utils/lsml_gate_locator_research.py', CODE / 'spectral_utils/prmbench.py',
                    CODE / 'scripts/experiments/calfix_common.py',
                    TCI_ROOT / 'tci_wide_stage1.py', TCI_ROOT / 'tci_wide_stage2.py',
                    TCI_ROOT / 'tci_triplet_selection.py']
    dump(OUT / 'CODE_MANIFEST.json', {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in source_files})
    print(f'Frozen protocol and input/code manifests; {len(input_manifest)} input files; no scores computed.')


def load_runtime():
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(CODE / 'scripts/experiments'))
    import spectral_utils
    spectral_utils.__path__.append(str(CODE / 'spectral_utils'))
    from calfix_common import Population
    spec = importlib.util.spec_from_file_location('monday_candidate_panel_inputs', ROOT / 'spectral_utils/monday_candidate_panel.py')
    cp = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(cp)
    from spectral_utils.monday_fusion_decision import rules
    from spectral_utils.monday_fusion_metrics import evaluate, top_flags, prmscore_counts
    pop = Population()
    evidence = cp.Evidence()
    data = cp.Inputs(pop, evidence)
    return pop, data, evidence, rules, evaluate, top_flags, prmscore_counts


def atomic_csv(path: Path, rows, columns=None):
    import pandas as pd
    tmp = path.with_suffix(path.suffix + '.tmp')
    pd.DataFrame(rows, columns=columns).to_csv(tmp, index=False, encoding='utf-8')
    tmp.replace(path)


def record_to_miss(first: int, target: int, erroneous: bool):
    if not erroneous:
        return None
    if first < 0:
        return 'no_flag'
    if first < target:
        return 'early'
    if first > target:
        return 'late'
    return 'hit'


def run():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import Counter
    from datetime import datetime, timezone
    from scipy.stats import rankdata

    prot = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))
    if prot.get('status') != 'FROZEN_BEFORE_EVALUATION':
        raise RuntimeError('Protocol is not frozen')
    for item in json.loads((OUT / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))['files'].values():
        p = Path(item['path'])
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise RuntimeError('Frozen input hash/size mismatch: ' + str(p))
    code_manifest = json.loads((OUT / 'CODE_MANIFEST.json').read_text(encoding='utf8'))
    for p, item in code_manifest.items():
        if sha(Path(p)) != item['sha256']:
            raise RuntimeError('Frozen code hash mismatch: ' + p)
    frozen_inputs = json.loads((OUT / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))['files']
    # All output-producing steps target this fresh run directory only.
    generated = ['METRICS.csv', 'F11_PRM_CATEGORY_METRICS.csv', 'F11_PRM_OVERLAP.csv.gz',
                 'F11_PRM_OVERLAP_SUMMARY.csv', 'F11_ERROR_COUNTS.csv',
                 'F12_MTG_ANSWER_FORENSICS.csv.gz', 'F12_MTG_BREAKDOWN.csv', 'CONTRASTS.csv',
                 'REPORT.md', 'RUN_STATUS.json']
    if any((OUT / name).exists() for name in generated) or (OUT / 'figures').exists():
        raise RuntimeError('Output exists; refusing to overwrite')
    pop, data, evidence, decision_rules, evaluate, top_flags, prmscore_counts = load_runtime()
    prm_meta = pop.meta
    mtg = np.asarray(data.mtg, float)
    if mtg.shape != (pop.total,) or not np.isfinite(mtg).all():
        raise RuntimeError('Mind the Gap score shape/non-finite check failed')
    expected_answers = np.load(STEP_EVIDENCE / 'OOF_STEP_SCORES.npz', mmap_mode='r')['offsets']
    if not np.array_equal(expected_answers, pop.off):
        raise RuntimeError('Frozen population offsets disagree')

    metric_rows, cat_rows, error_rows = [], [], []
    overlap_counts, pb_counts = {}, {}
    overlap_row_count = 0; pb_row_count = 0
    overlap_tmp = OUT / 'F11_PRM_OVERLAP.csv.gz.tmp'
    pb_tmp = OUT / 'F12_MTG_ANSWER_FORENSICS.csv.gz.tmp'
    if overlap_tmp.exists() or pb_tmp.exists():
        raise RuntimeError('Partial stream outputs exist; inspect/remove them before restarting')
    overlap_f = gzip.open(overlap_tmp, 'wt', encoding='utf8', newline='', compresslevel=6)
    overlap_writer = csv.DictWriter(overlap_f, fieldnames=['bank','answer_id','source_group','category','classification',
        'method','rule','n_errors','n_steps','both','only_ours','only_PRM','neither','ours_clean_flag','PRM_clean_flag'])
    overlap_writer.writeheader()
    pb_f = gzip.open(pb_tmp, 'wt', encoding='utf8', newline='', compresslevel=6)
    pb_writer = csv.DictWriter(pb_f, fieldnames=['bank','cell','answer_id','source_group','method','rule','target',
        'n_steps','length_bin','error_position_quartile','argmax_hit','decision_hit','MTG_native_argmax_hit',
        'ours_miss','MTG_miss','ours_clean_flag','MTG_clean_flag','MTG_has_abstention'])
    pb_writer.writeheader()
    # Sufficient arrays retained for the registered paired source-question bootstrap.
    prm_perf = {}  # (bank, method, rule) -> auc/confusion by answer
    pb_perf = {}    # (bank, method, rule) -> hit/clean/argmax arrays by answer
    prm_native = {}
    mtg_perf = {}
    cells = sorted(set(pop.cells))
    panel_df = pd.read_csv(PANEL / 'CONTENT_METRICS.csv', low_memory=False)
    stage2_df = pd.read_csv(TCI / 'STAGE2_CELL_METRICS.csv', low_memory=False)

    for bk in BANKS:
        for cell in cells:
            answers = np.flatnonzero(pop.cells == cell)
            rows = pop.rows(pop.cells == cell)
            off = np.r_[0, np.cumsum(pop.ns[answers])]
            bank = {'answers': answers, 'rows': rows, 'off': off}
            tci_path = TCI / 'scores' / f'{bk}__{cell}.npz'
            with np.load(tci_path) as z:
                if not np.array_equal(z['answers'], answers) or not np.array_equal(z['offsets'], off) or not np.array_equal(z['rows'], rows):
                    raise RuntimeError(f'TCI score population mismatch {bk}/{cell}')
                surv = z['surv'].astype(int)
            stem = f'{cell}__{bk}__groups_answer'
            dpath = INPUT_RUN / 'chunks' / f'{stem}.diagnostics.json'
            if not dpath.is_file():
                raise FileNotFoundError(dpath)
            record = json.loads(dpath.read_text(encoding='utf8'))
            feature_bank = data.bank(bk, cell, record)
            if not np.array_equal(feature_bank['answers'], answers) or not np.array_equal(feature_bank['rows'], rows):
                raise RuntimeError(f'Feature bank alignment mismatch {bk}/{cell}')
            if int(surv.max()) >= feature_bank['V'].shape[1]:
                raise RuntimeError(f'TCI survivor index out of bank range {bk}/{cell}')
            raw_surv = feature_bank['raw'][:, surv]
            scores, flags_by_method = {}, {}
            detectors_by_method = {}
            if cell != 'prmbench_qwen3_8b':
                mtg_hit = np.zeros(len(answers), dtype=bool)
                mtg_err = pop.target[answers] >= 0
                for j, i in enumerate(answers):
                    target_i = int(pop.target[i])
                    mtg_peak_i = int(np.argmax(mtg[pop.off[i]:pop.off[i+1]]))
                    mtg_hit[j] = mtg_peak_i == target_i
                mtg_perf[cell] = {'answers': answers.copy(), 'hit': mtg_hit, 'err': mtg_err}
            for method, (variant, recipe) in METHODS.items():
                if variant == 'tci':
                    with np.load(tci_path) as z:
                        score = z[recipe].astype(float)
                    flag_map, detectors, _ = decision_rules(score, feature_bank, raw_surv,
                        data.entropy[answers], pop.uids[answers])
                    scores[method] = score
                    flags_by_method[method] = flag_map
                    detectors_by_method[method] = detectors
                else:
                    path = PANEL / 'candidate_scores' / f'{cell}__{bk}__{variant}.npz'
                    with np.load(path) as z:
                        if not np.array_equal(z['answer_indices'], answers) or not np.array_equal(z['offsets'], off):
                            raise RuntimeError(f'Candidate panel population mismatch {bk}/{cell}/{method}')
                        scores[method] = z[f'{recipe}__score'].astype(float)
                        flags_by_method[method] = {rule: z[f'{recipe}__{rule}__flags'].astype(bool) for rule in RULES}
                    rebuilt, detectors, _ = decision_rules(scores[method], feature_bank, raw_surv,
                        data.entropy[answers], pop.uids[answers])
                    for rule in RULES:
                        if not np.array_equal(rebuilt[rule], flags_by_method[method][rule]):
                            raise RuntimeError(f'Frozen decision replay mismatch {bk}/{cell}/{method}/{rule}')
                    detectors_by_method[method] = detectors
                if scores[method].shape != (int(off[-1]),) or not np.isfinite(scores[method]).all():
                    raise RuntimeError(f'Incomplete/non-finite scores: {bk}/{cell}/{method}')
                for action in RULES + TOPS:
                    flags = flags_by_method[method][action] if action in RULES else top_flags(scores[method], off, np.full(len(answers), int(action[-1])))
                    detector = detectors_by_method[method].get(action) if action in RULES else None
                    met, per = evaluate(pop, bank, scores[method], flags, detector)
                    metric_rows.append({'bank': bk, 'cell': cell, 'method': method, 'decision_rule': action,
                                        'scope': 'development_only', **met})
                    # Candidate panel and TCI Stage 2 are hard replay gates on every cell.
                    if action in RULES and method != 'TCI_ONLY_ga':
                        variant, recipe = METHODS[method]
                        q = panel_df[(panel_df.bank == bk) & (panel_df.cell == cell) &
                                         (panel_df.variant == variant) & (panel_df.recipe == recipe) &
                                         (panel_df.decision_rule == action)]
                        if len(q) != 1:
                            raise RuntimeError(f'Candidate-panel row missing {bk}/{cell}/{method}/{action}')
                        for field in METRIC_FIELDS:
                            ref = q.iloc[0][field]
                            got = met[field]
                            if pd.isna(ref) and got is None:
                                continue
                            if got is None or abs(float(ref) - float(got)) > 1e-12:
                                raise RuntimeError(f'Candidate panel replay failed {bk}/{cell}/{method}/{action}/{field}: {got}/{ref}')
                    if action in RULES and method == 'TCI_ONLY_ga':
                        q = stage2_df[(stage2_df.bank == bk) & (stage2_df.cell == cell) &
                                          (stage2_df.arm == method) & (stage2_df.rule == action)]
                        if len(q) != 1:
                            raise RuntimeError(f'TCI Stage 2 row missing {bk}/{cell}/{method}/{action}')
                        for field in METRIC_FIELDS:
                            ref = q.iloc[0][field]
                            got = met[field]
                            if pd.isna(ref) and got is None:
                                continue
                            if got is None or abs(float(ref) - float(got)) > 1e-12:
                                raise RuntimeError(f'TCI Stage 2 replay failed {bk}/{cell}/{action}/{field}: {got}/{ref}')
                    if cell == 'prmbench_qwen3_8b':
                        prm_perf[(bk, method, action)] = {'auc': per['auc'].copy(), 'conf': per['confusion'].copy()}
                    else:
                        target = pop.target[answers]
                        hit = (per['first'] == target)
                        argmax_hit = (per['argmax'] == target)
                        pb_perf[(bk, method, action, cell)] = {'answers': answers.copy(), 'hit': hit,
                            'flag': per['flag_count'] > 0, 'argmax_hit': argmax_hit, 'err': target >= 0}
                        for j, i in enumerate(answers):
                            ns = int(pop.ns[i]); t = int(target[j]); first = int(per['first'][j])
                            length_bin = '1-5' if ns <= 5 else '6-10' if ns <= 10 else '11-20' if ns <= 20 else '21+'
                            quartile = 'clean' if t < 0 else str(min(3, int(4 * t / max(ns - 1, 1))))
                            mtg_peak = int(np.argmax(mtg[pop.off[i]:pop.off[i+1]]))
                            pb_row = {'bank': bk, 'cell': cell, 'answer_id': pop.ids[i],
                                'source_group': pop.groups[i], 'method': method, 'rule': action,
                                'target': t, 'n_steps': ns, 'length_bin': length_bin,
                                'error_position_quartile': quartile, 'argmax_hit': bool(argmax_hit[j]) if t >= 0 else None,
                                'decision_hit': bool(hit[j]), 'MTG_native_argmax_hit': mtg_peak == t if t >= 0 else None,
                                'ours_miss': record_to_miss(first, t, t >= 0),
                                'MTG_miss': ('early' if mtg_peak < t else 'late' if mtg_peak > t else 'hit') if t >= 0 else None,
                                'ours_clean_flag': bool(per['flag_count'][j] > 0) if t < 0 else None,
                                'MTG_clean_flag': None, 'MTG_has_abstention': False}
                            pb_writer.writerow(pb_row)
                            pb_row_count += 1
                            strata = [('answer_length', length_bin), ('error_position_quartile', quartile)] if t >= 0 else [('clean_answers', cell)]
                            for stratum, value in strata:
                                key_pb = (bk, method, action, stratum, value)
                                acc = pb_counts.setdefault(key_pb, [0, 0, 0, 0, 0, 0, 0, 0, 0])
                                acc[0] += 1
                                if t >= 0:
                                    acc[1] += int(bool(argmax_hit[j])); acc[2] += int(bool(hit[j]))
                                    acc[3] += int(mtg_peak == t)
                                    acc[4] += int(pb_row['ours_miss'] == 'early'); acc[5] += int(pb_row['ours_miss'] == 'late')
                                    acc[6] += int(pb_row['ours_miss'] == 'no_flag'); acc[7] += int(pb_row['ours_miss'] == 'hit')
                                else:
                                    acc[8] += int(bool(per['flag_count'][j] > 0))
                if cell == 'prmbench_qwen3_8b':
                    categories = {}
                    for i in answers:
                        meta = prm_meta[pop.ids[i]]
                        categories.setdefault(str(meta.get('category', 'unknown')), []).append(int(i))
                    for category, members in sorted(categories.items()):
                        loc = np.isin(answers, members)
                        conf = per['confusion'][loc].sum(axis=0)
                        auc = per['auc'][loc]
                        cat_rows.append({'bank': bk, 'category': category, 'method': method,
                            'rule': action, 'n_answers': int(loc.sum()), 'n_auc_answers': int(np.isfinite(auc).sum()),
                            'within_answer_auc': float(np.nanmean(auc)) if np.isfinite(auc).any() else None,
                            'prmscore': float(prmscore_counts(conf)) if conf.sum() > 0 else None,
                            'top2_post_hoc': action == 'TOP_2', 'scope': 'development_only'})
            if cell == 'prmbench_qwen3_8b':
                # Supervised PRM's own frozen labels and rewards, plus per-answer error-count distribution.
                labels = np.concatenate([np.asarray(prm_meta[pop.ids[i]]['labels']) for i in answers])
                reward = np.concatenate([np.asarray(prm_meta[pop.ids[i]]['rewards'], float) for i in answers])
                critic_score = -reward
                critic_flags = labels == 0
                for action in ['PRM_NATIVE'] + TOPS:
                    f = critic_flags if action == 'PRM_NATIVE' else top_flags(critic_score, off, np.full(len(answers), int(action[-1])))
                    met, per = evaluate(pop, bank, critic_score, f)
                    cat_map = {}
                    for i in answers:
                        cat_map.setdefault(str(prm_meta[pop.ids[i]].get('category', 'unknown')), []).append(int(i))
                    for category, members in sorted(cat_map.items()):
                        loc = np.isin(answers, members); conf = per['confusion'][loc].sum(axis=0); auc = per['auc'][loc]
                        cat_rows.append({'bank': bk, 'category': category, 'method': 'SUPERVISED_PRM',
                            'rule': action, 'n_answers': int(loc.sum()), 'n_auc_answers': int(np.isfinite(auc).sum()),
                            'within_answer_auc': float(np.nanmean(auc)) if np.isfinite(auc).any() else None,
                            'prmscore': float(prmscore_counts(conf)) if conf.sum() > 0 else None,
                            'top2_post_hoc': action == 'TOP_2', 'scope': 'development_only'})
                    if action == 'PRM_NATIVE':
                        prm_native[bk] = {'auc': per['auc'].copy(), 'conf': per['confusion'].copy()}
                for i in answers:
                    meta = prm_meta[pop.ids[i]]; y = pop.labels[pop.off[i]:pop.off[i+1]]
                    error_rows.append({'bank': bk, 'answer_id': pop.ids[i], 'source_group': pop.groups[i],
                        'category': meta.get('category'), 'classification': meta.get('classification'),
                        'n_errors': int(y.sum()), 'n_steps': int(len(y))})
                # Overlap of actual error-step detections, candidate flags vs supervised PRM flags.
                critic = critic_flags
                for method in METHODS:
                    for action in RULES + TOPS:
                        if action in RULES:
                            ff = flags_by_method[method][action]
                        else:
                            ff = top_flags(scores[method], off, np.full(len(answers), int(action[-1])))
                        start = 0
                        for j, i in enumerate(answers):
                            a, b = off[j:j+2]; gold = pop.labels[pop.off[i]:pop.off[i+1]]; ours = ff[a:b]; prm = critic[a:b]
                            ov_row = {'bank': bk, 'answer_id': pop.ids[i], 'source_group': pop.groups[i],
                                'category': prm_meta[pop.ids[i]].get('category'),
                                'classification': prm_meta[pop.ids[i]].get('classification'), 'method': method,
                                'rule': action, 'n_errors': int(gold.sum()), 'n_steps': int(b-a),
                                'both': int((gold & ours & prm).sum()), 'only_ours': int((gold & ours & ~prm).sum()),
                                'only_PRM': int((gold & ~ours & prm).sum()), 'neither': int((gold & ~ours & ~prm).sum()),
                                'ours_clean_flag': bool(ours.any()) if not gold.any() else None,
                                'PRM_clean_flag': bool(prm.any()) if not gold.any() else None}
                            overlap_writer.writerow(ov_row)
                            overlap_row_count += 1
                            key_ov = (bk, ov_row['category'], ov_row['classification'], method, action)
                            acc_ov = overlap_counts.setdefault(key_ov, [0, 0, 0, 0, 0, 0, 0])
                            for ix, name in enumerate(['n_errors','n_steps','both','only_ours','only_PRM','neither']):
                                acc_ov[ix] += int(ov_row[name])
                            acc_ov[6] += 1

    overlap_f.close(); pb_f.close()
    (OUT / 'F11_PRM_OVERLAP.csv.gz.tmp').replace(OUT / 'F11_PRM_OVERLAP.csv.gz')
    (OUT / 'F12_MTG_ANSWER_FORENSICS.csv.gz.tmp').replace(OUT / 'F12_MTG_ANSWER_FORENSICS.csv.gz')
    # Verify the immutable comparator source after all use and population mapping.
    evidence.verify()
    for key, item in frozen_inputs.items():
        p = Path(item['path'])
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise RuntimeError('Input changed during run: ' + key)
    if len(metric_rows) != len(BANKS) * len(cells) * len(METHODS) * (len(RULES) + len(TOPS)):
        raise RuntimeError(f'Metric coverage failure: {len(metric_rows)}')
    if len(prm_perf) != len(BANKS) * len(METHODS) * (len(RULES) + len(TOPS)) or len(pb_perf) != len(BANKS) * len(METHODS) * (len(RULES) + len(TOPS)) * (len(cells)-1):
        raise RuntimeError('F11/F12 candidate coverage failure')

    # Compact, auditable outputs. Large per-answer tables are gzip-compressed.
    atomic_csv(OUT / 'METRICS.csv', metric_rows)
    atomic_csv(OUT / 'F11_PRM_CATEGORY_METRICS.csv', cat_rows)
    atomic_csv(OUT / 'F11_ERROR_COUNTS.csv', error_rows)
    breakdown = []
    for (b, m, r, stratum, value), acc in pb_counts.items():
        row = {'bank': b, 'method': m, 'rule': r, 'stratum': stratum, 'value': value, 'n_answers': int(acc[0])}
        if stratum == 'clean_answers':
            row.update(candidate_clean_flag_rate=acc[8]/acc[0], MTG_has_abstention=False)
        else:
            row.update(candidate_argmax_hit=acc[1]/acc[0], candidate_decision_hit=acc[2]/acc[0],
                       MTG_native_argmax_hit=acc[3]/acc[0], early=int(acc[4]), late=int(acc[5]),
                       no_flag=int(acc[6]), hit=int(acc[7]))
        breakdown.append(row)
    atomic_csv(OUT / 'F12_MTG_BREAKDOWN.csv', breakdown)

    # F11 overlap totals by candidate/rule/bank/category; one row per stratum.
    ovsum = []
    for (b, category, classification, method, rule), acc in overlap_counts.items():
        ovsum.append({'bank': b, 'category': category, 'classification': classification,
            'method': method, 'rule': rule, 'n_errors': acc[0], 'n_steps': acc[1],
            'both': acc[2], 'only_ours': acc[3], 'only_PRM': acc[4], 'neither': acc[5],
            'n_answers': acc[6], 'error_hit_union': acc[2]+acc[3]+acc[4]})
    atomic_csv(OUT / 'F11_PRM_OVERLAP_SUMMARY.csv', ovsum)

    # A paired source-group bootstrap for method contrasts on the registered headline metrics.
    # Resampling weights are shared across banks/methods/rules within each benchmark.
    rng = np.random.default_rng(20261006); draws = 5000
    prm_answers = np.flatnonzero(pop.prm); prm_groups, prm_inverse = np.unique(pop.groups[prm_answers], return_inverse=True)
    Wprm = rng.multinomial(len(prm_groups), np.full(len(prm_groups), 1/len(prm_groups)), size=draws).astype(np.uint8)
    pb_answers = np.flatnonzero(pop.pb); pb_groups, pb_inverse = np.unique(pop.groups[pb_answers], return_inverse=True)
    Wpb = rng.multinomial(len(pb_groups), np.full(len(pb_groups), 1/len(pb_groups)), size=draws).astype(np.uint8)

    def prm_boot(method, action):
        auc_s = np.zeros((len(prm_groups), len(BANKS))); auc_n = auc_s.copy()
        conf_s = np.zeros((len(prm_groups), len(BANKS), 4))
        for bi, bk in enumerate(BANKS):
            x = prm_perf[(bk, method, action)]
            np.add.at(auc_s[:, bi], prm_inverse, np.nan_to_num(x['auc']))
            np.add.at(auc_n[:, bi], prm_inverse, np.isfinite(x['auc']).astype(float))
            np.add.at(conf_s[:, bi, :], prm_inverse, x['conf'])
        auc_sum = Wprm @ auc_s; auc_count = Wprm @ auc_n
        auc_bank = np.divide(auc_sum, auc_count, out=np.full_like(auc_sum, np.nan), where=auc_count > 0)
        auc_value = np.nanmean(auc_bank, axis=1)
        c = np.einsum('dg,gbk->dbk', Wprm, conf_s)
        ps = np.full((draws, len(BANKS)), np.nan)
        for bi in range(len(BANKS)):
            ps[:, bi] = prmscore_counts(c[:, bi, :])
        return {'within_answer_auc': auc_value, 'prmscore': np.nanmean(ps, axis=1)}

    def pb_boot(method, action):
        sums = np.zeros((len(pb_groups), len(BANKS), 7))
        full = {}
        for field in ('hit', 'flag', 'argmax_hit', 'err'):
            full[field] = np.zeros((len(BANKS), len(pb_answers)), dtype=bool)
        for bi, bk in enumerate(BANKS):
            for cell in cells:
                if cell == 'prmbench_qwen3_8b':
                    continue
                x = pb_perf[(bk, method, action, cell)]
                dest = np.searchsorted(pb_answers, x['answers'])
                for field in full:
                    full[field][bi, dest] = x[field]
            x = {field: full[field][bi] for field in full}
            e = x['err']; h = x['hit'].astype(float); f = x['flag'].astype(float); ah = x['argmax_hit'].astype(float)
            np.add.at(sums[:, bi, 0], pb_inverse, h * e); np.add.at(sums[:, bi, 1], pb_inverse, e.astype(float))
            np.add.at(sums[:, bi, 2], pb_inverse, h * (~e)); np.add.at(sums[:, bi, 3], pb_inverse, (~e).astype(float))
            np.add.at(sums[:, bi, 4], pb_inverse, f * (~e)); np.add.at(sums[:, bi, 5], pb_inverse, (~e).astype(float))
            np.add.at(sums[:, bi, 6], pb_inverse, ah * e)
        z = np.einsum('dg,gbk->dbk', Wpb, sums)
        ae = np.divide(z[:, :, 0], z[:, :, 1], out=np.full((draws, len(BANKS)), np.nan), where=z[:, :, 1] > 0)
        ac = np.divide(z[:, :, 2], z[:, :, 3], out=np.full((draws, len(BANKS)), np.nan), where=z[:, :, 3] > 0)
        f1 = np.divide(2*ae*ac, ae+ac, out=np.zeros_like(ae), where=(ae+ac)>0)
        cfr = np.divide(z[:, :, 4], z[:, :, 5], out=np.full_like(ae, np.nan), where=z[:, :, 5] > 0)
        return {'pb_official_f1': np.nanmean(f1, axis=1), 'pb_decision_error_accuracy': np.nanmean(ae, axis=1),
                'clean_flag_rate': np.nanmean(cfr, axis=1),
                'pb_argmax_first_error_accuracy': np.nanmean(np.divide(z[:, :, 6], z[:, :, 1],
                    out=np.full_like(ae, np.nan), where=z[:, :, 1] > 0), axis=1)}

    def mtg_boot():
        sums = np.zeros((len(pb_groups), len(BANKS), 2))
        for bi, bk in enumerate(BANKS):
            xhit = np.zeros(len(pb_answers), dtype=bool); xerr = pop.target[pb_answers] >= 0
            for cell in cells:
                if cell == 'prmbench_qwen3_8b':
                    continue
                x = mtg_perf[cell]; dest = np.searchsorted(pb_answers, x['answers']); xhit[dest] = x['hit']
            np.add.at(sums[:, bi, 0], pb_inverse, xhit.astype(float) * xerr)
            np.add.at(sums[:, bi, 1], pb_inverse, xerr.astype(float))
        z = np.einsum('dg,gbk->dbk', Wpb, sums)
        return np.nanmean(np.divide(z[:, :, 0], z[:, :, 1], out=np.full((draws, len(BANKS)), np.nan),
                                    where=z[:, :, 1] > 0), axis=1)

    def prm_point(method, action):
        auc_values, ps_values = [], []
        for bk in BANKS:
            x = prm_perf[(bk, method, action)]
            if np.isfinite(x['auc']).any():
                auc_values.append(float(np.nanmean(x['auc'])))
            if x['conf'].sum() > 0:
                ps_values.append(float(prmscore_counts(x['conf'].sum(0))))
        return {'within_answer_auc': float(np.mean(auc_values)), 'prmscore': float(np.mean(ps_values))}

    def pb_point(method, action):
        values = {k: [] for k in ('pb_official_f1', 'pb_decision_error_accuracy',
                                  'clean_flag_rate', 'pb_argmax_first_error_accuracy')}
        for bk in BANKS:
            fields = {k: [] for k in ('hit', 'flag', 'argmax_hit', 'err')}
            for cell in cells:
                if cell == 'prmbench_qwen3_8b':
                    continue
                x = pb_perf[(bk, method, action, cell)]
                for k in fields:
                    fields[k].append(x[k])
            hit = np.concatenate(fields['hit']); flag = np.concatenate(fields['flag'])
            argmax_hit = np.concatenate(fields['argmax_hit']); err = np.concatenate(fields['err'])
            ae = float(hit[err].mean()); ac = float(hit[~err].mean())
            values['pb_official_f1'].append(2*ae*ac/(ae+ac) if ae+ac else 0.)
            values['pb_decision_error_accuracy'].append(ae)
            values['clean_flag_rate'].append(float(flag[~err].mean()))
            values['pb_argmax_first_error_accuracy'].append(float(argmax_hit[err].mean()))
        return {k: float(np.mean(v)) for k, v in values.items()}

    def mtg_point():
        hits, errs = [], []
        for cell in cells:
            if cell == 'prmbench_qwen3_8b':
                continue
            hits.append(mtg_perf[cell]['hit']); errs.append(mtg_perf[cell]['err'])
        hit = np.concatenate(hits); err = np.concatenate(errs)
        return float(hit[err].mean())

    contrasts = []
    names = list(METHODS)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]
    for action in RULES:
        boot = {m: {**prm_boot(m, action), **pb_boot(m, action)} for m in names}
        point = {m: {**prm_point(m, action), **pb_point(m, action)} for m in names}
        for a, b in pairs:
            for metric in ['within_answer_auc', 'prmscore', 'pb_official_f1', 'pb_decision_error_accuracy',
                           'clean_flag_rate', 'pb_argmax_first_error_accuracy']:
                diff = boot[a][metric] - boot[b][metric]
                # Point estimates are recomputed from the same paired empirical populations.
                lo, hi = np.nanquantile(diff, [.025, .975])
                contrasts.append({'benchmark': 'PRMBench' if metric in ('within_answer_auc','prmscore') else 'ProcessBench',
                    'method_a': a, 'method_b': b, 'rule': action, 'metric': metric,
                    'delta_a_minus_b': float(point[a][metric] - point[b][metric]), 'lo95': float(lo), 'hi95': float(hi),
                    'draws': draws, 'unit': 'source_group', 'interval_scope': 'descriptive'})
        mtg_ref = mtg_boot()
        mtg_point_value = mtg_point()
        for method in names:
            compare = boot[method]['pb_argmax_first_error_accuracy'] - mtg_ref
            contrasts.append({'benchmark': 'ProcessBench', 'method_a': method, 'method_b': 'MTG_NATIVE_ARGMAX',
                'rule': action, 'metric': 'argmax_first_error_accuracy',
                'delta_a_minus_b': float(point[method]['pb_argmax_first_error_accuracy'] - mtg_point_value),
                'lo95': float(np.nanquantile(compare, .025)), 'hi95': float(np.nanquantile(compare, .975)),
                'draws': draws, 'unit': 'source_group', 'interval_scope': 'descriptive'})
    atomic_csv(OUT / 'CONTRASTS.csv', contrasts)

    # Figures are written in both light and dark themes; each has PNG and SVG.
    figdir = OUT / 'figures'; figdir.mkdir()
    catdf = pd.DataFrame(cat_rows)
    pbagg = pd.DataFrame(breakdown)
    def save_pair(fig, stem, dark=False):
        suffix = '_dark' if dark else '_light'
        bg = '#17191d' if dark else '#ffffff'; fg = '#f4f4f4' if dark else '#222222'
        fig.patch.set_facecolor(bg)
        for ax in fig.axes:
            ax.set_facecolor(bg); ax.tick_params(colors=fg); ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg); ax.title.set_color(fg)
            for sp in ax.spines.values(): sp.set_color(fg)
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(bg)
                for t in leg.get_texts(): t.set_color(fg)
        fig.savefig(figdir / f'{stem}{suffix}.png', dpi=180, facecolor=bg, bbox_inches='tight')
        fig.savefig(figdir / f'{stem}{suffix}.svg', facecolor=bg, bbox_inches='tight')
        plt.close(fig)
    for dark in (False, True):
        fg = '#f4f4f4' if dark else '#222222'
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sub = catdf[(catdf.method.isin(METHODS)) & (catdf.rule == 'ANSWER_Z_Q80')]
        for ax, metric, title in zip(axes, ['within_answer_auc', 'prmscore'], ['Within-answer AUC', 'PRMScore']):
            plot = sub.groupby(['category', 'method'])[metric].mean().unstack('method')
            plot.plot(kind='bar', ax=ax); ax.set_title(title, color=fg); ax.set_ylabel(metric); ax.tick_params(axis='x', rotation=30)
        fig.suptitle('F11 · PRMBench by error category · answer-z q80', color=fg)
        fig.tight_layout(); save_pair(fig, 'F11_PRM_categories', dark)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sub = pbagg[(pbagg.method.isin(METHODS)) & (pbagg.rule == 'R2_COUNT') & (pbagg.stratum.isin(['error_position_quartile','answer_length']))]
        for ax, stratum, title in zip(axes, ['error_position_quartile','answer_length'], ['Error position quartile','Answer length']):
            q = sub[sub.stratum == stratum].groupby(['value','method'])[['candidate_argmax_hit','candidate_decision_hit','MTG_native_argmax_hit']].mean()
            q.unstack('method').plot(kind='bar', ax=ax); ax.set_title(title, color=fg); ax.set_ylabel('First-error hit rate'); ax.tick_params(axis='x', rotation=25)
        fig.suptitle('F12 · ProcessBench first-error localization · R2_COUNT', color=fg)
        fig.tight_layout(); save_pair(fig, 'F12_MTG_position_length', dark)
    
    # Explicit run manifest and concise human-readable report.
    for name in generated:
        if not (OUT / name).is_file():
            raise RuntimeError('Missing required output: ' + name)
    status = {'status': 'COMPUTED_PENDING_INDEPENDENT_REVIEW', 'development_only': True,
              'n_banks': len(BANKS), 'n_cells': len(cells), 'n_methods': len(METHODS),
              'metric_rows': len(metric_rows), 'category_rows': len(cat_rows),
              'overlap_rows': overlap_row_count, 'pb_forensic_rows': pb_row_count,
              'candidate_panel_parity_rows': len(BANKS)*len(cells)*len(RULES)*3,
              'tci_stage2_parity_rows': len(BANKS)*len(cells)*len(RULES),
              'source_group_bootstrap_draws': draws,
              'mindgap': 'Frozen teacher-forced MTG reproduction; native argmax, no abstention.',
              'review': 'CLAUDE_PENDING; do not treat as final claims.'}
    dump(OUT / 'RUN_STATUS.json', status)
    (OUT / 'REPORT.md').write_text(
        '# F11–F12: ארבע גרסאות מול PRM ו־Mind the Gap\n\n'
        'כל התוצאות הן ניתוח פיתוח על האוכלוסייה שכבר נחשפה. אין כאן הערכה חיצונית לא־חשופה. '
        'Mind the Gap הוא השחזור המקומי הקפוא (teacher-forced, top-5 drop mean, native argmax), לא תוצאת המאמר; '
        'ל־MTG אין כלל abstention לתשובות נקיות.\n\n'
        f'- ארבע שיטות, {len(BANKS)} בנקים, {len(cells)} תאים.\n'
        f'- {len(metric_rows)} שורות מדדים, {overlap_row_count} רשומות overlap, {pb_row_count} רשומות ProcessBench.\n'
        '- שלושת כללי ההחלטה מדווחים בנפרד; top-1/2/3 הם בקרות דירוג, ו־top-2 מסומן כבחירה בדיעבד.\n'
        '- הניגודים המזווגים משתמשים ב־5,000 דגימות bootstrap לפי קבוצת שאלת מקור.\n'
        '- מצב ביקורת עצמאי: ממתין ל־Claude; עד אז התוצאות אינן מסקנות למצגת.\n\n'
        'קבצים: `METRICS.csv`, `F11_PRM_CATEGORY_METRICS.csv`, `F11_PRM_OVERLAP.csv.gz`, '
        '`F11_PRM_OVERLAP_SUMMARY.csv`, `F11_ERROR_COUNTS.csv`, `F12_MTG_ANSWER_FORENSICS.csv.gz`, '
        '`F12_MTG_BREAKDOWN.csv`, `CONTRASTS.csv`, וגרפי PNG/SVG בהירים וכהים.\n', encoding='utf8')
    status['completed_utc'] = datetime.now(timezone.utc).isoformat()
    dump(OUT / 'RUN_STATUS.json', status)
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true', help='Write the pre-score protocol and manifests only.')
    group.add_argument('--run', action='store_true', help='Run the frozen F11/F12 analysis.')
    args = ap.parse_args()
    freeze() if args.freeze else run()
