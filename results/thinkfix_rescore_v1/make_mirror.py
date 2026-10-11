"""Assemble the downstream mirror bundle: files keep their path relative to the main checkout.
On the cluster the bundle is unpacked into a directory literally named
'C:\\Users\\omris\\TAU\\hallucination_detection' inside the job's working directory, so the
frozen scripts' hardcoded Windows MAIN constants resolve to it without edits. Full Windows paths
stored INSIDE JSON manifests are single file names on Linux; LINKS.json lists them so the
cluster side can create those names as symlinks to the mirrored files."""
import hashlib, json, shutil, sys
from pathlib import Path

M = Path(r'C:\Users\omris\TAU\hallucination_detection')
DST = Path(sys.argv[1])
FILES = [
    'results/monday_external_panel_v1/score.py',
    'results/monday_nested_banks_v2/dev_run.py', 'results/monday_nested_banks_v2/PROTOCOL.json',
    'results/monday_nested_banks_v2/PANEL_A_ABSOLUTE.csv', 'results/monday_nested_banks_v2/PER_ANSWER.npz',
    'results/monday_nested_banks_v2/BANKS.json',
    'results/monday_cont_baselines_v1/run.py', 'results/monday_cont_baselines_v1/PROTOCOL.json',
    'results/estimator_input_v1/run_20261004_resume/INPUT_MANIFEST.json',
    'results/monday_candidate_panel_v1/run_20261004/candidate_scores/prmbench_qwen3_8b__B13__groups_answer.npz',
    'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl',
    '.worktrees/ssl-pseudolabel-residual-v1/results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv',
    '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_ANSWERS.csv',
    '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_STEP_SCORES.npz',
    '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/INPUT_FREEZE.json',
    '.worktrees/decision-rule-v1/results/answer_gate_v1/ANSWER_FEATURES.npz',
    'results/thinkfix_rescore_v1/run_prmbench_rescore.py',
]
DIRS = ['spectral_utils', 'results/monday_fusion_analysis_v1/code',
        'dataset_cache/four_localization/pb_prm_qwen25math7b_full', 'dataset_cache/four_localization/pb_critic_qwen72b_full']
manifest = {}
for rel in FILES:
    src = M / rel; dst = DST / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    manifest[rel] = hashlib.sha256(src.read_bytes()).hexdigest()
for rel in DIRS:
    shutil.copytree(M / rel, DST / rel, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=False)
    for p in sorted((DST / rel).rglob('*')):
        if p.is_file():
            manifest[str(p.relative_to(DST)).replace('\\', '/')] = hashlib.sha256(p.read_bytes()).hexdigest()
# Windows paths stored inside JSON manifests that the scorers open directly
fr = json.loads((M / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
man = json.loads((M / 'results/estimator_input_v1/run_20261004_resume/INPUT_MANIFEST.json').read_text(encoding='utf8'))
links = {}
for p in [fr['prm_metadata']['path'], man['inputs']['answer_features']['path']]:
    rel = str(Path(p).relative_to(M)).replace('\\', '/')
    assert (DST / rel).exists(), rel
    links[p] = rel
(DST / 'LINKS.json').write_text(json.dumps(links, indent=1))
(DST / 'MIRROR_MANIFEST.json').write_text(json.dumps({'source_root': str(M), 'files_sha256': manifest}, indent=1))
print(len(manifest), 'files;', sum(f.stat().st_size for f in DST.rglob('*') if f.is_file()) / 1e6, 'MB')
