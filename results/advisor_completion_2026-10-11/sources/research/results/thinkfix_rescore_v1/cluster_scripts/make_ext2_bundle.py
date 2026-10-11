"""Step 3 part 2 bundle: Monday panel methods on the think-fix external records (Hard2Verify, Socratic; Qwen3-8B).

Added on top of the chain-1 working copies ($SHARED/thinkfix_ext_<variant>/<mirror>), which already hold the
re-scored chain-1 shards, the inputs and the record links. Patched copies sit at their original paths (the
freezes hash scripts by path); originals are kept as *.frozen_original.py with *.THINKFIX_DIFF.txt.
Only changes: socratic_qwq32b removed from the cell lists (not re-collected; QwQ has no non-thinking mode),
and extract_external_banks_v4's tokenizer search also looks in the AIRCC HF cache. The family implementation
freeze is re-written (re-hashed) before scoring; labels are hashed there, not opened. The monday_nested_banks_v2
working directory receives the VARIANT's own development outputs (HIGHLIGHT, PANEL_A_ABSOLUTE, SOURCE_PARITY_GATE)
via --variant-dev so its external evaluation is compared with that variant's development panel.
    python make_ext2_bundle.py <out dir> <variant dev nested_banks dir>
"""
import difflib, hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

M = Path(r'C:\Users\omris\TAU\hallucination_detection')
SSL = '.worktrees/ssl-pseudolabel-residual-v1'
DST = Path('\\\\?\\' + str(Path(sys.argv[1]).resolve())); DEV = Path(sys.argv[2])  # extended-length path (Windows MAX_PATH)
NOTE = 'thinkfix_rescore_v1 step 3 part 2: socratic_qwq32b removed (not re-collected: QwQ template has no non-thinking mode)'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp(rel, src_root=M):
    d = DST / rel; d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src_root / rel, d)


def cptree(rel):
    shutil.copytree(M / rel, DST / rel, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=True)


def patch(rel, pairs):
    src = (M / rel).read_text(encoding='utf8'); out = src
    for old, new in pairs:
        if out.count(old) != 1: raise SystemExit(f'{rel}: anchor x{out.count(old)}: {old[:80]!r}')
        out = out.replace(old, new)
    d = DST / rel; d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(out.encode('utf8'))
    shutil.copy2(M / rel, d.with_name(d.stem + '.frozen_original.py'))
    diff = ''.join(difflib.unified_diff(src.splitlines(True), out.splitlines(True), fromfile=rel + ' (frozen)', tofile=rel + ' (thinkfix)'))
    d.with_name(d.stem + '.THINKFIX_DIFF.txt').write_text(f'# {NOTE}\n' + diff, encoding='utf8')


Q3 = "{'hard2verify_qwen3_8b': 'hard2verify', 'socratic_qwen3_8b': 'socratic', 'socratic_qwq32b': 'socratic'}"
Q2 = "{'hard2verify_qwen3_8b': 'hard2verify', 'socratic_qwen3_8b': 'socratic'}  # thinkfix: QwQ not re-collected"
# --- family features (chain C)
patch('scripts/run_family_external.py', [("CELLS=('hard2verify_qwen3_8b','socratic_qwen3_8b','socratic_qwq32b')",
                                          "CELLS=('hard2verify_qwen3_8b','socratic_qwen3_8b')  # thinkfix: QwQ not re-collected")])
fz = json.loads((M / 'results/family_tail_external_v1/IMPLEMENTATION_FREEZE.json').read_text())
for rel in fz['files']:
    if rel != 'scripts/run_family_external.py' and not rel.startswith('scratch/'):
        cp(rel)
cp('results/family_tail_external_v1/source_full_v1/GATE.json')
cp('results/family_tail_external_v1/IMPLEMENTATION_FREEZE.json')
(DST / 'results/family_tail_external_v1/IMPLEMENTATION_FREEZE.json').rename(DST / 'results/family_tail_external_v1/IMPLEMENTATION_FREEZE.frozen.json')
fz_new = dict(fz); fz_new['external_scoring_started'] = None
fz_new['files'] = {rel: None for rel in fz['files']}   # filled on the cluster, after linking scratch inputs (hash_family_freeze.py)
fz_new['thinkfix_note'] = ('Re-hashed for the think-fix re-scoring (2026-10-11): only scripts/run_family_external.py changed '
                           '(QwQ removed). Labels in scratch/... are hashed, never opened by this stage.')
(DST / 'results/family_tail_external_v1/IMPLEMENTATION_FREEZE.json').write_text(json.dumps(fz_new, indent=2) + '\n')
# --- v4 external banks (chain D)
patch(f'{SSL}/scripts/extract_external_banks_v4.py', [
    ("CELLS = {'hard2verify_qwen3_8b': ('hard2verify', 200), 'socratic_qwen3_8b': ('socratic', 2995),\n         'socratic_qwq32b': ('socratic', 2995)}",
     "CELLS = {'hard2verify_qwen3_8b': ('hard2verify', 200), 'socratic_qwen3_8b': ('socratic', 2995)}  # thinkfix: QwQ not re-collected"),
    ("    files += sorted(glob.glob(str(Path.home()/'.cache/huggingface/hub/models--Qwen--Q*/snapshots/*/tokenizer.json')))\n",
     "    files += sorted(glob.glob(str(Path.home()/'.cache/huggingface/hub/models--Qwen--Q*/snapshots/*/tokenizer.json')))\n"
     "    files += sorted(glob.glob('/shared/cycle2_tau_averbuch_prj/omrisegev1/hf_cache/hub/models--Qwen--Q*/snapshots/*/tokenizer.json'))  # thinkfix: AIRCC HF cache\n")])
cptree(f'{SSL}/spectral_utils')
for rel in (f'{SSL}/results/external_banks_v4/source_parity/GATE.json', f'{SSL}/results/external_banks_v4/source/FEATURES.npz',
            f'{SSL}/results/external_banks_v4/source/MANIFEST.json', f'{SSL}/results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv'):
    cp(rel)
for p in (M / 'results/automatic_group_free_phase_a6_s0a_v1/inputs').glob('qwen3-8b*/tokenizer.json'):
    cp(str(p.relative_to(M)))
# --- Monday panel (chain E/F), cont baselines (G), nested v2 (H)
patch('results/monday_external_panel_v1/score.py', [("CELLS = " + Q3, "CELLS = " + Q2)])
patch('results/monday_external_panel_v1/evaluate.py', [("CELLS = " + Q3, "CELLS = " + Q2)])
patch('results/monday_nested_banks_v2/evaluate_ext.py', [("CELLS = " + Q3, "CELLS = " + Q2)])
for rel in ('results/monday_external_panel_v1/PROTOCOL.json', 'results/monday_external_panel_v1/SOURCE_PARITY_GATE.json',
            'results/monday_cont_baselines_v1/run.py', 'results/monday_cont_baselines_v1/score_ext.py',
            'results/monday_cont_baselines_v1/evaluate_ext.py', 'results/monday_cont_baselines_v1/AMENDMENT_EXT.json',
            'results/monday_cont_baselines_v1/PROTOCOL.json',
            'results/monday_nested_banks_v2/dev_run.py', 'results/monday_nested_banks_v2/score_ext.py',
            'results/monday_nested_banks_v2/external_oriented.py', 'results/monday_nested_banks_v2/PROTOCOL.json',
            'results/monday_nested_banks_v2/AMENDMENT_A1.json', 'results/monday_nested_banks_v2/AMENDMENT_A2.json',
            'results/estimator_input_v1/run_20261004_resume/INPUT_MANIFEST.json',
            'results/monday_presentation_panels_v1/run_20261006/PANEL_A_ABSOLUTE.csv',
            '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_ANSWERS.csv'):
    cp(rel)
cptree('results/monday_fusion_analysis_v1/code')
cptree('scratch/external_generalization_private/sources')
for p in (M / 'results/monday_candidate_panel_v1/run_20261004/candidate_scores').glob('prmbench_qwen3_8b__B*__*.npz'):
    cp(str(p.relative_to(M)))
for p in (M / 'results/mom_filter_within_diagnostic_20261004/tci_wide_20261005/scores').glob('B*__prmbench_qwen3_8b.npz'):
    cp(str(p.relative_to(M)))
# variant's own development outputs for the nested external run
for name in ('HIGHLIGHT.json', 'PANEL_A_ABSOLUTE.csv', 'SOURCE_PARITY_GATE.json', 'BANKS.json'):
    d = DST / 'results/monday_nested_banks_v2' / name; shutil.copy2(DEV / name, d)
(DST / 'results/monday_nested_banks_v2/VARIANT_DEV_SOURCE.txt').write_text(
    f'HIGHLIGHT.json, PANEL_A_ABSOLUTE.csv, SOURCE_PARITY_GATE.json and BANKS.json here come from {DEV} '
    '(this variant\'s own thinkfix_rescore_v1 development run), not from the frozen monday_nested_banks_v2.\n')
print('bundle ready', sum(1 for _ in DST.rglob('*') if _.is_file()), 'files')
