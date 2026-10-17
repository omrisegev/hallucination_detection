"""Assemble the external-chain mirror (step 3, chain 1: locked LSML external evaluation) for the think-fix records.

Unpacked on AIRCC into <WD>/'C:\\Users\\omris\\TAU\\hallucination_detection' so ROOT-relative paths resolve.
Patched copies (QwQ cell removed: QwQ's template has no non-thinking mode, so it was not re-collected) sit at
the ORIGINAL paths, because the freeze files hash scripts by path; each original is kept beside it as
<name>.frozen_original.py with a unified diff. A NEW analysis freeze (ANALYSIS_FREEZE.json) is written here,
before any label is opened, hashing the patched scripts; the method freeze (scoring code + BUNDLE) is unchanged.
Labels (inputs/evaluator_only) are opened only by evaluate_external_locked_scores.py after all cells are sealed.
"""
import difflib, hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

M = Path(r'C:\Users\omris\TAU\hallucination_detection')
DST = Path(sys.argv[1])
EV = 'results/lsml_external_generalization_v1/evaluation'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp(rel):
    d = DST / rel; d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(M / rel, d)


def patch(rel, pairs, note):
    src = (M / rel).read_text(encoding='utf8'); out = src
    for old, new in pairs:
        if out.count(old) != 1: raise SystemExit(f'{rel}: anchor x{out.count(old)}: {old[:70]!r}')
        out = out.replace(old, new)
    d = DST / rel; d.parent.mkdir(parents=True, exist_ok=True)
    d.write_bytes(out.encode('utf8'))
    orig = d.with_name(d.stem + '.frozen_original.py'); shutil.copy2(M / rel, orig)
    diff = ''.join(difflib.unified_diff(src.splitlines(True), out.splitlines(True), fromfile=rel + ' (frozen)', tofile=rel + ' (thinkfix)'))
    d.with_name(d.stem + '.THINKFIX_DIFF.txt').write_text(f'# {note}\n' + diff, encoding='utf8')


shutil.copytree(M / 'spectral_utils', DST / 'spectral_utils', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for rel in ['docs/experiments/LSML_EXTERNAL_EVALUATION_LOCK_20260924.md',
            'results/localization_source_group_audit_v1/QUESTION_METADATA.json',
            'results/localization_source_group_audit_v1/PB_QUESTION_METADATA.json',
            f'{EV}/METHOD_FREEZE.json', f'{EV}/ANALYSIS_FREEZE.json'] + \
           [f'{EV}/source/{n}' for n in ('BUNDLE.json', 'INPUTS.json', 'DEPLOYMENT_WEIGHTS.csv', 'STRICT_REPLAY.json',
                                          'VALIDATION_FITS.json', 'VALIDATION_METRICS.json')]:
    cp(rel)
(DST / EV / 'ANALYSIS_FREEZE.json').rename(DST / EV / 'ANALYSIS_FREEZE.frozen_20260924.json')
note = 'thinkfix_rescore_v1 step 3: socratic_qwq32b removed (not re-collected: QwQ template has no non-thinking mode)'
patch('scripts/run_external_local_cpu.py',
      [("    cells=('hard2verify_qwen3_8b','socratic_qwen3_8b','socratic_qwq32b')\n",
        "    cells=('hard2verify_qwen3_8b','socratic_qwen3_8b')  # thinkfix: QwQ not re-collected\n"),
       ("'location':'local CPU fallback after SSH timeouts'", "'location':'AIRCC CPU node (native python), thinkfix_rescore_v1 step 3'")], note)
patch('scripts/evaluate_external_locked_scores.py',
      [("CELLS={'hard2verify_qwen3_8b':'hard2verify','socratic_qwen3_8b':'socratic','socratic_qwq32b':'socratic'}\n",
        "CELLS={'hard2verify_qwen3_8b':'hard2verify','socratic_qwen3_8b':'socratic'}  # thinkfix: QwQ not re-collected\n")], note)
old = json.loads((M / EV / 'ANALYSIS_FREEZE.json').read_text())
new = dict(old)
new.update(time_utc=datetime.now(timezone.utc).isoformat(), external_quality_inspected=False,
           files={k: sha(DST / k) for k in old['files']},
           supersedes={'file': 'ANALYSIS_FREEZE.frozen_20260924.json', 'sha256': sha(M / EV / 'ANALYSIS_FREEZE.json')},
           thinkfix_note=('Re-evaluation of the LOCKED method (unchanged METHOD_FREEZE, BUNDLE, scoring code) on think-fix '
                          'teacher-forced records. Only change: socratic_qwq32b removed from the cell list. The registered '
                          'contrast family stays 18 (Bonferroni unchanged); 12 of 18 contrasts exist with two cells, so '
                          "EVALUATION_PROVENANCE 'complete' is expected to be false. Written before any label is opened."))
(DST / EV / 'ANALYSIS_FREEZE.json').write_text(json.dumps(new, indent=2) + '\n')
assert new['method_freeze_sha256'] == sha(DST / EV / 'METHOD_FREEZE.json'), 'method freeze changed'
print('mirror ready', sum(1 for _ in DST.rglob('*') if _.is_file()), 'files')
