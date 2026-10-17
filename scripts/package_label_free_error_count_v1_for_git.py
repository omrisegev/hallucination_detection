"""Create/verify a Git reading delivery while preserving the full sealed study."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = Path('results/label_free_error_count_v1_2026-10-12')
SOURCE = ROOT / REL
BRANCH = 'codex/label-free-error-count-v1'
BASE = '07cb7a45fb01ab62ade51fa5e59717c1f27fd8ff'
SKIP = {'__pycache__', 'mpl_cache', '.mpl_cache', '.pytest_cache'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def verify(package):
    package = package.resolve()
    manifest = json.loads((package / 'GITHUB_DELIVERY_MANIFEST.json').read_text())
    for base, records in [(package, manifest['files']), (package.parents[1], manifest['runtime_files'])]:
        for record in records:
            path = (base / record['path']).resolve()
            if not path.is_relative_to(base) or not path.is_file() or sha(path) != record['sha256']:
                raise RuntimeError('Reading delivery mismatch: ' + record['path'])
    print('READING_DELIVERY_PASS', len(manifest['files']), 'artifacts', len(manifest['runtime_files']), 'runtime files')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--verify-delivery', type=Path)
    args = parser.parse_args()
    if args.verify_delivery:
        verify(args.verify_delivery)
        return
    if not args.destination: parser.error('--destination or --verify-delivery is required')
    destination = args.destination.resolve()
    if destination == ROOT.resolve(): raise RuntimeError('Never overwrite the analysis checkout')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=destination, text=True).strip()
    if head != BASE: raise RuntimeError('Unexpected publication base: ' + head)
    # Authenticate original outputs and source identities before copying anything.
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_label_free_error_count_delivery.py')], cwd=ROOT, check=True)
    target = destination / REL
    if target.exists(): raise RuntimeError('Use an empty delivery destination')
    target.mkdir(parents=True)
    omitted = []
    for path in sorted(SOURCE.rglob('*')):
        if not path.is_file() or SKIP.intersection(path.relative_to(SOURCE).parts): continue
        relative = str(path.relative_to(SOURCE))
        if path.suffix in {'.npz', '.pkl', '.pyc'}:
            omitted.append({'path': relative, 'size': path.stat().st_size, 'sha256': sha(path)})
            continue
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
    shutil.copyfile(target / 'README.md', target / 'README_ANALYSIS_SNAPSHOT.md')
    (target / '.gitignore').write_text('*.npz\n*.pkl\n__pycache__/\nmpl_cache/\n.mpl_cache/\n')
    with (target / 'RAW_DATA_CATALOG.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['path', 'size', 'sha256']); writer.writeheader(); writer.writerows(omitted)
    catalog = ['# Numerical payloads retained outside Git\n',
               'The full sealed local bundle remains at `' + str(SOURCE) + '`. These outputs have no verified cloud copy. Git contains all CSV/JSON, models, source, figures and review evidence, but not these NPZ/PKL arrays.\n',
               'Full arithmetic replay requires the matching local study and frozen advisor prerequisite. Do not disable checksum guards to make a partial clone pass.\n']
    for row in omitted:
        catalog.append(f"- `{row['path']}` — {row['size']} bytes; SHA256 `{row['sha256']}`")
    (target / 'RAW_DATA_CATALOG.md').write_text('\n'.join(catalog) + '\n')
    original = (target / 'report/index.html').read_text()
    banner = "<aside style='padding:16px;border:2px solid #c99a52;background:#fff7e8'><strong>GitHub reading delivery.</strong> Scientific results are unchanged from the reviewed local analysis. NPZ payloads are retained outside Git; <a href='../RAW_DATA_CATALOG.md'>payload catalog</a>. Original local/unpublished disclosures describe that analysis snapshot. <a href='../CLAUDE_REVIEW.md'>Reviewer entry point</a>.</aside>"
    at = original.index('<h1')
    (target / 'report/index_github.html').write_text(original[:at] + banner + original[at:])
    (target / 'README.md').write_text(f'''# Label-free error-count study — GitHub reading delivery

Branch: `{BRANCH}`, based on advisor v11 commit `{BASE}`.
The A–D family is complete. No primary candidate meets the combined success gate.
All evidence remains internal development validation on previously inspected data.

Start with [Claude review guide](CLAUDE_REVIEW.md), [Hebrew findings](FINDINGS_HE.md),
[reading HTML](report/index_github.html), and [independent review](review/FINAL_REVIEW.md).
The [original HTML](report/index.html) is preserved byte-for-byte.

## Delivery scope

All original CSV/JSON results, fitted-model JSON, checkpoints' identity/model
records, source code, logs, scientific SVG/PNG figures and review receipts are
included. NPZ/PKL payloads remain outside Git and are catalogued with sizes and
SHA256 in [RAW_DATA_CATALOG.md](RAW_DATA_CATALOG.md). The inherited advisor v11
reading delivery similarly retains its full numerical inputs outside Git.
**Git alone is a reading/audit delivery, not a complete numerical replay bundle.**

`PACKAGE_MANIFEST.json`, prediction seals and original review receipts describe
the full original local study, including omitted arrays. They remain unchanged;
they are not hashes of a complete clone. `GITHUB_DELIVERY_MANIFEST.json` hashes
this separate publication. [README_ANALYSIS_SNAPSHOT.md](README_ANALYSIS_SNAPSHOT.md)
preserves the exact original reproduction guide and local-publication status.

Verify the reading delivery using the configured local Python runtime:

```sh
.venv/bin/python scripts/package_label_free_error_count_v1_for_git.py --verify-delivery {REL}
.venv/bin/python -m unittest discover -s tests -p 'test_error_count*.py'
```

The synthetic tests do not require the omitted real-data arrays. Worktrees reuse
the main checkout's `/Users/osegev/Desktop/hallucination_detection/.venv/bin/python`.
Full replay requires the original complete study, the frozen advisor package,
its expected runtime, and a new output directory. No new inference, fit,
threshold search, or external confirmation accompanies this publication.
The Claude Artifact was not modified and no review was sent to an external service.
''')
    (target / 'CLAUDE_REVIEW.md').write_text('''# Review request: label-free error-count family A–D

The user wants Claude to examine the completed experiment. This document is a
handoff; it does not claim Claude has reviewed it or authorize new experiments.

## Read first

1. README.md and FINDINGS_HE.md; report/index_github.html.
2. PROTOCOL.json and chronological AMENDMENTS.json (especially the numerical
   bridge repair and the quota-mechanism diagnostic's information timeline).
3. review/FINAL_REVIEW.md and its independent arithmetic receipts.
4. evaluation/METRICS.csv, CONTRASTS.csv, SUCCESS.json, CHECKPOINT_STATUS.csv,
   and FOLD_METRICS.csv. Inspect per-type MS valid F1 and PS/DR/MS Sensitivity.
5. diagnostics/QUOTA_SCIENCE.json, EMISSION_CALIBRATION.json,
   COMPONENT_RESULTS.json and COUNT_ORACLES.json.

## Questions to examine

- Do fitting interfaces avoid target labels, types, PRM scores and holdout
  statistics? Are groups, weights, survivors, normalizations, thresholds and
  triple selection donor-only with source-disjoint FOLDS_V2 identities?
- Does the compensated raw/answer-z bridge preserve masks, affine coefficients
  and canonical group scores at the unchanged 1e-9 tolerance? Distinguish the
  first numerical failure from later genuine infeasible MoM emissions.
- Is IU-PCR canonical, fitted once per final bank/fold, and ordering shared
  across all quantity arms? Are ranking metrics strictly within answers?
- Are the positive-orientation, conditional voter independence and conditional
  step-independence assumptions justified or merely assumed? B selects one
  stable triple; stability and low correlation do not establish independence.
- Does D separate prior refitting (rho=0) from the structural-zero component,
  and report boundary/convergence/identification checks without silent repairs?
- Are posterior/count-PMF mean and lower median comparisons otherwise identical?
- Are noncontrols, synthetic controls and all true-zero cohorts distinguished,
  failure counts missing rather than zero, and partial coverage explicitly shown?
- Do all seven candidates remain in the corrected declaration family, with
  shared 20,000 source-group draws and the one-percentage-point recall allowance?
- Does the quota diagnostic support sensitivity without claiming MoM is
  mathematically the vote fraction or choosing q=10% because it matches truth?
- Is the conclusion balanced: no protected improvement, but LOCAL can improve
  over old MoM; R2 retains similar PRMScore and higher recall despite worse MAE;
  stable B improves count association without calibrated allocation; D improves
  clean abstention while missing more erroneous answers?
- Are oracle component gains kept nondeployable and nonadditive, and next-step
  continuous/dependence modelling clearly a hypothesis rather than a proven fix?

## Code and limits

Main code: spectral_utils/error_count.py, error_count_inputs.py,
error_count_evaluation.py; scripts/run_label_free_error_count_v1.py and
evaluate_label_free_error_count_v1.py. The separate final renderer is
scripts/report_label_free_error_count_v1.py. Synthetic tests are under tests/.
Diagnostic and independent audit scripts are archived beside their results.

Read-only static and aggregate arithmetic review can be done from Git. Full
array-level reconstruction needs the catalogued original numerical bundle and
the inherited advisor inputs. State what was actually reproduced; do not claim
full reproduction when arrays are unavailable. Do not refit, repair failed
arms, select thresholds, or overwrite any frozen artifact as part of this review.

Return accept/partial-accept/push-back findings with file/line evidence, severity,
and a distinction between arithmetic bugs, assumption failures and claim limits.
''')
    runtime_paths = list(json.loads((SOURCE / 'CODE_FREEZE.json').read_text()))
    runtime_paths += ['scripts/report_label_free_error_count_v1.py', 'scripts/reproduce_label_free_error_count_v1.py',
                      'scripts/verify_label_free_error_count_delivery.py', 'scripts/package_label_free_error_count_v1_for_git.py']
    runtime_paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'tests').glob('test_error_count*.py'))]
    for relative in runtime_paths:
        output = destination / relative; output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, output)
    progress = (ROOT / 'PROGRESS.md').read_text()
    section = progress[progress.index('**Label-free error-count completion'):progress.index('**Advisor CPU completion addendum')]
    old = (destination / 'PROGRESS.md').read_text(); title, rest = old.split('\n', 1)
    (destination / 'PROGRESS.md').write_text(title + '\n\n' + section + rest)
    history = (ROOT / 'HISTORY.md').read_text(); section = history[history.index('### Step 292 [label-free error count]'):]
    with (destination / 'HISTORY.md').open('a') as stream: stream.write('\n---\n\n' + section)
    for name, start, end in [('Research_Directions.md', '## PRMBench count estimation', '## Active priority — consolidated'),
                             ('Experiments_Report.md', '## PRMBench error-count addendum', '## Background')]:
        original_doc = (ROOT / name).read_text(); section = original_doc[original_doc.index(start):original_doc.index(end)]
        old = (destination / name).read_text(); split = old.index('---') + 3
        (destination / name).write_text(old[:split] + '\n\n' + section + old[split:])
    note = '\n**Git publication follow-up (2026-10-12):** The completed error-count reading delivery is on `' + BRANCH + '`, based on the advisor v11 publication. The full numerical bundles remain local; reading-delivery manifests and payload catalogs distinguish them. This does not update a Claude Artifact or claim external validation.\n'
    for name in ['PROGRESS.md', 'HISTORY.md']:
        with (destination / name).open('a') as stream: stream.write(note)
    records = [{'path': str(p.relative_to(target)), 'size': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(target.rglob('*')) if p.is_file()]
    runtime_records = [{'path': p, 'size': (destination / p).stat().st_size, 'sha256': sha(destination / p)}
                       for p in sorted(set(runtime_paths + ['PROGRESS.md', 'HISTORY.md', 'Research_Directions.md', 'Experiments_Report.md']))]
    receipt = {'scope': 'Git reading delivery; full original arrays remain outside Git', 'branch': BRANCH, 'base_commit': head,
               'source_package_manifest_sha256': sha(SOURCE / 'PACKAGE_MANIFEST.json'), 'source_prediction_seal_sha256': sha(SOURCE / 'PREDICTION_SEAL.json'),
               'source_html_sha256': sha(SOURCE / 'report/index.html'), 'preserved_html_sha256': sha(target / 'report/index.html'),
               'reading_html_sha256': sha(target / 'report/index_github.html'), 'files': records, 'runtime_files': runtime_records,
               'omitted_files': omitted, 'delivered_bytes': sum(r['size'] for r in records), 'omitted_bytes': sum(r['size'] for r in omitted),
               'transformations': ['README replaced by Git-specific guide; exact original preserved', 'separate HTML publication banner; scientific original unchanged', 'NPZ/PKL arrays catalogued rather than committed', 'only count-study doc sections appended to previous publication docs', 'Claude review handoff added; no external review sent']}
    assert receipt['source_html_sha256'] == receipt['preserved_html_sha256']
    dump(target / 'GITHUB_DELIVERY_MANIFEST.json', receipt)
    verify(target)
    print('PREPARED_GIT_DELIVERY', len(records), 'artifacts', receipt['delivered_bytes'], 'bytes')


if __name__ == '__main__': main()
