"""Pre-label amendment A2 (2026-10-11): METHOD_FREEZE stores Windows code paths; normalize separators so the
frozen-code comparison runs on Linux. Re-hash the patched evaluator in the new ANALYSIS_FREEZE. No label opened."""
import difflib, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])  # mirror root
rel = 'scripts/evaluate_external_locked_scores.py'
p = root / rel; src = p.read_text(encoding='utf8')
old = "frozen_code={str(Path(k).relative_to('spectral_utils/external_generalization')):v for k,v in freeze['code'].items()}"
new = ("frozen_code={str(Path(k.replace(chr(92),'/')).relative_to('spectral_utils/external_generalization')):v "
       "for k,v in freeze['code'].items()}  # thinkfix A2: freeze keys use Windows separators")
assert src.count(old) == 1, 'anchor'
out = src.replace(old, new); p.write_text(out, encoding='utf8')
orig = (root / 'scripts/evaluate_external_locked_scores.frozen_original.py').read_text(encoding='utf8')
d = ''.join(difflib.unified_diff(orig.splitlines(True), out.splitlines(True), fromfile=rel + ' (frozen)', tofile=rel + ' (thinkfix)'))
(root / 'scripts/evaluate_external_locked_scores.THINKFIX_DIFF.txt').write_text(
    '# thinkfix_rescore_v1 step 3: QwQ removed; A2 Windows-separator normalization of METHOD_FREEZE code keys\n' + d, encoding='utf8')
fz = root / 'results/lsml_external_generalization_v1/evaluation/ANALYSIS_FREEZE.json'
f = json.loads(fz.read_text())
for k in f['files']:
    f['files'][k] = hashlib.sha256((root / k).read_bytes()).hexdigest()
f.setdefault('amendments', []).append({'id': 'A2', 'time_utc': datetime.now(timezone.utc).isoformat(), 'labels_opened_before': False,
    'change': 'normalize Windows path separators in METHOD_FREEZE code keys before relative_to (failed on Linux, job 280816, '
              'before sealing; only OVERLAP.json had been written)'})
fz.write_text(json.dumps(f, indent=2) + '\n')
print('amended', root)
