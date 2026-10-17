"""Fill the re-written family IMPLEMENTATION_FREEZE.json hashes inside a mirror (run on AIRCC before scoring)."""
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1]); p = root / 'results/family_tail_external_v1/IMPLEMENTATION_FREEZE.json'
f = json.loads(p.read_text())
missing = [k for k in f['files'] if not (root / k).exists()]
if missing: raise SystemExit('missing frozen files: ' + ', '.join(missing))
f['files'] = {k: hashlib.sha256((root / k).read_bytes()).hexdigest() for k in f['files']}
f['rehashed_utc'] = datetime.now(timezone.utc).isoformat(); f['labels_opened'] = False
p.write_text(json.dumps(f, indent=2) + '\n'); print('family freeze re-hashed', len(f['files']), 'files')
