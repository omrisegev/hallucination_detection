#!/bin/bash
# CPU-only audit and immutable cluster-to-Drive backup after successful collection.
# Arguments: committed code snapshot, completed run directory, archive parent.
#SBATCH --job-name=external_archive
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
set -euo pipefail
umask 077
CODE=$1
RUN=$2
PARENT=$3
S=/shared/cycle2_tau_averbuch_prj/omrisegev1
NAME=$(basename "$RUN")
case "$RUN" in "$S"/results/lsml_external_generalization_v1/*_full_*) ;; *) exit 2;; esac
DEST="$PARENT/$NAME"
DRIVE="gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/full_archives/$NAME"
mkdir "$DEST"
python3 "$CODE/scripts/audit_external_telemetry.py" --run "$RUN" > "$RUN/AUDIT.json.tmp"
mv "$RUN/AUDIT.json.tmp" "$RUN/AUDIT.json"
for FILE in AUDIT.json TIMING.json TOKENIZATION.json ALIGNMENT_GATE.json; do
  cp "$RUN/$FILE" "$DEST/$FILE"
done
# Archive unchanged raw records privately; deterministic metadata supports retries.
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - -C "$RUN" . |
  gzip -n -1 > "$DEST/telemetry.tar.gz.tmp"
mv "$DEST/telemetry.tar.gz.tmp" "$DEST/telemetry.tar.gz"
python3 - "$DEST" "$RUN" "$DRIVE" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
dest, run, drive = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
timing = json.loads((dest/'TIMING.json').read_text())
audit = json.loads((dest/'AUDIT.json').read_text())
assert timing['complete'] and timing['identity']['mode'] == 'full'
assert audit['status'] == 'PASS' and not audit['quality_evaluated']
assert audit['answers'] == len(timing['identity']['selected_uids'])
hashes = {}
for path in sorted(dest.iterdir()):
    with path.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    hashes[path.name] = {'bytes':path.stat().st_size, 'sha256':sha}
manifest = {'schema':1, 'run':str(run), 'drive':drive,
            'job_id':os.environ.get('SLURM_JOB_ID'), 'files':hashes,
            'answers':audit['answers'], 'totals':audit['totals'],
            'identity':timing['identity'], 'quality_evaluated':False,
            'restore':'Download telemetry.tar.gz to private storage and extract; contains original records and audit.'}
(dest/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'audit':'PASS','answers':audit['answers'],'totals':audit['totals'],
                  'archive_bytes':hashes['telemetry.tar.gz']['bytes']}),flush=True)
PY
"$S/bin/rclone" --config "$S/.config/rclone/rclone.conf" copy "$DEST" "$DRIVE" --immutable
"$S/bin/rclone" --config "$S/.config/rclone/rclone.conf" check "$DEST" "$DRIVE" --one-way
printf 'ARCHIVE_VERIFIED %s %s\n' "$NAME" "$DRIVE"
