#!/bin/bash
# Downstream PRMBench/ProcessBench re-scoring for thinkfix_rescore_v1 (CPU only, no container).
# Working dir = $SLURM_SUBMIT_DIR, which contains the mirror directory literally named
# 'C:\Users\omris\TAU\hallucination_detection', so the frozen scripts' hardcoded Windows MAIN
# constants resolve to it unchanged. Full Windows paths stored inside JSON manifests are created
# as literal file names (symlinks) from LINKS.json. Libraries: pyenv_native_312_20261011
# (numpy 2.2.4, scipy 1.17.1, pandas 3.0.3, scikit-learn 1.8.0, torch 2.6.0 cpu = local versions).
# Usage: sbatch -p cpu --chdir=$SH/thinkfix_downstream downstream.sbatch <script rel. to mirror> [args]
#SBATCH --job-name=thinkfix_downstream
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
set -uo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/pyenv_native_312_20261011/activate.sh
export MPLBACKEND=Agg
cd "$SLURM_SUBMIT_DIR"
M='C:\Users\omris\TAU\hallucination_detection'
python3 - <<'EOF'
import json, os
M = r'C:\Users\omris\TAU\hallucination_detection'
links = json.load(open(os.path.join(M, 'LINKS.json')))
for win, rel in links.items():
    if not os.path.lexists(win):
        os.symlink(os.path.join(os.getcwd(), M, rel), win)
print('[links]', len(links))
EOF
echo "[sbatch] job $SLURM_JOB_ID on $(hostname) in $(pwd), args: $*"
python3 -c "import numpy, scipy, pandas, sklearn; print('[env]', numpy.__version__, scipy.__version__, pandas.__version__, sklearn.__version__)"
exec python3 -B "$M/$1" "${@:2}"
