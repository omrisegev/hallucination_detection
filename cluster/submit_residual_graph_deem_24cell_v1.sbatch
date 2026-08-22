#!/bin/bash
# Frozen AIRCC launcher. Submit partition/QoS on the sbatch command line.
# Long stages are submitted as afterany chains; cell/seed checkpoints make every
# repeated invocation idempotent.
#SBATCH --job-name=rgdeem24
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@900
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code

set -euo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
export PIP_CACHE_DIR=$SHARED/pip_cache
export PYTHONUSERBASE=/tmp/pyuserbase_${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$PYTHONUSERBASE" "$MPLCONFIGDIR"

# Compute nodes have no outbound DNS.  On AIRCC job 217597 every PyPI fetch
# failed with "Name or service not known", and because this package has no
# pyproject.toml pip fell back to the legacy `setup.py develop` path, which
# treats a dependency-fetch failure as non-fatal -- so deem was silently absent
# while pip still exited 0 and `set -e` never tripped.  deem is imported lazily
# (spectral_utils/deem_adapter.py) and only by the B1/B2 adapter controls, so
# that would not have surfaced until many hours into Stage A.
#
# Install deem + entmax from the wheelhouse staged on $SHARED from the login
# node (which does have outbound access), then install this package itself
# without touching the network.  Both are pure-Python py3-none-any wheels; the
# remaining deem requirements (torch, numpy, scipy, scikit-learn, pandas,
# joblib) already ship in the NGC image.
pip install -q --user --no-index --find-links="$SHARED/wheelhouse" "deem==0.2.0" "entmax"
pip install -q --user --no-deps -e .

# Fail loudly and immediately rather than deep inside Stage A.
python - <<'PYCHECK'
import deem, entmax, spectral_utils
version = getattr(deem, "__version__", "unknown")
if version not in ("0.2.0", "unknown"):
    raise SystemExit(f"deem {version} installed, expected 0.2.0")
print(f"dependency check ok: deem={version}")
PYCHECK

exec python cluster/run_residual_graph_deem_24cell_v1.py "$@"
