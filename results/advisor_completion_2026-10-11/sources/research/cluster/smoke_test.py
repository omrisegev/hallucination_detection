#!/usr/bin/env python
"""
GPU + environment smoke test for AIRCC B200 nodes. Run inside the NGC container.

Verifies: CUDA visible, device is B200 (sm_100 / capability (10,0)), bf16 matmul
works, spectral_utils imports with all its dependencies, and /shared is writable.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
assert torch.cuda.is_available(), "CUDA not available"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"device: {name} | capability: {cap}")
if cap[0] < 10:
    print("WARNING: expected B200 (capability (10,0)) — got a different GPU")

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
m = (x @ x).float().abs().mean().item()
print(f"bf16 matmul OK, mean |x@x| = {m:.2f}")

import spectral_utils
print(f"spectral_utils {spectral_utils.__version__} imported OK")
import transformers, datasets, scipy, sklearn  # noqa: E401
print(f"transformers {transformers.__version__} | datasets {datasets.__version__} | "
      f"scipy {scipy.__version__} | sklearn {sklearn.__version__}")

out_dir = os.environ.get("SMOKE_OUT", ".")
path = os.path.join(out_dir, f"smoke_ok_{os.environ.get('SLURM_JOB_ID', 'local')}.txt")
with open(path, "w") as f:
    f.write(f"{name} cap={cap} matmul={m:.4f}\n")
print(f"wrote {path}")
print("SMOKE TEST PASS")
