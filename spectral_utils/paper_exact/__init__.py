"""
paper_exact — the `paper_exact_acquisition_v1` contract.

One acquisition schema for every new cluster run in the localization / early-online
comparison cycle (HANDOFF_paper_exact_cluster_acquisition_2026-08-16.md).

The split is deliberate and load-bearing:

    cluster (GPU)   generation, teacher forcing, hidden-state extraction, forced closure
    laptop  (CPU)   feature construction, calibration, bootstraps, plots, reports

Everything the GPU produces is written once, immutably, sharded, hashed, and
manifested; every number in the final tables is then derived offline from that
acquisition without touching a GPU again. A run that cannot be re-derived offline from
its own shards is a failed run, however good its numbers look.

Modules
-------
manifest    RUN_MANIFEST.json — schema, builder, verifier (handoff §3.1)
shards      atomic sharded writer + INDEX.jsonl/STATUS.json/SHA-256 (handoff §3.3)
gates       GATE.json — machine-checkable pass/fail per stage (handoff §6)
evaluator   the ONE frozen metric library: answer parse, ProcessBench F1, SLA,
            AUROC/AUPRC, pass@1, grouped bootstrap (handoff §P0.5)
telemetry   per-token channel extraction, raw vs post-warper kept distinct (§3.2)
deepconf    the pinned official DeepConf confidence function + equality audit (§M1)
refrain     REFRAIN / "Stop When Enough" native policy, ACL 2026 (§S1)
leash       LEASH native policy with its declared sensitivity grid (§S2)

Fidelity labels (handoff §1) — every emitted row carries exactly one:

    official-exact            official data + checkpoint + code commit + prompt +
                              decoding + parser + metric
    paper-specified           implemented from a sufficiently detailed paper, no
                              runnable official code exists
    paper-specified-partial   the paper omits constants; we declare ours and run a
                              pre-registered sensitivity grid
    adapted-common-protocol   concept applied to our rows/closure/task
    published-context-only    number quoted, not rerun
    blocked-assets            required official asset unavailable

Published values are regression targets, never promotion gates. Nothing here decides
whether a run is good; it decides whether a run is *valid*.
"""

FIDELITY_LABELS = (
    "official-exact",
    "paper-specified",
    "paper-specified-partial",
    "adapted-common-protocol",
    "published-context-only",
    "blocked-assets",
)

#: The four result lanes. Never rank rows from different lanes in one table (handoff §1).
LANES = (
    "localization",
    "prefix-detection",
    "single-trace-stopping",
    "multi-trace-adaptive-compute",
)

SCHEMA_VERSION = "paper_exact_acquisition_v1"

from .manifest import (  # noqa: E402,F401
    build_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)
from .shards import ShardWriter, read_shards, verify_shards  # noqa: E402,F401
from .gates import Gate, write_gate  # noqa: E402,F401

__all__ = [
    "FIDELITY_LABELS",
    "LANES",
    "SCHEMA_VERSION",
    "build_manifest",
    "sha256_file",
    "verify_manifest",
    "write_manifest",
    "ShardWriter",
    "read_shards",
    "verify_shards",
    "Gate",
    "write_gate",
]
