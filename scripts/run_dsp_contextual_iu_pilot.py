#!/usr/bin/env python3
"""Orchestrate the frozen DSP-contextual IU pilot and enforce its gates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dsp_contextual_iu_pilot_v1"
PROTOCOL = ROOT / "docs" / "experiments" / "DSP_CONTEXTUAL_IU_PILOT_V1.md"
SYNTHETIC = ROOT / "scripts" / "dsp_contextual_iu_synthetic.py"
CORE = ROOT / "spectral_utils" / "contextual_iu.py"
UPCR = ROOT / "spectral_utils" / "upcr.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _stage_skip(stage: str, blocking_status: str) -> dict:
    value = {
        "stage": stage,
        "status": "SKIPPED_BY_S0",
        "executed": False,
        "blocking_status": blocking_status,
        "labels_accessed": False,
        "reason": "The frozen S0 gate failed; opening real-data labels is prohibited.",
    }
    _write_json(OUT / f"STAGE_{stage[1:]}_DECISION.json", value)
    return value


def _markdown(decision: dict, s0: dict) -> str:
    mechanics = s0["mechanics"]
    gates = s0["gates"]
    summary_path = OUT / "STAGE_0_SYNTHETIC_SUMMARY.csv"
    rows = []
    with summary_path.open(encoding="utf-8") as handle:
        import csv
        rows = list(csv.DictReader(handle))
    table = [
        "| world | IU AUROC | contextual AUROC | delta | wins | target-active mass | worst delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            "| {world} | {base:.4f} | {local:.4f} | {delta:+.4f} | {wins}/20 | {mass:.3f} | {worst:+.4f} |".format(
                world=row["world"],
                base=float(row["mean_baseline_auc"]),
                local=float(row["mean_contextual_auc"]),
                delta=float(row["mean_delta"]),
                wins=int(row["wins"]),
                mass=float(row["mean_target_active_family_mass"]),
                worst=float(row["worst_delta"]),
            )
        )
    failed = [name for name, passed in gates.items() if not passed]
    return "\n".join([
        "# DSP-Contextual IU Router pilot v1",
        "",
        "**Verdict: `STOP_NO_ROUTING_SIGNAL`.**",
        "",
        "The CPU-only S0 falsification gate failed before any real ProcessBench labels were opened. "
        "The implemented router therefore did not proceed to S1--S4.",
        "",
        "## Synthetic result",
        "",
        *table,
        "",
        "The failure is substantive, not merely low power. In the informative world the router "
        "placed 58.9% of its mass on the three target-active families (50% is neutral), yet that "
        "weakly correct preference still worsened AUROC. Under coherent nuisance the target-active "
        "mass collapsed to 34.1% and the score suffered a large loss. The context-independent null "
        "was approximately inert, so the problem is the target alignment and safety of adaptation "
        "rather than uncontrolled numerical drift.",
        "",
        "Failed gates: " + ", ".join(f"`{name}`" for name in failed) + ".",
        "",
        "## Mechanical audit",
        "",
        f"- Exact global fallback: `{mechanics['fallback_exact']}`.",
        f"- Question-duplication score delta: `{mechanics['question_duplication_max_abs_score_delta']:.3e}`.",
        f"- Question-duplication weight delta: `{mechanics['question_duplication_max_abs_weight_delta']:.3e}`.",
        f"- Observational-equivalence identity: `{mechanics['observational_equivalence_exact']}`.",
        "- Covariance-entry IU is regression-tested against ordinary `upcr_fit`.",
        "- The initial impossible `k=32`/`n_eff>=32` combination was corrected before the intentional run by adding eight neighbour questions of headroom.",
        "",
        "## Interpretation",
        "",
        "DSP states do contain regime structure, but this implementation cannot turn that structure "
        "into target-aligned family reliability.  Improving local manifold estimation with LPCA, "
        "LTSREx, or LEGO would make the same local geometry more stable; it would not supply the "
        "missing correctness contrast exposed by the coherent-nuisance failure.",
        "",
        "This closes the registered DSP-contextual covariance router.  It does not rule out a router "
        "fed by an independent target-relevant observation or a verified intervention.  No fresh "
        "inference run is requested.",
        "",
        "## Scope and audit trail",
        "",
        "All evidence is retrospective/synthetic premise evidence.  S1--S4 were skipped by the "
        "frozen gate; no real-data score, label, cache, GPU, cluster, or Drive operation occurred.",
        "",
    ])


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, str(SYNTHETIC)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    (OUT / "STAGE_0_STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
    (OUT / "STAGE_0_STDERR.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"S0 synthetic runner failed with exit code {completed.returncode}"
        )
    s0 = json.loads((OUT / "STAGE_0_DECISION.json").read_text(encoding="utf-8"))
    if s0["passed"]:
        raise RuntimeError(
            "S0 unexpectedly passed. Freeze and independently review the real-data score boundary before labels are opened."
        )

    skipped = {
        stage: _stage_skip(stage, s0["status"])
        for stage in ("S1", "S2", "S3", "S4")
    }
    decision = {
        "status": "STOP_NO_ROUTING_SIGNAL",
        "s0_passed": False,
        "stages": {"S0": s0["status"], **{key: value["status"] for key, value in skipped.items()}},
        "localization": "NOT_OPENED",
        "early_detection": "NOT_OPENED",
        "request_fresh_confirmation": False,
        "new_inference_recommended": False,
        "reason": "The registered router failed informative and coherent-nuisance S0 gates.",
        "retrospective_premise_evidence_only": True,
    }
    _write_json(OUT / "DECISION.json", decision)
    audit = {
        "status": "PASS_MECHANICS_FAIL_PREMISE",
        "labels_accessed": False,
        "real_caches_accessed": False,
        "new_inference": False,
        "gpu_hours": 0,
        "drive_mutation": False,
        "s0_gates": s0["gates"],
        "mechanics": s0["mechanics"],
        "protocol_sha256": _sha256(PROTOCOL),
        "core_sha256": _sha256(CORE),
        "upcr_sha256": _sha256(UPCR),
        "synthetic_runner_sha256": _sha256(SYNTHETIC),
    }
    _write_json(OUT / "AUDIT.json", audit)
    report = _markdown(decision, s0)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "REPORT.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>DSP Contextual IU Pilot</title>"
        "<style>body{font:16px/1.5 system-ui;max-width:980px;margin:40px auto;padding:0 20px}"
        "pre{white-space:pre-wrap}</style></head><body><pre>"
        + html.escape(report)
        + "</pre></body></html>\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "COMPLETE_STOPPED_AT_S0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "decision_sha256": _sha256(OUT / "DECISION.json"),
        "audit_sha256": _sha256(OUT / "AUDIT.json"),
        "report_sha256": _sha256(OUT / "REPORT.md"),
        "report_html_sha256": _sha256(OUT / "REPORT.html"),
        "new_inference": False,
        "gpu_hours": 0,
        "drive_mutation": False,
    }
    _write_json(OUT / "RUN_MANIFEST.json", manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
