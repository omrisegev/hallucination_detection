#!/usr/bin/env python
"""
P0 — environment, assets, and frozen-code audit. No GPU.

Handoff §P0. Runs before any generation and produces the evidence every later manifest
cites: the seven PDF hashes, the pinned official-code commits, the Streaming asset verdict,
and the environment/quota check. Nothing here trains, generates, or scores anything.

The rule this enforces: **do not fetch a floating branch, dataset, model or PDF inside a
full job** (§3.1). Everything resolvable is resolved and pinned here, once.

Usage (laptop, offline checks only):
    python scripts/paper_exact_p0_audit.py --out results/paper_exact/p0

Usage (cluster, with network + source cache):
    python scripts/paper_exact_p0_audit.py --out $SHARED/results/paper_exact/p0 \
        --clone-dir $SHARED/src --check-cluster
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spectral_utils.paper_exact.gates import Gate, write_blocked_assets  # noqa: E402
from spectral_utils.paper_exact.manifest import sha256_file, software_info  # noqa: E402

SOURCES_MD = os.path.join(REPO_ROOT, "papers", "PAPER_EXACT_SOURCES.md")

#: Official repositories to pin. `runnable` records whether the audit expects executable
#: code — REFRAIN's is a release-placeholder README and must never be treated as official
#: code just because the clone succeeded (handoff §P0.3).
OFFICIAL_REPOS = {
    "processbench": {"url": "https://github.com/QwenLM/ProcessBench", "runnable": True},
    "mind_the_gap": {"url": "https://github.com/QJ0114/evidence-drop", "runnable": True},
    "deepconf": {"url": "https://github.com/facebookresearch/deepconf", "runnable": True},
    "refrain": {"url": "https://github.com/RLSNLP/Adaptive-Reasoning", "runnable": False},
}

STREAMING_CODE_URL = (
    "https://anonymous.4open.science/r/Streaming-Hallucination-Detection-D186/")
STREAMING_REQUIRED_ASSETS = [
    "BBH/MuSiQue generated trajectories",
    "Claude Sonnet 4.5 step and prefix labels",
    "logical-filter decisions",
    "train/val/test split files",
    "selected representation layer specification",
    "trained probe checkpoints",
    "official evaluator / scoring code",
]

#: Models each GPU stage needs resident before submission.
REQUIRED_MODELS = {
    "L1_uprm_judge": ["Qwen/Qwen2.5-14B-Instruct"],
    "L2_prm_ceiling": ["Qwen/Qwen2.5-Math-PRM-7B", "Qwen/Qwen2.5-72B-Instruct"],
    "S1_refrain": ["Qwen/Qwen3-8B", "sentence-transformers/all-MiniLM-L6-v2"],
    "S2_leash": ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
                 "microsoft/Phi-3-mini-128k-instruct", "mistralai/Mistral-7B-v0.1"],
    "M1_M2_deepconf": ["Qwen/Qwen3-8B"],
    "C1_confirmation": ["openai/gpt-oss-20b"],
}


def _run(args, cwd=None, timeout=300):
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:  # noqa: BLE001 — an audit reports failures, it does not raise
        return -1, "", str(e)


# ── 1. PDF hashes ───────────────────────────────────────────────────────────────

def parse_sources_registry(path=SOURCES_MD) -> list:
    """Read the pinned (pdf, sha256) rows out of papers/PAPER_EXACT_SOURCES.md."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("|") or line.count("|") < 6:
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 5 or cells[0] in ("Lane", "---"):
                continue
            pdf = cells[1].strip("`")
            sha = cells[-1].strip("`")
            if re.fullmatch(r"[0-9a-f]{64}", sha):
                rows.append({"lane": cells[0], "pdf": pdf, "source": cells[2],
                             "version": cells[3], "sha256": sha})
    return rows


def audit_pdfs(gate: Gate) -> list:
    rows = parse_sources_registry()
    gate.check("sources_registry_parsed", len(rows) == 7,
               f"{len(rows)} pinned PDFs in PAPER_EXACT_SOURCES.md (expected 7)")
    report = []
    for row in rows:
        path = os.path.join(REPO_ROOT, "papers", row["pdf"])
        if not os.path.exists(path):
            report.append({**row, "present": False, "actual_sha256": None, "match": False})
            continue
        actual = sha256_file(path)
        report.append({**row, "present": True, "actual_sha256": actual,
                       "match": actual == row["sha256"]})
    bad = [r for r in report if not r["match"]]
    gate.check("pdf_hashes_match", not bad,
               "all seven committed PDFs match the registry" if not bad else
               f"{len(bad)} mismatched/missing: {[r['pdf'][:40] for r in bad]}",
               detail=bad or None)
    return report


# ── 2. official code ────────────────────────────────────────────────────────────

def audit_official_code(gate: Gate, clone_dir: str, allow_network: bool) -> dict:
    """Resolve each official repo to a pinned commit in a read-only source cache."""
    out = {}
    for name, spec in OFFICIAL_REPOS.items():
        url = spec["url"]
        entry = {"url": url, "expected_runnable": spec["runnable"]}
        if not allow_network:
            entry.update(status="skipped_no_network")
            out[name] = entry
            continue
        rc, so, se = _run(["git", "ls-remote", url, "HEAD"], timeout=90)
        if rc != 0 or not so:
            entry.update(status="unreachable", error=se[:400])
            out[name] = entry
            continue
        entry["remote_head"] = so.split()[0]
        # Resolving HEAD is enough to record a pin; cloning is what the cluster does so the
        # code is available offline inside a container with no outbound network.
        entry["status"] = "head_resolved"
        if clone_dir:
            dest = os.path.join(clone_dir, name)
            if not os.path.exists(os.path.join(dest, ".git")):
                os.makedirs(clone_dir, exist_ok=True)
                shutil.rmtree(dest, ignore_errors=True)
                rc, _, se = _run(["git", "clone", "--depth", "1", url, dest], timeout=900)
                if rc != 0:
                    entry.update(status="clone_failed", error=se[:400])
                    out[name] = entry
                    continue
            rc, commit, _ = _run(["git", "rev-parse", "HEAD"], cwd=dest)
            entry["pinned_commit"] = commit
            entry["path"] = dest
            files = []
            for root, _d, fs in os.walk(dest):
                if ".git" in root:
                    continue
                files.extend(os.path.join(root, f) for f in fs)
            py = [f for f in files if f.endswith(".py")]
            entry["n_files"] = len(files)
            entry["n_python_files"] = len(py)
            # A repo whose entire content is a README is a release placeholder, not code.
            entry["is_runnable"] = len(py) > 0
            entry["license"] = next(
                (os.path.basename(f) for f in files
                 if os.path.basename(f).upper().startswith(("LICENSE", "COPYING"))), None)
            entry["status"] = "pinned"
        out[name] = entry

    for name, spec in OFFICIAL_REPOS.items():
        e = out[name]
        if not spec["runnable"]:
            # REFRAIN: assert it is STILL a placeholder. If it ever ships real code, the
            # S1 fidelity label can be upgraded — but only deliberately, never by accident.
            gate.check(f"repo_{name}_placeholder_confirmed",
                       e.get("status") != "pinned" or not e.get("is_runnable", False),
                       "release placeholder as documented — S1 stays paper-specified"
                       if not e.get("is_runnable") else
                       "REPO NOW HAS CODE: re-audit before claiming paper-specified",
                       detail=e)
        else:
            gate.check(f"repo_{name}_pinned",
                       e.get("status") in ("pinned", "head_resolved", "skipped_no_network"),
                       f"{e.get('status')} "
                       f"{(e.get('pinned_commit') or e.get('remote_head') or '')[:12]}",
                       detail=e)
    return out


# ── 3. Streaming asset re-audit ─────────────────────────────────────────────────

def audit_streaming(gate: Gate, out_dir: str, allow_network: bool) -> dict:
    """Re-probe the anonymous Streaming endpoint (handoff §P0.4).

    On failure this writes BLOCKED_ASSETS.json and books no compute. Substituting a
    different corpus or labeller would produce a number that reads as a reproduction of
    published AUCs (87.83/86.70/93.27) and is not one.
    """
    evidence = {"url": STREAMING_CODE_URL,
                "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    reachable = False
    if allow_network:
        rc, so, se = _run(["curl", "-sS", "-m", "45", "-o", os.devnull,
                           "-w", "%{http_code}", STREAMING_CODE_URL], timeout=90)
        evidence.update(curl_rc=rc, http_code=so.strip(), curl_stderr=se[:400])
        reachable = (rc == 0 and so.strip().startswith("2"))
    else:
        evidence["note"] = "network probing disabled (--no-network)"

    evidence["reachable"] = reachable
    if not reachable:
        path = write_blocked_assets(out_dir, "W1-streaming-hallucination-detection",
                                    STREAMING_REQUIRED_ASSETS, evidence)
        evidence["blocked_assets_json"] = path
    # Not reaching the endpoint is an expected, publishable outcome — so the gate passes
    # as long as the blocked-assets row was emitted. What must never happen is a W1 number
    # produced from substitute assets, and that is structurally impossible without them.
    gate.check("streaming_assets_resolved",
               reachable or "blocked_assets_json" in evidence,
               "official assets reachable — W1 may proceed" if reachable else
               "unreachable; BLOCKED_ASSETS.json written, no compute booked",
               detail=evidence)
    return evidence


# ── 4. environment / cluster ────────────────────────────────────────────────────

def audit_environment(gate: Gate, check_cluster: bool) -> dict:
    env = {"software": software_info(), "cwd": os.getcwd(),
           "shared": os.environ.get("SHARED", "/shared/cycle2_tau_averbuch_prj/omrisegev1")}
    gate.check("evaluator_importable", True, "spectral_utils.paper_exact imports cleanly")

    rc, so, _ = _run([sys.executable, os.path.join(REPO_ROOT, "scripts", "test_paper_exact.py")],
                     timeout=900)
    env["p1_suite_rc"] = rc
    env["p1_suite_tail"] = so.splitlines()[-3:] if so else []
    gate.check("p1_regression_suite", rc == 0,
               "scripts/test_paper_exact.py passes" if rc == 0 else "P1 suite FAILED",
               detail=env["p1_suite_tail"])

    if check_cluster:
        shared = env["shared"]
        rc, so, _ = _run(["df", "-BG", "--output=avail", shared], timeout=60)
        avail = None
        if rc == 0:
            nums = re.findall(r"(\d+)G", so)
            avail = int(nums[0]) if nums else None
        env["shared_avail_gb"] = avail
        # M2's retained footprint is 20-60 GB under the scalar-rich contract; 500 GB of
        # headroom leaves room for that plus every other stage without a quota surprise.
        gate.check("shared_disk_headroom", avail is None or avail >= 500,
                   f"{avail} GB available on {shared}")
        rc, so, _ = _run(["sinfo", "-h", "-o", "%P %a %G"], timeout=60)
        env["partitions"] = so.splitlines() if rc == 0 else []
        gate.check("slurm_reachable", rc == 0, f"{len(env['partitions'])} partition lines")

        hub = os.path.join(shared, "hf_cache", "hub")
        present = set(os.listdir(hub)) if os.path.isdir(hub) else set()
        missing = {}
        for stage, models in REQUIRED_MODELS.items():
            want = [m for m in models
                    if f"models--{m.replace('/', '--')}" not in present]
            if want:
                missing[stage] = want
        env["missing_models"] = missing
        # Missing models are a prefetch task, not a failure — but they must be prefetched
        # in their own job, never downloaded inside a full run (§3.1).
        gate.check("models_prefetched", not missing,
                   "all stage models resident" if not missing else
                   f"prefetch needed: {missing}", detail=missing)
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "paper_exact", "p0"))
    ap.add_argument("--clone-dir", default=None, help="read-only official source cache")
    ap.add_argument("--check-cluster", action="store_true")
    ap.add_argument("--no-network", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    allow_net = not args.no_network
    gate = Gate("P0-assets-and-environment", args.out)

    print("=== P0: paper PDFs ===")
    pdfs = audit_pdfs(gate)
    print("\n=== P0: official code ===")
    repos = audit_official_code(gate, args.clone_dir, allow_net)
    print("\n=== P0: Streaming assets ===")
    streaming = audit_streaming(gate, args.out, allow_net)
    print("\n=== P0: environment ===")
    env = audit_environment(gate, args.check_cluster)

    report = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pdfs": pdfs, "official_code": repos, "streaming": streaming, "environment": env,
        "pinned_vllm_commit": "31f09c615f4f067dba765ce5fe7d00d880212a6d",
    }
    path = os.path.join(args.out, "P0_REPORT.json")
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, indent=2, default=str)
    os.replace(path + ".tmp", path)
    print(f"\nreport -> {path}")
    gate.finish(raise_on_fail=False)   # P0 reports; the per-stage gates block promotion
    print(f"\nP0 {'PASS' if gate.passed else 'ATTENTION: ' + ', '.join(gate.failures)}")


if __name__ == "__main__":
    main()
