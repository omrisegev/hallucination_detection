#!/usr/bin/env python
"""
Prefetch every model a paper-exact stage needs, into the shared HuggingFace hub cache.

Handoff §3.1: "Do not fetch a floating branch, dataset, model or PDF inside a full job.
Resolve and pin it in a prefetch/audit job first." This is that job. It also records the
resolved revision SHA of each repo, which the run manifests cite as `model_revision`, so a
row can be traced to the exact weights that produced it.

Downloads into `$HF_HOME/hub` (the standard cache `load_model` reads), NOT the `flat/`
directory used by `cluster/prefetch.sbatch` — that flat layout exists for Google Drive's
symlink-hostile FUSE and is not what the cluster drivers resolve.

Usage:
    sbatch -p power-gpu --qos=owner_880 -J pe_prefetch cluster/cpu_job.sbatch \
        scripts/paper_exact_prefetch.py --stages L1,S1,M1
    python scripts/paper_exact_prefetch.py --list
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: Stage -> models. Datasets are handled by `datasets` own cache on first use, which is
#: safe because they are small and revision-pinned by name.
STAGE_MODELS = {
    "L1": ["Qwen/Qwen2.5-14B-Instruct"],
    "L2": ["Qwen/Qwen2.5-Math-PRM-7B", "Qwen/Qwen2.5-72B-Instruct"],
    "S1": ["Qwen/Qwen3-8B", "sentence-transformers/all-MiniLM-L6-v2"],
    "S2": ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
           "microsoft/Phi-3-mini-128k-instruct", "mistralai/Mistral-7B-v0.1"],
    "M1": ["Qwen/Qwen3-8B"],
    "M2": ["Qwen/Qwen3-8B"],
    "C1": ["openai/gpt-oss-20b"],
}

#: Weight shards only — skip the duplicate .bin/.pth copies most repos carry alongside
#: safetensors, which would roughly double the download for no benefit.
IGNORE = ["*.pth", "*.bin", "*.msgpack", "*.h5", "*.onnx", "*.gguf"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--stages", default="L1,S1,M1")
    ap.add_argument("--models", default=None, help="explicit comma-separated repo ids")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.list:
        for stage, models in STAGE_MODELS.items():
            print(f"{stage}: {', '.join(models)}")
        return

    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        wanted = []
        for stage in [s.strip() for s in args.stages.split(",") if s.strip()]:
            if stage not in STAGE_MODELS:
                sys.exit(f"unknown stage {stage!r}; have {sorted(STAGE_MODELS)}")
            wanted += [m for m in STAGE_MODELS[stage] if m not in wanted]

    from huggingface_hub import snapshot_download
    from huggingface_hub import HfApi

    api = HfApi()
    report = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "hf_home": os.environ.get("HF_HOME", ""), "models": {}}
    failures = []
    for repo in wanted:
        print(f"\n=== {repo} ===", flush=True)
        entry = {"repo_id": repo}
        try:
            info = api.model_info(repo)
            entry["revision"] = info.sha
            path = snapshot_download(repo_id=repo, revision=info.sha, ignore_patterns=IGNORE)
            entry["local_path"] = path
            entry["bytes"] = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _d, fs in os.walk(path) for f in fs
                if os.path.exists(os.path.join(r, f)))
            entry["status"] = "ok"
            print(f"    revision {info.sha[:12]}  {entry['bytes'] / 1e9:.1f} GB  -> {path}",
                  flush=True)
        except Exception as e:  # noqa: BLE001 — one gated repo must not abort the rest
            entry.update(status="failed", error=repr(e)[:500])
            failures.append(repo)
            print(f"    FAILED: {e!r}", flush=True)
        report["models"][repo] = entry

    out = args.out or os.path.join(
        os.environ.get("SHARED", "/shared/cycle2_tau_averbuch_prj/omrisegev1"),
        "results", "paper_exact", "p0")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "MODEL_REVISIONS.json")
    # Merge rather than replace: prefetch runs stage by stage, and a later run must not
    # erase the pinned revisions an earlier one recorded.
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        old.get("models", {}).update(report["models"])
        old["written_utc"] = report["written_utc"]
        report = old
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, indent=2)
    os.replace(path + ".tmp", path)
    print(f"\nrevisions -> {path}", flush=True)
    if failures:
        print(f"FAILED: {failures} (likely gated repos needing HF_TOKEN acceptance)", flush=True)
        sys.exit(1)
    print("PREFETCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
