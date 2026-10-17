"""Copy immutable public releases and comparison references; never execute them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import subprocess
import urllib.request

PINS = {
    "apb": ("RUCBM/AgentProcessBench", "0a42606b178a8c69d40c5765dc05c342f921e578"),
    "tprm": ("David-Li0406/ToolPRMBench", "b43164fbb2cd2963e1906a6fe62a86e7ce05973e"),
    "aft_code": ("ZBox1005/AgentForesight", "af549083fcc6c3eff4ccc558534515e46ca0578e"),
}
DATASET = "ZBox008003/AFTraj"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CACHED_TOKENIZER_SHA = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "agent-step-telemetry-v1"})
    with urllib.request.urlopen(request, timeout=120, context=ssl.create_default_context()) as r:
        return r.read()


def tree(repo, revision):
    endpoint = f"repos/{repo}/git/trees/{revision}?recursive=1"
    try:
        raw = subprocess.check_output(["gh", "api", endpoint])
    except FileNotFoundError:
        raw = fetch("https://api.github.com/" + endpoint)
    result = json.loads(raw)
    if result.get("truncated"):
        raise ValueError("truncated source inventory")
    return {r["path"]: r for r in result["tree"] if r["type"] == "blob"}


def prepare(out, aft_revision=None, tokenizer_local=None):
    out = Path(out)
    if (out / "SOURCES.json").exists():
        raise FileExistsError("source freeze already exists; use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "files": [], "model": MODEL, "model_revision": MODEL_REVISION}

    def copy(key, repo, revision, path, url, size=None, purpose="data"):
        dest = out / key / path
        if dest.exists():
            raise FileExistsError(dest)
        data = fetch(url)
        if data.lstrip().startswith(b"<!DOCTYPE html>") or data.lstrip().startswith(b"<html"):
            raise ValueError(f"HTML/block page returned for source: {key}/{path}")
        if size is not None and len(data) != size:
            raise ValueError(f"source size mismatch: {key}/{path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        manifest["files"].append(dict(key=key, repo=repo, revision=revision,
            path=path, local_path=str(dest.relative_to(out)), bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), purpose=purpose, url=url))
        print(json.dumps({"copied": str(dest.relative_to(out)), "bytes": len(data)}), flush=True)

    for key, (repo, rev) in PINS.items():
        inventory = tree(repo, rev)
        if key == "apb":
            paths = [f"data/AgentProcessBench/{s}.jsonl" for s in ("bfcl", "gaia_dev", "hotpotqa", "tau2")]
            refs = [p for p in inventory if p == "eval/compare.py" or
                (p.startswith(("eval/results/Qwen3-8B/", "eval/results/Qwen3-4B/")) and p.endswith(".jsonl"))]
            if len(refs) != 9:
                raise ValueError("missing APB comparison references")
        elif key == "tprm":
            paths = [f"data/prmbench_{s}.json" for s in ("GTA", "bfcl", "tooltalk", "ToolSandbox")]
            refs = [p for p in inventory if p == "toolprmbench/evaluate.py" or
                    (p.startswith("results/") and p.endswith(".json"))]
            if "toolprmbench/evaluate.py" not in refs:
                raise ValueError("missing TPRM reference evaluator")
        else:
            paths = []
            refs = ["inference/data.py", "inference/prompts.py", "inference/metrics.py"]
        for path in paths + refs:
            copy(key, repo, rev, path, f"https://raw.githubusercontent.com/{repo}/{rev}/{path}",
                 inventory[path]["size"], "data" if path in paths else "reference_only")
    meta = json.loads(fetch(f"https://huggingface.co/api/datasets/{DATASET}" +
                           (f"/revision/{aft_revision}" if aft_revision else "")))
    rev = meta["sha"]
    manifest["aft_revision"] = rev
    for path in ("README.md", "aftraj_safe.parquet", "aftraj_unsafe.parquet", "splits_test.json"):
        copy("aft", DATASET, rev, path,
             f"https://huggingface.co/datasets/{DATASET}/resolve/{rev}/{path}",
             purpose="reference_only" if path == "README.md" else "data")
    # Tokenizer files only: no model weights, no inference.
    for path in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        url = f"https://huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/{path}"
        if tokenizer_local:
            data = (Path(tokenizer_local) / path).read_bytes()
            json.loads(data)
            if hashlib.sha256(data).hexdigest() != CACHED_TOKENIZER_SHA[path]:
                raise ValueError("cached tokenizer differs from verified pinned Qwen3-8B snapshot")
            dest = out / "tokenizer" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            manifest["files"].append(dict(key="tokenizer", repo=MODEL, revision=MODEL_REVISION,
                path=path, local_path=str(dest.relative_to(out)), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), purpose="tokenizer", url=url,
                cached_from=str(Path(tokenizer_local).resolve())))
        else:
            copy("tokenizer", MODEL, MODEL_REVISION, path, url, purpose="tokenizer")
    (out / "SOURCES.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--aft-revision")
    p.add_argument("--tokenizer-local", type=Path, help="Previously verified same-revision tokenizer snapshot")
    args = p.parse_args()
    prepare(args.out, args.aft_revision, args.tokenizer_local)
