"""Pin small public source/data releases; never download model weights locally."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "lsml-research/1"}), timeout=60) as response:
        return response.read()


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"repositories": {}, "models": {}, "datasets": {}, "files": []}
    for name, repo in [("hard2verify", "SalesforceAIResearch/Hard2Verify"),
                       ("socratic", "Xiang-Li-oss/Socratic-PRMBench"),
                       ("prmbench", "PRMBench/PRMBench")]:
        tree = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"))
        revision = tree["sha"]
        manifest["repositories"][name] = {"repo": repo, "revision": revision}
        for item in tree["tree"]:
            path = item["path"]
            keep = (name == "hard2verify" and path in ["utils.py", "model.py", "run_eval.py", "README.md", "LICENSE.txt"]
                    or name == "socratic" and path.endswith((".jsonl", "README.md"))
                    or name == "prmbench" and path.endswith(".py") and any(s in path.lower() for s in ["eval", "metric", "generative"]))
            if not keep:
                continue
            data = fetch(f"https://raw.githubusercontent.com/{repo}/{revision}/{path}")
            target = out / name / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest["files"].append({"path": str(target.relative_to(out)).replace("\\", "/"),
                                      "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        print(name, revision, flush=True)
    for model in ["Qwen/Qwen3-8B", "Qwen/QwQ-32B", "Qwen/Qwen2.5-Math-PRM-7B", "universalprm/Universal-PRM"]:
        metadata = json.loads(fetch(f"https://huggingface.co/api/models/{model}"))
        manifest["models"][model] = metadata["sha"]
    dataset = "Salesforce/Hard2Verify"
    metadata = json.loads(fetch(f"https://huggingface.co/api/datasets/{dataset}"))
    manifest["datasets"][dataset] = {"revision": metadata["sha"], "files": metadata.get("siblings", [])}
    (out / "SOURCES.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8", newline="\n")
    print("Pinned source manifest:", out / "SOURCES.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out)
