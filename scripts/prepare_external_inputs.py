"""Prepare label-separated benchmark inputs; no predictions or quality evaluation."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.external_generalization.contracts import adapt_hard2verify, adapt_socratic, overlap_manifest
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.sources / "SOURCES.json").read_text())
    for entry in manifest["files"]:
        if file_hash(args.sources / entry["path"]) != entry["sha256"]:
            raise ValueError("pinned source changed: " + entry["path"])
    module_path = args.sources / "hard2verify/utils.py"
    spec = importlib.util.spec_from_file_location("hard2verify_official", module_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    encrypted = args.sources / "hard2verify/encrypted_test.jsonl"
    rows = [official.decrypt_sample(json.loads(s)) for s in encrypted.read_text(encoding="utf8").splitlines() if s.strip()]
    hard, hg = adapt_hard2verify(rows)
    soc, sg = adapt_socratic(args.sources / "socratic")
    stats = {}
    for name, answers, gold in [("hard2verify", hard, hg), ("socratic", soc, sg)]:
        atomic_json(args.out / name / "answers.json", [asdict(a) for a in answers])
        atomic_json(args.out / "evaluator_only" / (name + ".json"), [asdict(g) for g in gold])
        stats[name] = {"answers": len(answers), "steps": sum(len(a.steps) for a in answers),
                       "empty_steps_retained": sum(not s for a in answers for s in a.steps),
                       "exact_question_groups": len({a.group for a in answers}),
                       "out_of_range_annotation_rows": sum(bool(g.out_of_range_error_indices) for g in gold),
                       "answers_sha256": file_hash(args.out / name / "answers.json")}
    atomic_json(args.out / "OVERLAP.json", overlap_manifest({"hard2verify": hard, "socratic": soc}))
    stats["source_development_overlap"] = "PENDING source-question join; do not claim disjoint transfer"
    stats["encrypted_hard2verify_sha256"] = file_hash(encrypted)
    atomic_json(args.out / "INPUT_MANIFEST.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
