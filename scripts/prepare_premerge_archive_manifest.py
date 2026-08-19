#!/usr/bin/env python3
"""Select reproducible intermediates for non-destructive pre-merge Drive archival."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


TOKENIZER_PREFIXES = (
    "results/automatic_group_free_phase_a6_tokenizers_v1/materialized/",
    "results/automatic_group_free_phase_a6_tokenizers_v2/materialized/",
)
EARLY_ONLINE_PREFIXES = (
    "results/early_online_existing_data_v1/",
    "results/early_online_localization_models_v1/",
)


def archive_reason(path: str) -> str | None:
    lower = path.lower()
    name = Path(path).name.lower()
    if path.startswith(TOKENIZER_PREFIXES):
        return "tokenizer_materialization"
    if "/cells/" in path:
        return "cell_directory"
    if path.startswith(EARLY_ONLINE_PREFIXES):
        return "early_online_per_cell_intermediate"
    if "per_question" in name:
        return "per_question_output"
    if ".partial." in name or name.endswith(".partial"):
        return "partial_output"
    if name == "per_trace_convergence.csv":
        return "per_trace_intermediate"
    if path.startswith("results/local_online_comprehensive_v1/") and "warnings" in lower:
        return "per_question_warning_output"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opening-manifest", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--files-from-output", required=True)
    parser.add_argument("--remote", required=True)
    args = parser.parse_args()

    root = Path.cwd()
    opening = json.loads((root / args.opening_manifest).read_text(encoding="utf-8"))
    selected = []
    for record in opening["untracked_files"]:
        reason = archive_reason(record["path"])
        if reason:
            selected.append({**record, "reason": reason})
    selected.sort(key=lambda item: item["path"])

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_opening_manifest": args.opening_manifest,
        "source_head": opening["head"],
        "remote": args.remote,
        "copy_semantics": "non-destructive rclone copy; local sources retained",
        "file_count": len(selected),
        "total_bytes": sum(int(item["size_bytes"]) for item in selected),
        "files": selected,
    }
    json_path = root / args.json_output
    list_path = root / args.files_from_output
    if json_path.exists() or list_path.exists():
        raise SystemExit("refusing to overwrite an existing archive manifest")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    list_path.write_text("".join(f"{item['path']}\n" for item in selected), encoding="utf-8")
    print(json.dumps({"file_count": payload["file_count"], "total_bytes": payload["total_bytes"]}))


if __name__ == "__main__":
    main()
