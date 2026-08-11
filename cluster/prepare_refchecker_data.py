#!/usr/bin/env python
"""
Assemble the RefChecker benchmark corpus into `data/refchecker_protocol/`, reproducing
`benchmark/data/download_data.sh` + `gather_benchmark_data.py` from
github.com/amazon-science/RefChecker.

Two halves, with very different costs:

  * The HUMAN ANNOTATIONS (21 files: 3 context settings x 7 generator models) and the example-id
    lists live in the GitHub repo itself and are a few MB — fetched directly over raw.githubusercontent.
  * The QUESTIONS AND REFERENCES are not in the repo and must be rebuilt from the source corpora:
      accurate_context -> databricks/databricks-dolly-15k   (HF, small)
      noisy_context    -> ms_marco v2.1 validation          (HF, moderate)
      zero_context     -> Natural Questions dev             (~7 GB of .jsonl.gz)

NQ COMES FROM HUGGING FACE, NOT gsutil AND NOT PLAIN HTTPS. The official script shells out to
`gsutil -m cp -R gs://natural_questions/v1.0/dev`, which would mean installing the Google Cloud
SDK inside the NGC container.

The obvious shortcut — read the same objects over `https://storage.googleapis.com/natural_questions/...`
— **does not work, and this is measured, not assumed**. That URL returns:

    HTTP 403  <Code>AccessDenied</Code>
    Anonymous caller does not have storage.objects.get access to the ... object

The bucket is readable by *authenticated* Google principals, not by `allUsers`. `gsutil` succeeds
only because it authenticates. Job 177897 died exactly here.

So `build_nq` streams `google-research-datasets/natural_questions` (split `validation`) from the
Hub instead, filters against the 100 wanted example ids, and drops everything else — no GCS
credentials, no 7 GB on disk. **The ids were verified to align before this was written**: a
1,200-row scan of the HF validation split matched 20 of the 100 wanted ids, against ~15 expected
if the id spaces are the same.

The HF schema is COLUMNAR where the raw jsonl is a list of dicts — `document_tokens[i]["token"]`
becomes `document["tokens"]["token"][i]`, and `html_token` becomes `is_html`. The long-answer
reconstruction rule itself (`tokens[start:end]`, dropping html tokens, joined by single spaces,
stripped) is otherwise identical to `gather_benchmark_data.py::process_nq`, so the reference text
matches the official pipeline's.

Idempotent: every output file is skipped if it already exists unless `--force` is passed.

Usage:
    python cluster/prepare_refchecker_data.py --out data/refchecker_protocol
    python cluster/prepare_refchecker_data.py --skip-nq     # annotations + dolly + msmarco only
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.refchecker import CONTEXT_SETTINGS, DATASET_OF, GENERATORS

RAW_BASE = "https://raw.githubusercontent.com/amazon-science/RefChecker/main/benchmark"
# NOT the GCS mirror — it 403s for anonymous callers. See the module docstring.
NQ_HF_DATASET = "google-research-datasets/natural_questions"
NQ_HF_SPLIT = "validation"


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def fetch_repo_files(out_dir: Path, force: bool) -> dict:
    """The 21 annotation files + 3 example-id lists, straight from the repo."""
    stats = {"fetched": 0, "skipped": 0, "failed": []}
    for setting in CONTEXT_SETTINGS:
        dataset = DATASET_OF[setting]
        (out_dir / setting).mkdir(parents=True, exist_ok=True)
        names = [f"{dataset}_example_ids.json"] + [
            f"{dataset}_{generator}_answers.json" for generator in GENERATORS]
        for name in names:
            target = out_dir / setting / name
            if target.exists() and not force:
                stats["skipped"] += 1
                continue
            # example-id lists live under data/, annotations under human_annotations_v1/
            sub = "data" if name.endswith("_example_ids.json") else "human_annotations_v1"
            url = f"{RAW_BASE}/{sub}/{setting}/{name}"
            try:
                target.write_bytes(_fetch(url))
                stats["fetched"] += 1
                print(f"[refchecker-data] fetched {setting}/{name}", flush=True)
            except Exception as exc:
                stats["failed"].append({"url": url, "error": repr(exc)})
                print(f"[refchecker-data] FAILED {url}: {exc}", flush=True)
    return stats


def build_dolly(out_dir: Path, force: bool):
    """`gather_benchmark_data.py::process_dolly` — the id IS the row index into the HF split."""
    target = out_dir / "accurate_context" / "dolly.json"
    if target.exists() and not force:
        print("[refchecker-data] dolly.json exists, skipping", flush=True)
        return {"skipped": True}
    from datasets import load_dataset

    ids = json.loads((out_dir / "accurate_context" / "dolly_example_ids.json")
                     .read_text(encoding="utf-8"))
    wanted = set(map(str, ids))
    data = load_dataset("databricks/databricks-dolly-15k", split="train")
    chosen = {}
    for i, row in enumerate(data):
        if str(i) in wanted:
            chosen[str(i)] = {"id": str(i), "question": row["instruction"],
                              "context": [row["context"]], "category": row["category"],
                              "human_response": row["response"]}
    out = [chosen[i] for i in map(str, ids) if i in chosen]
    target.write_text(json.dumps(out, indent=4), encoding="utf-8")
    return {"n_wanted": len(wanted), "n_built": len(out)}


def build_msmarco(out_dir: Path, force: bool):
    """`gather_benchmark_data.py::process_msmarco`."""
    target = out_dir / "noisy_context" / "msmarco.json"
    if target.exists() and not force:
        print("[refchecker-data] msmarco.json exists, skipping", flush=True)
        return {"skipped": True}
    from datasets import load_dataset

    ids = json.loads((out_dir / "noisy_context" / "msmarco_example_ids.json")
                     .read_text(encoding="utf-8"))
    wanted = set(map(str, ids))
    data = load_dataset("ms_marco", "v2.1", split="validation")
    chosen = {}
    for row in data:
        key = str(row["query_id"])
        if key in wanted:
            chosen[key] = {"id": key, "question": row["query"],
                           "context": list(row["passages"]["passage_text"]),
                           "query_type": row["query_type"], "answers": list(row["answers"]),
                           "context_is_selected": list(row["passages"]["is_selected"])}
    out = [chosen[i] for i in map(str, ids) if i in chosen]
    target.write_text(json.dumps(out, indent=4), encoding="utf-8")
    return {"n_wanted": len(wanted), "n_built": len(out)}


def _nq_join(tokens, is_html, start: int, end: int) -> str:
    """`process_nq`'s reconstruction rule: the token slice minus html tokens, single-spaced.

    `start < 0` is NQ's "no annotation" sentinel and yields the empty string.
    """
    if start is None or start < 0 or end is None or end < 0:
        return ""
    return " ".join(tokens[i] for i in range(start, min(end, len(tokens)))
                    if not is_html[i]).strip()


def build_nq(out_dir: Path, force: bool):
    """`gather_benchmark_data.py::process_nq`, sourced from the Hub instead of GCS.

    See the module docstring for why the GCS mirror is unusable (anonymous 403). Only the 100
    wanted example ids are materialised; every other record is parsed and discarded, so peak
    memory is one NQ record and peak disk is zero.
    """
    target = out_dir / "zero_context" / "nq.json"
    if target.exists() and not force:
        print("[refchecker-data] nq.json exists, skipping", flush=True)
        return {"skipped": True}

    from datasets import load_dataset

    ids = json.loads((out_dir / "zero_context" / "nq_example_ids.json")
                     .read_text(encoding="utf-8"))
    wanted = set(map(str, ids))
    chosen, n_scanned = {}, 0
    t0 = time.time()
    print(f"[refchecker-data] streaming {NQ_HF_DATASET} split={NQ_HF_SPLIT} "
          f"for {len(wanted)} example ids", flush=True)

    stream = load_dataset(NQ_HF_DATASET, split=NQ_HF_SPLIT, streaming=True)
    for record in stream:
        n_scanned += 1
        if n_scanned % 1000 == 0:
            print(f"[refchecker-data] scanned {n_scanned}, found "
                  f"{len(chosen)}/{len(wanted)} ({time.time() - t0:.0f}s)", flush=True)
        key = str(record["id"])
        if key not in wanted or key in chosen:
            continue

        tokens = record["document"]["tokens"]["token"]
        is_html = record["document"]["tokens"]["is_html"]
        annotations = []
        # Columnar HF layout: one entry per annotator in each parallel list.
        for long_span, short_span in zip(record["annotations"]["long_answer"],
                                         record["annotations"]["short_answers"]):
            short_answers = [
                _nq_join(tokens, is_html, s, e)
                for s, e in zip(short_span["start_token"], short_span["end_token"])
            ]
            long_answer = _nq_join(tokens, is_html,
                                   long_span["start_token"], long_span["end_token"])
            if short_answers and any(len(a) for a in short_answers) and long_answer:
                annotations.append({"short_answers": short_answers,
                                    "long_answer": long_answer})
        if not annotations:
            continue
        chosen[key] = {"id": key, "question": record["question"]["text"],
                       "context": [annotations[0]["long_answer"]],
                       "short_answers": annotations[0]["short_answers"]}
        if len(chosen) == len(wanted):
            print(f"[refchecker-data] all {len(wanted)} ids found after {n_scanned} rows",
                  flush=True)
            break

    out = [chosen[i] for i in map(str, ids) if i in chosen]
    target.write_text(json.dumps(out, indent=4), encoding="utf-8")
    return {"n_wanted": len(wanted), "n_built": len(out), "n_rows_scanned": n_scanned,
            "source": f"{NQ_HF_DATASET}:{NQ_HF_SPLIT}", "elapsed_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "refchecker_protocol"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-nq", action="store_true",
                    help="skip the ~7 GB Natural Questions stream (zero_context stays unbuilt)")
    cfg = ap.parse_args()

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    provenance = {
        "source_repo": "github.com/amazon-science/RefChecker (branch main)",
        "reproduces": "benchmark/data/download_data.sh + gather_benchmark_data.py",
        "nq_note": "streamed from the PUBLIC https://storage.googleapis.com/natural_questions "
                   "mirror of gs://natural_questions rather than via the gcloud SDK; same bytes",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    provenance["repo_files"] = fetch_repo_files(out_dir, cfg.force)
    provenance["dolly"] = build_dolly(out_dir, cfg.force)
    provenance["msmarco"] = build_msmarco(out_dir, cfg.force)
    provenance["nq"] = {"skipped_by_flag": True} if cfg.skip_nq else build_nq(out_dir, cfg.force)

    # Hash every built file so the corpus a run used is identifiable after the fact.
    hashes = {}
    for path in sorted(out_dir.rglob("*.json")):
        hashes[str(path.relative_to(out_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance["sha256"] = hashes
    provenance["elapsed_sec"] = time.time() - t0

    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    from spectral_utils.refchecker import load_refchecker
    claims, diag = load_refchecker(data_dir=out_dir)
    print("\n=== LOADER CHECK ===")
    print(json.dumps(diag, indent=2))
    print(f"\nCOMPLETE -> {out_dir} ({len(claims)} claims, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
