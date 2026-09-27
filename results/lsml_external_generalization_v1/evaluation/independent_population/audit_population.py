"""Independent population/identity audit. Never reads quality scores or emits text."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
INPUT = ROOT / "scratch/external_generalization_private/inputs"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_hash(text, normalized=True):
    assert isinstance(text, str) and text.strip()
    return hashlib.sha256((" ".join(text.split()) if normalized else text).encode("utf8")).hexdigest()


class Components:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def merge(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def group_map(ds, rows):
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[ds.find(i)].append(row["uid"])
    groups = {}
    for values in members.values():
        values.sort()
        key = "external_source:" + hashlib.sha256("\n".join(values).encode()).hexdigest()
        groups[key] = values
    return dict(sorted(groups.items()))


def main():
    paths = [INPUT / n / "answers.json" for n in ("hard2verify", "socratic")]
    paths += [INPUT / "evaluator_only" / (n + ".json") for n in ("hard2verify", "socratic")]
    audit_dir = ROOT / "results/localization_source_group_audit_v1"
    prm_path, pb_path = audit_dir / "QUESTION_METADATA.json", audit_dir / "PB_QUESTION_METADATA.json"
    joined_path = ROOT / "results/localization_full_benchmark_v3/evaluation/JOINED.json"
    paths += [prm_path, pb_path, joined_path]
    prm, pb, joined = read(prm_path)["rows"], read(pb_path)["rows"], read(joined_path)["records"]
    meta = {"prm": {x["row_id"]: x for x in prm}, "pb": {x["row_id"]: x for x in pb}}
    assert len(meta["prm"]) == len(prm) and len(meta["pb"]) == len(pb)
    dev_hashes, dev_exact = defaultdict(set), defaultdict(set)
    dev_kinds = defaultdict(set)
    dev_uids = defaultdict(set)
    represented = {"prm": set(), "pb": set()}
    for row in joined:
        kind = "pb" if row["cell"].startswith("pb_") else "prm"
        m = meta[kind][row["row_id"]]
        represented[kind].add(row["row_id"])
        dev_hashes[m["question_whitespace_sha256"]].add(row["group_id"])
        dev_exact[m["question_sha256"]].add(row["group_id"])
        dev_kinds[m["question_whitespace_sha256"]].add(kind)
        dev_uids[m["question_whitespace_sha256"]].add(row["uid"])
    assert all(represented[k] == set(meta[k]) for k in meta)

    rows, counts = [], {}
    for name, expected in (("hard2verify", 200), ("socratic", 2995)):
        answers, labels = read(INPUT/name/"answers.json"), read(INPUT/"evaluator_only"/(name+".json"))
        assert len(answers) == len(labels) == expected
        assert len({a["uid"] for a in answers}) == len(answers)
        assert len({a["uid"] for a in labels}) == len(labels)
        stats = Counter(answers=len(answers), annotations=len(labels))
        categories, source_prefixes = Counter(), Counter()
        original_hashes, evaluated_hashes, source_ids = set(), set(), set()
        for a, g in zip(answers, labels):
            assert a["uid"] == g["uid"] and a["benchmark"] == name
            assert len(a["steps"]) == len(g["correct"]) == len(g["include"])
            assert all(type(v) is bool for v in g["correct"] + g["include"])
            oh, eh = question_hash(a["original_question"]), question_hash(a["question"])
            orig_hit, eval_hit = dev_hashes.get(oh, set()), dev_hashes.get(eh, set())
            kinds = dev_kinds.get(oh, set()) | dev_kinds.get(eh, set())
            stats.update(steps=len(a["steps"]), included_steps=sum(g["include"]),
                         empty_steps=sum(not x for x in a["steps"]),
                         whitespace_only_steps=sum(not x.strip() for x in a["steps"]),
                         original_evaluated_differ=oh != eh,
                         missing_source_id=not a["source_id"],
                         direct_development_overlap=bool(orig_hit or eval_hit),
                         original_development_overlap=bool(orig_hit),
                         evaluated_development_overlap=bool(eval_hit),
                         pb_direct_overlap="pb" in kinds, prm_direct_overlap="prm" in kinds,
                         original_exact_overlap=question_hash(a["original_question"], False) in dev_exact,
                         evaluated_exact_overlap=question_hash(a["question"], False) in dev_exact,
                         rows_with_out_of_range_error_indices=bool(g["out_of_range_error_indices"]),
                         out_of_range_error_indices=len(g["out_of_range_error_indices"]))
            categories[g["category"]] += 1
            original_hashes.add(oh); evaluated_hashes.add(eh)
            if a["source_id"]:
                source_ids.add(a["source_id"])
                source_prefixes[a["source_id"].split("#")[0]] += 1
            rows.append({"uid": a["uid"], "benchmark": name, "source_id": a["source_id"],
                         "original_question_whitespace_sha256": oh,
                         "evaluated_question_whitespace_sha256": eh,
                         "steps": len(a["steps"]), "included_steps": sum(g["include"]),
                         "category": g["category"],
                         "direct_development_groups": sorted(orig_hit | eval_hit),
                         "original_development_groups": sorted(orig_hit),
                         "evaluated_development_groups": sorted(eval_hit),
                         "direct_development_kinds": sorted(kinds)})
        counts[name] = {**dict(stats), "unique_original_hashes": len(original_hashes),
                        "unique_evaluated_hashes": len(evaluated_hashes),
                        "nonempty_unique_source_ids": len(source_ids),
                        "categories": dict(categories), "source_prefix_rows": dict(source_prefixes)}

    ds, seen = Components(len(rows)), {}
    for i, row in enumerate(rows):
        keys = [("text", row["original_question_whitespace_sha256"]),
                ("text", row["evaluated_question_whitespace_sha256"])]
        if row["source_id"]:
            keys.append(("source:" + row["benchmark"], row["source_id"]))
        for key in keys:
            if key in seen:
                ds.merge(i, seen[key])
            seen[key] = i
    external_only = group_map(ds, rows)
    # Existing canonical development source identity can connect text variants.
    seen_development = {}
    for i, row in enumerate(rows):
        for key in row["direct_development_groups"]:
            if key in seen_development:
                ds.merge(i, seen_development[key])
            seen_development[key] = i
    groups = group_map(ds, rows)
    by_uid = {row["uid"]: row for row in rows}
    lists = {}; group_records = []
    for key, uids in groups.items():
        members = [by_uid[uid] for uid in uids]
        dev = sorted({g for row in members for g in row["direct_development_groups"]})
        datasets = sorted({row["benchmark"] for row in members})
        for row in members:
            row["bootstrap_group"] = key
            row["component_development_overlap"] = bool(dev)
            row["cross_external_component"] = len(datasets) > 1
        group_records.append({"group": key, "uids": uids, "datasets": datasets, "development_groups": dev})
    for name in counts:
        own = [row for row in rows if row["benchmark"] == name]
        lists[name] = {
            "all_uids": sorted(x["uid"] for x in own),
            "direct_development_overlap_uids": sorted(x["uid"] for x in own if x["direct_development_groups"]),
            "component_development_overlap_uids": sorted(x["uid"] for x in own if x["component_development_overlap"]),
            "observed_development_disjoint_uids": sorted(x["uid"] for x in own if not x["component_development_overlap"]),
            "cross_external_uids": sorted(x["uid"] for x in own if x["cross_external_component"])}
        counts[name].update(
            source_components=len({x["bootstrap_group"] for x in own}),
            observed_development_disjoint_rows=len(lists[name]["observed_development_disjoint_uids"]),
            component_development_overlap_rows=len(lists[name]["component_development_overlap_uids"]),
            observed_development_disjoint_groups=len({x["bootstrap_group"] for x in own if not x["component_development_overlap"]}),
            cross_external_rows=len(lists[name]["cross_external_uids"]),
            component_size_histogram=dict(sorted(Counter(len(v) for v in groups.values() if any(by_uid[u]["benchmark"] == name for u in v)).items())))

    provenance = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p),
                   "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()} for p in paths]
    summary = {
        "schema": "independent-external-population-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_reports_read": False, "quality_scores_computed": False,
        "checked_external_answers": len(rows), "registered_external_answers": 3195,
        "development": {"joined_rows_checked": len(joined), "prm_metadata_rows_checked": len(prm),
                        "pb_metadata_rows_checked": len(pb), "joined_source_groups": len({x["group_id"] for x in joined}),
                        "unique_normalized_question_hashes": len(dev_hashes), "all_metadata_rows_represented_in_joined": True,
                        "prm_source_seed_count": len({x["source_seed"] for x in prm})},
        "corpora": counts, "external_only_components": len(external_only),
        "components_with_development_identity_links": len(groups),
        "cross_external_components": sum(len(x["datasets"]) > 1 for x in group_records),
        "normalization": "SHA256 UTF8 of ' '.join(text.split()); preserve case/punctuation/math/Unicode",
        "group_rule": "union original/evaluated normalized hashes, nonempty dataset-qualified source_id, and canonical development source groups reached by text",
        "disjoint_definition": "no observed exact-normalized hash match anywhere in the connected source component against supplied development metadata",
        "limitations": [
            "PRMB metadata hashes cached evaluated question only; it does not enumerate every original_question and modified_question variant. Synthetic-correct rows represent only some originals.",
            "PRMB source_seed identity is prm_train/test_pN_number; Socratic source IDs such as OlympiadBench#number have no verified crosswalk to PRMB or PB IDs.",
            "All Hard2Verify source_id fields may be empty; source grouping then relies on exact-normalized question identity, not inferred parsing of answer UID.",
            "No semantic/paraphrase, mathematical equivalence, case-folded, or Unicode-normalized matching was possible against hash-only development metadata.",
            "Observed disjointness is not proof of untouched underlying problems, no pretraining contamination, or independence from other project exposure outside supplied development corpus.",
            "Annotations were read only to validate population/step alignment and inclusion; correctness was not used to select groups or disjoint UID lists."],
        "input_provenance": provenance,
        "audit_script_sha256": sha(Path(__file__)),
        "exact_command": 'python results/lsml_external_generalization_v1/evaluation/independent_population/audit_population.py'}
    for filename, content in (("AUDIT.json", summary), ("UID_LISTS.json", lists),
                              ("GROUPS.json", group_records), ("ROWS.json", rows)):
        target = OUT/filename
        assert not target.exists(), "Independent audit artifact already exists: " + str(target)
        target.write_text(json.dumps(content, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf8")
    print(json.dumps({k: summary[k] for k in ("checked_external_answers", "development", "corpora", "external_only_components", "components_with_development_identity_links", "cross_external_components")}, indent=2))


if __name__ == "__main__":
    main()
