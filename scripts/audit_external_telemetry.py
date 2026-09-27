"""Read-only raw-telemetry integrity audit; no feature fitting or quality labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit_row(row):
    ids = row["gen_token_ids"]
    n = len(ids)
    for key in ["token_offsets", "actual_token_logprobs", "token_spilled_energies",
                "token_entropies", "token_entropy_full", "token_logsumexp"]:
        if len(row[key]) != n:
            raise ValueError("unaligned " + key)
    lp, top_ids = row["top_k_logprobs"]["logprobs"], row["top_k_logprobs"]["ids"]
    if len(lp) != n or len(top_ids) != n:
        raise ValueError("unaligned top50")
    outside = 0
    for i, token in enumerate(ids):
        p, order = lp[i], top_ids[i]
        if len(p) != 50 or len(order) != 50 or len(set(order)) != 50:
            raise ValueError("invalid top50 length/identity")
        if not all(math.isfinite(x) and x <= 1e-6 for x in p):
            raise ValueError("invalid log probabilities")
        if any(a < b for a, b in zip(p, p[1:])) or sum(math.exp(v) for v in p) > 1.00001:
            raise ValueError("unsorted/unnormalized raw top50")
        actual = row["actual_token_logprobs"][i]
        if not math.isfinite(actual) or actual > 1e-6 or abs(actual + row["token_spilled_energies"][i]) > 1e-6:
            raise ValueError("invalid actual-token logprob")
        if token in order:
            if abs(actual-p[order.index(token)]) > 1e-4:
                raise ValueError("chosen/top50 disagreement")
        else:
            outside += 1
            if actual > p[-1]+1e-4:
                raise ValueError("outside-top50 probability exceeds boundary")
        for key in ["token_entropies", "token_entropy_full", "token_logsumexp"]:
            if not math.isfinite(row[key][i]):
                raise ValueError("nonfinite " + key)
        masses = [math.exp(v) for v in p[:15]]
        total = sum(masses)+1e-12
        expected = -sum(v/total*math.log(v/total+1e-12) for v in masses)
        if abs(expected-row["token_entropies"][i]) > 1e-4:
            raise ValueError("historical top15 entropy replay mismatch")
    spans = row["step_token_spans"]
    if len(spans) != len(row["step_char_spans"]) or any(not 0 <= a <= b <= n for a,b in spans):
        raise ValueError("invalid step spans")
    if any(a < previous[1] for previous,(a,b) in zip(spans,spans[1:])):
        raise ValueError("overlapping step spans")
    return {"tokens": n, "steps": len(spans), "outside_top50": outside,
            "empty_steps": sum(a==b for a,b in spans)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    timing = json.loads((args.run/"TIMING.json").read_text())
    if not timing["complete"] or timing["quality_evaluated"]:
        raise ValueError("not a completed timing-only run")
    records, hashes = {}, {}
    for path in sorted((args.run/"records").glob("*.record.json")):
        value = json.loads(path.read_text())
        if value["uid"] in records:
            raise ValueError("duplicate answer")
        records[value["uid"]] = audit_row(value["payload"]["telemetry"])
        with path.open("rb") as f:
            hashes[path.name] = hashlib.file_digest(f, "sha256").hexdigest()
    if set(records) != set(timing["identity"]["selected_uids"]):
        raise ValueError("incomplete or extra records")
    print(json.dumps({"status":"PASS", "quality_evaluated":False, "answers":len(records),
                      "totals": {key:sum(v[key] for v in records.values()) for key in next(iter(records.values()))},
                      "record_sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
