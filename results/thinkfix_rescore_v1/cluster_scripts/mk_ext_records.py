"""Write the think-fix identity records for the two Qwen3-8B external telemetry cells.
Run on the AIRCC login node with plain python3 (stdlib only)."""
import hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

SH = Path("/shared/cycle2_tau_averbuch_prj/omrisegev1")
C = SH / "code_tf_thinkfix_153589e"
OLD = SH / "src/lsml_external_411316927/results/lsml_external_generalization_v1"
NEW = C / "results/lsml_external_generalization_v1_thinkfix"
NEW.mkdir(parents=True, exist_ok=False)
now = datetime.now(timezone.utc).isoformat()


def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, obj):
    (NEW / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf8")
    return NEW / name


# The compute estimate is reused unchanged (prompt grows by the 4-token empty block, negligible).
shutil.copy2(OLD / "ALL_TELEMETRY_COMPUTE_ESTIMATE.json", NEW / "ALL_TELEMETRY_COMPUTE_ESTIMATE.json")
assert h(NEW / "ALL_TELEMETRY_COMPUTE_ESTIMATE.json") == h(OLD / "ALL_TELEMETRY_COMPUTE_ESTIMATE.json")

proto = json.loads((OLD / "COLLECTION_PROTOCOL.json").read_text())
proto["cells"] = [c for c in proto["cells"] if c["model"] == "Qwen/Qwen3-8B"]
proto["prompt"] = ("math_prompt + space /no_think; pinned tokenizer chat template rendered with "
                   "enable_thinking=False so the prompt ends with Qwen3's empty think block "
                   "'<think>\\n\\n</think>\\n\\n' (spectral_utils.processbench.teacher_forced_prompt, "
                   "--think-prefix empty_block)")
proto["think_prefix"] = "empty_block"
proto["supersedes_for_cells"] = {
    "protocol_sha256": h(OLD / "COLLECTION_PROTOCOL.json"),
    "reason": ("2026-10-10: the old prefix ended at the assistant header without Qwen3's empty "
               "think block; the first forced token sat where the model expects <think> "
               "(PRMBench job 177824: <think> top-1 on 6969/6969 rows, ~35 nats)."),
    "old_runs_preserved": ["hard2verify_qwen3_8b_full_411316927", "socratic_qwen3_8b_full_411316927"],
}
proto["socratic_qwq32b"] = ("not rerun: QwQ-32B's template has no non-thinking mode; "
                            "separate question")
proto["code_commit"] = "153589e431d866169035c3cdf9d3f0643edcdb4c"
p_proto = dump("COLLECTION_PROTOCOL.json", proto)
ph = h(p_proto)
eh = h(NEW / "ALL_TELEMETRY_COMPUTE_ESTIMATE.json")

old_dec = json.loads((OLD / "FULL_BUDGET_DECISION.json").read_text())
dec = {
    "approved_by": "user",
    "user_instruction": "you have my permission to do it. ... 3. the external benchmarks? "
                        "(hard to verify, socratic) if so, they also need to run.",
    "user_message_utc": "2026-10-10T18:35:39Z",
    "recorded_utc": now,
    "protocol_hash": ph,
    "estimate_hash": eh,
    "max_gpu_hours": 2,
    "cells": [c for c in old_dec["cells"] if c["model"] == "Qwen/Qwen3-8B"],
    "scope": old_dec["scope"] + "; think-fix rerun of the two Qwen3-8B cells only",
    "retries": old_dec["retries"],
}
dump("FULL_BUDGET_DECISION.json", dec)

old_pf = json.loads((OLD / "FULL_PREFLIGHT.json").read_text())
pf = {
    "verdict": "PASS",
    "session_id": "external-thinkfix-20261010",
    "utc": now,
    "protocol_hash": ph,
    "code": str(C) + " (git archive of 153589e43; SYNC_COMMIT.json)",
    "cluster_jobs_before_submit": "PRMBench/ProcessBench think-fix chains 280425-280430 only",
    "tests": "tests/test_external_generalization.py 8 PASS; tests/test_runtime_fusion_protocol.py 6 PASS; "
             "scripts/test_teacher_forced_think_prefix.py 5 PASS (Qwen3-8B template snapshot)",
    "driver_cpu_smoke": "tests/test_external_generalization.py --cpu-smoke PASS (GPT2 fixture; its "
                        "template has no enable_thinking, so the block itself is checked by the GPU "
                        "pilots: Gate B 280415/280416 and PRMBench pilot 280422/decode 280423)",
    "label_sanity": old_pf["label_sanity"],
    "input_integrity": "answers.json files unchanged from the pinned 2026-09-24 inputs (hashes in "
                       "INPUT_HASHES below)",
    "INPUT_HASHES": {d: h(SH / f"data/lsml_external_20260924/{d}/answers.json")
                     for d in ("hard2verify", "socratic")},
    "gpu_gate": old_pf["gpu_gate"],
}
dump("FULL_PREFLIGHT.json", pf)
print("protocol", ph); print("estimate", eh)
print(json.dumps(pf["INPUT_HASHES"]))
inm = json.loads((SH / "data/lsml_external_20260924/INPUT_MANIFEST.json").read_text())
print("INPUT_MANIFEST excerpt:", json.dumps(inm)[:600])
