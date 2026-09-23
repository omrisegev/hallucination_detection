"""Read-only backup verification for explicit historical cache candidates."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    ("cache/repgrid/inside_coqa_llama7b", "repgrid/inside_coqa_llama7b"),
    ("dataset_cache/repgrid/losnet_hotpotqa_mistral7b", "repgrid/losnet_hotpotqa_mistral7b"),
    ("dataset_cache/repgrid/noise_gsm8k_mistral7b", "repgrid/noise_gsm8k_mistral7b"),
    ("dataset_cache/repgrid/noise_gsm8k_phi3mini", "repgrid/noise_gsm8k_phi3mini"),
    ("dataset_cache/repgrid/lapeigvals_gsm8k_phi35", "repgrid/lapeigvals_gsm8k_phi35"),
    ("dataset_cache/repgrid/lapeigvals_gsm8k_nemo", "repgrid/lapeigvals_gsm8k_nemo"),
    ("dataset_cache/repgrid/lapeigvals_gsm8k_mistral24b", "repgrid/lapeigvals_gsm8k_mistral24b"),
    ("dataset_cache/repgrid/lapeigvals_gsm8k_llama3b", "repgrid/lapeigvals_gsm8k_llama3b"),
    ("dataset_cache/repgrid/evdrop_math_qwen3_8b", "evdrop_math_qwen3_8b"),
    ("dataset_cache/repgrid/evdrop_math_qwen3_4b", "evdrop_math_qwen3_4b"),
    ("dataset_cache/repgrid/evdrop_gsm8k_qwen3_8b", "evdrop_gsm8k_qwen3_8b"),
    ("dataset_cache/repgrid/evdrop_gsm8k_qwen3_4b", "evdrop_gsm8k_qwen3_4b"),
    ("dataset_cache/four_localization/hle_full", "hle_full"),
    ("cache/hle_full", "hle_full"),
    ("dataset_cache/ragtruth_ec_full/test", "ragtruth_ec_qwen25_15b_test"),
    ("dataset_cache/ragtruth_ec_full/dev", "ragtruth_ec_qwen25_15b_dev"),
]


def check(pair):
    local, remote = pair
    base = "gdrive:hallucination_detection/cluster_results/" + remote
    cmd = subprocess.run([shutil.which("rclone"), "lsjson", base, "--files-only", "--hash"],
                         capture_output=True, text=True, check=True)
    metadata = json.loads(cmd.stdout)
    found = {x["Name"]: x for x in metadata}
    verified = []
    for p in (ROOT / local).glob("*.pkl"):
        d = found.get(p.name)
        if not d or d["Size"] != p.stat().st_size or not d.get("Hashes", {}).get("md5"):
            continue
        with p.open("rb") as f:
            md5 = hashlib.file_digest(f, "md5").hexdigest()
        if md5 == d["Hashes"]["md5"]:
            verified.append({"local": str(p.resolve()), "remote": base + "/" + p.name,
                             "bytes": d["Size"], "md5": md5, "remote_mtime": d["ModTime"]})
    return {"local_dir": local, "remote": base, "verified": verified,
            "archive_metadata": metadata}


if __name__ == "__main__":
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(check, CANDIDATES):
            records.append(result)
            print(result["local_dir"], sum(x["bytes"] for x in result["verified"]), flush=True)
    out = ROOT / "results/lsml_external_generalization_v1/CLEANUP_VERIFICATION.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2)+"\n", encoding="utf8", newline="\n")
    print("Verified bytes", sum(v["bytes"] for r in records for v in r["verified"]))
