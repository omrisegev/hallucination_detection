"""Build the derived `*_thinkfix_mask1` cells from the `*_thinkfix` teacher-forced cells.

    python scripts/build_mask1_cells.py --results /shared/.../results [--only prmbench,pb8,pb4,ext]

Never overwrites: refuses if an output dir exists. Writes MASK.json per cell with the rule,
row/failure counts, and source/output sha256. Failures (a step that would become empty) are
listed explicitly; those rows are absent from the masked cell.
"""
import argparse, hashlib, json, os, pickle, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_utils.first_token_mask import RULE, mask_cache, mask_first_token, MaskError  # noqa

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def dump_atomic(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def write_mask_json(out, payload):
    payload = {"rule": RULE, "module": "spectral_utils/first_token_mask.py",
               "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "job_id": os.environ.get("SLURM_JOB_ID", ""), **payload}
    (out / "MASK.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "files"}, default=str)[:1500], flush=True)
    for f in payload.get("files", []):
        print("  ", f["name"], "rows", f["n_in"], "->", f["n_out"], "failures", f["n_failures"], flush=True)


def pkl_cell(src: Path, out: Path, names):
    out.mkdir(parents=False, exist_ok=False)
    files = []
    for name in names:
        t0 = time.time()
        cache = pickle.load(open(src / name, "rb"))
        masked, failures = mask_cache(cache)
        dump_atomic(masked, out / name)
        files.append({"name": name, "source_sha256": sha(src / name), "output_sha256": sha(out / name),
                      "n_in": len(cache), "n_out": len(masked), "n_failures": len(failures),
                      "failures": failures, "seconds": round(time.time() - t0, 1)})
        del cache, masked
    src_manifest = src / "manifest.json"
    if src_manifest.exists():
        (out / "SOURCE_manifest.json").write_text(src_manifest.read_text())
    write_mask_json(out, {"source_dir": str(src), "files": files})


def ext_cell(src: Path, out: Path):
    (out / "records").mkdir(parents=True, exist_ok=False)
    recs = sorted((src / "records").glob("*.record.json"))
    failures, n_out = [], 0
    for p in recs:
        data = json.loads(p.read_text(encoding="utf8"))
        try:
            data["payload"]["telemetry"] = mask_first_token(data["payload"]["telemetry"])
        except MaskError as ex:
            failures.append({"uid": data.get("uid"), "error": str(ex)})
            continue
        data["first_token_mask"] = {"rule": "mask1", "source_record_sha256": sha(p)}
        (out / "records" / p.name).write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                                          allow_nan=False), encoding="utf8")
        n_out += 1
    for name in ("ALIGNMENT_GATE.json", "TIMING.json", "TOKENIZATION.json", "THINKFIX_STATS.json"):
        if (src / name).exists():
            (out / f"SOURCE_{name}").write_text((src / name).read_text())
    if (src / "records/RUN.json").exists():
        (out / "records/SOURCE_RUN.json").write_text((src / "records/RUN.json").read_text())
    write_mask_json(out, {"source_dir": str(src), "n_in": len(recs), "n_out": n_out,
                          "n_failures": len(failures), "failures": failures,
                          "note": "step_char_spans and prompt_ids are unchanged (not per-token)",
                          "files": []})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--only", default="prmbench,pb8,pb4,ext")
    a = ap.parse_args()
    R, only = a.results, set(a.only.split(","))
    X = R / "lsml_external_generalization_v1"
    if "prmbench" in only:
        pkl_cell(R / "prmbench_qwen3_8b_telemetry_thinkfix", R / "prmbench_qwen3_8b_telemetry_thinkfix_mask1",
                 ["prmbench_telemetry.pkl"])
    for m in ("8b", "4b"):
        if f"pb{m[0]}" in only:
            pkl_cell(R / f"pb_qwen3_{m}_thinkfix", R / f"pb_qwen3_{m}_thinkfix_mask1",
                     [f"processbench_{s}.pkl" for s in PB_SUBSETS])
    if "ext" in only:
        for c in ("hard2verify", "socratic"):
            ext_cell(X / f"{c}_qwen3_8b_full_thinkfix_153589e", X / f"{c}_qwen3_8b_full_thinkfix_153589e_mask1")
    print("MASK1_DONE", flush=True)


if __name__ == "__main__":
    main()
