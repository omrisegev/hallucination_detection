"""Regenerate only incorrect contrast metadata from frozen OOF rows.

No fitting, inference, or resampling. Existing estimates and intervals are kept
ONLY after checking identical endpoint populations. Outputs go to a new folder.
Run: python -B scripts/repair_runtime_fusion_reports.py
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runtime_protocol", ROOT / "spectral_utils/runtime_fusion_protocol.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def main():
    out = ROOT / "results/runtime_fusion_protocol_v2"
    if out.exists():
        raise FileExistsError("Preserve existing repairs; choose a new version before rerunning")
    manifest = {"schema": "runtime-fusion-report-repair-v2", "fit_or_inference": False,
                "intervals_recomputed": False, "source_sha256": {}, "stages": {}}
    def read(path):
        data = path.read_bytes()
        manifest["source_sha256"][str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(data).hexdigest()
        return list(csv.DictReader(data.decode("utf-8-sig").splitlines()))
    repaired = {}
    for stage in ("S1", "S2", "S3", "S5"):
        source = ROOT / ".worktrees/ssl-pseudolabel-residual-v1/results/ssl_pseudolabel_residual_v1" / stage / "run_20260923"
        oof = read(source / "OOF_ANSWERS.csv")
        rows = read(source / "CONTRASTS.csv")
        assert len(oof) == len({r["uid"] for r in oof}) == 13769
        groups = [r["source_group"] for r in oof]
        def mask(method, endpoint):
            if endpoint == "pb_sla_macro8":
                return [r["cell"].startswith("pb_") and int(r["target"]) >= 0
                        and r[method + "__covered"] == "True" for r in oof]
            if endpoint != "prm_within_auc":
                raise ValueError(endpoint)
            return [not r["cell"].startswith("pb_") and bool(r[method + "__within_auc"])
                    and np.isfinite(float(r[method + "__within_auc"] or "nan")) for r in oof]
        for row in rows:
            a, b = row["contrast_id"].split(" - ")
            old = int(row["paired_N"])
            expected = round(int(row["B"]) * (1 - float(row["invalid_draw_rate"])))
            assert old == expected, "Original paired_N is not the claimed finite-draw count"
            row["legacy_paired_N"] = old
            row.update(P.paired_population(groups, mask(a, row["endpoint"]), mask(b, row["endpoint"]),
                                           draws=int(row["B"]), valid_draws=old))
        repaired[stage] = rows
        manifest["stages"][stage] = {"oof_answers": len(oof), "contrast_rows": len(rows),
                                     "paired_N": sorted({r["paired_N"] for r in rows})}
    # Only write after every source passed the population checks.
    out.mkdir(parents=True)
    for stage, rows in repaired.items():
        with (out / f"{stage}_CONTRASTS.csv").open("w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    audit_path = ROOT / "scratch/daily_experiment_review_20260923/AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    manifest["daily_audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest["window_native"] = audit["window_native_reconstruction"] if "window_native_reconstruction" in audit else {k: v for k, v in audit.items() if "window" in k}
    manifest["limitations"] = ["Window native points reconstructed from exact fallback identity; no matched native CI yet.",
                                "Residual-channel repair is unit-tested, not a newly evaluated localization method."]
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(manifest["stages"], indent=2))


if __name__ == "__main__":
    main()
