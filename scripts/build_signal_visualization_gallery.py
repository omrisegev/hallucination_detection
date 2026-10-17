"""Eight visual alternatives for the frozen channel-by-error-type analysis.

No scorers are fitted. Family profiles summarize member-channel AUCs, not a
new fusion score. Paired uncertainty is resampled at canonical source group.
"""
from __future__ import annotations
import os
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "2"
import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/advisor_completion_2026-10-11"
OUT = ROOT / "docs/reviews/signal_visualization_options_20261012"
AXES = [
    ("NR", "redundency", "Redundancy", "צעד מיותר"),
    ("NCL", "circular", "Circular logic", "הנמקה מעגלית"),
    ("DR", "deception", "Deception", "טענה מטעה"),
    ("PS", "missing_condition", "Missing prerequisite", "תנאי חסר"),
    ("CI", "confidence", "Confidence", "ביטחון מטעה"),
    ("DC", "domain_inconsistency", "Domain inconsistency", "אי־התאמה לתחום"),
    ("SC", "step_contradiction", "Step contradiction", "סתירה בין צעדים"),
    ("ES", "counterfactual", "Counterfactual", "טענה נגד עובדות"),
]
FAMILY_NAMES = {
    "F1": ("Uncertainty levels", "רמות אי־ודאות וביטחון"),
    "F2": ("Realized surprise", "הפתעת הטוקן שנכתב"),
    "F3": ("Change / innovation", "שינוי וחידוש"),
    "F4": ("Entropy shape", "צורת ה־entropy בתוך צעד"),
    "F5": ("Rhythm / spectrum", "קצב ומבנה ספקטרלי"),
    "F6": ("Variability / drift", "שונות וסטייה מצטברת"),
    "F7": ("Digit alternatives", "חלופות מספריות"),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (SOURCE / "statistics/CHANNEL_BY_TYPE.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    values = np.load(SOURCE / "statistics/CHANNEL_PER_ANSWER.npz")
    population = np.load(SOURCE / "inputs/population.npz")
    assert np.array_equal(values["idx"], population["idx"])
    names = list(values["columns"].astype(str))
    step = names.index("STEP_INDEX")
    tree = ast.parse((ROOT / "scripts/run_advisor_v11_lr.py").read_text())
    roster = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FAMILIES" for t in node.targets))
    membership = {channel: family for family, channels in roster.items() for channel in channels}
    n16 = {r["channel"] for r in rows if r["type"] == "ALL" and r["in_N16"] == "True"}
    assert len(n16) == 16 and len(names) == 57
    assert all(name in membership for name in names if name != "STEP_INDEX")
    lookup = {(r["type"], r["channel"]): r for r in rows}
    groups, inv = np.unique(population["group"], return_inverse=True)
    rng = np.random.default_rng(20261011)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=5000).astype(float)
    data = {"axes": [{"code": c, "key": k, "name": en, "he": he} for c, k, en, he in AXES],
            "familyNames": {k: {"en": v[0], "he": v[1]} for k, v in FAMILY_NAMES.items()},
            "features": [{"id": name, "family": membership[name], "inN16": name in n16,
                          "sign": int(values["lf_sign"][j])} for j, name in enumerate(names) if name != "STEP_INDEX"],
            "metrics": {}, "family": {}, "source": "Frozen v11 CHANNEL_BY_TYPE.csv / CHANNEL_PER_ANSWER.npz",
            "model": "Qwen3-8B · thinkfix + mask1", "bootstrap": {"draws": 5000, "seed": 20261011, "unit": "canonical source group"}}
    records = []
    def summary(vector, mask, reference=None):
        valid = mask & np.isfinite(vector)
        if reference is not None:
            valid &= np.isfinite(reference)
        x = vector - reference if reference is not None else vector
        sums = np.bincount(inv[valid], weights=x[valid], minlength=len(groups))
        counts = np.bincount(inv[valid], minlength=len(groups))
        den = weights @ counts
        boot = np.divide(weights @ sums, den, out=np.full(5000, np.nan), where=den > 0)
        lo, hi = np.nanquantile(boot, [.025, .975])
        return {"value": float(np.mean(x[valid])), "lo": float(lo), "hi": float(hi), "n": int(valid.sum())}
    for metric, key, pointfield, lowfield, highfield in [
        ("raw", "auc", "auc", "lo95", "hi95"),
        ("position", "position_auc", "position_controlled_auc", "position_lo95", "position_hi95")]:
        matrix = values[key]
        data["metrics"][metric] = {}
        data["family"][metric] = {}
        for _, typ, _, _ in AXES:
            mask = population["classification"] == typ
            category = {}
            for j, name in enumerate(names):
                row = lookup[(typ, name)]
                category[name] = {"value": float(row[pointfield]), "lo": float(row[lowfield]),
                                  "hi": float(row[highfield]),
                                  "n": int(row["n_answers" if metric == "raw" else "n_position_answers"]),
                                  "delta": summary(matrix[:, j], mask, matrix[:, step])}
                assert abs(category[name]["value"] - summary(matrix[:, j], mask)["value"]) < 1e-12
            data["metrics"][metric][typ] = category
            for pool in ["all", "n16"]:
                pooldata = data["family"][metric].setdefault(pool, {}).setdefault(typ, {})
                for family, members in roster.items():
                    selected = [name for name in members if name in names and (pool == "all" or name in n16)]
                    if not selected:
                        continue
                    ids = [names.index(name) for name in selected]
                    # All members have identical answer eligibility, but no NaN
                    # repair or zero filling is used for ineligible answers.
                    vector = np.mean(matrix[:, ids], axis=1)
                    result = summary(vector, mask)
                    result["delta"] = summary(vector, mask, matrix[:, step])
                    result["members"] = selected
                    member_points = [category[name]["value"] for name in selected]
                    result["q25"], result["q75"] = map(float, np.quantile(member_points, [.25, .75]))
                    result["member_min"], result["member_max"] = min(member_points), max(member_points)
                    assert abs(result["value"] - np.mean(member_points)) < 1e-12
                    pooldata[family] = result
                    records.append({"metric": metric, "pool": pool, "type": typ, "family": family,
                                    "n_channels": len(selected), "n_answers": result["n"],
                                    "mean_member_auc": result["value"], "ci_low": result["lo"], "ci_high": result["hi"],
                                    "delta_vs_step_index": result["delta"]["value"],
                                    "delta_ci_low": result["delta"]["lo"], "delta_ci_high": result["delta"]["hi"],
                                    "member_q25": result["q25"], "member_q75": result["q75"]})
        print("prepared", metric, flush=True)
    (OUT / "DATA.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (OUT / "FAMILY_SUMMARY.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(records[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(records)
    template = (OUT / "gallery.template.html").read_text()
    extension_path = ROOT / 'results/signal_gallery_extensions_2026-10-12/extension.json'
    extension = json.loads(extension_path.read_text())
    assert [f['id'] for f in extension['features']] == [f['id'] for f in data['features']]
    (OUT / 'EXTENSION_DATA.json').write_bytes(extension_path.read_bytes())
    page = template.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"))
    page = page.replace('__EXTENSION__', json.dumps(extension, ensure_ascii=False).replace('<', '\\u003c'))
    (OUT / "index.html").write_text(page)
    inputs = [SOURCE / "statistics/CHANNEL_BY_TYPE.csv", SOURCE / "statistics/CHANNEL_PER_ANSWER.npz",
              SOURCE / "inputs/population.npz", ROOT / "scripts/run_advisor_v11_lr.py",
              extension_path, ROOT / 'results/signal_gallery_extensions_2026-10-12/MANIFEST.json']
    (OUT / "MANIFEST.json").write_text(json.dumps({"scope": "Presentation alternatives; no new fusion fits; G/H use donor-Q80 feature decisions",
        "inputs": [{"path": str(f.relative_to(ROOT)), "sha256": sha(f)} for f in inputs],
        "builder_sha256": sha(Path(__file__)), "template_sha256": sha(OUT / "gallery.template.html"),
        "html_sha256": sha(OUT / "index.html"), "n_features": 56, "n_types": 8,
        "family_summary": "equal mean of member-channel AUC; not fused-family AUC",
        "family_ci": "5000 whole canonical source-group draws on per-answer mean-member AUC",
        "fixed_orientation": True, "MS_AUC": "undefined; intentionally absent",
        "dependence": "max of two class-conditional mean absolute error phi correlations; same defined peers in both classes; descriptive pairwise dependence only",
        "prm_family_summary": "equal mean of member-feature PRMScores, not a family fusion score; MS valid-step F1 only",
        "base_published_commit": "f54aa8113719d8ab9f96a9eea5dd7dc2ab5ca3b5",
        "published": False}, indent=2) + "\n")
    print("Gallery written", OUT / "index.html", flush=True)

if __name__ == "__main__": main()
