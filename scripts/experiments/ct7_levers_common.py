"""Shared plumbing for the three L-SML levers against CT7's failures (2026-09-23).

Items (plan `docs/research_notes/lsml_against_ct7_failures_2026-09-23.md`):
  2. CT7 family-equal              scripts/experiments/ct7_family_equal_v1.py
  3. token-level L-SML on CT7      scripts/experiments/ct7_token_lsml_v1.py
  4. window representation (B3)   scripts/diagnostics/window_pr_measurement_v1.py,
                                  scripts/experiments/window_answer_local_fusion_v1.py

Why this module exists. The frozen CT7 harness (`scripts/experiments/cvf_v2/`) hashes every
`cvf_v2/*.py` and five `spectral_utils` modules into `RUN_FREEZE.json` and asserts equality on
rerun, so nothing may be edited or added there; and `cvf_v2/data.py::Dataset.__init__` loads
token matrices, the Mind-the-Gap evidence, the token-fusion OOF scores and the PRMBench
metadata pickle, none of which a fixed linear combination of the seven CT7 profiles needs.
`light_dataset` builds only what `cvf_v2.scoring.{pb_metrics,prm_metrics,prmscores}` and
`cvf_v2.uncertainty.bootstrap` read, with the same anchor asserts as the frozen harness.

Nothing here reads a label into a fit. Labels are used by the scoring functions only.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CT7_VIEWS = ("H0lim", "ve0", "ve0.75", "ve1", "H0lim_prefix_innovation", "bocpd_residual",
             "chosen_token_z_despiked")
CT7_SHA256 = "9d10d2ff04402f56d6b04c643bfc87a55413786e67d721ad4a28b1578cd3b430"
CT7_MACRO_F1 = 0.41188745848863717
CT7_WITHIN_AUC = 0.7723966352864217
CT7_WITHIN_ELIGIBLE = 6030
STRATA = [("steps_2_to_5", 2, 5), ("steps_6_to_10", 6, 10), ("steps_11_plus", 11, 10 ** 9)]
DEVELOPMENT_NOTE = ("development evidence on the frozen 13,769-answer population (v3 labels, "
                    "source folds v2, frozen CT7 gate); no promotion; not untouched confirmation")


# ----------------------------------------------------------------------------- imports
def ensure_spectral_package() -> None:
    """Make `spectral_utils.<module>` importable without executing the package __init__.

    The package __init__ imports torch through model_utils; the fusion modules do not need
    it. When the real package imports, nothing changes; otherwise a bare package with the
    right __path__ is registered so that submodules and their relative imports resolve.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if "spectral_utils" in sys.modules:
        return
    try:
        importlib.import_module("spectral_utils")
    except Exception:
        pkg = types.ModuleType("spectral_utils")
        pkg.__path__ = [str(ROOT / "spectral_utils")]
        sys.modules["spectral_utils"] = pkg


def load_script(path: Path, name: str):
    """Import a script module by path (the `scripts/diagnostics` files are not a package)."""
    ensure_spectral_package()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cvf():
    """The frozen cvf_v2 package, imported read-only."""
    ensure_spectral_package()
    exp = ROOT / "scripts" / "experiments"
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    return SimpleNamespace(scoring=importlib.import_module("cvf_v2.scoring"),
                          uncertainty=importlib.import_module("cvf_v2.uncertainty"),
                          data=importlib.import_module("cvf_v2.data"),
                          core=importlib.import_module("cvf_v2.core"))


# ----------------------------------------------------------------------------- hashing
def digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value) -> None:
    def jsonable(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonable), encoding="utf8")


def run_freeze(out: Path, sources: list[Path], inputs: list[Path], settings: dict) -> dict:
    """Hash own sources and inputs; refuse to continue into an output directory whose freeze differs."""
    state = {"settings": settings,
             "sources": {str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p): digest(p)
                         for p in sources},
             "inputs": {str(p): {"bytes": Path(p).stat().st_size, "sha256": digest(p)} for p in inputs if Path(p).exists()},
             "development_only": True}
    dest = Path(out) / "RUN_FREEZE.json"
    if dest.exists():
        previous = json.loads(dest.read_text(encoding="utf8"))
        if previous != state:
            raise ValueError(f"frozen inputs/code changed: use a new output directory ({dest})")
    else:
        dump(dest, state)
    return state


# ----------------------------------------------------------------------------- dataset
class LightDataset:
    """Only the fields the frozen scorers read (see module docstring)."""

    def __init__(self, c: dict, *, roster: Path, joined: Path, folds: Path, ct7: Path | None,
                 prm_metadata: Path | None, out: Path, require_anchors: bool = True):
        self.c = c
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        j = json.loads(Path(roster).read_text(encoding="utf8"))
        self.records = j["records"]
        z = np.load(joined)
        self.off = np.asarray(z["offsets"], int); self.target = np.asarray(z["target"], int)
        self.labels = np.asarray(z["labels"], int)
        self.cells = np.array([r["cell"] for r in self.records]); self.groups = np.array([r["group_id"] for r in self.records])
        self.ids = np.array([r["row_id"] for r in self.records]); self.n = len(self.records)
        outer = json.loads(Path(folds).read_text(encoding="utf8"))["outer"]
        self.fold = np.array([outer[g] for g in self.groups], int)
        self.pb = np.char.startswith(self.cells, "pb_"); self.prm = ~self.pb
        assert set(self.fold) == set(range(5)) and len(self.off) == self.n + 1
        assert all(self.off[i + 1] - self.off[i] == r["steps"] for i, r in enumerate(self.records))
        assert all(-1 <= self.target[i] < self.off[i + 1] - self.off[i] for i in np.flatnonzero(self.pb))
        if require_anchors:
            assert (self.n, self.pb.sum(), self.prm.sum(), (self.pb & (self.target >= 0)).sum()) == (13769, 6800, 6969, 4442)
        self.references = {}
        if ct7 is not None:
            ct = np.load(ct7)
            self.gate = np.asarray(ct["gate"], bool); self.references["ct7"] = np.asarray(ct["step_scores"], float)
            assert self.gate.shape == (self.n,) and self.references["ct7"].shape == (int(self.off[-1]),)
        else:
            self.gate = np.ones(self.n, bool)
        self.meta_by_id = None
        if prm_metadata is not None and Path(prm_metadata).exists():
            self.meta_by_id = {m["idx"]: m for m in pickle.load(open(prm_metadata, "rb")).values()}
        # what cvf_v2.scoring.prmscores / uncertainty.bootstrap consult
        self.rosters = []; self.pb_extra_rosters = []
        self.fusion_arms = list(cvf().core.ARMS)

    def peaks(self, s):
        return np.array([int(np.argmax(s[a:b])) for a, b in zip(self.off[:-1], self.off[1:])])

    def strata(self) -> dict[str, np.ndarray]:
        steps = np.diff(self.off)
        m = {"all": np.ones(self.n, bool)}
        m.update({name: (steps >= lo) & (steps <= hi) for name, lo, hi in STRATA})
        return m


def light_dataset(config_path: Path, *, require_anchors: bool = True) -> LightDataset:
    """Build the light dataset from a config with `paths.{roster,joined,folds,ct7,prm_metadata,output}`."""
    config_path = Path(config_path).resolve()
    c = json.loads(config_path.read_text(encoding="utf8"))
    paths = {k: (config_path.parent / v).resolve() if not Path(v).is_absolute() else Path(v)
             for k, v in c["paths"].items()}
    c["paths"] = {k: str(v) for k, v in paths.items()}
    c.setdefault("candidate_id", "none-development-row")
    c.setdefault("seed", 20260923); c.setdefault("bootstrap_draws", 10000)
    c.setdefault("inner_threshold_quantiles", [0.5, 0.99, 50]); c.setdefault("development_only", True)
    d = LightDataset(c, roster=paths["roster"], joined=paths["joined"], folds=paths["folds"],
                     ct7=paths.get("ct7"), prm_metadata=paths.get("prm_metadata"), out=paths["output"],
                     require_anchors=require_anchors)
    if require_anchors:
        assert digest(paths["ct7"]) == CT7_SHA256, "CT7_DEV_SCORES.npz is not the frozen file"
        replay_ct7(d)
    return d


def replay_ct7(d: LightDataset) -> dict:
    """The same anchor asserts as cvf_v2.data.prepare: gated macro-F1 and PRMB within-AUC."""
    S = cvf().scoring
    m = method_from_scores(d, d.references["ct7"])
    pb = S.pb_metrics(d, m)
    assert abs(pb["macro8"]["f1"] - CT7_MACRO_F1) < 1e-12, pb["macro8"]["f1"]
    prm, auc = S.prm_metrics(d, m)
    assert prm["eligible"] == CT7_WITHIN_ELIGIBLE and abs(prm["within_auc"] - CT7_WITHIN_AUC) < 1e-12, prm["within_auc"]
    return {"macro8": pb["macro8"], "within_auc": prm["within_auc"], "eligible": prm["eligible"]}


def load_ct7_profiles(d: LightDataset, path: Path, validation: Path | None = None) -> np.ndarray:
    """The frozen seven CT7 step profiles [steps x 7]; sha checked, CT7 mean replayed."""
    profiles = np.load(path)
    assert profiles.shape == (int(d.off[-1]), 7) and np.isfinite(profiles).all()
    if validation is not None and Path(validation).exists():
        want = json.loads(Path(validation).read_text(encoding="utf8"))["profile_sha256"]
        assert digest(path) == want, "profiles.npy sha differs from PROFILE_VALIDATION.json"
    if "ct7" in d.references:
        mean = profiles.mean(1)
        assert np.max(np.abs(mean - d.references["ct7"])) < 1e-12
        assert np.array_equal(d.peaks(mean), d.peaks(d.references["ct7"]))
    return profiles


# ----------------------------------------------------------------------------- methods
def method_from_scores(d: LightDataset, scores: np.ndarray, *, fallback: np.ndarray | None = None) -> dict:
    """The `cvf_v2.ct7_evaluation.new_method` template for a per-step score vector."""
    s = np.asarray(scores, float)
    assert s.shape == (int(d.off[-1]),)
    m = {"scores": s, "pred": d.peaks(s), "median": np.full(d.n, -999, int),
         "fallback": np.zeros(d.n, bool) if fallback is None else np.asarray(fallback, bool),
         "valid": np.ones(d.n, bool)}
    return m


def answer_z(scores: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """One final within-answer standardization (mean/sd; constant answers -> 0).

    Applied to every arm before the pooled endpoints (PRMScore, pooled step AUROC), so that an
    answer-level scale difference between arms cannot reorder steps across answers. The argmax
    and the within-answer AUROC are invariant to it.
    """
    ensure_spectral_package()
    from spectral_utils.digitfree_broad50 import masked_answer_standardize
    s = np.asarray(scores, float)
    return masked_answer_standardize(s[:, None], np.isfinite(s)[:, None], np.asarray(offsets, int))[:, 0]


def group_draws(groups, mask, rng, draws):
    """Copied from scripts/diagnostics/stage_b_2x2_v1.py::_group_draws (resample source groups)."""
    order = np.argsort(groups[mask], kind="stable")
    flat = np.flatnonzero(mask)[order]
    _, starts, counts = np.unique(groups[flat], return_index=True, return_counts=True)
    for _ in range(draws):
        pick = rng.integers(0, len(starts), size=len(starts))
        take, base = counts[pick], starts[pick]
        ends = np.cumsum(take)
        within = np.arange(int(ends[-1])) - np.repeat(ends - take, take)
        yield flat[np.repeat(base, take) + within]


def interval(diff: np.ndarray) -> dict:
    dd = diff[np.isfinite(diff)]
    lo, hi = np.percentile(dd, 2.5), np.percentile(dd, 97.5)
    return {"point_pp": float(100 * dd.mean()), "ci95_pp": [float(100 * lo), float(100 * hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def stratum_sla_intervals(d: LightDataset, methods: dict, contrasts: list[tuple[str, str, str]],
                          *, seed: int = 20260918, draws: int = 10000) -> dict:
    """Paired source-group intervals of micro SLA (erroneous PB answers) within depth strata.

    The frozen `cvf_v2.uncertainty.bootstrap` gives intervals for the three primary endpoints
    only; the deciding contrast of item 2 lives on the 11+ stratum, so it gets its own paired
    draws here (same unit: source question; micro within the stratum because several cells
    have few 11+ erroneous answers).
    """
    names = list(methods)
    hit = np.column_stack([(methods[n]["pred"] == d.target) & methods[n]["valid"] for n in names]).astype(float)
    out = {}
    for stratum, smask in d.strata().items():
        mask = d.pb & (d.target >= 0) & smask
        if mask.sum() < 20:
            continue
        rng = np.random.default_rng(seed)
        acc = np.empty((draws, len(names)))
        for k, idx in enumerate(group_draws(d.groups, mask, rng, draws)):
            acc[k] = hit[idx].mean(0)
        point = {n: float(hit[mask, j].mean()) for j, n in enumerate(names)}
        rows = []
        for a, b, why in contrasts:
            if a in names and b in names:
                ia, ib = names.index(a), names.index(b)
                rows.append({"a": a, "b": b, "contrast": why, "delta_pp": 100 * (point[a] - point[b]),
                             **interval(acc[:, ia] - acc[:, ib])})
        out[stratum] = {"n_erroneous": int(mask.sum()), "point_sla": point, "contrasts": rows}
    return out


def evaluate_methods(d: LightDataset, methods: dict, *, contrasts_extra: list, strata_contrasts: list,
                     prmscore: bool = True, draws: int | None = None) -> dict:
    """PB metrics, PRMB metrics, the frozen paired bootstrap and the stratum intervals."""
    S, U = cvf().scoring, cvf().uncertainty
    if draws is not None:
        d.c["bootstrap_draws"] = int(draws)
    d.c["planned_contrasts_extra"] = [list(x) for x in contrasts_extra]
    started = time.perf_counter()
    pb = {name: S.pb_metrics(d, m) for name, m in methods.items()}
    prm, within = {}, {}
    for name, m in methods.items():
        prm[name], within[name] = S.prm_metrics(d, m)
    timing = {"metrics_seconds": time.perf_counter() - started}
    started = time.perf_counter()
    unc = U.bootstrap(d, methods, within)
    timing["bootstrap_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    strata = stratum_sla_intervals(d, methods, strata_contrasts, draws=d.c["bootstrap_draws"])
    timing["strata_seconds"] = time.perf_counter() - started
    tables = None
    if prmscore and d.meta_by_id is not None:
        started = time.perf_counter()
        tables = S.prmscores(d, methods)
        timing["prmscore_seconds"] = time.perf_counter() - started
    return {"pb": pb, "prm": prm, "uncertainty": unc, "strata": strata, "prmscore": tables, "timing": timing}


def summary_rows(d: LightDataset, methods: dict, result: dict) -> list[dict]:
    rows = []
    long_cells = [c for c in sorted(set(d.cells[d.pb])) if "olympiadbench" in c or "omnimath" in c]
    for name in methods:
        pb = result["pb"][name]; prm = result["prm"][name]
        strata = result["strata"]
        row = {"method": name, "sla_macro8": pb["macro8"]["sla"], "f1_common_gate": pb["macro8"]["f1"],
               "within_auc": prm["within_auc"], "early": pb["macro8"]["early"], "late": pb["macro8"]["late"],
               "mae": pb["macro8"]["mae"], "tolerance_one": pb["macro8"]["tolerance_one"],
               "sla_long_cells": float(np.mean([pb["cells"][c]["sla"] for c in long_cells])) if long_cells else None,
               "late_long_cells": float(np.mean([pb["cells"][c]["late"] for c in long_cells])) if long_cells else None}
        for stratum in strata:
            row["sla_" + stratum] = strata[stratum]["point_sla"].get(name)
        if result.get("prmscore") and name in result["prmscore"]:
            t = result["prmscore"][name]
            row["prmscore_q80"] = t["quantile_0.8"]["prmscore"]; row["prmscore_inner"] = t["inner_selected"]["prmscore"]
        rows.append(row)
    return rows


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    import csv
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------------- synthetic
def synthetic_dataset(out: Path, *, n_answers: int = 60, seed: int = 0, n_views: int = 7) -> tuple[LightDataset, np.ndarray]:
    """A small population through the real code path (`--dry-run`): PB and PRMB cells, 5 folds,
    source groups, labels, a gate, and profiles with a planted first-error step. No real data.
    """
    rng = np.random.default_rng(seed)
    # the frozen bootstrap packs exactly eight ProcessBench cells; keep the real cell set
    cells = [f"pb_{ds}_{q}" for ds in ("gsm8k", "math", "olympiadbench", "omnimath") for q in ("q4", "q8")]
    cells += ["prmbench_qwen3_8b"] * 3
    records, steps_all, target, labels, groups = [], [], [], [], []
    for i in range(n_answers):
        cell = cells[i % len(cells)]; steps = int(rng.integers(2, 14))
        steps_all.append(steps)
        gid = f"g{i // 2}"; groups.append(gid)
        if cell.startswith("pb_"):
            t = int(rng.integers(0, steps)) if rng.random() < 0.65 else -1
            target.append(t); lab = np.zeros(steps, int)
            if t >= 0:
                lab[t:] = 0; lab[t] = 1
        else:
            t = -1; target.append(t); lab = (rng.random(steps) < 0.2).astype(int)
            if lab.sum() == 0:
                lab[int(rng.integers(0, steps))] = 1
        labels.append(lab)
        records.append({"row": i, "row_id": f"{cell}::{i}", "group_id": gid, "tokens": steps * 20, "steps": steps,
                        "cell": cell, "uid": f"{cell}__{i}", "classification": "missing_condition"})
    off = np.concatenate([[0], np.cumsum(steps_all)])
    labels = np.concatenate(labels); target = np.asarray(target, int)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "JOINED.npz", offsets=off, target=target, labels=labels)
    (out / "JOINED.json").write_text(json.dumps({"records": records, "arms": []}), encoding="utf8")
    uniq = sorted(set(groups))
    (out / "FOLDS_V2.json").write_text(json.dumps({"outer": {g: k % 5 for k, g in enumerate(uniq)}}), encoding="utf8")
    # planted profiles: view j = signal at the true/labelled step + noise, answer-standardized
    profiles = rng.standard_normal((int(off[-1]), n_views))
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        if target[i] >= 0:
            profiles[a + target[i]] += 1.5
        else:
            profiles[a:b] += 1.5 * labels[a:b][:, None] * (rng.random() < 0.9)
    ensure_spectral_package()
    from spectral_utils.digitfree_broad50 import masked_answer_standardize
    profiles = masked_answer_standardize(profiles, np.ones(profiles.shape, bool), off)
    ct7 = profiles.mean(1)
    gate = np.array([rng.random() < 0.7 for _ in range(n_answers)])
    np.savez(out / "CT7_DEV_SCORES.npz", step_scores=ct7, gate=gate)
    np.save(out / "profiles.npy", profiles)
    c = {"paths": {k: str(out / v) for k, v in [("roster", "JOINED.json"), ("joined", "JOINED.npz"),
                                                 ("folds", "FOLDS_V2.json"), ("ct7", "CT7_DEV_SCORES.npz"),
                                                 ("output", "out")]},
         "seed": 20260923, "bootstrap_draws": 20, "inner_threshold_quantiles": [0.5, 0.99, 5],
         "development_only": True, "candidate_id": "synthetic-dry-run"}
    (out / "config.json").write_text(json.dumps(c), encoding="utf8")
    d = light_dataset(out / "config.json", require_anchors=False)
    return d, profiles
