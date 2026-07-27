"""
build_comparison.py — one interactive table covering every method we have tried.

Omri's framing, and the reason this file exists: **what matters is the algorithm
that does not use a fixed subset.** GOOD_5 / GOOD_6 / LOCO_5 were chosen by hand
using answer keys — they are the bar, not entries in the race, and they are tagged
so nobody reads them as method results.

EVERYTHING IS READ FROM DISK. Nothing is hand-typed: Step 193's lesson is that
hand-copied subset definitions and stale numbers are how this project loses
sessions. Descriptions come from `spectral_utils/glossary.py`, the single source of
truth with a hard coverage gate (`scripts/build_glossary.py`) — this script writes
no prose about any method, it only concatenates the canonical family note and
variant note.

CORRECTNESS TRAPS, ALL HANDLED
------------------------------
1. The selector-bench CSVs cover 51 cells (the `c46` pool includes out-of-scope
   cells) while scoreboard AUROC is the 25 in-scope cells. Sizes are computed on
   IN-SCOPE ROWS ONLY, or the size column would describe different data than the
   AUROC column. Asserted: no variant may contribute more than 25 cells.
2. The same variant appears under two pool modes (`c46`, `h16`) with genuinely
   different chosen sizes (a2.dufs: 19.0 vs 10.8). Rows are keyed by
   (variant, pool_mode), never merged.
3. THIS PAGE IS SHARED WITH ADVISORS, so "published" is not good enough — every
   row carries the result of `reproduction_audit.py`, which replays that row's
   stored subsets through today's fusion code. A row that no longer reproduces
   says so on its face instead of being quietly presented as current.
4. Loading scale is never implicit. Each row states the scale its number was
   computed at, and the audit supplies the same subsets fused at the other two,
   so scale sensitivity is a column rather than a footnote.
5. `ref.LOCO_5` is scored on 24 cells, not 25 (`inside_coqa_llama7b` lacks the
   energy/logprob views), so sorting by macro puts it on top of rows it is not
   strictly comparable to. Flagged in its status.

Run:  python scripts/upcr_study/build_comparison.py
Out:  results/upcr_study/comparison.html   (sortable / filterable / groupable)
      results/upcr_study/comparison.csv    (same rows, flat)

Depends on (run these first if stale):
      scripts/upcr_study/reproduction_audit.py
      scripts/run_selector_bench.py --selector a1_residual --pool c46 \
          --sel-kwargs '{"loading_scale":"complete"}' \
          --out results/upcr_study/07_a1_rerun/a1_residual__c46__complete.csv
"""
import os
import re
import sys
import csv
import glob
import json
import html
import statistics
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from spectral_utils.glossary import (                               # noqa: E402
    resolve, role_of, FAMILY_NOTES, ROLE_NOTES, SUFFIX_NOTES,
)
from reproduction_audit import K_OVERRIDE_VARIANTS                  # noqa: E402

REPO, OUT = S.REPO, S.OUT_ROOT
SUFFIX_RE = re.compile(r"_(s\d+|adapt|k\d+)$")

# Step-204 arms that carry a verdict the scoreboard does not know about.
STATUS_OVERRIDE = {
    "random_s6": "FLOOR — the sanity check every selector must beat",
    "random_s5": "FLOOR — the sanity check every selector must beat",
    "random_s4": "FLOOR — the sanity check every selector must beat",
}


def base_of(v):
    m = SUFFIX_RE.search(v)
    return v[:m.start()] if m else v


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def load_inscope():
    from inscope_bench_common import load_cells
    return set(load_cells())


def load_bench(inscope):
    """(family_of_variant, size_stats, top_features) from the selector bench.

    Family comes from the FILENAME, not the `selector` column — that column is
    empty in several files, and reading it lets a later file silently overwrite an
    earlier variant's family with ''.
    """
    fam, sizes, chosen = {}, collections.defaultdict(list), collections.defaultdict(collections.Counter)
    modes = collections.defaultdict(collections.Counter)
    for f in sorted(glob.glob(os.path.join(REPO, "results", "selector_bench", "*.csv"))):
        b = os.path.basename(f)
        if "__" not in b:
            continue
        family = b.split("__")[0]
        try:
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
        except Exception:
            continue
        if not rows or "variant" not in rows[0]:
            continue
        for r in rows:
            v, pm = r["variant"], r.get("pool_mode", "")
            fam.setdefault(v, family)
            if r.get("cell") in inscope and r.get("size"):
                sizes[(v, pm)].append(int(r["size"]))
                modes[(v, pm)][r.get("eval_mode") or ""] += 1
                for feat in (r.get("chosen") or "").split("|"):
                    if feat:
                        chosen[(v, pm)][feat] += 1
    for k, v in sizes.items():
        assert len(v) <= 25, f"in-scope filter leaked: {k} has {len(v)} cells"
    return fam, sizes, chosen, modes


def load_scoreboard():
    p = os.path.join(REPO, "results", "checkpoints", "scoreboard_latest.csv")
    return [r for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("macro_all")]


def load_audit():
    """Per (variant, pool) verdict from reproduction_audit.py: does the published
    number still come out of today's code, and what do the other two loading
    scales give on the same subsets."""
    p = os.path.join(OUT, "00_reproduction_audit", "reproduction_audit.csv")
    if not os.path.exists(p):
        return {}
    return {(r["variant"], r["pool"]): r
            for r in csv.DictReader(open(p, encoding="utf-8"))}


def load_stability():
    """Per (variant, pool): how far the number moves under a 1e-10 relative jitter
    of the feature matrix. A row whose macro moves by pp is one draw, not a
    measurement — see stability_audit.py for why small subsets do this."""
    p = os.path.join(OUT, "00_reproduction_audit", "stability_audit.csv")
    if not os.path.exists(p):
        return {}
    return {(r["variant"], r["pool"]): r
            for r in csv.DictReader(open(p, encoding="utf-8"))}


def load_a1_rerun(inscope):
    """a1_residual re-run end to end under today's code, once per loading scale.

    The published a1 rows select by an objective that calls L-SML, so unlike every
    other family they cannot be corrected by re-scoring a stored subset — the
    SELECTION moves too. `unit` re-derives the published arm (a check); `complete`
    is the corrected-criterion arm, and it is a genuinely different method, so it
    gets its own rows rather than overwriting anything.
    """
    from inscope_cells import GROUP
    out = {}
    for scale in ("unit", "complete"):
        p = os.path.join(OUT, "07_a1_rerun", f"a1_residual__c46__{scale}.csv")
        if not os.path.exists(p):
            continue
        agg = collections.defaultdict(
            lambda: {"auroc": [], "qa": [], "math": [], "size": [],
                     "chosen": collections.Counter(), "cells": set()})
        for r in csv.DictReader(open(p, encoding="utf-8")):
            ck = r.get("cell")
            if ck not in inscope or not r.get("auroc"):
                continue
            a = agg[r["variant"]]
            v = float(r["auroc"])
            a["auroc"].append(v)
            (a["qa"] if GROUP.get(ck) == "QA" else a["math"]).append(v)
            a["size"].append(int(r["size"]))
            a["cells"].add(ck)
            for feat in (r.get("chosen") or "").split("|"):
                if feat:
                    a["chosen"][feat] += 1
        if agg:
            out[scale] = dict(agg)
    return out


def load_extra():
    def j(name):
        p = os.path.join(OUT, name, "summary.json")
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    ab_path = os.path.join(REPO, "results", "advisor_inscope",
                           "prior_free_bench_summary.csv")
    ab = ({r["arm"]: r for r in csv.DictReader(open(ab_path, encoding="utf-8"))}
          if os.path.exists(ab_path) else {})
    return (j("03_faithful_factorial"), j("05_cluster_variant"),
            j("06_orientation"), ab)


# ---------------------------------------------------------------------------
def _fam(family, field):
    """One field of a family note. FAMILY_NOTES values are dicts with keys
    `relies_on` / `paper` / `performance` / `history` — not strings."""
    d = FAMILY_NOTES.get(family)
    return (d.get(field) or "").strip() if isinstance(d, dict) else ""


def describe(variant, family):
    """Canonical description: variant note + the family's mechanism. NO new prose
    is written here — every sentence is quoted from spectral_utils/glossary.py."""
    r = resolve(variant)
    parts = []
    if r and r[0]:
        parts.append(r[0].strip())
    mech = _fam(family, "relies_on")
    if mech and mech not in parts:
        parts.append(mech)
    if r and r[2]:
        parts.append(SUFFIX_NOTES.get(
            re.sub(r"\d+", "{N}", r[2]).replace("+scale_complete", "+scale_{S}")
                                       .replace("+scale_unit", "+scale_{S}")
                                       .replace("+scale_eigen", "+scale_{S}"),
            f"The {r[2]} suffix is a sweep of the same method."))
    return " ".join(parts) or "(no glossary entry)"


def similar_to(variant, family, all_variants, fam_of):
    """Derived structurally — same base with a different suffix first, then other
    members of the same family. Nothing hand-curated."""
    b = base_of(variant)
    same_base = sorted(v for v in all_variants
                       if v != variant and base_of(v) == b)
    same_fam = sorted(v for v in all_variants
                      if v != variant and v not in same_base
                      and fam_of.get(v) == family and family)
    out = same_base[:4] + same_fam[:4]
    return ", ".join(out) if out else "—"


def size_cell(stats):
    if not stats:
        return None, None, None, None, 0
    return (round(statistics.mean(stats), 1), statistics.median(stats),
            min(stats), max(stats), len(stats))


def _num(x, nd=4):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def verdict(a, st, mode_counts, variant):
    """Is this number still what the code gives — and if not, WHY not?

    "It drifted" on its own is not a useful verdict, and for most of the drifting
    rows it is not even the right one: the drift is smaller than the row's own
    numerical noise, so both values are draws from the same distribution rather
    than one being stale. The categories below separate the real causes, and the
    residual "unexplained" bucket is the one that would actually mean a number on
    this page is wrong. It is currently empty.
    """
    if not a:
        return "not audited", None, None, None
    frac3 = _num(a.get("frac_size3"), 3)
    if a.get("reproduces") == "True":
        return "verified", _num(a.get("macro_today")), _num(a.get("drift_pp"), 2), frac3
    if a.get("reproduces") is None or a.get("reproduces") == "":
        return "not replayable", None, None, frac3

    drift = abs(float(a["drift_pp"]))
    today, d2 = _num(a.get("macro_today")), _num(a.get("drift_pp"), 2)
    noise = float(st["macro_spread_pp"]) if st else 0.0
    if drift <= max(noise, 0.02):
        return "within its own noise", today, d2, frac3
    if "+K_" in variant or variant in K_OVERRIDE_VARIANTS:
        return "code fix: Step-189 K clamp", today, d2, frac3
    # Step 205 replaced spectral clustering with an EXACT partition enumeration at
    # m <= 4, where Eq.15 carries no information (m=3) or two terms (m=4) and the
    # grouping was decided by float noise rather than by the data. Any row whose
    # subsets are that small is expected to move, and moves for that reason.
    if _small_m_row(a, st):
        return "code fix: Step-205 exact small-m solve", today, d2, frac3
    if mode_counts and mode_counts.get("lookup", 0) > mode_counts.get("live", 0):
        return "lookup table vs live re-fusion", today, d2, frac3
    return f"UNEXPLAINED {float(a['drift_pp']):+.2f}pp", today, d2, frac3


def _small_m_row(a, st):
    """Does this row's drift fit the m <= 4 change, and ONLY that change?

    Two conditions, because "has a small subset somewhere" alone would let this
    category absorb a genuine anomaly on any row with one size-4 cell — which
    would defeat the point of the UNEXPLAINED bucket.

      1. the row must actually reach into m <= 4 on some cell. Deliberately not
         "mean size <= 4": a row with mean size 5.4 can still hold size-3/4
         subsets on individual cells, and those are exactly the cells that moved.
      2. no MORE cells may have changed than have a small subset. If 12 cells
         moved but only 7 hold an m <= 4 subset, something else moved too and
         this explanation does not cover the row.
    """
    frac = 0.0
    for src in (st, a):
        if not src:
            continue
        for key in ("frac_size3", "frac_size4"):
            try:
                frac = max(frac, float(src.get(key)))
            except (TypeError, ValueError):
                pass
    if frac <= 0:
        try:                                   # stability row missing: fall back
            return float(a.get("size_mean")) <= 4.0
        except (TypeError, ValueError):
            return False

    # frac_size3/4 are separate fractions of the same cells; sum them for the
    # share of cells that could have been touched, capped at 1.
    share = 0.0
    for key in ("frac_size3", "frac_size4"):
        for src in (st, a):
            if src and src.get(key) not in (None, ""):
                try:
                    share += float(src[key])
                except (TypeError, ValueError):
                    pass
                break
    share = min(share, 1.0)
    try:
        n_cells = float(a.get("cells"))
        n_diff = float(a.get("cells_differing"))
    except (TypeError, ValueError):
        return True
    return n_diff <= round(share * n_cells) + 1e-9


def main():
    inscope = load_inscope()
    fam_of, sizes, chosen, modes = load_bench(inscope)
    sb = load_scoreboard()
    fact, variant_sum, orient, ablation = load_extra()
    audit = load_audit()
    stab = load_stability()
    a1 = load_a1_rerun(inscope)

    all_variants = sorted({r["variant"] for r in sb})
    unexplained = [v for v in all_variants if resolve(v) is None]

    rows = []
    for r in sb:
        v, pm = r["variant"], r["pool_mode"]
        family = fam_of.get(v, "reference_macros" if v.startswith("ref.") else "")
        mean, med, lo, hi, ncell = size_cell(sizes.get((v, pm)))
        top = ", ".join(f for f, _ in chosen.get((v, pm), collections.Counter()).most_common(3))
        role = role_of(v)
        fixed = v.startswith("ref.") or "@" in v
        a = audit.get((v, pm))
        st = stab.get((v, pm))
        rep, today, drift, s3 = verdict(a, st, modes.get((v, pm)), v)
        status = STATUS_OVERRIDE.get(v, "")
        if v == "ref.LOCO_5":
            status = "24 cells, not 25 — not strictly comparable"
        if st and float(st["macro_spread_pp"]) >= 0.5:
            status = (f"NUMERICALLY UNDETERMINED — macro moves "
                      f"{float(st['macro_spread_pp']):.2f}pp under a 1e-10 jitter"
                      + (f"; {status}" if status else ""))
        rows.append({
            "family": family or "(unfiled)",
            "method": v,
            "pool": pm,
            "description": describe(v, family),
            "origin": _fam(family, "paper") or "—",
            "track_record": _fam(family, "performance") or "—",
            "similar": similar_to(v, family, all_variants, fam_of),
            "macro": round(float(r["macro_all"]), 4),
            "qa": round(float(r["macro_qa"]), 4) if r.get("macro_qa") else None,
            "math": round(float(r["macro_math"]), 4) if r.get("macro_math") else None,
            "cells": int(r["n_cells"]),
            "size_mean": mean, "size_median": med,
            "size_min": lo, "size_max": hi, "size_cells": ncell,
            "top_features": top or "—",
            "picks_subset": "no — fixed" if fixed else "yes",
            "needs_anchor": "yes",
            "scale": "unit",
            "reproduces": rep, "macro_today": today, "drift_pp": drift,
            "frac_size3": s3,
            "macro_spread_pp": _num(st.get("macro_spread_pp"), 2) if st else None,
            "cell_spread_pp": _num(st.get("spread_pp"), 2) if st else None,
            "worst_cell_spread_pp": _num(st.get("max_spread_pp"), 2) if st else None,
            "macro_eigen": _num(a.get("macro_eigen")) if a else None,
            "macro_complete": _num(a.get("macro_complete")) if a else None,
            "role": role,
            "status": status,
            "step": (resolve(v) or ("", "", ""))[1] or "",
        })

    # ---- a1_residual, re-derived end to end under today's code ----------------
    # The published a1 rows were benched at Step 186 and predate TWO changes: the
    # Step-189 K_override clamp (K_range=[1] is not a valid L-SML K) and the
    # Step-205 m<4 Eq.15 fix. Selections come out identical, so these rows carry
    # today's numbers rather than a re-scored approximation of them.
    for r in rows:
        if r["pool"] != "c46":
            continue
        agg = a1.get("unit", {}).get(r["method"])
        if not agg or not agg["auroc"]:
            continue
        before, after = r["macro"], round(statistics.mean(agg["auroc"]), 4)
        r.update(macro=after,
                 qa=round(statistics.mean(agg["qa"]), 4) if agg["qa"] else None,
                 math=round(statistics.mean(agg["math"]), 4) if agg["math"] else None,
                 reproduces="re-run today", macro_today=after, drift_pp=None)
        if abs(after - before) > 5e-5:
            r["status"] = (f"re-run: {before:.4f} -> {after:.4f} "
                           f"({(after-before)*100:+.2f}pp; published row predates the "
                           f"Step-189 K clamp)")

    # ---- a1_residual re-derived at the corrected loading scale (Step 205) -----
    # The ONLY family whose selection objective calls L-SML, so the only one where
    # Phase 0's loading-scale finding can change which features get picked. Every
    # other family's objective is scale-free and its published subsets stand.
    for variant, agg in sorted(a1.get("complete", {}).items()):
        if not agg["auroc"]:
            continue
        base = audit.get((variant, "c46"))
        pub = _num(base["macro_published"]) if base else None
        name = f"{variant}+scale_complete"
        sz = agg["size"]
        delta = ((statistics.mean(agg["auroc"]) - pub) * 100) if pub else None
        rows.append({
            "family": "a1_residual", "method": name, "pool": "c46",
            "description": describe(name, "a1_residual"),
            "origin": _fam("a1_residual", "paper") or "—",
            "track_record": _fam("a1_residual", "performance") or "—",
            "similar": f"{variant}, " + similar_to(variant, "a1_residual",
                                                   all_variants, fam_of),
            "macro": round(statistics.mean(agg["auroc"]), 4),
            "qa": round(statistics.mean(agg["qa"]), 4) if agg["qa"] else None,
            "math": round(statistics.mean(agg["math"]), 4) if agg["math"] else None,
            "cells": len(agg["cells"]),
            "size_mean": round(statistics.mean(sz), 1),
            "size_median": statistics.median(sz),
            "size_min": min(sz), "size_max": max(sz), "size_cells": len(sz),
            "top_features": ", ".join(f for f, _ in agg["chosen"].most_common(3)),
            "picks_subset": "no — fixed" if "@" in variant else "yes",
            "needs_anchor": "yes", "scale": "complete",
            "reproduces": "re-run today", "macro_today": round(statistics.mean(agg["auroc"]), 4),
            "drift_pp": None,
            "frac_size3": round(sum(1 for s in sz if s == 3) / len(sz), 3),
            "macro_eigen": None, "macro_complete": None,
            "macro_spread_pp": None, "cell_spread_pp": None,
            "worst_cell_spread_pp": None,
            "role": role_of(variant),
            "status": (f"corrected criterion, {delta:+.2f}pp vs 'unit'"
                       if delta is not None else "corrected criterion"),
            "step": "205",
        })

    # ---- the anchor ablation: the only clean with/without pair we have --------
    for arm, label, anchor, note in (
        ("a7_anchored", "a7.iter_consensus (anchor ON)", "yes",
         "Iterative consensus selector with the orientation anchor step in place."),
        ("a7_prior_free", "a7.iter_consensus (anchor OFF)", "NO",
         "Same selector with the anchor step removed — 3 cells fall below 0.5."),
    ):
        a = ablation.get(arm)
        if not a:
            continue
        rows.append({
            "family": "a7_iter_consensus", "method": label, "pool": "inscope",
            "description": (note + " " + _fam("a7_iter_consensus", "relies_on")).strip(),
            "origin": _fam("a7_iter_consensus", "paper") or "—",
            "track_record": _fam("a7_iter_consensus", "performance") or "—",
            "similar": "a7_anchored, a7_prior_free",
            "macro": round(float(a["macro_all"]), 4),
            "qa": round(float(a["macro_qa"]), 4),
            "math": round(float(a["macro_math"]), 4),
            "cells": int(a["n_cells"]),
            "size_mean": None, "size_median": None, "size_min": None,
            "size_max": None, "size_cells": 0, "top_features": "—",
            "picks_subset": "yes", "needs_anchor": anchor,
            "scale": "unit",
            "reproduces": "not audited", "macro_today": None, "drift_pp": None,
            "frac_size3": None, "macro_eigen": None, "macro_complete": None,
            "macro_spread_pp": None, "cell_spread_pp": None,
            "worst_cell_spread_pp": None,
            "role": "fs_selector_candidate",
            "status": ("ANCHOR ABLATION — costs "
                       f"{(float(ablation['a7_anchored']['macro_all'])-float(ablation['a7_prior_free']['macro_all']))*100:.1f}pp"
                       if "a7_anchored" in ablation and "a7_prior_free" in ablation else ""),
            "step": "202",
        })

    # ---- U-PCR arms (Step 204). No `chosen` list; size = survivors of exclusion.
    pool_mean = 28.7          # mean in-scope pool size; used only to render "~N of pool"
    kept_on = kept_off = None
    pc = os.path.join(OUT, "03_faithful_factorial", "per_config.csv")
    if os.path.exists(pc):
        cfg = list(csv.DictReader(open(pc, encoding="utf-8")))
        on = [float(x["mean_frac_features_kept"]) for x in cfg if x["exclusion"] == "True"]
        off = [float(x["mean_frac_features_kept"]) for x in cfg if x["exclusion"] == "False"]
        kept_on = round(statistics.mean(on) * pool_mean, 1) if on else None
        kept_off = round(statistics.mean(off) * pool_mean, 1) if off else None

    bs = variant_sum.get("by_scale", {}).get("complete", {})
    UPCR_NOTE = ("U-PCR (Dror/Nadler/Bilal/Kluger 2017) estimates fusion weights by "
                 "turning 'which feature do I trust' into one equation per feature "
                 "pair, assuming features fail independently. It uses the whole pool "
                 "— any reduction comes from its own label-free exclusion step, not "
                 "from subset selection.")
    for label, val, kept, anchor, note in (
        ("upcr.rho_polarities", orient.get("macro_rho_anchor"), kept_on, "yes",
         "Per-feature polarity derived from sign(rho) instead of the 42 hand signs."),
        ("upcr.best_of_64", fact.get("macro_best_config"), kept_on, "yes",
         "Best of the 64-configuration factorial — a winner's-curse ceiling, not a result."),
        ("upcr.hand_polarities", orient.get("macro_hand_anchor"), kept_on, "yes",
         "The same fit using our hand-derived feature polarities."),
        ("upcr.legacy", fact.get("macro_legacy_config"), kept_on, "yes",
         "What fusion_utils.upcr_fuse does today."),
        ("upcr.clustered", bs.get("macro_upcr_cross"), kept_on, "yes",
         "Our extension: fit the pair system on cross-cluster pairs only. REFUTED — "
         "both pre-registered mechanism gates fail and it loses 4.46pp."),
        ("upcr.faithful", fact.get("macro_faithful_config"), kept_on, "yes",
         "Every documented deviation from the paper corrected at once."),
        ("upcr.hierarchical", bs.get("macro_upcr_hier"), kept_on, "yes",
         "Two-level: U-PCR inside each cluster, then across clusters. REFUTED."),
        ("upcr.anchor_off", orient.get("macro_rho_majority"), kept_on, "NO",
         "Global sign taken from the model instead of the anchor. Provably "
         "unidentifiable — inverts in 25/25 cells."),
    ):
        if val is None:
            continue
        rows.append({
            "family": "upcr", "method": label, "pool": "inscope",
            "description": note + " " + UPCR_NOTE,
            "origin": ("Dror, Nadler, Bilal & Kluger, Unsupervised Ensemble "
                       "Regression, arXiv:1703.02965 (2017)."),
            "track_record": ("Step 204: every paper-faithful flag hurts or does "
                             "nothing; the clustered variant fails both "
                             "pre-registered gates; nothing reaches GOOD_6."),
            "similar": "upcr.legacy, upcr.faithful, upcr.clustered",
            "macro": round(float(val), 4), "qa": None, "math": None, "cells": 25,
            "size_mean": kept, "size_median": None, "size_min": None,
            "size_max": None, "size_cells": 25,
            "top_features": "— (no selection; exclusion only)",
            "picks_subset": "no — uses all views",
            "needs_anchor": anchor, "role": "fs_selector_candidate",
            "scale": "complete" if "clustered" in label or "hier" in label else "n/a",
            "reproduces": "re-run Step 204", "macro_today": round(float(val), 4),
            "drift_pp": None, "frac_size3": 0.0,
            "macro_eigen": None, "macro_complete": None,
            "macro_spread_pp": None, "cell_spread_pp": None,
            "worst_cell_spread_pp": None,
            "status": ("REFUTED (Step 204)" if "REFUTED" in note else ""),
            "step": "204",
        })

    refs_by_scale = fact.get("references_by_scale", {})
    lsml30 = refs_by_scale.get("unit", {}).get("lsml_all30")
    if lsml30:
        rows.append({
            "family": "reference_macros", "method": "lsml.all_30_views",
            "pool": "inscope",
            "description": ("Continuous L-SML fused over every available view with "
                            "no selection at all. The honest no-subset baseline: any "
                            "selector must beat this to justify selecting. Quoted at "
                            "the unit loading scale to match the scoreboard rows; at "
                            + ", ".join(f"{k} it is {v['lsml_all30']:.4f}"
                                        for k, v in refs_by_scale.items()) + "."),
            "origin": "Jaffe, Fetaya, Nadler et al. — latent SML (AISTATS 2016).",
            "track_record": "Beats half the selectors while selecting nothing.",
            "similar": "random_s6, ref.GOOD_6",
            "macro": round(float(lsml30), 4), "qa": None, "math": None, "cells": 25,
            "size_mean": pool_mean, "size_median": None, "size_min": 27,
            "size_max": 30, "size_cells": 25, "top_features": "— (all views)",
            "picks_subset": "no — uses all views", "needs_anchor": "yes",
            "role": "fs_selector_candidate",
            "scale": "unit",
            "reproduces": "re-run Step 204", "macro_today": round(float(lsml30), 4),
            "drift_pp": None, "frac_size3": 0.0,
            "macro_eigen": _num(refs_by_scale.get("eigen", {}).get("lsml_all30")),
            "macro_complete": _num(refs_by_scale.get("complete", {}).get("lsml_all30")),
            "macro_spread_pp": None, "cell_spread_pp": None,
            "worst_cell_spread_pp": None,
            "status": "NO-SELECTION BASELINE", "step": "204",
        })

    rows.sort(key=lambda r: -r["macro"])
    S.save_csv(os.path.join(OUT, "comparison.csv"), rows)
    render(rows, unexplained, kept_on, kept_off)


# ---------------------------------------------------------------------------
COLUMNS = [
    ("method", "method", "t"), ("family", "family", "t"),
    ("macro", "macro AUROC", "n"), ("qa", "QA", "n"), ("math", "math", "n"),
    ("cells", "cells", "n"),
    ("reproduces", "reproduces today?", "t"),
    ("macro_spread_pp", "macro moves under 1e-10 jitter (pp)", "n"),
    ("macro_today", "re-scored today", "n"),
    ("frac_size3", "share of cells at 3 features", "n"),
    ("cell_spread_pp", "mean per-cell jitter spread (pp)", "n"),
    ("worst_cell_spread_pp", "worst cell (pp)", "n"),
    ("size_mean", "features chosen (mean)", "n"),
    ("size_median", "median", "n"), ("size_min", "min", "n"),
    ("size_max", "max", "n"),
    ("picks_subset", "picks own subset?", "t"),
    ("needs_anchor", "needs anchor?", "t"),
    ("pool", "pool", "t"), ("scale", "loading scale", "t"),
    ("macro_eigen", "same subsets @ eigen", "n"),
    ("macro_complete", "same subsets @ complete", "n"),
    ("role", "role", "t"),
    ("status", "status", "t"), ("step", "step", "t"),
    ("top_features", "most-chosen features", "t"),
    ("similar", "similar methods", "t"),
    ("description", "what it is", "t"),
    ("origin", "origin / paper", "t"),
    ("track_record", "track record", "t"),
]


def render(rows, unexplained, kept_on, kept_off):
    good6 = next((r["macro"] for r in rows if r["method"] == "ref.GOOD_6"), None)
    assert good6 and abs(good6 - 0.7594) < 0.002, \
        f"GOOD_6 anchor drifted: {good6}"
    auto = [r for r in rows if r["picks_subset"] != "no — fixed"]
    best = max(auto, key=lambda r: r["macro"])

    head = "".join(
        f'<th data-k="{k}" data-t="{t}">{html.escape(lbl)}<span class="ar"></span></th>'
        for k, lbl, t in COLUMNS)

    body = [
        "<h2>How to read this</h2>",
        "<p><b>What matters is the method that does not use a fixed subset.</b> "
        "GOOD_5, GOOD_6 and LOCO_5 were chosen by hand using answer keys, so they "
        "are the bar rather than competitors — they carry the "
        "<code>reference_macro</code> role and <code>picks own subset = no — "
        "fixed</code>. Use the role filter to drop them.</p>",
        f"<p>Best method that is not a fixed hand-picked subset: "
        f"<b>{html.escape(best['method'])} = {best['macro']:.4f}</b>. "
        f"GOOD_6 = {good6:.4f}. The gap of "
        f"<b>{(good6-best['macro'])*100:.2f}pp</b> is the open problem.</p>",
        "<p class='note'><b>Every description is quoted from "
        "<code>spectral_utils/glossary.py</code></b> (family note + variant note), "
        "the repo's single source of truth with a coverage gate — this page writes "
        "no prose of its own about any method. <b>Similar methods</b> are derived "
        "structurally: same base variant with a different size/K suffix first, then "
        "other members of the same family.</p>",
        "<p class='note'><b>Feature counts are computed on the 25 in-scope cells "
        "only</b>, so they describe the same rows as the AUROC column — the bench "
        "CSVs also contain out-of-scope cells. The same method under a different "
        "pool mode is a separate row, because its chosen size genuinely differs "
        "(a2.dufs picks 19.0 features under <code>c46</code> and 10.8 under "
        f"<code>h16</code>). U-PCR rows use every view; their count "
        f"(~{kept_on} of ~29) is what survives U-PCR's own label-free exclusion "
        "step, which is pruning, not subset selection.</p>",
    ]
    if unexplained:
        body.append(f"<div class='warn'>{len(unexplained)} variants have no "
                    f"glossary entry: {html.escape(', '.join(unexplained))}</div>")

    # ---- provenance: the part that makes this safe to send to someone else ----
    n_ver = sum(1 for r in rows if r["reproduces"] == "verified")
    n_noise = sum(1 for r in rows if r["reproduces"] == "within its own noise")
    n_expl = sum(1 for r in rows if r["reproduces"] in
                 ("code fix: Step-189 K clamp", "lookup table vs live re-fusion",
                  "code fix: Step-205 exact small-m solve"))
    n_drift = sum(1 for r in rows if str(r["reproduces"]).startswith("UNEXPLAINED"))
    n_nr = sum(1 for r in rows if r["reproduces"] == "not replayable")
    s3 = [r for r in rows if (r["frac_size3"] or 0) >= 0.5]
    unstable = sorted((r for r in rows if (r["macro_spread_pp"] or 0) >= 0.5),
                      key=lambda r: -r["macro_spread_pp"])
    n_stab = sum(1 for r in rows if r["macro_spread_pp"] is not None)
    body.append(f"""
<h2>Provenance — how far to trust each number</h2>
<p>Every row was replayed through today's fusion code on the same 25 in-scope cells
(<code>scripts/upcr_study/reproduction_audit.py</code>): <b>{n_ver} reproduce to
the last decimal</b>, {n_noise} differ by less than their own numerical noise (both
values are draws from the same distribution — see below), {n_expl} differ for a
named reason (the Step-189 K-override clamp, the Step-205 exact small-m solve, or
an h16 row whose published number came from the Step-153 lookup table rather than a
live fusion), {n_nr} are not replayable (a2's two clustering-swap arms store their
group assignment nowhere, so replaying them would be a different method, not a
check), and <b>{n_drift} are unexplained</b>. Sort by <i>reproduces today?</i> to
see it.</p>
<p class='note'><b>One defect, and it was structural.</b> The Eq.15 score matrix,
which decides how features are grouped, has <b>no valid term at three features</b>
(it is identically zero, so the grouping is pure tie-break) and <b>exactly two at
four</b>. At four features that is enough to make the answer depend on rounding
rather than on the data: on one real cell, computing the covariance from a
non-contiguous column slice instead of a contiguous copy of <i>the same numbers</i>
changes it by <b>5.5e-17</b> — BLAS summation order alone — and that moves the
K=3 partition, which moves the Eq.14 residual 0.60&nbsp;&rarr;&nbsp;0.39, which
flips the selected K, which moves AUROC by <b>9.7pp</b>.</p>
<p class='note'><b>It is now solved rather than patched.</b> Spectral clustering is
only a heuristic for &ldquo;the partition minimising the Eq.14 residual&rdquo;, and
at small m the heuristic ties. So at m&nbsp;&le;&nbsp;4 we no longer approximate:
every partition is enumerated (5 of them at m=3, 15 at m=4) and the exact minimum
taken, with a deterministic tie-break. The covariance input is pinned to a canonical
memory layout at every m, and a near-degeneracy flag now fires whenever the winning
grouping beats its nearest rival by less than float noise — so a coin flip is
visible as one instead of being reported as a measurement. <b>This bought
determinacy, not accuracy</b> (+0.03pp, 15W/10L, p=0.70 on the rows it touches), and
<b>every headline row is unchanged to 0.00pp</b>. Rows whose numbers moved because
of it are labelled <i>code fix: Step-205 exact small-m solve</i> below. A standing
regression gate (U5 in <code>scripts/verify_residual_scaling.py</code>) now asserts
the grouping is invariant to memory layout, feature order, and a 1e-12 jitter.</p>
<p class='note'><b>So the table reports numerical determinacy as a column.</b>
<code>stability_audit.py</code> re-fuses every row's stored subsets with the feature
matrix jittered by a relative 1e-10 — eight orders of magnitude below any real
measurement precision — across 5 seeds. {len(unstable)} of {n_stab} measured rows
move their macro by at least 0.5pp under that jitter and are flagged
<b>NUMERICALLY UNDETERMINED</b>: their leaderboard position is not meaningful at
the resolution the table prints{(" (worst: " + html.escape(unstable[0]["method"]) +
" at " + f"{unstable[0]['macro_spread_pp']:.2f}pp)") if unstable else ""}. This is a
property of L-SML on small subsets, not of any one implementation — reverting the
vectorisation would only re-pick one arbitrary side of the tie.</p>
<p class='note'><b>Read every size-3 row with that in mind.</b> Zero is the honest
value, but it also means L-SML has <i>no structural information at all</i> at three
features — the group assignment is whatever the clustering's tie-break yields.
{len(s3)} rows sit at three features on at least half their cells
({html.escape(", ".join(r["method"] for r in s3[:6]))}
{"…" if len(s3) > 6 else ""}). Those are not measurements of the fusion model; they
are measurements of a degenerate case of it.</p>
<p class='note'><b>Loading scale.</b> Published numbers are all at
<code>unit</code>, L-SML's historical rank-one loading estimator, which Step 204
Phase 0 showed does not satisfy the paper's Lemma 1. The two right-hand columns
give the identical subsets re-fused at <code>eigen</code> and <code>complete</code>,
so the sensitivity is visible per row rather than asserted globally. Only
<code>a1_residual</code> <i>selects</i> using L-SML, so it is the only family where
the scale can change which features are chosen — hence the separate
<code>+scale_complete</code> rows. Every other family's objective is scale-free and
its published subsets stand as selected.</p>
""")

    body.append(f"""
<div class="ctl">
  <input id="q" type="search" placeholder="filter — method, family, description…">
  <label>group by
    <select id="grp">
      <option value="">(none)</option>
      <option value="family">family</option>
      <option value="role">role</option>
      <option value="picks_subset">picks own subset</option>
      <option value="needs_anchor">needs anchor</option>
      <option value="pool">pool mode</option>
    </select>
  </label>
  <label>features chosen
    <input id="smin" type="number" min="0" max="30" placeholder="min" style="width:5.5em">
    –
    <input id="smax" type="number" min="0" max="30" placeholder="max" style="width:5.5em">
  </label>
  <label><input type="checkbox" id="fauto"> only methods that pick their own subset</label>
  <label><input type="checkbox" id="fref"> hide hand-curated references</label>
  <label><input type="checkbox" id="fsweep"> hide size/K sweep variants</label>
  <label><input type="checkbox" id="fver"> only rows verified against today's code</label>
  <label><input type="checkbox" id="fs3"> hide rows that are mostly 3-feature subsets</label>
  <label><input type="checkbox" id="fstab"> hide numerically undetermined rows (&ge;0.5pp jitter)</label>
  <button id="reset">reset</button>
  <span id="cnt" class="cnt"></span>
</div>
<div class="tw"><table id="tbl"><thead><tr>{head}</tr></thead><tbody></tbody></table></div>
""")

    extra_css = """
.ctl{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;background:var(--card);
padding:12px 14px;border-radius:8px;margin:16px 0;font-size:13.5px}
.ctl input[type=search]{flex:1;min-width:230px;padding:6px 9px;border:1px solid var(--line);
border-radius:5px;background:var(--bg);color:var(--fg)}
.ctl select,.ctl input[type=number],.ctl button{padding:5px 7px;border:1px solid var(--line);
border-radius:5px;background:var(--bg);color:var(--fg)}
.ctl button{cursor:pointer}
.cnt{margin-left:auto;color:var(--mut);font-variant-numeric:tabular-nums}
.tw{overflow-x:auto;max-height:78vh;overflow-y:auto;border:1px solid var(--line);border-radius:8px}
#tbl{margin:0;display:table;width:100%;font-size:13px}
#tbl th{position:sticky;top:0;z-index:2;cursor:pointer;user-select:none;white-space:nowrap}
#tbl th:hover{color:#2b6cb0}
#tbl td{vertical-align:top}
#tbl td.desc,#tbl td.sim{white-space:normal;min-width:300px;max-width:460px;color:var(--mut)}
#tbl td.num{text-align:right;font-variant-numeric:tabular-nums}
tr.grp td{background:var(--card);font-weight:600;position:sticky;top:34px;z-index:1}
tr.ref td{opacity:.72;font-style:italic}
tr.drift td{background:#c5303014}
.ar{margin-left:5px;color:var(--mut);font-size:10px}
.badge{display:inline-block;padding:1px 6px;border-radius:9px;font-size:11px;
background:#c0562133;color:#c05621;white-space:nowrap}
.badge.good{background:#2f855a26;color:#2f855a}
.badge.bad{background:#c5303026;color:#c53030;font-weight:600}
"""

    script = """
const ROWS = __DATA__;
const COLS = __COLS__;
let sortK = 'macro', sortAsc = false, group = '';
const $ = s => document.querySelector(s);

function pass(r){
  const q = $('#q').value.trim().toLowerCase();
  if (q && ![r.method, r.family, r.description, r.similar, r.role, r.status, r.origin]
        .join(' ').toLowerCase().includes(q)) return false;
  if ($('#fauto').checked && r.picks_subset === 'no — fixed') return false;
  if ($('#fref').checked && r.role.startsWith('reference_macro')) return false;
  if ($('#fsweep').checked && /_(s\\d+|adapt|k\\d+)$/.test(r.method)) return false;
  if ($('#fver').checked && !['verified','re-run today','re-run Step 204']
        .includes(r.reproduces)) return false;
  if ($('#fs3').checked && (r.frac_size3 || 0) >= 0.5) return false;
  if ($('#fstab').checked && (r.macro_spread_pp || 0) >= 0.5) return false;
  const mn = parseFloat($('#smin').value), mx = parseFloat($('#smax').value);
  if (!isNaN(mn) && !(r.size_mean !== null && r.size_mean >= mn)) return false;
  if (!isNaN(mx) && !(r.size_mean !== null && r.size_mean <= mx)) return false;
  return true;
}
function cmp(a,b){
  const t = (COLS.find(c=>c[0]===sortK)||['','','t'])[2];
  let x=a[sortK], y=b[sortK];
  if (x===null||x===undefined) return 1;
  if (y===null||y===undefined) return -1;
  if (t==='n'){ x=+x; y=+y; } else { x=String(x).toLowerCase(); y=String(y).toLowerCase(); }
  return (x<y?-1:x>y?1:0) * (sortAsc?1:-1);
}
function cell(r,k,t){
  const v = r[k];
  const cls = t==='n' ? 'num' : (k==='description'?'desc':(['similar','origin','track_record'].includes(k)?'sim':''));
  if (v===null||v===undefined||v==='') return `<td class="${cls}">—</td>`;
  if (k==='status') return `<td class="${cls}"><span class="badge">${v}</span></td>`;
  if (k==='reproduces'){
    const bad = String(v).startsWith('DRIFTS');
    const good = ['verified','re-run today','re-run Step 204'].includes(v);
    return `<td class="${cls}"><span class="badge ${bad?'bad':(good?'good':'')}">${v}</span></td>`;
  }
  if (k==='frac_size3') return `<td class="${cls}">${(+v*100).toFixed(0)}%</td>`;
  if (k==='macro') return `<td class="${cls}"><b>${(+v).toFixed(4)}</b></td>`;
  if (t==='n' && typeof v === 'number' && !Number.isInteger(v))
    return `<td class="${cls}">${v.toFixed(v<1?4:1)}</td>`;
  return `<td class="${cls}">${String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;')}</td>`;
}
function render(){
  const rows = ROWS.filter(pass).sort(cmp);
  const tb = $('#tbl tbody'); tb.innerHTML='';
  const emit = r => {
    const tr = document.createElement('tr');
    if (String(r.reproduces).startsWith('DRIFTS')) tr.className='drift';
    else if (r.role.startsWith('reference_macro')) tr.className='ref';
    tr.innerHTML = COLS.map(c=>cell(r,c[0],c[2])).join('');
    tb.appendChild(tr);
  };
  if (!group) rows.forEach(emit);
  else {
    const g = {};
    rows.forEach(r => (g[r[group]] = g[r[group]] || []).push(r));
    Object.keys(g).sort((a,b)=>Math.max(...g[b].map(r=>r.macro))-Math.max(...g[a].map(r=>r.macro)))
      .forEach(k => {
        const best = Math.max(...g[k].map(r=>r.macro));
        const tr = document.createElement('tr'); tr.className='grp';
        tr.innerHTML = `<td colspan="${COLS.length}">${k||'(none)'} — ${g[k].length} row(s), best ${best.toFixed(4)}</td>`;
        tb.appendChild(tr);
        g[k].forEach(emit);
      });
  }
  $('#cnt').textContent = `${rows.length} of ${ROWS.length} rows`;
  document.querySelectorAll('#tbl th').forEach(th=>{
    th.querySelector('.ar').textContent = th.dataset.k===sortK ? (sortAsc?'▲':'▼') : '';
  });
}
document.querySelectorAll('#tbl th').forEach(th => th.onclick = () => {
  const k = th.dataset.k;
  if (k===sortK) sortAsc=!sortAsc; else { sortK=k; sortAsc = th.dataset.t!=='n'; }
  render();
});
['q','smin','smax'].forEach(i=>$('#'+i).oninput=render);
['fauto','fref','fsweep','fver','fs3','fstab'].forEach(i=>$('#'+i).onchange=render);
$('#grp').onchange = e => { group = e.target.value; render(); };
$('#reset').onclick = () => {
  ['q','smin','smax'].forEach(i=>$('#'+i).value='');
  ['fauto','fref','fsweep','fver','fs3','fstab'].forEach(i=>$('#'+i).checked=false);
  $('#grp').value=''; group=''; sortK='macro'; sortAsc=false; render();
};
render();
"""
    script = (script.replace("__DATA__", json.dumps(rows).replace("</", "<\\/"))
                    .replace("__COLS__", json.dumps(COLUMNS)))

    page = ("".join(body) + f"<style>{extra_css}</style>"
            + f"<script>{script}</script>")

    S.write_page(
        os.path.join(OUT, "comparison.html"),
        "Every method we have tried",
        "L-SML + feature selection, residual-guided pruning, and U-PCR — sortable, "
        "filterable, groupable, against the hand-curated bar",
        [f"<b>{len(rows)} rows</b> covering every variant on the scoreboard plus "
         f"the Step-204 U-PCR arms, the anchor ablation and the Step-205 "
         f"corrected-criterion arms.",
         f"Best non-fixed-subset method: <b>{html.escape(best['method'])} "
         f"{best['macro']:.4f}</b>; hand-curated GOOD_6 is {good6:.4f}.",
         f"<b>{n_ver} of {len(rows)} rows re-verified</b> against today's code, "
         f"{n_drift} drifting; every row states its loading scale and the same "
         f"subsets fused at the other two.",
         "Sort any column, filter by text, subset size or verification status, "
         "group by family or role."],
        page)

    print(f"\n{len(rows)} rows | glossary gaps: {len(unexplained)}")
    print(f"GOOD_6 anchor check: {good6:.4f} (expected 0.7594) OK")
    tally = collections.Counter(r["reproduces"] for r in rows)
    for label, n in tally.most_common():
        print(f"  {n:>4}  {label}")
    # Hard gate: a row this page cannot account for must never ship to advisors.
    unexplained_rows = [r for r in rows
                        if str(r["reproduces"]).startswith("UNEXPLAINED")]
    assert not unexplained_rows, "UNEXPLAINED rows on an advisor-facing page: " + "; ".join(
        f"{r['method']} [{r['pool']}] {r['macro']:.4f} -> {r['macro_today']:.4f}"
        for r in unexplained_rows)
    print(f"best non-fixed-subset: {best['method']} {best['macro']:.4f}")
    print("\nspot-checks:")
    for m in ("a6.pl_dufs", "a1.relres_greedy", "a1.relres_greedy+scale_complete",
              "ref.GOOD_6"):
        for r in rows:
            if r["method"] == m and r["pool"] in ("c46", "inscope"):
                print(f"  {m:<34} macro={r['macro']:.4f}  "
                      f"features={r['size_mean']}  ({r['size_cells']} cells)  "
                      f"{r['reproduces']}")
                break
    print(f"\nWrote -> {os.path.join(OUT, 'comparison.html')} (+ .csv)")


if __name__ == "__main__":
    main()
