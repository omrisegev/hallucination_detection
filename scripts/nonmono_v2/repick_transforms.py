#!/usr/bin/env python
"""
repick_transforms.py — re-derive the chosen transform under the agreed label policy.

WHY SEPARATE. `transform_selection.py` scored every option; choosing among them is
a policy decision, not a measurement, and re-running the hinge sweep (19 centres x
9 asymmetries x cross-fitting x 99 pairs) to change a policy would be wasteful.
This patches `chosen`/`why` in place from the stored scores.

THE POLICY, IN TIERS

  Tier A  STRICTLY LABEL-FREE — no answer key anywhere, on any cell.
          squared, dist_median, abs_rank, mode_centre, consensus_centre,
          consensus_map. Deployable on a cell we have never labelled.

  Tier B  OFFLINE CROSS-CELL — fitted on the other 23 cells, frozen, applied to
          the held-out one (Omri, 2026-08-02). loco_centre, loco_binmap.
          Deployable, but only if the shape transfers across cells; the whole
          point of fitting them leave-one-cell-out is that this is testable.

  DIAG    THIS CELL'S LABELS — best_centre, hinge. Never deployable. They mark
          the family's own ceiling and answer "wrong family or wrong centre?".

`transform_selection.py` restricted the pick to Tier A, which contradicts the
agreed policy. The pick here is the best of A+B; the Tier-A-only pick is retained
alongside so the cost of the stricter rule is visible rather than assumed.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "results", "nonmono_v2", "transform_selection.json")

TIER_A = {"squared", "dist_median", "abs_rank", "mode_centre",
          "consensus_centre", "consensus_map"}
TIER_B = {"loco_centre", "loco_binmap"}
DIAG = {"best_centre", "hinge"}


def pick(opts, allowed, min_gain):
    cand = [o for o in opts if o["name"] in allowed]
    if not cand:
        return None
    best = max(cand, key=lambda o: o["auc"])
    return best if best["delta_pp"] >= min_gain else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gain", type=float, default=2.0)
    args = ap.parse_args()

    with open(SRC, encoding="utf-8") as fh:
        data = json.load(fh)
    mg = args.min_gain
    data["min_gain_pp"] = mg
    data["policy"] = {"tier_a": sorted(TIER_A), "tier_b": sorted(TIER_B),
                      "diagnostic": sorted(DIAG), "min_gain_pp": mg}

    n_a = n_b = n_none = 0
    for p in data["panels"]:
        opts = p["options"]
        for o in opts:
            o["tier"] = ("A" if o["name"] in TIER_A else
                         "B" if o["name"] in TIER_B else
                         "DIAG" if o["name"] in DIAG else "baseline")
        base = next((o for o in opts if o["name"] == "identity"), None)
        hd = p["headroom_pp"]

        a = pick(opts, TIER_A, mg)
        b = pick(opts, TIER_A | TIER_B, mg)
        p["pick_strict"] = a["name"] if a else "identity"

        if b is None:
            p["chosen"] = "identity"
            n_none += 1
            worst = min(o["delta_pp"] for o in opts if o["name"] != "identity")
            p["why"] = (
                f"kept raw. Only {hd:.1f}pp of non-monotone headroom exists here, so "
                f"there is nothing for a fold to recover, and the best option on offer "
                f"fell short of the +{mg:.0f}pp bar (worst option costs {worst:.1f}pp)."
                if hd < 3 else
                f"kept raw. {hd:.1f}pp of headroom exists, but no deployable option "
                f"cleared +{mg:.0f}pp — the shape is real and this family does not "
                f"capture it.")
        else:
            p["chosen"] = b["name"]
            tier = "A" if b["name"] in TIER_A else "B"
            n_a += tier == "A"
            n_b += tier == "B"
            frac = (100 * b["delta_pp"] / hd) if hd > 0 else float("nan")
            src = ("uses no labels at all" if tier == "A"
                   else "fitted on the other cells, frozen before use")
            why = (f"+{b['delta_pp']:.1f}pp over the raw view "
                   f"({base['auc']:.3f} &#8594; {b['auc']:.3f}); {src}.")
            if hd > 0:
                why += f" That recovers {frac:.0f}% of the {hd:.1f}pp available."
            dg = pick(opts, DIAG, -99)
            if dg and dg["auc"] > b["auc"] + 0.01:
                why += (f" A centre fitted on this cell's own labels would reach "
                        f"{dg['auc']:.3f} — the {dg['auc']-b['auc']:.3f} difference is "
                        f"what the label-free choice costs.")
            if a and a["name"] != b["name"]:
                why += (f" The strictly label-free pick would be {a['name']} at "
                        f"{a['auc']:.3f}.")
            p["why"] = why

    cands = [p for p in data["panels"] if p["is_candidate"]]
    ctrls = [p for p in data["panels"] if not p["is_candidate"]]
    print(f"repicked under tiers A+B, min gain +{mg}pp")
    print(f"  Tier A chosen : {n_a}")
    print(f"  Tier B chosen : {n_b}")
    print(f"  kept raw      : {n_none}")
    print(f"\n  candidates transformed : "
          f"{sum(1 for p in cands if p['chosen']!='identity')}/{len(cands)}")
    print(f"  controls transformed   : "
          f"{sum(1 for p in ctrls if p['chosen']!='identity')}/{len(ctrls)}"
          f"   (false positives a label-free rule must pay for)")
    strict_diff = [p for p in data["panels"] if p["pick_strict"] != p["chosen"]]
    print(f"  strict-vs-agreed policy differ on {len(strict_diff)} views")

    with open(SRC, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"\npatched {SRC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
