#!/usr/bin/env python
"""
Pilot-gate direction-sanity check for the RAGTruth evidence-contrast campaign.

Preregistered gate (docs/research_notes/ragtruth_ec_preregistration_v1.md,
cluster/manifests/ragtruth_ec_v1.json "sequence"): on GROUNDED responses (no annotated
hallucination span, response_label == False), removing the evidence entirely should make the
model MORE surprised by the (faithful) published text than seeing the full context did —
i.e. mean(NLL_noctx - NLL_full) > 0. This is a sanity check on the scorer/prompt-surgery
pipeline, not a result: it only reads `response_label`, never the span-level annotations, and
is run on the small pilot before any full-scale submission or metric freeze.

NLL of a response under a condition = sum(token_spilled_energies) for that
(response_id, condition) cache entry — see CLAUDE.md's per-sample raw-inference-data table:
token_spilled_energies[n] = -log p(sampled/forced token) at position n.

Pure stdlib (pickle only) so it runs on a bare login-node python with no project deps.
"""
import argparse
import pickle
import statistics
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("pkl_path")
    args = ap.parse_args()

    with open(args.pkl_path, "rb") as f:
        cache = pickle.load(f)

    by_response = {}
    for key, entry in cache.items():
        rid, cond = entry["response_id"], entry["condition"]
        by_response.setdefault(rid, {})[cond] = entry

    n_total = len(by_response)
    n_missing_full_or_noctx = 0
    deltas_grounded, deltas_ungrounded = [], []
    for rid, conds in by_response.items():
        if "full" not in conds or "noctx" not in conds:
            n_missing_full_or_noctx += 1
            continue
        nll_full = sum(conds["full"]["token_spilled_energies"])
        nll_noctx = sum(conds["noctx"]["token_spilled_energies"])
        delta = nll_noctx - nll_full
        grounded = not conds["full"]["response_label"]
        (deltas_grounded if grounded else deltas_ungrounded).append(delta)

    print(f"responses total={n_total} missing_full_or_noctx={n_missing_full_or_noctx}")
    print(f"grounded (response_label=False) n={len(deltas_grounded)}")
    print(f"ungrounded (response_label=True) n={len(deltas_ungrounded)}")

    ok = True
    if deltas_grounded:
        mean_g = statistics.mean(deltas_grounded)
        frac_pos_g = sum(d > 0 for d in deltas_grounded) / len(deltas_grounded)
        print(f"grounded: mean(NLL_noctx - NLL_full)={mean_g:.4f} "
              f"frac_positive={frac_pos_g:.2f} "
              f"min={min(deltas_grounded):.4f} max={max(deltas_grounded):.4f}")
        ok = mean_g > 0
    else:
        print("grounded: NO DATA — cannot evaluate the direction-sanity gate")
        ok = False

    if deltas_ungrounded:
        mean_u = statistics.mean(deltas_ungrounded)
        frac_pos_u = sum(d > 0 for d in deltas_ungrounded) / len(deltas_ungrounded)
        print(f"ungrounded: mean(NLL_noctx - NLL_full)={mean_u:.4f} "
              f"frac_positive={frac_pos_u:.2f} "
              f"min={min(deltas_ungrounded):.4f} max={max(deltas_ungrounded):.4f}")

    print(f"\nGATE (mean grounded delta > 0): {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
