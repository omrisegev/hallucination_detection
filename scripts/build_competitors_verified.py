#!/usr/bin/env python
"""
build_competitors_verified.py — one auditable source for every published competitor
number that appears in the in-scope advisor deliverable.

Why this exists (Step 193, Phase 0.2). The competitor numbers lived in two places with
incompatible provenance:

  results/repgrid/published_baselines.csv   6 cells, AUROC on a 0-100 scale, WITH a
                                            `source` column naming paper + table.
  results/repgrid/scores_lsml_upcr.csv      19 cells, `published_Y`/`Y_method` on a 0-1
                                            scale, with NO citation at all — no paper,
                                            no table, no page, no scale note.

An advisor-facing claim of the form "we are competitive with the published numbers" is
only as good as those numbers, so each one was checked against `papers/extracted/<slug>.md`
(the raw extracted text, NOT the digest cards — `papers/index.md` records a digest pass
that fabricated datasets, models and venues across 9 papers).

Output: results/advisor_inscope/competitors_verified.csv, with a `verified_by` column
that is one of:
    extracted   — value, dataset, model and metric confirmed against the paper's own table
    unverified  — the paper is not in papers/, so nothing could be confirmed locally

Two corrections this audit produced are applied here and flagged in `caveat`:
  * HARP is SUPERVISED (the paper trains the detector with a binary cross-entropy loss on
    hallucination labels), not unsupervised as published_baselines.csv recorded.
  * HARP 92.8 is the Qwen-2.5-7B-Instruct row; the Llama-3.1-8B row (our cell's model) is
    92.9. The stored anchor was cross-model.

Usage:
    python scripts/build_competitors_verified.py
"""
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inscope_cells import INSCOPE

PUBLISHED = os.path.join(REPO, "results", "repgrid", "published_baselines.csv")
SCORES = os.path.join(REPO, "results", "repgrid", "scores_lsml_upcr.csv")
OUT = os.path.join(REPO, "results", "advisor_inscope", "competitors_verified.csv")

# ── Verification ledger ───────────────────────────────────────────────────────────────
# Keyed by the paper a method's number comes from. Every entry was confirmed by reading
# papers/extracted/<slug>.md and locating the exact table row. `columns` records the
# table's column order, because several of these audits turned on getting it right.
PAPERS = {
    "EPR": dict(
        slug="epr",
        table="Table 1 (K=15)",
        columns="SCGPT | EPR (ours) | HalluDetect | WEPR (ours)",
        evidence="TriviaQA (Hallucination Detection) / Mistral-Small-24B row: "
                 "79.0 | 74.6 | 78.7 | 82.0",
    ),
    "HCPD": dict(
        slug="zero-source-llm-hallucination-detection-with-human-like-crit",
        table="Table 2",
        columns="TriviaQA | SciQ | NQ Open | CoQA | Avg",
        evidence="Llama-3.1-8b block. HCPD 86.25|86.04|90.38; SemEntropy 78.71|77.81|61.04; "
                 "TSV 79.78|80.01|70.17; SAPLMA 78.51|85.63|76.23; Perplexity 80.62|66.12|57.92. "
                 "Caption: AUROC (%); the club glyph marks methods trained on fully labeled data.",
    ),
    "ARS": dict(
        slug="harnessing-reasoning-trajectories-for-hallucination-detectio",
        table="Table 1",
        columns="Single Sampling | Supervision | TruthfulQA | TriviaQA | GSM8K | MATH-500",
        evidence="DeepSeek-R1-Distill-Llama-8B block, ARS (CCS): 80.89 | 88.86 | 74.72 | 86.38. "
                 "Single Sampling = yes, Supervision = no. All values AUROC %.",
    ),
    "NOISE": dict(
        slug="enhancing-hallucination-detection-through-noise-injection",
        table="Table 4",
        columns="GSM8K | CSQA | TriviaQA",
        evidence="'w/ Noise' rows: Mistral-7B-Instruct-v0.3 78.50; Phi-3-mini-4k-instruct 72.51. "
                 "Detection AUROC, K=10 samples.",
    ),
    "SEMENERGY": dict(
        slug="semantic-energy-detecting-llm-hallucination-beyond-entropy",
        table="Table 1",
        columns="Semantic Entropy (AUROC/AUPR/FPR95) | Semantic Energy (AUROC/AUPR/FPR95)",
        evidence="Qwen3-8B / TriviaQA row: SE 69.6% vs Semantic Energy 74.8%.",
    ),
    "ALS": dict(
        slug="automatic-layer-selection-for-hallucination-detection",
        table="Table 2 (w=7)",
        columns="CoQA | SQuAD | HotpotQA | TriviaQA | PsiloQA | Avg",
        evidence="LlaMA-3.1-8B-Instruct block, reported as 0-1 decimals: Pred.Entropy "
                 ".5703/.6859; SemEntropy .5518; Lexical .5988/.6838; FEPoID .6377/.7516.",
    ),
    "LAPEIG": dict(
        slug="hallucination-detection-in-llms-using-spectral-features-of-a",
        table="Table 1 (temp=1.0, test AUROC)",
        columns="CoQA | GSM8K | HaluevalQA | NQOpen | SQuADv2 | TriviaQA | TruthfulQA",
        evidence="Binkowski et al. GSM8K column. AttentionScore: Llama3.2-3B .717, "
                 "Llama3.1-8B .720, Phi3.5 .666, Mistral-Nemo .630, Mistral-Small-24B .576. "
                 "LapEigvals: .870 / .872 / .885 / .890 / .925. Caption: 'a single run of "
                 "logistic regression training'; 'We mark results for AttentionScore in gray "
                 "as it is an unsupervised approach, not directly comparable to the others'.",
    ),
    "INSIDE": dict(
        slug="inside-llms-internal-states-retain-the",
        table="Table 1 (AUCs = sentence-similarity correctness)",
        columns="CoQA | SQuAD | NQ | TriviaQA, each as AUCs | AUCr | PCC",
        evidence="Chen et al., ICLR 2024 (Alibaba Cloud / Zhejiang). LLaMA-7B row, "
                 "EigenScore: CoQA AUCs 80.4. Implementation: 'The number of generations is "
                 "set to K = 10'; sentence embedding is 'the last token embedding ... in the "
                 "middle layer' -> multi-pass and white-box. INSIDE is the framework; the "
                 "score itself is called EigenScore.",
    ),
    "LOSNET": dict(
        slug="beyond-next-token-probabilities-learnable-fast-detection-of",
        table="Table 1 (test AUC, Mis-7b / L-3-8b)",
        columns="HotpotQA + 5 further HD/DCD settings",
        evidence="Bar-Shalom, Frasca et al. (Technion/MIT/Nvidia). HotpotQA/Mistral-7B: "
                 "LOS-Net 72.92+-0.45; Semantic Entropy 67.66; ATP+R-Transf. 69.70; "
                 "ATP+R-MLP 68.92; Act. Probe 73.00; Probas-mean 63; Logits-mean 61; "
                 "Logits-min 61; Logits-max 53; Probas-min 58; Probas-max 50; p(True) 54. "
                 "Caption: 'orange indicates baselines requiring additional "
                 "prompting/generations'; Activation Probes 'are incomparable as they access "
                 "model internals'. The paper's own taxonomy: white-box = model internals, "
                 "gray-box = operating only on LLM outputs (LOS-Net is gray-box).",
    ),
    "INTSTATES": dict(
        slug="hallucination-detection-via-internal-states-and-structured-r",
        table="Table 1 (AUROC, temp 0.8)",
        columns="TruthfulQA | TriviaQA | GSM8K",
        evidence="Song, Qiu, Zhang, Tang (Beijing Univ. of Posts and Telecommunications). "
                 "Qwen2.5-7B row, 'ours': 84.03 | 85.68 | 79.15 -> GSM8K 79.15. "
                 "Supervised: a cross-attention CLASSIFIER trained on "
                 "Hallucination/Non-Hallucination labels. White-box: 'LLM Internal States "
                 "Extraction'. Multi-pass: 'signals from three complementary reasoning "
                 "paths' (Answer, CoT Answer, Reverse Query).",
    ),
    "HARP": dict(
        slug="harp-hallucination-detection-via-reasoning-subspace-projecti",
        table="main results table",
        columns="Single | NQ Open | TruthfulQA | TriviaQA | TyDiQA",
        evidence="Qwen-2.5-7B-Instruct HARP TriviaQA 92.8; LLaMA-3.1-8B HARP TriviaQA 92.9. "
                 "Detector trained with binary cross-entropy on hallucination labels "
                 "(flag in {0,1}) -> SUPERVISED.",
    ),
}

# Which paper each stored method name was checked against. Methods absent here have no
# local PDF and stay `unverified`.
METHOD_PAPER = {
    "SelfCheckGPT": "EPR", "EPR": "EPR", "HalluDetect": "EPR", "WEPR": "EPR",
    "HCPD": "HCPD", "Perplexity": "HCPD", "Semantic Entropy": "HCPD",
    "SAPLMA": "HCPD", "TSV": "HCPD",
    "ARS (CCS)": "ARS",
    "Noise Injection": "NOISE",
    "Semantic Energy": "SEMENERGY",
    "Pred. Entropy (ALS)": "ALS", "Lexical Similarity (ALS)": "ALS",
    "Semantic Entropy (ALS)": "ALS", "FEPoID (ALS ceiling)": "ALS",
    "LapEigvals": "LAPEIG", "AttentionScore (LapEigvals paper)": "LAPEIG",
    "HARP": "HARP",
    "INSIDE": "INSIDE",
    "Internal-States+RC": "INTSTATES",
    "LOS-Net": "LOSNET", "ATP+R-MLP": "LOSNET", "ATP+R-Transf.": "LOSNET",
    "Activation Probes": "LOSNET", "Logits-mean": "LOSNET", "Logits-min": "LOSNET",
    "Logits-max": "LOSNET", "Probas-mean": "LOSNET", "Probas-min": "LOSNET",
    "Probas-max": "LOSNET", "p(True)": "LOSNET",
}

# ── LapEigvals (Step 193 follow-up) ───────────────────────────────────────────────────
# Omri pointed out the paper IS in papers/, filed under its title rather than the method
# name: "Hallucination Detection in LLMs Using Spectral Features of Attention" (Binkowski,
# Janiak, Sawczyn, Gabrys, Kajdanowicz). Verifying it showed all 5 stored anchors were
# mislabeled, so the 5 lapeigvals_* cells are rebuilt from this table instead:
#   * 4 of the 5 stored values are the paper's AttentionScore baseline, not LapEigvals.
#     That is the FAIR comparator for us (the caption calls AttentionScore "an unsupervised
#     approach, not directly comparable to the others") — the value was right, the name and
#     the supervision tag were wrong.
#   * lapeigvals_gsm8k_llama8b stored 0.925, which is Mistral-Small-24B's LapEigvals — a
#     different model. Its own model's numbers are .720 / .872.
#   * LapEigvals itself is a SUPERVISED logistic-regression probe over Laplacian eigenvalues
#     of attention maps, so it belongs against our LR oracle, not against our label-free score.
LAPEIG_GSM8K = {   # our cell -> (paper model, AttentionScore [unsup], LapEigvals [sup])
    "lapeigvals_gsm8k_llama3b":     ("Llama3.2-3B",       0.717, 0.870),
    "lapeigvals_gsm8k_llama8b":     ("Llama3.1-8B",       0.720, 0.872),
    "lapeigvals_gsm8k_phi35":       ("Phi3.5",            0.666, 0.885),
    "lapeigvals_gsm8k_nemo":        ("Mistral-Nemo",      0.630, 0.890),
    "lapeigvals_gsm8k_mistral24b":  ("Mistral-Small-24B", 0.576, 0.925),
}

# Methods with no PDF in papers/ — recorded so the report can name what is missing.
NO_LOCAL_PDF = {
    "Semantic Entropy (SE-ICLR'23)": "Kuhn et al. ICLR 2023 not in papers/",
    "TSV (arXiv 2503.01917)": "TSV paper not in papers/; the TSV rows inside HCPD Table 2 "
                              "are different datasets and cannot source the TruthfulQA anchor",
}

# Supervision for the single-anchor rows. scores_lsml_upcr.csv has no supervision column,
# but the grid must not put a supervised competitor next to our label-free score without
# saying so. Values below are read off the papers verified above; anything not listed is
# left blank rather than guessed.
ANCHOR_SUPERVISION = {
    "EPR": "unsupervised",                      # EPR Table 1, "Unsupervised" column header
    "HCPD (arXiv 2606.12900)": "unsupervised",  # HCPD Table 2, no club glyph on HCPD
    "ARS (CCS)": "unsupervised",                # ARS Table 1, Supervision = no
    "Noise Injection": "unsupervised",          # answer-entropy over K=10 noisy samples
    "Semantic Energy": "unsupervised",          # training-free energy over semantic clusters
}

# ── Method profile: what each competitor actually costs to run ────────────────────────
# Three axes an advisor needs in order to read the grid as a fair comparison:
#   supervision — does it need hallucination labels to fit anything?
#   access      — black-box (text only) / grey-box (output distributions, logprobs) /
#                 white-box (hidden states, attention maps, activations).
#                 The taxonomy is the LOS-Net paper's own: "probing techniques ... require
#                 restrictive white-box access to model internals ... gray-box methods relax
#                 these assumptions by operating only on LLM outputs".
#   passes      — generations per question at detection time (1 = single-pass).
# `src` is "paper" when all three were read off a paper in papers/extracted/, "inferred"
# when the paper is not in the repo and the profile comes from the method's description
# elsewhere — never silently mixed.
#
# OUR method for reference: unsupervised / grey-box (top-k logprobs) / 1 pass.
METHOD_PROFILE = {
    # --- EPR paper
    "EPR":                      ("unsupervised", "grey-box", "1", "paper"),
    "WEPR":                     ("supervised",   "grey-box", "1", "paper"),
    "HalluDetect":              ("supervised",   "grey-box", "1", "paper"),
    "SelfCheckGPT":             ("unsupervised", "black-box", "K (sampled)", "paper"),
    # --- HCPD paper
    "HCPD":                     ("unsupervised", "black-box", "K (multi-sample)", "paper"),
    "HCPD (arXiv 2606.12900)":  ("unsupervised", "black-box", "K (multi-sample)", "paper"),
    "Perplexity":               ("unsupervised", "grey-box", "1", "paper"),
    "Semantic Entropy":         ("unsupervised", "black-box", "K (sampled)", "paper"),
    "SAPLMA":                   ("supervised",   "white-box", "1", "paper"),
    "TSV":                      ("supervised",   "white-box", "1", "paper"),
    # --- ARS paper (Table 1 has explicit "Single Sampling" and "Supervision" columns)
    "ARS (CCS)":                ("unsupervised", "white-box", "1", "paper"),
    # --- Noise Injection paper (K=10 noisy samples, activation-space perturbation)
    "Noise Injection":          ("unsupervised", "white-box", "10", "paper"),
    # --- Semantic Energy paper (energy over semantic clusters of sampled answers)
    "Semantic Energy":          ("unsupervised", "grey-box", "K (sampled)", "paper"),
    # --- ALS paper (10 candidate answers per question; FEPoID is an MLP probe)
    "Pred. Entropy (ALS)":      ("unsupervised", "grey-box", "10", "paper"),
    "Lexical Similarity (ALS)": ("unsupervised", "black-box", "10", "paper"),
    "Semantic Entropy (ALS)":   ("unsupervised", "black-box", "10", "paper"),
    "FEPoID (ALS ceiling)":     ("supervised",   "white-box", "1", "paper"),
    # --- HARP paper (BCE-trained detector on reasoning-subspace projections)
    "HARP":                     ("supervised",   "white-box", "1", "paper"),
    # --- LapEigvals paper (Laplacian eigenvalues of attention maps -> LR probe)
    "LapEigvals":                        ("supervised",   "white-box", "1", "paper"),
    "AttentionScore (LapEigvals paper)": ("unsupervised", "white-box", "1", "paper"),
    # --- INSIDE paper: "The number of generations is set to K = 10"; embeddings taken from
    #     the middle layer -> white-box, multi-pass, no labels fitted.
    "INSIDE":                   ("unsupervised", "white-box", "10", "paper"),
    # --- LOS-Net paper (learnable net over the full output-distribution sequence)
    "LOS-Net":                  ("supervised",   "grey-box", "1", "paper"),
    "ATP+R-MLP":                ("supervised",   "grey-box", "1", "paper"),
    "ATP+R-Transf.":            ("supervised",   "grey-box", "1", "paper"),
    "Activation Probes":        ("supervised",   "white-box", "1", "paper"),
    "Logits-mean":              ("unsupervised", "grey-box", "1", "paper"),
    "Logits-min":               ("unsupervised", "grey-box", "1", "paper"),
    "Logits-max":               ("unsupervised", "grey-box", "1", "paper"),
    "Probas-mean":              ("unsupervised", "grey-box", "1", "paper"),
    "Probas-min":               ("unsupervised", "grey-box", "1", "paper"),
    "Probas-max":               ("unsupervised", "grey-box", "1", "paper"),
    "p(True)":                  ("unsupervised", "black-box", "2 (extra prompt)", "paper"),
    # --- no local PDF: profile inferred from the method's description elsewhere
    "Internal-States+RC":       ("supervised",   "white-box", "3 (multi-path)", "paper"),
    "Semantic Entropy (SE-ICLR'23)": ("unsupervised", "black-box", "K (sampled)", "inferred"),
    "TSV (arXiv 2503.01917)":   ("supervised",   "white-box", "1", "inferred"),
}

# Corrections this audit produced, applied to the emitted rows.
CORRECTIONS = {
    ("spilled_triviaqa_llama8b", "HARP"): dict(
        supervision="supervised",
        caveat="CORRECTED (Step 193): published_baselines.csv tagged HARP unsupervised, but the "
               "paper trains the detector with binary cross-entropy on hallucination labels. "
               "Also, 92.8 is the Qwen-2.5-7B-Instruct row; the LLaMA-3.1-8B row is 92.9, so the "
               "stored anchor was CROSS-MODEL. Treat as a supervised, cross-model reference.",
    ),
}


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def profile_fields(method, fallback_sup=""):
    """access / passes / supervision for one method, plus how the profile was sourced."""
    prof = METHOD_PROFILE.get(method)
    if prof is None:
        base = method.split(" (")[0].strip()
        prof = METHOD_PROFILE.get(base)
    if prof is None:
        return dict(access="", passes="", profile_src="", supervision=fallback_sup)
    sup, access, passes, src = prof
    return dict(access=access, passes=passes, profile_src=src,
                supervision=fallback_sup or sup)


def paper_fields(method):
    key = METHOD_PAPER.get(method)
    if key is None:
        return dict(verified_by="unverified", paper_slug="", paper_table="",
                    paper_columns="", paper_evidence="")
    p = PAPERS[key]
    return dict(verified_by="extracted", paper_slug=p["slug"], paper_table=p["table"],
                paper_columns=p["columns"], paper_evidence=p["evidence"])


def main():
    rows = []

    # ── source A: published_baselines.csv (0-100 scale, multi-competitor, cited) ───────
    for r in read_csv(PUBLISHED):
        cell = r["cell"]
        if cell not in INSCOPE:
            continue
        method = r["method"]
        rec = dict(
            cell=cell, dataset=r.get("dataset", ""), model=r.get("model", ""),
            method=method, supervision=r.get("supervision", ""),
            auroc=round(float(r["auroc"]) / 100.0, 4),
            auroc_as_published=r["auroc"], published_scale="0-100",
            role="table", source_csv="published_baselines.csv",
            source=r.get("source", ""), same_model="",
            caveat=r.get("note", ""),
        )
        rec.update(paper_fields(method))
        rec.update(profile_fields(method, fallback_sup=rec.get("supervision", "")))
        if rec["verified_by"] == "unverified":
            rec["caveat"] = (rec["caveat"] + " | UNVERIFIED: "
                             + NO_LOCAL_PDF.get(method, "no local PDF for this method")).strip(" |")
        corr = CORRECTIONS.get((cell, method))
        if corr:
            rec["supervision"] = corr.get("supervision", rec["supervision"])
            rec["caveat"] = corr["caveat"] + " | orig note: " + rec["caveat"]
        rows.append(rec)

    # ── source B: scores_lsml_upcr.csv published_Y (0-1 scale, single anchor, uncited) ─
    seen = set()
    for r in read_csv(SCORES):
        cell = r["cell"]
        if cell not in INSCOPE or cell in seen:
            continue
        if cell in LAPEIG_GSM8K:
            # rebuilt below from the verified table; the stored anchor is mislabeled
            seen.add(cell)
            model, attn, lap = LAPEIG_GSM8K[cell]
            stored = float(r.get("published_Y") or 0)
            pf = paper_fields("LapEigvals")
            wrong_model = abs(stored - attn) > 1e-6 and abs(stored - lap) > 1e-6
            base = dict(cell=cell, dataset=r.get("dataset", ""), model=r.get("model", ""),
                        auroc_as_published="", published_scale="0-1",
                        source_csv="scores_lsml_upcr.csv (rebuilt Step 193)",
                        source=f"Binkowski et al., {PAPERS['LAPEIG']['table']}, {model} row",
                        same_model=r.get("head_to_head", ""), **pf)
            rows.append(dict(base, method="AttentionScore (LapEigvals paper)",
                             **{k: v for k, v in profile_fields(
                                 "AttentionScore (LapEigvals paper)").items()
                                if k != "supervision"},
                             supervision="unsupervised", auroc=attn,
                             auroc_as_published=f"{attn}", role="anchor",
                             caveat=("CORRECTED Step 193: stored as 'LapEigvals' = "
                                     f"{stored}. " +
                                     ("That value is Mistral-Small-24B's LapEigvals, a "
                                      "DIFFERENT model; this cell's own model is "
                                      f"{model}. " if wrong_model else
                                      "The value was right but the method name and "
                                      "supervision tag were wrong. ") +
                                     "AttentionScore is the paper's only unsupervised "
                                     "method and therefore the like-for-like comparator "
                                     "for our label-free detector.")))
            rows.append(dict(base, method="LapEigvals",
                             **{k: v for k, v in profile_fields("LapEigvals").items()
                                if k != "supervision"},
                             supervision="supervised",
                             auroc=lap, auroc_as_published=f"{lap}", role="table",
                             caveat=("SUPERVISED: a logistic-regression probe over Laplacian "
                                     "eigenvalues of attention maps (needs model internals + "
                                     "labels). Compare against our LR oracle, not against our "
                                     "label-free score.")))
            continue
        y, meth = r.get("published_Y"), r.get("Y_method")
        if not y or not meth:
            continue
        seen.add(cell)
        rec = dict(
            cell=cell, dataset=r.get("dataset", ""), model=r.get("model", ""),
            method=meth, supervision=ANCHOR_SUPERVISION.get(meth, ""),
            auroc=round(float(y), 4), auroc_as_published=y, published_scale="0-1",
            role="anchor", source_csv="scores_lsml_upcr.csv",
            source="", same_model=r.get("head_to_head", ""),
            caveat="",
        )
        base = meth.split(" (")[0].strip()
        rec.update(paper_fields(meth if meth in METHOD_PAPER else base))
        rec.update(profile_fields(meth, fallback_sup=rec.get("supervision", "")))
        if rec["verified_by"] == "unverified":
            rec["caveat"] = ("UNVERIFIED: " + NO_LOCAL_PDF.get(
                meth, NO_LOCAL_PDF.get(base, "no local PDF for this method")))
        if not rec["source"]:
            rec["source"] = ("NO CITATION in scores_lsml_upcr.csv — that file carries "
                             "published_Y/Y_method only")
        rows.append(rec)

    rows.sort(key=lambda r: (r["cell"], r["role"], r["method"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["cell", "dataset", "model", "method", "supervision", "access",
            "passes", "profile_src", "auroc",
            "auroc_as_published", "published_scale", "role", "same_model",
            "verified_by", "source_csv", "source", "paper_slug", "paper_table",
            "paper_columns", "paper_evidence", "caveat"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    covered = {r["cell"] for r in rows}
    anchors = [r for r in rows if r["role"] == "anchor"]
    ver = [r for r in rows if r["verified_by"] == "extracted"]
    print(f"wrote {len(rows)} competitor rows -> {OUT}")
    print(f"  cells covered        : {len(covered)}/{len(INSCOPE)}")
    print(f"  no competitor at all : {sorted(set(INSCOPE) - covered)}")
    print(f"  verified vs extracted: {len(ver)}/{len(rows)} rows")
    print(f"  anchors              : {len(anchors)} "
          f"({sum(1 for r in anchors if r['verified_by'] == 'extracted')} verified)")
    unver = [r for r in rows if r["verified_by"] == "unverified"]
    by_method = {}
    for r in unver:
        by_method.setdefault(r["method"], []).append(r["cell"])
    for m, cells in sorted(by_method.items()):
        why = NO_LOCAL_PDF.get(m, NO_LOCAL_PDF.get(m.split(" (")[0], "no local PDF"))
        print(f"  UNVERIFIED {m:<32} {len(cells):>2} rows — {why}")


if __name__ == "__main__":
    main()
