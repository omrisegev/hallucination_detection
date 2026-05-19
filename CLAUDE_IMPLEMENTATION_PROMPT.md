# Claude Implementation Prompt: Benchmarking SOTA

## Instructions for Claude
You are tasked with implementing the benchmarking phase for the Hallucination Detection project. You must strictly follow the plan in `plans/SOTA_Comparison_Benchmarking_Plan.md`.

### Step 1: Environment Setup
1. Create a `baselines/` directory.
2. Clone the following repositories:
   - `https://github.com/graphml-lab-pwr/lapeigvals` (LapEigvals)
   - `https://github.com/BarSGuy/LLM-Output-Signatures-Network` (LOS-Net)
3. Install required dependencies: `transformers`, `scipy`, `evaluate`, `torch-geometric`, and `spacy`.

### Step 2: Implement Official Semantic Entropy
The current `lite_semantic_entropy` in `baselines.py` is insufficient. You must:
1. Implement a version using `cross-encoder/nli-deberta-v3-base` or `large` for entailment-based clustering.
2. Follow the protocol from Farquhar et al. (Nature 2024).

### Step 3: Run Comparisons
For each domain (Math, Science, RAG), run the baselines against the generations stored in the project's `cache/` or re-generate using `spectral_utils`.

**Critical Requirement**: Ensure that for the **RAG** domain, you are evaluating **Statement-level Grounding** using the citation markers `[1]`, `[2]` as slicing boundaries, matching our method's logic.

### Step 4: Output
Produce a final `Research_Phase12_Comparison_Results.md` file containing:
1. A table comparing Our AUROC vs. Competitor AUROC for every cell.
2. A summary of "Compute Cost" (e.g., 1-pass vs 10-pass).
3. A verification that our LOS-Net recreation matches their paper's ~72.9% on HotpotQA.

### Grounding Context
- **Our Method**: Gray-box, Unsupervised, 1-pass (Nadler Fusion of Entropy Spectrum).
- **Target AUC to beat**: 70% (The thesis "Gate").
- **Core Signal**: Spectral features of $H(n)$ (Entropy trajectory).
