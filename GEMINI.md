# GEMINI.md — Project Instructions & Handoff

You are the Gemini CLI agent working on the MV_EPR thesis project (Spectral Hallucination Detection). This file is your foundational mandate.

---

## 0. Mandatory Session Protocol

1. **Read `PROGRESS.md`**: Current status and immediate next actions.
2. **Read `Phase10_Pilot_Plan.md`**: The active locked plan.
3. **Read `Research_Directions.md`**: Thesis roadmap and decision gates.
4. **Follow `CLAUDE.md` Conventions**: While `GEMINI.md` is your primary guide, `CLAUDE.md` contains shared engineering standards (e.g., Colab setup, model loading rules).

---

## 1. Project Identity & Scope

**Thesis**: Hallucination detection via spectral analysis of per-token entropy trajectories $H(n)$ from a single gray-box forward pass.
**Core Method**: Extract spectral/time-domain features (FFT, STFT, variance peaks) and fuse them via **Nadler covariance-weighted leading eigenvector**.
**Scope**: Highly effective on reasoning-heavy domains (Math, GPQA). Ineffective on short factual QA (TriviaQA). Phase 10 extends this to **long-document RAG and agentic loops**.

---

## 2. Technical Mandates

### The `spectral_utils` Rule
**NEVER inline helpers in notebooks.** All code must live in `spectral_utils/`. Update the package and commit *before* running a notebook.
- `feature_utils.py`: Spectral & time-domain extraction.
- `fusion_utils.py`: Nadler & statistical fusion.
- `model_utils.py`: HF model/tokenizer loading & inference.
- `data_loaders.py`: Benchmark-specific loaders and prompts.

### Nadler Fusion Invariants
- **>= 3 views** required.
- **Z-score normalization** must be applied *after* sign orientation.
- **Correlation Filter**: Skip subsets with Spearman $|\rho| \geq 0.75$.

---

## 3. Gemini CLI Toolset Optimization

### Deep Research
Use the `gemini-deep-research` skill for multi-step investigations. It coordinates `google_web_search` and `web_fetch` to synthesize findings.
- Trigger: `/skills activate gemini-deep-research` or ask the agent to "run deep research on X".

### Codebase Navigation
- Use `codebase_investigator` for structural analysis or understanding complex dependencies.
- Use `grep_search` with `include_pattern` to find usage patterns before refactoring.

### Colab Integration
You write the `.ipynb` files; the user runs them on A100.
- Always include the **Standard Cell 1** (Git clone + `spectral_utils` import) from `CLAUDE.md`.
- Use `io_utils.save_cache` frequently in inference loops.

---

## 4. Documentation Hierarchy

| File | Purpose |
|------|---------|
| `PROGRESS.md` | Session handoff. **Update at the end of every session.** |
| `HISTORY.md` | Append-only step log. **Add a new step for every logical milestone.** |
| `Research_Directions.md` | Roadmap & Decision Gates. Update when phases complete. |
| `Experiments_Report.md` | Aggregated results for advisors. |

---

## 5. Branching & Git
- Primary branch: `master`.
- Use `gemini/` prefix for your feature branches (e.g., `gemini/phase10-pilot`).
- Commit messages: Imperative mood, prefix with Step number if applicable.

---

## 6. Advisor Context
- **Bracha Laufer-Goldshtein**: Focus on conformal prediction, risk control, and formal bounds.
- **Ofir Lindenbaum**: Focus on spectral methods, kernel fusion, and multi-view learning.
- **Goal**: Unsupervised detection (test-time labels are not available).
