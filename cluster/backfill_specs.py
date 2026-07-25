"""
Backfill cell registry — which cells get the teacher-forced view backfill and how
their exact generation context (prompt + sampling warp) is reconstructed.

Pure data + light JSON resolvers — no torch / transformers imports — so both the
cluster driver (cluster/backfill_views.py) and local tooling can import it cheaply,
same convention as cluster/presets.py.

A spec answers, per cell:
  - where the raw pkls live relative to --data-root      (data_dir, pkl_glob)
  - which model produced them                            (model, via preset for repgrid)
  - how the exact input ids are rebuilt                  (prompt_recipe)
  - which sampling warp the saved post-warp keys used    (warp per temperature)
  - what the pkl schema is                               (repgrid | flat)

Repgrid cells need one line each: everything derives from cluster/presets.py plus the
cell's own manifest.json (written at submission time). Rule when they disagree:
manifest wins for values that were actually run (temps, gen_top_k, gen_top_p,
logprob_top_k, prompt_suffix); preset wins for fields the manifest never carried
(raw_prompt).

Colab-era entries are authored from results/coverage_audit.csv (the Drive audit)
in Phase 3 of the full-coverage plan — see HANDOFF_full_coverage.md.

prompt_recipe kinds (priority order — a recipe may be a best guess; Gate B in
backfill_views.py proves it right or aborts the cell without writing):
  {"kind": "stored_ids",  "key": "prompt_token_ids"}
  {"kind": "stored_text", "key": "prompt", "add_special_tokens": True}
  {"kind": "dataset_fn",  "dataset": "<run_inference.DATASETS key>",
   "prompt_suffix": "", "raw_prompt": False}
  {"kind": "template",    "template": "...{question}...", "raw_prompt": False}
  {"kind": "irreconstructible"}   # tier-3: skipped + reported, decision escalated
"""
import glob
import json
import os
import re
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from presets import get_preset  # noqa: E402

# Raw top-K logprobs are capped at 50 regardless of the preset's logprob_top_k:
# losnet's post-warp capture used K=1000 (TDS wants it), but every Z_n/lp/xlp view
# only needs top-50, and K=1000 raw would add ~800 MB to that one cell.
RAW_TOPK_CAP = 50

# The 19 replication-grid analysis cells (matches build_repgrid_featcache discover()
# minus the _reject/_partial/_pilot archives). The backfill driver decides per
# candidate what is missing — listing a complete cell here is harmless (no-op), and
# the 7 already-complete cells are exactly the Gate-A validation set.
REPGRID_ANALYSIS_CELLS = [
    "ars_gsm8k_r1distill8b",
    "epr_triviaqa_mistral24b",
    "inside_coqa_llama7b",
    "internalstates_gsm8k_qwen25_7b",
    "lapeigvals_gsm8k_llama3b",
    "lapeigvals_gsm8k_llama8b",
    "lapeigvals_gsm8k_mistral24b",
    "lapeigvals_gsm8k_nemo",
    "lapeigvals_gsm8k_phi35",
    "losnet_hotpotqa_mistral7b",
    "noise_gsm8k_mistral7b",
    "noise_gsm8k_phi3mini",
    "sciq_llama8b",
    "se_nq_open_llama8b",
    "se_squad_v2_llama8b",
    "seiclr_triviaqa_opt30b",
    "semenergy_triviaqa_qwen3_8b",
    "spilled_triviaqa_llama8b",
    "truthfulqa_llama8b",
]

# Cluster-side layout verified 2026-07-18: cells live under $SHARED/results/repgrid/
# (locally they are fetched into cache/repgrid/ — pass --data-root accordingly).
BACKFILL_SPECS = {
    cid: {"origin": "repgrid", "preset_id": cid, "data_dir": f"results/repgrid/{cid}"}
    for cid in REPGRID_ANALYSIS_CELLS
}

# Per-cell warp corrections discovered by `--probe-warp` (a run whose saved traces
# carry a warp component the manifest never recorded — e.g. a model
# generation_config default that applied at generation time). Filled empirically;
# never guessed.
REPGRID_WARP_OVERRIDES = {}
# ── Colab-era cells (Phase 3, authored 2026-07-18 from coverage_audit.csv +
# schema_dump.json + notebook archaeology — see cluster/backfill_runbook.md).
# Data pushed to $SHARED/data/colab/<original dir name>/ via push_data.sh.
#
# ⚠ gpqa T-mislabel (fingerprint-verified, labels+trace-lengths match 1.000):
# ALL four small-model gpqa sweep cells labeled T1.0 actually hold the phase4
# T=1.5 generations (phase5's real T1.0 runs are different data). Warps below use
# the TRUE temperature; the cell-key relabel propagates at unified-rebuild time.

_LAPEI_GSM8K_TEMPLATE = (
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: {question}\n"
    'Your response should end with "The final answer is [answer]" '
    "where [answer] is the response to the problem."
)
_PHASE9_COT_TEMPLATE = (
    "Answer the following question. Think through your reasoning step by step, "
    "then state your final answer on its own line starting with 'Answer:'.\n\n"
    "Question: {question}\n\n"
    "Let me think step by step:"
)


def _colab(data_dir, pkl_glob, model, temp, recipe, cell_key, schema="flat",
           top_k=50, roundtrip=True):
    return {"origin": "colab", "data_dir": data_dir, "pkl_glob": pkl_glob,
            "schema": schema, "model": model,
            "warp": {"temperature": temp, "top_k": top_k, "top_p": None},
            "prompt_recipe": recipe, "target_cell_key": cell_key,
            "allow_roundtrip": roundtrip}


_P4 = "data/colab/epr_spectral_phase4"
_MATH_RECIPE = {"kind": "dataset_by_idx", "loader": "math500_300"}
_GPQA_RECIPE = {"kind": "dataset_by_idx", "loader": "gpqa_diamond"}

# math500 ×4 — phase4, T=1.5 (cells renamed *_T1.5 in Step 184)
for _dir, _model, _key in [
    ("Qwen2.5-Math-1.5B-Instruct__math500", "Qwen/Qwen2.5-Math-1.5B-Instruct",
     "Qwen2.5-Math-1.5B-Instruct_T1.5"),
    ("Qwen2.5-Math-7B-Instruct__math500", "Qwen/Qwen2.5-Math-7B-Instruct",
     "Qwen-Math-7B_T1.5"),
    ("deepseek-math-7b-instruct__math500", "deepseek-ai/deepseek-math-7b-instruct",
     "deepseek-math-7b-instruct_T1.5"),
    ("DeepSeek-R1-Distill-Llama-8B__math500", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
     "DeepSeek-R1-Distill-Llama-8B_T1.5"),
]:
    BACKFILL_SPECS[f"c_math500_{_dir.split('__')[0]}"] = _colab(
        f"{_P4}/{_dir}", "inference_cache.pkl", _model, 1.5, _MATH_RECIPE,
        ("math500", _key))

# gpqa ×4 small models — phase4, TRUE T=1.5 (mislabeled T1.0 in the sweep pool)
for _dir, _model, _key in [
    ("Llama-3.1-8B-Instruct__gpqa", "meta-llama/Llama-3.1-8B-Instruct", "Llama-8B_T1.0"),
    ("Mistral-7B-Instruct-v0.2__gpqa", "mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B_T1.0"),
    ("Qwen2.5-7B-Instruct__gpqa", "Qwen/Qwen2.5-7B-Instruct", "Qwen-7B_T1.0"),
    ("DeepSeek-R1-Distill-Llama-8B__gpqa", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
     "DeepSeek-R1-Distill-Llama-8B_T1.0"),
]:
    BACKFILL_SPECS[f"c_gpqa_{_dir.split('__')[0]}"] = _colab(
        f"{_P4}/{_dir}", "inference_cache.pkl", _model, 1.5, _GPQA_RECIPE,
        ("gpqa", _key))

# gpqa 72B AWQ — genuine T=1.0 (GPQA_Phase8_Fixed.ipynb: TEMP=1.0, MAX_NEW=1024)
BACKFILL_SPECS["c_gpqa_qwen72b_awq"] = _colab(
    "data/colab/epr_spectral_gpqa_72b/Qwen2.5-72B-Instruct-AWQ__gpqa_T1.0",
    "inference_cache.pkl", "Qwen/Qwen2.5-72B-Instruct-AWQ", 1.0, _GPQA_RECIPE,
    ("gpqa", "Qwen2.5-72B-Instruct-AWQ_T1.0"))

# gsm8k Llama-8B — LapEigvals Listing-5 verbatim template on the saved question
BACKFILL_SPECS["c_gsm8k_llama8b"] = _colab(
    "data/colab/epr_spectral_gsm8k_vs_lapei/Llama-3.1-8B-Instruct__gsm8k_T1.0",
    "inference_cache.pkl", "meta-llama/Llama-3.1-8B-Instruct", 1.0,
    {"kind": "template_question", "template": _LAPEI_GSM8K_TEMPLATE},
    ("gsm8k", "Llama-8B_T1.0"))

# qa ×4 — phase9, Falcon3-10B-Instruct, T=1.0 (list schema, question in item)
_P9 = "data/colab/spectral_phase9_cache"
for _pkl, _recipe, _key in [
    ("trivia_qa_cot_traces.pkl",
     {"kind": "template_question", "template": _PHASE9_COT_TEMPLATE},
     "spectral_phase9_cache_trivia_qa_cot_traces_T1.0"),
    ("webq_cot_traces.pkl",
     {"kind": "template_question", "template": _PHASE9_COT_TEMPLATE},
     "spectral_phase9_cache_webq_cot_traces_T1.0"),
    ("trivia_qa_traces.pkl",
     {"kind": "package_prompt", "fn": "trivia_qa_prompt"},
     "spectral_phase9_cache_trivia_qa_traces_T1.0"),
    ("webq_traces.pkl",
     {"kind": "package_prompt", "fn": "webq_prompt"},
     "spectral_phase9_cache_webq_traces_T1.0"),
]:
    BACKFILL_SPECS[f"c_qa_{_pkl.replace('.pkl','')}"] = _colab(
        _P9, _pkl, "tiiuae/Falcon3-10B-Instruct", 1.0, _recipe,
        ("qa", _key), schema="list")

# rag ×16 — phase10_main, T=1.0, prompt = lciteeval_prompt(saved row)
_RAG_MODELS = {
    "qwen7b": ("Qwen/Qwen2.5-7B-Instruct", "Qwen-7B"),
    "mistral24b": ("mistralai/Mistral-Small-24B-Instruct-2501", "Mistral-24B"),
    "qwen72b": ("Qwen/Qwen2.5-72B-Instruct-AWQ", "Qwen-72B"),
    "llama8b": ("meta-llama/Llama-3.1-8B-Instruct", "Llama-8B"),
}
_RAG_DATASETS = {"hotpotqa": "hotpotqa", "natural_questions": "natural-questions",
                 "2wikimultihopqa": "2wikimultihopqa", "narrativeqa": "narrativeqa"}
for _mk, (_model, _short) in _RAG_MODELS.items():
    for _ds, _dskey in _RAG_DATASETS.items():
        BACKFILL_SPECS[f"c_rag_{_mk}_{_ds}"] = _colab(
            "data/colab/phase10_main/raw", f"{_mk}__{_ds}.pkl", _model, 1.0,
            {"kind": "lciteeval"}, ("rag", f"{_short}_{_dskey}"), schema="phase10")

# Probe-proven warp corrections (job 123716, 2026-07-19): these runs inherited
# model generation_config defaults the code never passed explicitly.
BACKFILL_SPECS["c_gsm8k_llama8b"]["warp"]["top_p"] = 0.9            # Llama-3.1 default
BACKFILL_SPECS["c_gpqa_Qwen2.5-7B-Instruct"]["warp"].update(
    top_p=0.8, rep_penalty=1.05)                                     # Qwen2.5 defaults

# fp16 hypothesis (probe-fail follow-up, 2026-07-19): the phase4/5 notebooks loaded
# models with torch_dtype=torch.float16 while the backfill defaults to bf16 —
# the leading explanation for the remaining probe failures/near-misses in those
# families (deepseek-math 0.733, R1-Distill 0.54–0.68, gpqa Mistral 0.905 /
# Llama 0.869 vs the 0.90 bar). Falcon phase9 cells (0.86–0.90) are included in
# the fp16 re-probe to settle their dtype empirically even though the package was
# already bf16 by the phase9 commit. QwenMath math500 + gpqa Qwen-7B passed at
# bf16 and are deliberately left untouched.
for _cid in [
    "c_math500_deepseek-math-7b-instruct",
    "c_math500_DeepSeek-R1-Distill-Llama-8B",
    "c_gpqa_Llama-3.1-8B-Instruct",
    "c_gpqa_Mistral-7B-Instruct-v0.2",
    "c_gpqa_DeepSeek-R1-Distill-Llama-8B",
    "c_qa_trivia_qa_cot_traces",
    "c_qa_webq_cot_traces",
    "c_qa_trivia_qa_traces",
    "c_qa_webq_traces",
]:
    BACKFILL_SPECS[_cid]["dtype"] = "float16"

# phase15 ×9 — canonical rich schema incl. gen_token_ids (tier-2 clean),
# Qwen2.5-Math-7B, package math_prompt on the saved question (identical string)
_MATH_TEMPLATE = ("Solve the following competition math problem. "
                  "Show all your work step by step, then give your final answer "
                  "in \\boxed{{}}.\n\n{question}")
for _t, _r in [(0.3, 0), (0.6, 0), (1.0, 0), (1.0, 1), (1.0, 2), (1.0, 3), (1.0, 4),
               (1.5, 0), (2.0, 0)]:
    BACKFILL_SPECS[f"c_p15_T{_t}_run{_r}"] = _colab(
        "data/colab/phase15_temperature", f"math500_qwen7b_T{_t}_run{_r}.pkl",
        "Qwen/Qwen2.5-Math-7B-Instruct", _t,
        {"kind": "template_question", "template": _MATH_TEMPLATE},
        ("phase15", f"math500_qwen7b_T{_t}_run{_r}"), roundtrip=False)


_TEMP_RE = re.compile(r"_T([0-9.]+)\.pkl$")


def _pkl_temp(path):
    m = _TEMP_RE.search(os.path.basename(path))
    return float(m.group(1)) if m else None


def resolve_spec(cell_id, data_root):
    """Resolve a BACKFILL_SPECS entry into a flat namespace the driver consumes.

    Returns SimpleNamespace with:
      cell_id, origin, model, schema, dataset, logprob_top_k (post-warp K),
      raw_top_k (raw K, capped), prompt_recipe, pkls=[(temp, path)],
      warp_base={"top_k","top_p"} (temperature comes per pkl),
      repetition_penalty, no_repeat_ngram_size (must be None for post-warp keys),
      data_dir (absolute)
    """
    if cell_id not in BACKFILL_SPECS:
        raise KeyError(f"unknown backfill cell {cell_id!r}; known: {sorted(BACKFILL_SPECS)}")
    raw = BACKFILL_SPECS[cell_id]
    data_dir = os.path.join(data_root, raw["data_dir"])
    pkl_glob = raw.get("pkl_glob", "raw_*.pkl")
    pkls = sorted(glob.glob(os.path.join(data_dir, pkl_glob)))
    pkls = [p for p in pkls if not p.endswith(".tmp")]

    if raw["origin"] == "repgrid":
        preset = get_preset(raw["preset_id"])
        man_path = os.path.join(data_dir, "manifest.json")
        man = json.load(open(man_path)) if os.path.exists(man_path) else {}

        def mget(key, default=None):
            # manifest wins for actually-run values; preset is the fallback
            v = man.get(key)
            return v if v is not None else preset.get(key, default)

        temps = mget("temps", [1.0])
        pkl_list = []
        for p in pkls:
            t = _pkl_temp(p)
            if t is None:
                continue
            if not any(abs(t - mt) < 1e-9 for mt in temps):
                raise ValueError(f"{cell_id}: pkl temp {t} not in manifest temps {temps} ({p})")
            pkl_list.append((t, p))
        if len(pkl_list) != len(temps):
            missing = [t for t in temps if not any(abs(t - pt) < 1e-9 for pt, _ in pkl_list)]
            raise FileNotFoundError(f"{cell_id}: missing pkls for temps {missing} under {data_dir}")

        lp_topk = int(mget("logprob_top_k", 50) or 50)
        return SimpleNamespace(
            cell_id=cell_id,
            origin="repgrid",
            model=mget("model"),
            schema="repgrid",
            dataset=mget("dataset"),
            logprob_top_k=lp_topk,
            raw_top_k=min(lp_topk, RAW_TOPK_CAP),
            prompt_recipe={
                "kind": "dataset_fn",
                "dataset": mget("dataset"),
                "prompt_suffix": mget("prompt_suffix", "") or "",
                # manifest never carried raw_prompt — preset is authoritative
                "raw_prompt": bool(preset.get("raw_prompt", False)),
            },
            pkls=pkl_list,
            warp_base={"top_k": mget("gen_top_k", 50), "top_p": mget("gen_top_p"),
                       "rep_penalty": None,
                       **REPGRID_WARP_OVERRIDES.get(cell_id, {})},
            repetition_penalty=preset.get("repetition_penalty"),
            no_repeat_ngram_size=preset.get("no_repeat_ngram_size"),
            data_dir=data_dir,
            allow_roundtrip=False,  # every repgrid cell stores gen_token_ids
            dtype="bfloat16",       # every cluster preset generated in bf16
        )

    # colab / trace origins: everything explicit in the spec
    if not pkls:
        raise FileNotFoundError(f"{cell_id}: no pkls matching {pkl_glob} under {data_dir}")
    warp = raw.get("warp", {})
    lp_topk = int(raw.get("logprob_top_k", 50))
    return SimpleNamespace(
        cell_id=cell_id,
        origin=raw["origin"],
        model=raw["model"],
        schema=raw.get("schema", "flat"),
        dataset=raw.get("dataset"),
        logprob_top_k=lp_topk,
        raw_top_k=min(lp_topk, RAW_TOPK_CAP),
        prompt_recipe=raw["prompt_recipe"],
        pkls=[(warp.get("temperature", 1.0), p) for p in pkls],
        warp_base={"top_k": warp.get("top_k", 50), "top_p": warp.get("top_p"),
                   "rep_penalty": warp.get("rep_penalty")},
        repetition_penalty=raw.get("repetition_penalty"),
        no_repeat_ngram_size=raw.get("no_repeat_ngram_size"),
        data_dir=data_dir,
        allow_roundtrip=bool(raw.get("allow_roundtrip", False)),
        target_cell_key=raw.get("target_cell_key"),
        dtype=raw.get("dtype", "bfloat16"),
    )


def list_backfill_cells():
    return sorted(BACKFILL_SPECS)
