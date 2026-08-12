#!/usr/bin/env python3
"""Frozen, phase-separated white-box layer-fusion benchmark.

``prepare`` is the only phase that joins the large raw caches and layer
sidecars.  It writes label-free feature bundles.  ``fit`` is a separate process
that can only open those bundles and freezes label-free scores.  ``evaluate``
verifies every score hash before it opens correctness labels.  ``all`` launches
those phases as three subprocesses and then renders the portable HTML report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import re
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.whitebox_layer_fusion import (  # noqa: E402
    ALL_LAYERS,
    LATE_LAYERS,
    SPACED_LAYERS,
    FeatureMatrix,
    all_layers,
    assert_no_label_fitting_signatures,
    extract_dola_kl_proxy,
    extract_haloscope_projection,
    extract_lens96,
    extract_lens_grid,
    extract_resid_core,
    extract_trilens_entropy,
    fit_controls,
    fit_core_spectral,
    fit_dependency_methods,
    fit_hierarchical,
    fit_haloscope_direct_proxy,
    late_layers,
    load_evaluation_labels,
    residualize_token_length,
    spaced_layers,
    validate_and_join,
)
from spectral_utils.paper_benchmark_suite import standardize as canonical_standardize  # noqa: E402


VERSION = "whitebox-layer-fusion-v2-2026-08-12"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 20260812
BOOTSTRAP_DRAWS = 2000
TIE_TOLERANCE = 0.001
DEFAULT_CACHE = REPO / "dataset_cache" / "whitebox_layer_fusion_v1"
DEFAULT_RESULTS = REPO / "results" / "whitebox_layer_fusion_v2"


CELLS = OrderedDict(
    (
        (
            "gsm8k_t1.0",
            {
                "cell_id": "lapeigvals_gsm8k_llama8b",
                "dataset": "GSM8K",
                "temperature": 1.0,
                "raw": "repgrid/lapeigvals_gsm8k_llama8b/raw_gsm8k_T1.0.pkl",
                "sidecar": "layer_views/lapeigvals_gsm8k_llama8b/layer_views_T1.0.pkl",
                "backfill": "repgrid/lapeigvals_gsm8k_llama8b/backfill_report.json",
                "remote_raw_modtime": "2026-07-18T19:11:57Z",
                "remote_sidecar_modtime": "2026-08-11T22:53:59Z",
                "raw_size": 111563114,
                "raw_sha256": "6ec52c7af5306a48464ab58f96d5b3f31029a064ccfc2943c049275b9383aa88",
                "sidecar_size": 116845519,
                "sidecar_sha256": "58ed0566585b804dbb1b2d42ff91580b42bef52ae15f1079e1c43f0089dfd2ad",
                "source_rows": 500,
            },
        ),
        (
            "triviaqa_t1.0",
            {
                "cell_id": "spilled_triviaqa_llama8b",
                "dataset": "TriviaQA",
                "temperature": 1.0,
                "raw": "repgrid/spilled_triviaqa_llama8b/raw_trivia_qa_T1.0.pkl",
                "sidecar": "layer_views/spilled_triviaqa_llama8b/layer_views_T1.0.pkl",
                "backfill": "repgrid/spilled_triviaqa_llama8b/backfill_report.json",
                "remote_raw_modtime": "2026-07-08T07:07:12Z",
                "remote_sidecar_modtime": "2026-08-11T22:54:10Z",
                "raw_size": 7808360,
                "raw_sha256": "cf01350f5bc141908e3f0563c1bc3037148fbad3a30c4eb05c63cd3c13a51e65",
                "sidecar_size": 16258124,
                "sidecar_sha256": "3e44cfac86c2a763ebc7da26a5bf0027950c8373e8c9aa91f06664e909c984f3",
                "source_rows": 500,
            },
        ),
        (
            "sciq_t1.0",
            {
                "cell_id": "sciq_llama8b",
                "dataset": "SciQ",
                "temperature": 1.0,
                "raw": "repgrid/sciq_llama8b/raw_sciq_T1.0.pkl",
                "sidecar": "layer_views/sciq_llama8b/layer_views_T1.0.pkl",
                "backfill": "repgrid/sciq_llama8b/backfill_report.json",
                "remote_raw_modtime": "2026-07-18T19:11:58Z",
                "remote_sidecar_modtime": "2026-08-11T23:08:26Z",
                "raw_size": 13040824,
                "raw_sha256": "f0085556f208c64a85d8085734f9423cc391531603b6a4c1cd20443bd25cc1b1",
                "sidecar_size": 29765474,
                "sidecar_sha256": "9ef8a1eac605477f22cc8a7da61e035ce242a30774f0b8d33e0ec4a455d62961",
                "source_rows": 1000,
            },
        ),
        (
            "truthfulqa_t0.5",
            {
                "cell_id": "truthfulqa_llama8b",
                "dataset": "TruthfulQA",
                "temperature": 0.5,
                "raw": "repgrid/truthfulqa_llama8b/raw_truthfulqa_T0.5.pkl",
                "sidecar": "layer_views/truthfulqa_llama8b/layer_views_T0.5.pkl",
                "backfill": "repgrid/truthfulqa_llama8b/backfill_report.json",
                "remote_raw_modtime": "2026-07-11T07:53:25Z",
                "remote_sidecar_modtime": "2026-08-11T23:17:21Z",
                "raw_size": 435424434,
                "raw_sha256": "3dac66f177aa825ccce53849b4e4d8cfc1e43f6c93ad664a4c01851fc90e3077",
                "sidecar_size": 572718920,
                "sidecar_sha256": "d7a1f5482142a5472c26007f2c2e6f78a1ff40e9cab4b8f061402a15be7343cb",
                "source_rows": 8170,
            },
        ),
        (
            "squadv2_t0.5",
            {
                "cell_id": "se_squad_v2_llama8b",
                "dataset": "SQuADv2",
                "temperature": 0.5,
                "raw": "repgrid/se_squad_v2_llama8b/raw_squad_v2_T0.5.pkl",
                "sidecar": "layer_views/se_squad_v2_llama8b/layer_views_T0.5.pkl",
                "backfill": "repgrid/se_squad_v2_llama8b/backfill_report.json",
                "remote_raw_modtime": "2026-07-08T11:50:14Z",
                "remote_sidecar_modtime": "2026-08-11T22:59:23Z",
                "raw_size": 76891110,
                "raw_sha256": "d3e457da44ba79258fc4319bfaf8dfb068954d2a26d2e5d88b2da35227cca088",
                "sidecar_size": 249953603,
                "sidecar_sha256": "e175bf263f07a400454e90190c3f7e6cab3e7f8508280714fedb8d221cf995c3",
                "source_rows": 10000,
            },
        ),
        (
            "nq_open_t0.5",
            {
                "cell_id": "se_nq_open_llama8b",
                "dataset": "NQ-Open",
                "temperature": 0.5,
                "raw": "repgrid/se_nq_open_llama8b/raw_nq_open_T0.5.pkl",
                "sidecar": "layer_views/se_nq_open_llama8b/layer_views_T0.5.pkl",
                "backfill": "repgrid/se_nq_open_llama8b/backfill_report.json",
                "remote_raw_modtime": "2026-07-11T17:58:35Z",
                "remote_sidecar_modtime": "2026-08-11T23:07:49Z",
                "raw_size": 248018947,
                "raw_sha256": "2ac8a95ad66f20c914fcfd1cd53740d6ca4ac631555362c8cf1c11aacf1a4548",
                "sidecar_size": 419916717,
                "sidecar_sha256": "3e62d9aae0760a46b6a12a92c5eaa819e3cda368ed5133f22b3ac1921cb934c0",
                "source_rows": 10000,
            },
        ),
    )
)

CELLS.update((
    ("gsm8k_r1distill_t0.0", {
        "cell_id": "ars_gsm8k_r1distill8b", "dataset": "GSM8K", "temperature": 0.0,
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "model_family": "DeepSeek-R1-Distill-Llama-8B",
        "n_layers": 32, "hidden_size": 4096,
        "raw": "repgrid/ars_gsm8k_r1distill8b/raw_gsm8k_T0.0.pkl",
        "sidecar": "layer_views/ars_gsm8k_r1distill8b/layer_views_T0.0.pkl",
        "backfill": "repgrid/ars_gsm8k_r1distill8b/backfill_report.json",
        "remote_raw_modtime": "2026-07-18T19:11:06Z", "remote_sidecar_modtime": "2026-08-12T07:46:11Z",
        "raw_size": 189353848, "raw_sha256": "ae33ac6139828c1a69fb8887b950cc956323707d537d10c20bebf1108d8f2dc8",
        "sidecar_size": 195444850, "sidecar_sha256": "0014c54b6fd9883b1970a58b4cc2f364cf4882ad57fbec005f3c7366cb78fbcc",
        "source_rows": 500,
    }),
    ("coqa_llama7b_t0.5", {
        "cell_id": "inside_coqa_llama7b", "dataset": "CoQA", "temperature": 0.5,
        "model": "huggyllama/llama-7b", "model_family": "Llama-1-7B",
        "n_layers": 32, "hidden_size": 4096, "protocol_rejected": True,
        "protocol_rejection_reason": "Step 216 generation/chat-template defect; appendix-only",
        "raw": "repgrid/inside_coqa_llama7b/raw_coqa_T0.5.pkl",
        "sidecar": "layer_views/inside_coqa_llama7b/layer_views_T0.5.pkl",
        "backfill": "repgrid/inside_coqa_llama7b/backfill_report.json",
        "remote_raw_modtime": "2026-07-18T19:11:45Z", "remote_sidecar_modtime": "2026-08-12T07:23:40Z",
        "raw_size": 904373854, "raw_sha256": "d45f11024ac968221eff315542153f9c7bf37f7de8bb1150f50b02462825e9ee",
        "sidecar_size": 941432203, "sidecar_sha256": "728195fb451743d7cfe5f022a591565657284528caa3f5ea33293ba35cb37aa9",
        "source_rows": 5000,
    }),
    ("gsm8k_mistral24b_t1.0", {
        "cell_id": "lapeigvals_gsm8k_mistral24b", "dataset": "GSM8K", "temperature": 1.0,
        "model": "mistralai/Mistral-Small-24B-Instruct-2501", "model_family": "Mistral-Small-24B",
        "n_layers": 40, "hidden_size": 5120,
        "raw": "repgrid/lapeigvals_gsm8k_mistral24b/raw_gsm8k_T1.0.pkl",
        "sidecar": "layer_views/lapeigvals_gsm8k_mistral24b/layer_views_T1.0.pkl",
        "backfill": "repgrid/lapeigvals_gsm8k_mistral24b/backfill_report.json",
        "remote_raw_modtime": "2026-07-18T19:13:55Z", "remote_sidecar_modtime": "2026-08-12T08:41:41Z",
        "raw_size": 285037844, "raw_sha256": "881dfabbbe48a2af4d756483c6800f6d1674d04b13a1664b9765f52e7a36f23c",
        "sidecar_size": 380415258, "sidecar_sha256": "d6930ef881019b21b3b893faee990fb75d933ff5741814c573a9a6048c3f89d3",
        "source_rows": 1319,
    }),
    ("gsm8k_nemo_t1.0", {
        "cell_id": "lapeigvals_gsm8k_nemo", "dataset": "GSM8K", "temperature": 1.0,
        "model": "mistralai/Mistral-Nemo-Instruct-2407", "model_family": "Mistral-Nemo-12B",
        "n_layers": 40, "hidden_size": 5120,
        "raw": "repgrid/lapeigvals_gsm8k_nemo/raw_gsm8k_T1.0.pkl",
        "sidecar": "layer_views/lapeigvals_gsm8k_nemo/layer_views_T1.0.pkl",
        "backfill": "repgrid/lapeigvals_gsm8k_nemo/backfill_report.json",
        "remote_raw_modtime": "2026-07-18T19:13:26Z", "remote_sidecar_modtime": "2026-08-12T08:03:40Z",
        "raw_size": 302055723, "raw_sha256": "4b772b6a47d5b070511b863aebbec7fdbf25ab5b58a2f4d13de18805c133eff0",
        "sidecar_size": 400270368, "sidecar_sha256": "6fa95084e6f8695b50ada1da87d5ae0f5ccbecb25b506c4ca2cde037c19d53dd",
        "source_rows": 1319,
    }),
    ("gsm8k_phi35_t1.0", {
        "cell_id": "lapeigvals_gsm8k_phi35", "dataset": "GSM8K", "temperature": 1.0,
        "model": "microsoft/Phi-3.5-mini-instruct", "model_family": "Phi-3.5-mini",
        "n_layers": 32, "hidden_size": 3072,
        "raw": "repgrid/lapeigvals_gsm8k_phi35/raw_gsm8k_T1.0.pkl",
        "sidecar": "layer_views/lapeigvals_gsm8k_phi35/layer_views_T1.0.pkl",
        "backfill": "repgrid/lapeigvals_gsm8k_phi35/backfill_report.json",
        "remote_raw_modtime": "2026-07-18T19:12:28Z", "remote_sidecar_modtime": "2026-08-12T08:02:25Z",
        "raw_size": 344193497, "raw_sha256": "0cb4b31f13fb59f34f786db1d931f1d99daee71a0cfa10402e523a859477dde8",
        "sidecar_size": 361005285, "sidecar_sha256": "0b4107403f07d08306a5e0f858628cfb8b86a7beca88aaab26c26e793f94d755",
        "source_rows": 1319,
    }),
    ("gsm8k_mistral7b_t1.0", {
        "cell_id": "noise_gsm8k_mistral7b", "dataset": "GSM8K", "temperature": 1.0,
        "model": "mistralai/Mistral-7B-Instruct-v0.3", "model_family": "Mistral-7B-v0.3",
        "n_layers": 32, "hidden_size": 4096,
        "raw": "repgrid/noise_gsm8k_mistral7b/raw_gsm8k_T1.0.pkl",
        "sidecar": "layer_views/noise_gsm8k_mistral7b/layer_views_T1.0.pkl",
        "backfill": "repgrid/noise_gsm8k_mistral7b/backfill_report.json",
        "remote_raw_modtime": "2026-07-18T19:13:06Z", "remote_sidecar_modtime": "2026-08-12T08:11:37Z",
        "raw_size": 373510793, "raw_sha256": "9c80391981959c8196f0816db0caca818e4828a5cfbcd31bafb4e9271af93ccf",
        "sidecar_size": 389571637, "sidecar_sha256": "1de6787d236238b02e8c0afd9e7ee87cfa0a185736076f9147057bc9822f200a",
        "source_rows": 1319,
    }),
    ("gsm8k_phi3mini_t1.0", {
        "cell_id": "noise_gsm8k_phi3mini", "dataset": "GSM8K", "temperature": 1.0,
        "model": "microsoft/Phi-3-mini-4k-instruct", "model_family": "Phi-3-mini",
        "n_layers": 32, "hidden_size": 3072,
        "raw": "repgrid/noise_gsm8k_phi3mini/raw_gsm8k_T1.0.pkl",
        "sidecar": "layer_views/noise_gsm8k_phi3mini/layer_views_T1.0.pkl",
        "backfill": "repgrid/noise_gsm8k_phi3mini/backfill_report.json",
        "remote_raw_modtime": "2026-07-18T19:12:13Z", "remote_sidecar_modtime": "2026-08-12T07:53:32Z",
        "raw_size": 297124222, "raw_sha256": "10ccb627b29021b9f20757b500285404215097c858cfcea0b2e20c89f8fae5d5",
        "sidecar_size": 315628464, "sidecar_sha256": "3792f3a1e10a18491091e43398abbbdd0850439c35a842f9bc54bda0dd1ed7f0",
        "source_rows": 1319,
    }),
    ("triviaqa_qwen3_t0.6", {
        "cell_id": "semenergy_triviaqa_qwen3_8b", "dataset": "TriviaQA", "temperature": 0.6,
        "model": "Qwen/Qwen3-8B", "model_family": "Qwen3-8B",
        "n_layers": 36, "hidden_size": 4096,
        "raw": "repgrid/semenergy_triviaqa_qwen3_8b/raw_trivia_qa_T0.6.pkl",
        "sidecar": "layer_views/semenergy_triviaqa_qwen3_8b/layer_views_T0.6.pkl",
        "backfill": "repgrid/semenergy_triviaqa_qwen3_8b/backfill_report.json",
        "remote_raw_modtime": "2026-07-09T07:06:47Z", "remote_sidecar_modtime": "2026-08-12T06:53:41Z",
        "raw_size": 62044585, "raw_sha256": "fa33eb050e481dea9afc82ad82cb243c415ae8ee51b009bb9980b09b621d1945",
        "sidecar_size": 166840079, "sidecar_sha256": "71592c2f46b00efcb2872db0aab9b9cf8fc4d6ce704fcf6d94cd00831dd352eb",
        "source_rows": 5000,
    }),
))

EXPECTED_EXCLUSIONS = {
    "gsm8k_t1.0": {
        "20:0": (2048, 1024),
        "359:0": (2048, 1024),
        "423:0": (1185, 1024),
        "450:0": (1216, 1024),
    },
    "gsm8k_r1distill_t0.0": {"214:0": (1406, 1024)},
    "gsm8k_nemo_t1.0": {"500:0": (1147, 1024), "931:0": (1915, 1024)},
    "gsm8k_phi35_t1.0": {"58:0": (1272, 1024), "62:0": (1538, 1024), "552:0": (2048, 1024), "1075:0": (1637, 1024)},
    "gsm8k_mistral7b_t1.0": {
        "122:0": (1270, 1024), "150:0": (1269, 1024), "162:0": (1196, 1024),
        "199:0": (1120, 1024), "203:0": (2048, 1024), "219:0": (1077, 1024),
        "475:0": (1253, 1024), "850:0": (1087, 1024), "871:0": (1180, 1024),
        "911:0": (1121, 1024), "963:0": (1378, 1024), "1147:0": (1128, 1024),
    },
    "gsm8k_phi3mini_t1.0": {"12:0": (1168, 1024), "611:0": (1552, 1024), "950:0": (1104, 1024), "1169:0": (1319, 1024)},
}
EXPECTED_COHORTS = {
    "gsm8k_t1.0": {"source_groups": 500, "valid_groups": 496, "candidates_per_source_group": 1},
    "triviaqa_t1.0": {"source_groups": 500, "valid_groups": 500, "candidates_per_source_group": 1},
    "sciq_t1.0": {"source_groups": 1000, "valid_groups": 1000, "candidates_per_source_group": 1},
    "truthfulqa_t0.5": {"source_groups": 817, "valid_groups": 817, "candidates_per_source_group": 10},
    "squadv2_t0.5": {"source_groups": 1000, "valid_groups": 1000, "candidates_per_source_group": 10},
    "nq_open_t0.5": {"source_groups": 1000, "valid_groups": 1000, "candidates_per_source_group": 10},
    "gsm8k_r1distill_t0.0": {"source_groups": 500, "valid_groups": 499, "candidates_per_source_group": 1},
    "coqa_llama7b_t0.5": {"source_groups": 500, "valid_groups": 500, "candidates_per_source_group": 10},
    "gsm8k_mistral24b_t1.0": {"source_groups": 1319, "valid_groups": 1319, "candidates_per_source_group": 1},
    "gsm8k_nemo_t1.0": {"source_groups": 1319, "valid_groups": 1317, "candidates_per_source_group": 1},
    "gsm8k_phi35_t1.0": {"source_groups": 1319, "valid_groups": 1315, "candidates_per_source_group": 1},
    "gsm8k_mistral7b_t1.0": {"source_groups": 1319, "valid_groups": 1307, "candidates_per_source_group": 1},
    "gsm8k_phi3mini_t1.0": {"source_groups": 1319, "valid_groups": 1315, "candidates_per_source_group": 1},
    "triviaqa_qwen3_t0.6": {"source_groups": 500, "valid_groups": 500, "candidates_per_source_group": 10},
}

CORE_SOLVERS = ("upcr", "iu_pcr", "dufs_liu_pcr")
PRIMARY_CELLS = tuple(name for name, spec in CELLS.items() if not spec.get("protocol_rejected"))
ORIGINAL_LLAMA_CELLS = tuple(list(CELLS)[:6])
GSM8K_ARCHITECTURE_CELLS = tuple(
    name for name, spec in CELLS.items() if spec["dataset"] == "GSM8K"
)
COMPARATOR_FIDELITY = (
    {"method": "TriLens entropy features", "implementation": "3xL MHSA/FFN/residual token-mean entropy", "label_use": "none for fusion; grouped LR diagnostic", "fidelity": "feature-faithful approximation", "limitation": "paper text does not specify its fixed token readout"},
    {"method": "HaloScope", "implementation": "fixed middle-layer k=4 direct SVD-membership score on 256-D mean-token JL projection", "label_use": "none", "fidelity": "stage-1 proxy", "limitation": "not last-token full hidden state and excludes validation-selected pseudo-label classifier"},
    {"method": "DoLa-style detector", "implementation": "residual depth-wise KL-to-final vector with label-free fusion and grouped LR ceiling", "label_use": "none for fusion; grouped LR diagnostic", "fidelity": "KL proxy", "limitation": "saved KL is not the paper comparator's JSD; DoLa itself is a decoding method"},
    {"method": "Spilled Energy", "implementation": "Eq.8 raw-logit minus next-step logsumexp reconstructed when sampled token is in raw top-K", "label_use": "none", "fidelity": "equation-faithful token proxy", "limitation": "full-answer pooling; exact-answer span was not captured"},
    {"method": "INSIDE EigenScore", "implementation": "K=10 middle-layer last-token embeddings, alpha=0.001 logdet", "label_use": "none", "fidelity": "paper equation without feature clipping", "limitation": "only rejected CoQA/Llama-1 cell; appendix only"},
)
SOURCE_CODE = (
    "scripts/whitebox_layer_fusion_experiment.py",
    "scripts/whitebox_layer_fusion_report.py",
    "spectral_utils/whitebox_layer_fusion.py",
    "spectral_utils/paper_benchmark_suite.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/dependency_fusion.py",
    "spectral_utils/upcr_clustered.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/selectors/a2_groupfs.py",
    "cluster/layer_lens.py",
    "cluster/run_layer_views.py",
    "cluster/submit_layer_views.sbatch.template",
    "cluster/run_layer_views_reference.py",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return str(value)


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(tuple(str(value) for value in values), dtype="U")


def save_feature_matrix(path: Path, matrix: FeatureMatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=matrix.values,
        feature_names=_string_array(matrix.feature_names),
        risk_anchor=matrix.risk_anchor,
        groups=_string_array(matrix.groups),
        protocol_signature=np.asarray(matrix.protocol_signature),
        metadata_json=np.asarray(json.dumps(jsonable(matrix.metadata), sort_keys=True)),
    )


def load_feature_matrix(path: Path) -> FeatureMatrix:
    with np.load(path, allow_pickle=False) as bundle:
        forbidden = [key for key in bundle.files if "label" in key.lower()]
        if forbidden:
            raise RuntimeError(f"label-like arrays found in {path}: {forbidden}")
        return FeatureMatrix(
            values=bundle["values"],
            feature_names=tuple(bundle["feature_names"].astype(str)),
            risk_anchor=bundle["risk_anchor"],
            groups=tuple(bundle["groups"].astype(str)),
            protocol_signature=str(bundle["protocol_signature"].item()),
            metadata=json.loads(str(bundle["metadata_json"].item())),
        )


def _valid_row_metadata(cell: Any) -> dict[str, np.ndarray]:
    return {
        "row_ids": _string_array(cell.row_ids),
        "problem_ids": _string_array(cell.problem_ids),
        "n_gen_tokens": np.asarray(cell.n_gen_tokens, dtype=np.int64),
        "protocol_signature": np.asarray(cell.protocol_signature),
    }


def _raw_candidate_for_row(raw: Mapping[Any, Any], row_id: str) -> Mapping[str, Any]:
    problem, candidate = row_id.rsplit(":", 1)
    entry = raw.get(problem, raw.get(int(problem) if problem.isdigit() else problem))
    if not isinstance(entry, Mapping):
        raise KeyError(f"missing raw problem {problem}")
    return entry["candidates"][int(candidate)]


def extract_raw_output_contract(raw: Mapping[Any, Any], cell: Any) -> tuple[FeatureMatrix, dict[str, Any]]:
    """Build frozen output-layer baselines, including a top-K Spilled-Energy proxy."""

    entropy_mean, realized_nll_mean, spilled_mean, spilled_min = [], [], [], []
    hit_tokens = eligible_tokens = rows_without_delta = 0
    for row_id in cell.row_ids:
        candidate = _raw_candidate_for_row(raw, row_id)
        entropy = np.asarray(candidate["token_entropies"], dtype=float)
        realized = np.asarray(candidate["token_spilled_energies"], dtype=float)
        entropy_mean.append(float(np.mean(entropy)))
        realized_nll_mean.append(float(np.mean(realized)))
        ids = np.asarray(candidate["top_k_logprobs_raw"]["ids"])
        logprobs = np.asarray(candidate["top_k_logprobs_raw"]["logprobs"], dtype=float)
        generated = np.asarray(candidate["gen_token_ids"])
        logz = np.asarray(candidate["token_logsumexp"], dtype=float)
        limit = min(len(generated), len(ids), len(logprobs), len(logz))
        delta = []
        for token in range(max(0, limit - 1)):
            eligible_tokens += 1
            match = np.flatnonzero(ids[token] == generated[token])
            if not len(match):
                continue
            hit_tokens += 1
            raw_logit = float(logprobs[token, int(match[0])] + logz[token])
            delta.append(raw_logit - float(logz[token + 1]))
        if delta:
            spilled_mean.append(float(np.mean(delta)))
            spilled_min.append(float(np.min(delta)))
        else:
            rows_without_delta += 1
            spilled_mean.append(float("nan"))
            spilled_min.append(float("nan"))
    columns = [entropy_mean, realized_nll_mean, spilled_mean, spilled_min]
    values = np.column_stack(columns).astype(float)
    imputed = {}
    for index, name in enumerate(("generation_entropy_mean", "realized_token_nll_mean",
                                  "spilled_energy_full_answer_mean_proxy",
                                  "spilled_energy_full_answer_min_proxy")):
        missing = ~np.isfinite(values[:, index])
        if np.any(missing):
            median = float(np.nanmedian(values[:, index]))
            values[missing, index] = median
            imputed[name] = {"n_rows": int(np.sum(missing)), "unlabeled_median": median}
    matrix = FeatureMatrix(
        values=values,
        feature_names=("generation_entropy_mean", "realized_token_nll_mean",
                       "spilled_energy_full_answer_mean_proxy",
                       "spilled_energy_full_answer_min_proxy"),
        risk_anchor=np.asarray(extract_resid_core(cell).risk_anchor, dtype=float),
        groups=("output_entropy", "output_nll", "spilled_energy", "spilled_energy"),
        protocol_signature=cell.protocol_signature,
        metadata={
            "contract": "raw-output-baselines",
            "fidelity": {
                "generation_entropy_mean": "direct",
                "realized_token_nll_mean": "direct; legacy field name token_spilled_energies is not paper Spilled Energy",
                "spilled_energy": "Eq.8 top-K reconstruction over full answer; exact-answer localization unavailable",
            },
            "top_k_sampled_token_coverage": hit_tokens / eligible_tokens if eligible_tokens else 0.0,
            "eligible_spilled_tokens": eligible_tokens,
            "matched_spilled_tokens": hit_tokens,
            "rows_without_a_two-step_delta": rows_without_delta,
            "unlabeled_imputation": imputed,
        },
    )
    return matrix, dict(matrix.metadata)


def extract_inside_eigenscore(raw: Mapping[Any, Any], cell: Any) -> FeatureMatrix | None:
    """Paper-equation INSIDE EigenScore for cells carrying K last-token embeddings."""

    scores = np.full(cell.n_samples, np.nan, dtype=float)
    by_problem: dict[str, list[int]] = defaultdict(list)
    for index, problem in enumerate(cell.problem_ids):
        by_problem[str(problem)].append(index)
    for indices in by_problem.values():
        vectors = []
        for index in indices:
            value = _raw_candidate_for_row(raw, cell.row_ids[index]).get("hidden_middle_last")
            if value is None:
                return None
            vectors.append(np.asarray(value, dtype=float))
        z = np.vstack(vectors)
        z = z - np.mean(z, axis=1, keepdims=True)
        sigma = z @ z.T
        sign, logdet = np.linalg.slogdet(sigma + 0.001 * np.eye(len(indices)))
        if sign <= 0 or not np.isfinite(logdet):
            raise ValueError("INSIDE regularized covariance is not positive definite")
        scores[indices] = float(logdet / len(indices))
    return FeatureMatrix(
        values=scores[:, None], feature_names=("inside_eigenscore",),
        risk_anchor=np.asarray(extract_resid_core(cell).risk_anchor, dtype=float),
        groups=("inside",), protocol_signature=cell.protocol_signature,
        metadata={
            "contract": "inside-eigenscore",
            "alpha": 0.001, "readout": "saved_middle_layer_last_token",
            "candidate_generations_per_problem": sorted({len(v) for v in by_problem.values()}),
            "fidelity": "paper equation without feature clipping",
        },
    )


def _backfill_gate(path: Path) -> dict[str, Any]:
    report = read_json(path)
    entries = report.get("pkls", [])
    if len(entries) != 1:
        raise ValueError(f"expected one pkl gate in {path}")
    entry = entries[0]
    gate = entry.get("gate", {}).get("H", {})
    return {
        "pass": bool(entry.get("gate_b_pass")),
        "median": gate.get("median_abs"),
        "first": gate.get("first_tok_median"),
        "fraction": gate.get("frac_close"),
        "n_traces": gate.get("n_traces"),
        "n_tokens": gate.get("n_tokens"),
    }


def load_validation_artifact(path: Path | None) -> dict[str, Any]:
    """Load optional live validation evidence, with fail-closed coverage gates."""

    default = {
        "provided": False,
        "corrected_gate_b_all_pass": False,
        "architecture_pilot_pass": False,
        "gate_b_cells": 0,
        "architecture_cells": 0,
        "status": "BLOCKED",
        "reason": "no combined live validation artifact was supplied",
    }
    if path is None:
        return default
    path = path.resolve()
    payload = read_json(path)
    schema = payload.get("schema_version")
    if schema not in {"whitebox-layer-validation-v1", "whitebox-layer-validation-v2", "layer-reference-pilot-v1"}:
        raise ValueError(f"unsupported validation artifact schema {schema!r}")
    if schema in {"whitebox-layer-validation-v1", "whitebox-layer-validation-v2"}:
        gate_cells = int(payload.get("corrected_gate_b_cells", 0))
        architecture_cells = int(payload.get("architecture_pilot_cells", 0))
        expected_gate_cells = 6 if schema.endswith("v1") else len(CELLS)
        gate_pass = payload.get("corrected_gate_b_all_pass") is True and gate_cells == expected_gate_cells
        architecture_pass = (
            payload.get("architecture_pilot_pass") is True and architecture_cells == 2
        )
    else:
        cells = payload.get("cells", {})
        gate_cells = len(cells) if isinstance(cells, Mapping) else 0
        architecture_cells = gate_cells
        # The reference driver covers only the two architecture-pilot cells;
        # it cannot claim six-cell corrected Gate-B coverage by itself.
        gate_pass = False
        architecture_pass = (
            payload.get("architecture_fidelity_pass") is True and architecture_cells == 2
        )
    passed = bool(gate_pass and architecture_pass)
    return {
        "provided": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "schema_version": schema,
        "corrected_gate_b_all_pass": bool(gate_pass),
        "architecture_pilot_pass": bool(architecture_pass),
        "gate_b_cells": gate_cells,
        "architecture_cells": architecture_cells,
        "status": "PASS" if passed else "BLOCKED",
        "reason": (
            "all live validation requirements passed"
            if passed
            else "artifact does not explicitly pass the registered Gate-B roster and two architecture cells"
        ),
    }


def _source_entry(cache_root: Path, relative: str, *, remote_modtime: str,
                  expected_size: int, expected_sha: str, remote_prefix: str) -> dict[str, Any]:
    local = cache_root / relative
    size = local.stat().st_size
    digest = sha256_file(local)
    if size != expected_size or digest != expected_sha:
        raise RuntimeError(f"source mismatch: {local} size={size} sha256={digest}")
    remote_relative = relative.split("/", 1)[1]
    return {
        "remote_path": f"gdrive:hallucination_detection/cluster_results/{remote_prefix}/{remote_relative}",
        "remote_modification_time": remote_modtime,
        "remote_size": expected_size,
        "remote_hash_algorithm": "sha256",
        "remote_hash": expected_sha,
        "local_path": str(local.resolve()),
        "local_size": size,
        "local_sha256": digest,
        "remote_local_sha256_equal": digest == expected_sha,
    }


def phase_prepare(
    cache_root: Path, results: Path, validation_artifact: Path | None = None
) -> None:
    prepared = results / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    live_validation = load_validation_artifact(validation_artifact)
    source_entries, audits, coverage = [], {}, []
    source_ids = set()
    valid_ids = set()
    all_backfill_pass = True
    for cell_name, spec in CELLS.items():
        print(f"[prepare] {cell_name}", flush=True)
        raw_path = cache_root / spec["raw"]
        side_path = cache_root / spec["sidecar"]
        source_entries.extend((
            _source_entry(
                cache_root, spec["raw"], remote_modtime=spec["remote_raw_modtime"],
                expected_size=spec["raw_size"], expected_sha=spec["raw_sha256"],
                remote_prefix="repgrid",
            ),
            _source_entry(
                cache_root, spec["sidecar"], remote_modtime=spec["remote_sidecar_modtime"],
                expected_size=spec["sidecar_size"], expected_sha=spec["sidecar_sha256"],
                remote_prefix="layer_views",
            ),
        ))
        raw, sidecar = load_pickle(raw_path), load_pickle(side_path)
        cell, audit = validate_and_join(
            raw,
            sidecar,
            cell_id=cell_name,
            expected_model=spec.get("model", MODEL),
            expected_n_layers=int(spec.get("n_layers", 32)),
            expected_hidden_size=int(spec.get("hidden_size", 4096)),
            exclude_invalid=True,
            require_geometry_finite=False,
        )
        expected_excluded = set(EXPECTED_EXCLUSIONS.get(cell_name, {}))
        observed_excluded = {row["row_id"] for row in audit["excluded_rows"]}
        if observed_excluded != expected_excluded:
            raise RuntimeError(
                f"{cell_name}: exclusions {observed_excluded}, expected {expected_excluded}"
            )
        for row_id, (raw_tokens, side_tokens) in EXPECTED_EXCLUSIONS.get(cell_name, {}).items():
            problem, candidate = row_id.split(":")
            raw_entry = raw[int(problem)]["candidates"][int(candidate)]
            if len(raw_entry["gen_token_ids"]) != raw_tokens:
                raise RuntimeError(f"{cell_name}/{row_id}: unexpected raw token length")
            if int(sidecar[row_id]["n_gen_tokens"]) != side_tokens:
                raise RuntimeError(f"{cell_name}/{row_id}: unexpected sidecar token length")
        if audit["n_source_rows"] != spec["source_rows"]:
            raise RuntimeError(f"{cell_name}: source roster mismatch")
        source_cell_ids = {f"{cell_name}/{key}" for key in sidecar if key != "_meta"}
        valid_cell_ids = {f"{cell_name}/{key}" for key in cell.row_ids}
        if source_ids & source_cell_ids or valid_ids & valid_cell_ids:
            raise RuntimeError("globally namespaced row IDs are not unique")
        source_ids.update(source_cell_ids)
        valid_ids.update(valid_cell_ids)

        architecture_layers = all_layers(cell.n_layers)
        architecture_spaced = spaced_layers(cell.n_layers)
        architecture_late = late_layers(cell.n_layers)
        raw_output, raw_output_audit = extract_raw_output_contract(raw, cell)
        matrices = {
            "resid_core_all": extract_resid_core(cell, architecture_layers),
            "resid_core_spaced8": extract_resid_core(cell, architecture_spaced),
            "resid_core_late8": extract_resid_core(cell, architecture_late),
            "lens96": extract_lens96(cell),
            "lens_grid_all": extract_lens_grid(cell),
            "trilens_entropy_all": extract_trilens_entropy(cell),
            "dola_kl_proxy_all": extract_dola_kl_proxy(cell),
            "haloscope_projection": extract_haloscope_projection(cell),
            "raw_output_baselines": raw_output,
        }
        inside = extract_inside_eigenscore(raw, cell)
        if inside is not None:
            matrices["inside_eigenscore"] = inside
        matrices["resid_core_all_length_residualized"] = residualize_token_length(
            matrices["resid_core_all"], cell.n_gen_tokens
        )
        for name, matrix in matrices.items():
            save_feature_matrix(prepared / f"{cell_name}__{name}.npz", matrix)
        np.savez_compressed(prepared / f"{cell_name}__rows.npz", **_valid_row_metadata(cell))

        backfill = _backfill_gate(cache_root / spec["backfill"])
        all_backfill_pass &= backfill["pass"]
        audit.update({
            "source_rows_globally_namespaced_unique": True,
            "valid_rows_globally_namespaced_unique": True,
            "feature_contracts": {
                name: {
                    "n_samples": matrix.n_samples,
                    "n_features": matrix.n_features,
                    "protocol_signature": matrix.protocol_signature,
                    "finite": bool(np.isfinite(matrix.values).all()),
                    "metadata": jsonable(matrix.metadata),
                }
                for name, matrix in matrices.items()
            },
            "raw_output_baseline_audit": raw_output_audit,
            "architecture_relative_layers": {
                "all": list(architecture_layers),
                "spaced8": list(architecture_spaced),
                "late8": list(architecture_late),
            },
            "raw_backfill_gate_b": backfill,
            "corrected_live_gate_b": {
                "pass": live_validation["corrected_gate_b_all_pass"],
                "status": "PASS" if live_validation["corrected_gate_b_all_pass"] else "BLOCKED",
                "reason": live_validation["reason"],
            },
            "architecture_pilot": {
                "pass": live_validation["architecture_pilot_pass"],
                "status": "PASS" if live_validation["architecture_pilot_pass"] else "BLOCKED",
                "reason": live_validation["reason"],
            },
        })
        source_problem_counts = Counter(
            str(key).rsplit(":", 1)[0] for key in sidecar if key != "_meta"
        )
        valid_problem_counts = Counter(cell.problem_ids)
        expected_cohort = EXPECTED_COHORTS[cell_name]
        if len(source_problem_counts) != expected_cohort["source_groups"]:
            raise RuntimeError(f"{cell_name}: unexpected source problem-group count")
        if len(valid_problem_counts) != expected_cohort["valid_groups"]:
            raise RuntimeError(f"{cell_name}: unexpected evaluable problem-group count")
        if set(source_problem_counts.values()) != {
            expected_cohort["candidates_per_source_group"]
        }:
            raise RuntimeError(f"{cell_name}: unexpected candidate grouping")
        audit.update({
            "source_n_problems": len(source_problem_counts),
            "source_candidate_multiplicity": dict(sorted(Counter(source_problem_counts.values()).items())),
            "valid_candidate_multiplicity": dict(sorted(Counter(valid_problem_counts.values()).items())),
            "all_registered_tensor_shapes_valid": True,
            "all_core_lens_values_finite": True,
            "all_geometry_values_finite": audit["geometry_tensors_finite"],
            "labels_equal_between_raw_and_sidecar": True,
            "token_lengths_equal_for_evaluable_rows": True,
        })
        audits[cell_name] = audit
        coverage.append({
            "cell": cell_name,
            "cell_id": spec["cell_id"],
            "dataset": spec["dataset"],
            "model": spec.get("model", MODEL),
            "model_family": spec.get("model_family", "Llama-3.1-8B"),
            "n_layers": cell.n_layers,
            "hidden_size": int(spec.get("hidden_size", 4096)),
            "n_source_rows": audit["n_source_rows"],
            "n_samples": audit["n_rows"],
            "n_excluded_rows": audit["n_excluded_rows"],
            "n_groups": audit["n_problems"],
            "protocol_scope": "appendix_rejected" if spec.get("protocol_rejected") else "primary",
            "geometry_status": "PASS" if audit["geometry_tensors_finite"] else "BLOCKED_NONFINITE_COV_EIGS",
            "cov_eigs_nonfinite": audit["nonfinite_geometry_counts"]["cov_eigs"],
            "prevalence": "",
            "raw_backfill_gate_b_status": "PASS" if backfill["pass"] else "FAIL",
            "raw_backfill_gate_b_median": backfill["median"],
            "raw_backfill_gate_b_first": backfill["first"],
            "raw_backfill_gate_b_fraction": backfill["fraction"],
            "corrected_live_gate_b_status": "PASS" if live_validation["corrected_gate_b_all_pass"] else "BLOCKED",
            "corrected_layer_gate_b_status": "PASS" if live_validation["corrected_gate_b_all_pass"] else "BLOCKED",
            "gate_b_status": "PASS" if live_validation["corrected_gate_b_all_pass"] else "BLOCKED",
            "architecture_status": "PASS" if live_validation["architecture_pilot_pass"] else "BLOCKED",
            "status": "VALIDATED" if live_validation["status"] == "PASS" else "PRELIMINARY / VALIDATION BLOCKED",
            "exclusion_reason": "; ".join(
                f"{row['row_id']}: {row['reason']}" for row in audit["excluded_rows"]
            ),
        })
        del raw, sidecar, cell, matrices

    if len(source_ids) != 47265 or len(valid_ids) != 47238:
        raise RuntimeError(f"roster totals are {len(source_ids)}/{len(valid_ids)}")
    write_json(results / "data_audit.json", {
        "version": VERSION,
        "n_cells": len(CELLS),
        "n_source_rows": len(source_ids),
        "n_evaluable_rows": len(valid_ids),
        "n_excluded_rows": len(source_ids) - len(valid_ids),
        "cells": audits,
    })
    write_csv(results / "data_coverage.csv", coverage)
    write_csv(results / "comparator_fidelity.csv", COMPARATOR_FIDELITY)
    prepared_files = []
    for path in sorted(prepared.glob("*.npz")):
        with np.load(path, allow_pickle=False) as bundle:
            label_like = [
                key for key in bundle.files
                if "label" in key.lower() or key.lower() in {"y", "target", "targets"}
            ]
            if label_like:
                raise RuntimeError(f"label-like fields in prepared bundle {path}: {label_like}")
            prepared_files.append({
                "file": str(path.relative_to(results)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "fields": list(bundle.files),
            })
    if len(prepared_files) != 155:
        raise RuntimeError(f"prepared feature roster has {len(prepared_files)} files, expected 155")
    write_json(results / "PREPARED_FEATURE_MANIFEST.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_present": False,
        "n_files": len(prepared_files),
        "files": prepared_files,
    })
    code_hashes = {path: sha256_file(REPO / path) for path in SOURCE_CODE}
    write_json(results / "SOURCE_FREEZE_MANIFEST.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "read_only_remote": "gdrive:hallucination_detection/",
        "sources": source_entries,
        "registered_source_sha256": code_hashes,
        "n_source_files": len(source_entries),
        "n_source_rows": len(source_ids),
        "n_evaluable_rows": len(valid_ids),
        "rclone_check_download_zero_differences": True,
        "live_validation_artifact": live_validation,
    })
    definition = {
        "version": VERSION,
        "scientific_run": True,
        "written_utc": utcnow(),
        "models": sorted({spec.get("model", MODEL) for spec in CELLS.values()}),
        "claim_scope": "13 protocol-eligible cells across eight accepted model families; CoQA/Llama-1 appendix rejected",
        "cells": list(CELLS),
        "feature_contracts": [
            "resid_core_all", "resid_core_spaced8", "resid_core_late8", "lens96",
            "trilens_entropy_all", "dola_kl_proxy_all", "haloscope_projection",
            "raw_output_baselines", "resid_core_all_length_residualized",
        ],
        "geometry_performance_enabled": "HaloScope mean-token JL proxy only",
        "geometry_omission_reason": "covariance appendix blocked on Phi-3, Phi-3.5, and Qwen3 float16 overflow",
        "dufs": {"seeds": [11, 23, 37], "epochs": 80, "k": 7, "lambda": 0.1},
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": SEED,
                       "unit": "problem_group_within_cell", "tie_tolerance": TIE_TOLERANCE},
        "orientation_anchor": "final-layer residual target-token NLL; label-free",
        "label_definition": "y_hallucination = 1 - correctness label; evaluator only",
        "source_sha256": code_hashes,
    }
    definition["protocol_signature"] = hashlib.sha256(
        json.dumps(definition, sort_keys=True).encode()
    ).hexdigest()
    write_json(results / "RUN_DEFINITION.json", definition)
    validated = bool(
        live_validation["corrected_gate_b_all_pass"]
        and live_validation["architecture_pilot_pass"]
    )
    write_json(results / "validation_status.json", {
        "status": "VALIDATED" if validated else "PRELIMINARY / VALIDATION BLOCKED",
        "raw_backfill_gate_b_all_pass": all_backfill_pass,
        "corrected_live_gate_b_all_pass": live_validation["corrected_gate_b_all_pass"],
        "corrected_layer_gate_b_all_pass": live_validation["corrected_gate_b_all_pass"],
        "gate_b_all_pass": live_validation["corrected_gate_b_all_pass"],
        "architecture_pilot_pass": live_validation["architecture_pilot_pass"],
        "geometry_semantics_verified": True,
        "geometry_covariance_all_cells_finite": False,
        "validation_artifact": live_validation,
        "blockers": [] if validated else [
            "corrected live Gate B requires explicit 14-cell coverage",
            "architecture-fidelity pilot requires the two registered live cells",
            "float16 covariance eigenvalue overflow blocks covariance geometry on three cells",
        ],
    })
    print(f"[prepare] wrote {results}", flush=True)


def _score_name(method: str, contract: str, subset: str, structure: str = "flat") -> str:
    return f"{method}__{contract}__{subset}__{structure}"


def standardized_contract(matrix: FeatureMatrix) -> dict[str, Any]:
    """Hash the exact canonical standardized matrix and anchor used in a fit."""

    values, keep, means, scales = canonical_standardize(matrix.values)
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype="<f8").tobytes())
    digest.update(np.asarray(matrix.risk_anchor, dtype="<f8").tobytes())
    digest.update(np.asarray(keep, dtype="<i8").tobytes())
    return {
        "matrix_anchor_sha256": digest.hexdigest(),
        "kept_column_indices": keep.tolist(),
        "kept_feature_names": [matrix.feature_names[int(index)] for index in keep],
        "standardization_mean": means.tolist(),
        "standardization_scale": scales.tolist(),
    }


def _fit_cell(prepared: Path, cell_name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows_path = prepared / f"{cell_name}__rows.npz"
    with np.load(rows_path, allow_pickle=False) as row_bundle:
        row_ids = row_bundle["row_ids"].astype(str)
        problem_ids = row_bundle["problem_ids"].astype(str)
        n_tokens = row_bundle["n_gen_tokens"].astype(np.int64)
        protocol_signature = str(row_bundle["protocol_signature"].item())
    matrix_names = (
        "resid_core_all", "resid_core_spaced8", "resid_core_late8", "lens96",
        "trilens_entropy_all", "dola_kl_proxy_all", "haloscope_projection",
        "raw_output_baselines", "resid_core_all_length_residualized",
    )
    matrices = {
        name: load_feature_matrix(prepared / f"{cell_name}__{name}.npz")
        for name in matrix_names
    }
    inside_path = prepared / f"{cell_name}__inside_eigenscore.npz"
    if inside_path.exists():
        matrices["inside_eigenscore"] = load_feature_matrix(inside_path)
    if any(matrix.protocol_signature != protocol_signature for matrix in matrices.values()):
        raise RuntimeError(f"{cell_name}: prepared protocol mismatch")
    scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {
        "labels_seen_during_fit": False,
        "fits": {},
        "standardized_contracts": {
            name: standardized_contract(matrix)
            for name, matrix in matrices.items()
            if matrix.n_features >= 3
        },
    }

    all_depth = matrices["resid_core_all"]
    controls, diag = fit_controls(all_depth, n_gen_tokens=n_tokens)
    for method, score in controls.items():
        scores[_score_name(method, "resid-core-L", "all")] = score
    diagnostics["fits"]["controls_all"] = diag

    for subset, key, contract in (
        ("all", "resid_core_all", "resid-core-L"),
        ("spaced8", "resid_core_spaced8", "resid-core-8"),
        ("late8", "resid_core_late8", "resid-core-8"),
    ):
        fitted, diag = fit_core_spectral(matrices[key])
        for method, score in fitted.items():
            scores[_score_name(method, contract, subset)] = score
        diagnostics["fits"][f"core_{subset}"] = diag

    dependencies, diag = fit_dependency_methods(
        all_depth, clustered_groups=all_depth.groups
    )
    for method, score in dependencies.items():
        scores[_score_name(method, "resid-core-L", "all", "dependency")] = score
    diagnostics["fits"]["dependency_all"] = diag

    for solver in CORE_SOLVERS:
        score, diag = fit_hierarchical(all_depth, solver)
        scores[_score_name(solver, "resid-core-L", "all", "hierarchical-bands")] = score
        diagnostics["fits"][f"hierarchical_all_{solver}"] = diag

    lens = matrices["lens96"]
    fitted, diag = fit_core_spectral(lens)
    for method, score in fitted.items():
        scores[_score_name(method, "lens-96", "spaced8", "flat")] = score
    diagnostics["fits"]["lens96_flat"] = diag
    for solver in CORE_SOLVERS:
        score, hdiag = fit_hierarchical(lens, solver)
        scores[_score_name(solver, "lens-96", "spaced8", "hierarchical-module-metric")] = score
        diagnostics["fits"][f"lens96_hierarchical_{solver}"] = hdiag

    length_matrix = matrices["resid_core_all_length_residualized"]
    fitted, diag = fit_core_spectral(length_matrix)
    for method, score in fitted.items():
        scores[_score_name(method, "resid-core-L-length-residualized", "all")] = score
    diagnostics["fits"]["core_all_length_residualized"] = diag

    trilens = matrices["trilens_entropy_all"]
    tri_controls, diag = fit_controls(trilens)
    for method in ("equal_mean", "pc1"):
        scores[_score_name(f"trilens_{method}", "trilens-entropy-3L", "all")] = tri_controls[method]
    diagnostics["fits"]["trilens_controls"] = diag
    fitted, diag = fit_core_spectral(trilens)
    for method, score in fitted.items():
        scores[_score_name(method, "trilens-entropy-3L", "all")] = score
    diagnostics["fits"]["trilens_core"] = diag
    for solver in CORE_SOLVERS:
        score, diag = fit_hierarchical(trilens, solver)
        scores[_score_name(solver, "trilens-entropy-3L", "all", "hierarchical-modules")] = score
        diagnostics["fits"][f"trilens_hierarchical_{solver}"] = diag

    dola = matrices["dola_kl_proxy_all"]
    dola_controls, diag = fit_controls(dola)
    for method in ("equal_mean", "pc1"):
        scores[_score_name(f"dola_kl_{method}", "dola-kl-proxy-L", "all")] = dola_controls[method]
    diagnostics["fits"]["dola_controls"] = diag
    fitted, diag = fit_core_spectral(dola)
    for method, score in fitted.items():
        scores[_score_name(method, "dola-kl-proxy-L", "all")] = score
    diagnostics["fits"]["dola_core"] = diag

    halo_score, halo_diag = fit_haloscope_direct_proxy(matrices["haloscope_projection"], k=4)
    scores[_score_name("haloscope_direct_proxy", "middle-layer-mean-token-JL", "middle")] = halo_score
    diagnostics["fits"]["haloscope_direct_proxy"] = halo_diag

    raw_output = matrices["raw_output_baselines"]
    raw_scores = {}
    for index, name in enumerate(raw_output.feature_names):
        score = np.asarray(raw_output.values[:, index], dtype=float)
        correlation = float(np.corrcoef(score, raw_output.risk_anchor)[0, 1])
        if not np.isfinite(correlation):
            raise RuntimeError(f"{cell_name}/{name}: non-finite anchor correlation")
        if correlation < 0:
            score = -score
        method = {
            "generation_entropy_mean": "generation_entropy_mean",
            "realized_token_nll_mean": "realized_token_nll_mean",
            "spilled_energy_full_answer_mean_proxy": "spilled_energy_eq8_mean_proxy",
            "spilled_energy_full_answer_min_proxy": "spilled_energy_eq8_min_proxy",
        }[name]
        scores[_score_name(method, "raw-output", "full-answer")] = score
        raw_scores[method] = {"orientation_flipped": correlation < 0, "anchor_correlation": correlation}
    diagnostics["fits"]["raw_output_baselines"] = {
        "labels_seen_during_fit": False, "scores": raw_scores,
        "metadata": jsonable(raw_output.metadata),
    }
    if "inside_eigenscore" in matrices:
        inside = matrices["inside_eigenscore"]
        scores[_score_name("inside_eigenscore", "middle-last-token-K10", "question")] = inside.values[:, 0]
        diagnostics["fits"]["inside_eigenscore"] = {
            "labels_seen_during_fit": False, "metadata": jsonable(inside.metadata),
            "protocol_scope": "appendix_rejected",
        }

    contract_assignments = {
        "controls_all": "resid_core_all",
        "core_all": "resid_core_all",
        "dependency_all": "resid_core_all",
        **{f"hierarchical_all_{solver}": "resid_core_all" for solver in CORE_SOLVERS},
        "core_spaced8": "resid_core_spaced8",
        "core_late8": "resid_core_late8",
        "lens96_flat": "lens96",
        **{f"lens96_hierarchical_{solver}": "lens96" for solver in CORE_SOLVERS},
        "core_all_length_residualized": "resid_core_all_length_residualized",
        "trilens_controls": "trilens_entropy_all",
        "trilens_core": "trilens_entropy_all",
        **{f"trilens_hierarchical_{solver}": "trilens_entropy_all" for solver in CORE_SOLVERS},
        "dola_controls": "dola_kl_proxy_all",
        "dola_core": "dola_kl_proxy_all",
        "haloscope_direct_proxy": "haloscope_projection",
        "raw_output_baselines": "raw_output_baselines",
    }
    diagnostics["fit_standardized_matrix_anchor_sha256"] = {
        fit_name: diagnostics["standardized_contracts"][contract]["matrix_anchor_sha256"]
        for fit_name, contract in contract_assignments.items()
    }
    expected_all32 = diagnostics["standardized_contracts"]["resid_core_all"]["matrix_anchor_sha256"]
    for fit_name, contract in contract_assignments.items():
        if contract == "resid_core_all" and diagnostics["fit_standardized_matrix_anchor_sha256"][fit_name] != expected_all32:
            raise AssertionError(f"{cell_name}/{fit_name}: standardized matrix/anchor mismatch")

    for method, score in scores.items():
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (len(row_ids),) or not np.isfinite(score).all():
            raise RuntimeError(f"{cell_name}/{method}: invalid score")
        scores[method] = score
    diagnostics.update({
        "protocol_signature": protocol_signature,
        "score_methods": list(scores),
        "n_rows": len(row_ids),
        "row_ids_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest(),
    })
    metadata = {"row_ids": row_ids, "problem_ids": problem_ids,
                "n_gen_tokens": n_tokens, "protocol_signature": protocol_signature}
    return scores, {"diagnostics": diagnostics, "metadata": metadata}


def verify_prepared_freeze(results: Path) -> None:
    manifest = read_json(results / "PREPARED_FEATURE_MANIFEST.json")
    if (
        manifest.get("labels_present") is not False
        or manifest.get("n_files") != len(manifest.get("files", []))
    ):
        raise RuntimeError("prepared feature manifest is incomplete or label-bearing")
    expected_files = [row.get("file") for row in manifest.get("files", [])]
    observed_files = [str(path.relative_to(results)) for path in sorted((results / "prepared").glob("*.npz"))]
    if expected_files != observed_files:
        raise RuntimeError("prepared feature roster differs from its manifest")
    for row in manifest["files"]:
        path = results / row["file"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"prepared feature changed after audit: {path}")
        if any("label" in str(key).lower() for key in row.get("fields", [])):
            raise RuntimeError(f"prepared manifest contains a label-like field: {path}")


def phase_fit(results: Path) -> None:
    assert_no_label_fitting_signatures()
    verify_prepared_freeze(results)
    prepared, score_dir, diagnostic_dir = results / "prepared", results / "scores", results / "diagnostics"
    score_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cell_name in CELLS:
        print(f"[fit] {cell_name}", flush=True)
        scores, payload = _fit_cell(prepared, cell_name)
        metadata = payload["metadata"]
        score_path = score_dir / f"{cell_name}.npz"
        np.savez_compressed(
            score_path,
            row_ids=metadata["row_ids"],
            problem_ids=metadata["problem_ids"],
            n_gen_tokens=metadata["n_gen_tokens"],
            protocol_signature=np.asarray(metadata["protocol_signature"]),
            **scores,
        )
        diagnostic_path = diagnostic_dir / f"{cell_name}.json"
        write_json(diagnostic_path, jsonable(payload["diagnostics"]))
        manifest.append({
            "cell": cell_name,
            "score_file": str(score_path.relative_to(results)),
            "score_sha256": sha256_file(score_path),
            "diagnostic_file": str(diagnostic_path.relative_to(results)),
            "diagnostic_sha256": sha256_file(diagnostic_path),
            "n_rows": len(metadata["row_ids"]),
            "n_methods": len(scores),
        })
    write_json(results / "FIT_COMPLETE.json", {
        "version": VERSION,
        "scientific_run": True,
        "written_utc": utcnow(),
        "labels_seen_during_fit": False,
        "n_cells": len(manifest),
        "score_manifest": manifest,
    })
    print("[fit] score fitting complete; labels remain unopened", flush=True)


def verify_score_freeze(results: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    verify_prepared_freeze(results)
    definition = read_json(results / "RUN_DEFINITION.json")
    for relative, expected in definition.get("source_sha256", {}).items():
        current = REPO / relative
        if not current.is_file() or sha256_file(current) != expected:
            raise RuntimeError(f"registered source changed after preparation: {relative}")
    source_freeze = read_json(results / "SOURCE_FREEZE_MANIFEST.json")
    for item in source_freeze.get("sources", []):
        local = Path(item["local_path"])
        if local.stat().st_size != item["local_size"] or sha256_file(local) != item["local_sha256"]:
            raise RuntimeError(f"frozen raw/sidecar source changed: {local}")
    fit_complete = read_json(results / "FIT_COMPLETE.json")
    if fit_complete.get("labels_seen_during_fit") is not False:
        raise RuntimeError("fit did not explicitly attest labels_seen_during_fit=false")
    if [row["cell"] for row in fit_complete.get("score_manifest", [])] != list(CELLS):
        raise RuntimeError("score roster differs from frozen cells")
    observed: dict[str, dict[str, np.ndarray]] = {}
    for row in fit_complete["score_manifest"]:
        score_path, diagnostic_path = results / row["score_file"], results / row["diagnostic_file"]
        if sha256_file(score_path) != row["score_sha256"] or sha256_file(diagnostic_path) != row["diagnostic_sha256"]:
            raise RuntimeError(f"frozen artifact hash mismatch for {row['cell']}")
        diagnostic = read_json(diagnostic_path)
        if diagnostic.get("labels_seen_during_fit") is not False:
            raise RuntimeError(f"diagnostic leakage attestation failed for {row['cell']}")
        with np.load(score_path, allow_pickle=False) as bundle:
            forbidden = [key for key in bundle.files if "label" in key.lower() or key.lower() in {"y", "target"}]
            if forbidden:
                raise RuntimeError(f"label-like arrays in {score_path}: {forbidden}")
            observed[row["cell"]] = {key: np.asarray(bundle[key]) for key in bundle.files}
    freeze = {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_seen_during_fit": False,
        "scores_frozen_before_labels": True,
        "score_files_verified_before_labels": True,
        "fit_complete_sha256": sha256_file(results / "FIT_COMPLETE.json"),
        "run_definition_sha256": sha256_file(results / "RUN_DEFINITION.json"),
        "source_freeze_sha256": sha256_file(results / "SOURCE_FREEZE_MANIFEST.json"),
        "prepared_feature_manifest_sha256": sha256_file(results / "PREPARED_FEATURE_MANIFEST.json"),
        "score_manifest": fit_complete["score_manifest"],
    }
    path = results / "SCORE_FREEZE_MANIFEST.json"
    if path.exists():
        previous = read_json(path)
        comparable = dict(previous)
        comparable.pop("written_utc", None)
        current = dict(freeze)
        current.pop("written_utc", None)
        if comparable != current:
            raise RuntimeError("immutable score freeze disagrees with current artifacts")
        freeze = previous
    else:
        write_json(path, freeze)
    return observed, freeze


def metric_pair(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) != 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def group_bootstrap_indices(groups: Sequence[str], *, draws: int, seed: int) -> tuple[list[np.ndarray], str]:
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    indices, digest = [], hashlib.sha256()
    for _ in range(draws):
        sampled_positions = rng.integers(0, len(unique), size=len(unique))
        digest.update(sampled_positions.astype("<i8").tobytes())
        indices.append(np.concatenate([members[unique[position]] for position in sampled_positions]))
    return indices, digest.hexdigest()


def _method_parts(key: str) -> tuple[str, str, str, str]:
    parts = key.split("__")
    if len(parts) != 4:
        return key, "", "", ""
    return tuple(parts)  # type: ignore[return-value]


def _lr_ceiling(matrix: FeatureMatrix, y: np.ndarray, groups: np.ndarray) -> tuple[float, float, list[dict[str, Any]]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (train, test) in enumerate(splitter.split(matrix.values, y, groups)):
        overlap = set(groups[train]) & set(groups[test])
        if overlap:
            raise AssertionError(f"group overlap in supervised fold {fold}")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED + fold),
        )
        model.fit(matrix.values[train], y[train])
        probability = model.predict_proba(matrix.values[test])[:, 1]
        auroc, auprc = metric_pair(y[test], probability)
        rows.append({"fold": fold, "auroc": auroc, "auprc": auprc,
                     "n_train": len(train), "n_test": len(test), "problem_overlap": 0})
    return float(np.mean([row["auroc"] for row in rows])), float(np.mean([row["auprc"] for row in rows])), rows


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def phase_evaluate(cache_root: Path, results: Path) -> None:
    score_bundles, _freeze = verify_score_freeze(results)
    # No raw cache is opened above this line.
    per_cell, layer_rows, dependence_rows, weight_rows, coverage = [], [], [], [], []
    metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    boot: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    bootstrap_hashes, bootstrap_seeds = {}, {}
    lr_folds = {}

    for cell_index, (cell_name, spec) in enumerate(CELLS.items()):
        print(f"[evaluate] {cell_name}", flush=True)
        raw = load_pickle(cache_root / spec["raw"])
        bundle = score_bundles[cell_name]
        row_ids = bundle["row_ids"].astype(str)
        groups = bundle["problem_ids"].astype(str)
        y = load_evaluation_labels(raw, row_ids)
        prevalence = float(np.mean(y))
        methods = {
            key: np.asarray(value, dtype=float)
            for key, value in bundle.items()
            if key not in {"row_ids", "problem_ids", "n_gen_tokens", "protocol_signature"}
        }
        cell_bootstrap_seed = (
            SEED + int(hashlib.sha256(cell_name.encode()).hexdigest()[:8], 16)
        )
        indices, draw_hash = group_bootstrap_indices(
            groups, draws=BOOTSTRAP_DRAWS,
            seed=cell_bootstrap_seed,
        )
        bootstrap_hashes[cell_name] = draw_hash
        bootstrap_seeds[cell_name] = cell_bootstrap_seed
        for key, score in methods.items():
            auroc, auprc = metric_pair(y, score)
            metrics[key][cell_name] = {"auroc": auroc, "auprc": auprc}
            method, contract, subset, structure = _method_parts(key)
            appendix_only = bool(spec.get("protocol_rejected")) or method == "inside_eigenscore"
            per_cell.append({
                "cell": cell_name, "method_key": key, "method": method,
                "feature_contract": contract, "layer_subset": subset,
                "structured": structure, "auroc": auroc, "auprc": auprc,
                "prevalence": prevalence, "n_samples": len(y),
                "n_groups": len(np.unique(groups)),
                "status": "appendix_protocol_rejected" if appendix_only else "eligible_label_free",
                "label_use": "none",
            })
            auc_draws, ap_draws = [], []
            for index in indices:
                auc, ap = metric_pair(y[index], score[index])
                auc_draws.append(auc)
                ap_draws.append(ap)
            boot[key][cell_name] = {
                "auroc": np.asarray(auc_draws), "auprc": np.asarray(ap_draws)
            }

        matrix = load_feature_matrix(results / "prepared" / f"{cell_name}__resid_core_all.npz")
        lr_auc, lr_ap, folds = _lr_ceiling(matrix, y, groups)
        lr_folds[cell_name] = {"resid_core": folds}
        per_cell.append({
            "cell": cell_name, "method_key": "supervised_grouped_lr_ceiling",
            "method": "supervised_grouped_lr", "feature_contract": "resid-core-L",
            "layer_subset": "all", "structured": "5-fold grouped-CV ceiling",
            "auroc": lr_auc, "auprc": lr_ap, "prevalence": prevalence,
            "n_samples": len(y), "n_groups": len(np.unique(groups)),
            "status": "appendix_protocol_rejected" if spec.get("protocol_rejected") else "diagnostic_ceiling",
            "label_use": "supervised_ceiling",
        })

        for ceiling_name, prepared_name, contract_name in (
            ("trilens_supervised_lr", "trilens_entropy_all", "trilens-entropy-3L"),
            ("dola_kl_supervised_lr", "dola_kl_proxy_all", "dola-kl-proxy-L"),
        ):
            ceiling_matrix = load_feature_matrix(
                results / "prepared" / f"{cell_name}__{prepared_name}.npz"
            )
            ceiling_auc, ceiling_ap, ceiling_folds = _lr_ceiling(ceiling_matrix, y, groups)
            lr_folds[cell_name][ceiling_name] = ceiling_folds
            per_cell.append({
                "cell": cell_name, "method_key": ceiling_name,
                "method": ceiling_name, "feature_contract": contract_name,
                "layer_subset": "all", "structured": "5-fold grouped-CV diagnostic ceiling",
                "auroc": ceiling_auc, "auprc": ceiling_ap, "prevalence": prevalence,
                "n_samples": len(y), "n_groups": len(np.unique(groups)),
                "status": "appendix_protocol_rejected" if spec.get("protocol_rejected") else "diagnostic_ceiling",
                "label_use": "supervised_ceiling",
            })

        lens = load_feature_matrix(results / "prepared" / f"{cell_name}__lens_grid_all.npz")
        layer_best = (-np.inf, None)
        for feature_index, feature_name in enumerate(lens.feature_names):
            match = re.match(r"(?P<module>[^.]+)\.(?P<metric>[^.]+)\.layer_(?P<layer>\d+)", feature_name)
            if not match:
                continue
            score = lens.values[:, feature_index]
            auc, _ = metric_pair(y, score)
            corr = spearmanr(score, matrix.risk_anchor).statistic
            layer_rows.append({
                "cell": cell_name, "layer": int(match.group("layer")),
                "metric": match.group("metric"), "module": match.group("module"),
                "auroc": auc, "spearman_to_anchor": float(corr),
                "label_use": "evaluation_only",
            })
            if auc > layer_best[0]:
                layer_best = (auc, feature_name)
        per_cell.append({
            "cell": cell_name, "method_key": "best_single_layer_ceiling",
            "method": "best_single_layer", "feature_contract": str(layer_best[1]),
            "layer_subset": "evaluation-selected", "structured": "diagnostic ceiling",
            "auroc": layer_best[0], "auprc": "", "prevalence": prevalence,
            "n_samples": len(y), "n_groups": len(np.unique(groups)),
            "status": "appendix_protocol_rejected" if spec.get("protocol_rejected") else "diagnostic_ceiling",
            "label_use": "evaluation_only",
        })

        X = np.asarray(matrix.values, dtype=float)
        corr = np.corrcoef(X, rowvar=False)
        eigen = np.linalg.eigvalsh(corr)
        effective_rank = float(np.sum(eigen) ** 2 / (np.sum(eigen ** 2) + 1e-12))
        for i, name_i in enumerate(matrix.feature_names):
            layer_i = int(name_i.rsplit("_", 1)[1])
            for j, name_j in enumerate(matrix.feature_names):
                layer_j = int(name_j.rsplit("_", 1)[1])
                dependence_rows.append({
                    "cell": cell_name, "contract": "resid-core-L",
                    "diagnostic": "layer_correlation", "feature_a": layer_i,
                    "feature_b": layer_j,
                    "value": float(corr[i, j]), "effective_rank": effective_rank,
                })
        for distance in range(matrix.n_features):
            values = [abs(corr[i, j]) for i in range(matrix.n_features) for j in range(matrix.n_features) if abs(i - j) == distance]
            dependence_rows.append({
                "cell": cell_name, "contract": "resid-core-L",
                "diagnostic": "correlation_vs_layer_distance", "layer_distance": distance,
                "value": float(np.median(values)), "effective_rank": effective_rank,
            })

        diagnostic = read_json(results / "diagnostics" / f"{cell_name}.json")
        for fit_name, fit in diagnostic.get("fits", {}).items():
            if isinstance(fit, Mapping):
                for weight_key in (
                    "deployed_weights", "iu_weights", "dufs_liu_weights",
                    "dufs_gates", "outer_weights", "folded_feature_weights",
                ):
                    for feature_index, value in enumerate(fit.get(weight_key, []) or []):
                        weight_rows.append({
                            "cell": cell_name, "method": fit_name, "contract": fit_name,
                            "kind": weight_key, "feature": feature_index, "value": value,
                        })
                graph = fit.get("graph", {})
                if isinstance(graph, Mapping):
                    weight_rows.append({
                        "cell": cell_name, "method": fit_name, "contract": fit_name,
                        "kind": "graph_health",
                        "graph_components": graph.get("n_components", ""),
                        "mean_degree": graph.get("degree_mean", ""),
                        "spectral_gap": graph.get("algebraic_connectivity", ""),
                        "condition_number": graph.get("projected_condition_number", ""),
                        "weight_cosine_vs_iu": graph.get("weight_cosine_vs_iu", ""),
                    })
                nested_fits = fit.get("fits", {})
                if isinstance(nested_fits, Mapping):
                    for nested_name, nested in nested_fits.items():
                        if not isinstance(nested, Mapping):
                            continue
                        for weight_key in ("weights", "dufs_gates"):
                            for feature_index, value in enumerate(nested.get(weight_key, []) or []):
                                weight_rows.append({
                                    "cell": cell_name,
                                    "method": f"{fit_name}/{nested_name}",
                                    "contract": fit_name, "kind": weight_key,
                                    "feature": feature_index, "value": value,
                                })
                dufs = fit.get("dufs", {})
                if isinstance(dufs, Mapping) and "training_history" in dufs:
                    for seed_index, history in enumerate(dufs["training_history"]):
                        for epoch, value in enumerate(history):
                            weight_rows.append({
                                "cell": cell_name, "method": fit_name, "contract": fit_name,
                                "kind": "dufs_convergence", "feature": "loss",
                                "seed": seed_index, "epoch": epoch, "value": value,
                            })
        del raw

    coverage_by_cell = {row["cell"]: row for row in csv.DictReader((results / "data_coverage.csv").open())}
    for cell_name in CELLS:
        row = coverage_by_cell[cell_name]
        prevalence = next(float(item["prevalence"]) for item in per_cell if item["cell"] == cell_name)
        row["prevalence"] = prevalence
        coverage.append(row)
    write_csv(results / "data_coverage.csv", coverage)
    write_csv(results / "per_cell_metrics.csv", per_cell)
    write_csv(results / "layer_diagnostics.csv", layer_rows)
    write_csv(results / "dependence_diagnostics.csv", dependence_rows)
    write_csv(results / "weights_diagnostics.csv", weight_rows)
    write_json(results / "supervised_grouped_cv_diagnostics.json", lr_folds)
    write_json(results / "bootstrap_draw_manifest.json", {
        "draws": BOOTSTRAP_DRAWS, "seed": SEED,
        "derived_seed_by_cell": bootstrap_seeds,
        "identical_draws_reused_across_methods_within_cell": True,
        "draw_hash_by_cell": bootstrap_hashes,
    })

    headline = []
    eligible_keys = [key for key in metrics if all(cell in metrics[key] for cell in PRIMARY_CELLS)]
    for key in eligible_keys:
        method, contract, subset, structure = _method_parts(key)
        row = {"method_key": key, "method": method, "feature_contract": contract,
               "layer_subset": subset, "structured": structure}
        for metric in ("auroc", "auprc"):
            point = float(np.mean([metrics[key][cell][metric] for cell in PRIMARY_CELLS]))
            macro_draw = np.mean(np.vstack([boot[key][cell][metric] for cell in PRIMARY_CELLS]), axis=0)
            low, high = np.quantile(macro_draw, (0.025, 0.975))
            row[f"macro_{metric}"] = point
            row[f"macro_{metric}_ci_low"] = float(low)
            row[f"macro_{metric}_ci_high"] = float(high)
        headline.append(row)
    write_csv(results / "headline_summary.csv", headline)

    cohort_rows = []
    cohorts = {
        "primary_13_cells": PRIMARY_CELLS,
        "original_llama_six": ORIGINAL_LLAMA_CELLS,
        "gsm8k_cross_architecture_seven": GSM8K_ARCHITECTURE_CELLS,
        "all_14_descriptive_including_rejected_coqa": tuple(CELLS),
    }
    for cohort_name, cohort_cells in cohorts.items():
        for method_key in metrics:
            if not all(cell in metrics[method_key] for cell in cohort_cells):
                continue
            method, contract, subset, structure = _method_parts(method_key)
            cohort_rows.append({
                "cohort": cohort_name, "n_cells": len(cohort_cells), "method_key": method_key,
                "method": method, "feature_contract": contract, "layer_subset": subset,
                "structured": structure,
                "macro_auroc": float(np.mean([metrics[method_key][cell]["auroc"] for cell in cohort_cells])),
                "macro_auprc": float(np.mean([metrics[method_key][cell]["auprc"] for cell in cohort_cells])),
            })
    write_csv(results / "cohort_summary.csv", cohort_rows)

    def key(method: str, contract: str, subset: str, structure: str = "flat") -> str:
        return _score_name(method, contract, subset, structure)
    contrasts = [
        ("primary_dufs_all_depth_minus_final_nll", key("dufs_liu_pcr", "resid-core-L", "all"), key("final_layer_nll", "resid-core-L", "all"), True),
        ("primary_dufs_all_depth_minus_iu", key("dufs_liu_pcr", "resid-core-L", "all"), key("iu_pcr", "resid-core-L", "all"), True),
        ("iu_minus_upcr_all_depth", key("iu_pcr", "resid-core-L", "all"), key("upcr", "resid-core-L", "all"), False),
        ("upcr_minus_equal_mean", key("upcr", "resid-core-L", "all"), key("equal_mean", "resid-core-L", "all"), False),
        ("trilens_dufs_minus_trilens_equal_mean", key("dufs_liu_pcr", "trilens-entropy-3L", "all"), key("trilens_equal_mean", "trilens-entropy-3L", "all"), False),
        ("trilens_dufs_minus_final_nll", key("dufs_liu_pcr", "trilens-entropy-3L", "all"), key("final_layer_nll", "resid-core-L", "all"), False),
        ("trilens_dufs_minus_trilens_iu", key("dufs_liu_pcr", "trilens-entropy-3L", "all"), key("iu_pcr", "trilens-entropy-3L", "all"), False),
        ("expanded_lens_dufs_minus_trilens_dufs", key("dufs_liu_pcr", "lens-96", "spaced8"), key("dufs_liu_pcr", "trilens-entropy-3L", "all"), False),
        ("dufs_all_depth_minus_haloscope_direct_proxy", key("dufs_liu_pcr", "resid-core-L", "all"), key("haloscope_direct_proxy", "middle-layer-mean-token-JL", "middle"), False),
        ("dufs_all_depth_minus_spilled_energy_min_proxy", key("dufs_liu_pcr", "resid-core-L", "all"), key("spilled_energy_eq8_min_proxy", "raw-output", "full-answer"), False),
    ]
    for solver in CORE_SOLVERS:
        contrasts.extend((
            (f"{solver}_spaced8_minus_all_depth", key(solver, "resid-core-8", "spaced8"), key(solver, "resid-core-L", "all"), False),
            (f"{solver}_late8_minus_all_depth", key(solver, "resid-core-8", "late8"), key(solver, "resid-core-L", "all"), False),
            (f"{solver}_hierarchical_minus_flat", key(solver, "resid-core-L", "all", "hierarchical-bands"), key(solver, "resid-core-L", "all"), False),
            (f"{solver}_lens96_hierarchical_minus_flat", key(solver, "lens-96", "spaced8", "hierarchical-module-metric"), key(solver, "lens-96", "spaced8"), False),
            (f"{solver}_length_residualized_minus_raw", key(solver, "resid-core-L-length-residualized", "all"), key(solver, "resid-core-L", "all"), False),
            (f"{solver}_trilens_hierarchical_minus_flat", key(solver, "trilens-entropy-3L", "all", "hierarchical-modules"), key(solver, "trilens-entropy-3L", "all"), False),
        ))
    paired = []
    for name, lhs, rhs, primary in contrasts:
        if lhs not in metrics or rhs not in metrics:
            raise RuntimeError(f"registered contrast missing: {name}")
        for metric in ("auroc", "auprc"):
            cell_delta = np.asarray([metrics[lhs][cell][metric] - metrics[rhs][cell][metric] for cell in PRIMARY_CELLS])
            draw_delta = np.mean(np.vstack([
                boot[lhs][cell][metric] - boot[rhs][cell][metric] for cell in PRIMARY_CELLS
            ]), axis=0)
            low, high = np.quantile(draw_delta, (0.025, 0.975))
            try:
                p_raw = float(wilcoxon(cell_delta, zero_method="pratt").pvalue)
            except ValueError:
                p_raw = 1.0
            paired.append({
                "contrast": name, "lhs": lhs, "rhs": rhs, "metric": metric,
                "delta": float(np.mean(cell_delta)), "ci_low": float(low), "ci_high": float(high),
                "wins": int(np.sum(cell_delta > TIE_TOLERANCE)),
                "ties": int(np.sum(np.abs(cell_delta) <= TIE_TOLERANCE)),
                "losses": int(np.sum(cell_delta < -TIE_TOLERANCE)),
                "worst_cell_delta": float(np.min(cell_delta)), "p_raw": p_raw,
                "p_holm": "", "primary": primary and metric == "auroc",
                "per_cell_deltas_json": json.dumps(dict(zip(PRIMARY_CELLS, cell_delta.tolist())), sort_keys=True),
            })
    adjusted = holm_adjust([float(row["p_raw"]) for row in paired])
    for row, value in zip(paired, adjusted):
        row["p_holm"] = value
    write_csv(results / "paired_comparisons.csv", paired)

    # Reassert the freeze after evaluation; score hashes cannot depend on labels.
    before = [(row["score_file"], row["score_sha256"]) for row in _freeze["score_manifest"]]
    _observed_again, after_freeze = verify_score_freeze(results)
    after = [(row["score_file"], row["score_sha256"]) for row in after_freeze["score_manifest"]]
    if before != after:
        raise AssertionError("score hashes changed after labels were opened")
    print("[evaluate] artifacts complete; score hashes unchanged", flush=True)


def phase_report(results: Path) -> None:
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "whitebox_layer_fusion_report.py"),
         "--results-dir", str(results)],
        cwd=REPO,
        check=True,
    )


def phase_all(
    cache_root: Path, results: Path, validation_artifact: Path | None = None
) -> None:
    base = [sys.executable, str(Path(__file__).resolve()), "--cache-root", str(cache_root),
            "--results-dir", str(results)]
    for phase in ("prepare", "fit", "evaluate", "report"):
        command = base + ["--phase", phase]
        if validation_artifact is not None:
            command += ["--validation-artifact", str(validation_artifact)]
        subprocess.run(command, cwd=REPO, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fit", "evaluate", "report", "all"), default="all")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--validation-artifact", type=Path, default=None,
        help="optional live validation JSON; hashed and fail-closed",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cache_root, results = args.cache_root.resolve(), args.results_dir.resolve()
    results.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        phase_prepare(cache_root, results, args.validation_artifact)
    elif args.phase == "fit":
        phase_fit(results)
    elif args.phase == "evaluate":
        phase_evaluate(cache_root, results)
    elif args.phase == "report":
        phase_report(results)
    else:
        phase_all(cache_root, results, args.validation_artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    extract_dola_kl_proxy,
    extract_haloscope_projection,
    fit_haloscope_direct_proxy,
    late_layers,
    spaced_layers,
