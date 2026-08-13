"""Target-firewalled data preparation for automatic group-free IU Phase A5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import re
import unicodedata
from typing import Mapping, Sequence

import numpy as np

from .feature_utils import extract_all_features
from .answer_span import answer_token_slice
from .feature_utils import compute_spilled_energy_features


CORE_FEATURES = (
    "cusum_max",
    "cusum_max_energy",
    "cusum_max_spilled",
    "epr",
    "epr_energy",
    "epr_spilled",
    "logprob_margin",
    "mean_logprob_entropy",
    "mean_top1_logprob",
    "min_energy",
    "renyi_entropy_2",
    "rpdi",
    "sw_var_peak",
    "sw_var_peak_energy",
    "sw_var_peak_spilled",
    "topk_tail_mass",
    "varentropy",
)
A0_SOURCE_ENVIRONMENTS = (
    "ars_gsm8k_r1distill8b", "internalstates_gsm8k_qwen25_7b",
    "lapeigvals_gsm8k_llama3b", "lapeigvals_gsm8k_llama8b",
    "lapeigvals_gsm8k_mistral24b", "lapeigvals_gsm8k_nemo",
    "lapeigvals_gsm8k_phi35", "noise_gsm8k_mistral7b",
    "noise_gsm8k_phi3mini", "trace_gsm8k_llama8b_k10",
    "losnet_hotpotqa_mistral7b", "math500_dsmath7b",
    "math500_qwenmath7b", "math500_r1distill8b",
    "math500_r1distill8b_mn4096", "trace_math500_qwenmath15b_k10",
    "se_nq_open_llama8b", "sciq_llama8b", "se_squad_v2_llama8b",
    "epr_triviaqa_mistral24b", "seiclr_triviaqa_opt30b",
    "semenergy_triviaqa_qwen3_8b", "truthfulqa_llama8b",
)
TELEMETRY_KEYS = (
    "token_entropies",
    "token_spilled_energies",
    "token_logsumexp",
    "top_k_logprobs",
)
TARGET_LIKE = re.compile(
    r"(?:^|_)(?:label|labels|correct|correctness|is_correct|gold|answer|answers|"
    r"reference|target|first_error|error_label)(?:$|_)",
    re.IGNORECASE,
)


def _energy_features_from_logsumexp(values) -> dict:
    output = compute_spilled_energy_features(np.asarray(values, dtype=float))
    return {
        "epr_energy": output["epr_spilled"],
        "min_energy": output["min_spilled"],
        "sw_var_peak_energy": output["sw_var_peak_spilled"],
        "cusum_max_energy": output["cusum_max_spilled"],
    }


def _logprob_features(top_k_logprobs) -> dict:
    logprob = np.asarray(top_k_logprobs["logprobs"], dtype=float)
    top1, top2 = logprob[:, 0], logprob[:, 1]
    probability = np.exp(logprob)
    probability /= probability.sum(axis=1, keepdims=True) + 1e-12
    entropy = -(probability * np.log(probability + 1e-12)).sum(axis=1)
    surprisal = -np.log(probability + 1e-12)
    mean_surprisal = (probability * surprisal).sum(axis=1, keepdims=True)
    varentropy = (probability * (surprisal - mean_surprisal) ** 2).sum(axis=1)
    renyi2 = -np.log((probability ** 2).sum(axis=1) + 1e-12)
    tail_k = min(5, logprob.shape[1])
    tail_mass = np.clip(1.0 - probability[:, :tail_k].sum(axis=1), 0.0, 1.0)
    return {
        "mean_top1_logprob": float(np.mean(top1)),
        "logprob_margin": float(np.mean(top1 - top2)),
        "mean_logprob_entropy": float(np.mean(entropy)),
        "varentropy": float(np.mean(varentropy)),
        "renyi_entropy_2": float(np.mean(renyi2)),
        "topk_tail_mass": float(np.mean(tail_mass)),
    }


@dataclass(frozen=True)
class SafeCandidate:
    environment_id: str
    dataset_revision: str
    split: str
    dataset_family: str
    item_group_id: str
    candidate_ordinal: int
    canonical_item_id: str
    content_hash: str
    features: np.ndarray
    trace_length: int
    unexpected_candidate_keys: tuple[str, ...]


@dataclass(frozen=True)
class SourceExpectation:
    environment_id: str
    dataset: str
    split: str
    dataset_family: str
    expected_admitted_count: int
    admission_mode: str
    source_sha256: str
    source_size: int
    source_mtime: str
    manifest_sha256: str


@dataclass(frozen=True)
class FrozenSourceSpec:
    environment_id: str
    dataset: str
    split: str
    dataset_family: str
    expected_admitted_count: int
    admission_mode: str
    raw_relative_path: str
    source_sha256: str
    source_size: int
    manifest_sha256: str


@dataclass(frozen=True)
class RawSourceArtifact:
    rows: Mapping
    source_sha256: str
    source_size: int
    source_mtime: str
    manifest_sha256: str


@dataclass(frozen=True)
class TargetFreeBoundary:
    primary_rows: tuple[SafeCandidate, ...]
    all_admitted_rows: tuple[SafeCandidate, ...]
    audit: dict


# Immutable registry copied from the hash-frozen A0 audit and the committed LFS
# pointer OIDs.  Production boundary construction never accepts caller-defined
# source metadata.
FROZEN_A0_SOURCE_SPECS = (
    FrozenSourceSpec("ars_gsm8k_r1distill8b", "gsm8k", "test", "gsm8k", 500, "complete_h16", "dataset_cache/repgrid/ars_gsm8k_r1distill8b/raw_gsm8k_T0.0.pkl", "ae33ac6139828c1a69fb8887b950cc956323707d537d10c20bebf1108d8f2dc8", 189353848, "bf578c9a22b60438f9f54e76cc1d08f267508a50110aa1f43d215a622e2e5292"),
    FrozenSourceSpec("internalstates_gsm8k_qwen25_7b", "gsm8k", "test", "gsm8k", 500, "complete_h16", "dataset_cache/repgrid/internalstates_gsm8k_qwen25_7b/raw_gsm8k_T0.8.pkl", "7ff68214158c740a88baf3959dff6484b68f43f40e6f6096ca6163fb64b5f82c", 146217844, "0951afa2aa81c7a8f6c52f4ba6077eadc40261fca11ce231a5dee5c87a3ca4bf"),
    FrozenSourceSpec("lapeigvals_gsm8k_llama3b", "gsm8k", "test", "gsm8k", 1319, "complete_h16", "dataset_cache/repgrid/lapeigvals_gsm8k_llama3b/raw_gsm8k_T1.0.pkl", "595310c86caf867978e261d563de1af60ca782e8981491fa0fba646524af1270", 261092570, "0e33912a26cd0638e5f6e2f89becfc66f668661a1f6ccf386af88d7a06f3cee5"),
    FrozenSourceSpec("lapeigvals_gsm8k_llama8b", "gsm8k", "test", "gsm8k", 500, "complete_h16", "dataset_cache/repgrid/lapeigvals_gsm8k_llama8b/raw_gsm8k_T1.0.pkl", "6ec52c7af5306a48464ab58f96d5b3f31029a064ccfc2943c049275b9383aa88", 111563114, "ce86ae1174909dcb0115b61a2479f9ba52c431142c904181b1126260fb16c809"),
    FrozenSourceSpec("lapeigvals_gsm8k_mistral24b", "gsm8k", "test", "gsm8k", 1319, "complete_h16", "dataset_cache/repgrid/lapeigvals_gsm8k_mistral24b/raw_gsm8k_T1.0.pkl", "881dfabbbe48a2af4d756483c6800f6d1674d04b13a1664b9765f52e7a36f23c", 285037844, "5780f5b15e53a55c5330af7b8a70834c55f7f1a322e7c835f1c328181eb4b0f8"),
    FrozenSourceSpec("lapeigvals_gsm8k_nemo", "gsm8k", "test", "gsm8k", 1319, "complete_h16", "dataset_cache/repgrid/lapeigvals_gsm8k_nemo/raw_gsm8k_T1.0.pkl", "4b772b6a47d5b070511b863aebbec7fdbf25ab5b58a2f4d13de18805c133eff0", 302055723, "fffb99dd235b74b8bdf50a926f6d7b74c39131a91911d0c51e886fa8281b98e6"),
    FrozenSourceSpec("lapeigvals_gsm8k_phi35", "gsm8k", "test", "gsm8k", 1319, "complete_h16", "dataset_cache/repgrid/lapeigvals_gsm8k_phi35/raw_gsm8k_T1.0.pkl", "0cb4b31f13fb59f34f786db1d931f1d99daee71a0cfa10402e523a859477dde8", 344193497, "d7693a5ebec7ae5a3e58fc4ff9bf7fef08a5e6c2c8376d328408dbf360f399b1"),
    FrozenSourceSpec("noise_gsm8k_mistral7b", "gsm8k", "test", "gsm8k", 1319, "complete_h16", "dataset_cache/repgrid/noise_gsm8k_mistral7b/raw_gsm8k_T1.0.pkl", "9c80391981959c8196f0816db0caca818e4828a5cfbcd31bafb4e9271af93ccf", 373510793, "33ced15503844eb2ea4b7fbb03e2dd03d4f6ece2b99ec883deca6ed77e0c697b"),
    FrozenSourceSpec("noise_gsm8k_phi3mini", "gsm8k", "test", "gsm8k", 1319, "complete_h16", "dataset_cache/repgrid/noise_gsm8k_phi3mini/raw_gsm8k_T1.0.pkl", "10ccb627b29021b9f20757b500285404215097c858cfcea0b2e20c89f8fae5d5", 297124222, "c1307b4813225cc5843fde410020bf66a606ea67966b3678993db6438a1e3817"),
    FrozenSourceSpec("trace_gsm8k_llama8b_k10", "gsm8k", "test", "gsm8k", 5000, "complete_h16", "dataset_cache/repgrid/trace_gsm8k_llama8b_k10/raw_gsm8k_T1.0.pkl", "eae1924f5b7790e74da5721b6e99ef1792627a8abf6400eee0ae42a605aa4281", 1119977097, "f3817da665454da910a7dfb433025959610e17e7395a56ca23565eb5defd9495"),
    FrozenSourceSpec("losnet_hotpotqa_mistral7b", "hotpotqa", "validation", "hotpotqa", 500, "complete_h16", "dataset_cache/repgrid/losnet_hotpotqa_mistral7b/raw_hotpotqa_T0.0.pkl", "0bf2a99e7a368fdab45b2bb490418ab1f9323339fe93b3d64cd33db4be263ce7", 944868365, "c10fc07b392dc17b0afdeca65802c6a7176282525ab395060186460e090f30d0"),
    FrozenSourceSpec("math500_dsmath7b", "math500", "test", "math500", 300, "complete_h16", "dataset_cache/repgrid/math500_dsmath7b/raw_math500_T1.5.pkl", "dc03941c3713227f0afaacc6f60b634c6b907d69d676a5201fd8ea7621d71f86", 134163160, "c36225a59ee15de11864934213eb456a842ab0a62e5760801238952ad3be298a"),
    FrozenSourceSpec("math500_qwenmath7b", "math500", "test", "math500", 300, "complete_h16", "dataset_cache/repgrid/math500_qwenmath7b/raw_math500_T1.5.pkl", "34e4c6c7c23b2694f75f72ba247cb00d00f82437c6e6a201369173c6c83c5e8b", 399256006, "89d6d31356fffce0e623845379b61053af6fb29c25bc6b309e7160c96d6efe89"),
    FrozenSourceSpec("math500_r1distill8b", "math500", "test", "math500", 300, "complete_h16", "dataset_cache/repgrid/math500_r1distill8b/raw_math500_T1.5.pkl", "da2da9c73f6d3a5c658dad2d3f9d4348ac9ee5272c2cf64eb87157cfdd40ba6c", 395029741, "04f602131b9394432ce6a9a8a27f6794d474aa13af294a62b271e6c8df41619b"),
    FrozenSourceSpec("math500_r1distill8b_mn4096", "math500", "test", "math500", 300, "complete_h16", "dataset_cache/repgrid/math500_r1distill8b_mn4096/raw_math500_T1.5.pkl", "7d1f8376f6d25004ca532af89a28bbc446c4a36025d877a89a2c07b8d122face", 684761665, "2f2b02d0b155cbbfc79b8f9fe4adc5f04b1e661cb7e31cf217e5540cb935cb8d"),
    FrozenSourceSpec("trace_math500_qwenmath15b_k10", "math500", "test", "math500", 3000, "complete_h16", "dataset_cache/repgrid/trace_math500_qwenmath15b_k10/raw_math500_T1.0.pkl", "fff54f9e7078316f936ea933061b04d53367b297bf26c51088c6c9567882c0c3", 1283184154, "53a0b436c58c63e216b785fce5a3fc988788fea098f774e8f23561c25e46db24"),
    FrozenSourceSpec("se_nq_open_llama8b", "nq_open", "validation", "nq_open", 8460, "complete_h16", "dataset_cache/repgrid/se_nq_open_llama8b/raw_nq_open_T0.5.pkl", "2ac8a95ad66f20c914fcfd1cd53740d6ca4ac631555362c8cf1c11aacf1a4548", 248018947, "68a8ee7568b03b855860318fc68a71a3c4bdd6849c1a4998b4f9e0bccb684a6d"),
    FrozenSourceSpec("sciq_llama8b", "sciq", "validation", "sciq", 198, "complete_h16", "dataset_cache/repgrid/sciq_llama8b/raw_sciq_T1.0.pkl", "f0085556f208c64a85d8085734f9423cc391531603b6a4c1cd20443bd25cc1b1", 13040824, "a9117e4556ae140b5ebccfda90daa34c1a6c811fb60f55a17676cf7980f4f8ea"),
    FrozenSourceSpec("se_squad_v2_llama8b", "squad_v2", "validation", "squad_v2", 2933, "complete_h16", "dataset_cache/repgrid/se_squad_v2_llama8b/raw_squad_v2_T0.5.pkl", "d3e457da44ba79258fc4319bfaf8dfb068954d2a26d2e5d88b2da35227cca088", 76891110, "8c3b4ca838f6306fdabc8738db67745b675aa5ff5c54c99926e062924e65b3eb"),
    FrozenSourceSpec("epr_triviaqa_mistral24b", "trivia_qa_wiki", "validation", "triviaqa", 621, "complete_h16", "dataset_cache/repgrid/epr_triviaqa_mistral24b/raw_trivia_qa_wiki_T1.0.pkl", "597ab481bf35317247abd1f61003cd405df0fd59491e207533218aa7365012b8", 16453637, "69382282fc47e88950b88446c08cf9a6f7686f8345593574ac560b3748f27cb8"),
    FrozenSourceSpec("seiclr_triviaqa_opt30b", "trivia_qa_rougel", "validation", "triviaqa", 5000, "cropped_all_rows", "dataset_cache/repgrid/seiclr_triviaqa_opt30b/raw_trivia_qa_rougel_T0.5.pkl", "91c70610dafbb3c76af42135716ba97a129b912703479e671d70db81af9097ed", 269574582, "84632855439a99dc9d18cd668e7b4038a6dac3791ad5217e9817234278071f99"),
    FrozenSourceSpec("semenergy_triviaqa_qwen3_8b", "trivia_qa", "validation", "triviaqa", 4392, "complete_h16", "dataset_cache/repgrid/semenergy_triviaqa_qwen3_8b/raw_trivia_qa_T0.6.pkl", "fa33eb050e481dea9afc82ad82cb243c415ae8ee51b009bb9980b09b621d1945", 62044585, "208decb84431a30609af1f4c4e60b487281a5b64fdffed400cf38d62e265f19c"),
    FrozenSourceSpec("truthfulqa_llama8b", "truthfulqa", "validation", "truthfulqa", 7633, "complete_h16", "dataset_cache/repgrid/truthfulqa_llama8b/raw_truthfulqa_T0.5.pkl", "3dac66f177aa825ccce53849b4e4d8cfc1e43f6c93ad664a4c01851fc90e3077", 435424434, "4fd419712a945fff6207e106bb81d6bf67873e8b51dd982748b53efd84f895a3"),
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_raw_sources(repo_root: str | Path) -> tuple[dict, tuple[SourceExpectation, ...]]:
    """Hash/metadata verify every frozen A0 source before any pickle executes."""
    root = Path(repo_root).resolve()
    verified = []
    for spec in FROZEN_A0_SOURCE_SPECS:
        raw_path = (root / spec.raw_relative_path).resolve()
        expected_parent = (root / "dataset_cache" / "repgrid" / spec.environment_id).resolve()
        if raw_path.parent != expected_parent:
            raise ValueError("CLOSE_INVALID_SOURCE_PATH")
        manifest_path = expected_parent / "manifest.json"
        if not raw_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"missing frozen A0 source: {spec.environment_id}")
        manifest_hash = _sha256_file(manifest_path)
        if manifest_hash != spec.manifest_sha256:
            raise ValueError("CLOSE_SOURCE_MANIFEST_MISMATCH")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != spec.dataset or manifest.get("split") != spec.split:
            raise ValueError("CLOSE_INVALID_SOURCE_ROSTER: dataset/split mismatch")
        cells = manifest.get("cells")
        if not isinstance(cells, list) or len(cells) != 1:
            raise ValueError("CLOSE_INVALID_SOURCE_ROSTER: expected one frozen cell")
        if cells[0].get("pkl") != raw_path.name:
            raise ValueError("CLOSE_INVALID_SOURCE_ROSTER: raw filename mismatch")
        source_size = raw_path.stat().st_size
        if source_size != spec.source_size:
            raise ValueError("CLOSE_SOURCE_ARTIFACT_MISMATCH: byte size")
        source_hash = _sha256_file(raw_path)
        if source_hash != spec.source_sha256:
            raise ValueError("CLOSE_SOURCE_ARTIFACT_MISMATCH: sha256")
        source_mtime = "noncanonical-filesystem-mtime-excluded"
        verified.append((spec, raw_path, source_hash, source_size,
                         source_mtime, manifest_hash))

    # Atomic boundary rule: no pickle executes until all 23 immutable manifests
    # and source byte streams have verified.
    artifacts, expectations = {}, []
    for spec, raw_path, source_hash, source_size, source_mtime, manifest_hash in verified:
        with raw_path.open("rb") as handle:
            rows = pickle.load(handle)
        artifacts[spec.environment_id] = RawSourceArtifact(
            rows=rows, source_sha256=source_hash, source_size=source_size,
            source_mtime=source_mtime, manifest_sha256=manifest_hash,
        )
        expectations.append(SourceExpectation(
            environment_id=spec.environment_id, dataset=spec.dataset,
            split=spec.split, dataset_family=spec.dataset_family,
            expected_admitted_count=spec.expected_admitted_count,
            admission_mode=spec.admission_mode, source_sha256=source_hash,
            source_size=source_size, source_mtime=source_mtime,
            manifest_sha256=manifest_hash,
        ))
    return artifacts, tuple(expectations)


def normalize_question(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("top-level question must be a string")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("normalized question must not be empty")
    return normalized


def canonical_item_id(dataset_revision: str, split: str, item_group_id: str) -> str:
    return _sha256(f"{dataset_revision}\0{split}\0{item_group_id}")


def question_content_hash(question: str) -> str:
    return _sha256(normalize_question(question))


def dataset_family(dataset: str) -> str:
    value = str(dataset).lower()
    if value == "gsm8k":
        return "gsm8k"
    if value == "math500":
        return "math500"
    if value.startswith("trivia_qa"):
        return "triviaqa"
    allowed = {
        "hotpotqa", "nq_open", "sciq", "squad_v2", "truthfulqa",
    }
    if value not in allowed:
        raise KeyError(f"unregistered A5 dataset family: {dataset}")
    return value


def canonical_revision(dataset: str, split: str) -> str:
    family = dataset_family(dataset)
    # The split is hashed separately by ``canonical_item_id``.  Returning the
    # frozen GSM8K/MATH500 revision makes only their known source aliases share
    # an ID namespace.  TriviaQA variants share a bootstrap family but remain
    # distinct manifest revisions, as preregistered.
    if family in {"gsm8k", "math500"}:
        return family
    return str(dataset)


def _validated_telemetry(candidate: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping]:
    """Return aligned finite telemetry without touching non-whitelisted values."""
    missing = [name for name in TELEMETRY_KEYS if name not in candidate or candidate[name] is None]
    if missing:
        raise KeyError(f"missing A5 telemetry: {missing}")
    entropy = candidate["token_entropies"]
    spilled = candidate["token_spilled_energies"]
    logsum = candidate["token_logsumexp"]
    topk = candidate["top_k_logprobs"]
    entropy_array = np.asarray(entropy, dtype=float)
    spilled_array = np.asarray(spilled, dtype=float)
    logsum_array = np.asarray(logsum, dtype=float)
    if not isinstance(topk, Mapping) or "logprobs" not in topk:
        raise TypeError("top_k_logprobs must contain a logprobs matrix")
    topk_array = np.asarray(topk["logprobs"], dtype=float)
    if entropy_array.ndim != 1 or spilled_array.ndim != 1 or logsum_array.ndim != 1:
        raise ValueError("A5 telemetry streams must be one-dimensional")
    if topk_array.ndim != 2 or topk_array.shape[1] < 2:
        raise ValueError("top-k logprobs must have shape (T,K>=2)")
    lengths = (len(entropy_array), len(spilled_array), len(logsum_array), len(topk_array))
    if min(lengths) <= 0 or len(set(lengths)) != 1:
        raise ValueError(f"A5 telemetry token lengths disagree: {lengths}")
    if not all(np.isfinite(value).all() for value in (
        entropy_array, spilled_array, logsum_array, topk_array
    )):
        raise ValueError("A5 telemetry contains non-finite values")
    return entropy_array, spilled_array, logsum_array, topk


def _cropped_telemetry_only(candidate: Mapping) -> dict:
    """Isolated inherited A0 span crop; emits telemetry and no text/target field."""
    entropy, spilled, logsum, topk = _validated_telemetry(candidate)
    selected = answer_token_slice(candidate)
    if selected is None:
        lo, hi = 0, len(entropy)
    else:
        lo, hi = selected
    ids = np.asarray(topk.get("ids")) if "ids" in topk else None
    output_topk = {"logprobs": np.asarray(topk["logprobs"])[lo:hi]}
    if ids is not None and ids.shape[0] >= hi:
        output_topk["ids"] = ids[lo:hi]
    return {
        "token_entropies": entropy[lo:hi],
        "token_spilled_energies": spilled[lo:hi],
        "token_logsumexp": logsum[lo:hi],
        "top_k_logprobs": output_topk,
    }


def _feature_row(candidate: Mapping, *, allow_short: bool = False) -> tuple[np.ndarray, int]:
    # Access only the four explicitly whitelisted telemetry fields.  In
    # particular, membership scans over candidate values are prohibited.
    entropy_array, spilled_array, logsum_array, topk = _validated_telemetry(candidate)
    values = extract_all_features(
        entropy_array, spilled_energies=spilled_array, allow_short=allow_short
    ) or {}
    values.update(_energy_features_from_logsumexp(logsum_array))
    values.update(_logprob_features(topk))
    output = np.asarray([values.get(name, np.nan) for name in CORE_FEATURES], dtype=float)
    if output.shape != (len(CORE_FEATURES),) or not np.isfinite(output).all():
        missing_features = [
            name for name, value in zip(CORE_FEATURES, output) if not np.isfinite(value)
        ]
        raise ValueError(f"nonfinite A5 core feature(s): {missing_features}")
    return output, int(len(entropy_array))


def _a0_admitted(candidate: Mapping, admission_mode: str) -> bool:
    if admission_mode == "cropped_all_rows":
        return True
    if admission_mode != "complete_h16":
        raise ValueError(f"unknown frozen A0 admission mode: {admission_mode}")
    features = extract_all_features(
        candidate["token_entropies"],
        spilled_energies=candidate["token_spilled_energies"],
        allow_short=False,
    )
    # This is the exact label-free complete-case population rule used by the
    # A0 repgrid feature cache for every non-cropped source.
    return features is not None and all(
        name in features and np.isfinite(features[name])
        for name in (
            "epr", "trace_length", "spectral_entropy", "low_band_power",
            "high_band_power", "hl_ratio", "dominant_freq",
            "spectral_centroid", "stft_max_high_power",
            "stft_spectral_entropy", "rpdi", "sw_var_peak", "pe_mean",
            "hurst_exponent", "cusum_max", "cusum_shift_idx",
        )
    )


def sanitize_source_row(
    *,
    environment_id: str,
    dataset: str,
    split: str,
    item_group_id: object,
    row: Mapping,
    allow_short_features: bool = False,
) -> tuple[SafeCandidate, ...]:
    """Copy only A5-permitted values out of one raw source row.

    `row` may contain prompts, gold objects and labels.  Only top-level
    `question` and `candidates` are indexed; candidate access is limited to the
    telemetry whitelist.
    """
    if not isinstance(row, Mapping):
        raise TypeError("source row must be a mapping")
    question = row["question"]
    candidates = row["candidates"]
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("source candidates must be a sequence")
    item_group_id = str(item_group_id)
    revision = canonical_revision(dataset, split)
    family = dataset_family(dataset)
    canonical = canonical_item_id(revision, str(split), item_group_id)
    content = question_content_hash(question)
    output = []
    for ordinal, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError("candidate must be a mapping")
        features, length = _feature_row(candidate, allow_short=allow_short_features)
        unexpected = tuple(sorted(set(candidate) - set(TELEMETRY_KEYS)))
        output.append(SafeCandidate(
            environment_id=str(environment_id),
            dataset_revision=revision,
            split=str(split),
            dataset_family=family,
            item_group_id=item_group_id,
            candidate_ordinal=int(ordinal),
            canonical_item_id=canonical,
            content_hash=content,
            features=features,
            trace_length=length,
            unexpected_candidate_keys=unexpected,
        ))
    return tuple(output)


def connected_content_groups(rows: Sequence[SafeCandidate]) -> dict[tuple[str, str, int], str]:
    """Join records transitively when canonical ID or content hash agrees."""
    n = len(rows)
    boundary_keys = [
        (row.environment_id, row.item_group_id, row.candidate_ordinal) for row in rows
    ]
    if len(set(boundary_keys)) != len(boundary_keys):
        raise ValueError("duplicate A5 boundary key")
    hashes_by_canonical: dict[str, set[str]] = {}
    for row in rows:
        hashes_by_canonical.setdefault(row.canonical_item_id, set()).add(row.content_hash)
    conflicts = {
        canonical: sorted(hashes)
        for canonical, hashes in hashes_by_canonical.items()
        if len(hashes) != 1
    }
    if conflicts:
        raise ValueError(
            "CLOSE_INVALID_GLOBAL_ITEM_BOUNDARY: canonical item IDs map to "
            f"conflicting normalized-question hashes: {conflicts}"
        )
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    by_canonical: dict[str, int] = {}
    by_content: dict[str, int] = {}
    for index, row in enumerate(rows):
        for table, value in (
            (by_canonical, row.canonical_item_id),
            (by_content, row.content_hash),
        ):
            if value in table:
                union(index, table[value])
            else:
                table[value] = index
    members: dict[int, list[int]] = {}
    for index in range(n):
        members.setdefault(find(index), []).append(index)
    component_hash = {
        root: _sha256("\0".join(sorted(
            {rows[index].canonical_item_id for index in indexes}
            | {rows[index].content_hash for index in indexes}
        )))
        for root, indexes in members.items()
    }
    return {
        (row.environment_id, row.item_group_id, row.candidate_ordinal): component_hash[find(index)]
        for index, row in enumerate(rows)
    }


def select_primary_responses(
    rows: Sequence[SafeCandidate], content_groups: Mapping[tuple[str, str, int], str]
) -> tuple[SafeCandidate, ...]:
    grouped: dict[tuple[str, str], list[SafeCandidate]] = {}
    for row in rows:
        key = (row.environment_id, row.item_group_id, row.candidate_ordinal)
        component = str(content_groups[key])
        grouped.setdefault((row.environment_id, component), []).append(row)

    output = []
    for (environment, component), candidates in sorted(grouped.items()):
        def selection_key(row: SafeCandidate):
            payload = (
                "A5-primary-response\0" + environment + "\0" + component + "\0"
                + row.item_group_id + "\0" + str(row.candidate_ordinal)
            )
            return _sha256(payload), row.item_group_id, row.candidate_ordinal
        output.append(min(candidates, key=selection_key))
    return tuple(output)


def sha256_ordered_boundary_keys(rows: Sequence[SafeCandidate]) -> str:
    payload = [
        [row.environment_id, row.item_group_id, row.candidate_ordinal]
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_target_free_boundary(
    raw_sources: Mapping[str, RawSourceArtifact],
    expectations: Sequence[SourceExpectation],
) -> TargetFreeBoundary:
    return _build_target_free_boundary(raw_sources, expectations, enforce_frozen=True)


def _build_target_free_boundary(
    raw_sources: Mapping[str, RawSourceArtifact],
    expectations: Sequence[SourceExpectation],
    *, enforce_frozen: bool,
) -> TargetFreeBoundary:
    """Build and audit the exact no-target A5 source boundary.

    This routine is intentionally in-memory so a caller can resolve Drive/LFS
    artifacts and verify their file hashes before unpickling them.  It never
    accepts labels or a prepared archive containing label arrays.
    """
    expectation_by_id = {value.environment_id: value for value in expectations}
    if len(expectation_by_id) != len(expectations):
        raise ValueError("duplicate source expectation")
    if (
        set(raw_sources) != set(expectation_by_id)
        or set(raw_sources) != set(A0_SOURCE_ENVIRONMENTS)
    ):
        raise ValueError("CLOSE_INVALID_SOURCE_ROSTER: expected exact 23 A0 environments")
    if enforce_frozen:
        frozen = {value.environment_id: value for value in FROZEN_A0_SOURCE_SPECS}
        for expectation in expectations:
            spec = frozen.get(expectation.environment_id)
            fixed = (
                expectation.dataset, expectation.split, expectation.dataset_family,
                expectation.expected_admitted_count, expectation.admission_mode,
                expectation.source_sha256, expectation.source_size,
                expectation.manifest_sha256,
            )
            required = (
                spec.dataset, spec.split, spec.dataset_family,
                spec.expected_admitted_count, spec.admission_mode,
                spec.source_sha256, spec.source_size, spec.manifest_sha256,
            ) if spec is not None else None
            if fixed != required:
                raise ValueError("CLOSE_INVALID_SOURCE_ROSTER: frozen registry mismatch")
    admitted_rows: list[SafeCandidate] = []
    source_audits = []
    normalized_by_source: dict[str, dict[str, str]] = {}
    canonical_by_source: dict[str, set[str]] = {}
    for environment_id in sorted(raw_sources):
        expectation = expectation_by_id[environment_id]
        if dataset_family(expectation.dataset) != expectation.dataset_family:
            raise ValueError("CLOSE_INVALID_SOURCE_ROSTER: dataset family mismatch")
        artifact = raw_sources[environment_id]
        if not isinstance(artifact, RawSourceArtifact):
            raise TypeError("raw source must carry verified artifact metadata")
        if (
            artifact.source_sha256 != expectation.source_sha256
            or artifact.source_size != expectation.source_size
            or artifact.source_mtime != expectation.source_mtime
            or artifact.manifest_sha256 != expectation.manifest_sha256
        ):
            raise ValueError("CLOSE_SOURCE_ARTIFACT_MISMATCH")
        source = artifact.rows
        if not isinstance(source, Mapping):
            raise TypeError("raw source must be a mapping")
        source_rows = []
        id_to_content = {}
        assigned_problem_rows = 0
        unexpected_counts: dict[str, int] = {}
        for problem_key in sorted(source, key=lambda value: str(value)):
            raw_row = source[problem_key]
            # Question/ID assignment coverage is audited before admission.
            try:
                normalized = normalize_question(raw_row["question"])
            except (KeyError, TypeError, ValueError):
                continue
            assigned_problem_rows += 1
            revision = canonical_revision(expectation.dataset, expectation.split)
            canonical = canonical_item_id(revision, expectation.split, str(problem_key))
            content = question_content_hash(normalized)
            id_to_content[canonical] = content
            candidates = raw_row.get("candidates")
            if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                raise TypeError("source candidates must be a sequence")
            for ordinal, candidate in enumerate(candidates):
                feature_candidate = (
                    _cropped_telemetry_only(candidate)
                    if expectation.admission_mode == "cropped_all_rows"
                    else candidate
                )
                # Validate the aligned four-stream telemetry even for a short row
                # that the frozen complete-case rule will exclude.
                _validated_telemetry(feature_candidate)
                if not _a0_admitted(feature_candidate, expectation.admission_mode):
                    continue
                safe = sanitize_source_row(
                    environment_id=environment_id,
                    dataset=expectation.dataset,
                    split=expectation.split,
                    item_group_id=problem_key,
                    row={"question": normalized, "candidates": [feature_candidate]},
                    allow_short_features=(expectation.admission_mode == "cropped_all_rows"),
                )[0]
                # sanitize_source_row sees a one-element list; restore the raw
                # candidate ordinal without inspecting any candidate target.
                safe = SafeCandidate(**{**safe.__dict__, "candidate_ordinal": int(ordinal)})
                for key in safe.unexpected_candidate_keys:
                    unexpected_counts[key] = unexpected_counts.get(key, 0) + 1
                source_rows.append(safe)
        assignment_fraction = assigned_problem_rows / max(len(source), 1)
        if assignment_fraction < 0.999:
            raise ValueError("CLOSE_INADEQUATE_GLOBAL_GROUP_BOUNDARY")
        if len(source_rows) != expectation.expected_admitted_count:
            raise ValueError(
                "CLOSE_A0_POPULATION_MISMATCH: "
                f"{environment_id} admitted {len(source_rows)} expected "
                f"{expectation.expected_admitted_count}"
            )
        normalized_by_source[environment_id] = id_to_content
        canonical_by_source[environment_id] = set(id_to_content)
        admitted_rows.extend(source_rows)
        source_audits.append({
            "environment_id": environment_id,
            "dataset_revision": canonical_revision(expectation.dataset, expectation.split),
            "dataset_family": expectation.dataset_family,
            "split": expectation.split,
            "raw_problem_count": len(source),
            "assigned_problem_count": assigned_problem_rows,
            "assignment_fraction": assignment_fraction,
            "admitted_candidate_count": len(source_rows),
            "expected_admitted_count": expectation.expected_admitted_count,
            "ordered_admitted_key_sha256": sha256_ordered_boundary_keys(source_rows),
            "source_sha256": expectation.source_sha256,
            "source_size": expectation.source_size,
            "source_mtime": expectation.source_mtime,
            "manifest_sha256": expectation.manifest_sha256,
            "unexpected_candidate_key_counts": unexpected_counts,
            "labels_accessed": False,
        })

    overlap_rows = []
    ids = sorted(raw_sources)
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1:]:
            left_expectation, right_expectation = expectation_by_id[left], expectation_by_id[right]
            shared_revision = canonical_revision(
                left_expectation.dataset, left_expectation.split
            ) == canonical_revision(right_expectation.dataset, right_expectation.split)
            expected_ids = canonical_by_source[left] & canonical_by_source[right]
            matching_ids = {
                value for value in expected_ids
                if normalized_by_source[left][value] == normalized_by_source[right][value]
            }
            if shared_revision and matching_ids != expected_ids:
                raise ValueError("CLOSE_INVALID_GLOBAL_ITEM_BOUNDARY")
            overlap_rows.append({
                "source_a": left,
                "source_b": right,
                "shared_canonical_revision": shared_revision,
                "expected_canonical_id_overlap": len(expected_ids),
                "matching_question_hash_overlap": len(matching_ids),
            })

    content_groups = connected_content_groups(admitted_rows)
    primary = select_primary_responses(admitted_rows, content_groups)
    observed_components_by_source = {
        environment_id: {
            content_groups[(row.environment_id, row.item_group_id, row.candidate_ordinal)]
            for row in admitted_rows if row.environment_id == environment_id
        }
        for environment_id in ids
    }
    for overlap in overlap_rows:
        observed = len(
            observed_components_by_source[overlap["source_a"]]
            & observed_components_by_source[overlap["source_b"]]
        )
        overlap["observed_shared_components"] = observed
        if overlap["shared_canonical_revision"] and (
            observed < overlap["expected_canonical_id_overlap"]
        ):
            raise ValueError("CLOSE_INVALID_GLOBAL_ITEM_BOUNDARY")
    audit = {
        "source_count": len(expectations),
        "source_roster": ids,
        "source_rows": source_audits,
        "overlap_rows": overlap_rows,
        "all_admitted_count": len(admitted_rows),
        "primary_structural_count": len(primary),
        "all_admitted_key_sha256": sha256_ordered_boundary_keys(admitted_rows),
        "primary_key_sha256": sha256_ordered_boundary_keys(primary),
        "labels_accessed": False,
    }
    return TargetFreeBoundary(tuple(primary), tuple(admitted_rows), audit)


def assert_no_target_fields(payload: Mapping) -> None:
    offending = sorted(str(key) for key in payload if TARGET_LIKE.search(str(key)))
    if offending:
        raise ValueError(f"target-like keys reached public A5 boundary: {offending}")


__all__ = [
    "CORE_FEATURES",
    "A0_SOURCE_ENVIRONMENTS",
    "RawSourceArtifact",
    "FrozenSourceSpec",
    "FROZEN_A0_SOURCE_SPECS",
    "SafeCandidate",
    "SourceExpectation",
    "TargetFreeBoundary",
    "assert_no_target_fields",
    "canonical_item_id",
    "canonical_revision",
    "build_target_free_boundary",
    "load_frozen_raw_sources",
    "connected_content_groups",
    "dataset_family",
    "normalize_question",
    "question_content_hash",
    "sanitize_source_row",
    "select_primary_responses",
    "sha256_ordered_boundary_keys",
]
