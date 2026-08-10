#!/usr/bin/env python3
"""Regrade SemGrad generation caches with the official BEM model.

The implementation follows SemGrad's BEM evaluator: score every reference
answer, take the maximum score for each candidate, and call the candidate
correct when the score is at least the registered threshold.

The input pickle is never modified. The script writes a scored copy and a
JSON manifest beside it (or under --output-dir).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import platform
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SEMGRAD_COMMIT = "118b6949f9641df3872caa7ad65a797f4ae28d63"
VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
VOCAB_URL = (
    "https://raw.githubusercontent.com/mingdali6717/SemGrad/"
    f"{SEMGRAD_COMMIT}/uncertainty/generation_evaluation/metrics/vocab.txt"
)
BEM_MODEL = "https://tfhub.dev/google/answer_equivalence/bem/1"
MAX_LENGTH = 512


class RestrictedUnpickler(pickle.Unpickler):
    """Load the cluster cache without allowing arbitrary pickle globals."""

    _ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked unsafe pickle global: {module}.{name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path) -> dict[Any, Any]:
    with path.open("rb") as handle:
        value = RestrictedUnpickler(handle).load()
    if not isinstance(value, dict):
        raise ValueError("Expected the cache root to be a dictionary")
    return value


def download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == expected_sha256:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    actual = sha256_file(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded vocabulary hash mismatch: expected {expected_sha256}, got {actual}"
        )
    temporary.replace(destination)
    return destination


def resolve_vocab(explicit_path: Path | None, asset_dir: Path) -> Path:
    if explicit_path is not None:
        if sha256_file(explicit_path) != VOCAB_SHA256:
            raise ValueError(
                "The supplied vocabulary is not the pinned SemGrad BEM vocabulary "
                f"(expected SHA-256 {VOCAB_SHA256})"
            )
        return explicit_path
    return download_verified(VOCAB_URL, asset_dir / "vocab.txt", VOCAB_SHA256)


@dataclass(frozen=True)
class CandidateRef:
    problem_id: Any
    candidate_index: int
    question: str
    candidate: str
    references: tuple[str, ...]


def iter_candidates(cache: dict[Any, Any]) -> Iterable[CandidateRef]:
    for problem_id, row in cache.items():
        if not isinstance(row, dict):
            raise ValueError(f"Problem {problem_id!r} is not a dictionary")
        question = str(row.get("question") or row.get("gold_row", {}).get("question") or "")
        answers = row.get("gold_row", {}).get("truthful_answers")
        if isinstance(answers, str):
            answers = [answers]
        if not answers or not all(isinstance(answer, str) for answer in answers):
            raise ValueError(f"Problem {problem_id!r} has no valid truthful_answers")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Problem {problem_id!r} has no candidates")
        for candidate_index, candidate_row in enumerate(candidates):
            if not isinstance(candidate_row, dict) or "full_text" not in candidate_row:
                raise ValueError(
                    f"Problem {problem_id!r}, candidate {candidate_index} has no full_text"
                )
            candidate_text = str(candidate_row["full_text"] or "None")
            yield CandidateRef(
                problem_id=problem_id,
                candidate_index=candidate_index,
                question=question,
                candidate=candidate_text,
                references=tuple(answers),
            )


def expand_examples(candidates: Sequence[CandidateRef]) -> tuple[list[dict[str, str]], list[slice]]:
    examples: list[dict[str, str]] = []
    groups: list[slice] = []
    for candidate in candidates:
        start = len(examples)
        examples.extend(
            {
                "question": candidate.question,
                "reference": reference,
                "candidate": candidate.candidate,
            }
            for reference in candidate.references
        )
        groups.append(slice(start, len(examples)))
    return examples, groups


def aggregate_max(scores: Sequence[float], groups: Sequence[slice]) -> list[float]:
    return [float(max(scores[group])) for group in groups]


class BemScorer:
    """TensorFlow implementation matching SemGrad's BemCalculator."""

    def __init__(self, vocab_path: Path, model_path: str, device: str):
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            import tensorflow_text as text
        except ImportError as error:
            raise RuntimeError(
                "BEM dependencies are missing. Install scripts/bem-requirements.in "
                "in a dedicated Python 3.11 environment."
            ) from error

        self.tf = tf
        self.device, self.device_kind = self._choose_device(device)
        with tf.device(self.device):
            self.model = hub.load(model_path)

        vocab_table = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                filename=str(vocab_path),
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            ),
            num_oov_buckets=1,
        )
        self.cls_id, self.sep_id = tf.cast(
            vocab_table.lookup(tf.convert_to_tensor(["[CLS]", "[SEP]"])), tf.int64
        )
        self.tokenizer = text.BertTokenizer(
            vocab_lookup_table=vocab_table,
            token_out_type=tf.int64,
            preserve_unused_token=True,
            lower_case=True,
        )

    def _choose_device(self, requested: str) -> tuple[str, str]:
        gpus = self.tf.config.list_physical_devices("GPU")
        if requested == "gpu":
            if not gpus:
                raise RuntimeError("--device gpu was requested, but TensorFlow sees no GPU")
            return "/GPU:0", "gpu"
        if requested == "cpu":
            return "/CPU:0", "cpu"
        return ("/GPU:0", "gpu") if gpus else ("/CPU:0", "cpu")

    @staticmethod
    def _pad(values: np.ndarray) -> np.ndarray:
        return np.append(values, np.zeros(MAX_LENGTH - values.shape[-1], np.int32))

    def _bertify(self, examples: Sequence[dict[str, str]]) -> dict[str, np.ndarray]:
        tf = self.tf
        flat = [value for ex in examples for value in (ex["question"], ex["reference"], ex["candidate"])]
        tokens = self.tokenizer.tokenize(flat).merge_dims(1, 2)
        questions = tf.concat([tokens[i : i + 1] for i in range(0, len(flat), 3)], 0)
        references = tf.concat([tokens[i + 1 : i + 2] for i in range(0, len(flat), 3)], 0)
        candidates = tf.concat([tokens[i + 2 : i + 3] for i in range(0, len(flat), 3)], 0)

        shortened = []
        max_content_length = MAX_LENGTH - 4
        for question, reference, candidate in zip(questions, references, candidates):
            overflow = int(question.shape[0] + reference.shape[0] + candidate.shape[0]) - max_content_length
            shortened.append(candidate[:-overflow] if overflow > 0 else candidate)
        candidates = tf.ragged.stack(shortened).with_row_splits_dtype(questions.dtype)
        input_ids, segment_ids = self.tf_text_combine(
            candidates, references, questions
        )
        input_ids = input_ids.numpy()
        segment_ids = segment_ids.numpy()
        return {
            "input_ids": np.stack([self._pad(row) for row in input_ids]),
            "segment_ids": np.stack([self._pad(row) for row in segment_ids]),
        }

    def tf_text_combine(self, candidates: Any, references: Any, questions: Any) -> tuple[Any, Any]:
        import tensorflow_text as text

        return text.combine_segments(
            (candidates, references, questions), self.cls_id, self.sep_id
        )

    def score(self, examples: Sequence[dict[str, str]], batch_size: int) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs = self._bertify(batch)
            with self.tf.device(self.device):
                logits = np.asarray(self.model(inputs), dtype=np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            scores.extend(probabilities[:, 1].tolist())
            print(f"Scored {min(start + batch_size, len(examples))}/{len(examples)} references", flush=True)
        return scores


def write_outputs(
    cache: dict[Any, Any],
    candidates: Sequence[CandidateRef],
    scores: Sequence[float],
    threshold: float,
    input_path: Path,
    output_dir: Path,
    scorer: BemScorer,
    batch_size: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_cache = copy.deepcopy(cache)
    correct = 0
    for candidate, score in zip(candidates, scores):
        row = scored_cache[candidate.problem_id]["candidates"][candidate.candidate_index]
        row["bem_score"] = float(score)
        row["bem_correct"] = bool(score >= threshold)
        correct += int(score >= threshold)

    output_pickle = output_dir / f"{input_path.stem}_bem.pkl"
    with output_pickle.open("wb") as handle:
        pickle.dump(scored_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_pickle.resolve()),
        "output_sha256": sha256_file(output_pickle),
        "candidate_count": len(candidates),
        "reference_pair_count": sum(len(candidate.references) for candidate in candidates),
        "bem_correct_count": correct,
        "bem_incorrect_count": len(candidates) - correct,
        "bem_accuracy": correct / len(candidates),
        "threshold": threshold,
        "aggregation": "maximum BEM score over truthful_answers",
        "original_label_preserved": True,
        "model": BEM_MODEL,
        "semgrad_commit": SEMGRAD_COMMIT,
        "vocab_sha256": VOCAB_SHA256,
        "device": scorer.device,
        "device_kind": scorer.device_kind,
        "batch_size": batch_size,
        "python": sys.version,
        "platform": platform.platform(),
    }
    manifest_path = output_dir / f"{input_path.stem}_bem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_pickle, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Raw SemGrad pickle cache(s)")
    parser.add_argument("--output-dir", type=Path, help="Output directory; default: beside each input")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--model-path", default=BEM_MODEL)
    parser.add_argument("--vocab-path", type=Path)
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(os.environ.get("BEM_ASSET_DIR", Path.home() / ".cache" / "hallucination_detection" / "bem")),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate and summarize inputs without downloading or loading BEM",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    loaded: list[tuple[Path, dict[Any, Any], list[CandidateRef]]] = []
    for input_path in args.inputs:
        cache = load_cache(input_path)
        candidates = list(iter_candidates(cache))
        reference_count = sum(len(candidate.references) for candidate in candidates)
        print(f"{input_path}: {len(candidates)} candidates, {reference_count} candidate-reference pairs")
        loaded.append((input_path, cache, candidates))
    if args.inspect_only:
        return 0

    vocab_path = resolve_vocab(args.vocab_path, args.asset_dir)
    scorer = BemScorer(vocab_path, args.model_path, args.device)
    print(f"Using TensorFlow device {scorer.device}")
    for input_path, cache, candidates in loaded:
        examples, groups = expand_examples(candidates)
        expanded_scores = scorer.score(examples, args.batch_size)
        scores = aggregate_max(expanded_scores, groups)
        output_dir = args.output_dir or input_path.parent
        output_pickle, manifest = write_outputs(
            cache, candidates, scores, args.threshold, input_path, output_dir, scorer, args.batch_size
        )
        print(f"Wrote {output_pickle}")
        print(f"Wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
