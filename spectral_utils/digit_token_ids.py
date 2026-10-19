"""Tokenizer-aware digit token ids for the digit feature family, with loud failure.

The digit channels (digit_alternative, digit_spread, digit_alternative_innovation) look for the ten
single-character tokens '0'..'9' in each position's top-K list. external_banks_v4.DIGIT_IDS hard-codes
15..24, which is correct for the Qwen2.5/Qwen3 and Llama-3 byte-level BPE vocabularies. In the
Mistral/Llama-2 SentencePiece vocabulary (Math-Shepherd's backbone) 15..24 are byte-fallback tokens,
so the digit channels came out constant (results/prm_boosting_generalization_v1/run_v1/RESULTS_README.md,
Amendment A2). Resolve the ids from the run's own tokenizer.json instead, and fail when they disagree.

Self-contained (json + numpy only, no package-relative imports), so the AIRCC bank builders that run
from a frozen code snapshot can load this file by path. No labels are accepted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DIGIT_CHARS = "0123456789"
QWEN_DIGIT_IDS = tuple(range(15, 25))  # the historical external_banks_v4.DIGIT_IDS


def load_vocab(tokenizer_json):
    """token -> id from a Hugging Face tokenizer.json (BPE/WordPiece dict vocab or Unigram list vocab)."""
    model = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))["model"]
    vocab = model["vocab"]
    if isinstance(vocab, dict):
        return {str(k): int(v) for k, v in vocab.items()}
    if isinstance(vocab, list):  # Unigram: [[piece, score], ...], id = position
        return {str(entry[0]): i for i, entry in enumerate(vocab)}
    raise ValueError(f"unsupported tokenizer.json vocab type {type(vocab).__name__} in {tokenizer_json}")


def digit_ids_from_vocab(vocab):
    """Ids of the ten single-character tokens '0'..'9', in digit order. Raises if any is missing."""
    missing = [c for c in DIGIT_CHARS if c not in vocab]
    if missing:
        raise ValueError(f"tokenizer has no single-character token for digit(s) {missing}; "
                         "the digit feature family is undefined for this vocabulary")
    ids = tuple(int(vocab[c]) for c in DIGIT_CHARS)
    if len(set(ids)) != 10:
        raise ValueError(f"digit tokens are not ten distinct ids: {ids}")
    return ids


def resolve_digit_ids(tokenizer_json, expected=None):
    """Digit ids of this tokenizer, plus a provenance record. With ``expected``, raise on any mismatch."""
    path = Path(tokenizer_json)
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found: {path}")
    ids = digit_ids_from_vocab(load_vocab(path))
    if expected is not None and tuple(int(x) for x in expected) != ids:
        raise ValueError(f"digit ids {tuple(expected)} do not match {path}: '0'..'9' are {ids}")
    record = {"tokenizer_json": str(path), "tokenizer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "digit_ids": list(ids), "matches_qwen_15_24": ids == QWEN_DIGIT_IDS}
    return ids, record


def check_digit_channels_vary(values, names, digit_names, tol=1e-12):
    """Raise if any digit channel is constant over the whole cell.

    A constant digit channel on a math benchmark means the digit ids never occur in the top-K lists, the
    signature of a tokenizer mismatch. ``values`` is the [steps, channels] bank matrix, ``names`` its
    channel names.
    """
    x = np.asarray(values, float)
    names = [str(n) for n in names]
    const = [n for n in digit_names if n in names and float(np.ptp(x[:, names.index(n)])) <= tol]
    if const:
        raise ValueError(f"digit channel(s) constant over the whole cell: {const}. "
                         "Check that the digit ids match this run's tokenizer (resolve_digit_ids).")
