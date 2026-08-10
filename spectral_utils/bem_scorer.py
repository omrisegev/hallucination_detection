"""
Offline BEM (Bulian et al. 2022, "Answer Equivalence") scorer.

Reproduces SemGrad's correctness grading exactly (full protocol detail in
data/semgrad_protocol/PROVENANCE.md) WITHOUT the official code's tensorflow_hub /
tensorflow_text dependencies.

Runs LOCALLY, never on the AIRCC cluster: BEM only scores already-generated
(question, reference, candidate) text triples -- it never touches the LLM -- so there is
no reason to fight TensorFlow + tensorflow_hub/tensorflow_text against the B200's sm_100
kernels inside the NGC PyTorch container. A normal CPU or GPU box is enough; a 200-400
sample pilot scores in well under a minute even on CPU.

Reimplements uncertainty/generation_evaluation/metrics/bem.py's `bertify_examples`
tokenization + segment-packing logic using transformers.BertTokenizer over the SAME
vendored vocab.txt (byte-identical WordPiece splits) instead of
tensorflow_text.BertTokenizer, and calls the downloaded SavedModel's forward pass
directly via tf.saved_model.load() instead of tensorflow_hub's hub.load() -- both are
thin wrappers over the same SavedModel loader for a local path, so this drops the
tensorflow_text / tensorflow_hub dependency entirely. Only `tensorflow` (forward pass)
and `transformers` (already a project dependency, for tokenization) are needed.

Segment packing (input_ids = [CLS] + candidate + [SEP] + reference + [SEP] + question +
[SEP], segment_ids incrementing per block) reconstructs tensorflow_text.combine_segments'
documented behavior for N segments -- this one piece was not read verbatim from the
official source (which calls the opaque tensorflow_text primitive), so verify against a
known-good BEM score before trusting it (see the __main__ self-test below, and
cross-check a handful of scores against the official repo's own run if possible).

One-time setup:
    pip install tensorflow kagglehub
    kagglehub needs a Kaggle account + API token (~/.kaggle/kaggle.json or the
    KAGGLE_USERNAME / KAGGLE_KEY env vars) to download the checkpoint.

Threshold = 0.8, NOT 0.5 -- this value exists only in the official code
(compute_score(threshold=0.8) default), never stated in the SemGrad paper text.
"""
import os

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VOCAB_PATH = os.path.join(_REPO_ROOT, "data", "semgrad_protocol", "vocab.txt")

MAX_LENGTH = 512
THRESHOLD = 0.8

_bem_model = None
_bem_tokenizer = None


def _get_tokenizer():
    global _bem_tokenizer
    if _bem_tokenizer is None:
        from transformers import BertTokenizer
        if not os.path.exists(_VOCAB_PATH):
            raise FileNotFoundError(
                f"BEM vocab not found at {_VOCAB_PATH} -- see data/semgrad_protocol/"
                f"PROVENANCE.md for how it was vendored.")
        _bem_tokenizer = BertTokenizer(vocab_file=_VOCAB_PATH, do_lower_case=True)
    return _bem_tokenizer


def load_bem_model():
    """
    Download (if needed, via kagglehub) and load the BEM SavedModel, replacing the dead
    tfhub.dev URL the official paper/code used. Returns a callable tf.saved_model object
    -- call it as model({'input_ids': ..., 'segment_ids': ...}) to get raw logits, exactly
    like the official code's hub.load(...) object was called.
    """
    global _bem_model
    if _bem_model is not None:
        return _bem_model
    import kagglehub
    import tensorflow as tf

    model_dir = kagglehub.model_download("google/bert/tensorFlow2/answer-equivalence-bem")
    load_dir = model_dir
    if not os.path.exists(os.path.join(load_dir, "saved_model.pb")):
        for root, _dirs, files in os.walk(model_dir):
            if "saved_model.pb" in files:
                load_dir = root
                break
        else:
            raise FileNotFoundError(
                f"No saved_model.pb found under {model_dir} -- kagglehub's download "
                f"layout may have changed; inspect the directory manually.")

    with tf.device("/cpu:0"):
        _bem_model = tf.saved_model.load(load_dir)
    return _bem_model


def _tokenize_ids(text: str) -> list:
    """WordPiece token ids for one string, no [CLS]/[SEP] -- matches the official
    tensorflow_text.BertTokenizer(preserve_unused_token=True, lower_case=True) split
    against the same vendored vocab.txt."""
    return _get_tokenizer().encode(text, add_special_tokens=False)


def _bertify_batch(examples: list) -> dict:
    """
    Reimplements bem.py::BemCalculator.bertify_examples for a batch of
    {'question', 'reference', 'candidate'} dicts.

    Segment order is (candidate, reference, question) -- candidate first. This order is
    load-bearing and matches the official code exactly; do not reorder.
    """
    tok = _get_tokenizer()
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    max_len = MAX_LENGTH - 4  # official constant: reserves room for [CLS] + 3x[SEP]

    input_ids_batch, segment_ids_batch = [], []
    for ex in examples:
        q_ids = _tokenize_ids(ex["question"])
        r_ids = _tokenize_ids(ex["reference"])
        c_ids = _tokenize_ids(ex["candidate"])

        ex_len = len(q_ids) + len(r_ids) + len(c_ids)
        if ex_len > max_len:
            # Candidate is truncated FROM THE END by the overflow amount; question and
            # reference are never truncated. Matches c[:-(ex_len-max_len)] in the
            # official code (Python list slicing already yields [] if overflow >=
            # len(c_ids), so no extra guard is needed).
            c_ids = c_ids[:-(ex_len - max_len)]

        # text.combine_segments((candidate, reference, question), cls_id, sep_id):
        # [CLS] + cand + [SEP] + ref + [SEP] + quest + [SEP]; segment_ids 0/1/2 per block.
        input_ids = [cls_id] + c_ids + [sep_id] + r_ids + [sep_id] + q_ids + [sep_id]
        segment_ids = ([0] * (2 + len(c_ids)) + [1] * (1 + len(r_ids))
                        + [2] * (1 + len(q_ids)))

        pad_n = MAX_LENGTH - len(input_ids)
        if pad_n < 0:
            raise ValueError(
                f"Packed sequence length {len(input_ids)} exceeds MAX_LENGTH="
                f"{MAX_LENGTH} even after truncating the candidate to zero -- the "
                f"question+reference alone are too long; unexpected for SciQ/TruthfulQA.")
        input_ids_batch.append(input_ids + [0] * pad_n)
        segment_ids_batch.append(segment_ids + [0] * pad_n)

    return {"input_ids": np.array(input_ids_batch, dtype=np.int64),
            "segment_ids": np.array(segment_ids_batch, dtype=np.int64)}


def bem_score_batch(examples: list, batch_size: int = 25) -> list:
    """
    examples: list of {'question': str, 'reference': str, 'candidate': str}.
    Returns the BEM positive-class probability for each example
    (softmax(logits, axis=1)[:, 1]), matching the official BemCalculator.__call__.

    Empty candidates are replaced with the literal string "None" first, per the
    official uncertainty/generation_evaluation/__init__.py preprocessing.
    """
    from scipy.special import softmax

    model = load_bem_model()
    examples = [{**ex, "candidate": ex["candidate"] if ex["candidate"] != "" else "None"}
                for ex in examples]

    scores = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        inputs = _bertify_batch(batch)
        raw_outputs = model(inputs)
        batch_scores = softmax(np.asarray(raw_outputs), axis=1)[:, 1]
        scores.extend(float(s) for s in batch_scores)
    return scores


def bem_score(question: str, candidate: str, references: list) -> float:
    """Max BEM score over all reference answers -- the official multi-reference
    aggregation (uncertainty/generation_evaluation/__init__.py, metric == 'bem')."""
    examples = [{"question": question, "reference": r, "candidate": candidate}
                for r in references]
    return max(bem_score_batch(examples), default=0.0)


def bem_correct(question: str, candidate: str, references: list,
                 threshold: float = THRESHOLD) -> bool:
    return bem_score(question, candidate, references) >= threshold


def bem_label_cache(cache: dict, threshold: float = THRESHOLD, checkpoint=None,
                     checkpoint_every: int = 25, batch_size: int = 25,
                     on_progress=None) -> int:
    """
    Relabel every candidate in a run cache with BEM -- the SemGrad protocol's
    authoritative correctness metric. Mirrors judge_utils.judge_label_cache's convention
    exactly: the prior label is preserved as `label_lexical`, `label` is overwritten with
    the BEM verdict, and `label_bem=True` marks a candidate done (idempotent -- a re-run
    only scores candidates not already marked). The raw `bem_score` float is also saved
    so the 0.8 threshold can be revisited offline without rescoring.

    entry["gold_row"] must carry "truthful_answers": list[str] (the schema
    load_semgrad_sciq / load_semgrad_truthfulqa produce).

    Scores every not-yet-labeled candidate x every one of its references as ONE flat
    batch, mirroring the official code's own 'bem' metric branch
    (uncertainty/generation_evaluation/__init__.py) -- much faster than one bem_score()
    call per candidate, especially for TruthfulQA's multi-reference questions.

    Args:
        cache:      {idx: {question, gold_row, candidates:[...]}} run cache (mutated in place).
        checkpoint: zero-arg callable that persists `cache` (e.g. save_cache_atomic partial).
    Returns the number of candidates scored this call.
    """
    targets = []  # (idx, question, references, candidate_dict)
    for idx, entry in cache.items():
        refs = entry.get("gold_row", {}).get("truthful_answers")
        if not refs:
            continue
        q = entry.get("question", "")
        for c in entry["candidates"]:
            if c.get("label_bem"):
                continue
            targets.append((idx, q, refs, c))
    if not targets:
        return 0

    flat_examples, bounds = [], [0]
    for _, q, refs, c in targets:
        cand_text = c.get("full_text", "")
        for r in refs:
            flat_examples.append({"question": q, "reference": r, "candidate": cand_text})
        bounds.append(len(flat_examples))

    flat_scores = bem_score_batch(flat_examples, batch_size=batch_size)

    n = 0
    for i, (idx, _q, _refs, c) in enumerate(targets):
        seg = flat_scores[bounds[i]:bounds[i + 1]]
        score = max(seg) if seg else 0.0
        c["label_lexical"] = bool(c.get("label", False))
        c["bem_score"] = score
        c["label"] = score >= threshold
        c["label_bem"] = True
        n += 1
        if on_progress:
            on_progress(idx, n)
        if checkpoint and (n % checkpoint_every == 0):
            checkpoint()
    if checkpoint:
        checkpoint()
    return n


if __name__ == "__main__":
    # Official repo's own commented-out sanity example (bem.py). Run this once TF +
    # kagglehub are set up to confirm the reimplementation loads and scores sanely
    # before trusting it on the pilot's 400 candidates.
    demo = [{"question": "why is the sky blue", "reference": "light scattering",
             "candidate": "scattering of light"}]
    print("BEM score (expect high, this is a near-paraphrase):", bem_score_batch(demo))
