"""
REFRAIN — "Stop When Enough: Adaptive Early-Stopping for Chain-of-Thought Reasoning"
(Sun, Cheng, Li, Chen, Wang; ACL 2026 Long Papers, 2026.acl-long.1256).

Handoff §S1. The official repository (RLSNLP/Adaptive-Reasoning) was a release-placeholder
README when audited, so this is `paper-specified` / `paper-specified-partial`, never
`official-exact`.

The trigger vocabulary is not a guess
-------------------------------------
Table 5 prints four categories, but its caption says underlining marks the **Section 5.2
in-category expansions** and bold marks the **new category** — and the main REFRAIN row
uses neither (Table 2 reports In-cat Expansion and New Category as separate, different
variants: 91.60/1.65M and 91.20/1.69M against REFRAIN's 91.20/1.61M).

Plain text extraction loses underlining, so the base sets below were recovered from the
PDF itself: page 14's underlines are drawn line segments, and each character's midpoint
was tested against them (`scripts/recover_refrain_vocabulary.py`). `V_BASE` is therefore
the vocabulary the paper's headline numbers actually used, and `V_INCAT_EXPANSION` /
`V_NEW_CATEGORY` reproduce its two ablation rows.

Declared deviations (this run is `paper-specified-partial`)
-----------------------------------------------------------
1. **Provisional-answer cue `c`.** The paper gives only "e.g., 'answer is/should be'".
   We freeze `PROVISIONAL_ANSWER_CUES` and declare it.
2. **Reward timing / running-length update.** Eq. 9 uses a running mean `L̄` without
   saying whether the current sample is included. We compute the reward against `L̄` over
   *previous* rounds, then update — the standard reading of "running mean", declared.
3. **Cold-start arm order and tie-breaking.** Alg. 2 says "arbitrary such t" for
   cold-start and does not define UCB ties. We take ascending τ in both cases so the run
   is deterministic and reproducible.

None of these is tuned. They are frozen before generation and recorded in the manifest;
tuning any of them against the published 91.20/1.61M target would be exactly the
"tune toward the published number" failure the handoff §6 forbids.

State crosses questions
-----------------------
SW-UCB reward buffers persist across the whole dataset, so REFRAIN is **not** a per-question
method and cannot be sharded across GPUs. The MATH-500 row order is frozen in the manifest's
`dataset_order_sha256`; shuffling it changes the algorithm (phase-1 checkpoint §7.13).
"""
import math
import re
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ── Table 5 vocabulary, base sets (underline-free = used by the headline numbers) ──
V_CHECK = ("wait", "let me check", "hold on", "have made a mistake")
V_SHIFT = ("alternatively", "let me try", "think of it as", "let me consider")
V_UNCERT = ("not sure", "looks like", "that seems", "hmm", "perhaps", "maybe i")
V_RETRO = ("earlier we saw", "from before", "so now we have", "recall that", "let me go back")

V_BASE = tuple(V_CHECK + V_SHIFT + V_UNCERT + V_RETRO)

#: Section 5.2 "In-cat Expansion" ablation (Table 2: 91.60 / 1.65M) — underlined in Table 5.
V_INCAT_EXPANSION = V_BASE + (
    "let me double check", "wait a moment", "is that correct", "let me re-read",
    "what if we try", "let's think from a different angle",
    "an alternative method would be", "instead of doing that",
    "i'm not certain", "it seems", "i suspect", "my guess is",
    "as we established previously", "based on our previous result",
    "remember that we found", "the value from step",
)

#: Section 5.2 "New Category" ablation (Table 2: 91.20 / 1.69M) — bold in Table 5.
V_NEW_CATEGORY = V_BASE + (
    "simplify this problem", "the core of the problem is", "this is equivalent to",
    "this is equal to", "the key insight here is", "break this down",
    "the overall plan is to", "the plan is to",
)

VOCABULARIES = {
    "base": V_BASE,
    "incat_expansion": V_INCAT_EXPANSION,
    "new_category": V_NEW_CATEGORY,
}

#: DECLARED DEVIATION 1 — the paper gives this only as "e.g., 'answer is/should be'".
PROVISIONAL_ANSWER_CUES = ("answer is", "answer should be")

#: Prompt P0, verbatim from Appendix (used for the headline table).
PROMPT_P0 = ("{question}\nPlease answer step by step. End your response with: "
             "Final Answer: \\boxed{{your final answer here}}. "
             "Make sure to wrap your final answer in \\boxed{{}}.")

#: Forced-closure prompt after a stop (§3.3).
CLOSURE_PROMPT = "Final Answer: \\boxed{"

# ── frozen hyperparameters (§ Experimental setup + Appendix) ─────────────────────
TAU_GRID = (0.60, 0.65, 0.70, 0.75, 0.80)
UCB_C = 1.0
WINDOW_W = 100
LAMBDA = 0.2
COLD_START_COEF = 1e-4
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 16384
SEED = 42
# Qwen3 thinking-mode decoding (Appendix): T=0.6, top-p 0.95, top-k 20.
QWEN3_DECODING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
# gpt-oss decoding, for the C1 confirmation cell.
GPTOSS_DECODING = {"temperature": 1.0, "top_p": 1.0, "top_k": 50}

STEP_DELIM = re.compile(r"\n\s*\n")


def format_p0(question: str) -> str:
    return PROMPT_P0.format(question=question)


def segment_steps(text: str) -> list:
    """Split a reasoning trace on blank-line delimiters (§3.1)."""
    return [s.strip() for s in STEP_DELIM.split(text or "") if s.strip()]


def _norm(s: str) -> str:
    """Lowercase and fold the curly apostrophe/quote forms.

    Table 5 prints U+2019 ("let's", "i'm not certain") while models emit ASCII "'". Without
    folding, two of the expansion-set phrases could never match anything a model writes,
    which would silently make the In-cat Expansion ablation a different vocabulary again.
    """
    return (s or "").lower().replace("’", "'").replace("‘", "'")


def _contains_any(haystack: str, needles) -> bool:
    low = _norm(haystack)
    return any(_norm(n) in low for n in needles)


def is_reflective(step: str, vocab=V_BASE) -> bool:
    """r_n = I(exists v in V such that v is a substring of s_n)  — Eq. 2."""
    return _contains_any(step, vocab)


def has_provisional_answer(prior_steps, cues=PROVISIONAL_ANSWER_CUES) -> bool:
    """h_{n-1} = I(exists j < n such that c is a substring of s_j)  — Eq. 3.

    Note the index: the cue must appear in a step **strictly before** the current one, so
    a single step that both proposes an answer and reflects on it cannot trigger a stop.
    """
    return any(_contains_any(s, cues) for s in prior_steps)


class SWUCB:
    """Sliding-window UCB over the τ grid — Algorithm 2.

    One instance per *dataset run*, not per question: the buffers are the cross-question
    state that makes REFRAIN adaptive, and resetting them per question would silently
    turn it into a fixed-threshold method that always plays the cold-start arm.
    """

    def __init__(self, arms=TAU_GRID, window: int = WINDOW_W, c: float = UCB_C):
        self.arms = list(arms)
        self.window = int(window)
        self.c = float(c)
        self.buffers = {a: deque(maxlen=self.window) for a in self.arms}
        self.k = 0            # round index, 1-based after the first select()
        self.history = []

    def select(self):
        """Return (arm, diagnostics) for the next round."""
        self.k += 1
        # Cold start: any unplayed arm. DECLARED DEVIATION 3 — ascending τ, for determinism.
        for a in self.arms:
            if len(self.buffers[a]) == 0:
                return a, {"reason": "cold_start", "round": self.k}
        t_eff = min(self.k, self.window * len(self.arms))
        scores = {}
        for a in self.arms:
            n = max(1, len(self.buffers[a]))
            scores[a] = float(np.mean(self.buffers[a])) + self.c * math.sqrt(
                2.0 * math.log(max(t_eff, 2)) / n)
        # Ties break toward the lowest index (ascending τ) — DECLARED DEVIATION 3.
        top = max(scores.values())
        best = next(a for a in self.arms if scores[a] == top)
        return best, {"reason": "ucb", "round": self.k, "t_eff": t_eff, "ucb": scores}

    def update(self, arm, reward: float):
        self.buffers[arm].append(float(reward))
        self.history.append({"round": self.k, "arm": arm, "reward": float(reward)})

    def state(self) -> dict:
        return {
            "round": self.k,
            "arms": self.arms,
            "window": self.window,
            "c": self.c,
            "buffers": {a: list(b) for a, b in self.buffers.items()},
            "means": {a: (float(np.mean(b)) if b else None) for a, b in self.buffers.items()},
            "n_pulls_in_window": {a: len(b) for a, b in self.buffers.items()},
        }

    def load_state(self, st: dict):
        """Restore after preemption. Without this a requeued job would restart the bandit
        cold mid-dataset, which is a different algorithm from the one that ran before the
        preemption — and the resulting trace file would look perfectly normal."""
        self.k = int(st["round"])
        self.arms = list(st["arms"])
        self.window = int(st["window"])
        self.c = float(st["c"])
        self.buffers = {a: deque(st["buffers"].get(str(a), st["buffers"].get(a, [])),
                                 maxlen=self.window) for a in self.arms}


@dataclass
class RewardState:
    """Running mean of total output length L̄, for Eq. 9."""
    n: int = 0
    total: float = 0.0

    @property
    def mean(self):
        return (self.total / self.n) if self.n else None

    def reward(self, score: float, length: int) -> dict:
        """R = Score(y|x) - λ·L/L̄, with a 1e-4·L cold start on the first sample.

        DECLARED DEVIATION 2: computed against L̄ over *previous* rounds, then updated.
        """
        mean = self.mean
        if mean is None or mean <= 0:
            penalty = COLD_START_COEF * float(length)
            mode = "cold_start"
        else:
            penalty = LAMBDA * float(length) / mean
            mode = "running_mean"
        r = float(score) - penalty
        return {"reward": r, "score": float(score), "penalty": penalty,
                "L": int(length), "L_bar": mean, "mode": mode}

    def observe(self, length: int):
        self.n += 1
        self.total += float(length)

    def state(self) -> dict:
        return {"n": self.n, "total": self.total}

    def load_state(self, st: dict):
        self.n, self.total = int(st["n"]), float(st["total"])


@dataclass
class RefrainConfig:
    tau_grid: tuple = TAU_GRID
    ucb_c: float = UCB_C
    window: int = WINDOW_W
    lam: float = LAMBDA
    cold_start_coef: float = COLD_START_COEF
    vocabulary: str = "base"
    cues: tuple = PROVISIONAL_ANSWER_CUES
    sbert_model: str = SBERT_MODEL
    max_new_tokens: int = MAX_NEW_TOKENS
    seed: int = SEED
    decoding: dict = field(default_factory=lambda: dict(QWEN3_DECODING))

    @property
    def vocab(self):
        return VOCABULARIES[self.vocabulary]

    def as_manifest(self) -> dict:
        return {
            "tau_grid": list(self.tau_grid), "ucb_c": self.ucb_c, "window": self.window,
            "lambda": self.lam, "cold_start_coef": self.cold_start_coef,
            "vocabulary": self.vocabulary, "n_trigger_phrases": len(self.vocab),
            "provisional_cues": list(self.cues), "sbert_model": self.sbert_model,
            "max_new_tokens": self.max_new_tokens, "seed": self.seed,
            "decoding": dict(self.decoding),
        }


class StepStopper:
    """Algorithm 1 — the two-stage discriminator, driven incrementally.

    Fed the growing decoded text after every token; it re-segments, and when a *new*
    completed step appears it evaluates the stop rule on that step. Only completed steps
    are judged: a half-written step would match a trigger phrase before its redundancy is
    measurable, which would stop far earlier than the paper's rule.
    """

    def __init__(self, encoder, tau: float, cfg: RefrainConfig):
        self.encoder = encoder
        self.tau = float(tau)
        self.cfg = cfg
        self.steps = []
        self.embs = []
        self.n_seen = 0
        self.fired = None

    def __call__(self, text_so_far: str, channels=None) -> bool:
        segs = segment_steps(text_so_far)
        # The last segment is still being written unless the text ends on a delimiter.
        complete = segs if STEP_DELIM.search(text_so_far or "") and \
            STEP_DELIM.split(text_so_far)[-1].strip() == "" else segs[:-1]
        while self.n_seen < len(complete):
            step = complete[self.n_seen]
            self.n_seen += 1
            prior = list(self.steps)
            emb = self.encoder.encode([step], normalize_embeddings=True)[0]
            phi = float(max((float(np.dot(emb, e)) for e in self.embs), default=0.0))
            self.steps.append(step)
            self.embs.append(emb)
            r = is_reflective(step, self.cfg.vocab)
            h = has_provisional_answer(prior, self.cfg.cues)
            if h and r and phi >= self.tau:
                self.fired = {"step_index": self.n_seen - 1, "phi": phi, "tau": self.tau,
                              "reflective": True, "has_provisional": True, "step_text": step}
                return True
        return False

    def diagnostics(self) -> dict:
        return {"n_steps": len(self.steps), "fired": self.fired, "tau": self.tau}


class MiniLMEncoder:
    """all-MiniLM-L6-v2 sentence embeddings on plain `transformers`.

    Exposes the one method REFRAIN needs, with sentence-transformers' signature, so it is a
    drop-in for `SentenceTransformer` and the CPU tests can still stub it.

    Why not just import sentence_transformers: the AIRCC compute nodes can reach the
    HuggingFace hub but not PyPI, so an in-job `pip install sentence-transformers` fails with
    a DNS error and takes the whole S1 run with it. The library's contribution here is
    exactly three lines — encode, attention-masked mean-pool, L2-normalise — which is the
    documented pooling configuration for this checkpoint, so reproducing it removes a
    network dependency without changing a number.
    """

    def __init__(self, model_name: str = SBERT_MODEL, device: str = None,
                 max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if device:
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        # all-MiniLM-L6-v2's own config truncates at 256 word pieces; a REFRAIN step can be
        # longer, and silently letting it through would change the similarity.
        self.max_length = int(max_length)

    def encode(self, sentences, normalize_embeddings: bool = True, batch_size: int = 32):
        import numpy as _np
        torch = self.torch
        if isinstance(sentences, str):
            sentences = [sentences]
        out = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = [str(s) for s in sentences[i:i + batch_size]]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.float().cpu().numpy())
        return _np.concatenate(out, axis=0) if out else _np.zeros((0, 384))


def load_sbert(model_name: str = SBERT_MODEL, device: str = None):
    """Load the redundancy scorer. Behind a function so the CPU smoke tests can stub it."""
    return MiniLMEncoder(model_name, device=device)


def extract_boxed_answer_ids(tok, text: str):
    """Token IDs of the boxed answer region, for Eq. 6's answer-only likelihood.

    Returns (answer_text, token_ids) or (None, []). Scoring the whole trace instead would
    make the reward a length-and-style statistic rather than an answer-confidence one,
    which is exactly what §3.3 says it is avoiding.
    """
    from .evaluator import extract_boxed
    ans = extract_boxed(text)
    if ans is None:
        return None, []
    return ans, tok(ans, add_special_tokens=False).input_ids
