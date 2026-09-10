"""Step readout controls and numeral-provenance rollback of token surprise.

Protocol: docs/experiments/READOUT_LENGTH_CONTROL_AND_PROVENANCE_V1.md.
Everything here is a fixed rule on one answer's own tokens: no labels, no
fitting, no other answers. Step scores are returned as float arrays whose
``np.argmax`` is the rule's predicted step (first-max tie rule preserved).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

TOP_K = 10
NEAR_MAX_SD = 0.25
DIGIT = re.compile(r"^\d$")
SPACE_DIGIT = re.compile(r"^ \d$")
JOINER = {".", ","}
CHAR_NUMERAL = re.compile(r"\d+(?:[.,]\d+)*")


def top_k_mean(x: np.ndarray, k: int = TOP_K) -> float:
    x = np.asarray(x, float)
    if not len(x):
        return -np.inf
    return float(np.sort(x)[-k:].mean()) if len(x) >= k else float(x.mean())


def step_scores_top_k(tokens: np.ndarray, starts: np.ndarray, ends: np.ndarray, k: int = TOP_K) -> np.ndarray:
    return np.array([top_k_mean(tokens[u:w], k) for u, w in zip(starts, ends)], float)


# ---------------------------------------------------------------------------
# Stage 0 controls
# ---------------------------------------------------------------------------

def length_scores(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return (np.asarray(ends) - np.asarray(starts)).astype(float)


def random_scores(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=n_steps)


def position_first_scores(n_steps: int) -> np.ndarray:
    return -np.arange(n_steps, dtype=float)


# ---------------------------------------------------------------------------
# Onset-style fixed readouts on frozen step scores
# ---------------------------------------------------------------------------

def rise_vs_history(s: np.ndarray) -> np.ndarray:
    """s_k minus the mean of all earlier step scores (s_0 minus the answer mean)."""
    s = np.asarray(s, float)
    out = np.empty_like(s)
    for k in range(len(s)):
        out[k] = s[k] - (s[:k].mean() if k > 0 else s.mean())
    return out


def first_near_max(s: np.ndarray, sd_fraction: float = NEAR_MAX_SD) -> np.ndarray:
    """Earliest step within ``sd_fraction`` SD of the maximum wins; the near-max set
    is lifted above the maximum in earliest-first order, all other ranks unchanged."""
    s = np.asarray(s, float)
    if not len(s):
        return s.copy()
    mx, sd = s.max(), s.std()
    near = np.flatnonzero(s >= mx - sd_fraction * sd)
    out = s.copy()
    eps = 1e-6 * max(sd, 1e-12)
    for rank, k in enumerate(near):  # near is ascending in k
        out[k] = mx + eps * (len(near) - rank)
    return out


# ---------------------------------------------------------------------------
# Numeral provenance
# ---------------------------------------------------------------------------

@dataclass
class Numeral:
    start: int          # first token index (inclusive)
    end: int            # last token index (exclusive)
    literal: str        # digits with '.' kept and ',' removed
    step: int           # step index the numeral lies in
    origin: int = -1    # earliest step containing the same literal (filled later)
    given: bool = False


def numeral_runs(token_texts: list[str]) -> list[tuple[int, int, str]]:
    """Maximal runs of single-digit tokens, optionally joined by one '.' or ','
    token between digit runs. Single-digit literals are dropped."""
    runs: list[tuple[int, int, str]] = []
    n = len(token_texts)
    i = 0
    while i < n:
        t = token_texts[i]
        if DIGIT.match(t) or SPACE_DIGIT.match(t):
            j = i + 1
            digits = [t.strip()]
            while j < n:
                tj = token_texts[j]
                if DIGIT.match(tj):
                    digits.append(tj)
                    j += 1
                elif tj in JOINER and j + 1 < n and DIGIT.match(token_texts[j + 1]):
                    digits.append("." if tj == "." else "")
                    j += 1
                else:
                    break
            literal = "".join(digits)
            if len(literal.replace(".", "")) >= 2:
                runs.append((i, j, literal))
            i = j
        else:
            i += 1
    return runs


def given_literals(text: str) -> set[str]:
    out = set()
    for m in CHAR_NUMERAL.finditer(text or ""):
        lit = m.group(0).replace(",", "")
        if len(lit.replace(".", "")) >= 2:
            out.add(lit)
    return out


def step_of_token(starts: np.ndarray, ends: np.ndarray, n_tokens: int) -> np.ndarray:
    step = np.full(n_tokens, -1, int)
    for k, (u, w) in enumerate(zip(starts, ends)):
        step[u:w] = k
    return step


def build_numerals(token_texts: list[str], starts: np.ndarray, ends: np.ndarray, given: set[str]) -> list[Numeral]:
    """Numerals with step, given flag and earliest-step origin. Token indices are
    relative to the answer (same frame as ``starts``/``ends``)."""
    sot = step_of_token(starts, ends, len(token_texts))
    nums = [Numeral(a, b, lit, int(sot[a]), given=lit in given) for a, b, lit in numeral_runs(token_texts)]
    nums = [x for x in nums if x.step >= 0]
    first: dict[str, int] = {}
    for x in nums:  # token order == chronological order
        first.setdefault(x.literal, x.step)
        x.origin = first[x.literal]
    return nums


@dataclass
class Reassignment:
    pairs: list[tuple[int, int]] = field(default_factory=list)  # (token, step) memberships
    moved_tokens: int = 0
    inherited_numerals: int = 0
    numerals: int = 0
    given_numerals: int = 0


def reassign(nums: list[Numeral], starts: np.ndarray, ends: np.ndarray, n_tokens: int,
             mode: str, rng: np.random.Generator | None = None) -> Reassignment:
    """Token->step memberships after moving inherited, non-given numerals.

    mode 'reassign': inherited tokens leave their step and join origin(v).
    mode 'duplicate': inherited tokens stay AND are copied to origin(v).
    mode 'shuffled': as 'reassign' but the target is a uniform random earlier step.
    """
    if mode not in ("reassign", "duplicate", "shuffled"):
        raise ValueError(mode)
    sot = step_of_token(starts, ends, n_tokens)
    target = sot.copy()
    dup: list[tuple[int, int]] = []
    r = Reassignment(numerals=len(nums), given_numerals=sum(x.given for x in nums))
    for x in nums:
        if x.given or x.origin >= x.step:
            continue
        r.inherited_numerals += 1
        if mode == "shuffled":
            if rng is None:
                raise ValueError("shuffled mode needs rng")
            dest = int(rng.integers(0, x.step))  # uniform over earlier steps
        else:
            dest = x.origin
        for t in range(x.start, x.end):
            r.moved_tokens += 1
            if mode == "duplicate":
                dup.append((t, dest))
            else:
                target[t] = dest
    r.pairs = [(t, int(target[t])) for t in range(n_tokens) if target[t] >= 0] + dup
    return r


def step_scores_from_pairs(tokens: np.ndarray, pairs: list[tuple[int, int]], n_steps: int, k: int = TOP_K) -> np.ndarray:
    """Top-k mean over each step's membership bucket. A step emptied by
    reassignment gets a finite floor (answer minimum minus one) so it is never
    selected but the answer stays valid for the evaluator."""
    buckets: list[list[float]] = [[] for _ in range(n_steps)]
    for t, s in pairs:
        buckets[s].append(float(tokens[t]))
    floor = float(np.min(tokens)) - 1.0 if len(tokens) else 0.0
    return np.array([top_k_mean(np.asarray(b), k) if b else floor for b in buckets], float)


def normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


# ---------------------------------------------------------------------------
# Step-mass attribution along numeral dependencies
# (docs/experiments/STEP_MASS_ATTRIBUTION_V1.md)
# ---------------------------------------------------------------------------

def dependency_counts(nums: list[Numeral]) -> dict[tuple[int, int], int]:
    """n_kj: inherited non-given numerals in step k whose origin is step j < k."""
    counts: dict[tuple[int, int], int] = {}
    for x in nums:
        if x.given or x.origin >= x.step:
            continue
        key = (x.step, x.origin)
        counts[key] = counts.get(key, 0) + 1
    return counts


def zscore_steps(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else np.zeros_like(s)


def attribute_step_mass(z: np.ndarray, counts: dict[tuple[int, int], int], alpha: float,
                        mode: str = "dependency", rng: np.random.Generator | None = None) -> np.ndarray:
    """score'_j = (1 - alpha*[j has parents]) z_j + alpha * sum_k A(k->j) z_k.

    mode 'dependency': A(k->j) = n_kj / n_k.
    mode 'shuffled'  : each (k, j) parent edge is redirected to a uniform random
                       earlier step of k (weights kept), rng required.
    mode 'uniform'   : each step with parents sends alpha uniformly to all earlier steps.
    """
    if mode not in ("dependency", "shuffled", "uniform"):
        raise ValueError(mode)
    z = np.asarray(z, float); K = len(z)
    out = z.copy()
    senders: dict[int, dict[int, float]] = {}
    for (k, j), n in counts.items():
        senders.setdefault(k, {})[j] = senders.get(k, {}).get(j, 0.0) + float(n)
    for k, parents in senders.items():
        total = sum(parents.values())
        if total <= 0 or k <= 0:
            continue
        out[k] -= alpha * z[k]  # conservation: sender keeps (1 - alpha) of its own mass
        if mode == "uniform":
            share = alpha * z[k] / k
            out[:k] += share
            continue
        if mode == "shuffled" and rng is None:
            raise ValueError("shuffled mode needs rng")
        for j, n in parents.items():
            w = n / total
            dest = int(rng.integers(0, k)) if mode == "shuffled" else j
            out[dest] += alpha * w * z[k]
    return out
