"""
localization_data.py — dataset loaders added for the Evidence Drop replication (Extension F).

New file rather than an edit to `data_loaders.py`: this branch follows the Step-186 worktree
convention (new files only, so merge-back never conflicts on a shared module). Everything here
is still inside `spectral_utils`, so the CLAUDE.md "never inline helpers in notebooks" rule holds.

WHY A FULL-MATH LOADER EXISTS
-----------------------------
"Mind the Gap" (ICML 2026) evaluates on **MATH (Hendrycks et al., 2021)**. The strings
`MATH-500`, `MATH500` and `500` appear nowhere in that paper — it is the full benchmark, not
the 500-problem Lightman/`lighteval` subset our replication grid uses everywhere else.

That distinction is not cosmetic here. The paper's headline metrics are **selective accuracy**
and **AURC**, and both are monotone functions of the base error rate, so a reproduction run on a
different subset with a different difficulty mix is not comparable to their tables no matter how
faithfully the estimator is implemented. `load_math_full` therefore targets the real MATH test
split; `data_loaders.load_math500` stays untouched as the grid's continuity cell.

SAMPLING
--------
The MATH test split is 5,000 problems across 7 subjects. `EleutherAI/hendrycks_math` exposes one
config per subject, so a naive concatenation + `[:n]` would return **all algebra** — a silently
biased sample. Rows are therefore concatenated in a fixed subject order and then shuffled with a
**fixed seed** before truncation, so an `n < 5000` request is subject-representative and, more
importantly, byte-reproducible across runs and machines.
"""
import random

# Fixed so `load_math_full(n)` returns the same n problems on every machine and every rerun.
# Changing this invalidates comparability with any already-scored cell.
MATH_SAMPLE_SEED = 0

# Canonical MATH test subjects. Order is fixed (not `set` iteration) so the pre-shuffle
# concatenation is deterministic before the seeded shuffle is applied.
_HENDRYCKS_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _normalize_math_row(row: dict, subject: str = "") -> dict:
    """Coerce a MATH row to the {problem, solution} contract `math_prompt` / `is_correct_math` read.

    `data_loaders.math_prompt` looks for problem/query/question and `is_correct_math` looks for
    solution/answer/output, so a Hendrycks row already satisfies both. This only fills `subject`
    (absent from the per-config rows, useful for stratified reporting) and guarantees the two
    keys exist so a schema surprise fails here rather than mid-run on the GPU.
    """
    out = dict(row)
    if "problem" not in out:
        raise KeyError(f"MATH row has no 'problem' key; got {sorted(out)[:8]}")
    if "solution" not in out:
        raise KeyError(f"MATH row has no 'solution' key; got {sorted(out)[:8]}")
    out.setdefault("subject", subject or out.get("type", ""))
    return out


def load_math_full(n_samples: int = 1500, split: str = "test") -> list:
    """Full MATH (Hendrycks et al., 2021) test problems — NOT the 500-problem subset.

    Tries, in order:
      1. one `EleutherAI/hendrycks_math` config per subject, concatenated (the real 5,000-row
         test split);
      2. `nlile/hendrycks-MATH-benchmark`, a single-config mirror of the same data;
      3. `EleutherAI/hendrycks_math` with `name="all"` — kept last because "all" is not a
         documented config and may not resolve; `data_loaders.load_math500` lists it optimistically.

    Raises RuntimeError if every source fails, rather than silently degrading to MATH-500 —
    a silent fallback would reintroduce exactly the comparability defect this loader exists to fix.
    """
    from datasets import load_dataset

    rows = []
    errors = []

    # 1. per-subject configs (the documented layout)
    try:
        for subject in _HENDRYCKS_SUBJECTS:
            ds = load_dataset("EleutherAI/hendrycks_math", name=subject, split=split)
            rows.extend(_normalize_math_row(ds[i], subject) for i in range(len(ds)))
        print(f"Loaded {len(rows)} MATH problems from EleutherAI/hendrycks_math "
              f"({len(_HENDRYCKS_SUBJECTS)} subject configs).")
    except Exception as e:
        errors.append(f"per-subject configs: {type(e).__name__}: {e}")
        rows = []

    # 2. single-config mirror
    if not rows:
        for path in ("nlile/hendrycks-MATH-benchmark",):
            try:
                ds = load_dataset(path, split=split)
                rows = [_normalize_math_row(ds[i]) for i in range(len(ds))]
                print(f"Loaded {len(rows)} MATH problems from {path}.")
                break
            except Exception as e:
                errors.append(f"{path}: {type(e).__name__}: {e}")

    # 3. undocumented "all" config
    if not rows:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", name="all", split=split)
            rows = [_normalize_math_row(ds[i]) for i in range(len(ds))]
            print(f"Loaded {len(rows)} MATH problems from EleutherAI/hendrycks_math (name='all').")
        except Exception as e:
            errors.append(f"hendrycks_math name='all': {type(e).__name__}: {e}")

    if not rows:
        raise RuntimeError(
            "Could not load the full MATH test split from any source. Refusing to fall back to "
            "MATH-500 — the Evidence Drop replication needs the full benchmark for its selective "
            "accuracy / AURC numbers to be comparable. Attempts:\n  " + "\n  ".join(errors)
        )

    # Seeded shuffle BEFORE truncation, so n < len(rows) stays subject-representative.
    random.Random(MATH_SAMPLE_SEED).shuffle(rows)
    return rows[:n_samples]


def smoke() -> None:
    """Known-answer checks that need no network (auto-discovered by the localization smoke gate).

    Only the pure logic is covered here — `load_math_full` itself is network-bound and is
    validated by the N=30 cluster pilot, which prints the loaded row count and accuracy.
    """
    # _normalize_math_row fills `subject` and enforces the two keys the graders read.
    r = _normalize_math_row({"problem": "1+1?", "solution": r"\boxed{2}"}, "algebra")
    assert r["subject"] == "algebra", r
    assert r["problem"] == "1+1?" and r["solution"] == r"\boxed{2}"

    # A row missing either key must raise here, not silently produce an ungradeable cell.
    for bad in ({"solution": "x"}, {"problem": "x"}):
        try:
            _normalize_math_row(bad)
        except KeyError:
            pass
        else:
            raise AssertionError(f"_normalize_math_row accepted a malformed row: {bad}")

    # The seeded shuffle must be reproducible across calls (this is what makes an n<5000
    # request comparable between the pilot and the full run).
    a = list(range(50))
    b = list(range(50))
    random.Random(MATH_SAMPLE_SEED).shuffle(a)
    random.Random(MATH_SAMPLE_SEED).shuffle(b)
    assert a == b, "seeded shuffle is not reproducible"
    assert a != list(range(50)), "seeded shuffle was a no-op"

    print("localization_data.smoke: PASS")
