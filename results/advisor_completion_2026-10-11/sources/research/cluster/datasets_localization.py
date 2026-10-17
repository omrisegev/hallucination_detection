"""
datasets_localization.py — extra `run_inference.DATASETS` entries for the Evidence Drop replication.

New file per the Step-186 worktree convention. `run_inference.py` picks these up through a
three-line `try/except` at the bottom of its DATASETS block, so merge-back never conflicts.

Contract is identical to `run_inference.DATASETS`:
    dataset_key -> (loader(n, split) -> list[row], prompt_fn(row) -> str, grader(gen, row) -> bool)

Key naming note: the new key is **`math`**, not `math_full`. `scripts/smoke_preset.py`'s
`_fixture_family` already routes `dataset in ("math500", "math")` to the `math_family` grader
fixtures, so using `math` means the smoke gate covers these presets with zero edits to that
shared file. `math` = the full Hendrycks test split; `math500` = the 500-problem Lightman subset
the rest of the grid uses. Both stay available and are never mixed inside one cell.
"""


def register_all(datasets: dict) -> dict:
    """Merge the localization dataset entries into `run_inference.DATASETS` (in place).

    Refuses to shadow an existing key — a silent override would repoint an already-scored cell's
    loader or grader without changing its preset id, which is the kind of staleness Step 193
    spent a whole step chasing.
    """
    from spectral_utils.data_loaders import math_prompt, is_correct_math
    from spectral_utils.localization_data import load_math_full

    new = {
        # Full MATH test split (Hendrycks et al., 2021) — the benchmark "Mind the Gap" actually
        # uses. Same prompt + grader as math500; only the underlying problem set differs.
        "math": (lambda n, split="test": load_math_full(n, split or "test"),
                 math_prompt, is_correct_math),
    }

    clash = sorted(set(new) & set(datasets))
    if clash:
        raise KeyError(
            f"datasets_localization would shadow existing DATASETS key(s): {clash}. "
            "Rename the new key instead of overriding — an override silently changes what an "
            "existing preset id means."
        )
    datasets.update(new)
    return datasets
