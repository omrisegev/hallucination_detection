"""
Registry of label-free per-cell feature-subset selectors (Step 186+).

A selector family registers one callable under a short name:

    from spectral_utils.selectors import register

    @register('a1_residual')
    def a1_residual(cell, rng, cache=None):
        ...
        return [ {'variant': 'a1.minres_exh_s5', 'cols': ..., 'diag': {...}},
                 ... ]

Contract (enforced by spectral_utils/selector_bench.py):
  * `cell` is an UnlabeledCell — no labels, no positive rate, ever.
  * `rng` is a per-cell-seeded numpy Generator (determinism).
  * `cache` is the label-free slice of the cell's exhaustive-sweep npz
    (mask/size/residual/K/rho stats — NO auroc) or None; selectors must
    degrade gracefully without it.
  * returns a LIST of selection dicts: variant (str), cols (indices into
    cell.V columns), optional groups / K / fusion ('lsml'|'upcr') /
    fallback (bool) / diag (JSON-able).

Worktree convention: a2_groupfs / a3_concrete_ae are developed on their own
branches; the try/except imports below pre-stub them so merge-back never
touches this file. Each selector module should also expose a ``smoke()``
known-answer test — auto-discovered by scripts/smoke_selectors.py.
"""

import importlib

_REGISTRY = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_selector(name):
    if name not in _REGISTRY:
        raise KeyError(f"unknown selector {name!r}; registered: "
                       f"{sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered():
    return dict(_REGISTRY)


# Selectors developed inline on master (register on import).
from . import a1_residual       # noqa: E402,F401  residual-guided (Nadler/Kluger)
from . import a5_mrmr           # noqa: E402,F401  mRMR relevance-vs-redundancy (Step 189)
from . import classical_fs      # noqa: E402,F401  Laplacian Score / SPEC / MCFS
from . import simple_stats      # noqa: E402,F401  random / MAD / kurtosis / decorr
from . import reference_macros  # noqa: E402,F401  GOOD_5/6, STABLE_H9, ... baselines

# Worktree-developed selector modules — pre-stubbed so branch merges never
# conflict on this file. Absent modules are simply skipped.
for _optional in ('a2_groupfs', 'a3_concrete_ae', 'a4_antigravity'):
    try:
        importlib.import_module(f'.{_optional}', __name__)
    except ImportError:
        pass
