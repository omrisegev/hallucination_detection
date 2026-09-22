"""
label_sanity.py — one place for "is this label set even scorable?" checks.

Why (PROJECT_RETROSPECTIVE_2026-09-22 §4.3, §7.5): every label bug in five months
(Steps 34, 41, 82, 144, 160, 182, 216, 313) produced a plausible AUROC and was caught
only by noticing an impossible accuracy, a handful of positives, or traces pinned at
the generation cap. None of the scorers checked those before writing a CSV row.
`report_figs.gate_flag` flagged bad cells, but post hoc, at report time.

Two tiers, mirroring the desk policy formalised 2026-07-12 in report_figs.py:
  HARD  — the AUROC would estimate the wrong quantity. Scorers refuse to write the row
          unless `--allow-degenerate REASON` is passed (the reason lands in the CSV).
  FLAG  — the AUROC is a noisy estimate of the right quantity. Written, tagged, and
          excluded from headline win tallies.

Thresholds live here and only here. `cluster/presets.py`, `scripts/report_figs.py`
and `scripts/labelfree_standing_report.py` import them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

# ── thresholds ─────────────────────────────────────────────────────────────────
HARD_ACC_BAND = (0.05, 0.98)     # outside: one class is (nearly) empty -> AUROC is noise
FLAG_ACC_BAND = (0.20, 0.85)     # desk band since 2026-07-12; outside -> CEILING/FLOOR flag
MIN_MINORITY_HARD = 10           # fewer positives/negatives than this -> refuse (Step 82: 2 positives)
MIN_MINORITY_FLAG = 30           # presets.DEFAULT_MIN_MINORITY; below -> flag
CAP_PINNED_HARD = 0.10           # >=10% of traces at max_new AND ...
LEAK_DIFF_HARD = 0.15            # ... label rate differs by >=15 pp between pinned and free traces
                                 # -> truncation leaks into labels (Steps 144, 168-172). Pinning alone
                                 # is only a flag: base-model QA cells (answer-span cropped) legitimately
                                 # run to the cap without the label depending on it.
CAP_PINNED_FLAG = 0.02


@dataclass
class LabelSanity:
    n: int
    n_pos: int
    n_neg: int
    acc: float
    cap_pinned_frac: Optional[float]
    cap_leak_diff: Optional[float] = None      # acc(pinned) - acc(free), when both groups exist
    hard: list = field(default_factory=list)   # reasons that make the AUROC meaningless
    flags: list = field(default_factory=list)  # reasons that make it noisy

    @property
    def ok(self) -> bool:
        return not self.hard

    def flag_string(self) -> str:
        """Compact tag for a CSV column: '' | 'FLAG:...' | 'DEGENERATE:...'."""
        parts = []
        if self.hard:
            parts.append("DEGENERATE:" + "|".join(self.hard))
        if self.flags:
            parts.append("FLAG:" + "|".join(self.flags))
        return ";".join(parts)

    def summary(self) -> str:
        cap = "n/a" if self.cap_pinned_frac is None else f"{self.cap_pinned_frac:.1%}"
        if self.cap_leak_diff is not None:
            cap += f" (acc pinned-free {self.cap_leak_diff:+.2f})"
        verdict = "OK" if self.ok else "DEGENERATE"
        s = (f"label sanity: {verdict}  n={self.n} pos={self.n_pos} neg={self.n_neg} "
             f"acc={self.acc:.3f} cap_pinned={cap}")
        if self.hard:
            s += "\n   HARD: " + "; ".join(self.hard)
        if self.flags:
            s += "\n   flags: " + "; ".join(self.flags)
        return s


def check_labels(labels: Iterable, lengths: Optional[Sequence[int]] = None,
                 max_new: Optional[int] = None,
                 hard_acc_band=HARD_ACC_BAND, flag_acc_band=FLAG_ACC_BAND,
                 min_minority_hard=MIN_MINORITY_HARD, min_minority_flag=MIN_MINORITY_FLAG,
                 cap_hard=CAP_PINNED_HARD, cap_flag=CAP_PINNED_FLAG,
                 leak_hard=LEAK_DIFF_HARD) -> LabelSanity:
    """Classify a binary label vector (and optionally per-row trace lengths vs the
    generation cap) as OK / flagged / degenerate. Pure function, no I/O."""
    y_raw = np.asarray(list(labels), dtype=float)
    keep = ~np.isnan(y_raw)
    y = y_raw[keep]
    n = int(len(y))
    n_pos = int((y > 0.5).sum())
    n_neg = n - n_pos
    acc = n_pos / n if n else float("nan")
    res = LabelSanity(n=n, n_pos=n_pos, n_neg=n_neg, acc=acc, cap_pinned_frac=None)

    if n == 0:
        res.hard.append("no labels")
        return res
    minority = min(n_pos, n_neg)
    if minority == 0:
        res.hard.append(f"single-class labels (acc={acc:.3f})")
    elif minority < min_minority_hard:
        res.hard.append(f"minority class has {minority} rows (< {min_minority_hard})")
    elif minority < min_minority_flag:
        res.flags.append(f"minority class has {minority} rows (< {min_minority_flag})")

    if n and minority:
        if not (hard_acc_band[0] <= acc <= hard_acc_band[1]):
            res.hard.append(f"acc={acc:.3f} outside hard band {hard_acc_band}")
        elif acc < flag_acc_band[0]:
            res.flags.append("FLOOR")
        elif acc > flag_acc_band[1]:
            res.flags.append("CEILING")

    if lengths is not None and max_new:
        L_all = np.asarray([np.nan if x is None else float(x) for x in lengths], dtype=float)
        if len(L_all) == len(y_raw):
            L_all = L_all[keep]
            yL = y
        else:  # lengths not row-aligned with labels; use them for the fraction only
            yL = None
        L_ok = ~np.isnan(L_all)
        L = L_all[L_ok]
        if len(L):
            pinned = L >= int(max_new)
            frac = float(pinned.mean())
            res.cap_pinned_frac = frac
            leak = None
            if yL is not None and pinned.any() and (~pinned).any():
                yy = yL[L_ok]
                leak = float(yy[pinned].mean() - yy[~pinned].mean())
                res.cap_leak_diff = leak
            if frac >= cap_hard and leak is not None and abs(leak) >= leak_hard:
                res.hard.append(f"{frac:.1%} of traces pinned at max_new={max_new} and label rate differs "
                                f"by {leak:+.2f} between pinned and free traces: truncation leaks into labels")
            elif frac >= cap_flag:
                res.flags.append(f"{frac:.1%} traces at max_new={max_new}"
                                 + (f" (acc pinned-free {leak:+.2f})" if leak is not None else ""))
    return res


def trace_lengths_from_candidates(cands: Iterable[dict]) -> list:
    """Per-candidate generated length, preferring gen_token_ids, then token_entropies."""
    out = []
    for c in cands:
        ids = c.get("gen_token_ids")
        if ids is not None:
            out.append(len(ids))
            continue
        H = c.get("token_entropies")
        out.append(len(H) if H is not None else None)
    return out


def check_pkl(data: dict, max_new: Optional[int] = None, label_key: str = "label") -> LabelSanity:
    """Convenience for the raw replication-grid pkl schema {idx: {"candidates": [...]}}."""
    cands = [c for i in sorted(data.keys()) for c in data[i]["candidates"]]
    labels = [bool(c.get(label_key, c.get("label", False))) for c in cands]
    return check_labels(labels, trace_lengths_from_candidates(cands), max_new)


def feasibility_tag(n_scored: int, n_population: Optional[int]) -> str:
    """'FEASIBILITY' when a result was computed on fewer rows than the registered full
    population (Omri, 2026-09-07: subsets are feasibility checks only, never evidence).
    Empty string when the population is unknown or fully covered."""
    if n_population is None or n_population <= 0:
        return ""
    return "FEASIBILITY" if int(n_scored) < int(n_population) else ""


def gate_flag(acc) -> str:
    """'' if inside FLAG_ACC_BAND, else 'FLOOR'/'CEILING'. Same contract as the
    historical report_figs.gate_flag; kept here so there is one definition."""
    try:
        a = float(acc)
    except (TypeError, ValueError):
        return ""
    if a != a:
        return ""
    if a < FLAG_ACC_BAND[0]:
        return "FLOOR"
    if a > FLAG_ACC_BAND[1]:
        return "CEILING"
    return ""
