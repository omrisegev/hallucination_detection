"""Rebuild nested telemetry banks and the CHAIN L-SML reference from a FEATURES file.

Two exact ports, kept here so experiments outside the cluster mirror use the same arithmetic:

* :func:`build_nested_bank` is ``results/monday_external_panel_v1/score.py:build_bank`` for banks without
  ``lf__`` (level-oriented) channels, which covers every nested bank N11..N56: plain channels are standardized
  inside each answer, the three digit channels only over their active steps.
* :func:`chain_lsml` is the CHAIN construction of ``results/prm_boost_ablation5_v1/ablation5.py``: continuous L-SML
  on the bank, answer-z of its group virtual classifiers, then continuous L-SML over the PRM channel and those
  scores with every view its own group, sign anchored to the PRM channel.

Callers must check parity against a published score before trusting a rebuilt bank.
"""

import numpy as np

from .fusion_utils import lsml_continuous_guarded
from .lsml_gate_locator_research import answer_standardize


DIGIT = ('digit_alternative', 'digit_spread', 'digit_alternative_innovation')
LEVEL = ('q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level')


def masked_std(x, offsets, active):
    """Answer standardization over active steps only; inactive steps stay zero (score.py:masked_std)."""
    out = np.zeros(x.shape)
    for a, b in zip(offsets[:-1], offsets[1:]):
        for j in range(x.shape[1]):
            v = active[a:b, j]; y = x[a:b, j][v]
            if len(y) and y.std() > 1e-12:
                out[a:b, j][v] = (y - y.mean()) / y.std()
    return out


def build_nested_bank(values, names, digit_active, offsets, columns):
    """Bank matrix (steps x columns) exactly as score.py:build_bank, for banks without lf__ channels."""
    names = [str(n) for n in names]
    if any(str(c).startswith('lf__') for c in columns):
        raise ValueError('level-oriented lf__ channels are not supported by this port')
    plain = [c for c in columns if c not in DIGIT]
    Vp = answer_standardize(np.asarray(values, float)[:, [names.index(c) for c in plain]], offsets)
    Vd = masked_std(np.asarray(values, float)[:, [names.index(c) for c in DIGIT]], offsets,
                    np.asarray(digit_active, bool))
    out = np.zeros((len(values), len(columns)))
    for j, c in enumerate(columns):
        out[:, j] = Vd[:, DIGIT.index(c)] if c in DIGIT else Vp[:, plain.index(c)]
    return out


def level_score(values, names, offsets):
    """The line's label-free level: mean answer-z of the five LEVEL channels (score.py Population.level)."""
    names = [str(n) for n in names]
    return answer_standardize(np.asarray(values, float)[:, [names.index(c) for c in LEVEL]], offsets).mean(1)


def answer_z(s, offsets):
    """Per-answer z-score with the ablation5 / evaluate_collection floor (std floor 1e-8)."""
    out = np.zeros(len(s))
    for a, b in zip(offsets[:-1], offsets[1:]):
        v = s[a:b]; out[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return out


def anchored_lsml(V, groups=None, anchor=None):
    """ablation5.lsml: guarded continuous L-SML, flipped when it correlates negatively with the anchor."""
    s, meta = lsml_continuous_guarded(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float)
    ref = V[:, 0] if anchor is None else anchor
    if np.corrcoef(s, ref)[0, 1] < 0:
        s = -s
    return s, meta


def chain_lsml(zp, V, offsets):
    """CHAIN of ablation5: returns (chain_score, tele_score, tele_meta)."""
    tele, meta = anchored_lsml(V, anchor=V.mean(1))
    vg = np.asarray(meta['virtual_classifiers'], float)
    VC = np.column_stack([zp] + [answer_z(vg[:, g], offsets) for g in range(vg.shape[1])])
    chain, _ = anchored_lsml(VC, groups=np.arange(VC.shape[1]))
    return chain, tele, meta


__all__ = ['DIGIT', 'LEVEL', 'anchored_lsml', 'answer_z', 'build_nested_bank', 'chain_lsml', 'level_score',
           'masked_std']
