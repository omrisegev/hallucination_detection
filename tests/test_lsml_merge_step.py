import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
import lsml_merge_step as MS  # noqa: E402


def block_R(sizes, within, between=0.0, cross=None):
    """Correlation matrix of blocks with equicorrelation `within[b]`, `between` elsewhere, and optional cross[(b1, b2)]."""
    lab = np.repeat(np.arange(len(sizes)), sizes); p = len(lab); R = np.full((p, p), between)
    for b, r in enumerate(within):
        R[np.ix_(lab == b, lab == b)] = r
    for (b1, b2), r in (cross or {}).items():
        R[np.ix_(lab == b1, lab == b2)] = r; R[np.ix_(lab == b2, lab == b1)] = r
    np.fill_diagonal(R, 1.0)
    return R, lab


def test_merges_only_the_split_common_factor():
    # one factor of 6 channels split into groups 0 and 1, three unrelated blocks of 3
    R, _ = block_R([3, 3, 3, 3, 3], [0.6, 0.6, 0.5, 0.5, 0.5], between=0.05, cross={(0, 1): 0.6})
    g0 = np.repeat(np.arange(5), 3)
    g, seq = MS.absorb_merge(R, g0)
    assert np.array_equal(g, np.repeat([0, 0, 1, 2, 3], 3))
    assert seq[0]['a'] == 0 and seq[0]['b'] == 1 and seq[0]['rho'] == pytest.approx(0.4 / 2.2)
    assert seq[-1]['stop'] == 'rho' and seq[-1]['rho'] >= 0.5


def test_unrelated_groups_give_rho_one_whatever_their_sizes():
    for sizes in ([1, 1, 1], [2, 5, 3], [8, 1, 4]):
        R, lab = block_R(sizes, [0.5] * len(sizes), between=0.0)
        g, seq = MS.absorb_merge(R, lab)
        assert np.array_equal(g, lab) and seq == [{'stop': 'rho', 'rho': pytest.approx(1.0)}]


def test_halves_of_one_factor_rho_formula():
    # rho = (1 - r) / (1 + (p - 1) r) for two halves of size p of one equicorrelated block (plus two other groups)
    for p, r in [(1, 0.8), (2, 0.6), (4, 0.3)]:
        R, lab = block_R([p, p, 2, 2], [r, r, 0.5, 0.5], cross={(0, 1): r})
        g, seq = MS.absorb_merge(R, lab, thr=0.99)
        assert seq[0]['rho'] == pytest.approx((1 - r) / (1 + (p - 1) * r))


def test_never_below_min_groups():
    R, lab = block_R([2, 2, 2], [0.7, 0.7, 0.7], between=0.7)          # one factor over three groups
    g, seq = MS.absorb_merge(R, lab)
    assert g.max() + 1 == 3 and seq == [{'stop': 'min_groups', 'rho': pytest.approx(0.3 / 1.7)}]
    R4, lab4 = block_R([2, 2, 2, 2], [0.7] * 4, between=0.7)
    g4, seq4 = MS.absorb_merge(R4, lab4)
    assert g4.max() + 1 == 3 and 'a' in seq4[0] and seq4[-1]['stop'] == 'min_groups'


def test_singletons_merge_iff_abs_r_above_half():
    R, lab = block_R([1, 1, 2, 2], [1, 1, 0.5, 0.5], cross={(0, 1): 0.8})
    assert MS.absorb_merge(R, lab)[0].max() + 1 == 3
    R, lab = block_R([1, 1, 2, 2], [1, 1, 0.5, 0.5], cross={(0, 1): 0.3})
    assert MS.absorb_merge(R, lab)[0].max() + 1 == 4


def test_labels_are_canonical_and_input_untouched():
    R, _ = block_R([2, 2, 2, 2], [0.6] * 4, cross={(0, 3): 0.6})
    g0 = np.array([7, 7, 3, 3, 5, 5, 9, 9]); keep = g0.copy()
    g, seq = MS.absorb_merge(R, g0)
    assert np.array_equal(g0, keep) and set(g.tolist()) == {0, 1, 2}
    assert g[0] == g[6] and len({g[0], g[2], g[4]}) == 3


def test_rejects_bad_matrix():
    with pytest.raises(ValueError):
        MS.absorb_merge(np.full((3, 3), np.nan), [0, 1, 2])
    with pytest.raises(ValueError):
        MS.absorb_merge(np.eye(4), [0, 1, 2])


def test_band_select_boundaries():
    keep, flip, drop = MS.band_select([0.6, 0.55, 0.5, 0.45, 0.44, 0.551, 0.449])
    assert keep.tolist() == [0, 5] and flip.tolist() == [4, 6] and drop.tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        MS.band_select([0.5, np.nan])


def test_dependence_split():
    R, lab = block_R([2, 2], [0.8, 0.6], between=0.1)
    d = MS.dependence_split(R, lab)
    assert d['between'] == {'pairs': 4, 'mean_abs': pytest.approx(0.1), 'max_abs': pytest.approx(0.1)}
    assert d['within']['pairs'] == 2 and d['within']['mean_abs'] == pytest.approx(0.7)
