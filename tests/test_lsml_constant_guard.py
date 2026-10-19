"""fusion_utils.lsml_continuous_guarded: constant views are dropped before L-SML and get weight 0."""
import numpy as np
import pytest

from spectral_utils.fusion_utils import lsml_continuous, lsml_continuous_guarded


def views(n=600, seed=0):
    """Eight z-scored views of one latent class, two of them sharing a nuisance (one dependent pair)."""
    rng = np.random.default_rng(seed)
    y = rng.choice([-1., 1.], n); shared = rng.normal(size=n)
    out = []
    for j, a in enumerate((.9, .8, .7, .7, .6, .6, .5, .5)):
        x = a * y + rng.normal(size=n) + (1.2 * shared if j in (4, 5) else 0.)
        out.append((x - x.mean()) / x.std())
    return out


def test_without_constant_views_it_is_lsml_continuous():
    v = views()
    f0, m0 = lsml_continuous(*v)
    f1, m1 = lsml_continuous_guarded(*v)
    assert np.array_equal(f0, f1) and np.array_equal(m0['c'], m1['c']) and m1['dropped_constant'] == []
    g = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    assert np.array_equal(lsml_continuous(*v, groups=g)[0], lsml_continuous_guarded(*v, groups=g)[0])


@pytest.mark.parametrize('where', [(8, 9, 10), (0, 5, 10), (2,)])
def test_constant_views_are_dropped_and_reinserted(where):
    v = views(); ref, mref = lsml_continuous(*v)
    full = list(v)
    for k in sorted(where):
        full.insert(k, np.zeros_like(v[0]) if k % 2 else np.full_like(v[0], 3.0))
    fused, meta = lsml_continuous_guarded(*full)
    assert np.allclose(fused, ref) and meta['K'] == mref['K']
    assert meta['dropped_constant'] == sorted(where) and len(meta['c']) == len(full)
    assert all(meta['c'][k] == -1 for k in where)
    kept = [j for j in range(len(full)) if j not in where]
    assert np.array_equal(meta['c'][kept], mref['c'])
    used = sorted(int(i) for idx, _ in meta['group_weights'] for i in idx)
    assert used == kept


def test_supplied_groups_skip_dropped_views():
    v = views(); g = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    ref, _ = lsml_continuous(*v, groups=g)
    fused, meta = lsml_continuous_guarded(*(v + [np.zeros_like(v[0])]), groups=np.r_[g, 9])
    assert np.allclose(fused, ref) and meta['c'][-1] == -1 and meta['K'] == 4


def test_too_few_informative_views_fail_loudly():
    v = views()[:2] + [np.zeros(600), np.ones(600)]
    with pytest.raises(ValueError, match='only 2 non-constant views'):
        lsml_continuous_guarded(*v)
    with pytest.raises(ValueError, match='groups has length'):
        lsml_continuous_guarded(*views(), groups=[0, 1])
