import numpy as np
from spectral_utils.family_tail_transfer import tail_marks, build_representations, load_lock, score_locked


def test_quota_ties_centering_and_short_answers():
    x = np.array([[0., 7.], [1., 7.], [2., 7.], [2., 7.], [2., 7.]])
    raw = tail_marks(x, np.array([0, 5]), centred=False)
    np.testing.assert_allclose(raw[:, 0], [0, 0, 1/3, 1/3, 1/3])
    np.testing.assert_allclose(raw.sum(0), 1)
    centered = tail_marks(x, np.array([0, 5]))
    np.testing.assert_allclose(centered.mean(0), 0, atol=1e-16)
    np.testing.assert_array_equal(centered[:, 1], 0)
    np.testing.assert_array_equal(tail_marks(x[:1], np.array([0, 1])), 0)
    # The same recipe without ties really has binary uncentered marks.
    distinct = tail_marks(np.arange(10.)[:, None], np.array([0, 10]), centred=False)
    np.testing.assert_array_equal(distinct[:, 0], [0]*8+[1]*2)


def test_named_alignment_missing_channels_and_frozen_threshold():
    names = load_lock()['recipe']['channels_28']
    x = np.random.default_rng(24).normal(size=(30, 28)); off = np.array([0, 10, 30])
    first = score_locked(x, names, off)
    order = np.arange(28)[::-1]
    second = score_locked(x[:, order], [names[i] for i in order], off)
    for method in first:
        np.testing.assert_allclose(first[method]['scores'], second[method]['scores'])
        np.testing.assert_array_equal(first[method]['pred_valid'], second[method]['pred_valid'])
        assert first[method]['threshold'] == load_lock()['deployment'][method]['q80_threshold_fold4']
    try:
        build_representations(x[:, :-1], names[:-1], off)
    except ValueError as exc:
        assert 'missing locked channels' in str(exc)
    else:
        raise AssertionError('missing source feature accepted')
