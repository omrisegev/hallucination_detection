import numpy as np
from spectral_utils.joint_structured_stress import structured_augmentation,standardize_block
from spectral_utils.joint_noise_aware import score_noise_aware_joint


def test_perturbation_contract_and_identity_stability():
    rng=np.random.default_rng(9);x=np.vstack((standardize_block(rng.normal(size=(120,51))),np.zeros((1,51))))
    for kind in ('near_copies','structured_noise'):
        a=structured_augmentation(x,[0,120,121],['a','b'],kind)
        b=structured_augmentation(np.vstack((x[120:],x[:120])),[0,1,121],['b','a'],kind)
        np.testing.assert_array_equal(a[:,:51],x)
        np.testing.assert_array_equal(a[:120,51:],b[1:,51:])
        np.testing.assert_array_equal(a[120,51:],np.zeros(15))
        np.testing.assert_allclose(a[:120,51:].mean(0),0,atol=1e-14)
        np.testing.assert_allclose(a[:120,51:].std(0),1,atol=1e-14)
        if kind=='near_copies':
            assert not np.array_equal(a[:120,51:],x[:120,:15])
            assert all(np.corrcoef(x[:120,i],a[:120,51+i])[0,1]>.99 for i in range(15))
        else:
            c=np.corrcoef(a[:120,51:],rowvar=False)
            assert np.median(c[np.triu_indices(15,1)])>.4


def test_invalid_model_requires_explicit_fallback():
    model=dict(valid=False,input_width=3,anchor_index=1,failure='fixture')
    x=np.arange(12).reshape(4,3)
    try:score_noise_aware_joint(model,x)
    except ValueError as exc:assert 'INVALID_JOINT_MODEL' in str(exc)
    else:raise AssertionError('hidden fallback')
    np.testing.assert_array_equal(score_noise_aware_joint(model,x,allow_anchor_fallback=True),x[:,1])
