import unittest
import numpy as np
import torch
from spectral_utils.temporal_context_models import (ConditionalFlow, flow_objective, flow_dot,
    probability_features, probability_pgd, history_windows, TelemetryTCN, residual_step_score)
from spectral_utils.renyi_alpha_sweep import escort_varentropy


class FlowFixtures(unittest.TestCase):
    def test_straight_and_curved_paths(self):
        class Straight(torch.nn.Module):
            def forward(self, x, t, c): return torch.ones_like(x)*2
        class Curved(torch.nn.Module):
            def forward(self, x, t, c): return torch.ones_like(x)*(2*t)
        x = torch.zeros(3, 4, dtype=torch.float64); c = torch.zeros(3, 1, dtype=torch.float64)
        endpoint, dot = flow_dot(Straight(), c, x, steps=50)
        torch.testing.assert_close(endpoint, torch.full_like(x, 2))
        self.assertLess(float(dot.max()), 1e-12)
        endpoint, curved = flow_dot(Curved(), c, x, steps=50, solver="heun")
        torch.testing.assert_close(endpoint, torch.ones_like(x))
        expected = sum(abs((i/50)**2-i/50) for i in range(1,51))
        torch.testing.assert_close(curved, torch.full_like(curved, expected))

    def test_zero_penalties_exact_fm_gradient(self):
        torch.manual_seed(15)
        model = ConditionalFlow(6, width=8, depth=1)
        x, c, u = torch.randn(7,4), torch.randn(7,6), torch.randn(7,4)
        a, _ = flow_objective(model, x, .5, u, c, -c, repel_weight=0, curve_weight=0)
        b = (model(x,.5,c)-u).square().sum(-1).mean()
        ga = torch.autograd.grad(a, tuple(model.parameters()))
        gb = torch.autograd.grad(b, tuple(model.parameters()))
        for left, right in zip(ga,gb): torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_probability_features_and_pgd_constraint(self):
        torch.manual_seed(2)
        logits = torch.randn(8,15,dtype=torch.float64)
        q = torch.softmax(logits,-1).numpy()
        expected = np.column_stack((np.log(q+1e-12).mean(-1), *[escort_varentropy(q,a) for a in (0,.75,1)]))
        actual = probability_features(logits, torch.ones(4,dtype=torch.float64))
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)
        model = ConditionalFlow(4,width=8,depth=1).double()
        x,u = torch.randn(8,4,dtype=torch.float64),torch.randn(8,4,dtype=torch.float64)
        make = lambda lp: probability_features(lp,torch.ones(4,dtype=torch.float64))
        negative, adv = probability_pgd(model,x,.2,u,logits,make,epsilon=.1)
        self.assertLessEqual(float((adv-logits).abs().max()), .100000000001)
        torch.testing.assert_close(negative,make(adv))
        torch.testing.assert_close(torch.softmax(adv,-1).sum(-1),torch.ones(8,dtype=torch.float64))

    def test_no_future_observation_in_context(self):
        x = np.arange(80).reshape(20,4).astype(float)
        before, mask = history_windows(x,[0,7,19],history=4)
        self.assertFalse(mask[0].any()); np.testing.assert_array_equal(before[1],x[3:7])
        changed=x.copy(); changed[7:]=9999
        np.testing.assert_array_equal(history_windows(changed,[7],history=4)[0][0],before[1])
        current,_=history_windows(x,[7],history=4,include_current=True)
        np.testing.assert_array_equal(current[0],x[3:8])

    def test_tcn_causal_internal_block_and_residual_identity(self):
        torch.manual_seed(3); model=TelemetryTCN(width=8)
        x=torch.randn(2,5,16); changed=x.clone();changed[:,:,8:]=100
        torch.testing.assert_close(model.blocks(x)[:,:,:8],model.blocks(changed)[:,:,:8])
        mean,var=model(torch.randn(2,16,4),torch.ones(2,16),torch.tensor([.3,.7]))
        self.assertEqual(mean.shape,(2,4));self.assertTrue(bool((var>0).all()))
        base=np.array([1.,3.,-2.]); auxiliary=np.array([0.,1.,0.])
        np.testing.assert_array_equal(residual_step_score(base,auxiliary,0),base)
        np.testing.assert_array_equal(residual_step_score(base,np.ones(3),1),base)


if __name__ == "__main__": unittest.main()
