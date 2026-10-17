import unittest
import torch
from spectral_utils.temporal_context_models import TelemetryTCN


class TCNArchitecture(unittest.TestCase):
    def test_effective_receptive_field_is_fifteen(self):
        torch.manual_seed(388);torch.set_num_threads(1)
        model=TelemetryTCN(dimensions=5).eval()
        history=torch.randn(3,16,5,requires_grad=True)
        mask=torch.ones(3,16,dtype=torch.bool);position=torch.tensor([.1,.5,.9])
        mean,var=model(history,mask,position)
        mean.sum().backward()
        self.assertEqual(float(history.grad[:,0].abs().sum()),0.)
        self.assertGreater(float(history.grad[:,1:].abs().sum()),0.)
        changed=history.detach().clone();changed[:,0]+=1000
        with torch.no_grad():other_mean,other_var=model(changed,mask,position)
        torch.testing.assert_close(other_mean,mean.detach(),rtol=0,atol=0)
        torch.testing.assert_close(other_var,var.detach(),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
