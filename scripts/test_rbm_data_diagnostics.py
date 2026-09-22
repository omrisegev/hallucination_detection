"""Numerical and label-unit acceptance tests, no learned models."""
import ast
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_utils.rbm_data_diagnostics import *


class DiagnosticsTests(unittest.TestCase):
    def test_claude_exact_rule(self):
        source=ROOT.parent/'readout-provenance-v1/spectral_utils/provenance_readout.py'
        tree=ast.parse(source.read_text(encoding='utf8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='first_near_max')
        ns={'np':np,'NEAR_MAX_SD':.25}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        rng=np.random.default_rng(192)
        for s in [np.array([]),np.array([1.]),np.zeros(8),np.array([1.,1.,.99,0.]),
                  *[rng.normal(size=n) for n in range(2,50)]]:
            np.testing.assert_array_equal(first_near_max(s),ns['first_near_max'](s))

    def test_unknown_and_step_labels(self):
        x=np.array([[1.],[3.],[100.]])
        auc,counts=auc_columns(x,np.array([0,1,-1]))
        self.assertEqual(counts,(1,1));self.assertEqual(auc[0],1)
        result=class_diagnostics(x,np.array([0,1,-1]),np.ones((1,3),bool))
        self.assertTrue(np.isnan(result['class_log_variance_ratio']).all())
        x=np.array([[1.],[1.],[2.],[3.]])
        result=class_diagnostics(x,np.array([0,0,1,1]),np.ones((1,4),bool))
        self.assertTrue(result['class_zero_variance'][0,0,0])
        self.assertTrue(np.isnan(result['class_log_variance_ratio'][0,0]))

    def test_synthetic_exact_moments(self):
        a=np.array([.2,-.5]);w=np.array([1.2,-.7]);b=-.3
        pi=expit(b+a@w+.5*w@w)
        x=sample_model(200000,a,w,b,np.random.default_rng(775))
        np.testing.assert_allclose(x.mean(axis=0),a+pi*w,atol=.012,rtol=0)
        np.testing.assert_allclose(np.cov(x,rowvar=False),np.eye(2)+pi*(1-pi)*np.outer(w,w),atol=.015,rtol=0)

    def test_lag_uses_actual_positions(self):
        x=np.arange(12.)[:,None];mask=np.arange(12)%2==0
        corr,count=lag_correlations(x,mask[None])
        self.assertEqual(count[0,0],0);self.assertEqual(count[0,1],5)
        self.assertTrue(np.isnan(corr[0,0,0]));self.assertAlmostEqual(corr[0,1,0],1)

    def test_driver_has_no_fitting_calls(self):
        tree=ast.parse((ROOT/'scripts/run_rbm_data_diagnostics.py').read_text())
        names=[n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else ''
               for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(set(names)&{'fit','fit_rbm','fit_all','minimize','backward','upcr_fit'})


if __name__=='__main__':unittest.main()
