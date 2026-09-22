"""Regression checks for independent ProcessBench metric units."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.review_rbm_literature_completion_v2 import pb_harmonic

class MetricUnits(unittest.TestCase):
    def test_fraction_and_display_percent(self):
        self.assertEqual(pb_harmonic(1.,1.),1.)
        self.assertAlmostEqual(pb_harmonic(.5,.25),1/3)
        self.assertAlmostEqual(100*pb_harmonic(.5,.25),100/3)
    def test_zero_component_is_zero(self):
        self.assertEqual(pb_harmonic(0.,0.),0.)
        self.assertEqual(pb_harmonic(1.,0.),0.)
        self.assertEqual(pb_harmonic(0.,1.),0.)

if __name__=='__main__':unittest.main()
