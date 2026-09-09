import math
import unittest
import numpy as np
from reproduction.stage2ac.verify import cabg_score


class ScoreReconstruction(unittest.TestCase):
    def test_formal_temperature_matters_for_nonconstant_map(self):
        a=np.zeros((1,4,1024),dtype=np.float32);a[:,:,:512]=1
        actual=cabg_score(a,np.zeros_like(a))
        tau=1/math.log(1024)
        expected=tau*math.log((math.exp(.5/tau)+1)/2)
        self.assertAlmostEqual(actual,expected,places=12)
        self.assertGreater(abs(actual-math.log((math.exp(.5)+1)/2)),.05)

    def test_zero_gap_and_constant_gap(self):
        a=np.zeros((1,4,1024),dtype=np.float32)
        self.assertAlmostEqual(cabg_score(a,a),0,places=12)
        self.assertAlmostEqual(cabg_score(a+1,a),.5,places=12)

    def test_wrong_geometry_fails(self):
        with self.assertRaises(AssertionError):cabg_score(np.zeros((1,4,32)),np.zeros((1,4,32)))


if __name__=='__main__':unittest.main()
