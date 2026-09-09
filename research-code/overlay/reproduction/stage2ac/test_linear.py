import unittest
import numpy as np
from scipy.special import expit
from reproduction.stage2ac.linear import fit_head,predict


class LinearTests(unittest.TestCase):
    def test_convex_fit_stationarity_and_constant_feature(self):
        x=np.array([[-2,1],[-1,1],[0,1],[1,1],[2,1],[3,1]],dtype=float)
        y=np.array([0,0,0,1,1,1])
        h=fit_head(x,y);scores=predict(h,x)
        self.assertTrue(np.all((scores>=0)==y))
        self.assertEqual(h['scale'][1],1.)
        z=(x-h['mean'])/h['scale'];res=(expit(scores)-y)/len(y)
        self.assertLess(np.max(np.abs(z.T@res+h['penalty']*h['weight'])),1e-6)

    def test_test_distribution_does_not_change_normalization(self):
        x=np.arange(12,dtype=float).reshape(6,2);y=np.array([0,0,0,1,1,1])
        h=fit_head(x,y);before=h['mean'].copy()
        predict(h,np.ones((10,2))*1000)
        np.testing.assert_array_equal(h['mean'],before)
        np.testing.assert_array_equal(h['mean'],x.mean(0))

    def test_row_order_does_not_select_different_head(self):
        x=np.array([[0,1],[2,0],[1,2],[4,3],[3,4]],dtype=float);y=np.array([0,0,1,1,1])
        a=fit_head(x,y);b=fit_head(x[::-1],y[::-1])
        np.testing.assert_allclose(predict(a,x),predict(b,x),atol=1e-6,rtol=1e-6)

    def test_invalid_training_inputs_fail(self):
        with self.assertRaises(ValueError):fit_head([[0],[1]],[0,0])
        with self.assertRaises(ValueError):fit_head([[float('nan')],[1]],[0,1])


if __name__=='__main__':unittest.main()
