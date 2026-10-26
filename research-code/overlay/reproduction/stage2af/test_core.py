"""Independent analytic checks for loss derivatives and applied gradient budget."""
import math
import unittest
import numpy as np
import torch
from reproduction.stage2af.core import compute_lce,compose,DynamicController,multiplier,fixed_coefficient

class CoreTests(unittest.TestCase):
    def test_lce_numpy_value_and_analytic_derivative(self):
        a=torch.linspace(-3,4,4096,dtype=torch.float32).reshape(1,4,1024).requires_grad_()
        n=torch.cos(torch.arange(4096,dtype=torch.float32)/17).reshape(1,4,1024).requires_grad_()
        result=compute_lce(a,n,['abnormal']);ga,gn=torch.autograd.grad(result['per_image_loss'].sum(),[a,n])
        gap=a.detach().numpy().astype('float64')-n.detach().numpy().astype('float64')
        c=(gap/(1+abs(gap))).mean(1);tau=1/math.log(1024)
        e=np.exp((c-c.max(1,keepdims=True))/tau);weights=e/e.sum(1,keepdims=True)
        score=c.max(1)+tau*np.log(e.mean(1));loss=np.logaddexp(0,-score)
        derivative=(-1/(1+np.exp(score)))[:,None,None]*weights[:,None,:]/4/(1+abs(gap))**2
        np.testing.assert_allclose(result['score'].detach(),score,atol=2e-6,rtol=0)
        np.testing.assert_allclose(result['per_image_loss'].detach(),loss,atol=2e-6,rtol=0)
        np.testing.assert_allclose(ga,derivative,atol=2e-7,rtol=3e-5)
        np.testing.assert_allclose(gn,-derivative,atol=2e-7,rtol=3e-5)

    def fixture(self):
        lm=[{'p':torch.tensor([1.,2.]),'q':torch.tensor([.2])},
            {'p':torch.tensor([2.,-.5]),'q':torch.tensor([.1])},
            {'p':torch.tensor([.5,1.]),'q':torch.tensor([.4])}]
        aux=[{'p':torch.tensor([2.,.5])},{'p':torch.tensor([-1.,2.])},{'p':torch.tensor([1.,3.])}]
        return lm,aux,['normal','abnormal','abnormal']

    def test_fixed_is_genuinely_fixed_without_dynamic_cap(self):
        lm,aux,labels=self.fixture();coefficient=3.
        actual,details=compose(lm,aux,labels,'fixed_lce',DynamicController(),coefficient,('p','q'),('p',))
        base=np.mean([np.r_[g['p'].numpy(),g['q'].numpy()] for g in lm],axis=0)
        extra=.5*aux[0]['p'].numpy()+.25*(aux[1]['p'].numpy()+aux[2]['p'].numpy())
        expected=base.copy();expected[:2]+=coefficient*extra
        expected/=max(1,np.linalg.norm(expected))
        np.testing.assert_allclose(np.r_[actual['p'],actual['q']],expected,atol=2e-6,rtol=0)
        self.assertEqual(details['applied_coefficient'],3.)
        self.assertFalse(details['applied_dynamic_cap'])
        self.assertGreater(coefficient,details['budget']['lambda_cap'])

    def test_dynamic_respects_cap_and_lm_is_zero_auxiliary(self):
        lm,aux,labels=self.fixture()
        out,details=compose(lm,aux,labels,'cabg_lce',DynamicController(),None,('p','q'),('p',))
        self.assertLessEqual(details['applied_coefficient'],details['budget']['lambda_cap'])
        out,details=compose(lm,aux,labels,'lm_only',DynamicController(),None,('p','q'),('p',))
        expected=np.mean([np.r_[g['p'].numpy(),g['q'].numpy()] for g in lm],axis=0)
        expected/=max(1,np.linalg.norm(expected))
        np.testing.assert_allclose(np.r_[out['p'],out['q']],expected,atol=2e-6,rtol=0)
        self.assertEqual(details['applied_coefficient'],0)

    def test_horizon_and_fixed_selection(self):
        self.assertEqual(multiplier(0),1);self.assertGreater(multiplier(24),.99)
        self.assertAlmostEqual(multiplier(500),.5);self.assertEqual(multiplier(1000),0)
        self.assertEqual(fixed_coefficient([{'coefficient':x} for x in range(1,9)]),4.5)
        with self.assertRaises(ValueError):fixed_coefficient([{'coefficient':1}]*7)

if __name__=='__main__':unittest.main()
