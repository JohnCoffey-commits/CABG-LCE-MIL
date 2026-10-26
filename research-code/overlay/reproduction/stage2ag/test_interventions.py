"""Small synthetic contract tests, run in the existing server environment."""
import unittest
import torch
from reproduction.stage2ag.interventions import replacement

class Tests(unittest.TestCase):
    def setUp(self):
        self.a=torch.linspace(-3,6,4096).reshape(1,4,1,1024)
        self.n=torch.linspace(2,-4,4096).reshape(1,4,1,1024)
        self.e=torch.sigmoid(self.a)-torch.sigmoid(self.n)
    def test_identity_and_clone(self):
        self.assertIs(replacement(self.a,self.n,self.e,'native'),self.e)
        self.assertTrue(torch.equal(replacement(self.a,self.n,self.e,'clone'),self.e))
    def test_centered_common_shift_and_layer_rms(self):
        x=replacement(self.a,self.n,self.e,'center_rms')
        # Independent definition from centered sigmoid probabilities.
        d=(self.a-self.n)/2
        independent=torch.sigmoid(d)-torch.sigmoid(-d)
        independent*=self.e.square().mean(-1,keepdim=True).sqrt()/independent.square().mean(-1,keepdim=True).sqrt()
        self.assertTrue(torch.allclose(x,independent,atol=3e-7,rtol=1e-5))
        self.assertTrue(torch.allclose(x.square().mean(-1),self.e.square().mean(-1),atol=1e-7))
        # With target RMS fixed, common mode is mathematically removed.
        shifted=replacement(self.a+8,self.n+8,self.e,'center_rms')
        self.assertTrue(torch.allclose(x,shifted,atol=1e-6))
    def test_sham_preserves_values_but_changes_location(self):
        x=replacement(self.a,self.n,self.e,'softsign_rms')
        z=replacement(self.a,self.n,self.e,'softsign_shuffle_rms')
        self.assertTrue(torch.equal(x.sort(-1).values,z.sort(-1).values))
        self.assertFalse(torch.equal(x,z))
    def test_zero_is_finite_and_unknown_rejected(self):
        z=torch.zeros_like(self.a)
        self.assertTrue(torch.equal(replacement(z,z,z,'center_rms'),z))
        with self.assertRaises(ValueError):replacement(z,z,z,'unknown')

if __name__=='__main__':unittest.main()
