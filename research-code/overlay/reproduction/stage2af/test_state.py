import copy,unittest
import numpy as np
import torch
from reproduction.stage2af.exact_state import equal_state
class StateTests(unittest.TestCase):
    def test_numpy_rng_and_tensor_mutations(self):
        x={'rng':('MT19937',np.arange(8,dtype=np.uint32),2,0,0.),'master':torch.tensor([1.,2.])}
        equal_state(x,copy.deepcopy(x))
        y=copy.deepcopy(x);y['rng'][1][3]+=1
        with self.assertRaises(AssertionError):equal_state(x,y)
        y=copy.deepcopy(x);y['master'][0]+=1
        with self.assertRaises(AssertionError):equal_state(x,y)
if __name__=='__main__':unittest.main()
