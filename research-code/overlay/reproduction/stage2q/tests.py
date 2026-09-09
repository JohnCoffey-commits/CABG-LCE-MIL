import unittest
import torch
from reproduction.stage2q.verify import compare_tensor,compare_map,small_enough
from reproduction.stage2q.execution import AttentionExecution
from reproduction.stage2q.run import runtime_state,restore_runtime
from reproduction.stage2i.rng_state import snapshot_rng_state,restore_rng_state,rng_state_fingerprint


class ReproTests(unittest.TestCase):
    def test_exact_and_sign_partition(self):
        x=torch.tensor([0.,1.,-1.,1e-14]);y=torch.tensor([0.,1.,-1.,-1e-14])
        self.assertTrue(compare_tensor(x,x)["exact"])
        c=compare_tensor(x,y);self.assertEqual(c["meaningful_sign_disagreements"],0);self.assertEqual(c["near_zero_sign_disagreements"],1)
        y[1]=-1;self.assertEqual(compare_tensor(x,y)["meaningful_sign_disagreements"],1)

    def test_nonfinite_and_support_fail_closed(self):
        with self.assertRaises(RuntimeError):compare_tensor(torch.tensor([float('nan')]),torch.ones(1))
        with self.assertRaises(RuntimeError):compare_map({'a':torch.ones(1)},{'b':torch.ones(1)})
        with self.assertRaises(RuntimeError):AttentionExecution('unknown')

    def test_rng_restored(self):
        s=snapshot_rng_state();before=rng_state_fingerprint(s);torch.rand(5);restore_rng_state(s)
        self.assertEqual(before,rng_state_fingerprint(snapshot_rng_state()))

    def test_buffers_and_cache_restored(self):
        model=torch.nn.Linear(2,2);model.register_buffer('test',torch.ones(2));model.rope_deltas=None
        s=runtime_state(model);model.test.zero_();model.rope_deltas=torch.ones(1);restore_runtime(model,s)
        self.assertTrue(torch.equal(model.test,torch.ones(2)));self.assertIsNone(model.rope_deltas)

    def test_residual_not_automatically_accepted(self):
        c=compare_map({'a':torch.tensor([1.,2.])},{'a':torch.tensor([1.001,2.])})
        self.assertFalse(small_enough(c))

    def test_deterministic_backward_is_enforced_at_actual_entrypoint(self):
        execution=AttentionExecution('flash_deterministic').install()
        try:
            with self.assertRaises(RuntimeError):
                execution.interface._wrapped_flash_attn_backward(*([None]*16+[False]))
            with self.assertRaises(RuntimeError):
                execution.interface._wrapped_flash_attn_varlen_backward(*([None]*20+[False]))
        finally:execution.close()

    def test_grad_collection_does_not_mutate_parameters_or_grad(self):
        p=torch.nn.Parameter(torch.tensor([2.,3.]));before=p.detach().clone()
        first=torch.autograd.grad((p*p).sum(),p)[0]
        second=torch.autograd.grad((p*p).sum(),p)[0]
        self.assertIsNone(p.grad);self.assertTrue(torch.equal(p,before));self.assertTrue(torch.equal(first,second))

    def test_map_cosine_and_l2(self):
        c=compare_map({'p':torch.tensor([1.,0.])},{'p':torch.tensor([0.,1.])})
        self.assertAlmostEqual(c['absolute_l2'],2**.5);self.assertEqual(c['cosine'],0.)

    def test_no_unregistered_horizon_in_producer(self):
        from pathlib import Path
        source=(Path(__file__).parent/'run.py').read_text()
        self.assertIn('args.blocks not in (4,12)',source)
        self.assertIn('args.blocks != 13',source)
        self.assertNotIn('evaluate_scout',source)


if __name__=='__main__':unittest.main()
