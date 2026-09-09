"""Synthetic mathematical and execution-boundary tests, not research outcomes."""
import math,unittest
import torch
from reproduction.stage2x.common import variants,forbidden,ANCHORS
from reproduction.stage2x.math import response,contrast,decision
from reproduction.stage2x.verify import measure,effects,verdict

class Contracts(unittest.TestCase):
    def raw(self,x=0.):
        return dict(maps=dict(abnormal_raw=torch.full((1,4,1024),x),normal_raw=torch.zeros(1,4,1024)),lm_logits=torch.tensor([[0.,0.],[0.,0.]]),targets=torch.tensor([0,1]))
    def reports(self,a,b,eq=False,clip=True):
        return {str(i):dict(means=dict(log_support=dict(total=a),top11=dict(total=b)),policy_raw_equal=eq,clip_raw_equal=clip) for i in range(10)}
    def test_uniform(self):
        x=self.raw();r=response(x['maps'],'normal');self.assertAlmostEqual(r['log_support'],math.log(1024));self.assertAlmostEqual(r['top11'],11/1024)
    def test_independent_uniform_lm(self):
        self.assertAlmostEqual(measure(self.raw(),'normal')['lm_loss'],math.log(2))
    def test_independent_bf16_attention(self):
        x=self.raw();x['maps']={k:v.bfloat16() for k,v in x['maps'].items()}
        self.assertAlmostEqual(measure(x,'normal')['log_support'],math.log(1024))
    def test_nonuniform_independent(self):
        x=self.raw();x['maps']['abnormal_raw'][0,:,0]=7.
        a=response(x['maps'],'abnormal');b=measure(x,'abnormal')
        for k in a:self.assertAlmostEqual(a[k],b[k],places=11)
    def test_concentration_direction(self):
        x=self.raw();x['maps']['abnormal_raw'][0,:,0]=7.;r=response(x['maps'],'normal')
        self.assertLess(r['log_support'],math.log(1024));self.assertGreater(r['top11'],11/1024)
    def test_labels_change_loss_only(self):
        x=self.raw(.5);a=response(x['maps'],'normal');b=response(x['maps'],'abnormal')
        self.assertEqual(a['log_support'],b['log_support']);self.assertGreater(a['lce_loss'],b['lce_loss'])
    def test_two_factor_closure(self):
        x={k:dict(y=v) for k,v in zip(('00','01','10','11'),(0.,2.,3.,11.))}
        self.assertEqual(contrast(x),effects(x));self.assertEqual(contrast(x)['y']['direct']+contrast(x)['y']['clip'],11.)
    def test_no_clip(self):
        x={k:dict(y=v) for k,v in zip(('00','01','10','11'),(1.,1.,4.,4.))};self.assertEqual(contrast(x)['y']['clip'],0.)
    def test_negative_component(self):
        x={k:dict(y=v) for k,v in zip(('00','01','10','11'),(0.,-1.,3.,2.))};self.assertLess(contrast(x)['y']['clip'],0.)
    def test_all_concentrating(self):
        r=self.reports(-.1,.1);self.assertEqual(decision(r),verdict(r));self.assertEqual(decision(r)['policy_response'],'CONSISTENT_LOCAL_CONCENTRATING_RESPONSE')
    def test_all_diffusing(self):
        r=self.reports(.1,-.1);self.assertEqual(decision(r),verdict(r));self.assertEqual(decision(r)['policy_response'],'CONSISTENT_LOCAL_DIFFUSING_RESPONSE')
    def test_mixed_includes_zero(self):
        r=self.reports(-.1,.1);r['9']['means']['log_support']['total']=0.
        self.assertEqual(decision(r)['policy_response'],'STATE_DEPENDENT_OR_MIXED_LOCAL_RESPONSE')
    def test_null(self):
        r=self.reports(0.,0.,eq=True);self.assertEqual(decision(r),verdict(r));self.assertEqual(decision(r)['policy_response'],'NO_RESOLVED_LOCAL_POLICY_RESPONSE')
    def test_clipping_raw_label(self):
        r=self.reports(0.,0.);r['0']['clip_raw_equal']=False;self.assertEqual(decision(r)['clipping_response'],'CLIP_MEDIATED_ATTENTION_RESPONSE')
    def test_exact_full_hash_alias(self):
        r=dict(corners={k:dict(bf16_sha256=h) for k,h in zip(('00','01','10','11'),('a','a','b','b'))})
        self.assertEqual(variants(r),(['00','10'],dict(zip(('00','01','10','11'),('00','00','10','10')))))
    def test_unique_anchors(self):
        self.assertEqual(len(ANCHORS),10);self.assertEqual({c['step'] for c in ANCHORS.values()},{1,9,13,21})
    def test_training_forbidden(self):
        with self.assertRaises(RuntimeError):forbidden()
if __name__=='__main__':unittest.main()
