"""Synthetic analytic/native agreement and falsifiable attribution cases."""
import copy,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from reproduction.stage2w.common import ANCHORS,PHASES,GROUPS,TRAINABLE_NAMES,S9_NAMES,state_digest,encode_master
from reproduction.stage2w.math import corners,native,attribution,memory_attribution,geometry,decision
from reproduction.stage2w.verify import numpy_attribution,numpy_geometry,numpy_memory,analytic,equal,decode_numpy

def toy(step=4):
    before={n:torch.tensor([.0,.07,-.2]) for n in TRAINABLE_NAMES}
    ps=[torch.nn.Parameter(v.clone()) for v in before.values()]
    optimizer=torch.optim.AdamW(ps,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=True)
    if step:
        for p in ps:optimizer.state[p]=dict(step=torch.tensor(float(step)),exp_avg=torch.tensor([.02,-.03,.08]),exp_avg_sq=torch.tensor([.003,.004,.002]))
    return before,ps,optimizer

class Contracts(unittest.TestCase):
    def test_fixed_coverage(self):
        self.assertEqual(sum(map(len,PHASES.values())),72);self.assertEqual(len(ANCHORS),10)
        self.assertEqual(sum(len(a['references']) for a in ANCHORS.values()),12)
        self.assertEqual({a['step'] for a in ANCHORS.values()},{1,9,13,21})
        self.assertEqual(set(GROUPS['T12'])&set(S9_NAMES),set())
    def test_native_matches_real_cpu_adam_with_state(self):
        before,ps,opt=toy();g={n:torch.tensor([.15,-.05,-.04]) for n in before};saved=copy.deepcopy(opt.state_dict());h=state_digest((before,saved))
        after,states=native(before,saved,g,'cpu')
        self.assertEqual(h,state_digest((before,saved)))
        for n,p in zip(before,ps):p.grad=g[n]
        opt.step()
        for i,(n,p) in enumerate(zip(before,ps)):
            self.assertTrue(torch.equal(after[n],p));self.assertTrue(torch.equal(states[i]['exp_avg'],opt.state[p]['exp_avg']))
        self.assertLessEqual(analytic(before,saved,{'11':g},5,{'11':after}),2e-7)
    def test_initial_adam_zero_state(self):
        before,ps,opt=toy(0);g={n:torch.tensor([1e-10,2.,-3.]) for n in before};after,_=native(before,opt.state_dict(),g,'cpu')
        for n,p in zip(before,ps):p.grad=g[n]
        opt.step()
        self.assertTrue(all(torch.equal(after[n],p) for n,p in zip(before,ps)))
    def test_no_clipping_no_spillover(self):
        before,_,opt=toy();lm={n:torch.ones(3)*.001 for n in before};aux={n:torch.ones(3)*.002 for n in S9_NAMES};gs,cs=corners(lm,aux,.1)
        self.assertEqual(cs,(1.,1.));values={k:native(before,opt.state_dict(),g,'cpu')[0] for k,g in gs.items()}
        a=attribution(values);self.assertEqual(a['T12']['fp32_policy'],0);self.assertEqual(a['T21']['clip_squared_norm'],0.)
        equal(a,numpy_attribution(values))
    def test_clipping_spillover_and_zero_direct_T12(self):
        before,_,opt=toy();lm={n:torch.ones(3)*.2 for n in before};aux={n:torch.ones(3)*.5 for n in S9_NAMES};gs,cs=corners(lm,aux,.2)
        self.assertNotEqual(cs[0],cs[1]);values={k:native(before,opt.state_dict(),g,'cpu')[0] for k,g in gs.items()};a=attribution(values)
        self.assertGreater(a['T12']['fp32_policy'],0);self.assertGreater(a['T12']['bf16_policy'],0);self.assertEqual(a['T12']['fp32_direct_c0'],0)
        equal(a,numpy_attribution(values));self.assertEqual(decision({'x':{'attribution':a}})['spillover'],'CLIP_MEDIATED_BF16_SPILLOVER')
    def test_bf16_invisible_fp32_change(self):
        base={n:torch.ones(3) for n in TRAINABLE_NAMES};values={k:{n:v.clone() for n,v in base.items()} for k in ('00','01','10','11')}
        for k in ('01','11'):
            for n in values[k]:values[k][n]+=1e-5
        a=attribution(values);self.assertGreater(a['T12']['fp32_policy'],0);self.assertEqual(a['T12']['bf16_policy'],0)
        self.assertEqual(decision({'x':{'attribution':a}})['spillover'],'FP32_ONLY_CLIP_SPILLOVER')
    def test_zero_effect_defined_as_null(self):
        values={k:{n:torch.ones(3) for n in TRAINABLE_NAMES} for k in ('00','01','10','11')};a=attribution(values)
        self.assertIsNone(a['T21']['clip_projection']);self.assertEqual(decision({'x':{'attribution':a}})['undefined_anchors'],1)
    def test_geometry_opposition_and_zero_aux(self):
        lm={n:torch.ones(3) for n in TRAINABLE_NAMES};aux={n:-torch.ones(3) for n in S9_NAMES};g=geometry(lm,aux,1.)
        self.assertAlmostEqual(g['S9']['cosine'],-1.);self.assertEqual(g['S9']['cancellation_ratio'],0.);self.assertIsNone(g['T12']['cosine'])
        equal(g,numpy_geometry(lm,aux,1.),1e-10)
    def test_memory_numerator_and_zero_initial(self):
        before,_,opt=toy();g={n:torch.tensor([.1,-.2,.3]) for n in before};a=memory_attribution(before,opt.state_dict(),g,5)
        equal(a,numpy_memory(opt.state_dict(),g,5))
        for v in a.values():self.assertAlmostEqual(v['memory_projection']+v['current_projection'],1.)
        _,_,zero=toy(0);z=memory_attribution(before,zero.state_dict(),g,1);self.assertEqual(z['T21']['memory_projection'],0.)
    def test_independent_numpy_codec(self):
        before={n:torch.randn(3,4) for n in TRAINABLE_NAMES};after={n:v+torch.randn_like(v)*1e-5 for n,v in before.items()};bits=encode_master(before,after)
        for n in before:
            arr=np.bitwise_xor(before[n].numpy().view(np.int32),bits[n].numpy()).view(np.float32)
            self.assertTrue(np.array_equal(arr,after[n].numpy()))
    def test_scalar_and_matrix_lossless_numpy_decode(self):
        for before in (torch.tensor(1.),torch.randn(2,3),torch.tensor(-0.)):
            after=before+torch.ones_like(before)*1e-5;bits=torch.bitwise_xor(before.view(torch.int32),after.view(torch.int32))
            actual=decode_numpy(before,bits)
            self.assertEqual(actual.shape,before.shape);self.assertTrue(torch.equal(actual,after))
    def test_gpu_requires_explicit_execute_before_io(self):
        from reproduction.stage2w import gpu
        with tempfile.TemporaryDirectory() as folder,patch('sys.argv',['gpu','--output',folder+'/new','--code-commit','x','--phase','reference']):
            with self.assertRaises(RuntimeError):gpu.main()
            self.assertFalse(Path(folder+'/new').exists())
    def test_verifier_does_not_import_producer(self):
        source=Path(__file__).with_name('verify.py').read_text();self.assertNotIn('from reproduction.stage2w.math',source)
    def test_class_aggregation_preserves_negative_zero_identity(self):
        from reproduction.stage2w.verify import independent_gradients
        lm=[{n:torch.full((3,),.001,dtype=torch.bfloat16) for n in TRAINABLE_NAMES} for _ in range(3)]
        aux=[{n:torch.full((3,),-0.,dtype=torch.bfloat16) for n in S9_NAMES} for _ in range(3)]
        mean={n:torch.stack([g[n].float() for g in lm]).mean(0) for n in TRAINABLE_NAMES}
        balanced={n:(aux[0][n].float()+torch.stack([aux[1][n].float(),aux[2][n].float()]).mean(0))*.5 for n in S9_NAMES}
        gs,_=corners(mean,balanced,.1)
        r=dict(images=[dict(label=x) for x in ('normal','abnormal','abnormal')],aggregate_lm=state_digest(mean),aggregate_lce=state_digest(balanced),controller={'lambda_final':.1},policy={'arm':'dynamic'},applied_sha256=state_digest(gs['11']))
        with patch('reproduction.stage2w.verify.check_spatial'):
            _,actual,_,_=independent_gradients(dict(lm=lm,lce=aux,attention_maps=[None]*3),r)
        self.assertEqual(state_digest(actual),state_digest(balanced))
    def test_negative_dominance_not_hidden(self):
        values={k:{n:torch.zeros(1) for n in TRAINABLE_NAMES} for k in ('00','01','10','11')}
        for n in S9_NAMES:values['10'][n]+=2;values['01'][n]-=1;values['11'][n]+=1
        a=attribution(values);self.assertLess(a['T21']['clip_projection'],0);self.assertEqual(decision({'x':{'attribution':a}})['dominance'],'STATE_DEPENDENT_MIXED')

if __name__=='__main__':unittest.main()
