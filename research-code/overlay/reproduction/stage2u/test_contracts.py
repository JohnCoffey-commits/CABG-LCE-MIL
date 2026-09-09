"""CPU-only intervention, preservation, lossless evidence and authorization checks."""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from reproduction.stage2i.fingerprint import canonical_json_sha256
from reproduction.stage2l.controller import DynamicController,compute_block_gradients
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2q.execution import cpu_tree
from reproduction.stage2q.run import make_optimizer,update
from reproduction.stage2s.common import state_digest
from reproduction.stage2t.verify import spatial,adam
from reproduction.stage2u.common import encode_master,decode_master,require_execution,Budget,LIMITS
from reproduction.stage2u.policy import gradients,selected_lambda
from reproduction.stage2u.verify import reconstruct

class Causal(unittest.TestCase):
    def setUp(self):torch.manual_seed(9);torch.set_num_threads(2)
    def fixture(self):
        lm=[{n:torch.randn(7).bfloat16().float() for n in TRAINABLE_NAMES} for _ in range(3)]
        lce=[{n:torch.randn(7).bfloat16().float() for n in S9_NAMES} for _ in range(3)]
        return lm,lce,['normal','abnormal','abnormal']
    def test_explicit_execution_gate_before_work(self):
        for arm in ('dynamic','frozen_ratio','lce_off'):
            with self.assertRaisesRegex(RuntimeError,'explicit user'):require_execution(SimpleNamespace(execute=False,arm=arm))
        with self.assertRaises(ValueError):require_execution(SimpleNamespace(execute=True,arm='other'))
    def test_cli_blocks_without_execute(self):
        result=subprocess.run([sys.executable,'-m','reproduction.stage2u.train','--campaign','/nonexistent/forbidden','--plan','missing','--arm','dynamic','--code-commit','bad'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('explicit user GPU approval',result.stderr)
        self.assertFalse(Path('/nonexistent/forbidden').exists())
    def test_policy_uses_only_parent_ratio_and_current_cap(self):
        for cap in (.01,.2):
            budget=dict(lambda_final=.007,lambda_cap=cap)
            self.assertEqual(selected_lambda('dynamic',budget,.1),.007)
            self.assertEqual(selected_lambda('frozen_ratio',budget,.1),min(.1,cap))
            self.assertEqual(selected_lambda('lce_off',budget,.1),0)
        with self.assertRaises(ValueError):selected_lambda('frozen_ratio',budget,float('nan'))
    def test_dynamic_exact_and_one_shadow_update(self):
        lm,lce,labels=self.fixture();a=DynamicController();b=DynamicController()
        x,d=compute_block_gradients(lm,lce,labels,trainable_names=TRAINABLE_NAMES,s9_names=S9_NAMES,controller=a)
        y,e=gradients(lm,lce,labels,controller=b,arm='dynamic',frozen=.1)
        self.assertEqual(state_digest(x),state_digest(y));self.assertEqual(a.state_dict(),b.state_dict());self.assertEqual(b.valid_blocks,1)
    def test_all_policies_keep_raw_gradients_and_shadow(self):
        lm,lce,labels=self.fixture();original=state_digest([lm,lce]);states=[]
        for arm in ('dynamic','frozen_ratio','lce_off'):
            c=DynamicController();g,d=gradients(lm,lce,labels,controller=c,arm=arm,frozen=.1)
            self.assertEqual(original,state_digest([lm,lce]));states.append(c.state_dict())
        self.assertEqual(states[0],states[1]);self.assertEqual(states[0],states[2])
    def test_xor_codec_exact_including_signed_zero(self):
        a={'v':torch.tensor([0.,-0.,1.,-3.,1e-30]),'w':torch.randn(7,4)}
        b={'v':torch.tensor([-0.,0.,1.000001,-2.99999,-1e-30]),'w':a['w']+.00001}
        result=decode_master(a,encode_master(a,b));self.assertEqual(state_digest(result),state_digest(b))
    def test_shadow_and_actual_policy_independent_reconstruction(self):
        lm,lce,labels=self.fixture()
        for arm in ('dynamic','frozen_ratio','lce_off'):
            controller=DynamicController();before=controller.state_dict()
            applied,d=gradients(lm,lce,labels,controller=controller,arm=arm,frozen=.1)
            maps=[{k:torch.randn(1,4,1024).bfloat16() for k in ('abnormal_raw','normal_raw')} for _ in labels]
            rows=[dict(label=l,**spatial(m,l)) for m,l in zip(maps,labels)]
            raw={'lm':[{n:v.bfloat16() for n,v in g.items()} for g in lm],'lce':[{n:v.bfloat16() for n,v in g.items()} for g in lce],'attention_maps':maps}
            rec=dict(images=rows,block=1,controller=d['budget'],controller_before=before,policy=d['policy'],gradient=d['gradient'],aggregate_lm=state_digest(d['aggregate_lm']),aggregate_lce=state_digest(d['aggregate_lce']),applied_sha256=state_digest(applied))
            self.assertEqual(state_digest(reconstruct(raw,rec,.1)),state_digest(applied))
            rec['policy']['selected_lambda']+=.001
            with self.assertRaises(AssertionError):reconstruct(raw,rec,.1)
    def test_off_preserves_parent_moments_then_adam_updates_them(self):
        initial={n:torch.randn(7).bfloat16() for n in TRAINABLE_NAMES};masters,opt,sched=make_optimizer(initial,'cpu')
        for _ in range(12):update(masters,opt,sched,{n:torch.randn(7)*.01 for n in TRAINABLE_NAMES})
        checkpoint=copy.deepcopy(opt.state_dict());lm,lce,labels=self.fixture();controller=DynamicController(lm_ema=1.,lce_ema=1.,valid_blocks=12)
        applied,details=gradients(lm,lce,labels,controller=controller,arm='lce_off',frozen=.1)
        self.assertEqual(state_digest(opt.state_dict()),state_digest(checkpoint))
        before=cpu_tree(masters);group=checkpoint['param_groups'][0]
        moments={n:(checkpoint['state'][i]['exp_avg'].double(),checkpoint['state'][i]['exp_avg_sq'].double()) for n,i in zip(TRAINABLE_NAMES,group['params'])}
        lr=opt.param_groups[0]['lr'];update(masters,opt,sched,applied)
        self.assertLessEqual(adam({'applied':applied,'updated_master':cpu_tree(masters)},dict(block=13,update={'learning_rate_used':lr}),before,moments),2e-7)
        self.assertTrue(all(float(s['step'])==13 for s in opt.state.values()))
    def test_write_reservation_tracks_actual_payload_bound(self):
        from reproduction.stage2u.common import disk_guard,tensor_bytes
        self.assertEqual(tensor_bytes({'x':torch.ones(7),'y':[torch.ones(3).bfloat16()]}),34)
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'budget.json').write_text('{}')
            with patch('reproduction.stage2u.common.shutil.disk_usage',return_value=SimpleNamespace(free=6_000_000_000)),patch('reproduction.stage2u.common.artifact_bytes',return_value=11_200_000_000):
                disk_guard(t,needed=100_000_000)
                with self.assertRaisesRegex(RuntimeError,'artifact'):disk_guard(t,needed=900_000_000)
            with patch('reproduction.stage2u.common.shutil.disk_usage',return_value=SimpleNamespace(free=4_050_000_000)):
                with self.assertRaisesRegex(RuntimeError,'free-space'):disk_guard(t,needed=100_000_000)
    def test_cumulative_operation_limits(self):
        with tempfile.TemporaryDirectory() as t,patch('reproduction.stage2u.common.disk_guard'),patch('reproduction.stage2u.common.signal.signal'),patch('reproduction.stage2u.common.signal.alarm'):
            b=Budget(t)
            for kind,count in LIMITS.items():
                for _ in range(count):b.reserve(kind,'CPU test')
                with self.assertRaises(RuntimeError):b.reserve(kind,'overflow')
            self.assertEqual(json.loads(b.path.read_text())['counts'],LIMITS)

if __name__=='__main__':unittest.main()
