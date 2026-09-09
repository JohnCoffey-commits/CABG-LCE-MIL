"""CPU regressions for registered phases, untouched common prefixes and fail-closed gates."""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2l.controller import DynamicController
from reproduction.stage2q.run import make_optimizer,update
from reproduction.stage2q.execution import cpu_tree
from reproduction.stage2s.common import state_digest,write
from reproduction.stage2s.trajectory import predict_next
from reproduction.stage2u.policy import gradients
from reproduction.stage2u.common import encode_master,decode_master
from reproduction.stage2v.common import PHASES,EVALUATIONS,configuration,require_execution,require_phase_ready,parent,eval_checkpoint,Budget,LIMITS,disk_guard
from reproduction.stage2v.verify import decide

class Contracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def plan(self):return dict(code_commit='fixed',source_sha256='source',protocol_sha256='protocol',train_manifest_sha256='train',eval_manifest_sha256='eval')
    def test_only_seed_changes_original_configuration(self):
        a=configuration(42)
        for seed in (43,44):
            b=configuration(seed);self.assertEqual({k:v for k,v in a.items() if k!='seed'},{k:v for k,v in b.items() if k!='seed'})
            self.assertEqual(b['seed'],seed)
        with self.assertRaises(ValueError):configuration(45)
    def test_budgeted_phase_graph_has_no_extra_cutoff_or_arm(self):
        self.assertEqual(sum(c['end']-c['start'] for c in PHASES.values()),74)
        self.assertEqual(sum(c['end']-c['start'] for c in PHASES.values() if c['mode']=='train'),72)
        self.assertEqual(len(EVALUATIONS)*32,224)
        self.assertEqual({c['seed'] for c in PHASES.values() if c['mode']=='train'},{43,44})
        self.assertTrue(all(c['start']==12 for c in PHASES.values() if c['arm']=='lce_off'))
    def test_no_update_replay_clones_match_real_toy_adam(self):
        torch.manual_seed(4);initial={n:torch.randn(7).bfloat16() for n in TRAINABLE_NAMES}
        masters,opt,sched=make_optimizer(initial,'cpu')
        for _ in range(12):update(masters,opt,sched,{n:torch.randn(7)*.01 for n in TRAINABLE_NAMES})
        applied={n:torch.randn(7)*.01 for n in TRAINABLE_NAMES}
        before=state_digest([cpu_tree(masters),cpu_tree(opt.state_dict()),sched.state_dict()])
        predicted=predict_next(masters,opt,applied)
        self.assertEqual(before,state_digest([cpu_tree(masters),cpu_tree(opt.state_dict()),sched.state_dict()]))
        update(masters,opt,sched,applied)
        self.assertLess(max(float((predicted[n]-masters[n]).abs().max()) for n in TRAINABLE_NAMES),2e-7)
    def test_same_parent_first_raw_shadow_and_moments_preserved(self):
        torch.manual_seed(43)
        initial={n:torch.randn(7).bfloat16() for n in TRAINABLE_NAMES};masters,opt,sched=make_optimizer(initial,'cpu')
        for _ in range(12):update(masters,opt,sched,{n:torch.randn(7)*.01 for n in TRAINABLE_NAMES})
        parent=copy.deepcopy(opt.state_dict());moment_sha=state_digest(parent)
        lm=[{n:torch.randn(7).bfloat16().float() for n in TRAINABLE_NAMES} for _ in range(3)]
        aux=[{n:torch.randn(7).bfloat16().float() for n in S9_NAMES} for _ in range(3)]
        raw_sha=state_digest([lm,aux]);outputs=[];states=[]
        for arm in ('dynamic','lce_off'):
            c=DynamicController(lm_ema=1.,lce_ema=1.,valid_blocks=12)
            g,d=gradients(lm,aux,['normal','abnormal','abnormal'],controller=c,arm=arm,frozen=0.)
            outputs.append(state_digest(g));states.append(c.state_dict())
            self.assertEqual(raw_sha,state_digest([lm,aux]));self.assertEqual(moment_sha,state_digest(opt.state_dict()))
        self.assertNotEqual(outputs[0],outputs[1]);self.assertEqual(states[0],states[1])
    def test_parents_are_seed_local_and_fixed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);plan=self.plan()
            for seed in (43,44):
                paths=[parent(root,f's{seed}-{arm}',plan)[0] for arm in ('dynamic','lce_off')]
                self.assertEqual(paths,[root/f's{seed}-prefix/block12-checkpoint/checkpoint.pt']*2)
                self.assertIsNone(parent(root,f's{seed}-prefix',plan)[0])
                self.assertEqual(eval_checkpoint(root,f'eval-s{seed}-parent12',plan)[0],paths[0])
    def test_missing_execution_flag_fails_before_files(self):
        with self.assertRaises(RuntimeError):require_execution(SimpleNamespace(execute=False,phase='s43-prefix'))
        with self.assertRaises(ValueError):require_execution(SimpleNamespace(execute=True,phase='s45-prefix'))
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'must-not-exist'
            p=subprocess.run([sys.executable,'-m','reproduction.stage2v.train','--campaign',str(root),'--code-commit','bad','--phase','s43-prefix'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('--execute',p.stderr);self.assertFalse(root.exists())
    def test_bridge_and_all_training_gate_development_access(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            with self.assertRaises(FileNotFoundError):require_phase_ready(root,'s43-prefix')
            write(root/'bridge-verification.json',{'status':'PASS'})
            require_phase_ready(root,'s43-prefix')
            with self.assertRaises(FileNotFoundError):require_phase_ready(root,'eval-s43-parent12',evaluation=True)
            for name,cfg in PHASES.items():
                if cfg['mode']=='train':
                    (root/name).mkdir();write(root/name/'result.json',{'status':'SUCCESS'})
            require_phase_ready(root,'eval-s43-parent12',evaluation=True)
            require_phase_ready(root,'s44-lce_off')
    def test_scientific_nonreplication_is_valid_and_no_seed_replacement(self):
        report={str(s):{'dynamic':{'spatial':{'concentrated':True}},'lce_off':{'spatial':{'stable':True}}} for s in (43,44)}
        self.assertEqual(decide(report)[0],'SPATIAL_CONTRAST_REPLICATED_2_OF_2')
        report['44']['dynamic']['spatial']['concentrated']=False
        self.assertEqual(decide(report)[0],'INITIALIZATION_SENSITIVE_OR_NOT_REPLICATED')
        self.assertEqual(decide(report)[1],{'43':True,'44':False})
        report['44']['dynamic']['spatial']['concentrated']=True;report['44']['lce_off']['spatial']['stable']=False
        self.assertFalse(decide(report)[1]['44'])
    def test_cumulative_hard_limits_survive_budget_reopen(self):
        with tempfile.TemporaryDirectory() as t,patch('reproduction.stage2v.common.disk_guard'),patch('reproduction.stage2v.common.signal.signal'),patch('reproduction.stage2v.common.signal.alarm'):
            b=Budget(t)
            for kind,count in LIMITS.items():
                for _ in range(count):b.reserve(kind,'CPU test')
                with self.assertRaises(RuntimeError):Budget(t).reserve(kind,'overflow')
            self.assertEqual(json.loads(b.path.read_text())['counts'],LIMITS)
    def test_storage_reservation_refuses_excess(self):
        with tempfile.TemporaryDirectory() as t:
            write(Path(t)/'budget.json',{})
            with patch('reproduction.stage2v.common.shutil.disk_usage',return_value=SimpleNamespace(free=40_000_000_000)),patch('reproduction.stage2v.common.artifact_bytes',return_value=29_600_000_000):
                disk_guard(t,100_000_000)
                with self.assertRaises(RuntimeError):disk_guard(t,500_000_000)
            with patch('reproduction.stage2v.common.shutil.disk_usage',return_value=SimpleNamespace(free=16_000_000_001)):
                with self.assertRaises(RuntimeError):disk_guard(t,2)

if __name__=='__main__':unittest.main()
