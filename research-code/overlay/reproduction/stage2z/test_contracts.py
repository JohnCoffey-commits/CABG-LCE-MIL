"""Synthetic fixed-baseline policy, recovery, budget and verdict contracts."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2l.controller import DynamicController
from reproduction.stage2q.run import make_optimizer,update
from reproduction.stage2q.verify import norm32
from reproduction.stage2q.execution import cpu_tree
from reproduction.stage2s.common import state_digest,write
from reproduction.stage2s.trajectory import predict_next
from reproduction.stage2u.policy import gradients
from reproduction.stage2z.common import PHASES,EVALUATIONS,SEEDS,BOUNDARIES,PLANNED,LIMITS,Budget,parent,require_execution,require_phase_ready,original_phase,disk_guard
from reproduction.stage2z.analysis import relation,decide,summary
from reproduction.stage2z.verify import verdict,start_identity
from reproduction.stage2i.rng_state import snapshot_rng_state

class Contracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def plan(self):return dict(code_commit='fixed',source_sha256='source',protocol_sha256='protocol',train_manifest_sha256='train',eval_manifest_sha256='eval')
    def test_start_identity_handles_numpy_rng_and_detects_changes(self):
        value=dict(trainable={'p':torch.ones(3,dtype=torch.bfloat16)},runtime={'cache':torch.zeros(2)},rng=snapshot_rng_state())
        self.assertEqual(start_identity(value),start_identity(copy.deepcopy(value)))
        for field in ('trainable','runtime','rng'):
            changed=copy.deepcopy(value)
            if field=='rng':changed['rng']['numpy'][1][0]^=1
            elif field=='runtime':changed[field]['cache'][0]=1
            else:changed[field]['p'][0]=2
            self.assertNotEqual(start_identity(value),start_identity(changed))
    def test_start_identity_rejects_unchecked_fields(self):
        with self.assertRaises(AssertionError):start_identity(dict(trainable={},runtime={},rng={},extra='unchecked'))
    def summaries(self,delta=.1):
        return {str(s):dict(lm_only=dict(auroc=.7,average_precision=.7),dynamic=dict(auroc=.7+delta,average_precision=.7+delta),early_only=dict(auroc=.7+delta,average_precision=.7+delta)) for s in SEEDS}
    def check_decision(self,summaries):
        expected,comparisons=verdict(summaries)
        producer={s:{a:relation(x[a],x['lm_only']) for a in ('dynamic','early_only')} for s,x in summaries.items()}
        self.assertEqual(comparisons,producer);self.assertEqual(expected,decide(producer));return expected
    def test_complete_phase_graph(self):
        self.assertEqual(sum(c['end']-c['start'] for c in PHASES.values()),50)
        self.assertEqual(sum(c['end']-c['start'] for c in PHASES.values() if c['mode']=='train'),48)
        self.assertEqual((len(EVALUATIONS)*32,len(PHASES)+len(EVALUATIONS)),(128,8))
        self.assertEqual(PLANNED,dict(update=48,block=50,eval=128,model=8))
    def test_zero_policy_from_initialization_only_fixed_seeds(self):
        self.assertEqual({c['arm'] for c in PHASES.values()},{'lce_off'});self.assertEqual({c['seed'] for c in PHASES.values()},{43,44})
        self.assertTrue(all((c['start'],c['end'])==(0,24) for c in PHASES.values() if c['mode']=='train'))
        self.assertEqual(BOUNDARIES,(12,24))
    def test_own_midpoint_parent(self):
        root=Path('/synthetic');plan=self.plan()
        for s in SEEDS:
            self.assertEqual(parent(root,f'replay-s{s}-b13',plan)[0],root/f's{s}-lm_only/block12-checkpoint/checkpoint.pt')
            self.assertEqual(parent(root,f's{s}-lm_only',plan),(None,None))
    def test_execution_authorization_gate(self):
        with self.assertRaises(RuntimeError):require_execution(SimpleNamespace(execute=False,phase='s43-lm_only'))
        with self.assertRaises(ValueError):require_execution(SimpleNamespace(execute=True,phase='s45-lm_only'))
    def test_training_and_reference_gates_precede_new_eval(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t)
            with self.assertRaises(FileNotFoundError):require_phase_ready(r,'eval-ref-s43-dynamic24',True)
            write(r/'training-verification.json',dict(status='PASS'))
            require_phase_ready(r,'eval-ref-s43-dynamic24',True)
            with self.assertRaises(FileNotFoundError):require_phase_ready(r,'eval-s43-lm_only24',True)
            write(r/'reference-verification.json',dict(status='PASS'));require_phase_ready(r,'eval-s43-lm_only24',True)
    def test_all_train_before_replay(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t)
            for s in SEEDS:
                with self.assertRaises(FileNotFoundError):require_phase_ready(r,'replay-s43-b13')
                (r/f's{s}-lm_only').mkdir();write(r/f's{s}-lm_only/result.json',dict(status='SUCCESS'))
            require_phase_ready(r,'replay-s43-b13')
    def test_pure_lm_gradient_and_shadow_independence(self):
        torch.manual_seed(11)
        lm=[{n:torch.randn(7).bfloat16().float() for n in TRAINABLE_NAMES} for _ in range(3)]
        aux=[{n:torch.randn(7).bfloat16().float() for n in S9_NAMES} for _ in range(3)]
        outputs=[];states=[]
        for factor in (1.,8.):
            c=DynamicController();g,d=gradients(lm,[{n:v*factor for n,v in x.items()} for x in aux],['normal','abnormal','abnormal'],controller=c,arm='lce_off',frozen=0.)
            self.assertEqual(d['policy']['selected_lambda'],0.);outputs.append(g);states.append(c.state_dict())
        expected={n:torch.stack([g[n] for g in lm]).mean(0) for n in TRAINABLE_NAMES};scale=min(1.,1./(norm32(expected)+1e-12))
        self.assertTrue(all(torch.equal(outputs[0][n],expected[n]*scale) for n in TRAINABLE_NAMES))
        self.assertEqual(state_digest(outputs[0]),state_digest(outputs[1]));self.assertNotEqual(states[0],states[1])
    def test_zero_applied_lce_keeps_full_t21_learning(self):
        torch.manual_seed(9);initial={n:torch.zeros(7,dtype=torch.bfloat16) for n in TRAINABLE_NAMES};m,o,s=make_optimizer(initial,'cpu')
        lm=[{n:torch.ones(7)*.01 for n in TRAINABLE_NAMES} for _ in range(3)];aux=[{n:torch.ones(7) for n in S9_NAMES} for _ in range(3)]
        g,_=gradients(lm,aux,['normal','abnormal','abnormal'],controller=DynamicController(),arm='lce_off',frozen=0.)
        update(m,o,s,g);self.assertTrue(all(torch.count_nonzero(m[n])==7 for n in TRAINABLE_NAMES));self.assertEqual(len(o.state),21)
    def test_midpoint_prediction_is_read_only_and_matches_update(self):
        torch.manual_seed(5);initial={n:torch.randn(7).bfloat16() for n in TRAINABLE_NAMES};m,o,s=make_optimizer(initial,'cpu')
        for _ in range(12):update(m,o,s,{n:torch.randn(7)*.01 for n in TRAINABLE_NAMES})
        g={n:torch.randn(7)*.01 for n in TRAINABLE_NAMES};before=state_digest([cpu_tree(m),cpu_tree(o.state_dict()),s.state_dict()])
        prediction=predict_next(m,o,g);self.assertEqual(before,state_digest([cpu_tree(m),cpu_tree(o.state_dict()),s.state_dict()]))
        update(m,o,s,g);self.assertLess(max(float((prediction[n]-m[n]).abs().max()) for n in TRAINABLE_NAMES),2e-7)
    def test_reference_phase_follows_original_full_trajectory(self):
        self.assertEqual([original_phase(43,k) for k in (1,12,13,24)],['s43-prefix','s43-prefix','s43-dynamic','s43-dynamic'])
        with self.assertRaises(AssertionError):original_phase(45,1)
    def test_both_descriptive_gain(self):
        self.assertEqual(set(self.check_decision(self.summaries()).values()),{'DESCRIPTIVE_LCE_GAIN_2_OF_2'})
    def test_lm_ties_are_retained(self):
        self.assertEqual(set(self.check_decision(self.summaries(0.)).values()),{'LM_ONLY_MATCHES_OR_EXCEEDS_2_OF_2'})
    def test_mixed_seeds(self):
        s=self.summaries();s['44']=self.summaries(-.1)['44']
        self.assertEqual(set(self.check_decision(s).values()),{'MIXED_OR_METRIC_DEPENDENT_GAIN'})
    def test_metric_discordance_and_single_metric_tie(self):
        for value in (.6,.7):
            s=self.summaries();s['43']['dynamic']['average_precision']=value
            self.assertEqual(self.check_decision(s)['dynamic'],'MIXED_OR_METRIC_DEPENDENT_GAIN')
    def test_early_only_is_separate_not_selected(self):
        s=self.summaries();s['44']['early_only']=dict(auroc=.6,average_precision=.6)
        d=self.check_decision(s);self.assertEqual(d['dynamic'],'DESCRIPTIVE_LCE_GAIN_2_OF_2');self.assertEqual(d['early_only'],'MIXED_OR_METRIC_DEPENDENT_GAIN')
    def test_missing_seed_invalid(self):
        s=self.summaries();del s['44']
        with self.assertRaises(AssertionError):verdict(s)
        with self.assertRaises(AssertionError):decide({'43':{}})
    def test_tie_aware_metrics(self):
        rows=[dict(label=l,score=.5,effective_support=500.,top11_mass=.1,lm_loss=.2,valid_tokens=2) for l in ('normal','abnormal')]
        m=summary(rows);self.assertEqual((m['auroc'],m['average_precision']),(.5,.5))
    def test_budget_cumulative_across_reopen(self):
        with tempfile.TemporaryDirectory() as t,patch('reproduction.stage2z.common.disk_guard'),patch('reproduction.stage2z.common.signal.signal'),patch('reproduction.stage2z.common.signal.alarm'):
            b=Budget(t)
            for kind,count in LIMITS.items():
                for _ in range(count):b.reserve(kind,'synthetic')
                with self.assertRaises(RuntimeError):Budget(t).reserve(kind,'overflow')
            self.assertEqual(json.loads(b.path.read_text())['counts'],LIMITS)
    def test_new_storage_ceiling(self):
        with tempfile.TemporaryDirectory() as t:
            write(Path(t)/'budget.json',{})
            with patch('reproduction.stage2z.common.shutil.disk_usage',return_value=SimpleNamespace(free=40_000_000_000)),patch('reproduction.stage2z.common.artifact_bytes',return_value=23_600_000_000):
                disk_guard(t,100_000_000)
                with self.assertRaises(RuntimeError):disk_guard(t,500_000_000)
if __name__=='__main__':unittest.main()
