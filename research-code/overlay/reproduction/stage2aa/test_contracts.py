"""Synthetic timing, inherited-state, exact-rank and bounded execution contracts."""
import copy,tempfile,unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from reproduction.stage2l.controller import DynamicController
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2q.run import make_optimizer,update
from reproduction.stage2q.execution import cpu_tree
from reproduction.stage2s.common import state_digest,write
from reproduction.stage2s.trajectory import predict_next
from reproduction.stage2u.policy import gradients
from reproduction.stage2aa.common import PHASES,EVALUATIONS,SEEDS,BOUNDARIES,PLANNED,LIMITS,CONTRASTS,parent,eval_checkpoint,require_execution,require_phase_ready,Budget,disk_guard
from reproduction.stage2aa.analysis import exact_ranks,relations,decide
from reproduction.stage2aa.verify import exact_ranking,verdict,INDEPENDENT_CONTRASTS,start_identity
from reproduction.stage2i.rng_state import snapshot_rng_state

class Contracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def plan(self):return dict(code_commit='fixed',source_sha256='source',protocol_sha256='protocol',train_manifest_sha256='train',eval_manifest_sha256='eval')
    def ranks(self,none=Fraction(1,2),early=Fraction(3,4),late=Fraction(2,3),full=Fraction(4,5)):
        return {str(s):{h:dict(auroc=v,average_precision=v) for h,v in dict(none=none,early=early,late=late,full=full).items()} for s in SEEDS}
    def compare(self,ranks):
        v,c=verdict(ranks);p={s:relations(r) for s,r in ranks.items()};self.assertEqual(c,p);self.assertEqual(v,decide(p));return v
    def test_complete_fixed_phase_graph(self):
        self.assertEqual(PLANNED,dict(update=24,block=26,eval=128,model=8))
        self.assertEqual(sum(c['end']-c['start'] for c in PHASES.values()),26)
        self.assertEqual(sum(c['end']-c['start'] for c in PHASES.values() if c['mode']=='train'),24)
        self.assertEqual(len(EVALUATIONS)*32,128);self.assertEqual(len(PHASES)+len(EVALUATIONS),8)
        self.assertEqual(BOUNDARIES,(20,24))
    def test_registered_dynamic_timing_only(self):
        self.assertEqual(SEEDS,(43,44));self.assertTrue(all(c['arm']=='dynamic' for c in PHASES.values()))
        self.assertEqual([(c['start'],c['end']) for c in PHASES.values()],[(12,24)]*2+[(20,21)]*2)
    def test_exact_source_parent_and_own_replay(self):
        class Fake:
            def zpath(self,n):return Path('/old')/n
            def zjs(self,n):return {'original':n}
        with patch('reproduction.stage2aa.common.Inputs',Fake):
            for s in SEEDS:
                path,prov=parent('/new',f's{s}-late_only',self.plan())
                self.assertEqual(path,Path(f'/old/s{s}-lm_only/block12-checkpoint/checkpoint.pt'))
                self.assertEqual(prov,{'original':f's{s}-lm_only/provenance.json'})
                self.assertEqual(parent('/new',f'replay-s{s}-b21',self.plan())[0],Path(f'/new/s{s}-late_only/block20-checkpoint/checkpoint.pt'))
    def test_reference_checkpoint_is_original_lm_only(self):
        class Fake:
            def zpath(self,n):return Path('/old')/n
            def zjs(self,n):return {'original':n}
        with patch('reproduction.stage2aa.common.Inputs',Fake):
            p,_=eval_checkpoint('/new','eval-ref-s43-lm_only24',self.plan());self.assertEqual(p,Path('/old/s43-lm_only/block24-checkpoint/checkpoint.pt'))
    def test_no_unregistered_execution(self):
        with self.assertRaises(RuntimeError):require_execution(SimpleNamespace(execute=False,phase='s43-late_only'))
        with self.assertRaises(ValueError):require_execution(SimpleNamespace(execute=True,phase='s45-late_only'))
    def test_all_training_before_replay_and_reference_before_new_eval(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for s in SEEDS:
                with self.assertRaises(FileNotFoundError):require_phase_ready(root,'replay-s43-b21')
                (root/f's{s}-late_only').mkdir();write(root/f's{s}-late_only/result.json',dict(status='SUCCESS'))
            require_phase_ready(root,'replay-s43-b21')
            with self.assertRaises(FileNotFoundError):require_phase_ready(root,'eval-ref-s43-lm_only24',True)
            write(root/'training-verification.json',dict(status='PASS'));require_phase_ready(root,'eval-ref-s43-lm_only24',True)
            with self.assertRaises(FileNotFoundError):require_phase_ready(root,'eval-s43-late_only24',True)
            write(root/'reference-verification.json',dict(status='PASS'));require_phase_ready(root,'eval-s43-late_only24',True)
    def test_dynamic_activation_keeps_shadow_history(self):
        torch.manual_seed(8);c=DynamicController()
        lm=[{n:torch.randn(5).bfloat16().float() for n in TRAINABLE_NAMES} for _ in range(3)]
        aux=[{n:torch.randn(5).bfloat16().float() for n in S9_NAMES} for _ in range(3)];labels=['normal','abnormal','abnormal']
        for _ in range(12):gradients(lm,aux,labels,controller=c,arm='lce_off',frozen=0.)
        parent_state=copy.deepcopy(c.state_dict());off=DynamicController();off.load_state_dict(parent_state)
        go,do=gradients(lm,aux,labels,controller=off,arm='lce_off',frozen=0.)
        gd,dd=gradients(lm,aux,labels,controller=c,arm='dynamic',frozen=0.)
        self.assertEqual(c.state_dict(),off.state_dict());self.assertEqual(dd['budget'],do['budget'])
        self.assertEqual(c.state_dict()['valid_blocks'],13);self.assertGreater(dd['policy']['selected_lambda'],0)
        self.assertNotEqual(state_digest(gd),state_digest(go))
    def test_fresh_optimizer_restore_preserves_24_schedule(self):
        torch.manual_seed(4);initial={n:torch.randn(5).bfloat16() for n in TRAINABLE_NAMES};m,o,s=make_optimizer(initial,'cpu')
        for _ in range(12):update(m,o,s,{n:torch.randn(5)*.01 for n in TRAINABLE_NAMES})
        nm,no,ns=make_optimizer({n:p.detach().bfloat16() for n,p in m.items()},'cpu')
        with torch.no_grad():
            for n in nm:nm[n].copy_(m[n])
        no.load_state_dict(copy.deepcopy(o.state_dict()));ns.load_state_dict(s.state_dict())
        self.assertEqual(no.param_groups[0]['lr'],o.param_groups[0]['lr']);self.assertEqual(ns.state_dict(),s.state_dict())
        self.assertEqual(state_digest(cpu_tree(no.state_dict())),state_digest(cpu_tree(o.state_dict())))
        g={n:torch.randn(5)*.01 for n in TRAINABLE_NAMES}
        update(m,o,s,g);update(nm,no,ns,g);self.assertEqual(state_digest(cpu_tree(m)),state_digest(cpu_tree(nm)))
    def test_step21_prediction_is_read_only(self):
        torch.manual_seed(6);m,o,s=make_optimizer({n:torch.randn(5).bfloat16() for n in TRAINABLE_NAMES},'cpu')
        for _ in range(20):update(m,o,s,{n:torch.randn(5)*.01 for n in TRAINABLE_NAMES})
        g={n:torch.randn(5)*.01 for n in TRAINABLE_NAMES};before=state_digest([cpu_tree(m),cpu_tree(o.state_dict()),s.state_dict()]);p=predict_next(m,o,g)
        self.assertEqual(before,state_digest([cpu_tree(m),cpu_tree(o.state_dict()),s.state_dict()]))
        update(m,o,s,g);self.assertLess(max(float((p[n]-m[n]).abs().max()) for n in TRAINABLE_NAMES),2e-7)
    def test_numpy_start_identity_regression(self):
        x=dict(trainable={'p':torch.ones(2)},runtime={},rng=snapshot_rng_state());self.assertEqual(start_identity(x),start_identity(copy.deepcopy(x)))
        y=copy.deepcopy(x);y['rng']['numpy'][1][0]^=1;self.assertNotEqual(start_identity(x),start_identity(y))
    def test_tie_aware_exact_rank_definition(self):
        rows=[dict(label=l,score=s) for l,s in [('abnormal',.8),('abnormal',.5),('normal',.5),('normal',.2)]]
        expected=dict(auroc=Fraction(7,8),average_precision=Fraction(5,6))
        self.assertEqual(exact_ranks(rows),expected);self.assertEqual(exact_ranking(rows),expected)
        self.assertEqual(exact_ranks(rows[::-1]),expected)
    def test_exact_all_ties_and_perfect_ranks(self):
        for scores,expected in [([.5]*4,dict(auroc=Fraction(1,2),average_precision=Fraction(1,2))),([.9,.8,.3,.2],dict(auroc=Fraction(1),average_precision=Fraction(1)))]:
            rows=[dict(label='abnormal' if i<2 else 'normal',score=x) for i,x in enumerate(scores)]
            self.assertEqual(exact_ranks(rows),expected);self.assertEqual(exact_ranking(rows),expected)
    def test_gain_and_early_higher_are_separate(self):
        d=self.compare(self.ranks());self.assertEqual(d,dict(late_gain='DESCRIPTIVE_LATE_ONLY_GAIN_2_OF_2',early_late_order='EARLY_HIGHER_BOTH_2_OF_2'))
    def test_no_gain_and_mixed_seed_results(self):
        self.assertEqual(self.compare(self.ranks(late=Fraction(1,2)))['late_gain'],'NO_LATE_ONLY_GAIN_2_OF_2')
        r=self.ranks();r['44']=self.ranks(late=Fraction(1,3))['44'];self.assertEqual(self.compare(r)['late_gain'],'MIXED_OR_METRIC_DEPENDENT_LATE_GAIN')
    def test_late_early_exact_tie_not_fake_gain(self):
        d=self.compare(self.ranks(late=Fraction(3,4)));self.assertEqual(d['early_late_order'],'LATE_MATCHES_OR_EXCEEDS_EARLY_2_OF_2')
        self.assertEqual(relations(self.ranks(none=Fraction(2,3))['43'])['late_vs_none']['pattern'],'NO_GAIN_BOTH')
    def test_metric_discordance_retained(self):
        r=self.ranks();r['43']['late']['average_precision']=Fraction(1,3)
        self.assertEqual(self.compare(r)['late_gain'],'MIXED_OR_METRIC_DEPENDENT_LATE_GAIN')
        self.assertEqual(self.compare(r)['early_late_order'],'EARLY_HIGHER_BOTH_2_OF_2')
    def test_missing_seed_or_history_cannot_pass(self):
        r=self.ranks();del r['44']
        with self.assertRaises(AssertionError):verdict(r)
        r=self.ranks();del r['43']['none']
        with self.assertRaises(AssertionError):verdict(r)
    def test_complete_fixed_contrasts_and_interaction(self):
        self.assertEqual(CONTRASTS,INDEPENDENT_CONTRASTS);self.assertEqual(len(CONTRASTS),6)
        value={'none':1,'early':3,'late':4,'full':8}
        c={n:sum(w*value[h] for h,w in weights.items()) for n,weights in CONTRASTS.items()}
        self.assertEqual(c['interaction'],c['late_with_early']-c['late_without_early'])
        self.assertEqual(c['interaction'],c['early_with_late']-c['early_without_late'])
    def test_cumulative_budget_across_reopen(self):
        with tempfile.TemporaryDirectory() as t,patch('reproduction.stage2aa.common.disk_guard'),patch('reproduction.stage2aa.common.signal.signal'),patch('reproduction.stage2aa.common.signal.alarm'):
            for kind,count in LIMITS.items():
                for _ in range(count):Budget(t).reserve(kind,'synthetic')
                with self.assertRaises(RuntimeError):Budget(t).reserve(kind,'overflow')
    def test_storage_floor_and_campaign_ceiling(self):
        with tempfile.TemporaryDirectory() as t:
            write(Path(t)/'budget.json',{})
            with patch('reproduction.stage2aa.common.shutil.disk_usage',return_value=SimpleNamespace(free=40_000_000_000)),patch('reproduction.stage2aa.common.artifact_bytes',return_value=17_600_000_000):
                disk_guard(t,100_000_000)
                with self.assertRaises(RuntimeError):disk_guard(t,500_000_000)
if __name__=='__main__':unittest.main()
