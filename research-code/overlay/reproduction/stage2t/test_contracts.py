"""CPU checks for meaningful numerical, resume and exclusion boundaries."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from reproduction.stage2i.fingerprint import canonical_json_sha256
from reproduction.stage2l.controller import DynamicController,compute_block_gradients
from reproduction.stage2k.producer_math import compute_lce
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2m.metrics import summarize_metrics
from reproduction.stage2m.train_scout import matched_configuration
from reproduction.stage2p import CONFIG_SHA256
from reproduction.stage2q.run import make_optimizer,update
from reproduction.stage2q.execution import cpu_tree
from reproduction.stage2s.trajectory import predict_next
from reproduction.stage2s.common import state_digest,write
from reproduction.stage2t.common import Budget,LIMITS,save_raw,load_raw,validate_range
from reproduction.stage2t.evaluate import lm_terms
from reproduction.stage2t.verify import aggregation,spatial,adam,metrics,phenotype,paired

class Contracts(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(2)
    def test_data_registry_binding_precedes_import(self):
        import os
        from types import SimpleNamespace
        from reproduction.stage2t.evaluate import bind_data_registry
        def imported(name):
            self.assertEqual(name,'qwenvl.data')
            self.assertEqual(os.environ['MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION'],'fixed-eval.json')
            return SimpleNamespace(MEDIC_AD_STAGE2H_CALIBRATION={'annotation_path':'fixed-eval.json','data_path':'images'})
        with patch('reproduction.stage2t.evaluate.importlib.import_module',side_effect=imported):
            self.assertEqual(bind_data_registry('fixed-eval.json','images')['annotation_path'],'fixed-eval.json')
    def test_stale_registry_fails_before_model_loading(self):
        from types import SimpleNamespace
        from reproduction.stage2t.evaluate import bind_data_registry
        stale=SimpleNamespace(MEDIC_AD_STAGE2H_CALIBRATION={'annotation_path':'old-default.json','data_path':'images'})
        with patch('reproduction.stage2t.evaluate.importlib.import_module',return_value=stale):
            with self.assertRaisesRegex(RuntimeError,'fresh process'):bind_data_registry('fixed-eval.json','images')
    def test_configuration_unchanged(self):self.assertEqual(canonical_json_sha256(matched_configuration()),CONFIG_SHA256)
    def test_ranges_fail_closed(self):
        validate_range('train',0,24,None,'R0');validate_range('train',12,24,'own','R1')
        validate_range('replay',23,24,'own','R0')
        for args in [('replay',11,12,'ck','R0'),('train',12,24,None,'R1'),('train',0,24,'ck','R0'),('train',0,25,None,'R0'),('train',0,24,None,'unknown')]:
            with self.assertRaises(ValueError):validate_range(*args)
    def test_budget_reservation_hard_limits(self):
        with tempfile.TemporaryDirectory() as t,patch('reproduction.stage2t.common.signal.signal'),patch('reproduction.stage2t.common.signal.alarm'),patch('reproduction.stage2t.common.disk_guard'):
            b=Budget(t)
            for kind,limit in LIMITS.items():
                for i in range(limit):b.reserve(kind,str(i))
                with self.assertRaises(RuntimeError):b.reserve(kind,'overflow')
            self.assertEqual(json.loads(b.path.read_text())['counts'],LIMITS)
            with self.assertRaises(ValueError):b.reserve('typo','x')
    def test_lossless_archive_and_exclusive_write(self):
        with tempfile.TemporaryDirectory() as t,patch('reproduction.stage2t.common.disk_guard'):
            p=Path(t)/'raw.pt.gz';d={'bf16':torch.randn(12).bfloat16(),'fp32':torch.randn(12)}
            save_raw(p,d);self.assertEqual(state_digest(d),state_digest(load_raw(p)))
            with self.assertRaises(FileExistsError):save_raw(p,d)
    def test_lm_shift_and_ignored_targets(self):
        logits=torch.randn(1,7,13);labels=torch.tensor([[-100,3,5,-100,7,2,-100]])
        terms=lm_terms(logits,labels);expected=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,13),labels[:,1:].reshape(-1))
        self.assertEqual(terms['valid_tokens'],4)
        self.assertAlmostEqual(float((terms['logsumexp']-terms['target_logit']).mean()),float(expected),places=6)
    def test_raw_spatial_formula(self):
        for label in ('normal','abnormal'):
            maps={k:torch.randn(1,4,1024).bfloat16() for k in ('abnormal_raw','normal_raw')}
            x=spatial(maps,label);y=compute_lce(maps['abnormal_raw'],maps['normal_raw'],[label])
            for k in x:self.assertAlmostEqual(x[k],float(y['per_image_loss' if k=='lce_loss' else k][0]),places=5)
    def test_metrics_ties(self):
        rows=[dict(label=l,score=s,lm_loss=1.,valid_tokens=4,effective_support=256,top11_mass=.1) for l,s in [('normal',.5),('abnormal',.5),('abnormal',.8),('normal',.1)]]
        independent=metrics(rows);producer=summarize_metrics(rows)
        self.assertEqual(independent['auroc'],.875);self.assertAlmostEqual(independent['average_precision'],5/6)
        for k in ('auroc','average_precision'):self.assertEqual(independent[k],producer[k])
    def test_phenotype_strict_boundary(self):
        e=dict(support_median=127,support_below128=16,top11_above035=16)
        self.assertTrue(phenotype(e,[True,True,False,False])['concentrated'])
        self.assertFalse(phenotype(dict(e,support_median=128),[True]*4)['concentrated'])
        self.assertFalse(phenotype(e,[True,False,False,False])['concentrated'])
    def test_pair_detects_single_vpt_tensor_drift(self):
        from reproduction.stage2s.verify_single import VPT
        before={n:torch.zeros(3) for n in TRAINABLE_NAMES}
        d={'lm':[{n:torch.ones(3).bfloat16() for n in TRAINABLE_NAMES}]*3,
           'lce':[{n:torch.ones(3).bfloat16() for n in S9_NAMES}]*3,
           'attention_maps':[], 'applied':{n:torch.ones(3) for n in TRAINABLE_NAMES},
           'updated_master':{n:torch.ones(3)*.01 for n in TRAINABLE_NAMES}}
        keys=('inputs_sha256','rng','controller_before','controller','gradient','runtime_before','runtime_after','optimizer_before','optimizer_devices','aggregate_lm','aggregate_lce','master_before','master_after','images','optimizer_after')
        r=dict.fromkeys(keys,'identical');e=copy.deepcopy(d)
        self.assertTrue(paired(d,e,r,r,before,before)['exact'])
        e['updated_master'][VPT[0]][0]+=1e-7
        self.assertFalse(paired(d,e,r,r,before,before)['exact'])
    def test_runtime_modes_and_cache_restore(self):
        from reproduction.stage2s.common import runtime_state,restore_runtime
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.5))
        model[1].eval();model.rope_deltas=torch.ones(1);state=runtime_state(model)
        model.train();model.rope_deltas=torch.zeros(1);model.last_visual_tokens=torch.ones(3)
        restore_runtime(model,state)
        self.assertEqual(state_digest(runtime_state(model)),state_digest(state))
        self.assertFalse(model[1].training);self.assertFalse(hasattr(model,'last_visual_tokens'))
    def test_24step_adam_and_fresh_state_prediction(self):
        initial={n:torch.randn(7).bfloat16() for n in TRAINABLE_NAMES}
        masters,opt,sched=make_optimizer(initial,'cpu');moments={}
        for step in range(1,25):
            applied={n:torch.randn(7)*.01 for n in TRAINABLE_NAMES};before=cpu_tree(masters);lr=opt.param_groups[0]['lr']
            if step in (13,24):
                other,oo,ss=make_optimizer(initial,'cpu')
                for n in other:other[n].data.copy_(masters[n])
                oo.load_state_dict(copy.deepcopy(opt.state_dict()));ss.load_state_dict(sched.state_dict())
                prediction=predict_next(other,oo,applied);unchanged=state_digest(cpu_tree(oo.state_dict()))
                self.assertEqual(unchanged,state_digest(cpu_tree(opt.state_dict())))
            update(masters,opt,sched,applied)
            if step in (13,24):self.assertEqual(state_digest(prediction),state_digest(cpu_tree(masters)))
            error=adam({'applied':applied,'updated_master':cpu_tree(masters)},dict(block=step,update={'learning_rate_used':lr}),before,moments)
            self.assertLessEqual(error,2e-7)
    def test_independent_class_controller_reconstruction(self):
        lm=[{n:torch.randn(7).bfloat16().float() for n in TRAINABLE_NAMES} for _ in range(3)]
        lce=[{n:torch.randn(7).bfloat16().float() for n in S9_NAMES} for _ in range(3)]
        labels=['normal','abnormal','abnormal'];controller=DynamicController();before=controller.state_dict()
        applied,d=compute_block_gradients(lm,lce,labels,trainable_names=TRAINABLE_NAMES,s9_names=S9_NAMES,controller=controller)
        maps=[{k:torch.randn(1,4,1024).bfloat16() for k in ('abnormal_raw','normal_raw')} for _ in range(3)]
        images=[dict(label=l,**spatial(m,l)) for m,l in zip(maps,labels)]
        raw={'lm':[{n:v.bfloat16() for n,v in g.items()} for g in lm],'lce':[{n:v.bfloat16() for n,v in g.items()} for g in lce],'applied':applied,'attention_maps':maps}
        r=dict(images=images,block=1,controller_before=before,controller=d['budget'],gradient=d['gradient'],aggregate_lm=state_digest(d['aggregate_lm']),aggregate_lce=state_digest(d['aggregate_lce']))
        self.assertTrue(aggregation(raw,r)['aggregation_and_applied_exact'])
        raw['applied'][TRAINABLE_NAMES[0]][0]+=1e-3
        with self.assertRaises(AssertionError):aggregation(raw,r)

if __name__=='__main__':unittest.main()
