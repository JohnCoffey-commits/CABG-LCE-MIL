import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from reproduction.stage2ad.readiness import GATES, gate_decision, precision, source_facts, sha
from reproduction.stage2ad.verify import verify

EVIDENCE=Path(__file__).resolve().parents[3]/'document/cabg_mil_v1_2/external-final-readiness'


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        (self.root/'source').write_text('Synthetic source for gate-control testing, not medical evidence.')
        self.c={'gates':{k:dict(status='PASS',reason='synthetic test judgment',
                    sources=[dict(path='source',sha256=sha(self.root/'source'))]) for k in GATES}}

    def tearDown(self): self.temp.cleanup()

    def test_all_pass_still_requires_execution_lock(self):
        self.assertEqual(gate_decision(self.c,self.root)[0],'REVIEWED_GATES_PASS_REQUIRES_EXECUTION_LOCK')

    def test_every_unknown_or_failed_gate_blocks(self):
        for gate in GATES:
            for status in ('UNKNOWN','FAIL'):
                c=copy.deepcopy(self.c);c['gates'][gate]['status']=status
                with self.subTest(gate=gate,status=status):
                    self.assertEqual(gate_decision(c,self.root),('NOT_ADMITTED',[gate]))

    def test_missing_extra_or_invalid_gate_rejected(self):
        for mutation in ('missing','extra','invalid','no_source','blank_reason'):
            c=copy.deepcopy(self.c)
            if mutation=='missing':del c['gates']['grouping']
            if mutation=='extra':c['gates']['magic']='PASS'
            if mutation=='invalid':c['gates']['rights']['status']='true'
            if mutation=='no_source':c['gates']['rights']['sources']=[]
            if mutation=='blank_reason':c['gates']['rights']['reason']=' '
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):gate_decision(c,self.root)

    def test_changed_source_rejected(self):
        (self.root/'source').write_text('Changed evidence')
        with self.assertRaises(ValueError):gate_decision(self.c,self.root)

    def test_escaping_source_rejected(self):
        self.c['gates']['rights']['sources'][0]['path']='../not-in-evidence'
        with self.assertRaises(ValueError):gate_decision(self.c,self.root)

    def test_precision_rejects_pseudo_and_invalid_n(self):
        for a,b in [(0,30),(-1,30),(25.0,11),(True,11)]:
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):precision(a,b,.2)
        for q in (-.1,1.1,float('nan'),float('inf')):
            with self.subTest(q=q),self.assertRaises(ValueError):precision(25,11,q)

    def test_class_imbalance_and_zero_width_guard(self):
        self.assertGreater(precision(25,11,.2)['approximate_half_width'],precision(18,18,.2)['approximate_half_width'])
        self.assertGreater(precision(25,11,0)['zero_discordance_simultaneous_absolute_bound'],.20)
        self.assertFalse(precision(25,11,.2)['observed'])


@unittest.skipUnless(EVIDENCE.exists(),'Actual source evidence required for integration/mutation tests')
class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for name in ('source-metadata-v1','inherited-source-evidence-v1'):
            shutil.copytree(EVIDENCE/name,self.root/name)
        for name in ('readiness-report-v1.json','admission-register-v1.json'):
            shutil.copy2(EVIDENCE/name,self.root/name)

    def tearDown(self):self.temp.cleanup()

    def test_real_independent_verification(self):
        self.assertEqual(verify(self.root)['status'],'PASS_INDEPENDENT_METADATA_AND_PRECISION_CHECK')

    def test_outcome_or_precision_tampering_rejected(self):
        path=self.root/'readiness-report-v1.json'; original=json.loads(path.read_text())
        for change in ('zero_effect','fake_gpu','narrow_ci','promote_admission'):
            x=copy.deepcopy(original)
            if change=='zero_effect':x['independent_effect']=0
            if change=='fake_gpu':x['model_loads']=1
            if change=='narrow_ci':x['precision_design_scenarios']['ds001226_all_cases_upper_ceiling'][0]['approximate_half_width']=.001
            if change=='promote_admission':x['candidates']['ds001226_and_postop']['status']='ADMITTED'
            path.write_text(json.dumps(x))
            with self.subTest(change=change),self.assertRaises(AssertionError):verify(self.root)

    def test_source_tampering_rejected(self):
        with (self.root/'source-metadata-v1/ds001226-participants.tsv').open('a') as f:f.write('\nforged')
        with self.assertRaises(AssertionError):verify(self.root)

    def test_missing_mask_or_truncated_tree_rejected(self):
        path=self.root/'source-metadata-v1/ds001226-tree.json';original=json.loads(path.read_text())
        for change in ('mask','truncated'):
            x=copy.deepcopy(original)
            if change=='truncated':x['truncated']=True
            else:x['tree']=[e for e in x['tree'] if not e['path'].endswith('sub-PAT01_space_T1_label-tumor.nii')]
            path.write_text(json.dumps(x))
            with self.subTest(change=change),self.assertRaises(ValueError):source_facts(self.root/'source-metadata-v1')


if __name__=='__main__':unittest.main()
