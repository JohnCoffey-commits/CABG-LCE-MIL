"""Failure-oriented tests: ties, class errors, leakage and altered evidence."""
import copy
import unittest
from reproduction.stage2ab.readiness import partition, ranking, unpack


class ReadinessTests(unittest.TestCase):
    def row(self, key):
        return {'sample_id':key,'relative_path':key+'.jpg','sha256':key}

    def data(self):
        return {'pool':[self.row(x) for x in 'abcd'], 'd3r':[self.row('a')],
                'pilot':[self.row('b')], 'train':[self.row('c'),self.row('c')],
                'development':[self.row('d')]}

    def test_repeated_training_exposures_are_not_new_cases(self):
        r=partition(self.data())
        self.assertTrue(all(x['train']==1 and x['unused']==0 for x in r.values()))

    def test_reusing_development_for_training_is_rejected(self):
        d=self.data();d['train'].append(self.row('d'))
        with self.assertRaises(ValueError):partition(d)

    def test_different_id_same_content_still_rejected(self):
        d=self.data();d['development'][0]['sha256']='c'
        with self.assertRaises(ValueError):partition(d)

    def test_exclusion_overlap_rejected(self):
        d=self.data();d['pilot']=[self.row('a')]
        with self.assertRaises(ValueError):partition(d)

    def test_unknown_source_member_rejected(self):
        d=self.data();d['development']=[self.row('e')]
        with self.assertRaises(ValueError):partition(d)

    def test_unused_member_remains_visible(self):
        d=self.data();d['pool'].append(self.row('e'))
        self.assertEqual(partition(d)['sha256']['unused'],1)

    def test_altered_body_cannot_be_accepted(self):
        s={'files':{'pool':{'text':'[]','sha256':'bad','bytes':2,'path':'pool.json'}}}
        with self.assertRaisesRegex(ValueError,'identity mismatch'):unpack(s)

    def test_ties_have_half_auc_and_prevalence_ap(self):
        rows=[{'label':x,'score':1} for x in ['abnormal','normal','normal']]
        self.assertEqual(ranking(rows),{'auroc':.5,'average_precision':1/3})

    def test_perfect_and_reversed_order(self):
        rows=[{'label':'abnormal','score':1},{'label':'normal','score':0}]
        self.assertEqual(ranking(rows),{'auroc':1.,'average_precision':1.})
        rows[0]['score']=-1
        self.assertEqual(ranking(rows),{'auroc':0.,'average_precision':.5})

    def test_score_shift_does_not_create_rank_gain(self):
        rows=[{'label':l,'score':s} for l,s in [('normal',0),('abnormal',2),('normal',2),('abnormal',1)]]
        shifted=copy.deepcopy(rows)
        for r in shifted:r['score']+=100
        self.assertEqual(ranking(rows),ranking(shifted))

    def test_single_class_rejected(self):
        with self.assertRaises(ValueError):ranking([{'label':'normal','score':1}])

    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):ranking([{'label':'unknown','score':1}])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):ranking([{'label':'normal','score':1},{'label':'abnormal','score':float('nan')}])


if __name__=='__main__':unittest.main()
