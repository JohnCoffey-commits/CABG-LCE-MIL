import unittest
from reproduction.stage2af.metrics import parse_answer,summary,paired

class MetricTests(unittest.TestCase):
    def rows(self):
        return [dict(sample_id=str(i),pixel_sha256_prepared=str(i),target=y,parsed=p,score=s) for i,(y,p,s) in enumerate([(0,0,.5),(0,None,0.),(1,1,.5),(1,0,.5)])]
    def test_whole_response_and_invalid_denominator(self):
        self.assertEqual(parse_answer(' Yes! '),1);self.assertEqual(parse_answer('No.'),0)
        for s in ['Yes, but','probably no','yes no','','No?']:self.assertIsNone(parse_answer(s))
        r=summary(self.rows());self.assertEqual(r['invalid'],1);self.assertEqual(r['accuracy'],.5);self.assertEqual(r['balanced_accuracy'],.5)
    def test_score_ties(self):
        r=summary(self.rows());self.assertAlmostEqual(r['auroc'],.75);self.assertAlmostEqual(r['average_precision'],2/3)
    def test_paired_matching(self):
        r=paired(self.rows(),self.rows());self.assertEqual(r['delta_balanced_accuracy'],0)
        rows=self.rows();rows[0]['pixel_sha256_prepared']='changed'
        with self.assertRaises(AssertionError):paired(self.rows(),rows)
if __name__=='__main__':unittest.main()
