"""Synthetic design/transfer tests; all numerical measurements inherit tested stage2x."""
import copy,unittest
from reproduction.stage2y.common import ANCHORS,BLOCKS,queries,indices_for,variants,LIMITS,SECONDS
from reproduction.stage2y.math import signs,summarize,decision
from reproduction.stage2y.verify import verdict,averages

class Contracts(unittest.TestCase):
    def reports(self,a=-.1,b=.1,eq=False,clip=True):
        cell=dict(means=dict(log_support=dict(total=a),top11=dict(total=b)),policy_raw_equal=eq,clip_raw_equal=clip)
        return {n:dict(queries={str(k):copy.deepcopy(cell) for k in BLOCKS}) for n in ANCHORS}
    def test_full_matrix_coverage(self):
        cells=[(n,b) for n,c in ANCHORS.items() for b in queries(c['step'])]
        self.assertEqual(len(set(cells)),40);self.assertEqual(sum(b!=ANCHORS[n]['step'] for n,b in cells),30)
    def test_diagonal_first(self):
        for c in ANCHORS.values():
            q=queries(c['step']);self.assertEqual(q[0],c['step']);self.assertEqual(q[1:],sorted(set(BLOCKS)-{c['step']}))
    def test_twelve_unique_exposures(self):
        ids=[i for b in BLOCKS for i in indices_for(b)];self.assertEqual(len(set(ids)),12)
        self.assertEqual(ids,[0,1,2,24,25,26,36,37,38,60,61,62])
    def test_forwards_and_aliases(self):
        total=0;diagonal=0
        for n in ANCHORS:
            h=('a','b','c','d') if 'parent12' in n else ('a','a','b','b')
            reps,_=variants(dict(corners={k:dict(bf16_sha256=v) for k,v in zip(('00','01','10','11'),h)}))
            total+=12*(1+len(reps));diagonal+=3*(1+len(reps))
        self.assertEqual((total,diagonal,total-diagonal),(408,102,306))
    def test_budget_bounded(self):
        self.assertEqual(LIMITS,dict(process=14,forward=528));self.assertEqual(SECONDS,5400)
    def test_all_transfer(self):
        r=self.reports();self.assertEqual(decision(r),verdict(r));self.assertEqual(decision(r)['fully_transferring_anchors'],10)
    def test_state_specific_transfer(self):
        r=self.reports();r['s44-initial']=self.reports(.2,-.1)['s44-initial']
        self.assertEqual(decision(r),verdict(r));self.assertEqual(decision(r)['transfer_response'],'ALL_ANCHOR_DIRECTIONS_TRANSFER')
    def test_one_input_flip(self):
        r=self.reports();r['s43-initial']['queries']['9']['means']['log_support']['total']=.1
        d=decision(r);self.assertEqual(d,verdict(r));self.assertEqual(d['fully_transferring_anchors'],9);self.assertEqual(d['matching_off_batch_cells'],29)
        self.assertEqual(d['transfer_response'],'INPUT_DEPENDENT_OR_MIXED_TRANSFER')
    def test_diagonal_not_counted(self):
        r=self.reports();r['s43-initial']['queries']['1']['means']['top11']['total']=-.1
        self.assertEqual(decision(r)['matching_off_batch_cells'],27)
    def test_zero_not_dropped(self):
        r=self.reports();r['s43-initial']['queries']['9']['means']['log_support']['total']=0.
        self.assertEqual(decision(r),verdict(r));self.assertEqual(decision(r)['matching_off_batch_cells'],29)
    def test_no_epsilon(self):
        self.assertEqual(signs(dict(log_support=dict(total=-1e-30),top11=dict(total=1e-30))),(-1,1))
    def test_null_precedence(self):
        r=self.reports(0.,0.,True);self.assertEqual(decision(r),verdict(r));self.assertEqual(decision(r)['transfer_response'],'NO_RESOLVED_OFF_BATCH_POLICY_RESPONSE')
    def test_diagonal_clip_not_offbatch(self):
        r=self.reports();r['s43-initial']['queries']['1']['clip_raw_equal']=False
        self.assertEqual(decision(r)['clipping_response'],'NO_RESOLVED_OFF_BATCH_CLIP_RESPONSE')
    def test_offbatch_clip(self):
        r=self.reports();r['s43-initial']['queries']['9']['clip_raw_equal']=False
        self.assertEqual(decision(r),verdict(r));self.assertEqual(decision(r)['clipping_response'],'OFF_BATCH_CLIP_MEDIATED_ATTENTION_RESPONSE')
    def test_missing_cell_rejected(self):
        r=self.reports();del r['s43-initial']['queries']['9']
        for fn in (decision,verdict):
            with self.assertRaises(AssertionError):fn(r)
    def test_missing_anchor_rejected(self):
        r=self.reports();del r['s43-initial']
        for fn in (decision,verdict):
            with self.assertRaises(AssertionError):fn(r)
    def test_offbatch_mean_all_nine(self):
        rows=[dict(x=dict(total=float(i))) for i in range(9)]
        self.assertEqual(summarize(rows)['x']['total'],4.)
        self.assertEqual(summarize(rows),averages([dict(effects=r) for r in rows]))
    def test_no_four_batch_pooled_mean(self):
        with self.assertRaises(AssertionError):summarize([dict(x=dict(total=1.))]*12)
    def test_query_validated(self):
        for fn in (queries,indices_for):
            with self.assertRaises(AssertionError):fn(2)
if __name__=='__main__':unittest.main()
