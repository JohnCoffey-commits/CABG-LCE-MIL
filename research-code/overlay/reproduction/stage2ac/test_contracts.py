import unittest
from reproduction.stage2ac.contracts import parse_answer, answer_metrics, selected_development, unique_training, require_time, QUESTION


class Contracts(unittest.TestCase):
    def test_parser_does_not_extract_agreeable_word(self):
        for text in ('Yes, but no.', 'Probably yes', '', 'Yes\nNo', 'No tumor', 'yes..'):
            self.assertIsNone(parse_answer(text))
        for text,value in [(' YES. ',1),('no',0),('No!',0)]:self.assertEqual(parse_answer(text),value)

    def test_invalid_remains_in_denominator(self):
        r=answer_metrics([{'parsed':1,'target':1},{'parsed':None,'target':0}])
        self.assertEqual(r,dict(n=2,correct=1,invalid=1,accuracy=.5))

    def test_selection_uses_original_rank_not_scores(self):
        rows=[dict(sha256=str(i),scout_eval_rank=i,scout_label='normal' if i%2 else 'abnormal') for i in range(32)]
        self.assertEqual([r['scout_eval_rank'] for r in selected_development(list(reversed(rows)))],[0,1,2,3])

    def test_duplicate_cases_rejected(self):
        rows=[dict(sha256='same',scout_eval_rank=i,scout_label='normal') for i in range(32)]
        with self.assertRaises(ValueError):selected_development(rows)

    def test_repeated_exposure_is_not_new_training_case(self):
        row=dict(sha256='h',sample_id='x',scout_label='normal')
        self.assertEqual(len(unique_training([row,row])),1)
        with self.assertRaises(ValueError):unique_training([row,{**row,'scout_label':'abnormal'}])

    def test_closure_reserve(self):
        require_time({'deadline_epoch':36000},30599)
        with self.assertRaises(RuntimeError):require_time({'deadline_epoch':36000},30600)

    def test_prompt_is_original_literal(self):
        from pathlib import Path
        import ast
        module=ast.parse((Path(__file__).resolve().parents[2]/'run_anomaly.py').read_text())
        value=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='QUESTION' for t in n.targets))
        self.assertEqual(QUESTION,value)


if __name__=='__main__':unittest.main()
