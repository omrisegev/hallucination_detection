import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cohort',ROOT/'spectral_utils/fusion_replication_cohort.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class CohortIsolation(unittest.TestCase):
    def test_excludes_source_family_across_cells_and_does_not_fill_shortages(self):
        rows=[{'row_id':rid,'group_id':group,'tokens':length,'steps':2} for rid,group,length in (
            ('old_variant','excluded',80),('a1','a',80),('a2','a',90),('b','b',100),('long','a',300),('short','s',63))]
        release={'cells':{'first':{'rows':rows},'second':{'rows':[{'row_id':'other_answer','group_id':'b','tokens':80,'steps':2}]}}}
        selected,support=c.select_cohort(release,{'excluded'},'fixed',quota=3,cells=('first','second'),bins=((64,255),(256,1023)))
        self.assertEqual({r['group_id'] for r in selected},{'a','b'})
        self.assertEqual(len(selected),2);self.assertEqual([x['selected'] for x in support],[2,0,0,0])

    def test_selection_is_order_invariant_and_label_blind(self):
        rows=[{'row_id':str(i),'group_id':'g'+str(i//2),'tokens':100,'steps':3} for i in range(12)]
        original={'cells':{'x':{'rows':rows}}}
        changed={'cells':{'x':{'rows':[{**r,'target':1,'fit_success':False} for r in reversed(rows)]}}}
        a,_=c.select_cohort(original,set(),'same',quota=3,cells=('x',),bins=((64,255),))
        b,_=c.select_cohort(changed,set(),'same',quota=3,cells=('x',),bins=((64,255),))
        self.assertEqual([r['row_id'] for r in a],[r['row_id'] for r in b])


if __name__=='__main__':unittest.main()
