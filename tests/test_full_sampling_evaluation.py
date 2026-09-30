"""Numerical edge cases and continuity checks, without performance selection."""
import ast
import importlib.util
from html.parser import HTMLParser
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


class FullSamplingEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core=module('sampling_metrics_fixture','spectral_utils/full_sampling_evaluation.py')
        cls.ref=module('anchor_metrics_fixture','scripts/evaluate_localization_full_anchors_v3.py')
        cls.ref.DRAWS=40

    def test_all_previous_contrasts_preserved(self):
        core=self.core;keys={(d['left'],d['right'],d['scope']) for d in core.contrasts()}
        self.assertEqual(len(keys),142)
        for file,selectors in [('run_fusion_sampling_replication_v1.py',core.SELECTORS[:6]),
                               ('run_fusion_entropy_sampling_v1.py',core.SELECTORS[-2:])]:
            tree=ast.parse((ROOT/'scripts'/file).read_text())
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pairs')
            namespace=dict(SELECTORS=selectors,CORES=core.CORES,
                arm_name=lambda s,c:f'sample_{s}__{c}',name=lambda s,c:f'sample_{s}__{c}',
                key=lambda d:(d['left'],d['right'],d['scope']))
            exec(compile(ast.Module(body=[fn],type_ignores=[]),file,'exec'),namespace)
            previous=namespace['pairs']()
            old={(d['left'],d['right'],d['scope']) if isinstance(d,dict) else (*d,'all') for d in previous}
            self.assertTrue(old<=keys)

    def test_grouped_intervals_match_explicit_resampling(self):
        core,ref=self.core,self.ref
        arms=[f'sample_{s}__{c}' for s in core.SELECTORS for c in core.CORES]+['dual__equal_graph_perm']
        records=[];folds=[]
        for f in range(5):
            for cell in ('prm_fixture','pb_fixture_q4','pb_fixture_q8'):
                for i in range(4):
                    records.append(dict(cell=cell,group_id=f'g{f}_{i}',steps=2));folds.append(f)
        n,k=len(records),len(arms);folds=np.array(folds)
        a=dict(offsets=np.arange(n+1)*2,labels=np.tile([0,1],n),target=np.tile([-1,-1,0,1],n//4),
               scores=np.zeros((2*n,k)),valid=np.ones((n,k),bool),decision=np.ones((n,k),bool),
               predictions=np.repeat(np.tile([-1,-1,0,0],n//4)[:,None],k,axis=1),
               within=np.full((n,k),.5))
        prm=np.array([r['cell'].startswith('prm') for r in records]);a['target'][prm]=-2
        left='sample_entropy_tails__graph010';right='sample_entropy_tails__joint0'
        j,h=arms.index(left),arms.index(right)
        a['scores'][:,j]=np.tile([0.,1.],n);a['within'][:,j]=1.
        a['predictions'][~prm,j]=a['target'][~prm]
        # One invalid PRMB row must be dropped on BOTH sides of a paired AUC.
        a['valid'][0,j]=False
        # An invalid clean PB decision must remain a failure in its denominator.
        bad=next(i for i,r in enumerate(records) if r['cell']=='pb_fixture_q8')
        a['decision'][bad,j]=False
        eligible=np.ones(n,bool);native=np.ones((n,k),bool)
        result=core.intervals(ref,records,arms,a,folds,eligible,native)
        pair=next(d for d in result['paired'] if d['left']==left and d['right']==right and d['scope']=='all')
        self.assertEqual(pair['prm_common_answers'],19)
        self.assertEqual(pair['delta']['prm_fold'],.5)
        self.assertEqual(pair['delta']['within'],.5)
        self.assertEqual(pair['intervals']['within']['ci95'],[.5,.5])
        groups=sorted({r['group_id'] for r in records});lookup={g:i for i,g in enumerate(groups)}
        weights=np.random.default_rng(ref.SEED).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=ref.DRAWS)
        draws=[]
        for w in weights:
            repeated=[i for i,r in enumerate(records) if r['cell']=='pb_fixture_q8' for _ in range(w[lookup[r['group_id']]])]
            clean=[i for i in repeated if a['target'][i]==-1];error=[i for i in repeated if a['target'][i]>=0]
            if not clean or not error:continue
            def score(column):
                success=lambda indices:sum(bool(a['decision'][i,column]) and a['predictions'][i,column]==a['target'][i] for i in indices)/len(indices)
                c,e=success(clean),success(error)
                return 2*c*e/(c+e) if c+e else 0.
            draws.append(score(j)-score(h))
        np.testing.assert_allclose(pair['intervals']['pb_q8']['ci95'],np.quantile(draws,[.025,.975]),atol=1e-14)
        point=core.fold_points(ref,records,a,j,folds)
        self.assertEqual(point['fold_mean_auc'],1.)

    def test_no_eligible_population_is_not_zero_performance(self):
        # Empty AUC inputs are undefined, including a missing fixed fold.
        ref=self.ref
        records=[dict(cell='prm_fixture',group_id='g0',steps=2)]
        a=dict(offsets=np.array([0,2]),labels=np.array([0,1]),scores=np.array([[0.],[1.]]),valid=np.ones((1,1),bool))
        result=self.core.fold_points(ref,records,a,0,np.array([0]))
        self.assertIsNone(result['fold_mean_auc'])
        self.assertEqual(result['fold_aucs'],[1.,None,None,None,None])

    def test_render_preserves_method_values(self):
        renderer=module('sampling_report_fixture','scripts/evaluate_full_sampling_v3.py')
        renderer.OUT=ROOT/'scratch/full_sampling_report_fixture'
        renderer.OUT.mkdir(parents=True,exist_ok=True)
        arms=['fixture_A','fixture_B'];metrics={}
        for i,arm in enumerate(arms):
            metrics[arm]=dict(access='answer only; no labels',
                prm=dict(fold_mean_auc=.61+i*.01,auroc=.62+i*.01,within_answer_auc=.63+i*.01,answers=6900+i),
                pb=dict(macros=dict(q4=.21+i*.01,q8=.22+i*.01,all=.215+i*.01),
                        cells={'fixture':dict(valid_decisions=6790+i)}))
        renderer.render(self.ref,arms,metrics,dict(paired=[]),{},[])
        class Table(HTMLParser):
            def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None
            def handle_starttag(self,tag,attrs):
                if tag=='tr':self.row=[]
                if tag=='td':self.cell=''
            def handle_data(self,text):
                if self.cell is not None:self.cell+=text
            def handle_endtag(self,tag):
                if tag=='td':self.row.append(self.cell);self.cell=None
                if tag=='tr' and self.row:self.rows.append(self.row)
        parser=Table();parser.feed((renderer.OUT/'REPORT.html').read_text(encoding='utf-8'))
        self.assertEqual(parser.rows[0],['fixture_A','answer only; no labels','0.6100','0.6200','0.6300','6900/6969','21.00%','22.00%','21.50%','6790/6800'])
        self.assertEqual(parser.rows[1][0],'fixture_B')
        self.assertEqual(parser.rows[1][7],'23.00%')


if __name__=='__main__':unittest.main()
