"""CPU lifecycle and telemetry tests with a tiny fixture adapter; no HF weights."""
import copy
import importlib.util
import tempfile
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from spectral_utils.prm_adapters import PRMAdapter
from spectral_utils import save_cache_atomic, load_cache

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prm_driver',ROOT/'cluster/run_prmbench_prm_telemetry.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


class FixtureAdapter(PRMAdapter):
    def __init__(self, context=10):
        self.backbone=torch.nn.Linear(8,8)
        self.model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=context))
        self.bind_head(torch.randn(32,8),'own_used')
        self.head_meta.update(adapter='rlhflow_prm',official_wrapper_commit='fixture')
        self.forward_count=0
    def load(self, model_id): return self.model,None,self.head_meta
    def build(self,tok,question,steps):
        n=12 if question=='long' else 4
        return dict(input_ids=list(range(1,n+1)),score_positions=[n-1],
                    step_token_positions=[list(range(1,n-1))],n_step_text_mismatch=0,n_boundary_whitespace=0)
    def forward(self,x,batch):
        self.forward_count+=1
        return torch.arange(x.shape[1]*8,dtype=torch.float32).reshape(1,-1,8)/100,{}
    def rewards(self,outputs,batch): return [.8]


def row(idx,question='short'):
    return dict(idx=idx,source_idx=idx,classification='confidence',category='soundness',
                question=question,steps=['fixture'],error_steps=[])


class Driver(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123);D.STOP['flag']=False
        self.temp=tempfile.TemporaryDirectory();self.path=str(Path(self.temp.name)/'telemetry.pkl')
        self.cfg=SimpleNamespace(model='fixture',adapter='rlhflow_prm',verify_rows=0,logprob_top_k=15,checkpoint_every=1)
    def tearDown(self):self.temp.cleanup();D.STOP['flag']=False
    def test_context_failure_is_counted_and_never_forwarded(self):
        adapter=FixtureAdapter()
        with patch.object(D,'get_adapter',return_value=adapter):
            complete,_,_=D.run_prm(self.cfg,[row('ok'),row('bad','long')],self.path)
        self.assertTrue(complete);self.assertEqual(adapter.forward_count,1)
        cache=load_cache(self.path)
        self.assertEqual(cache[1]['failure'],'context_overflow')
        self.assertEqual(cache[1]['n_input_tokens'],12)
        self.assertEqual(cache[0]['native_labels'],[1])
    def test_sigterm_checkpoints_and_resumes(self):
        adapter=FixtureAdapter();D._on_sigterm(None,None)
        with patch.object(D,'get_adapter',return_value=adapter):
            ok,_,_=D.run_prm(self.cfg,[row('a')],self.path)
        self.assertFalse(ok);self.assertEqual(adapter.forward_count,0)
        self.assertEqual(load_cache(self.path),{})
        D.STOP['flag']=False
        with patch.object(D,'get_adapter',return_value=adapter):
            ok,_,_=D.run_prm(self.cfg,[row('a')],self.path)
            ok2,_,_=D.run_prm(self.cfg,[row('a')],self.path)
        self.assertTrue(ok and ok2);self.assertEqual(adapter.forward_count,1)
    def test_main_writes_incomplete_manifest_and_exits_85(self):
        source = str(Path(self.temp.name)/'meta.pkl'); save_cache_atomic({0:row('a')}, source)
        output = str(Path(self.temp.name)/'run'); adapter=FixtureAdapter()
        argv=['driver','--adapter','rlhflow_prm','--meta-pkl',source,'--out',output]
        D.STOP['flag']=True
        with patch.object(sys,'argv',argv), patch.object(D,'get_adapter',return_value=adapter), patch.object(D.signal,'signal'):
            with self.assertRaises(SystemExit) as stopped: D.main()
        self.assertEqual(stopped.exception.code,85)
        manifest=json.loads((Path(output)/'manifest.json').read_text())
        self.assertEqual(manifest['status'],'INCOMPLETE')
        self.assertEqual(manifest['n_rows'],0)
        self.assertTrue((Path(output)/'run_identity.json').exists())
        self.assertTrue((Path(output)/'prmbench_telemetry.pkl').exists())
        # A changed row must not resume into the same checkpoint.
        save_cache_atomic({0:row('a','changed question')},source);D.STOP['flag']=False
        with patch.object(sys,'argv',argv), patch.object(D,'get_adapter',return_value=adapter), patch.object(D.signal,'signal'):
            with self.assertRaisesRegex(ValueError,'resume identity differs'): D.main()

    def test_main_writes_counted_context_failures_and_exits_2(self):
        source=str(Path(self.temp.name)/'meta.pkl')
        save_cache_atomic({0:row('a'),1:row('b','long')},source)
        output=str(Path(self.temp.name)/'run');adapter=FixtureAdapter()
        argv=['driver','--adapter','rlhflow_prm','--meta-pkl',source,'--out',output,'--logprob-top-k','15']
        with patch.object(sys,'argv',argv), patch.object(D,'get_adapter',return_value=adapter), patch.object(D.signal,'signal'):
            with self.assertRaises(SystemExit) as stopped:D.main()
        self.assertEqual(stopped.exception.code,2)
        manifest=json.loads((Path(output)/'manifest.json').read_text())
        self.assertEqual(manifest['status'],'COMPLETE_WITH_FAILURES')
        self.assertEqual(manifest['n_rows_context_overflow'],1)
        self.assertEqual(manifest['n_successful_rows'],1)

    def test_main_allows_pilot_prefix_extension(self):
        source=str(Path(self.temp.name)/'meta.pkl');save_cache_atomic({0:row('a')},source)
        output=str(Path(self.temp.name)/'run');adapter=FixtureAdapter()
        argv=['driver','--adapter','rlhflow_prm','--meta-pkl',source,'--out',output,'--logprob-top-k','15']
        with patch.object(sys,'argv',argv), patch.object(D,'get_adapter',return_value=adapter), patch.object(D.signal,'signal'), patch.object(D,'free_memory'):
            D.main()
            save_cache_atomic({0:row('a'),1:row('b')},source)
            D.main()
        self.assertEqual(adapter.forward_count,2)
        self.assertEqual(len(load_cache(str(Path(output)/'prmbench_telemetry.pkl'))),2)

    def test_chunked_telemetry_matches_one_projection(self):
        adapter=FixtureAdapter();h=torch.randn(1,2050,8);pos=list(range(1,2050));ids=[i%32 for i in range(2050)]
        one=D.chunk_quantities(adapter,h,pos,ids,15,len(pos))
        chunk=D.chunk_quantities(adapter,h,pos,ids,15,1024)
        for key in ['token_entropies','token_spilled_energies','token_logsumexp']:
            np.testing.assert_allclose(one[key],chunk[key],atol=1e-5,rtol=0)
        np.testing.assert_array_equal(one['top_k_logprobs']['ids'],chunk['top_k_logprobs']['ids'])
        np.testing.assert_allclose(one['top_k_logprobs']['logprobs'],chunk['top_k_logprobs']['logprobs'],atol=1e-5,rtol=0)
    def test_regression_comparator_catches_corruption(self):
        adapter=FixtureAdapter()
        with patch.object(D,'get_adapter',return_value=adapter):D.run_prm(self.cfg,[row('a')],self.path)
        cache=load_cache(self.path)
        self.assertEqual(D.compare_caches(cache,self.path,1)['status'],'PASS')
        bad=copy.deepcopy(cache);bad[0]['gen_token_ids'][0]+=1
        with self.assertRaisesRegex(ValueError,'gen_token_ids'):D.compare_caches(bad,self.path,1)
        bad=copy.deepcopy(cache);bad[0]['rewards'][0]+=.001
        with self.assertRaisesRegex(ValueError,'rewards'):D.compare_caches(bad,self.path,1)
        with self.assertRaisesRegex(ValueError,'requires 40'):D.compare_caches(cache,self.path)


if __name__=='__main__':unittest.main()
