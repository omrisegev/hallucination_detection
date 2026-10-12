"""Source mapping, real Qwen3 template, spans, blindness and CPU model/resume gates."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from copy import deepcopy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_utils.agent_bench.contracts import (AgentItem,adapt_apb,adapt_tprm,adapt_aft,
    history_GTA,history_bfcl,history_tooltalk,history_ToolSandbox,blind_keys,validate)
from spectral_utils.agent_bench.render_qwen3 import render,tokenize,native,HEADER,EMPTY
from spectral_utils.external_generalization.artifacts import RecordStore,atomic_json


def fixtures():
    apb=[]
    for i in range(3):
        row={"total_index":i,"query_index":0,"messages":[{"role":"system","content":"system"},
              {"role":"user","content":"question"},{"role":"assistant","content":"abc"},
              {"role":"tool","content":"def"},{"role":"assistant","content":"xyz"}],
              "tools":[],"step_labels":{"2":1,"4":-1},"final_label":-1}
        apb.append(adapt_apb(row,"bfcl")[0])
    row={"sample_id":"fixture","history":[{"role":"system","content":"system"},{"role":"user","content":"question"}],
         "functions":[],"action_chosen":json.dumps({"api_name":"a","parameters":{}}),
         "action_rejected":json.dumps({"api_name":"b","parameters":{}})}
    tprm=adapt_tprm(row,"tooltalk")[0]
    row=deepcopy(row);row['sample_id']='fixture2'
    tprm+=adapt_tprm(row,"tooltalk")[0][:1]
    aft=[]
    for i in range(3):
        row={"conv_id":str(i),"domain":"math","task":"question","num_turns":3,"tools":[],
             "turns":[{"role":"user","content":"abc"},{"role":"agent","thought":"def","action":"xyz"},
                      {"role":"tool","content":""}]}
        aft.append(adapt_aft(row,True,set())[0])
    return apb+tprm+aft


def tokenizer_directory():
    return Path(os.environ.get("AGENT_TOKENIZER_DIR",
        "/Users/osegev/Desktop/hallucination_detection/local_cache/agent_step_telemetry_v1/sources_verified/tokenizer"))


class CharTokenizer:
    """Reversible tiny-vocabulary tokenizer using the actual pinned Qwen3 Jinja template."""
    def __init__(self):
        self.chat_template=(ROOT/'tests/fixtures/agent_qwen3_chat_template.jinja').read_text()
        self.chars={}; self.reverse={}
    def apply_chat_template(self,messages,**kwargs):
        from transformers.utils.chat_template_utils import _compile_jinja_template
        return _compile_jinja_template(self.chat_template).render(messages=messages,**kwargs)
    def __call__(self,text,**kwargs):
        ids=[]
        for c in text:
            if c not in self.chars:
                index=len(self.chars)+1
                if index>=64:
                    raise ValueError("fixture exceeds tiny vocabulary")
                self.chars[c]=index;self.reverse[index]=c
            ids.append(self.chars[c])
        return {"input_ids":ids,"offset_mapping":[(i,i+1) for i in range(len(text))]}
    def decode(self,ids,**kwargs):
        return "".join(self.reverse[i] for i in ids)


class Contracts(unittest.TestCase):
    def test_apb_raw_arguments_and_second_system(self):
        row={"total_index":1,"query_index":2,"messages":[{"role":"system","content":"a"},
            {"role":"system","content":"state"},{"role":"assistant","content":None,
            "tool_calls":[{"function":{"name":"f","arguments":"{not json}"}}]}],
            "tools":[{"name":"f","description":"d","parameters":{},"response":{"x":1}}],
            "step_labels":{"2":1},"final_label":1}
        diagnostics=[];a,g=adapt_apb(row,"bfcl",diagnostics)
        self.assertEqual(a.messages[1],row['messages'][1])
        self.assertEqual(a.messages[2]['tool_calls'],row['messages'][2]['tool_calls'])
        self.assertEqual(a.tools[0]['function']['response'],{"x":1})
        self.assertEqual([d['kind'] for d in diagnostics],['null_content','invalid_argument_json'])
        row['step_labels']={'2':-1};row['final_label']=-1
        self.assertEqual(a,adapt_apb(row,'bfcl')[0])
        row['step_labels']={}
        with self.assertRaises(ValueError):adapt_apb(row,'bfcl')

    def test_reasoning_preservation_and_unexpected_tag(self):
        row={"total_index":1,"query_index":2,"messages":[{"role":"user","content":"q"},
            {"role":"assistant","content":"<think>\nreason\n</think>\nabc"}],"tools":[],"step_labels":{"1":1}}
        a,g=adapt_apb(row,'hotpotqa'); text,spans=render(a,CharTokenizer())
        self.assertIn('<think>\nreason\n</think>\n\nabc',text)
        self.assertEqual(text[slice(*spans['steps'][0]['reasoning_char_span'])],'reason')
        with self.assertRaises(ValueError):adapt_apb(row,'bfcl')

    def test_history_tooltalk(self):
        r={'history':[{'role':'user','content':' q\n'}],'functions':[{'name':'f','parameters':{}}]}
        ms,tools=history_tooltalk(r)
        self.assertEqual(ms,r['history']);self.assertEqual(tools[0]['function'],r['functions'][0])

    def test_history_GTA(self):
        r={'history':[{'role':'system','content':'tool list'}],'functions':None}
        self.assertEqual(history_GTA(r),(r['history'],[]))
        r['functions']=[]
        with self.assertRaises(ValueError):history_GTA(r)

    def test_history_bfcl(self):
        r={'history':[{'role':'assistant','content':[{'f':'{ "x": 1 }'},{'g':'{}'}]},
                      {'role':'tool','content':' result\n'}],
           'functions':[{'api_names':['Class.f','f'],'parameters':{},'response':{'type':'dict'}}]}
        ms,tools=history_bfcl(r)
        self.assertEqual([c['function']['name'] for c in ms[0]['tool_calls']],['f','g'])
        self.assertEqual(ms[0]['tool_calls'][0]['function']['arguments'],'{ "x": 1 }')
        self.assertEqual(ms[1],r['history'][1]);self.assertEqual(tools[0]['function']['name'],'Class.f')

    def test_history_ToolSandbox(self):
        r={'history':[{'role':'assistant','content':'thinking','tool':{'name':'f','arguments':{'x':1}}}],
           'functions':[]}
        ms,tools=history_ToolSandbox(r)
        self.assertEqual(ms[0]['content'],'thinking');self.assertEqual(ms[0]['tool_calls'][0]['function'],r['history'][0]['tool'])

    def test_pair_blindness_and_identity_all_subsets(self):
        for subset in ('GTA','bfcl','tooltalk','ToolSandbox'):
            r={'sample_id':'x','history':[{'role':'user','content':'q'}],'functions':None if subset=='GTA' else []}
            if subset=='GTA':a,b='a','b'
            elif subset=='bfcl':a,b={'name':'f','parameters':{}},{'name':'g','parameters':{}}
            elif subset=='tooltalk':a,b=json.dumps({'api_name':'f','parameters':{}}),json.dumps({'api_name':'g','parameters':{}})
            else:a,b={'role':'assistant','content':'','tool':{'name':'f','arguments':{}}},{'role':'assistant','content':'','tool':{'name':'g','arguments':{}}}
            r.update(action_chosen=a,action_rejected=b,rationale='never in prompt')
            rows,g=adapt_tprm(r,subset); tok=CharTokenizer()
            texts=[render(x,tok) for x in rows];sp=[x[1]['step_char_spans'][0] for x in texts]
            self.assertEqual(texts[0][0][:sp[0][0]],texts[1][0][:sp[1][0]])
            self.assertNotEqual(texts[0][0][slice(*sp[0])],texts[1][0][slice(*sp[1])])
            self.assertFalse(blind_keys(rows[0].to_dict()))

    def test_aft_all_fields_no_truncation_empty_steps_and_index_zero(self):
        r={'conv_id':'x','domain':'math','task':' task\n','num_turns':3,'tools':[{'name':'f','description':'d'}],
           'gold_answer':'secret','mistake_step':0,'unsafe_source':'injected',
           'turns':[{'role':'agent','thought':'T','action':'A','content':'C'*2100},
                    {'role':'tool','content':''},{'role':'user','content':'Z'}]}
        a,g=adapt_aft(r,False,{'x'});self.assertEqual(g['mistake_step'],0);self.assertTrue(g['split_test'])
        self.assertEqual(a.messages[1]['content'],r['task']);self.assertNotIn('secret',str(a.messages))
        self.assertIn('C'*2100,a.messages[2]['content']);self.assertEqual(a.entry_spans[1][0],a.entry_spans[1][1])
        self.assertEqual(a.entry_to_history,[0,1,2]);self.assertEqual(a.entry_field_spans[0]['thought'][1]-a.entry_field_spans[0]['thought'][0],1)
        with self.assertRaises(ValueError):adapt_aft(r,True,set())

    def test_exact_spans_context_tools_and_prefix_parity(self):
        tok=CharTokenizer();a=fixtures()[0];text,s=render(a,tok)
        self.assertEqual(text.count(EMPTY),2)
        for c in s['context_spans']:
            self.assertEqual(text[slice(*c['char_span'])],a.messages[c['message_index']]['content'])
        item=tokenize(a,tok,max_context=5);self.assertTrue(item['overlength'])
        for (left,right),(c,d) in zip(item['step_char_spans'],item['step_token_spans']):
            self.assertEqual(tok.decode(item['gen_ids'][c:d]),text[left:right])
        prefix=AgentItem('x','apb','bfcl','x','x',[{'role':'user','content':'q'},{'role':'assistant','content':'a'}],[],[1])
        text,sp=render(prefix,tok)
        self.assertEqual(text[:sp['step_char_spans'][0][0]],native(tok,prefix.messages[:1],[],generation=True))
        # Native full-prefix parity cannot hold after earlier scored turns were corrected.
        corrected=render(a,tok)[0];self.assertNotEqual(corrected,native(tok,a.messages,a.tools))

    def test_recursive_blind_keys_and_duplicate_step(self):
        self.assertEqual(blind_keys({'tools':[{'groundtruth':'x'}]}),['$.tools[0].groundtruth'])
        a=fixtures()[0].to_dict();a['step_message_indices']=[2,2]
        with self.assertRaises(ValueError):validate(a)

    def test_approved_source_amendments(self):
        row={'sample_id':'same','history':[{'role':'assistant','content':'<think>\nraw'}],
             'functions':[],'action_chosen':{'role':'assistant','content':'a'},
             'action_rejected':{'role':'assistant','content':'b'}}
        a,_=adapt_tprm(row,'ToolSandbox',case_index=0)
        b,_=adapt_tprm(row,'ToolSandbox',case_index=1)
        self.assertNotEqual(a[0].uid,b[0].uid);self.assertEqual(a[0].group,b[0].group)
        self.assertEqual(a[0].messages[0]['content'],'<think>\nraw')
        self.assertIn(a[0].messages[-1]['content'],('a','b'))
        row['history']=[{'role':'assistant','content':'a</think>\nb'}]
        rows,_=adapt_tprm(row,'ToolSandbox',case_index=2)
        text,_=render(rows[0],CharTokenizer())
        self.assertIn('a</think>\nb',text)
        row['history']=[{'role':'user','content':'q'}]
        row['action_chosen']={'tool_name':'f','arguments':{'x':1}}
        row['action_rejected']={'tool_name':'f','arguments':{}}
        rows,_=adapt_tprm(row,'ToolSandbox',case_index=3)
        self.assertNotEqual(rows[0].messages[-1]['tool_calls'],rows[1].messages[-1]['tool_calls'])

    def test_sources_checked_before_read_and_mutation_refused(self):
        import hashlib
        spec=importlib.util.spec_from_file_location('prepare_inputs',ROOT/'scripts/agent_bench/prepare_agent_inputs.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'x').write_bytes(b'abc')
            (p/'SOURCES.json').write_text(json.dumps({'files':[{'key':'apb','path':'x','local_path':'x','bytes':3,
                                                               'sha256':hashlib.sha256(b'abc').hexdigest()}]}))
            module.verify_sources(p)
            (p/'x').write_bytes(b'abd')
            with self.assertRaises(ValueError):module.verify_sources(p)

    def test_store_changed_identity_and_single_writer(self):
        with tempfile.TemporaryDirectory() as d:
            with RecordStore(d,{'code':1}) as store:
                store.put('x',{'a':1})
                with self.assertRaises(FileExistsError):
                    with RecordStore(d,{'code':1}):pass
            with self.assertRaises(ValueError):
                with RecordStore(d,{'code':2}):pass

    def test_native_bpe_boundary_is_reported_without_changing_text(self):
        from transformers import AutoTokenizer
        if not (tokenizer_directory()/'tokenizer.json').exists():
            self.skipTest('prepare pinned tokenizer or set AGENT_TOKENIZER_DIR for native BPE test')
        tok=AutoTokenizer.from_pretrained(tokenizer_directory(),local_files_only=True,use_fast=True)
        row={'conv_id':'boundary','domain':'math','task':'q','num_turns':2,'tools':[],
             'turns':[{'role':'agent','content':'abc.'},{'role':'tool','content':'xyz.'}]}
        item,_=adapt_aft(row,True,set());text,spans=render(item,tok)
        self.assertEqual(text[slice(*spans['step_char_spans'][0])],'[Content] abc.')
        with self.assertRaisesRegex(ValueError,'decoded tokens differ'):
            tokenize(item,tok)

    def test_empty_transcript_header_has_no_extra_newline(self):
        item=fixtures()[-1]
        self.assertTrue(item.messages[2]['content'].endswith('[2] tool:'))

    def test_empty_scored_turn_uses_one_forward_and_keeps_zero_span(self):
        import torch
        from transformers import GPT2Config,GPT2LMHeadModel
        spec=importlib.util.spec_from_file_location('agent_empty_driver',ROOT/'cluster/run_agent_telemetry.py')
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        model=GPT2LMHeadModel(GPT2Config(vocab_size=64,n_positions=512,n_embd=16,n_layer=1,n_head=2)).eval()
        item=AgentItem('empty','apb','fixture','x','x',[{'role':'user','content':'q'},
                      {'role':'assistant','content':''}],[],[1])
        tokenized=tokenize(item,CharTokenizer())
        self.assertEqual(tokenized['step_token_spans'][0][0],tokenized['step_token_spans'][0][1])
        with mock.patch.object(model,'forward',wraps=model.forward) as forward:
            payload=driver.collect_quantities(model,tokenized)
            self.assertEqual(forward.call_count,1)
        self.assertEqual(payload['top_k_token_indices'],[])
        self.assertEqual(payload['top_k_logprobs'],{'ids':[],'logprobs':[]})


def cpu_smoke(out):
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    spec=importlib.util.spec_from_file_location('agent_driver',ROOT/'cluster/run_agent_telemetry.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    torch.manual_seed(0)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=64,n_positions=512,n_embd=16,n_layer=1,n_head=2,
                                  bos_token_id=1,eos_token_id=2)).eval()
    tok=CharTokenizer();items=driver.make_items([a.to_dict() for a in fixtures()],tok,512)
    assert len(items)==9 and not any(i['overlength'] for i in items)
    gate=driver.alignment_gate(model,min(items,key=lambda i:len(i['gen_ids'])))
    identity={'cpu_smoke':1,'code':driver.code_identity()}
    calls=[]
    def interrupt(item):
        payload=driver.collect_quantities(model,item);calls.append(item['uid'])
        if len(calls)==4:driver.stop()
        return payload
    with RecordStore(out/'records',identity) as store:
        complete,rows=driver.run(items,store,interrupt,range(9));assert not complete and len(rows)==4
    driver.STOP=False
    with RecordStore(out/'records',identity) as store:
        complete,rows=driver.run(items,store,lambda i:driver.collect_quantities(model,i),range(9))
        assert complete and len(rows)==9
        complete,rows=driver.run(items,store,lambda i:(_ for _ in ()).throw(AssertionError('rescored')),range(9))
        assert complete
    with mock.patch.object(driver,'STOP',False):
        with RecordStore(out/'second',identity) as store:
            complete,_=driver.run(items,store,lambda i:driver.collect_quantities(model,i),range(9));assert complete
    with RecordStore(out/'records',identity) as first,RecordStore(out/'second',identity) as second:
        for item in items:
            assert first.get(item['uid'])['telemetry']==second.get(item['uid'])['telemetry']
    result={'status':'PASS','items':9,'per_benchmark':{'apb':3,'tprm':3,'aft':3},'interrupted_after':4,
            'resume_identical_to_fresh':True,'alignment_gate':gate,'model':'random GPT2 64 vocab, 1 layer, 16 dim'}
    atomic_json(out/'G6.json',result);return result


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--cpu-smoke':
        print(json.dumps(cpu_smoke(Path(sys.argv[2]))))
    else:unittest.main()
