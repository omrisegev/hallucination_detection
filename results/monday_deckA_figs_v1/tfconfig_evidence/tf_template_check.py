import sys, os
sys.path.insert(0, r"c:\Users\omris\TAU\hallucination_detection")
os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoTokenizer
from spectral_utils.data_loaders import math_prompt
from spectral_utils.processbench import NO_THINK_SUFFIX, build_chain, step_token_spans
for mid in ["Qwen/Qwen3-4B","Qwen/Qwen3-8B"]:
    tok = AutoTokenizer.from_pretrained(mid)
    msg = math_prompt({"problem":"What is 3 times 4?"}) + NO_THINK_SUFFIX
    # mirrors spectral_utils.model_utils.fmt_prompt (system_message=None)
    ours = tok.apply_chat_template([{"role":"user","content":msg}], tokenize=False, add_generation_prompt=True)
    off = tok.apply_chat_template([{"role":"user","content":msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print("=====", mid)
    print("OURS repr tail:", repr(ours[-120:]))
    print("enable_thinking=False tail:", repr(off[-120:]))
    print("identical:", ours==off)
    text, spans = build_chain(["We multiply 3 by 4.", "3 x 4 = 12."])
    ids, sp = step_token_spans(tok, text, spans)
    full = tok(ours).input_ids + ids
    print("first forced tokens:", [tok.decode([i]) for i in ids[:6]])
    print("last prompt tokens:", [tok.decode([i]) for i in tok(ours).input_ids[-6:]])
    print("think ids:", tok.convert_tokens_to_ids(["<think>","</think>"]))
    # would a generated /no_think answer (as in the template with an assistant turn) contain think block?
    hist = tok.apply_chat_template([{"role":"user","content":msg},{"role":"assistant","content":"We multiply."}], tokenize=False)
    print("template rendering of a past assistant turn:", repr(hist[-80:]))
