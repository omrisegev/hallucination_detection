"""CPU preflight for the standalone runs: real tokenizer of each standalone model on 3 rows of its PRM's backbone pkl.
Prints the decoded prompt and checks one span per step, the prompt's BOS, and the context length. No model weights.

    python3 tokenizer_smoke.py <CODE_DIR> <prm>=<snapshot dir>=<prompt style>=<backbone pkl> [...]
"""
import pickle, sys
from transformers import AutoTokenizer

code = sys.argv[1]; sys.path[:0] = [code, code + '/cluster']
import importlib.util
spec = importlib.util.spec_from_file_location('tel', code + '/cluster/run_prmbench_prm_telemetry.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
from spectral_utils.prmbench import build_chain, step_token_spans, assert_alignment

bad = 0
for arg in sys.argv[2:]:
    prm, snap, style, pkl = arg.split('=')
    tok = AutoTokenizer.from_pretrained(snap)
    rows = [r for _, r in sorted(pickle.load(open(pkl, 'rb')).items())][:3]
    print(f'== {prm} {snap.split("/models--")[-1][:60]} style={style} chat_template={bool(tok.chat_template)} bos={tok.bos_token_id}')
    for r in rows:
        ids = D.standalone_prompt_ids(tok, style, r['question'])
        text, cs = build_chain(r['steps']); g, sp = step_token_spans(tok, text, cs)
        sp = D.fill_empty_spans(sp); diag = assert_alignment(g, sp, r['steps'], strict=False)
        ok = len(sp) == len(r['steps'])
        bad += not ok
        print(f"  row {r['idx']}: prompt {len(ids)} + chain {len(g)} tokens, {len(sp)} spans for {len(r['steps'])} steps, ok={ok}")
    print('  prompt:', repr(tok.decode(ids))[:400])
print('TOKENIZER_SMOKE', 'FAIL' if bad else 'PASS')
sys.exit(1 if bad else 0)
