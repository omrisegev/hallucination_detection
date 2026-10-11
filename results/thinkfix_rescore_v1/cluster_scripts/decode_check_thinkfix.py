"""Decode check for the think-fix PRMBench pilot: rebuild item 0 exactly as the driver does and
print the tail of its prompt and the head of its forced ids, plus the saved pkl's first ids."""
import sys, pickle
sys.path.insert(0, "cluster"); sys.path.insert(0, ".")
from transformers import AutoTokenizer
from spectral_utils.prmbench import load_prmbench
from run_prmbench_teacher_forced import build_items
from spectral_utils.processbench import NO_THINK_SUFFIX, EMPTY_THINK_BLOCK

pkl = sys.argv[1]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
meta, _ = load_prmbench(n_samples=60, seed=0)
items = build_items(tok, meta, NO_THINK_SUFFIX, "empty_block")
it = items[0]
p = it["prompt_ids"]
print("prompt tail repr:", repr(tok.decode(p[-12:])))
print("prompt ends with empty block:", tok.decode(p).endswith(EMPTY_THINK_BLOCK))
print("gen head repr:", repr(tok.decode(it["gen_ids"][:20])))
cache = pickle.load(open(pkl, "rb"))
e = cache[sorted(cache)[0]]
print("pkl row idx", e["idx"], "matches item idx", it["row"]["idx"])
print("pkl gen head repr:", repr(tok.decode(list(e["gen_token_ids"][:20]))))
print("pkl gen ids == rebuilt:", list(e["gen_token_ids"]) == list(it["gen_ids"]))
print("first step text head:", repr(e["steps"][0][:80]))
print("first-token surprisal:", float(e["token_spilled_energies"][0]),
      "top1 id:", int(e["top_k_logprobs"]["ids"][0][0]), repr(tok.decode([int(e["top_k_logprobs"]["ids"][0][0])])))
import numpy as np
for k in sorted(cache)[:8]:
    e = cache[k]; ids = e["top_k_logprobs"]["ids"][0]; lps = e["top_k_logprobs"]["logprobs"][0]
    g = int(e["gen_token_ids"][0]); hit = [float(l) for i, l in zip(ids, lps) if int(i) == g]
    print(k, "gen0", repr(tok.decode([g])), "logp(gen0)", hit[0] if hit else "not in top50",
          "spilled0", round(float(e["token_spilled_energies"][0]), 3), "lse0", round(float(e["token_logsumexp"][0]), 3),
          "H0", round(float(e["token_entropies"][0]), 4),
          "top3", [(repr(tok.decode([int(i)])), round(float(l), 3)) for i, l in zip(ids[:3], lps[:3])])
