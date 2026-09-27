"""Bounded diagnostic for QwQ prefix-length numerical differences; no quality."""
import json
import os
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_external_telemetry import make_items
from backfill_views import forward_batch
from spectral_utils.external_generalization.contracts import Answer
from spectral_utils.external_generalization.artifacts import atomic_json


def compare(model, item):
    x = dict(item, gen_ids=item["gen_ids"][:16])
    with torch.no_grad():
        reference = forward_batch(model, [x])[0].float().log_softmax(-1)
        results = []
        for j in [0, 8, 15]:
            prefix = x["prompt_ids"] + x["gen_ids"][:j]
            ids = torch.tensor([prefix], device=model.device)
            plain = model(input_ids=ids, use_cache=False).logits[0,-1].float().log_softmax(-1)
            masked = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0,-1].float().log_softmax(-1)
            changed = dict(x, gen_ids=x["gen_ids"][:j]+[0]*(len(x["gen_ids"])-j))
            causal = forward_batch(model, [changed])[0][j].float().log_softmax(-1)
            token = x["gen_ids"][j]
            results.append({"position":j, "reference_actual_logprob":float(reference[j,token]),
                            "plain_prefix_delta":float(plain[token]-reference[j,token]),
                            "masked_prefix_delta":float(masked[token]-reference[j,token]),
                            "same_length_future_perturbation_max_delta":float((causal-reference[j]).abs().max())})
    return results


def main():
    base = Path('/shared/cycle2_tau_averbuch_prj/omrisegev1')
    model_id, revision = 'Qwen/QwQ-32B', '976055f8c83f394f35dbd3ab09a285a984907bd0'
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    answers = [Answer(**r) for r in json.loads((base/'data/lsml_external_20260924/socratic/answers.json').read_text())]
    items = make_items(answers, tok, 32768)
    item = min(items, key=lambda r:len(r['prompt_ids'])+len(r['gen_ids']))
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16,
                                                device_map='auto', attn_implementation='sdpa').eval()
    result = {'uid':item['uid'], 'job_id':os.environ.get('SLURM_JOB_ID'), 'quality_evaluated':False,
              'bf16':compare(model,item), 'use_sliding_window':model.config.use_sliding_window}
    # Same already-quantized weights in fp32 isolate arithmetic/layout error.
    model.float()
    result['same_bf16_weights_fp32_arithmetic'] = compare(model,item)
    atomic_json(base/'results/lsml_external_generalization_v1/QWQ_ALIGNMENT_DIAGNOSTIC.json',result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
