"""Compare existing smoke/full overlap without inference, labels or quality scores."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(root):
    cells = [('hard2verify_qwen3_8b', 'bfdd74490'),
             ('socratic_qwen3_8b', 'bfdd74490'),
             ('socratic_qwq32b', '956a1aa19_complete')]
    result = []
    for cell, smoke_revision in cells:
        smoke = root / (cell + '_smoke_' + smoke_revision)
        full = root / (cell + '_full_411316927')
        smoke_timing = json.loads((smoke / 'TIMING.json').read_text())
        full_timing = json.loads((full / 'TIMING.json').read_text())
        for key in ['model', 'revision', 'dtype', 'attn', 'answers_sha256',
                    'protocol_sha256', 'chat_template_sha256', 'prompt_suffix', 'thinking_mode']:
            assert smoke_timing['identity'][key] == full_timing['identity'][key], key
        row = dict(cell=cell, smoke_job=smoke_timing['job_id'], full_job=full_timing['job_id'],
                   answers=0, tokens=0, identical_telemetry_answers=0,
                   differing_top50_id_rows=0, differing_fields={}, scalar_deltas={})
        for path in sorted((smoke/'records').glob('*.record.json')):
            a = json.loads(path.read_text())
            b = json.loads((full/'records'/path.name).read_text())
            assert a['uid'] == b['uid']
            x, y = a['payload']['telemetry'], b['payload']['telemetry']
            assert set(x) == set(y)
            for key in ['gen_token_ids', 'prompt_ids', 'token_offsets', 'step_token_spans', 'step_char_spans']:
                assert x[key] == y[key], (a['uid'], key)
            row['answers'] += 1
            row['tokens'] += len(x['gen_token_ids'])
            row['identical_telemetry_answers'] += x == y
            for key in x:
                if x[key] != y[key]:
                    row['differing_fields'][key] = row['differing_fields'].get(key, 0) + 1
            row['differing_top50_id_rows'] += sum(i != j for i,j in zip(x['top_k_logprobs']['ids'],y['top_k_logprobs']['ids']))
            for key in ['actual_token_logprobs','token_entropy_full','token_entropies','token_logsumexp']:
                entry = row['scalar_deltas'].setdefault(key, dict(max_abs=0.0, sum_abs=0.0, unequal_tokens=0))
                for i,j in zip(x[key],y[key]):
                    delta = abs(i-j)
                    entry['max_abs'] = max(entry['max_abs'],delta)
                    entry['sum_abs'] += delta
                    entry['unequal_tokens'] += i != j
        assert row['answers'] == len(smoke_timing['identity']['selected_uids']) == 12
        for entry in row['scalar_deltas'].values():
            entry['mean_abs'] = entry.pop('sum_abs') / row['tokens']
        result.append(row)
    return dict(scope='Existing smoke/full overlap: two executions per example, 12 answers per cell',
                new_inference=False, quality_evaluated=False, cells=result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(compare(args.root),indent=2))
