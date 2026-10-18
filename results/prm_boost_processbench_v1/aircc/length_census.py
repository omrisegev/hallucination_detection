"""Tokenizer-only context census for ProcessBench x the 5 PRM adapters (AIRCC CPU, no model weights, no GPU).

For every answer, builds the adapter's exact input with spectral_utils.prm_adapters.<adapter>.build (the same
call the telemetry driver makes) and compares len(input_ids) with the checkpoint's max_position_embeddings, the
driver's own context-failure test. Over-long answers are NOT truncated: the driver records them as counted
failures (failure='context_overflow'). This census predicts that count before any GPU time, and also counts
step-text mismatches (n_step_text_mismatch), the check rlhflow_prm failed on Hard2Verify.

Run with PYTHONPATH = the telemetry code dir (code_integration_7534b8bf4) and the offline HF cache:
    python length_census.py --answers $SHARED/data/prm_boost_processbench_v1/answers.json \
        --out $SHARED/results/prm_boost_processbench_v1/census [--adapters qwen_prm ...]
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np

ADAPTERS5 = ('qwen_prm', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd')


def census(adapter, tok, context, rows):
    lens, mism, over, errors = [], 0, [], {}
    for r in rows:
        try:
            b = adapter.build(tok, r['question'], r['steps'])
        except Exception as exc:  # a build error is itself a finding; the driver would crash on it
            errors[r['uid']] = repr(exc)[:300]; continue
        n = len(b['input_ids']); lens.append(n); mism += int(b['n_step_text_mismatch'] > 0)
        if n > context:
            over.append({'uid': r['uid'], 'n_input_tokens': n})
    L = np.array(lens) if lens else np.zeros(1)
    by = {}
    for r in rows:
        s = r.get('benchmark'); by.setdefault(s, 0)
    for o in over:
        by[next(r['benchmark'] for r in rows if r['uid'] == o['uid'])] += 1
    return {'context_limit': int(context), 'n_rows': len(rows), 'n_built': len(lens),
            'n_context_overflow': len(over), 'overflow_by_subset': by, 'overflow_rows': over,
            'n_rows_step_text_mismatch': mism, 'build_errors': errors,
            'max_input_tokens': int(L.max()), 'p99_input_tokens': float(np.percentile(L, 99)),
            'mean_input_tokens': float(L.mean()), 'total_input_tokens': int(L.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--adapters', nargs='+', default=list(ADAPTERS5))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoConfig, AutoTokenizer
    from spectral_utils.prm_adapters import get_adapter
    rows = json.loads(a.answers.read_text(encoding='utf8')); summary = {}
    for name in a.adapters:
        t0 = time.time(); ad = get_adapter(name)
        tok = AutoTokenizer.from_pretrained(ad.model_id, trust_remote_code=True)
        context = AutoConfig.from_pretrained(ad.model_id, trust_remote_code=True).max_position_embeddings
        res = dict(adapter=name, model=ad.model_id, **census(ad, tok, context, rows), seconds=round(time.time() - t0, 1),
                   job_id=os.environ.get('SLURM_JOB_ID', ''))
        (a.out / f'CENSUS_{name}.json').write_text(json.dumps(res, indent=1))
        summary[name] = {k: res[k] for k in ('context_limit', 'n_context_overflow', 'overflow_by_subset',
                                             'n_rows_step_text_mismatch', 'max_input_tokens', 'total_input_tokens')}
        summary[name]['n_build_errors'] = len(res['build_errors'])
        print(name, summary[name], flush=True)
    (a.out / 'CENSUS_SUMMARY.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
