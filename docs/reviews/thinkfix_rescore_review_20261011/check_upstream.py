"""Compare declared pooled F1 with the upstream task's actual numerical functions.

This is evaluator parity on the exported slice, not a full official dataset run.
No upstream imports, model inference, installs, or remote writes.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--meta', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    tree = ast.parse(a.upstream.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {'evaluate_function', 'eval_on_hallucination_step'}]
    assert len(nodes) == 2
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(a.upstream), 'exec'), namespace)
    b = np.load(a.bank, allow_pickle=False)
    metadata = json.loads(a.meta.read_text())['rows']
    scores = b['S__MEAN_MOM_ga']
    q = np.quantile(scores, .8)
    results = []
    for idx, start, end in zip(b['idx'], b['offsets'][:-1], b['offsets'][1:]):
        results.append(dict(idx=str(idx), scores=dict(
            step_level_validity_labels=(scores[start:end] <= q).tolist())))
    result = namespace['evaluate_function'](results, metadata)
    total = result['total_hallucination_results']
    cats = result['hallucination_type_results']
    c = {name: {m: cats[m][name] for m in ['f1', 'negative_f1']}
         for name in cats['f1']}
    for v in c.values():
        v['half_f1_plus_negative_f1'] = (v['f1'] + v['negative_f1']) / 2
    report = dict(
        upstream_source_sha256=hashlib.sha256(a.upstream.read_bytes()).hexdigest(),
        scope='Upstream numerical task functions, exported deduplicated slice, validity-head fallback, global answer-z Q80; no official dataset loader or model runner',
        overall=dict(f1=total['f1'], negative_f1=total['negative_f1'],
                     half_f1_plus_negative_f1=(total['f1']+total['negative_f1'])/2),
        by_class=c,
        validity_rate=result['validitiy_rate'])
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
