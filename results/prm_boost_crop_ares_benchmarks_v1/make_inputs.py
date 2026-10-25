"""Build label-free answers.json files for the CROP / ARES benchmarks we did not have, plus separate gold files.

Benchmarks (plan: docs/experiments/PRM_BOOST_MISSING_BENCHMARKS_2026-10-26_HE.md):
  crv_arith      facebook/crv arith.nt{3,5,7,10}.annotated.json, CROP's 1,500-trace subset
  crv_gsm8k      facebook/crv gsm8k.annotated.json, all 1,319 traces
  math_shepherd  peiyi9979/Math-Shepherd train, first 10,000 rows (CROP)
  prm800k        openai/prm800k phase2_test then phase2_train, first 8,000 traces (CROP)
  deltabench     OpenStellarTeam/DeltaBench Deltabench_v1.csv, one step per section (ARES)

The selection and label rules re-implement CROP's code (github.com/matthewyccheung/crop: crop/cheap_baselines.py
select_target_traces/_sample_group, crop/scripts/import_external_process.py). CROP's code is read, never run.
Counts are checked against CROP Table 4; the build stops on a mismatch unless --allow-count-mismatch.

--src layout (fetched by fetch_sources.sh):
  crv/arithmetic_expressions/arith.nt*.annotated.json, crv/gsm8k_expressions/gsm8k.annotated.json
  math_shepherd/math-shepherd.first10000.jsonl          (first 10,000 lines = first 10,000 streamed rows)
  prm800k/phase2_test.jsonl, prm800k/phase2_train.first12000.jsonl
  deltabench/Deltabench_v1.csv

Outputs:
  <out>/<bench>/answers.json        [{uid, benchmark, source_id, question, original_question, steps, generator}]
  <out>/<bench>/answers_timing12.json   12 deterministic rows (seed 20261026)
  <gold>/<bench>.json               [{uid, correct, include, category}]  (correct[i] = step i is not an error)
  <out>/INPUT_MANIFEST.json         counts, CROP Table 4 targets, sha256 of every source and output

    python results/prm_boost_crop_ares_benchmarks_v1/make_inputs.py --src <src> --out <inputs> --gold <local>
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

CROP_SEED = 2806
TIMING_SEED, TIMING_N = 20261026, 12
# CROP Table 4 (traces, steps, step error rate)
CROP_TABLE4 = {'crv_arith': (1500, 10316, 0.034), 'crv_gsm8k': (1319, 8995, 0.073),
               'math_shepherd': (10000, 37514, 0.476), 'prm800k': (8000, 70698, 0.042)}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def answer(uid, bench, source_id, question, steps, generator=None):
    return {'uid': uid, 'benchmark': bench, 'source_id': str(source_id), 'question': question,
            'original_question': question, 'steps': list(steps), 'generator': generator}


def gold(uid, correct, category):
    return {'uid': uid, 'correct': [bool(c) for c in correct], 'include': [True] * len(correct), 'category': category}


# --- CRV (CROP select_target_traces) -------------------------------------------------------------------------
def crv_traces(paths, domain):
    out = []
    for p in paths:
        m = re.search(r'nt(\d+)', p.name)
        for e in json.loads(p.read_text(encoding='utf8')):
            steps = e.get('step_expressions', [])
            if not steps:
                continue
            out.append({'trace_id': f'{domain}:{p.stem}:{e.get("expression_id")}', 'source_id': str(e.get('expression_id')),
                        'question': str(e.get('original_expression', '')), 'complexity': int(m.group(1)) if m else None,
                        'steps': [str(s.get('step_content', '')) for s in steps],
                        'correct': [bool(s.get('step_label')) for s in steps]})
    return out


def sample_group(traces, n, rng):
    if n <= 0 or len(traces) <= n:
        return list(traces)
    idx = rng.choice(len(traces), size=n, replace=False)
    return [traces[int(i)] for i in np.sort(idx)]


def build_crv(src):
    root = src / 'crv'
    arith_files = sorted((root / 'arithmetic_expressions').glob('arith.nt*.annotated.json'))
    rng = np.random.default_rng(CROP_SEED)
    per_file = int(math.ceil(1500 / max(len(arith_files), 1)))
    pool = []
    for p in arith_files:
        pool.extend(sample_group(crv_traces([p], 'arithmetic'), per_file, rng))
    arith = sorted(sample_group(pool, 1500, rng), key=lambda t: t['trace_id'])
    gsm = sorted(crv_traces([root / 'gsm8k_expressions' / 'gsm8k.annotated.json'], 'gsm8k'), key=lambda t: t['trace_id'])
    res = {}
    for bench, tt in (('crv_arith', arith), ('crv_gsm8k', gsm)):
        res[bench] = ([answer(t['trace_id'], bench, t['source_id'], t['question'], t['steps'], 'Llama-3.1-8B-Instruct')
                       for t in tt],
                      [gold(t['trace_id'], t['correct'], f'nt{t["complexity"]}' if t['complexity'] else 'gsm8k') for t in tt])
    srcs = arith_files + [root / 'gsm8k_expressions' / 'gsm8k.annotated.json']
    return res, srcs


# --- Math-Shepherd (CROP _parse_math_shepherd_steps) ---------------------------------------------------------
def parse_math_shepherd(text, label_text):
    matches = list(re.finditer(r'Step\s+\d+\s*:\s*', text))
    if not matches:
        parts = [x.strip() for x in re.split(r'\n\s*(?:Step\s+\d+\s*:)?', text) if x.strip()]
        return parts or [text], set()
    steps, labels = [], []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        steps.append(text[m.end():end].replace('ки', '').strip())
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(label_text)
        seg = label_text[start:end] if start < len(label_text) else ''
        labels.append(not ('*' in seg or ' - ' in seg))
    return steps, {i for i, ok in enumerate(labels[:len(steps)]) if not ok}


def build_math_shepherd(src):
    p = src / 'math_shepherd' / 'math-shepherd.first10000.jsonl'
    ans, gld = [], []
    for idx, line in enumerate(p.read_text(encoding='utf8').splitlines()[:10000]):
        row = json.loads(line)
        text, lab = str(row.get('input', '')), str(row.get('label', ''))
        steps, err = parse_math_shepherd(text, lab)
        if not steps:
            continue
        uid = f'math_shepherd:{idx}'
        ans.append(answer(uid, 'math_shepherd', idx, text.split('Step 1:', 1)[0].strip(), steps))
        gld.append(gold(uid, [i not in err for i in range(len(steps))], str(row.get('task', 'unknown')).lower()))
    return {'math_shepherd': (ans, gld)}, [p]


# --- PRM800K (CROP _prm800k_step / prm800k_traces) -----------------------------------------------------------
def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def prm800k_step(step):
    chosen, comps = step.get('chosen_completion'), step.get('completions') or []
    if chosen is not None:
        try:
            item = comps[int(chosen)]
        except (IndexError, TypeError, ValueError):
            return None
        text, rating = str(item.get('text', '')).strip(), item.get('rating')
    else:
        for item in comps:
            if _int(item.get('rating')) < 0:
                text = str(item.get('text', '')).strip()
                if text:
                    return text, False
        if comps:
            text = str(comps[0].get('text', '')).strip()
            if text:
                return text, _int(comps[0].get('rating')) >= 0
        human = step.get('human_completion')
        if not isinstance(human, dict):
            return None
        text, rating = str(human.get('text', '')).strip(), 1
    if not text:
        return None
    return text, _int(rating) >= 0


def build_prm800k(src):
    paths = [src / 'prm800k' / 'phase2_test.jsonl', src / 'prm800k' / 'phase2_train.first12000.jsonl']
    stems = ['phase2_test', 'phase2_train']   # CROP's trace ids use the original file stems
    ans, gld = [], []
    for p, stem in zip(paths, stems):
        for li, line in enumerate(p.read_text(encoding='utf8').splitlines()):
            if len(ans) >= 8000:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            steps, correct = [], []
            for raw in (row.get('label') or {}).get('steps') or []:
                parsed = prm800k_step(raw)
                if parsed is not None:
                    steps.append(parsed[0]); correct.append(parsed[1])
            if not steps:
                continue
            uid = f'prm800k:{stem}:{li}'
            ans.append(answer(uid, 'prm800k', f'{stem}:{li}', str((row.get('question') or {}).get('problem', '')), steps))
            gld.append(gold(uid, correct, stem))
    if len(ans) < 8000:
        raise SystemExit(f'prm800k: only {len(ans)} traces in the fetched prefix; fetch more phase2_train lines')
    return {'prm800k': (ans, gld)}, paths


# --- DeltaBench (ARES) ---------------------------------------------------------------------------------------
def build_deltabench(src):
    p = src / 'deltabench' / 'Deltabench_v1.csv'
    csv.field_size_limit(1 << 30)
    ans, gld = [], []
    with open(p, encoding='utf8', newline='') as f:
        for row in csv.DictReader(f):
            secs = ast.literal_eval(row['sections'])
            err = set(ast.literal_eval(row['reason_error_section_numbers']))
            nums = [int(s.get('section_number', i + 1)) if 'section_number' in s else i + 1 for i, s in enumerate(secs)]
            steps = [str(s['content']) for s in secs]
            uid = f'deltabench:{row["id"]}'
            ans.append(answer(uid, 'deltabench', row['id'], row['question'], steps, row.get('origin')))
            gld.append(gold(uid, [n not in err for n in nums], row['task_l1']))
    return {'deltabench': (ans, gld)}, [p]


BUILDERS = {'crv': build_crv, 'math_shepherd': build_math_shepherd, 'prm800k': build_prm800k,
            'deltabench': build_deltabench}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gold', type=Path, required=True, help='gold directory, kept apart from the AIRCC inputs')
    ap.add_argument('--only', nargs='*', default=list(BUILDERS))
    ap.add_argument('--allow-count-mismatch', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); a.gold.mkdir(parents=True, exist_ok=True)
    mpath = a.out / 'INPUT_MANIFEST.json'
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {'benchmarks': {}, 'sources': {}}
    failures = []
    for key in a.only:
        built, srcs = BUILDERS[key](a.src)
        for s in srcs:
            manifest['sources'][str(s.relative_to(a.src)).replace('\\', '/')] = sha(s)
        for bench, (ans, gld) in built.items():
            assert len({x['uid'] for x in ans}) == len(ans), bench
            assert all(len(x['steps']) == len(g['correct']) and x['uid'] == g['uid'] for x, g in zip(ans, gld)), bench
            assert all('correct' not in x and 'label' not in x for x in ans), bench
            d = a.out / bench; d.mkdir(exist_ok=True)
            rng = np.random.default_rng(TIMING_SEED)
            timing = [ans[i] for i in sorted(rng.choice(len(ans), min(TIMING_N, len(ans)), replace=False).tolist())]
            for path, obj in ((d / 'answers.json', ans), (d / 'answers_timing12.json', timing), (a.gold / f'{bench}.json', gld)):
                path.write_text(json.dumps(obj, ensure_ascii=False, indent=0), encoding='utf8')
            n_steps = sum(len(g['correct']) for g in gld)
            n_err = sum(not c for g in gld for c in g['correct'])
            cats = {}
            for g in gld:
                c = cats.setdefault(g['category'], [0, 0, 0])
                c[0] += 1; c[1] += len(g['correct']); c[2] += sum(not x for x in g['correct'])
            chars = np.array([len(x['question']) + sum(len(s) for s in x['steps']) for x in ans])
            rec = {'n_answers': len(ans), 'n_steps': n_steps, 'n_error_steps': n_err,
                   'step_error_rate': round(n_err / n_steps, 4), 'answers_with_error': sum(any(not c for c in g['correct']) for g in gld),
                   'empty_steps': sum(not s.strip() for x in ans for s in x['steps']),
                   'by_category': {k: {'answers': v[0], 'steps': v[1], 'error_rate': round(v[2] / v[1], 4)} for k, v in sorted(cats.items())},
                   'chars_median': int(np.median(chars)), 'chars_max': int(chars.max()),
                   'timing12_uids': [x['uid'] for x in timing],
                   'sha256': {'answers.json': sha(d / 'answers.json'), 'answers_timing12.json': sha(d / 'answers_timing12.json'),
                              'gold': sha(a.gold / f'{bench}.json')}}
            if bench in CROP_TABLE4:
                t = CROP_TABLE4[bench]
                ok = rec['n_answers'] == t[0] and rec['n_steps'] == t[1] and abs(rec['step_error_rate'] - t[2]) < 0.0015
                rec['crop_table4'] = {'answers': t[0], 'steps': t[1], 'error_rate': t[2], 'match': ok}
                if not ok:
                    failures.append(bench)
            manifest['benchmarks'][bench] = rec
            print(bench, {k: rec[k] for k in ('n_answers', 'n_steps', 'step_error_rate', 'empty_steps', 'chars_median')},
                  rec.get('crop_table4'))
    manifest['label_rule'] = 'gold correct[i] is True when step i is not annotated as an error; answers.json carries no labels'
    mpath.write_text(json.dumps(manifest, indent=1))
    if failures and not a.allow_count_mismatch:
        raise SystemExit(f'counts differ from CROP Table 4: {failures}')


if __name__ == '__main__':
    main()
