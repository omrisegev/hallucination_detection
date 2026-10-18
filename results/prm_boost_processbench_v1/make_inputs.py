"""Build the label-free ProcessBench answers.json for the PRM-telemetry driver, and a separate gold file.

Source: the local copy of all 3,400 Qwen/ProcessBench rows carried in the 2026-08-10 Qwen2.5-Math-PRM-7B run
(dataset_cache/four_localization/pb_prm_qwen25math7b_full/pb_prm_<subset>.pkl, AIRCC job 177776). Only the dataset
fields are read (id, generator, problem, steps, label); that run's rewards/predictions are ignored here.
--hf reads the same rows from the Hugging Face cache instead (spectral_utils.processbench.load_processbench).

Outputs (in --out):
  answers.json             [{uid, question, steps, benchmark, source_id}]   -> AIRCC (no label field)
  answers_timing12.json    3 deterministic rows per subset (seed 20261018)   -> AIRCC
  ../local/processbench_gold.json [{uid, subset, label, n_steps, generator}] -> stays local, never to AIRCC
  INPUT_MANIFEST.json      counts, sha256 of every file, ordered-id hash checked against the repo's audit

Label convention (ProcessBench, NOT PRMBench): `label` is the 0-based index of the first erroneous step, -1 when
the solution is correct. Do not pass it through spectral_utils.prm_label_contract (one-based PRMBench contract).

    .venv/bin/python results/prm_boost_processbench_v1/make_inputs.py --out results/prm_boost_processbench_v1/inputs
"""
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spectral_utils.processbench import SUBSETS, NO_ERROR, validate_row  # noqa: E402
from spectral_utils.fair_comparisons.folds import ordered_id_sha256  # noqa: E402
from spectral_utils.fair_comparisons.processbench import canonical_processbench_id  # noqa: E402

DEFAULT_PKL_DIR = Path('/Users/osegev/Desktop/hallucination_detection/dataset_cache/four_localization/pb_prm_qwen25math7b_full')
EXPECTED = {'gsm8k': 400, 'math': 1000, 'olympiadbench': 1000, 'omnimath': 1000}
# results/fair_paper_exact_comparisons_v1/PROCESSBENCH_SOURCE_AUDIT.json (population processbench@e8024636bcab::official-3400)
AUDIT_ORDERED_ID_SHA256 = '7a7edc7a6e4a67ac16968c915900805620fc86314368105afe05a4d1ffd20e10'
TIMING_SEED, TIMING_PER_SUBSET = 20261018, 3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rows(pkl_dir, use_hf):
    rows, sources = {}, {}
    for s in SUBSETS:
        if use_hf:
            from spectral_utils.processbench import load_processbench
            rr = load_processbench(s)
        else:
            p = pkl_dir / f'pb_prm_{s}.pkl'
            d = pickle.load(open(p, 'rb')); sources[s] = {'path': str(p), 'sha256': sha(p)}
            rr = [d[k] for k in sorted(d)]
        rows[s] = [{k: r[k] for k in ('id', 'generator', 'problem', 'steps', 'label')} for r in rr]
    return rows, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gold-out', type=Path, default=None, help='gold directory (default: <out>/../local)')
    ap.add_argument('--pkl-dir', type=Path, default=DEFAULT_PKL_DIR)
    ap.add_argument('--hf', action='store_true', help='read Qwen/ProcessBench from the HF cache instead of the pkls')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    a.gold_out = a.gold_out or a.out.parent / 'local'; a.gold_out.mkdir(parents=True, exist_ok=True)
    rows, sources = load_rows(a.pkl_dir, a.hf)
    answers, gold, ordered = [], [], []
    for s in SUBSETS:
        rr = rows[s]
        if len(rr) != EXPECTED[s]: raise SystemExit(f'{s}: {len(rr)} rows, expected {EXPECTED[s]}')
        ids = [r['id'] for r in rr]
        if ids != [f'{s}-{i}' for i in range(EXPECTED[s])]: raise SystemExit(f'{s}: ids are not {s}-0..{s}-{EXPECTED[s]-1}')
        for r in rr:
            validate_row(r)
            if any(not st.strip() for st in r['steps']):
                print('note: whitespace-only step in', r['id'])
            answers.append({'uid': r['id'], 'question': r['problem'], 'steps': list(r['steps']),
                            'benchmark': s, 'source_id': r['id']})
            gold.append({'uid': r['id'], 'subset': s, 'label': int(r['label']), 'n_steps': len(r['steps']),
                         'generator': r['generator']})
            ordered.append(canonical_processbench_id(s, r['id']))
    order_hash = ordered_id_sha256(ordered)
    if order_hash != AUDIT_ORDERED_ID_SHA256:
        raise SystemExit(f'ordered-id hash {order_hash} != audit {AUDIT_ORDERED_ID_SHA256}')
    assert len({x['uid'] for x in answers}) == len(answers) == 3400
    assert all('label' not in x for x in answers)
    rng = np.random.default_rng(TIMING_SEED); timing = []
    for s in SUBSETS:
        sub = [x for x in answers if x['benchmark'] == s]
        timing += [sub[i] for i in sorted(rng.choice(len(sub), TIMING_PER_SUBSET, replace=False).tolist())]
    files = {'answers.json': (a.out, answers), 'answers_timing12.json': (a.out, timing),
             'processbench_gold.json': (a.gold_out, gold)}
    for name, (d, obj) in files.items():
        (d / name).write_text(json.dumps(obj, ensure_ascii=False, indent=0), encoding='utf8')

    def chars(xs): return float(np.mean([len(x['question']) + sum(len(t) for t in x['steps']) for x in xs]))
    lab = np.array([g['label'] for g in gold]); nst = np.array([g['n_steps'] for g in gold])
    sub = np.array([g['subset'] for g in gold])
    per = {s: {'n_answers': int((sub == s).sum()), 'n_error': int(((sub == s) & (lab != NO_ERROR)).sum()),
               'n_correct': int(((sub == s) & (lab == NO_ERROR)).sum()), 'n_steps': int(nst[sub == s].sum())}
           for s in SUBSETS}
    err = lab != NO_ERROR
    manifest = {
        'source': 'local pkls of AIRCC job 177776 (dataset fields only)' if not a.hf else 'Qwen/ProcessBench via HF cache',
        'source_files': sources, 'population_id': 'processbench@e8024636bcab::official-3400',
        'ordered_id_sha256': order_hash, 'ordered_id_sha256_matches_audit': True,
        'label_convention': '0-based first erroneous step, -1 = correct (ProcessBench)',
        'n_answers': len(answers), 'n_steps_total': int(nst.sum()), 'per_subset': per,
        'steps_labelled_prefix_domain': int(nst[~err].sum() + (lab[err] + 1).sum()),
        'n_first_error_steps': int(err.sum()),
        'n_answers_label0': int((lab == 0).sum()),
        'timing12_uids': [x['uid'] for x in timing], 'timing_seed': TIMING_SEED,
        'mean_chars_per_answer': {'all': chars(answers), 'timing12': chars(timing),
                                  **{s: chars([x for x in answers if x['benchmark'] == s]) for s in SUBSETS}},
        'sha256': {n: sha(d / n) for n, (d, _) in files.items()}, 'gold_dir': 'local/ (never synced to AIRCC)',
        'labels_in_answers_json': False}
    (a.out / 'INPUT_MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ('n_answers', 'n_steps_total', 'per_subset', 'sha256')}, indent=1))


if __name__ == '__main__':
    main()
