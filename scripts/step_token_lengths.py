"""Per-step token spans (answer-local [lo, hi)) for the 13769 JOINED answers, from the OLD raw cells.

Replaces TOKEN_MATRICES.npz['step_spans'] (no longer on disk) for monday dev_run's STEP_LENGTH
reference and its 'longest step' ProcessBench strata. Step text is identical across the old,
thinkfix and mask1 cells, so one array serves all variants (it describes the text, not the scorer).
    python scripts/step_token_lengths.py --results <results> --out <file.npz>
"""
import argparse, hashlib, json, pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json'
SUBSETS = ('gsm8k', 'math', 'olympiadbench', 'omnimath')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', choices=['old', 'thinkfix', 'thinkfix_mask1'], default='old')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('exists: '+str(a.out))
    roster = json.loads(ROSTER.read_text())['records']
    off = np.r_[0, np.cumsum([r['steps'] for r in roster])]
    spans = np.full((off[-1], 2), -1, np.int64); seen = np.zeros(len(roster), bool); src = {}
    s = {'old': '', 'thinkfix': '_thinkfix', 'thinkfix_mask1': '_thinkfix_mask1'}[a.variant]
    shift = 1 if a.variant == 'thinkfix_mask1' else 0
    cells = [(f'pb_{d}_q{m[0]}', a.results/f'pb_qwen3_{m}{s}'/f'processbench_{d}.pkl', d) for m in ('4b', '8b') for d in SUBSETS]
    prm = 'prmbench_qwen3_8b_telemetry_full' if a.variant == 'old' else 'prmbench_qwen3_8b_telemetry'+s
    cells.append(('prmbench_qwen3_8b', a.results/prm/'prmbench_telemetry.pkl', None))
    for cell, path, d in cells:
        src[cell] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        payload = pickle.load(open(path, 'rb')); rows = list(payload.values()) if isinstance(payload, dict) else payload
        lookup = {(f'{d}::{r.get("id")}' if d else str(r.get('idx'))): r for r in rows if isinstance(r, dict)}
        for i, r in enumerate(roster):
            if r['cell'] != cell:
                continue
            row = lookup[str(r['row_id'])]; sp = row['step_token_spans']
            assert len(sp) == r['steps'] and len(row['gen_token_ids']) == r['tokens'] - shift, r['uid']
            spans[off[i]:off[i+1]] = [(-1, -1) if s is None else (int(s[0]), int(s[1])) for s in sp]
            seen[i] = True
        print('CELL', cell, flush=True)
    assert seen.all()
    np.savez_compressed(a.out, step_spans=spans, offsets=off, uids=np.asarray([r['uid'] for r in roster]))
    a.out.with_suffix('.json').write_text(json.dumps({'variant': a.variant, 'sources': src, 'roster_sha256': hashlib.sha256(ROSTER.read_bytes()).hexdigest(),
        'unmapped_steps': int((spans[:, 0] < 0).sum()), 'note': 'answer-local half-open token spans from old raw step_token_spans'}, indent=1))
    print('DONE', spans.shape, 'unmapped', int((spans[:, 0] < 0).sum()), flush=True)


if __name__ == '__main__':
    main()
