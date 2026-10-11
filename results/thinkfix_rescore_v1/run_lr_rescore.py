"""thinkfix_rescore_v1: re-run the supervised LR reference (monday_lr_reference_v1) and its weight analysis
(monday_lr_weights_v1) on one variant's rebuilt bank. Run after run_prmbench_rescore.py for that variant.

    python results/thinkfix_rescore_v1/run_lr_rescore.py --variant old|thinkfix|thinkfix_mask1 --bank <dir> --root <out root>

Changes relative to the frozen scripts (patched copies + DIFF.txt are written beside the outputs):
  * ext.FE -> --bank; OUT -> <root>/<variant>/{lr_reference,lr_weights};
  * monday_lr_reference_v1: V1 and V2 (the panel sources it reads PRMBench/ProcessBench method numbers, per-answer
    arrays and fit diagnostics from) -> this variant's nested_banks run, so no frozen number enters a new variant;
    ORIENTED_CONTROL.csv for the variant is derived from its cont_baselines ABSOLUTE.csv (ALL_EQ_ORIENTED rows);
  * monday_lr_weights_v1: REF -> this variant's lr_reference; the N16 'OOF within-AUC = 0.822366' assert stays a
    HARD gate for variant=old and is recorded (not enforced) for the think-fix variants; the cached-fold replay
    check (1e-9, within the same variant) stays hard for every variant;
  * figure stages are not run.
"""
from __future__ import annotations
import argparse, difflib, importlib.util, json, sys, time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; MAIN = HERE.parents[1]
LRR = MAIN / 'results/monday_lr_reference_v1'; LRW = MAIN / 'results/monday_lr_weights_v1'


def patch(src, old, new):
    if src.count(old) != 1:
        raise SystemExit(f'patch anchor found {src.count(old)} times: {old[:90]!r}')
    return src.replace(old, new)


def write(src_path, dst, text, header):
    dst.write_text(header + text, encoding='utf8')
    d = difflib.unified_diff(src_path.read_text(encoding='utf8').splitlines(True), dst.read_text(encoding='utf8').splitlines(True),
                             fromfile=str(src_path), tofile=str(dst))
    (dst.parent / 'DIFF.txt').write_text(''.join(d), encoding='utf8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['old', 'thinkfix', 'thinkfix_mask1'], required=True)
    ap.add_argument('--bank', type=Path, required=True); ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args(); t0 = time.time()
    vroot = a.root / a.variant; nested = vroot / 'nested_banks'; cont = vroot / 'cont_baselines'
    for need in (nested / 'PER_ANSWER.npz', nested / 'PANEL_A_ABSOLUTE.csv', nested / 'BANKS.json', cont / 'ABSOLUTE.csv'):
        if not need.exists(): raise SystemExit('run run_prmbench_rescore.py first; missing ' + str(need))
    oc = nested / 'ORIENTED_CONTROL.csv'
    if not oc.exists():
        ab = pd.read_csv(cont / 'ABSOLUTE.csv'); ab = ab[ab.scorer == 'ALL_EQ_ORIENTED']
        ab[['bank', 'scorer', 'prmb_within_auc', 'pb_argmax_hit', 'prmb_top1_on_error']].to_csv(oc, index=False)
        (nested / 'ORIENTED_CONTROL.README.txt').write_text('Derived by run_lr_rescore.py from ../cont_baselines/ABSOLUTE.csv '
            '(ALL_EQ_ORIENTED rows) for monday_lr_reference_v1.panel_values(); the frozen one came from supplement_a2.py.\n')
    out_r = vroot / 'lr_reference'; out_w = vroot / 'lr_weights'
    out_r.mkdir(parents=True, exist_ok=False); out_w.mkdir(parents=True, exist_ok=False)
    for src, dst in ((LRR / 'PROTOCOL.json', out_r / 'PROTOCOL.json'), (LRW / 'PROTOCOL.json', out_w / 'PROTOCOL.json')):
        dst.write_bytes(src.read_bytes())
    s = (LRR / 'run.py').read_text(encoding='utf8')
    s = patch(s, "V1 = MAIN / 'results/monday_nested_banks_v1'; V2 = MAIN / 'results/monday_nested_banks_v2'",
              f"V1 = Path(r'{nested}'); V2 = Path(r'{nested}')")
    write(LRR / 'run.py', out_r / 'run_patched.py', s,
          f'# PATCHED COPY of results/monday_lr_reference_v1/run.py (thinkfix_rescore_v1, variant={a.variant}): '
          'V1/V2 -> this variant\'s nested_banks; ext.FE set by the wrapper.\n')
    w = (LRW / 'run.py').read_text(encoding='utf8')
    w = patch(w, "REF = MAIN / 'results/monday_lr_reference_v1'", f"REF = Path(r'{out_r}')")
    w = patch(w, "_spec = importlib.util.spec_from_file_location('lr_ref', REF / 'run.py')",
              "_spec = importlib.util.spec_from_file_location('lr_ref', REF / 'run_patched.py')")
    w = patch(w, "    if abs(auc - 0.822366) > 1e-6: raise SystemExit('GATE FAIL: N16 AUC not reproduced')\n",
              "    out['N16_refit']['enforced'] = _ENFORCE_N16\n"
              "    if abs(auc - 0.822366) > 1e-6 and _ENFORCE_N16: raise SystemExit('GATE FAIL: N16 AUC not reproduced')\n")
    write(LRW / 'run.py', out_w / 'run_patched.py', w,
          f'# PATCHED COPY of results/monday_lr_weights_v1/run.py (thinkfix_rescore_v1, variant={a.variant}): '
          'REF -> this variant\'s lr_reference; N16 0.822366 gate enforced only for variant=old.\n')
    spec = importlib.util.spec_from_file_location('lr_w', out_w / 'run_patched.py'); W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    W._ENFORCE_N16 = a.variant == 'old'; R = W.ref; R.ext.FE = a.bank
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        R.fit_stage(); R.scores_stage(); R.aggregate_stage()
        print('lr_reference done', round(time.time() - t0), flush=True)
        W.gate_stage(); [W.bank_stage(b) for b in W.BANKS]; W.aggregate_stage()
    (vroot / 'RESCORE_STATUS_lr.json').write_text(json.dumps({'variant': a.variant, 'bank': str(a.bank),
        'runtime_s': round(time.time() - t0), 'n16_gate_enforced': a.variant == 'old'}, indent=1))
    print('LR RESCORE DONE', a.variant, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
