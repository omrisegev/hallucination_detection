"""Correction 2026-09-13: mutually exclusive loss categories for the depth_amended suite.

Confirmed defect (Codex review, verified by Claude): in run_rbm_depth_amended.evaluate the
categories lost_early / lost_late / lost_gate / lost_failure overlap, because an invalid (failed)
answer has peak = -1 and satisfies peak < target, so it is counted as both "early" and "failure".
In the original suite driver the same rule exists but every suite had full coverage, so no
overlap occurred there (verified below).

This script recomputes the categories from the SAVED step scores and predictions only (no refit,
no metric change), with mutually exclusive definitions:
  failure : lost and the arm's step scores are not all finite (declared failure)
  gate    : lost, valid, peak == target, prediction == -1 (correct peak suppressed by the gate)
  early   : lost, valid, peak < target
  late    : lost, valid, peak > target
  other   : lost, valid, peak == target, prediction != -1  (should be empty; asserted)
and asserts failure + gate + early + late + other == lost for every comparison. It also re-derives
the PB macro from the saved predictions and asserts equality with METRICS.json (point scores are
unchanged by this bookkeeping correction). Outputs LOSS_BREAKDOWN_CORRECTION_20260913.{json,csv}.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_rbm_literature_completion as run  # noqa: E402
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402

SUITES = ('variance', 'capacity', 'temporal', 'stability', 'depth_amended')


def categories(records, joined, S, a, b):
    off, target = joined['offsets'], joined['target']
    cells = np.array([r['cell'] for r in records])
    pb = np.char.startswith(cells, 'pb_')
    error = pb & (target >= 0)

    def peaks(name):
        flat = S['steps__' + name]
        pk = np.full(len(records), -1)
        valid = np.zeros(len(records), bool)
        for i in range(len(records)):
            s = flat[off[i]:off[i + 1]]
            if len(s) and np.isfinite(s).all():
                valid[i] = True
                pk[i] = int(np.argmax(s))
        return pk, valid

    pa, va = peaks(a)
    pred_a, pred_b = S['prediction__' + a], S['prediction__' + b]
    dva, dvb = S['valid__' + a], S['valid__' + b]
    new = error & dva & (pred_a == target)
    old = error & dvb & (pred_b == target)
    loss = old & ~new
    gain = new & ~old
    out = dict(gained=int(gain.sum()), lost=int(loss.sum()),
               lost_failure=int(np.sum(loss & ~va)),
               lost_gate=int(np.sum(loss & va & (pa == target) & (pred_a == -1))),
               lost_early=int(np.sum(loss & va & (pa < target))),
               lost_late=int(np.sum(loss & va & (pa > target))),
               lost_other=int(np.sum(loss & va & (pa == target) & (pred_a != -1))),
               saved_rule_lost_early=int(np.sum(loss & (pa < target))))
    assert out['lost_failure'] + out['lost_gate'] + out['lost_early'] + out['lost_late'] + out['lost_other'] == out['lost']
    assert out['lost_other'] == 0, (a, b, out)
    return out, dict(pb=pb, target=target, cells=cells)


def main():
    source = ROOT.parents[1]
    records, joined, _ = run.load_contract(source)
    rows, summary = [], {}
    for suite in SUITES:
        d = run.PROGRAM / suite
        if not (d / 'METRICS.json').exists():
            continue
        m = json.load((d / 'METRICS.json').open())
        S = np.load(d / 'SCORES.npz')
        overlap = 0
        for key, c in m['contrasts'].items():
            a, b = key.split('_minus_')
            out, ctx = categories(records, joined, S, a, b)
            assert out['lost'] == c['lost'] and out['gained'] == c['gained'], (suite, key)
            changed = out['saved_rule_lost_early'] != out['lost_early']
            overlap += changed
            rows.append(dict(suite=suite, comparison=key, saved_lost_early=c.get('lost_early'), saved_lost_late=c.get('lost_late'),
                             saved_lost_gate=c.get('lost_gate'), saved_lost_failure=c.get('lost_failure'), **out,
                             overlap_in_saved=changed))
        # point metrics unchanged: re-derive PB macro from saved predictions for every method
        checked = 0
        for name, v in m['metrics'].items():
            pb, target, cells = ctx['pb'], ctx['target'], ctx['cells']
            res = pb_metrics(target[pb], S['prediction__' + name][pb], S['valid__' + name][pb], cells[pb])
            np.testing.assert_allclose(res['macros']['all'], v['pb_all8'], atol=1e-12)
            checked += 1
        summary[suite] = dict(comparisons=len(m['contrasts']), comparisons_with_overlap=overlap, methods_pb_macro_reverified=checked)
        print('[breakdown]', suite, summary[suite], flush=True)
    out = run.PROGRAM / 'depth_amended'
    run.csv_write(out / 'LOSS_BREAKDOWN_CORRECTION_20260913.csv', rows)
    run.base.atomic_json(out / 'LOSS_BREAKDOWN_CORRECTION_20260913.json', dict(
        correction='2026-09-13 mutually exclusive loss categories (failure / gate / early / late); saved rule counted failed answers (peak=-1) as early',
        definitions=dict(failure='lost and arm scores not all finite', gate='lost, valid, peak==target, prediction==-1',
                         early='lost, valid, peak<target', late='lost, valid, peak>target', other='lost, valid, peak==target, prediction!=-1 (asserted empty)'),
        point_scores_changed=False, summary=summary, rows=rows))
    print('[breakdown] written; suites', summary)


if __name__ == '__main__':
    main()
