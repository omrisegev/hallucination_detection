#!/usr/bin/env python
"""Refactor stage-0 guard: check the active PRMBench line's headline numbers.

Reads configs/current_line_registry.json. Level A (always): every entry's value is read from its
frozen CSV and compared with the pinned number. Level B (report only): lists which raw inputs exist
on this machine, so a later stage-1 re-fit replay knows what it can recompute locally.

    python scripts/verify_current_line.py [--root DIR] [--raw-root DIR]

Exit 0 = all level-A entries match; 1 = a mismatch or a missing frozen file (fails loudly).
"""
import argparse
import csv
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join('configs', 'current_line_registry.json')


def read_value(root, entry):
    path = os.path.join(root, entry['file'])
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f)
                if all(r.get(k) == v for k, v in entry['match'].items())]
    if len(rows) != 1:
        raise ValueError(f"{entry['id']}: {len(rows)} rows match {entry['match']} in {entry['file']}")
    return float(rows[0][entry['column']])


def verify(root=ROOT, raw_root=None):
    with open(os.path.join(root, REGISTRY), encoding='utf-8') as f:
        reg = json.load(f)
    tol = reg['tolerance']
    failures, report = [], []
    for e in reg['entries']:
        try:
            got = read_value(root, e)
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f"{e['id']}: {exc}")
            continue
        ok = math.isclose(got, e['expected'], rel_tol=0, abs_tol=tol)
        report.append((e['id'], e['expected'], got, ok))
        if not ok:
            failures.append(f"{e['id']}: expected {e['expected']!r}, got {got!r}")
    raw = []
    for r in reg.get('raw_inputs', []):
        present = None if r.get('remote') else os.path.exists(os.path.join(raw_root or root, r['path']))
        raw.append((r['path'], present))
    return report, raw, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--raw-root', default=None, help='checkout holding untracked raw inputs')
    a = ap.parse_args()
    report, raw, failures = verify(a.root, a.raw_root)
    for i, exp, got, ok in report:
        print(f"{'PASS' if ok else 'FAIL'}  {i:36s} expected {exp:.10g}  got {got:.10g}")
    for p, present in raw:
        print(f"RAW   {'remote' if present is None else ('present' if present else 'missing'):8s} {p}")
    if failures:
        print('\n'.join(['', 'LEVEL A FAILED:'] + failures))
        return 1
    print(f'\nLEVEL A PASS: {len(report)} entries')
    return 0


if __name__ == '__main__':
    sys.exit(main())
