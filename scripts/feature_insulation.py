#!/usr/bin/env python3
"""
Feature Insulation Analysis (Gemini R4)
========================================

Question: does L-SML's grouping insulate the fused score from *volatile* features
(features that are strong in one domain and near-random in another), compared to a
flat average that cannot insulate?

Reads the artifacts produced by method_comparison.py (no model, no GPU):
  results/method_comparison_table1.csv        per-cell AUROC for every variant
  results/method_comparison_table2.csv        per-group feature lists + vAUROC
  results/method_comparison_table4_feat_aurocs.csv  per-feature per-cell AUROC

Produces (stdout only — nothing written, for review):
  1. Feature volatility ranking (cross-domain AUROC range).
  2. Method robustness: cross-domain std of each method's domain means.
     If CONT's domain-to-domain swing is smaller than avg5's, grouping insulates.
  3. spectral_entropy case study: solo AUROC vs the vAUROC of the group it lands in,
     per cell — does grouping rescue it, dilute a strong group, or isolate it?
  4. Isolation pattern: how often each feature is placed in a size-1 group (isolated)
     vs merged — an insulating algorithm should isolate the volatile features.
"""

import csv
import os
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
T1 = os.path.join(RESULTS, 'method_comparison_table1.csv')
T2 = os.path.join(RESULTS, 'method_comparison_table2.csv')
T4 = os.path.join(RESULTS, 'method_comparison_table4_feat_aurocs.csv')

DOMAINS = ['math500', 'gsm8k', 'gpqa', 'rag', 'qa']
GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']


def pct(s):
    """'84.3%' -> 0.843 ; '' -> None."""
    s = (s or '').strip().rstrip('%')
    if not s:
        return None
    try:
        return float(s) / 100.0
    except ValueError:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


# ---------------------------------------------------------------------------
# Load table4 — per-feature per-cell AUROC
# ---------------------------------------------------------------------------
def load_feat_aurocs():
    # feat -> domain -> [auroc per cell]
    data = defaultdict(lambda: defaultdict(list))
    with open(T4, newline='') as f:
        for row in csv.DictReader(f):
            a = pct(row['auroc'])
            if a is not None:
                data[row['feature']][row['domain']].append(a)
    return data


# ---------------------------------------------------------------------------
# Load table1 — per-cell AUROC per variant (skip aggregate rows)
# ---------------------------------------------------------------------------
def load_variant_aurocs():
    # variant -> domain -> [auroc per cell]
    data = defaultdict(lambda: defaultdict(list))
    with open(T1, newline='') as f:
        reader = csv.DictReader(f)
        variant_cols = [c for c in reader.fieldnames
                        if c not in ('domain', 'cell_key', 'n', 'n_pos', 'saturated_feats')]
        for row in reader:
            dom = row['domain']
            ck = row['cell_key']
            if not ck or 'MEAN' in ck or dom == 'MACRO_AVG' or ck == 'DOMAIN_MEAN':
                continue
            for v in variant_cols:
                a = pct(row.get(v))
                if a is not None:
                    data[v][dom].append(a)
    return data, variant_cols


# ---------------------------------------------------------------------------
# Load table2 — group membership per (cell, variant)
# ---------------------------------------------------------------------------
def load_groups():
    # list of dicts: {domain, cell_key, variant, feature_names:[...], size, vAUROC_cont}
    rows = []
    with open(T2, newline='') as f:
        for row in csv.DictReader(f):
            names = [x for x in (row['feature_names'] or '').split(',') if x]
            rows.append({
                'domain': row['domain'],
                'cell_key': row['cell_key'],
                'variant': row['variant'],
                'feature_names': names,
                'size': int(row['size']),
                'vAUROC_cont': pct(row['vAUROC_cont']),
                'vAUROC_bin': pct(row['vAUROC_bin']),
            })
    return rows


def domain_means(by_domain):
    """dict domain->list -> dict domain->mean (only DOMAINS present)."""
    return {d: mean(by_domain[d]) for d in DOMAINS if by_domain.get(d)}


def macro(by_domain):
    dm = domain_means(by_domain)
    return mean(list(dm.values()))


def main():
    feat = load_feat_aurocs()
    variants, vcols = load_variant_aurocs()
    groups = load_groups()

    # ---- 1. Feature volatility -------------------------------------------
    print('=' * 78)
    print('1. FEATURE VOLATILITY  (per-feature AUROC, oriented; domain = mean of cells)')
    print('=' * 78)
    print(f'{"feature":<22} {"macro":>6} {"min_dom":>8} {"max_dom":>8} {"range":>7}  most volatile->')
    vol = {}
    rows = []
    for fname, by_dom in feat.items():
        dm = domain_means(by_dom)
        if len(dm) < 2:
            continue
        lo, hi = min(dm.values()), max(dm.values())
        rng = hi - lo
        vol[fname] = rng
        rows.append((fname, macro(by_dom), lo, hi, rng,
                     min(dm, key=dm.get), max(dm, key=dm.get)))
    for fname, mac, lo, hi, rng, dlo, dhi in sorted(rows, key=lambda r: -r[4]):
        star = ' GOOD_5' if fname in GOOD_5 else ''
        print(f'{fname:<22} {mac*100:>5.1f}% {lo*100:>7.1f}% {hi*100:>7.1f}% {rng*100:>6.1f}pp'
              f'  {dlo}->{dhi}{star}')

    # ---- 2. Method robustness (cross-domain stability) -------------------
    print()
    print('=' * 78)
    print('2. METHOD ROBUSTNESS  (does grouping insulate vs flat average?)')
    print('=' * 78)
    print('Lower cross-domain std = more robust = better insulated from per-domain swings.')
    print(f'{"method":<26} {"macro":>6} {"min_dom":>8} {"max_dom":>8} {"x-dom std":>9}')
    # table1 columns use SHORT variant names
    focus = ['CONT', 'lsml16c', 'avg5', 'lsml16', 'PROD', 'best1', 'flat5']
    # also synthesise "best fixed single feature" macro from table4
    fixed_feat_macro = {fn: macro(by) for fn, by in feat.items()}
    best_fixed = max(fixed_feat_macro, key=lambda k: fixed_feat_macro[k])
    for v in focus:
        if v not in variants:
            continue
        dm = domain_means(variants[v])
        if not dm:
            continue
        print(f'{v:<26} {macro(variants[v])*100:>5.1f}% {min(dm.values())*100:>7.1f}%'
              f' {max(dm.values())*100:>7.1f}% {std(list(dm.values()))*100:>8.1f}pp')
    # best fixed feature row (cross-domain std of that one feature)
    bfd = domain_means(feat[best_fixed])
    print(f'{"best_fixed:"+best_fixed:<26} {fixed_feat_macro[best_fixed]*100:>5.1f}%'
          f' {min(bfd.values())*100:>7.1f}% {max(bfd.values())*100:>7.1f}%'
          f' {std(list(bfd.values()))*100:>8.1f}pp')

    # ---- 3. spectral_entropy case study ----------------------------------
    print()
    print('=' * 78)
    print('3. spectral_entropy CASE STUDY  (the volatile GOOD_5 feature)')
    print('=' * 78)
    print('For each cell: its solo AUROC, and the vAUROC_cont of the L-SML group it')
    print('lands in. If group vAUROC >= solo, grouping rescued it; if the group it joins')
    print('has HIGHER vAUROC than spectral_entropy solo, it was carried (not diluting).')
    for target_variant in ('lsml_5_signs_continuous', 'lsml_16_continuous'):
        print(f'\n--- variant: {target_variant} ---')
        print(f'{"domain/cell":<48} {"solo":>6} {"grp_size":>8} {"grp_vAUROC":>10} {"verdict":>10}')
        for gr in groups:
            if gr['variant'] != target_variant:
                continue
            if 'spectral_entropy' not in gr['feature_names']:
                continue
            # solo AUROC for spectral_entropy in this cell (per-cell lookup from table4)
            solo = SOLO_LOOKUP.get((gr['domain'], gr['cell_key'], 'spectral_entropy'))
            grpv = gr['vAUROC_cont']
            if solo is None or grpv is None:
                verdict = '?'
            elif grpv >= solo + 0.02:
                verdict = 'rescued'
            elif grpv <= solo - 0.02:
                verdict = 'diluted'
            else:
                verdict = 'neutral'
            label = f"{gr['domain']}/{gr['cell_key']}"[:46]
            s = f'{solo*100:.1f}%' if solo is not None else '  --'
            g = f'{grpv*100:.1f}%' if grpv is not None else '  --'
            print(f'{label:<48} {s:>6} {gr["size"]:>8} {g:>10} {verdict:>10}')

    # ---- 4. Isolation pattern --------------------------------------------
    print()
    print('=' * 78)
    print('4. ISOLATION PATTERN  (insulating algorithm should isolate volatile features)')
    print('=' * 78)
    print('Per feature, across all cells of the CONT variant: how often is it ALONE in')
    print('its group (size-1, isolated) vs merged with others?')
    for target_variant in ('lsml_5_signs_continuous', 'lsml_16_continuous'):
        iso = defaultdict(lambda: [0, 0])  # feat -> [isolated, merged]
        for gr in groups:
            if gr['variant'] != target_variant:
                continue
            for fn in gr['feature_names']:
                if gr['size'] == 1:
                    iso[fn][0] += 1
                else:
                    iso[fn][1] += 1
        print(f'\n--- variant: {target_variant} ---')
        print(f'{"feature":<22} {"isolated":>9} {"merged":>7} {"%isolated":>10} {"volatility":>11}')
        for fn in sorted(iso, key=lambda k: -vol.get(k, 0)):
            i, m = iso[fn]
            tot = i + m
            pctiso = 100 * i / tot if tot else 0
            vstr = f'{vol.get(fn, 0)*100:.1f}pp' if fn in vol else '   --'
            print(f'{fn:<22} {i:>9} {m:>7} {pctiso:>9.0f}% {vstr:>11}')


# Build a per-cell solo-AUROC lookup from table4 once (module-level).
def _build_solo_lookup():
    lut = {}
    with open(T4, newline='') as f:
        for row in csv.DictReader(f):
            a = pct(row['auroc'])
            if a is not None:
                lut[(row['domain'], row['cell_key'], row['feature'])] = a
    return lut


SOLO_LOOKUP = _build_solo_lookup()

if __name__ == '__main__':
    main()
