"""Assemble the Stage-1 account tables from reviewed result files only (no recomputation).

Writes results/rbm_literature_completion_v1/STAGE1_COMPARISON_TABLE.{md,csv} and
STAGE1_CONTRASTS.md. Rows are taken from RBM_FUSION_COMPARISON.csv (already hash-bound to their
sources) plus the stability and depth_amended METRICS.json when those suites are COMPLETE with a
PASS review. Missing suites are reported as missing, never filled in.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_rbm_literature_completion as run  # noqa: E402
from scripts.build_rbm_literature_summary import display_name  # noqa: E402

OUT = run.PROGRAM
ROWS = [  # (display_name in RBM_FUSION_COMPARISON.csv, label for the account, classification)
    ('Token entropy', 'Token entropy (retained reference)', 'simple reference'),
    ('Varentropy, top 15 probabilities', 'Varentropy15 raw', 'simple reference'),
    ('Varentropy, top 50 probabilities', 'Varentropy50 raw', 'simple reference'),
    ('Varentropy15 contributions, equal fusion', 'Varentropy15 contributions, equal', 'fixed fusion'),
    ('Varentropy15 contributions, IU-PCR fusion', 'Varentropy15 contributions, IU-PCR', 'learned fusion'),
    ('Step length control', 'Longest-step control', 'simple control'),
    ('Random step control', 'Random-step control', 'simple control'),
    ('RBM6, trained, posterior', 'RBM6 (order-3 bank), posterior', 'learned fusion'),
    ('RBM6, before training, posterior', 'RBM6 before training', 'fixed fusion'),
    ('RBM12, trained, logit', 'RBM12 (order-6 bank), logit', 'learned fusion'),
    ('RBM12, trained, posterior', 'RBM12 (order-6 bank), posterior', 'learned fusion'),
    ('RBM12, before training, posterior', 'RBM12 before training', 'fixed fusion'),
    ('RBM6 with weight shrinkage', 'RBM6 weight shrinkage', 'learned fusion'),
    ('RBM6 with shared diagonal variance, previous experiment', 'RBM6 shared diagonal variance', 'learned fusion'),
    ('Gaussian fusion, shared state variance, 12 features, posterior', 'Two-state shared variance, bank12, posterior', 'learned fusion'),
    ('Gaussian fusion, separate state variances, 12 features, logit', 'Two-state separate variance, bank12, logit', 'learned fusion'),
    ('RBM, 1 hidden unit, CD-10 training, 12 features, posterior', 'CD-10 H1, bank12, posterior', 'learned fusion (fixed epoch budget)'),
    ('RBM, 4 hidden units, exact training, 12 features, logit', 'Exact H4, bank12, logit (maxiter 100)', 'learned fusion (iteration cap)'),
    ('RBM, 4 hidden units, exact training, 6 features, posterior', 'Exact H4, bank6, posterior (maxiter 100)', 'learned fusion (iteration cap)'),
    ('RBM with token sequence fusion across steps, 12 features, logit', 'Token-chain Markov, bank12, logit', 'learned fusion'),
    ('RBM with shuffled token order, control, 12 features, logit', 'Token-chain shuffled control, bank12, logit', 'control'),
    ('RBM12 with chronological-half weight adaptation', 'Position-conditioned RBM12', 'learned fusion'),
    ('DUFS selects six of 12 moment features; trained RBM', 'DUFS-selected 6 of 12, RBM', 'learned selection + fusion'),
    ('Low-correlation selection of six of 12 moment features; trained RBM', 'Low-correlation 6 of 12, RBM', 'selection control + fusion'),
    ('RBM, 48 raw rank-power features, before training', '48 raw rank-power columns, RBM before training', 'fixed fusion'),
    ('RBM, 48 raw rank-power features, trained', '48 raw rank-power columns, RBM trained', 'learned fusion'),
    ('RBM12 with supervised step-BCE coefficient correction', 'Supervised step-BCE correction of RBM12 (labels, other answers)', 'supervised diagnostic'),
]
SUITE_ROWS = {
    'stability': [('b6_best_exact1_posterior', 'Stability: best-of-3 exact H1, bank6, posterior'),
                  ('b6_best_exact4_posterior', 'Stability: best-of-3 exact H4, bank6, posterior'),
                  ('b12_best_exact1_logit', 'Stability: best-of-3 exact H1, bank12, logit'),
                  ('b12_best_exact4_logit', 'Stability: best-of-3 exact H4, bank12, logit')],
    'depth_amended': [('b6_layer2_exact_posterior', 'Depth: exact second layer on H4 posteriors, bank6'),
                      ('b6_layer2_cd_posterior', 'Depth: CD-10 second layer on H4 posteriors, bank6'),
                      ('b6_layer2_logit_exact_posterior', 'Depth: exact second layer on H4 logits, bank6 (amendment)'),
                      ('b6_layer2_logit_cd_posterior', 'Depth: CD-10 second layer on H4 logits, bank6 (amendment)'),
                      ('b12_layer2_exact_logit', 'Depth: exact second layer on H4 posteriors, bank12'),
                      ('b12_layer2_cd_logit', 'Depth: CD-10 second layer on H4 posteriors, bank12'),
                      ('b12_layer2_logit_exact_logit', 'Depth: exact second layer on H4 logits, bank12 (amendment)'),
                      ('b12_layer2_logit_cd_logit', 'Depth: CD-10 second layer on H4 logits, bank12 (amendment)')],
}


def f(x, pct=False, d=4):
    if x in (None, ''):
        return 'n/a'
    x = float(x)
    return f'{100 * x:.{d}f}' if pct else f'{x:.{d + 2}f}'


def main():
    comp = list(csv.DictReader((OUT / 'RBM_FUSION_COMPARISON.csv').open(encoding='utf-8-sig', newline='')))
    by_name = {}
    for r in comp:
        by_name.setdefault(r['display_name'], r)
    table, csv_rows, missing = [], [], []
    for name, label, kind in ROWS:
        r = by_name.get(name)
        if r is None:
            missing.append(name)
            continue
        row = dict(label=label, classification=kind, panel=r['panel'], pb_all8_pct=100 * float(r['pb_all8']),
                   pb_all8_conditional_pct=100 * float(r['pb_all8']), coverage=int(r['valid_answers']) / 13769,
                   prm_within=r['prm_within'], prm_pooled=r['prm_pooled'], prmscore_q08=r['prmscore_q08'],
                   valid_answers=r['valid_answers'], source=r['source_path'], sha256=r['source_sha256'])
        csv_rows.append(row)
    for suite, entries in SUITE_ROWS.items():
        d = OUT / suite
        state = json.loads((d / 'RUN_STATE.json').read_text()) if (d / 'RUN_STATE.json').exists() else {}
        review = json.loads((d / 'RESULT_REVIEW.json').read_text()) if (d / 'RESULT_REVIEW.json').exists() else {}
        if state.get('status') != 'COMPLETE' or review.get('status') != 'PASS':
            missing.append(f'{suite}: status={state.get("status", "NOT_STARTED")} review={review.get("status", "none")}')
            continue
        payload = json.loads((d / 'METRICS.json').read_text())
        metrics, cond = payload['metrics'], payload.get('conditional', {})
        digest = run.base.old.sha256_file(d / 'METRICS.json')
        for m, label in entries:
            if m not in metrics:
                missing.append(f'{suite}:{m}')
                continue
            v = metrics[m]
            c = cond.get(m, {})
            csv_rows.append(dict(label=label, classification='learned fusion', panel='answer_local',
                                 pb_all8_pct=100 * v['pb_all8'],
                                 pb_all8_conditional_pct=100 * (c.get('pb_all8') if c.get('pb_all8') is not None else v['pb_all8']),
                                 coverage=v['valid_answers'] / 13769, prm_within=v['prm_within'], prm_pooled=v['prm_pooled'],
                                 prmscore_q08=v['prmscore_q08'] if v['prmscore_q08'] is not None else f"{v['prmscore_conditional']:.6f} (conditional)",
                                 valid_answers=v['valid_answers'], source=str(d / 'METRICS.json'), sha256=digest))
    lines = ['| Row | Class | PB all-8 % (full population) | PB all-8 % (covered) | Coverage | PRMB within | PRMB pooled | PRMScore | Valid |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in csv_rows:
        lines.append(f"| {r['label']} | {r['classification']} | {r['pb_all8_pct']:.4f} | {r['pb_all8_conditional_pct']:.4f} | "
                     f"{r['coverage']:.4f} | {f(r['prm_within'])} | {f(r['prm_pooled'])} | "
                     f"{r['prmscore_q08'] if isinstance(r['prmscore_q08'], str) and 'conditional' in r['prmscore_q08'] else f(r['prmscore_q08'])} | {r['valid_answers']} |")
    if missing:
        lines += ['', 'Missing rows (not fabricated): ' + '; '.join(missing)]
    (OUT / 'STAGE1_COMPARISON_TABLE.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    run.csv_write(OUT / 'STAGE1_COMPARISON_TABLE.csv', csv_rows)
    # Primary contrasts of the completed suites, verbatim from their METRICS.json
    out = ['| Suite | Contrast | PB delta pp | PB CI | within delta | within CI | gained / lost |', '|---|---|---:|---|---:|---|---:|']
    for suite in ('variance', 'capacity', 'temporal', 'stability', 'depth_amended'):
        d = OUT / suite / 'METRICS.json'
        if not d.exists():
            out.append(f'| {suite} | not available | | | | | |')
            continue
        payload = json.loads(d.read_text())
        for key, c in payload['contrasts'].items():
            if not c.get('primary'):
                continue
            a, b = key.split('_minus_')
            out.append(f"| {suite} | {display_name(a)} − {display_name(b)} | {100 * c['pb_delta']:+.4f} | "
                       f"[{100 * c['pb_ci'][0]:+.4f}, {100 * c['pb_ci'][1]:+.4f}] | "
                       f"{c['prm_within_delta_common']:+.6f} | [{c['prm_within_ci'][0]:+.6f}, {c['prm_within_ci'][1]:+.6f}] | "
                       f"{c['gained']} / {c['lost']} |")
        for key, c in payload.get('conditional_contrasts', {}).items():
            if not c.get('primary'):
                continue
            a, b = key.split('_minus_')
            out.append(f"| {suite} (conditional, common covered answers n={c['common_answers']}) | {display_name(a)} − {display_name(b)} | "
                       f"{(100 * c['pb_delta']) if c['pb_delta'] is not None else float('nan'):+.4f} | "
                       f"[{100 * c['pb_ci'][0]:+.4f}, {100 * c['pb_ci'][1]:+.4f}] | {c['prm_within_delta_common']:+.6f} | "
                       f"[{c['prm_within_ci'][0]:+.6f}, {c['prm_within_ci'][1]:+.6f}] | {c['gained']} / {c['lost']} |")
    (OUT / 'STAGE1_CONTRASTS.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('rows', len(csv_rows), 'missing', missing)


if __name__ == '__main__':
    main()
