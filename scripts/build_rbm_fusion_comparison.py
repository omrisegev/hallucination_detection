"""Readable cross-experiment RBM comparison from reviewed, saved results only.

No scoring, model fitting, threshold changes or performance-based row selection.
The prior raw ledger remains exhaustive; this roster removes repeated aliases
while keeping the distinct RBM experiments and their controls visible.
"""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_rbm_literature_completion as run


def roster():
    rows = []

    def add(experiment, method, name, panel='answer_local',
            fit='unlabeled fitting within the current answer',
            readout='posterior; Top10 token mean; earliest maximum'):
        rows.append(dict(experiment=experiment, method=method, display_name=name,
                         panel=panel, fusion_fit=fit, score_readout=readout))

    logit = 'logit; Top10 token mean; earliest maximum'
    for degree in (3, 4, 5, 6):
        bank = 2 * degree
        for method, label in [('rbm', 'RBM'), ('rbm_initial', 'RBM before training'),
                              ('iu', 'IU-PCR'), ('equal', 'Equal fusion')]:
            add('higher-moment-fusion-v1', f'd{degree}__{method}',
                f'{label}, {bank} features, moments through order {degree}',
                fit=('fixed coefficients; no fitting' if method in ('rbm_initial', 'equal')
                     else 'unlabeled fitting within the current answer'),
                readout=('posterior; Top10 token mean; earliest maximum' if method.startswith('rbm')
                         else 'continuous fusion score; Top10 token mean; earliest maximum'))

    for key, label in [('b3', 'B3, six moment features'),
                       ('b3_initial', 'B3 before training, six moment features')]:
        add('moment-rbm-fusion-v1', key, label,
            fit='fixed coefficients; no fitting' if key.endswith('initial')
            else 'unlabeled fitting within the current answer',
            readout='B3 score; Top10 token mean; earliest maximum')

    for key, label in [
        ('mom5_rbm', 'RBM, five features, distribution third moment removed'),
        ('mom6_rbm', 'RBM, six features, third-moment ablation reference'),
        ('mom6_shared_rbm', 'RBM, six features, common five-feature orientation control'),
        ('power48_rbm', 'RBM, 48 raw rank-power features, trained'),
        ('power48_initial', 'RBM, 48 raw rank-power features, before training')]:
        add('rbm-m3-powers-v1', key, label,
            fit='fixed coefficients; no fitting' if key.endswith('initial')
            else 'unlabeled fitting within the current answer')

    add('rbm-weight-shrinkage-v1', 'rbm_shrinkage', 'RBM6 with weight shrinkage')
    add('rbm-diagonal-variance-v1', 'rbm_diagonal',
        'RBM6 with shared diagonal variance, previous experiment')
    add('rbm-diagonal-variance-v1', 'rbm_continued',
        'RBM6 with fixed variance, matched continued-training control')

    for bank in (6, 12):
        for initial in (False, True):
            stem = f'initial{bank}' if initial else f'rbm{bank}'
            name = f'RBM{bank}, ' + ('before training' if initial else 'trained')
            fit = ('fixed coefficients; no fitting' if initial
                   else 'unlabeled fitting within the current answer')
            for score in ('posterior', 'logit'):
                key = stem + ('__logit' if score == 'logit' else '_')
                for near in (False, True):
                    method = key + ('_near' if near else '_old')
                    readout = (f'{score}; Top10 token mean; first_near_max score transform'
                               if near else f'{score}; Top10 token mean; earliest maximum')
                    add('rbm-logit-readout-v1', method,
                        name + f', {score}' + (', first_near_max diagnostic' if near else ''),
                        panel='readout_diagnostic' if near else 'answer_local',
                        fit=fit, readout=readout)

    for stem, label, learned in [
        ('entropy', 'Token entropy', False), ('var15', 'Varentropy, top 15 probabilities', False),
        ('var50', 'Varentropy, top 50 probabilities', False),
        ('var15_equal', 'Varentropy15 contributions, equal fusion', False),
        ('var15_iu', 'Varentropy15 contributions, IU-PCR fusion', True),
        ('shrinkage', 'RBM6 weight shrinkage', True),
        ('diagonal', 'RBM6 shared diagonal variance, previous experiment', True)]:
        # Ordinary shrinkage/diagonal rows above retain their original source.
        for near in ((True,) if stem in ('shrinkage', 'diagonal') else (False, True)):
            add('rbm-logit-readout-v1', f'{stem}__' + ('near' if near else 'old'),
                label + (', first_near_max diagnostic' if near else ''),
                panel='readout_diagnostic' if near else 'answer_local',
                fit='unlabeled fitting within the current answer' if learned
                else 'fixed formula; no fitting',
                readout='saved token score; Top10 token mean; ' +
                        ('first_near_max score transform' if near else 'earliest maximum'))
    for key, label in [('length__old', 'Step length control'), ('random__old', 'Random step control')]:
        add('rbm-logit-readout-v1', key, label, panel='simple_control',
            fit='no fitting', readout='original control definition, unchanged')

    for key, label in [('position__max', 'RBM12 with chronological-half weight adaptation'),
                       ('permuted__max', 'RBM12 with shuffled-step-group weight adaptation'),
                       ('shared__max', 'RBM12 with shared weight adaptation')]:
        add('rbm-position-fusion-v1', key, label, readout=logit)
    for key, label, fit in [
        ('unsupervised_update', 'RBM12 with unlabeled other-answer coefficient correction',
         'unlabeled correction using other training-fold answers'),
        ('supervised_update', 'RBM12 with supervised step-BCE coefficient correction',
         'supervised correction using other training-fold answers')]:
        add('rbm-supervision-matched-v1', key, label, panel='other_answer_diagnostic',
            fit=fit, readout=logit)
    add('rbm-first-error-objective-v1', 'first_error_pb',
        'RBM12 with supervised first-error objective; PRMB inherited unchanged',
        panel='other_answer_diagnostic', fit='supervised correction using other training-fold answers',
        readout=logit)
    return rows


def main():
    # Import only at execution: the suite summary calls this module after it
    # finishes writing its separate summary artifacts.
    from scripts.build_rbm_literature_summary import display_name

    out = run.PROGRAM
    prior = list(csv.DictReader((out / 'PRIOR_METHODS.csv').open(encoding='utf-8-sig', newline='')))
    indexed = {(r['experiment'], r['method']): r for r in prior}
    inventory = {r['experiment']: r for r in json.loads((out / 'PRIOR_EXPERIMENTS.json').read_text())}
    selected = roster()
    assert len({(r['experiment'], r['method']) for r in selected}) == len(selected)
    source_data = {}
    rows = []

    def append(spec, values, path, digest):
        rows.append(dict(**spec, source_path=str(path), source_sha256=digest,
                         benchmark='localization v3; 13769 answers; canonical source folds',
                         calibration='external-fold entropy q0.3 gate and PRMScore q0.8',
                         evidence='reviewed full development population; not untouched confirmation',
                         pb_macro_percent=100 * values['pb_all8'],
                         **{k: values[k] for k in run.METRIC_KEYS},
                         valid_answers=values['valid_answers'],
                         prm_within_n=values['prm_within_n']))

    for spec in selected:
        exp, method = spec['experiment'], spec['method']
        entry = inventory[exp]
        assert entry['status'] == 'COMPLETE' and entry['source_review'] == 'PASS', exp
        old = indexed[(exp, method)]
        path = Path(old['source_path'])
        if path not in source_data:
            digest = run.base.old.sha256_file(path)
            assert digest == old['source_sha256'], path
            source_data[path] = (json.loads(path.read_text())['metrics'], digest)
        data, digest = source_data[path]
        append(spec, data[method], path, digest)

    stages = []
    for suite in run.SUITES + ('depth_amended',):
        directory = out / suite
        state = directory / 'RUN_STATE.json'
        review = directory / 'RESULT_REVIEW.json'
        status = json.loads(state.read_text()) if state.exists() else {'status': 'NOT_STARTED'}
        passed = review.exists() and json.loads(review.read_text()).get('status') == 'PASS'
        stages.append(dict(suite=suite, status=status['status'], review_pass=passed))
        if status['status'] != 'COMPLETE' or not passed:
            continue
        path = directory / 'METRICS.json'
        digest = run.base.old.sha256_file(path)
        payload = json.loads(path.read_text())
        metrics = payload['metrics']
        if suite in run.VARIANTS:
            suite_methods = run.methods(suite)
        else:
            suite_methods = tuple(f'b{bank}_{v}_{s}' for bank in (6, 12) for v in payload['variants']
                                  for s in ('logit', 'posterior'))
        for method in suite_methods:
            coverage = metrics[method]['valid_answers'] / 13769
            spec = dict(experiment=f'literature-{suite}', method=method,
                        display_name=display_name(method) + (f' [coverage {coverage:.4f}]' if coverage < 1 else ''),
                        panel='answer_local',
                        fusion_fit='unlabeled fitting within the current answer'
                        + ('; declared per-answer failures counted as missed decisions' if coverage < 1 else ''),
                        score_readout=method.rsplit('_', 1)[-1] + '; Top10 token mean; earliest maximum')
            append(spec, metrics[method], path, digest)

    dufs = ROOT.parents[1] / '.worktrees/dufs-moment-selection-v1/results/dufs_moment_selection_v1'
    dufs_checks = {}
    for filename in ('RUN_STATE.json', 'PIPELINE_STATE.json', 'RESULT_REVIEW.json', 'STATE_REVIEW.json'):
        path = dufs / filename
        dufs_checks[filename] = json.loads(path.read_text()).get('status') if path.exists() else 'MISSING'
    dufs_complete = all(dufs_checks[name] == expected for name, expected in (
        ('RUN_STATE.json', 'COMPLETE'), ('PIPELINE_STATE.json', 'COMPLETE'),
        ('RESULT_REVIEW.json', 'PASS'), ('STATE_REVIEW.json', 'PASS')))
    if dufs_complete:
        path = dufs / 'METRICS.json'
        digest = run.base.old.sha256_file(path)
        metrics = json.loads(path.read_text())['metrics']
        for bank, label in [('all12', 'All 12 moment features'),
                            ('original6', 'Original six moment features'),
                            ('dufs6', 'DUFS selects six of 12 moment features'),
                            ('correlation6', 'Low-correlation selection of six of 12 moment features')]:
            for solver in ('rbm', 'rbm_initial'):
                method = f'{bank}__{solver}'
                fitting = ('RBM fitted without labels' if solver == 'rbm' else 'fixed RBM initial coefficients')
                if bank in ('dufs6', 'correlation6'):
                    fitting = 'unlabeled feature selection; ' + fitting
                spec = dict(experiment='dufs-moment-selection-v1', method=method,
                            display_name=label + ('; trained RBM' if solver == 'rbm' else '; RBM before training'),
                            panel='answer_local', fusion_fit=fitting + '; current answer only',
                            score_readout='posterior; Top10 token mean; earliest maximum')
                append(spec, metrics[method], path, digest)

    run.csv_write(out / 'RBM_FUSION_COMPARISON.csv', rows)
    run.base.atomic_json(out / 'RBM_FUSION_COMPARISON_COVERAGE.json', dict(
        status='PASS', rows=len(rows), prior_roster_rows=len(selected), stages=stages,
        dufs_included=dufs_complete, dufs_reported_states=dufs_checks,
        sources={str(p): value[1] for p, value in source_data.items()},
        scope='Distinct completed RBM-related experiments and controls; repeated source aliases retained only where informative.',
        panels='Readout and other-answer diagnostics are explicit, not interchangeable unlabeled candidates.',
        all_prior_aliases=str(out / 'PRIOR_METHODS.csv'),
        omissions='This is not a claim that every historical project method or published competitor has been refitted.',
        uncertainty='Use the source paired contrasts; this table does not select a winner or invent row-level intervals.'))
    print(json.dumps(dict(rows=len(rows), prior_roster_rows=len(selected), stages=stages)))


if __name__ == '__main__':
    main()
