"""Pre-registered presentation repair for the F11 category table and F11/F12 plots.

The original run computed and saved the full-population metric/contrast tables;
an indentation defect limited candidate F11 category rows to TOP_3. This script
recomputes only PRMBench category aggregates from the same frozen scores, checks
their per-bank metrics against METRICS.csv, and renders the missing figures/report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT = ROOT / 'results/monday_fusion_competitor_extension_v1/run_20261006'
RUNNER = OUT / 'run.py'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n', encoding='utf8')
    tmp.replace(path)


def load_runner():
    spec = importlib.util.spec_from_file_location('monday_fusion_competitor_run', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def freeze() -> None:
    runner = load_runner()
    if (OUT / 'REPAIR_AMENDMENT.json').exists() or (OUT / 'REPAIR_CODE_MANIFEST.json').exists():
        raise RuntimeError('Repair amendment already frozen; refusing to replace it')
    originals = ['METRICS.csv', 'CONTRASTS.csv', 'F11_PRM_CATEGORY_METRICS.csv',
                 'F11_PRM_OVERLAP.csv.gz', 'F11_PRM_OVERLAP_SUMMARY.csv', 'F11_ERROR_COUNTS.csv',
                 'F12_MTG_ANSWER_FORENSICS.csv.gz', 'F12_MTG_BREAKDOWN.csv']
    for name in originals:
        if not (OUT / name).is_file():
            raise FileNotFoundError(OUT / name)
    if (OUT / 'RUN_STATUS.json').exists() or any((OUT / 'figures').glob('*')):
        raise RuntimeError('Repair outputs already exist; refusing to overwrite')
    inputs = {name: {'path': str(OUT / name), 'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)} for name in originals}
    main = json.loads((OUT / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))['files']
    inputs.update({'frozen_input/' + k: v for k, v in main.items()})
    dump(OUT / 'REPAIR_AMENDMENT.json', {
        'schema': 'monday-fusion-competitor-repair-v1',
        'status': 'FROZEN_BEFORE_REPAIR_EVALUATION',
        'reason': 'The first complete pass calculated metrics and paired bootstrap, but the F11 candidate category aggregation was indented outside the registered decision-rule loop, leaving candidate rows only for TOP_3. The chart then correctly failed because ANSWER_Z_Q80 rows were absent.',
        'scope': 'Recompute F11 PRMBench category summaries for all 4 candidates, 3 registered decision rules, and top-1/2/3 from the frozen saved scores; verify global per-bank metrics against METRICS.csv; render F11/F12 figures and report. No ProcessBench, contrast, or bootstrap metric changes.',
        'unchanged': ['PROTOCOL.json', 'INPUT_MANIFEST.json', 'METRICS.csv', 'CONTRASTS.csv',
                      'F11_PRM_OVERLAP.csv.gz', 'F11_PRM_OVERLAP_SUMMARY.csv', 'F11_ERROR_COUNTS.csv',
                      'F12_MTG_ANSWER_FORENSICS.csv.gz', 'F12_MTG_BREAKDOWN.csv'],
        'validation': ['All candidate decision flags replay from frozen rule code.',
                       'For each bank/method/rule, aggregating category PRMScore confusion counts reproduces METRICS.csv PRMScore within 1e-12.',
                       'Category-weighted within-answer AUC reproduces METRICS.csv within-answer AUC within 1e-12.',
                       'All frozen input and main-run artifact hashes are checked before and after repair.'],
        'frozen_inputs': inputs,
        'base_run_code_sha256': sha(RUNNER),
    })
    source = [Path(__file__), RUNNER]
    dump(OUT / 'REPAIR_CODE_MANIFEST.json', {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in source})
    print(f'Repair amendment frozen; {len(inputs)} input/artifact hashes; no metrics computed.')


def repair() -> None:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    amendment_path = OUT / 'REPAIR_AMENDMENT.json'
    amendment = json.loads(amendment_path.read_text(encoding='utf8'))
    if amendment.get('status') != 'FROZEN_BEFORE_REPAIR_EVALUATION':
        raise RuntimeError('Repair amendment is not frozen')
    code = json.loads((OUT / 'REPAIR_CODE_MANIFEST.json').read_text(encoding='utf8'))
    for p, item in code.items():
        if sha(Path(p)) != item['sha256']:
            raise RuntimeError('Repair code hash mismatch: ' + p)
    for key, item in amendment['frozen_inputs'].items():
        p = Path(item['path'])
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise RuntimeError('Frozen repair input hash mismatch: ' + key)
    if (OUT / 'RUN_STATUS.json').exists() or any((OUT / 'figures').glob('*')):
        raise RuntimeError('Repair outputs already exist')

    runner = load_runner()
    pop, data, evidence, make_rules, evaluate, top_flags, prmscore_counts = runner.load_runtime()
    panel_metrics = pd.read_csv(OUT / 'METRICS.csv', low_memory=False)
    rows = []
    for bk in runner.BANKS:
        cell = 'prmbench_qwen3_8b'
        answers = np.flatnonzero(pop.cells == cell)
        step_rows = pop.rows(pop.cells == cell)
        off = np.r_[0, np.cumsum(pop.ns[answers])]
        bank = {'answers': answers, 'rows': step_rows, 'off': off}
        score_path = runner.TCI / 'scores' / f'{bk}__{cell}.npz'
        with np.load(score_path) as z:
            if not np.array_equal(z['answers'], answers) or not np.array_equal(z['offsets'], off) or not np.array_equal(z['rows'], step_rows):
                raise RuntimeError(f'TCI population mismatch in F11 repair: {bk}')
            surv = z['surv'].astype(int)
            tci_score = z['TCI_ONLY_ga'].astype(float)
        stem = f'{cell}__{bk}__groups_answer'
        diagnostic = runner.INPUT_RUN / 'chunks' / f'{stem}.diagnostics.json'
        record = json.loads(diagnostic.read_text(encoding='utf8'))
        features = data.bank(bk, cell, record)
        if int(surv.max()) >= features['raw'].shape[1] or not np.array_equal(features['rows'], step_rows):
            raise RuntimeError(f'F11 feature-bank alignment mismatch: {bk}')
        raw_surv = features['raw'][:, surv]
        method_scores, method_flags, method_detectors = {}, {}, {}
        for method, (variant, recipe) in runner.METHODS.items():
            if variant == 'tci':
                score = tci_score
            else:
                p = runner.PANEL / 'candidate_scores' / f'{cell}__{bk}__{variant}.npz'
                with np.load(p) as z:
                    if not np.array_equal(z['answer_indices'], answers) or not np.array_equal(z['offsets'], off):
                        raise RuntimeError(f'Candidate bank alignment mismatch: {bk}/{method}')
                    score = z[f'{recipe}__score'].astype(float)
            flags, detectors, _ = make_rules(score, features, raw_surv, data.entropy[answers], pop.uids[answers])
            method_scores[method] = score; method_flags[method] = flags; method_detectors[method] = detectors
            if variant != 'tci':
                p = runner.PANEL / 'candidate_scores' / f'{cell}__{bk}__{variant}.npz'
                with np.load(p) as z:
                    for rule in runner.RULES:
                        if not np.array_equal(flags[rule], z[f'{recipe}__{rule}__flags']):
                            raise RuntimeError(f'Decision flag replay mismatch: {bk}/{method}/{rule}')

        for method in runner.METHODS:
            score = method_scores[method]
            for action in runner.RULES + runner.TOPS:
                ff = method_flags[method][action] if action in runner.RULES else top_flags(score, off, np.full(len(answers), int(action[-1])))
                det = method_detectors[method].get(action) if action in runner.RULES else None
                _, per = evaluate(pop, bank, score, ff, det)
                for category in sorted({str(pop.meta[pop.ids[i]].get('category', 'unknown')) for i in answers}):
                    loc = np.asarray([str(pop.meta[pop.ids[i]].get('category', 'unknown')) == category for i in answers])
                    conf = per['confusion'][loc].sum(axis=0); auc = per['auc'][loc]
                    rows.append({'bank': bk, 'category': category, 'method': method, 'rule': action,
                        'n_answers': int(loc.sum()), 'n_auc_answers': int(np.isfinite(auc).sum()),
                        'within_answer_auc': float(np.nanmean(auc)) if np.isfinite(auc).any() else None,
                        'prmscore': float(prmscore_counts(conf)) if conf.sum() > 0 else None,
                        'top2_post_hoc': action == 'TOP_2', 'scope': 'development_only'})
                # Cross-check category aggregates to the full-population headline row.
                ref = panel_metrics[(panel_metrics.bank == bk) & (panel_metrics.cell == cell) &
                    (panel_metrics.method == method) & (panel_metrics.decision_rule == action)]
                if len(ref) != 1:
                    raise RuntimeError(f'METRICS row missing for F11 repair: {bk}/{method}/{action}')
                selected = [r for r in rows if r['bank'] == bk and r['method'] == method and r['rule'] == action]
                n_auc = sum(r['n_auc_answers'] for r in selected)
                auc_num = sum((r['within_answer_auc'] or 0.) * r['n_auc_answers'] for r in selected)
                got_auc = auc_num / n_auc if n_auc else None
                ref_auc = ref.iloc[0]['within_answer_auc']
                if not (pd.isna(ref_auc) and got_auc is None) and (got_auc is None or abs(got_auc - float(ref_auc)) > 1e-12):
                    raise RuntimeError(f'Category AUC does not reconstruct full metric: {bk}/{method}/{action}')
                conf_total = np.zeros(4)
                for i, cat in enumerate(sorted({str(pop.meta[pop.ids[j]].get('category', 'unknown')) for j in answers})):
                    mask = np.asarray([str(pop.meta[pop.ids[j]].get('category', 'unknown')) == cat for j in answers])
                    conf_total += per['confusion'][mask].sum(axis=0)
                got_ps = float(prmscore_counts(conf_total)) if conf_total.sum() else None
                ref_ps = ref.iloc[0]['prmscore']
                if not (pd.isna(ref_ps) and got_ps is None) and (got_ps is None or abs(got_ps - float(ref_ps)) > 1e-12):
                    raise RuntimeError(f'Category PRMScore counts do not reconstruct full metric: {bk}/{method}/{action}')

        # Supervised PRM comparison by the actual frozen error category.
        labels = np.concatenate([np.asarray(pop.meta[pop.ids[i]]['labels']) for i in answers])
        rewards = np.concatenate([np.asarray(pop.meta[pop.ids[i]]['rewards'], float) for i in answers])
        prm_score = -rewards; native_flags = labels == 0
        for action in ['PRM_NATIVE'] + runner.TOPS:
            ff = native_flags if action == 'PRM_NATIVE' else top_flags(prm_score, off, np.full(len(answers), int(action[-1])))
            _, per = evaluate(pop, bank, prm_score, ff)
            for category in sorted({str(pop.meta[pop.ids[i]].get('category', 'unknown')) for i in answers}):
                loc = np.asarray([str(pop.meta[pop.ids[i]].get('category', 'unknown')) == category for i in answers])
                conf = per['confusion'][loc].sum(axis=0); auc = per['auc'][loc]
                rows.append({'bank': bk, 'category': category, 'method': 'SUPERVISED_PRM', 'rule': action,
                    'n_answers': int(loc.sum()), 'n_auc_answers': int(np.isfinite(auc).sum()),
                    'within_answer_auc': float(np.nanmean(auc)) if np.isfinite(auc).any() else None,
                    'prmscore': float(prmscore_counts(conf)) if conf.sum() > 0 else None,
                    'top2_post_hoc': action == 'TOP_2', 'scope': 'development_only'})

    evidence.verify()
    for key, item in amendment['frozen_inputs'].items():
        p = Path(item['path'])
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise RuntimeError('Input changed during F11 repair: ' + key)
    repaired = pd.DataFrame(rows)
    counts = repaired.groupby(['method', 'rule']).size().to_dict()
    if any(v == 0 for v in counts.values()) or len(repaired) == 0:
        raise RuntimeError('F11 category repair has incomplete candidate coverage')
    temp = OUT / 'F11_PRM_CATEGORY_METRICS.csv.repair.tmp'
    repaired.to_csv(temp, index=False)
    temp.replace(OUT / 'F11_PRM_CATEGORY_METRICS.csv')

    figdir = OUT / 'figures'
    figdir.mkdir(exist_ok=True)
    if any(figdir.iterdir()):
        raise RuntimeError('Figure files exist; refusing to overwrite')
    pb = pd.read_csv(OUT / 'F12_MTG_BREAKDOWN.csv')
    for dark in (False, True):
        bg = '#17191d' if dark else '#ffffff'; fg = '#f4f4f4' if dark else '#222222'
        suffix = '_dark' if dark else '_light'
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sub = repaired[(repaired.method.isin(list(runner.METHODS))) & (repaired.rule == 'ANSWER_Z_Q80')]
        for ax, metric, title in zip(axes, ['within_answer_auc', 'prmscore'], ['Within-answer AUC', 'PRMScore']):
            plot = sub.groupby(['category', 'method'])[metric].mean().unstack('method')
            plot = plot.apply(pd.to_numeric, errors='coerce')
            if not plot.notna().any().any():
                raise RuntimeError('F11 category figure has no numeric values: ' + metric)
            plot.plot(kind='bar', ax=ax); ax.set_title(title, color=fg); ax.set_ylabel(metric); ax.tick_params(axis='x', rotation=30)
        fig.suptitle('F11 · PRMBench by error category · answer-z q80', color=fg); fig.tight_layout()
        fig.patch.set_facecolor(bg)
        for ax in fig.axes:
            ax.set_facecolor(bg); ax.tick_params(colors=fg); ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)
            for sp in ax.spines.values(): sp.set_color(fg)
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(bg)
                for t in leg.get_texts(): t.set_color(fg)
        fig.savefig(figdir / f'F11_PRM_categories{suffix}.png', dpi=180, facecolor=bg, bbox_inches='tight')
        fig.savefig(figdir / f'F11_PRM_categories{suffix}.svg', facecolor=bg, bbox_inches='tight'); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        subpb = pb[(pb.method.isin(list(runner.METHODS))) & (pb.rule == 'R2_COUNT') &
                   (pb.stratum.isin(['error_position_quartile', 'answer_length']))]
        for ax, stratum, title in zip(axes, ['error_position_quartile', 'answer_length'], ['Error position quartile', 'Answer length']):
            x = subpb[subpb.stratum == stratum].groupby(['value', 'method'])[
                ['candidate_argmax_hit', 'candidate_decision_hit', 'MTG_native_argmax_hit']].mean()
            x.unstack('method').plot(kind='bar', ax=ax); ax.set_title(title, color=fg); ax.set_ylabel('First-error hit rate'); ax.tick_params(axis='x', rotation=25)
        fig.suptitle('F12 · ProcessBench first-error localization · R2_COUNT', color=fg); fig.tight_layout()
        fig.patch.set_facecolor(bg)
        for ax in fig.axes:
            ax.set_facecolor(bg); ax.tick_params(colors=fg); ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)
            for sp in ax.spines.values(): sp.set_color(fg)
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(bg)
                for t in leg.get_texts(): t.set_color(fg)
        fig.savefig(figdir / f'F12_MTG_position_length{suffix}.png', dpi=180, facecolor=bg, bbox_inches='tight')
        fig.savefig(figdir / f'F12_MTG_position_length{suffix}.svg', facecolor=bg, bbox_inches='tight'); plt.close(fig)

    run_status = {'status': 'COMPUTED_PENDING_INDEPENDENT_REVIEW', 'development_only': True,
        'n_banks': len(runner.BANKS), 'n_cells': len(sorted(set(pop.cells))), 'n_methods': len(runner.METHODS),
        'metric_rows': len(pd.read_csv(OUT / 'METRICS.csv')), 'category_rows': len(repaired),
        'overlap_rows': sum(1 for _ in __import__('gzip').open(OUT / 'F11_PRM_OVERLAP.csv.gz', 'rt', encoding='utf8')) - 1,
        'pb_forensic_rows': sum(1 for _ in __import__('gzip').open(OUT / 'F12_MTG_ANSWER_FORENSICS.csv.gz', 'rt', encoding='utf8')) - 1,
        'candidate_panel_parity_rows': len(runner.BANKS) * len(sorted(set(pop.cells))) * len(runner.RULES) * 3,
        'tci_stage2_parity_rows': len(runner.BANKS) * len(sorted(set(pop.cells))) * len(runner.RULES),
        'paired_source_group_bootstrap': {'draws': 5000, 'seed': 20261006},
        'repair': {'status': 'PASS', 'reason': 'category loop corrected; metrics/contrasts/bootstrap unchanged'},
        'review': 'CLAUDE_PENDING; not yet cleared for final presentation'}
    dump(OUT / 'RUN_STATUS.json', run_status)
    (OUT / 'REPORT.md').write_text('# F11–F12: ארבע גרסאות מול PRM ו־Mind the Gap\n\n'
        'כל התוצאות הן ניתוח פיתוח על האוכלוסייה שכבר נחשפה, ולא אישור חיצוני לא־חשוף. '
        'F11 מציג את קטגוריית שגיאת ה־PRM לפי השדה הקפוא `category`; `classification` נשמר בנפרד. '
        'F12 משווה ל־Mind the Gap המקומי: teacher-forced, paper EMA span 5, later-token attribution, '
        'top-5 drop mean ו־native argmax. זה אינו מספר הכותרת של המאמר, ול־MTG אין כלל abstention לתשובות נקיות.\n\n'
        f'- {run_status["metric_rows"]:,} שורות מדדים; {run_status["category_rows"]:,} שורות קטגוריה/בנק/שיטה/כלל.\n'
        f'- {run_status["overlap_rows"]:,} רשומות overlap PRM ו־{run_status["pb_forensic_rows"]:,} רשומות ProcessBench.\n'
        '- כללי ההחלטה מוצגים בנפרד. top-1/2/3 הן בקרות דירוג; top-2 מסומן כבחירה בדיעבד.\n'
        '- ניגודים מזווגים לפי source group, 5,000 דגימות bootstrap; רווחים תיאוריים של 95%.\n'
        '- חישובי הכותרת התאימו לשורות פאנל המועמדים / שלב 2 עד 1e-12. תיקון F11 לא שינה את METRICS.csv או CONTRASTS.csv.\n'
        '- ביקורת עצמאית של Claude עדיין ממתינה.\n', encoding='utf8')
    dump(OUT / 'REPAIR_STATUS.json', {'status': 'PASS', 'repaired': 'F11_PRM_CATEGORY_METRICS.csv and F11/F12 figures/report',
        'metrics_changed': False, 'contrasts_changed': False, 'repair_amendment_sha256': sha(amendment_path),
        'completed_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
    print(json.dumps(run_status, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze', action='store_true')
    g.add_argument('--repair', action='store_true')
    a = ap.parse_args()
    freeze() if a.freeze else repair()
