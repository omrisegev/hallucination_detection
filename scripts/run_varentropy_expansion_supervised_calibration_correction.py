"""Correction 2026-09-13 of the completed supervised diagnostic (varentropy expansion fusion v1).

Writes ONLY to results/varentropy_expansion_fusion_v1/supervised/correction_20260913/.
The completed run (CHECKPOINT.sqlite, METRICS.json, CONTRASTS.json, MANIFEST.json,
RUN_STATE.json, SCORES.npz) is opened read-only and never rewritten.

Stages (``--stage``):
  health       FIT_HEALTH_SUPERVISED.json from the saved fit records of the completed run.
  cache        CACHE_VERIFICATION.json: every cached answer's spans re-derived from the frozen
               BENCH/scores/<uid>.npz and compared with the cached spans; uid/row/step/token
               counts checked; CURRENT raw-source hashes recorded with an explicit statement
               that extraction-time source state was not recorded.
  calibration  Held-fold-blind PRMScore calibration (Finding 1).  In the completed run the
               0.8-quantile threshold q_f was taken from saved scores of training folds g != f,
               each produced by model_g which was trained WITH fold f's labels.  Here, for each
               bank and each held outer fold f, inner models fit on folds not in {f, h} score
               fold h (h != f), giving inner out-of-fold calibration scores for every
               outer-training answer from models that never saw fold f.  q_f is the
               0.8-quantile of those scores under the exact concatenation/quantile convention
               of ``evaluate_arrays``; held fold f is scored by the EXISTING saved outer fit
               (same provenance: trained on folds != f; reused, not refit).  ProcessBench is
               unaffected and reuses the saved scores.  Same fit(), label rules, ridge, optimizer.
Same q=0.8, same v2 source groups / outer folds, same step labels, same population, same banks,
same top-10 readout.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_direct_probability_temporal as base
from scripts.run_varentropy_expansion_fusion_v1 import MEMORY_GATE_KB, atomic_json_retry, clean, wait_for_memory
from scripts.run_varentropy_expansion_supervised_v1 import METHODS, OUT as SUPERVISED, load_answers, raw_source_paths
from spectral_utils.varentropy_expansion_supervised import (BANKS, bank_columns, calibration_plan, calibration_quantile, fit, fit_health,
                                                            held_fold_blind_check, score_steps, spans_sha256, training_matrix)

CORR = SUPERVISED / 'correction_20260913'
CELL = 'prmbench_qwen3_8b'
Q = .8
base.atomic_json = atomic_json_retry
FROZEN = ('CHECKPOINT.sqlite', 'METRICS.json', 'CONTRASTS.json', 'MANIFEST.json', 'RUN_STATE.json', 'SCORES.npz')


def open_readonly(path):
    return sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True)


def frozen_hashes():
    return {name: base.old.sha256_file(SUPERVISED / name) for name in FROZEN}


def load_benchmark():
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    if len(records) != 13769: raise ValueError('benchmark roster mismatch')
    folds = json.loads(base.old.FOLDS.read_text(encoding='utf8'))['outer']
    outer = np.array([int(folds[r['group_id']]) for r in records])
    return records, joined, outer


def summary(values):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    return dict(n=int(len(v)), mean=float(v.mean()) if len(v) else None, std=float(v.std()) if len(v) else None,
                q80=float(np.quantile(v, Q)) if len(v) else None)


# ----------------------------------------------------------------------------- health

def stage_health():
    metrics = json.loads((SUPERVISED / 'METRICS.json').read_text(encoding='utf8'))
    health = fit_health(metrics['fits'])
    expected = dict(n_fits=90, n_converged=67, n_iteration_limit=23, n_stalled=0)
    for k, v in expected.items():
        if health[k] != v: raise AssertionError(f'fit health {k}: {health[k]} != expected {v}')
    payload = dict(schema='varentropy-expansion-supervised-fit-health-20260913', source='supervised/METRICS.json fits',
                   verified_against=expected, **{k: v for k, v in health.items() if k != 'by_cell_bank_fold'},
                   iteration_limit_fits=sorted(k for k, o in health['by_cell_bank_fold'].items() if o == 'FIT_ITERATION_LIMIT'),
                   note='FIT_ITERATION_LIMIT fits are finite models that stopped on the L-BFGS-B iteration budget; they were used for scoring but are not converged.')
    base.atomic_json(CORR / 'FIT_HEALTH_SUPERVISED.json', clean(payload))
    print('[health]', {k: health[k] for k in expected}, flush=True)


# ----------------------------------------------------------------------------- cache verification

def stage_cache(records, joined, outer):
    started = time.perf_counter(); offsets = joined['offsets']
    con = open_readonly(SUPERVISED / 'CHECKPOINT.sqlite')
    stored = json.loads(con.execute('SELECT payload FROM manifest WHERE id=1').fetchone()[0])
    manifest_file = json.loads((SUPERVISED / 'MANIFEST.json').read_text(encoding='utf8'))
    n_rows = con.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
    mismatches = []; per_cell = {}; checked = 0; provenance_recorded = 0
    for idx, blob, info_json in con.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
        r = records[idx]; info = json.loads(info_json); cell = r['cell']; per_cell.setdefault(cell, 0)
        problems = []
        if info.get('uid') != r['uid']: problems.append(f'uid {info.get("uid")} != {r["uid"]}')
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            spans = np.asarray(z['spans'], int); n_tokens = int(len(z['z']))
        with np.load(base.old.BENCH / 'scores' / f'{r["uid"]}.npz', allow_pickle=False) as b:
            starts, ends = np.asarray(b['step_starts'], int), np.asarray(b['step_ends'], int)
        if spans.shape != (int(r['steps']), 2): problems.append(f'span shape {spans.shape} != ({r["steps"]}, 2)')
        elif not (np.array_equal(spans[:, 0], starts) and np.array_equal(spans[:, 1], ends)): problems.append('spans differ from frozen BENCH boundaries')
        if int(r['steps']) != int(offsets[idx + 1] - offsets[idx]): problems.append('record steps != JOINED offsets')
        if info.get('n_steps') != len(spans): problems.append('cached n_steps != spans')
        if info.get('n_tokens') != n_tokens: problems.append('cached n_tokens != bank rows')
        if int(r['tokens']) != n_tokens: problems.append(f'benchmark tokens {r["tokens"]} != cached bank rows {n_tokens}')
        if len(spans) and (np.any(spans[:, 0] < 0) or np.any(spans[:, 1] <= spans[:, 0]) or np.any(spans[:, 1] > n_tokens)): problems.append('span outside cached answer')
        if 'provenance' in info: provenance_recorded += 1
        if problems: mismatches.append(dict(idx=int(idx), uid=r['uid'], problems=problems))
        per_cell[cell] += 1; checked += 1
        if checked % 2000 == 0: print('[cache]', checked, '/', n_rows, round(time.perf_counter() - started, 1), 's', flush=True)
    con.close()
    print('[cache] hashing current raw sources', flush=True)
    raw_now = {str(p): base.old.sha256_file(p) for p in raw_source_paths()}
    payload = dict(
        schema='varentropy-expansion-supervised-cache-verification-20260913',
        checkpoint=str(SUPERVISED / 'CHECKPOINT.sqlite'), n_cached_answers=int(n_rows), n_benchmark_answers=len(records), n_checked=checked,
        per_cell=per_cell, n_mismatches=len(mismatches), mismatches=mismatches,
        checks=['info.uid == JOINED record uid', 'cached spans == frozen BENCH/scores/<uid>.npz step_starts/step_ends (values)',
                'span count == record steps == JOINED offsets', 'cached n_tokens == bank rows == record tokens', 'spans inside cached answer'],
        status='PASS' if not mismatches and checked == len(records) else 'FAIL',
        stored_manifest_equals_manifest_json=(stored == manifest_file),
        stored_manifest_schema=stored.get('schema'), stored_manifest_hashed_paths=sorted(stored.get('hashes', {})),
        cached_info_has_provenance=provenance_recorded,
        extraction_time_source_state='NOT RECORDED. The completed run\'s manifest hashed code, JOINED/FOLDS and the gate only; the raw pickles were not hashed at extraction time and the cached info carries no per-answer source provenance. The hashes below are the CURRENT state of the raw artifacts and are not proof of the state read during extraction.',
        current_raw_source_hashes=raw_now, frozen_run_hashes=frozen_hashes(), elapsed_seconds=time.perf_counter() - started)
    base.atomic_json(CORR / 'CACHE_VERIFICATION.json', clean(payload))
    print('[cache]', payload['status'], 'checked', checked, 'mismatches', len(mismatches), round(payload['elapsed_seconds'], 1), 's', flush=True)


# ----------------------------------------------------------------------------- calibration

def calibration_manifest():
    files = [Path(__file__), ROOT / 'scripts/run_varentropy_expansion_supervised_v1.py', ROOT / 'spectral_utils/varentropy_expansion_supervised.py',
             ROOT / 'spectral_utils/varentropy_expansion_fusion.py', ROOT / 'scripts/test_varentropy_expansion_supervised.py',
             base.old.BENCH / 'evaluation/JOINED.json', base.old.BENCH / 'evaluation/JOINED.npz', base.old.FOLDS,
             base.old.FIXED_GATE / 'DETECTORS.npz', base.old.FIXED_GATE / 'METRICS.json', base.old.PRMB_LABELS]
    hashes = {str(p): base.old.sha256_file(p) for p in files}
    hashes.update({f'supervised/{k}': v for k, v in frozen_hashes().items()})
    return dict(schema='varentropy-expansion-supervised-calibration-correction-20260913', cell=CELL, banks=list(BANKS), quantile=Q,
                access='supervised; other-answer step labels; source-group-disjoint outer folds; held-fold-blind inner calibration',
                hashes=hashes, source_root=str(base.old.SOURCE_ROOT))


def flip_fold_labels(labels, offsets, indices, outer, held):
    flipped = np.array(labels, copy=True); n = 0
    for i in indices:
        if int(outer[i]) != int(held): continue
        sl = slice(int(offsets[i]), int(offsets[i + 1])); block = flipped[sl]; known = block >= 0
        block[known] = 1 - block[known]; flipped[sl] = block; n += int(known.sum())
    return flipped, n


def run_plan(con, answers, records, joined_labels, target, offsets, outer, prm, bank, held, *, store=True, log=print):
    """Fit (or reload) the inner models for held fold ``held``; returns (flat calibration scores, per-model info)."""
    cols = bank_columns(bank); flat = np.full(int(offsets[-1]), np.nan); models = []
    for h, train, score in calibration_plan(prm, outer, held):
        key = f'{CELL}|{bank}|held{held}|inner{h}'
        train_groups = held_fold_blind_check(records, outer, held, train, score)
        # index-level label check: the step indices whose labels this model consumes never touch fold `held`
        consumed = np.zeros(int(offsets[-1]), bool); held_steps = np.zeros(int(offsets[-1]), bool)
        for i in train: consumed[offsets[i]:offsets[i + 1]] = True
        for i in prm:
            if int(outer[i]) == int(held): held_steps[offsets[i]:offsets[i + 1]] = True
        if (consumed & held_steps).any(): raise AssertionError(f'{key}: consumed label indices intersect held fold')
        row = con.execute('SELECT theta,info FROM fits WHERE key=?', (key,)).fetchone() if store else None
        if row is not None:
            theta = np.load(io.BytesIO(row[0]))['theta'] if row[0] is not None else None; info = json.loads(row[1])
        else:
            started = time.perf_counter(); theta = None
            try:
                if not train or not score: raise ValueError('empty training or scoring fold (declared fold failure)')
                x, spans, y = training_matrix(answers, train, cols, kind='prm', target=target, labels=joined_labels, offsets=offsets)
                theta, info = fit(x, spans, y)
                info.update(n_train_answers=len(train), n_score_answers=len(score), n_train_tokens=int(len(x)))
                if info['status'] == 'STALLED': theta = None
                del x
            except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as error:
                info = dict(status='FAILED', reason=f'{type(error).__name__}: {error}', n_train_answers=len(train), n_score_answers=len(score))
            info.update(seconds=time.perf_counter() - started, cell=CELL, bank=bank, held_fold=int(held), inner_fold=int(h),
                        training_folds=sorted({int(outer[i]) for i in train}), n_training_groups=len(train_groups),
                        training_groups_sha256=hashlib.sha256('\n'.join(train_groups).encode('utf8')).hexdigest(),
                        held_fold_label_indices_consumed=0, held_fold_groups_intersect_training=0)
            if store:
                con.execute('INSERT INTO fits VALUES (?,?,?)', (key, base.packed(theta=theta) if theta is not None else None, base.dumps(clean(info))))
                con.commit()
            log('[calibration-fit]', key, info['status'], info.get('message', info.get('reason', '')), round(info['seconds'], 1), 's')
        if theta is not None:
            for i in score:
                z, sp = answers[i]; flat[offsets[i]:offsets[i + 1]] = score_steps(z[:, cols], sp, theta)
        models.append(dict(key=key, theta=theta, info=info, training_groups=train_groups, score_indices=score))
    return flat, models


def stage_calibration(records, joined, outer, *, gate_kb, perturbation_bank, perturbation_fold, skip_perturbation):
    started = time.perf_counter(); offsets = joined['offsets']; labels = joined['labels']; target = joined['target']
    prm = [i for i, r in enumerate(records) if r['cell'] == CELL]
    if len(prm) != 6969: raise ValueError('PRMBench roster mismatch')
    manifest = calibration_manifest()
    con = base.connect(CORR / 'CALIBRATION_FITS.sqlite', manifest)
    con.execute('CREATE TABLE IF NOT EXISTS fits (key TEXT PRIMARY KEY, theta BLOB, info TEXT NOT NULL)'); con.commit()
    base.atomic_json(CORR / 'CALIBRATION_MANIFEST.json', manifest)
    original_metrics = json.loads((SUPERVISED / 'METRICS.json').read_text(encoding='utf8'))['metrics']
    saved = np.load(SUPERVISED / 'SCORES.npz', allow_pickle=False)
    scores = {m: saved['steps__' + m] for m in METHODS}
    outer_fits = {}
    original = open_readonly(SUPERVISED / 'CHECKPOINT.sqlite')
    for bank in BANKS:
        for f in range(5):
            theta_blob, info_json = original.execute('SELECT theta,info FROM fits WHERE key=?', (f'{CELL}|{bank}|{f}',)).fetchone()
            outer_fits[(bank, f)] = dict(info=json.loads(info_json), theta_sha256=hashlib.sha256(theta_blob).hexdigest() if theta_blob else None)
    wait_for_memory(gate_kb); print('[calibration] loading', len(prm), 'PRMBench answers from the frozen checkpoint (read-only)', flush=True)
    answers = load_answers(original, prm); original.close()
    if len(answers) != len(prm): raise ValueError('frozen checkpoint does not hold every PRMBench answer')
    thresholds = {}; provenance = {}; calibration_flat = {}; model_records = []
    for bank in ('B2d_sel', 'B2_sel'):
        method = f'sup__{bank}'; thresholds[method] = {}; provenance[method] = {}
        for f in range(5):
            flat, models = run_plan(con, answers, records, labels, target, offsets, outer, prm, bank, f, log=lambda *a: print(*a, flush=True))
            calibration_flat[(bank, f)] = flat
            valid_outer = np.array([np.isfinite(scores[method][offsets[i]:offsets[i + 1]]).all() for i in range(len(records))])
            train = [i for i in prm if outer[i] != f and valid_outer[i]]; held = [i for i in prm if outer[i] == f and valid_outer[i]]
            missing = [i for i in train if not np.isfinite(flat[offsets[i]:offsets[i + 1]]).all()]
            q_corr = calibration_quantile(flat, train, offsets, Q) if not missing else None
            q_orig = float(original_metrics[method]['prmscore_thresholds'][str(f)])
            q_replay = calibration_quantile(scores[method], train, offsets, Q)
            if abs(q_replay - q_orig) > 1e-12: raise AssertionError(f'{method} fold {f}: original threshold does not replay from saved scores ({q_replay} vs {q_orig})')
            outer_training_saved = np.concatenate([scores[method][offsets[i]:offsets[i + 1]] for i in train])
            inner_oof = np.concatenate([flat[offsets[i]:offsets[i + 1]] for i in train if i not in missing]) if len(train) > len(missing) else np.zeros(0)
            held_saved = np.concatenate([scores[method][offsets[i]:offsets[i + 1]] for i in held])
            thresholds[method][str(f)] = dict(
                original_q=q_orig, original_q_replayed_from_saved_scores=q_replay, corrected_q=q_corr, delta=(q_corr - q_orig) if q_corr is not None else None,
                n_outer_training_answers=len(train), n_held_answers=len(held), n_missing_calibration_answers=len(missing),
                scale=dict(inner_oof_calibration_scores=summary(inner_oof), outer_training_saved_scores_used_originally=summary(outer_training_saved),
                           held_fold_saved_scores=summary(held_saved)),
                held_fold_model=dict(key=f'{CELL}|{bank}|{f}', reused_existing_outer_fit=True, theta_sha256=outer_fits[(bank, f)]['theta_sha256'],
                                     status=outer_fits[(bank, f)]['info']['status'], converged=outer_fits[(bank, f)]['info'].get('converged'),
                                     iterations=outer_fits[(bank, f)]['info'].get('iterations'), n_train_answers=outer_fits[(bank, f)]['info'].get('n_train_answers')),
                inner_models=[dict(key=m['key'], training_folds=m['info'].get('training_folds'), n_train_answers=m['info'].get('n_train_answers'),
                                   n_train_steps=m['info'].get('n_train_steps'), n_train_tokens=m['info'].get('n_train_tokens'), status=m['info']['status'],
                                   converged=m['info'].get('converged'), iterations=m['info'].get('iterations'), message=m['info'].get('message', m['info'].get('reason')),
                                   seconds=m['info'].get('seconds'), gradient_max=m['info'].get('gradient_max'), theta_norm=m['info'].get('theta_norm'),
                                   n_scored_answers=len(m['score_indices'])) for m in models])
            held_groups = sorted({records[i]['group_id'] for i in prm if outer[i] == f})
            provenance[method][str(f)] = dict(
                held_fold=f, held_fold_groups=len(held_groups), held_fold_answers=len([i for i in prm if outer[i] == f]),
                contributing_models=[dict(key=m['key'], training_folds=m['info'].get('training_folds'), training_groups=m['training_groups'],
                                          intersects_held_fold_groups=bool(set(m['training_groups']) & set(held_groups)),
                                          held_fold_rows_in_training=False, held_fold_label_indices_consumed=0) for m in models],
                assertion='no contributing model trained on a row or source group of the held fold (index-level and group-level checks raised nothing)')
            if any(c['intersects_held_fold_groups'] for c in provenance[method][str(f)]['contributing_models']): raise AssertionError('provenance intersection')
            model_records.extend(m['info'] for m in models)
            base.atomic_json(CORR / 'THRESHOLDS.json', clean(dict(schema='varentropy-expansion-supervised-thresholds-20260913', quantile=Q, cell=CELL,
                                                                    status='PARTIAL', thresholds=thresholds)))
            print('[calibration]', method, 'fold', f, 'q original', round(q_orig, 6), 'corrected', None if q_corr is None else round(q_corr, 6), flush=True)
    np.savez_compressed(CORR / 'CALIBRATION_SCORES.npz', **{f'calibration__{b}__held{f}': v for (b, f), v in calibration_flat.items()})
    # --- evaluation: reuse the saved outer scores; only the PRMScore thresholds change
    complete = all(v['corrected_q'] is not None for m in thresholds.values() for v in m.values())
    replay, _ = base.evaluate_arrays(records, joined, scores)
    compare_metric_bundles(replay, original_metrics, note='replay of the original evaluation from saved scores')
    corrected = None
    if complete:
        thr = {m: {f: v['corrected_q'] for f, v in t.items()} for m, t in thresholds.items()}
        corrected, _ = base.evaluate_arrays(records, joined, scores, calibration_thresholds=thr)
        compare_metric_bundles(corrected, original_metrics, note='corrected evaluation vs original', skip=('prmscore_q08', 'prmscore_conditional', 'prmscore_thresholds'))
    health = fit_health(model_records)
    base.atomic_json(CORR / 'THRESHOLDS.json', clean(dict(schema='varentropy-expansion-supervised-thresholds-20260913', quantile=Q, cell=CELL,
                                                            status='COMPLETE' if complete else 'DECLARED_INCOMPLETE', thresholds=thresholds,
                                                            inner_fit_health={k: v for k, v in health.items() if k != 'by_cell_bank_fold'})))
    base.atomic_json(CORR / 'METRICS_CORRECTED.json', clean(dict(
        schema='varentropy-expansion-supervised-metrics-corrected-20260913', access='supervised; other-answer access; matched diagnostic, not a ceiling',
        correction='PRMScore threshold q_f from held-fold-blind inner out-of-fold calibration scores; PB and PRMB ranking metrics reuse the saved scores and are asserted identical',
        status='COMPLETE' if complete else 'DECLARED_INCOMPLETE', metrics=corrected, original_metrics=original_metrics,
        prmscore=({m: dict(original=original_metrics[m]['prmscore_q08'], corrected=corrected[m]['prmscore_q08'],
                           delta=corrected[m]['prmscore_q08'] - original_metrics[m]['prmscore_q08']) for m in METHODS} if complete else None),
        identical_to_original=['pb_all8', 'pb_q4', 'pb_q8', 'pb_cells', 'prm_within', 'prm_pooled', 'valid_answers', 'pb_clean_accuracy', 'pb_raw_exact'],
        inner_fit_health={k: v for k, v in health.items() if k != 'by_cell_bank_fold'}, n_inner_models=len(model_records),
        elapsed_seconds=time.perf_counter() - started)))
    for m in METHODS:
        print('[metrics]', m, 'PRMScore original', original_metrics[m]['prmscore_q08'], 'corrected', corrected[m]['prmscore_q08'] if corrected else None, flush=True)
    # --- label-invariance demonstration
    perturbation = dict(skipped=True)
    if not skip_perturbation:
        perturbation = perturbation_test(con, answers, records, labels, target, offsets, outer, prm, perturbation_bank, perturbation_fold, calibration_flat)
    base.atomic_json(CORR / 'PROVENANCE.json', clean(dict(
        schema='varentropy-expansion-supervised-provenance-20260913', cell=CELL, quantile=Q,
        held_fold_scoring='held fold f is scored by the existing saved outer fit prmbench_qwen3_8b|{bank}|{f} (trained on folds != f); reused, not refit',
        index_level_label_check='for every calibration model the consumed label step indices were asserted disjoint from the held fold\'s step indices, and its training rows/groups disjoint from the held fold (held_fold_blind_check)',
        thresholds=provenance, label_invariance_perturbation=perturbation, calibration_manifest=manifest, elapsed_seconds=time.perf_counter() - started)))
    con.close()
    print('[calibration] done', round(time.perf_counter() - started, 1), 's', flush=True)


def perturbation_test(con, answers, records, labels, target, offsets, outer, prm, bank, held, calibration_flat):
    """Flip every known step label of held fold ``held``, refit its calibration models, require bit-identical theta and q."""
    started = time.perf_counter()
    flipped, n_flipped = flip_fold_labels(labels, offsets, prm, outer, held)
    if n_flipped == 0: raise AssertionError('perturbation flipped no labels')
    if np.array_equal(flipped, labels): raise AssertionError('perturbation left labels unchanged')
    ro = open_readonly(CORR / 'CALIBRATION_FITS.sqlite'); stored = {}
    for h in range(5):
        if h == held: continue
        row = ro.execute('SELECT theta FROM fits WHERE key=?', (f'{CELL}|{bank}|held{held}|inner{h}',)).fetchone()
        stored[h] = np.load(io.BytesIO(row[0]))['theta'] if row and row[0] is not None else None
    ro.close()
    flat, models = run_plan(con, answers, records, flipped, target, offsets, outer, prm, bank, held, store=False, log=lambda *a: print('[perturbation]', *a, flush=True))
    comparisons = []
    for m in models:
        h = m['info']['inner_fold']; a = stored[h]; b = m['theta']
        identical = (a is None and b is None) or (a is not None and b is not None and a.tobytes() == b.tobytes())
        comparisons.append(dict(key=m['key'], theta_bit_identical=bool(identical), theta_sha256_original=hashlib.sha256(a.tobytes()).hexdigest() if a is not None else None,
                                theta_sha256_perturbed=hashlib.sha256(b.tobytes()).hexdigest() if b is not None else None))
        if not identical: raise AssertionError(f'{m["key"]}: theta changed under a held-fold label flip')
    train = [i for i in prm if outer[i] != held]
    q_orig = calibration_quantile(calibration_flat[(bank, held)], train, offsets, Q); q_pert = calibration_quantile(flat, train, offsets, Q)
    if q_orig != q_pert: raise AssertionError(f'q changed under a held-fold label flip: {q_orig} vs {q_pert}')
    if not np.array_equal(np.nan_to_num(flat, nan=-1e300), np.nan_to_num(calibration_flat[(bank, held)], nan=-1e300)): raise AssertionError('calibration scores changed under held-fold flip')
    print('[perturbation]', bank, 'held', held, 'flipped', n_flipped, 'labels; theta and q bit-identical', flush=True)
    return dict(skipped=False, bank=bank, held_fold=held, n_labels_flipped=n_flipped, n_models=len(models), theta_bit_identical=True,
                q_original=q_orig, q_perturbed=q_pert, q_bit_identical=(q_orig == q_pert), calibration_scores_identical=True,
                models=comparisons, seconds=time.perf_counter() - started,
                statement='labels of the held fold were flipped (0<->1, excluded untouched) and every calibration model for that fold refit in the same process; theta bytes and q_f are identical because fold f is never read')


def compare_metric_bundles(actual, expected, *, note, skip=()):
    for m in METHODS:
        for key, value in expected[m].items():
            if key in skip: continue
            a = clean(actual[m].get(key))
            if not _equal(a, value): raise AssertionError(f'{note}: {m}.{key} differs: {a!r} vs {value!r}')
    print('[assert]', note, 'identical on', [k for k in expected[METHODS[0]] if k not in skip], flush=True)


def _equal(a, b, tol=1e-12):
    if isinstance(a, dict) and isinstance(b, dict): return set(a) == set(b) and all(_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)): return len(a) == len(b) and all(_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None: return a is b
        return abs(float(a) - float(b)) <= tol
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--stage', choices=('all', 'health', 'cache', 'calibration'), default='all')
    p.add_argument('--memory-gate-prmb-kb', type=int, default=MEMORY_GATE_KB['prm'])
    p.add_argument('--perturbation-bank', default='B2d_sel', choices=BANKS); p.add_argument('--perturbation-fold', type=int, default=0)
    p.add_argument('--skip-perturbation', action='store_true')
    args = p.parse_args(); base.old.configure_source_root(args.source_root.resolve()); CORR.mkdir(parents=True, exist_ok=True)
    state = CORR / ('STATE_' + args.stage + '.json')
    base.atomic_json(state, dict(status='RUNNING', stage=args.stage, started=time.time()))
    try:
        with threadpool_limits(limits=1):
            records, joined, outer = load_benchmark()
            if args.stage in ('all', 'health'): stage_health()
            if args.stage in ('all', 'cache'): stage_cache(records, joined, outer)
            if args.stage in ('all', 'calibration'):
                stage_calibration(records, joined, outer, gate_kb=args.memory_gate_prmb_kb, perturbation_bank=args.perturbation_bank,
                                  perturbation_fold=args.perturbation_fold, skip_perturbation=args.skip_perturbation)
        base.atomic_json(state, dict(status='COMPLETE', stage=args.stage))
    except BaseException as error:
        base.atomic_json(state, dict(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', stage=args.stage, error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
