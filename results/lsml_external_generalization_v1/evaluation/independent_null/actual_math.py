"""Full-population algebra checks on sealed predictions, no summary imports."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT'):
    os.environ[key] = '1'
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash, require_seal
OUT = Path(__file__).resolve().parent
EVAL = OUT.parent
def load(p): return json.loads(p.read_text(encoding='utf8'))
bundle = load(EVAL/'source/BUNDLE.json')
bundle_hash = file_hash(EVAL/'source/BUNDLE.json')
allseal = load(EVAL/'ALL_CELLS_SEALED.json')
assert bundle_hash == allseal['bundle_sha256']
result = {}
for cell in ('hard2verify_qwen3_8b','socratic_qwen3_8b','socratic_qwq32b'):
    rows = load(EVAL/cell/'PREDICTIONS.json')
    seal = load(EVAL/cell/'SEAL.json')
    assert seal == allseal['seals'][cell]
    require_seal(rows,seal,bundle_hash)
    counts = dict(answers=len(rows), native=0, fallback=0, empty_steps=0, step_arm_decisions_checked=0,
                  degenerate_native=0, guarded_native=0, flagged_small_group_native=0, zero_anchor_rho=0)
    maxmean = maxsd = maxl1 = 0.
    reasons = {}
    for uid,row in rows.items():
        valid = np.asarray(row['nonempty'],bool)
        counts['empty_steps'] += int((~valid).sum())
        for arm, values in row['scores'].items():
            s = np.asarray([np.nan if v is None else v for v in values],float)
            p = np.asarray(row['predictions'][arm])
            assert np.array_equal(np.isfinite(s),valid),(uid,arm,'finite')
            assert np.array_equal(p[valid],s[valid] < bundle['thresholds'][arm]),(uid,arm,'threshold')
            assert np.all(p[~valid] == 0),(uid,arm,'empty')
            v = s[valid]
            maxmean = max(maxmean,abs(float(v.mean())))
            if v.std() > 1e-8: maxsd = max(maxsd,abs(float(v.std())-1))
            else: assert np.all(v == 0),(uid,arm,'constant')
            counts['step_arm_decisions_checked'] += len(p)
        d = row['local']
        if d['native']:
            counts['native'] += 1
            w = np.asarray(d['weights'],float)
            assert np.isfinite(w).all() and len(w) == len(d['groups']) == len(d['active_channels'])
            assert d['fit_rows'] >= 3*len(w) and len(w) >= 3
            assert 0 in d['active_channels'] and d['anchor_spearman'] >= 0
            assert np.isfinite(d['residual'])
            maxl1 = max(maxl1,abs(float(np.abs(w).sum())-1))
            counts['degenerate_native'] += bool(d.get('grouping_diag',{}).get('degenerate',False))
            counts['guarded_native'] += bool(d.get('small_m_guarded'))
            counts['flagged_small_group_native'] += bool(d.get('small_m_flags'))
            counts['zero_anchor_rho'] += d['anchor_spearman'] == 0
        else:
            counts['fallback'] += 1
            reasons[d.get('reason','MISSING')] = reasons.get(d.get('reason','MISSING'),0)+1
            for arm in ('local_equal','local_partition_equal'):
                assert row['scores'][arm] == row['scores']['local_lsml'],(uid,'matched fallback scores')
            # Decisions use separately source-calibrated per-arm thresholds;
            # matched fallback scores need not imply identical predictions.
    assert maxmean < 1e-10 and maxsd < 1e-10 and maxl1 < 1e-10
    counts.update(max_abs_mean=maxmean,max_std_error=maxsd,max_weight_l1_error=maxl1,fallback_reasons=reasons)
    result[cell] = counts
atomic_json(OUT/'ACTUAL_MATH.json',{'status':'PASS','checked':sum(v['answers'] for v in result.values()),
    'total':6190,'bundle_sha256':bundle_hash,'script_sha256':file_hash(__file__),'cells':result,
    'fallback_note':'all local arm fallback score curves are identical; per-arm source thresholds can yield different decisions'})
print(json.dumps(result,sort_keys=True),flush=True)
