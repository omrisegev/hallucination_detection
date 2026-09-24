"""One answer, seven locked arms, explicit empty-step routing; no labels."""
import time
import numpy as np
from .fusion import ARMS, frozen_scores, local_scores, validate_telemetry
from .ct7 import step_scores

ALL_ARMS = ARMS + ('ct7',)


def score_answer(row, bundle):
    started = time.perf_counter()
    cpu_started = time.process_time()
    spans = np.asarray(row['step_token_spans'],int)
    valid = spans[:,1] > spans[:,0]
    if not valid.any(): raise ValueError('answer has no nonempty steps')
    matrix = validate_telemetry(row)
    scores = frozen_scores(matrix,spans[valid],bundle['fit'])
    local,diag = local_scores(matrix,spans[valid]); scores.update(local)
    scores['ct7'] = step_scores(row,spans[valid])
    output, predictions = {}, {}
    for arm in ALL_ARMS:
        expanded = np.full(len(spans),np.nan); expanded[valid] = scores[arm]
        output[arm] = [float(v) if np.isfinite(v) else None for v in expanded]
        decision = np.zeros(len(spans),int)
        decision[valid] = scores[arm] < bundle['thresholds'][arm]
        predictions[arm] = decision.tolist()
    return {'scores':output,'predictions':predictions,'nonempty':valid.tolist(),
            'local':diag,'tokens':len(matrix),'cpu_seconds':time.perf_counter()-started,
            'process_cpu_seconds':time.process_time()-cpu_started}
