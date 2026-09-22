"""Reviewer for the amended depth suite: the v2 reviewer plus replay of the logit-input variant.

Delegates every check to review_rbm_literature_completion_v2 (saved-state replay, separate metric
arithmetic, fold-threshold checks). Adds an independent replay for models of type ``stacked_logit``.
Writes SMOKE_REVIEW.json / RESULT_REVIEW.json and RUN_STATE.json under depth_amended/.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_rbm_literature_completion as run  # noqa: E402
from scripts import review_rbm_literature_completion_v2 as review  # noqa: E402
from spectral_utils import rbm_literature_completion as model  # noqa: E402

run.SUITES = tuple(run.SUITES) + ('depth_amended',)
_original_replay = review.replay_one


def replay_one(data, arrays, info, independent=False):
    _, uid, spans, anchor, banks = data
    plain = dict(info, models={k: v for k, v in info['models'].items() if v['type'] != 'stacked_logit'})
    checks = _original_replay(data, arrays, plain, independent)
    for key, d in info['models'].items():
        if d['type'] != 'stacked_logit':
            continue
        bank = int(key.split('_')[0][1:])
        x = banks[bank]['x']
        p = x.shape[1]
        get = lambda name: arrays[key + '::' + name]  # noqa: E731
        _, w, b = model.unpack(get('first'), p, 4)
        hidden = (x @ w + b) * get('firstsign')
        keep = get('keep').astype(bool)
        z = (hidden[:, keep] - get('mean')[keep]) / get('scale')[keep]
        assert z.shape[1] >= 3
        _, w1, b1 = model.unpack(get('theta'), z.shape[1], 1)
        logit = ((z @ w1 + b1) * get('signs'))[:, 0]
        post = expit(logit)
        for s, token in (('logit', logit), ('posterior', post)):
            step = np.array([np.mean(sorted(token[a:b_], reverse=True)[:10]) for a, b_ in spans])
            np.testing.assert_allclose(step, arrays['score::' + key + '_' + s], atol=2e-10, rtol=2e-10,
                                       err_msg=uid + ':' + key + ':' + s)
            checks += 1
    return checks


review.replay_one = replay_one

if __name__ == '__main__':
    review.main()
