"""Explicit PRMBench one-based annotation to zero-based array contract."""
import numbers
import numpy as np


def prm_error_flags(error_steps, n_steps):
    """Out-of-range annotations stay inert, as in the official step loop.

    Do not guess the index base from a row's values: a valid list need not
    contain either the first or the last step. ProcessBench has a different
    contract and must not pass through this conversion.
    """
    if not isinstance(n_steps,numbers.Integral) or isinstance(n_steps,bool) or n_steps<0:
        raise ValueError('INVALID_STEP_COUNT')
    flags=np.zeros(n_steps,dtype=np.int64)
    for step in error_steps:
        if not isinstance(step,numbers.Integral) or isinstance(step,bool):
            raise ValueError('PRMB_ANNOTATION_MUST_BE_INTEGER')
        if 1<=step<=n_steps:flags[step-1]=1
    return flags
