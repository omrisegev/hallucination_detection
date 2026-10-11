"""Numerical tie policy: arithmetic roundoff must not defeat earliest-mode ties."""
import numpy as np

def earliest_mode(mass):
    mass=np.asarray(mass,float)
    # PAVA differences of equal rational masses can differ by a few ulps.
    # This tolerance handles arithmetic equality, not a preference for early steps.
    tolerance=8*np.finfo(float).eps*max(1.,float(np.max(abs(mass))))
    return int(np.flatnonzero(mass>=mass.max()-tolerance)[0])
