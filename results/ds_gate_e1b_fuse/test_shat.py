"""Unit checks for votes_e1b (run before the real E1b run)."""
import numpy as np
import votes_e1b as VB

rng = np.random.default_rng(1)
n, m = 20000, 7
y = rng.random(n) < .3
# conditionally independent voters given y
Xi = np.column_stack([np.where(rng.random(n) < np.where(y, .8, .2), 1., -1.) for _ in range(m)])
# same plus a shared nuisance factor that pushes 4 voters together
u = rng.random(n) < .5
Xd = Xi.copy()
for j in range(4): Xd[:, j] = np.where(rng.random(n) < .5, np.where(u, 1., -1.), Xi[:, j])
s_ind = VB.shat(*VB.moments(Xi)[1:]); s_dep = VB.shat(*VB.moments(Xd)[1:])
print('S-hat independent', round(s_ind, 4), 'dependent', round(s_dep, 4)); assert s_dep > 5 * s_ind

# incremental update inside fuse_taus equals a full recomputation: rerun with a tiny problem and compare
Z = rng.normal(size=(3000, 5)) + y[:3000, None] * 1.0; act = np.ones_like(Z, bool)
q, s0, s1, tr = VB.fuse_taus(Z, act)
V, _ = VB._votes(Z, act, q)
full = VB.shat(*VB.moments(np.where(V > 0, 1., -1.))[1:])
print('final S-hat from search', round(s1, 6), 'recomputed', round(full, 6), 'q', q); assert abs(full - s1) < 1e-9
assert all(b <= a + 1e-12 for a, b in zip(tr, tr[1:]))
print('PASS')
