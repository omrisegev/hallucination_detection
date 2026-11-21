"""Builder-environment parity on cycle4: FEATURES.npz rebuilt from the stored H2V C grafted.pkl vs the stored bank_v3.
Rule (v2, 2026-11-22): names, uids, offsets, digit_active and the NaN pattern exact; every column |diff| <= 1e-9 * max(1,
max |stored column|); the 16 TELE16 channels (first 11 columns + 5 step channels) |diff| <= 1e-12.
v1 used one absolute 1e-9 for all 56 columns and failed in job 372108 at 1.05e-9 on hist_entropy_rolling_hl_ratio
(values up to 8.7e4, relative 1.2e-14; not a TELE16 channel). The 11 token channels were bit-identical, the 5 step
channels <= 1.6e-15. No labels.
    python bank_parity.py NEW_FEATURES.npz STORED_FEATURES.npz"""
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
ok = all(np.array_equal(a[k], b[k]) for k in ('names', 'uids', 'offsets', 'digit_active'))
x, y = a['values'], b['values']; ok &= bool(np.array_equal(np.isnan(x), np.isnan(y)))
d = np.nan_to_num(np.abs(x - y)); scale = np.maximum(1., np.nanmax(np.abs(y), 0))
worst_rel = float((d.max(0) / scale).max()); ok &= worst_rel <= 1e-9
n = list(a['names']); tele = n[:11] + ['realized_z', 'realized_drv', 'digit_alternative', 'digit_spread', 'digit_alternative_innovation']
tele_max = float(max(d[:, n.index(c)].max() for c in tele)); ok &= tele_max <= 1e-12
print('BANK PARITY', 'PASS' if ok else 'FAIL', 'max_abs_diff', float(d.max()), 'worst_scaled', worst_rel, 'tele16_max_abs', tele_max)
sys.exit(0 if ok else 6)
