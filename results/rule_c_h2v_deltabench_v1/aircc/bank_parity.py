"""Builder-environment parity on cycle4: FEATURES.npz rebuilt from the stored H2V C grafted.pkl must equal the stored
bank_v3 FEATURES.npz (names, uids, offsets, digit_active exact; values allclose 1e-9). No labels.
    python bank_parity.py NEW_FEATURES.npz STORED_FEATURES.npz"""
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
ok = all(np.array_equal(a[k], b[k]) for k in ('names', 'uids', 'offsets', 'digit_active'))
d = float(np.nanmax(np.abs(a['values'] - b['values']))); nan_eq = bool(np.array_equal(np.isnan(a['values']), np.isnan(b['values'])))
ok = ok and nan_eq and d <= 1e-9
print('BANK PARITY', 'PASS' if ok else 'FAIL', 'max_abs_diff', d, 'nan_pattern_equal', nan_eq); sys.exit(0 if ok else 6)
