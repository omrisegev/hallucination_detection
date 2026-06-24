import numpy as np
from spectral_utils.feature_utils import extract_all_features, FEAT_NAMES

def test_features():
    print("Testing new features implementation...")
    
    # 1. Sine wave + noise (standard case)
    t = np.linspace(0, 10, 100)
    ents = np.sin(t) + np.random.normal(0, 0.1, 100)
    feats = extract_all_features(ents)
    
    if feats is None:
        print("FAILED: extract_all_features returned None for len=100")
        return

    print("\nSine + Noise (len=100):")
    for name in FEAT_NAMES:
        val = feats.get(name)
        print(f"  {name:20}: {val}")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            print(f"FAILED: {name} is {val}")

    # 2. Short trace (edge case)
    ents_short = np.array([1.0, 2.0, 1.5, 2.5])
    feats_short = extract_all_features(ents_short)
    print("\nShort Trace (len=4):")
    if feats_short is None:
        print("  Correctly returned None for trace < min_len (8)")
    else:
        print("  ERROR: Should have returned None for trace < 8")

    # 3. Constant signal (edge case for variance/std)
    ents_const = np.ones(50) * 1.5
    feats_const = extract_all_features(ents_const)
    print("\nConstant Signal (len=50):")
    for name in ["pe_min", "pe_mean", "hurst_exponent", "cusum_max", "cusum_shift_idx"]:
        val = feats_const.get(name)
        print(f"  {name:20}: {val}")
        if val is not None and isinstance(val, float) and np.isnan(val):
             print(f"FAILED: {name} is NaN")

    # 4. Regime shift signal
    # First 50: random noise around 1.0
    # Next 50: random noise around 2.0
    ents_shift = np.concatenate([
        np.random.normal(1.0, 0.1, 50),
        np.random.normal(2.0, 0.1, 50)
    ])
    feats_shift = extract_all_features(ents_shift)
    print("\nRegime Shift (len=100):")
    print(f"  cusum_max          : {feats_shift['cusum_max']}")
    print(f"  cusum_shift_idx    : {feats_shift['cusum_shift_idx']} (Expected near 0.5)")

if __name__ == "__main__":
    test_features()
