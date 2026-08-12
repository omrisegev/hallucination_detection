# Atomic NRM candidate v1 — label-free structural audit

- Version: `atomic-neutral-residual-projector-cs-iu-candidate-v1-2026-08-13`
- Source telemetry: frozen 23-cell original roster; no correctness field was loaded.
- Atoms seen / eligible in every cell: 30 / 17
- Eligible atoms: `cusum_max, cusum_max_energy, cusum_max_spilled, epr, epr_energy, epr_spilled, logprob_margin, mean_logprob_entropy, mean_top1_logprob, min_energy, renyi_entropy_2, rpdi, sw_var_peak, sw_var_peak_energy, sw_var_peak_spilled, topk_tail_mass, varentropy`
- Excluded for incomplete source coverage: `cusum_shift_idx, dominant_freq, high_band_power, hl_ratio, hurst_exponent, low_band_power, min_spilled, pe_mean, spectral_centroid, spectral_entropy, stft_max_high_power, stft_spectral_entropy, trace_length`
- Permutation-null simultaneous interval: [0.934489, 1.070026]
- Neutral dimension: 2
- Retained eigenvalues: `0.960685, 1.025557`
- Symmetric-anchor retained norm: 0.456515
- Leave-one-cell direction |cosine|: min 0.975505, median 0.994124, max 0.999434
- Minimum relative IU weight across source cells: 0.00531307; no atom was numerically inactive.
- Fixed correction scale: `1/sqrt(p)` standard deviations, with p=17.
- Direction SHA-256: `d7de9faeb68825ac540cbaa70868aeb52dcf548d6b7e11e480664f446e952edb`
- Covariance SHA-256: `9ef9010eb9f7969831603db2ff0a484c77c1b05b9455901e0e83449b707ca3ed`

This audit establishes only null geometry and affine/invariance properties.  It
does not identify a hallucination-target direction and reports no AUROC.
