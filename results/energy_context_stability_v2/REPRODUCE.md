# Reproduce the real-bank label-free stability diagnostic

This is an all-answer landmark diagnostic, not a full-token quality evaluation.
The fixed protocol and its numerical amendment are under docs/experiments.

From the repository root, with the dependencies already used by the temporal
research environment (NumPy/SciPy/PyTorch/threadpoolctl):

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m unittest tests.test_energy_context_stability tests.test_cca_iu_isolation
python -c "from scripts import run_energy_context_stability as r; r.OUT=r.ROOT/'results/energy_context_stability_replay'; r.main()"
python -c "from scripts import run_energy_context_stability as r; r.OUT=r.ROOT/'results/energy_context_stability_replay'; from scripts import report_energy_context_stability as report; report.main()"
```

Required local inputs: results/temporal_context_data_v1/{MANIFEST.json,
METADATA.json,features.npy}. Exact hashes are in PREPARED.json. The runner checks
them before fitting. Do not substitute another cache because names match.
No correctness-label file is required or opened. The existing bundle can be
reconstructed through the prior temporal context data preparation workflow.

Git contains the protocol, implementation, fit metadata, summaries, independent
audits, source-group NLL aggregates, report and SHA256 inventories. Large
LANDMARKS.npz and per-fit DIAGNOSTICS.npz remain local generated artifacts; their
hashes are retained. For reproduction in a fresh checkout, remove NO source
artifacts: save or use a separate output directory for a fresh rerun. A summary
checkout without its large checkpoints cannot be resumed as though complete.
The runner deliberately rejects missing/mismatched completed-fit artifacts.

v1 is a preserved partial numerical run. NUMERICAL_ANALYSIS.json and
QP_REGRESSION.npz document the boundary-case solver correction. The full v2 run
recomputed all45 fits. NUMERICAL_REVISION_AUDIT.json verifies33 old completed
fits, with bitwise-identical non-simplex arrays. Its optional replay requires
the local v1 DIAGNOSTICS.npz archives:

```powershell
python scripts/audit_energy_numerical_revision.py
```

Do not describe feature NLL, stability ratios, or synthetic S0 AUC as PB or
PRMB error-localization improvements. The Step381 innovation5 quality reference
remains unchanged. Model fitting is label-free; historical bank selection used
development labels. The neural queue remains paused separately.
