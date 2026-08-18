"""
Marks `cluster/` as a package so the offline smoke gate can import the drivers as modules
(`scripts/smoke_paper_exact_drivers.py`) instead of exec-ing them from a path.

This does not change how the drivers run on the cluster: each one puts the repo root on
`sys.path` itself, and `run_inference.py` additionally puts `cluster/` on `sys.path` so its
`import presets` still resolves as a top-level module.
"""
