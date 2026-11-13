# GSP status review (2026-11-13)

Omri asked: did we continue with GSP although it did not deliver? Three review agents.
- HISTORY_AUDIT.md: two-week decision timeline, matched head-to-heads, component attribution.
- LITERATURE.md: cited benchmarks, published comparators, gaps for a paper.
- Fusion ablation (ablate.py; TABLES.md, BOOT_*.csv, RESULTS_*.csv, GATES.txt 72/72 pass): standalone TELE16 channels, no PRM, GAP flags for all arms, R0 and R1 input.
  F1 mean, F2 L-SML (= TELE16 on R0), F3 GSP-T, F4a GSP-within + equal across, F4b GSP-within + continuous rank-1 across,
  F5 L-SML-within + binary across, F6 = A2 readout calibration (min rank-1 misfit) then F1..F4.
  Paired bootstrap vs F2@R0, 2,000 draws. Per-step cells (99 MB) are not in git.
Run: COLLECTION_DATA=... python -B ablate.py <bench> (paths point to the scratch dir of the run; adjust OUT).
Summary (means of 4 cells): PRMBench best F2@R0 67.97; Socratic best F3@R0 66.07; Hard2Verify F2@R1 44.36 (no arm separable);
ProcessBench SLA best mean@R0 36.5, all binary cross-group arms lose (F3@R0 30.45).
