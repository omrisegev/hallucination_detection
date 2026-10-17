# Gate definition and protocol wording correction

The executable and frozen artifact used **whole-answer Tail15 Top10 mean**:

1. At token t, tail(t) = max(0, 1 - sum(exp(logp(t,r)), r=1..15)).
2. Answer score = mean of the largest min(10,N) tail values across the answer.
3. Rank answer scores within each PB model/dataset cell using midranks
   (average_rank - 1)/(cell_size - 1). Gate opens at rank >= .33.
4. If open, predict the fusion score's argmax step. Otherwise predict no error.

No digit IDs, correctness labels, or provided-token identity enter this gate.
The threshold was fixed in this run, but historically selected on development
outcomes. Cell rank calibration is transductive. PRMB within-answer AUC uses no
gate. All compared arms share the same gate, including historical references.

The frozen protocol incorrectly called the loaded `gate_raw` score
"answer-prominence". That term belongs to a different atlas feature which
subtracts the answer-wide mean. **No mean subtraction occurred in this run.**
The intended and executed artifact was consistently the frozen historical gate;
the driver and its hash, scores, threshold and all results are unchanged. Keep
the original protocol as a frozen record and this explicit wording correction.

Provenance: `scripts/run_temporal_research_baseline.py:100` computes the raw
Top10 mean and stores it in `SCORES_FROZEN.npz:gate_raw`. Both historical PB
anchors replay independently. `GATE_PROVENANCE_CHECK.json` also compares the
raw formula directly with frozen values on all 400 answers of one source cell;
this check is implementation verification, not a subset quality experiment.
