# Handoff: from the view-selection searches to a defensible feature set (2026-10-26)

## For Omri, in short (Hebrew)

חיפשנו בשלוש ריצות (שלבים 0, 1 ו-2) איזה סט קטן של ערוצי telemetry עוזר ל-PRM, ואיזה עובד גם בלי ה-PRM.
- **עם ה-PRM:** אין סט אחד שמנצח בכל 10 התאים.
- **בלי ה-PRM:** יש 380 סטים שמנצחים את הממוצע של עצמם בכל 10 התאים, אבל אף אחד מהם לא מנצח את ה-PRM ב-Socratic.
- **בשני המצבים:** סט אחד עובד גם עם ה-PRM וגם בלעדיו, אבל רק ב-8 תאים בלי Math-Shepherd.

השיחה הבאה צריכה לעבור על הממצאים ולבחור את הסט שהכי קל להצדיק מתמטית ואלגוריתמית. כלומר: לנסח את ההיגיון
שמסביר למה דווקא הערוצים האלה ("reverse engineering"), ולבדוק את ההיגיון הזה על נתונים שלא נבחרו עליהם.

**Lesson about communication (Omri, 2026-10-26: "אני לא מצליח להבין אותך", i.e. "I can't understand you").** The
previous session's answers were too dense. Answer in plain Hebrew:
- one idea per paragraph;
- the intuition before any number;
- at most one small table per answer;
- define every term the first time it appears.

See CLAUDE.md, "Communication style".

## 1. Goal of the next session

Omri: "תעבור על הממצאים ותחקור סביב איזה סט פיצ׳רים יהיה לנו הכי קל לבסס את הטיעון שלנו מבחינה מתמטית ריגורוזית
ואלגוריתמית, לתת reasoning לבחירת הפיצ׳רים שבחרנו (קצת reverse engineering)".

Translation: review the findings and find the feature set for which our argument is easiest to ground with
mathematical and algorithmic rigour, and give the reasoning for the features we chose (a bit of reverse engineering).

Concretely:
1. Choose one feature set, or two (one with the PRM, one standalone), from the candidates below.
2. Write a mathematical and algorithmic argument for why these channels and not others. Ground it in:
   - the L-SML model's assumptions (Jaffe, Fetaya and Nadler 2016);
   - the PRM's measured failure modes;
   - the channel definitions.
3. Turn that argument into a rule that can be applied without labels, and check whether the rule picks the same set.
4. Test the frozen set and rule on data not used for selection. Hard2Verify has N16 channels for all 5 PRMs; it is
   small (177 answers) and CPU-only. ProcessBench may also have channels (`results/prm_boost_processbench_v1/`) —
   verify. Qwen3's own answers would need GPU inference, which needs approval.

Two rules:
- **Reasoning written after seeing the results is a hypothesis, not evidence.** It becomes evidence only if a rule
  derived from it picks the set and the set holds on held-out data. Pre-register before any new run, as stages 0–2 did.
- **Ask Omri before running anything new.** The CLAUDE.md "ASK which method" rule applies.

## 2. What exists (read these first, in order)

| What | Where | Key fact |
|---|---|---|
| Stage 0 (descriptive) | `results/prm_view_selection_v1/stage0/README.md`, Step 478 | From PRMBench to Socratic the PRM improves and the telemetry degrades. Label-free CI estimators over-trust the telemetry consensus and, on Socratic, rank channels wrongly for RLHFlow and Math-Shepherd |
| Stage 1 (with PRM) | `stage1/README.md`, Step 479 | 0 of 12,247 valid subsets beat both the PRM and the same-set mean in all 10 cells; the best wins 9/10. Socratic is the bottleneck. Selection does not transfer across benchmarks or PRMs. The small Socratic wins depend on DS-EM's probability floor |
| Stage 1b (proposed, not run) | `docs/experiments/PRM_VIEW_SELECTION_STAGE1B_PROPOSAL_2026-10-26.md` | A PRM weight anchor plus position residualization (fork diagnosis: Math-Shepherd's PRM is position-driven) |
| Stage 2 (no PRM) | `stage2/README.md`, Step 480 | 380 telemetry-only subsets beat their own mean in all 10 cells; none beats the PRM on strong-PRM Socratic. The two runs' best subsets do not overlap |
| Competitors | `stage2/COMPETITORS.csv`, `figures/6_*` | Under CROP, ARES, PRISM and SLIDER protocols, Qwen-7B, in-sample |
| Charts | `stage2/figures/1-6` | |
| Search data | `stage1/SEARCH.npz`, `stage2/SEARCH.npz` | All 64,839 subsets × 10 cells. Bit i = channel i in the N16 order below |
| Method code | `spectral_utils/prm_anchored_lsml.py`, `stage1/stage1.py`, `stage2/stage2.py` | Light per-cell caches in `stage*/cache/` (untracked; rebuild with `python stage1.py cache` / `stage2.py cache`) |
| Math background | `docs/research_notes/math_rigor_2026-10-24/FRAMING_THEORY.md`, `GAPS_STATUS_HE.md`, `docs/research_notes/math_peer_review_2026-10-24/MATH_PEER_REVIEW_HE.md` | Latent-class framing; the PRM as the identifying anchor (Step 476) |
| Paper | `papers/extracted/unsupervised-ensemble-learning-with-dependent-classifiers.md` | "Classifier selection": under CI, take the most accurate classifiers; with groups, the most accurate ones from different groups. This is the cleanest published rule to build on |

The N16 channel order is: q15_H1, q15_VE1, chosen_surprisal, logprob_margin, true_tail50, energy_level,
energy_innovation, top15_turnover, top50_js, dominant_freq16, bocpd_p0, realized_z, realized_drv, digit_alternative,
digit_spread, digit_alternative_innovation.

## 3. Candidate sets

All of these were selected with the labels of PRMBench and Socratic, so they are in-sample.

| Id | Channels | With PRM (stage 1) | Without PRM (stage 2) |
|---|---|---|---|
| SUB5 | q15_H1, chosen_surprisal, top15_turnover, dominant_freq16, bocpd_p0, digit_spread | Beats PRM and mean in 8/8 cells without Math-Shepherd, but collapses on Socratic Math-Shepherd (−16). Socratic gains are tie-level | Beats its mean in 9/10 (fails PRMBench Math-Shepherd, −0.5) |
| SUB4 | q15_VE1, chosen_surprisal, energy_innovation, top15_turnover, realized_z, digit_alternative, digit_alternative_innovation | 4/10 cells; essentially the PRM on Socratic | 9/10 (fails Socratic Math-Shepherd, −1.0); most consistent on PRMBench |
| S2-best | q15_VE1, chosen_surprisal, logprob_margin, energy_innovation, top50_js | 6/10 | **10/10** against its own mean (min +1.62), but in absolute terms below TELE_N16 and the PRM |
| S1-9/10 | q15_VE1, chosen_surprisal, top15_turnover, dominant_freq16, bocpd_p0 | 9/10 (fails Socratic Math-Shepherd, −15.7) | 8/10 |

Patterns worth explaining:
- **chosen_surprisal is in every candidate,** and in 67–77% of the best subsets of both runs.
- **With the PRM, the best subsets favour realized_z and true_tail50.** Without it they favour energy_innovation and
  realized_drv, and realized_z appears in only 10% of stage-2 successes.
- **Near-duplicate channels exist:** digit_alternative / digit_alternative_innovation (residual partial correlation
  .997); also top15_turnover / top50_js and q15_H1 / logprob_margin.
- **The PRM's between-group weight** is 70–86% in the Socratic cells where fusion survives. Fusion "works" there by
  staying close to the PRM.

## 4. Known caveats that any argument must respect

- **Everything is development data.** Selection used PRMBench and Socratic labels. The PRMBench pool excludes the 758
  label-known control answers.
- **DS-EM's unregistered probability floor (1e-6)** decides the small Socratic wins with the PRM. The secondary counts
  break at 1e-4.
- **Math-Shepherd's PRM correlates +.47 to +.50 with step position.** On Socratic, the PRM-anchored orientation then
  flips 10/16 channels (fork diagnosis, `results/prm_view_selection_v1/position_probe/`, untracked). Dropping
  Math-Shepherd needs a stated reason and an amendment.
- **Omri's own label-free structural conditions do not predict success** in stage 1: correlation with the PRM above
  the mean correlation with the rest of the set (Spearman with the min-margin −.06), and low pairwise correlation
  (+.27).
- **A supervised LR on the same channels** beats the label-free fusion in most cells. Advisor Bracha suggested an LR
  on the PRM's training data.

## 5. Suggested approach (for discussion with Omri, not yet approved)

1. **Map each channel to what it measures.** Use the definitions in the feature code, e.g. `spectral_utils/` and the
   N16 bank definition in `results/monday_nested_banks_v2/BANKS.json`. Then map it to which PRM failure it could
   cover: PRMBench category, position, step length.
2. **Use the stage-0 per-channel estimates** (`stage0/BINARY_VIEWS.csv`: label-free accuracy, PRM-anchored loading,
   nuisance loading, CI misfit). They show which channels satisfy the L-SML model on both benchmarks. That is the
   mathematically defensible core.
3. **Write a rule, then check it.** Formulate a label-free rule from the 2016 paper's selection principle, i.e. one
   channel per dependency group, ranked by anchored accuracy. Then check whether the rule reproduces one of the
   candidates. If it does, the reasoning is not only post-hoc.
4. **Pre-register and test** the frozen set and rule on Hard2Verify (and ProcessBench, if channels exist) against the
   PRM, CHAIN_N16, TELE_N16 and the same-set mean.

## 6. Repository state

- **Branch** `claude/prm-boosting`, pushed. The last results commit is 621e987a (Step 480); this file is committed on
  top of it.
- **Untracked work from a fork session** (not committed by this session): `results/prm_view_selection_v1/competitors/`
  and `results/prm_view_selection_v1/position_probe/`.
- **Other sessions:**
  - "ביקורת מתמטית PRM boosting" (math rigor; its branch is merged up to Step 476);
  - a fork named "PRM boosting".
