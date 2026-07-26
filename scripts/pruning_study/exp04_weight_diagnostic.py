"""
Experiment 4 - Where do the label-free weights actually go wrong?

Our detector has to guess how much to trust each measurement using structure
alone. A model trained on answer keys learns those trust levels directly, and
scores about 1.45 points higher. This experiment measures WHERE the guess goes
wrong, so that any attempted repair is aimed at a real failure rather than a
guessed one.

Three things are measured, per test set:

  1. Does the "everything is a noisy reading of one hidden thing" picture hold?
     If a second hidden thing is present, the whole weighting recipe is built on
     a premise that does not apply.
  2. How redundant are the measurements? Near-duplicates break the assumption
     the weighting recipe needs.
  3. Do the guessed trust levels agree with the learned ones - in RANK
     (does it order measurements correctly), in SIGN (does it get the direction
     right), and at the TOP (does it agree on which few matter most)?

Writes results/pruning_study/04_weight_diagnostic/
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as S                                          # noqa: E402


def learned_weights(V, y):
    """Trust levels a model learns directly from answer keys."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0))
    pipe.fit(V, y)
    return pipe.named_steps["logisticregression"].coef_.ravel()


def guessed_weights(cell, cols):
    """Trust levels our detector infers from structure alone - the effective
    per-measurement coefficient in the final combined score."""
    _, meta = S.fuse_meta(cell, cols)
    m = len(cols)
    w = np.zeros(m)
    cross = np.asarray(meta["cross_weights"], float)
    for gi, (idx, wg) in enumerate(meta["group_weights"]):
        cw = cross[gi] if gi < len(cross) else 1.0
        w[np.asarray(idx, int)] = np.asarray(wg, float) * cw
    return w


def main():
    out = S.outdir("04_weight_diagnostic")
    cells = S.load()
    S.validity_check(cells)

    rows, feat_rows = [], []
    for ck, cell in cells.items():
        V, y = cell["V"], cell["labels"]
        p = V.shape[1]
        cols = list(range(p))
        R = np.cov(V.T)
        Roff = R - np.diag(np.diag(R))

        ev = np.linalg.eigvalsh(Roff)
        ev = ev[np.argsort(-np.abs(ev))]
        second_vs_first = abs(ev[1]) / abs(ev[0])
        top_share = ev[0] ** 2 / np.sum(ev ** 2)

        d = np.sqrt(np.diag(R))
        C = R / np.outer(d, d)
        iu = np.triu_indices(p, 1)
        offdiag = np.abs(C[iu])

        w_learn = learned_weights(V, y)
        w_guess = guessed_weights(cell, cols)

        rank = spearmanr(np.abs(w_learn), np.abs(w_guess)).statistic
        # The combined score carries ONE global +/-1 ambiguity, resolved later
        # against the anchor measurement. Comparing raw signs would charge that
        # single bit to every measurement, so resolve it first and report both.
        raw_sgn = float(np.mean(np.sign(w_learn) == np.sign(w_guess)))
        sgn_match = max(raw_sgn, 1.0 - raw_sgn)
        if raw_sgn < 0.5:
            w_guess = -w_guess
        try:
            lin = pearsonr(w_learn / (np.abs(w_learn).max() + 1e-12),
                           w_guess / (np.abs(w_guess).max() + 1e-12)).statistic
        except Exception:
            lin = np.nan
        top_learn = set(np.argsort(-np.abs(w_learn))[:5])
        top_guess = set(np.argsort(-np.abs(w_guess))[:5])
        overlap = len(top_learn & top_guess)
        conc = float(np.sum(np.sort(np.abs(w_guess))[-3:]) /
                     (np.sum(np.abs(w_guess)) + 1e-12))

        rows.append({
            "test_set": S.plain_cell(ck), "test_set_code": ck,
            "answers": int(V.shape[0]),
            "second_factor_vs_first": float(second_vs_first),
            "share_explained_by_one_factor": float(top_share),
            "largest_correlation_between_measurements": float(offdiag.max()),
            "pairs_above_0.75": int((offdiag > 0.75).sum()),
            "rank_agreement_guessed_vs_learned": float(rank),
            "sign_agreement": sgn_match,
            "sign_agreement_before_global_flip": raw_sgn,
            "global_flip_applied": bool(raw_sgn < 0.5),
            "shape_agreement": float(lin),
            "top5_overlap_out_of_5": overlap,
            "top3_weight_concentration": conc,
        })
        for j in range(p):
            feat_rows.append({
                "test_set": S.plain_cell(ck), "test_set_code": ck,
                "measurement": S.plain(cell["pool"][j]),
                "measurement_code": cell["pool"][j],
                "learned_weight": float(w_learn[j]),
                "guessed_weight": float(w_guess[j]),
            })
        print(f"  {ck[:32]:32s} 2nd/1st={second_vs_first:.3f} "
              f"rank={rank:+.3f} sign={sgn_match:.2f} top5={overlap}/5")

    S.save_csv(os.path.join(out, "weight_diagnostic_per_test_set.csv"), rows)
    S.save_csv(os.path.join(out, "weights_per_measurement.csv"), feat_rows)

    arr = lambda k: np.array([r[k] for r in rows], float)          # noqa: E731
    med_2nd = float(np.median(arr("second_factor_vs_first")))
    med_share = float(np.median(arr("share_explained_by_one_factor")))
    med_rank = float(np.median(arr("rank_agreement_guessed_vs_learned")))
    med_sign = float(np.median(arr("sign_agreement")))
    med_top = float(np.median(arr("top5_overlap_out_of_5")))
    med_conc = float(np.median(arr("top3_weight_concentration")))

    ch1 = S.bar_chart(
        [r["test_set"] for r in sorted(rows, key=lambda r: -r["second_factor_vs_first"])],
        [r["second_factor_vs_first"] for r in sorted(rows, key=lambda r: -r["second_factor_vs_first"])],
        "Strength of a SECOND hidden factor, relative to the first "
        "(0 = the one-factor picture holds exactly)",
        value_fmt="{:.3f}", bar_h=22)
    ch2 = S.scatter_chart(
        arr("rank_agreement_guessed_vs_learned"), arr("sign_agreement"),
        "Rank agreement: does the guess order measurements like the learned model?",
        "Sign agreement: does the guess get each direction right?")

    tbl = S.html_table(
        ["Test set", "Answers", "2nd factor / 1st", "Share in one factor",
         "Largest correlation", "Pairs > 0.75", "Rank agree", "Sign agree",
         "Top-5 overlap"],
        [[r["test_set"], f"{r['answers']:,}",
          f"{r['second_factor_vs_first']:.3f}",
          f"{r['share_explained_by_one_factor']:.1%}",
          f"{r['largest_correlation_between_measurements']:.3f}",
          r["pairs_above_0.75"],
          f"{r['rank_agreement_guessed_vs_learned']:+.3f}",
          f"{r['sign_agreement']:.2f}", f"{r['top5_overlap_out_of_5']}/5"]
         for r in sorted(rows, key=lambda r: -r["second_factor_vs_first"])],
        numeric_cols=(1, 2, 3, 4, 5, 6, 7, 8))

    body = f"""
<h2>The gap being diagnosed</h2>
<p>Using the same 30 measurements and the same directions, a model trained on
answer keys scores <b>0.7809</b> while simply averaging them scores
<b>0.7664</b>. That <b>1.45 point</b> difference is purely the value of knowing
how much to trust each measurement. Our detector has to guess those trust levels
from structure alone. This experiment measures how the guess fails.</p>

<h2>1. Does the "one hidden cause" picture hold?</h2>
<p>The whole weighting recipe assumes every measurement is a noisy reading of a
single hidden thing &mdash; whether the answer is right. If that were exactly
true, the relationships between measurements would be explainable by one factor
and a second factor would be absent.</p>
{ch1}
<ul>
<li>Median strength of a second factor relative to the first:
<b>{med_2nd:.3f}</b>. A single factor explains a median of
<b>{med_share:.0%}</b> of the structure.</li>
<li><b>The premise is only approximately true.</b> There is real structure the
one-factor picture does not capture &mdash; which is exactly the room a better
estimator would have to exploit.</li>
</ul>

<h2>2. How redundant are the measurements?</h2>
<p>Every test set contains near-duplicate measurements &mdash; largest
correlation between {min(arr('largest_correlation_between_measurements')):.3f}
and {max(arr('largest_correlation_between_measurements')):.3f}, with
{int(min(arr('pairs_above_0.75')))}&ndash;{int(max(arr('pairs_above_0.75')))}
pairs above 0.75. Near-duplicates are precisely what breaks the assumption the
weighting recipe needs, and they are also what Experiment 2's localizer finds
sitting in the worst-fitting group.</p>

<h2>3. Where the guessed trust levels differ from the learned ones</h2>
{ch2}
{tbl}
<ul>
<li><b>Rank agreement</b> (does the guess order measurements the same way?):
median <b>{med_rank:+.3f}</b>.</li>
<li><b>Sign agreement</b> (does it get each measurement's direction right?):
median <b>{med_sign:.2f}</b>. This is measured <em>after</em> resolving the single
global flip that the combined score carries anyway &mdash; charging that one bit
to all 30 measurements would have understated agreement.</li>
<li><b>Top-5 overlap</b> (do they agree on which few matter most?): median
<b>{med_top:.0f}/5</b>.</li>
<li><b>Weight concentration</b>: the top three measurements carry a median
<b>{med_conc:.0%}</b> of the total guessed weight.</li>
</ul>

<h2>What this points at</h2>
<p>The failure is not one thing. Read the three columns together:</p>
<ul>
<li>Sign agreement near {med_sign:.2f} means direction errors are
{'a substantial part of the problem' if med_sign < 0.75 else 'not the main problem'}.</li>
<li>Rank agreement of {med_rank:+.3f} means the ordering is
{'largely wrong' if med_rank < 0.2 else 'partly right'} &mdash; the guess and the
learned model do not agree about which measurements matter.</li>
<li>A second factor at {med_2nd:.2f} of the first means repairs that assume a
single clean factor are working against the data, and the honest options are
either to model a second factor or to remove the measurements that create it.</li>
</ul>
<p class="note">This experiment deliberately produces no winner and no gate. Its
only job is to say which repair is worth attempting.</p>

<h2>Saved data</h2>
<ul>
<li><code>weight_diagnostic_per_test_set.csv</code> &mdash; all diagnostics per test set</li>
<li><code>weights_per_measurement.csv</code> &mdash; the guessed and learned trust level
for every measurement in every test set, for direct inspection</li>
</ul>
"""
    S.write_page(
        os.path.join(out, "index.html"),
        "Experiment 4 - Where the label-free weights go wrong",
        "Diagnosing the 1.45-point gap between guessed and learned trust levels, "
        "before attempting any repair.",
        [f"The one-hidden-cause premise is only approximate: a second factor sits at "
         f"<b>{med_2nd:.2f}</b> of the first, and one factor explains a median "
         f"{med_share:.0%} of the structure.",
         "Every test set contains near-duplicate measurements (largest correlation "
         "0.99-1.00), which is exactly what breaks the weighting recipe.",
         f"Guessed vs learned trust levels: rank agreement <b>{med_rank:+.3f}</b>, "
         f"sign agreement <b>{med_sign:.2f}</b>, top-5 overlap <b>{med_top:.0f}/5</b>.",
         "No winner is declared here - this experiment exists to aim the repair, "
         "not to pick one."],
        body)
    print("\nExperiment 4 complete.")


if __name__ == "__main__":
    main()
