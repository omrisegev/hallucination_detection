# Uniform multiscale Renyi/varentropy fusion v1

Status: **COMPLETE / REVIEW PASS** on the frozen 13,769-answer,
145,597-step development population. This is adaptive development selection,
not untouched confirmation.

## Result in one sentence

The frozen benchmark-uniform rule selects the four-view q15 raw fusion after
per-view Top10 readout. q50 improves PRMB only slightly at the bank level while
losing 0.634 ProcessBench points, q15+q50 does not dominate q15, fixed supervised
simplex weights are worse, and preserving the answer-level feature mean remains
necessary for calibration.

## Uniform results

All rows use one definition across all benchmark cells. `Fold pooled` is the
mean of five held-fold pooled AUROCs; `OOF pooled` is the descriptive
concatenation and is not the primary cross-fitted ranking summary.

| Bank / weighting | PB all-8 | PB raw exact | PRMB within | Fold pooled | OOF pooled | PRMScore |
|---|---:|---:|---:|---:|---:|---:|
| **q15 raw equal — selected** | **36.6201%** | 33.1607% | .753436 | .722708 | .722305 | .634412 |
| q15 global-scale equal | 36.6362% | **33.7010%** | .749026 | .718537 | .718092 | .631668 |
| q15 supervised simplex | 36.2965% | 33.1157% | .739728 | .711479 | .711022 | .625976 |
| q50 raw equal | 35.9864% | 32.9131% | **.754843** | **.726011** | **.725580** | .635939 |
| q50 global-scale equal | 36.2258% | 32.8456% | .752030 | .723943 | .723461 | .635755 |
| q50 supervised simplex | 36.1495% | 32.8906% | .749773 | .721798 | .721225 | .635240 |
| q15+q50 raw equal | 36.2524% | 33.0482% | .753748 | .724872 | .724443 | **.636436** |
| q15+q50 global-scale equal | **36.6745%** | 33.5885% | .751616 | .722371 | .721917 | .634813 |
| q15+q50 supervised simplex | 36.1495% | 32.8906% | .749773 | .721798 | .721225 | .635240 |
| Previous q15 fusion-before-Top10 reference | 36.1674% | 32.6655% | .751115 | .721815 | .721407 | .634805 |

The `scale_natural` controls reproduce raw-equal within-answer ordering and PB
metrics exactly, as registered. Their calibration values differ only at the
fourth/fifth decimal because PRMScore calibration uses separately nested scale
contexts.

## Frozen primary contrasts

Intervals are family-wise 99.375% whole-source-group bootstrap intervals.
ProcessBench deltas are shown in percentage points.

| Contrast | PB delta [interval] | PRMB-within delta [interval] | Conclusion |
|---|---:|---:|---|
| q50 raw - q15 raw | -0.634 [-1.662, +0.366] | +.001407 [-.001390, .004207] | No universal q50 gain |
| q15+q50 raw - q15 raw | -0.368 [-1.192, +0.407] | +.000312 [-.001497, .002100] | Multiscale raw adds no clear value |
| q50 scale-equal - q15 scale-equal | -0.410 [-1.475, +0.683] | +.003003 [.000066, .005990] | q50 helps PRMB but not both axes |
| multiscale scale-equal - q15 scale-equal | +0.038 [-0.741, +0.827] | +.002590 [.000567, .004588] | Gain only versus the weaker scaled q15 control |
| q15 simplex - q15 scale-equal | -0.340 [-1.117, +0.416] | -.009298 [-.011372, -.007258] | Supervised simplex rejected |
| q50 simplex - q50 scale-equal | -0.076 [-0.868, +0.698] | -.002256 [-.004422, -.000099] | Supervised simplex rejected |
| multiscale simplex - scale-equal | -0.525 [-1.187, +0.121] | -.001843 [-.003505, -.000078] | Supervised simplex rejected |
| multiscale full - centered | 0 [0, 0] | 0 [0, 0] | Exact ordering invariance |

## Answers to the four questions

### 1. q15 versus q50 under one rule

q50 is not the uniform replacement. Its raw four-view bank gains only .001407
PRMB within AUROC over q15, with a corrected interval crossing zero, while its
ProcessBench point estimate is 0.634 points lower. The predeclared normalized
worst-regret rule therefore selects `q15_raw_equal` once for every benchmark.

The best individual PB row is multiscale scale-equal at 36.6745%, only 0.0545
points above the selected q15 row. The best individual PRMB-within row is q50
raw at .754843, only .001407 above q15. q15 is the smallest common compromise;
this is not a claim that it wins each metric separately.

### 2. Should both supports be retained?

Not in the fixed bank. Raw q15+q50 changes PB by -0.368 points and PRMB within
by only +.000312 versus q15; both corrected intervals cross zero. Scaling all
eight views improves PRMB relative to scaled q15, but not relative to the
stronger q15 raw candidate. There is no evidence that duplicating the same four
definitions at two supports gives a better uniform locator.

The error sets are not identical: versus q50 raw, q15 uniquely localizes 174 PB
errors and q50 uniquely localizes 163. On 6,030 valid PRMB answers q15 is better
on 1,319 and q50 on 1,525, with 3,186 ties. This complementarity is real but the
tested static average does not exploit it.

### 3. Are natural scales nuisance or useful weights?

They are useful in this representation, though they do not constitute balanced
fusion. In the selected q15 raw score, the approximate standardized-coordinate
shares are 61.76% `VE0`, 30.02% `H0lim`, 5.48% `VE0.75`, and 2.73% `VE1`.
Removing those scale ratios with global scale-equal preprocessing reduces PRMB
within by .004409; its descriptive 95% interval is [.002219,.006632], while the
PB change is negligible and uncertain.

At q50, raw natural scaling similarly beats scale-equal by .002813 PRMB within
(95% interval [.001306,.004354]). Thus the raw result is not an equal-expert
success: it is a reproducible low-alpha-heavy weighting that is useful for the
current development population and must be frozen before model transfer.

### 4. Does a sparse supervised simplex solve the weighting problem?

No. The five outer q15 fits stably assign about 22.2% to `VE0.75` and 77.8% to
`VE1`, setting `H0lim` and `VE0` to zero. q50 assigns about 72.7% to `VE0.75`
and 27.3% to `VE1`. The eight-view fit sets every q15 feature and the two q50
low-alpha features to zero, becoming numerically the q50 simplex arm.

Although this lowers its class-balanced training BCE, it worsens the actual
within-answer AUROC and exact-error objectives. The tested global step-BCE
surrogate therefore selects the wrong fixed mixture for localization. This
rejects this fixed supervised simplex objective, not all time-varying or
conditional selectors.

## Answer-level versus local channel

Centering the simplex scores inside every answer leaves PB and PRMB-within
identical by construction, but removes cross-answer calibration:

| Bank | Full-minus-centered fold pooled | Full-minus-centered PRMScore |
|---|---:|---:|
| q15 | +.011578 | +.009610 |
| q50 | +.005997 | +.007539 |
| q15+q50 | +.005997 | +.007539 |

The architecture should therefore retain both the answer-level mean and local
deviation. A centered local channel is safe only if the answer-level channel is
carried separately and recombined; it must not replace the full score.

## Top10 placement

Applying Top10 separately to each view and then fusing improves q15 PRMB-within
by .002321 over the previous fusion-before-Top10 implementation (descriptive
95% interval [.001509,.003171]). PB is +0.453 points with an interval crossing
zero; PRMScore is essentially unchanged/slightly lower (.634412 versus .634805).
The agreed Top10 readout is therefore retained before fixed feature fusion.

## Decision and next step

- `SELECT_Q15_RAW_EQUAL_UNIFORMLY_ON_DEVELOPMENT`
- `REJECT_BENCHMARK_SPECIFIC_Q15_Q50_SELECTION`
- `DO_NOT_ADD_STATIC_Q15_Q50_DUPLICATION`
- `RETAIN_NATURAL_SCALE_RATIOS_AS_FROZEN_WEIGHTS`
- `REJECT_FIXED_GLOBAL_STEP_BCE_SIMPLEX`
- `PRESERVE_ANSWER_LEVEL_MEAN_ALONGSIDE_LOCAL_DEVIATION`

The next ordered experiment may optimize the answer-error gate, beginning with
Top10 readouts, while holding the selected q15 locator representation fixed.
The selected feature rule still requires frozen new-model confirmation.

## Integrity record

- 6/6 mechanism tests and a 360-answer all-cell/all-fold real-data smoke pass.
- Full coverage: 13,769 answers and 145,597 official steps.
- All 15 shared outer/nested models fit; 41,645 score contexts were reviewed.
- No held source group entered a scale or supervised weight fit.
- q15 replays exactly; saved q50 views replay within `6.22e-15`.
- OOF scores were frozen before aggregate evaluation at SHA256
  `2e87a3d11f4ee77594b64775457b1e99e864efb7d5eaca8350f9688b4adc14f4`.
- No package installation, GPU inference, Drive mutation, commit or push.

