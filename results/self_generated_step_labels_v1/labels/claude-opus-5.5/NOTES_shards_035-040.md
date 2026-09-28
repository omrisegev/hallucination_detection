# Notes: judge claude-opus-5.5, shards 035-040

Start: 2026-09-29 00:44 (local). End: 2026-09-29 00:53 (local).

Suspected reference errors (candidate judged on the mathematics, reference_suspect = true):
- J0883: the reference uses 10*1.5 for the third race instead of 1.5*11 = 16.5. The net average loss should be 3.5, not 3.
- J0888: the reference sums Julie's 10 instead of Sasha's 14. Sasha's total is 18, not 14.
- J0921: the reference adds 400 for Sarah instead of $300. The total is 2180, not 2280.
- J1008: the reference solution validates both x = -2 and x = 1, but its final answer lists only 1.

Ambiguous wording (labelled against the reference's reading, low confidence):
- J0990: "six groups of equal size" vs groups of six. J0998: "half of what is left is sold equally". J0930: simple vs compound depreciation. J0933: "three times more". J0944: gives the breakdown 4 blue + 6 red instead of the total 10; treated as matching.

Items where the correct final answer rests on a false or unjustified intermediate statement (step flagged although the answer matches):
- J0893 step 12, J0927 step 5, J0948 step 2, J0949 step 2 ("or" instead of "and"), J0977 step 29.
- J0902 step 0 states "0 days left", which is not used later. It is flagged as the first error, ahead of the step-4 arithmetic error.

Split and localization choices: J0909 (j-component sign error, spread over steps 4-5; step 4 chosen), J0947 (step 0's "time to heat up to 400" label vs step 1's use of it; step 1 chosen), J0961 (the plan in step 0 omits the sour oranges; step 0 chosen).
