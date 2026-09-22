# Complementarity of the leading single views (post hoc, descriptive)

ProcessBench erroneous answers: 4442. The no-error gate is identical across arms, so only error peaks differ.

## PB exact-hit rate on erroneous answers

| arm | hits | rate |
|---|---:|---:|
| VE_0 | 1170 | 0.263 |
| VE_0.75 | 1235 | 0.278 |
| varentropy15 | 1189 | 0.268 |
| H0lim | 1170 | 0.263 |
| entropy | 1169 | 0.263 |

## Pairwise overlap of PB hits

| pair | both | only A | only B | neither | union rate | Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| VE_0 vs VE_0.75 | 828 | 342 | 407 | 2865 | 0.355 | 0.53 |
| VE_0 vs varentropy15 | 865 | 305 | 324 | 2948 | 0.336 | 0.58 |
| VE_0.75 vs varentropy15 | 928 | 307 | 261 | 2946 | 0.337 | 0.62 |
| H0lim vs VE_0.75 | 796 | 374 | 439 | 2833 | 0.362 | 0.49 |
| H0lim vs entropy | 1048 | 122 | 121 | 3151 | 0.291 | 0.81 |
| VE_0 vs H0lim | 1076 | 94 | 94 | 3178 | 0.285 | 0.85 |

Union of all five arms (oracle pick per answer): 0.410; intersection: 0.141.

## PB hit rate by first-error position (absolute)

| stratum | n | VE_0 | VE_0.75 | varentropy15 | H0lim | entropy | union | VE_0 early/late miss | VE_0.75 early/late miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step0 | 544 | 0.246 | 0.333 | 0.324 | 0.281 | 0.285 | 0.461 | 0.00/0.92 | 0.00/0.91 |
| step1 | 1054 | 0.313 | 0.299 | 0.304 | 0.308 | 0.313 | 0.454 | 0.18/0.74 | 0.22/0.70 |
| step2 | 984 | 0.273 | 0.257 | 0.264 | 0.263 | 0.266 | 0.407 | 0.30/0.61 | 0.33/0.61 |
| step3-5 | 1340 | 0.257 | 0.284 | 0.254 | 0.253 | 0.244 | 0.411 | 0.46/0.48 | 0.46/0.46 |
| step6+ | 520 | 0.179 | 0.204 | 0.177 | 0.181 | 0.183 | 0.265 | 0.57/0.34 | 0.57/0.33 |

## PB hit rate by first-error position (relative)

| stratum | n | VE_0 | VE_0.75 | varentropy15 | H0lim | entropy | union | VE_0 early/late miss | VE_0.75 early/late miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first_third | 1798 | 0.217 | 0.246 | 0.240 | 0.226 | 0.235 | 0.377 | 0.11/0.82 | 0.12/0.81 |
| middle_third | 1590 | 0.294 | 0.293 | 0.291 | 0.292 | 0.289 | 0.440 | 0.36/0.56 | 0.37/0.55 |
| last_third | 1054 | 0.296 | 0.309 | 0.279 | 0.285 | 0.272 | 0.420 | 0.66/0.24 | 0.67/0.21 |

## PB hit rate by first-error position (length)

| stratum | n | VE_0 | VE_0.75 | varentropy15 | H0lim | entropy | union | VE_0 early/late miss | VE_0.75 early/late miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-4 steps | 684 | 0.404 | 0.450 | 0.444 | 0.396 | 0.398 | 0.591 | 0.32/0.57 | 0.43/0.45 |
| 5-8 steps | 2270 | 0.290 | 0.290 | 0.283 | 0.292 | 0.289 | 0.444 | 0.35/0.58 | 0.34/0.58 |
| 9+ steps | 1488 | 0.159 | 0.181 | 0.163 | 0.159 | 0.161 | 0.273 | 0.29/0.64 | 0.30/0.63 |

## Where the unique hits of VE_0 and VE_0.75 fall (relative position of the first error)

| stratum | only VE_0 | only VE_0.75 | both |
|---|---:|---:|---:|
| first_third | 120 | 172 | 271 |
| middle_third | 144 | 143 | 323 |
| last_third | 78 | 92 | 234 |

## PRMBench per-answer within-AUC (6030 mixed answers)

| pair | corr of per-answer AUC | A better by >0.1 | B better by >0.1 | mean A−B |
|---|---:|---:|---:|---:|
| VE_0 vs VE_0.75 | 0.796 | 0.227 | 0.157 | +0.0211 |
| VE_0 vs varentropy15 | 0.873 | 0.184 | 0.121 | +0.0156 |
| VE_0.75 vs varentropy15 | 0.885 | 0.130 | 0.145 | -0.0055 |
| H0lim vs VE_0.75 | 0.787 | 0.209 | 0.178 | +0.0117 |
| H0lim vs entropy | 0.934 | 0.120 | 0.064 | +0.0139 |
| VE_0 vs H0lim | 0.956 | 0.079 | 0.043 | +0.0094 |

## PRMBench within-AUC by relative position of the first labelled error

| stratum | n | VE_0 | VE_0.75 | varentropy15 | H0lim | entropy |
|---|---:|---:|---:|---:|---:|---:|
| first_third | 1195 | 0.6808 | 0.6887 | 0.6913 | 0.6806 | 0.6747 |
| middle_third | 2575 | 0.7598 | 0.7387 | 0.7467 | 0.7536 | 0.7400 |
| last_third | 2260 | 0.7845 | 0.7481 | 0.7522 | 0.7664 | 0.7482 |