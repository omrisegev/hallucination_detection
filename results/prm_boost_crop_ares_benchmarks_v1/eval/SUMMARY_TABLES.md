# CROP / ARES benchmark collection: label evaluation (development data)

Population: answers scored by every cell of the benchmark (common set): crv_arith 1500, crv_gsm8k 1319, deltabench 447, hard2verify 177, math_shepherd 9999, prm800k 8000, processbench 3400, socratic 2995.

## A. Within-answer AUC, PRM + its own backbone telemetry (prm mode)

|                                    |   PRM |   TELE_N16 (backbone) |   CHAIN_N16 |   HALF_N16 |   position only |
|:-----------------------------------|------:|----------------------:|------------:|-----------:|----------------:|
| ('crv_arith', 'qwen_prm')          | 0.902 |                 0.74  |       0.768 |      0.867 |           0.957 |
| ('crv_arith', 'qwen_prm_72b')      | 0.961 |                 0.744 |       0.773 |      0.925 |           0.957 |
| ('crv_arith', 'pure_prm')          | 0.87  |                 0.277 |       0.394 |      0.761 |           0.957 |
| ('crv_arith', 'rlhflow_prm')       | 0.686 |                 0.345 |       0.448 |      0.602 |           0.957 |
| ('crv_arith', 'math_shepherd')     | 0.948 |                 0.219 |       0.936 |      0.865 |           0.957 |
| ('crv_gsm8k', 'qwen_prm')          | 0.743 |                 0.621 |       0.646 |      0.707 |           0.769 |
| ('crv_gsm8k', 'qwen_prm_72b')      | 0.76  |                 0.629 |       0.665 |      0.736 |           0.769 |
| ('crv_gsm8k', 'pure_prm')          | 0.804 |                 0.531 |       0.572 |      0.754 |           0.769 |
| ('crv_gsm8k', 'rlhflow_prm')       | 0.648 |                 0.526 |       0.607 |      0.644 |           0.769 |
| ('crv_gsm8k', 'math_shepherd')     | 0.749 |                 0.521 |       0.507 |      0.728 |           0.769 |
| ('math_shepherd', 'qwen_prm')      | 0.445 |                 0.426 |       0.435 |      0.448 |           0.355 |
| ('math_shepherd', 'qwen_prm_72b')  | 0.446 |                 0.437 |       0.431 |      0.444 |           0.355 |
| ('math_shepherd', 'pure_prm')      | 0.446 |                 0.455 |       0.45  |      0.444 |           0.355 |
| ('math_shepherd', 'rlhflow_prm')   | 0.405 |                 0.455 |       0.422 |      0.414 |           0.355 |
| ('math_shepherd', 'math_shepherd') | 0.361 |                 0.413 |       0.386 |      0.375 |           0.355 |
| ('prm800k', 'qwen_prm')            | 0.929 |                 0.719 |       0.835 |      0.913 |           0.995 |
| ('prm800k', 'qwen_prm_72b')        | 0.926 |                 0.838 |       0.885 |      0.92  |           0.995 |
| ('prm800k', 'pure_prm')            | 0.942 |                 0.623 |       0.76  |      0.923 |           0.995 |
| ('prm800k', 'rlhflow_prm')         | 0.382 |                 0.644 |       0.49  |      0.425 |           0.995 |
| ('prm800k', 'math_shepherd')       | 0.614 |                 0.402 |       0.477 |      0.625 |           0.995 |
| ('deltabench', 'qwen_prm')         | 0.656 |                 0.655 |       0.671 |      0.683 |           0.555 |
| ('deltabench', 'qwen_prm_72b')     | 0.63  |                 0.65  |       0.665 |      0.654 |           0.555 |
| ('deltabench', 'pure_prm')         | 0.648 |                 0.591 |       0.395 |      0.679 |           0.555 |
| ('deltabench', 'rlhflow_prm')      | 0.536 |                 0.584 |       0.582 |      0.562 |           0.555 |
| ('deltabench', 'math_shepherd')    | 0.476 |                 0.605 |       0.617 |      0.519 |           0.555 |

## B. Macro-F1 (PRMScore / ARES form, x100; positive = valid step). Native = PRM reward <= 0.5; GAP = label-free count; ORACLE = true count (reference)

|                                    |   PRM native |   PRM GAP |   CHAIN_N16 GAP |   HALF_N16 GAP |   PRM ORACLE |   CHAIN_N16 ORACLE |
|:-----------------------------------|-------------:|----------:|----------------:|---------------:|-------------:|-------------------:|
| ('crv_arith', 'qwen_prm')          |         77.3 |      44   |            50.7 |           46.4 |         92.1 |               82.1 |
| ('crv_arith', 'qwen_prm_72b')      |         83.8 |      45.1 |            49.5 |           45.4 |         96.7 |               83.7 |
| ('crv_arith', 'pure_prm')          |         68   |      45.3 |            43   |           45.1 |         90.8 |               69.5 |
| ('crv_arith', 'rlhflow_prm')       |         67.9 |      45.1 |            40.9 |           41.2 |         81.6 |               70.3 |
| ('crv_arith', 'math_shepherd')     |         71.9 |      47.4 |            47   |           46.3 |         97.3 |               95.7 |
| ('crv_gsm8k', 'qwen_prm')          |         72.5 |      51.2 |            49.9 |           50.7 |         80.9 |               75   |
| ('crv_gsm8k', 'qwen_prm_72b')      |         77.4 |      53.6 |            49.5 |           53.3 |         80.6 |               75.3 |
| ('crv_gsm8k', 'pure_prm')          |         70.9 |      52   |            48.3 |           49.8 |         83.5 |               70.8 |
| ('crv_gsm8k', 'rlhflow_prm')       |         62.7 |      49.9 |            47.6 |           48.9 |         75.8 |               72.7 |
| ('crv_gsm8k', 'math_shepherd')     |         67.5 |      50.7 |            46.5 |           49.9 |         81.1 |               70.1 |
| ('math_shepherd', 'qwen_prm')      |         44.1 |      47.1 |            46.2 |           47.2 |         63.6 |               63.3 |
| ('math_shepherd', 'qwen_prm_72b')  |         41.7 |      47   |            46.5 |           47   |         63.5 |               63.1 |
| ('math_shepherd', 'pure_prm')      |         44.8 |      47.1 |            47.2 |           47.2 |         63.2 |               63.8 |
| ('math_shepherd', 'rlhflow_prm')   |         45.2 |      44.6 |            46   |           45.1 |         62.4 |               62.5 |
| ('math_shepherd', 'math_shepherd') |         42   |      42.1 |            43.4 |           42   |         60   |               61.2 |
| ('prm800k', 'qwen_prm')            |         75.2 |      56.7 |            55.9 |           56.4 |         91.1 |               81.9 |
| ('prm800k', 'qwen_prm_72b')        |         80.4 |      55.8 |            56.5 |           56.4 |         91.4 |               85.9 |
| ('prm800k', 'pure_prm')            |         62   |      61   |            60.8 |           61.7 |         92.9 |               76   |
| ('prm800k', 'rlhflow_prm')         |         54.1 |      48.5 |            49.7 |           47.5 |         61.6 |               61.8 |
| ('prm800k', 'math_shepherd')       |         41.7 |      49.1 |            47.4 |           50   |         72.6 |               62.8 |
| ('deltabench', 'qwen_prm')         |         56.3 |      58   |            58.2 |           59.3 |         62   |               63.6 |
| ('deltabench', 'qwen_prm_72b')     |         57   |      57.8 |            58.9 |           59.1 |         60.5 |               61.9 |
| ('deltabench', 'pure_prm')         |         53.9 |      56.6 |            48.7 |           59.6 |         61.9 |               51   |
| ('deltabench', 'rlhflow_prm')      |         49.9 |      50.9 |            53.8 |           52.5 |         52.3 |               58.2 |
| ('deltabench', 'math_shepherd')    |         44.9 |      48.3 |            54.4 |           51   |         53.3 |               58.9 |

## C. Standalone base-model telemetry only (L-SML TELE, plain MEAN)

|                                    |   TELE_N16 |   MEAN_N16 |   TELE_N13 |   pooled AUROC TELE_N16 |   macro-F1 TELE_N16 GAP |
|:-----------------------------------|-----------:|-----------:|-----------:|------------------------:|------------------------:|
| ('crv_arith', 'qwen_prm')          |      0.683 |      0.583 |      0.612 |                   0.609 |                  47.635 |
| ('crv_arith', 'qwen_prm_72b')      |      0.538 |      0.517 |      0.465 |                   0.531 |                  46.417 |
| ('crv_arith', 'pure_prm')          |      0.289 |      0.316 |      0.282 |                   0.407 |                  42.271 |
| ('crv_arith', 'rlhflow_prm')       |      0.479 |      0.452 |      0.505 |                   0.48  |                  46.203 |
| ('crv_arith', 'math_shepherd')     |      0.223 |      0.247 |      0.157 |                   0.356 |                  42.188 |
| ('crv_gsm8k', 'qwen_prm')          |      0.578 |      0.581 |      0.55  |                   0.557 |                  48.411 |
| ('crv_gsm8k', 'qwen_prm_72b')      |      0.555 |      0.547 |      0.542 |                   0.536 |                  47.301 |
| ('crv_gsm8k', 'pure_prm')          |      0.515 |      0.518 |      0.515 |                   0.508 |                  47.455 |
| ('crv_gsm8k', 'rlhflow_prm')       |      0.55  |      0.549 |      0.545 |                   0.519 |                  47.741 |
| ('crv_gsm8k', 'math_shepherd')     |      0.478 |      0.498 |      0.483 |                   0.489 |                  45.678 |
| ('math_shepherd', 'qwen_prm')      |      0.439 |      0.469 |      0.416 |                   0.45  |                  46.458 |
| ('math_shepherd', 'qwen_prm_72b')  |      0.444 |      0.471 |      0.426 |                   0.46  |                  46.813 |
| ('math_shepherd', 'pure_prm')      |      0.487 |      0.468 |      0.454 |                   0.487 |                  49.725 |
| ('math_shepherd', 'rlhflow_prm')   |      0.407 |      0.449 |      0.391 |                   0.43  |                  44.882 |
| ('math_shepherd', 'math_shepherd') |      0.559 |      0.516 |      0.535 |                   0.53  |                  53.391 |
| ('prm800k', 'qwen_prm')            |      0.461 |      0.679 |      0.519 |                   0.495 |                  50.366 |
| ('prm800k', 'qwen_prm_72b')        |      0.433 |      0.649 |      0.502 |                   0.492 |                  50.985 |
| ('prm800k', 'pure_prm')            |      0.446 |      0.625 |      0.453 |                   0.519 |                  51.468 |
| ('prm800k', 'rlhflow_prm')         |      0.503 |      0.684 |      0.509 |                   0.549 |                  53.377 |
| ('prm800k', 'math_shepherd')       |      0.316 |      0.591 |      0.368 |                   0.386 |                  49.114 |
| ('deltabench', 'qwen_prm')         |      0.666 |      0.665 |      0.646 |                   0.662 |                  57.913 |
| ('deltabench', 'qwen_prm_72b')     |      0.619 |      0.62  |      0.617 |                   0.623 |                  55.514 |
| ('deltabench', 'pure_prm')         |      0.59  |      0.598 |      0.579 |                   0.595 |                  54.127 |
| ('deltabench', 'rlhflow_prm')      |      0.601 |      0.609 |      0.584 |                   0.602 |                  54.067 |
| ('deltabench', 'math_shepherd')    |      0.589 |      0.601 |      0.587 |                   0.602 |                  55.165 |
| ('processbench', 'qwen_prm')       |      0.773 |      0.771 |      0.684 |                   0.733 |                 nan     |
| ('processbench', 'qwen_prm_72b')   |      0.746 |      0.754 |      0.666 |                   0.739 |                 nan     |
| ('processbench', 'pure_prm')       |      0.681 |      0.704 |      0.612 |                   0.713 |                 nan     |
| ('processbench', 'rlhflow_prm')    |      0.698 |      0.706 |      0.636 |                   0.715 |                 nan     |
| ('processbench', 'math_shepherd')  |      0.6   |      0.615 |      0.531 |                   0.665 |                 nan     |
| ('socratic', 'qwen_prm')           |      0.696 |      0.681 |      0.706 |                   0.679 |                  63.289 |
| ('socratic', 'qwen_prm_72b')       |      0.695 |      0.68  |      0.684 |                   0.678 |                  65.306 |
| ('socratic', 'pure_prm')           |      0.663 |      0.658 |      0.65  |                   0.643 |                  63.805 |
| ('socratic', 'rlhflow_prm')        |      0.662 |      0.653 |      0.663 |                   0.642 |                  62.452 |
| ('socratic', 'math_shepherd')      |      0.617 |      0.613 |      0.594 |                   0.597 |                  60.736 |
| ('hard2verify', 'qwen_prm')        |      0.54  |      0.533 |      0.515 |                   0.543 |                  47.225 |
| ('hard2verify', 'qwen_prm_72b')    |      0.56  |      0.552 |      0.551 |                   0.548 |                  43.475 |
| ('hard2verify', 'pure_prm')        |      0.489 |      0.478 |      0.463 |                   0.499 |                  39.327 |
| ('hard2verify', 'rlhflow_prm')     |      0.528 |      0.53  |      0.524 |                   0.524 |                  40.241 |
| ('hard2verify', 'math_shepherd')   |      0.512 |      0.516 |      0.51  |                   0.52  |                  40.027 |

## D. CROP Table 6 protocol (10 splits, alpha = 0.05; CROP's calibration labels for every arm): prefix kept %

|                                    |   prm:PRMPROB@fixed101 |   prm:PRMPROB@q201 |   prm:CHAIN_N16 |   prm:HALF_N16 |   prm:TELE_N16 |   sa:TELE_N16 |   sa:MEAN_N16 |
|:-----------------------------------|-----------------------:|-------------------:|----------------:|---------------:|---------------:|--------------:|--------------:|
| ('crv_arith', 'math_shepherd')     |                   93.8 |               93.6 |            63.8 |           57.8 |           38.3 |          49.7 |          51.3 |
| ('crv_arith', 'pure_prm')          |                   90.4 |               90.3 |            42.6 |           57.1 |           43.6 |          49.9 |          50.4 |
| ('crv_arith', 'qwen_prm')          |                   97.4 |               96.9 |            24   |           39.1 |           22.3 |          43.4 |          44.8 |
| ('crv_arith', 'qwen_prm_72b')      |                   96.7 |               97   |            35.6 |           64.9 |           32.7 |          36.4 |          41.7 |
| ('crv_arith', 'rlhflow_prm')       |                   95.6 |               95.1 |            31.2 |           34.9 |           52.5 |          52.5 |          52.2 |
| ('crv_gsm8k', 'math_shepherd')     |                   60.3 |               61.4 |            16.4 |           29.6 |           34.7 |          31   |          29.9 |
| ('crv_gsm8k', 'pure_prm')          |                   62.5 |               63   |            33.3 |           45.5 |           30.5 |          32.2 |          32.4 |
| ('crv_gsm8k', 'qwen_prm')          |                   79.8 |               80.9 |            32.5 |           40.1 |           27.8 |          27.6 |          28.8 |
| ('crv_gsm8k', 'qwen_prm_72b')      |                   80.8 |               81.4 |            30.9 |           42.9 |           28.2 |          33.2 |          31.8 |
| ('crv_gsm8k', 'rlhflow_prm')       |                   59.8 |               62.3 |            25.4 |           27.2 |           33.2 |          33.3 |          32.1 |
| ('math_shepherd', 'math_shepherd') |                    0   |                2.4 |             2.5 |            2.7 |            3.2 |           6.8 |           5.2 |
| ('math_shepherd', 'pure_prm')      |                    3.2 |                3.4 |             3.2 |            2.9 |            3.5 |           4.2 |           4.1 |
| ('math_shepherd', 'qwen_prm')      |                    1   |                2.4 |             2.3 |            2.6 |            2.3 |           3.7 |           4.4 |
| ('math_shepherd', 'qwen_prm_72b')  |                    2.4 |                3.2 |             3.1 |            3   |            4   |           4.1 |           4.1 |
| ('math_shepherd', 'rlhflow_prm')   |                    0   |                2.7 |             3   |            2.5 |            3.4 |           2.9 |           4   |
| ('prm800k', 'math_shepherd')       |                   36.3 |               36.5 |            12.3 |           29.4 |            9.8 |          16.8 |          17.1 |
| ('prm800k', 'pure_prm')            |                   61.6 |               61.8 |            30.8 |           58.1 |           20.2 |          17.3 |          17.1 |
| ('prm800k', 'qwen_prm')            |                   68.3 |               69.1 |            35   |           47.8 |           29.1 |          14.7 |          20.1 |
| ('prm800k', 'qwen_prm_72b')        |                   75.8 |               75.4 |            39.3 |           45.6 |           34.5 |          17.9 |          19.9 |
| ('prm800k', 'rlhflow_prm')         |                   36.9 |               37.3 |             7.8 |           10.3 |           13.5 |          22.8 |          24.2 |

## E. CROP Table 6 protocol: pooled step AUROC (test splits)

|                                    |   prm:PRMPROB@q201 |   prm:CHAIN_N16 |   prm:HALF_N16 |   prm:TELE_N16 |   sa:TELE_N16 |   sa:MEAN_N16 |
|:-----------------------------------|-------------------:|----------------:|---------------:|---------------:|--------------:|--------------:|
| ('crv_arith', 'math_shepherd')     |              0.942 |           0.765 |          0.707 |          0.35  |         0.359 |         0.374 |
| ('crv_arith', 'pure_prm')          |              0.877 |           0.461 |          0.627 |          0.392 |         0.41  |         0.413 |
| ('crv_arith', 'qwen_prm')          |              0.943 |           0.69  |          0.684 |          0.68  |         0.614 |         0.548 |
| ('crv_arith', 'qwen_prm_72b')      |              0.978 |           0.691 |          0.719 |          0.679 |         0.53  |         0.503 |
| ('crv_arith', 'rlhflow_prm')       |              0.86  |           0.461 |          0.535 |          0.414 |         0.485 |         0.472 |
| ('crv_gsm8k', 'math_shepherd')     |              0.843 |           0.516 |          0.672 |          0.509 |         0.489 |         0.496 |
| ('crv_gsm8k', 'pure_prm')          |              0.847 |           0.542 |          0.668 |          0.515 |         0.505 |         0.507 |
| ('crv_gsm8k', 'qwen_prm')          |              0.849 |           0.6   |          0.652 |          0.582 |         0.554 |         0.554 |
| ('crv_gsm8k', 'qwen_prm_72b')      |              0.879 |           0.615 |          0.673 |          0.593 |         0.534 |         0.533 |
| ('crv_gsm8k', 'rlhflow_prm')       |              0.797 |           0.555 |          0.572 |          0.517 |         0.52  |         0.521 |
| ('math_shepherd', 'math_shepherd') |              0.428 |           0.415 |          0.409 |          0.435 |         0.53  |         0.505 |
| ('math_shepherd', 'pure_prm')      |              0.456 |           0.46  |          0.455 |          0.468 |         0.487 |         0.474 |
| ('math_shepherd', 'qwen_prm')      |              0.455 |           0.44  |          0.456 |          0.435 |         0.448 |         0.478 |
| ('math_shepherd', 'qwen_prm_72b')  |              0.451 |           0.449 |          0.456 |          0.453 |         0.459 |         0.478 |
| ('math_shepherd', 'rlhflow_prm')   |              0.44  |           0.438 |          0.444 |          0.463 |         0.427 |         0.465 |
| ('prm800k', 'math_shepherd')       |              0.654 |           0.419 |          0.611 |          0.368 |         0.393 |         0.583 |
| ('prm800k', 'pure_prm')            |              0.878 |           0.738 |          0.848 |          0.64  |         0.525 |         0.614 |
| ('prm800k', 'qwen_prm')            |              0.93  |           0.783 |          0.852 |          0.68  |         0.5   |         0.654 |
| ('prm800k', 'qwen_prm_72b')        |              0.95  |           0.835 |          0.86  |          0.79  |         0.497 |         0.634 |
| ('prm800k', 'rlhflow_prm')         |              0.748 |           0.481 |          0.474 |          0.558 |         0.553 |         0.661 |
