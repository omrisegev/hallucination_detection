| Suite | Contrast | PB delta pp | PB CI | within delta | within CI | gained / lost |
|---|---|---:|---|---:|---|---:|
| variance | Gaussian fusion, separate state variances, 12 features, logit − Gaussian fusion, shared state variance, 12 features, logit | -15.0897 | [-17.4746, -12.7660] | -0.048659 | [-0.053922, -0.043877] | 283 / 934 |
| variance | Gaussian fusion, separate state variances, 6 features, posterior − Gaussian fusion, shared state variance, 6 features, posterior | +0.2753 | [-0.6084, +1.1796] | +0.005166 | [+0.002576, +0.007737] | 191 / 175 |
| capacity | RBM, 4 hidden units, exact training, 12 features, logit − RBM, 1 hidden unit, exact training, 12 features, logit | -7.6159 | [-9.4943, -5.7564] | -0.023511 | [-0.027552, -0.019642] | 299 / 701 |
| capacity | RBM, 4 hidden units, exact training, 6 features, posterior − RBM, 1 hidden unit, exact training, 6 features, posterior | -10.8812 | [-13.0988, -8.7371] | -0.025882 | [-0.029364, -0.022462] | 285 / 802 |
| temporal | RBM with token sequence fusion across steps, 12 features, logit − RBM with shuffled token order, control, 12 features, logit | -0.3303 | [-0.9615, +0.3041] | -0.002814 | [-0.003966, -0.001703] | 75 / 108 |
| temporal | RBM with token sequence fusion across steps, 6 features, posterior − RBM with shuffled token order, control, 6 features, posterior | -0.3342 | [-0.9826, +0.3015] | -0.003561 | [-0.005372, -0.001824] | 85 / 119 |
| stability | RBM, 4 hidden units, best density fit of 3 starts, 12 features, logit − RBM, 4 hidden units, exact training, 12 features, logit | -1.3291 | [-2.1655, -0.5145] | +0.000124 | [-0.001624, +0.001934] | 79 / 133 |
| stability | RBM, 4 hidden units, best density fit of 3 starts, 6 features, posterior − RBM, 4 hidden units, exact training, 6 features, posterior | -0.6144 | [-1.3230, +0.0784] | -0.001438 | [-0.002739, -0.000160] | 44 / 71 |
| depth_amended | not available | | | | | |
