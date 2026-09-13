# Whole-answer position fusion (RBM12 bank, Top10 fixed)

Full development benchmark; other-answer unlabelled fitting. Rank means loading-map rank, not latent classes.

| Method | PB % | PRMB within AUC | PRMScore | Coverage |
|---|---:|---:|---:|---:|
| Original answer-local RBM12 | 36.271 | 0.74520 | 0.62222 | 13769/13769 |
| Gaussian factor: fixed weights | 36.372 | 0.74611 | 0.60734 | 13769/13769 |
| Gaussian factor: fixed weights, position mean | 28.630 | 0.74322 | 0.60385 | 13769/13769 |
| Gaussian factor: answer position, rank 1 | 21.160 | 0.74136 | 0.60124 | 13769/13769 |
| Gaussian factor: answer position, rank 2 | 33.543 | 0.75591 | 0.60514 | 13769/13769 |
| Gaussian factor: shuffled positions, rank 2 | 30.493 | 0.74619 | 0.60722 | 13769/13769 |
| IU-PCR: fixed weights | 20.071 | 0.69118 | 0.56890 | 13769/13769 |
| IU-PCR: fixed weights, position mean | 22.893 | 0.72424 | 0.58698 | 13769/13769 |
| IU-PCR: answer position | 35.516 | 0.76573 | 0.61442 | 13769/13769 |
| IU-PCR: shuffled positions | 20.653 | 0.69106 | 0.56907 | 13769/13769 |
| Gaussian RBM: fixed weights | 36.240 | 0.74448 | 0.59718 | 13769/13769 |
| Gaussian RBM: answer position, rank 1 | 35.717 | 0.74723 | 0.60018 | 13769/13769 |
| Gaussian RBM: answer position, rank 2 | 22.187 | 0.68793 | 0.56679 | 13769/13769 |
| Gaussian RBM: shuffled positions, rank 2 | 36.029 | 0.74805 | 0.60019 | 13769/13769 |
| Equal feature weights | 20.181 | 0.68910 | 0.56780 | 13769/13769 |
| Position only: early | 18.756 | 0.33827 | 0.41457 | 13769/13769 |
| Position only: late | 6.704 | 0.66173 | 0.54332 | 13769/13769 |
