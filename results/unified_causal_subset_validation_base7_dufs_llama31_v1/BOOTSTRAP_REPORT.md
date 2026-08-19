# Frozen scorer-model validation bootstrap

Control: `base7_full28`. Paired source-question bootstrap, 2000 replicates.

| candidate | ΔGlobal [95% CI] | ΔLocalization [95% CI] | ΔEarly [95% CI] | gate |
|---|---:|---:|---:|:---:|
| base7_full28__dufs_l0p1 | -0.0016 [-0.0024, -0.0008] | +0.0015 [-0.0077, +0.0113] | -0.0014 [-0.0019, -0.0010] | fail |
| base7_full28__dufs_l0p3 | -0.0039 [-0.0055, -0.0022] | -0.0056 [-0.0180, +0.0075] | -0.0032 [-0.0041, -0.0022] | fail |
| base7_full28__dufs_l1 | -0.0079 [-0.0110, -0.0048] | -0.0301 [-0.0481, -0.0127] | -0.0062 [-0.0079, -0.0043] | fail |
| base7_full28__dufs_l3 | -0.0117 [-0.0158, -0.0073] | -0.0528 [-0.0726, -0.0328] | -0.0084 [-0.0108, -0.0060] | fail |
| base7_full28__rw_a0p5 | -0.0126 [-0.0169, -0.0084] | -0.0319 [-0.0498, -0.0145] | -0.0090 [-0.0116, -0.0065] | fail |
| raw9_full36 | +0.0016 [-0.0009, +0.0041] | -0.0175 [-0.0273, -0.0077] | +0.0029 [+0.0012, +0.0045] | fail |

The gate requires a positive lower CI on at least one task and lower CIs above the frozen noninferiority margins (-0.010 Global, -0.010 Localization, -0.015 Early) on all remaining tasks.
