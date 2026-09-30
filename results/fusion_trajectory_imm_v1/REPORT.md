# Full-trajectory fusion and IMM - Step318

Review PASS. No consistent two-task winner. Same110 development answers, corrected v3 labels/v2 groups.

| Method | PRMB AUC | Within-answer | PB native | Fixed-IU gate | Clean /33 | Exact /53 | Raw peak /53 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dual__iu | 0.68131 | 0.76881 | 30.16% | 30.37% | 18 | 12 | 20 |
| dual__cond100_graph010 | 0.65545 | 0.75348 | 30.22% | 29.53% | 17 | 13 | 18 |
| sample_risk_top__equal_graph_perm | 0.76842 | 0.78661 | 33.92% | 29.35% | 18 | 13 | 20 |
| traj_iu_joint_graph__mean | 0.66944 | 0.75609 | 31.98% | 30.37% | 15 | 13 | 19 |
| traj_iu_joint_graph__gls | 0.68338 | 0.79012 | 30.35% | 28.73% | 15 | 12 | 17 |
| traj_iu_joint_graph__hold | 0.68043 | 0.78387 | 30.35% | 28.73% | 15 | 12 | 18 |
| traj_iu_joint_graph__imm | 0.69325 | 0.75120 | 16.69% | 26.28% | 11 | 7 | 17 |
| traj_iu_joint0__mean | 0.66888 | 0.76859 | 30.26% | 30.37% | 13 | 13 | 19 |
| traj_iu_joint0__gls | 0.68342 | 0.80759 | 31.09% | 27.72% | 16 | 12 | 16 |
| traj_iu_joint0__hold | 0.67911 | 0.80134 | 31.09% | 27.72% | 16 | 12 | 17 |
| traj_iu_joint0__imm | 0.69301 | 0.75120 | 18.16% | 26.28% | 14 | 6 | 17 |
| traj_iu_joint_perm__mean | 0.67191 | 0.76859 | 30.26% | 30.37% | 13 | 13 | 19 |
| traj_iu_joint_perm__gls | 0.68654 | 0.80475 | 31.47% | 28.73% | 15 | 13 | 17 |
| traj_iu_joint_perm__hold | 0.68111 | 0.79225 | 33.10% | 28.73% | 15 | 14 | 18 |
| traj_iu_joint_perm__imm | 0.69190 | 0.75120 | 23.79% | 27.61% | 14 | 9 | 18 |
| traj_equal_graph__mean | 0.67811 | 0.77549 | 25.83% | 29.35% | 16 | 10 | 18 |
| traj_equal_graph__gls | 0.68474 | 0.76924 | 25.91% | 29.35% | 16 | 10 | 17 |
| traj_equal_graph__hold | 0.68107 | 0.76299 | 25.91% | 29.35% | 16 | 10 | 18 |
| traj_equal_graph__imm | 0.70776 | 0.78105 | 20.87% | 26.91% | 14 | 8 | 19 |
| traj_equal_perm__mean | 0.68518 | 0.77838 | 32.05% | 29.35% | 17 | 12 | 19 |
| traj_equal_perm__gls | 0.68554 | 0.76422 | 28.30% | 30.31% | 14 | 12 | 18 |
| traj_equal_perm__hold | 0.68570 | 0.75797 | 28.30% | 30.31% | 14 | 12 | 19 |
| traj_equal_perm__imm | 0.71152 | 0.79471 | 29.75% | 28.23% | 14 | 12 | 21 |
| traj_iu__hold | 0.67995 | 0.76881 | 30.16% | 30.37% | 18 | 12 | 21 |
| traj_iu__imm | 0.68970 | 0.78458 | 25.94% | 27.92% | 14 | 10 | 20 |
| traj_joint_graph__hold | 0.65817 | 0.75695 | 30.22% | 29.53% | 17 | 13 | 19 |
| traj_joint_graph__imm | 0.65721 | 0.69399 | 14.64% | 26.60% | 16 | 6 | 18 |
| traj_equal__hold | 0.68147 | 0.77782 | 25.55% | 29.35% | 15 | 10 | 19 |
| traj_equal__imm | 0.69381 | 0.77932 | 24.52% | 27.92% | 11 | 11 | 20 |
| traj_iu_joint_graph__imm_permuted | 0.65225 | 0.71873 | 18.70% | 16.83% | 11 | 8 | 14 |

27 additions +149 unchanged anchors =176 total;50 registered contrasts. All final outputs valid110;3 original Joint fallbacks inherited.
GLS vsIU intervals include0 on both primary endpoints. Primary IMM loses13 correct PB decisions/gains4 versus hold. Its two modes are dynamical, not semantic labels.
Post-evaluation fixed peak/gate exchanges and serial-dependence summaries are diagnostics, not new candidates. See GATE_AUDIT.json and REPORT.html.
Scoring52.80s; contrasts25.64s; review113.13s. Seven tests, vector IMM/math/raw-label/metric and bootstrap review PASS.
Next audit the no-error decision under serially dependent fused trajectories before a bounded gate-only change. Keep all wider research obligations active.
