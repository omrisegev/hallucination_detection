The first independent calibration replay differed by3.93e-10 in one null maximum.
Its covariance reduction used a different memory order before the iterative
loading initializer. The audit now preserves the original contiguous covariance
reduction for that initializer while independently constructing the null from
signed observations (not source Gram sums). Original tight tolerance passed.
No fitting code, penalty, support, score or scientific parameter was changed.
The experiment was not restarted; only the independent audit was rerun.
