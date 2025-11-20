# 在线实验汇总

## INSECTS_abrupt_balanced

| model | acc_final | acc_max | kappa_final | drift_events | seen_samples | MDR | MTD | MTFA | MTR |
|-------|-----------|---------|-------------|--------------|--------------|-----|-----|------|-----|
| baseline_student | 0.2209 | 0.2337 | 0.0651 | 8 | 52847 | 0.200 | 5349.5 | 14933.3 | 3.489 |
| ts_drift_adapt | 0.2122 | 0.2182 | 0.0546 | 8 | 52847 | 0.200 | 5349.5 | 14933.3 | 3.489 |
| mean_teacher | 0.2008 | 0.2122 | 0.0410 | 8 | 52847 | 0.200 | 5349.5 | 14933.3 | 3.489 |

## sea_abrupt4

| model | acc_final | acc_max | kappa_final | drift_events | seen_samples | MDR | MTD | MTFA | MTR |
|-------|-----------|---------|-------------|--------------|--------------|-----|-----|------|-----|
| mean_teacher | 0.8710 | 0.8710 | 0.7054 | 32 | 50000 | 0.000 | 727.0 | 1792.6 | 2.466 |
| baseline_student | 0.8692 | 0.8692 | 0.7018 | 31 | 50000 | 0.000 | 663.0 | 1858.5 | 2.803 |
| ts_drift_adapt | 0.8661 | 0.8661 | 0.6922 | 30 | 50000 | 0.000 | 903.0 | 1899.5 | 2.104 |

## sine_abrupt4

| model | acc_final | acc_max | kappa_final | drift_events | seen_samples | MDR | MTD | MTFA | MTR |
|-------|-----------|---------|-------------|--------------|--------------|-----|-----|------|-----|
| ts_drift_adapt | 0.7578 | 0.9251 | 0.5155 | 0 | 50000 | 1.000 | nan | nan | nan |
| baseline_student | 0.6833 | 0.8981 | 0.3650 | 0 | 50000 | 1.000 | nan | nan | nan |
| mean_teacher | 0.6755 | 0.9255 | 0.3500 | 0 | 50000 | 1.000 | nan | nan | nan |

## stagger_abrupt3

| model | acc_final | acc_max | kappa_final | drift_events | seen_samples | MDR | MTD | MTFA | MTR |
|-------|-----------|---------|-------------|--------------|--------------|-----|-----|------|-----|
| baseline_student | 0.8810 | 0.9948 | 0.7534 | 0 | 60000 | 1.000 | nan | nan | nan |
| ts_drift_adapt | 0.8727 | 0.9914 | 0.7358 | 0 | 60000 | 1.000 | nan | nan | nan |
| mean_teacher | 0.8705 | 0.9915 | 0.7306 | 0 | 60000 | 1.000 | nan | nan | nan |
