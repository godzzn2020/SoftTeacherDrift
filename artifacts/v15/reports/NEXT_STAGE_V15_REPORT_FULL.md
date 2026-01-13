# NEXT_STAGE V15.1 Report FULL（Permutation-test Confirm + vote_score）

- 生成时间：2026-01-13 00:00:20
- 环境确认：
  - `which python` -> `/home/ylh/anaconda3/envs/ZZNSTD/bin/python`
  - `python -V` -> `Python 3.10.19`

## 关键口径与硬约束
- drift 约束：`sea_abrupt4` + `sine_abrupt4` 上 `miss_tol500_mean==0` 且 `conf_P90_mean<500`
- 目标：在满足约束下最小化 no-drift `confirm_rate_per_10k`（`sea_nodrift` + `sine_nodrift` 平均；次选最大化 no-drift `MTFA_win`）

## Track AL：Perm-confirm sweep
- 产物：`artifacts/v15/tables/TRACKAL_PERM_CONFIRM_SWEEP_V15.csv`

| group | phase | confirm_rule | perm_stat | perm_alpha | perm_pre_n | perm_post_n | delta_k | sea_miss | sea_confP90 | sine_miss | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_weighted_n5 | quick | weighted | N/A | N/A | N/A | N/A | N/A | 0.000 | 229.4 | 0.000 | 286.7 | 26.540 | 368.7 | 0.7577 |
| P_perm_delta_fused_score_a0.005_pre200_post10_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 10 | 25 | 0.150 | 479.0 | 0.150 | 419.5 | 17.880 | 530.0 | 0.7594 |
| P_perm_delta_fused_score_a0.005_pre200_post10_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 10 | 50 | 0.200 | 602.8 | 0.150 | 522.2 | 17.420 | 563.7 | 0.7619 |
| P_perm_delta_fused_score_a0.005_pre200_post20_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 20 | 25 | 0.200 | 495.6 | 0.150 | 482.5 | 18.640 | 524.6 | 0.7587 |
| P_perm_delta_fused_score_a0.005_pre200_post20_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 20 | 50 | 0.350 | 576.0 | 0.200 | 473.2 | 17.780 | 540.7 | 0.7577 |
| P_perm_delta_fused_score_a0.005_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 30 | 25 | 0.050 | 337.6 | 0.300 | 664.3 | 19.640 | 497.4 | 0.7609 |
| P_perm_delta_fused_score_a0.005_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 30 | 50 | 0.150 | 479.3 | 0.400 | 608.0 | 18.800 | 529.8 | 0.7596 |
| P_perm_delta_fused_score_a0.005_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 50 | 25 | 0.100 | 419.2 | 0.150 | 385.9 | 20.500 | 478.2 | 0.7623 |
| P_perm_delta_fused_score_a0.005_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 50 | 50 | 0.150 | 448.9 | 0.100 | 469.4 | 19.860 | 490.2 | 0.7541 |
| P_perm_delta_fused_score_a0.005_pre500_post10_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 10 | 25 | 0.200 | 550.4 | 0.350 | 851.2 | 16.360 | 585.2 | 0.7590 |
| P_perm_delta_fused_score_a0.005_pre500_post10_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 10 | 50 | 0.100 | 454.0 | 0.400 | 832.9 | 15.560 | 611.2 | 0.7550 |
| P_perm_delta_fused_score_a0.005_pre500_post20_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 20 | 25 | 0.050 | 380.8 | 0.050 | 300.1 | 19.120 | 508.5 | 0.7546 |
| P_perm_delta_fused_score_a0.005_pre500_post20_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 20 | 50 | 0.150 | 520.3 | 0.350 | 673.6 | 18.660 | 515.4 | 0.7593 |
| P_perm_delta_fused_score_a0.005_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 30 | 25 | 0.100 | 435.2 | 0.250 | 613.7 | 19.220 | 514.8 | 0.7567 |
| P_perm_delta_fused_score_a0.005_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 30 | 50 | 0.100 | 461.7 | 0.250 | 560.6 | 18.720 | 526.4 | 0.7624 |
| P_perm_delta_fused_score_a0.005_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 50 | 25 | 0.050 | 359.0 | 0.100 | 290.2 | 19.640 | 503.2 | 0.7542 |
| P_perm_delta_fused_score_a0.005_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 50 | 50 | 0.050 | 334.7 | 0.200 | 442.2 | 19.820 | 480.7 | 0.7601 |
| P_perm_delta_fused_score_a0.01_pre200_post10_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 10 | 25 | 0.250 | 518.7 | 0.300 | 587.5 | 17.480 | 549.1 | 0.7599 |
| P_perm_delta_fused_score_a0.01_pre200_post10_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 10 | 50 | 0.300 | 563.2 | 0.200 | 381.4 | 17.440 | 557.0 | 0.7671 |
| P_perm_delta_fused_score_a0.01_pre200_post20_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 20 | 25 | 0.000 | 335.3 | 0.150 | 383.0 | 19.080 | 517.8 | 0.7532 |
| P_perm_delta_fused_score_a0.01_pre200_post20_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 20 | 50 | 0.100 | 359.6 | 0.150 | 489.2 | 18.500 | 514.7 | 0.7575 |
| P_perm_delta_fused_score_a0.01_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 30 | 25 | 0.100 | 369.9 | 0.250 | 496.6 | 19.400 | 509.7 | 0.7551 |
| P_perm_delta_fused_score_a0.01_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 30 | 50 | 0.100 | 361.2 | 0.150 | 340.1 | 19.780 | 487.3 | 0.7627 |
| P_perm_delta_fused_score_a0.01_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 50 | 25 | 0.150 | 379.8 | 0.100 | 445.4 | 20.500 | 475.2 | 0.7485 |
| P_perm_delta_fused_score_a0.01_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 50 | 50 | 0.100 | 341.7 | 0.100 | 385.6 | 19.640 | 493.7 | 0.7612 |
| P_perm_delta_fused_score_a0.01_pre500_post10_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 10 | 25 | 0.200 | 667.2 | 0.150 | 491.5 | 16.660 | 574.9 | 0.7525 |
| P_perm_delta_fused_score_a0.01_pre500_post10_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 10 | 50 | 0.300 | 615.0 | 0.250 | 747.8 | 16.800 | 570.3 | 0.7562 |
| P_perm_delta_fused_score_a0.01_pre500_post20_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 20 | 25 | 0.100 | 424.0 | 0.300 | 572.1 | 19.240 | 510.9 | 0.7591 |
| P_perm_delta_fused_score_a0.01_pre500_post20_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 20 | 50 | 0.150 | 589.7 | 0.250 | 534.7 | 18.760 | 503.0 | 0.7556 |
| P_perm_delta_fused_score_a0.01_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 30 | 25 | 0.150 | 352.3 | 0.250 | 532.1 | 19.820 | 495.0 | 0.7622 |
| P_perm_delta_fused_score_a0.01_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 30 | 50 | 0.200 | 439.0 | 0.200 | 537.9 | 19.200 | 509.6 | 0.7567 |
| P_perm_delta_fused_score_a0.01_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 50 | 25 | 0.050 | 340.4 | 0.150 | 450.8 | 20.460 | 476.2 | 0.7549 |
| P_perm_delta_fused_score_a0.01_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 50 | 50 | 0.100 | 391.6 | 0.250 | 538.5 | 20.680 | 466.2 | 0.7568 |
| P_perm_delta_fused_score_a0.02_pre200_post10_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 10 | 25 | 0.200 | 495.0 | 0.350 | 703.0 | 17.320 | 562.5 | 0.7602 |
| P_perm_delta_fused_score_a0.02_pre200_post10_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 10 | 50 | 0.100 | 404.8 | 0.300 | 560.3 | 16.820 | 597.3 | 0.7509 |
| P_perm_delta_fused_score_a0.02_pre200_post20_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 20 | 25 | 0.000 | 318.4 | 0.300 | 605.1 | 18.560 | 527.8 | 0.7569 |
| P_perm_delta_fused_score_a0.02_pre200_post20_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 20 | 50 | 0.100 | 462.4 | 0.150 | 530.5 | 19.540 | 500.4 | 0.7572 |
| P_perm_delta_fused_score_a0.02_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 30 | 25 | 0.100 | 340.1 | 0.050 | 314.2 | 19.640 | 488.1 | 0.7549 |
| P_perm_delta_fused_score_a0.02_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 30 | 50 | 0.050 | 339.8 | 0.150 | 486.7 | 19.420 | 508.9 | 0.7562 |
| P_perm_delta_fused_score_a0.02_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 50 | 25 | 0.000 | 356.1 | 0.200 | 481.2 | 21.120 | 455.9 | 0.7631 |
| P_perm_delta_fused_score_a0.02_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 50 | 50 | 0.100 | 416.9 | 0.100 | 371.8 | 20.060 | 479.5 | 0.7653 |
| P_perm_delta_fused_score_a0.02_pre500_post10_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 10 | 25 | 0.300 | 523.8 | 0.400 | 706.5 | 17.060 | 573.0 | 0.7580 |
| P_perm_delta_fused_score_a0.02_pre500_post10_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 10 | 50 | 0.150 | 451.2 | 0.100 | 433.6 | 17.920 | 541.3 | 0.7538 |
| P_perm_delta_fused_score_a0.02_pre500_post20_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 20 | 25 | 0.050 | 381.4 | 0.300 | 632.6 | 19.580 | 489.5 | 0.7580 |
| P_perm_delta_fused_score_a0.02_pre500_post20_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 20 | 50 | 0.300 | 651.5 | 0.150 | 413.4 | 19.920 | 486.0 | 0.7644 |
| P_perm_delta_fused_score_a0.02_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 30 | 25 | 0.100 | 442.2 | 0.300 | 638.7 | 19.320 | 492.7 | 0.7647 |
| P_perm_delta_fused_score_a0.02_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 30 | 50 | 0.100 | 417.6 | 0.100 | 363.5 | 19.220 | 512.3 | 0.7554 |
| P_perm_delta_fused_score_a0.02_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 50 | 25 | 0.150 | 401.9 | 0.250 | 619.2 | 20.260 | 487.2 | 0.7619 |
| P_perm_delta_fused_score_a0.02_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 50 | 50 | 0.000 | 282.5 | 0.250 | 660.1 | 21.100 | 461.5 | 0.7566 |
| P_perm_delta_fused_score_a0.05_pre200_post10_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 10 | 25 | 0.150 | 429.7 | 0.350 | 598.0 | 20.480 | 468.3 | 0.7628 |
| P_perm_delta_fused_score_a0.05_pre200_post10_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 10 | 50 | 0.000 | 303.6 | 0.250 | 569.9 | 18.780 | 514.9 | 0.7603 |
| P_perm_delta_fused_score_a0.05_pre200_post20_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 20 | 25 | 0.050 | 306.8 | 0.200 | 509.7 | 20.080 | 486.8 | 0.7575 |
| P_perm_delta_fused_score_a0.05_pre200_post20_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 20 | 50 | 0.150 | 464.9 | 0.300 | 571.8 | 19.860 | 488.6 | 0.7564 |
| P_perm_delta_fused_score_a0.05_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 30 | 25 | 0.100 | 370.5 | 0.050 | 296.3 | 19.420 | 507.8 | 0.7586 |
| P_perm_delta_fused_score_a0.05_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 30 | 50 | 0.050 | 380.1 | 0.300 | 521.2 | 20.020 | 487.3 | 0.7494 |
| P_perm_delta_fused_score_a0.05_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 50 | 25 | 0.050 | 391.0 | 0.050 | 367.6 | 20.920 | 460.3 | 0.7585 |
| P_perm_delta_fused_score_a0.05_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 50 | 50 | 0.100 | 380.4 | 0.300 | 540.4 | 20.860 | 473.3 | 0.7561 |
| P_perm_delta_fused_score_a0.05_pre500_post10_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 10 | 25 | 0.100 | 431.0 | 0.150 | 401.2 | 17.920 | 564.1 | 0.7569 |
| P_perm_delta_fused_score_a0.05_pre500_post10_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 10 | 50 | 0.100 | 455.3 | 0.250 | 556.8 | 19.520 | 495.7 | 0.7631 |
| P_perm_delta_fused_score_a0.05_pre500_post20_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 20 | 25 | 0.150 | 371.2 | 0.200 | 418.5 | 19.740 | 489.5 | 0.7574 |
| P_perm_delta_fused_score_a0.05_pre500_post20_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 20 | 50 | 0.150 | 520.0 | 0.200 | 568.0 | 20.800 | 462.9 | 0.7550 |
| P_perm_delta_fused_score_a0.05_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 30 | 25 | 0.100 | 418.5 | 0.200 | 535.6 | 19.800 | 504.6 | 0.7618 |
| P_perm_delta_fused_score_a0.05_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 30 | 50 | 0.100 | 374.0 | 0.150 | 529.2 | 20.960 | 459.0 | 0.7581 |
| P_perm_delta_fused_score_a0.05_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 50 | 25 | 0.050 | 325.4 | 0.150 | 473.2 | 22.420 | 436.4 | 0.7626 |
| P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 50 | 50 | 0.050 | 319.3 | 0.200 | 559.3 | 20.520 | 475.0 | 0.7534 |
| P_perm_fused_score_a0.005_pre200_post10_n5 | quick | perm_test | fused_score | 0.005 | 200 | 10 | N/A | 0.150 | 508.1 | 0.150 | 548.1 | 16.660 | 588.1 | 0.7494 |
| P_perm_fused_score_a0.005_pre200_post20_n5 | quick | perm_test | fused_score | 0.005 | 200 | 20 | N/A | 0.300 | 535.0 | 0.150 | 380.8 | 16.800 | 582.0 | 0.7533 |
| P_perm_fused_score_a0.005_pre200_post30_n5 | quick | perm_test | fused_score | 0.005 | 200 | 30 | N/A | 0.200 | 507.8 | 0.250 | 537.9 | 17.180 | 575.2 | 0.7603 |
| P_perm_fused_score_a0.005_pre200_post50_n5 | quick | perm_test | fused_score | 0.005 | 200 | 50 | N/A | 0.200 | 488.3 | 0.150 | 276.4 | 18.280 | 530.9 | 0.7537 |
| P_perm_fused_score_a0.005_pre500_post10_n5 | quick | perm_test | fused_score | 0.005 | 500 | 10 | N/A | 0.350 | 647.0 | 0.400 | 720.6 | 14.900 | 650.2 | 0.7568 |
| P_perm_fused_score_a0.005_pre500_post20_n5 | quick | perm_test | fused_score | 0.005 | 500 | 20 | N/A | 0.000 | 332.4 | 0.100 | 273.2 | 15.080 | 645.4 | 0.7556 |
| P_perm_fused_score_a0.005_pre500_post30_n5 | quick | perm_test | fused_score | 0.005 | 500 | 30 | N/A | 0.200 | 535.0 | 0.200 | 480.6 | 16.660 | 586.6 | 0.7606 |
| P_perm_fused_score_a0.005_pre500_post50_n5 | quick | perm_test | fused_score | 0.005 | 500 | 50 | N/A | 0.200 | 595.5 | 0.150 | 468.8 | 16.680 | 579.9 | 0.7630 |
| P_perm_fused_score_a0.01_pre200_post10_n5 | quick | perm_test | fused_score | 0.01 | 200 | 10 | N/A | 0.150 | 449.6 | 0.200 | 479.0 | 16.240 | 603.7 | 0.7601 |
| P_perm_fused_score_a0.01_pre200_post20_n5 | quick | perm_test | fused_score | 0.01 | 200 | 20 | N/A | 0.100 | 452.8 | 0.150 | 418.5 | 17.260 | 568.0 | 0.7573 |
| P_perm_fused_score_a0.01_pre200_post30_n5 | quick | perm_test | fused_score | 0.01 | 200 | 30 | N/A | 0.250 | 517.7 | 0.050 | 310.7 | 17.420 | 561.8 | 0.7566 |
| P_perm_fused_score_a0.01_pre200_post50_n5 | quick | perm_test | fused_score | 0.01 | 200 | 50 | N/A | 0.250 | 459.8 | 0.050 | 267.2 | 18.060 | 535.8 | 0.7478 |
| P_perm_fused_score_a0.01_pre500_post10_n5 | quick | perm_test | fused_score | 0.01 | 500 | 10 | N/A | 0.200 | 532.1 | 0.150 | 480.6 | 14.660 | 671.0 | 0.7555 |
| P_perm_fused_score_a0.01_pre500_post20_n5 | quick | perm_test | fused_score | 0.01 | 500 | 20 | N/A | 0.200 | 537.9 | 0.050 | 381.4 | 16.200 | 608.2 | 0.7581 |
| P_perm_fused_score_a0.01_pre500_post30_n5 | quick | perm_test | fused_score | 0.01 | 500 | 30 | N/A | 0.250 | 498.2 | 0.100 | 393.6 | 16.540 | 578.2 | 0.7517 |
| P_perm_fused_score_a0.01_pre500_post50_n5 | quick | perm_test | fused_score | 0.01 | 500 | 50 | N/A | 0.150 | 389.1 | 0.100 | 411.8 | 17.080 | 555.4 | 0.7599 |
| P_perm_fused_score_a0.02_pre200_post10_n5 | quick | perm_test | fused_score | 0.02 | 200 | 10 | N/A | 0.200 | 530.8 | 0.100 | 342.4 | 15.400 | 653.0 | 0.7597 |
| P_perm_fused_score_a0.02_pre200_post20_n5 | quick | perm_test | fused_score | 0.02 | 200 | 20 | N/A | 0.150 | 522.2 | 0.100 | 360.6 | 18.400 | 526.8 | 0.7601 |
| P_perm_fused_score_a0.02_pre200_post30_n5 | quick | perm_test | fused_score | 0.02 | 200 | 30 | N/A | 0.250 | 459.5 | 0.150 | 481.2 | 17.420 | 559.3 | 0.7621 |
| P_perm_fused_score_a0.02_pre200_post50_n5 | quick | perm_test | fused_score | 0.02 | 200 | 50 | N/A | 0.200 | 505.9 | 0.050 | 243.2 | 18.340 | 522.0 | 0.7691 |
| P_perm_fused_score_a0.02_pre500_post10_n5 | quick | perm_test | fused_score | 0.02 | 500 | 10 | N/A | 0.250 | 532.8 | 0.250 | 614.0 | 14.940 | 651.6 | 0.7496 |
| P_perm_fused_score_a0.02_pre500_post20_n5 | quick | perm_test | fused_score | 0.02 | 500 | 20 | N/A | 0.100 | 493.1 | 0.100 | 287.0 | 16.300 | 603.0 | 0.7516 |
| P_perm_fused_score_a0.02_pre500_post30_n5 | quick | perm_test | fused_score | 0.02 | 500 | 30 | N/A | 0.250 | 550.0 | 0.100 | 381.4 | 16.800 | 578.8 | 0.7652 |
| P_perm_fused_score_a0.02_pre500_post50_n5 | quick | perm_test | fused_score | 0.02 | 500 | 50 | N/A | 0.050 | 306.5 | 0.100 | 330.2 | 17.180 | 565.9 | 0.7659 |
| P_perm_fused_score_a0.05_pre200_post10_n5 | quick | perm_test | fused_score | 0.05 | 200 | 10 | N/A | 0.150 | 491.5 | 0.250 | 520.9 | 17.180 | 556.3 | 0.7604 |
| P_perm_fused_score_a0.05_pre200_post20_n5 | quick | perm_test | fused_score | 0.05 | 200 | 20 | N/A | 0.100 | 404.4 | 0.300 | 613.7 | 18.260 | 535.2 | 0.7621 |
| P_perm_fused_score_a0.05_pre200_post30_n5 | quick | perm_test | fused_score | 0.05 | 200 | 30 | N/A | 0.200 | 463.0 | 0.200 | 474.2 | 18.700 | 516.8 | 0.7515 |
| P_perm_fused_score_a0.05_pre200_post50_n5 | quick | perm_test | fused_score | 0.05 | 200 | 50 | N/A | 0.150 | 463.6 | 0.100 | 377.2 | 18.700 | 520.4 | 0.7641 |
| P_perm_fused_score_a0.05_pre500_post10_n5 | quick | perm_test | fused_score | 0.05 | 500 | 10 | N/A | 0.150 | 449.6 | 0.150 | 416.0 | 15.720 | 610.3 | 0.7600 |
| P_perm_fused_score_a0.05_pre500_post20_n5 | quick | perm_test | fused_score | 0.05 | 500 | 20 | N/A | 0.200 | 568.0 | 0.150 | 365.7 | 17.080 | 568.4 | 0.7560 |
| P_perm_fused_score_a0.05_pre500_post30_n5 | quick | perm_test | fused_score | 0.05 | 500 | 30 | N/A | 0.200 | 481.6 | 0.100 | 369.6 | 17.280 | 565.1 | 0.7550 |
| P_perm_fused_score_a0.05_pre500_post50_n5 | quick | perm_test | fused_score | 0.05 | 500 | 50 | N/A | 0.150 | 394.8 | 0.000 | 239.3 | 17.960 | 548.6 | 0.7613 |
| P_perm_vote_score_a0.005_pre200_post10_n5 | quick | perm_test | vote_score | 0.005 | 200 | 10 | N/A | 0.000 | 290.5 | 0.000 | 249.2 | 26.400 | 369.7 | 0.7633 |
| P_perm_vote_score_a0.005_pre200_post20_n5 | quick | perm_test | vote_score | 0.005 | 200 | 20 | N/A | 0.000 | 274.8 | 0.000 | 276.8 | 25.400 | 385.7 | 0.7607 |
| P_perm_vote_score_a0.005_pre200_post30_n5 | quick | perm_test | vote_score | 0.005 | 200 | 30 | N/A | 0.000 | 301.4 | 0.000 | 239.6 | 26.440 | 374.0 | 0.7547 |
| P_perm_vote_score_a0.005_pre200_post50_n5 | quick | perm_test | vote_score | 0.005 | 200 | 50 | N/A | 0.000 | 305.9 | 0.000 | 191.3 | 25.860 | 378.5 | 0.7554 |
| P_perm_vote_score_a0.005_pre500_post10_n5 | quick | perm_test | vote_score | 0.005 | 500 | 10 | N/A | 0.000 | 242.8 | 0.000 | 235.8 | 25.560 | 380.5 | 0.7643 |
| P_perm_vote_score_a0.005_pre500_post20_n5 | quick | perm_test | vote_score | 0.005 | 500 | 20 | N/A | 0.000 | 265.6 | 0.000 | 232.3 | 25.840 | 382.1 | 0.7575 |
| P_perm_vote_score_a0.005_pre500_post30_n5 | quick | perm_test | vote_score | 0.005 | 500 | 30 | N/A | 0.000 | 302.4 | 0.000 | 218.5 | 26.300 | 372.9 | 0.7644 |
| P_perm_vote_score_a0.005_pre500_post50_n5 | quick | perm_test | vote_score | 0.005 | 500 | 50 | N/A | 0.000 | 284.1 | 0.000 | 193.6 | 26.360 | 369.2 | 0.7718 |
| P_perm_vote_score_a0.01_pre200_post10_n5 | quick | perm_test | vote_score | 0.01 | 200 | 10 | N/A | 0.000 | 258.2 | 0.000 | 255.6 | 26.560 | 364.4 | 0.7646 |
| P_perm_vote_score_a0.01_pre200_post20_n5 | quick | perm_test | vote_score | 0.01 | 200 | 20 | N/A | 0.000 | 252.1 | 0.000 | 212.4 | 26.240 | 374.9 | 0.7683 |
| P_perm_vote_score_a0.01_pre200_post30_n5 | quick | perm_test | vote_score | 0.01 | 200 | 30 | N/A | 0.000 | 225.9 | 0.000 | 233.9 | 25.620 | 382.4 | 0.7623 |
| P_perm_vote_score_a0.01_pre200_post50_n5 | quick | perm_test | vote_score | 0.01 | 200 | 50 | N/A | 0.000 | 255.3 | 0.000 | 241.2 | 26.640 | 367.9 | 0.7595 |
| P_perm_vote_score_a0.01_pre500_post10_n5 | quick | perm_test | vote_score | 0.01 | 500 | 10 | N/A | 0.000 | 252.8 | 0.000 | 232.9 | 25.640 | 375.3 | 0.7612 |
| P_perm_vote_score_a0.01_pre500_post20_n5 | quick | perm_test | vote_score | 0.01 | 500 | 20 | N/A | 0.000 | 281.2 | 0.000 | 286.4 | 25.940 | 375.9 | 0.7624 |
| P_perm_vote_score_a0.01_pre500_post30_n5 | quick | perm_test | vote_score | 0.01 | 500 | 30 | N/A | 0.000 | 248.0 | 0.000 | 252.1 | 25.820 | 378.8 | 0.7655 |
| P_perm_vote_score_a0.01_pre500_post50_n5 | quick | perm_test | vote_score | 0.01 | 500 | 50 | N/A | 0.000 | 246.0 | 0.000 | 248.0 | 27.100 | 359.7 | 0.7713 |
| P_perm_vote_score_a0.02_pre200_post10_n5 | quick | perm_test | vote_score | 0.02 | 200 | 10 | N/A | 0.000 | 252.4 | 0.000 | 267.5 | 26.440 | 373.7 | 0.7601 |
| P_perm_vote_score_a0.02_pre200_post20_n5 | quick | perm_test | vote_score | 0.02 | 200 | 20 | N/A | 0.000 | 279.0 | 0.000 | 229.1 | 25.800 | 381.1 | 0.7600 |
| P_perm_vote_score_a0.02_pre200_post30_n5 | quick | perm_test | vote_score | 0.02 | 200 | 30 | N/A | 0.000 | 237.4 | 0.000 | 282.8 | 26.460 | 373.1 | 0.7576 |
| P_perm_vote_score_a0.02_pre200_post50_n5 | quick | perm_test | vote_score | 0.02 | 200 | 50 | N/A | 0.000 | 275.5 | 0.000 | 232.0 | 25.540 | 377.2 | 0.7571 |
| P_perm_vote_score_a0.02_pre500_post10_n5 | quick | perm_test | vote_score | 0.02 | 500 | 10 | N/A | 0.000 | 235.5 | 0.000 | 249.9 | 26.940 | 369.5 | 0.7642 |
| P_perm_vote_score_a0.02_pre500_post20_n5 | quick | perm_test | vote_score | 0.02 | 500 | 20 | N/A | 0.000 | 282.5 | 0.000 | 225.6 | 24.920 | 390.6 | 0.7612 |
| P_perm_vote_score_a0.02_pre500_post30_n5 | quick | perm_test | vote_score | 0.02 | 500 | 30 | N/A | 0.000 | 256.9 | 0.000 | 249.2 | 26.180 | 373.2 | 0.7607 |
| P_perm_vote_score_a0.02_pre500_post50_n5 | quick | perm_test | vote_score | 0.02 | 500 | 50 | N/A | 0.000 | 287.3 | 0.000 | 222.7 | 25.920 | 379.7 | 0.7575 |
| P_perm_vote_score_a0.05_pre200_post10_n5 | quick | perm_test | vote_score | 0.05 | 200 | 10 | N/A | 0.000 | 222.0 | 0.000 | 232.9 | 26.680 | 368.3 | 0.7602 |
| P_perm_vote_score_a0.05_pre200_post20_n5 | quick | perm_test | vote_score | 0.05 | 200 | 20 | N/A | 0.000 | 261.4 | 0.000 | 209.9 | 26.500 | 367.4 | 0.7574 |
| P_perm_vote_score_a0.05_pre200_post30_n5 | quick | perm_test | vote_score | 0.05 | 200 | 30 | N/A | 0.000 | 224.0 | 0.000 | 267.5 | 26.040 | 370.5 | 0.7577 |
| P_perm_vote_score_a0.05_pre200_post50_n5 | quick | perm_test | vote_score | 0.05 | 200 | 50 | N/A | 0.000 | 292.8 | 0.000 | 237.1 | 26.880 | 364.8 | 0.7635 |
| P_perm_vote_score_a0.05_pre500_post10_n5 | quick | perm_test | vote_score | 0.05 | 500 | 10 | N/A | 0.000 | 219.5 | 0.000 | 262.4 | 26.260 | 370.1 | 0.7609 |
| P_perm_vote_score_a0.05_pre500_post20_n5 | quick | perm_test | vote_score | 0.05 | 500 | 20 | N/A | 0.000 | 256.9 | 0.000 | 251.2 | 26.160 | 375.5 | 0.7578 |
| P_perm_vote_score_a0.05_pre500_post30_n5 | quick | perm_test | vote_score | 0.05 | 500 | 30 | N/A | 0.000 | 257.2 | 0.000 | 262.0 | 25.280 | 392.5 | 0.7597 |
| P_perm_vote_score_a0.05_pre500_post50_n5 | quick | perm_test | vote_score | 0.05 | 500 | 50 | N/A | 0.000 | 213.4 | 0.000 | 242.5 | 26.160 | 376.1 | 0.7633 |

**winner 选择规则（写死）**
- Step1: sea_abrupt4 & sine_abrupt4 满足 miss_tol500_mean==0 且 conf_P90_mean<500；Step2: 最小化 no-drift confirm_rate_per_10k（sea_nodrift+sine_nodrift 平均）；Step3: 并列时最大化 no-drift MTFA_win；Step4: drift_acc_final_mean 不低于 best-0.01。

**winner**
- `P_perm_vote_score_a0.005_pre500_post10_n5`
- no-drift confirm_rate_per_10k 下降幅度（vs baseline）：26.540 → 25.560 (Δ=-0.980)

## Track AM：机制诊断（可选）
- - winner：`P_perm_vote_score_a0.005_pre500_post10_n5`（见 Track AL）
- - no-drift confirm_rate_per_10k（平均）变化：26.540 → 25.560 (Δ=-0.980)
- - perm_test 最佳降误报候选（网格内）：`P_perm_fused_score_a0.01_pre500_post10_n5` no_drift_rate=14.660, sea(miss=0.200,confP90=532.1), sine(miss=0.150,confP90=480.6)
- - Track AM：已生成 `artifacts/v15/tables/TRACKAM_PERM_DIAG_V15.csv` rows=8（p-value 分布/confirmed-candidate 比例见诊断表）

## 可复制运行命令
- `source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- `python experiments/trackAL_perm_confirm_sweep.py`
- `python experiments/trackAM_perm_diagnostics.py`  （可选）
- `python scripts/summarize_next_stage_v15.py`
