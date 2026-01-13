# NEXT_STAGE V14 Report（Permutation-test Confirm）

- 生成时间：2026-01-11 20:30:47
- 环境确认：
  - `which python` -> `/home/ylh/anaconda3/envs/ZZNSTD/bin/python`
  - `python -V` -> `Python 3.10.19`

## 关键口径与硬约束
- drift 约束：`sea_abrupt4` + `sine_abrupt4` 上 `miss_tol500_mean==0` 且 `conf_P90_mean<500`
- 目标：在满足约束下最小化 no-drift `confirm_rate_per_10k`（`sea_nodrift` + `sine_nodrift` 平均；次选最大化 no-drift `MTFA_win`）

## Track AL：Perm-confirm sweep
- 产物：`artifacts/v14/tables/TRACKAL_PERM_CONFIRM_SWEEP.csv`

| group | phase | confirm_rule | perm_stat | perm_alpha | perm_pre_n | perm_post_n | delta_k | sea_miss | sea_confP90 | sine_miss | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_weighted_n5 | quick | weighted | N/A | N/A | N/A | N/A | N/A | 0.000 | 273.9 | 0.000 | 232.0 | 26.880 | 373.1 | 0.7658 |
| P_perm_delta_fused_score_a0.005_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 30 | 25 | 0.150 | 437.4 | 0.200 | 490.2 | 18.860 | 502.4 | 0.7570 |
| P_perm_delta_fused_score_a0.005_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 30 | 50 | 0.200 | 531.5 | 0.100 | 439.0 | 17.800 | 551.4 | 0.7523 |
| P_perm_delta_fused_score_a0.005_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 50 | 25 | 0.200 | 512.6 | 0.150 | 517.7 | 19.500 | 502.8 | 0.7634 |
| P_perm_delta_fused_score_a0.005_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 50 | 50 | 0.200 | 392.3 | 0.000 | 196.4 | 20.660 | 472.8 | 0.7515 |
| P_perm_delta_fused_score_a0.005_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 30 | 25 | 0.100 | 382.0 | 0.300 | 539.5 | 19.340 | 502.5 | 0.7580 |
| P_perm_delta_fused_score_a0.005_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 30 | 50 | 0.200 | 466.5 | 0.200 | 490.2 | 19.260 | 506.6 | 0.7644 |
| P_perm_delta_fused_score_a0.005_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 50 | 25 | 0.000 | 283.2 | 0.050 | 378.2 | 20.240 | 481.0 | 0.7660 |
| P_perm_delta_fused_score_a0.005_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 50 | 50 | 0.150 | 440.0 | 0.050 | 243.2 | 20.080 | 487.2 | 0.7604 |
| P_perm_delta_fused_score_a0.01_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 30 | 25 | 0.150 | 438.0 | 0.250 | 458.8 | 19.280 | 502.1 | 0.7590 |
| P_perm_delta_fused_score_a0.01_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 30 | 50 | 0.050 | 375.6 | 0.150 | 381.7 | 19.740 | 499.4 | 0.7638 |
| P_perm_delta_fused_score_a0.01_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 50 | 25 | 0.200 | 449.9 | 0.300 | 514.5 | 20.860 | 467.9 | 0.7553 |
| P_perm_delta_fused_score_a0.01_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 50 | 50 | 0.150 | 568.3 | 0.200 | 476.8 | 20.240 | 486.0 | 0.7583 |
| P_perm_delta_fused_score_a0.01_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 30 | 25 | 0.100 | 406.7 | 0.200 | 470.0 | 19.620 | 501.3 | 0.7583 |
| P_perm_delta_fused_score_a0.01_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 30 | 50 | 0.300 | 586.5 | 0.150 | 476.4 | 19.380 | 497.3 | 0.7562 |
| P_perm_delta_fused_score_a0.01_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 50 | 25 | 0.100 | 464.0 | 0.200 | 553.9 | 20.140 | 488.4 | 0.7571 |
| P_perm_delta_fused_score_a0.01_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 50 | 50 | 0.050 | 321.9 | 0.250 | 637.1 | 20.400 | 486.0 | 0.7561 |
| P_perm_delta_fused_score_a0.02_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 30 | 25 | 0.150 | 353.6 | 0.250 | 529.6 | 19.700 | 498.3 | 0.7561 |
| P_perm_delta_fused_score_a0.02_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 30 | 50 | 0.050 | 417.6 | 0.200 | 398.0 | 19.380 | 504.4 | 0.7593 |
| P_perm_delta_fused_score_a0.02_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 50 | 25 | 0.050 | 370.5 | 0.200 | 460.1 | 20.420 | 476.1 | 0.7584 |
| P_perm_delta_fused_score_a0.02_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 50 | 50 | 0.100 | 469.1 | 0.200 | 367.6 | 20.440 | 478.8 | 0.7516 |
| P_perm_delta_fused_score_a0.02_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 30 | 25 | 0.150 | 392.0 | 0.200 | 548.8 | 19.420 | 496.7 | 0.7631 |
| P_perm_delta_fused_score_a0.02_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 30 | 50 | 0.250 | 500.4 | 0.200 | 522.2 | 19.300 | 502.0 | 0.7584 |
| P_perm_delta_fused_score_a0.02_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 50 | 25 | 0.050 | 357.7 | 0.100 | 462.0 | 21.300 | 459.3 | 0.7604 |
| P_perm_delta_fused_score_a0.02_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 50 | 50 | 0.200 | 424.0 | 0.250 | 573.7 | 21.540 | 451.8 | 0.7566 |
| P_perm_delta_fused_score_a0.05_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 30 | 25 | 0.100 | 461.1 | 0.200 | 406.7 | 19.880 | 493.0 | 0.7636 |
| P_perm_delta_fused_score_a0.05_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 30 | 50 | 0.050 | 335.3 | 0.150 | 385.2 | 19.160 | 523.3 | 0.7556 |
| P_perm_delta_fused_score_a0.05_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 50 | 25 | 0.050 | 341.4 | 0.250 | 514.8 | 20.560 | 475.4 | 0.7580 |
| P_perm_delta_fused_score_a0.05_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 50 | 50 | 0.200 | 453.4 | 0.200 | 469.1 | 20.880 | 462.3 | 0.7584 |
| P_perm_delta_fused_score_a0.05_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 30 | 25 | 0.100 | 408.0 | 0.250 | 557.1 | 21.360 | 453.5 | 0.7490 |
| P_perm_delta_fused_score_a0.05_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 30 | 50 | 0.050 | 399.0 | 0.150 | 547.2 | 19.840 | 492.4 | 0.7563 |
| P_perm_delta_fused_score_a0.05_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 50 | 25 | 0.050 | 391.3 | 0.100 | 332.8 | 21.540 | 446.4 | 0.7589 |
| P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 50 | 50 | 0.150 | 420.1 | 0.000 | 145.2 | 20.640 | 475.1 | 0.7536 |
| P_perm_fused_score_a0.005_pre200_post30_n5 | quick | perm_test | fused_score | 0.005 | 200 | 30 | N/A | 0.150 | 485.7 | 0.250 | 506.8 | 16.500 | 585.2 | 0.7531 |
| P_perm_fused_score_a0.005_pre200_post50_n5 | quick | perm_test | fused_score | 0.005 | 200 | 50 | N/A | 0.300 | 620.4 | 0.250 | 506.2 | 18.060 | 535.9 | 0.7650 |
| P_perm_fused_score_a0.005_pre500_post30_n5 | quick | perm_test | fused_score | 0.005 | 500 | 30 | N/A | 0.150 | 527.3 | 0.150 | 439.0 | 16.660 | 585.7 | 0.7578 |
| P_perm_fused_score_a0.005_pre500_post50_n5 | quick | perm_test | fused_score | 0.005 | 500 | 50 | N/A | 0.150 | 534.0 | 0.100 | 381.7 | 16.920 | 586.0 | 0.7603 |
| P_perm_fused_score_a0.01_pre200_post30_n5 | quick | perm_test | fused_score | 0.01 | 200 | 30 | N/A | 0.200 | 461.4 | 0.050 | 299.8 | 17.920 | 541.8 | 0.7580 |
| P_perm_fused_score_a0.01_pre200_post50_n5 | quick | perm_test | fused_score | 0.01 | 200 | 50 | N/A | 0.100 | 377.6 | 0.100 | 345.6 | 17.520 | 565.4 | 0.7549 |
| P_perm_fused_score_a0.01_pre500_post30_n5 | quick | perm_test | fused_score | 0.01 | 500 | 30 | N/A | 0.050 | 352.0 | 0.100 | 366.7 | 16.920 | 572.1 | 0.7514 |
| P_perm_fused_score_a0.01_pre500_post50_n5 | quick | perm_test | fused_score | 0.01 | 500 | 50 | N/A | 0.150 | 452.4 | 0.300 | 637.4 | 17.160 | 569.0 | 0.7604 |
| P_perm_fused_score_a0.02_pre200_post30_n5 | quick | perm_test | fused_score | 0.02 | 200 | 30 | N/A | 0.300 | 629.1 | 0.250 | 591.6 | 18.520 | 531.8 | 0.7587 |
| P_perm_fused_score_a0.02_pre200_post50_n5 | quick | perm_test | fused_score | 0.02 | 200 | 50 | N/A | 0.150 | 423.0 | 0.050 | 277.7 | 18.980 | 508.7 | 0.7587 |
| P_perm_fused_score_a0.02_pre500_post30_n5 | quick | perm_test | fused_score | 0.02 | 500 | 30 | N/A | 0.200 | 467.2 | 0.100 | 342.7 | 17.200 | 557.3 | 0.7554 |
| P_perm_fused_score_a0.02_pre500_post50_n5 | quick | perm_test | fused_score | 0.02 | 500 | 50 | N/A | 0.100 | 368.0 | 0.200 | 508.4 | 17.820 | 538.4 | 0.7514 |
| P_perm_fused_score_a0.05_pre200_post30_n5 | quick | perm_test | fused_score | 0.05 | 200 | 30 | N/A | 0.200 | 464.3 | 0.050 | 340.8 | 18.700 | 526.8 | 0.7613 |
| P_perm_fused_score_a0.05_pre200_post50_n5 | quick | perm_test | fused_score | 0.05 | 200 | 50 | N/A | 0.000 | 339.2 | 0.100 | 344.6 | 19.400 | 503.0 | 0.7585 |
| P_perm_fused_score_a0.05_pre500_post30_n5 | quick | perm_test | fused_score | 0.05 | 500 | 30 | N/A | 0.200 | 520.6 | 0.250 | 552.3 | 17.260 | 560.0 | 0.7597 |
| P_perm_fused_score_a0.05_pre500_post50_n5 | quick | perm_test | fused_score | 0.05 | 500 | 50 | N/A | 0.050 | 393.6 | 0.100 | 368.3 | 18.840 | 513.8 | 0.7608 |

**winner 选择规则（写死）**
- Step1: sea_abrupt4 & sine_abrupt4 满足 miss_tol500_mean==0 且 conf_P90_mean<500；Step2: 最小化 no-drift confirm_rate_per_10k（sea_nodrift+sine_nodrift 平均）；Step3: 并列时最大化 no-drift MTFA_win；Step4: drift_acc_final_mean 不低于 best-0.01。

**winner**
- `A_weighted_n5`
- no-drift confirm_rate_per_10k 下降幅度（vs baseline）：26.880 → 26.880 (Δ=+0.000)

## Track AM：机制诊断（可选）
- - winner：`A_weighted_n5`（见 Track AL）
- - no-drift confirm_rate_per_10k（平均）变化：26.880 → 26.880 (Δ=+0.000)
- - perm_test 最佳降误报候选（网格内）：`P_perm_fused_score_a0.005_pre200_post30_n5` no_drift_rate=16.500, sea(miss=0.150,confP90=485.7), sine(miss=0.250,confP90=506.8)
- - Track AM：已生成 `artifacts/v14/tables/TRACKAM_PERM_DIAG.csv` rows=8（p-value 分布/confirmed-candidate 比例见诊断表）

## 可复制运行命令
- `source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- `python experiments/trackAL_perm_confirm_sweep.py`
- `python experiments/trackAM_perm_diagnostics.py`  （可选）
- `python scripts/summarize_next_stage_v14.py`
