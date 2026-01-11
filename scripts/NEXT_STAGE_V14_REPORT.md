# NEXT_STAGE V14 Report（Permutation-test Confirm）

- 生成时间：2026-01-10 21:28:37
- 环境确认：
  - `which python` -> `/home/ylh/anaconda3/envs/ZZNSTD/bin/python`
  - `python -V` -> `Python 3.10.19`

## 关键口径与硬约束
- drift 约束：`sea_abrupt4` + `sine_abrupt4` 上 `miss_tol500_mean==0` 且 `conf_P90_mean<500`
- 目标：在满足约束下最小化 no-drift `confirm_rate_per_10k`（`sea_nodrift` + `sine_nodrift` 平均；次选最大化 no-drift `MTFA_win`）

## Track AL：Perm-confirm sweep
- 产物：`scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv`

| group | phase | confirm_rule | perm_stat | perm_alpha | perm_pre_n | perm_post_n | delta_k | sea_miss | sea_confP90 | sine_miss | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_weighted_n20 | full | weighted | N/A | N/A | N/A | N/A | N/A | 0.013 | 286.0 | 0.000 | 244.0 | 29.295 | 340.6 | 0.7733 |
| A_weighted_n5 | quick | weighted | N/A | N/A | N/A | N/A | N/A | 0.000 | 271.6 | 0.000 | 245.7 | 29.312 | 340.5 | 0.7660 |
| P_perm_delta_fused_score_a0.005_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 30 | 25 | 0.050 | 360.9 | 0.150 | 520.6 | 20.633 | 480.7 | 0.7614 |
| P_perm_delta_fused_score_a0.005_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 30 | 50 | 0.150 | 456.3 | 0.200 | 421.7 | 20.711 | 478.9 | 0.7606 |
| P_perm_delta_fused_score_a0.005_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 50 | 25 | 0.100 | 449.2 | 0.150 | 450.8 | 22.629 | 438.2 | 0.7639 |
| P_perm_delta_fused_score_a0.005_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 200 | 50 | 50 | 0.150 | 441.2 | 0.300 | 428.4 | 22.207 | 446.7 | 0.7663 |
| P_perm_delta_fused_score_a0.005_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 30 | 25 | 0.300 | 543.3 | 0.150 | 452.8 | 21.066 | 471.2 | 0.7648 |
| P_perm_delta_fused_score_a0.005_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 30 | 50 | 0.100 | 441.2 | 0.150 | 523.8 | 20.615 | 481.2 | 0.7686 |
| P_perm_delta_fused_score_a0.005_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 50 | 25 | 0.050 | 433.6 | 0.150 | 441.2 | 22.150 | 447.7 | 0.7636 |
| P_perm_delta_fused_score_a0.005_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.005 | 500 | 50 | 50 | 0.050 | 371.8 | 0.100 | 496.3 | 21.538 | 461.8 | 0.7642 |
| P_perm_delta_fused_score_a0.01_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 30 | 25 | 0.250 | 564.8 | 0.150 | 397.1 | 21.440 | 463.3 | 0.7617 |
| P_perm_delta_fused_score_a0.01_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 30 | 50 | 0.100 | 404.4 | 0.150 | 478.7 | 21.704 | 459.3 | 0.7673 |
| P_perm_delta_fused_score_a0.01_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 50 | 25 | 0.100 | 452.4 | 0.200 | 424.0 | 22.257 | 444.2 | 0.7651 |
| P_perm_delta_fused_score_a0.01_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 200 | 50 | 50 | 0.100 | 462.0 | 0.200 | 488.9 | 21.974 | 451.3 | 0.7624 |
| P_perm_delta_fused_score_a0.01_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 30 | 25 | 0.250 | 471.3 | 0.250 | 598.7 | 21.560 | 459.0 | 0.7590 |
| P_perm_delta_fused_score_a0.01_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 30 | 50 | 0.150 | 476.1 | 0.250 | 629.7 | 21.619 | 458.5 | 0.7646 |
| P_perm_delta_fused_score_a0.01_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 50 | 25 | 0.150 | 452.1 | 0.400 | 615.6 | 22.587 | 440.4 | 0.7643 |
| P_perm_delta_fused_score_a0.01_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.01 | 500 | 50 | 50 | 0.100 | 448.3 | 0.200 | 593.2 | 22.328 | 443.9 | 0.7642 |
| P_perm_delta_fused_score_a0.02_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 30 | 25 | 0.200 | 576.0 | 0.150 | 384.9 | 21.280 | 466.8 | 0.7625 |
| P_perm_delta_fused_score_a0.02_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 30 | 50 | 0.150 | 415.6 | 0.350 | 606.4 | 21.641 | 456.7 | 0.7660 |
| P_perm_delta_fused_score_a0.02_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 50 | 25 | 0.050 | 343.6 | 0.050 | 358.7 | 22.493 | 440.8 | 0.7574 |
| P_perm_delta_fused_score_a0.02_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 200 | 50 | 50 | 0.050 | 442.8 | 0.300 | 491.5 | 22.352 | 445.5 | 0.7666 |
| P_perm_delta_fused_score_a0.02_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 30 | 25 | 0.100 | 379.2 | 0.150 | 441.9 | 21.882 | 453.0 | 0.7667 |
| P_perm_delta_fused_score_a0.02_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 30 | 50 | 0.100 | 452.8 | 0.150 | 524.1 | 21.980 | 451.7 | 0.7659 |
| P_perm_delta_fused_score_a0.02_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 50 | 25 | 0.000 | 350.7 | 0.150 | 464.6 | 22.650 | 437.6 | 0.7607 |
| P_perm_delta_fused_score_a0.02_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.02 | 500 | 50 | 50 | 0.200 | 451.8 | 0.200 | 483.2 | 23.001 | 430.7 | 0.7622 |
| P_perm_delta_fused_score_a0.05_pre200_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 30 | 25 | 0.100 | 425.6 | 0.250 | 511.0 | 22.071 | 449.0 | 0.7704 |
| P_perm_delta_fused_score_a0.05_pre200_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 30 | 50 | 0.050 | 365.1 | 0.100 | 465.2 | 21.463 | 460.3 | 0.7697 |
| P_perm_delta_fused_score_a0.05_pre200_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 50 | 25 | 0.000 | 314.5 | 0.200 | 397.4 | 23.199 | 427.1 | 0.7624 |
| P_perm_delta_fused_score_a0.05_pre200_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 200 | 50 | 50 | 0.150 | 510.7 | 0.100 | 470.4 | 23.020 | 430.7 | 0.7604 |
| P_perm_delta_fused_score_a0.05_pre500_post30_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 30 | 25 | 0.100 | 399.6 | 0.300 | 558.4 | 22.391 | 442.7 | 0.7607 |
| P_perm_delta_fused_score_a0.05_pre500_post30_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 30 | 50 | 0.150 | 403.8 | 0.200 | 469.4 | 22.608 | 438.9 | 0.7638 |
| P_perm_delta_fused_score_a0.05_pre500_post50_dk25_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 50 | 25 | 0.100 | 400.3 | 0.250 | 600.0 | 23.600 | 420.1 | 0.7741 |
| P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5 | quick | perm_test | delta_fused_score | 0.05 | 500 | 50 | 50 | 0.000 | 326.0 | 0.050 | 347.2 | 23.615 | 419.3 | 0.7629 |
| P_perm_fused_score_a0.005_pre200_post30_n5 | quick | perm_test | fused_score | 0.005 | 200 | 30 | N/A | 0.250 | 504.6 | 0.300 | 587.8 | 18.719 | 530.2 | 0.7704 |
| P_perm_fused_score_a0.005_pre200_post50_n5 | quick | perm_test | fused_score | 0.005 | 200 | 50 | N/A | 0.150 | 419.5 | 0.100 | 342.7 | 19.172 | 517.0 | 0.7643 |
| P_perm_fused_score_a0.005_pre500_post30_n5 | quick | perm_test | fused_score | 0.005 | 500 | 30 | N/A | 0.500 | 740.1 | 0.050 | 317.1 | 17.044 | 582.0 | 0.7649 |
| P_perm_fused_score_a0.005_pre500_post50_n5 | quick | perm_test | fused_score | 0.005 | 500 | 50 | N/A | 0.250 | 485.4 | 0.150 | 470.4 | 17.745 | 557.7 | 0.7616 |
| P_perm_fused_score_a0.01_pre200_post30_n5 | quick | perm_test | fused_score | 0.01 | 200 | 30 | N/A | 0.150 | 426.2 | 0.200 | 342.7 | 19.453 | 509.1 | 0.7656 |
| P_perm_fused_score_a0.01_pre200_post50_n5 | quick | perm_test | fused_score | 0.01 | 200 | 50 | N/A | 0.200 | 517.7 | 0.100 | 321.2 | 20.220 | 490.3 | 0.7623 |
| P_perm_fused_score_a0.01_pre500_post30_n5 | quick | perm_test | fused_score | 0.01 | 500 | 30 | N/A | 0.250 | 568.6 | 0.300 | 707.2 | 17.650 | 561.9 | 0.7671 |
| P_perm_fused_score_a0.01_pre500_post50_n5 | quick | perm_test | fused_score | 0.01 | 500 | 50 | N/A | 0.100 | 473.9 | 0.150 | 412.1 | 18.383 | 538.5 | 0.7633 |
| P_perm_fused_score_a0.02_pre200_post30_n5 | quick | perm_test | fused_score | 0.02 | 200 | 30 | N/A | 0.150 | 411.8 | 0.200 | 376.0 | 20.119 | 493.3 | 0.7667 |
| P_perm_fused_score_a0.02_pre200_post50_n5 | quick | perm_test | fused_score | 0.02 | 200 | 50 | N/A | 0.200 | 457.6 | 0.050 | 285.7 | 20.258 | 490.6 | 0.7579 |
| P_perm_fused_score_a0.02_pre500_post30_n5 | quick | perm_test | fused_score | 0.02 | 500 | 30 | N/A | 0.250 | 527.6 | 0.100 | 340.1 | 17.905 | 552.9 | 0.7639 |
| P_perm_fused_score_a0.02_pre500_post50_n5 | quick | perm_test | fused_score | 0.02 | 500 | 50 | N/A | 0.150 | 531.2 | 0.200 | 490.2 | 18.997 | 522.7 | 0.7622 |
| P_perm_fused_score_a0.05_pre200_post30_n5 | quick | perm_test | fused_score | 0.05 | 200 | 30 | N/A | 0.200 | 505.2 | 0.200 | 385.6 | 19.864 | 498.5 | 0.7687 |
| P_perm_fused_score_a0.05_pre200_post50_n5 | quick | perm_test | fused_score | 0.05 | 200 | 50 | N/A | 0.150 | 425.2 | 0.000 | 173.1 | 20.795 | 476.0 | 0.7599 |
| P_perm_fused_score_a0.05_pre500_post30_n5 | quick | perm_test | fused_score | 0.05 | 500 | 30 | N/A | 0.200 | 527.6 | 0.050 | 235.2 | 18.730 | 529.7 | 0.7633 |
| P_perm_fused_score_a0.05_pre500_post50_n5 | quick | perm_test | fused_score | 0.05 | 500 | 50 | N/A | 0.150 | 607.0 | 0.150 | 602.5 | 19.545 | 506.8 | 0.7615 |

**winner 选择规则（写死）**
- Step1: sea_abrupt4 & sine_abrupt4 满足 miss_tol500_mean==0 且 conf_P90_mean<500；Step2: 最小化 no-drift confirm_rate_per_10k（sea_nodrift+sine_nodrift 平均）；Step3: 并列时最大化 no-drift MTFA_win；Step4: drift_acc_final_mean 不低于 best-0.01。

**winner**
- `A_weighted_n5`
- no-drift confirm_rate_per_10k 下降幅度（vs baseline）：29.295 → 29.312 (Δ=+0.017)

## Track AM：机制诊断（可选）
- - winner：`A_weighted_n5`（见 Track AL）
- - no-drift confirm_rate_per_10k（平均）变化：29.295 → 29.312 (Δ=+0.017)
- - perm_test 最佳降误报候选（网格内）：`P_perm_fused_score_a0.005_pre500_post30_n5` no_drift_rate=17.044, sea(miss=0.500,confP90=740.1), sine(miss=0.050,confP90=317.1)
- - Track AM：已生成 `scripts/TRACKAM_PERM_DIAG.csv` rows=8（p-value 分布/confirmed-candidate 比例见诊断表）

## 可复制运行命令
- `source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- `python experiments/trackAL_perm_confirm_sweep.py`
- `python experiments/trackAM_perm_diagnostics.py`  （可选）
- `python scripts/summarize_next_stage_v14.py`
