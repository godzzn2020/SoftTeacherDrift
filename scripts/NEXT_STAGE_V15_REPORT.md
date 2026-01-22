# NEXT_STAGE V15 Report（TopK + 双最优点 + 验收前置）

- 生成时间：2026-01-13 00:00:21
- TrackAL：`scripts/TRACKAL_PERM_CONFIRM_SWEEP_V15.csv`
- TrackAM：`scripts/TRACKAM_PERM_DIAG_V15.csv`
- 全量报告：`scripts/NEXT_STAGE_V15_REPORT_FULL.md`（含全表）

## 说明（口径）
- V15 在 V14 pipeline 上新增 `vote_score` 的 `perm_test` 统计量分支，并支持更小的 `post_n` 与分片并行。
- 本节验收：只在 hard-ok 集合（sea/sine miss==0 且 confP90<500）内比较。
- baseline：`A_weighted_n5` no_drift_rate=26.540（一律取本次聚合/metrics_table 口径）。

## 双最优点（前置）
- best_drift_acc_among_hard_ok=0.7718（只在 hard-ok 集合内取最大）
- best_acceptance：`P_perm_vote_score_a0.02_pre500_post20_n5` (no_drift_rate=24.920, Δ_vs_baseline=-1.620; MTFA=390.6; drift_acc=0.7612)
- best_with_guardrail：`P_perm_vote_score_a0.005_pre500_post10_n5` (no_drift_rate=25.560, Δ_vs_baseline=-0.980; MTFA=380.5; drift_acc=0.7643)
- 注：best_acceptance 与 best_with_guardrail 不同，原因是 Step4 guardrail 过滤导致。

## Top-K Hard-OK candidates（K=20）
| group | perm_stat | perm_alpha | perm_pre_n | perm_post_n | pass_guardrail | sea_miss | sine_miss | sea_confP90 | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P_perm_vote_score_a0.02_pre500_post20_n5 | vote_score | 0.02 | 500 | 20 | False | 0.000 | 0.000 | 282.5 | 225.6 | 24.920 | 390.6 | 0.7612 |
| P_perm_vote_score_a0.05_pre500_post30_n5 | vote_score | 0.05 | 500 | 30 | False | 0.000 | 0.000 | 257.2 | 262.0 | 25.280 | 392.5 | 0.7597 |
| P_perm_vote_score_a0.005_pre200_post20_n5 | vote_score | 0.005 | 200 | 20 | False | 0.000 | 0.000 | 274.8 | 276.8 | 25.400 | 385.7 | 0.7607 |
| P_perm_vote_score_a0.02_pre200_post50_n5 | vote_score | 0.02 | 200 | 50 | False | 0.000 | 0.000 | 275.5 | 232.0 | 25.540 | 377.2 | 0.7571 |
| P_perm_vote_score_a0.005_pre500_post10_n5 | vote_score | 0.005 | 500 | 10 | True | 0.000 | 0.000 | 242.8 | 235.8 | 25.560 | 380.5 | 0.7643 |
| P_perm_vote_score_a0.01_pre200_post30_n5 | vote_score | 0.01 | 200 | 30 | True | 0.000 | 0.000 | 225.9 | 233.9 | 25.620 | 382.4 | 0.7623 |
| P_perm_vote_score_a0.01_pre500_post10_n5 | vote_score | 0.01 | 500 | 10 | False | 0.000 | 0.000 | 252.8 | 232.9 | 25.640 | 375.3 | 0.7612 |
| P_perm_vote_score_a0.02_pre200_post20_n5 | vote_score | 0.02 | 200 | 20 | False | 0.000 | 0.000 | 279.0 | 229.1 | 25.800 | 381.1 | 0.7600 |
| P_perm_vote_score_a0.01_pre500_post30_n5 | vote_score | 0.01 | 500 | 30 | True | 0.000 | 0.000 | 248.0 | 252.1 | 25.820 | 378.8 | 0.7655 |
| P_perm_vote_score_a0.005_pre500_post20_n5 | vote_score | 0.005 | 500 | 20 | False | 0.000 | 0.000 | 265.6 | 232.3 | 25.840 | 382.1 | 0.7575 |
| P_perm_vote_score_a0.005_pre200_post50_n5 | vote_score | 0.005 | 200 | 50 | False | 0.000 | 0.000 | 305.9 | 191.3 | 25.860 | 378.5 | 0.7554 |
| P_perm_vote_score_a0.02_pre500_post50_n5 | vote_score | 0.02 | 500 | 50 | False | 0.000 | 0.000 | 287.3 | 222.7 | 25.920 | 379.7 | 0.7575 |
| P_perm_vote_score_a0.01_pre500_post20_n5 | vote_score | 0.01 | 500 | 20 | True | 0.000 | 0.000 | 281.2 | 286.4 | 25.940 | 375.9 | 0.7624 |
| P_perm_vote_score_a0.05_pre200_post30_n5 | vote_score | 0.05 | 200 | 30 | False | 0.000 | 0.000 | 224.0 | 267.5 | 26.040 | 370.5 | 0.7577 |
| P_perm_vote_score_a0.05_pre500_post20_n5 | vote_score | 0.05 | 500 | 20 | False | 0.000 | 0.000 | 256.9 | 251.2 | 26.160 | 375.5 | 0.7578 |
| P_perm_vote_score_a0.05_pre500_post50_n5 | vote_score | 0.05 | 500 | 50 | True | 0.000 | 0.000 | 213.4 | 242.5 | 26.160 | 376.1 | 0.7633 |
| P_perm_vote_score_a0.02_pre500_post30_n5 | vote_score | 0.02 | 500 | 30 | False | 0.000 | 0.000 | 256.9 | 249.2 | 26.180 | 373.2 | 0.7607 |
| P_perm_vote_score_a0.01_pre200_post20_n5 | vote_score | 0.01 | 200 | 20 | True | 0.000 | 0.000 | 252.1 | 212.4 | 26.240 | 374.9 | 0.7683 |
| P_perm_vote_score_a0.05_pre500_post10_n5 | vote_score | 0.05 | 500 | 10 | False | 0.000 | 0.000 | 219.5 | 262.4 | 26.260 | 370.1 | 0.7609 |
| P_perm_vote_score_a0.005_pre500_post30_n5 | vote_score | 0.005 | 500 | 30 | True | 0.000 | 0.000 | 302.4 | 218.5 | 26.300 | 372.9 | 0.7644 |

## 产物
- `scripts/NEXT_STAGE_V15_REPORT.md`
- `scripts/NEXT_STAGE_V15_REPORT_FULL.md`
