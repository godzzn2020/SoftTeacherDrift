# NEXT_STAGE V15.1 Report（TopK + 验收前置）

- 生成时间：2026-01-12 20:25:46
- TrackAL：`scripts/TRACKAL_PERM_CONFIRM_SWEEP_V15P1.csv`
- TrackAM：`scripts/TRACKAM_PERM_DIAG_V15P1.csv`
- 全量报告：`scripts/NEXT_STAGE_V15P1_REPORT_FULL.md`（含全表）

## 说明（口径）
- V15 在 V14 pipeline 上新增 `vote_score` 的 `perm_test` 统计量分支，并支持更小的 `post_n` 与分片并行。
- 本节验收：只检查是否存在满足硬约束（sea/sine `miss==0` 且 `confP90<500`）且 no-drift 低于 baseline 的 `perm_test` 配置。
- 参考：历史 V14 报告中 `A_weighted_n5` no_drift_rate=26.880（仅供对照，不作为本次 baseline）。
- 本次 baseline：`A_weighted_n5` no_drift_rate=27.280（取自本次 TrackAL 聚合/metrics_table 口径）。

## 验收结论（前置）
- ✅ 存在满足硬约束且优于 baseline 的 perm_test：`P_perm_vote_score_a0.02_pre500_post10_n5` (no_drift_rate=25.060, Δ=-2.220; MTFA=391.4; drift_acc=0.7582)

## Top-K Hard-OK candidates（K=20）
| group | perm_stat | perm_alpha | perm_pre_n | perm_post_n | sea_miss | sine_miss | sea_confP90 | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P_perm_vote_score_a0.02_pre500_post10_n5 | vote_score | 0.02 | 500 | 10 | 0.000 | 0.000 | 236.1 | 234.5 | 25.060 | 391.4 | 0.7582 |
| P_perm_vote_score_a0.025_pre500_post10_n5 | vote_score | 0.025 | 500 | 10 | 0.000 | 0.000 | 279.0 | 202.2 | 25.140 | 393.3 | 0.7619 |
| P_perm_vote_score_a0.015_pre500_post10_n5 | vote_score | 0.015 | 500 | 10 | 0.000 | 0.000 | 272.6 | 279.6 | 25.660 | 383.6 | 0.7575 |
| P_perm_vote_score_a0.01_pre500_post10_n5 | vote_score | 0.01 | 500 | 10 | 0.000 | 0.000 | 290.8 | 228.8 | 25.860 | 377.0 | 0.7637 |
| P_perm_vote_score_a0.015_pre500_post20_n5 | vote_score | 0.015 | 500 | 20 | 0.000 | 0.000 | 239.0 | 242.5 | 25.900 | 381.5 | 0.7545 |
| P_perm_vote_score_a0.01_pre500_post20_n5 | vote_score | 0.01 | 500 | 20 | 0.000 | 0.000 | 280.0 | 246.0 | 25.900 | 377.2 | 0.7676 |
| P_perm_vote_score_a0.01_pre500_post30_n5 | vote_score | 0.01 | 500 | 30 | 0.000 | 0.000 | 238.7 | 250.5 | 25.980 | 378.1 | 0.7614 |
| P_perm_vote_score_a0.025_pre500_post20_n5 | vote_score | 0.025 | 500 | 20 | 0.000 | 0.000 | 276.8 | 218.8 | 25.980 | 370.7 | 0.7632 |
| P_perm_vote_score_a0.015_pre500_post30_n5 | vote_score | 0.015 | 500 | 30 | 0.000 | 0.000 | 245.7 | 235.2 | 26.080 | 375.8 | 0.7640 |
| P_perm_vote_score_a0.03_pre500_post10_n5 | vote_score | 0.03 | 500 | 10 | 0.000 | 0.000 | 261.4 | 218.5 | 26.180 | 365.8 | 0.7687 |
| P_perm_vote_score_a0.03_pre500_post20_n5 | vote_score | 0.03 | 500 | 20 | 0.000 | 0.000 | 265.6 | 195.2 | 26.360 | 364.1 | 0.7591 |
| P_perm_vote_score_a0.03_pre500_post30_n5 | vote_score | 0.03 | 500 | 30 | 0.000 | 0.000 | 288.0 | 205.7 | 26.500 | 362.7 | 0.7620 |
| P_perm_vote_score_a0.025_pre500_post30_n5 | vote_score | 0.025 | 500 | 30 | 0.000 | 0.000 | 245.1 | 228.4 | 26.560 | 368.7 | 0.7644 |
| P_perm_vote_score_a0.02_pre500_post30_n5 | vote_score | 0.02 | 500 | 30 | 0.000 | 0.000 | 266.2 | 271.0 | 26.580 | 363.4 | 0.7615 |
| P_perm_vote_score_a0.02_pre500_post20_n5 | vote_score | 0.02 | 500 | 20 | 0.000 | 0.000 | 276.8 | 213.7 | 26.940 | 360.4 | 0.7671 |

## 产物
- `scripts/NEXT_STAGE_V15P1_REPORT.md`
- `scripts/NEXT_STAGE_V15P1_REPORT_FULL.md`
