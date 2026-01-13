# NEXT_STAGE V15.2 Report（TopK + 双最优点 + 验收前置）

- 生成时间：2026-01-12 23:58:39
- TrackAL：`artifacts/v15p2/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P2.csv`
- TrackAM：`artifacts/v15p2/tables/TRACKAM_PERM_DIAG_V15P2.csv`
- 全量报告：`artifacts/v15p2/reports/NEXT_STAGE_V15P2_REPORT_FULL.md`（含全表）

## 说明（口径）
- V15 在 V14 pipeline 上新增 `vote_score` 的 `perm_test` 统计量分支，并支持更小的 `post_n` 与分片并行。
- 本节验收：只在 hard-ok 集合（sea/sine miss==0 且 confP90<500）内比较。
- baseline：`A_weighted_n5` no_drift_rate=27.050（一律取本次聚合/metrics_table 口径）。

## 双最优点（前置）
- best_drift_acc_among_hard_ok=0.7641（只在 hard-ok 集合内取最大）
- best_acceptance：`P_perm_vote_score_a0.02_pre500_post10_n5` (no_drift_rate=26.340, Δ_vs_baseline=-0.710; MTFA=371.4; drift_acc=0.7624)
- best_with_guardrail：`P_perm_vote_score_a0.02_pre500_post10_n5` (no_drift_rate=26.340, Δ_vs_baseline=-0.710; MTFA=371.4; drift_acc=0.7624)

## 稳定性复核摘要（前置）
- best_acceptance vs baseline：Δ_no_drift_rate mean=-0.710 var=1.0521 p10=-2.420 p50=-0.500 p90=0.310
- best_with_guardrail vs baseline：Δ_no_drift_rate mean=-0.710 var=1.0521 p10=-2.420 p50=-0.500 p90=0.310
- hard constraint 破坏：存在最差 case group=P_perm_vote_score_a0.02_pre500_post10_n5 dataset=stagger_abrupt3 seed=3 miss=0.5 confP90=10000.0

## Top-K Hard-OK candidates（K=20）
| group | perm_stat | perm_alpha | perm_pre_n | perm_post_n | pass_guardrail | sea_miss | sine_miss | sea_confP90 | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P_perm_vote_score_a0.02_pre500_post10_n5 | vote_score | 0.02 | 500 | 10 | True | 0.000 | 0.000 | 257.4 | 228.6 | 26.340 | 371.4 | 0.7624 |
| P_perm_vote_score_a0.03_pre500_post10_n5 | vote_score | 0.03 | 500 | 10 | True | 0.000 | 0.000 | 234.4 | 228.1 | 26.570 | 368.9 | 0.7640 |
| P_perm_vote_score_a0.025_pre500_post10_n5 | vote_score | 0.025 | 500 | 10 | True | 0.000 | 0.000 | 285.7 | 226.7 | 26.840 | 364.1 | 0.7641 |
| A_weighted_n5 | N/A | N/A | N/A | N/A | True | 0.000 | 0.000 | 279.5 | 226.8 | 27.050 | 370.1 | 0.7621 |

## 产物
- `artifacts/v15p2/reports/NEXT_STAGE_V15P2_REPORT.md`
- `artifacts/v15p2/reports/NEXT_STAGE_V15P2_REPORT_FULL.md`
