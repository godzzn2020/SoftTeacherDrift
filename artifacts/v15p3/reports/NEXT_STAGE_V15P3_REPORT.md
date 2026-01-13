# NEXT_STAGE V15.3 Report（TopK + 双最优点 + 验收前置）

- 生成时间：2026-01-13 01:18:33
- TrackAL：`artifacts/v15p3/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P3.csv`
- TrackAM：`artifacts/v15p3/tables/TRACKAM_PERM_DIAG_V15P3.csv`
- 全量报告：`artifacts/v15p3/reports/NEXT_STAGE_V15P3_REPORT_FULL.md`（含全表）

## 验收结论（失败 case 定向诊断 + 最小修复验证）
- sea/sine hard-ok：成立（best_acceptance=`P_perm_vote_score_a0.03_pre500_post10_n5` 在 `sea_abrupt4/sine_abrupt4` 上均为 miss=0 且 confP90<500）。
- no-drift 相对 baseline：仍平均更低（best_acceptance vs `A_weighted_n5`：Δ_no_drift_rate mean=-0.570，p50=-0.800，p90=0.250）。
- stagger_abrupt3 seed=3 worst-case：已修复（原 V15.2 的 `miss=0.5, confP90=10000` 由 GT anchors 口径错误触发；本次同组 `P_perm_vote_score_a0.02_pre500_post10_n5` 在 `stagger_abrupt3 seed=3` 上 miss=0 且 confP90≈59.8）。
- stagger 诊断证据来源：来自本次 stability runs 的 `.summary.json`（同样被 `experiments/trackAM_perm_diagnostics.py` 汇总进 `artifacts/v15p3/tables/TRACKAM_PERM_DIAG_V15P3.csv`，包含 `stagger_abrupt3` 行）。

## 说明（口径）
- V15 在 V14 pipeline 上新增 `vote_score` 的 `perm_test` 统计量分支，并支持更小的 `post_n` 与分片并行。
- 本节验收：只在 hard-ok 集合（sea/sine miss==0 且 confP90<500）内比较。
- baseline：`A_weighted_n5` no_drift_rate=26.780（一律取本次聚合/metrics_table 口径）。

## 双最优点（前置）
- best_drift_acc_among_hard_ok=0.7676（只在 hard-ok 集合内取最大）
- best_acceptance：`P_perm_vote_score_a0.03_pre500_post10_n5` (no_drift_rate=26.210, Δ_vs_baseline=-0.570; MTFA=374.5; drift_acc=0.7624)
- best_with_guardrail：`P_perm_vote_score_a0.03_pre500_post10_n5` (no_drift_rate=26.210, Δ_vs_baseline=-0.570; MTFA=374.5; drift_acc=0.7624)

## 稳定性复核摘要（前置）
- best_acceptance vs baseline：Δ_no_drift_rate mean=-0.570 var=0.8446 p10=-1.430 p50=-0.800 p90=0.250
- best_with_guardrail vs baseline：Δ_no_drift_rate mean=-0.570 var=0.8446 p10=-1.430 p50=-0.800 p90=0.250
- hard constraint 破坏：存在最差 case group=P_perm_vote_score_a0.025_pre500_post10_n5_side2s dataset=sine_abrupt4 seed=1 miss=0.25 confP90=576.6000000000001

## Top-K Hard-OK candidates（K=20）
| group | perm_stat | perm_alpha | perm_pre_n | perm_post_n | perm_side | pass_guardrail | sea_miss | sine_miss | sea_confP90 | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P_perm_vote_score_a0.03_pre500_post10_n5 | vote_score | 0.03 | 500 | 10 | one_sided_pos | True | 0.000 | 0.000 | 258.2 | 179.2 | 26.210 | 374.5 | 0.7624 |
| P_perm_vote_score_a0.03_pre500_post10_n5_side2s | vote_score | 0.03 | 500 | 10 | two_sided | True | 0.000 | 0.000 | 263.6 | 214.7 | 26.300 | 370.9 | 0.7624 |
| P_perm_vote_score_a0.02_pre500_post10_n5_side2s | vote_score | 0.02 | 500 | 10 | two_sided | True | 0.000 | 0.000 | 262.7 | 222.4 | 26.450 | 368.3 | 0.7671 |
| P_perm_vote_score_a0.02_pre500_post10_n5 | vote_score | 0.02 | 500 | 10 | one_sided_pos | True | 0.000 | 0.000 | 267.3 | 239.5 | 26.470 | 366.3 | 0.7647 |
| P_perm_vote_score_a0.025_pre500_post10_n5 | vote_score | 0.025 | 500 | 10 | one_sided_pos | True | 0.000 | 0.000 | 267.0 | 241.7 | 26.520 | 365.7 | 0.7631 |
| A_weighted_n5 | N/A | N/A | N/A | N/A | N/A | True | 0.000 | 0.000 | 242.7 | 242.0 | 26.780 | 371.9 | 0.7676 |

## 产物
- `artifacts/v15p3/reports/NEXT_STAGE_V15P3_REPORT.md`
- `artifacts/v15p3/reports/NEXT_STAGE_V15P3_REPORT_FULL.md`
