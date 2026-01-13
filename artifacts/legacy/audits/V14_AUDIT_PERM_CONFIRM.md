# V14 审计（Permutation-test Confirm）

- 生成时间：2026-01-11 15:24:17
- 输入范围：严格按 prompt 列表仅读取指定 CSV/MD；run drill-down 仅尝试 `artifacts/v14/logs/<run_id>/*.summary.json`。

## 1) 硬约束复核（表格级）
- drift 约束：`sea_abrupt4` + `sine_abrupt4` 上 `miss_tol500_mean==0` 且 `conf_P90_mean<500`
- 满足硬约束 Step1 的组数：1（应为 1 或极少）
- 复核 winner（按 CSV 计算）：`A_weighted_n5`；report 声明：`A_weighted_n5` -> 一致

| group | sea_miss | sea_confP90 | sine_miss | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_weighted_n5 | 0 | 271.64 | 0 | 245.72 | 29.312 | 340.5 | 0.766 |

- drift_acc_final_mean 容差口径：best=0.766，允许 >= best-0.01

### 1.1 Top-5 “最接近满足硬约束”的 perm_test 组
- 排序：`sea_miss+sine_miss` 升序；再 `max(sea_confP90,sine_confP90)` 升序；再 `no_drift_rate` 升序

| rank | group | perm_alpha | pre_n | post_n | delta_k | sea_miss | sine_miss | sea_confP90 | sine_confP90 | no_drift_rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5 | 0.05 | 500 | 50 | 50 | 0 | 0.05 | 326.04 | 347.16 | 23.615 |
| 2 | P_perm_delta_fused_score_a0.02_pre200_post50_dk25_n5 | 0.02 | 200 | 50 | 25 | 0.05 | 0.05 | 343.64 | 358.68 | 22.493 |
| 3 | P_perm_fused_score_a0.05_pre200_post50_n5 | 0.05 | 200 | 50 | N/A | 0.15 | 0 | 425.24 | 173.08 | 20.795 |
| 4 | P_perm_delta_fused_score_a0.02_pre500_post50_dk25_n5 | 0.02 | 500 | 50 | 25 | 0 | 0.15 | 350.68 | 464.6 | 22.65 |
| 5 | P_perm_delta_fused_score_a0.05_pre200_post30_dk50_n5 | 0.05 | 200 | 30 | 50 | 0.05 | 0.1 | 365.08 | 465.24 | 21.463 |

## 2) Track AM 诊断：dataset×group 审计汇总表
- 数据源：`scripts/TRACKAM_PERM_DIAG.csv`（注意：该文件当前只覆盖 2 个 group，缺失项已标 N/A，不做臆测）

| group | dataset | pvalue_P50 | pvalue_P90 | pvalue_P99 | p_le_alpha_ratio | accept_ratio* | candidate_count_mean | confirmed_over_candidate | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A_weighted_n20 | sea_abrupt4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| A_weighted_n20 | sine_abrupt4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| A_weighted_n20 | sea_nodrift | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| A_weighted_n20 | sine_nodrift | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| A_weighted_n5 | sea_abrupt4 | N/A | N/A | N/A | N/A | N/A | 148.4 | 1 |  |
| A_weighted_n5 | sine_abrupt4 | N/A | N/A | N/A | N/A | N/A | 148.2 | 1 |  |
| A_weighted_n5 | sea_nodrift | N/A | N/A | N/A | N/A | N/A | 148.4 | 1 |  |
| A_weighted_n5 | sine_nodrift | N/A | N/A | N/A | N/A | N/A | 148.2 | 1 |  |
| P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5 | sea_abrupt4 | 0.2184 | 1 | 1 | 0.4683 | 0.4683 | 136.4 | 0.893 |  |
| P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5 | sine_abrupt4 | 0.2478 | 1 | 1 | 0.4616 | 0.4616 | 131.2 | 0.8932 |  |
| P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5 | sea_nodrift | 0.2184 | 1 | 1 | 0.4683 | 0.4683 | 136.4 | 0.893 |  |
| P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5 | sine_nodrift | 0.2478 | 1 | 1 | 0.4616 | 0.4616 | 131.2 | 0.8932 |  |
| P_perm_delta_fused_score_a0.02_pre200_post50_dk25_n5 | sea_abrupt4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_delta_fused_score_a0.02_pre200_post50_dk25_n5 | sine_abrupt4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_delta_fused_score_a0.02_pre200_post50_dk25_n5 | sea_nodrift | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_delta_fused_score_a0.02_pre200_post50_dk25_n5 | sine_nodrift | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_fused_score_a0.05_pre200_post50_n5 | sea_abrupt4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_fused_score_a0.05_pre200_post50_n5 | sine_abrupt4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_fused_score_a0.05_pre200_post50_n5 | sea_nodrift | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_fused_score_a0.05_pre200_post50_n5 | sine_nodrift | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_fused_score_a0.005_pre500_post30_n5 | sea_abrupt4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_fused_score_a0.005_pre500_post30_n5 | sine_abrupt4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_fused_score_a0.005_pre500_post30_n5 | sea_nodrift | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |
| P_perm_fused_score_a0.005_pre500_post30_n5 | sine_nodrift | N/A | N/A | N/A | N/A | N/A | N/A | N/A | diag缺失->N/A |

*注：本次将 `accept_ratio` 记为 `perm_pvalue<=alpha` 的比例（`perm_pvalue_le_alpha_ratio_mean`）；原表无 `accept_count/test_count` 字段。*

### 2.1 归因结论（<=12 行）
1) 主因：B（窗口对齐/统计量错配）；次因：A（power 不足/不稳定）。
2) 判据：该 perm_test 组在 AM 里 `pvalue_P90=1.0` 且 `pvalue_P99=1.0`，但 `p_le_alpha_ratio≈0.4683`（同一表可复核）→ pvalue 呈“极小/极大”混合，更像窗口/统计量对齐敏感而非稳定显著。
3) 与硬约束冲突的直接证据：最接近硬约束的 `P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5` 仍有 `sine_miss=0.05`（硬约束要求为 0）→ 即便 `confP90<500` 也会被淘汰。
4) no-drift 降误报的代价：no-drift 最低的 `P_perm_fused_score_a0.005_pre500_post30_n5` 虽 `no_drift_rate≈17.044`，但在 `sea_abrupt4` 上 `sea_miss≈0.5` 且 `sea_confP90≈740.12`（>500）→ 同时破坏 miss 与延迟两条硬约束。
5) C（状态机/逻辑副作用）的证据不足：当前 perm_test 的 `confirmed_over_candidate_mean≈0.893` 并不低；且本次 drill-down 允许路径下未找到 summary.json，无法进一步核验“accept 触发但确认未发生”。

## 3) 逐 run drill-down（3 组；每组<=2 run_id）
- 选择组：winner / 最接近硬约束（rank1）/ no-drift 最低
- 约束：只按 `artifacts/v14/tables/NEXT_STAGE_V14_RUN_INDEX.csv` 取 run_id；只尝试打开 `artifacts/v14/logs/<run_id>/*.summary.json`。

### group: `A_weighted_n5`
- run_id：`20260110-195141-kkz_A_weighted_n5`
  - summary 定位：目录不存在：logs/20260110-195141-kkz_A_weighted_n5
  - 字段：N/A（summary 文件不存在或不在允许路径下）

### group: `P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5`
- run_id：`20260110-202442-pl1_P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5`
  - summary 定位：目录不存在：logs/20260110-202442-pl1_P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5
  - 字段：N/A（summary 文件不存在或不在允许路径下）

### group: `P_perm_fused_score_a0.005_pre500_post30_n5`
- run_id：`20260110-201159-3kp_P_perm_fused_score_a0.005_pre500_post30_n5`
  - summary 定位：目录不存在：logs/20260110-201159-3kp_P_perm_fused_score_a0.005_pre500_post30_n5
  - 字段：N/A（summary 文件不存在或不在允许路径下）

## 4) 最终解释：为何 perm_test 网格内无法同时满足硬约束并降低 no-drift 误报
从 `TRACKAL_PERM_CONFIRM_SWEEP.csv` 的同一套数值可复核到：一旦 perm_test 把 `no_drift_rate` 压下去（例如最低约 `17.044`），drift 侧会出现 `miss_tol500_mean` 上升和/或 `conf_P90_mean` 推迟到 >500；而满足硬约束的只有 `A_weighted_n5`，其 `no_drift_rate≈29.312` 明显更高。在当前可用的 `TRACKAM_PERM_DIAG.csv` 里，perm_test 的 pvalue 高分位（P90/P99=1.0）与 `p_le_alpha_ratio` 并存也更像对齐/稳定性问题在不同窗口上被放大，形成“降低误报 ↔ 增加 drift miss/延迟”的结构性冲突。
