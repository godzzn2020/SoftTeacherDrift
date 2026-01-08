# V7-Track Q：口径一致性审计

## acc_min 的来源与定义（代码位置）
- `training/loop.py`：训练结束写 sidecar `*.summary.json` 时，`acc_min = df["metric_accuracy"].min()`（不跳过 warmup、不平滑、不做窗口聚合）。
- `scripts/summarize_next_stage_v6.py`：仅从 Track CSV/summary 中读取 `acc_min` 做聚合，不会二次处理。

## V6 不一致现象复现（同参对齐，run_id 不同）
| seed | TrackO_run_id | TrackP_run_id | acc_min_raw(O) | acc_min_raw(P) | Δraw(P-O) | acc_min@2000(O) | acc_min@2000(P) | Δwarm(P-O) |
|---|---|---|---|---|---|---|---|---|
| 1 | 20260108-103723-gug_theta0.50_w1_cd0 | 20260108-110543-ey9_baseline_theta0.50_w1_cd0 | 0.682292 | 0.682292 | 0.000000 | 0.770020 | 0.766113 | -0.003906 |
| 2 | 20260108-103723-gug_theta0.50_w1_cd0 | 20260108-110543-ey9_baseline_theta0.50_w1_cd0 | 0.703125 | 0.712500 | 0.009375 | 0.739258 | 0.743652 | 0.004395 |
| 3 | 20260108-103723-gug_theta0.50_w1_cd0 | 20260108-110543-ey9_baseline_theta0.50_w1_cd0 | 0.328125 | 0.661458 | 0.333333 | 0.728693 | 0.737689 | 0.008996 |
| 4 | 20260108-103723-gug_theta0.50_w1_cd0 | 20260108-110543-ey9_baseline_theta0.50_w1_cd0 | 0.593750 | 0.625000 | 0.031250 | 0.762695 | 0.746582 | -0.016113 |
| 5 | 20260108-103723-gug_theta0.50_w1_cd0 | 20260108-110543-ey9_baseline_theta0.50_w1_cd0 | 0.312500 | 0.406250 | 0.093750 | 0.748047 | 0.723633 | -0.024414 |

## 根因结论
- 不一致不是“计算口径不同”，而是 **不同 run_id 的实际轨迹不同**；`acc_min_raw` 对早期瞬时下探极敏感。
- 当改用 `acc_min@sample_idx>=2000`（跳过早期 warmup），Track O / Track P 的差异显著收敛（Δwarm 的均值更接近 0）。

## 最终采用口径（V7 统一）
- sea 统一采用：`acc_min@sample_idx>=2000`（并在表中同时保留 `acc_min_raw` 作为参考）。
- INSECTS 仍以 `post_min@W1000` 作为“谷底”主口径（与 V5/V6 一致）。
