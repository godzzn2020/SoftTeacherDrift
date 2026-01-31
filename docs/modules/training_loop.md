# 训练循环与日志

## 流程概览

1. 从 `batch_stream` 获取 `(X_labeled, y_labeled, X_unlabeled, y_unlabeled_true)`。
2. `FeatureVectorizer` / `LabelEncoder` 转换成张量。
3. 学生前向 + 损失：监督 CE + 无监督（伪标签 + 一致性）。
4. 反向传播与学生参数更新；EMA 更新教师。
5. 计算漂移信号，交给 `DriftMonitor` 得到 `candidate_flag/drift_flag` 与 `monitor_severity`（并额外记录 `monitor_vote_*`/`monitor_fused_severity`、confirm_delay、perm_test 诊断字段等）。
6. `SeverityCalibrator` 维护三信号基线，并在 `drift_flag=1` 时将 error/divergence/entropy 的相对增量转换为 `drift_severity_raw` / 归一化 `drift_severity`（`entropy_mode` 可切换熵项“危险方向”）。
7. **Severity-Aware v2（可选）**：维护 `severity_carry = max(severity_carry * decay, drift_severity)`，并支持 `freeze_baseline_steps`（漂移后冻结若干步 baseline 更新），使严重度在漂移后的窗口期持续影响调参。
8. `update_hparams` 或 `update_hparams_with_severity` 输出新的超参与 regime：
   - v1：传入 `drift_severity`；
   - v2：传入 `severity_carry`；
   - 严重度版本会在漂移时按 `severity_norm × severity_scheduler_scale` 进一步缩放 alpha/lr/lambda_u/tau。`severity_scheduler_scale=0.0` 退化为 baseline；`>1.0` 更激进。随后调整优化器学习率。
9. 更新在线指标（Accuracy、Cohen Kappa），记录完整日志。
10. 若配置 `log_path`，在结束后写 CSV；并额外写入 `.summary.json`（用于离线 TrackAL/AM 汇总，避免扫描大日志）。

## 日志字段

每个 batch 对应一行，核心字段：

- **定位**：`step`（从 1 开始）、`seen_samples`（累计样本数）、`sample_idx`（0-based 累计样本位置，漂移相关时间戳统一用它）。
- **实验信息**：`dataset_name`, `dataset_type`, `model_variant`, `seed`.
- **指标**：`metric_accuracy`, `metric_kappa`.
- **漂移信号**：`student_error_rate`, `teacher_entropy`, `divergence_js`.
- **检测输出**：
  - `monitor_preset`, `monitor_preset_base`, `monitor_ph_params`, `monitor_ph_overrides`（PH 参数会被写入 JSON，便于离线审计/复现）；
  - `trigger_mode`, `trigger_k`, `trigger_threshold`, `trigger_weights`（用于复现实验配置）；
  - `confirm_window`（two_stage 的候选有效期；单位 step/batch）；
  - `confirm_cooldown`（从 trigger_weights 透传；单位 sample_idx）、`effective_confirm_cooldown`（含自适应冷却后的实际值）；
  - `confirm_cooldown_active`, `confirm_cooldown_remaining`, `confirm_rate_per10k`（冷却诊断/密度统计）；
  - `adaptive_cooldown_enabled` 及 `adaptive_*`（自适应冷却参数，均从 trigger_weights 透传，仅用于复现/审计）；
  - `drift_flag`（0/1）, `monitor_severity`（原口径：仅在 detector 触发时的 max-delta），`monitor_fused_severity`（每步 max-delta），`monitor_vote_count`, `monitor_vote_score`；
  - `candidate_flag`（0/1，two_stage 的候选触发）、`confirm_delay`（candidate→confirm 的步数，非 two_stage 为 -1）；
  - `candidate_count_total` / `confirmed_count_total`（two_stage 的累计计数，便于离线汇总）；
  - `confirm_rule_effective`（实际 confirm 规则：weighted/k_of_n/weighted_error_gate/perm_test）；
  - perm_test（two_stage+perm_test 才有意义）：`last_perm_pvalue`, `last_perm_effect`, `perm_test_count_total`, `perm_accept_count_total`, `perm_reject_count_total`；
  - `drift_severity_raw`（SeverityCalibrator 原值），`drift_severity`（压缩到 [0,1] 的严重度），`severity_carry`（v2 的持续严重度），`regime`.
- **严重度配置**：`severity_scheduler_scale`, `use_severity_v2`, `entropy_mode`, `decay`, `freeze_baseline_steps`。
- **严重度 gating（可选）**：`severity_gate`（`none/confirmed_only`）、`severity_confirmed`（0/1）。
- **调参**：`alpha`, `lr`, `lambda_u`, `tau`.
- **损失与时间**：`supervised_loss`, `unsupervised_loss`, `timestamp`.

## log_path 使用

- `TrainingConfig.log_path` 控制日志写入位置（默认 `None` 表示仅返回 DataFrame）。
- 推荐使用 `soft_drift/utils/run_paths.py:ExperimentRun` 生成结构化路径（见 `docs/modules/outputs_and_logs.md`），并自动写 legacy symlink 以兼容旧脚本。

## `.summary.json` sidecar（TrackAL/AM 数据源）

- 写入位置：`Path(log_path).with_suffix(".summary.json")`（`training/loop.py:run_training_loop`）。
- 目的：为后续 TrackAL/TrackAM 汇总提供 run-level 小文件，避免扫描大 CSV 日志。
- 关键字段（主干）：
  - 配置镜像：`monitor_preset/_base/monitor_ph_params/trigger_mode/trigger_threshold/trigger_weights/confirm_window/...`
  - 轨迹：`candidate_sample_idxs`, `confirmed_sample_idxs`, `acc_series`
  - 计数：`candidate_count_total`, `confirmed_count_total`
  - perm_test 诊断：`perm_alpha`, `perm_pvalue_p50/p90/p99`, `perm_pvalue_le_alpha_ratio`, `perm_effect_p50/p90/p99` 及 accept/reject/test 总数
  - 典型读取脚本：`experiments/trackAL_perm_confirm_stability_v15p3.py:read_run_summary`、`experiments/trackAM_perm_diagnostics.py:read_summary`

## TODO

- 在日志中追加资源监控字段（如耗时、GPU 占用）以便后续排查。
- 为漂移检测器提供更细粒度的事件流（例如 drift_id）。
- 允许在线可视化 hook（例如将日志推送到 TensorBoard 或 wandb）。
