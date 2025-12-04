# 训练循环与日志

## 流程概览

1. 从 `batch_stream` 获取 `(X_labeled, y_labeled, X_unlabeled, y_unlabeled_true)`。
2. `FeatureVectorizer` / `LabelEncoder` 转换成张量。
3. 学生前向 + 损失：监督 CE + 无监督（伪标签 + 一致性）。
4. 反向传播与学生参数更新；EMA 更新教师。
5. 计算漂移信号，交给 `DriftMonitor` 得到 `drift_flag` 与 `monitor_severity`，再由 `SeverityCalibrator` 将 error/divergence/entropy 的相对增量转换为 `drift_severity_raw` / 归一化 `drift_severity`。
6. `update_hparams` 或 `update_hparams_with_severity` 输出新的超参与 regime（严重度版本会在漂移时按 s_norm × `severity_scheduler_scale` 进一步缩放 alpha/lr/lambda_u/tau）。当 `TrainingConfig.severity_scheduler_scale=0.0` 时就完全退化为 baseline；`>1.0` 会得到更激进的调节。随后调整优化器学习率。
7. 更新在线指标（Accuracy、Cohen Kappa），记录完整日志。
8. 若配置 `log_path`，在结束后写 CSV；同时函数返回 DataFrame 供脚本使用。

## 日志字段

每个 batch 对应一行，核心字段：

- **定位**：`step`（从 1 开始）、`seen_samples`（截至当前的累计样本数）。
- **实验信息**：`dataset_name`, `dataset_type`, `model_variant`, `seed`.
- **指标**：`metric_accuracy`, `metric_kappa`.
- **漂移信号**：`student_error_rate`, `teacher_entropy`, `divergence_js`.
- **检测输出**：`drift_flag`（0/1）, `monitor_severity`（检测器的增量），`drift_severity_raw`（SeverityCalibrator 加权求和后的原值），`drift_severity`（压缩到 [0,1] 的严重度），`regime`.
- **调参**：`alpha`, `lr`, `lambda_u`, `tau`.
- **损失与时间**：`supervised_loss`, `unsupervised_loss`, `timestamp`.

## log_path 使用

- `TrainingConfig.log_path` 控制日志写入位置（默认 `None` 表示仅返回 DataFrame）。
- 推荐命名：`logs/{dataset_name}/{dataset_name}__{model_variant}__seed{seed}.csv`。
- `run_experiment.py` 与批量实验脚本都会传入该路径，保证日志可被评估脚本直接读取。

## TODO

- 在日志中追加资源监控字段（如耗时、GPU 占用）以便后续排查。
- 为漂移检测器提供更细粒度的事件流（例如 drift_id）。
- 允许在线可视化 hook（例如将日志推送到 TensorBoard 或 wandb）。
