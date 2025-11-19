# 漂移信号与超参调度

## 漂移信号

- 位于 `drift/signals.py`，对每个 batch 计算：
  - `batch_error_rate`：学生在有标签样本上的错误率；
  - `batch_teacher_entropy`：教师概率分布的平均熵，反映不确定性；
  - `batch_divergence`：教师与学生预测之间的 Jensen–Shannon 或 KL 散度。
- `compute_signals(...)` 返回一个字典，供训练循环写入日志和监控。

## 漂移检测器

- `drift/detectors.py` 中的 `DriftMonitor` 封装 river 检测器：
  - 默认为 `ADWIN` 监控错误率、`PageHinkley` 监控散度；
  - `update(values, step)` 返回 `drift_flag` 与 `severity`，并记录检测历史。
- 检测值应避开 `NaN` / 空批次（训练循环已做防护）。

## 超参调度（Scheduler）

- `scheduler/hparam_scheduler.py` 维护 `SchedulerState` 与 `HParams`：
  - Regime 划分：
    - **stable**：无漂移或长期稳定 → 更高 `alpha`，更小 `lr`，较大 `lambda_u`。
    - **mild_drift**：检测器报警且严重度不大 → 略降 `alpha`、略升 `lr`、适度降低 `lambda_u`、适度降低 `tau`。
    - **severe_drift**：严重漂移 → 显著降低 `alpha`、提升 `lr`、减小 `lambda_u`、下调 `tau`。
  - `update_hparams` 输出新的超参并返回当前 regime，训练循环据此写入日志与调节优化器。

## TODO

- 根据实际评估结果调整阈值（例如 severity 划分），并支持按指标准确率动态调节。
- 引入多检测器融合策略或权重，减少单一检测器误报影响。
- 将未来的 Lukats 指标反馈（如 MTD/MDR）纳入长周期超参调度策略。

