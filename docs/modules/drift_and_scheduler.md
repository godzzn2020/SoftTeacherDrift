# 漂移信号与超参调度

## 漂移信号

- 位于 `drift/signals.py`，对每个 batch 计算：
  - `batch_error_rate`：学生在有标签样本上的错误率；
  - `batch_teacher_entropy`：教师概率分布的平均熵，反映不确定性；
  - `batch_divergence`：教师与学生预测之间的 Jensen–Shannon 或 KL 散度。
- `compute_signals(...)` 返回一个字典，供训练循环写入日志和监控。

## 漂移检测器

- `drift/detectors.py` 中的 `DriftMonitor` 封装 river 检测器：
  - 通过 `MONITOR_PRESETS` 管理多种组合（当前实现）：
    - `error_ph_meta`：仅监控学生误差率，使用 `PageHinkley(delta=0.005, alpha=0.15, threshold=0.2, min_instances=25)`；
    - `divergence_ph_meta`：仅监控 JS 散度，使用 `PageHinkley(delta=0.005, alpha=0.1, threshold=0.05, min_instances=30)`；
    - `error_divergence_ph_meta`：同时监控误差率与散度。
  - `build_default_monitor(preset="error_ph_meta")` 会根据预设构建对应的 `DriftMonitor`。
  - `update(values, step)`：
    - 通过 `SIGNAL_ALIASES` 从 `values` 中抽取对应的信号（例如 `"error_rate" → student_error_rate"`，`"divergence" → divergence_js`）；
    - 调用 river 的 `detector.update(value)`；
    - 通过 `get_drift_flag(detector)` 统一读取 `drift_detected` 标志，若任一检测器触发则置 `drift_flag=True`；
    - 以当前值与上一次值的差的绝对值作为该信号的增量，取所有信号中的最大值作为本批次 `severity`；
    - 将发生漂移的 `step` 追加到 `history`，更新 `last_drift_step`。
- 检测值应避开 `NaN` / 空批次（训练循环已做防护）。

### PageHinkley 预设（信号组合）

| preset 名称 | 监控信号 | PageHinkley 参数 |
|-------------|-----------|------------------|
| `error_only_ph_meta` / `error_ph_meta` | `student_error_rate` | `delta=0.005, alpha=0.15, threshold=0.2, min_instances=25` |
| `entropy_only_ph_meta` | `teacher_entropy` | `delta=0.01, alpha=0.3, threshold=0.5, min_instances=20` |
| `divergence_only_ph_meta` / `divergence_ph_meta` | `divergence_js` | `delta=0.005, alpha=0.1, threshold=0.05, min_instances=30` |
| `error_entropy_ph_meta` | `student_error_rate + teacher_entropy` | 同上两种信号各自使用对应参数 |
| `error_divergence_ph_meta` | `student_error_rate + divergence_js` | 同上两种信号各自使用对应参数 |
| `entropy_divergence_ph_meta` | `teacher_entropy + divergence_js` | 同上两种信号各自使用对应参数 |
| `all_signals_ph_meta` | `student_error_rate + teacher_entropy + divergence_js` | 三种信号分别使用各自的参数 |

> 说明：多个信号组合时，会为每个信号创建独立的 PageHinkley 实例，只要任一信号触发就视为漂移。旧名称 `error_ph_meta` 与 `divergence_ph_meta` 仍保持兼容，等价于单信号版本。

## 超参调度（Scheduler）

- `scheduler/hparam_scheduler.py` 维护 `SchedulerState` 与 `HParams`：
  - Regime 划分：
    - **stable**：无漂移或长期稳定 → 更高 `alpha`，更小 `lr`，较大 `lambda_u`。
    - **mild_drift**：检测器报警且严重度不大 → 略降 `alpha`、略升 `lr`、适度降低 `lambda_u`、适度降低 `tau`。
    - **severe_drift**：严重漂移 → 显著降低 `alpha`、提升 `lr`、减小 `lambda_u`、下调 `tau`。
  - `update_hparams` 输出新的超参并返回当前 regime，训练循环据此写入日志与调节优化器。

## 关键代码片段

- 漂移检测器预设与更新逻辑（`drift/detectors.py`）：

  ```python
  MONITOR_PRESETS = {
      "error_ph_meta": _build_error_ph_meta,          # error_rate + PH
      "divergence_ph_meta": _build_divergence_ph_meta,# divergence_js + PH
      "error_divergence_ph_meta": _build_error_divergence_meta,
  }

  SIGNAL_ALIASES = {
      "error_rate": ("error_rate", "student_error_rate"),
      "divergence": ("divergence", "divergence_js"),
      "teacher_entropy": ("teacher_entropy",),
  }

  def get_drift_flag(detector):
      if hasattr(detector, "drift_detected"):
          return bool(detector.drift_detected)
      if hasattr(detector, "change_detected"):
          return bool(detector.change_detected)
      return False

  class DriftMonitor:
      def update(self, values: Dict[str, float], step: int) -> Tuple[bool, float]:
          drift_flag = False
          severity = 0.0
          for name, detector in self.detectors.items():
              value = _resolve_signal(values, name)   # 根据别名取 error_rate / divergence_js
              if isinstance(value, float) and math.isnan(value):
                  continue
              detector.update(value)
              if get_drift_flag(detector):
                  drift_flag = True
                  delta = abs(value - self._prev_values.get(name, value))
                  severity = max(severity, delta)
                  self._prev_values[name] = value
          if drift_flag:
              self.history.append(step)
              self.last_drift_step = step
          return drift_flag, severity
  ```

- 调度器核心逻辑（`scheduler/hparam_scheduler.py`）：

  ```python
  def update_hparams(state: SchedulerState, prev: HParams,
                     drift_flag: bool, severity: float) -> Tuple[HParams, str]:
      mild_threshold, severe_threshold = 0.05, 0.15
      if math.isnan(severity):
          severity = 0.0
      regime = "stable"
      if drift_flag:
          state.last_drift_step = state.step
          if severity >= severe_threshold:
              regime = "severe_drift"
          elif severity >= mild_threshold:
              regime = "mild_drift"
      else:
          if state.time_since_drift() > 20:
              regime = "stable"
          else:
              regime = "mild_drift"
      target = _regime_target(regime, state.base_hparams)
      return HParams(
          alpha=_blend(prev.alpha,     target.alpha,     0.5),
          lr=_blend(prev.lr,           target.lr,        0.5),
          lambda_u=_blend(prev.lambda_u, target.lambda_u,0.5),
          tau=_blend(prev.tau,         target.tau,       0.5),
      ), regime
  ```

## 框架级算法流程（检测 + 适应）

以下是 `training/loop.py` 中一整个 batch 的高层流程（简化）：

1. 从 `batch_stream` 拿到 `(X_labeled, y_labeled, X_unlabeled, y_unlabeled_true)`，通过 `FeatureVectorizer` / `LabelEncoder` 转成张量。
2. 学生网络前向、计算监督 + 无监督损失，并通过 `TeacherStudentModel.update_teacher(alpha)` 做 EMA 更新教师。
3. 调用 `_collect_statistics` 从 `LossOutputs.details` 中取出：
   - `teacher_probs`、`student_probs_unlabeled`（无标签部分概率）；
   - `student_logits_labeled`（有标签部分 logits）。
   然后用 `compute_signals(...)` 得到：
   - `error_rate`（学生分类错误率）；
   - `teacher_entropy`（教师预测熵）；
   - `divergence`（教师–学生 JS 散度）。
4. 将这些信号传入 `DriftMonitor.update(signals, step)`：
   - 根据预设选择监控 `error_rate` / `divergence` 或两者；
   - 如果某条检测器触发，返回 `drift_flag=True` 与当前批次的 `severity`。
5. 调用 `update_hparams(scheduler_state, current_hparams, drift_flag, severity)`：
   - 根据 `drift_flag` 和 `severity` 选择 regime（stable/mild_drift/severe_drift 或短期 mild）；
   - 计算新的目标超参并与当前值平滑插值，得到新的 `(alpha, lr, lambda_u, tau)`；
   - 在训练循环中使用新的 `lr` 更新优化器，并将新的超参与 regime 写入日志。
6. 日志中同时记录：
   - 漂移信号：`student_error_rate`, `teacher_entropy`, `divergence_js`；
   - 检测输出：`drift_flag`, `drift_severity`, `regime`；
   - 调度结果：`alpha`, `lr`, `lambda_u`, `tau`。

这使得后续的 offline/online 评估脚本可以直接基于日志中的 `sample_idx`、`drift_flag` 与 meta 中的真值位置计算 MDR/MTD/MTFA/MTR，并可用 `monitor_preset` 快速切换不同检测配置。

## TODO

- 根据实际评估结果调整阈值（例如 severity 划分），并支持按指标准确率动态调节。
- 引入多检测器融合策略或权重，减少单一检测器误报影响。
- 将未来的 Lukats 指标反馈（如 MTD/MDR）纳入长周期超参调度策略。
