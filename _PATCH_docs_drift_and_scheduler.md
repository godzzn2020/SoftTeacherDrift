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
  - `build_default_monitor(preset="error_ph_meta")` 会根据预设构建对应的 `DriftMonitor`，并支持多 detector 的融合触发策略 `trigger_mode`：
    - `or`：任一 detector 触发即 `drift_flag=True`（默认，保持旧行为）。
    - `k_of_n`：至少 `k` 个 detector 同时触发才 `drift_flag=True`（例如 `k=2` 更保守）。
    - `weighted`：计算 `vote_score = Σ w_i * I(detector_i_drift)`，当 `vote_score >= threshold` 才触发。
      - 建议默认权重：`error_rate=0.5, divergence=0.3, teacher_entropy=0.2`；阈值可从 `0.5` 起调（更大更保守）。
    - `two_stage`：两阶段触发（candidate→confirm）
      - candidate：使用 OR（任一 detector 触发即候选）；
      - confirm：候选触发后在 `confirm_window` 步内，若 `vote_score >= threshold` 则确认；
      - confirm_rule：通过 confirm 侧规则控制确认行为（默认 weighted）；也可启用 `perm_test`（置换检验 confirm）。
        - `weighted`：沿用原逻辑（`vote_score >= threshold` 立即确认）。
        - `perm_test`：在 confirm 阶段比较 candidate 前后窗口的统计量（连续值）差异是否显著；no-drift 下通常不显著→拒绝确认。
          - 通过 `trigger_weights` 透传：`__confirm_rule=perm_test`，并可配置 `__perm_pre_n/__perm_post_n/__perm_n_perm/__perm_alpha/__perm_stat/__perm_min_effect/__perm_rng_seed/__perm_delta_k`。
          - `__perm_stat` 支持：`fused_score`（Σ w_i * ph_evidence_i；ph_evidence 来自 PageHinkley 内部累积量）与 `delta_fused_score`（fused_score_t - median(fused_score_{t-k..t-1})）。
      - 只有 confirmed 才置 `drift_flag=1`（候选通过 `candidate_flag` 记录），并记录 `confirm_delay`（candidate→confirm 的步数）。
  - `update(values, step)`：
    - 通过 `SIGNAL_ALIASES` 从 `values` 中抽取对应的信号（例如 `"error_rate" → student_error_rate"`，`"divergence" → divergence_js`）；
    - 调用 river 的 `detector.update(value)`；
    - 通过 `get_drift_flag(detector)` 统一读取 `drift_detected` 标志，并按 `trigger_mode` 计算融合后的 `drift_flag`；
    - `monitor_severity`：保持原定义（只在 detector 触发时，取 `abs(value - prev_on_drift)` 的 max）；
    - `monitor_fused_severity`：新增调试口径（每步 max `abs(value - prev_all)`，与 drift_flag 无关）；
    - 额外记录：`monitor_vote_count`（本步触发的 detector 数量）、`monitor_vote_score`（weighted 时的票分；其它模式也会给出一致口径的得分）。
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

### PageHinkley 参数覆盖（用于延迟优化）

为了在不新增大量 preset 的情况下快速调参，`build_default_monitor(...)` 支持通过 `monitor_preset` 字符串内联覆盖部分 PH 参数（保持 base preset 名称不变）：

- 语法：`<base_preset>@error.threshold=0.1,error.min_instances=10,divergence.threshold=0.02`
- 支持信号前缀：`error` / `divergence` / `entropy`
- 支持参数名：`threshold` / `delta` / `alpha` / `min_instances`

训练日志会记录最终生效的参数，便于离线汇总：

- `monitor_preset_base`：解析后的 base preset（去掉 `@...` 覆盖部分）
- `monitor_ph_params`：最终 PH 参数 JSON（按 signal 分组）
- 便捷列：`ph_error_*` / `ph_divergence_*` / `ph_entropy_*`

## 严重度校准（SeverityCalibrator）

- `soft_drift/severity.py` 在训练循环中实时维护误差/散度/熵的 EMA 基线：
  - `baseline_error`, `baseline_div`, `baseline_entropy` 通过 `ema_momentum` 慢速更新；
  - 当 `drift_flag = 1` 时，计算三种“危险方向”的正向增量：
    - `x_error_pos = max(0, error - baseline_error)`（学生错误率的上升）；
    - `x_div_pos = max(0, divergence - baseline_div)`（师生分歧的扩大）；
    - `x_entropy_pos` 支持 `entropy_mode`：
      - `overconfident`：`max(0, baseline_entropy - entropy)`（保持旧行为：教师更自信）；
      - `uncertain`：`max(0, entropy - baseline_entropy)`（教师更不确定）；
      - `abs`：`abs(entropy - baseline_entropy)`（双向偏离）。
  - 对 `x_*` 在线估计 mean/std，变换为 `z_*` 后按 `(0.6, 0.3, 0.1)` 加权得到 `drift_severity_raw`；
  - 将 `drift_severity_raw` 映射到 `[severity_low, severity_high]`（默认 `[0, 2]`）区间，再压缩到 `[0, 1]` 记作 `drift_severity`，供调度器使用。
- 日志字段：
  - `monitor_severity`：`DriftMonitor` 给出的单步增量（兼容旧版 `drift_severity` 定义）；
  - `drift_severity_raw`：三信号融合后的原始严重度；
  - `drift_severity`：归一化严重度，缺省仅在 `drift_flag=1` 时为正。
  - `severity_carry`：Severity-Aware v2 的“持续严重度”（默认关闭；见训练循环文档）。

## 超参调度（Scheduler）

- `scheduler/hparam_scheduler.py` 维护 `SchedulerState` 与 `HParams`：
  - Regime 划分：
    - **stable**：无漂移或长期稳定 → 更高 `alpha`，更小 `lr`，较大 `lambda_u`。
    - **mild_drift**：检测器报警且严重度不大 → 略降 `alpha`、略升 `lr`、适度降低 `lambda_u`、适度降低 `tau`。
    - **severe_drift**：严重漂移 → 显著降低 `alpha`、提升 `lr`、减小 `lambda_u`、下调 `tau`。
  - `update_hparams` 输出新的超参并返回当前 regime，训练循环据此写入日志与调节优化器。
- `update_hparams_with_severity`：在基础调度的结果上，再按照 `severity_norm` 缩放（v1 默认使用 `drift_severity`，v2 可用 `severity_carry` 替代；由训练循环决定）：
    - `alpha *= (1 - alpha_scale * s)`（默认 `alpha_scale=0.3`），严重漂移时更快“忘掉”旧教师；
    - `lambda_u *= (1 - lambda_u_scale * s)`（默认 `0.7`），抑制伪标签误差传播；
    - `tau += tau_delta * s`（默认 `0.15`），降低伪标签置信门槛；
    - `lr *= (1 + lr_scale * s)`（默认 `0.5`），加速恢复；
    - 缩放系数由 `SeveritySchedulerConfig` 管理，可在不同实验下调参；
    - 额外的 `severity_scale`（默认为 `1.0`）提供整体验证强度的“主旋钮”，训练脚本可通过 `--severity_scheduler_scale` 调整：
        - `1.0` → 维持当前强度；
        - `0.0` → 完全退化为 baseline，仅保留漂移触发但不按严重度缩放；
        - `>1.0` → 放大严重度响应，使 `alpha/lambda_u` 更快下降，`lr/tau` 更快上升。

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
   - 根据预设选择监控 `error_rate` / `divergence` 等；
   - 如果某条检测器触发，返回 `drift_flag=True` 与当前批次的 `monitor_severity`。
5. `SeverityCalibrator` 更新基线，并在 `drift_flag=1` 时输出 `drift_severity_raw`/`drift_severity`。
6. 调度：
   - baseline：`update_hparams(scheduler_state, current_hparams, drift_flag, monitor_severity)`；
   - severity-aware：`update_hparams_with_severity(..., severity_norm=drift_severity)`，在检测到漂移时按严重度进一步缩放 `(alpha, lr, lambda_u, tau)`。
7. 日志中同时记录：
   - 漂移信号：`student_error_rate`, `teacher_entropy`, `divergence_js`；
   - 检测输出：`drift_flag`, `monitor_severity`, `drift_severity_raw`, `drift_severity`, `regime`；
   - 调度结果：`alpha`, `lr`, `lambda_u`, `tau`。

这使得后续的 offline/online 评估脚本可以直接基于日志中的 `sample_idx`、`drift_flag` 与 meta 中的真值位置计算 MDR/MTD/MTFA/MTR，并可用 `monitor_preset` 快速切换不同检测配置。

## TODO

- 根据实际评估结果调整阈值（例如 severity 划分），并支持按指标准确率动态调节。
- 引入多检测器融合策略或权重，减少单一检测器误报影响。
- 将未来的 Lukats 指标反馈（如 MTD/MDR）纳入长周期超参调度策略。
