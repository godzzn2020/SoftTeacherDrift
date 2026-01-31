# STD（SoftTeacherDrift）工程架构与创新点摘要（main / V15.3）

> 面向最终报告/论文读者：本文尽量做到“可定位”。每个关键机制都给出对应的文件路径与核心符号名（函数/类），便于 `rg` 追溯。

#### 0) TL;DR（10 行内）

- 系统做什么：在非平稳表格数据流上运行 Teacher-Student(EMA) 在线学习，同时做 drift detection，并通过 two-stage confirm（含 permutation-test confirm）抑制误报、保证可审计落盘。
- 关键创新点（含 V15.3）：
  - `two_stage`：candidate(OR)→confirm(window) 的两阶段触发，把“报警候选”和“最终确认”解耦（`drift/detectors.py:DriftMonitor.update`）。
  - `perm_test` confirm：在 confirm 阶段对 candidate 前后窗口做置换检验，用 p-value/effect 决定 confirm/reject，并输出可审计统计（`drift/detectors.py:_perm_test`, `training/loop.py:run_training_loop`）。
  - `vote_score` 作为 `perm_stat`：把多 detector 的 0/1 触发融合成有界票分，支持作为置换检验统计量（`drift/detectors.py:DriftMonitor.update`；实验组装见 `experiments/trackAL_perm_confirm_stability_v15p3.py`）。
  - `perm_side`：支持 one-sided / two-sided 的 effect 口径（signed vs abs），V15.3 用 `two_sided` 做方向不敏感复核（`drift/detectors.py:_perm_test`）。
  - TrackAL/TrackAM 可复核产物链：用 `.summary.json` sidecar 避免扫描大日志，生成稳定性表与诊断表并可追溯到具体 run/log（`experiments/trackAL_perm_confirm_stability_v15p3.py`, `experiments/trackAM_perm_diagnostics.py`）。

#### 1) 总体数据流（Dataflow）

1. 数据进入 pipeline（流式样本 → batch）
   - 入口：`run_experiment.py:main` → `experiments/first_stage_experiments.py:run_experiment`
   - 数据流构建：`data/streams.py:build_stream`（支持 `sea/sine/stagger/hyperplane/...`），再由 `data/streams.py:batch_stream` 按 `batch_size` 切 batch 并按 `labeled_ratio` 拆有/无标签子集。

2. 训练/推理（Teacher-Student + EMA）
   - 训练循环：`training/loop.py:run_training_loop`
   - 模型与损失：`models/teacher_student.py:TeacherStudentModel.compute_losses`、`models/teacher_student.py:TeacherStudentModel.update_teacher`

3. 计算 drift 信号（每 batch 一次）
   - 信号：`drift/signals.py:compute_signals` → `{error_rate, teacher_entropy, divergence}`

4. 漂移检测（trigger_mode 融合 + two_stage confirm）
   - 检测器封装：`drift/detectors.py:build_default_monitor`（构造 `DriftMonitor` + PH 参数解析）
   - 运行时更新：`drift/detectors.py:DriftMonitor.update(values, step, sample_idx=...)`
   - two_stage 关键输出：`candidate_flag`（候选）、`drift_flag`（confirmed）、`confirm_delay`、`candidate_count_total/confirmed_count_total`

5. （可选）drift severity 与超参调度
   - 严重度：`soft_drift/severity.py:SeverityCalibrator.compute_severity`
   - 调度：`scheduler/hparam_scheduler.py:update_hparams` / `scheduler/hparam_scheduler.py:update_hparams_with_severity`

6. 产物落盘（logs + sidecar summary）
   - 每步日志 CSV：`training/loop.py:run_training_loop`（`TrainingConfig.log_path`）
   - sidecar：`{log_path}.summary.json`（同函数写入；包含 candidate/confirmed sample_idx、perm pvalue 分位数等，供离线脚本直接读取）
   - run 目录规范：`soft_drift/utils/run_paths.py:ExperimentRun.prepare_dataset_run`（并写 legacy symlink）

7. TrackAL（稳定性/扫参）聚合表生成
   - 稳定性（V15.3）：`experiments/trackAL_perm_confirm_stability_v15p3.py:main`
   - sweep（V14/V15）：`experiments/trackAL_perm_confirm_sweep.py:main`
   - 产物：`TRACKAL_PERM_CONFIRM_STABILITY_*.csv` + `*_RAW.csv`（按 dataset+group 聚合/按 seed 明细）

8. TrackAM（perm_test 诊断表）
   - 入口：`experiments/trackAM_perm_diagnostics.py:main`
   - 逻辑：读取 TrackAL 的 `run_index_json`，只打开对应 run 的 `.summary.json`，统计 confirmed/candidate、p-value 分布等
   - 产物：`TRACKAM_PERM_DIAG_*.csv`

9. 报告/Top-K 表
   - 汇总入口：`scripts/summarize_next_stage_v15.py:main`（会复用 `scripts/summarize_next_stage_v14.py` 生成 run_index/metrics_table，并产出 Top-K hard-ok 表；表中包含 `perm_side` 列，若 TrackAL CSV 未提供则显示 `N/A`）

#### 2) 核心模块清单（Modules）

- `data/streams.py`
  - 作用：统一构建合成/真实数据流，并提供 batch+半监督拆分。
  - 关键接口：`build_stream`, `batch_stream`, `StreamInfo`
  - 输入输出：输入 dataset_type/name；输出可迭代的 `(x_dict, y)` 或 batch 形式 `(X_l, y_l, X_u, y_u_true)`。
  - 依赖关系：被 `experiments/first_stage_experiments.py:run_experiment` 调用。

- `models/teacher_student.py`
  - 作用：Teacher-Student(EMA) 半监督学习主体。
  - 关键接口：`TeacherStudentModel.compute_losses`, `TeacherStudentModel.update_teacher`, `LossOutputs`
  - 输入输出：输入张量与 `HParams(alpha, lr, lambda_u, tau)`；输出损失与细节（用于信号计算）。
  - 依赖关系：被 `training/loop.py:run_training_loop` 调用。

- `drift/signals.py`
  - 作用：把模型输出压缩成三类 drift signal。
  - 关键接口：`compute_signals`, `batch_error_rate`, `batch_teacher_entropy`, `batch_divergence`
  - 输入输出：输入 numpy probs/y；输出 `{error_rate, teacher_entropy, divergence}`。
  - 依赖关系：被 `training/loop.py:_collect_statistics` → `run_training_loop` 调用。

- `drift/detectors.py`
  - 作用：river PageHinkley 多信号检测 + 触发融合 + two_stage confirm + perm_test confirm。
  - 关键接口：
    - 构建：`build_default_monitor`
    - 核心更新：`DriftMonitor.update`
    - perm_test：`DriftMonitor._resolve_perm_test_cfg`, `DriftMonitor._perm_test`（含 `perm_side`）
  - 主要输入输出：
    - 输入：`values: Dict[str,float]`（signals）、`step`、`sample_idx`
    - 输出：`(drift_flag, monitor_severity)`
    - 额外状态（供日志/诊断读取）：`last_vote_score/last_vote_count/last_candidate_flag/last_confirm_delay/last_perm_pvalue/last_perm_effect/...`
  - 依赖关系：由 `training/loop.py:run_training_loop` 调用；构建由 `experiments/first_stage_experiments.py:run_experiment` 触发。
  - 必须覆盖点（定位）：
    - two_stage confirm：`drift/detectors.py:DriftMonitor.update`（candidate 注册、deadline、confirm_hit/confirm_delay、cooldown）
    - perm_test confirm + `perm_side`：`drift/detectors.py:_resolve_perm_test_cfg`, `drift/detectors.py:_perm_test`
    - `vote_score` 作为 `perm_stat`：`drift/detectors.py:DriftMonitor.update`（`perm_stat == "vote_score"` 分支）

- `training/loop.py`
  - 作用：在线训练主循环、写入日志 CSV 与 `.summary.json`（TrackAL/AM 的数据源）。
  - 关键接口：`run_training_loop`, `TrainingConfig`
  - 主要输入输出：
    - 输入：`batch_iter`、模型/优化器、`DriftMonitor`、`SchedulerState` 等
    - 输出：DataFrame；落盘：`log_path` + `log_path.with_suffix(".summary.json")`
  - 与其他模块关系：调用 `compute_signals`、`DriftMonitor.update`、`SeverityCalibrator.compute_severity`、`update_hparams*`。

- `experiments/trackAL_perm_confirm_stability_v15p3.py`
  - 作用：V15.3 stability 复核：固定候选组 + baseline，跨 seeds 输出 mean/std，并修复 `stagger_abrupt3` GT anchors。
  - 关键接口：`main`, `ensure_log`, `read_run_summary`
  - 输出：聚合 `TRACKAL_PERM_CONFIRM_STABILITY_*.csv` 与明细 `*_RAW.csv`
  - 依赖关系：内部会调用 `experiments/first_stage_experiments.py:run_experiment` 以补齐缺失 logs。

- `experiments/trackAM_perm_diagnostics.py`
  - 作用：TrackAM 可审计诊断：confirmed/candidate 比例 + p-value 分布（不扫大日志）。
  - 关键接口：`main`, `pick_winner`, `read_summary`
  - 输入输出：输入 TrackAL CSV（含 `run_index_json`）与目标 datasets/groups；输出 `TRACKAM_PERM_DIAG_*.csv`。

- `scripts/summarize_next_stage_v15.py`
  - 作用：把 TrackAL/TrackAM 的 CSV 进一步汇总为报告与 Top-K 表（并产出 run_index/metrics_table）。
  - 关键接口：`main`, `_write_main_report`
  - 依赖关系：会调用 `scripts/summarize_next_stage_v14.py`（subprocess 复用 V14 汇总逻辑）。

#### 3) V15.3 相关“创新点/设计点”详解（最重要）

##### A. `vote_score` 统计量

- 代码位置：
  - 票分计算：`drift/detectors.py:DriftMonitor.update`（`vote_score += weights.get(name, 1.0)`）
  - 作为 `perm_stat`：同函数内 `perm_stat == "vote_score"` 分支
  - 典型实验组装：`experiments/trackAL_perm_confirm_stability_v15p3.py`（`tw["__perm_stat"] = "vote_score"`）
- 设计动机：相比直接对 0/1 `drift_flag` 做检验，`vote_score` 保留“多信号共识程度”（加权票数），同时上界固定、可解释，便于跨数据集/多 seed 做稳定性比较。
- 行为解释（机制）：
  - 每个 signal detector（PH）给出 0/1 drift；`vote_score` 将这些 0/1 通过权重累加成标量。
  - 在 `weighted` 模式：`vote_score >= trigger_threshold` 直接触发 drift。
  - 在 `two_stage + perm_test`：`vote_score` 可作为 perm_test 的检验统计量（pre/post 的均值差）。

##### B. `perm_test` confirm（n_perm/alpha/pre/post/window 语义 + confirm/reject/not_tested）

- 代码位置：
  - 参数解析：`drift/detectors.py:DriftMonitor._resolve_perm_test_cfg`
  - 检验实现：`drift/detectors.py:DriftMonitor._perm_test` / `_perm_test_one_sided`
  - two_stage 触发：`drift/detectors.py:DriftMonitor.update`（candidate 注册、收集 pre/post、计算 `perm_ok`）
  - 落盘与统计：`training/loop.py:run_training_loop`（日志列 + `.summary.json` 的 `perm_pvalue_p50/p90/p99` 等）
- 设计动机：在 no-drift 场景下，PH 偶发误报会导致 `two_stage(weighted)` 仍可能确认；引入置换检验把“短期波动”与“统计显著的分布变化证据”区分开，从而系统性降低误报。
- 行为解释（机制）：
  - `pre_n/post_n`：以 `sample_idx` 为单位的窗口长度（`DriftMonitor.update` 内通过 `batch_n = current_pos - prev_pos` 把 batch 统计展开为样本级序列）。
  - `confirm_window`：以 `step`（batch 次数）为单位的候选有效期；超过 deadline 未确认则清空 pending。
  - `n_perm/alpha/min_effect`：在窗口齐备后，计算 p-value；当 `p <= alpha` 且 effect 满足最小阈值才 `perm_ok=True`。
  - confirm/reject/not_tested：
    - confirm：`perm_ok=True` → 本步 `confirm_hit=True` → 满足 rule/gate 后确认
    - reject：窗口齐备但 `perm_ok=False`（p 大或 effect 小）→ 继续等待/直至过期
    - not_tested：`len(pre_seq)<pre_n` 或 `len(post_seq)<post_n` 时不会计算置换检验，`perm_ok=False`（`.summary.json` 中 p-value 分位数会是 `NaN`，可作为“未发生有效检验”的信号）

##### C. `perm_side` 三种 side 的差异与 effect 口径（signed vs abs）

- 代码位置：
  - 参数：`drift/detectors.py:DriftMonitor._resolve_perm_test_cfg`（`__perm_side` / `perm_side`）
  - 语义与实现：`drift/detectors.py:DriftMonitor._perm_test`
  - 语义自检：`drift/detectors.py:_quickcheck_perm_test_sides`
- 三种 side：
  - `one_sided_pos`：只接受“post.mean - pre.mean > 0”的正向变化；若 `obs<=0` 直接返回 `p=1`，effect 输出为 **signed(obs)**。
  - `two_sided`：方向不敏感；以 `abs(post.mean-pre.mean)` 为 effect，p-value 按绝对差计算；effect 输出为 **abs(obs)**。
  - `abs_one_sided`：当前实现与 `two_sided` 等价（同样用 abs(effect)），保留为命名兼容/未来扩展入口。
- V15.3 设计点：在 stability 脚本中显式加入 `two_sided` 变体（`experiments/trackAL_perm_confirm_stability_v15p3.py`），用于验证“方向假设”是否影响 no-drift 误报与 drift 漏检的 trade-off。

##### D. two_stage confirm + cooldown/window 门控交互（为何 confirmed/candidate < 1）

- 代码位置：
  - `drift/detectors.py:DriftMonitor.update`（cooldown 清空 pending；deadline 过期清空；candidate_count_total/confirmed_count_total）
  - 日志与汇总：`training/loop.py:run_training_loop`（`candidate_flag`/`drift_flag`/计数/`confirm_delay`）
- 设计动机：candidate 设计为“敏感”，confirm 设计为“保守”；通过 `confirm_window` 与 `confirm_cooldown` 控制确认密度，避免密集误报造成不必要的重置/调参抖动。
- 行为解释（机制）：
  - `candidate_flag`：任一 detector drift 即候选（`vote_count>=1`）；但只有满足 confirm 规则才 `drift_flag=True`。
  - `confirm_window`：候选注册后必须在窗口内确认，否则过期丢弃（candidate++，confirmed 不变）。
  - `confirm_cooldown`：最近一次 confirmed 之后的冷却期内禁止新 confirm，并清空 pending，避免“过期后补确认”的晚检（confirmed 被抑制；candidate 的注册也会被冷却逻辑阻断）。
  - 对 `perm_test` 而言，early candidate 可能因 `pre_n` 不足而 not_tested，进而更容易窗口过期 → confirmed/candidate 进一步降低。

#### 4) “贡献归因”线索（用于后续消融）

> 入口主要在 `run_experiment.py` 的 CLI（`--trigger_mode/--trigger_threshold/--confirm_window/--trigger_weights`）以及 `drift/detectors.py` 对 `trigger_weights` 的约定键。

- gating-only（confirm 永远通过，但保留 cooldown/window）
  - 现成配置思路：使用 `trigger_mode=two_stage`，把 `trigger_threshold` 设到极低（如 0），使 confirm_hit 恒为真；冷却与窗口仍生效。
  - 参数位置：`run_experiment.py` 的 `--trigger_mode two_stage --trigger_threshold 0 --confirm_window ...`；冷却用 `--trigger_weights "__confirm_cooldown=200"`（`training/loop.py:run_training_loop` 注入到 monitor）。

- perm-only（保留 perm_test，但尽量关闭/放宽 cooldown/window）
  - 现成配置思路：`__confirm_rule=perm_test`，同时 `__confirm_cooldown=0` 且 `--confirm_window` 取大值（避免 candidate 过期）。
  - 参数位置：`drift/detectors.py:_resolve_perm_test_cfg`（`__perm_*`）、`drift/detectors.py:_resolve_confirm_cooldown`（`__confirm_cooldown`）、`run_experiment.py:--confirm_window`。

- baseline（不启用 perm_test）
  - 现成配置思路：不传 `__confirm_rule=perm_test`；使用 `weighted` 或 `two_stage+weighted`（默认 confirm 规则）。
  - 对照组实现：`experiments/trackAL_perm_confirm_stability_v15p3.py` 中的 `A_weighted_n5`。

#### 5) 风险点 / 易误用点（工程角度）

1. 单位混淆：`confirm_window` 用 step，而 `perm_pre_n/perm_post_n/confirm_cooldown` 用 sample_idx（`drift/detectors.py:DriftMonitor.update`）。
2. `trigger_weights` 既承载数值权重也承载字符串开关（如 `__confirm_rule=perm_test`）；命名写错不会报错但会静默退化（`run_experiment.py:parse_trigger_weights`, `drift/detectors.py:_resolve_confirm_rule_name`）。
3. perm_test 历史 ring buffer 上限 4096（`drift/detectors.py:DriftMonitor.__post_init__`）；若设 `pre_n` 过大可能永远 not_tested。
4. `vote_score` 分布离散、可能产生大量 ties；在 no-drift 下 p-value 可能集中在 1.0，需要结合 `min_effect` 或改用 `fused_score`（`drift/detectors.py:_perm_test` + 实验表）。
5. `fused_score` 依赖 river PageHinkley 内部字段 `_sum_increase/_sum_decrease`（`drift/detectors.py:DriftMonitor.update`），river 版本升级可能破坏该读取。
6. candidate/confirmed 计数语义：冷却期内 candidate 注册会被阻断（cooldown 分支清空 pending 且不注册），因此 `candidate_count_total` 不是“所有 candidate_flag 的次数”（`drift/detectors.py:DriftMonitor.update`）。
7. `stagger_abrupt3` 漂移 GT anchors 容易误用：默认等分 anchor 会引入“虚假漂移点”；V15.3 在 stability 脚本中显式修复（`experiments/trackAL_perm_confirm_stability_v15p3.py`）。
8. 多卡/并行输出冲突：TrackAL 支持 `--num_shards/--shard_idx`，必须隔离 `logs_root` 与 `out_csv/out_raw_csv`（`experiments/trackAL_perm_confirm_stability_v15p3.py` 头部注释；`scripts/merge_csv_shards_concat.py`）。

#### 6) 关键产物字段字典（面向写报告）

> 以下以当前仓库内已生成的 V15P3 文件头为准（也与脚本写入逻辑一致）。

- `TRACKAL_PERM_CONFIRM_STABILITY_*.csv`（聚合表；例：`scripts/TRACKAL_PERM_CONFIRM_STABILITY_V15P3.csv`）
  - `track/phase`：Track 编号与 quick/full 阶段（来自脚本）
  - `dataset/dataset_kind/base_dataset_name`：数据集别名、是否 nodrift、base 名
  - `group`：实验组名（通常编码 confirm_rule/perm 参数）
  - `monitor_preset/confirm_theta/confirm_window/confirm_cooldown/confirm_rule`：检测与确认配置
  - `perm_stat/delta_k/perm_alpha/perm_pre_n/perm_post_n/perm_n_perm/weights`：置换检验与融合权重
  - 指标：`acc_final_mean/std`, `miss_tol500_mean`, `conf_P90_mean`, `confirm_rate_per_10k_mean/std`, `MTFA_win_mean/std`
  - `run_index_json`：可追溯定位（dataset→runs[{seed,run_id,log_path}]）

- `TRACKAL_PERM_CONFIRM_STABILITY_*_RAW.csv`（逐 run 明细；例：`scripts/TRACKAL_PERM_CONFIRM_STABILITY_V15P3_RAW.csv`）
  - 关键定位：`seed/run_id/log_path/horizon`
  - 关键指标：`acc_final`, `miss_tol500`, `conf_P90`, `confirm_rate_per_10k`, `MTFA_win`
  - 关键配置镜像：`confirm_rule/perm_stat/perm_alpha/...`

- `TRACKAM_PERM_DIAG_*.csv`（诊断表；例：`scripts/TRACKAM_PERM_DIAG_V15P3.csv`）
  - confirmed/candidate：`candidate_count_mean/std`, `confirmed_count_mean/std`, `confirmed_over_candidate_mean`
  - p-value 统计：`perm_pvalue_p50_mean/p90_mean/p99_mean`, `perm_pvalue_le_alpha_ratio_mean`
  - 这些字段来自 `.summary.json` 的 `candidate_count_total/confirmed_count_total/perm_*`（`training/loop.py:run_training_loop` 写入）

- summarize 生成的 Top-K 表（`scripts/summarize_next_stage_v15.py`）
  - 列：`group, perm_stat, perm_alpha, perm_pre_n, perm_post_n, perm_side, pass_guardrail, sea_miss, sine_miss, sea_confP90, sine_confP90, no_drift_rate, no_drift_MTFA, drift_acc_final`
  - 说明：`no_drift_rate/MTFA/drift_acc_final` 为对 sea/sine 的聚合二次计算；`perm_side` 从 TrackAL CSV 读取（缺失则 `N/A`）。

#### 7) 附录：定位信息

- 参考的 docs：
  - `docs/PROJECT_OVERVIEW.md`
  - `docs/modules/drift_and_scheduler.md`
  - `docs/modules/training_loop.md`
  - `docs/modules/outputs_and_logs.md`
  - `docs/modules/test_scripts.md`

- 关键文件 Top10（按“读源码定位重要性”排序）：
  1. `drift/detectors.py`
  2. `training/loop.py`
  3. `run_experiment.py`
  4. `experiments/first_stage_experiments.py`
  5. `experiments/trackAL_perm_confirm_stability_v15p3.py`
  6. `experiments/trackAM_perm_diagnostics.py`
  7. `scripts/summarize_next_stage_v15.py`
  8. `data/streams.py`
  9. `soft_drift/utils/run_paths.py`
  10. `drift/signals.py`

