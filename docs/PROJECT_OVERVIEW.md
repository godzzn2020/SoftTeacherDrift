# PROJECT_OVERVIEW

## 项目目标

- 基于 **Teacher–Student + EMA** 的半监督框架，在非平稳表格数据流上完成：
  - **概念漂移检测**：尽量依赖漂移信号与无监督检测器，减少对标签的需求；
  - **漂移理解**：跟踪教师–学生统计、漂移强度/类型以及历史记录；
  - **漂移自适应**：通过在线超参调度缓解灾难性遗忘（例如动态调整 alpha、学习率、伪标签阈值等）。
- 支持多种数据来源：
  - **合成流**：SEA、Hyperplane，可生成含真值的 parquet + meta.json；
  - **真实流**：USPDS 仓库中的 Electricity、Airlines、INSECTS 等，按 CSV 或 river 自带数据读取。

## 代码结构（高层）

- `data/streams.py`：统一的数据入口，包含实时生成、保存/加载合成流、批次切分等。
- `models/teacher_student.py`：学生 MLP、教师 EMA 同构网络、监督 + 无监督损失。
- `drift/signals.py` 与 `drift/detectors.py`：计算误差/熵/散度等批次信号，封装 ADWIN、PageHinkley。
- `scheduler/hparam_scheduler.py`：基于漂移状态切换 stable / mild / severe regime，并更新 alpha、lr、lambda_u、tau；同时提供 severity-aware 调度接口。
- `soft_drift/severity.py`：在线维护误差/熵/散度的漂移基线，输出严重度 `drift_severity_raw` / `drift_severity`，供 scheduler 使用。
- `training/loop.py`：在线训练主循环，记录 batch 级日志；`run_experiment.py` & `experiments/first_stage_experiments.py` 提供 CLI 入口及批量实验脚本。
- `evaluation/`：（已启动）聚焦漂移指标计算与实验日志评估，将持续扩展。

## 阶段性离线实验

- **Phase0 — 全监督 Tabular MLP 基线**：`experiments/phase0_offline_supervised.py` + `artifacts/legacy/results/phase0_offline_supervised/`（历史产物归档目录），在固定的 train/val/test 划分上评估 MLP 上限，输出 run-level 与 summary。
- **Phase1 — 半监督 + EMA Teacher**：`experiments/phase1_offline_tabular_semi_ema.py` + `artifacts/legacy/results/phase1_offline_semisup/`（历史产物归档目录），复用 Phase0 划分，在低标签率（0.05/0.1）下验证 Teacher-Student 机制；教师验证精度用于 early stopping，并记录 teacher/student 的 val/test 指标。

## 数据与日志约定

- 实验产物（报告/表格/命令/归档日志）统一归档在 `artifacts/<version>/`，版本入口见 `artifacts/INDEX.md`；目录规范见 `docs/EXPERIMENT_ARTIFACT_RULES.md`。
- 合成数据存放在 `data/synthetic/{dataset_name}/`，文件命名为：
  - `{dataset_name}__seed{seed}_data.parquet`：包含时间步 `t`、`concept_id`、`is_drift`、`drift_id`、`y` 及特征列；
  - `{dataset_name}__seed{seed}_meta.json`：记录漂移真值、概念区间、生成器参数等。
- 训练日志历史产物已归档到 `artifacts/legacy/logs/`；核心字段包括：
  - `step`, `seen_samples`, `dataset_name`, `dataset_type`, `model_variant`, `seed`
  - 评估：`metric_accuracy`, `metric_kappa`
  - 漂移：`student_error_rate`, `teacher_entropy`, `divergence_js`, `drift_flag`, `monitor_severity`（检测器增量）、`drift_severity_raw`（严重度原值）、`drift_severity`（归一化 0–1 的严重度），`regime`
  - 调度：`alpha`, `lr`, `lambda_u`, `tau`
  - 训练：`supervised_loss`, `unsupervised_loss`, `timestamp`

## 漂移评估指标

- **合成流**：使用 MDR、MTD、MTFA、MTR，依赖 meta.json 中的真实漂移位置来匹配检测器报警。
- **真实流**：使用 lift-per-drift (lpd)，比较启用检测器与 baseline（无漂移检测）的 prequential accuracy 差异。

## 给代码助手的规则

1. 修改任意模块前先查看 `docs/INDEX.md`，再定位到对应的 `docs/modules/*.md` 获取背景与约定。
2. 每次对核心逻辑进行调整后，及时更新相关模块文档中的“当前实现”与“TODO”小节。
3. 有实质性改动（新增特性/接口/重要 bugfix）时，在 `changelog/CHANGELOG.md` 追加记录，包含日期与 1–3 条要点。
4. 文档风格统一使用简洁短段落与 bullet list，避免冗长大段文字，确保在 token 受限场景下可快速阅读。
5. 若新增子模块或指标，请同步更新 `docs/INDEX.md` 的索引，以便后续对话能够发现新文档。
6. **新增或调整测试/评估脚本时，必须同步更新 `docs/modules/test_scripts.md`**，记录 CLI 参数、默认值与典型命令，并确保脚本开头注入 `Path(__file__).resolve().parents[1]` 到 `sys.path`，以便直接运行脚本时能找到项目模块。
