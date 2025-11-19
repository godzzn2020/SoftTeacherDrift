# Stage1 实验设置

## 目标

- 在少量合成流（SEA、Hyperplane）与真实流（Electricity、Airlines、Insects）上验证 Teacher–Student 框架的端到端流程。
- 统一生成日志、漂移检测输出与评估文件，为后续阶段（更复杂的数据/方法）打基础。

## 实现

- `experiments/first_stage_experiments.py`：
  - 定义 `ExperimentConfig`（数据、模型、训练超参、日志路径等）。
  - 提供 `run_experiment(config, device)`，内部构建 stream、模型、监控器、scheduler，并调用 `run_training_loop`。
  - CLI 支持批量运行 `--datasets`、`--models` 参数，自动保存日志到 `logs/{dataset}/...`.
- 默认数据集（可在 `_default_experiment_configs` 中查看）：
  - `sea_abrupt4`, `hyperplane_slow`, `electricity`, `airlines`, `insects_abrupt_balanced`。
  - 每个配置包含 `n_steps`, `batch_size`, `labeled_ratio`, `initial_alpha`, `initial_lr`, `lambda_u`, `tau` 等。

## 示例命令

```bash
# 单个数据集 + 单模型
python experiments/first_stage_experiments.py \
  --device cuda \
  --seed 42 \
  --datasets sea_abrupt4 \
  --models ts_drift_adapt

# 多数据集 + 多模型
python experiments/first_stage_experiments.py \
  --device cuda \
  --seed 1 \
  --datasets sea_abrupt4,hyperplane_slow,electricity \
  --models baseline_student,ts_drift_adapt
```

## TODO

- 规划 Stage2（大规模真实流 + 更复杂模型）、Stage3（多指标自动评估）并在此文档中扩展描述。
- 引入配置文件（YAML/JSON）以便无需修改脚本即可定义新实验。
- 支持实验并行调度与失败重试。

