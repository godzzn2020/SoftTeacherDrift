# Stage1 实验设置

## 目标

- 在少量合成流（SEA、Hyperplane）与真实流（Electricity、Airlines、Insects）上验证 Teacher–Student 框架的端到端流程。
- 统一生成日志、漂移检测输出与评估文件，为后续阶段（更复杂的数据/方法）打基础。

## 实现

- `experiments/first_stage_experiments.py`：
  - 定义 `ExperimentConfig`（数据、模型、训练超参、日志路径等）。
  - 提供 `run_experiment(config, device)`，内部构建 stream、模型、监控器、scheduler，并调用 `run_training_loop`。
  - CLI 支持批量运行 `--datasets`、`--models` 参数，自动保存日志到 `logs/{dataset}/...`.
- `experiments/abrupt_stage1_experiments.py`：
  - 专门运行 “突变漂移” 套餐：默认包含 `sea_abrupt4`, `sine_abrupt4`, `stagger_abrupt3`, `INSECTS_abrupt_balanced`。
  - 内置三种模型变体：`baseline_student`, `mean_teacher`, `ts_drift_adapt`，并支持多 seed。
  - 自动调用 `data.streams.generate_default_abrupt_synth_datasets` 生成合成流 parquet，再调用 `run_experiment.py` 写日志。
- 默认配置（可在 `_default_experiment_configs` 或脚本常量中查看）：
  - 合成流：段长、漂移位置、种子等已固定；
  - 真实流：读取 `datasets/real/INSECTS_abrupt_balanced.csv` 并配套 meta（0-based positions）。

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

# 突变漂移批量实验（合成 + INSECTS）
python experiments/abrupt_stage1_experiments.py \
  --device cuda \
  --seeds 1 2 3
```

## TODO

- 规划 Stage2（大规模真实流 + 更复杂模型）、Stage3（多指标自动评估）并在此文档中扩展描述。
- 引入配置文件（YAML/JSON）以便无需修改脚本即可定义新实验。
- 支持实验并行调度与失败重试。
