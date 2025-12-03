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
- `experiments/offline_detector_sweep.py`：
  - 读取已有 `logs/{dataset}/{dataset}__{model}__seed{seed}.csv` 与对应 meta，离线重放多组 drift detector 组合。
  - 内置 `DetectorConfig` 网格（error/entropy/divergence × ADWIN/PageHinkley），输出 `results/offline_detector_grid.csv` 与 per-dataset Markdown Top-N。
  - CLI 支持 `--datasets`, `--seeds`, `--model_variant`, `--out_*`，以及 `--debug_sanity` 快速验证 detector 是否会触发，方便批量筛选稳定的漂移检测配置。
- `experiments/summarize_online_results.py`：
  - 汇总在线训练日志（`logs/`）并生成 `results/online_runs.csv`、`results/online_summary.csv`、`results/online_summary.md`。
  - 自带轻量级 SVG 绘图，将每个数据集的 Accuracy 曲线输出到 `figures/online_accuracy/`，并在 `figures/online_detections/` 绘制检测时间线；同时按日志中的 `drift_flag` 计算在线 MDR/MTD/MTFA/MTR，方便与离线 detector 结果对照。
- `evaluation/phaseA_signal_drift_analysis_synth.py`：
  - 针对合成流日志的信号层面对齐分析脚本，会读取 `logs/` 中的 `student_error_rate`、`teacher_entropy`、`divergence_js` 与 `data/synthetic/..._meta.json` 的真值漂移。
  - 输出每个 run 的信号 + 漂移叠加图，以及漂移前后窗口的统计（写入 `summary_pre_post_stats.csv`），用于验证 offline 选择的信号是否与真实漂移一致。
- `evaluation/phaseB_signal_drift_analysis_real.py`：
  - 针对真实流日志的数据探索脚本，会绘制三种信号与实际检测事件（`drift_flag==1`）的时间曲线，并输出每个 run 的检测次数、平均严重度等摘要。
  - 适用于 Electricity/NOAA/Airlines/INSECTS 等无真值漂移的场景，观察 detector 行为是否与业务直觉一致。
- `evaluation/phaseB_detection_ablation_synth.py`：
  - 离线重放 `DriftMonitor`，在合成流上比较 error/entropy/divergence 信号的 7 种 PageHinkley 组合的 MDR/MTD/MTFA/MTR，并输出 run 级与 dataset+preset 级别的统计结果。
  - 可用于分析信号组合的优劣，为线上选择 `monitor_preset` 提供依据。
- `experiments/stage1_multi_seed.py`：
  - 为多数据集 × 多模型 × 多随机种子循环调用 `run_experiment.py`，默认 datasets=`sea_abrupt4,sine_abrupt4,stagger_abrupt3`、models=`baseline_student,mean_teacher,ts_drift_adapt`、seeds=`1 2 3 4 5`，并统一透传 `--monitor_preset`（默认 `error_ph_meta`）。
  - 执行前会自动调用 `data.streams.generate_default_abrupt_synth_datasets`，确保所需的合成流 parquet/meta 在 `data/synthetic/` 下生成（覆盖所有传入 seed）。
  - 运行结束后调用 online summarizer 逻辑收集指标，输出 `results/stage1_multi_seed_raw.csv`（逐 seed）、`results/stage1_multi_seed_summary.csv`（按 dataset+model 的 mean/std），以及 `results/stage1_multi_seed_md/{dataset}_multi_seed_summary.md`（Markdown 表）。
  - 示例命令：
    ```bash
    python experiments/stage1_multi_seed.py \
      --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3 \
      --models baseline_student,mean_teacher,ts_drift_adapt \
      --seeds 1 2 3 4 5 \
      --monitor_preset error_ph_meta \
      --gpus 0,1 \
      --max_jobs_per_gpu 3
    ```
- `experiments/parallel_stage1_launcher.py`：
  - 基于 `_default_experiment_configs` 构建 (dataset × model × seed) 组合。
  - 通过指定 `--gpus` 与 `--max_jobs_per_gpu`，并行启动多个 `run_experiment.py` 子进程（内部设置 `CUDA_VISIBLE_DEVICES`），可将 12 个 Stage-1 任务平均分摊到多张卡上。
  - CLI 复用 Stage-1 常用参数（datasets/models/seeds/monitor_preset）并支持 `--python_bin`、`--sleep_interval` 等调度选项。
- `run_experiment.py` 与 `experiments/first_stage_experiments.py` 现均支持 `--monitor_preset`：
  - `none`：禁用漂移检测，保持旧版 baseline 行为；
  - `error_ph_meta`：使用 offline sweep 得到的学生误差 + PageHinkley(`alpha=0.15, delta=0.005, threshold=0.2, min_instances=25`);
  - `divergence_ph_meta`：使用 JS 散度 + PageHinkley(`alpha=0.1, delta=0.005, threshold=0.05, min_instances=30`);
  - `error_divergence_ph_meta`：同时启用上述两个 detector；
  - 这些 preset 直接映射到 `DriftMonitor` 内部的 detector 组合，可与 scheduler 联动，也可以设为 `none` 只记录原始信号。
- 默认配置（可在 `_default_experiment_configs` 或脚本常量中查看）：
  - 合成流：目前包含 `sea_abrupt4`, `sine_abrupt4`, `stagger_abrupt3`（均 n_steps=800, batch_size=64, labeled_ratio=0.1, 初始超参同 sea）；
  - 真实流：读取本地 `datasets/real/INSECTS_abrupt_balanced.csv`（`dataset_type=insects_real`），避免 river 在线下载 404，meta 信息仍来自配套 JSON（0-based positions）。

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
