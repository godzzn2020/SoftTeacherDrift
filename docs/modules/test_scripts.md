# 测试脚本与常用命令

## 目的

- 汇总当前仓库中“可直接运行”的训练、评估、诊断脚本，描述 CLI 参数、默认值与示例命令。
- 约定：**新增或调整此类脚本时，必须同步更新本文件**，保证测试入口始终可查。

## 离线 / 评估脚本

### experiments/offline_detector_sweep.py

- 作用：在已有训练日志和 meta.json 上离线网格搜索 detector 组合，输出 MDR/MTD/MTFA/MTR、CSV 及 per-dataset Markdown。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--datasets` | 逗号分隔数据集列表 | `sea_abrupt4,sine_abrupt4,stagger_abrupt3,INSECTS_abrupt_balanced` |
| `--model_variant` | 使用哪种模型日志 | `baseline_student` |
| `--seeds` | 需要复现的随机种子 | `1 2 3` |
| `--logs_root` / `--synth_root` / `--insects_meta` | 数据输入路径 | `logs` / `data/synthetic` / `datasets/real/INSECTS_abrupt_balanced.json` |
| `--out_csv` / `--out_md_dir` | 输出文件夹 | `results/offline_detector_grid.csv` / `results/offline_md` |
| `--top_k` | Markdown 中每个数据集保留的配置数 | `10` |
| `--debug_sanity` | 仅运行 ADWIN/PH 演示序列，确认是否能触发漂移 | `False` |

- 示例：

```bash
python experiments/offline_detector_sweep.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3,INSECTS_abrupt_balanced \
  --model_variant baseline_student \
  --seeds 1 2 3
```

### experiments/analyze_abrupt_results.py

- 作用：汇总 Stage-1 突变漂移实验日志，重算指标、绘制 Accuracy/Drift timeline。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--datasets` | 待分析的数据集（必填，可多选） | — |
| `--models` | 需要对比的模型（必填） | — |
| `--seeds` | 随机种子列表（必填） | — |
| `--logs_root` / `--synth_root` / `--results_root` / `--fig_root` | 输入输出路径 | `logs` / `data/synthetic` / `results` / `figures/abrupt` |
| `--insects_meta` | INSECTS meta 路径 | `datasets/real/INSECTS_abrupt_balanced.json` |

- 示例：

```bash
python experiments/analyze_abrupt_results.py \
  --datasets sea_abrupt4 sine_abrupt4 stagger_abrupt3 INSECTS_abrupt_balanced \
  --models baseline_student mean_teacher ts_drift_adapt \
  --seeds 1 2 3
```

## 训练 / 回归脚本

### run_experiment.py

- 作用：单次 Teacher-Student 训练入口，可通过 `--monitor_preset` 选择 DriftMonitor 预设。
- 核心参数（节选）：

| 参数 | 说明 |
| --- | --- |
| `--dataset_type`, `--dataset_name` | 必填，指定数据来源（sea/hyperplane/...）及变体 |
| `--model_variant` | baseline_student / mean_teacher / ts_drift_adapt 等 |
| `--batch_size`, `--labeled_ratio`, `--n_steps`, `--hidden_dims`, `--dropout` | 训练配置 |
| `--initial_alpha`, `--initial_lr`, `--lambda_u`, `--tau` | Teacher-Student 超参 |
| `--monitor_preset` | `none` / `error_ph_meta` / `divergence_ph_meta` / `error_divergence_ph_meta` |
| `--log_path` | CSV 输出路径 |
| `--device`, `--seed` | 运行设备与随机种子 |

- 示例：

```bash
python run_experiment.py \
  --dataset_type sea \
  --dataset_name sea_abrupt4 \
  --model_variant ts_drift_adapt \
  --monitor_preset error_divergence_ph_meta \
  --log_path logs/sea_abrupt4/test.csv
```

### experiments/first_stage_experiments.py

- 作用：Stage-1 批量训练脚本，自动遍历默认数据集/模型，支持统一 `--monitor_preset`。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--device` | `cpu` / `cuda` | 自动检测 |
| `--seed` | 随机种子 | `42` |
| `--datasets` | 逗号分隔列表或 `all` | `sea_abrupt4,hyperplane_slow,electricity,airlines,insects_abrupt_balanced` |
| `--models` | 逗号分隔模型或 `all` | `ts_drift_adapt` |
| `--monitor_preset` | 同 run_experiment | `none` |

- 示例：

```bash
python experiments/first_stage_experiments.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3,INSECTS_abrupt_balanced \
  --models baseline_student,mean_teacher,ts_drift_adapt \
  --monitor_preset error_divergence_ph_meta
```

### experiments/parallel_stage1_launcher.py

- 作用：在多张 GPU（或 CPU）上并行调度 Stage-1 组合（数据集 × 模型 × 种子），内部调用 `run_experiment.py` 并按照 `CUDA_VISIBLE_DEVICES` 轮询。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--datasets` / `--models` | 同 Stage-1，支持 `all` | `sea_abrupt4,...` / `baseline_student,...` |
| `--seeds` | 可一次指定多个随机种子 | `1` |
| `--monitor_preset` | 传递给每个 run_experiment | `error_divergence_ph_meta` |
| `--gpus` | 逗号分隔 GPU 列表，例如 `0,1`；填 `none` 时表示 CPU | `0,1` |
| `--max_jobs_per_gpu` | 单卡最大并发任务数 | `1` |
| `--device` | run_experiment 的 --device（如 `cuda`） | `cuda` |
| `--python_bin` | 子进程使用的 Python 解释器 | 当前解释器 |
| `--sleep_interval` | 轮询任务状态的间隔（秒） | `2.0` |

- 示例：将 4 个数据集 × 3 模型 × seed=1 平均分配到两张 GPU，并发数每卡 2：

```bash
python experiments/parallel_stage1_launcher.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3,INSECTS_abrupt_balanced \
  --models baseline_student,mean_teacher,ts_drift_adapt \
  --seeds 1 \
  --gpus 0,1 \
  --max_jobs_per_gpu 2 \
  --monitor_preset error_divergence_ph_meta
```

### experiments/summarize_online_results.py

- 作用：读取 `logs/{dataset}/{dataset}__{model}__seed{seed}.csv`，统计最终准确率/Kappa/漂移事件数量以及在线检测指标（MDR/MTD/MTFA/MTR），并输出
  `results/online_runs.csv`、`results/online_summary.csv`、`results/online_summary.md`。同时生成 Accuracy SVG (`figures/online_accuracy/`) 和检测时间线 SVG (`figures/online_detections/`)，与离线结果对照。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--datasets` / `--models` | 需要统计的组合（或 `all`） | `sea_abrupt4,...` / `baseline_student,...` |
| `--seeds` | 复数 seed | `1 2 3` |
| `--logs_root` | 日志根目录 | `logs` |
| `--out_runs_csv` / `--out_summary_csv` / `--out_md` | 输出表路径 | `results/online_runs.csv` 等 |
| `--fig_dir` / `--detection_fig_dir` | Accuracy / Detection SVG 输出目录 | `figures/online_accuracy` / `figures/online_detections` |
| `--top_k` | Markdown 每个数据集保留的配置数 | `10` |
| `--insects_meta` / `--synth_meta_root` | 真值漂移 meta 的路径 | `datasets/real/INSECTS_abrupt_balanced.json` / `data/synthetic` |

- 示例：

```bash
python experiments/summarize_online_results.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3,INSECTS_abrupt_balanced \
  --models baseline_student,mean_teacher,ts_drift_adapt \
  --seeds 1 \
  --fig_dir figures/online_accuracy
```

## 维护规则

- 若新增测试/评估脚本，或为现有脚本添加重要 CLI 参数，请务必同步更新本文件：
  - 描述脚本用途；
  - 给出主要命令行参数（含默认值/适用范围）；
  - 提供典型命令示例；
  - 标注依赖或注意事项（如需要先生成日志）。  
- 同时在相关模块文档中提示读者查看本文件，确保所有实验入口都有文档可查。
