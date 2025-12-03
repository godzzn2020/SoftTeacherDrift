# 测试脚本与常用命令

## 目的

- 汇总当前仓库中“可直接运行”的训练、评估、诊断脚本，描述 CLI 参数、默认值与示例命令。
- 约定：  
  - **新增或调整此类脚本时，必须同步更新本文件**，保证测试入口始终可查；  
  - **所有 CLI 脚本都要在文件开头注入 `Path(__file__).resolve().parents[1]` 到 `sys.path`**（示例如下），以便直接 `python xxx.py` 时能找到 `data/`、`models/` 等模块。

```python
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
```

## 离线 / 评估脚本

### evaluation/phaseA_signal_drift_analysis_synth.py

- 作用：针对合成数据流，读取 `logs/{dataset}/{dataset}__{model}__seed{seed}.csv` 和 `data/synthetic/{dataset}/{dataset}__seed{seed}_meta.json`，绘制三种漂移信号（student_error_rate / teacher_entropy / divergence_js）与真实漂移的对齐图，并在漂移前后窗口统计信号均值差异，输出汇总 CSV，帮助分析 offline 搜索到的信号是否与真值一致。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--logs_root` | 日志根目录 | `logs` |
| `--synthetic_root` | 合成数据 meta 根目录 | `data/synthetic` |
| `--datasets` | 需要分析的合成数据集（逗号分隔） | **必填** |
| `--model_variant_pattern` | 可选字符串过滤器，只分析包含该子串的模型 | 空 |
| `--output_dir` | 图像/CSV 输出目录 | `results/phaseA_synth_analysis` |
| `--window` | 漂移前后窗口大小（以 step 计） | `50` |

- 输出：`{output_dir}/plots/{dataset}__{model}__seed{seed}.png` 以及 `summary_pre_post_stats.csv`（包含每个漂移点的 pre/post 均值与差值）。

- 示例命令：

```bash
python evaluation/phaseA_signal_drift_analysis_synth.py \
  --datasets sea_abrupt4,sine_abrupt4 \
  --model_variant_pattern ts_drift_adapt \
  --window 100
```

### evaluation/phaseB_signal_drift_analysis_real.py

- 作用：针对真实数据集的日志，绘制三种漂移信号与检测事件（`drift_flag`）的时间曲线，并输出每个 run 的检测统计，帮助在没有真值漂移的情况下检视 detector 行为。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--logs_root` | 日志根目录 | `logs` |
| `--datasets` | 逗号分隔的数据集名称（例如 Electricity,NOAA） | **必填** |
| `--model_variant_pattern` | 只分析满足该子串的模型（默认 `ts_drift_adapt`） | `ts_drift_adapt` |
| `--output_dir` | 图像与 summary 输出目录 | `results/phaseB_real_analysis` |

- 输出：`plots/{dataset}__{model}__seed{seed}.png`（信号 + detection 纵线）与 `summary_detection_stats.csv`。

- 示例：

```bash
python evaluation/phaseB_signal_drift_analysis_real.py \
  --datasets Electricity,NOAA,INSECTS_abrupt_balanced,Airlines \
  --model_variant_pattern ts_drift_adapt
```

### evaluation/phaseB_detection_ablation_synth.py

- 作用：针对合成流日志，离线重放 `DriftMonitor`，对 7 种 PageHinkley 信号组合（error / entropy / divergence 的 1/2/3 组合）进行检测指标消融，输出 run-level 和 dataset+preset 汇总的 MDR/MTD/MTFA/MTR。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--logs_root` / `--synthetic_root` | 日志与 meta 根目录 | `logs` / `data/synthetic` |
| `--datasets` | 合成数据集列表（逗号分隔） | **必填** |
| `--model_variant_pattern` | 只评估名称包含该子串的 model_variant | 空 |
| `--presets` | 逗号分隔的 preset 名列表 | 7 种组合全覆盖 |
| `--match_tolerance` | 真漂移与检测匹配的容差（样本数） | `500` |
| `--min_separation` | 去重检测事件的最小间隔（样本数） | `200` |
| `--output_dir` | 结果输出目录 | `results/phaseB_ablation_synth` |

- 示例：

```bash
python evaluation/phaseB_detection_ablation_synth.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3 \
  --model_variant_pattern ts_drift_adapt \
  --presets error_only_ph_meta,entropy_only_ph_meta,divergence_only_ph_meta,error_entropy_ph_meta,error_divergence_ph_meta,entropy_divergence_ph_meta,all_signals_ph_meta \
  --match_tolerance 500 \
  --min_separation 200
```

### evaluation/phaseC_severity_analysis_synth.py

- 作用：在合成流上针对每个真实漂移点计算三种信号（error / teacher_entropy / divergence_js）以及 accuracy 的漂移前后变化，输出 per-drift 统计与 dataset 级别的信号变化 vs 性能掉幅相关性，用于定义“漂移严重度”指标。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--logs_root` / `--synthetic_root` | 日志与 meta 根目录 | `logs` / `data/synthetic` |
| `--datasets` | 合成数据集列表（逗号分隔） | **必填** |
| `--model_variant_pattern` | 只分析名称包含该子串的 model_variant | 空 |
| `--window` | 漂移前后窗口大小（样本数） | `500` |
| `--output_dir` | 结果输出目录 | `results/phaseC_severity_analysis` |

- 示例：

```bash
python evaluation/phaseC_severity_analysis_synth.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3 \
  --model_variant_pattern ts_drift_adapt \
  --window 500
```

### evaluation/phaseC_severity_score_fit.py

- 作用：基于 Phase C1 生成的 `per_drift_stats.csv`，构造三信号联合的严重度得分（手工权重版 + 线性回归版），并评估它们与 `drop_min_acc` 的相关性，为后续调度器使用统一 severity score 做准备。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--per_drift_path` | Phase C1 输出的 per-drift 统计路径 | `results/phaseC_severity_analysis/per_drift_stats.csv` |
| `--output_dir` | 保存 severity 表与相关性汇总的目录 | `results/phaseC_severity_score` |
| `--min_drop` | 拟合/标准化使用的最小 `drop_min_acc`（过滤极小漂移） | `0.0` |
| `--standardize / --no-standardize` | 是否对特征做 z-score（默认开启，可用 `--no-standardize` 关闭） | `True` |

- 示例：

```bash
python evaluation/phaseC_severity_score_fit.py \
  --per_drift_path results/phaseC_severity_analysis/per_drift_stats.csv \
  --output_dir results/phaseC_severity_score \
  --min_drop 0.0 \
  --standardize
```

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

### experiments/stage1_multi_seed.py

- 作用：批量运行 Stage-1 多 seed 实验（默认覆盖 `sea_abrupt4,sine_abrupt4,stagger_abrupt3` × `baseline_student,mean_teacher,ts_drift_adapt` × `seed=1..5`），并将结果聚合成 Raw/Summary CSV 与 per-dataset Markdown 表。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--datasets` / `--models` | 逗号分隔列表 | `sea_abrupt4,sine_abrupt4,stagger_abrupt3` / `baseline_student,mean_teacher,ts_drift_adapt` |
| `--seeds` | 多个 seed | `1 2 3 4 5` |
| `--monitor_preset` | 统一传给 run_experiment | `error_ph_meta` |
| `--device` | run_experiment 的 --device | `cuda` |
| `--gpus` / `--max_jobs_per_gpu` | 并行运行时的 GPU 设置 | `0,1` / `2`（填 `none` 可顺序运行） |
| `--logs_root` | 日志根目录 | `logs` |
| `--out_csv_raw` / `--out_csv_summary` / `--out_md_dir` | 输出路径 | `results/stage1_multi_seed_raw.csv` 等 |

- 示例：

```bash
python experiments/stage1_multi_seed.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3 \
  --models baseline_student,mean_teacher,ts_drift_adapt \
  --seeds 1 2 3 4 5 \
  --monitor_preset error_ph_meta \
  --gpus 0,1 \
  --max_jobs_per_gpu 3
```

### experiments/run_real_adaptive.py

- 作用：在指定的真实数据集（默认 `datasets/real/Electricity.csv`, `NOAA.csv`, `INSECTS_abrupt_balanced.csv`, `Airlines.csv`）上仅运行 `ts_drift_adapt` 模型，循环多个 seed，并把日志写入 `logs/{dataset}/{dataset}__ts_drift_adapt__seed{seed}.csv`。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--datasets` | 逗号分隔列表（需在脚本内置的 `REAL_DATASETS` 中） | `Electricity,NOAA,INSECTS_abrupt_balanced,Airlines` |
| `--seeds` | 多个 seed | `1 2 3` |
| `--monitor_preset` | 传给 run_experiment 的漂移检测配置 | `error_divergence_ph_meta` |
| `--device` | 训练设备 | `cuda` |
| `--logs_root` | 日志输出根目录 | `logs` |

- 脚本自动为每个数据集指定 `dataset_type`、`csv_path`、`batch_size/n_steps/alpha/lr/lambda_u/tau` 等配置，与 Stage-1 真实流默认设置一致。

- 示例：

```bash
python experiments/run_real_adaptive.py \
  --datasets Electricity,NOAA,INSECTS_abrupt_balanced,Airlines \
  --seeds 1 2 3 \
  --monitor_preset error_divergence_ph_meta \
  --device cuda
```

## 维护规则

- 若新增测试/评估脚本，或为现有脚本添加重要 CLI 参数，请务必同步更新本文件：
  - 描述脚本用途；
  - 给出主要命令行参数（含默认值/适用范围）；
  - 提供典型命令示例；
  - 标注依赖或注意事项（如需要先生成日志）。  
- 同时在相关模块文档中提示读者查看本文件，确保所有实验入口都有文档可查。
