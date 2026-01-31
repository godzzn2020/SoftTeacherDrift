# 测试脚本与常用命令

## 目的

- 汇总当前仓库中“可直接运行”的训练、评估、诊断脚本，描述 CLI 参数、默认值与示例命令。
- 约定：  
  - **新增或调整此类脚本时，必须同步更新本文件**，保证测试入口始终可查；  
  - **所有 CLI 脚本都要在文件开头注入 `Path(__file__).resolve().parents[1]` 到 `sys.path`**（示例如下），以便直接 `python xxx.py` 时能找到 `data/`、`models/` 等模块。
  - 日志与结果的统一目录规范参见《docs/modules/outputs_and_logs.md》，所有示例命令均会自动写入带 `run_id` 的子目录。
  - **多数据集实验默认策略**：当需要在多张 GPU 上并行跑多个数据集时，应按“**每个数据集独立进程、seeds 在进程内部串行**”的方式启动，所有数据集按顺序平均分配到可用 GPU（不足时轮询）。参考 `scripts/run_phase0_offline_supervised.sh`；后续新增的多数据集脚本若需并行，也要遵守这一模式。

```python
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
```

## 离线 / 评估脚本

### scripts/summarize_next_round.py

- 作用：针对“下一轮实验（Track A/B/C）”的指定 `run_name`（附加在 `run_id` 后的 token），自动在 `logs/` 与 `results/` 中定位对应 `run_id`，并汇总关键表格到 Markdown。
- 约束：脚本仅使用标准库 `csv/json/os/statistics`（不依赖 pandas），并复用 `evaluation/drift_metrics.py`（其本身也仅依赖标准库）在 INSECTS 上重算 MDR/MTD/MTFA/MTR。
- 关键参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--logs_root` | 日志根目录 | `logs` |
| `--results_root` | 结果根目录 | `results` |
| `--out_md` | 输出 Markdown 路径 | `NEXT_ROUND_TRACK_REPORT.md` |
| `--out_run_index_csv` | 输出本轮 run 索引 CSV | `NEXT_ROUND_RUN_INDEX.csv` |
| `--insects_meta` | INSECTS 真值 meta 路径 | `datasets/real/INSECTS_abrupt_balanced.json` |
| `--track_run_name_map` | 可选 JSON 覆盖默认 Track->(experiment,run_name_token) 映射 | 空 |

- 示例：

```bash
python scripts/summarize_next_round.py \
  --logs_root logs \
  --results_root results \
  --out_md NEXT_ROUND_TRACK_REPORT.md \
  --out_run_index_csv NEXT_ROUND_RUN_INDEX.csv
```

### scripts/summarize_next_round_v3.py

- 作用：汇总 Track D/E（Severity-Aware v2 + trigger_mode 融合策略）实验，输出：
  - `scripts/NEXT_ROUND_V3_REPORT.md`
  - `scripts/NEXT_ROUND_V3_RUN_INDEX.csv`
  - `scripts/NEXT_ROUND_V3_METRICS_TABLE.csv`
- 关键参数（节选）：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--recovery_window` | post-drift 恢复窗口 W（样本尺度） | `1000` |
| `--match_tolerance` | tolerance 口径的匹配容差（样本数） | `500` |
| `--min_separation` | tolerance 口径的去重间隔（样本数） | `200` |

- 示例：

```bash
python scripts/summarize_next_round_v3.py
```

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

### evaluation/phaseC_scheduler_ablation_synth.py

- 作用：对比 baseline (`ts_drift_adapt`) 与 severity-aware (`ts_drift_adapt_severity`) scheduler，在现有合成流日志 + meta 上重算 `mean_acc`、`final_acc` 以及漂移窗口内的 `drop_min_acc`，量化严重度调度是否降低性能谷底。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--logs_root` / `--synthetic_root` | 日志与 meta 根目录 | `logs` / `data/synthetic` |
| `--datasets` | 合成流列表（逗号分隔） | **必填** |
| `--model_variants` | 需要比较的模型 | `ts_drift_adapt,ts_drift_adapt_severity` |
| `--seeds` | 随机种子列表（逗号或空格分隔） | `1,2,3,4,5` |
| `--window` | `drop_min_acc` 的样本窗口 | `500` |
| `--final_window` | final accuracy 的尾部窗口（按行数/批次数） | `200` |
| `--output_dir` | 结果输出目录 | `results/phaseC_scheduler_ablation_synth` |

- 示例：

```bash
python evaluation/phaseC_scheduler_ablation_synth.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3 \
  --seeds 1,2,3,4,5 \
  --model_variants ts_drift_adapt,ts_drift_adapt_severity \
  --window 500 \
  --final_window 200
```

### evaluation/phaseC_scheduler_ablation_real.py

- 作用：在真实流日志上比较 baseline 与 severity-aware scheduler 的 `mean_acc` / `final_acc` / `drop_min_acc`，漂移位置来源于检测事件（去重后）。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--logs_root` | 日志根目录 | `logs` |
| `--datasets` | 真实数据集列表 | `Electricity,NOAA,INSECTS_abrupt_balanced,Airlines` |
| `--model_variants` | 模型列表 | `ts_drift_adapt,ts_drift_adapt_severity` |
| `--seeds` | 随机种子 | `1,2,3` |
| `--window` | 漂移后窗口（样本尺度） | `500` |
| `--final_window` | final accuracy 的尾部窗口（行数） | `200` |
| `--min_separation` | 合并检测事件的最小间隔 | `200` |
| `--output_dir` | 结果输出目录 | `results/phaseC_scheduler_ablation_real` |

- 示例：

```bash
python evaluation/phaseC_scheduler_ablation_real.py \
  --datasets Electricity,NOAA,INSECTS_abrupt_balanced,Airlines \
  --model_variants ts_drift_adapt,ts_drift_adapt_severity \
  --seeds 1,2,3 \
  --window 500 \
  --final_window 200 \
  --min_separation 200
```

### evaluation/phase0_offline_summary.py

- 作用：遍历 `results/phase0_offline_supervised/**/summary.json`，整理所有 Phase0 离线 MLP run 的 `best_val_acc` / `test_acc` 等指标，并按 dataset 聚合输出。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--root_dir` | Phase0 训练结果根目录 | `results/phase0_offline_supervised` |
| `--datasets` | 逗号分隔的 dataset 过滤列表 | 空（表示全部） |
| `--run_name` / `--run_id` | 汇总脚本 run_id 的附加名称 / 覆盖值 | 空 |
| `--results_root` / `--logs_root` | 汇总 run 的输出根目录 | `results` / `logs` |

- 输出：`results/phase0_offline_summary/summary/{run_id}/run_level_metrics.csv` 与 `summary_by_dataset.csv`。

- 示例：

```bash
python evaluation/phase0_offline_summary.py \
  --root_dir results/phase0_offline_supervised \
  --datasets Airlines,Electricity \
  --run_name phase0_report
```

### evaluation/phaseC_scheduler_ablation_real.py

- 作用：读取真实数据集日志（Electricity/NOAA/INSECTS_abrupt_balanced/Airlines 等），以 `drift_flag==1` 的检测事件为锚点，比较 baseline (`ts_drift_adapt`) 与 severity-aware (`ts_drift_adapt_severity`) 的 `mean_acc`、`final_acc`、`mean/max drop_min_acc`。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--logs_root` | 日志根目录 | `logs` |
| `--datasets` | 数据集目录列表（逗号分隔） | **必填** |
| `--model_variants` | 需要比较的模型列表 | `ts_drift_adapt,ts_drift_adapt_severity` |
| `--seeds` | 随机种子列表（逗号或空格分隔） | `1,2,3` |
| `--detection_window` | 检测事件前后窗口大小（样本数） | `500` |
| `--min_separation` | 合并检测事件的最小间隔（样本数） | `200` |
| `--final_window` | final accuracy 的尾部窗口（步数/批次数） | `200` |
| `--output_dir` | 结果输出目录 | `results/phaseC_scheduler_ablation_real` |

- 示例：

```bash
python evaluation/phaseC_scheduler_ablation_real.py \
  --datasets Electricity,NOAA,INSECTS_abrupt_balanced,Airlines \
  --seeds 1,2,3 \
  --model_variants ts_drift_adapt,ts_drift_adapt_severity \
  --detection_window 500 \
  --final_window 200
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
| `--trigger_mode` | `or` / `k_of_n` / `weighted` / `two_stage`（多 detector 融合触发） |
| `--trigger_k` | `k_of_n` 的 k（至少 k 个 detector 同时触发） |
| `--trigger_weights`, `--trigger_threshold` | `weighted` 的权重与阈值 |
| `--confirm_window` | `two_stage` 的 confirm window（候选后窗口内确认） |
| `--use_severity_v2` | 启用 Severity-Aware v2（carry+decay 持续影响调度） |
| `--severity_gate` | `none` / `confirmed_only`（仅高置信 drift 更新 carry/freeze） |
| `--entropy_mode` | `overconfident` / `uncertain` / `abs` |
| `--severity_decay`, `--freeze_baseline_steps` | v2 的 decay 与 baseline 冻结步数 |
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

- 作用：批量运行 Stage-1 多 seed 实验（默认覆盖 `sea_abrupt4,sine_abrupt4,stagger_abrupt3` × `baseline_student,mean_teacher,ts_drift_adapt,ts_drift_adapt_severity` × `seed=1..5`），并将结果聚合成 Raw/Summary CSV 与 per-dataset Markdown 表。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--datasets` / `--models` | 逗号分隔列表 | `sea_abrupt4,sine_abrupt4,stagger_abrupt3` / `baseline_student,mean_teacher,ts_drift_adapt,ts_drift_adapt_severity` |
| `--seeds` | 多个 seed | `1 2 3 4 5` |
| `--monitor_preset` | 统一传给 run_experiment | `error_ph_meta` |
| `--trigger_mode` / `--trigger_k` / `--trigger_weights` / `--trigger_threshold` / `--confirm_window` | 透传给 run_experiment（监控融合策略；two_stage 需 confirm_window） | `or` / `2` / 空 / `0.5` / `200` |
| `--use_severity_v2` / `--severity_gate` / `--entropy_mode` / `--severity_decay` / `--freeze_baseline_steps` | 透传给 run_experiment（Severity-Aware v2 + gating） | 关闭 / `none` / `overconfident` / `0.95` / `0` |
| `--device` | run_experiment 的 --device | `cuda` |
| `--gpus` / `--max_jobs_per_gpu` | 并行运行时的 GPU 设置 | `0,1` / `2`（填 `none` 可顺序运行） |
| `--logs_root` | 日志根目录 | `logs` |
| `--out_csv_raw` / `--out_csv_summary` / `--out_md_dir` | 输出路径 | `results/stage1_multi_seed_raw.csv` 等 |

- 示例：

```bash
python experiments/stage1_multi_seed.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3 \
  --models baseline_student,mean_teacher,ts_drift_adapt,ts_drift_adapt_severity \
  --seeds 1 2 3 4 5 \
  --monitor_preset error_ph_meta \
  --gpus 0,1 \
  --max_jobs_per_gpu 3

### scripts/run_c3_synth_severity.sh

- 作用：Phase C3 合成流 severity 调度实验的快捷脚本，串行提交 baseline（`ts_drift_adapt`）与 severity-aware（`ts_drift_adapt_severity`）两组多 seed 任务，默认配置 `datasets=sea_abrupt4,sine_abrupt4,stagger_abrupt3`、`seeds=1..5`、`gpus=0,1`、`max_jobs_per_gpu=2`、`monitor_preset=error_divergence_ph_meta`。
- 使用前可根据机器资源修改脚本顶部的变量，然后执行：

```bash
bash scripts/run_c3_synth_severity.sh
```

- 若想手动调用同样的命令，可运行：

```bash
python experiments/stage1_multi_seed.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3 \
  --models ts_drift_adapt \
  --seeds 1 2 3 4 5 \
  --monitor_preset error_divergence_ph_meta \
  --gpus 0,1 --max_jobs_per_gpu 2

python experiments/stage1_multi_seed.py \
  --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3 \
  --models ts_drift_adapt_severity \
  --seeds 1 2 3 4 5 \
  --monitor_preset error_divergence_ph_meta \
  --gpus 0,1 --max_jobs_per_gpu 2
```
```

### experiments/run_real_adaptive.py

- 作用：在指定的真实数据集（默认 `datasets/real/Electricity.csv`, `NOAA.csv`, `INSECTS_abrupt_balanced.csv`, `Airlines.csv`）上批量运行 `ts_drift_adapt` / `ts_drift_adapt_severity` 等模型，循环多个 seed，并把日志写入 `logs/{dataset}/{dataset}__{model_variant}__seed{seed}.csv`。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--datasets` | 逗号分隔列表（需在脚本内置的 `REAL_DATASETS` 中） | `Electricity,NOAA,INSECTS_abrupt_balanced,Airlines` |
| `--seeds` | 多个 seed | `1 2 3` |
| `--model_variants` | 需要运行的模型列表 | `ts_drift_adapt` |
| `--monitor_preset` | 传给 run_experiment 的漂移检测配置 | `error_divergence_ph_meta` |
| `--trigger_mode` / `--trigger_k` / `--trigger_weights` / `--trigger_threshold` / `--confirm_window` | 透传给 run_experiment（监控融合策略；two_stage 需 confirm_window） | `or` / `2` / 空 / `0.5` / `200` |
| `--use_severity_v2` / `--severity_gate` / `--entropy_mode` / `--severity_decay` / `--freeze_baseline_steps` | 透传给 run_experiment（Severity-Aware v2 + gating） | 关闭 / `none` / `overconfident` / `0.95` / `0` |
| `--device` | 训练设备 | `cuda` |
| `--logs_root` | 日志输出根目录 | `logs` |

- 脚本自动为每个数据集指定 `dataset_type`、`csv_path`、`batch_size/n_steps/alpha/lr/lambda_u/tau` 等配置，与 Stage-1 真实流默认设置一致。

- 示例：

```bash
python experiments/run_real_adaptive.py \
  --datasets Electricity,NOAA,INSECTS_abrupt_balanced,Airlines \
  --seeds 1 2 3 \
  --model_variants ts_drift_adapt,ts_drift_adapt_severity \
  --monitor_preset error_divergence_ph_meta \
  --device cuda
```

### experiments/trackF_weighted_threshold_sweep.py

- 作用：Track F 的 weighted 阈值扫描（θ sweep），在 `sea_abrupt4`/`stagger_abrupt3` 上输出逐 run 指标表，并在 `scripts/figures/` 生成 trade-off 曲线图。
- 输出：
  - `scripts/TRACKF_THRESHOLD_SWEEP.csv`
  - `scripts/figures/trackF_theta_mdr_mtfa.png`
  - `scripts/figures/trackF_theta_acc_final.png`
- 示例：

```bash
python experiments/trackF_weighted_threshold_sweep.py \
  --datasets sea_abrupt4,stagger_abrupt3 \
  --seeds 1 2 3 4 5 \
  --thetas 0.3 0.4 0.5 0.6 0.7 \
  --monitor_preset error_divergence_ph_meta
```

### experiments/trackG_two_stage_eval.py

- 作用：Track G 的三组对照：`or` vs `weighted` vs `two_stage(candidate OR → confirm weighted)`，并额外统计 candidate/confirmed 计数与确认延迟。
- 输出：`scripts/TRACKG_TWO_STAGE.csv`
- 示例：

```bash
python experiments/trackG_two_stage_eval.py \
  --datasets sea_abrupt4,stagger_abrupt3 \
  --seeds 1 2 3 4 5 \
  --theta_weighted 0.5 \
  --theta_confirm 0.5 \
  --confirm_window 200
```

### experiments/trackH_severity_gating_insects.py

- 作用：Track H（最小验证）：在 INSECTS（默认 seeds=1/3/5）上对比 v2 vs v2+gate（`confirmed_only`），检验 gating 是否缓解 v2 的负迁移。
- 输出：`scripts/TRACKH_SEVERITY_GATING.csv`
- 示例：

```bash
python experiments/trackH_severity_gating_insects.py \
  --seeds 1 3 5 \
  --theta_confirm 0.6
```

### scripts/summarize_next_stage.py

- 作用：读取 `scripts/TRACKF_THRESHOLD_SWEEP.csv` / `scripts/TRACKG_TWO_STAGE.csv`（以及可选的 Track H CSV），自动生成：`scripts/NEXT_STAGE_REPORT.md`。
- 示例：

```bash
python scripts/summarize_next_stage.py
```

### experiments/trackAL_perm_confirm_sweep.py

- 作用：Track AL（扫参）：在固定的 two_stage 设置下 sweep `perm_test` 的 `__perm_*` 网格，对比 baseline（weighted confirm）与 perm_test confirm，并输出按 `(dataset, group)` 聚合的表（含 `run_index_json` 精确定位到每个 run）。
- 入口：`experiments/trackAL_perm_confirm_sweep.py:main`
- 输出：默认写到 `scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv`（也可用 `--out_csv` 指定）。
- 关键点：perm_test/perm_stat/perm_side 等均通过 `trigger_weights` 透传（见 `drift/detectors.py:DriftMonitor._resolve_perm_test_cfg` 与 `run_experiment.py:parse_trigger_weights`）。

### experiments/trackAL_perm_confirm_stability_v15p3.py

- 作用：V15.3 稳定性复核（tiny ablation）：固定候选组 + baseline，跨 seeds 输出 mean/std，并修复 `stagger_abrupt3` 的 GT drift anchors。
- 入口：`experiments/trackAL_perm_confirm_stability_v15p3.py:main`
- 输出：
  - 聚合：`scripts/TRACKAL_PERM_CONFIRM_STABILITY_V15P3*.csv`
  - 明细：`scripts/TRACKAL_PERM_CONFIRM_STABILITY_V15P3*_RAW.csv`
- 注意：支持 `--num_shards/--shard_idx` 并行，必须隔离 `--out_csv/--out_raw_csv/--log_root_suffix`。

### experiments/trackAM_perm_diagnostics.py

- 作用：Track AM（可审计诊断）：基于 TrackAL 的 `run_index_json`，只读取对应 run 的 `.summary.json`，统计 confirmed/candidate 比例与 p-value 分布（避免扫描大日志）。
- 入口：`experiments/trackAM_perm_diagnostics.py:main`
- 输出：默认写到 `artifacts/v15p3/tables/TRACKAM_PERM_DIAG_V15P3.csv`（也可用 `--out_csv` 指定）。

### scripts/summarize_next_stage_v15.py

- 作用：V15 汇总入口：读取 TrackAL/TrackAM CSV，生成主报告/全量报告/metrics_table/run_index，并输出 Top-K hard-ok 表（列包含 `perm_side`；若 TrackAL CSV 未提供则显示 `N/A`）。
- 入口：`scripts/summarize_next_stage_v15.py:main`

### scripts/merge_csv_shards_concat.py

- 作用：将 TrackAL 的分片 CSV（同 header）直接拼接合并，常用于 two GPU 或 `--num_shards>1` 的产物回收。
- 示例：

```bash
python scripts/merge_csv_shards_concat.py \
  --out_csv scripts/TRACKAL_PERM_CONFIRM_STABILITY_V15P3.csv \
  --inputs scripts/TRACKAL_PERM_CONFIRM_STABILITY_V15P3_shard0.csv,scripts/TRACKAL_PERM_CONFIRM_STABILITY_V15P3_shard1.csv
```

### experiments/phase0_offline_supervised.py

- 作用：在真实数据集上离线训练 Tabular MLP 基线，并按 run_id 写入 `results/phase0_offline_supervised/{dataset}/tabular_mlp_baseline/seed{seed}/{run_id}/`。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--datasets` | 真实数据集列表 | `Airlines,Electricity,NOAA,INSECTS_abrupt_balanced` |
| `--seeds` | 逗号或空格分隔的随机种子 | `1` |
| `--labeled_ratio` | 训练集标注占比（1.0 / 0.1 / 0.05 等） | `1.0` |
| `--hidden_dim` / `--num_layers` / `--dropout` / `--activation` / `--no_batchnorm` | Tabular MLP 结构参数 | `512` / `4` / `0.2` / `relu` / 默认开启 BatchNorm |
| `--lr` / `--weight_decay` / `--batch_size` / `--max_epochs` | 训练超参 | `1e-3` / `1e-5` / `1024` / `50` |
| `--lr_scheduler` | `none` / `cosine` / `step` | `cosine` |
| `--max_samples` | 每个数据集仅取前 N 条样本（sanity check 用） | 空 |
| `--run_tag` / `--run_id` | 输出 run_id 的附加标识 / 覆盖值 | 空 |
| `--results_root` / `--logs_root` | 结果/日志根目录（run_paths 使用） | `results` / `logs` |

- 输出：每个组合生成 `train.log`、`metrics.csv`、`summary.json`，并在 `results/phase0_offline_supervised/summary/{run_id}/` 写入 `run_level_metrics.csv` + `summary.md`。脚本内部会按 `datasets × seeds` 顺序串行执行。
- 建议先用 `--max_samples 5000 --max_epochs 2` 做 sanity check，确认流程和目录结构正确后再跑全量。

- 示例：

```bash
python experiments/phase0_offline_supervised.py \
  --datasets Airlines,Electricity,NOAA,INSECTS_abrupt_balanced \
  --seeds 1,2,3 \
  --labeled_ratio 1.0 \
  --hidden_dim 512 \
  --num_layers 4 \
  --dropout 0.2 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --batch_size 1024 \
  --max_epochs 50 \
  --run_tag phase0_mlp_full_supervised
```

### scripts/run_phase0_offline_supervised.sh

- 作用：批量运行 Phase0 离线实验，自动把 DATASETS 平均分配到多张 GPU（默认两张），并在各自的 `CUDA_VISIBLE_DEVICES` 下并行启动 `experiments/phase0_offline_supervised.py`。
- 核心环境变量（可在执行脚本前 export 或直接前缀设置）：

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `DATASETS` | 以空格或逗号分隔的真实数据集列表 | `Airlines Electricity NOAA INSECTS_abrupt_balanced` |
| `GPU_IDS` | 逗号分隔的 GPU 编号列表 | `0,1` |
| `SEEDS` | 逗号分隔的随机种子 | `1,2,3` |
| `LABELED_RATIO`, `HIDDEN_DIM`, `NUM_LAYERS`, `DROPOUT`, `LR`, `WEIGHT_DECAY`, `BATCH_SIZE`, `MAX_EPOCHS`, `LR_SCHEDULER`, `RUN_TAG` | 透传给训练脚本 | 与 CLI 默认一致 |
| `EXTRA_ARGS` | 追加到 CLI 的自定义参数（例如 `--max_samples 5000`） | 空 |

- 脚本会对每个数据集单独启动一个 Python 进程，保持 `seeds` 串行但各数据集并行运行；若数据集数量多于 GPU 数量，会按顺序轮流分配。

- 示例（两张显卡并行）：

```bash
GPU_IDS="0,1" \
DATASETS="Airlines Electricity NOAA INSECTS_abrupt_balanced" \
SEEDS="1,2,3" \
RUN_TAG="phase0_mlp_full_supervised" \
./scripts/run_phase0_offline_supervised.sh
```

### scripts/run_phase0_offline_supervised_full_parallel.sh

- 作用：在显存足够的机器上，将 Phase0 的 **所有数据集 × seed** 组合一次性并行运行；根据数据集规模设置权重（Airlines 最大，NOAA 最小），再把每个 `(dataset, seed)` 组合分配到 `GPU_IDS` 中当前负载最低的 GPU。
- 环境变量：

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `DATASETS` / `SEEDS` | 与前述脚本一致 | `Airlines Electricity NOAA INSECTS_abrupt_balanced` / `1,2,3` |
| `GPU_IDS` | 逗号分隔 GPU 列表；每个组合都会启动一个独立进程 | `0,1` |
| 其它（`LABELED_RATIO`, `HIDDEN_DIM`, `NUM_LAYERS`, `DROPOUT`, `LR`, `WEIGHT_DECAY`, `BATCH_SIZE`, `MAX_EPOCHS`, `LR_SCHEDULER`, `RUN_TAG`, `EXTRA_ARGS`） | 与 `phase0_offline_supervised.py` 的 CLI 对应 | 与 CLI 默认一致 |

- 使用注意：
  - 同一数据集的不同 seed 也会被拆到不同 GPU 上并行运行，请确保硬件足够支撑；
  - 数据集权重默认 `Airlines=10`, `INSECTS=5`, `Electricity=3`, `NOAA=2`，可根据需要修改脚本。

- 示例（12 个组合全部并行）：

```bash
GPU_IDS="0,1" \
DATASETS="Airlines Electricity NOAA INSECTS_abrupt_balanced" \
SEEDS="1,2,3" \
RUN_TAG="phase0_mlp_full_supervised" \
./scripts/run_phase0_offline_supervised_full_parallel.sh
```

### scripts/multi_gpu_launcher.py

- 作用：统一在多张 GPU 上调度 Phase0 或 Stage1 任务。通过 Job 队列 + GPU cost 启发式，确保同一数据集的多个 seed 顺序执行，但不同数据集/任务可以并行。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--plan` | `phase0_mlp`（Airlines/Electricity/NOAA/INSECTS）或 `stage1_synth_default`（sea/sine/stagger × 2 模型） | 必填 |
| `--seeds` | 逗号或空格分隔的 seeds（传递给底层脚本，顺序运行） | `1,2,3` |
| `--gpu-ids` | 逗号分隔 GPU 列表（缺省使用 `CUDA_VISIBLE_DEVICES` 或 `0,1`） | `0,1` |
| `--max-jobs-per-gpu` | 单卡最大并发 job 数 | `2` |
| `--dry-run` | 只打印调度计划，不启动任务 | `False` |

- 调度策略：根据 `DATASET_GPU_COST`（Airlines/INSECTS=1.5，Electricity/NOAA=1.0，合成流=1.0）和 `GPU_CAPACITY=3.0` 做简单的 best-fit；一旦 job 启动就绑定 `CUDA_VISIBLE_DEVICES=<gpu>`，并在完成时输出耗时与退出码。

- 示例：

```bash
# Phase0，四个数据集 × seeds=1,2,3
python scripts/multi_gpu_launcher.py --plan phase0_mlp --seeds 1,2,3 --gpu-ids 0,1 --max-jobs-per-gpu 2

# Stage1 合成流，seeds=1..5
python scripts/multi_gpu_launcher.py --plan stage1_synth_default --seeds 1,2,3,4,5 --gpu-ids 0,1 --max-jobs-per-gpu 2

# 仅检查调度计划
python scripts/multi_gpu_launcher.py --plan phase0_mlp --seeds 1,2 --gpu-ids 0,1 --dry-run
```

### experiments/phase1_offline_tabular_semi_ema.py

- 作用：运行 Phase1 离线半监督（Teacher-Student EMA）实验，对固定的 train/val/test 划分在多个 labeled_ratio 下训练 `tabular_mlp_semi_ema`，并输出 run-level 与 dataset-level summary。
- 核心参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--datasets` | 真实数据集列表 | `Airlines,Electricity,NOAA,INSECTS_abrupt_balanced` |
| `--seeds` | 逗号或空格分隔的随机种子 | `1,2,3` |
| `--labeled_ratios` | 逗号分隔的 labeled_ratio | `0.05,0.1` |
| `--output_dir` | 输出目录 | `results/phase1_offline_semisup` |
| `--max_epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--optimizer` | 训练超参 | `50`, `1024`, `1e-3`, `1e-5`, `AdamW` |
| `--hidden_dim`, `--num_layers`, `--dropout`, `--activation`, `--no_batchnorm` | MLP 结构 | `512`, `4`, `0.2`, `relu`, 默认开启 BN |
| `--ema_momentum`, `--lambda_u`, `--rampup_epochs`, `--confidence_threshold` | 半监督/EMA 设置 | `0.99`, `1.0`, `5`, `0.8` |
| `--device`, `--num_workers` | 运行设备与 DataLoader worker 数 | `cuda`, `0` |

- 输出：
  - `run_level_metrics.csv`：每个 `(dataset, seed, labeled_ratio)` 的教师/学生验证/测试表现；
  - `summary_metrics_by_dataset_variant.csv`：按 dataset + variant + labeled_ratio 聚合 teacher accuracy 的 mean/std；
  - 每条记录包含 `run_id = YYYYMMDD-HHMMSS-xxx_phase1_mlp_semi_ema`。

- 示例：

```bash
python experiments/phase1_offline_tabular_semi_ema.py \
  --datasets Airlines,Electricity,NOAA,INSECTS_abrupt_balanced \
  --seeds 1,2,3 \
  --labeled_ratios 0.05,0.1 \
  --output_dir results/phase1_offline_semisup

python experiments/phase1_offline_tabular_semi_ema.py \
  --datasets INSECTS_abrupt_balanced \
  --seeds 1,2,3 \
  --labeled_ratios 0.05 \
  --max_epochs 30 \
  --ema_momentum 0.995 \
  --output_dir results/phase1_offline_semisup
```

### scripts/run_c3_synth_severity.sh

- 作用：Phase C3 合成流 severity 调度实验的多卡批处理脚本，顺序调用 `experiments/stage1_multi_seed.py` 运行 baseline (`ts_drift_adapt`) 与 severity-aware (`ts_drift_adapt_severity`)。
- 关键变量：

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `DATASETS` | 逗号分隔数据集列表 | `sea_abrupt4,sine_abrupt4,stagger_abrupt3` |
| `SEEDS_BASELINE` / `SEEDS_SEVERITY` | baseline/严重度版本的 seed 列表 | `1 2 3 4 5` |
| `GPUS` | 多卡列表（传入 `--gpus`） | `0,1` |
| `MAX_JOBS` | 单卡并行任务数量 | `2` |
| `MONITOR` | 传给 CLI 的 `--monitor_preset` | `error_divergence_ph_meta` |

- 使用方式：根据实际 GPU/seed 修改脚本内变量后执行 `bash scripts/run_c3_synth_severity.sh`。脚本会依次运行 baseline 与 severity 版本，日志写入 `logs/{dataset}/...`.

## 维护规则

- 若新增测试/评估脚本，或为现有脚本添加重要 CLI 参数，请务必同步更新本文件：
  - 描述脚本用途；
  - 给出主要命令行参数（含默认值/适用范围）；
  - 提供典型命令示例；
  - 标注依赖或注意事项（如需要先生成日志）。  
- 同时在相关模块文档中提示读者查看本文件，确保所有实验入口都有文档可查。
