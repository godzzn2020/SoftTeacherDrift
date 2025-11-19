# 数据流与合成生成

## 简介

`data/streams.py` 提供统一的数据入口，贯穿合成流、真实流、批次划分以及保存/加载工具。模块输出 `StreamInfo`（含特征数、类别数）以供训练循环配置模型。

## 支持的数据集类型

- `sea` / `hyperplane`：即时生成的 river 合成流，通过 `SEA_CONFIGS`、`HYPERPLANE_CONFIGS` 控制概念序列与漂移。
- `uspds_csv`：从本地 CSV 读取（默认最后一列为标签），使用 `river.stream.iter_pandas` 返回 `(x_dict, y)`。
- `insects_river`：使用 `river.datasets.Insects` 中的各类变体。
- `*_saved`：`sea_saved`、`hyperplane_saved`、`synth_saved` 走 `load_saved_synth_stream`，从 `data/synthetic/{dataset}/` 读取 parquet + meta。

## 合成流生成/保存

- `generate_and_save_synth_stream(dataset_type, dataset_name, n_samples, seed, out_root, **cfg)`：
  - 生成多段概念序列，写出：
    - `{dataset_name}__seed{seed}_data.parquet`（字段：`t`, `concept_id`, `is_drift`, `drift_id`, `y`, `f0...`）
    - `{dataset_name}__seed{seed}_meta.json`（漂移真值、concept segments、生成器参数等）
  - 支持 SEA（突变，按 concept_id 切换 threshold）、Hyperplane（按段调整 `mag_change` 等）。
- `load_saved_synth_stream` 读取上述文件并重新构建 `StreamInfo`，供 `build_stream` 复用。

## 批次切分

- `batch_stream(stream, batch_size, labeled_ratio, seed)`：
  - 将任意迭代器划分为 minibatch；
  - 随机抽取 `labeled_ratio` 作为有标签部分，其余作为无标签但保留真值，用于指标评估。

## TODO

- 支持渐变漂移脚本（目前仅实现突变概念切换）。
- 在 meta.json 中扩展更多统计（例如每段特征分布）以便后续评估。
- 提供更细粒度的真实数据预处理（缺失值、类别特征编码等）。

