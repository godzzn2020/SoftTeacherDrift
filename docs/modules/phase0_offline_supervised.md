# Phase0 离线监督基线

## 目标与背景

- 在 Electricity / NOAA / Airlines / INSECTS_abrupt_balanced 等真实数据集上构建离线监督 Tabular MLP，上界参考用。
- 保持与在线流实验一致的特征编码、日志目录结构与 run_id 机制，方便后续半监督阶段横向比较。
- 默认按时间顺序切分 60/20/20（train/val/test），同时保留 `labeled_ratio` 接口，为 Phase1/Phase2 半监督做准备。

## 关键组件

- `datasets/preprocessing.py`
  - `load_tabular_csv_dataset`：统一把真实 CSV 中的类别列编码为整数、数值列 cast 为 float32，标签统一为 int64。
  - 该函数同时被 `data/streams.py`（在线流）与 Phase0 离线数据划分复用，确保 FeatureVectorizer/LabelEncoder 的输入一致。

- `datasets/offline_real_datasets.py`
  - `OfflineDatasetConfig`：可配置 `name`、`labeled_ratio`、`val_ratio`、`test_ratio`、`max_samples`（快速 sanity check）等字段。
  - `load_offline_real_dataset`：按时间序列切分 train/val/test，并在 `labeled_ratio<1` 时返回 `X_train_unlabeled` 以保留未标注样本。
  - dataclass `OfflineDatasetSplits` 包含 `input_dim`、`num_classes`、特征名、类别集合等元信息，训练脚本直接使用。

- `models/tabular_mlp_baseline.py`
  - `TabularMLPConfig`：可调 `hidden_dim`、`num_layers`、`dropout`、`activation`、`use_batchnorm`。
  - `build_tabular_mlp`：构建多层全连接 MLP，Layer 结构为 `Linear -> (BatchNorm) -> Activation -> (Dropout)`，末层输出 logits。

- `experiments/phase0_offline_supervised.py`
  - CLI 支持 `--datasets`、`--seeds`、`--labeled_ratio`、`--hidden_dim`、`--num_layers`、`--dropout`、`--lr`、`--weight_decay`、`--batch_size`、`--max_epochs`、`--max_samples`、`--run_tag` 等。
  - 每个 `(dataset, seed)` 会创建 `results/phase0_offline_supervised/{dataset}/tabular_mlp_baseline/seed{seed}/{run_id}/`，写入：
    - `train.log`（逐 epoch 文本日志）
    - `metrics.csv`（epoch, train_loss, train_acc, val_loss, val_acc, lr）
    - `summary.json`（best_epoch、best_val_acc、test_acc、样本数量、超参）
  - 运行结束还会写 `results/phase0_offline_supervised/summary/{run_id}/run_level_metrics.csv` 与 `summary.md`。

- `evaluation/phase0_offline_summary.py`
  - 遍历 `results/phase0_offline_supervised/**/summary.json`，生成新的汇总 run（`results/phase0_offline_summary/summary/{run_id}/`）。
  - 输出 `run_level_metrics.csv`（逐 dataset/seed/run_id）与 `summary_by_dataset.csv`（按 dataset+model 统计 mean/std）。

## 当前实现状态

- 数据切分沿用 csv 原始顺序，不打乱，适合时间序列上游。
- `labeled_ratio` 默认 1.0，上层 CLI 亦可设置 0.1/0.05；未标注样本保留在 `X_train_unlabeled` 中供未来半监督使用。
- Tabular MLP 采用 AdamW + 可选 Cosine / Step LR，训练日志全部落到 per-run 子目录， run_id 格式 `YYYYMMDD-HHMMSS-xxx[_tag]`。
- 汇总脚本复用 run_paths 工具，可按 dataset 过滤、独立输出新的 summary run，方便多次训练结果比对。

## 使用方法

- 训练示例（可根据需要调参）：

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
      --lr_scheduler cosine \
  --run_tag phase0_mlp_full_supervised
  ```

  - 小规模 sanity check 可加 `--max_samples 5000 --max_epochs 2 --batch_size 256`，确保脚本流程正确后再跑全量。

- 批量多 GPU 脚本：

  ```bash
  GPU_IDS="0,1" \
  DATASETS="Airlines Electricity NOAA INSECTS_abrupt_balanced" \
  SEEDS="1,2,3" \
  RUN_TAG="phase0_mlp_full_supervised" \
  ./scripts/run_phase0_offline_supervised.sh
  ```

- 脚本会把数据集按顺序平均分配到 `GPU_IDS` 指定的显卡并行运行，其他训练参数通过环境变量复用 `experiments/phase0_offline_supervised.py` 的 CLI。
- 每个数据集会独立启动一个训练进程（Seeds 仍在该进程内串行运行），若数据集数量大于 GPU 数量则轮流分配。
- 若需要把 **所有 `(dataset, seed)` 组合一次性并行**（例如 4 个数据集 × 3 个 seed = 12 个作业），可以运行 `scripts/run_phase0_offline_supervised_full_parallel.sh`，它会基于数据集规模（Airlines 最大）把组合分配到 `GPU_IDS` 中当前负载最小的 GPU，seed 之间也完全并行；使用方式与上方脚本类似，只是每个组合都会启动一个独立 CLI 进程，请注意硬件资源。

- 汇总示例：

  ```bash
  python evaluation/phase0_offline_summary.py \
      --root_dir results/phase0_offline_supervised \
      --datasets Airlines,Electricity \
      --run_name phase0_report
  ```

  - 汇总后的 CSV/Markdown 将写入 `results/phase0_offline_summary/summary/{run_id}/`。

更多 CLI 细节与参数表见《docs/modules/test_scripts.md》。
