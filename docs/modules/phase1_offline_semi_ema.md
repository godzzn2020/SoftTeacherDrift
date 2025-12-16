# Phase1 离线半监督（Tabular MLP + EMA Teacher）

## 目的

- 在与 Phase0 相同的离线划分（train/val/test）上验证半监督 Teacher-Student 机制是否稳定；
- 仅在训练集上抽取部分样本（`labeled_ratio ∈ {0.05, 0.1}`）作为监督信号，剩余样本通过教师伪标签训练；
- 通过离线实验确认模型结构、EMA、伪标签策略是否可行，再迁移到在线/漂移场景。

## 核心组件

- **模型结构**：学生与教师完全一致，均复用 `models/tabular_mlp_baseline.py` 的 `TabularMLP`（多层 FC + BatchNorm + Dropout）。
- **训练配置 (`training/tabular_semi_ema.py`)**：
  - `TabularSemiEMATrainingConfig` 控制 `max_epochs`, `batch_size`, `lr`, `weight_decay`, `ema_momentum`, `lambda_u`, `rampup_epochs`, `confidence_threshold`, `labeled_ratio`, `device`, `num_workers`。
  - 训练集抽样：对 `splits.X_train` 按 `labeled_ratio` 生成布尔 mask，labeled 子集计算监督 CE，剩余样本在 teacher 置信度达阈值时参与伪标签损失。
  - λ_u 线性 ramp-up：前 `rampup_epochs` 从 0 增长到 `lambda_u`，避免初期噪声伪标签。
  - EMA 更新：每个 step 后执行 `teacher = m * teacher + (1 - m) * student`。
  - 评估：每个 epoch 计算 student/teacher 的验证准确率，按 `teacher_val_acc` 选择 best epoch，最终载入 best state 在测试集分别评估 student/teacher。

- **实验脚本 (`experiments/phase1_offline_tabular_semi_ema.py`)**：
  - CLI 支持 `--datasets`、`--seeds`、`--labeled_ratios`、MLP/EMA 超参、`--output_dir`、`--device` 等；
  - 对每个 `(dataset, seed, labeled_ratio)` 生成 `run_id=YYYYMMDD-HHMMSS-xyz_phase1_mlp_semi_ema`，调用训练函数并记录指标；
  - `run_level_metrics.csv` 追加字段：
    - `dataset_name, model_variant=tabular_mlp_semi_ema, seed, run_id, labeled_ratio`
    - `train_samples, train_labeled_samples, val_samples, test_samples`
    - `best_epoch, best_val_acc_teacher, test_acc_teacher, best_val_acc_student, test_acc_student`
    - 训练与半监督超参（`max_epochs`, `batch_size`, `lr`, `weight_decay`, `hidden_dim`, `num_layers`, `dropout`, `activation`, `use_batchnorm`, `ema_momentum`, `lambda_u`, `rampup_epochs`, `confidence_threshold`）
  - `summary_metrics_by_dataset_variant.csv`：按 `(dataset_name, model_variant, labeled_ratio)` 聚合 `best_val_acc_teacher` 与 `test_acc_teacher` 的 mean/std，并附带 `runs` 统计数量。

## 输出目录

- 单 run log（可选）打印在控制台；
- `results/phase1_offline_semisup/run_level_metrics.csv`：所有 run 的明细，追加写入；
- `results/phase1_offline_semisup/summary_metrics_by_dataset_variant.csv`：运行结束后重新生成；
- run_id 只写在 CSV 中，不会再额外生成 per-run 子目录（以减轻 I/O）。

## 示例命令

```bash
# Phase1：四个数据集 + 3 seeds + 两种 labeled_ratio
python experiments/phase1_offline_tabular_semi_ema.py \
  --datasets Airlines,Electricity,NOAA,INSECTS_abrupt_balanced \
  --seeds 1,2,3 \
  --labeled_ratios 0.05,0.1 \
  --max_epochs 50 \
  --ema_momentum 0.99 \
  --lambda_u 1.0 \
  --confidence_threshold 0.8 \
  --output_dir results/phase1_offline_semisup

# 只跑 INSECTS，标签 5%，λ_u 更小
python experiments/phase1_offline_tabular_semi_ema.py \
  --datasets INSECTS_abrupt_balanced \
  --seeds 1,2,3 \
  --labeled_ratios 0.05 \
  --lambda_u 0.5 \
  --rampup_epochs 10 \
  --output_dir results/phase1_offline_semisup
```

## 当前状态 & TODO

- ✅ 复用 Phase0 划分，保证可比性；
- ✅ 支持 0.05/0.1 标签率、Teacher EMA、伪标签阈值、λ_u ramp-up；
- ✅ 自动聚合 run-level & summary CSV；
- ⏳ TODO：后续 Phase2/在线实验可复用同一训练配置，将该离线实现迁移到流式数据处理 pipeline。
