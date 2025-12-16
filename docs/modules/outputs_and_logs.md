# 运行输出与目录约定

自本次整理起，所有训练 / 评估脚本都会为**每一次调用**生成唯一 `run_id`，并把日志、CSV、Markdown 等结果写入独立的子目录，避免覆盖历史记录。核心规则如下：

1. **run_id 生成**
   - 默认格式：`YYYYMMDD-HHMMSS-xyz`，其中 `xyz` 为 3 位随机后缀。
   - `--run_name foo_bar` 会追加 `_foo_bar` 方便手工标记；也可使用 `--run_id xxx` 强制指定（谨慎，容易覆盖）。

2. **目录层级**
   - 训练日志文件（CSV）：

     ```
     logs/{experiment_name}/{dataset_name}/{model_variant}/seed{seed}/{run_id}/{dataset}__{model}__seed{seed}.csv
     ```

     例如：`logs/stage1_multi_seed/sea_abrupt4/ts_drift_adapt/seed1/20250205-103012-abc/sea_abrupt4__ts_drift_adapt__seed1.csv`

   - 与日志相关的中间文件（如 run metadata）放在同一 `logs/.../{run_id}/` 目录中。

   - 结果与评估输出：

     ```
     results/{experiment_name}/{dataset_name}/{model_variant}/seed{seed}/{run_id}/...
     ```

     批量/汇总类型脚本还会写入 `results/{experiment_name}/summary/{run_id}/...`，其中包含：
       - `run_level_metrics.csv`
       - `summary_metrics_by_dataset_variant.csv`
       - 可选的 Markdown 报告

3. **向前兼容**
   - 旧版日志依然保存在 `logs/{dataset}/{dataset}__{model}__seed{seed}.csv`，评估脚本会优先读取新路径，找不到时退回旧路径。
   - 运行参数 `--logs_root` / `--results_root` 依旧可用，只是现在根目录下会自动生成 `experiment_name/run_id` 的层级。

4. **典型命令示例**

   ```bash
   # Stage-1 合成流多 seed（自动 run_id），输出位于 results/stage1_multi_seed/summary/{run_id}/
   python experiments/stage1_multi_seed.py \
     --datasets sea_abrupt4,sine_abrupt4 \
     --models ts_drift_adapt \
     --seeds 1 2 \
     --run_name trial_a

   # 真实流运行（日志写到 logs/run_real_adaptive/.../{run_id}/）
   python experiments/run_real_adaptive.py \
     --datasets Electricity \
     --model_variants ts_drift_adapt_severity_s1p0 \
     --seeds 1 \
     --run_name elec_debug

   # Phase-C 真实流调度评估，读取 run_real_adaptive 的 run_id，结果存到 results/phaseC_scheduler_ablation_real/summary/{run_id}/
   python evaluation/phaseC_scheduler_ablation_real.py \
     --log_experiment run_real_adaptive \
     --log_run_id 20250205-120001-abc \
     --run_name summary_real
   ```

更多脚本的 CLI 说明，可参考 `docs/modules/test_scripts.md` 或查看脚本自带的 `--help`。
