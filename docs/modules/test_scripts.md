# 测试脚本与常用命令

## 目的

- 汇总当前仓库中“可直接运行的测试/评估脚本”，便于快速回顾用途、输入与典型命令。
- 约定：**新增或修改测试脚本时必须同步更新本文件**，保持文档与代码一致。

## 离线 / 评估脚本

- **experiments/offline_detector_sweep.py**  
  - 作用：在现有训练日志 + meta.json 上离线网格搜索 detector 组合，输出 MDR/MTD/MTFA/MTR 及 Markdown 总结。  
  - 示例命令：  
    ```bash
    python experiments/offline_detector_sweep.py \
      --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3,INSECTS_abrupt_balanced \
      --model_variant baseline_student \
      --seeds 1 2 3
    ```  
  - 调试：`--debug_sanity` 可快速验证 River detector 会触发漂移。

- **experiments/analyze_abrupt_results.py**  
  - 作用：汇总 stage1 突变漂移实验产生的日志，重算指标、画 Accuracy/Drift timeline。  
  - 示例命令：  
    ```bash
    python experiments/analyze_abrupt_results.py \
      --datasets sea_abrupt4 sine_abrupt4 stagger_abrupt3 INSECTS_abrupt_balanced \
      --models baseline_student mean_teacher ts_drift_adapt \
      --seeds 1 2 3
    ```

## 训练 / 回归脚本

- **run_experiment.py**  
  - 单次 Teacher-Student 训练入口，支持 `--monitor_preset`（none / error_ph_meta / divergence_ph_meta / error_divergence_ph_meta）。  
  - 示例命令：  
    ```bash
    python run_experiment.py \
      --dataset_type sea \
      --dataset_name sea_abrupt4 \
      --model_variant ts_drift_adapt \
      --monitor_preset error_divergence_ph_meta \
      --log_path logs/sea_abrupt4/test.csv
    ```

- **experiments/first_stage_experiments.py**  
  - Stage-1 批量实验脚本，现也支持 `--monitor_preset`，会自动写入 `logs/{dataset}/...`.  
  - 示例命令：  
    ```bash
    python experiments/first_stage_experiments.py \
      --datasets sea_abrupt4,sine_abrupt4,stagger_abrupt3,INSECTS_abrupt_balanced \
      --models baseline_student,mean_teacher,ts_drift_adapt \
      --monitor_preset error_divergence_ph_meta
    ```

## 维护规则

- 若新增新的评估/测试脚本，或为现有脚本添加关键 CLI 功能，请在提交中同步更新本文件：
  - 描述脚本用途；
  - 给出典型命令示例；
  - 标注依赖或注意事项。  
- 同时在相关模块文档（例如 `docs/modules/experiments_stage1.md`）中提示读者查看本文件，以确保所有测试入口都有文档可查。
