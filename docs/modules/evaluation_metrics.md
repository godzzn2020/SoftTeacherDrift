# 漂移评估指标

## 合成流指标

- **Missed Detection Rate (MDR)**  
  `MDR = (# 未被检测到的真实漂移) / (# 真实漂移)`  
  若某真实漂移在下一个漂移前都没有对应报警，则视为漏检。
- **Mean Time to Detection (MTD)**  
  对每个被检测的漂移，记录 `delay = detection_time - ground_truth_time`；MTD 为这些延迟的平均值。
- **Mean Time between False Alarms (MTFA)**  
  所有假警报之间的平均间隔；若假警报数 < 2，则为 `NaN`。
- **Mean Time Ratio (MTR)**  
  综合指标：`MTR = MTFA / (MTD * (1 - MDR))`，在漏检少、检测快、假警报稀疏时更大。
- 真实漂移位置源自 `meta.json` 的 `drifts[start]`，检测时间取日志的 `sample_idx`（0-based）；实现位于 `evaluation/drift_metrics.py`。
- 对 `INSECTS_abrupt_balanced` 也可使用同样的指标——漂移真值来自 `datasets/real/INSECTS_abrupt_balanced.json` 中的 `positions`（同样 0-based）。

## 真实流指标

- **Lift-per-Drift (lpd)**  
  对比启用漂移检测的模型与 baseline（无检测）之间的整体准确率提升：  
  `lpd = (acc_d - acc_0) / #drifts`，当 `#drifts = 0` 时返回 0。  
  可用于比较在真实流中启用检测是否值得其开销。
- 需要两份日志：带检测 (`model_variant`) 与 baseline (`baseline_variant`，默认 `baseline_student`)。

## 评估脚本

- `evaluation/eval_synthetic_and_real.py`：
  - `--mode synthetic`：读取 meta.json + 对应日志，打印 MDR/MTD/MTFA/MTR + 最终 accuracy。
  - `--mode real`：读取两份日志，计算 `acc_d`、`acc0`、`n_drifts`、`lpd`。对于有真值的真实流（例如 INSECTS）也可以另行调用 `compute_detection_metrics`。
- 指标实现细节在 `evaluation/drift_metrics.py` 中，可按需扩展更多指标。

## TODO

- 增加窗口级 precision/recall、检测延迟分布等补充指标。
- 将评估结果写回统一表格或可视化报告，方便跨实验对比。
- 对真实流加入更多 baseline（如固定策略）以提升 lpd 的解释力。
