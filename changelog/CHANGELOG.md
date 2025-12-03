# 变更记录（CHANGELOG）

记录格式：每次较大改动追加一个小节，包含日期与 1–3 条要点（涉及文件、功能、风险等）。保持倒序（最新在前）。

## 2025-XX-XX  文档体系初始化

- 创建 `docs/PROJECT_OVERVIEW.md` 与 `docs/INDEX.md`，概述项目目标与文档索引。
- 为核心模块添加 `docs/modules/*.md` 草稿，说明接口与 TODO。
- 新建 `changelog/CHANGELOG.md`，约定未来重大更新都在此记录。

## 2025-XX-XX  Abrupt Stage1 扩展

- `data/streams.py` 新增 Sine/STAGGER 合成流生成、默认批量生成 CLI，以及 INSECTS_real 支持与 sample 元信息。
- `training/loop.py` 日志增加 0-based `sample_idx`，统一漂移检测时间戳。
- 新增 `experiments/abrupt_stage1_experiments.py` 与 `experiments/analyze_abrupt_results.py`，提供突变漂移批量运行、汇总与绘图；相应文档与指标说明同步更新。

2025-11-20  [assistant]  修改 summary：新增离线 detector 网格搜索脚本，支持多信号/检测器组合的批量评估  
影响文件：experiments/offline_detector_sweep.py, docs/modules/experiments_stage1.md

2025-11-20  [assistant]  修改 summary：修复 offline detector sweep 脚本的模块导入路径问题  
影响文件：experiments/offline_detector_sweep.py

2025-11-20  [assistant]  修改 summary：纠正 PageHinkley 参数名（lambda_→alpha），确保离线网格搜索能运行 PH 组合  
影响文件：experiments/offline_detector_sweep.py

2025-11-20  [assistant]  修改 summary：统一 river 漂移检测标志读取，修复在线/离线 detector 永不触发的问题，并补充 sanity check CLI  
影响文件：experiments/offline_detector_sweep.py, drift/detectors.py, docs/modules/experiments_stage1.md

2025-11-20  [assistant]  修改 summary：新增离线校准得出的 monitor presets，并在 Stage-1 CLI 暴露 `--monitor_preset`（none/error_ph_meta/divergence_ph_meta/error_divergence_ph_meta）  
影响文件：drift/detectors.py, experiments/first_stage_experiments.py, run_experiment.py, docs/modules/experiments_stage1.md

2025-11-20  [assistant]  修改 summary：整理可用测试脚本，新增 `docs/modules/test_scripts.md` 并在索引/总览中强调 CLI 维护规则  
影响文件：docs/modules/test_scripts.md, docs/INDEX.md, docs/PROJECT_OVERVIEW.md

2025-11-20  [assistant]  修改 summary：修复 Stage-1 脚本直接运行时的模块导入错误（添加项目根路径）  
影响文件：experiments/first_stage_experiments.py

2025-11-20  [assistant]  修改 summary：新增 `experiments/parallel_stage1_launcher.py`，支持多 GPU 并行调度 Stage-1 组合，并在文档中记录 CLI 参数  
影响文件：experiments/parallel_stage1_launcher.py, docs/modules/test_scripts.md, docs/modules/experiments_stage1.md

2025-11-20  [assistant]  修改 summary：Stage-1 默认 INSECTS 改为使用本地 CSV（`insects_real`），避免 river 下载 404  
影响文件：experiments/first_stage_experiments.py, docs/modules/experiments_stage1.md

2025-11-20  [assistant]  修改 summary：新增 `experiments/summarize_online_results.py`（生成在线 runs/summary/Markdown/Accuracy & Detection SVG，并计算在线 MDR/MTD/MTFA/MTR），同时完善图表轴标签/图例，并在相关文档补充 CLI  
影响文件：experiments/summarize_online_results.py, docs/modules/test_scripts.md, docs/modules/experiments_stage1.md

2025-11-20  [assistant]  修改 summary：Stage-1 默认实验新增 `sine_abrupt4` 与 `stagger_abrupt3`，确保并行/批量脚本能覆盖离线调参的合成流  
影响文件：experiments/first_stage_experiments.py, docs/modules/experiments_stage1.md

2025-11-20  [assistant]  修改 summary：统一 CLI 脚本的 `sys.path` 注入规范（以避免 `ModuleNotFoundError`），并在文档中明确要求  
影响文件：run_experiment.py, docs/modules/test_scripts.md, docs/PROJECT_OVERVIEW.md

2025-11-20  [assistant]  修改 summary：新增 `experiments/stage1_multi_seed.py`（自动运行多 seed 实验、汇总 Raw/Summary CSV 与 Markdown，并在训练前自动生成对应 seed 的合成流 parquet/meta），文档同步添加使用说明  
影响文件：experiments/stage1_multi_seed.py, docs/modules/experiments_stage1.md, docs/modules/test_scripts.md

2025-11-20  [assistant]  修改 summary：新增 `experiments/run_real_adaptive.py`，可在 Electricity/NOAA/INSECTS/Airlines 等真实流上批量运行 `ts_drift_adapt` 多 seed 实验，并在测试脚本文档登记  
影响文件：experiments/run_real_adaptive.py, docs/modules/test_scripts.md

2025-11-20  [assistant]  修改 summary：新增 `evaluation/phaseB_signal_drift_analysis_real.py`，用于真实数据集的信号/检测可视化，并在文档登记该脚本  
影响文件：evaluation/phaseB_signal_drift_analysis_real.py, docs/modules/test_scripts.md, docs/modules/experiments_stage1.md

2025-11-20  [assistant]  修改 summary：新增 Phase C2 严重度拟合脚本 `evaluation/phaseC_severity_score_fit.py`，并在测试脚本文档记录 CLI，用于统一信号严重度指标  
影响文件：evaluation/phaseC_severity_score_fit.py, docs/modules/test_scripts.md, changelog/CHANGELOG.md

2025-11-20  [assistant]  修改 summary：引入 `SeverityCalibrator` + severity-aware scheduler（`ts_drift_adapt_severity`），训练日志新增 `monitor_severity`/`drift_severity_raw`/`drift_severity` 字段，并提供 Phase C3 调度消融脚本  
影响文件：soft_drift/severity.py, scheduler/hparam_scheduler.py, training/loop.py, experiments/first_stage_experiments.py, experiments/stage1_multi_seed.py, experiments/parallel_stage1_launcher.py, evaluation/phaseB_signal_drift_analysis_real.py, evaluation/phaseC_scheduler_ablation_synth.py, docs/PROJECT_OVERVIEW.md, docs/modules/drift_and_scheduler.md, docs/modules/training_loop.md, docs/modules/experiments_stage1.md, docs/modules/test_scripts.md

2025-11-20  [assistant]  修改 summary：扩展 `drift/detectors.py` 的 PageHinkley 预设为 error/entropy/divergence 的 7 种组合，并新增合成流检测消融脚本 `evaluation/phaseB_detection_ablation_synth.py`  
影响文件：drift/detectors.py, evaluation/phaseB_detection_ablation_synth.py, docs/modules/drift_and_scheduler.md, docs/modules/test_scripts.md, docs/modules/experiments_stage1.md
