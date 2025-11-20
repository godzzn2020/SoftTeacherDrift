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
