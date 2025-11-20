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

2025-11-20  [assistant]  修改 summary：新增 `experiments/summarize_online_results.py`（生成在线 runs/summary/Markdown/Accuracy & Detection SVG，并计算在线 MDR/MTD/MTFA/MTR），并在相关文档补充 CLI  
影响文件：experiments/summarize_online_results.py, docs/modules/test_scripts.md, docs/modules/experiments_stage1.md
