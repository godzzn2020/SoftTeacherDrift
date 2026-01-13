# 文档索引（INDEX）

## 核心总览
- [PROJECT_OVERVIEW](PROJECT_OVERVIEW.md) — 项目简介、代码结构、协作规则
- [EXPERIMENT_ARTIFACT_RULES](EXPERIMENT_ARTIFACT_RULES.md) — 实验产物/日志归档规范（artifacts 目录）
- [artifacts/INDEX](../artifacts/INDEX.md) — 各版本入口索引（报告/CSV/日志/命令）

## 模块级文档
- [数据流与合成生成](modules/data_streams.md) — `data/streams.py` 及相关工具
- [教师–学生模型](modules/teacher_student.md) — `models/teacher_student.py`
- [漂移信号与超参调度](modules/drift_and_scheduler.md) — `drift/*`, `scheduler/*`
- [训练循环与日志](modules/training_loop.md) — `training/loop.py`, `run_experiment.py`
- [Stage1 实验设置](modules/experiments_stage1.md) — `experiments/first_stage_experiments.py`
- [漂移评估指标](modules/evaluation_metrics.md) — `evaluation/` 目录
- [测试脚本与常用命令](modules/test_scripts.md) — 训练/评估脚本 CLI 参数与示例
- [Phase0 离线监督基线](modules/phase0_offline_supervised.md) — 真实数据划分、Tabular MLP、Phase0 训练与汇总脚本
- [Phase1 离线半监督](modules/phase1_offline_semi_ema.md) — Tabular MLP + EMA Teacher、离线半监督实验脚本

## 变更记录
- [CHANGELOG](../changelog/CHANGELOG.md)
