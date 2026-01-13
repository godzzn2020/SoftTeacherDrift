# v14 日志归档与定位

本版本相关日志包含但不限于：
- `artifacts/v14/logs/logs_v14fix1/`（仓库中历史目录名；根目录 `logs_v14fix1` 保留软链兼容）

最小审计路径建议从：
- `artifacts/v14/tables/NEXT_STAGE_V14_RUN_INDEX.csv`（包含 run_id/log_path）
- 再定位到具体 `log_path` 与同目录 `*.summary.json`

