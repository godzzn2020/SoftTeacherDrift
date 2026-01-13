# v15 日志归档与定位

v15 相关日志通常对应 `logs_v15vote_s0/`、`logs_v15vote_s1/` 与 `logs_two_gpu/...`（若已归档则位于本目录下）。

最小审计路径建议从：
- `artifacts/v15/tables/TRACKAL_PERM_CONFIRM_SWEEP_V15.csv` 的 `run_index_json`
- 进一步定位到具体 run 的 `log_path` 与同目录 `*.summary.json`

