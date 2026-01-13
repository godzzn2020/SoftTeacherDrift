# v15p2 日志归档与定位

本目录用于说明 v15p2 的日志/summary.json 在哪里，以及如何从报告定位到最小可审计路径。

## two-gpu 分片
- shard0（cuda0）：通常对应 `log_root_suffix=v15p2_s0`（写入 `logs_v15p2_s0/`）
- shard1（cuda1）：通常对应 `log_root_suffix=v15p2_s1`（写入 `logs_v15p2_s1/`）

## 最小可审计路径
- 报告 -> RAW：`artifacts/v15p2/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P2_RAW.csv`
- RAW 行内 `log_path` 指向单次 run 的 csv；同目录 `*.summary.json` 为诊断字段来源。

