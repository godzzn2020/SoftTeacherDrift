# v15p3 日志归档与定位

本目录用于说明 v15p3 的日志/summary.json 在哪里，以及如何从报告定位到最小可审计路径。

## 归档策略
- 若本目录下存在 `logs_v15p3_s0/`、`logs_v15p3_s1/`：表示已从仓库根目录归档至此（仅做搬运，不改变实验逻辑）。
- 若不存在：请在仓库根目录查找 `logs_v15p3_s0/`、`logs_v15p3_s1/` 与 `logs_two_gpu/v15p3_stability/`。

## two-gpu 分片
- shard0（cuda0）：对应 `log_root_suffix=v15p3_s0`（通常写入 `logs_v15p3_s0/`）
- shard1（cuda1）：对应 `log_root_suffix=v15p3_s1`（通常写入 `logs_v15p3_s1/`）

## 最小可审计路径（示例）
- 报告 -> RAW：`artifacts/v15p3/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P3_RAW.csv`
- RAW 行内 `log_path` 指向单次 run 的 csv；其同目录 `*.summary.json` 包含 `last_perm_pvalue/last_perm_effect/perm_pvalue_pXX` 等诊断字段。

