# 实验产物归档规则（SoftTeacherDrift）

本文件定义“代码 vs 产物”的目录职责、版本命名、产物命名、日志隔离与可审计性要求；用于保证任何人只看目录即可定位到某个版本（如 v15p3）的脚本/命令/日志/CSV/报告及其对应关系。

## 1) 目录规范（强制）

- `experiments/`：实验运行脚本（会生成日志/summary.json 与 TrackAL 产物）。
- `scripts/`：可执行的汇总/分析/工具脚本（summarize/merge/diag 等）；不应放置新产物。
- `artifacts/`：所有实验产物与日志的归档根目录（按版本分层）。
- `docs/`：规则、索引与项目说明文档。

## 2) 版本命名规范（强制）

- 目录名统一小写：`v<major>p<patch>`（例如 `v15p3` 对应 `V15P3` / `V15.3`）。
- 若无 `p`（例如 `V15`），目录使用 `v15`。
- 报告标题可保留 `V15P3`/`V15.3` 的展示形式，但路径与目录名使用小写目录名。

## 3) 产物命名规范（强制）

每个版本目录 `artifacts/<version>/` 下建议固定子目录：

- `reports/`
  - `NEXT_STAGE_<VERSION>_REPORT.md`
  - `NEXT_STAGE_<VERSION>_REPORT_FULL.md`
- `tables/`
  - `TRACKAL_*.csv`（聚合表）
  - `*_RAW.csv`（逐 seed 明细表）
  - `TRACKAM_*.csv`（诊断表）
  - `<VERSION>_METRICS_TABLE.csv` / `<VERSION>_RUN_INDEX.csv`（汇总/索引）
- `commands/`：运行命令（cmd-file、可复制命令段落等）。
- `logs/`：日志与最小审计指针（见下一节）。
- `MANIFEST.md`：该版本入口文件（强制，见第 5 节）。

任何新脚本产物文件名必须携带版本号与用途（例如 `..._V15P4_...`），避免跨版本覆盖。

## 4) 日志规范（强制/建议）

强制：
- 输出隔离：并行/分片运行时，必须显式隔离 `--log_root_suffix` 与 `--out_csv/--out_raw_csv`（或等价 output_dir）。
- two-gpu 并行仅在“无共享写入/无依赖”且每 shard 输出隔离时允许；禁止用 `&` 后台并行。

建议：
- 日志归档到 `artifacts/<version>/logs/`；若日志过大不移动，也必须在 `artifacts/<version>/logs/README.md` 写明：
  - 原始日志根目录位置
  - shard/cuda 对应关系（cuda0/cuda1、`--num_shards/--shard_idx`）
  - 报告中关键 case 的最小可审计定位路径（到 `.summary.json` 或 `.csv`）

## 5) 可审计性要求（强制）

- 报告必须引用产物路径：报告 -> TrackAL CSV ->（可选）RAW CSV -> logs 目录/`*.summary.json`。
- TrackAL/RAW 至少包含：`group/dataset/seed` 与关键参数列（例如 `perm_side`）。
- 每个版本必须存在 `artifacts/<version>/MANIFEST.md`，列出：
  - 关键产物路径（report/full/trackal/raw/trackam/metrics/run_index）
  - 运行命令（最小复现：至少能重跑 summarize/diag）
  - 日志根目录与分片说明
  - 关键结论摘要（3-5 行）

## 6) 兼容策略（强制）

- 本仓库不保留旧路径下的日志/CSV/报告；一律以 `artifacts/` 为准，避免同名产物在多个目录并存导致审计混乱。
- 不允许在 `scripts/` 下新增任何“产物文件”（CSV/MD/图片/日志等）；`scripts/` 仅保留可执行脚本。
