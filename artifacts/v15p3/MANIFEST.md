# MANIFEST: v15p3

## 入口
- 报告：`artifacts/v15p3/reports/NEXT_STAGE_V15P3_REPORT.md`
- 全量报告：`artifacts/v15p3/reports/NEXT_STAGE_V15P3_REPORT_FULL.md`
- TrackAL（聚合）：`artifacts/v15p3/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P3.csv`
- TrackAL（逐 seed RAW）：`artifacts/v15p3/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P3_RAW.csv`
- TrackAM（诊断）：`artifacts/v15p3/tables/TRACKAM_PERM_DIAG_V15P3.csv`
- metrics_table：`artifacts/v15p3/tables/NEXT_STAGE_V15P3_METRICS_TABLE.csv`
- run_index：`artifacts/v15p3/tables/NEXT_STAGE_V15P3_RUN_INDEX.csv`

## 运行命令（最小复现）
- 生成 TrackAM（默认指向 v15p3 stability 产物）：`/home/ylh/anaconda3/envs/ZZNSTD/bin/python experiments/trackAM_perm_diagnostics.py`
- 生成主报告/全量报告：`python scripts/summarize_next_stage_v15.py --trackal_csv artifacts/v15p3/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P3.csv --trackam_csv artifacts/v15p3/tables/TRACKAM_PERM_DIAG_V15P3.csv --out_report artifacts/v15p3/reports/NEXT_STAGE_V15P3_REPORT.md --out_report_full artifacts/v15p3/reports/NEXT_STAGE_V15P3_REPORT_FULL.md --out_run_index artifacts/v15p3/tables/NEXT_STAGE_V15P3_RUN_INDEX.csv --out_metrics_table artifacts/v15p3/tables/NEXT_STAGE_V15P3_METRICS_TABLE.csv --raw_csv artifacts/v15p3/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P3_RAW.csv`

## 分片/日志定位
- two-gpu 命令文件：`artifacts/v15p3/commands/V15P3_vote_score_stability_cmds.txt`
- 日志说明：`artifacts/v15p3/logs/README.md`

## 关键结论（摘要）
- sea/sine hard-ok：成立（见报告“验收结论”段落）。
- no-drift 相对 baseline：仍平均更低（见报告“稳定性复核摘要”段落）。
- stagger_abrupt3 seed=3 worst-case：已修复为可审计口径（见报告“验收结论”段落）。

