# V14 审计（Permutation-test Confirm）V6：D4 修复后只读审计

- 生成时间：2026-01-11 21:25:25
- 环境确认：
  - `which python` -> `/home/ylh/anaconda3/envs/ZZNSTD/bin/python`
  - `python -V` -> `Python 3.10.19`

## 0) 审计范围声明（强约束）
- 只读：未修改任何既有代码/配置；未重跑训练/未生成新 runs（本审计仅读取既有产物与 summary）。
- 未做任何全局搜索/递归扫描（未使用 find/rg/grep -R/os.walk/glob("**")）。
- 逐 run 定位唯一来源：`artifacts/v14/tables/NEXT_STAGE_V14_RUN_INDEX.csv` 的 `log_path`。
- summary 定位规则固定：`summary_path = Path(log_path).with_suffix(".summary.json")`；不 listdir、不 glob。

## 1) RUN_INDEX 总览
- RUN_INDEX：`artifacts/v14/tables/NEXT_STAGE_V14_RUN_INDEX.csv` rows=980
- R1 推断 logs root：`logs_v14fix1`

## 2) R 规则（fail-fast）
| rule | status | first_evidence |
|---|---|---|
| R1 | PASS | expected_logs_root=logs_v14fix1 |
| R2 | PASS | rows=980 dataset_mismatch=0 missing_summary=0 seed_mismatch=0 seed_schema_missing=0 |
| R3 | PASS | dup_clusters=0 |
| R4 | PASS | sea_nodrift/sine_nodrift label_purity=1.0 |
| R5 | PASS | tables consistent with stats |

## 2.1) 旁证：V14_STRICT_PATH_AUDIT（独立脚本输出）
- RUN_INDEX：`artifacts/v14/tables/NEXT_STAGE_V14_RUN_INDEX.csv`
- 总行数：980
- 违反行数：0

## 3) 关键结论
- dataset 对齐：dataset mismatch=0、missing_summary=0、seed mismatch=0（见表 `ALIGNMENT_STATS_BY_DATASET`）。
- no-drift 纯度：sea_nodrift/sine_nodrift `label_purity=1.0`（见表 `NODRIFT_LABEL_PURITY_AUDIT`）。
- 复用检查：跨 dataset 的 run_id/log_path 复用簇=0（见表 `RUN_INDEX_DUP_BY_RUN_ID` / `RUN_INDEX_DUP_BY_LOG_PATH`）。
- 结论一句话：D4 修复后 dataset 对齐恢复且未推翻 winner 声明（见 `WINNER_RECOMPUTE_CHECK` 与 `DIFF_VS_REPORT_CLAIMS`）。

## 4) 产物
- `V14_AUDIT_PERM_CONFIRM_V6.md`
- `V14_AUDIT_PERM_CONFIRM_V6_TABLES.csv`
