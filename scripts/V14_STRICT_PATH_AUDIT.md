# V14 STRICT PATH AUDIT（fail-fast）

## 范围声明（强约束）
- 未做任何递归扫描（不 os.walk/find/glob/listdir）。
- summary 读取仅按固定规则：`log_path` 末尾 `.csv -> .summary.json`（否则仅追加一次 `.summary.json`）。

## 总览
- RUN_INDEX：`scripts/NEXT_STAGE_V14_RUN_INDEX.csv`
- 总行数：980
- 违反行数：0
- 输出：`scripts/V14_STRICT_PATH_AUDIT.md`、`scripts/V14_STRICT_PATH_AUDIT.csv`

## 按 dataset 违反统计
| dataset | n_rows | n_violations | n_missing_summary | n_dataset_mismatch | n_path_mismatch |
|---|---|---|---|---|---|
| sea_abrupt4 | 245 | 0 | 0 | 0 | 0 |
| sea_nodrift | 245 | 0 | 0 | 0 | 0 |
| sine_abrupt4 | 245 | 0 | 0 | 0 | 0 |
| sine_nodrift | 245 | 0 | 0 | 0 | 0 |

## R3 复用检查（Top-20）
### dup_by_log_path
_N/A_

### dup_by_run_id
_N/A_

## Top-50 违反样例
_N/A_

