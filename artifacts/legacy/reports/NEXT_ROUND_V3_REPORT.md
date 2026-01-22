# NEXT_ROUND V3 Report (Track D/E)

- 生成时间：2026-01-07 14:01:53
- logs_root：`logs`；results_root：`results`
- 产物：`scripts/NEXT_ROUND_V3_REPORT.md` / `scripts/NEXT_ROUND_V3_RUN_INDEX.csv` / `scripts/NEXT_ROUND_V3_METRICS_TABLE.csv`

## 0) Baseline 机制回顾（必须引用）
- 多信号 drift monitor 的默认触发为 **OR**：任一 detector 触发即 `drift_flag=1`（现支持 `or/k_of_n/weighted`，默认仍为 OR）。
- `drift_severity` 默认仅在 `drift_flag=1` 时为正（非 drift 时为 0），用于驱动 severity-aware 调度。
- v1 熵项默认是 **overconfident**：`x_entropy_pos = max(0, baseline_entropy - entropy)`（教师更自信被视为 drift 风险）。
- v1 severity-aware 调度仅在 `drift_flag` 的时刻通过 `drift_severity` 缩放 `alpha/lr/lambda_u/tau`。

## 1) What changed
- `soft_drift/severity.py`：新增 `entropy_mode ∈ {overconfident, uncertain, abs}`。
- `training/loop.py`：新增 `severity_carry = max(severity_carry*decay, drift_severity)` 与 `freeze_baseline_steps`（v2 默认关闭）。
- `drift/detectors.py`：新增 `trigger_mode ∈ {or, k_of_n, weighted}`（默认 OR），并记录 `monitor_vote_*`/`monitor_fused_severity`。
- 推荐 weighted 参数（本轮 Track E 使用）：`w(error)=0.5, w(div)=0.3, w(entropy)=0.2, threshold=0.5`（偏向“error 必须触发”）。

## 2) Track D：Severity-Aware v2（恢复收益）
- 评估：acc_final/acc_min + drift metrics 两套口径（window-based vs tolerance=500/min_sep=200）+ 恢复指标 W=1000。

### D-Table（汇总）
| dataset | group | n_runs | acc_final(mean±std) | acc_min(mean±std) | post_mean_acc@W1000(mean±std) | post_min_acc@W1000(mean±std) | MDR_win | MTD_win | MTFA_win | MTR_win | MDR_tol(t=500) | MTD_tol(t=500) | MTFA_tol | MTR_tol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| INSECTS_abrupt_balanced | A) baseline | 5 | 0.2005±0.0133 | 0.1488±0.0338 | 0.1929±0.0111 | 0.1922±0.0110 | 0.2000±0.0000 | 5400.7000±114.4867 | 14933.3333±0.0000 | 3.4576±0.0713 | 1.0000±0.0000 | N/A | 6400.0000±0.0000 | N/A |
| INSECTS_abrupt_balanced | B) v1 | 5 | 0.2005±0.0050 | 0.1440±0.0373 | 0.1902±0.0066 | 0.1894±0.0066 | 0.2000±0.0000 | 5400.7000±114.4867 | 14950.4000±38.1622 | 3.4614±0.0628 | 1.0000±0.0000 | N/A | 6407.3143±16.3552 | N/A |
| INSECTS_abrupt_balanced | C) v2(unc,d=0.95,f=5) | 5 | 0.1962±0.0101 | 0.1345±0.0451 | 0.1901±0.0084 | 0.1892±0.0082 | 0.2400±0.0894 | 4491.0667±1418.1811 | 13465.6000±2010.2861 | 4.5552±2.5268 | 0.9200±0.1789 | 221.0000±NaN | 6772.4800±1326.1473 | 102.5158±NaN |
| sea_abrupt4 | A) baseline | 5 | 0.8668±0.0081 | 0.4672±0.1092 | 0.8272±0.0253 | 0.8262±0.0258 | 0.0000±0.0000 | 727.0000±153.8831 | 1791.9006±66.5668 | 2.5366±0.4329 | 0.6000±0.1369 | 346.2000±75.5989 | 1677.4288±49.5111 | 13.1963±2.4058 |
| sea_abrupt4 | B) v1 | 5 | 0.8720±0.0046 | 0.4917±0.1611 | 0.8320±0.0246 | 0.8311±0.0249 | 0.0000±0.0000 | 848.6000±170.3080 | 1857.1852±78.7732 | 2.2567±0.4283 | 0.6500±0.1369 | 325.4000±111.0531 | 1685.9154±46.5362 | 20.5236±16.5503 |
| sea_abrupt4 | C) v2(unc,d=0.95,f=5) | 5 | 0.8704±0.0054 | 0.4907±0.1680 | 0.8303±0.0244 | 0.8293±0.0248 | 0.0000±0.0000 | 711.0000±100.5584 | 1792.0684±59.8865 | 2.5533±0.3068 | 0.5500±0.1118 | 303.0000±111.1396 | 1667.2102±41.9798 | 17.2952±16.2674 |

### D-结论（要点）
- 重点看 `acc_min` 与 post-drift 恢复指标（是否抬高谷底/加速恢复），其次看 `acc_final` 是否保持。
- 若 v2 不提升：结合 `drift_flag_count`、`drift_severity` 与 carry 的持续性，判断是否 carry 衰减过快/基线被过快吸收。
- sea_abrupt4：v1/v2 的 acc_min 均值分别为 0.4917 / 0.4907，高于 baseline 0.4672；但 v2 未明显优于 v1。
- INSECTS：v2 的 acc_final/acc_min 均值 0.1962/0.1345 低于 baseline 0.2005/0.1488，本轮未体现恢复收益。

## 3) Track E：监控融合策略（trigger_mode）

### E-Table（汇总）
| dataset | trigger_mode | n_runs | acc_final(mean±std) | mean_acc(mean±std) | MDR_win | MTD_win | MTFA_win | MTR_win | MDR_tol(t=500) | MTD_tol(t=500) | MTFA_tol | MTR_tol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sea_abrupt4 | or | 5 | 0.8745±0.0080 | 0.8294±0.0099 | 0.0000±0.0000 | 762.2000±123.1065 | 1809.1496±53.0164 | 2.4162±0.3379 | 0.6500±0.2236 | 377.0000±98.2785 | 1679.2664±51.1592 | 11.3710±3.2865 |
| sea_abrupt4 | k_of_n(k=2) | 5 | 0.8528±0.0094 | 0.8099±0.0106 | 1.0000±0.0000 | N/A | N/A | N/A | 1.0000±0.0000 | N/A | N/A | N/A |
| sea_abrupt4 | weighted(w=0.5/0.3/0.2,t=0.5) | 5 | 0.8721±0.0053 | 0.8274±0.0034 | 0.0000±0.0000 | 727.0000±89.7998 | 1897.0240±36.0026 | 2.6354±0.2680 | 0.6000±0.1369 | 275.8000±130.3043 | 1728.9820±28.3668 | 29.1092±33.6777 |
| stagger_abrupt3 | or | 5 | 0.9319±0.0051 | 0.9464±0.0075 | 0.0000±0.0000 | 59.8000±17.5271 | 3800.3172±409.0571 | 69.0712±23.8726 | 0.0000±0.0000 | 59.8000±17.5271 | 3878.3972±466.9572 | 70.7324±25.3340 |
| stagger_abrupt3 | k_of_n(k=2) | 5 | 0.8540±0.0063 | 0.9145±0.0062 | 1.0000±0.0000 | N/A | N/A | N/A | 1.0000±0.0000 | N/A | N/A | N/A |
| stagger_abrupt3 | weighted(w=0.5/0.3/0.2,t=0.5) | 5 | 0.9304±0.0026 | 0.9467±0.0041 | 0.0000±0.0000 | 59.8000±17.5271 | 4831.5572±539.1527 | 84.7708±18.3945 | 0.0000±0.0000 | 59.8000±17.5271 | 4831.5572±539.1527 | 84.7708±18.3945 |

### E-结论（要点）
- `k_of_n(k=2)` 在本轮 preset=error+divergence 下相当激进：检测显著减少但漏检会上升，且分类往往受影响（见表）。
- `weighted(w=0.5/0.3/0.2,t=0.5)` 倾向“error 必须触发”，可在减少误报的同时维持检测覆盖（具体以 MDR/MTFA 为准）。
- sea_abrupt4：k=2 明显掉点（acc_final 均值 0.8528 vs OR 0.8745），weighted 接近 OR（0.8721）。

## 4) 讨论：为何 v1 不显著、v2 为何更合理
- v1 只在 `drift_flag` 当步做一次性缩放，若漂移影响持续多步（或检测延迟），单步调参可能不足以帮助恢复。
- v2 的 `severity_carry` 让严重度在漂移后窗口内持续生效（可解释为“恢复期”的调参记忆），并通过 `decay` 控制影响时长。
- `freeze_baseline_steps` 防止漂移后 baseline 过快贴合新分布导致严重度迅速归零，从而丢失恢复驱动。
- `entropy_mode=uncertain/abs` 能覆盖“教师更不确定”的漂移形态，避免只对过度自信敏感。

## 5) 下一步建议
- 若 v2 在 `acc_min`/恢复指标上稳定提升：建议 sweep `decay∈{0.9,0.95,0.98}` 与 `freeze_baseline_steps∈{0,5,10}`，并对 entropy_mode 做 `overconfident vs uncertain vs abs` 对照。
- 若 weighted 融合更稳：建议在 `all_signals_ph_meta` 下引入 entropy detector，并把权重/阈值与误报成本绑定（例如阈值从 0.5→0.7）。

