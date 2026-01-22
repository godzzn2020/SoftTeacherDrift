# NEXT_STAGE V4 Report (Track I/J/K/L)

- 生成时间：2026-01-08 00:40:26
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`/home/ylh/anaconda3/envs/ZZNSTD/bin/python` / `3.10.19`

## 最小定位（入口与关键产物路径）
- 入口代码：`experiments/trackI_delay_diag.py`、`experiments/trackJ_twostage_sweep.py`、`experiments/trackK_generalization.py`、`experiments/trackL_gating_sweep.py`
- 关键实现：`drift/detectors.py`、`training/loop.py`、`soft_drift/severity.py`
- 汇总脚本：`scripts/summarize_next_stage_v4.py`
- 既有产物（F/G/H）：`scripts/TRACKF_THRESHOLD_SWEEP.csv`、`scripts/TRACKG_TWO_STAGE.csv`、`scripts/TRACKH_SEVERITY_GATING.csv`、`scripts/NEXT_STAGE_REPORT.md`
- 本阶段产物（I/J/K/L）：`scripts/TRACKI_DELAY_DIAG.csv`、`scripts/TRACKJ_TWOSTAGE_SWEEP.csv`、`scripts/TRACKK_GENERALIZATION.csv`、`scripts/TRACKL_GATING_SWEEP.csv`

## 口径统一（window vs tol500）
- 时间轴：本报告对合成/INSECTS 的 GT drift 均使用 `sample_idx`（与 meta.json 的 positions 对齐）。
- window 口径（Lukats window-based）：只要在“当前 drift 到下一个 drift 之间”出现过一次报警，就不算漏检；因此**晚检也会被视为命中**。
- tol500 口径：要求在 `drift_pos + 500` 内出现报警才算命中；因此**晚检会被计为漏检**。
- 结论：当出现 `MDR_win≈0` 但 `MDR_tol500` 较高时，通常不是完全漏检，而是**检测延迟显著（P90/P99>500）**。Track I 会给出延迟分位数来解释该冲突。

## Track I：延迟诊断（必须）
| dataset | mode | n(drifts×seeds) | miss_win | miss_tol500 | cand_P50 | cand_P90 | cand_P99 | conf_P50 | conf_P90 | conf_P99 |
|---|---|---|---|---|---|---|---|---|---|---|
| sea_abrupt4 | or | 20 | 0.000 | 0.700 | 799.0 | 1199.0 | 1678.5 | 799.0 | 1199.0 | 1678.5 |
| sea_abrupt4 | two_stage | 20 | 0.000 | 0.700 | 831.0 | 1283.8 | 1443.6 | 831.0 | 1283.8 | 1443.6 |
| sea_abrupt4 | weighted_t0.40 | 20 | 0.000 | 0.800 | 799.0 | 1199.0 | 1523.0 | 799.0 | 1199.0 | 1523.0 |
| sea_abrupt4 | weighted_t0.50 | 20 | 0.000 | 0.750 | 799.0 | 1199.0 | 1302.7 | 799.0 | 1199.0 | 1302.7 |
| stagger_abrupt3 | or | 10 | 0.000 | 0.000 | 63.0 | 98.2 | 124.1 | 63.0 | 98.2 | 124.1 |
| stagger_abrupt3 | two_stage | 10 | 0.000 | 0.000 | 63.0 | 95.0 | 95.0 | 63.0 | 95.0 | 95.0 |
| stagger_abrupt3 | weighted_t0.40 | 10 | 0.000 | 0.000 | 63.0 | 98.2 | 124.1 | 63.0 | 98.2 | 124.1 |
| stagger_abrupt3 | weighted_t0.50 | 10 | 0.000 | 0.000 | 63.0 | 98.2 | 124.1 | 63.0 | 98.2 | 124.1 |

**Track I 结论（<=8 行）**
- 若某个 mode 的 `miss_win` 低但 `miss_tol500` 高，同时 `conf_P90` 或 `conf_P99` > 500：说明主要问题是“晚检”，论文中应把 tol500 作为辅助口径用于强调实时性。
- `or` 往往 candidate 最早但误报更多；`weighted`/`two_stage` 倾向降低误报但可能带来确认延迟。

## Track J：two_stage 小扫（稳健默认）
| dataset | theta | window | acc_final(mean±std) | acc_min(mean±std) | miss_tol500(mean) | delay_P90(mean) | MDR_tol(mean) | MTR_win(mean) |
|---|---|---|---|---|---|---|---|---|
| sea_abrupt4 | 0.30 | 1 | 0.8731±0.0127 | 0.5437±0.1514 | 0.550 | 1170.5 | 0.550 | 2.491 |
| sea_abrupt4 | 0.30 | 2 | 0.8657±0.0067 | 0.5000±0.2024 | 0.700 | 1082.8 | 0.700 | 2.544 |
| sea_abrupt4 | 0.30 | 3 | 0.8723±0.0073 | 0.4703±0.1875 | 0.650 | 1077.1 | 0.650 | 2.570 |
| sea_abrupt4 | 0.40 | 1 | 0.8651±0.0071 | 0.5238±0.1727 | 0.800 | 1101.1 | 0.800 | 2.497 |
| sea_abrupt4 | 0.40 | 2 | 0.8725±0.0118 | 0.3781±0.1395 | 0.600 | 1079.0 | 0.600 | 2.933 |
| sea_abrupt4 | 0.40 | 3 | 0.8725±0.0073 | 0.4406±0.1341 | 0.850 | 1156.4 | 0.850 | 2.217 |
| sea_abrupt4 | 0.50 | 1 | 0.8684±0.0094 | 0.4906±0.1460 | 0.700 | 1190.7 | 0.700 | 2.380 |
| sea_abrupt4 | 0.50 | 2 | 0.8657±0.0071 | 0.4188±0.1748 | 0.700 | 1079.0 | 0.700 | 2.649 |
| sea_abrupt4 | 0.50 | 3 | 0.8706±0.0082 | 0.4223±0.1572 | 0.600 | 1153.9 | 0.600 | 2.702 |
| stagger_abrupt3 | 0.30 | 1 | 0.9280±0.0038 | 0.6625±0.2203 | 0.000 | 84.1 | 0.000 | 61.472 |
| stagger_abrupt3 | 0.30 | 2 | 0.9314±0.0030 | 0.7539±0.1290 | 0.000 | 95.6 | 0.000 | 51.286 |
| stagger_abrupt3 | 0.30 | 3 | 0.9321±0.0054 | 0.4469±0.3203 | 0.000 | 84.1 | 0.000 | 64.293 |
| stagger_abrupt3 | 0.40 | 1 | 0.9264±0.0044 | 0.1125±0.0434 | 0.000 | 84.1 | 0.000 | 77.464 |
| stagger_abrupt3 | 0.40 | 2 | 0.9291±0.0045 | 0.4375±0.2909 | 0.000 | 84.1 | 0.000 | 75.134 |
| stagger_abrupt3 | 0.40 | 3 | 0.9253±0.0047 | 0.4356±0.3312 | 0.000 | 66.2 | 0.000 | 86.308 |
| stagger_abrupt3 | 0.50 | 1 | 0.9299±0.0012 | 0.3625±0.3002 | 0.000 | 84.1 | 0.000 | 83.435 |
| stagger_abrupt3 | 0.50 | 2 | 0.9291±0.0043 | 0.5344±0.3041 | 0.000 | 84.1 | 0.000 | 79.890 |
| stagger_abrupt3 | 0.50 | 3 | 0.9329±0.0037 | 0.4844±0.2815 | 0.000 | 95.6 | 0.000 | 70.483 |

**推荐组合（按数据集）**
- sea_abrupt4: theta=0.3, window=1（规则：acc_final_mean≥best-0.01，先最小化 miss_tol500，再最大化 acc_min（再看 MTR_win））
- stagger_abrupt3: theta=0.3, window=2（规则：acc_final_mean≥best-0.01，先最小化 miss_tol500，再最大化 acc_min（再看 MTR_win））

**推荐组合（全局默认）**
- theta=0.3, window=1（全局默认：平均 miss_tol500 最小（0.275），且平均 acc_min 最大（0.603））

## Track K：泛化验证（INSECTS 必须）
| dataset | mode | n_runs | acc_final(mean±std) | acc_min(mean±std) | post_mean@W1000(mean±std) | post_min@W1000(mean±std) | recovery_time_to_pre90(mean±std) | MDR_win(mean±std) | MDR_tol500(mean±std) |
|---|---|---|---|---|---|---|---|---|---|
| INSECTS_abrupt_balanced | or | 10 | 0.1975±0.0100 | 0.1279±0.0408 | 0.1868±0.0093 | 0.1860±0.0093 | 178.4±0.0 | 0.200±0.094 | 0.940±0.135 |
| INSECTS_abrupt_balanced | two_stage | 10 | 0.1999±0.0072 | 0.1522±0.0188 | 0.1906±0.0089 | 0.1898±0.0089 | 178.4±0.0 | 0.220±0.063 | 0.960±0.126 |
| INSECTS_abrupt_balanced | weighted | 10 | 0.2008±0.0094 | 0.1482±0.0351 | 0.1902±0.0088 | 0.1895±0.0088 | 178.4±0.0 | 0.240±0.084 | 0.920±0.169 |

**Track K 结论（<=8 行）**
- 重点看 `acc_min`、`post_*@W1000` 与 `recovery_time_to_pre90` 是否在 two_stage 下更稳（且 `acc_final` 不明显下降）。
- 若 two_stage 的 `MDR_tol500` 下降且延迟统计改善，同时分类不掉点：可作为论文默认触发策略。

## Track L：severity v2 gating 强化验证（INSECTS）
| dataset | group | n_runs | acc_final(mean±std) | acc_min(mean±std) | post_mean@W1000(mean±std) | post_min@W1000(mean±std) | recovery_time_to_pre90(mean±std) |
|---|---|---|---|---|---|---|---|
| INSECTS_abrupt_balanced | baseline | 10 | 0.1975±0.0083 | 0.1392±0.0429 | 0.1869±0.0056 | 0.1861±0.0055 | 178.4±0.0 |
| INSECTS_abrupt_balanced | v2 | 10 | 0.1974±0.0065 | 0.1432±0.0347 | 0.1898±0.0037 | 0.1890±0.0038 | 178.4±0.0 |
| INSECTS_abrupt_balanced | v2_gate_m1 | 10 | 0.1996±0.0071 | 0.1288±0.0418 | 0.1883±0.0054 | 0.1875±0.0054 | 178.4±0.0 |
| INSECTS_abrupt_balanced | v2_gate_m3 | 10 | 0.2040±0.0098 | 0.1398±0.0377 | 0.1922±0.0068 | 0.1914±0.0068 | 178.4±0.0 |
| INSECTS_abrupt_balanced | v2_gate_m5 | 10 | 0.1986±0.0087 | 0.1563±0.0280 | 0.1905±0.0068 | 0.1896±0.0067 | 178.4±0.0 |

**Track L 结论（<=8 行）**
- 对比 `v2` vs `v2_gate_*`：若 gating 组的 `acc_min` 与 `post_min@W1000` 更高且方差更小，同时 `acc_final` 不显著下降，可支持“confirmed drift gating 缓解负迁移”的结论。
- 若 gating 过强导致 `acc_final` 下滑或恢复指标变差：说明确认条件太苛刻（可回退更小的 min_streak 或降低 trigger_threshold）。

## 最终建议（论文可用默认）
- 主口径建议：window-based 作为主表（与既有文献一致）；tol500 作为辅表强调实时性，并用 Track I 的 delay 分位数解释冲突来源。
- 默认触发建议：two_stage（candidate=OR，confirm=weighted），采用 Track J 的“全局默认”组合；并在论文方法段明确 candidate/confirm 统计（candidate_count/confirmed_count/confirm_delay）。
- severity v2 建议：默认开启 confirmed drift gating（优先 `confirmed_streak`），以 Track L 的恢复指标与 acc_min 作为核心证据。
