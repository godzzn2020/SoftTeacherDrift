# NEXT_STAGE V11 Report（gradual 口径修正 + no-drift 约束校准 + 恢复统计CI）

- 生成时间：2026-01-09 23:45:47
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`/home/ylh/anaconda3/envs/ZZNSTD/bin/python / 3.10.19`

================================================
V11-Track AD：Gradual drift 的区间 GT 口径重算（stagger_gradual_frequent）
================================================

产物：`scripts/TRACKAD_GRADUAL_TOL_AUDIT.csv`

| group | n_runs | miss_tol500_start | miss_tol500_mid | miss_tol500_end | delay_start_P90 | delay_mid_P90 | delay_end_P90 |
|---|---|---|---|---|---|---|---|
| B_or_tunedPH | 5 | 0.400±0.137 | 0.250±0.000 | 0.250±0.000 | 589.1±295.0 | -410.9±295.0 | -1410.9±295.0 |
| C_weighted_tunedPH | 5 | 0.450±0.112 | 0.450±0.112 | 0.450±0.112 | 79.6±29.9 | -920.4±29.9 | -1920.4±29.9 |
| D_two_stage_tunedPH_cd200 | 5 | 0.450±0.112 | 0.400±0.137 | 0.400±0.137 | 674.8±999.0 | -325.2±999.0 | -1325.2±999.0 |

**结论（回答“口径问题 vs 机制问题”）**
- 若 `miss_tol500_start` 高但 `miss_tol500_end` 低，说明主要是“GT 锚点过早（start）”导致的口径问题（检测更接近 transition 后段）。
- 若 `miss_tol500_end` 仍高，则更可能是机制/参数导致的真实漏检（需要进一步调 detector 或 confirm 结构）。
- 本轮结论：区间口径（start→end）能降低 miss（部分组从 ~0.40 降到 ~0.25），但 `miss_tol500_end` 仍显著>0，说明 stagger_gradual 的高 miss 不只是口径问题，更偏机制/参数导致的真实漏检。

================================================
V11-Track AE：no-drift 约束下的 PH 校准（降低误报密度）
================================================

产物：`scripts/TRACKAE_PH_CALIBRATION_WITH_NODRIFT.csv`

| error.threshold | error.min_instances | sea_miss | sea_conf_P90 | sine_miss | sine_conf_P90 | no_drift_rate | no_drift_MTFA | constraints_ok |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 5 | 0.000 | 206.7 | 0.000 | 234.5 | 30.160 | 330.3 | Y |
| 0.05 | 10 | 0.000 | 438.7 | 0.200 | 520.6 | 15.480 | 641.7 |  |
| 0.05 | 25 | 0.750 | 1479.0 | 0.750 | 1479.0 | 6.200 | 1600.0 |  |
| 0.08 | 5 | 0.000 | 278.7 | 0.000 | 273.9 | 30.040 | 331.7 | Y |
| 0.08 | 10 | 0.000 | 450.2 | 0.100 | 442.2 | 15.560 | 640.6 |  |
| 0.08 | 25 | 0.750 | 1479.0 | 0.750 | 1479.0 | 6.200 | 1600.0 |  |
| 0.10 | 5 | 0.000 | 235.5 | 0.000 | 231.6 | 29.440 | 339.3 | Y |
| 0.10 | 10 | 0.050 | 457.9 | 0.200 | 485.4 | 15.440 | 642.5 |  |
| 0.10 | 25 | 0.750 | 1479.0 | 0.750 | 1395.8 | 6.200 | 1600.0 |  |

**推荐 PH（写入默认配置）**
- error.threshold=0.10,error.min_instances=5
- 选择规则：规则：drift(sea+sine) miss_mean==0 且 conf_P90_mean<500；在满足下最小化 no-drift confirm_rate_per_10k（次选最大化 MTFA_win）；并要求 drift acc_final_mean 不低于 best-0.01。
- 备注：在 drift 约束（miss=0 且 conf_P90<500）下，仅 `min_instances=5` 可行；因此 no-drift 的 confirm_rate 降幅有限（主要依赖 error.threshold 微调）。

================================================
V11-Track AF：INSECTS 恢复收益的配对置信区间（不新跑）
================================================

输入：`scripts/TRACKAC_RECOVERY_WINDOW_SWEEP.csv`（seeds=40，baseline vs v2）

| window | metric | delta_mean | ci95 | n_seeds |
|---|---|---|---|---|
| W500 | Δpost_min(v2-baseline) | 0.0023 | [-0.0004,0.0049] | 40 |
| W500 | Δpost_mean(v2-baseline) | 0.0023 | [-0.0004,0.0049] | 40 |
| W1000 | Δpost_min(v2-baseline) | 0.0022 | [-0.0004,0.0049] | 40 |
| W1000 | Δpost_mean(v2-baseline) | 0.0023 | [-0.0004,0.0049] | 40 |
| W2000 | Δpost_min(v2-baseline) | 0.0022 | [-0.0005,0.0048] | 40 |
| W2000 | Δpost_mean(v2-baseline) | 0.0022 | [-0.0005,0.0048] | 40 |

**解读**
- 点估计为正但 95% CI 跨 0，说明 v2 的收益较小且在本轮设定下未达到统计显著（但方向一致）。

================================================
V11 后“一句话默认配置”
================================================

- 检测：`two_stage(candidate=OR,confirm=weighted)` + `error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5`，confirm_theta=0.50，confirm_window=1，confirm_cooldown=200；主口径 `acc_min@sample_idx>=2000`。
- 恢复：INSECTS 默认启用 `severity v2`（Track AF 给出 Δpost_min/Δpost_mean 的配对 CI：点估计为正但 CI 跨 0，收益较小）。 
