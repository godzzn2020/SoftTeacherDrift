# NEXT_STAGE V12 Report（confirm-side 降误报 + stagger gradual 补救）

- 生成时间：2026-01-10 00:55:20
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`/home/ylh/anaconda3/envs/ZZNSTD/bin/python / 3.10.19`

================================================
V12-Track AG（必做）：confirm-side sweep（no-drift 约束优化）
================================================

产物：`artifacts/tracks/TRACKAG_CONFIRM_SIDE_NODRIFT.csv`

| theta | window | cooldown | sea_miss | sea_confP90 | sine_miss | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 200 | 0.000 | 276.4 | 0.000 | 261.1 | 29.400 | 338.9 | 0.7538 |
| 0.50 | 1 | 200 | 0.000 | 289.9 | 0.000 | 241.6 | 29.520 | 336.8 | 0.7561 |
| 0.50 | 2 | 200 | 0.000 | 286.4 | 0.000 | 229.1 | 29.880 | 333.5 | 0.7569 |

**推荐 confirm-side（winner）**
- theta=0.50, window=3, cooldown=200
- 选择规则：规则：drift(sea+sine) miss==0 且 conf_P90<500；目标最小化 no-drift confirm_rate_per_10k（次选最大化 no-drift MTFA）；并要求 drift_acc_final_mean≥best-0.01。
- 误报变化（相对 baseline=theta0.50/w1/cd200）：no-drift confirm_rate_per_10k: 29.520 → 29.400 (Δ=-0.120)
- 结论：在本轮约束下（drift 实时性必须满足），confirm-side 对 no-drift 误报密度的下降幅度很小，未出现“显著下降”。

================================================
V12-Track AH（必做）：stagger_gradual_frequent 补救（专项小扫）
================================================

产物：`artifacts/tracks/TRACKAH_STAGGER_GRADUAL_SENSITIVITY.csv`

| error.threshold | n_runs | miss_start | miss_mid | miss_end | end_delay_P90 | end_delay_P99 |
|---|---|---|---|---|---|---|
| 0.02 | 5 | 0.520±0.110 | 0.480±0.110 | 0.440±0.089 | -984.0±1318.7 | -767.5±1617.6 |
| 0.03 | 5 | 0.440±0.089 | 0.360±0.089 | 0.360±0.089 | -1018.0±783.5 | -739.2±1072.6 |
| 0.05 | 5 | 0.560±0.089 | 0.480±0.110 | 0.440±0.089 | -530.3±1713.4 | -209.4±2103.4 |
| 0.08 | 5 | 0.600±0.000 | 0.520±0.110 | 0.520±0.110 | -721.0±2202.9 | -447.4±2698.2 |
| 0.10 | 5 | 0.560±0.089 | 0.520±0.110 | 0.520±0.110 | -625.6±1367.6 | -331.3±1672.6 |

**结论**
- end 口径下最优仍较高（best miss_end=0.360 @ thr=0.03），更像 stagger_gradual 的边界条件：gradual transition + 信号弱/迟。

================================================
V12 后“一句话默认配置”
================================================

- 检测：`two_stage(candidate=OR,confirm=weighted)` + `error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5`（divergence 默认 0.05/30）。
- confirm-side：theta=0.50, window=3, cooldown=200
- 恢复：INSECTS 默认启用 `severity v2`（V11 Track AF：收益较小、CI 跨 0，但方向一致）。
