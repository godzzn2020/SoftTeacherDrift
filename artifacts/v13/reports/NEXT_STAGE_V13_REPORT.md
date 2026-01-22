# NEXT_STAGE V13 Report（Confirm 规则降低 no-drift 误报）

- 生成时间：2026-01-10 12:53:17
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`/home/ylh/anaconda3/envs/ZZNSTD/bin/python / 3.10.19`

## 结论摘要
- 在本轮 drift 约束（sea+sine：miss==0 且 confP90<500）下，仅 baseline 能同时满足；其余规则虽然可显著降 no-drift confirm_rate，但会导致 drift miss 或 confP90 超标。
- 观测：divergence_gate / k_of_n / error_gate 这类“更严格 confirm”在 gradual_frequent 上容易把确认推迟到 transition 后段，从而在 start 口径下计为 miss。

================================================
V13-Track AJ（必做）：Confirm-rule ablation for no-drift
================================================

产物：`scripts/TRACKAJ_CONFIRM_RULE_NODRIFT.csv`

| group | divgate_value_thr | errgate_thr | sea_miss | sea_confP90 | sine_miss | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|
| A_weighted | N/A | N/A | 0.000 | 263.6 | 0.000 | 233.2 | 29.880 | 342.3 | 0.7658 |
| B_k_of_n_k2 | N/A | N/A | 1.000 | 10000.0 | 1.000 | 10000.0 | 0.000 | N/A | 0.7680 |
| C_weighted_divgate_LstepsW_thr0.01 | 0.010 | N/A | 0.150 | 455.3 | 0.750 | 9812.6 | 23.760 | 434.1 | 0.7451 |
| D_weighted_divgate_Lsamples500_thr0.01 | 0.010 | N/A | 0.200 | 568.9 | 0.700 | 10000.0 | 25.160 | 407.9 | 0.7464 |
| E_weighted_errgate_thr0.167 | N/A | 0.167 | 0.600 | 1580.4 | 0.050 | 323.8 | 6.680 | 1648.0 | 0.7730 |
| E_weighted_errgate_thr0.2 | N/A | 0.200 | 0.700 | 3806.7 | 0.000 | 229.7 | 6.280 | 1614.8 | 0.7657 |
| E_weighted_errgate_thr0.25 | N/A | 0.250 | 0.700 | 2885.2 | 0.000 | 271.3 | 5.520 | 1786.6 | 0.7724 |

**赢家选择**
- 规则：规则：drift(sea+sine) miss==0 且 confP90<500；目标最小化 no-drift confirm_rate_per_10k（次选最大化 MTFA_win）；并要求 drift_acc_final≥best-0.01。
- winner：`A_weighted`
- no-drift confirm_rate_per_10k 下降幅度（vs baseline A_weighted）：29.880 → 29.880 (Δ=+0.000)

================================================
V13-Track AK（可选）：stagger_gradual 新信号最小验证（entropy）
================================================

- 未运行（可选项）

================================================
V13 后“一句话默认配置”
================================================

- 检测：`two_stage(candidate=OR,confirm=*)` + `error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5`（divergence 默认 0.05/30）。
- confirm 规则：`A_weighted`（详见 Track AJ）。
- 恢复：INSECTS 默认启用 `severity v2`（V11 Track AF：收益较小、CI 跨 0，但方向一致）。
