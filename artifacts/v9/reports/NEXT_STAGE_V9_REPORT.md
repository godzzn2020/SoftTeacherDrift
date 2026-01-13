# NEXT_STAGE V9 Report (gating 收敛 + detector×gating 联动验证)

- 生成时间：2026-01-09 00:46:06
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`/home/ylh/anaconda3/envs/ZZNSTD/bin/python / 3.10.19`

========================
V9-Track X：INSECTS gating 强度 sweep（在 V8 定稿检测下）
========================

| group | n_runs | acc_final | acc_min@2000 | post_mean@W1000 | post_min@W1000 | confirm_rate_per_10k |
|---|---|---|---|---|---|---|
| A_baseline | 40 | 0.2004±0.0075 | 0.1611±0.0072 | 0.1887±0.0063 | 0.1879±0.0063 | 7.375±0.184 |
| B_v2 | 40 | 0.2017±0.0088 | 0.1625±0.0100 | 0.1910±0.0081 | 0.1902±0.0081 | 7.366±0.184 |
| v2_gate_m1 | 40 | 0.2018±0.0095 | 0.1630±0.0084 | 0.1914±0.0091 | 0.1906±0.0091 | 7.347±0.218 |
| v2_gate_m3 | 40 | 0.2019±0.0081 | 0.1633±0.0076 | 0.1913±0.0072 | 0.1905±0.0072 | 7.328±0.201 |
| v2_gate_m5 | 40 | 0.2008±0.0078 | 0.1628±0.0080 | 0.1913±0.0084 | 0.1905±0.0084 | 7.375±0.179 |

**结论与赢家**
- 规则：acc_final_mean≥best-0.01，优先最大化 post_min@W1000_mean，再最大化 acc_min@2000_mean；best_acc=0.2019
- winner=`v2_gate_m1`：post_min@W1000=0.1906，acc_final=0.2018，confirm_rate/10k=7.347

**Δpost_min@W1000 的 bootstrap 95% CI（逐 seed 配对差）**
Δpost_min@W1000(v2_gate_m1-B_v2) = 0.0004 (95% CI [-0.0029, 0.0034], n=40)
| winner | ref | delta_mean | ci95 | n_seeds |
|---|---|---|---|---|
| v2_gate_m1 | B_v2 | 0.0004 | [-0.0029,0.0034] | 40 |
- 结论：本轮 `Δpost_min@W1000(gate-v2)` 的 95% CI 跨 0，优势不显著；默认配置不强制启用 gating。

========================
V9-Track Y：detector 敏感度 × gating 联动（机制验证）
========================

| detector | recovery | n_runs | acc_final | acc_min@2000 | post_min@W1000 | confirm_rate_per_10k |
|---|---|---|---|---|---|---|
| clean | v2 | 20 | 0.2006±0.0087 | 0.1647±0.0086 | 0.1901±0.0099 | 7.342±0.145 |
| clean | v2_gate_m1 | 20 | 0.2011±0.0095 | 0.1607±0.0071 | 0.1890±0.0076 | 7.332±0.229 |
| sensitive | v2 | 20 | 0.2027±0.0074 | 0.1653±0.0067 | 0.1921±0.0070 | 7.418±0.169 |
| sensitive | v2_gate_m1 | 20 | 0.2025±0.0098 | 0.1641±0.0086 | 0.1907±0.0081 | 7.408±0.154 |

**联动解释（用 confirm_rate 解释机制）**
- detector 设定：clean err_thr≈0.050，sensitive err_thr≈0.020；Δpost_min@W1000(gate-v2)：clean=-0.0011，sensitive=-0.0014（若 sensitive 更大则支持联动机制）；TrackX winner_m=1

## 回答三问（必须）
1) 在 V8 定稿检测条件下，gating 是否仍优于 v2？最优 m 是多少？
- TrackX 赢家：`v2_gate_m1`；是否优于 v2 以 `post_min@W1000` 及其 bootstrap CI 为准。
2) gating 的价值是否在 sensitive detector 下更明显？
- TrackY 对比 clean vs sensitive 下的 `Δpost_min(gate-v2)` 以及 confirm_rate 的上升。
3) 下一版最终默认配置（检测侧固定 + 恢复侧）
- 检测侧（固定）：`two_stage(candidate=OR,confirm=weighted)` + tuned PH（`error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5`），confirm_theta=0.50，confirm_window=1，confirm_cooldown=200。
- 恢复侧：`severity v2`（默认：v2（若 gating 的提升不显著则不强推复杂机制））。
