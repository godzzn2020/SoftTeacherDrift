# NEXT_STAGE V7 Report (统一口径 + cooldown 机制解释 + 自适应 cooldown)

- 生成时间：2026-01-08 19:29:50
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`/home/ylh/anaconda3/envs/ZZNSTD/bin/python / 3.10.19`

## 结论摘要
- 统一口径：sea 的“谷底”最终采用 `acc_min@sample_idx>=2000`；`acc_min_raw` 仅作参考。
- Track Q 产物：`scripts/TRACKQ_METRIC_AUDIT.md`
- Track R 产物：`scripts/TRACKR_CONFIRM_DENSITY.csv`

========================
V7-Track Q：口径一致性审计
========================

- 关键现象：V6 sea 在 Track O vs Track P 的 `acc_min_raw` 会因早期瞬时下探而出现 run 间差异；改用 warmup 后差异收敛。
- 量化（同参对齐 baseline）：mean Δraw=0.093542，mean Δwarm=-0.006209

### V6 表重算（统一口径）
- sea 的谷底统一使用 `acc_min@sample_idx>=2000`。

**V6-Track O（sea_abrupt4）重算**
| config_tag | theta | window | cooldown | acc_final | acc_min_raw | acc_min@2000 | miss_tol500 | conf_P90 | MTFA_win |
|---|---|---|---|---|---|---|---|---|---|
| theta0.50_w1_cd200 | 0.50 | 1 | 200 | 0.8790±0.0049 | 0.5703±0.0963 | 0.7558±0.0038 | 0.000 | 256.6 | 340.4±2.3 |
| theta0.50_w2_cd200 | 0.50 | 2 | 200 | 0.8790±0.0097 | 0.4979±0.1142 | 0.7338±0.0215 | 0.000 | 258.2 | 344.9±9.2 |
| theta0.30_w1_cd0 | 0.30 | 1 | 0 | 0.8771±0.0050 | 0.4562±0.1637 | 0.7450±0.0308 | 0.000 | 271.0 | 337.2±0.5 |
| theta0.30_w2_cd0 | 0.30 | 2 | 0 | 0.8691±0.0050 | 0.5503±0.1340 | 0.7338±0.0181 | 0.000 | 272.6 | 339.8±1.8 |
| theta0.50_w1_cd0 | 0.50 | 1 | 0 | 0.8756±0.0084 | 0.5240±0.1905 | 0.7497±0.0168 | 0.000 | 287.0 | 341.0±5.9 |
| theta0.40_w2_cd200 | 0.40 | 2 | 200 | 0.8772±0.0047 | 0.5123±0.1599 | 0.7396±0.0116 | 0.000 | 288.6 | 344.7±6.9 |
| theta0.50_w3_cd0 | 0.50 | 3 | 0 | 0.8738±0.0108 | 0.5542±0.1320 | 0.7262±0.0167 | 0.000 | 288.6 | 343.8±8.3 |
| theta0.30_w1_cd200 | 0.30 | 1 | 200 | 0.8735±0.0050 | 0.4813±0.0744 | 0.7457±0.0137 | 0.000 | 288.6 | 341.8±3.3 |
| theta0.40_w1_cd200 | 0.40 | 1 | 200 | 0.8721±0.0105 | 0.5050±0.1561 | 0.7290±0.0304 | 0.000 | 288.6 | 341.4±2.4 |
| theta0.50_w2_cd0 | 0.50 | 2 | 0 | 0.8739±0.0069 | 0.4479±0.1437 | 0.7425±0.0259 | 0.000 | 288.6 | 341.1±5.4 |
| theta0.40_w3_cd200 | 0.40 | 3 | 200 | 0.8761±0.0062 | 0.3820±0.1724 | 0.7054±0.0499 | 0.000 | 303.0 | 342.9±3.0 |
| theta0.50_w3_cd200 | 0.50 | 3 | 200 | 0.8771±0.0103 | 0.4490±0.1619 | 0.7229±0.0424 | 0.000 | 303.0 | 342.3±7.1 |

**Track O 推荐点（按统一口径重选）**
- 规则：acc_final_mean≥best-0.01，先最小化 miss_tol500_mean（再看 conf_P90），再最大化 MTFA_win_mean，再最大化 acc_min@2000_mean；best_acc=0.8794
- best=`theta0.50_w1_cd200`：theta=0.50 window=1 cooldown=200 miss=0.000 conf_P90=256.6 MTFA=340.4 acc_min@2000=0.7558

**V6-Track P 重算（sea_abrupt4）**
| group | n_runs | theta | window | cooldown | acc_final | acc_min_raw | acc_min@2000 | miss_tol500 | conf_P90 | MTFA_win |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 5 | 0.50 | 1 | 0 | 0.8776±0.0055 | 0.6175±0.1223 | 0.7435±0.0154 | 0.000 | 251.5 | 341.4 |
| cooldown | 5 | 0.50 | 1 | 200 | 0.8733±0.0055 | 0.4625±0.1369 | 0.7302±0.0205 | 0.000 | 271.3 | 339.4 |
| v2 | 5 | 0.50 | 1 | 0 | 0.8801±0.0110 | 0.5219±0.1573 | 0.7456±0.0318 | 0.000 | 280.0 | 345.7 |
| v2_gate_m1 | 5 | 0.50 | 1 | 0 | 0.8800±0.0014 | 0.4625±0.2138 | 0.7427±0.0262 | 0.000 | 220.8 | 341.8 |
| v2_gate_m1_cd | 5 | 0.50 | 1 | 200 | 0.8749±0.0079 | 0.5284±0.2066 | 0.7382±0.0354 | 0.000 | 242.8 | 341.1 |
| v2_gate_m3 | 5 | 0.50 | 1 | 0 | 0.8749±0.0059 | 0.4313±0.1141 | 0.7379±0.0170 | 0.000 | 264.6 | 340.5 |
| v2_gate_m3_cd | 5 | 0.50 | 1 | 200 | 0.8731±0.0034 | 0.5016±0.1389 | 0.7429±0.0292 | 0.000 | 244.8 | 340.0 |
| v2_gate_m5 | 5 | 0.50 | 1 | 0 | 0.8796±0.0064 | 0.5036±0.1377 | 0.7468±0.0127 | 0.000 | 247.3 | 342.3 |
| v2_gate_m5_cd | 5 | 0.50 | 1 | 200 | 0.8693±0.0056 | 0.3734±0.1681 | 0.7010±0.0334 | 0.000 | 288.6 | 343.5 |

**V6-Track P 重算（INSECTS_abrupt_balanced）**
| group | n_runs | theta | window | cooldown | acc_final | acc_min_raw | acc_min@2000 | post_min@W1000 |
|---|---|---|---|---|---|---|---|---|
| baseline | 10 | 0.50 | 1 | 0 | 0.1985±0.0078 | 0.1245±0.0358 | 0.1612±0.0108 | 0.1885±0.0080 |
| cooldown | 10 | 0.50 | 1 | 200 | 0.1993±0.0087 | 0.1533±0.0099 | 0.1627±0.0054 | 0.1894±0.0056 |
| v2 | 10 | 0.50 | 1 | 0 | 0.2009±0.0089 | 0.1495±0.0217 | 0.1598±0.0067 | 0.1887±0.0055 |
| v2_gate_m1 | 10 | 0.50 | 1 | 0 | 0.2021±0.0070 | 0.1392±0.0427 | 0.1623±0.0068 | 0.1911±0.0078 |
| v2_gate_m1_cd | 10 | 0.50 | 1 | 200 | 0.2019±0.0111 | 0.1503±0.0275 | 0.1632±0.0093 | 0.1908±0.0086 |
| v2_gate_m3 | 10 | 0.50 | 1 | 0 | 0.2035±0.0049 | 0.1437±0.0359 | 0.1651±0.0087 | 0.1921±0.0064 |
| v2_gate_m3_cd | 10 | 0.50 | 1 | 200 | 0.1985±0.0106 | 0.1570±0.0215 | 0.1633±0.0061 | 0.1883±0.0080 |
| v2_gate_m5 | 10 | 0.50 | 1 | 0 | 0.2042±0.0059 | 0.1582±0.0153 | 0.1652±0.0051 | 0.1926±0.0056 |
| v2_gate_m5_cd | 10 | 0.50 | 1 | 200 | 0.2011±0.0057 | 0.1314±0.0398 | 0.1584±0.0044 | 0.1886±0.0043 |

**Track P 推荐组（按统一口径复核）**
- best_group=`v2_gate_m5`（规则：acc_final_mean≥best-0.01，优先最大化 post_min@W1000_mean；best_acc=0.2042）

### cooldown 机制解释（写清楚）
- fixed cooldown：距离上次 confirmed 小于 `confirm_cooldown` 时，禁止新 confirm，并清空 two_stage 的 pending（避免“过期后补确认”的晚检）。
- adaptive cooldown：在最近窗口 `adaptive_window` 内统计 confirmed 数量，换算 `confirm_rate_per_10k`；若 >upper 切到高 cooldown（默认 500），若 <lower 切到低 cooldown（默认 200）。

========================
V7-Track R：触发密度诊断（V6 runs）
========================

相关性（run 粒度）：
| dataset | y | x | pearson_r | spearman_r |
|---|---|---|---|---|
| sea_abrupt4 | acc_min@2000 | confirm_rate_per_10k | 0.0017 | -0.0037 |
| sea_abrupt4 | acc_min@2000 | median_gap_between_confirms | N/A | N/A |
| sea_abrupt4 | acc_min@2000 | p10_gap_between_confirms | N/A | N/A |
| INSECTS_abrupt_balanced | post_min@W1000 | confirm_rate_per_10k | 0.2730 | 0.2897 |
| INSECTS_abrupt_balanced | post_min@W1000 | median_gap_between_confirms | N/A | N/A |
| INSECTS_abrupt_balanced | post_min@W1000 | p10_gap_between_confirms | N/A | N/A |

========================
V7-Track S：自适应 cooldown
========================

### sea_abrupt4
| group | n_runs | acc_final | acc_min_raw | acc_min_warmup | miss_tol500 | conf_P90 | MTFA_win | post_min@W1000 |
|---|---|---|---|---|---|---|---|---|
| adaptive_cd | 5 | 0.8752±0.0046 | 0.3906±0.0574 | 0.7448±0.0133 | 0.100 | 448.6 | 588.7 | N/A |
| fixed_cd0 | 5 | 0.8730±0.0090 | 0.5094±0.1489 | 0.7410±0.0301 | 0.000 | 262.0 | 339.0 | N/A |
| fixed_cd200 | 5 | 0.8741±0.0069 | 0.5250±0.1460 | 0.7390±0.0242 | 0.000 | 250.5 | 341.9 | N/A |

### INSECTS_abrupt_balanced
| group | n_runs | acc_final | acc_min_raw | acc_min_warmup | miss_tol500 | conf_P90 | MTFA_win | post_min@W1000 |
|---|---|---|---|---|---|---|---|---|
| adaptive_cd | 10 | 0.2036±0.0085 | 0.1552±0.0210 | 0.1627±0.0083 | N/A | N/A | N/A | 0.1920 |
| fixed_cd0 | 10 | 0.2030±0.0089 | 0.1396±0.0313 | 0.1652±0.0075 | N/A | N/A | N/A | 0.1915 |
| fixed_cd200 | 10 | 0.2059±0.0073 | 0.1462±0.0233 | 0.1635±0.0075 | N/A | N/A | N/A | 0.1935 |

## 回答三问（必须）
1) acc_min 不一致根因：见 `scripts/TRACKQ_METRIC_AUDIT.md`，核心是 `acc_min_raw` 包含 warmup 段且对瞬时下探敏感；最终口径采用 `acc_min@sample_idx>=2000`。
2) cooldown 是否通过确认密度影响谷底/恢复：见 `scripts/TRACKR_CONFIRM_DENSITY.csv` 与上表相关性。
3) adaptive cooldown 是否兼顾 tol500 与谷底：对比 Track S 的 miss/conf_P90 与 acc_min_warmup/post_min@W1000。
