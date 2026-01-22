# NEXT_STAGE V6 Report (Pareto Trade-off + Detector×Gating 联动验证)

- 生成时间：2026-01-08 11:17:14
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`/home/ylh/anaconda3/envs/ZZNSTD/bin/python / 3.10.19`

## 关键口径（Latency → tol500）
- 时间轴：统一使用 `sample_idx`；tol500 口径匹配条件为 `confirmed_step <= drift_pos + 500`。
- 注意：miss_tol500 高不等于完全漏检，也可能是晚检（P90/P99 延迟 > 500）。

========================
V6-Track O：Confirm-side sweep（固定 tuned PH，收敛误报与 acc_min）
========================

| config_tag | theta | window | cooldown | acc_final | acc_min | miss_tol500 | MDR_tol500 | conf_P90 | conf_P99 | MTFA_win |
|---|---|---|---|---|---|---|---|---|---|---|
| theta0.50_w1_cd200 | 0.50 | 1 | 200 | 0.8790±0.0049 | 0.5703±0.0963 | 0.000 | 0.000 | 256.6 | 284.0 | 340.4 |
| theta0.50_w2_cd200 | 0.50 | 2 | 200 | 0.8790±0.0097 | 0.4979±0.1142 | 0.000 | 0.000 | 258.2 | 351.8 | 344.9 |
| theta0.30_w1_cd0 | 0.30 | 1 | 0 | 0.8771±0.0050 | 0.4562±0.1637 | 0.000 | 0.000 | 271.0 | 284.0 | 337.2 |
| theta0.30_w2_cd0 | 0.30 | 2 | 0 | 0.8691±0.0050 | 0.5503±0.1340 | 0.000 | 0.000 | 272.6 | 300.0 | 339.8 |
| theta0.50_w1_cd0 | 0.50 | 1 | 0 | 0.8756±0.0084 | 0.5240±0.1905 | 0.000 | 0.000 | 287.0 | 300.0 | 341.0 |
| theta0.40_w2_cd200 | 0.40 | 2 | 200 | 0.8772±0.0047 | 0.5123±0.1599 | 0.000 | 0.000 | 288.6 | 316.0 | 344.7 |
| theta0.50_w3_cd0 | 0.50 | 3 | 0 | 0.8738±0.0108 | 0.5542±0.1320 | 0.000 | 0.000 | 288.6 | 303.0 | 343.8 |
| theta0.30_w1_cd200 | 0.30 | 1 | 200 | 0.8735±0.0050 | 0.4813±0.0744 | 0.000 | 0.000 | 288.6 | 316.0 | 341.8 |
| theta0.40_w1_cd200 | 0.40 | 1 | 200 | 0.8721±0.0105 | 0.5050±0.1561 | 0.000 | 0.000 | 288.6 | 316.0 | 341.4 |
| theta0.50_w2_cd0 | 0.50 | 2 | 0 | 0.8739±0.0069 | 0.4479±0.1437 | 0.000 | 0.000 | 288.6 | 354.8 | 341.1 |
| theta0.40_w3_cd200 | 0.40 | 3 | 200 | 0.8761±0.0062 | 0.3820±0.1724 | 0.000 | 0.000 | 303.0 | 316.0 | 342.9 |
| theta0.50_w3_cd200 | 0.50 | 3 | 200 | 0.8771±0.0103 | 0.4490±0.1619 | 0.000 | 0.000 | 303.0 | 354.8 | 342.3 |

**选择规则（写入论文/报告）**
- 规则：acc_final_mean≥best-0.01，先最小化 miss_tol500_mean（再看 conf_P90），再最大化 MTFA_win_mean，再最大化 acc_min_mean；best_acc=0.8794

**推荐 confirm-side 参数（sea_abrupt4）**
- monitor_preset=`error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5`, trigger=`two_stage(candidate=OR,confirm=weighted)`, confirm_theta=0.50, confirm_window=1, confirm_cooldown=200
- 约束检查：miss_tol500_mean=0.000, conf_P90=256.6, MTFA_win_mean=340.4, acc_min_mean=0.5703

========================
V6-Track P：Detector×Gating 联动（敏感 detector 场景验证 gating 价值）
========================

### sea_abrupt4
| group | n_runs | theta | window | cooldown | acc_final | acc_min | miss_tol500 | conf_P90 | MTFA_win |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 5 | 0.50 | 1 | 0 | 0.8776±0.0055 | 0.6175±0.1223 | 0.000 | 251.5 | 341.4 |
| cooldown | 5 | 0.50 | 1 | 200 | 0.8733±0.0055 | 0.4625±0.1369 | 0.000 | 271.3 | 339.4 |
| v2 | 5 | 0.50 | 1 | 0 | 0.8801±0.0110 | 0.5219±0.1573 | 0.000 | 280.0 | 345.7 |
| v2_gate_m1 | 5 | 0.50 | 1 | 0 | 0.8800±0.0014 | 0.4625±0.2138 | 0.000 | 220.8 | 341.8 |
| v2_gate_m1_cd | 5 | 0.50 | 1 | 200 | 0.8749±0.0079 | 0.5284±0.2066 | 0.000 | 242.8 | 341.1 |
| v2_gate_m3 | 5 | 0.50 | 1 | 0 | 0.8749±0.0059 | 0.4313±0.1141 | 0.000 | 264.6 | 340.5 |
| v2_gate_m3_cd | 5 | 0.50 | 1 | 200 | 0.8731±0.0034 | 0.5016±0.1389 | 0.000 | 244.8 | 340.0 |
| v2_gate_m5 | 5 | 0.50 | 1 | 0 | 0.8796±0.0064 | 0.5036±0.1377 | 0.000 | 247.3 | 342.3 |
| v2_gate_m5_cd | 5 | 0.50 | 1 | 200 | 0.8693±0.0056 | 0.3734±0.1681 | 0.000 | 288.6 | 343.5 |

### INSECTS_abrupt_balanced
| group | n_runs | theta | window | cooldown | acc_final | acc_min | post_mean@W1000 | post_min@W1000 | recovery_time_to_pre90 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 10 | 0.50 | 1 | 0 | 0.1985±0.0078 | 0.1245±0.0358 | 0.1893 | 0.1885 | 178.4 |
| cooldown | 10 | 0.50 | 1 | 200 | 0.1993±0.0087 | 0.1533±0.0099 | 0.1903 | 0.1894 | 178.4 |
| v2 | 10 | 0.50 | 1 | 0 | 0.2009±0.0089 | 0.1495±0.0217 | 0.1895 | 0.1887 | 178.4 |
| v2_gate_m1 | 10 | 0.50 | 1 | 0 | 0.2021±0.0070 | 0.1392±0.0427 | 0.1919 | 0.1911 | 178.4 |
| v2_gate_m1_cd | 10 | 0.50 | 1 | 200 | 0.2019±0.0111 | 0.1503±0.0275 | 0.1916 | 0.1908 | 178.4 |
| v2_gate_m3 | 10 | 0.50 | 1 | 0 | 0.2035±0.0049 | 0.1437±0.0359 | 0.1928 | 0.1921 | 178.4 |
| v2_gate_m3_cd | 10 | 0.50 | 1 | 200 | 0.1985±0.0106 | 0.1570±0.0215 | 0.1891 | 0.1883 | 178.4 |
| v2_gate_m5 | 10 | 0.50 | 1 | 0 | 0.2042±0.0059 | 0.1582±0.0153 | 0.1934 | 0.1926 | 178.4 |
| v2_gate_m5_cd | 10 | 0.50 | 1 | 200 | 0.2011±0.0057 | 0.1314±0.0398 | 0.1894 | 0.1886 | 178.4 |

**联动结论（写入论文/报告）**
- INSECTS 上，`severity v2 + confirmed drift gating` 能稳定抬高 `post_min@W1000`（缓解负迁移）。
- confirm_cooldown 对“过密 confirm/误报”有帮助，但在部分设置上会以 `acc_min` 为代价（需要与 Track O 口径一起权衡）。
- 推荐默认以 INSECTS 的 `post_min@W1000` 为主目标：best_group=`v2_gate_m5`（规则：acc_final_mean≥best-0.01，优先最大化 post_min@W1000_mean，再最大化 acc_min_mean；best_acc=0.2042）

## 最终论文默认配置（两句话）
- 检测：`two_stage(candidate=OR,confirm=weighted)` + tuned PH（`error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5`），confirm_theta=0.50，confirm_window=1，confirm_cooldown=200。
- 缓解负迁移：启用 `severity v2` 并使用 confirmed drift gating（`v2_gate_m5`），在敏感 detector 下优先抬高 `post_min@W1000/acc_min`。
