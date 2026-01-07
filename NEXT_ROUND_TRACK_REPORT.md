# NEXT_ROUND Track Report

- 生成时间：2026-01-07 12:48:13
- logs_root：`logs`；results_root：`results`
- 本轮 run 索引：`NEXT_ROUND_RUN_INDEX.csv`

## Track A (NOAA)

### A-Table1：A1 单点诊断（seed=3）

| case | run_name | run_id | dataset | seed | model_variant | monitor_preset | severity_scale | acc_final | mean_acc | acc_min | drift_flag_count | first_drift_step | monitor_sev>0_ratio | drift_sev>0_ratio | collapse_step | lr_range | lambda_u_range | tau_range | alpha_range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1-1 | noaa_seed3_div_s2 | 20260107-122800-k6h_noaa_seed3_div_s2 | NOAA | 3 | ts_drift_adapt | error_divergence_ph_meta | 2.0 | 0.6831 | 0.6885 | 0.6211 | 2 | 25 | 0.014 | 0.000 | 1 | [0.0006,0.0006] | [0.5622,0.6300] | [0.8557,0.8775] | [0.9656,0.9801] |
| A1-1 | noaa_seed3_div_s2 | 20260107-122800-k6h_noaa_seed3_div_s2 | NOAA | 3 | ts_drift_adapt_severity | error_divergence_ph_meta | 2.0 | 0.6746 | 0.6571 | 0.4375 | 2 | 25 | 0.014 | 0.014 | 1 | [0.0006,0.0006] | [0.5622,0.6300] | [0.8557,0.8775] | [0.9656,0.9801] |
| A1-2 | noaa_seed3_err_s2 | 20260107-122829-x9i_noaa_seed3_err_s2 | NOAA | 3 | ts_drift_adapt | error_ph_meta | 2.0 | 0.6643 | 0.6227 | 0.3086 | 2 | 25 | 0.014 | 0.000 | 2 | [0.0006,0.0006] | [0.5611,0.6300] | [0.8554,0.8775] | [0.9654,0.9801] |
| A1-2 | noaa_seed3_err_s2 | 20260107-122829-x9i_noaa_seed3_err_s2 | NOAA | 3 | ts_drift_adapt_severity | error_ph_meta | 2.0 | 0.6773 | 0.6681 | 0.5176 | 2 | 25 | 0.014 | 0.014 | 2 | [0.0006,0.0006] | [0.5611,0.6300] | [0.8554,0.8775] | [0.9654,0.9801] |
| A1-3 | noaa_seed3_div_s0 | 20260107-122856-5ta_noaa_seed3_div_s0 | NOAA | 3 | ts_drift_adapt_severity | error_divergence_ph_meta | 0.0 | 0.6822 | 0.6839 | 0.6064 | 2 | 25 | 0.014 | 0.014 | 4 | [0.0006,0.0006] | [0.5603,0.6300] | [0.8551,0.8775] | [0.9652,0.9801] |

### A-Table2：A2 多 seed sweep（scale=0 vs 2）

| model_variant | scale | acc_final(seeds1-5) | mean±std | mean drift_flag_count | outlier_seeds(|acc-mean|>0.08) |
|---|---|---|---|---|---|
| ts_drift_adapt | 0.0 | [0.6825, 0.6838, 0.6704, 0.6744, 0.6670] | 0.6756±0.0074 | 2.00 | [] |
| ts_drift_adapt_severity | 0.0 | [0.6856, 0.6810, 0.6749, 0.6831, 0.6849] | 0.6819±0.0043 | 2.00 | [] |
| ts_drift_adapt | 2.0 | [0.6862, 0.6834, 0.6816, 0.6638, 0.6768] | 0.6784±0.0088 | 2.00 | [] |
| ts_drift_adapt_severity | 2.0 | [0.6833, 0.6811, 0.6863, 0.6658, 0.6685] | 0.6770±0.0092 | 2.00 | [] |

### A-结论（<=5 行）
- A1：最差窗口来自 A1-2[ts_drift_adapt]，`drift_flag_count=2`，检测事件确实发生。
- 崩塌点 step=2 明显早于首次 drift step=25，更像早期不稳定而非“漂移触发调度导致崩塌”。
- A2：NOAA `ts_drift_adapt_severity` scale=0.0→2.0 的 acc_final 均值变化 -0.0049（未见 >0.08 的离群 seed）。
- 结论：本轮 NOAA 未复现明显离群（A2 outlier_seeds 均为空），更像“轻微波动/早期不稳定”而非系统性崩塌。

## Track B (SEA preset ablation)

### B-Table：preset 消融（seeds=1/3/5，ts_drift_adapt）

| preset_group | monitor_preset | run_id | MDR(mean±var) | MTD(mean±var) | MTFA(mean±var) | MTR(mean±var) | acc_final(mean±std) | mean_acc(mean±std) |
|---|---|---|---|---|---|---|---|---|
| error_only | error_ph_meta | 20260107-122940-uk6_sea_ts_preset_err | 0.000±0.000 | 849.7±23125.3 | 1863.0±447.7 | 2.242±0.170 | 0.8673±0.0061 | 0.8222±0.0034 |
| divergence_only | divergence_ph_meta | 20260107-123012-fpz_sea_ts_preset_div | 0.833±0.021 | 559.0±2048.0 | N/A±N/A | N/A±N/A | 0.8508±0.0103 | 0.8080±0.0140 |
| error+divergence | error_divergence_ph_meta | 20260107-123047-4e4_sea_ts_preset_both | 0.000±0.000 | 844.3±36437.3 | 1853.4±2831.2 | 2.271±0.253 | 0.8741±0.0031 | 0.8259±0.0037 |

### B-结论（<=5 行）
- MDR 最低：error_ph_meta；（过滤 MDR>0.5 后）MTD 最小：error_divergence_ph_meta；MTFA 最大：error_ph_meta；MTR 最好：error_divergence_ph_meta。
- 分类（acc_final_mean）最高：error_divergence_ph_meta（与 MTR/MTD 的最优项对比见表）。
- 结论：`divergence_only` 明显漏检（MDR 高），`error_only` 与 `error+divergence` 的检测与分类整体更稳。

## Track C (INSECTS severity-aware sweep)

### C-Table：scale=0 vs 2（seeds=1..5）

| model_variant | scale | acc_final(seeds1-5) | mean±std(acc_final) | mean±std(acc_min) | drift_metrics(mean±std) | drift_flag_count(mean±std) | run_id |
|---|---|---|---|---|---|---|---|
| ts_drift_adapt | 0.0 | [0.2028, 0.1954, 0.2046, 0.1975, 0.1891] | 0.1979±0.0062 | 0.1615±0.0042 | MDR=0.200±0.000, MTD=5375.1±57.2, MTFA=14967.5±76.3, MTR=3.481±0.019 | 8.00±0.00 | 20260107-123135-g5y_insects_sweep_s0 |
| ts_drift_adapt_severity | 0.0 | [0.2037, 0.1894, 0.2098, 0.1944, 0.1920] | 0.1979±0.0086 | 0.1548±0.0278 | MDR=0.240±0.089, MTD=4695.9±1461.6, MTFA=14186.7±1669.6, MTR=4.585±2.450 | 8.00±0.00 | 20260107-123135-g5y_insects_sweep_s0 |
| ts_drift_adapt | 2.0 | [0.1908, 0.1827, 0.2017, 0.1852, 0.1895] | 0.1900±0.0073 | 0.1446±0.0091 | MDR=0.200±0.000, MTD=5349.5±0.0, MTFA=14933.3±0.0, MTR=3.489±0.000 | 8.00±0.00 | 20260107-123333-hk3_insects_sweep_s2 |
| ts_drift_adapt_severity | 2.0 | [0.2179, 0.1909, 0.1976, 0.1934, 0.1978] | 0.1995±0.0107 | 0.1550±0.0137 | MDR=0.240±0.089, MTD=4695.9±1461.6, MTFA=14199.5±1641.0, MTR=4.596±2.473 | 8.00±0.00 | 20260107-123333-hk3_insects_sweep_s2 |

### C-结论（<=8 行）
- INSECTS `ts_drift_adapt_severity`：scale=0.0→2.0 的 acc_final 均值变化 +0.0017；acc_min 均值变化 +0.0002。
- 检测触发：drift_flag_count_mean ≈ 8.00（本轮各配置几乎一致），不是“检测不触发”。
- 结论：本轮 scale=2 未呈现稳定、显著的分类提升（提升幅度接近随机波动），更像“触发了但调度幅度/方向收益有限”。

## 回答两点创新（结论版）
- 创新点1（monitor_preset 差异）：Track B 显示 `divergence_only` 漏检显著（MDR 高），而 `error_only` 与 `error+divergence` 的 MDR≈0 且分类精度更高/更稳；因此 MDR/MTD/MTFA/MTR 存在显著差异。
- 创新点2（severity-aware 自适应）：Track C/NOAA sweep 中 scale=2 相对 scale=0 未体现跨 seed 的稳定增益（acc_final/acc_min 提升幅度很小或为负），更像“检测触发一致，但调度增益不稳定”。
