# NEXT_STAGE V10 Report（泛化 + 边界条件 + 恢复窗口稳健性）

- 生成时间：2026-01-09 01:57:40
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`/home/ylh/anaconda3/envs/ZZNSTD/bin/python / 3.10.19`

================================================
V10-Track AA（必做）：Gradual/Frequent drift 泛化（合成流）
================================================

产物：`scripts/TRACKAA_GENERALIZATION_NONABRUPT.csv`

说明：本仓库默认仅内置 abrupt 合成流；Track AA 先用 `data.streams.generate_and_save_synth_stream` 生成 non-abrupt（gradual）版本落盘到临时目录，再在运行期间“临时替换” `data/synthetic/<base_dataset_name>/`，跑完立即恢复原始数据。

| dataset | group | n_runs | acc_final | acc_min@2000 | miss_tol500 | MDR_tol500 | cand_P90 | conf_P90 | MTFA_win |
|---|---|---|---|---|---|---|---|---|---|
| sea_gradual_frequent | A_or_defaultPH | 5 | 0.8726±0.0074 | 0.7472±0.0136 | 0.800±0.209 | 0.800±0.209 | 1223.0±195.9 | 1223.0±195.9 | 1843.6±75.7 |
| sea_gradual_frequent | B_or_tunedPH | 5 | 0.8796±0.0042 | 0.7382±0.0277 | 0.000±0.000 | 0.000±0.000 | 247.0±53.6 | 247.0±53.6 | 337.5±6.0 |
| sea_gradual_frequent | C_weighted_tunedPH | 5 | 0.8776±0.0085 | 0.7265±0.0304 | 0.000±0.000 | 0.000±0.000 | 240.3±57.9 | 240.3±57.9 | 344.4±5.7 |
| sea_gradual_frequent | D_two_stage_tunedPH_cd200 | 5 | 0.8722±0.0015 | 0.7368±0.0210 | 0.000±0.000 | 0.000±0.000 | 249.9±39.5 | 249.9±39.5 | 339.6±3.3 |
| sine_gradual_frequent | A_or_defaultPH | 5 | 0.6440±0.0129 | 0.6215±0.0115 | 0.667±0.079 | 0.667±0.079 | 1474.6±387.2 | 1474.6±387.2 | 2315.0±103.2 |
| sine_gradual_frequent | B_or_tunedPH | 5 | 0.6587±0.0132 | 0.6402±0.0129 | 0.000±0.000 | 0.000±0.000 | 273.9±49.5 | 273.9±49.5 | 355.6±3.0 |
| sine_gradual_frequent | C_weighted_tunedPH | 5 | 0.6713±0.0242 | 0.6423±0.0250 | 0.000±0.000 | 0.000±0.000 | 264.0±42.3 | 264.0±42.3 | 356.3±4.5 |
| sine_gradual_frequent | D_two_stage_tunedPH_cd200 | 5 | 0.6573±0.0111 | 0.6351±0.0150 | 0.000±0.000 | 0.000±0.000 | 256.3±30.5 | 256.3±30.5 | 356.7±7.4 |
| stagger_gradual_frequent | A_or_defaultPH | 5 | 0.9305±0.0027 | 0.9133±0.0143 | 0.500±0.000 | 0.500±0.000 | 9147.4±1201.8 | 9147.4±1201.8 | 4066.7±571.2 |
| stagger_gradual_frequent | B_or_tunedPH | 5 | 0.9360±0.0048 | 0.8994±0.0473 | 0.400±0.137 | 0.400±0.137 | 7215.7±111.1 | 7215.7±111.1 | 925.2±99.0 |
| stagger_gradual_frequent | C_weighted_tunedPH | 5 | 0.9371±0.0055 | 0.9084±0.0180 | 0.450±0.112 | 0.450±0.112 | 9408.6±1322.5 | 9408.6±1322.5 | 993.5±104.0 |
| stagger_gradual_frequent | D_two_stage_tunedPH_cd200 | 5 | 0.9374±0.0041 | 0.8859±0.0353 | 0.450±0.112 | 0.450±0.112 | 8440.8±1456.8 | 8440.8±1456.8 | 965.2±84.3 |

**主方法组 D 约束检查（miss_tol500≈0 且 conf_P90<500）**
- Track AA 主方法组 D 约束通过：2/3（sea_gradual_frequent, sine_gradual_frequent）
  - sea_gradual_frequent: miss_mean=0.000, conf_P90_mean=249.9
  - sine_gradual_frequent: miss_mean=0.000, conf_P90_mean=256.3
  - stagger_gradual_frequent: miss_mean=0.450, conf_P90_mean=8440.8
- 未通过数据集（看 miss/conf_P90）：stagger_gradual_frequent
- 边界条件：gradual drift 的 `drift_start`→`tol500` 口径更苛刻，可能出现“检测在 transition 后段才触发”导致 conf_P90≫500。

================================================
V10-Track AB（必做）：No-drift sanity（误报成本）
================================================

产物：`scripts/TRACKAB_NODRIFT_SANITY.csv`

| dataset | group | n_runs | confirm_rate_per_10k | MTFA_win | acc_final | mean_acc |
|---|---|---|---|---|---|---|
| sea_nodrift | B_or_tunedPH | 5 | 30.480±0.303 | 327.8±3.1 | 0.8739±0.0077 | 0.8238±0.0093 |
| sea_nodrift | C_weighted_tunedPH | 5 | 30.080±0.415 | 331.6±4.8 | 0.8737±0.0057 | 0.8221±0.0107 |
| sea_nodrift | D_two_stage_tunedPH_cd200 | 5 | 30.240±0.219 | 330.5±2.1 | 0.8721±0.0099 | 0.8245±0.0138 |

**解读**
- Track AB 用 no-drift 合成流（positions/drifts 为空）评估误报：看 confirm_rate_per_10k 与 MTFA_win。
- 本轮 no-drift 观测：confirm_rate_per_10k≈[30.08,30.48]，MTFA_win≈[327.8,331.6]。

================================================
V10-Track AC（必做）：INSECTS 恢复窗口稳健性（baseline vs v2）
================================================

产物：`scripts/TRACKAC_RECOVERY_WINDOW_SWEEP.csv`（由 Track X 的 `log_path/run_id` 精确定位并重算）

| group | n_runs | acc_final | acc_min@2000 | post_min@W500 | post_min@W1000 | post_min@W2000 | post_mean@W500 | post_mean@W1000 | post_mean@W2000 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 40 | 0.2004±0.0075 | 0.1611±0.0072 | 0.1885±0.0063 | 0.1879±0.0063 | 0.1856±0.0062 | 0.1889±0.0063 | 0.1887±0.0063 | 0.1879±0.0062 |
| v2 | 40 | 0.2017±0.0088 | 0.1625±0.0100 | 0.1908±0.0081 | 0.1902±0.0081 | 0.1878±0.0080 | 0.1911±0.0081 | 0.1910±0.0081 | 0.1901±0.0080 |

**解读**
- Track AC 用同一批 seeds(1..40) 对比 baseline vs v2，并在 W=500/1000/2000 上重算 post_min/post_mean。
- Δpost_min（v2 - baseline，逐 seed 配对均值）：
  - W500: 0.0023
  - W1000: 0.0022
  - W2000: 0.0022

**配对差（v2 - baseline，逐 seed 均值）**
- Δpost_min@W500 = 0.0023
- Δpost_min@W1000 = 0.0022
- Δpost_min@W2000 = 0.0022

================================================
回答三问（必须）
================================================

1) non-abrupt drift 下：看 Track AA 主方法组 D 的 `miss_tol500` 与 `conf_P90`（上面已给约束检查）。
2) no-drift 情况下误报：看 Track AB 的 `confirm_rate_per_10k` 与 `MTFA_win`（并同时核对 `acc_final/mean_acc` 波动）。
3) v2 的恢复收益跨窗口一致性：看 Track AC 的 `post_min@W500/W1000/W2000`（baseline vs v2）。

## V10 后最终默认配置（一句话）
- 检测：`two_stage(candidate=OR,confirm=weighted)` + tuned PH（`error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5`），confirm_theta=0.50，confirm_window=1，confirm_cooldown=200；主口径 `acc_min@sample_idx>=2000`。
- 恢复：INSECTS 上默认启用 `severity v2`（不强制 gating）。
