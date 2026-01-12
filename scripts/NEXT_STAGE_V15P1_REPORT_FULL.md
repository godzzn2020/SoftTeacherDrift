# NEXT_STAGE V15.1 Report FULL（Permutation-test Confirm + vote_score）

- 生成时间：2026-01-12 20:25:46
- 环境确认：
  - `which python` -> `/home/ylh/anaconda3/envs/ZZNSTD/bin/python`
  - `python -V` -> `Python 3.10.19`

## 关键口径与硬约束
- drift 约束：`sea_abrupt4` + `sine_abrupt4` 上 `miss_tol500_mean==0` 且 `conf_P90_mean<500`
- 目标：在满足约束下最小化 no-drift `confirm_rate_per_10k`（`sea_nodrift` + `sine_nodrift` 平均；次选最大化 no-drift `MTFA_win`）

## Track AL：Perm-confirm sweep
- 产物：`scripts/TRACKAL_PERM_CONFIRM_SWEEP_V15P1.csv`

| group | phase | confirm_rule | perm_stat | perm_alpha | perm_pre_n | perm_post_n | delta_k | sea_miss | sea_confP90 | sine_miss | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_weighted_n5 | quick | weighted | N/A | N/A | N/A | N/A | N/A | 0.000 | 224.0 | 0.000 | 265.9 | 27.280 | 366.7 | 0.7542 |
| P_perm_vote_score_a0.015_pre500_post10_n5 | quick | perm_test | vote_score | 0.015 | 500 | 10 | N/A | 0.000 | 272.6 | 0.000 | 279.6 | 25.660 | 383.6 | 0.7575 |
| P_perm_vote_score_a0.015_pre500_post20_n5 | quick | perm_test | vote_score | 0.015 | 500 | 20 | N/A | 0.000 | 239.0 | 0.000 | 242.5 | 25.900 | 381.5 | 0.7545 |
| P_perm_vote_score_a0.015_pre500_post30_n5 | quick | perm_test | vote_score | 0.015 | 500 | 30 | N/A | 0.000 | 245.7 | 0.000 | 235.2 | 26.080 | 375.8 | 0.7640 |
| P_perm_vote_score_a0.01_pre500_post10_n5 | quick | perm_test | vote_score | 0.01 | 500 | 10 | N/A | 0.000 | 290.8 | 0.000 | 228.8 | 25.860 | 377.0 | 0.7637 |
| P_perm_vote_score_a0.01_pre500_post20_n5 | quick | perm_test | vote_score | 0.01 | 500 | 20 | N/A | 0.000 | 280.0 | 0.000 | 246.0 | 25.900 | 377.2 | 0.7676 |
| P_perm_vote_score_a0.01_pre500_post30_n5 | quick | perm_test | vote_score | 0.01 | 500 | 30 | N/A | 0.000 | 238.7 | 0.000 | 250.5 | 25.980 | 378.1 | 0.7614 |
| P_perm_vote_score_a0.025_pre500_post10_n5 | quick | perm_test | vote_score | 0.025 | 500 | 10 | N/A | 0.000 | 279.0 | 0.000 | 202.2 | 25.140 | 393.3 | 0.7619 |
| P_perm_vote_score_a0.025_pre500_post20_n5 | quick | perm_test | vote_score | 0.025 | 500 | 20 | N/A | 0.000 | 276.8 | 0.000 | 218.8 | 25.980 | 370.7 | 0.7632 |
| P_perm_vote_score_a0.025_pre500_post30_n5 | quick | perm_test | vote_score | 0.025 | 500 | 30 | N/A | 0.000 | 245.1 | 0.000 | 228.4 | 26.560 | 368.7 | 0.7644 |
| P_perm_vote_score_a0.02_pre500_post10_n5 | quick | perm_test | vote_score | 0.02 | 500 | 10 | N/A | 0.000 | 236.1 | 0.000 | 234.5 | 25.060 | 391.4 | 0.7582 |
| P_perm_vote_score_a0.02_pre500_post20_n5 | quick | perm_test | vote_score | 0.02 | 500 | 20 | N/A | 0.000 | 276.8 | 0.000 | 213.7 | 26.940 | 360.4 | 0.7671 |
| P_perm_vote_score_a0.02_pre500_post30_n5 | quick | perm_test | vote_score | 0.02 | 500 | 30 | N/A | 0.000 | 266.2 | 0.000 | 271.0 | 26.580 | 363.4 | 0.7615 |
| P_perm_vote_score_a0.03_pre500_post10_n5 | quick | perm_test | vote_score | 0.03 | 500 | 10 | N/A | 0.000 | 261.4 | 0.000 | 218.5 | 26.180 | 365.8 | 0.7687 |
| P_perm_vote_score_a0.03_pre500_post20_n5 | quick | perm_test | vote_score | 0.03 | 500 | 20 | N/A | 0.000 | 265.6 | 0.000 | 195.2 | 26.360 | 364.1 | 0.7591 |
| P_perm_vote_score_a0.03_pre500_post30_n5 | quick | perm_test | vote_score | 0.03 | 500 | 30 | N/A | 0.000 | 288.0 | 0.000 | 205.7 | 26.500 | 362.7 | 0.7620 |

**winner 选择规则（写死）**
- Step1: sea_abrupt4 & sine_abrupt4 满足 miss_tol500_mean==0 且 conf_P90_mean<500；Step2: 最小化 no-drift confirm_rate_per_10k（sea_nodrift+sine_nodrift 平均）；Step3: 并列时最大化 no-drift MTFA_win；Step4: drift_acc_final_mean 不低于 best-0.01。

**winner**
- `P_perm_vote_score_a0.025_pre500_post10_n5`
- no-drift confirm_rate_per_10k 下降幅度（vs baseline）：27.280 → 25.140 (Δ=-2.140)

## Track AM：机制诊断（可选）
- - winner：`P_perm_vote_score_a0.025_pre500_post10_n5`（见 Track AL）
- - no-drift confirm_rate_per_10k（平均）变化：27.280 → 25.140 (Δ=-2.140)
- - perm_test 最佳降误报候选（网格内）：`P_perm_vote_score_a0.02_pre500_post10_n5` no_drift_rate=25.060, sea(miss=0.000,confP90=236.1), sine(miss=0.000,confP90=234.5)
- - Track AM：已生成 `scripts/TRACKAM_PERM_DIAG_V15P1.csv` rows=8（p-value 分布/confirmed-candidate 比例见诊断表）

## 可复制运行命令
- `source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- `python experiments/trackAL_perm_confirm_sweep.py`
- `python experiments/trackAM_perm_diagnostics.py`  （可选）
- `python scripts/summarize_next_stage_v15.py`
