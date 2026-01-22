# NEXT_STAGE V15.3 Report FULL（Permutation-test Confirm + vote_score）

- 生成时间：2026-01-13 01:18:32
- 环境确认：
  - `which python` -> `/home/ylh/anaconda3/envs/ZZNSTD/bin/python`
  - `python -V` -> `Python 3.10.19`

## 关键口径与硬约束
- drift 约束：`sea_abrupt4` + `sine_abrupt4` 上 `miss_tol500_mean==0` 且 `conf_P90_mean<500`
- 目标：在满足约束下最小化 no-drift `confirm_rate_per_10k`（`sea_nodrift` + `sine_nodrift` 平均；次选最大化 no-drift `MTFA_win`）

## Track AL：Perm-confirm sweep
- 产物：`scripts/TRACKAL_PERM_CONFIRM_STABILITY_V15P3.csv`

| group | phase | confirm_rule | perm_stat | perm_alpha | perm_pre_n | perm_post_n | delta_k | sea_miss | sea_confP90 | sine_miss | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_weighted_n5 | stability | weighted | N/A | N/A | N/A | N/A | N/A | 0.000 | 242.7 | 0.000 | 242.0 | 26.780 | 371.9 | 0.7676 |
| P_perm_vote_score_a0.025_pre500_post10_n5 | stability | perm_test | vote_score | 0.025 | 500 | 10 | N/A | 0.000 | 267.0 | 0.000 | 241.7 | 26.520 | 365.7 | 0.7631 |
| P_perm_vote_score_a0.025_pre500_post10_n5_side2s | stability | perm_test | vote_score | 0.025 | 500 | 10 | N/A | 0.000 | 280.9 | 0.025 | 247.6 | 26.240 | 379.0 | 0.7586 |
| P_perm_vote_score_a0.02_pre500_post10_n5 | stability | perm_test | vote_score | 0.02 | 500 | 10 | N/A | 0.000 | 267.3 | 0.000 | 239.5 | 26.470 | 366.3 | 0.7647 |
| P_perm_vote_score_a0.02_pre500_post10_n5_side2s | stability | perm_test | vote_score | 0.02 | 500 | 10 | N/A | 0.000 | 262.7 | 0.000 | 222.4 | 26.450 | 368.3 | 0.7671 |
| P_perm_vote_score_a0.03_pre500_post10_n5 | stability | perm_test | vote_score | 0.03 | 500 | 10 | N/A | 0.000 | 258.2 | 0.000 | 179.2 | 26.210 | 374.5 | 0.7624 |
| P_perm_vote_score_a0.03_pre500_post10_n5_side2s | stability | perm_test | vote_score | 0.03 | 500 | 10 | N/A | 0.000 | 263.6 | 0.000 | 214.7 | 26.300 | 370.9 | 0.7624 |

**winner 选择规则（写死）**
- Step1: sea_abrupt4 & sine_abrupt4 满足 miss_tol500_mean==0 且 conf_P90_mean<500；Step2: 最小化 no-drift confirm_rate_per_10k（sea_nodrift+sine_nodrift 平均）；Step3: 并列时最大化 no-drift MTFA_win；Step4: drift_acc_final_mean 不低于 best-0.01。

**winner**
- `P_perm_vote_score_a0.03_pre500_post10_n5`
- no-drift confirm_rate_per_10k 下降幅度（vs baseline）：26.780 → 26.210 (Δ=-0.570)

## Track AM：机制诊断（可选）
- - winner：`P_perm_vote_score_a0.03_pre500_post10_n5`（见 Track AL）
- - no-drift confirm_rate_per_10k（平均）变化：26.780 → 26.210 (Δ=-0.570)
- - perm_test 最佳降误报候选（网格内）：`P_perm_vote_score_a0.03_pre500_post10_n5` no_drift_rate=26.210, sea(miss=0.000,confP90=258.2), sine(miss=0.000,confP90=179.2)
- - Track AM：已生成 `scripts/TRACKAM_PERM_DIAG_V15P3.csv` rows=10（含 stagger_abrupt3；p-value 分布/confirmed-candidate 比例见诊断表）

## 可复制运行命令
- `source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- `python experiments/trackAL_perm_confirm_sweep.py`
- `python experiments/trackAM_perm_diagnostics.py`  （可选）
- `python scripts/summarize_next_stage_v15.py`
