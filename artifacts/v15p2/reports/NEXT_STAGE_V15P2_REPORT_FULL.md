# NEXT_STAGE V15.1 Report FULL（Permutation-test Confirm + vote_score）

- 生成时间：2026-01-12 23:58:39
- 环境确认：
  - `which python` -> `/home/ylh/anaconda3/envs/ZZNSTD/bin/python`
  - `python -V` -> `Python 3.10.19`

## 关键口径与硬约束
- drift 约束：`sea_abrupt4` + `sine_abrupt4` 上 `miss_tol500_mean==0` 且 `conf_P90_mean<500`
- 目标：在满足约束下最小化 no-drift `confirm_rate_per_10k`（`sea_nodrift` + `sine_nodrift` 平均；次选最大化 no-drift `MTFA_win`）

## Track AL：Perm-confirm sweep
- 产物：`artifacts/v15p2/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P2.csv`

| group | phase | confirm_rule | perm_stat | perm_alpha | perm_pre_n | perm_post_n | delta_k | sea_miss | sea_confP90 | sine_miss | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_weighted_n5 | stability | weighted | N/A | N/A | N/A | N/A | N/A | 0.000 | 279.5 | 0.000 | 226.8 | 27.050 | 370.1 | 0.7621 |
| P_perm_vote_score_a0.025_pre500_post10_n5 | stability | perm_test | vote_score | 0.025 | 500 | 10 | N/A | 0.000 | 285.7 | 0.000 | 226.7 | 26.840 | 364.1 | 0.7641 |
| P_perm_vote_score_a0.02_pre500_post10_n5 | stability | perm_test | vote_score | 0.02 | 500 | 10 | N/A | 0.000 | 257.4 | 0.000 | 228.6 | 26.340 | 371.4 | 0.7624 |
| P_perm_vote_score_a0.03_pre500_post10_n5 | stability | perm_test | vote_score | 0.03 | 500 | 10 | N/A | 0.000 | 234.4 | 0.000 | 228.1 | 26.570 | 368.9 | 0.7640 |

**winner 选择规则（写死）**
- Step1: sea_abrupt4 & sine_abrupt4 满足 miss_tol500_mean==0 且 conf_P90_mean<500；Step2: 最小化 no-drift confirm_rate_per_10k（sea_nodrift+sine_nodrift 平均）；Step3: 并列时最大化 no-drift MTFA_win；Step4: drift_acc_final_mean 不低于 best-0.01。

**winner**
- `P_perm_vote_score_a0.02_pre500_post10_n5`
- no-drift confirm_rate_per_10k 下降幅度（vs baseline）：27.050 → 26.340 (Δ=-0.710)

## Track AM：机制诊断（可选）
- - winner：`P_perm_vote_score_a0.02_pre500_post10_n5`（见 Track AL）
- - no-drift confirm_rate_per_10k（平均）变化：27.050 → 26.340 (Δ=-0.710)
- - perm_test 最佳降误报候选（网格内）：`P_perm_vote_score_a0.02_pre500_post10_n5` no_drift_rate=26.340, sea(miss=0.000,confP90=257.4), sine(miss=0.000,confP90=228.6)
- - Track AM：已生成 `artifacts/v15p2/tables/TRACKAM_PERM_DIAG_V15P2.csv` rows=8（p-value 分布/confirmed-candidate 比例见诊断表）

## 可复制运行命令
- `source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- `python experiments/trackAL_perm_confirm_sweep.py`
- `python experiments/trackAM_perm_diagnostics.py`  （可选）
- `python scripts/summarize_next_stage_v15.py`
