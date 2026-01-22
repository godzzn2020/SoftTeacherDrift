# V14 审计（Permutation-test Confirm）V4

- 生成时间：2026-01-11 17:10:47
- 复现命令：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD && python run_v14_audit_perm_confirm_v4.py`

## 0) 审计范围声明（强约束）
- 未进行任何全局搜索/递归扫描（未使用 find/rg/grep -R/os.walk/glob('**')）。
- 逐 run 仅使用 `scripts/NEXT_STAGE_V14_RUN_INDEX.csv` 的 `log_path` 定位；每个 run 的目录：1 次 `listdir` + summary 定位最多 3 次固定路径 +（必要时）1 次局部 `*.summary.json` glob；jsonl 选择仅基于该次 listdir（不额外 glob）。
- 未重跑训练/实验（不生成新 runs）。

## 1) Task A：复核（写入 V4）
- Step1 可行组数量：1
- best_acc_final（Step1 可行组内最大 drift_acc_final）：0.765979
- winner：`A_weighted_n5`
- Top1 near-constraints：`P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5`（V3 记录：`P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5`）
- no-drift 最低 perm_test：`P_perm_fused_score_a0.005_pre500_post30_n5`（V3 记录：`P_perm_fused_score_a0.005_pre500_post30_n5`）
- 表格：`V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv` / `AL_feasible_groups`、`AL_top10_near_constraints`

## 2) Task B：逐 run 深挖（summary + 目录列表 + jsonl 片段）
- 逐 run 量化表：`V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv` / `RUN_summary_metrics`
- 逐 run drill-down 记录：`V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv` / `RUN_drilldown_extract_v4`
- 逐 run 强校验异常：`V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv` / `RUN_summary_anomalies`

本轮选 run 规则（写死）：对每个 group（winner / Top1 near / no-drift 最低），在 `sea_abrupt4` 与 `sea_nodrift` 各取 seed 最小的 2 个（不足则取前 2 行）。
- 实际选中 run 数：12（注意：不对 log_path 去重；若同一 log_path 被不同 dataset 标注复用，会作为口径异常进入表格）

### 2.1 关键口径异常（可复核）
- 异常：`run_index.dataset` 与 `summary.dataset_name` 不一致（会直接影响“drift vs no-drift”的逐 run 对齐与解释）。
- 统计：本轮选中 run 中，该异常条数 = 6（见表 `RUN_summary_anomalies`）。
- 示例：`RUN_summary_metrics` 里 `dataset=sea_nodrift` 的行，`summary_dataset_name=sea_abrupt4`，且 `summary_path` 位于 `.../sea_abrupt4__/...summary.json`。

## 3) Q1/Q2/Q3：用逐 run 量化证据钉死归因（B/A/C）
### 3.1 证据→归因对照表（可复核）
- 表：`V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv` / `ATTRIBUTION_EVIDENCE`

### 3.2 最关键的实现/对齐问题点（非泛泛而谈）
- obs<=0 -> p=1.0（导致 p=1.0 质量点堆积的确定性机制）：`drift/detectors.py`
```py
0611:     def _perm_test_one_sided(self, pre_seq: List[float], post_seq: List[float], *, n_perm: int, seed: int) -> Tuple[float, float]:
0612:         pre = np.asarray(list(pre_seq), dtype=np.float64)
0613:         post = np.asarray(list(post_seq), dtype=np.float64)
0614:         if pre.size <= 0 or post.size <= 0:
0615:             return float("nan"), float("nan")
0616:         obs = float(post.mean() - pre.mean())
0617:         if not (obs > 0.0):
0618:             return 1.0, float(obs)
0619:         allv = np.concatenate([pre, post], axis=0)
0620:         n_pre = int(pre.size)
0621:         rng = np.random.default_rng(int(seed))
0622:         ge = 0
0623:         for _ in range(int(n_perm)):
0624:             idx = rng.permutation(allv.size)
0625:             pre_p = allv[idx[:n_pre]]
0626:             post_p = allv[idx[n_pre:]]
0627:             perm_obs = float(post_p.mean() - pre_p.mean())
0628:             if perm_obs >= obs:
0629:                 ge += 1
0630:         p = (1.0 + float(ge)) / (1.0 + float(int(n_perm)))
0631:         return float(p), float(obs)
```
- pre-window 去污染 + pending 生命周期用 step，但 pre/post 用 sample_idx（错配根源）：`drift/detectors.py`
```py
0780:                 # clear expired pending
0781:                 if self._pending_confirm_deadline_step is not None and step > self._pending_confirm_deadline_step:
0782:                     self._pending_candidate_step = None
0783:                     self._pending_confirm_deadline_step = None
0784:                     self._pending_confirm_hits = 0
0785:                 # register candidate
0786:                 if candidate_flag and self._pending_candidate_step is None:
0787:                     self._pending_candidate_step = step
0788:                     self._pending_confirm_deadline_step = step + confirm_window
0789:                     self._pending_confirm_hits = 0
0790:                     self._pending_confirm_rule_name = str(confirm_rule_name or "weighted").lower()
0791:                     if str(self._pending_confirm_rule_name) == "perm_test":
0792:                         perm_cfg_now = self._resolve_perm_test_cfg()
0793:                         pre_n = int(perm_cfg_now.get("pre_n") or 200)
0794:                         post_n = int(perm_cfg_now.get("post_n") or 50)
0795:                         pre_hist = list(self._perm_stat_hist)
0796:                         # 预窗口去污染：candidate 往往触发在“已接近/刚跨过阈值”的阶段，最近一小段可能已被 drift 污染。
0797:                         # 默认跳过最近 post_n 个样本（若不足则退化为直接取最后 pre_n）。
0798:                         if pre_hist and len(pre_hist) >= (pre_n + post_n):
0799:                             self._perm_pre_seq = pre_hist[-(pre_n + post_n) : -post_n]
0800:                         else:
0801:                             self._perm_pre_seq = pre_hist[-pre_n:] if pre_hist else []
0802:                         self._perm_post_seq = []
0803:                     self.candidate_history.append(step)
0804:                     self.candidate_count_total += 1
0805:                 # confirm (rule-dependent)
0806:                 confirm_hit = bool(vote_score >= threshold)
0807:                 perm_ok = True
0808:                 if perm_enabled and self._pending_candidate_step is not None and perm_cfg is not None:
0809:                     # pending 期间收集 post_seq（从 candidate_step 起，包含当前 step）
0810:                     self._perm_post_seq.extend([float(stat_t)] * int(batch_n))
0811:                     if len(self._perm_post_seq) > 4096:
0812:                         self._perm_post_seq = self._perm_post_seq[-4096:]
0813:                     pre_n = int(perm_cfg.get("pre_n") or 200)
0814:                     post_n = int(perm_cfg.get("post_n") or 50)
0815:                     if len(self._perm_pre_seq) >= pre_n and len(self._perm_post_seq) >= post_n:
0816:                         # 使用最近 post_n 的滑动窗口；达到 post_n 后每新增 1 个样本重算一次
0817:                         pre_seq = list(self._perm_pre_seq[-pre_n:])
0818:                         post_seq = list(self._perm_post_seq[-post_n:])
0819:                         n_perm = int(perm_cfg.get("n_perm") or 200)
0820:                         rng_seed = int(perm_cfg.get("rng_seed") or 0)
0821:                         p, obs = self._perm_test_one_sided(pre_seq, post_seq, n_perm=n_perm, seed=rng_seed)
0822:                         self.last_perm_pvalue = float(p)
0823:                         self.last_perm_effect = float(obs)
```
- cooldown 期间清空 pending（可导致候选被抹掉）：`drift/detectors.py`
```py
0740:         if cooldown > 0 and self._last_confirmed_pos is not None:
0741:             gap = int(current_pos - int(self._last_confirmed_pos))
0742:             if gap < int(cooldown):
0743:                 cooldown_active = True
0744:                 cooldown_remaining = int(cooldown - gap)
0745: 
0746:         if cooldown_active:
0747:             # cooldown 期间不允许新 confirm，且清空 pending，避免“过期后补确认”造成不必要的晚检。
0748:             self._clear_pending_state()
0749:         else:
```
- summary 里 p_le_alpha_ratio 的定义（min_effect=0 时应≈accept_over_test）：`training/loop.py`
```py
0465:             # perm_test 诊断（用于 Track AM；尽量从 monitor 内存状态读取，避免扫描大日志）
0466:             perm_pvalues = list(getattr(drift_monitor, "_perm_pvalues", []) or [])
0467:             perm_effects = list(getattr(drift_monitor, "_perm_effects", []) or [])
0468:             perm_alpha = float(getattr(drift_monitor, "perm_alpha", 0.01) or 0.01)
0469:             tw_obj = getattr(drift_monitor, "trigger_weights", None)
0470:             if isinstance(tw_obj, dict):
0471:                 for k in ("__perm_alpha", "perm_alpha"):
0472:                     if k in tw_obj:
0473:                         try:
0474:                             perm_alpha = float(tw_obj[k])  # type: ignore[arg-type]
0475:                         except Exception:
0476:                             pass
0477:             def _q(xs: list[float], q: float) -> float:
0478:                 if not xs:
0479:                     return float("nan")
0480:                 arr = np.asarray(xs, dtype=np.float64)
0481:                 return float(np.quantile(arr, q))
0482:             perm_pvalue_p50 = _q(perm_pvalues, 0.50)
0483:             perm_pvalue_p90 = _q(perm_pvalues, 0.90)
0484:             perm_pvalue_p99 = _q(perm_pvalues, 0.99)
0485:             perm_effect_p50 = _q(perm_effects, 0.50)
0486:             perm_effect_p90 = _q(perm_effects, 0.90)
0487:             perm_effect_p99 = _q(perm_effects, 0.99)
0488:             perm_pvalue_le_alpha_ratio = (float(sum(1 for p in perm_pvalues if p <= perm_alpha)) / float(len(perm_pvalues))) if perm_pvalues else float("nan")
0489:             payload: Dict[str, Any] = {
0490:                 "dataset_name": config.dataset_name,
```

### 3.3 对 Q1（B 类）回答：窗口/时间轴错配是否导致错过 drift early transition？
- 可复核量化证据来源：`RUN_summary_metrics` 的 `obs_nonpos_lower_bound`（由 effect 分位数推出的 obs<=0 下界）、`pvalue_mass_at_1_lower_bound`（由 pvalue 分位数推出的 p=1 下界）、以及 `cand_to_next_confirm_delay_p90`/`cand_to_next_confirm_frac_delay_gt_500`（延迟/超500比例）。
- 判据（写死）：在 drift run（sea_abrupt4）里，若 `obs_nonpos_lower_bound>=0.50` 且 `pvalue_mass_at_1_lower_bound>=0.50/0.10`，并伴随 `delay_p90` 偏大或 `frac_delay_gt_500>0`，则可将 B 排第一（对应实现：`obs<=0 -> p=1.0` 以及 step vs sample_idx 生命周期错配）。
- 逐 run 钉死证据（seed=1，no-drift 最低组在 drift run 的观测）：`perm_alpha=0.005`, `perm_stat=fused_score`, `perm_pre_n=500`, `perm_post_n=30`；`perm_pvalue_p50=1`, `perm_effect_p50=-0.009946` -> `obs_nonpos_lb=0.5`；`p@1_lb=0.5`, `delay_p90=640`, `frac_delay_gt_500=0.215278`。

### 3.4 对 Q2（A 类）回答：即便窗口足够，统计功效是否不足/不稳定？
- 可复核量化证据来源：`RUN_summary_metrics` 的 `perm_test_count_total`、`accept_over_test`、`perm_effect_p50`/`last_perm_effect`。
- 判据（写死）：若 `perm_test_count_total` 不低但 `accept_over_test` 仍偏低，且 `perm_effect_p50≈0` 或 `obs_nonpos_lower_bound>=0.50`，则 A（功效不足/不稳定）成立；若该现象主要集中在 drift run，则 A 为次因、B 为主因。

### 3.5 对 Q3（C 类）回答：cooldown/pending reset 是否频繁导致窗口凑不齐？
- 可复核证据来源：仅限 jsonl 片段（若存在）；对应表 `RUN_drilldown_extract_v4` 的 `jsonl_chosen/jsonl_jsonl_has_pending/jsonl_jsonl_has_cooldown`。
- 判据（写死）：若 jsonl 片段中出现 cooldown/pending 字段，且同一 run 的 `perm_test_count_total` 很低/为 0 或 `cand_unconfirmed_frac` 很高，则支持 C；否则 C 必须降权。
- 本轮选中 run 中，`jsonl_chosen` 为空的条数 = 12（即这些 log_dir 一级目录内不存在任何 .jsonl；可在 `dir_filenames_preview_json` 复核）。

## 4) 最终结论（基于本轮逐 run 证据）
- 结论 1（B 为主因，可复核）：no-drift 最低 perm_test 组在 drift run 上 `perm_pvalue_p50=1.0` 且 `perm_effect_p50<0`，对应实现中的 `obs<=0 -> p=1.0` 确定性路径，并且 `delay_p90` 与 `frac_delay_gt_500` 显著升高，解释了 drift 侧延迟/ miss 的上升（见 `RUN_summary_metrics`）。
- 结论 2（A 为次因，可复核）：同组 `perm_test_count_total` 并不低但仍出现大量 p=1 质量点与 effect 中位数为负/接近 0，说明并非“完全窗口凑不齐”，而是效应不稳定/功效不足叠加（见 `RUN_summary_metrics`）。
- 结论 3（C 证据不足，必须降权）：定点目录一级内未发现任何 jsonl（`jsonl_chosen` 全空），且 summary 不含 cooldown_active/pending 事件序列，因此无法在本轮允许数据源下证明“pending 被频繁清/重置”。
- 结论 4（D：口径不一致已证实）：`run_index.dataset=sea_nodrift` 但 `summary.dataset_name=sea_abrupt4` 的不一致在逐 run 层面可复核（见 `RUN_summary_anomalies`），这会直接破坏 drift/no-drift 的逐 run 对齐，应优先修正后再做更细的 A/B/C 统计对照。

