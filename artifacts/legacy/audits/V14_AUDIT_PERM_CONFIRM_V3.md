# V14 审计（Permutation-test Confirm）V3

- 生成时间：2026-01-11 16:52:31
- 复现命令：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD && python run_v14_audit_perm_confirm_v3.py`

## 0) 审计范围声明（强约束）
- 未进行任何全局搜索/扫描（未对 `logs/` 做递归查找、未做全仓库 grep/rg）。
- 逐 run drill-down 仅使用 `scripts/NEXT_STAGE_V14_RUN_INDEX.csv` 的 `log_path` 定位；只在该 `log_path` 所在目录内按固定顺序尝试 + 最多 1 次局部 glob。
- 未重跑任何训练/实验（不生成新 runs）。

## 1) Task A：表格级复核（TRACKAL 聚合口径）
- Step1 可行组数量：1
- best_acc_final（Step1 可行组内最大 drift_acc_final）：0.765979
- winner（Step1→Step2→并列规则）：`A_weighted_n5`
- NEXT_STAGE_V14_REPORT 声明 winner：`A_weighted_n5`
- 既有 V14_AUDIT（根目录白名单文件）winner：`N/A`（文件不存在则为 N/A）

### 1.1 可行组总表（A2）
- 对应表：`V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv` / `table_name=AL_feasible_groups`

### 1.2 Top-10 最接近硬约束的 perm_test（A3）
- 对应表：`V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv` / `table_name=AL_top10_near_constraints`
- rank=1：`P_perm_delta_fused_score_a0.05_pre500_post50_dk50_n5`
- no-drift 最低 perm_test：`P_perm_fused_score_a0.005_pre500_post30_n5`（no_drift_rate≈17.044062）

## 2) Task B：AM 诊断口径与一致性核查
### 2.1 B1 口径定义（来自代码，可复核）
- drift/detectors.py（perm cfg 透传与覆盖）：`drift/detectors.py`
```py
0537:     def _resolve_perm_test_cfg(self) -> Dict[str, Any]:
0538:         tw = getattr(self, "trigger_weights", None)
0539: 
0540:         def _int_from(keys: Tuple[str, ...], default: int) -> int:
0541:             if isinstance(tw, dict):
0542:                 for k in keys:
0543:                     if k not in tw:
0544:                         continue
0545:                     try:
0546:                         return int(float(tw[k]))  # type: ignore[arg-type]
0547:                     except Exception:
0548:                         continue
0549:             return int(default)
0550: 
0551:         def _float_from(keys: Tuple[str, ...], default: float) -> float:
0552:             if isinstance(tw, dict):
0553:                 for k in keys:
0554:                     if k not in tw:
0555:                         continue
0556:                     try:
0557:                         return float(tw[k])  # type: ignore[arg-type]
0558:                     except Exception:
0559:                         continue
0560:             return float(default)
0561: 
0562:         def _str_from(keys: Tuple[str, ...], default: str) -> str:
0563:             if isinstance(tw, dict):
0564:                 for k in keys:
0565:                     if k not in tw:
0566:                         continue
0567:                     v = tw.get(k)
0568:                     if v is None:
0569:                         continue
0570:                     s = str(v).strip()
0571:                     if s:
0572:                         return s
0573:             return str(default)
0574: 
0575:         pre_n = _int_from(("__perm_pre_n", "perm_pre_n"), int(getattr(self, "perm_pre_n", 200) or 200))
0576:         post_n = _int_from(("__perm_post_n", "perm_post_n"), int(getattr(self, "perm_post_n", 50) or 50))
0577:         n_perm = _int_from(("__perm_n_perm", "perm_n_perm"), int(getattr(self, "perm_n_perm", 200) or 200))
0578:         alpha = _float_from(("__perm_alpha", "perm_alpha"), float(getattr(self, "perm_alpha", 0.01) or 0.01))
0579:         stat = _str_from(("__perm_stat", "perm_stat"), str(getattr(self, "perm_stat", "fused_score") or "fused_score")).lower()
0580:         min_eff = _float_from(("__perm_min_effect", "perm_min_effect"), float(getattr(self, "perm_min_effect", 0.0) or 0.0))
0581:         rng_seed = _int_from(("__perm_rng_seed", "perm_rng_seed"), int(getattr(self, "perm_rng_seed", 0) or 0))
0582:         delta_k = _int_from(("__perm_delta_k", "perm_delta_k"), int(getattr(self, "perm_delta_k", 50) or 50))
0583: 
0584:         pre_n = max(1, int(pre_n))
0585:         post_n = max(1, int(post_n))
0586:         n_perm = max(1, int(n_perm))
0587:         alpha = float(alpha)
0588:         if not (0.0 < alpha <= 1.0):
0589:             alpha = 0.01
0590:         if stat not in {"fused_score", "delta_fused_score"}:
0591:             stat = "fused_score"
0592:         min_eff = float(min_eff)
0593:         delta_k = max(1, int(delta_k))
0594:         return {
0595:             "pre_n": pre_n,
0596:             "post_n": post_n,
0597:             "n_perm": n_perm,
0598:             "alpha": alpha,
0599:             "stat": stat,
0600:             "min_effect": min_eff,
0601:             "rng_seed": int(rng_seed),
0602:             "delta_k": delta_k,
0603:         }
```
- drift/detectors.py（one-sided pvalue：obs<=0 -> p=1.0）：`drift/detectors.py`
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
- drift/detectors.py（accept 定义：p<=alpha 且 obs>=min_effect；不足窗口 perm_ok=False）：`drift/detectors.py`
```py
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
0824:                         self.perm_test_count_total += 1
0825:                         if len(self._perm_pvalues) < 10000 and not math.isnan(float(p)):
0826:                             self._perm_pvalues.append(float(p))
0827:                         if len(self._perm_effects) < 10000 and not math.isnan(float(obs)):
0828:                             self._perm_effects.append(float(obs))
0829:                         alpha = float(perm_cfg.get("alpha") or 0.01)
0830:                         min_eff = float(perm_cfg.get("min_effect") or 0.0)
0831:                         perm_ok = bool(float(p) <= float(alpha) and float(obs) >= float(min_eff))
0832:                         if perm_ok:
0833:                             self.perm_accept_count_total += 1
0834:                         else:
0835:                             self.perm_reject_count_total += 1
0836:                     else:
0837:                         perm_ok = False
0838:                 if perm_enabled:
0839:                     confirm_hit = bool(perm_ok)
0840:                 # k_of_n confirm：hit 定义为 vote_score>=threshold（与 weighted 的阈值语义一致）
```
- training/loop.py（summary 里 perm_pvalue_* 与 le_alpha_ratio 的生成）：`training/loop.py`
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
0491:                 "dataset_type": config.dataset_type,
0492:                 "model_variant": config.model_variant,
0493:                 "seed": int(config.seed),
0494:                 "monitor_preset": config.monitor_preset,
0495:                 "monitor_preset_base": monitor_preset_base,
0496:                 "monitor_ph_params": monitor_ph_params,
0497:                 "monitor_ph_overrides": monitor_ph_overrides,
0498:                 "trigger_mode": config.trigger_mode,
0499:                 "trigger_k": int(config.trigger_k),
0500:                 "trigger_threshold": float(config.trigger_threshold),
0501:                 "trigger_weights": config.trigger_weights,
0502:                 "confirm_window": int(config.confirm_window),
0503:                 "confirm_cooldown": int(confirm_cooldown),
0504:                 "adaptive_cooldown_enabled": int(bool(adaptive_enabled)),
0505:                 "adaptive_window": int(adaptive_window),
0506:                 "adaptive_lower_per10k": float(adaptive_lower_per10k),
0507:                 "adaptive_upper_per10k": float(adaptive_upper_per10k),
0508:                 "adaptive_cooldown_low": int(adaptive_cooldown_low),
0509:                 "adaptive_cooldown_high": int(adaptive_cooldown_high),
0510:                 "severity_scheduler_scale": float(config.severity_scheduler_scale),
0511:                 "use_severity_v2": int(bool(config.use_severity_v2)),
0512:                 "severity_gate": str(config.severity_gate),
0513:                 "severity_gate_min_streak": int(getattr(config, "severity_gate_min_streak", 1) or 1),
0514:                 "entropy_mode": str(config.entropy_mode),
0515:                 "severity_decay": float(config.severity_decay),
0516:                 "freeze_baseline_steps": int(config.freeze_baseline_steps),
0517:                 "n_steps": int(config.n_steps),
0518:                 "horizon": horizon,
0519:                 "acc_final": acc_final,
0520:                 "mean_acc": mean_acc,
0521:                 "acc_min": acc_min,
0522:                 "candidate_sample_idxs": candidate_sample_idxs,
0523:                 "confirmed_sample_idxs": confirmed_sample_idxs,
0524:                 "acc_series": [[int(x), float(a)] for x, a in acc_series],
0525:                 "candidate_count_total": int(last["candidate_count_total"]) if last is not None else 0,
0526:                 "confirmed_count_total": int(last["confirmed_count_total"]) if last is not None else 0,
0527:                 "created_at": float(time.time()),
0528:                 "confirm_rule_effective": str(getattr(drift_monitor, "last_confirm_rule", "")),
0529:                 "last_perm_pvalue": float(getattr(drift_monitor, "last_perm_pvalue", float("nan"))),
0530:                 "last_perm_effect": float(getattr(drift_monitor, "last_perm_effect", float("nan"))),
0531:                 "perm_test_count_total": int(getattr(drift_monitor, "perm_test_count_total", 0)),
0532:                 "perm_accept_count_total": int(getattr(drift_monitor, "perm_accept_count_total", 0)),
0533:                 "perm_reject_count_total": int(getattr(drift_monitor, "perm_reject_count_total", 0)),
0534:                 "perm_alpha": float(perm_alpha),
0535:                 "perm_pvalue_p50": float(perm_pvalue_p50),
0536:                 "perm_pvalue_p90": float(perm_pvalue_p90),
0537:                 "perm_pvalue_p99": float(perm_pvalue_p99),
0538:                 "perm_pvalue_le_alpha_ratio": float(perm_pvalue_le_alpha_ratio),
0539:                 "perm_effect_p50": float(perm_effect_p50),
0540:                 "perm_effect_p90": float(perm_effect_p90),
0541:                 "perm_effect_p99": float(perm_effect_p99),
0542:             }
0543:             summary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
```
- experiments/trackAL_perm_confirm_sweep.py（注入 __perm_*）：`experiments/trackAL_perm_confirm_sweep.py`
```py
0340:     delta_ks = parse_int_list(str(args.delta_ks))
0341:     perm_n_perm = int(args.perm_n_perm)
0342: 
0343:     for stat in ("fused_score", "delta_fused_score"):
0344:         for alpha in perm_alphas:
0345:             for pre_n in perm_pre_ns:
0346:                 for post_n in perm_post_ns:
0347:                     if stat == "delta_fused_score":
0348:                         for dk in delta_ks:
0349:                             name = f"P_perm_{stat}_a{alpha:g}_pre{pre_n}_post{post_n}_dk{dk}_n5"
0350:                             tw = dict(weights_base)
0351:                             tw["__confirm_rule"] = "perm_test"
0352:                             tw["__perm_stat"] = stat
0353:                             tw["__perm_alpha"] = float(alpha)
0354:                             tw["__perm_pre_n"] = float(pre_n)
0355:                             tw["__perm_post_n"] = float(post_n)
0356:                             tw["__perm_n_perm"] = float(perm_n_perm)
0357:                             tw["__perm_delta_k"] = float(dk)
0358:                             tw["__perm_min_effect"] = 0.0
0359:                             tw["__perm_rng_seed"] = 0.0
0360:                             groups.append(
0361:                                 {
0362:                                     "group": name,
0363:                                     "confirm_rule": "perm_test",
0364:                                     "perm_stat": stat,
0365:                                     "delta_k": int(dk),
0366:                                     "perm_alpha": float(alpha),
0367:                                     "perm_pre_n": int(pre_n),
0368:                                     "perm_post_n": int(post_n),
0369:                                     "perm_n_perm": int(perm_n_perm),
0370:                                     "trigger_weights": tw,
0371:                                 }
0372:                             )
0373:                     else:
0374:                         name = f"P_perm_{stat}_a{alpha:g}_pre{pre_n}_post{post_n}_n5"
0375:                         tw = dict(weights_base)
0376:                         tw["__confirm_rule"] = "perm_test"
0377:                         tw["__perm_stat"] = stat
0378:                         tw["__perm_alpha"] = float(alpha)
0379:                         tw["__perm_pre_n"] = float(pre_n)
0380:                         tw["__perm_post_n"] = float(post_n)
0381:                         tw["__perm_n_perm"] = float(perm_n_perm)
0382:                         tw["__perm_min_effect"] = 0.0
0383:                         tw["__perm_rng_seed"] = 0.0
0384:                         groups.append(
0385:                             {
0386:                                 "group": name,
0387:                                 "confirm_rule": "perm_test",
0388:                                 "perm_stat": stat,
0389:                                 "delta_k": "N/A",
0390:                                 "perm_alpha": float(alpha),
0391:                                 "perm_pre_n": int(pre_n),
0392:                                 "perm_post_n": int(post_n),
0393:                                 "perm_n_perm": int(perm_n_perm),
0394:                                 "trigger_weights": tw,
0395:                             }
```
- experiments/trackAM_perm_diagnostics.py（AM 读取每个 run 的 *.summary.json 并聚合）：`experiments/trackAM_perm_diagnostics.py`
```py
0205: 
0206:             ds_info = run_index.get(d) if isinstance(run_index, dict) else None
0207:             runs = (ds_info or {}).get("runs") if isinstance(ds_info, dict) else None
0208:             if not isinstance(runs, list):
0209:                 continue
0210: 
0211:             cand_counts: List[Optional[float]] = []
0212:             conf_counts: List[Optional[float]] = []
0213:             ratios: List[Optional[float]] = []
0214:             p50s: List[Optional[float]] = []
0215:             p90s: List[Optional[float]] = []
0216:             p99s: List[Optional[float]] = []
0217:             le_alpha: List[Optional[float]] = []
0218: 
0219:             for item in runs:
0220:                 try:
0221:                     log_path = Path(str(item.get("log_path")))
0222:                 except Exception:
0223:                     continue
0224:                 summ = read_summary(log_path)
0225:                 ccand = _safe_float(summ.get("candidate_count_total"))
0226:                 cconf = _safe_float(summ.get("confirmed_count_total"))
0227:                 cand_counts.append(ccand)
0228:                 conf_counts.append(cconf)
0229:                 if ccand is not None and ccand > 0 and cconf is not None:
0230:                     ratios.append(float(cconf) / float(ccand))
0231:                 else:
0232:                     ratios.append(None)
0233:                 p50s.append(_safe_float(summ.get("perm_pvalue_p50")))
0234:                 p90s.append(_safe_float(summ.get("perm_pvalue_p90")))
0235:                 p99s.append(_safe_float(summ.get("perm_pvalue_p99")))
0236:                 le_alpha.append(_safe_float(summ.get("perm_pvalue_le_alpha_ratio")))
0237: 
0238:             out_rows.append(
0239:                 {
0240:                     "track": "AM",
0241:                     "dataset": d,
0242:                     "group": g,
0243:                     "phase": str(r.get("phase") or ""),
0244:                     "confirm_rule": str(r.get("confirm_rule") or ""),
0245:                     "perm_stat": str(r.get("perm_stat") or ""),
0246:                     "delta_k": str(r.get("delta_k") or ""),
0247:                     "perm_alpha": str(r.get("perm_alpha") or ""),
0248:                     "perm_pre_n": str(r.get("perm_pre_n") or ""),
0249:                     "perm_post_n": str(r.get("perm_post_n") or ""),
0250:                     "perm_n_perm": str(r.get("perm_n_perm") or ""),
0251:                     "n_runs": int(len(runs)),
0252:                     "candidate_count_mean": mean(cand_counts),
0253:                     "candidate_count_std": std(cand_counts),
0254:                     "confirmed_count_mean": mean(conf_counts),
0255:                     "confirmed_count_std": std(conf_counts),
0256:                     "confirmed_over_candidate_mean": mean(ratios),
0257:                     "perm_pvalue_p50_mean": mean(p50s),
0258:                     "perm_pvalue_p90_mean": mean(p90s),
0259:                     "perm_pvalue_p99_mean": mean(p99s),
0260:                     "perm_pvalue_le_alpha_ratio_mean": mean(le_alpha),
```

口径小结：
- `perm_pvalue`：`drift/detectors.py` 的 `_perm_test_one_sided()` 计算；当 `obs = post.mean - pre.mean <= 0` 时直接返回 `p=1.0`（强烈把“非正向变化”压到 1.0）。
- `accept`：在 `confirm_rule=perm_test` 时，`perm_ok = (p<=alpha) and (obs>=min_effect)`；窗口不足时 `perm_ok=False`，且该次不会增加 `perm_test_count_total`。
- `perm_pvalue_le_alpha_ratio`：在 `training/loop.py` summary 里定义为 `perm_pvalues` 列表中 `p<=perm_alpha` 的比例（`perm_pvalues` 来自 monitor 内存 `_perm_pvalues`）。当 `min_effect==0` 时与 `accept_count_total/test_count_total` 口径一致；若未来 `min_effect>0` 则两者会分叉。

### 2.2 B2 一致性校验结果（异常清单）
- 对应表：`V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv` / `table_name=AM_consistency_anomalies`
- 说明：diag 不含 raw pvalue 序列，因此“bimodal 证据”的 unique/占比统计在表格层面为 N/A；但实现层面可由 `obs<=0 -> p=1.0` 解释为何容易出现 `p90/p99=1.0` 的质量点。

## 3) Task C：逐 run drill-down（严格使用 log_path；不乱找）
- 对应表：`V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv` / `table_name=RUN_drilldown_extract`
- 对照表：`V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv` / `table_name=DIFF_diag_vs_summary`
- 关键约束复述：每个 run 的 summary 定位最多 3 次固定路径尝试 + 1 次局部 glob（仅在该 log_path 所在目录内）。

## 4) Task D：失败归因（可证伪分类 + 证据链）
### 4.1 结论摘要
- 现象（来自 `scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv`）：Step1 硬约束可行组极少（本次为 1 个：`A_weighted_n5`）；perm_test 网格里 no-drift 误报可降低，但 drift 侧 miss/延迟同时恶化，导致无法同时满足 Step1。
- 主要归因排序：B（窗口对齐/统计量错配） > A（统计功效不足/不稳定） > C（状态机副作用） > D（实现 bug/口径不一致：未发现硬 bug，但存在可解释的实现行为导致 bimodal）。

### 4.2 分类证据与“如何推翻”
**A：power 不足/不稳定**
- 证据：perm_test 的确认依赖 `obs=post.mean-pre.mean` 的 one-sided 置换检验；在 drift 早期/窗口污染情况下，`obs` 不稳定会导致大量 `p=1.0` 或 reject，从而推迟确认并造成 `conf_P90_mean` 上升/`miss_tol500_mean` 上升。
- 如何推翻：若能在 run summary 中看到 drift 数据集里 `perm_pvalue_P90` 明显小于 alpha 且 `perm_pvalue_le_alpha_ratio` 高，同时仍出现高 miss/高 confP90，则 A 不是主因。

**B：窗口对齐/统计量错配**
- 证据：实现中 pre/post 窗口以 `sample_idx` 展开（`batch_n=current_pos-prev_pos`），但 confirm 的 pending 生命周期由 `confirm_window`（step 计数）控制，且 pre-window 还会“去污染”跳过最近 `post_n`（`drift/detectors.py` 795-801）。这些都会使 drift early transition 的 effect 被稀释/错过。
- 如何推翻：若在 drift 数据集里，run summary 显示 `test_count` 很高、`p_le_alpha_ratio` 很高，但 `confirmed_count_total/candidate_count_total` 仍很低且 pending 常在 deadline 之前满足 post_n，则需要转向 C/D。

**C：状态机副作用**
- 证据：cooldown 期间直接清空 pending（`drift/detectors.py` 746-749），会导致“候选触发了，但在 cooldown 内被抹掉”，从而错过 confirm 窗口（表现为延迟变大/甚至 miss）。
- 如何推翻：若在 drift 数据集里 cooldown_active 很少、且 pending 未被频繁清空，但仍出现同样的 miss/延迟，则 C 不是主要原因。

**D：实现 bug / 口径不一致**
- 证据（未发现硬 bug）：AM diag 的 `perm_pvalue_*` 与 `perm_pvalue_le_alpha_ratio` 明确来自每个 run 的 `.summary.json`（`experiments/trackAM_perm_diagnostics.py`）；summary 中 `perm_pvalue_le_alpha_ratio` 的计算也与字段名一致（`training/loop.py` 488）。
- 风险点（实现行为导致“看起来异常”）：`obs<=0 -> p=1.0` 的早返回会天然造成 pvalue 分布在 1.0 处堆积，叠加 drift/no-drift 下不同窗口行为，会形成 bimodal 统计形态，可能被误读为“数据/口径异常”。
- 如何推翻：若发现同一 run 的 `accept_over_test` 与 `p_le_alpha_ratio` 大幅不一致（且 `min_effect==0`），或发现 `perm_alpha` 在 summary 与 trigger_weights/__perm_alpha 不一致，则属于 D 类问题。

## 5) 白名单文件存在性（缺失标 N/A）
- scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv：OK
- scripts/TRACKAM_PERM_DIAG.csv：OK
- scripts/NEXT_STAGE_V14_RUN_INDEX.csv：OK
- scripts/NEXT_STAGE_V14_METRICS_TABLE.csv：OK
- scripts/NEXT_STAGE_V14_REPORT.md：OK
- V14_AUDIT_PERM_CONFIRM.md：N/A
- V14_AUDIT_PERM_CONFIRM_TABLES.csv：N/A

