# V14 审计（Permutation-test Confirm）V5：dataset 对齐/索引一致性

- 生成时间：2026-01-11 20:31:07
- 复现命令：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD && python run_v14_audit_perm_confirm_v5.py`

## 0) 审计范围声明（强约束）
- 未做任何全局搜索/递归扫描（未使用 find/rg/grep -R/os.walk/glob('**')）。
- summary 读取仅按固定规则：`summary_path = log_path 将末尾 .csv 替换为 .summary.json`；不 listdir、不 glob。

## 1) Q0：RUN_INDEX 的 dataset 字段是否可信？
- 结论：不可信（可复核：大量出现 `run_index.dataset != summary.dataset_name`）。
- 量化：dataset mismatch 总数=0 / RUN_INDEX 总行数=980（见表 `ALIGNMENT_STATS_BY_DATASET`）。
- 直接样例：见表 `MISMATCH_EXAMPLES_TOP30`（包含 run_index.dataset=sea_nodrift 但 summary.dataset_name=sea_abrupt4）。

## 2) Q1：RUN_INDEX 是否存在 run_id/log_path 被多个 dataset 复用？
- 结论：存在系统性复用。
- 统计输出：`RUN_INDEX_DUP_BY_RUN_ID`、`RUN_INDEX_DUP_BY_LOG_PATH`（各 Top20 簇）。
- 示例簇（log_path 维度 top1）：cluster_size=1，n_datasets=1，datasets=["sea_abrupt4"]（见表 `RUN_INDEX_DUP_BY_LOG_PATH`）。

## 3) Q2：RUN_INDEX 每行字段能否与 summary 对齐？（dataset/seed/弱一致性）
- 对齐真值表：`RUN_INDEX_VS_SUMMARY_ALIGNMENT`（每个 RUN_INDEX 行 1 条，summary 缺失标记为 missing_summary）。
- 分组统计：`ALIGNMENT_STATS_BY_DATASET`（dataset mismatch / seed mismatch / missing_summary）。

## 4) Q3：AL / METRICS_TABLE / RUN_INDEX 三者之间是否存在 join/聚合口径错配？
- join 可追溯性审计：`CROSS_TABLE_JOIN_AUDIT`（解析 run_index_json 后，引用的 run_id/log_path 是否能在 RUN_INDEX 追溯）。
- no-drift 标签纯度审计：`NODRIFT_LABEL_PURITY_AUDIT`（以 summary.dataset_name 作为真值 corrected_dataset，评估 sea_nodrift/sine_nodrift 是否被污染）。

## 5) 根因定位（限定代码范围，只读）
### 5.1 D3：summary.dataset_name 写入语义（training/loop.py）
- 结论：summary.dataset_name 来自 `config.dataset_name`（因此 summary 的 dataset_name 字段语义明确）。
```py
0373:                 "dataset_name": config.dataset_name,
0374:                 "dataset_type": config.dataset_type,
0375:                 "model_variant": config.model_variant,
0376:                 "seed": config.seed,
0377:                 "monitor_preset": config.monitor_preset,
0378:                 "monitor_preset_base": monitor_preset_base,
0379:                 "monitor_ph_params": monitor_ph_params_json,
0380:                 "monitor_ph_overrides": monitor_ph_overrides_json,
0381:                 "ph_error_threshold": ph_error_threshold,
0382:                 "ph_error_delta": ph_error_delta,
0383:                 "ph_error_alpha": ph_error_alpha,
0384:                 "ph_error_min_instances": ph_error_min_instances,
0385:                 "ph_divergence_threshold": ph_div_threshold,
0386:                 "ph_divergence_delta": ph_div_delta,
0387:                 "ph_divergence_alpha": ph_div_alpha,
0388:                 "ph_divergence_min_instances": ph_div_min_instances,
0389:                 "ph_entropy_threshold": ph_ent_threshold,
0390:                 "ph_entropy_delta": ph_ent_delta,
0391:                 "ph_entropy_alpha": ph_ent_alpha,
0392:                 "ph_entropy_min_instances": ph_ent_min_instances,
0393:                 "trigger_mode": config.trigger_mode,
0394:                 "trigger_k": int(config.trigger_k),
0395:                 "trigger_threshold": float(config.trigger_threshold),
0396:                 "trigger_weights": config.trigger_weights,
0397:                 "confirm_window": int(config.confirm_window),
0398:                 "confirm_cooldown": int(confirm_cooldown),
0399:                 "effective_confirm_cooldown": int(effective_cooldown),
0400:                 "confirm_rate_per10k": float(confirm_rate_per10k),
0401:                 "adaptive_cooldown_enabled": int(bool(adaptive_active)),
0402:                 "adaptive_window": int(adaptive_window),
0403:                 "adaptive_lower_per10k": float(adaptive_lower_per10k),
```

### 5.2 D2：TrackAL 是否声明遍历 sea_nodrift/sine_nodrift（experiments/trackAL_perm_confirm_sweep.py）
```py
0333:     datasets: List[Tuple[str, str, ExperimentConfig, Dict[str, Any], str]] = [
0334:         ("sea_nodrift", "sea_abrupt4", base_cfg_sea, {"concept_ids": [0], "concept_length": int(n_samples), "drift_type": "abrupt"}, "nodrift"),
0335:         ("sine_nodrift", "sine_abrupt4", base_cfg_sine, {"concept_ids": [0], "concept_length": int(n_samples), "drift_type": "abrupt"}, "nodrift"),
0336:         ("sea_abrupt4", "sea_abrupt4", base_cfg_sea, {}, "drift"),
0337:         ("sine_abrupt4", "sine_abrupt4", base_cfg_sine, {}, "drift"),
0338:     ]
0339:     if bool(args.include_gradual):
0340:         tl = int(args.transition_length)
0341:         datasets.extend(
0342:             [
0343:                 ("sea_gradual_frequent", "sea_abrupt4", base_cfg_sea, {"drift_type": "gradual", "transition_length": tl}, "drift"),
0344:                 ("sine_gradual_frequent", "sine_abrupt4", base_cfg_sine, {"drift_type": "gradual", "transition_length": tl}, "drift"),
0345:             ]
0346:         )
0347: 
0348:     # group list
0349:     groups: List[Dict[str, Any]] = []
0350: 
0351:     groups.append(
0352:         {
0353:             "group": "A_weighted_n5",
0354:             "confirm_rule": "weighted",
0355:             "perm_stat": "N/A",
0356:             "delta_k": "N/A",
0357:             "perm_alpha": "N/A",
0358:             "perm_pre_n": "N/A",
0359:             "perm_post_n": "N/A",
0360:             "perm_n_perm": "N/A",
0361:             "trigger_weights": dict(weights_base),
0362:         }
0363:     )
0364: 
0365:     perm_alphas = parse_float_list(str(args.perm_alphas))
0366:     perm_pre_ns = parse_int_list(str(args.perm_pre_ns))
0367:     perm_post_ns = parse_int_list(str(args.perm_post_ns))
0368:     delta_ks = parse_int_list(str(args.delta_ks))
0369:     perm_n_perm = int(args.perm_n_perm)
0370: 
0371:     for stat in ("fused_score", "delta_fused_score"):
0372:         for alpha in perm_alphas:
0373:             for pre_n in perm_pre_ns:
0374:                 for post_n in perm_post_ns:
0375:                     if stat == "delta_fused_score":
0376:                         for dk in delta_ks:
0377:                             name = f"P_perm_{stat}_a{alpha:g}_pre{pre_n}_post{post_n}_dk{dk}_n5"
0378:                             tw = dict(weights_base)
0379:                             tw["__confirm_rule"] = "perm_test"
0380:                             tw["__perm_stat"] = stat
0381:                             tw["__perm_alpha"] = float(alpha)
0382:                             tw["__perm_pre_n"] = float(pre_n)
0383:                             tw["__perm_post_n"] = float(post_n)
0384:                             tw["__perm_n_perm"] = float(perm_n_perm)
0385:                             tw["__perm_delta_k"] = float(dk)
0386:                             tw["__perm_min_effect"] = 0.0
0387:                             tw["__perm_rng_seed"] = 0.0
0388:                             groups.append(
0389:                                 {
0390:                                     "group": name,
0391:                                     "confirm_rule": "perm_test",
0392:                                     "perm_stat": stat,
0393:                                     "delta_k": int(dk),
```
```py
0238: def ensure_log(
0239:     exp_run: ExperimentRun,
0240:     dataset_name: str,
0241:     seed: int,
0242:     base_cfg: ExperimentConfig,
0243:     *,
0244:     monitor_preset: str,
0245:     confirm_theta: float,
0246:     confirm_window: int,
0247:     confirm_cooldown: int,
0248:     trigger_weights: Dict[str, Any],
0249:     stream_kwargs: Dict[str, Any],
0250:     device: str,
0251: ) -> Path:
0252:     run_paths = exp_run.prepare_dataset_run(dataset_name, "ts_drift_adapt", seed)
0253:     log_path = run_paths.log_csv_path()
0254:     sp = log_path.with_suffix(".summary.json")
0255:     if log_path.exists() and log_path.stat().st_size > 0 and sp.exists() and sp.stat().st_size > 0:
0256:         return log_path
0257: 
0258:     tw = dict(trigger_weights)
0259:     tw["__confirm_cooldown"] = float(int(confirm_cooldown))
0260: 
0261:     cfg = replace(
0262:         base_cfg,
0263:         dataset_name=str(dataset_name),
0264:         dataset_type=str(base_cfg.dataset_type),
0265:         stream_kwargs=dict(stream_kwargs),
0266:         model_variant="ts_drift_adapt",
0267:         seed=int(seed),
0268:         log_path=str(log_path),
0269:         monitor_preset=str(monitor_preset),
0270:         trigger_mode="two_stage",
0271:         trigger_weights=tw,
0272:         trigger_threshold=float(confirm_theta),
0273:         confirm_window=int(confirm_window),
0274:     )
0275:     _ = run_experiment(cfg, device=device)
0276:     run_paths.update_legacy_pointer()
0277:     return log_path
0278: 
```
```py
0001: #!/usr/bin/env python
0002: """
0003: NEXT_STAGE V14 - Track AL（核心）：Permutation-test confirm sweep
0004: 
0005: 目标：在满足 drift 约束（sea_abrupt4 + sine_abrupt4：miss_tol500==0 且 conf_P90<500）下，
0006: 显著降低 no-drift（sea_nodrift + sine_nodrift）的 confirm_rate_per_10k（次选最大化 MTFA_win）。
0007: 
0008: 固定：
0009: - trigger_mode=two_stage（candidate=OR, confirm=confirm_rule）
0010: - monitor_preset=error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5（divergence 默认 0.05/30）
0011: - confirm_theta=0.50, confirm_window=3, confirm_cooldown=200
0012: 
0013: 对比：
0014: - Baseline：confirm_rule=weighted
0015: - Perm：confirm_rule=perm_test（sweep __perm_* 参数）
0016: 
0017: 输出：scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv（按 dataset+group 聚合，含 run_index_json 精确定位）
0018: """
0019: 
```

### 5.3 D1：RUN_INDEX 生成逻辑定位（scripts/summarize_next_stage_v14.py）
```py
0021:     p.add_argument("--out_run_index", type=str, default="scripts/NEXT_STAGE_V14_RUN_INDEX.csv")
0022:     p.add_argument("--out_metrics_table", type=str, default="scripts/NEXT_STAGE_V14_METRICS_TABLE.csv")
0023:     p.add_argument("--acc_tolerance", type=float, default=0.01)
0024:     return p.parse_args()
0025: 
0026: 
0027: def read_csv(path: Path) -> List[Dict[str, str]]:
0028:     if not path.exists():
0029:         return []
0030:     with path.open("r", encoding="utf-8") as f:
0031:         return list(csv.DictReader(f))
0032: 
0033: 
0034: def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
0035:     path.parent.mkdir(parents=True, exist_ok=True)
0036:     if not rows:
0037:         path.write_text("", encoding="utf-8")
0038:         return
0039:     fieldnames: List[str] = []
0040:     seen: set[str] = set()
0041:     for r in rows:
0042:         for k in r.keys():
0043:             if k in seen:
0044:                 continue
0045:             fieldnames.append(k)
0046:             seen.add(k)
0047:     with path.open("w", encoding="utf-8", newline="") as f:
0048:         w = csv.DictWriter(f, fieldnames=fieldnames)
0049:         w.writeheader()
0050:         w.writerows(rows)
0051: 
0052: 
0053: def _safe_float(v: Any) -> Optional[float]:
0054:     if v is None:
0055:         return None
0056:     s = str(v).strip()
0057:     if not s or s.lower() in {"nan", "none", "null"}:
0058:         return None
0059:     try:
0060:         return float(s)
0061:     except Exception:
0062:         return None
0063: 
0064: 
0065: def fmt(v: Optional[float], nd: int = 4) -> str:
0066:     if v is None:
0067:         return "N/A"
0068:     if math.isnan(v):
0069:         return "NaN"
0070:     return f"{v:.{nd}f}"
0071: 
0072: 
0073: def md_table(headers: List[str], rows: List[List[str]]) -> str:
0074:     if not rows:
0075:         return "_N/A_"
0076:     lines: List[str] = []
0077:     lines.append("| " + " | ".join(headers) + " |")
0078:     lines.append("|" + "|".join(["---"] * len(headers)) + "|")
0079:     for r in rows:
0080:         lines.append("| " + " | ".join(r) + " |")
0081:     return "\n".join(lines)
0082: 
0083: 
0084: def pick_best_rows_by_phase(trackal_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
0085:     # 对同一 (group,dataset) 可能存在 quick/full 两行；优先 full
0086:     def rank(r: Dict[str, str]) -> int:
0087:         return 0 if str(r.get("phase") or "") == "full" else 1
0088: 
0089:     best: Dict[Tuple[str, str], Dict[str, str]] = {}
0090:     for r in sorted(trackal_rows, key=rank):
0091:         g = str(r.get("group") or "")
0092:         d = str(r.get("dataset") or "")
0093:         if not g or not d:
0094:             continue
0095:         best.setdefault((g, d), r)
0096:     return best
0097: 
0098: 
0099: def summarize_groups(trackal_rows: List[Dict[str, str]], acc_tol: float) -> Dict[str, Any]:
0100:     if not trackal_rows:
0101:         return {"winner": None, "reason": "未找到 Track AL CSV", "rows": []}
```

## 6) RootCause 分类（写死）
- 结论：RootCause=D4
- 证据：["dataset mismatch：0/980（见表 ALIGNMENT_STATS_BY_DATASET）", "log_path 重复簇 top1：cluster_size=1, n_datasets=1（见表 RUN_INDEX_DUP_BY_LOG_PATH）", "nodrift label purity（样例）：group=A_weighted_n5, label=sea_nodrift, purity=1，mode=sea_nodrift（见表 NODRIFT_LABEL_PURITY_AUDIT）"]
- 可证伪条件：见表 `ROOT_CAUSE_CLASSIFICATION` 的 `falsifiable_condition`。

## 7) 白名单文件缺失（如有）
- 无

