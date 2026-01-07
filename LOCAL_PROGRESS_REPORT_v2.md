# LOCAL_PROGRESS_REPORT_v2

生成时间：`2026-01-07 11:38:58`

## Step0：仓库信息与运行环境（ZZNSTD）

- git HEAD: `92e5be5`
- git status --porcelain: `M changelog/CHANGELOG.md
?? LOCAL_PROGRESS_REPORT.md
?? RUN_INDEX.csv`
- python: `Python 3.10.19`
- which python: `/home/ylh/anaconda3/envs/ZZNSTD/bin/python`
- pip show (Name/Version/Location)：
  - Name: numpy
  - Version: 2.2.6
  - Location: /home/ylh/.local/lib/python3.10/site-packages
  - Name: pandas
  - Version: 2.3.3
  - Location: /home/ylh/.local/lib/python3.10/site-packages
  - Name: pyarrow
  - Version: 22.0.0
  - Location: /home/ylh/anaconda3/envs/ZZNSTD/lib/python3.10/site-packages
- 关键目录存在性：
  - `logs/`: OK
  - `results/`: OK
  - `experiments/`: OK
  - `evaluation/`: OK
  - `drift/`: OK
  - `scheduler/`: OK
  - `training/`: OK
  - `docs/`: OK

## Step1：扫描最近 runs（run_id 新结构 + legacy 扁平结构）

- 已生成索引：`RUN_INDEX.csv`（包含 logs/results 的 run_id 目录 + legacy 文件）
### Top10 run_id 目录（logs+results，按 modified_time）
- `20251205-012259-ww5_phase0_mlp_full_supervised` exp=phase0_offline_supervised dataset=Airlines model=tabular_mlp_baseline seed=2 time=2025-12-05 01:27:51
- `20251205-012259-ww5_phase0_mlp_full_supervised` exp=phase0_offline_supervised dataset=summary model=20251205-012259-ww5_phase0_mlp_full_supervised seed=- time=2025-12-05 01:27:51
- `20251205-012259-807_phase0_mlp_full_supervised` exp=phase0_offline_supervised dataset=Airlines model=tabular_mlp_baseline seed=1 time=2025-12-05 01:27:44
- `20251205-012259-807_phase0_mlp_full_supervised` exp=phase0_offline_supervised dataset=summary model=20251205-012259-807_phase0_mlp_full_supervised seed=- time=2025-12-05 01:27:44
- `20251205-012259-uf7_phase0_mlp_full_supervised` exp=phase0_offline_supervised dataset=Airlines model=tabular_mlp_baseline seed=3 time=2025-12-05 01:27:07
- `20251205-012259-uf7_phase0_mlp_full_supervised` exp=phase0_offline_supervised dataset=summary model=20251205-012259-uf7_phase0_mlp_full_supervised seed=- time=2025-12-05 01:27:07
- `20251205-012259-thf_phase0_mlp_full_supervised` exp=phase0_offline_supervised dataset=INSECTS_abrupt_balanced model=tabular_mlp_baseline seed=1 time=2025-12-05 01:23:30
- `20251205-012259-thf_phase0_mlp_full_supervised` exp=phase0_offline_supervised dataset=summary model=20251205-012259-thf_phase0_mlp_full_supervised seed=- time=2025-12-05 01:23:30
- `20251205-012259-owj_phase0_mlp_full_supervised` exp=phase0_offline_supervised dataset=INSECTS_abrupt_balanced model=tabular_mlp_baseline seed=2 time=2025-12-05 01:23:29
- `20251205-012259-bup_phase0_mlp_full_supervised` exp=phase0_offline_supervised dataset=INSECTS_abrupt_balanced model=tabular_mlp_baseline seed=3 time=2025-12-05 01:23:29

### Top10 legacy 扁平日志（logs/<dataset>/*.csv）
- `logs/Airlines/Airlines__ts_drift_adapt_severity_s2p0__seed3.csv` dataset=Airlines model=ts_drift_adapt_severity_s2p0 seed=3 time=2025-12-03 18:59:49
- `logs/Airlines/Airlines__ts_drift_adapt_severity_s2p0__seed2.csv` dataset=Airlines model=ts_drift_adapt_severity_s2p0 seed=2 time=2025-12-03 18:59:34
- `logs/Airlines/Airlines__ts_drift_adapt_severity_s2p0__seed1.csv` dataset=Airlines model=ts_drift_adapt_severity_s2p0 seed=1 time=2025-12-03 18:59:19
- `logs/INSECTS_abrupt_balanced/INSECTS_abrupt_balanced__ts_drift_adapt_severity_s2p0__seed3.csv` dataset=INSECTS_abrupt_balanced model=ts_drift_adapt_severity_s2p0 seed=3 time=2025-12-03 18:59:04
- `logs/INSECTS_abrupt_balanced/INSECTS_abrupt_balanced__ts_drift_adapt_severity_s2p0__seed2.csv` dataset=INSECTS_abrupt_balanced model=ts_drift_adapt_severity_s2p0 seed=2 time=2025-12-03 18:58:56
- `logs/INSECTS_abrupt_balanced/INSECTS_abrupt_balanced__ts_drift_adapt_severity_s2p0__seed1.csv` dataset=INSECTS_abrupt_balanced model=ts_drift_adapt_severity_s2p0 seed=1 time=2025-12-03 18:58:49
- `logs/NOAA/NOAA__ts_drift_adapt_severity_s2p0__seed3.csv` dataset=NOAA model=ts_drift_adapt_severity_s2p0 seed=3 time=2025-12-03 18:58:41
- `logs/NOAA/NOAA__ts_drift_adapt_severity_s2p0__seed2.csv` dataset=NOAA model=ts_drift_adapt_severity_s2p0 seed=2 time=2025-12-03 18:58:35
- `logs/NOAA/NOAA__ts_drift_adapt_severity_s2p0__seed1.csv` dataset=NOAA model=ts_drift_adapt_severity_s2p0 seed=1 time=2025-12-03 18:58:29
- `logs/Electricity/Electricity__ts_drift_adapt_severity_s2p0__seed3.csv` dataset=Electricity model=ts_drift_adapt_severity_s2p0 seed=3 time=2025-12-03 18:58:24

## Step2：统一抽取关键指标（分类 + 漂移检测）

- 已生成：`UNIFIED_METRICS_TABLE.csv`
- acc_final 口径：优先 PhaseC `run_level_metrics.csv` 的 tail-mean；否则用日志 `metric_accuracy` 的 last-200 mean
- window-based 漂移指标：使用 `evaluation/drift_metrics.py`
- tolerance-based 漂移指标：tol=500, min_sep=200（复刻 `evaluation/phaseB_detection_ablation_synth.py::compute_metrics`）

### 精简版（legacy_flat 在线日志）：dataset×model 的 mean±std

| dataset | model | acc_final | mean_acc | MTR (window) | MTR_tol500 | n |
| --- | --- | --- | --- | --- | --- | --- |
| Airlines | ts_drift_adapt | 0.6350±0.0024 | 0.6372±0.0088 |  |  | 3 |
| Airlines | ts_drift_adapt_severity | 0.6362±0.0007 | 0.6397±0.0004 |  |  | 3 |
| Airlines | ts_drift_adapt_severity_s0p5 | 0.6354±0.0006 | 0.6406±0.0011 |  |  | 3 |
| Airlines | ts_drift_adapt_severity_s1p0 | 0.6348±0.0021 | 0.6367±0.0094 |  |  | 3 |
| Airlines | ts_drift_adapt_severity_s2p0 | 0.6366±0.0001 | 0.6414±0.0013 |  |  | 3 |
| Electricity | ts_drift_adapt | 0.5760±0.0191 | 0.5760±0.0191 |  |  | 3 |
| Electricity | ts_drift_adapt_severity | 0.5536±0.0101 | 0.5536±0.0101 |  |  | 3 |
| Electricity | ts_drift_adapt_severity_s0p5 | 0.5724±0.0066 | 0.5724±0.0066 |  |  | 3 |
| Electricity | ts_drift_adapt_severity_s1p0 | 0.5633±0.0075 | 0.5633±0.0075 |  |  | 3 |
| Electricity | ts_drift_adapt_severity_s2p0 | 0.5695±0.0152 | 0.5695±0.0152 |  |  | 3 |
| INSECTS_abrupt_balanced | baseline_student | 0.2656±0.0477 | 0.2476±0.0357 | 3.4894 |  | 3 |
| INSECTS_abrupt_balanced | mean_teacher | 0.2728±0.0399 | 0.2549±0.0269 | 3.3301 |  | 3 |
| INSECTS_abrupt_balanced | ts_drift_adapt | 0.1960±0.0052 | 0.1960±0.0043 | 3.5298±0.0571 | 150.9434 | 3 |
| INSECTS_abrupt_balanced | ts_drift_adapt_severity | 0.1933±0.0055 | 0.1927±0.0048 | 3.4426±0.0662 |  | 3 |
| INSECTS_abrupt_balanced | ts_drift_adapt_severity_s0p5 | 0.1907±0.0085 | 0.1901±0.0081 | 3.3584±0.1034 |  | 3 |
| INSECTS_abrupt_balanced | ts_drift_adapt_severity_s1p0 | 0.1977±0.0015 | 0.1973±0.0014 | 3.4894 |  | 3 |
| INSECTS_abrupt_balanced | ts_drift_adapt_severity_s2p0 | 0.1919±0.0050 | 0.1920±0.0049 | 3.4426±0.0662 |  | 3 |
| NOAA | ts_drift_adapt | 0.6766±0.0298 | 0.6766±0.0298 |  |  | 3 |
| NOAA | ts_drift_adapt_severity | 0.6197±0.0221 | 0.6197±0.0221 |  |  | 3 |
| NOAA | ts_drift_adapt_severity_s0p5 | 0.6682±0.0275 | 0.6682±0.0275 |  |  | 3 |
| NOAA | ts_drift_adapt_severity_s1p0 | 0.6565±0.0290 | 0.6565±0.0290 |  |  | 3 |
| NOAA | ts_drift_adapt_severity_s2p0 | 0.6436±0.0606 | 0.6436±0.0606 |  |  | 3 |
| sea_abrupt4 | baseline_student | 0.8571±0.0054 | 0.8168±0.0077 | 2.6665±0.4422 | 19.7499±5.9191 | 5 |
| sea_abrupt4 | mean_teacher | 0.8565±0.0079 | 0.8184±0.0093 | 2.7286±0.3026 | 19.1053±14.5958 | 5 |
| sea_abrupt4 | ts_drift_adapt | 0.8591±0.0148 | 0.8168±0.0157 | 2.2282±0.3322 | 26.8968±15.2499 | 6 |
| sea_abrupt4 | ts_drift_adapt_severity | 0.8660±0.0049 | 0.8242±0.0037 | 2.4519±0.3013 | 28.7399±30.5201 | 5 |
| sea_abrupt4 | ts_drift_adapt_severity_s1p0 | 0.8592±0.0090 | 0.8221±0.0073 | 2.5447±0.1775 | 15.8111±5.9620 | 5 |
| sine_abrupt4 | baseline_student | 0.6465±0.0147 | 0.7203±0.0097 | 2.7077±0.3697 | 19.9835±6.6757 | 5 |
| sine_abrupt4 | mean_teacher | 0.6256±0.0073 | 0.7106±0.0143 | 2.5611±0.4061 | 29.1130±18.3874 | 5 |
| sine_abrupt4 | ts_drift_adapt | 0.6446±0.0118 | 0.7224±0.0150 | 2.7574±0.3978 | 46.1009±50.0736 | 5 |
| sine_abrupt4 | ts_drift_adapt_severity | 0.6403±0.0134 | 0.7235±0.0203 | 2.7444±0.3005 | 15.4134±1.9866 | 5 |
| sine_abrupt4 | ts_drift_adapt_severity_s1p0 | 0.6427±0.0138 | 0.7211±0.0125 | 2.8487±0.1193 | 38.0743±49.8255 | 5 |
| stagger_abrupt3 | baseline_student | 0.9242±0.0058 | 0.9350±0.0132 | 74.7877±24.9380 | 74.7877±24.9380 | 5 |
| stagger_abrupt3 | mean_teacher | 0.9254±0.0034 | 0.9389±0.0075 | 77.2882±18.9806 | 77.2882±18.9806 | 5 |
| stagger_abrupt3 | ts_drift_adapt | 0.9315±0.0031 | 0.9468±0.0086 | 61.5052±16.9604 | 62.9515±18.6603 | 5 |
| stagger_abrupt3 | ts_drift_adapt_severity | 0.9315±0.0029 | 0.9447±0.0056 | 59.5977±19.3213 | 59.5977±19.3213 | 5 |
| stagger_abrupt3 | ts_drift_adapt_severity_s1p0 | 0.9319±0.0038 | 0.9455±0.0110 | 64.1997±15.5387 | 64.1997±15.5387 | 5 |

## Step3：重点排雷 ① —— NOAA severity_s2p0 seed=3 离群分析

### NOAA 各模型变体 acc_final（按 seed）
- `ts_drift_adapt`: seed1:0.6344(phaseC_run_level_tail_mean), seed2:0.6979(phaseC_run_level_tail_mean), seed3:0.6974(phaseC_run_level_tail_mean)
- `ts_drift_adapt_severity_s0p5`: seed1:0.6297(phaseC_run_level_tail_mean), seed2:0.6923(phaseC_run_level_tail_mean), seed3:0.6827(phaseC_run_level_tail_mean)
- `ts_drift_adapt_severity_s1p0`: seed1:0.6158(phaseC_run_level_tail_mean), seed2:0.6725(phaseC_run_level_tail_mean), seed3:0.6811(phaseC_run_level_tail_mean)
- `ts_drift_adapt_severity_s2p0`: seed1:0.6802(phaseC_run_level_tail_mean), seed2:0.6923(phaseC_run_level_tail_mean), seed3:0.5582(phaseC_run_level_tail_mean)

### 离群点（|acc - median| > 0.08）
- variant=`ts_drift_adapt_severity_s2p0` seed=3 acc_final=0.5582 vs median=0.6802

### 目标 run：`NOAA/ts_drift_adapt_severity_s2p0/seed3`
- log: `logs/NOAA/NOAA__ts_drift_adapt_severity_s2p0__seed3.csv`
- 说明：当前真实流 runs 为 legacy 扁平 CSV，通常没有 train.log/metrics.csv/summary.json；若需要这些产物建议用 run_id 结构补跑。
- 最近 200 行超参/严重度统计（min/max/mean）：
  - alpha: min=0.944317, max=0.994067, mean=0.968583
  - lr: min=0.000407812, max=0.000707812, mean=0.000619366
  - lambda_u: min=0.520625, max=0.695625, mean=0.577746
  - tau: min=0.855, max=0.942188, mean=0.867042
  - drift_severity: min=0, max=0, mean=0
  - drift_severity_raw: min=0, max=0, mean=0
  - monitor_severity: min=0, max=0.166667, mean=0.00234742
  - regime_changes_lastN: 6
  - drift_events_lastN: 2

- 日志尾部 20 行（关键列）：
```text
step, sample_idx, metric_accuracy, alpha, lr, lambda_u, tau, regime, drift_flag, drift_severity
52, 13311, 0.6364933894230769, 0.9599417970904498, 0.0006644531231955625, 0.5501562510104849, 0.8599218756495974, mild_drift, 0, 0.0
53, 13567, 0.6364239386792453, 0.9625458985452249, 0.0006572265615977813, 0.5550781255052424, 0.8574609378247987, mild_drift, 0, 0.0
54, 13823, 0.6353443287037037, 0.9638479492726124, 0.0006536132807988907, 0.5575390627526211, 0.8562304689123994, mild_drift, 0, 0.0
55, 14079, 0.6349431818181818, 0.9644989746363062, 0.0006518066403994454, 0.5587695313763106, 0.8556152344561997, mild_drift, 0, 0.0
56, 14335, 0.63623046875, 0.9648244873181531, 0.0006509033201997227, 0.5593847656881552, 0.8553076172280998, mild_drift, 0, 0.0
57, 14591, 0.6383634868421053, 0.9649872436590765, 0.0006504516600998614, 0.5596923828440776, 0.8551538086140499, mild_drift, 0, 0.0
58, 14847, 0.6375942887931034, 0.9650686218295382, 0.0006502258300499307, 0.5598461914220387, 0.855076904307025, mild_drift, 0, 0.0
59, 15103, 0.6381091101694916, 0.965109310914769, 0.0006501129150249654, 0.5599230957110193, 0.8550384521535125, mild_drift, 0, 0.0
60, 15359, 0.6386067708333333, 0.9651296554573845, 0.0006500564575124827, 0.5599615478555096, 0.8550192260767562, mild_drift, 0, 0.0
61, 15615, 0.6377433401639344, 0.9651398277286922, 0.0006500282287562414, 0.5599807739277547, 0.8550096130383781, mild_drift, 0, 0.0
62, 15871, 0.6365927419354839, 0.9651449138643461, 0.0006500141143781207, 0.5599903869638774, 0.8550048065191891, mild_drift, 0, 0.0
63, 16127, 0.6382688492063492, 0.965147456932173, 0.0006500070571890603, 0.5599951934819387, 0.8550024032595945, mild_drift, 0, 0.0
64, 16383, 0.639404296875, 0.9651487284660865, 0.0006500035285945303, 0.5599975967409694, 0.8550012016297972, mild_drift, 0, 0.0
65, 16639, 0.6380408653846154, 0.9651493642330432, 0.0006500017642972651, 0.5599987983704846, 0.8550006008148986, mild_drift, 0, 0.0
66, 16895, 0.638375946969697, 0.9651496821165215, 0.0006500008821486325, 0.5599993991852423, 0.8550003004074493, mild_drift, 0, 0.0
67, 17151, 0.6398087686567164, 0.9651498410582607, 0.0006500004410743164, 0.5599996995926211, 0.8550001502037247, mild_drift, 0, 0.0
68, 17407, 0.6395335477941176, 0.9651499205291303, 0.0006500002205371582, 0.5599998497963106, 0.8550000751018623, mild_drift, 0, 0.0
69, 17663, 0.638077445652174, 0.9651499602645651, 0.0006500001102685792, 0.5599999248981553, 0.8550000375509311, mild_drift, 0, 0.0
70, 17919, 0.638671875, 0.9651499801322825, 0.0006500000551342897, 0.5599999624490777, 0.8550000187754656, mild_drift, 0, 0.0
71, 18157, 0.6393875977530565, 0.9800749900661412, 0.0005250000275671448, 0.6299999812245388, 0.9000000093877328, stable, 0, 0.0
```

## Step4：重点排雷 ② —— 指标口径一致性验证（SEA vs STAGGER）

### SEA：sea_abrupt4 / ts_drift_adapt / seed=3
- log: `logs/sea_abrupt4/sea_abrupt4__ts_drift_adapt__seed3.csv`
- n_true_drifts=4, detections_raw=30, detections_merged(min_sep=200)=30
- window-based: MDR=0.000, MTD=935.0, MTFA=1894.4, MTR=2.026
- tol500-based: MDR=0.750, MTD=143.0, MTFA=1691.4, MTR=47.313

### STAGGER：stagger_abrupt3 / ts_drift_adapt / seed=3
- log: `logs/stagger_abrupt3/stagger_abrupt3__ts_drift_adapt__seed3.csv`
- n_true_drifts=2, detections_raw=15, detections_merged(min_sep=200)=14
- window-based: MDR=0.000, MTD=47.0, MTFA=3738.7, MTR=79.546
- tol500-based: MDR=0.000, MTD=47.0, MTFA=4078.5, MTR=86.778

### 量纲差异解释（简要）
- MTR 的主导项通常是 MTFA；当漂移间隔大或假警报很稀疏时，MTFA 会变大，从而让 MTR 显著变大。
- window-based 以“相邻真漂移之间”为窗口；窗口内第一个报警记为检测，其余报警计为假警报。
- tol500-based 只允许在 [drift, drift+500] 内匹配，并先做 min_sep=200 合并，对“报警过密”更敏感。

## Step5：实验下一步“最小补跑集合”建议

| dataset | model | monitor_preset | severity_scale | seeds | 预期回答的问题 |
| --- | --- | --- | --- | --- | --- |
| NOAA | ts_drift_adapt, ts_drift_adapt_severity_s0p5/s1p0/s2p0 | error_divergence_ph_meta | 0.5/1.0/2.0 | 1 2 3 4 5 | 复现/定位 NOAA severity 离群；检查 scale 是否单调影响 mean/final/drop |
| INSECTS_abrupt_balanced | ts_drift_adapt vs ts_drift_adapt_severity_s1p0 | error_divergence_ph_meta | 1.0 | 1 2 3 4 5 | 在有真值漂移的真实流上验证自适应是否提升 final_acc 与 MTR |
| sea_abrupt4 | ts_drift_adapt | error_ph_meta vs divergence_ph_meta vs error_divergence_ph_meta | - | 1 3 5 | 验证创新点1：detector preset 对 MDR/MTD/MTFA/MTR 的影响（最小对照） |

## Step6：输出物

- `LOCAL_PROGRESS_REPORT_v2.md`
- `RUN_INDEX.csv`
- `UNIFIED_METRICS_TABLE.csv`

## 容错记录

- 本次在 ZZNSTD（Python 3.10.19）下生成报告；解析 CSV/JSON 全程使用标准库 csv/json。
