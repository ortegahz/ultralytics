# 交接档 · 服务器夜间第 7 轮（2026-11-04 夜间 → 2026-11-04 上午 CST）

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e`
> **python**：`/home/huangzhe/anaconda3/envs/uav/bin/python`（conda env `uav`）
> **线程**：`01a199f3-5523-7591-b039-9e07b7a1fb58`（host 18:30 关机，早晨接回）
> **收尾时间**：2026-11-04 上午 CST · 收尾记录 `/mnt/data/codex-overnight/round10_final.log`
> **上位交接档**：`manu/memory/handoff_20261104_server_round9.md`（第 5/6 轮）

---

## 1. 一句话

**外挂已在物理极限** —— 用**三条互相独立**的证据判定：

| 证据 | 脚本 | 数字 | 含义 |
|:--|:--|:--|:--|
| 误差三分解 | `diag_error_decomp2.py` | 输出误差 p50 **1.670 px**（＝锚点继承 e(f) **1.566 px** 与位移残差 δ **0.870 px** 的**矢量和**）；零延迟管线自身 e(t) p50 **1.561 px**（8 序 25469 帧） | **外挂比零延迟管线只多 0.11 px**；δ(0.870) ≪ e(f)(1.566) ⇒ 该修**锚点（检测器）** |
| L2 不可救 | `diag_gap_coast.py` | 快核穿「检测器空窗」hit@5px：001324 **18.95%** / 000322 **17.80%** / 002111 **31.78%**，**全部低于 kinematic 臂 38.68%** | **7.06%** 的缺口来自**检测器边界**，不是外挂 |
| 误差不可平均 | （24 序 123966 对） | 管线误差向量帧间相关 ρ(lag1)=**+0.778**、ρ(lag8)=**+0.567** | 多源平均只降 ~16%，不值 3× 算力 |

⇒ **剩余 4.33 pp 命中缺口要靠换/改检测器，不是外挂**；**外挂旋钮已扫完，不再在外挂侧调参**。

本轮同时：**新档位 `stride`（分段配准）实现 ⇒ 实测 ⇒ 否决**；**报告口径两处硬伤修正**；**memory 导航表校验从 FAIL(88/112) 修回 PASS(112/112)**。

---

## 2. 交付档（本轮**未改**，仅复验）

```bash
bash manu/tracking/lookahead/run_delivery.sh both
```
（8 序 86.5 s / 24 序 460.8 s，服务器 test-Super-Server，2026-11-04 夜间）

| 口径 | 臂 | F1% | FAR/帧 | hit@5px | CLE p50(池) | p95(池) | max_jitter |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **24 序官方** | original(天花板/上界) | 94.7469 | 0.020820 | 85.8957 | 1.562 | 6.844 | 76.52 |
| | stale(老逻辑,真吃320ms,**交付基线**) | 77.5225 | 0.155917 | 44.8232 | 5.293 | 56.620 | 76.52 |
| | kinematic(纯前向外推) | 75.7641 | 0.169734 | 38.6764 | 6.421 | 61.517 | 188.95 |
| | **replay(交付档)** | **93.1879** | **0.029548** | **81.5612** | **1.719** | **12.946** | **60.00** |
| **8 序 gate** | original(天花板/上界) | 96.4002 | 0.006277 | 88.1316 | 1.568 | 5.970 | 65.24 |
| | stale(老逻辑,**交付基线**) | 77.2034 | 0.180743 | 38.1850 | 6.878 | 57.035 | 65.24 |
| | **replay(交付档)** | **95.4440** | **0.015381** | **85.2206** | **1.674** | **9.935** | **60.00** |

- **24 序红线 0.030 ⇒ 余量 +0.000452**
- 相对 `stale`：**F1 +15.67 pp / hit@5px +36.74 pp / FAR −0.126 / p95 −43.67 px / max_jitter −16.52 px**
- **320 ms 滞后账追回：24 序 F1 90.9% / hit 89.4%；8 序 F1 95.0% / hit 94.2%**
- 离零延迟天花板：24 序差 **F1 1.56 pp / hit 4.33 pp**

> ⚠️ 口径提醒（用户验收时最容易被做反的一处）：
> **`original` 臂是「零延迟天花板（oracle 上界）」，不是交付基线**；**交付基线是 `stale` 臂**（老逻辑真吃 320 ms 滞后）。

---

## 3. 新档位 `stride`（分段配准）：**已实现、已实测、已否决**

- **机理**：`direct` 跨 8 帧做一次大匹配 ⇒ 模板老化；`stride` 改成每 N 帧配一次、每段刷新模板。
- **实现**：`extrapolator.py` 新增 `_xcorr_stride_refine`；`LookaheadConfig.replay_xcorr_stride`；CLI `--replay-xcorr-stride`；`--replay-xcorr-mode` choices 增加 `stride`。单测 `manu/tracking/tests/test_xcorr_stride.py`。
- **★ 坑（务必记住）**：`xcorr_peak_last` 必须记为**各段峰值最弱值**，否则 `replay_abstain_peak_below` 弃权门**静默失效**（FAR 红线破防）。
- **生产端到端实测**（8 序，其余旋钮一字不改）—— **单调变差 ⇒ 否决**：

| 档位 | F1% | FAR/帧 | hit@5px | CLE p95(池) |
|:--|--:|--:|--:|--:|
| **direct（交付档）** | **95.4440** | **0.015381** | **85.2206** | **9.94** |
| stride N=1 | 95.2769 | 0.016415 | 85.0587 | 9.49 |
| stride N=2 | 94.6833 | 0.019623 | 84.3705 | 12.71 |
| stride N=4 | 94.2266 | 0.023589 | 83.2886 | 16.69 |

- **根因（本轮最有价值的认识）**：离线 bench 的 `direct` 用**锚点**当搜索中心（罩不住快速目标 ⇒ **严重低估 direct**）；
  **生产**的 direct 用**质心链末位置** `chain_pos` 当搜索中心。逐帧对比 direct vs stride1 位置差 **p50 仅 0.309 px**
  ⇒ 两者在生产里本来就几乎一样，stride 的额外结构只增加了出错面。
  ⇒ **离线 bench 若不与生产口径对齐，会给出完全误导的排序**（本轮差点据此砍掉冠军路线）。
- 产物：`runs/lookahead/final/final8stride/{arms_STR1,arms_STR2,arms_STR4}.json` + `preds_STR*`。

---

## 4. 失手漏斗（24 序交付档，GT 133257）

```
L1 源帧不存在 184 (0.14%)
 → L2 源帧无检测 9404 (7.06%)      ← 主要缺口，且不可救（见 §1）
 → L3 43
 → L4 能重推但未上报 1598 (1.20%)  ← 已在甜点（abstain 扫描复核）
 → L5 上报但位置错 14198 (10.65%)  ← (5,10] 6886 / (10,20] 2874 / (20,50] 2487 / >50 1951
 → 命中 108629 (81.52%)
```

- 重灾区序列（hit@5px 率）：015740 **6.98%** / 000322 **48.44%**（源无检 4563）/ 001324 65.24% / 002111 68.80% / 002001 73.06%（位置错 3356）
  ⇒ 这 5 条合计失手 **17265 / 24628 = 70.1%**；按失手帧排名的**前 5 条**（000322/002001/001324/002959/002438）= **19353 / 24628 = 78.6%**。
- **GT 逐帧位移 p50 仅 0.5~3.5 px（目标基本悬停）⇒ 高速机动不是主因**（推翻"高速帧是主因"的直觉）。
- 产物：`final24/diag_miss_funnel_rb16.json`、`final24/diag_gap_coast.json`、`final8/diag_error_decomp2.json`（8 序版漏斗：`final8/diag_miss_funnel.json`）。

---

## 5. Gate A 复验（尺子1 逐位一致）—— PASS

| 口径 | 参照 | md5 比对 |
|:--|:--|:--|
| 8 序 | `runs/lookahead/ref/preds_original_v3/`（9 份） | **IDENTICAL=9 DIFFER=0 MISSING=0** |
| 24 序 | `runs/lookahead/final/ref_preds_original_v3_24seq/`（25 份） | **IDENTICAL=25 DIFFER=0 MISSING=0** |

- 脚本：`/mnt/data/codex-overnight/gateA_newcode.sh`；产物 `runs/lookahead/final/gateA_newcode/{arms_off_8seq,arms_off_24seq}.json`。
- **⚠️ 坑**：第一次 8/24 全 DIFFER，根因是 `--cache-root` 传**相对路径**（参照产物记的是绝对路径）
  ⇒ `config.cache_root` 字符串不同 ⇒ md5 不同，而 `per_frame` 实际 **0 帧不同**。
  ⇒ **Gate A 的 md5 含 `config` 块，路径必须是绝对路径。**
- 单测：`pytest manu/tracking/tests -q` → **155 passed / 10 skipped**（无回归）。

---

## 6. OSD 三臂视频（当前交付档重渲）

`runs/lookahead/osd/videos/<SEQ>_3arm_{stale,replay,original}.mp4`（脚本 `/mnt/data/codex-overnight/render_osd_one.sh`）

| 序列 | 窗口 | stale hit5 | replay hit5 | original hit5 |
|:--|:--|--:|--:|--:|
| 001324 | f8549–8848 | **19.7%**（CLE p50 20.93 / max 192.24） | **90.3%**（p50 1.98 / max 10.47） | 98.7% |
| 015000 | f726–1025 | **25.0%**（p50 12.21 / max 102.97） | **97.0%**（p50 1.85） | 99.0% |

画面可直接看出「**老逻辑抽搐脱靶 vs 外挂平滑跟住**」。

---

## 7. ★ 本轮修掉的 4 个硬伤（都会让结论做反）

| # | 硬伤 | 症状 | 修法 | 影响面 |
|:--|:--|:--|:--|:--|
| 1 | **报告总体行 CLE 用错口径** | 24 序 replay 报 `CLE p50 = 146.94 px`，与 `hit@5px = 81.56%` 自相矛盾 | `make_report_table.py` 总体行改用 `cle_pooled_*`（**全帧池化**）；旧的 `cle_p*` 是**逐序列最差值** worst_over_sequences | 24 序 replay p50 **146.94 → 1.72 px**；p95 **287.88 → 12.95 px** |
| 2 | **退化/收益小节跨产物混用基线** | 8 序 replay 拿 **24 序 stale** 当基线，算出无意义的「追回 **104.0%**」 | 基线与被比臂必须**同产物**；退化表按产物分组 | 8 序追回改为 **F1 95.0% / hit 94.2%**（24 序仍 90.9% / 89.4%） |
| 3 | **报告同名臂重复行** | 24 序 replay 与 8 序 replay 都叫 `replay(视觉重推)`，读者无从分辨 | 摘要表/退化表/尾部效应一律用「产物 · 臂」全名 | 全部对照表 |
| 4 | **memory 导航表校验 FAIL** | `validate_nav_table.py` ⇒ **88/112**（改动前 110/110 PASS），`memory.md` 组 24 条全 FAIL | 根因：新加的 §30.9.28 导航行写 `6020`，而 `memory.md` 组的**自洽 offset 是 −410**（唯一解）⇒ 映射到 local 6430 > 文件 6099 行。改成 `6020 − 410 = **5610**` | **112/112 PASS** |

> 修 #4 的可复现推理：`memory.md` 组 24 行的 offset 候选里，**−410 是唯一被每行同时命中的值**；
> 新增行必须落在同一 offset 下 ⇒ 行号 = 标题实际行号 − 410（与 §30.9.27 的 `5900 − 410 = 5490` 同约定）。

---

## 8. 本轮改动文件清单（全部在服务器磁盘上）

**新增**
| 路径 | 说明 |
|:--|:--|
| `manu/tracking/lookahead/diag_error_decomp2.py` | 误差三分解 |
| `manu/tracking/lookahead/diag_miss_funnel.py` | 失手漏斗（逐层） |
| `manu/tracking/lookahead/diag_gap_coast.py` | L2「检测器空窗」不可救 |
| `manu/tracking/tests/test_xcorr_stride.py` | stride 档位单测 |
| `/mnt/data/codex-overnight/{sweep8_stride.sh,gateA_newcode.sh}` | 复现脚本 |
| `manu/memory/handoff_20261104_server_round10.md` | 本档 |

**修改**
| 路径 | 改动 |
|:--|:--|
| `manu/tracking/lookahead/extrapolator.py` | 新增 `_xcorr_stride_refine`（stride 档位，已否决但保留可复现） |
| `manu/tracking/lookahead/lookahead_longquan.py` | `--replay-xcorr-stride` / `--replay-xcorr-mode stride` |
| `manu/tracking/lookahead/make_report_table.py` | 总体行 CLE 改池化口径 + 按产物分组 + 全名消歧（见 §7 #1~#3） |
| `manu/tracking/lookahead/bench_xcorr_estimators.py` | 修真 bug：`results` 定义在序列循环外 ⇒ 所有序列显示最后一条数字 |
| `manu/memory/lookahead_extrapolator_20261102.md` | 新增 **§34**（15 个子节，4323 行起） |
| `manu/memory/memory.md` | 新增 **§30.9.28**（6020 行起）+ 导航条目 |
| `manu/tracking/lookahead/README.md` | 新增 **§5.18** |

**产物**
| 路径 | 说明 |
|:--|:--|
| `runs/lookahead/final/final8/arms_delivery.json` + `preds_delivery/` | 8 序交付档（86.5 s） |
| `runs/lookahead/final/final24/arms_delivery.json` + `preds_delivery/` | 24 序交付档（460.8 s） |
| `runs/lookahead/final/final8stride/arms_STR{1,2,4}.json` + `preds_STR*` | stride 扫描（**已否决**） |
| `runs/lookahead/final/gateA_newcode/{arms_off_8seq,arms_off_24seq}.json` | 尺子1 逐位复验 |
| `runs/lookahead/final/report_delivery.md` | ★ 最终对照表（已按池化口径重生成） |
| `runs/lookahead/final/final8/diag_error_decomp2.json` + `final8/diag_miss_funnel.json` + `final24/diag_miss_funnel_rb16.json` + `final24/diag_gap_coast.json` | 三个诊断 |
| `runs/lookahead/osd/videos/*_3arm_*.mp4` | 三臂 OSD 视频 |

---

## 8b. ★ host 同步包（换机直接用）

```
runs/lookahead/final/code_sync_20261104.tar.gz
```

**权威 md5 看 sidecar**（不内嵌，避免自引用）：`runs/lookahead/final/code_sync_20261104.tar.gz.md5`
（包内文件清单 md5：`runs/lookahead/final/code_sync_20261104.manifest.md5`）。

**包内自洽可验**：`tar xzf` 后直接 `python3 manu/memory/validate_nav_table.py` ⇒ **112/112 PASS**
（已把 `manu/memory/archive/` 00..07、顶层全部专题档、`validate_nav_table.py` 一并打进包；
否则换机后校验器会报「归档文件不存在」⇒ 38/112）。
重建命令：`bash /mnt/data/codex-overnight/build_sync_pkg.sh`。

---

## 9. ⛔ 新增「不要再试」清单（既有清单仍然有效）

1. **`stride` 分段配准**（生产口径）
2. **多源回推取中位数**（误差帧间强相关 ρ(lag1)=+0.778，收益 ~16% 不值 3× 算力）
3. **快核穿「检测器空窗」**（L2 层不可救：18.95%/17.80%/31.78% < kinematic 臂 38.68%）
4. 既有：观测层收缩 / 位移层收缩 / 低速冻结 / 位移自适应融合 / peak_centroid / refine / 一致性门单独用 /
   `--replay-anchor pred` / `--fast-min-contrast` / rolling 链 / core_frac / th_k≥2.5 / 大半径重试 / 各类单门 / chain 家族

---

## 10. 下一步（给 host 侧）

1. **不要再在外挂侧调参** —— 已被三条独立证据封死。下一步若要继续提精度，**方向是换/改检测器**
   （尤其 000322 / 002111 / 001324 的**源帧漏检**，以及 002001 的**上报位置错 3356 帧**）。
2. **本轮结论已可直接交付评审**：`report_delivery.md` + `README.md §5.18` + `memory §34` 三件套齐了。
3. **口径统一提醒**：凡引用 CLE 数字，必须写明是**池化**（`cle_pooled_*`）还是**逐序列最差值**（`cle_p*`）。
4. **换机复跑**：`bash manu/tracking/lookahead/run_delivery.sh both`（服务器，conda env `uav`）；
   Gate A 用 `/mnt/data/codex-overnight/gateA_newcode.sh`（**记得 `--cache-root` 必须是绝对路径**）。
