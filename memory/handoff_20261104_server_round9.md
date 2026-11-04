# 交接档 · 服务器夜间第 5 / 6 轮（2026-11-04 18:30 → 次日 07:50 CST）

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e`
> **python**：`/home/huangzhe/anaconda3/envs/uav/bin/python`（conda env `uav`）
> **线程**：`01a199f3-5523-7591-b039-9e07b7a1fb58`（host 18:30 关机，早晨接回）
> **收尾时间**：2026-11-04 07:50 CST · 收尾记录 `/mnt/data/codex-overnight/round9_final.log`

## 1. 一句话

**§32 的 8 序冠军 `CD_MP060` 在 24 序官方口径上被证伪**（FAR 0.045260 破红线 0.030、hit@5px −2.87 pp、F1 −1.77 pp）
⇒ 它是 **8 序过拟合**。**2×2 因子隔离**进一步证明：**FAR 破线的元凶是 `chain` 档位本身**，与 ROI 前推上限无关。
据此把交付档升级为 **`direct` + `--replay-roi-selfpred-max-px 16`** ⇒ **FAR 红线余量 0.000162 → 0.000452（×2.8）**。

## 2. 交付档最终形态（唯一改动：`--replay-roi-selfpred-max-px 4 → 16`）

```bash
bash manu/tracking/lookahead/run_delivery.sh both
```

| 口径 | 臂 | 尺子2 F1% | FAR/帧 | hit@5px | CLE p50(池) | CLE p95(池) | max_jitter |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **24 序官方** | original(天花板) | 94.7469 | 0.020820 | 85.8957 | 1.56 | 6.84 | 76.52 |
| | stale(老逻辑,真吃 320 ms) | 77.5225 | 0.155917 | 44.8232 | 5.29 | 56.62 | 76.52 |
| | kinematic(纯前向外推) | 75.7641 | 0.169734 | 38.6764 | 6.42 | 61.52 | 188.95 |
| | **replay(交付档)** | **93.1879** | **0.029548** | **81.5612** | 1.72 | 12.95 | **60.00** |
| **8 序 gate** | original(天花板) | 96.4002 | 0.006277 | 88.1316 | 1.57 | 5.97 | 65.24 |
| | stale(老逻辑) | 77.2034 | 0.180743 | 38.1850 | 6.88 | 57.04 | 65.24 |
| | **replay(交付档)** | **95.4440** | **0.015381** | **85.2206** | 1.67 | 9.94 | **60.00** |

- **24 序红线 0.030 ⇒ 余量 +0.000452**
- 相对 `stale`：**F1 +15.67 pp / hit@5px +36.74 pp / FAR −0.126 / p95 −43.67 px / mj −16.52 px**
- **320 ms 滞后账追回：F1 90.95%、hit@5px 89.45%**；离零延迟天花板只差 **F1 1.56 pp / hit 4.33 pp**

## 3. 三道 Gate 全部 PASS（换机后可直接复跑）

| Gate | 内容 | 结果 |
|:--|:--|:--|
| Gate A（8 序） | `--lookahead off` vs `runs/lookahead/ref/preds_original_v3` 逐位 | **IDENTICAL=9 DIFFER=0**，ref/off F1 均 0.964002 |
| Gate A（24 序） | 同上 24 序 | **IDENTICAL=25 DIFFER=0**，ref/off F1 均 0.947469 |
| Gate B（本轮新增） | 当前 `run_delivery.sh` CFG 重跑 vs 原产物 | 三臂 × 两组指标全 **IDENTICAL**；**9/9 per-frame JSON 逐字节相同** |
| 单测 | `pytest manu/tracking/tests` | **151 passed / 10 skipped / 3 subtests**（无回归） |

## 4. 本轮最重要的方法论结论（写给下一轮）

1. **8 序 gate 集只覆盖 24 序的 20.39% GT / 17.55% 帧** ⇒
   **8 序只能做「机理是否存在」的快速筛选，不能用来排序冠军**。
   chain 爆 FAR 的重灾区（015740 FAR 0.40 / 002251 FAR 0.10 / 002111 / 002438 / 001810）**全在 gate 集之外**。
2. **「不上报」是 hit@5px 的大漏斗**：8 序 GT 27173 帧里 **1619 帧弃权（5.96%）**、2397 帧上报但落错（8.82%）
   ⇒ 若弃权帧全部命中，hit 上限 **91.18%**（当前 85.22%）。FAR 红线是约束它的那个闸门。
3. **可见性天花板**：真实 headroom 只有 **+3.52 pp**（8 序可见帧内：original 93.53% vs replay 90.01%），
   不是「88.13 − 85.22 = 2.91 pp」。**001324 的病根在检测器本身**（当前帧不可见率 14.85%、源帧漏检 15.3%）。
4. **min_peak 在 chain 类档位越高越差**（逐段判据，任一段峰值低 ⇒ 整链作废 ⇒ 该帧不上报 ⇒ 必然脱靶）
   ⇒ 正确的对比轴是 **(档位 × min_peak) 二维网格**。
5. **xcorr 的半像素约定**：`make_template` 必须返回 `(weights, origin)`（模板左上角绝对坐标），
   `match()` 上报位置 = **anchor + 模板左上角位移**（纯位移口径）。

## 5. 关键路径（服务器）

| 用途 | 路径 |
|:--|:--|
| 项目根 | `/tmp/pycharm_project_10ae9e2e` |
| python | `/home/huangzhe/anaconda3/envs/uav/bin/python`（conda env `uav`） |
| 8 序 gate 缓存 | `runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache` |
| 24 序缓存 | `runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache` |
| GT（**平铺** `<SEQ>__<idx>.txt`） | `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010` |
| 帧 | `/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg` |
| **帧缓存（逐位安全）** | `/mnt/data/siping/frame_cache_lq/`（24 条 .npy + .json） |
| 交付参数 | `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json` |
| 交付档产物 | `runs/lookahead/final/final8/arms_delivery.json` · `runs/lookahead/final/final24/arms_delivery.json` |
| 上一版交付档备份 | `runs/lookahead/final/_prev_champion_backup/arms_delivery_{8,24}seq_rb16mp090.json` |
| OSD 三臂视频 | `runs/lookahead/osd/videos/<SEQ>_3arm_{stale,replay,original}.mp4`（8 序 × 3 臂 = 24 段） |
| 文档 | `manu/memory/lookahead_extrapolator_20261102.md` §32 / §33 · `manu/memory/memory.md` §30.9.26 / §30.9.27 |

## 6. 复现入口（服务器 `/mnt/data/codex-overnight/`）

```bash
# 交付档（8 序 + 24 序）
bash manu/tracking/lookahead/run_delivery.sh both
# 24 序候选扫描 / 2×2 因子隔离 / ROI 前推扫描
bash /mnt/data/codex-overnight/sweep24_final.sh
bash /mnt/data/codex-overnight/sweep24_factorial.sh
bash /mnt/data/codex-overnight/sweep24_roimax.sh
# 汇总对照 / Gate B / 8 序 roimax16
python3 /mnt/data/codex-overnight/cmp24_sweep.py
bash /mnt/data/codex-overnight/verify_delivery_unchanged.sh
bash /mnt/data/codex-overnight/test8_roimax16.sh
# 重建 host 同步包
bash /mnt/data/codex-overnight/build_sync_pkg.sh
```

## 7. 尚未解决 / 下一步（按优先级）

1. **24 序 FAR 红线余量只有 0.000452** ⇒ 任何新机制先过 FAR 再谈 hit。
2. **剩余 hit 缺口**（8 序可见帧内 +3.52 pp / 24 序 +4.33 pp）需要**先降 FAR** 才能换 hit。
3. **001324 已到检测器改动的边界**（当前帧不可见率 14.85%）⇒ 想再涨 hit，得先动检测器（漏检率 / 帧率），
   不是外挂能解决的。
4. **`diag_disp_radius.py` 的 `--original` 参数当前未被使用**（审计代理发现）⇒ 可删或真正用于锚点分析。
5. **诊断脚本的 Δ 空间不统一**：`diag_disp_radius.py` / `diag_oracle_split.py` 的源帧用**视频帧号**空间，
   臂用 **record 序号**空间（001324 有 358 帧错位，影响 0.01~0.11 pp，方向不变）⇒ 后续统一。
6. **`ruff` 环境未装**；`diag_disp_radius.py:191` 有 1 处 124 字符超长行（超 pyproject 的 120）。

## 8. 铁律（下一轮照抄）

- **重活只在服务器跑**（80 核 / 125 GB）；host 内存与 CPU 有限，**host 不要跑重活**。
- **24 序调参才是排序依据**；8 序只做机理筛选。
- **关开关必须逐位一致**：任何新旋钮默认值必须是「关」；`--lookahead off` 的产物必须与
  `runs/lookahead/ref/preds_original_v3/` 逐位相同。
- **不 commit / 不 push**；**不修改 `manu/evaluation/*` 原文件**（外挂全在 `manu/tracking/lookahead/`）。
- CLI 参数名一律 grep 自 `manu/tracking/lookahead/lookahead_longquan.py`。
- 每个结论必须带：**数字 + 完整命令 + 执行机 + 产物路径 + 时间戳**。
- 已被实测否决、不要再试的旋钮家族：观测层收缩 / 位移层收缩 / 低速冻结 / 位移自适应融合 / peak_centroid /
  refine / 一致性门单独用 / `--replay-anchor pred` / `--fast-min-contrast` / rolling 链 / core_frac /
  th_k≥2.5 / 大半径重试(r48/64/96/128) / 唯一性比单门 / 模板位置-链位置分歧单门 / 链位移单门 / 峰值单门 /
  二次意见采纳就弃权 / 精修失败就弃权 / dev 单门 / disp 单门 / rematch 弃权 / xcorr_fail 弃权 /
  换运动学搜索中心 / 纯峰值门 / **chain 家族（§33.8 已在 2×2 因子隔离上证明其破 FAR 红线，见 §33.8）**。
