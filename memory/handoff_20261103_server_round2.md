> ⚠️ **本文已被下一轮取代**：2026-11-04 服务器夜间第 1 轮的交接说明见
> **`manu/memory/handoff_20261104_server_round3.md`**（新冠军：8 序列 F1 **95.266%** / hit@5px **85.17%** / FAR **0.0163**）。

# 交接说明 · 2026-11-03 夜间第 3 段（服务器）→ host

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e`
> **python**：`/home/huangzhe/anaconda3/envs/uav/bin/python`（conda env `uav`）
> **线程 id**：`01a199f3-5523-7591-b039-9e07b7a1fb58`
> **host 关机**：2026-11-03 18:30（host 侧本轮没有任何写入）
> **本轮时段**：2026-11-03 20:50 – 22:20（服务器夜间自主推进 · 第 1 轮）

## 一、结论速览（本轮有**重大进展**）

| 口径 | `original`（天花板，**不是基线**） | `stale`（**交付基线** = 老逻辑） | `replay` 亮假设（旧交付 `WTH15`） | **`replay` auto + ratio10（本轮）** |
|:--|--:|--:|--:|--:|
| **8 序列**（官方 gate 集）F1 / hit@5px / FAR | 96.40 / 88.13 / 0.0063 | 77.20 / 38.18 / 0.1807 | 95.077 / 84.67 / 0.0180 | **95.062 / 84.96 / 0.0182** ✅ 三闸全过 |
| **24 序列**（官方口径）F1 / hit@5px / FAR | 94.75 / 85.90 / 0.0208 | 77.52 / 44.82 / 0.1559 | 90.46 / 77.51 / **0.0542** ❌ | **92.828 / 81.55 / 0.0356** ⚠️ 超线从 80.8% → 18.7% |

- **本轮新增机理 = 目标极性（亮/暗）自适应**：上一段归因出的「24 序列 FAR 缺口的 54.9% 来自 `014453`」
  **已被这一条补丁基本抹平**（`014453` FAR 0.3608 → **0.0021**，F1 63.49% → **99.75%**）。
- **交付配置 = 冠军 `WTH15` + `--replay-polarity auto`**（其余参数一字不改）。
  默认 `--replay-polarity bright` = 历史行为 ⇒ **逐位不变**。
- **★ 纠正上一段的错误前提**：极性是**逐帧**属性，**不能按序列分类**
  （`015000` 只有前 175 帧是暗的、`014227` 末 240 帧是暗的；旧探针按文件名字典序取样 ⇒ 只采到序列开头 ⇒ 误判）。

## 二、本轮（20:50–22:20）做了什么

1. **★ 新机理：目标极性自适应**（新增 `manu/tracking/lookahead/polarity.py`）
   - 快核与 xcorr 模板都写死了「比背景亮」；暗目标上权重恒为 0、xcorr 模板为空 ⇒ 配准峰锁到最亮的云边。
   - 合成图实证 + **归一化相关的符号坑**（暗极性必须把模板整体取反）。
   - 实现：`--replay-polarity {bright,dark,auto}`（默认 `bright`）+ `--replay-polarity-ratio`（默认 **10.0**）。
2. **★ 24 序列复评**：F1 90.461 → **92.828%**、FAR 0.0542 → **0.0356**、hit@5px 77.51 → **81.55%**、
   池化 CLE p95 28.48 → **15.16**、FP 8963 → **5889**。
3. **★ 8 序列复评**：hit@5px 84.67 → **84.96%**，F1 −0.015 pp，FAR +0.0002 ⇒ **三闸仍全过**；
   7 条亮目标序列**逐位不变**。
4. **置信倍数扫描**（4 档 × 11 序列）：**ratio = 10 最优**（`001810` 的 68 条误判链 → 2 条、
   `014453` 保留 8318/8319 条暗链）；ratio 30 反而伤到 `014453` 最弱的链。
5. **16 段 OSD 四臂视频**（4 序列 × 4 臂）—— `014453` 窗口上 **旧 replay hit@5px 9.0% → 新 replay 100.0%**。
6. **回归防线**：**Gate A 8 序列 9/9 IDENTICAL / 0 DIFFER**（F1 均 0.964002）+
   **24 序列 25/25 IDENTICAL / 0 DIFFER**（F1 均 0.947469）⇒ 零回归；
   新增 `manu/tracking/tests/test_polarity.py`（**7 用例**）+ 既有套件 ⇒ **70 passed**；
   一键复现 `bash manu/tracking/lookahead/run_delivery_polarity.sh [8|24|both]`。
7. **文档**：专题档 **§25**（283 行）· 主档 **§30.9.20** · README **§5.9 / §3 / §6 / §7 / §8**；
   导航表 `[PASS] **95/95**`（原 93/93）。
8. **同步包重建**：151 MB / **328 文件**，md5 `181afa750909957839c463fc8d291280`
   （含 `polarity.py` / `test_polarity.py` / `run_delivery_polarity.sh` / 补齐后的三处文档）。

## 三、早上 host 接回要做的三件事

```bash
# ① 把本轮产物拉回 host
scp huangzhe@192.168.99.40:/tmp/pycharm_project_10ae9e2e/runs/lookahead/final/code_sync_20261103.tar.gz .
tar xzf code_sync_20261103.tar.gz
#    md5 见 runs/lookahead/final/code_sync_20261103.tar.gz.md5（服务器侧 181afa750909957839c463fc8d291280）
#    服务器侧备份：/mnt/data/codex-overnight/backup_20261103/

# ② 同步 host 侧代码（host 的 xcorr.py / extrapolator.py / lookahead_longquan.py 仍是**旧的**，
#    且**没有** polarity.py）⇒ 从解包出来的 manu/tracking/lookahead/ 覆盖过去

# ③ 恢复会话
export CODEX_HOME=$HOME/.codex
codex resume 01a199f3-5523-7591-b039-9e07b7a1fb58
```

## 四、下一步（按优先级）

1. **24 序列 FAR 还差 0.0056**（0.0356 vs 红线 0.030）—— 剩余主体是**管线自身**问题
   （`015740` 0.4428 / original 0.4277、`002251` 0.1054 / original 0.0997、`003002` 0.0707、`002111` 0.0651）
   ⇒ 属于扩展缺口，不为此牺牲 hit@5px。
2. **极性方向的两个小补丁**（见专题档 §25.12）：
   ① 暗/亮**切换**时沿用上一链的极性（时间平滑）⇒ 修 `015000` 的 F1 −0.78 pp；
   ② `min_energy` / `min_contrast` **按极性分别设阈值**。
3. `max_jitter` 122.2 px（24 序列）⇒ 板端加一级输出限速/低通。
4. 001324 重捕获（需动快核本体）。
5. 板端移植契约（xcorr 精修 + **极性自适应** + 输出限幅 + 兜底策略 + 无状态性）。

## 五、关键产物路径（服务器磁盘）

| 产物 | 路径 |
|:--|:--|
| **8 序列主产物（本轮）** | `runs/lookahead/final/pol8f/arms_F1_auto10.json` + `preds_F1_auto10/`（亮基线 `arms_F0_bright.json`） |
| **24 序列主产物（本轮）** | `runs/lookahead/final/pol24f/arms_F1_auto10.json` + `preds_F1_auto10/`（亮基线 `pol24/arms_R0_bright.json`） |
| 极性置信扫描 | `runs/lookahead/final/polratio/arms_r{1,3,10,30}.json` |
| 逐链诊断 | `runs/lookahead/final/poldiag/arms_diag.json`（`replay_polarity_chains` / `_dark` / `_ratio_ge{3,10,30}`） |
| `014453` 单序列 | `runs/lookahead/final/pol14453/arms_Q{0,1,2,3}*.json` |
| **OSD 四臂视频** | `runs/lookahead/osd/videos/<SEQ>_pol_{stale,replay_old,replay_new,original}.mp4`（4 序列 × 4 臂 = 16 段） |
| 上一轮 FAR 归因 | `runs/lookahead/final/diag_far_breakdown.{json,md}` · `diag_014453*` · `sim_policy_offline.{json,md}` |
| 同步包 | `runs/lookahead/final/code_sync_20261103.tar.gz`（151 MB / **328 文件**，md5 `181afa750909957839c463fc8d291280`）+ `.manifest.md5` |
| 备份 | `/mnt/data/codex-overnight/backup_20261103/` |
| memory | `manu/memory/memory.md` **§30.9.20** · `manu/memory/lookahead_extrapolator_20261102.md` **§25** · `manu/tracking/lookahead/README.md` **§5.9** |

## 六、本轮代码基线（md5）

| 文件 | md5 | 状态 |
|:--|:--|:--|
| `manu/tracking/lookahead/polarity.py` | `2cd585ad87bad8438f6ecfc7272f0553` | **新建** |
| `manu/tracking/lookahead/xcorr.py` | `56bcc444a250988ebab57a377434d7c1` | 改（加 polarity） |
| `manu/tracking/lookahead/extrapolator.py` | `b0e010c2c326241392e85e1407f33985` | 改（接入极性） |
| `manu/tracking/lookahead/lookahead_longquan.py` | `5bc5ec136d7c02eac51639134833b09a` | 改（加 CLI） |
| `manu/tracking/lookahead/run_delivery_polarity.sh` | `b42701f930a5ac06c1fce4c7148b0870` | 新增（一键复现 8/24/both） |
| `manu/tracking/tests/test_polarity.py` | `338779b0278d2e61aad27ffeaf17e2bd` | 新增（7 用例） |
| `manu/evaluation/system_sota_longquan.py` | `3361e7b70e76d3a92346fd8fd9879dad` | **零改动**（mtime ≤ 10/22） |
