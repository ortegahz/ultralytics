# 交接说明 · 2026-11-04 服务器夜间第 1 轮（服务器）→ host

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e`
> **python**：`/home/huangzhe/anaconda3/envs/uav/bin/python`（conda env `uav`）
> **线程 id**：`01a199f3-5523-7591-b039-9e07b7a1fb58`
> **host 关机**：2026-11-03 18:30（host 侧本轮没有任何写入）
> **本轮时段**：2026-11-03 23:20 → 2026-11-04 02:00（服务器夜间自主推进 · 第 1 轮）
> **上一轮交接**：`handoff_20261103_server_round2.md`（已被本文取代）
> **⚠️ 本文已被 `handoff_20261104_server_round4.md` 取代**（2026-11-04 服务器夜间第 2 轮：「不自信就不上报」，24 序列 FAR 缺口关闭）

## 一、结论速览（本轮**再进一步**）

| 口径 | `original`（天花板，**不是基线**） | `stale`（**交付基线** = 老逻辑） | 上一档（极性自适应） | **本轮新冠军（唯一性二次意见 τ=0.6）** |
|:--|--:|--:|--:|--:|
| **8 序列** F1 / hit@5px / FAR | 96.400 / 88.13 / 0.0063 | 77.203 / 38.18 / 0.1807 | 95.062 / 84.96 / 0.0182 | **95.266 / 85.17 / 0.0163** ✅ 三闸全过 |
| **24 序列** F1 / hit@5px / FAR | 94.747 / 85.90 / 0.0208 | 77.523 / 44.82 / 0.1559 | 92.828 / 81.55 / 0.0356 | **92.924 / 81.64 / 0.0349** ⚠️ 超线 18.7% → **16.4%** |

- **本轮新增机理 =「相关峰唯一性比 + 二次意见重配」**（新旋钮 `--replay-xcorr-uniq-gate`，**默认 0.0 = 关**）。
- **交付配置 = 上一档 + `--replay-xcorr-uniq-gate 0.6`**（其余参数一字不改）。
  一键：`bash manu/tracking/lookahead/run_delivery.sh [8|24|both]`。
- **相对上一档**：8 序列 F1 **+0.205 pp** / FAR **−10.2%** / CLE p95 **−2.91 px** / p99_jitter **−3.87 px**；
  24 序列 F1 **+0.096 pp** / FAR **−2.0%** / max_jitter **−13.15 px**，逐序列 **ΔFP 合计 −125**。
- **F1 追回 94.09% 的 320 ms 滞后损失**（上一档 93.04%）；相对零延迟天花板只差 **1.13 pp**。

## 二、本轮做了什么

1. **★ walk-off 机理定量归因**（24 序列 8 条、51205 帧离线诊断，脚本 `/tmp/diag_walkoff.py` + `/tmp/diag_uniq.py`）
   - **位移测量本身很准**：好帧方向余弦 p50 **0.983–0.994**、实测位移 ≈ 真位移。
   - **坏帧不是「卡住不动」，是「链锁到错误特征后一路跑飞」**：实测位移 22.9–51.3 px（比真位移还大）、
     方向乱、10–28% **反向**、与同帧 `original` 差 25–39 px。
   - ⇒ **改位移精度（换匹配器/调模板/调核）都治不了它**；必须能识别「这次匹配不可信」。
2. **★ 由此一次性关掉三个历史旋钮家族（都带上了数字）**
   - 一致性门（`|d_meas−d_kin|>20`）：门内 16,401 帧 ⇒ **ΔFP +6822**（净负）
   - 低速冻结（`|d_meas|>30 且 |d_kin|<3`）：门内 659 帧 ⇒ **ΔFP +474**（净负）
   - 位移自适应融合：24 序列旧准则 τ=0.5 ⇒ F1 92.734 / FAR 0.0364 / hit 81.43（净负）
   - **「回退锚点」形态**：8 序列 τ=0.3 ⇒ **F1 95.062 → 87.03、hit@5px 84.96 → 67.20、FAR 0.0182 → 0.0913**（灾难性）
   - 根因：判定「目标很慢」的速度估计不可靠（001810 上锚点能救回 95% 坏帧，但同一门在别处误伤 **427** 个好帧）。
3. **★ 唯一可用的无状态离群信号：相关峰唯一性比**
   `uniqueness = 次峰/最佳峰`（次峰须排除与最佳峰重叠邻域）：TP 帧 p50 **0.000** / FP 帧 p50 **0.593**。
   判别力：τ=0.6 抓 49.4% 坏帧 / 误伤 3.9% 好帧。
4. **★ 落地形态：二次意见（不是回退）—— 最坏是「无变化」，不可能回归**
   歧义时以**锚点**为搜索中心**重配一次**，**三条同时满足才采纳**（重配唯一 `uniq2 ≤ τ` ∧ 重配更自信 `peak2 > peak` ∧ 峰值达标）；
   否则**保持原结果不动**。**② 不能省**：省掉时高速目标被「更靠近锚点但错误」的结构骗走
   （24 序列 τ=0.5：002438 FP **+69**、000322 +42、001415 +27）；加严后收窄到 002438 **+18**。
5. **代码改动（全部默认关 ⇒ Gate A 不受影响）**
   `xcorr.py`（`match()` 增加 `diagnostics` 出参算唯一性比，**返回值不变**）·
   `extrapolator.py`（`replay_xcorr_uniq_gate: float = 0.0` + 二次意见 + 6 个新计数器）·
   `lookahead_longquan.py`（`--replay-xcorr-uniq-gate` **默认 0.0=关** + 接线 + 配置快照 + `stream_stats` 汇总）·
   `run_delivery.sh`（**由 `run_delivery_polarity.sh` 改名**，`$CFG` 加 `--replay-xcorr-uniq-gate 0.6`）。
   **板端算力代价**：τ=0.6 时 **1698/24944 = 6.8%** 的匹配多做一次 `matchTemplate` ⇒ 平均 **+6.8% MAC**。
6. **τ 扫描（4 批）**：`uniqrematch2/`（加严＝交付口径，**τ=0.6 最优**）· `uniqrematch/`（宽松，收益小）·
   `uniq/`（回退锚点，**全档净负，永久否决**）· `uniq24/` + `uniq24b/`（24 序列，加严口径同样最优）。
7. **回归防线**：Gate A 8 序 **9/9 IDENTICAL / 0 DIFFER** + 24 序 **25/25 / 0 DIFFER**；
   **τ=0 时 replay 臂与上一档交付产物 9/9 md5 逐字节一致**；单测 **149 项（10 skipped）OK**。
8. **文档**：专题档 **§26**（284 行）+ 主档 **§30.9.21**（134 行）+ README **§5.10 / §6 / §7 / §8** +
   板端契约 **§3.6 契约六**（+ R15 风险条目）；导航表 `[PASS] **97/97**`（原 95/95）。

### 二' OSD 三臂对照（★ 验收画面，4 序列 × 3 臂 = 12 段）

单元格 = 窗口内 **CLE 均值 / hit@5px / CLE max**；窗口判据 = `mean(cle[stale]) − mean(cle[replay])` 最大。

| 序列 | 窗口 | `stale`（老逻辑，真吃 320 ms） | **`replay`（外挂交付档）** | `original`（零延迟天花板） |
|:--|:--|:--|:--|:--|
| **`000703`** | 218–517 | 33.26 px / 6.7% / max 108.7 | **3.04 px / 93.0% / max 40.3** | 1.87 px / 99.0% / max 6.2 |
| **`001324`** | 8538–8837 | 32.89 px / 22.7% / max 192.2 | **2.41 px / 91.7% / max 10.3** | 1.98 px / 99.7% / max 5.7 |
| **`001531`** | 8–107 | 22.67 px / 4.0% / max 111.9 | **1.80 px / 98.0% / max 6.4** | 1.61 px / 100.0% / max 3.5 |
| **`015000`** | 339–638 | 18.91 px / 14.3% / max 68.3 | **2.23 px / 98.7% / max 6.0** | 2.03 px / 100.0% / max 5.0 |

⇒ 画面上就是「**老逻辑抽搐脱靶 vs 外挂平滑跟住**」：stale 的红十字明显偏离 GT 绿框（最大 192 px），
replay 的红十字锁在目标内（最大 6~40 px）。

## 三、早上 host 接回要做的三件事

```bash
# ① 把本轮产物拉回 host
scp huangzhe@192.168.99.40:/tmp/pycharm_project_10ae9e2e/runs/lookahead/final/code_sync_20261104.tar.gz .
tar xzf code_sync_20261104.tar.gz
#    md5 以服务器侧 runs/lookahead/final/code_sync_20261104.tar.gz.md5 为准
#    服务器侧备份：/mnt/data/codex-overnight/backup_20261104/

# ② 同步 host 侧代码（host 的 xcorr.py / extrapolator.py / lookahead_longquan.py 仍是**旧的**，
#    且**没有** polarity.py）⇒ 从解包出来的 manu/tracking/lookahead/ 覆盖过去

# ③ 恢复会话
export CODEX_HOME=$HOME/.codex
codex resume 01a199f3-5523-7591-b039-9e07b7a1fb58
```

## 四、下一步（按优先级）

1. **`001531` 的剩余缺口**（8 序列）：天花板 99.57% 而外挂 **91.30%** —— 这是 8 序列上唯一还明显落后的一条。
2. **24 序列 `002438` +18 FP**（唯一劣化的序列）：查它的歧义帧为什么「二次意见」没救回来。
3. **24 序列 FAR 还差 0.0049**（0.0349 vs 红线 0.030）：主体是**管线自身**
   （`015740` 0.4428 / original 0.4277、`002251` 0.1034 / original 0.0997、`003002` 0.0707）⇒ **不为此牺牲 hit@5px**。
4. **极性方向的两个小补丁**（专题档 §25.12）：① 暗/亮**切换**时沿用上一链极性（时间平滑）；
   ② `min_energy` / `min_contrast` **按极性分别设阈值**。
5. **板端移植**：契约六条（xcorr 精修 / 极性自适应 / 输出限幅 / 兜底策略 / 无状态性 / **唯一性二次意见**）；
   板端必须加一级输出限速/低通（`p99_jitter` 36.56 px 仍大于 stale 的 18.9 px）。

## 五、关键产物路径（服务器磁盘）

| 产物 | 路径 |
|:--|:--|
| **8 序列主产物（本轮交付档）** | `runs/lookahead/final/final8/arms_delivery.json` + `final8/preds_delivery/` + `final8/delivery.log` + `final8/report_delivery_vs_prev.md` |
| **24 序列主产物（本轮交付档）** | `runs/lookahead/final/final24/arms_delivery.json` + `final24/preds_delivery/` + `final24/delivery.log` |
| τ 扫描（加严＝交付口径） | `runs/lookahead/final/uniqrematch2/arms_u{030,050,060,070}.json` |
| τ 扫描（宽松 / 回退锚点） | `runs/lookahead/final/uniqrematch/arms_u{020…070}.json` · `runs/lookahead/final/uniq/arms_u{0,030,…,085}.json` |
| τ 扫描（24 序列） | `runs/lookahead/final/uniq24/arms_u{050,060,070}.json` · `runs/lookahead/final/uniq24b/arms_u{050,060}.json` |
| 上一档（对照臂） | `runs/lookahead/final/pol8f/arms_F1_auto10.json` · `runs/lookahead/final/pol24f/arms_F1_auto10.json` |
| **OSD 三臂视频（本轮新交付档）** | `runs/lookahead/osd/videos/<SEQ>_3arm_{stale,replay,original}.mp4`（4 序列 × 3 臂 = **12 段**）· 窗口 `runs/lookahead/osd/windows/<SEQ>_3arm.json` · 脚本 `manu/tracking/lookahead/osd/render_osd_3arm.sh` |
| OSD 四臂视频（上一档） | `runs/lookahead/osd/videos/<SEQ>_pol_{stale,replay_old,replay_new,original}.mp4`（4 序列 × 4 臂 = 16 段） |
| 同步包 | `runs/lookahead/final/code_sync_20261104.tar.gz`（约 **206 MB / 425 文件**，含 12 段新 OSD 视频）+ `.manifest.md5` + `.tar.gz.md5`；**md5 以磁盘上的 `code_sync_20261104.tar.gz.md5` 为准**（本文件会被打进 tar 并回填该值 ⇒ 硬写 md5 会自引用漂移，故此处只给指针） |
| 备份 | `/mnt/data/codex-overnight/backup_20261104/` |
| memory | `manu/memory/memory.md` **§30.9.21** · `manu/memory/lookahead_extrapolator_20261102.md` **§26** · `manu/memory/lookahead_board_port_contract_20261103.md` **§3.6** · `manu/tracking/lookahead/README.md` **§5.10** |

## 六、本轮代码基线（md5）

| 文件 | md5 | 状态 |
|:--|:--|:--|
| `manu/tracking/lookahead/xcorr.py` | `1c9ddf5848b1d86c0997b9f35df170e9` |
| `manu/tracking/lookahead/extrapolator.py` | `0ff908ec5ac042483de8cb46503b7676` |
| `manu/tracking/lookahead/lookahead_longquan.py` | `d12bb291ffe2b7fb53090bd0e6095eac` |
| `manu/tracking/lookahead/polarity.py` | `2cd585ad87bad8438f6ecfc7272f0553` |
| `manu/tracking/lookahead/run_delivery.sh` | `573928ca2f1591cdf4d5ff7e1602fa60` |
| `manu/tracking/tests/test_polarity.py` | `338779b0278d2e61aad27ffeaf17e2bd` |
| `manu/evaluation/system_sota_longquan.py` | `3361e7b70e76d3a92346fd8fd9879dad` |
