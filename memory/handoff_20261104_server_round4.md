# 交接说明 · 2026-11-04 服务器夜间第 2 轮（服务器）→ host

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e`
> **python**：`/home/huangzhe/anaconda3/envs/uav/bin/python`（conda env `uav`）
> **线程 id**：`01a199f3-5523-7591-b039-9e07b7a1fb58`
> **host 关机**：2026-11-03 18:30（host 侧本轮没有任何写入）
> **本轮时段**：2026-11-04 00:27 → 02:20（服务器夜间自主推进 · 第 2 轮）
> **上一轮交接**：`handoff_20261104_server_round3.md`（已被本文取代）
> **下一轮交接**：`handoff_20261104_server_round5.md`（**第 3 轮已接手，交付档未改**）

## 一、结论速览（本轮把 **24 序列 FAR 红线缺口关掉**）

| 口径 | `original`（天花板，**不是基线**） | `stale`（**交付基线** = 老逻辑） | 上一档（唯一性二次意见） | **本轮新冠军（不自信就不上报）** |
|:--|--:|--:|--:|--:|
| **8 序列** F1 / hit@5px / FAR | 96.4002% / 88.1316% / 0.0063 | 77.2034% / 38.1850% / 0.1807 | 95.2664% / 85.1691% / 0.0163 | **95.3038% / 85.1728% / 0.0151** ✅ 三闸全过 |
| **24 序列** F1 / hit@5px / FAR | 94.7469% / 85.8957% / 0.0208 | 77.5225% / 44.8232% / 0.1559 | 92.9245% / 81.6362% / 0.0349 | **93.0865% / 81.4974% / 0.0298** ✅ **红线 0.030 以内** |

- **本轮新增机理 =「不自信就不上报」**（新旋钮 `--replay-abstain-peak-below` / `--replay-abstain-disp-px`，**默认 0.0 = 关**）。
- **交付配置 = 上一档 + `--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24`**（其余参数一字不改）。
  一键：`bash manu/tracking/lookahead/run_delivery.sh [8|24|both]`。
- **相对上一档**：8 序列 F1 **+0.0374 pp** /
  FAR **-0.0012** / hit@5px **+0.00 pp** /
  CLE p95(池化) **-0.49 px**；
  24 序列 F1 **+0.1621 pp** /
  FAR **-0.0051** /
  FP **-838** 个。
- **24 序列 FAR 缺口关闭**：0.0349 → **0.0298**（交付红线 0.030），
  且 **8 序列主交付档的 hit@5px 一格没掉**（85.1691% → 85.1728%）。

## 二、本轮做了什么

1. **★ 把 5764 个误报逐帧归属**（新脚本 `manu/tracking/lookahead/diag_fp_attrib.py`，
   replay 臂 vs 零延迟臂同帧对账）：
   | 归属 | 数量 | 占 FP | 插件能不能改 |
   |:--|--:|--:|:--|
   | `lost_by_replay`（零延迟臂对、重推臂错 ⇒ 重推跟丢） | **1658** | 28.8% | **能** |
   | `both_missed`（两臂都错 ⇒ 检测器自身噪声） | 1367 | 23.7% | 不能 |
   | `no_ground_truth`（该帧 GT **显式为空**的标注稀疏） | 2739 | 47.5% | 不能 |
   ⇒ **FAR 里近一半是标注稀疏，不是算法能修的**；真正能修的只有 `lost_by_replay`。
2. **★ 新机理「不自信就不上报」**（`manu/tracking/lookahead/extrapolator.py`）
   - 模板配准（xcorr）的**相关峰**是这次重推唯一的自洽性证据；峰值低有两种成因：
     (a) 目标**机动太大**（模板跟不上但快核跟得住 ⇒ **位置是对的**）、
     (b) 快核**跟丢**（**位置是错的**）。
   - 用**链位移**区分：跟丢时链从锚点跑很远（`001415` 低峰帧 p50 = 80 px），机动时链几乎没跑动（`001324` 低峰帧 p50 = 0 px）。
   - ⇒ **弃权条件 = 峰值 < 0.30 且 链位移 > 24 px ⇒ 该帧不上报**（转台保持上一帧指令）。
   - **板端代价为零**：峰值与链位移都是重推链**已经在算**的量，判据只是一次比较。
3. **★ 离线策略扫描器**（新脚本 `manu/tracking/lookahead/diag_abstain_scan.py`）
   - `--dump`：按**交付档 argv**（脚本把 argv 喂给生产 CLI 解析器，避免手抄参数漂移）跑一遍 replay，
     逐帧记录上报坐标 + 链诊断（peak / uniq / chain_dev / anchor_disp / rematch / held / final_peak / xcorr_ok）；
   - `--scan`：在这份 dump 上**秒级**复算任意弃权规则（F1 / FAR / hit@5px / jitter），
     不必为每个候选重跑重推。
   - **正确性闸门**：空规则必须逐位复现交付档（8 序 F1 95.2664%、
     24 序 F1 92.9245%）—— PASS。
   - **被扫描否决的规则家族**（全部带数字）：唯一性比单门、模板位置-链位置分歧单门、链位移单门、
     峰值单门（误伤 `001324` 高速机动）、「二次意见采纳就弃权」、「精修失败就弃权」。
   - 阈值细扫（peak × disp 全网格，8/24 双口径）：`/mnt/data/codex-overnight/fine_grid.json`。
4. **代码改动（全部默认关 ⇒ 关开关逐位一致）**
   `extrapolator.py`（`LookaheadConfig.replay_abstain_peak_below` / `replay_abstain_disp_px`，**默认 0.0**）·
   `lookahead_longquan.py`（`--replay-abstain-peak-below` / `--replay-abstain-disp-px`，**默认 0.0=关** + 接线）·
   `run_delivery.sh`（`$CFG` 加两个新旋钮，8/24 同一份配置）·
   `manu/tracking/tests/test_lookahead_plugin.py`（新增 3 项单测）。
5. **回归防线（全部 PASS）**
   - Gate A 8 序：**IDENTICAL=9 / DIFFER=0**（F1 均 0.964002）；
   - Gate A 24 序：**IDENTICAL=25 / DIFFER=0**（F1 均 0.947469）；
   - 尺子 1（历史帧等价性）：8 序 / 24 序 F1 0.964127 / 0.947527，与上一档**逐位相同**（ΔF1 = 0.000000）；
   - 新旋钮**关闭**时 replay 臂 vs **上一档冠军产物**：**8 序 IDENTICAL=9 / DIFFER=0**（`noguard8/preds_delivery` vs `uniqrematch2/preds_u060`）、**24 序 IDENTICAL=25 / DIFFER=0**（`noguard24/preds_delivery` vs `uniq24b/preds_u060`）；
   - 单测 **152 项 OK**；导航表 `manu/memory/validate_nav_table.py` **[PASS]**。

### 二' OSD 三臂对照（★ 验收画面，4 序列 × 3 臂 = 12 段）

单元格 = 窗口内 **CLE 均值 / hit@5px / CLE max**；窗口判据 = `mean(cle[stale]) − mean(cle[replay])` 最大。
视频：`runs/lookahead/osd/videos/<SEQ>_3arm_<arm>.mp4`（h264）。

| 序列 | 窗口 | `stale`（老逻辑，真吃 320 ms） | **`replay`（外挂交付档）** | `original`（零延迟天花板） |
|:--|:--|:--|:--|:--|
| **`000703`** | 68–367 | 29.95 px / 15.3% / 108.7 | 3.37 px / 87.0% / 26.5 | 2.50 px / 92.3% / 7.0 |
| **`001324`** | 8538–8837 | 32.89 px / 22.7% / 192.2 | 2.41 px / 91.7% / 10.3 | 1.98 px / 99.7% / 5.7 |
| **`001531`** | 8–107 | 22.67 px / 4.0% / 111.9 | 1.80 px / 98.0% / 6.4 | 1.61 px / 100.0% / 3.5 |
| **`015000`** | 339–638 | 18.91 px / 14.3% / 68.3 | 2.23 px / 98.7% / 6.0 | 2.03 px / 100.0% / 5.0 |

⇒ 画面上就是「**老逻辑抽搐脱靶 vs 外挂平滑跟住**」。

## 三、产物索引（全部在服务器磁盘）

| 用途 | 路径 |
|:--|:--|
| **8 序列定稿** | `runs/lookahead/final/final8/arms_delivery.json` + `final8/preds_delivery/` + `final8/delivery.log` |
| **24 序列定稿** | `runs/lookahead/final/final24/arms_delivery.json` + `final24/preds_delivery/` + `final24/delivery.log` |
| 上一档对照（无弃权） | `runs/lookahead/final/noguard8/` · `runs/lookahead/final/noguard24/` |
| FP 归属拆解 | `runs/lookahead/final/final24/diag_fp_attrib.json` + `.md` |
| 弃权策略扫描 | `runs/lookahead/final/final8/diag_abstain_scan.json` + `final24/diag_abstain_scan.json`（+ `.md`） |
| 逐帧 dump（扫描器输入，**大文件，不进同步包**） | `runs/lookahead/final/final8/abstain_diag/` · `final24/abstain_diag/`（`--dump` 可重建） |
| 阈值细扫 | `/mnt/data/codex-overnight/fine_grid.json` |
| OSD 三臂视频 | `runs/lookahead/osd/videos/` |
| 同步包 | `runs/lookahead/final/code_sync_20261104.tar.gz` + `.md5` + `.manifest.md5` |

## 四、文档落点（本轮新增）

| 文件 | 新增章节 |
|:--|:--|
| `manu/memory/lookahead_extrapolator_20261102.md` | **§27**（专题档，完整机理 + 全部表格） |
| `manu/memory/memory.md` | **§30.9.22**（主档速览） |
| `manu/tracking/lookahead/README.md` | **§5.11** |
| `manu/memory/lookahead_board_port_contract_20261103.md` | **§3.7 契约七** + 风险条目 **R16 / R17** |
| `manu/memory/handoff_20261104_server_round4.md` | 本文 |

## 五、下一步（host 开机后可直接接续）

1. **8 序列剩余 hit@5px 差距**：2.96 pp
   （天花板 88.1316% vs 交付档 85.1728%），
   其中 `001324` 单独占 ~2.0 pp，而它自身天花板只有 70.53%。
2. **24 序列逐序列 FAR 余量**：看还有哪条序列离红线最近。
3. **标注稀疏口径**：`48%` 的误报落在 GT **显式为空**的帧上，
   建议与标注方确认这批帧的判定口径。
4. **板端移植**：按 `manu/memory/lookahead_board_port_contract_20261103.md` §3.7 契约七实现弃权判据
   （**零额外图像运算**），并把两个旋钮做成可调 + 保留关闭位。
5. **不要再试的旋钮家族**（已被实测否决，见专题档 §27.4）：唯一性比单门 / 模板位置-链位置分歧单门 /
   链位移单门 / 峰值单门 / 二次意见弃权 / 精修失败弃权。
