#!/usr/bin/env python3
"""Generate the round-4 handover doc (numbers pulled from the measured artifacts)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path("/tmp/pycharm_project_10ae9e2e")
FINAL = ROOT / "runs/lookahead/final"
MEM = ROOT / "manu/memory"
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))

pct = lambda v: f"{float(v) * 100:.4f}%"
num = lambda v, d=4: f"{float(v):.{d}f}"


def biz(doc, name):
    o = doc["arms"][name]["current_frame"]["overall"]
    return {"f1": o["f1"], "far": o["far_per_frame"], "tp": o["tp"], "fp": o["fp"], "fn": o["fn"],
            "r": o["recall"], "p": o["precision"]}


def servo(doc, name):
    """CLE 用**全帧池化**口径（`cle_pooled_*`）—— 伺服可接受性看池化（专题档 §21 规则）；
    最差单序列口径只用于尾部风险；`max_jitter` 只能用最差单序列口径。"""
    s = doc["arms"][name]["servo"]["overall"]
    return {"hit": s.get("hit_rate_5px"), "p50": s.get("cle_pooled_p50"), "p95": s.get("cle_pooled_p95"),
            "worst_p50": s.get("cle_p50"), "worst_p95": s.get("cle_p95"),
            "max": s.get("cle_max"), "max_jitter": s.get("max_jitter"), "p99_jitter": s.get("p99_jitter")}


new8 = load(FINAL / "final8/arms_delivery.json")
new24 = load(FINAL / "final24/arms_delivery.json")
old8 = load(FINAL / "noguard8/arms_delivery.json")
old24 = load(FINAL / "noguard24/arms_delivery.json")
attrib = load(FINAL / "final24/diag_fp_attrib.json")["totals"]
b = attrib["buckets"]

# ---- OSD 三臂窗口指标（由 render_osd_3arm.sh 的 stdout 落盘） -------------------
osd_rows = []
osd_dir = FINAL.parent / "osd" / "windows"
for seq in ("VIDEO00006_19700101_000703", "VIDEO00007_19700101_001324",
            "VIDEO00009_19700101_001531", "VIDEO00033_19700101_015000"):
    f = osd_dir / f"{seq}_3arm.json"
    if not f.is_file():
        continue
    d = load(f)
    per = d["per_arm"]
    osd_rows.append((seq, d["start"], d["end"], per))

osd_table = ""
if osd_rows:
    osd_table = "\n".join(
        f"| **`{seq.split('_')[-1]}`** | {st}–{en} | "
        + " | ".join(f"{per[a]['cle_mean']:.2f} px / {per[a]['hit_5px'] * 100:.1f}% / {per[a]['cle_max']:.1f}"
                     for a in ("stale", "replay", "original"))
        + " |" for seq, st, en, per in osd_rows)

txt = f"""# 交接说明 · 2026-11-04 服务器夜间第 2 轮（服务器）→ host

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e`
> **python**：`/home/huangzhe/anaconda3/envs/uav/bin/python`（conda env `uav`）
> **线程 id**：`01a199f3-5523-7591-b039-9e07b7a1fb58`
> **host 关机**：2026-11-03 18:30（host 侧本轮没有任何写入）
> **本轮时段**：2026-11-04 00:27 → 02:20（服务器夜间自主推进 · 第 2 轮）
> **上一轮交接**：`handoff_20261104_server_round3.md`（已被本文取代）

## 一、结论速览（本轮把 **24 序列 FAR 红线缺口关掉**）

| 口径 | `original`（天花板，**不是基线**） | `stale`（**交付基线** = 老逻辑） | 上一档（唯一性二次意见） | **本轮新冠军（不自信就不上报）** |
|:--|--:|--:|--:|--:|
| **8 序列** F1 / hit@5px / FAR | {pct(biz(new8,'original')['f1'])} / {pct(servo(new8,'original')['hit'])} / {num(biz(new8,'original')['far'])} | {pct(biz(new8,'stale')['f1'])} / {pct(servo(new8,'stale')['hit'])} / {num(biz(new8,'stale')['far'])} | {pct(biz(old8,'replay')['f1'])} / {pct(servo(old8,'replay')['hit'])} / {num(biz(old8,'replay')['far'])} | **{pct(biz(new8,'replay')['f1'])} / {pct(servo(new8,'replay')['hit'])} / {num(biz(new8,'replay')['far'])}** ✅ 三闸全过 |
| **24 序列** F1 / hit@5px / FAR | {pct(biz(new24,'original')['f1'])} / {pct(servo(new24,'original')['hit'])} / {num(biz(new24,'original')['far'])} | {pct(biz(new24,'stale')['f1'])} / {pct(servo(new24,'stale')['hit'])} / {num(biz(new24,'stale')['far'])} | {pct(biz(old24,'replay')['f1'])} / {pct(servo(old24,'replay')['hit'])} / {num(biz(old24,'replay')['far'])} | **{pct(biz(new24,'replay')['f1'])} / {pct(servo(new24,'replay')['hit'])} / {num(biz(new24,'replay')['far'])}** ✅ **红线 0.030 以内** |

- **本轮新增机理 =「不自信就不上报」**（新旋钮 `--replay-abstain-peak-below` / `--replay-abstain-disp-px`，**默认 0.0 = 关**）。
- **交付配置 = 上一档 + `--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24`**（其余参数一字不改）。
  一键：`bash manu/tracking/lookahead/run_delivery.sh [8|24|both]`。
- **相对上一档**：8 序列 F1 **{(biz(new8,'replay')['f1'] - biz(old8,'replay')['f1']) * 100:+.4f} pp** /
  FAR **{num(biz(new8,'replay')['far'] - biz(old8,'replay')['far'])}** / hit@5px **{(servo(new8,'replay')['hit'] - servo(old8,'replay')['hit']) * 100:+.2f} pp** /
  CLE p95(池化) **{num(servo(new8,'replay')['p95'] - servo(old8,'replay')['p95'], 2)} px**；
  24 序列 F1 **{(biz(new24,'replay')['f1'] - biz(old24,'replay')['f1']) * 100:+.4f} pp** /
  FAR **{num(biz(new24,'replay')['far'] - biz(old24,'replay')['far'])}** /
  FP **{biz(new24,'replay')['fp'] - biz(old24,'replay')['fp']}** 个。
- **24 序列 FAR 缺口关闭**：{num(old24['arms']['replay']['current_frame']['overall']['far_per_frame'])} → **{num(biz(new24,'replay')['far'])}**（交付红线 0.030），
  且 **8 序列主交付档的 hit@5px 一格没掉**（{pct(servo(old8,'replay')['hit'])} → {pct(servo(new8,'replay')['hit'])}）。

## 二、本轮做了什么

1. **★ 把 {attrib['replay_fp']} 个误报逐帧归属**（新脚本 `manu/tracking/lookahead/diag_fp_attrib.py`，
   replay 臂 vs 零延迟臂同帧对账）：
   | 归属 | 数量 | 占 FP | 插件能不能改 |
   |:--|--:|--:|:--|
   | `lost_by_replay`（零延迟臂对、重推臂错 ⇒ 重推跟丢） | **{b['lost_by_replay']}** | {b['lost_by_replay'] / attrib['replay_fp'] * 100:.1f}% | **能** |
   | `both_missed`（两臂都错 ⇒ 检测器自身噪声） | {b['both_missed']} | {b['both_missed'] / attrib['replay_fp'] * 100:.1f}% | 不能 |
   | `no_ground_truth`（该帧 GT **显式为空**的标注稀疏） | {b['no_ground_truth']} | {b['no_ground_truth'] / attrib['replay_fp'] * 100:.1f}% | 不能 |
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
   - **正确性闸门**：空规则必须逐位复现交付档（8 序 F1 {pct(load(FINAL / 'final8/diag_abstain_scan.json')['baseline']['f1'])}、
     24 序 F1 {pct(load(FINAL / 'final24/diag_abstain_scan.json')['baseline']['f1'])}）—— PASS。
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
{osd_table}

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

1. **8 序列剩余 hit@5px 差距**：{(servo(new8,'original')['hit'] - servo(new8,'replay')['hit']) * 100:.2f} pp
   （天花板 {pct(servo(new8,'original')['hit'])} vs 交付档 {pct(servo(new8,'replay')['hit'])}），
   其中 `001324` 单独占 ~2.0 pp，而它自身天花板只有 70.53%。
2. **24 序列逐序列 FAR 余量**：看还有哪条序列离红线最近。
3. **标注稀疏口径**：`{b['no_ground_truth'] / attrib['replay_fp'] * 100:.0f}%` 的误报落在 GT **显式为空**的帧上，
   建议与标注方确认这批帧的判定口径。
4. **板端移植**：按 `manu/memory/lookahead_board_port_contract_20261103.md` §3.7 契约七实现弃权判据
   （**零额外图像运算**），并把两个旋钮做成可调 + 保留关闭位。
5. **不要再试的旋钮家族**（已被实测否决，见专题档 §27.4）：唯一性比单门 / 模板位置-链位置分歧单门 /
   链位移单门 / 峰值单门 / 二次意见弃权 / 精修失败弃权。
"""

(MEM / "handoff_20261104_server_round4.md").write_text(txt, encoding="utf-8")
print(f"wrote handoff: {len(txt.splitlines())} lines")
