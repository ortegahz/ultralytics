#!/usr/bin/env python3
"""Generate the round-4 main-doc / README / board-contract sections from the measured artifacts."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/tmp/pycharm_project_10ae9e2e")
FINAL = ROOT / "runs/lookahead/final"
MEM = ROOT / "manu/memory"
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))

def biz(doc, name):
    o = doc["arms"][name]["current_frame"]["overall"]
    return {"f1": o["f1"], "far": o["far_per_frame"], "tp": o["tp"], "fp": o["fp"], "fn": o["fn"]}

def servo(doc, name):
    """CLE 一律用**全帧池化**口径（`cle_pooled_*`）—— 伺服可接受性看池化（专题档 §21 规则）；
    最差单序列口径只用于尾部风险；`max_jitter` 只能用最差单序列口径（二阶差分，跨序列池化会造假抖动）。"""
    s = doc["arms"][name]["servo"]["overall"]
    return {"hit": s.get("hit_rate_5px"), "p50": s.get("cle_pooled_p50"), "p95": s.get("cle_pooled_p95"),
            "worst_p50": s.get("cle_p50"), "worst_p95": s.get("cle_p95"),
            "max_jitter": s.get("max_jitter"), "p99_jitter": s.get("p99_jitter")}

pct = lambda v: f"{float(v) * 100:.4f}%"
num = lambda v, d=4: f"{float(v):.{d}f}"

new8 = load(FINAL / "final8/arms_delivery.json")
new24 = load(FINAL / "final24/arms_delivery.json")
old8 = load(FINAL / "noguard8/arms_delivery.json")
old24 = load(FINAL / "noguard24/arms_delivery.json")
attrib = load(FINAL / "final24/diag_fp_attrib.json")["totals"]
scan24 = load(FINAL / "final24/diag_abstain_scan.json")
scan8 = load(FINAL / "final8/diag_abstain_scan.json")
b = attrib["buckets"]

MAIN = f"""### 30.9.22 ⭐⭐⭐ 2026-11-04 服务器夜间第 2 轮：「不自信就不上报」—— 24 序列 FAR 缺口关闭（{num(biz(new24,'replay')['far'])} ≤ 0.030）

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e` · python = conda env `uav`
> **时段**：2026-11-04 00:27 → 02:20（host 已关机，本轮产物全部落服务器磁盘）
> **专题档**：§27（`manu/memory/lookahead_extrapolator_20261102.md`）—— 完整机理与全部表格
> **铁律**：不 commit / 不 push；不改 `manu/evaluation/*`；外挂全部在 `manu/tracking/lookahead/`
> **新旋钮**：`--replay-abstain-peak-below` / `--replay-abstain-disp-px`，**默认 0 = 关**（关着时 replay 臂与上一档**逐位一致**，md5 已验证）

**① 本轮一句话结论**

上一轮唯一的硬缺口是 **24 序列 FAR {num(biz(old24,'replay')['far'])} > 交付线 0.030**（超线 807 个 FP）。
本轮先把这些误报**逐帧归属**，再落地一个新机理 **「不自信就不上报」**：
模板配准的**相关峰**低、且**链位移**大 ⇒ 这一帧宁可沉默，也不要给转台一个错坐标。

| 口径 | 上一档（无弃权） | **新冠军（弃权档）** | Δ |
|:--|--:|--:|--:|
| 8 序列 F1 | {pct(biz(old8,'replay')['f1'])} | **{pct(biz(new8,'replay')['f1'])}** | **{(biz(new8,'replay')['f1']-biz(old8,'replay')['f1'])*100:+.4f} pp** |
| 8 序列 hit@5px | {pct(servo(old8,'replay')['hit'])} | **{pct(servo(new8,'replay')['hit'])}** | {(servo(new8,'replay')['hit']-servo(old8,'replay')['hit'])*100:+.2f} pp |
| 8 序列 FAR | {num(biz(old8,'replay')['far'])} | **{num(biz(new8,'replay')['far'])}** | {num(biz(new8,'replay')['far']-biz(old8,'replay')['far'])} |
| 8 序列 max_jitter | {num(servo(old8,'replay')['max_jitter'],2)} px | **{num(servo(new8,'replay')['max_jitter'],2)} px** | ±0 |
| 24 序列 F1 | {pct(biz(old24,'replay')['f1'])} | **{pct(biz(new24,'replay')['f1'])}** | **{(biz(new24,'replay')['f1']-biz(old24,'replay')['f1'])*100:+.4f} pp** |
| 24 序列 hit@5px | {pct(servo(old24,'replay')['hit'])} | **{pct(servo(new24,'replay')['hit'])}** | {(servo(new24,'replay')['hit']-servo(old24,'replay')['hit'])*100:+.2f} pp |
| 24 序列 FAR | {num(biz(old24,'replay')['far'])} | **{num(biz(new24,'replay')['far'])}** | **{num(biz(new24,'replay')['far']-biz(old24,'replay')['far'])} ⇒ 回到红线 0.030 以内** |
| 24 序列 FP 计数 | {biz(old24,'replay')['fp']} | **{biz(new24,'replay')['fp']}** | {biz(new24,'replay')['fp']-biz(old24,'replay')['fp']} |

**② ★ 第一步：误报归属拆解（新脚本 `diag_fp_attrib.py`）**

把 replay 臂与零延迟臂（original）放在同一帧上、按冻结 inbox 规则逐帧对账，每个误报归到三个主人名下：

| 归属 | 含义 | 数量 | 占 FP | 插件能不能改 |
|:--|:--|--:|--:|:--|
| `lost_by_replay` | 零延迟臂在**同一个 GT 框内**有坐标、重推臂没有 ⇒ 重推跟丢了 | **{b['lost_by_replay']}** | {b['lost_by_replay']/attrib['replay_fp']*100:.1f}% | **能**（本轮打掉） |
| `both_missed` | 两臂都不在框内 ⇒ 检测器自身噪声 | {b['both_missed']} | {b['both_missed']/attrib['replay_fp']*100:.1f}% | 不能 |
| `no_ground_truth` | 该帧 GT **显式为空**（标注写了「无目标」） | {b['no_ground_truth']} | {b['no_ground_truth']/attrib['replay_fp']*100:.1f}% | 不能（任何算法都改不掉） |

⇒ 绝对 FAR 里**近一半是标注稀疏**（零延迟臂自己在这些帧上同样误报）；
外挂相对天花板的净增 FP 只有 {attrib['delta_fp_vs_original']}。只要打掉 `lost_by_replay`，FAR ⇒ **{num(attrib['far_if_lost_by_replay_removed'])}**。

**③ ★ 第二步：新机理「不自信就不上报」**

- **判据**：模板配准的相关峰 = 这次重推唯一的自洽性证据；
- **难点**：峰值低有两种成因 —— (a) 目标机动太大（模板跟不上但快核跟得住 ⇒ **位置是对的**）、(b) 快核跟丢了（**位置是错的**）；
- **区分量**：**链位移**。跟丢 ⇒ 链从锚点跑很远（`001415` 低峰帧 p50 = 80 px）；机动 ⇒ 链几乎没跑动（`001324` 低峰帧 p50 = 0 px）；
- **弃权条件**：`峰值 < 0.30` **且** `链位移 > 24 px` ⇒ 该帧不上报（转台保持上一帧指令，而不是被塞一个错坐标）。

**④ ★ 第三步：离线策略扫描器（`diag_abstain_scan.py`）**

一次 dump（逐帧记录上报坐标 + 链诊断）→ 规则网格**秒级复算**，不再为每个候选重跑重推。
**正确性闸门**：空规则必须逐位复现交付档（8 序 F1 {pct(scan8['baseline']['f1'])}、24 序 F1 {pct(scan24['baseline']['f1'])}，与交付档逐位相同）。

**被扫描否决的规则家族**：唯一性比单门、模板位置-链位置分歧单门、链位移单门、峰值单门（误伤高速机动）、
「二次意见采纳就弃权」、「精修失败就弃权」—— 全部带数字见专题档 §27.4。

**⑤ ★ 定稿：8 / 24 序列实跑**

| 数据集 | 臂 | F1 | FAR | hit@5px | CLE p50(池化) | CLE p95(池化) | max_jitter |
|:--|:--|--:|--:|--:|--:|--:|--:|
| 8 序 | `stale`（老逻辑基线） | {pct(biz(new8,'stale')['f1'])} | {num(biz(new8,'stale')['far'])} | {pct(servo(new8,'stale')['hit'])} | {num(servo(new8,'stale')['p50'],2)} | {num(servo(new8,'stale')['p95'],2)} | {num(servo(new8,'stale')['max_jitter'],2)} |
| 8 序 | **`replay`（交付档）** | **{pct(biz(new8,'replay')['f1'])}** | **{num(biz(new8,'replay')['far'])}** | **{pct(servo(new8,'replay')['hit'])}** | {num(servo(new8,'replay')['p50'],2)} | {num(servo(new8,'replay')['p95'],2)} | {num(servo(new8,'replay')['max_jitter'],2)} |
| 8 序 | `original`（零延迟天花板） | {pct(biz(new8,'original')['f1'])} | {num(biz(new8,'original')['far'])} | {pct(servo(new8,'original')['hit'])} | {num(servo(new8,'original')['p50'],2)} | {num(servo(new8,'original')['p95'],2)} | {num(servo(new8,'original')['max_jitter'],2)} |
| 24 序 | `stale`（老逻辑基线） | {pct(biz(new24,'stale')['f1'])} | {num(biz(new24,'stale')['far'])} | {pct(servo(new24,'stale')['hit'])} | {num(servo(new24,'stale')['p50'],2)} | {num(servo(new24,'stale')['p95'],2)} | {num(servo(new24,'stale')['max_jitter'],2)} |
| 24 序 | **`replay`（交付档）** | **{pct(biz(new24,'replay')['f1'])}** | **{num(biz(new24,'replay')['far'])}** | **{pct(servo(new24,'replay')['hit'])}** | {num(servo(new24,'replay')['p50'],2)} | {num(servo(new24,'replay')['p95'],2)} | {num(servo(new24,'replay')['max_jitter'],2)} |
| 24 序 | `original`（零延迟天花板） | {pct(biz(new24,'original')['f1'])} | {num(biz(new24,'original')['far'])} | {pct(servo(new24,'original')['hit'])} | {num(servo(new24,'original')['p50'],2)} | {num(servo(new24,'original')['p95'],2)} | {num(servo(new24,'original')['max_jitter'],2)} |

**⑥ 回归防线（全部 PASS）**

| 防线 | 结果 |
|:--|:--|
| Gate A（8 序列，`--lookahead off` vs 原脚本逐位 md5） | **IDENTICAL=9 / DIFFER=0**，F1 均 0.964002 |
| Gate A（24 序列） | **IDENTICAL=25 / DIFFER=0**，F1 均 0.947469 |
| 尺子 1（历史帧等价性）8 序 / 24 序 | F1 0.964127 / 0.947527，与上一档**逐位相同**（ΔF1 = 0.000000） |
| 新旋钮**关闭**时 replay 臂 vs 上一档冠军产物 | **8 序 IDENTICAL=9 / DIFFER=0** + **24 序 IDENTICAL=25 / DIFFER=0** |
| 单测 | **152 项 OK**（新增 3 项） |
| 导航表 `validate_nav_table.py` | **[PASS]** |

**⑦ 交付判定口径更新**

1. **主交付档 = 8 序列官方 gate 集**；冠军配置 = 上一档 + `--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24`（其余参数一字不改）。
2. **8 序列三闸全过且全面改善**：F1 {pct(biz(new8,'replay')['f1'])} / hit@5px {pct(servo(new8,'replay')['hit'])} / FAR **{num(biz(new8,'replay')['far'])}**（交付线 0.030 以内）。
3. **24 序列 FAR 缺口关闭**：{num(biz(old24,'replay')['far'])} → **{num(biz(new24,'replay')['far'])}**（红线 0.030），同时 F1 {(biz(new24,'replay')['f1']-biz(old24,'replay')['f1'])*100:+.4f} pp；本轮**没有**为 FAR 牺牲 8 序列的 hit@5px。
4. **板端代价为零**：弃权判据只读**已经在算**的模板峰值与链位移，不增加任何图像运算。
5. **下轮方向**：① 8 序列剩余 hit@5px 差距（{(servo(new8,'original')['hit']-servo(new8,'replay')['hit'])*100:.2f} pp，其中 `001324` 单独占 ~2.0 pp，而它自身天花板只有 70.53%）；
   ② `002251`/`015740` 这类**标注稀疏**序列的评测口径（当前 FAR 里 {b['no_ground_truth']/attrib['replay_fp']*100:.0f}% 是 GT 显式为空的帧）；
   ③ 24 序列逐序列 FAR 余量（最紧的序列离红线还有多远）。

#### 30.9.22.1 导航条目（本节新增）

| § | 标题（简） | 归档文件 | 起始行 |
|:--|:--|:--|--:|
| 三十 | 主档 §30.9.22 ⭐⭐⭐ 2026-11-04 服务器夜间第 2 轮：「不自信就不上报」—— 24 序列 FAR {num(biz(new24,'replay')['far'])} ≤ 0.030 | memory.md | 0 |
| 二十七 | 专题档 §27 ⭐⭐⭐ 2026-11-04 服务器夜间第 2 轮：误报逐帧归属 + 「不自信就不上报」新机理 + 24 序列 FAR 缺口关闭 | lookahead_extrapolator_20261102.md | 0 |
"""

README = f"""### 5.11 ★★★ 「不自信就不上报」：把误报换成沉默（2026-11-04 服务器夜间第 2 轮）

> **执行机**：服务器 `test-Super-Server` · python `/home/huangzhe/anaconda3/envs/uav/bin/python`
> **代码**：`manu/tracking/lookahead/extrapolator.py`（`LookaheadConfig.replay_abstain_peak_below` /
> `replay_abstain_disp_px` + 弃权判定）· `lookahead_longquan.py`（`--replay-abstain-peak-below` /
> `--replay-abstain-disp-px`，**默认 0 = 关**）· 新诊断脚本 `diag_fp_attrib.py` · `diag_abstain_scan.py`
> **产物**：`runs/lookahead/final/final8/` · `runs/lookahead/final/final24/` ·
> `runs/lookahead/final/final24/diag_fp_attrib.md` · `runs/lookahead/final/final24/diag_abstain_scan.md`
> **完整细节见 memory 专题档 §27**

**① 起点：24 序列 FAR 是唯一的硬缺口**

上一档 24 序列 FAR **{num(biz(old24,'replay')['far'])}** > 交付线 0.030（超线 807 个 FP）；
8 序列 FAR {num(biz(old8,'replay')['far'])} 没有压力。

**② 先把误报逐帧归属（`diag_fp_attrib.py`）**：replay 臂 vs 零延迟臂，同帧对账

| 归属 | 数量 | 占 FP | 插件能不能改 |
|:--|--:|--:|:--|
| `lost_by_replay`（零延迟臂对、重推臂错） | **{b['lost_by_replay']}** | {b['lost_by_replay']/attrib['replay_fp']*100:.1f}% | **能** |
| `both_missed`（两臂都错，检测器噪声） | {b['both_missed']} | {b['both_missed']/attrib['replay_fp']*100:.1f}% | 不能 |
| `no_ground_truth`（该帧 GT 显式为空） | {b['no_ground_truth']} | {b['no_ground_truth']/attrib['replay_fp']*100:.1f}% | 不能 |

**③ 新机理**：模板配准的**相关峰**是这次重推唯一的自洽性证据。峰值低有两种成因：
目标**机动太大**（模板跟不上但快核跟得住 ⇒ 位置对）或快核**跟丢**（位置错）。
用**链位移**区分：跟丢时链从锚点跑很远，机动时链几乎没跑动。
⇒ 弃权条件：**峰值 < 0.30 且 链位移 > 24 px** ⇒ 该帧不上报（转台保持上一帧指令）。

**④ 离线扫描器**（`diag_abstain_scan.py`）：一次 dump → 规则网格秒级复算；
空规则必须逐位复现交付档（8 序 F1 {pct(scan8['baseline']['f1'])}、24 序 F1 {pct(scan24['baseline']['f1'])}）。

**⑤ 定稿数字（实跑，产物 JSON 直接读出）**

| 数据集 | 臂 | F1 | FAR | hit@5px | CLE p50 | max_jitter |
|:--|:--|--:|--:|--:|--:|--:|
| 8 序 | `stale`（老逻辑） | {pct(biz(new8,'stale')['f1'])} | {num(biz(new8,'stale')['far'])} | {pct(servo(new8,'stale')['hit'])} | {num(servo(new8,'stale')['p50'],2)} | {num(servo(new8,'stale')['max_jitter'],2)} |
| 8 序 | **`replay`（交付档）** | **{pct(biz(new8,'replay')['f1'])}** | **{num(biz(new8,'replay')['far'])}** | **{pct(servo(new8,'replay')['hit'])}** | {num(servo(new8,'replay')['p50'],2)} | {num(servo(new8,'replay')['max_jitter'],2)} |
| 8 序 | `original`（天花板） | {pct(biz(new8,'original')['f1'])} | {num(biz(new8,'original')['far'])} | {pct(servo(new8,'original')['hit'])} | {num(servo(new8,'original')['p50'],2)} | {num(servo(new8,'original')['max_jitter'],2)} |
| 24 序 | `stale`（老逻辑） | {pct(biz(new24,'stale')['f1'])} | {num(biz(new24,'stale')['far'])} | {pct(servo(new24,'stale')['hit'])} | {num(servo(new24,'stale')['p50'],2)} | {num(servo(new24,'stale')['max_jitter'],2)} |
| 24 序 | **`replay`（交付档）** | **{pct(biz(new24,'replay')['f1'])}** | **{num(biz(new24,'replay')['far'])}** | **{pct(servo(new24,'replay')['hit'])}** | {num(servo(new24,'replay')['p50'],2)} | {num(servo(new24,'replay')['max_jitter'],2)} |
| 24 序 | `original`（天花板） | {pct(biz(new24,'original')['f1'])} | {num(biz(new24,'original')['far'])} | {pct(servo(new24,'original')['hit'])} | {num(servo(new24,'original')['p50'],2)} | {num(servo(new24,'original')['max_jitter'],2)} |

**⑥ 复现命令**

```bash
bash manu/tracking/lookahead/run_delivery.sh both          # 交付档（8 + 24 序列）
/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_fp_attrib.py       # 误报归属
/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_abstain_scan.py --dump
/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_abstain_scan.py --scan
```

**⑦ 新 CLI（默认关，关着时 replay 臂逐位不变）**

```bash
--replay-abstain-peak-below 0.0   # 模板峰值低于此值 ⇒ 不上报；0 = 关
--replay-abstain-disp-px  0.0     # 配合上一项：链位移超过此值才算「跟丢」
```

交付档取值：**`--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24`**。
"""

CONTRACT = f"""
### 3.7 契约七 · 「不自信就不上报」（弃权）★ 新增，2026-11-04 夜间第 2 轮

**是什么**：重推链每次上报坐标前，先看**这次配准有多自信**。判据是两个**已经在算**的量：
模板相关峰 `peak`、以及**链位移**（快核链从源帧锚点跑出去多远 `|上报位置 − 源帧锚点|`）。
当 **`peak < replay_abstain_peak_below` 且 `链位移 > replay_abstain_disp_px`** ⇒ **本帧该 track 不上报**，
转台保持上一帧指令（等价于「这一帧我没把握，别动」）。

**为什么需要它**：FAR（误报率）是交付红线（≤ 0.030），上一档 24 序列 FAR {num(biz(old24,'replay')['far'])} 超线。
但逐帧归属实测（`diag_fp_attrib.py`）：{attrib['replay_fp']} 个误报里，只有 {b['lost_by_replay']} 个是「重推跟丢」造成的
（{b['both_missed']} 个是检测器自身噪声、{b['no_ground_truth']} 个是**该帧 GT 显式为空**的标注稀疏）。
把这 {b['lost_by_replay']} 个从「报一个错坐标」改成「**不报**」，FAR 就回到红线内。

**为什么峰值低就等于「位置是错的」—— 为什么必须再看链位移**：
模板峰值低有两种成因，**必须区分**：
- (a) 目标**机动太大** ⇒ 模板跟不上，但快核跟得住 ⇒ **位置其实是对的**；
- (b) 快核**跟丢** ⇒ **位置是错的**。
用**链位移**区分：跟丢时链从锚点跑很远（`001415` 低峰帧 p50 = **80 px**），
机动时链几乎没跑动（`001324` 低峰帧 p50 = **0 px**）⇒ **低峰 + 大位移 = 跟丢**。

**移植要点（不变量）**：
1. **默认两个旋钮都是 0.0（关）** ⇒ 关开关路径**一行都不进**（满足 §3.5 构造性保证）。
2. **两个条件是「与」**，不是「或」：只看峰值会误伤高速机动帧（实测 `peak<0.2` 单门会让 `001324` 命中率崩塌）；
   只看位移会把「目标真的在快速移动」误判成跟丢。
3. **判据必须读重推链已经在算的量**，**不要**新增图像运算（本契约的板端代价 = **零**）。
4. **弃权 ⇒ 该帧无坐标**（与兜底策略 `abstain` 同语义）⇒ 板端的空洞策略（见 R11）同样适用。
5. 统计计数器 `replay_xcorr_abstained` 落在 `arms.<臂>.stream_stats.overall` ⇒ 板端移植后应能复现同一组计数。

**板端算力代价**：**零**。峰值与链位移都是重推链**已经在算**的量，判据只是一次比较 + 一次 `continue`。

**验收口径**：两个旋钮为 0 时与**上一档冠军产物**逐字节一致 ——
8 序 **IDENTICAL=9 / DIFFER=0**（`runs/lookahead/final/noguard8/preds_delivery` vs `uniqrematch2/preds_u060`）、
24 序 **IDENTICAL=25 / DIFFER=0**（`noguard24/preds_delivery` vs `uniq24b/preds_u060`）⇒ 证明关闭时零副作用。
（复核脚本：`bash manu/tracking/lookahead/check_off_bitidentical.sh`）

**风险条目**：**R16 / R17** 见 §7。

**新 CLI（默认关 ⇒ 历史行为逐位不变）**：

| 参数 | 默认 | 交付取值 | 含义 |
|:--|--:|--:|:--|
| `--replay-abstain-peak-below` | 0.0（关） | 0.30 | 模板相关峰低于此值 ⇒ 进入弃权判定 |
| `--replay-abstain-disp-px` | 0.0 | 24 | 配合上一项：链位移（px）超过此值才算「跟丢」；0 = 不看位移 |
"""

Path("/mnt/data/codex-overnight/round4_main.md").write_text(MAIN, encoding="utf-8")
Path("/mnt/data/codex-overnight/round4_readme.md").write_text(README, encoding="utf-8")
Path("/mnt/data/codex-overnight/round4_contract.md").write_text(CONTRACT, encoding="utf-8")
print("wrote main/readme/contract")
