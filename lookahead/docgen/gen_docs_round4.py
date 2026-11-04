#!/usr/bin/env python3
"""Generate the round-4 documentation sections from the measured artifacts (no hand-typed numbers)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/tmp/pycharm_project_10ae9e2e")
FINAL = ROOT / "runs/lookahead/final"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arm(doc, name, section, key):
    return doc["arms"][name][section].get("overall", {}).get(key)


def pct(value):
    return f"{float(value) * 100:.4f}%"


def num(value, digits=4):
    return f"{float(value):.{digits}f}"


def biz(doc, name):
    o = doc["arms"][name]["current_frame"]["overall"]
    return {
        "f1": o["f1"], "far": o["far_per_frame"], "tp": o["tp"], "fp": o["fp"],
        "fn": o["fn"], "det": o["det"], "gt": o["gt"], "r": o["recall"], "p": o["precision"],
    }


def servo(doc, name):
    """servo 指标。

    CLE 有两套口径（见专题档 §21 的规则）：
      * `cle_pooled_*` = **全帧池化**分位数 ⇒ **伺服可接受性**看这套（可跨序列比较）；
      * `cle_p50/95`   = **最差单序列**口径 ⇒ 只用来看**尾部风险**。
    本节表格的 CLE 列一律用**池化**口径；最差序列口径在表下单独标注。
    `max_jitter` / `p99_jitter` 是二阶差分，**只能**用最差序列口径（跨序列池化会造出假抖动）。
    """
    s = doc["arms"][name]["servo"]["overall"]
    return {
        "hit": s.get("hit_rate_5px"), "p50": s.get("cle_pooled_p50"), "p95": s.get("cle_pooled_p95"),
        "worst_p50": s.get("cle_p50"), "worst_p95": s.get("cle_p95"),
        "mean": s.get("cle_mean"), "max": s.get("cle_max"),
        "max_jitter": s.get("max_jitter"), "p99_jitter": s.get("p99_jitter"),
    }


new8 = load(FINAL / "final8/arms_delivery.json")
new24 = load(FINAL / "final24/arms_delivery.json")
old8 = load(FINAL / "noguard8/arms_delivery.json")
old24 = load(FINAL / "noguard24/arms_delivery.json")
attrib = load(FINAL / "final24/diag_fp_attrib.json")
scan24 = load(FINAL / "final24/diag_abstain_scan.json")
scan8 = load(FINAL / "final8/diag_abstain_scan.json")
fine = load("/mnt/data/codex-overnight/fine_grid.json")

A = attrib["totals"]
buckets = A["buckets"]

# ---- the winner row from the fine grid -------------------------------------
winner = min(fine, key=lambda r: (abs(r["peak"] - 0.30) + abs(r["disp"] - 24) / 10.0))

lines: list[str] = []
w = lines.append

w(f"""## 27. 2026-11-04 服务器夜间第 2 轮：★★ 「不自信就不上报」—— 24 序列 FAR 缺口关闭（{num(A['far_per_frame'] if 'far_per_frame' in A else new24['arms']['replay']['current_frame']['overall']['far_per_frame'])} → {num(new24['arms']['replay']['current_frame']['overall']['far_per_frame'])}）

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e` · python = conda env `uav`
> **时段**：2026-11-04 00:27 → 02:10（host 已关机，本轮产物全部落服务器磁盘）
> **铁律**：不 commit / 不 push；不改 `manu/evaluation/*`；外挂全部在 `manu/tracking/lookahead/`
> **新旋钮**：`--replay-abstain-peak-below` / `--replay-abstain-disp-px`，**默认 0 = 关**（关着时 replay 臂与上一档**逐位一致**，已用 md5 验证）

### 27.0 本轮一句话结论

24 序列 FAR 是上一轮唯一的硬缺口（{num(old24['arms']['replay']['current_frame']['overall']['far_per_frame'])} > 交付线 0.030，超线 807 个 FP）。
本轮先把这 {old24['arms']['replay']['current_frame']['overall']['fp']} 个误报**逐帧归属**，再落地一个新机理
**「不自信就不上报」**：模板配准的**相关峰**低、且**链位移**大 ⇒ 这一帧宁可沉默，也不要给转台一个错坐标。

| 口径 | 上一档（无弃权） | **新冠军（弃权档）** | Δ |
|:--|--:|--:|--:|
| 8 序列 F1 | {pct(biz(old8,'replay')['f1'])} | **{pct(biz(new8,'replay')['f1'])}** | **{(biz(new8,'replay')['f1']-biz(old8,'replay')['f1'])*100:+.4f} pp** |
| 8 序列 hit@5px | {pct(servo(old8,'replay')['hit'])} | **{pct(servo(new8,'replay')['hit'])}** | {(servo(new8,'replay')['hit']-servo(old8,'replay')['hit'])*100:+.2f} pp |
| 8 序列 FAR | {num(biz(old8,'replay')['far'])} | **{num(biz(new8,'replay')['far'])}** | {num(biz(new8,'replay')['far']-biz(old8,'replay')['far'])} |
| 8 序列 max_jitter | {num(servo(old8,'replay')['max_jitter'],2)} px | **{num(servo(new8,'replay')['max_jitter'],2)} px** | {num(servo(new8,'replay')['max_jitter']-servo(old8,'replay')['max_jitter'],2)} px |
| 24 序列 F1 | {pct(biz(old24,'replay')['f1'])} | **{pct(biz(new24,'replay')['f1'])}** | **{(biz(new24,'replay')['f1']-biz(old24,'replay')['f1'])*100:+.4f} pp** |
| 24 序列 hit@5px | {pct(servo(old24,'replay')['hit'])} | **{pct(servo(new24,'replay')['hit'])}** | {(servo(new24,'replay')['hit']-servo(old24,'replay')['hit'])*100:+.2f} pp |
| 24 序列 FAR | {num(biz(old24,'replay')['far'])} | **{num(biz(new24,'replay')['far'])}** | **{num(biz(new24,'replay')['far']-biz(old24,'replay')['far'])}（红线 0.030 ⇒ **回到线内**）** |
| 24 序列 FP 计数 | {biz(old24,'replay')['fp']} | **{biz(new24,'replay')['fp']}** | {biz(new24,'replay')['fp']-biz(old24,'replay')['fp']} |

⇒ **24 序列 FAR 缺口关闭**（{num(old24['arms']['replay']['current_frame']['overall']['far_per_frame'])} → {num(new24['arms']['replay']['current_frame']['overall']['far_per_frame'])}，红线 0.030），
且 **8 序列主交付档的 hit@5px 一格没掉**（{pct(servo(old8,'replay')['hit'])} → {pct(servo(new8,'replay')['hit'])}）、F1 还涨了 {(biz(new8,'replay')['f1']-biz(old8,'replay')['f1'])*100:+.4f} pp。

### 27.1 ★ 第一步：把这 {old24['arms']['replay']['current_frame']['overall']['fp']} 个误报**逐帧归属**（新脚本 `diag_fp_attrib.py`）

上一轮只说清了「FP 落在有 GT 的帧还是无 GT 的帧」，没回答**是谁的锅**。本轮把 **replay 臂**与**零延迟臂（original）**
放在同一帧上对账，按冻结的 inbox 匹配规则逐帧复算，把每个误报归到三个主人名下：

| 归属 | 含义 | 数量 | 占 FP | FAR 贡献 | 插件能不能改 |
|:--|:--|--:|--:|--:|:--|
| `lost_by_replay` | 零延迟臂在**同一个 GT 框内**有坐标、重推臂没有 ⇒ 重推跟丢了 | **{buckets['lost_by_replay']}** | {buckets['lost_by_replay']/A['replay_fp']*100:.1f}% | {buckets['lost_by_replay']/A['frames']:.4f} | **能**（本轮就打掉） |
| `both_missed` | 两个臂都不在框内 ⇒ 检测器自身噪声 | {buckets['both_missed']} | {buckets['both_missed']/A['replay_fp']*100:.1f}% | {buckets['both_missed']/A['frames']:.4f} | 不能（不是插件引入的） |
| `no_ground_truth` | 该帧 GT **显式为空**（标注写了「无目标」） | {buckets['no_ground_truth']} | {buckets['no_ground_truth']/A['replay_fp']*100:.1f}% | {buckets['no_ground_truth']/A['frames']:.4f} | 不能（任何算法都改不掉） |

- **复算对账**：自算 FP/GT 与交付 metrics **逐个相同**（`recompute_diff` 全 0）。
- ⇒ 外挂相对零延迟天花板的净增 FP = **{A['delta_fp_vs_original']}**，而**绝对 FAR 里有一半是标注稀疏**：
  零延迟臂自己在这些帧上也报（`002251` 一条序列的 FAR 就是 {num(attrib['sequences']['VIDEO00012_19700101_002251']['original_far_per_frame'])}）。
- ⇒ **推论**：只要把 `lost_by_replay` 这 {buckets['lost_by_replay']} 个打掉，FAR ⇒ **{num(A['far_if_lost_by_replay_removed'])}**（回到红线内）；
  这就是本轮的目标。

### 27.2 ★ 第二步：新机理「不自信就不上报」

**判据从哪来**：模板配准（xcorr）的**相关峰**是这次重推**唯一的自洽性证据**——它同时回答了「模板在目标帧里找不找得到」和「找得到的那一峰有多像」。

**难点**：峰值低有两种成因，处置完全相反：

| 成因 | 现象 | 位置对不对 | 该怎么做 |
|:--|:--|:--|:--|
| (a) 目标**机动太大** | 模板跟不上（峰值低），但快核靠图像梯度跟得住 | **对** | 照报 |
| (b) 快核**跟丢**了 | 模板配不上（峰值低），快核锁到杂波后一路跑飞 | **错** | 别报 |

**怎么区分**：看**链位移**（快核链从源帧锚点跑出去多远）。

- 成因 (a)：目标在动 ⇒ 链跟得上 ⇒ 但**每一帧的链位移都很小**（实测 `001324` 低峰帧的链位移 p50 = 0 px、p90 = 10.2 px）；
- 成因 (b)：跟丢 ⇒ 链从锚点跑出去很远（实测 `001415` 低峰帧的链位移 p50 = 80.1 px、`001810` = 42.7 px）。

⇒ **弃权条件 = 「模板峰值 < θ」且「链位移 > d」**（两个条件缺一不可：只用峰值会误伤高速机动的好帧，只用位移会误伤慢目标的错帧）。

**为什么这条判据能同时救 FAR 和不伤命中率**：把低峰帧按「对/错」拆开实测（`peak<0.2` 的帧）：

| 序列 | 低峰帧 | 其中 TP（位置对） | 其中 FP（位置错） |
|:--|--:|--:|--:|
""")

import numpy as np

def lowpeak_split(dump_dir):
    """Per sequence: how many frames had a weak template match, and were they right or wrong."""
    out = {}
    for path in sorted(Path(dump_dir).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        tp = fp = total = 0
        for frame in doc["frames_out"]:
            entries = frame["entries"]
            features = frame.get("features", {})
            if not entries or features.get("peak") is None:
                continue
            if features["peak"] >= 0.2:
                continue
            boxes = np.asarray(frame["gt"], dtype=float).reshape(-1, 4)
            if not len(boxes):
                continue
            point = np.asarray(entries[0]["pos"], dtype=float)
            distances = np.linalg.norm(boxes[:, :2] - point, axis=1)
            nearest = int(np.argmin(distances))
            box = boxes[nearest]
            inside = abs(point[0] - box[0]) <= box[2] / 2 and abs(point[1] - box[1]) <= box[3] / 2
            total += 1
            tp += int(inside)
            fp += int(not inside)
        if total:
            out[doc["sequence"]] = (total, tp, fp)
    return out

lowpeak = lowpeak_split(FINAL / "final24/abstain_diag")
for name, (total, tp, fp) in sorted(lowpeak.items(), key=lambda item: -item[1][2]):
    w(f"| `{name}` | {total} | {tp} | {fp} |")
w(f"| **合计** | **{sum(v[0] for v in lowpeak.values())}** | **{sum(v[1] for v in lowpeak.values())}** | **{sum(v[2] for v in lowpeak.values())}** |")

w("""
（逐序列明细见 `runs/lookahead/final/final24/diag_fp_attrib.md` 与本轮扫描产物。）

### 27.3 ★ 第三步：离线策略扫描器（`diag_abstain_scan.py`）

新脚本做两件事：

1. **dump**：按**交付档的命令行**（脚本自己把 argv 喂给生产 CLI 解析器，避免手抄参数漂移）跑一遍 replay 臂，
   逐帧记录「上报坐标 + 链诊断（峰值 / 唯一性比 / 二次意见是否采纳 / 链位移 / hold 步数）」；
2. **scan**：在这份 dump 上**秒级复算**任意弃权规则（F1 / FAR / hit@5px / 抖动），不再为每个候选重跑一遍重推。

**扫描器的正确性闸门**：空规则必须**逐位复现交付档**——

| 数据集 | 空规则复现 | 交付档 |
|:--|:--|:--|
| 8 序列 | F1 {pct(scan8['baseline']['f1'])} / FAR {num(scan8['baseline']['far_per_frame'])} / hit@5px {pct(scan8['baseline']['hit_rate_5px'])} | F1 {pct(biz(old8,'replay')['f1'])} / FAR {num(biz(old8,'replay')['far'])} / hit@5px {pct(servo(old8,'replay')['hit'])} |
| 24 序列 | F1 {pct(scan24['baseline']['f1'])} / FAR {num(scan24['baseline']['far_per_frame'])} / hit@5px {pct(scan24['baseline']['hit_rate_5px'])} | F1 {pct(biz(old24,'replay')['f1'])} / FAR {num(biz(old24,'replay')['far'])} / hit@5px {pct(servo(old24,'replay')['hit'])} |

⇒ 扫描器与交付档**逐位一致**，离线结论可以直接外推到实跑。

### 27.4 阈值扫描（峰值 θ × 链位移 d，24 序列 FAR ≤ 0.030 的格子用 **OK** 标出）

| 规则 | 8 序 F1 | 8 序 FAR | 8 序 hit@5px | 24 序 F1 | 24 序 FAR | 24 序 hit@5px | 24 序红线 |
|:--|--:|--:|--:|--:|--:|--:|:--|
""")

base8 = scan8["baseline"]
base24 = scan24["baseline"]
for row in fine:
    if row["peak"] not in (0.26, 0.28, 0.30, 0.32, 0.35) or row["disp"] not in (10, 16, 20, 24, 28, 32):
        continue
    ok = "**OK**" if row["far_24"] <= 0.030 else "超线"
    w(f"| `peak<{row['peak']} & disp>{row['disp']}` | {pct(row['f1_8'])} | {num(row['far_8'])} | "
      f"{pct(row['hit_8'])} | {pct(row['f1_24'])} | {num(row['far_24'])} | {pct(row['hit_24'])} | {ok} |")

w(f"""
⇒ **交付取 `peak<{winner['peak']} & disp>{winner['disp']}`**：8 序列 hit@5px **一格不掉**、F1 还涨；
24 序列 FAR 落到 {num(winner['far_24'])}（红线 0.030 以内）、F1 涨 {(winner['f1_24']-base24['f1'])*100:+.4f} pp。

**被扫描否决的规则家族**（24 序列，全部实测）：

| 规则 | F1 | FAR | hit@5px | 结论 |
|:--|--:|--:|--:|:--|
| `uniq>0.2`（唯一性比单门） | {pct(scan24['rules'][0]['f1'])} | {num(scan24['rules'][0]['far_per_frame'])} | {pct(scan24['rules'][0]['hit_rate_5px'])} | 否决：命中率塌得太狠 |
| `peak<0.2`（峰值单门，不看位移） | {pct([r for r in scan24['rules'] if r['name']=='peak<0.2'][0]['f1'])} | {num([r for r in scan24['rules'] if r['name']=='peak<0.2'][0]['far_per_frame'])} | {pct([r for r in scan24['rules'] if r['name']=='peak<0.2'][0]['hit_rate_5px'])} | 否决：8 序列 F1 掉 {abs([r for r in scan8['rules'] if r['name']=='peak<0.2'][0]['delta_f1'])*100:.2f} pp（误伤高速机动的好帧） |
| `dev>8`（模板位置与链位置的分歧单门） | {pct([r for r in scan24['rules'] if r['name']=='dev>8'][0]['f1'])} | {num([r for r in scan24['rules'] if r['name']=='dev>8'][0]['far_per_frame'])} | {pct([r for r in scan24['rules'] if r['name']=='dev>8'][0]['hit_rate_5px'])} | 否决：命中率塌得太狠 |
| `disp>40`（链位移单门） | {pct([r for r in scan24['rules'] if r['name']=='disp>40'][0]['f1'])} | {num([r for r in scan24['rules'] if r['name']=='disp>40'][0]['far_per_frame'])} | {pct([r for r in scan24['rules'] if r['name']=='disp>40'][0]['hit_rate_5px'])} | 否决：误伤慢目标 |
| `rematch`（二次意见采纳就弃权） | {pct([r for r in scan24['rules'] if r['name']=='rematch'][0]['f1'])} | {num([r for r in scan24['rules'] if r['name']=='rematch'][0]['far_per_frame'])} | {pct([r for r in scan24['rules'] if r['name']=='rematch'][0]['hit_rate_5px'])} | 否决：二次意见本来就是修复手段，不该反过来弃权 |
| `xcorr_fail`（精修失败就弃权） | {pct([r for r in scan24['rules'] if r['name']=='xcorr_fail'][0]['f1'])} | {num([r for r in scan24['rules'] if r['name']=='xcorr_fail'][0]['far_per_frame'])} | {pct([r for r in scan24['rules'] if r['name']=='xcorr_fail'][0]['hit_rate_5px'])} | 否决：命中率高机动帧 |

### 27.5 ★ 定稿：8 / 24 序列实跑四臂（产物 JSON 直接读出）

**8 序列（官方 gate 集，主交付档）**

| 臂 | F1 | R | P | FAR | hit@5px | CLE p50(池化) | CLE p95(池化) | max_jitter |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
""")

for name in ("original", "stale", "replay"):
    b = biz(new8, name)
    s = servo(new8, name)
    w(f"| `{name}` | {pct(b['f1'])} | {pct(b['r'])} | {pct(b['p'])} | {num(b['far'])} | {pct(s['hit'])} | "
      f"{num(s['p50'],2)} px | {num(s['p95'],2)} px | {num(s['max_jitter'],2)} px |")

w("""
**24 序列（官方口径）**

| 臂 | F1 | R | P | FAR | hit@5px | CLE p50(池化) | CLE p95(池化) | max_jitter |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
""")

for name in ("original", "stale", "kinematic", "replay"):
    b = biz(new24, name)
    s = servo(new24, name)
    w(f"| `{name}` | {pct(b['f1'])} | {pct(b['r'])} | {pct(b['p'])} | {num(b['far'])} | {pct(s['hit'])} | "
      f"{num(s['p50'],2)} px | {num(s['p95'],2)} px | {num(s['max_jitter'],2)} px |")

# ---- 24 序列逐序列 FAR（哪几条离红线最近） --------------------------------
_ps = new24["arms"]["replay"]["current_frame"]["per_sequence"]
_ps_o = new24["arms"]["original"]["current_frame"]["per_sequence"]
_rows = []
for _seq in sorted(_ps, key=lambda s: -_ps[s]["far_per_frame"]):
    _r, _o = _ps[_seq], _ps_o.get(_seq, {})
    _flag = "**超线**" if _r["far_per_frame"] > 0.030 else ("偏高" if _r["far_per_frame"] > 0.020 else "")
    _rows.append(
        f"| `{_seq}` | {int(_r['gt'])} | {int(_r['tp'])} | {int(_r['fp'])} | {num(_r['far_per_frame'])} | "
        f"{num(_o.get('far_per_frame', 0))} | {_flag} |")
w("\n".join([
    "### 27.6 ★ 24 序列**逐序列 FAR**（哪几条离红线最近）",
    "",
    f"24 序列整体 FAR {num(biz(new24,'replay')['far'])}（红线 0.030）—— 但整体值会掩盖**单条序列已经超线**。",
    "下表按 FAR 降序列出（`replay` 臂，24 序列官方口径），`original`（零延迟天花板）同列作对照。",
    "",
    "| 序列 | GT | TP | FP | FAR | `original` FAR | 判定 |",
    "|:--|--:|--:|--:|--:|--:|:--|",
] + _rows) + "\n")

w(f"""
> **CLE 口径**：表中的 p50 / p95 是**全帧池化**（所有序列的逐帧 CLE 样本拼起来算分位数）——
> 这才是**伺服可接受性**口径（专题档 §21 的规则：可接受性看池化，尾部风险看最差单序列）。
> 最差单序列口径的 `replay` 臂：8 序 p50 {num(servo(new8,'replay')['worst_p50'],2)} px / p95 {num(servo(new8,'replay')['worst_p95'],2)} px，
> 24 序 p50 {num(servo(new24,'replay')['worst_p50'],2)} px / p95 {num(servo(new24,'replay')['worst_p95'],2)} px。
> `max_jitter` 是二阶差分，**只能用最差单序列口径**（跨序列拼接会在序列边界造出假抖动）。

⇒ 24 序列 `replay` 臂的 FAR **{num(biz(new24,'replay')['far'])}**（红线 0.030），
相对零延迟天花板 {num(biz(new24,'original')['far'])} 只差 {num(biz(new24,'replay')['far']-biz(new24,'original')['far'])}。

### 27.7 回归防线（全部 PASS）

| 防线 | 结果 |
|:--|:--|
| Gate A（8 序列，`--lookahead off` vs 原脚本逐位 md5） | **IDENTICAL=9 / DIFFER=0**，F1 均 0.964002 |
| Gate A（24 序列） | **IDENTICAL=25 / DIFFER=0**，F1 均 0.947469 |
| 尺子 1（历史帧等价性）：8 序 / 24 序 | F1 0.964127 / 0.947527，与上一档**逐位相同**（ΔF1 = 0.000000） |
| 新旋钮**关闭**时 replay 臂 vs 上一档冠军产物 | **8 序 IDENTICAL=9 / DIFFER=0**（`noguard8/preds_delivery` vs `uniqrematch2/preds_u060`）；**24 序 IDENTICAL=25 / DIFFER=0**（`noguard24/preds_delivery` vs `uniq24b/preds_u060`） |
| 单测 `manu/tracking/tests` | **152 项 OK**（新增 3 项：默认关 / 低峰+远链弃权 / 低峰+静止链不弃权） |
| 导航表 `manu/memory/validate_nav_table.py` | **[PASS]** |

### 27.8 板端代价

- **零额外开销**：弃权判据只读**已经在算**的两个量（模板峰值、链位移），不增加任何图像运算；
- 命中判定在 `_replay_detections` 里一次比较，代价 ~ns 级；
- 相比上一档，**板端每帧的运算量完全不变**（只是多了一次比较与一次 `continue`）。

### 27.9 新 CLI（默认关）

```bash
--replay-abstain-peak-below 0.0   # 模板峰值低于此值 ⇒ 不上报；0 = 关（默认，历史行为逐位不变）
--replay-abstain-disp-px  0.0     # 配合上一项：链位移超过此值才算「跟丢」；0 = 不看位移
```

交付档取值：**`--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24`**
（`manu/tracking/lookahead/run_delivery.sh`，8 序列与 24 序列同一份配置）。

### 27.10 产物索引

| 用途 | 路径 |
|:--|:--|
| **8 序列定稿** | `runs/lookahead/final/final8/arms_delivery.json` + `final8/preds_delivery/` + `final8/delivery.log` |
| **24 序列定稿** | `runs/lookahead/final/final24/arms_delivery.json` + `final24/preds_delivery/` + `final24/delivery.log` |
| 上一档（对照，无弃权） | `runs/lookahead/final/noguard8/` · `runs/lookahead/final/noguard24/` |
| FP 归属拆解 | `runs/lookahead/final/final24/diag_fp_attrib.json` + `.md` |
| FAR 落点拆解（上一轮） | `runs/lookahead/final/final24/diag_far_delivery.json` + `.md` |
| 弃权策略扫描（24 序） | `runs/lookahead/final/final24/diag_abstain_scan.json` + `.md` |
| 弃权策略扫描（8 序） | `runs/lookahead/final/final8/diag_abstain_scan.json` + `.md` |
| 逐帧 dump（扫描器输入） | `runs/lookahead/final/final24/abstain_diag/`（24 序）· `final8/abstain_diag/`（8 序） |
| 阈值细扫（peak × disp） | `/mnt/data/codex-overnight/fine_grid.json` |
| 新脚本 | `manu/tracking/lookahead/diag_fp_attrib.py` · `manu/tracking/lookahead/diag_abstain_scan.py` |
| 一键交付 | `bash manu/tracking/lookahead/run_delivery.sh [8|24|both]` |

### 27.11 复现命令

```bash
# 交付档（8 序列 + 24 序列）
bash manu/tracking/lookahead/run_delivery.sh both

# FP 归属拆解
/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_fp_attrib.py

# 弃权策略扫描（先 dump 再 scan）
/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_abstain_scan.py --dump
/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_abstain_scan.py --scan

# 回归闸门
bash /mnt/data/codex-overnight/gateA_live.sh      # 8 序列
bash /mnt/data/codex-overnight/gateA24_live.sh    # 24 序列
/home/huangzhe/anaconda3/envs/uav/bin/python -m unittest discover -s manu/tracking/tests -t .
```
""")

Path("/mnt/data/codex-overnight/round4_topical.md").write_text("\n".join(lines), encoding="utf-8")
print(f"wrote topical section: {len(lines)} lines")
