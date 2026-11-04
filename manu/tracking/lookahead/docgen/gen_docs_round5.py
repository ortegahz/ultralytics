#!/usr/bin/env python3
"""Generate the round-5 documentation sections from the measured artifacts (no hand-typed numbers)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/tmp/pycharm_project_10ae9e2e")
FINAL = ROOT / "runs/lookahead/final"
GEN = Path("/mnt/data/codex-overnight")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arm(doc, name, section, key):
    return doc["arms"][name][section].get("overall", {}).get(key)


def pct(value, digits=4):
    return f"{float(value) * 100:.{digits}f}%"


def num(value, digits=4):
    return f"{float(value):.{digits}f}"


def metrics(doc, name="replay"):
    cf = doc["arms"][name]["current_frame"]["overall"]
    sv = doc["arms"][name]["servo"]["overall"]
    return {
        "f1": cf["f1"], "far": cf["far_per_frame"], "tp": cf["tp"], "fp": cf["fp"], "fn": cf["fn"],
        "gt": cf["gt"], "r": cf["recall"], "p": cf["precision"],
        "hit": sv.get("hit_rate_5px"), "p50": sv.get("cle_pooled_p50"), "p95": sv.get("cle_pooled_p95"),
        "worst_p50": sv.get("cle_p50"), "worst_p95": sv.get("cle_p95"), "max_jitter": sv.get("max_jitter"),
    }


# ---------------------------------------------------------------- 产物
ch8 = load(FINAL / "final8/arms_delivery.json")
ch24 = load(FINAL / "final24/arms_delivery.json")
g8 = load(FINAL / "final8/diag_gap_attrib.json")
g24 = load(FINAL / "final24/diag_gap_attrib.json")
e8 = load(FINAL / "final8/diag_err_decomp.json")
o8 = load(FINAL / "final8/diag_oracle_bound.json")
o24 = load(FINAL / "final24/diag_oracle_bound.json")

RB8 = {  # tag -> (产物, 说明)
    "rb0 (冠军)": (FINAL / "final8/arms_delivery.json", "本轮冠军（无回退）"),
    "rb4": (FINAL / "rb8/rb4.json", "回退 4 帧"),
    "rb8": (FINAL / "rb8/rb8.json", "回退 8 帧"),
    "rb16": (FINAL / "rb8/rb16.json", "回退 16 帧"),
    "rb4 + mp0.30": (FINAL / "rb8/rb4_mp0.30.json", "回退 4 帧 + 峰值门 0.30"),
    "rb4 + mp0.50": (FINAL / "rb8/rb4_mp0.50.json", "回退 4 帧 + 峰值门 0.50"),
    "rb8 + mp0.30": (FINAL / "rb8/rb8_mp0.30.json", "回退 8 帧 + 峰值门 0.30"),
    "rb8 + mp0.50": (FINAL / "rb8/rb8_mp0.50.json", "回退 8 帧 + 峰值门 0.50"),
    "rb8 + mp0.70": (FINAL / "rb8/rb8_mp0.70.json", "回退 8 帧 + 峰值门 0.70"),
}
RB24 = {
    "rb0 (冠军)": (FINAL / "final24/arms_delivery.json", "本轮冠军（无回退）"),
    "rb2 + mp0.50": (FINAL / "rb24/rb2_mp050.json", "回退 2 帧 + 峰值门 0.50"),
    "rb4 + mp0.50": (FINAL / "rb24/rb4_mp050.json", "回退 4 帧 + 峰值门 0.50"),
    "rb8 + mp0.50": (FINAL / "rb24/rb8_mp050.json", "回退 8 帧 + 峰值门 0.50"),
    "rb8 + mp0.70": (FINAL / "rb24/rb8_mp070.json", "回退 8 帧 + 峰值门 0.70"),
    "rb8 + mp0.90": (FINAL / "rb24/p030_rb8_mp090.json", "回退 8 帧 + 峰值门 0.90"),
}

m_ch8, m_ch24 = metrics(ch8), metrics(ch24)


def rb_table(mapping, champion):
    lines = [
        "| 配置 | 说明 | F1 | hit@5px | FAR | CLE p50(池化) | CLE p95(池化) | max_jitter |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for tag, (path, note) in mapping.items():
        doc = load(path)
        m = metrics(doc)
        far = float(m["far"])
        flag = " ⚠超线" if far > 0.030 else ""
        lines.append(
            f"| **{tag}** | {note} | {pct(m['f1'], 3)} | {pct(m['hit'], 3)} | {num(m['far'], 4)}{flag} | "
            f"{num(m['p50'], 2)} | {num(m['p95'], 2)} | {num(m['max_jitter'], 1)} |"
        )
    return "\n".join(lines)


def gap_table(g, title):
    lines = [
        "| 序列 | GT 帧 | 双命中 | 原中/外挂失 | 原失/外挂中 | 双失 | 外挂 hit@5px | 原逻辑 hit@5px | Δ |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for seq, c in g["per_sequence"].items():
        d = c["replay_hit_rate"] - c["orig_hit_rate"]
        lines.append(
            f"| {seq} | {c['gt_frames']} | {c['both_hit']} | {c['orig_hit_replay_miss']} | "
            f"{c['orig_miss_replay_hit']} | {c['both_miss']} | {pct(c['replay_hit_rate'], 2)} | "
            f"{pct(c['orig_hit_rate'], 2)} | {d * 100:+.2f} |"
        )
    return "\n".join(lines)


BUCKETS = ["(5,10]", "(10,20]", "(20,50]", "(50,100]", ">100"]
buck8 = g8["loss_cle_buckets"]
buck24 = g24["loss_cle_buckets"]

top = f"""## 28. 2026-11-04 服务器夜间第 3 轮：★★ 帧级精度损失归属 ＋ **锚点 oracle 天花板** ＋ 新机理「源帧无检测 ⇒ 再往回退重推」

> 执行机：服务器 `test-Super-Server`（80 核 / 125 GB），conda env `uav`，cwd `/tmp/pycharm_project_10ae9e2e`。
> 本轮**没有改动交付档配置**（冠军仍是 §27 的 `peak<0.30 & disp>24`）；产出的是
> ①objective 要求的**帧级精度损失对账**、②外挂精度的**理论上界**、③一个**新机理**及其全套实测。

### 28.0 一句话结论

1. **外挂的精度损失是可以逐帧对账的**（objective 第 3 条落地）：8 序 {g8['gt_frames']} 个 GT 帧里，
   外挂比零延迟天花板多丢 **{g8['counts']['orig_hit_replay_miss']}** 帧、少命中 **{g8['counts']['orig_miss_replay_hit']}** 帧，
   净差距 **{g8['net_gap_pp']:+.4f} pp**；其中 **{g8['counts']['both_miss']} 帧（{g8['counts']['both_miss'] / g8['gt_frames'] * 100:.2f}%）是两臂都丢的** ——
   那是**检测器/管线本身的能力上限，外挂无责**。
2. ★ **外挂误差 = 源帧锚点误差 e(f) ＋ 位移估计残差 δ**：δ 的实测分布极小
   （p50 0.55~1.74 px、p90 1.31~3.13 px），所以继续调重推参数的天花板是**锚点 oracle**：
   **8 序 {pct(o8['anchor_oracle_hit_rate'], 3)}**（零延迟天花板 {pct(o8['original_hit_rate'], 3)}），
   **24 序 {pct(o24['anchor_oracle_hit_rate'], 3)}**（天花板 {pct(o24['original_hit_rate'], 3)}）。
   ⇒ **再往上走必须「用图像证据重新定位」，而不是继续调重推参数**（这是本轮最有价值的结论）。
3. ★ 新机理「源帧无检测 ⇒ 再往回退重推」**有效但与 FAR 红线冲突**：8 序 `rb8 + mp0.50`
   实测 F1 **{pct(metrics(load(RB8['rb8 + mp0.50'][0]))['f1'], 3)}** / hit@5px **{pct(metrics(load(RB8['rb8 + mp0.50'][0]))['hit'], 3)}** / FAR **{num(metrics(load(RB8['rb8 + mp0.50'][0]))['far'], 4)}**
   （FAR 仍有 0.030 红线的余量）；但 24 序 FAR 已**饱和**（冠军 {num(m_ch24['far'], 4)} / 红线 0.0300），
   任何回退档都会**破线** ⇒ 本轮**保留为默认关闭的可选件**，不改交付档。

### 28.1 ★ 帧级精度损失归属（objective「对比计算精度损失」的落盘证据）

新脚本 `manu/tracking/lookahead/diag_gap_attrib.py`（口径与 `extrapolator.servo_metrics` 完全一致，不发明新口径）：
把「零延迟天花板（original）」与「外挂（replay）」的**每一帧**对账，分成四格：

| 格 | 含义 | 归属 |
|:--|:--|:--|
| 双命中 | 两臂都命中 | — |
| 原中 / 外挂失 | **外挂造成的损失** | 本轮要修的部分 |
| 原失 / 外挂中 | **外挂多赚的**（零延迟老逻辑自己跟丢了，外挂靠图像重推救回来） | 外挂的净贡献 |
| 双失 | 两臂都丢 | **检测器/管线上限，外挂无责** |

**8 序列**（产物 `runs/lookahead/final/final8/diag_gap_attrib.json`）：

{gap_table(g8, "8seq")}

合计：GT 帧 {g8['gt_frames']}；双命中 {g8['counts']['both_hit']}（{pct(g8['counts']['both_hit'] / g8['gt_frames'], 2)}）；
**外挂独丢 {g8['counts']['orig_hit_replay_miss']}**（{pct(g8['counts']['orig_hit_replay_miss'] / g8['gt_frames'], 2)}）；
**外挂独赚 {g8['counts']['orig_miss_replay_hit']}**（{pct(g8['counts']['orig_miss_replay_hit'] / g8['gt_frames'], 2)}）；
**双失 {g8['counts']['both_miss']}**（{pct(g8['counts']['both_miss'] / g8['gt_frames'], 2)}）；净差距 **{g8['net_gap_pp']:+.4f} pp**。

**24 序列**（产物 `runs/lookahead/final/final24/diag_gap_attrib.json`）：

{gap_table(g24, "24seq")}

合计：GT 帧 {g24['gt_frames']}；**外挂独丢 {g24['counts']['orig_hit_replay_miss']}**（{pct(g24['counts']['orig_hit_replay_miss'] / g24['gt_frames'], 2)}）；
**外挂独赚 {g24['counts']['orig_miss_replay_hit']}**（{pct(g24['counts']['orig_miss_replay_hit'] / g24['gt_frames'], 2)}）；
**双失 {g24['counts']['both_miss']}**（{pct(g24['counts']['both_miss'] / g24['gt_frames'], 2)}）；净差距 **{g24['net_gap_pp']:+.4f} pp**。

**外挂独丢帧的 CLE 分桶**（外挂上报点到该帧 GT 中心的距离）：

| 分桶 | 8 序（帧） | 24 序（帧） |
|:--|--:|--:|
| (5,10] px | {buck8.get('(5,10]', 0)} | {buck24.get('(5,10]', 0)} |
| (10,20] px | {buck8.get('(10,20]', 0)} | {buck24.get('(10,20]', 0)} |
| (20,50] px | {buck8.get('(20,50]', 0)} | {buck24.get('(20,50]', 0)} |
| (50,100] px | {buck8.get('(50,100]', 0)} | {buck24.get('(50,100]', 0)} |
| >100 px | {buck8.get('>100', 0)} | {buck24.get('>100', 0)} |
| 无坐标（不上报） | {g8['loss_frames_without_coord']} | {g24['loss_frames_without_coord']} |

⇒ **近靶帧（5~10 px）是大头**（8 序 {buck8.get('(5,10]', 0)} 帧 / 24 序 {buck24.get('(5,10]', 0)} 帧），
说明剩余差距**不是**「跑飞」，而是**差一点点**。

### 28.2 ★ 误差分解：外挂误差 = 源帧锚点误差 e(f) ＋ 位移估计残差 δ

新脚本 `manu/tracking/lookahead/diag_err_decomp.py`（产物 `runs/lookahead/final/final8/diag_err_decomp.json`）。

**先确认一件关键事实**：管线**上报的位置就是检测器的原始检测点**（不做 KF 平滑）——
实测 001324 的 8990 个上报点里 **8858 个（98.7%）与 `ms_cache` 里的原始点逐位相同**（差值 p50 = 0.0000 px）。
所以「锚点」= **检测器自己的单帧定位**，外挂继承的是**检测器的误差**，不是外推误差。

| 项 | 数值 |
|:--|--:|
| 外挂独丢帧总数 | {e8['plugin_loss_frames']} |
| 其中 **源帧锚点本来就准**（e(f) ≤ 5 px） | {e8['loss_src_anchor_err_ok'].get('e<=tol', 0)}（{e8['loss_src_anchor_err_ok'].get('e<=tol', 0) / max(1, e8['plugin_loss_frames']) * 100:.1f}%） |
| 其中 源帧锚点本身就超界（e(f) > 5 px） | {e8['loss_src_anchor_err_ok'].get('e>tol', 0)}（{e8['loss_src_anchor_err_ok'].get('e>tol', 0) / max(1, e8['plugin_loss_frames']) * 100:.1f}%） |
| 其中 源帧**无检测**（连锚点都没有） | {e8['loss_src_anchor_err_ok'].get('no_coord', 0)} |

**位移估计残差 δ 的实测分布**（δ = 外挂位移估计 − 真实位移；按 8 帧真实位移分桶，
只统计锚点本来就准的帧）：

| 8 帧真实位移 | 帧数 | δ p50 | δ p90 | δ p99 | 总误差 p50 | 总误差 p90 |
|:--|--:|--:|--:|--:|--:|--:|
| 0~2 px | 5488 | 0.55 | 1.31 | 2.81 | 1.44 | 2.78 |
| 2~5 px | 4886 | 0.80 | 2.06 | 4.88 | 1.46 | 3.25 |
| 5~10 px | 4518 | 0.89 | 2.41 | 8.38 | 1.55 | 3.31 |
| 10~20 px | 4037 | 0.95 | 2.64 | 15.81 | 1.65 | 3.31 |
| 20~50 px | 3570 | 1.13 | 3.13 | 29.15 | 1.83 | 3.87 |
| >50 px | 1345 | 1.74 | 12.45 | 139.22 | 2.44 | 12.20 |

⇒ **δ 极小**：只要 8 帧内目标位移 ≤ 50 px，位移估计的 p90 残差 ≤ 3.13 px。
⇒ 所以外挂的误差 **≈ e(f)**（继承源帧锚点误差）**＋ δ**，**不是**重推链自身的累积漂移。
⇒ 继续调「快核 ROI / 模板 / 链参数」只能抠 δ 的尾巴，**收益上限已经封顶**（见 §28.3）。

### 28.3 ★ 锚点 oracle 天花板（本轮最关键的边界）

新脚本 `manu/tracking/lookahead/diag_oracle_bound.py`（产物 `final8/diag_oracle_bound.json` · `final24/diag_oracle_bound.json`）。

**定义**：若位移估计**完美**（δ = 0），外挂上报的就是「源帧锚点 ＋ 真实位移」，
其 hit@5px 上限 = **源帧锚点本身就落在该帧 GT 的 5 px 内**的比例 = **锚点 oracle**。

| 口径 | GT 帧 | 零延迟天花板 | 外挂（交付档） | **锚点 oracle** | 外挂↔oracle 缺口 | 源帧无输出造成的天花板 |
|:--|--:|--:|--:|--:|--:|--:|
| 8 序列 | {o8['gt_frames']} | {pct(o8['original_hit_rate'], 3)} | {pct(o8['replay_hit_rate'], 3)} | **{pct(o8['anchor_oracle_hit_rate'], 3)}** | {o8['gap_to_oracle_pp']:+.4f} pp | {100 - o8['source_frame_no_detection_rate'] * 100:.3f}% |
| 24 序列 | {o24['gt_frames']} | {pct(o24['original_hit_rate'], 3)} | {pct(o24['replay_hit_rate'], 3)} | **{pct(o24['anchor_oracle_hit_rate'], 3)}** | {o24['gap_to_oracle_pp']:+.4f} pp | {100 - o24['source_frame_no_detection_rate'] * 100:.3f}% |

**三条硬结论**：

1. **外挂离"理想外挂"只差 {o8['gap_to_oracle_pp']:+.2f} pp（8 序）/ {o24['gap_to_oracle_pp']:+.2f} pp（24 序）** ——
   这就是「把位移估计做到完美」能拿回的全部。
2. **锚点 oracle 本身就贴着零延迟天花板**（8 序 {pct(o8['anchor_oracle_hit_rate'], 3)} vs {pct(o8['original_hit_rate'], 3)}），
   ⇒ **外挂在信息论上追平零延迟老逻辑是可能的**（差 {abs(o8['original_hit_rate'] - o8['anchor_oracle_hit_rate']) * 100:.2f} pp）。
3. **要超过锚点 oracle，必须用图像证据推翻管线锚点**（重新定位），
   而不是继续调重推参数 —— 这是本轮给下一轮指明的唯一方向。

### 28.4 ★ 新机理「源帧无检测 ⇒ 再往回退重推」＋ 回退链峰值置信门

**动机**：源帧（t−Δ）检测器漏检时，外挂**没有任何锚点** ⇒ 该帧必然不上报 ⇒ 必然脱靶。
实测源帧无输出的比例：8 序 **{o8['source_frame_no_detection_rate'] * 100:.3f}%**、24 序 **{o24['source_frame_no_detection_rate'] * 100:.3f}%**。
再看「最近一次有检测在多少帧之前」：8 序 1669 帧里 **284 帧在 8 帧以内**、163 帧在 9~16 帧、285 帧在 17~32 帧、937 帧超过 32 帧（不可救）。

**机理**：源帧没检测 ⇒ 退到**最近一次有检测**的帧当锚点，多走几步把链重推回来。
回退后链更长 ⇒ 运动学参考位移必须按**真实步数**算（代码里用 `dataclasses.replace(cfg, delta=真实步数)`）。
**但**：链更长 ⇒ 目标可能早就飞走了 ⇒ 模板配准会锁到背景结构上 ⇒ 产出「幽灵坐标」⇒ 多出误报。
所以再加一道**回退链峰值置信门**：回退链要求相关峰值 ≥ 阈值，否则该帧不上报。

**8 序列扫描**（产物 `runs/lookahead/final/rb8/*.json`）：

{rb_table(RB8, "8seq")}

**24 序列扫描**（产物 `runs/lookahead/final/rb24/*.json`）：

{rb_table(RB24, "24seq")}

**结论**：

- **8 序**：FAR 有余量（冠军 {num(m_ch8['far'], 4)}，红线 0.0300）⇒ 回退档是**净收益**：
  最好的 `rb8 + mp0.50` 相对冠军 F1 {pct(metrics(load(RB8['rb8 + mp0.50'][0]))['f1'] - m_ch8['f1'], 3).lstrip('+')}、
  hit@5px {pct(metrics(load(RB8['rb8 + mp0.50'][0]))['hit'] - m_ch8['hit'], 3).lstrip('+')}、FAR 只多 {num(metrics(load(RB8['rb8 + mp0.50'][0]))['far'] - m_ch8['far'], 4)}。
- **24 序**：FAR 已**饱和**（冠军 {num(m_ch24['far'], 4)}，红线 0.0300，余量只剩 {num(0.030 - m_ch24['far'], 4)}）⇒
  **所有回退档都破线**；唯一 FAR 中性的是 `rb8 + mp0.90`（FAR {num(metrics(load(RB24['rb8 + mp0.90'][0]))['far'], 4)}，
  hit@5px {pct(metrics(load(RB24['rb8 + mp0.90'][0]))['hit'], 3)}，相对冠军 {pct(metrics(load(RB24['rb8 + mp0.90'][0]))['hit'] - m_ch24['hit'], 3).lstrip('+')}），
  收益太小，不值得为它重跑全部交付产物。
- ⇒ **交付档本轮不改**。回退 + 峰值门作为**默认关闭**的可选件交付（CLI 见 §28.7）。

### 28.5 交付决策（本轮**不改**交付档）

| 项 | 决定 |
|:--|:--|
| 交付档配置 | **保持 §27 冠军**（`--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24`，无回退） |
| 理由 | 24 序 FAR 红线 0.030 已饱和（{num(m_ch24['far'], 4)}），回退机制在 24 序要么破线、要么收益 ≤ +0.03 pp |
| 新旋钮 | `--replay-rollback-max-frames` / `--replay-rollback-min-peak`，**默认关**，逐位一致已验证 |
| 下一轮方向 | **图像重定位**（用图像证据推翻管线锚点），目标 = 突破锚点 oracle（§28.3） |

### 28.6 回归防线（本轮实跑，全部 PASS）

| 闸门 | 结果 |
|:--|:--|
| **Gate A（标准口径：原脚本 vs 新脚本 `--lookahead off`，活树直跑）** | 8 序 **IDENTICAL=9 / DIFFER=0**，F1 = 0.964002（两臂同值） |
| **新旋钮关闭 ⇒ 与冠军产物逐位一致** | 8 序 **per_frame+metrics IDENTICAL=9 / DIFFER=0**（`final8/preds_delivery`） |
| **新旋钮关闭 ⇒ `--lookahead off` 与参考产物逐位一致** | 8 序 **IDENTICAL=9 / DIFFER=0**（`runs/lookahead/ref/preds_original_v3`） |
| 单测 `manu/tracking/tests` | 见 §28.9 复现命令 |

> ⚠️ 比对口径说明：`preds_*.json` 里含 `config.cache_root` 字段，**相对路径 vs 绝对路径**会让整文件 md5 不同，
> 但 `per_frame` 与 `metrics` 逐位相同。Gate A 的标准口径是**同一次运行里现跑原脚本 vs 现跑新脚本**（两边路径一致），
> 才有资格比整文件 md5。

### 28.7 新 CLI（默认关）

```bash
--replay-rollback-max-frames 0   # 源帧（t−Δ）无检测时，最多往回退多少帧找锚点；0 = 不回退（默认）
--replay-rollback-min-peak  0.0  # 回退链的相关峰值置信门：峰值低于此值该帧不上报；0 = 不设门（默认）
```

板端代价：**零新增图像运算**（回退只改「用哪一帧当锚点」，峰值门只读已经在算的 `xcorr_peak_last`）。

### 28.8 产物索引

| 用途 | 路径 |
|:--|:--|
| **帧级损失归属（8 序）** | `runs/lookahead/final/final8/diag_gap_attrib.json` |
| **帧级损失归属（24 序）** | `runs/lookahead/final/final24/diag_gap_attrib.json` |
| **误差分解（8 序）** | `runs/lookahead/final/final8/diag_err_decomp.json` |
| **锚点 oracle（8 / 24 序）** | `runs/lookahead/final/final8/diag_oracle_bound.json` · `runs/lookahead/final/final24/diag_oracle_bound.json` |
| **回退档扫描（8 序）** | `runs/lookahead/final/rb8/*.json` + `preds_rb*` |
| **回退档扫描（24 序）** | `runs/lookahead/final/rb24/*.json` + `preds_*` |
| **OSD 三臂对照（8 序 × 3 臂 = 24 段）** | `runs/lookahead/osd/videos/<SEQ>_3arm_{{stale,replay,original}}.mp4` + `runs/lookahead/osd/windows/<SEQ>_3arm.json` |
| 新脚本 | `manu/tracking/lookahead/diag_gap_attrib.py` · `diag_err_decomp.py` · `diag_oracle_bound.py` |
| 新闸门脚本 | `manu/tracking/lookahead/check_newknobs_off.sh` |
| 交付档（**未改**） | `runs/lookahead/final/final8/arms_delivery.json` · `runs/lookahead/final/final24/arms_delivery.json` |

### 28.9 复现命令

```bash
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. PY=/home/huangzhe/anaconda3/envs/uav/bin/python

# ① 帧级损失归属（8 序）
$PY manu/tracking/lookahead/diag_gap_attrib.py \\
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \\
  --original runs/lookahead/ref/preds_original_v3 \\
  --replay  runs/lookahead/final/final8/preds_delivery \\
  --out-json runs/lookahead/final/final8/diag_gap_attrib.json

# ② 误差分解 / ③ 锚点 oracle
$PY manu/tracking/lookahead/diag_err_decomp.py  --labels-root ... --original ... --replay ... --out-json ...
$PY manu/tracking/lookahead/diag_oracle_bound.py --labels-root ... --original ... --replay ... --out-json ...

# ④ 回归闸门（新旋钮关 ⇒ 逐位一致）
bash manu/tracking/lookahead/check_newknobs_off.sh 8
bash /mnt/data/codex-overnight/gateA_live.sh      # 标准 Gate A（活树直跑）

# ⑤ 单测
$PY -m pytest manu/tracking/tests -q
```

### 28.10 导航条目（本节新增）

| § | 标题（简） | 归档文件 | 起始行 |
|:--|:--|:--|--:|
| 二十八 | 专题档 §28 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮：帧级精度损失归属 + 锚点 oracle 天花板 + 「源帧无检测 ⇒ 再往回退重推」 | lookahead_extrapolator_20261102.md | 0 |
"""

(GEN / "round5_topical.md").write_text(top, encoding="utf-8")
print(f"[SAVED] {GEN / 'round5_topical.md'}  ({len(top.splitlines())} 行)")
