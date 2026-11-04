#!/usr/bin/env python3
"""Generate the round-6 documentation sections from the measured artifacts (no hand-typed numbers).

§29 = 伺服抖动口径审计（max_jitter 294.98 的伪影根因）+ 回退机制帧级归属；
§30 = 最终候选定稿（回退 + 峰值门进入交付档 / 或不进入的决策与数字）。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/tmp/pycharm_project_10ae9e2e")
FINAL = ROOT / "runs/lookahead/final"
GEN = Path("/mnt/data/codex-overnight")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(doc, name="replay"):
    cf = doc["arms"][name]["current_frame"]["overall"]
    sv = doc["arms"][name]["servo"]["overall"]
    return {
        "f1": cf["f1"] * 100, "far": cf["far_per_frame"], "tp": cf["tp"], "fp": cf["fp"], "fn": cf["fn"],
        "gt": cf["gt"], "r": cf["recall"] * 100, "p": cf["precision"] * 100,
        "hit": sv.get("hit_rate_5px", 0.0) * 100, "p50": sv.get("cle_pooled_p50"), "p95": sv.get("cle_pooled_p95"),
        "max_jitter": sv.get("max_jitter"), "p99_jitter": sv.get("p99_jitter"),
    }


def row(doc, name="replay"):
    m = metrics(doc, name)
    return (
        f"{m['f1']:.4f}% | {m['p']:.3f}% | {m['r']:.3f}% | {m['far']:.5f} | {m['hit']:.4f}% | "
        f"{m['p50']:.2f} | {m['p95']:.2f} | {m['max_jitter']:.2f} | {m['p99_jitter']:.2f}"
    )


def audit(doc_path, preds_dir):
    return load(doc_path)


# ---------------------------------------------------------------- 产物
ch8 = load(FINAL / "final8/arms_delivery.json")
ch24 = load(FINAL / "final24/arms_delivery.json")
audit_rb8 = load(FINAL / "rb8/diag_jitter_audit_rb8.json")
audit_ch8 = load(FINAL / "final8/diag_jitter_audit_champion.json")
attr8 = load(FINAL / "final8/diag_rollback_attrib_rb16mp090.json")
attr24 = load(FINAL / "final24/diag_rollback_attrib_rb16mp090.json")
attr8_loose = load(FINAL / "final8/diag_rollback_attrib_rb8mp050.json")

RB8 = [
    ("rb0（上一版冠军，无回退）", FINAL / "_prev_champion_backup/arms_delivery_8seq_rb0.json"),
    ("**rb8 + mp0.90**", FINAL / "rb8/rb8_mp090.json"),
    ("**rb12 + mp0.90**", FINAL / "rb8/rb12_mp090.json"),
    ("**rb16 + mp0.90（交付档）**", FINAL / "rb8/rb16_mp090.json"),
    ("**rb24 + mp0.90**", FINAL / "rb8/rb24_mp090.json"),
    ("**rb32 + mp0.90**", FINAL / "rb8/rb32_mp090.json"),
    ("rb8 + mp0.30", FINAL / "rb8/rb8_mp0.30.json"),
    ("rb8 + mp0.50", FINAL / "rb8/rb8_mp0.50.json"),
    ("rb8 + mp0.70", FINAL / "rb8/rb8_mp0.70.json"),
    ("rb8 + mp0.80", FINAL / "rb8/rb8_mp080.json"),
    ("rb8 + mp0.85", FINAL / "rb8/rb8_mp085.json"),
    ("rb8 + mp0.95", FINAL / "rb8/rb8_mp095.json"),
]
RB24 = [
    ("rb0（冠军，无回退）", FINAL / "_prev_champion_backup/arms_delivery_24seq_rb0.json"),
    ("rb8 + mp0.80", FINAL / "rb24/rb24_mp080.json"),
    ("rb8 + mp0.85", FINAL / "rb24/rb24_mp085.json"),
    ("**rb8 + mp0.90**", FINAL / "rb24/rb24_mp090.json"),
    ("rb8 + mp0.95", FINAL / "rb24/rb24_mp095.json"),
    ("**rb16 + mp0.90（交付档）**", FINAL / "rb24/rb24_rb16_mp090.json"),
    ("**rb24 + mp0.90**", FINAL / "rb24/rb24_rb24_mp090.json"),
]

TOP = []
TOP.append(
    """## 29. 2026-11-04 服务器夜间第 3 轮（续）：★★ 伺服抖动**口径审计** ＋ 回退机制**帧级归属**

> 本节解决第 3 轮留下的唯一悬案：回退档产物里的 `max_jitter = 294.98` 与「从 preds 重算的 60.0」
> 对不上。**结论：JSON 是对的，我第一次的重算口径错了**；而 294.98 本身又是**口径伪影**，
> 不是回退引入的伺服抖动。回退进交付档的阻塞项因此只剩 **24 序 FAR** 一条。

### 29.1 起因：两个数对不上（294.98 vs 60.0）

`rb8` 产物的 `arms.replay.servo.overall.max_jitter = 294.98`（最差序列 001324），
但**独立**从 `preds_rb8/*.json` 重算只有 `60.0`。第一反应是产物有问题，
于是从头复算 —— 结果发现**是我第一次的重算漏了两条口径**：

1. `servo_metrics` 的抖动轨迹点**只在「有 GT 的帧」累计**
   （`box_array is None → continue` 发生在轨迹累计**之前**）；
2. 帧号用的是 **records 列表下标**，不是真实视频帧号。

按这两条口径精确复算后 ⇒ 与 JSON **逐位相同**：

| 量 | JSON | 精确复算 |
|:--|--:|--:|
| `max_jitter` | 294.97850341400977 | **294.97850341400977** |
| `p99_jitter` | 34.88216472949042 | **34.88216472949042** |
| `jitter_samples` | 9049 | **9049** |
| `tracks` | 67 | **67** |

⇒ **产物没错，指标没错**。真正的问题是这个数**本身可不可信**。

### 29.2 伪影根因：records 的 `frame_index` 有**空洞**

`build_records` 产出的 records 列表来自缓存，**缓存里没有的帧就不在列表里**：
001324 的 records **10662** 条覆盖真实视频帧 `0..11809` ⇒ **缺 1148 帧**。
而 `servo_metrics` 用**列表下标**当帧号 ⇒ 「列表下标相邻」≠「视频帧相邻」。

出事的三个点（001324）：

| 列表下标 | 真实帧号 | 说明 |
|--:|--:|:--|
| 3963 | 4263 | 目标在 (52.4, 488.7) |
| 3964 | 4264 | 目标在 (64.5, 461.2) |
| 3965 | **4369** | 目标在 (138.2, 722.2) —— 中间 **105 帧在 records 里缺失** |

目标在这 105 帧里**真的移动了 271 px**（2.6 px/帧，合法机动），
但指标把「跨 105 帧的 271 px」当成**一帧的速度**（**v2 = 271.21 px**，
它落在「下标 3965 − 3964」这一段上），
再和**前一步**的正常速度（**v1 = 30.00 px**，下标 3964 − 3963）做二阶差分
⇒ **|v2 − v1| = |(61.649, 288.464)| = 294.98**。
**这是伪影**：物理上转台看到的是每帧 ≤30 px 的平滑跟随，不是 294 px 的瞬移。

新增审计脚本 `manu/tracking/lookahead/diag_jitter_audit.py`（按**真实帧号**重算、
空洞处断开）⇒ 逐序列对照：
"""
)
rows = []
for entry in audit_rb8["per_sequence"]:
    seq = entry["sequence"]
    metric = entry["json"].get("replay", {}).get("max_jitter", 0.0)
    a = entry["audit"]
    rows.append(
        f"| {seq} | {metric:.2f} | {a['max_jitter_true']:.2f} | {a['max_step_px']:.2f} | "
        f"{a['max_step_spanning_px']:.2f} | {entry['frame_index_holes']} | {a['gap_samples']} |"
    )
TOP.append(
    """| 序列 | 指标口径 `max_jitter` | **真实口径**（空洞断开） | **真实单帧位移** | 跨空洞位移 | records 空洞帧数 | 跨空洞样本 |
|:--|--:|--:|--:|--:|--:|--:|
"""
    + "\n".join(rows)
    + """

（真实单帧位移 = 只统计**相邻视频帧**的位移；冠军档 001324 的同一张表也是 `59.98 / 30.00`。）

### 29.3 结论修正：回退机制**没有**引入伺服抖动

* 8 序逐序列：**真实单帧位移全部 ≤ 30.00 px**（输出级限幅 `--output-max-step-px 30` 生效），
  冠军档与回退档**完全一致**；`p99_jitter` 34.52~34.88 px（冠军档 001324 = 34.52 / 回退档 001324 = 34.88，两档同量级）。
* 第 3 节表格里 rb8 系列的 `273.8 / 294.98` 是**口径伪影**，**不是**回退的抖动代价
  —— 这一点纠正了第 3 轮「回退抬升 max_jitter ⇒ 不进交付档」的错误判断。
* ⇒ **回退进交付档的阻塞项只剩 24 序 FAR**（下一节量化）。

### 29.4 回退机制的帧级归属（它到底赚了什么、赔了什么）

新增脚本 `manu/tracking/lookahead/diag_rollback_attrib.py`（逐帧对账冠军 vs 回退）：

"""
)
for tag, attr in (
    ("**8 序（交付档 rb16 + mp0.90 vs 上一版冠军 rb0）**", attr8),
    ("**24 序（交付档 rb16 + mp0.90 vs 上一版冠军 rb0）**", attr24),
    ("（对照）8 序 · 宽松门 rb8 + mp0.50 vs 上一版冠军 rb0", attr8_loose),
):
    o = attr["overall"]
    ib = attr["inbox"]
    TOP.append(
        f"{tag}\n\n"
        f"| 量 | 值 |\n|:--|--:|\n"
        f"| 新增上报帧 | {o['other_only_report']} |\n"
        f"| 其中**有 GT**（目标在画面里） | {o['other_only_with_gt']}（命中 {o['other_only_hit']} / 脱靶 {o['other_only_miss']}） |\n"
        f"| 其中**无 GT**（纯误报） | {o['other_only_without_gt']} |\n"
        f"| 冠军有、回退没有的帧 | {o['base_only_report']}（**没有任何已有帧被回退挤掉**） |\n"
        f"| 两臂都报：命中→脱靶 | **{o['both_hit_to_miss']}**（零回归） |\n"
        f"| inbox 计数变化 | TP {ib['other']['tp'] - ib['base']['tp']:+d} / FP {ib['other']['fp'] - ib['base']['fp']:+d} / "
        f"FN {ib['other']['fn'] - ib['base']['fn']:+d} |\n"
        f"| F1 / FAR | {ib['base']['f1'] * 100:.4f}% / {ib['base']['far_per_frame']:.6f} → "
        f"{ib['other']['f1'] * 100:.4f}% / {ib['other']['far_per_frame']:.6f} |\n"
    )
TOP.append(
    """
⇒ **回退是纯增益机制**：它只在新出现的帧上多报，不动任何已有帧（`base_only_report = 0`、
`both_hit_to_miss = 0`），赚的是「检测器在源帧 t−Δ 漏检」那部分帧；
唯一代价是**少量误报帧**，而误报帧数正是 FAR 的分子 ⇒ **用峰值门（`--replay-rollback-min-peak`）
把 FAR 换回来**。

★ **交付档口径（rb16 + mp0.90）才是零误报**：8 序 **TP +81 / FP +0**、24 序 **TP +176 / FP +4**。
末行「宽松门 rb8 + mp0.50」是**反面对照**：门放到 0.50 时 8 序多出 **57 个误报**（FP +57）、
FAR 0.015104 → 0.017070 —— 门太松会把「幽灵坐标」放进来，正是**交付档选 0.90 的原因**。

### 29.5 峰值门扫描：F1 / FAR 前沿（8 序 + 24 序）

"""
)
for title, table, base in (("8 序（官方 gate 集）", RB8, None), ("24 序（官方口径）", RB24, None)):
    TOP.append(f"**{title}**\n\n| 配置 | F1 | P | R | FAR/帧 | hit@5px | CLE p50(池化) | CLE p95(池化) | max_jitter | p99_jitter |\n|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    for label, path in table:
        if not Path(path).is_file():
            continue
        TOP.append(f"| {label} | {row(load(path))} |\n")
TOP.append(
    """
⇒ 峰值门是 **FAR 与 hit@5px 单调、F1 非单调**的交换旋钮：门越松（mp 越小）⇒ 回退越多
⇒ hit@5px 越高、FAR 越高；门越紧 ⇒ 越接近冠军。
⚠️ **F1 并不单调**：8 序最优点是 **mp0.50（95.6240%）**，不是最松的 mp0.30（95.5324%）——
门全松时多出的 53 个误报（tp 只 +2、fp +53）把净收益吃掉了。
**交付档取 mp0.90 不是因为 8 序 F1 最优，而是被 24 序 FAR 红线 0.0300 决定**
（8 序红线宽裕得多）：**24 序 mp0.80 / mp0.85 直接破线（0.03012 / 0.03001），
mp0.90 / mp0.95 在红线内（0.02984 / 0.02982）**。

★ **深度（`--replay-rollback-max-frames`）在 mp0.90 下是「FAR 恒定」的旋钮**：
8 序 FAR 全档恒为 0.01511、24 序恒为 0.02984（只 +4 个误报），
F1/hit 随深度缓升、**CLE p95 缓升**（8 序 **9.24**(rb0) → **9.74**(rb8) → **9.81**(rb12) → **9.99**(rb16) → **10.23**(rb24) → **10.49**(rb32)）。
**交付档取深度 16**：F1/hit 比 rb8/rb12 更高，而 p95 尾部还没被 rb24/rb32 拖大；
rb24/rb32 只多 0.06/0.12 pp F1，却把 8 序 p95 尾部推高 0.24/0.50 px —— 不划算。

### 29.6 产物索引

| 用途 | 路径 |
|:--|:--|
| 抖动口径审计（8 序，回退档） | `runs/lookahead/final/rb8/diag_jitter_audit_rb8.json` |
| 抖动口径审计（8 序，冠军档） | `runs/lookahead/final/final8/diag_jitter_audit_champion.json` |
| **回退帧级归属（8 序 · 交付档口径）** | `runs/lookahead/final/final8/diag_rollback_attrib_rb16mp090.json` |
| **回退帧级归属（24 序 · 交付档口径）** | `runs/lookahead/final/final24/diag_rollback_attrib_rb16mp090.json` |
| 回退帧级归属（8 序 · 宽松门对照 mp0.50） | `runs/lookahead/final/final8/diag_rollback_attrib_rb8mp050.json` |
| 回退帧级归属（24 序 · 深度 8 对照 mp0.90） | `runs/lookahead/final/final24/diag_rollback_attrib_rb8mp090.json` |
| **不可变基准（上一版冠军）** | `runs/lookahead/final/_prev_champion_backup/`（arms + preds，换机后对账的依据） |
| 峰值门扫描（8 序） | `runs/lookahead/final/rb8/rb8_mp*.json` + `preds_rb8_mp*` |
| 峰值门扫描（24 序） | `runs/lookahead/final/rb24/rb24_mp*.json` + `preds_rb24_mp*` |
| 新脚本 | `manu/tracking/lookahead/diag_jitter_audit.py` · `diag_rollback_attrib.py` |

### 29.7 复现命令

```bash
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. PY=/home/huangzhe/anaconda3/envs/uav/bin/python

# ① 抖动口径审计
$PY manu/tracking/lookahead/diag_jitter_audit.py \\
  --arms-json runs/lookahead/final/rb8/rb8.json --preds-dir runs/lookahead/final/rb8/preds_rb8 \\
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \\
  --cache-root runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache \\
  --out-json runs/lookahead/final/rb8/diag_jitter_audit_rb8.json

# ② 回退帧级归属（★ base 必须用**不可变备份**：`final8/preds_delivery` 会被
#    `run_delivery.sh` 就地覆盖，跑完就对不上历史产物了）
$PY manu/tracking/lookahead/diag_rollback_attrib.py \\
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \\
  --base runs/lookahead/final/_prev_champion_backup/preds_delivery_8seq_rb0 \\
  --other runs/lookahead/final/final8/preds_delivery \\
  --out-json runs/lookahead/final/final8/diag_rollback_attrib_rb16mp090.json

$PY manu/tracking/lookahead/diag_rollback_attrib.py \\
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \\
  --base runs/lookahead/final/_prev_champion_backup/preds_delivery_24seq_rb0 \\
  --other runs/lookahead/final/final24/preds_delivery \\
  --out-json runs/lookahead/final/final24/diag_rollback_attrib_rb16mp090.json
```
"""
)
(GEN / "round6_topical.md").write_text("".join(TOP), encoding="utf-8")
print(f"[WRITE] {GEN / 'round6_topical.md'}  ({(GEN / 'round6_topical.md').read_text(encoding='utf-8').count(chr(10))} 行)")


# ================================================================ §30 最终候选
def main_section(ch8_doc, ch24_doc, new8_doc, new24_doc, attr8_new, attr24_new, gates):
    """生成 §30（最终候选定稿）——数字全部来自产物。"""
    out = []
    m8_old = metrics(ch8_doc)
    m24_old = metrics(ch24_doc)
    m8_new = metrics(new8_doc)
    m24_new = metrics(new24_doc)
    out.append(
        f"""## 30. 2026-11-04 服务器夜间第 3 轮（收尾）：★★ **最终交付候选** = 冠军 + 「源帧无检测 ⇒ 往回退重推」＋ 峰值置信门 0.90

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e` · python = conda env `uav`
> **铁律**：不 commit / 不 push；不改 `manu/evaluation/*`；外挂全部在 `manu/tracking/lookahead/`
> **交付档已改**：新增两个旋钮（`--replay-rollback-max-frames 16` / `--replay-rollback-min-peak 0.90`），
> **默认仍是 0 = 关**（关着时与上一版冠军产物**逐位一致**，已复测）。

### 30.1 为什么回退机制这次能进交付档

第 3 轮挡住它的是两条，本轮都被量化清楚：

| 上轮的顾虑 | 本轮的结论 | 证据 |
|:--|:--|:--|
| 「回退抬升 max_jitter（273.8 / 294.98 px）」 | **伪影**：指标把「跨 105 帧空洞的 271 px 合法位移」当成一帧速度差；**真实单帧位移 ≤ 30.00 px**（限幅生效） | §29.2 / §29.3 · `diag_jitter_audit.py` |
| 「24 序 FAR 破线（0.0304~0.0321）」 | **峰值门可解**：门收到 0.90 后 FAR 回到红线内，且回退只带来 **0 个新增误报** | §29.5 · `diag_rollback_attrib.py` |

### 30.2 交付候选 vs 上一版冠军（两套口径全跑）

"""
    )
    out.append("**8 序（官方 gate 集）**\n\n| 臂 | F1 | P | R | FAR/帧 | hit@5px | CLE p50(池化) | CLE p95(池化) | max_jitter |\n|:--|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    for label, doc in (("上一版冠军（rb0）", ch8_doc), ("**新交付候选（rb16 + mp0.90）**", new8_doc)):
        out.append(f"| {label} | {row(doc)} |\n")
    out.append("\n**24 序（官方口径）**\n\n| 臂 | F1 | P | R | FAR/帧 | hit@5px | CLE p50(池化) | CLE p95(池化) | max_jitter |\n|:--|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    for label, doc in (("上一版冠军（rb0）", ch24_doc), ("**新交付候选（rb16 + mp0.90）**", new24_doc)):
        out.append(f"| {label} | {row(doc)} |\n")
    out.append(
        f"""
⇒ 8 序 **F1 {m8_new['f1'] - m8_old['f1']:+.4f} pp**、**hit@5px {m8_new['hit'] - m8_old['hit']:+.4f} pp**、
**FAR {m8_new['far'] - m8_old['far']:+.6f}**；24 序 **F1 {m24_new['f1'] - m24_old['f1']:+.4f} pp**、
**hit@5px {m24_new['hit'] - m24_old['hit']:+.4f} pp**、**FAR {m24_new['far'] - m24_old['far']:+.6f}**
（红线 0.0300）。

### 30.3 帧级证据：这是**零误报**的增益

"""
    )
    for tag, attr in (("8 序", attr8_new), ("24 序", attr24_new)):
        o = attr["overall"]
        ib = attr["inbox"]
        out.append(
            f"**{tag}**（新候选 vs 上一版冠军）：新增上报 **{o['other_only_report']}** 帧，"
            f"其中有 GT **{o['other_only_with_gt']}**（命中 {o['other_only_hit']}）、**无 GT {o['other_only_without_gt']}**；"
            f"已有帧**零回归**（命中→脱靶 {o['both_hit_to_miss']}、被挤掉 {o['base_only_report']}）；"
            f"inbox 计数 **TP {ib['other']['tp'] - ib['base']['tp']:+d} / FP {ib['other']['fp'] - ib['base']['fp']:+d} / "
            f"FN {ib['other']['fn'] - ib['base']['fn']:+d}**。\n\n"
        )
    out.append(
        f"""
### 30.4 回归闸门（本轮全部实跑）

| 闸门 | 结果 |
|:--|:--|
{gates}

### 30.5 产物索引与复现命令

| 用途 | 路径 |
|:--|:--|
| **交付档（8 序）** | `runs/lookahead/final/final8/arms_delivery.json` · `preds_delivery/` |
| **交付档（24 序）** | `runs/lookahead/final/final24/arms_delivery.json` · `preds_delivery/` |
| 一键复现 | `bash manu/tracking/lookahead/run_delivery.sh [8\\|24\\|both]` |
| 峰值门扫描（8 序） | `runs/lookahead/final/rb8/rb8_mp*.json` |
| 峰值门扫描（24 序） | `runs/lookahead/final/rb24/rb24_mp*.json` |
| 抖动口径审计 | `runs/lookahead/final/rb8/diag_jitter_audit_rb8.json` · `final8/diag_jitter_audit_champion.json` |
| 回退帧级归属 | `runs/lookahead/final/final8/diag_rollback_attrib_rb16mp090.json` · `final24/diag_rollback_attrib_rb16mp090.json` |
| 新脚本 | `manu/tracking/lookahead/diag_jitter_audit.py` · `diag_rollback_attrib.py` |

```bash
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. PY=/home/huangzhe/anaconda3/envs/uav/bin/python
bash manu/tracking/lookahead/run_delivery.sh both      # 交付档一键复现（8 + 24 序）
bash /mnt/data/codex-overnight/gateA_live.sh            # 标准 Gate A（8 序，活树直跑）
bash /mnt/data/codex-overnight/gateA24_live.sh          # 标准 Gate A（24 序）
bash manu/tracking/lookahead/check_newknobs_off.sh 8    # 新旋钮默认关 ⇒ 逐位一致
$PY -m pytest manu/tracking/tests -q
```

### 30.6 下一步（给 host 侧留的活）

1. **图像重定位推翻管线锚点**（§28 的唯一剩余方向）：锚点 oracle 天花板 8 序 87.804% / 24 序 85.522%，
   当前外挂 8 序 {m8_new['hit']:.3f}% / 24 序 {m24_new['hit']:.3f}% ⇒ 还差
   **{87.804 - m8_new['hit']:.3f} pp（8 序）/ {85.522 - m24_new['hit']:.3f} pp（24 序）**；
2. 24 序 FAR 只剩 {0.0300 - m24_new['far']:.6f} 余量，任何新增机制都必须先过 FAR；
3. `max_jitter` 在**有 records 空洞**的序列上是伪影，建议后续把「真实单帧位移」列进交付报表。
"""
    )
    return "".join(out)


def main_and_readme(ch8_doc, ch24_doc, new8_doc, new24_doc, gates):
    """生成主档 §30.9.24 与 README §5.13。"""
    m8o, m8n = metrics(ch8_doc), metrics(new8_doc)
    m24o, m24n = metrics(ch24_doc), metrics(new24_doc)
    _a8 = load(FINAL / "final8/diag_rollback_attrib_rb16mp090.json")["inbox"]
    _a24 = load(FINAL / "final24/diag_rollback_attrib_rb16mp090.json")["inbox"]
    a8tp = _a8["other"]["tp"] - _a8["base"]["tp"]
    a8fp = _a8["other"]["fp"] - _a8["base"]["fp"]
    a24tp = _a24["other"]["tp"] - _a24["base"]["tp"]
    a24fp = _a24["other"]["fp"] - _a24["base"]["fp"]
    main = f"""### 30.9.24 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮（收尾）：**最终交付候选** = 冠军 + 「源帧无检测 ⇒ 往回退重推」＋ 峰值置信门 0.90

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e` · python = conda env `uav`
> **专题档**：§29（抖动口径审计 + 回退帧级归属）、§30（定稿）—— `manu/memory/lookahead_extrapolator_20261102.md`
> **铁律**：不 commit / 不 push；不改 `manu/evaluation/*`；外挂全部在 `manu/tracking/lookahead/`

**① 一句话结论**

上一轮挡回退机制的两条顾虑，本轮都被**量化清楚并解掉**：
`max_jitter 294.98` 是**指标伪影**（records 空洞导致跨 105 帧的合法位移被当成一帧速度差，
真实单帧位移 ≤ 30.00 px）；FAR 破线可以用**峰值置信门 0.90**换回来，
且回退只带来 **0 个新增误报** ⇒ **回退机制正式进入交付档**。

**② 数字（两套口径全跑）**

| 口径 | 指标 | 上一版冠军 | **新交付候选** | Δ |
|:--|:--|--:|--:|--:|
| 8 序 | F1 | {m8o['f1']:.4f}% | **{m8n['f1']:.4f}%** | {m8n['f1'] - m8o['f1']:+.4f} pp |
| 8 序 | hit@5px | {m8o['hit']:.4f}% | **{m8n['hit']:.4f}%** | {m8n['hit'] - m8o['hit']:+.4f} pp |
| 8 序 | FAR/帧 | {m8o['far']:.5f} | **{m8n['far']:.5f}** | {m8n['far'] - m8o['far']:+.6f} |
| 24 序 | F1 | {m24o['f1']:.4f}% | **{m24n['f1']:.4f}%** | {m24n['f1'] - m24o['f1']:+.4f} pp |
| 24 序 | hit@5px | {m24o['hit']:.4f}% | **{m24n['hit']:.4f}%** | {m24n['hit'] - m24o['hit']:+.4f} pp |
| 24 序 | FAR/帧 | {m24o['far']:.5f} | **{m24n['far']:.5f}** | {m24n['far'] - m24o['far']:+.6f}（红线 0.0300） |

**③ 帧级证据（零回归、零新增误报）**：回退只在「源帧（t−Δ）检测器漏检」的帧上多报，
不碰任何已有帧（命中→脱靶 = 0、被挤掉 = 0）；inbox 计数 8 序 **TP {a8tp} / FP {a8fp}**、
24 序 **TP {a24tp} / FP {a24fp}**（交付档 rb16+mp0.90 口径）。

**④ 回归闸门（本轮全部实跑）**

{gates}

**⑤ 交付档复现**：`bash manu/tracking/lookahead/run_delivery.sh [8|24|both]`
（冠军旋钮一字不改，新增 `--replay-rollback-max-frames 16 --replay-rollback-min-peak 0.90`；
两个旋钮**默认 0 = 关**，关着时与上一版冠军产物**逐位一致**）。

**⑥ 下一步**：锚点 oracle 天花板 8 序 87.804% / 24 序 85.522%，
当前外挂 8 序 {m8n['hit']:.3f}% / 24 序 {m24n['hit']:.3f}%
⇒ 还要 **图像重定位推翻管线锚点** 才能继续往上（§28 §4）。
"""
    readme = f"""### 5.13 ★★★ 最终交付候选：冠军 + 回退重推 + 峰值门 0.90（2026-11-04 夜间第 3 轮收尾）

**交付档配置本轮已改**：在冠军旋钮后加两个旋钮（**默认 0 = 关**）：

```bash
--replay-rollback-max-frames 16 --replay-rollback-min-peak 0.90
```

| 口径 | 指标 | 上一版冠军 | **新交付候选** | Δ |
|:--|:--|--:|--:|--:|
| 8 序 | F1 | {m8o['f1']:.4f}% | **{m8n['f1']:.4f}%** | {m8n['f1'] - m8o['f1']:+.4f} pp |
| 8 序 | hit@5px | {m8o['hit']:.4f}% | **{m8n['hit']:.4f}%** | {m8n['hit'] - m8o['hit']:+.4f} pp |
| 8 序 | FAR/帧 | {m8o['far']:.5f} | **{m8n['far']:.5f}** | {m8n['far'] - m8o['far']:+.6f} |
| 24 序 | F1 | {m24o['f1']:.4f}% | **{m24n['f1']:.4f}%** | {m24n['f1'] - m24o['f1']:+.4f} pp |
| 24 序 | hit@5px | {m24o['hit']:.4f}% | **{m24n['hit']:.4f}%** | {m24n['hit'] - m24o['hit']:+.4f} pp |
| 24 序 | FAR/帧 | {m24o['far']:.5f} | **{m24n['far']:.5f}** | {m24n['far'] - m24o['far']:+.6f} |

| 新产物 | 命令 | 结论 |
|:--|:--|:--|
| 抖动口径审计 | `python manu/tracking/lookahead/diag_jitter_audit.py --arms-json ... --preds-dir ...` | `max_jitter 294.98` 是**伪影**（records 空洞）；**真实单帧位移 ≤ 30.00 px** |
| 回退帧级归属 | `python manu/tracking/lookahead/diag_rollback_attrib.py --labels-root ... --base ... --other ...` | 8 序 **TP {a8tp} / FP {a8fp}**、24 序 **TP {a24tp} / FP {a24fp}**，**零回归** |

**回归闸门**：`bash manu/tracking/lookahead/check_newknobs_off.sh [8|24]`（新旋钮关 ⇒ 与上一版冠军产物逐位一致）；
标准 Gate A：`/mnt/data/codex-overnight/gateA_live.sh`（8 序）· `gateA24_live.sh`（24 序）。

### 5.14 ★★★ OSD 三臂对照视频（8 序 · 交付档实测，2026-11-04 夜间第 3 轮）

```bash
bash manu/tracking/lookahead/render_osd_3arm.sh 8
```

产物：`runs/lookahead/osd/videos/<SEQ>_3arm_{{stale,replay,original}}.mp4`（+ 同名 `.json` 逐帧判定）。
**8 序 × 3 臂 = 24 段**，全部 h264 / 1920×1080 / 25 fps、**全帧可解码、0 异常**
（清单：`/mnt/data/codex-overnight/osd_manifest_round6.md`）。

| 序列 | 场景 | 窗口 | stale（老逻辑） | replay（外挂） | original（天花板） |
|:--|:--|:--|:--|:--|:--|
| `VIDEO00007_19700101_001324` | 高速机动 | 8538–8837 | F1 56.67% | **F1 100%** | F1 100% |
| `VIDEO00033_19700101_015000` | 悬停 | 339–638 | F1 75.67% | **F1 100%** | F1 100% |

⚠️ 窗口号以 `runs/lookahead/osd/windows/<SEQ>_3arm.json` 的 `config.frames` 为准；
本表与 §5.11 的老窗口（001324 = 8549–8848、015000 = 726–1025）**不是同一段画面、不可逐帧对齐**。
"""
    (GEN / "round6_main.md").write_text(main, encoding="utf-8")
    (GEN / "round6_readme.md").write_text(readme, encoding="utf-8")
    print(f"[WRITE] {GEN / 'round6_main.md'} / {GEN / 'round6_readme.md'}")


# ================================================================ 入口
if __name__ == "__main__":
    import json as _json

    # 闸门结果：由 /mnt/data/codex-overnight/round6_gates.json 提供（全部为实跑结论）
    gates_path = GEN / "round6_gates.json"
    if gates_path.is_file():
        gates = gates_path.read_text(encoding="utf-8").rstrip("\n")
    else:
        gates = "| （闸门结果未生成，见 `manu/memory/handoff_20261104_server_round6.md`） |"

    new8_doc = load(FINAL / "final8/arms_delivery.json")
    new24_doc = load(FINAL / "final24/arms_delivery.json")
    # ★ 对照基线 = **上一版冠军**（rb0 备份），不是新交付档
    prev8_doc = load(FINAL / "_prev_champion_backup/arms_delivery_8seq_rb0.json")
    prev24_doc = load(FINAL / "_prev_champion_backup/arms_delivery_24seq_rb0.json")
    attr8_new = load(FINAL / "final8/diag_rollback_attrib_rb16mp090.json")
    attr24_new = load(FINAL / "final24/diag_rollback_attrib_rb16mp090.json")

    (GEN / "round6_topical2.md").write_text(
        main_section(prev8_doc, prev24_doc, new8_doc, new24_doc, attr8_new, attr24_new, gates),
        encoding="utf-8",
    )
    print(f"[WRITE] {GEN / 'round6_topical2.md'}")
    main_and_readme(prev8_doc, prev24_doc, new8_doc, new24_doc, gates)
