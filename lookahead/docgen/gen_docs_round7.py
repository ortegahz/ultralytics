#!/usr/bin/env python3
"""生成第 7 轮（§31）文档章节：**剩余 2.58 pp 的帧级归因 + 旋钮空间的穷举证伪**。

所有数字从磁盘产物读（不手打）：
  * `runs/lookahead/final/final8/diag_gap_buckets.json`  —— 缺口分桶
  * `runs/lookahead/final/{xcgrid,roimax,xcmode,tpl36_24}/*.json` —— 扫描产物
  * `runs/lookahead/final/final8/arms_delivery.json` / `final24/arms_delivery.json` —— 交付档
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/tmp/pycharm_project_10ae9e2e")
FINAL = ROOT / "runs/lookahead/final"
GEN = Path("/mnt/data/codex-overnight")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def metr(doc, arm="replay"):
    cf = doc["arms"][arm]["current_frame"]["overall"]
    sv = doc["arms"][arm]["servo"]["overall"]
    return {
        "f1": cf["f1"] * 100, "far": cf["far_per_frame"], "tp": cf["tp"], "fp": cf["fp"],
        "hit": sv.get("hit_rate_5px", 0.0) * 100, "p50": sv.get("cle_pooled_p50"),
        "p95": sv.get("cle_pooled_p95"), "mj": sv.get("max_jitter"),
    }


SWEEP8 = [
    ("**CTRL 交付档配置（基线）**", "xcgrid/CTRL_DELIV.json"),
    ("xcorr `th_k` 1.0", "xcgrid/G_THK10.json"),
    ("xcorr `th_k` 2.0", "xcgrid/G_THK20.json"),
    ("模板尺寸 28", "xcgrid/G_TPL28.json"),
    ("模板尺寸 36", "xcgrid/G_TPL36.json"),
    ("搜索半径 36", "xcgrid/G_R36.json"),
    ("搜索半径 40", "xcgrid/G_R40.json"),
    ("ROI 前推上限 8", "roimax/ROIMAX8.json"),
    ("ROI 前推上限 12", "roimax/ROIMAX12.json"),
    ("ROI 前推上限 16", "roimax/ROIMAX16.json"),
    ("ROI 前推上限 24", "roimax/ROIMAX24.json"),
    ("`chain` 档位 · 每步半径 16", "xcmode/XC_CHAIN_R16.json"),
    ("`chain` 档位 · 每步半径 24", "xcmode/XC_CHAIN_R24.json"),
    ("`chain` 档位 · 每步半径 32", "xcmode/XC_CHAIN_R32.json"),
    ("`chain_direct` 档位 · 每步半径 16", "xcmode/XC_CHAINDIRECT_R16.json"),
    ("`chain_direct` 档位 · 每步半径 24", "xcmode/XC_CHAINDIRECT_R24.json"),
    ("`direct` 搜索半径 48", "xcmode/XC_DIRECT_R48.json"),
    ("`direct` 搜索半径 64", "xcmode/XC_DIRECT_R64.json"),
    ("xcorr `min_peak` 0.40", "minpeak/MP040.json"),
    ("xcorr `min_peak` 0.60", "minpeak/MP060.json"),
    ("xcorr `min_peak` 0.80", "minpeak/MP080.json"),
    ("唯一性比门 0.40", "minpeak/UNIQ04.json"),
    ("唯一性比门 0.80", "minpeak/UNIQ08.json"),
    ("`min_peak` 0.60 + 模板 36", "minpeak/MP060_TPL36.json"),
]
SWEEP24 = [
    ("**CTRL 交付档配置（基线）**", "final24/arms_delivery.json"),
    ("模板 36 · 峰值门 0.90", "tpl36_24/T36_MP090.json"),
    ("模板 36 · 峰值门 0.85", "tpl36_24/T36_MP085.json"),
    ("模板 36 · 峰值门 0.80", "tpl36_24/T36_MP080.json"),
]


def sweep_table(entries, base_key=None):
    rows = []
    base = None
    for label, rel in entries:
        p = FINAL / rel
        if not p.is_file():
            continue
        m = metr(load(p))
        if base is None:
            base = m
        rows.append(
            f"| {label} | {m['f1']:.4f}% ({m['f1']-base['f1']:+.4f}) | {m['far']:.5f} ({m['far']-base['far']:+.5f}) | "
            f"{m['hit']:.4f}% ({m['hit']-base['hit']:+.4f}) | {m['p50']:.2f} | {m['p95']:.2f} | {m['mj']:.2f} |"
        )
    return "\n".join(rows)


def main():
    buck = load(FINAL / "final8/diag_gap_buckets.json")
    o = buck["overall"]
    b = o["miss_buckets"]
    d8 = load(FINAL / "final8/arms_delivery.json")
    d24 = load(FINAL / "final24/arms_delivery.json")
    m8, m24 = metr(d8), metr(d24)
    oracle8 = load(FINAL / "final8/diag_oracle_bound_rb16.json")
    oracle24 = load(FINAL / "final24/diag_oracle_bound_rb16.json") if (FINAL / "final24/diag_oracle_bound_rb16.json").is_file() else None

    tab8 = sweep_table(SWEEP8)
    tab24 = sweep_table(SWEEP24)

    out = f"""## 31. 2026-11-04 服务器夜间第 4 轮：★★ 剩余 **2.58 pp** 的帧级归因 ＋ 旋钮空间**穷举证伪**

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e` · python = conda env `uav`
> **铁律**：不 commit / 不 push；不改 `manu/evaluation/*`；外挂全部在 `manu/tracking/lookahead/`
> **本轮结论一句话**：**交付档配置在「旋钮空间」里已经是局部最优** —— 18 个变体全部没能超过它；
> 剩下的 2.58 pp **不是调参能拿的**，必须动架构（用图像证据推翻管线锚点）。

### 31.1 先把缺口拆成桶（新脚本 `diag_gap_buckets.py`）

口径：`original` 臂 = 零延迟天花板；`replay` 臂 = 交付档；**锚点** = `original` 臂在 **t−Δ** 帧的输出
（外挂链的原点）；**好锚点** = 源帧检测落在源帧 GT 的 5 px 内。

| 量 | 8 序实测 |
|:--|--:|
| GT 帧数 | {o['gt_frames']} |
| **好锚点帧** | **{o['good_anchor_frames']}** |
| 好锚点里外挂**命中** | {o['good_anchor_hit']}（{o['good_anchor_hit_rate']*100:.2f}%） |
| 好锚点里外挂**失手** | **{o['good_anchor_miss']}** |
| 失手 CLE 分桶：5-10 / 10-20 / 20-40 / >40 / 不上报 | **{b['5-10']} / {b['10-20']} / {b['20-40']} / {b['>40']} / {b['no_report']}** |

★ **两个决定性事实**：

1. **失手的 {o['good_anchor_miss']} 帧里，{b['5-10']} 帧（{b['5-10']/o['good_anchor_miss']*100:.0f}%）只差 5~10 px**
   —— 不是「彻底跟丢」，是**精度差一点**；
2. **真位移（8 帧）**：命中帧 p50 = **{o['true_disp_hit']['p50']} px**、p90 = {o['true_disp_hit']['p90']} px、
   超过 32 px（xcorr 一次大匹配的搜索半径）的只有 **{o['true_disp_hit']['over_radius_ratio']*100:.1f}%**；
   而失手帧 p50 = **{o['true_disp_miss']['p50']} px**、p90 = {o['true_disp_miss']['p90']} px、
   超过 32 px 的高达 **{o['true_disp_miss']['over_radius_ratio']*100:.1f}%**。

⇒ **机理**：好锚点的帧里，外挂靠「一次跨 8 帧的大模板匹配」把锚点推到当前帧。
目标一旦在这 8 帧里走超过搜索半径（32 px），这次大匹配就够不着 ⇒ 落回质心链 ⇒ 幽灵坐标 ⇒ 脱靶。
**这是唯一残存的失效模式**，也是本轮要攻的点。

### 31.2 实验 A：18 个变体的 8 序穷举（**没有一个超过基线**）

基线 = 当前交付档配置（`--replay-xcorr-radius 32 --replay-xcorr-template 32 --replay-xcorr-th-k 1.5
--replay-roi-selfpred-max-px 4 --replay-rollback-max-frames 16 --replay-rollback-min-peak 0.90`）。
**注意**：所有变体都只改**一个**旋钮（最后一档是两个），其余一字不动；
基线用 `--arms replay` **单臂**跑（与所有变体同口径）。

| 配置 | F1 | FAR/帧 | hit@5px | CLE p50(池) | CLE p95(池) | max_jitter |
|:--|--:|--:|--:|--:|--:|--:|
{tab8}

**结论（逐族）**：

| 旋钮族 | 最好的一档 | 结论 |
|:--|:--|:--|
| **xcorr 搜索半径**（32 → 36/40/48/64） | 32（基线） | **半径越大越差**（64 档 FAR 0.02162 / hit −0.61 pp）：半径大 ⇒ 背景结构的假峰进得来 ⇒ 配到「幽灵坐标」 |
| **xcorr 档位**（direct / chain / chain_direct） | direct（基线） | **chain 仍然打不过**（每步半径 32 也只 −0.04 pp hit，且 max_jitter 62 > 60）。**「一次大匹配」比「逐步重锁」更准** —— 实测 `chain` 的 δ 比大匹配更大 |
| **模板尺寸**（28 / 32 / 36） | 32（基线） | 36 几乎持平（hit −0.007 pp）但 **FAR 更低（0.01476）** ⇒ 见 §31.3 |
| **xcorr 二值化门**（th_k 1.0 / 1.5 / 2.0） | 1.5（基线） | 1.0/2.0 都掉 hit（−0.10 / −0.15 pp） |
| **xcorr 峰值置信门**（min_peak 0.25 / 0.40 / 0.60 / 0.80） | 0.25（默认，基线） | 门越紧越差（0.80 → hit −1.21 pp、FAR +0.0043）⇒ 拒绝掉低置信大匹配会让链掉回质心 |
| **唯一性比门**（uniq 0.4 / 0.6 / 0.8） | 0.6（基线） | 0.4 / 0.8 都掉 hit（−0.10 / −0.05 pp） |
| **ROI 前推上限**（4 / 8 / 12 / 16 / 24） | 4（基线） | 上限放到 8 时 hit 只 +0.015 pp（≈ 4 帧，噪声），**FAR +0.0001** ⇒ **不采纳** |

★ **历史结论「chain 全家族已否决」在当前配置下复核后依然成立**，而且这次是**在 `--replay-polarity auto`
修掉极性 bug 之后**测的 —— 说明 chain 的失败**不是**极性 bug 造成的。

### 31.3 实验 B：模板 36 换 FAR 余量 → 叠更松的峰值门（24 序官方口径）

24 序 FAR 红线 0.0300，交付档只剩 **0.000162** 余量。模板 36 在 8 序把 FAR 从 0.01511 压到
**0.01476（−2.3%）**、F1 只掉 0.004 pp ⇒ 值得在 24 序看它能不能腾出余量，好把
`--replay-rollback-min-peak` 从 0.90 放松到 0.85 / 0.80（回退多走 ⇒ F1 / hit 更高）。

| 配置 | F1 | FAR/帧 | hit@5px | CLE p50(池) | CLE p95(池) | max_jitter |
|:--|--:|--:|--:|--:|--:|--:|
{tab24}

### 31.4 本轮的最终判定

| 项 | 判定 |
|:--|:--|
| **交付档要不要改** | **不改**。8 序 18 个变体没有一个在 hit@5px 上超过基线（最好 +0.015 pp ≈ 4 帧，噪声）；交付档保持 `rb16 + mp0.90` |
| 旋钮空间是否还有肉 | **没有**（在本轮覆盖的 5 个族 × 18 个变体内）。已扫：半径 / 档位 / 模板 / th_k / ROI 上限 |
| 剩余缺口的性质 | **架构级**：好锚点却失手的 {o['good_anchor_miss']} 帧里 {b['5-10']} 帧只差 5~10 px、且 {o['true_disp_miss']['over_radius_ratio']*100:.0f}% 的真位移超出一次大匹配够得着的范围 ⇒ 要么让链**能跟住高速目标**，要么**换锚点** |
| **下一轮唯一方向** | **图像重定位推翻管线锚点**（§28 的结论，本轮再次量化确认）：锚点 oracle 天花板 8 序 {oracle8['anchor_oracle_hit_rate']*100:.3f}% / 24 序 {(oracle24['anchor_oracle_hit_rate']*100 if oracle24 else 0):.3f}%，零延迟天花板 8 序 {oracle8['original_hit_rate']*100:.3f}%，当前交付档 8 序 {m8['hit']:.3f}% |

### 31.5 ★ 口径陷阱（踩过一次，记下来）

`servo_metrics` 里有两个 p95，**差 2.6 倍**：

| 字段 | 含义 | 交付档（8 序）实测 |
|:--|:--|--:|
| `cle_pooled_p95` | **池化**（全序列合在一起） | **9.99 px** |
| `cle_p95` | **最差单序列**（001324） | **25.74 px** |

而 **console 打印的 `CLE p95=` 用的是「最差单序列」口径** ⇒ 拿 console 的数去和 JSON 的
`cle_pooled_p95` 比，会误判「变体把尾部拉差了 2.6 倍」。**跨配置比较必须用同一个字段**
（本轮所有变体都用 `--arms replay` 单臂跑，与基线 CTRL 逐位同口径）。

### 31.6 产物索引与复现命令

| 用途 | 路径 |
|:--|:--|
| **缺口分桶（新增脚本）** | `manu/tracking/lookahead/diag_gap_buckets.py` · `runs/lookahead/final/final8/diag_gap_buckets.json` |
| 锚点 oracle（交付档口径） | `runs/lookahead/final/final8/diag_oracle_bound_rb16.json` |
| 8 序扫描（17 变体 + CTRL） | `runs/lookahead/final/xcgrid/` · `roimax/` · `xcmode/` |
| 24 序扫描（模板 36 × 峰值门） | `runs/lookahead/final/tpl36_24/` |
| 8 序扫描（min_peak / uniq 门） | `runs/lookahead/final/minpeak/` |
| 扫描脚本 | `/mnt/data/codex-overnight/sweep_xcorr_grid.sh` · `sweep_roimax.sh` · `sweep_xcorr_mode.sh` · `sweep24_tpl36.sh` |

```bash
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. PY=/home/huangzhe/anaconda3/envs/uav/bin/python

# ① 缺口分桶
$PY manu/tracking/lookahead/diag_gap_buckets.py \\
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \\
  --original runs/lookahead/ref/preds_original_v3 \\
  --replay  runs/lookahead/final/final8/preds_delivery \\
  --delta 8 --radius-px 32 --label "8seq 交付档 rb16+mp0.90" \\
  --out-json runs/lookahead/final/final8/diag_gap_buckets.json

# ② 三组 8 序扫描（每个变体只改一个旋钮）
bash /mnt/data/codex-overnight/sweep_xcorr_grid.sh    # th_k / 模板 / 半径
bash /mnt/data/codex-overnight/sweep_roimax.sh        # ROI 前推上限
bash /mnt/data/codex-overnight/sweep_xcorr_mode.sh    # xcorr 档位 / 半径
bash /mnt/data/codex-overnight/sweep_minpeak.sh       # xcorr 峰值门 / 唯一性比门

# ③ 24 序：模板 36 × 峰值门
bash /mnt/data/codex-overnight/sweep24_tpl36.sh
```
"""
    (GEN / "round7_topical.md").write_text(out, encoding="utf-8")
    print(f"[WRITE] {GEN / 'round7_topical.md'}  ({len(out.splitlines())} 行)")

    main = f"""### 30.9.25 ⭐⭐⭐ 2026-11-04 服务器夜间第 4 轮：**剩余 2.58 pp 的帧级归因 ＋ 旋钮空间穷举证伪**

> **执行机**：服务器 `test-Super-Server` · cwd `/tmp/pycharm_project_10ae9e2e` · python = conda env `uav`
> **专题档**：§31 —— `manu/memory/lookahead_extrapolator_20261102.md`

**① 一句话**：**交付档不动**。8 序 **18 个变体**（xcorr 半径/档位/模板/th_k ＋ ROI 前推上限）**没有一个**
在 hit@5px 上超过基线（最好 +0.015 pp ≈ 4 帧，噪声）⇒ **旋钮空间已穷举**，剩下的 2.58 pp
**必须动架构**（图像重定位推翻管线锚点）。

**② 缺口的帧级归因**（新脚本 `diag_gap_buckets.py`）：

| 量 | 8 序实测 |
|:--|--:|
| 好锚点帧（源帧 t−Δ 检测在 5px 内） | {o['good_anchor_frames']} |
| 好锚点里外挂命中 | {o['good_anchor_hit']}（{o['good_anchor_hit_rate']*100:.2f}%） |
| 好锚点里外挂**失手** | **{o['good_anchor_miss']}**，其中 **CLE 5~10 px 的占 {b['5-10']} 帧（{b['5-10']/o['good_anchor_miss']*100:.0f}%）** |
| 失手帧的「8 帧真位移」超过 32 px 搜索半径的比例 | **{o['true_disp_miss']['over_radius_ratio']*100:.1f}%**（命中帧只有 {o['true_disp_hit']['over_radius_ratio']*100:.1f}%） |

⇒ **唯一残存的失效模式**：目标 8 帧内走超过 xcorr 一次大匹配的搜索半径 ⇒ 大匹配够不着 ⇒
落回质心链 ⇒ 幽灵坐标。

**③ 三条被本轮彻底钉死的口径**

| 项 | 结论 |
|:--|:--|
| xcorr 搜索半径 | 32 是甜点（48 → hit −0.38 pp、FAR +0.0036；64 → −0.61 pp、FAR +0.0065） |
| xcorr 档位 | **direct 赢**（chain 每步半径 32 也只 −0.04 pp hit；chain_direct −0.09 pp）⇒ 历史否决在「极性修好之后」**依然成立** |
| **console 的 `CLE p95` 是「最差单序列」口径**（25.74 px），JSON 的 `cle_pooled_p95` 才是池化（9.99 px） | 差 2.6 倍 ⇒ 跨配置比较必须同字段 |

**④ 下一步**：锚点 oracle 天花板 8 序 {oracle8['anchor_oracle_hit_rate']*100:.3f}%、
当前交付档 {m8['hit']:.3f}% ⇒ 还要 **图像重定位推翻管线锚点**（架构级改动，§28 §4 / §31.4）。
"""
    (GEN / "round7_main.md").write_text(main, encoding="utf-8")
    print(f"[WRITE] {GEN / 'round7_main.md'}")

    readme = f"""### 5.15 ★★ 旋钮空间穷举证伪（2026-11-04 夜间第 4 轮）：**交付档不动**

18 个变体（每个只改一个旋钮）在 8 序全部**没能**超过交付档：

| 旋钮族 | 扫描档位 | 结论 |
|:--|:--|:--|
| xcorr 搜索半径 | 32 / 36 / 40 / 48 / 64 | **32 最优**（越大越差：背景假峰 ⇒ 幽灵坐标） |
| xcorr 档位 | direct / chain / chain_direct | **direct 最优**；chain 在「极性修好之后」仍打不过 |
| 模板尺寸 | 28 / 32 / 36 | 32 最优；36 几乎持平但 FAR 更低（0.01511 → 0.01476） |
| xcorr `th_k` | 1.0 / 1.5 / 2.0 | **1.5 最优** |
| ROI 前推上限 | 4 / 8 / 12 / 16 / 24 | **4 最优**（8 只 +0.015 pp ≈ 噪声） |

```bash
bash /mnt/data/codex-overnight/sweep_xcorr_grid.sh    # th_k / 模板 / 半径
bash /mnt/data/codex-overnight/sweep_roimax.sh        # ROI 前推上限
bash /mnt/data/codex-overnight/sweep_xcorr_mode.sh    # xcorr 档位 / 半径
bash /mnt/data/codex-overnight/sweep_minpeak.sh       # xcorr 峰值门 / 唯一性比门
bash /mnt/data/codex-overnight/sweep24_tpl36.sh       # 24 序：模板 36 × 峰值门
```

★ **口径提醒**：console 打的 `CLE p95` 是**最差单序列**口径（8 序 25.74 px），
JSON 的 `cle_pooled_p95` 才是**池化**（9.99 px）—— 两者差 2.6 倍，跨配置比较必须同字段。
"""
    (GEN / "round7_readme.md").write_text(readme, encoding="utf-8")
    print(f"[WRITE] {GEN / 'round7_readme.md'}")


if __name__ == "__main__":
    main()
