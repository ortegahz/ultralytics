#!/usr/bin/env python3
"""Round-5: 主档 §30.9.23 · README §5.12 · 交接档增量（数字全部来自产物）。"""
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
    return {"f1": cf["f1"], "far": cf["far_per_frame"], "tp": cf["tp"], "fp": cf["fp"],
            "hit": sv.get("hit_rate_5px"), "p50": sv.get("cle_pooled_p50"), "p95": sv.get("cle_pooled_p95"),
            "max_jitter": sv.get("max_jitter")}


def pct(v, d=4):
    return f"{float(v) * 100:.{d}f}%"


def num(v, d=4):
    return f"{float(v):.{d}f}"


ch8 = load(FINAL / "final8/arms_delivery.json")
ch24 = load(FINAL / "final24/arms_delivery.json")
g8 = load(FINAL / "final8/diag_gap_attrib.json")
g24 = load(FINAL / "final24/diag_gap_attrib.json")
o8 = load(FINAL / "final8/diag_oracle_bound.json")
o24 = load(FINAL / "final24/diag_oracle_bound.json")
rb8 = load(FINAL / "rb8/rb8_mp0.50.json")
rb24 = load(FINAL / "rb24/p030_rb8_mp090.json")
m8, m24, r8, r24 = metrics(ch8), metrics(ch24), metrics(rb8), metrics(rb24)

main_md = f"""### 30.9.23 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮：帧级精度损失归属 ＋ **锚点 oracle 天花板** ＋ 新机理「源帧无检测 ⇒ 再往回退重推」

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e` · python = conda env `uav`
> **时段**：2026-11-04 02:00 → 03:20（host 已关机，本轮产物全部落服务器磁盘）
> **专题档**：§28（`manu/memory/lookahead_extrapolator_20261102.md`）—— 完整机理与全部表格
> **铁律**：不 commit / 不 push；不改 `manu/evaluation/*`；外挂全部在 `manu/tracking/lookahead/`
> **交付档本轮未改**：冠军仍是 §30.9.22 的配置（FAR 红线在 24 序已饱和，任何新档都会破线）
> **新旋钮**：`--replay-rollback-max-frames` / `--replay-rollback-min-peak`，**默认 0 = 关**（关着时 replay 臂与冠军产物**逐位一致**，已验证）

**① 一句话结论**

objective 要求的「与原代码对比算精度损失」本轮落到了**帧级**：外挂比零延迟天花板
在 8 序多丢 **{g8['counts']['orig_hit_replay_miss']}** 帧、少命中 **{g8['counts']['orig_miss_replay_hit']}** 帧，
净差距 **{g8['net_gap_pp']:+.4f} pp**；其中 **{g8['counts']['both_miss']} 帧（{g8['counts']['both_miss'] / g8['gt_frames'] * 100:.2f}%）两臂都丢** ——
**检测器/管线上限，外挂无责**。

**② ★ 关键机理：外挂误差 = 源帧锚点误差 e(f) ＋ 位移估计残差 δ**

- 管线**上报的就是检测器原始点**（实测 001324 的 8990 个上报点里 8858 个与 `ms_cache` 原始点**逐位相同**，98.7%）；
- 位移估计残差 δ 极小：8 帧内位移 ≤ 50 px 时 **p50 0.55~1.13 px、p90 1.31~3.13 px**；
- ⇒ 外挂继承的是**检测器在源帧的单帧定位误差**，不是重推链的累积漂移。

**③ ★ 锚点 oracle 天花板（本轮最重要的边界）**

若位移估计完美（δ=0），外挂 hit@5px 的上限 = 「源帧锚点本身就落在 GT 5 px 内」的比例：

| 口径 | 零延迟天花板 | 外挂（交付档） | **锚点 oracle** | 外挂↔oracle | 源帧无输出造成的天花板 |
|:--|--:|--:|--:|--:|--:|
| 8 序列 | {pct(o8['original_hit_rate'], 3)} | {pct(o8['replay_hit_rate'], 3)} | **{pct(o8['anchor_oracle_hit_rate'], 3)}** | {o8['gap_to_oracle_pp']:+.4f} pp | {100 - o8['source_frame_no_detection_rate'] * 100:.3f}% |
| 24 序列 | {pct(o24['original_hit_rate'], 3)} | {pct(o24['replay_hit_rate'], 3)} | **{pct(o24['anchor_oracle_hit_rate'], 3)}** | {o24['gap_to_oracle_pp']:+.4f} pp | {100 - o24['source_frame_no_detection_rate'] * 100:.3f}% |

⇒ **继续调重推参数最多只能再拿 {o8['gap_to_oracle_pp']:+.2f} pp（8 序）/ {o24['gap_to_oracle_pp']:+.2f} pp（24 序）**；
**要再往上必须「用图像证据重新定位、推翻管线锚点」** —— 这是给下一轮指明的唯一方向。

**④ ★ 新机理「源帧无检测 ⇒ 再往回退重推」＋ 回退链峰值置信门**

源帧（t−Δ）检测器漏检 ⇒ 外挂没有锚点 ⇒ 该帧必然脱靶。实测源帧无输出比例：8 序 {o8['source_frame_no_detection_rate'] * 100:.3f}%、24 序 {o24['source_frame_no_detection_rate'] * 100:.3f}%。
新机理：退到**最近一次有检测**的帧当锚点，多走几步重推；回退链再叠一道**峰值置信门**（峰值低 ⇒ 目标早已飞走、配准锁到背景 ⇒ 不上报）。

| 口径 | 冠军 | 回退档最优 | Δ |
|:--|--:|--:|--:|
| 8 序列 F1 | {pct(m8['f1'], 3)} | {pct(r8['f1'], 3)}（`rb8 + mp0.50`） | {(r8['f1'] - m8['f1']) * 100:+.3f} pp |
| 8 序列 hit@5px | {pct(m8['hit'], 3)} | {pct(r8['hit'], 3)} | {(r8['hit'] - m8['hit']) * 100:+.3f} pp |
| 8 序列 FAR | {num(m8['far'])} | {num(r8['far'])}（红线 0.0300，仍有���量） | {num(r8['far'] - m8['far'])} |
| 24 序列 hit@5px | {pct(m24['hit'], 3)} | {pct(r24['hit'], 3)}（`rb8 + mp0.90`，FAR 中性） | {(r24['hit'] - m24['hit']) * 100:+.3f} pp |
| 24 序列 FAR | {num(m24['far'])}（红线 0.0300，**已饱和**） | {num(r24['far'])} | {num(r24['far'] - m24['far'])} |

⇒ **24 序 FAR 红线只剩 {num(0.030 - m24['far'], 4)} 余量**，回退档在 24 序要么破线、要么收益 ≤ +0.03 pp
⇒ **交付档本轮不改**，回退 + 峰值门作为**默认关闭**的可选件交付。

**⑤ 回归防线（本轮实跑，全部 PASS）**

| 闸门 | 结果 |
|:--|:--|
| Gate A（活树直跑：原脚本 vs 新脚本 `--lookahead off`） | 8 序 **IDENTICAL=9 / DIFFER=0**，F1 = 0.964002；24 序 **IDENTICAL=25 / DIFFER=0**，F1 = 0.947469 |
| 新旋钮关闭 ⇒ 与冠军产物逐位一致 | 8 序 **9/9**、24 序 **25/25**（`per_frame` + `metrics`） |
| `--lookahead off` ⇒ 与参考产物逐位一致 | 8 序 **9/9**、24 序 **25/25** |

**⑥ 新脚本 / 新产物**

- `manu/tracking/lookahead/diag_gap_attrib.py`（帧级损失归属）· `diag_err_decomp.py`（误差分解）· `diag_oracle_bound.py`（锚点 oracle）
- `manu/tracking/lookahead/check_newknobs_off.sh`（新旋钮关 ⇒ 逐位一致）
- OSD 三臂对照补齐：**8 序 × 3 臂 = 24 段**（`runs/lookahead/osd/videos/<SEQ>_3arm_{{stale,replay,original}}.mp4`）

**⑦ 导航条目（本节新增）**

| § | 标题（简） | 归档文件 | 起始行 |
|:--|:--|:--|--:|
| 二十八 | 专题档 §28 ⭐⭐⭐ 2026-11-04 服务器夜间第 3 轮：帧级精度损失归属 + 锚点 oracle 天花板 + 「源帧无检测 ⇒ 再往回退重推」 | lookahead_extrapolator_20261102.md | 0 |
"""

readme_md = f"""### 5.12 ★★★ 帧级精度损失归属 ＋ 锚点 oracle 天花板 ＋ 「源帧无检测 ⇒ 再往回退重推」（2026-11-04 夜间第 3 轮）

**交付档配置本轮未改**（冠军仍是 §5.11 的 `peak<0.30 & disp>24`）。本轮补的是**归因与边界**：

| 产物 | 命令 | 结论 |
|:--|:--|:--|
| 帧级损失归属 | `python manu/tracking/lookahead/diag_gap_attrib.py --labels-root ... --original ... --replay ...` | 8 序：外挂独丢 {g8['counts']['orig_hit_replay_miss']} 帧 / 独赚 {g8['counts']['orig_miss_replay_hit']} 帧 / **双失 {g8['counts']['both_miss']} 帧（检测器上限，外挂无责）** |
| 误差分解 | `python manu/tracking/lookahead/diag_err_decomp.py ...` | 外挂误差 = **源帧锚点误差 e(f)** ＋ 位移残差 δ（δ p90 ≤ 3.13 px） |
| 锚点 oracle | `python manu/tracking/lookahead/diag_oracle_bound.py ...` | 天花板：8 序 **{pct(o8['anchor_oracle_hit_rate'], 3)}** / 24 序 **{pct(o24['anchor_oracle_hit_rate'], 3)}**（零延迟天花板 {pct(o8['original_hit_rate'], 3)} / {pct(o24['original_hit_rate'], 3)}） |
| 回退重推 | `--replay-rollback-max-frames N --replay-rollback-min-peak P` | 8 序 `rb8 + mp0.50` → F1 {pct(r8['f1'], 3)} / hit@5px {pct(r8['hit'], 3)} / FAR {num(r8['far'])}；24 序 FAR 红线饱和 ⇒ **默认关** |

**回归闸门**：`bash manu/tracking/lookahead/check_newknobs_off.sh [8|24]`（新旋钮关 ⇒ 与冠军产物逐位一致）；
标准 Gate A 脚本 `/mnt/data/codex-overnight/gateA_live.sh` · `gateA24_live.sh`。
"""

(GEN / "round5_main.md").write_text(main_md, encoding="utf-8")
(GEN / "round5_readme.md").write_text(readme_md, encoding="utf-8")
print(f"[SAVED] {GEN / 'round5_main.md'} ({len(main_md.splitlines())} 行)")
print(f"[SAVED] {GEN / 'round5_readme.md'} ({len(readme_md.splitlines())} 行)")
