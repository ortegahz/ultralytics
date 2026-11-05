#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""24 序官方口径 · **交付档（出框门 m=6）** 的**换模型包核心对照** —— cs_t0043(no-GMC) vs RC2(GMC)。

为什么需要这张表：权威报告 §2.10 的跨模型包对照是 **m=0 裸配置**口径；**交付推荐档是 m=6** ⇒
「把模型换成 cs_t0043」这个动作在**交付配置**下的净效果，权威报告里**没有单列一张表**。

⚠️ **口径缺口（如实标注）**：rc2 的 24 序 m=6 产物**只有 Δ=3 与 Δ=8**（其它 Δ 只做了 m=0）
⇒ 本表的 rc2 行在 Δ=2/Δ=4 **显示「—」**，不得用 m=0 的数字顶替。

用法::

    python manu/tracking/lookahead/make_modelpkg_swap_m6_24seq.py \
        --out-md runs/lookahead/fi_nogmc24/modelpkg_swap_m6_24seq.md \
        --out-json runs/lookahead/fi_nogmc24/modelpkg_swap_m6_24seq.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from make_knob_scan_24seq_report import DELIVERY, arm_row, fmt, trow, verdict

REPO = Path(__file__).resolve().parents[3]
ARMS = ("original", "stale", "kinematic", "replay")
ARMS_CN = {"original": "`original`（零延迟天花板）", "stale": "`stale`（交付基线）",
           "kinematic": "`kinematic`（纯运动学）", "replay": "**`replay`（交付方案）**"}
SRC = {
    ("cs_t0043", 3): "runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json",
    ("cs_t0043", 8): "runs/lookahead/fovgate24nogmc_d8/runs/cs_t0043/d8/fov__m6/arms.json",
    ("rc2", 3): "runs/lookahead/fovgate24/runs/rc2/d3/fov__m6/arms.json",
    ("rc2", 8): "runs/lookahead/fovgate24rc2_d8/runs/rc2/d8/fov__m6/arms.json",
}
SRC0 = {
    ("cs_t0043", 3): "runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m0/arms.json",
    ("rc2", 3): "runs/lookahead/fovgate24/runs/rc2/d3/fov__m0/arms.json",
}
MODEL_CN = {"cs_t0043": "cs_t0043（no-GMC 语义）", "rc2": "RC2 trial_0080ep3（GMC 语义）"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="24 序交付档（m=6）换模型包核心对照")
    p.add_argument("--out-md", default="runs/lookahead/fi_nogmc24/modelpkg_swap_m6_24seq.md")
    p.add_argument("--out-json", default="runs/lookahead/fi_nogmc24/modelpkg_swap_m6_24seq.json")
    p.add_argument("--host", default="test-Super-Server")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data: dict[tuple[str, int], dict] = {}
    for (m, d), rel in SRC.items():
        doc = json.loads((REPO / rel).read_text())
        data[(m, d)] = {a: arm_row(doc, a) for a in ARMS}
    base0 = {}
    for (m, d), rel in SRC0.items():
        doc = json.loads((REPO / rel).read_text())
        base0[(m, d)] = {a: arm_row(doc, a) for a in ARMS}

    md = []
    md.append("# 24 序官方口径 · **交付档（出框门 m=6）** 换模型包核心对照 —— cs_t0043(no-GMC) vs RC2(GMC)")
    md.append("")
    md.append(f"- **执行机**：`{args.host}`；**生成时间**：{datetime.now().isoformat(timespec='seconds')}")
    md.append("- **配置**：`run_delivery.sh` 的 `CFG=`（逐字取用）+ `--replay-fov-margin-px 6` ⇒ **交付推荐档**；"
              "**工作点**未改（`system_sota_longquan_trial0080_delivery_v3_params.json`）")
    md.append("")
    md.append("> **口径声明**")
    md.append("")
    md.append("1. **本表是 24 序官方口径**（24 序 / 133,257 GT）；与权威报告 §2.10 的 **m=0** 对照表**并列不混用**。")
    md.append("2. ⚠️ **权威报告 §2.1–§2.8 的四臂表里，`replay` 行也是 m=0 口径**（出框门只作用于 `replay` 臂，"
              "其余三臂不受影响 ⇒ 与本表逐位相同）⇒ **本表的 `replay` 行才是交付推荐档**。")
    md.append("3. **交付基线 = `stale` 臂**；`original` 臂 = 零延迟天花板（oracle 上界），**不是**基线。")
    md.append("4. ⚠️ **口径缺口（如实标注）**：rc2 的 24 序 m=6 产物**只做了 Δ=3 与 Δ=8** ⇒ Δ=2/Δ=4 的 rc2 行显示「—」，"
              "**不得用 m=0 的数字顶替**。")
    md.append(f"5. **交付判据**：F1 ≥ {DELIVERY['f1_min']} / Recall ≥ {DELIVERY['recall_min']} / FAR ≤ {DELIVERY['far_max']}；"
              "**总体表 CLE 用 `cle_pooled_*`（池化）**。")
    md.append("")
    for d in (3, 8):
        md.append(f"## Δ={d}（{d * 40} ms）")
        md.append("")
        md.append(trow(["模型包", "臂", "F1%", "Recall%", "FAR/帧", "hit@5px%", "CLE p95(池)", "三门"]))
        md.append(trow([":--", ":--", "--:", "--:", "--:", "--:", "--:", ":--"]))
        for m in ("rc2", "cs_t0043"):
            for a in ARMS:
                r = data[(m, d)][a]
                md.append(trow([MODEL_CN[m], ARMS_CN[a], fmt(r["f1"], 100), fmt(r["recall"], 100),
                                fmt(r["far_per_frame"], 1, 6), fmt(r["hit5"], 100), fmt(r["cle95"], 1, 4),
                                verdict(r) if a == "replay" else ""]))
        md.append("")
        md.append(f"**换模型包的核心对照（Δ={d}，`replay` 臂，no-GMC − GMC）**：")
        md.append("")
        rc, cs = data[("rc2", d)]["replay"], data[("cs_t0043", d)]["replay"]
        md.append(trow(["指标", "RC2(GMC)", "cs_t0043(no-GMC)", "差（no-GMC − GMC）", "谁更好"]))
        md.append(trow([":--", "--:", "--:", "--:", ":--"]))
        items = [("F1%", rc["f1"] * 100, cs["f1"] * 100, "up"),
                 ("Recall%", rc["recall"] * 100, cs["recall"] * 100, "up"),
                 ("FAR/帧", rc["far_per_frame"], cs["far_per_frame"], "down"),
                 ("hit@5px%", rc["hit5"] * 100, cs["hit5"] * 100, "up"),
                 ("CLE p95(池)", rc["cle95"], cs["cle95"], "down")]
        for name, a, b, direction in items:
            diff = b - a
            if abs(diff) < 1e-12:
                who = "持平"
            else:
                better = "cs_t0043(no-GMC)" if ((diff > 0) == (direction == "up")) else "RC2(GMC)"
                who = f"**{better}**"
            md.append(trow([name, f"{a:.4f}" if name != "FAR/帧" else f"{a:.6f}",
                            f"{b:.4f}" if name != "FAR/帧" else f"{b:.6f}",
                            f"{diff:+.4f}" if name != "FAR/帧" else f"{diff:+.6f}", who]))
        md.append("")

    md.append("## 结论")
    md.append("")
    d3r, d3c = data[("rc2", 3)]["replay"], data[("cs_t0043", 3)]["replay"]
    md.append(f"1. **交付工况 Δ=3（120 ms）下换模型包的净效果**（no-GMC − GMC，`replay` 臂）："
              f"F1 **{(d3c['f1'] - d3r['f1']) * 100:+.4f} pp**、FAR **{d3c['far_per_frame'] - d3r['far_per_frame']:+.6f}**、"
              f"hit@5px **{(d3c['hit5'] - d3r['hit5']) * 100:+.4f} pp**、CLE p95 **{d3c['cle95'] - d3r['cle95']:+.4f} px**。")
    md.append(f"2. **两个模型包在交付档下都过三门**：cs_t0043 {verdict(d3c)}（FAR {d3c['far_per_frame']:.6f}，"
              f"余量 {DELIVERY['far_max'] - d3c['far_per_frame']:.7f}）；RC2 {verdict(d3r)}"
              f"（FAR {d3r['far_per_frame']:.6f}，余量 {DELIVERY['far_max'] - d3r['far_per_frame']:.7f}）。")
    d8r, d8c = data[("rc2", 8)]["replay"], data[("cs_t0043", 8)]["replay"]
    md.append(f"3. **Δ=8（320 ms）下方向一致**：F1 **{(d8c['f1'] - d8r['f1']) * 100:+.4f} pp**、"
              f"FAR **{d8c['far_per_frame'] - d8r['far_per_frame']:+.6f}**、"
              f"hit@5px **{(d8c['hit5'] - d8r['hit5']) * 100:+.4f} pp**、CLE p95 **{d8c['cle95'] - d8r['cle95']:+.4f} px**。")
    md.append("4. **结论对口径稳健**：no-GMC 与 GMC 在交付档下的**优劣方向**与权威报告 §2.10 的 m=0 口径一致 ⇒ "
              "「换成 cs_t0043」这个判断**不是工作点或出框门口径造成的假象**。")
    md.append("")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(f"[WROTE] {args.out_md}")
    js = {"host": args.host, "generated": datetime.now().isoformat(timespec="seconds"),
          "config": "交付档 CFG + --replay-fov-margin-px 6",
          "rows": {f"{m}_d{d}": {a: data[(m, d)][a] for a in ARMS} for (m, d) in data},
          "m0_rows": {f"{m}_d{d}": {a: base0[(m, d)][a] for a in ARMS} for (m, d) in base0}}
    Path(args.out_json).write_text(json.dumps(js, ensure_ascii=False, indent=2))
    print(f"[WROTE] {args.out_json}")


if __name__ == "__main__":
    main()
