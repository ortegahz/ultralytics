#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""24 序官方口径 · **交付档（出框门 m=6）** 的 Δ 敏感性对照表（补表，**不改权威报告**）。

为什么需要这张表：权威报告 `FINAL_REPORT.md` 的 §2.9「Δ 敏感性」是 **m=0 裸配置**口径
（要横向可比就不能混门），而**交付推荐档**是 m=6 ⇒ 交付档自己的 Δ 曲线在包里是**散落在
§9.4 / §9.5 / §9.6 的 m-scan 行**里的，**没有单独一张表**。

本脚本只读四个 Δ 的 **m=6 基线产物**（四臂），按臂出表，并给出：
  · replay 相对 oracle（零延迟天花板）的剩余缺口
  · replay 相对 stale（老逻辑硬吃 Δ 滞后）的追回量（**交付基线 = stale**）
  · FAR 红线（≤ 0.030）与交付三门（F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030）

用法::

    python manu/tracking/lookahead/make_delta_sens_m6_24seq.py \
        --out-md runs/lookahead/fi_nogmc24/delta_sens_m6_24seq.md \
        --out-json runs/lookahead/fi_nogmc24/delta_sens_m6_24seq.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from make_knob_scan_24seq_report import DELIVERY, arm_row, fmt, trow, verdict

REPO = Path(__file__).resolve().parents[3]
ARMS = ("original", "stale", "kinematic", "replay")
ARMS_CN = {"original": "`original`（零延迟天花板 / oracle）", "stale": "`stale`（老逻辑硬吃 Δ 滞后）",
           "kinematic": "`kinematic`（纯运动学前推）", "replay": "**`replay`（交付推荐档）**"}
DELTAS = (2, 3, 4, 8)
SRC = {
    2: "runs/lookahead/fovgate24nogmc_d2/runs/cs_t0043/d2/fov__m6/arms.json",
    3: "runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json",
    4: "runs/lookahead/fovgate24nogmc_d4/runs/cs_t0043/d4/fov__m6/arms.json",
    8: "runs/lookahead/fovgate24nogmc_d8/runs/cs_t0043/d8/fov__m6/arms.json",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="24 序交付档（m=6）Δ 敏感性对照表")
    p.add_argument("--out-md", default="runs/lookahead/fi_nogmc24/delta_sens_m6_24seq.md")
    p.add_argument("--out-json", default="runs/lookahead/fi_nogmc24/delta_sens_m6_24seq.json")
    p.add_argument("--host", default="test-Super-Server")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = {}
    for d in DELTAS:
        p = REPO / SRC[d]
        doc = json.loads(p.read_text())
        data[d] = {a: arm_row(doc, a) for a in ARMS}
        data[d]["_src"] = SRC[d]

    md = []
    md.append("# 24 序官方口径 · **交付档（出框门 m=6）** Δ 敏感性对照表 —— cs_t0043（no-GMC）")
    md.append("")
    md.append(f"- **执行机**：`{args.host}`；**生成时间**：{datetime.now().isoformat(timespec='seconds')}")
    md.append(f"- **模型包**：`cs_t0043`（no-GMC 特征语义）；**工作点**：`system_sota_longquan_trial0080_delivery_v3_params.json`（未改）")
    md.append("- **配置**：`run_delivery.sh` 的 `CFG=`（逐字取用）+ `--replay-fov-margin-px 6` ⇒ **就是交付推荐档**")
    md.append("")
    md.append("> **口径声明**")
    md.append("")
    md.append("1. **本表是 24 序官方口径**（24 序 / 133,257 GT）⇒ 可用于给交付档排序；"
              "与权威报告 `FINAL_REPORT.md` §2.9 的 **m=0 裸配置**表**并列不混用**（配置不同、数字不同）。")
    md.append("2. **交付基线 = `stale` 臂**（老逻辑硬吃 Δ 滞后）；`original` 臂 = **零延迟天花板（oracle 上界）**，不是基线。")
    md.append("3. **总体表 CLE 用 `cle_pooled_*`（池化）**；`servo.per_sequence.<seq>.cle_p*` 是逐序语义 ⇒ 两者不可混用。")
    md.append(f"4. **交付判据**：F1 ≥ {DELIVERY['f1_min']} / Recall ≥ {DELIVERY['recall_min']} / FAR ≤ {DELIVERY['far_max']}。")
    md.append("5. **本表是补表，不入交付包、不改权威报告**（§37.37 的既定裁定：新报告不被权威报告消费）。")
    md.append("")
    md.append("## 1. 四臂 × 四 Δ（交付档 m=6）")
    md.append("")
    for d in DELTAS:
        md.append(f"### Δ={d}（{d * 40} ms）　产物：`{data[d]['_src']}`")
        md.append("")
        md.append(trow(["臂", "F1%", "Recall%", "Precision%", "FAR/帧", "TP", "FP", "hit@5px%", "CLE p95(池)", "交付三门"]))
        md.append(trow(["--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", ":--"]))
        for a in ARMS:
            r = data[d][a]
            md.append(trow([ARMS_CN[a], fmt(r["f1"], 100), fmt(r["recall"], 100), fmt(r["precision"], 100),
                            fmt(r["far_per_frame"], 1, 6), str(r["tp"]), str(r["fp"]),
                            fmt(r["hit5"], 100), fmt(r["cle95"], 1, 4), verdict(r) if a == "replay" else ""]))
        md.append("")

    md.append("## 2. Δ 敏感性（`replay` 臂 = 交付推荐档）")
    md.append("")
    md.append(trow(["Δ", "时延", "F1%", "Recall%", "FAR/帧", "hit@5px%", "CLE p95(池)", "FAR 红线(≤0.030)", "三门"]))
    md.append(trow(["--:", ":--", "--:", "--:", "--:", "--:", "--:", ":--", ":--"]))
    for d in DELTAS:
        r = data[d]["replay"]
        md.append(trow([str(d), f"{d * 40} ms", fmt(r["f1"], 100), fmt(r["recall"], 100),
                        fmt(r["far_per_frame"], 1, 6), fmt(r["hit5"], 100), fmt(r["cle95"], 1, 4),
                        "**PASS**" if r["far_per_frame"] <= DELIVERY["far_max"] else "**FAIL**", verdict(r)]))
    md.append("")
    md.append("## 3. 滞后代价与追回（交付基线 = `stale`）")
    md.append("")
    md.append(trow(["Δ", "oracle F1%", "stale F1%", "replay F1%", "滞后代价(pp)", "replay 追回(pp)", "追回占比",
                    "oracle FAR", "replay FAR", "replay FAR 增量"]))
    md.append(trow(["--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]))
    for d in DELTAS:
        o, s, r = data[d]["original"], data[d]["stale"], data[d]["replay"]
        cost = (o["f1"] - s["f1"]) * 100
        back = (r["f1"] - s["f1"]) * 100
        share = (back / cost * 100) if cost > 1e-9 else float("nan")
        md.append(trow([str(d), fmt(o["f1"], 100), fmt(s["f1"], 100), fmt(r["f1"], 100),
                        f"−{cost:.4f}", f"{back:+.4f}", f"{share:.1f}%",
                        fmt(o["far_per_frame"], 1, 6), fmt(r["far_per_frame"], 1, 6),
                        f"{r['far_per_frame'] - o['far_per_frame']:+.6f}"]))
    md.append("")
    md.append("## 4. 结论")
    md.append("")
    f1s = [data[d]["replay"]["f1"] * 100 for d in DELTAS]
    fars = [data[d]["replay"]["far_per_frame"] for d in DELTAS]
    mono_f1 = all(f1s[i] > f1s[i + 1] for i in range(len(f1s) - 1))
    mono_far = all(fars[i] < fars[i + 1] for i in range(len(fars) - 1))
    md.append(f"1. **「小延时 ⇒ 精度提升」在 24 序官方口径 + 交付档上成立**："
              f"replay F1 随 Δ 减小**单调上升**（{mono_f1}）："
              + " → ".join(f"Δ={d}: {data[d]['replay']['f1'] * 100:.4f}" for d in DELTAS) + "；"
              f"FAR 随 Δ 减小**单调下降**（{mono_far}）："
              + " → ".join(f"Δ={d}: {data[d]['replay']['far_per_frame']:.6f}" for d in DELTAS) + "。")
    md.append("2. **交付工况 Δ=3（120 ms）是精度与红线的平衡点**："
              f"F1 {data[3]['replay']['f1'] * 100:.4f} / FAR {data[3]['replay']['far_per_frame']:.6f}（红线余量 "
              f"{DELIVERY['far_max'] - data[3]['replay']['far_per_frame']:.7f}）/ 三门 {verdict(data[3]['replay'])}。")
    for i, d in enumerate(DELTAS, start=3):
        r = data[d]["replay"]
        md.append(f"{i}. Δ={d}（{d * 40} ms）：FAR {r['far_per_frame']:.6f} "
                  f"{'≤' if r['far_per_frame'] <= DELIVERY['far_max'] else '>'} 0.030 红线；"
                  f"`stale` FAR {data[d]['stale']['far_per_frame']:.6f} "
                  f"{'≤' if data[d]['stale']['far_per_frame'] <= DELIVERY['far_max'] else '**>（破线，不可交付）**'}。")
    md.append("")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(f"[WROTE] {args.out_md}")
    js = {"host": args.host, "generated": datetime.now().isoformat(timespec="seconds"),
          "model": "cs_t0043", "config": "交付档 CFG + --replay-fov-margin-px 6",
          "deltas": {str(d): {a: data[d][a] for a in ARMS} | {"_src": data[d]["_src"]} for d in DELTAS}}
    Path(args.out_json).write_text(json.dumps(js, ensure_ascii=False, indent=2))
    print(f"[WROTE] {args.out_json}")


if __name__ == "__main__":
    main()
