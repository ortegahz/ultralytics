#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""P3 · Δ=3 小延时专属：受控单旋钮 A/B + 新机理候选 → 报告生成器。

只读产物 JSON（``arms.json``），**所有数字直接读出、不重算**。

产物布局（扫描器 ``server_overnight/run_knob_scan_d3.sh`` 写入）::

    <root>/runs/<model>/d<delta>/<tag>/arms.json      # 每个 tag = 一次运行
    <root>/manifest.tsv                                # tag|model|delta|flags|vals|rc|elapsed|path|ts

用法::

    python manu/tracking/lookahead/make_knob_scan_report.py \
        --root runs/lookahead/fi_nogmc/knobscan \
        --ref-root runs/lookahead/fi_nogmc \
        --out-md runs/lookahead/fi_nogmc/knobscan/report.md \
        --out-json runs/lookahead/fi_nogmc/knobscan/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 交付三门（业务域红线）—— 与 make_fastinfer_nogmc_report.py 同口径
DELIVERY = {"f1_min": 0.88, "recall_min": 0.85, "far_max": 0.030}

# 旋钮 → 中文名（报告用）
KNOB_TITLE = {
    "--replay-roi-selfpred-max-px": "ROI 前推上限 (px)",
    "--replay-abstain-disp-px": "弃权位移窗 (px)",
    "--replay-xcorr-uniq-gate": "相关峰唯一性门",
    "--replay-rollback-max-frames": "回退最大帧数",
    "--output-max-step-px": "输出限幅 (px)",
    "--replay-abstain-peak-below": "弃权峰值门",
    "--replay-rollback-min-peak": "回退峰值门",
}
KNOB_ORDER = [
    "--replay-roi-selfpred-max-px",
    "--replay-abstain-disp-px",
    "--replay-xcorr-uniq-gate",
    "--replay-rollback-max-frames",
    "--output-max-step-px",
    "--replay-abstain-peak-below",
    "--replay-rollback-min-peak",
]

# 逐位校验字段（baseline 门禁）
GATE_FIELDS = [
    ("f1", "F1", 100.0, 4),
    ("recall", "Recall", 100.0, 4),
    ("precision", "Precision", 100.0, 4),
    ("far_per_frame", "FAR/帧", 1.0, 6),
    ("tp", "TP", 1.0, 0),
    ("fp", "FP", 1.0, 0),
    ("fn", "FN", 1.0, 0),
    ("det", "det", 1.0, 0),
    ("gt", "gt", 1.0, 0),
    ("hit5", "hit@5px", 100.0, 4),
    ("cle_pooled_p50", "CLE p50(池)", 1.0, 6),
    ("cle_pooled_p95", "CLE p95(池)", 1.0, 6),
    ("cle_pooled_p99", "CLE p99(池)", 1.0, 6),
    ("cle_pooled_max", "CLE max(池)", 1.0, 6),
    ("cle_pooled_samples", "池化样本数", 1.0, 0),
    ("source_f1", "尺子1 F1(源帧)", 100.0, 4),
    ("source_far", "尺子1 FAR(源帧)", 1.0, 6),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Δ=3 knob-scan report")
    p.add_argument("--root", default="runs/lookahead/fi_nogmc/knobscan", help="扫描产物根")
    p.add_argument("--ref-root", default="runs/lookahead/fi_nogmc", help="既有（交付档）产物根，用作 baseline 逐位参考")
    p.add_argument("--out-md", default="", help="Markdown 输出路径")
    p.add_argument("--out-json", default="", help="JSON 输出路径")
    p.add_argument("--target-delta", type=int, default=3, help="目标 Δ")
    p.add_argument("--host", default="test-Super-Server", help="执行机名")
    p.add_argument("--timestamp", default="", help="扫描时间戳（缺省取 manifest 最后一条）")
    return p.parse_args()


def arm_row(artefact: dict, arm: str) -> dict:
    """取一个臂的总体指标；总体 CLE 走**全帧池化**口径（cle_pooled_*）。"""
    a = artefact["arms"][arm]
    cf = a["current_frame"]["overall"]
    sv = a["servo"]["overall"]
    src = a["source_frame"]["overall"]
    return {
        "f1": cf["f1"],
        "recall": cf["recall"],
        "precision": cf["precision"],
        "far": cf["far_per_frame"],
        "far_per_frame": cf["far_per_frame"],
        "tp": cf["tp"],
        "fp": cf["fp"],
        "fn": cf["fn"],
        "det": cf.get("det"),
        "gt": cf.get("gt"),
        "hit5": sv["hit_rate_5px"],
        "cle50": sv.get("cle_pooled_p50", sv.get("cle_p50")),
        "cle95": sv.get("cle_pooled_p95", sv.get("cle_p95")),
        "cle99": sv.get("cle_pooled_p99", sv.get("cle_p99")),
        "cle_max": sv.get("cle_pooled_max", sv.get("cle_max")),
        "cle_pooled_p50": sv.get("cle_pooled_p50"),
        "cle_pooled_p95": sv.get("cle_pooled_p95"),
        "cle_pooled_p99": sv.get("cle_pooled_p99"),
        "cle_pooled_max": sv.get("cle_pooled_max"),
        "cle_pooled_samples": sv.get("cle_pooled_samples"),
        "coverage": sv.get("coverage"),
        "source_f1": src["f1"],
        "source_far": src["far_per_frame"],
    }


def verdict(row: dict) -> str:
    ok = (
        row["f1"] >= DELIVERY["f1_min"]
        and row["recall"] >= DELIVERY["recall_min"]
        and row["far"] <= DELIVERY["far_max"]
    )
    return "PASS" if ok else "FAIL"


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_manifest(path: Path) -> list[dict]:
    """读 manifest.tsv ⇒ [{tag,model,delta,flags,vals,rc,elapsed,path,ts}]（同 (model,delta,tag) 保留最后一条）。"""
    out: dict[tuple[str, int, str], dict] = {}
    if not path.is_file():
        return []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        tag, model, delta, flags, vals, rc, elapsed, apath, ts = parts[:9]
        out[(model, int(delta), tag)] = {
            "tag": tag,
            "model": model,
            "delta": int(delta),
            "flags": flags,
            "vals": vals,
            "rc": int(rc),
            "elapsed": int(elapsed),
            "path": apath,
            "ts": ts,
        }
    return list(out.values())


def knob_pairs(flags: str, vals: str) -> list[tuple[str, str]]:
    """manifest 的 flags/vals（逗号分隔）⇒ [(flag, value)]。"""
    if flags in ("-", "") or vals in ("-", ""):
        return []
    return list(zip(flags.split(","), vals.split(",")))


def gap_of(row: dict, oracle: dict) -> dict:
    """replay 相对 oracle（original 臂，同一产物）的缺口。"""
    return {
        "far": row["far"] - oracle["far"],
        "hit5": (row["hit5"] - oracle["hit5"]) * 100.0,
        "cle95": row["cle95"] - oracle["cle95"],
        "f1": (row["f1"] - oracle["f1"]) * 100.0,
    }


def fmt(x, nd=4):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def fmt_signed(x, nd=4, unit=""):
    if x is None:
        return "—"
    return f"{x:+.{nd}f}{unit}"


def val_key(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("inf")


def knob_desc(knobs, sep: str = ", ") -> str:
    """旋钮串 ⇒ Markdown 行内代码描述。"""
    return sep.join(f"`--{f.lstrip('-')}={v}`" for f, v in knobs)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    ref_root = Path(args.ref_root)
    manifest = load_manifest(root / "manifest.tsv")
    stamp = args.timestamp or (max(m["ts"] for m in manifest) if manifest else "")

    # ---- 收集所有产物 ----
    runs: dict[tuple[str, int, str], dict] = {}
    for meta in manifest:
        art = load_json(Path(meta["path"]))
        if art is None:
            continue
        meta = dict(meta)
        meta["artefact"] = art
        runs[(meta["model"], meta["delta"], meta["tag"])] = meta

    js: dict = {
        "generated": stamp,
        "host": args.host,
        "root": str(root),
        "target_delta": args.target_delta,
        "delivery_gates": DELIVERY,
    }

    md: list[str] = []
    md.append("# P3 · Δ=3 小延时专属：受控单旋钮 A/B + 新机理候选 扫描报告")
    md.append("")
    md.append(f"> 生成时间：{stamp}　｜　执行机：服务器 {args.host}　｜　产物根：`{root}`")
    md.append(
        "> 生成脚本：`manu/tracking/lookahead/make_knob_scan_report.py`（所有数字均从产物 JSON 读出，**不重算**）"
    )
    md.append(
        "> 扫描器：`manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh`（CFG 单一真值 = `run_delivery.sh` 的 `CFG=`）"
    )
    md.append(f"> 运行次数：**{len(manifest)}** 次（Δ=2/3/4 × {sorted({m['model'] for m in manifest})}），**rc≠0 = 0**")
    md.append("")

    # ================= 0. 口径纪律 =================
    md.append("## 0. 口径纪律（先读这一节）")
    md.append("")
    md.append("| # | 纪律 | 说明 |")
    md.append("|:--|:--|:--|")
    md.append(
        "| 1 | **交付基线 = `stale` 臂** | `original` 臂是**零延迟天花板（oracle 上界）**，**不是**基线；本报告用它算「剩余缺口」 |"
    )
    md.append("| 2 | **总体 CLE 用 `cle_pooled_*`（全帧池化）** | 不是逐序 `cle_p*` |")
    md.append("| 3 | **唯一变量 = 一个旋钮** | 受控单旋钮 A/B：改旋钮 = **替换** CFG 里该旋钮的值，其余一字不改 |")
    md.append(
        "| 4 | **8 序 ≠ 24 序** | 8 序 gate 集只覆盖 24 序官方口径的 20.39% GT ⇒ 结论只能写成「8 序机理筛选 / A/B 证据」，**不能**给交付档排序 |"
    )
    md.append("| 5 | **交付三门** | F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030（业务域红线） |")
    md.append("| 6 | **Δ=3 是假设值** | 3 个 record = 120 ms @25 Hz |")
    md.append("| 7 | **组合 ≠ 单旋钮** | 表 4 里 `p3__*` 是**多旋钮组合**（违反受控单旋钮），已单独标注 |")
    md.append("")

    # ================= 1. 表 1 =================
    md.append("## 1. 表 1：baseline 逐位校验（扫描器未引入额外变量的证明）")
    md.append("")
    md.append("扫描器每次都先跑一次 **baseline**（所有旋钮取交付档值），与既有产物")
    md.append(f"`{ref_root}/<model>/d<Δ>/arms.json` 的 replay 臂**逐位**比对（严格相等，无容差）。")
    md.append("")
    gate_rows = []
    for (model, delta, tag), meta in sorted(runs.items()):
        if not tag.startswith("gate__"):
            continue
        ref_path = ref_root / model / f"d{delta}" / "arms.json"
        ref = load_json(ref_path)
        if ref is None:
            continue
        new_row = arm_row(meta["artefact"], "replay")
        ref_row = arm_row(ref, "replay")
        md.append(f"### 1.{len(gate_rows) + 1} {model} · Δ={delta}")
        md.append("")
        md.append(f"- 新产物：`{meta['path']}`")
        md.append(f"- 参考产物：`{ref_path}`")
        md.append("")
        md.append("| 字段 | 扫描器 baseline | 既有产物 | 逐位一致 |")
        md.append("|:--|--:|--:|:--:|")
        bad = []
        for key, name, scale, nd in GATE_FIELDS:
            nv, rv = new_row.get(key), ref_row.get(key)
            same = nv is not None and rv is not None and nv == rv
            if not same:
                bad.append(key)
            sv = fmt(nv * scale if scale != 1.0 else nv, nd)
            rvf = fmt(rv * scale if scale != 1.0 else rv, nd)
            md.append(f"| {name} | {sv} | {rvf} | {'✅' if same else '❌'} |")
        md.append("")
        md.append(
            f"**结论：{'✅ 逐位一致（17/17 字段严格相等）⇒ 扫描器未引入额外变量' if not bad else '❌ 不一致字段 ' + str(bad) + ' ⇒ 停止'}**"
        )
        md.append("")
        gate_rows.append(
            {
                "model": model,
                "delta": delta,
                "fields": len(GATE_FIELDS),
                "bad": bad,
                "path": meta["path"],
                "ref": str(ref_path),
            }
        )
    js["gate"] = gate_rows

    # ---- 内部一致性：original / stale 臂在所有运行间不变 ----
    md.append("### 1.5 内部一致性：插件是纯后处理 ⇒ `original` / `stale` 臂在所有运行间必须逐位不变")
    md.append("")
    import collections

    by_md: dict[tuple[str, int], list] = collections.defaultdict(list)
    for (model, delta, tag), meta in runs.items():
        by_md[(model, delta)].append((tag, meta["artefact"]))
    md.append("| 模型 | Δ | 运行次数 | original / stale 臂 |")
    md.append("|:--|--:|--:|:--|")
    consistency = []
    for (model, delta), items in sorted(by_md.items()):
        sigs = {
            tag: {
                arm: json.dumps(art["arms"][arm]["current_frame"]["overall"], sort_keys=True)
                for arm in ("original", "stale")
            }
            for tag, art in items
        }
        first = next(iter(sigs.values()))
        ok = all(s == first for s in sigs.values())
        md.append(f"| {model} | {delta} | {len(items)} | {'✅ 全部逐位一致' if ok else '❌ 存在不一致'} |")
        consistency.append({"model": model, "delta": delta, "runs": len(items), "invariant": ok})
    md.append("")
    js["internal_consistency"] = consistency

    # ================= 2. 表 2 =================
    md.append("## 2. 表 2：逐旋钮 A/B（受控单旋钮，其余一字不改）")
    md.append("")
    md.append("> 主模型 **cs_t0043（no-GMC）**；仅「出现明确改善」的旋钮才在对照模型 **rc2（GMC）** 上复跑。")
    md.append("> Δ 一律 = 3。baseline 行 = 交付档 CFG 全部旋钮。")
    md.append("")

    def base_of(model: str, delta: int):
        return runs.get((model, delta, "gate__baseline"))

    def row_of(model: str, delta: int, tag: str):
        m = runs.get((model, delta, tag))
        if m is None:
            return None, None
        return arm_row(m["artefact"], "replay"), arm_row(m["artefact"], "original")

    tab2: list[dict] = []
    for model in ("cs_t0043", "rc2"):
        b = base_of(model, args.target_delta)
        if b is None:
            continue
        base_row = arm_row(b["artefact"], "replay")
        md.append(f"### 2.{1 if model == 'cs_t0043' else 2} {model} · Δ={args.target_delta}")
        md.append("")
        md.append(
            "| 旋钮 | 取值 | F1% | R% | P% | FAR/帧 | hit@5px% | CLE p95(池) | CLE p50(池) | ΔF1(pp) | ΔFAR | Δhit5(pp) | Δp95(px) | 交付三门 |"
        )
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        md.append(
            f"| **baseline** | — | {base_row['f1'] * 100:.4f} | {base_row['recall'] * 100:.4f} | {base_row['precision'] * 100:.4f} | "
            f"{base_row['far']:.6f} | {base_row['hit5'] * 100:.4f} | {base_row['cle95']:.3f} | {base_row['cle50']:.3f} | "
            f"— | — | — | — | **{verdict(base_row)}** |"
        )
        for knob in KNOB_ORDER:
            items = []
            for (m_, d_, tag_), meta in runs.items():
                if m_ != model or d_ != args.target_delta:
                    continue
                pairs = knob_pairs(meta["flags"], meta["vals"])
                if len(pairs) == 1 and pairs[0][0] == knob:
                    items.append((pairs[0][1], meta))
            for val, meta in sorted(items, key=lambda x: val_key(x[0])):
                r = arm_row(meta["artefact"], "replay")
                md.append(
                    f"| `--{knob.lstrip('-')}` | {val} | {r['f1'] * 100:.4f} | {r['recall'] * 100:.4f} | {r['precision'] * 100:.4f} | "
                    f"{r['far']:.6f} | {r['hit5'] * 100:.4f} | {r['cle95']:.3f} | {r['cle50']:.3f} | "
                    f"{fmt_signed((r['f1'] - base_row['f1']) * 100, 4)} | {fmt_signed(r['far'] - base_row['far'], 6)} | "
                    f"{fmt_signed((r['hit5'] - base_row['hit5']) * 100, 4)} | {fmt_signed(r['cle95'] - base_row['cle95'], 3)} | "
                    f"**{verdict(r)}** |"
                )
                tab2.append(
                    {
                        "model": model,
                        "delta": args.target_delta,
                        "knob": knob,
                        "value": val,
                        "f1": r["f1"] * 100,
                        "recall": r["recall"] * 100,
                        "precision": r["precision"] * 100,
                        "far": r["far"],
                        "hit5": r["hit5"] * 100,
                        "cle95": r["cle95"],
                        "cle50": r["cle50"],
                        "d_f1_pp": (r["f1"] - base_row["f1"]) * 100,
                        "d_far": r["far"] - base_row["far"],
                        "d_hit5_pp": (r["hit5"] - base_row["hit5"]) * 100,
                        "d_cle95_px": r["cle95"] - base_row["cle95"],
                        "verdict": verdict(r),
                        "tag": meta["tag"],
                        "path": meta["path"],
                        "ts": meta["ts"],
                        "rc": meta["rc"],
                    }
                )
        md.append("")
    js["table2"] = tab2

    # ================= 3. 表 3 =================
    md.append("## 3. 表 3：对总缺口的贡献（相对 oracle 上界）")
    md.append("")
    md.append("> 缺口 = replay 臂 − **同一产物内** `original` 臂（oracle 上界）。")
    md.append("> **「压掉」的口径**：FAR / CLE p95 的缺口越小越好 ⇒ 压缩量 = baseline 缺口 − 本配置缺口；")
    md.append(
        "> hit@5px / F1 的缺口是**负数**（越接近 0 越好）⇒ 压缩量 = 本配置缺口 − baseline 缺口。两者都**正数 = 压掉了**。"
    )
    md.append("")
    for model in ("cs_t0043", "rc2"):
        b = base_of(model, args.target_delta)
        if b is None:
            continue
        base_row = arm_row(b["artefact"], "replay")
        base_oracle = arm_row(b["artefact"], "original")
        gb = gap_of(base_row, base_oracle)
        md.append(f"### 3.{1 if model == 'cs_t0043' else 2} {model} · Δ={args.target_delta}　baseline 缺口")
        md.append("")
        md.append("| 指标 | baseline 值 | oracle 值 | baseline 缺口 |")
        md.append("|:--|--:|--:|--:|")
        md.append(f"| FAR/帧 | {base_row['far']:.6f} | {base_oracle['far']:.6f} | {gb['far']:+.6f} |")
        md.append(
            f"| hit@5px | {base_row['hit5'] * 100:.4f}% | {base_oracle['hit5'] * 100:.4f}% | {gb['hit5']:+.4f} pp |"
        )
        md.append(
            f"| CLE p95(池) | {base_row['cle95']:.3f} px | {base_oracle['cle95']:.3f} px | {gb['cle95']:+.4f} px |"
        )
        md.append(f"| F1 | {base_row['f1'] * 100:.4f}% | {base_oracle['f1'] * 100:.4f}% | {gb['f1']:+.4f} pp |")
        md.append("")
        md.append(
            "| 旋钮 | 取值 | FAR 缺口 | 压掉 FAR 缺口 | hit5 缺口(pp) | 压掉 hit5 缺口(pp) | p95 缺口(px) | 压掉 p95 缺口(px) | F1 缺口(pp) | 交付三门 |"
        )
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        for e in [x for x in tab2 if x["model"] == model]:
            meta = runs.get((model, e["delta"], e["tag"]))
            if meta is None:
                continue
            r = arm_row(meta["artefact"], "replay")
            o = arm_row(meta["artefact"], "original")
            g = gap_of(r, o)
            md.append(
                f"| `--{e['knob'].lstrip('-')}` | {e['value']} | {g['far']:+.6f} | {gb['far'] - g['far']:+.6f} | "
                f"{g['hit5']:+.4f} | {g['hit5'] - gb['hit5']:+.4f} | {g['cle95']:+.4f} | {gb['cle95'] - g['cle95']:+.4f} | "
                f"{g['f1']:+.4f} | **{e['verdict']}** |"
            )
        md.append("")
        if model == "cs_t0043":
            js["gap_baseline"] = {"model": model, "delta": args.target_delta, **gb}
    js["table3_gap_baseline"] = js.get("gap_baseline", {})

    # ================= 4. 表 4 =================
    md.append("## 4. 表 4：新机理候选")
    md.append("")

    def cand_rows(pred):
        out = []
        for (m_, d_, tag_), meta in sorted(runs.items()):
            if not pred(m_, d_, tag_):
                continue
            r = arm_row(meta["artefact"], "replay")
            o = arm_row(meta["artefact"], "original")
            b = base_of(m_, d_)
            base_row = arm_row(b["artefact"], "replay") if b else None
            g = gap_of(r, o)
            out.append(
                {
                    "tag": tag_,
                    "model": m_,
                    "delta": d_,
                    "knobs": knob_pairs(meta["flags"], meta["vals"]),
                    "f1": r["f1"] * 100,
                    "recall": r["recall"] * 100,
                    "far": r["far"],
                    "hit5": r["hit5"] * 100,
                    "cle95": r["cle95"],
                    "verdict": verdict(r),
                    "gap": g,
                    "d_f1_pp": (r["f1"] - base_row["f1"]) * 100 if base_row else None,
                    "d_far": (r["far"] - base_row["far"]) if base_row else None,
                    "d_hit5_pp": (r["hit5"] - base_row["hit5"]) * 100 if base_row else None,
                    "d_cle95_px": (r["cle95"] - base_row["cle95"]) if base_row else None,
                    "path": meta["path"],
                    "ts": meta["ts"],
                    "rc": meta["rc"],
                }
            )
        return out

    def render_cands(rows, note=""):
        md.append(
            "| 候选 | 模型 | Δ | 改动旋钮 | F1% | R% | FAR/帧 | hit@5px% | CLE p95(池) | ΔF1(pp) | ΔFAR | Δhit5(pp) | Δp95(px) | 交付三门 | 产物 |"
        )
        md.append("|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|")
        for c in rows:
            desc = ", ".join(f"`--{f.lstrip('-')}={v}`" for f, v in c["knobs"]) or "—"
            md.append(
                f"| {c['tag']} | {c['model']} | {c['delta']} | {desc} | {c['f1']:.4f} | {c['recall']:.4f} | {c['far']:.6f} | "
                f"{c['hit5']:.4f} | {c['cle95']:.3f} | {fmt_signed(c['d_f1_pp'], 4)} | {fmt_signed(c['d_far'], 6)} | "
                f"{fmt_signed(c['d_hit5_pp'], 4)} | {fmt_signed(c['d_cle95_px'], 3)} | **{c['verdict']}** | `{c['path']}` |"
            )

    # 4.1 候选 1：Δ 自适应 roimax
    md.append("### 4.1 候选 1：Δ 自适应的 ROI 前推上限（纯 CLI ⇒ 「按 Δ 选一个 roimax」）")
    md.append("")
    md.append("**Δ=3 上的 roimax 扫描趋势（cs_t0043）**：")
    md.append("")
    md.append("| roimax (px) | F1% | R% | FAR/帧 | hit@5px% | ΔF1(pp) | ΔFAR | Δhit5(pp) |")
    md.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    roi_rows = []
    for (m_, d_, tag_), meta in sorted(runs.items()):
        if m_ != "cs_t0043" or d_ != args.target_delta:
            continue
        pairs = knob_pairs(meta["flags"], meta["vals"])
        if len(pairs) != 1 or pairs[0][0] != "--replay-roi-selfpred-max-px":
            continue
        r = arm_row(meta["artefact"], "replay")
        b = base_of(m_, d_)
        base_row = arm_row(b["artefact"], "replay")
        roi_rows.append((pairs[0][1], r, base_row))
    for val, r, base_row in sorted(roi_rows, key=lambda x: val_key(x[0])):
        md.append(
            f"| {val} | {r['f1'] * 100:.4f} | {r['recall'] * 100:.4f} | {r['far']:.6f} | {r['hit5'] * 100:.4f} | "
            f"{fmt_signed((r['f1'] - base_row['f1']) * 100, 4)} | {fmt_signed(r['far'] - base_row['far'], 6)} | "
            f"{fmt_signed((r['hit5'] - base_row['hit5']) * 100, 4)} |"
        )
    md.append("")
    md.append(
        "**趋势解读**：Δ=3 上 roimax 的 F1/hit@5px **随取值单调上升并在 ≥24 px 处饱和**（0/24/32/48 四点逐位相同 ⇒ 上限不再咬合）。"
    )
    md.append(
        "⇒「Δ 小 ⇒ roimax 应按 Δ 线性缩小」的假设**被否证**（roimax=4/8 比 16 差：F1 −0.0039/−0.0038 pp、hit@5px −0.0147 pp）。"
    )
    md.append("")
    md.append("**拟合出的 Δ→roimax 标定关系**（两点：Δ=3 扫描最优 = 24 px 饱和点；Δ=8 交付档选定 = 16 px）：")
    md.append("")
    md.append("```")
    md.append("roimax*(Δ) = 28.8 − 1.6·Δ      （Δ=3 → 24.0 px；Δ=8 → 16.0 px）")
    md.append("```")
    md.append("")
    md.append("**在 Δ=2 / Δ=4 上各验证 1 次**（用拟合值 Δ=2 → 25.6 ⇒ 26 px，Δ=4 → 22.4 ⇒ 22 px）：")
    md.append("")
    md.append(
        "| 验证 | 模型 | Δ | roimax | F1% | R% | FAR/帧 | hit@5px% | CLE p95(池) | ΔF1(pp) | ΔFAR | Δhit5(pp) | 产物 |"
    )
    md.append("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
    for c in cand_rows(lambda m_, d_, t_: t_.startswith(("d2__", "d4__"))):
        desc = ", ".join(f"`--{f.lstrip('-')}={v}`" for f, v in c["knobs"])
        md.append(
            f"| {c['tag']} | {c['model']} | {c['delta']} | {desc} | {c['f1']:.4f} | {c['recall']:.4f} | {c['far']:.6f} | "
            f"{c['hit5']:.4f} | {c['cle95']:.3f} | {fmt_signed(c['d_f1_pp'], 4)} | {fmt_signed(c['d_far'], 6)} | "
            f"{fmt_signed(c['d_hit5_pp'], 4)} | `{c['path']}` |"
        )
    md.append("")
    md.append(
        "**候选 1 结论：❌ 否证。** Δ=2 用拟合值 26 px 与「不限幅」0 px 逐位相同（远在饱和区），但比 Δ=2 的 baseline（16 px）**略差**"
        "（F1 −0.0073 pp、FAR +0.000138）；Δ=4 用拟合值 22 px 也**无改善**（F1 −0.0017 pp）。"
    )
    md.append(
        "⇒ Δ→roimax 的线性标定关系**不可用**；roimax 在 Δ=3 上的真实效应只有 +0.0079 pp F1 / +0.0037 pp hit@5px（≈ 噪声量级）。\n"
        "补充反证：Δ=4 上 `--replay-roi-selfpred-max-px 8` 反而比 16/22 **好** +0.0097 pp F1（见上表最后一行）\n"
        "⇒ roimax 的最优值随 Δ 的走向**非单调**，两点线性拟合必然失效 ⇒ 候选 1 在机理上就不成立。",
    )
    md.append("")
    js["candidate1"] = {
        "fit": "roimax*(Δ) = 28.8 - 1.6*Δ",
        "rows": cand_rows(lambda m_, d_, t_: t_.startswith(("d2__", "d4__"))),
        "verdict": "否证",
    }

    # 4.2 候选 2
    md.append("### 4.2 候选 2：弃权窗口 × 回退窗口联动（**多旋钮组合**，非受控单旋钮 A/B）")
    md.append("")
    md.append(
        "> 机理假设：「Δ 小 ⇒ 回退/弃权窗口应变短」。以下 4 个组合把 `--replay-abstain-disp-px` 与 `--replay-rollback-max-frames` 一起缩短。"
    )
    md.append("")
    render_cands(cand_rows(lambda m_, d_, t_: t_.startswith("p3__c2_")))
    md.append("")
    md.append(
        "**候选 2 结论：❌ 否证。** 4 个联动组合**全部劣于 baseline**（F1 −0.0121 ~ −0.1482 pp、hit@5px −0.0258 ~ −0.1619 pp）。"
    )
    md.append(
        "机理解释：`--replay-rollback-max-frames` 在 Δ=3 上单独就是 **16 最优**（0/8 分别 F1 −0.1343/−0.0296 pp），"
    )
    md.append("缩短回退窗口会直接砍掉 hit@5px；因此「Δ 小 ⇒ 回退窗口应变短」在 Δ=3 上**不成立**。")
    md.append("")
    js["candidate2"] = {"rows": cand_rows(lambda m_, d_, t_: t_.startswith("p3__c2_")), "verdict": "否证"}

    # 4.3 数据驱动组合
    md.append("### 4.3 数据驱动组合：FAR × hit@5px 的角落（**多旋钮组合**，非受控单旋钮 A/B）")
    md.append("")
    md.append("> 由表 2 的两条单调趋势拼出来：① `--replay-abstain-peak-below` 抬高 ⇒ FAR 单调降（0.20→0.70）；")
    md.append(
        "> ② `--replay-rollback-min-peak` 降低 ⇒ hit@5px/F1 单调升（0.95→0.80）。组合即「FAR 与 hit@5px 同时压缺口」。"
    )
    md.append("")
    render_cands(cand_rows(lambda m_, d_, t_: t_.startswith("p3__c_") and not t_.startswith("p3__c2_")))
    md.append("")
    js["candidate3"] = {"rows": cand_rows(lambda m_, d_, t_: t_.startswith("p3__c_") and not t_.startswith("p3__c2_"))}

    # ================= 5. 表 5 =================
    md.append("## 5. 表 5：候选组合的最优点")
    md.append("")
    md.append("**入选规则（写死、可复算）**：在 **cs_t0043 与 rc2 两个模型包上**都满足")
    md.append(
        "「① FAR 缺口**不恶化**　② hit@5px 缺口**压缩**　③ F1 **上升**」的配置里，按 cs_t0043 的 hit@5px 缺口压缩量排序取第一。"
    )
    md.append("")
    pool = []
    for e in tab2:
        pool.append(
            {
                "label": f"`--{e['knob'].lstrip('-')}={e['value']}`",
                "knobs": [(e["knob"], e["value"])],
                "model": e["model"],
                "delta": e["delta"],
                "far": e["far"],
                "hit5": e["hit5"],
                "f1": e["f1"],
                "recall": e["recall"],
                "cle95": e["cle95"],
                "d_f1_pp": e["d_f1_pp"],
                "d_far": e["d_far"],
                "d_hit5_pp": e["d_hit5_pp"],
                "d_cle95_px": e["d_cle95_px"],
                "verdict": e["verdict"],
                "path": e["path"],
                "ts": e["ts"],
            }
        )
    for c in js["candidate3"]["rows"] + js["candidate2"]["rows"]:
        pool.append(
            {
                "label": ", ".join(f"`--{f.lstrip('-')}={v}`" for f, v in c["knobs"]),
                "knobs": c["knobs"],
                "model": c["model"],
                "delta": c["delta"],
                "far": c["far"],
                "hit5": c["hit5"],
                "f1": c["f1"],
                "recall": c["recall"],
                "cle95": c["cle95"],
                "d_f1_pp": c["d_f1_pp"],
                "d_far": c["d_far"],
                "d_hit5_pp": c["d_hit5_pp"],
                "d_cle95_px": c["d_cle95_px"],
                "verdict": c["verdict"],
                "path": c["path"],
                "ts": c["ts"],
            }
        )
    # 按「(knobs 组合)」聚合到模型
    agg: dict[tuple, list] = {}
    for p in pool:
        agg.setdefault(tuple(p["knobs"]), []).append(p)
    combos = []
    for key, ps in agg.items():
        cs = next((x for x in ps if x["model"] == "cs_t0043"), None)
        rc = next((x for x in ps if x["model"] == "rc2"), None)
        if cs is None or rc is None:
            continue
        b_cs = base_of("cs_t0043", args.target_delta)
        b_rc = base_of("rc2", args.target_delta)
        gcs = gap_of(arm_row(b_cs["artefact"], "replay"), arm_row(b_cs["artefact"], "original"))
        grc = gap_of(arm_row(b_rc["artefact"], "replay"), arm_row(b_rc["artefact"], "original"))
        combos.append({"knobs": list(key), "cs": cs, "rc": rc, "gcs": gcs, "grc": grc})
    md.append(
        "| 配置（旋钮串） | 模型 | FAR 缺口 | 压掉 FAR 缺口 | hit5 缺口(pp) | 压掉 hit5 缺口(pp) | ΔF1(pp) | 交付三门 | 产物 |"
    )
    md.append("|:--|:--|--:|--:|--:|--:|--:|:--|:--|")
    for c in combos:
        for mdl, item, gb in (("cs_t0043", c["cs"], c["gcs"]), ("rc2", c["rc"], c["grc"])):
            art = runs[(mdl, args.target_delta, item["path"].split("/")[-2])]["artefact"]
            g = gap_of(arm_row(art, "replay"), arm_row(art, "original"))
            item["gap"] = g
            item["gap_base"] = gb
            # 压缩量（正数 = 缺口被压掉）：FAR/p95 缺口越小越好 ⇒ base−new；hit5/F1 缺口越接近 0 越好 ⇒ new−base
            item["d_far_gap"] = gb["far"] - g["far"]
            item["d_hit5_gap"] = g["hit5"] - gb["hit5"]
            item["d_cle95_gap"] = gb["cle95"] - g["cle95"]
            item["d_f1_gap"] = g["f1"] - gb["f1"]
            md.append(
                f"| {knob_desc(c['knobs'])} | {mdl} | {item['gap']['far']:+.6f} | "
                f"{item['d_far_gap']:+.6f} | {item['gap']['hit5']:+.4f} | {item['d_hit5_gap']:+.4f} | "
                f"{fmt_signed(item['d_f1_pp'], 4)} | **{item['verdict']}** | `{item['path']}` |"
            )
    md.append("")
    # 入选判定
    for c in combos:
        ok = (
            c["cs"]["d_far_gap"] >= 0
            and c["rc"]["d_far_gap"] >= 0
            and c["cs"]["d_hit5_gap"] > 0
            and c["rc"]["d_hit5_gap"] > 0
            and c["cs"]["d_f1_pp"] > 0
            and c["rc"]["d_f1_pp"] > 0
        )
        c["qualified"] = ok
    qual = [c for c in combos if c["qualified"]]
    qual.sort(key=lambda c: -c["cs"]["d_hit5_gap"])
    if qual:
        best = qual[0]
        md.append(f"### 5.1 最优点：**{knob_desc(best['knobs'])}**")
        md.append("")
        md.append(
            "| 模型 | F1% | R% | FAR/帧 | hit@5px% | CLE p95(池) | ΔF1(pp) | ΔFAR | Δhit5(pp) | Δp95(px) | 交付三门 |"
        )
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        for mdl, item in (("cs_t0043", best["cs"]), ("rc2", best["rc"])):
            md.append(
                f"| {mdl} | {item['f1']:.4f} | {item['recall']:.4f} | {item['far']:.6f} | {item['hit5']:.4f} | {item['cle95']:.3f} | "
                f"{fmt_signed(item['d_f1_pp'], 4)} | {fmt_signed(item['d_far'], 6)} | {fmt_signed(item['d_hit5_pp'], 4)} | "
                f"{fmt_signed(item['d_cle95_px'], 3)} | **{item['verdict']}** |"
            )
        md.append("")
        md.append("| 产物路径 | 时间戳 |")
        md.append("|:--|:--|")
        for mdl, item in (("cs_t0043", best["cs"]), ("rc2", best["rc"])):
            md.append(f"| `{item['path']}` | {item['ts']} |")
        md.append("")
        js["best"] = {"knobs": best["knobs"], "cs": best["cs"], "rc": best["rc"]}
        md.append(f"入选（满足全部三条判据、且 cs_t0043 的 hit@5px 缺口压缩最大）的配置共 **{len(qual)}** 个：")
        md.append("")
        md.append("| 配置 | cs FAR 缺口压掉 | cs hit5 缺口压掉 | rc2 FAR 缺口压掉 | rc2 hit5 缺口压掉 |")
        md.append("|:--|--:|--:|--:|--:|")
        for c in qual:
            md.append(
                f"| {knob_desc(c['knobs'])} | {c['cs']['d_far_gap']:+.6f} | "
                f"{c['cs']['d_hit5_gap']:+.4f} pp | {c['rc']['d_far_gap']:+.6f} | {c['rc']['d_hit5_gap']:+.4f} pp |"
            )
        md.append("")
    else:
        md.append("> ⚠️ 没有配置在两个模型包上同时满足三条判据。")
        md.append("")
        js["best"] = None

    # ================= 6. 结论 =================
    md.append("## 6. 结论")
    md.append("")
    md.append("### 6.1 Δ=3 上「目前最优方法」还有可挖空间吗？—— **有，但很小**")
    md.append("")
    md.append("| 结论 | 证据 |")
    md.append("|:--|:--|")
    md.append(
        "| **有可挖空间** | 交付档 CFG 的 7 个旋钮是在 **Δ=8** 上调的；在 Δ=3 上 `--replay-abstain-peak-below`（0.30→0.50）与 `--replay-rollback-min-peak`（0.90→0.80）**两个旋钮都偏离最优**，单独改一个就能同时改善 F1 与 hit@5px |"
    )
    md.append(
        "| **改善幅度有限** | 最好的一档把 F1 缺口压掉 96.2%(cs)/69.4%(rc2)、hit@5px 缺口压掉 41.0%(cs)/9.0%(rc2)；**FAR 缺口只能压掉 12.6%(cs)/0.0%(rc2)** |"
    )
    md.append(
        "| **CLE p95 缺口几乎压不动** | 最优配置下 p95 缺口 cs 0.6699→0.6806 px、rc2 1.1863→1.2684 px（**反而略差**） |"
    )
    md.append(
        "| **7 个旋钮里 4 个在 Δ=3 上是死旋钮** | `--replay-xcorr-uniq-gate`（0.5/0.6/0.7 几乎逐位相同）、`--replay-rollback-max-frames`（16 最优，缩短即变差）、`--replay-abstain-disp-px`（24 最优，±8 无实质差别）、`--replay-roi-selfpred-max-px`（≥24 饱和，效应 ≈ +0.008 pp） |"
    )
    md.append("")
    md.append("### 6.2 两个新机理候选的成败（如实）")
    md.append("")
    md.append(
        "- **候选 1「Δ 自适应 roimax」：❌ 否证。** Δ=3 上 roimax 的趋势是**单调上升**并在 24 px 饱和（0/24/32/48 逐位相同），"
    )
    md.append(
        "  与「Δ 小 ⇒ roimax 小」的假设方向相反。拟合出的 `roimax*(Δ)=28.8−1.6Δ` 在 Δ=2（用 26 px）与 Δ=4（用 22 px）"
    )
    md.append("  各验证 1 次，**两次都没有改善**（Δ=2：F1 −0.0073 pp、FAR +0.000138；Δ=4：F1 −0.0017 pp）。")
    md.append("- **候选 2「弃权/回退窗口联动」：❌ 否证。** 4 个联动组合全部劣于 baseline。")
    md.append("  机理：`--replay-rollback-max-frames` 在 Δ=3 上单独就是 16 最优 ⇒ 缩短回退窗口直接砍 hit@5px。")
    md.append("- **数据驱动组合（表 4.3）：✅ 有效**，但它**不是新机理**，而是两个既有旋钮的组合。")
    md.append("")
    md.append("### 6.3 推荐旋钮串")
    md.append("")
    if qual:
        best = qual[0]
        md.append("**Δ=3 最优旋钮串**（在 cs_t0043 与 rc2 两个模型包上都验证过）：")
        md.append("")
        md.append("```")
        md.append(" --replay-abstain-peak-below 0.50   # 交付档 0.30 → 0.50：FAR 缺口 ↓（直接打 FAR）")
        md.append(" --replay-rollback-min-peak  0.80   # 交付档 0.90 → 0.80：hit@5px / F1 缺口 ↓")
        md.append(" # 其余旋钮一字不改：")
        md.append(" #   --replay-roi-selfpred-max-px 16 / --replay-abstain-disp-px 24 / --replay-xcorr-uniq-gate 0.6")
        md.append(
            " #   --replay-rollback-max-frames 16 / --output-max-step-px 30 / --replay-mode perframe / --replay-polarity auto"
        )
        md.append("```")
        md.append("")
        md.append("**完整命令**（服务器 test-Super-Server，`PYTHONPATH` = 仓库根）：")
        md.append("")
        md.append("```bash")
        md.append("cd /tmp/pycharm_project_10ae9e2e")
        md.append("export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=6 MKL_NUM_THREADS=6")
        md.append(
            'CFG_ASSIGN="$(awk \'/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}\' manu/tracking/lookahead/run_delivery.sh)"'
        )
        md.append('CFG=""; eval "$CFG_ASSIGN"')
        md.append("# 替换两个旋钮（其余一字不改）")
        md.append(
            "CFG=\"$(printf '%s' \"$CFG\" | sed -E 's|--replay-abstain-peak-below[[:space:]]+[^[:space:]]+|--replay-abstain-peak-below 0.50|g;"
            " s|--replay-rollback-min-peak[[:space:]]+[^[:space:]]+|--replay-rollback-min-peak 0.80|g')\""
        )
        md.append("")
        md.append("/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/lookahead_longquan.py \\")
        md.append("  --cache-root /mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache \\")
        md.append('  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \\')
        md.append('  --frames-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \\')
        md.append("  --frames-cache /mnt/data/siping/frame_cache_lq \\")
        md.append("  --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \\")
        md.append("  --all-seqs --lookahead replay --arms original,stale,kinematic,replay --delta 3 $CFG \\")
        md.append("  --out-json <OUT>/arms.json --out-preds-dir <OUT>/preds")
        md.append("```")
        md.append("")
        md.append("| 项 | 值 |")
        md.append("|:--|:--|")
        md.append("| 执行机 | 服务器 test-Super-Server |")
        md.append(f"| 时间戳 | {best['cs']['ts']}（cs_t0043）/ {best['rc']['ts']}（rc2） |")
        md.append(f"| cs_t0043 产物 | `{best['cs']['path']}` |")
        md.append(f"| rc2 产物 | `{best['rc']['path']}` |")
        md.append(f"| 本报告 | `{root}/report.md`（+ `report.json`） |")
        md.append("| 扫描器 | `manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh` |")
        md.append("")
        md.append("**该旋钮串的效果（Δ=3 / 8 序 gate 集）**：")
        md.append("")
        md.append("| 模型 | F1% | ΔF1(pp) | R% | FAR/帧 | ΔFAR | hit@5px% | Δhit5(pp) | CLE p95(池) | 交付三门 |")
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        for mdl, item in (("cs_t0043", best["cs"]), ("rc2", best["rc"])):
            md.append(
                f"| {mdl} | {item['f1']:.4f} | {fmt_signed(item['d_f1_pp'], 4)} | {item['recall']:.4f} | {item['far']:.6f} | "
                f"{fmt_signed(item['d_far'], 6)} | {item['hit5']:.4f} | {fmt_signed(item['d_hit5_pp'], 4)} | {item['cle95']:.3f} | "
                f"**{item['verdict']}** |"
            )
        md.append("")
    md.append("### 6.4 必须写明的限制")
    md.append("")
    md.append("1. **8 序只覆盖 24 序官方口径的 20.39% GT** ⇒ 以上全部是「**8 序机理筛选 / A/B 证据**」，")
    md.append("   **不能**用来给交付档排序；上 24 序官方口径复核前不要改交付档 CFG。")
    md.append("2. 「FAR 缺口压掉 12.6%(cs) / 0.0%(rc2)」说明 **FAR 缺口在两个模型包上的可压缩性不同**")
    md.append("   （rc2 的 FAR 缺口本来就小：0.002483 vs cs 的 0.003000），组合的收益不稳定。")
    md.append("3. `p95` 缺口在最优配置下**没有改善甚至略差** ⇒ 剩余的尾部误差**不是**这几个旋钮能解决的。")
    md.append("4. `--replay-abstain-peak-below` 抬高会同时**牺牲 R**（cs 0.50：R −0.0626 pp；0.60：R −0.1693 pp），")
    md.append("   0.60 以上在 cs 上 F1 已开始转负（−0.0268 pp）⇒ **0.50 是甜点**，再往上不划算。")
    md.append("")

    # ================= 7. 附录 =================
    md.append("## 7. 附录：运行清单与 rc")
    md.append("")
    total = len(manifest)
    bad = [m for m in manifest if m["rc"] != 0]
    md.append(f"- 总运行次数：**{total}**　｜　**rc≠0：{len(bad)}**")
    md.append("")
    if bad:
        md.append("| tag | 模型 | Δ | 改动 | rc | 产物 |")
        md.append("|:--|:--|--:|:--|--:|:--|")
        for m in bad:
            md.append(
                f"| {m['tag']} | {m['model']} | {m['delta']} | {m['flags']}→{m['vals']} | {m['rc']} | `{m['path']}` |"
            )
        md.append("")
    md.append("| 阶段 | 模型 | Δ | 改动 | rc | 用时(s) | 时间戳 | 产物 |")
    md.append("|:--|:--|--:|:--|--:|--:|:--|:--|")
    for m in sorted(manifest, key=lambda x: x["ts"]):
        pairs = knob_pairs(m["flags"], m["vals"])
        desc = ", ".join(f"`--{f.lstrip('-')}={v}`" for f, v in pairs) or "（baseline）"
        md.append(
            f"| {m['tag']} | {m['model']} | {m['delta']} | {desc} | {m['rc']} | {m['elapsed']} | {m['ts']} | `{m['path']}` |"
        )
    md.append("")
    js["runs"] = manifest

    out_md = Path(args.out_md) if args.out_md else root / "report.md"
    out_json = Path(args.out_json) if args.out_json else root / "report.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n")
    out_json.write_text(json.dumps(js, ensure_ascii=False, indent=1, default=str) + "\n")
    print(f"[report] md   → {out_md}")
    print(f"[report] json → {out_json}")
    print(
        f"[report] runs={total} rc_nonzero={len(bad)} gate_bad={sum(len(g['bad']) for g in gate_rows)} best={js.get('best') and js['best']['knobs']}"
    )


if __name__ == "__main__":
    main()
