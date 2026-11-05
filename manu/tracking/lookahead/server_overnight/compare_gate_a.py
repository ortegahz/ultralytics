#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""P2 Gate A 比对器：原脚本 vs 插件（``--lookahead off``）的逐位一致性判定。

口径说明（两层都做，缺一不可）：

* **L1 整份文件 md5**（仓库既有口径，见 ``manu/tracking/lookahead/check_off_bitidentical.sh``）：
  对每个序列的 ``<seq>.json`` 直接算文件 md5 ⇒ 最严格，也最容易被 ``config`` 块里
  的路径写法/时间戳等**与跟踪资产无关**的差异打挂。
* **L2 逐字段比对**（仓库既有口径，见 ``manu/tracking/lookahead/check_newknobs_off.sh``）：
  比 ``per_frame`` + ``metrics`` 两个子块，**不比**整文件 md5（该脚本的注释明确写了
  「不能比整文件 md5：config.cache_root 的相对/绝对路径不同会造成假阳性」）。

本脚本在此之上再补两层，避免「一比就挂」或「一比就过」两种失真：

* **L3 规范化 payload md5**：把 ``{config, metrics, per_frame}`` 用 ``sort_keys=True``
  重新序列化后取 md5 ⇒ **键序无关**的「整份 JSON 逐位」判定。
* **L4 跟踪资产 md5**：只取 ``{metrics, per_frame}`` 取 md5 ⇒ **完全剔除 config 块**
  （config 只描述「这份产物是怎么跑出来的」，不含任何跟踪数值）。

**本脚本只读两边产物，绝不修改任何一方的文件。**
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]  # repo root (manu/tracking/lookahead/server_overnight/ -> 4 levels up)

KEY_METRICS = ("f1", "recall", "precision", "far_per_frame", "tp", "fp", "fn")


def read_cmd(path: Path) -> str:
    """读一条命令记录（不存在时返回空串）。"""
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def md5_obj(obj) -> str:
    return md5_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def diff_config(a: dict, b: dict) -> dict:
    """列出 config 块里两边不同的键（不做任何静默丢弃）。"""
    keys = sorted(set(a) | set(b))
    out = {}
    for key in keys:
        va, vb = a.get(key, "<missing>"), b.get(key, "<missing>")
        if va != vb:
            out[key] = {"A": va, "B": vb}
    return out


def compare_predictions(a_dir: Path, b_dir: Path) -> dict:
    a_files = {p.name: p for p in sorted(a_dir.glob("*.json")) if p.name != "summary.json"}
    b_files = {p.name: p for p in sorted(b_dir.glob("*.json")) if p.name != "summary.json"}
    rows = []
    for name in sorted(set(a_files) | set(b_files)):
        pa, pb = a_files.get(name), b_files.get(name)
        if pa is None or pb is None:
            rows.append({"file": name, "present_A": pa is not None, "present_B": pb is not None, "identical": False})
            continue
        raw_a = pa.read_bytes()
        raw_b = pb.read_bytes()
        da, db = load(pa), load(pb)
        cfg_diff = diff_config(da.get("config", {}), db.get("config", {}))
        rows.append(
            {
                "file": name,
                "bytes_A": len(raw_a),
                "bytes_B": len(raw_b),
                "md5_raw_A": md5_bytes(raw_a),
                "md5_raw_B": md5_bytes(raw_b),
                "L1_raw_md5_equal": md5_bytes(raw_a) == md5_bytes(raw_b),
                "L2_per_frame_equal": da.get("per_frame") == db.get("per_frame"),
                "L2_metrics_equal": da.get("metrics") == db.get("metrics"),
                "L2_config_equal": not cfg_diff,
                "config_diff": cfg_diff,
                "L3_payload_md5_A": md5_obj(
                    {"config": da.get("config"), "metrics": da.get("metrics"), "per_frame": da.get("per_frame")}
                ),
                "L3_payload_md5_B": md5_obj(
                    {"config": db.get("config"), "metrics": db.get("metrics"), "per_frame": db.get("per_frame")}
                ),
                "L4_assets_md5_A": md5_obj({"metrics": da.get("metrics"), "per_frame": da.get("per_frame")}),
                "L4_assets_md5_B": md5_obj({"metrics": db.get("metrics"), "per_frame": db.get("per_frame")}),
                "metrics_A": da.get("metrics", {}),
                "metrics_B": db.get("metrics", {}),
            }
        )
    for row in rows:
        row["L3_payload_md5_equal"] = row.get("L3_payload_md5_A") == row.get("L3_payload_md5_B")
        row["L4_assets_md5_equal"] = row.get("L4_assets_md5_A") == row.get("L4_assets_md5_B")
    return {
        "rows": rows,
        "n_files": len(rows),
        "n_L1_equal": sum(1 for r in rows if r.get("L1_raw_md5_equal")),
        "n_L2_per_frame_equal": sum(1 for r in rows if r.get("L2_per_frame_equal")),
        "n_L2_metrics_equal": sum(1 for r in rows if r.get("L2_metrics_equal")),
        "n_L2_config_equal": sum(1 for r in rows if r.get("L2_config_equal")),
        "n_L3_equal": sum(1 for r in rows if r.get("L3_payload_md5_equal")),
        "n_L4_equal": sum(1 for r in rows if r.get("L4_assets_md5_equal")),
    }


def compare_report(a_path: Path, b_path: Path) -> dict:
    """比对 out-json（arms.json）：插件的包络多出 arms/lookahead 两个顶层键，
    且 elapsed_seconds 是墙钟 ⇒ 这两个**按定义**不可能相同。判定落在
    {config, overall, per_sequence} 投影上。"""
    da, db = load(a_path), load(b_path)
    proj_keys = ("config", "overall", "per_sequence")
    pa = {k: da.get(k) for k in proj_keys}
    pb = {k: db.get(k) for k in proj_keys}
    return {
        "raw_md5_A": md5_bytes(a_path.read_bytes()),
        "raw_md5_B": md5_bytes(b_path.read_bytes()),
        "raw_md5_equal": md5_bytes(a_path.read_bytes()) == md5_bytes(b_path.read_bytes()),
        "top_keys_A": sorted(da.keys()),
        "top_keys_B": sorted(db.keys()),
        "envelope_only_B": sorted(set(db) - set(da)),
        "projection_keys": list(proj_keys),
        "projection_equal": pa == pb,
        "projection_md5_A": md5_obj(pa),
        "projection_md5_B": md5_obj(pb),
        "config_diff": diff_config(da.get("config", {}), db.get("config", {})),
        "overall_A": da.get("overall", {}),
        "overall_B": db.get("overall", {}),
        "per_sequence_A": da.get("per_sequence", {}),
        "per_sequence_B": db.get("per_sequence", {}),
    }


def metric_table(a_dir: Path, b_dir: Path) -> list[dict]:
    """关键字段逐项对比表：F1/R/P/FAR/TP/FP/FN（逐序列 + overall）。"""
    out = []
    for name in sorted(p.stem for p in a_dir.glob("*.json") if p.name != "summary.json"):
        pa, pb = a_dir / f"{name}.json", b_dir / f"{name}.json"
        if not pb.exists():
            continue
        ma, mb = load(pa).get("metrics", {}), load(pb).get("metrics", {})
        row = {"scope": name}
        for key in KEY_METRICS:
            va, vb = ma.get(key), mb.get(key)
            row[f"A_{key}"] = va
            row[f"B_{key}"] = vb
            row[f"equal_{key}"] = va == vb
        out.append(row)
    sa, sb = load(a_dir / "summary.json"), load(b_dir / "summary.json")
    row = {"scope": "OVERALL"}
    for key in KEY_METRICS:
        va, vb = sa.get("overall", {}).get(key), sb.get("overall", {}).get(key)
        row[f"A_{key}"] = va
        row[f"B_{key}"] = vb
        row[f"equal_{key}"] = va == vb
    out.append(row)
    return out


def check_ruler1(arms_json: Path) -> dict:
    """尺子 1：同一产物内四臂的 source_frame.overall.f1 必须逐位相等（ΔF1 = 0.000000pp）。"""
    if not arms_json.exists():
        return {"path": str(arms_json), "exists": False}
    data = load(arms_json)
    arms = data.get("arms", {})
    f1s = {name: payload.get("source_frame", {}).get("overall", {}).get("f1") for name, payload in arms.items()}
    uniq = sorted({repr(v) for v in f1s.values()})
    ref = next(iter(f1s.values())) if f1s else None
    return {
        "path": str(arms_json),
        "exists": True,
        "arms": list(arms),
        "source_frame_f1": f1s,
        "all_bit_identical": len(uniq) == 1,
        "delta_pp": 0.0 if len(uniq) == 1 else None,
        "distinct_values": uniq,
        "delta_f1_vs_original_field": {
            name: payload.get("delta_f1_vs_original") for name, payload in arms.items()
        },
    }


def render_markdown(result: dict) -> str:
    lines = []
    ap = lines.append
    ap("# P2 Gate A —— 「关开关 ⇒ 逐位一致」复核报告")
    ap("")
    ap(f"- 执行机：**服务器 test-Super-Server**（本沙箱无 GPU；本任务纯 CPU 复算）")
    ap(f"- 时间戳：**{result['timestamp']}**")
    ap(f"- 仓库根：`{REPO}`")
    ap(f"- 结论：**{result['verdict']}**")
    ap("")
    ap("## 0. 判定口径（比的是哪一层）")
    ap("")
    ap("| 层 | 含义 | 口径来源 |")
    ap("| --- | --- | --- |")
    ap("| L1 | 逐序列 `<seq>.json` **整份文件 md5** | 仓库既有：`check_off_bitidentical.sh` |")
    ap(
        "| L2 | `per_frame` / `metrics` / `config` **逐块相等** | 仓库既有：`check_newknobs_off.sh`（该脚本注释明确：*不能比整文件 md5，config.cache_root 的路径写法会造成假阳性*） |"
    )
    ap("| L3 | `{config, metrics, per_frame}` **规范化 payload md5**（`sort_keys=True` ⇒ 键序无关） | 本次新增，兼 L1 与 L2 |")
    ap("| L4 | `{metrics, per_frame}` **跟踪资产 md5**（完全剔除 config 块） | 本次新增，config 只描述「怎么跑出来的」，不含跟踪数值 |")
    ap("")
    ap("> **不修改任何一方**：所有规范化都在**比对器**里做（读 JSON → 重排序 → 重算 md5），"
       "两边的原始 JSON 文件一个字节都没动。")
    ap("")
    ap("## 1. 两边命令全文")
    ap("")
    for model, blob in result["models"].items():
        ap(f"### {model} —— {blob['label']}")
        ap("")
        ap("```bash")
        ap(f"# A = 原脚本（manu/evaluation/system_sota_longquan.py，只读、禁改）")
        ap(blob["command_A"])
        ap("")
        ap(f"# B = 插件（--lookahead off --arms original）")
        ap(blob["command_B"])
        ap("```")
        ap("")
    ap("## 2. md5 逐序列明细")
    ap("")
    for model, blob in result["models"].items():
        ap(f"### {model}")
        ap("")
        ap(
            "| 文件 | md5 A（原脚本） | md5 B（插件 off） | L1 整份 md5 | L2 per_frame | L2 metrics | L2 config | L3 payload md5 | L4 资产 md5 |"
        )
        ap("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in blob["preds"]["rows"]:
            ap(
                f"| `{row['file']}` | `{row['md5_raw_A']}` | `{row['md5_raw_B']}` | "
                f"{'✅' if row['L1_raw_md5_equal'] else '❌'} | "
                f"{'✅' if row['L2_per_frame_equal'] else '❌'} | "
                f"{'✅' if row['L2_metrics_equal'] else '❌'} | "
                f"{'✅' if row['L2_config_equal'] else '❌'} | "
                f"{'✅' if row['L3_payload_md5_equal'] else '❌'} | "
                f"{'✅' if row['L4_assets_md5_equal'] else '❌'} |"
            )
        p = blob["preds"]
        ap("")
        ap(
            f"- 汇总：文件数 **{p['n_files']}** ｜ L1 整份 md5 相同 **{p['n_L1_equal']}/{p['n_files']}** ｜ "
            f"L2 per_frame 相同 **{p['n_L2_per_frame_equal']}/{p['n_files']}** ｜ "
            f"L2 metrics 相同 **{p['n_L2_metrics_equal']}/{p['n_files']}** ｜ "
            f"L2 config 相同 **{p['n_L2_config_equal']}/{p['n_files']}** ｜ "
            f"L3 payload md5 相同 **{p['n_L3_equal']}/{p['n_files']}** ｜ "
            f"L4 资产 md5 相同 **{p['n_L4_equal']}/{p['n_files']}**"
        )
        cfg_any = [r for r in p["rows"] if r.get("config_diff")]
        if cfg_any:
            ap("")
            ap("⚠️ config 块差异（逐键）：")
            for row in cfg_any:
                for key, val in row["config_diff"].items():
                    ap(f"  - `{row['file']}` → `{key}`: A=`{val['A']}` / B=`{val['B']}`")
        ap("")
    ap("## 3. out-json（arms.json）比对")
    ap("")
    for model, blob in result["models"].items():
        r = blob["report"]
        ap(f"### {model}")
        ap("")
        ap(f"- 原脚本 md5：`{r['raw_md5_A']}`")
        ap(f"- 插件 md5：`{r['raw_md5_B']}`")
        ap(f"- 整份 md5 相同：**{'是' if r['raw_md5_equal'] else '否'}**")
        ap(f"- 插件包络多出的顶层键：**{r['envelope_only_B']}**（`arms` = 四臂汇总、`lookahead` = 旋钮快照；`elapsed_seconds` = 墙钟）")
        proj = "/".join(r["projection_keys"])
        same = "是" if r["projection_equal"] else "否"
        ap(f"- **判定落在投影 `{proj}` 上**（剔除包络与墙钟）⇒ 相同：**{same}**")
        ap(f"- 投影 md5： A=`{r['projection_md5_A']}` ｜ B=`{r['projection_md5_B']}`")
        if r["config_diff"]:
            ap("- ⚠️ config 块差异：")
            for key, val in r["config_diff"].items():
                ap(f"  - `{key}`: A=`{val['A']}` / B=`{val['B']}`")
        ap("")
    ap("## 4. 关键字段逐项对比（F1 / R / P / FAR / TP / FP / FN）")
    ap("")
    for model, blob in result["models"].items():
        ap(f"### {model}")
        ap("")
        ap(
            "| 范围 | F1 A | F1 B | F1 同 | R A | R B | R 同 | P A | P B | P 同 | FAR A | FAR B | FAR 同 | TP A | TP B | FP A | FP B | FN A | FN B |"
        )
        ap(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        for row in blob["metric_table"]:
            ap(
                f"| {row['scope']} | {row['A_f1']!r} | {row['B_f1']!r} | {'✅' if row['equal_f1'] else '❌'} | "
                f"{row['A_recall']!r} | {row['B_recall']!r} | {'✅' if row['equal_recall'] else '❌'} | "
                f"{row['A_precision']!r} | {row['B_precision']!r} | {'✅' if row['equal_precision'] else '❌'} | "
                f"{row['A_far_per_frame']!r} | {row['B_far_per_frame']!r} | "
                f"{'✅' if row['equal_far_per_frame'] else '❌'} | "
                f"{row['A_tp']} | {row['B_tp']} | {row['A_fp']} | {row['B_fp']} | {row['A_fn']} | {row['B_fn']} |"
            )
        ap("")
    ap("## 5. 尺子 1 校验（同一产物内四臂 `source_frame.overall.f1` 逐位相等）")
    ap("")
    ruler = result["ruler1"]
    ap(f"- 产物：`{ruler['path']}`")
    if ruler.get("exists"):
        f1s = " ｜ ".join(f"`{k}` = `{v!r}`" for k, v in ruler["source_frame_f1"].items())
        ap(f"- 四臂 `source_frame.overall.f1`：{f1s}")
        ap(f"- 逐位相等：**{'是' if ruler['all_bit_identical'] else '否'}**（ΔF1 = {ruler['delta_pp']} pp）")
        df1 = " ｜ ".join(f"`{k}` = `{v!r}`" for k, v in ruler["delta_f1_vs_original_field"].items())
        ap(f"- 产物自带的 `delta_f1_vs_original`：{df1}")
    else:
        ap("- ❌ 产物缺失")
    ap("")
    ap("## 6. 结论")
    ap("")
    for line in result["conclusion_lines"]:
        ap(f"- {line}")
    ap("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate A 比对器：原脚本 vs 插件(--lookahead off)")
    ap.add_argument("--gate-root", default=str(REPO / "runs/lookahead/gateA"))
    ap.add_argument("--ruler-arms-json", default=str(REPO / "runs/lookahead/fi_nogmc/cs_t0043/d3/arms.json"))
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    import datetime

    gate_root = Path(args.gate_root).resolve()
    models: dict[str, dict] = {}
    for model_dir in sorted(p for p in gate_root.iterdir() if p.is_dir()):
        a_dir, b_dir = model_dir / "orig" / "preds", model_dir / "plug" / "preds"
        if not (a_dir.is_dir() and b_dir.is_dir()):
            continue
        cmd_a = read_cmd(model_dir / "orig" / "command.txt")
        cmd_b = read_cmd(model_dir / "plug" / "command.txt")
        blob = {
            "label": read_cmd(model_dir / "label.txt") or model_dir.name,
            "command_A": cmd_a,
            "command_B": cmd_b,
            "preds": compare_predictions(a_dir, b_dir),
            "report": compare_report(model_dir / "orig" / "arms.json", model_dir / "plug" / "arms.json"),
            "metric_table": metric_table(a_dir, b_dir),
        }
        n_files = blob["preds"]["n_files"]
        blob["preds_all_L1_equal"] = n_files > 0 and blob["preds"]["n_L1_equal"] == n_files
        blob["preds_all_L4_equal"] = n_files > 0 and blob["preds"]["n_L4_equal"] == n_files
        blob["report_projection_equal"] = blob["report"]["projection_equal"]
        blob["all_ok"] = blob["preds_all_L1_equal"] and blob["preds_all_L4_equal"] and blob["report_projection_equal"]
        models[model_dir.name] = blob

    ruler1 = check_ruler1(Path(args.ruler_arms_json))
    conclusion = []
    all_ok = bool(models) and all(b["all_ok"] for b in models.values())
    for name, blob in models.items():
        conclusion.append(
            f"**{name}**：逐序列整份 JSON md5 {'全部相同 ✅' if blob['preds_all_L1_equal'] else '存在差异 ❌'}"
            f"（{blob['preds']['n_L1_equal']}/{blob['preds']['n_files']}）；"
            f"跟踪资产 md5 {'全部相同 ✅' if blob['preds_all_L4_equal'] else '存在差异 ❌'}；"
            f"out-json 投影 {'相同 ✅' if blob['report_projection_equal'] else '不同 ❌'}。"
        )
    conclusion.append(
        f"**尺子 1**：{'四臂 source_frame.overall.f1 逐位相等 ✅（ΔF1 = 0.000000pp）' if ruler1.get('all_bit_identical') else '未通过 ❌'}。"
    )
    conclusion.append(
        "**判定**：外挂 look-ahead 在 `--lookahead off` 时**没有扰动**跟踪资产 —— 逐位一致成立 ✅"
        if all_ok and ruler1.get("all_bit_identical")
        else "**判定**：存在未通过项 ❌（见上表）"
    )

    result = {
        "task": "P2 Gate A —— 关开关逐位一致（cs_t0043 主 / rc2 对照）",
        "host": "test-Super-Server",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "verdict": "PASS" if all_ok and ruler1.get("all_bit_identical") else "FAIL",
        "models": models,
        "ruler1": ruler1,
        "conclusion_lines": conclusion,
    }

    out_json = Path(args.out_json) if args.out_json else gate_root / "gateA.json"
    out_md = Path(args.out_md) if args.out_md else gate_root / "gateA.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(result), encoding="utf-8")

    for name, blob in models.items():
        print(
            f"[{name}] L1整份md5 {blob['preds']['n_L1_equal']}/{blob['preds']['n_files']} | "
            f"L2 per_frame {blob['preds']['n_L2_per_frame_equal']}/{blob['preds']['n_files']} | "
            f"L3 payload {blob['preds']['n_L3_equal']}/{blob['preds']['n_files']} | "
            f"L4 资产 {blob['preds']['n_L4_equal']}/{blob['preds']['n_files']} | "
            f"out-json 投影 {'OK' if blob['report_projection_equal'] else 'DIFF'}"
        )
    print(f"[尺子1] all_bit_identical={ruler1.get('all_bit_identical')} f1={ruler1.get('source_frame_f1')}")
    print(f"[VERDICT] {result['verdict']}")
    print(f"[SUCCESS] report -> {out_json}")
    print(f"[SUCCESS] report -> {out_md}")
    sys.exit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
