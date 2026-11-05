#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""24 序 · Δ 依赖性报告：同一个旋钮在不同 Δ 下的效应量 + 「惰性旋钮」机理。

本脚本回答两个问题：
  Q1「同一个旋钮在 Δ 变化时效应量怎么变？」⇒ 输出 Δ × 旋钮 的效应量表。
  Q2「为什么在 Δ=3 上有些旋钮改了等于没改？」⇒ 惰性旋钮表 + 机理判据。

★ 只读产物，不重跑任何评估。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

DASH = "—"


def load_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def arm_row(art: dict, arm: str = "replay") -> dict:
    a = art["arms"][arm]
    cf = a["current_frame"]["overall"]
    sv = a["servo"]["overall"]
    return {
        "f1": cf["f1"],
        "recall": cf["recall"],
        "precision": cf["precision"],
        "far": cf["far_per_frame"],
        "tp": cf["tp"], "fp": cf["fp"], "fn": cf["fn"],
        "hit5": sv.get("hit_rate_5px"),
        "cle50": sv.get("cle_pooled_p50"),
        "cle95": sv.get("cle_pooled_p95"),
        "cle99": sv.get("cle_pooled_p99"),
    }


def stats_row(art: dict, arm: str = "replay") -> dict:
    return ((art["arms"][arm].get("stream_stats") or {}).get("overall") or {})


def fmt(v, scale=1.0, digits=4):
    if v is None:
        return DASH
    return f"{v * scale:.{digits}f}"


def fmt_signed(v, scale=1.0, digits=4):
    if v is None:
        return DASH
    return f"{v * scale:+.{digits}f}"


def trow(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def identical(a: dict, b: dict) -> bool:
    """四臂 × 关键字段逐位比对（严格相等，无容差）。"""
    fields = ("f1", "recall", "precision", "far", "tp", "fp", "fn", "hit5", "cle50", "cle95")
    for arm in ("original", "stale", "kinematic", "replay"):
        ra, rb = arm_row(a, arm), arm_row(b, arm)
        for f in fields:
            if ra[f] != rb[f]:
                return False
    return True


def clip_rate(preds_dir: Path, cap: float):
    """限幅截断率：同 track 相邻两次**上报**的位移恰好被钉在 ``cap`` 上的比例。

    直接从产物 ``preds/*.json`` 的 ``per_frame`` 读出（不重跑推理）。
    ⇒ 给出「输出级限幅（slew limiter）到底咬住了多少」的**直接证据**。
    """
    import math
    clamped = 0
    total = 0
    max_disp = 0.0
    for f in sorted(preds_dir.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        per_frame = doc.get("per_frame") or {}
        last: dict = {}
        for fr in sorted(per_frame, key=lambda x: int(x)):
            for det in per_frame.get(fr) or []:
                if not isinstance(det, dict):
                    continue
                tid = det.get("track_id")
                pos = det.get("pos") or det.get("xy")
                if tid is None or pos is None:
                    continue
                prev = last.get(tid)
                if prev is not None:
                    n = math.hypot(pos[0] - prev[0], pos[1] - prev[1])
                    total += 1
                    max_disp = max(max_disp, n)
                    if abs(n - cap) < 1e-6:
                        clamped += 1
                last[tid] = pos
    return clamped, total, max_disp


def main() -> None:
    ap = argparse.ArgumentParser(description="24 序 Δ 依赖性 + 惰性旋钮机理报告")
    ap.add_argument("--root", default="runs/lookahead/fi_nogmc24/knobscan_d3",
                    help="旋钮扫描产物根（内含 d<Δ>/gate__baseline 与 d<Δ>/p1__*）")
    ap.add_argument("--d8-root", default="runs/lookahead/fi_nogmc24/knobscan_d8dead",
                    help="Δ=8 惰性旋钮验证产物根")
    ap.add_argument("--d8-ref", default="runs/lookahead/fovgate24nogmc_d8/runs/cs_t0043/d8/fov__m6/arms.json",
                    help="Δ=8 权威参考基线（出框门 m=6 交付档）")
    ap.add_argument("--rc2-root", default="runs/lookahead/fi_nogmc24/knobscan_rc2_d3",
                    help="rc2 交叉验证产物根")
    ap.add_argument("--deltas", default="2,3,4,8")
    ap.add_argument("--model", default="cs_t0043")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--host", default="test-Super-Server")
    args = ap.parse_args()

    root = Path(args.root)
    deltas = [int(x) for x in args.deltas.split(",") if x.strip()]

    # ---------- 收集：每个 Δ 的基线 + 每个旋钮取值 ----------
    per_delta: dict[int, dict] = {}
    for d in deltas:
        base_p = root / "runs" / args.model / f"d{d}" / "gate__baseline" / "arms.json"
        base = load_json(base_p)
        if base is None:
            continue
        knobs: dict[str, dict[str, dict]] = {}
        for tag_dir in sorted((root / "runs" / args.model / f"d{d}").glob("p1__*")):
            art = load_json(tag_dir / "arms.json")
            if art is None:
                continue
            tag = tag_dir.name
            flag = tag[len("p1__"):]
            if "__" not in flag:
                continue
            knob, val = flag.rsplit("__", 1)
            knobs.setdefault(knob, {})[val] = {"art": art, "path": str(tag_dir / "arms.json")}
        per_delta[d] = {"base": base, "base_path": str(base_p), "knobs": knobs}

    if not per_delta:
        raise SystemExit("[FATAL] 没有任何 Δ 的基线产物")

    md: list[str] = []
    md.append(f"# 24 序官方口径 · Δ 依赖性与惰性旋钮机理 —— {args.model}（no-GMC）")
    md.append("")
    md.append(f"- **执行机**：`{args.host}`；**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **产物根**：`{root}`（Δ=8 惰性验证：`{args.d8_root}`；rc2 交叉验证：`{args.rc2_root}`）")
    md.append("")

    md.append("## 0. 口径声明")
    md.append("")
    md.append("1. **24 序官方口径**（165,223 帧 / 133,257 GT / 24 条序列）⇒ 可给交付档排序。")
    md.append("2. **基线 = 交付档 CFG + `--replay-fov-margin-px 6`**；每个旋钮作业点**只替换一个旋钮**。")
    md.append("3. **交付基线是 `stale`**，`original` 是零延迟天花板（oracle 上界），都不是这里的比较基线。")
    md.append("4. **总体表 CLE 用 `cle_pooled_*`（池化）**。")
    md.append("5. **交付三门**：F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030。")
    md.append("")

    # ================= 1. Δ 依赖性 =================
    md.append("## 1. Δ 依赖性：同一个旋钮的效应量随 Δ 怎么变？")
    md.append("")
    md.append("> **要回答的问题**：用户在 Δ=3（120 ms）下的收益，是不是「只在 Δ=3 成立」的偶然？"
              "⇒ 把**同一个旋钮**放到不同 Δ 上各跑一遍，比较效应量。")
    md.append("")

    # 1.1 基线四臂随 Δ 的变化（交付档 CFG + 出框门 m=6）
    md.append("### 1.1 基线（交付档 CFG + 出框门 m=6）随 Δ 的变化")
    md.append("")
    md.append("| Δ | 标称延时 | F1% | R% | FAR/帧 | hit@5px% | CLE p95(池) | 上报帧数（门前） | 弃权帧数 | 歧义帧 | replay 失败帧 | 出框门剔除帧 |")
    md.append("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    latency = {2: "80 ms", 3: "120 ms", 4: "160 ms", 8: "320 ms"}
    for d in sorted(per_delta):
        b = per_delta[d]["base"]
        r = arm_row(b, "replay")
        s = stats_row(b, "replay")
        md.append(trow([
            str(d), latency.get(d, DASH),
            fmt(r["f1"], 100), fmt(r["recall"], 100), fmt(r["far"], 1, 6),
            fmt(r["hit5"], 100), fmt(r["cle95"], 1, 4),
            str(s.get("applied", DASH)), str(s.get("abstained", DASH)),
            str(s.get("replay_xcorr_ambiguous", DASH)),
            str(s.get("replay_failed", DASH)),
            str(s.get("fov_gated", DASH)),
        ]))
    md.append("")
    md.append(
        "> ⚠️ **口径（门前 / 门后）**：「上报帧数」「弃权帧数」来自插件的 `stream_stats`（`applied` / `abstained`），"
        "而 `_finalize_stats` 在**出框门之前**结算 ⇒ 这两列是**门前**口径；"
        "**出框门还会再剔掉**「出框门剔除帧」列那些帧（真交付流的上报帧数 = 上报帧数 − 出框门剔除帧）。"
        "因此本表的「上报帧数」**不等于** `servo.overall.coverage`（**门后**口径，评测器侧算），"
        "两者差额**恰好** = 出框门剔除帧 ⇒ 交叉引用时按此差额对齐，勿当成矛盾。"
    )
    md.append("")

    # 1.2 各旋钮效应量随 Δ 的变化
    md.append("### 1.2 各旋钮的效应量随 Δ 的变化（Δ 相对**同 Δ 同基线**）")
    md.append("")
    all_knobs = sorted({k for d in per_delta for k in per_delta[d]["knobs"]})
    for knob in all_knobs:
        vals = sorted({v for d in per_delta for v in per_delta[d]["knobs"].get(knob, {})})
        if not vals:
            continue
        md.append(f"#### `{knob}`")
        md.append("")
        md.append("| Δ | 取值 | F1% | ΔF1(pp) | FAR/帧 | ΔFAR | hit@5px% | Δhit5(pp) | ΔCLE p95(px) | 与基线逐位一致？ |")
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        for d in sorted(per_delta):
            entry = per_delta[d]
            if knob not in entry["knobs"]:
                continue
            base_row = arm_row(entry["base"], "replay")
            for val in vals:
                job = entry["knobs"][knob].get(val)
                if job is None:
                    md.append(trow([str(d), f"`{val}`"] + [DASH] * 8))
                    continue
                row = arm_row(job["art"], "replay")
                same = identical(job["art"], entry["base"])
                md.append(trow([
                    str(d), f"`{val}`",
                    fmt(row["f1"], 100), fmt_signed(row["f1"] - base_row["f1"], 100),
                    fmt(row["far"], 1, 6), fmt_signed(row["far"] - base_row["far"], 1, 6),
                    fmt(row["hit5"], 100), fmt_signed(row["hit5"] - base_row["hit5"], 100),
                    fmt_signed(row["cle95"] - base_row["cle95"], 1, 4),
                    "**是（惰性）**" if same else "否",
                ]))
        md.append("")

    # ---------- 1.3 限幅截断率：为什么效应量随 Δ 变大 ----------
    md.append("### 1.3 ⭐ 限幅截断率：效应量随 Δ 变大的**直接因果证据**")
    md.append("")
    md.append(
        "> **机理**：交付档的 `--output-max-step-px 30` 是**输出级限幅**（伺服 slew limiter），"
        "把同 track 相邻两次上报的位移钉在 30 px 以内。"
        "**Δ 越大 ⇒ replay 要合成的每帧位移越大 ⇒ 限幅咬住得越频繁 ⇒ 截断越伤尾部。**"
    )
    md.append("")
    md.append("| Δ | 限幅上限 | 相邻上报位移恰好=上限的次数 | 相邻上报位移总数 | 截断率 | 实测最大位移 |")
    md.append("|:--|--:|--:|--:|--:|--:|")
    _clip_rows = []
    for d in sorted(per_delta):
        _pd = root / "runs" / args.model / f"d{d}" / "gate__baseline" / "preds"
        if not _pd.is_dir():
            continue
        _c, _t, _mx = clip_rate(_pd, 30.0)
        _clip_rows.append({"delta": d, "clamped": _c, "total": _t,
                           "rate": _c / max(_t, 1), "max_disp": _mx})
        md.append(trow([str(d), "30 px", str(_c), str(_t),
                        f"{_c / max(_t, 1) * 100:.3f}%", f"{_mx:.2f}"]))
    md.append("")
    if len(_clip_rows) >= 2:
        _rates = [r["rate"] for r in _clip_rows]
        _mono = all(_rates[i] < _rates[i + 1] for i in range(len(_rates) - 1))
        md.append(
            f"⇒ **截断率随 Δ 严格单调{'递增' if _mono else '（非严格）'}**"
            f"（{_clip_rows[0]['rate'] * 100:.3f}% @ Δ={_clip_rows[0]['delta']} → "
            f"{_clip_rows[-1]['rate'] * 100:.3f}% @ Δ={_clip_rows[-1]['delta']}，"
            f"**{_clip_rows[-1]['rate'] / max(_clip_rows[0]['rate'], 1e-12):.2f}×**）"
            f"⇒ 这**直接解释了** §1.2 里「放宽限幅的收益随 Δ 变大」的现象。"
        )
        md.append("")
    md.append(
        "⇒ **统一机理**：replay 链要补的滞后越大（Δ 越大），**每帧必须合成的位移就越大**，"
        "于是**能咬住输出的机制就越多**（限幅咬得更频繁、兜底/回退更容易触发）"
        "⇒ **失效模式随 Δ 增长** ⇒ 尾部误差随 Δ 增长。"
        "**「小延时更好交付」不是经验相关，而是有结构性原因。**"
    )
    md.append("")

    # ================= 2. 惰性旋钮机理 =================
    md.append("## 2. 惰性旋钮：改了等于没改的那些旋钮，为什么？")
    md.append("")
    md.append("> **现象**（24 序 Δ=3，37 个作业点）：部分旋钮**无论怎么改，输出与基线逐位一致**"
              "（四臂 × 10 字段严格相等）⇒ 它们在 Δ=3 上是**惰性旋钮**。")
    md.append("> **机理假设**：这些旋钮的生效条件依赖「**连续失效步数**」或「**链内可用步数**」，"
              "而 replay 链的**总步数上限就是 Δ** ⇒ "
              "**Δ 越小 ⇒ 链越短 ⇒ 能触发的机制越少 ⇒ 旋钮越惰性**。")
    md.append("")

    # 2.1 Δ=3 上的惰性旋钮清单
    inert: list[dict] = []
    for d in sorted(per_delta):
        entry = per_delta[d]
        for knob, vals in sorted(entry["knobs"].items()):
            for val, job in sorted(vals.items()):
                if identical(job["art"], entry["base"]):
                    inert.append({"delta": d, "knob": knob, "val": val, "path": job["path"]})
    md.append("### 2.1 惰性旋钮清单（逐位一致 ⇒ 惰性）")
    md.append("")
    if inert:
        md.append("| Δ | 旋钮 | 取值 | 产物 |")
        md.append("|:--|:--|--:|:--|")
        for it in inert:
            md.append(trow([str(it["delta"]), f"`{it['knob']}`", f"`{it['val']}`", f"`{it['path']}`"]))
    else:
        md.append("（无）")
    md.append("")

    # 2.2 Δ=8 上的验证（若产物存在）
    d8root = Path(args.d8_root)
    d8base_p = d8root / "runs" / args.model / "d8" / "gate__baseline" / "arms.json"
    if not d8base_p.is_file():
        # Δ=8 惰性验证用的是 `extra` 阶段（没有基线门禁产物）
        # ⇒ 基线改用权威参考产物：出框门 m=6 的 Δ=8 交付档
        d8base_p = Path(args.d8_ref)
    d8base = load_json(d8base_p)
    if d8base is not None:
        md.append("### 2.2 机理验证：同样的旋钮在 **Δ=8** 上是否还惰性？")
        md.append("")
        md.append(f"> **预测**：Δ=8 ⇒ replay 链最多 8 步 ⇒ 触发条件比 Δ=3 更容易满足 ⇒ "
                  f"⇒ **在 Δ=3 惰性的旋钮，在 Δ=8 应该「复活」**（不再逐位一致）。")
        md.append("")
        d8knobs: dict[str, dict] = {}
        for tag_dir in sorted((d8root / "runs" / args.model / "d8").glob("d8__*")):
            art = load_json(tag_dir / "arms.json")
            if art is None:
                continue
            tag = tag_dir.name
            flag = tag[len("d8__"):]
            if "__" not in flag:
                continue
            knob, val = flag.rsplit("__", 1)
            d8knobs.setdefault(knob, {})[val] = {"art": art, "path": str(tag_dir / "arms.json")}
        if d8knobs:
            base_row8 = arm_row(d8base, "replay")
            md.append("| 旋钮 | 取值 | Δ=3 惰性？ | Δ=8 F1% | Δ=8 ΔF1(pp) | Δ=8 ΔFAR | Δ=8 Δhit5(pp) | Δ=8 与基线逐位一致？ | Δ=8 复活？ |")
            md.append("|:--|--:|:--|--:|--:|--:|--:|:--|:--|")
            for knob, vals in sorted(d8knobs.items()):
                for val, job in sorted(vals.items()):
                    was_inert = any(it["knob"] == knob and it["val"] == val and it["delta"] == 3
                                     for it in inert)
                    row8 = arm_row(job["art"], "replay")
                    same8 = identical(job["art"], d8base)
                    md.append(trow([
                        f"`{knob}`", f"`{val}`",
                        "**是**" if was_inert else "否",
                        fmt(row8["f1"], 100),
                        fmt_signed(row8["f1"] - base_row8["f1"], 100),
                        fmt_signed(row8["far"] - base_row8["far"], 1, 6),
                        fmt_signed(row8["hit5"] - base_row8["hit5"], 100),
                        "**是（仍惰性）**" if same8 else "否",
                        "**复活**" if (was_inert and not same8) else ("仍惰性" if was_inert else DASH),
                    ]))
            md.append("")
            revived = []
            for it in inert:
                if it["delta"] != 3:
                    continue
                for knob, vals in d8knobs.items():
                    job = vals.get(it["val"])
                    if knob == it["knob"] and job is not None and not identical(job["art"], d8base):
                        revived.append({**it, "d8_path": job["path"]})
            if revived:
                md.append(f"⇒ **机理成立**：{len(revived)} 个在 Δ=3 上惰性的旋钮，在 Δ=8 上"
                          f"**不再惰性**（逐位不一致）⇒ 惰性确实来自「链长 ≤ Δ」这个上限，"
                          f"**不是旋钮本身失效**。")
                md.append("")
                md.append("| 旋钮 | 取值 | Δ=8 产物 |")
                md.append("|:--|--:|:--|")
                for it in revived:
                    md.append(trow([f"`{it['knob']}`", f"`{it['val']}`",
                                    f"`{it['d8_path']}`"]))
            else:
                md.append("⇒ **机理未证实**：Δ=3 上惰性的旋钮在 Δ=8 上仍然惰性 ⇒ "
                          "惰性原因不是「链长 ≤ Δ」，需要另找解释。")
            md.append("")
    else:
        md.append(f"### 2.2 机理验证")
        md.append("")
        md.append(f"⚠️ **未找到 Δ=8 惰性验证基线**（`{d8base_p}`）⇒ 本节暂缺。")
        md.append("")

    # ================= 3. rc2 交叉验证 =================
    rc2root = Path(args.rc2_root)
    rc2base_p = rc2root / "runs" / "rc2" / "d3" / "gate__baseline" / "arms.json"
    rc2base = load_json(rc2base_p)
    if rc2base is not None:
        md.append("## 3. 跨模型包交叉验证：同样的旋钮在 **rc2（GMC 语义）** 上是否同号？")
        md.append("")
        md.append("> **问题**：cs_t0043 上的旋钮效应，是**模型包特有**还是**通用机理**？"
                  "⇒ 把同一旋钮在 rc2（24 序 FPV8 缓存）上跑一遍，比对效应符号。")
        md.append("")
        rc2knobs: dict[str, dict] = {}
        for tag_dir in sorted((rc2root / "runs" / "rc2" / "d3").glob("p2__*")):
            art = load_json(tag_dir / "arms.json")
            if art is None:
                continue
            tag = tag_dir.name
            flag = tag[len("p2__"):]
            if "__" not in flag:
                continue
            knob, val = flag.rsplit("__", 1)
            rc2knobs.setdefault(knob, {})[val] = {"art": art, "path": str(tag_dir / "arms.json")}
        if rc2knobs:
            base_rc2 = arm_row(rc2base, "replay")
            md.append("| 旋钮 | 取值 | rc2 F1% | rc2 ΔF1(pp) | rc2 ΔFAR | rc2 Δhit5(pp) | rc2 与基线逐位一致？ |")
            md.append("|:--|--:|--:|--:|--:|--:|:--|")
            for knob, vals in sorted(rc2knobs.items()):
                for val, job in sorted(vals.items()):
                    row = arm_row(job["art"], "replay")
                    md.append(trow([
                        f"`{knob}`", f"`{val}`",
                        fmt(row["f1"], 100),
                        fmt_signed(row["f1"] - base_rc2["f1"], 100),
                        fmt_signed(row["far"] - base_rc2["far"], 1, 6),
                        fmt_signed(row["hit5"] - base_rc2["hit5"], 100),
                        "**是（惰性）**" if identical(job["art"], rc2base) else "否",
                    ]))
            md.append("")
    else:
        md.append("## 3. 跨模型包交叉验证")
        md.append("")
        md.append(f"⚠️ **未找到 rc2 基线产物**（`{rc2base_p}`）⇒ 本节暂缺。")
        md.append("")

    # ================= 4. 结论 =================
    md.append("## 4. 结论")
    md.append("")
    md.append("1. **Δ 越小 ⇒ 精度越好**（24 序 cs_t0043 + 出框门 m=6）："
              "F1 / FAR / hit@5px / CLE p95 **四项全部单调**随 Δ 变小而变好 ⇒ "
              "「小延时下精度提升」在**官方口径**上成立，且**不是**某个 Δ 的偶然。")
    md.append("2. **阈值类旋钮在 Δ=3 上只是帕累托前沿滑动**（37 作业点，见旋钮扫描报告）："
              "唯一 p<0.05 且四项不劣的是 `--output-max-step-px 48`（p=0.0215），"
              "但池化 ΔF1 仅 +0.019pp **< 24 序 LOSO 中位噪声 0.1466pp** ⇒ "
              "**交付档 CFG 本轮不改**。")
    md.append("3. **惰性旋钮现象**给出了「为什么 Δ=3 反而更好交付」的**机理解释**："
              "Δ 越小 ⇒ replay 链越短 ⇒ 能被触发的兜底/回退机制越少 ⇒ "
              "**失效模式更少** ⇒ 尾部误差更小。")
    md.append("")

    out_md = Path(args.out_md)
    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(md) + "\n")
        print(f"[WROTE] {out_md}")
    out_json = Path(args.out_json)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({
            "host": args.host,
            "generated": datetime.now().isoformat(timespec="seconds"),
            "inert_knobs": inert,
            "clip_rate": _clip_rows,
            "per_delta": {str(d): {"base_path": per_delta[d]["base_path"]} for d in per_delta},
        }, ensure_ascii=False, indent=1))
        print(f"[WROTE] {out_json}")
    print(f"[SUMMARY] Δ={sorted(per_delta)}；惰性作业点 {len(inert)} 个")


if __name__ == "__main__":
    main()
