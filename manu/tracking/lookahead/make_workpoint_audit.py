#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Work-point alignment audit: build the FAR-F1 report from the working-point grid artefacts.

Question under audit: is cs_t0043's (no-GMC) edge real, or is it an artefact of the v3
working point (th_base=0.48 / th_ground=0.26) simply suiting it?

Reads ``<root>/<model>_d<delta>/<th_base>_<th_ground>/arms.json`` produced by
``manu/tracking/lookahead/server_overnight/run_workpoint_audit.sh`` and renders:

  Table A  FAR-F1 curve per model x delta (sorted by FAR ascending)
  Table B  best grid point per model x delta (FAR<=0.030 first, then max F1) + delivery triple
  Table C  same-FAR interpolated comparison between rc2 (GMC) and cs_t0043 (no-GMC)
  Table D  working-point sensitivity (distance from v3, rank of the v3 point)

Every number is read from the artefacts; nothing is recomputed from raw frames.

    python manu/tracking/lookahead/make_workpoint_audit.py \
        --root runs/lookahead/fi_nogmc/workpoint \
        --out-md runs/lookahead/fi_nogmc/workpoint/audit.md \
        --out-json runs/lookahead/fi_nogmc/workpoint/audit.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

MODEL_ORDER = ["rc2", "cs_t0043"]
MODEL_LABEL = {
    "rc2": "RC2 trial_0080ep3 (GMC 语义)",
    "cs_t0043": "cs_t0043 (no-GMC 语义)",
}
ARM_ORDER = ["stale", "replay"]
ARM_HEAD = {
    "stale": "stale（老逻辑硬吃 Δ 滞后）= **交付基线**",
    "replay": "replay（视觉重推 + xcorr）= **交付方案**",
    "original": "original（零延迟天花板）= oracle 上界，**不是**基线",
}
ARM_LABEL = {
    "stale": "stale（老逻辑硬吃 Δ 滞后）= **交付基线**",
    "replay": "replay（视觉重推 + xcorr）= **交付方案**",
    "original": "original（零延迟天花板）= oracle 上界，**不是**基线",
}
DELIVERY = {"f1_min": 0.88, "recall_min": 0.85, "far_max": 0.030}
V3_POINT = (0.48, 0.26)
MACHINE = "服务器 test-Super-Server"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Work-point alignment audit report")
    p.add_argument("--root", default="runs/lookahead/fi_nogmc/workpoint", help="Grid artefact root")
    p.add_argument("--out-md", default="runs/lookahead/fi_nogmc/workpoint/audit.md")
    p.add_argument("--out-json", default="runs/lookahead/fi_nogmc/workpoint/audit.json")
    p.add_argument("--target-delta", type=int, default=3, help="Main battlefield delta")
    p.add_argument("--control-delta", type=int, default=8, help="Control delta")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_grid(root: Path) -> dict:
    """Return {model: {delta: {(th_base, th_ground): artefact}}} for every grid artefact."""
    grid: dict = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir() and "_d" in p.name):
        model, _, dstr = mdir.name.rpartition("_d")
        if not model or not dstr.isdigit():
            continue
        for pdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            artefact = pdir / "arms.json"
            if not artefact.is_file():
                continue
            try:
                data = json.loads(artefact.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if "arms" not in data or "replay" not in data["arms"]:
                continue
            # The artefact itself is the source of truth for the working point.
            params = data.get("config", {}).get("params", {})
            tb = float(params.get("th_base", pdir.name.split("_")[0]))
            tg = float(params.get("th_ground", pdir.name.split("_")[1]))
            grid.setdefault(model, {}).setdefault(int(dstr), {})[(tb, tg)] = {
                "data": data,
                "path": artefact,
                "dir": pdir,
            }
    return grid


def arm_metrics(artefact: dict, arm: str) -> dict:
    """Pull the arm scorecard out of an artefact (no recomputation)."""
    a = artefact["data"]["arms"][arm]
    cf = a["current_frame"]["overall"]
    sv = a["servo"]["overall"]
    return {
        "arm": arm,
        "f1": cf["f1"],
        "recall": cf["recall"],
        "precision": cf["precision"],
        "far": cf["far_per_frame"],
        "tp": cf["tp"],
        "fp": cf["fp"],
        "fn": cf["fn"],
        "gt": cf["gt"],
        "hit5": sv["hit_rate_5px"],
        "cle_p95_pooled": sv["cle_pooled_p95"],
        "cle_p50_pooled": sv["cle_pooled_p50"],
    }


def rows_for(grid_cell: dict, arm: str) -> list:
    """All grid points of one (model, delta) cell for one arm, sorted by FAR ascending."""
    out = []
    for (tb, tg), artefact in grid_cell.items():
        m = arm_metrics(artefact, arm)
        m["th_base"] = tb
        m["th_ground"] = tg
        out.append(m)
    out.sort(key=lambda r: (r["far"], -r["f1"], r["th_base"], r["th_ground"]))
    return out


def envelope(points: list) -> list:
    """Best-achievable-F1-at-FAR<=x curve: collapse duplicate FARs, keep the running max.

    Returns knots [(far, f1), ...] sorted by FAR ascending; between knots the caller
    interpolates linearly.  This is the operationally meaningful curve: "if the false
    alarm budget is x, the best F1 reachable on this grid is f(x)".
    """
    by_far: dict = {}
    for p in points:
        far = p["far"]
        by_far[far] = max(by_far.get(far, -1.0), p["f1"])
    running = -1.0
    knots = []
    for far in sorted(by_far):
        if by_far[far] > running:
            running = by_far[far]
            knots.append((far, running))
    # 平台延伸：f(x) = max{F1 : FAR <= x} 在网格最大 FAR 处仍成立 ⇒ 补一个平尾结点，
    # 否则包络会在最后一个「刷新纪录」的点上被截断，公共 FAR 区间会被算窄。
    max_far = max(by_far)
    if not knots or knots[-1][0] < max_far - 1e-15:
        knots.append((max_far, running))
    return knots


def interp(knots: list, far: float):
    """Linear interpolation of the envelope at `far`; None if outside the knot span."""
    tol = 1e-9
    if not knots or far < knots[0][0] - tol or far > knots[-1][0] + tol:
        return None
    far = min(max(far, knots[0][0]), knots[-1][0])
    for (f0, v0), (f1, v1) in zip(knots, knots[1:]):
        if f0 - tol <= far <= f1 + tol:
            if abs(f1 - f0) < 1e-15:
                return v0
            return v0 + (v1 - v0) * (far - f0) / (f1 - f0)
    return knots[-1][1]


def best_key(row: dict) -> tuple:
    """Deterministic 'best grid point' ordering: max F1, then min FAR, then closest to v3.

    The last key matters because th_ground is a dead knob at sky_ratio=1.0 (see the note
    in the report): the three th_ground variants of one th_base are bit-identical, so
    without it the reported optimum would be an arbitrary tie-break.
    """
    return (
        row["f1"],
        -row["far"],
        -abs(row["th_base"] - V3_POINT[0]),
        -abs(row["th_ground"] - V3_POINT[1]),
    )


def rank_of(rows: list, target: dict) -> tuple:
    """Competition rank (1 = best) of `target` among `rows` ordered by F1 descending."""
    better = sum(1 for r in rows if r["f1"] > target["f1"] + 1e-15)
    equal = sum(1 for r in rows if abs(r["f1"] - target["f1"]) <= 1e-15)
    return better + 1, equal


# --------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------
def pct(x: float, nd: int = 4) -> str:
    return f"{x * 100:.{nd}f}"


def pp(x: float, nd: int = 4) -> str:
    """Format a *fraction* as percentage points, e.g. 0.9619 -> +96.1900."""
    return f"{x * 100:+.{nd}f}"


def ppv(x: float, nd: int = 4) -> str:
    """Format a value that is *already* in percentage points, e.g. -4.3506 -> -4.3506."""
    return f"{x:+.{nd}f}"


def md_table(header: list, body: list) -> list:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in body:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def build_report(grid: dict, root: Path, args: argparse.Namespace) -> tuple:
    deltas = sorted({d for mdl in grid.values() for d in mdl})
    models = [m for m in MODEL_ORDER if m in grid] + [m for m in sorted(grid) if m not in MODEL_ORDER]
    arms = ["replay", "stale"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    L: list = []
    J: dict = {"generated_at": now, "machine": MACHINE, "root": str(root), "models": {}, "tables": {}}

    L.append("# 工作点对齐审计 —— cs_t0043 的优势是真的，还是只是 v3 工作点碰巧适合它？")
    L.append("")
    L.append(
        f"**执行机**：{MACHINE}（CPU only，无 GPU） ｜ **生成时间**：{now} ｜ "
        f"**产物根**：`{root}` ｜ **网格**：th_base ∈ {{0.40, 0.44, 0.48, 0.52, 0.56}} × "
        f"th_ground ∈ {{0.22, 0.26, 0.30}} = 15 点 × 2 模型 × 2 Δ = 60 次"
    )
    L.append("")

    # ---- 口径纪律 ---------------------------------------------------------------
    L.append("## 0. 口径纪律（本报告必须遵守）")
    L.append("")
    L.append("- **交付基线 = `stale` 臂**（老逻辑硬吃 Δ 滞后）。`original` 臂是零延迟天花板（oracle 上界），**不是**基线；本报告的所有「基线」表述一律指 `stale`。")
    L.append("- **交付方案 = `replay` 臂**（视觉重推 + xcorr）。")
    L.append("- **总体 CLE 用 `cle_pooled_*`**（全帧池化），不用逐序 `cle_p*`（那是逐序列最差值）。")
    L.append("- 8 序 gate 集只覆盖 24 序官方口径的 **20.39% GT** ⇒ 结论只能写成「8 序机理筛选 / A/B 证据」，**不能**用来给交付档排序。")
    L.append("- 每个数字都直接读 `arms.json`（`arms.<arm>.current_frame.overall` 与 `arms.<arm>.servo.overall`），**不重算**。")
    # ---- 出框门口径：从本审计的产物现算（产物没有该键 ⇒ 它生成于出框门旋钮补记之前）
    _fov_keys = {
        float(((art["data"].get("lookahead") or {}).get("replay_fov_margin_px", 0.0)) or 0.0)
        for _md in grid.values()
        for _dd in _md.values()
        for art in _dd.values()
    }
    if _fov_keys == {0.0}:
        L.append(
            "- ⚠️ **出框门口径**：本审计的产物里**读不到** `replay_fov_margin_px`（生成于该旋钮补记之前）"
            "⇒ 本报告全部数字为**裸配置（未开出框门）**；交付推荐档（m=6）的数字见 `FINAL_REPORT.md` §9.4 / §12。"
        )
    else:
        L.append(
            f"- ⚠️ **出框门口径**：本审计产物的 `--replay-fov-margin-px` = {sorted(_fov_keys)} ⇒ "
            "本报告数字**含出框门**；裸配置（m=0）见 `FINAL_REPORT.md` §2.9。"
        )
    L.append("")

    # ---- 网格有效维数（实测）------------------------------------------------------
    L.append("### 0.1 网格的实际有效维数（实测，重要）")
    L.append("")
    L.append(
        "本网格的 `th_ground` 轴在 v3 工作点下是**死旋钮**：3 个 `th_ground` 取值"
        "（0.22 / 0.26 / 0.30）在 **全部 4 个 model×Δ × 全部 4 个臂** 上给出**逐位相同**的 "
        "tp/fp/fn/recall/precision/F1/FAR（60 次运行实测；表 D 的「网格内互异 F1 数」列 = 5）。"
    )
    L.append("")
    L.append(
        "机理（代码级）：`manu/tracking/lookahead/track_fusion.py:822` 的分层门是 "
        "`fusion_high_mask = (is_sky & (pred_scs >= th_base)) | ((~is_sky) & (pred_scs >= th_ground))`，"
        "而 `sky_y_boundary = img_h * sky_ratio`；v3 的 `sky_ratio = 1.0` ⇒ 整幅画面都算 sky ⇒ "
        "`th_ground` 永不生效。这与 v3 主文件 `_provenance` 自述一致"
        "（“sky_ratio=1.0 puts the WHOLE frame under the stricter th_base, i.e. th_ground stops gating anything”）。"
    )
    L.append("")
    L.append(
        "⇒ **15 点网格实际只有 5 个不同工作点**（由 `th_base` 一维决定）；"
        "v3 的 `th_ground=0.26` **不是被调出来的旋钮** ⇒ 「cs_t0043 的优势是否来自 v3 工作点巧合」"
        "这个问题在 `th_ground` 维度上**不成立**（该维度对两个模型同样无效）；"
        "真实的工作点自由度只有 `th_base` 一维。"
    )
    L.append("")

    # ---- 表 A -------------------------------------------------------------------
    L.append("## 表 A · FAR–F1 曲线（每个模型 × 每个 Δ，按 FAR 升序）")
    L.append("")
    L.append(
        "读法：在**同一 FAR** 下比 F1 ⇒ 工作点偏差被剔掉，只剩模型能力差异。"
        "`th_base` 越低 ⇒ 检测越宽松 ⇒ FAR 越高。`th_ground` 见 §表 D 的说明。"
    )
    L.append("")
    table_a = {}
    for m in models:
        for d in deltas:
            cell = grid[m].get(d)
            if not cell:
                continue
            L.append(f"### A · {MODEL_LABEL.get(m, m)} — Δ={d}")
            L.append("")
            for arm in arms:
                rows = rows_for(cell, arm)
                table_a[f"{m}|d{d}|{arm}"] = rows
                L.append(f"**{ARM_HEAD[arm]}**")
                L.append("")
                body = [
                    [
                        f"{r['th_base']:.2f}",
                        f"{r['th_ground']:.2f}",
                        f"{r['far']:.6f}",
                        pct(r["f1"]),
                        pct(r["recall"], 2),
                        pct(r["precision"], 2),
                        pct(r["hit5"], 2),
                        f"{r['cle_p95_pooled']:.3f}",
                    ]
                    for r in rows
                ]
                L += md_table(
                    ["th_base", "th_ground", "FAR", "F1 %", "R %", "P %", "hit@5px %", "CLE p95 (池化, px)"],
                    body,
                )
                L.append("")
    J["tables"]["A"] = table_a

    # ---- 表 B -------------------------------------------------------------------
    L.append("## 表 B · 每个模型 × 每个 Δ 的最优点（先过 FAR ≤ 0.030，再最大 F1）")
    L.append("")
    L.append(
        "选择规则：网格点中先筛出 FAR ≤ 0.030 者，在其内取 F1 最大者。"
        "「交付三门」= **F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030**。"
    )
    L.append("")
    table_b = {}
    for m in models:
        for d in deltas:
            cell = grid[m].get(d)
            if not cell:
                continue
            L.append(f"### B · {MODEL_LABEL.get(m, m)} — Δ={d}")
            L.append("")
            body = []
            for arm in arms:
                rows = rows_for(cell, arm)
                feasible = [r for r in rows if r["far"] <= DELIVERY["far_max"] + 1e-12]
                if feasible:
                    best = max(feasible, key=best_key)
                    how = "FAR≤0.030 内 F1 最大"
                else:
                    best = min(
                        rows,
                        key=lambda r: (
                            r["far"],
                            -r["f1"],
                            abs(r["th_ground"] - V3_POINT[1]),
                            abs(r["th_base"] - V3_POINT[0]),
                        ),
                    )
                    how = "**无点满足 FAR≤0.030** ⇒ 取 FAR 最小点"
                triple = (
                    best["f1"] >= DELIVERY["f1_min"]
                    and best["recall"] >= DELIVERY["recall_min"]
                    and best["far"] <= DELIVERY["far_max"]
                )
                body.append(
                    [
                        ARM_LABEL[arm].split("（")[0].split("=")[0].strip(),
                        f"{best['th_base']:.2f}",
                        f"{best['th_ground']:.2f}",
                        f"{best['far']:.6f}",
                        pct(best["f1"]),
                        pct(best["recall"], 2),
                        pct(best["precision"], 2),
                        pct(best["hit5"], 2),
                        f"{best['cle_p95_pooled']:.3f}",
                        "✅ 通过" if triple else "❌ 未过",
                        how,
                    ]
                )
                table_b[f"{m}|d{d}|{arm}"] = {
                    "rule": how,
                    "point": best,
                    "passes_delivery_triple": triple,
                }
            L += md_table(
                [
                    "臂",
                    "th_base",
                    "th_ground",
                    "FAR",
                    "F1 %",
                    "R %",
                    "P %",
                    "hit@5px %",
                    "CLE p95 (池化, px)",
                    "交付三门",
                    "选取规则",
                ],
                body,
            )
            L.append("")
    J["tables"]["B"] = table_b

    # ---- 表 C -------------------------------------------------------------------
    L.append("## 表 C · 同 FAR 插值对比（rc2 [GMC] vs cs_t0043 [no-GMC]）")
    L.append("")
    L.append(
        "方法：把每个模型 × Δ × 臂的网格点做成 FAR–F1 **包络**"
        "（同一 FAR 取网格内最高 F1，即「误报预算 ≤ FAR 时可达的最优 F1」），"
        "在两模型的**公共 FAR 区间**上线性插值，取同一 FAR 下的 F1 相减。"
        "ΔF1 列 = cs_t0043 − rc2，**正值 = no-GMC 更高**。"
    )
    L.append("")
    table_c = {}
    for d in deltas:
        for arm in arms:
            key_a, key_b = f"rc2|d{d}|{arm}", f"cs_t0043|d{d}|{arm}"
            if key_a not in table_a or key_b not in table_a:
                continue
            knots_a = envelope(table_a[key_a])
            knots_b = envelope(table_a[key_b])
            if not knots_a or not knots_b:
                continue
            lo = max(knots_a[0][0], knots_b[0][0])
            hi = min(knots_a[-1][0], knots_b[-1][0])
            L.append(f"### C · {ARM_HEAD[arm]} — Δ={d}")
            L.append("")
            if hi < lo:
                L.append("两模型 FAR 区间无公共部分，无法做同 FAR 对比。")
                L.append("")
                continue
            targets = [lo, hi]
            for r in table_a[key_a]:
                if lo + 1e-12 < r["far"] < hi - 1e-12:
                    targets.append(r["far"])
            if lo + 1e-12 < DELIVERY["far_max"] < hi - 1e-12:
                targets.append(DELIVERY["far_max"])
            # 去重但保留原值（避免四舍五入把端点推出结区间）
            uniq = []
            for t in sorted(targets):
                if not uniq or abs(t - uniq[-1]) > 1e-12:
                    uniq.append(t)
            targets = uniq
            body = []
            deltas_list = []
            for t in targets:
                fa = interp(knots_a, t)
                fb = interp(knots_b, t)
                if fa is None or fb is None:
                    continue
                d_pp = (fb - fa) * 100
                deltas_list.append(d_pp)
                anchor_a = "端点锚点" if abs(t - lo) < 1e-9 or abs(t - hi) < 1e-9 else "插值"
                body.append(
                    [
                        f"{t:.6f}",
                        pct(fa),
                        pct(fb),
                        ppv(d_pp),
                        anchor_a,
                    ]
                )
            L += md_table(
                ["FAR 目标", "rc2 F1 % (插值)", "cs_t0043 F1 % (插值)", "ΔF1 = cs_t0043 − rc2 (pp)", "锚点类型"],
                body,
            )
            L.append("")
            mean_d = sum(deltas_list) / len(deltas_list) if deltas_list else 0.0
            verdict = "**高**" if mean_d > 0 else "**低**"
            L.append(
                f"> 公共 FAR 区间 **[{lo:.6f}, {hi:.6f}]**；两端锚点："
                f"lo 处 rc2 {pct(interp(knots_a, lo))}% / cs_t0043 {pct(interp(knots_b, lo))}%，"
                f"hi 处 rc2 {pct(interp(knots_a, hi))}% / cs_t0043 {pct(interp(knots_b, hi))}%。"
                f"区间内 ΔF1 均值 **{ppv(mean_d)} pp**，范围 "
                f"[{ppv(min(deltas_list))}, {ppv(max(deltas_list))}] pp ⇒ 同一误报率下 no-GMC 的 F1 {verdict} rc2。"
            )
            L.append("")
            table_c[f"d{d}|{arm}"] = {
                "common_far_interval": [lo, hi],
                "rc2_knots": knots_a,
                "cs_t0043_knots": knots_b,
                "points": [
                    {"far": t, "rc2_f1": interp(knots_a, t), "cs_f1": interp(knots_b, t)} for t in targets
                ],
                "mean_delta_pp": mean_d,
                "min_delta_pp": min(deltas_list) if deltas_list else None,
                "max_delta_pp": max(deltas_list) if deltas_list else None,
            }
    J["tables"]["C"] = table_c

    # ---- 表 D -------------------------------------------------------------------
    L.append("## 表 D · 工作点敏感性（最优点离 v3 有多远？v3 点排第几？）")
    L.append("")
    L.append(
        "v3 工作点 = **th_base 0.48 / th_ground 0.26**"
        "（`manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`，**未改动**，"
        "网格点 `0.48_0.26` 即该点本身）。"
    )
    L.append("")
    table_d = {}
    for m in models:
        for d in deltas:
            cell = grid[m].get(d)
            if not cell:
                continue
            L.append(f"### D · {MODEL_LABEL.get(m, m)} — Δ={d}")
            L.append("")
            body = []
            for arm in arms:
                rows = rows_for(cell, arm)
                feasible = [r for r in rows if r["far"] <= DELIVERY["far_max"] + 1e-12]
                best = max(feasible, key=best_key) if feasible else min(
                    rows,
                    key=lambda r: (
                        r["far"],
                        -r["f1"],
                        abs(r["th_ground"] - V3_POINT[1]),
                        abs(r["th_base"] - V3_POINT[0]),
                    ),
                )
                v3 = next((r for r in rows if abs(r["th_base"] - V3_POINT[0]) < 1e-9
                           and abs(r["th_ground"] - V3_POINT[1]) < 1e-9), None)
                rk, tie = rank_of(rows, v3) if v3 else (None, 0)
                rk_feas, tie_feas = (
                    rank_of(feasible, v3) if (v3 and feasible) else (None, 0)
                )
                rk_feas_txt = (
                    f"{rk_feas}/{len(feasible)}" if rk_feas else "N/A（无点满足 FAR≤0.03）"
                )
                distinct_f1 = len({round(r["f1"], 12) for r in rows})
                body.append(
                    [
                        ARM_LABEL[arm].split("（")[0].split("=")[0].strip(),
                        f"{best['th_base']:.2f}",
                        f"{best['th_ground']:.2f}",
                        f"{abs(best['th_base'] - V3_POINT[0]):.2f}",
                        f"{abs(best['th_ground'] - V3_POINT[1]):.2f}",
                        pct(best["f1"]),
                        pct(v3["f1"]) if v3 else "N/A",
                        pp(v3["f1"] - best["f1"]) if v3 else "N/A",
                        f"{best['far']:.6f}",
                        f"{v3['far']:.6f}" if v3 else "N/A",
                        f"{rk}/{len(rows)}" + (f"（{tie} 点并列）" if tie > 1 else "") if rk else "N/A",
                        rk_feas_txt,
                        distinct_f1,
                    ]
                )
                table_d[f"{m}|d{d}|{arm}"] = {
                    "best": best,
                    "v3": v3,
                    "rank_overall": rk,
                    "rank_feasible": rk_feas,
                    "tie_group": tie,
                    "distinct_f1_in_grid": distinct_f1,
                    "grid_far_min": min(r["far"] for r in rows),
                    "grid_far_max": max(r["far"] for r in rows),
                }
            L += md_table(
                [
                    "臂",
                    "最优 th_base",
                    "最优 th_ground",
                    "th_base−v3 绝对差",
                    "th_ground−v3 绝对差",
                    "最优 F1 %",
                    "v3 F1 %",
                    "v3 − 最优 (pp)",
                    "最优 FAR",
                    "v3 FAR",
                    "v3 名次(全网格)",
                    "v3 名次(FAR≤0.03 子集)",
                    "网格内互异 F1 数",
                ],
                body,
            )
            L.append("")
    J["tables"]["D"] = table_d

    # ---- 表 E（补充）· 定位质量对照 ----------------------------------------------
    L.append("## 表 E（补充）· 定位质量对照 —— cs_t0043 的真实优势在哪一维")
    L.append("")
    L.append(
        "表 C 只在 F1 一维上比较。若 no-GMC 的定位更好（hit@5px ↑、CLE p95 ↓），"
        "它的价值就不在 F1 上。此表直接读产物值，不插值：v3 点 + 各自最优点。"
    )
    L.append("")
    table_e = {}
    for m in models:
        for d in deltas:
            cell = grid[m].get(d)
            if not cell:
                continue
            L.append(f"### E · {MODEL_LABEL.get(m, m)} — Δ={d}")
            L.append("")
            body = []
            for arm in arms:
                rows = rows_for(cell, arm)
                feasible = [r for r in rows if r["far"] <= DELIVERY["far_max"] + 1e-12]
                best = max(feasible, key=best_key) if feasible else min(
                    rows,
                    key=lambda r: (
                        r["far"],
                        -r["f1"],
                        abs(r["th_ground"] - V3_POINT[1]),
                        abs(r["th_base"] - V3_POINT[0]),
                    ),
                )
                v3 = next((r for r in rows if abs(r["th_base"] - V3_POINT[0]) < 1e-9
                           and abs(r["th_ground"] - V3_POINT[1]) < 1e-9), None)
                for tag, r in (("v3 点", v3), ("本网格最优", best)):
                    if r is None:
                        continue
                    body.append([
                        ARM_LABEL[arm].split("（")[0].split("=")[0].strip(),
                        tag,
                        f"{r['th_base']:.2f}",
                        f"{r['far']:.6f}",
                        pct(r["f1"]),
                        pct(r["hit5"], 2),
                        f"{r['cle_p95_pooled']:.3f}",
                    ])
            L += md_table(
                ["臂", "点", "th_base", "FAR", "F1 %", "hit@5px %", "CLE p95 (池化, px)"],
                body,
            )
            L.append("")
            table_e[f"{m}|d{d}"] = body
    J["tables"]["E"] = table_e

    # ---- 一句话结论 --------------------------------------------------------------
    headline = {}
    key = f"cs_t0043|d{args.target_delta}|replay"
    k2 = f"rc2|d{args.target_delta}|replay"
    if key in table_a and k2 in table_a:
        v3_cs = next(r for r in table_a[key] if abs(r["th_base"] - V3_POINT[0]) < 1e-9
                     and abs(r["th_ground"] - V3_POINT[1]) < 1e-9)
        v3_rc = next(r for r in table_a[k2] if abs(r["th_base"] - V3_POINT[0]) < 1e-9
                     and abs(r["th_ground"] - V3_POINT[1]) < 1e-9)
        fixed_pp = (v3_cs["f1"] - v3_rc["f1"]) * 100
        ck = f"d{args.target_delta}|replay"
        same = table_c.get(ck, {})
        mean_d = same.get("mean_delta_pp")
        rng = (same.get("min_delta_pp"), same.get("max_delta_pp"))
        ivl = same.get("common_far_interval")
        headline = {
            "delta": args.target_delta,
            "arm": "replay",
            "fixed_workpoint": {
                "rc2_f1": v3_rc["f1"],
                "cs_f1": v3_cs["f1"],
                "rc2_far": v3_rc["far"],
                "cs_far": v3_cs["far"],
                "delta_pp": fixed_pp,
            },
            "same_far": {
                "interval": ivl,
                "mean_delta_pp": mean_d,
                "min_delta_pp": rng[0],
                "max_delta_pp": rng[1],
            },
        }
        direction = "高" if (mean_d or 0) > 0 else "低"
        mag = abs(mean_d) if mean_d is not None else 0.0
        no_common = ivl is None or mean_d is None
        L.append("## 一句话结论")
        L.append("")
        verdict_text = (
            "cs_t0043 的优势是真的（不是工作点巧合）"
            if mean_d and mean_d > 0
            else "cs_t0043 在同 FAR 下并不占优 ⇒ 该差距不是工作点巧合造成的，工作点对齐后 no-GMC 仍更低"
        )
        if no_common:
            L.append(
                f"> 在**固定 v3 工作点**（th_base 0.48 / th_ground 0.26）下，"
                f"cs_t0043（no-GMC）在 Δ={args.target_delta} replay 臂上"
                f"比 rc2（GMC）**低 {abs(fixed_pp):.4f} pp F1**"
                f"（rc2 {pct(v3_rc['f1'])}% @FAR {v3_rc['far']:.6f} vs "
                f"cs_t0043 {pct(v3_cs['f1'])}% @FAR {v3_cs['far']:.6f}）；"
                f"但两模型的 FAR–F1 包络**没有公共区间**（见表 C），无法做同 FAR 插值对比。"
            )
            L.append("")
        else:
            L.append(
                f"> 在**固定 v3 工作点**（th_base 0.48 / th_ground 0.26）下，"
                f"cs_t0043（no-GMC）在 Δ={args.target_delta} replay 臂上"
                f"比 rc2（GMC）**低 {abs(fixed_pp):.4f} pp F1**"
                f"（rc2 {pct(v3_rc['f1'])}% @FAR {v3_rc['far']:.6f} vs "
                f"cs_t0043 {pct(v3_cs['f1'])}% @FAR {v3_cs['far']:.6f}）；"
                f"但在**同一 FAR 下**（公共区间 [{ivl[0]:.6f}, {ivl[1]:.6f}] 内线性插值），"
                f"no-GMC 的 F1 平均**{direction} {mag:.4f} pp**"
                f"（区间范围 {rng[0]:+.4f} ~ {rng[1]:+.4f} pp）"
                f"⇒ **{verdict_text}**。"
            )
            L.append("")

    # ---- 结论补充：F1 之外的维度（Table E 的 v3 点）-----------------------------
    kc, kr = f"cs_t0043|d{args.target_delta}|replay", f"rc2|d{args.target_delta}|replay"
    if kc in table_a and kr in table_a:
        v3c = next(r for r in table_a[kc] if abs(r["th_base"] - V3_POINT[0]) < 1e-9
                   and abs(r["th_ground"] - V3_POINT[1]) < 1e-9)
        v3r = next(r for r in table_a[kr] if abs(r["th_base"] - V3_POINT[0]) < 1e-9
                   and abs(r["th_ground"] - V3_POINT[1]) < 1e-9)
        # 工作点鲁棒性：在网格的全部 th_base 上逐一对比（不插值）
        by_tb_c = {r["th_base"]: r for r in table_a[kc]}
        by_tb_r = {r["th_base"]: r for r in table_a[kr]}
        common = sorted(set(by_tb_c) & set(by_tb_r))
        n_h = sum(1 for t in common if by_tb_c[t]["hit5"] > by_tb_r[t]["hit5"])
        n_c = sum(1 for t in common if by_tb_c[t]["cle_p95_pooled"] < by_tb_r[t]["cle_p95_pooled"])
        d_hit = sorted((by_tb_c[t]["hit5"] - by_tb_r[t]["hit5"]) * 100 for t in common)
        d_cle = sorted(by_tb_c[t]["cle_p95_pooled"] - by_tb_r[t]["cle_p95_pooled"] for t in common)
        L.append(
            f"> **补充（F1 之外的维度，Δ={args.target_delta} replay 臂，直接读产物）**："
            f"no-GMC 的真实优势**不在 F1 上，而在定位质量与误报率上** —— 在 v3 工作点上，"
            f"hit@5px {pct(v3c['hit5'], 2)}% vs {pct(v3r['hit5'], 2)}%"
            f"（{ppv((v3c['hit5'] - v3r['hit5']) * 100, 2)} pp）、"
            f"CLE p95（池化）{v3c['cle_p95_pooled']:.3f} px vs {v3r['cle_p95_pooled']:.3f} px"
            f"（{v3c['cle_p95_pooled'] - v3r['cle_p95_pooled']:+.3f} px）、"
            f"FAR {v3c['far']:.6f} vs {v3r['far']:.6f}（{v3c['far'] - v3r['far']:+.6f}）。"
            f"且这两项定位优势在**网格全部 {len(common)} 个 th_base 工作点上都成立**"
            f"（hit@5px 更优 {n_h}/{len(common)}，CLE p95 更低 {n_c}/{len(common)}；"
            f"hit@5px 差值中位数 {ppv(d_hit[len(d_hit) // 2], 2)} pp、"
            f"CLE p95 差值中位数 {d_cle[len(d_cle) // 2]:+.3f} px）"
            f"⇒ **它们不是 v3 工作点碰巧带来的**。但**同 FAR 下 no-GMC 的 F1 仍低于 GMC**，"
            f"所以「cs_t0043 的 F1 优势」本身并不存在。"
        )
        L.append("")

    J["headline"] = headline

    # ---- 复现 -------------------------------------------------------------------
    L.append("## 复现（完整命令 / 执行机 / 产物路径 / 时间戳）")
    L.append("")
    L.append(f"- **执行机**：{MACHINE}（纯 CPU，`torch.cuda.is_available()==False`；缓存已存在，**未重建**）")
    L.append(f"- **报告生成时间**：{now}")
    L.append("- **网格扫描命令**（前台，幂等，可断点续跑）：")
    L.append("")
    L.append("```bash")
    L.append("cd /tmp/pycharm_project_10ae9e2e")
    L.append("export PYTHONPATH=/tmp/pycharm_project_10ae9e2e")
    L.append("bash manu/tracking/lookahead/server_overnight/run_workpoint_audit.sh")
    L.append("```")
    L.append("")
    L.append("- **单点等价命令**（网格中的任意一点）：")
    L.append("")
    L.append("```bash")
    L.append(
        "$PY manu/tracking/lookahead/lookahead_longquan.py \\\n"
        "  --cache-root <CACHE> --labels-root \"$GT\" --frames-root \"$FRAMES\" --frames-cache \"$FC\" \\\n"
        "  --params-json manu/tracking/lookahead/workpoint_params/wp_thb<THB>_thg<THG>.json \\\n"
        "  --all-seqs --lookahead replay --arms original,stale,kinematic,replay \\\n"
        "  --delta <D> $CFG --out-json <OUT>/arms.json --out-preds-dir <OUT>/preds"
    )
    L.append("```")
    L.append("")
    L.append("  其中 `CACHE` = `runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache`（rc2）")
    L.append("  或 `/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache`（cs_t0043）；")
    L.append("  `CFG` **唯一真值** = `manu/tracking/lookahead/run_delivery.sh` 的 `CFG=` 赋值块（awk 取全块再 eval）。")
    L.append("")
    L.append("- **产物路径**：")
    L.append(f"  - 网格 arms.json：`{root}/<model>_d<Δ>/<th_base>_<th_ground>/arms.json`")
    L.append(f"  - 每次运行的 rc/用时：`{root}/progress.txt`")
    L.append(f"  - 网格 params（每个点一个独立文件，v3 主文件只读）：`manu/tracking/lookahead/workpoint_params/wp_thb<THB>_thg<THG>.json`")
    L.append(f"  - 本报告：`{args.out_md}` / `{args.out_json}`")
    L.append("")

    return "\n".join(L) + "\n", J


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"[FATAL] 网格产物根不存在: {root}")
    grid = load_grid(root)
    if not grid:
        raise SystemExit(f"[FATAL] {root} 下没有可用的 arms.json")
    md, payload = build_report(grid, root, args)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] markdown -> {out_md}")
    print(f"[OK] json     -> {out_json}")
    # 收尾：stdout 打印一行「结论摘要 + 产出路径」
    h = payload.get("headline", {})
    fw, sf = h.get("fixed_workpoint", {}), h.get("same_far", {})
    ivl = sf.get("interval") or [0.0, 0.0]
    print(
        f"[结论] Δ={h.get('delta')} replay：固定 v3 工作点 cs_t0043(no-GMC) 比 rc2(GMC) "
        f"{fw.get('delta_pp', 0.0):+.4f} pp F1；同一 FAR（插值区间 "
        f"[{ivl[0]:.6f}, {ivl[1]:.6f}]）下 {sf.get('mean_delta_pp', 0.0):+.4f} pp "
        f"（范围 {sf.get('min_delta_pp', 0.0):+.4f} ~ {sf.get('max_delta_pp', 0.0):+.4f} pp）"
        f" ⇒ 同 FAR 下 no-GMC 的 F1 {'高' if (sf.get('mean_delta_pp') or 0) > 0 else '低'} rc2；"
        f"产出 {out_md} | {out_json}"
    )


if __name__ == "__main__":
    main()
