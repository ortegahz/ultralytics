#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**前沿审计**（只读）：把台账压成「相对当前基线的七轴差值表」。

为什么需要它
------------
`six_criteria_audit.py` 只回答「六项全过」。但**六项全过 != 可升轨**：
升轨还要「至少一轴严格优」（`auto_promote_watch._strictly_better`：六轴
nano/nano4/f1/t1/t2/t3 任一 > 基线，或 t2_fp < 基线）**且**板端 T4 PASS。
现场实测：台账 13,765 臂里六项全过的 57 臂**全部与基线逐位相同** ⇒ 零增量。

本脚本直接回答三个问题：
  Q1 有没有「六项全过 ∧ 至少一轴严格优」的臂？（= 可升轨候选）
  Q2 哪些臂在 **T1 轴**严格为正？它们**还缺哪几轴**？（近失候选榜）
  Q3 哪些臂能当**供体**去补缺轴？混多少能补平、且不伤已占优的轴？

口径（六项判过 = `>=` 基线，T2 虚警越少越好）：
    六项判过：nano / nano4 / f1 / t1 / t3 >= 基线，且 t2_fp <= 基线
    严格优轴：nano / nano4 / f1 / t1 / t2 / t3 任一 > 基线，或 t2_fp < 基线
    ⚠ T2 轨 F1（`t2`）是**严格优轴**但**不是六项判过轴**（与 `promote_candidate`
      的手工通道一致：`promote_candidate._strictly_better` 没有 t2 轴）。

用法：
    python audit_frontier.py                      # 审计默认台账
    python audit_frontier.py --pair-base 2120000  # 用某臂当基臂做配对预测
    python audit_frontier.py --json /tmp/fr.json  # 同时落机器可读结果
"""
from __future__ import annotations

import argparse
import json

DEFAULT_LEDGER = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"

AXES_P = ("nano", "nano4", "f1")
AXES_O = ("t1", "t2", "t3")
SIX_AXES = ("nano", "nano4", "f1", "t1", "t3")
STRICT_AXES = ("nano", "nano4", "f1", "t1", "t2", "t3")
EPS = 1e-12


def _f(d, key, default=None):
    """安全取浮点：缺失 / 脏值一律返回 default（绝不 raise）。"""
    v = (d or {}).get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_rows(ledger_path):
    """读台账 → (led, 基线七轴, 行表)。行表每行含绝对读数与相对基线的差值。"""
    with open(ledger_path, encoding="utf-8") as fh:
        led = json.load(fh)
    base = led.get("baseline_v2") or {}
    base_ax = {
        "nano": _f(base, "nano"),
        "nano4": _f(base, "nano4"),
        "f1": _f(base, "f1"),
        "t1": _f(base, "t1"),
        "t2": _f(base, "t2"),
        "t3": _f(base, "t3"),
        "t2_fp": _f(base, "t2_fp"),
    }
    rows = []
    for sid, arm in (led.get("arms") or {}).items():
        picked = arm.get("picked") or {}
        official = arm.get("official") or {}
        if not isinstance(official, dict) or official.get("stale"):
            continue
        ax = {
            "nano": _f(picked, "nano"),
            "nano4": _f(picked, "nano4"),
            "f1": _f(picked, "f1"),
            "t1": _f(official, "t1"),
            "t2": _f(official, "t2"),
            "t3": _f(official, "t3"),
            "t2_fp": _f(official, "t2_fp"),
        }
        if any(ax[k] is None for k in AXES_P + AXES_O):
            continue
        rows.append(
            {
                "sid": str(sid),
                "ax": ax,
                "d": {k: ax[k] - base_ax[k] for k in STRICT_AXES},
                "d_t2_fp": ax["t2_fp"] - base_ax["t2_fp"],
                "members": arm.get("soup_members"),
                "weights": arm.get("soup_weights"),
                "desc": arm.get("desc") or "",
                "synth": bool(arm.get("synth")),
                "board_ms": arm.get("board_ms"),
                "board_pass_t4": arm.get("board_pass_t4"),
                "six_ok": official.get("six_ok", arm.get("six_ok")),
            }
        )
    return led, base_ax, rows


def six_pass(row):
    """六项判过（>= 基线；T2 虚警 <= 基线）。"""
    if row["d_t2_fp"] > 0:
        return False
    return all(row["d"][k] >= -EPS for k in SIX_AXES)


def strict_axes(row):
    """严格胜出轴列表（含 T2 轨 F1 与 T2 虚警）。"""
    out = [k for k in STRICT_AXES if row["d"][k] > EPS]
    if row["d_t2_fp"] < 0:
        out.append("t2_fp")
    return out


def missing_axes(row):
    """还缺哪几轴（严格负 = 欠账）。"""
    out = [k for k in SIX_AXES if row["d"][k] < -EPS]
    if row["d_t2_fp"] > 0:
        out.append("t2_fp")
    return out


def _fmt_row(row, extra=""):
    d = row["d"]
    return (
        "sid=%-9s dNano=%+.6f dNano4=%+.6f dF1=%+.6f dT1=%+.6f dT2=%+.6f dT3=%+.6f dT2fp=%+d %s"
        % (
            row["sid"],
            d["nano"],
            d["nano4"],
            d["f1"],
            d["t1"],
            d["t2"],
            d["t3"],
            int(row["d_t2_fp"]),
            extra,
        )
    )


def _recipe_of(row):
    """还原一条臂的配方文本（汤臂用成员+权重，训练臂用 desc）。"""
    mem = row.get("members")
    wts = row.get("weights")
    if mem:
        parts = []
        for i, m in enumerate(mem):
            trial, ep = (list(m) + [None, None])[:2]
            w = None
            if wts and i < len(wts):
                w = wts[i]
            parts.append("%s@%s:w%s" % (trial, ep, w if w is not None else "?"))
        return "soup(" + ",".join(parts) + ")"
    return row.get("desc") or "(无配方)"


def pair_predict(base_row, donor_row, floor=0.005, cap=0.5):
    """线性配对预测：基臂与供体按权重 w 混合，补平基臂的欠账轴。

    ⚠ 权重空间线性、**读数非线性** ⇒ 这是**一阶近似**，用来排配方优先级，
      不是最终结论；最终以台账实测读数为准。
    """
    out = {}
    for axis in SIX_AXES:
        need = -base_row["d"][axis]
        if need <= EPS:
            continue
        gain = donor_row["d"][axis]
        if gain <= EPS:
            out[axis] = None
            continue
        out[axis] = need / gain
    # T2 虚警：越低越好，单独算
    if base_row["d_t2_fp"] > 0 and donor_row["d_t2_fp"] < 0:
        out["t2_fp"] = base_row["d_t2_fp"] / (-donor_row["d_t2_fp"])
    if not out:
        return None
    feasible = {k: v for k, v in out.items() if v is not None}
    if not feasible:
        return None
    w = max(feasible.values())
    if w < floor:
        w = floor
    if w > cap:
        return None
    pred = {k: (1 - w) * base_row["d"][k] + w * donor_row["d"][k] for k in STRICT_AXES}
    pred["t2_fp"] = (1 - w) * base_row["d_t2_fp"] + w * donor_row["d_t2_fp"]
    pred["w"] = w
    pred["need"] = out
    return pred


def merge_recipe(base_row, donor_row, w):
    """把供体按权重 w 掺进基臂 ⇒ 合并后的成员权重表（可照抄进 night_soup3）。"""
    acc = {}
    for src, scale in ((base_row, 1.0), (donor_row, w)):
        mem = src.get("members")
        wts = src.get("weights")
        if not mem:
            continue
        for i, m in enumerate(mem):
            trial, ep = (list(m) + [None, None])[:2]
            ww = 1.0
            if wts and i < len(wts):
                try:
                    ww = float(wts[i])
                except (TypeError, ValueError):
                    ww = 1.0
            key = (trial, ep)
            acc[key] = acc.get(key, 0.0) + ww * scale
    total = sum(acc.values()) or 1.0
    return [(t, e, round(v / total, 6)) for (t, e), v in acc.items()]


def main():
    ap = argparse.ArgumentParser(description="台账前沿审计（只读）")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--pair-base", default="", help="当基臂的 sid（做配对预测）")
    ap.add_argument("--donor-top", type=int, default=12)
    ap.add_argument("--json", default="", help="落机器可读结果")
    args = ap.parse_args()

    led, base_ax, rows = load_rows(args.ledger)
    bname = (led.get("baseline_v2") or {}).get("name")
    btrial = (led.get("baseline_v2") or {}).get("trial")
    print("===== 前沿审计（只读）=====")
    print("基线 = %s / trial_%s" % (bname, btrial))
    print("基线六轴: " + " ".join("%s=%.7f" % (k, base_ax[k]) for k in ("nano", "nano4", "f1", "t1", "t2", "t3")))
    print("可读臂数 = %d（台账总臂数 %d）" % (len(rows), len(led.get("arms") or {})))

    by_sid = {r["sid"]: r for r in rows}

    # ── S1 六项全过 ∧ 至少一轴严格优 ───────────────────────────────
    s1 = [r for r in rows if six_pass(r) and strict_axes(r)]
    s1.sort(key=lambda r: (r["d"]["t1"], r["d"]["nano4"]), reverse=True)
    print()
    print("--- S1 六项全过 ∧ 至少一轴严格优（= 可升轨候选，不含板端判定）: %d 个 ---" % len(s1))
    for r in s1[:15]:
        print("  " + _fmt_row(r, "| 胜出=%s | %s" % (",".join(strict_axes(r)), _recipe_of(r))))
    if not s1:
        print("  （无）⇒ 当前台账里**没有任何**六项全过且严格占优的臂")

    # ── S2 T1 严格为正（近失候选）────────────────────────────────
    s2 = [r for r in rows if r["d"]["t1"] > EPS]
    s2.sort(key=lambda r: (len(missing_axes(r)), -r["d"]["t1"]))
    print()
    print("--- S2 dT1>0 的臂: %d 个（按「欠账轴数」排序，少的在前）---" % len(s2))
    for r in s2[:15]:
        print("  " + _fmt_row(r, "| 欠账=%s | %s" % (",".join(missing_axes(r)) or "无", _recipe_of(r))))
    if not s2:
        print("  （无）")

    # ── S3 可当供体的臂（修欠账轴）──────────────────────────────
    for axis, label in (("nano4", "修 dNano4"), ("t3", "修 dT3"), ("nano", "修 dNano")):
        pool = [r for r in rows if r["d"][axis] > EPS]
        pool.sort(key=lambda r: (-r["d"][axis], -r["d"]["t1"]))
        print()
        print("--- S3 %s 供体（d%s>0）: %d 个，按 d%s 降序取前 %d ---"
              % (label, axis, len(pool), axis, args.donor_top))
        for r in pool[: args.donor_top]:
            print("  " + _fmt_row(r, "| %s" % _recipe_of(r)))

    # ── S4 配对预测 ──────────────────────────────────────────────
    base_row = by_sid.get(str(args.pair_base))
    if base_row is None:
        cand = [r for r in rows if r["d"]["t1"] > EPS]
        if cand:
            cand.sort(key=lambda r: -r["d"]["t1"])
            base_row = cand[0]
    if base_row is not None:
        print()
        print("--- S4 配对预测：基臂 sid=%s %s ---" % (base_row["sid"], _recipe_of(base_row)))
        print("    基臂欠账: %s" % (",".join(missing_axes(base_row)) or "无"))
        donors = [r for r in rows if r["sid"] != base_row["sid"] and r["d"]["nano4"] > EPS]
        donors.sort(key=lambda r: -r["d"]["nano4"])
        shown = 0
        for donor in donors:
            pred = pair_predict(base_row, donor)
            if pred is None:
                continue
            rec = merge_recipe(base_row, donor, pred["w"])
            print(
                "    供体 sid=%-9s dNano4=%+.6f dT1=%+.6f dT3=%+.6f ⇒ w=%.4f ⇒ 预测 dNano=%+.6f dNano4=%+.6f dT1=%+.6f dT3=%+.6f"
                % (
                    donor["sid"],
                    donor["d"]["nano4"],
                    donor["d"]["t1"],
                    donor["d"]["t3"],
                    pred["w"],
                    pred["nano"],
                    pred["nano4"],
                    pred["t1"],
                    pred["t3"],
                )
            )
            print("        配方: %s" % ",".join("%s@%s:w%s" % (t, e, w) for t, e, w in rec))
            shown += 1
            if shown >= 8:
                break
        if shown == 0:
            print("    （没有可行配对：要么供体补不动欠账轴，要么需要权重 > 0.5）")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "baseline": {"name": bname, "trial": btrial, "axes": base_ax},
                    "n_rows": len(rows),
                    "s1_strictly_better": [r["sid"] for r in s1],
                    "s2_t1_positive": [
                        {"sid": r["sid"], "d": r["d"], "missing": missing_axes(r)} for r in s2
                    ],
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        print()
        print("JSON 已落: %s" % args.json)


if __name__ == "__main__":
    main()
