#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**T3 前沿地图**：把多个 study 的实测点画成 T1–T3 前沿，并用**受控对**量出各旋钮的汇率。

（2026-11-03 T3 前沿轮的收尾分析脚本。与 `analyze_t3_reachability.py` 互补：
 那个回答「T3 还能不能更高」，这个回答「**扩空间之后前沿有没有被推开**、以及**离 RC2 还有多远**」。）

## 为什么不用回归、只用受控对
==========================
15 维超参空间里 `translate`/`weight_decay` 近乎常数 ⇒ OLS 系数爆到 17/50、pooled R² = **−0.65**
⇒ 回归结论不可用。**受控对**（两条记录除某一维外逐位相同）没有这个病：
  · 受控对必须读 **`study.db` 原始 float**（CSV 的浮点格式化会造出/抹掉对子）；
  · categorical 要按各 study 自己的 `distribution_json` 解码
    （focus 的 `init_from=[rc2,base474]`，weekend 的 `[base474,rc2]`，**顺序相反**）；
  · 方向要按**数值**归一（`mix_spec` 的 choices 含 `natural`，字典序会把它当最大值）。

## 用法（服务器，conda `uav`，根 /tmp/pycharm_project_10ae9e2e）
==========================================================
```bash
python manu/training/analyze_t3_frontier.py \
  --study "focus=/mnt/data/siping/runs/nogmc_focus/study.db:mix_ratio_search_nogmc_focus" \
  --study "weekend=/mnt/data/siping/runs/nogmc_weekend/study.db:mix_ratio_search_nogmc_weekend" \
  --study "t3=/mnt/data/siping/runs/nogmc_t3/study.db:mix_ratio_search_nogmc_t3" \
  --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --out-md /mnt/data/siping/runs/nogmc_t3/T3_FRONTIER.md \
  --out-json /mnt/data/siping/runs/nogmc_t3/T3_FRONTIER.json
```
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sqlite3
import statistics as st
from pathlib import Path

# 配方键用的 15 维（+ init_from_x）；缺项按 "" 处理 ⇒ 跨 study 可比
CORE_COLS = ("mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
             "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn",
             "init_from", "epochs", "backbone_lr_scale", "freeze_backbone_epochs")
KEY_COLS = CORE_COLS + ("init_from_x",)
INT_COLS = {"epochs", "freeze_backbone_epochs"}


def _decode_params(conn) -> dict[int, dict]:
    """trial_id -> 参数（categorical 按各自 study 的 choices 解码）。"""
    out: dict[int, dict] = {}
    for tid, num, stt in conn.execute("SELECT trial_id, number, state FROM trials"):
        out[tid] = {"num": num, "state": stt, "params": {}, "attrs": {}}
    for tid, pn, pv, dj in conn.execute(
            "SELECT trial_id, param_name, param_value, distribution_json FROM trial_params"):
        if tid not in out:
            continue
        v = pv
        try:
            d = json.loads(dj)
            ch = (d.get("attributes") or {}).get("choices")
            if ch:
                v = ch[int(pv)]
        except Exception:
            pass
        out[tid]["params"][pn] = v
    for tid, k, vj in conn.execute("SELECT trial_id, key, value_json FROM trial_user_attributes"):
        if tid not in out:
            continue
        try:
            out[tid]["attrs"][k] = json.loads(vj)
        except Exception:
            out[tid]["attrs"][k] = vj
    return out


def load_csv_metrics(path: str) -> dict:
    """读 `optuna_summary.csv` ⇒ {trial_XXXX: {t1_f1, t3_f1, ...}}。

    ⚠️ **为什么需要 CSV 兜底**：2026-11-03 之前跑的 study（focus / weekend）
    的 `trial_user_attributes` 里**只有 role/enqueued_by/seed**（早于「把 harvest 结果写进
    user_attrs」的改动）⇒ 三轨指标**只在 CSV 和 trial_result.json 里**。
    而**新 study（t3）走的是新调度器 ⇒ 指标在 user_attrs 里** ⇒ 两个来源都要支持。
    """
    out = {}
    if not path or not Path(path).exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r.get("trial", "")] = r
    return out


def load_study(db_path: str, study_name: str, tag: str,
               csv_metrics: dict | None = None) -> list[dict]:
    conn = sqlite3.connect(db_path)
    tr = _decode_params(conn)
    # ⚠️ `studies` 表**没有 trial_id 列** ⇒ 要先取 study_id，再按 `trials.study_id` 过滤
    # ⚠️ `studies` 的主键列名是 **study_id**（不是 id）
    ids = [r[0] for r in conn.execute("SELECT study_id FROM studies WHERE study_name=?", (study_name,))]
    keep = set()
    for sid in ids:
        keep |= {r[0] for r in conn.execute("SELECT trial_id FROM trials WHERE study_id=?", (sid,))}
    recs = []
    for tid, t in tr.items():
        if keep and tid not in keep:
            continue
        p = t["params"]
        if not p:
            continue
        a = t["attrs"]
        m = a.get("metrics") or {}
        # ⚠️⚠️ **2026-11-04 15:25 修（键名不对 ⇒ 既丢 user_attrs 又跳过 CSV 兜底）**：
        #    `user_attrs["metrics"]` 是把 `trial_result.json` 的 `metrics` 段**原样**写进去的
        #    ⇒ 键名是 `track1_f1/track2_f1/t3_fp…`，而下游统一用 `t1_f1/t2_f1/t3_f1…`。
        #    ⇒ `m` 非空 ⇒ **CSV 兜底被跳过** ⇒ 取 `m.get("t1_f1")` 全是 None
        #    ⇒ 该 trial 的 T1/T3 变 None ⇒ **被静默排除出配置组**（实测 t49/t50 归组后 n 仍是 1）。
        #    ⇒ 这里先做一次**键名归一化**，再决定要不要走 CSV 兜底。
        if m:
            _alias = {"track1_f1": "t1_f1", "track1_fp": "t1_fp",
                      "track2_f1": "t2_f1", "track2_fp": "t2_fp",
                      "track3_f1": "t3_f1", "track3_fp": "t3_fp"}
            m = {_alias.get(k, k): v for k, v in m.items()}
        if not m and csv_metrics:
            # CSV 兜底：指标键名与 metrics 一致（t1_f1/t2_f1/t2_fp/t3_f1/t3_fp）
            row = csv_metrics.get("trial_%04d" % t["num"]) or {}
            m = {k: row.get(k) for k in ("t1_f1", "t2_f1", "t2_fp", "t3_f1", "t3_fp")
                 if row.get(k) not in (None, "")}
        recs.append({
            "tag": tag, "num": t["num"], "state": t["state"], "params": p,
            "role": a.get("role", ""), "seed": a.get("seed"),
            "verdict": a.get("gate_verdict", ""),
            "t1": _num(m.get("t1_f1")), "t2": _num(m.get("t2_f1")), "t2fp": _num(m.get("t2_fp")),
            "t3": _num(m.get("t3_f1")), "t3fp": _num(m.get("t3_fp")),
            "utility": a.get("utility"), "utility_vs_rc1": a.get("utility_vs_rc1"),
            "tier1_ok": a.get("tier1_ok"),
        })
    conn.close()
    return recs


def _num(v):
    try:
        return float(v)
    except Exception:
        return None


def key_of(r: dict) -> tuple:
    out = []
    p = r["params"]
    for c in KEY_COLS:
        v = p.get(c, "")
        if c in ("mix_spec", "init_from", "init_from_x"):
            out.append(str(v))
        elif c in INT_COLS:
            try:
                out.append(int(float(v)))
            except Exception:
                out.append(str(v))
        elif c == "freeze_bn":
            out.append(str(v).lower() in ("true", "1", "yes"))
        else:
            try:
                out.append(round(float(v), 6))
            except Exception:
                out.append(str(v))
    return tuple(out)


def mix_num(m) -> float | None:
    """mix 比值（`natural` 单独标记 ⇒ 返回 0.0 表示「不比其它比值大」）。"""
    s = str(m)
    if s == "natural":
        return 0.0
    try:
        return float(s.lstrip("w"))
    except Exception:
        return None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else None


def controlled_pairs(recs: list[dict], knob: str, lo_tag: str | None = None):
    """同配方（除 knob 外逐位相同）的受控对。返回 [(lo, hi, span, dT3, dT1)]。"""
    out = []
    for a, b in itertools.combinations(recs, 2):
        if key_of(a) == key_of(b):
            continue
        ka, kb = key_of(a), key_of(b)
        if tuple(x for i, x in enumerate(ka) if KEY_COLS[i] != knob) != \
           tuple(x for i, x in enumerate(kb) if KEY_COLS[i] != knob):
            continue
        va, vb = a["params"].get(knob), b["params"].get(knob)
        if va is None and vb is None:
            continue
        if a["t3"] is None or b["t3"] is None or a["t1"] is None or b["t1"] is None:
            continue
        if knob == "mix_spec":
            na, nb = mix_num(va), mix_num(vb)
            if na is None or nb is None or na == nb:
                continue
            lo, hi = (a, b) if na < nb else (b, a)
            span = abs(nb - na)
        else:
            fa = fb = None
            try:
                fa, fb = float(va), float(vb)
            except Exception:
                fa = fb = None
            if fa is None or fb is None:
                # ⚠️ **categorical 旋钮**（`init_from` / `init_from_x`）走字符串分支。
                #    2026-11-04 09:00 修：原来这里一律 `float(va)` ⇒ 遇到
                #    `'rc2+soup_strict3'` 直接抛异常被 `continue` 掉；
                #    且更早一步 `if va is None or vb is None: continue` 把
                #    **rc2 起手臂的 `init_from_x = None`**（「没有 soup 初始化」是
                #    **合法取值**）当「缺失」丢掉 ⇒
                #    ⇒ `soup` 这个**唯一把前沿向外推的旋钮**的受控对**一个都配不出来**
                #    （实测 rates.init_from.n = 0 / init_from_x.n = 0）。
                #    ⇒ 现在：只要**不是两边都 None** 就是合法配对；
                #       排序用字符串（`""` 代表「无」，排在最前 ⇒ 方向是「rc2 → soup」）。
                sa = "" if va is None else str(va)
                sb = "" if vb is None else str(vb)
                if sa == sb:
                    continue
                lo, hi = (a, b) if sa < sb else (b, a)
                span = 1.0
            else:
                if fa == fb:
                    continue
                lo, hi = (a, b) if fa < fb else (b, a)
                span = abs(fb - fa)
        out.append((lo, hi, span, hi["t3"] - lo["t3"], hi["t1"] - lo["t1"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", action="append", required=True,
                    help="tag=db路径:study名（可重复）")
    ap.add_argument("--csv", action="append", default=[],
                    help="tag=optuna_summary.csv 路径（早于 user_attrs 存指标的 study 需要它）")
    ap.add_argument("--ref", required=True, help="reference_frame.json")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--rc2-t1", type=float, default=0.9008967482400269,
                    help="带 GMC 的 RC2 的 T1（对照目标；参照系包里没有这个值）")
    ap.add_argument("--rc2-t3", type=float, default=0.897694,
                    help="带 GMC 的 RC2 的 T3（对照目标）")
    args = ap.parse_args()

    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    # ⚠️ **参照系包里的 `frozen_baseline` 是「无 GMC 的冻结基线」（trial_0474）**，
    #    **不是**带 GMC 的 RC2！两者 T1 相同（0.9008967），但 T3 差 14 pp：
    #       无 GMC 基线：T1 0.900897 / T3 0.757812
    #       带 GMC 的 RC2：T1 0.900897 / T3 0.897694
    #    ⇒ 两个参照必须分开写，混用会把结论带偏。
    fb = ref.get("frozen_baseline") or ref.get("gate_base") or {}
    base_t1 = float(fb.get("track1_f1") or 0.0)
    base_t3 = float(fb.get("track3_f1") or 0.0)
    base_t2 = float(fb.get("track2_f1") or 0.0)
    rc2_t1 = float(args.rc2_t1)
    rc2_t3 = float(args.rc2_t3)
    red_t1 = base_t1 + float((ref.get("gate") or {}).get("track1_min_delta") or -0.005)

    csv_by_tag = {}
    for cs in (args.csv or []):
        tag, path = cs.split("=", 1)
        csv_by_tag[tag] = load_csv_metrics(path)

    recs: list[dict] = []
    for s in args.study:
        tag, rest = s.split("=", 1)
        db, _, name = rest.rpartition(":")
        got = load_study(db, name, tag, csv_by_tag.get(tag))
        recs += got
        print("[FRONTIER] %-8s %-46s %d 条" % (tag, name, len(got)))

    # ⚠️ **不按 state 过滤**（2026-11-03）：`PRUNED` 的 trial **也有完整三轨指标**
    #    （调度器在一级红线不过时记 PRUNED，但指标照常落盘）⇒
    #    只取 COMPLETE 会**漏掉本轮 T3 最高的那批臂**（它们的 verdict 是
    #    FAIL_TRACK1_RETREAT ⇒ 既不在严格通道、也不在「放宽 PASS」子集，
    #    但它们恰恰是「T3 冲得最高」的那批）。
    #    ⇒ 判据改成「三轨指标齐全」，state 只作标注。
    done = [r for r in recs if r["t1"] and r["t2"] and r["t3"]]
    print("[FRONTIER] COMPLETE 且有三轨指标：%d 条" % len(done))

    # ---- 受控对统计 ----
    knobs = ["lr_scale", "epochs", "mix_spec", "init_from", "init_from_x",
             "backbone_lr_scale", "freeze_backbone_epochs", "offset_weight", "focal_beta",
             "mosaic", "scale", "translate", "clip_max_norm", "warmup_epochs",
             "weight_decay", "freeze_bn"]
    rate = {}
    for kn in knobs:
        ps = controlled_pairs(done, kn)
        if not ps:
            rate[kn] = {"n": 0, "note": "没有受控对 ⇒ 效应未识别"}
            continue
        d3 = [p[3] for p in ps]
        d1 = [p[4] for p in ps]
        up = sum(1 for x in d3 if x > 0)
        rate[kn] = {
            "n": len(ps),
            "dT3_median_pp": 100 * st.median(d3),
            "dT1_median_pp": 100 * st.median(d1),
            "t3_up": up, "t3_up_ratio": f"{up}/{len(ps)}",
            "pairs": [{"lo": f"{p[0]['tag']}t{p[0]['num']:04d}", "hi": f"{p[1]['tag']}t{p[1]['num']:04d}",
                       "lo_val": p[0]["params"].get(kn), "hi_val": p[1]["params"].get(kn),
                       "span": p[2], "dT3_pp": 100 * p[3], "dT1_pp": 100 * p[4]}
                      for p in ps],
        }

    # ---- 前沿（配置 = 同配方） ----
    groups: dict[tuple, list[dict]] = {}
    for r in done:
        groups.setdefault(key_of(r), []).append(r)
    cfgs = []
    for k, rs in groups.items():
        p = rs[0]["params"]
        cfgs.append({
            "mix": p.get("mix_spec"), "lr": p.get("lr_scale"), "ep": p.get("epochs"),
            "init": p.get("init_from"), "init_x": p.get("init_from_x", ""),
            "n": len(rs),
            "t1_med": st.median([x["t1"] for x in rs]),
            "t3_med": st.median([x["t3"] for x in rs]),
            "t1_min": min(x["t1"] for x in rs), "t1_max": max(x["t1"] for x in rs),
            "t3_min": min(x["t3"] for x in rs), "t3_max": max(x["t3"] for x in rs),
            "verdict": rs[0]["verdict"],
            "members": [f"{x['tag']}t{x['num']:04d}" for x in rs],
            "roles": [x["role"] for x in rs],
        })
    cfgs.sort(key=lambda c: -c["t3_med"])

    # ---- 前沿斜率（用 Pareto 前沿上的相邻点） ----
    pareto = []
    for c in sorted(cfgs, key=lambda c: -c["t1_med"]):
        if not pareto or c["t3_med"] > pareto[-1]["t3_med"]:
            pareto.append(c)
    slope = None
    if len(pareto) >= 2:
        # pareto 是「T1 递减 ⇒ T3 递增」的序列 ⇒ 取**两端**算总前沿斜率：
        #   每丢 1 pp T1，能换来多少 pp T3（正数 = 越靠近 RC2 越划算）
        a, b = pareto[0], pareto[-1]      # a: T1 最高 / T3 最低；b: T1 最低 / T3 最高
        dt1 = (a["t1_med"] - b["t1_med"])
        dt3 = (b["t3_med"] - a["t3_med"])
        if abs(dt1) > 1e-9:
            slope = dt3 / dt1
    # **局部斜率**（T3 最优端的两点）—— 与「某旋钮的汇率」比较时**必须用局部**：
    # 全局斜率被「低 T3 高 T1」的远端点拉偏，会把旋钮的性价比判错。
    slope_local = None
    if len(pareto) >= 2:
        a, b = pareto[-2], pareto[-1]
        dt1 = (a["t1_med"] - b["t1_med"])
        dt3 = (b["t3_med"] - a["t3_med"])
        if abs(dt1) > 1e-9:
            slope_local = dt3 / dt1

    # ---- 与 RC2 的距离 ----
    best_t3 = max(done, key=lambda r: r["t3"]) if done else None
    best_t1 = max(done, key=lambda r: r["t1"]) if done else None
    strict = [r for r in done if r["verdict"] == "PASS"]
    best_strict_t3 = max(strict, key=lambda r: r["t3"]) if strict else None

    out = {
        "rc2_with_gmc": {"t1": rc2_t1, "t3": rc2_t3},
        "baseline_no_gmc": {"t1": base_t1, "t3": base_t3, "t2": base_t2},
        "red_line_t1": red_t1,
        "n_complete": len(done),
        "pearson_t1_t3": pearson([r["t1"] for r in done], [r["t3"] for r in done]),
        "rates": rate,
        "configs": cfgs,
        "pareto": [{"mix": c["mix"], "lr": c["lr"], "ep": c["ep"], "n": c["n"],
                    "t1_med": c["t1_med"], "t3_med": c["t3_med"]} for c in pareto],
        "frontier_slope_t3_per_t1": slope,
        "frontier_slope_local_t3_per_t1": slope_local,
        "best_t3": best_t3 and {"tag": best_t3["tag"], "num": best_t3["num"], "role": best_t3["role"],
                                "t1": best_t3["t1"], "t3": best_t3["t3"], "verdict": best_t3["verdict"]},
        "best_t1": best_t1 and {"tag": best_t1["tag"], "num": best_t1["num"], "role": best_t1["role"],
                                "t1": best_t1["t1"], "t3": best_t1["t3"], "verdict": best_t1["verdict"]},
        "best_strict_t3": best_strict_t3 and {
            "tag": best_strict_t3["tag"], "num": best_strict_t3["num"], "role": best_strict_t3["role"],
            "t1": best_strict_t3["t1"], "t3": best_strict_t3["t3"], "verdict": best_strict_t3["verdict"]},
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- markdown ----
    L = []
    A = L.append
    A("# T3 前沿地图（no-GMC）")
    A("")
    A(f"- 数据：{len(done)} 条**三轨指标齐全**的 trial（不限 state：PRUNED 的臂指标同样完整）")
    A(f"- **对照目标 RC2（带 GMC）**：T1 **{rc2_t1:.6f}** / T3 **{rc2_t3:.6f}**")
    A(f"- 同语义冻结基线（**无 GMC**，trial_0474）：T1 **{base_t1:.6f}** / T3 **{base_t3:.6f}**")
    A(f"- T1 红线 = 基线 − 0.005 = **{red_t1:.6f}**；T3 红线 = 0.87")
    A(f"- Pearson(T1, T3) = **{out['pearson_t1_t3']:.4f}**" if out["pearson_t1_t3"] is not None else "- Pearson: n/a")
    if slope:
        A(f"- 前沿斜率（全局，两端点）：**每丢 1 pp T1 换 {slope:.1f} pp T3**")
    if slope_local:
        A(f"- 前沿斜率（**局部**，T3 最优端）：**每丢 1 pp T1 换 {slope_local:.1f} pp T3**"
          f" ⇒ 旋钮汇率优于这个数，才算「把前沿向外推」而不是沿前沿滑动")
    A("")
    A("## 1. 逐旋钮汇率（受控对）")
    A("")
    A("| 旋钮 | 受控对 | ΔT3 中位 | ΔT1 中位 | T3 变好 |")
    A("|---|--:|--:|--:|:--:|")
    for kn, v in rate.items():
        if not v.get("n"):
            A(f"| `{kn}` | 0 | — | — | **未识别** |")
        else:
            A(f"| `{kn}` | {v['n']} | {v['dT3_median_pp']:+.4f} pp | "
              f"{v['dT1_median_pp']:+.4f} pp | {v['t3_up_ratio']} |")
    A("")
    A("## 2. 配置级前沿（按 T3 中位排序，前 25）")
    A("")
    A("| mix | lr | ep | init | n | T1 中位 | T3 中位 | T1 缺口(vs 红线) | 距 RC2 的 T3 |")
    A("|---|--:|--:|---|--:|--:|--:|--:|--:|")
    red = red_t1
    for c in cfgs[:25]:
        A(f"| {c['mix']} | {c['lr']} | {c['ep']} | {c['init']}"
          f"{('/'+c['init_x']) if c['init_x'] else ''} | {c['n']} | "
          f"{c['t1_med']:.6f} | {c['t3_med']:.6f} | {100*(c['t1_med']-red):+.4f} pp | "
          f"{100*(c['t3_med']-rc2_t3):+.4f} pp |")
    A("")
    A("## 3. 与 RC2 的距离")
    A("")
    if best_t3:
        A(f"- T3 最高：{best_t3['tag']}t{best_t3['num']:04d}（{best_t3['role']}）"
          f" T3 **{best_t3['t3']:.6f}** / T1 {best_t3['t1']:.6f} / {best_t3['verdict']}"
          f" ⇒ 距 RC2 的 T3 还差 **{100*(rc2_t3-best_t3['t3']):+.4f} pp**")
    if best_strict_t3:
        A(f"- **严格通道** T3 最高：{best_strict_t3['tag']}t{best_strict_t3['num']:04d}"
          f" T3 {best_strict_t3['t3']:.6f} / T1 {best_strict_t3['t1']:.6f} / {best_strict_t3['verdict']}")
    A("")
    if args.out_md:
        Path(args.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
        print("[FRONTIER] 已写 %s" % args.out_md)
    print("[FRONTIER] 完成：T3 最高 %s，前沿斜率 %s" % (
        ("%.6f" % best_t3["t3"]) if best_t3 else "n/a",
        ("%.1f pp T3 / 1 pp T1" % slope) if slope else "n/a"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
