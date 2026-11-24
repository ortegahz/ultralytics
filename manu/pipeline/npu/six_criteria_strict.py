#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六项判据**严格口径**全盘审计（只读）。

🟢 2026-11-23 新增：回答「有没有**真·增量**的模型可升轨」的**唯一权威口径**。

🔴 为什么必须另立一个脚本（`six_criteria_audit.py` 不够用）：
  ① `six_criteria_audit.judge()` 六轴**全部用 `>=`** ⇒ 「与基线逐位打平」的臂也算「六项全过」
     ⇒ 晨报里那 57 个「六项全过」臂**全是基线权重的逐位复现**（零增量），
        直接拿它升轨 = **基线名通胀**（2026-11-18 事故：同一臂重复升轨 82 次）。
  ② 晨报/升轨命令清单的打印精度是 6~7 位小数，而 nano 的 1 阶台阶只有 `2.3629489e-4`
     ⇒ **6~7 位小数在基线附近根本分不出「严格更优」与「严格更劣」**
     （例：基线 `0.7242438563327032` 与候选 `0.7242439` 在 7 位小数下**完全同形**）。
  ③ 台账 `arms[*].nanobench[ep]` 存的是**满精度 float** ⇒ 严格比较必须用满精度算。

⇒ 本脚本做四件事（全部只读：`json.load` + `glob` + 读文件，不写任何状态）：
  **A** 六项**严格**全过（T1/T3 已在磁盘门禁报告里）⇒ 真冠军候选；
  **B** 六项 `>=` 全过但严格口径不过 ⇒ 区分「基线逐位克隆」与「只差一个台阶」；
  **C** **三轴**（nano/nano4/f1）严格全过、但 **T1/T3 从未门禁** ⇒ 最可能产出真冠军的待门禁臂；
  **D** 台账全量扫描（不限有门禁报告的臂）里的三轴严格全过臂排行。

严格口径（与 2026-11-23 开箱清单 §1 一致）：

    | 指标 | 判据 |
    |---|---|
    | nano  | **严格 >** 基线 |
    | nano4 | **严格 >** 基线 |
    | F1    | **严格 >** 基线 |
    | T1    | **>=** 基线 |
    | T3    | **>=** 基线 |
    | t2_fp | **<=** max(基线 t2_fp, 5) |

台阶量纲（1 阶）：nano/nano4 `2.3629489e-4` ｜ F1 ≈ `2.24e-4` ｜ T1 `1.858e-5` ｜ T3 ≈ `1.95e-5`。

用法：
    python manu/pipeline/npu/six_criteria_strict.py
    python manu/pipeline/npu/six_criteria_strict.py --json-out /tmp/strict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import six_criteria_audit as SCA  # noqa: E402

LEDGER = SCA.LEDGER
GATE_DIR = SCA.GATE_DIR

# 🟢 1 阶台阶量纲（2026-11-23 开箱清单 §1）
STEP = {
    "nano": 2.3629489e-4,
    "nano4": 2.3629489e-4,
    "f1": 2.24e-4,
    "t1": 1.858e-5,
    "t3": 1.95e-5,
}

# 🔴 严格轴 = 必须**严格优于**基线；其余轴 = 允许打平（>=）
STRICT_AXES = ("nano", "nano4", "f1")
GE_AXES = ("t1", "t3")
ALL_AXES = ("nano", "nano4", "f1", "t1", "t2fp", "t3")

# 🔴 门禁短路时 T2/T3 的**哨兵值**（不是真实测量值）
SENTINEL_FP = 999


def _t2t3_measured(row: dict) -> tuple[bool, bool]:
    """T2 / T3 是否**真测过**。

    🔴 门禁在 T1 不达标时**短路**（跳过龙泉山缓存 + T2/T3 评测）
       ⇒ 台账 `official` 里写的是哨兵值：`t2_fp=999` / `t3=-1.0` / `t3_fp=999`。
       把哨兵值当真实读数会算出 −99,880 阶这种荒谬的 Δ ⇒ 必须先识别。
    """
    t3 = row.get("t3")
    t3_measured = (
        t3 is not None
        and float(t3) >= 0
        and int(row.get("t3_fp") or 0) < SENTINEL_FP
    )
    t2_fp = row.get("t2_fp")
    t2_measured = t2_fp is not None and int(t2_fp) < SENTINEL_FP
    return t2_measured, t3_measured


def collect_ledger_official(led: dict) -> list[dict]:
    """从**台账 `arms[trial].official`** 补齐六轴读数（磁盘门禁报告缺失时）。

    🔴 根因（2026-11-23 定位）：`clean_infra.py` 在门禁 FAIL 后**直接 rmtree 门禁目录**
       （证据：`nano_search/clean_infra.log` 的「清理 infra 失败门禁目录」行）
       ⇒ 任何以 `gate_report.json` **存在与否**为判据的审计，
          都会把「**测过且 FAIL**」的臂误报成「**从未门禁**」。
       ⇒ 双证据源：① 磁盘 `gate_report.json`（权威报告，优先）
                     ② 台账 `arms[*].official`（durable：FAIL 后依然在台账里）
    """
    arms = led.get("arms") or {}
    out = []
    for key, st in arms.items():
        if not isinstance(st, dict):
            continue
        o = st.get("official") or {}
        if not isinstance(o, dict) or o.get("t1") is None:
            continue
        ep = st.get("picked_epoch")
        ep = str(ep) if ep is not None else ""
        nb = (st.get("nanobench") or {}).get(ep)
        if not nb:
            continue
        out.append({
            "trial": int(key),
            "ep": ep,
            "nano": nb.get("nano"),
            "nano4": nb.get("nano4"),
            "f1": nb.get("f1"),
            "t1": float(o["t1"]),
            "t1_fp": o.get("t1_fp"),
            "t2": o.get("t2"),
            "t2_fp": o.get("t2_fp"),
            "t3": o.get("t3"),
            "t3_fp": o.get("t3_fp"),
            "veto": o.get("veto_failures") or [],
            "verdict": o.get("verdict"),
            "skipped_t2t3": bool(o.get("skipped_t2t3")),
            "desc": st.get("desc") or "",
            "src": "ledger_official",
        })
    return out


def strict_flags(row: dict, base: dict, t2_cap: int) -> dict:
    """六轴严格判定（返回逐轴 bool + 逐轴 Δ + 逐轴台阶数）。"""
    out = {}
    for k in STRICT_AXES:
        v = row.get(k)
        b = float(base[k])
        out[k] = (v is not None and float(v) > b)
    for k in GE_AXES:
        v = row.get(k)
        b = float(base[k])
        out[k] = (v is not None and float(v) >= b)
    out["t2fp"] = row.get("t2_fp") is not None and int(row["t2_fp"]) <= t2_cap
    return out


def deltas(row: dict, base: dict) -> dict:
    """逐轴 Δ（原始单位）与台阶数（1 阶 = STEP[k]）。"""
    out = {}
    for k in ("nano", "nano4", "f1", "t1", "t3"):
        v = row.get(k)
        if v is None:
            out[k] = None
            continue
        d = float(v) - float(base[k])
        out[k] = {"delta": d, "steps": d / STEP[k]}
    out["t2fp"] = row.get("t2_fp")
    out["t2fp_delta"] = (int(row["t2_fp"]) - int(base["t2_fp"])) if row.get("t2_fp") is not None else None
    return out


def is_clone(row: dict, base: dict) -> bool:
    """基线**逐位克隆** = 六轴读数与基线**完全相等**（float 精确相等）。"""
    for k in ("nano", "nano4", "f1", "t1", "t3"):
        v = row.get(k)
        if v is None or float(v) != float(base[k]):
            return False
    return row.get("t2_fp") is not None and int(row["t2_fp"]) == int(base["t2_fp"])


def five_axis_zero(row: dict, base: dict) -> bool:
    """五个**指标轴**（nano/nano4/f1/t1/t3）与基线**逐位相同** ⇒ 指标零增量。

    🔴 与 `is_clone()` 的区别：`is_clone()` 还要求 `t2_fp` 也相同。
    实战里存在**五轴逐位相同、但 t2_fp 比基线多 1 个虚警**的臂 ——
    它 `>=` 过六项门禁，却**没有任何指标增量**，而且 T2 严格**更差**。
    ⇒ 这种臂**不是「近失」，是「零增量且更差」** ⇒ 绝不能升轨，
    且必须与真·近失臂分开统计，否则会把 0 增量臂误读成候选。
    """
    for k in ("nano", "nano4", "f1", "t1", "t3"):
        v = row.get(k)
        if v is None or float(v) != float(base[k]):
            return False
    return True


def fmt(v, nd=17):
    return "None" if v is None else f"{float(v):.{nd}g}"


def ledger_three_axis_strict(led: dict, base: dict) -> list:
    """台账全量扫描：不要求有磁盘门禁报告，只看 `arms[*].nanobench[ep]` 的三轴。

    ⇒ 这是「**还没送官方门禁**」的臂池，也是最可能产出真冠军的地方
      （2026-11-23 清单 §3：11042xx 家族 nano/nano4 双过基线但 T1 从未门禁）。
    """
    arms = led.get("arms") or {}
    out = []
    for key, st in arms.items():
        if not isinstance(st, dict):
            continue
        nb = st.get("nanobench") or {}
        for ep, v in nb.items():
            if not isinstance(v, dict):
                continue
            if v.get("nano") is None or v.get("nano4") is None or v.get("f1") is None:
                continue
            row = {
                "trial": int(key),
                "ep": ep,
                "nano": v.get("nano"),
                "nano4": v.get("nano4"),
                "f1": v.get("f1"),
                "t1": None,
                "t2_fp": None,
                "t3": None,
                "desc": st.get("desc") or "",
                "soup_family": st.get("soup_family"),
                # 台账自己记的 official 读数：若存在但没有 gate_report.json ⇒
                # 「测过但报告文件丢了」⇒ 不能当成「从未测过」重复送门禁。
                "has_official": bool(st.get("official")),
            }
            ok3 = all(float(row[k]) > float(base[k]) for k in STRICT_AXES)
            row["three_axis_strict"] = ok3
            row["n_strict_axes"] = sum(1 for k in STRICT_AXES if float(row[k]) > float(base[k]))
            row["d"] = deltas(row, base)
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="六项严格口径审计（只读）")
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--gate-dir", default=GATE_DIR)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    led = json.load(open(args.ledger, encoding="utf-8"))
    base = SCA.live_base(led)
    t2c = SCA.t2_cap(base)
    bn = led.get("baseline_v2") or {}
    btrial = int(bn.get("trial") or 0)
    bep = bn.get("epoch")

    print("===== 六项**严格口径**审计（满精度；nano/nano4/F1 须严格 > 基线）=====")
    print(f"基线 = {bn.get('name')} / trial_{btrial} ep{bep}")
    print(f"  nano  {fmt(base['nano'])}   nano4 {fmt(base['nano4'])}   F1 {fmt(base['f1'])}")
    print(f"  T1    {fmt(base['t1'])}   T3    {fmt(base['t3'])}   t2_fp ≤ {t2c}")
    print("  判据：nano/nano4/F1 **严格 >**；T1/T3 **>=**；t2_fp **<=** "
          f"{t2c}　｜　1 阶：nano/nano4 {STEP['nano']:.4g}、F1 {STEP['f1']:.3g}、"
          f"T1 {STEP['t1']:.3g}、T3 {STEP['t3']:.3g}")

    rows = SCA.collect(args.ledger, args.gate_dir)
    print(f"\n有完整六轴读数的 (臂,epoch) = {len(rows)}")

    for r in rows:
        r["npass_ge"] = SCA.judge(r, base)[0]
        fl = strict_flags(r, base, t2c)
        r["strict_ok"] = fl
        r["npass_strict"] = sum(1 for v in fl.values() if v)
        r["miss_strict"] = [k for k, v in fl.items() if not v]
        r["clone"] = is_clone(r, base)
        r["zero_metric"] = five_axis_zero(r, base)
        r["d"] = deltas(r, base)
        r["desc"] = ((led.get("arms") or {}).get(str(r["trial"])) or {}).get("desc") or ""

    # 台账全量扫描（三轴读数，不要求有门禁报告）⇒ C/D 两节共用
    all_arms = ledger_three_axis_strict(led, base)
    gated_keys = {(r["trial"], str(r["ep"])) for r in rows}

    # ── A. 六项严格全过（T1/T3 已门禁）────────────────────────────────────────
    champs = [r for r in rows if r["npass_strict"] == 6]
    champs.sort(key=lambda r: -(r["d"]["nano"]["steps"]))
    print(f"\n===== A. 六项**严格**全过臂：{len(champs)} 个 =====")
    if champs:
        for r in champs[: args.top]:
            d = r["d"]
            print(f"  ✔ trial={r['trial']} ep={r['ep']}　{r['desc'][:90]}")
            print(f"     nano {fmt(r['nano'])} (+{d['nano']['steps']:.2f}阶)　"
                  f"nano4 {fmt(r['nano4'])} (+{d['nano4']['steps']:.2f}阶)　"
                  f"F1 {fmt(r['f1'])} (+{d['f1']['steps']:.2f}阶)")
            print(f"     T1 {fmt(r['t1'])} (+{d['t1']['steps']:.2f}阶)　"
                  f"T2_fp {r['t2_fp']} (Δ{d['t2fp_delta']})　"
                  f"T3 {fmt(r['t3'])} (+{d['t3']['steps']:.2f}阶)")
    else:
        print("  （无）")

    # ── B. `>=` 全过但严格口径不过（三层分类，别把 0 增量臂误读成候选）-------
    ge_all = [r for r in rows if r["npass_ge"] == 6]
    clones = [r for r in ge_all if r["clone"]]
    zero_metric = [r for r in ge_all if not r["clone"] and r["zero_metric"]]
    near_strict = [r for r in ge_all if not r["clone"] and not r["zero_metric"]]
    print(f"\n===== B. `>=` 六项全过臂：{len(ge_all)} 个 =====")
    print(f"  ├ B1 六轴逐位克隆（基线原样）        ：{len(clones)} 个 ⇒ 零增量，不可升轨")
    print(f"  ├ B2 五轴逐位相同、t2_fp 与基线不同   ：{len(zero_metric)} 个 ⇒ **零增量且 T2 更差**，不可升轨")
    print(f"  └ B3 真·近失（至少一轴有真实增量）    ：{len(near_strict)} 个 ⇒ 才值得看")
    near_strict.sort(key=lambda r: -r["npass_strict"])
    for r in near_strict[: args.top]:
        d = r["d"]
        print(f"  △ trial={r['trial']} ep={r['ep']}　严格 {r['npass_strict']}/6　"
              f"差 {r['miss_strict']}")
        print(f"     nano {fmt(r['nano'])} ({d['nano']['steps']:+.2f}阶)　"
              f"nano4 {fmt(r['nano4'])} ({d['nano4']['steps']:+.2f}阶)　"
              f"F1 {fmt(r['f1'])} ({d['f1']['steps']:+.2f}阶)　"
              f"T1 {fmt(r['t1'])} ({d['t1']['steps']:+.2f}阶)　"
              f"T2_fp {r['t2_fp']} (Δ{d['t2fp_delta']})　"
              f"T3 {fmt(r['t3'])} ({d['t3']['steps']:+.2f}阶)")

    # ── C. 三轴严格全过 ⇒ 扣掉已门禁的 ⇒ **真正待门禁**臂池 ------------------
    # 🔴 旧版把 C 建在 `rows`（= SCA.collect()，只含**已有门禁报告**的臂）上
    #    ⇒ `t1 is None or t3 is None` 恒为 False ⇒ 该节**永远输出 0**，是死代码。
    #    正确做法：在**台账全量**（§D 的源）上取三轴严格全过，再扣掉已门禁的。
    tri_full = [r for r in all_arms if r["three_axis_strict"]]
    tri_pending = [r for r in tri_full if (r["trial"], str(r["ep"])) not in gated_keys]
    tri_gated = [r for r in tri_full if (r["trial"], str(r["ep"])) in gated_keys]
    print(f"\n===== C. **三轴**（nano/nano4/F1）严格全过臂（台账全量）：{len(tri_full)} 个　"
          f"｜已门禁 {len(tri_gated)}｜**待门禁 {len(tri_pending)}** =====")
    if not tri_pending:
        print("  （无待门禁臂）")
    for r in sorted(tri_pending, key=lambda r: -(r["d"]["f1"]["steps"]))[: args.top]:
        d = r["d"]
        print(f"  ○ trial={r['trial']} ep={r['ep']}　{r['desc'][:90]}")
        print(f"     nano {fmt(r['nano'])} (+{d['nano']['steps']:.2f}阶)　"
              f"nano4 {fmt(r['nano4'])} (+{d['nano4']['steps']:.2f}阶)　"
              f"F1 {fmt(r['f1'])} (+{d['f1']['steps']:.2f}阶)　"
              f"⇒ T1/T3 **未门禁** ⇒ 需送官方门禁才知道是不是真冠军"
              f"{'　⚠ 台账已记 official（报告文件缺失）' if r.get('has_official') else ''}")

    # ── D. 台账全量扫描（不限有门禁报告）--------------------------------------
    t3_ok = [r for r in all_arms if r["three_axis_strict"]]
    t3_near = sorted(
        (r for r in all_arms if not r["three_axis_strict"]),
        key=lambda r: -r["n_strict_axes"],
    )
    print(f"\n===== D. 台账全量扫描：{len(all_arms)} 个 (臂,epoch) 有三轴读数　"
          f"｜三轴**严格**全过：{len(t3_ok)} 个 =====")
    for r in sorted(t3_ok, key=lambda r: -(r["d"]["f1"]["steps"]))[: args.top]:
        d = r["d"]
        print(f"  ★ trial={r['trial']} ep={r['ep']}　{r['desc'][:90]}")
        print(f"     nano {fmt(r['nano'])} (+{d['nano']['steps']:.2f}阶)　"
              f"nano4 {fmt(r['nano4'])} (+{d['nano4']['steps']:.2f}阶)　"
              f"F1 {fmt(r['f1'])} (+{d['f1']['steps']:.2f}阶)")
    print(f"  -- 三轴里只差一轴的臂（按接近度）--")
    for r in t3_near[: min(args.top, 10)]:
        d = r["d"]
        print(f"     trial={r['trial']} ep={r['ep']}　严格 {r['n_strict_axes']}/3　"
              f"nano {d['nano']['steps']:+.1f}阶 n4 {d['nano4']['steps']:+.1f}阶 "
              f"F1 {d['f1']['steps']:+.1f}阶　{r['desc'][:60]}")

    if args.json_out:
        payload = {
            "baseline": {"name": bn.get("name"), "trial": btrial, "epoch": bep,
                         "values": {k: base[k] for k in ("nano", "nano4", "f1", "t1", "t3")},
                         "t2_fp": base.get("t2_fp"), "t2_cap": t2c},
            "step": STEP,
            "A_strict_champs": champs,
            "B_ge_all": [{"trial": r["trial"], "ep": r["ep"], "clone": r["clone"],
                          "zero_metric": r["zero_metric"],
                          "npass_strict": r["npass_strict"], "miss": r["miss_strict"],
                          "d": r["d"], "desc": r["desc"]} for r in ge_all],
            "C_three_axis_pending_gate": tri_pending,
            "C_three_axis_already_gated": len(tri_gated),
            "D_ledger_three_axis": sorted(t3_ok, key=lambda r: -(r["d"]["f1"]["steps"])),
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        print(f"\n（JSON 产物：{args.json_out}）")


if __name__ == "__main__":
    main()
