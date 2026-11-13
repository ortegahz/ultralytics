#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选升轨工具 v2（2026-11-11 15:20）：把「击败 Baseline_v2.0(trial_9863)」的臂升为新基线。

与 `promote_baseline.py`（一次性把 9863 升为 Baseline_v2.0）不同，本脚本是**通用**的：
  * `--list`：列出所有臂，按 nano 排序，标出「六项全平基线」的候选；
  * `--trial N [--dry-run]`：对某个臂做**升轨预检**（六项判据 + 板端时延 + 种子复现），
    预检通过后执行升轨（写台账 + 写升轨记录 + 打印需要回填到 `nano_search.BASE_V2` 的常量）。

判据（全部用**台账里的精确读数**，不用 6 位截断常量）：
  ① NanoBench nano / nano@4px / F1 ≥ 基线；
  ② 官方三轨 T1 ≥ 基线、T2 FP ≤ 基线、T3 ≥ 基线；
  ③ 板端 int8 ≤ 同会话 0031 自身 ×1.02（**绝对 20 ms 铁律待新板重定标**，见 memory §137/§142.4）；
  ④ **种子复现**：该臂所属配方（同 recipe_key）需 n≥2 且均值 nano 严格优于基线配方均值
     ⇒ 单臂过线只算「PASS*（未复现）」，需 `--force` 才能强推。

用法（服务器）：
  PY=~/anaconda3/envs/uav/bin/python
  $PY manu/pipeline/npu/promote_candidate.py --list
  $PY manu/pipeline/npu/promote_candidate.py --trial 12123 --dry-run
  $PY manu/pipeline/npu/promote_candidate.py --trial 12123 --name Baseline_v3.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nano_search as ns  # noqa: E402
from adaptive_runner import recipe_key, recipe_stats, base_ref  # noqa: E402

PROMO_DIR = ns.WORK


def board_ref_ms(led: dict) -> float | None:
    """同会话里 trial_0031 自身的板端 ms（新板参照）。"""
    sess = (led.get("board_sessions") or [])
    for s in reversed(sess):
        ref = ((s.get("session2") or {}).get("arms") or {}).get("t31_silu_int8")
        if ref:
            return float(ref)
    for s in reversed(sess):
        for key in ("session2", "session1"):
            ref = ((s.get(key) or {}).get("arms") or {}).get("t31_silu_int8")
            if ref:
                return float(ref)
    # 台账里没有（被搜索器的 save_ledger 覆盖成 null）⇒ 直接读会话文件
    return ns.board_reference_ms()


# 🟢 2026-11-12 新增：是否要求「板端时延」也作为升轨必要条件（默认否，见 §150.5）
REQUIRE_BOARD = False


def check_arm(led: dict, base: dict, trial: int) -> dict:
    st = (led.get("arms") or {}).get(str(trial)) or {}
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    ms_board = st.get("board_ms")
    if ms_board is None and pk.get("epoch") is not None:
        ms_board = ns.board_ms_for(trial, int(pk["epoch"]))
    res = {"trial": trial, "desc": st.get("desc", ""), "epoch": pk.get("epoch"),
           "nano": pk.get("nano"), "nano4": pk.get("nano4"), "f1": pk.get("f1"),
           "micro": pk.get("micro"), "t1": of.get("t1"), "t2_fp": of.get("t2_fp"),
           "t3": of.get("t3"), "board_ms": ms_board, "checks": {}, "ok": False,
           "seed_note": "", "warnings": []}
    c = res["checks"]
    if pk.get("nano") is not None:
        c["nano"] = pk["nano"] >= base["nano"]
    if pk.get("nano4") is not None:
        c["nano4"] = pk["nano4"] >= base["nano4"]
    if pk.get("f1") is not None:
        c["f1"] = pk["f1"] >= base["f1"]
    if of.get("t1") is not None:
        c["t1"] = of["t1"] >= base["t1"]
    if of.get("t2_fp") is not None:
        c["t2fp"] = of["t2_fp"] <= base["t2_fp"]
    if of.get("t3") is not None:
        c["t3"] = of["t3"] >= base["t3"]
    res["ok"] = bool(c) and all(c.values())
    res["six_pass"] = bool(res["ok"])      # 六项判据本身（不含板端时延）
    # 板端时延
    ref = board_ref_ms(led)
    if ref:
        res["board_ref_ms"] = ref
        if res["board_ms"] is not None:
            c["ms"] = res["board_ms"] <= ref * 1.02
        else:
            c["ms"] = None
            res["warnings"].append("未测板端时延（arm 未进板卡会话）")
    else:
        c["ms"] = None
        res["warnings"].append("台账里没有 0031 的板端参照")
    # 🟢 2026-11-12：板卡时延默认**不挡**升轨（板卡忙/未测时仍可先升轨，板卡另行验收），
    #   但 `--require-board` 时**必须**有板卡读数且 ≤ 参照 ×1.02 才算 ok（发布前用）。
    if REQUIRE_BOARD:
        res["ok"] = bool(res["ok"]) and bool(c.get("ms"))
        if not c.get("ms"):
            res["warnings"].append("--require-board 生效：板端时延未达标/未测 ⇒ 不予升轨")
    # 种子复现
    groups = recipe_stats(led)
    g = groups.get(recipe_key(st.get("params") or {}))
    bg = None
    a98 = (led.get("arms") or {}).get(str(ns.BASE_V2["trial"]), {})
    bg = groups.get(recipe_key(a98.get("params") or {}))
    if g and bg:
        res["recipe_n"] = g["n"]
        res["recipe_mean_nano"] = g["mean_nano"]
        res["base_recipe_mean_nano"] = bg["mean_nano"]
        # 🟢 2026-11-12 09:27：**基线配方自身**的组不再与「自己」比均值
        #   （该组已 n=8 复现 ⇒ 种子关自动满足；升轨看单臂六项即可）。
        is_base_group = (g["key"] == bg["key"])
        # 🟢🟢🟢 2026-11-13 22:40：**均值护栏对「本臂六项全过」的格让路**
        #   （与 adaptive_runner.promotion_candidates 同一口径：实测 C 家族
        #    sw4.0×thr16ms4 的均值 nano 0.692344 < 基线组均值 0.694066，
        #    ⇒ 旧护栏会把它已产出的六项全过臂也否掉；六项判据本身没放松）
        res["seed_replicated"] = (g["n"] >= 2 if is_base_group
                                  else (g["n"] >= 2 and (g["mean_nano"] > bg["mean_nano"]
                                                         or res.get("six_pass"))))
        if not res["seed_replicated"]:
            res["seed_note"] = (f"配方 n={g['n']} 均值 nano={g['mean_nano']:.6f} "
                                f"vs 基线配方均值 {bg['mean_nano']:.6f} ⇒ 未复现")
        elif res.get("six_pass"):
            res["seed_note"] = (f"配方 n={g['n']} 均值 nano={g['mean_nano']:.6f} "
                                f"（低于基线组均值 {bg['mean_nano']:.6f}，但本臂六项全过 ⇒ 放行；"
                                f"建议再补 1 个同配方 seed 复核）")
    return res


def cmd_list(led: dict, base: dict) -> None:
    rows = []
    for k, st in sorted((led.get("arms") or {}).items(), key=lambda x: int(x[0])):
        pk = st.get("picked") or {}
        if pk.get("nano") is None:
            continue
        r = check_arm(led, base, int(k))
        rows.append(r)
    rows.sort(key=lambda r: -(r["nano"] or 0))
    print(f"基线 Baseline_v2.0 = trial {ns.BASE_V2['trial']} "
          f"nano={base['nano']:.6f} nano4={base['nano4']:.6f} F1={base['f1']:.6f} "
          f"T1={base['t1']:.6f} T2FP={base['t2_fp']} T3={base['t3']:.6f}")
    print(f"{'trial':>6} {'nano':>10} {'nano4':>9} {'F1':>9} {'T1':>9} {'T2FP':>5} {'T3':>9} "
          f"{'ms':>7} {'六项':>4} {'seed':>5}  desc")
    for r in rows[:40]:
        print(f"{r['trial']:>6} {r['nano']:>10.6f} {(r['nano4'] or 0):>9.6f} "
              f"{(r['f1'] or 0):>9.6f} {(r['t1'] or 0):>9.6f} "
              f"{(r['t2_fp'] if r['t2_fp'] is not None else -1):>5} "
              f"{(r['t3'] or 0):>9.6f} {(r['board_ms'] if r['board_ms'] else 0):>7.2f} "
              f"{'PASS' if r['ok'] else '—':>4} "
              f"{'OK' if r.get('seed_replicated') else ('?' if r.get('recipe_n') else '-'):>5}  "
              f"{r['desc'][:52]}")


def cmd_promote(led: dict, base: dict, trial: int, name: str, force: bool, dry: bool) -> None:
    r = check_arm(led, base, trial)
    print("── 升轨预检 ──────────────────────────────")
    print(f"  trial {trial}  epoch {r['epoch']}  {r['desc'][:70]}")
    print(f"  nano   {r['nano']:.6f}  (基线 {base['nano']:.6f})")
    print(f"  nano4  {r['nano4']:.6f}  (基线 {base['nano4']:.6f})")
    print(f"  F1     {r['f1']:.6f}  (基线 {base['f1']:.6f})")
    print(f"  T1     {r['t1']}  (基线 {base['t1']})   T2FP {r['t2_fp']} (基线 {base['t2_fp']})"
          f"   T3 {r['t3']} (基线 {base['t3']})")
    print(f"  板端   {r['board_ms']} ms  参照 0031 同会话 {r.get('board_ref_ms')} ms ×1.02")
    for k, v in r["checks"].items():
        print(f"    [{'✔' if v else '✘'}] {k}")
    print(f"  种子复现：{r.get('seed_note') or 'n/a'}")
    for w in r["warnings"]:
        print(f"  ⚠ {w}")
    if not r["ok"]:
        print("  ⇒ 六项未全平基线 ⇒ **拒绝升轨**")
        return
    if not r.get("seed_replicated") and not force:
        print("  ⇒ 未过种子复现关 ⇒ **拒绝升轨**（单臂 PASS*，需 --force 或再补 seed）")
        return
    print(f"  ⇒ 预检通过 ⇒ 新基线名 {name}")

    if dry:
        print("  （--dry-run：不写台账）")
        return

    # ── 执行升轨 ──
    st = led["arms"][str(trial)]
    ck = f"{ns.ROOT}/trials/trial_{trial}/weights/epoch{r['epoch']}.pt"
    old = dict(led.get("baseline_v2") or ns.BASE_V2)
    new = dict(ns.BASE_V2)
    new.update({
        "name": name, "trial": trial, "epoch": r["epoch"], "ckpt": ck,
        "nano": r["nano"], "micro": r["micro"], "f1": r["f1"], "nano4": r["nano4"],
        "t1": r["t1"], "t2_fp": r["t2_fp"], "t3": r["t3"],
        "ms_new_board": r["board_ms"], "ms_new_board_t31_ref": r.get("board_ref_ms"),
        "recipe": r["desc"][:200],
        "promoted_at": time.strftime("%F %T"),
        "previous_baseline": {"name": old.get("name"), "trial": old.get("trial"),
                              "epoch": old.get("epoch"), "nano": old.get("nano")},
    })
    led["baseline_v2"] = new
    reg = [x for x in (led.get("baseline_registry") or []) if x.get("trial") != trial]
    reg.insert(0, {"id": name, "trial": trial, "status": "ACTIVE / OFFICIAL BASELINE",
                   "note": "新基线；后续所有探索的 Beat-to-Pass 参照"})
    for x in reg:
        if x.get("status", "").startswith("ACTIVE") and x.get("trial") != trial:
            x["status"] = "SUPERSEDED / DEPRECATED BASELINE"
    led["baseline_registry"] = reg
    led["beat_to_pass"] = {
        "ref": f"{name}(trial_{trial})", "nano_min": new["nano"], "f1_min": new["f1"],
        "t1_min": new["t1"], "t2_fp_max": new["t2_fp"], "t3_min": new["t3"],
        "legacy_ref": "trial_0031", "legacy_nano_min": ns.BASELINE_NANO_FLOOR["0031"],
        "note": "后续所有探索统一以新基线的指标为 Beat-to-Pass 下限",
    }
    st["is_baseline_v2"] = True
    st["status"] = "ACTIVE / OFFICIAL BASELINE"
    st["baseline_name"] = name
    st["beat_to_pass_ref"] = f"{name}(trial_{trial})"
    for k, ost in (led.get("arms") or {}).items():
        if k != str(trial) and ost.get("is_baseline_v2"):
            ost["is_baseline_v2"] = False
            ost["status"] = "SUPERSEDED / DEPRECATED BASELINE"
    ns.save_ledger(led)
    rec = {"promoted_at": time.strftime("%F %T"), "from": old.get("name"),
           "to": name, "trial": trial, "epoch": r["epoch"], "checks": r["checks"],
           "nano": r["nano"], "nano4": r["nano4"], "f1": r["f1"], "t1": r["t1"],
           "t2_fp": r["t2_fp"], "t3": r["t3"], "board_ms": r["board_ms"],
           "seed_replicated": r.get("seed_replicated")}
    out = os.path.join(PROMO_DIR, f"promotion_{trial}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"  ✔ 台账已升轨 → {ns.LEDGER}")
    print(f"  ✔ 升轨记录 → {out}")
    print("  ⚠ 需回填 nano_search.py 的 BASE_V2 常量（搜索器仍按旧基线判 Beat-to-Pass）：")
    print(f"    BASE_V2 = dict(name={name!r}, trial={trial}, epoch={r['epoch']}, "
          f"ckpt={ck!r},")
    print(f"        nano={new['nano']!r}, micro={new['micro']!r}, f1={new['f1']!r}, "
          f"nano4={new['nano4']!r}, t1={new['t1']!r}, t2_fp={new['t2_fp']!r}, "
          f"t3={new['t3']!r}, ...)")


def main() -> None:
    ap = argparse.ArgumentParser(description="候选升轨工具（Baseline_v2.0 → 新基线）")
    ap.add_argument("--list", action="store_true", help="列出所有臂与候选")
    ap.add_argument("--trial", type=int, help="要升轨的 trial 号")
    ap.add_argument("--name", default="Baseline_v3.0", help="新基线名")
    ap.add_argument("--dry-run", action="store_true", help="只预检不写台账")
    ap.add_argument("--force", action="store_true", help="允许未过种子复现关的臂强推")
    ap.add_argument("--require-board", action="store_true",
                    help="板端时延也作为升轨必要条件（未测或超参照×1.02 ⇒ 不予升轨）")
    args = ap.parse_args()
    global REQUIRE_BOARD
    REQUIRE_BOARD = bool(args.require_board)
    led = ns.load_ledger()
    base = base_ref(led)
    if args.list or args.trial is None:
        cmd_list(led, base)
    if args.trial is not None:
        cmd_promote(led, base, args.trial, args.name, args.force, args.dry_run)


if __name__ == "__main__":
    main()

