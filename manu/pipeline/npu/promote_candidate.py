
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
# 🟢 2026-11-16：加**仓库根** ⇒ `from manu.pipeline.npu import family_key` 可用
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..", "..")))
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
    return None


def polish_family_key(st: dict) -> str:
    """抛光臂的「家族键」= 抛光超参组合（**不含锚点**）。

    理由：抛光轴的可复现性问的是「同一套抛光配方作用在**不同锚点**上，是否反复
    产出六项全过的模型」⇒ 家族按 (lr, λ, freeze_bn, epochs) 分组，n = 用同一套
    抛光配方处理过的**不同锚点个数**。

    ⚠️ 抛光臂的 `init` 是 `epoch*.pt`（不是 base init）⇒ `recipe_stats()` 只收
    `init == BASE_INIT` 的臂 ⇒ 抛光臂**不在**普通配方组里 ⇒ 必须单独建家族。
    """
    p = st.get("params") or {}
    return (f"polish|lr={p.get('lr')}|l2={p.get('anchor_l2')}"
            f"|fb={p.get('freeze_bn')}|ep={p.get('epochs')}")


def polish_family_stats(led: dict) -> dict:
    """抛光家族统计：n（锚点数）＋ n_six_ok（其中六项全过的个数）。"""
    fams: dict[str, dict] = {}
    for k, st in (led.get("arms") or {}).items():
        p = st.get("params") or {}
        # 抛光臂有两种来源：旁路 `polish_nearmiss`（st["polish"]=1）与
        # 调度器的 warm 格（params.warm_key 有值）⇒ 两者都是同一类「抛光配方」，
        # 必须合并统计，否则复现证据被低估。
        is_polish = bool(st.get("polish")) or bool(p.get("warm_key"))
        if not is_polish:
            continue
        key = polish_family_key(st)
        f = fams.setdefault(key, {"key": key, "n": 0, "six_ok": 0,
                                  "trials": [], "anchors": []})
        f["n"] += 1
        f["trials"].append(int(k))
        anc = st.get("source_anchor")
        if anc:
            f["anchors"].append(str(anc))
        if st.get("six_ok"):
            f["six_ok"] += 1
    return fams


def polish_family_key(st: dict) -> str:
    """抛光臂的「家族键」= 抛光超参组合（**不含锚点**）。

    理由：抛光轴的可复现性问的是「同一套抛光配方作用在**不同锚点**上，是否反复
    产出六项全过的模型」⇒ 家族按 (lr, λ, freeze_bn, epochs) 分组，n = 用同一套
    抛光配方处理过的**不同锚点个数**。

    ⚠️ 抛光臂的 `init` 是 `epoch*.pt`（不是 base init）⇒ `recipe_stats()` 只收
    `init == BASE_INIT` 的臂 ⇒ 抛光臂**不在**普通配方组里 ⇒ 必须单独建家族。
    """
    p = st.get("params") or {}
    return (f"polish|lr={p.get('lr')}|l2={p.get('anchor_l2')}"
            f"|fb={p.get('freeze_bn')}|ep={p.get('epochs')}")


def polish_family_stats(led: dict) -> dict:
    """抛光家族统计：n（锚点数）＋ n_six_ok（其中六项全过的个数）。"""
    fams: dict[str, dict] = {}
    for k, st in (led.get("arms") or {}).items():
        p = st.get("params") or {}
        # 抛光臂有两种来源：旁路 `polish_nearmiss`（st["polish"]=1）与
        # 调度器的 warm 格（params.warm_key 有值）⇒ 两者都是同一类「抛光配方」，
        # 必须合并统计，否则复现证据被低估。
        is_polish = bool(st.get("polish")) or bool(p.get("warm_key"))
        if not is_polish:
            continue
        key = polish_family_key(st)
        f = fams.setdefault(key, {"key": key, "n": 0, "six_ok": 0,
                                  "trials": [], "anchors": []})
        f["n"] += 1
        f["trials"].append(int(k))
        anc = st.get("source_anchor")
        if anc:
            f["anchors"].append(str(anc))
        if st.get("six_ok"):
            f["six_ok"] += 1
    return fams


def soup_family_stats(led: dict, st: dict) -> dict | None:
    """🟢 2026-11-17 14:10：**汤臂（synth）的复现性统计**。

    为什么必须补这个（否则汤臂永远升不了轨）：
    · `recipe_stats()` 只收 `init == cs_t0043.pt` 的**真训练臂**（跨模型 soup 不在同一盆地）
    · ⇒ 汤臂（`synth=True`，init = 自己的 ckpt）**不在任何配方组里** ⇒ `g is None`
    · ⇒ `seed_replicated` 永远为 False ⇒ `cmd_promote` 判「未过种子复现关 ⇒ 拒绝升轨」
      ⇒ **今夜辛苦打出来的六项全过汤臂（1106580 等）会在最后一步被卡掉**。
    修法（口径对齐升轨关本身）：
    · 「同一配方」的汤版本 = **同 (parent, 权重) 家族**（`soup_family`，如 `bs640001:w55`）
    · 复现证据 = **该族成员数 ≥ 2**（不同 child ⇒ 不同样本）且 **族均值 nano 严格优于基线族均值**
      （基线 610000 本身也是汤 ⇒ 用**同样的口径**算基线族均值 ⇒ 苹果对苹果）
    """
    fam = st.get("soup_family")
    if not fam:
        return None
    members = []
    for k, s in (led.get("arms") or {}).items():
        if s.get("soup_family") != fam:
            continue
        pk = s.get("picked") or {}
        if pk.get("nano") is None:
            continue
        members.append((int(k), pk))
    if not members:
        return None
    nano = [m[1]["nano"] for m in members]
    nano4 = [m[1].get("nano4") or 0.0 for m in members]
    f1 = [m[1].get("f1") or 0.0 for m in members]
    return {"key": fam, "n": len(members), "trials": sorted(m[0] for m in members),
            "mean_nano": sum(nano) / len(nano), "max_nano": max(nano),
            "mean_nano4": sum(nano4) / len(nano4), "mean_f1": sum(f1) / len(f1)}


def soup_family_six_ok(led: dict, fam: str) -> list:
    """同族里**六项全过**的成员（🔴 以磁盘上的 gate_report.json 为权威，台账写竞争会丢读数）。"""
    try:
        from manu.pipeline.npu import six_criteria_audit as A  # noqa: PLC0415
        six = {int(r["trial"]) for r in A.collect() if (A.judge(r)[0] == 6)}
    except Exception:  # noqa: BLE001
        six = set()
    out = []
    for k, st in (led.get("arms") or {}).items():
        if st.get("soup_family") != fam:
            continue
        try:
            t = int(k)
        except Exception:  # noqa: BLE001
            continue
        if t in six:
            out.append(t)
    return sorted(out)


def check_arm(led: dict, base: dict, trial: int) -> dict:
    st = (led.get("arms") or {}).get(str(trial)) or {}
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    _ep = pk.get("epoch")
    if _ep is None:
        # 🟢 2026-11-17 14:15：汤臂的 picked 可能缺 epoch ⇒ 兜底 5（汤统一存 epoch5.pt）
        _ep = 5 if (st.get("synth") or os.path.isfile(os.path.join(
            ns.ROOT, "trials", f"trial_{trial}", "weights", "epoch5.pt"))) else 5
    res = {"trial": trial, "desc": st.get("desc", ""), "epoch": _ep,
           "nano": pk.get("nano"), "nano4": pk.get("nano4"), "f1": pk.get("f1"),
           "micro": pk.get("micro"), "t1": of.get("t1"), "t2_fp": of.get("t2_fp"),
           "t3": of.get("t3"), "board_ms": st.get("board_ms"), "checks": {}, "ok": False,
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
    # 板端时延
    # 🟢🟢 2026-11-17 16:45 修**假检查**：原来 c["ms"] 是在 res["ok"] **之后**才写进 checks
    #   ⇒ 板端超线的臂照样判 ok=True ⇒ 板端红线形同虚设。
    #   修法：先算板端检查，再算 res["ok"]（有板端读数 ⇒ 时延进判定；没读数 ⇒ 保持原语义 + 警告）。
    ref = board_ref_ms(led)
    if ref:
        res["board_ref_ms"] = ref
        if res["board_ms"] is not None:
            c["ms"] = res["board_ms"] <= ref * 1.02
        else:
            res["warnings"].append("未测板端时延（arm 未进板卡会话）")
    else:
        res["warnings"].append("台账里没有 0031 的板端参照")
    res["ok"] = bool(c) and all(c.values())
    # 种子复现
    groups = recipe_stats(led)
    g = groups.get(recipe_key(st.get("params") or {}))
    if g is None and st.get("polish"):
        # 🔴 抛光臂不在 recipe_stats 里（init ≠ BASE_INIT）⇒ 单独按「抛光家族」判种子关
        pf = polish_family_stats(led).get(polish_family_key(st))
        if pf:
            n_anchor = len(set(pf["anchors"]))
            res["polish_family"] = {"n": pf["n"], "n_anchor": n_anchor,
                                    "six_ok": pf["six_ok"], "key": pf["key"]}
            res["seed_replicated"] = bool(n_anchor >= 2)
            res["seed_note"] = (
                f"抛光家族 {pf['key']}：n={pf['n']}（不同锚点 {n_anchor} 个）"
                f"，其中六项全过 {pf['six_ok']} 个 ⇒ "
                + ("已复现（同一抛光配方作用在 ≥2 个锚点上）"
                   if res["seed_replicated"]
                   else "未复现（同一抛光配方目前只作用在 1 个锚点上 ⇒ 需 ≥2 个）"))
    if g is None and st.get("polish"):
        # 🔴 抛光臂不在 recipe_stats 里（init ≠ BASE_INIT）⇒ 单独按「抛光家族」判种子关
        pf = polish_family_stats(led).get(polish_family_key(st))
        if pf:
            n_anchor = len(set(pf["anchors"]))
            res["polish_family"] = {"n": pf["n"], "n_anchor": n_anchor,
                                    "six_ok": pf["six_ok"], "key": pf["key"]}
            res["seed_replicated"] = bool(n_anchor >= 2)
            res["seed_note"] = (
                f"抛光家族 {pf['key']}：n={pf['n']}（不同锚点 {n_anchor} 个）"
                f"，其中六项全过 {pf['six_ok']} 个 ⇒ "
                + ("已复现（同一抛光配方作用在 ≥2 个锚点上）"
                   if res["seed_replicated"]
                   else "未复现（同一抛光配方目前只作用在 1 个锚点上 ⇒ 需 ≥2 个）"))
    bg = None
    a98 = (led.get("arms") or {}).get(str(ns.BASE_V2["trial"]), {})
    bg = groups.get(recipe_key(a98.get("params") or {}))
    # 🟢 2026-11-17 14:10：**汤臂的复现关**（否则永远拒绝升轨，见 soup_family_stats 注释）
    if g is None and st.get("synth"):
        sf = soup_family_stats(led, st)
        if sf:
            res["soup_family"] = sf
            bst = (led.get("arms") or {}).get(str(ns.BASE_V2["trial"]), {}) or {}
            bsf = soup_family_stats(led, bst)
            bmean = (bsf or {}).get("mean_nano", base.get("nano"))
            six_ok = soup_family_six_ok(led, sf["key"])
            res["soup_family"]["six_ok"] = six_ok
            res["seed_replicated"] = bool(len(six_ok) >= 2)
            res["seed_note"] = (
                f"汤族 {sf['key']}：n={sf['n']}（成员 {sf['trials'][:8]}），"
                f"族均值 nano={sf['mean_nano']:.6f} vs 基线族均值 {bmean:.6f}；"
                f"**同族六项全过 {len(six_ok)} 个 {six_ok}** ⇒ "
                + ("已复现（同一 parent×权重 作用在 ≥2 个 child 上，**且 ≥2 个成员六项全过**）"
                   if res["seed_replicated"] else "未复现（同族六项全过的成员 <2 ⇒ 单臂彩票）"))
    if g and bg:
        res["recipe_n"] = g["n"]
        res["recipe_mean_nano"] = g["mean_nano"]
        res["base_recipe_mean_nano"] = bg["mean_nano"]
        res["seed_replicated"] = g["n"] >= 2 and g["mean_nano"] > bg["mean_nano"]
        if not res["seed_replicated"]:
            res["seed_note"] = (f"配方 n={g['n']} 均值 nano={g['mean_nano']:.6f} "
                                f"vs 基线配方均值 {bg['mean_nano']:.6f} ⇒ 未复现")
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


def cmd_promote(led: dict, base: dict, trial: int, name: str, force: bool, dry: bool,
                require_board: bool = False) -> None:
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
    # 🔴 2026-11-17 16:45：`auto_promote_watch` 一直带 `--require-board` 调本脚本，
    #   而本脚本**根本没有这个参数** ⇒ argparse 直接 usage error ⇒ **自动升轨永远失败**
    #   （昨夜 11:17 卡死过一次，今夜若不修则一整晚的票全废）。
    #   ⇒ 这里补上：候选臂必须有板端 T4 读数，且 ≤ 19.989 ms 硬红线。
    if require_board:
        bms = r.get("board_ms")
        if bms is None:
            print("  ⇒ --require-board：候选臂**没有板端 T4 读数** ⇒ **拒绝升轨**")
            return
        if float(bms) > 19.989:
            print(f"  ⇒ --require-board：板端 {bms} ms > 19.989 ms 硬红线 ⇒ **拒绝升轨**")
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
                    help="强制要求候选臂有板端 T4 读数且 ≤ 19.989 ms 硬红线")
    args = ap.parse_args()
    led = ns.load_ledger()
    base = base_ref(led)
    if args.list or args.trial is None:
        cmd_list(led, base)
    if args.trial is not None:
        cmd_promote(led, base, args.trial, args.name, args.force, args.dry_run, args.require_board)


if __name__ == "__main__":
    main()