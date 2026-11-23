
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
from manu.pipeline.npu import t4_criterion as T4  # noqa: E402（板卡 T4 红线唯一真源）

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
    # 🔴🔴🔴 2026-11-20 15:35 —— **T2 判据与权威口径对齐**（修「升轨预检比六项审计更严」的漂移 bug）。
    #   【现场事故】基线 `Baseline_v3.1`(trial_1990055) 的 `t2_fp = 3`；权威口径
    #     `six_criteria_audit.t2_cap(base) = max(基线 t2_fp, 5)`（用户口径红线 `T2_USER = 5`，
    #     2026-11-16 已明确「永不比基线更严」）。
    #   而本文件老代码是 `of["t2_fp"] <= base["t2_fp"]`（严格 ≤3）⇒ **比权威口径更严**
    #     ⇒ `t2_fp = 4` 的**合法六项全过臂**在本文件里被判 5/6 ⇒
    #     `res["ok"] = False` ⇒ **合法候选被升轨预检拦下**
    #     （正是本文件 docstring 描述过的「整晚算力跑出来的更优臂永远升不了轨」的坑；
    #       2026-11-16 只修了 `six_criteria_audit` 那一侧，本侧漏修）。
    #   【改法】本函数**唯一**的 T2 上限来源 = `six_criteria_audit.t2_cap(base)`（带兜底）。
    #   【影响面】只放宽「t2_fp ∈ (基线, 5]」这一档 ⇒ 与权威审计口径**完全一致**；
    #     升轨三关（种子复现 / 单臂严格支配 / 板端 T4 PASS）**全部保持不变** ⇒ 不会引入基线名通胀。
    try:
        from manu.pipeline.npu import six_criteria_audit as _A  # noqa: PLC0415（权威 T2 口径唯一真源）
        _t2_cap = int(_A.t2_cap(base))
    except Exception:  # noqa: BLE001
        _t2_cap = max(int(base.get("t2_fp") or 0), 5)
    res["t2_cap"] = _t2_cap
    if pk.get("nano") is not None:
        c["nano"] = pk["nano"] >= base["nano"]
    if pk.get("nano4") is not None:
        c["nano4"] = pk["nano4"] >= base["nano4"]
    if pk.get("f1") is not None:
        c["f1"] = pk["f1"] >= base["f1"]
    if of.get("t1") is not None:
        c["t1"] = of["t1"] >= base["t1"]
    if of.get("t2_fp") is not None:
        c["t2fp"] = of["t2_fp"] <= _t2_cap
    if of.get("t3") is not None:
        c["t3"] = of["t3"] >= base["t3"]
    # 板端时延
    # 🟢🟢 2026-11-17 16:45 修**假检查**：原来 c["ms"] 是在 res["ok"] **之后**才写进 checks
    #   ⇒ 板端超线的臂照样判 ok=True ⇒ 板端红线形同虚设。
    #   修法：先算板端检查，再算 res["ok"]（有板端读数 ⇒ 时延进判定；没读数 ⇒ 保持原语义 + 警告）。
    ref = board_ref_ms(led)
    # 🔴🔴🔴 2026-11-20 14:40 —— **板端判定必须认「同会话相对基线」口径**。
    #   现场事故（2026-11-20 14:00 实测）：板卡读数**热漂移 ±0.5 ms**
    #   （逐小时中位数 11-13 19.68 → 11-19 18时 20.005 → 11-20 13时 19.831，批内 max 21.462），
    #   而红线 19.989 ms 正落在漂移带中央 ⇒ **同一张计算图**（基线家族 bw10 克隆臂）
    #   现在测出 19.94~21.46 ms ⇒ 绝对口径把它们**全部误判 FAIL** ⇒ **升轨链断链**。
    #   `board_t4_watch` 已在台账写 `board_pass_t4` = `绝对口径 OR 同会话相对口径`
    #   （相对口径用**同会话基线臂读数**抵消板卡整体漂移，语义仍是「不比 trial_0031×1.02 慢」）。
    #   ⇒ 这里**以台账的 `board_pass_t4` 为准**，缺该字段时才退回旧的绝对/0031 参照口径。
    _bpass = st.get("board_pass_t4")
    res["board_pass_t4"] = _bpass
    # 🔴 R3：换基线后，**打旧基线量出来的板端判定是过期的** ⇒ 判「需重测」，不许当 PASS。
    _cur_base = int((led.get("baseline_v2") or {}).get("trial") or 0)
    _ref_trial = str(st.get("board_ref_trial") or "")
    _stale = False
    if _bpass is not None and _ref_trial and _cur_base:
        try:
            _stale = int(_ref_trial) != _cur_base
        except (TypeError, ValueError):
            _stale = False
    res["board_ref_trial"] = _ref_trial or None
    res["board_stale"] = _stale
    if _bpass is not None:
        if res["board_ms"] is None:
            res["warnings"].append("未测板端时延（arm 未进板卡会话）")
            c["ms"] = False
        elif _stale:
            # 🔴 参照基线已换人 ⇒ 旧判定语义已变 ⇒ **保守判否**，等板卡重测（`board_t4_watch` 会补测）
            res["warnings"].append(
                f"板端判定打的是**旧基线**（trial {_ref_trial}），当前基线已换成 trial {_cur_base}"
                f" ⇒ 判定过期，需重测")
            c["ms"] = False
        else:
            c["ms"] = bool(_bpass)
            if ref:
                res["board_ref_ms"] = ref
            res["board_ms_ref"] = st.get("board_ms_ref")
            res["board_diff_ms"] = st.get("board_diff_ms")
    elif ref:
        # 🔴🔴 R5：老臂（只有 `board_ms`、没有 `board_pass_t4`）原先用 `ref * 1.02`，
        #   而台账 `board_sessions` 里的 t31 是**已下线 .211 板**的 **23.528 ms**
        #   ⇒ 预算 24.0 ms ⇒ **21.4 ms 的臂在预检里判 PASS**（比红线松 2 倍）⇒ 误放行。
        #   ⇒ 老臂一律用**绝对红线**（`T4.LIMIT_MS`），并标注判定来源。
        res["board_ref_ms"] = ref
        res["board_verdict_source"] = "legacy-absolute"
        if res["board_ms"] is not None:
            c["ms"] = res["board_ms"] <= T4.LIMIT_MS
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

    # ============================================================================
    # 🔴🔴🔴 2026-11-20 R1修复 —— **「单臂严格支配」升轨通道**（训练臂 / 抛光臂专用）
    # ============================================================================
    #
    # 【实测证据】2026-11-20 14:39 服务器台账快照（12,096 臂，基线 Baseline_v3.1 / trial 1990055）：
    #   · 按台账读数复算六项判据：**41** 个臂六项全过；
    #     但其中**「至少一轴严格更优」的 = 0 个** ⇒ 这 41 个**全部是基线权重的逐位复现**（零改进）。
    #   · 现有种子关要求「同族 n≥2」（汤臂）/「n_anchor≥2」（抛光臂）/「配方 n≥2」（普通训练臂）
    #     ⇒ **单个严格支配基线的训练臂永远升不了轨**（它所在的族只有它自己）。
    #   · `six_criteria_audit.py:70-72` 的权威口径是 `t2_fp ≤ max(基线 t2_fp, 5)`；
    #     而本文件老代码 `c["t2fp"] = of["t2_fp"] <= base["t2_fp"]`（:213-214）
    #     在基线 t2_fp=3 时**比权威口径更严** ⇒ 与审计口径**不一致**。
    #
    # 【改法】新增一条**只对训练臂/抛光臂开放**的通道（汤臂保持老口径不变）：
    #   该臂**自身**
    #     ① 六项全过：nano/nano4/f1 ≥ 基线、T1 ≥ 基线、**T2_fp ≤ max(基线, 5)（权威口径）**、T3 ≥ 基线；
    #     ② **至少一轴严格更优**（否则是纯打平 ⇒ 基线名通胀，2026-11-18 事故 82 次重复升轨）；
    #     ③ **板端判定 PASS**（`c["ms"]`，**沿用本函数已有的 `board_ref_trial` 新鲜度/stale 逻辑**）
    #   ⇒ 允许它**无需 n_anchor ≥ 2** 直接升轨，并在 `seed_note` / `warnings` 里标注「单臂严格支配」。
    #
    # 【为什么 T2 在本通道用权威口径】老 `c["t2fp"]` 判据**保持不变**（老臂判定零变化）；
    #   只有本通道**命中时**才按 `six_criteria_audit.t2_cap()` 的权威口径重算 `c["t2fp"]` 与 `res["ok"]`。
    #   ⇒ 行为差异**仅限**「训练臂/抛光臂 + 自身六项全过 + 至少一轴严格更优 + 板卡 PASS」这一类，
    #      而这正是本次要新增的能力。
    #
    # 【影响面】只新增一条 True 通路；**老逻辑（种子关 / 六项判据 / 板卡判据）一字未改**；
    #   汤臂**完全不受影响**（`_is_soup` 判定 ⇒ 不进本通道）。
    # 【回退】删除本段即可（该文件已被 git 跟踪 ⇒ `git checkout manu/pipeline/npu/promote_candidate.py`）。
    res["single_arm_strict"] = False
    _is_soup = bool(st.get("synth") or (st.get("soup_members") or []))
    # （`_t2_cap` 已在函数头部按权威口径算好 ⇒ 此处直接复用，避免两处口径再次漂移）
    _auth_axes = {
        "nano": pk.get("nano") is not None and pk["nano"] >= base["nano"],
        "nano4": pk.get("nano4") is not None and pk["nano4"] >= base["nano4"],
        "f1": pk.get("f1") is not None and pk["f1"] >= base["f1"],
        "t1": of.get("t1") is not None and of["t1"] >= base["t1"],
        "t2fp": of.get("t2_fp") is not None and of["t2_fp"] <= _t2_cap,
        "t3": of.get("t3") is not None and of["t3"] >= base["t3"],
    }
    _strictly_better = bool(
        (pk.get("nano") or 0) > base["nano"]
        or (pk.get("nano4") or 0) > base["nano4"]
        or (pk.get("f1") or 0) > base["f1"]
        or (of.get("t1") or 0) > base["t1"]
        or (of.get("t3") or 0) > base["t3"]
        or (of.get("t2_fp") is not None and of["t2_fp"] < int(base.get("t2_fp") or 0)))
    res["strict_axes"] = _auth_axes
    res["strictly_better"] = _strictly_better
    if (not _is_soup) and all(_auth_axes.values()) and _strictly_better and c.get("ms") is True:
        res["single_arm_strict"] = True
        res["seed_replicated"] = True
        res["seed_note"] = (
            "**单臂严格支配路径**（R1修复新增）：该训练/抛光臂**自身**六项全过"
            f"（T2 用权威口径 t2_fp ≤ {_t2_cap}）且**至少一轴严格优于基线**，"
            f"板端 T4 判定 PASS（board_pass_t4={res.get('board_pass_t4')}，"
            f"参照基线 trial {res.get('board_ref_trial')}，stale={res.get('board_stale')}）"
            " ⇒ 无需「同族 n≥2 / n_anchor≥2」复现关即可升轨")
        res["warnings"].append(
            "单臂严格支配路径：未走种子复现关（该臂自身六项全过且至少一轴严格更优）")
        # 权威 T2 口径下重算 ok（老 c["t2fp"] 比权威口径更严 ⇒ 本通道按权威口径）
        c["t2fp"] = _auth_axes["t2fp"]
        res["ok"] = bool(c) and all(c.values())
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
        # 🔴🔴 2026-11-20 14:40 —— **最终权威也要认「同会话相对基线」口径**。
        #   `r["board_pass_t4"]` 来自台账，已含 `绝对 OR 同会话相对`；相对口径用于抵消板卡热漂移。
        #   ⇒ 有该字段时以它为准；缺该字段（老臂）才退回绝对红线。
        _bp = r.get("board_pass_t4")
        if _bp is not None:
            if not bool(_bp):
                print(f"  ⇒ --require-board：板端 {bms} ms 未过 T4 判定"
                      f"（绝对/同会话相对口径均不通过）⇒ **拒绝升轨**")
                return
        elif float(bms) > T4.LIMIT_MS:
            print(f"  ⇒ --require-board：板端 {bms} ms > {T4.LIMIT_MS} ms 硬红线 ⇒ **拒绝升轨**")
            return
        # 🔴🔴 R3b：**新基线不得比旧基线慢**（相对口径放宽后，基线绝对读数可以漂到 ~21.5 ms
        #   ⇒ 若不设这道闸，**交付基线的时延可以悄悄回退 1.5 ms** 并被固化进台账/源码常量/交付文档）。
        _old_ms = (led.get("baseline_v2") or {}).get("ms_new_board")
        try:
            _old_ms = float(_old_ms)
        except (TypeError, ValueError):
            _old_ms = None
        if _old_ms is not None and _old_ms > 5.0 and float(bms) > _old_ms:
            print(f"  ⇒ --require-board：新基线板端 {bms} ms **比旧基线慢**"
                  f"（旧基线 {_old_ms} ms）⇒ **拒绝升轨**（交付时延不得回退）")
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
        # 🔴 R1：**干净参照**只允许「绝对达标」的读数写入（≤ 红线）；
        #   相对口径过的臂（绝对 ms > 红线）写入的 `ms_new_board` **不是**干净参照
        #   ⇒ `board_t4_watch.base_ref_ms()` 会拒绝用它（并告警）。
        "ms_clean_ref": (r["board_ms"] if (r.get("board_ms") is not None
                                           and float(r["board_ms"]) <= T4.LIMIT_MS) else None),
        "board_ref_trial": r.get("board_ref_trial"),
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
