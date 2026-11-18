#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""血缘汤**批次轴**（§196 追加6）—— 零训练成本，把「同家族 n≥4」升轨硬门槛顶满。

为什么必须做这一轴（11-17 上午实测，不是猜测）
------------------------------------------------
· 家族键口径：`soup:<盆地>:<标签><成员数>` ⇒ **同一个 parent + 同一种加权方式 + 2 个成员**
  ⇒ 只要有 **4 个不同 child** 与同一个 parent 做同权重汤，就凑齐「同家族 n≥4」。
· 实测「亲子汤」比成员**更好**：`1105000` = 98905@5(sw4.0) ⊕ 1100275@1(sw4.2) 等权
  ⇒ nano .721408 / nano4 .655718 / F1 .890615 **三轴全过**（两个成员都没这么高）
  ⇒ 「汤 ≠ 成员加权均值」的直觉成立（平均相当于正则化的集成效应）。
· 官方门禁 T1 = .890596（≥ 基线 .8904939）⇒ **只剩 T2 虚警 / T3 两项未知**。

做法
----
1. 读台账 ⇒ 找 parent（锚点）与它的 child（child 的 `params.init` = parent 的 ckpt）；
2. 按「三轴最小边际 σ」给 child 排序 ⇒ 取前 K 个；
3. 对 (parent, child, w) 三元组做**权重平均** ⇒ 落 `trial_<sid>/weights/epoch5.pt` + sidecar；
4. 建缓存 + NanoBench ⇒ **三轴全过才跑官方门禁**（门禁 9.5 min/臂 ⇒ 必须限量）；
5. 回写台账：`synth/soup_members/soup_weights/soup_family/official/six_ok`
   ⇒ `soup_family` = `bs<parent>:<标签>`（标签如 `eq2`/`w06`）⇒ **同 parent 同权重 ⇒ 同家族**；
6. **幂等**：已在台账里的 (parent, child, w) 组合直接跳过（不重复烧算力、不重复占臂号）；
7. **资源铁律**：每步前查 MemAvailable（< `--mem-gate-gb` 就等 120 s），超时/超限即收工。

用法（服务器）
    python -u manu/pipeline/npu/night_soup_batch.py --gpu 3 \
        --parents 98905:5,69901:5 --children 4 --weights 0.5,0.6 \
        --max-gates 14 --deadline "2026-11-18 07:00"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
W = "/mnt/data/siping/runs/npu20ms/nano_search"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
SIDE_CAR = "basin_soup_synth.json"
SID_BASE = 1105100
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

from manu.pipeline.npu import nano_search as NS  # noqa: E402
from manu.pipeline.npu import route2_acceptance as RA  # noqa: E402

LOGF = os.path.join(W, "soup_batch.log")


def log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGF, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def mem_avail_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    return 999.0


def wait_mem(threshold: float, deadline: float) -> bool:
    while mem_avail_gb() < threshold:
        if time.time() > deadline:
            return False
        log(f"[内存闸门] MemAvailable={mem_avail_gb():.1f}GB < {threshold:.0f}GB ⇒ 等待 120s")
        time.sleep(120)
    return True


def ck_path(trial: int, epoch) -> str:
    ep = int(epoch) if str(epoch).replace(".", "").isdigit() else epoch
    return os.path.join(ROOT, "trials", f"trial_{int(trial)}", "weights", f"epoch{ep}.pt")


def margin(picked: dict, base: dict) -> float:
    return min((picked.get("nano", 0) - base["nano"]) / 0.0042,
               (picked.get("nano4", 0) - base["nano4"]) / 0.0060,
               (picked.get("f1", 0) - base["f1"]) / 0.0008)


def wtag(w: float) -> str:
    """权重 ⇒ 家族标签（等权 → eq2；否则 w06/w04…）⇒ 同 parent 同标签 = 同家族。"""
    if abs(w - 0.5) < 1e-9:
        return "eq2"
    return ("w%02d" % int(round(w * 100))).replace(".", "")


# 🔴🔴 2026-11-17 16:30 修**跨通道撞号**（实测：wave0 与 wave2 撞了 20 个臂号）
#   老逻辑：启动时取「台账最大臂号+1」⇒ 两个通道**同时**启动就发同一批号
#   ⇒ 同号两种配方 ⇒ trial_<sid>/weights/epoch5.pt 互相覆盖
#   ⇒ 磁盘上的 gate_report.json 与台账配方对不上号 ⇒ **可能伪造升轨票**（实测已撞 1160007~1160026）
#   修法：① 每个通道一个**独占号段** [sid_base, sid_base+sid_span)
#        ② flock 原子占用；③ 占用即写台账占位（断点续跑可复用同配方同号）。
LOCK_PATH = "/tmp/sid_alloc.lock"


def reserve_sid(base: int, span: int, members, weights, family: str):
    """在独占号段里原子占一个臂号；返回 None ⇒ 号段已满。"""
    import fcntl
    key = tuple(sorted((int(m[0]), int(m[1])) for m in members))
    wkey = tuple(round(float(x), 6) for x in weights)
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    with open(LOCK_PATH, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            led = NS.load_ledger()
            arms = led.get("arms") or {}
            pick = None
            for cand in range(int(base), int(base) + int(span)):
                st = arms.get(str(cand))
                if st is None:
                    pick = cand
                    break
                mem = tuple(sorted((int(m[0]), int(m[1])) for m in (st.get("soup_members") or [])))
                ww = tuple(round(float(x), 6) for x in (st.get("soup_weights") or []))
                if mem == key and ww == wkey:
                    pick = cand          # 断点续跑：同配方同号 ⇒ 复用
                    break
            if pick is None:
                return None
            if str(pick) not in arms:
                st = NS.arm_state(led, pick)
                st.update({"trial": int(pick), "synth": True,
                           "soup_members": [[int(m[0]), int(m[1])] for m in members],
                           "soup_weights": [float(x) for x in weights],
                           "soup_family": family, "picked": None, "reserved": True,
                           "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                NS.save_ledger(led)
            return pick
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--parents", default="98905:5,69901:5,610000:5")
    ap.add_argument("--children", type=int, default=4)
    ap.add_argument("--children-arms", default="",
                    help="显式 child 列表 trial:epoch（逗号分隔）⇒ 与 --parents 构成全笛卡尔积；"
                         "用于 cross-basin（跨家族交叉汤）与精确对照，默认空=按血缘自动挑")
    ap.add_argument("--weights", default="0.5,0.6")
    ap.add_argument("--max-gates", type=int, default=14)
    ap.add_argument("--gate-family-quota", type=int, default=5,
                    help="每个家族最多送几个去门禁（升轨要 n≥4 ⇒ 必须按家族连打）")
    ap.add_argument("--only-gate", action="store_true",
                    help="只跑门禁：从台账捞「三轴全过且还没门禁」的汤臂去门禁（不再合成新汤）")
    ap.add_argument("--gate-family-prefix", default="",
                    help="🔴 2026-11-17 18:00：只门禁该家族前缀的臂（如 bs640001）"
                         "⇒ 把有限门禁算力集中在唯一 T2fp≤1 的家族上")
    ap.add_argument("--gate-sids", default="",
                    help="只门禁这些臂号（逗号分隔）；空 = 由 --only-gate 自动挑")
    ap.add_argument("--mem-gate-gb", type=float, default=45.0)
    ap.add_argument("--deadline", default="2026-11-18 07:00")
    ap.add_argument("--sid-base", type=int, default=SID_BASE)
    ap.add_argument("--sid-span", type=int, default=1000,
                    help="独占臂号段宽度 [sid_base, sid_base+sid_span)（防跨通道撞号）")
    args = ap.parse_args()

    dl = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))
    base = NS.BASE_V2
    ws = [float(x) for x in args.weights.split(",") if x.strip()]
    parents = []
    for tok in args.parents.split(","):
        if not tok.strip():
            continue
        t, e = tok.split(":")
        parents.append((int(t), int(e)))

    from manu.pipeline.npu.ckpt_synth import average_checkpoints  # noqa: PLC0415

    # ---- 1) 找 parent 的 child（血缘），按三轴最小边际排序 ----
    led = NS.load_ledger()
    arms = led.get("arms") or {}
    pat = re.compile(r"trials/trial_(\d+)/weights/epoch(\d+)\.pt$")
    children: dict[tuple, list] = {}
    for key, st in arms.items():
        p = st.get("params") or {}
        m = pat.search(str(p.get("init") or ""))
        if not m:
            continue
        pk = st.get("picked") or {}
        if not pk or "nano" not in pk:
            continue
        ptrial, pep = int(m.group(1)), int(m.group(2))
        if ptrial == int(key):
            continue
        children.setdefault((ptrial, pep), []).append(
            (int(key), pk.get("epoch"), pk, p.get("sw"), p.get("cp_prob")))

    # ---- 2) 已在台账里的汤组合 ⇒ 跳过（幂等）----
    done_pairs = set()
    for st in arms.values():
        if st.get("synth") and st.get("soup_members"):
            mem = st["soup_members"]
            w = (st.get("soup_weights") or [0.5, 0.5])[0]
            done_pairs.add((int(mem[0][0]), int(mem[0][1]), int(mem[1][0]), int(mem[1][1]), round(float(w), 6)))

    pending_gate: list = []
    if args.only_gate:
        _led = NS.load_ledger()
        for _k, _st in (_led.get("arms") or {}).items():
            try:
                _sid = int(_k)
            except Exception:  # noqa: BLE001
                continue
            if not _st.get("synth") or not _st.get("soup_members"):
                continue
            _of = _st.get("official") or {}
            if _of.get("t1") is not None and _of.get("t3") is not None:
                continue                      # 已有门禁读数
            # 🟢 磁盘上的门禁报告是**权威**（台账写入可能因并发写竞争丢失 ⇒ 不能只看台账）
            _gdir = os.path.join(ROOT, "gate", f"NANO_{_sid}_ep5_official")
            if os.path.isfile(os.path.join(_gdir, "gate_report.json")):
                log(f"[汤批次] sid={_sid} 磁盘已有门禁报告 ⇒ 跳过（台账可能丢写）")
                continue
            # 🟢 2026-11-17 12:10：**门禁目录已存在但报告未落** = 另一个门禁进程正在打这个臂
            #   ⇒ 必须跳过，否则两个门禁并发写同一个 out_dir 会把报告写坏。
            if os.path.isdir(_gdir):
                log(f"[汤批次] sid={_sid} 门禁目录已存在（报告未落 ⇒ 正在打）⇒ 跳过")
                continue
            _pk = _st.get("picked") or {}
            # 🟢 2026-11-17 16:45：判据改成**非严格 >=**（与「严格六项全过」口径一致：
            #   六项全过允许**恰好打平**基线 ⇒ 这里用 > 会把「打平也能过」的候选臂漏掉）
            if not (_pk.get("nano", 0) >= base["nano"] and _pk.get("nano4", 0) >= base["nano4"]
                    and _pk.get("f1", 0) >= base["f1"]):
                continue                      # 只打三轴全过的
            pending_gate.append((margin(_pk, base), _sid, _st.get("soup_family") or "?", _pk))
        if args.gate_family_prefix:
            # 🟢🟢 2026-11-18 10:25 支持**逗号分隔多前缀** ⇒ 可把同一家族的不同权重段拆给不同门禁通道
            #   （例：通道 A 打 w55/w56 家族、通道 C 打 w7*/w8*/w9* 高权重段 ⇒ 两条通道不抢同一批臂）
            _pfxs = [p.strip() for p in str(args.gate_family_prefix).split(",") if p.strip()]
            _n0 = len(pending_gate)
            pending_gate = [x for x in pending_gate
                            if any(str(x[2]).startswith(_pf) for _pf in _pfxs)]
            log(f"[汤批次] 家族前缀过滤 {'|'.join(_pfxs)}：{_n0} ⇒ {len(pending_gate)} 个候选")
        if args.gate_sids:
            _want = set(int(x) for x in args.gate_sids.split(",") if x.strip())
            pending_gate = [x for x in pending_gate if x[1] in _want]
        log(f"[汤批次] 只门禁模式：候选 {len(pending_gate)} 个（{args.gate_sids or '自动'}）")
        for x in pending_gate:
            log(f"   待门禁 sid={x[1]} fam={x[2]} m={x[0]:+.3f}σ")

    # ---- 显式 child（cross-basin / 精确对照）：直接取台账里的 picked + params.sw ----
    explicit: list = []
    for tok in args.children_arms.split(","):
        tok = tok.strip()
        if not tok:
            continue
        t, e = tok.split(":")
        explicit.append((int(t), int(e)))
    by_trial: dict = {}
    for key, st in arms.items():
        try:
            by_trial[int(key)] = st
        except Exception:  # noqa: BLE001
            continue

    combos = []
    for (pt, pe) in parents:
        if explicit:
            for (ct, ce) in explicit:
                cst = by_trial.get(ct) or {}
                pk = cst.get("picked") or {}
                sw = (cst.get("params") or {}).get("sw")
                mg = margin(pk, base) if pk.get("nano") is not None else -9.0
                for w in ws:
                    combos.append((pt, pe, ct, ce, w, sw, mg))
        else:
            kids = children.get((pt, pe)) or []
            kids.sort(key=lambda x: -margin(x[2], base))
            for (ct, ce, pk, sw, cp) in kids[: args.children]:
                for w in ws:
                    combos.append((pt, pe, ct, ce, w, sw, margin(pk, base)))
    # 已做过的排前面
    combos.sort(key=lambda c: (c[0], c[2], c[4]))
    todo = [c for c in combos if (c[0], c[1], c[2], c[3], round(c[4], 6)) not in done_pairs]
    log(f"[汤批次] parent {parents}｜children/parent "
        f"{args.children if not explicit else str(len(explicit)) + '(显式)'}｜权重 {ws} ⇒ 组合 {len(combos)}（跳过已做 {len(combos) - len(todo)}）")
    for c in combos:
        log(f"   计划 parent={c[0]}@{c[1]} ← child={c[2]}@{c[3]} w={c[4]} (child sw={c[5]} m={c[6]:+.3f}σ)"
            f"{'  [已做]' if (c[0], c[1], c[2], c[3], round(c[4], 6)) in done_pairs else ''}")

    # 🟢 2026-11-17 12:10：断点续跑时**不能**从 args.sid_base 重新发号
    #   （已用过的臂号在台账里，重新发号 = 同号换配方 = 假读数）⇒ 从「台账最大臂号+1」起。
    # 🔴 2026-11-17 16:30：不再用「台账最大臂号+1」（会跨通道撞号）⇒ 改成**独占号段**
    sid = args.sid_base
    log(f"[汤批次] 独占臂号段 = [{args.sid_base}, {args.sid_base + args.sid_span})")
    n_gate = 0
    n_soup = 0
    stats = {"triad": 0, "six": 0}
    pending = []          # (边际, sid, fam, gate 是否已跑)
    for (pt, pe, ct, ce, w, csw, cm) in ([] if args.only_gate else todo):
        if time.time() > dl:
            log("[汤批次] 已过截止 ⇒ 收工")
            break
        sid = reserve_sid(args.sid_base, args.sid_span,
                          [(int(pt), int(pe)), (int(ct), int(ce))], [w, 1.0 - w],
                          f"bs{int(pt)}:{wtag(w)}")
        if sid is None:
            log("[汤批次] 独占臂号段已满 ⇒ 收工")
            break
        p_ck, c_ck = ck_path(pt, pe), ck_path(ct, ce)
        if not (os.path.isfile(p_ck) and os.path.isfile(c_ck)):
            log(f"[汤批次] 权重缺失 {p_ck} / {c_ck} ⇒ 跳过")
            continue
        out_dir = os.path.join(ROOT, "trials", f"trial_{sid}", "weights")
        os.makedirs(out_dir, exist_ok=True)
        out_ck = os.path.join(out_dir, "epoch5.pt")
        fam = f"bs{int(pt)}:{wtag(w)}"
        try:
            average_checkpoints([p_ck, c_ck], out_ck, weights=[w, 1.0 - w], epoch=5,
                                note=f"lineage_soup_{pt}@{pe}+{ct}@{ce}_w{w}")
        except Exception as exc:  # noqa: BLE001
            log(f"[汤批次] 合成失败 {pt}+{ct}：{exc}")
            sid += 1
            continue
        if not wait_mem(args.mem_gate_gb, dl):
            log("[汤批次] 内存闸门/截止 ⇒ 收工")
            break
        pkl = f"/mnt/data/siping/runs/npu20ms/tiny_ref/soupbatch_{sid}.pkl"
        try:
            if not RA.build_cache(out_ck, pkl, args.gpu, extra="--no-peak-attrs"):
                log(f"[汤批次] 缓存失败 sid={sid}")
                sid += 1
                continue
            res = NS._nb_pool().submit(NS._nb_worker, (pkl, f"soupbatch_{sid}")).result()
        except Exception as exc:  # noqa: BLE001
            log(f"[汤批次] 评测异常 sid={sid}：{exc}")
            sid += 1
            continue
        if not res:
            log(f"[汤批次] 评测失败 sid={sid}")
            sid += 1
            continue
        m = {"nano": res["nano"], "nano4": res["nano4"], "f1": res["f1"],
             "micro": res.get("micro"), "small": res.get("small"), "regular": res.get("regular"),
             "f1_4": res.get("f1_4"), "cache": pkl}
        triad = bool(m["nano"] > base["nano"] and m["nano4"] > base["nano4"] and m["f1"] > base["f1"])
        stats["triad"] += int(triad)
        mg = margin(m, base)
        log(f"[汤批次] sid={sid} 成员=({pt}@{pe},{ct}@{ce}) w={w} fam={fam} ⇒ "
            f"nano={m['nano']:.6f} nano4={m['nano4']:.6f} F1={m['f1']:.6f} "
            f"m={mg:+.3f}σ 三轴全过={triad}")
        n_soup += 1
        # ---- 回写台账（写前重新 load，规避与训练轴/其它进程的写竞争）----
        led2 = NS.load_ledger()
        st = NS.arm_state(led2, sid)
        st.update({
            "trial": sid, "synth": True, "source_trial": int(pt),
            "soup_members": [[int(pt), int(pe)], [int(ct), int(ce)]],
            "soup_weights": [w, 1.0 - w], "soup_family": fam,
            "params": {"init": out_ck, "sw": csw, "lr": None, "lr_floor": None,
                       "mix": "soup", "fb": None, "ow": None, "sched": None,
                       "backbone_lr": None, "ema_decay": None, "swb": None,
                       "swb_thr": None, "cp_prob": None, "cp_max_px": None},
            "seed": None,
            "nanobench": {"5": m}, "picked": dict(m), "picked_epoch": 5,
            "official": None, "six_ok": False,
            "desc": f"[SB] soup({pt}@{pe},{ct}@{ce}) w={w}",
            "channel": "SB", "registered_by": "night_soup_batch",
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        NS.save_ledger(led2)
        try:
            with open(os.path.join(out_dir, SIDE_CAR), "w", encoding="utf-8") as fh:
                json.dump({"key": f"lineage:{pt}@{pe}:{ct}@{ce}:{w}", "scored": True,
                           "members": [[int(pt), int(pe)], [int(ct), int(ce)]],
                           "weights": [w, 1.0 - w], "family": fam, "tag": "soupbatch",
                           "basin": int(pt),
                           "nanobench": {k: m.get(k) for k in ("nano", "nano4", "f1")},
                           "triad": triad, "gate": None,
                           "entry": {"family": fam, "tag": "soupbatch", "members": [[int(pt), int(pe)], [int(ct), int(ce)]],
                                     "weights": [w, 1.0 - w], "synth_id": sid,
                                     "nanobench": m, "triad": triad, "gate": None}},
                          fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            log(f"[汤批次] sidecar 写失败：{exc}")
        pending.append((mg, sid, fam, m))
        sid += 1

    # ---- 阶段2：对**三轴全过**的汤跑官方门禁 ⇒ 六项全过 + 家族 n ----
    # 🔴 门禁是**最贵的一步**（~10~35 min/臂）⇒ 配额要**按家族分配**：
    #   升轨关 = 「同家族六项全过 n≥4」⇒ 只挑全局最好的几个臂去门禁没用（它们分属不同家族，n=1）
    #   ⇒ 必须**一个家族连门禁 4~6 个成员**才可能凑够 n≥4。
    pending = list(pending_gate) + pending
    pending.sort(key=lambda x: -x[0])
    by_fam: dict[str, list] = {}
    for item in pending:
        by_fam.setdefault(item[2], []).append(item)
    # 家族排序 = 该家族最好成员的边际（先押最有希望的家族）
    fam_order = sorted(by_fam.items(), key=lambda kv: -kv[1][0][0])
    quota = max(1, args.gate_family_quota)
    ordered = []
    for fam, items in fam_order:
        ordered.extend(items[:quota])
    log(f"[汤批次] 门禁排期：三轴全过 {len(pending)} 个 / 家族 {len(by_fam)} 个 ⇒ "
        f"按家族配额 {quota} 排出 {len(ordered)} 个待门禁")
    for mg, s_id, fam, m in ordered:
        if time.time() > dl:
            log("[汤批次] 已过截止 ⇒ 门禁阶段收工")
            break
        if n_gate >= args.max_gates:
            log(f"[汤批次] 门禁次数用满（{n_gate}/{args.max_gates}）⇒ 收工")
            break
        if not wait_mem(args.mem_gate_gb, dl):
            log("[汤批次] 内存闸门/截止 ⇒ 门禁阶段收工")
            break
        n_gate += 1
        g = NS.run_official_gate(s_id, 5, args.gpu)
        gate = {k: (g or {}).get(k) for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")}
        gate["six_ok"] = bool(gate.get("t1") is not None and gate.get("t3") is not None
                             and gate["t1"] >= base["t1"]
                             and gate.get("t2_fp") is not None and gate["t2_fp"] <= base["t2_fp"]
                             and gate["t3"] >= base["t3"])
        stats["six"] += int(bool(gate["six_ok"]))
        log(f"[汤批次] 门禁 sid={s_id} fam={fam} ⇒ T1={gate.get('t1')} T2fp={gate.get('t2_fp')} "
            f"T3={gate.get('t3')} six_ok={gate.get('six_ok')}")
        led2 = NS.load_ledger()
        st = NS.arm_state(led2, s_id)
        st["official"] = gate
        st["six_ok"] = bool(gate.get("six_ok"))
        NS.save_ledger(led2)
        sc = os.path.join(ROOT, "trials", f"trial_{s_id}", "weights", SIDE_CAR)
        try:
            with open(sc, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            d["gate"] = gate
            if isinstance(d.get("entry"), dict):
                d["entry"]["gate"] = gate
            with open(sc, "w", encoding="utf-8") as fh:
                json.dump(d, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            log(f"[汤批次] sidecar 回写失败：{exc}")
    # ---- 家族统计（同 parent 同权重 ⇒ 同家族 ⇒ 目标 n≥4）----
    led3 = NS.load_ledger()
    famcnt = {}
    for st in (led3.get("arms") or {}).values():
        if st.get("synth") and st.get("soup_family"):
            f = st["soup_family"]
            famcnt.setdefault(f, [0, 0])
            famcnt[f][0] += 1
            famcnt[f][1] += int(bool(st.get("six_ok")))
    for f, (n, k) in sorted(famcnt.items()):
        log(f"[家族] {f}：成员 {n} 个，其中六项全过 {k} 个"
            + ("  ✔ 可升轨" if k >= 4 else ""))
    log(f"[汤批次] 收工：合成 {n_soup} 个，送门禁 {n_gate} 个，三轴全过 {stats['triad']}，六项全过 {stats['six']}")


if __name__ == "__main__":
    main()
