#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**多成员**权重汤通道（night_soupN）—— 零训练成本，专打「同家族 n≥4」升轨硬门槛。

为什么要有这条通道（2026-11-17 16:00 实测诊断，不是猜测）
----------------------------------------------------------
台账全量复算 + 逐臂诊断得出两个关键事实：

① **parent=640001 的 2 成员汤：三轴全过率只有 ~4%**
   （120 个臂里只有 5 个满足 nano/nano4/F1 同时 ≥ 基线）⇒ 想在
   `soup:bs640001:w2` 家族凑 4 个六项全过，需要上百个臂 ⇒ 门禁预算根本不够。
   绑点不是 nano（能到 .727），而是 **F1 ≥ .890398**（CX 家族 F1 掉到 .885~.889）
   与 **T2_fp ≤ 1**（2 成员汤普遍 fp=2~4）。

② **基线本身就是「5 成员等权汤」，而且它的家族里已经躺着 1 票**
   基线 610000 = soup(98900@5, 12679@3, 99105@5, 98905@5, 98901@5)，等权，
   `soup_family = "bs12679:eq"` ⇒
     · 升轨关 A（家族 `soup:bs12679:eq5` 六项全过 n≥4）**已有基线这 1 票** ⇒ 只差 3 票；
     · 升轨关 B（同 `soup_family` 字符串组六项全过 n≥2）**已有基线** ⇒ 只差 1 票。
   而台账里所有同族 5 成员汤（600004 / 620002 / 620004 / 98903 …）**T2_fp 恒为 1**
   ⇒ **成员越多 ⇒ 虚警越少**（2 成员汤 fp=2~4，5 成员汤 fp=1）⇒ 多成员汤是**降虚警**的主杠杆。

本通道做的事
------------
把基线的 5 个成员当「骨架」，逐个换血：
  · **K=5 换血**：CORE5 − {98900,98901,98905,99105} + {强子臂}
  · **K=6 加血**：CORE5 + {强子臂}
等权合成 ⇒ `soup_family = "bs12679:eq"` ⇒ 落进**与基线同一个升轨家族/组**。

用法（服务器）
    python -u manu/pipeline/npu/night_soupN.py --gpu 2 \
        --sets-file /tmp/eq_sets.txt --family bs12679:eq \
        --max-gates 24 --deadline "2026-11-18 07:30"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
W = "/mnt/data/siping/runs/npu20ms/nano_search"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
SIDE_CAR = "soupN_synth.json"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

from manu.pipeline.npu import nano_search as NS  # noqa: E402
from manu.pipeline.npu import route2_acceptance as RA  # noqa: E402

LOGF = os.path.join(W, "soupN.log")


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


def parse_sets(path: str) -> list:
    """一行一个组合：`label|member@ep,member@ep[,w1,w2,...]`。"""
    out = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            label = parts[0].strip()
            mem_s = parts[1].strip()
            weights = None
            if len(parts) > 2 and parts[2].strip():
                weights = [float(x) for x in parts[2].split(",") if x.strip()]
            mem = []
            for tok in mem_s.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                t, e = tok.split("@")
                mem.append((int(t), int(e)))
            if len(mem) < 2:
                continue
            out.append({"label": label, "members": mem, "weights": weights})
    return out


def _run_gate(s_id: int, m: dict, args, base: dict, dl: float, stats: dict) -> None:
    """官方门禁 + 回写台账（幂等：目录已存在则跳过，避免并发写坏报告）。"""
    gdir = os.path.join(ROOT, "gate", f"NANO_{s_id}_ep5_official")
    if os.path.isdir(gdir):
        log(f"[门禁] sid={s_id} 门禁目录已存在 ⇒ 跳过")
        return
    if not wait_mem(args.mem_gate_gb, dl):
        log(f"[门禁] 内存闸门/截止 ⇒ 放弃 sid={s_id}")
        return
    g = NS.run_official_gate(s_id, 5, args.gpu)
    gate = {k: (g or {}).get(k) for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")}
    gate["six_ok"] = bool(gate.get("t1") is not None and gate.get("t3") is not None
                         and gate["t1"] >= base["t1"]
                         and gate.get("t2_fp") is not None and gate["t2_fp"] <= base["t2_fp"]
                         and gate["t3"] >= base["t3"])
    stats["six"] += int(bool(gate["six_ok"]))
    log(f"[门禁] sid={s_id} fam={args.family} ⇒ T1={gate.get('t1')} T2fp={gate.get('t2_fp')} "
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
        log(f"[sidecar 回写失败] {exc}")


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
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--sets-file", required=True)
    ap.add_argument("--family", default="bs12679:eq")
    ap.add_argument("--max-gates", type=int, default=24)
    ap.add_argument("--mem-gate-gb", type=float, default=45.0)
    ap.add_argument("--deadline", default="2026-11-18 07:30")
    ap.add_argument("--sid-base", type=int, default=1180000)
    ap.add_argument("--sid-span", type=int, default=1000)
    ap.add_argument("--only-gate", action="store_true")
    args = ap.parse_args()

    dl = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))
    base = NS.BASE_V2
    from manu.pipeline.npu.ckpt_synth import average_checkpoints  # noqa: PLC0415

    led = NS.load_ledger()
    arms = led.get("arms") or {}
    done = set()
    for st in arms.values():
        if st.get("synth") and st.get("soup_members"):
            mem = tuple(sorted((int(m[0]), int(m[1])) for m in st["soup_members"]))
            w = tuple(round(float(x), 6) for x in (st.get("soup_weights") or []))
            done.add((mem, w))

    specs = parse_sets(args.sets_file)
    log(f"[通道] family={args.family}｜组合 {len(specs)}｜台账已有汤 {len(done)}")
    log(f"[基线] nano={base['nano']:.7f} nano4={base['nano4']:.7f} f1={base['f1']:.7f} "
        f"t1={base['t1']:.7f} t2fp={base['t2_fp']} t3={base['t3']:.7f}")

    # 只门禁模式：捞台账里「三轴全过且还没门禁」的汤臂
    pending_gate = []
    if args.only_gate:
        for sid, st in (NS.load_ledger().get("arms") or {}).items():
            if not st.get("synth") or not st.get("soup_members"):
                continue
            if st.get("soup_family") != args.family:
                continue
            gdir = os.path.join(ROOT, "gate", f"NANO_{sid}_ep5_official")
            if os.path.isdir(gdir):
                continue
            pk = st.get("picked") or {}
            if not (pk.get("nano", 0) >= base["nano"] and pk.get("nano4", 0) >= base["nano4"]
                    and pk.get("f1", 0) >= base["f1"]):
                continue
            pending_gate.append((margin(pk, base), int(sid), args.family, pk))
        pending_gate.sort(key=lambda x: -x[0])
        log(f"[只门禁] 候选 {len(pending_gate)} 个")

    # ---- 阶段1：合成 + 三轴评测 ----
    sid = args.sid_base
    log(f"[臂号] 独占号段 = [{args.sid_base}, {args.sid_base + args.sid_span})")
    n_soup = 0
    pending = list(pending_gate)
    stats = {"triad": 0, "six": 0}
    n_gate = 0
    for spec in ([] if args.only_gate else specs):
        if time.time() > dl:
            log("[截止] 合成阶段收工")
            break
        mem = spec["members"]
        weights = spec["weights"] or [1.0 / len(mem)] * len(mem)
        key = (tuple(sorted(mem)), tuple(round(float(x), 6) for x in weights))
        if key in done:
            log(f"[跳过] 已存在 {spec['label']} ⇒ {mem}")
            continue
        sid = reserve_sid(args.sid_base, args.sid_span, mem, weights, args.family)
        if sid is None:
            log("[臂号] 独占号段已满 ⇒ 收工")
            break
        cks = [ck_path(t, e) for t, e in mem]
        missing = [c for c in cks if not os.path.isfile(c)]
        if missing:
            log(f"[跳过] {spec['label']} 缺权重 {missing}")
            continue
        out_dir = os.path.join(ROOT, "trials", f"trial_{sid}", "weights")
        os.makedirs(out_dir, exist_ok=True)
        out_ck = os.path.join(out_dir, "epoch5.pt")
        try:
            average_checkpoints(cks, out_ck, weights=weights, epoch=5,
                                note=f"soupN_{spec['label']}")
        except Exception as exc:  # noqa: BLE001
            log(f"[合成失败] {spec['label']}：{exc}")
            sid += 1
            continue
        pkl = f"/mnt/data/siping/runs/npu20ms/tiny_ref/soupN_{sid}.pkl"
        try:
            # 🔴 必须先建**推理缓存**（跑一遍模型落 pred/gt），nanobench 才有得评
            if not RA.build_cache(out_ck, pkl, args.gpu, extra="--no-peak-attrs"):
                log(f"[缓存失败] {spec['label']} sid={sid}")
                sid += 1
                continue
            res = NS._nb_pool().submit(NS._nb_worker, (pkl, f"soupN_{sid}")).result()
        except Exception as exc:  # noqa: BLE001
            log(f"[评测异常] {spec['label']}：{exc}")
            sid += 1
            continue
        if not res:
            log(f"[评测失败] {spec['label']}")
            sid += 1
            continue
        m = {"nano": res["nano"], "nano4": res["nano4"], "f1": res["f1"],
             "micro": res.get("micro"), "small": res.get("small"), "regular": res.get("regular"),
             "f1_4": res.get("f1_4"), "cache": pkl}
        triad = bool(m["nano"] >= base["nano"] and m["nano4"] >= base["nano4"] and m["f1"] >= base["f1"])
        mg = margin(m, base)
        stats["triad"] += int(triad)
        n_soup += 1
        log(f"[汤N] sid={sid} {spec['label']} 成员={len(mem)} ⇒ nano={m['nano']:.7f} "
            f"nano4={m['nano4']:.7f} f1={m['f1']:.7f} m={mg:+.3f}σ 三轴全过={triad}")
        led2 = NS.load_ledger()
        st = NS.arm_state(led2, sid)
        st.update({
            "trial": sid, "synth": True, "source_trial": int(mem[0][0]),
            "soup_members": [[int(t), int(e)] for t, e in mem],
            "soup_weights": [float(x) for x in weights], "soup_family": args.family,
            "params": {"init": out_ck, "sw": None, "lr": None, "lr_floor": None,
                       "mix": "soupN", "fb": None, "ow": None, "sched": None,
                       "backbone_lr": None, "ema_decay": None, "swb": None,
                       "swb_thr": None, "cp_prob": None, "cp_max_px": None},
            "seed": None,
            "nanobench": {"5": m}, "picked": dict(m), "picked_epoch": 5,
            "official": None, "six_ok": False,
            "desc": f"[SN] soupN {spec['label']} k={len(mem)}",
            "channel": "SN", "registered_by": "night_soupN",
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        NS.save_ledger(led2)
        try:
            with open(os.path.join(out_dir, SIDE_CAR), "w", encoding="utf-8") as fh:
                json.dump({"key": spec["label"], "scored": True,
                           "members": [[int(t), int(e)] for t, e in mem],
                           "weights": list(weights), "family": args.family,
                           "tag": "soupN",
                           "nanobench": {k: m.get(k) for k in ("nano", "nano4", "f1")},
                           "triad": triad, "gate": None,
                           "entry": {"family": args.family, "tag": "soupN",
                                     "members": [[int(t), int(e)] for t, e in mem],
                                     "weights": list(weights), "synth_id": sid,
                                     "nanobench": m, "triad": triad, "gate": None}},
                          fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            log(f"[sidecar 写失败] {exc}")
        pending.append((mg, sid, args.family, m))
        sid += 1
        # **边合成边门禁**（升轨要的是「家族投票数」⇒ 早门禁早拿票）
        if triad and not args.only_gate and n_gate < args.max_gates and time.time() < dl:
            _run_gate(sid - 1, m, args, base, dl, stats)
            n_gate += 1

    # ---- 阶段2：把还没门禁的三轴全过臂补齐（升轨 n≥4 ⇒ 家族连打）----
    pending.sort(key=lambda x: -x[0])
    for mg, s_id, fam, m in pending:
        if time.time() > dl:
            log("[截止] 门禁阶段收工")
            break
        if n_gate >= args.max_gates:
            log(f"[门禁] 次数用满（{n_gate}/{args.max_gates}）⇒ 收工")
            break
        if not wait_mem(args.mem_gate_gb, dl):
            log("[门禁] 内存闸门/截止 ⇒ 收工")
            break
        _run_gate(s_id, m, args, base, dl, stats)
        n_gate += 1
    log(f"[收工] 合成 {n_soup} 个，送门禁 {n_gate} 个，三轴全过 {stats['triad']}，六项全过 {stats['six']}")


if __name__ == "__main__":
    main()
