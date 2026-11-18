#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同血缘**亲子汤**探针（§196 追加5）—— 零训练成本，专破 nano/nano4/F1 的 Pareto 张力。

动机（今夜实测，不是猜测）
--------------------------
· 昨夜唯一三轴全过臂 `1100275@1`（**sw4.2**）：nano .720936 ✓ / nano4 .657136 ✓ / F1 .890828 ✓
  ⇒ **只差官方门禁 t2_fp=3（红线 ≤1）**、t3 差 4e-6；
· 今夜 sw4.0 子代 `1102001@2`（**从 1100275@1 抛光**）：nano .712902 ✗ / **nano4 .667533 ✓✓（+2σ）** / F1 .890908 ✓
⇒ 三个轴互相拉扯：**sw 是唯一在动 nano 的旋钮，但 sw↑ ⇒ 长泉金池 T2 虚警涨**（sw↓ ⇒ nano↓、nano4↑）。

假设
----
同血缘（parent → child 只隔 1~2 epoch、同一盆地）的两个 ckpt **权重平均**：
  · 平均后落在**同一条损失盆地**里 ⇒ 不会像跨家族汤那样把特化平均掉；
  · 保留 parent 的 nano，同时吸收 child 的 nano4 / F1；
  · **sw4.2 的偏置被稀释** ⇒ 有机会把 t2_fp 从 3 拉回 ≤1 ⇒ 一次拿到六项全过。
⇒ 这是与**已被证伪的「跨家族汤」机制不同**的新轴（证伪的是跨配方平均把 nano 特化平均掉）。

执行
----
1. 读台账 ⇒ 找血缘对（child 的 `params["init"]` 就是 parent 的 epoch 权重 ⇒ 同一血缘）；
2. 按 child 的边际排序取前 N 对；
3. 每对生成 **等权汤(0.5/0.5)** 与 **加权汤(0.7/0.3 / 0.3/0.7)** ⇒ 落 `trial_<sid>/weights/epoch5.pt` + sidecar；
4. 建缓存 + NanoBench ⇒ **三轴全过才跑官方门禁** ⇒ 回写台账（synth/soup_members/soup_family/official/six_ok）；
5. **内存闸门**：MemAvailable < 45 GB 时等待（不与训练轴抢内存，§194 铁律）。

用法（服务器）
    python -u manu/pipeline/npu/night_lineage_soup.py --gpu 3 --max-soups 8 --deadline "2026-11-18 07:30"
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
SYNTH_BASE = 1105000
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

from manu.pipeline.npu import nano_search as NS  # noqa: E402
from manu.pipeline.npu import route2_acceptance as RA  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


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
    """内存闸门：MemAvailable < threshold 就等待（不与训练轴抢内存）。"""
    while mem_avail_gb() < threshold:
        if time.time() > deadline:
            return False
        log(f"[内存闸门] MemAvailable={mem_avail_gb():.1f}GB < {threshold}GB ⇒ 等待 120s")
        time.sleep(120)
    return True


def ck_path(trial: int, epoch) -> str:
    ep = int(epoch) if str(epoch).replace(".", "").isdigit() else epoch
    return os.path.join(ROOT, "trials", f"trial_{int(trial)}", "weights", f"epoch{ep}.pt")


def lineage_pairs(led: dict) -> list[tuple]:
    """血缘对 = (parent_trial, parent_ep, parent_sw, child_trial, child_ep, child_sw, child_state)。"""
    arms = led.get("arms") or {}
    out = []
    pat = re.compile(r"trials/trial_(\d+)/weights/epoch(\d+)\.pt$")
    for key, st in arms.items():
        p = st.get("params") or {}
        init = str(p.get("init") or "")
        m = pat.search(init)
        if not m:
            continue
        ptrial, pep = int(m.group(1)), int(m.group(2))
        if ptrial == int(key):
            continue
        picked = st.get("picked") or {}
        if not picked or "nano" not in picked:
            continue
        pst = arms.get(str(ptrial)) or {}
        psw = (pst.get("params") or {}).get("sw")
        out.append((ptrial, pep, psw, int(key), picked.get("epoch"),
                    p.get("sw"), picked))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--max-soups", type=int, default=12)
    ap.add_argument("--mem-gate-gb", type=float, default=45.0)
    ap.add_argument("--deadline", default="2026-11-18 07:30")
    ap.add_argument("--gate-limit", type=int, default=3, help="最多送几个汤跑官方门禁")
    args = ap.parse_args()
    dl = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))
    base = NS.BASE_V2

    led = NS.load_ledger()
    arms = led.get("arms") or {}

    def margin(picked: dict) -> float:
        return min((picked.get("nano", 0) - base["nano"]) / 0.0042,
                   (picked.get("nano4", 0) - base["nano4"]) / 0.0060,
                   (picked.get("f1", 0) - base["f1"]) / 0.0008)

    pairs = lineage_pairs(led)
    pairs.sort(key=lambda x: -margin(x[6]))
    log(f"[血缘汤] 血缘对 {len(pairs)} 个；边际前 8：")
    for pt, pe, psw, ct, ce, csw, pk in pairs[:8]:
        log(f"   parent {pt}@ep{pe}(sw{psw}) ← child {ct} ep{ce}(sw{csw}) nano={pk.get('nano'):.6f} "
            f"nano4={pk.get('nano4'):.6f} F1={pk.get('f1'):.6f} m={margin(pk):+.3f}σ")

    def sw_gap(x) -> float:
        try:
            return abs(float(x[2]) - float(x[5]))
        except Exception:  # noqa: BLE001
            return 0.0

    # A 组：**sw 对比对**（parent 与 child 的 sw 差 ≥ 0.1）⇒ 直击「稀释 sw4.2 偏置」的核心假设
    # B 组：边际最好的血缘对（同配方血缘，作为对照）
    grp_a = [x for x in pairs if sw_gap(x) >= 0.1]
    grp_b = [x for x in pairs if sw_gap(x) < 0.1]
    log(f"[血缘汤] sw 对比对 {len(grp_a)} 个（A 组优先），同 sw 对 {len(grp_b)} 个")
    # 特例：**parent 是 sw4.2 家族**的对（1100275 那类）⇒ 「稀释 sw4.2 偏置」的直接检验
    special = [x for x in grp_a if (x[2] is not None and float(x[2]) >= 4.15)][:1]
    chosen = (grp_a[:3] + special + grp_b[:1])[: max(1, args.max_soups // 3)]
    log(f"[血缘汤] 选中血缘对 {len(chosen)} 个："
        + "；".join(f"{x[0]}@ep{x[1]}(sw{x[2]})→{x[3]}@ep{x[4]}(sw{x[5]})" for x in chosen))
    combos = []
    for pt, pe, psw, ct, ce, csw, pk in chosen:
        for w in (0.5, 0.7, 0.3):
            combos.append(((pt, pe), (ct, ce), w, psw, csw))
    combos = combos[: args.max_soups]
    log(f"[血缘汤] 本轮汤计划 {len(combos)} 个")

    from manu.pipeline.npu.ckpt_synth import average_checkpoints  # noqa: PLC0415

    sid = SYNTH_BASE
    n_gate = 0
    for (pt, pe), (ct, ce), w, psw, csw in combos:
        if time.time() > dl:
            log("[血缘汤] 已过截止 ⇒ 收工")
            break
        p_ck = ck_path(pt, pe)
        c_ck = ck_path(ct, ce)
        if not (os.path.isfile(p_ck) and os.path.isfile(c_ck)):
            log(f"[血缘汤] 权重缺失 {p_ck} / {c_ck} ⇒ 跳过")
            continue
        out_dir = os.path.join(ROOT, "trials", f"trial_{sid}", "weights")
        os.makedirs(out_dir, exist_ok=True)
        out_ck = os.path.join(out_dir, "epoch5.pt")
        tag = "lin"
        try:
            average_checkpoints([p_ck, c_ck], out_ck,
                                weights=[w, 1.0 - w], epoch=5,
                                note=f"lineage_soup_{pt}@{pe}+{ct}@{ce}_w{w}")
        except Exception as exc:  # noqa: BLE001
            log(f"[血缘汤] 合成失败 {pt}+{ct}：{exc}")
            sid += 1
            continue
        if not wait_mem(args.mem_gate_gb, dl):
            log("[血缘汤] 内存闸门/截止 ⇒ 收工")
            break
        pkl = f"/mnt/data/siping/runs/npu20ms/tiny_ref/lineagesoup_{sid}.pkl"
        try:
            if not RA.build_cache(out_ck, pkl, args.gpu, extra="--no-peak-attrs"):
                log(f"[血缘汤] 缓存失败 sid={sid}")
                sid += 1
                continue
            res = NS._nb_pool().submit(NS._nb_worker, (pkl, f"lineagesoup_{sid}")).result()
        except Exception as exc:  # noqa: BLE001
            log(f"[血缘汤] 评测异常 sid={sid}：{exc}")
            sid += 1
            continue
        if not res:
            log(f"[血缘汤] 评测失败 sid={sid}")
            sid += 1
            continue
        m = {"nano": res["nano"], "nano4": res["nano4"], "f1": res["f1"],
             "micro": res.get("micro"), "small": res.get("small"), "regular": res.get("regular"),
             "f1_4": res.get("f1_4"), "cache": pkl}
        triad = bool(m["nano"] > base["nano"] and m["nano4"] > base["nano4"] and m["f1"] > base["f1"])
        mg = margin(m)
        log(f"[血缘汤] sid={sid} 成员=({pt}@{pe} sw{psw},{ct}@{ce} sw{csw}) w={w} ⇒ nano={m['nano']:.6f} "
            f"nano4={m['nano4']:.6f} F1={m['f1']:.6f} m={mg:+.3f}σ 三轴全过={triad}")
        gate = None
        if triad and n_gate < args.gate_limit and wait_mem(args.mem_gate_gb, dl):
            n_gate += 1
            g = NS.run_official_gate(sid, 5, args.gpu)
            gate = {k: (g or {}).get(k) for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")}
            gate["six_ok"] = bool(gate.get("t1") is not None and gate.get("t3") is not None
                                 and gate["t1"] >= base["t1"]
                                 and (gate.get("t2_fp") is not None and gate["t2_fp"] <= base["t2_fp"])
                                 and gate["t3"] >= base["t3"])
            log(f"[血缘汤] 门禁 sid={sid} ⇒ T1={gate.get('t1')} T2fp={gate.get('t2_fp')} "
                f"T3={gate.get('t3')} six_ok={gate.get('six_ok')}")
        # ---- 登记进台账（家族键 = bs<parent>:eq2 ⇒ 同 parent 的等权汤归一家族，n≥4 可达）----
        equal = abs(w - 0.5) < 1e-9
        fam = f"bs{int(pt)}:eq2" if equal else f"soup:{int(pt)}:w{w}"
        members = [[int(pt), int(pe)], [int(ct), int(ce)]]
        led2 = NS.load_ledger()
        st = NS.arm_state(led2, sid)
        st.update({
            "trial": sid, "synth": True, "source_trial": int(pt),
            "soup_members": members, "soup_weights": [w, 1.0 - w],
            "soup_family": fam,
            "params": {"init": out_ck, "sw": psw, "lr": None, "lr_floor": None,
                       "mix": "soup", "fb": None, "ow": None, "sched": None,
                       "backbone_lr": None, "ema_decay": None, "swb": None,
                       "swb_thr": None, "cp_prob": None, "cp_max_px": None},
            "seed": None,
            "nanobench": {"5": m}, "picked": dict(m), "picked_epoch": 5,
            "official": gate, "six_ok": bool((gate or {}).get("six_ok")),
            "desc": f"[LIN] soup({pt}@{pe},{ct}@{ce}) w={w}",
            "channel": "LIN",
            "registered_by": "night_lineage_soup",
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        NS.save_ledger(led2)
        try:
            with open(os.path.join(out_dir, SIDE_CAR), "w", encoding="utf-8") as fh:
                json.dump({"key": f"lineage:{pt}@{pe}:{ct}@{ce}:{w}", "scored": True,
                           "members": members, "weights": [w, 1.0 - w], "family": fam,
                           "tag": tag, "basin": int(pt),
                           "nanobench": {k: m.get(k) for k in ("nano", "nano4", "f1")},
                           "triad": triad, "gate": gate,
                           "entry": {"family": fam, "tag": tag, "members": members,
                                     "weights": [w, 1.0 - w], "synth_id": sid,
                                     "nanobench": m, "triad": triad, "gate": gate}},
                          fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            log(f"[血缘汤] sidecar 写失败：{exc}")
        sid += 1
    log(f"[血缘汤] 收工：合成 {sid - SYNTH_BASE} 个，送门禁 {n_gate} 个")


if __name__ == "__main__":
    main()
