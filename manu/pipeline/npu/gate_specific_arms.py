#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对**指定**的 (trial, epoch) 补跑官方三轨门禁（幂等：gate_report.json 存在则复用）。

═══ 为什么需要（2026-11-12 09:20）═══
今晨实测发现：**官方 T2（金池）随 swl（大目标降权）单调塌陷**
（swl=1.0 → 0.994~0.999；swl=0.1 → 0.795~0.851），而 NanoBench(no-GMC) 看不见。
⇒ 需要对「基线配方、epochs=5」的臂单独补跑官方三轨，用来回答两个问题：
   ① epochs=5 是否真的与 epochs=20 逐位等价（§142.3 的代码级结论是否成立）；
   ② epochs=5 的基线配方臂官方 T2 是否也塌（若塌 ⇒ 搜索器的 epochs=5 捷径是致命的）。

用法（服务器）：
    setsid nohup python manu/pipeline/npu/gate_specific_arms.py \
        --arms 12040:5 12041:5 12042:5 --gpus 0,1,2,3 \
        > /mnt/data/siping/runs/npu20ms/nano_search/specific_gate.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nano_search as ns  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="形如 12040:5（trial:epoch）")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.replace(" ", "").split(",") if x.strip()]

    led = ns.load_ledger()
    base = led.get("baseline_v2") or {}
    for spec in args.arms:
        t, e = spec.split(":")
        trial, ep = int(t), int(e)
        st = (led.get("arms") or {}).get(str(trial)) or {}
        print(f"[spec-gate] trial {trial} ep{ep} | {st.get('desc','')[:70]}")
    if args.dry_run:
        return

    # ⚠️ 不走 ns.phase_gate：它会按规则**重挑 epoch**（all3_pass_max_nano ⇒ 12053 会被挑到 ep4）
    #   ⇒ 本脚本要的是「**指定 epoch** 的官方读数」（例如 12053 的 ep5 才是与 9863 ep5 同口径的对照）。
    import concurrent.futures as cf  # noqa: PLC0415

    def _gate(spec_trial: int, spec_ep: int, gpu: int) -> tuple:
        try:
            g = ns.run_official_gate(spec_trial, spec_ep, gpu)
        except Exception as e:  # noqa: BLE001
            g = {"error": str(e)}
        led2 = ns.load_ledger()
        st = ns.arm_state(led2, spec_trial)
        # 🔴 2026-11-12 修：官方读数**按 epoch 分槽**，绝不互相覆盖。
        #   原逻辑 st["official"] = g + st["picked_epoch"] = spec_ep：
        #     ① 同臂连跑两个 epoch ⇒ 后写的覆盖先写的（实测 12053 的 ep4 读数被 ep5/ep3 冲掉）；
        #     ② 覆盖 picked_epoch ⇒ 把 runner 选好的 epoch 改掉（09:30 的 re-pick 才纠正回来，纯侥幸）。
        by_ep = st.setdefault("official_by_epoch", {})
        by_ep[str(spec_ep)] = g
        # 只有与**台账当前 picked_epoch 一致**的读数才写进 official 槽（升轨判定只认这一个槽）。
        if int(st.get("picked_epoch") or -1) == int(spec_ep):
            st["official"] = g
        ns.save_ledger(led2)
        return spec_trial, spec_ep, g

    specs = []
    for spec in args.arms:
        t, e = spec.split(":")
        specs.append((int(t), int(e)))

    results = []
    with cf.ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futs = [ex.submit(_gate, t, e, gpus[i % len(gpus)]) for i, (t, e) in enumerate(specs)]
        for fu in futs:                       # futs 与 specs 同序 ⇒ results 保持请求顺序
            t, e, g = fu.result()
            results.append((t, e, g))
            print(f"[spec-gate] trial {t} ep{e} ⇒ T1={g.get('t1')} T2={g.get('t2')}(fp {g.get('t2_fp')}) "
                  f"T3={g.get('t3')}(fp {g.get('t3_fp')}) verdict={g.get('verdict')} err={g.get('error')}")

    # 🔴 2026-11-12 修：报告**直接用本次算出的结果**，不再从台账回读。
    #   原逻辑从台账读 official/picked_epoch ⇒ 同臂跑两个 epoch 时**两行读数完全一样**
    #   （实测 12053 ep5 出现两次、ep3 消失）⇒ 报告会说谎。
    out = {"ts": time.strftime("%F %T"),
           "base": {k: base.get(k) for k in ("trial", "nano", "nano4", "f1", "t1", "t2", "t2_fp", "t3", "t3_fp")},
           "results": []}
    for t, e, g in results:
        row = {"trial": t, "epoch": e,
               "t1": g.get("t1"), "t2": g.get("t2"), "t2_fp": g.get("t2_fp"),
               "t3": g.get("t3"), "t3_fp": g.get("t3_fp"), "verdict": g.get("verdict")}
        row["t2_ok"] = bool(g.get("t2") is not None and g["t2"] >= (base.get("t2") or 0) - 0.002)
        out["results"].append(row)
        print(f"[spec-gate] trial {t} ep{row['epoch']} | T1={row['t1']} T2={row['t2']}(fp {row['t2_fp']}) "
              f"T3={row['t3']}(fp {row['t3_fp']}) ⇒ T2 不塌={row['t2_ok']}")
    path = f"{ns.WORK}/specific_gate_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[spec-gate] 报告已落盘：{path}")


if __name__ == "__main__":
    main()
