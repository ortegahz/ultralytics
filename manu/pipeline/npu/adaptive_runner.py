#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自适应搜索调度器（2026-11-11）：**以击败 Baseline_v2.0(trial_9863) 为目标**。

与 `nano_search.py` 的固定波次不同，这里是**闭环自适应**：
  每轮读台账 → 选当前 champion → 生成「零训练合成臂」+「训练臂」→ 训练 → NanoBench
  → 官方三轨 → 联合门禁 → 重新选 champion → 下一轮，直到 deadline。

两类臂：
  ① **合成臂（不训练，只花评测时间）**：Model Soup（不同超参/种子的同构权重平均）
     与 SWA（同一臂相邻 epoch 的权重平均）—— 这是**此前从未探过的轴**，且成本近乎为零。
  ② **训练臂**：以 champion 权重为 init 的单变量微调（剂量邻域 / 宽小目标定义 /
     种子复现 / 退火周期 / 长训取早期 epoch）。

所有结果落盘：
  - 台账 `nano_search_ledger.json`（顶层 `rounds`）
  - `final_report.md` / `beat_to_pass_report.md`
  - `adaptive.log`

用法（服务器，常驻）：
  setsid nohup /home/huangzhe/anaconda3/envs/uav/bin/python \
      manu/pipeline/npu/adaptive_runner.py --deadline "2026-11-12 09:00" \
      >> /mnt/data/siping/runs/npu20ms/nano_search/adaptive.log 2>&1 &
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nano_search as ns  # noqa: E402
from ckpt_synth import average_checkpoints  # noqa: E402

TRIAL_BASE = 12000          # 自适应波次 trial 号段（与 W1~W11 不冲突）
SYNTH_EPOCH = 5             # 合成臂统一用的 epoch 文件名（评测按 epoch 号索引）
EPOCHS_EVAL = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] [ADAPT] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 基线与 champion
# ---------------------------------------------------------------------------
def base_ref(led: dict) -> dict:
    """取 Baseline_v2.0 的**精确读数**（常量是 6 位截断，会让基线臂自己判负）。"""
    base = dict(ns.BASE_V2)
    a98 = (led.get("arms") or {}).get(str(ns.BASE_V2["trial"]), {})
    pk = a98.get("picked") or {}
    of = a98.get("official") or {}
    for k in ("nano", "micro", "small", "regular", "f1", "nano4"):
        if pk.get(k) is not None:
            base[k] = pk[k]
    for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp"):
        if of.get(k) is not None:
            base[k] = of[k]
    return base


def ckpt_path(trial: int, epoch: int) -> str:
    return f"{ns.ROOT}/trials/trial_{trial}/weights/epoch{epoch}.pt"


def core_checks(st: dict, base: dict) -> dict:
    """Beat-to-Pass 六项（ms 项另算，见 ns.phase_report）。"""
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    out = {}
    if pk.get("nano") is not None:
        out["nano"] = pk["nano"] >= base["nano"]
    if pk.get("nano4") is not None:
        out["nano4"] = pk["nano4"] >= base["nano4"]
    if pk.get("f1") is not None:
        out["f1"] = pk["f1"] >= base["f1"]
    if of.get("t1") is not None:
        out["t1"] = of["t1"] >= base["t1"]
    if of.get("t2_fp") is not None:
        out["t2fp"] = of["t2_fp"] <= base["t2_fp"]
    if of.get("t3") is not None:
        out["t3"] = of["t3"] >= base["t3"]
    return out


def champion_score(st: dict, base: dict) -> tuple:
    """分数 = nano 主导 + 达标项奖励 − 违约项惩罚。返回 (score, detail)。"""
    pk = st.get("picked") or {}
    if pk.get("nano") is None:
        return None, {}
    chk = core_checks(st, base)
    ok = sum(1 for v in chk.values() if v)
    bad = len(chk) - ok
    score = pk["nano"] + 0.004 * ok - 0.010 * bad
    return score, chk


def pick_champion(led: dict, base: dict) -> dict | None:
    """选当前最强臂：必须有 picked 读数 + ckpt 文件真实存在。"""
    best = None
    for k, st in (led.get("arms") or {}).items():
        pk = st.get("picked") or {}
        if not pk:
            continue
        ep = pk.get("epoch")
        if ep is None:
            continue
        path = ckpt_path(int(k), int(ep))
        if not os.path.isfile(path):
            continue
        score, chk = champion_score(st, base)
        if score is None:
            continue
        cand = {"trial": int(k), "epoch": int(ep), "ckpt": path, "score": score,
                "checks": chk, "nano": pk.get("nano"), "nano4": pk.get("nano4"),
                "f1": pk.get("f1"), "micro": pk.get("micro"),
                "params": st.get("params") or {}, "desc": st.get("desc", "")}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


# ---------------------------------------------------------------------------
# ① 合成臂：Model Soup / SWA（零训练）
# ---------------------------------------------------------------------------
def soup_recipes(led: dict, champ: dict, base: dict) -> list[tuple[str, list[str], str]]:
    """产出 (配方名, 成员 ckpt 路径列表, 说明)。成员必须存在且同构。"""
    arms = led.get("arms") or {}
    scored = []
    for k, st in arms.items():
        pk = st.get("picked") or {}
        if not pk or pk.get("nano") is None:
            continue
        ep = pk.get("epoch")
        p = ckpt_path(int(k), int(ep)) if ep is not None else ""
        if p and os.path.isfile(p):
            scored.append({"trial": int(k), "epoch": int(ep), "path": p,
                           "nano": pk["nano"], "params": st.get("params") or {},
                           "desc": st.get("desc", "")})
    scored.sort(key=lambda x: -x["nano"])
    if not scored:
        return []

    champ_sw = (champ.get("params") or {}).get("sw")
    out = []

    top = scored[:3]
    if len(top) >= 2:
        out.append(("soup_top3", [x["path"] for x in top],
                    f"Model Soup：nano 前 3 名等权平均（{[x['trial'] for x in top]}）"))

    same_sw = [x for x in scored if (x["params"] or {}).get("sw") == champ_sw]
    if len(same_sw) >= 2:
        out.append(("soup_same_sw", [x["path"] for x in same_sw[:3]],
                    f"Model Soup：同 sw={champ_sw} 前 3 名等权平均"
                    f"（{[x['trial'] for x in same_sw[:3]]}，含种子复现臂）"))

    # SWA：冠军臂相邻 epoch
    cdir = os.path.dirname(champ["ckpt"])
    neigh = []
    for ep in range(max(1, champ["epoch"] - 2), champ["epoch"] + 3):
        p = f"{cdir}/epoch{ep}.pt"
        if os.path.isfile(p):
            neigh.append(p)
    if len(neigh) >= 2:
        out.append(("swa_neigh", neigh,
                    f"SWA：champion 臂 epoch{champ['epoch']-2}~{champ['epoch']+2} 等权平均"))

    late = []
    for ep in range(champ["epoch"], champ["epoch"] + 5):
        p = f"{cdir}/epoch{ep}.pt"
        if os.path.isfile(p):
            late.append(p)
    if len(late) >= 2:
        out.append(("swa_late", late,
                    f"SWA：champion 臂 epoch{champ['epoch']}~{champ['epoch']+4} 等权平均"))

    return out


def build_synth_arms(led: dict, champ: dict, base: dict, round_idx: int,
                     next_trial: list[int]) -> list[tuple]:
    """生成合成臂并把权重写到 trials/trial_X/weights/epoch5.pt。"""
    arms = []
    for name, members, note in soup_recipes(led, champ, base):
        trial = next_trial[0]
        next_trial[0] += 1
        out = f"{ns.ROOT}/trials/trial_{trial}/weights/epoch{SYNTH_EPOCH}.pt"
        try:
            average_checkpoints(members, out, epoch=SYNTH_EPOCH, note=note)
        except Exception as e:  # noqa: BLE001
            log(f"[synth] {name} 放弃：{e}")
            continue
        st = ns.arm_state(led, trial)
        st.update(trial=trial, desc=f"★ 合成臂 {name}：{note}", trained=True,
                  synth=True, synth_members=members, synth_note=note,
                  params={"init": champ["ckpt"], "synth": name, "epochs": SYNTH_EPOCH,
                          "sched": SYNTH_EPOCH, "sw": (champ.get("params") or {}).get("sw"),
                          "mix": "w2.0", "warmup": 1.5, "lr": 3.5, "fb": 2.75, "ow": 0.4,
                          "wd": 0.000269, "mosaic": 0.15, "scale": 0.1, "translate": 0.05,
                          "clip": 5.65, "backbone_lr": 1.0, "uf": 0, "head_src": "p3",
                          "native_grid": 0, "mr": 1, "swl": 1.0, "thr": 16.0,
                          "maxside": 4.0, "lr_floor": 0.0, "hf": 0, "hfsrc": "b1", "hfch": 1})
        log(f"[synth] {name} ⇒ trial {trial}（成员 {len(members)} 个）")
        arms.append((trial, 20000 + trial, st["params"], st["desc"]))
    ns.save_ledger(led)
    return arms


# ---------------------------------------------------------------------------
# ② 训练臂：以 champion 为 init 的单变量微调
# ---------------------------------------------------------------------------
def build_train_arms(champ: dict, round_idx: int, next_trial: list[int]) -> list[tuple]:
    p0 = dict(champ.get("params") or {})
    sw = float(p0.get("sw") or 1.0)
    init = champ["ckpt"]
    seed0 = 20000 + round_idx * 97
    specs = [
        (dict(p0, init=init, sw=sw, epochs=5, sched=5),
         f"★ 冠军自续训（sw={sw}，5ep）——检验稳定性/是否已收敛到平台"),
        (dict(p0, init=init, sw=round(sw * 1.15, 3), epochs=5, sched=5),
         f"★ 剂量上推 sw={round(sw * 1.15, 3)}"),
        (dict(p0, init=init, sw=round(sw * 0.85, 3), epochs=5, sched=5),
         f"★ 剂量下推 sw={round(sw * 0.85, 3)}"),
        (dict(p0, init=init, sw=sw, thr=25.0, maxside=6.0, epochs=5, sched=5),
         "★ 更宽的小目标定义（thr25/maxside6）——直击 nano@4px 严格档"),
        (dict(p0, init=init, sw=sw, epochs=5, sched=5),
         "★ 种子复现（确认增益可复现）"),
        (dict(p0, init=init, sw=sw, epochs=10, sched=10),
         "★ 退火周期 10ep"),
        (dict(p0, init=init, sw=sw, epochs=20, sched=5),
         "★ 长训 20ep（退火周期仍 5，取早期 epoch）"),
        (dict(p0, init=init, sw=sw, epochs=5, sched=5, lr=max(1.0, float(p0.get("lr") or 3.5) * 0.7)),
         "★ 更低 LR 精修（lr_scale×0.7）"),
    ]
    arms = []
    for i, (p, desc) in enumerate(specs):
        trial = next_trial[0]
        next_trial[0] += 1
        arms.append((trial, seed0 + i, p, f"[R{round_idx}] {desc}"))
    return arms


# ---------------------------------------------------------------------------
# 一轮
# ---------------------------------------------------------------------------
def run_round(round_idx: int, deadline_ts: float, gpus: list, next_trial: list) -> dict:
    led = ns.load_ledger()
    base = base_ref(led)
    champ = pick_champion(led, base)
    if champ is None:
        log("找不到可用 champion（需有 picked 读数 + ckpt 存在）⇒ 本轮跳过")
        return {"round": round_idx, "skipped": "no_champion"}

    log(f"[R{round_idx}] champion = trial {champ['trial']} ep{champ['epoch']} "
        f"nano={champ['nano']:.6f} score={champ['score']:.4f} checks={champ['checks']}")

    synth = build_synth_arms(led, champ, base, round_idx, next_trial)
    trains = build_train_arms(champ, round_idx, next_trial)
    all_arms = synth + trains
    for t, s, p, d in all_arms:
        ns.arm_state(led, t).update(trial=t, seed=s, desc=d, params=p)
    ns.save_ledger(led)

    stop_ts = min(deadline_ts, time.time() + 5 * 3600)
    log(f"[R{round_idx}] 训练 {len(trains)} 臂（合成 {len(synth)} 臂免训练）")
    ns.phase_train(trains, stop_ts, gpus)

    led = ns.load_ledger()
    gpu_of = {t: gpus[i % len(gpus)] for i, (t, _s, _p, _d) in enumerate(all_arms)}
    ns.phase_nanobench(all_arms, EPOCHS_EVAL, gpu_of)
    ns.phase_gate(all_arms, gpu_of)
    ns.phase_dual(all_arms)

    led = ns.load_ledger()
    base = base_ref(led)
    champ2 = pick_champion(led, base)
    beat = []
    for t, _s, _p, _d in all_arms:
        st = (led.get("arms") or {}).get(str(t)) or {}
        chk = core_checks(st, base)
        if chk and all(chk.values()):
            beat.append(t)
    # 板卡测速：只对「打平基线的臂」+ 当轮冠军做（其余臂不必占板卡）
    board_targets = [a for a in all_arms if a[0] in set(beat)]
    if champ2 and champ2["trial"] not in set(beat):
        board_targets += [a for a in all_arms if a[0] == champ2["trial"]]
    if board_targets:
        try:
            log(f"[R{round_idx}] 板卡测速：{[a[0] for a in board_targets]}")
            ns.phase_board(board_targets)
        except Exception as e:  # noqa: BLE001
            log(f"[R{round_idx}] 板卡测速失败（不影响本轮结论）：{e}")

    rec = {"round": round_idx, "ts": time.strftime("%F %T"),
           "champion_before": champ["trial"], "champion_after": (champ2 or {}).get("trial"),
           "arms": [t for t, _s, _p, _d in all_arms], "beat": beat,
           "board_measured": [a[0] for a in board_targets]}
    led.setdefault("rounds", []).append(rec)
    ns.save_ledger(led)
    ns.phase_report()
    try:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "beat_report.py")], check=False, timeout=600)
    except Exception as e:  # noqa: BLE001
        log(f"[beat_report] 失败：{e}")
    log(f"[R{round_idx}] 完成：champion → trial {(champ2 or {}).get('trial')}；"
        f"打平基线的臂 = {beat or '无'}")
    return rec


def selftest() -> None:
    """CPU-only 自检：验证 champion 选择 / 合成臂生成 / ckpt 写盘与回读（不碰台账、不占 GPU）。"""
    import torch
    led = ns.load_ledger()
    base = base_ref(led)
    champ = pick_champion(led, base)
    assert champ is not None, "champion 选择失败"
    log(f"champion = trial {champ['trial']} ep{champ['epoch']} nano={champ['nano']:.6f} "
        f"score={champ['score']:.4f} checks={champ['checks']}")
    recipes = soup_recipes(led, champ, base)
    assert recipes, "没有生成任何合成配方"
    tmpdir = "/tmp/adaptive_selftest"
    os.makedirs(tmpdir, exist_ok=True)
    for name, members, note in recipes:
        out = f"{tmpdir}/{name}.pt"
        average_checkpoints(members, out, epoch=SYNTH_EPOCH, note=note)
        ck = torch.load(out, map_location="cpu")
        src = torch.load(members[0], map_location="cpu")
        assert set(ck["model"]) == set(src["model"]), f"{name}: key 集合不一致"
        for k in ("no_p2", "head_level", "act", "head_conv", "late_upsample", "stride"):
            assert ck.get(k) == src.get(k), f"{name}: 结构字段 {k} 不一致"
        log(f"  ✓ {name}: {len(members)} 成员 → {os.path.getsize(out)} B，结构自描述一致，"
            f"keys={len(ck['model'])}")
    trains = build_train_arms(champ, 1, [TRIAL_BASE])
    log(f"训练臂 {len(trains)} 个：")
    for t, seed, pp, desc in trains:
        log(f"  · trial {t} seed={seed} sw={pp.get('sw')} thr={pp.get('thr')} "
            f"maxside={pp.get('maxside')} ep={pp.get('epochs')}/{pp.get('sched')} lr={pp.get('lr')} :: {desc}")
    log("SELFTEST PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description="自适应搜索调度器（目标：击败 Baseline_v2.0 / trial_9863）")
    ap.add_argument("--deadline", default="2026-11-12 09:00", help="硬截止")
    ap.add_argument("--rounds", type=int, default=99, help="最多轮数")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--trial-base", type=int, default=TRIAL_BASE)
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    ap.add_argument("--selftest", action="store_true", help="CPU-only 自检（不碰台账、不占 GPU）")
    args = ap.parse_args()

    deadline_ts = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))
    gpus = [int(x) for x in args.gpus.replace(" ", "").split(",") if x.strip()]
    next_trial = [args.trial_base]

    if args.selftest:
        selftest()
        return

    log(f"=== ADAPTIVE RUNNER START | deadline={args.deadline} | gpus={gpus} ===")
    for r in range(1, args.rounds + 1):
        if time.time() > deadline_ts - 600:
            log(f"已到截止前 10 min ⇒ 收工")
            break
        try:
            run_round(r, deadline_ts, gpus, next_trial)
        except Exception as e:  # noqa: BLE001
            log(f"[R{r}] 异常：{e}（继续下一轮）")
            import traceback
            log(traceback.format_exc())
        if time.time() > deadline_ts:
            log("超过截止 ⇒ 退出")
            break
    log("=== ADAPTIVE RUNNER DONE ===")


if __name__ == "__main__":
    main()
