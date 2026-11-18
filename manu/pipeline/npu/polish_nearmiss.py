#!/usr/bin/env python
"""近失臂抛光 side-job（2026-11-13 17:00 新增）。

**为什么要有这个旁路**
  · 当前最接近升轨的两个臂都是 5/6 近失：
      - trial_12679 @ ep3：nano 0.718809 / nano4 0.655718 / F1 0.890329 / T1 0.890550
                           / T2 fp 1 ✅ / **T3 0.948332（只差 1.57e-4）**
      - trial_96835 @ ep5（SWA of 12679 ep2+ep3，权重 0.3/0.7）：
                           nano 0.718336 / nano4 0.652647 / T3 0.948343（差 1.46e-4）
  · 但自适应调度器的抛光槽（``build_warm_arms``）是「锚点优先」的两层循环
    ⇒ 第一个锚点（12869@4）一个人吃掉全部 16 槽 ⇒ 上面这两个臂**一个抛光臂都拿不到**。
  ⇒ 本脚本作为**旁路**独立抛光这两个臂，并把读数**登记进台账**
    ⇒ 六项审计 / 板卡 T4 守望 / 后续抛光锚点都能看到它们。

**口径纪律**
  · 配方沿用**锚点真实配方**（12679 = sw4.0 / thr12 / maxside3 / mix w2.0），
    绝不用 ``ns.BASE``（sw=1.0）—— 那正是 v1 canary 掉 nano 1pp 的根因。
  · 输出目录必须在 ``ns.ROOT`` 下（官方门禁按固定路径 ``ROOT/trials/trial_X/weights/`` 找权重）。
  · L2 锚定指向**锚点自身**（``--anchor-ckpt`` = init）⇒ 抛光不偏离近失点。
  · 只用 GPU 0/1、nice 19 ⇒ 不抢自适应调度器的训练槽。
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
sys.path.insert(0, REPO + "/manu/pipeline/npu")
sys.path.insert(0, REPO)
os.chdir(REPO)

import nano_search as ns  # noqa: E402
from manu.pipeline.npu import route2_acceptance as RA  # noqa: E402

ROOT = ns.ROOT
CACHE_DIR = "/mnt/data/siping/runs/npu20ms/tiny_ref"
REPORT = "/mnt/data/siping/runs/npu20ms/nano_search/polish_nearmiss.json"

BASE = {
    "nano": 0.708648393194707,
    "nano4": 0.6472117202268431,
    "f1": 0.8897117275427806,
    "t1": 0.88808649190182,
    "t3": 0.9484893511639424,
    "t2_fp": 5,
}

# ---- 抛光锚点（5/6 近失臂）----------------------------------------------------
ANCHORS = [
    {"trial": 12679, "epoch": 3, "ckpt": f"{ROOT}/trials/trial_12679/weights/epoch3.pt"},
    {"trial": 96835, "epoch": 5, "ckpt": f"{ROOT}/trials/trial_96835/weights/epoch5.pt"},
]
# ---- 抛光配置 (lr_scale, L2 锚定 λ, freeze_bn, epochs) ------------------------
CONFIGS = [
    (0.20, 1e-3, 1, 1, 0.0, 12.0, 1.0, 1.0, 0),
    (0.60, 1e-3, 1, 1, 0.0, 12.0, 1.0, 1.0, 0),
    (0.35, 5e-4, 1, 2, 0.0, 12.0, 1.0, 1.0, 0),
]
TRIAL_BASE = 999821
GPUS = [0, 1]

_LOCK = threading.Lock()
_STATE: dict = {}


def log(msg: str) -> None:
    with _LOCK:
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)
        _STATE.setdefault("log", []).append(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}")


def save_report() -> None:
    with _LOCK:
        tmp = f"{REPORT}.tmp.{os.getpid()}"
        os.makedirs(os.path.dirname(REPORT), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_STATE, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, REPORT)


def anchor_params(led: dict, anchor: dict) -> dict:
    """锚点配方：台账里取原配方；台账没有（96835 是 SWA 合成臂）⇒ 退回同源臂 12679。"""
    st = (led.get("arms") or {}).get(str(anchor["trial"])) or {}
    p = dict(st.get("params") or {})
    if not p:
        src = (led.get("arms") or {}).get("12679") or {}
        p = dict(src.get("params") or {})
    for k, v in ns.BASE.items():
        p.setdefault(k, v)
    return p


def _used_warm_keys(led: dict) -> set:
    """台账里**真跑出读数**的抛光臂 warm_key（与调度器 dedup 同口径）。

    ⇒ 旁路第二批不会重复烧已被调度器/前一批用掉的 (锚, 配置) 组合。
    """
    out = set()
    for _st in (led.get("arms") or {}).values():
        _wk = (_st.get("params") or {}).get("warm_key")
        if _wk and ((_st.get("picked") or {}).get("nano") is not None
                    or bool(_st.get("nanobench"))):
            out.add(str(_wk))
    return out


def build_arms(led: dict, anchors: list[dict] | None = None,
               configs: list[tuple] | None = None,
               trial_base: int | None = None) -> list[dict]:
    """构造抛光臂（**配置序 × 锚点序**轮转 ⇒ 每个锚点先各领一个配置）。

    轮转理由（2026-11-13）：旧写法两层循环（锚点在外）⇒ 第一个锚点吃光所有配置；
    抛光轴的核心价值是「从多个近失臂各捞一个配置」，故改成配置优先。
    """
    anchors = anchors if anchors is not None else ANCHORS
    configs = configs if configs is not None else CONFIGS
    tid = int(trial_base if trial_base is not None else TRIAL_BASE)
    used = _used_warm_keys(led)
    # ---- 锚点过滤（权重必须存在）----
    valid = []
    for a in anchors:
        if not os.path.isfile(a["ckpt"]):
            log(f"[cfg] 跳过锚点 {a['trial']}@{a['epoch']}（权重不存在：{a['ckpt']}）")
            continue
        valid.append(a)
    # ---- 配置序 × 锚点序 轮转 ----
    combos = []
    for _cfg in configs:
        for _a in valid:
            combos.append((_a, _cfg))
    arms: list[dict] = []
    for (a, _cfg) in combos:
        (lr, l2, fb, ep, cp_prob, cp_px, sc_min, sc_max, rep) = _cfg
        _wk = f"w{a['trial']}e{a['epoch']}|lr{lr}|l2{l2}|fb{fb}|ep{ep}"
        if cp_prob > 0:
            # 🟢 cp 抛光臂必须与「同 (lr,l2,fb,ep) 的普通抛光臂」区分开
            #   ⇒ warm_key 加 cp 后缀 ⇒ 否则会互相顶掉（去重是跨进程共享的）
            _wk += f"|cp{cp_prob:g}px{cp_px:g}sc{sc_min:g}x{sc_max:g}"
        if rep > 0:
            # 🟢 复现臂：同一 (锚点, lr, λ, fb, ep, cp) 但**换随机种子**
            #   ⇒ warm_key 必须带 rep 标记，否则会被跨进程去重表顶掉。
            _wk += f"|rep{rep}"
        if _wk in used:
            log(f"[cfg] 跳过重复组合（台账已跑过）：{_wk}")
            continue
        used.add(_wk)
        base_p = anchor_params(led, a)
        p = dict(base_p)
        p.update({
            "init": a["ckpt"], "epochs": int(ep), "sched": int(ep), "warmup": 0.1,
            "lr": float(lr), "anchor_l2": float(l2), "anchor_ckpt": a["ckpt"],
            "freeze_bn": int(fb),
            "warm_key": _wk,
            "warm_anchor": f"{a['trial']}@{a['epoch']}",
            "rep": int(rep),
        })
        if cp_prob > 0:
            p.update({"cp_prob": float(cp_prob), "cp_max_px": float(cp_px),
                      "cp_scale_min": float(sc_min), "cp_scale_max": float(sc_max),
                      "cp_ctx_px": 3.0, "cp_feather_px": 2.0, "cp_per_img": 3,
                      "cp_min_gap_px": 2.0})
        arms.append({"trial": tid, "gpu": None, "seed": 50000 + tid + rep * 1_000_000,
                     "params": p, "anchor": a,
                     "cfg": (lr, l2, fb, ep, cp_prob, cp_px, sc_min, sc_max, rep)})
        tid += 1
    return arms


def train(arm: dict, gpu: int) -> bool:
    p = arm["params"]
    cmd = ns.train_cmd(arm["trial"], gpu, arm["seed"], p)
    log(f"[train] trial={arm['trial']} gpu={gpu} anchor={p['warm_anchor']} "
        f"lr={p['lr']} λ={p['anchor_l2']} fb={p['freeze_bn']} ep={p['epochs']}")
    pr = subprocess.Popen(["bash", "-c", "nice -n 19 bash -c " + shlex.quote(cmd)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    rc = pr.wait()
    ck = f"{ROOT}/trials/trial_{arm['trial']}/weights/epoch{p['epochs']}.pt"
    ok = rc == 0 and os.path.isfile(ck)
    log(f"[train] trial={arm['trial']} rc={rc} ckpt={'OK' if ok else 'MISSING'}")
    return ok


def evaluate(arm: dict, gpu: int) -> dict | None:
    p = arm["params"]
    ck = f"{ROOT}/trials/trial_{arm['trial']}/weights/epoch{p['epochs']}.pt"
    if not os.path.isfile(ck):
        return None
    pkl = f"{CACHE_DIR}/polish_{arm['trial']}.pkl"
    if not RA.build_cache(ck, pkl, gpu, extra="--no-peak-attrs"):
        log(f"[eval] trial={arm['trial']} 建缓存失败")
        return None
    r = RA.nanobench(pkl, f"polish_{arm['trial']}")
    if not r:
        log(f"[eval] trial={arm['trial']} NanoBench 失败")
        return None
    rb8 = r["peak_tol8"]["recall_by_bucket"]
    rb4 = r["peak_tol4"]["recall_by_bucket"]
    m = {"nano": float(rb8["nano"]), "nano4": float(rb4["nano"]),
         "f1": float(r["peak_tol8"]["f1"]), "micro": float(rb8["micro"]),
         "small": float(rb8["small"]), "regular": float(rb8["regular"]),
         "f1_4": float(r["peak_tol4"]["f1"]), "cache": pkl,
         "epoch": int(p["epochs"]), "epoch_key": str(int(p["epochs"])),
         "ckpt": f"epoch{p['epochs']}.pt"}
    triad = (m["nano"] >= BASE["nano"] and m["nano4"] >= BASE["nano4"]
             and m["f1"] >= BASE["f1"])
    m["triad"] = bool(triad)
    log(f"[eval] trial={arm['trial']} nano={m['nano']:.6f} nano4={m['nano4']:.6f} "
        f"f1={m['f1']:.6f} triad={triad}")
    if triad:
        g = ns.run_official_gate(arm["trial"], int(p["epochs"]), gpu)
        if g and "error" not in g:
            m["official"] = {k: g.get(k) for k in
                             ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")}
            six = (g["t1"] >= BASE["t1"] and g["t2_fp"] <= BASE["t2_fp"]
                   and g["t3"] >= BASE["t3"] and triad)
            m["six_ok"] = bool(six)
            log(f"[gate] trial={arm['trial']} T1={g.get('t1')} T2fp={g.get('t2_fp')} "
                f"T3={g.get('t3')} six_ok={six}")
        else:
            log(f"[gate] trial={arm['trial']} 门禁失败：{(g or {}).get('error')}")
    return m


def register(arm: dict, m: dict) -> None:
    """把抛光臂登记进台账（口径与普通臂一致 ⇒ 六项审计 / T4 守望 / 抛光锚点都能看到）。"""
    p = dict(arm["params"])
    p["init"] = arm["params"]["init"]
    nb = {k: m.get(k) for k in ("nano", "nano4", "f1", "micro", "small", "regular",
                                "f1_4", "epoch", "epoch_key", "ckpt", "cache")}
    st = {
        "trial": arm["trial"],
        "polish": True,
        "source_anchor": arm["params"]["warm_anchor"],
        "polish_cfg": {"lr": arm["params"]["lr"], "anchor_l2": arm["params"]["anchor_l2"],
                       "freeze_bn": arm["params"]["freeze_bn"], "epochs": arm["params"]["epochs"],
                       "cp_prob": arm["params"].get("cp_prob", 0.0),
                       "cp_max_px": arm["params"].get("cp_max_px"),
                       "cp_scale_min": arm["params"].get("cp_scale_min"),
                       "cp_scale_max": arm["params"].get("cp_scale_max"),
                       "rep": arm["params"].get("rep", 0)},
        "params": p,
        "seed": None,
        "nanobench": {str(m["epoch"]): nb},
        "picked": dict(nb),
        "picked_epoch": int(m["epoch"]),
        "registered_by": "polish_nearmiss",
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if m.get("official"):
        st["official"] = m["official"]
        st["six_ok"] = bool(m.get("six_ok"))
    led = ns.load_ledger()
    arms = led.get("arms") or {}
    arms[str(arm["trial"])] = st
    led["arms"] = arms
    ns.save_ledger(led)
    log(f"[登记] trial={arm['trial']} 已进台账")


def worker(gpu: int, arms: list[dict], deadline: float | None = None) -> None:
    for arm in arms:
        if deadline is not None and time.time() > deadline:
            log(f"[截止] 已过截止时间 ⇒ 停止本 GPU 的剩余抛光臂（{gpu}）")
            return
        try:
            if not train(arm, gpu):
                continue
            m = evaluate(arm, gpu)
            if m:
                register(arm, m)
                with _LOCK:
                    _STATE.setdefault("results", {})[str(arm["trial"])] = {
                        "anchor": arm["params"]["warm_anchor"],
                        "cfg": {"lr": arm["params"]["lr"],
                                "anchor_l2": arm["params"]["anchor_l2"],
                                "freeze_bn": arm["params"]["freeze_bn"],
                                "epochs": arm["params"]["epochs"],
                                "cp_prob": arm["params"].get("cp_prob", 0.0),
                                "cp_max_px": arm["params"].get("cp_max_px"),
                                "cp_scale_min": arm["params"].get("cp_scale_min"),
                                "cp_scale_max": arm["params"].get("cp_scale_max")},
                        **{k: m.get(k) for k in ("nano", "nano4", "f1", "triad",
                                                 "official", "six_ok")},
                    }
                save_report()
        except Exception as exc:  # noqa: BLE001 —— 单臂失败不掀掉整轮
            log(f"[err] trial={arm.get('trial')} 失败：{type(exc).__name__}: {exc}")
        finally:
            save_report()


def _parse_anchors(spec: str) -> list[dict]:
    """``trial:epoch,trial:epoch`` ⇒ 锚点列表（权重路径按 ROOT 约定拼）。"""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"锚点格式应为 trial:epoch，收到 {tok!r}")
        t, e = tok.split(":")
        out.append({"trial": int(t), "epoch": int(e),
                    "ckpt": f"{ROOT}/trials/trial_{int(t)}/weights/epoch{int(e)}.pt"})
    return out


def _parse_configs(spec: str) -> list[tuple]:
    """``lr:l2:fb:ep[:cp_prob[:cp_max_px[:cp_scale_min:cp_scale_max[:rep]]]],...``

    ⇒ 统一返回 9 元组 ``(lr, l2, fb, ep, cp_prob, cp_max_px, cp_scale_min, cp_scale_max, rep)``。

    🟢🟢 2026-11-13 新增 **抛光 + Copy-Paste 组合轴**：
       在抛光的最后 1 个 epoch 里同时开微小目标 Copy-Paste ⇒
       「已打穿的抛光（L2 锚定，保 T3）」＋「cp（补 ≤4px 正样本，抬 nano4）」同时作用。
       cp 字段缺省 ⇒ cp_prob=0 ⇒ 与原抛光臂**逐字等价**（旧配置串照旧可用）。

    🟢🟢 2026-11-13 新增 **rep 复现维**：
       第 9 段 = 复现序号（默认 0）⇒ 同一配方换随机种子重跑
       ⇒ 用于判定「六项全过臂」是**配方**还是**运气臂**（对应 memory 的 12869 教训）。
    """
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) not in (4, 5, 6, 7, 8, 9):
            raise ValueError(
                f"配置格式应为 lr:l2:fb:ep[:cp_prob[:cp_max_px[:sc_min:sc_max[:rep]]]]，收到 {tok!r}")
        lr, l2, fb, ep = parts[:4]
        cp_prob = float(parts[4]) if len(parts) >= 5 else 0.0
        cp_px = float(parts[5]) if len(parts) >= 6 else 12.0
        sc_min = float(parts[6]) if len(parts) >= 7 else 1.0
        sc_max = float(parts[7]) if len(parts) >= 8 else 1.0
        rep = int(parts[8]) if len(parts) >= 9 else 0
        out.append((float(lr), float(l2), int(fb), int(ep),
                    cp_prob, cp_px, sc_min, sc_max, rep))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="近失臂抛光 side-job")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--anchors", default="", help="trial:epoch,trial:epoch（默认=内置锚点）")
    ap.add_argument("--configs", default="", help="lr:l2:fb:ep,...（默认=内置配置）")
    ap.add_argument("--trial-base", type=int, default=TRIAL_BASE)
    ap.add_argument("--gpus", default="", help="逗号分隔 GPU 列表（默认=内置）")
    ap.add_argument("--deadline", default="", help="本地时间 'YYYY-MM-DD HH:MM' ⇒ 到点停")
    ap.add_argument("--report", default="", help="结果 JSON 路径（默认=内置）")
    args = ap.parse_args()

    anchors = _parse_anchors(args.anchors) if args.anchors else None
    configs = _parse_configs(args.configs) if args.configs else None
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()] if args.gpus else GPUS
    dl = None
    if args.deadline:
        dl = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))

    if args.report:
        global REPORT
        REPORT = args.report
    led = ns.load_ledger()
    arms = build_arms(led, anchors=anchors, configs=configs, trial_base=args.trial_base)
    log(f"[cfg] 抛光臂 {len(arms)} 个；GPU={gpus}；"
        + ("截止 " + args.deadline if args.deadline else "无截止"))
    for a in arms:
        p = a["params"]
        log(f"[cfg] trial={a['trial']} anchor={p['warm_anchor']} lr={p['lr']} "
            f"λ={p['anchor_l2']} fb={p['freeze_bn']} ep={p['epochs']} "
            f"cp={p.get('cp_prob', 0)} px={p.get('cp_max_px', '-')} "
            f"sw={p.get('sw')} thr={p.get('thr')} ms={p.get('maxside')}")
    if args.dry_run:
        return
    save_report()

    buckets: dict[int, list[dict]] = {g: [] for g in gpus}
    for i, a in enumerate(arms):
        buckets[gpus[i % len(gpus)]].append(a)
    ths = [threading.Thread(target=worker, args=(g, buckets[g], dl), daemon=False) for g in gpus]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    log("[done] 全部抛光臂处理完毕")
    save_report()


if __name__ == "__main__":
    main()
