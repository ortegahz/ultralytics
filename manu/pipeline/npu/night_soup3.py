#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**三路汤轴（3-way soup）** —— 2026-11-18 白天波。

为什么必须开这条轴（2026-11-18 08:50 实测，不是猜测）
--------------------------------------------------
昨夜把「六项判据」拆成了三个**互不兼容**的物理量：

1. ``nano / nano4 / f1``（台账 bench，微小目召回）：由**子臂权重**决定。
   * 冠军子臂 ``1100296@1`` 家族（``bs640001:*``）⇒ bench **钉死**在
     nano .7204631 / nano4 .6561909 / f1 .8904027（= 当前基线，**没有一格更高**）。
   * 换子臂 ``1100277@2`` 家族（``bs1120000``/``bs640001:w55|w58|w62``）⇒ bench **更高**：
     nano .7216446（+1.18e-3）/ nano4 .6568998（+7.1e-4）/ f1 .8904760 / **T1 .8908**（+2.3e-4），
     且 **T2_fp 仍然是 1**（同属 bs640001 盆地）⇒ **只差 T3**。
2. ``T3``（盲测硬例段 macro recall）：**量化台阶**，取值集合
   {.9483168, .9483740, .9484311, .9484730, .9485822, .9486405, .9487598, .9489159, …}，
   步长 ~1.1e-4；判据线 .9485822 正好卡在两阶之间。
   实测：**同 parent + 同子臂 ⇒ 权重 w 在 .53~.67 全域内 T3 恒定**
   （bs640001+1100296@1 ⇒ 恒 .9485822；bs640001+1100277@2 ⇒ 恒 .9484730）
   ⇒ **T3 是「子臂决定的台阶」，不是「权重可微调」的量** ⇒ 细扫 w 永远翻不过 T3。
3. ⇒ **唯一出路 = 把两个台阶混在一个汤里**：
   ``汤 = a·640001@5（干净父, T2_fp=1 的来源） + b·1100296@1（T3 好台阶锚） + c·1100277@2（bench 增益子）``
   期望：b 足够大 ⇒ 保住 T3 .9485822；c 提供 nano/nano4 增益；a 保证 T2_fp=1。
   **这正是昨夜 premix 轴没能做到的**（premix 只在 1100296@1/@2 之间混 ⇒ bench 的 nano4 反而掉到 .6545~.6552 ⇒ 死路）。

本脚本做的事
------------
* 支持 **N 路（≥2）权重汤**：``--recipes "640001:5:0.40,1100296:1:0.45,1100277:2:0.15;..."``
* 合成 ⇒ 建缓存 ⇒ NanoBench（nano/nano4/f1）⇒ 回写台账 ⇒ 对「三轴 ≥ 基线」的臂跑官方门禁；
* **零训练成本**（权重平均），单臂合成 <1 s、bench ~1 min、门禁 ~10~35 min；
* 资源铁律：起任务前查内存（MemAvailable < --mem-gate-gb ⇒ 等 120 s），门禁并发靠调用方限流；
* 幂等：同 (成员, 权重) 组合复用同臂号（跨进程用 flock 原子占号）。

用法（服务器）
    python -u manu/pipeline/npu/night_soup3.py --gpu 3 \
        --recipes "640001:5:0.40,1100296:1:0.45,1100277:2:0.15" \
        --sid-base 1400000 --sid-span 400 --max-gates 40 --deadline "2026-11-19 07:00"
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
SIDE_CAR = "basin_soup_synth.json"
LOCK_PATH = "/tmp/sid_alloc3.lock"

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

from manu.pipeline.npu import nano_search as NS  # noqa: E402
from manu.pipeline.npu import route2_acceptance as RA  # noqa: E402

LOGF = os.path.join(W, "soup3.log")

# 🔴🔴 号段独占锁：同一个 `--sid-base` 号段**同时只能有一个合成进程**。
#
# 为什么需要（2026-11-23 10:29~10:31 实测事故）：
#   `t1close_synth.sh` 的 flock 挂在**包装 shell**上，而真正干活的 `night_soup3.py`（经 timeout）
#   是它的**子进程**。包装 shell 一死（SIGHUP/SIGTERM/被守望收割）⇒ flock 释放 ⇒
#   守望按「PID 文件已死」拉起第二个实例 ⇒ **两个 `night_soup3` 拿着同一个 `--sid-base`
#   和同一批配方并行跑**（实测 pid 819483 与 820183 的 `--recipes` 逐字节相同）。
#   台账本身是安全的（`reserve_sid` 有 flock + 读-合并-写事务 ⇒ 不会撞号），
#   但 `trial_<sid>/weights/epoch5.pt` 与 `soup3_<sid>.pkl` **原本直接写最终路径**
#   ⇒ 两个进程并发写同一个文件 ⇒ 产物可能被截断/串写。
#
# ⇒ 在**进程内**按号段加锁：这是「号段独占」这条不变式的最后一道闸，
#   与「谁启动的」（cron / 守望 / 手工 ssh）完全无关。
_SIDRANGE_LOCK_FD = None


def acquire_sid_range_lock(base: int, span: int):
    """按号段加**非阻塞**独占锁。成功返回持有锁的 fd（须保持存活），失败返回 None。"""
    import fcntl  # noqa: PLC0415

    path = os.path.join("/tmp", f"soup3_sidrange_{int(base)}.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


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
        log(f"[内存闸门] MemAvailable={mem_avail_gb():.1f}GB < {threshold:.0f}GB ⇒ 等 120s")
        time.sleep(120)
    return True


def ck_path(trial: int, epoch) -> str:
    ep = int(epoch) if str(epoch).replace(".", "").isdigit() else epoch
    return os.path.join(ROOT, "trials", f"trial_{int(trial)}", "weights", f"epoch{ep}.pt")


def margin(picked: dict, base: dict) -> float:
    return min((picked.get("nano", 0) - base["nano"]) / 0.0042,
               (picked.get("nano4", 0) - base["nano4"]) / 0.0060,
               (picked.get("f1", 0) - base["f1"]) / 0.0008)


def fam_tag(members, weights) -> str:
    """家族标签：**成员集合 + 增益子权重档（1% 粒度）** ⇒ 同家族 = 同一方法、权重邻域。

    🔴 为什么不能按「权重精确到 1e-4」分组（2026-11-18 09:50 实测）：
      升轨的两道家族关都按 `soup_family` 字符串分组
      （`promote_candidate.soup_family_stats/soup_family_six_ok` + `auto_promote_watch` 的 n≥4）
      ⇒ 若每份配方各成一族，**每族永远 n=1 ⇒ 永远升不了轨**。
      两路汤的老口径是 `bs<parent>:<权重档>`（权重取整到 1%）⇒ 同档邻域的臂天然同族。
      本函数沿用同一思想：成员集合（方法）+ **最后一个成员（增益子）的权重档**。
    """
    parts = [f"{int(t)}@{int(e)}" for (t, e) in members]
    bw = int(round(float(weights[-1]) * 100)) if weights else 0
    return "sb3:" + "+".join(parts) + f":bw{bw:02d}"


def reserve_sid(base: int, span: int, members, weights, family: str):
    """独占号段 + flock 原子占号（同配方同号 ⇒ 断点续跑幂等）。"""
    import fcntl
    key = tuple(sorted((int(m[0]), int(m[1])) for m in members))
    wkey = tuple(round(float(x), 6) for x in weights)
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
                    pick = cand
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


def parse_recipes(text: str):
    """'640001:5:0.40,1100296:1:0.45,1100277:2:0.15;...' ⇒ [[(t,e,w), ...], ...]"""
    out = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        members = []
        for tok in chunk.split(","):
            tok = tok.strip()
            if not tok:
                continue
            t, e, w = tok.split(":")
            members.append((int(t), int(e), float(w)))
        if len(members) >= 2:
            out.append(members)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--recipes", default="",
                    help="N 路汤配方，';' 分隔多个，',' 分隔成员，成员 = trial:epoch:weight"
                         "（--only-gate 模式可为空）")
    ap.add_argument("--sid-base", type=int, default=1400000)
    ap.add_argument("--sid-span", type=int, default=400)
    ap.add_argument("--max-gates", type=int, default=40)
    ap.add_argument("--mem-gate-gb", type=float, default=44.0)
    ap.add_argument("--deadline", default="2026-11-19 07:00")
    ap.add_argument("--only-gate", action="store_true",
                    help="只跑门禁（不合成）：捞台账里 sb3 家族、三轴 ≥ 基线 且未门禁的臂")
    args = ap.parse_args()

    # 🔴 号段独占（双实例的最后一道闸；见 acquire_sid_range_lock 的事故说明）
    global _SIDRANGE_LOCK_FD
    if not args.only_gate:
        _SIDRANGE_LOCK_FD = acquire_sid_range_lock(args.sid_base, args.sid_span)
        if _SIDRANGE_LOCK_FD is None:
            log(f"[3汤] 🔴 号段 [{args.sid_base},{args.sid_base + args.sid_span}) 已被另一个合成进程"
                f"占用 ⇒ 本实例退出（防双实例重复合成 / 并发写同一 ckpt+pkl）")
            raise SystemExit(3)
        log(f"[3汤] 号段锁已持有 sid=[{args.sid_base},{args.sid_base + args.sid_span})")

    dl = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))
    base = NS.BASE_V2
    recipes = parse_recipes(args.recipes)
    log(f"[3汤] 配方 {len(recipes)} 个｜基线 nano={base['nano']:.7f} nano4={base['nano4']:.7f} "
        f"f1={base['f1']:.7f} t1={base['t1']:.7f} t3={base['t3']:.7f}")
    for r in recipes:
        log(f"[3汤] 配方 = {[(t, e, round(w, 4)) for t, e, w in r]}")

    from manu.pipeline.npu.ckpt_synth import average_checkpoints  # noqa: PLC0415

    pending = []
    n_gate = 0
    n_soup = 0
    if args.only_gate:
        led = NS.load_ledger()
        for k, st in sorted(((k, v) for k, v in (led.get("arms") or {}).items()), key=lambda x: int(x[0])):
            if not str(st.get("soup_family") or "").startswith("sb3:"):
                continue
            pk = st.get("picked") or {}
            of = st.get("official") or {}
            if of.get("t1") is not None and of.get("t3") is not None:
                continue
            gdir = os.path.join(ROOT, "gate", f"NANO_{int(k)}_ep5_official")
            if os.path.isfile(os.path.join(gdir, "gate_report.json")):
                continue
            if os.path.isdir(gdir):
                continue
            if not (pk.get("nano", 0) >= base["nano"] and pk.get("nano4", 0) >= base["nano4"]
                    and pk.get("f1", 0) >= base["f1"]):
                continue
            pending.append((margin(pk, base), int(k), st.get("soup_family") or "?", pk))
        pending.sort(key=lambda x: -x[0])
        log(f"[3汤] 只门禁模式：候选 {len(pending)} 个")
        for x in pending:
            log(f"   待门禁 sid={x[1]} fam={x[2]} m={x[0]:+.3f}σ")
    else:
        for members in recipes:
            if time.time() > dl:
                log("[3汤] 已过截止 ⇒ 收工")
                break
            ws = [w for (_t, _e, w) in members]
            mt = [(t, e) for (t, e, _w) in members]
            fam = fam_tag(mt, ws)
            sid = reserve_sid(args.sid_base, args.sid_span, mt, ws, fam)
            if sid is None:
                log(f"[3汤] 独占号段已满 ⇒ 收工")
                break
            # 🟢🟢 2026-11-18 10:55 **已合成 ⇒ 跳过**（幂等续跑）
            #   坑：night_soup3 每轮循环都会把同一批配方**重新合成 + 重建缓存 + 重评测**
            #        ⇒ 一轮 D2/D3 波（~60 臂）每轮重复消耗数小时算力，真正的新配方反而排不上。
            #   修法：台账已有 picked + ckpt + pkl 都在 ⇒ 直接跳过（合成是确定性的，重做无意义）。
            _led0 = NS.load_ledger()
            _st0 = (_led0.get("arms") or {}).get(str(sid))
            _ck0 = os.path.join(ROOT, "trials", f"trial_{sid}", "weights", "epoch5.pt")
            _pk0 = os.path.join("/mnt/data/siping/runs/npu20ms/tiny_ref", f"soup3_{sid}.pkl")
            if _st0 and _st0.get("picked") and os.path.isfile(_ck0) and os.path.isfile(_pk0):
                log(f"[3汤] sid={sid} 已合成 ⇒ 跳过")
                continue
            paths = [ck_path(t, e) for (t, e) in mt]
            if not all(os.path.isfile(p) for p in paths):
                log(f"[3汤] 权重缺失 {paths} ⇒ 跳过")
                continue
            out_dir = os.path.join(ROOT, "trials", f"trial_{sid}", "weights")
            os.makedirs(out_dir, exist_ok=True)
            out_ck = os.path.join(out_dir, "epoch5.pt")
            # 🔴 原子发布：先写「同目录临时文件」再 os.replace ⇒ 任何时刻 out_ck 要么是旧的
            #    完整文件、要么是新的完整文件，**绝不会是写了一半的**。
            #    （守望的「收割孤儿」会 kill -9 掉写盘中途的进程 ⇒ 直接写最终路径必然留残档）
            tmp_ck = f"{out_ck}.tmp.{os.getpid()}"
            try:
                average_checkpoints(paths, tmp_ck, weights=ws, epoch=5,
                                    note="soup3_" + "_".join(f"{t}@{e}w{w:.4f}" for t, e, w in members))
                os.replace(tmp_ck, out_ck)
            except Exception as exc:  # noqa: BLE001
                log(f"[3汤] 合成失败 {fam}：{exc}")
                _unlink(tmp_ck)
                continue
            pkl = f"/mnt/data/siping/runs/npu20ms/tiny_ref/soup3_{sid}.pkl"
            tmp_pkl = f"{pkl}.tmp.{os.getpid()}"
            try:
                _unlink(tmp_pkl)  # 清掉上一次崩溃留下的残档（否则 build_cache 会误判「已存在」）
                if not RA.build_cache(out_ck, tmp_pkl, args.gpu, extra="--no-peak-attrs"):
                    log(f"[3汤] 缓存失败 sid={sid}")
                    _unlink(tmp_pkl)
                    continue
                res = NS._nb_pool().submit(NS._nb_worker, (tmp_pkl, f"soup3_{sid}")).result()
            except Exception as exc:  # noqa: BLE001
                log(f"[3汤] 评测异常 sid={sid}：{exc}")
                _unlink(tmp_pkl)
                continue
            if not res:
                log(f"[3汤] 评测失败 sid={sid}")
                _unlink(tmp_pkl)
                continue
            # 评测通过才发布缓存 ⇒ 挂到台账上的永远是「已评测过」的完整缓存
            os.replace(tmp_pkl, pkl)
            m = {"nano": res["nano"], "nano4": res["nano4"], "f1": res["f1"],
                 "micro": res.get("micro"), "small": res.get("small"), "regular": res.get("regular"),
                 "f1_4": res.get("f1_4"), "cache": pkl}
            # 🟢🟢 2026-11-18 15:40 **三轴筛选放宽（F1 容差 + nano4 增益例外）**
            #   背景：基线抬到 v2.7 后 f1=.8905687 ⇒ 「f1 差 1e-4 但 nano4 高 5e-3」的臂被挡在门外，
            #   而这类臂恰恰是**小目标召回更强**的候选（用户最关心的 nano4 轴）⇒ 需要门禁读数才能进晨报。
            #   规则（**仅决定「是否值得花门禁机时」**，六项判据本身不变）：
            #     · nano 必须 > 基线、nano4 必须 > 基线；
            #     · 且（f1 > 基线）**或**（f1 > 基线 − F1_TOL 且 nano4 ≥ 基线 + NANO4_GAIN）。
            F1_TOL = 2.0e-4
            NANO4_GAIN = 3.0e-3
            triad = bool(m["nano"] > base["nano"] and m["nano4"] > base["nano4"]
                         and (m["f1"] > base["f1"]
                              or (m["f1"] > base["f1"] - F1_TOL
                                  and m["nano4"] >= base["nano4"] + NANO4_GAIN)))
            mg = margin(m, base)
            log(f"[3汤] sid={sid} fam={fam} ⇒ nano={m['nano']:.7f} nano4={m['nano4']:.7f} "
                f"F1={m['f1']:.7f} m={mg:+.3f}σ 三轴全过={triad}")
            n_soup += 1
            led2 = NS.load_ledger()
            st = NS.arm_state(led2, sid)
            st.update({
                "trial": sid, "synth": True, "source_trial": int(mt[0][0]),
                "soup_members": [[int(t), int(e)] for (t, e) in mt],
                "soup_weights": [float(w) for w in ws], "soup_family": fam,
                "params": {"init": out_ck, "sw": None, "lr": None, "lr_floor": None,
                           "mix": "soup3", "fb": None, "ow": None, "sched": None,
                           "backbone_lr": None, "ema_decay": None, "swb": None,
                           "swb_thr": None, "cp_prob": None, "cp_max_px": None},
                "seed": None,
                "nanobench": {"5": m}, "picked": dict(m), "picked_epoch": 5,
                "official": None, "six_ok": False,
                "desc": "[SB3] soup(" + ",".join(f"{t}@{e}:w{w:.4f}" for (t, e, w) in members) + ")",
                "channel": "SB3", "registered_by": "night_soup3",
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            NS.save_ledger(led2)
            try:
                with open(os.path.join(out_dir, SIDE_CAR), "w", encoding="utf-8") as fh:
                    json.dump({"key": "soup3:" + fam, "scored": True,
                               "members": [[int(t), int(e)] for (t, e) in mt],
                               "weights": [float(w) for w in ws], "family": fam, "tag": "soup3",
                               "basin": int(mt[0][0]),
                               "nanobench": {k: m.get(k) for k in ("nano", "nano4", "f1")},
                               "triad": triad, "gate": None,
                               "entry": {"family": fam, "tag": "soup3",
                                         "members": [[int(t), int(e)] for (t, e) in mt],
                                         "weights": [float(w) for w in ws], "synth_id": sid,
                                         "nanobench": m, "triad": triad, "gate": None}},
                              fh, ensure_ascii=False, indent=2)
            except Exception as exc:  # noqa: BLE001
                log(f"[3汤] sidecar 写失败：{exc}")
            pending.append((mg, sid, fam, m))

    # ---- 门禁：只打「三轴 ≥ 基线」的臂；家族配额 + 内存闸门 + 截止 ----
    pending.sort(key=lambda x: -x[0])
    log(f"[3汤] 待门禁 {len(pending)} 个")
    for mg, s_id, fam, m in pending:
        if time.time() > dl:
            log("[3汤] 已过截止 ⇒ 门禁阶段收工")
            break
        if n_gate >= args.max_gates:
            log(f"[3汤] 门禁次数用满（{n_gate}/{args.max_gates}）⇒ 收工")
            break
        if not wait_mem(args.mem_gate_gb, dl):
            log("[3汤] 内存闸门/截止 ⇒ 门禁阶段收工")
            break
        n_gate += 1
        g = NS.run_official_gate(s_id, 5, args.gpu)
        gate = {k: (g or {}).get(k) for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")}
        gate["six_ok"] = bool(gate.get("t1") is not None and gate.get("t3") is not None
                             and gate["t1"] >= base["t1"]
                             and gate.get("t2_fp") is not None and gate["t2_fp"] <= base["t2_fp"]
                             and gate["t3"] >= base["t3"])
        log(f"[3汤] 门禁 sid={s_id} fam={fam} ⇒ T1={gate.get('t1')} T2fp={gate.get('t2_fp')} "
            f"T3={gate.get('t3')} six_ok={gate.get('six_ok')}")
        led2 = NS.load_ledger()
        st = NS.arm_state(led2, s_id)
        st["official"] = gate
        st["six_ok"] = bool(gate["six_ok"])
        NS.save_ledger(led2)
    log(f"[3汤] 收工：合成 {n_soup} 个，送门禁 {n_gate} 个")


if __name__ == "__main__":
    main()
