#!/usr/bin/env python3
"""夜间流水线看门狗（主机 18:00 关机后仍自动运行，不依赖主机在线）。

背景（2026-11-09 17:45 审计结论）：
  overnight.py 的 `sh()` 用 subprocess.run(timeout=...)，但**没有 try/except**。
  ⇒ 任何一个 arm 的 screen / train / build_rknn / board_measure 超时，异常会一路冒泡出 main()，
    脚本当场死掉，且 `overnight_result.json`（写在阶段2之后）**永远不会被写出**，
    官方三轨门禁（阶段4）也因此**永远不会执行**。
  这是「整条流水线因单点失败全灭」的真实风险，看门狗就是为此准备的。

职责（严格保守，不与 overnight.py 抢活）：
  1) 轮询 overnight.py 是否存活；**只要它活着，看门狗只观察不动手**（避免与它并发碰板卡）。
  2) overnight.py 死亡后：独立从 `overnight_screen_*.log` 汇总所有 silu 臂的 F1
     （不依赖 overnight.py 的内存状态，它一死那些状态就没了）。
  3) 若阶段2 的 8 个种子臂没跑完 ⇒ 用 4 卡并行补齐（复用 overnight.py 的 train_silu / screen）。
  4) 若最佳 silu 臂 > trial_0031(0.886899) 且**该臂尚未上板测过** ⇒ 建 rknn + 一次会话上板测速。
  5) 若最佳臂还没跑官方门禁 ⇒ 跑官方三轨门禁。
  6) 无论成败，最后**一定**写出 overnight_result.json（arm 汇总 + board + gate）。

安全约束：
  * 板卡驱动脆弱（反复长任务 → soft reset → rknn_init 永远 132 ⇒ 只能物理断电）。
    ⇒ 板卡只在「overnight.py 已死」且「该臂未测过」时碰，且**只开一次会话**测完所有臂。
  * 不覆盖任何已存在的 board / gate 结果。
  * 有硬截止：08:45 之后不再启动新的训练臂，只把已有结果写盘后退出。
"""

import json
import os
import re
import subprocess
import sys
import time

RUNS = "/mnt/data/siping/runs/npu20ms"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
REPO = "/tmp/pycharm_project_10ae9e2e"
RESULT = f"{RUNS}/overnight_result.json"
STATE = f"{RUNS}/watchdog_state.json"
LOG = f"{RUNS}/watchdog.log"
T31_F1 = 0.886899
STOP_NEW_TRAIN = time.strftime("%H:%M", time.localtime())  # 运行时按当前时刻判断
HARD_STOP_H = 8  # 08:45 之后不再启动新训练臂

sys.path.insert(0, RUNS)
try:
    import overnight as ov  # 复用 train_silu / screen / build_rknn / board_measure
except Exception as _e:  # pragma: no cover
    ov = None
    print(f"[WD] import overnight 失败: {_e}", flush=True)

PEAK_RE = re.compile(
    r"\[PEAK\] trial_\S+: th=([\d.]+)\s+F1=([\d.]+)\s+P=([\d.]+)%\s+R=([\d.]+)%\s+TP=(\d+)\s+FP=(\d+)"
)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout=7200):
    # 必须显式用 bash：/bin/sh 在本机是 dash，没有 `source` 内建 ⇒ conda 前缀命令会秒退（rc=127）
    return subprocess.run(
        cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=timeout
    )


def alive():
    """overnight.py 是否还活着。

    注意两个坑：
    1) cmdline 可能带 `-u`（`python -u overnight.py`），所以不能只匹配 `python overnight.py`；
       用 `bin/python.*overnight\\.py` 两种都能命中。
    2) 不能匹配到本进程自己：看门狗 cmdline 里是 `overnight_watchdog.py`，
       其中不含字面量 `overnight.py`（`overnight_watchdog.py` 的 "overnight" 后面跟的是 `_watchdog.py`），
       因此 `overnight\\.py`（转义点号）不会误匹配。
    """
    r = subprocess.run(
        ["pgrep", "-f", r"bin/python.*overnight\.py"], capture_output=True, text=True
    )
    pids = [p for p in r.stdout.split() if p.strip()]
    if not pids:
        return False
    # 再按 cmdline 复核一遍，排除 pgrep/bash -c 之类的自匹配
    for p in pids:
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().decode(errors="ignore").replace("\0", " ")
        except OSError:
            continue
        if "overnight.py" in cmd and "watchdog" not in cmd:
            return True
    return False


def hours_to_deadline():
    """距离硬截止（次日 08:45）还有几小时。当前实现取绝对时间戳，不用「小时数」做判断
    （旧版写成 `tm_hour >= 8` ⇒ 下午 17 点也被判定为「已过 8 点」，导致补训全部被跳过）。"""
    import datetime

    now = datetime.datetime.now()
    dl = now.replace(hour=8, minute=45, second=0, microsecond=0)
    if dl <= now:
        dl += datetime.timedelta(days=1)
    return (dl - now).total_seconds() / 3600.0, dl


def may_start_training():
    """离截止不足 1.5 小时就不再启动新的训练臂（已经开跑的让它跑完）。"""
    h, dl = hours_to_deadline()
    return h > 1.5


def collect_arms():
    """从所有 overnight_screen_*.log 汇总臂的 T1 快筛结果（取该臂最好的阈值行）。"""
    arms = {}
    if not os.path.isdir(RUNS):
        return arms
    for fn in os.listdir(RUNS):
        m = re.match(r"overnight_screen_(.+)\.log$", fn)
        if not m:
            continue
        tag = m.group(1)
        try:
            txt = open(f"{RUNS}/{fn}", errors="ignore").read()
        except OSError:
            continue
        best = None
        for mm in PEAK_RE.finditer(txt):
            cand = dict(
                f1=float(mm.group(2)),
                P=float(mm.group(3)),
                R=float(mm.group(4)),
                TP=int(mm.group(5)),
                FP=int(mm.group(6)),
            )
            if best is None or cand["f1"] > best["f1"]:
                best = cand
        if best:
            arms[tag] = dict(tag=tag, **best)
    return arms


def arm_trial(tag):
    if tag.startswith("SILU"):
        return tag.replace("SILU", "")
    if tag.startswith("S2"):
        return tag.replace("S2", "")
    return None


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def save_result(extra):
    cur = load_json(RESULT, {})
    if not isinstance(cur, dict):
        cur = {}
    cur.update(extra)
    with open(RESULT, "w") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)
    return cur


def board_already_done(tag, state):
    """该臂是否已经上板测过（避免重复碰板卡）。"""
    if state.get("board_done") == tag:
        return True
    for p in (RESULT, f"{RUNS}/fast_track_result.json"):
        d = load_json(p, {})
        b = d.get("board")
        if isinstance(b, dict) and b:
            return True
    return False


def run_stage2(arms, state):
    """补齐阶段2 的 8 个种子臂（4 卡并行）。"""
    if ov is None:
        return arms
    trials = [9301, 9302, 9303, 9304, 9305, 9306, 9307, 9308]
    seeds = [7501, 7502, 7503, 7504, 7505, 7506, 7507, 7508]
    gpus = [0, 1, 2, 3]
    rec = state.get("recipe")
    if not rec:
        # 与 overnight.py 同口径：配方由阶段1 最优 silu 臂决定
        best_silu = max(
            (v for k, v in arms.items() if k.startswith("SILU")),
            key=lambda v: v["f1"],
            default=None,
        )
        tr = (best_silu or {}).get("trial") or ""
        rec = {"9221": (1, 3.5), "9222": (0, 3.5), "9223": (1, 1.5)}.get(tr, (1, 3.5))
        log(f"[WD] 补训配方 = uf={rec[0]} lr_scale={rec[1]}（来自阶段1 最优臂 {tr or '默认'}）")

    def one(i):
        t, s, g = trials[i], seeds[i], gpus[i % 4]
        tag = f"S2{t}"
        if tag in arms:
            return
        if not may_start_training():
            h, dl = hours_to_deadline()
            log(f"[WD] 距截止（{dl:%m-%d %H:%M}）仅 {h:.1f} h，放弃补训 {tag}")
            return
        log(f"[WD] 补训 {tag} seed={s} gpu={g} recipe={rec}")
        try:
            ov.train_silu(t, g, s, rec[0], rec[1])
        except Exception as e:
            log(f"[WD] {tag} 训练异常: {e}")
            return
        try:
            r = ov.screen(tag, str(t), g)
        except Exception as e:
            log(f"[WD] {tag} 筛查异常: {e}")
            r = None
        log(f"[WD] {tag} => {r}")
        if r:
            arms[tag] = dict(tag=tag, trial=str(t), **r)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, range(len(trials))))
    return arms


def main():
    state = load_json(STATE, {})
    log("=== WATCHDOG START ===")
    log(f"[WD] overnight.py 当前存活={alive()}")

    # 阶段 A：观察期。只要 overnight.py 活着就只看不动。
    while alive():
        time.sleep(60)
    log("[WD] overnight.py 已退出 ⇒ 进入恢复流程")

    arms = collect_arms()
    log(f"[WD] 汇总到 {len(arms)} 个臂: {sorted(arms)}")
    save_result({"watchdog_arms": arms, "watchdog_state": "recovering"})

    # 阶段 B：补齐阶段2
    need2 = [f"S2{t}" for t in range(9301, 9309)]
    missing = [t for t in need2 if t not in arms]
    if missing:
        log(f"[WD] 阶段2 缺失 {missing} ⇒ 补齐")
        arms = run_stage2(arms, state)
        save_result({"watchdog_arms": arms})

    # 阶段 C：挑最佳 silu 臂
    silu = {k: v for k, v in arms.items() if k.startswith(("SILU", "S2"))}
    best = max(silu, key=lambda k: silu[k]["f1"]) if silu else None
    log(f"[WD] best silu arm = {best} F1={silu[best]['f1'] if best else None}")

    # 阶段 D：上板测速（只一次会话；已测过就跳过）
    board = {}
    if best and silu[best]["f1"] > T31_F1:
        trial = arm_trial(best)
        ck = f"{ROOT}/trials/trial_{trial}/weights/epoch8.pt"
        if board_already_done(best, state):
            log(f"[WD] {best} 已上板测过 ⇒ 跳过（不再碰板卡）")
            board = load_json(RESULT, {}).get("board", {}) or load_json(
                f"{RUNS}/fast_track_result.json", {}
            ).get("board", {})
        elif ov is not None and os.path.isfile(ck):
            try:
                p = ov.build_rknn(f"best_{best}", ck, "silu")
                if p:
                    board = ov.board_measure(f"best_{best}", p)
                    state["board_done"] = best
                    with open(STATE, "w") as f:
                        json.dump(state, f, ensure_ascii=False, indent=2)
            except Exception as e:
                log(f"[WD] 上板测速异常（不影响门禁）: {e}")
        else:
            log(f"[WD] {best} 无 ckpt({ck}) ⇒ 跳过上板")
    save_result({"board": board, "watchdog_arms": arms})

    # 阶段 E：官方三轨门禁
    gate = {}
    if best and silu[best]["f1"] > T31_F1:
        trial = arm_trial(best)
        ck = f"{ROOT}/trials/trial_{trial}/weights/epoch8.pt"
        out = f"{ROOT}/gate/SILUOVN_{trial}_official"
        prev = load_json(f"{out}/gate_report.json", None)
        if prev:
            tr = prev.get("tracks", {}) or {}
            gate = {
                "t1": ((tr.get("track1_uav") or {}).get("finetune_peak") or {}).get("f1"),
                "t2": (tr.get("track2_golden") or {}).get("f1"),
                "t2_fp": (tr.get("track2_golden") or {}).get("fp"),
                "t3": (tr.get("track3_blind") or {}).get("f1"),
                "t3_fp": (tr.get("track3_blind") or {}).get("fp"),
                "verdict": prev.get("verdict"),
                "out": out,
                "reused": True,
            }
            log(f"[WD] 复用已有门禁报告 {out}")
        elif os.path.isfile(ck):
            os.makedirs(out, exist_ok=True)
            log(f"[WD] 跑官方门禁 {out}")
            try:
                sh(
                    f"cd {REPO} && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav && "
                    f"export PYTHONPATH=. && python -u manu/training/auto_epoch_gate_check.py --weights {ck} "
                    f"--out-dir {out} --device 0 --infer-gpu 0 --gpus 1 > {out}/gate.log 2>&1",
                    timeout=5400,
                )
            except Exception as e:
                log(f"[WD] 门禁执行异常: {e}")
            g = load_json(f"{out}/gate_report.json", {})
            tr = g.get("tracks", {}) or {}
            gate = {
                "t1": ((tr.get("track1_uav") or {}).get("finetune_peak") or {}).get("f1"),
                "t2": (tr.get("track2_golden") or {}).get("f1"),
                "t2_fp": (tr.get("track2_golden") or {}).get("fp"),
                "t3": (tr.get("track3_blind") or {}).get("f1"),
                "t3_fp": (tr.get("track3_blind") or {}).get("fp"),
                "verdict": g.get("verdict"),
                "out": out,
            }
    save_result({"gate": gate, "watchdog_arms": arms, "watchdog_state": "done"})
    log("[WD] === WATCHDOG DONE ===")


if __name__ == "__main__":
    main()
