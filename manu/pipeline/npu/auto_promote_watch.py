
# -*- coding: utf-8 -*-
"""自动升轨守望（host 关机期间让「更好的模型」在夜里真正落地）。

动机（2026-11-13 17:45）
------------------------
服务器要自主跑到 2026-11-16 09:00。如果夜里跑出**六项全过 + n≥4 复现 + 板卡 T4 过线**的候选，
却没有人在旁边点「升轨」⇒ 早上的报告只能说「有个候选」，而**基线没换** ⇒ 目标没达成。
⇒ 本守望每 20 min 做一次：
   ① 全盘复算六项（复用 `six_criteria_audit` 的权威口径）；
   ② 按**配方家族**分组（普通臂按 params 配方键；抛光臂按抛光配置键；合成汤按成员集合）；
   ③ 要求某家族里 **六项全过的臂 ≥ 4 个**（种子复现关）且其中至少一个**板卡 T4 PASS**；
   ④ 选该家族 nano 最高的臂 → `promote_candidate.py --trial T --require-board` 执行升轨；
   ⑤ 回填 `nano_search.BASE_V2` / `six_criteria_audit.BASE` 常量 + 同步 watchdog 的 code_md5
      （⚠️ 不回填 ⇒ 调度器仍按旧基线判 Beat-to-Pass ⇒ 守望下一轮会重跑，幂等）；
   ⑥ 写 `promotion_ready.json` + 追加 memory。

🔒 安全阀：
   · **不**用 `--force`（promote_candidate 自己的种子复现关必须过）；
   · 板卡 T4 必须 ≤ 19.989 ms（红线）；
   · 升轨前后都写台账记录（可回滚：记录里带 from/to 全文）；
   · 每轮只升一次（升过就不再动，除非出现更严格的候选）。

用法（服务器）：
    python -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 1200
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

from manu.pipeline.npu import six_criteria_audit as A  # noqa: E402
from manu.pipeline.npu import nano_search as ns  # noqa: PLC0415
from manu.pipeline.npu import family_key as FK  # noqa: PLC0415  （家族键唯一权威口径）

PY = "/home/huangzhe/anaconda3/envs/uav/bin/python"
RUNS = "/mnt/data/siping/runs/npu20ms/nano_search"
LEDGER = f"{RUNS}/nano_search_ledger.json"
MEASURED = "/mnt/data/siping/runs/npu20ms/board_measure_auto/measured.json"
MEM = os.path.join(REPO, "manu", "memory", "memory.md")
LIMIT_MS = 19.989


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def family_key(row: dict, led: dict) -> str:
    """配方家族键（**唯一权威口径** ⇒ 委托给 `family_key` 模块）。

    🟢🟢 2026-11-16 11:05：原先本文件内嵌的实现有两个口径 bug，已移到
    `manu/pipeline/npu/family_key.py` 统一修掉：
      ① 数值未归一化（`thr=12` vs `thr=12.0` ⇒ 同一配方被劈成两个家族 ⇒ n≥4 永远凑不齐）；
      ② 合成汤家族键没带成员数（`bs12679:eq` 把 k=2/3/4/5/7 混成一家 ⇒ 复现关被悄悄放宽）。
    判据本身不变：同家族 n≥4 六项全过 + 板卡 T4 ≤ 19.989 ms。
    """
    return FK.arm_family_key((led.get("arms") or {}).get(str(row.get("trial"))) or {})


def board_pass_map() -> dict:
    try:
        d = json.load(open(MEASURED))
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for k, v in (d or {}).items():
        ms = v.get("ms")
        if ms is None:
            continue
        m = re.match(r"^(\d+)_ep(\d+)$", str(k))
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = (float(ms), bool(float(ms) <= LIMIT_MS))
    return out


def _baseline_metrics(led: dict, trial: int) -> dict:
    """新基线的**完整**读数（picked/official + 台账 baseline_v2 兜底）。"""
    st = (led.get("arms") or {}).get(str(trial)) or {}
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    b = led.get("baseline_v2") or {}
    m: dict = {}
    for k in ("nano", "micro", "small", "regular", "f1", "nano4"):
        v = pk.get(k)
        if v is None:
            v = b.get(k)
        m[k] = v
    for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp"):
        v = of.get(k)
        if v is None:
            v = b.get(k)
        m[k] = v
    for k in ("ms_delivery_board", "ms_new_board", "ms_new_board_t31_ref", "name", "recipe"):
        m[k] = b.get(k)
    return m


# 🟢🟢🟢 2026-11-16 11:45 补**缺失的读写助手**：回填函数用了 rd()/wr()/STAMP，
#   但模块里**根本没定义** ⇒ 升轨那一脚抛 `name 'rd' is not defined`
#   ⇒ 常量（BASE_V2 / six_criteria_audit.BASE）**从来没被回填过**。
STAMP = time.strftime("%Y%m%d_%H%M%S")


def rd(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def wr(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def backfill_constants(trial: int, epoch: int) -> None:
    """把新基线写回两个常量文件 + 同步 watchdog 的 code_md5。

    🟢🟢 2026-11-16 09:55：**指标也要回填**（原来只改 trial/epoch ⇒ 常量里的
      nano/f1/T1/T3 还是旧基线的 ⇒ 升轨后判据口径没抬起来）。
    """
    ns_p = os.path.join(REPO, "manu", "pipeline", "npu", "nano_search.py")
    aud_p = os.path.join(REPO, "manu", "pipeline", "npu", "six_criteria_audit.py")
    try:
        led = json.load(open(LEDGER, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        led = {}
    m = _baseline_metrics(led, trial)

    def g(k, default):
        v = m.get(k)
        return default if v is None else v

    # ── ① six_criteria_audit.BASE ──
    if os.path.isfile(aud_p) and m.get("nano") is not None:
        src = rd(aud_p)
        old_base = re.search(r"^BASE = \{.*?^\}", src, re.S | re.M)
        if old_base:
            new_base = ("BASE = {\n"
                        f"    \"nano\": {float(g('nano', 0.0))!r},\n"
                        f"    \"nano4\": {float(g('nano4', 0.0))!r},\n"
                        f"    \"f1\": {float(g('f1', 0.0))!r},\n"
                        f"    \"t1\": {float(g('t1', 0.0))!r},\n"
                        f"    \"t3\": {float(g('t3', 0.0))!r},\n"
                        f"    \"t2_fp\": {int(g('t2_fp', 5))},\n"
                        "}\n")
            shutil.copy(aud_p, aud_p + ".bak_" + STAMP)
            wr(aud_p, src[:old_base.start()] + new_base + src[old_base.end():])
            log(f"[回填] six_criteria_audit.BASE ⇒ trial={trial} 指标已同步")

    # ── ② nano_search.BASE_V2 ──
    if os.path.isfile(ns_p) and m.get("nano") is not None:
        src = rd(ns_p)
        ck = f"{ns.ROOT}/trials/trial_{trial}/weights/epoch{epoch}.pt"
        name = m.get("name") or "Baseline_v2.1"
        recipe = m.get("recipe") or f"trial_{trial} ep{epoch}"
        block = ("BASE_V2_CKPT = " + repr(ck) + "\n"
                 "BASE_V2 = dict(\n"
                 f"    name={name!r}, trial={trial}, epoch={epoch}, ckpt=BASE_V2_CKPT,\n"
                 f"    nano={float(g('nano', 0.0))!r}, micro={g('micro', None)!r}, small={g('small', None)!r}, "
                 f"regular={g('regular', None)!r}, f1={float(g('f1', 0.0))!r}, nano4={float(g('nano4', 0.0))!r},\n"
                 f"    t1={float(g('t1', 0.0))!r}, t1_fp={g('t1_fp', None)!r}, t2={g('t2', None)!r}, "
                 f"t2_fp={int(g('t2_fp', 5))}, t3={float(g('t3', 0.0))!r}, t3_fp={g('t3_fp', None)!r},\n"
                 f"    ms_delivery_board={g('ms_delivery_board', None)!r}, ms_new_board={g('ms_new_board', None)!r}, "
                 f"ms_new_board_t31_ref={g('ms_new_board_t31_ref', None)!r},\n"
                 f"    recipe={recipe!r},\n"
                 ")\n")
        m1 = re.search(r"^BASE_V2_CKPT = .*?\nBASE_V2 = dict\(.*?^\)\n", src, re.S | re.M)
        if not m1:
            raise SystemExit("[FAIL] nano_search.BASE_V2 块没找到")
        shutil.copy(ns_p, ns_p + ".bak_" + STAMP)
        wr(ns_p, src[:m1.start()] + block + src[m1.end():])
        log(f"[回填] nano_search.BASE_V2 ⇒ trial={trial} epoch={epoch} 指标已同步")
    # watchdog 的 code_md5 必须同步，否则它会判「代码已更新」并再重启一次调度器
    try:
        cur = subprocess.run(
            "cd %s && md5sum manu/pipeline/npu/adaptive_runner.py manu/pipeline/npu/nano_search.py | md5sum | cut -d' ' -f1" % REPO,
            shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
        if cur:
            sp = f"{RUNS}/watchdog_state.json"
            st = {}
            if os.path.isfile(sp):
                try:
                    st = json.load(open(sp))
                except Exception:  # noqa: BLE001
                    st = {}
            st["code_md5"] = cur
            json.dump(st, open(sp, "w"))
            log(f"[回填] watchdog code_md5 = {cur}")
    except Exception as exc:  # noqa: BLE001
        log(f"[回填] watchdog md5 失败：{exc}")


def restart_scheduler(deadline: str = "2026-11-17 09:00") -> None:
    """升轨后重启调度器（让它按**新基线**判 Beat-to-Pass）。

    🟢🟢 2026-11-16 10:10：升轨后**已在跑的调度器**内存里还是旧 BASE_V2 ⇒
      判 Beat-to-Pass 仍按旧基线 ⇒ 新基线抬起来的门槛对老进程无效（算力浪费在
      「不如新基线」的臂上）。⇒ 升轨成功后**只重启调度器自己的进程组**：
      不用 watchdog 那个全局 `pkill optuna_mix_ratio_search.py`
      （那会误杀 polish / soup 等并行通道的训练进程）。
    """
    try:
        out = subprocess.run(["pgrep", "-f", "adaptive_runner.py --deadline"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        pids = [int(x) for x in out if str(x).strip().isdigit()]
        if not pids:
            log("[重启] 没找到调度器进程 ⇒ 跳过")
            return
        for pid in pids:
            pgid = subprocess.run(["ps", "-o", "pgid=", "-p", str(pid)],
                                  capture_output=True, text=True, timeout=30).stdout.strip()
            if pgid and pgid.isdigit():
                subprocess.run(["kill", "-9", "-%s" % pgid], timeout=30)
        time.sleep(5)
        nxt = 1
        try:
            st = json.load(open(f"{RUNS}/round_state.json"))
            nxt = max(nxt, int(st.get("last_started_round") or 1))
        except Exception:  # noqa: BLE001
            pass
        try:
            led = json.load(open(LEDGER))
            comp = max([int(r.get("round", 0)) for r in (led.get("rounds") or [])]
                       or [0])
            nxt = max(nxt, comp + 1)
        except Exception:  # noqa: BLE001
            pass
        # 🔴🔴 2026-11-17 16:55：**不许自动重启训练调度器**。
        #   `run_adaptive_night.sh` 会拉起 `optuna_mix_ratio_search.py --worker`
        #   （每个再 fork 10 个 2.2GB 子进程 ⇒ **单条训练臂 ≈ 22GB**）
        #   ⇒ 昨夜 15:00 的**服务器内存爆炸**就是这么来的（MemAvailable 23GB→79GB）。
        #   ⇒ 升轨成功后只回填常量，**不再拉起任何训练进程**；
        #      确实需要重训时由人显式跑（`PROMOTE_RESTART_SCHEDULER=1` 才允许）。
        if os.environ.get("PROMOTE_RESTART_SCHEDULER") == "1":
            with open(os.devnull, "wb") as devnull:
                subprocess.Popen(["bash", "manu/pipeline/npu/run_adaptive_night.sh",
                                  deadline, "6", "18", str(nxt)],
                                 cwd=REPO, stdout=devnull, stderr=devnull,
                                 stdin=devnull, start_new_session=True)
            log(f"[重启] 调度器已重启（start-round={nxt}）⇒ 按新基线判 Beat-to-Pass")
        else:
            log(f"[重启] **跳过**（不自动拉起训练调度器 ⇒ 防 optuna worker 内存爆炸）；"
                f"start-round={nxt} 需人工确认后再跑")
    except Exception as exc:  # noqa: BLE001
        log(f"[重启] 失败（不影响升轨结果）：{exc}")


def append_memory(text: str) -> None:
    try:
        with open(MEM, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception as exc:  # noqa: BLE001
        log(f"[memory] 写入失败：{exc}")


def run_once() -> None:
    led = ns.load_ledger()
    base = led.get("baseline_v2") or {}
    base_trial = int(base.get("trial") or 9863)
    rows = A.collect()
    board = board_pass_map()
    fam = collections.defaultdict(list)
    for r in rows:
        n, _miss = A.judge(r)
        if n != 6:
            continue
        key = family_key(r, led)
        fam[key].append(r)
    log(f"[扫描] 六项全过臂 {sum(len(v) for v in fam.values())} 个 / {len(fam)} 个家族｜基线 = {base_trial}")

    # 🟢🟢🟢 2026-11-17 17:20 —— **种子复现关的家族分组**（升轨链路的隐藏坑）
    #   `promote_candidate` 的种子关用的是**候选臂自己的 `soup_family` 字符串**（同 parent 同权重标签，
    #   如 `bs640001:w56`）⇒ 候选臂所在的权重标签里**必须还有第二个六项全过臂**。
    #   ⇒ 若守望只按 nano 取最优臂，一旦那个臂的权重标签是「孤儿」（只有它自己六项全过），
    #      `promote_candidate` 必定拒绝 ⇒ **整晚的票白费**（旧逻辑还不会换下一个臂试）。
    fam_six: dict[str, list[int]] = collections.defaultdict(list)
    for r in rows:
        if A.judge(r)[0] != 6:
            continue
        _st = (led.get("arms") or {}).get(str(r.get("trial"))) or {}
        _sf = _st.get("soup_family")
        if _sf:
            fam_six[_sf].append(int(r.get("trial")))

    def seed_ok(r: dict) -> bool:
        """该臂所在权重标签里是否已有 ≥2 个六项全过臂（= 种子复现关可通过）。"""
        _st = (led.get("arms") or {}).get(str(r.get("trial"))) or {}
        _sf = _st.get("soup_family")
        return bool(_sf) and len(fam_six.get(_sf, [])) >= 2

    # 找满足「同家族 ≥4 个六项全过」且「其中有板卡 T4 PASS」的家族
    best = None
    cands: list = []
    for key, members in fam.items():
        if len(members) < 4:
            continue
        with_board = []
        for r in members:
            # 🟢🟢🟢 2026-11-16 11:35 修**根因**：audit 行的 epoch 键名是 **"ep"**（不是 "epoch"）
            #   ⇒ 原来 `r.get("epoch")` 恒为 None ⇒ 板卡 PASS 永远查不到 ⇒
            #     任何家族都被判「无板卡候选」⇒ **升轨永远不触发**（今天 11:17 就是这么卡住的）。
            ep = r.get("ep") if r.get("ep") is not None else r.get("epoch")
            tr = r.get("trial")
            b = board.get((int(tr), int(float(ep)))) if ep is not None else None
            if b and b[1]:
                with_board.append((r, b[0]))
        if not with_board:
            continue
        members.sort(key=lambda r: (not seed_ok(r), -(r.get("nano") or 0), -(r.get("nano4") or 0), -(r.get("f1") or 0)))
        # 🟢🟢 2026-11-16 11:05：升轨候选**优先取「自己已有板卡 PASS」的臂**。
        #   原因：promote_candidate --require-board 要求**候选臂自身**带板卡读数；
        #   若只按 nano 排序取最优臂，它可能还没上板 ⇒ 升轨被拒 + 白白空转一个周期。
        with_board.sort(key=lambda rm: (not seed_ok(rm[0]), -(rm[0].get("nano") or 0), -(rm[0].get("nano4") or 0), -(rm[0].get("f1") or 0)))
        cand = with_board[0][0]
        log(f"[候选] 家族 {key}：六项全过 {len(members)} 个｜板卡 PASS {len(with_board)} 个｜"
            f"最优 trial={cand.get('trial')} ep={cand.get('epoch')} nano={cand.get('nano'):.6f} "
            f"种子关={'✔' if seed_ok(cand) else '✘'}")
        if best is None or (cand.get("nano") or 0) > (best[0].get("nano") or 0):
            best = (cand, key, len(members), with_board)
        # 🟢🟢 2026-11-17 17:20：**把所有「板卡 PASS 的家族成员」都放进候选队列**，
        #   排序 = 种子关先过 → nano 高 ⇒ 预检失败时能自动换下一个臂试（不再整夜空转）
        for r, ms in with_board:
            cands.append((r, key, len(members), ms))
    if best is None:
        log("[结论] 无满足「n≥4 + 板卡 T4」的新候选 ⇒ 不动基线")
        return
    cands.sort(key=lambda c: (not seed_ok(c[0]), -(c[0].get("nano") or 0)))
    # 🔴🔴 2026-11-18 09:10 修**基线名通胀**（昨夜真实事故）
    #   现象：守望在 6 小时内把**同一个 trial_1117002** 反复升轨 82 次
    #         ⇒ 基线名从 Baseline_v2.4 一路涨到 **Baseline_v2.86** ⇒ registry 89 条垃圾。
    #   根因：候选筛选只要求「六项 ≥ 基线」⇒ **打平**（tie）也被当成新候选 ⇒ 自我升轨循环。
    #   修法：候选必须 (a) 不是当前基线本体，(b) 在六项里**至少严格优于一项**
    #        （nano / nano4 / f1 / t1 / t3 任一严格大于当前基线）⇒ 打平不再升轨。
    _bm = {
        "nano": float(base.get("nano") or 0), "nano4": float(base.get("nano4") or 0),
        "f1": float(base.get("f1") or 0), "t1": float(base.get("t1") or 0),
        "t3": float(base.get("t3") or 0),
    }

    def _strictly_better(r: dict) -> bool:
        if int(r.get("trial")) == base_trial:
            return False
        return (float(r.get("nano") or 0) > _bm["nano"]
                or float(r.get("nano4") or 0) > _bm["nano4"]
                or float(r.get("f1") or 0) > _bm["f1"]
                or float(r.get("t1") or 0) > _bm["t1"]
                or float(r.get("t3") or 0) > _bm["t3"])

    _n0 = len(cands)
    cands = [c for c in cands if _strictly_better(c[0])]
    log(f"[过滤] 剔除「基线本体/纯打平」候选 {_n0} ⇒ {len(cands)}（防基线名通胀）")
    # 2026-11-17 18:10: 基线名必须动态递增(原来硬编码 Baseline_v2.1
    #   => 与历史名 Baseline_v2.1(trial_69913) 撞车, 且从 v2.2 升上去却叫 v2.1)
    try:
        _cur = str((ns.load_ledger().get("baseline_v2") or {}).get("name") or "Baseline_v2.0")
    except Exception:
        _cur = "Baseline_v2.0"
    _mm = re.match(r"Baseline_v2\.(\d+)", _cur)
    name = "Baseline_v2.%d" % (int(_mm.group(1)) + 1 if _mm else 1)
    log(f"[命名] 当前基线 {_cur} => 新基线名 {name}")
    # 🟢🟢 2026-11-17 17:20：**逐个候选试**（种子关/板卡关任一不过就换下一个臂）
    cand = key = nmem = with_board = None
    for _c, _k, _n, _ms in cands[:8]:
        trial = int(_c.get("trial"))
        _ep = _c.get("ep") if _c.get("ep") is not None else _c.get("epoch")
        epoch = int(float(_ep)) if _ep is not None else 0
        log(f"[升轨] 试候选 trial={trial} ep={epoch}（家族 {_k}，n={_n}，板卡 {_ms} ms，"
            f"种子关={'✔' if seed_ok(_c) else '✘'}）")
        pre = subprocess.run([PY, "manu/pipeline/npu/promote_candidate.py", "--trial", str(trial),
                              "--name", name, "--require-board", "--dry-run"],
                             capture_output=True, text=True, timeout=1800)
        log(f"[预检] rc={pre.returncode}\n{(pre.stdout or '')[-1500:]}\n{(pre.stderr or '')[-500:]}")
        if pre.returncode != 0:
            log(f"[结论] 预检未过（trial={trial}）⇒ 换下一个候选")
            continue
        real = subprocess.run([PY, "manu/pipeline/npu/promote_candidate.py", "--trial", str(trial),
                               "--name", name, "--require-board"],
                              capture_output=True, text=True, timeout=1800)
        log(f"[升轨] rc={real.returncode}\n{(real.stdout or '')[-2000:]}\n{(real.stderr or '')[-500:]}")
        if real.returncode != 0:
            log(f"[结论] 升轨命令失败（trial={trial}）⇒ 换下一个候选")
            continue
        cand, key, nmem, with_board = _c, _k, _n, [(_c, _ms)]
        break
    if cand is None:
        log("[结论] 所有候选预检/升轨均未过 ⇒ 不动基线")
        return
    trial = int(cand.get("trial"))
    _ep = cand.get("ep") if cand.get("ep") is not None else cand.get("epoch")
    epoch = int(float(_ep)) if _ep is not None else 0
    # 🟢 2026-11-16 11:20：回填/重启**不许把守望进程带死**
    #   （SystemExit/异常会终止整个 watcher ⇒ 后面再来的候选就没人升轨了）
    try:
        backfill_constants(trial, epoch)
    except Exception as exc:  # noqa: BLE001
        log(f"[回填] 失败（不影响已升轨的台账）：{exc}")
    try:
        restart_scheduler()
    except Exception as exc:  # noqa: BLE001
        log(f"[重启] 失败：{exc}")
    rec = {"ts": time.strftime("%F %T"), "name": name, "trial": trial, "epoch": epoch,
           "family": key, "n_six_pass": nmem, "board_ms": with_board[0][1] if with_board else None,
           "nano": cand.get("nano"), "nano4": cand.get("nano4"), "f1": cand.get("f1"),
           "prev_base_trial": base_trial}
    with open(f"{RUNS}/promotion_ready.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    append_memory(u"\n### 🟢🟢 自动升轨记录（%s）\n* **%s = trial_%d@%d**（家族 %s，同配方六项全过 n=%d，板卡 T4 %.3f ms）\n"
                 u"* 指标：nano %.6f / nano4 %.6f / F1 %.6f（前基线 trial_%d）\n"
                 u"* 判据：六项全过（nano/nano4/F1/T1/T2_fp/T3 对前基线）+ 板卡 T4 ≤ %.3f ms\n"
                 % (rec["ts"], name, trial, epoch, key, nmem, rec["board_ms"] or 0.0,
                    cand.get("nano") or 0, cand.get("nano4") or 0, cand.get("f1") or 0, base_trial, LIMIT_MS))
    log("[完成] 升轨完成")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=1200)
    ap.add_argument("--deadline", default="2026-11-16 09:00")
    args = ap.parse_args()
    dl = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))
    while True:
        try:
            run_once()
        except Exception as exc:  # noqa: BLE001
            log(f"[异常] {exc}")
        if not args.watch:
            break
        if time.time() > dl:
            log("[截止] 到点退出")
            break
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    main()
