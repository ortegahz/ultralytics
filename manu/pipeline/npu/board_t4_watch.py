
"""板卡 T4 自动测速守望（host 关机期间让「六项全过」候选**自动**补上板端时延）。

🟢 2026-11-13 14:50 新增（配合 host 断电）：
   调度器只跑**离线**六项（nano / nano4 / F1 / T1 / T2虚警 / T3），
   但**交付**还要过板卡 T4（RKNN 推理时延 ≤ trial_0031×1.02 = 19.989 ms）。
   ⇒ 夜间若出了新的六项全过候选，没人测它的板端时延 ⇒ 明早拿不到完整结论。
   ⇒ 本守望进程每 20 min 扫一次磁盘门禁报告（six_criteria_audit 口径），
     对**尚未测过**的六项全过候选：导出 RKNN → 上板 → 同会话 A/B 测速 → 落盘。

🔒 纪律（与 board_beat_measure.py 一致）：
   · 板卡**只读预检**（load / 是否有别人的 rknn 进程 / 根分区余量）⇒ 忙就**跳过本轮**，绝不抢。
   · **绝不 kill 板卡上任何进程**（板卡是共享资源）。
   · 测速协议同 session3_old64：CONTROL_NAME=base_int8.rknn LOOPS=30 WARMUP=10 SETTLE=5。

用法（服务器）：
    python -u manu/pipeline/npu/board_t4_watch.py --once      # 跑一轮
    python -u manu/pipeline/npu/board_t4_watch.py --watch    # 常驻（默认 20 min/轮）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
if not os.path.isdir(REPO):
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

from manu.pipeline.npu import six_criteria_audit as A  # noqa: E402
from manu.pipeline.npu import route2_acceptance as RA  # noqa: E402
from manu.pipeline.npu import nano_search as ns  # noqa: E402
import board_speed_search as B  # noqa: E402

OUT = "/mnt/data/siping/runs/npu20ms/board_measure_auto"
DONE = os.path.join(OUT, "measured.json")
KEY = "-i /home/huangzhe/.ssh/id_board -o StrictHostKeyChecking=no -o ConnectTimeout=10"
LIMIT_MS = 19.989          # trial_0031 19.597 ms × 1.02（交付时延红线）
# 🟢 2026-11-17 17:30：板卡是**共享资源** ⇒ 临界带（红线 ±0.6 ms）内的读数要复测取中位数
BORDER_LO = LIMIT_MS - 0.6
BORDER_HI = LIMIT_MS + 0.6
# 🟢🟢 2026-11-19 12:40 —— **T4 判定改「中位数 + 样本池」**
#   现场事实（2026-11-19 实测）：同一臂跨会话时延抖动可达 **±2.9 ms**
#   （例：1740004 三次读数 19.719 / 19.990 / 21.396；1612001 19.735 → 21.720），
#   而红线 19.989 的余量只有 0.2~0.4 ms ⇒ **单次读数判定等于抛硬币**。
#   ⇒ 改为：每次测速会把板上**所有臂**重测一遍 ⇒ 免费得到全体臂的新样本；
#     每个臂攒够 MIN_SAMPLES 个样本后，用**中位数**判定，并保留全部样本供审计。
MIN_SAMPLES = 5
MAX_SAMPLES = 9
SAMPLES = os.path.join(OUT, "t4_samples.json")
_ARM_RE = re.compile(r"^arm_n(\d+)_ep(\d+)_int8$")


def load_pools() -> dict:
    """读回「臂 -> 全部历史样本」的池子（键形如 `1674017_ep5`）。"""
    pools: dict = {}
    if os.path.isfile(SAMPLES):
        try:
            raw = json.load(open(SAMPLES, encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            raw = {}
        if raw:
            for k, v in raw.items():
                xs = [float(x) for x in (v or [])]
                if xs:
                    pools[k] = xs
            return pools
    for src in (DONE,):
        if not os.path.isfile(src):
            continue
        try:
            d = json.load(open(src, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if src == DONE:
            for k, v in (d or {}).items():
                xs = [float(x) for x in ((v or {}).get("samples") or [])]
                if not xs and (v or {}).get("ms") is not None:
                    xs = [float(v["ms"])]
                if xs:
                    pools.setdefault(k, []).extend(xs)
        else:
            for k, v in (d or {}).items():
                xs = [float(x) for x in (v or [])]
                if xs:
                    pools.setdefault(k, []).extend(xs)
    return pools


def save_pools(pools: dict) -> None:
    tmp = SAMPLES + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({k: [round(float(x), 4) for x in v] for k, v in pools.items()}, fh,
                  indent=1, ensure_ascii=False)
    os.replace(tmp, SAMPLES)


def pool_median(xs: list) -> float:
    s = sorted(float(x) for x in xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def merge_session(pools: dict, all_ms: dict) -> list:
    """把一次会话里**所有臂**的读数并进各自样本池 ⇒ 返回被更新的臂键列表。"""
    touched = []
    for ak, av in (all_ms or {}).items():
        m = _ARM_RE.match(str(ak).strip())
        if not m:
            continue
        key = "%s_ep%s" % (m.group(1), m.group(2))
        pools.setdefault(key, []).append(float(av))
        touched.append(key)
    return touched

# 🟢 2026-11-17 11:45：任务书交付红线（memory §136.6）—— 与「内部严格六项」是两套口径。
#   交付判据：F1 > 0.887122（0031）· nano ≥ 0.6666 · 官方 T1 ≥ 0.885997 · T2 FP ≤ 20（官方 36）
#              · T3 ≥ 0.948330 · int8 ≤ 20 ms
#   ⇒ 凡是过这条线的臂都是**可交付件**；若板卡时延也过 ⇒ 就是比 trial_0031 更好的交付件。
#   ⚠️ gate_report 的 verdict 用的是冻结基线 0.905736（连当前基线都过不了）⇒ 不能用它判交付。
DELIVER = {"f1": 0.887122, "nano": 0.6666, "t1": 0.885997, "t2_fp": 20, "t3": 0.948330,
           "ms": 20.0}


def deliver_ok(r: dict) -> bool:
    """该臂是否过**任务书交付红线**（不含板卡时延）。"""
    try:
        return bool(r.get("f1") is not None and r.get("nano") is not None
                    and r.get("t1") is not None and r.get("t2_fp") is not None
                    and r.get("t3") is not None
                    and float(r["f1"]) > DELIVER["f1"] and float(r["nano"]) >= DELIVER["nano"]
                    and float(r["t1"]) >= DELIVER["t1"] and int(r["t2_fp"]) <= DELIVER["t2_fp"]
                    and float(r["t3"]) >= DELIVER["t3"])
    except Exception:  # noqa: BLE001
        return False
CKPT_DIR = "/mnt/data/siping/runs/npu20ms_offcal/trials"


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def board_ready() -> bool:
    """板卡只读预检：忙 / 空间不足 / 连不上 ⇒ False（本轮跳过，绝不抢别人的板）。

    🟢🟢 2026-11-18 18:00 —— **修「板卡永远被判忙」的致命自匹配**（今晚升轨链最后一环）：
      原实现：`ps -eo cmd | grep -E '[r]knn|[e]2e_|[m]easure_hm_speed' | wc -l`
      方括号技巧只能让 **grep 自己**不匹配自己，但**探针的父 shell**（ssh 远端起的
      `bash -c "ps ... grep ..."`）的命令行里就含这些字面量 ⇒ ps 会把它列出来、grep 会命中它
      ⇒ 计数恒 ≥ 1 ⇒ **板卡永远判忙** ⇒ 17:37 / 17:47 两轮预检在板卡**完全空闲**
      （load=0.01、NPU 0%、无任何用户进程）时仍然跳过 ⇒ 所有候选的板卡 T4 读数全缺
      ⇒ 升轨链最后一环永远打不通（与算力无关的**纯逻辑阻塞**）。
      修法：① 改用 `ps -eo args | awk ...` 并显式排除 **awk / grep / ps -eo**（整条探针链）；
            ② 增加 **NPU 负载** 判据（`/sys/kernel/debug/rknpu/load`）—— 抢板卡的
               别人任务一定占 NPU，CPU load 只是辅助信号。
    """
    try:
        pre = subprocess.run(
            ["bash", "-c",
             f"ssh {KEY} {B.BOARD} \"awk '{{print \\$1}}' /proc/loadavg; "
             "ps -eo args | awk '/rknn_mosaic|e2e_v|measure_hm_speed/ && !/awk/ && !/grep/ "
             "&& !/ps -eo/ {c++} END {print c+0}'; "
             "df -m / | tail -1 | awk '{print \\$4}'; "
             "(cat /sys/kernel/debug/rknpu/load 2>/dev/null | grep -o '[0-9]\\+%' | tr -d '%' "
             "| sort -n | tail -1) || echo 0\""],
            capture_output=True, text=True, timeout=90)
    except Exception as exc:  # noqa: BLE001
        log(f"[precheck] 异常（忽略）：{exc}")
        return False
    lines = [x.strip() for x in (pre.stdout or "").splitlines() if x.strip()]
    if len(lines) < 3:
        log(f"[precheck] 板卡无响应（输出 {len(lines)} 行）⇒ 本轮跳过")
        return False
    try:
        load = float(lines[0])
        nproc = int(lines[1])
        disk_free_mb = int(lines[2])
        npu_pct = int(lines[3]) if len(lines) > 3 and lines[3].isdigit() else 0
    except ValueError:
        log(f"[precheck] 输出异常 ⇒ 本轮跳过：{lines[:4]}")
        return False
    if load > 1.0 or nproc > 0 or disk_free_mb < 400 or npu_pct > 10:
        log(f"[precheck] 板卡忙（load={load} 别人的进程={nproc} 根分区余量={disk_free_mb}MB "
            f"NPU={npu_pct}%）⇒ 本轮跳过")
        return False
    log(f"[precheck] 板卡空闲（load={load} 别人的进程={nproc} 根分区余量={disk_free_mb}MB "
        f"NPU={npu_pct}%）⇒ 本轮可测")
    return True


def _norm_arm(key: str) -> str:
    """板卡脚本会把臂名写成 `arm_<name>_int8` ⇒ 归一化后再比对。"""
    k = key.strip()
    if k.startswith("arm_"):
        k = k[4:]
    if k.endswith("_int8"):
        k = k[:-5]
    return k


def _pick_ms(res: dict, trial: int, ep: str) -> float | None:
    """从 @@SUMMARY@@ 结果里挑出本候选的 rknn_run_avg_ms。"""
    want = _norm_arm(f"n{trial}_ep{ep}")
    for k, v in res.items():
        if _norm_arm(k) == want:
            return float(v)
    cand = [float(v) for k, v in res.items()
            if _norm_arm(k).startswith(f"n{trial}_ep")]
    return cand[0] if len(cand) == 1 else None


def measure(trial: int, ep: str) -> dict | None:
    """导出 RKNN → 上板 → 同会话 A/B 测速 ⇒ 返回结果 dict（失败返回 None）。"""
    ck_name = ns.epoch_ckpt_name(ep) if hasattr(ns, "epoch_ckpt_name") else f"epoch{ep}.pt"
    ck = os.path.join(CKPT_DIR, f"trial_{trial}", "weights", ck_name)
    if not os.path.isfile(ck):
        log(f"[T4] 权重不存在，跳过：{ck}")
        return None
    name = f"n{trial}_ep{ep}".replace(".", "_")
    try:
        rknn, stride = RA.export_rknn(ck, name)
    except Exception as exc:  # noqa: BLE001
        log(f"[T4] 导出失败 {trial}@{ep}：{exc}")
        return None
    if not rknn:
        log(f"[T4] 导出返回空 {trial}@{ep}")
        return None
    # 🔴 2026-11-18 17:50：远端命令加 `timeout` + **先落文件再 cat**
    #   原因：远端脚本若留下任何后台子进程（持有 ssh 的 stdout 管道）⇒
    #   `subprocess.run(capture_output=True)` 会**永远等 EOF** ⇒ 看板静默挂死（实测 20+ 分钟无输出）。
    #   ⇒ 远端 stdout 重定向到文件，再 cat 出来；管道只由 cat 持有 ⇒ 一定能收尾。
    subprocess.run(["bash", "-c",
                    f"timeout 60 ssh {KEY} {B.BOARD} 'mkdir -p {B.BOARD_DIR}/arms'"],
                   capture_output=True, text=True, timeout=90)
    up = subprocess.run(
        ["bash", "-c",
         f"timeout 300 scp {KEY} {rknn} {B.BOARD}:{B.BOARD_DIR}/arms/ && "
         f"timeout 120 ssh {KEY} {B.BOARD} 'printf {stride} > {B.BOARD_DIR}/arms/{name}_int8.stride'"],
        capture_output=True, text=True, timeout=600)
    if up.returncode != 0:
        log(f"[T4] 上传失败 {trial}@{ep}：{(up.stderr or '')[-300:]}")
        return None
    r = subprocess.run(
        ["bash", "-c",
         f"timeout 900 ssh {KEY} {B.BOARD} 'cd {B.BOARD_DIR} && BOARD_DIR={B.BOARD_DIR} "
         "CONTROL_NAME=base_int8.rknn LOOPS=30 WARMUP=10 SETTLE=5 "
         "timeout 780 bash measure_hm_speed_board.sh > /tmp/t4_out.txt 2>&1; cat /tmp/t4_out.txt'"],
        capture_output=True, text=True, timeout=1200)
    txt = (r.stdout or "") + (r.stderr or "")
    # 🟢 2026-11-19：同臂复测会**覆盖**同名会话文件 ⇒ 历史样本丢失（这正是
    #   1702011 只有 3 个样本、1740004 判成 FAIL 的根因之一）⇒ 改为带时间戳落盘。
    _stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT, f"session_{trial}_ep{ep}_{_stamp}.txt"), "w") as fh:
        fh.write(txt)
    res: dict[str, float] = {}
    for m in re.finditer(r"@@SUMMARY@@ (\S+)\|rknn_run_avg_ms\|([\d.]+)", txt):
        res[m.group(1)] = float(m.group(2))
    ms = _pick_ms(res, trial, ep)
    out = {"trial": trial, "ep": ep, "ckpt": ck, "ms": ms, "limit_ms": LIMIT_MS,
           "pass_t4": bool(ms is not None and ms <= LIMIT_MS),
           "all_ms": res, "ts": time.strftime("%F %T")}
    return out


def prune_remote_arms(keep: int = 25) -> None:
    """🔴🔴 2026-11-19 11:20：**板卡根分区只剩 1.0GB**，而每次测速要上传 2.96MB 的 rknn。

    现场事实：`/root/hm_speed/arms` 已 198MB / 150 个文件，板卡 `df -h /` = **94%（余 1021MB）**。
    测速节奏 ≈ 每 8 分钟一个臂 ⇒ **22MB/小时 ⇒ 约 46 小时就把板卡写满**
    ⇒ 后续上传/测速全部失败 ⇒ **T4 读数断链 ⇒ 升轨链最后一环断**（deadline 是 11-23 09:00）。

    修法：每次测速后，只保留板卡上**最新的 keep 个** rknn（连同对应 .stride），其余删掉。
    保留 25 个 ≈ 75MB ⇒ 稳态占用约 75MB，1GB 余量足够撑到 deadline。
    """
    cmd = (
        f"timeout 120 ssh {KEY} {B.BOARD} 'cd {B.BOARD_DIR}/arms 2>/dev/null || exit 0; "
        f"ls -1t *.rknn 2>/dev/null | tail -n +{keep + 1} | "
        f"while read f; do rm -f \"$f\" \"${{f%.rknn}}.stride\"; done; "
        f"ls -1 *.rknn 2>/dev/null | wc -l'"
    )
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=180)
        n = ((r.stdout or "").strip().splitlines() or ["?"])[-1].strip()
        log(f"[清理] 板卡 arms 只保留最新 {keep} 个 rknn（现存 {n}）")
    except Exception as exc:  # noqa: BLE001
        log(f"[清理] 板卡 arms 清理失败（不致命）：{exc}")


def write_back_ledger(trial: int, ep: str, ms: float) -> None:
    """把板端 T4 读数**写回台账**（2026-11-13 新增）。

    为什么必需：`promote_candidate.check_arm()` 的板端参照只认台账里的板卡会话
    （`st["board_ms"]` / `ns.board_ms_for`）⇒ 自动测速只写 `measured.json` 的话，
    升轨工具会误判「未测板端时延」⇒ `--require-board` 直接拒。
    """
    try:
        led = ns.load_ledger()
        arms = led.get("arms") or {}
        st = arms.get(str(trial))
        if st is None:
            log(f"[T4] 台账里没有 trial={trial} ⇒ 跳过回写")
            return
        st["board_ms"] = float(ms)
        st.setdefault("board_ms_epochs", {})[str(ep)] = float(ms)
        st["board_pass_t4"] = bool(float(ms) <= LIMIT_MS)
        st["board_limit_ms"] = LIMIT_MS
        st["board_measured_at"] = time.strftime("%F %T")
        arms[str(trial)] = st
        led["arms"] = arms
        ns.save_ledger(led)
        log(f"[T4] 台账回写：trial={trial} ep={ep} board_ms={ms} "
            f"pass={st['board_pass_t4']}")
    except Exception as exc:  # noqa: BLE001
        log(f"[T4] 台账回写失败（忽略）：{exc}")


def write_back_ledger_batch(items: list) -> None:
    """🟢 2026-11-19：一次会话会带回 20+ 个臂的读数 ⇒ **批量回写**（一次 load/save）。

    为什么要批量：`write_back_ledger` 每次都 load+save 整个台账 ⇒ 25 次调用就是
    25 个「读-改-写」窗口 ⇒ 极易与门禁通道的并发写互相覆盖。批量只开 1 个窗口。
    """
    try:
        led = ns.load_ledger()
        arms = led.get("arms") or {}
        n = 0
        for trial, ep, ms in items:
            st = arms.get(str(trial))
            if st is None:
                continue
            st["board_ms"] = float(ms)
            st.setdefault("board_ms_epochs", {})[str(ep)] = float(ms)
            st["board_pass_t4"] = bool(float(ms) <= LIMIT_MS)
            st["board_limit_ms"] = LIMIT_MS
            st["board_measured_at"] = time.strftime("%F %T")
            arms[str(trial)] = st
            n += 1
        led["arms"] = arms
        ns.save_ledger(led)
        if n:
            log(f"[T4] 台账批量回写 {n} 个臂的板端读数（中位数口径）")
    except Exception as exc:  # noqa: BLE001
        log(f"[T4] 台账批量回写失败（忽略）：{exc}")


def sync_measured_to_ledger() -> None:
    """把 `measured.json` 里**已测过**的候选同步进台账（启动时跑一次）。

    背景：`measured.json` 里的读数是**先于**「回写台账」功能产生的（2026-11-13 15:24
    之前只有文件、没有台账字段）⇒ 升轨工具会误判「未测板端时延」。
    ⇒ 启动时补一次同步，保证台账与实测文件一致。
    """
    if not os.path.isfile(DONE):
        return
    try:
        with open(DONE, encoding="utf-8") as fh:
            done = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        log(f"[sync] measured.json 读取失败（跳过）：{exc}")
        return
    items = []
    for key, v in (done or {}).items():
        ms = v.get("ms")
        if ms is None:
            continue
        try:
            trial = int(v.get("trial") or key.split("_ep")[0])
            ep = str(v.get("ep") or key.split("_ep")[-1])
        except Exception:  # noqa: BLE001
            continue
        items.append((trial, ep, float(ms)))
    # 🟢 2026-11-19：一次写全（原来逐臂 load+save 84 次 ⇒ 丢更新风险 + 慢）
    write_back_ledger_batch(items)
    log(f"[sync] 已测候选 {len(items)} 个 ⇒ 同步进台账（中位数口径）")


def candidates() -> list[dict]:
    try:
        rows = A.collect()
    except Exception as exc:  # noqa: BLE001  （台账正在写盘时可能读坏 ⇒ 下轮再来）
        log(f"[scan] 台账/门禁读取失败（下轮再来）：{exc}")
        return []
    out = []
    near = []
    six = []
    for r in rows:
        n, miss = A.judge(r)
        if n == 6:
            six.append(r)
        elif n == 5:
            # 🟢 2026-11-17 11:35：**5/6 候选也要上板卡**。
            #   理由：内部「六项全压过基线」判据里 T2 虚警要求 ≤ 基线(=1)，
            #   而**任务书交付红线是 T2_fp ≤ 20（官方门禁 ≤ 36）** ⇒ 「只差 T2 一项」的候选
            #   仍属**交付可用**（如 1105000 汤：五项全压过基线，T2_fp=3 远低于红线），
            #   ⇒ 必须给出它的板卡时延，否则无法回答「有没有更好的模型」。
            near.append(r)
    near.sort(key=lambda r: -(r.get("nano") or 0.0))
    # 🔥 2026-11-18 13:55：**六项全过的臂排在队首**（升轨候选）。
    #   原因：原实现把 6/6、5/6、交付候选混在一起按 nano 排序
    #   ⇧ nano 最高的是**老臂**（nano .7238 但 T1/T3 差）→ 真正的升轨候选
    #   （1540030 / 1572001 / 1572034）被挤到队尾→ 板卡 T4 读数拖到每 10 分钟才 1 个。
    # 🟢🟢 2026-11-19 08:58 —— **板卡队列改成「升轨关键臂优先」**（昨夜板卡全程不可达后的补救）
    #   昨夜事实：板卡 18:13 起 SSH 全超时 ⇒ 1605008/1617002/1616005/1622004 四个关键臂全部
    #   「上传失败」⇒ **一整夜零板卡读数**（已测数始终停在 68）⇒ 升轨链最后一环全断。
    #   板卡今晨恢复后，若队列仍按 nano 单键排序 ⇒ 会先测**家族只有 1 臂**的臂（如 1612001，
    #   nano .724244 但它家族六项全过只有它自己）⇒ **测了也升不了轨**（升轨要求同家族 n≥4 + 板卡 PASS）。
    #   修法：六项全过桶里「同家族六项全过臂 ≥4」的臂整体提到最前，桶内再按 nano → nano4 降序
    #         ⇒ 板卡一恢复就直冲能触发升轨的家族（bw07/bw06/bw05/bw04/bw38）。
    _famcount: dict = {}
    try:
        from manu.pipeline.npu import family_key as _FK  # noqa: PLC0415
        _led2 = ns.load_ledger()
        _cnt: dict = {}
        for r in rows:
            _arm = (_led2.get("arms") or {}).get(str(r.get("trial"))) or {}
            _k = _FK.arm_family_key(_arm)
            _cnt[_k] = _cnt.get(_k, 0) + 1
        for r in rows:
            _arm = (_led2.get("arms") or {}).get(str(r.get("trial"))) or {}
            _famcount[int(r.get("trial"))] = _cnt.get(_FK.arm_family_key(_arm), 0)
    except Exception as exc:  # noqa: BLE001
        log(f"[queue] 家族计数失败（退回 nano 排序）：{exc}")
    six.sort(key=lambda r: (0 if _famcount.get(int(r.get("trial")), 0) >= 4 else 1,
                             -(r.get("nano") or 0.0), -(r.get("nano4") or 0.0)))
    out.extend(six)
    # 🔥 2026-11-18 17:40：**Track-B 桶提到 near/deliver 之前**
    #   理由：Track-B 臂（五项精度全 ≥ 基线 + T2_fp ≤ 5）是**今晚唯一可能被升轨的一类**
    #   ⇒ 它的板卡 T4 是升轨链最后一环 ⇒ 必须最先测（每臂测速 ~8~13 min，队列顺序直接决定能否升轨）。
    #   near / deliver 桶只是「历史可交付臂」的信息性补测 ⇒ 排到 Track-B 之后。
    # 🟢 2026-11-17 11:45：**任务书交付红线候选也要上板卡**。
    #   理由：全盘有 100+ 个臂过交付红线（其中老臂 14049/14445 的 nano .7328 / F1 .8931
    #   明显优于当前基线与 1105000）⇒ 只有补上板卡时延，才能回答「有没有更好的**可交付**模型」。
    # 🟢🟢 2026-11-18 16:20：**Track-B 桶**（五项精度指标全 ≥ 基线 + T2_fp ≤ 5）
    #   为什么单开一桶：Track-B 臂在**严格轨**里差在 T2_fp（≤5 但 >1）⇒ 判 4/6 ⇒
    #   既进不了 6/6 桶、也进不了 5/6 桶 ⇒ 只剩「交付红线桶」排队 ⇒ 板卡读数常年缺
    #   ⇒ Track-B 候选**无法升轨**（升轨链要求板卡 T4 PASS）。
    seen = {int(r["trial"]) for r in out}
    trackb = []
    try:
        from trackb_core import T2_USER, base_of, load_ledger, qualifies  # noqa: PLC0415
        _led = load_ledger()
        _base = base_of(_led)
        for r in rows:
            if int(r["trial"]) in seen:
                continue
            try:
                ok, _miss, _win = qualifies(r, _base, T2_USER, require_board=False)
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                trackb.append(r)
    except Exception as exc:  # noqa: BLE001
        log(f"[scan] Track-B 桶取数失败（跳过）：{exc}")
    trackb.sort(key=lambda r: -(r.get("nano") or 0.0))
    out.extend(trackb[: max(1, args_trackb_max())])
    #   ⇒ near 桶放到 Track-B 之后（见上方注释）
    out.extend(near[: max(1, args_near_max())])
    seen = {int(r["trial"]) for r in out}
    deliver = [r for r in rows if int(r["trial"]) not in seen and deliver_ok(r)]
    deliver.sort(key=lambda r: -(r.get("nano") or 0.0))
    out.extend(deliver[: max(1, args_deliver_max())])
    return out


def args_deliver_max() -> int:
    try:
        return int(os.environ.get("BOARD_T4_DELIVER_MAX", "6"))
    except Exception:  # noqa: BLE001
        return 6


def args_trackb_max() -> int:
    """Track-B 桶（16:20 新增）：五项精度全 ≥ 基线、T2_fp ≤ 5，但严格轨差在 T2_fp ⇒ 5/6 桶收不到。"""
    try:
        return int(os.environ.get("BOARD_T4_TRACKB_MAX", "8"))
    except Exception:  # noqa: BLE001
        return 8


def args_near_max() -> int:
    try:
        return int(os.environ.get("BOARD_T4_NEAR_MAX", "16"))
    except Exception:  # noqa: BLE001
        return 8


def board_busy() -> bool:
    """🔴 2026-11-18 17:55：板卡是**共享资源** ⇒ 别人在跑任务时读数不可信 ⇒ 本轮跳过。

    事实（今晚）：板卡 192.168.0.64 上有别人的 `rknn_mosaic_e2e_v9` 全序列任务（8 核全占 + NPU）。
    此时跑我们的 measure_hm_speed_board.sh，读数会被别人的负载抬高 ⇒ **污染 T4 读数**
    ⇒ 宁可不出读数，也不能出**错的**读数（升轨红线只剩 0.2~0.4 ms 余量，噪声一次就能判死好臂）。
    """
    remote = "ps -eo args | grep -E 'rknn_mosaic|e2e_v|measure_hm' | grep -v grep | wc -l"
    try:
        r = subprocess.run(
            ["bash", "-c", f'timeout 30 ssh {KEY} {B.BOARD} "{remote}"'],
            capture_output=True, text=True, timeout=60)
        n = int((r.stdout or "0").strip() or 0)
        if n > 0:
            log(f"[precheck] 板卡上有 {n} 个其它测速任务在跑 ⇒ 本轮跳过（读数会污染）")
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        log(f"[precheck] 板卡忙检查失败（保守跳过）：{exc}")
        return True


def run_once() -> None:
    os.makedirs(OUT, exist_ok=True)
    done = {}
    if os.path.isfile(DONE):
        try:
            done = json.load(open(DONE))
        except Exception:  # noqa: BLE001
            done = {}
    pools = load_pools()
    cands = candidates()
    log(f"[scan] 六项全过候选 {len(cands)} 个；已测 {len(done)} 个；样本池 {len(pools)} 个臂")
    if not cands:
        save_pools(pools)
        return
    if not board_ready():
        return
    if board_busy():          # 🔴 共享板卡：别人在跑 ⇒ 读数污染 ⇒ 跳过本轮
        return
    # 🟢🟢 2026-11-19：**样本不足的候选要补测**（不是「测过一次就跳过」）
    #   老逻辑 `if key in done: continue` ⇒ 一次噪声读数就把好臂永久判死（1702011/1740004）。
    #   新逻辑：只要**样本数 < MIN_SAMPLES** 就继续测，直到中位数稳定。
    try:
        B = A.live_base()
    except Exception:  # noqa: BLE001
        B = {}

    def _strict(r: dict) -> int:
        n = 0
        for k, bk in (("nano", "nano"), ("nano4", "nano4"), ("f1", "f1"),
                      ("t1", "t1"), ("t3", "t3")):
            v, bv = r.get(k), (B or {}).get(bk)
            if v is not None and bv is not None and v > bv:
                n += 1
        if r.get("t2_fp") is not None and (B or {}).get("t2_fp") is not None \
                and r["t2_fp"] < B["t2_fp"]:
            n += 1
        return n

    todo = []
    for r in cands:
        key = f"{r['trial']}_ep{r['ep']}"
        pool = pools.get(key) or []
        # 🟢 2026-11-19：**判 FAIL 的也要继续补样本**（上限 MAX_SAMPLES）
        #   现场：1740004 中位数一度 19.990（只超红线 0.001 ms）⇒ 不能一票否决。
        need = len(pool) < MIN_SAMPLES
        if not need and len(pool) < MAX_SAMPLES and pool and pool_median(pool) > LIMIT_MS:
            need = True
        if need:
            todo.append((_strict(r), r, key))
    # 严格优于的轴多的优先 ⇒ 最可能成为新基线的臂先攒样本
    # 🔴🔴 2026-11-19 13:55 **加「可升轨家族」加权**（与 night_soup_batch 同根因）
    #   现场事故：bw10 族（1940007 六项全过、nano4 严格更好 +7.09e-4）需要板卡 T4 才能升轨，
    #     但 todo 按 (-严格优轴, -nano) 排序 ⇒ **nano 更高但已判死的针族臂**（nano .7244802）
    #     永远排前面 ⇒ 板卡机时全花在死路上，真正能升轨的 bw10 臂排到队尾。
    #   修法：族内存在「六项全过且至少一轴严格优于基线」的臂 ⇒ 该族为可升轨家族 ⇒ 整体提前。
    #     （纯打平族如 bw08 的 9 个六项全过臂不满足「严格更好」⇒ 不被提前。）
    _promo_fams: set = set()
    try:
        _led3 = ns.load_ledger()
        _b3 = ns.BASE_V2
        for _st3 in ((_led3.get("arms") or {}).values()):
            _of3 = _st3.get("official") or {}
            if not _of3.get("six_ok"):
                continue
            _sf3 = str(_st3.get("soup_family") or "")
            if not _sf3:
                continue
            _pk3 = _st3.get("picked") or {}
            _st = False
            for _k in ("nano", "nano4", "f1"):
                if (_pk3.get(_k) or 0) > (_b3.get(_k) or 0) + 1e-12:
                    _st = True
            for _k in ("t1", "t3"):
                if (_of3.get(_k) is not None) and (_b3.get(_k) is not None) \
                        and _of3[_k] > _b3[_k] + 1e-12:
                    _st = True
            if _of3.get("t2_fp") is not None and _b3.get("t2_fp") is not None \
                    and _of3["t2_fp"] < _b3["t2_fp"]:
                _st = True
            if _st:
                _promo_fams.add(_sf3)
        log(f"[T4] 可升轨家族 {len(_promo_fams)} 个：{sorted(_promo_fams)[:6]}")
    except Exception as exc:  # noqa: BLE001
        log(f"[T4] 可升轨家族判定失败（退回原排序）：{exc}")
    try:
        from manu.pipeline.npu import family_key as _FK2  # noqa: PLC0415
        _led4 = ns.load_ledger()
        _pf = {}
        for r in todo:
            _arm4 = (_led4.get("arms") or {}).get(str(r[1].get("trial"))) or {}
            # 🔴 2026-11-19 14:12 **家族键口径统一**（根因修复）
            #   `_promo_fams` 取自台账 `soup_family`（"sb3:..."），
            #   这里原来用 `_FK2.arm_family_key`（"soup3:sb3:..."）⇒ 前缀对不上 ⇒ 加权 100% 失效。
            _pf[r[1].get("trial")] = str(_arm4.get("soup_family")
                                         or _FK2.arm_family_key(_arm4))
    except Exception as exc:  # noqa: BLE001
        log(f"[T4] 家族键取数失败（退回原排序）：{exc}")
        _pf = {}
    # 🔴🔴 2026-11-19 14:12 **升轨预备硬优先级**（只差板卡就能升轨的臂必须先测）
    #   判据：official.six_ok=True 且 同族 six_ok 成员 >=2（种子复现关已过）
    #         且 至少一轴严格优于基线 ⇒ 该臂只差板卡 T4 ⇒ 板卡机时优先给它。
    #   现场：bw10 族 1940007/2020022/2020035 六项全过 + 种子关已过 ⇒ 只差板卡 ms 就能升轨，
    #     但 todo 按 nano 排序 ⇒ 已被判死的针族臂（nano 更高）永远排前 ⇒ 板卡机时全烧死路。
    _seed_ready = set()
    try:
        _led5 = ns.load_ledger()
        _b5 = ns.BASE_V2
        _cnt5 = {}
        _cand5 = []
        for _st5 in ((_led5.get("arms") or {}).values()):
            _of5 = _st5.get("official") or {}
            if not _of5.get("six_ok"):
                continue
            _sf5 = str(_st5.get("soup_family") or "")
            if not _sf5:
                continue
            _cnt5[_sf5] = _cnt5.get(_sf5, 0) + 1
            _pk5 = _st5.get("picked") or {}
            _st = False
            for _k in ("nano", "nano4", "f1"):
                if (_pk5.get(_k) or 0) > (_b5.get(_k) or 0) + 1e-12:
                    _st = True
            for _k in ("t1", "t3"):
                if (_of5.get(_k) is not None) and (_b5.get(_k) is not None) \
                        and _of5[_k] > _b5[_k] + 1e-12:
                    _st = True
            if _of5.get("t2_fp") is not None and _b5.get("t2_fp") is not None \
                    and _of5["t2_fp"] < _b5["t2_fp"]:
                _st = True
            if _st:
                _cand5.append((int(_st5.get("trial") or 0), _sf5))
        for _sid5, _sf5 in _cand5:
            if _cnt5.get(_sf5, 0) >= 2:
                _seed_ready.add(_sid5)
        log(f"[T4] 升轨预备臂（种子已过+只差板卡）：{sorted(_seed_ready)}")
    except Exception as exc:  # noqa: BLE001
        log(f"[T4] 升轨预备判定失败（退回原排序）：{exc}")
        _seed_ready = set()
    todo.sort(key=lambda t: (0 if int(t[1].get("trial") or 0) in _seed_ready else 1,
                             -t[0], 0 if _pf.get(t[1].get("trial")) in _promo_fams else 1,
                             -(t[1].get("nano") or 0.0)))
    log(f"[T4] 待补样本候选 {len(todo)} 个（样本 < {MIN_SAMPLES}）")
    for _ns, r, key in todo:
        log(f"[T4] 开始测 {key}（nano={r.get('nano')}｜严格优轴={_ns}｜已有样本 {len(pools.get(key) or [])}）")
        out = measure(int(r["trial"]), str(r["ep"]))
        prune_remote_arms()          # 🔴 测速后立刻清理板卡 rknn（防磁盘写满断链）
        if not out:
            continue
        touched = merge_session(pools, out.get("all_ms") or {})
        if out["ms"] is not None and key not in touched:
            pools.setdefault(key, []).append(float(out["ms"]))
            touched.append(key)
        for k2 in sorted(set(touched)):
            xs = pools.get(k2) or []
            if not xs:
                continue
            med = pool_median(xs)
            rec = done.get(k2) or {"trial": int(k2.split("_ep")[0]), "ep": k2.split("_ep")[-1]}
            rec.update({"ms": round(med, 4), "ms_last": xs[-1],
                        "samples": [round(float(x), 4) for x in xs],
                        "n_samples": len(xs), "limit_ms": LIMIT_MS,
                        "pass_t4": bool(med <= LIMIT_MS),
                        "all_ms": out.get("all_ms"), "ts": time.strftime("%F %T")})
            done[k2] = rec
        save_pools(pools)
        with open(DONE, "w") as fh:
            json.dump(done, fh, indent=1, ensure_ascii=False)
        # 台账回写：候选臂 + 本会话顺带测到的其它臂（升轨工具只认台账里的 board_ms）
        _bw = []
        for k2 in sorted(set(touched)):
            rec = done.get(k2) or {}
            if rec.get("ms") is None:
                continue
            _bw.append((int(k2.split("_ep")[0]), k2.split("_ep")[-1], float(rec["ms"])))
        if _bw:
            write_back_ledger_batch(_bw)
        rec = done.get(key) or {}
        log(f"[T4] {key} ⇒ 样本 {rec.get('n_samples')} 个 {rec.get('samples')} "
            f"中位数 {rec.get('ms')} ms（红线 {LIMIT_MS}）⇒ "
            f"{'PASS ✔' if rec.get('pass_t4') else 'FAIL ✘'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="只跑一轮")
    ap.add_argument("--watch", action="store_true", help="常驻（默认 20 min/轮）")
    ap.add_argument("--interval", type=int, default=1200, help="轮间隔秒")
    ap.add_argument("--deadline", default="2026-11-16 09:00", help="截止时间")
    a = ap.parse_args()
    dl = time.mktime(time.strptime(a.deadline, "%Y-%m-%d %H:%M"))
    try:
        sync_measured_to_ledger()      # 启动时把已测结果补进台账（升轨工具要读）
    except Exception as exc:  # noqa: BLE001
        log(f"[sync] 启动同步异常（忽略）：{exc}")
    while True:
        try:
            run_once()
        except Exception as exc:  # noqa: BLE001  ⇒ 任何异常都不能让守望退出
            log(f"[error] 本轮异常：{exc}")
        if a.once or time.time() > dl:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()