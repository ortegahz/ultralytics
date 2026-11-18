
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
    """板卡只读预检：忙 / 空间不足 / 连不上 ⇒ False（本轮跳过，绝不抢别人的板）。"""
    try:
        pre = subprocess.run(
            ["bash", "-c",
             f"ssh {KEY} {B.BOARD} \"awk '{{print \\$1}}' /proc/loadavg; "
             "ps -eo cmd | grep -E '[r]knn|[e]2e_|[m]easure_hm_speed' | wc -l; "
             "df -m / | tail -1 | awk '{print \\$4}'\""],
            capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        log(f"[precheck] 异常（忽略）：{exc}")
        return False
    lines = [x.strip() for x in (pre.stdout or "").splitlines() if x.strip()]
    if len(lines) < 3:
        log("[precheck] 板卡无响应 ⇒ 本轮跳过")
        return False
    try:
        load, nproc, free_mb = float(lines[0]), int(lines[1]), int(lines[2])
    except ValueError:
        log(f"[precheck] 输出异常 ⇒ 本轮跳过：{lines[:3]}")
        return False
    if load > 1.0 or nproc > 0 or free_mb < 400:
        log(f"[precheck] 板卡忙（load={load} 别人的进程={nproc} 余量={free_mb}MB）⇒ 本轮跳过")
        return False
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
    up = subprocess.run(
        ["bash", "-c",
         f"scp {KEY} {rknn} {B.BOARD}:{B.BOARD_DIR}/arms/ && "
         f"ssh {KEY} {B.BOARD} 'printf {stride} > {B.BOARD_DIR}/arms/{name}_int8.stride'"],
        capture_output=True, text=True, timeout=1800)
    if up.returncode != 0:
        log(f"[T4] 上传失败 {trial}@{ep}：{(up.stderr or '')[-300:]}")
        return None
    r = subprocess.run(
        ["bash", "-c",
         f"ssh {KEY} {B.BOARD} 'cd {B.BOARD_DIR} && BOARD_DIR={B.BOARD_DIR} "
         "CONTROL_NAME=base_int8.rknn LOOPS=30 WARMUP=10 SETTLE=5 bash measure_hm_speed_board.sh'"],
        capture_output=True, text=True, timeout=3600)
    txt = (r.stdout or "") + (r.stderr or "")
    with open(os.path.join(OUT, f"session_{trial}_ep{ep}.txt"), "w") as fh:
        fh.write(txt)
    res: dict[str, float] = {}
    for m in re.finditer(r"@@SUMMARY@@ (\S+)\|rknn_run_avg_ms\|([\d.]+)", txt):
        res[m.group(1)] = float(m.group(2))
    ms = _pick_ms(res, trial, ep)
    out = {"trial": trial, "ep": ep, "ckpt": ck, "ms": ms, "limit_ms": LIMIT_MS,
           "pass_t4": bool(ms is not None and ms <= LIMIT_MS),
           "all_ms": res, "ts": time.strftime("%F %T")}
    return out


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
    n = 0
    for key, v in (done or {}).items():
        ms = v.get("ms")
        if ms is None:
            continue
        try:
            trial = int(v.get("trial") or key.split("_ep")[0])
            ep = str(v.get("ep") or key.split("_ep")[-1])
        except Exception:  # noqa: BLE001
            continue
        write_back_ledger(trial, ep, float(ms))
        n += 1
    log(f"[sync] 已测候选 {n} 个 ⇒ 同步进台账")


def candidates() -> list[dict]:
    try:
        rows = A.collect()
    except Exception as exc:  # noqa: BLE001  （台账正在写盘时可能读坏 ⇒ 下轮再来）
        log(f"[scan] 台账/门禁读取失败（下轮再来）：{exc}")
        return []
    out = []
    near = []
    for r in rows:
        n, miss = A.judge(r)
        if n == 6:
            out.append(r)
        elif n == 5:
            # 🟢 2026-11-17 11:35：**5/6 候选也要上板卡**。
            #   理由：内部「六项全压过基线」判据里 T2 虚警要求 ≤ 基线(=1)，
            #   而**任务书交付红线是 T2_fp ≤ 20（官方门禁 ≤ 36）** ⇒ 「只差 T2 一项」的候选
            #   仍属**交付可用**（如 1105000 汤：五项全压过基线，T2_fp=3 远低于红线），
            #   ⇒ 必须给出它的板卡时延，否则无法回答「有没有更好的模型」。
            near.append(r)
    near.sort(key=lambda r: -(r.get("nano") or 0.0))
    out.extend(near[: max(1, args_near_max())])
    # 🟢 2026-11-17 11:45：**任务书交付红线候选也要上板卡**。
    #   理由：全盘有 100+ 个臂过交付红线（其中老臂 14049/14445 的 nano .7328 / F1 .8931
    #   明显优于当前基线与 1105000）⇒ 只有补上板卡时延，才能回答「有没有更好的**可交付**模型」。
    seen = {int(r["trial"]) for r in out}
    deliver = [r for r in rows if int(r["trial"]) not in seen and deliver_ok(r)]
    deliver.sort(key=lambda r: -(r.get("nano") or 0.0))
    out.extend(deliver[: max(1, args_deliver_max())])
    out.sort(key=lambda r: -(r.get("nano") or 0.0))
    return out


def args_deliver_max() -> int:
    try:
        return int(os.environ.get("BOARD_T4_DELIVER_MAX", "6"))
    except Exception:  # noqa: BLE001
        return 6


def args_near_max() -> int:
    try:
        return int(os.environ.get("BOARD_T4_NEAR_MAX", "8"))
    except Exception:  # noqa: BLE001
        return 8


def run_once() -> None:
    os.makedirs(OUT, exist_ok=True)
    done = {}
    if os.path.isfile(DONE):
        try:
            done = json.load(open(DONE))
        except Exception:  # noqa: BLE001
            done = {}
    cands = candidates()
    log(f"[scan] 六项全过候选 {len(cands)} 个；已测 {len(done)} 个")
    if not cands:
        return
    if not board_ready():
        return
    for r in cands:
        key = f"{r['trial']}_ep{r['ep']}"
        if key in done:
            continue
        log(f"[T4] 开始测 {key}（nano={r.get('nano')}）")
        out = measure(int(r["trial"]), str(r["ep"]))
        if not out:
            continue
        if out["ms"] is None:
            log(f"[T4] {key} 未取到 rknn_run_avg_ms ⇒ 不落盘（下轮重试）")
            continue
        # 🟢🟢 2026-11-17 17:30 —— **板卡是共享资源 ⇒ 临界读数要复测**
        #   事实：板卡同时有别人的任务在跑 ⇒ 单次 rknn_run_avg_ms 会**被其它负载抬高**，
        #   而升轨红线是 19.989 ms（余量常常只有 0.2~0.4 ms）⇒ 一次噪声就可能把好臂判成 FAIL。
        #   做法：**只对六项全过的候选**（真正要升轨的那几个）且**读数落在临界带**时复测，
        #        取 3 次的**中位数**（抗单次异常），并把全部样本记进 done（晨报可核对）。
        samples = [out["ms"]]
        try:
            _six = A.judge(r)[0] == 6
        except Exception:  # noqa: BLE001
            _six = False
        if _six and BORDER_LO <= out["ms"] <= BORDER_HI:
            for _rep in range(2):
                log(f"[T4] {key} 读数 {out['ms']:.3f} ms 落在临界带 ⇒ 复测 {_rep + 1}/2")
                out2 = measure(int(r["trial"]), str(r["ep"]))
                if out2 and out2.get("ms") is not None:
                    samples.append(float(out2["ms"]))
                    log(f"[T4] {key} 复测读数 {out2['ms']:.3f} ms")
        if len(samples) > 1:
            samples.sort()
            med = samples[len(samples) // 2]
            log(f"[T4] {key} 样本 {samples} ⇒ 取中位数 {med:.3f} ms")
            out["ms"] = med
            out["pass_t4"] = bool(med <= LIMIT_MS)
        out["samples"] = samples
        done[key] = out
        with open(DONE, "w") as fh:
            json.dump(done, fh, indent=1, ensure_ascii=False)
        write_back_ledger(int(r["trial"]), str(r["ep"]), out["ms"])
        log(f"[T4] {key} ⇒ {out['ms']} ms（红线 {LIMIT_MS}）⇒ "
            f"{'PASS ✔' if out['pass_t4'] else 'FAIL ✘'}")


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