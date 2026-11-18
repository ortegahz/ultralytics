#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""臂内细粒度 SWA 扫描器（**零训练**）—— 在「近失」臂内部搜 epoch 权重组合。

═══ 目的 ═══
基线 Baseline_v2.0 = trial_9863，六项判据（比较口径：严格 ≥ / ≤）：
    nano  ≥ 0.708648393194707   （NanoBench 8px 桶 nano）
    nano4 ≥ 0.6472117202268431   （NanoBench 4px 桶 nano）
    f1    ≥ 0.8897117275427806   （NanoBench 8px 桶 F1）
    t1    ≥ 0.888086             （官方 GMC 三轨 T1）
    t2_fp ≤ 5                    （官方 GMC 三轨 T2 虚警）
    t3    ≥ 0.948489             （官方 GMC 三轨 T3）
「臂三件套 / triad」= 前三项（nano / nano4 / f1）同时过线。

本脚本**不训练**：只在已有 ckpt 上做**臂内 SWA（epoch 权重平均）**——
把一个近失臂的若干 epoch 平均成一个合成臂，用同一口径评测
⇒ 找 triad 乃至六项全过的组合。

═══ 口径（与主流水线逐位一致）═══
  ① 合成 ckpt = manu.pipeline.npu.ckpt_synth.average_checkpoints
     （只平均**同拓扑** ckpt：
     no_p2/head_level/act/head_conv/late_upsample/stride 签名不一致直接拒绝）；
  ② 评测 = manu.pipeline.npu.route2_acceptance.build_cache(..., extra="--no-peak-attrs") 建 **no-GMC** 缓存
          + route2_acceptance.nanobench 读 nano / nano4 / f1
     ⇒ **GMC 与 no-GMC 严禁混用**：本脚本的 nano/nano4/f1 一律走 no-GMC；
  ③ 官方三轨 = manu.pipeline.npu.nano_search.run_official_gate（GMC 语义），
     只在 ``--gate`` 且候选 triad 时才跑（很贵，约 7 min GPU）。

═══ 候选生成规则（每个臂内）═══
  ① 任意无序 epoch 对，等权 [0.5, 0.5]；
  ② 相邻 epoch 对（磁盘 epoch 升序相邻），权重 [0.3, 0.7] 与 [0.7, 0.3]；
  ③ 该臂磁盘上**全部** epoch 等权平均。
  ⇒ 单 epoch 组合（= 已评过的原始 ckpt）跳过；成员+权重完全相同的组合去重。
  ⇒ 全局顺序：**最佳单 epoch 的 nano 越高 ⇒ 该臂越靠前**（先扫高价值臂）。

═══ 合成臂 id 与幂等规则 ═══
  * 合成臂 id = 96000 + sha1("trial|epochs|weights") % 1000 ⇒ **跨次运行稳定**（重跑不换臂）；
  * 合成 ckpt 落盘：{ROOT}/trials/trial_<synth_id>/weights/epoch5.pt
    （epoch 号固定 5，官方门禁按 epoch5 读）；
  * 同目录写标记文件 ``weights/fine_swa_synth.json``（来源臂 / 成员 / 权重 / 合成 id）⇒
    **护栏**：只允许写「目录不存在」或「目录里有本脚本标记」的合成臂目录
    ⇒ 绝不覆盖别人的 trial 目录；
  * 缓存 pkl 命名 fine_swa_<synth_id>.pkl ⇒ build_cache 命中即跳过 ⇒ 重跑不重复建缓存；
  * 报告写入前先合并磁盘旧报告（--resume 跳过已评过的候选）⇒ 中断可续跑。

用法（服务器）：
    python manu/pipeline/npu/fine_swa_sweep.py --dry-run --limit 3      # 只列候选，不建 ckpt/缓存
    python manu/pipeline/npu/fine_swa_sweep.py --limit 6 --gpu 2        # 扫描近失臂（不跑官方门禁）
    python manu/pipeline/npu/fine_swa_sweep.py --gate --resume          # triad 候选补跑官方三轨
    python manu/pipeline/npu/fine_swa_sweep.py --selftest               # CPU 自检（不碰 /mnt/data）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from typing import Any

# ---- 服务器路径常量（口径冻结，勿改）----
SERVER_REPO = "/tmp/pycharm_project_10ae9e2e"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
RUNS = "/mnt/data/siping/runs/npu20ms"
WORK = f"{RUNS}/nano_search"
LEDGER = f"{WORK}/nano_search_ledger.json"
REPORT = f"{WORK}/fine_swa_report.json"
SWEEP_LOG = f"{WORK}/fine_swa_sweep.log"
CACHE_DIR = f"{RUNS}/tiny_ref"

# ---- 基线（Baseline_v2.0 = trial_9863）六项精确线 ----
BASE = dict(
    nano=0.708648393194707,
    nano4=0.6472117202268431,
    f1=0.8897117275427806,
    t1=0.888086,
    t2_fp=5,
    t3=0.948489,
)

# ---- 近失（near-miss）判定的宽容度 ----
NEAR_NANO_SLACK = 0.012   # nano 允许差这么多
NEAR_NANO4_SLACK = 0.02   # nano4 允许差这么多

# ---- 合成臂 id 分配（确定性）----
SYNTH_ID_BASE = 96000      # 合成臂 id 基址
SYNTH_ID_SPAN = 1000       # 合成臂 id 空间 [96000, 96999]
SIDE_CAR = "fine_swa_synth.json"   # 合成臂标记文件（写盘护栏）
MARKER = "fine_swa_sweep"          # 标记文件里的 marker 字段

_HELPERS: tuple | None = None


# ============================================================================
# 基础设施
# ============================================================================
def log(msg: str) -> None:
    """终端日志（与主流水线同风格）。"""
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def repo_root() -> str:
    """定位仓库根：向上找同时含 ``manu/pipeline/npu`` 的目录。"""
    cur = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(cur, "manu", "pipeline", "npu")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.dirname(os.path.abspath(__file__))
        cur = parent


def ensure_repo_on_path() -> None:
    """把仓库根放进 ``sys.path`` ⇒ ``import manu.pipeline.npu.*`` 可用。

    ``manu/`` 下没有 ``__init__.py``（命名空间包），所以必须让**仓库根**在 path 上；
    服务器上的项目根 ``/tmp/pycharm_project_10ae9e2e`` 也一并放进去（存在才放）。
    """
    for p in (repo_root(), SERVER_REPO):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def import_helpers() -> tuple:
    """延迟导入三个**既有** helper（import 期不做任何重活）。

    返回 ``(average_checkpoints, route2_acceptance, nano_search)``。
    """
    global _HELPERS
    if _HELPERS is None:
        ensure_repo_on_path()
        from manu.pipeline.npu.ckpt_synth import average_checkpoints  # noqa: PLC0415
        from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415
        from manu.pipeline.npu import nano_search as ns  # noqa: PLC0415

        _HELPERS = (average_checkpoints, RA, ns)
    return _HELPERS


def load_ledger(path: str) -> dict:
    """读台账（不存在 / 坏文件 ⇒ 抛异常由调用方降级处理）。"""
    with open(path, encoding="utf-8") as f:
        led = json.load(f)
    return led if isinstance(led, dict) else {}


def trial_weights_dir(trial: int, weights_root: str = ROOT) -> str:
    """某臂的 weights 目录。"""
    return os.path.join(weights_root, "trials", f"trial_{trial}", "weights")


def epochs_on_disk(trial: int, weights_root: str = ROOT) -> list[int]:
    """磁盘上**真实存在**的 epoch ckpt（升序）。"""
    d = trial_weights_dir(trial, weights_root)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        if fn.startswith("epoch") and fn.endswith(".pt"):
            try:
                out.append(int(fn[5:-3]))
            except ValueError:
                continue
    return sorted(out)


def arm_nanobench(arm: dict) -> dict[int, dict]:
    """台账里该臂每个 epoch 的 NanoBench 读数（跳过 ``error`` 行与坏 epoch 键）。"""
    traj = arm.get("nanobench") or {}
    out: dict[int, dict] = {}
    for k, v in traj.items():
        if not isinstance(v, dict) or "f1" not in v:
            continue
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def metrics_of(entry: dict) -> dict:
    """从台账读数里抽出 nano / nano4 / f1 三项。"""
    return {
        "nano": float(entry.get("nano") or 0.0),
        "nano4": float(entry.get("nano4") or 0.0),
        "f1": float(entry.get("f1") or 0.0),
    }


def is_triad(m: dict, base: dict = BASE) -> bool:
    """臂三件套：nano / nano4 / f1 **同时** ≥ 基线（严格 ≥）。"""
    return (m["nano"] >= base["nano"] and m["nano4"] >= base["nano4"] and m["f1"] >= base["f1"])


def compute_deltas(m: dict, base: dict = BASE) -> dict:
    """相对基线三项阈值的差值（正 = 更好）。"""
    return {
        "d_nano": m["nano"] - base["nano"],
        "d_nano4": m["nano4"] - base["nano4"],
        "d_f1": m["f1"] - base["f1"],
    }


# ============================================================================
# 臂筛选与候选生成
# ============================================================================
def select_arms(led: dict, weights_root: str = ROOT, base: dict = BASE) -> list[dict]:
    """挑出「近失臂」。

    入选条件（两条都满足）：
      ① 磁盘上 **≥2 个** epoch ckpt 真实存在；
      ② 近失：存在某个 epoch 已 triad，**或**（各 epoch 最大 nano ≥ 基线−0.012 且
         最大 nano4 ≥ 基线−0.02）。

    返回按「最佳单 epoch 的 nano」降序排好的臂列表。
    """
    arms = led.get("arms") or {}
    picked: list[dict] = []
    for key, arm in arms.items():
        if not isinstance(arm, dict):
            continue
        try:
            trial = int(key)
        except (TypeError, ValueError):
            continue
        eps = epochs_on_disk(trial, weights_root)
        if len(eps) < 2:                       # 条件①：磁盘上至少两个 epoch ckpt
            continue
        traj = arm_nanobench(arm)
        ms = {ep: metrics_of(traj[ep]) for ep in traj}
        if not ms:
            continue
        any_triad = any(is_triad(m, base) for m in ms.values())
        max_nano = max(m["nano"] for m in ms.values())
        max_nano4 = max(m["nano4"] for m in ms.values())
        near = ((max_nano >= base["nano"] - NEAR_NANO_SLACK)
                and (max_nano4 >= base["nano4"] - NEAR_NANO4_SLACK))
        if not (any_triad or near):            # 条件②：近失
            continue
        best_ep = max(ms, key=lambda e: (ms[e]["nano"], -e))
        picked.append({
            "trial": trial,
            "epochs": eps,
            "metrics": ms,
            "best_epoch": best_ep,
            "best_nano": ms[best_ep]["nano"],
            "triad_epochs": sorted(e for e in ms if is_triad(ms[e], base)),
            "any_triad": any_triad,
            "near": near,
        })
    picked.sort(key=lambda a: (-a["best_nano"], a["trial"]))
    return picked


def generate_candidates(epochs: list[int]) -> list[dict]:
    """生成一个臂的候选 epoch 组合（成员升序、权重等长）。

    规则：① 任意无序对等权 [0.5, 0.5]；② 相邻对 [0.3, 0.7] 与 [0.7, 0.3]；
    ③ 全部 epoch 等权平均。**单 epoch 组合跳过**（= 已评过的原始 ckpt），
    成员+权重完全相同的组合**去重**（先出现的保留）。
    """
    eps = sorted({int(e) for e in epochs})
    cands: list[dict] = []
    seen: set[tuple] = set()

    def _add(members: list[int], weights: list[float]) -> None:
        if len(members) < 2:
            return
        key = (tuple(members), tuple(round(float(w), 6) for w in weights))
        if key in seen:
            return
        seen.add(key)
        cands.append({"members": list(members), "weights": [float(w) for w in weights]})

    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            _add([eps[i], eps[j]], [0.5, 0.5])
    for i in range(len(eps) - 1):
        _add([eps[i], eps[i + 1]], [0.3, 0.7])
        _add([eps[i], eps[i + 1]], [0.7, 0.3])
    if len(eps) >= 2:
        w = 1.0 / len(eps)
        _add(eps, [w] * len(eps))
    return cands


def plan_candidates(arms: list[dict]) -> list[dict]:
    """把各臂的候选串成一张全局清单（臂顺序 = 最佳单 epoch nano 降序）。"""
    plan: list[dict] = []
    for a in arms:
        for c in generate_candidates(a["epochs"]):
            plan.append({
                "trial": a["trial"],
                "members": c["members"],
                "weights": c["weights"],
                "arm_best_epoch": a["best_epoch"],
                "arm_best_nano": a["best_nano"],
            })
    return plan


def cand_key(trial: int, members: list[int], weights: list[float]) -> str:
    """候选的唯一键（报告 / sidecar / 合成 id 三处共用 ⇒ 幂等的锚点）。"""
    return (f"{trial}|{','.join(str(m) for m in members)}|"
            f"{','.join(f'{float(w):.6f}' for w in weights)}")


# ============================================================================
# 合成臂：id 分配 / 写盘护栏 / 平均
# ============================================================================
def synth_dir(synth_id: int, weights_root: str = ROOT) -> str:
    """合成臂目录（与普通臂同构 ⇒ 官方门禁能直接按 trial_<id> 读）。"""
    return os.path.join(weights_root, "trials", f"trial_{synth_id}")


def sidecar_path(synth_id: int, weights_root: str = ROOT) -> str:
    """合成臂标记文件（与 ckpt 同目录 ⇒ 护栏判据）。"""
    return os.path.join(synth_dir(synth_id, weights_root), "weights", SIDE_CAR)


def read_sidecar(synth_id: int, weights_root: str = ROOT) -> dict | None:
    """读合成臂标记文件（不存在 / 坏文件 ⇒ None）。"""
    p = sidecar_path(synth_id, weights_root)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def can_write_synth(synth_id: int, weights_root: str = ROOT) -> bool:
    """写盘护栏：只允许写「目录不存在」或「目录里有本脚本标记」的合成臂目录。"""
    if not os.path.isdir(synth_dir(synth_id, weights_root)):
        return True
    return read_sidecar(synth_id, weights_root) is not None


def alloc_synth_id(trial: int, members: list[int], weights: list[float],
                   weights_root: str = ROOT, probe: bool = True) -> int:
    """确定性合成臂 id：``96000 + sha1(key) % 1000`` ⇒ **跨次运行稳定**。

    ``probe=True`` 时做线性探测：撞到**别的**组合占用的 id 就顺延到下一个
    （撞到**同一**组合 ⇒ 复用该 id ⇒ 重跑幂等）。
    """
    key = cand_key(trial, members, weights)
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    for k in range(SYNTH_ID_SPAN):
        sid = SYNTH_ID_BASE + (h + k) % SYNTH_ID_SPAN
        if not probe:
            return sid
        d = synth_dir(sid, weights_root)
        if not os.path.isdir(d):
            return sid
        side = read_sidecar(sid, weights_root)
        if side and side.get("key") == key:
            return sid
    raise RuntimeError("合成臂 id 空间已耗尽")


def build_synth_ckpt(cand: dict, weights_root: str = ROOT) -> tuple[str, dict]:
    """把候选组合平均成合成 ckpt + 写标记 sidecar。返回 ``(ckpt 路径, sidecar)``。"""
    average_checkpoints, _RA, _ns = import_helpers()
    trial = cand["trial"]
    members = cand["members"]
    paths = [os.path.join(trial_weights_dir(trial, weights_root), f"epoch{m}.pt") for m in members]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"缺 ckpt：{missing}")
    sid = cand["synth_id"]
    if not can_write_synth(sid, weights_root):
        raise PermissionError(f"trial_{sid} 已存在且无本脚本标记 ⇒ 拒绝覆盖")
    out_dir = os.path.join(synth_dir(sid, weights_root), "weights")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "epoch5.pt")
    average_checkpoints(paths, out, weights=cand["weights"], epoch=5,
                        note=f"fine-swa:trial{trial}:" + "+".join(f"ep{m}" for m in members))
    side = {
        "marker": MARKER,
        "key": cand["key"],
        "source_trial": trial,
        "members": list(members),
        "member_paths": paths,
        "weights": list(cand["weights"]),
        "synth_id": sid,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, SIDE_CAR), "w", encoding="utf-8") as f:
        json.dump(side, f, ensure_ascii=False, indent=2)
    return out, side


# ============================================================================
# 评测 / 门禁
# ============================================================================
def evaluate_candidate(cand: dict, gpu: int, cache_dir: str = CACHE_DIR,
                       weights_root: str = ROOT) -> dict:
    """按主流水线**同口径**评测：no-GMC 缓存（--no-peak-attrs）+ NanoBench 分层。"""
    _average, RA, _ns = import_helpers()
    ck = os.path.join(synth_dir(cand["synth_id"], weights_root), "weights", "epoch5.pt")
    pkl = os.path.join(cache_dir, f"fine_swa_{cand['synth_id']}.pkl")
    os.makedirs(cache_dir, exist_ok=True)
    if not RA.build_cache(ck, pkl, gpu, extra="--no-peak-attrs"):
        raise RuntimeError(f"建缓存失败：{pkl}")
    res = RA.nanobench(pkl, f"fswa_{cand['synth_id']}")
    if not res:
        raise RuntimeError(f"NanoBench 失败：{pkl}")
    rb8 = res["peak_tol8"]["recall_by_bucket"]
    rb4 = res["peak_tol4"]["recall_by_bucket"]
    m = {
        "nano": float(rb8["nano"]),
        "nano4": float(rb4["nano"]),
        "f1": float(res["peak_tol8"]["f1"]),
        "micro": float(rb8["micro"]),
        "small": float(rb8["small"]),
        "regular": float(rb8["regular"]),
        "f1_4": float(res["peak_tol4"]["f1"]),
        "cache": pkl,
    }
    m["triad"] = is_triad(m)
    m["deltas"] = compute_deltas(m)
    return m


def gate_candidate(cand: dict, gpu: int) -> dict:
    """官方三轨（GMC）门禁 ⇒ t1 / t2_fp / t3 + ``six_ok``（六项全过）。

    注：``nano_search.run_official_gate`` 内部固定读服务器 ``ROOT`` ⇒ 合成臂须落在 ROOT 下。
    """
    _average, _RA, ns = import_helpers()
    g = ns.run_official_gate(cand["synth_id"], 5, gpu)
    if not g or "error" in g:
        return {"error": (g or {}).get("error", "no_report")}
    out = {
        "t1": g.get("t1"),
        "t1_fp": g.get("t1_fp"),
        "t2": g.get("t2"),
        "t2_fp": g.get("t2_fp"),
        "t3": g.get("t3"),
        "t3_fp": g.get("t3_fp"),
        "verdict": g.get("verdict"),
    }
    out["six_ok"] = bool(
        out["t1"] is not None and out["t2_fp"] is not None and out["t3"] is not None
        and out["t1"] >= BASE["t1"] and out["t2_fp"] <= BASE["t2_fp"]
        and out["t3"] >= BASE["t3"] and bool(cand.get("triad"))
    )
    return out


# ============================================================================
# 报告 / 日志
# ============================================================================
def write_report(path: str, report: dict) -> None:
    """原子写 JSON 报告（先写临时文件再 os.replace ⇒ 中断不留半截文件）。"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_sweep_log(line: str, path: str = SWEEP_LOG) -> None:
    """追加一行人类可读日志（失败只提示，不掀掉整轮）。"""
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:  # noqa: BLE001
        log(f"写 sweep 日志失败：{e}")


def fmt_candidate_line(entry: dict) -> str:
    """一个候选的一行人类可读摘要。"""
    members = "+".join(f"ep{m}" for m in entry.get("members", []))
    weights = "[" + ",".join(f"{float(w):g}" for w in entry.get("weights", [])) + "]"
    head = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} | trial={entry.get('trial')} "
            f"synth={entry.get('synth_id')} | {members} {weights}")
    if entry.get("error"):
        return f"{head} | ERROR {entry['error']}"
    d = entry.get("deltas") or {}
    line = (f"{head} | nano={entry.get('nano'):.6f} nano4={entry.get('nano4'):.6f} "
            f"f1={entry.get('f1'):.6f} | triad={entry.get('triad')} | "
            f"d_nano={d.get('d_nano', 0.0):+.6f} d_nano4={d.get('d_nano4', 0.0):+.6f} "
            f"d_f1={d.get('d_f1', 0.0):+.6f}")
    gate = entry.get("gate")
    if gate:
        if "error" in gate:
            line += f" | gate=ERR({gate['error']})"
        else:
            line += (f" | gate=t1={gate.get('t1')} t2_fp={gate.get('t2_fp')} t3={gate.get('t3')} "
                     f"six_ok={gate.get('six_ok')}")
    return line


# ============================================================================
# 台账登记（2026-11-13 16:45 新增）
# ============================================================================
def register_synth_arms(entries: dict, ledger_path: str = LEDGER) -> int:
    """把 SWA 合成候选**登记进台账**（2026-11-13 16:45 新增）。

    背景：SWA 候选此前只写自己的 ``fine_swa_report.json`` ⇒ 台账里查不到 ⇒
      ① ``six_criteria_audit`` / ``board_t4_watch`` 都看不到它 ⇒ 不会被自动测 T4；
      ② ``adaptive_runner`` 的抛光锚点选择（``warm_anchor_candidates``）也看不到它
         ⇒ 5/6 近失臂（如 trial_96835）拿不到任何抛光机会。
    登记内容与普通臂同口径：
      · ``nanobench["5"]``（六项审计脚本按这个键配对）+ ``picked``（调度器读这个）
      · ``official``（T1 / T2 虚警 / T3，来自官方三轨门禁）
      · ``params.init`` = 合成 ckpt，``params.synth`` = 1（⇒ 不算「训练臂」，不污染配方去重）
    """
    try:
        _average, _RA, ns = import_helpers()
    except Exception as exc:  # noqa: BLE001
        log(f"[登记] helper 导入失败（跳过）：{exc}")
        return 0
    try:
        led = ns.load_ledger()
    except Exception as exc:  # noqa: BLE001
        log(f"[登记] 台账读取失败（跳过）：{exc}")
        return 0
    arms = led.get("arms") or {}
    n = 0
    for e in entries.values():
        if e.get("error"):
            continue
        sid = e.get("synth_id")
        if sid is None or not e.get("triad"):
            continue
        g = e.get("gate") or {}
        if g.get("error") or g.get("t1") is None or g.get("t3") is None:
            continue
        ck = os.path.join(synth_dir(int(sid), ROOT), "weights", "epoch5.pt")
        if not os.path.isfile(ck):
            continue
        src = (arms.get(str(e.get("trial"))) or {}).get("params") or {}
        p = dict(src)
        p["init"] = ck
        p["synth"] = 1
        nb = {"nano": e.get("nano"), "nano4": e.get("nano4"), "f1": e.get("f1"),
              "micro": e.get("micro"), "small": e.get("small"), "regular": e.get("regular"),
              "f1_4": e.get("f1_4"), "epoch": 5, "epoch_key": "5", "ckpt": "epoch5.pt",
              "cache": e.get("cache")}
        arms[str(int(sid))] = {
            "trial": int(sid),
            "synth": True,
            "source_trial": e.get("trial"),
            "swa_members": e.get("members"),
            "swa_weights": e.get("weights"),
            "params": p,
            "seed": None,
            "nanobench": {"5": nb},
            "picked": dict(nb),
            "picked_epoch": 5,
            "official": {"t1": g.get("t1"), "t1_fp": g.get("t1_fp"),
                         "t2": g.get("t2"), "t2_fp": g.get("t2_fp"),
                         "t3": g.get("t3"), "t3_fp": g.get("t3_fp"),
                         "verdict": g.get("verdict")},
            "registered_by": "fine_swa_sweep",
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        n += 1
    if n:
        led["arms"] = arms
        try:
            ns.save_ledger(led)
            log(f"[登记] {n} 个 SWA 候选已进台账（⇒ 六项审计 / 板卡 T4 / 抛光锚点都能看到）")
        except Exception as exc:  # noqa: BLE001
            log(f"[登记] 台账写盘失败：{exc}")
            return 0
    return n


# ============================================================================
# 主流程
# ============================================================================
def main() -> None:
    """CLI 入口。"""
    ap = argparse.ArgumentParser(description="臂内细粒度 SWA 扫描（零训练）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少个近失臂（0=不限）")
    ap.add_argument("--max-candidates", type=int, default=0, help="最多评测多少个候选（0=不限）")
    ap.add_argument("--gpu", type=int, default=2, help="建缓存 / 官方门禁用的 GPU 序号")
    ap.add_argument("--gate", action="store_true", help="triad 候选额外跑官方三轨门禁")
    ap.add_argument("--selftest", action="store_true", help="CPU 自检（不碰 /mnt/data 与台账）")
    ap.add_argument("--report", default=REPORT, help="JSON 报告路径")
    ap.add_argument("--ledger", default=LEDGER, help="台账路径")
    ap.add_argument("--weights-root", default=ROOT, help="trials/ 根目录（ckpt 所在处）")
    ap.add_argument("--dry-run", action="store_true", help="只列候选：不建 ckpt / 缓存 / 门禁")
    ap.add_argument("--triad-first", action="store_true",
                    help="把「已有 triad epoch」的臂排到最前（实测：按裸 nano 排序时"
                         "前 12 名是官方 T1/T3 已崩的另一个域的臂，真·近失臂排在 79~137 位）")
    ap.add_argument("--resume", action="store_true", help="跳过报告里已存在的候选")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    # ---- 台账（不存在 ⇒ 本机没挂服务器产物 ⇒ 优雅降级，退出码 0）----
    if not os.path.isfile(args.ledger):
        log(f"台账不存在：{args.ledger} ⇒ 本机未挂载 /mnt/data（正常）⇒ 无可扫描的臂，结束。")
        return
    try:
        led = load_ledger(args.ledger)
    except Exception as e:  # noqa: BLE001
        log(f"台账读取失败：{e}（{args.ledger}）⇒ 直接结束。")
        return

    arms = select_arms(led, args.weights_root, BASE)
    if args.triad_first:
        arms = sorted(arms, key=lambda a: (0 if a.get("triad_epochs") else 1,
                                           -a["best_nano"], a["trial"]))
    if args.limit > 0:
        arms = arms[:args.limit]
    plan = plan_candidates(arms)
    if args.max_candidates > 0:
        plan = plan[:args.max_candidates]
    log(f"近失臂 {len(arms)} 个 ｜ 候选 {len(plan)} 个 ｜ 基线 nano={BASE['nano']:.6f} "
        f"nano4={BASE['nano4']:.6f} f1={BASE['f1']:.6f}")

    if args.dry_run:
        print(f"{'臂':>8}  {'成员':<18} {'权重':<16} 合成臂id")
        for c in plan:
            sid = alloc_synth_id(c["trial"], c["members"], c["weights"], probe=False)
            members = "+".join(f"ep{m}" for m in c["members"])
            weights = "[" + ",".join(f"{float(w):g}" for w in c["weights"]) + "]"
            print(f"{c['trial']:>8}  {members:<18} {weights:<16} {sid}")
        print(f"[fine-swa] dry-run：列了 {len(plan)} 个候选，未建任何 ckpt / 缓存。")
        return

    # ---- 报告骨架（--resume ⇒ 合并磁盘旧报告）----
    entries: dict[str, dict] = {}
    if os.path.isfile(args.report):
        try:
            with open(args.report, encoding="utf-8") as f:
                old = json.load(f)
            for c in (old.get("candidates") or []):
                if isinstance(c, dict) and c.get("key"):
                    entries[c["key"]] = c
        except Exception as e:  # noqa: BLE001
            log(f"旧报告读取失败（从头扫）：{e}")

    for i, cand in enumerate(plan, 1):
        key = cand_key(cand["trial"], cand["members"], cand["weights"])
        cand["key"] = key
        _e0 = entries.get(key) or {}
        _need_gate = bool(args.gate) and bool(_e0.get("triad")) and "gate" not in _e0
        if args.resume and key in entries and not _need_gate:
            log(f"[{i}/{len(plan)}] 跳过（报告里已有）：trial={cand['trial']} "
                f"{cand['members']} {cand['weights']}")
            continue
        if _need_gate:
            log(f"[{i}/{len(plan)}] 补跑官方门禁（报告里已有候选但缺 gate）：{key}")
        entry = {"key": key, "trial": cand["trial"], "members": cand["members"],
                 "weights": cand["weights"], "arm_best_epoch": cand["arm_best_epoch"],
                 "arm_best_nano": cand["arm_best_nano"]}
        try:
            cand["synth_id"] = alloc_synth_id(cand["trial"], cand["members"], cand["weights"],
                                              args.weights_root)
            build_synth_ckpt(cand, args.weights_root)
            entry["synth_id"] = cand["synth_id"]
            m = evaluate_candidate(cand, args.gpu, weights_root=args.weights_root)
            entry.update(m)
            log(f"[{i}/{len(plan)}] trial={cand['trial']} synth={cand['synth_id']} "
                f"{cand['members']} {cand['weights']} nano={m['nano']:.6f} "
                f"nano4={m['nano4']:.6f} f1={m['f1']:.6f} triad={m['triad']}")
            if args.gate and m["triad"]:
                entry["gate"] = gate_candidate(cand, args.gpu)
        except Exception as e:  # noqa: BLE001 —— 单候选失败不掀掉整轮
            entry["error"] = f"{type(e).__name__}: {e}"
            log(f"[{i}/{len(plan)}] 候选失败（继续）：{key} ⇒ {entry['error']}")
        entries[key] = entry
        append_sweep_log(fmt_candidate_line(entry))
        write_report(args.report, build_report(entries, arms, args.ledger))
        register_synth_arms(entries, args.ledger)

    report = build_report(entries, arms, args.ledger)
    write_report(args.report, report)
    log(f"报告写入 {args.report}")

    # ---- 终端汇总 ----
    print("\n%-10s %-20s %-16s %-10s %-10s %-10s %s" %
          ("臂", "成员", "权重", "nano", "nano4", "f1", "triad"))
    print("-" * 96)
    n_triad = 0
    for e in entries.values():
        if e.get("triad"):
            n_triad += 1
        members_s = "+".join(f"ep{m}" for m in e.get("members", []))
        weights_s = "[" + ",".join(f"{float(w):g}" for w in e.get("weights", [])) + "]"
        if e.get("error"):
            print("%-10s %-20s %-16s %s" % (e.get("trial"), members_s, weights_s,
                                           "ERROR " + str(e["error"])[:40]))
            continue
        print("%-10s %-20s %-16s %-10.6f %-10.6f %-10.6f %s" % (
            e.get("trial"), members_s, weights_s,
            e.get("nano", 0.0), e.get("nano4", 0.0), e.get("f1", 0.0),
            "★" if e.get("triad") else "-"))
    print(f"\n候选 {len(entries)} 个 ｜ triad {n_triad} 个 ｜ 报告 {args.report}")
    for e in entries.values():
        gate = e.get("gate") or {}
        if gate.get("six_ok"):
            print(f"★ SIX_OK trial={e.get('trial')} synth={e.get('synth_id')} "
                  f"members={e.get('members')} weights={e.get('weights')} "
                  f"t1={gate.get('t1')} t2_fp={gate.get('t2_fp')} t3={gate.get('t3')}")


def build_report(entries: dict, arms: list[dict], ledger_path: str) -> dict:
    """组装报告（每个候选一条 + 汇总）。"""
    cands = list(entries.values())
    n_triad = sum(1 for c in cands if c.get("triad"))
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ledger": ledger_path,
        "base": dict(BASE),
        "arms": [{"trial": a["trial"], "epochs": a["epochs"], "best_epoch": a["best_epoch"],
                  "best_nano": a["best_nano"], "triad_epochs": a["triad_epochs"],
                  "any_triad": a["any_triad"], "near": a["near"]} for a in arms],
        "candidates": cands,
        "summary": {
            "n_arms_considered": len(arms),
            "n_candidates": len(cands),
            "n_triad": n_triad,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


# ============================================================================
# 自检（CPU-only、不碰 /mnt/data 与台账）
# ============================================================================
class _StubTensor:
    """最小张量：够 ``average_checkpoints`` 做「浮点加权平均 + 整数取第一个」。

    仅在**本机没有 torch** 时随 :func:`_install_torch_stub` 一起注册，
    让 ``--selftest`` 在无 torch 的机器上也能跑通（服务器有真 torch ⇒ 不用它）。
    """

    __slots__ = ("data", "dtype")

    def __init__(self, data, dtype: str = "float32") -> None:
        self.data = [float(x) for x in data]
        self.dtype = dtype if isinstance(dtype, str) else str(dtype)

    def is_floating_point(self) -> bool:
        return "float" in str(self.dtype).lower()

    def float(self) -> "_StubTensor":
        return _StubTensor(self.data, "float32")

    def clone(self) -> "_StubTensor":
        return _StubTensor(self.data, self.dtype)

    def to(self, dtype) -> "_StubTensor":
        return _StubTensor(self.data, dtype if isinstance(dtype, str) else str(dtype))

    def tolist(self) -> list[float]:
        return list(self.data)

    def __mul__(self, other) -> "_StubTensor":
        return _StubTensor([x * float(other) for x in self.data], self.dtype)

    __rmul__ = __mul__

    def __add__(self, other) -> "_StubTensor":
        return _StubTensor([a + b for a, b in zip(self.data, other.data)], self.dtype)

    def __truediv__(self, other) -> "_StubTensor":
        return _StubTensor([x / float(other) for x in self.data], self.dtype)

    def __eq__(self, other) -> bool:
        return isinstance(other, _StubTensor) and self.data == other.data

    def __len__(self) -> int:
        return len(self.data)


def _install_torch_stub() -> bool:
    """本机没有 torch 时注册一个最小 stub ⇒ ``--selftest`` 也能跑（返回是否装了 stub）。"""
    try:
        import torch  # noqa: PLC0415,F401

        return False
    except Exception:  # noqa: BLE001
        pass
    import types  # noqa: PLC0415

    stub = types.ModuleType("torch")

    def _save(obj, path, *_a, **_k) -> None:
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def _load(path, *_a, **_k):
        with open(path, "rb") as f:
            return pickle.load(f)

    stub.save = _save
    stub.load = _load
    stub.is_tensor = lambda x: isinstance(x, _StubTensor)
    stub.tensor = lambda data, dtype="float32": _StubTensor(data, dtype)
    stub.zeros = lambda n, dtype="float32": _StubTensor([0.0] * n, dtype)
    stub.ones = lambda n, dtype="float32": _StubTensor([1.0] * n, dtype)
    stub.float32 = "float32"
    stub.int64 = "int64"
    sys.modules["torch"] = stub
    return True


def _int64_dtype() -> Any:
    """int64 dtype（真 torch / stub 两种世界都能用）。"""
    import torch  # noqa: PLC0415

    return getattr(torch, "int64", "int64")


def _write_fake_ckpt(path: str, vals: list[float], num: int, head_level: str = "p3") -> None:
    """写一个**结构自描述**的假 ckpt（average_checkpoints 只看结构键 + model）。"""
    import torch  # noqa: PLC0415

    ck = {
        "no_p2": 1,
        "head_level": head_level,
        "act": "silu",
        "head_conv": 64,
        "late_upsample": 1,
        "stride": 2,
        "epoch": 5,
        "model": {"w": torch.tensor(vals), "n": torch.tensor([num]).to(_int64_dtype())},
    }
    torch.save(ck, path)


def run_selftest() -> None:
    """CPU 自检：假 ckpt → 平均 → 结构拒绝 → 候选生成 → triad/差值。"""
    import tempfile  # noqa: PLC0415

    stubbed = _install_torch_stub()
    import torch  # noqa: PLC0415

    average_checkpoints, _RA, _ns = import_helpers()
    tmp = tempfile.mkdtemp(prefix="fine_swa_selftest_")
    try:
        # (a) 两个同构假 ckpt
        a = os.path.join(tmp, "a.pt")
        b = os.path.join(tmp, "b.pt")
        _write_fake_ckpt(a, [1.0, 2.0, 3.0], 7)
        _write_fake_ckpt(b, [3.0, 4.0, 5.0], 7)

        # (b) 平均 ⇒ 等于均值（整数张量取第一个，epoch 被写成 5）
        out = os.path.join(tmp, "avg.pt")
        average_checkpoints([a, b], out, weights=[0.5, 0.5], epoch=5, note="selftest")
        avg = torch.load(out, map_location="cpu")
        got = [float(x) for x in avg["model"]["w"].tolist()]
        for g, exp in zip(got, [2.0, 3.0, 4.0]):
            assert abs(g - exp) < 1e-6, f"均值不对：{got}"
        assert [float(x) for x in avg["model"]["n"].tolist()] == [7.0], "整数张量应取第一个"
        assert avg.get("epoch") == 5, "epoch 应写成 5"

        # (c) 结构不一致的 ckpt 对 ⇒ 必须被拒绝
        c = os.path.join(tmp, "c.pt")
        _write_fake_ckpt(c, [1.0, 2.0, 3.0], 7, head_level="p2")
        try:
            average_checkpoints([a, c], os.path.join(tmp, "bad.pt"))
        except ValueError:
            pass
        else:
            raise AssertionError("结构不一致的 ckpt 对必须被拒绝")
        try:
            average_checkpoints([a], os.path.join(tmp, "one.pt"))
        except ValueError:
            pass
        else:
            raise AssertionError("少于 2 个 ckpt 必须被拒绝")

        # (d) 候选生成：计数 / 顺序 / 去重 / 单 epoch 跳过
        cands = generate_candidates([3, 2, 4])            # 乱序输入 ⇒ 内部排序
        assert len(cands) == 8, f"3 epoch 应有 8 个候选，得到 {len(cands)}"
        assert [c["members"] for c in cands] == [
            [2, 3], [2, 4], [3, 4],          # 等权对
            [2, 3], [2, 3], [3, 4], [3, 4],  # 相邻对 × [0.3,0.7] / [0.7,0.3]
            [2, 3, 4],                      # 全部等权
        ], f"候选顺序不对：{cands}"
        assert cands[0]["weights"] == [0.5, 0.5]
        assert cands[3]["weights"] == [0.3, 0.7] and cands[4]["weights"] == [0.7, 0.3]
        assert len(generate_candidates([2, 3])) == 3, "2 epoch 应有 3 个候选"
        assert len(generate_candidates([5])) == 0, "单 epoch 不该有候选"
        assert len(generate_candidates([2, 2, 3])) == 3, "重复 epoch 应先去重"

        # (d2) 近失臂筛选 + 全局排序（临时假盘 ⇒ 不碰 /mnt/data）
        root = os.path.join(tmp, "runs")

        def _mk_arm(trial: int, eps: list[int]) -> None:
            d = os.path.join(root, "trials", f"trial_{trial}", "weights")
            os.makedirs(d, exist_ok=True)
            for e in eps:
                with open(os.path.join(d, f"epoch{e}.pt"), "wb"):
                    pass

        for t, e in ((111, [2, 3]), (222, [2, 3, 4]), (333, [2, 3])):
            _mk_arm(t, e)
        led = {"arms": {
            "111": {"trial": 111, "nanobench": {
                "2": {"f1": 0.89, "nano": 0.700, "nano4": 0.645},
                "3": {"f1": 0.89, "nano": 0.705, "nano4": 0.646}}},
            "222": {"trial": 222, "nanobench": {
                "2": {"f1": 0.89, "nano": 0.720, "nano4": 0.660},
                "3": {"f1": 0.90, "nano": 0.710, "nano4": 0.650},
                "4": {"f1": 0.89, "nano": 0.690, "nano4": 0.640}}},
            "333": {"trial": 333, "nanobench": {
                "2": {"f1": 0.50, "nano": 0.200, "nano4": 0.100},
                "3": {"f1": 0.50, "nano": 0.210, "nano4": 0.110}}},
        }}
        arms = select_arms(led, root, BASE)
        assert [x["trial"] for x in arms] == [222, 111], f"臂排序不对：{arms}"
        assert arms[0]["best_epoch"] == 2 and abs(arms[0]["best_nano"] - 0.72) < 1e-9
        assert len(plan_candidates(arms)) == 11, "222(8) + 111(3) = 11 个候选"

        # (e) triad / 差值
        m = {"nano": BASE["nano"], "nano4": BASE["nano4"], "f1": BASE["f1"]}
        assert is_triad(m), "恰好等于基线应算 triad"
        d = compute_deltas(m)
        assert abs(d["d_nano"]) < 1e-12 and abs(d["d_nano4"]) < 1e-12 and abs(d["d_f1"]) < 1e-12
        m2 = {"nano": BASE["nano"] - 1e-6, "nano4": BASE["nano4"], "f1": BASE["f1"]}
        assert not is_triad(m2), "差 1e-6 就该判负"
        assert compute_deltas(m2)["d_nano"] < 0

        # (f) 合成臂 id 确定性 + 写盘护栏
        sid1 = alloc_synth_id(222, [2, 3], [0.5, 0.5], root, probe=False)
        sid2 = alloc_synth_id(222, [2, 3], [0.5, 0.5], root, probe=False)
        assert sid1 == sid2, "合成臂 id 必须稳定"
        assert sid1 != alloc_synth_id(222, [2, 4], [0.5, 0.5], root, probe=False)
        other = os.path.join(root, "trials", f"trial_{sid1}")
        os.makedirs(os.path.join(other, "weights"), exist_ok=True)
        assert not can_write_synth(sid1, root), "别人的目录必须被护栏挡住"
        with open(sidecar_path(sid1, root), "w", encoding="utf-8") as f:
            json.dump({"marker": MARKER, "key": "x"}, f)
        assert can_write_synth(sid1, root), "有本脚本标记的目录应可写"

        print(f"[selftest] torch={'stub' if stubbed else 'real'} | tmp={tmp}")
        print("[selftest] 假 ckpt 平均 / 结构拒绝 / 候选生成 / triad 差值 / id 幂等 全部通过")
        print("SELFTEST PASS")
    finally:
        import shutil  # noqa: PLC0415

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
