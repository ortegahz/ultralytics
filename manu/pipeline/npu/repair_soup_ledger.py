#!/usr/bin/env python3
# 🟢🟢 2026-11-20 18:05 新增：修复「并发写竞争吃掉」的汤臂 picked 读数
#
# 🔴 根因（已实测定位）：
#   `nano_search._merge_disk_ledger` 的 arms 合并是 **整臂覆盖**（`merged.update(m_arms)`）
#   ⇒ **内存快照陈旧的一臂会把别的进程刚写进去的真读数整臂覆盖掉**（典型 lost update）。
#   实测：`save_ledger` 明明「读-合并-写」整体加了 flock，但**合并语义是「内存覆盖磁盘」**
#   ⇒ 训练轴（长驻、内存快照老）会把汤批次/门禁通道刚写的 `picked` 覆盖成 `None`。
#   ⇒ **磁盘上的 sidecar 才是权威**（代码注释里也写了：「磁盘上的门禁报告是权威」）。
#
# 🔴 实测影响面（18:05 快照）：
#   全台账 **3911 臂 `picked` 缺失**；其中 **518 臂在磁盘有 sidecar**（含真读数）⇒ **可完整恢复**；
#   SB4 段（≥2109000）当时已丢 4 臂（2109001/2109002/2109012/2109504）。
#   ⇒ 周一审计若只看台账 ⇒ **这 518 臂里若有六项全过候选，会被整体漏掉**。
#
# 🟢 本脚本的安全设计（不制造新的写竞争）：
#   ① **全程持有 `nano_search` 用的同一把跨进程 flock**（`$WORK/ledger.lock`）
#      ⇒ 与 `save_ledger` 的「读-合并-写」事务**互斥** ⇒ 修复期间训练轴/门禁无法并发写台账。
#   ② **持锁后重新读盘**（不复用任何外部快照）⇒ 看到的是最新台账。
#   ③ **只补「picked 缺失」的臂，已有值一律不动**（绝不覆盖别人的新读数）。
#   ④ 写盘用 **临时文件 + os.replace**（同目录 ⇒ 原子替换）。
#   ⑤ **幂等**：已修好的臂不再命中；`--dry-run` 可先看清单再落盘。
#
# 用法：
#   python manu/pipeline/npu/repair_soup_ledger.py --dry-run     # 只列清单
#   python manu/pipeline/npu/repair_soup_ledger.py                # 实际修复
#   python manu/pipeline/npu/repair_soup_ledger.py --min-sid 2109000   # 只修某段以后
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time

WORK = "/mnt/data/siping/runs/npu20ms/nano_search"
LEDGER = os.path.join(WORK, "nano_search_ledger.json")
LOCKFILE = os.path.join(WORK, "ledger.lock")
TRIALS = "/mnt/data/siping/runs/npu20ms_offcal/trials"
SIDE_CAR = "basin_soup_synth.json"
SIDECAR_TRIALS = os.path.join(TRIALS)


def _sidecar_path(sid: int) -> str:
    return os.path.join(SIDECAR_TRIALS, f"trial_{sid}", "weights", SIDE_CAR)


def _read_sidecar(sid: int) -> dict | None:
    p = _sidecar_path(sid)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def _needs_repair(arm: dict) -> bool:
    """`picked` 缺失（None / 非 dict）或读数不完整（nano 为空）⇒ 需要补。

    🔴 2026-11-23 修正（老口径漏修 557 臂）：
      老口径还要求「nanobench 也缺」才算要修 ⇒ **把「台账 nanobench 有完整读数、
      但 picked 是 null」的臂全挡住了** —— 而那正是被并发写吃掉读数的那一批。
      现在只要 `picked` 缺读数就命中 ⇒ 读数源优先用**台账自己的 nanobench**。
    """
    pk = arm.get("picked")
    if not isinstance(pk, dict):
        return True
    return pk.get("nano") is None


def _ledger_nb(arm: dict) -> tuple[dict | None, str | None]:
    """台账里自带的 nanobench 读数（优先 picked epoch）。

    比磁盘 sidecar **更全**：含 micro / small / regular / micro4 / f1_4 / cache。
    返回 (读数 dict, epoch_key)。
    """
    nb = arm.get("nanobench")
    if not isinstance(nb, dict):
        return None, None
    cands = {str(k): v for k, v in nb.items() if isinstance(v, dict) and v.get("nano") is not None}
    if not cands:
        return None, None
    pe = arm.get("picked_epoch")
    if pe is not None and str(pe) in cands:
        return cands[str(pe)], str(pe)

    def _num(k: str) -> float:
        try:
            return float(k)
        except (TypeError, ValueError):
            return -1.0

    ek = max(cands, key=_num)
    return cands[ek], ek


def _patch(arm: dict, nb: dict, epoch_key: str, from_ledger: bool, sc: dict | None) -> bool:
    """把权威读数补进臂；返回是否真的补了东西。

    `nb`    —— 权威读数（台账 nanobench 优先，其次磁盘 sidecar）
    `from_ledger` —— True 表示读数本来就来自台账 nanobench（写回是幂等 no-op）
    """
    nano, nano4, f1 = nb.get("nano"), nb.get("nano4"), nb.get("f1")
    if nano is None:
        return False
    try:
        ep_int = int(float(epoch_key))
    except (TypeError, ValueError):
        ep_int = arm.get("picked_epoch") or 5
    picked = {
        "nano": nano, "nano4": nano4, "f1": f1,
        "micro": nb.get("micro"), "small": nb.get("small"),
        "regular": nb.get("regular"), "micro4": nb.get("micro4"),
        "f1_4": nb.get("f1_4"),
    }
    for opt in ("cache", "th", "fp", "fn"):
        if nb.get(opt) is not None:
            picked[opt] = nb[opt]
    picked["epoch"] = ep_int
    picked["epoch_key"] = str(epoch_key)
    ckpt = (arm.get("params") or {}).get("ckpt")
    if ckpt:
        picked["ckpt"] = ckpt
    arm["picked"] = picked
    arm["picked_epoch"] = arm.get("picked_epoch") or ep_int
    if not from_ledger:
        arm.setdefault("nanobench", {})[str(epoch_key)] = dict(nb)
    if sc:
        if sc.get("members") and not arm.get("soup_members"):
            arm["soup_members"] = sc["members"]
        if sc.get("weights") and not arm.get("soup_weights"):
            arm["soup_weights"] = sc["weights"]
        if sc.get("family") and not arm.get("soup_family"):
            arm["soup_family"] = sc["family"]
    arm.setdefault("synth", True)
    arm.setdefault("channel", "SB")
    arm["repaired_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    arm["repaired_from"] = "ledger_nanobench" if from_ledger else "sidecar"
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-sid", type=int, default=0, help="只修 arm 号 >= 该值的臂")
    ap.add_argument("--max-sid", type=int, default=0, help="只修 arm 号 <= 该值的臂（0=不限）")
    args = ap.parse_args()

    with open(LOCKFILE, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(LEDGER, "r", encoding="utf-8") as fh:
                led = json.load(fh)
            arms = led.get("arms") or {}
            targets: list[tuple[int, dict | None, str, bool]] = []
            sidecar_only = 0
            for k, v in arms.items():
                try:
                    sid = int(k)
                except Exception:  # noqa: BLE001
                    continue
                if sid < args.min_sid:
                    continue
                if args.max_sid and sid > args.max_sid:
                    continue
                if not isinstance(v, dict) or not _needs_repair(v):
                    continue
                nb, ek = _ledger_nb(v)                 # ① 台账自带读数优先
                if nb is not None:
                    targets.append((sid, nb, ek, True))
                    continue
                sc = _read_sidecar(sid)                # ② sidecar 兜底
                if not sc:
                    continue
                snb = sc.get("nanobench") or (sc.get("entry") or {}).get("nanobench") or {}
                if snb.get("nano") is None:
                    continue
                targets.append((sid, snb, str(v.get("picked_epoch") or 5), False))
                sidecar_only += 1
            targets.sort()
            print(f"[修复] 命中待修臂 {len(targets)} 个"
                  f"（台账自带读数 {len(targets) - sidecar_only} ｜ sidecar 兜底 {sidecar_only}）"
                  f"{'（dry-run，不落盘）' if args.dry_run else ''}")
            for sid, nb, ek, _fl in targets[:15]:
                print(f"   sid={sid} ep={ek} nano={nb.get('nano')} "
                      f"nano4={nb.get('nano4')} f1={nb.get('f1')}")
            if len(targets) > 15:
                print(f"   … 另有 {len(targets) - 15} 个")
            if args.dry_run or not targets:
                return 0
            fixed = 0
            for sid, nb, ek, from_ledger in targets:
                arm = arms.get(str(sid))
                if arm is None or not _needs_repair(arm):
                    continue
                sc = None if from_ledger else _read_sidecar(sid)
                if _patch(arm, nb, ek, from_ledger, sc):
                    fixed += 1
            tmp = f"{LEDGER}.tmp.repair.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(led, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, LEDGER)
            print(f"[修复] 已补 {fixed} 个臂的 picked/nanobench ⇒ {LEDGER}")
            return 0
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


if __name__ == "__main__":
    sys.exit(main())
