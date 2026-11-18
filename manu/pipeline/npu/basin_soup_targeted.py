
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""定向盆地权重汤扫描（2026-11-13 18:20 新增）—— 只冲「官方轨最强盆地」。

🟢 为什么必须单独写（与 `family_soup.py` 的分工）：
  `family_soup.py` 的簇排序偏向「簇内 F1 最干净」，`--limit` 又是**全局**预算
  ⇒ 实测 2026-11-13 17:23：8 个预算被 5 个簇吃光，而其中最强簇
  {96630,96565,12844,96719} 的**成员上限** T1 0.887946 < 基线 0.888086、
  T3 0.948343 < 基线 0.948489 ⇒ **该盆地任何成员组合都不可能六项全过**（死路）。
  本脚本反过来：**先锁定官方轨最强的盆地（12679/12519 盆地）**，再枚举它的成员组合。

🟢🟢 该盆地的互补性（台账实测，2026-11-13 18:10 复算）：
    12679@3  : nano .718809 nano4 .655718 F1 .890329 | T1 .890550 ✓ | T3 .948332 ✗(−1.6e-4)
    12519@3  : nano .715974 nano4 .657372 F1 .890534 | T1 .890586 ✓ | T3 .948374 ✗(−1.2e-4)
    96835@5  : nano .718336 nano4 .652647 F1 .889977 | T1 .889552 ✓ | T3 .948343 ✗
    1000003@1: nano .713138 nano4 .648629 F1 .889770 | T1 .889758 ✓ | T3 .948637 ✓ ⇒ 六项全过
    999821@1 : nano .711011 nano4 .651938 F1 .889831 | T1 .889851 ✓ | T3 .948610 ✓ ⇒ 六项全过
    12830@5  : T3 .948827（盆地最高）但 nano .707467 ✗、F1 .889624 ✗
  ⇒ **高 nano 的臂缺 T3、高 T3 的臂缺 nano/F1** ⇒ **同盆地等权平均**可同时补齐
  ⇒ 跨盆地平均会把模型平均坏（见 memory §176：head cos≈0.02 ⇒ 头被重初始化）。

复现纪律（升轨门槛「同家族 n≥4」的落地方式）：
  · **rep 家族** = 从同一成员池用**固定随机种子**抽 k 个成员做等权汤
    ⇒ 每个 seed 是一次独立抽样；4 个 seed 全部六项全过 ⇒「同盆地汤」这个**方法**可复现
    ⇒ 台账里写 `soup_family = bs<basin>:eq<k>` ⇒ auto_promote 按
      「同 soup_family ≥4 个六项全过 + 其中有板卡 T4 PASS」自动升轨。
  · 每个汤都**独立跑官方三轨门禁**（不是拿母臂读数顶替）⇒ 每次抽样是一次完整测量。

用法（服务器）：
    python manu/pipeline/npu/basin_soup_targeted.py --gpu 0 --rep-seeds 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
if os.path.isdir(REPO):
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

LEDGER = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"
CACHE_DIR = "/mnt/data/siping/runs/npu20ms/tiny_ref"
SIDE_CAR = "basin_soup_synth.json"
SYNTH_BASE = 98900          # ⚠️ 2026-11-16 10:35：原 69900 段与**调度器臂号**（69916+）撞号
                          #    ⇒ 挪到保留段 98900（family_soup 69800 / cross_soup 97900 / fine_swa 97000 之外）
COS_MIN = 0.90
BASE = {"nano": 0.708648393194707, "nano4": 0.6472117202268431, "f1": 0.8897117275427806}


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def synth_dir(sid: int) -> str:
    return os.path.join(ROOT, "trials", f"trial_{int(sid)}")


def sidecar_path(sid: int) -> str:
    return os.path.join(synth_dir(sid), "weights", SIDE_CAR)


def read_sidecar(sid: int):
    p = sidecar_path(sid)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


SYNTH_BASE_OVERRIDE = None


SID_SPAN = 2000          # 🟢 2026-11-16：100 → 2000（多轮换种子扫描要用）
_SID_INDEX: dict = None
_SID_USED: set = set()


def _load_sid_index(base: int) -> None:
    """一次性扫 base..base+SID_SPAN，建 sid→key 索引 + 已占用集合。

    🟢🟢 2026-11-16 11:25 修根因：原来**每分配一个 id 就重扫 100 个 sidecar**，
      且 id 空间只有 100 ⇒ 多轮扫描（换种子）必然撞「合成 id 空间耗尽」。
      现在：一次建索引 + 2000 空间 ⇒ 每轮用独立 base 段（见 soup_k5_loop2.sh）。
    """
    global _SID_INDEX, _SID_USED
    if _SID_INDEX is not None:
        return
    _SID_INDEX = {}
    _SID_USED = set()
    for sid in range(base, base + SID_SPAN):
        sc = read_sidecar(sid)
        if sc:
            _SID_INDEX[sid] = str(sc.get("key") or "")
            _SID_USED.add(sid)
        elif os.path.isdir(synth_dir(sid)):
            _SID_USED.add(sid)


def alloc_synth_id(key: str) -> int:
    base = SYNTH_BASE_OVERRIDE or SYNTH_BASE
    _load_sid_index(base)
    for sid, k in _SID_INDEX.items():
        if k == key:
            return sid
    for sid in range(base, base + SID_SPAN):
        if sid not in _SID_USED:
            _SID_USED.add(sid)
            _SID_INDEX[sid] = key
            return sid
    raise RuntimeError("合成 id 空间耗尽")


def arm_ckpt(trial: int, epoch: int) -> str:
    return os.path.join(ROOT, "trials", f"trial_{int(trial)}", "weights", f"epoch{int(epoch)}.pt")


def head_vec(path: str):
    import torch  # noqa: PLC0415
    ck = torch.load(path, map_location="cpu")
    m = ck.get("model") or ck
    sd = m.state_dict() if hasattr(m, "state_dict") else m
    for k in sd:
        if k.endswith("head_in.conv.weight"):
            return sd[k].float().flatten()
    return None


def cos(a, b) -> float:
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def basin_members(led: dict, ref: tuple[int, int], min_nano: float) -> list[dict]:
    """盆地成员 = 与 ref 的 head_in.conv.weight 余弦 ≥ COS_MIN 的臂（且权重在盘）。"""
    ref_path = arm_ckpt(*ref)
    rv = head_vec(ref_path)
    out = []
    for t, arm in (led.get("arms") or {}).items():
        p = arm.get("picked") or {}
        n = p.get("nano")
        if n is None or n < min_nano:
            continue
        ep = arm.get("picked_epoch")
        if ep is None:
            continue
        path = arm_ckpt(int(t), int(ep))
        if not os.path.isfile(path):
            continue
        try:
            v = head_vec(path)
        except Exception:  # noqa: BLE001
            continue
        if v is None or rv is None:
            continue
        if cos(rv, v) < COS_MIN:
            continue
        off = arm.get("official") or {}
        out.append({"trial": int(t), "epoch": int(ep), "ckpt": path,
                    "nano": float(n), "nano4": float(p.get("nano4") or 0.0), "f1": float(p.get("f1") or 0.0),
                    "t1": off.get("t1"), "t3": off.get("t3"), "t2_fp": off.get("t2_fp"),
                    # 🟢 2026-11-16 12:00：带上「这个成员本身是不是汤」的信息
                    #   （98900/99000/99100 是**同一个汤**的三份拷贝 ⇒ 抽样池必须去重，
                    #     否则同一份权重会被抽进同一个汤 ⇒ 等价于给那个成员双倍权重）
                    "soup_members": arm.get("soup_members"), "soup_weights": arm.get("soup_weights")})
    out.sort(key=lambda c: -c["nano"])
    return out


def build_combos(members: list[dict], base_t3: float, rep_seeds: int, rep_k: list[int],
                 basin: int = 0, only_rep: bool = False, seed_offset: int = 0,
                 pool_nano_min: float = 0.0, pool_t3_min: float = 0.0,
                 mix_hi_t3: bool = False, mix_hi_t1: bool = False,
                 curated: bool = False, pool_t1_min: float = 0.0,
                 set_t3_min: float = 0.0, set_nano_min: float = 0.0,
                 set_t1_min: float = 0.0, set_f1_min: float = 0.0,
                 set_nano4_min: float = 0.0) -> list[dict]:
    """核心池 = F1 过线成员；extras = T3 过线但 F1 不足的成员（用来补 T3）。"""
    core = [m for m in members if m["f1"] >= BASE["f1"]]
    extras = [m for m in members if m["f1"] < BASE["f1"]
              and m["t3"] is not None and float(m["t3"]) >= base_t3]
    combos: list[dict] = []
    seen: set[tuple] = set()

    def add(mem: list[dict], tag: str, fam: str, weights=None) -> None:
        sig = tuple(sorted((m["trial"], m["epoch"]) for m in mem))
        if sig in seen or len(mem) < 2:
            return
        seen.add(sig)
        combos.append({"members": mem, "weights": weights or [1.0 / len(mem)] * len(mem),
                       "tag": tag, "family": fam, "k": len(mem)})

    # ① top-k 等权（确定性）—— only_rep 时跳过（多实例并行时避免重复算同一批）
    if not only_rep:
        for k in range(2, len(core) + 1):
            add(core[:k], f"eq{k}", f"bs{basin or eq_k_family(core)}:eq")
    # ② rep 家族：固定种子随机抽 k 个（⇒ 独立抽样，可做 n≥4 复现）
    # 🟢🟢 2026-11-16 11:50 **定向抽样**（升轨 v2.1 之后必须）：
    #   新基线 69913 把 T3 抬到 0.948526，而**高 nano 的盆地臂 T3 低、高 T3 的臂 nano 低**
    #   （实测：69901 nano 0.720463 / T3 0.948322；98905 T3 0.948572 / nano 0.711720）
    #   ⇒ 从全池随机抽 ⇒ 绝大多数组合的 T3 均值天然在红线下 ⇒ 白烧 13 min/个门禁。
    #   现在：**只从「nano 过新基线 + T3 过新基线−δ」的成员池里抽**，
    #   汤的 T3 ≈ 成员 T3 均值 ±0.00016 ⇒ 命中率提高一个量级。
    pool = [m for m in core
            if (m.get("nano") or 0) >= pool_nano_min
            and (m.get("t3") is None or float(m["t3"]) >= pool_t3_min)]
    if pool_nano_min or pool_t3_min:
        log(f"[定向] 成员池 {len(core)} → {len(pool)} 个（nano≥{pool_nano_min} 且 T3≥{pool_t3_min}）")
    # 🟢🟢🟢 2026-11-16 11:50 **混合双组抽样**（冲 Baseline_v2.2 的关键配方）
    #   观察：新基线 69913 = [96835, 12519@3, **1000003**, **999821**, 97712]
    #     = 「2 个高 nano 盆地臂 + 2 个高 T3 **抛光臂** + 1 个弱臂」
    #     抛光臂 T3 最高（1000003=0.948637 / 999821=0.948610）但 nano 低（0.713/0.711）、
    #     盆地臂 nano 最高（0.718~0.720）但 T3 低（0.94832~0.94843）
    #     ⇒ **纯 nano 池 ⇒ T3 不够；纯 T3 池 ⇒ nano 不够；必须两样都有。**
    #   A 组 = 高 nano（≥pool_nano_min）；B 组 = 高 T3（≥pool_t3_min，来自**全体盆地成员**，
    #   不受 F1 核心池限制 —— 抛光臂 F1 略低但 T3 是它的长项）。
    #   每个汤固定抽 k//2 个 A + 余下 B 个 B。
    def _wsample(rng, pop, weights, k):
        """按权重**无放回**抽样（权重递减 ⇒ 偏向列表前部）。"""
        pop, w, out = list(pop), list(weights), []
        for _ in range(k):
            tot = sum(w)
            if tot <= 0 or not pop:
                break
            r = rng.random() * tot
            acc = 0.0
            for i, x in enumerate(w):
                acc += x
                if r <= acc:
                    out.append(pop[i]); pop.pop(i); w.pop(i)
                    break
        return out

    def _msig(m: dict) -> tuple:
        sm = m.get("soup_members")
        if sm:
            return ("soup", tuple(sorted((int(a), int(b)) for a, b in sm)))
        return ("arm", int(m["trial"]), int(m["epoch"]))

    def _dedup(seq: list[dict]) -> list[dict]:
        seen: set = set()
        out = []
        for m in seq:
            s = _msig(m)
            if s in seen:
                continue
            seen.add(s)
            out.append(m)
        return out

    pool_a = _dedup([m for m in core if (m.get("nano") or 0) >= pool_nano_min]) if pool_nano_min else _dedup(core)
    pool_b = _dedup([m for m in members
                     if m.get("t3") is not None and float(m["t3"]) >= pool_t3_min
                     and (pool_nano_min == 0 or (m.get("nano") or 0) < pool_nano_min)])
    if mix_hi_t3:
        log(f"[双组] A组(高nano) {len(pool_a)} 个 / B组(高T3) {len(pool_b)} 个")
    # 🟢🟢🟢 2026-11-16 13:20 **策展模式（curated）**：不再随机抽，而是**枚举**组合并按
    #   「集合均值同时守住 nano / nano4 / F1 / T1 / T3」过滤 ⇒ 只把**大概率过六项**的汤送门禁。
    #   依据（13:00~13:20 实测）：
    #     · 汤的 nano ≈ 成员 nano 均值 + 0.002~0.005；T1 ≈ 成员均值 ±0.0002；T3 ≈ 成员均值 ±0.0001
    #     · 盆地成员在 (nano,T1) 与 (T1,T3) 两轴上互斥 ⇒ 随机抽几乎必然踩到某一轴
    #   ⇒ 先筛「个体就强」的池（T1≥阈值 且 T3≥阈值），再枚举 k 元组，按集合均值过线。
    if curated:
        import itertools
        pool_p = _dedup([m for m in members
                         if (m.get("t3") is not None and float(m["t3"]) >= pool_t3_min)
                         and (m.get("t1") is not None and float(m["t1"]) >= pool_t1_min)])
        log(f"[策展] 优质成员池 {len(pool_p)} 个（T1≥{pool_t1_min} 且 T3≥{pool_t3_min}）")
        for k in rep_k:
            kept = []
            for comb in itertools.combinations(pool_p, k):
                nn = sum(m["nano"] for m in comb) / k
                n4 = sum(m["nano4"] for m in comb) / k
                f1 = sum(m["f1"] for m in comb) / k
                t1 = sum(float(m["t1"]) for m in comb) / k
                t3 = sum(float(m["t3"]) for m in comb) / k
                if nn < set_nano_min or n4 < set_nano4_min or f1 < set_f1_min:
                    continue
                if t1 < set_t1_min or t3 < set_t3_min:
                    continue
                kept.append((comb, nn, n4, f1, t1, t3))
            # 按「最弱轴的余量」排序 ⇒ 优先送最有把握的
            kept.sort(key=lambda x: -min(x[4] - set_t1_min, x[5] - set_t3_min,
                                          x[1] - set_nano_min, x[2] - set_nano4_min))
            log(f"[策展] k={k}: 通过集合质量过滤 {len(kept)} 个")
            for i, (comb, nn, n4, f1, t1, t3) in enumerate(kept[: max(rep_seeds, 1)], 1):
                mem = sorted(list(comb), key=lambda m: -m["nano"])
                add(mem, f"cur{k}s{i}", f"bs{basin or eq_k_family(core)}:eq")
            if not kept:
                log(f"[策展] k={k}: 无组合满足集合门槛 ⇒ 放宽一轮")
                for comb in itertools.combinations(pool_p, k):
                    mem = sorted(list(comb), key=lambda m: -m["nano"])
                    add(mem, f"cur{k}sx{len(comb)}", f"bs{basin or eq_k_family(core)}:eq")
                    if len(combos) > 400:
                        break
        return combos
    for k in rep_k:
        for s in range(1, rep_seeds + 1):
            rng = random.Random(seed_offset + 1000 + k * 100 + s)
            if mix_hi_t3 and pool_b:
                na = max(1, k // 2)
                nb = k - na
                if mix_hi_t1:
                    # 🟢 2026-11-16 12:10：A 组（高 nano）内部**按 T1 降序加权抽样**
                    #   （升轨 v2.1 后卡点已从 T3 转到 **T1 ≥ 0.890013**：
                    #    高 T3 臂的 T1 偏低（1011000 T1 0.8894）⇒ A 组必须尽量取高 T1 的）
                    a_sorted = sorted(pool_a, key=lambda m: -(m.get("t1") or 0))
                    wts = [2.718281828 ** (-i / 2.5) for i in range(len(a_sorted))]
                    pick_a = _wsample(rng, a_sorted, wts, min(na, len(a_sorted)))
                else:
                    pick_a = rng.sample(pool_a, min(na, len(pool_a)))
                rest_b = [m for m in pool_b if m not in pick_a]
                if mix_hi_t1 and rest_b:
                    # 🟢 2026-11-16 12:35：**B 组也按 T1 降序加权**
                    #   （B 组里 1010720 T1=0.88708 / 1010721 T1=0.887735 是「T3 极高但 T1 极低」的坑，
                    #     均匀抽到它们 ⇒ 整个汤的 T1 直接塌 ⇒ 六项全过变五项）
                    b_sorted = sorted(rest_b, key=lambda m: -(m.get("t1") or 0))
                    bw = [2.718281828 ** (-i / 2.5) for i in range(len(b_sorted))]
                    pick_b = _wsample(rng, b_sorted, bw, min(nb, len(b_sorted)))
                else:
                    pick_b = rng.sample(rest_b, min(nb, len(rest_b))) if rest_b else []
                mem = list(pick_a) + list(pick_b)
                if len(mem) < k:  # 补齐
                    used = {id(x) for x in mem}
                    extra = [m for m in (pool_a + pool_b) if id(m) not in used]
                    mem += rng.sample(extra, min(k - len(mem), len(extra)))
                if len(mem) < 2:
                    continue
            else:
                src_pool = pool if len(pool) >= k else core
                mem = rng.sample(src_pool, k) if k <= len(src_pool) else list(src_pool)
            mem = sorted(mem, key=lambda m: -m["nano"])
            add(mem, f"rep{k}s{s}", f"bs{basin or eq_k_family(core)}:eq")
    # ③ best0.5：最强成员占一半权重
    if len(core) >= 2:
        w = [0.5] + [0.5 / (len(core) - 1)] * (len(core) - 1)
        add(core, "best0.5", f"bs{basin or eq_k_family(core)}:w0.5", w)
    # ④ T3 补强：核心池 top-3/4 + T3-strong 成员
    for extra in extras:
        for k in (3, 4):
            if len(core) >= k:
                add(core[:k] + [extra], f"t3k{k}+{extra['trial']}", f"bs{basin or eq_k_family(core)}:eqT3")
    return combos


def eq_k_family(core: list[dict]) -> int:
    return core[0]["trial"] if core else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ref", default="12679:3", help="盆地参考臂 trial:epoch")
    ap.add_argument("--min-nano", type=float, default=0.705)
    ap.add_argument("--rep-seeds", type=int, default=4)
    ap.add_argument("--rep-k", default="4,5")
    ap.add_argument("--gate-top", type=int, default=4, help="额外（rep 家族之外）要送官方门禁的汤数上限")
    ap.add_argument("--max-eval", type=int, default=40)
    ap.add_argument("--report", default="/mnt/data/siping/runs/npu20ms/nano_search/basin_soup_targeted.json")
    ap.add_argument("--synth-base", type=int, default=0,
                    help="合成臂 id 段起点（并行多实例时各占一段 ⇒ 互不撞号；0=用默认）")
    ap.add_argument("--pool-nano-min", type=float, default=0.0,
                    help="定向抽样：成员 nano 下限（= 新基线 nano 时只抽能守住 nano 的成员）")
    ap.add_argument("--curated", action="store_true",
                    help="策展模式：枚举组合 + 集合均值过线过滤（只把稳过的汤送门禁）")
    ap.add_argument("--pool-t1-min", type=float, default=0.0,
                    help="策展池：成员 T1 下限")
    ap.add_argument("--set-t3-min", type=float, default=0.0, help="集合 T3 均值下限")
    ap.add_argument("--set-nano-min", type=float, default=0.0, help="集合 nano 均值下限")
    ap.add_argument("--set-nano4-min", type=float, default=0.0, help="集合 nano4 均值下限")
    ap.add_argument("--set-f1-min", type=float, default=0.0, help="集合 F1 均值下限")
    ap.add_argument("--set-t1-min", type=float, default=0.0, help="集合 T1 均值下限")
    ap.add_argument("--mix-hi-t1", action="store_true",
                    help="A 组（高 nano）内部按 T1 降序加权抽样（卡点转到 T1 时用）")
    ap.add_argument("--mix-hi-t3", action="store_true",
                    help="双组抽样：每个汤固定一半成员来自高 nano 池、一半来自高 T3 池")
    ap.add_argument("--pool-t3-min", type=float, default=0.0,
                    help="定向抽样：成员 T3 下限（= 新基线 T3 − δ，δ≈0.0002 时命中率最高）")
    ap.add_argument("--only-rep", action="store_true",
                    help="只做 rep 随机抽样（跳过确定性 eq-k ⇒ 多实例并行不重复算）")
    ap.add_argument("--rep-seed-offset", type=int, default=0,
                    help="rep 抽样随机种子偏移（多实例并行时各用不同偏移 ⇒ 抽样不重复）")
    ap.add_argument("--curated-sets", default="",
                    help="策展集合 JSON（list of {members,tag,pred}）⇒ 离线枚举好的高胜率汤")
    ap.add_argument("--curated-offset", type=int, default=0,
                    help="策展集合起始下标（多轮循环时推进）")
    ap.add_argument("--curated-limit", type=int, default=12,
                    help="本轮策展集合个数（= 门禁预算）")
    args = ap.parse_args()
    if args.synth_base:
        global SYNTH_BASE_OVERRIDE
        SYNTH_BASE_OVERRIDE = int(args.synth_base)
        log(f"[id段] 合成臂 id 段起点 = {SYNTH_BASE_OVERRIDE}")

    from manu.pipeline.npu.ckpt_synth import average_checkpoints  # noqa: PLC0415
    from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415
    from manu.pipeline.npu import nano_search as ns  # noqa: PLC0415

    rt, rep = args.ref.split(":")
    ref = (int(rt), int(rep))
    with open(LEDGER, encoding="utf-8") as fh:
        led = json.load(fh)
    base = led.get("baseline_v2") or {}
    bt1 = float(base.get("t1") or 0.0)
    bt2fp = int(base.get("t2_fp") or 5)
    bt3 = float(base.get("t3") or 0.0)
    # 🟢🟢🟢 2026-11-16 12:55：**BASE 必须跟台账走**（原来是这个文件里写死的旧基线常量）
    #   后果：升轨 v2.1 后 nano 门槛已抬到 0.715737，但 triad 判定仍按 0.708648
    #   ⇒ nano 只有 0.711 的候选照样送官方门禁 ⇒ **每个白烧 13 min**（T1/T2fp/T3 用的是新口径，
    #   只有 nano/nano4/F1 用旧口径 ⇒ 口径分裂 ⇒ 假阳性候选）。
    global BASE
    _barm = (led.get("arms") or {}).get(str(base.get("trial"))) or {}
    _bpk = _barm.get("picked") or {}
    BASE = {
        "nano": float(_bpk.get("nano") if _bpk.get("nano") is not None else (base.get("nano") or BASE["nano"])),
        "nano4": float(_bpk.get("nano4") if _bpk.get("nano4") is not None else (base.get("nano4") or BASE["nano4"])),
        "f1": float(_bpk.get("f1") if _bpk.get("f1") is not None else (base.get("f1") or BASE["f1"])),
    }
    log(f"[基线] nano={BASE['nano']:.6f} nano4={BASE['nano4']:.6f} F1={BASE['f1']:.6f} "
        f"T1={bt1:.6f} T2fp={bt2fp} T3={bt3:.6f}｜参考盆地臂 {ref}")

    core = []
    if args.curated_sets:
        with open(args.curated_sets, encoding="utf-8") as fh:
            raw = json.load(fh)
        items = raw.get("sets") if isinstance(raw, dict) else raw
        off = max(0, int(args.curated_offset))
        lim = max(1, int(args.curated_limit))
        items = items[off:off + lim]
        combos = []
        for it in items:
            mem = []
            ok = True
            for pair in (it.get("members") or []):
                tt, ee = int(pair[0]), int(pair[1])
                pa = arm_ckpt(tt, ee)
                if not os.path.isfile(pa):
                    ok = False
                    break
                mem.append({"trial": tt, "epoch": ee, "ckpt": pa})
            if not ok or len(mem) < 2:
                log(f"[策展] 跳过（权重缺失）：{it.get('tag')}")
                continue
            combos.append({"members": mem, "weights": [1.0 / len(mem)] * len(mem),
                           "tag": it.get("tag") or "repcur",
                           "family": f"bs{ref[0]}:eq", "k": len(mem)})
        log(f"[策展] {len(combos)} 个集合（{args.curated_sets} off={off} lim={lim}）")
    else:
        members = basin_members(led, ref, args.min_nano)
        log(f"[盆地] cos≥{COS_MIN} 的成员 {len(members)} 个")
        for m in members:
            log(f"    {m['trial']}@{m['epoch']} nano={m['nano']:.6f} nano4={m['nano4']:.6f} "
                f"F1={m['f1']:.6f} T1={m['t1']} T3={m['t3']}")
        core = [m for m in members if m["f1"] >= BASE["f1"]]
        if len(core) < 2:
            log("[退出] F1 过线成员不足 2 个 ⇒ 无汤可做")
            return
        rep_k = [int(x) for x in args.rep_k.split(",") if x.strip()]
        combos = build_combos(members, bt3, args.rep_seeds, rep_k, basin=ref[0],
                              only_rep=args.only_rep, seed_offset=args.rep_seed_offset,
                              pool_nano_min=args.pool_nano_min, pool_t3_min=args.pool_t3_min,
                              mix_hi_t3=args.mix_hi_t3, mix_hi_t1=args.mix_hi_t1)
        log(f"[计划] {len(combos)} 个汤（核心池 {len(core)} 个）")

    entries = []
    n_gate = 0
    for cand in combos[: args.max_eval]:
        mem = cand["members"]
        members_key = [(m["trial"], m["epoch"]) for m in mem]
        key = "bs|" + "|".join(f"{t}@{e}" for t, e in members_key) + "|" + cand["tag"]
        sid = alloc_synth_id(key)
        sc = read_sidecar(sid)
        if sc and sc.get("scored"):
            log(f"[跳过] 已算过 {key}")
            entries.append(sc.get("entry") or {})
            if sc.get("gate"):
                n_gate += 1
            continue
        out_dir = os.path.join(synth_dir(sid), "weights")
        os.makedirs(out_dir, exist_ok=True)
        out_ck = os.path.join(out_dir, "epoch5.pt")
        try:
            average_checkpoints([m["ckpt"] for m in mem], out_ck,
                                weights=cand["weights"], epoch=5)
        except Exception as exc:  # noqa: BLE001
            log(f"[合成失败] {key}：{exc}")
            continue
        pkl = os.path.join(CACHE_DIR, f"basinsoup_{sid}.pkl")
        if not RA.build_cache(out_ck, pkl, args.gpu, extra="--no-peak-attrs"):
            log(f"[缓存失败] {key}")
            continue
        res = RA.nanobench(pkl, f"basinsoup_{sid}")
        if not res:
            log(f"[评测失败] {key}")
            continue
        rb8 = res["peak_tol8"]["recall_by_bucket"]
        rb4 = res["peak_tol4"]["recall_by_bucket"]
        m = {"nano": float(rb8["nano"]), "nano4": float(rb4["nano"]), "f1": float(res["peak_tol8"]["f1"]),
             "micro": float(rb8["micro"]), "small": float(rb8["small"]), "regular": float(rb8["regular"]),
             "f1_4": float(res["peak_tol4"]["f1"]), "cache": pkl}
        triad = bool(m["nano"] >= BASE["nano"] and m["nano4"] >= BASE["nano4"] and m["f1"] >= BASE["f1"])
        log(f"[评测][{cand['tag']}] 成员={members_key} ⇒ nano={m['nano']:.6f} nano4={m['nano4']:.6f} "
            f"F1={m['f1']:.6f} triad={triad}")

        # 送官方门禁的优先级：rep 家族（可做 n≥4 复现）优先，其次 eq-k（nano 高）
        is_rep = cand["tag"].startswith("rep")
        want_gate = triad and ((is_rep and n_gate < 12) or (not is_rep and n_gate < 12 + args.gate_top))
        gate = None
        if want_gate:
            n_gate += 1
            g = ns.run_official_gate(sid, 5, args.gpu)
            if g and "error" not in g:
                gate = {k: g.get(k) for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")}
                gate["six_ok"] = bool(gate.get("t1") is not None and gate.get("t3") is not None
                                     and gate["t1"] >= bt1 and gate["t2_fp"] <= bt2fp and gate["t3"] >= bt3)
            else:
                gate = {"error": (g or {}).get("error", "no_report")}
            log(f"[门禁] synth={sid} tag={cand['tag']} T1={gate.get('t1')} fp={gate.get('t2_fp')} "
                f"T3={gate.get('t3')} six_ok={gate.get('six_ok')}")
        entry = {"family": cand["family"], "tag": cand["tag"], "members": members_key,
                 "weights": cand["weights"], "synth_id": sid, "nanobench": m, "triad": triad,
                 "gate": gate, "cache": pkl}
        entries.append(entry)
        with open(sidecar_path(sid), "w", encoding="utf-8") as fh:
            json.dump({"key": key, "scored": True, "members": [list(x) for x in members_key],
                       "weights": cand["weights"], "family": cand["family"], "tag": cand["tag"],
                       "basin": int(ref[0]),
                       "basin": int(ref[0]),
                       "nanobench": {k: m.get(k) for k in ("nano", "nano4", "f1")},
                       "triad": triad, "gate": gate, "entry": entry}, fh, ensure_ascii=False, indent=2)

    # 登记进台账（⇒ board_t4_watch 自动补板端时延 ⇒ auto_promote 自动升轨）
    n_reg = 0
    try:
        from manu.pipeline.npu import family_soup as FS  # noqa: PLC0415
        for e in entries:
            if e.get("members"):
                e["members"] = [tuple(x) for x in e["members"]]
        n_reg = FS.register_synth_arms(entries, ns)
    except Exception as exc:  # noqa: BLE001
        log(f"[登记] 失败：{exc}")
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump({"ts": time.strftime("%F %T"), "ref": list(ref), "entries": entries,
                   "registered": n_reg}, fh, ensure_ascii=False, indent=2)
    log(f"[完成] 组合 {len(entries)} 个｜登记 {n_reg} 个｜报告 {args.report}")


if __name__ == "__main__":
    main()