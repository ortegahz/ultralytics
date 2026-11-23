#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""权重空间工具（2026-11-20 16:20）——**把「汤」展平成基成员向量，并度量它离基线多远**。

🔴🔴🔴 为什么需要它（台账 2026-11-20 13:27 快照，6121 个已合成臂的实测）：

  **平均是线性的 ⇒「二阶汤」在数学上 ≡「扁平汤」**
    （`ckpt_synth.average_checkpoints` 用 `acc / sum(w)` 归一化 ⇒ 权重和不必为 1；
      `soup(soup(a,b,c),d)` 与其展开后的扁平汤**逐位等价**）
    ⇒ 台账里 **1491 个「汤套汤」的臂 nano ≥ 基线** ⇒ **「二阶汤必塌」这个结论是错的**
    （被证伪的是**具体那 59 个臂**，不是「二阶汤」这个机制）。

  **真正的机制（三条，全部实测）**：
    ① **量测饱和（resolution floor）**：把每个臂展平成「基成员权重向量」后，
       与基线向量的 **maxΔ（逐成员最大权重差）** 相比：
         · maxΔ ≤ 0.005（407 臂）：**nano / nano4 读数 0/407 非零**（与基线逐位相同）
         · maxΔ ≤ 0.010（757 臂）：nano 0/757 非零；nano4 仅 5/757 非零且**全为负**
         · 最小能改变 nano 读数的 maxΔ = **0.012**
       ⇒ **任何 maxΔ < 0.012 的配方 = 纯烧 GPU**（读数必然与基线逐位相同）。
    ② **单调损失区**：0.010 < maxΔ ≤ 0.030（370 臂）里
       nano **最大增益 = 0**（71 臂变化**全为负**，最差 −8.7e-3）、
       T1 **最大增益 = 0**、T3 **最大增益 = 0**、F1 最大增益 +1.85e-5（≈ 噪声 σ 的 1/77）
       ⇒ **这一段只有「变差」的方向**。
    ③ **前沿池退化**：按六项判据排前 24 的臂，其展平向量两两 **maxΔ ≤ 0.0033**
       ⇒ 把它们两两互汤 = **自交**（信息量为 0）；动态轴唯一真正的扰动是 3% 掺料，
       而 3% 掺料的 maxΔ ≈ **0.0307**（比前沿池自身离散度大 9 倍）⇒ **恰好踩在悬崖边**。

  **⇒ 本模块给两个生成器提供同一套判据**：
     · `Flattener`  展平（递归 soup_members → 基成员权重向量；叶子 epoch 取台账 `picked_epoch`）
     · `max_delta`  与基线向量的 maxΔ / L1 距离
     · `prune_below_floor` **低于分辨率地板的配方一律丢弃**（⇒ 生成器再也不会吐纯浪费配方）
"""
import json
import os

W = "/mnt/data/siping/runs/npu20ms/nano_search"
LEDGER = os.environ.get("NANO_LEDGER") or os.path.join(W, "nano_search_ledger.json")

# 🔴 分辨率地板（实测）：maxΔ < 0.012 ⇒ nano/nano4 读数与基线**逐位相同**
FLOOR_MAX_DELTA = 0.012
# 🔴 单调损失区上界：0.010 < maxΔ ≤ 0.030 ⇒ 只有变差的方向
LOSS_ZONE_MAX_DELTA = 0.030
FLAT_MIN_W = 0.0005      # 展平后低于此权重的成员剔除
FLAT_MAX_DEPTH = 8       # 递归展开深度上限（防环）


def load_ledger(path=None):
    path = path or LEDGER
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Flattener(object):
    """把任意臂（可能是汤）递归展平成 {基成员(trial, epoch): 权重}。"""

    def __init__(self, arms):
        self.arms = arms or {}

    def _soup_of(self, sid):
        """返回 (members, weights)；无汤 ⇒ None；半汤（长度不符）⇒ 'BAD'（整个臂作废）。"""
        st = self.arms.get(str(sid)) or {}
        sm = st.get("soup_members")
        sw = st.get("soup_weights")
        if not sm or not sw:
            return None
        if len(sm) != len(sw):
            return "BAD"
        return list(sm), [float(x) for x in sw]

    def leaf_epoch(self, sid):
        """叶子臂的真实 ckpt epoch；非整数 epoch ⇒ None（不安全）。"""
        st = self.arms.get(str(sid)) or {}
        ep = st.get("picked_epoch")
        if ep is None:
            return 5
        try:
            f = float(ep)
        except Exception:
            return 5
        if abs(f - round(f)) > 1e-9:
            return None
        return int(round(f))

    def flatten(self, sid, w=1.0, depth=0):
        """返回 {基成员: 权重}；不可安全展平时返回 None。"""
        if depth >= FLAT_MAX_DEPTH:
            return None
        soup = self._soup_of(sid)
        if soup == "BAD":
            return None
        if soup is not None:
            sm, sw = soup
            acc = {}
            for mem, mw in zip(sm, sw):
                sub = self.flatten(int(mem[0]), float(mw) * w, depth + 1)
                if sub is None:
                    return None
                for k, v in sub.items():
                    acc[k] = acc.get(k, 0.0) + v
            return acc
        ep = self.leaf_epoch(sid)
        if ep is None:
            return None
        return {(int(sid), ep): float(w)}

    def is_soup_member(self, sid):
        st = self.arms.get(str(sid)) or {}
        return bool(st.get("soup_members"))

    def flat_of_members(self, members):
        """members = [(trial, epoch, weight), ...] ⇒ 基成员权重向量（展平后）。"""
        acc = {}
        for (t, e, w) in members:
            sub = self.flatten(int(t), float(w))
            if sub is None:
                return None
            for k, v in sub.items():
                acc[k] = acc.get(k, 0.0) + v
        return acc


def max_delta(a, b):
    """两个展平向量之间的距离：逐成员最大权重差 + L1 距离。"""
    keys = set(a or {}) | set(b or {})
    if not keys:
        return 0.0, 0.0
    mx = max(abs((a or {}).get(k, 0.0) - (b or {}).get(k, 0.0)) for k in keys)
    l1 = sum(abs((a or {}).get(k, 0.0) - (b or {}).get(k, 0.0)) for k in keys)
    return mx, l1


def parse_recipe(rec):
    """'1608003:5:0.65,1612001:5:0.04' ⇒ [(1608003, 5, 0.65), ...]"""
    out = []
    for part in str(rec).split(","):
        part = part.strip()
        if not part:
            continue
        t, e, w = part.split(":")
        out.append((int(t), int(e), float(w)))
    return out


def baseline_flat(ledger=None, flattener=None):
    """基线臂的展平权重向量。"""
    led = ledger if ledger is not None else load_ledger()
    fl = flattener or Flattener(led.get("arms") or {})
    return fl.flatten(led["baseline_v2"]["trial"], 1.0)


def prune_below_floor(recipes, base_vec, flattener, floor=FLOOR_MAX_DELTA):
    """**低于分辨率地板的配方一律丢弃**（实测：这类配方读数必然与基线逐位相同 ⇒ 纯烧 GPU）。

    返回 (kept, dropped)；dropped 是 [(rec, maxΔ), ...] 便于日志审计。
    """
    kept, dropped = [], []
    for rec in recipes:
        members = parse_recipe(rec)
        vec = flattener.flat_of_members(members)
        if vec is None:
            dropped.append((rec, None))
            continue
        mx, _l1 = max_delta(vec, base_vec)
        if mx < floor:
            dropped.append((rec, mx))
        else:
            kept.append((rec, mx))
    return kept, dropped
