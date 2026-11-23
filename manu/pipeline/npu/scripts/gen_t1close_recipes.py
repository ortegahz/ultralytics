#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1 收口扫网配方生成器（t1close）—— 2026-11-23 新增轴。

★ 为什么有这条轴（**全部来自实测，不是推理**；台账快照 2026-11-23 09:50）
────────────────────────────────────────────────────────────────────────
① **T1 是唯一真缺口**：FAM5 家族（{1608003,1612001,1530006,1543006,1623003}）89 个已门禁臂里
   一大批是 **5/6**（nano / nano4 / f1 / t2_fp / t3 全部 ≥ 基线），**只差 T1 0.5~1.2 阶**
   （5e-5~1.2e-4 ≈ 1~4 个检出）。实测最优两条前沿：
     · `1920123`（FAM5 5 成员）w=(.25,.40,.15,.15,.05)
         dNano 0.0 / dNano4 **+4.7 阶** / dF1 +0.1 / **dT3 +1.6 阶** / **dT1 −0.7 阶** / t2fp 3
     · `2000343`（FAM5 + T1 供体 6 成员）w=(.28,.25,.20,.10,.05,.10)
         dNano +2.4 / dNano4 0.0 / dF1 +0.7 / dT3 +0.1 / **dT1 −0.6 阶** / t2fp 3
   ⇒ **两条前沿各自差 T1 不到 1 阶** ⇒ 只差「撞对一两格」。

② **T1 台阶量化在 ≈1.5e-5（0.15 阶）** ⇒ 在这个分辨率上 T1 局部看是**离散细胞**（不是平滑函数）
   ⇒ **只能靠密集采样去撞，不能靠线性外推** ⇒ 本轴是**细网格**，不是插值。

③ **T1 杠杆 = A 叶纯汤比例**：台账里 dT1>0 的臂，T1 最高的一批全是 A 叶纯汤
   （`sb3:1100275@1+1102004@1:bwXX`）：`1603002` T1 .8910085（+0.85 阶）、
   `1610001` .8909990、`1603004` .8909854、`1609001` .8909763；**四者 t2_fp 都是 3**（不顶虚警）。

④ 🔴 **w1530006 上探假设已被实测证伪**：`w3 > .2116` 的 91 个已门禁臂，T3 分布
   `{.9488856 : 75, .9488335 : 16}` ⇒ **没有一个超过基线** ⇒ 本轴**不推 w3**，只做邻域细扫。

⑤ 🔴 **分辨率地板（决定本轴不踩雷区）**：把臂展平成「基成员权重向量」后与基线比 **maxΔ**：
   · maxΔ < 0.012 ⇒ nano/nano4 读数与基线**逐位相同**（纯烧 GPU）
   · 0.012 ≤ maxΔ ≤ 0.030 ⇒ nano / T1 / T3 **最大增益 = 0**（单调损失区）
   · **0.03 ≤ maxΔ ≲ 0.16 才是「活带」**
   ⇒ 本轴所有配方都落在**实测有产出的盒子**内（对照实测：FAM5 家族 maxΔ 0.055~0.095）。

⑥ **供体剂量必须小**：供体自身 nano4 低 19~30 阶 ⇒ 剂量大了会把 nano4 拖下去。
   实测 6 成员最优臂的供体剂量都是 **0.05** ⇒ 本轴供体剂量只取 {0.04, 0.05, 0.06, 0.08}。

⑦ **家族 n≥4 是升轨硬门槛** ⇒ 成员**顺序**必须与既有同族臂一致（`fam_tag` 用末位成员权重定桶）：
   6 成员用 `[1608003,1612001,1530006,1543006,<供体>,1623003]`、5 成员用
   `[1608003,1612001,1530006,1543006,1623003]` ⇒ 与既有 1990344~1990350 / 2000339~2000344 / 19xxxx 同族同桶。

用法（服务器）
    python gen_t1close_recipes.py [batch]      # batch 0..N，见下面 BATCHES
输出：`trial:epoch:w,trial:epoch:w;...`（`night_soup3.py --recipes` 的格式）
"""
from __future__ import annotations

import sys

# ── 成员常量（台账实测）──────────────────────────────────────────────
A8003 = (1608003, 5)   # FAM5 槽1
A1201 = (1612001, 5)   # FAM5 槽2（nano4 顶格关键供体）
A1    = (1530006, 5)   # FAM5 槽3
A2306 = (1543006, 5)   # FAM5 槽4（dT3 严格更优的来源）
A2303 = (1623003, 5)   # FAM5 槽5（末位 ⇒ 家族桶由它的权重定）

# T1 供体（A 叶纯汤）：自身 T1 高于基线 0.5~0.85 阶，t2_fp 均为 3
DONORS = [
    ((1603002, 5), "T1D_02"),   # t1 .8910085 / n4 .6578450
    ((1610001, 5), "T1D_01"),   # t1 .8909990 / n4 .6590265（供体里 n4 最好）
    ((1603004, 5), "T1D_04"),   # t1 .8909854 / n4 .6585539
    ((1609001, 5), "T1D_90"),   # t1 .8909763 / n4 .6576087 / nano .7240076（供体里 nano 最高）
]

# 细网格分辨率：0.01（实测响应是离散细胞 ⇒ 必须细）
STEP = 0.01
OFFSETS = (-0.02, -0.01, 0.0, 0.01, 0.02)

# 6 成员前沿锚点（台账实测的 5/6 臂）
#   槽序 = 1608003(slack) / 1612001 / 1530006 / 1543006 / <供体> / 1623003
#   ⚠️ 台账里这些臂的权重和**不严格等于 1**（如 2000343 = 0.98）⇒ `average_checkpoints`
#      会按 sum(w) 归一化 ⇒ 本轴统一**归一化到和 = 1**（与 gen_w13 的 emit 契约一致）。
A6_ANCHORS = [
    # (w1201, w1530, w1543, w_donor, w1623) —— w1608003 作松弛槽
    (0.25, 0.20, 0.10, 0.05, 0.10),   # 2000343  dT1 −0.6 / dT3 +0.1 / dNano +2.4 / dF1 +0.7
    (0.25, 0.20, 0.10, 0.08, 0.05),   # 1990349  dT1 −0.7 / dT3 +1.1 / dN4 +2.4
    (0.35, 0.20, 0.10, 0.05, 0.05),   # 2000341  dT1 −0.9 / dT3 +1.1 / dN4 +4.7
    (0.15, 0.20, 0.10, 0.05, 0.05),   # 1990345  dT1 −0.9 / dT3 +1.6 / dN4 +2.4 / dF1 +0.8
]
# 5 成员 FAM5 前沿锚点
A5_ANCHORS = [
    (0.40, 0.15, 0.15, 0.05),   # 1920123  dT1 −0.7 / dT3 +1.6 / dN4 +4.7
    (0.30, 0.20, 0.15, 0.05),   # 1940122  dT1 −0.9 / dT3 +1.6 / dN4 +4.7
    (0.10, 0.20, 0.10, 0.15),   # 1920144  dT1 −0.7 / dT3 +1.0 / dN4 +2.4 / dF1 +0.8
]

LO_W1, HI_W1 = 0.20, 0.55   # w1608003 合法区间（实测有产出的范围）


def _emit(members, weights):
    """权重和必须精确 = 1（否则残差会塞进末位 ⇒ 家族桶漂移）；且必须全部 ≥ 0。"""
    w = [round(float(x), 6) for x in weights]
    tot = round(sum(w), 6)
    if abs(tot - 1.0) > 1e-9:
        raise ValueError("权重和 != 1: %r (sum=%r)" % (w, tot))
    if any(x < 0 for x in w):
        raise ValueError("负权重: %r" % (w,))
    return ",".join("%d:%d:%.6f" % (m[0], m[1], x) for m, x in zip(members, w))


def _mk6(donor, w2, w3, w4, w5, w6, w1=None):
    """6 成员：1608003(slack) / 1612001 / 1530006 / 1543006 / <供体> / 1623003。"""
    if w1 is None:
        w1 = round(1.0 - w2 - w3 - w4 - w5 - w6, 6)
    if not (LO_W1 - 1e-9 <= w1 <= HI_W1 + 1e-9):
        return None
    if w1 < 0 or w2 < 0 or w3 < 0 or w4 < 0 or w5 < 0 or w6 < 0:
        return None
    return _emit([A8003, A1201, A1, A2306, donor, A2303], [w1, w2, w3, w4, w5, w6])


def _mk5(w2, w3, w4, w5, w1=None):
    """5 成员 FAM5：1608003 / 1612001 / 1530006 / 1543006 / 1623003。"""
    if w1 is None:
        w1 = round(1.0 - w2 - w3 - w4 - w5, 6)
    if not (LO_W1 - 1e-9 <= w1 <= HI_W1 + 1e-9):
        return None
    return _emit([A8003, A1201, A1, A2306, A2303], [w1, w2, w3, w4, w5])


def _patch6(donor, anchor, axes):
    """在 6 成员锚点周围，对 axes 指定的两个槽做 5×5 细网格（步长 0.01）。

    🔴 **松弛槽 = 1608003**：两个被扰动的槽各自独立变动，**w1608003 自动吸收差额**
    ⇒ 权重和恒等于 1（`emit` 契约），且 (0,0) 偏移**精确复现锚点**。
    """
    out = []
    a = list(anchor)
    i, j = axes
    for di in OFFSETS:
        for dj in OFFSETS:
            b = list(a)
            b[i] = round(a[i] + di, 6)
            b[j] = round(a[j] + dj, 6)
            rec = _mk6(donor, b[0], b[1], b[2], b[3], b[4])
            if rec:
                out.append(rec)
    return out


def _patch5(anchor, axes):
    out = []
    a = list(anchor)
    i, j = axes
    for di in OFFSETS:
        for dj in OFFSETS:
            b = list(a)
            b[i] = round(a[i] + di, 6)
            b[j] = round(a[j] + dj, 6)
            rec = _mk5(b[0], b[1], b[2], b[3])
            if rec:
                out.append(rec)
    return out


def _coarse():
    """粗网格：把 6 成员盒子整个扫一遍（供体剂量小、w1 落在实测区间）。"""
    out = []
    w1201s = (0.15, 0.25, 0.35)
    w1530s = (0.15, 0.20)
    w1543s = (0.10, 0.15)
    w1623s = (0.05, 0.10, 0.15)
    for donor, _name in DONORS[:2]:
        for d in (0.04, 0.05, 0.06, 0.08):
            for w2 in w1201s:
                for w3 in w1530s:
                    for w4 in w1543s:
                        for w6 in w1623s:
                            rec = _mk6(donor, w2, w3, w4, d, w6)
                            if rec:
                                out.append(rec)
    return out


def _coarse5():
    """5 成员 FAM5 粗网格（不供体，保住 dT3 增益）。"""
    out = []
    for w2 in (0.10, 0.30, 0.40):
        for w3 in (0.15, 0.20):
            for w4 in (0.10, 0.15):
                for w5 in (0.05, 0.10, 0.15):
                    rec = _mk5(w2, w3, w4, w5)
                    if rec:
                        out.append(rec)
    return out


def _donor_only():
    """供体单独替换 1543006 的 5 成员族（对照轴：T1 杠杆更大、但可能砸掉 dT3）。"""
    out = []
    for donor, _name in DONORS:
        for d in (0.05, 0.08, 0.10, 0.12):
            for w2 in (0.20, 0.30, 0.40):
                for w3 in (0.15, 0.20):
                    for w6 in (0.05, 0.10):
                        w1 = round(1.0 - w2 - w3 - d - w6, 6)
                        if not (LO_W1 <= w1 <= HI_W1):
                            continue
                        out.append(_emit([A8003, A1201, A1, donor, A2303],
                                          [w1, w2, w3, d, w6]))
    return out


# ── 批次（batch 越大越靠后；每批 ~100~200 条 ⇒ 合成 ~2~3 h/批）────────
def _batch0():
    """最高 EV：6 成员前沿锚点的 5×5 细网格（槽位对 = (1612001,1530006) 与 (1543006,供体)）。"""
    out = []
    for donor, _n in DONORS[:2]:
        for anchor in A6_ANCHORS:
            out += _patch6(donor, anchor, (0, 1))
            out += _patch6(donor, anchor, (2, 3))
    return out


def _batch1():
    """5 成员 FAM5 前沿锚点的细网格。"""
    out = []
    for anchor in A5_ANCHORS:
        out += _patch5(anchor, (0, 1))
        out += _patch5(anchor, (2, 3))
    return out


def _batch2():
    """6 成员粗网格。"""
    return _coarse()


def _batch3():
    """5 成员 FAM5 粗网格 + 供体替换轴。"""
    return _coarse5() + _donor_only()


def _batch4():
    """6 成员粗网格的扩展（另外两个供体）。"""
    out = []
    w1201s = (0.15, 0.25, 0.35)
    w1530s = (0.15, 0.20)
    w1543s = (0.10, 0.15)
    w1623s = (0.05, 0.10, 0.15)
    for donor, _name in DONORS[2:]:
        for d in (0.04, 0.05, 0.06, 0.08):
            for w2 in w1201s:
                for w3 in w1530s:
                    for w4 in w1543s:
                        for w6 in w1623s:
                            rec = _mk6(donor, w2, w3, w4, d, w6)
                            if rec:
                                out.append(rec)
    return out


BATCHES = [_batch0, _batch1, _batch2, _batch3, _batch4]


def main() -> None:
    """用法：`gen_t1close_recipes.py <batch> [chunk] [nchunk]`

    · `batch` < 0 ⇒ 全部批次（去重后）
    · `chunk`/`nchunk` ⇒ 把结果**轮转切片**（合成通道按片跑 ⇒ 不会一次性把 GPU 压满，
      且早出的片先被门禁通道消化）
    """
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nchunk = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    seen, out = set(), []
    fns = BATCHES if batch < 0 else [BATCHES[batch % len(BATCHES)]]
    for fn in fns:
        for rec in fn():
            key = tuple(sorted(rec.split(",")))
            if key in seen:
                continue
            seen.add(key)
            out.append(rec)
    if nchunk > 1:
        out = out[chunk % nchunk::nchunk]
    sys.stderr.write("[t1close] batch=%d chunk=%d/%d 配方 %d 条\n" % (batch, chunk, nchunk, len(out)))
    print(";".join(out))


if __name__ == "__main__":
    main()
