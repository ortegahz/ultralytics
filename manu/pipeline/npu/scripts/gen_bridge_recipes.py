#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**桥接家族**配方生成器（bridge）—— 2026-11-23 新增轴（取代 t1close 的 FAM5 细网格）。

★ 为什么换轴（**全部实测**，台账快照 2026-11-23 10:05）
────────────────────────────────────────────────────────────────────────
① **全库扫描结论：六项全过 ∧ 严格更优 = 0 臂**（13,765 臂）⇒ 汤轴的"前沿"全是**基线克隆**。
   · 57 臂六项全过 ⇒ **逐位相同**于基线（nano/nano4/T1/T3 全同）。
   · 115 臂 T1 逐位等于基线 ⇒ **nano 也逐位等于基线**（平台上 nano 与 T1 完全联动）。

② **T1 优势臂全带 nano 赤字**：全库 47 个 dT1>0 的臂，**dNano 无一为正**（最好 −23.63e-5）
   ⇒ **汤的"加成员"路线结构性地撞墙**：FAM5+供体 125 个已门禁臂 dT1 ∈ [−22.23, −3.68]e-5，
   **无一触到 0** ⇒ 继续在 FAM5 盒子里细扫是**确定性浪费**（故停掉 t1close 扫网）。

③ 🔴 **新杠杆：桥接家族 `sb3:1100275@1+1102004@1+1612001@5:bw50`**
   台账实测 8 臂，**dT1 随 w(1100275) 单调上升**，且**扫描恰好停在 0.32（从未越界）**：

       w(1100275)  0.23    0.24    0.25    0.26    0.27    0.29    0.30    0.32
       dNano      -23.63   +0.00   +0.00   +0.00   +0.00   +0.00   +0.00   +0.00
       dNano4    -141.78 -165.41 -141.78 -141.78 -141.78 -141.78 -118.15 -141.78
       dF1         +3.25   +5.56   +7.87   +8.32  +10.18  +16.19  +12.03  +14.34
       dT1        -13.43  -13.43  -13.43   -7.40   -3.22   -1.37   +2.81   +2.35
       dT3        -21.85  -21.85  -21.85  -16.64  -11.94  -16.64  -16.64  -16.64

   ⇒ **1612001@5 正是基线成员**（基线 = {1608003@5, 1612001@5, 1530006@5, 1623003@5}）
   ⇒ 这个家族是**基线族与 A 叶纯汤血统之间的天然桥梁** ⇒ dNano 能守住 0 而 dT1 爬到 +2.81。

④ **本轴四条线**（按 EV 排序）：
   · batch0 = **续扫 w(1100275) 0.33→0.48**（bw50 ⇒ **直接并入既有家族 n=8** ⇒ 家族票直接累积）
   · batch1 = 桥接 2-D（换 w(1612001) 锚 ⇒ 修 dNano4 / dT3）
   · batch2 = **基线 6 成员桥接**（基线权重整体缩放 + 桥接块按剂量 d 掺入）
   · batch3 = 换锚（用基线另外三个成员当锚，验证 dNano=0 是否只有 1612001 能做到）
   · batch4 = 换血统对（1104407 / 1102001 / 1102002 等 A 叶成员配对）
   · batch5 = 基线桥接 2-D（剂量 d × 桥接内部比例）

用法：`python gen_bridge_recipes.py <batch> [chunk] [nchunk]`
输出：`trial:epoch:w,trial:epoch:w;...`（`night_soup3.py --recipes` 格式，权重和恒 = 1）
"""
from __future__ import annotations

import sys

# ── 成员常量（台账实测 + 权重文件已核存在 /mnt/data/siping/runs/npu20ms_offcal）────
L0275 = (1100275, 1)   # A 叶纯汤血统（自身 dT1 高基线）
L2004 = (1102004, 1)   # A 叶纯汤血统
L4407 = (1104407, 1)
L2001 = (1102001, 1)
L2002 = (1102002, 1)
B8003 = (1608003, 5)   # 基线成员
B1201 = (1612001, 5)   # 基线成员（桥接锚：dNano=0 的来源）
B3006 = (1530006, 5)   # 基线成员
B2303 = (1623003, 5)   # 基线成员

# 基线权重（trial_1990055 @ ep5）
BASE_W = {B8003: 0.6524, B1201: 0.036, B3006: 0.2116, B2303: 0.10}
# 桥接块内部比例（沿用实测最优点 1970025：0.30 / 0.20 / 0.50）
BRIDGE_RATIO = (0.30, 0.20, 0.50)   # (1100275, 1102004, 1612001)


def _emit(members, weights) -> str:
    """权重和必须精确 = 1（否则残差塞进末位 ⇒ 家族桶漂移）；且必须全部 ≥ 0。"""
    w = [round(float(x), 6) for x in weights]
    tot = round(sum(w), 6)
    if abs(tot - 1.0) > 1e-9:
        raise ValueError("权重和 != 1: %r (sum=%r)" % (w, tot))
    if any(x < 0 for x in w):
        raise ValueError("负权重: %r" % (w,))
    return ",".join("%d:%d:%.6f" % (m[0], m[1], x) for m, x in zip(members, w))


def _mk3(pair, anchor, w_anchor, w_a):
    """3 成员桥接：<pairA@1, pairB@1, anchor@5>，锚权重固定，pair 之间分配剩余。"""
    w_b = round((1.0 - w_anchor) - w_a, 6)
    if w_b < 0:
        return None
    return _emit([pair[0], pair[1], anchor], [w_a, w_b, w_anchor])


def _mk_base_bridge(d, ratio=BRIDGE_RATIO):
    """基线 6 成员桥接：基线权重整体缩 (1-d)，桥接块按 d 掺入（含 1612001@5）。"""
    r0, r1, r2 = ratio
    w8003 = round(BASE_W[B8003] * (1.0 - d), 6)
    w3006 = round(BASE_W[B3006] * (1.0 - d), 6)
    w2303 = round(BASE_W[B2303] * (1.0 - d), 6)
    w1201 = round(BASE_W[B1201] * (1.0 - d) + r2 * d, 6)
    w0275 = round(r0 * d, 6)
    w2004 = round(r1 * d, 6)
    return _emit([B8003, B1201, B3006, B2303, L0275, L2004],
                 [w8003, w1201, w3006, w2303, w0275, w2004])


# ── 各批次 ─────────────────────────────────────────────────────────────────
def _batch0() -> list:
    """续扫 bw50 家族（并入既有 n=8 ⇒ 家族票直接累积）。"""
    out = []
    i = 33
    while i <= 48:
        w_a = round(i / 100.0, 6)
        rec = _mk3((L0275, L2004), B1201, 0.50, w_a)
        if rec:
            out.append(rec)
        i += 1
    return out


def _batch1() -> list:
    """桥接 2-D：换 w(1612001) 锚（修 dNano4 / dT3）。"""
    out = []
    for w_anchor in (0.40, 0.45, 0.55, 0.60):
        for w_a in (0.26, 0.30, 0.34, 0.38, 0.42, 0.46):
            rec = _mk3((L0275, L2004), B1201, w_anchor, w_a)
            if rec:
                out.append(rec)
    return out


def _batch2() -> list:
    """基线 6 成员桥接：剂量 d 扫描。

    ⚠️ **d >= 0.15 是硬下限**：基线桥接的新成员坐标权重 = 0.30d / 0.20d ⇒ 展平 maxΔ 同阶；
    实测 maxΔ < 0.012 ⇒ 读数逐位同格；0.012~0.030 ⇒ nano/T1/T3 最大增益 = 0（死区）
    ⇒ d=0.05/0.10 的 maxΔ 只有 0.015/0.030 ⇒ **纯烧 GPU**，故从 0.15 起扫。
    """
    out = []
    d = 0.15
    while d <= 0.5001:
        out.append(_mk_base_bridge(round(d, 4)))
        d += 0.05
    return out


def _batch3() -> list:
    """换锚：用基线另外三个成员当锚，验证 dNano=0 是否只有 1612001 能做到。"""
    out = []
    for anchor in (B3006, B8003, B2303):
        for w_a in (0.26, 0.30, 0.34, 0.38, 0.42, 0.46):
            rec = _mk3((L0275, L2004), anchor, 0.50, w_a)
            if rec:
                out.append(rec)
    return out


def _batch4() -> list:
    """换血统对：其它 A 叶成员配 1612001 锚。"""
    out = []
    for pair in ((L0275, L4407), (L0275, L2001), (L0275, L2002), (L2004, L4407)):
        for w_a in (0.26, 0.30, 0.34, 0.38, 0.42, 0.46):
            rec = _mk3(pair, B1201, 0.50, w_a)
            if rec:
                out.append(rec)
    return out


def _batch5() -> list:
    """基线桥接 2-D：剂量 d × 桥接内部比例。"""
    out = []
    ratios = [(0.30, 0.20, 0.50), (0.40, 0.10, 0.50), (0.20, 0.30, 0.50),
              (0.34, 0.16, 0.50), (0.26, 0.24, 0.50)]
    for d in (0.15, 0.20, 0.30, 0.40):
        for r in ratios:
            out.append(_mk_base_bridge(d, r))
    return out


BATCHES = [_batch0, _batch1, _batch2, _batch3, _batch4, _batch5]


def main() -> None:
    """用法：`gen_bridge_recipes.py <batch> [chunk] [nchunk]`

    · `batch` < 0 ⇒ 全部批次（去重后）
    · `chunk`/`nchunk` ⇒ 轮转切片（合成通道按片跑 ⇒ 早出的片先被门禁通道消化）
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
    sys.stderr.write("[bridge] batch=%d chunk=%d/%d 配方 %d 条\n" % (batch, chunk, nchunk, len(out)))
    print(";".join(out))


if __name__ == "__main__":
    main()
