#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**nano 抬升**配方生成器（nanolift）—— 2026-11-23 新增轴。

★ 这个生成器解决什么问题
────────────────────────────────────────────────────────────────────────
基线 = `Baseline_v3.1 / trial_1990055 @ ep5`，六项实测读数：
    nano = .7242438563327032   nano4 = .6609168241965974   f1 = .8906832039622916
    t1   = .8909238824339264    t2    = .9931631171345595   t2_fp = 3   t3 = .9488855869242199
基线配方 = soup(1608003@5:w0.6524, 1612001@5:w0.0360, 1530006@5:w0.2116, 1623003@5:w0.1000)

**全库扫描（13,765 臂）六项全过 ∧ 严格更优 = 0 臂** ⇒ 汤轴的"前沿"全是基线克隆
⇒ 想抬 nano/nano4 必须**换血统结构**（掺 A 叶纯汤成员），不是继续微调基线四成员的权重。
本轴落点：**桥接家族 `sb3` 是全库唯一"nano 打平（dNano=0）且 T1 严格为正"的家族**，
但它**只欠 nano4 与 T3** ⇒ 只需四类"补缺"动作：掺 nano4 供体 / 掺 T3 供体 /
细扫 1608003 三成员族 / 外推 1614000 家族。

★ 依据哪些实测数据（**全部实测，这些数字不要改**）
────────────────────────────────────────────────────────────────────────
① **成员级读数（相对基线的差值）**：
    1100275@1  dNano −0.003308  dNano4 −0.003781  dF1 +0.000144  dT1 −0.000165  dT2 +0.000061  dT3 −0.000365  dT2fp  0
    1104407@1  dNano −0.003781  dNano4 +0.003781  dF1 −0.000306  dT1 −0.001874  dT2 −0.000061  dT3 +0.000619  dT2fp  0
    1102033@1  dNano +0.000236  dNano4 +0.012051  dF1 +0.000236  dT1 −0.002494  dT2 −0.000671  dT3 −0.000584  dT2fp  0
    1612001@5  dNano +0.000000  dNano4 −0.000236  dF1 −0.000098  dT1 −0.000056  dT2 −0.000061  dT3 −0.000015  dT2fp +1
    1608003@5  dNano +0.000000  dNano4 −0.000236  dF1 −0.000005  dT1 −0.000060  dT2 −0.000030  dT3 −0.000114  dT2fp  0
    1530006@5  dNano −0.000473  dNano4 +0.000709  dF1 −0.000208  dT1 −0.000051  dT2 +0.000304  dT3 +0.000327  dT2fp  0
    1623003@5  dNano +0.000000  dNano4 −0.001654  dF1 +0.000157  dT1 −0.000153  dT2 −0.000091  dT3 −0.000015  dT2fp +1
    1614000@5  dNano −0.000709  dNano4 −0.002363  dF1 +0.000088  dT1 −0.000031  dT2 −0.000121  dT3 −0.000114  dT2fp +1
    14963@4    dNano −0.005198  dNano4 +0.000945  dF1 +0.003581  dT1 +0.000395  dT2 +0.003612  dT3 −0.000117  dT2fp +1
                （训练臂，**唯一 nano4 正 + T1 正**的供体）
    2106006@5  dNano −0.011578  dNano4 +0.004017  dF1 +0.001156  dT1 +0.000310  dT2 +0.004548  dT3 +0.001278  dT2fp +1
                （训练臂，**T1 正**的供体）
② **桥接骨架实测**（`sb3:1100275@1 + 1102004@1 + 1612001@5`，比例 .30/.20/.50）：
    dNano **+0.000000** / dNano4 −0.001181 / dF1 +0.000120 / dT1 **+0.000028** / dT2 −0.000030 /
    dT3 −0.000166 / dT2fp 0 ⇒ **4/6，只欠 nano4 与 T3**
    ⇒ batch0 以它为骨架掺供体；batch3 以它为块掺进基线配方。
③ **1608003 三成员族** `{1608003@5, 1543006@5, 1623003@5}`：最佳臂 1930087[.4,.3,.3] /
    1940085[.5,.3,.2] / 1910083[.6,.3,.1] 都是 **5/6，只欠 T1**（dT1 −0.000167 / −0.000143 /
    −0.000162，其余轴全 ≥ 0）⇒ batch1 细扫找 T1 转正；该族 dT1 最优点是
    [0.5,0.25,0.25] 与 [0.6,0.25,0.15]（dT1 −0.000102，但 nano4 退到 −0.000473）。
④ **1614000 家族** `{1614000@5, 1104407@1}`：w(1104407)=0.062 → dT1 −0.000088 / dNano4 −0.000709；
    0.066 / 0.068 / 0.070 → dT1 −0.000065 / dNano4 −0.000709（**走平**）
    ⇒ batch2 外推 0.070~0.090 撞台阶。
⑤ **成员 ckpt 已核存在**（2026-11-23，SSH 只读 stat）：
    `/mnt/data/siping/runs/npu20ms_offcal/trials/trial_<TRIAL>/weights/epoch<EP>.pt`
    本轴用到的 10 个 (trial, epoch) 全部存在：1100275@1 / 1102004@1 / 1612001@5 / 1530006@5 /
    1104407@1 / 1102033@1 / 1623003@5 / 1608003@5 / 1543006@5 / 1614000@5。

★ 怎么用
────────────────────────────────────────────────────────────────────────
    python gen_nanolift_recipes.py <batch> [chunk] [nchunk]
    · `batch` < 0 ⇒ 全部批次（去重后）
    · `chunk`/`nchunk` ⇒ 步进切片 `recs[chunk::nchunk]`（不同 chunk 不重叠，合成通道按片跑）
    · `--ckpt-root <dir>` ⇒ 额外做 ckpt 存在性检查（只读 stat）：缺失成员被剔除、
      剩余成员权重重新归一化，剔除明细写 stderr（默认不做检查，保持纯生成器）
⑥ **A 叶四元组 × 桥接锚**（台账实测臂 1900096）：成员集
    `{1100275@1, 1102004@1, 1104407@1, 1623003@5}` 比例 [.52,.37,.095,.015]
    ⇒ dNano +0.000000 / **dNano4 +0.000000** / dF1 +0.000014 / dT1 −0.000060 / dT3 −0.000114
    ⇒ **4/6（只欠 T1 与 T3），nano4 比桥接家族好 0.001181**。
    ⇒ batch5 把它与桥接锚 1612001@5 合并：桥接家族靠 1612001 把 dT1 顶正（+0.000028），
      A 叶四元组靠 1104407@1 把 dNano4 补到 0 ⇒ **目标是拿到「dNano4≥0 且 dT1>0」的点**
      （实测全库尚不存在此点：dT1>0 ∧ dNano4>0 = 0 臂）。

输出（stdout）：`trial:epoch:w,trial:epoch:w;...` —— `night_soup3.py --recipes` 格式，
**权重归一化到和 = 1，保留 4 位小数**；统计信息写 stderr。
"""
from __future__ import annotations

import argparse
import os
import sys

# ── 成员常量（trial, epoch）；ckpt 均已核存在 ─────────────────────────────
L0275 = (1100275, 1)   # 桥接骨架：dNano 顶格来源
L2004 = (1102004, 1)   # 桥接骨架：dF1 正贡献
L4407 = (1104407, 1)   # 供体：nano4 +0.003781 / T3 +0.000619
L2033 = (1102033, 1)   # 供体：nano4 +0.012051（全集最强）
B8003 = (1608003, 5)   # 基线成员（主权重 .6524）
B1201 = (1612001, 5)   # 基线成员（桥接锚：dNano=0 的来源）
B3006 = (1530006, 5)   # 基线成员 + 免费 T3 供体（dT1 −0.000051）
B2303 = (1623003, 5)   # 基线成员（末端槽：家族桶由它的权重定）
B5430 = (1543006, 5)   # 1608003 三成员族槽 2
B1400 = (1614000, 5)   # 1614000 家族槽 1

BASE_ORDER = (B8003, B1201, B3006, B2303)          # 基线成员顺序（trial_1990055）
BASE_W = {B8003: 0.6524, B1201: 0.0360, B3006: 0.2116, B2303: 0.1000}
BRIDGE_MEMBERS = (L0275, L2004, B1201)              # 桥接骨架 sb3
BRIDGE_RATIO = (0.30, 0.20, 0.50)                   # 实测最优点（1970025）
TRIMEM_RATIO = (0.50, 0.30, 0.20)                  # 1608003 三成员族最佳臂比例
DONORS = (B3006, L4407, L2033, B2303)               # batch0 供体：T3 / nano4+T3 / nano4 / F1 对照
# 🔴 2026-11-23 11:50 —— **A 叶四元组**（台账实测臂 1900096 / 1870128）：
#   成员集 {1100275@1, 1102004@1, 1104407@1, 1623003@5}，比例 [.52,.37,.095,.015]
#   ⇒ dNano **+0.000000** / dNano4 **+0.000000** / dF1 +0.000014 / dT1 −0.000060 / dT3 −0.000114
#   ⇒ **4/6，只欠 T1 与 T3**（比桥接家族 dNano4 −0.001181 好得多，但 dT1 是负的）
#   ⇒ batch5 把它与桥接锚 1612001@5 合并：**桥接家族靠 1612001 把 dT1 顶正**，
#      A 叶四元组靠 1104407@1 把 dNano4 补到 0 ⇒ 两家各取所长。
QUAD_MEMBERS = (L0275, L2004, L4407, B2303)
QUAD_RATIO = (0.52, 0.37, 0.095, 0.015)
DOSES = (0.05, 0.10, 0.15, 0.20)
MIN_WEIGHT = 0.05                                    # 成员权重下限（过小的成员读数不稳）
SCALE = 10000                                        # 权重精度 4 位小数（1e-4 整数域求和）


def _emit(members, weights):
    """权重归一化到和 = 1（4 位小数），返回 `trial:epoch:w` 逗号串。

    · 整数域（1e-4）求和 ⇒ 和 **精确** = 1.0000，不吃浮点累加残差
    · 舍入余数由权重最大的成员吸收（对齐 t1close 的"松弛槽吸收差额"约定）
    · 任一成员权重必须 > 0
    """
    units = [round(float(w) * SCALE) for w in weights]
    residual = SCALE - sum(units)
    if residual:
        units[max(range(len(units)), key=lambda i: units[i])] += residual
    if sum(units) != SCALE:
        raise ValueError("权重和 != 1: %r" % (units,))
    if any(u <= 0 for u in units):
        raise ValueError("成员权重必须 > 0: %r" % (units,))
    return ",".join("%d:%d:%.4f" % (m[0], m[1], u / SCALE) for m, u in zip(members, units))


def _merge(members, weights):
    """同 (trial, epoch) 出现在多处 ⇒ 合并成一项、权重相加（1612001@5 两边都有）。"""
    order, acc = [], {}
    for m, w in zip(members, weights):
        if m not in acc:
            acc[m] = 0.0
            order.append(m)
        acc[m] += w
    return order, [acc[m] for m in order]


def _drop_missing(pairs, root):
    """只读 stat 剔除 ckpt 不存在的成员；剩余成员重新归一化；成员 < 2 的配方整条丢弃。"""
    if not root:
        return pairs
    if not os.path.isdir(root):
        sys.stderr.write("[nanolift] ckpt-root %s 不存在 ⇒ 跳过 ckpt 存在性检查\n" % root)
        return pairs
    kept, missing = [], {}
    for members, weights in pairs:
        keep_m, keep_w, gone = [], [], []
        for m, w in zip(members, weights):
            path = os.path.join(root, "trial_%d" % m[0], "weights", "epoch%d.pt" % m[1])
            if os.path.isfile(path):
                keep_m.append(m)
                keep_w.append(w)
            else:
                gone.append(m)
        for m in gone:
            missing[m] = missing.get(m, 0) + 1
        if len(keep_m) < 2:
            continue
        total = sum(keep_w)
        kept.append((keep_m, [w / total for w in keep_w]))
    for m, cnt in sorted(missing.items()):
        sys.stderr.write("[nanolift] ckpt 缺失 trial_%d@ep%d ⇒ 剔除该成员（%d 条配方）\n"
                         % (m[0], m[1], cnt))
    return kept


# ── 各批次 ─────────────────────────────────────────────────────────────────
def _batch0():
    """batch0 = 桥接骨架 + 单一供体（补 nano4 与 T3 两个缺口）⇒ 4 供体 × 4 剂量 = 16 条。

    骨架 `sb3`（1100275 .30 / 1102004 .20 / 1612001 .50）整体按 (1-d) 缩放，
    供体按剂量 d 掺入。供体各自补的轴（成员级实测）：
    1530006@5 → T3 +0.000327 且 T1 −0.000051（近乎免费）；1104407@1 → nano4 +0.003781 且
    T3 +0.000619；1102033@1 → nano4 +0.012051（最强但 T1 −0.002494）；1623003@5 → 基线成员，
    F1 +0.000157 但 nano4 −0.001654（作对照档）。
    """
    out = []
    for donor in DONORS:
        for d in DOSES:
            members = list(BRIDGE_MEMBERS) + [donor]
            weights = [r * (1.0 - d) for r in BRIDGE_RATIO] + [d]
            out.append((members, weights))
    return out


def _batch1():
    """batch1 = 1608003 三成员族细扫（补 T1 缺口）⇒ w(1608003) 5 × w(1543006) 4 = 20 条。

    既有最佳臂 1930087 / 1940085 / 1910083 都是 5/6 只欠 T1 ⇒ 在该族内部细扫 T1 转正，
    且成员权重不过小（任一权重 < 0.05 的组合直接过滤）。
    """
    out = []
    for w8003 in (0.40, 0.45, 0.50, 0.55, 0.60):
        for w5430 in (0.20, 0.25, 0.30, 0.35):
            w2303 = round(1.0 - w8003 - w5430, 6)
            if min(w8003, w5430, w2303) < MIN_WEIGHT - 1e-9:
                continue
            out.append(([B8003, B5430, B2303], [w8003, w5430, w2303]))
    return out


def _batch2():
    """batch2 = 1614000 家族外推（撞 dT1 台阶）⇒ w(1104407) 8 档 = 8 条。

    实测 w(1104407) 到 0.070 时 dT1 走平在 −0.000065 ⇒ 继续外推 0.072~0.090 找台阶口。
    """
    out = []
    for w4407 in (0.070, 0.072, 0.074, 0.076, 0.078, 0.080, 0.085, 0.090):
        out.append(([B1400, L4407], [round(1.0 - w4407, 6), w4407]))
    return out


def _batch3():
    """batch3 = 基线配方 + 桥接块小剂量 ⇒ d ∈ {.10,.15,.20,.25,.30} = 5 条。

    基线四成员整体按 (1-d) 缩放、桥接块按 d 掺入；**1612001@5 两边都有 ⇒ 必须合并成一项**
    并把权重相加（合并后仍按基线成员顺序排列，A 叶成员追加在末位）。
    """
    out = []
    for d in (0.10, 0.15, 0.20, 0.25, 0.30):
        raw_m = list(BASE_ORDER) + [m for m in BRIDGE_MEMBERS]
        raw_w = [BASE_W[m] * (1.0 - d) for m in BASE_ORDER]
        raw_w += [r * d for r in BRIDGE_RATIO]
        members, weights = _merge(raw_m, raw_w)
        out.append((members, weights))
    return out


def _batch4():
    """batch4 = 1608003 三成员骨架 + T3 供体 ⇒ d ∈ {.05,.10,.15,.20,.25} = 5 条。

    骨架比例取最佳臂 1940085 的 [.5,.3,.2]（5/6 只欠 T1），按 (1-d) 缩放后掺
    1530006@5（dT3 +0.000327 / dNano4 +0.000709 / dT1 −0.000051）补 T3。
    """
    out = []
    for d in (0.05, 0.10, 0.15, 0.20, 0.25):
        members = [B8003, B5430, B2303, B3006]
        weights = [r * (1.0 - d) for r in TRIMEM_RATIO] + [d]
        out.append((members, weights))
    return out


def _batch5():
    """batch5 = **A 叶四元组 × 桥接锚 1612001**（两条最好家族的交叉点）⇒ 16 条。

    5a（12 条）四成员 `{1100275,1102004,1104407,1612001}` 二维扫：
        w(1612001) ∈ {.25,.35,.45,.50} × w(1104407) ∈ {.05,.10,.15}，
        余下权重按桥接比例 3:2 分给 1100275/1102004。
    5b（4 条）五成员 `{1100275,1102004,1104407,1623003,1612001}`：
        以 A 叶四元组实测比例 [.52,.37,.095,.015] 按 (1-d) 缩放，1612001@5 掺入剂量
        d ∈ {.10,.20,.30,.40}。
    依据：桥接家族 dT1=+0.000028 但 dNano4=−0.001181；A 叶四元组 dNano4=0 但
    dT1=−0.000060 ⇒ 合并后看能否同时拿到 dNano4≥0 与 dT1>0（**全库尚无此点**）。
    """
    out = []
    for w1201 in (0.25, 0.35, 0.45, 0.50):
        for w4407 in (0.05, 0.10, 0.15):
            rest = round(1.0 - w1201 - w4407, 6)
            w0275 = round(rest * 0.60, 6)
            w2004 = round(rest * 0.40, 6)
            if min(w0275, w2004, w4407, w1201) < MIN_WEIGHT - 1e-9:
                continue
            out.append(([L0275, L2004, L4407, B1201], [w0275, w2004, w4407, w1201]))
    for d in (0.10, 0.20, 0.30, 0.40):
        members = list(QUAD_MEMBERS) + [B1201]
        weights = [r * (1.0 - d) for r in QUAD_RATIO] + [d]
        out.append((members, weights))
    return out


BATCHES = [_batch0, _batch1, _batch2, _batch3, _batch4, _batch5]


def main():
    """用法：`gen_nanolift_recipes.py <batch> [chunk] [nchunk]`

    · `batch` < 0 ⇒ 全部批次（去重后）
    · `chunk`/`nchunk` ⇒ 步进切片（合成通道按片跑 ⇒ 不会一次性把 GPU 压满）
    """
    ap = argparse.ArgumentParser(description="nano 抬升轴配方生成器（nanolift）")
    ap.add_argument("batch", nargs="?", type=int, default=0,
                    help="批次号；< 0 表示全部批次")
    ap.add_argument("chunk", nargs="?", type=int, default=0, help="切片序号（步进切片）")
    ap.add_argument("nchunk", nargs="?", type=int, default=1, help="切片总数")
    ap.add_argument("--ckpt-root", default=None,
                    help="trials 目录（如 /mnt/data/siping/runs/npu20ms_offcal/trials）"
                         "⇒ 做 ckpt 存在性检查；默认不做")
    args = ap.parse_args()

    idxs = range(len(BATCHES)) if args.batch < 0 else [args.batch % len(BATCHES)]
    pairs, counts = [], []
    for idx in idxs:
        batch_pairs = _drop_missing(BATCHES[idx](), args.ckpt_root)
        pairs.extend(batch_pairs)
        counts.append((idx, len(batch_pairs)))

    seen, recs = set(), []
    for members, weights in pairs:
        rec = _emit(members, weights)
        key = tuple(sorted(rec.split(",")))
        if key in seen:
            continue
        seen.add(key)
        recs.append(rec)

    if args.nchunk > 1:
        recs = recs[args.chunk % args.nchunk::args.nchunk]
    sys.stderr.write("[nanolift] batch=%d chunk=%d/%d 配方 %d 条\n"
                     % (args.batch, args.chunk, args.nchunk, len(recs)))
    sys.stderr.write("[nanolift] 各批次（切片前）：%s\n"
                     % " ".join("batch%d=%d" % (i, c) for i, c in counts))
    print(";".join(recs))


if __name__ == "__main__":
    main()
