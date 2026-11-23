#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动态前沿配方生成器（2026-11-19 11:00 / 2026-11-20 16:30 **三处根因修复**）

🔴🔴🔴 2026-11-20 16:30 **根因重判：原版「二阶汤必塌」是错的，真因是「量测饱和 + 前沿池退化」**

  台账 2026-11-20 13:27 快照实测（6121 个已合成臂）：
    ① **平均是线性的**（`ckpt_synth.average_checkpoints` 用 `acc / sum(w)` 归一化）
       ⇒ `soup(soup(a,b,c),d)` ≡ 展开后的**扁平汤**（逐位等价）
       ⇒ 台账里 **1491 个「汤套汤」的臂 nano ≥ 基线**
       ⇒ **「二阶汤必塌」这个机制被证伪**（被证伪的是那 59 个具体臂）。
    ② **真因一：量测饱和（resolution floor）**
       把臂展平成「基成员权重向量」后与基线比 maxΔ（逐成员最大权重差）：
         · maxΔ ≤ 0.005（407 臂）：nano / nano4 **0/407 非零**（与基线逐位相同）
         · maxΔ ≤ 0.010（757 臂）：nano **0/757 非零**；nano4 仅 5/757 且**全为负**
         · 最小能改变 nano 读数的 maxΔ = **0.012**
       ⇒ **maxΔ < 0.012 的配方 = 纯烧 GPU**。
    ③ **真因二：前沿池退化**
       按六项判据排前 24 的臂，展平向量两两 **maxΔ ≤ 0.0033** ⇒ 两两互汤 = **自交**，
       信息量为 0；而原版唯一的真实扰动是 **3% 掺料（maxΔ ≈ 0.0307，比池内离散度大 9 倍）**
       ⇒ 恰好踩在悬崖边（0.010<maxΔ≤0.030 区间 **nano/T1/T3 的最大增益都是 0，只有变差的方向**）
       ⇒ 59 个臂全部落在同一个死点（nano = .7157372，**逐位相同**，因为它们其实是同一个模型）。
    ⇒ **修复方向（不是「展平」）**：
       (a) **池子按「权重空间离散度」选**，不再按分数取前 N（同分近邻 = 无信息）；
       (b) **只配对「彼此相距够远」的臂**（maxΔ ≥ MIN_PAIR_DELTA）；
       (c) **最终配方过分辨率地板**（maxΔ ≥ 0.012）⇒ 低地板配方自动被丢弃；
       (d) **3% 掺料默认关闭**（它是唯一的大扰动，且落在单调损失区）。

  ⚠ 「展平」本身仍然保留且必要：它让**分数不同、但底座相同的**臂能被统一比较与去重，
    也让「叶子 epoch」取台账真实 `picked_epoch`（原版硬编码 epoch 5 ⇒ 指向不存在的 ckpt）。

🔴 为什么需要它：
  · 静态周末空间锁在两个「只差一项」的臂上，但门禁读数要 10~30min 才回来 ⇒ 夜里前沿一动，静态列表就过时。
  · 本脚本**每次调用都重新读台账**：按六项判据给全盘臂打分 ⇒ 取当前最接近全过的 K 个臂，
    **按权重空间离散度**挑出**互不相同**的一组 ⇒ 两两互汤（落在地板之上）⇒ 每轮产出**新信息**。

用法： gen_dyn_recipes.py <chan 0..3> [topk]
      环境变量 NANO_LEDGER 可覆盖台账路径（本地自检用）；
              NANO_CHECK_TRIALS=0 跳过权重文件存在性检查（本地自检用）；
              DYN_GARNISH_W 掺料权重（默认 **0 = 关闭**；>0 时会同时抬高 maxΔ）；
              DYN_FLOOR 分辨率地板（默认 0.012）；
              DYN_MIN_PAIR_DELTA 配对最小权重距离（默认 0.030）；
              DYN_MAX_DELTA 配对后成品臂的 maxΔ 上限（默认 0.180）。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weight_space as WS  # noqa: E402

W = "/mnt/data/siping/runs/npu20ms/nano_search"
LEDGER = os.environ.get("NANO_LEDGER") or os.path.join(W, "nano_search_ledger.json")
ROOT = os.environ.get("NANO_ROOT") or "/mnt/data/siping/runs/npu20ms_offcal"
GATE = os.path.join(ROOT, "gate")
CHECK_TRIALS = os.environ.get("NANO_CHECK_TRIALS", "1") != "0"
GARNISH = (1104407, 1)
GARNISH_W = float(os.environ.get("DYN_GARNISH_W", "0.0"))   # 🔴 默认关闭（原版 0.03 是唯一的大扰动）
FLOOR = float(os.environ.get("DYN_FLOOR", str(WS.FLOOR_MAX_DELTA)))
MIN_PAIR_DELTA = float(os.environ.get("DYN_MIN_PAIR_DELTA", "0.030"))
# 🔴 上界：实测 [0.20,0.30) 区间 nano 128/128 全负；活带 = [0.03, 0.16]
MAX_DELTA = float(os.environ.get("DYN_MAX_DELTA", "0.180"))
# 🔴 候选臂的 maxΔ 活带窗口（相对基线）
MIN_ARM_DELTA = float(os.environ.get("DYN_MIN_ARM_DELTA", "0.030"))
MAX_ARM_DELTA = float(os.environ.get("DYN_MAX_ARM_DELTA", "0.160"))
MIN_NPASS = int(os.environ.get("DYN_MIN_NPASS", "4"))
MIX_RATIOS = (0.40, 0.50, 0.60)
MAX_OUT = 24


def log(msg):
    sys.stderr.write("[dyn] %s\n" % msg)
    sys.stderr.flush()


def load_official(sid, st):
    of = st.get("official") or {}
    if of.get("t1") is not None and of.get("t3") is not None:
        return of
    d = os.path.join(GATE, "NANO_%s_ep5_official" % sid)
    rp = os.path.join(d, "gate_report.json")
    if not os.path.isfile(rp):
        return of
    try:
        rep = json.load(open(rp, encoding="utf-8"))
    except Exception:
        return of
    tr = rep.get("tracks") or {}
    t1 = (tr.get("track1_uav") or {}).get("finetune_peak") or {}
    t2 = tr.get("track2_golden") or {}
    t3 = tr.get("track3_blind") or {}
    if t1.get("f1") is None or t3.get("f1") is None:
        return of
    return {"t1": t1["f1"], "t2_fp": t2.get("fp"), "t3": t3["f1"]}


def score_arm(sid, st, base):
    """按六项判据给一个臂打分 ⇒ (npass, -dist, sid, picked, official) 或 None。"""
    pk = st.get("picked") or {}
    if not pk.get("nano") or not pk.get("nano4") or not pk.get("f1"):
        return None
    if CHECK_TRIALS:
        ep = st.get("picked_epoch")
        try:
            ep = int(round(float(ep)))
        except Exception:
            ep = 5
        if not os.path.isfile(
                os.path.join(ROOT, "trials", "trial_%s" % sid, "weights", "epoch%d.pt" % ep)):
            return None
    of = load_official(sid, st)
    npass = 0
    gaps = [0.0] * 6
    checks = (
        (pk["nano"] >= base["nano"], base["nano"] - pk["nano"]),
        (pk["nano4"] >= base["nano4"], base["nano4"] - pk["nano4"]),
        (pk["f1"] >= base["f1"], base["f1"] - pk["f1"]),
        ((of.get("t1") or -1) >= base["t1"], base["t1"] - (of.get("t1") or 0)),
        ((of.get("t3") or -1) >= base["t3"], base["t3"] - (of.get("t3") or 0)),
        ((of.get("t2_fp") if of.get("t2_fp") is not None else 99) <= base["t2_fp"],
         (of.get("t2_fp") or 0) - base["t2_fp"]),
    )
    for i, (ok, gap) in enumerate(checks):
        if ok:
            npass += 1
        else:
            gaps[i] = max(gap, 0.0)
    dist = (sum(gaps[:3]) / 1e-4) + (sum(gaps[3:5]) / 1e-5) + max(gaps[5], 0.0)
    return (npass, -dist, int(sid), pk, of)


def select_diverse_pool(scored, flattener, base_vec, topk):
    """🔴 池子按**权重空间离散度**选（不再按分数取前 N）。

    贪心 max-min（k-center）：
      ① 先取分数最好的臂作种子；
      ② 之后每轮取「到已选集合的**最小权重距离**最大」的臂（并列时分数高者优先）。
    ⇒ 池里每个臂都**代表一个不同的方向**，互汤才有信息量
      （原版按分数取前 24 ⇒ 两两 maxΔ ≤ 0.0033 ⇒ 互汤 = 自交）。
    """
    cands = []
    for row in scored:
        if row[0] < MIN_NPASS:
            continue
        vec = flattener.flatten(row[2], 1.0)
        if vec is None:
            continue
        # 🔴 **活带窗口**（实测）：候选臂必须落在 [MIN_ARM_DELTA, MAX_ARM_DELTA]。
        #   · maxΔ < 0.012 ⇒ 读数与基线逐位相同 ⇒ 纯烧 GPU；
        #   · maxΔ > 0.16 ⇒ 实测 [0.20,0.30) 区间 nano 128/128 全负 ⇒ 只有变差的方向。
        #   不设窗口时 k-center 贪心会把池子推到**塌陷区**（实测 24/24 全被上界剪掉 ⇒ 出 0 条）。
        mx = WS.max_delta(vec, base_vec)[0]
        if mx < MIN_ARM_DELTA or mx > MAX_ARM_DELTA:
            continue
        cands.append((row, vec))
    if not cands:
        return []
    cands.sort(key=lambda c: (c[0][0], c[0][1], -c[0][2]), reverse=True)
    pool = [cands[0]]
    while len(pool) < topk:
        best, best_d = None, -1.0
        for row, vec in cands:
            if any(row[2] == p[0][2] for p in pool):
                continue
            d = min(WS.max_delta(vec, p[1])[0] for p in pool)
            if d > best_d + 1e-12 or (abs(d - best_d) <= 1e-12 and best is not None
                                     and (row[0], row[1], -row[2]) > (best[0][0], best[0][1], -best[0][2])):
                best, best_d = (row, vec), d
        if best is None or best_d < 1e-9:
            break
        pool.append(best)
    return pool


def emit_flat(acc):
    """把展平后的权重表变成一条配方串（权重和精确 = 1，末位取残差）。"""
    items = [(t, e, w) for (t, e), w in acc.items() if w >= WS.FLAT_MIN_W]
    if len(items) < 2:
        return None
    items.sort(key=lambda x: (-x[2], x[0]))
    tot = sum(w for _t, _e, w in items)
    if tot <= 0:
        return None
    items = [(t, e, w / tot) for t, e, w in items]
    head = items[:-1]
    last_t, last_e, _last_w = items[-1]
    acc_w = 1.0 - sum(w for _t, _e, w in head)
    if acc_w < 0:
        return None
    parts = ["%d:%d:%.6f" % (t, e, w) for t, e, w in head]
    parts.append("%d:%d:%.6f" % (last_t, last_e, acc_w))
    if any(float(x.rsplit(":", 1)[1]) < 0 for x in parts):
        return None
    return ",".join(parts)


def validate(rec, flattener, armed):
    mem = WS.parse_recipe(rec)
    if len(mem) < 2:
        return False, "members<2"
    tot = sum(w for _t, _e, w in mem)
    # %.6f 打印 ⇒ n 个成员最多偏 n*5e-7（15 成员 ⇒ ~7.5e-6）⇒ 容差取 2e-5
    if abs(tot - 1.0) > 2e-5:
        return False, "sum!=1 (%r)" % tot
    if any(w < 0 for _t, _e, w in mem):
        return False, "negative weight"
    if len(set((t, e) for t, e, _w in mem)) != len(mem):
        return False, "dup member"
    for t, _e, _w in mem:
        if flattener.is_soup_member(t):
            return False, "SOUP-OF-SOUP member %d" % t
    sig = (tuple(sorted((t, e) for t, e, _w in mem)),
           tuple(sorted(round(w, 4) for _t, _e, w in mem)))
    if sig in armed:
        return False, "already synthesized"
    return True, "ok"


def main():
    chan = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    led = WS.load_ledger(LEDGER)
    base = led["baseline_v2"]
    arms = led.get("arms") or {}
    flattener = WS.Flattener(arms)
    base_vec = flattener.flatten(base["trial"], 1.0)
    scored = []
    for sid, st in arms.items():
        row = score_arm(sid, st, base)
        if row is not None:
            scored.append(row)
    scored.sort(key=lambda t: (t[0], t[1], -t[2]), reverse=True)
    pool = select_diverse_pool(scored, flattener, base_vec, topk)
    log("arms=%d scored=%d pool=%d (min_npass=%d) base_trial=%s floor=%.3f garnish=%.3f "
        "arm_window=[%.3f,%.3f]" % (
            len(arms), len(scored), len(pool), MIN_NPASS, base.get("trial"), FLOOR, GARNISH_W,
            MIN_ARM_DELTA, MAX_ARM_DELTA))
    if len(pool) < 2:
        log("pool<2 ⇒ 不生成（前沿读数不足，等门禁回读）")
        return

    armed = set()
    for sid, st in arms.items():
        sm = st.get("soup_members")
        sw = st.get("soup_weights")
        if sm and sw and len(sm) == len(sw):
            armed.add((tuple(sorted((int(m[0]), int(m[1])) for m in sm)),
                       tuple(sorted(round(float(x), 4) for x in sw))))

    # 🔴 只配对「彼此相距够远」的臂（否则互汤 = 自交）
    pairs = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            d = WS.max_delta(pool[i][1], pool[j][1])[0]
            if d < MIN_PAIR_DELTA:
                continue
            pairs.append((d, pool[i], pool[j]))
    pairs.sort(key=lambda p: -p[0])
    pairs = pairs[chan::4]
    log("diverse_pairs=%d (chan %d, min_pair_delta=%.3f)" % (len(pairs), chan, MIN_PAIR_DELTA))

    out, seen = [], set()
    for (d, a, b) in pairs[:MAX_OUT]:
        for aw in MIX_RATIOS:
            bw = 1.0 - GARNISH_W - aw
            if bw <= 0:
                continue
            acc = dict(a[1])
            for k, v in b[1].items():
                acc[k] = acc.get(k, 0.0) + v
            if GARNISH_W > 0:
                gk = (GARNISH[0], GARNISH[1])
                acc[gk] = acc.get(gk, 0.0) + GARNISH_W
                hosts = [k for k, v in acc.items() if v >= GARNISH_W and k != gk]
                if not hosts:
                    continue
                host = max(hosts, key=lambda k: acc[k])
                acc[host] -= GARNISH_W
            rec = emit_flat(acc)
            if not rec or rec in seen:
                continue
            ok, reason = validate(rec, flattener, armed)
            if not ok:
                log("reject %s (%s)" % (rec, reason))
                continue
            seen.add(rec)
            out.append(rec)
    # 🔴 分辨率地板：maxΔ < FLOOR 的配方一律丢弃（读数必然与基线逐位相同 ⇒ 纯烧 GPU）
    kept, dropped = WS.prune_below_floor(out, base_vec, flattener, FLOOR)
    # 🔴 上界护栏：实测 [0.20, 0.30) 区间 nano **128/128 全负** ⇒ 超出 MAX_DELTA 的也丢
    over = []
    kept2 = []
    for rec, mx in kept:
        if mx is not None and mx > MAX_DELTA:
            over.append((rec, mx))
        else:
            kept2.append((rec, mx))
    kept = kept2
    dropped = dropped + over
    log("dropped_above_max_delta=%d (max_delta=%.3f)" % (len(over), MAX_DELTA))
    log("emitted=%d kept=%d dropped=%d (max_deltas=%s)" % (
        len(out), len(kept), len(dropped),
        ",".join("%.4f" % mx for _r, mx in kept[:8])))
    print(";".join(r for r, _mx in kept))


if __name__ == "__main__":
    main()
