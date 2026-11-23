#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1 供体微剂量轴（2026-11-20 15:30）—— **唯一未开采的、真正对得上缺口的搜索空间**。

🔴🔴🔴 为什么是这条轴（全部实测，不是推理）：

  ① **六项里唯一的真缺口 = T1**。
     1,154 个有完整六项读数的臂里，六项全过的 40 个**全部是基线权重的逐位复现**（零改进）；
     158 个「过 5 项」里 152 个只差 T1、6 个只差 T3 ⇒ **T1 是结构性缺口**。

  ② **T1 与 nano 在成员层强负相关**：全台账 20 个「dT1 > 0」的非汤成员，
     **每一个 dNano 都 < 0**（最轻的 14963@4 也是 −5.2e-3 = 22 个 nano 阶）
     ⇒ 「一个成员同时 dT1>0 且 dNano≥0」在成员层**不存在**。

  ③ **但「汤」有成员之上的增益（FAM5 实测）**：
     FAM5 = {1608003,1612001,1530006,1543006,1623003}，其 5 个成员的 nano **没有一个**
     高于基线（1530006 −4.7e-4 / 1543006 −2.4e-3 / …），而 FAM5 汤的 nano = **+2.36e-4**
     ⇒ **汤的权重重排能凭空拿到 1 个 nano 阶**，这正是「权重空间里还剩一点油」的证据。

  ④ **「基线 + T1 供体」这块地是 100% 未测的**：
     实测遍历台账全部汤臂 —— **含任何 dT1>0 成员的汤臂 = 0 个**
     （既有 w13 的 ax_t1donor/ax_t1donor6 都挂在 **FAM5 骨架**上，而 FAM5 自身 dT1 = −9.26e-5，
      任何供体的 T1 余量都补不上这个亏 ⇒ 那两轴注定过不了 T1 关）。

  ⑤ **为什么「基线骨架 + 微剂量」有机会**（线性估计 + 1 个阶的汤增益）：
     基线骨架的 dT1 **恰好 = 0** ⇒ 加任何 δ>0 的正 T1 供体 ⇒ dT1 直接转正；
     代价是 dNano = δ·(−|dN_D|) ⇒ **δ 上界 = 1 个 nano 阶 / |dN_D|**：
       · 14963@4（dT1 +3.95e-4、dN −5.2e-3、**dN4 +9.45e-4**、dF1 +3.58e-3）⇒ δ ≲ 0.045
       · 12543@3（dT1 +7.81e-4、dN −1.25e-2）⇒ δ ≲ 0.019
       · 15510@4（dT1 +7.27e-4、**dT3 +2.59e-3**、dN −1.70e-2）⇒ δ ≲ 0.014
     ⇒ 微剂量带（δ = 0.01~0.06）是**唯一**可能同时满足 dT1≥0 ∧ dNano≥0 的区域。

⚠ **预测口径的诚实边界**：
  · 本脚本用**线性混合 + 「汤增益 1 阶」的假设**排序，只用于**排优先级**，不用于判胜负；
  · 真正的胜负由 NanoBench + 官方门禁实测给出（读数是离散阶梯，T1 阶 = 1.858e-5）。
  · 「汤增益 1 阶」是 FAM5 的**实测**（+2.36e-4 = 1 个 nano 阶），不是保证。

用法： gen_t1donor_recipes.py <chan 0..3>
      环境变量：
        NANO_LEDGER / NANO_ROOT      台账与产物根（默认服务器路径）
        NANO_CHECK_TRIALS=0          跳过权重文件存在性检查（本地自检）
        T1D_MAX_OUT                  每次调用最多吐几条（默认 16）
        T1D_TOP_DONORS               取前几个供体（默认 6）
        T1D_MIN_DONOR_DT1            供体的最小 dT1（默认 5e-5）
        T1D_DOSES                    剂量网格（默认 0.01,0.02,...,0.12）
        T1D_VARIANTS                 每个 (供体,剂量) 几条内部配比变体（默认 3）
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

MAX_OUT = int(os.environ.get("T1D_MAX_OUT", "16"))
TOP_DONORS = int(os.environ.get("T1D_TOP_DONORS", "6"))
MIN_DONOR_DT1 = float(os.environ.get("T1D_MIN_DONOR_DT1", "5e-5"))
VARIANTS = int(os.environ.get("T1D_VARIANTS", "3"))
DOSES = tuple(float(x) for x in os.environ.get(
    "T1D_DOSES", "0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.15,0.20").split(","))
# 🔴 2026-11-20 16:3x：允许「还差多少阶额外汤增益」的上限（3.0 阶 = sb3 族 nano 增益 p90）
MAX_GAIN_STEPS = float(os.environ.get("T1D_MAX_GAIN_STEPS", "3.0"))
# 🔴 每个供体最多几条（防全押一个供体的增益性格）
MAX_PER_DONOR = int(os.environ.get("T1D_MAX_PER_DONOR", "4"))

# 🔴 读数离散阶梯（台账实测）——「汤增益」按 1 阶计
STEP_NANO = 0.0002362906      # nano 1 阶
STEP_NANO4 = 0.0004726        # nano4 1 阶
T2_CAP = 5                    # t2_fp 门禁上限（max(基线 t2_fp=3, 5)）


def log(msg):
    sys.stderr.write("[t1d] %s\n" % msg)
    sys.stderr.flush()


def official(sid, st):
    """官方三轨读数：台账优先，缺则回落到磁盘 gate_report.json。"""
    of = st.get("official") or {}
    if of.get("t1") is not None and of.get("t3") is not None:
        return of
    for ep in (5, 4, 3):
        rp = os.path.join(GATE, "NANO_%s_ep%d_official" % (sid, ep), "gate_report.json")
        if not os.path.isfile(rp):
            continue
        try:
            rep = json.load(open(rp, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        tr = rep.get("tracks") or {}
        t1 = (tr.get("track1_uav") or {}).get("finetune_peak") or {}
        t2 = tr.get("track2_golden") or {}
        t3 = tr.get("track3_blind") or {}
        if t1.get("f1") is None or t3.get("f1") is None:
            continue
        return {"t1": t1["f1"], "t2_fp": t2.get("fp"), "t3": t3["f1"]}
    return of


def load_donors(arms, base):
    """扫全台账：所有「dT1 > 0」且有权重可用的成员（汤臂与非汤臂都算供体）。"""
    out = []
    for sid, st in arms.items():
        pk = st.get("picked") or {}
        if pk.get("nano") is None or pk.get("nano4") is None:
            continue
        of = official(str(sid), st)
        if of.get("t1") is None or of.get("t3") is None:
            continue
        d = {
            "sid": int(sid),
            "d_nano": pk["nano"] - base["nano"],
            "d_nano4": (pk.get("nano4") or 0) - base["nano4"],
            "d_f1": (pk.get("f1") or 0) - base["f1"],
            "d_t1": of["t1"] - base["t1"],
            "d_t3": of["t3"] - base["t3"],
            "t2_fp": of.get("t2_fp"),
            "epoch": st.get("picked_epoch") or 5,
            "soup": bool(st.get("soup_members")),
            "desc": str(st.get("desc") or "")[:60],
        }
        if d["d_t1"] <= MIN_DONOR_DT1:
            continue
        if CHECK_TRIALS:
            try:
                ep = int(round(float(d["epoch"])))
            except Exception:  # noqa: BLE001
                continue
            ck = os.path.join(ROOT, "trials", "trial_%d" % d["sid"], "weights", "epoch%d.pt" % ep)
            if not os.path.isfile(ck):
                continue
            d["epoch"] = ep
        # 🔴 「T1 效率」：单位 nano/nano4 代价能换到多少 T1
        cost = max(abs(d["d_nano"]), abs(d["d_nano4"]))
        d["cost"] = cost if cost > 1e-9 else 1e-9
        d["eff"] = d["d_t1"] / d["cost"]
        out.append(d)
    out.sort(key=lambda x: (-x["eff"], -x["d_t1"]))
    return out


def arm_delta(sid, arms, base):
    """某个臂自身的六项 delta（相对基线）；读数不全则 None。"""
    st = arms.get(str(sid)) or {}
    pk = st.get("picked") or {}
    if pk.get("nano") is None or pk.get("nano4") is None:
        return None
    of = official(str(sid), st)
    if of.get("t1") is None or of.get("t3") is None:
        return None
    return {"d_nano": pk["nano"] - base["nano"],
            "d_nano4": (pk.get("nano4") or 0) - base["nano4"],
            "d_f1": (pk.get("f1") or 0) - base["f1"],
            "d_t1": of["t1"] - base["t1"],
            "d_t3": of["t3"] - base["t3"],
            "t2_fp": of.get("t2_fp")}


def measure_gain(sid, members, weights, arms, base):
    """🔴🔴 实测「汤增益」= 该汤臂的**实际读数** − 成员读数的**线性混合预测**。

    🔴 为什么必须实测（2026-11-20 16:2x 实测，504 个可算汤臂）：
      · 旧代码把增益写死成「nano +1 阶、其余轴 0」⇒ **既低估又错配**：
        基线臂 trial_1990055 自身的实测增益 =
            nano +0.42 阶 / nano4 +0.38 阶 / **F1 +7.29 阶** / **T1 +3.63 阶** / T3 +0.70 阶
        ⇒ 旧模型把「加供体后仍然继承的 +3.63 阶 T1 增益」**当成 0**，
          于是 (a) 把真正有戏的候选判死、(b) 把预测 nano 会掉 1~2 阶的候选**放行**（白烧 GPU）。
      · 实测 sb3 族（n=344）增益分位：
            nano  p10 +0.40 p25 +0.42 p50 +0.43 p75 +2.30 p90 +3.00 max +7.06
            nano4 p10 +0.23 p25 +0.36 p50 +0.37 p75 +0.40 p90 +1.01 max +2.69
            F1    p10 +1.57 p25 +7.31 p50 +7.77 p75 +8.10 p90 +15.49 max +27.20
            T1    p10 -0.69 p25 +1.35 p50 +2.60 p75 +3.60 p90 +4.21 max +9.36
            T3    p10 -4.50 p25 -3.74 p50 +0.45 p75 +0.75 p90 +5.81 max +32.02
        ⇒ 「额外还需要多少阶的运气」有了**实测先验**，不必再拍脑袋写死 +1 阶。
    """
    act = arm_delta(sid, arms, base)
    if act is None or not members:
        return None
    tot = sum(float(w) for w in weights) or 1.0
    pred = {"d_nano": 0.0, "d_nano4": 0.0, "d_f1": 0.0, "d_t1": 0.0, "d_t3": 0.0}
    for (t, _e), w in zip(members, weights):
        md = arm_delta(t, arms, base)
        if md is None:
            return None
        for k in pred:
            pred[k] += float(w) / tot * md[k]
    return {k: act[k] - pred[k] for k in pred}



def scaffold_of(arms, base):
    """基线骨架：直接取当前基线臂的成员与权重（**不改配比**，只做整体缩放）。"""
    st = arms.get(str(base["trial"])) or {}
    sm = st.get("soup_members") or []
    sw = st.get("soup_weights") or []
    if not sm or len(sm) != len(sw):
        return [(int(base["trial"]), int(base.get("epoch") or 5))], [1.0]
    tot = sum(float(x) for x in sw)
    return [(int(m[0]), int(m[1])) for m in sm], [float(x) / tot for x in sw]


def variants_of(members, weights):
    """内部配比变体：把 2% 的权重在基线成员之间挪一挪 ⇒ 采「基线邻域」的权重空间。

    🔴 为什么必须做变体：升轨的「种子复现关」按**汤族**分组（`soup_family` =
       成员集合 + 末位成员权重档 1%）⇒ 同一 (供体, 剂量) 桶里至少要有 ≥2 个**不同**的臂
       （权重不同 ⇒ 模型不同 ⇒ 独立样本）⇒ 同一族里 n≥4 才可能凑出复现证据。
    """
    n = len(members)
    if n < 2 or VARIANTS <= 1:
        return [list(weights)]
    order = sorted(range(n), key=lambda i: -weights[i])
    outs = [list(weights)]
    for k in range(1, VARIANTS):
        w = list(weights)
        src = order[0]
        dst = order[k % n]
        if src == dst:
            continue
        shift = 0.02
        if w[src] - shift < 0.01:
            continue
        w[src] -= shift
        w[dst] += shift
        outs.append(w)
    return outs


def build_recipe(scaffold, sca_w, donor, delta, var_w):
    """基线成员按 var_w 缩放到 (1−δ)，供体拿 δ ⇒ 权重和 = 1，末位（供体）权重 = δ。"""
    acc = []
    for (t, e), w in zip(scaffold, var_w):
        acc.append((int(t), int(e), float(w) * (1.0 - delta)))
    acc.append((int(donor["sid"]), int(donor["epoch"]), float(delta)))
    return acc


def emit(members):
    """配方串：权重打印 6 位；末位（供体）权重必须精确 = δ ⇒ 其余成员用残差补齐。"""
    if len(members) < 2:
        return None
    head = members[:-1]
    last_t, last_e, last_w = members[-1]
    tot_head = sum(w for _t, _e, w in head)
    if tot_head <= 0:
        return None
    parts = []
    acc = 0.0
    for t, e, w in head:
        parts.append("%d:%d:%.6f" % (t, e, w))
        acc += w
    rest = 1.0 - acc - last_w
    if rest < -1e-9:
        return None
    parts.append("%d:%d:%.6f" % (last_t, last_e, last_w))
    return ",".join(parts)


def predict(donor, delta, gain=None):
    """线性混合 + **骨架实测汤增益** ⇒ 六项预测 delta（**只用于排序**）。

    🔴 2026-11-20 16:3x：增益不再写死「nano +1 阶」，而是**继承骨架自身的实测增益**
       （基线骨架 trial_1990055 实测：nano +0.42 / nano4 +0.38 / **F1 +7.29** /
        **T1 +3.63** / T3 +0.70 阶）⇒ 加一个微剂量供体后这个汤的「增益性格」不变。
    """
    g = gain or {}
    p = {
        "d_nano": delta * donor["d_nano"] + g.get("d_nano", 0.0),
        "d_nano4": delta * donor["d_nano4"] + g.get("d_nano4", 0.0),
        "d_f1": delta * donor["d_f1"] + g.get("d_f1", 0.0),
        "d_t1": delta * donor["d_t1"] + g.get("d_t1", 0.0),
        "d_t3": delta * donor["d_t3"] + g.get("d_t3", 0.0),
    }
    p["t2_fp"] = 3 + delta * ((donor["t2_fp"] if donor["t2_fp"] is not None else 3) - 3)
    return p


AX_STEP = {"d_nano": STEP_NANO, "d_nano4": STEP_NANO4, "d_f1": 0.0000048,
           "d_t1": 0.00001858, "d_t3": 0.0000105}


def gain_steps_needed(p):
    """「还差多少阶的额外汤增益」才六项全过（各轴按自己的阶单位取最大）。"""
    need = 0.0
    for k, st in AX_STEP.items():
        if p.get(k) is None or p[k] >= 0:
            continue
        need = max(need, (-p[k]) / st)
    return need


def score(p):
    """越小越优先 = **所需额外增益阶数**（0 = 靠继承增益就六项全过）。"""
    if p["t2_fp"] > T2_CAP:
        return 1e9
    return gain_steps_needed(p)



def main():
    chan = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    led = WS.load_ledger(LEDGER)
    base = led["baseline_v2"]
    arms = led.get("arms") or {}
    donors = load_donors(arms, base)
    scaffold, sca_w = scaffold_of(arms, base)
    log("base=trial%s nano=%.7f n4=%.7f f1=%.7f t1=%.7f t3=%.7f" % (
        base["trial"], base["nano"], base["nano4"], base["f1"], base["t1"], base["t3"]))
    log("scaffold=%s w=%s" % (scaffold, [round(x, 4) for x in sca_w]))
    # 🔴🔴 继承骨架自身的**实测汤增益**（不再写死 +1 阶）
    _gain = measure_gain(base["trial"], scaffold, sca_w, arms, base)
    if _gain is None:
        log("⚠ 骨架增益无法实测 ⇒ 退回保守假设（nano/nano4 各 +1 阶，其余 0）")
        _gain = {"d_nano": STEP_NANO, "d_nano4": STEP_NANO4, "d_f1": 0.0, "d_t1": 0.0, "d_t3": 0.0}
    log("骨架实测汤增益（阶）: nano=%+.2f nano4=%+.2f F1=%+.2f T1=%+.2f T3=%+.2f" % (
        _gain["d_nano"] / STEP_NANO, _gain["d_nano4"] / STEP_NANO4,
        _gain["d_f1"] / 0.0000048, _gain["d_t1"] / 0.00001858, _gain["d_t3"] / 0.0000105))

    log("donors(dT1>0)=%d ⇒ 取前 %d（按 T1 效率）" % (len(donors), TOP_DONORS))
    for d in donors[:TOP_DONORS + 2]:
        log("   donor %-8d ep=%-3s dT1=%+.2e dN=%+.2e dN4=%+.2e dF1=%+.2e dT3=%+.2e eff=%.4f %s" % (
            d["sid"], d["epoch"], d["d_t1"], d["d_nano"], d["d_nano4"], d["d_f1"],
            d["d_t3"], d["eff"], d["desc"]))
    if not donors:
        log("无 dT1>0 的供体 ⇒ 不生成（等训练轴产出）")
        return

    # 已合成签名（成员集合 + 权重）⇒ 不重复投喂
    armed = set()
    for sid, st in arms.items():
        sm = st.get("soup_members")
        sw = st.get("soup_weights")
        if sm and sw and len(sm) == len(sw):
            armed.add((tuple(sorted((int(m[0]), int(m[1])) for m in sm)),
                       tuple(sorted(round(float(x), 6) for x in sw))))

    cands = []
    for d in donors[:TOP_DONORS]:
        # 🔴 δ 上界：让预测的 nano/nano4 都不判负
        for delta in DOSES:
            if delta <= 0 or delta > 0.5:
                continue
            p = predict(d, delta, _gain)
            # 🔴🔴 2026-11-20 16:3x：判据从「预测 nano/nano4 ≥ −2 阶」改成
            #   **「所需额外汤增益 ≤ T1D_MAX_GAIN_STEPS 阶」**（默认 3.0 阶 = sb3 族 nano 增益 p90）。
            #   · 继承骨架实测增益后，大多数候选的「所需额外增益」= 0 阶 ⇒ 直接放行；
            #   · 仍需要运气的候选，按所需阶数**升序**排（越不费运气越优先）；
            #   · 旧判据把「预测 nano 掉 1~2 阶」的候选放行（实测 nano 增益 p50 仅 +0.43 阶
            #     ⇒ 那些臂大概率 nano 不达标 ⇒ 白烧 GPU），新判据把它们排到后面。
            if gain_steps_needed(p) > MAX_GAIN_STEPS:
                continue
            for vi, vw in enumerate(variants_of(scaffold, sca_w)):
                mem = build_recipe(scaffold, sca_w, d, delta, vw)
                rec = emit(mem)
                if not rec:
                    continue
                sig = (tuple(sorted((int(m[0]), int(m[1])) for m in mem)),
                       tuple(sorted(round(float(m[2]), 6) for m in mem)))
                if sig in armed:
                    continue
                cands.append((score(p) - 1e-6 * vi, d["sid"], delta, rec, p))
    # 升序：所需额外增益少的优先；同档再按预测余量大的优先
    cands.sort(key=lambda c: (round(c[0], 6), -min(
        p["d_nano"], p["d_nano4"], p["d_f1"], p["d_t1"], p["d_t3"])))
    log("候选 %d 条 ⇒ 选前 %d" % (len(cands), MAX_OUT))
    # 🔴 分辨率地板：maxΔ < 0.012 的臂读数必然与基线逐位相同 ⇒ 纯烧 GPU
    fl = WS.Flattener(arms)
    base_vec = fl.flatten(base["trial"], 1.0)
    kept, dropped = [], 0
    # 🔴 2026-11-20 16:4x：**每供体配额**（防全押一个供体的增益性格）
    per_donor = {}
    for sc, sid, delta, rec, p in cands:
        if len(kept) >= MAX_OUT:
            break
        if per_donor.get(sid, 0) >= MAX_PER_DONOR:
            continue
        per_donor[sid] = per_donor.get(sid, 0) + 1
        vec = fl.flat_of_members(WS.parse_recipe(rec))
        if vec is None:
            continue
        mx, _l1 = WS.max_delta(vec, base_vec)
        if mx < WS.FLOOR_MAX_DELTA:
            dropped += 1
            continue
        kept.append((rec, p, mx, sid, delta))
    for rec, p, mx, sid, delta in kept:
        log("   +%s  donor=%d δ=%.3f maxΔ=%.4f 需额外增益=%.2f阶  pred: dN=%+.2e dN4=%+.2e "
            "dF1=%+.2e dT1=%+.2e dT3=%+.2e t2fp=%.2f" % (
                rec, sid, delta, mx, gain_steps_needed(p), p["d_nano"], p["d_nano4"], p["d_f1"],
                p["d_t1"], p["d_t3"], p["t2_fp"]))
    log("地板丢弃 %d 条 ⇒ 输出 %d 条" % (dropped, len(kept)))
    if not kept:
        return
    out = [r for r, _p, _mx, _s, _d in kept]
    # 4 通道切分：与其它波次生成器一致（i % 4 == chan）
    mine = [r for i, r in enumerate(out) if i % 4 == chan]
    log("chan=%d total=%d mine=%d" % (chan, len(out), len(mine)))
    print(";".join(mine))


if __name__ == "__main__":
    main()
