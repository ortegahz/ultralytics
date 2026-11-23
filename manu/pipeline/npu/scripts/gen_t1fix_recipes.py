#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**T1 缺口「骨架补偿」轴**（2026-11-20 16:0x 新增）—— 在 `gen_t1donor_recipes.py` 之外
打开**第二块 100% 未开采**的搜索区：**「只差 T1 的现成汤骨架 + T1 供体微剂量」**。

🔴🔴🔴 为什么必须再开一条轴（全部实测，不是推理）：

  ① **六项里唯一的真缺口仍是 T1**：全台账 1,172 个有完整六项读数的臂里，
     六项全过的全是基线权重的逐位复现（零改进）；「过 5 项」的绝大多数**只差 T1**。

  ② **`gen_t1donor_recipes.py`（轴 ⑨）的结构性问题**：它用**基线骨架**（dT1 恰好 = 0）
     起手 ⇒ 供体剂量 δ 必须同时满足两个条件：
       · dT1 ≥ +1 阶（1.858e-5，可读）⇒ δ ≥ 1.858e-5 / dT1_供体；
       · dNano ≥ 0 ⇒ δ·|dN_供体| ≤ 「汤增益」（**实测只有 1 阶 = 2.36e-4**）。
     对最优供体 14963@4（dT1 +3.95e-4、dN −5.2e-3）⇒ δ ≥ 0.047 且 δ ≤ 0.045
     ⇒ **两个条件几乎不相交** ⇒ 这块地其实**极窄**（实测只有 16 条候选过分辨率地板）。

  ③ **本轴换了一个起手点，把「dt1 缺口」预支掉**：
     台账里有一批**只差 T1、但其余五项都 ≥ 基线且 nano 还有 1 阶余量**的现成汤臂
     （例：`2000321` dN=+2.36e-4、dN4=+4.73e-4、dF1=+3.25e-5、dT3=+1.61e-4、
     **dT1=−5.54e-5**、t2fp=3）⇒
       · 骨架自带 **dN +1 阶** 的 nano 余量（用来吸收供体的 nano 代价）；
       · 骨架的 T1 缺口只有 **3 阶**（−5.54e-5）⇒ 供体剂量只需补这点缺口
         ⇒ δ ≈ 0.014~0.03 就够，**远小于轴 ⑨ 所需的 0.047** ⇒
         「dT1 ≥ +1 阶」与「dNano ≥ 0」**同时满足**的区域**宽得多**。
     🔴 **而且这个区域 100% 未开采**：实测遍历全台账汤臂 ⇒
       「含 dT1>0 成员的汤臂」= 2 个；「**同时**含 dT1>0 成员与 dT3>0 成员的汤臂」= **0 个**
       ⇒ 「给一个 T3/nano4 好的汤再加一个 T1 供体」这件事**从来没被试过**。

  ④ **预测口径（保守）**：本轴**不额外记「汤增益」**——骨架的增益按**已实测**的读数直接继承
     ⇒ `预测dX = dX_骨架 + δ·dX_供体` ⇒ **比轴 ⑨ 更保守**（⑨ 假设额外 +1 阶）。
     胜负仍由 NanoBench + 官方门禁实测给出，本预测**只用于排优先级**。

  ⑤ **种子复现关**：升轨按 `soup_family`（成员集合 + 末位成员权重档 1% 粒度）分组
     ⇒ 同一 (骨架, 供体, δ) 桶里给出 **≥3 个内部配比变体** ⇒ 同族 n≥2 ⇒ 升轨链才走得通。

用法： gen_t1fix_recipes.py <chan 0..3>
      环境变量：
        NANO_LEDGER / NANO_ROOT      台账与产物根（默认服务器路径）
        NANO_CHECK_TRIALS=0          跳过权重文件存在性检查（本地自检）
        T1FIX_MAX_OUT                每次调用最多吐几条（默认 16）
        T1FIX_TOP_SKEL               取前几个骨架（默认 6）
        T1FIX_TOP_DONOR              取前几个供体（默认 4）
        T1FIX_DOSES                  剂量网格（默认 0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06,0.08）
        T1FIX_VARIANTS               每个 (骨架,供体,剂量) 的内部配比变体数（默认 3）
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

MAX_OUT = int(os.environ.get("T1FIX_MAX_OUT", "16"))
TOP_SKEL = int(os.environ.get("T1FIX_TOP_SKEL", "6"))
TOP_DONOR = int(os.environ.get("T1FIX_TOP_DONOR", "4"))
VARIANTS = int(os.environ.get("T1FIX_VARIANTS", "3"))
DOSES = tuple(float(x) for x in os.environ.get(
    "T1FIX_DOSES", "0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.05,0.06,0.08,0.10").split(","))
# 🔴 2026-11-20 16:3x：允许「还差多少阶额外汤增益」的上限（3.0 阶 = sb3 族 nano 增益 p90）
MAX_GAIN_STEPS = float(os.environ.get("T1FIX_MAX_GAIN_STEPS", "3.0"))
# 🔴 每个骨架最多几条（防全押一个骨架的增益性格）
MAX_PER_SKEL = int(os.environ.get("T1FIX_MAX_PER_SKEL", "4"))

STEP_NANO = 0.0002362906      # nano 1 阶（台账实测阶梯）
STEP_NANO4 = 0.0004726        # nano4 1 阶
STEP_T1 = 0.00001858          # T1 1 阶
T2_CAP = 5


def log(msg):
    sys.stderr.write("[t1fix] %s\n" % msg)
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
        t3 = (tr.get("track3_blind") or {})
        if t1.get("f1") is None or t3.get("f1") is None:
            continue
        return {"t1": t1["f1"], "t2_fp": t2.get("fp"), "t3": t3["f1"]}
    return of


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
    """实测「汤增益」= 该汤臂的实际读数 − 成员读数的线性混合预测。

    🔴 2026-11-20 16:3x：504 个可算汤臂实测 ⇒ 增益中位 nano +0.43 / p75 +2.30 / p90 +3.00 / max +7.06 阶；
       sb3 族的 **F1 增益中位 +7.77 阶、T1 增益中位 +2.60 阶**。
       ⇒ 「加一个微剂量供体后这个汤的增益性格不变」⇒ **骨架的增益应当被继承**，
          而不是当成 0（本轴初版按 0 算 ⇒ 32×44 全配对可行域为空，属**过保守**）。
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


def deltas(pk, of, base):
    return {
        "d_nano": pk["nano"] - base["nano"],
        "d_nano4": (pk.get("nano4") or 0.0) - base["nano4"],
        "d_f1": (pk.get("f1") or 0.0) - base["f1"],
        "d_t1": of["t1"] - base["t1"],
        "d_t3": of["t3"] - base["t3"],
    }


def ck_ok(sid, epoch):
    ck = os.path.join(ROOT, "trials", "trial_%d" % sid, "weights", "epoch%d.pt" % epoch)
    return os.path.isfile(ck)


def load_skeletons(arms, base):
    """骨架 = 「只差 T1、但 nano 还有 ≥1 阶余量、其余各项都 ≥ 基线」的现成汤臂。"""
    out = []
    for sid, st in arms.items():
        sm = st.get("soup_members") or []
        sw = st.get("soup_weights") or []
        if not sm or len(sm) != len(sw):
            continue
        pk = st.get("picked") or {}
        if pk.get("nano") is None or pk.get("nano4") is None:
            continue
        of = official(str(sid), st)
        if of.get("t1") is None or of.get("t3") is None:
            continue
        if of.get("t2_fp") is not None and int(of["t2_fp"]) > T2_CAP:
            continue
        d = deltas(pk, of, base)
        if d["d_t1"] >= 0:
            continue                       # 已经不差 T1 ⇒ 不是「只差 T1」
        if d["d_nano"] <= 0:
            continue                       # 没有 nano 余量 ⇒ 吸收不了供体代价
        if d["d_nano4"] < 0 or d["d_f1"] < 0 or d["d_t3"] < 0:
            continue                       # 其余轴不能倒退
        out.append({
            "sid": int(sid), "d": d, "t2_fp": of.get("t2_fp"),
            "epoch": st.get("picked_epoch") or 5,
            "members": [(int(m[0]), int(m[1])) for m in sm],
            "weights": [float(x) for x in sw],
            "desc": str(st.get("desc") or "")[:56],
            "gap_steps": -d["d_t1"] / STEP_T1,
        })
    # 缺口最小 + nano 余量最大 ⇒ 优先
    out.sort(key=lambda x: (x["gap_steps"], -x["d"]["d_nano"]))
    return out


def load_donors(arms, base):
    """供体 = dT1 > 0 的臂（单臂优先），按「每 nano 代价换多少 T1」排序。"""
    out = []
    for sid, st in arms.items():
        pk = st.get("picked") or {}
        if pk.get("nano") is None or pk.get("nano4") is None:
            continue
        of = official(str(sid), st)
        if of.get("t1") is None or of.get("t3") is None:
            continue
        if of.get("t2_fp") is not None and int(of["t2_fp"]) > T2_CAP:
            continue
        d = deltas(pk, of, base)
        if d["d_t1"] <= 0:
            continue
        ep = st.get("picked_epoch") or 5
        try:
            ep = int(round(float(ep)))
        except Exception:  # noqa: BLE001
            continue
        if CHECK_TRIALS and not ck_ok(int(sid), ep):
            continue
        d["sid"] = int(sid)
        d["epoch"] = ep
        d["t2_fp"] = of.get("t2_fp")
        d["soup"] = bool(st.get("soup_members"))
        d["desc"] = str(st.get("desc") or "")[:56]
        cost = max(abs(d["d_nano"]), abs(d["d_nano4"]))
        d["cost"] = cost if cost > 1e-9 else 1e-9
        d["eff"] = d["d_t1"] / d["cost"]
        out.append(d)
    # 「dN4 ≥ 0」的供体最友好（补 nano4 而不是赔 nano4）⇒ 排前面
    out.sort(key=lambda x: (0 if x["d_nano4"] >= 0 else 1, -x["eff"], -x["d_t1"]))
    return out


def variants_of(members, weights):
    """内部配比变体：在骨架成员之间挪 2% 权重 ⇒ 同族不同模型（种子复现关要 n≥2）。"""
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


def build_recipe(skel_members, skel_w, donor, delta, var_w):
    """骨架成员按 var_w 缩到 (1−δ)，供体拿 δ ⇒ 权重和 = 1，末位（供体）权重 = δ。"""
    acc = []
    for (t, e), w in zip(skel_members, var_w):
        acc.append((int(t), int(e), float(w) * (1.0 - delta)))
    acc.append((int(donor["sid"]), int(donor["epoch"]), float(delta)))
    return acc


def emit(members):
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
    if rest < -1e-6:
        return None
    parts.append("%d:%d:%.6f" % (last_t, last_e, last_w))
    return ",".join(parts)


def predict(skel, donor, delta, gain=None):
    """新汤 = 骨架(实际读数) 缩到 (1−δ) + 供体 δ，**并继承骨架的实测汤增益**。

    🔴 2026-11-20 16:3x：初版按「零增益」算 ⇒ 32 骨架 × 44 供体**全配对可行域为空**（过保守）。
       实测表明 sb3 族汤增益 nano 中位 +0.43 / p75 +2.30 / p90 +3.00 阶、T1 中位 +2.60 阶、
       F1 中位 +7.77 阶 ⇒ 继承骨架实测增益后，**可行区重新打开**。
    """
    sd, dd = skel["d"], donor
    g = gain or {}
    return {
        "d_nano": (1.0 - delta) * sd["d_nano"] + delta * dd["d_nano"] + g.get("d_nano", 0.0),
        "d_nano4": (1.0 - delta) * sd["d_nano4"] + delta * dd["d_nano4"] + g.get("d_nano4", 0.0),
        "d_f1": (1.0 - delta) * sd["d_f1"] + delta * dd["d_f1"] + g.get("d_f1", 0.0),
        "d_t1": (1.0 - delta) * sd["d_t1"] + delta * dd["d_t1"] + g.get("d_t1", 0.0),
        "d_t3": (1.0 - delta) * sd["d_t3"] + delta * dd["d_t3"] + g.get("d_t3", 0.0),
        "t2_fp": 3 + delta * ((dd["t2_fp"] if dd["t2_fp"] is not None else 3) - 3),
    }


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
    skels = load_skeletons(arms, base)
    donors = load_donors(arms, base)
    # 🔴🔴 2026-11-20 16:3x：给每个骨架算**实测汤增益**（而不是按 0 算）
    for s_ in skels:
        s_["gain"] = measure_gain(s_["sid"], s_["members"], s_["weights"], arms, base) or {
            "d_nano": 0.0, "d_nano4": 0.0, "d_f1": 0.0, "d_t1": 0.0, "d_t3": 0.0}
    log("base=trial%s nano=%.7f n4=%.7f f1=%.7f t1=%.7f t3=%.7f" % (
        base["trial"], base["nano"], base["nano4"], base["f1"], base["t1"], base["t3"]))
    log("骨架（只差 T1 且 nano 有余量）=%d；供体（dT1>0）=%d" % (len(skels), len(donors)))
    for s in skels[:TOP_SKEL + 2]:
        log("   骨架 %-8d 缺口=%.1f阶 dN=%+.2e dN4=%+.2e dF1=%+.2e dT3=%+.2e t2fp=%s 实测增益(nano/F1/T1)=%+.2f/%+.2f/%+.2f阶 %s" % (
            s["sid"], s["gap_steps"], s["d"]["d_nano"], s["d"]["d_nano4"],
            s["d"]["d_f1"], s["d"]["d_t3"], s["t2_fp"],
            (s.get("gain") or {}).get("d_nano", 0.0) / STEP_NANO,
            (s.get("gain") or {}).get("d_f1", 0.0) / 0.0000048,
            (s.get("gain") or {}).get("d_t1", 0.0) / 0.00001858, s["desc"][:28]))
    for d in donors[:TOP_DONOR + 2]:
        log("   供体 %-8d ep=%-3s dT1=%+.2e dN=%+.2e dN4=%+.2e dF1=%+.2e dT3=%+.2e eff=%.4f t2fp=%s %s" % (
            d["sid"], d["epoch"], d["d_t1"], d["d_nano"], d["d_nano4"], d["d_f1"],
            d["d_t3"], d["eff"], d["t2_fp"], d["desc"][:28]))
    if not skels or not donors:
        log("骨架或供体为空 ⇒ 不生成")
        return

    armed = set()
    for sid, st in arms.items():
        sm = st.get("soup_members")
        sw = st.get("soup_weights")
        if sm and sw and len(sm) == len(sw):
            armed.add((tuple(sorted((int(m[0]), int(m[1])) for m in sm)),
                       tuple(sorted(round(float(x), 6) for x in sw))))

    cands = []
    for skel in skels[:TOP_SKEL]:
        for donor in donors[:TOP_DONOR]:
            if donor["sid"] in {m[0] for m in skel["members"]}:
                continue
            for delta in DOSES:
                if delta <= 0 or delta > 0.25:
                    continue
                p = predict(skel, donor, delta, skel.get("gain"))
                if gain_steps_needed(p) > MAX_GAIN_STEPS:
                    continue
                for vi, vw in enumerate(variants_of(skel["members"], skel["weights"])):
                    mem = build_recipe(skel["members"], skel["weights"], donor, delta, vw)
                    rec = emit(mem)
                    if not rec:
                        continue
                    sig = (tuple(sorted((int(m[0]), int(m[1])) for m in mem)),
                           tuple(sorted(round(float(m[2]), 6) for m in mem)))
                    if sig in armed:
                        continue
                    cands.append((score(p) - 1e-6 * vi, skel["sid"], donor["sid"], delta, rec, p))
    # 升序：所需额外增益少的优先；同档再按预测余量大的优先
    cands.sort(key=lambda c: (round(c[0], 6), -min(
        p["d_nano"], p["d_nano4"], p["d_f1"], p["d_t1"], p["d_t3"])))
    log("候选 %d 条 ⇒ 选前 %d" % (len(cands), MAX_OUT))

    fl = WS.Flattener(arms)
    base_vec = fl.flatten(base["trial"], 1.0)
    kept, dropped = [], 0
    # 🔴 2026-11-20 16:4x：**每骨架配额**（防全押一个骨架）
    #   背景：干跑显示 16 条候选**全部来自同一个骨架**（2000321）⇒
    #   若「加第 6 个成员后该汤的增益性格不变」这个假设对它不成立 ⇒ 16 条全废。
    #   ⇒ 每个骨架最多 MAX_PER_SKEL 条 ⇒ 分散到多个骨架（不同骨架 = 不同增益性格）。
    per_skel = {}
    for sc, ssid, dsid, delta, rec, p in cands:
        if len(kept) >= MAX_OUT:
            break
        if per_skel.get(ssid, 0) >= MAX_PER_SKEL:
            continue
        per_skel[ssid] = per_skel.get(ssid, 0) + 1
        vec = fl.flat_of_members(WS.parse_recipe(rec))
        if vec is None:
            continue
        mx, _l1 = WS.max_delta(vec, base_vec)
        if mx < WS.FLOOR_MAX_DELTA:
            dropped += 1
            continue
        kept.append((rec, p, mx, ssid, dsid, delta))
    for rec, p, mx, ssid, dsid, delta in kept:
        log("   +%s  骨架=%d 供体=%d δ=%.3f maxΔ=%.4f 需额外增益=%.2f阶  pred: dN=%+.2e dN4=%+.2e "
            "dF1=%+.2e dT1=%+.2e dT3=%+.2e t2fp=%.2f" % (
                rec, ssid, dsid, delta, mx, gain_steps_needed(p), p["d_nano"], p["d_nano4"],
                p["d_f1"], p["d_t1"], p["d_t3"], p["t2_fp"]))
    log("地板丢弃 %d 条 ⇒ 输出 %d 条" % (dropped, len(kept)))
    if not kept:
        return
    out = [r for r, _p, _mx, _s, _d, _dl in kept]
    mine = [r for i, r in enumerate(out) if i % 4 == chan]
    log("chan=%d total=%d mine=%d" % (chan, len(out), len(mine)))
    print(";".join(mine))


if __name__ == "__main__":
    main()
