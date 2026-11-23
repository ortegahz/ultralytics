#!/usr/bin/env python3
"""守望用：自动维护门禁优先队列 .prio_sids，并回收僵死 claim。

2026-11-18 13:00 v4 —— 排序口径 =「近失锚点局部加密」：
  锚点定义：**只差一项**（T1✓/T2fp✓ 但 T3 差一点点）的臂 ⇒ 本例 1540016
            w=[640001@5:0.71, 1100296@1:0.10, 1100277@2:0.19]，T3 差 5.1e-6。
  锚点从**磁盘上的 gate_report.json** 取（台账 arms[*].official 缺 1540016 ⇒ 口径会错）。
  排序：① 同成员集且权重离锚点最近 ② 同成员集其他 ③ 其他成员集按 bench 边际。
  同时回收僵死 claim（按 /proc 里真实在跑的 --gate-sids 反推）。
"""
import glob
import json
import os
import re

W = "/mnt/data/siping/runs/npu20ms/nano_search"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
LEDGER = os.path.join(W, "nano_search_ledger.json")
PRIO = os.path.join(W, ".prio_sids")
CLAIM = os.path.join(W, ".gate_claim")
LINE = re.compile(r"sid=(\d+).*?fam=(\S+).*?m=([+-][0-9.]+)σ.*?三轴全过=(True|False)")
# 🟢 2026-11-18 15:35 **新方向波次提权**：严格轨（bw22 族）已升轨 ⇒ 门禁机时必须让位给
#   **新家族波**（h16/h17/h18/h19）⇒ 否则新家族臂永远排在 tier2 ⇒ 饿死。
#   规则：日志文件命中新波次 ⇒ tier = -1（插队到队首）；其余规则不变。
# 🟢 2026-11-18 16:45：加入 h3x（F1×T1 交汇窗口 h32/h33/h34 + 供体混合 h26 等）
NEW_WAVE = re.compile(r"soup3_(h(1[6-9]|2[0-9]|3[0-9])|relaxed)\.log$")


def running_sids():
    out = set()
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            c = open(p, "rb").read().decode("utf-8", "ignore").replace("\0", " ")
        except Exception:
            continue
        if "--gate-sids" not in c:
            continue
        toks = c.split()
        for i, t in enumerate(toks):
            if t == "--gate-sids" and i + 1 < len(toks):
                for s in toks[i + 1].split(","):
                    if s.strip().isdigit():
                        out.add(s.strip())
    return out


def mkey(members):
    return tuple(sorted((int(m[0]), int(m[1])) for m in members))


def read_gate(sid):
    f = os.path.join(ROOT, "gate", "NANO_%s_ep5_official" % sid, "gate_report.json")
    if not os.path.isfile(f):
        return None
    try:
        g = json.load(open(f))
    except Exception:
        return None
    t = g.get("tracks") or {}
    a = t.get("track1_uav") or {}
    b = t.get("track2_golden") or {}
    c = t.get("track3_blind") or {}
    t1 = (a.get("finetune_peak") or {}).get("f1")
    t2 = b.get("fp")
    t3 = c.get("f1")
    if t1 is None or t3 is None:
        return None
    return t1, t2, t3


def main():
    run = running_sids()
    freed = 0
    import time
    now = time.time()
    for f in glob.glob(os.path.join(CLAIM, "*")):
        sid = os.path.basename(f)
        if sid in run:
            continue
        if os.path.isdir(os.path.join(ROOT, "gate", "NANO_%s_ep5_official" % sid)):
            continue
        # ⚠️ 只回收「陈旧」claim（>10min）：刚 claim 但批次还没起跑的那 10 分钟窗口里
        #    回收会导致两个通道抢同一个臂（11-18 12:26 真实踩到：两批都是 1572010,...）
        try:
            age = now - os.path.getmtime(f)
        except OSError:
            continue
        if age < 600:
            continue
        try:
            os.remove(f)
            freed += 1
        except OSError:
            pass

    try:
        led = json.load(open(LEDGER))
    except Exception:
        led = {}
    base = led.get("baseline_v2") or {}
    arms = led.get("arms") or {}
    b1, b2, b3 = base.get("t1", 0), base.get("t2_fp", 0), base.get("t3", 0)
    btrial = int(base.get("trial", -1))

    # ---- 找锚点：六项里**只差一项**、且该项相对缺口最小（离过线最近的刀刃点）----
    bn, bn4, bf1 = base.get("nano", 0), base.get("nano4", 0), base.get("f1", 0)
    anchor = None
    for d in glob.glob(os.path.join(ROOT, "gate", "NANO_*_ep5_official")):
        sid = os.path.basename(d).split("_")[1]
        if not sid.isdigit() or int(sid) == btrial:
            continue
        g = read_gate(sid)
        if not g:
            continue
        t1, t2, t3 = g
        st = arms.get(sid) or {}
        pk = st.get("picked") or {}
        mem = st.get("soup_members")
        w = st.get("soup_weights")
        if not mem or not w or not pk:
            continue
        crit = [
            (pk.get("nano"), bn, "ge"),
            (pk.get("nano4"), bn4, "ge"),
            (pk.get("f1"), bf1, "ge"),
            (t1, b1, "ge"),
            (t2, b2, "le"),
            (t3, b3, "ge"),
        ]
        miss = []
        for val, ref, op in crit:
            if val is None:
                miss = None
                break
            ok = (val >= ref) if op == "ge" else (val <= ref)
            if not ok:
                miss.append(abs(val - ref) / max(abs(ref), 1e-12))
        if miss is None:
            continue
        # 锚点优先级：六项全过（最强，2026-11-18 13:15 起有 trial_1572002）> 只差一项（缺口越小越近）
        if len(miss) == 0:
            key = (0, 0.0)
        elif len(miss) == 1:
            key = (-1, miss[0])
        else:
            continue
        if anchor is None or key > anchor[0]:
            anchor = (key, mkey(mem), list(w), sid)
    akey = anchor[1] if anchor else None
    aw = anchor[2] if anchor else None

    # ---- 已门禁臂的权重向量（去重：同成员集同权重 ⇒ 结果必然相同 ⇒ 不重复占用门禁机时）----
    gated_w = set()
    for d in glob.glob(os.path.join(ROOT, "gate", "NANO_*_ep5_official")):
        sid = os.path.basename(d).split("_")[1]
        if not sid.isdigit():
            continue
        st = arms.get(sid) or {}
        mem = st.get("soup_members")
        w = st.get("soup_weights")
        if mem and w:
            gated_w.add((mkey(mem), tuple(round(float(x), 5) for x in w)))

    cand = {}
    for lg in glob.glob(os.path.join(W, "soup3*.log")):
        try:
            for line in open(lg, errors="ignore"):
                m = LINE.search(line)
                if not m:
                    continue
                sid, fam, mg, ok = m.group(1), m.group(2), float(m.group(3)), m.group(4) == "True"
                if sid in run:
                    continue
                # 🔴🔴 2026-11-19 10:35 **不信日志里的「三轴全过」标记，改用当前基线实时判定**
                #   坑 1：night_soup3 打那个标记用的是**严格 >**（nano/nano4），而升轨口径是 **>=**
                #        ⇒ 凡是 nano **恰好等于**基线的臂（SB3 家族在 .7242439 阶梯平台上**全都相等**）
                #        ⇒ 标记恒 False ⇒ 好臂一律被 refresh_prio 丢弃。
                #   坑 2：那个标记是「打**老基线** .6666 时代」留下的历史值 ⇒ .prio_sids 头部
                #        全是当时的三轴全过臂，而那些臂现在的 nano/nano4 **早已低于**新基线 ⇒ 死路臂。
                #   修法：用台账 picked 实时按 **>= 当前基线** 判定。
                _st0 = arms.get(sid) or {}
                _pk0 = _st0.get("picked") or {}
                ok = (_pk0.get("nano", 0) >= bn and _pk0.get("nano4", 0) >= bn4
                      and _pk0.get("f1", 0) >= bf1)
                if not ok:
                    continue
                if os.path.isdir(os.path.join(ROOT, "gate", "NANO_%s_ep5_official" % sid)):
                    continue
                st = arms.get(sid) or {}
                mem = st.get("soup_members")
                w = st.get("soup_weights")
                k = mkey(mem) if mem else ("fam", fam)
                if mem and w and (k, tuple(round(float(x), 5) for x in w)) in gated_w:
                    continue  # 同权重已门禁过 ⇒ 重复臂，跳过
                if NEW_WAVE.search(os.path.basename(lg)):
                    # 🟢 2026-11-18 15:35 提权组内部按**边际 m（σ）**排序（高者先）
                    #   理由：新方向臂里，**「nano/nano4/F1 三项同时优于基线」**的 Track-B 族边际最高
                    #   （m≈+0.7σ）⇒ 它们才是 v2.8 的真正候选 ⇒ 先打门禁；
                    #   冠军族邻域臂边际≈0 ⇒ 排在后面（它们已接近局部最优）。
                    tier, dist = -1, -mg
                elif aw and w and k == akey and len(w) == len(aw):
                    tier, dist = 0, round(sum(abs(x - y) for x, y in zip(w, aw)), 6)
                elif k == akey:
                    tier, dist = 1, 9.0
                else:
                    tier, dist = 2, 9.0
                cand[sid] = (tier, dist, -mg)
        except OSError:
            continue
    order = sorted(cand.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[1][2], -int(kv[0])))
    sids = [s for s, _ in order][:24]   # 🟢 15:40 12→24：提权后候选变多，队列头 24 个够三通道轮转
    tmp = PRIO + ".tmp"
    with open(tmp, "w") as f:
        f.write(",".join(sids))
    os.replace(tmp, PRIO)
    print("[refresh_prio] freed=%d live=%d anchor=%s w=%s prio=%s"
          % (freed, len(cand), anchor[3] if anchor else None, aw, ",".join(sids)))


if __name__ == "__main__":
    main()
