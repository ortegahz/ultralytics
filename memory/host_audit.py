#!/usr/bin/env python3
"""host 侧独立验收器：按**真实台账 schema** 复算「六项全过 / 同家族计数 / 候选排序」，
   不依赖服务器上的 promote_candidate.py，用于晨报复核（防止自动升轨误判）。

台账 schema（实测）：
  arms[tid] = {trial, seed, desc, params, trained, nanobench:{ep:{...}},
               picked_epoch, picked:{nano,nano4,f1,...},
               official:{verdict,t1,t2_fp,t3,official_ok,deliver_ok},
               board_ms, speed_ok, dual:{...}, family?, members?}
  顶层：baseline_v2{...} / beat_to_pass{...} / deliver_criteria{...}

用法: python3 host_audit.py <ledger.json> [--out report.json] [--top 25]
"""
from __future__ import annotations
import argparse, json, math

BOARD_LIMIT = 19.989
FALLBACK = {"nano": 0.715737, "nano4": 0.650756, "f1": 0.890043,
            "t1": 0.890013, "t3": 0.948526, "t2_fp": 2}


def pick(d, names):
    for n in names:
        if isinstance(d, dict) and isinstance(d.get(n), (int, float)):
            return float(d[n])
    return None


def arm_metrics(arm):
    """⇒ {nano,nano4,f1,t1,t2_fp,t3,board_ms}，缺项为 None"""
    nb = arm.get("picked")
    if not isinstance(nb, dict):
        pe = arm.get("picked_epoch")
        nball = arm.get("nanobench")
        nb = nball.get(str(pe), {}) if isinstance(nball, dict) and pe is not None else {}
    if not isinstance(nb, dict):
        nb = {}
    off = arm.get("official")
    off = off if isinstance(off, dict) else {}
    m = {
        "nano":  pick(nb, ["nano", "nano_recall"]),
        "nano4": pick(nb, ["nano4", "nano_4"]),
        "f1":    pick(nb, ["f1", "F1"]),
        "t1":    pick(off, ["t1", "T1"]),
        "t2_fp": pick(off, ["t2_fp", "T2_fp"]),
        "t3":    pick(off, ["t3", "T3"]),
        "board_ms": pick(arm, ["board_ms", "t4_ms", "board_measure_ms", "latency_ms"]),
    }
    return m


def thresholds(ledger):
    """门槛取自「当前基线」自身指标（live base 口径），缺失才回落到固定值。"""
    base = ledger.get("baseline_v2") if isinstance(ledger, dict) else None
    th = dict(FALLBACK)
    src = {}
    if isinstance(base, dict):
        for k in ("nano", "nano4", "f1", "t1", "t3", "t2_fp"):
            v = pick(base, [k])
            if v is not None:
                th[k] = v
                src[k] = "baseline_v2"
    btp = ledger.get("beat_to_pass") if isinstance(ledger, dict) else None
    if isinstance(btp, dict):
        m = {"nano": "nano_min", "f1": "f1_min", "t1": "t1_min",
             "t3": "t3_min", "t2_fp": "t2_fp_max"}
        for k, kk in m.items():
            if src.get(k) != "baseline_v2" and isinstance(btp.get(kk), (int, float)):
                th[k] = float(btp[kk])
                src[k] = "beat_to_pass"
    dc = ledger.get("deliver_criteria") if isinstance(ledger, dict) else None
    if isinstance(dc, dict):
        for k, kk in {"t1": "t1_ge", "t3": "t3_ge", "t2_fp": "t2_fp_le"}.items():
            if src.get(k) not in ("baseline_v2",) and isinstance(dc.get(kk), (int, float)):
                th[k] = float(dc[kk])
                src[k] = "deliver_criteria"
    return th, src


def family_of(arm, tid):
    for k in ("family", "family_key", "soup_family", "fam"):
        v = arm.get(k)
        if isinstance(v, str) and v:
            return v
    p = arm.get("params")
    if isinstance(p, dict):
        for k in ("family", "family_key", "soup_family"):
            v = p.get(k)
            if isinstance(v, str) and v:
                return v
    return "unknown"


def signature_of(arm, tid):
    """同一模型只算一次：汤按（成员集合 + 权重），普通臂按 seed。"""
    for k in ("signature", "dedup_key", "model_signature", "soup_signature"):
        v = arm.get(k)
        if isinstance(v, str) and v:
            return v
    for k in ("members", "soup_members", "member_trials", "members_list"):
        v = arm.get(k)
        if isinstance(v, list) and v:
            mem = ",".join(sorted(str(x) for x in v))
            w = arm.get("weights") or arm.get("member_weights")
            ws = ",".join(str(x) for x in w) if isinstance(w, list) else ""
            return f"{mem}|{ws}"
    p = arm.get("params")
    if isinstance(p, dict):
        for k in ("members", "soup_members"):
            v = p.get(k)
            if isinstance(v, list) and v:
                mem = ",".join(sorted(str(x) for x in v))
                w = p.get("weights")
                ws = ",".join(str(x) for x in w) if isinstance(w, list) else ""
                return f"{mem}|{ws}"
    seed = arm.get("seed")
    return f"trial:{tid}" if seed is None else f"seed:{seed}"


def six_pass(m, th):
    bad = []
    missing = [k for k in ("nano", "nano4", "f1", "t1", "t2_fp", "t3") if m.get(k) is None]
    if missing:
        return False, ["缺指标:" + ",".join(missing)]
    for k, v in m.items():
        if k in ("board_ms",) or v is None:
            continue
        t = th.get(k)
        if t is None:
            continue
        if k == "t2_fp":
            if v > t:
                bad.append(f"{k}={v:.6g}>{t:.6g}")
        elif v < t:
            bad.append(f"{k}={v:.6g}<{t:.6g}")
    return (not bad), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    ledger = json.load(open(a.ledger, encoding="utf-8"))
    arms = ledger
    if isinstance(ledger, dict):
        for k in ("arms", "trials", "entries", "records"):
            if isinstance(ledger.get(k), (dict, list)):
                arms = ledger[k]
                break
    if isinstance(arms, list):
        arms = {str(x.get("trial", i)): x for i, x in enumerate(arms) if isinstance(x, dict)}

    th, src = thresholds(ledger)
    fams, sig_seen, metrics = {}, {}, {}
    n_arms = 0
    for tid, arm in arms.items():
        if not isinstance(arm, dict):
            continue
        n_arms += 1
        m = arm_metrics(arm)
        metrics[str(tid)] = m
        sig = signature_of(arm, str(tid))
        if sig in sig_seen:
            continue
        sig_seen[sig] = str(tid)
        ok, why = six_pass(m, th)
        fam = family_of(arm, str(tid))
        fams.setdefault(fam, []).append({"tid": str(tid), "ok": ok, "why": why, "m": m})

    fam_stat = {f: {"total": len(l), "six_pass": sum(1 for x in l if x["ok"]),
                     "ids": [x["tid"] for x in l if x["ok"]][:12]}
                for f, l in fams.items()}
    ready = {f: s for f, s in fam_stat.items() if s["six_pass"] >= 4}

    cands = []
    for fam, lst in fams.items():
        for x in lst:
            m = x["m"]
            if m.get("nano") is None or m.get("t1") is None:
                continue
            score = (m.get("nano4") or 0) + 3 * (m.get("t3") or 0) + 3 * (m.get("t1") or 0) + (m.get("f1") or 0)
            cands.append({"tid": x["tid"], "fam": fam, "six_pass": x["ok"], "why": x["why"],
                          "board_ms": m.get("board_ms"), "score": score,
                          **{k: m.get(k) for k in ("nano", "nano4", "f1", "t1", "t3", "t2_fp")}})
    cands.sort(key=lambda c: (c["six_pass"], c["score"]), reverse=True)

    out = {"ledger": a.ledger, "arms_total": n_arms, "thresholds": th, "threshold_source": src,
           "six_pass_arms": sum(1 for l in fams.values() for x in l if x["ok"]),
           "families": dict(sorted(fam_stat.items(), key=lambda kv: -kv[1]["six_pass"])[:20]),
           "families_ready_to_promote": ready,
           "baseline_v2": ledger.get("baseline_v2") if isinstance(ledger, dict) else None,
           "board_limit_ms": BOARD_LIMIT, "top_candidates": cands[: a.top]}
    print(json.dumps({k: v for k, v in out.items() if k != "top_candidates"}, ensure_ascii=False, indent=2))
    print("\n=== TOP 候选 ===")
    hdr = f"{'trial':<12}{'fam':<20}{'6项':<4}{'nano':<11}{'nano4':<11}{'T1':<11}{'T3':<11}{'T2fp':<6}{'board':<9}未过原因"
    print(hdr)
    for c in cands[: a.top]:
        f = lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else "-"
        print(f"{c['tid']:<12}{str(c['fam'])[:19]:<20}{'Y' if c['six_pass'] else 'n':<4}"
              f"{f(c['nano']):<11}{f(c['nano4']):<11}{f(c['t1']):<11}{f(c['t3']):<11}"
              f"{(str(c['t2_fp']) if c['t2_fp'] is not None else '-'):<6}"
              f"{(str(c['board_ms']) if c['board_ms'] is not None else '-'):<9}"
              f"{';'.join(c['why'])[:56]}")
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n报告写入 {a.out}")


if __name__ == "__main__":
    main()
