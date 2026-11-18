#!/usr/bin/env python3
"""从已策展的 k=5 集合派生子家族集合（k=4 / k=6），用于「同家族六项臂 ≥4」的升轨判据。

动机：家族键 = `soup:bs<盆地>:eq<成员数>` ⇒ eq4/eq6 各自成族，任一族凑够 4 个六项臂即可升轨。
k=4：平均稀释更小 ⇒ 单臂六项全过概率更高；且成员必然落在已通过余弦过滤的池内（避免 14199 类毒臂）。

用法:
  python3 make_family_sets.py --ledger <ledger.json> --curated <k5_sets.json> \
      --k 4 --out <k4_sets.json> --shards 4 --per-shard 40 [--format list]
"""
from __future__ import annotations
import argparse, json, math, itertools, re
from collections import defaultdict

AXES = ["nano", "nano4", "f1", "t1", "t3"]
# §191 标定：汤 − 成员均值 的系统偏移（nano 不可预测 ⇒ 偏移置 0）
SOUP_DELTA = {"t1": +0.000059, "t3": -0.000053, "f1": +0.000052, "nano4": -0.000589, "nano": 0.0}
# 六项门槛（Baseline_v2.1 口径）
THRESH = {"nano": (">=", 0.715737), "nano4": (">=", 0.650756), "f1": (">=", 0.890043),
          "t1": (">=", 0.890013), "t3": (">=", 0.948526)}
KEYS = {"nano": ["nano", "nano_recall"], "nano4": ["nano4", "nano_4"], "f1": ["f1", "F1"],
        "t1": ["t1", "T1"], "t3": ["t3", "T3"], "t2_fp": ["t2_fp", "T2_fp", "t2fp"]}
NEST = ["picked", "metrics", "official", "metric", "eval"]


def dig(arm, name):
    for nest in NEST:
        sub = arm.get(nest)
        if isinstance(sub, dict):
            for k in KEYS.get(name, [name]):
                v = sub.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
    for k in KEYS.get(name, [name]):
        v = arm.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def pick(d, names):
    for n in names:
        if isinstance(d, dict) and isinstance(d.get(n), (int, float)):
            return float(d[n])
    return None


def load_arms(ledger):
    raw = json.load(open(ledger, encoding="utf-8"))
    arms = raw
    if isinstance(raw, dict):
        for k in ("arms", "trials", "entries", "records"):
            if isinstance(raw.get(k), (dict, list)):
                arms = raw[k]
                break
    if isinstance(arms, list):
        arms = {str(i): a for i, a in enumerate(arms)}
    out = {}
    for tid, a in arms.items():
        if not isinstance(a, dict):
            continue
        nb = a.get("picked")
        if not isinstance(nb, dict):
            pe = a.get("picked_epoch")
            nball = a.get("nanobench")
            nb = nball.get(str(pe), {}) if isinstance(nball, dict) and pe is not None else {}
        if not isinstance(nb, dict):
            nb = {}
        off = a.get("official") if isinstance(a.get("official"), dict) else {}
        m = {"nano": pick(nb, ["nano"]), "nano4": pick(nb, ["nano4", "nano_4"]),
             "f1": pick(nb, ["f1", "F1"]), "t1": pick(off, ["t1", "T1"]),
             "t3": pick(off, ["t3", "T3"]), "t2_fp": pick(off, ["t2_fp", "T2_fp"])}
        if m.get("nano") is None:  # 兼容另一种 schema：metrics/official 平铺
            for n in AXES + ["t2_fp"]:
                if m.get(n) is None:
                    m[n] = dig(a, n)
        if m.get("nano") is not None:
            out[str(tid)] = m
    return out


def norm_id(x):
    """'trial_96835@5' / '96835@5' / '96835' → 统一 key"""
    s = str(x).strip()
    s = re.sub(r"^trial_", "", s)
    return s


def metrics_of(members, arms):
    """成员 id → 指标（arms key 可能是 'trial_96835' / '96835' / '96835@5'）"""
    vals = {}
    for mem in members:
        key = norm_id(mem)
        m = arms.get(key) or arms.get(f"trial_{key}") or arms.get(mem)
        if m is None and "@" in key:
            base, ep = key.split("@", 1)
            m0 = arms.get(base) or arms.get(f"trial_{base}")
            if m0 is not None:
                pe = m0.get("picked_epoch")
                try:
                    ep_i = int(float(ep))
                except ValueError:
                    ep_i = None
                if ep_i is None or (pe is not None and int(pe) == ep_i):
                    m = m0
        if m is None:
            return None
        if any(m.get(a) is None for a in AXES):
            return None
        vals[key] = m
    return vals


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--curated", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--per-shard", type=int, default=40)
    ap.add_argument("--format", choices=["list", "dict"], default="list")
    ap.add_argument("--min-prob", type=float, default=0.0, help="预计六项全过概率下限")
    a = ap.parse_args()

    arms = load_arms(a.ledger)
    if not arms:
        raise SystemExit("台账里没有可用臂（指标为空）")
    # σ：全体池各轴标准差（用于 z 分数）
    stats = {}
    for ax in AXES:
        xs = [m[ax] for m in arms.values() if m.get(ax) is not None]
        mu = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)) or 1e-9
        stats[ax] = (mu, sd)

    raw = json.load(open(a.curated, encoding="utf-8"))
    groups = []
    if isinstance(raw, dict):
        for v in raw.values():
            groups.extend(v if isinstance(v, list) else [])
    elif isinstance(raw, list):
        groups = raw
    # 统一成「成员列表」
    norm_groups = []
    for g in groups:
        if isinstance(g, list) and g and all(isinstance(x, (str, int)) for x in g):
            norm_groups.append([str(x) for x in g])
        elif isinstance(g, dict):
            for k in ("members", "set", "trials", "ids", "group"):
                if isinstance(g.get(k), list):
                    norm_groups.append([str(x) for x in g[k]])
                    break
    print(f"策展集合 {len(norm_groups)} 组（k 源={len(norm_groups[0]) if norm_groups else 0}）")

    seen, cands = set(), []
    for g in norm_groups:
        if len(g) < a.k:
            continue
        for combo in itertools.combinations(sorted(g), a.k):
            key = "|".join(combo)
            if key in seen:
                continue
            seen.add(key)
            vals = metrics_of(list(combo), arms)
            if vals is None:
                continue
            mean = {ax: sum(v[ax] for v in vals.values()) / a.k for ax in AXES}
            pred = {ax: mean[ax] + SOUP_DELTA[ax] for ax in AXES}
            z = {ax: (pred[ax] - stats[ax][0]) / stats[ax][1] for ax in AXES}
            score = 1.0
            for ax in AXES:
                score *= phi(z[ax])
            # 六项全过的「必要条件」：集合均值本身必须已过线（家族均值口径）
            feasible = all((mean[ax] >= THRESH[ax][1]) for ax in AXES)
            cands.append({"members": list(combo), "mean": mean, "pred": pred,
                          "z": z, "score": score, "feasible": feasible})
    if not cands:
        raise SystemExit("没有可用候选（检查台账/集合 schema）")
    cands.sort(key=lambda c: (c["feasible"], c["score"]), reverse=True)
    feas = [c for c in cands if c["feasible"]]
    print(f"候选 {len(cands)} 组，其中集合均值已过线 {len(feas)} 组；"
          f"最高 ΠΦ(z)={feas[0]['score']:.4f}" if feas else f"候选 {len(cands)} 组（无均值过线者）")

    # 按 mean-nano 分三层取样，抵消「回归均值」风险
    pool = feas if feas else cands
    by_nano = sorted(pool, key=lambda c: -c["mean"]["nano"])
    n = len(by_nano)
    # 三层（按 mean-nano）分层，层内按 ΠΦ(z) 降序取 ⇒ 既对冲回归均值，又不浪费命中
    hi = sorted(by_nano[: n // 3], key=lambda c: -c["score"])
    mid = sorted(by_nano[n // 3: 2 * n // 3], key=lambda c: -c["score"])
    lo = sorted(by_nano[2 * n // 3:], key=lambda c: -c["score"])
    picked, idx = [], 0
    per = max(1, a.per_shard // 3)
    for trio in (hi, mid, lo):
        picked.extend(trio[:per])
        idx += 1
    picked = picked[: a.per_shard * a.shards]

    shards = []
    for s in range(a.shards):
        part = picked[s * a.per_shard: (s + 1) * a.per_shard]
        shards.append(part)
    out_sets = [{"members": c["members"], "score": round(c["score"], 6),
                 "pred": {ax: round(c["pred"][ax], 8) for ax in AXES}} for c in picked]
    if a.format == "list":
        payload = [c["members"] for c in picked]
    else:
        payload = out_sets
    json.dump(payload, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for s, part in enumerate(shards):
        json.dump([c["members"] for c in part], open(f"{a.out[:-5]}_s{s}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"写出 {a.out}：{len(payload)} 组 / {a.shards} 片（每片 {a.per_shard}）")
    if picked:
        c = picked[0]
        print("示例（hi-nano 层最佳）:", c["members"], "score=%.4f" % c["score"],
              "pred:", {ax: round(c["pred"][ax], 6) for ax in AXES})


if __name__ == "__main__":
    main()
