import csv
from collections import defaultdict
from itertools import combinations

RED = 0.8958967482400269

PARAMS = [
    "mix_spec",
    "lr_scale",
    "warmup_epochs",
    "weight_decay",
    "focal_beta",
    "offset_weight",
    "mosaic",
    "scale",
    "translate",
    "clip_max_norm",
    "freeze_bn",
    "init_from",
    "epochs",
    "backbone_lr_scale",
    "freeze_backbone_epochs",
]


def load(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def num(r, k):
    try:
        return float(r.get(k, ""))
    except Exception:
        return None


rows = []
for tag, p in [
    ("W", "/mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv"),
    ("F", "/mnt/data/siping/runs/nogmc_focus/optuna_summary.csv"),
]:
    for r in load(p):
        if num(r, "t1_f1") is None or num(r, "t3_f1") is None:
            continue
        r["_tag"] = tag
        rows.append(r)

print("rows with metrics:", len(rows))

# ---- 1) 同参数 = 复跑/换种子 => 估计种子噪声 ----
allk = tuple(PARAMS)
g = defaultdict(list)
for r in rows:
    g[tuple(r.get(k) for k in allk)].append(r)
noise_t1 = []
noise_t3 = []
dup_groups = 0
for k, v in g.items():
    if len(v) < 2:
        continue
    dup_groups += 1
    for a, b in combinations(v, 2):
        noise_t1.append(abs(num(a, "t1_f1") - num(b, "t1_f1")))
        noise_t3.append(abs(num(a, "t3_f1") - num(b, "t3_f1")))
        print(
            "    SEED-PAIR %s%s vs %s%s : dT1=%+.6f dT3=%+.6f (mix=%s lr=%s ep=%s bb=%s/%s init=%s)"
            % (
                a["_tag"],
                a["trial"][-4:],
                b["_tag"],
                b["trial"][-4:],
                num(b, "t1_f1") - num(a, "t1_f1"),
                num(b, "t3_f1") - num(a, "t3_f1"),
                a.get("mix_spec"),
                a.get("lr_scale"),
                a.get("epochs"),
                a.get("backbone_lr_scale"),
                a.get("freeze_backbone_epochs"),
                a.get("init_from"),
            )
        )
print(
    "identical-param groups: %d, pairs: %d" % (dup_groups, len(noise_t1))
)
if noise_t1:
    import statistics as st

    print(
        "  |dT1| median=%.6f max=%.6f ; |dT3| median=%.6f max=%.6f"
        % (
            st.median(noise_t1),
            max(noise_t1),
            st.median(noise_t3),
            max(noise_t3),
        )
    )

# ---- 2) 只差一个参数的受控对 ----
print("\n########## 受控对：只差一个参数 ##########")
summary = []
for p in PARAMS:
    others = [k for k in PARAMS if k != p]
    gg = defaultdict(list)
    for r in rows:
        gg[tuple(r.get(k) for k in others)].append(r)
    pairs = []
    for k, v in gg.items():
        if len(v) < 2:
            continue
        vals = set(r.get(p) for r in v)
        if len(vals) < 2:
            continue
        for a, b in combinations(v, 2):
            if a.get(p) == b.get(p):
                continue
            pairs.append((a, b))
    if not pairs:
        print("\n--- %s : 0 对 ---" % p)
        continue
    dt1 = []
    dt3 = []
    detail = []
    for a, b in pairs:
        va, vb = num(a, p), num(b, p)
        if va is None or vb is None:
            continue
        d = vb - va
        if d == 0:
            continue
        t1 = (num(b, "t1_f1") - num(a, "t1_f1")) / d
        t3 = (num(b, "t3_f1") - num(a, "t3_f1")) / d
        dt1.append(t1)
        dt3.append(t3)
        detail.append(
            "%s->%s [%s %s->%s] dT1=%+.6f dT3=%+.6f"
            % (
                a["_tag"] + a["trial"][-4:],
                b["_tag"] + b["trial"][-4:],
                p,
                a.get(p),
                b.get(p),
                num(b, "t1_f1") - num(a, "t1_f1"),
                num(b, "t3_f1") - num(a, "t3_f1"),
            )
        )
    if not dt1:
        continue
    import statistics as st

    summary.append(
        (
            p,
            len(dt1),
            st.mean(dt1),
            st.mean(dt3),
            sum(1 for x in dt1 if x > 0),
            sum(1 for x in dt3 if x > 0),
        )
    )
    print("\n--- %s : %d 对 ---" % (p, len(dt1)))
    for d in detail:
        print("    " + d)
    print(
        "    均值: dT1/unit=%+.6f  dT3/unit=%+.6f   T1↑%d/%d  T3↑%d/%d"
        % (st.mean(dt1), st.mean(dt3), sum(1 for x in dt1 if x > 0), len(dt1), sum(1 for x in dt3 if x > 0), len(dt3))
    )

print("\n########## 汇总（按 |dT1/unit| 排序：T1 杠杆强度）##########")
summary.sort(key=lambda x: -abs(x[2]))
print("%-24s %5s %14s %14s %8s %8s" % ("param", "n", "dT1/unit", "dT3/unit", "T1up", "T3up"))
for p, n, m1, m3, u1, u3 in summary:
    print("%-24s %5d %+14.6f %+14.6f %5d/%-3d %5d/%-3d" % (p, n, m1, m3, u1, n, u3, n))

print("\n########## T3 杠杆强度排序 ##########")
summary.sort(key=lambda x: -abs(x[3]))
print("%-24s %5s %14s %14s" % ("param", "n", "dT1/unit", "dT3/unit"))
for p, n, m1, m3, u1, u3 in summary:
    print("%-24s %5d %+14.6f %+14.6f" % (p, n, m1, m3))
