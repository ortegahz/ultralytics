import csv
import statistics as st

RED = 0.8958967482400269
T3GOAL = 0.87

def load(p):
    with open(p) as f:
        return list(csv.DictReader(f))

def num(r, k):
    try:
        return float(r.get(k, ""))
    except Exception:
        return None

rows = []
for tag, p in [("W", "/mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv"),
               ("F", "/mnt/data/siping/runs/nogmc_focus/optuna_summary.csv")]:
    for r in load(p):
        if num(r, "t1_f1") is None or num(r, "t3_f1") is None:
            continue
        r["_tag"] = tag
        r["_name"] = tag + ":" + r["trial"][-4:]
        rows.append(r)

MIXV = {"w1.0": 1.0, "w1.5": 1.5, "w2.0": 2.0, "w2.5": 2.5, "w3.0": 3.0, "w3.5": 3.5,
        "w4.0": 4.0, "w5.0": 5.0, "w6.0": 6.0, "w8.0": 8.0, "natural": 3.09}


def mixv(r):
    return MIXV.get(r.get("mix_spec"), 3.09)


def family(r):
    return "ep%s_%s_bb%s%s" % (r.get("epochs"), r.get("init_from"),
                               r.get("backbone_lr_scale"), r.get("freeze_backbone_epochs"))


print("=== 各家族样本量与 T1/T3 均值（按家族）===")
fams = {}
for r in rows:
    fams.setdefault(family(r), []).append(r)
for f, v in sorted(fams.items(), key=lambda x: -len(x[1])):
    if len(v) < 3:
        continue
    t1 = st.mean([num(r, "t1_f1") for r in v])
    t3 = st.mean([num(r, "t3_f1") for r in v])
    print("%-28s n=%-3d T1=%.6f(%+0.4fpp) T3=%.6f  FP=%d" % (
        f, len(v), t1, (t1 - RED) * 100, t3,
        st.mean([num(r, "t2_fp") or 0 for r in v])))

print("\n=== ep8 + rc2 + bb1/0 家族（family C 核心）逐点 ===")
core = [r for r in rows if r.get("epochs") == "8" and r.get("init_from") == "rc2"
        and r.get("backbone_lr_scale") == "1" and r.get("freeze_backbone_epochs") == "0"]
for r in sorted(core, key=lambda r: num(r, "lr_scale")):
    print("  %-10s lr=%-8s mix=%-8s T1=%.6f(%+0.4fpp) T3=%.6f FP=%-4s U=%s" % (
        r["_name"], r.get("lr_scale"), r.get("mix_spec"), num(r, "t1_f1"),
        (num(r, "t1_f1") - RED) * 100, num(r, "t3_f1"), r.get("t2_fp"), r.get("utility")))

def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = st.mean(xs); my = st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    ss = sum(r * r for r in resid)
    syy = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss / syy if syy else float("nan")
    return a, b, r2, math.sqrt(ss / max(n - 2, 1)) if n > 2 else float("nan")

import math

for fam_name, sel in [
    ("ep8/rc2/bb1-0 (family C)", core),
    ("ep8/rc2 全骨干档", [r for r in rows if r.get("epochs") == "8" and r.get("init_from") == "rc2"]),
    ("ep15 全家族", [r for r in rows if r.get("epochs") == "15"]),
]:
    if len(sel) < 3:
        continue
    print("\n=== 线性拟合 %s (n=%d) ===" % (fam_name, len(sel)))
    for label, target in [("T1", "t1_f1"), ("T3", "t3_f1")]:
        for xname, xf in [("lr", lambda r: num(r, "lr_scale")), ("mix", mixv)]:
            xs = [xf(r) for r in sel if xf(r) is not None]
            ys = [num(r, target) for r in sel if xf(r) is not None]
            fit = linfit(xs, ys)
            if fit:
                a, b, r2, se = fit
                print("  %s ~ %-4s : slope=%+.6f/unit  R2=%.3f  residSE=%.6f" % (label, xname, b, r2, se))

print("\n=== 全库 T1 过线 + T3>=0.87 的可行性扫描：按 (lr,mix) 在 ep8/rc2/bb1-0 家族内插值 ===")
print("    (用 family C 逐点包络，只看观测到的 lr×mix 邻域)")
grid = []
for r in core:
    grid.append((num(r, "lr_scale"), mixv(r), num(r, "t1_f1"), num(r, "t3_f1"), r["_name"]))
for lr, mx, t1, t3, nm in sorted(grid, key=lambda g: -g[3]):
    flag = "T1过线" if t1 >= RED else "T1差%+.4fpp" % ((t1 - RED) * 100)
    flag2 = "T3过线" if t3 >= T3GOAL else "T3差%+.4f" % (t3 - T3GOAL)
    print("  %-10s lr=%-8s mix=%-6s T1=%.6f T3=%.6f  %s / %s" % (nm, lr, mx, t1, t3, flag, flag2))
