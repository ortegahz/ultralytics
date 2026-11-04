import csv
import json
from collections import defaultdict
from itertools import combinations

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

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


def load_csv(p):
    with open(p) as f:
        return {r["trial"]: r for r in csv.DictReader(f)}


def num(r, k):
    try:
        return float(r.get(k, ""))
    except Exception:
        return None


def study_params(db_path, study_name):
    st = optuna.study.load_study(storage="sqlite:///" + db_path, study_name=study_name)
    out = {}
    for t in st.trials:
        p = dict(t.params)
        if not p:
            p = dict(t.system_attrs.get("fixed_params", {}))
        seed = t.user_attrs.get("seed")
        if seed is None:
            seed = t.system_attrs.get("seed")
        out["trial_%04d" % t.number] = {"params": p, "seed": seed, "state": str(t.state)}
    return out


def norm(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "%.12g" % v
    return str(v)


SOURCES = [
    ("W", "/mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv",
     "/mnt/data/siping/runs/nogmc_weekend/study.db", "mix_ratio_search_nogmc_weekend"),
    ("F", "/mnt/data/siping/runs/nogmc_focus/optuna_summary.csv",
     "/mnt/data/siping/runs/nogmc_focus/study.db", "mix_ratio_search_nogmc_focus"),
]
csv_all = {}
sp = {}
for tag, cpath, dbpath, sname in SOURCES:
    for k, v in load_csv(cpath).items():
        csv_all["%s:%s" % (tag, k)] = v
    for k, v in study_params(dbpath, sname).items():
        sp["%s:%s" % (tag, k)] = v

rows = []
for name, r in csv_all.items():
    if num(r, "t1_f1") is None or num(r, "t3_f1") is None:
        continue
    meta = sp.get(name)
    if meta is None:
        continue
    p = meta["params"]
    missing = [k for k in PARAMS if k not in p]
    if missing:
        continue
    rec = {
        "name": name,
        "params": {k: norm(p[k]) for k in PARAMS},
        "raw": p,
        "seed": meta["seed"],
        "t1": num(r, "t1_f1"),
        "t3": num(r, "t3_f1"),
        "fp": num(r, "t2_fp"),
        "u": num(r, "utility"),
        "verdict": r.get("gate_verdict"),
    }
    rows.append(rec)

print("rows joined with exact study params:", len(rows))
seeds_known = sum(1 for r in rows if r["seed"] is not None)
print("rows with known seed:", seeds_known)

# ---- A) 同参数组（换种子）=> 种子噪声 ----
print("\n########## A) 同参数换种子：种子噪声实测 ##########")
g = defaultdict(list)
for r in rows:
    g[tuple(r["params"][k] for k in PARAMS)].append(r)
noise_t1, noise_t3 = [], []
for k, v in g.items():
    if len(v) < 2:
        continue
    for a, b in combinations(v, 2):
        d1 = b["t1"] - a["t1"]
        d3 = b["t3"] - a["t3"]
        noise_t1.append(abs(d1))
        noise_t3.append(abs(d3))
        print(
            "  %-11s vs %-11s seeds=%s/%s : dT1=%+.6f dT3=%+.6f | %s mix=%s lr=%s ep=%s bb=%s/%s init=%s"
            % (
                a["name"],
                b["name"],
                a["seed"],
                b["seed"],
                d1,
                d3,
                "mix=%s lr=%s ep=%s bb=%s/%s init=%s" % (
                    a["params"]["mix_spec"],
                    a["params"]["lr_scale"],
                    a["params"]["epochs"],
                    a["params"]["backbone_lr_scale"],
                    a["params"]["freeze_backbone_epochs"],
                    a["params"]["init_from"],
                ),
                a["params"]["mix_spec"],
                a["params"]["lr_scale"],
                a["params"]["epochs"],
                a["params"]["backbone_lr_scale"],
                a["params"]["freeze_backbone_epochs"],
                a["params"]["init_from"],
            )
        )
if noise_t1:
    import statistics as st

    print(
        "  => |dT1| median=%.6f max=%.6f ; |dT3| median=%.6f max=%.6f"
        % (st.median(noise_t1), max(noise_t1), st.median(noise_t3), max(noise_t3))
    )

# ---- B) 只差一个参数 ----
print("\n########## B) 受控对（精确参数，只差一项）##########")
summary = []
for p in PARAMS:
    others = [k for k in PARAMS if k != p]
    gg = defaultdict(list)
    for r in rows:
        gg[tuple(r["params"][k] for k in others)].append(r)
    pairs = []
    for k, v in gg.items():
        if len(v) < 2:
            continue
        vals = set(r["params"][p] for r in v)
        if len(vals) < 2:
            continue
        for a, b in combinations(v, 2):
            if a["params"][p] == b["params"][p]:
                continue
            pairs.append((a, b))
    if not pairs:
        print("\n--- %s : 0 对 ---" % p)
        continue
    det = []
    u1, u3 = [], []
    for a, b in pairs:
        va = float(a["params"][p]) if p not in ("mix_spec", "init_from", "freeze_bn") else None
        vb = float(b["params"][p]) if p not in ("mix_spec", "init_from", "freeze_bn") else None
        d1 = b["t1"] - a["t1"]
        d3 = b["t3"] - a["t3"]
        u1.append(d1)
        u3.append(d3)
        det.append(
            "    %-11s(%s=%-12s) -> %-11s(%s=%-12s)  dT1=%+.6f  dT3=%+.6f  dFP=%+.0f"
            % (a["name"], p, a["params"][p], b["name"], p, b["params"][p], d1, d3, (b["fp"] or 0) - (a["fp"] or 0))
        )
    import statistics as st

    print("\n--- %s : %d 对 ---" % (p, len(pairs)))
    for d in det:
        print(d)
    print(
        "    均值 dT1=%+.6f  dT3=%+.6f   T1↑%d/%d  T3↑%d/%d"
        % (
            st.mean(u1),
            st.mean(u3),
            sum(1 for x in u1 if x > 0),
            len(u1),
            sum(1 for x in u3 if x > 0),
            len(u3),
        )
    )
    summary.append((p, len(pairs), st.mean(u1), st.mean(u3)))

print("\n########## 汇总 ##########")
print("%-24s %4s %12s %12s" % ("param", "n", "dT1/对", "dT3/对"))
for p, n, m1, m3 in sorted(summary, key=lambda x: -abs(x[2])):
    print("%-24s %4d %+12.6f %+12.6f" % (p, n, m1, m3))
