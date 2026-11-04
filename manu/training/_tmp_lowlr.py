import csv

RED = 0.8958967482400269


def load(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def num(r, k):
    try:
        return float(r.get(k, ""))
    except Exception:
        return None


w = load("/mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv")
f = load("/mnt/data/siping/runs/nogmc_focus/optuna_summary.csv")
allr = [("W", r) for r in w] + [("F", r) for r in f]


def show(s, r):
    t1 = num(r, "t1_f1")
    t3 = num(r, "t3_f1")
    return (
        "%s %-10s T3=%.6f T1=%.6f(%+0.4fpp) FP=%-4s U=%-22s mix=%-8s lr=%-8s ep=%-3s bb=%s/%s init=%-8s wu=%-6s ver=%s"
        % (
            s,
            r.get("trial"),
            t3,
            t1,
            (t1 - RED) * 100,
            r.get("t2_fp"),
            r.get("utility") or "None",
            r.get("mix_spec"),
            r.get("lr_scale"),
            r.get("epochs"),
            r.get("backbone_lr_scale"),
            r.get("freeze_backbone_epochs"),
            r.get("init_from"),
            r.get("warmup_epochs"),
            r.get("gate_verdict"),
        )
    )


print("###### A) 全库中 lr_scale<=1.5 的行（低 lr 家族）######")
rows = [(s, r) for s, r in allr if (num(r, "lr_scale") or 9) <= 1.5]
for s, r in sorted(rows, key=lambda x: -(num(x[1], "t3_f1") or 0)):
    print(show(s, r))

print("\n###### B) epochs>=10 且 lr<=2.0 的行 ######")
rows = [(s, r) for s, r in allr if (num(r, "epochs") or 0) >= 10 and (num(r, "lr_scale") or 9) <= 2.0]
for s, r in sorted(rows, key=lambda x: -(num(x[1], "t3_f1") or 0)):
    print(show(s, r))

print("\n###### C) epochs>=14 的行按 T3 排序（长训家族）######")
rows = [(s, r) for s, r in allr if (num(r, "epochs") or 0) >= 14]
for s, r in sorted(rows, key=lambda x: -(num(x[1], "t3_f1") or 0))[:20]:
    print(show(s, r))

print("\n###### D) init_from=base474 且 epochs>=14 的行（T3 家族 base474）######")
rows = [
    (s, r)
    for s, r in allr
    if (num(r, "epochs") or 0) >= 14 and r.get("init_from") == "base474"
]
for s, r in sorted(rows, key=lambda x: -(num(x[1], "t3_f1") or 0)):
    print(show(s, r))

print("\n###### E) 全部严格 PASS（gate PASS）里 epochs>=10 的行 ######")
rows = [(s, r) for s, r in allr if r.get("gate_verdict") == "PASS" and (num(r, "epochs") or 0) >= 10]
for s, r in sorted(rows, key=lambda x: -(num(x[1], "t3_f1") or 0)):
    print(show(s, r))

print("\n###### F) mix_spec=natural 的行按 T3 排序 top 15 ######")
rows = [(s, r) for s, r in allr if r.get("mix_spec") == "natural"]
for s, r in sorted(rows, key=lambda x: -(num(x[1], "t3_f1") or 0))[:15]:
    print(show(s, r))
