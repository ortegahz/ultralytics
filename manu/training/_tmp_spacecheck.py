import json
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

FOCUS_CHOICES = {
    "init_from": ["rc2", "base474"],
    "epochs": [6, 8, 10, 15],
    "mix_spec": ["w2.0", "w3.0", "w6.0", "w8.0", "natural"],
    "backbone_lr_scale": [0.0, 0.1, 0.3, 1.0],
    "freeze_backbone_epochs": [0, 2],
}
FLOAT_RANGE = {
    "lr_scale": (0.8, 3.0),
    "warmup_epochs": (0.2, 1.5),
    "weight_decay": (1e-5, 5e-4),
    "focal_beta": (2.2, 2.9),
    "offset_weight": (0.40, 0.70),
    "mosaic": (0.0, 0.15),
    "scale": (0.10, 0.30),
    "translate": (0.05, 0.15),
    "clip_max_norm": (2.0, 8.0),
}
STEP = {"focal_beta": 0.05, "offset_weight": 0.05, "mosaic": 0.05, "scale": 0.05, "translate": 0.05}


def check(p):
    bad = []
    for k, choices in FOCUS_CHOICES.items():
        v = p.get(k)
        if v is None:
            continue
        if v not in choices:
            bad.append("%s=%r 不在 %s" % (k, v, choices))
    for k, (lo, hi) in FLOAT_RANGE.items():
        v = p.get(k)
        if v is None:
            continue
        v = float(v)
        if not (lo - 1e-9 <= v <= hi + 1e-9):
            bad.append("%s=%r 越界 [%s,%s]" % (k, v, lo, hi))
        st = STEP.get(k)
        if st:
            off = abs(round((v - lo) / st) - (v - lo) / st)
            if off > 1e-6:
                bad.append("%s=%r 不在 step=%s 的格点(lo=%s)" % (k, v, st, lo))
    return bad


plan = json.load(open("/mnt/data/siping/runs/nogmc_focus/../nogmc_focus_plan.json")) \
    if False else json.load(open("/tmp/pycharm_project_10ae9e2e/manu/training/nogmc_focus_plan.json"))
plan_by_role = {x["role"]: x for x in plan}

st = optuna.study.load_study(
    storage="sqlite:////mnt/data/siping/runs/nogmc_focus/study.db",
    study_name="mix_ratio_search_nogmc_focus")

role_of = {}
for t in st.trials:
    ua = t.user_attrs or {}
    if ua.get("role"):
        role_of[ua["role"]] = t

print("study 共 %d 个 trial；带 role 的 %d 个" % (len(st.trials), len(role_of)))
print("\n########## A) plan 臂的运行状态 + 越界体检 ##########")
bad_roles = []
for role, item in plan_by_role.items():
    t = role_of.get(role)
    p = item.get("params") or {}
    bad = check(p)
    if t is None:
        state = "NOT_ENQUEUED"
        num = "-"
    else:
        state = t.state.value
        num = "t%04d" % t.number
    mark = "❌" if bad else "✅"
    if bad:
        bad_roles.append((role, bad, state))
    print("%s %-36s %-12s %-14s %s" % (
        mark, role, num, state, ("; ".join(bad)) if bad else ""))

print("\n########## B) 越界臂汇总 ##########")
for role, bad, state in bad_roles:
    print("  %-36s state=%-12s %s" % (role, state, "; ".join(bad)))
print("越界臂共 %d 条（会在 trial 启动时抛 ValueError ⇒ 直接 FAIL，不产出结果）" % len(bad_roles))

print("\n########## C) 已入队臂的实际 state 分布 ##########")
from collections import Counter
c = Counter(t.state.value for t in st.trials)
print(dict(c))
print("\n各臂 role → trial/state：")
for t in sorted(st.trials, key=lambda t: t.number):
    ua = t.user_attrs or {}
    if ua.get("role"):
        print("   t%-4d %-36s %-10s seed=%s" % (t.number, ua.get("role"), t.state.value, ua.get("seed")))
