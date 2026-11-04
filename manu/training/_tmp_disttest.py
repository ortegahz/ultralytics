"""在**真实 study 的副本**上验证：扩 FOCUS_CHOICES 后，Optuna 会不会报 dynamic value space。"""
import shutil
import sys

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SRC = "/mnt/data/siping/runs/nogmc_focus/study.db"
DST = "/tmp/_study_copy_test.db"
shutil.copy(SRC, DST)

sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")
import manu.training.optuna_mix_ratio_search as M  # noqa: E402

M.FOCUS_SPACE = True
M.EXTENDED_SPACE = True

storage = "sqlite:///" + DST
st = optuna.study.load_study(storage=storage, study_name="mix_ratio_search_nogmc_focus")

print("副本里 trial 数:", len(st.trials))
dist_hist = set()
for t in st.trials:
    for k, d in (t.distributions or {}).items():
        dist_hist.add((k, str(d)))
print("study 里已记录的分布类型数:", len(dist_hist))
for k, d in sorted(dist_hist):
    print("   ", k, "->", d[:110])

# --- 关键测试：扩 choices 后 ask+suggest 会不会炸 ---
NEW_CHOICES = {
    "init_from": ["rc2", "base474", "soup_strict3", "soup_high3_nobn"],
    "epochs": [6, 8, 10, 15],
    "mix_spec": ["w2.0", "w2.5", "w3.0", "w4.0", "w6.0", "w8.0", "natural"],
    "backbone_lr_scale": [0.0, 0.1, 0.3, 0.5, 1.0],
    "freeze_backbone_epochs": [0, 1, 2],
}
OLD = dict(M.FOCUS_CHOICES)

print("\n=== 测试 1：用**扩大的** choices 在新 trial 上 suggest ===")
M.FOCUS_CHOICES.update(NEW_CHOICES)
M.WARMUP_EPOCHS_HIGH = 1.8
try:
    t = st.ask()
    p = M.suggest_params(t)
    print("   ✅ 没报错，新 trial suggest 成功：")
    for k, v in p.items():
        print("      ", k, "=", v)
except Exception as exc:
    print("   ❌ 炸了：", type(exc).__name__, exc)

print("\n=== 测试 2：老 trial（已 COMPLETE，分布已记录）再被 ask 会不会炸 ===")
try:
    for t2 in st.trials:
        if t2.state == optuna.trial.TrialState.COMPLETE and t2.distributions:
            print("   参考：t%04d 已记录分布 %d 项" % (t2.number, len(t2.distributions)))
            break
    # 用 study.optimize 走一遍（它会做分布一致性检查）
    def obj(tr):
        p = M.suggest_params(tr)
        return 1.0

    st2 = optuna.study.load_study(storage=storage, study_name="mix_ratio_search_nogmc_focus")
    st2.optimize(obj, n_trials=1)
    print("   ✅ study.optimize + suggest_params 没炸")
except Exception as exc:
    print("   ❌ 炸了：", type(exc).__name__, exc)

print("\n=== 测试 3：直接给一个固定超参（out-of-space）建 study，看 suggest 行为 ===")
st3 = optuna.create_study(storage="sqlite:////tmp/_fresh_test.db", study_name="fresh")
st3.enqueue_trial({"mix_spec": "w2.5", "epochs": 8, "init_from": "rc2"})
M.FOCUS_CHOICES.update(OLD)  # 恢复老 choices
try:
    def obj2(tr):
        p = M.suggest_params(tr)
        print("      suggest 成功 ->", p.get("mix_spec"))
        return 1.0

    st3.optimize(obj2, n_trials=1)
    print("   ✅ out-of-space 固定参数**没有**报错（说明 Optuna 不校验固定参数）")
except Exception as exc:
    print("   ❌ 炸了（Optuna 会校验固定参数）：", type(exc).__name__, exc)
