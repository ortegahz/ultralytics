import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("hasattr(optuna,'delete_trial'):", hasattr(optuna, "delete_trial"))
print("hasattr(study,'delete_trial'):", hasattr(optuna.study.Study, "delete_trial"))
if hasattr(optuna, "delete_trial"):
    import inspect

    print(inspect.signature(optuna.delete_trial))
    print(inspect.getdoc(optuna.delete_trial)[:600])

st = optuna.study.load_study(
    storage="sqlite:////mnt/data/siping/runs/nogmc_focus/study.db",
    study_name="mix_ratio_search_nogmc_focus")

print("\n=== 每个 trial 的 distributions / fixed_params ===")
for t in st.trials:
    d = t.distributions or {}
    fx = (t.system_attrs or {}).get("fixed_params")
    mix_d = d.get("mix_spec")
    print(
        "t%-4d %-9s dists=%-3d fixed=%-5s mix_dist=%s"
        % (
            t.number,
            t.state.value,
            len(d),
            "YES" if fx else "no",
            (str(mix_d)[:60] if mix_d is not None else "-"),
        )
    )

print("\n=== ask() 会返回哪个 trial？===")
st2 = optuna.study.load_study(
    storage="sqlite:////mnt/data/siping/runs/nogmc_focus/study.db",
    study_name="mix_ratio_search_nogmc_focus")
waiting = [t for t in st2.trials if t.state == optuna.trial.TrialState.WAITING]
print("WAITING trial 号:", [t.number for t in waiting])
if waiting:
    t0 = waiting[0]
    print("最老 WAITING t%d 的 distributions 条数 = %d" % (t0.number, len(t0.distributions or {})))
    print("  fixed_params 键:", sorted((t0.system_attrs or {}).get("fixed_params", {}).keys()))
