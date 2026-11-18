import sys, time, json
sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")
import manu.pipeline.npu.nano_search as NS

ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
REPAIR = {1220012: (640001, 5, 1100296, 1, 0.5575),
          1220014: (640001, 5, 1100296, 1, 0.565)}
for sid, (pt, pe, ct, ce, w) in REPAIR.items():
    pkl = f"/mnt/data/siping/runs/npu20ms/tiny_ref/soupbatch_{sid}.pkl"
    out_ck = f"{ROOT}/trials/trial_{sid}/weights/epoch5.pt"
    try:
        res = NS._nb_pool().submit(NS._nb_worker, (pkl, f"soupbatch_{sid}")).result()
    except Exception as e:
        print(sid, "评测失败", e); continue
    if not res:
        print(sid, "评测空"); continue
    m = {"nano": res["nano"], "nano4": res["nano4"], "f1": res["f1"],
         "micro": res.get("micro"), "small": res.get("small"), "regular": res.get("regular"),
         "f1_4": res.get("f1_4"), "cache": pkl}
    fam = f"bs{pt}:w{int(round(w*100))}"
    led2 = NS.load_ledger()
    st = NS.arm_state(led2, sid)
    st.update({
        "trial": sid, "synth": True, "source_trial": int(pt),
        "soup_members": [[int(pt), int(pe)], [int(ct), int(ce)]],
        "soup_weights": [w, 1.0 - w], "soup_family": fam,
        "params": {"init": out_ck, "sw": 4.0, "lr": None, "lr_floor": None,
                   "mix": "soup", "fb": None, "ow": None, "sched": None,
                   "backbone_lr": None, "ema_decay": None, "swb": None,
                   "swb_thr": None, "cp_prob": None, "cp_max_px": None},
        "seed": None,
        "nanobench": {"5": m}, "picked": dict(m), "picked_epoch": 5,
        "official": None, "six_ok": False,
        "desc": f"[SB] soup({pt}@{pe},{ct}@{ce}) w={w}",
        "channel": "SB", "registered_by": "repair_ledger",
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    NS.save_ledger(led2)
    print(sid, "修复完成 nano=%.6f nano4=%.6f f1=%.6f fam=%s" % (m["nano"], m["nano4"], m["f1"], fam))
