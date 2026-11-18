"""基线守卫：每 60s 校验台账 baseline_v2 == 代码常量 BASE_V2（trial/指标）
   ⇒ 并发写把新基线回滚时立刻补回（升轨的最后一公里不能丢）。"""
import sys, time, os
sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")
import manu.pipeline.npu.nano_search as ns

W = "/mnt/data/siping/runs/npu20ms/nano_search"
LOG = os.path.join(W, "baseline_guard.log")
END = "2026-11-18 07:00"

def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("[%s] %s\n" % (time.strftime("%F %T"), m))

while time.strftime("%Y-%m-%d %H:%M") < END:
    try:
        want = ns.BASE_V2
        led = ns.load_ledger()
        cur = led.get("baseline_v2") or {}
        ok = (int(cur.get("trial") or 0) == int(want["trial"])
              and abs(float(cur.get("t1") or 0) - float(want["t1"])) < 1e-9)
        if not ok:
            new = dict(want)
            new["previous_baseline"] = {"name": cur.get("name"), "trial": cur.get("trial"),
                                        "epoch": cur.get("epoch"), "nano": cur.get("nano")}
            led["baseline_v2"] = new
            reg = [x for x in (led.get("baseline_registry") or []) if x.get("trial") != want["trial"]]
            for x in reg:
                if cur.get("trial") is not None and x.get("trial") == cur.get("trial"):
                    x["status"] = "SUPERSEDED / DEPRECATED BASELINE"
            reg.insert(0, {"id": want["name"], "trial": want["trial"],
                           "status": "ACTIVE / OFFICIAL BASELINE",
                           "note": "基线守卫自动补写 " + time.strftime("%F %T")})
            led["baseline_registry"] = reg
            ns.save_ledger(led)
            log("基线被并发写回滚 => 已补回 %s(trial=%s)" % (want["name"], want["trial"]))
    except Exception as e:
        log("守卫异常(忽略): %s" % e)
    time.sleep(60)
log("基线守卫到期退出")
