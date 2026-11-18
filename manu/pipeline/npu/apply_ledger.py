import sys, json, time, os
sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")
import manu.pipeline.npu.nano_search as ns

W = "/mnt/data/siping/runs/npu20ms/nano_search"
prom = os.path.join(W, "promotion_1106580.json")
promoted_at = "2026-11-17 17:54:29"
if os.path.isfile(prom):
    promoted_at = json.load(open(prom)).get("promoted_at") or promoted_at

led = ns.load_ledger()
old = dict(led.get("baseline_v2") or {})
new = dict(ns.BASE_V2)
new["name"] = "Baseline_v2.3"
new["trial"] = 1106580
new["promoted_at"] = promoted_at
new["previous_baseline"] = {"name": old.get("name"), "trial": old.get("trial"),
                            "epoch": old.get("epoch"), "nano": old.get("nano")}
led["baseline_v2"] = new

reg = [x for x in (led.get("baseline_registry") or []) if x.get("trial") != 1106580]
for x in reg:
    if x.get("trial") == 610000:
        x["status"] = "SUPERSEDED / DEPRECATED BASELINE"
        x["note"] = "2026-11-17 17:54 被 trial_1106580 取代"
reg.insert(0, {"id": "Baseline_v2.3", "trial": 1106580, "status": "ACTIVE / OFFICIAL BASELINE",
               "note": "2026-11-17 17:54 升轨; 家族 soup:bs640001:w2 n=4 + 种子关 bs640001:w55 + 板卡 19.706ms"})
led["baseline_registry"] = reg
led["beat_to_pass"] = {
    "ref": "Baseline_v2.3(trial_1106580)", "nano_min": new["nano"], "f1_min": new["f1"],
    "t1_min": new["t1"], "t2_fp_max": new["t2_fp"], "t3_min": new["t3"],
    "legacy_ref": "trial_0031", "legacy_nano_min": ns.BASELINE_NANO_FLOOR["0031"],
    "note": "2026-11-17 18:10 升轨后统一以 Baseline_v2.3 为 Beat-to-Pass 参照"}
ns.save_ledger(led)

v = ns.load_ledger().get("baseline_v2") or {}
print("台账基线:", v.get("name"), v.get("trial"), v.get("promoted_at"))
print("registry[0]:", (ns.load_ledger().get("baseline_registry") or [{}])[0])
