import json, os, torch
root = "/mnt/data/siping/runs/npu20ms_seed2/trials"
for i in range(0, 8):
    t = "trial_%04d" % i
    d = os.path.join(root, t)
    if not os.path.isdir(d):
        continue
    ck = os.path.join(d, "weights", "epoch20.pt")
    gr = os.path.join(d, "gate", "epoch20", "gate_report.json")
    st = {}
    if os.path.isfile(ck):
        try:
            c = torch.load(ck, map_location="cpu", weights_only=False)
            st = {k: c.get(k) for k in ("head_level", "act", "head_conv", "late_upsample", "p2_out_scale", "stride", "unfreeze_b0b4")}
        except Exception as e:
            st = {"ERR": str(e)[:60]}
    if os.path.isfile(gr):
        j = json.load(open(gr)); tr = j.get("tracks", {})
        t1 = (tr.get("track1_uav") or {}).get("finetune_peak") or {}
        t2 = tr.get("track2_golden") or {}
        t3 = tr.get("track3_blind") or {}
        g = "T1=%s (tp=%s fp=%s) T2=%s T3=%s verdict=%s" % (
            t1.get("f1"), t1.get("tp"), t1.get("fp"), t2.get("f1"), t3.get("f1"), j.get("verdict"))
    else:
        g = "(无门禁报告)"
    print("%s: %s" % (t, st))
    print("    " + g)
