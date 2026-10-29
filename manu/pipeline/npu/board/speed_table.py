import os, re

ARMS = [
 ("CANON8_opportunistic", "融合·旧二进制·GMC 4-7/4anchor"),
 ("CANON14_fused_ref",     "融合·新二进制·GMC 4-7/4anchor"),
 ("CANON15_fused",         "融合·用户拓扑  GMC 4-5/2anchor"),
 ("CANON13_dec",           "解耦·用户拓扑  GMC 4-5/2anchor"),
 ("CANON16_dec_gmc47",     "解耦·GMC 满核  GMC 4-7/4anchor"),
 ("CANON21_dec_a55",       "解耦·阶段下沉A55 GMC 4-7/4anchor"),
 ("CANON19_dec_jit67",     "解耦·压占6-7"),
 ("CANON20_fused_jit67",   "融合·压占6-7"),
 ("CANON17_dec_jit",       "解耦·全核饱和压占"),
 ("CANON22_dec_bufa16",    "解耦·深度3(实际)"),
 ("CANON23_dec_a16",       "解耦·1:1·深度16"),
 ("CANON26_dec_xsvc_d3",   "解耦·跨投放·深度3"),
 ("CANON27_dec_xsvc_d8",   "解耦·跨投放·深度8"),
]
SEC = ["gmc_total","gmc_post","gmc_fit","rga_import","rga_improcess","upload","rknn_run",
       "outputs_get","backproject_cc","total_wall","post_wait","gmc_wait","gmc_ingest_in",
       "gmc_ingest_window","gmc_ingest_space","npu_wait"]
KV = ["frames","gmc_groups_done","features_dropped","detections_dropped","dropped",
      "bufA_dropped","bufA_full_drops","bufB_dropped","npu_util_pct","fps",
      "stage1_ms","stage1_wait_ms","stage2_ms","stage2_wait_ms","bufA_upload_ms","npu_busy_ms"]

def parse(path):
    t = open(path).read()
    d = {}
    for k in KV:
        m = re.search(r'^%s=([-\d.]+)' % re.escape(k), t, re.M)
        if m: d[k] = float(m.group(1))
    for k in SEC:                      # section rows:  name  TOTAL_ms  avg_ms
        m = re.search(r'^%s\s+([\d.]+)\s+([\d.]+)\s*$' % re.escape(k), t, re.M)
        if m: d[k] = (float(m.group(1)), float(m.group(2)))
    return d

def main():
    rows = []
    for arm, desc in ARMS:
        p = '/root/psearch/results/%s/timing.txt' % arm
        if not os.path.exists(p):
            continue
        d = parse(p)
        wall = d["total_wall"][0]          # total ms of the run
        fr = d.get("frames") or 0
        gmc_rate = (d.get("gmc_groups_done") or 0) / (wall / 1000.0)

        # ---- Stage 2 (NPU) capacity -------------------------------------
        s2ms = d.get("stage2_ms")          # KV: total ms over all frames
        if s2ms and s2ms > 0:              # decoupled
            per_frame = s2ms / fr if fr else 0.0
            cap = 2 * 1000.0 / per_frame  # two NPU services
            busy = 100.0 * s2ms / (wall * 2)
            src = "解耦"
        else:                             # fused: upload+run on the NPU thread
            per_frame = d["upload"][1] + d["rknn_run"][1] + d["outputs_get"][1]
            cap = 2 * 1000.0 / per_frame
            busy = 100.0 * (d["upload"][1] + d["rknn_run"][1]) * fr / (wall * 2)
            src = "融合"
        rows.append((arm, desc, d, fr, wall, gmc_rate, cap, busy, src, per_frame))

    print("=== 各阶段「每帧成本 / 理论产能 / 实测忙闲」 ===\n")
    h = ("%-22s %-32s %6s %8s %9s %8s %8s %8s %8s %7s" %
         ("臂", "拓扑", "fps", "GMC组/s", "Stage2组/s", "upload", "rknn_run", "backproj", "Stage2忙", "丢帧"))
    print(h); print("-" * len(h))
    for arm, desc, d, fr, wall, gmc_rate, cap, busy, src, per in rows:
        drops = (d.get("bufA_dropped") or 0) + (d.get("features_dropped") or 0) + \
                (d.get("bufA_full_drops") or 0) + (d.get("bufB_dropped") or 0)
        print("%-22s %-32s %6.3f %8.2f %9.2f %8.2f %8.2f %8.2f %7.1f%% %7d" % (
            arm, desc, d.get("fps") or 0, gmc_rate, cap,
            d["upload"][1] if "upload" in d else 0.0,
            d["rknn_run"][1] if "rknn_run" in d else 0.0,
            d["backproject_cc"][1] if "backproject_cc" in d else 0.0,
            busy, drops))

    print("\n=== 每个阶段的理论产能（谁最小谁就是瓶颈） ===\n")
    h2 = ("%-22s %10s %10s %10s %10s   %s" %
          ("臂", "GMC组/s", "Stage1组/s", "Stage2组/s", "Stage3组/s", "瓶颈"))
    print(h2); print("-" * len(h2))
    for arm, desc, d, fr, wall, gmc_rate, cap, busy, src, per in rows:
        s1 = 1000.0 / (d["rga_import"][1] + d["rga_improcess"][1] + 0.9) if "rga_import" in d else 0
        s3 = 1000.0 / d["backproject_cc"][1] if "backproject_cc" in d else 0
        caps = {"GMC": gmc_rate, "Stage1": s1, "Stage2": cap, "Stage3": s3}
        bn = min(caps, key=caps.get)
        print("%-22s %10.2f %10.2f %10.2f %10.2f   %s (%.2f)" % (
            arm, gmc_rate, s1, cap, s3, bn, caps[bn]))

main()
