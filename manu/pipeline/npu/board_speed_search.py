#!/usr/bin/env python3
"""板卡速度搜索：把候选结构导成 int8 rknn，在 RK3588 上**一次会话**实测延迟。

目标（2026-11-10 新目标）：保留 P1/P2 高分辨率路径 + 通道瘦身 ⇒ int8 ≤ 30 ms，
同时保住微小目标召回。本脚本负责**速度维度**（召回由后续训练负责）。

设计要点：
1. **延迟只由结构决定**（int8 稠密卷积，与权重值无关）⇒ 本轮用「结构探针权重」
   （RC2 权重 + 打补丁的结构元数据，head 形状不匹配的部分被跳过 ⇒ 随机初始化）测延迟是**有效的**。
   这些探针模型**不可交付**，只用来定速度白名单。
2. act 统一 **relu**（用户口径：NPU 推理效率最高）。用官方 `--act relu` 探针通道（TIMING PROBE）。
3. 板卡驱动脆弱 ⇒ **一次会话测完全部臂**；控制臂必带（防驱动漂移）。
4. 每个臂写 `.stride` 边车：160 输出 ⇒ 4；320 输出 ⇒ 2；80 输出 ⇒ 8
   （板上工具会校验 640/stride，不传就默认 2 ⇒ 被 FATAL 拒）。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

RUNS = "/mnt/data/siping/runs/npu20ms"
REPO = "/tmp/pycharm_project_10ae9e2e"
PY = "/home/huangzhe/anaconda3/envs/uav/bin/python"
PY_RKNN = "/home/huangzhe/anaconda3/envs/rknn/bin/python"
RC2 = "/mnt/data/siping/runs/nogmc_weekend/trials/trial_0080/weights/epoch8.pt"
CALIB = f"{RUNS}/_chain_test/calib_nogmc_train_server.txt"
WORK = f"{RUNS}/speed_search"
BOARD = "root@192.168.0.64"
BOARD_DIR = "/root/hm_speed"
KEY = "-i /home/huangzhe/.ssh/id_board -o ConnectTimeout=10 -o StrictHostKeyChecking=no"

# ---- 候选结构（全部 relu；P2 = 4px 特征 / P1 = 2px 特征）--------------------
SPECS = [
    # name,              head_src,    head_level,   hc,  late, sc,  stride, 备注
    ("p2f_hc64",         "p2_fused",  "p2_fused",   64,  0,    1.0,  4, "P2 满宽基线"),
    ("p2f_hc48",         "p2_fused",  "p2_fused",   48,  0,    1.0,  4, "通道瘦身"),
    ("p2f_hc40",         "p2_fused",  "p2_fused",   40,  0,    1.0,  4, "中间宽度"),
    ("p2f_hc32",         "p2_fused",  "p2_fused",   32,  0,    1.0,  4, "深度瘦身"),
    ("p2f_hc24",         "p2_fused",  "p2_fused",   24,  0,    1.0,  4, "极窄"),
    ("p2f_hc64_late",    "p2_fused",  "p2_fused",   64,  1,    1.0,  2, "320 输出(160 计算)"),
    ("p2f_hc32_late",    "p2_fused",  "p2_fused",   32,  1,    1.0,  2, "★P2+late+瘦身"),
    ("p2f_hc32_late_s05", "p2_fused", "p2_fused",   32,  1,    0.5,  2, "neck 再收窄"),
    ("p1_hc64",          "p1",        "p1",         64,  0,    1.0,  2, "P1 满宽对照(≈55ms)"),
    ("p1_hc32",          "p1",        "p1",         32,  0,    1.0,  2, "P1 瘦身(召回上限候选)"),
]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(f"{WORK}/speed_search.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout=3600):
    return subprocess.run(cmd, shell=True, executable="/bin/bash",
                          capture_output=True, text=True, timeout=timeout)


def build_probe_ckpt(spec, work=WORK):
    """给 RC2 ckpt 打结构元数据补丁 ⇒ 导出器按新结构重建图。"""
    import torch
    name, head_src, head_level, hc, late, sc, stride, _ = spec
    d = torch.load(RC2, map_location="cpu", weights_only=False)
    d["head_src"] = head_src
    d["head_level"] = head_level
    d["head_conv"] = hc
    d["late_upsample"] = bool(late)
    d["p2_out_scale"] = sc
    d["no_p2"] = False
    d["stride"] = stride
    d["probe"] = True          # 标记：结构探针，权重随机，**不可交付**
    out = f"{work}/probes/{name}.pt"
    torch.save(d, out)
    return out


def export_and_convert(spec, work=WORK):
    name, head_src, head_level, hc, late, sc, stride, note = spec
    ck = build_probe_ckpt(spec, work)
    onnx = f"{work}/onnx/{name}.onnx"
    rknn = f"{work}/arms/{name}_int8.rknn"
    r = sh(f"cd {REPO} && export PYTHONPATH=. && {PY} manu/pipeline/npu/hm_export_onnx.py "
           f"--weights {ck} --out {onnx} --act relu --ablate-p0 --skip-check", timeout=1800)
    if not os.path.isfile(onnx):
        log(f"[{name}] 导出失败 rc={r.returncode}: {(r.stdout or '')[-400:]} {(r.stderr or '')[-400:]}")
        return None
    r = sh(f"cd {REPO} && {PY_RKNN} manu/pipeline/npu/hm_convert_rknn.py --onnx {onnx} "
           f"--out {rknn} --calib-list {CALIB} --calib-list-only", timeout=3600)
    if not os.path.isfile(rknn):
        log(f"[{name}] 转 rknn 失败 rc={r.returncode}: {(r.stdout or '')[-400:]}")
        return None
    log(f"[{name}] OK  onnx={os.path.getsize(onnx)}B rknn={os.path.getsize(rknn)}B  ({note})")
    return dict(name=name, rknn=rknn, stride=stride, note=note, hc=hc,
                head_src=head_src, late=late, sc=sc)


def upload(built):
    """把 rknn + .stride 边车传到板上（不测速，只是搬文件）。"""
    for b in built:
        sh(f"scp {KEY} {b['rknn']} {BOARD}:{BOARD_DIR}/arms/", timeout=600)
        sh(f"ssh {KEY} {BOARD} 'printf {b['stride']} > {BOARD_DIR}/arms/{b['name']}_int8.stride'", timeout=120)
    log(f"已上传 {len(built)} 个臂到板卡")


def archive_board_arms():
    """把板上 arms/ 里已有的 .rknn 归档到 arms_archive/。

    为什么必须归档：板上 measure_hm_speed_board.sh 遍历 **arms/ 下全部 *.rknn** 逐个测速，
    而板卡驱动脆弱（长会话 → job timeout → soft reset → rknn_init 132，只能物理断电恢复）。
    ⇒ 一次会话只装本轮要测的臂，否则会话被拉长、失败面变大。
    """
    r = sh(f"ssh {KEY} {BOARD} 'cd {BOARD_DIR} && mkdir -p arms_archive && "
           f"for f in arms/*.rknn; do [ -e \"$f\" ] && mv \"$f\" arms_archive/ 2>/dev/null; done; "
           f"for f in arms/*.stride; do [ -e \"$f\" ] && mv \"$f\" arms_archive/ 2>/dev/null; done; "
           f"echo arms=$(ls arms/ | wc -l) archived=$(ls arms_archive/ | wc -l)'", timeout=300)
    log(f"板上旧臂归档：{((r.stdout or '').strip())}")


def load_specs(specs_file):
    """从 JSON 文件读探针集：[[name, head_src, head_level, hc, late, sc, stride, note], ...]"""
    with open(specs_file) as f:
        raw = json.load(f)
    out = []
    for it in raw:
        if len(it) < 8:
            raise ValueError(f"探针定义需要 8 个字段，实际 {len(it)}: {it}")
        out.append(tuple(it))
    return out


def measure_on_board(work=WORK):
    """一次会话测完全部臂（板上 arms/ 下所有 *.rknn，含控制臂）。"""
    r = sh(f"ssh {KEY} {BOARD} 'cd {BOARD_DIR} && BOARD_DIR={BOARD_DIR} CONTROL_NAME=base_int8.rknn "
           f"LOOPS=30 WARMUP=10 SETTLE=5 bash measure_hm_speed_board.sh'", timeout=1800)
    txt = (r.stdout or "") + (r.stderr or "")
    open(f"{work}/board_session.txt", "w").write(txt)
    res = {}
    for m in re.finditer(r"@@SUMMARY@@ (\S+)\|rknn_run_avg_ms\|([\d.]+)", txt):
        res[m.group(1)] = float(m.group(2))
    res2 = {}
    for m in re.finditer(r"@@SUMMARY@@ (\S+)\|npu_dequant_avg_ms\|([\d.]+)", txt):
        res2[m.group(1)] = float(m.group(2))
    res3 = {}
    for m in re.finditer(r"@@SUMMARY@@ (\S+)\|full_frame_avg_ms\|([\d.]+)", txt):
        res3[m.group(1)] = float(m.group(2))
    return res, res2, res3, txt


def main():
    ap = argparse.ArgumentParser(description="板卡速度搜索（结构探针 → rknn → 一次会话测速）")
    ap.add_argument("--specs-file", default="", help="自定义探针集 JSON（缺省用内置 SPECS）")
    ap.add_argument("--work", default=WORK, help="工作目录（默认 speed_search）")
    ap.add_argument("--out-json", default=f"{RUNS}/board_speed_search.json")
    ap.add_argument("--archive-arms", type=int, default=1,
                    help="测速前归档板上 arms/ 里已有 rknn（默认 1：保证一次会话只测本轮臂）")
    ap.add_argument("--skip-board", type=int, default=0, help="只建探针不测速")
    args = ap.parse_args()

    work = args.work
    specs = load_specs(args.specs_file) if args.specs_file else SPECS

    os.makedirs(f"{work}/probes", exist_ok=True)
    os.makedirs(f"{work}/onnx", exist_ok=True)
    os.makedirs(f"{work}/arms", exist_ok=True)
    log(f"=== 板卡速度搜索 START：{len(specs)} 个结构探针（全部 relu，基线 RC2 结构元数据）===")

    built = []
    from concurrent.futures import ThreadPoolExecutor

    def work_one(spec):
        try:
            return export_and_convert(spec, work)
        except Exception as e:
            log(f"[{spec[0]}] 异常: {e}")
            return None

    with ThreadPoolExecutor(max_workers=4) as ex:
        for b in ex.map(work_one, specs):
            if b:
                built.append(b)

    if not built:
        log("没有可用探针 ⇒ 终止")
        return

    # ⚠️ 归档必须在上传**之前**：归档会把 arms/ 下全部 *.rknn 与 *.stride 一起搬走，
    #    若先上传再归档，本轮刚建的臂会被自己扫掉（实测：会话 9s 结束、所有臂 null）。
    if args.archive_arms:
        archive_board_arms()

    upload(built)

    if args.skip_board:
        log("=== 按 --skip-board 只建探针，不开板卡会话 ===")
        return

    log("板上一次会话测速中（驱动脆弱，一次测完）…")
    t0 = time.time()
    r1, r2, r3, txt = measure_on_board(work)
    log(f"板卡会话结束，耗时 {time.time()-t0:.0f}s")

    rows = []
    for b in built:
        key = f"arm_{b['name']}_int8"
        rows.append(dict(name=b["name"], note=b["note"], head_src=b["head_src"], hc=b["hc"],
                         late=b["late"], p2_out_scale=b["sc"], stride=b["stride"],
                         rknn_run_ms=r1.get(key), npu_dequant_ms=r2.get(key),
                         full_frame_ms=r3.get(key),
                         under_30=(r1.get(key) is not None and r1[key] <= 30.0)))
    ctrl = r1.get("control_base_cs_t0043")
    out = dict(control_rknn_run_ms=ctrl, budget_ms=30.0, rows=rows,
               raw={k: v for k, v in r1.items()})
    with open(args.out_json, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n%-22s %-12s %-9s %-8s %-9s %-9s %s" % ("arm", "结构", "hc", "stride", "rknn_ms", "整帧ms", "≤30ms"))
    print("-" * 88)
    print("%-22s %-12s %-9s %-8s %-9s %-9s %s" % ("control(cs_t0043)", "p1/silu", "-", "2",
          f"{ctrl:.3f}" if ctrl else "-", "-", "参照"))
    for r in rows:
        print("%-22s %-12s %-9s %-8s %-9s %-9s %s" % (
            r["name"], r["head_src"] + ("+late" if r["late"] else ""), r["hc"], r["stride"],
            f"{r['rknn_run_ms']:.3f}" if r["rknn_run_ms"] else "-",
            f"{r['full_frame_ms']:.3f}" if r["full_frame_ms"] else "-",
            "★" if r["under_30"] else ""))
    log("=== 板卡速度搜索 DONE ===")


if __name__ == "__main__":
    main()
