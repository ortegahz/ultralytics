#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
结构谱系探针：把「结构 → 板卡 NPU 实测时间」测成一张表，并拟合出**离线速度预测器**。

为什么需要它
------------
新目标（2026-11-06）：**NPU int8 ≤ 20 ms** 且精度下降非常微小，deadline 下周一 09:00，
**板卡今天 18:30 断电**，而服务器训练窗口有 ~62.5 h（≈75 条臂的预算）。

⇒ **板卡测速是一次性稀缺资源**，不可能给 75 条臂每条都测。
⇒ 但 **NPU 推理时间只取决于「结构」（图的形状/算子数/搬运量），不取决于「权重」**
   （唯一例外是塌陷模型被常量折叠，那是病态，必须用 null control 排除）。
⇒ 所以正确的分工是：
   * **今天（板卡在线）**：把候选**结构**谱系测全 ⇒ 得到「结构 → 实测 ms」表 + 离线预测器；
   * **周末（板卡离线）**：训练出的任意臂，结构已知 ⇒ **速度查表 / 预测**；
     精度用服务器上的三轨门禁测。

本脚本做的事
------------
1. 用 `cs_t0043` 的权重直接搭出每个探针结构（**不训练**，只测速度）
   ⇒ 继承率会打印出来（结构变了 ⇒ 权重形状不变 ⇒ 头仍可继承）
2. 导出 ONNX（按探针决定是否 `--ablate-p0`）→ 转 RKNN int8
3. 从构建日志解析 **NPU 周期 / DDR 周期 / RW(MB)**（RKNN 自带的算子表）
4. 上板实测 `rknn_run`（一次会话打包全部探针，避开驱动脆弱）
5. 产出 `struct_speed_table.json` + `struct_speed_table.md`，
   并用所有点拟合 `ms = a + b*cycles(M) + c*RW(MB)`

用法（宿主，rknn230 环境）：
    python manu/pipeline/npu/probe_struct_speed.py --specs manu/pipeline/npu/struct_speed_specs.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE = REPO_ROOT / "manu" / "pipeline" / "npu"
HERE = Path(__file__).resolve().parent
CS_T0043 = "/home/manu/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt"
CALIB = "/home/manu/mnt/data/siping/runs/nogmc_struct2/npu_speed/calib_nogmc_train.txt"
BOARD = "root@192.168.0.64"
BOARD_DIR = "/root/hm_speed"
GOOD_LIB = "/run/media/manu/SSD960-LOCAL/nfs/npu/librknnrt.so"
GOOD_MD5 = "56e425dcc7b59ab0e1aec685b178b99f"

CYCLE_RE = re.compile(
    r'^D RKNN: \[[0-9:.]+\]\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+.*?\s(\d+)/(\d+)/(\d+)\s+(\d+)\s+(\S+)\s*$'
)



def _stride_of(spec: dict) -> int:
    """板卡工具校验 input/grid ratio == 640/stride ⇒ stride 必须按结构推。"""
    hs = spec.get("head_src", "p1")
    if spec.get("late_upsample"):
        return 2                      # 头在 160 算、2x 上采样回 320 ⇒ 口径是 stride 2
    return {"p1": 2, "p2_fused": 4, "p2_raw": 4, "p3": 8}.get(hs, 2)


def build_probe_ckpt(spec: dict, out: Path) -> int:
    """用 cs_t0043 的权重搭出探针结构（不训练）。返回继承的张量数。"""
    import torch

    from manu.models.heatmap_model import YOLO26HeatmapDetector

    ck = torch.load(CS_T0043, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    m = YOLO26HeatmapDetector(
        stride=2,
        head_src=spec.get("head_src", "p1"),
        act=spec.get("act", "relu"),
        head_conv=spec.get("head_conv", 64),
        late_upsample=bool(spec.get("late_upsample", False)),
        p2_out_scale=float(spec.get("p2_out_scale", 1.0)),
    )
    own = m.state_dict()
    matched = 0
    for k, v in sd.items():
        c = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if c in own and own[c].shape == v.shape:
            own[c].copy_(v)
            matched += 1
    new_state = dict(m.state_dict())
    # ⚠️ p0 张量要带回（与训练侧保存逻辑一致）⇒ 导出时 --ablate-p0 才有东西可 drop
    p0 = {k: v for k, v in sd.items() if k.startswith("p0_highway")}
    new_state.update(p0)
    new_ck = dict(ck)
    new_ck["model"] = new_state
    new_ck.update({
        "stride": m.stride,
        "p2_out_scale": float(spec.get("p2_out_scale", 1.0)),
        "no_p2": spec.get("head_src") in ("p3", "p2_raw"),
        "head_level": spec.get("head_src", "p1"),
        "act": spec.get("act", "relu"),
        "head_conv": spec.get("head_conv", 64),
        "late_upsample": bool(spec.get("late_upsample", False)),
        "head_src": spec.get("head_src", "p1"),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ck, out)
    return matched


def parse_cycles(log: Path) -> dict:
    rows = []
    if not log.is_file():
        return {}
    for ln in log.read_text(errors="ignore").splitlines():
        m = CYCLE_RE.match(ln)
        if m:
            _, op, dt, tgt, ddr, npu, tot, rw, name = m.groups()
            rows.append((op, tgt, int(ddr), int(npu), int(rw)))
    if not rows:
        return {}
    return {
        "ops": len(rows),
        "npu_cycles": sum(r[3] for r in rows),
        "ddr_cycles": sum(r[2] for r in rows),
        "rw_mb": sum(r[4] for r in rows) / 1024.0,
        "cpu_ops": sum(1 for r in rows if r[1] == "CPU"),
    }


def sh(cmd: list[str], log: Path | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if log:
        log.write_text(out)
    return p.returncode, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=str(HERE / "struct_speed_specs.json"))
    ap.add_argument("--outdir", default="/media/manu/1TB-Volume/rknn_work/hm_speed/probes")
    ap.add_argument("--only", default="", help="逗号分隔的探针名")
    ap.add_argument("--no-board", action="store_true", help="只做导出/转换，不上板")
    args = ap.parse_args()

    specs = json.loads(Path(args.specs).read_text(encoding="utf-8"))
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        specs = [s for s in specs if s["name"] in want]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logs = outdir / "logs"
    logs.mkdir(exist_ok=True)

    # ---------- 1) 搭 ckpt + 导出 + 转换（宿主） ----------
    arms = []
    for sp in specs:
        name = sp["name"]
        ck = outdir / f"{name}.pt"
        onnx = outdir / f"{name}.onnx"
        rknn = outdir / f"{name}_int8.rknn"
        print(f"\n=== {name}: {sp}", flush=True)
        if not ck.is_file():
            n = build_probe_ckpt(sp, ck)
            print(f"  继承 {n} 个张量（不训练，只为测速）", flush=True)
        if not onnx.is_file():
            cmd = [sys.executable, "-u", str(HERE / "hm_export_onnx.py"), "--weights", str(ck),
                   "--out", str(onnx), "--skip-check"]
            if sp.get("ablate_p0", True):
                cmd.append("--ablate-p0")
            rc, out = sh(cmd, logs / f"export_{name}.log")
            if rc != 0:
                print(f"  ❌ export 失败：\n{out[-800:]}", flush=True)
                continue
        if not rknn.is_file():
            rc, out = sh([sys.executable, "-u", str(HERE / "hm_convert_rknn.py"),
                          "--onnx", str(onnx), "--out", str(rknn),
                          "--calib-list", CALIB, "--calib-list-only"], logs / f"convert_{name}.log")
            if rc != 0:
                print(f"  ❌ convert 失败：\n{out[-800:]}", flush=True)
                continue
        cyc = parse_cycles(logs / f"convert_{name}.log")
        print(f"  rknn {rknn.stat().st_size/1e6:.2f} MB | 算子 {cyc.get('ops')} | "
              f"NPU周期 {cyc.get('npu_cycles',0)/1e6:.2f} M | RW {cyc.get('rw_mb',0):.1f} MB", flush=True)
        arms.append({"name": name, "spec": sp, "rknn": str(rknn), "cycles": cyc})

    (outdir / "arms.json").write_text(json.dumps(arms, indent=2, ensure_ascii=False) + "\n")
    if args.no_board:
        print("\n[OK] 只做导出/转换（--no-board）")
        return 0

    # ---------- 2) 上板一次会话测全部 ----------
    ssh = ["ssh", "-p", "22", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    scp = ["scp", "-P", "22", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    # 2.1 好库就位
    rc, out = sh(ssh + [BOARD, f"md5sum {BOARD_DIR}/lib/librknnrt.so 2>/dev/null | cut -d' ' -f1"])
    if GOOD_MD5 not in out:
        print("  暂存已知可用 runtime…", flush=True)
        sh(ssh + [BOARD, f"mkdir -p {BOARD_DIR}/lib"])
        rc, out = sh(scp + [GOOD_LIB, f"{BOARD}:{BOARD_DIR}/lib/librknnrt.so"])
        if rc != 0:
            print(f"  ❌ 暂存 runtime 失败：{out}", flush=True)
            return 3
    # 2.2 暂存 arms
    sh(ssh + [BOARD, f"mkdir -p {BOARD_DIR}/arms && rm -f {BOARD_DIR}/arms/*"])
    for a in arms:
        rc, out = sh(scp + [a["rknn"], f"{BOARD}:{BOARD_DIR}/arms/"])
        if rc != 0:
            print(f"  ❌ scp {a['name']} 失败：{out}", flush=True)
            return 3
        # stride 边车（板卡工具校验 input/grid ratio == 640/stride）
        st = a["spec"].get("stride") or _stride_of(a["spec"])
        sh(ssh + [BOARD, f"echo {st} > {BOARD_DIR}/arms/{Path(a['rknn']).stem}.stride"])
    # 2.3 一次会话测完（脚本**本地写 + scp + 执行** ⇒ 避免长命令行的 ssh 引号/长度问题）
    lines = [f"cd {BOARD_DIR}", "export LD_LIBRARY_PATH=$BOARD_DIR/lib:/usr/lib"]
    for a in arms:
        nm = Path(a["rknn"]).stem
        st = a["spec"].get("stride") or _stride_of(a["spec"])
        lines.append(f"echo '===== {a['name']} stride={st} ====='")
        lines.append(f"./rknn_heatmap_smoke {BOARD_DIR}/arms/{nm}.rknn {BOARD_DIR}/feat.jpg "
                     f"--loops 30 --warmup 10 --stride {st} --tag {nm} 2>&1 | "
                     f"grep -E 'rknn_run|NPU.dequant|FATAL|CHANGED'")
    script = "\n".join(lines) + "\n"
    local_sh = outdir / "probe_board.sh"
    local_sh.write_text(script)
    rc, out = sh(scp + [str(local_sh), f"{BOARD}:{BOARD_DIR}/probe_board.sh"])
    if rc != 0:
        print(f"  ❌ scp 板卡脚本失败：{out}", flush=True)
        return 3
    rc, out = sh(ssh + [BOARD, f"bash {BOARD_DIR}/probe_board.sh"])
    print("\n" + out, flush=True)
    (outdir / "board_session.log").write_text(out)

    # ---------- 3) 解析 + 拟合 ----------
    res = {}
    for ln in out.splitlines():
        m = re.match(r"\s*rknn_run\s+avg\s+([0-9]+\.[0-9]+)", ln)
        if m:
            res.setdefault("_last", {})["ms"] = float(m.group(1))
    # 按 "===== name =====" 分段重新解析
    cur = None
    for ln in out.splitlines():
        m = re.match(r"=====\s+(\S+)\s+stride=(\d+)", ln)
        if m:
            cur = m.group(1)
            res.setdefault(cur, {})
            continue
        m2 = re.match(r"\s*rknn_run\s+avg\s+([0-9]+\.[0-9]+)", ln)
        if m2 and cur:
            res[cur]["ms"] = float(m2.group(1))
        if cur and "FATAL" in ln:
            res[cur]["fatal"] = ln.strip()
    for a in arms:
        r = res.get(a["name"], {})
        a["ms"] = r.get("ms")
        a["fatal"] = r.get("fatal")
    (outdir / "arms.json").write_text(json.dumps(arms, indent=2, ensure_ascii=False) + "\n")

    # ---------- 4) 表 + 预测器 ----------
    print("\n" + "=" * 118)
    print(f"{'探针':<22} {'head_src':<10} {'hc':>4} {'P0':>4} {'act':>5} {'算子':>5} "
          f"{'NPU周期M':>10} {'RW(MB)':>8} {'实测ms':>9} {'预测ms':>9} {'残差':>8}")
    print("-" * 118)
    pts = []
    for a in arms:
        s, c = a["spec"], a["cycles"]
        ms = a.get("ms")
        if ms is not None:
            pts.append((c["npu_cycles"] / 1e6, c["rw_mb"], ms))
        print(f"{a['name']:<22} {s.get('head_src','p1'):<10} {s.get('head_conv',64):>4} "
              f"{('off' if s.get('ablate_p0',True) else 'ON'):>4} {s.get('act','relu'):>5} "
              f"{c.get('ops',0):>5} {c.get('npu_cycles',0)/1e6:>10.2f} {c.get('rw_mb',0):>8.1f} "
              f"{(ms if ms is not None else float('nan')):>9.2f} "
              f"{'':>9} {'':>8}" + (f"  FATAL: {a.get('fatal','')}" if a.get("fatal") else ""))
    print("=" * 118)

    fit = None
    if len(pts) >= 3:
        fit = lstsq3(pts)
        print(f"\n【离线速度预测器】ms ≈ {fit[0]:.2f} + {fit[1]:.3f}×NPU周期(M) + {fit[2]:.3f}×RW(MB)")
        for cyc, rw, ms in pts:
            pred = fit[0] + fit[1] * cyc + fit[2] * rw
            print(f"   cycles={cyc:>6.2f}M RW={rw:>6.1f}MB  实测={ms:>7.2f}  预测={pred:>7.2f}  残差={ms-pred:>+6.2f}")
        fitd = {"a": fit[0], "b": fit[1], "c": fit[2], "n_points": len(pts)}
        (outdir / "speed_predictor.json").write_text(json.dumps(fitd, indent=2) + "\n")
        for a in arms:
            c = a["cycles"]
            if a.get("ms") is not None:
                a["pred_ms"] = fit[0] + fit[1] * c["npu_cycles"] / 1e6 + fit[2] * c["rw_mb"]
                a["pred_resid"] = a["ms"] - a["pred_ms"]

    tbl = {"arms": arms, "predictor": fit}
    (outdir / "struct_speed_table.json").write_text(json.dumps(tbl, indent=2, ensure_ascii=False) + "\n")
    md = ["# 结构谱系 → 板卡 NPU 速度表（2026-11-06，同板同日）", "",
          "| 探针 | head_src | head_conv | P0 | act | 算子 | NPU周期(M) | RW(MB) | 实测ms | 预测ms |",
          "|:--|:--|--:|:--|:--|--:|--:|--:|--:|--:|"]
    for a in arms:
        s, c = a["spec"], a["cycles"]
        md.append(f"| `{a['name']}` | {s.get('head_src','p1')} | {s.get('head_conv',64)} | "
                  f"{'off' if s.get('ablate_p0',True) else 'ON'} | {s.get('act','relu')} | {c.get('ops',0)} | "
                  f"{c.get('npu_cycles',0)/1e6:.2f} | {c.get('rw_mb',0):.1f} | "
                  f"{a.get('ms') if a.get('ms') is not None else 'FATAL'} | "
                  f"{a.get('pred_ms', float('nan')):.2f} |")
    if fit:
        md += ["", f"**离线预测器**：`ms ≈ {fit[0]:.2f} + {fit[1]:.3f}×NPU周期(M) + {fit[2]:.3f}×RW(MB)`"]
    (outdir / "struct_speed_table.md").write_text("\n".join(md) + "\n")
    print(f"\n[OK] 表 -> {outdir/'struct_speed_table.md'}")
    return 0


def lstsq3(pts):
    """最小二乘拟合 ms = a + b*cycles(M) + c*RW(MB)。"""
    A = [[1.0, p[0], p[1]] for p in pts]
    y = [p[2] for p in pts]
    n = 3
    ATA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] for i in range(n)]
    ATy = [sum(A[k][i] * y[k] for k in range(len(A))) for i in range(n)]
    M = [ATA[i][:] + [ATy[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i and M[i][i] != 0:
                f = M[r][i] / M[i][i]
                for c in range(i, n + 1):
                    M[r][c] -= f * M[i][c]
    return [M[i][n] / M[i][i] for i in range(n)]


if __name__ == "__main__":
    raise SystemExit(main())
