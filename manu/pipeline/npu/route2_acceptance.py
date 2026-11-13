#!/usr/bin/env python3
"""路线二候选臂的**联合验收流水线**（速度 × 微小目标召回）。

一条命令跑完：
    ① 按候选 epoch 列表建 no-GMC 缓存（T1 val 31613 帧）
    ② NanoBench-T1 分层评测（nano/micro/small/regular + F1）
    ③ 每臂选 F1 最好的 epoch ⇒ 导出 ONNX → 转 int8 rknn
    ④ **一次**板卡会话测完全部候选臂 + 控制臂（板卡驱动脆弱 ⇒ 必须一次测完）
    ⑤ 联合门禁判定（int8 ≤30 ms 且 nano ≥ 下限）⇒ JSON + 终端表

用法（服务器）：
    python manu/pipeline/npu/route2_acceptance.py \
        --arm 9603=/mnt/data/.../trial_9603/weights \
        --arm 9643=/mnt/data/.../trial_9643/weights \
        --epochs 10,20,30,40,50 \
        --gpu 0 --budget-ms 30 --nano-min 0.6978

设计约束：
* **口径红线**：全部走 no-GMC（`uav_gmc_median_nogmc`）；GMC 与 no-GMC 严禁混用。
* **板卡一次会话**：测速前先归档板上 arms/（否则会把本轮臂一起扫走），再上传。
* **缓存与参照同源**：val 切分恰好 31613 帧 ⇒ 与 NanoBench 参照卡逐位可比。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
PY = "/home/huangzhe/anaconda3/envs/uav/bin/python"
PY_RKNN = "/home/huangzhe/anaconda3/envs/rknn/bin/python"
RUNS = "/mnt/data/siping/runs/npu20ms"
WORK = f"{RUNS}/route2_acceptance"
TINY = f"{RUNS}/tiny_ref"
DATA_YAML = "/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml"
BASE_CACHE = "/mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl"
CALIB = f"{RUNS}/_chain_test/calib_nogmc_train_server.txt"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(WORK, exist_ok=True)
    with open(f"{WORK}/acceptance.log", "a") as f:
        f.write(line + "\n")


def sh(cmd: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, executable="/bin/bash",
                          capture_output=True, text=True, timeout=timeout)


def build_cache(ckpt: str, out_pkl: str, gpu: int, extra: str = "") -> bool:
    """建 no-GMC 缓存（口径与参照卡一致：val 31613 帧 / conf 0.02 / top-k 100）。

    ``extra``：追加到建缓存命令的额外开关（默认空 ⇒ 行为完全不变）。
    🟢 2026-11-12 17:00：NanoBench 路径传 ``--no-peak-attrs`` ⇒ 跳过连通域 area/sums 计算
       （实测那一步占建缓存 CPU 时间的绝大部分），``pred_points`` / ``pred_scores`` **逐位不变**
       （同一 ``extract_peaks`` 调用），而 ``tiny_target_subset.eval_cache`` **不读**
       ``pred_area`` / ``pred_sums`` ⇒ NanoBench 读数不变。
    """
    if os.path.isfile(out_pkl):
        log(f"缓存已存在，跳过：{out_pkl}")
        return True
    cmd = (f"cd {REPO} && export PYTHONPATH=. && {PY} -u manu/inference/cache_trial0474_inferences.py "
           f"--weights {ckpt} --data {DATA_YAML} --output {out_pkl} --imgsz 640 --stride 2 "
           f"--batch 32 --device {gpu} --conf-thresh 0.02 --top-k 100 --ablate-p0"
           + (f" {extra}" if extra else ""))
    r = sh(cmd)
    ok = os.path.isfile(out_pkl)
    log(f"[cache] {os.path.basename(ckpt)} -> {'OK' if ok else 'FAIL rc=' + str(r.returncode)} "
        f"{(r.stdout or '')[-300:]}{(r.stderr or '')[-300:] if not ok else ''}")
    return ok


def nanobench(cache: str, label: str) -> dict | None:
    """跑 NanoBench-T1（no-GMC）分层评测。"""
    sys.path.insert(0, REPO)
    from manu.evaluation.tiny_target_subset import eval_cache  # noqa: PLC0415

    try:
        res = eval_cache(cache, "nogmc", tols=(4.0, 8.0), img_w=640, img_h=512, label=label)
    except Exception as e:  # noqa: BLE001
        log(f"[nanobench] {label} 异常：{e}")
        return None
    p8 = res["peak_tol8"]
    rb = p8["recall_by_bucket"]
    log(f"[nanobench] {label}: F1={p8['f1']:.6f} th={p8['th']} FP={p8['fp']} FN={p8['fn']} | "
        f"nano={rb['nano']:.4f} micro={rb['micro']:.4f} small={rb['small']:.4f} regular={rb['regular']:.4f}")
    return res


def export_rknn(ckpt: str, name: str, act_override: str = "") -> tuple[str | None, int]:
    """导出 ONNX → 转 int8 rknn。返回 (rknn 路径, stride)。

    ⚠️ **act 必须跟随 ckpt 自描述**（2026-11-10 修正）：
    历史版本这里硬编码 ``--act relu`` ⇒ 对 **silu 训练的权重**导出会得到「silu 权重 + relu 图」
    的数值错配模型（memory §103.1 实测 **F1 = 0.000000，彻底死模型**）⇒ 交付级事故。
    现在默认读 ckpt 的 ``act`` 字段（``hm_export_onnx.py::build_model`` 同样按 ckpt 重建结构），
    只有显式传 ``act_override="relu"`` 才做 relu 探针（探针只测时延，不可用于验收）。
    """
    import torch  # noqa: PLC0415

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    stride = int(d.get("stride", 2) or 2)
    act = act_override or str(d.get("act", "silu") or "silu")
    onnx = f"{WORK}/onnx/{name}.onnx"
    rknn = f"{WORK}/arms/{name}_int8.rknn"
    os.makedirs(f"{WORK}/onnx", exist_ok=True)
    os.makedirs(f"{WORK}/arms", exist_ok=True)
    if not os.path.isfile(rknn):
        r = sh(f"cd {REPO} && export PYTHONPATH=. && {PY} manu/pipeline/npu/hm_export_onnx.py "
               f"--weights {ckpt} --out {onnx} --act {act} --ablate-p0 --skip-check", timeout=3600)
        if not os.path.isfile(onnx):
            log(f"[export] {name} 导出失败 rc={r.returncode}: {(r.stdout or '')[-300:]}{(r.stderr or '')[-300:]}")
            return None, stride
        r = sh(f"cd {REPO} && {PY_RKNN} manu/pipeline/npu/hm_convert_rknn.py --onnx {onnx} "
               f"--out {rknn} --calib-list {CALIB} --calib-list-only", timeout=7200)
        if not os.path.isfile(rknn):
            log(f"[rknn] {name} 转换失败 rc={r.returncode}: {(r.stdout or '')[-300:]}")
            return None, stride
    log(f"[rknn] {name} OK stride={stride} act={act} size={os.path.getsize(rknn)}B")
    return rknn, stride


def board_measure(arms: list[tuple[str, str, int]]) -> dict[str, float]:
    """**一次**板卡会话测完全部臂（含控制臂）⇒ 返回 {name: rknn_run_ms}。"""
    sys.path.insert(0, REPO)
    from manu.pipeline.npu import board_speed_search as B  # noqa: PLC0415

    built = [dict(name=n, rknn=p, stride=s, note="", hc=0, head_src="", late=0, sc=1.0)
             for n, p, s in arms]
    B.archive_board_arms()   # ⚠️ 必须先归档再上传（否则会扫掉本轮臂）
    B.upload(built)
    r1, _r2, _r3, _txt = B.measure_on_board(WORK)
    out = {}
    for n, _p, _s in arms:
        v = r1.get(f"arm_{n}_int8")
        if v is not None:
            out[n] = v
    log(f"[board] 会话结束：{out} | control={r1.get('control_base_cs_t0043')}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="路线二候选臂联合验收（速度 × 微小召回）")
    ap.add_argument("--arm", action="append", default=[], help="trial=<weights_dir>，可重复")
    ap.add_argument("--epochs", default="20,30,40,50", help="候选 epoch 列表（逗号分隔）")
    ap.add_argument("--gpu", type=int, default=0, help="建缓存用的 GPU")
    ap.add_argument("--budget-ms", type=float, default=30.0)
    ap.add_argument("--nano-min", type=float, default=0.6978)
    ap.add_argument("--skip-board", type=int, default=0)
    ap.add_argument("--act-override", default="",
                    help="强制导出的 act（默认 = 跟随 ckpt 自描述）。"
                         "'relu' 仅作**时延探针**（数值错配 ⇒ 死模型），不可用于验收")
    ap.add_argument("--out", default=f"{WORK}/route2_acceptance.json")
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(TINY, exist_ok=True)

    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    arms_spec: dict[str, str] = {}
    for a in args.arm:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        arms_spec[k] = v
    if not arms_spec:
        log("没有 --arm 参数 ⇒ 退出")
        return

    # ---- ① ② 每臂逐 epoch：建缓存 + NanoBench ----
    per_arm: dict[str, list[dict]] = {}
    for trial, wdir in arms_spec.items():
        cand = []
        for ep in epochs:
            ck = os.path.join(wdir, f"epoch{ep}.pt")
            if not os.path.isfile(ck):
                continue
            pkl = f"{TINY}/r2_{trial}_ep{ep}.pkl"
            if not build_cache(ck, pkl, args.gpu):
                continue
            res = nanobench(pkl, f"r2_{trial}_ep{ep}")
            if res:
                cand.append(dict(trial=trial, epoch=ep, ckpt=ck, cache=pkl,
                                 f1=res["peak_tol8"]["f1"], nano=res["peak_tol8"]["recall_by_bucket"]["nano"],
                                 micro=res["peak_tol8"]["recall_by_bucket"]["micro"],
                                 regular=res["peak_tol8"]["recall_by_bucket"]["regular"],
                                 fp=res["peak_tol8"]["fp"], fn=res["peak_tol8"]["fn"],
                                 th=res["peak_tol8"]["th"]))
        per_arm[trial] = cand
        if cand:
            best = max(cand, key=lambda c: c["f1"])
            log(f"★ {trial} 最佳 epoch={best['epoch']} F1={best['f1']:.6f} nano={best['nano']:.4f}")
        else:
            log(f"★ {trial} 无有效 epoch")

    # ---- ③ 每臂选 epoch → rknn ----
    # ⚠️ **选 epoch 不能只看 F1**：实测 9603 的 nano 随训练**下降**（ep3 0.6281 → ep9 0.5971），
    #    而 F1 走平 ⇒ 纯 F1 选臂会挑到 nano 最差的晚期。
    #    判据：**先取「nano ≥ 下限」的 epoch 集合，其中 F1 最高者**；若一个都没有 ⇒ 取 nano 最高者（并标记 nano 未达标）。
    def pick_epoch(cand: list[dict]) -> tuple[dict, str]:
        if not cand:
            return None, "none"
        ok = [c for c in cand if c["nano"] >= args.nano_min]
        if ok:
            return max(ok, key=lambda c: c["f1"]), "f1_within_nano_floor"
        return max(cand, key=lambda c: c["nano"]), "max_nano_nofloor"

    board_arms: list[tuple[str, str, int]] = []
    picks: dict[str, dict] = {}
    for trial, cand in per_arm.items():
        if not cand:
            continue
        best, how = pick_epoch(cand)
        picks[trial] = dict(best, select_rule=how)
        best_nano = max(cand, key=lambda c: c["nano"])
        log(f"★ {trial} 选 epoch={best['epoch']}（{how}）F1={best['f1']:.6f} nano={best['nano']:.4f} | "
            f"nano 峰值在 epoch{best_nano['epoch']}（nano={best_nano['nano']:.4f}）")
        rknn, stride = export_rknn(best["ckpt"], f"r2_{trial}_ep{best['epoch']}",
                                   act_override=args.act_override)
        if rknn:
            board_arms.append((f"r2_{trial}_ep{best['epoch']}", rknn, stride))

    # ---- ④ 一次板卡会话 ----
    ms_map: dict[str, float] = {}
    if board_arms and not args.skip_board:
        ms_map = board_measure(board_arms)
    elif board_arms:
        log("按 --skip-board 跳过测速")

    # ---- ⑤ 联合门禁 ----
    rows = []
    for trial, cand in per_arm.items():
        if not cand:
            continue
        best = picks[trial]
        best_nano = max(cand, key=lambda c: c["nano"])
        name = f"r2_{trial}_ep{best['epoch']}"
        ms = ms_map.get(name)
        speed = "UNKNOWN" if ms is None else ("PASS" if ms <= args.budget_ms else "FAIL")
        recall = "PASS" if best["nano"] >= args.nano_min else "FAIL"
        verdict = ("FAIL" if "FAIL" in (speed, recall)
                   else ("PASS_WITH_GAP" if speed == "UNKNOWN" else "PASS"))
        rows.append(dict(name=name, trial=trial, epoch=best["epoch"], board_ms=ms,
                         speed_verdict=speed, recall_verdict=recall, verdict=verdict,
                         select_rule=best["select_rule"],
                         nano=best["nano"], micro=best["micro"], regular=best["regular"],
                         f1=best["f1"], fp=best["fp"], fn=best["fn"], th=best["th"],
                         nano_peak_epoch=best_nano["epoch"], nano_peak=best_nano["nano"],
                         nano_peak_f1=best_nano["f1"],
                         cache=best["cache"]))

    out = dict(budget_ms=args.budget_ms, nano_min=args.nano_min, rows=rows,
               all_epochs=per_arm, board_ms=ms_map)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    log(f"written {args.out}")

    print("\n%-22s %-9s %-9s %-9s %-9s %-10s %-9s %s" %
          ("候选", "板卡ms", "SPEED", "nano", "micro", "T1_F1", "RECALL", "总判定"))
    print("-" * 96)
    for r in rows:
        print("%-22s %-9s %-9s %-9.4f %-9.4f %-10.6f %-9s %s" % (
            r["name"], f"{r['board_ms']:.3f}" if r["board_ms"] else "-", r["speed_verdict"],
            r["nano"], r["micro"], r["f1"], r["recall_verdict"], r["verdict"]))
    print(f"\n判定：预算 int8 ≤ {args.budget_ms} ms；nano 下限 {args.nano_min}")


if __name__ == "__main__":
    main()
