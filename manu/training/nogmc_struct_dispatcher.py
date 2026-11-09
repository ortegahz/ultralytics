#!/usr/bin/env python
"""no-GMC **结构搜索** dispatcher。

搜索线
------
底模/初始化 = ``cs_t0043``（3 粒子同配方 Model Soup）
结构改造   = 删 neck 的 P2 融合级（``--no-p2``）+ 头层级 ``--head-level {p3,p2}``
             + 全部 ReLU（``--act relu``）
目标       = **速度大幅提升 + 精度微量下降**（不是过 T1 红线）

为什么自己写 dispatcher（而不是复用 ``optuna_mix_ratio_search.py``）
--------------------------------------------------------------------
1. ``optuna_mix_ratio_search.py`` 是**上一轮 no-GMC 混合比例搜索**的活 dispatcher，
   与其它 agent 共用 ⇒ 不能改它（会污染那个 study）。
2. 本线要的是**固定结构的受控对照**（不是 TPE 采样）⇒ 需要一个"计划驱动"的调度器：
   先跑完结构扫描，再据此**动态**展开超参阶段。
3. 本线的判据（ΔT3 vs cs_t0043 + 板卡 NPU 实测）不同于上一轮的红线判据。

调度器职责
----------
* **代码就绪闸门**：训练器由主控 agent 并发改造（``--act`` / ``--no-p2`` /
  ``--head-level`` …）。派单前**检查训练器源码是否已含新旗标**，不含就
  **等待并重试**（绝不因为"旗标不存在"就退出）。
* 4 卡并行（一条臂一张卡），每臂独立目录/日志，权重按 epoch 落盘。
* 每臂跑完自动做**一次**三轨门禁（不用训练器自带的每-epoch 门禁 ——
  门禁一次约 20 min，20 臂 × 15 epoch 的每-epoch 门禁会直接吃光预算）。
* **断点续跑**：``arm_summary.json`` 存在 ⇒ 跳过。
* **hard_cut**：过了截止时刻不再派新臂；预计跑不完的臂直接跳过。
* 产出 ``arms.csv`` 与 ``SUMMARY.md``。

用法（一般由 ``launch_nogmc_struct_search.sh`` 调用）::

    python manu/training/nogmc_struct_dispatcher.py \
        --plan runs/nogmc_struct/plan.json \
        --out-root runs/nogmc_struct \
        --gpus 0,1,2,3 \
        --hard-cut "2026-11-06 08:30"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import manu.training.build_nogmc_struct_plan as B  # noqa: E402

# ---------------------------------------------------------------- 固定路径
NOGMC_LQ_DATA = "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/data.yaml"
NOGMC_UAV_DATA = "/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml"
NOGMC_REF_DIR = "/mnt/data/siping/runs/nogmc_ref_weekend"
NOGMC_BASELINE_CACHE = f"{NOGMC_REF_DIR}/uav_cache_base.pkl"

#: no-GMC 语义下的冻结基线（trial_0474）—— 门禁的 ``--base-*`` 注入值
NOGMC_GATE_BASE = {
    "base_track1_f1": 0.9008967482400269,
    "base_track1_tp": 21499,
    "base_track1_fp": 1118,
    "base_track2_f1": 0.981074481074481,
    "base_track2_tp": 16070,
    "base_track2_fp": 52,
    "base_track2_fn": 568,
    "base_track3_f1": 0.7578120617110798,
    "base_track3_tp": 6754,
    "base_track3_fp": 536,
    "base_track3_fn": 3781,
}

#: 训练器必须支持的旗标（缺任一 ⇒ 等待并重试，不退出）
REQUIRED_TRAINER_FLAGS = ("--act", "--no-p2", "--head-level")


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="no-GMC 结构搜索 dispatcher")
    p.add_argument("--plan", default="runs/nogmc_struct/plan.json")
    p.add_argument("--out-root", default="runs/nogmc_struct")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--trainer", default="manu/training/mixed_ft_train.py")
    p.add_argument("--shim", default="manu/training/nogmc_struct_train_shim.py")
    p.add_argument("--gate-script", default="manu/training/auto_epoch_gate_check.py")
    p.add_argument("--hard-cut", default="2026-11-06 08:30",
                   help="硬截止（墙钟）—— 之后不再派新臂")
    p.add_argument("--stop-at", default="", help="软截止 —— 之后不再派新臂（留门禁/收尾窗口）")
    p.add_argument("--poll-seconds", type=int, default=60,
                   help="训练器未就绪时的重试间隔")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--gate-infer-batch", type=int, default=8)
    p.add_argument("--gate-cache-batch", type=int, default=4)
    p.add_argument("--force", action="store_true", help="忽略已有 arm_summary.json 重跑")
    p.add_argument("--dry-run", action="store_true", help="只打印将要派发的臂，不训练")
    return p.parse_args()


# ==============================================================================
# 训练器就绪探测
# ==============================================================================
def trainer_flags(trainer: Path) -> set[str]:
    try:
        text = trainer.read_text(encoding="utf-8")
    except OSError:
        return set()
    found = set()
    for flag in ("--act", "--no-p2", "--head-level", "--head-conv", "--late-upsample",
                 "--lr-scale", "--mix-spec"):
        if f'"{flag}"' in text or f"'{flag}'" in text:
            found.add(flag)
    return found


def wait_for_trainer(trainer: Path, hard_cut: float, poll: int) -> bool:
    """等待训练器含新旗标。返回 True=就绪，False=到 hard_cut 仍未就绪。"""
    missing = [f for f in REQUIRED_TRAINER_FLAGS if f not in trainer_flags(trainer)]
    if not missing:
        return True
    waited = 0
    while True:
        now = time.time()
        if now >= hard_cut:
            log(f"[READY-GATE] 到 hard_cut 仍未就绪（缺 {missing}）⇒ 不派单，退出")
            return False
        log(
            f"[READY-GATE] 训练器尚未含 {missing} ⇒ 等待并重试（已等 {waited}s，"
            f"距 hard_cut 还剩 {int(hard_cut - now)}s）"
        )
        time.sleep(min(poll, max(1, int(hard_cut - now))))
        waited += poll
        missing = [f for f in REQUIRED_TRAINER_FLAGS if f not in trainer_flags(trainer)]
        if not missing:
            log("[READY-GATE] 训练器已就绪 ⇒ 开始派单")
            return True


# ==============================================================================
# 阶段解析
# ==============================================================================
def _struct_key(a: dict) -> tuple:
    return (a["head_level"], bool(a["late_upsample"]), int(a["head_conv"]))


def resolve_stage(stage: dict, summaries: dict[str, dict]) -> list[dict]:
    """把「动态阶段」按已完成阶段的结果展开成具体臂列表。"""
    res = stage.get("resolve")
    if not res:
        return list(stage.get("arms", []))

    ok = [
        s for s in summaries.values()
        if s.get("status") == "ok" and s.get("t3_f1") is not None
    ]

    if "pick_structures" in res:
        from_stage = res["from_stage"]
        pool = [s for s in ok if s.get("stage") == from_stage] or ok
        best_per_struct: dict[tuple, dict] = {}
        for s in pool:
            k = _struct_key(s)
            if k not in best_per_struct or (s["t3_f1"] or -1) > (best_per_struct[k]["t3_f1"] or -1):
                best_per_struct[k] = s
        ranked = sorted(
            best_per_struct.values(),
            key=lambda s: (-(s["t3_f1"] or -1), -(s["t1_f1"] or -1)),
        )
        picked = ranked[: int(res["pick_structures"])]
        log(
            f"[RESOLVE] {stage['name']}：从 {len(pool)} 条已完成臂里按 T3 降序挑出 "
            + ", ".join(
                f"{s['head_level']}/lu{int(s['late_upsample'])}/h{s['head_conv']}"
                f"(T3={s['t3_f1']:.6f})" for s in picked
            )
        )
        arms = []
        for s in picked:
            for lr in res["lr_scale"]:
                for mix in res["mix_spec"]:
                    arms.append(B.make_arm(
                        stage=stage["name"],
                        head_level=s["head_level"],
                        late=bool(s["late_upsample"]),
                        head_conv=int(s["head_conv"]),
                        lr=lr,
                        epochs=int(res["epochs"]),
                        mix=mix,
                        seed=0,
                    ))
        return arms

    if "pick_best" in res:
        if not ok:
            log(f"[RESOLVE] {stage['name']}：无可用结果 ⇒ 跳过该阶段")
            return []
        best = sorted(ok, key=lambda s: (-(s["t3_f1"] or -1), -(s["t1_f1"] or -1)))[0]
        log(
            f"[RESOLVE] {stage['name']}：最佳臂 = {best['arm_id']} "
            f"(T3={best['t3_f1']:.6f} T1={best['t1_f1']:.6f})"
        )
        epochs_list = res.get("epochs")
        seeds = res.get("seed", [0])
        arms = []
        for ep in (epochs_list if isinstance(epochs_list, list) else [best["epochs"]]):
            for sd in seeds:
                arms.append(B.make_arm(
                    stage=stage["name"],
                    head_level=best["head_level"],
                    late=bool(best["late_upsample"]),
                    head_conv=int(best["head_conv"]),
                    lr=float(best["lr_scale"]),
                    epochs=int(ep),
                    mix=best["mix_spec"],
                    seed=int(sd),
                ))
        return arms

    return []


# ==============================================================================
# 单臂执行
# ==============================================================================
def build_train_cmd(arm: dict, gpu: str, arm_dir: Path, args: argparse.Namespace,
                    native_flags: set[str]) -> list[str]:
    cmd = [
        sys.executable, "-u", args.shim,
        "--longquan-data", NOGMC_LQ_DATA,
        "--uav-data", NOGMC_UAV_DATA,
        "--weights", arm["init_from"],
        "--out-dir", str(arm_dir),
        "--epochs", str(arm["epochs"]),
        "--batch", str(args.batch),
        "--workers", str(args.workers),
        "--imgsz", str(args.imgsz),
        "--device", gpu,
        "--seed", str(arm["seed"]),
        "--no-gate",
    ]
    # ---- 结构旗标（全部为主控约定的契约）----
    cmd += ["--act", "relu"]
    cmd += ["--no-p2"]
    cmd += ["--head-level", arm["head_level"]]
    # ---- 可选结构旗标：训练器不支持时不传（并在 arm_summary 里留证，避免
    #      "静默跑了错的结构"这种最坏情况）----
    if "--head-conv" in native_flags:
        cmd += ["--head-conv", str(arm["head_conv"])]
    if arm["late_upsample"] and "--late-upsample" in native_flags:
        cmd += ["--late-upsample"]
    # ---- 超参：训练器原生支持就透传，否则由 shim 打补丁 ----
    if "--lr-scale" in native_flags:
        cmd += ["--lr-scale", str(arm["lr_scale"])]
    if "--mix-spec" in native_flags:
        cmd += ["--mix-spec", arm["mix_spec"]]
    return cmd


def build_gate_cmd(weights: Path, gate_dir: Path, gpu: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "-u", args.gate_script,
        "--weights", str(weights),
        "--out-dir", str(gate_dir),
        "--gpus", "1",
        "--device", gpu,
        "--infer-gpu", gpu,
        "--cache-gpus", gpu,
        "--strict-gate",
        # ---- no-GMC 语义（特征集 + 基线数字 + 基线缓存）----
        "--uav-data", NOGMC_UAV_DATA,
        "--ft-root", "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc",
        "--baseline-cache", NOGMC_BASELINE_CACHE,
    ]
    for k, v in NOGMC_GATE_BASE.items():
        cmd += [f"--{k}", str(v)]
    # ---- 降 batch（4 卡并行时每卡只剩 ~11 GB）----
    cmd += ["--infer-batch", str(args.gate_infer_batch)]
    cmd += ["--cache-batch-size", str(args.gate_cache_batch)]
    return cmd


def extract_metrics(gate_report: dict) -> dict:
    tr = gate_report.get("tracks", {}) or {}
    t1 = tr.get("track1_uav", {}) or {}
    t1p = t1.get("finetune_peak", {}) or {}
    t2 = tr.get("track2_golden", {}) or {}
    t3 = tr.get("track3_blind", {}) or {}
    return {
        "t1_f1": t1p.get("f1"),
        "t1_tp": t1p.get("tp"),
        "t1_fp": t1p.get("fp"),
        "t2_f1": t2.get("f1"),
        "t2_tp": t2.get("tp"),
        "t2_fp": t2.get("fp"),
        "t2_fn": t2.get("fn"),
        "t3_f1": t3.get("f1"),
        "t3_tp": t3.get("tp"),
        "t3_fp": t3.get("fp"),
        "t3_fn": t3.get("fn"),
        "gate_verdict": gate_report.get("verdict"),
    }


def run_arm(arm: dict, gpu: str, args: argparse.Namespace, native_flags: set[str],
            hard_cut: float) -> dict:
    # ---- 每条臂都**重新**探测一次训练器旗标 ----
    # 训练器由主控 agent 并发改造，dispatcher 可能比训练器先启动；
    # 启动时那份快照会过期（新增的 --head-conv / --lr-scale 等会漏掉）。
    # ⇒ 派单前重新读源码，避免"静默跑了错的结构"。
    trainer_path = Path(args.trainer)
    if not trainer_path.is_absolute():
        trainer_path = PROJECT_ROOT / trainer_path
    fresh = trainer_flags(trainer_path)
    if fresh != native_flags:
        log(f"[ARM] 训练器旗标已变化：{sorted(fresh)}（原 {sorted(native_flags)}）")
        native_flags = fresh

    arm_dir = Path(args.out_root) / "arms" / arm["arm_id"]
    arm_dir.mkdir(parents=True, exist_ok=True)
    sum_path = arm_dir / "arm_summary.json"

    base = {
        "arm_id": arm["arm_id"], "stage": arm["stage"], "gpu": gpu,
        "head_level": arm["head_level"], "late_upsample": arm["late_upsample"],
        "head_conv": arm["head_conv"], "act": arm["act"], "no_p2": arm["no_p2"],
        "lr_scale": arm["lr_scale"], "epochs": arm["epochs"], "mix_spec": arm["mix_spec"],
        "seed": arm["seed"], "init_from": arm["init_from"],
        "anchor_t1_f1": B.ANCHOR["t1_f1"], "anchor_t2_f1": B.ANCHOR["t2_f1"],
        "anchor_t3_f1": B.ANCHOR["t3_f1"],
    }

    if sum_path.is_file() and not args.force:
        log(f"[ARM] {arm['arm_id']} 已完成 ⇒ 跳过（断点续跑）")
        return json.loads(sum_path.read_text(encoding="utf-8"))

    if time.time() >= hard_cut:
        log(f"[ARM] {arm['arm_id']} 已过 hard_cut ⇒ 跳过")
        return {**base, "status": "skipped_no_time"}

    cmd = build_train_cmd(arm, gpu, arm_dir, args, native_flags)
    (arm_dir / "cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    # ---- 留证：计划结构 vs 实际传下去的旗标（防止"静默跑了错的结构"）----
    dropped = []
    if "--head-conv" not in native_flags:
        dropped.append("--head-conv")
    if arm["late_upsample"] and "--late-upsample" not in native_flags:
        dropped.append("--late-upsample")
    if dropped:
        log(f"[ARM] ⚠️ {arm['arm_id']}：训练器不支持 {dropped} ⇒ 本臂**未完全复现**计划结构！")
    log(f"[ARM] 启动 {arm['arm_id']} on GPU{gpu} | $ {' '.join(cmd)}")

    env = dict(os.environ)
    env["STRUCT_LR_SCALE"] = str(arm["lr_scale"])
    env["STRUCT_MIX_SPEC"] = arm["mix_spec"]
    env["STRUCT_TRAINER"] = args.trainer

    t0 = time.time()
    with open(arm_dir / "train.log", "a", encoding="utf-8") as fh:
        fh.write(f"\n\n===== {arm['arm_id']} @ {time.strftime('%F %T')} =====\n")
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(PROJECT_ROOT), env=env)
    train_s = time.time() - t0
    log(f"[ARM] {arm['arm_id']} 训练结束 rc={rc}，耗时 {train_s/60:.1f} min")

    # ---- 取最后一个 epoch 的权重 ----
    wdir = arm_dir / "weights"
    cks = []
    if wdir.is_dir():
        for p in wdir.glob("epoch*.pt"):
            m = re.search(r"epoch(\d+)", p.name)
            if m:
                cks.append((int(m.group(1)), p))
    cks.sort()
    if not cks:
        rec = {**base, "status": "failed_no_weights", "train_rc": rc,
               "train_min": round(train_s / 60, 1)}
        sum_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        return rec
    last_ep, last_w = cks[-1]
    log(f"[ARM] {arm['arm_id']} 门禁权重 = {last_w.name}（共 {len(cks)} 个 epoch 落盘）")

    # ---- 三轨门禁 ----
    gate_dir = arm_dir / "gate"
    gcmd = build_gate_cmd(last_w, gate_dir, gpu, args)
    (arm_dir / "gate_cmd.txt").write_text(" ".join(gcmd) + "\n", encoding="utf-8")
    with open(arm_dir / "gate.log", "a", encoding="utf-8") as fh:
        fh.write(f"\n\n===== gate {arm['arm_id']} @ {time.strftime('%F %T')} =====\n")
        fh.write("$ " + " ".join(gcmd) + "\n\n")
        fh.flush()
        grc = subprocess.call(gcmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))

    rep_path = gate_dir / "gate_report.json"
    rec = {**base, "status": "ok", "train_rc": rc, "gate_rc": grc,
           "train_min": round(train_s / 60, 1), "weights": str(last_w),
           "last_epoch": last_ep, "n_epochs_ckpt": len(cks),
           "flag_fallback": dropped}
    if rep_path.is_file():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        rec.update(extract_metrics(rep))
    else:
        rec["status"] = "failed_no_gate_report"
    for key, anchor in (("t1_f1", "t1_f1"), ("t2_f1", "t2_f1"), ("t3_f1", "t3_f1")):
        v = rec.get(key)
        if isinstance(v, (int, float)):
            rec[f"d_{key}_pp"] = round((v - B.ANCHOR[anchor]) * 100.0, 4)
    sum_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        f"[ARM] {arm['arm_id']} 完成 | T1={rec.get('t1_f1')} T2={rec.get('t2_f1')} "
        f"T3={rec.get('t3_f1')} | ΔT3={rec.get('d_t3_f1_pp')} pp | ΔT1={rec.get('d_t1_f1_pp')} pp"
    )
    return rec


# ==============================================================================
# 阶段调度
# ==============================================================================
def run_stage(arms: list[dict], args: argparse.Namespace, native_flags: set[str],
              hard_cut: float, stop_at: float) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if not arms:
        return results
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    arm_q: queue.Queue = queue.Queue()
    for a in arms:
        arm_q.put(a)
    gpu_q: queue.Queue = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    lock = threading.Lock()

    def worker():
        while True:
            try:
                arm = arm_q.get_nowait()
            except queue.Empty:
                return
            gpu = gpu_q.get()
            try:
                if stop_at and time.time() >= stop_at:
                    rec = {**arm, "status": "skipped_stop_at"}
                else:
                    rec = run_arm(arm, gpu, args, native_flags, hard_cut)
            except Exception as exc:  # noqa: BLE001
                rec = {"arm_id": arm["arm_id"], "stage": arm["stage"], "status": "exception",
                       "error": f"{type(exc).__name__}: {exc}"}
                log(f"[ARM] {arm['arm_id']} 异常: {exc!r}")
            finally:
                gpu_q.put(gpu)
            with lock:
                results[arm["arm_id"]] = rec
            arm_q.task_done()

    threads = [threading.Thread(target=worker, daemon=True, name=f"slot{i}")
               for i in range(len(gpus))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ==============================================================================
# 汇总
# ==============================================================================
def write_summaries(out_root: Path, summaries: dict[str, dict], plan: dict) -> None:
    rows = []
    for rec in sorted(summaries.values(), key=lambda r: r.get("arm_id", "")):
        rows.append(rec)

    csv_path = out_root / "arms.csv"
    cols = [
        "arm_id", "stage", "status", "head_level", "late_upsample", "head_conv", "act",
        "no_p2", "lr_scale", "epochs", "mix_spec", "seed", "gpu",
        "t1_f1", "d_t1_f1_pp", "t2_f1", "d_t2_f1_pp", "t3_f1", "d_t3_f1_pp",
        "t2_fp", "t3_fp", "gate_verdict", "train_min", "last_epoch", "weights",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"[SUMMARY] {csv_path}")

    crit = plan.get("criteria", {})
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("t3_f1") is not None]
    ok_rows.sort(key=lambda r: -(r.get("t3_f1") or -1))
    md = out_root / "SUMMARY.md"
    lines = [
        "# no-GMC 结构搜索 —— 结果汇总",
        "",
        f"* 生成时间：{time.strftime('%F %T')}",
        f"* 初始化：`{plan.get('init_from')}`",
        f"* 目标结构：删 neck 的 P2 融合级（`--no-p2`）+ `--head-level {{p3,p2}}` + all ReLU",
        f"* 判据锚点（cs_t0043 on no-GMC）：T1={crit.get('anchor',{}).get('t1_f1')} "
        f"T2={crit.get('anchor',{}).get('t2_f1')} T3={crit.get('anchor',{}).get('t3_f1')}",
        f"* 本线判据：**T3 相对 cs_t0043 的下降 ≤ {crit.get('t3_drop_max_pp')} pp**"
        f"；T1 下降记录在案；NPU 板卡实测对照基线 **{crit.get('npu_baseline_ms')} ms**",
        f"* 汇率红线：新旋钮须优于 **{crit.get('frontier_exchange_rate')} pp T3 / 1 pp T1**",
        "",
        "## 逐臂结果",
        "",
        "| arm_id | 结构 | lr_scale | ep | mix | T1 | ΔT1 pp | T2 | T3 | ΔT3 pp | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        struct = f"{r.get('head_level')}/lu{int(bool(r.get('late_upsample')))}/h{r.get('head_conv')}"
        lines.append(
            f"| {r.get('arm_id')} | {struct} | {r.get('lr_scale')} | {r.get('epochs')} | "
            f"{r.get('mix_spec')} | {r.get('t1_f1')} | {r.get('d_t1_f1_pp')} | "
            f"{r.get('t2_f1')} | {r.get('t3_f1')} | {r.get('d_t3_f1_pp')} | "
            f"{r.get('gate_verdict')} |"
        )
    lines += ["", "## T3 排序（Top）", ""]
    for r in ok_rows[:10]:
        lines.append(
            f"* {r.get('arm_id')}：T3={r.get('t3_f1'):.6f}（Δ{r.get('d_t3_f1_pp')} pp）"
            f" / T1={r.get('t1_f1')}（Δ{r.get('d_t1_f1_pp')} pp）"
            f" / T2={r.get('t2_f1')} / gate={r.get('gate_verdict')}"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[SUMMARY] {md}")


# ==============================================================================
def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "arms").mkdir(parents=True, exist_ok=True)

    plan_path = Path(args.plan)
    if not plan_path.is_file():
        log(f"[FATAL] 计划文件不存在：{plan_path}")
        return 2
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    log(f"[PLAN] {plan_path} | {plan.get('description')}")
    log(f"[PLAN] 判据：ΔT3 ≤ {plan['criteria']['t3_drop_max_pp']} pp（vs cs_t0043）"
        f" | NPU 基线 {plan['criteria']['npu_baseline_ms']} ms")

    hard_cut = time.mktime(time.strptime(args.hard_cut, "%Y-%m-%d %H:%M"))
    stop_at = time.mktime(time.strptime(args.stop_at, "%Y-%m-%d %H:%M")) if args.stop_at else 0.0

    trainer = Path(args.trainer)
    if not trainer.is_absolute():
        trainer = PROJECT_ROOT / trainer

    # ---- 代码就绪闸门（等待并重试，不退出）----
    if not wait_for_trainer(trainer, hard_cut, args.poll_seconds):
        return 3
    native_flags = trainer_flags(trainer)
    log(f"[READY-GATE] 训练器旗标：{sorted(native_flags)}")

    summaries: dict[str, dict] = {}
    for stage in plan["stages"]:
        arms = resolve_stage(stage, summaries)
        log("=" * 100)
        log(f"[STAGE] {stage['name']}：{len(arms)} 臂 | {stage.get('desc','')}")
        for a in arms:
            log(f"         - {a['arm_id']}  est={a['est_min']} min")
        if args.dry_run:
            log("[DRY-RUN] 不训练")
            continue
        res = run_stage(arms, args, native_flags, hard_cut, stop_at)
        summaries.update(res)
        write_summaries(out_root, summaries, plan)

    write_summaries(out_root, summaries, plan)
    log("[DONE] 全部阶段结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
