#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optuna 超参搜索：**dataloader 混合比例** 为主变量（口径 1 = Batch 内采样概率）。

背景与目标
----------
路线 B 首轮试跑（5 epoch）已把 Ep3 固化为 RC1，但其源域大盘付出 **−0.38pp** 的代价
（Track1 F1 0.905736 → 0.901918）。本搜索要回答的问题只有一个：
**把 Anti-UAV 的采样比重从自然的 1:3.09 调到严格的 1:4（甚至更偏），
能不能把源域那笔损失救回来，同时不牺牲难例轨 001324 的收益？**

"真正的 1:4" 的定义（唯一口径 = 口径 1：Batch 内采样概率）
------------------------------------------------------------
| 口径 | 定义 | 1:4 是否成立 |
|:--|:--|:--|
| 口径1 | **Batch 内采样概率** P(LQ)=0.20 / P(UAV)=0.80 | ✅ **本搜索采用** |
| 口径2 | 数据集总帧数（两边各遍历一次） | ❌ 物理不成立：13,918×4 = 55,672 > 43,008，实际极限 1:3.09 |
| 口径3 | GT 目标框数 | 1:3.36（12,120 : 40,761） |
| 口径4 | 局部切片张数 | 路线 A 之后才成立 |

实现方式：**WeightedRandomSampler**（零数据复制、零磁盘开销），
通过**只在官方 `build_dataloader` 调用期间替换索引采样器**来实现，
不重写 DataLoader（§十四 教训：自写 DataLoader 曾致样本腰斩 + 凭空多 576 处假警）。

判定标准：三级漏斗（见 `manu/training/mix_ratio_judgement.py`）
-------------------------------------------------------------
* **一级** 安全红线（相对冻结基线）：T2 F1 ≥ 0.985 且 ΔFP_T2 ≤ +10 且 ΔF1_T1 ≥ −0.005
  → 触碰任一条 ⇒ **剪枝 Pruned**（不参与 TPE 采样）
* **三级** 效用分作为 Optuna 标量 objective：
  `U = ΔF1_T3 + 0.5·ΔF1_T2 − 2.5·max(0,−ΔF1_T1) − 0.0005·max(0,ΔFP_T2)`
  （**RC1/Ep3 的效用分 = 0.027579**，是"更好"的及格线）
* **二级** 帕累托占优用于最终选型（`select_best`），与 objective 同尺度，不参与搜索排序

工程特性
--------
* **4 卡并行**，每个 trial **单 GPU**、**3 epoch**；门禁在同卡串行跑
* **可随时停止 / 随时继续**：Optuna SQLite study 持久化 + 断点续训状态 + 陈旧 RUNNING trial 自愈
* **后台常驻**：`setsid nohup` 启动，与 SSH 会话完全解耦（宿主机断电不影响）
* **状态可查**：`manu/training/status_mix_ratio_search.py` 随时读库/读日志/读 GPU
* **有截止时间**：`--max-hours` 到点自动收尾（保证第二天早上有结果）

用法（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`，
根目录 `/tmp/pycharm_project_10ae9e2e`）：

```bash
# 启动（后台常驻，宿主机断电无影响）
bash manu/training/launch_mix_ratio_search.sh

# 看状态（任何时候、任何机器都能查）
python manu/training/status_mix_ratio_search.py

# 停止（可恢复）
bash manu/training/stop_mix_ratio_search.sh
```
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.utils.data import WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from torch.cuda.amp import GradScaler, autocast  # noqa: E402
from torch.utils.data import ConcatDataset

from manu.data.preannotate_hm_bbox import letterbox_gray, load_hm_model, resolve_weights  # noqa: E402
from manu.models.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets  # noqa: E402
from manu.training.mix_ratio_judgement import (  # noqa: E402
    FROZEN_BASELINE,
    RC1_EPOCH3,
    evaluate_model_quality,
    hard_gate,
    metrics_from_gate_report,
    utility,
)

# ==============================================================================
# 常量
# ==============================================================================
LQ_EXPECT, UAV_EXPECT = 13918, 43008  # 满额护栏（§十四 腰斩事故）
TOTAL_EXPECT = LQ_EXPECT + UAV_EXPECT

# 路线 B 的分层参数组（3e-6 : 1.5e-5 : 3e-5），搜索只缩放整体倍率
GROUP_LRS = {"G2_p4p5": 3e-6, "G3_neck": 1.5e-5, "G4_head": 3e-5}
FROZEN_BACKBONE = ("b0", "b1", "b2", "b3", "b4")

# 混合比例候选（mix_spec）：natural = 路线 B 对照（自然 1:3.09），wN = 加权采样 1:N
MIX_SPECS = [
    "natural",
    "w1.0", "w1.5", "w2.0", "w2.5", "w3.0", "w3.5", "w4.0", "w5.0", "w6.0", "w8.0",
]

# 路线 B 已验证的超参（作为种子与缩放中心）
ROUTE_B_DEFAULTS = {
    "lr_scale": 1.0,
    "warmup_epochs": 0.5,
    "weight_decay": 1e-4,
    "focal_beta": 2.7,
    "offset_weight": 0.55,
    "mosaic": 0.05,
    "scale": 0.20,
    "translate": 0.10,
    "clip_max_norm": 5.0,
    "freeze_bn": False,
    "lr_min": 1e-6,
}

# 门禁阈值（与用户一级红线一致）
GATE_TRACK2_F1_MIN = 0.985
GATE_TRACK2_FP_MAX_INCREMENT = 10
GATE_TRACK1_MIN_DELTA = -0.005
# auto_epoch_gate_check.py 的 --max-track2-f1/--max-track2-fp 语义是**绝对阈值**
# （默认 36 = 基线 FP 26 + 允许增量 10），而我们内部一律用"相对冻结基线的增量"。
# 直接把增量 10 传给它会被当成绝对上限 ⇒ 只要 FP > 10 就误判一票否决。
# 这里换算成绝对上限，保证脚本自带 verdict 与我们的 hard_gate() 判定一致。
GATE_TRACK2_FP_ABS_MAX = FROZEN_BASELINE["track2_fp"] + GATE_TRACK2_FP_MAX_INCREMENT  # = 36
# 同理，--min-track1-f1 是**绝对** F1 下限。我们的红线是"相对冻结基线 ΔF1 ≥ -0.005"，
# 换算成绝对值 = 0.905736 - 0.005 = 0.900736。这样门禁脚本自带的 verdict 与我们的
# hard_gate() 完全等价 ⇒ gate_report.json 里的 pass/veto 字段可直接审计，无需另读代码。
GATE_TRACK1_F1_ABS_MIN = FROZEN_BASELINE["track1_f1"] + GATE_TRACK1_MIN_DELTA  # = 0.900736
UTILITY_ALPHA, UTILITY_BETA, UTILITY_GAMMA = 0.5, 2.5, 0.0005


# ==============================================================================
# 1. 加权采样器：只在官方 build_dataloader 期间替换索引采样器
# ==============================================================================
@contextlib.contextmanager
def weighted_sampler_patch(weights: np.ndarray, num_samples: int, seed: int):
    """在官方 build_dataloader 调用期间，把索引采样器换成 WeightedRandomSampler。

    为什么要这么做（而不是自己写 DataLoader）
    ------------------------------------------
    torch 1.13 的 `DataLoader.__init__` 在 `sampler is None and shuffle=True` 时
    使用**模块级** `RandomSampler(dataset, generator=generator)`。
    我们只在调用窗口内替换这个名字，于是：
      · 数据集仍是官方 `build_yolo_dataset`（图片/标签扫描、缓存、mosaic 全官方）
      · BatchSampler / _RepeatSampler / collate_fn / worker_init_fn / generator / prefetch 全官方
      · **唯一**变化是"抽哪些下标"由 WeightedRandomSampler 决定
    ⇒ 索引语义（batch_idx）与官方完全一致（collate_fn 里 batch_idx += 批内位置），
      不会出现 §十四 的"样本腰斩 + 凭空多假警"。
    """
    import torch.utils.data.dataloader as dl

    original = dl.RandomSampler

    def _factory(dataset, generator=None, **kwargs):  # noqa: ANN001
        g = generator if generator is not None else torch.Generator()
        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,
            generator=g,
        )

    dl.RandomSampler = _factory
    try:
        yield
    finally:
        dl.RandomSampler = original


# ==============================================================================
# 2. 参数空间
# ==============================================================================
def suggest_params(trial: optuna.Trial) -> dict:
    return {
        # ★ 关键参数：dataloader 混合比例（口径 1 = Batch 内采样概率）
        "mix_spec": trial.suggest_categorical("mix_spec", MIX_SPECS),
        # 其余超参
        "lr_scale": trial.suggest_float("lr_scale", 0.3, 3.0, log=True),
        "warmup_epochs": trial.suggest_float("warmup_epochs", 0.2, 1.0),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
        "focal_beta": trial.suggest_float("focal_beta", 2.2, 2.9, step=0.05),
        "offset_weight": trial.suggest_float("offset_weight", 0.40, 0.70, step=0.05),
        "mosaic": trial.suggest_float("mosaic", 0.0, 0.15, step=0.05),
        "scale": trial.suggest_float("scale", 0.10, 0.30, step=0.05),
        "translate": trial.suggest_float("translate", 0.05, 0.15, step=0.05),
        "clip_max_norm": trial.suggest_float("clip_max_norm", 2.0, 8.0),
        "freeze_bn": trial.suggest_categorical("freeze_bn", [True, False]),
    }


def parse_mix_spec(spec: str) -> tuple[str, float]:
    """mix_spec → (mode, uav_per_lq)。natural 模式的实际帧比是 43008/13918 ≈ 3.09。"""
    if spec == "natural":
        return "natural", UAV_EXPECT / LQ_EXPECT
    return "weighted", float(spec[1:])


def read_runtime_config(out_root: Path) -> dict:
    """读取 ``<out_root>/runtime_config.json``（**可选**，解析失败一律回落到空字典）。

    用途：**不重启调度器**就能改后续 trial 的训练协议。
    典型场景（2026-10-21）：把后续 trial 从 3 epoch 切到 5 epoch，
    以便复现 RC1 的完整 5-epoch 曲线（Ep3 与 Ep5 都在同一次运行里）。

    键：
      · ``train_epochs`` (int)  后续 trial 训练几个 epoch
      · ``gate_epoch``   (int)  门禁量第几个 epoch（0/缺省 = 最后一个）
      · ``workers``      (int)  dataloader worker 数
    """
    path = Path(out_root) / "runtime_config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {k: data[k] for k in ("train_epochs", "gate_epoch", "workers") if k in data}
    except Exception as exc:
        print(f"[RUNTIME] runtime_config.json 解析失败（忽略，用启动参数）：{exc}", flush=True)
        return {}


# ==============================================================================
# 3. Worker：训练 + 门禁
# ==============================================================================
def run_worker() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--trial-number", type=int, required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--longquan-data", required=True)
    ap.add_argument("--uav-data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gate-script", default="manu/training/auto_epoch_gate_check.py")
    ap.add_argument("--baseline-cache", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-epochs", default="last", help="last=只门禁最后一个 epoch")
    ap.add_argument("--gate-epoch", type=int, default=0,
                    help="门禁量第几个 epoch 的权重（0=最后一个）。"
                         "5-epoch 训练时可用 3 去量『RC1 同款 Ep3』，"
                         "从而在同一次 5-epoch 运行里既拿到 Ep5 权重又复现 RC1/Ep3")
    ap.add_argument("--max-iters", type=int, default=0, help=">0 时每 epoch 只跑这么多 iteration（冒烟用）")
    ap.add_argument("--skip-gate", action="store_true", help="跳过门禁（冒烟用）")
    # 搜索参数
    ap.add_argument("--mix-spec", required=True)
    ap.add_argument("--lr-scale", type=float, required=True)
    ap.add_argument("--warmup-epochs", type=float, required=True)
    ap.add_argument("--weight-decay", type=float, required=True)
    ap.add_argument("--focal-beta", type=float, required=True)
    ap.add_argument("--offset-weight", type=float, required=True)
    ap.add_argument("--mosaic", type=float, required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--translate", type=float, required=True)
    ap.add_argument("--clip-max-norm", type=float, required=True)
    ap.add_argument("--freeze-bn", type=int, required=True)
    ap.add_argument(
        "--sched-epochs", type=int, default=5,
        help="cosine 退火**周期**按几个 epoch 算（默认 5，与 RC1 同口径）。"
             "0 表示跟随 --epochs。训练仍只跑 --epochs 个 epoch，"
             "但 lr 的退火曲线按 --sched-epochs 的全长走 ⇒ trial 的末 epoch 权重"
             "与 RC1/Ep3（路线 B 5 epoch 实验的第 3 个 epoch）**同 lr 轨迹**，可直接比较。",
    )
    args = ap.parse_args()

    trial_name = f"trial_{args.trial_number:04d}"
    save_dir = Path(args.output_root) / "trials" / trial_name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "logs").mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mix_mode, uav_per_lq = parse_mix_spec(args.mix_spec)
    print("=" * 100, flush=True)
    print(f"[WORKER] {trial_name} | GPU {args.gpu_id} | PID {os.getpid()}", flush=True)
    print(f"[WORKER] mix_spec={args.mix_spec} (mode={mix_mode}, uav_per_lq={uav_per_lq:.3f})", flush=True)
    print(
        f"[WORKER] lr_scale={args.lr_scale:.3f} warmup={args.warmup_epochs} wd={args.weight_decay} "
        f"beta={args.focal_beta} offset={args.offset_weight} clip={args.clip_max_norm} "
        f"freeze_bn={bool(args.freeze_bn)}",
        flush=True,
    )
    print("=" * 100, flush=True)

    # ---- 模型：与底模完全同构（去 P0），并带回源 ckpt 的 p0 张量 ----
    model, model_stride = load_hm_model(
        resolve_weights(args.weights), device, stride_fallback=2, use_p0_highway=False
    )
    model.to(device)
    feat_h = feat_w = args.imgsz // model_stride
    src_ckpt = torch.load(resolve_weights(args.weights), map_location="cpu")
    src_state = src_ckpt["model"] if "model" in src_ckpt else src_ckpt.get("state_dict", src_ckpt)
    if hasattr(src_state, "state_dict"):
        src_state = src_state.state_dict()
    p0_tensors = {k: v for k, v in src_state.items() if k.startswith("p0_highway")}
    p0_kwargs = src_ckpt.get("p0_kwargs", {})

    # ---- 数据：官方 build_yolo_dataset × 2 → ConcatDataset ----
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.hsv_h = cfg.hsv_s = cfg.hsv_v = 0.0
    cfg.degrees = cfg.shear = cfg.perspective = 0.0
    cfg.translate = args.translate
    cfg.scale = args.scale
    cfg.mosaic = args.mosaic
    cfg.fliplr = 0.5
    cfg.flipud = cfg.mixup = cfg.copy_paste = 0.0

    datasets = []
    for tag, data_yaml in (("longquan", args.longquan_data), ("uav", args.uav_data)):
        data_dict = check_det_dataset(data_yaml)
        cfg.data = data_yaml
        ds = build_yolo_dataset(
            cfg, data_dict["train"], batch=args.batch, data=data_dict, mode="train", stride=32
        )
        print(f"[DATA] {tag:<9} 官方 build_yolo_dataset 样本数 = {len(ds):,}", flush=True)
        datasets.append(ds)
    n_lq, n_uav = len(datasets[0]), len(datasets[1])
    if n_lq != LQ_EXPECT or n_uav != UAV_EXPECT:
        raise RuntimeError(
            f"[DATA-GATE] 样本数异常: 龙泉山 {n_lq} (期望 {LQ_EXPECT}) / "
            f"Anti-UAV {n_uav} (期望 {UAV_EXPECT}) —— 立即中止"
        )
    print(f"[DATA-GATE] PASS 满额 {n_lq:,} + {n_uav:,} = {n_lq + n_uav:,}", flush=True)

    mixed = ConcatDataset(datasets)
    mixed.collate_fn = YOLODataset.collate_fn  # 官方静态方法（collate_fn 内 batch_idx += 批内位置）

    if mix_mode == "weighted":
        # 口径 1：Batch 内采样概率 P(LQ) = 1/(1+r)，P(UAV) = r/(1+r)
        p_lq = 1.0 / (1.0 + uav_per_lq)
        p_uav = 1.0 - p_lq
        weights = np.concatenate(
            [
                np.full(n_lq, p_lq / n_lq, dtype=np.float64),
                np.full(n_uav, p_uav / n_uav, dtype=np.float64),
            ]
        )
        num_samples = n_lq + n_uav  # 保持每 epoch 总采样量不变 ⇒ 各 trial 算力与耗时可比
        with weighted_sampler_patch(weights, num_samples, seed=args.seed + args.trial_number):
            loader = build_dataloader(mixed, batch=args.batch, workers=args.workers, shuffle=True)
        print(
            f"[MIX] 口径1 加权采样：P(LQ)={p_lq:.4f} P(UAV)={p_uav:.4f} | "
            f"每 epoch 抽样 {num_samples:,} 次（LQ 期望 {num_samples*p_lq:,.0f} / UAV {num_samples*p_uav:,.0f}）",
            flush=True,
        )
    else:
        loader = build_dataloader(mixed, batch=args.batch, workers=args.workers, shuffle=True)
        print(
            f"[MIX] 对照模式：自然遍历（ConcatDataset 全量，帧比 1:{n_uav/n_lq:.3f}）",
            flush=True,
        )

    iters_per_epoch = len(loader)
    total_steps = iters_per_epoch * args.epochs
    print(f"[DATA] iters/epoch = {iters_per_epoch} | 总优化步数 = {total_steps}", flush=True)

    # ---- 参数组：G1 严格冻结，G2/G3/G4 按路线 B 比例 × lr_scale ----
    buckets: dict[str, list[torch.nn.Parameter]] = {k: [] for k in GROUP_LRS}
    frozen: list[torch.nn.Parameter] = []
    unmatched: list[str] = []
    for name, param in model.named_parameters():
        if any(name == g or name.startswith(g + ".") for g in FROZEN_BACKBONE):
            param.requires_grad_(False)
            frozen.append(param)
            continue
        if name.startswith("head"):
            key = "G4_head"
        elif name.startswith("b5") or name.startswith("b6") or name.startswith("b7") or name.startswith("b8") \
                or name.startswith("b9") or name.startswith("b10"):
            key = "G2_p4p5"
        else:
            key = "G3_neck"
        param.requires_grad_(True)
        buckets[key].append(param)
    if unmatched:
        raise RuntimeError(f"[PARAM-GATE] 有参数未归组: {unmatched[:10]}")
    groups = []
    for key, base_lr in GROUP_LRS.items():
        params = buckets[key]
        if not params:
            continue
        lr = base_lr * args.lr_scale
        n = sum(p.numel() for p in params)
        groups.append({"params": params, "lr": lr, "base_lr": lr, "name": key, "numel": n})
        print(f"[PARAM] {key:<9} lr={lr:.3e} (基准 {base_lr:.1e} × {args.lr_scale:.3f})  参数 {n:,}", flush=True)
    print(f"[PARAM] G1_frozen 参数 {sum(p.numel() for p in frozen):,}（严格冻结）", flush=True)

    optimizer = torch.optim.AdamW(groups, lr=groups[0]["lr"], weight_decay=args.weight_decay)
    for pg, g in zip(optimizer.param_groups, groups):
        pg["lr"] = g["base_lr"]
        pg["initial_lr"] = g["base_lr"]

    # 调度：warmup（按单 epoch 步数算）→ cosine → 绝对下限
    #
    # ⚠️ **cosine 周期与训练总步数解耦**：
    #    RC1/Ep3 是"路线 B 跑 5 epoch"里的第 3 个 epoch 权重，它经历的是
    #    **5 epoch 全长的 cosine 曲线**（总步 8895），在 Ep3 结束时 lr 仍处高位
    #    （G2 ≈ 1.24e-6 = 峰值的 41%）。
    #    若搜索按 3 epoch 自己的 cosine（总步 5337），Ep3 结束时 lr 已退到绝对下限
    #    （G2 = 1.00e-6 = 峰值的 3.3%）⇒ **每个 trial 的末 epoch 权重都系统性地
    #    "比 RC1 学得少一点"**，与 mix_spec 的真实效应混在一起，无法归因。
    #    ⇒ 统一用 --sched-epochs（默认 5）作为 cosine 周期，使 natural 档成为
    #    **RC1 的精确复现**（一个强 sanity check），其余档位与 RC1 直接可比。
    sched_epochs = args.sched_epochs if args.sched_epochs > 0 else args.epochs
    sched_total_steps = sched_epochs * iters_per_epoch
    warmup_steps = max(1, int(round(args.warmup_epochs * iters_per_epoch)))
    warmup_steps = min(warmup_steps, max(1, sched_total_steps - 1))

    def make_lambda(base_lr: float):
        floor_ratio = min(1.0, ROUTE_B_DEFAULTS["lr_min"] / base_lr) if base_lr > 0 else 0.0

        def fn(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, sched_total_steps - warmup_steps)))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cos

        return fn

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[make_lambda(g["base_lr"]) for g in optimizer.param_groups]
    )
    print(f"[SCHED] 训练 {args.epochs} ep（{total_steps} 步）| warmup {warmup_steps} 步"
          f"（{args.warmup_epochs} ep）| cosine 周期 {sched_epochs} ep（{sched_total_steps} 步）"
          f" → {ROUTE_B_DEFAULTS['lr_min']:.0e}"
          f"{'  ← 与 RC1 同口径' if sched_epochs != args.epochs else ''}", flush=True)

    criterion = HeatmapLoss(
        hm_weight=1.0, offset_weight=args.offset_weight, focal_alpha=2.0, focal_beta=args.focal_beta
    )
    scaler = GradScaler(enabled=True)

    csv_path = save_dir / "logs" / "train_log.csv"
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_file.write(
        "epoch,iter,loss_total,loss_hm,loss_offset,"
        + ",".join(f"lr_{g['name']}" for g in groups)
        + ",batch_lq,batch_uav\n"
    )
    csv_file.flush()

    def set_frozen_eval() -> None:
        model.train()
        for name, module in model.named_modules():
            if any(name == g or name.startswith(g + ".") for g in FROZEN_BACKBONE):
                module.eval()
            if args.freeze_bn and name.endswith("bn"):
                module.eval()  # 冻结 BN running stats（用户点名的下一轮方向）

    global_step = 0
    vram_peak = 0.0
    t_start = time.time()
    last_weights = None
    for epoch in range(1, args.epochs + 1):
        set_frozen_eval()
        ep_loss = ep_hm = ep_off = 0.0
        nb = 0
        lq_seen = uav_seen = 0
        t_ep = time.time()
        for it, batch in enumerate(loader, start=1):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"].to(device, non_blocking=True)
            b_idx = batch["batch_idx"].to(device, non_blocking=True)
            bs = imgs.shape[0]
            targets = generate_heatmaps_and_targets(
                batch_bboxes=bboxes, batch_idx=b_idx, batch_size=bs,
                feat_shape=(feat_h, feat_w), stride=model_stride, min_radius=1, device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                preds = model(imgs)
                loss, loss_items = criterion(preds, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], max_norm=args.clip_max_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            ep_loss += loss_items["loss_total"]
            ep_hm += loss_items["loss_hm"]
            ep_off += loss_items["loss_offset"]
            nb += 1
            if device.type == "cuda":
                vram_peak = max(vram_peak, torch.cuda.max_memory_allocated(device) / 2**30)
            im_files = batch.get("im_file", []) or []
            n_lq_batch = sum(1 for f in im_files if "longquanshan_ft_gmc_median_official" in str(f))
            lq_seen += n_lq_batch
            uav_seen += int(bs - n_lq_batch)
            if it % args.log_every == 0 or it == 1:
                csv_file.write(
                    f"{epoch},{it},{loss_items['loss_total']:.6f},{loss_items['loss_hm']:.6f},"
                    f"{loss_items['loss_offset']:.6f},"
                    + ",".join(f"{optimizer.param_groups[i]['lr']:.10e}" for i in range(len(groups)))
                    + f",{n_lq_batch},{int(bs - n_lq_batch)}\n"
                )
                csv_file.flush()
                print(
                    f"[Ep{epoch:02d} it{it:5d}/{iters_per_epoch}] loss {loss_items['loss_total']:.4f} "
                    f"(hm {loss_items['loss_hm']:.4f}/off {loss_items['loss_offset']:.4f}) | "
                    f"mix LQ/UAV {n_lq_batch}/{int(bs-n_lq_batch)} | vram {vram_peak:.2f}GB | "
                    f"{time.time()-t_start:.0f}s",
                    flush=True,
                )
            if args.max_iters and it >= args.max_iters:
                print(f"[SMOKE] 达到 max-iters={args.max_iters}，提前结束本 epoch", flush=True)
                break
        csv_file.write(
            f"{epoch},END,{ep_loss/max(nb,1):.6f},{ep_hm/max(nb,1):.6f},{ep_off/max(nb,1):.6f},"
            + ",".join(f"{optimizer.param_groups[i]['lr']:.10e}" for i in range(len(groups)))
            + f",{lq_seen},{uav_seen}\n"
        )
        csv_file.flush()
        ratio = uav_seen / max(lq_seen, 1)
        print(
            f"[EPOCH {epoch}] loss {ep_loss/max(nb,1):.4f} | 实测混样 LQ {lq_seen:,} / UAV {uav_seen:,} "
            f"= 1:{ratio:.3f}（目标 1:{uav_per_lq:.3f}） | {time.time()-t_ep:.1f}s",
            flush=True,
        )
        state = dict(model.state_dict())
        state.update(p0_tensors)
        last_weights = weights_dir / f"epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": state,
                "metrics": {"train_loss": ep_loss / max(nb, 1)},
                "stride": model_stride,
                "imgsz": args.imgsz,
                "p0_kwargs": p0_kwargs,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "finetune": {
                    "route": "mix_ratio_search",
                    "mix_spec": args.mix_spec,
                    "mix_mode": mix_mode,
                    "uav_per_lq_target": uav_per_lq,
                    "measured_lq": lq_seen,
                    "measured_uav": uav_seen,
                    "measured_ratio": ratio,
                    "groups": {g["name"]: g["base_lr"] for g in groups},
                    "freeze_bn": bool(args.freeze_bn),
                    "sched_epochs": sched_epochs,
                    "sched_total_steps": sched_total_steps,
                    "train_total_steps": total_steps,
                },
            },
            last_weights,
        )
        print(f"[SAVE] {last_weights}", flush=True)
    csv_file.close()

    # ---- 门禁：默认只门禁最后一个 epoch（保证吞吐；**每个 epoch 的权重全部保留**）----
    # ⚠️ `--gate-epoch N` 可改为门禁**第 N 个 epoch** 的权重：
    #    训练 5 epoch 时，Ep3 的权重**就是 RC1/Ep3 的等价物**（同 LR 轨迹），
    #    Ep5 的权重**就是 RC1/Ep5 的等价物** ⇒ 一次运行同时覆盖两个参照系。
    gate_epochs = [args.epochs] if args.gate_epochs == "last" else list(range(1, args.epochs + 1))
    gate_epoch = args.gate_epoch if args.gate_epoch > 0 else args.epochs
    if gate_epoch < 1 or gate_epoch > args.epochs:
        raise RuntimeError(f"[GATE-ERR] --gate-epoch {gate_epoch} 越界（训练了 {args.epochs} epoch）")
    gate_weights = weights_dir / f"epoch{gate_epoch}.pt"
    if not gate_weights.is_file():
        raise RuntimeError(f"[GATE-ERR] 门禁权重不存在：{gate_weights}")
    gate_dir = save_dir / "gate" / f"epoch{gate_epoch}"
    gate_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, args.gate_script,
        "--weights", str(gate_weights),
        "--out-dir", str(gate_dir),
        "--gpus", "1",
        "--device", str(args.gpu_id),
        "--infer-gpu", str(args.gpu_id),
        "--cache-gpus", str(args.gpu_id),
        "--baseline-cache", args.baseline_cache,
        "--min-track2-f1", str(GATE_TRACK2_F1_MIN),
        # 绝对上限语义：= 基线 FP 26 + 增量 10 = 36（等价于我们自己的 ΔFP ≤ +10）
        "--max-track2-fp", str(GATE_TRACK2_FP_ABS_MAX),
        "--min-track1-f1", str(GATE_TRACK1_F1_ABS_MIN),  # = 基线 0.905736 + (-0.005)，与我们的 ΔF1 红线等价
    ]
    print(f"\n[GATE] 三轨门禁（{args.mix_spec} / epoch{gate_epoch}"
          f"{'（末 epoch）' if gate_epoch == args.epochs else f'（训练了 {args.epochs} epoch）'}）…"
          f"\n$ {' '.join(cmd)}\n", flush=True)
    if args.skip_gate:
        print("[SMOKE] --skip-gate，跳过门禁", flush=True)
        cmd = ["true"]
    t_gate = time.time()
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    print(f"[GATE] rc={rc} 用时 {time.time()-t_gate:.1f}s", flush=True)

    result = {
        "trial": args.trial_number,
        "trial_name": trial_name,
        "mix_spec": args.mix_spec,
        "mix_mode": mix_mode,
        "uav_per_lq_target": uav_per_lq,
        "measured_ratio": ratio,
        "measured_lq": lq_seen,
        "measured_uav": uav_seen,
        "gate_rc": rc,
        "gate_dir": str(gate_dir),
        "train_epochs": args.epochs,
        "gate_epoch": gate_epoch,
        "sched_epochs": sched_epochs,
        "weights": str(gate_weights),
        "last_weights": str(last_weights),
        "train_loss": ep_loss / max(nb, 1),
        "vram_peak_gb": vram_peak,
        "elapsed_s": time.time() - t_start,
    }
    rep_path = gate_dir / "gate_report.json"
    if rep_path.is_file():
        try:
            metrics = metrics_from_gate_report(rep_path)
        except Exception as exc:
            # ⚠️ 门禁**基础设施失败**（最常见：Track1 推理 CUDA OOM ⇒ 报告缺字段）。
            # 绝不能让异常冒出去把整个 worker 打死 —— 那样会丢掉本 trial 的 27 分钟训练
            # 且 trial_result.json 缺失 ⇒ 无任何诊断信息（实测踩过：trial_0003 因此变成
            # CRASHED，根因要从 track1_infer.log 里翻半天）。
            # 这里捕获后写入**结构化诊断**，调度器据此记为 FAIL（而非"触红线被剪枝"）。
            result["metrics"] = None
            result["utility"] = None
            result["gate_verdict"] = "GATE_REPORT_INCOMPLETE"
            result["tier1_ok"] = False
            result["tier1_reason"] = "GATE_REPORT_INCOMPLETE"
            result["gate_error"] = f"{type(exc).__name__}: {exc}"
            result["infrastructure_failure"] = True
            print(f"[GATE-ERR] 门禁报告不完整 ⇒ 记为基础设施失败：{result['gate_error']}", flush=True)
            print(f"[GATE-ERR] 见 {gate_dir}/track1_infer.log 等子日志", flush=True)
            metrics = None
        if metrics is not None:
            score, reason, detail = evaluate_model_quality(
                metrics, FROZEN_BASELINE,
                track2_f1_min=GATE_TRACK2_F1_MIN,
                track2_fp_max_increment=GATE_TRACK2_FP_MAX_INCREMENT,
                track1_min_delta=GATE_TRACK1_MIN_DELTA,
                alpha=UTILITY_ALPHA, beta=UTILITY_BETA, gamma=UTILITY_GAMMA,
            )
            ok, greason, gdetail = hard_gate(
                metrics, FROZEN_BASELINE,
                track2_f1_min=GATE_TRACK2_F1_MIN,
                track2_fp_max_increment=GATE_TRACK2_FP_MAX_INCREMENT,
                track1_min_delta=GATE_TRACK1_MIN_DELTA,
            )
            result["metrics"] = metrics
            result["utility"] = score if ok else None
            result["gate_verdict"] = reason
            result["gate_detail"] = detail
            result["tier1_ok"] = ok
            result["tier1_reason"] = greason
            u_rc1, _ = utility(metrics, RC1_EPOCH3)
            result["utility_vs_rc1"] = u_rc1
            print(f"[JUDGE] {reason} | U(vs 冻结基线)={score if ok else float('-inf')} | U(vs RC1)={u_rc1:+.6f}", flush=True)
    else:
        result["metrics"] = None
        result["utility"] = None
        result["gate_verdict"] = "NO_REPORT"
        result["tier1_ok"] = False
        result["tier1_reason"] = "NO_REPORT"
        result["infrastructure_failure"] = True
        result["gate_error"] = "门禁未产出 gate_report.json（见 gate 目录下的子日志）"

    (save_dir / "trial_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[RESULT] {save_dir / 'trial_result.json'}", flush=True)
    return 0 if result.get("tier1_ok") else 0  # 一级红线失败不视为进程失败（由调度器剪枝）


# ==============================================================================
# 4. 调度器
# ==============================================================================
def read_trial_result(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_summary_row(csv_path: Path, trial_name: str, params: dict, res: dict | None, status: str,
                      duration_min: float) -> None:
    header = (
        "trial,status,mix_spec,mix_mode,target_ratio,measured_ratio,gate_verdict,"
        "t1_f1,t1_fp,t2_f1,t2_fp,t3_f1,t3_fp,utility,utility_vs_rc1,"
        "lr_scale,warmup_epochs,weight_decay,focal_beta,offset_weight,mosaic,scale,translate,"
        "clip_max_norm,freeze_bn,duration_min\n"
    )
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(header)
    m = (res or {}).get("metrics") or {}

    def g(key, default=""):
        v = m.get(key, default)
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    def _f3(key, default=0.0):
        try:
            return float((res or {}).get(key, default))
        except Exception:
            return default

    row = (
        f"{trial_name},{status},{params.get('mix_spec')},{(res or {}).get('mix_mode','')},"
        f"{_f3('uav_per_lq_target'):.3f},{_f3('measured_ratio'):.3f},"
        f"{(res or {}).get('gate_verdict','')},"
        f"{g('track1_f1')},{m.get('track1_fp','')},{g('track2_f1')},{m.get('track2_fp','')},"
        f"{g('track3_f1')},{m.get('track3_fp','')},"
        f"{(res or {}).get('utility', '')},{(res or {}).get('utility_vs_rc1', '')},"
        f"{params.get('lr_scale'):.4f},{params.get('warmup_epochs'):.3f},{params.get('weight_decay'):.8f},"
        f"{params.get('focal_beta'):.2f},{params.get('offset_weight'):.2f},{params.get('mosaic'):.2f},"
        f"{params.get('scale'):.2f},{params.get('translate'):.2f},{params.get('clip_max_norm'):.2f},"
        f"{params.get('freeze_bn')},{duration_min:.2f}\n"
    )
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(row)


def main() -> int:
    if "--worker" in sys.argv:
        return run_worker()

    ap = argparse.ArgumentParser(description="Optuna 混合比例超参搜索（4 卡并行 / 可断点续跑）")
    ap.add_argument("--longquan-data", default="/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/data.yaml")
    ap.add_argument("--uav-data", default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml")
    ap.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    ap.add_argument("--baseline-cache", default="runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl")
    ap.add_argument("--gate-script", default="manu/training/auto_epoch_gate_check.py")
    ap.add_argument("--output-root", default="runs/optuna_mix_ratio")
    ap.add_argument("--study-name", default="mix_ratio_search")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--n-trials", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gate-epochs", default="last")
    ap.add_argument("--max-hours", type=float, default=20.0, help="到点自动收尾（不再派新 trial）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-seed-trials", action="store_true", help="不注入种子 trial")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    (out_root / "trials").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    # 调度器自写 PID 文件（比 shell 的 $! 可靠：setsid 在某些情况下会 fork，$! 可能是父进程 PID）
    (out_root / "dispatcher.pid").write_text(str(os.getpid()), encoding="utf-8")
    summary_csv = out_root / "optuna_summary.csv"
    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip().isdigit()]
    if not gpu_list:
        raise ValueError(f"没有合法 GPU：{args.gpus}")

    print("=" * 100, flush=True)
    print("🚀 Optuna 混合比例超参搜索", flush=True)
    print(f"   底模        : {args.weights}", flush=True)
    print(f"   数据       : LQ {args.longquan_data}", flush=True)
    print(f"                UAV {args.uav_data}", flush=True)
    print(f"   GPU        : {gpu_list}（{len(gpu_list)} 并行，每 trial 单卡）", flush=True)
    print(f"   Trial/Epoch: {args.n_trials} / {args.epochs}", flush=True)
    print(f"   截止       : {args.max_hours} 小时后自动收尾", flush=True)
    print(f"   输出       : {out_root.resolve()}", flush=True)
    print("=" * 100 + "\n", flush=True)

    db_path = out_root / "study.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{db_path.resolve()}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=12, multivariate=True),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 陈旧 RUNNING trial 自愈（进程被断电/杀死后残留）
    #
    # ⚠️ 必须用公共 API study.tell(trial_number, state=FAIL)，**不能**直接调
    #    study._storage.set_trial_state_values(t.number, ...)。
    # 原因（两层坑，都是实测复现）：
    #    1. Optuna 的 RDB 存储里 trial_id 是 **1 起**的自增主键，而 Trial.number 是
    #       **0 起**的业务编号，两者差 1；set_trial_state_values() 收的是前者。
    #       传 number 会改错行，对已完成的 trial 直接抛 UpdateFinishedTrialError
    #       ⇒ 断点续跑时 dispatcher 直接崩溃。
    #    2. study.get_trials(deepcopy=False) 返回的是 **FrozenTrial**，不是 Trial，
    #       传给 study.tell(trial_obj) 会抛 "Trial must be a trial object or trial number"。
    #    study.tell(number) 两个坑都避开：内部按业务编号取回正确 trial 并用其 _trial_id。
    healed = 0
    for t in study.get_trials(deepcopy=False):
        if t.state == optuna.trial.TrialState.RUNNING:
            try:
                study.tell(t.number, state=optuna.trial.TrialState.FAIL)
                healed += 1
            except Exception as exc:  # 已被别的进程结算完等情况，忽略
                print(f"[HEAL] trial {t.number} 自愈跳过：{exc}", flush=True)
    if healed:
        print(f"[HEAL] 自愈 {healed} 个陈旧 RUNNING trial（标为 FAIL，不污染 TPE）", flush=True)

    # 种子 trial：对照（natural）+ 1:2 / 1:3 / 1:4 / 1:6
    if not args.no_seed_trials and len(study.trials) == 0:
        seeds = []
        for spec in ("natural", "w2.0", "w3.0", "w4.0", "w6.0"):
            p = dict(ROUTE_B_DEFAULTS)
            p["mix_spec"] = spec
            seeds.append(p)
        for p in seeds:
            study.enqueue_trial(p, skip_if_exists=True)
        print(f"[SEED] 注入 {len(seeds)} 个种子 trial（natural 对照 + w2/w3/w4/w6）", flush=True)

    stop_flag = {"stop": False}

    def _on_signal(signum, _frame):
        print(f"\n[SIGNAL] 收到信号 {signum} ⇒ 停止派新 trial，等待在跑的收尾 …", flush=True)
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    running: list[dict] = []
    available = list(gpu_list)
    deadline = time.time() + args.max_hours * 3600
    done_count = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    registry_path = out_root / "workers.json"

    def save_registry() -> None:
        payload = {
            "updated": time.time(),
            "workers": {
                str(info["trial_number"]): {
                    "pid": info["process"].pid,
                    "gpu": info["gpu_id"],
                    "trial_name": info["trial_name"],
                    "started_at": info["start_time"],
                }
                for info in running
            },
        }
        registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    while True:
        now = time.time()
        if stop_flag["stop"]:
            print("[STOP] 不再派新 trial", flush=True)
            break
        if now > deadline:
            print(f"[DEADLINE] 到达 {args.max_hours} 小时截止时间，收尾 …", flush=True)
            break
        if done_count >= args.n_trials:
            print(f"[DONE] 已完成 {done_count} 个 trial", flush=True)
            break

        # 派新 trial
        while available and done_count + len(running) < args.n_trials and time.time() < deadline \
                and not stop_flag["stop"]:
            gpu_id = available.pop(0)
            trial = study.ask()
            params = suggest_params(trial)

            # ---- 运行时配置（可在**不重启调度器**的前提下改后续 trial 的协议）----
            # 文件：<out_root>/runtime_config.json
            #   {"train_epochs": 5}   → 后续 trial 训练 5 epoch（cosine 周期仍为 5 ⇒ 与 RC1 同 LR 轨迹）
            #   {"gate_epoch": 3}     → 门禁量第 3 个 epoch（默认 = 最后一个）
            #   {"workers": 16}       → dataloader worker 数
            # 解析失败一律回落到启动参数（绝不因配置问题打死调度器）。
            rt = read_runtime_config(out_root)
            train_epochs = int(rt.get("train_epochs") or args.epochs)
            default_gate_epoch = int(rt.get("gate_epoch") or 0)
            default_workers = int(rt.get("workers") or args.workers)

            # ---- 每 trial 覆盖（user_attrs，由 enqueue_control_trials.py 写入）----
            ua = dict(getattr(trial, "user_attrs", {}) or {})
            seed = int(ua.get("seed", args.seed + trial.number))
            workers = int(ua.get("workers", default_workers))
            gate_epoch = int(ua.get("gate_epoch", default_gate_epoch))

            trial_name = f"trial_{trial.number:04d}"
            log_file = out_root / "logs" / f"{trial_name}.log"
            cmd = [
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--worker",
                "--trial-number", str(trial.number),
                "--gpu-id", str(gpu_id),
                "--output-root", str(out_root),
                "--longquan-data", args.longquan_data,
                "--uav-data", args.uav_data,
                "--weights", args.weights,
                "--gate-script", args.gate_script,
                "--baseline-cache", args.baseline_cache,
                "--epochs", str(train_epochs),
                "--batch", str(args.batch),
                "--imgsz", str(args.imgsz),
                "--workers", str(workers),
                "--gate-epochs", args.gate_epochs,
                "--gate-epoch", str(gate_epoch),
                "--seed", str(seed),
                "--mix-spec", params["mix_spec"],
                "--lr-scale", str(params["lr_scale"]),
                "--warmup-epochs", str(params["warmup_epochs"]),
                "--weight-decay", str(params["weight_decay"]),
                "--focal-beta", str(params["focal_beta"]),
                "--offset-weight", str(params["offset_weight"]),
                "--mosaic", str(params["mosaic"]),
                "--scale", str(params["scale"]),
                "--translate", str(params["translate"]),
                "--clip-max-norm", str(params["clip_max_norm"]),
                "--freeze-bn", "1" if params["freeze_bn"] else "0",
            ]
            fh = log_file.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
            running.append({
                "trial": trial, "trial_number": trial.number, "trial_name": trial_name,
                "gpu_id": gpu_id, "params": params, "process": proc, "log_file": log_file,
                "log_handle": fh, "start_time": time.time(),
            })
            save_registry()
            gate_desc = "量末Ep" if gate_epoch in (0, train_epochs) else f"量Ep{gate_epoch}"
            print(f"[LAUNCH] {trial_name} → GPU {gpu_id} | mix={params['mix_spec']} | "
                  f"lr_scale={params['lr_scale']:.3f} | 协议 {train_epochs}ep/{gate_desc} "
                  f"| seed={seed} workers={workers} | 日志 {log_file.name}", flush=True)

        # 收割
        still = []
        for info in running:
            ret = info["process"].poll()
            if ret is None:
                still.append(info)
                continue
            info["log_handle"].close()
            available.append(info["gpu_id"])
            save_registry()
            res = read_trial_result(out_root / "trials" / info["trial_name"] / "trial_result.json")
            dur = (time.time() - info["start_time"]) / 60.0
            trial = info["trial"]
            if res is None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                write_summary_row(summary_csv, info["trial_name"], info["params"], None, "CRASHED", dur)
                print(f"[CRASH] {info['trial_name']} rc={ret}（{dur:.1f} min）", flush=True)
            else:
                if res.get("tier1_ok") and res.get("utility") is not None:
                    score = float(res["utility"])
                    study.tell(trial, score)
                    status = "SUCCESS"
                    for k, v in res.items():
                        try:
                            trial.set_user_attr(k, v)
                        except Exception:
                            pass
                elif res.get("infrastructure_failure"):
                    # 门禁**基础设施失败**（如 Track1 CUDA OOM）⇒ 记 FAIL 而不是"触红线被剪枝"，
                    # 两者语义不同：剪枝会进 TPE 的"已探索"统计，失败则应视为"没测成"。
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    status = "CRASHED"
                    for k, v in res.items():
                        try:
                            trial.set_user_attr(k, v)
                        except Exception:
                            pass
                    print(f"[INFRA-FAIL] {info['trial_name']} 门禁基础设施失败：{res.get('gate_error')}",
                          flush=True)
                else:
                    # 一级红线失败 ⇒ 剪枝（不参与 TPE 采样）
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    status = "PRUNED"
                    for k, v in res.items():
                        try:
                            trial.set_user_attr(k, v)
                        except Exception:
                            pass
                write_summary_row(summary_csv, info["trial_name"], info["params"], res, status, dur)
                m = res.get("metrics") or {}
                print(
                    f"[{status}] {info['trial_name']} mix={info['params']['mix_spec']} "
                    f"实测 1:{res.get('measured_ratio', 0):.2f} | {res.get('gate_verdict')} | "
                    f"U={res.get('utility')} | UvsRC1={res.get('utility_vs_rc1')} | {dur:.1f} min",
                    flush=True,
                )
            done_count += 1
        running = still
        save_registry()

        # 心跳（供状态查询）
        (out_root / "status.json").write_text(
            json.dumps(
                {
                    "updated": time.time(),
                    "elapsed_h": round((time.time() - (deadline - args.max_hours * 3600)) / 3600, 3),
                    "done": done_count,
                    "running": [
                        {"trial": i["trial_name"], "gpu": i["gpu_id"], "pid": i["process"].pid,
                         "mix": i["params"]["mix_spec"]}
                        for i in running
                    ],
                    "free_gpus": available,
                    "best_value": (study.best_value if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else None),
                    "best_trial": (study.best_trial.number if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else None),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        time.sleep(3)

    # 收尾：等在跑的 trial 结束（或杀掉）
    if running:
        print(f"[WAIT] 等待 {len(running)} 个在跑的 trial 收尾 …", flush=True)
        for info in running:
            try:
                info["process"].wait(timeout=1800)
            except Exception:
                info["process"].kill()
            info["log_handle"].close()
    save_registry()

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print("\n" + "=" * 100, flush=True)
    print(f"🎉 搜索结束：完成 {len(complete)} / 剪枝 {len(pruned)} / 总计 {len(study.trials)}", flush=True)
    if complete:
        best = study.best_trial
        print(f"🏆 最佳 trial #{best.number}  U={study.best_value:+.6f}", flush=True)
        print(f"   参数：{best.params}", flush=True)
        print(f"   效用分及格线：RC1/Ep3 = 0.027579", flush=True)
    print(f"   汇总表：{summary_csv.resolve()}", flush=True)
    print("=" * 100 + "\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
