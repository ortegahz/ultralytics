#!/usr/bin/env python
"""路线 B：全帧直接对等混训（龙泉山 official 1 : Anti-UAV 3.09）—— 首轮试跑。

设计铁律（全部有实测/文献依据，见文件末尾「依据」）
------------------------------------------------
1. **数据管线只用官方 Ultralytics**（铁律〇 之四的延伸）：
   ``build_yolo_dataset`` + ``build_dataloader``，两个源各自建集后用
   ``ConcatDataset`` 拼接。**不写任何 OpenCV imread/resize/letterbox**。
   依据：``manu/memory/falsified_archive.md`` §十四「自定义 DataLoader 导致基模
   精度虚假下挫与样本丢失陷阱」——手写 OpenCV DataLoader 曾使 43,008 样本腰斩成
   21,414，Ep 00 基线 F1 从 0.9064 暴跌到 0.8953、凭空多 576 处假警。
2. **通道序零漂移**：官方 ``Format._format_img`` 会做 BGR→RGB 反转，磁盘序
   [I_t, diff, res] 因此以模型序 [res, diff, I_t] 进入模型。已实测
   （``probe_channel_order_final.py``）：官方管线与龙泉山评估管线的**内容区逐字节
   相同**（max|d|=0），两套训练集磁盘通道序同构（I_t 均在 0 号通道）。
3. **分层冻结**：Group1(b0..b4) 严格冻结且强制 eval()（冻结 BN 统计量）；
   Group2(b5..b10) lr=3e-6；Group3(Neck) lr=1.5e-5；Group4(Head) lr=3e-5。
4. **调度**：每 iteration 推进。warmup 0.5 epoch 线性，之后 cosine，每个参数组
   退火到**绝对**下限 1e-6（组内比例 = 1e-6 / 该组基准 lr）。
5. **判据口径不动**：三轨评测完全沿用各自现有脚本与匹配规则（Anti-UAV 8px 点对点、
   龙泉山 point-in-bbox），本脚本只负责产出权重。

用法::

    # 预演（不训练，仅验证显存/耗时/日志）
    python manu/training/mixed_ft_train.py --dry-run --max-iters 30

    # 正式跑
    python manu/training/mixed_ft_train.py --epochs 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils import DEFAULT_CFG  # noqa: E402
from torch.cuda.amp import GradScaler, autocast  # noqa: E402

from manu.data.preannotate_hm_bbox import load_hm_model, resolve_weights  # noqa: E402
from manu.models.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets  # noqa: E402

# ---------------------------------------------------------------- 分层冻结定义
# 依据 manu/models/heatmap_model.py:620-630 的真实层级注释
GROUP_SPECS = [
    # (组名, 正则, 基准 lr, 描述)
    ("G2_p4p5", r"^b(5|6|7|8|9|10)\.", 3e-6, "P4/P5 特征层"),
    ("G3_neck", r"^(up1|c13|up2|c16|up3|c19|down1|c22|up_p2|fuse_p2|fuse_p1)\.", 1.5e-5, "Neck (FPN+PAN+融合)"),
    ("G4_head", r"^head\.", 3e-5, "Head 分类/回归"),
]
G1_FROZEN_RE = r"^b[0-4]\."  # Stem/P1 + Stage1/P2 + Stage2/P3 —— 严格冻结


def assign_group(name: str) -> str | None:
    import re

    if re.search(G1_FROZEN_RE, name):
        return "G1_frozen"
    for group, pattern, _lr, _desc in GROUP_SPECS:
        if re.search(pattern, name):
            return group
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="路线B 全帧对等混训（1:3.09）")
    p.add_argument("--longquan-data", default="/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/data.yaml")
    p.add_argument("--uav-data", default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml")
    p.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    p.add_argument("--out-dir", default="runs/mixed_ft_route_b")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--device", default="0")
    p.add_argument("--lr-min", type=float, default=1e-6, help="cosine 退火的绝对下限 lr")
    p.add_argument("--warmup-epochs", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-max-norm", type=float, default=5.0)
    # 以下三项取 trial_0474 实际训练值（manu/memory/spire_antiuav_debug.md:367-387），
    # 而非脚本默认值 —— 微调必须与底模当初的训练配方一致，才能保证 Epoch-0 零回归。
    p.add_argument("--offset-weight", type=float, default=0.55)
    p.add_argument("--focal-beta", type=float, default=2.7)
    p.add_argument("--focal-alpha", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="只跑前向反向预演，不做优化器更新")
    p.add_argument("--max-iters", type=int, default=0, help=">0 时只跑这么多 iteration（预演用）")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume-from", default="", help="从某个 epoch 的 pt 续训（只恢复模型权重）")
    # ---- 每 epoch 自动三轨门禁 ----
    p.add_argument("--gate-script", default="manu/training/auto_epoch_gate_check.py")
    p.add_argument("--gate-gpus", type=int, default=4, help="重建龙泉山缓存时的 GPU 分片数")
    p.add_argument("--no-gate", action="store_true", help="不跑门禁（仅调试）")
    p.add_argument(
        "--fuse-mode",
        choices=["stop", "record"],
        default="stop",
        help="门禁未过时：stop=立即熔断停训（默认）；record=记录熔断信号但继续采集后续 epoch，"
        "用于在红线已破时仍取完整 5 轮曲线（调度仍是单条连续 cosine）",
    )
    p.add_argument("--skip-gate-epoch0", action="store_true", help="跳过开训前的 Epoch-0 对齐门禁")
    p.add_argument("--gate-device", default="0", help="Track1 推理所用 GPU")
    p.add_argument(
        "--gate-infer-gpu",
        default="1",
        help="门禁 Track1 推理 GPU（默认 1，避开训练进程占用的 GPU 0）",
    )
    p.add_argument(
        "--gate-cache-gpus",
        default="1,2,3",
        help="门禁龙泉山缓存分片 GPU（逗号分隔；默认避开训练进程占用的 GPU 0）",
    )
    return p.parse_args()


def build_mixed_loader(args: argparse.Namespace) -> tuple:
    """用官方 build_yolo_dataset/build_dataloader 建两个源，拼成 ConcatDataset。"""
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    # 官方超参：与 optuna_p0_nas_distributed.py 训练底模时完全一致。
    # translate/scale/mosaic 取 trial_0474 实际值（0.1 / 0.2 / 0.05），非脚本默认值 —— 见 spire_antiuav_debug.md
    cfg.hsv_h = 0.0
    cfg.hsv_s = 0.0
    cfg.hsv_v = 0.0
    cfg.degrees = 0.0
    cfg.shear = 0.0
    cfg.perspective = 0.0
    cfg.translate = 0.10
    cfg.scale = 0.20
    cfg.mosaic = 0.05
    cfg.fliplr = 0.5
    cfg.flipud = 0.0
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0

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
    total = n_lq + n_uav
    print(
        f"[DATA] 合计 {total:,} 帧 | 龙泉山 {n_lq:,} ({n_lq/total:.2%}) "
        f"| Anti-UAV {n_uav:,} ({n_uav/total:.2%}) | 帧比 1 : {n_uav/n_lq:.3f}",
        flush=True,
    )
    # 满额校验（§十四 的 21,414 腰斩事故的护栏）
    EXPECT_LQ, EXPECT_UAV = 13918, 43008
    if n_lq != EXPECT_LQ or n_uav != EXPECT_UAV:
        raise RuntimeError(
            f"[DATA-GATE] 样本数异常: 龙泉山 {n_lq} (期望 {EXPECT_LQ}) / "
            f"Anti-UAV {n_uav} (期望 {EXPECT_UAV}) —— 疑似扫描/步长映射错配，立即中止"
        )
    print(f"[DATA-GATE] PASS 样本满额 {n_lq:,} + {n_uav:,} = {total:,}", flush=True)

    mixed = ConcatDataset(datasets)
    # ConcatDataset 没有 collate_fn，build_dataloader 会 getattr 找不到 → 显式挂官方静态方法。
    # collate_fn 内部对 batch_idx 做「+= 批内样本位置」的偏移，因此混批后索引依然正确。
    mixed.collate_fn = YOLODataset.collate_fn
    loader = build_dataloader(mixed, batch=args.batch, workers=args.workers, shuffle=True)
    return loader, mixed, n_lq, n_uav


def build_param_groups(model: torch.nn.Module) -> tuple[list[dict], dict]:
    """按分层方案构造优化器参数组；Group1 严格冻结（requires_grad=False + eval 模式）。"""
    buckets: dict[str, list[torch.nn.Parameter]] = {name: [] for name, _p, _l, _d in GROUP_SPECS}
    frozen: list[torch.nn.Parameter] = []
    unmatched: list[str] = []
    for name, param in model.named_parameters():
        group = assign_group(name)
        if group == "G1_frozen":
            param.requires_grad_(False)
            frozen.append(param)
        elif group is None:
            unmatched.append(name)
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            buckets[group].append(param)
    if unmatched:
        raise RuntimeError(f"[PARAM-GATE] 有参数未归组，拒绝开训: {unmatched[:10]}")

    groups = []
    for name, _pattern, lr, desc in GROUP_SPECS:
        params = buckets[name]
        if not params:
            continue
        n = sum(p.numel() for p in params)
        groups.append({"params": params, "lr": lr, "base_lr": lr, "name": name, "desc": desc, "numel": n})
        print(f"[PARAM] {name:<9} lr={lr:.2e}  参数 {n:>10,}  ({desc})", flush=True)
    print(f"[PARAM] G1_frozen  参数 {sum(p.numel() for p in frozen):>10,}  (严格冻结 + eval 保护 BN)", flush=True)
    return groups, {"frozen": frozen, "buckets": buckets}


def set_frozen_eval(model: torch.nn.Module) -> None:
    """冻结的骨干段强制 eval()，避免 BN running stats 被更新（§十七 教训：BN 漂移毁掉底模）。"""
    model.train()
    for name, module in model.named_modules():
        if any(name == g or name.startswith(g + ".") for g in ("b0", "b1", "b2", "b3", "b4")):
            module.eval()


def run_gate(
    weights: Path, gate_root: Path, epoch: int, args: argparse.Namespace, history: list
) -> bool:
    """跑一次三轨门禁。

    返回值语义：
      True  = 通过
      False = 指标未达标（质量熔断）
    基础设施失败（门禁退出码 3，如推理 OOM / 缓存重建失败）会直接抛 RuntimeError，
    **绝不**被当成"模型退化"而误熔断。
    """
    gate_dir = gate_root / f"epoch{epoch}"
    cmd = [
        sys.executable,
        args.gate_script,
        "--weights",
        str(weights),
        "--out-dir",
        str(gate_dir),
        "--gpus",
        str(args.gate_gpus),
        "--device",
        args.gate_device,
        "--infer-gpu",
        args.gate_infer_gpu,
        "--cache-gpus",
        args.gate_cache_gpus,
        "--strict-gate",
    ]
    print(f"\n[GATE] 触发 Epoch{epoch} 三轨门禁 …\n$ {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    rep_path = gate_dir / "gate_report.json"
    rec: dict = {"epoch": epoch, "weights": str(weights), "rc": rc, "elapsed_s": round(time.time() - t0, 1)}
    if rep_path.is_file():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        rec["verdict"] = rep.get("verdict")
        rec["tracks"] = rep.get("tracks", {})
        rec["gates"] = rep.get("gates", {})
    history.append(rec)
    (gate_root / "gate_history.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open(gate_root / "gate_history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if rc == 3:
        raise RuntimeError(
            f"[GATE-INFRA] Epoch{epoch} 门禁因基础设施失败退出（rc=3），"
            f"**不是**指标未达标。详见 {gate_dir}/。"
        )
    return rc == 0


def make_scheduler(optimizer: torch.optim.Optimizer, groups: list[dict], args, total_steps: int, iters_per_epoch: int):
    """每 iteration 推进：warmup 线性 → cosine → 绝对下限 lr_min。

    ⚠️ warmup 长度必须按 **单 epoch 步数**算（``warmup_epochs × iters_per_epoch``），
    早期版本误用 ``total_steps``，导致 0.5 epoch 的 warmup 被拉长成 2.5 epoch。
    """
    warmup_steps = max(1, int(round(args.warmup_epochs * iters_per_epoch)))
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))

    def make_lambda(base_lr: float):
        floor_ratio = min(1.0, args.lr_min / base_lr) if base_lr > 0 else 0.0

        def fn(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cos

        return fn

    lambdas = [make_lambda(g["base_lr"]) for g in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    print(
        f"[SCHED] 总步数 {total_steps} | warmup {warmup_steps} 步（{args.warmup_epochs} epoch）| "
        f"cosine 退火到绝对下限 lr={args.lr_min:.1e}",
        flush=True,
    )
    return scheduler


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("=" * 100, flush=True)
    print(f"  路线 B 全帧对等混训 | device={device} | epochs={args.epochs} | batch={args.batch}", flush=True)
    print("=" * 100, flush=True)

    # ---- 模型：与底模完全同构（去 P0），并复用权威加载函数 ----
    model, model_stride = load_hm_model(
        resolve_weights(args.weights), device, stride_fallback=2, use_p0_highway=False
    )
    model.to(device)
    feat_h = feat_w = args.imgsz // model_stride

    # 保存时需要把源 ckpt 的 p0 张量原样带回，否则评测侧 --ablate-p0 会触发
    # ABLATION-P0 GATE（"0 p0 tensors were dropped"）而拒载。
    src_ckpt = torch.load(resolve_weights(args.weights), map_location="cpu")
    src_state = src_ckpt["model"] if "model" in src_ckpt else src_ckpt.get("state_dict", src_ckpt)
    if hasattr(src_state, "state_dict"):
        src_state = src_state.state_dict()
    p0_tensors = {k: v for k, v in src_state.items() if k.startswith("p0_highway")}
    p0_kwargs = src_ckpt.get("p0_kwargs", {})
    print(f"[CKPT] 源 ckpt 带回 p0 张量 {len(p0_tensors)} 个（仅供评测侧 ablation 门控，本身不参与前向）", flush=True)

    resume_state: dict = {}
    if args.resume_from:
        blob = torch.load(args.resume_from, map_location="cpu")
        sd = blob["model"] if "model" in blob else blob
        own = model.state_dict()
        n = 0
        for k, v in sd.items():
            kk = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v)
                n += 1
        print(f"[CKPT] 从 {args.resume_from} 恢复 {n} 个张量", flush=True)
        if "optimizer" in blob:
            resume_state = blob
            print(
                f"[CKPT] 完整续训状态就绪：epoch={blob.get('epoch')} "
                f"global_step={blob.get('global_step')}",
                flush=True,
            )

    # ---- 数据 ----
    loader, mixed, n_lq, n_uav = build_mixed_loader(args)
    iters_per_epoch = len(loader)
    total_steps = iters_per_epoch * args.epochs
    print(f"[DATA] iters/epoch = {iters_per_epoch} | 总优化步数 = {total_steps}", flush=True)

    # ---- 参数组 ----
    groups, group_info = build_param_groups(model)
    optimizer = torch.optim.AdamW(groups, lr=groups[0]["lr"], weight_decay=args.weight_decay)
    # AdamW 的 lr 由 param_group 各自持有；构造后显式写回各组基准 lr
    for pg, g in zip(optimizer.param_groups, groups):
        pg["lr"] = g["base_lr"]
        pg["initial_lr"] = g["base_lr"]
    scheduler = make_scheduler(optimizer, groups, args, total_steps, iters_per_epoch)

    # ---- 断点续训：还原优化器/调度器/RNG，使 LR 调度与 epoch 计数连续 ----
    start_epoch = 1
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        for pg, g in zip(optimizer.param_groups, groups):
            pg["initial_lr"] = g["base_lr"]
        torch.set_rng_state(resume_state["torch_rng_state"])
        np.random.set_state(resume_state["numpy_rng_state"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        print(
            f"[RESUME] 从 epoch {resume_state.get('epoch')} 续训 → 起始 epoch {start_epoch} | "
            f"调度器 last_epoch={scheduler.last_epoch} | 当前 lr="
            f"{[round(pg['lr'], 12) for pg in optimizer.param_groups]}",
            flush=True,
        )

    criterion = HeatmapLoss(
        hm_weight=1.0,
        offset_weight=args.offset_weight,
        focal_alpha=args.focal_alpha,
        focal_beta=args.focal_beta,
    )
    scaler = GradScaler(enabled=not args.no_amp)

    csv_path = logs_dir / "train_log.csv"
    csv_header = (
        "epoch,iter,global_step,loss_total,loss_hm,loss_offset,"
        + ",".join(f"lr_{g['name']}" for g in groups)
        + ",elapsed_s,lr_mem_gb,vram_peak_gb,batch_lq,batch_uav\n"
    )
    # 续训时追加，不覆盖既有日志
    csv_mode = "a" if resume_state and csv_path.is_file() else "w"
    csv_file = open(csv_path, csv_mode, encoding="utf-8", newline="")
    if csv_mode == "w":
        csv_file.write(csv_header)
    csv_file.flush()
    jsonl_path = logs_dir / "train_log.jsonl"

    def log_row(row: dict) -> None:
        csv_file.write(
            f"{row['epoch']},{row['iter']},{row['global_step']},{row['loss_total']:.6f},"
            f"{row['loss_hm']:.6f},{row['loss_offset']:.6f},"
            + ",".join(f"{row['lrs'][g['name']]:.10e}" for g in groups)
            + f",{row['elapsed_s']:.1f},{row['lr_mem_gb']:.3f},{row['vram_peak_gb']:.3f},"
            f"{row['batch_lq']},{row['batch_uav']}\n"
        )
        csv_file.flush()
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    gate_root = out_dir / "gate"
    gate_root.mkdir(parents=True, exist_ok=True)
    gate_history: list = []

    # ---- Epoch-0 权重对齐门禁：起跑线必须精确复现冻结基线，否则后续对比不可信 ----
    if not args.no_gate and not args.skip_gate_epoch0 and not args.dry_run:
        print("\n[EPOCH-0] 开训前先验证起跑线（原始权重）复现冻结基线 …\n", flush=True)
        src_ckpt_path = Path(resolve_weights(args.weights))
        if not run_gate(src_ckpt_path, gate_root, 0, args, gate_history):
            print("[FATAL] Epoch-0 门禁未过 —— 起跑线无法复现冻结基线，中止开训", flush=True)
            csv_file.close()
            return 1
        print("[EPOCH-0] PASS 起跑线对齐", flush=True)

    global_step = int(resume_state.get("global_step", 0)) if resume_state else 0
    vram_peak = 0.0
    t_start = time.time()
    stop = False

    for epoch in range(start_epoch, args.epochs + 1):
        set_frozen_eval(model)
        ep_loss = ep_hm = ep_off = 0.0
        nb = 0
        t_ep = time.time()
        for it, batch in enumerate(loader, start=1):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"].to(device, non_blocking=True)
            b_idx = batch["batch_idx"].to(device, non_blocking=True)
            bs = imgs.shape[0]

            targets = generate_heatmaps_and_targets(
                batch_bboxes=bboxes,
                batch_idx=b_idx,
                batch_size=bs,
                feat_shape=(feat_h, feat_w),
                stride=model_stride,
                min_radius=1,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=not args.no_amp):
                preds = model(imgs)
                loss, loss_items = criterion(preds, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Ep{epoch} it{it} loss 非有限值，跳过该步", flush=True)
                continue

            scaler.scale(loss).backward()
            if not args.dry_run:
                # dry-run 只做前反向用于显存/耗时预演，不碰优化器状态
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

            if it % args.log_every == 0 or it == 1:
                lrs = {g["name"]: optimizer.param_groups[i]["lr"] for i, g in enumerate(groups)}
                # 实测每 batch 的两源混样占比（从 im_file 数，不靠假设）
                im_files = batch.get("im_file", []) or []
                n_lq_batch = sum(
                    1
                    for f in im_files
                    if "longquanshan_ft_gmc_median_official" in str(f)
                )
                row = {
                    "epoch": epoch,
                    "iter": it,
                    "global_step": global_step,
                    "loss_total": loss_items["loss_total"],
                    "loss_hm": loss_items["loss_hm"],
                    "loss_offset": loss_items["loss_offset"],
                    "lrs": lrs,
                    "elapsed_s": round(time.time() - t_start, 1),
                    "lr_mem_gb": round(torch.cuda.memory_reserved(device) / 2**30, 3) if device.type == "cuda" else 0.0,
                    "vram_peak_gb": round(vram_peak, 3),
                    "batch_lq": n_lq_batch,
                    "batch_uav": int(bs - n_lq_batch),
                    "note": "dry-run" if args.dry_run else "train",
                }
                log_row(row)
                print(
                    f"[Ep{epoch:02d} it{it:5d}/{iters_per_epoch}] "
                    f"loss {loss_items['loss_total']:.4f} (hm {loss_items['loss_hm']:.4f} / off {loss_items['loss_offset']:.4f}) | "
                    + " | ".join(f"lr {g['name']} {optimizer.param_groups[i]['lr']:.3e}" for i, g in enumerate(groups))
                    + f" | mix LQ/UAV {n_lq_batch}/{int(bs-n_lq_batch)}"
                    + f" | vram_peak {vram_peak:.2f}GB | {time.time()-t_start:.0f}s",
                    flush=True,
                )

            if args.max_iters and it >= args.max_iters:
                stop = True
                break

        avg = (ep_loss / max(nb, 1), ep_hm / max(nb, 1), ep_off / max(nb, 1))
        print(
            f"[EPOCH {epoch}] loss {avg[0]:.4f} (hm {avg[1]:.4f} / off {avg[2]:.4f}) | "
            f"{time.time()-t_ep:.1f}s | vram_peak {vram_peak:.2f}GB",
            flush=True,
        )

        if not args.dry_run:
            state = dict(model.state_dict())
            state.update(p0_tensors)  # 带回 p0 张量，满足评测侧 --ablate-p0 的门控
            ckpt = {
                "epoch": epoch,
                "model": state,
                "metrics": {"train_loss": avg[0], "train_loss_hm": avg[1], "train_loss_offset": avg[2]},
                "stride": model_stride,
                "imgsz": args.imgsz,
                "p0_kwargs": p0_kwargs,
                # —— 断点续训所需：优化器/调度器/RNG 全量状态 ——
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "finetune": {
                    "route": "B_mixed_equal_full_frame",
                    "groups": {g["name"]: g["base_lr"] for g in groups},
                    "g1_frozen": True,
                    "dataset": {"longquan": n_lq, "uav": n_uav, "total": n_lq + n_uav},
                    "clip_max_norm": args.clip_max_norm,
                    "warmup_epochs": args.warmup_epochs,
                    "lr_min": args.lr_min,
                    "weight_decay": args.weight_decay,
                },
            }
            path = weights_dir / f"epoch{epoch}.pt"
            torch.save(ckpt, path)
            print(f"[SAVE] {path}", flush=True)

            # ---- 每 epoch 自动三轨门禁，失败即熔断 ----
            if not args.no_gate:
                ok = run_gate(path, gate_root, epoch, args, gate_history)
                if not ok:
                    if args.fuse_mode == "record":
                        print(
                            f"[FUSE-RECORD] Epoch{epoch} 三轨门禁未过 —— **已记录熔断信号**，"
                            f"按 --fuse-mode record 继续采集后续 epoch（权重已保留）",
                            flush=True,
                        )
                    else:
                        print(
                            f"[FUSE] Epoch{epoch} 三轨门禁未过 —— 熔断，停止训练（权重已保留）",
                            flush=True,
                        )
                        stop = True
                        break
                else:
                    print(f"[GATE] Epoch{epoch} 门禁通过，继续训练", flush=True)

        if stop:
            break

    csv_file.close()
    print("=" * 100, flush=True)
    print(f"  完成 | 总耗时 {time.time()-t_start:.1f}s | vram_peak {vram_peak:.2f}GB", flush=True)
    print(f"  日志: {csv_path}", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
