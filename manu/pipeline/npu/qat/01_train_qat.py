#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
01_train_qat.py -- the QAT fine-tune of trial_0080 for the RK3588 head.

WHY THIS RUN EXISTS
-------------------
The int8 (PTQ) arm of trial_0080 loses accuracy ONLY to false positives: on the
same 2000 frames the board reports

    fp16 : F1 0.7395   TP 1273   FP 184
    int8 : F1 0.6448   TP 1359   FP 527      <- TP goes UP, FP nearly triples

i.e. the activation quantisation noise manufactures spurious heatmap peaks.
Calibration-set design does not fix it (4 stratified sets differ by 0.048 F1),
the KL algorithm kills the model (output range clipped to 0.2399 < th 0.3) and
mmse does not finish in 4 hours.  The only remaining lever is to make the network
ROBUST to the quantisation during training.

WHAT THIS SCRIPT DOES
---------------------
  1. loads trial_0080/epoch3.pt (the delivered weights);
  2. measures the PRE-quantisation activation statistics of the three head sites
     on TRAIN frames only (never on val) -> the initial clip;
  3. inserts an LSQ fake quantiser at those sites (learnable step size, STE);
  4. fine-tunes with the trial's own loss / optimiser protocol, so the only
     difference against the PTQ arm is the quantisation noise;
  5. writes a checkpoint that carries BOTH the re-trained weights and the
     learned clips, plus the pre-QAT clips (the A1 control arm).

The head is the only quantised part; the backbone stays float16.  That makes the
deployed model a HYBRID (fp16 body + int8 head), which is what the three-arm
comparison A0 (pure fp16) / A1 (int8 head, PTQ clips) / A2 (int8 head, QAT clips)
separates.

DATA HYGIENE
------------
Training uses ONLY the `train` split of the official feature datasets:

    /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/train
    /mnt/data/siping/datasets/manu/uav_gmc_median/images/train

The val split (which contains the 001324 blind track and the golden pool) is
NEVER touched here -- the frames are read from `images/train` and their labels
from `labels/train` only.

Usage (server):
    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
    cd /tmp/pycharm_project_10ae9e2e
    python manu/pipeline/npu/qat/01_train_qat.py \
        --weights runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt \
        --out-dir /mnt/data/siping/ptq_work/qat/trial_0080 --epochs 1 --gpu 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.pipeline.npu.qat import qat_common as Q  # noqa: E402


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class FeatureDataset(Dataset):
    """(image, boxes-on-canvas) pairs out of one feature dataset split."""

    def __init__(self, items: list[tuple[str, str]], imgsz: int = Q.IMGSZ):
        self.items = items
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.items[idx]
        rgb = Q.load_frame_rgb(img_path)
        src_hw = rgb.shape[:2]
        canvas, r, pad_l, pad_t = Q.letterbox_rgb(rgb, self.imgsz)
        boxes_src = Q.read_label_norm(lbl_path)
        boxes = Q.boxes_to_canvas_norm(boxes_src, src_hw, r, pad_l, pad_t, self.imgsz)
        img = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).float() / 255.0
        return img, torch.from_numpy(boxes)


def collate(batch):
    """Stack the images and concatenate the boxes with their batch index."""
    imgs = torch.stack([b[0] for b in batch])
    boxes, bidx = [], []
    for i, (_, b) in enumerate(batch):
        if len(b):
            boxes.append(b)
            bidx.append(torch.full((len(b),), i, dtype=torch.long))
    if boxes:
        boxes = torch.cat(boxes, 0)
        bidx = torch.cat(bidx, 0)
    else:
        boxes = torch.zeros((0, 4))
        bidx = torch.zeros((0,), dtype=torch.long)
    return imgs, boxes, bidx


class MixedSampler(torch.utils.data.Sampler):
    """Draws `num_samples` indices with P(LQ) = 1/(1+r), P(UAV) = r/(1+r).

    Same mouth as the trial's `weighted` mix mode (manu/training/
    optuna_mix_ratio_search.py): the per-epoch sample COUNT is unchanged, only the
    sampling probability moves.
    """

    def __init__(self, n_lq: int, n_uav: int, uav_per_lq: float, num_samples: int, seed: int):
        self.n_lq, self.n_uav = n_lq, n_uav
        self.num_samples = num_samples
        p_lq = 1.0 / (1.0 + uav_per_lq)
        self.p_lq = p_lq
        rng = np.random.default_rng(seed)
        pick_lq = rng.random(num_samples) < p_lq
        lq_idx = rng.integers(0, max(1, n_lq), num_samples)
        uav_idx = rng.integers(0, max(1, n_uav), num_samples)
        self._idx = np.where(pick_lq, lq_idx, n_lq + uav_idx)
        self.measured_lq = int(pick_lq.sum())
        self.measured_uav = int(num_samples - pick_lq.sum())

    def __iter__(self):
        return iter(self._idx.tolist())

    def __len__(self) -> int:
        return self.num_samples


# --------------------------------------------------------------------------------------
# Calibration: the initial clip of every quantised site
# --------------------------------------------------------------------------------------
@torch.no_grad()
def calibrate(model, fq_modules, loader, device, max_batches: int) -> dict:
    """Per-site max of the PRE-quantisation activation, over TRAIN frames only.

    The initial step size is max/255 -- the same min/max rule the RKNN toolkit's
    `normal` algorithm uses, so the A1 control arm (PTQ clips, no QAT) is a fair
    stand-in for "what the toolkit would have picked".
    """
    stats = {name: {"max": 0.0, "n": 0} for name, _ in fq_modules}
    hooks = []

    def mk(name):
        def hook(_mod, inputs):
            t = inputs[0].detach().float()
            st = stats[name]
            st["max"] = max(st["max"], float(t.max().item()))
            st["n"] += t.numel()
        return hook

    for name, fq in fq_modules:
        fq.enabled = False
        hooks.append(fq.register_forward_pre_hook(mk(name)))

    model.eval()
    for i, (imgs, _, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(imgs.to(device, non_blocking=True).float())
    for h in hooks:
        h.remove()
    for _, fq in fq_modules:
        fq.enabled = True
    return {k: v for k, v in stats.items()}


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
FROZEN_BACKBONE = ("b0", "b1", "b2", "b3", "b4")
GROUP_LRS = {"G2_p4p5": 3e-6, "G3_neck": 1.5e-5, "G4_head": 3e-5}


def group_of(name: str) -> str:
    if name.startswith("head"):
        return "G4_head"
    if name.startswith(("b5", "b6", "b7", "b8", "b9", "b10")):
        return "G2_p4p5"
    return "G3_neck"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(Q.SERVER_ROOT /
                                            "runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt"))
    ap.add_argument("--out-dir", default=str(Q.QAT_ROOT / "trial_0080"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="multiplier on the trial_0080 learning rates")
    ap.add_argument("--scale-lr", type=float, default=1e-4,
                    help="learning rate of the LSQ step sizes (they move on their own clock)")
    ap.add_argument("--warmup-epochs", type=float, default=0.5)
    ap.add_argument("--sched-epochs", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--focal-beta", type=float, default=2.7)
    ap.add_argument("--offset-weight", type=float, default=0.55)
    ap.add_argument("--clip-max-norm", type=float, default=5.0)
    ap.add_argument("--uav-per-lq", type=float, default=2.5,
                    help="the trial_0080 mix ratio (natural = 43008/13918 = 3.09)")
    ap.add_argument("--n-samples", type=int, default=0, help="0 = n_lq + n_uav (the trial protocol)")
    ap.add_argument("--max-iters", type=int, default=0, help=">0 -> stop early (smoke test)")
    ap.add_argument("--calib-batches", type=int, default=16, help="calib frames = batches x batch")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ablate-p0", type=int, default=1,
                    help="1 = the p0_off variant (the gate / business-domain mouth)")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 100, flush=True)
    print(f"[QAT] weights={args.weights}")
    print(f"[QAT] out={out_dir}  gpu={args.gpu}  batch={args.batch}  epochs={args.epochs}")
    print("=" * 100, flush=True)

    # ---- model --------------------------------------------------------------
    from manu.pipeline.npu.hm_export_onnx import build_model

    model, ckpt = build_model(args.weights, ablate_p0=bool(args.ablate_p0))
    model_stride = int(ckpt.get("stride", 2))
    feat_h = feat_w = args.imgsz // model_stride

    # ---- data ---------------------------------------------------------------
    lq_items = Q.list_split(Q.FEATURE_LQ, "train")
    uav_items = Q.list_split(Q.FEATURE_UAV, "train")
    print(f"[DATA] LQ train {len(lq_items):,} | UAV train {len(uav_items):,}", flush=True)
    for it in lq_items[:3] + uav_items[:3]:
        print(f"[DATA]   {it[0]}", flush=True)

    items = lq_items + uav_items
    ds = FeatureDataset(items, args.imgsz)
    n_samples = args.n_samples or (len(lq_items) + len(uav_items))
    sampler = MixedSampler(len(lq_items), len(uav_items), args.uav_per_lq, n_samples, args.seed)
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                        collate_fn=collate, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    iters_per_epoch = len(sampler) // args.batch
    print(f"[DATA] iters/epoch={iters_per_epoch}  samples/epoch={n_samples:,} "
          f"(LQ {sampler.measured_lq:,} / UAV {sampler.measured_uav:,} = "
          f"1:{sampler.measured_uav/max(1,sampler.measured_lq):.3f}, target 1:{args.uav_per_lq})",
          flush=True)

    # ---- fake-quant + calibration ------------------------------------------
    placeholder = {name: 1.0 for name, _, _ in Q.QAT_SITES}
    fq_modules = Q.attach_sites(model, placeholder)
    # the fake-quant parameters are created AFTER the model was built, so the
    # device move has to happen here -- otherwise the LSQ step sizes stay on the
    # CPU and the AMP GradScaler trips over a CPU gradient.
    model.to(device)
    calib_loader = DataLoader(FeatureDataset(lq_items, args.imgsz), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers, collate_fn=collate,
                              pin_memory=True)
    t0 = time.time()
    stats = calibrate(model, fq_modules, calib_loader, device, args.calib_batches)
    calib_scales = {}
    for name, _ in fq_modules:
        mx = max(stats[name]["max"], 1e-6)
        calib_scales[name] = mx / Q.QMAX
        print(f"[CALIB] {name:<12} max={mx:.6f}  -> initial step size {calib_scales[name]:.8f} "
              f"(range [0, {mx:.4f}])", flush=True)
    print(f"[CALIB] {args.calib_batches} batches in {time.time()-t0:.1f}s", flush=True)

    # set the calibrated step sizes
    for name, fq in fq_modules:
        with torch.no_grad():
            fq.scale.fill_(calib_scales[name])

    # ---- the output site (the heatmap that leaves the graph) ----------------
    out_fq = Q.attach_output_fq(model)
    print(f"[CALIB] {Q.OUT_SITE[0]:<12} fixed step size {Q.OUT_SCALE:.8f} "
          f"(the post-sigmoid heatmap lives in (0,1))", flush=True)

    # the fake-quant parameters are created AFTER the model was built, so the
    # device move has to happen here -- otherwise the LSQ step sizes stay on the
    # CPU and the AMP GradScaler trips over a CPU gradient.
    model.to(device)

    # ---- optimiser ----------------------------------------------------------
    buckets = {k: [] for k in GROUP_LRS}
    frozen = []
    for name, p in model.named_parameters():
        if any(name == g or name.startswith(g + ".") for g in FROZEN_BACKBONE):
            p.requires_grad_(False)
            frozen.append(p)
            continue
        if ".fq." in name or name.endswith(".fq.scale"):
            continue
        p.requires_grad_(True)
        buckets[group_of(name)].append(p)

    groups = []
    for key, base_lr in GROUP_LRS.items():
        ps = buckets[key]
        if not ps:
            continue
        lr = base_lr * args.lr_scale
        groups.append({"params": ps, "lr": lr, "base_lr": lr, "name": key,
                       "numel": sum(p.numel() for p in ps)})
        print(f"[PARAM] {key:<9} lr={lr:.3e}  params {groups[-1]['numel']:,}", flush=True)
    scale_params = [fq.scale for _, fq in fq_modules]
    groups.append({"params": scale_params, "lr": args.scale_lr, "base_lr": args.scale_lr,
                   "name": "LSQ_scale", "numel": len(scale_params)})
    print(f"[PARAM] {'LSQ_scale':<9} lr={args.scale_lr:.3e}  params {len(scale_params)}", flush=True)
    print(f"[PARAM] frozen backbone params {sum(p.numel() for p in frozen):,}", flush=True)

    optimizer = torch.optim.AdamW(groups, lr=groups[0]["lr"], weight_decay=args.weight_decay)

    sched_total = max(1, args.sched_epochs * iters_per_epoch)
    warmup = max(1, int(round(args.warmup_epochs * iters_per_epoch)))
    warmup = min(warmup, max(1, sched_total - 1))

    def lam(base_lr):
        floor_ratio = min(1.0, 1e-6 / base_lr) if base_lr > 0 else 0.0

        def fn(step):
            if step < warmup:
                return (step + 1) / warmup
            prog = min(1.0, max(0.0, (step - warmup) / max(1, sched_total - warmup)))
            return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
        return fn

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lam(g["base_lr"]) for g in groups])

    from manu.models.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
    from torch.cuda.amp import GradScaler, autocast

    criterion = HeatmapLoss(hm_weight=1.0, offset_weight=args.offset_weight,
                            focal_alpha=2.0, focal_beta=args.focal_beta)
    scaler = GradScaler(enabled=not args.no_amp)

    # ---- the loop -----------------------------------------------------------
    csv_path = out_dir / "qat_train_log.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        cf.write("epoch,iter,loss_total,loss_hm,loss_offset,lr_head,lr_scale,lr_lsq,"
                 + ",".join(f"scale_{n}" for n, _ in fq_modules) + "\n")

    def set_train_mode():
        model.train()
        for name, module in model.named_modules():
            if any(name == g or name.startswith(g + ".") for g in FROZEN_BACKBONE):
                module.eval()

    t_start = time.time()
    gstep = 0
    for epoch in range(1, args.epochs + 1):
        set_train_mode()
        ep_loss = ep_hm = ep_off = 0.0
        nb = 0
        t_ep = time.time()
        for it, (imgs, boxes, bidx) in enumerate(loader, start=1):
            imgs = imgs.to(device, non_blocking=True).float()
            boxes = boxes.to(device, non_blocking=True)
            bidx = bidx.to(device, non_blocking=True)
            targets = generate_heatmaps_and_targets(
                batch_bboxes=boxes, batch_idx=bidx, batch_size=imgs.shape[0],
                feat_shape=(feat_h, feat_w), stride=model_stride, min_radius=1, device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=not args.no_amp):
                preds = model(imgs)
                loss, items = criterion(preds, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], max_norm=args.clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            # keep the step size positive
            with torch.no_grad():
                for _, fq in fq_modules:
                    fq.scale.clamp_(min=1e-12)
            scheduler.step()
            gstep += 1
            ep_loss += items["loss_total"]
            ep_hm += items["loss_hm"]
            ep_off += items["loss_offset"]
            nb += 1
            if it % args.log_every == 0 or it == 1:
                scales = Q.learned_scales(model)
                lrs = {g["name"]: optimizer.param_groups[i]["lr"] for i, g in enumerate(groups)}
                with open(csv_path, "a", encoding="utf-8", newline="") as cf:
                    cf.write(f"{epoch},{it},{items['loss_total']:.6f},{items['loss_hm']:.6f},"
                             f"{items['loss_offset']:.6f},"
                             f"{lrs.get('G4_head', 0.0):.10e},"
                             f"{lrs.get('G3_neck', 0.0):.10e},"
                             f"{lrs.get('LSQ_scale', 0.0):.10e},"
                             + ",".join(f"{scales.get(n, 0.0):.10f}" for n, _ in fq_modules) + "\n")
                print(f"[Ep{epoch:02d} it{it:5d}/{iters_per_epoch}] loss {items['loss_total']:.4f} "
                      f"(hm {items['loss_hm']:.4f}/off {items['loss_offset']:.4f}) | "
                      f"lr_head {lrs.get('G4_head', 0):.2e} | "
                      + " ".join(f"{n}={scales.get(n, 0):.6f}" for n, _ in fq_modules)
                      + f" | {time.time()-t_start:.0f}s", flush=True)
            if args.max_iters and it >= args.max_iters:
                print(f"[SMOKE] max-iters={args.max_iters} reached", flush=True)
                break

        print(f"[EPOCH {epoch}] loss {ep_loss/max(nb,1):.4f} (hm {ep_hm/max(nb,1):.4f} / "
              f"off {ep_off/max(nb,1):.4f}) | {time.time()-t_ep:.1f}s", flush=True)

        scales = Q.learned_scales(model)
        ck = {
            "epoch": epoch,
            "model": Q.strip_wrappers(model.state_dict()),
            "qat_scales": scales,
            "calib_scales": calib_scales,
            "out_scale": Q.OUT_SCALE,
            "stride": model_stride,
            "imgsz": args.imgsz,
            "p0_kwargs": ckpt.get("p0_kwargs", {}),
            "ablate_p0": bool(args.ablate_p0),
            "metrics": {"train_loss": ep_loss / max(nb, 1)},
            "qat": {
                "sites": [(n, p) for n, p, _ in Q.QAT_SITES],
                "out_site": Q.OUT_SITE[0],
                "batch": args.batch, "lr_scale": args.lr_scale, "scale_lr": args.scale_lr,
                "sched_epochs": args.sched_epochs, "warmup_epochs": args.warmup_epochs,
                "uav_per_lq": args.uav_per_lq, "n_samples": n_samples,
                "focal_beta": args.focal_beta, "offset_weight": args.offset_weight,
                "weight_decay": args.weight_decay, "clip_max_norm": args.clip_max_norm,
                "amp": not args.no_amp, "elapsed_s": time.time() - t_start,
                "iters": nb, "seed": args.seed,
            },
        }
        wpath = out_dir / f"qat_epoch{epoch}.pt"
        torch.save(ck, wpath)
        print(f"[SAVE] {wpath}", flush=True)
        (out_dir / f"qat_epoch{epoch}_meta.json").write_text(json.dumps(
            {k: v for k, v in ck.items() if k != "model"}, ensure_ascii=False, indent=1))
    print(f"[DONE] {time.time()-t_start:.0f}s total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
