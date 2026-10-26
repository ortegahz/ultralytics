#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
01b_check_loss.py -- is the QAT training loss measuring what we think it measures?

THE QUESTION
------------
The trial_0080 training log (runs/optuna_mix_ratio/trials/trial_0080/logs/train_log.csv)
shows a per-iteration loss of 1.07 on the FIRST iteration and ~0.20-0.85 (mean 0.349)
by epoch 3.  The QAT run's first iteration prints 2.09, which is ~7x the converged
loss.  Two very different explanations are possible and they need DIFFERENT fixes:

  (a) the LOADER / TARGETS are wrong (boxes not mapped onto the letterboxed canvas
      the way the ultralytics dataloader maps them) -> the model is being trained
      against the wrong supervision;
  (b) the FAKE-QUANT is genuinely destroying the head -> the clip is far too coarse
      and the LSQ has a much harder job than expected.

This script separates them by measuring the SAME batch three times:

    L_float   : the epoch3.pt model, NO fake-quant, train mode (batch-stat BN)
    L_quant   : the same batch WITH the calibrated fake-quant
    L_eval    : the same batch, NO fake-quant, eval mode (running-stat BN)

  * if L_float ~ 0.3 and L_quant ~ 2.0  -> (b) the clip is the problem
  * if L_float ~ 2.0                     -> (a) the loader/targets are the problem

It also prints the number of positive objects per batch, because the focal loss is
normalised by num_pos and an empty / sparse batch silently rescales the loss.

Run (server):
    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
    python manu/pipeline/npu/qat/01b_check_loss.py --n-batches 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.pipeline.npu.qat import qat_common as Q  # noqa: E402
from manu.pipeline.npu.qat import importlib.util as _il  # noqa: E402  (placeholder, unused)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(Q.SERVER_ROOT /
                                            "runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt"))
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    from manu.pipeline.npu.hm_export_onnx import build_model
    from manu.models.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
    from torch.cuda.amp import autocast

    # reuse the trainer's dataset/collate so this check measures the SAME pipeline
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qat_train", REPO_ROOT / "manu/pipeline/npu/qat/01_train_qat.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    model, ckpt = build_model(args.weights, ablate_p0=True)
    model_stride = int(ckpt.get("stride", 2))
    feat_h = feat_w = Q.IMGSZ // model_stride

    lq_items = Q.list_split(Q.FEATURE_LQ, "train")
    uav_items = Q.list_split(Q.FEATURE_UAV, "train")
    items = lq_items + uav_items
    ds = T.FeatureDataset(items, Q.IMGSZ)
    sampler = T.MixedSampler(len(lq_items), len(uav_items), 2.5,
                             args.n_batches * args.batch, 0)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=6,
                        collate_fn=T.collate, pin_memory=True)

    criterion = HeatmapLoss(hm_weight=1.0, offset_weight=0.55, focal_alpha=2.0, focal_beta=2.7)

    def set_train_mode():
        model.train()
        for name, module in model.named_modules():
            if name.split(".")[0] in ("b0", "b1", "b2", "b3", "b4"):
                module.eval()

    def run(tag, quant: bool):
        vals = []
        npos = []
        for i, (imgs, boxes, bidx) in enumerate(loader):
            if i >= args.n_batches:
                break
            imgs = imgs.to(device).float()
            boxes = boxes.to(device)
            bidx = bidx.to(device)
            targets = generate_heatmaps_and_targets(
                batch_bboxes=boxes, batch_idx=bidx, batch_size=imgs.shape[0],
                feat_shape=(feat_h, feat_w), stride=model_stride, min_radius=1, device=device)
            npos.append(int(targets["mask"].sum().item()))
            with autocast(enabled=True):
                preds = model(imgs)
                loss, items = criterion(preds, targets)
            vals.append((float(loss), float(items["loss_hm"]), float(items["loss_offset"])))
        a = np.asarray(vals)
        print(f"[{tag}] loss={a[:,0].mean():.4f} (hm {a[:,1].mean():.4f} / off {a[:,2].mean():.4f}) "
              f"| per-batch {[round(v[0],3) for v in vals]} | positives/batch {npos}", flush=True)
        return float(a[:, 0].mean())

    set_train_mode()
    l_float = run("L_float  (train mode, NO fake-quant)", quant=False)

    model.eval()
    run("L_eval   (eval  mode, NO fake-quant)", quant=False)

    # ---- now attach the fake quant with the calibrated clips -----------------
    placeholder = {name: 1.0 for name, _, _ in Q.QAT_SITES}
    fq_modules = Q.attach_sites(model, placeholder)
    model.to(device)
    calib_loader = DataLoader(T.FeatureDataset(lq_items, Q.IMGSZ), batch_size=args.batch,
                              shuffle=True, num_workers=6, collate_fn=T.collate)
    stats = T.calibrate(model, fq_modules, calib_loader, device, 4)
    scales = {n: max(stats[n]["max"], 1e-6) / Q.QMAX for n, _ in fq_modules}
    print(f"[calib] scales = {scales}", flush=True)
    for name, fq in fq_modules:
        with torch.no_grad():
            fq.scale.fill_(scales[name])
    Q.attach_output_fq(model)
    model.to(device)

    set_train_mode()
    l_quant = run("L_quant  (train mode, WITH fake-quant)", quant=True)

    print(f"\n[verdict] L_float={l_float:.4f}  L_quant={l_quant:.4f}  "
          f"delta={l_quant - l_float:+.4f}", flush=True)
    if l_float > 1.5:
        print("[verdict] the LOADER/TARGETS are the problem (even the float model scores "
              "~7x the trial's converged loss)", flush=True)
    elif l_quant - l_float > 1.0:
        print("[verdict] the FAKE-QUANT is the problem (the float model is fine, the "
              "quantised one is not) -- the clip needs to be tighter/softer", flush=True)
    else:
        print("[verdict] both are in the trial's range -- the QAT premise holds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
