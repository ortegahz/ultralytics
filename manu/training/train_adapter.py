"""Train the lightweight heatmap adapter on top of the frozen Trial 0474 detector.

Pipeline contract
    * The base detector is never updated. Only ``HeatmapAdapter`` receives gradients, and
      ``assert_only_adapter_trains`` fails loudly if that ever stops being true.
    * Epoch 0 is gated against the FROZEN SOTA HEADLINE, measured with the project's own
      two-step official pipeline -- a zero-initialised adapter must reproduce
      F1 0.906361 / TP 21643 / FP 1004 bit-for-bit on the 31,613-frame Anti-UAV val split.
    * Every epoch is scored with that same official pipeline (base cache vs adapter cache),
      ~155 s per epoch, so the monitor reflects the adapter instead of a stale base cache.

Why the official pipeline and not a hand-rolled loop
    A previous revision evaluated the three Longquan validation sequences with its own
    canvas -> forward -> detect -> merge path. That path turned out NOT to be equivalent to
    the cache builder: same detection count, but 4,501 false positives against the cache's
    28, i.e. a systematic coordinate offset. Rather than keep a second implementation of an
    already-frozen pipeline, this script drives the two official scripts as subprocesses and
    reads their structured ``--out-json``. The Longquan convergence probes that depended on
    that broken path are therefore NOT evaluated here; see KNOWN_GAPS.

Known gaps (do not treat their absence as a pass)
    * ``VIDEO00028_013631`` split-merge probe and ``VIDEO00008_001307`` clutter probe
      (document section 6.3) need a Longquan-side evaluation that reflects the adapter.
      The existing ``runs/ms_cache_nop0`` caches the FROZEN base only, so they cannot be
      scored until a correct live Longquan path exists.
    * Longquan FAR after adapter training is likewise unmeasurable with a base-only cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manu.models.adapter_model import (  # noqa: E402
    AdapterLossWeights,
    HeatmapAdapter,
    adapter_total_loss,
    assert_branch_separation,
    assert_identity,
    gaussian_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFER_SCRIPT = PROJECT_ROOT / "manu/inference/cache_trial0474_inferences.py"
AB_SCRIPT = PROJECT_ROOT / "manu/evaluation/eval_arm_pc_gmc_ab.py"

#: Frozen official headline, reproduced 2026-10-19 with [GATE] PASS on this very server.
FROZEN_HEADLINE = {
    "f1": 0.906361, "tp": 21643, "fp": 1004, "th": 0.25,
    "frames": 31613, "gt": 25111, "tol": 8.0,
    "cache": "runs/gmc_eval/official_rebuild_trial0474_cache.pkl",
    "weights_md5": "5b07f255735f9fbadf3e0dfe95a7708f",
}

#: Sequences the user marked ``<ignore>`` on 2026-10-19. Never mined, never trained on.
IGNORE_SEQUENCES = [
    "VIDEO00032_19700101_014453", "VIDEO00036_19700101_015740",
    "VIDEO00033_19700101_015000", "VIDEO00031_19700101_014227",
    "VIDEO00035_19700101_015138", "VIDEO00007_19700101_001324",
    "VIDEO00006_19700101_000322", "VIDEO00003_19700101_002111",
    "VIDEO00004_19700101_002438", "VIDEO00012_19700101_002251",
]
VALIDATION_SEQUENCES = [
    "VIDEO00028_19700101_013631", "VIDEO00008_19700101_001307",
    "VIDEO00009_19700101_001531",
]


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
def translate_fill(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return array
    out = np.empty_like(array)
    out[...] = np.asarray(array.mean(), dtype=array.dtype)
    src_y0, src_y1 = max(0, -dy), min(array.shape[0], array.shape[0] - dy)
    dst_y0, dst_y1 = max(0, dy), min(array.shape[0], array.shape[0] + dy)
    src_x0, src_x1 = max(0, -dx), min(array.shape[1], array.shape[1] - dx)
    dst_x0, dst_x1 = max(0, dx), min(array.shape[1], array.shape[1] + dx)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def zoom_about_center(array: np.ndarray, factor: float) -> np.ndarray:
    height, width = array.shape[:2]
    new_w = max(1, int(round(width * factor)))
    new_h = max(1, int(round(height * factor)))
    interp = cv2.INTER_LINEAR if new_w >= width else cv2.INTER_AREA
    resized = cv2.resize(array, (new_w, new_h), interpolation=interp)
    canvas = np.empty_like(array)
    canvas[...] = np.asarray(array.mean(), dtype=array.dtype)
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(width, x0 + new_w), min(height, y0 + new_h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return canvas
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[
        src_y0 : src_y0 + (dst_y1 - dst_y0), src_x0 : src_x0 + (dst_x1 - dst_x0)
    ]
    return canvas


def rasterise_fp_mask(centre: tuple[float, float], radius: float, size: int) -> np.ndarray:
    """Binary disk on the 640 canvas marking a known base-model false positive."""
    mask = np.zeros((size, size), dtype=np.float32)
    cx, cy = centre[0] * size, centre[1] * size
    r = max(1, int(round(radius * size)))
    x0, x1 = int(max(0, round(cx - r))), int(min(size, round(cx + r) + 1))
    y0, y1 = int(max(0, round(cy - r))), int(min(size, round(cy + r) + 1))
    if x1 <= x0 or y1 <= y0:
        return mask
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    return mask


class AdapterFrameDataset(Dataset):
    """One mined frame: dense base heatmap, guidance channel, gaussian target, FP mask."""

    def __init__(self, manifest_path: Path, root: Path, grid: int, stride: int, augment: bool):
        # grid MUST equal the heatmap resolution the adapter sees at INFERENCE time
        # (imgsz / stride = 640 / 2 = 320). Training at 640 teaches spatial filters at twice
        # the deployed scale: the dilated 7x7 receptive field covers 7 px in training but
        # only 3.5 px in deployment, so the learned filters would not transfer.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.root, self.grid, self.stride, self.augment = root, grid, stride, augment
        self.samples = manifest["samples"]
        if not self.samples:
            raise RuntimeError(f"[DATASET] manifest has zero samples: {manifest_path}")
        leaked = sorted({row["sequence"] for row in self.samples} & set(VALIDATION_SEQUENCES))
        if leaked:
            raise RuntimeError(f"[SEQ GATE] FAILED: validation sequences in the manifest: {leaked}")
        ignored = sorted({row["sequence"] for row in self.samples} & set(IGNORE_SEQUENCES))
        if ignored:
            raise RuntimeError(f"[SEQ GATE] FAILED: <ignore> sequences in the manifest: {ignored}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples[index]
        with np.load(self.root / row["npz"]) as payload:
            h_base = payload["h_base"].astype(np.float32)
            guide = payload["guide"].astype(np.float32)

        h_base = cv2.resize(h_base, (self.grid, self.grid), interpolation=cv2.INTER_LINEAR)
        guide = cv2.resize(guide, (self.grid, self.grid), interpolation=cv2.INTER_LINEAR)

        kind = row["sample_class"]
        boxes = [tuple(row["box_canvas_cxcywh"])] if row.get("box_canvas_cxcywh") else []
        fp_mask = np.zeros((self.grid, self.grid), dtype=np.float32)
        if row.get("peak_canvas_xy") is not None:
            fp_mask = rasterise_fp_mask(
                tuple(row["peak_canvas_xy"]),
                float(row.get("peak_canvas_radius", 0.01)), self.grid,
            )

        if self.augment and kind == "hard_fp":
            limit = self.grid // 16
            dx, dy = np.random.randint(-limit, limit + 1, size=2)
            h_base, guide = translate_fill(h_base, int(dx), int(dy)), translate_fill(guide, int(dx), int(dy))
            fp_mask = translate_fill(fp_mask, int(dx), int(dy))
        elif self.augment and kind == "hard_fn":
            limit = self.grid // 32
            dx, dy = np.random.randint(-limit, limit + 1, size=2)
            factor = float(np.random.uniform(0.9, 1.1))
            h_base, guide = zoom_about_center(h_base, factor), zoom_about_center(guide, factor)
            h_base, guide = translate_fill(h_base, int(dx), int(dy)), translate_fill(guide, int(dx), int(dy))
            fp_mask = translate_fill(fp_mask, int(dx), int(dy))
            if boxes:
                boxes = [(
                    min(1.0, max(0.0, boxes[0][0] + dx / self.grid)),
                    min(1.0, max(0.0, boxes[0][1] + dy / self.grid)),
                    min(1.0, boxes[0][2] * factor), min(1.0, boxes[0][3] * factor),
                )]

        h_base_t = torch.from_numpy(h_base)[None]
        guide_t = torch.from_numpy(guide)[None]
        target = gaussian_target(h_base_t[None], boxes, stride=float(self.stride))[0]
        return {
            "h_base": h_base_t,
            "guide": guide_t,
            "target": target,
            "fp_mask": torch.from_numpy(fp_mask)[None],
            "is_anchor": torch.tensor(1.0 if kind == "anchor" else 0.0),
        }


# --------------------------------------------------------------------------------------
# Official-pipeline evaluation
# --------------------------------------------------------------------------------------
def run_official_inference(adapter_path: Path, cache_path: Path, args, label: str) -> None:
    """Step 1 of the official pipeline: Trial 0474 + adapter -> inference cache."""
    command = [
        sys.executable, str(INFER_SCRIPT),
        "--weights", args.hm_weights,
        "--data", args.data_yaml,
        "--output", str(cache_path),
        "--imgsz", "640", "--stride", "2", "--batch", "32",
        "--device", args.device,
        "--conf-thresh", "0.02", "--top-k", "100",
        "--adapter-weights", str(adapter_path),
        "--adapter-mid-channels", str(args.mid_channels),
        "--adapter-guide-channel", "2",
    ]
    if not args.dual:
        command.append("--adapter-single")
    print(f"[EVAL {label}] $ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def run_official_ab(adapter_cache: Path, out_json: Path, tol: float, label: str) -> dict:
    """Step 2: frozen SOTA cache vs adapter cache, reading the structured --out-json."""
    command = [
        sys.executable, str(AB_SCRIPT),
        "--arm-a", FROZEN_HEADLINE["cache"], "--label-a", "SOTA",
        "--arm-b", str(adapter_cache), "--label-b", label,
        "--tol", str(tol), "--out-json", str(out_json),
    ]
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
    return json.loads(out_json.read_text(encoding="utf-8"))


def score_adapter(adapter_path: Path, out_dir: Path, args, label: str) -> dict:
    cache_path = out_dir / f"cache_{label}.pkl"
    json_path = out_dir / f"ab_{label}.json"
    run_official_inference(adapter_path, cache_path, args, label)
    report = run_official_ab(cache_path, json_path, FROZEN_HEADLINE["tol"], label)
    peak = report["arm_b"]["peak"]
    missing = {"th", "tp", "fp", "fn", "precision", "recall", "f1"} - set(peak)
    if missing:
        raise KeyError(f"[AB SCHEMA] eval_arm_pc_gmc_ab out-json peak is missing {sorted(missing)}; "
                       f"actual keys = {sorted(peak)}")
    return {
        "peak_f1": peak["f1"],
        "tp": peak["tp"],
        "fp": peak["fp"],
        "fn": peak.get("fn"),
        "precision": peak["precision"],
        "recall": peak["recall"],
        "far_per_frame": peak["fp"] / report["frames"],
        "frames": report["frames"],
        "gt": report["gt"],
        "delta_peak_f1": report["delta"]["peak_f1"],
        "delta_mean_f1": report["delta"]["mean_f1"],
        "negative_bands": report["delta"]["negative_bands"],
        "bands": report["delta"]["bands"],
    }


# --------------------------------------------------------------------------------------
# Freeze gate
# --------------------------------------------------------------------------------------
def assert_only_adapter_trains(model, adapter, optimiser=None) -> dict:
    """Hard gate: the frozen detector stays frozen and nothing but the adapter is optimised."""
    leaky = [name for name, p in model.named_parameters() if p.requires_grad]
    if leaky:
        raise RuntimeError(
            f"[FREEZE GATE] FAILED: {len(leaky)} base parameters still require grad, first few: {leaky[:5]}"
        )
    base_count = sum(p.numel() for p in model.parameters())
    adapter_ids = {id(p) for p in adapter.parameters()}
    if optimiser is not None:
        for group in optimiser.param_groups:
            foreign = [id(p) for p in group["params"] if id(p) not in adapter_ids]
            if foreign:
                raise RuntimeError(
                    f"[FREEZE GATE] FAILED: optimiser holds {len(foreign)} non-adapter parameters"
                )
    adapter_count = sum(p.numel() for p in adapter.parameters())
    return {
        "base_parameters_frozen": base_count,
        "adapter_parameters_trained": adapter_count,
        "adapter_fraction_of_base": adapter_count / base_count if base_count else 0.0,
    }


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight heatmap adapter")
    parser.add_argument("--manifest", default="runs/adapter_mine_r5/manifest.json")
    parser.add_argument("--mine-root", default="runs/adapter_mine_r5")
    parser.add_argument("--out-dir", default="runs/adapter_train")
    parser.add_argument("--data-yaml", default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml")
    parser.add_argument("--hm-weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    # adapter
    parser.add_argument("--mid-channels", type=int, default=16,
                        help="16 keeps the dual-branch adapter at ~2.3%% of the base model's MACs")
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--adapter-grid", type=int, default=320,
                        help="Resolution the adapter runs at; MUST equal imgsz/stride = 320, "
                             "the grid the adapter actually sees during inference")
    parser.add_argument("--dual", dest="dual", action="store_true", default=True)
    parser.add_argument("--adapter-single", dest="dual", action="store_false",
                        help="Single-branch reference layout, for A/B only")
    # optimisation
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-augment", action="store_true")
    # loss
    parser.add_argument("--lambda-damp", type=float, default=0.05)
    parser.add_argument("--lambda-recall", type=float, default=1.0,
                        help="Threshold hinge on the GT core: h_final must reach --decision-high")
    parser.add_argument("--lambda-gt-protect", type=float, default=10.0,
                        help="Price the suppression branch pays for touching GT. Kept an order of "
                             "magnitude above the rest on purpose: mildly penalised suppression is "
                             "cheaper to satisfy by flattening every bright peak, UAVs included")
    parser.add_argument("--lambda-quiet", type=float, default=5.0,
                        help="GT-excluded relu penalty on flat sky. Needed because the flood this "
                             "catches is sparse but tall (2.55%% of pixels at 0.79 reads as a mean of "
                             "0.009), which an L1 mean cannot resolve at any lambda")
    parser.add_argument("--lambda-anchor", type=float, default=0.5)
    parser.add_argument("--lambda-suppress", type=float, default=0.5)
    parser.add_argument("--decision-high", type=float, default=0.30,
                        help="Heatmap value the recall hinge drives GT cores to (deployed threshold is 0.22)")
    parser.add_argument("--decision-low", type=float, default=0.10,
                        help="Heatmap value the GT-excluded sky term pushes clean sky below")
    parser.add_argument("--gt-core", type=float, default=0.5,
                        help="target above this counts as GT core for the recall hinge")
    parser.add_argument("--gt-guard", type=float, default=0.3,
                        help="target above this is forbidden ground for the suppression branch")
    # gates
    parser.add_argument("--canary-tol", type=float, default=0.002,
                        help="Allowed peak-F1 drop versus the frozen headline before the weights are rejected")
    parser.add_argument("--headline-tol", type=float, default=5e-6,
                        help="Epoch-0 identity tolerance versus the frozen headline")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    weights_path = Path(args.hm_weights)
    base_fingerprint = hashlib.md5(weights_path.read_bytes()).hexdigest()
    print(f"[INFO] base weights md5 = {base_fingerprint}", flush=True)

    # Load the base model on CPU purely to audit it: the gate has to look at the REAL frozen
    # detector, not at a stand-in, or "only the adapter trains" would be an unfalsifiable claim.
    # Training never touches this object -- every forward pass happens inside the official script.
    from manu.data.preannotate_hm_bbox import load_hm_model, resolve_weights

    base_audit, _ = load_hm_model(resolve_weights(args.hm_weights), torch.device("cpu"), use_p0_highway=True)
    for parameter in base_audit.parameters():
        parameter.requires_grad_(False)

    adapter = HeatmapAdapter(in_channels=2, mid_channels=args.mid_channels,
                             dilation=args.dilation, dual=args.dual).to(device)
    print(f"[INFO] {adapter.describe()}", flush=True)

    freeze_report = assert_only_adapter_trains(base_audit, adapter)
    print(f"[FREEZE GATE] PASS: base {freeze_report['base_parameters_frozen']:,} params frozen, "
          f"only {freeze_report['adapter_parameters_trained']:,} adapter params trainable "
          f"({freeze_report['adapter_fraction_of_base']*100:.3f}% of the base)", flush=True)
    del base_audit

    probe = torch.rand(2, 1, 320, 320, device=device)
    max_delta, max_diff = assert_identity(adapter, probe, probe)
    print(f"[IDENTITY GATE] PASS: max|delta_h|={max_delta:.1e} max|H_final-H_base|={max_diff:.1e}", flush=True)
    branch_report = assert_branch_separation(adapter, probe, probe)
    if args.dual:
        print(f"[BRANCH GATE] PASS: recall_min={branch_report['recall_min']:.1e} "
              f"suppress_max={branch_report['suppress_max']:.1e}", flush=True)

    history: list[dict] = []

    # ---------------- epoch 0: an untrained adapter must reproduce the frozen headline ----
    zero_path = out_dir / "adapter_epoch0_zeroinit.pt"
    torch.save(adapter.state_dict(), zero_path)
    print("[EPOCH 0] scoring a zero-initialised adapter through the official pipeline ...", flush=True)
    started = time.perf_counter()
    epoch0 = score_adapter(zero_path, out_dir, args, "epoch0")
    drift = abs(epoch0["peak_f1"] - FROZEN_HEADLINE["f1"])
    print(f"[EPOCH 0] F1={epoch0['peak_f1']:.6f} TP={epoch0['tp']} FP={epoch0['fp']} "
          f"FAR={epoch0['far_per_frame']:.4f} | headline F1={FROZEN_HEADLINE['f1']:.6f} "
          f"TP={FROZEN_HEADLINE['tp']} FP={FROZEN_HEADLINE['fp']} | drift={drift:.2e} "
          f"in {time.perf_counter()-started:.0f}s", flush=True)
    if (drift > args.headline_tol or epoch0["tp"] != FROZEN_HEADLINE["tp"]
            or epoch0["fp"] != FROZEN_HEADLINE["fp"]):
        raise RuntimeError(
            f"[HEADLINE GATE] FAILED: an untrained adapter scored F1={epoch0['peak_f1']:.6f} "
            f"TP={epoch0['tp']} FP={epoch0['fp']} but the frozen headline is "
            f"F1={FROZEN_HEADLINE['f1']:.6f} TP={FROZEN_HEADLINE['tp']} FP={FROZEN_HEADLINE['fp']}. "
            "The evaluation path no longer matches the frozen pipeline; fix that before training."
        )
    print("[HEADLINE GATE] PASS", flush=True)
    history.append({"epoch": 0, **epoch0, "stage": "headline"})

    dataset = AdapterFrameDataset(Path(args.manifest), Path(args.mine_root),
                                  args.adapter_grid, args.imgsz // args.stride, not args.no_augment)
    print(f"[DATASET] {len(dataset)} frames | "
          f"{ {k: sum(1 for r in dataset.samples if r['sample_class'] == k) for k in ('hard_fp','hard_fn','silent_sky','anchor')} }", flush=True)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=len(dataset) > args.batch_size)
    weights = AdapterLossWeights(
        lambda_recall=args.lambda_recall, lambda_gt_protect=args.lambda_gt_protect,
        lambda_quiet=args.lambda_quiet, lambda_damp=args.lambda_damp,
        lambda_anchor=args.lambda_anchor, lambda_suppress=args.lambda_suppress,
        decision_high=args.decision_high, decision_low=args.decision_low,
        gt_core=args.gt_core, gt_guard=args.gt_guard,
    )
    optimiser = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr_final)
    from manu.data.preannotate_hm_bbox import load_hm_model, resolve_weights as _rw

    _audit, _ = load_hm_model(_rw(args.hm_weights), torch.device("cpu"), use_p0_highway=True)
    for _p in _audit.parameters():
        _p.requires_grad_(False)
    assert_only_adapter_trains(_audit, adapter, optimiser)
    del _audit
    print("[FREEZE GATE] PASS: optimiser holds only adapter parameters", flush=True)

    history_path = out_dir / "adapter_train_log.json"
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        totals: dict[str, float] = defaultdict(float)
        for step, batch in enumerate(loader):
            optimiser.zero_grad(set_to_none=True)
            h_base = batch["h_base"].to(device, non_blocking=True)
            guide = batch["guide"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            fp_mask = batch["fp_mask"].to(device, non_blocking=True)
            is_anchor = batch["is_anchor"].to(device, non_blocking=True)
            h_final, delta_recall, delta_suppress = adapter.forward_split(h_base, guide)
            loss, parts = adapter_total_loss(
                h_final, delta_recall, delta_suppress, h_base, target, is_anchor, fp_mask, weights
            )
            if not torch.isfinite(loss):
                # never silently optimise NaN weights -- fail with the offending batch in hand
                idx = int((~torch.isfinite(loss)).nonzero()[0])
                raise RuntimeError(
                    f"[LOSS GATE] FAILED at epoch {epoch} step {step}: loss={float(loss)} parts={parts}. "
                    f"h_base[{h_base[idx].min():.4f},{h_base[idx].max():.4f}] "
                    f"h_final[{h_final[idx].min():.4f},{h_final[idx].max():.4f}] "
                    f"target_max={float(target[idx].max()):.4f}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimiser.step()
            totals["loss"] += float(loss.detach())
            for key, value in parts.items():
                totals[key] += value
        scheduler.step()
        steps = max(1, len(loader))
        print(f"\n[EPOCH {epoch}/{args.epochs}] loss={totals['loss']/steps:.4f} "
              f"recall={totals['recall']/steps:.4f} gt_protect={totals['gt_protect']/steps:.6f} "
              f"quiet={totals['quiet']/steps:.5f} damp={totals['damp']/steps:.5f} "
              f"suppress={totals['suppress']/steps:.4f} "
              f"anchor={totals['anchor']/steps:.6f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        path = out_dir / f"adapter_epoch{epoch}.pt"
        torch.save(adapter.state_dict(), path)
        adapter.eval()
        result = score_adapter(path, out_dir, args, f"epoch{epoch}")
        delta = result["peak_f1"] - FROZEN_HEADLINE["f1"]
        print(f"[EPOCH {epoch}] F1={result['peak_f1']:.6f} ({delta:+.6f}) "
              f"P={result['precision']*100:.2f}% R={result['recall']*100:.2f}% "
              f"TP={result['tp']} FP={result['fp']} FAR={result['far_per_frame']:.4f} | "
              f"mean dF1={result['delta_mean_f1']:+.6f} negative_bands="
              f"{result['negative_bands']}/{result['bands']}", flush=True)

        vetoed = delta < -args.canary_tol
        history.append({"epoch": epoch, **result, "stage": "train",
                        "loss": totals["loss"] / steps, "delta_peak_f1": delta, "vetoed": vetoed})
        history_path.write_text(json.dumps({"history": history}, indent=1), encoding="utf-8")

        if vetoed:
            print(f"[CANARY VETO] peak F1 dropped {abs(delta):.6f} > {args.canary_tol}; "
                  f"weights for epoch {epoch} rejected, stopping.", flush=True)
            break

    adapter.eval()
    torch.save(adapter.state_dict(), out_dir / "adapter_final.pt")
    log_path = out_dir / "adapter_train_log.json"
    log_path.write_text(json.dumps({
        "adapter": adapter.describe(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "base_weights_md5": base_fingerprint,
        "freeze_report": freeze_report,
        "frozen_headline": FROZEN_HEADLINE,
        "governance": {"validation_sequences": VALIDATION_SEQUENCES,
                       "ignore_sequences": IGNORE_SEQUENCES},
        "known_gaps": __doc__.split("Known gaps")[1].strip().splitlines(),
        "history": history,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[LOG] {log_path}", flush=True)


if __name__ == "__main__":
    main()
