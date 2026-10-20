"""Lightweight Spatial Residual Adapter for the frozen Trial 0474 heatmap detector.

The base detector is 100% frozen: this module only learns a residual correction ``delta_h``
that is added to the base heatmap. The delta head is zero-initialised, so epoch 0 is a
bit-exact identity mapping and the frozen baseline is reproduced by construction.

Design constraints inherited from the RK3588 delivery target:
  * no custom ops, only Conv2d / GroupNorm / GELU / clamp, all RKNN-convertible;
  * the adapter runs on the same 640x640 letterboxed canvas the native-scale base forward
    uses, so no extra resampling is introduced on the NPU path.

Reference parameter counts (bias=False on the two 3x3 convs):
    mid=32 -> 9,953      mid=48 -> 21,841      mid=56 -> 29,513 (~30k)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def positive_part(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Smooth, strictly non-negative surrogate for ``relu`` that is exactly 0 at the origin.

    A plain ``relu`` would deadlock here: both heads are zero-initialised, so their output sits
    exactly at the kink where ``relu'(0) = 0`` and the branch could never learn anything.
    This form keeps a non-zero gradient at 0 (= 0.5), is monotonically increasing, and never
    returns a negative value -- so the recall branch is physically incapable of suppressing and
    the suppress branch of boosting, rather than merely being asked not to.
    """
    return 0.5 * (x + torch.sqrt(x * x + eps * eps) - eps)


class HeatmapAdapter(nn.Module):
    """Two-branch residual adapter over ``[H_base, guide]``.

    ``dual=True`` gives each job its own trunk and its own head, and -- more importantly --
    constrains their signs, so the separation is physical rather than merely encouraged by the
    loss:

      * recall branch   emits ``+positive_part(...)``  -> can only ever raise the heatmap
      * suppress branch emits ``-positive_part(...)``  -> can only ever lower it

    ``dual=False`` reproduces the single-trunk reference implementation for A/B comparison.
    """

    def __init__(self, in_channels: int = 2, mid_channels: int = 32, dilation: int = 2, dual: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.dilation = dilation
        self.dual = dual

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.GELU(),
        )

        def trunk() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                ),
                nn.GroupNorm(4, mid_channels),
                nn.GELU(),
            )

        if dual:
            self.branch_recall = trunk()
            self.branch_suppress = trunk()
            self.head_recall = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
            self.head_suppress = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
            # Zero-init BOTH heads: epoch 0 stays a bit-exact identity map.
            for head in (self.head_recall, self.head_suppress):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        else:
            self.branch = trunk()
            self.head = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, h_base: torch.Tensor, guide: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(torch.cat([h_base, guide], dim=1))
        if self.dual:
            delta_recall = positive_part(self.head_recall(self.branch_recall(feat)))
            delta_suppress = -positive_part(self.head_suppress(self.branch_suppress(feat)))
            delta_h = delta_recall + delta_suppress
        else:
            delta_h = self.head(self.branch(feat))
        return torch.clamp(h_base + delta_h, 0.0, 1.0), delta_h

    def forward_split(self, h_base: torch.Tensor, guide: torch.Tensor):
        """Same as ``forward`` but also returns the two branch deltas separately (dual only)."""
        feat = self.stem(torch.cat([h_base, guide], dim=1))
        if not self.dual:
            delta = self.head(self.branch(feat))
            return torch.clamp(h_base + delta, 0.0, 1.0), delta, None
        delta_recall = positive_part(self.head_recall(self.branch_recall(feat)))
        delta_suppress = -positive_part(self.head_suppress(self.branch_suppress(feat)))
        return (
            torch.clamp(h_base + delta_recall + delta_suppress, 0.0, 1.0),
            delta_recall,
            delta_suppress,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def branch_report(self) -> dict:
        report = {"dual": self.dual, "total": self.count_parameters()}
        if self.dual:
            report["stem"] = sum(p.numel() for p in self.stem.parameters())
            report["recall_branch"] = sum(
                p.numel() for p in list(self.branch_recall.parameters()) + list(self.head_recall.parameters())
            )
            report["suppress_branch"] = sum(
                p.numel()
                for p in list(self.branch_suppress.parameters()) + list(self.head_suppress.parameters())
            )
        else:
            report["stem"] = sum(p.numel() for p in self.stem.parameters())
            report["single_branch"] = sum(p.numel() for p in list(self.branch.parameters()) + list(self.head.parameters()))
        return report

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v:,}" for k, v in self.branch_report().items() if k != "dual")
        return (
            f"HeatmapAdapter(in={self.in_channels}, mid={self.mid_channels}, "
            f"dilation={self.dilation}, dual={self.dual}) {parts}"
        )


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CenterNet's positive-sample radius for a box of ``height x width``."""
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1**2 - 4 * a1 * c1) ** 0.5 / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2**2 - 4 * a2 * c2) ** 0.5 / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3**2 - 4 * a3 * c3) ** 0.5 / 2
    return max(0.0, min(r1, r2, r3))


def draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: int) -> None:
    """Render one Gaussian peak into ``heatmap[0, 0]`` in place (diameter is odd)."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    device = heatmap.device
    y = torch.arange(-radius, radius + 1, device=device, dtype=heatmap.dtype)
    x = torch.arange(-radius, radius + 1, device=device, dtype=heatmap.dtype)
    g = torch.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / (2 * sigma**2))
    height, width = heatmap.shape[-2:]
    left, right = min(cx, radius), min(width - cx, radius + 1)
    top, bottom = min(cy, radius), min(height - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    masked = heatmap[0, 0, cy - top : cy + bottom, cx - left : cx + right]
    torch.maximum(masked, g[radius - top : radius + bottom, radius - left : radius + right], out=masked)


def gaussian_target(
    h_base: torch.Tensor,
    boxes_norm: list[tuple[float, float, float, float]],
    stride: float,
) -> torch.Tensor:
    """Build ``Y_gt`` for one sample; ``boxes_norm`` are cxcywh fractions of the canvas."""
    height, width = h_base.shape[-2:]
    target = torch.zeros((1, 1, height, width), dtype=h_base.dtype, device=h_base.device)
    for cx_n, cy_n, w_n, h_n in boxes_norm:
        cx = int(cx_n * width)
        cy = int(cy_n * height)
        if not (0 <= cx < width and 0 <= cy < height):
            continue
        radius = int(gaussian_radius(h_n * height / stride, w_n * width / stride))
        draw_gaussian(target, cx, cy, max(0, radius))
    return target


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------
def gaussian_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss, averaged over the whole batch.

    The penalty on negative positions is what pushes ``delta_h`` below zero over hard-FP
    regions, so this single term carries both the boost and the suppression signal.
    """
    # min MUST be eps, not 0.0: with min=0 a clamped pixel of exactly 0 gives log(0) = +inf,
    # and inf * pos(=0 on a negative sample) = NaN, which poisons the whole batch.
    pred = pred.clamp(min=eps, max=1.0 - eps)
    pos = target.eq(1.0).float()
    neg = 1.0 - pos
    neg_weight = torch.pow(1.0 - target, beta)
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = (
        -torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weight * neg
    )
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def damping_loss(
    delta_h: torch.Tensor,
    h_base: torch.Tensor,
    target: torch.Tensor,
    base_threshold: float = 0.15,
    target_threshold: float = 0.10,
) -> torch.Tensor:
    """Pin ``delta_h -> 0`` on flat sky so the adapter cannot drift globally."""
    mask = (h_base < base_threshold) & (target < target_threshold)
    if not bool(mask.any()):
        return delta_h.sum() * 0.0
    return delta_h.abs().mul(mask.float()).sum() / mask.float().sum()


def anchor_protect_loss(h_base: torch.Tensor, h_final: torch.Tensor, is_anchor: torch.Tensor) -> torch.Tensor:
    """One-sided protection on Anti-UAV anchors: only penalise suppression, never boost.

    ``is_anchor`` is a per-sample float/bool mask shaped ``[B]``.
    """
    weights = is_anchor.reshape(-1, 1, 1, 1).to(h_base.dtype)
    per_sample = F.relu(h_base - h_final).mean(dim=(1, 2, 3))
    return (per_sample * weights.reshape(-1)).sum() / weights.sum().clamp(min=1.0)


def fp_suppression_loss(h_final: torch.Tensor, fp_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Background binary cross-entropy restricted to hard-false-positive pixels.

    This is the term that makes the adapter actively *quiet* known clutter: on every pixel
    flagged as a confident base-model false positive it pushes ``H_final`` toward 0, i.e. it
    drives ``delta_h`` negative exactly where the base model is confidently wrong.
    """
    mask = fp_mask.to(h_final.dtype)
    denominator = mask.sum().clamp(min=1.0)
    if float(denominator) <= 1.0 and float(mask.max()) == 0.0:
        return h_final.sum() * 0.0
    probability = h_final.clamp(eps, 1.0 - eps)
    return -(torch.log(1.0 - probability) * mask).sum() / denominator


@dataclass
class AdapterLossWeights:
    """Weights for the evidence-caged adapter loss.

    ``lambda_gt_protect`` is deliberately an order of magnitude above everything else. In infrared
    a UAV and a street lamp are both bright local peaks, so a mildly-penalised suppression term is
    cheaper to satisfy by flattening "bright spots" than by learning the difference -- and the
    measured price was recall 94.92% -> 79.17% on Longquan. The cage has to be expensive to breach.
    """

    lambda_recall: float = 1.0
    lambda_gt_protect: float = 10.0
    lambda_quiet: float = 5.0
    lambda_damp: float = 0.05
    lambda_anchor: float = 0.5
    lambda_suppress: float = 0.5
    damp_base_threshold: float = 0.15
    damp_target_threshold: float = 0.10
    decision_high: float = 0.30
    decision_low: float = 0.10
    gt_core: float = 0.5
    gt_guard: float = 0.3


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``value`` over ``mask``, normalised by the mask's OWN pixel count.

    Every term goes through here. That normalisation is the point: a plain ``.sum() / num_pos``
    lets a 320x320 grid's ~102k background pixels out-vote its ~1-5 positive pixels by four orders
    of magnitude, which is what turned a correction head into a global downward bias.
    """
    weights = mask.to(value.dtype)
    return (value * weights).sum() / weights.sum().clamp(min=1e-6)


def recall_hinge_loss(
    h_final: torch.Tensor, target: torch.Tensor, decision_high: float = 0.30, core: float = 0.5
) -> torch.Tensor:
    """Threshold hinge: every GT core must clear ``decision_high`` (0.22 deployed + 0.08 margin).

    This is the replacement for the focal positive term, which asked for a log-probability nudge
    of order 0.01 while the decision it had to cross was 0.22 -- a push that could never flip a
    pixel. The hinge is flat once a detection already clears the bar, so its gradient is spent
    exactly on the misses.
    """
    core_mask = (target > core).float()
    return _masked_mean(F.relu(decision_high - h_final), core_mask)


def gt_protect_loss(delta_suppress: torch.Tensor, target: torch.Tensor, guard: float = 0.3) -> torch.Tensor:
    """Hard one-way insulation: the suppression branch may not touch ground truth at all.

    ``fp_suppression_loss`` only sees ``fp_mask``, which is zero on targets, so without this term
    crushing a real peak was free. Measured on VIDEO00028 the suppression was 20.7x stronger on GT
    than on background -- the branch had simply learned "bright peak => push down", and a UAV is a
    bright peak too.
    """
    gt_mask = (target > guard).float()
    return _masked_mean(F.relu(-delta_suppress), gt_mask)


def sky_quiet_loss(
    h_final: torch.Tensor,
    h_base: torch.Tensor,
    target: torch.Tensor,
    decision_low: float = 0.10,
    base_threshold: float = 0.15,
    target_threshold: float = 0.10,
) -> torch.Tensor:
    """Keep flat sky down, AMPLITUDE-SENSITIVE and never touching ground truth.

    Why this is a ``relu`` and not the ``damping_loss`` L1 mean that was supposed to cover the sky:
    the v7 failure was a *sparse but tall* flood -- 2.55% of background pixels pushed up to 0.79,
    which showed up as a background mean d_recall of just 0.0090. A mean over ~100k sky pixels
    cannot tell "2.55% of pixels at 0.79" from "everything at 0.009"; it is diluted by the 97.45%
    that did not move, so no lambda can rescue it. ``relu(h_final - decision_low)`` scales with how
    far a pixel was actually lifted, so the flooded pixels dominate the gradient on their own.

    Two masks keep it inside the user's "no downward pressure on targets" rule:
    ``h_base < 0.15`` (only where the base model was already flat) and ``target < 0.10`` (never on
    or near a target). It is therefore incapable of reproducing the v5 collapse, which came from a
    whole-image negative term weighted by ``pred^2`` and so hit the strongest peaks hardest.
    """
    mask = (h_base < base_threshold) & (target < target_threshold)
    return _masked_mean(F.relu(h_final - decision_low), mask)


def adapter_total_loss(
    h_final: torch.Tensor,
    delta_recall: torch.Tensor,
    delta_suppress: torch.Tensor,
    h_base: torch.Tensor,
    target: torch.Tensor,
    is_anchor: torch.Tensor,
    fp_mask: torch.Tensor,
    weights: AdapterLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total adapter loss: an evidence cage, not a global field.

    Three rules, each aimed at one measured failure of the previous focal-based objective:

    1. ``recall_hinge``  -- GT core must reach the real 0.22 decision threshold, not a 0.01 nudge.
    2. ``gt_protect``    -- the suppression branch pays 10x to touch GT, closing the hole that let
                            it crush targets 20.7x harder than background.
    3. ``suppress``      -- hard-FP pixels ONLY; ``quiet`` covers flat sky but is GT-excluded and
                            relu-shaped (a plain L1 mean is blind to sparse tall floods, see
                            ``sky_quiet_loss``); ``damp`` keeps delta -> 0 on sky without sign.

    ``gaussian_focal_loss`` is deliberately NOT used any more. Its negative branch weights every
    background pixel by ``(1 - target)^4 = 1`` while the grid holds ~102k background pixels per
    sample against ~1-5 positive pixels, all divided by the same ``num_pos``. The adapter was
    therefore trained ~99.995% as "push every heatmap down", and because the negative weight grows
    with ``pred^2`` the strongest peaks -- the true targets -- were pushed down hardest.
    """
    delta_h = delta_recall + delta_suppress
    recall = recall_hinge_loss(h_final, target, weights.decision_high, weights.gt_core)
    protect = gt_protect_loss(delta_suppress, target, weights.gt_guard)
    quiet = sky_quiet_loss(
        h_final, h_base, target, weights.decision_low,
        weights.damp_base_threshold, weights.damp_target_threshold,
    )
    damp = damping_loss(
        delta_h, h_base, target, weights.damp_base_threshold, weights.damp_target_threshold
    )
    anchor = anchor_protect_loss(h_base, h_final, is_anchor)
    suppress = fp_suppression_loss(h_final, fp_mask)
    total = (
        weights.lambda_recall * recall
        + weights.lambda_gt_protect * protect
        + weights.lambda_quiet * quiet
        + weights.lambda_damp * damp
        + weights.lambda_anchor * anchor
        + weights.lambda_suppress * suppress
    )
    parts = {
        "recall": float(recall.detach()),
        "gt_protect": float(protect.detach()),
        "quiet": float(quiet.detach()),
        "damp": float(damp.detach()),
        "anchor": float(anchor.detach()),
        "suppress": float(suppress.detach()),
    }
    parts["total"] = float(total.detach())
    return total, parts


def assert_suppress_gt_is_punished(
    model: HeatmapAdapter,
    h_base: torch.Tensor,
    guide: torch.Tensor,
    box: tuple[float, float, float, float] = (0.5, 0.5, 0.10, 0.10),
    crush: float = 0.5,
    weights: AdapterLossWeights | None = None,
) -> dict:
    """Regression gate: driving the suppression branch down on a GT peak must COST loss.

    This is the invariant the old focal-based loss violated, and it is what produced the measured
    collapse (GT suppression 20.7x background, recall 94.92% -> 79.17%). The check is behavioural,
    not structural: it runs the real total loss before and after a synthetic GT crush, so any
    future re-weighting that lets "push targets down" pay for itself fails here.
    """
    weights = weights or AdapterLossWeights()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        target = gaussian_target(h_base, [box], stride=2.0)
        h_final, delta_recall, delta_suppress = model.forward_split(h_base, guide)
        zero_anchor = torch.zeros(h_base.shape[0], dtype=h_base.dtype, device=h_base.device)
        zero_fp = torch.zeros_like(h_base)
        before, _ = adapter_total_loss(
            h_final, delta_recall, delta_suppress, h_base, target, zero_anchor, zero_fp, weights
        )
        hit = (target > 0.30).to(h_base.dtype)
        crushed_suppress = delta_suppress - crush * hit
        crushed_final = torch.clamp(h_base + delta_recall + crushed_suppress, 0.0, 1.0)
        after, _ = adapter_total_loss(
            crushed_final, delta_recall, crushed_suppress, h_base, target, zero_anchor, zero_fp, weights
        )
    model.train(was_training)
    penalty = float(after - before)
    if penalty <= 0.0:
        raise AssertionError(
            f"[EVIDENCE GATE] FAILED: crushing a GT peak changed the loss by {penalty:+.6f}; "
            "the objective still rewards suppressing real targets"
        )
    return {
        "penalty": penalty,
        "recall_hole": float(recall_hinge_loss(h_final, target, weights.decision_high, weights.gt_core)),
        "protect_at_hole": float(gt_protect_loss(delta_suppress - crush * hit, target, weights.gt_guard)),
    }


# --------------------------------------------------------------------------------------
# Identity guard
# --------------------------------------------------------------------------------------
@torch.no_grad()
def assert_identity(
    model: HeatmapAdapter,
    h_base: torch.Tensor,
    guide: torch.Tensor,
    atol: float = 0.0,
) -> tuple[float, float]:
    """Epoch-0 gate: the zero-init head must make ``H_final`` bit-identical to ``H_base``."""
    was_training = model.training
    model.eval()
    h_final, delta_h = model(h_base, guide)
    model.train(was_training)
    max_delta = float(delta_h.abs().max())
    max_diff = float((h_final - h_base).abs().max())
    if max_delta > atol or max_diff > atol:
        raise RuntimeError(
            f"[IDENTITY GATE] FAILED: max|delta_h|={max_delta:.3e} max|H_final-H_base|={max_diff:.3e} "
            f"(tolerance {atol:.3e}). The delta head is not zero-initialised."
        )
    return max_delta, max_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HeatmapAdapter self-check")
    parser.add_argument("--mid-channels", type=int, default=32)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--single", action="store_true", help="Force the single-branch reference layout")
    return parser.parse_args()


@torch.no_grad()
def assert_branch_separation(model: HeatmapAdapter, h_base: torch.Tensor, guide: torch.Tensor) -> dict:
    """Dual mode must never let the recall branch push down or the suppress branch push up."""
    if not model.dual:
        return {"dual": False}
    _, delta_recall, delta_suppress = model.forward_split(h_base, guide)
    worst_up = float(delta_recall.min())
    worst_down = float(delta_suppress.max())
    if worst_up < 0.0 or worst_down > 0.0:
        raise RuntimeError(
            f"[BRANCH GATE] FAILED: recall branch went negative ({worst_up:.3e}) or suppress "
            f"branch went positive ({worst_down:.3e}); the two jobs are no longer separated"
        )
    return {
        "dual": True,
        "recall_min": worst_up,
        "suppress_max": worst_down,
        "recall_abs_mean": float(delta_recall.abs().mean()),
        "suppress_abs_mean": float(delta_suppress.abs().mean()),
    }


if __name__ == "__main__":
    args = parse_args()
    probe_base = torch.rand(2, 1, 64, 64)
    probe_guide = torch.rand(2, 1, 64, 64)

    single = HeatmapAdapter(in_channels=2, mid_channels=args.mid_channels,
                            dilation=args.dilation, dual=False)
    print(single.describe())
    delta, diff = assert_identity(single, probe_base, probe_guide)
    print(f"[IDENTITY GATE] PASS: max|delta_h|={delta:.1e} max|H_final-H_base|={diff:.1e}")

    print()
    dual = HeatmapAdapter(in_channels=2, mid_channels=args.mid_channels,
                          dilation=args.dilation, dual=True)
    print(dual.describe())
    delta, diff = assert_identity(dual, probe_base, probe_guide)
    print(f"[IDENTITY GATE] PASS: max|delta_h|={delta:.1e} max|H_final-H_base|={diff:.1e}")
    report = assert_branch_separation(dual, probe_base, probe_guide)
    print(f"[BRANCH GATE] PASS: recall_min={report['recall_min']:.1e} (<=0) "
          f"suppress_max={report['suppress_max']:.1e} (>=0)")

    # a real gradient must survive the zero-init heads, which is what relu would have killed
    dual.train()
    h_final, delta_h = dual(probe_base, probe_guide)
    loss = h_final.mean()
    loss.backward()
    grad = dual.head_recall.weight.grad
    live = float(grad.abs().max()) if grad is not None else 0.0
    print(f"[GRADIENT GATE] {'PASS' if live > 0 else 'FAIL'}: "
          f"head_recall grad max={live:.3e} (a relu head would be 0 here and stuck)")
    assert live > 0, "[GRADIENT GATE] zero-init head received no gradient -- branch is deadlocked"

    dual.zero_grad(set_to_none=True)
    # the evidence gate only means something against a realistic heatmap: mostly flat with rare
    # peaks. torch.rand() has mean 0.5, which is nothing like a detector output.
    probe_heat = torch.rand(2, 1, 64, 64) ** 6
    report = assert_suppress_gt_is_punished(dual, probe_heat, probe_guide)
    print(f"[EVIDENCE GATE] PASS: crushing one GT peak costs {report['penalty']:+.4f} loss "
          f"(must be > 0); recall_hole={report['recall_hole']:.4f} "
          f"protect_at_hole={report['protect_at_hole']:.4f}")
