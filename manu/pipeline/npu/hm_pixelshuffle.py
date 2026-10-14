#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
An ONNX/RKNN-safe, bit-exact replacement for ``nn.PixelUnshuffle``.

Why
---
``nn.PixelUnshuffle`` is implemented in PyTorch as

    x.reshape(B, C, H/f, f, W/f, f) -> permute(0,1,3,5,2,4) -> reshape(...)

For f=2 on a (1,16,640,640) input the permuted tensor is (1,16,2,2,320,320).
rknn-toolkit2 folds that into a single ``Reshape`` to **(1,16,4,102400)** and
then needs a Transpose, which trips the runtime limit:

    W RKNN: Transpose will fallback to CPU, because input shape has exceeded
            the max limit, height(4) * width(102400) = 409600,
            required product no larger than 16384!

409600 is 25x over the 16384 ceiling, so two Transposes leave the NPU and the
P0 highway's downsampling runs on the Cortex cores, with an NPU->CPU->NPU
context switch around it.

The fix
-------
Pixel-unshuffling is just a *strided slice plus a channel concat*. Expressing it
that way keeps every intermediate tensor at a legal shape:

    (B,C,H,W) -> four slices (B,C,H/2,W/2) -> concat on dim 1 -> (B,4C,H/2,W/2)

Channel order is fixed by PyTorch's definition

    out[:, c*f*f + i*f + j, y, x] == in[:, c, y*f + i, x*f + j]

so for f=2 the concat order is [::2,::2], [::2,1::2], [1::2,::2], [1::2,1::2]
and the result is bit-identical to ``nn.PixelUnshuffle`` -- not "close", exact.

This module is used ONLY for the ONNX export path. Training and every CPU-side
reference keep the original ``nn.PixelUnshuffle`` so the checkpoint stays
portable and nothing else changes behaviour.

Verification
------------
    python hm_pixelshuffle.py --verify
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn


class PixelUnshuffleSliceConcat(nn.Module):
    """Bit-exact ``nn.PixelUnshuffle`` via strided slices + channel concat.

    Parameters
    ----------
    downscale_factor:
        Must be 2. The general case would need ``f*f`` slices and a concat of
        ``f*f`` tensors, which is legal but only f=2 occurs here, and an
        unverified general path is worse than an explicit refusal.
    """

    def __init__(self, downscale_factor: int = 2):
        super().__init__()
        if downscale_factor != 2:
            raise ValueError(
                f"only downscale_factor=2 is implemented, got {downscale_factor}"
            )
        self.downscale_factor = downscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.downscale_factor
        # PyTorch's channel layout is out[:, c*f*f + i*f + j] = in[:, c, y*f+i, x*f+j],
        # i.e. the four sub-grids are INTERLEAVED per input channel. A plain
        # torch.cat of the four slices would instead lay them out as blocks
        # (all channels of (0,0), then all of (0,1), ...) -- the right values in
        # the wrong channel order, which is the kind of mistake that still
        # produces a plausible-looking heatmap. stack() keeps the interleaved
        # order and the following reshape is a free view, so ONNX emits only
        # Slice + Stack + Reshape and never the oversized Transpose.
        B, C, H, W = x.shape
        h2, w2 = H // f, W // f
        subs = [x[:, :, i::f, j::f] for i in range(f) for j in range(f)]
        return torch.stack(subs, dim=2).reshape(B, C * f * f, h2, w2)

    def extra_repr(self) -> str:
        return f"downscale_factor={self.downscale_factor}, impl=slice_concat"


def swap_pixel_unshuffle(module: nn.Module, verbose: bool = True) -> int:
    """Recursively replace every ``nn.PixelUnshuffle`` under ``module``.

    DEPRECATED in favour of :func:`swap_to_strided_conv`. Kept because it is
    bit-exact and was needed to establish that the problem is the *layout
    change itself*, not the spelling of the graph -- but it does NOT fix the
    CPU fallback, because RKNN emulates strided Slice with a Transpose and the
    2-D product limit still applies. Do not use it to chase NPU residency.

    Returns the number of replacements.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.PixelUnshuffle):
            setattr(module, name, PixelUnshuffleSliceConcat(child.downscale_factor))
            n += 1
            if verbose:
                print(f"    swapped PixelUnshuffle at .{name} "
                      f"(f={child.downscale_factor}) -> slice+concat")
        else:
            n += swap_pixel_unshuffle(child, verbose=verbose)
    return n


class PixelUnshuffleStridedConv(nn.Module):
    """Bit-exact ``nn.PixelUnshuffle`` as a fixed stride-2 depthwise convolution.

    Why this form
    -------------
    RKNN's NPU accepts a Transpose only when the 2-D product ``H*W`` is at most
    16384 (i.e. about 128x128). Pixel-unshuffling a 640x640 tensor is a layout
    change over 409600 elements, so *every* spelling of it -- reshape+permute,
    strided slice+stack, strided slice+concat -- lands on the CPU.

    A convolution does not need a transpose: it reads a window and writes a
    smaller map in one step, and stride-2 depthwise conv is a first-class RKNN
    operator. Four fixed delta planes produce all four sub-grids at once.

    With ``groups=C`` PyTorch numbers the output channels as ``c * f*f + k``
    for a depthwise conv, which is exactly PyTorch's PixelUnshuffle layout

        out[:, c*f*f + i*f + j, y, x] == in[:, c, y*f + i, x*f + j]

    so no channel permutation is needed and the result is bit-identical.

    The weights are 0/1 and non-learnable, so nothing in the checkpoint changes.
    """

    def __init__(self, downscale_factor: int = 2):
        super().__init__()
        if downscale_factor != 2:
            raise ValueError(
                f"only downscale_factor=2 is implemented, got {downscale_factor}"
            )
        f = downscale_factor
        self.downscale_factor = f
        # The weight is built from Python literals on the first forward call so
        # that ONNX folds it into a graph initializer. Building it any other way
        # (a buffer, a .repeat(), torch.tile) exports a runtime Tile/Expand op
        # instead, and rknn-toolkit2 2.3.0 then dies inside its fold_constant
        # pass with:
        #   UFuncNoLoopError: ufunc 'multiply' did not contain a loop with
        #   signature matching types (dtype('<U21'), dtype('<U21'))
        # i.e. the optimizer tries to fold a tensor that is still a symbolic
        # shape expression. Build it as constants, keep the optimizer happy.
        self._weight: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.downscale_factor
        C = x.shape[1]
        need = C * f * f
        if self._weight is None or self._weight.shape[0] != need:
            w = torch.zeros(need, 1, f, f, dtype=x.dtype)
            for i in range(f):
                for j in range(f):
                    # depthwise numbering: out channel = c*f*f + i*f + j
                    w[i * f + j :: f * f, 0, i, j] = 1.0
            self._weight = w
        return torch.nn.functional.conv2d(x, self._weight, None, stride=f, groups=C)

    def extra_repr(self) -> str:
        return (f"downscale_factor={self.downscale_factor}, "
                f"impl=strided_depthwise_conv")



def verify(verbose: bool = True) -> None:
    """Prove every replacement is bit-identical to nn.PixelUnshuffle."""
    torch.manual_seed(0)
    cases = [
        (1, 16, 640, 640),
        (1, 3, 320, 320),
        (2, 16, 128, 96),
        (1, 64, 8, 8),
    ]
    impls = [
        ("slice+stack+reshape", PixelUnshuffleSliceConcat(2)),
        ("strided depthwise conv", PixelUnshuffleStridedConv(2)),
    ]
    ok_all = True
    for label, alt in impls:
        print(f"\n  --- {label} ---")
        ok_impl = True
        for shape in cases:
            ref = nn.PixelUnshuffle(2)
            x = torch.randint(-128, 128, shape, dtype=torch.float32)
            a, b = ref(x), alt(x)
            ok = a.shape == b.shape and torch.equal(a, b)
            ok_impl &= ok
            if verbose:
                print(f"    {str(shape):<22} ref{tuple(a.shape)} alt{tuple(b.shape)} "
                      f"-> {'EXACT' if ok else 'MISMATCH'}")
        ok_all &= ok_impl
        if not ok_impl:
            raise SystemExit(f"[FATAL] {label} is NOT bit-exact")
        print(f"    [ok] all cases bit-identical to nn.PixelUnshuffle")
    if not ok_all:
        raise SystemExit("[FATAL] a replacement is NOT bit-exact")
    print("\n  [ok] every replacement is bit-exact")


def swap_to_strided_conv(module: nn.Module, verbose: bool = True) -> int:
    """Recursively replace every ``nn.PixelUnshuffle`` with the conv form.

    This is the variant that actually keeps the NPU: a stride-2 depthwise
    convolution is a native RKNN operator, so no transpose of a 640x640 tensor
    is ever required. The weights are four fixed delta planes (0/1) and carry no
    learned parameters, so the checkpoint is unaffected.

    Returns the number of replacements.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.PixelUnshuffle):
            setattr(module, name, PixelUnshuffleStridedConv(child.downscale_factor))
            n += 1
            if verbose:
                print(f"    swapped PixelUnshuffle at .{name} "
                      f"(f={child.downscale_factor}) -> strided depthwise conv")
        else:
            n += swap_to_strided_conv(child, verbose=verbose)
    return n



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="prove bit-exactness against nn.PixelUnshuffle and exit")
    args = ap.parse_args()
    if args.verify:
        verify()
        return
    verify()


if __name__ == "__main__":
    main()
