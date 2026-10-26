#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
ptq_qat_common.py -- shared contracts for the RK3588 QAT (quantisation-aware training) run.

What this module owns
---------------------
Everything the four QAT consumers must agree on byte-for-byte:

  1. 21_train_qat.py        -- the LSQ fake-quant + the fine-tune loop
  2. 02_export_qdq_onnx.py  -- freezing the learned clips into ONNX Q/DQ nodes
  3. 03_build_rknn.py       -- rknn.build(do_quantization=False) + export_rknn
  4. 04_eval_pc_sim.py      -- the PC-simulator scoring (same mouth as the PTQ arm)

--------------------------------------------------------------------------------------
THE PIXEL / NORMALISATION CONTRACT (inherited from manu/pipeline/npu/ptq/ptq_common.py)
--------------------------------------------------------------------------------------
  * features are read with cv2.imread()  -> BGR  -> reversed to RGB (model order),
    which is byte-identical to what the board's stbi_load() hands to the NPU;
  * letterbox is 640x640 with pad value 114;
  * the graph input is UINT8 NHWC and the /255 lives INSIDE the graph
    (manu/pipeline/npu/make_uint8_input_onnx.py), so rknn.config() must be called
    with mean_values=None / std_values=None.

--------------------------------------------------------------------------------------
THE FAKE-QUANT == Q/DQ EQUIVALENCE (the property the whole experiment rests on)
--------------------------------------------------------------------------------------
The training-time fake quantiser and the exported ONNX Q/DQ pair must be the SAME
function of the input, otherwise "QAT" only trains against a simulator of the thing
that gets deployed.

    fake-quant (training) : q = clamp(round(x / s), 0, 255)          ; xq = q * s
    ONNX  QuantizeLinear  : q = clamp(round(x / y_scale) + y_zp, 0, 255)
    ONNX  DequantizeLinear: xq = (q - y_zp) * y_scale

With y_zp = 0 and y_scale = s the two are IDENTICAL, including the rounding tie
break (torch.round and ONNX round are both round-half-to-even).

The BOARD sees the same grid, because the RKNN toolkit re-centres a uint8
zero-point of 0 to the int8 zero-point -128 and keeps the scale, and the board
driver dequantises with

    value = (q_int8 - zp_int8) * scale      ->  (q_uint8) * scale     (zp = -128)

so the board's heatmap values sit exactly on the 1/s grid that QAT trained against.

--------------------------------------------------------------------------------------
THE ZERO-POINT CONVENTION (why every site uses zp = 0)
--------------------------------------------------------------------------------------
Every quantised site in this run is a POST-ReLU activation or the post-sigmoid
heatmap, i.e. its value range starts at 0.  zp = 0 therefore wastes no level and
keeps the ONNX pair legal for uint8 (y_zero_point must live in [0, 255]).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Paths (all overridable from the environment, same convention as ptq_common.py)
# --------------------------------------------------------------------------------------
SERVER_ROOT = Path(os.environ.get("PTQ_SERVER_ROOT", "/tmp/pycharm_project_10ae9e2e"))
WORK_ROOT = Path(os.environ.get("PTQ_WORK_ROOT", "/mnt/data/siping/ptq_work"))
QAT_ROOT = Path(os.environ.get("QAT_ROOT", str(WORK_ROOT / "qat")))
LOG_ROOT = Path(os.environ.get("QAT_LOG_ROOT", str(WORK_ROOT / "logs")))

FEATURE_LQ = Path(os.environ.get(
    "QAT_FEATURE_LQ", "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official"))
FEATURE_UAV = Path(os.environ.get(
    "QAT_FEATURE_UAV", "/mnt/data/siping/datasets/manu/uav_gmc_median"))

# --------------------------------------------------------------------------------------
# Model / IO constants (identical to the PTQ arm)
# --------------------------------------------------------------------------------------
IMGSZ = 640
PAD_VALUE = 114
STRIDE = 2
TH_BASE = 0.3
TOPK = 100

# --------------------------------------------------------------------------------------
# The quantised sites.
#
# Rationale (this is the experiment's one lever): the int8 arm loses ~100% of its
# accuracy to FALSE POSITIVES (FP 527 vs 184 for fp16 on the same 2000 frames, with
# TP going UP), i.e. activation quantisation noise manufactures spurious heatmap
# peaks.  The heatmap is produced by the HEAD, so the head's activations are where
# the noise is injected and where the network can learn to be robust to it.
#
# Only the head is quantised: the backbone stays float16, which is what makes the
# QDQ model a HYBRID (fp16 body + int8 head).  That is the point of the A0/A1/A2
# three-arm comparison -- it separates "int8 head" from "QAT-learned clips".
# --------------------------------------------------------------------------------------
# (logical name, attribute path on the detector, kind)
#   kind "act" -> LSQ, learnable per-tensor scale (the post-ReLU activation)
#   kind "out" -> fixed scale, the post-sigmoid heatmap that leaves the graph
QAT_SITES = [
    ("feat_conv1", "head.feat_conv.1", "act"),   # 64ch @320x320, the input of both branches
    ("heatmap0", "head.heatmap.0", "act"),       # 64ch @320x320, the input of the 1x1 heatmap conv
    ("offset0", "head.offset.0", "act"),         # 64ch @320x320, the input of the 1x1 offset conv
]
OUT_SITE = ("heatmap_out", None, "out")          # the sigmoid heatmap itself

# the post-sigmoid heatmap lives in (0, 1); 1/255 is the finest legal grid
OUT_SCALE = 1.0 / 255.0
QMIN, QMAX = 0, 255


# --------------------------------------------------------------------------------------
# The fake quantiser
# --------------------------------------------------------------------------------------
class _FakeQuantSTE(torch.autograd.Function):
    """STE forward + LSQ step-size gradient.

    forward : xq = round(clamp(x / s, qmin, qmax)) * s
    backward: dx/ds = (vq - v) / s * g   (the LSQ form, g = 1/sqrt(N*Qp))
              dx/dx = 1 inside the clamp range, 0 outside
    """

    @staticmethod
    def forward(ctx, x, s, qmin, qmax):
        v = torch.clamp(x / s, float(qmin), float(qmax))
        vq = torch.round(v)
        ctx.save_for_backward(v, vq, s)
        ctx.qmin = float(qmin)
        ctx.qmax = float(qmax)
        ctx.g = 1.0 / math.sqrt(max(1, x.numel()) * float(qmax))
        return vq * s

    @staticmethod
    def backward(ctx, grad_out):
        v, vq, s = ctx.saved_tensors
        mask = ((v >= ctx.qmin) & (v <= ctx.qmax)).to(grad_out.dtype)
        grad_x = grad_out * mask
        grad_s = grad_out * ctx.g * (vq - v) / s
        return grad_x, grad_s, None, None


class ActFakeQuantLSQ(nn.Module):
    """uint8-grid fake quantiser with a learnable per-tensor step size.

    ``scale`` is the LSQ step size s; the represented range is [0, 255*s].
    The forward pass is bit-identical to ONNX QuantizeLinear + DequantizeLinear
    with y_scale = s and y_zero_point = 0 (see the module docstring).
    """

    def __init__(self, scale: float, learnable: bool = True, qmin: int = QMIN,
                 qmax: int = QMAX, name: str = ""):
        super().__init__()
        self.qmin, self.qmax = int(qmin), int(qmax)
        self.learnable = bool(learnable)
        self.site_name = name
        # `enabled=False` makes the module the identity -- that is how the
        # calibration pass measures the PRE-quantisation activation statistics
        # without having to rebuild the model.
        self.enabled = True
        s = torch.tensor(float(scale), dtype=torch.float32)
        if self.learnable:
            self.scale = nn.Parameter(s)
        else:
            self.register_buffer("scale", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        s = self.scale.clamp(min=1e-12)
        if not torch.is_grad_enabled():
            x32 = x.float()
            v = torch.clamp(x32 / s, float(self.qmin), float(self.qmax))
            return (torch.round(v) * s).to(x.dtype)
        xq = _FakeQuantSTE.apply(x.float(), s, self.qmin, self.qmax)
        return xq.to(x.dtype)

    def extra_repr(self) -> str:
        return f"site={self.site_name}, scale={float(self.scale):.8f}, learnable={self.learnable}"


class ExportFakeQuant(nn.Module):
    """The ONNX-native twin of ActFakeQuantLSQ.

    It calls torch.fake_quantize_per_tensor_affine, whose opset-13 symbolic emits
    a real QuantizeLinear + DequantizeLinear pair -- MEASURED on torch 1.13.1 by
    manu/pipeline/npu/ptq/21_probe_onnx_fakequant.py:

        op histogram: {'Conv': 1, 'Relu': 1, 'Constant': 2,
                       'QuantizeLinear': 1, 'DequantizeLinear': 1}
        max|ONNXRuntime - PyTorch| = 0.000e+00

    The numerics are identical to ActFakeQuantLSQ (clamp(round(x/s), 0, 255) * s,
    zero-point 0), so the exported graph is bit-for-bit the function the network
    was trained against.
    """

    def __init__(self, scale: float, name: str = ""):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))
        self.site_name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fake_quantize_per_tensor_affine(x, self.scale, 0, 0, 255)

    def extra_repr(self) -> str:
        return f"site={self.site_name}, scale={float(self.scale):.8f} (ONNX Q/DQ)"


def attach_export_sites(model, scales: dict, sites=None):
    """Wrap the quantised sites with the ONNX-native fake quantiser."""
    sites = QAT_SITES if sites is None else sites
    handles = []
    for name, path, kind in sites:
        block = get_submodule(model, path)
        fq = ExportFakeQuant(scales[name], name=name)
        set_submodule(model, path, FakeQuantWrap(block, fq, name=name))
        handles.append((name, fq))
    return handles


def attach_export_output_fq(model, out_scale: float = OUT_SCALE):
    """Patch the head forward so the heatmap OUTPUT carries a Q/DQ pair too."""
    fq = ExportFakeQuant(out_scale, name=OUT_SITE[0])
    head = model.head
    orig = head.forward

    def patched(x):
        out = orig(x)
        out["heatmap"] = fq(out["heatmap"])
        return out

    head.forward = patched
    return fq


class FakeQuantWrap(nn.Module):
    """Wraps a block with the fake quantiser.

    The wrapped block stays under the attribute name ``block`` so that the QAT
    checkpoint can be exported back to the ORIGINAL parameter names by stripping
    ``.block.`` -- that is what lets the exported ONNX be loaded by the unchanged
    manu/pipeline/npu/hm_export_onnx.py machinery.
    """

    def __init__(self, block, fq, name: str = ""):
        super().__init__()
        self.block = block
        self.fq = fq
        self.site_name = name

    def forward(self, x):
        return self.fq(self.block(x))


class OutputFakeQuantWrap(nn.Module):
    """Reference wrapper (NOT used by the pipeline).

    Kept only as the documented alternative to the in-place head patch done by
    attach_output_fq(); the pipeline uses the patch because the patch keeps the
    checkpoint keys unchanged.
    """

    def __init__(self, model, fq, name: str = "heatmap_out"):
        super().__init__()
        self.model = model
        self.fq = fq
        self.site_name = name

    def forward(self, x):
        out = self.model(x)
        out["heatmap"] = self.fq(out["heatmap"])
        return out


# --------------------------------------------------------------------------------------
# Site attachment
# --------------------------------------------------------------------------------------
def get_submodule(root, path: str):
    """Resolve 'head.feat_conv.1' to the module object."""
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def set_submodule(root, path: str, value) -> None:
    """Assign into 'head.feat_conv.1' without touching the module tree otherwise."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def attach_sites(model, scales: dict, sites=None, out_scale: float = OUT_SCALE):
    """Replace the listed sub-modules with FakeQuantWrap(block, LSQ).

    ``scales`` maps the logical site name to the initial step size.  Returns the
    list of (site_name, fq_module) so the caller can read the learned scales back.
    """
    sites = QAT_SITES if sites is None else sites
    handles = []
    for name, path, kind in sites:
        block = get_submodule(model, path)
        fq = ActFakeQuantLSQ(scales[name], learnable=True, name=name)
        set_submodule(model, path, FakeQuantWrap(block, fq, name=name))
        handles.append((name, fq))
    return handles


def attach_output_fq(model, out_scale: float = OUT_SCALE):
    """Patch the head's forward so the post-sigmoid heatmap is fake-quantised.

    The head is patched in place instead of wrapping the detector, so that the
    parameter names (and therefore the checkpoint keys) stay EXACTLY the ones
    manu/pipeline/npu/hm_export_onnx.py expects.

    The deployed graph quantises the heatmap OUTPUT (the board reads it as int8
    with the toolkit's scale), so the training has to see the same grid --
    otherwise the loss is computed on a heatmap the board will never produce.
    """
    fq = ActFakeQuantLSQ(out_scale, learnable=False, name=OUT_SITE[0])
    head = model.head
    orig = head.forward

    def patched(x):
        out = orig(x)
        out["heatmap"] = fq(out["heatmap"])
        return out

    head.forward = patched
    return fq


def strip_wrappers(state_dict: dict) -> dict:
    """'.block.' -> '.' and drop the fake-quant parameters; returns the clean dict."""
    out = {}
    for k, v in state_dict.items():
        if ".fq." in k:
            continue
        out[k.replace(".block.", ".")] = v
    return out


def learned_scales(model) -> dict:
    """{site_name: learned scale} for every wrapped site."""
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "site_name") and hasattr(mod, "scale"):
            out[mod.site_name] = float(mod.scale.detach().abs().item())
    return out


# --------------------------------------------------------------------------------------
# Dataset: the OFFICIAL feature images + their YOLO labels
# --------------------------------------------------------------------------------------
def list_split(root: Path, split: str) -> list[tuple[str, str]]:
    """[(image_path, label_path)] for one split; the names match 1:1.

    The UAV train split is NOT uniformly .jpg -- it carries 41,050 .jpg AND
    1,958 .png (43,008 in total, which is the number the trial's DATA-GATE
    expects), so the glob has to take every image suffix or the QAT run would
    silently train on 95% of the Anti-UAV data.
    """
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    if not img_dir.is_dir():
        raise SystemExit(f"[FATAL] no image dir: {img_dir}")
    paths = [p for p in img_dir.iterdir()
             if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    paths.sort(key=lambda p: p.name)
    out = [(str(p), str(lbl_dir / (p.stem + ".txt"))) for p in paths]
    if not out:
        raise SystemExit(f"[FATAL] no images under {img_dir}")
    return out


def letterbox_rgb(img_rgb: np.ndarray, size: int = IMGSZ):
    """Letterbox an RGB uint8 image to size x size with pad 114.

    Returns (canvas, scale, pad_left, pad_top).  Mirrors
    manu/pipeline/npu/ptq/ptq_common.py:letterbox() exactly (same rounding), so the
    QAT training sees the same pixels the board sees.
    """
    import cv2

    h, w = img_rgb.shape[:2]
    r = min(size / h, size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (size - new_unpad[0]) / 2
    dh = (size - new_unpad[1]) / 2
    if new_unpad != (w, h):
        img_rgb = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    canvas = cv2.copyMakeBorder(img_rgb, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return canvas, r, left, top


def read_label_norm(label_path: str) -> np.ndarray:
    """(M,4) cx cy w h, normalised to the SOURCE image, from a YOLO label file."""
    rows = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                fl = line.split()
                if len(fl) >= 5:
                    rows.append([float(fl[1]), float(fl[2]), float(fl[3]), float(fl[4])])
    except OSError:
        pass
    if not rows:
        return np.zeros((0, 4), np.float32)
    return np.asarray(rows, np.float32)


def boxes_to_canvas_norm(boxes_src: np.ndarray, src_hw, scale: float,
                         pad_left: int, pad_top: int, size: int = IMGSZ) -> np.ndarray:
    """Map source-image normalised boxes onto the letterboxed canvas, still normalised."""
    if len(boxes_src) == 0:
        return np.zeros((0, 4), np.float32)
    h, w = src_hw
    cx = (boxes_src[:, 0] * w * scale + pad_left) / size
    cy = (boxes_src[:, 1] * h * scale + pad_top) / size
    bw = boxes_src[:, 2] * w * scale / size
    bh = boxes_src[:, 3] * h * scale / size
    return np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)


def load_frame_rgb(path: str) -> np.ndarray:
    """cv2.imread -> BGR -> reverse -> RGB, i.e. the model's channel order.

    This is the SAME byte order the board's stbi_load() produces.
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img[..., ::-1]
