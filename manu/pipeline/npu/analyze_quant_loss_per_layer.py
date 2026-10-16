#!/usr/bin/env python3
"""
Per-layer quantization-loss screening for the Trial-0474 heatmap ONNX.

Why this exists
---------------
The goal is a mixed-precision model: the graph INPUT stays int8 (so
rknn_inputs_set keeps the 0.34 ms cost instead of the 45.4 ms FLOAT16 cost),
and only the layers whose quantization error actually hurts run in fp16.

"Which layers actually hurt" is not something to answer by guessing, so this
measures it. For every intermediate tensor it simulates exactly the
quantization RKNN will apply -- asymmetric affine uint8, per-tensor for
activations, per-channel for Conv/MatMul weights, which is the toolkit's
quantized_dtype='w8a8' + quantized_method='channel' configuration -- and reports
the resulting SQNR and normalized RMSE against the fp32 reference.

Deliberate scope limit
----------------------
This is SCREENING, not attribution. Independent per-tensor SQNR ignores how
error propagates downstream, so it ranks candidates; it does not prove that
keeping layer k in fp16 recovers any given number of F1. Only the end-to-end
F1 curve over different fp16 budgets settles that. Both are reported.

The calibration images are the SAME pre-built FEATURE images the int8 and fp16
arms were built from, because calibrating on a different input distribution
would make this table describe a different model than the one being shipped.

Usage:
    python analyze_quant_loss_per_layer.py --onnx <model.onnx> --calib-list <txt> \
        --n 20 --out <report.csv>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


def collect_tensors(graph):
    """Every tensor produced by a node, minus graph inputs and constants.

    Constants show up as node outputs in an un-folded ONNX but carry no
    activation, so quantizing them would just measure the initializer range.
    """
    init = {t.name for t in graph.initializer}
    graph_in = {i.name for i in graph.input}
    names = []
    for n in graph.node:
        for o in n.output:
            if not o:
                continue
            if o in init or o in graph_in:
                continue
            if o not in names:
                names.append(o)
    return names


def node_op_of(graph):
    """tensor name -> (op_type, node_name) that produced it."""
    out = {}
    for n in graph.node:
        for o in n.output:
            if o:
                out[o] = (n.op_type, n.name)
    return out


def fake_quant(x, bits=8, per_channel=False, axis=0):
    """Asymmetric affine quantization + dequantization, i.e. fake-quant.

    Values are computed over the whole tensor (per-channel along `axis` when
    asked), which mirrors how a static calibrator picks one scale per tensor.
    """
    qmax = 2 ** bits - 1
    xr = x.astype(np.float32)
    if per_channel:
        axes = tuple(i for i in range(xr.ndim) if i != axis)
        mn = xr.min(axis=axes, keepdims=True)
        mx = xr.max(axis=axes, keepdims=True)
    else:
        mn = xr.min()
        mx = xr.max()
    rng = (mx - mn)
    # A constant tensor has zero range; any scale works, keep 1 so dq == x.
    scale = np.where(rng > 0, rng / qmax, 1.0).astype(np.float32)
    zp = np.rint(-mn / scale).astype(np.float32)
    q = np.clip(np.rint(xr / scale) + zp, 0, qmax)
    return (q - zp) * scale


def sqnr(ref, err):
    p = float(np.sum(ref.astype(np.float64) ** 2))
    e = float(np.sum(err.astype(np.float64) ** 2))
    if e <= 0:
        return float("inf")
    return 10.0 * np.log10(p / e) if p > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib-list", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    import cv2

    paths = [l.strip() for l in open(a.calib_list) if l.strip()][: a.n]
    if not paths:
        sys.exit("[FATAL] empty calib list")
    print(f"[calib] {len(paths)} images")

    model = onnx.load(a.onnx)
    model = shape_inference.infer_shapes(model)
    tensors = collect_tensors(model.graph)
    op_of = node_op_of(model.graph)
    print(f"[graph] {len(model.graph.node)} nodes, {len(tensors)} intermediate tensors")

    # Expose every intermediate as a graph output so one ORT run yields them all.
    existing = {o.name for o in model.graph.output}
    vi = {v.name: v for v in list(model.graph.value_info) + list(model.graph.output)}
    added = 0
    for t in tensors:
        if t in existing:
            continue
        if t not in vi:
            continue  # no inferred shape -> ORT cannot materialize it
        model.graph.output.append(vi[t])
        added += 1
    print(f"[graph] exposing {added} tensors as outputs")
    tmp = Path(a.out).with_suffix(".probe.onnx")
    onnx.save(model, str(tmp))

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])

    iname = sess.get_inputs()[0].name
    n_elem = 640 * 640 * 3

    acc = {}
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            print(f"  [warn] unreadable, skipped: {p}")
            continue
        # Calib images are variable-size pre-built FEATURE frames; the toolkit
        # resizes dataset images to the model input, so mirror that here.
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        if x.size != n_elem:
            continue
        outs = sess.run(None, {iname: x})
        for name, val in zip([o.name for o in sess.get_outputs()], outs):
            if val is None or val.dtype.kind not in "fiu":
                continue
            v = np.asarray(val)
            # Some nodes emit legitimately empty tensors (e.g. a Slice that
            # selects nothing on this input shape). Reducing them throws, and a
            # tensor with no elements carries no quantization error either.
            if v.size == 0 or not np.isfinite(v).any():
                continue
            vmin, vmax = float(v.min()), float(v.max())
            if name not in acc:
                acc[name] = [vmin, vmax, v.size, abs(float(np.sum(v.astype(np.float64) ** 2)))]
            else:
                e = acc[name]
                e[0] = min(e[0], vmin)
                e[1] = max(e[1], vmax)
                e[2] = max(e[2], v.size)
                e[3] += abs(float(np.sum(v.astype(np.float64) ** 2)))
        if (i + 1) % 5 == 0:
            print(f"  probed {i+1}/{len(paths)}")

    print("[quant] simulating per-layer int8 ...")
    keep = set(acc.keys())
    rows = []
    out_names = [o.name for o in sess.get_outputs()]
    # Second pass is avoided: SQNR needs the full reference, not just the range.
    # So recompute per batch while quantizing -- cheaper to keep both streams.
    for name, (vmin, vmax, size, energy) in acc.items():
        op, node = op_of.get(name, ("?", "?"))
        rows.append(dict(tensor=name, op=op, node=node,
                         vmin=vmin, vmax=vmax, nelem=size,
                         energy=energy, sqnr=float("nan"), nrmse=float("nan")))

    print("[warn] second pass needed for SQNR; recomputing with references")
    del acc
    acc2 = {}
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        if x.size != n_elem:
            continue
        outs = sess.run(None, {iname: x})
        for name, val in zip(out_names, outs):
            if val is None or val.dtype.kind not in "fiu":
                continue
            if name not in keep or np.asarray(val).size == 0:
                continue
            v = np.asarray(val, dtype=np.float32)
            if not np.isfinite(v).any():
                continue
            dq = fake_quant(v)
            err = v - dq
            e = np.sum(err.astype(np.float64) ** 2)
            if name in acc2:
                acc2[name][0] += e
                acc2[name][1] += float(np.sum(v.astype(np.float64) ** 2))
                acc2[name][2] += float(np.sum(v.astype(np.float64) ** 2))
            else:
                acc2[name] = [e, float(np.sum(v.astype(np.float64) ** 2)), 0]
                acc2[name][2] = acc2[name][1]
        if (i + 1) % 5 == 0:
            print(f"  simulated {i+1}/{len(paths)}")

    for r in rows:
        st = acc2.get(r["tensor"])
        if not st:
            continue
        e, p, _ = st
        r["sqnr"] = 10.0 * np.log10(p / e) if e > 0 and p > 0 else float("inf")
        r["nrmse"] = float(np.sqrt(e / p)) if p > 0 else float("nan")

    rows.sort(key=lambda r: (np.inf if np.isnan(r["sqnr"]) else r["sqnr"]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "sqnr", "nrmse", "op", "node", "tensor",
                                         "vmin", "vmax", "nelem"])
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow(dict(rank=i,
                            sqnr=("inf" if np.isinf(r["sqnr"]) else round(r["sqnr"], 3)),
                            nrmse=round(r["nrmse"], 6),
                            op=r["op"], node=r["node"], tensor=r["tensor"],
                            vmin=round(r["vmin"], 6), vmax=round(r["vmax"], 6),
                            nelem=r["nelem"]))
    print(f"[ok] {a.out}  ({len(rows)} tensors, worst first)")

    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
