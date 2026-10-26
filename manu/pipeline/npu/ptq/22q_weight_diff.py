#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
22q_weight_diff.py -- which tensors did the QAT training actually move?

The float ONNX with the QAT weights already explodes (4.1x the detections of
the base weights on tier1), and the graph structure is identical, so the
question is WHICH weights moved and by how much.  A bias shift in the heatmap
head's last convolution moves the logit and therefore the sigmoid scale, which
is exactly the symptom (an over-confident heatmap: 35.2 px>0.5/frame instead
of 10.4).

It also checks the thing that would be a REAL export bug: whether any tensor
in the QAT checkpoint is bit-identical to the base model (which would mean the
QAT training silently failed to write it) and whether the shapes match.

Usage (server, conda env `uav`):
    python manu/pipeline/npu/ptq/22q_weight_diff.py \
        --ckpt /mnt/data/siping/ptq_work/qat/trial_0080/qat_epoch3.pt \
        --base /tmp/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt \
        --out /mnt/data/siping/ptq_work/qat/trial_0080/report/weight_diff.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    qat = ck["model"]
    base_sd = torch.load(args.base, map_location="cpu", weights_only=False)
    base = base_sd["model"] if isinstance(base_sd, dict) and "model" in base_sd else base_sd

    rows = []
    for k, v in qat.items():
        b = base.get(k)
        if b is None:
            rows.append({"key": k, "status": "not_in_base", "shape": list(v.shape)})
            continue
        if tuple(b.shape) != tuple(v.shape):
            rows.append({"key": k, "status": "shape_mismatch",
                         "shape_qat": list(v.shape), "shape_base": list(b.shape)})
            continue
        d = (v.float() - b.float())
        rel = float(d.norm() / max(1e-12, b.float().norm()))
        rows.append({"key": k, "status": "ok", "shape": list(v.shape),
                     "rel_l2": rel, "abs_max_delta": float(d.abs().max()),
                     "base_mean": float(b.float().mean()),
                     "qat_mean": float(v.float().mean()),
                     "base_absmax": float(b.float().abs().max()),
                     "qat_absmax": float(v.float().abs().max())})

    ok = [r for r in rows if r["status"] == "ok"]
    ok.sort(key=lambda r: -r["rel_l2"])
    bad = [r for r in rows if r["status"] != "ok"]

    # the tensors that can move the heatmap's absolute scale
    head_keys = [r for r in ok if any(s in r["key"] for s in
                                      ("head.heatmap", "head.offset", "head.feat_conv"))]

    res = {
        "ckpt": args.ckpt, "base": args.base,
        "n_qat_tensors": len(qat), "n_base_tensors": len(base),
        "n_compared": len(ok), "n_problem": len(bad), "problems": bad,
        "identical_tensors": [r["key"] for r in ok if r["rel_l2"] == 0.0],
        "n_identical": sum(1 for r in ok if r["rel_l2"] == 0.0),
        "top_movers": ok[:25],
        "head_movers": sorted(head_keys, key=lambda r: -r["rel_l2"])[:15],
        "mean_rel_l2": (sum(r["rel_l2"] for r in ok) / len(ok)) if ok else None,
        "max_rel_l2": ok[0]["rel_l2"] if ok else None,
        "qat_scales": ck.get("qat_scales"),
        "calib_scales": ck.get("calib_scales"),
        "train_loss": ck.get("metrics", {}).get("train_loss"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))

    print(f"[wdiff] tensors: qat={len(qat)} base={len(base)} compared={len(ok)} "
          f"problems={len(bad)} identical={res['n_identical']}")
    if bad:
        for b in bad[:10]:
            print(f"[wdiff]   PROBLEM {b}")
    print(f"[wdiff] mean rel_l2 = {res['mean_rel_l2']:.6f}   max = {res['max_rel_l2']:.6f}")
    print(f"[wdiff] top movers:")
    for r in res["top_movers"][:12]:
        print(f"[wdiff]   {r['key']:<52} rel_l2={r['rel_l2']:.6f} "
              f"absmax {r['base_absmax']:.4f}->{r['qat_absmax']:.4f} "
              f"mean {r['base_mean']:+.5f}->{r['qat_mean']:+.5f}")
    print(f"[wdiff] head movers:")
    for r in res["head_movers"][:12]:
        print(f"[wdiff]   {r['key']:<52} rel_l2={r['rel_l2']:.6f} "
              f"absmax {r['base_absmax']:.4f}->{r['qat_absmax']:.4f}")
    print(f"[wdiff] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
