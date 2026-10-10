#!/usr/bin/env python3
"""
fit_tradeoff.py -- walk the GMC fit accuracy/performance curve and print it as a table.

For each configuration it runs gmc_stream_ocl twice on the same sequence:
  * once with the frozen defaults (the reference arm)
  * once with the configuration under test
and reports BOTH the fit time and how far the transform chain moved, which is
the quantity the accuracy knobs actually trade against.

Two accuracy views, because they answer different questions:

  max|dtx|, max|dty|   The translation of each 2x3 affine, in ORIGINAL-resolution
                       pixels (the half-res fit is scaled by `downscale` on the
                       way out). This is the same unit as the warping error, so
                       a sub-pixel number means the warped edges land within a
                       sub-pixel box of each other.
  cos(theta)           Angle between the rotation parts, over the frames whose
                       |dtx|+|dty| exceeds that frame's 90th percentile of
                       baseline motion. Isolating the frames that actually
                       move keeps a near-zero rotation delta on static frames
                       from diluting the number.

A configuration is only worth its time if the transform deviation stays small
AND the fit time target is met. Both are printed together so neither can be
quoted without the other.
"""

import argparse
import math
import struct
import subprocess
import sys


def run(binary, seq_dir, limit, extra, mats_path):
    cmd = [binary, "--raw-root", seq_dir, "--sequence", "", "--limit", str(limit),
           "--fused", "cpu", "--mats-all", mats_path, "--timing"] + extra
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    fit = None
    for line in out.splitlines():
        if "SEQTIMING" in line and "fit=" in line:
            fit = float(line.split("fit=")[1].split()[0])
    if fit is None:
        sys.exit("no timing in output:\n" + out[-2000:])
    data = open(mats_path, "rb").read()
    n = len(data) // 4
    return fit, struct.unpack("<%df" % n, data), n // (21 * 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/gmc_stream_ocl")
    ap.add_argument("--seq", default="/home/manu/mnt/nfs/ocvparity/bmp2")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--tmp", default="/tmp/fit_tradeoff")
    ap.add_argument("--configs", default="",
                    help="semicolon-separated arg strings, e.g. "
                         "'--pyr-cache;--pyr-cache --fit-corners 120'")
    args = ap.parse_args()

    ref_fit, ref, ref_frames = run(args.bin, args.seq, args.limit, [], args.tmp + "_ref.bin")
    print(f"reference (frozen defaults): fit={ref_fit:.3f} ms  frames={ref_frames}")
    print()

    configs = [c.strip() for c in args.configs.split(";") if c.strip()]
    hdr = (f"{'config':<40} {'fit ms':>8} {'x':>6} {'max|dtx|':>9} {'max|dty|':>9} "
           f"{'mean|d|':>9} {'rot cos':>9} {'|ds/s|':>8}")
    print(hdr)
    print("-" * len(hdr))

    for cfg in configs:
        extra = cfg.split()
        fit, got, frames = run(args.bin, args.seq, args.limit, extra, args.tmp + "_t.bin")
        if frames != ref_frames:
            print(f"{cfg:<44} FRAME COUNT MISMATCH {frames} vs {ref_frames}")
            continue

        # Per-frame motion of the reference, to pick the frames that matter.
        per_frame = []
        for f in range(ref_frames):
            m = ref[f * 126:(f + 1) * 126]
            dx = sum(abs(m[6 * k + 2]) for k in range(21))
            dy = sum(abs(m[6 * k + 5]) for k in range(21))
            per_frame.append(dx + dy)

        thr = sorted(per_frame)[int(0.9 * len(per_frame))]  # 90th percentile

        max_dx = max_dy = 0.0
        sum_d = 0.0
        cnt = 0
        cos_num = cos_den = 0.0
        sum_scale = 0.0
        n_scale = 0
        ref_fail = got_fail = 0
        for f in range(ref_frames):
            a = ref[f * 126:(f + 1) * 126]
            b = got[f * 126:(f + 1) * 126]
            for k in range(21):
                r = a[6 * k:6 * k + 6]
                c = b[6 * k:6 * k + 6]
                # identity == fit failed (fewer than 6 tracked survivors)
                if abs(r[0] - 1) < 1e-9 and abs(r[4] - 1) < 1e-9 and \
                   abs(r[1]) < 1e-9 and abs(r[3]) < 1e-9 and \
                   abs(r[2]) < 1e-9 and abs(r[5]) < 1e-9:
                    ref_fail += 1
                if abs(c[0] - 1) < 1e-9 and abs(c[4] - 1) < 1e-9 and \
                   abs(c[1]) < 1e-9 and abs(c[3]) < 1e-9 and \
                   abs(c[2]) < 1e-9 and abs(c[5]) < 1e-9:
                    got_fail += 1
                dtx, dty = abs(c[2] - r[2]), abs(c[5] - r[5])
                max_dx, max_dy = max(max_dx, dtx), max(max_dy, dty)
                sum_d += dtx + dty
                cnt += 1
                if per_frame[f] >= thr:
                    # Layout is [a b tx / c d ty] and estimateAffinePartial2D
                    # returns a SIMILARITY: a = s*cos(th), c = s*sin(th). So
                    # (a, c) is the rotation only up to the scale; comparing
                    # them raw yields s^2, not a cosine (and exceeds 1 whenever
                    # the fit reports any zoom). Normalise both sides.
                    ra = math.hypot(r[0], r[3])
                    rb = math.hypot(c[0], c[3])
                    if ra > 1e-12 and rb > 1e-12:
                        cos_num += (r[0] * c[0] + r[3] * c[3]) / (ra * rb)
                        cos_den += 1
                    sum_scale += abs(rb / ra - 1.0)
                    n_scale += 1

        mean_d = sum_d / max(1, cnt)
        cosv = (cos_num / cos_den) if cos_den else float("nan")
        dscale = (sum_scale / n_scale) if n_scale else float("nan")
        speedup = ref_fit / fit if fit else float("nan")
        print(f"{cfg:<40} {fit:8.3f} {speedup:5.2f}x {max_dx:9.4f} {max_dy:9.4f} "
              f"{mean_d:9.5f} {cosv:9.7f} {dscale:8.5f}")


if __name__ == "__main__":
    main()