#!/usr/bin/env python3
"""Acceptance check for the SERVICE pipeline under a 25 fps camera.

The engineering requirement this script verifies:

  * the GMC feature stream must be COMPLETE -- a feature for EVERY frame,
    bit-identical to the reference implementation;
  * the inference stage is allowed to SKIP frames -- a skipped frame costs one
    missed detection and nothing else;
  * a frame lost at the CAMERA queue would break the GMC registration chain,
    so that counter must be 0.

Inputs:
  --feature-md5   the per-frame feature fingerprint written by the run
  --feature-md5-ref   the reference fingerprint of the same sequence
  --tsv           the detection TSV of the run
  --tsv-ref       the certified detection TSV of the same sequence
  --timing        the timing.txt of the run

Usage:
  python 44_svc_acceptance.py --feature-md5 A --feature-md5-ref B \
      --tsv C --tsv-ref D --timing E
"""
from __future__ import annotations

import argparse
from pathlib import Path


def read_feature_md5(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        try:
            out[int(parts[1])] = parts[2]
        except ValueError:
            continue
    return out


def read_tsv(path: Path) -> dict[int, str]:
    """frame id -> the whole row (the detections), verbatim."""
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        stem = line.split("\t", 1)[0]
        # The scorer's stem_to_frame_index() takes the digits after the last "__".
        if "__" not in stem:
            continue
        tail = stem.rsplit("__", 1)[1]
        try:
            out[int(tail)] = line
        except ValueError:
            continue
    return out


def read_timing(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and not parts[0].startswith("#"):
            try:
                out[parts[0]] = float(parts[2])
                continue
            except ValueError:
                pass
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            try:
                out[key.strip()] = float(value.strip())
            except ValueError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-md5", required=True)
    ap.add_argument("--feature-md5-ref", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--tsv-ref", required=True)
    ap.add_argument("--timing", required=True)
    args = ap.parse_args()

    feat = read_feature_md5(Path(args.feature_md5))
    feat_ref = read_feature_md5(Path(args.feature_md5_ref))
    tsv = read_tsv(Path(args.tsv))
    tsv_ref = read_tsv(Path(args.tsv_ref))
    tm = read_timing(Path(args.timing))

    ok = True
    print("=" * 74)
    print("SERVICE ACCEPTANCE -- 25 fps camera, GMC must not lose a frame")
    print("=" * 74)

    # ---- 1. the feature stream must be COMPLETE and bit-identical ---------
    print("\n[1] the GMC feature stream")
    print(f"    reference frames : {len(feat_ref)}")
    print(f"    produced frames  : {len(feat)}")
    missing = sorted(set(feat_ref) - set(feat))
    extra = sorted(set(feat) - set(feat_ref))
    same = sum(1 for k, v in feat.items() if feat_ref.get(k) == v)
    diff = [k for k, v in feat.items() if k in feat_ref and feat_ref[k] != v]
    print(f"    missing (never computed) : {len(missing)}")
    print(f"    extra                   : {len(extra)}")
    print(f"    identical               : {same}")
    print(f"    DIFFERENT               : {len(diff)}")
    if missing:
        print(f"    -> the GMC did NOT produce a feature for every frame: "
              f"{len(missing)} frame(s) missing, first = {missing[:10]}")
        ok = False
    if diff:
        print(f"    -> the produced features are NOT bit-identical: first = {diff[:10]}")
        ok = False
    if not missing and not diff:
        print("    -> OK: a bit-identical feature for EVERY frame.")

    # ---- 2. the detections that WERE produced must be bit-identical -------
    print("\n[2] the detections (the inference may skip frames)")
    common = sorted(set(tsv) & set(tsv_ref))
    rows_same = sum(1 for k in common if tsv[k] == tsv_ref[k])
    rows_diff = [k for k in common if tsv[k] != tsv_ref[k]]
    print(f"    reference frames with detections : {len(tsv_ref)}")
    print(f"    produced frames with detections  : {len(tsv)}")
    print(f"    processed by both                : {len(common)}")
    print(f"    identical rows                   : {rows_same}")
    print(f"    DIFFERENT rows                   : {len(rows_diff)}")
    if rows_diff:
        print(f"    -> the processed frames are NOT bit-identical: first = {rows_diff[:10]}")
        ok = False
    else:
        print("    -> OK: every frame the inference DID process has the "
              "bit-identical detection row.")

    # ---- 3. the counters: which drop point fired --------------------------
    print("\n[3] the three drop points")
    cam = tm.get("dropped", 0.0)
    feat_drop = tm.get("features_dropped", 0.0)
    det_drop = tm.get("detections_dropped", 0.0)
    blows = tm.get("deadline_blows", 0.0)
    pushed = tm.get("pushed", 0.0)
    print(f"    camera-queue drops      : {int(cam)}   "
          f"(MUST be 0: this breaks the GMC registration)")
    print(f"    features skipped (NPU)  : {int(feat_drop)}   (benign: a missed detection)")
    print(f"    detections skipped      : {int(det_drop)}   (benign)")
    print(f"    GMC fits degraded       : {int(blows)}   (the watchdog fired)")
    if cam > 0:
        print("    -> FAIL: the GMC did not see every frame.")
        ok = False
    else:
        print("    -> OK: the GMC saw every frame.")

    # ---- 4. the GMC's own capacity against the 25 fps budget -------------
    print("\n[4] the GMC service against the 40 ms/frame budget")
    gmc_total = tm.get("gmc_total", 0.0)
    workers = tm.get("gmc_workers", 2.0)
    per_frame = gmc_total / workers if workers else 0.0
    print(f"    gmc_total {gmc_total:.3f} ms over {int(workers)} engine(s) "
          f"= {per_frame:.3f} ms per frame of the 40 ms budget")
    print(f"    margin: {40.0 - per_frame:.3f} ms "
          f"({100.0 * (40.0 - per_frame) / 40.0:.1f}%)   "
          f"capacity {1000.0 / per_frame if per_frame else 0:.1f} fps")
    if per_frame > 40.0:
        print("    -> FAIL: the GMC service cannot keep up with a 25 fps source.")
        ok = False
    else:
        print("    -> OK: the GMC service can keep up with a 25 fps source.")

    print("\n" + "=" * 74)
    print("ACCEPTANCE: " + ("PASS" if ok else "FAIL"))
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
