#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
03_make_fast_eval_manifest.py -- build the fast-eval manifests and stage the BMPs.

Three products
--------------
  1. 001324_sub2000.json    2,000 continuous frames of the blind-test death spot
                            (frames 4000..5999 of VIDEO00007_19700101_001324).
                            001324 is 11,810 frames long and is where the delivery
                            recall is decided, so the core engagement window is
                            what has to be fast.
  2. golden_pool_sub1200.json
                            ~1,200 frames spread over the 7 Track-2 golden-pool
                            sequences (Track2 = 17,187 frames / gt 16,638).
  3. the FULL Tier-3 staging set (optional, --stage-full)
                            every official-semantic frame of the 8 gate sequences
                            (28,997 frames = Track2 17,187 + Track3 11,810).
                            Track1 (Anti-UAV, 31,613 frames) already exists on the
                            board as pcfeat_bmp/images -- only the T2/T3 part has
                            to be staged.

Naming
------
The board's driver (`run_npu_pcfeat_board_generic.sh`) reads `<FEAT>/<seq>__<id>.bmp`
and the GT is `<seq>__<id>.txt`, so the id in the BMP name must be the SAME id as in
the label file. The 龙泉山 label ids start at 000000 (unlike Anti-UAV, which starts
at 000003), so the ids are taken from the feature filenames verbatim -- never
re-indexed -- because a one-frame offset silently turns every detection into a miss.

Why BMP
------
The board reads images with stb_image, the PC reads them with OpenCV. If the file
is a JPEG the two decoders are not guaranteed to produce identical pixels, so the
"quantisation loss" would be contaminated by decoder differences. A BMP is lossless
and byte-exact on both sides. Every staged file is therefore a BMP.

Usage
-----
    python 03_make_fast_eval_manifest.py --out /mnt/data/siping/ptq_work/fast
    python 03_make_fast_eval_manifest.py --stage-full
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

FAST_001324_FRAMES = 2000
FAST_001324_START = 4000          # core engagement window
FAST_GOLDEN_TOTAL = 1200


def stage_sequence(seq: str, frame_ids: list[int], bmp_root: Path,
                   src_root: Path, workers: int = 32) -> list[dict]:
    """Convert the feature JPGs of one sequence to BMP and return the manifest rows."""
    import multiprocessing as mp

    out_dir = bmp_root / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    todo = []
    for fid in frame_ids:
        src = C.official_feature_path(src_root, seq, fid)
        if not src.is_file():
            raise SystemExit(f"[FATAL] missing feature image: {src}")
        dst = out_dir / f"{seq}__{fid:06d}.bmp"
        rows.append({
            "seq": seq,
            "frame_id": fid,
            "bmp": str(dst),
            "feature_src": str(src),
            "gt": str(C.gt_path(seq, fid)),
        })
        if not dst.is_file():
            todo.append((str(src), str(dst)))

    if todo:
        args = [(s, d) for s, d in todo]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(workers, len(args))) as pool:
            pool.map(_convert_one, args, chunksize=32)

    for r in rows:
        if not Path(r["bmp"]).is_file():
            raise SystemExit(f"[FATAL] staged file missing: {r['bmp']}")
    return rows


def _convert_one(pair):
    src, dst = pair
    C.write_feature_bmp(src, dst)
    return dst


def golden_frames(src_root: Path, total: int) -> dict[str, list[int]]:
    """Deterministic, sequence-stratified subsample of the golden pool."""
    per_seq = {}
    for seq in C.TRACK2_GOLDEN:
        ids = C.seq_frames_from_official(src_root, seq)
        if not ids:
            raise SystemExit(f"[FATAL] no official features for golden sequence {seq}")
        per_seq[seq] = ids
    # equal share per sequence, then a deterministic stride inside each sequence
    quota = max(1, total // len(per_seq))
    out = {}
    for seq, ids in per_seq.items():
        n = min(quota, len(ids))
        step = max(1, len(ids) // n)
        picked = [ids[i] for i in range(0, len(ids), step)][:n]
        out[seq] = sorted(picked)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.WORK_ROOT / "fast"))
    ap.add_argument("--bmp-root", default=str(C.BMP_ROOT))
    ap.add_argument("--src-root", default=str(C.FEATURE_OFFICIAL_VAL),
                    help="official-semantic feature root (never the FPV one)")
    ap.add_argument("--stage-full", action="store_true",
                    help="also stage every frame of the 8 gate sequences (Tier 3)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-stage", action="store_true", help="manifest only, no BMPs")
    args = ap.parse_args()

    src_root = C.require(Path(args.src_root), "official feature root")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    bmp_root = Path(args.bmp_root)

    # ---------------- manifest 1: 001324 fast window ----------------
    blind_seq = C.TRACK3_BLIND[0]
    blind_ids = C.seq_frames_from_official(src_root, blind_seq)
    if not blind_ids:
        raise SystemExit(f"[FATAL] no official features for {blind_seq}")
    lo = FAST_001324_START
    hi = lo + FAST_001324_FRAMES
    fast_blind = [i for i in blind_ids if lo <= i < hi]
    if len(fast_blind) < FAST_001324_FRAMES:
        # fall back to the last FAST_001324_FRAMES frames of the sequence
        fast_blind = blind_ids[-FAST_001324_FRAMES:]
        print(f"[warn] window {lo}..{hi} short ({len(fast_blind)} frames) -- "
              f"falling back to the tail of the sequence")
    rows_blind = [] if args.no_stage else stage_sequence(
        blind_seq, fast_blind, bmp_root, src_root, args.workers)
    if not rows_blind:
        rows_blind = [{"seq": blind_seq, "frame_id": i,
                       "bmp": str(bmp_root / blind_seq / f"{blind_seq}__{i:06d}.bmp"),
                       "feature_src": str(C.official_feature_path(src_root, blind_seq, i)),
                       "gt": str(C.gt_path(blind_seq, i))} for i in fast_blind]
    C.write_json(out_root / "001324_sub2000.json", {
        "name": "001324_sub2000",
        "seq": blind_seq,
        "frame_count": len(rows_blind),
        "frame_range": [min(fast_blind), max(fast_blind)],
        "src_root": str(src_root),
        "items": rows_blind,
    })
    print(f"[001324_sub2000] {len(rows_blind)} frames "
          f"(ids {min(fast_blind)}..{max(fast_blind)}) -> {out_root / '001324_sub2000.json'}")

    # ---------------- manifest 2: golden pool fast subset ----------------
    per_seq = golden_frames(src_root, FAST_GOLDEN_TOTAL)
    rows_golden = []
    for seq, ids in per_seq.items():
        rows_golden.extend([] if args.no_stage else stage_sequence(
            seq, ids, bmp_root, src_root, args.workers))
    if not rows_golden:
        for seq, ids in per_seq.items():
            rows_golden.extend({"seq": seq, "frame_id": i,
                                "bmp": str(bmp_root / seq / f"{seq}__{i:06d}.bmp"),
                                "feature_src": str(C.official_feature_path(src_root, seq, i)),
                                "gt": str(C.gt_path(seq, i))} for i in ids)
    C.write_json(out_root / "golden_pool_sub1200.json", {
        "name": "golden_pool_sub1200",
        "sequences": {k: len(v) for k, v in per_seq.items()},
        "frame_count": len(rows_golden),
        "src_root": str(src_root),
        "items": rows_golden,
    })
    print(f"[golden_pool_sub1200] {len(rows_golden)} frames -> "
          f"{out_root / 'golden_pool_sub1200.json'}")

    # ---------------- optional: full Tier-3 staging ----------------
    if args.stage_full:
        all_rows = []
        for seq in list(C.TRACK2_GOLDEN) + list(C.TRACK3_BLIND):
            ids = C.seq_frames_from_official(src_root, seq)
            all_rows.extend(stage_sequence(seq, ids, bmp_root, src_root, args.workers))
            print(f"[stage-full] {seq}: {len(ids)} frames staged")
        C.write_json(out_root / "tier3_full_official.json", {
            "name": "tier3_full_official",
            "sequences": {s: sum(1 for r in all_rows if r["seq"] == s)
                          for s in list(C.TRACK2_GOLDEN) + list(C.TRACK3_BLIND)},
            "frame_count": len(all_rows),
            "src_root": str(src_root),
            "items": all_rows,
        })
        print(f"[stage-full] total {len(all_rows)} frames -> "
              f"{out_root / 'tier3_full_official.json'}")

    # ---------------- summary ----------------
    print("\n[summary]")
    for p in sorted(out_root.glob("*.json")):
        d = json.loads(p.read_text())
        print(f"  {p.name:32s} {d.get('frame_count')} frames")


if __name__ == "__main__":
    main()
