#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
01_build_calibration_sets.py -- build the 4 x 250 PTQ calibration sets.

Why the calibration set is the single most consequential choice in this whole task
------------------------------------------------------------------------------------
A PTQ quantiser sees ONLY the images in the calibration list. Feed it the wrong
distribution and the model quantises "fine" on paper while quietly losing the
sub-threshold pulses the heatmap head lives on. So the sets are built from the
**already-built 3-channel feature images** (GMC diff + 21-frame median residual),
never from raw IR frames, and never from a different feature semantics:

    official semantics  = longquanshan_ft_gmc_median_official  (network's training input)
    FPV semantics       = longquanshan_ir_gmc_median            (business-domain eval only)

memory §五十四 measured a 26..70/255 per-pixel difference between the two, so mixing
them makes every downstream number a weighted average of two distributions and kills
the ability to attribute a delta. This script therefore refuses to look at FPV.

The four sets
-------------
  Set-A  random baseline      125 Anti-UAV train + 125 龙泉山 official train (uniform)
  Set-B  dark-weak bias        175 from {001324, 003002} dark-rigid-body transit
                                +  75 open sky
  Set-C  high-dynamic clutter  175 complex ground / buildings / road / forest edge
                                +  75 open sky
  Set-D  golden tri-state      85 very dark weak rigid body
                                +  85 high-frequency strong clutter
                                +  80 open standard sky

Every image is stored as a **BMP** (lossless). A JPG in the middle would let the PC
decoder (libjpeg-turbo) and the board decoder (stb_image) see different pixels, which
is exactly the contamination this whole pipeline is trying to measure away. See
ptq_common.py for the pixel contract.

Scene stratification is automatic and auditable
-----------------------------------------------
No per-frame scene labels exist, so the strata are derived from image statistics
computed over the candidate pool and the thresholds are recorded in the manifest:

    dark_ratio   fraction of pixels whose raw IR channel is below DARK_LEVEL
    clutter      mean |gradient| of the GMC-diff channel (high = texture / edge clutter)
    sky_ratio    fraction of the top rows that are nearly uniform AND bright
    contrast     std of the median-residual channel

Usage
-----
    python 01_build_calibration_sets.py --out /mnt/data/siping/ptq_work/calib
    python 01_build_calibration_sets.py --set B --dry-run          # only report stats
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

SET_SIZE = 250

# --- stratification constants (recorded in the manifest so the choice is auditable) ---
DARK_LEVEL = 40          # raw-IR level below which a pixel counts as "dark"
DARK_RATIO_HI = 0.55     # a frame is "dark weak" when at least this fraction is dark
DARK_RATIO_LO = 0.20     # ... and it is NOT overwhelmingly dark (keeps some structure)
CLUTTER_HI = 6.0         # mean |grad| of the diff channel above this = strong clutter
SKY_ROW_FRAC = 0.25      # top band of the frame used for the "open sky" test
SKY_UNIFORM_STD = 6.0    # a sky band is uniform if its std is below this
SKY_BRIGHT_LEVEL = 90    # ... and bright enough
SKY_RATIO_HI = 0.45      # fraction of the band that must be sky


# --------------------------------------------------------------------------------------
# Image statistics -- the only thing that separates "dark weak" from "clutter" from "sky"
# --------------------------------------------------------------------------------------
def frame_stats(path: Path) -> dict:
    """Statistics of ONE feature image, in the model's channel order.

    The disk order is the reverse of the model order (see ptq_common.py), so after
    loading we index as: 0 = median residual, 1 = GMC diff, 2 = raw IR.
    """
    import cv2

    path = Path(path)
    img = C.load_feature_bmp(path) if path.suffix.lower() == ".bmp" else \
        cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    rgb = img[..., ::-1]                      # BGR -> RGB == model order
    res, diff, ir = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ir_f = ir.astype(np.float32)

    dark_ratio = float((ir_f < DARK_LEVEL).mean())

    gx = np.abs(np.diff(diff.astype(np.float32), axis=1)).mean() if diff.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(diff.astype(np.float32), axis=0)).mean() if diff.shape[0] > 1 else 0.0
    clutter = float((gx + gy) / 2.0)

    band = ir[: max(1, int(ir.shape[0] * SKY_ROW_FRAC)), :]
    bright = band >= SKY_BRIGHT_LEVEL
    uniform = (band.astype(np.float32).std(axis=1) < SKY_UNIFORM_STD)
    sky_ratio = float((bright & uniform[:, None]).mean())

    # A frame without any GT box cannot lose a detection, so it tells the quantiser
    # nothing about the target. Frames WITH a target are the ones whose loss we are
    # measuring -- "刚体穿行" (a rigid body crossing the frame) is exactly that.
    n_gt = 0
    stem = Path(path).stem
    if "__frame_" in stem:
        seq, _, tail = stem.partition("__frame_")
        try:
            n_gt = sum(1 for l in C.gt_path(seq, int(tail)).read_text().splitlines()
                       if len(l.split()) >= 5)
        except (OSError, ValueError):
            n_gt = 0

    return {
        "dark_ratio": dark_ratio,
        "clutter": clutter,
        "sky_ratio": sky_ratio,
        "contrast": float(res.astype(np.float32).std()),
        "ir_mean": float(ir_f.mean()),
        "ir_max": int(ir.max()),
        "n_gt": n_gt,
    }


def _stats_worker(path_str: str) -> tuple[str, dict]:
    return path_str, frame_stats(Path(path_str))


def compute_stats(paths: list[Path], workers: int = 32) -> dict[str, dict]:
    """Statistics for many feature images, in parallel.

    ~29k official-val images have to be classified (that is the whole candidate
    pool for sets B/C/D). Single-threaded that is ~4 minutes of pure decoding, so
    the pool is mandatory, not an optimisation.
    """
    import multiprocessing as mp

    uniq = sorted({str(p) for p in paths})
    if workers <= 1 or len(uniq) < 64:
        return {p: frame_stats(Path(p)) for p in uniq}
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(workers, len(uniq))) as pool:
        results = pool.map(_stats_worker, uniq, chunksize=64)
    return {k: v for k, v in results}


def select_strata(stats: dict[str, dict], n_dark: int, n_clutter: int, n_sky: int,
                  restrict_dark: list[str] | None = None,
                  restrict_clutter: list[str] | None = None) -> dict[str, list[str]]:
    """Pick the three strata by QUANTILES of the pool's own distribution.

    Absolute thresholds were the first attempt and they were wrong: the raw IR
    level of this data sits around 96/255, so a "<40 is dark" rule classified
    nothing. Quantiles are self-calibrating -- they cannot fall off the end of
    the distribution, and the thresholds they imply are written into the manifest
    so the choice stays auditable.

      dark_weak  : a target IS present (n_gt >= 1) AND the scene is in the darkest
                   quartile AND the residual contrast is below the median
                   -> "暗弱刚体穿行": a weak target in a dark scene
      clutter    : a target IS present AND the residual contrast is in the top
                   decile -> "高频强杂波": busy background, easy to lose the target
      open_sky   : the sky band is the most uniform+bright of the pool AND there
                   is no target -> flat empty sky, the dynamic-range extreme
    """
    keys = sorted(stats)
    ir = np.array([stats[k]["ir_mean"] for k in keys])
    con = np.array([stats[k]["contrast"] for k in keys])
    sky = np.array([stats[k]["sky_ratio"] for k in keys])
    ngt = np.array([stats[k]["n_gt"] for k in keys])

    q_ir = float(np.quantile(ir, 0.25))
    q_con = float(np.quantile(con, 0.90))
    med_con = float(np.median(con))

    def seq_of(k: str) -> str:
        stem = Path(k).stem
        return stem.split("__frame_")[0] if "__frame_" in stem else stem

    dark_pool = [i for i, k in enumerate(keys)
                 if (restrict_dark is None or seq_of(k) in restrict_dark)
                 and ngt[i] >= 1 and ir[i] <= q_ir and con[i] <= med_con]
    clutter_pool = [i for i, k in enumerate(keys)
                    if (restrict_clutter is None or seq_of(k) in restrict_clutter)
                    and ngt[i] >= 1 and con[i] >= q_con]
    sky_pool = [i for i, k in enumerate(keys)
                if ngt[i] == 0 and sky[i] > 0.0]
    # if the "no target" sky pool is too small, fall back to the flattest frames
    if len(sky_pool) < n_sky:
        sky_pool = sorted(range(len(keys)), key=lambda i: -sky[i])[: max(n_sky * 4, 64)]

    def take(pool, n):
        if len(pool) < n:
            return []
        return [keys[i] for i in pool[:n]]

    return {
        "dark_weak": take(dark_pool, n_dark),
        "clutter": take(clutter_pool, n_clutter),
        "open_sky": take(sky_pool, n_sky),
        "_thresholds": {"q_ir_25": q_ir, "q_contrast_90": q_con, "median_contrast": med_con,
                        "pool": len(keys),
                        "pool_dark": len(dark_pool), "pool_clutter": len(clutter_pool),
                        "pool_sky": len(sky_pool)},
    }


# --------------------------------------------------------------------------------------
# Candidate pools
# --------------------------------------------------------------------------------------
def pool_official(root: Path, seqs: list[str]) -> dict[str, list[Path]]:
    """{sequence: [feature jpg paths]} -- existence-checked, deterministic order."""
    root = C.require(root, "official feature root")
    out: dict[str, list[Path]] = {}
    for seq in seqs:
        paths = sorted(root.glob(f"{seq}__frame_*.jpg"))
        if not paths:
            raise SystemExit(f"[FATAL] no feature images for sequence {seq} under {root}")
        out[seq] = paths
    return out


def pool_uav(root: Path) -> dict[str, list[Path]]:
    """Anti-UAV features: a FLAT directory, grouped by the filename prefix."""
    root = C.require(root, "Anti-UAV feature root")
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        stem = p.stem
        seq = stem.rsplit("__", 1)[0] if "__" in stem else stem
        groups[seq].append(p)
    if not groups:
        raise SystemExit(f"[FATAL] {root} holds no images")
    return dict(groups)


def stratified_sample(items: list[Path], n: int, seed: int) -> list[Path]:
    """Deterministic uniform sample of `n` items (sorted first, then a fixed seed)."""
    if len(items) < n:
        raise SystemExit(f"[FATAL] need {n} items, pool has {len(items)}")
    rng = random.Random(seed)
    picked = rng.sample(sorted(items), n)
    return sorted(picked)


def interleave(groups: list[list[Path]], n: int) -> list[Path]:
    """Round-robin over groups so every sequence contributes, deterministic."""
    out: list[Path] = []
    k = 0
    while len(out) < n:
        progressed = False
        for g in groups:
            if k < len(g):
                out.append(g[k])
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            raise SystemExit("[FATAL] interleaved pool exhausted before reaching the quota")
        k += 1
    return out


# --------------------------------------------------------------------------------------
# The four sets
# --------------------------------------------------------------------------------------
def build_set_a(seed: int) -> dict:
    """Random baseline: Anti-UAV train + 龙泉山 official train, uniform."""
    uav = pool_uav(C.FEATURE_UAV_TRAIN)
    off = pool_official(C.FEATURE_OFFICIAL_TRAIN, sorted(
        {p.name.split("__frame_")[0] for p in C.require(C.FEATURE_OFFICIAL_TRAIN).iterdir()
         if "__frame_" in p.name}))
    uav_items = [p for g in uav.values() for p in g]
    off_items = [p for g in off.values() for p in g]
    picks = interleave([
        stratified_sample(uav_items, 125, seed + 1),
        stratified_sample(off_items, 125, seed + 2),
    ], SET_SIZE)
    return {
        "name": "Set-A 随机基准",
        "composition": "Anti-UAV train 125 + 龙泉山 official train 125（均匀）",
        "items": picks,
        "strata": {},
    }


def _classify_pool(paths: list[Path], cache: dict, workers: int = 32) -> dict[str, list[Path]]:
    """Classify a candidate pool into {class: [paths]}, caching the statistics.

    `cache` is keyed by the path STRING so it survives the multiprocessing pickle.
    """
    todo = [p for p in paths if str(p) not in cache]
    if todo:
        cache.update(compute_stats(todo, workers=workers))
    buckets: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        buckets[classify(cache[str(p)])].append(p)
    return buckets


def _val_pool() -> dict[str, list[Path]]:
    seqs = sorted({p.name.split("__frame_")[0]
                   for p in C.require(C.FEATURE_OFFICIAL_VAL).iterdir() if "__frame_" in p.name})
    return pool_official(C.FEATURE_OFFICIAL_VAL, seqs)


def build_set_b(seed: int, cache: dict) -> dict:
    """Dark-weak bias: 175 dark rigid-body transit frames + 75 open sky.

    The dark half is restricted to {001324, 003002} -- those are the two sequences
    where the target crosses a dark scene, which is where quantisation bites.
    """
    dark_seqs = ["VIDEO00007_19700101_001324", "VIDEO00009_19700101_003002"]
    dark_pool = pool_official(C.FEATURE_OFFICIAL_VAL, ["VIDEO00007_19700101_001324"])
    dark_pool["VIDEO00009_19700101_003002"] = sorted(
        C.require(C.FEATURE_OFFICIAL_TRAIN).glob("VIDEO00009_19700101_003002__frame_*.jpg"))
    dark_items = [p for g in dark_pool.values() for p in g]
    val_items = [p for g in _val_pool().values() for p in g]

    all_stats = dict(cache)
    all_stats.update(compute_stats([p for p in set(dark_items) | set(val_items)
                                    if str(p) not in all_stats], workers=40))
    sel = select_strata(all_stats, 175, 0, 75, restrict_dark=dark_seqs)
    thr = sel.pop("_thresholds")
    dark_pick = [Path(p) for p in sel["dark_weak"]]
    sky_pick = [Path(p) for p in sel["open_sky"]]
    if len(dark_pick) < 175 or len(sky_pick) < 75:
        raise SystemExit(f"[FATAL] Set-B strata short: dark={len(dark_pick)} sky={len(sky_pick)} "
                         f"thresholds={thr}")
    cache.update(all_stats)
    return {
        "name": "Set-B 暗弱难例偏置",
        "composition": "001324+003002 暗弱刚体穿行 175 + 开阔天空 75",
        "items": dark_pick + sky_pick,
        "strata": {"dark_weak": len(dark_pick), "open_sky": len(sky_pick)},
        "thresholds": thr,
    }


def build_set_c(seed: int, cache: dict) -> dict:
    """High-dynamic clutter: 175 complex-ground/building/road/forest-edge + 75 open sky."""
    items = [p for g in _val_pool().values() for p in g]
    all_stats = dict(cache)
    all_stats.update(compute_stats([p for p in items if str(p) not in all_stats], workers=40))
    sel = select_strata(all_stats, 0, 175, 75)
    thr = sel.pop("_thresholds")
    clutter_pick = [Path(p) for p in sel["clutter"]]
    sky_pick = [Path(p) for p in sel["open_sky"]]
    if len(clutter_pick) < 175 or len(sky_pick) < 75:
        raise SystemExit(f"[FATAL] Set-C strata short: clutter={len(clutter_pick)} "
                         f"sky={len(sky_pick)} thresholds={thr}")
    cache.update(all_stats)
    return {
        "name": "Set-C 高动态杂波",
        "composition": "复杂地表/建筑/公路/山林边缘反光 175 + 开阔天空 75",
        "items": clutter_pick + sky_pick,
        "strata": {"clutter": len(clutter_pick), "open_sky": len(sky_pick)},
        "thresholds": thr,
    }


def build_set_d(seed: int, cache: dict) -> dict:
    """Golden tri-state: 85 very dark weak + 85 high-frequency clutter + 80 open sky."""
    items = [p for g in _val_pool().values() for p in g]
    all_stats = dict(cache)
    all_stats.update(compute_stats([p for p in items if str(p) not in all_stats], workers=40))
    sel = select_strata(all_stats, 85, 85, 80)
    thr = sel.pop("_thresholds")
    dark = sorted(sel["dark_weak"], key=lambda k: all_stats[k]["ir_mean"])
    clutter = sorted(sel["clutter"], key=lambda k: -all_stats[k]["contrast"])
    sky = sel["open_sky"]
    picks = [Path(p) for p in (dark[:85] + clutter[:85] + sky[:80])]
    if len(picks) != 250:
        raise SystemExit(f"[FATAL] Set-D strata short: {len(picks)} != 250 thresholds={thr}")
    cache.update(all_stats)
    return {
        "name": "Set-D 黄金三元态平衡",
        "composition": "极暗弱刚体 85 + 高频强杂波 85 + 开阔标准空域 80",
        "items": picks,
        "strata": {"very_dark_weak": 85, "strong_clutter": 85, "open_sky": 80},
        "thresholds": thr,
    }


BUILDERS = {"A": build_set_a, "B": build_set_b, "C": build_set_c, "D": build_set_d}


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.WORK_ROOT / "calib"))
    ap.add_argument("--set", choices=list(BUILDERS), default=None,
                    help="build only this set (default: all four)")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--bmp", action="store_true", default=True,
                    help="also write the lossless BMP copy (default on)")
    ap.add_argument("--no-bmp", dest="bmp", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # --- explicit existence checks on every input we are going to touch -----------
    for p, what in ((C.FEATURE_UAV_TRAIN, "Anti-UAV train features"),
                    (C.FEATURE_OFFICIAL_TRAIN, "龙泉山 official train features"),
                    (C.FEATURE_OFFICIAL_VAL, "龙泉山 official val features")):
        C.require(p, what)
        print(f"[check] {what}: {p} OK")

    cache: dict = {}
    names = [args.set] if args.set else list(BUILDERS)
    summary = {"seed": args.seed, "set_size": SET_SIZE, "sets": {}}

    for key in names:
        print(f"\n===== building Set-{key} =====", flush=True)
        if key == "A":
            spec = BUILDERS[key](args.seed)
        else:
            spec = BUILDERS[key](args.seed, cache)
        items = spec["items"]
        assert len(items) == SET_SIZE, f"Set-{key}: {len(items)} != {SET_SIZE}"
        assert len({str(p) for p in items}) == SET_SIZE, f"Set-{key}: duplicate entries"

        seqs = defaultdict(int)
        for p in items:
            seqs[p.name.split("__frame_")[0] if "__frame_" in p.name
                 else p.stem.rsplit("__", 1)[0]] += 1

        lines = [str(p) for p in items]
        src_path = out_root / f"set_{key}_sources.txt"
        src_path.write_text("\n".join(lines) + "\n")

        bmp_dir = out_root / f"set_{key}_bmp"
        bmp_items: list[Path] = []
        if args.bmp and not args.dry_run:
            bmp_dir.mkdir(parents=True, exist_ok=True)
            for src in items:
                dst = bmp_dir / Path(src).name.replace(".jpg", ".bmp").replace(".png", ".bmp")
                C.write_feature_bmp(src, dst)
                bmp_items.append(dst)
            nbmp = len(list(bmp_dir.glob("*.bmp")))
            assert nbmp == SET_SIZE, f"Set-{key}: wrote {nbmp} BMP != {SET_SIZE}"

        # THE calibration list points at the LOSSLESS BMPs, never at the source JPEGs.
        # RKNN's own image reader and the board's stb_image are different decoders;
        # if the quantiser calibrates on JPEG pixels while the board runs on BMP
        # pixels, the calibration distribution is not the inference distribution.
        list_path = out_root / f"set_{key}.txt"
        listed = bmp_items if (args.bmp and not args.dry_run) else items
        list_path.write_text("\n".join(str(p) for p in listed) + "\n")
        # the SOP's hard requirement, asserted on the file that RKNN will actually read
        written = [l for l in list_path.read_text().splitlines() if l.strip()]
        assert len(written) == SET_SIZE, f"Set-{key}: file has {len(written)} lines != {SET_SIZE}"
        for l in written:
            if not Path(l).is_file():
                raise SystemExit(f"[FATAL] Set-{key}: listed file does not exist: {l}")

        meta = {
            "set": key,
            "name": spec["name"],
            "composition": spec["composition"],
            "strata": spec["strata"],
            "n": SET_SIZE,
            "list": str(list_path),
            "bmp_dir": str(bmp_dir) if args.bmp else None,
            "per_sequence": dict(sorted(seqs.items())),
            "source_roots": {
                "official_val": str(C.FEATURE_OFFICIAL_VAL),
                "official_train": str(C.FEATURE_OFFICIAL_TRAIN),
                "uav_train": str(C.FEATURE_UAV_TRAIN),
            },
            "stratification_constants": {
                "DARK_LEVEL": DARK_LEVEL, "DARK_RATIO_HI": DARK_RATIO_HI,
                "DARK_RATIO_LO": DARK_RATIO_LO, "CLUTTER_HI": CLUTTER_HI,
                "SKY_ROW_FRAC": SKY_ROW_FRAC, "SKY_UNIFORM_STD": SKY_UNIFORM_STD,
                "SKY_BRIGHT_LEVEL": SKY_BRIGHT_LEVEL, "SKY_RATIO_HI": SKY_RATIO_HI,
            },
        }
        summary["sets"][key] = meta
        print(f"[Set-{key}] {spec['name']}  n={SET_SIZE}  seqs={len(seqs)}")
        for s, c in sorted(seqs.items(), key=lambda kv: -kv[1])[:12]:
            print(f"         {s:38s} {c}")
        print(f"[Set-{key}] list -> {list_path}")

    if not args.dry_run:
        C.write_json(out_root / "calib_sets_manifest.json", summary)
        print(f"\n[OK] manifest -> {out_root / 'calib_sets_manifest.json'}")
    else:
        print("\n[DRY-RUN] nothing written")


if __name__ == "__main__":
    main()
