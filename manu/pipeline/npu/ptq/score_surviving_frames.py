#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""score_surviving_frames.py -- score an arm on the frames it actually survived.

WHY THIS TOOL EXISTS
--------------------
``manu/evaluation/system_sota_longquan.py`` builds its per-frame records from
``GT frames UNION candidate frames`` (``make_records``, L196-231).  A live arm
that produced only 3793 of the 11810 frames therefore collects ~8000 GT frames
as pure FN and its F1 collapses -- a number that measures *frame coverage*, not
per-frame detection quality.  ``score_pipeline_arms.sh`` documents the same
caveat ("only FULL-SEQUENCE arms are comparable").

THE "SURVIVING FRAMES" CALIBER
------------------------------
  * ``S`` = the frame numbers that appear in the **arm TSV**.  A row with 0
    detections still counts: the row means "the pipeline produced an artefact
    for this frame", which is the definition of "the pipeline ran on it".
  * The ground truth is restricted to ``S`` by symlinking only those label
    files into a scratch directory.  **The original label tree is never
    modified.**
  * The **reference arm is clipped to ``S`` too**, and both arms are scored
    against the *same* restricted label set.  Otherwise the delta between the
    two F1s would mix "frame coverage" with "per-frame detection quality" and
    would not be attributable.
  * ``arm@fullGT`` is reported alongside as the un-restricted control, so the
    report shows both calibers at once.

IMPLEMENTATION NOTES
--------------------
* The frozen scorer is never modified; it is invoked as a subprocess with
  ``--no-pipeline`` exactly like ``38_mosaic_final_report.py`` does.
* ``parse_mosaic_tsv`` / ``build_cache`` are re-implemented here (mirroring
  ``38_mosaic_final_report.py``) instead of imported: that file starts with a
  digit and cannot be imported by name, and importing it would drag a whole
  report script into a single-number tool.
* ``--params-json`` accepts the gate artefact directly.  Its ``config.params``
  are extracted and written out as a flat dict, because the scorer does
  ``payload.get("params", payload)``: handing it the nested gate file unchanged
  would silently fall back to the scorer defaults (th_base=0.22 /
  dedup_radius=32) and produce a completely different number.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCORER = PROJECT_ROOT / "manu/evaluation/system_sota_longquan.py"


# --------------------------------------------------------------------------------------
# board TSV -> ms_cache-shaped dict (mirrors 38_mosaic_final_report.py)
# --------------------------------------------------------------------------------------
def parse_mosaic_tsv(path: Path) -> dict[str, list[tuple]]:
    """-> {stem: [(tag, cx, cy, w, h, peak, sum), ...]}"""
    out: dict[str, list[tuple]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        stem = parts[0]
        n = int(parts[1])
        dets = []
        for field in parts[2:]:
            v = field.split()
            if len(v) != 8:
                raise ValueError(f"{path}: bad detection field {field!r}")
            dets.append((int(v[0]), float(v[1]), float(v[2]), int(v[3]), int(v[4]),
                         float(v[6]), float(v[7])))
        if len(dets) != n:
            raise ValueError(f"{path}: {stem} declares {n} detections but carries {len(dets)}")
        out[stem] = dets
    return out


def build_cache(dets: dict[str, list[tuple]], meta: dict) -> dict:
    """ms_cache-shaped dict -- dtype layout copied from multiscale_heatmap_cache.py:437-476."""
    frame_index, points, sizes, scores, sums, areas, tags = [], [], [], [], [], [], []
    offset = [0]
    for stem in sorted(dets, key=lambda s: int(s.rsplit("__", 1)[1])):
        frame_index.append(int(stem.rsplit("__", 1)[1]))
        for tag, cx, cy, bw, bh, peak, total in dets[stem]:
            points.append((cx, cy))
            sizes.append((bw, bh))
            scores.append(peak)
            sums.append(total)
            areas.append(1)          # the scorer never reads `area` either
            tags.append(tag)
        offset.append(len(points))
    return {
        "meta": {**meta, "frames": len(frame_index)},
        "frame_index": np.asarray(frame_index, dtype=np.int32),
        "offset": np.asarray(offset, dtype=np.int64),
        "points": np.asarray(points, dtype=np.float32).reshape(-1, 2),
        "sizes": np.asarray(sizes, dtype=np.float32).reshape(-1, 2),
        "scores": np.asarray(scores, dtype=np.float16),
        "sums": np.asarray(sums, dtype=np.float32),
        "area": np.asarray(areas, dtype=np.int32),
        "tag": np.asarray(tags, dtype=np.uint8),
    }


def meta_for(source: str, seq: str) -> dict:
    """The cache meta the scorer reads (native_size drives the GT de-normalisation)."""
    return {
        "sequence": seq, "frames": 0, "native_size": [640, 512], "imgsz": 640,
        "stride": 2, "scales": [0, 160, 80], "fused_tag": 255,
        "coordinate_space": "native_pixel", "region_threshold": 0.06,
        "main_threshold": 0.22, "fusion_dilate": 32, "min_area": 4,
        "use_p0_highway": False, "source": source,
        "features_root": f"/mnt/manu/ptq/bmp/{seq}",
        "hm_weights": "trial_0080/weights/epoch3.pt",
    }


def frame_of(stem: str) -> int:
    return int(stem.rsplit("__", 1)[1])


# --------------------------------------------------------------------------------------
# scratch artefacts
# --------------------------------------------------------------------------------------
def write_cache(root: Path, seq: str, cache: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{seq}.pkl"
    path.write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
    return path


def link_restricted_labels(labels_root: Path, seq: str, frames: set[int], dest: Path) -> dict:
    """Symlink the label files of ``frames`` into ``dest``.  The source tree stays read-only."""
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f"{seq}__"
    linked = 0
    for path in sorted(labels_root.glob(f"{seq}__*.txt")):
        digits = path.name[len(prefix):-4]
        if not digits.isdigit() or int(digits) not in frames:
            continue
        target = dest / path.name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(path)
        linked += 1
    return {"linked": linked, "missing": len(frames) - linked}


def extract_params(params_json: Path) -> dict:
    """Normalise the gate artefact to a flat params dict.

    The scorer does ``payload.get("params", payload)``.  The gate artefacts nest
    the params under ``config.params`` and carry no top-level ``params`` key, so
    they MUST be unwrapped -- otherwise the scorer keeps its own defaults.
    """
    payload = json.loads(params_json.read_text(encoding="utf-8"))
    config = payload.get("config")
    if isinstance(config, dict) and isinstance(config.get("params"), dict):
        return dict(config["params"])
    if isinstance(payload.get("params"), dict):
        return dict(payload["params"])
    return {k: v for k, v in payload.items() if not isinstance(v, (dict, list))}


def run_scorer(cache_root: Path, labels_root: Path, seq: str, params_json: Path, out_json: Path) -> dict:
    """Run the frozen scorer (single-frame arm) and return its report."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCORER),
        "--cache-root", str(cache_root),
        "--labels-root", str(labels_root),
        "--seqs", seq,
        "--no-pipeline",
        "--params-json", str(params_json),
        "--out-json", str(out_json),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f"scorer failed ({' '.join(cmd)}):\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    report = json.loads(out_json.read_text(encoding="utf-8"))
    return {"overall": report["overall"], "per_sequence": report["per_sequence"],
            "stdout": result.stdout.strip()}


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
METRIC_KEYS = ("tp", "fp", "fn", "gt", "recall", "precision", "f1", "far_per_frame")


def delta(a: dict, b: dict) -> dict:
    """b - a over the reported metric keys."""
    return {key: b[key] - a[key] for key in METRIC_KEYS}


def render_markdown(payload: dict) -> str:
    cov = payload["surviving_frames"]
    arms = payload["arms"]
    deltas = payload["deltas"]
    params = payload["params"]
    out: list[str] = []
    out.append(f"# 存活帧口径精度对照 —— `{payload['sequence']}`\n")
    out.append(f"> 生成时间 {payload['generated']} · 生成脚本 `manu/pipeline/npu/ptq/score_surviving_frames.py`")
    out.append("> 打分器：`manu/evaluation/system_sota_longquan.py --no-pipeline`（单帧臂，**无 track**）")
    out.append(f"> 口径参数来自 `{payload['params_source']}`"
               f"（th_base={params.get('th_base')}, dedup_radius={params.get('dedup_radius')}）\n")

    out.append("## 1. 帧覆盖\n")
    out.append("| 项 | 值 |")
    out.append("|:--|--:|")
    out.append(f"| 存活帧数 \\|S\\|（arm TSV 行数） | {cov['arm_tsv_frames']} |")
    out.append(f"| ref TSV 行数（全量） | {cov['ref_tsv_frames']} |")
    out.append(f"| ref 在 S 上的帧数 | {cov['ref_frames_in_S']} |")
    out.append(f"| ref 对 S 的覆盖率 | {cov['ref_coverage_on_S']:.6f} |")
    out.append(f"| S 中 ref 未覆盖的帧数 | {cov['arm_frames_missing_in_ref']} |")
    out.append(f"| S 的帧号范围 | {cov['frame_min']} .. {cov['frame_max']} |")
    out.append(f"| S 中有 GT 标签文件的帧数 | {cov['gt_files_linked']} |")
    out.append(f"| S 中无 GT 标签文件的帧数 | {cov['gt_files_missing']} |")
    out.append(f"| arm TSV 中 0 检测的行数 | {cov['arm_zero_detection_rows']} |")
    out.append("")

    out.append("## 2. 精度对照\n")
    out.append("| 口径 | 评测帧数 | GT | TP | FP | FN | Recall | Precision | F1 | FAR/帧 |")
    out.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for key in ("arm@S", "ref@S", "arm@fullGT", "ref@fullGT"):
        row = arms[key]
        m = row["metrics"]
        out.append(
            f"| **{key}** | {row['frames']} | {m['gt']} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{m['recall']:.6f} | {m['precision']:.6f} | {m['f1']:.6f} | {m['far_per_frame']:.6f} |"
        )
    out.append("")

    out.append("## 3. 差值\n")
    out.append("| 对比 | ΔTP | ΔFP | ΔFN | ΔGT | ΔRecall | ΔPrecision | ΔF1 | ΔFAR/帧 |")
    out.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for key, (base, target) in deltas.items():
        d = delta(arms[base]["metrics"], arms[target]["metrics"])
        out.append(
            f"| {target} − {base} | {d['tp']:+d} | {d['fp']:+d} | {d['fn']:+d} | {d['gt']:+d} | "
            f"{d['recall']:+.6f} | {d['precision']:+.6f} | {d['f1']:+.6f} | {d['far_per_frame']:+.6f} |"
        )
    out.append("")

    out.append("## 4. 口径声明（必读）\n")
    out.append("1. **`arm@S` / `ref@S` 是同一帧宇宙上的对比**：两臂都用同一份「帧号 ∈ S」的 GT，"
               "且 ref 的 cache 也被裁到 S ⇒ ΔF1 里只有单帧检测质量，没有帧覆盖率。")
    out.append("2. **存活帧 = TSV 里出现的帧**（0 检测的行也算）：该行代表管线为这一帧产出了产物。"
               "若某个版本的管线对 0 检测帧不写行，S 会偏小——这是口径本身的约定，不是本工具引入的偏差。")
    out.append("3. **`arm@fullGT` 是对照组**：arm cache + 全量 GT ⇒ 未覆盖帧全部记 FN，"
               "F1 反映的是「覆盖 + 质量」的合成量，与部分臂不可横向比较。")
    out.append("4. **参数已归一化**：`--params-json` 若指向 gate 产物（params 嵌在 `config.params` 里），"
               "本脚本会取出后另存为扁平 dict 再喂给打分器；直接把嵌套文件丢给打分器会让它退回自带默认值。")
    out.append("5. **GT 目录只读**：受限 GT 由临时目录里的软链构成，原始标签目录未做任何改动。\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-tsv", required=True, help="the arm's <SEQ>_mosaic.tsv")
    ap.add_argument("--ref-tsv", required=True, help="the reference arm's <SEQ>_mosaic.tsv")
    ap.add_argument("--seq", required=True, help="sequence name, e.g. VIDEO00007_19700101_001324")
    ap.add_argument("--labels-root", required=True, help="the ORIGINAL (read-only) label directory")
    ap.add_argument("--params-json", required=True, help="gate artefact with the scoring params")
    ap.add_argument("--work-dir", required=True, help="writable scratch directory")
    ap.add_argument("--out-json", required=True, help="output report (json)")
    ap.add_argument("--out-md", required=True, help="output report (markdown)")
    ap.add_argument("--clean-work", action="store_true", help="delete the scratch directory when done")
    args = ap.parse_args()

    started = time.perf_counter()
    seq = args.seq
    arm_tsv = Path(args.arm_tsv)
    ref_tsv = Path(args.ref_tsv)
    labels_root = Path(args.labels_root)
    params_src = Path(args.params_json)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # ---- 1. parse the two TSVs and derive the surviving frame set S ------------------
    dets_arm = parse_mosaic_tsv(arm_tsv)
    dets_ref = parse_mosaic_tsv(ref_tsv)
    prefix = f"{seq}__"
    for stem in list(dets_arm) + list(dets_ref):
        if not stem.startswith(prefix):
            raise ValueError(f"{stem!r} does not belong to sequence {seq!r}")
    frames_arm = {frame_of(stem) for stem in dets_arm}
    frames_ref = {frame_of(stem) for stem in dets_ref}
    S = frames_arm
    zero_rows = sum(1 for stem, dets in dets_arm.items() if not dets)
    print(f"[S] surviving frames |S| = {len(S)}  (arm TSV rows = {len(dets_arm)}, "
          f"ref TSV rows = {len(dets_ref)}, arm 0-detection rows = {zero_rows})", flush=True)

    # ---- 2. the three caches --------------------------------------------------------
    cache_arm = build_cache(dets_arm, meta_for("arm_live", seq))
    cache_ref = build_cache(dets_ref, meta_for("ref_nodrop", seq))
    # the reference arm is clipped to S as well, otherwise the two F1s would differ in
    # frame coverage as well as in detection quality.
    dets_ref_at_s = {stem: dets for stem, dets in dets_ref.items() if frame_of(stem) in S}
    cache_ref_at_s = build_cache(dets_ref_at_s, meta_for("ref_nodrop_at_S", seq))
    if len(cache_ref_at_s["frame_index"]) != len(cache_arm["frame_index"]):
        raise RuntimeError(
            f"frame universe mismatch: arm cache has {len(cache_arm['frame_index'])} frames, "
            f"ref cache clipped to S has {len(cache_ref_at_s['frame_index'])}")

    root_arm = work / "cache_arm"
    root_ref = work / "cache_ref"
    root_ref_s = work / "cache_ref_at_S"
    write_cache(root_arm, seq, cache_arm)
    write_cache(root_ref, seq, cache_ref)
    write_cache(root_ref_s, seq, cache_ref_at_s)

    # ---- 3. the restricted GT (symlinks only, the original tree is untouched) -------
    labels_s = work / "labels_S"
    link = link_restricted_labels(labels_root, seq, S, labels_s)
    print(f"[GT] restricted labels: {link['linked']} files symlinked into {labels_s} "
          f"({link['missing']} of the |S| frames have no label file)", flush=True)

    # ---- 4. the normalised params ---------------------------------------------------
    params = extract_params(params_src)
    params_json = work / "params.json"
    params_json.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PARAMS] th_base={params.get('th_base')} dedup_radius={params.get('dedup_radius')} "
          f"(normalised from {params_src})", flush=True)

    # ---- 5. the four scorings ------------------------------------------------------
    scores_dir = work / "scores"
    # (cache_root, labels_root, evaluated frame count of that cache)
    runs = {
        "arm@S": (root_arm, labels_s, len(cache_arm["frame_index"])),
        "ref@S": (root_ref_s, labels_s, len(cache_ref_at_s["frame_index"])),
        "arm@fullGT": (root_arm, labels_root, len(cache_arm["frame_index"])),
        "ref@fullGT": (root_ref, labels_root, len(cache_ref["frame_index"])),
    }
    arms: dict[str, dict] = {}
    for key, (cache_root, gt_root, n_frames) in runs.items():
        report = run_scorer(cache_root, gt_root, seq, params_json, scores_dir / f"{key}.json")
        arms[key] = {
            "metrics": report["overall"],
            "per_sequence": report["per_sequence"],
            "frames": n_frames,
            "cache_root": str(cache_root),
            "labels_root": str(gt_root),
        }
        m = report["overall"]
        print(f"[{key:>9}] frames={arms[key]['frames']} GT={m['gt']} TP={m['tp']} FP={m['fp']} "
              f"FN={m['fn']} R={m['recall'] * 100:.3f}% P={m['precision'] * 100:.3f}% "
              f"F1={m['f1']:.6f} FAR={m['far_per_frame']:.6f}", flush=True)

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequence": seq,
        "arm_tsv": str(arm_tsv),
        "ref_tsv": str(ref_tsv),
        "labels_root": str(labels_root),
        "params_source": str(params_src),
        "params": params,
        "surviving_frames": {
            "count": len(S),
            "arm_tsv_frames": len(dets_arm),
            "ref_tsv_frames": len(dets_ref),
            "ref_frames_in_S": len(frames_ref & S),
            "ref_coverage_on_S": (len(frames_ref & S) / len(S)) if S else 0.0,
            "arm_frames_missing_in_ref": len(S - frames_ref),
            "arm_zero_detection_rows": zero_rows,
            "frame_min": min(S) if S else -1,
            "frame_max": max(S) if S else -1,
            "gt_files_linked": link["linked"],
            "gt_files_missing": link["missing"],
        },
        "arms": arms,
        "deltas": {
            "arm@S vs ref@S": ("ref@S", "arm@S"),
            "arm@S vs arm@fullGT": ("arm@fullGT", "arm@S"),
            "ref@S vs ref@fullGT": ("ref@fullGT", "ref@S"),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"[ok] -> {out_json}")
    print(f"[ok] -> {out_md}")
    print(f"[ok] elapsed {payload['elapsed_seconds']:.1f}s")

    if args.clean_work:
        shutil.rmtree(work, ignore_errors=True)
        print(f"[ok] scratch dir removed: {work}")


if __name__ == "__main__":
    main()
