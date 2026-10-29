#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""load_and_align_subset.py -- subset alignment validator for the opportunistic scheduler.

WHY THIS TOOL EXISTS
--------------------
The pipeline used to be compared with the golden baseline by a FULL-SEQUENCE
diff: "the arm must have exactly the same rows as the reference".  That is
only possible when the arm computes EVERY frame.  The opportunistic latest
frame scheduler (dynamic latest-frame scheduling) does NOT: it picks the
NEWEST frame whenever the GMC engine is free, so the number of processed
frames depends on the load -- it is N, not 11810, and N is different on every
run.

So the comparison has to change shape: instead of comparing ROW COUNTS, the
arm's OWN output declares which physical frame numbers it actually processed,
and the golden baseline is filtered onto exactly that set.  Everything is then
compared frame-by-frame inside that subset.

THE THREE KEYS
--------------
1. **The truth key is the arm's runtime frame list.**  The board writes one
   line per computed group into ``feature_md5.txt`` (``seq \\t %06d(frame_id)
   \\t md5``); that file IS the list of frames the pipeline really produced.
   Nothing is inferred from a row count and nothing is assumed dense.
2. **The golden baseline is filtered onto that set.**  ``ref_subset = {k: v
   for k, v in golden.items() if k in surviving_frames}`` -- no frame of the
   baseline that the arm did not process takes part.
3. **The ground truth is restricted to the same subset**, and the reference
   arm is clipped to it as well, so ``delta F1`` measures per-frame DETECTION
   QUALITY and not FRAME COVERAGE.

WHAT IS CHECKED
---------------
* **Bit-exactness**: per-frame MD5 equality on the subset.  The contract is
  ``different == 0`` on the frames whose history was complete.  This is the
  strongest possible statement: if the features are bit-identical, the
  detections cannot differ either.
* **@S precision**: TP / FP / FN / Recall / Precision / F1 of the arm and of
  the reference on the SAME restricted label set, plus ``delta F1 = arm - ref``
  (which must be >= 0).
* **Hole isolation**: frames the library itself reported as having a hole
  (parsed from the board run log) are separated out, so a stretched temporal
  window can never hide inside an aggregate number.

The scorer itself is NOT re-implemented: ``score_surviving_frames.py`` in this
directory already owns the cache building, the label restriction and the call
into the frozen scorer, and is imported.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_surviving_frames import (  # noqa: E402  (the path insert has to come first)
    METRIC_KEYS,
    build_cache,
    delta,
    extract_params,
    frame_of,
    link_restricted_labels,
    meta_for,
    parse_mosaic_tsv,
    run_scorer,
    write_cache,
)

# The library's own hole report, one line per holed frame:
#   [WARN] GMC frame 10944 (engine 0): the PHYSICAL history has a HOLE: ...
HOLE_RE = re.compile(r"GMC frame (\d+) \(engine \d+\): the PHYSICAL history has a HOLE")


# --------------------------------------------------------------------------------------
# the runtime manifest: feature_md5.txt -> {frame_id: md5}
# --------------------------------------------------------------------------------------
def parse_feature_md5(path: Path) -> dict[int, str]:
    """-> {frame_id: md5}.  The file is ``seq \\t %06d(frame_id) \\t md5``."""
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"{path}: malformed line {line!r}")
        out[int(parts[1])] = parts[2].strip()
    return out


def parse_gap_frames(gap_log: Path | None, gap_frames: Path | None) -> set[int]:
    """The frames the library itself reported as having a hole."""
    holes: set[int] = set()
    if gap_log is not None:
        for line in gap_log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = HOLE_RE.search(line)
            if m:
                holes.add(int(m.group(1)))
    if gap_frames is not None:
        for line in gap_frames.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                holes.add(int(line))
    return holes


def parse_tsv_rows(path: Path, prefix: str) -> dict[int, str]:
    """-> {frame_id: the detection payload of that frame}.

    The payload is everything AFTER the ``<seq>__<frame_id>`` stem, tab
    separated: the detection count and then one field group per detection.
    Two arms agree on a frame iff these payloads are byte-identical -- that is
    the per-frame evidence that a change in the INPUT path (the upload, the
    canvas, the tile selection) did not change a single detection.
    """
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stem, _, payload = line.partition("\t")
        fid = int(stem.rsplit("__", 1)[1])
        if not stem.startswith(prefix):
            raise ValueError(f"{path}: {stem!r} does not belong to sequence {prefix}")
        out[fid] = payload
    return out


def compare_detection_streams(arm: dict[int, str], ref: dict[int, str]) -> dict:
    """Per-frame, byte-exact comparison of two detection streams."""
    common = sorted(set(arm) & set(ref))
    identical = [f for f in common if arm[f] == ref[f]]
    different = [f for f in common if arm[f] != ref[f]]
    samples = []
    for f in different[:5]:
        samples.append({"frame_id": f, "arm": arm[f][:300], "ref": ref[f][:300]})
    return {
        "common": len(common),
        "identical": len(identical),
        "different": len(different),
        "only_in_arm": len(set(arm) - set(ref)),
        "only_in_ref": len(set(ref) - set(arm)),
        "different_ids": different[:5000],
        "samples": samples,
    }


def runs_of(sorted_ids: list[int], limit: int = 12) -> list[tuple[int, int]]:
    """Contiguous runs of frame ids, e.g. [1,2,3,7] -> [(1,3),(7,7)]."""
    if not sorted_ids:
        return []
    out: list[tuple[int, int]] = []
    start = prev = sorted_ids[0]
    for fid in sorted_ids[1:]:
        if fid == prev + 1:
            prev = fid
            continue
        out.append((start, prev))
        start = prev = fid
    out.append((start, prev))
    return out[:limit]


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def render_markdown(payload: dict) -> str:
    sub = payload["subset"]
    bit = payload["bit_exact"]
    arms = payload["arms"]
    out: list[str] = []
    out.append(f"# 子集对齐校验 —— `{payload['sequence']}`\n")
    out.append(f"> 生成时间 {payload['generated']} · 生成脚本 "
               "`manu/pipeline/npu/ptq/load_and_align_subset.py`")
    out.append("> 打分器：`manu/evaluation/system_sota_longquan.py --no-pipeline`（单帧臂，**无 track**）")
    out.append(f"> 真值键 = 板端运行态产出清单 `{payload['arm_md5']}`")
    out.append(f"> 黄金基准 = `{payload['ref_md5']}`\n")

    out.append("## 1. 子集对齐（帧号集合）\n")
    out.append("| 项 | 值 |")
    out.append("|:--|--:|")
    out.append(f"| 基准总帧数 | {sub['ref_frames']} |")
    out.append(f"| **运行态实际处理帧数 \\|S\\|** | **{sub['surviving_frames']}** |")
    out.append(f"| S 中基准也有的帧数 | {sub['ref_in_S']} |")
    out.append(f"| S 中基准没有的帧数 | {sub['S_missing_in_ref']} |")
    out.append(f"| S 中 TSV 有检出行的帧数 | {sub['S_in_arm_tsv']} |")
    out.append(f"| 运行态清单与 TSV 的帧号差异 | {sub['manifest_vs_tsv']} |")
    out.append(f"| S 的帧号范围 | {sub['frame_min']} .. {sub['frame_max']} |")
    out.append("")

    out.append("## 2. 逐位 MD5 校验\n")
    out.append("| 项 | 值 |")
    out.append("|:--|--:|")
    out.append(f"| 参与比对的帧数 | {bit['compared']} |")
    out.append(f"| **逐位相同** | **{bit['identical']}** |")
    out.append(f"| **逐位不同** | **{bit['different']}** |")
    out.append(f"| bit-exact 率 | {bit['rate'] * 100:.4f}% |")
    out.append(f"| 无洞帧：相同 / 不同 | {bit['clean_identical']} / {bit['clean_different']} |")
    out.append(f"| 有洞帧：相同 / 不同 | {bit['holed_identical']} / {bit['holed_different']} |")
    out.append(f"| 板端自报有洞帧数 | {bit['hole_frames_reported']} |")
    out.append("")
    if bit["different_ids"]:
        out.append("**逐位不同的帧号（前 12 段连续区间）**：")
        out.append("")
        out.append("```")
        out.append(", ".join(f"{a}..{b}" if a != b else f"{a}" for a, b in runs_of(bit["different_ids"])))
        out.append("```")
        out.append("")

    det = payload["det_stream"]
    out.append("## 2b. 检测流逐帧比对（下游输入路径的证据）\n")
    out.append("> 特征 MD5 覆盖的是 GMC 输出，**位于拼图画布与上传之前**；"
               "所以 MD5 相同**不能**证明 NPU 吃到的字节相同。"
               "这里比对的是两臂 TSV 里每一帧的**检测载荷**（检出数 + 每个检出的全部字段），"
               "逐字节。\n")
    out.append("| 项 | 值 |")
    out.append("|:--|--:|")
    out.append(f"| 两臂共有的检测帧数 | {det['common']} |")
    out.append(f"| **逐帧完全相同** | **{det['identical']}** |")
    out.append(f"| **逐帧不同** | **{det['different']}** |")
    out.append(f"| 只在 arm 里的帧 | {det['only_in_arm']} |")
    out.append(f"| 只在 ref 里的帧 | {det['only_in_ref']} |")
    out.append("")
    if det["samples"]:
        out.append("**不同帧样例（前 5 帧）**：")
        out.append("")
        out.append("| frame_id | arm | ref |")
        out.append("|--:|:--|:--|")
        for s in det["samples"]:
            out.append(f"| {s['frame_id']} | `{s['arm']}` | `{s['ref']}` |")
        out.append("")

    if arms:
        out.append("## 3. @S 口径精度（同一受限帧宇宙）\n")
        out.append("| 口径 | 评测帧数 | GT | TP | FP | FN | Recall | Precision | F1 |")
        out.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
        for key in ("arm@S", "ref@S"):
            row = arms[key]
            m = row["metrics"]
            out.append(
                f"| **{key}** | {row['frames']} | {m['gt']} | {m['tp']} | {m['fp']} | {m['fn']} | "
                f"{m['recall']:.6f} | {m['precision']:.6f} | {m['f1']:.6f} |")
        d = delta(arms["ref@S"]["metrics"], arms["arm@S"]["metrics"])
        out.append("")
        out.append("| 差值 | ΔTP | ΔFP | ΔFN | ΔGT | ΔRecall | ΔPrecision | **ΔF1** |")
        out.append("|:--|--:|--:|--:|--:|--:|--:|--:|")
        out.append(f"| arm@S − ref@S | {d['tp']:+d} | {d['fp']:+d} | {d['fn']:+d} | {d['gt']:+d} | "
                   f"{d['recall']:+.6f} | {d['precision']:+.6f} | **{d['f1']:+.6f}** |")
        out.append("")

    out.append("## 4. 判定\n")
    out.append("| 判据 | 结果 | 状态 |")
    out.append("|:--|:--|:--:|")
    for item in payload["verdict"]["items"]:
        out.append(f"| {item['name']} | {item['value']} | {'✅ PASS' if item['ok'] else '❌ FAIL'} |")
    out.append("")
    out.append(f"**总判定：{'✅ PASS' if payload['verdict']['pass'] else '❌ FAIL'}**\n")

    out.append("## 5. 口径声明（必读）\n")
    out.append("1. **真值键是板端自己产出的物理帧号集合**（`feature_md5.txt`），不是推算的行数、"
               "不是配置里的 cadence、也不是假定稠密的帧号序列。")
    out.append("2. **基准被裁到同一子集**：只有 S 里的帧参与逐位比对与精度比对，"
               "因此 ΔF1 只反映单帧检测质量，不含帧覆盖率。")
    out.append("3. **GT 被限制到 S**（软链到临时目录，原始标签树只读），基准臂同样裁到 S，"
               "两臂在同一份受限 GT 上打分。")
    out.append("4. **有洞帧被隔离**：板端自报有洞（`gmc_history_gaps > 0`）的帧单独统计，"
               "它们的历史窗口被拉伸，不允许混进「无洞帧必须 100% 逐位相同」的结论里。")
    out.append("5. **逐位相同 ⇒ 精度必然相同**：特征 MD5 相同则特征张量逐位相同，"
               "下游 NPU 与 POST 的输出必然相同，所以「精度完全一致」的最强判据是 `different == 0`。\n")
    return "\n".join(out)


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-md5", required=True,
                    help="the arm's runtime feature_md5.txt (the truth key: the frames it really processed)")
    ap.add_argument("--arm-tsv", required=True, help="the arm's <SEQ>_mosaic.tsv")
    ap.add_argument("--ref-md5", required=True, help="the golden baseline's feature_md5.txt")
    ap.add_argument("--ref-tsv", required=True, help="the golden baseline's <SEQ>_mosaic.tsv")
    ap.add_argument("--seq", required=True, help="sequence name, e.g. VIDEO00007_19700101_001324")
    ap.add_argument("--labels-root", help="the ORIGINAL (read-only) label directory")
    ap.add_argument("--params-json", help="gate artefact with the scoring params")
    ap.add_argument("--work-dir", help="writable scratch directory")
    ap.add_argument("--out-json", required=True, help="output report (json)")
    ap.add_argument("--out-md", required=True, help="output report (markdown)")
    ap.add_argument("--gap-log", help="the board run log; the library's HOLE warnings are parsed from it")
    ap.add_argument("--gap-frames", help="a file with one holed frame id per line")
    ap.add_argument("--manifest-out",
                    help="write the merged runtime manifest (frame_id,feature_md5,n_dets) here")
    ap.add_argument("--skip-score", action="store_true",
                    help="only do the bit-exact part (no ground truth needed)")
    ap.add_argument("--clean-work", action="store_true", help="delete the scratch directory when done")
    args = ap.parse_args()

    started = time.perf_counter()
    seq = args.seq
    arm_md5 = Path(args.arm_md5)
    arm_tsv = Path(args.arm_tsv)
    ref_md5 = Path(args.ref_md5)
    ref_tsv = Path(args.ref_tsv)

    # ---- 1. the runtime manifest IS the truth key -----------------------------------
    arm_finger = parse_feature_md5(arm_md5)
    ref_finger = parse_feature_md5(ref_md5)
    dets_arm = parse_mosaic_tsv(arm_tsv)
    dets_ref = parse_mosaic_tsv(ref_tsv)
    prefix = f"{seq}__"
    for stem in list(dets_arm) + list(dets_ref):
        if not stem.startswith(prefix):
            raise ValueError(f"{stem!r} does not belong to sequence {seq!r}")
    frames_arm_tsv = {frame_of(stem) for stem in dets_arm}
    frames_ref_tsv = {frame_of(stem) for stem in dets_ref}

    S = set(arm_finger)                     # the frames the pipeline REALLY produced
    S_tsv = S & frames_arm_tsv              # ... and the ones that carry detections
    ref_subset = {k: v for k, v in ref_finger.items() if k in S}
    holes = parse_gap_frames(
        Path(args.gap_log) if args.gap_log else None,
        Path(args.gap_frames) if args.gap_frames else None)
    print(f"[S] surviving frames |S| = {len(S)}  (arm TSV rows = {len(dets_arm)}, "
          f"ref frames = {len(frames_ref_tsv)}, ref in S = {len(ref_subset)}, "
          f"hole frames = {len(holes)})", flush=True)

    # ---- 2. the bit-exact comparison -------------------------------------------------
    common = sorted(set(ref_subset) & set(arm_finger))
    identical = [f for f in common if ref_subset[f] == arm_finger[f]]
    different = [f for f in common if ref_subset[f] != arm_finger[f]]
    clean = [f for f in common if f not in holes]
    holed = [f for f in common if f in holes]
    clean_identical = sum(1 for f in clean if ref_subset[f] == arm_finger[f])
    holed_identical = sum(1 for f in holed if ref_subset[f] == arm_finger[f])
    bit = {
        "compared": len(common),
        "identical": len(identical),
        "different": len(different),
        "rate": (len(identical) / len(common)) if common else 0.0,
        "clean_frames": len(clean),
        "clean_identical": clean_identical,
        "clean_different": len(clean) - clean_identical,
        "holed_frames": len(holed),
        "holed_identical": holed_identical,
        "holed_different": len(holed) - holed_identical,
        "hole_frames_reported": len(holes),
        "different_ids": different[:5000],
        "frames_only_in_arm": sorted(set(arm_finger) - set(ref_finger))[:5000],
        "frames_only_in_ref": len(set(ref_finger) - set(arm_finger)),
    }
    print(f"[MD5] compared={bit['compared']} identical={bit['identical']} "
          f"different={bit['different']}  (clean {clean_identical}/{len(clean)}, "
          f"holed {holed_identical}/{len(holed)})", flush=True)

    # ---- 2b. THE DETECTION STREAM, frame by frame -----------------------------------
    #
    # The feature MD5 covers the GMC output, which sits UPSTREAM of the canvas
    # and of the upload into the NPU.  So an MD5 match says NOTHING about the
    # input path.  The per-frame comparison of the detection payloads is the
    # evidence that the NPU actually consumed the same bytes.
    det_stream = compare_detection_streams(
        parse_tsv_rows(arm_tsv, prefix), parse_tsv_rows(ref_tsv, prefix))
    print(f"[DETS] common={det_stream['common']} identical={det_stream['identical']} "
          f"different={det_stream['different']}  (only_arm {det_stream['only_in_arm']}, "
          f"only_ref {det_stream['only_in_ref']})", flush=True)

    # ---- 3. the merged runtime manifest ----------------------------------------------
    if args.manifest_out:
        manifest = Path(args.manifest_out)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf-8") as fh:
            fh.write("frame_id,feature_md5,n_dets\n")
            for fid in sorted(arm_finger):
                stem = f"{seq}__{fid:06d}"
                fh.write(f"{fid},{arm_finger[fid]},{len(dets_arm.get(stem, []))}\n")
        print(f"[ok] runtime manifest -> {manifest}", flush=True)

    # ---- 4. the @S precision ----------------------------------------------------------
    arms: dict[str, dict] = {}
    if not args.skip_score:
        if not (args.labels_root and args.params_json and args.work_dir):
            raise SystemExit("--labels-root / --params-json / --work-dir are required unless --skip-score")
        labels_root = Path(args.labels_root)
        params_src = Path(args.params_json)
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)

        dets_arm_s = {stem: d for stem, d in dets_arm.items() if frame_of(stem) in S_tsv}
        dets_ref_s = {stem: d for stem, d in dets_ref.items() if frame_of(stem) in S_tsv}
        cache_arm = build_cache(dets_arm_s, meta_for("arm_live", seq))
        cache_ref_s = build_cache(dets_ref_s, meta_for("ref_nodrop_at_S", seq))
        if len(cache_arm["frame_index"]) != len(cache_ref_s["frame_index"]):
            raise RuntimeError(
                f"frame universe mismatch: arm {len(cache_arm['frame_index'])} vs "
                f"ref clipped {len(cache_ref_s['frame_index'])}")
        root_arm = work / "cache_arm"
        root_ref_s = work / "cache_ref_at_S"
        write_cache(root_arm, seq, cache_arm)
        write_cache(root_ref_s, seq, cache_ref_s)

        labels_s = work / "labels_S"
        link = link_restricted_labels(labels_root, seq, S_tsv, labels_s)
        print(f"[GT] restricted labels: {link['linked']} symlinked ({link['missing']} of "
              f"{len(S_tsv)} frames have no label file)", flush=True)

        params = extract_params(params_src)
        params_json = work / "params.json"
        params_json.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[PARAMS] th_base={params.get('th_base')} dedup_radius={params.get('dedup_radius')}",
              flush=True)

        scores_dir = work / "scores"
        for key, (cache_root, n_frames) in {
            "arm@S": (root_arm, len(cache_arm["frame_index"])),
            "ref@S": (root_ref_s, len(cache_ref_s["frame_index"])),
        }.items():
            report = run_scorer(cache_root, labels_s, seq, params_json, scores_dir / f"{key}.json")
            arms[key] = {"metrics": report["overall"], "per_sequence": report["per_sequence"],
                         "frames": n_frames, "cache_root": str(cache_root),
                         "labels_root": str(labels_s)}
            m = report["overall"]
            print(f"[{key:>6}] frames={n_frames} GT={m['gt']} TP={m['tp']} FP={m['fp']} FN={m['fn']} "
                  f"R={m['recall'] * 100:.3f}% P={m['precision'] * 100:.3f}% F1={m['f1']:.6f}",
                  flush=True)

    # ---- 5. the verdict ----------------------------------------------------------------
    items: list[dict] = []
    items.append({"name": "无洞帧逐位相同率 = 100%",
                  "value": f"{clean_identical}/{len(clean)}",
                  "ok": (len(clean) - clean_identical) == 0})
    items.append({"name": "全子集逐位不同帧数 = 0",
                  "value": str(bit["different"]),
                  "ok": bit["different"] == 0})
    items.append({"name": "板端自报有洞帧数",
                  "value": str(len(holes)),
                  "ok": True})
    items.append({"name": "检测流逐帧一致（共同帧）",
                  "value": f"{det_stream['identical']}/{det_stream['common']}",
                  "ok": det_stream["different"] == 0})
    if arms:
        d = delta(arms["ref@S"]["metrics"], arms["arm@S"]["metrics"])
        items.append({"name": "ΔF1 = arm@S − ref@S ≥ 0",
                      "value": f"{d['f1']:+.6f}",
                      "ok": d["f1"] >= -1e-9})
    verdict = {"items": items, "pass": all(i["ok"] for i in items)}

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequence": seq,
        "arm_md5": str(arm_md5),
        "arm_tsv": str(arm_tsv),
        "ref_md5": str(ref_md5),
        "ref_tsv": str(ref_tsv),
        "labels_root": args.labels_root,
        "params_source": args.params_json,
        "subset": {
            "ref_frames": len(ref_finger),
            "surviving_frames": len(S),
            "ref_in_S": len(ref_subset),
            "S_missing_in_ref": len(S - set(ref_finger)),
            "S_in_arm_tsv": len(S_tsv),
            "manifest_vs_tsv": len(S ^ frames_arm_tsv),
            "frame_min": min(S) if S else -1,
            "frame_max": max(S) if S else -1,
        },
        "bit_exact": bit,
        "det_stream": det_stream,
        "arms": arms,
        "hole_frames": sorted(holes)[:5000],
        "verdict": verdict,
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
    print(f"[ok] VERDICT {'PASS' if verdict['pass'] else 'FAIL'}  "
          f"({payload['elapsed_seconds']:.1f}s)")

    if args.clean_work and args.work_dir:
        shutil.rmtree(args.work_dir, ignore_errors=True)
        print(f"[ok] scratch dir removed: {args.work_dir}")

    sys.exit(0 if verdict["pass"] else 1)


if __name__ == "__main__":
    main()
