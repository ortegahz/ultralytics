#!/usr/bin/env python3
"""Check that the rendered OSD mp4s are complete, readable, and of THIS run.

The file-count check in verify_all.py only proves that 24 mp4 files exist.  It
does not prove they are playable or that they contain the whole sequence: a
render that dies half-way still leaves a file behind.  This script closes that
gap with ffprobe, comparing each mp4 against the predictions it was rendered
from:

  * ffprobe must parse the file (not a truncated/corrupt container)
  * nb_frames must equal len(per_frame) of the matching predictions json
    -> no dropped or cut-off frames
  * frame rate must be exactly 25 fps, resolution 1920x1080
    -> the OSD matches the 25 fps evaluation caliber
  * the F1 baked into each mp4 filename, the f1_pct column of
    per_sequence_hyperparams.tsv and the pass/un-pass grouping must all equal
    the evaluation json
    -> the OSD was rendered from the delivered parameters, not from leftovers

usage: python3 check_osd_media.py [--osd-dir DIR] [--preds-dir DIR] [--json-out FILE]
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

DEFAULT_OSD = "/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_optgapfit_20261119"
DEFAULT_PREDS = "/mnt/data/siping/runs/trial0080_onnx/eval/preds_v3_optgapfit"
DEFAULT_EVAL = "/mnt/data/siping/runs/trial0080_onnx/eval/track_all_v3_optgapfit.json"


def probe(path):
    """Return {nb_frames, fps, width, height, duration} for an mp4, or None if unreadable."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,r_frame_rate,width,height",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=False).stdout
    try:
        j = json.loads(out)
        s = j["streams"][0]
        num, _, den = s.get("r_frame_rate", "0/1").partition("/")
        return {"nb_frames": int(s["nb_frames"]), "fps": float(num) / float(den) if float(den) else 0.0,
                "width": int(s["width"]), "height": int(s["height"]),
                "duration": float(j["format"]["duration"])}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osd-dir", default=DEFAULT_OSD)
    ap.add_argument("--preds-dir", default=DEFAULT_PREDS)
    ap.add_argument("--eval-json", default=DEFAULT_EVAL)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.osd_dir, "pass", "*.mp4")) +
                   glob.glob(os.path.join(args.osd_dir, "un-pass", "*.mp4")))
    results, failures = [], []
    per_seq = json.load(open(args.eval_json, encoding="utf-8"))["per_sequence"]
    for path in files:
        name = os.path.basename(path).rsplit("_F1-", 1)[0]
        info = probe(path)
        problems = []
        if info is None:
            problems.append("ffprobe 无法解析")
            frames = None
        else:
            frames = info["nb_frames"]
            if abs(info["fps"] - args.fps) > 0.05:
                problems.append(f"帧率 {info['fps']:.3f} != {args.fps}")
            if (info["width"], info["height"]) != (args.width, args.height):
                problems.append(f"分辨率 {info['width']}x{info['height']}")
            if info["duration"] <= 0:
                problems.append("时长为 0")
        pj = os.path.join(args.preds_dir, name + ".json")
        if not os.path.exists(pj):
            problems.append("缺少对应预测 json")
        else:
            expected = len(json.load(open(pj))["per_frame"])
            if frames is not None and frames != expected:
                problems.append(f"帧数 {frames} != 处理帧数 {expected}")
        m = re.match(r"(.+)_F1-([0-9.]+)\.mp4$", os.path.basename(path))
        if m:
            embedded = float(m.group(2))
            ref = per_seq.get(m.group(1), {}).get("f1")
            if ref is not None and abs(embedded - round(ref * 100, 3)) > 0.0005:
                problems.append(f"文件名 F1 {embedded} != 评测 F1 {round(ref * 100, 3)}")
        else:
            problems.append("文件名不含 _F1- 标记")
        ok = not problems
        failures += problems
        results.append({"file": os.path.basename(path), "sequence": name, "frames": frames, "ok": ok,
                        "problems": problems})
        print(f"  [{'PASS' if ok else 'FAIL'}] {os.path.basename(path)}" +
              (f"  frames={frames}" if frames is not None else "") +
              ("" if ok else "  <- " + "; ".join(problems)))

    tsv = os.path.join(args.osd_dir, "per_sequence_hyperparams.tsv")
    tsv_mismatch = []
    group_mismatch = []
    if os.path.exists(tsv):
        rows = list(csv.DictReader([l for l in open(tsv, encoding="utf-8") if not l.startswith("#")],
                                   delimiter="\t"))
        for r in rows:
            ref = per_seq.get(r["sequence"], {}).get("f1")
            if ref is not None and abs(float(r["f1_pct"]) - round(ref * 100, 3)) > 0.0005:
                tsv_mismatch.append((r["sequence"], float(r["f1_pct"]), round(ref * 100, 3)))
        on_disk = {os.path.basename(p).rsplit("_F1-", 1)[0]: "pass"
                   for p in glob.glob(os.path.join(args.osd_dir, "pass", "*.mp4"))}
        on_disk.update({os.path.basename(p).rsplit("_F1-", 1)[0]: "un-pass"
                        for p in glob.glob(os.path.join(args.osd_dir, "un-pass", "*.mp4"))})
        for r in rows:
            if on_disk.get(r["sequence"]) != r["group"]:
                group_mismatch.append((r["sequence"], r["group"], on_disk.get(r["sequence"])))
    else:
        tsv_mismatch.append(("(tsv 缺失)", None, None))

    summary = {"osd_dir": args.osd_dir, "preds_dir": args.preds_dir, "eval_json": args.eval_json,
               "tsv_mismatch": tsv_mismatch, "group_mismatch": group_mismatch, "checked": len(results),
               "passed": sum(r["ok"] for r in results), "failed": sum(not r["ok"] for r in results),
               "results": results}
    print("=" * 60)
    print(f"OSD 媒体完整性: {summary['passed']}/{summary['checked']} 通过, {summary['failed']} 失败")
    print(f"TSV 逐序列 F1 vs 评测 JSON: {'一致' if not tsv_mismatch else tsv_mismatch}")
    print(f"TSV group 列 vs 目录分组: {'一致' if not group_mismatch else group_mismatch}")
    failures += tsv_mismatch + group_mismatch
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"已写出 {args.json_out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
