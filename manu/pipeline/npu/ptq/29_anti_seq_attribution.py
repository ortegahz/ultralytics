#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
29_anti_seq_attribution.py -- attribute the Anti-UAV near-total misses to
EITHER int8 quantisation OR the float model itself.

THE QUESTION
------------
Three Anti-UAV sequences are almost completely missed by the int8 board model
(DJI_0051_2, DJI_0175_2, wg2022_ir_020_split_03).  Is that the quantisation
shaving the heat-map peaks, or is the float model blind on those frames too?

WHAT THIS SCRIPT DOES
---------------------
1. reads the board's int8 detections ($W/full_scale_int8/anti_board.tsv) and
   aggregates them PER SEQUENCE against the Anti-UAV GT tree;
2. re-runs the PT float reference arm on the SAME uint8 BMP bytes, with the
   SAME decoder (extract_peaks_np) and the SAME greedy matcher
   (C.match_distance, 8 px) that `23_full_scale_threshold_sweep.py --pt`
   used -- and PERSISTS the per-frame detections (23 threw them away);
3. decomposes the quantisation loss per sequence into dTP / dFP / dFN and
   classifies every sequence:
     cat 1  the float model is blind there too  (NOT a quantisation problem)
     cat 2  quantisation shaved the peaks        (R_PT - R_int8 > 0.10)
     cat 3  quantisation added false alarms      (dFP over threshold)
4. self-checks the per-sequence sums against the independent sweep file
   ($W/full_scale_sweep/anti_sweep.json :: board_at_030 / pt_at_030): TP/FP/FN/F1
   must match EXACTLY, otherwise the decoding / matching protocol drifted.
5. runs the paired McNemar test (exact binomial) on the per-frame pairs
   (int8 detected vs PT detected) and on the per-GT pairs -- never on two
   independent recall numbers (rules.md 3/6c).

Usage
-----
    python manu/pipeline/npu/ptq/29_anti_seq_attribution.py \
        --run-dir  /mnt/data/siping/ptq_work/full_scale_int8 \
        --anti-root /mnt/data/siping/ptq_work/pcfeat_bmp/images \
        --sweep-json /mnt/data/siping/ptq_work/full_scale_sweep/anti_sweep.json \
        --out      /mnt/data/siping/ptq_work/anti_seq_attribution \
        --pt-device cuda:1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

# The repo root is parents[4] (ptq -> npu -> pipeline -> manu -> <root>).
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent

TARGET_SEQS = ["DJI_0051_2", "DJI_0175_2", "wg2022_ir_020_split_03"]

# numeric criteria of the attribution classes (printed into the report)
CRIT = {
    "cat1_model_blind": "R_PT < 0.10  (the float arm is blind on this sequence too)",
    "cat2_quant_miss": "R_PT - R_int8 > 0.10  (quantisation shaved the peaks)",
    "cat3_quant_false_alarm": "dFP >= 5  or  (FP_PT >= 10 and dFP >= 0.5 * FP_PT)",
}


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_s04 = _load("ptq_04", "04_run_board_fast_eval.py")
_s23 = _load("ptq_23", "23_full_scale_threshold_sweep.py")

parse_board_tsv = _s04.parse_board_tsv
run_pt_reference = _s04.run_pt_reference


# --------------------------------------------------------------------------------------
# matching -- byte-for-byte the rule of C.match_distance, plus the per-GT hit mask
# --------------------------------------------------------------------------------------
def match_with_mask(dets: np.ndarray, gts: np.ndarray,
                    dist_thr: float = C.DIST_THR) -> tuple[int, int, int, np.ndarray]:
    """Same greedy rule as C.match_distance, plus the per-GT hit mask.

    The mask is what the object-level paired McNemar needs; the (tp, fp, fn)
    triple is asserted to be identical to C.match_distance on every frame.
    """
    if len(gts) == 0:
        return 0, len(dets), 0, np.zeros(0, bool)
    if len(dets) == 0:
        return 0, 0, len(gts), np.zeros(len(gts), bool)
    used = np.zeros(len(gts), bool)
    tp = 0
    for i in np.argsort(-dets[:, 2]):
        d = np.hypot(gts[:, 0] - dets[i, 0], gts[:, 1] - dets[i, 1])
        j = int(d.argmin())
        if d[j] <= dist_thr and not used[j]:
            used[j] = True
            tp += 1
    return tp, len(dets) - tp, int((~used).sum()), used


def score_frames(det_by_name: dict, rows: list[dict],
                 src_hw: dict) -> list[dict]:
    """One pass over the frames: per-frame (tp, fp, fn, n_det, gt hit mask).

    The geometry / matching code is a copy of 04_run_board_fast_eval.py
    ::score_rows, so the aggregates below are the same numbers the sweep
    reported -- only the per-frame resolution is new.
    """
    out: list[dict] = []
    for r in rows:
        seq = r["seq"]
        name = Path(r["bmp"]).name
        dets = det_by_name.get(name, [])
        h, w = src_hw[seq]
        pts = np.asarray([[d[0], d[1]] for d in dets], np.float32).reshape(-1, 2)
        scores = np.asarray([d[2] for d in dets], np.float32).reshape(-1)
        geom = C.letterbox_geometry((h, w))
        pts_src = C.canvas_to_src(pts, geom) if len(pts) else pts
        gpts = []
        gp = Path(r["gt"])
        if gp.is_file():
            for line in gp.read_text().splitlines():
                fl = line.split()
                if len(fl) >= 5:
                    gpts.append([float(fl[1]) * w, float(fl[2]) * h])
        gpts = np.asarray(gpts, np.float32).reshape(-1, 2)
        d = (np.concatenate([pts_src, scores[:, None]], 1) if len(pts_src)
             else np.zeros((0, 3), np.float32))
        a, b, c = C.match_distance(d, gpts)
        a2, b2, c2, hit = match_with_mask(d, gpts)
        if (a, b, c) != (a2, b2, c2):
            raise SystemExit("[FATAL] matching replication diverged on {}: "
                             "{} vs {}".format(name, (a, b, c), (a2, b2, c2)))
        out.append({
            "name": name, "seq": seq,
            "tp": int(a), "fp": int(b), "fn": int(c),
            "n_det": len(dets), "gt": int(len(gpts)), "hit": hit,
            "max_score": float(scores.max()) if len(scores) else 0.0,
        })
    return out


def aggregate(frames: list[dict], arm: str, keep: set[str] | None = None) -> dict:
    """tp / fp / fn / det over the frames whose sequence is in `keep`."""
    tp = fp = fn = det = 0
    for f in frames:
        if keep is not None and f["seq"] not in keep:
            continue
        tp += f[arm]["tp"]
        fp += f[arm]["fp"]
        fn += f[arm]["fn"]
        det += f[arm]["n_det"]
    return metrics(tp, fp, fn, det)


def metrics(tp: int, fp: int, fn: int, det: int | None = None) -> dict:
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "gt": tp + fn,
            "det": det if det is not None else tp + fp,
            "precision": prec, "recall": rec, "f1": f1}


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial (McNemar) p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * p)


def mcnemar_frames(frames: list[dict], keep: set[str] | None = None) -> dict:
    """Frame-level paired test: 'detected' = at least one detection >= th."""
    both = only_i8 = only_pt = neither = 0
    for f in frames:
        if keep is not None and f["seq"] not in keep:
            continue
        d8 = f["int8"]["n_det"] > 0
        dpt = f["pt"]["n_det"] > 0
        if d8 and dpt:
            both += 1
        elif d8:
            only_i8 += 1
        elif dpt:
            only_pt += 1
        else:
            neither += 1
    # b = int8 only (int8 better), c = PT only (PT better)
    b, c = only_i8, only_pt
    return {"both": both, "int8_only": b, "pt_only": c, "neither": neither,
            "discordant": b + c, "p_value": mcnemar_exact(b, c),
            "note": "b=int8-detected-only (int8 better), c=PT-detected-only (PT better)"}


def mcnemar_objects(frames: list[dict], keep: set[str] | None = None) -> dict:
    """Object-level paired test: for every GT box, did each arm hit it?"""
    b = c = 0
    n_gt = 0
    for f in frames:
        if keep is not None and f["seq"] not in keep:
            continue
        h8 = f["int8"]["hit"]
        hpt = f["pt"]["hit"]
        n_gt += len(h8)
        for i in range(len(h8)):
            if h8[i] and not hpt[i]:
                b += 1
            elif hpt[i] and not h8[i]:
                c += 1
    return {"int8_only": b, "pt_only": c, "discordant": b + c,
            "gt_total": n_gt, "p_value": mcnemar_exact(b, c),
            "note": "b=int8-hit-only (int8 better), c=PT-hit-only (PT better)"}


def classify(m_i8: dict, m_pt: dict) -> tuple[str, list[str]]:
    flags = []
    if m_pt["recall"] < 0.10:
        flags.append("cat1_model_blind")
    if m_pt["recall"] - m_i8["recall"] > 0.10:
        flags.append("cat2_quant_miss")
    d_fp = m_i8["fp"] - m_pt["fp"]
    if d_fp >= 5 or (m_pt["fp"] >= 10 and d_fp >= 0.5 * m_pt["fp"]):
        flags.append("cat3_quant_false_alarm")
    if not flags:
        return "cat0_no_quant_effect", flags
    for name in ("cat1_model_blind", "cat2_quant_miss", "cat3_quant_false_alarm"):
        if name in flags:
            return name, flags
    return "cat0_no_quant_effect", flags


# --------------------------------------------------------------------------------------
# detection persistence (23_full_scale_threshold_sweep.py threw the PT detections away)
# --------------------------------------------------------------------------------------
def write_det_tsv(path: Path, det_by_name: dict) -> None:
    with path.open("w") as f:
        for name in sorted(det_by_name):
            dets = det_by_name[name]
            cells = [name, str(len(dets))]
            for x, y, s in dets:
                cells.append("{} {} {}".format(x, y, s))
            f.write("\t".join(cells) + "\n")


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/mnt/data/siping/ptq_work/full_scale_int8")
    ap.add_argument("--bmp-root", default="/mnt/data/siping/ptq_work/bmp")
    ap.add_argument("--anti-root",
                    default="/mnt/data/siping/ptq_work/pcfeat_bmp/images")
    ap.add_argument("--sweep-json",
                    default="/mnt/data/siping/ptq_work/full_scale_sweep/anti_sweep.json")
    ap.add_argument("--out", default="/mnt/data/siping/ptq_work/anti_seq_attribution")
    ap.add_argument("--th", type=float, default=0.30)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--variant", default="p0_off", choices=list(C.VARIANTS))
    ap.add_argument("--pt-device", default="cuda:1")
    ap.add_argument("--targets", nargs="+", default=TARGET_SEQS)
    ap.add_argument("--reuse-pt", action="store_true",
                    help="re-score from the saved PT detections instead of re-running")
    args = ap.parse_args()

    t_start = time.time()
    run_dir = Path(args.run_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    tsv = run_dir / "anti_board.tsv"
    if not tsv.is_file():
        raise SystemExit("[FATAL] no board TSV at {}".format(tsv))

    # ---------------- frame set (the TSV is the truth) + GT + source size -------------
    rows, src_hw = _s23.build_rows("anti", tsv, Path(args.bmp_root),
                                   Path(args.anti_root))
    seqs = sorted({r["seq"] for r in rows})
    print("[rows] {} frames / {} sequences".format(len(rows), len(seqs)), flush=True)
    for s in args.targets:
        if s not in seqs:
            raise SystemExit("[FATAL] target sequence {} is NOT in {} -- "
                             "check the naming before computing anything"
                             .format(s, tsv))

    # ---------------- int8 arm: the board's own detections ---------------------------
    board_raw = parse_board_tsv(tsv)
    board_at_th = {k: [d for d in v if d[2] >= args.th] for k, v in board_raw.items()}
    n_raw = sum(len(v) for v in board_raw.values())
    n_at_th = sum(len(v) for v in board_at_th.values())
    print("[int8] {} raw detections -> {} at th={:.2f}".format(
        n_raw, n_at_th, args.th), flush=True)

    # ---------------- PT float arm: same bytes, same decoder, same matcher ------------
    pt_file = out_root / "pt_detections.tsv"
    if args.reuse_pt and pt_file.is_file():
        pt_raw = parse_board_tsv(pt_file)
        print("[PT] reusing {}".format(pt_file), flush=True)
        pt_seconds = 0.0
    else:
        print("[PT] running the float reference on the same BMPs ...", flush=True)
        t0 = time.time()
        pt_raw = run_pt_reference(rows, src_hw, args.variant, topk=args.topk,
                                  device=args.pt_device)
        pt_seconds = round(time.time() - t0, 1)
        write_det_tsv(pt_file, pt_raw)
        print("[PT] done in {} s -> {}".format(pt_seconds, pt_file), flush=True)
    pt_at_th = {k: [d for d in v if d[2] >= args.th] for k, v in pt_raw.items()}

    # ---------------- per-frame scoring (one pass, both arms) -------------------------
    fr8 = score_frames(board_at_th, rows, src_hw)
    frpt = score_frames(pt_at_th, rows, src_hw)
    assert [f["name"] for f in fr8] == [f["name"] for f in frpt]
    frames = []
    for a, b in zip(fr8, frpt):
        frames.append({
            "name": a["name"], "seq": a["seq"],
            "int8": {"tp": a["tp"], "fp": a["fp"], "fn": a["fn"],
                     "n_det": a["n_det"], "hit": a["hit"],
                     "max_score": a["max_score"]},
            "pt": {"tp": b["tp"], "fp": b["fp"], "fn": b["fn"],
                   "n_det": b["n_det"], "hit": b["hit"],
                   "max_score": b["max_score"]},
        })

    # ---------------- per-sequence aggregates -----------------------------------------
    per_seq = {}
    for s in seqs:
        sub = [f for f in frames if f["seq"] == s]
        m8 = aggregate(sub, "int8")
        mpt = aggregate(sub, "pt")
        cat, flags = classify(m8, mpt)
        sc8 = [f["int8"]["max_score"] for f in sub if f["gt"] > 0]
        scpt = [f["pt"]["max_score"] for f in sub if f["gt"] > 0]
        per_seq[s] = {
            "frames": len(sub),
            "int8": m8, "pt": mpt,
            "delta": {"tp": m8["tp"] - mpt["tp"],
                      "fp": m8["fp"] - mpt["fp"],
                      "fn": m8["fn"] - mpt["fn"],
                      "recall": m8["recall"] - mpt["recall"]},
            "category": cat, "flags": flags,
            "intermediate": {
                "median_max_score_int8": statistics.median(sc8) if sc8 else 0.0,
                "median_max_score_pt": statistics.median(scpt) if scpt else 0.0,
                "frames_no_det_int8": sum(1 for f in sub if f["int8"]["n_det"] == 0),
                "frames_no_det_pt": sum(1 for f in sub if f["pt"]["n_det"] == 0),
                "frames_int8_only": sum(1 for f in sub
                                        if f["int8"]["n_det"] > 0 and f["pt"]["n_det"] == 0),
                "frames_pt_only": sum(1 for f in sub
                                      if f["pt"]["n_det"] > 0 and f["int8"]["n_det"] == 0),
            },
        }

    # ---------------- full-track self-check against the independent sweep -------------
    all_i8 = aggregate(frames, "int8")
    all_pt = aggregate(frames, "pt")
    sweep = json.loads(Path(args.sweep_json).read_text())
    check = {}
    for key, mine in (("board_at_030", all_i8), ("pt_at_030", all_pt)):
        ref = sweep[key]
        row = {"mine": {k: mine[k] for k in ("tp", "fp", "fn", "det")},
               "ref": {k: ref[k] for k in ("tp", "fp", "fn", "det")},
               "f1_mine": mine["f1"], "f1_ref": ref["f1"],
               "f1_abs_diff": abs(mine["f1"] - ref["f1"])}
        row["ok"] = all(row["mine"][k] == row["ref"][k]
                        for k in ("tp", "fp", "fn", "det")) and row["f1_abs_diff"] < 1e-9
        check[key] = row
    print("\n================ SELF-CHECK vs {} ================".format(args.sweep_json))
    print("{:<14}{:<6}{:>12}{:>12}{:>14}{:>10}".format(
        "arm", "item", "mine", "ref", "f1(mine/ref)", "ok"))
    for key, row in check.items():
        for k in ("tp", "fp", "fn", "det"):
            print("{:<14}{:<6}{:>12}{:>12}{:>14}{:>10}".format(
                key, k, row["mine"][k], row["ref"][k], "", "OK" if row["ok"] else "MISMATCH"))
        print("{:<14}{:<6}{:>12}{:>12}{:>14}{:>10}".format(
            key, "f1", "", "", "{:.6f}/{:.6f}".format(row["f1_mine"], row["f1_ref"]),
            "OK" if row["ok"] else "MISMATCH"))
    if not all(r["ok"] for r in check.values()):
        raise SystemExit("[FATAL] self-check FAILED -- the decoding / matching "
                         "protocol drifted, do not quote the attribution numbers")

    # ---------------- contribution of the target sequences ----------------------------
    tgt = set(args.targets)
    dji = {s for s in seqs if s.upper().startswith("DJI")}
    all_seqs = set(seqs)
    ex_tgt = all_seqs - tgt
    ex_dji = all_seqs - dji

    def subset_metrics(keep: set[str]) -> dict:
        return {"frames": sum(1 for f in frames if f["seq"] in keep),
                "int8": aggregate(frames, "int8", keep),
                "pt": aggregate(frames, "pt", keep)}

    contribution = {
        "target_seqs": sorted(tgt),
        "dji_seqs": sorted(dji),
        "frames": {"targets": sum(1 for f in frames if f["seq"] in tgt),
                   "dji": sum(1 for f in frames if f["seq"] in dji),
                   "total": len(frames)},
        "gt": {"targets": sum(per_seq[s]["int8"]["gt"] for s in tgt),
               "dji": sum(per_seq[s]["int8"]["gt"] for s in dji),
               "total": all_i8["gt"]},
        "int8_fn_in_targets": sum(per_seq[s]["int8"]["fn"] for s in tgt),
        "int8_fn_in_dji": sum(per_seq[s]["int8"]["fn"] for s in dji),
        "d_tp_total": all_i8["tp"] - all_pt["tp"],
        "d_tp_targets": sum(per_seq[s]["delta"]["tp"] for s in tgt),
        "d_tp_dji": sum(per_seq[s]["delta"]["tp"] for s in dji),
    }
    contribution["fn_share_targets"] = contribution["int8_fn_in_targets"] / max(1, all_i8["fn"])
    contribution["fn_share_dji"] = contribution["int8_fn_in_dji"] / max(1, all_i8["fn"])
    contribution["dtp_share_targets"] = contribution["d_tp_targets"] / max(1, contribution["d_tp_total"])
    contribution["dtp_share_dji"] = contribution["d_tp_dji"] / max(1, contribution["d_tp_total"])
    contribution["frame_share_targets"] = contribution["frames"]["targets"] / max(1, len(frames))
    contribution["gt_share_targets"] = contribution["gt"]["targets"] / max(1, all_i8["gt"])

    subsets = {
        "full": subset_metrics(all_seqs),
        "ex_targets": subset_metrics(ex_tgt),
        "ex_dji": subset_metrics(ex_dji),
        "targets_only": subset_metrics(tgt),
        "dji_only": subset_metrics(dji),
    }

    mcn = {
        "all_frames": mcnemar_frames(frames),
        "ex_dji_frames": mcnemar_frames(frames, ex_dji),
        "dji_frames": mcnemar_frames(frames, dji),
        "targets_frames": mcnemar_frames(frames, tgt),
        "ex_targets_frames": mcnemar_frames(frames, ex_tgt),
        "all_objects": mcnemar_objects(frames),
        "ex_dji_objects": mcnemar_objects(frames, ex_dji),
        "targets_objects": mcnemar_objects(frames, tgt),
    }

    report = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": str(run_dir), "anti_root": args.anti_root,
            "sweep_json": args.sweep_json, "out": str(out_root),
            "th": args.th, "topk": args.topk, "variant": args.variant,
            "weights": str(C.WEIGHTS), "pt_device": args.pt_device,
            "pt_seconds": pt_seconds, "seconds_total": round(time.time() - t_start, 1),
            "criteria": CRIT,
            "weights_md5": C.md5_file(C.WEIGHTS) if C.WEIGHTS.is_file() else None,
        },
        "self_check": check,
        "full_track": {"int8": all_i8, "pt": all_pt},
        "per_sequence": per_seq,
        "contribution": contribution,
        "subsets": subsets,
        "mcnemar": mcn,
        "targets": {s: per_seq[s] for s in args.targets},
    }
    C.write_json(out_root / "anti_seq_attribution.json", report)

    # ---------------- per-frame dump (reproducibility) --------------------------------
    with (out_root / "per_frame.tsv").open("w") as f:
        f.write("frame\tseq\tgt\ti8_tp\ti8_fp\ti8_fn\ti8_det\ti8_maxscore\t"
                "pt_tp\tpt_fp\tpt_fn\tpt_det\tpt_maxscore\n")
        for fr in frames:
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6f}\t{}\t{}\t{}\t{}\t{:.6f}\n".format(
                fr["name"], fr["seq"], fr["int8"]["tp"] + fr["int8"]["fn"],
                fr["int8"]["tp"], fr["int8"]["fp"], fr["int8"]["fn"],
                fr["int8"]["n_det"], fr["int8"]["max_score"],
                fr["pt"]["tp"], fr["pt"]["fp"], fr["pt"]["fn"],
                fr["pt"]["n_det"], fr["pt"]["max_score"]))

    write_markdown(out_root / "anti_seq_attribution.md", report, args)
    print_report(report, args)
    print("\n[out] {}".format(out_root))


def print_report(r: dict, args) -> None:
    print("\n================ PER-SEQUENCE ATTRIBUTION ================")
    hdr = ("{:<24}{:>7}{:>7}{:>7}{:>7}{:>8}{:>8} | {:>7}{:>7}{:>7}{:>8}{:>8}"
           .format("sequence", "GT", "i8_TP", "i8_FP", "i8_FN", "i8_R", "i8_P",
                   "pt_TP", "pt_FP", "pt_FN", "pt_R", "pt_P"))
    print(hdr)
    for s in r["meta"].get("_order", list(r["targets"].keys())):
        e = r["targets"][s]
        i8, pt = e["int8"], e["pt"]
        print("{:<24}{:>7}{:>7}{:>7}{:>7}{:>8.4f}{:>8.4f} | {:>7}{:>7}{:>7}{:>8.4f}{:>8.4f}"
              .format(s, i8["gt"], i8["tp"], i8["fp"], i8["fn"], i8["recall"],
                      i8["precision"], pt["tp"], pt["fp"], pt["fn"], pt["recall"],
                      pt["precision"]))
        print("{:<24}  dTP {:+d}  dFP {:+d}  dFN {:+d}  dR {:+.4f}  -> {}".format(
            "", e["delta"]["tp"], e["delta"]["fp"], e["delta"]["fn"],
            e["delta"]["recall"], e["category"]))


def write_markdown(path: Path, r: dict, args) -> None:
    L: list[str] = []
    A = L.append
    meta = r["meta"]
    A("# Anti-UAV 三序列漏检归因（int8 量化 vs PT float 模型本身）")
    A("")
    A("- 生成：`{}`　脚本：`manu/pipeline/npu/ptq/29_anti_seq_attribution.py`".format(meta["generated"]))
    A("- 权重：`{}`　md5 `{}`　口径：variant `{}`，th **0.30**，topk {}".format(
        meta["weights"], meta["weights_md5"], meta["variant"], meta["topk"]))
    A("- int8 臂：板端 TSV `{}`；PT 臂：同 uint8 BMP、同解码器 "
      "`extract_peaks_np`、同匹配 `C.match_distance`(8px)，本次重跑并**保留逐帧检测**".format(
          meta["run_dir"]))
    A("- 对照文件：`{}`；产物目录：`{}`".format(meta["sweep_json"], meta["out"]))
    A("")
    A("## 0. 全轨自检（逐序列加总 vs 独立 sweep 文件）")
    A("")
    A("| 臂 | TP | FP | FN | det | F1(我) | F1(参照) | 一致 |")
    A("|:---|---:|---:|---:|---:|---:|---:|:--:|")
    for key in ("board_at_030", "pt_at_030"):
        c = r["self_check"][key]
        A("| {} | {} | {} | {} | {} | {:.6f} | {:.6f} | {} |".format(
            key, c["mine"]["tp"], c["mine"]["fp"], c["mine"]["fn"], c["mine"]["det"],
            c["f1_mine"], c["f1_ref"], "✅" if c["ok"] else "❌"))
    A("")
    A("## 1. 三个目标序列的逐序归因")
    A("")
    A("| 序列 | 帧 | GT | int8 TP | int8 FP | int8 FN | int8 R | int8 P | "
      "PT TP | PT FP | PT FN | PT R | PT P | ΔTP | ΔFP | ΔFN | 归因 |")
    A("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
    for s, e in r["targets"].items():
        i8, pt = e["int8"], e["pt"]
        A("| `{}` | {} | {} | {} | {} | {} | {:.4f} | {:.4f} | {} | {} | {} | {:.4f} | {:.4f} "
          "| {:+d} | {:+d} | {:+d} | **{}** |".format(
              s, e["frames"], i8["gt"], i8["tp"], i8["fp"], i8["fn"], i8["recall"],
              i8["precision"], pt["tp"], pt["fp"], pt["fn"], pt["recall"], pt["precision"],
              e["delta"]["tp"], e["delta"]["fp"], e["delta"]["fn"], e["category"]))
    A("")
    A("判据：" + "；".join("**{}** = {}".format(k, v) for k, v in meta["criteria"].items()))
    A("")
    A("### 1b. 中间量（同序列逐帧最大检出分数中位数 / 无检出的帧数）")
    A("")
    A("| 序列 | int8 最大分中位数 | PT 最大分中位数 | int8 零检出帧 | PT 零检出帧 | "
      "int8-only 帧 | PT-only 帧 |")
    A("|:---|---:|---:|---:|---:|---:|---:|")
    for s, e in r["targets"].items():
        im = e["intermediate"]
        A("| `{}` | {:.3f} | {:.3f} | {} | {} | {} | {} |".format(
            s, im["median_max_score_int8"], im["median_max_score_pt"],
            im["frames_no_det_int8"], im["frames_no_det_pt"],
            im["frames_int8_only"], im["frames_pt_only"]))
    A("")
    A("## 2. 三条序列对全轨的贡献")
    A("")
    c = r["contribution"]
    A("| 项 | 值 | 占全轨 |")
    A("|:---|---:|---:|")
    A("| 帧数 | {} / {} | {:.2%} |".format(c["frames"]["targets"], c["frames"]["total"],
                                          c["frame_share_targets"]))
    A("| GT 数 | {} / {} | {:.2%} |".format(c["gt"]["targets"], c["gt"]["total"],
                                            c["gt_share_targets"]))
    A("| int8 的 FN（4301 个）落在三序列 | {} | {:.2%} |".format(
        c["int8_fn_in_targets"], c["fn_share_targets"]))
    A("| 全轨 ΔTP（int8−PT = {:+d}）落在三序列 | {:+d} | {:.2%} |".format(
        c["d_tp_total"], c["d_tp_targets"], c["dtp_share_targets"]))
    A("")
    A("## 3. 剔除后的全轨指标（回答「DJI 序列是不是量化损失的主要来源」）")
    A("")
    A("| 子集 | 臂 | 帧 | GT | TP | FP | FN | P | R | F1 |")
    A("|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, label in (("full", "全轨"), ("ex_targets", "剔除三条目标序列"),
                        ("ex_dji", "剔除两个 DJI 序列"),
                        ("targets_only", "仅三条目标序列"),
                        ("dji_only", "仅两个 DJI 序列")):
        sub = r["subsets"][name]
        for arm in ("int8", "pt"):
            m = sub[arm]
            A("| {} | {} | {} | {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
                label, arm, sub["frames"], m["gt"], m["tp"], m["fp"], m["fn"],
                m["precision"], m["recall"], m["f1"]))
    A("")
    A("## 4. 配对 McNemar（精确二项，双侧）——复核 trial-0474 的 DJI 结论")
    A("")
    A("### 4a. 帧级（检出 = 该帧有 ≥1 个 ≥0.30 的检测）")
    A("")
    A("| 口径 | 两边都检出 | int8-only | PT-only | 都没检出 | 不一致合计 | p |")
    A("|:---|---:|---:|---:|---:|---:|---:|")
    for name, label in (("all_frames", "全轨 31613 帧"),
                        ("ex_dji_frames", "剔除 DJI（历史口径）"),
                        ("dji_frames", "仅 DJI"),
                        ("targets_frames", "仅三条目标序列"),
                        ("ex_targets_frames", "剔除三条目标序列")):
        m = r["mcnemar"][name]
        A("| {} | {} | {} | {} | {} | {} | {:.4g} |".format(
            label, m["both"], m["int8_only"], m["pt_only"], m["neither"],
            m["discordant"], m["p_value"]))
    A("")
    A("### 4b. GT 级（每个标注框是否被命中）")
    A("")
    A("| 口径 | GT 总数 | int8-only | PT-only | 不一致合计 | p |")
    A("|:---|---:|---:|---:|---:|---:|")
    for name, label in (("all_objects", "全轨"), ("ex_dji_objects", "剔除 DJI"),
                        ("targets_objects", "仅三条目标序列")):
        m = r["mcnemar"][name]
        A("| {} | {} | {} | {} | {} | {:.4g} |".format(
            label, m["gt_total"], m["int8_only"], m["pt_only"], m["discordant"],
            m["p_value"]))
    A("")
    A("### 4c. 与历史对照（trial-0474，600 帧分层样本，NPU int8 v2 vs CPU fp32）")
    A("")
    A("| 项 | 历史 trial-0474 | 本次 trial-0080（全量 Anti-UAV） |")
    A("|:---|:---|:---|")
    exd = r["subsets"]["ex_dji"]
    A("| ex-DJI recall（float 臂） | 0.8955 | {:.4f} |".format(exd["pt"]["recall"]))
    A("| ex-DJI recall（int8 臂） | 0.8727 | {:.4f} |".format(exd["int8"]["recall"]))
    A("| ex-DJI 召回差（float−int8） | −2.27 pp | {:+.2f} pp |".format(
        100.0 * (exd["pt"]["recall"] - exd["int8"]["recall"])))
    A("| ex-DJI 帧级 McNemar | 12 / 2，p=0.0129 | {} / {}，p={:.4g} |".format(
        r["mcnemar"]["ex_dji_frames"]["int8_only"], r["mcnemar"]["ex_dji_frames"]["pt_only"],
        r["mcnemar"]["ex_dji_frames"]["p_value"]))
    A("")
    A("## 5. 结论")
    A("")
    A(write_conclusions(r))
    A("")
    path.write_text("\n".join(L) + "\n")


def write_conclusions(r: dict) -> str:
    lines = []
    for s, e in r["targets"].items():
        i8, pt = e["int8"], e["pt"]
        if e["category"] == "cat1_model_blind":
            verdict = ("**模型本身不行**：PT float 召回也只有 {:.4f}（{}/{}），"
                       "量化不是主因".format(pt["recall"], pt["tp"], pt["gt"]))
        elif e["category"] == "cat2_quant_miss":
            verdict = ("**量化致漏**：PT float 召回 {:.4f} vs int8 {:.4f}"
                       "（差 {:+.4f}）".format(pt["recall"], i8["recall"],
                                              pt["recall"] - i8["recall"]))
        elif e["category"] == "cat3_quant_false_alarm":
            verdict = ("**量化致虚警**：ΔFP {:+d}（int8 {} vs PT {}）"
                       .format(e["delta"]["fp"], i8["fp"], pt["fp"]))
        else:
            verdict = "两臂接近，量化影响不显著"
        lines.append("- `{}`：{}。".format(s, verdict))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
