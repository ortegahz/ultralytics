#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
04_run_board_fast_eval.py -- run the RKNN models on the board and score them
against the PC PyTorch reference, frame by frame, on the SAME bytes.

The comparison this script exists to make
------------------------------------------
    PT  (PC, PyTorch, float)   -- the baseline / reference
    INT8 (board, real NPU)     -- the candidate

Both arms read the SAME BMP files, both decode the peaks with the SAME code
(extract_peaks_np is a 1:1 numpy port of the board's C++ extract_peaks), both
match against the SAME GT with the SAME rule. The only difference between the
two arms is the arithmetic inside the network -- i.e. the quantisation loss.

How the board is driven
-----------------------
The board sees the host's NFS export at /mnt/manu, so a BMP staged on the host
at <BOARD_NFS>/ptq/bmp/... is /mnt/manu/ptq/bmp/... on the board. The driver
writes one list.txt per sequence (one line per BMP, the board's absolute path),
runs the binary and pulls the resulting TSV back.

Board output format (produced by manu/pipeline/npu/rknn_heatmap_smoke.cpp):
    <bmp name>\\t<ndet>\\t<x> <y> <score>\\t<x> <y> <score>\\t...
ONE TAB-SEPARATED COLUMN PER DETECTION, with x, y in the 640x640 letterbox
canvas. A frame without any detection is "<bmp name>\\t0".

Usage
-----
    python 04_run_board_fast_eval.py --models m1.rknn m2.rknn \\
        --manifest /mnt/data/siping/ptq_work/fast/001324_sub2000.json \\
        --variant p0_off --out /mnt/data/siping/ptq_work/tier2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# The repo root is parents[4] (ptq -> npu -> pipeline -> manu -> <root>). With
# only four ".parent" hops the path resolved to <root>/manu, and running this
# file directly (python manu/pipeline/npu/ptq/04_....py) died with
# ModuleNotFoundError: No module named 'manu'.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

BOARD_HOST = "ptq-board"   # ssh alias on the server: key-based, no password
BOARD_NFS_MOUNT = "/mnt/manu/ptq"     # how the board sees the host staging root
BOARD_BIN = "/root/rknn_heatmap_smoke"
BOARD_WORK = "/root/ptq"              # scratch on the board's own filesystem
# The per-sequence board run is HARD CAPPED.  A bad model once made the board
# hang for 26 minutes without returning (manu/memory/PTQ_RK3588_int8_报告.md
# §7, the p0_on int8 arm), and an uncapped ssh would wait for it every time.
# 900 s is ~20x the measured cost of the largest sequence (the board runs at
# ~9.3 fps, the NFS read dominates), so it never fires on a healthy model.
BOARD_RUN_TIMEOUT = 900
# The scratch directory MUST be per-run.  The board is driven over ssh as
# `bash -c '/root/rknn_heatmap_smoke ...'`, so two concurrent drivers used to
# share /root/ptq and therefore the same det_<seq>.tsv path: one driver
# deleted the file the other one was still writing, and the victim failed with
# "cannot pull the detection TSV ... No such file or directory".
#
# The tag alone is NOT enough: the two selection axes evaluate the SAME model
# at the same time (axis1 = 001324, axis2 = the golden pool), so the tag is
# identical for them.  The PID is appended, which makes every concurrent run
# own its directory.  The board's root filesystem has only ~1.6 GB free, so the
# directory is emptied and removed again at the end of the run.
def board_work_dir(tag: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in tag)
    return f"/root/ptq_{safe}_{os.getpid()}"


def sh(cmd: list[str], timeout: int = 3600) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def parse_board_tsv(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    """<name>\\t<ndet>\\t<x> <y> <score>\\t<x> <y> <score> ... -> {name: [(x,y,score), ...]}

    The board writes ONE TAB-SEPARATED COLUMN PER DETECTION
    (manu/pipeline/npu/rknn_heatmap_smoke.cpp:688-691):

        detf << base << "\\t" << dets.size();
        for (size_t k = 0; k < dets.size(); k++)
          detf << "\\t" << dets[k].x << " " << dets[k].y << " " << dets[k].score;

    so the detections do NOT all live in a single column. Reading only column 2
    (and then breaking out of the loop as soon as that column ran out of tokens)
    kept the FIRST detection of every frame and silently threw the rest away,
    which made the board arm's FP / recall / F1 meaningless. A zero-detection
    line is "<name>\\t0" and must stay an empty detection list.
    """
    out: dict[str, list[tuple[float, float, float]]] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[0]
        n = int(parts[1]) if parts[1].strip().isdigit() else 0
        dets = []
        for k in range(n):
            if 2 + k >= len(parts):
                break
            fl = parts[2 + k].split()
            if len(fl) < 3:
                break
            try:
                dets.append((float(fl[0]), float(fl[1]), float(fl[2])))
            except ValueError:
                break
        out[name] = dets
    return out


# The board prints the model's tensor attributes once per model load:
#     input dims=[1,640,640,3] fmt=NHWC type=INT8
#     out[0] dims=[1,1,320,320] fmt=NCHW type=INT8 zp=-128 scale=0.00392157
# They are archived in the report: a broken arm (zero detections) is usually a
# wrong quantisation / zero-point / layout, and that is visible right here.
_RE_INPUT_ATTR = re.compile(r"input dims=\[([^\]]*)\]\s*fmt=(\S+)\s+type=(\S+)")
_RE_OUT_ATTR = re.compile(
    r"out\[(\d+)\] dims=\[([^\]]*)\]\s*fmt=(\S+)\s+type=(\S+)\s+zp=(-?\d+)\s+scale=([0-9eE+\-.]+)")


def parse_board_attrs(stdout: str) -> dict:
    """Extract the input/output tensor attributes from the board's stdout."""
    out: dict = {"input_attr": None, "out_attrs": []}
    text = stdout or ""
    m = _RE_INPUT_ATTR.search(text)
    if m:
        out["input_attr"] = {
            "dims": [int(x) for x in m.group(1).split(",") if x.strip()],
            "fmt": m.group(2), "type": m.group(3), "raw": m.group(0).strip()}
    for mo in _RE_OUT_ATTR.finditer(text):
        out["out_attrs"].append({
            "index": int(mo.group(1)),
            "dims": [int(x) for x in mo.group(2).split(",") if x.strip()],
            "fmt": mo.group(3), "type": mo.group(4),
            "zp": int(mo.group(5)), "scale": float(mo.group(6)),
            "raw": mo.group(0).strip()})
    return out


def score_rows(det_by_name: dict[str, list[tuple[float, float, float]]],
                rows: list[dict], src_hw: dict[str, tuple[int, int]]) -> dict:
    """Greedy distance matching against the GT, per sequence and overall."""
    import cv2

    tp = fp = fn = 0
    n_det = 0
    per_seq: dict[str, dict] = {}
    for r in rows:
        seq = r["seq"]
        name = Path(r["bmp"]).name
        dets = det_by_name.get(name, [])
        n_det += len(dets)
        h, w = src_hw[seq]
        pts = np.asarray([[d[0], d[1]] for d in dets], np.float32).reshape(-1, 2)
        scores = np.asarray([d[2] for d in dets], np.float32).reshape(-1)
        # canvas -> source pixels
        geom = C.letterbox_geometry((h, w))
        pts_src = C.canvas_to_src(pts, geom) if len(pts) else pts
        # GT is normalised by the SOURCE image size
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
        tp += a; fp += b; fn += c
        s = per_seq.setdefault(seq, {"tp": 0, "fp": 0, "fn": 0, "det": 0, "frames": 0})
        s["tp"] += a; s["fp"] += b; s["fn"] += c; s["det"] += len(dets); s["frames"] += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "det": n_det,
            "precision": prec, "recall": rec, "f1": f1, "per_sequence": per_seq}


# --------------------------------------------------------------------------------------
# The PT reference arm -- the same bytes, the same decoder, on the PC
# --------------------------------------------------------------------------------------
def run_pt_reference(rows: list[dict], src_hw: dict[str, tuple[int, int]],
                     variant: str, topk: int = 100, batch: int = 8,
                     device: str = "cuda:0") -> dict:
    """PyTorch float reference on exactly the BMPs the board will read."""
    import torch

    from manu.pipeline.npu.hm_export_onnx import build_model

    model, ckpt = build_model(str(C.WEIGHTS), ablate_p0=C.VARIANTS[variant]["ablate_p0"])
    assert not model.training, "the PT reference must run in eval mode"
    stride = ckpt.get("stride", 2)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    det_by_name: dict[str, list[tuple[float, float, float]]] = {}
    model = model.to(dev).eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            xs = np.concatenate([C.bmp_to_input_nchw_float(r["bmp"]) for r in chunk], axis=0)
            out = model(torch.from_numpy(xs).to(dev))
            for j, r in enumerate(chunk):
                hm = out["heatmap"][j, 0].float().cpu().numpy()
                off = out["offset"][j].float().cpu().numpy()
                det = C.extract_peaks_np(hm, off, stride=stride,
                                         conf_thresh=C.TH_BASE, top_k=topk)
                det_by_name[Path(r["bmp"]).name] = [
                    (float(p[0]), float(p[1]), float(s))
                    for p, s in zip(det["points"], det["scores"])]
    return det_by_name


# --------------------------------------------------------------------------------------
# Board driver
# --------------------------------------------------------------------------------------
def run_on_board(model_path: Path, model_md5: str, rows: list[dict],
                 board_nfs: Path, th: float, topk: int, tag: str,
                 board_bin: str = BOARD_BIN,
                 board_models_dir: str = "/mnt/manu/ptq/models",
                 timeout: int = BOARD_RUN_TIMEOUT,
                 board_cpu: str = "6") -> dict:
    """Stage the lists on the NFS share, run the board binary, pull the TSVs back."""
    out_dir = Path("/tmp/ptq_board_out") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # The server's own path is NOT visible to the board. The models are staged on
    # the host's NFS export under <staging>/models/, which the board sees at
    # /mnt/manu/ptq/models/. Map by file name so the two sides never have to
    # agree on an absolute prefix.
    board_model = f"{board_models_dir}/{model_path.name}"

    # group the rows per sequence
    per_seq: dict[str, list[dict]] = {}
    for r in rows:
        per_seq.setdefault(r["seq"], []).append(r)

    # the board's absolute path for each BMP: the host path is <BOARD_NFS>/... and
    # the board sees the same tree under /mnt/manu
    rel_root = board_nfs
    # The board reads its frames from the NFS export, and the export holds TWO
    # pools at DIFFERENT paths:
    #   the Longquan BMPs      -> /mnt/manu/ptq/bmp/<seq>/<seq>__<id>.bmp
    #   the Anti-UAV val BMPs  -> /mnt/manu/pcfeat_bmp/images/<seq>__<id>.bmp
    # A manifest row that is not under C.BMP_ROOT used to fall through to its
    # SERVER-side absolute path, which the board cannot read at all -- the run
    # then "succeeded" with 0 frames and 0 detections on every sequence.
    BOARD_POOLS = [
        (C.BMP_ROOT, f"{BOARD_NFS_MOUNT}/bmp"),
        (C.WORK_ROOT / "pcfeat_bmp" / "images", "/mnt/manu/pcfeat_bmp/images"),
    ]
    lists = []
    unmapped = 0
    for seq, srows in per_seq.items():
        lp = out_dir / f"list_{seq}.txt"
        lines = []
        for r in sorted(srows, key=lambda x: x["frame_id"]):
            bmp = Path(r["bmp"])
            # the server keeps the BMPs under one of the known roots; the board
            # needs the same tree re-anchored at its own mount point
            mapped = None
            for root, board_root in BOARD_POOLS:
                try:
                    rel = bmp.relative_to(root)
                except ValueError:
                    continue
                mapped = f"{board_root}/{rel}"
                break
            if mapped is None:
                unmapped += 1
                mapped = str(bmp)
            lines.append(mapped)
        if unmapped:
            print(f"  [warn] {unmapped} frame path(s) are not under a known pool "
                  f"(the board cannot read server-side paths)", flush=True)
        lp.write_text("\n".join(lines) + "\n")
        lists.append((seq, lp, len(lines)))

    # copy the lists onto the board's own filesystem (small) and run there
    board_work = board_work_dir(tag)
    rc, out, err = sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_HOST,
                       "mkdir", "-p", board_work])
    if rc != 0:
        raise SystemExit(f"[FATAL] cannot reach the board: {err[:300]}")

    # ---- the model that RUNS on the board must be the model we SCORE ----
    # The board reads the model from the NFS staging area; the server may have
    # rebuilt it since the last staging. A stale copy would make the board run
    # an OLD model while the report quotes the NEW model's md5 -- the numbers
    # would then be attributed to the wrong model.
    rc, out, err = sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_HOST,
                       f"md5sum {board_model}"])
    board_md5 = out.split()[0] if (rc == 0 and out.split()) else ""
    if not board_md5:
        raise SystemExit(
            f"[FATAL] cannot read the model md5 on the board: {board_model}\n"
            f"       rc={rc} err={err[:200]} out={out[:200]}")
    if board_md5 != model_md5:
        raise SystemExit(
            f"[FATAL] model md5 MISMATCH -- the board would run a different model\n"
            f"       than the one being scored.\n"
            f"       server: {model_md5}  {model_path}\n"
            f"       board : {board_md5}  {board_model}\n"
            f"       re-stage the model into {board_models_dir} and retry.")
    print(f"  [{tag}] board model md5 verified: {board_md5}", flush=True)

    results = {}
    t0 = time.time()
    total = 0
    n_det_total = 0
    stdout_head = ""
    stderr_head = ""
    for seq, lp, n in lists:
        total += n
        remote_list = f"{board_work}/{lp.name}"
        rc, out, err = sh(["scp", "-o", "StrictHostKeyChecking=no", str(lp),
                           f"{BOARD_HOST}:{remote_list}"])
        if rc != 0:
            raise SystemExit(f"[FATAL] scp list failed: {err[:300]}")
        remote_tsv = f"{board_work}/det_{seq}.tsv"
        # CPU affinity: rknn_inputs_set's type conversion is a SINGLE-THREADED
        # scalar loop, and it costs 45.41 ms on an A55 (cpu 0-3, 1.8 GHz) but
        # only 11.15 ms on an A76 (cpu 4-7, 2.35 GHz) -- 4.1x for free. taskset
        # is used instead of a binary flag so this works with the CERTIFIED
        # board binary as well. Measured 2026-10-23, see manu/memory/
        # fp16_input_io_mem.md (the zero-copy alternative was a net LOSS: the
        # runtime moves the staging into rknn_run at ~180 MB/s).
        pin = f"taskset -c {board_cpu} " if board_cpu else ""
        cmd = (f"{pin}{board_bin} {board_model} --batch {remote_list} --th {th} "
               f"--topk {topk} --detections-out {remote_tsv}")
        rc, out, err = sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_HOST,
                           f"rm -f {remote_tsv}; {cmd}"], timeout=timeout)
        # the board prints the model's tensor attributes once per model load
        if not stdout_head and out:
            stdout_head = out[:8000]
        if not stderr_head and err:
            stderr_head = err[:2000]
        # pull the TSV back
        rc2, o2, e2 = sh(["scp", "-o", "StrictHostKeyChecking=no",
                          f"{BOARD_HOST}:{remote_tsv}", str(out_dir / f"det_{seq}.tsv")])
        if rc2 != 0:
            raise SystemExit(f"[FATAL] cannot pull the detection TSV {remote_tsv}: "
                             f"{e2[:300]}")
        tsv = out_dir / f"det_{seq}.tsv"
        dets = parse_board_tsv(tsv)
        got = len(dets)
        ndet = sum(len(v) for v in dets.values())
        n_det_total += ndet
        status = "OK" if (rc == 0 and got == n) else f"RC{rc}_ROWS{got}_WANT{n}"
        results[seq] = {"status": status, "want": n, "got": got, "det": ndet}
        print(f"  [{tag}] {seq}: {status} ({got}/{n} frames, {ndet} detections)",
              flush=True)
    # The board's root filesystem has only ~1.6 GB free, so the per-run
    # scratch directory is emptied and removed again.  Without this the
    # per-PID directories would pile up over a long night.
    sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_HOST,
        f"find {board_work} -mindepth 1 -delete 2>/dev/null; rmdir {board_work} 2>/dev/null; true"])
    elapsed = time.time() - t0
    print(f"  [{tag}] total {total} frames in {elapsed:.1f}s "
          f"({total / max(1e-9, elapsed):.2f} fps)", flush=True)
    run = {"per_sequence": results, "seconds": elapsed, "frames": total,
           "fps": total / max(1e-9, elapsed),
           "detections": n_det_total,
           "model": str(model_path), "model_md5": model_md5,
           "board_model": board_model, "board_model_md5": board_md5,
           "th": th, "topk": topk}
    run.update(parse_board_attrs(stdout_head))
    if stdout_head:
        run["board_stdout"] = stdout_head
    if stderr_head:
        run["board_stderr"] = stderr_head
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="RKNN model paths")
    ap.add_argument("--manifest", required=True, help="001324_sub2000.json / golden_pool_sub1200.json")
    ap.add_argument("--variant", default="p0_off", choices=list(C.VARIANTS))
    ap.add_argument("--out", default=str(C.WORK_ROOT / "tier2"))
    ap.add_argument("--board-nfs", default=str(C.BOARD_NFS))
    ap.add_argument("--th", type=float, default=C.TH_BASE)
    ap.add_argument("--topk", type=int, default=100,
                    help="detections kept per frame; MUST match between the PT and "
                         "the board arm, otherwise FP is not comparable")
    ap.add_argument("--board-bin", default=BOARD_BIN)
    ap.add_argument("--board-cpu", default="6",
                    help="CPU list the board run is pinned to (default 6 = an A76 "
                         "big core; empty string disables the pinning)")
    ap.add_argument("--board-models-dir", default="/mnt/manu/ptq/models",
                    help="where the models live ON THE BOARD (host NFS export)")
    ap.add_argument("--board-timeout", type=int, default=BOARD_RUN_TIMEOUT,
                    help="hard cap on ONE board sequence run, in seconds")
    ap.add_argument("--skip-pt", action="store_true", help="board only, no PT reference")
    ap.add_argument("--pt-device", default="cuda:0")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = C.read_json(args.manifest)
    # a manifest may be {"items": [...]} or a bare list of frame rows
    if isinstance(manifest, list):
        rows, mname = manifest, Path(args.manifest).stem
    else:
        rows, mname = manifest["items"], manifest.get("name", Path(args.manifest).stem)
    print(f"[manifest] {args.manifest}: {len(rows)} frames ({mname})")

    # the source size per sequence, read from the staged BMPs (they are the truth)
    src_hw: dict[str, tuple[int, int]] = {}
    for r in rows:
        seq = r["seq"]
        if seq not in src_hw:
            import cv2
            img = cv2.imread(r["bmp"], cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"[FATAL] cannot read staged bmp: {r['bmp']}")
            src_hw[seq] = (img.shape[0], img.shape[1])
    print(f"[geom] {len(src_hw)} sequences, sizes "
          f"{sorted({v for v in src_hw.values()})}")

    report = {"manifest": args.manifest, "manifest_name": mname,
              "variant": args.variant, "th": args.th, "topk": args.topk,
              "frames": len(rows), "arms": {}}

    # ---------------- PT reference arm ----------------
    if not args.skip_pt:
        print("[PT] running the PyTorch reference on the same BMPs ...", flush=True)
        t0 = time.time()
        det_by_name = run_pt_reference(rows, src_hw, args.variant, topk=args.topk,
                                      device=args.pt_device)
        pt_score = score_rows(det_by_name, rows, src_hw)
        pt_score["seconds"] = round(time.time() - t0, 1)
        report["arms"]["pt_float"] = pt_score
        print(f"[PT] F1={pt_score['f1']:.4f} R={pt_score['recall']:.4f} "
              f"P={pt_score['precision']:.4f} FP={pt_score['fp']} FN={pt_score['fn']} "
              f"({pt_score['seconds']}s)")

    # ---------------- board arms ----------------
    for mp in args.models:
        mpath = Path(mp)
        C.require(mpath, "RKNN model")
        md5 = C.md5_file(mpath)
        tag = mpath.stem
        print(f"\n[board] {tag}  md5={md5}", flush=True)
        # the board's copy of the model is resolved inside run_on_board (and its
        # md5 is checked there) -- the server path is not visible to the board.
        board_run = run_on_board(mpath, md5, rows, C.BOARD_NFS, args.th,
                                 args.topk, tag, args.board_bin,
                                 board_models_dir=args.board_models_dir,
                                 timeout=args.board_timeout,
                                 board_cpu=args.board_cpu)
        # pull the TSVs and score
        det_by_name: dict[str, list[tuple[float, float, float]]] = {}
        det_dir = Path("/tmp/ptq_board_out") / tag
        for tsv in sorted(det_dir.glob("det_*.tsv")):
            det_by_name.update(parse_board_tsv(tsv))
        score = score_rows(det_by_name, rows, src_hw)
        score["board_run"] = board_run
        # A model that detects NOTHING is broken, not "quantised badly". Reporting
        # it as F1=0 reads like a quantisation loss and hides a dead model.
        zero_det = score["det"] == 0
        if zero_det:
            score["status"] = "ZERO_DETECTIONS"
            board_run["status"] = "ZERO_DETECTIONS"
            print(f"[FATAL] board arm {tag}: 0 detections over {len(rows)} frames -- "
                  f"this arm is INVALID (a broken model, NOT a quantisation loss).",
                  flush=True)
            print(f"        board input attr: {board_run.get('input_attr')}", flush=True)
            print(f"        board out  attrs: {board_run.get('out_attrs')}", flush=True)
            if board_run.get("board_stderr"):
                print(f"        board stderr    : "
                      f"{board_run['board_stderr'][:400]}", flush=True)
        report["arms"][f"board_{tag}"] = score
        if zero_det:
            print(f"[board] {tag}: [FATAL] ZERO_DETECTIONS -- no F1 is reported for "
                  f"this arm (the other arms are still scored)")
        else:
            print(f"[board] {tag}: F1={score['f1']:.4f} R={score['recall']:.4f} "
                  f"P={score['precision']:.4f} FP={score['fp']} FN={score['fn']}")

    # ---------------- arbitration ----------------
    if "pt_float" in report["arms"]:
        pt = report["arms"]["pt_float"]
        print("\n=== quantisation loss vs the PT reference ===")
        invalid = [n for n, a in report["arms"].items()
                   if n != "pt_float" and a.get("status") == "ZERO_DETECTIONS"]
        if invalid:
            print(f"[FATAL] {len(invalid)} arm(s) produced ZERO detections and are "
                  f"INVALID (excluded below, NOT a quantisation loss): "
                  f"{', '.join(invalid)}")
        print(f"{'arm':38s} {'F1':>8s} {'dF1':>9s} {'Recall':>8s} {'dRecall':>9s} "
              f"{'FP':>7s} {'dFP':>6s} {'fps':>7s}")
        print(f"{'PT float (reference)':38s} {pt['f1']:8.4f} {'--':>9s} "
              f"{pt['recall']:8.4f} {'--':>9s} {pt['fp']:7d} {'--':>6s} {'--':>7s}")
        for name, arm in report["arms"].items():
            if name == "pt_float":
                continue
            if arm.get("status") == "ZERO_DETECTIONS":
                continue        # already reported as [FATAL] above
            fps = (arm.get("board_run") or {}).get("fps")
            print(f"{name:38s} {arm['f1']:8.4f} {arm['f1'] - pt['f1']:+9.4f} "
                  f"{arm['recall']:8.4f} {arm['recall'] - pt['recall']:+9.4f} "
                  f"{arm['fp']:7d} {arm['fp'] - pt['fp']:+6d} "
                  f"{(f'{fps:.2f}' if fps else '--'):>7s}")

    # The report name carries the manifest AND the models: "<manifest>_<variant>_
    # <model stems>_report.json". A single name per manifest made every model /
    # every manifest overwrite the previous report.
    model_tags = "_".join(Path(m).stem for m in args.models) or "nomodel"
    report_path = out_root / (f"{Path(args.manifest).stem}_{args.variant}_"
                               f"{model_tags}_report.json")
    C.write_json(report_path, report)
    print(f"\n[report] -> {report_path}")


if __name__ == "__main__":
    main()
