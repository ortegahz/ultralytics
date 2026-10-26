#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
20_finalize_and_launch.py -- pick the final int8 model AND start the one and
only full-scale run, with NO dependency on the workstation.

WHY THIS SCRIPT EXISTS
----------------------
The workstation (the "host") is switched off at 18:30.  That kills the NFS
export the board used to read the frames from, and it kills the session that
started all of this.  The SERVER and the BOARD, however, stay up all night.

So the last mile of the pipeline must not live in anybody's shell: this script
runs on the server, waits for the selection sweep to finish, decides the
final model by a written-down rule, and launches the full-scale run by
itself.  If the workstation dies at 18:30 the experiment still finishes.

WHAT IT DOES
------------
  1. wait for 15_night_selection.sh to exit and to have written
     <selection>/best_model.json;
  2. re-score EVERY candidate's board detections at every threshold of
     interest (offline, on the TSVs the sweep left in
     /tmp/ptq_board_out/<tag>/) -- the sweep itself was run at th=0.3 only;
  3. apply the selection rule (below) and write final_choice.json;
  4. launch 14_run_full_scale_nofs.py on the three tracks with the winner.

THE SELECTION RULE (deployment口径, written down so it is not a judgement call)
-------------------------------------------------------------------------------
  * REJECT any candidate whose heatmap output ceiling (255 * out[0] scale) is
    <= 0.30: its peaks are clipped below the decision threshold, so its F1 is
    a broken-model artefact, not a quantisation result.
  * primary key : the candidate's best F1 over the threshold sweep on the
    GOLDEN POOL (axis 2, 1197 frames).  The golden pool is the selection axis;
    the blind track's 11,810 frames are reserved for the final model.
  * tie-break 1 : the best F1 over the same sweep on 001324/2000 frames
    (axis 1, the comparative/diagnostic axis).
  * tie-break 2 : the F1 at the single fixed threshold 0.30 on the golden pool
    (the strict same-threshold quantisation-loss number).
  * tie-break 3 : the output ceiling.
  * tie-break 4 : the model name (stable, deterministic).
  * The best threshold of the winner is recorded and reported, because the
    int8 optimum is measurably HIGHER than 0.30 (the quantisation shifts the
    score distribution), and the board-side threshold has to follow it.

THE FULL-SCALE RUN
-------------------
  * collection threshold stays 0.30 on purpose: that is the threshold every
    earlier experiment in this study used, so the full-scale quantisation
    loss is directly comparable with the whole chain of measurements.  The
    detections below 0.30 are irrelevant anyway, because the winner's
    optimal threshold is ABOVE 0.30.
  * the PT reference arm runs on cuda:1, because cuda:0 is reserved for the
    QAT training that runs in parallel.
  * 14_run_full_scale_nofs.py is resumable and re-entrant (it flock()s its
    own output dir), so a re-run picks up where it stopped.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent

# --- load the sibling modules whose filenames start with a digit ------------
def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_s19 = _load("ptq_19", "19_board_threshold_sweep.py")
_s04 = _load("ptq_04", "04_run_board_fast_eval.py")

SWEEP_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
REJECT_CEILING = 0.30


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_selection(selection_dir: Path, timeout_s: int) -> bool:
    """Wait for the selection sweep to finish and to have written a winner."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        running = subprocess.run(
            ["bash", "-c", "ps -eo args | grep -c '[1]5_night_selection[.]sh'"],
            capture_output=True, text=True).stdout.strip()
        if running in ("0", "") and (selection_dir / "best_model.json").is_file():
            log(f"selection sweep finished (waited {int(time.time()-t0)}s)")
            return True
        time.sleep(20)
    log("TIMEOUT waiting for the selection sweep")
    return False


def sweep_one(tag: str, manifest: Path) -> list[dict]:
    """Offline threshold sweep of one candidate's board detections."""
    tsv_dir = Path("/tmp/ptq_board_out") / tag
    if not tsv_dir.is_dir():
        return []
    det_by_name: dict = {}
    for tsv in sorted(tsv_dir.glob("*.tsv")):
        det_by_name.update(_s19.parse_board_tsv(tsv))
    if not det_by_name:
        return []
    rows, _ = _s19.load_manifest(str(manifest))
    src_hw = _s19.read_src_hw(rows)
    return _s19.sweep(det_by_name, rows, src_hw, SWEEP_THRESHOLDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-dir", default="/mnt/data/siping/ptq_work/selection")
    ap.add_argument("--work", default="/mnt/data/siping/ptq_work")
    ap.add_argument("--axis1", default="/mnt/data/siping/ptq_work/fast/001324_sub2000.json")
    ap.add_argument("--axis2", default="/mnt/data/siping/ptq_work/fast/golden_pool_sub1200.json")
    ap.add_argument("--wait-timeout", type=int, default=7200)
    ap.add_argument("--no-launch", action="store_true",
                    help="only decide, do not start the full-scale run")
    ap.add_argument("--pt-device", default="cuda:1")
    args = ap.parse_args()

    sel_dir = Path(args.selection_dir)
    work = Path(args.work)
    out_dir = work / "final_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wait_for_selection(sel_dir, args.wait_timeout):
        raise SystemExit("[FATAL] the selection sweep did not finish in time")

    models: list[dict] = []
    for line in (sel_dir / "model_list.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            models.append({"server_path": parts[0], "board_name": parts[1]})

    # the ceiling + the fixed-threshold numbers come from the sweep reports
    table = json.loads((sel_dir / "selection_table.json").read_text())
    ceiling = {r["model"]: r.get("out_ceiling") for r in table.get("rows", [])}
    f1_30 = {(r["model"], "axis1"): r.get("axis1_f1") for r in table.get("rows", [])}
    f1_30.update({(r["model"], "axis2"): r.get("axis2_f1") for r in table.get("rows", [])})

    rows_out = []
    for m in models:
        tag = Path(m["server_path"]).stem
        log(f"sweeping thresholds for {tag} ...")
        c1 = sweep_one(tag, Path(args.axis1))
        c2 = sweep_one(tag, Path(args.axis2))
        best1 = max(c1, key=lambda c: c["f1"]) if c1 else None
        best2 = max(c2, key=lambda c: c["f1"]) if c2 else None
        rows_out.append({
            "model": tag,
            "server_path": m["server_path"],
            "board_name": m["board_name"],
            "out_ceiling": ceiling.get(tag),
            "axis1_best": best1, "axis2_best": best2,
            "axis1_curve": c1, "axis2_curve": c2,
            "axis1_f1_at_030": f1_30.get((tag, "axis1")),
            "axis2_f1_at_030": f1_30.get((tag, "axis2")),
        })
        log(f"  {tag}: axis1 best F1={best1['f1']:.4f}@th={best1['th']} | "
            f"axis2 best F1={best2['f1']:.4f}@th={best2['th']}" if best1 and best2
            else f"  {tag}: no TSV to sweep")

    def key(r: dict):
        c = r.get("out_ceiling")
        if c is not None and c <= REJECT_CEILING:
            return (-1, -1, -1, -1, r["model"])      # rejected outright
        b2 = r.get("axis2_best")
        b1 = r.get("axis1_best")
        return (
            b2["f1"] if b2 else -1,
            b1["f1"] if b1 else -1,
            r.get("axis2_f1_at_030") or -1,
            c if c is not None else -1,
            r["model"],
        )

    rejected = [r["model"] for r in rows_out
                if r.get("out_ceiling") is not None
                and r["out_ceiling"] <= REJECT_CEILING]
    alive = [r for r in rows_out if r["model"] not in rejected]
    alive.sort(key=key, reverse=True)
    winner = alive[0] if alive else None

    choice = {
        "rule": {
            "reject_if_out_ceiling_le": REJECT_CEILING,
            "primary": "best-over-threshold F1 on the golden pool (axis2)",
            "tie_breakers": ["best-over-threshold F1 on axis1",
                             "F1 at th=0.30 on axis2", "out ceiling", "name"],
            "sweep_thresholds": SWEEP_THRESHOLDS,
        },
        "rejected": rejected,
        "rows": rows_out,
        "winner": winner,
    }
    (out_dir / "final_choice.json").write_text(
        json.dumps(choice, ensure_ascii=False, indent=1))

    print("\n=== final selection (deployment口径: best F1 over the threshold sweep) ===")
    print(f"{'model':<26}{'ax2_best':>10}{'@th':>6}{'ax1_best':>10}{'@th':>6}"
          f"{'ax2@.30':>10}{'ceiling':>9}")
    for r in rows_out:
        b2 = r.get("axis2_best") or {}
        b1 = r.get("axis1_best") or {}
        print(f"{r['model']:<26}{b2.get('f1', float('nan')):>10.4f}"
              f"{b2.get('th', 0):>6.2f}{b1.get('f1', float('nan')):>10.4f}"
              f"{b1.get('th', 0):>6.2f}"
              f"{(r.get('axis2_f1_at_030') or float('nan')):>10.4f}"
              f"{(r.get('out_ceiling') or float('nan')):>9.4f}")
    for m in rejected:
        print(f"  REJECTED {m}: out ceiling <= {REJECT_CEILING}")
    if not winner:
        raise SystemExit("[FATAL] no candidate survived the selection rule")
    print(f"\n[winner] {winner['model']}  "
          f"golden-pool F1={winner['axis2_best']['f1']:.4f} @ th={winner['axis2_best']['th']}")

    if args.no_launch:
        return

    # ---------------- launch the one and only full-scale run ----------------
    full_out = work / "full_scale"
    cmd = [
        sys.executable, "-u", str(HERE / "14_run_full_scale_nofs.py"),
        "--tracks", "blind", "golden", "anti",
        "--model", winner["server_path"],
        "--frames-root", str(work / "bmp"),
        "--frames-root", f"anti={work / 'pcfeat_bmp' / 'images'}",
        "--stage-dir", "/userdata/ptq_stage",
        "--chunk", "500", "--th", "0.30", "--topk", "100",
        "--out", str(full_out),
        "--pt-device", args.pt_device,
    ]
    log("launching the full-scale run:")
    log(" ".join(cmd))
    logfile = work / "logs" / "full_scale.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True,
                                cwd="/tmp/pycharm_project_10ae9e2e")
    (out_dir / "full_scale_pid").write_text(str(proc.pid))
    log(f"full-scale run started, pid={proc.pid}, log={logfile}")


if __name__ == "__main__":
    main()
