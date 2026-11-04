"""Pick the OSD window that shows each arm's failure most clearly.

For one sequence it loads the per-frame predictions of the three arms (stale / replay / original),
computes the centre-location error against the frame's ground-truth centre, and picks the
`--window`-frame CONTIGUOUS run of real frame numbers where the pair named by `--criterion`
separates most.

The three arms are the delivery-candidate comparison:

    stale     manu/tracking/lookahead --lookahead stale  (old logic, hard 320 ms lag)
    replay    WTH15 external replay lookahead
    original  --lookahead off (zero-latency ceiling)

Default arm directories point at the delivery-candidate products under runs/lookahead/; the stale arm
is regenerated per sequence into runs/lookahead/osd/preds_stale_<SEQ>/ because no 24-sequence stale
product exists.

``--criterion base:cand`` maximises mean(cle[base]) - mean(cle[cand]) over the window:

    stale:replay       the old logic is the one that loses the target (high-speed manoeuvre)
    replay:original    the external replay arm drifts onto the wrong thing (dark hover, cloud edge)

``per_frame`` is keyed by the REAL frame number but only exists on the recorded-frame set (candidate
frames plus frames that carry ground truth), which is NOT 0..N-1 in general -- e.g. 001415 has a
1541-frame hole and 002111 a 1212-frame hole. Frames absent from the key set carry no report and are
skipped here; the window is always a contiguous run of real frame numbers.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/tmp/pycharm_project_10ae9e2e")
sys.path.insert(0, str(REPO_ROOT))
from manu.videos.render_pred_gt_osd_video import load_ground_truth  # noqa: E402

FINAL = REPO_ROOT / "runs/lookahead/final"
OSD = REPO_ROOT / "runs/lookahead/osd"
DEFAULT_ARMS = {
    "stale": OSD / "preds_stale_{seq}",
    "replay": FINAL / "preds_final_WTH15_24seq",
    "original": FINAL / "ref_preds_original_v3_24seq",
}


def load_arm(preds_dir: str, sequence: str) -> dict:
    """Read one arm's per-frame predictions into {frame: [(x, y, score), ...]}."""
    path = Path(preds_dir) / f"{sequence}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    per_frame: dict[int, list] = {}
    for key, rows in (data.get("per_frame") or {}).items():
        dets = []
        for row in rows or []:
            pos = row.get("pos")
            if pos is None:
                continue
            dets.append((float(pos[0]), float(pos[1]), float(row.get("score", 0.0))))
        per_frame[int(key)] = dets
    return per_frame


def centre_errors(per_frame: dict, truth: dict) -> dict:
    """Distance from the closest predicted centre to the closest ground-truth centre, per frame."""
    out = {}
    for frame, dets in per_frame.items():
        boxes = truth.get(frame)
        if boxes is None or not len(boxes) or not dets:
            continue
        centres = np.asarray(boxes, np.float64)[:, :2]
        best = None
        for x, y, _score in dets:
            d = float(np.sqrt(((centres - np.asarray([x, y])) ** 2).sum(axis=1)).min())
            best = d if best is None else min(best, d)
        out[frame] = best
    return out


def contiguous_runs(frames: list[int]) -> list[list[int]]:
    """Split a sorted frame list into runs of consecutive frame numbers (a hole breaks a run)."""
    runs, current = [], []
    for frame in frames:
        if current and frame != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(frame)
    if current:
        runs.append(current)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        help="name=preds_dir; defaults to the delivery-candidate stale/replay/original products",
    )
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument(
        "--criterion",
        default="stale:replay",
        help="base:cand -- maximise mean(cle[base]) - mean(cle[cand]) over the window",
    )
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    specs = args.arm or [f"{name}={path}" for name, path in DEFAULT_ARMS.items()]
    arms = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        arms[name] = path.format(seq=args.seq) if "{seq}" in path else path

    base, _, cand = args.criterion.partition(":")
    for name in (base, cand):
        if name not in arms:
            raise SystemExit(f"criterion arm {name!r} not loaded; available: {sorted(arms)}")

    loaded = {name: load_arm(path, args.seq) for name, path in arms.items()}

    frames = sorted({f for pf in loaded.values() for f in pf})
    truth = load_ground_truth(Path(args.labels_root), args.seq, args.width, args.height, set(frames))
    cle = {name: centre_errors(pf, truth) for name, pf in loaded.items()}

    # Only frames on which every arm reported a detection can compare the arms against each other.
    common = sorted(set.intersection(*(set(cle[name]) for name in cle)))
    best = None
    for run in contiguous_runs(common):
        if len(run) < args.window:
            continue  # a shorter run would give a short, unrepresentative clip
        for start in range(0, len(run) - args.window + 1):
            window = run[start : start + args.window]
            b = np.asarray([cle[base][f] for f in window])
            c = np.asarray([cle[cand][f] for f in window])
            # Separation = how much worse the base arm is than the candidate arm over this window.
            score = float(b.mean() - c.mean())
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "start": int(window[0]),
                    "limit": int(args.window),
                    "frames": len(window),
                    "end": int(window[-1]),
                }

    if best is None:
        raise SystemExit(f"no {args.window}-frame contiguous run for {args.seq}")

    window = [f for f in common if best["start"] <= f <= best["end"]]
    stats = {}
    for name, series in cle.items():
        picked = [series[f] for f in window if f in series]
        values = np.asarray(picked) if picked else np.asarray([])
        stats[name] = {
            "frames": int(len(values)),
            "cle_mean": float(values.mean()) if len(values) else None,
            "cle_p50": float(np.percentile(values, 50)) if len(values) else None,
            "cle_p90": float(np.percentile(values, 90)) if len(values) else None,
            "cle_p95": float(np.percentile(values, 95)) if len(values) else None,
            "cle_max": float(values.max()) if len(values) else None,
            "hit_5px": float((values <= 5.0).mean()) if len(values) else None,
            "hit_inbox_8px": float((values <= 8.0).mean()) if len(values) else None,
        }

    result = {
        "sequence": args.seq,
        "window": args.window,
        "criterion": f"{base}:{cand}",
        "base_arm": base,
        "candidate_arm": cand,
        "arm_dirs": arms,
        "start": best["start"],
        "end": best["end"],
        "limit": best["limit"],
        "frames_compared": best["frames"],
        "separation_score": best["score"],
        "per_arm": stats,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
