#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Reconcile the BOARD 24-sequence run against the SERVER reference of the same parameter group.

The board runs the *mosaic* chain on the RK3588 NPU; the server publishes the *multiscale* chain
predictions for the very same 24 sequences under the delivered parameter group ``cc2_md10``
(``optgapfit``).  Both sides are scored against the SAME ground truth with the SAME matcher
(``count_matches(..., 0.0, 8.0, "inbox")``) over the SAME frame window, so the numbers are
directly comparable.

The global row follows the delivery convention of
``manu/evaluation/system_sota_longquan.py``: the per-sequence COUNT KEYS are summed and
``metrics()`` is recomputed ONCE on the summed counts (per-sequence F1 is never averaged).
The FAR denominator is the total frame count of the 24 sequences (165,223 for the full set).

Usage::

    python board_vs_server_24seq.py \
        --board-tsv-dir  /mnt/data/siping/runs/trial0080_board24_20261120/tsv \
        --server-preds-dir /mnt/data/siping/runs/trial0080_onnx/eval/shortlist/preds_cc2_md10 \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --total-frames-json /mnt/data/siping/runs/trial0080_board24_20261120/total_frames.json \
        --server-global /mnt/data/siping/runs/trial0080_onnx/eval/track_all_v3_optgapfit.json \
        --out /mnt/data/siping/runs/trial0080_board24_20261120/board_vs_server_24seq.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = os.environ.get(
    "ULTRALYTICS_ROOT", "/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics"
)
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.board_far_parity_check import (  # noqa: E402
    parse_board_tsv,
    parse_server_preds,
    score_side,
)
from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    COUNT_KEYS,
    load_ground_truth,
    metrics,
)


def aggregate(rows: list[dict], num_frames: int) -> dict:
    """Sum the per-sequence counts and recompute the metrics once (no per-sequence F1 averaging)."""
    counts = {key: int(sum(row["board"][key] for row in rows)) for key in COUNT_KEYS}
    return metrics(counts, num_frames)


def aggregate_server(rows: list[dict], num_frames: int) -> dict:
    counts = {key: int(sum(row["server"][key] for row in rows)) for key in COUNT_KEYS}
    return metrics(counts, num_frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-tsv-dir", required=True, type=Path)
    parser.add_argument("--server-preds-dir", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--total-frames-json", type=Path, default=None)
    parser.add_argument("--server-global", type=Path, default=None,
                        help="Official server 24-seq json; used as an aggregation self-check.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tol", type=float, default=8.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    totals: dict[str, int] = {}
    if args.total_frames_json:
        totals = json.loads(args.total_frames_json.read_text(encoding="utf-8"))

    seqs = sorted(p.name[: -len("_fused.tsv")] for p in args.board_tsv_dir.glob("*_fused.tsv"))
    if not seqs:
        print(f"[ERR] no <SEQ>_fused.tsv under {args.board_tsv_dir}")
        return 1

    rows: list[dict] = []
    for seq in seqs:
        board_tsv = args.board_tsv_dir / f"{seq}_fused.tsv"
        server_json = args.server_preds_dir / f"{seq}.json"
        if not server_json.is_file():
            print(f"[ERR] missing server reference for {seq}: {server_json}")
            return 1
        truth = load_ground_truth(args.labels_root, seq, args.width, args.height)
        board = parse_board_tsv(board_tsv)
        server = parse_server_preds(server_json)

        n_frames = int(totals.get(seq, max([0, *board.keys(), *server.keys(), *truth.keys()]) + 1))
        window = (0, n_frames)
        board_m = score_side(board, truth, window, args.tol)
        server_m = score_side(server, truth, window, args.tol)
        rows.append({
            "sequence": seq,
            "frames": n_frames,
            "gt_frames": len(truth),
            "board": board_m,
            "server": server_m,
            "d_f1": board_m["f1"] - server_m["f1"],
            "d_far": board_m["far_per_frame"] - server_m["far_per_frame"],
            "d_tp": board_m["tp"] - server_m["tp"],
            "d_fp": board_m["fp"] - server_m["fp"],
            "d_fn": board_m["fn"] - server_m["fn"],
        })
        print(f"[SEQ] {seq:<30} frames={n_frames:<6} "
              f"board f1={board_m['f1']:.7f} far={board_m['far_per_frame']:.7f} "
              f"tp={board_m['tp']} fp={board_m['fp']} fn={board_m['fn']} | "
              f"server f1={server_m['f1']:.7f} far={server_m['far_per_frame']:.7f} "
              f"tp={server_m['tp']} fp={server_m['fp']} fn={server_m['fn']} | "
              f"dF1={rows[-1]['d_f1']:+.7f} dFAR={rows[-1]['d_far']:+.7f}")

    num_frames = sum(row["frames"] for row in rows)
    g_board = aggregate(rows, num_frames)
    g_server = aggregate_server(rows, num_frames)

    print("=" * 118)
    print(f"[GLOBAL] frames={num_frames}  seqs={len(rows)}")
    print(f"[BOARD ] tp={g_board['tp']} fp={g_board['fp']} fn={g_board['fn']} gt={g_board['gt']} "
          f"recall={g_board['recall']:.7f} precision={g_board['precision']:.7f} "
          f"f1={g_board['f1']:.10f} far_per_frame={g_board['far_per_frame']:.10f}")
    print(f"[SERVER] tp={g_server['tp']} fp={g_server['fp']} fn={g_server['fn']} gt={g_server['gt']} "
          f"recall={g_server['recall']:.7f} precision={g_server['precision']:.7f} "
          f"f1={g_server['f1']:.10f} far_per_frame={g_server['far_per_frame']:.10f}")
    print(f"[DELTA ] df1={g_board['f1'] - g_server['f1']:+.10f} "
          f"dfar={g_board['far_per_frame'] - g_server['far_per_frame']:+.10f} "
          f"dtp={g_board['tp'] - g_server['tp']} dfp={g_board['fp'] - g_server['fp']} "
          f"dfn={g_board['fn'] - g_server['fn']}")

    out = {"num_frames": num_frames, "num_sequences": len(rows), "per_sequence": rows,
           "global": {"board": g_board, "server": g_server,
                      "delta": {"f1": g_board["f1"] - g_server["f1"],
                                "far_per_frame": g_board["far_per_frame"] - g_server["far_per_frame"],
                                "tp": g_board["tp"] - g_server["tp"],
                                "fp": g_board["fp"] - g_server["fp"],
                                "fn": g_board["fn"] - g_server["fn"]}}}

    if args.server_global:
        official = json.loads(args.server_global.read_text(encoding="utf-8"))
        om = official.get("overall") or official.get("metrics") or official
        out["server_official"] = om
        same = (abs(float(om["f1"]) - g_server["f1"]) < 1e-9
                and abs(float(om["far_per_frame"]) - g_server["far_per_frame"]) < 1e-12
                and int(om["tp"]) == int(g_server["tp"])
                and int(om["fp"]) == int(g_server["fp"])
                and int(om["fn"]) == int(g_server["fn"]))
        n_official = len(official.get("per_sequence") or {})
        covered = len(rows) >= n_official
        if not covered:
            out["server_aggregation_selfcheck"] = f"N/A（本次仅覆盖 {len(rows)}/{n_official} 序 ⇒ 全局聚合自检不适用）"
            print(f"[SELFCHECK] 本脚本聚合服务器逐序结果 vs 官方 {args.server_global.name}: "
                  f"N/A（仅覆盖 {len(rows)}/{n_official} 序 ⇒ 官方全局是 {n_official} 序口径，不可直接比）")
            print(f"           official f1={float(om['f1']):.10f} far={float(om['far_per_frame']):.10f} "
                  f"tp={om['tp']} fp={om['fp']} fn={om['fn']}")
        else:
            out["server_aggregation_selfcheck"] = "PASS" if same else "FAIL"
            print(f"[SELFCHECK] 本脚本聚合服务器逐序结果 vs 官方 {args.server_global.name}: "
                  f"{'PASS ✅ 逐位一致' if same else 'FAIL ❌'}")
            print(f"           official f1={float(om['f1']):.10f} far={float(om['far_per_frame']):.10f} "
                  f"tp={om['tp']} fp={om['fp']} fn={om['fn']}")
        per_seq_diff = []
        for row in rows:
            off = (official.get("per_sequence") or {}).get(row["sequence"])
            if not off:
                continue
            for key in ("tp", "fp", "fn"):
                if int(off[key]) != int(row["server"][key]):
                    per_seq_diff.append({"sequence": row["sequence"], "key": key,
                                         "official": int(off[key]), "recomputed": int(row["server"][key])})
        out["server_per_sequence_selfcheck"] = "PASS" if not per_seq_diff else "FAIL"
        out["server_per_sequence_diffs"] = per_seq_diff
        n_checked = sum(1 for row in rows if (official.get("per_sequence") or {}).get(row["sequence"]))
        print(f"[SELFCHECK] 逐序服务器计数 vs 官方 per_sequence: "
              f"{'PASS ✅ ' + str(n_checked - len(per_seq_diff)) + '/' + str(n_checked) + ' 序一致' if not per_seq_diff else 'FAIL ❌ ' + str(per_seq_diff[:5])}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OUT] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
