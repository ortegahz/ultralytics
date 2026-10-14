#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
hm_val_verify.py -- Anti-UAV val end-to-end accuracy verification for RK3588.

What this tool is responsible for
---------------------------------
The board arm and the CPU authority arm are two DIFFERENT things, and this tool
only compares the one it owns:

  CPU authority arm = FROZEN features, built once on the server
      uav_gmc_median/images/val/<SEQ>__<id>.<png|jpg>
  board arm         = features recomputed live on the board
      <board-root>/<SEQ>/feat/<SEQ>_<push:06d>.bmp   (gmc_stream_ocl, Mali fused tail)

So this tool answers exactly two questions:

  L1  does the board feature stream land on the frozen feature, frame by frame?
  S   how fast does it do that, and does it hold the 40 ms red line?

L2 (detection recall vs GT) and L3 (NPU vs CPU paired) are NOT here -- they are
owned by hm_score_board_det.py / hm_demo_report.py and must stay untouched.

Three silent-failure modes this tool is built to make impossible
-----------------------------------------------------------------
1. Hard-coded frozen-feature extension. 20 of 24 sequences are `.jpg` (27957
   frames) and only 4 are `.png` (3656) -- and the whole directory mixes both.
   `hm_demo_report.py:119` hard-codes `.png` and `continue`s when the read
   fails, which silently skips 88% of the set and still prints a pretty number.
   Here the suffix is resolved from the manifest stem and every frame in the
   manifest is a hard obligation: read it, or record why it could not be read.

2. A missing frame must never be `continue`. Every unreadable / missing /
   shape-mismatched frame is counted with a reason, printed as
   expected / compared / skipped, and makes the exit code non-zero. A run that
   compares fewer frames than the manifest promises is a FAIL, not a footnote.

3. Latency must never be pooled across resolutions. Preprocessing cost scales
   roughly linearly with pixel count (project rule section 4 / 9h), so the
   speed table is always stratified 512x640 vs 512x512 first; the combined
   number is printed only in addition to the two strata, never instead of them.

The push -> id offset is an ASSUMPTION, so it gets measured
-----------------------------------------------------------
`--push-offset` (default 1) was inferred from the single-sequence 34_1 run as
push = raw_id - 1. If that relationship is not constant across sequences, a
whole-sequence run is silently misaligned: nothing errors, every comparison is
simply against the wrong frame, and the tool reports "features do not match at
all". That is a plausible-looking false conclusion, which is exactly the failure
mode `--verify-offset` exists to kill: for every sequence we sample frames and
score each candidate k in {0,1,2,3,-1} (candidate raw_id = push + k), then keep
the k with the best mean identical-pixel fraction, and require it to beat the
runner-up by a real margin.

Usage:
    python hm_val_verify.py \\
        --frames-list  L.txt --frozen-root FR --board-root NFS/demo \\
        --verify-offset [--l1] [--speed] \\
        --out report.md --perframe-out perframe.csv \\
        [--push-offset 1] [--offset-samples 8] [--gmc-warmup 100]

Exit codes:
    0  every requested check passed
    1  a check FAILED (offset unverifiable, frame accounting short, etc.)
    2  fatal: bad invocation, unreadable manifest, missing required CSV columns
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# --- stage ledger contract -------------------------------------------------
# The header is CONFIRMED against the board's real gmc_profile.csv:
#   seq_id,frame_idx,seq_frame,io_ms,prep_ms,fit_ms,warp_ms,median_ms,
#   img_up_ms,mats_up_ms,kernel_ms,readback_ms,total_ms,resid_ms
# It is still parsed dynamically below -- these names are the EXPECTED contract
# and a missing column is a hard error, never a silently dropped column.
STAGE_COLS = [
    "io_ms",
    "prep_ms",
    "fit_ms",
    "warp_ms",
    "median_ms",
    "img_up_ms",
    "mats_up_ms",
    "kernel_ms",
    "readback_ms",
]
REQUIRED_COLS = STAGE_COLS + ["total_ms"]

# Column-name patterns that would carry a per-frame corner count. Absent from
# the confirmed header, so corner bucketing must degrade to global quantiles.
CORNER_COL_PATTERNS = ("corner", "n_corners", "ncorners", "npts", "n_pts", "pts")

# Candidate offsets tested by --verify-offset, in report order.
OFFSET_CANDIDATES = (0, 1, 2, 3, -1)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def pct(a: np.ndarray, q: float) -> float:
    """Percentile that degrades gracefully on tiny samples instead of raising."""
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))


def stats_block(a: np.ndarray) -> dict:
    """p50/p90/p99/max/std/mean for one column -- the shape every table uses."""
    if a.size == 0:
        return {k: float("nan") for k in ("p50", "p90", "p99", "max", "std", "mean")}
    return {
        "p50": pct(a, 50),
        "p90": pct(a, 90),
        "p99": pct(a, 99),
        "max": float(a.max()),
        "std": float(a.std()),
        "mean": float(a.mean()),
    }


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a GitHub-flavoured markdown table as a list of lines."""
    if not rows:
        rows = [["--"] * len(headers)]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def f3(x: float) -> str:
    return "n/a" if x != x else f"{x:.3f}"


def f4(x: float) -> str:
    return "n/a" if x != x else f"{x:.4f}"


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
class Frame:
    """One row of the manifest: a frozen feature the run is OBLIGED to compare."""

    __slots__ = ("seq", "path", "raw_id", "stem")

    def __init__(self, seq: str, path: Path, raw_id: int):
        self.seq = seq
        self.path = path
        self.raw_id = raw_id
        self.stem = path.stem


def load_manifest(frames_list: str, frozen_root: str) -> list[Frame]:
    """Parse the manifest, preserving order and resolving the TRUE suffix.

    The suffix comes from the manifest line itself, never from a constant. The
    `.png`-hardcoding in hm_demo_report.py is the bug this exists to not repeat.
    """
    lp = Path(frames_list)
    if not lp.is_file():
        raise SystemExit(f"[fatal] frames-list not found: {lp}")
    root = Path(frozen_root)
    out: list[Frame] = []
    bad = 0
    for raw in lp.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        p = Path(line)
        stem = p.stem
        if "__" not in stem:
            bad += 1
            continue
        seq, idx = stem.rsplit("__", 1)
        try:
            raw_id = int(idx)
        except ValueError:
            bad += 1
            continue
        # manifest may be relative to --frozen-root or already absolute
        resolved = p if p.is_absolute() else root / p.name
        out.append(Frame(seq, resolved, raw_id))
    if not out:
        raise SystemExit(f"[fatal] no usable frame parsed from {lp}")
    if bad:
        print(f"[manifest] 跳过 {bad} 行无法解析的条目 (缺少 '__<id>')")
    return out


def group_by_seq(frames: list[Frame]) -> "dict[str, list[Frame]]":
    g: dict[str, list[Frame]] = defaultdict(list)
    for f in frames:
        g[f.seq].append(f)
    for v in g.values():
        v.sort(key=lambda f: f.raw_id)
    return dict(g)


# ---------------------------------------------------------------------------
# board side
# ---------------------------------------------------------------------------
def board_feat(board_root: str, seq: str, push: int) -> Path:
    return Path(board_root) / seq / "feat" / f"{seq}_{push:06d}.bmp"


def board_feat_count(board_root: str, seq: str) -> int:
    d = Path(board_root) / seq / "feat"
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() == ".bmp")


def board_resolution(board_root: str, seq: str) -> str | None:
    """WxH of the board feature, read from one BMP. None if the seq has no BMP."""
    d = Path(board_root) / seq / "feat"
    if not d.is_dir():
        return None
    for p in sorted(d.iterdir()):
        if p.suffix.lower() != ".bmp":
            continue
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            return None
        return f"{im.shape[1]}x{im.shape[0]}"
    return None


# ---------------------------------------------------------------------------
# 1) offset measurement
# ---------------------------------------------------------------------------
def _compare(ba: np.ndarray, ra: np.ndarray) -> tuple[int, float, float] | None:
    """(max_abs, mean_abs, identical_frac) or None if shapes disagree."""
    if ba.shape != ra.shape:
        return None
    d = np.abs(ba.astype(np.int16) - ra.astype(np.int16))
    return int(d.max()), float(d.mean()), float((d == 0).mean())


def verify_offset(
    seq: str,
    seq_frames: list[Frame],
    board_root: str,
    n_samples: int,
    gap_min: float,
    lossless_min: float,
) -> dict:
    """Measure push -> raw_id offset for ONE sequence by brute-force correlation.

    Returns a verdict dict. The candidate scores are means over whichever
    sampled pushes had that candidate id present, so a push near the sequence
    head (where some ids simply do not exist) cannot bias one k over another.
    """
    by_id = {f.raw_id: f for f in seq_frames}
    n_board = board_feat_count(board_root, seq)
    suffix = Path(next(iter(by_id.values())).path).suffix.lower() if by_id else "?"
    container = "png" if suffix == ".png" else "jpg" if suffix in (".jpg", ".jpeg") else suffix
    res = {
        "seq": seq,
        "n_frames": len(seq_frames),
        "n_board": n_board,
        "container": container,
        "samples": [],
        "per_k": {},
        "best_k": None,
        "best_ident": float("nan"),
        "second_ident": float("nan"),
        "gap": float("nan"),
        "best_mean_abs": float("nan"),
        "second_mean_abs": float("nan"),
        "verdict": "FAIL",
        "reason": "",
    }
    if n_board == 0:
        res["reason"] = f"板端无特征目录 feat/ ({seq})"
        return res

    # evenly spaced deterministic sample over the whole push range
    pushes = sorted({int(round(v)) for v in
                     np.linspace(0, n_board - 1, max(1, min(n_samples, n_board)))})
    acc: dict[int, list[tuple[float, float]]] = defaultdict(list)  # k -> [(ident, mean_abs)]
    used: list[tuple[int, int, int]] = []  # (push, k_best_at_push, -)
    for push in pushes:
        bp = board_feat(board_root, seq, push)
        if not bp.is_file():
            continue
        bimg = cv2.imread(str(bp), cv2.IMREAD_COLOR)
        if bimg is None:
            continue
        row = {"push": push, "cands": {}}
        for k in OFFSET_CANDIDATES:
            fr = by_id.get(push + k)
            if fr is None or not fr.path.is_file():
                row["cands"][k] = None
                continue
            rimg = cv2.imread(str(fr.path), cv2.IMREAD_COLOR)
            if rimg is None:
                row["cands"][k] = None
                continue
            r = _compare(bimg, rimg)
            row["cands"][k] = r
            if r is not None:
                acc[k].append((r[2], r[1]))
        res["samples"].append(row)
        # per-push argmax, for the report's readability
        valid = {k: v for k, v in row["cands"].items() if v is not None}
        used.append((push, max(valid, key=lambda k: valid[k][2]) if valid else -999))

    if not acc:
        res["reason"] = "所有抽样帧都没有可用的候选 id"
        return res

    for k, v in acc.items():
        res["per_k"][k] = {
            "n": len(v),
            "ident": float(np.mean([x[0] for x in v])),
            "mean_abs": float(np.mean([x[1] for x in v])),
        }

    ranked = sorted(res["per_k"].items(), key=lambda kv: (-kv[1]["ident"], kv[1]["mean_abs"]))
    best_k, best = ranked[0]
    res["best_k"] = best_k
    res["best_ident"] = best["ident"]
    res["best_mean_abs"] = best["mean_abs"]
    if len(ranked) > 1:
        _, second = ranked[1]
        res["second_ident"] = second["ident"]
        res["second_mean_abs"] = second["mean_abs"]
        res["gap"] = best["ident"] - second["ident"]
    res["runner_up_k"] = ranked[1][0] if len(ranked) > 1 else None

    # --- binding rules ----------------------------------------------------
    # (a) contrast: the winner must actually beat the runner-up, otherwise the
    #     offset is simply not determined by this data.
    if not (res["gap"] >= gap_min):
        res["reason"] = f"无法判定: 最优与次优 k 的逐位相同率差 {res['gap']:.4f} < {gap_min}"
        return res
    # (b) absolute floor. This is only a valid gate for a LOSSLESS container:
    #     a jpg frozen feature already carries compression error against the
    #     board's lossless BMP, so its ceiling is below 1.0 by construction and
    #     gating it at 0.9 would fail all 20 jpg sequences for a reason that has
    #     nothing to do with alignment. jpg is therefore judged by (a) alone,
    #     and its absolute number is reported as advisory.
    if container == "png" and best["ident"] < lossless_min:
        res["reason"] = (f"无损容器逐位相同率 {best['ident']:.4f} < {lossless_min}; "
                         "特征未对齐")
        return res
    res["verdict"] = "PASS"
    res["reason"] = "无损容器绝对门槛" if container == "png" else "有损容器按对比度判定"
    return res


# ---------------------------------------------------------------------------
# 2) L1 alignment
# ---------------------------------------------------------------------------
def run_l1(
    seq_frames_by_seq: dict[str, list[Frame]],
    board_root: str,
    offsets: dict[str, int],
    perframe_path: Path | None,
) -> dict:
    recs: list[dict] = []
    skip_reasons: dict[str, int] = defaultdict(int)
    expected = 0
    board_total = 0
    board_unpaired = 0

    for seq in sorted(seq_frames_by_seq):
        k = offsets.get(seq)
        if k is None:
            for f in seq_frames_by_seq[seq]:
                expected += 1
                skip_reasons["无实测偏移"] += 1
            continue
        n_board = board_feat_count(board_root, seq)
        board_total += n_board
        matched_pushes: set[int] = set()
        for f in seq_frames_by_seq[seq]:
            expected += 1
            push = f.raw_id - k
            bp = board_feat(board_root, seq, push)
            if push < 0:
                skip_reasons["push<0"] += 1
                continue
            if not bp.is_file():
                skip_reasons["板端特征缺失"] += 1
                continue
            if not f.path.is_file():
                skip_reasons["冻结特征缺失"] += 1
                continue
            B = cv2.imread(str(bp), cv2.IMREAD_COLOR)
            if B is None:
                skip_reasons["板端特征不可读"] += 1
                continue
            R = cv2.imread(str(f.path), cv2.IMREAD_COLOR)
            if R is None:
                skip_reasons["冻结特征不可读"] += 1
                continue
            if B.shape != R.shape:
                skip_reasons[f"形状不符({B.shape[1]}x{B.shape[0]} vs {R.shape[1]}x{R.shape[0]})"] += 1
                continue
            matched_pushes.add(push)
            d = np.abs(B.astype(np.int16) - R.astype(np.int16))
            recs.append({
                "seq": seq,
                "frozen_name": f.path.name,
                "push": push,
                "id": f.raw_id,
                "max_abs": int(d.max()),
                "mean_abs": float(d.mean()),
                "identical_frac": float((d == 0).mean()),
                "res": f"{B.shape[1]}x{B.shape[0]}",
                "container": "png" if f.path.suffix.lower() == ".png" else "jpg",
            })
        board_unpaired += max(0, n_board - len(matched_pushes))

    if perframe_path is not None and recs:
        perframe_path.parent.mkdir(parents=True, exist_ok=True)
        with perframe_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seq", "frozen_name", "push", "id",
                        "max_abs", "mean_abs", "identical_frac"])
            for r in recs:
                w.writerow([r["seq"], r["frozen_name"], r["push"], r["id"],
                            r["max_abs"], f"{r['mean_abs']:.6f}", f"{r['identical_frac']:.6f}"])
        print(f"[L1] 逐帧明细已写: {perframe_path}  ({len(recs)} 行)")

    n_cmp = len(recs)
    return {
        "recs": recs,
        "expected": expected,
        "n_cmp": n_cmp,
        "skipped": expected - n_cmp,
        "skip_reasons": dict(skip_reasons),
        "board_total": board_total,
        "board_unpaired": board_unpaired,
    }


def l1_summary(recs: list[dict], label: str) -> dict:
    if not recs:
        return {"label": label, "n": 0}
    mx = np.array([r["max_abs"] for r in recs], float)
    mn = np.array([r["mean_abs"] for r in recs], float)
    idf = np.array([r["identical_frac"] for r in recs], float)
    return {
        "label": label,
        "n": len(recs),
        "global_max_abs": int(mx.max()),
        "frame_max_p50": pct(mx, 50),
        "frame_max_p95": pct(mx, 95),
        "frame_max_p99": pct(mx, 99),
        "mean_abs_mean": float(mn.mean()),
        "mean_abs_max": float(mn.max()),
        "ident_min": float(idf.min()),
        "ident_mean": float(idf.mean()),
        "ident_max": float(idf.max()),
        "exact_frames": int((mx == 0).sum()),
    }


def fmt_l1_row(s: dict) -> list[str]:
    if not s.get("n"):
        return [s["label"], "0", "--", "--", "--", "--", "--", "--", "--", "--"]
    return [
        s["label"], str(s["n"]), str(s["global_max_abs"]),
        f3(s["frame_max_p50"]), f3(s["frame_max_p95"]), f3(s["frame_max_p99"]),
        f4(s["ident_min"]), f4(s["ident_mean"]), f4(s["ident_max"]),
        str(s["exact_frames"]),
    ]


L1_HDR = ["分层", "n_cmp", "global_max", "帧max p50", "帧max p95", "帧max p99",
          "identical min", "identical mean", "identical max", "全帧逐位相同帧数"]


# ---------------------------------------------------------------------------
# 3) speed
# ---------------------------------------------------------------------------
def read_profile_csv(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read the stage ledger. Header is parsed, never assumed.

    Returns (column -> float array, header order). Raises SystemExit when a
    required column is absent -- a missing stage must never degrade into a
    quietly shorter table.
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open(newline="") as fh:
        rdr = csv.DictReader(fh)
        header = rdr.fieldnames or []
        missing = [c for c in REQUIRED_COLS if c not in header]
        if missing:
            raise SystemExit(
                f"[fatal] {path} 缺少必需列 {missing}; 实际表头 = {header}")
        rows = list(rdr)
    cols: dict[str, np.ndarray] = {}
    for c in header:
        try:
            cols[c] = np.array([float(r[c]) for r in rows], dtype=float)
        except (TypeError, ValueError):
            cols[c] = np.array([], dtype=float)
    return cols, header


def find_corner_col(header: list[str]) -> str | None:
    for c in header:
        low = c.lower()
        if any(p in low for p in CORNER_COL_PATTERNS):
            return c
    return None


BUDGET_RE = re.compile(
    r"\[budget\]\s*(?:(\d+(?:\.\d+)?)\s*ms\s*red line)?.*?p99\s+(PASS|FAIL)\s*\(([\d.]+)\s*ms\)"
    r".*?max\s+(PASS|FAIL)\s*\(([\d.]+)\s*ms\)", re.I)
END2END_RE = re.compile(r"==\s*END-TO-END TOTAL\s*==\s*(.+)$", re.I)
FPS_RE = re.compile(r"steady-state fps\s*@p50\s*=\s*([\d.]+)\s*\|\s*@p99\s*=\s*([\d.]+)"
                    r"\s*\|\s*@max\s*=\s*([\d.]+)", re.I)


def parse_logs(log_path: Path) -> dict:
    out = {"budget": [], "end_to_end": [], "fps": [], "warmup_declared": None}
    if not log_path.is_file():
        return out
    for line in log_path.read_text(errors="replace").splitlines():
        m = BUDGET_RE.search(line)
        if m:
            out["budget"].append({
                "red_line_ms": float(m.group(1)) if m.group(1) else 40.0,
                "p99_verdict": m.group(2).upper(),
                "p99_ms": float(m.group(3)),
                "max_verdict": m.group(4).upper(),
                "max_ms": float(m.group(5)),
            })
        m = END2END_RE.search(line)
        if m:
            nums = [float(x) for x in re.findall(r"[\d.]+", m.group(1))]
            # board order is: mean p50 p95 p99 max std
            if len(nums) >= 6:
                out["end_to_end"].append({
                    "mean": nums[0], "p50": nums[1], "p95": nums[2],
                    "p99": nums[3], "max": nums[4], "std": nums[5],
                })
        m = FPS_RE.search(line)
        if m:
            out["fps"].append({"p50": float(m.group(1)), "p99": float(m.group(2)),
                               "max": float(m.group(3))})
        m = re.search(r"warm-?up discarded=(\d+)", line, re.I)
        if m:
            out["warmup_declared"] = int(m.group(1))
    return out


def run_speed(board_root: str, seqs: list[str], warmup_req: int, red_line: float) -> dict:
    per_seq: dict[str, dict] = {}
    notes: list[str] = []
    corner_col_global: str | None = None
    header_global: list[str] | None = None

    for seq in seqs:
        sd = Path(board_root) / seq
        csvp = sd / "gmc_profile.csv"
        if not csvp.is_file():
            notes.append(f"{seq}: 无 gmc_profile.csv, 未纳入速度统计")
            continue
        try:
            cols, header = read_profile_csv(csvp)
        except FileNotFoundError:
            notes.append(f"{seq}: gmc_profile.csv 不可读")
            continue
        header_global = header
        ccol = find_corner_col(header)
        if ccol:
            corner_col_global = ccol
        n = len(cols["total_ms"])
        if n == 0:
            notes.append(f"{seq}: gmc_profile.csv 无数据行")
            continue
        w = min(warmup_req, n // 5)
        sl = slice(w, n) if w else slice(0, n)
        keep = [c for c in REQUIRED_COLS]
        per_seq[seq] = {
            "res": board_resolution(board_root, seq),
            "n_total": n,
            "warmup_req": warmup_req,
            "warmup_used": w,
            "warmup_degraded": w < warmup_req,
            "cols": {c: cols[c][sl] for c in keep},
            "corner_col": ccol,
            "corner_vals": cols[ccol][sl] if ccol else None,
            "logs": parse_logs(sd / "gmc_run.log"),
        }
        tag = " (降级: n//5)" if w < warmup_req else ""
        print(f"[speed] {seq}: {n} 行, 剔除冷启动 {w} 帧 (要求 {warmup_req}){tag}, "
              f"计入 {n - w} 帧, 分辨率 {per_seq[seq]['res']}")

    # ---- per-stage, stratified by resolution (never pooled alone) ----------
    stage_tab: dict[str, dict] = {}
    resolutions = sorted({v["res"] for v in per_seq.values() if v["res"]})
    for res in resolutions:
        rows = []
        for c in STAGE_COLS + ["total_ms"]:
            pool = np.concatenate([v["cols"][c] for v in per_seq.values()
                                   if v["res"] == res and v["cols"][c].size]) \
                if any(v["res"] == res and v["cols"][c].size for v in per_seq.values()) \
                else np.array([])
            s = stats_block(pool)
            rows.append([c, str(pool.size), f3(s["p50"]), f3(s["p90"]), f3(s["p99"]),
                         f3(s["max"]), f3(s["std"]), f3(s["mean"])])
        stage_tab[res] = rows
    combined = []
    all_cols = [c for c in STAGE_COLS + ["total_ms"]
                if any(v["cols"][c].size for v in per_seq.values())]
    for c in all_cols:
        pool = np.concatenate([v["cols"][c] for v in per_seq.values() if v["cols"][c].size])
        s = stats_block(pool)
        combined.append([c, str(pool.size), f3(s["p50"]), f3(s["p90"]), f3(s["p99"]),
                         f3(s["max"]), f3(s["std"]), f3(s["mean"])])

    # ---- 40 ms red line, judged on p50 AND p99, per resolution -------------
    redline = []
    for seq, v in per_seq.items():
        lg = v["logs"]
        if not lg["budget"]:
            redline.append([seq, v["res"] or "?", "--", "--", "--", "日志无 [budget] 行"])
            continue
        b = lg["budget"][-1]
        e2e = lg["end_to_end"][-1] if lg["end_to_end"] else None
        p50 = e2e["p50"] if e2e else float("nan")
        p99 = e2e["p99"] if e2e else b["p99_ms"]
        rl = b["red_line_ms"] or red_line
        v50 = "PASS" if p50 == p50 and p50 <= rl else "FAIL"
        v99 = "PASS" if p99 <= rl else "FAIL"
        redline.append([seq, v["res"] or "?", f3(p50), v50, f3(p99), v99])

    # ---- fit_ms bucketed by corner count, ONLY if such a column exists ----
    corner_buckets = None
    if corner_col_global:
        edges = [0, 100, 200, 300, 400, 500, 10 ** 9]
        labels = ["<100", "100-199", "200-299", "300-399", "400-499", ">=500"]
        corner_buckets = []
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            sel = []
            for v in per_seq.values():
                if v["corner_vals"] is None:
                    continue
                m = (v["corner_vals"] >= lo) & (v["corner_vals"] < hi)
                if m.any():
                    sel.append(v["cols"]["fit_ms"][m])
            if sel:
                pool = np.concatenate(sel)
                s = stats_block(pool)
                corner_buckets.append([lab, str(pool.size), f3(s["p50"]), f3(s["p90"]),
                                       f3(s["p99"]), f3(s["max"]), f3(s["mean"])])

    return {
        "per_seq": per_seq, "resolutions": resolutions, "stage_tab": stage_tab,
        "combined": combined, "redline": redline, "corner_col": corner_col_global,
        "corner_buckets": corner_buckets, "notes": notes, "header": header_global,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def build_report(args, frames, seqs, off, l1, sp, fails, expected_total) -> list[str]:
    L: list[str] = []
    L.append("# Anti-UAV val 全量精度验证报告 (板端特征 vs 冻结特征)")
    L.append("")
    L.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 帧清单: `{args.frames_list}`")
    L.append(f"- 冻结特征根 (CPU 权威臂): `{args.frozen_root}`")
    L.append(f"- 板端产物根: `{args.board_root}`")
    L.append(f"- 序列数: {len(seqs)}   清单帧数 (期望): {expected_total}")
    L.append(f"- 参数: push_offset={args.push_offset} offset_samples={args.offset_samples} "
             f"gmc_warmup={args.gmc_warmup}")
    L.append("")
    L.append("> 本报告只覆盖 L1 (特征对齐) 与 S (速度)。L2/L3 精度由 "
             "`hm_score_board_det.py` + `hm_demo_report.py` 负责, 此处不做任何改动。")
    L.append("")

    L.append("## 0. 判定摘要")
    L.append("")
    L.append(f"**总判定: {'FAIL' if fails else 'PASS'}**  "
             f"失败项 {len(fails)} 个")
    L.append("")
    if fails:
        for f in fails:
            L.append(f"- [FAIL] {f}")
    else:
        L.append("- 所有请求的校验项均通过")
    L.append("")

    # ---- 1) offset -------------------------------------------------------
    if off is not None:
        L.append("## 1. push -> id 偏移实测校验")
        L.append("")
        L.append(f"`--push-offset {args.push_offset}` 只是从单序列 34_1 推来的假设。"
                 f"逐序列抽样 {args.offset_samples} 帧, 对候选 k "
                 f"{list(OFFSET_CANDIDATES)} (候选 raw_id = push + k) 做逐像素比对, "
                 f"取逐位相同率最高者。")
        L.append("")
        hdr = ["序列", "容器", "清单帧", "板端帧", "best k", "best 相同率",
               "次优 k", "次优相同率", "差距", "判定"]
        rows = []
        for seq in seqs:
            r = off.get(seq)
            if r is None:
                rows.append([seq, "--", "--", "--", "--", "--", "--", "--", "--", "FAIL"])
                continue
            rows.append([
                seq, r["container"], str(r["n_frames"]), str(r["n_board"]),
                str(r["best_k"]), f4(r["best_ident"]),
                str(r.get("runner_up_k")), f4(r["second_ident"]),
                f4(r["gap"]), r["verdict"],
            ])
        L += md_table(hdr, rows)
        L.append("")
        L.append("### 判定规则")
        L.append("")
        L.append(f"- 差距 (最优 - 次优 逐位相同率) 必须 >= `{args.offset_gap}`, "
                 f"否则判为「无法判定」并使退出码非零。")
        L.append(f"- **无损容器 (png)**: 最优逐位相同率必须 >= `{args.lossless_min}`。")
        L.append("- **有损容器 (jpg)**: **不施加该绝对门槛**。冻结特征本身是 jpg, "
                 "它与板端无损 BMP 之间本就存在压缩误差, 逐位相同率的上限由容器决定而非由对齐决定; "
                 "对 jpg 用 0.9 判决会把全部 20 个 jpg 序列无一例外地判 FAIL, 而失败原因与对齐无关。"
                 "jpg 序列改由上面的「差距」门槛判定, 其绝对数值作为参考量报告。")
        L.append("")
        # per-k detail
        L.append("### 各候选 k 的抽样得分 (仅列出有可用候选的 k)")
        L.append("")
        for seq in seqs:
            r = off.get(seq)
            if r is None or not r["per_k"]:
                continue
            parts = ", ".join(
                f"k={k}: n={v['n']} 相同率={v['ident']:.4f} mean|d|={v['mean_abs']:.4f}"
                for k, v in sorted(r["per_k"].items()))
            L.append(f"- `{seq}`: {parts}")
        L.append("")
        ks = sorted({r["best_k"] for r in off.values() if r and r["best_k"] is not None})
        if ks:
            if len(ks) == 1:
                k = ks[0]
                L.append(f"**全序列实测偏移一致: k = {k}** (id = push {k:+d})  "
                         + (f"与 `--push-offset {args.push_offset}` 相符。\n"
                            if k == args.push_offset else
                            f"与 `--push-offset {args.push_offset}` **不相符**。\n"))
            else:
                L.append(f"**实测偏移跨序列不一致: {ks}** —— 全量比对不能用一个固定偏移, "
                         "已按序列分别使用其实测值。\n")
        L.append("")

    # ---- 2) L1 -----------------------------------------------------------
    if l1 is not None:
        L.append("## 2. L1 特征对齐 (板端特征 vs 冻结特征)")
        L.append("")
        L.append("逐元素绝对差 `np.abs(B.astype(int16) - R.astype(int16))`, HxWx3 全通道; "
                 "`identical_frac` = 该帧 3 通道全部逐位相同的元素占比。")
        L.append("")
        L.append("### 帧数账目 (三者必须相等, 否则 FAIL)")
        L.append("")
        L += md_table(["项", "值"], [
            ["期望帧数 (清单)", str(l1["expected"])],
            ["实际比对帧数 n_cmp", str(l1["n_cmp"])],
            ["跳过帧数", str(l1["skipped"])],
            ["板端特征总帧数", str(l1["board_total"])],
            ["板端有特征但清单无对应帧", str(l1["board_unpaired"])],
        ])
        L.append("")
        if l1["skip_reasons"]:
            L.append("跳过原因明细:")
            for k, v in sorted(l1["skip_reasons"].items(), key=lambda kv: -kv[1]):
                L.append(f"- `{k}`: {v} 帧")
            L.append("")
        else:
            L.append("跳过原因明细: 无。")
            L.append("")

        recs = l1["recs"]
        if recs:
            L.append("### 2.1 全量汇总")
            L.append("")
            L += md_table(L1_HDR, [fmt_l1_row(l1_summary(recs, "全部"))])
            L.append("")
            L.append("### 2.2 按分辨率分层 (预处理时延与精度都随分辨率变化, 禁止合并后再看)")
            L.append("")
            res_rows = []
            for res in sorted({r["res"] for r in recs}):
                res_rows.append(fmt_l1_row(
                    l1_summary([r for r in recs if r["res"] == res], res)))
            L += md_table(L1_HDR, res_rows)
            L.append("")
            L.append("### 2.3 按冻结容器分层 (png 无损 vs jpg 有损)")
            L.append("")
            L.append("> jpg 帧的 `identical_frac` 天然低于 png: 冻结特征容器本身有损, "
                     "它与板端无损 BMP 之间存在压缩误差。这是**测量口径差异**, 不是对齐 bug, "
                     "所以两层必须分开读, 不能把 jpg 的低相同率当成板端错帧。")
            L.append("")
            L += md_table(L1_HDR, [fmt_l1_row(
                l1_summary([r for r in recs if r["container"] == c], c))
                for c in ("png", "jpg") if any(r["container"] == c for r in recs)])
            L.append("")
            L.append("### 2.4 逐序列")
            L.append("")
            seq_rows = []
            for seq in sorted({r["seq"] for r in recs}):
                sub = [r for r in recs if r["seq"] == seq]
                mx = np.array([r["max_abs"] for r in sub], float)
                idf = np.array([r["identical_frac"] for r in sub], float)
                conts = sorted({r["container"] for r in sub})
                res = sub[0]["res"]
                seq_rows.append([
                    seq, "/".join(conts), res, str(len(sub)),
                    f3(float(mx.max())), f3(pct(mx, 50)), f3(pct(mx, 99)),
                    f4(float(idf.mean())), f4(float(idf.min())),
                    str(int((mx == 0).sum())),
                ])
            L += md_table(["序列", "容器", "分辨率", "n", "max_abs", "帧max p50",
                           "帧max p99", "identical mean", "identical min", "全帧逐位相同"],
                          seq_rows)
            L.append("")

    # ---- 3) speed --------------------------------------------------------
    if sp is not None:
        L.append("## 3. 速度汇总 (gmc_profile.csv)")
        L.append("")
        if sp["header"]:
            L.append(f"实际 CSV 表头: `{' , '.join(sp['header'])}`")
            L.append("")
        L.append("> 每阶段 p50/p90/p99/max/std/mean 均由 CSV 原始行现算, "
                 "冷启动按 `--gmc-warmup` 逐序列剔除, 不引用日志里的预聚合表。")
        L.append("")
        for res, rows in sp["stage_tab"].items():
            L.append(f"### 3.1 分辨率 {res}")
            L.append("")
            L += md_table(["stage", "n", "p50", "p90", "p99", "max", "std", "mean"], rows)
            L.append("")
        L.append("### 3.2 合并 (仅在分层之后参考, 不可单独引用)")
        L.append("")
        L += md_table(["stage", "n", "p50", "p90", "p99", "max", "std", "mean"],
                      sp["combined"])
        L.append("")
        L.append("`fit_ms` = GMC 角点+特征拟合耗时, 即「角点拟合耗时」。")
        L.append("")
        L.append("### 3.3 40 ms 红线 (p50 与 p99 同时判定)")
        L.append("")
        L += md_table(["序列", "分辨率", "p50 (ms)", "p50 判定",
                       "p99 (ms)", "p99 判定"], sp["redline"])
        L.append("")
        L.append("### 3.4 fit_ms 按角点数分桶")
        L.append("")
        if sp["corner_col"]:
            L.append(f"本表存在角点数列 `{sp['corner_col']}`, 分桶如下:")
            L.append("")
            L += md_table(["角点数区间", "n", "p50", "p90", "p99", "max", "mean"],
                          sp["corner_buckets"] or [["--"] * 7])
        else:
            L.append("**本表无角点数列。** `gmc_profile.csv` 的实际表头里不含任何角点/特征点"
                     "计数列, 因此角点耗时分布**只能给全局分位数** (见 3.1 / 3.2 的 `fit_ms` 行), "
                     "无法按角点数分桶。板端日志里只有序列级汇总 "
                     "`corners mean=... over ... dets`, 粒度是整序列而非逐帧, "
                     "不足以做分桶, 故此处不做任何角点数区间的推断。")
        L.append("")
        if sp["notes"]:
            L.append("### 3.5 备注")
            L.append("")
            for n in sp["notes"]:
                L.append(f"- {n}")
            L.append("")

    L.append("## 4. 数据异常与口径声明")
    L.append("")
    L.append("- 本报告不生成任何图片/图表, 因此不存在图内 CJK 文本。")
    L.append("- 冻结容器混用: 20 个序列为 jpg (27957 帧), 4 个序列为 png (3656 帧), "
             "全目录混用。扩展名一律由清单 stem 解析, 不硬编码。")
    L.append("- 速度统计一律先按分辨率分层再给合并值 (项目铁律 4/9h)。")
    L.append("")
    return L


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Anti-UAV val 全量精度验证: 偏移实测 + L1 特征对齐 + 速度汇总")
    ap.add_argument("--frames-list", required=True)
    ap.add_argument("--frozen-root", required=True)
    ap.add_argument("--board-root", required=True)
    ap.add_argument("--verify-offset", action="store_true")
    ap.add_argument("--l1", action="store_true")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--perframe-out", default=None)
    ap.add_argument("--push-offset", type=int, default=1)
    ap.add_argument("--offset-samples", type=int, default=8)
    ap.add_argument("--offset-gap", type=float, default=0.05)
    ap.add_argument("--lossless-min", type=float, default=0.9)
    ap.add_argument("--gmc-warmup", type=int, default=100)
    ap.add_argument("--red-line-ms", type=float, default=40.0)
    args = ap.parse_args()

    if not (args.verify_offset or args.l1 or args.speed):
        raise SystemExit("[fatal] 至少指定 --verify-offset / --l1 / --speed 之一")

    frames = load_manifest(args.frames_list, args.frozen_root)
    seqs_map = group_by_seq(frames)
    seqs = sorted(seqs_map)
    expected_total = len(frames)
    print("=" * 78)
    print(f"[manifest] {expected_total} 帧 / {len(seqs)} 序列")
    conts: dict[str, int] = defaultdict(int)
    for f in frames:
        conts[f.path.suffix.lower()] += 1
    print(f"[manifest] 冻结容器分布: {dict(conts)}")
    print("=" * 78)

    fails: list[str] = []
    off: dict[str, dict] | None = None
    l1: dict | None = None
    sp: dict | None = None

    # ---- 1) offset -------------------------------------------------------
    offsets: dict[str, int] = {s: args.push_offset for s in seqs}
    if args.verify_offset:
        print("\n" + "=" * 78)
        print("偏移实测校验 (--verify-offset)")
        print("=" * 78)
        off = {}
        hdr = (f"| {'序列':26} | {'容器':4} | {'best k':>6} | {'best 相同率':>10} | "
               f"{'次优 k':>6} | {'次优相同率':>10} | {'差距':>7} | {'判定':>4} |")
        print(hdr)
        print("|" + "-" * (len(hdr) - 2) + "|")
        for seq in seqs:
            r = verify_offset(seq, seqs_map[seq], args.board_root,
                              args.offset_samples, args.offset_gap, args.lossless_min)
            off[seq] = r
            print(f"| {seq:26} | {r['container']:4} | {str(r['best_k']):>6} | "
                  f"{r['best_ident']:10.4f} | {str(r.get('runner_up_k')):>6} | "
                  f"{r['second_ident']:10.4f} | {r['gap']:7.4f} | {r['verdict']:>4} |")
            if r["verdict"] != "PASS":
                fails.append(f"偏移校验 {seq}: {r['reason']}")
            else:
                offsets[seq] = r["best_k"]
        ks = sorted({r["best_k"] for r in off.values() if r["best_k"] is not None})
        print(f"\n[offset] 实测 k 集合 = {ks}")
        if len(ks) == 1:
            k = ks[0]
            same = k == args.push_offset
            print(f"[offset] 全序列一致: k={k}; 与 --push-offset {args.push_offset} "
                  + ("相符" if same else "不相符"))
            if not same:
                fails.append(f"实测偏移 k={k} 与 --push-offset {args.push_offset} 不符")
        else:
            fails.append(f"实测偏移跨序列不一致: {ks}")

    # ---- 2) L1 -----------------------------------------------------------
    if args.l1:
        print("\n" + "=" * 78)
        print("L1 特征对齐 (--l1)")
        print("=" * 78)
        t0 = datetime.now()
        l1 = run_l1(seqs_map, args.board_root, offsets,
                    Path(args.perframe_out) if args.perframe_out else None)
        dt = (datetime.now() - t0).total_seconds()
        print(f"\n[L1] 期望帧数   : {l1['expected']}")
        print(f"[L1] 实际比对帧数: {l1['n_cmp']}")
        print(f"[L1] 跳过帧数    : {l1['skipped']}")
        print(f"[L1] 板端特征总帧: {l1['board_total']}  "
              f"(其中 {l1['board_unpaired']} 帧在清单中无对应冻结帧)")
        for k, v in sorted(l1["skip_reasons"].items(), key=lambda kv: -kv[1]):
            print(f"[L1]   跳过原因 {k}: {v}")
        if l1["n_cmp"] != l1["expected"]:
            fails.append(f"L1 帧数不平: 期望 {l1['expected']} / 实际比对 {l1['n_cmp']} / "
                         f"跳过 {l1['skipped']}")
        if l1["recs"]:
            s = l1_summary(l1["recs"], "全部")
            print(f"[L1] global max_abs = {s['global_max_abs']}   "
                  f"帧max p50/p95/p99 = {s['frame_max_p50']:.1f}/{s['frame_max_p95']:.1f}/"
                  f"{s['frame_max_p99']:.1f}")
            print(f"[L1] identical_frac min/mean/max = {s['ident_min']:.4f}/"
                  f"{s['ident_mean']:.4f}/{s['ident_max']:.4f}   "
                  f"全帧逐位相同 {s['exact_frames']}/{l1['n_cmp']}")
            print("[L1] 按分辨率分层:")
            for res in sorted({r['res'] for r in l1['recs']}):
                sr = l1_summary([r for r in l1['recs'] if r['res'] == res], res)
                print(f"[L1]   {res:10} n={sr['n']:6d}  max_abs={sr['global_max_abs']:3d}  "
                      f"identical mean={sr['ident_mean']:.4f} min={sr['ident_min']:.4f}")
            print("[L1] 按冻结容器分层:")
            for c in ("png", "jpg"):
                if any(r['container'] == c for r in l1['recs']):
                    sr = l1_summary([r for r in l1['recs'] if r['container'] == c], c)
                    print(f"[L1]   {c:4} n={sr['n']:6d}  max_abs={sr['global_max_abs']:3d}  "
                          f"identical mean={sr['ident_mean']:.4f} min={sr['ident_min']:.4f}")
        else:
            fails.append("L1 无可比帧")
        print(f"[L1] 耗时 {dt:.1f}s")

    # ---- 3) speed --------------------------------------------------------
    if args.speed:
        print("\n" + "=" * 78)
        print("速度汇总 (--speed)")
        print("=" * 78)
        try:
            sp = run_speed(args.board_root, seqs, args.gmc_warmup, args.red_line_ms)
        except SystemExit:
            raise
        print(f"\n[speed] 纳入 {len(sp['per_seq'])} 个序列; 分辨率分层 = {sp['resolutions']}")
        print(f"[speed] CSV 实际表头 = {sp['header']}")
        if sp["corner_col"]:
            print(f"[speed] 角点数列 = {sp['corner_col']}")
        else:
            print("[speed] 本表无角点数列, 角点耗时分布只能给全局分位数")
        for res, rows in sp["stage_tab"].items():
            print(f"\n[speed] 分辨率 {res}:")
            print(f"  {'stage':12} {'n':>7} {'p50':>8} {'p90':>8} {'p99':>8} "
                  f"{'max':>8} {'std':>7} {'mean':>8}")
            for r in rows:
                print(f"  {r[0]:12} {r[1]:>7} {r[2]:>8} {r[3]:>8} {r[4]:>8} "
                      f"{r[5]:>8} {r[6]:>7} {r[7]:>8}")
        print("\n[speed] 合并 (分层之后参考):")
        print(f"  {'stage':12} {'n':>7} {'p50':>8} {'p90':>8} {'p99':>8} "
              f"{'max':>8} {'std':>7} {'mean':>8}")
        for r in sp["combined"]:
            print(f"  {r[0]:12} {r[1]:>7} {r[2]:>8} {r[3]:>8} {r[4]:>8} "
                  f"{r[5]:>8} {r[6]:>7} {r[7]:>8}")
        print("\n[speed] 40ms 红线 (p50 与 p99 同时判定):")
        print(f"  {'序列':26} {'分辨率':10} {'p50':>8} {'p50':>5} {'p99':>8} {'p99':>5}")
        for r in sp["redline"]:
            print(f"  {r[0]:26} {r[1]:10} {r[2]:>8} {r[3]:>5} {r[4]:>8} {r[5]:>5}")
            if "FAIL" in (r[3], r[5]):
                fails.append(f"40ms 红线 {r[0]} ({r[1]}): p50={r[3]} p99={r[5]}")
        for n in sp["notes"]:
            print(f"[speed] note: {n}")

    # ---- outputs ---------------------------------------------------------
    if args.out:
        rep = build_report(args, frames, seqs, off, l1, sp, fails, expected_total)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(rep) + "\n")
        print(f"\n[out] 报告已写: {args.out}")

    print("\n" + "=" * 78)
    if fails:
        print(f"[判定] FAIL  失败项 {len(fails)} 个:")
        for f in fails:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print("[判定] PASS")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())