#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
35_mosaic_board_eval.py -- RK3588 board-side f16 mosaic inference vs the PC f32 multi-scale baseline.

What this script exists for
---------------------------
    The PC reference (ms_cache) is a 3-scale, 3-forward-pass product.
    The board candidate is a 1-forward-pass product on a mosaic canvas.
    Those two are NOT the same computation graph, so "board F1 < PC F1" is
    NOT a quantisation number until the multi-scale contribution has been
    subtracted out.  This script produces the arms that make the subtraction
    possible, and it scores EVERY arm with ONE scorer (the frozen
    ``manu/evaluation/system_sota_longquan.py --no-pipeline`` path), so the
    table that comes out is auditable line by line.

The arms
--------
    pc-all               PC f32, all 3 scales (native / 160 / 80)      -> the frozen baseline
    pc-native            PC f32, tag 0 only (same pkl, subsetted)      -> multi-scale contribution
    board-single         board f16, single-scale canvas                 -> f16 + board quantisation
    board-mosaic-native  board f16, mosaic canvas, native tile only     -> canvas perturbation
    board-mosaic         board f16, mosaic canvas, 3 tiles              -> the product

THE EVALUATION CONTRACT (do not change it silently)
---------------------------------------------------
    scorer  = manu/evaluation/system_sota_longquan.py --no-pipeline
    params  = taken VERBATIM from the gate product's own ``config.params``
              (runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/track3_blind.json)
              -- NOT hand-copied, because the script's own DEFAULT_PARAMS carry
              th_base=0.22 while the gate ran with th_base=0.30, and the two
              do NOT produce the same numbers (0.84833 vs 0.89769 on T3).
    GT      = /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010
              (11,810 label files for 001324, 10,535 valid boxes)

    Frozen T3 baseline that MUST be reproduced by --self-check:
        tp=8972  fp=482  fn=1563  gt=10535  det=9454  f1=0.8976937315523539

Usage
-----
    # 0) the guard -- if this does not reproduce the frozen baseline, NOTHING
    #    downstream is trustworthy, so it exits non-zero and stops the run.
    python manu/pipeline/npu/ptq/35_mosaic_board_eval.py --self-check

    # 1) the PC arms (no board, no GPU)
    python manu/pipeline/npu/ptq/35_mosaic_board_eval.py --arm pc-all    --out <dir>
    python manu/pipeline/npu/ptq/35_mosaic_board_eval.py --arm pc-native --out <dir>

    # 2) the board arms (drives the board over ssh)
    python manu/pipeline/npu/ptq/35_mosaic_board_eval.py --arm board-mosaic --out <dir>

    # 3) the report
    python manu/pipeline/npu/ptq/36_mosaic_vs_f32_report.py --out <dir>

Exit codes: 0 = ok, 1 = a check/assertion failed, 2 = a setup error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# The repo root is parents[4] (ptq -> npu -> pipeline -> manu -> <root>).
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCORER = REPO_ROOT / "manu" / "evaluation" / "system_sota_longquan.py"

# --------------------------------------------------------------------------------------
# The frozen contract.  Every number here is reproduced by --self-check; nothing in this
# file is allowed to drift away from it silently.
# --------------------------------------------------------------------------------------
SEQUENCE = "VIDEO00007_19700101_001324"

FROZEN_T3 = {
    "tp": 8972,
    "fp": 482,
    "fn": 1563,
    "gt": 10535,
    "det": 9454,
    "f1": 0.8976937315523539,
}
FROZEN_T3_F1_TOL = 1e-12

NATIVE_SIZE = (640, 512)          # (W, H) of the native frame, in pixels
SCALES = [0, 160, 80]             # native / thumb160 / thumb80
FUSED_TAG = 255                   # the baked fused row -- deliberately NOT used here
SCALE_TAGS = (0, 1, 2)           # the three rows the scorer is allowed to read

# The board TSV contract (8 fields per detection, ONE TAB-SEPARATED COLUMN per detection):
#   <stem>\t<ndet>\t<tag> <cx> <cy> <w> <h> <area> <peak> <sum>\t...
BOARD_DET_FIELDS = 8

# The 8 timing segments the board program is expected to report.
TIMING_SEGMENTS = (
    "decode",
    "rga_import",
    "rga_improcess",
    "fp16_or_upload",
    "rknn_run",
    "outputs_get",
    "backproject_cc",
    "write",
)

ARMS = ("pc-all", "pc-native", "board-mosaic", "board-mosaic-native", "board-single")
PC_ARMS = ("pc-all", "pc-native")
BOARD_ARMS = ("board-mosaic", "board-mosaic-native", "board-single")

ARM_SPEC = {
    "pc-all": {
        "kind": "pc",
        "desc": "PC f32, 3 scales (native/160/80), 3 forward passes -- the frozen baseline",
    },
    "pc-native": {
        "kind": "pc",
        "desc": "PC f32, tag 0 only (same pkl, subsetted) -- native single scale",
    },
    "board-mosaic": {
        "kind": "board",
        "canvas_mode": "mosaic",
        "tiles": "0,1,2",
        "desc": "board f16, mosaic canvas with 3 tiles -- the product",
    },
    "board-mosaic-native": {
        "kind": "board",
        "canvas_mode": "mosaic",
        "tiles": "0",
        "desc": "board f16, mosaic canvas, native tile only -- isolates the canvas perturbation",
    },
    "board-single": {
        "kind": "board",
        "canvas_mode": "single",
        "tiles": "0",
        "desc": "board f16, single-scale canvas -- isolates f16 + board quantisation",
    },
}

# --------------------------------------------------------------------------------------
# Default locations.  Every one of them is overridable (CLI > env > default) so the
# script runs unchanged on the workstation and on the training server.
# --------------------------------------------------------------------------------------
DEF_CACHE_ROOT = os.environ.get(
    "MOSAIC_PC_CACHE_ROOT",
    "/tmp/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache",
)
DEF_LABELS_ROOT = os.environ.get(
    "MOSAIC_LABELS_ROOT",
    "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010",
)
DEF_GATE_JSON = os.environ.get("MOSAIC_GATE_JSON", "")
DEF_OUT = os.environ.get("MOSAIC_WORK_ROOT", "/mnt/data/siping/ptq_work/mosaic_vs_f32")
DEF_BOARD_HOST = os.environ.get("MOSAIC_BOARD_HOST", "root@192.168.0.64")
DEF_BOARD_PORT = int(os.environ.get("MOSAIC_BOARD_PORT", "22"))
DEF_BOARD_NFS = os.environ.get("MOSAIC_BOARD_NFS", "/mnt/manu/ptq")
DEF_BOARD_BIN_LOCAL = os.environ.get(
    "MOSAIC_BOARD_BIN", str(REPO_ROOT / "manu" / "pipeline" / "npu" / "out" / "rknn_mosaic_heatmap")
)
DEF_BOARD_BIN_REMOTE = os.environ.get("MOSAIC_BOARD_BIN_REMOTE", "/root/rknn_mosaic_heatmap")
DEF_BOARD_MODEL = os.environ.get("MOSAIC_BOARD_MODEL", "/mnt/manu/npu/data/trial0474_hm_fp16.rknn")
DEF_BOARD_OUT = os.environ.get("MOSAIC_BOARD_OUT", "/mnt/manu/ptq/mosaic_out")

# The board only tolerates 2 RKNN contexts; a leftover process from another run
# silently corrupts a measurement, so the driver refuses to start with one alive.
BOARD_MAX_RKNN_CTX = 0
BOARD_ROOT_SPACE_MARGIN_MB = 64


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[35] {msg}", flush=True)


def die(msg: str, code: int = 2) -> "NoReturn":  # noqa: F821
    print(f"[35][FATAL] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def sh(cmd: list[str], timeout: int = 3600) -> tuple[int, str, str]:
    """Run a command, capture stdout/stderr.  Never raises on a non-zero exit code."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(path: Path, default=None):
    if not Path(path).is_file():
        return default
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# --------------------------------------------------------------------------------------
# params: taken from the gate product, never hand-copied
# --------------------------------------------------------------------------------------
def resolve_params_json(args) -> tuple[Path, str]:
    """Return (params json path, the gate json it came from).

    Priority: --params-json  >  --gate-json (the gate product's own json)  >  auto-discovery
    next to the cache root.  The params are EXTRACTED from ``config.params`` of the
    gate json -- hand-writing them is how the 0.22 / 0.30 confusion started.
    """
    if args.params_json:
        path = Path(args.params_json)
        if not path.is_file():
            die(f"--params-json does not exist: {path}")
        return path, ""

    gate = args.gate_json
    if not gate:
        candidate = Path(args.cache_root).parent / "track3_blind.json"
        gate = str(candidate) if candidate.is_file() else ""
    if not gate:
        die("cannot find the gate json -- pass --gate-json (the product that owns the "
            "frozen numbers) or --params-json (an already extracted params file)")
    gate_path = Path(gate)
    if not gate_path.is_file():
        die(f"--gate-json does not exist: {gate_path}")
    payload = read_json(gate_path)
    if not isinstance(payload, dict) or "config" not in payload or "params" not in payload["config"]:
        die(f"{gate_path} has no config.params -- it is not a gate product json")
    out = Path(args.work) / "gate_params.json"
    write_json(out, payload["config"]["params"])
    log(f"params extracted from {gate_path} -> {out}")
    log(f"  th_base={payload['config']['params'].get('th_base')} "
        f"dedup_radius={payload['config']['params'].get('dedup_radius')}")
    return out, str(gate_path.resolve())


def load_params(params_json: Path) -> dict:
    payload = read_json(params_json)
    return payload.get("params", payload) if isinstance(payload, dict) else {}


# --------------------------------------------------------------------------------------
# the scorer -- one path, used by every arm and by the self-check
# --------------------------------------------------------------------------------------
def score_cache(cache_root: Path, labels_root: Path, sequence: str, params_json: Path,
                out_json: Path, log_path: Path) -> dict:
    """Run the frozen scorer on one cache root and return its per-sequence metrics.

    The scorer is invoked as a SUBPROCESS on purpose: the arm numbers must come out
    of ``system_sota_longquan.py`` itself, not out of a re-implementation of it here.
    """
    if not SCORER.is_file():
        die(f"the scorer is missing: {SCORER}")
    cmd = [
        sys.executable, str(SCORER),
        "--cache-root", str(cache_root),
        "--labels-root", str(labels_root),
        "--seq", sequence,
        "--no-pipeline",
        "--params-json", str(params_json),
        "--out-json", str(out_json),
    ]
    rc, out, err = sh(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n",
        encoding="utf-8",
    )
    if rc != 0:
        die(f"the scorer failed (rc={rc}); see {log_path}\n{err[-2000:]}", code=1)
    payload = read_json(out_json)
    if not payload:
        die(f"the scorer wrote no readable report: {out_json}", code=1)
    metrics = payload.get("per_sequence", {}).get(sequence)
    if not metrics:
        die(f"the scorer report has no per_sequence[{sequence}]: {out_json}", code=1)
    return metrics


# --------------------------------------------------------------------------------------
# ms_cache surgery
# --------------------------------------------------------------------------------------
def load_cache(cache_root: Path, sequence: str) -> dict:
    path = Path(cache_root) / f"{sequence}.pkl"
    if not path.is_file():
        die(f"cache not found: {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def subset_cache(cache: dict, keep_frames: set[int] | None = None,
                 keep_tags: set[int] | None = None) -> dict:
    """Rebuild a cache with a subset of frames and/or tags.

    The offsets are rebuilt from scratch -- a cache whose offset does not match its
    rows is silently truncated by every reader, so this never reuses the old offsets.
    """
    frame_index = cache["frame_index"]
    offset = cache["offset"]
    rows = {key: [] for key in ("points", "sizes", "scores", "sums", "area", "tag")}
    new_frame_index, new_offset = [], [0]
    for position in range(len(frame_index)):
        index = int(frame_index[position])
        if keep_frames is not None and index not in keep_frames:
            continue
        start, end = int(offset[position]), int(offset[position + 1])
        selected = np.arange(start, end, dtype=np.int64)
        if keep_tags is not None and len(selected):
            selected = selected[np.isin(cache["tag"][selected], list(keep_tags))]
        for key in rows:
            rows[key].append(cache[key][selected])
        new_frame_index.append(index)
        new_offset.append(new_offset[-1] + int(len(selected)))
    out = {"meta": dict(cache["meta"])}
    out["frame_index"] = np.asarray(new_frame_index, dtype=np.int32)
    out["offset"] = np.asarray(new_offset, dtype=np.int64)
    for key in ("points", "sizes", "scores", "sums", "area", "tag"):
        out[key] = np.concatenate(rows[key]) if rows[key] else cache[key][:0]
    return out


def write_cache(cache: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def per_tag_counts(cache: dict) -> dict:
    counts = {str(tag): 0 for tag in SCALE_TAGS}
    tags = cache["tag"]
    for tag in SCALE_TAGS:
        counts[str(tag)] = int(np.count_nonzero(tags == tag))
    counts["255"] = int(np.count_nonzero(tags == FUSED_TAG))
    return counts


# --------------------------------------------------------------------------------------
# the board TSV
# --------------------------------------------------------------------------------------
def parse_mosaic_tsv(path: Path) -> dict[str, list[dict]]:
    """Parse the board mosaic TSV into {stem: [det, ...]}.

    The board writes ONE TAB-SEPARATED COLUMN PER DETECTION (same convention as
    rknn_heatmap_smoke.cpp and 04_run_board_fast_eval.py:88), so a frame with
    3 detections has 5 tab-separated columns -- NOT 3 columns with 3 fields each.
    Reading only column 2 (and stopping when it runs out) keeps the first
    detection of every frame and throws the rest away; that bug is documented in
    04_run_board_fast_eval.py, so it is not repeated here.

    A detection is ``tag cx cy w h area peak sum``:
        tag   0=native(640x512) 1=thumb160(160x128) 2=thumb80(80x64)
        cx cy the centre in NATIVE 640x512 pixels
        w h   the connected-component bounding box size (integer px)
        area  the connected-component pixel count (integer)
        peak  the heatmap maximum inside the component  -> the score
        sum   the heatmap integral inside the component
    """
    if not path.is_file():
        die(f"the board TSV is missing: {path}")
    out: dict[str, list[dict]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            die(f"{path}:{lineno}: a TSV line needs at least <stem>\\t<ndet>, got {line[:120]!r}")
        stem = parts[0].strip()
        # the stem must carry the frame number: the PC-side frame indices are
        # 0..N-1 and the two sides are matched on them
        stem_to_frame_index(stem)
        try:
            ndet = int(parts[1].strip())
        except ValueError:
            die(f"{path}:{lineno}: <ndet> is not an integer: {parts[1]!r}")
        if ndet < 0:
            die(f"{path}:{lineno}: negative <ndet>={ndet}")
        if len(parts) - 2 < ndet:
            die(f"{path}:{lineno}: <ndet>={ndet} but only {len(parts) - 2} detection column(s) present")
        dets = []
        for k in range(ndet):
            fields = parts[2 + k].split()
            if len(fields) < BOARD_DET_FIELDS:
                die(f"{path}:{lineno}: detection {k} has {len(fields)} field(s), "
                    f"expected {BOARD_DET_FIELDS} (tag cx cy w h area peak sum)")
            try:
                tag = int(fields[0])
                cx, cy = float(fields[1]), float(fields[2])
                box_w, box_h = int(fields[3]), int(fields[4])
                area = int(fields[5])
                peak = float(fields[6])
                total = float(fields[7])
            except ValueError as exc:
                die(f"{path}:{lineno}: detection {k} is not parsable: {exc}")
            if tag not in SCALE_TAGS:
                die(f"{path}:{lineno}: detection {k} has tag={tag}, expected one of {SCALE_TAGS}")
            if area < 0 or box_w < 0 or box_h < 0:
                die(f"{path}:{lineno}: detection {k} has a negative geometry")
            dets.append({
                "tag": tag, "cx": cx, "cy": cy, "w": box_w, "h": box_h,
                "area": area, "peak": peak, "sum": total,
            })
        if stem in out:
            die(f"{path}:{lineno}: duplicate frame {stem}")
        out[stem] = dets
    return out


def stem_to_frame_index(stem: str) -> int:
    """'VIDEO00007_19700101_001324__000123' -> 123."""
    if "__" not in stem:
        die(f"the frame stem carries no frame number: {stem!r}")
    tail = stem.rsplit("__", 1)[1]
    if not tail.isdigit():
        die(f"the frame stem does not end in a frame number: {stem!r}")
    return int(tail)


def build_board_cache(dets_by_stem: dict[str, list[dict]], frame_indices: list[int],
                      meta: dict) -> dict:
    """Turn the board TSV into an ms_cache-shaped dict.

    The dtype layout is copied from manu/data/multiscale_heatmap_cache.py:437-476
    EXACTLY, because the scorer reads this file with the same reader:

        frame_index int32 (N,)     offset int64 (N+1,)
        points     float32 (M,2)   sizes  float32 (M,2)
        scores     float16 (M,)    sums   float32 (M,)
        area       int32   (M,)    tag    uint8  (M,)

    There is NO tag=255 fused row: the evaluation arms never read it, and a
    fabricated fused row would be a fake number in an auditable table.
    """
    points, sizes, scores, sums, areas, tags = [], [], [], [], [], []
    frame_list = []
    offset = [0]
    by_index: dict[int, str] = {}
    for stem in dets_by_stem:
        index = stem_to_frame_index(stem)
        if index in by_index:
            die(f"two TSV rows carry the same frame index {index}")
        by_index[index] = stem
    for index in frame_indices:
        stem = by_index.get(index)
        if stem is None:
            die(f"the board TSV has no frame {index}")
        dets = dets_by_stem[stem]
        for det in dets:
            points.append((det["cx"], det["cy"]))
            sizes.append((det["w"], det["h"]))
            scores.append(det["peak"])
            sums.append(det["sum"])
            areas.append(det["area"])
            tags.append(det["tag"])
        frame_list.append(index)
        offset.append(offset[-1] + len(dets))
    return {
        "meta": meta,
        "frame_index": np.asarray(frame_list, dtype=np.int32),
        "offset": np.asarray(offset, dtype=np.int64),
        "points": np.asarray(points, dtype=np.float32).reshape(-1, 2),
        "sizes": np.asarray(sizes, dtype=np.float32).reshape(-1, 2),
        "scores": np.asarray(scores, dtype=np.float16),
        "sums": np.asarray(sums, dtype=np.float32),
        "area": np.asarray(areas, dtype=np.int32),
        "tag": np.asarray(tags, dtype=np.uint8),
    }


# --------------------------------------------------------------------------------------
# the board driver
# --------------------------------------------------------------------------------------
class Board:
    """Everything that touches the board over ssh."""

    def __init__(self, host: str, port: int, timeout: int = 3600):
        self.host = host
        self.port = port
        self.timeout = timeout

    def _ssh_base(self) -> list[str]:
        return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
                "-p", str(self.port), self.host]

    def sh(self, command: str, timeout: int | None = None) -> tuple[int, str, str]:
        return sh(self._ssh_base() + [command], timeout or self.timeout)

    def scp_to(self, local: Path, remote: str) -> tuple[int, str, str]:
        cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
               "-P", str(self.port), str(local), f"{self.host}:{remote}"]
        return sh(cmd, self.timeout)

    def scp_from(self, remote: str, local: Path) -> tuple[int, str, str]:
        local.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
               "-P", str(self.port), f"{self.host}:{remote}", str(local)]
        return sh(cmd, self.timeout)

    def put_text(self, remote: str, text: str) -> tuple[int, str]:
        """Write a text file on the board over ssh stdin (no shell quoting games)."""
        proc = subprocess.Popen(self._ssh_base() + [f"cat > {remote}"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate(text.encode("utf-8"), timeout=self.timeout)
        return proc.returncode, (out + err).decode("utf-8", "replace")

    # -- guards -----------------------------------------------------------------
    def rknn_context_count(self) -> int:
        """How many RKNN contexts are alive on the board right now.

        The board's ``ps -eo comm`` truncates the command name to 15 characters, so
        the check matches the ``^rknn_`` PREFIX instead of the full binary name.
        """
        rc, out, err = self.sh("ps -eo comm | grep -c '^rknn_'")
        text = out.strip().splitlines()
        if not text:
            die(f"cannot read the board's process list (rc={rc}): {err[:300]}")
        try:
            return int(text[0].strip())
        except ValueError:
            die(f"cannot parse the board's RKNN context count: {out[:200]!r}")

    def root_free_mb(self) -> float:
        rc, out, err = self.sh("df -k /root | tail -1")
        if rc != 0:
            die(f"cannot read the board's /root free space: {err[:300]}")
        fields = out.split()
        if len(fields) < 4:
            die(f"cannot parse `df -k /root`: {out.strip()!r}")
        return float(fields[3]) / 1024.0

    def list_bmp(self, board_dir: str) -> list[str]:
        """The board's absolute BMP paths in that directory, sorted by file name."""
        rc, out, err = self.sh(f"ls -1 {board_dir}/*.bmp 2>/dev/null")
        if rc != 0:
            die(f"cannot list the board features {board_dir}/*.bmp: {err[:300]}")
        paths = [line.strip() for line in out.splitlines() if line.strip().endswith(".bmp")]
        return sorted(paths, key=lambda item: item.rsplit("/", 1)[-1])


def select_board_frames(bmp_paths: list[str], start: int, limit: int | None) -> list[str]:
    """Slice the board's BMP list into the arm's frame slice.

    ``--start`` skips frames at the HEAD of the sequence and ``--limit`` caps the
    count.  The manifest is then exactly the arm's slice, so the board is never
    asked to skip it a second time (that would desynchronise the frame indices).
    """
    out = list(bmp_paths[start:] if start else bmp_paths)
    if limit is not None:
        out = out[:limit]
    return out


def build_board_command(args, remote_bin: str, remote_manifest: str, board_out: str,
                        timing_remote: str, tiles: str, canvas_mode: str,
                        supported: set[str]) -> tuple[str, list[str]]:
    """Build the board command line from the flags the binary actually advertises.

    Returns (command, warnings).  The board program is owned by the C++ worker and
    its CLI grew flags on top of the original contract, so nothing is assumed: a
    flag that the binary does not advertise is simply not passed (and the warning
    says so), instead of making the program die on an unknown option.
    """
    cmd = (f"{remote_bin} --model {args.board_model} --list {remote_manifest} "
           f"--out {board_out}")
    warnings = []
    if "--tiles" in supported:
        cmd += f" --tiles {tiles}"
    else:
        warnings.append(f"the board binary does not advertise --tiles -- the arm's tile "
                        f"selection ({tiles}) CANNOT be expressed")
    if "--canvas-mode" in supported:
        cmd += f" --canvas-mode {canvas_mode}"
    else:
        raise SystemExit("[35][FATAL] the board binary does not advertise --canvas-mode "
                         "-- it cannot run the mosaic / single arms")
    if "--timing" in supported:
        cmd += f" --timing {timing_remote}"
    if "--input-mode" in supported and args.board_input_mode:
        cmd += f" --input-mode {args.board_input_mode}"
    if "--cpu-affinity" in supported and args.board_cpu:
        cmd += f" --cpu-affinity {args.board_cpu}"
    elif args.board_cpu:
        cmd = f"taskset -c {args.board_cpu} {cmd}"
    if limit_value(args) is not None:
        cmd += f" --limit {limit_value(args)}"
    return cmd, warnings


def limit_value(args) -> int | None:
    return args.limit


def probe_board_flags(board: Board, remote_bin: str) -> dict:
    """Ask the board binary which flags it understands.

    The board program is owned by the C++ worker and its CLI grew an
    ``--input-mode`` flag on top of the contract this script was written against,
    so the driver PROBES instead of assuming: only the flags the binary actually
    advertises are passed.  ``--help`` is handled by the program before it opens an
    RKNN context (rknn_mosaic_heatmap.cpp:851), so the probe is side-effect free.
    """
    rc, out, err = board.sh(f"{remote_bin} --help 2>&1 | head -40")
    text = (out or "") + (err or "")
    flags = sorted(set(_FLAG.findall(text)))
    return {"flags": flags, "help": text[:4000]}


def push_binary(board: Board, local_bin: Path, remote_bin: str) -> dict:
    """Copy the board binary to the board and prove the bytes are identical."""
    if not local_bin.is_file():
        die(f"the board binary does not exist: {local_bin}\n"
            "       (it is produced by the C++ worker; build it before running the board arms)")
    size_mb = local_bin.stat().st_size / (1024 * 1024)
    free_mb = board.root_free_mb()
    log(f"board /root free: {free_mb:.1f} MB, binary: {size_mb:.2f} MB")
    if free_mb < size_mb + BOARD_ROOT_SPACE_MARGIN_MB:
        die(f"not enough room on the board's /root: {free_mb:.1f} MB free, "
            f"need {size_mb + BOARD_ROOT_SPACE_MARGIN_MB:.1f} MB")
    local_md5 = md5_file(local_bin)
    rc, out, err = board.sh(f"md5sum {remote_bin} 2>/dev/null")
    previous_md5 = out.split()[0] if (rc == 0 and out.split()) else ""
    if previous_md5 and previous_md5 != local_md5:
        log(f"[warn] the board already has a DIFFERENT binary at {remote_bin} "
            f"({previous_md5}); it is being replaced by {local_md5}")
    rc, _out, err = board.scp_to(local_bin, remote_bin)
    if rc != 0:
        die(f"cannot push the board binary: {err[:300]}")
    rc, out, err = board.sh(f"md5sum {remote_bin}")
    remote_md5 = out.split()[0] if (rc == 0 and out.split()) else ""
    if remote_md5 != local_md5:
        die(f"the board binary md5 MISMATCH after the push\n"
            f"       local : {local_md5}\n       board : {remote_md5 or 'none'}")
    log(f"board binary pushed and md5-verified: {remote_md5}")
    return {"local": str(local_bin), "remote": remote_bin, "md5": local_md5,
            "previous_md5_on_board": previous_md5}


def board_model_md5(board: Board, board_model: str) -> str:
    """The md5 of the model the board will actually run."""
    rc, out, err = board.sh(f"md5sum {board_model}")
    if rc != 0 or not out.split():
        die(f"cannot read the model md5 on the board: {board_model}: {err[:300]}")
    return out.split()[0]


# --------------------------------------------------------------------------------------
# timing.txt -- the board program owns the format, so the parser is tolerant
# --------------------------------------------------------------------------------------
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_FLAG = re.compile(r"--[a-z0-9][a-z0-9_-]*")


def parse_timing(text: str) -> dict:
    """Parse the board's timing.txt into {segments_ms, ms_per_frame, wall_ms, fps, raw}.

    The board program owns this format (manu/pipeline/npu/rknn_mosaic_heatmap.cpp:1344-1366):

        # rknn_mosaic_heatmap timing
        canvas_mode=mosaic
        input_mode=set
        seq=VIDEO00007_19700101_001324
        frames=11810
        segment           total_ms    avg_ms/frame   frames
        decode                12.500         0.0010     11810
        ...
        sum_of_segments      123.456         0.0105     11810
        total_wall          456.789         0.0388     11810
        fps=15.290

    The parser is deliberately tolerant: it understands that table layout, the
    ``key: value`` / ``key = value`` one-liners, and it keeps the raw text next to
    the parsed values so the report can quote the original numbers verbatim.
    """
    result: dict = {
        "segments_ms": {}, "segments_ms_per_frame": {}, "wall_ms": None, "fps": None,
        "frames": None, "canvas_mode": None, "input_mode": None, "seq": None,
        "sum_of_segments_ms": None, "other": {}, "raw": text,
    }
    if not text.strip():
        return result

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        name = parts[0].lower()
        numbers = []
        for token in parts[1:]:
            try:
                numbers.append(float(token))
            except ValueError:
                break
        # the table rows
        if name in TIMING_SEGMENTS and numbers:
            result["segments_ms"][name] = numbers[0]
            if len(numbers) >= 2:
                result["segments_ms_per_frame"][name] = numbers[1]
        elif name in ("sum_of_segments", "sum_of_segment"):
            if numbers:
                result["sum_of_segments_ms"] = numbers[0]
        elif name in ("total_wall", "wall", "wall_ms", "wallclock", "wall_clock", "elapsed"):
            if numbers:
                result["wall_ms"] = numbers[0]
        elif name == "fps" and numbers:
            result["fps"] = numbers[0]
        # the key=value one-liners
        if "=" in stripped and " " not in stripped.split("=", 1)[0]:
            key, _, value = stripped.partition("=")
            key, value = key.strip().lower(), value.strip()
            if key in ("canvas_mode", "input_mode", "seq"):
                result[key] = value
            elif key == "frames":
                try:
                    result["frames"] = int(value)
                except ValueError:
                    pass
            elif key == "fps":
                try:
                    result["fps"] = float(value)
                except ValueError:
                    pass
            else:
                try:
                    result["other"][key] = float(value)
                except ValueError:
                    result["other"][key] = value

    # tolerant fallback for a "decode: 12.5 ms" style file
    for line in text.splitlines():
        low = line.lower()
        for segment in TIMING_SEGMENTS:
            if segment in low and segment not in result["segments_ms"]:
                tail = low.split(segment, 1)[1]
                found = _NUMBER.findall(tail)
                if found:
                    result["segments_ms"][segment] = float(found[0])
    if result["wall_ms"] is None:
        for line in text.splitlines():
            low = line.lower()
            if "wall" in low:
                found = _NUMBER.findall(low.split("wall", 1)[1])
                if found:
                    result["wall_ms"] = float(found[0])
                    break
    if result["fps"] is None:
        for line in text.splitlines():
            low = line.lower()
            if "fps" in low:
                found = _NUMBER.findall(low)
                if found:
                    result["fps"] = float(found[-1])
                    break
    return result


# --------------------------------------------------------------------------------------
# the GT subset -- so a --limit run scores exactly the same frames on both sides
# --------------------------------------------------------------------------------------
def build_gt_subset(labels_root: Path, sequence: str, frames: list[int], target: Path) -> Path:
    """A labels root that contains ONLY the selected frames (as symlinks).

    The scorer unions the cache frames with the ground-truth frames, so an
    unfiltered GT root would add every other frame of the sequence as a false
    negative and make a --limit run incomparable.  Symlinks keep the bytes in one
    place; the scorer only ever reads them.
    """
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    missing = []
    for index in frames:
        source = Path(labels_root) / f"{sequence}__{index:06d}.txt"
        if not source.is_file():
            missing.append(source.name)
            continue
        link = target / source.name
        if not link.exists():
            link.symlink_to(source)
    if missing:
        die(f"{len(missing)} ground-truth file(s) are missing under {labels_root} "
            f"(first: {missing[0]})")
    return target


# --------------------------------------------------------------------------------------
# the PC arms
# --------------------------------------------------------------------------------------
def run_pc_arm(args, arm: str, cache: dict, params_json: Path, labels_root: Path,
               out_dir: Path) -> dict:
    """Score one PC arm.  ``cache`` is already the subsetted cache for this arm."""
    cache_dir = out_dir / "ms_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl = write_cache(cache, cache_dir / f"{args.sequence}.pkl")

    metrics = score_cache(cache_dir, labels_root, args.sequence, params_json,
                          out_dir / "score.json", out_dir / "score.log")
    return {
        "arm": arm,
        "kind": "pc",
        "description": ARM_SPEC[arm]["desc"],
        "sequence": args.sequence,
        "frames": int(len(cache["frame_index"])),
        "limit": args.limit,
        "start": args.start,
        "derived_from": (
            f"{args.cache_root}/{args.sequence}.pkl, tag 0 rows only"
            if arm == "pc-native" else
            f"{args.cache_root}/{args.sequence}.pkl (used as is, never rebuilt)"
        ),
        "metrics": metrics,
        "per_tag_counts": per_tag_counts(cache),
        "artifacts": {
            "pkl": str(pkl),
            "score_json": str(out_dir / "score.json"),
            "score_log": str(out_dir / "score.log"),
        },
    }


# --------------------------------------------------------------------------------------
# the board arms
# --------------------------------------------------------------------------------------
def run_board_arm(args, arm: str, board: Board, pc_cache: dict, pc_frames: list[int],
                  pc_frames_all: list[int], params_json: Path, labels_root: Path,
                  out_dir: Path) -> dict:
    """Drive the board for one arm and score the result with the frozen scorer."""
    spec = ARM_SPEC[arm]
    canvas_mode = args.canvas_mode or spec["canvas_mode"]
    tiles = args.tiles if args.tiles is not None else spec["tiles"]

    # -- 1) the board must be idle ------------------------------------------------
    contexts = board.rknn_context_count()
    log(f"board RKNN contexts in use: {contexts}")
    if contexts > BOARD_MAX_RKNN_CTX:
        die(f"the board still has {contexts} RKNN context(s) alive "
            f"(limit {BOARD_MAX_RKNN_CTX}); a leftover process corrupts the measurement")

    # -- 2) the manifest -----------------------------------------------------------
    board_features = args.board_features or f"{args.board_nfs}/bmp/{args.sequence}"
    bmp_paths = board.list_bmp(board_features)
    if not bmp_paths:
        die(f"the board feature directory is empty: {board_features}")
    # The board must hold the WHOLE sequence: the manifest is truncated afterwards,
    # but a board that is missing frames would silently shrink the comparison.
    if len(bmp_paths) != len(pc_frames_all):
        die(f"the board has {len(bmp_paths)} BMP(s) under {board_features} but the PC cache "
            f"has {len(pc_frames_all)} frame(s) -- the two sides are not the same frame set")
    # --start selects a slice of the sequence; the manifest is already that slice,
    # so the board is never asked to skip it a second time.
    bmp_paths = select_board_frames(bmp_paths, int(args.start or 0), args.limit)
    # the BMP file names must map onto the PC frame indices, in the same order
    board_frame_indices = [
        stem_to_frame_index(path.rsplit("/", 1)[-1].rsplit(".", 1)[0]) for path in bmp_paths
    ]
    if board_frame_indices != list(pc_frames):
        die(f"the board BMPs do not line up with the PC frames\n"
            f"       board: {board_frame_indices[:5]}... ({len(board_frame_indices)})\n"
            f"       pc   : {list(pc_frames)[:5]}... ({len(pc_frames)})")
    manifest_lines = list(bmp_paths)
    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    log(f"manifest: {len(manifest_lines)} BMP(s) -> {manifest}")

    # -- 3) the binary and the model ------------------------------------------------
    binary = push_binary(board, Path(args.board_bin), args.board_bin_remote)
    model_md5 = board_model_md5(board, args.board_model)
    probe = probe_board_flags(board, args.board_bin_remote)
    supported = set(probe["flags"])
    log(f"board binary flags: {probe['flags']}")

    # -- 4) the board run -----------------------------------------------------------
    board_out = f"{args.board_out}/{arm}"
    board_run_dir = f"{args.board_out}/_run_{arm}"
    rc, out, err = board.sh(f"mkdir -p {board_run_dir}")
    if rc != 0:
        die(f"cannot create the board output dir {board_run_dir}: {err[:300]}")
    timing_remote = f"{board_run_dir}/timing.txt"
    stdout_remote = f"{board_run_dir}/stdout.log"
    stderr_remote = f"{board_run_dir}/stderr.log"
    # a stale TSV from an earlier run would be pulled back and scored as if it
    # were this run's output -> it is deleted before the run starts
    rc, _out, err = board.sh(f"rm -f {board_out}/{args.sequence}_mosaic.tsv "
                              f"{board_out}/timing.txt {timing_remote}")
    remote_manifest = f"{board_run_dir}/manifest.txt"
    rc, err = board.put_text(remote_manifest, "\n".join(manifest_lines) + "\n")
    if rc != 0:
        die(f"cannot push the manifest to the board: {err[:300]}")

    cmd, warnings = build_board_command(
        args, args.board_bin_remote, remote_manifest, board_out, timing_remote,
        tiles, canvas_mode, supported)
    for warning in warnings:
        log(f"[warn] {warning}")
    # The board's librga writes debug logs to STDOUT, so stdout is NEVER a data
    # channel here: it is redirected to a file and the TSV is read from disk.
    run_cmd = f"{cmd} > {stdout_remote} 2> {stderr_remote}"
    log(f"board run: {run_cmd}")
    started = time.time()
    rc, out, err = board.sh(run_cmd, timeout=args.board_timeout)
    elapsed = time.time() - started
    log(f"board run finished rc={rc} in {elapsed:.1f}s")
    if rc != 0:
        rc2, so, se = board.sh(f"tail -40 {stderr_remote} 2>/dev/null; tail -20 {stdout_remote} 2>/dev/null")
        die(f"the board run failed (rc={rc}); tail of the board logs:\n{so[-1500:]}\n{se[-1500:]}",
            code=1)

    # -- 5) pull the artefacts back --------------------------------------------------
    tsv_name = f"{args.sequence}_mosaic.tsv"
    remote_tsv = f"{board_out}/{tsv_name}"
    rc, out, err = board.sh(f"ls -1 {board_out} 2>/dev/null")
    if rc != 0 or tsv_name not in out.split():
        die(f"the board produced no {tsv_name} under {board_out}; board dir listing:\n{out[:1000]}")
    local_tsv = out_dir / tsv_name
    rc, _out, err = board.scp_from(remote_tsv, local_tsv)
    if rc != 0:
        die(f"cannot pull the board TSV {remote_tsv}: {err[:300]}")
    local_timing = out_dir / "timing.txt"
    rc, _out, err = board.scp_from(timing_remote, local_timing)
    if rc != 0:
        # the board writes <out>/timing.txt when --timing is not given
        rc, _out, err = board.scp_from(f"{board_out}/timing.txt", local_timing)
    if rc != 0:
        log(f"[warn] cannot pull the timing file: {err[:200]}")
    local_stdout = out_dir / "board_stdout.log"
    local_stderr = out_dir / "board_stderr.log"
    board.scp_from(stdout_remote, local_stdout)
    board.scp_from(stderr_remote, local_stderr)

    # the board prints its own run summary on stderr -- keep it, it is the audit
    # trail of what actually ran (canvas mode, input mode, frame count)
    board_summary = ""
    if local_stderr.is_file():
        for line in local_stderr.read_text(encoding="utf-8", errors="replace").splitlines():
            if "canvas-mode=" in line:
                board_summary = line.strip()
                break

    # -- 6) the TSV -> ms_cache --------------------------------------------------------
    dets_by_stem = parse_mosaic_tsv(local_tsv)
    board_frames = sorted(stem_to_frame_index(stem) for stem in dets_by_stem)
    expected_frames = sorted(pc_frames)
    if board_frames != expected_frames:
        only_board = sorted(set(board_frames) - set(expected_frames))[:5]
        only_pc = sorted(set(expected_frames) - set(board_frames))[:5]
        die(f"the board frame indices do not match the PC cache\n"
            f"       board-only: {only_board}\n       pc-only: {only_pc}\n"
            f"       (board {len(board_frames)} frames vs pc {len(expected_frames)} frames)")

    meta = {
        "sequence": args.sequence,
        "frames": len(board_frames),
        "native_size": [NATIVE_SIZE[0], NATIVE_SIZE[1]],
        "imgsz": 640,
        "stride": 2,
        "scales": list(SCALES),
        "fused_tag": FUSED_TAG,
        "coordinate_space": "native_pixel",
        "source": "board_mosaic_f16",
        "region_threshold": 0.06,
        "main_threshold": 0.22,
        "fusion_dilate": 32,
        "min_area": 4,
        # the PC-side cache leaves features_root EMPTY -- a known provenance gap.
        # The board product closes it: the exact BMP directory the board read.
        "features_root": board_features,
        "hm_weights": args.board_model,
        "use_p0_highway": False,
        "adapter_weights": "",
        "adapter_mid_channels": 0,
        "adapter_guide_channel": -1,
        "adapter_single": False,
        # --- provenance of the board product (not part of the PC cache layout) ---
        "board_binary": args.board_bin_remote,
        "board_binary_md5": binary["md5"],
        "board_model": args.board_model,
        "board_model_md5": model_md5,
        "board_arm": arm,
        "board_canvas_mode": canvas_mode,
        "board_tiles": tiles,
        "board_output_dir": board_out,
    }
    board_cache = build_board_cache(dets_by_stem, board_frames, meta)
    cache_dir = out_dir / "ms_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl = write_cache(board_cache, cache_dir / f"{args.sequence}.pkl")

    # -- 7) score it with the SAME scorer and the SAME params ----------------------------
    metrics = score_cache(cache_dir, labels_root, args.sequence, params_json,
                          out_dir / "score.json", out_dir / "score.log")

    timing = parse_timing(local_timing.read_text(encoding="utf-8") if local_timing.is_file() else "")
    frames = max(1, len(board_frames))
    timing["frames"] = timing.get("frames") or len(board_frames)
    # the board's own per-frame column wins; total/frames is the fallback
    timing["ms_per_frame"] = {
        segment: timing.get("segments_ms_per_frame", {}).get(
            segment, value / frames)
        for segment, value in timing["segments_ms"].items()
    }
    timing["wall_seconds_measured"] = elapsed

    return {
        "arm": arm,
        "kind": "board",
        "description": spec["desc"],
        "sequence": args.sequence,
        "frames": len(board_frames),
        "limit": args.limit,
        "start": args.start,
        "metrics": metrics,
        "per_tag_counts": per_tag_counts(board_cache),
        "board": {
            "host": board.host,
            "port": board.port,
            "canvas_mode": canvas_mode,
            "tiles": tiles,
            "input_mode": timing.get("input_mode") or args.board_input_mode,
            "model": args.board_model,
            "model_md5": model_md5,
            "binary": binary,
            "output_dir": board_out,
            "run_command": run_cmd,
            "supported_flags": probe["flags"],
            "board_summary": board_summary,
            "wall_seconds": elapsed,
        },
        "timing": timing,
        "artifacts": {
            "pkl": str(pkl),
            "tsv": str(local_tsv),
            "timing": str(local_timing) if local_timing.is_file() else "",
            "manifest": str(manifest),
            "score_json": str(out_dir / "score.json"),
            "score_log": str(out_dir / "score.log"),
            "board_stdout": str(local_stdout),
            "board_stderr": str(local_stderr),
        },
    }


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------
def run_self_check(args, params_json: Path, labels_root: Path) -> int:
    """Reproduce the frozen T3 baseline through the EXACT scoring path the arms use.

    If this does not reproduce tp/fp/fn/gt/f1/det bit for bit, the whole comparison
    table is meaningless -- so the script exits non-zero and says what differs.
    """
    cache_root = Path(args.cache_root)
    metrics = score_cache(cache_root, labels_root, args.sequence, params_json,
                          Path(args.work) / "self_check_score.json",
                          Path(args.work) / "self_check_score.log")
    params = load_params(params_json)
    log("self-check: expected vs actual")
    log(f"  {'key':>6} {'expected':>18} {'actual':>18}  ok")
    failures = []
    for key in ("tp", "fp", "fn", "gt", "det"):
        expected, actual = FROZEN_T3[key], int(metrics.get(key, -1))
        ok = expected == actual
        failures.append(key) if not ok else None
        log(f"  {key:>6} {expected:>18d} {actual:>18d}  {'OK' if ok else 'MISMATCH'}")
    f1_expected, f1_actual = FROZEN_T3["f1"], float(metrics.get("f1", -1.0))
    f1_ok = abs(f1_actual - f1_expected) < FROZEN_T3_F1_TOL
    if not f1_ok:
        failures.append("f1")
    log(f"  {'f1':>6} {f1_expected:>18.12f} {f1_actual:>18.12f}  {'OK' if f1_ok else 'MISMATCH'}")
    log(f"  {'th_base':>6} {params.get('th_base', '?'):>18} {'':>18}  "
        f"(the gate product's own value -- the script default 0.22 is NOT this task's contract)")
    log(f"  {'dedup':>6} {params.get('dedup_radius', '?'):>18} {'':>18}")
    if failures:
        print(f"[35][GATE FAIL] the frozen baseline was NOT reproduced: {failures}\n"
              f"       downstream arms are not comparable -- fix the params / cache / GT first.",
              file=sys.stderr, flush=True)
        return 1
    print("[35][GATE PASS] the frozen T3 baseline is reproduced bit for bit", flush=True)
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RK3588 board-side f16 mosaic inference vs the PC f32 multi-scale baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arm", choices=ARMS, help="which arm to run")
    parser.add_argument("--self-check", action="store_true",
                        help="reproduce the frozen T3 baseline and exit")
    parser.add_argument("--sequence", default=SEQUENCE)
    parser.add_argument("--cache-root", default=DEF_CACHE_ROOT,
                        help="the PC ms_cache directory (read-only; never rebuilt)")
    parser.add_argument("--labels-root", default=DEF_LABELS_ROOT)
    parser.add_argument("--gate-json", default=DEF_GATE_JSON,
                        help="the gate product json whose config.params defines the scoring contract")
    parser.add_argument("--params-json", default="",
                        help="an already extracted params json (overrides --gate-json)")
    parser.add_argument("--out", default=DEF_OUT, help="the output directory of this run")
    parser.add_argument("--start", type=int, default=None,
                        help="first frame of the slice -- the PC side is subsetted to the SAME slice")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N frames -- the PC side is subsetted to the SAME frames")
    parser.add_argument("--board-host", default=DEF_BOARD_HOST)
    parser.add_argument("--board-port", type=int, default=DEF_BOARD_PORT)
    parser.add_argument("--board-bin", default=DEF_BOARD_BIN_LOCAL,
                        help="the board binary to push (produced by the C++ worker)")
    parser.add_argument("--board-bin-remote", default=DEF_BOARD_BIN_REMOTE)
    parser.add_argument("--board-model", default=DEF_BOARD_MODEL,
                        help="the model path AS THE BOARD SEES IT")
    parser.add_argument("--board-nfs", default=DEF_BOARD_NFS,
                        help="the board's NFS mount point (the PC sees the same tree at a host path)")
    parser.add_argument("--board-features", default="",
                        help="the board-side feature directory (default <board-nfs>/bmp/<sequence>)")
    parser.add_argument("--board-out", default=DEF_BOARD_OUT,
                        help="the board-side output root (NFS, so the artefacts survive the run)")
    parser.add_argument("--board-cpu", default="6",
                        help="CPU list for the board run (the program's own "
                             "--cpu-affinity when supported, else taskset; "
                             "empty disables the pinning)")
    parser.add_argument("--board-input-mode", default="",
                        help="the board program's --input-mode (set / io-mem-nchw / "
                             "io-mem-nhwc); empty = let the program use its own default")
    parser.add_argument("--board-timeout", type=int, default=7200,
                        help="hard cap on one board run, in seconds")
    parser.add_argument("--canvas-mode", default="", choices=["", "mosaic", "single"],
                        help="override the arm's canvas mode")
    parser.add_argument("--tiles", default=None,
                        help="override the arm's tile list (e.g. 0 or 0,1,2)")
    args = parser.parse_args(argv)
    if not args.self_check and not args.arm:
        parser.error("either --arm or --self-check is required")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive frame count")
    if args.start is not None and args.start < 0:
        parser.error("--start must not be negative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    work = out_root / "_work"
    work.mkdir(parents=True, exist_ok=True)
    args.work = str(work)

    labels_root = Path(args.labels_root)
    if not labels_root.is_dir():
        die(f"--labels-root does not exist: {labels_root}")
    params_json, gate_json = resolve_params_json(args)
    params = load_params(params_json)

    if args.self_check:
        return run_self_check(args, params_json, labels_root)

    arm = args.arm
    out_dir = out_root / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    # the PC cache is read-only and NEVER rebuilt
    pc_cache = load_cache(Path(args.cache_root), args.sequence)
    pc_frames_all = [int(value) for value in pc_cache["frame_index"]]
    # --start selects a slice; --limit caps it.  The board manifest is built from
    # the SAME slice, so both sides always score the same frames.
    start = int(args.start or 0)
    pc_frames = pc_frames_all[start:]
    if args.limit is not None:
        pc_frames = pc_frames[:args.limit]
    pc_frame_set = set(pc_frames)
    if not pc_frames:
        die(f"--start {start} / --limit {args.limit} selects no frame")
    log(f"PC cache: {args.cache_root}/{args.sequence}.pkl | frames={len(pc_frames)} "
        f"| tags={per_tag_counts(pc_cache)}")

    # the labels the arms are scored against: the scorer unions the cache frames
    # with the ground-truth frames, so a slice that is not the whole sequence needs
    # a matching GT root -- otherwise the frames OUTSIDE the slice come back as
    # false negatives and the two sides stop being comparable.
    if args.limit is None and not start:
        arm_labels = labels_root
    else:
        arm_labels = build_gt_subset(labels_root, args.sequence, pc_frames,
                                     out_dir / "gt_subset")

    if arm in PC_ARMS:
        keep_tags = {0} if arm == "pc-native" else None
        cache = subset_cache(pc_cache, keep_frames=pc_frame_set, keep_tags=keep_tags)
        result = run_pc_arm(args, arm, cache, params_json, arm_labels, out_dir)
    else:
        board = Board(args.board_host, args.board_port, timeout=args.board_timeout)
        result = run_board_arm(args, arm, board, pc_cache, pc_frames, pc_frames_all,
                               params_json, arm_labels, out_dir)

    result["params"] = params
    result["params_json"] = str(params_json)
    result["gate_json"] = gate_json
    result["config"] = {
        "cache_root": str(args.cache_root),
        "labels_root": str(arm_labels),
        "sequence": args.sequence,
        "no_pipeline": True,
        "scorer": str(SCORER),
        "img_w": NATIVE_SIZE[0],
        "img_h": NATIVE_SIZE[1],
        "match_mode": "inbox",
        "tol": 8.0,
        "scale_tags": list(SCALE_TAGS),
    }
    result["frozen_baseline"] = FROZEN_T3
    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result["host"] = os.uname().nodename
    result["python"] = sys.version.split()[0]
    result["numpy"] = np.__version__
    write_json(out_dir / "result.json", result)
    log(f"arm {arm}: F1={result['metrics']['f1']:.6f} TP={result['metrics']['tp']} "
        f"FP={result['metrics']['fp']} FN={result['metrics']['fn']} GT={result['metrics']['gt']}")
    log(f"result -> {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
