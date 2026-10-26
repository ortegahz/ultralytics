#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
14_run_full_scale_nofs.py -- server-side, fully autonomous, resumable, chunk-verified
full-scale board evaluation driver for the RK3588 PTQ sweep.

Why this script exists
----------------------
The previous board driver (04_run_board_fast_eval.py) reads the frames over NFS
from the workstation.  The workstation is switched off at 18:30, the NFS mount
dies with it and the board can no longer read /mnt/manu/ptq/bmp/... .  The
board's own root filesystem only has 1.6 GB free, so the full-scale set
(28,997 official-val frames + 31,613 Anti-UAV frames, ~45 GB as BMP) cannot be
staged on the board either.

This driver therefore runs ON THE SERVER (which owns all the data and never goes
away) and pushes the frames to the board over SSH in small chunks:

    server BMP --(cv2 -> PNG)--> server tmp --(tar | ssh)--> board /userdata
    board runs the NPU --> out.tsv --(scp)--> server --> 04.score_rows()

The frames are staged as PNG because PNG is lossless -- cv2.imread(png) is
byte-identical to cv2.imread(bmp) -- and roughly 3x smaller than BMP, which is
what keeps the board's /userdata peak under 1 GB.  The board's stbi_load()
decodes the PNG to exactly the same RGB bytes it would have decoded from the
BMP, so the "quantisation loss" is not contaminated by a format change.  Every
chunk asserts that equality on a sample of frames before anything is pushed.

Autonomy (the whole point of this script)
-----------------------------------------
The pipeline must not depend on the survival of the session that started it:

  * SIGHUP is ignored, so `setsid nohup ... &` keeps the run alive.
  * --out/.lock is flock()ed, so two instances can never drive the board at the
    same time (the board only tolerates 2 RKNN contexts; a third one soft-resets
    the NPU).
  * --out/progress.json is rewritten atomically (tmp file + os.replace) after
    every chunk.  On the next start a chunk whose record says OK, whose model
    md5 still matches, and whose TSV still passes the row/name check is skipped
    => --resume.
  * A failing chunk is recorded in status.json["errors"] and does NOT abort the
    run; the process exits non-zero at the end so the outer scheduler sees it.
  * The board's staging directory is emptied chunk by chunk, so the peak
    /userdata usage stays at one chunk (<< 1 GB).

Usage
-----
    setsid nohup python 14_run_full_scale_nofs.py \
        --tracks blind golden anti \
        --model /mnt/data/siping/ptq_work/screen/p0_off_A_none.rknn \
        --frames-root /mnt/data/siping/ptq_work/bmp \
        --frames-root anti=/mnt/data/siping/ptq_work/pcfeat_bmp/images \
        --stage-dir /userdata/ptq_stage \
        --chunk 500 --th 0.3 --topk 100 \
        --out /mnt/data/siping/ptq_work/full_scale \
        --skip-pt > /mnt/data/siping/ptq_work/logs/full_scale.log 2>&1 &

    # dry run (plan only, nothing is pushed)
    python 14_run_full_scale_nofs.py --tracks blind --model <m.rknn> \
        --out <dir> --chunk 40 --limit-frames 40 --dry-run

Exit codes: 0 = everything scored, 1 = at least one chunk failed,
2 = another instance holds the lock, 3 = fatal setup error.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# The repo root is parents[4] (ptq -> npu -> pipeline -> manu -> <root>).
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent

# The board is ALWAYS reached through the server's key-based ssh alias.  No board
# IP and no board password ever appear in this file.
BOARD_HOST = "ptq-board"
BOARD_BIN = "/root/rknn_heatmap_smoke"

# The three tracks of the gate.  blind/golden live in per-sequence sub
# directories, anti is one flat directory (no sequence grouping, see
# ptq_common.TRACK1_SEQS = None).
TRACKS = ("blind", "golden", "anti")
TRACK_SEQS = {
    "blind": list(C.TRACK3_BLIND),
    "golden": list(C.TRACK2_GOLDEN),
    "anti": None,
}

# Default frame roots on the server.
DEFAULT_FRAMES_ROOT = {
    "blind": C.BMP_ROOT,
    "golden": C.BMP_ROOT,
    "anti": C.WORK_ROOT / "pcfeat_bmp" / "images",
}

# Default GT label roots on the server.  blind/golden come from the 龙泉山 label
# tree; the anti track carries its labels in a different tree (same naming and
# same content format), so it gets its own root -- pointing it at C.GT_ROOT
# makes every anti detection a FP (GT files 0/31613).
DEFAULT_GT_ROOT = {
    "blind": C.GT_ROOT,
    "golden": C.GT_ROOT,
    "anti": C.ANTI_GT_ROOT,
}

STOP = False          # set by SIGINT/SIGTERM -> stop after the current chunk


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def sh(cmd: list[str], timeout: int = 3600) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def board_sh(cmd: str, timeout: int = 3600) -> tuple[int, str, str]:
    """Run one shell command string on the board through the ssh alias."""
    return sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_HOST, cmd], timeout=timeout)


def write_json_atomic(path: Path, obj) -> Path:
    """Write JSON via a temp file + rename, so a crash can never leave a
    half-written progress/status file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def load_script(name: str):
    """Load a sibling script whose filename starts with a digit (not importable)."""
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sig_stop(signum, frame):        # noqa: ARG001
    global STOP
    STOP = True
    print(f"\n[signal] {signal.Signals(signum).name} received -- finishing the "
          f"current chunk and exiting", flush=True)


# --------------------------------------------------------------------------------------
# the lock -- one driver per board
# --------------------------------------------------------------------------------------
def acquire_lock(lock_path: Path):
    """flock() the output dir so two drivers never fight over the board.

    flock() is released by the kernel when the holder dies, so a crashed run
    never leaves a lock that blocks the next --resume.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.seek(0)
        holder = fh.read().strip() or "<unknown>"
        fh.close()
        print(f"[FATAL] another instance already holds {lock_path} "
              f"(holder: {holder}) -- refusing to start a second driver "
              f"(the board only tolerates 2 RKNN contexts).", flush=True)
        raise SystemExit(2)
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} host={os.uname().nodename} started={now_iso()}\n")
    fh.flush()
    return fh


# --------------------------------------------------------------------------------------
# frame discovery
# --------------------------------------------------------------------------------------
def collect_rows(track: str, root: Path, limit: int | None,
                 gt_root: Path | None = None) -> list[dict]:
    """All frames of one track, ordered deterministically by (seq, frame_id).

    The order is what makes the chunking reproducible, and a reproducible
    chunking is what makes --resume meaningful.

    `gt_root` is the label root of THIS track (see DEFAULT_GT_ROOT); it is
    None for the tracks that use the global C.GT_ROOT.
    """
    rows: list[dict] = []
    if not root.is_dir():
        raise SystemExit(f"[FATAL] frames root does not exist: {root}")

    if track == "anti":
        # one flat directory: <seq>__<id>.bmp
        for p in sorted(root.glob("*.bmp")):
            seq, fid = C.split_seq_frame(p.name)
            if fid < 0:
                continue
            rows.append({"seq": seq, "frame_id": fid, "bmp": str(p),
                         "gt": str(C.gt_path(seq, fid, root=gt_root))})
    else:
        for seq in TRACK_SEQS[track]:
            d = root / seq
            if not d.is_dir():
                print(f"[warn] track {track}: no such sequence dir {d}", flush=True)
                continue
            for p in sorted(d.glob("*.bmp")):
                s2, fid = C.split_seq_frame(p.name)
                if fid < 0 or s2 != seq:
                    continue
                rows.append({"seq": seq, "frame_id": fid, "bmp": str(p),
                             "gt": str(C.gt_path(seq, fid, root=gt_root))})

    rows.sort(key=lambda r: (r["seq"], r["frame_id"]))
    if limit is not None and limit > 0:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"[FATAL] track {track}: no frames found under {root}")
    return rows


def chunk_rows(rows: list[dict], chunk: int, track: str) -> list[dict]:
    """Split the track into fixed-size chunks with stable ids."""
    out = []
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        out.append({"chunk_id": f"{track}_{i // chunk:05d}", "index": i // chunk,
                    "rows": part})
    return out


def src_hw_of(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """Source (h, w) per sequence, read from the first frame of each sequence."""
    import cv2

    hw: dict[str, tuple[int, int]] = {}
    for r in rows:
        if r["seq"] in hw:
            continue
        img = cv2.imread(r["bmp"], cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"[FATAL] cannot read frame: {r['bmp']}")
        hw[r["seq"]] = (img.shape[0], img.shape[1])
    return hw


# --------------------------------------------------------------------------------------
# the board
# --------------------------------------------------------------------------------------
def board_model_path(board_model_dir: str, model: Path) -> str:
    return f"{board_model_dir}/{model.name}"


def stage_model(model: Path, board_model_dir: str) -> tuple[str, str]:
    """Copy the model onto the board and prove the bytes are identical.

    The md5 is checked BEFORE (what the board would run) and AFTER (what the
    board actually got).  A mismatch is fatal: the numbers would be attributed
    to a different model than the report quotes.
    """
    server_md5 = C.md5_file(model)
    bpath = board_model_path(board_model_dir, model)
    rc, out, err = board_sh(f"md5sum {bpath}")
    board_md5 = out.split()[0] if (rc == 0 and out.split()) else ""
    if board_md5 == server_md5:
        print(f"[model] board copy already matches: {board_md5} ({bpath})", flush=True)
        return server_md5, board_md5

    print(f"[model] pushing {model} -> {bpath} "
          f"(server md5 {server_md5}, board md5 {board_md5 or 'none'})", flush=True)
    board_sh(f"mkdir -p {board_model_dir}")
    tmp_remote = f"{bpath}.part"
    # stream the file into a .part name and rename it: an interrupted push can
    # never leave a half-written model in place.
    p = subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_HOST,
                          f"cat > {tmp_remote}"], stdin=subprocess.PIPE)
    try:
        with open(model, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                p.stdin.write(b)
    finally:
        p.stdin.close()
    if p.wait() != 0:
        board_sh(f"rm -f {tmp_remote}")
        raise SystemExit("[FATAL] cannot push the model to the board")

    rc, out, err = board_sh(f"md5sum {tmp_remote}")
    part_md5 = out.split()[0] if (rc == 0 and out.split()) else ""
    if part_md5 != server_md5:
        board_sh(f"rm -f {tmp_remote}")
        raise SystemExit(f"[FATAL] model md5 MISMATCH after the push\n"
                         f"       server: {server_md5}\n"
                         f"       board : {part_md5}\n"
                         f"       {err[:300]}")
    board_sh(f"mv {tmp_remote} {bpath}")

    rc, out, err = board_sh(f"md5sum {bpath}")
    board_md5 = out.split()[0] if (rc == 0 and out.split()) else ""
    if board_md5 != server_md5:
        raise SystemExit(f"[FATAL] model md5 MISMATCH after the move\n"
                         f"       server: {server_md5}\n"
                         f"       board : {board_md5}")
    print(f"[model] board model md5 verified: {board_md5}", flush=True)
    return server_md5, board_md5


def board_clean_chunk(stage_dir: str, chunk_id: str) -> None:
    """Empty one chunk's staging dir on the board.

    `find <dir> -mindepth 1 -delete` is used on purpose instead of `rm -rf`:
    it is scoped to exactly the directory this driver created and can never
    walk outside of it.
    """
    d = f"{stage_dir}/{chunk_id}"
    board_sh(f"if [ -d {d} ]; then find {d} -mindepth 1 -delete; rmdir {d} 2>/dev/null; fi")


def board_push_chunk(img_dir: Path, stage_dir: str, chunk_id: str,
                     timeout: int = 3600) -> tuple[int, str, str]:
    """tar over ssh: server -> board, exactly one chunk of frames."""
    board_sh(f"mkdir -p {stage_dir}/{chunk_id}")
    tar_p = subprocess.Popen(["tar", "-C", str(img_dir), "-cf", "-", "."],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ssh_p = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", BOARD_HOST,
         f"tar -C {stage_dir}/{chunk_id} -xf -"],
        stdin=tar_p.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tar_p.stdout.close()
    ssh_out, ssh_err = ssh_p.communicate(timeout=timeout)
    tar_err = tar_p.stderr.read().decode("utf-8", "replace")
    tar_p.stderr.close()
    tar_rc = tar_p.wait()
    if tar_rc != 0 or ssh_p.returncode != 0:
        return 1, ssh_out.decode("utf-8", "replace"), \
            f"tar_rc={tar_rc} ssh_rc={ssh_p.returncode} tar_err={tar_err[:300]} " \
            f"ssh_err={ssh_err.decode('utf-8', 'replace')[:300]}"
    return 0, ssh_out.decode("utf-8", "replace"), ssh_err.decode("utf-8", "replace")


def board_put_file(local: Path, remote: str) -> tuple[int, str]:
    with open(local, "rb") as f:
        p = subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_HOST,
                              f"cat > {remote}"], stdin=f)
        p.wait()
    if p.returncode != 0:
        return p.returncode, f"cannot write {remote} on the board"
    return 0, ""


def board_pull_file(remote: str, local: Path) -> tuple[int, str]:
    local.parent.mkdir(parents=True, exist_ok=True)
    rc, out, err = sh(["scp", "-o", "StrictHostKeyChecking=no",
                       f"{BOARD_HOST}:{remote}", str(local)])
    return rc, err


def board_demo_processes() -> str:
    """`ps` of the board -- a leftover rknn_heatmap_smoke pollutes the next run
    (the board only tolerates 2 RKNN contexts), so this is checked and printed."""
    rc, out, err = board_sh("ps -eo args | grep rknn_heatmap_smoke | grep -v grep")
    return out.strip()


def board_demo_count(bin_path: str = BOARD_BIN) -> int:
    """How many RKNN contexts are really in use on the board right now.

    Only the demo processes themselves count: the `sh -c` wrapper that ssh
    spawns also contains the binary's name in its command line, and counting it
    would report 2 contexts where there is only 1.
    """
    rc, out, err = board_sh(
        f"ps -eo args= | awk -v b='{bin_path}' '$1==b' | wc -l")
    try:
        return int(out.strip().splitlines()[0])
    except (IndexError, ValueError):
        return 0


def board_own_processes(stage_dir: str) -> str:
    """The board processes that belong to THIS driver's staging area.

    A leftover here would keep an RKNN context alive and pollute the next
    measurement, so it must be empty once a chunk is done.
    """
    rc, out, err = board_sh(
        f"ps -eo pid,args | grep -F '{stage_dir}' | grep -v grep")
    return out.strip()


def wait_for_board_slot(max_ctx: int, timeout: int, bin_path: str = BOARD_BIN) -> int:
    """Wait until the board has a free RKNN context.

    The board tolerates at most 2 concurrent RKNN contexts; a third one soft-
    resets the NPU and corrupts the measurement.  Another driver (e.g. the old
    04_run_board_fast_eval.py, which holds a context for the whole 2,000-frame
    batch) may be using one, so wait for a slot instead of racing it.
    """
    t0 = time.time()
    while True:
        n = board_demo_count(bin_path)
        if n < max_ctx:
            return n
        waited = time.time() - t0
        if waited > timeout:
            raise RuntimeError(f"board busy: {n} demo process(es) running "
                               f"(limit {max_ctx}) after {waited:.0f}s -- giving up "
                               f"on this chunk")
        print(f"  [board] {n} demo process(es) running, waiting for a free "
              f"context (limit {max_ctx}, waited {waited:.0f}s)", flush=True)
        time.sleep(10)


def board_userdata_usage() -> str:
    rc, out, err = board_sh("du -sh /userdata 2>/dev/null | cut -f1")
    return out.strip() or (err.strip() or "n/a")


# --------------------------------------------------------------------------------------
# one chunk
# --------------------------------------------------------------------------------------
def prepare_images(rows: list[dict], img_dir: Path, fmt: str) -> list[Path]:
    """BMP -> PNG (lossless) on the server, into the chunk's temp dir.

    PNG keeps /userdata small.  `cv2.imwrite` on a PNG is lossless, so
    cv2.imread(png) is byte-identical to cv2.imread(bmp) -- asserted below.
    """
    import cv2

    img_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for r in rows:
        src = Path(r["bmp"])
        if fmt == "png":
            dst = img_dir / (src.stem + ".png")
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"cannot read frame: {src}")
            if not cv2.imwrite(str(dst), img):
                raise RuntimeError(f"cannot write PNG: {dst}")
        else:
            dst = img_dir / src.name
            shutil.copyfile(src, dst)
        out.append(dst)
    return out


def verify_pixels(rows: list[dict], img_dir: Path, fmt: str, sample: int) -> list[dict]:
    """Assert PNG == BMP pixel-by-pixel on a sample of the chunk.

    This is the assertion that keeps the format change honest: the board's
    stbi_load() must end up with exactly the bytes it would have gotten from
    the BMP, otherwise the "quantisation loss" would also contain a decoder
    difference.
    """
    import cv2

    n = len(rows)
    if n == 0 or sample <= 0:
        return []
    k = min(sample, n)
    idxs = sorted({int(round(v)) for v in np.linspace(0, n - 1, k)})
    res = []
    for i in idxs:
        r = rows[i]
        src = Path(r["bmp"])
        png = img_dir / (src.stem + (".png" if fmt == "png" else ".bmp"))
        a = cv2.imread(str(src), cv2.IMREAD_COLOR)
        b = cv2.imread(str(png), cv2.IMREAD_COLOR)
        ok = a is not None and b is not None and np.array_equal(a, b)
        res.append({"frame": src.name, "equal": bool(ok),
                    "shape": list(a.shape) if a is not None else None})
        print(f"  [pixel] {src.name}: np.array_equal(bmp, {fmt}) = {ok} "
              f"shape={res[-1]['shape']}", flush=True)
        if not ok:
            raise RuntimeError(f"PIXEL MISMATCH: {src.name} -- the staged {fmt} is "
                               f"not pixel-identical to the BMP, aborting the chunk")
    return res


def tsv_names(path: Path) -> list[str]:
    """First column of every non-empty line of a board TSV."""
    names = []
    if not path.is_file():
        return names
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        names.append(line.split("\t", 1)[0])
    return names


def run_chunk(track: str, chunk: dict, args, model_md5: str, board_model: str,
              out_root: Path) -> dict:
    """Execute one chunk end to end.  Raises on any failed assertion."""
    rows = chunk["rows"]
    chunk_id = chunk["chunk_id"]
    n = len(rows)
    tmp_dir = Path(args.tmp_dir) / chunk_id
    img_dir = tmp_dir / "img"

    # A previous run may have died in the middle of this chunk -> start clean.
    if tmp_dir.exists():
        # shutil.rmtree on a directory this script created itself (the temp
        # staging area), never on anything the user owns.
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    board_clean_chunk(args.stage_dir, chunk_id)

    t0 = time.time()
    print(f"[chunk {chunk_id}] {n} frames ...", flush=True)

    # ---- 1. stage the frames on the server as PNG -------------------------------
    prepare_images(rows, img_dir, args.fmt)
    nbytes = sum(p.stat().st_size for p in img_dir.iterdir())
    print(f"  staged {n} {args.fmt} on the server: {nbytes / 1e6:.1f} MB", flush=True)

    # ---- 2. pixel assertion (sample) ------------------------------------------
    px = verify_pixels(rows, img_dir, args.fmt, args.sample_pixels)

    # ---- 3. push the chunk to the board ---------------------------------------
    rc, out, err = board_push_chunk(img_dir, args.stage_dir, chunk_id,
                                    timeout=args.board_timeout)
    if rc != 0:
        raise RuntimeError(f"push failed: {err}")
    board_list = f"{args.stage_dir}/{chunk_id}/list.txt"
    board_tsv = f"{args.stage_dir}/{chunk_id}/out.tsv"

    # ---- 4. the board's manifest: one absolute board path per line ------------
    lines = []
    for r in rows:
        lines.append(f"{args.stage_dir}/{chunk_id}/{Path(r['bmp']).stem}.{args.fmt}")
    list_local = tmp_dir / "list.txt"
    list_local.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rc, err = board_put_file(list_local, board_list)
    if rc != 0:
        raise RuntimeError(f"cannot write the board manifest: {err}")

    # the manifest must be exactly what we think it is
    rc, out, err = board_sh(f"wc -l < {board_list}")
    try:
        got = int(out.strip().split()[0])
    except (IndexError, ValueError):
        got = -1
    if got != n:
        raise RuntimeError(f"board manifest has {got} lines, expected {n}")

    # ---- 5. board inference ----------------------------------------------------
    # never become the 3rd RKNN context on the board
    nctx = wait_for_board_slot(args.max_board_contexts, args.board_wait,
                               args.board_bin)
    if nctx:
        print(f"  [board] {nctx} other demo process(es) already running "
              f"(this driver uses 1 of {args.max_board_contexts} allowed)",
              flush=True)
    # CPU affinity: rknn_inputs_set's type conversion is a SINGLE-THREADED
    # scalar loop -- 45.41 ms on an A55 (cpu 0-3) vs 11.15 ms on an A76
    # (cpu 4-7), i.e. 4.1x for free. taskset keeps this working with the
    # CERTIFIED board binary. Measured 2026-10-23; the zero-copy alternative
    # was a net loss (the runtime moves the staging into rknn_run at
    # ~180 MB/s) -- see manu/memory/fp16_input_io_mem.md.
    pin = f"taskset -c {args.board_cpu} " if args.board_cpu else ""
    cmd = (f"{pin}{args.board_bin} {board_model} --batch {board_list} "
           f"--th {args.th} --topk {args.topk} --detections-out {board_tsv}")
    rc, out, err = board_sh(cmd, timeout=args.board_timeout)
    if rc != 0:
        raise RuntimeError(f"board inference failed rc={rc}: {err[:400]}")

    # ---- 6. pull the TSV back and verify it ------------------------------------
    tsv_local = out_root / "chunks" / track / f"{chunk_id}.tsv"
    rc, err = board_pull_file(board_tsv, tsv_local)
    if rc != 0:
        raise RuntimeError(f"cannot pull {board_tsv}: {err[:300]}")

    names = tsv_names(tsv_local)
    expect = [Path(r["bmp"]).stem + f".{args.fmt}" for r in rows]
    if len(names) != n:
        raise RuntimeError(f"TSV has {len(names)} lines, expected {n} "
                           f"(one line per frame)")
    got_set, want_set = set(names), set(expect)
    missing = sorted(want_set - got_set)
    extra = sorted(got_set - want_set)
    dupes = sorted({x for x in names if names.count(x) > 1}) if len(set(names)) != len(names) else []
    if missing or extra or dupes:
        raise RuntimeError(f"TSV frame-name mismatch: missing={missing[:10]} "
                           f"({len(missing)}) extra={extra[:10]} ({len(extra)}) "
                           f"duplicated={dupes[:10]} ({len(dupes)})")

    ndet = 0
    for line in tsv_local.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip().isdigit():
            ndet += int(parts[1])

    # ---- 7. clean up: the board keeps only one chunk at a time ------------------
    board_clean_chunk(args.stage_dir, chunk_id)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- 7b. this driver must not leave a board process of its own running ----
    # (it would hold an RKNN context and pollute the next chunk)
    left = board_own_processes(args.stage_dir)
    if left:
        print(f"  [warn] board still shows processes of THIS driver:\n{left}",
              flush=True)

    dt = time.time() - t0
    print(f"  [chunk {chunk_id}] OK  {n} frames, {ndet} detections, "
          f"{dt:.1f}s ({n / max(1e-9, dt):.2f} fps), board /userdata="
          f"{board_userdata_usage()}", flush=True)
    return {"track": track, "chunk_id": chunk_id, "frames": n, "tsv_path": str(tsv_local),
            "ts": now_iso(), "status": "OK", "md5_model": model_md5,
            "detections": ndet, "seconds": round(dt, 2), "pixel_ok": len(px),
            "board_left": left}


# --------------------------------------------------------------------------------------
# merging + scoring (the scoring口径 is REUSED from 04, never copied)
# --------------------------------------------------------------------------------------
def merge_track(track: str, rows: list[dict], records: list[dict], args,
                model_md5: str, out_root: Path, mod,
                gt_root: Path | None = None) -> dict:
    """Merge the chunk TSVs into <track>_board.tsv and score them."""
    by_chunk = {r["chunk_id"]: r for r in records if r["track"] == track
                and r.get("status") == "OK"}
    name_of = {Path(r["bmp"]).stem + f".{args.fmt}": Path(r["bmp"]).name for r in rows}
    # sort key: (seq, frame_id).  Sorting by the bare file name would interleave
    # sequences whose names have different lengths (the Anti-UAV sequences are
    # "01_4485_1167-2666" and "25_1", and '_' > '2' in ASCII).
    sort_key = {Path(r["bmp"]).name: (r["seq"], r["frame_id"]) for r in rows}

    merged: list[tuple[tuple, str]] = []
    for rec in records:
        if rec["track"] != track or rec.get("status") != "OK":
            continue
        tsv = Path(rec["tsv_path"])
        if not tsv.is_file():
            continue
        for line in tsv.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            head, _, tail = line.partition("\t")
            bmp_name = name_of.get(head, head)
            merged.append((sort_key.get(bmp_name, (bmp_name, 0)), f"{bmp_name}\t{tail}"))
    merged.sort(key=lambda x: x[0])
    out_tsv = out_root / f"{track}_board.tsv"
    out_tsv.write_text("".join(line + "\n" for _, line in merged), encoding="utf-8")
    print(f"[merge {track}] {len(merged)} frames -> {out_tsv}", flush=True)

    det_by_name = mod.parse_board_tsv(out_tsv)
    hw = src_hw_of(rows)
    score = mod.score_rows(det_by_name, rows, hw)
    n_missing_gt = sum(1 for r in rows if not Path(r["gt"]).is_file())
    failed = [r["chunk_id"] for r in records if r["track"] == track
              and r.get("status") != "OK"]
    report = {
        "track": track,
        "model": str(args.model),
        "model_md5": model_md5,
        "board_model": board_model_path(args.board_model_dir, Path(args.model)),
        "gt_root": str(gt_root) if gt_root else str(C.GT_ROOT),
        "th": args.th,
        "topk": args.topk,
        "fmt": args.fmt,
        "frames_expected": len(rows),
        "frames_scored": len(merged),
        "chunks_ok": len(by_chunk),
        "chunks_failed": failed,
        "status": "PARTIAL" if failed else "OK",
        "n_missing_gt": n_missing_gt,
        "score": score,
    }
    if score["det"] == 0:
        report["status"] = "ZERO_DETECTIONS"
    write_json_atomic(out_root / f"{track}_report.json", report)
    print(f"[report {track}] F1={score['f1']:.4f} R={score['recall']:.4f} "
          f"P={score['precision']:.4f} TP={score['tp']} FP={score['fp']} "
          f"FN={score['fn']} det={score['det']} "
          f"(status={report['status']}, missing GT files={n_missing_gt}, "
          f"gt_root={report['gt_root']})", flush=True)
    return report


def run_pt_arm(track: str, rows: list[dict], args, model_md5: str, out_root: Path,
               mod) -> dict:
    """The PC PyTorch reference on exactly the same frames (same bytes, same
    decoder, same matcher) so the quantisation loss = PT - INT8 is meaningful."""
    print(f"[PT {track}] running the PyTorch reference on {len(rows)} frames ...",
          flush=True)
    t0 = time.time()
    hw = src_hw_of(rows)
    det_by_name = mod.run_pt_reference(rows, hw, args.pt_variant, topk=args.topk,
                                      device=args.pt_device)
    score = mod.score_rows(det_by_name, rows, hw)
    score["seconds"] = round(time.time() - t0, 1)
    report = {"track": track, "arm": "pt_float", "variant": args.pt_variant,
              "weights": str(C.WEIGHTS), "frames": len(rows),
              "th": args.th, "topk": args.topk, "score": score}
    write_json_atomic(out_root / f"{track}_pt_report.json", report)
    print(f"[PT {track}] F1={score['f1']:.4f} R={score['recall']:.4f} "
          f"P={score['precision']:.4f} ({score['seconds']}s)", flush=True)
    return report


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Server-side autonomous full-scale board evaluation driver.")
    ap.add_argument("--tracks", nargs="+", default=list(TRACKS), choices=list(TRACKS),
                    help="which tracks to run")
    ap.add_argument("--model", required=True, help="the RKNN model on the SERVER")
    ap.add_argument("--frames-root", action="append", default=[], metavar="[TRACK=]PATH",
                    help="server-side frame root; repeatable, optionally per track "
                         "(e.g. --frames-root anti=/mnt/.../pcfeat_bmp/images)")
    ap.add_argument("--gt-root", action="append", default=[], metavar="[TRACK=]PATH",
                    help="server-side GT label root; repeatable, optionally per track "
                         "(e.g. --gt-root anti=/mnt/.../uav_gmc_median/labels/val). "
                         "Defaults per track: blind/golden -> the 龙泉山 label tree, "
                         "anti -> the Anti-UAV label tree.")
    ap.add_argument("--stage-dir", default="/userdata/ptq_stage",
                    help="the BOARD's staging directory")
    ap.add_argument("--chunk", type=int, default=500, help="frames per chunk")
    ap.add_argument("--th", type=float, default=C.TH_BASE)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--out", required=True, help="server-side output directory")
    ap.add_argument("--tmp-dir", default=None,
                    help="server-side temp dir for the staged frames "
                         "(default <out>/_tmp, emptied chunk by chunk)")
    ap.add_argument("--fmt", choices=("png", "bmp"), default="png",
                    help="the format staged on the board (png = lossless and ~3x "
                         "smaller, bmp = the raw files)")
    ap.add_argument("--sample-pixels", type=int, default=5,
                    help="frames per chunk on which the PNG/BMP pixel equality is asserted")
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="only the first N frames of every track (smoke tests)")
    ap.add_argument("--board-bin", default=BOARD_BIN)
    ap.add_argument("--board-cpu", default="6",
                    help="CPU list the board run is pinned to (default 6 = an A76 "
                         "big core; empty string disables the pinning)")
    ap.add_argument("--board-model-dir", default="/userdata/ptq_models")
    ap.add_argument("--board-timeout", type=int, default=7200)
    ap.add_argument("--max-board-contexts", type=int, default=2,
                    help="the board tolerates at most 2 concurrent RKNN contexts; "
                         "a 3rd one soft-resets the NPU")
    ap.add_argument("--board-wait", type=int, default=1800,
                    help="seconds to wait for a free RKNN context on the board")
    ap.add_argument("--pt", dest="pt", action="store_true", default=False,
                    help="also run the PC PyTorch reference arm")
    ap.add_argument("--skip-pt", dest="pt", action="store_false")
    ap.add_argument("--pt-variant", default="p0_off", choices=list(C.VARIANTS))
    ap.add_argument("--pt-device", default="cuda:0")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--fresh", dest="resume", action="store_false",
                    help="ignore progress.json and redo every chunk")
    ap.add_argument("--dry-run", action="store_true", help="plan only, push nothing")
    return ap.parse_args()


def resolve_frames_roots(args) -> dict[str, Path]:
    roots = {t: Path(DEFAULT_FRAMES_ROOT[t]) for t in TRACKS}
    for spec in args.frames_root:
        if "=" in spec:
            track, _, path = spec.partition("=")
            track = track.strip()
            if track not in roots:
                raise SystemExit(f"[FATAL] unknown track in --frames-root: {track}")
            roots[track] = Path(path.strip())
        else:
            for t in roots:
                roots[t] = Path(spec)
    return roots


def resolve_gt_roots(args) -> dict[str, Path]:
    """The GT label root of every track.

    Same parsing style as --frames-root: "TRACK=PATH" sets one track, a bare
    "PATH" sets every track.  The per-track defaults are DEFAULT_GT_ROOT.
    """
    roots = {t: Path(DEFAULT_GT_ROOT[t]) for t in TRACKS}
    for spec in args.gt_root:
        if "=" in spec:
            track, _, path = spec.partition("=")
            track = track.strip()
            if track not in roots:
                raise SystemExit(f"[FATAL] unknown track in --gt-root: {track}")
            roots[track] = Path(path.strip())
        else:
            for t in roots:
                roots[t] = Path(spec)
    return roots


def main() -> None:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)     # survive the session's death
    signal.signal(signal.SIGINT, _sig_stop)
    signal.signal(signal.SIGTERM, _sig_stop)

    args = parse_args()
    if args.chunk <= 0:
        raise SystemExit("[FATAL] --chunk must be > 0")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.tmp_dir is None:
        args.tmp_dir = str(out_root / "_tmp")
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)

    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f"[FATAL] model does not exist on the server: {model}")
    model_md5 = C.md5_file(model)
    print(f"[model] {model} md5={model_md5}", flush=True)

    roots = resolve_frames_roots(args)
    gt_roots = resolve_gt_roots(args)
    plans = {}
    for t in args.tracks:
        rows = collect_rows(t, roots[t], args.limit_frames or None, gt_roots[t])
        plans[t] = {"rows": rows, "chunks": chunk_rows(rows, args.chunk, t)}
        nseq = len({r["seq"] for r in rows})
        ngt = sum(1 for r in rows if Path(r["gt"]).is_file())
        print(f"[plan] track={t} root={roots[t]} seqs={nseq} frames={len(rows)} "
              f"chunks={len(plans[t]['chunks'])} (chunk={args.chunk}) "
              f"gt_root={gt_roots[t]} gt_files={ngt}/{len(rows)}", flush=True)

    if args.dry_run:
        total = 0
        for t, p in plans.items():
            print(f"\n=== dry-run plan: track {t} ===")
            for c in p["chunks"]:
                first = Path(c["rows"][0]["bmp"]).name
                last = Path(c["rows"][-1]["bmp"]).name
                est = sum(os.path.getsize(r["bmp"]) for r in c["rows"]) * (
                    0.35 if args.fmt == "png" else 1.0)
                print(f"  chunk {c['chunk_id']}: {len(c['rows'])} frames  "
                      f"{first} .. {last}  ~{est / 1e6:.1f} MB  "
                      f"-> board {args.stage_dir}/{c['chunk_id']}/")
                total += len(c["rows"])
        print(f"\n[dry-run] total {total} frames, nothing was pushed.",
              flush=True)
        return

    lock_fh = acquire_lock(out_root / ".lock")

    mod = load_script("04_run_board_fast_eval.py")     # score_rows / parse_board_tsv

    progress_path = out_root / "progress.json"
    status_path = out_root / "status.json"
    progress = read_json(progress_path, None) or {}
    if not args.resume:
        progress = {}
    records: list[dict] = list(progress.get("chunks", []))
    errors: list[dict] = []

    # the model must be on the board, byte-identical, before anything is scored
    server_md5, board_md5 = stage_model(model, args.board_model_dir)
    board_model = board_model_path(args.board_model_dir, model)

    # index the already-done chunks (for --resume)
    done = {(r["track"], r["chunk_id"]): r for r in records if r.get("status") == "OK"}

    total_chunks = sum(len(p["chunks"]) for p in plans.values())
    frames_total = sum(len(p["rows"]) for p in plans.values())
    status = {"last_update": now_iso(), "done_chunks": 0, "total_chunks": total_chunks,
              "frames_done": 0, "current_track": args.tracks[0], "errors": errors,
              "model": str(model), "model_md5": model_md5, "pid": os.getpid(),
              "frames_total": frames_total}

    def flush_status(**kw):
        status.update(kw)
        status["last_update"] = now_iso()
        write_json_atomic(status_path, status)

    def flush_progress():
        write_json_atomic(progress_path, {
            "version": 1, "model": str(model), "model_md5": model_md5,
            "updated": now_iso(), "chunks": records})

    flush_status(done_chunks=len(done), frames_done=sum(
        r.get("frames", 0) for r in done.values()))
    flush_progress()

    for track in args.tracks:
        p = plans[track]
        flush_status(current_track=track)
        for chunk in p["chunks"]:
            if STOP:
                print("[stop] stopping before the next chunk", flush=True)
                break
            cid = chunk["chunk_id"]
            prev = done.get((track, cid))
            if args.resume and prev is not None:
                # a chunk is only reusable if it was scored with the SAME model
                # and its TSV still passes the row/name check.
                if prev.get("md5_model") != model_md5:
                    print(f"[chunk {cid}] re-run: the model md5 changed", flush=True)
                else:
                    tsv = Path(prev.get("tsv_path", ""))
                    names = tsv_names(tsv)
                    expect = [Path(r["bmp"]).stem + f".{args.fmt}" for r in chunk["rows"]]
                    if (tsv.is_file() and len(names) == len(expect)
                            and set(names) == set(expect)):
                        print(f"[chunk {cid}] SKIP (resume: {len(names)} frames "
                              f"already scored)", flush=True)
                        continue
                    print(f"[chunk {cid}] re-run: the stored TSV does not verify",
                          flush=True)

            try:
                rec = run_chunk(track, chunk, args, model_md5, board_model, out_root)
            except Exception as exc:                     # noqa: BLE001
                err = {"track": track, "chunk_id": cid, "error": f"{type(exc).__name__}: {exc}",
                       "ts": now_iso()}
                errors.append(err)
                print(f"[chunk {cid}] FAILED: {err['error']}", flush=True)
                # keep the board AND the server clean even after a failure
                board_clean_chunk(args.stage_dir, cid)
                shutil.rmtree(Path(args.tmp_dir) / cid, ignore_errors=True)
                rec = {"track": track, "chunk_id": cid, "frames": len(chunk["rows"]),
                       "tsv_path": "", "ts": now_iso(), "status": "FAILED",
                       "md5_model": model_md5, "error": err["error"]}
            records = [r for r in records
                       if not (r["track"] == track and r["chunk_id"] == cid)]
            records.append(rec)
            flush_progress()
            flush_status(done_chunks=sum(1 for r in records if r.get("status") == "OK"),
                         frames_done=sum(r.get("frames", 0) for r in records
                                         if r.get("status") == "OK"))

        # ---- score this track --------------------------------------------------
        ok_records = [r for r in records if r["track"] == track]
        if any(r.get("status") == "OK" for r in ok_records):
            merge_track(track, p["rows"], records, args, model_md5, out_root, mod,
                        gt_roots[track])
            if args.pt:
                try:
                    run_pt_arm(track, p["rows"], args, model_md5, out_root, mod)
                except Exception as exc:                 # noqa: BLE001
                    errors.append({"track": track, "chunk_id": f"{track}_pt",
                                   "error": f"{type(exc).__name__}: {exc}",
                                   "ts": now_iso()})
                    print(f"[PT {track}] FAILED: {exc}", flush=True)
        else:
            print(f"[{track}] no chunk succeeded -- no report written", flush=True)

        if STOP:
            break

    # ---- final state -------------------------------------------------------------
    left = board_demo_processes()
    own = board_own_processes(args.stage_dir)
    flush_status(done_chunks=sum(1 for r in records if r.get("status") == "OK"),
                 frames_done=sum(r.get("frames", 0) for r in records
                                 if r.get("status") == "OK"),
                 errors=errors, current_track="done",
                 board_demo_processes=left, board_own_processes=own,
                 board_userdata=board_userdata_usage())
    flush_progress()
    lock_fh.close()

    ok = sum(1 for r in records if r.get("status") == "OK")
    failed = sum(1 for r in records if r.get("status") != "OK")
    print(f"\n=== summary ===")
    print(f"  chunks ok      : {ok}")
    print(f"  chunks failed  : {failed}")
    print(f"  frames scored  : {sum(r.get('frames', 0) for r in records if r.get('status') == 'OK')}"
          f" / {frames_total}")
    print(f"  board /userdata : {board_userdata_usage()}")
    print(f"  board demo procs: {left if left else '<none>'}")
    if own:
        print(f"  [FATAL] this driver left board processes running:\n{own}",
              flush=True)
    print(f"  status         : {status_path}")
    print(f"  progress       : {progress_path}")
    if failed or errors or own:
        print(f"[FATAL] {failed} chunk(s) failed, {len(errors)} error(s) recorded "
              f"-- see {status_path}", flush=True)
        raise SystemExit(1)
    if STOP:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
