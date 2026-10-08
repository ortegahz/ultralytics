#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Unit tests for verify_cpp_port.py's pure helpers.

Why this file exists
--------------------
`verify_cpp_port.py` is the gate the whole C++ port rests on. A bug in it does not fail
loudly -- it produces a *confident wrong verdict*. The worst instance found while writing
this port was an inverted dictionary in `build_push_to_manifest`:

    list_to_frame = {li: fi for fi, li in idx_map.items()}   # keyed by LIST INDEX
    li = list_to_frame.get(fi)                              # ... looked up by FRAME NUMBER

That matched 0/10 frames on a synthetic tree, yet the gate would still have printed a clean
table -- comparing unrelated frames while reporting no drift. This is precisely the failure
mode the function's own docstring warns about, so the mapping gets its own tests, including
a regression guard that reproduces the inverted logic and asserts it does NOT match.

What is covered: the push-index <-> manifest-name mapping, the frozen-builder helpers it
reuses, and the C++ stdout regex the report depends on. What is NOT covered: any actual
feature comparison -- that is verify_cpp_port.py's G1/G2 job, on real data.

No dataset, no OpenCV, no GPU. Runs in under a second.

Usage:
    python manu/pipeline/cpp/test_verify_helpers.py
"""

from __future__ import annotations

import importlib.util
import re
import struct
import sys
import tempfile
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1] / "verify_cpp_port.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_cpp_port", VERIFIER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {VERIFIER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_tree(root: Path, frame_nums: list[int], seq: str, sep: str = "___",
               foreign: bool = True) -> tuple[Path, Path, str]:
    raw = root / "raw"
    frozen = root / "frozen"
    (raw / seq).mkdir(parents=True, exist_ok=True)
    (frozen / "images" / "val").mkdir(parents=True, exist_ok=True)
    for fn in frame_nums:
        (raw / seq / f"{fn:06d}.jpg").write_bytes(b"x")
    (raw / seq / "notes.txt").write_text("not an image")  # must be ignored by the filter
    for fn in frame_nums:
        (frozen / "images" / "val" / f"{seq}{sep}{fn}.jpg").write_bytes(b"y")
    if foreign:
        (frozen / "images" / "val" / f"OTHERSEQ{sep}3.jpg").write_bytes(b"z")
    return raw, frozen, seq


def expected_map(v, frame_nums: list[int], seq: str, sep: str = "___") -> dict[int, str]:
    """Ground truth: push order is natural_key over raw names; push i carries frame[i]'s name."""
    names = sorted([f"{fn:06d}.jpg" for fn in frame_nums], key=v.natural_key)
    return {i: f"{seq}{sep}{int(Path(n).stem)}.jpg" for i, n in enumerate(names)}


def main() -> int:
    v = load_verifier()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, bool(cond), detail))

    print("=" * 92)
    print("   verify_cpp_port.py HELPER TESTS  (push-index <-> manifest-name mapping)")
    print("=" * 92)
    print(f"  verifier : {VERIFIER}")
    print("=" * 92)

    noncontig = [3, 42, 7, 100, 12, 99, 5, 1000, 21, 8]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        raw, fro, seq = build_tree(root / "a", noncontig, "wg2022_ir_052_split_08")
        got = v.build_push_to_manifest(str(raw), str(fro), seq, 0)
        exp = expected_map(v, noncontig, seq)
        check("non-contiguous frame numbers", got == exp,
              f"{len(got)}/{len(exp)} mapped")

        raw, fro, seq = build_tree(root / "b", [10, 2, 7], "DJI_0051_2", sep="__")
        got2 = v.build_push_to_manifest(str(raw), str(fro), seq, 0)
        check("'__' manifest separator", got2 == expected_map(v, [10, 2, 7], seq, "__"))

        raw, fro, seq = build_tree(root / "c", list(range(1, 9)), "02_6321")
        got3 = v.build_push_to_manifest(str(raw), str(fro), seq, 3)
        check("--limit truncates to push indices", sorted(got3) == [0, 1, 2],
              f"keys={sorted(got3)}")

        raw, fro, seq = build_tree(root / "d", [3, 5, 7, 21, 99, 100], "wg011")
        got4 = v.build_push_to_manifest(str(raw), str(fro), seq, 0)
        check("keys are dense push indices, not frame numbers",
              sorted(got4) == list(range(len(got4))), f"keys={sorted(got4)}")
        check("foreign sequence never leaks in",
              all("OTHERSEQ" not in s for s in got4.values()))

        # The historical bug: look a FRAME NUMBER up in a dict keyed by LIST INDEX.
        rawfiles = sorted([p.name for p in (raw / seq).iterdir()
                           if p.suffix.lower() in v.IMAGE_SUFFIXES], key=v.natural_key)
        idx_map: dict[int, int] = {}
        for li, f in enumerate(rawfiles):
            m = re.search(r"(\d+)$", Path(f).stem)
            idx_map[int(m.group(1)) if m else li] = li
        inverted = {li: fi for fi, li in idx_map.items()}
        buggy = {}
        for f in rawfiles:
            fi = int(Path(f).stem)
            li = inverted.get(fi)
            if li is not None:
                buggy[li] = f"{seq}___{fi}.jpg"
        check("regression guard: inverted dict does NOT reproduce the correct map",
              buggy != got4, f"inverted matched {len(buggy)}/{len(got4)}")

        missing = v.build_push_to_manifest(str(raw), str(root / "does_not_exist"), seq, 0)
        check("missing frozen dir returns empty, no crash", missing == {})

    # ---- [MATS] collection: which frame, and which sequence -------------------------
    # Two alignment bugs were found here, both silent: the C++ binary writes _mats.npy at
    # push index 0 while the Python side read pipe.last_mats (the LAST frame), and a single
    # shared variable was compared against sequence[0] on multi-sequence runs.
    import argparse as _ap
    import numpy as np

    class FakePipe:
        def __init__(self, **kw):
            self.kw = kw
            self.seq = None
            self.last_mats = {}
            self.n = 0

        def run(self, src):
            self.seq = src.seq
            for i in range(src.n_frames):
                self.n += 1
                # Distinct value per (sequence, frame) so a frame mix-up is detectable.
                # n*100 + len(seq): frames step by 100, sequences offset by name length.
                self.last_mats = {2: np.full((2, 3), self.n * 100 + len(self.seq), np.float32)}
                yield np.full((1, 1, 1), self.n, np.uint8)

    class FakeSrc:
        def __init__(self, root, seq, limit=0):
            self.seq = seq
            self.n_frames = 4

    v.OnlineFeaturePipeline = FakePipe
    v.SequenceFrameSource = FakeSrc
    args = _ap.Namespace(window=21, stride_step=2, anchor_step=2, downscale=2,
                         raw_root="unused", limit=0)
    # Different name lengths so the two sequences carry distinguishable values.
    seqs = ["SEQ_A", "SEQ_BB"]
    _, _, mats_by_seq = v.python_side(args, seqs)
    check("[MATS] matrices collected per sequence", sorted(mats_by_seq) == sorted(seqs),
          f"keys={sorted(mats_by_seq)}")
    # Each sequence restarts its own pipeline, so frame 0 is n=1 for both.
    # A LAST-frame capture would give n=4 (last frame of 4).
    a0 = float(mats_by_seq["SEQ_A"][2][0, 0])
    b0 = float(mats_by_seq["SEQ_BB"][2][0, 0])
    check("[MATS] captures FRAME 0, not the last frame", a0 == 105.0 and b0 == 106.0,
          f"SEQ_A={a0} SEQ_BB={b0}; last-frame capture would give 405/406")
    check("[MATS] sequences do not overwrite each other", a0 != b0,
          f"distinct values {a0} vs {b0} (a shared variable would collapse to one)")

    # compare_mats must align rows with sorted lag keys, and report SKIP not PASS when absent.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        lags = [2, 4, 6]
        rows = np.arange(18, dtype=np.float32).reshape(3, 6)
        dict_str = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 6), }"
        while (10 + len(dict_str) + 1) % 64:
            dict_str += " "
        dict_str += "\n"
        with open(d / "S_000000_mats.npy", "wb") as f:
            f.write(bytes([0x93]) + b"NUMPY" + bytes([1, 0]))
            f.write(struct.pack("<H", len(dict_str)))
            f.write(dict_str.encode())
            f.write(rows.tobytes())
        py = {lag: rows[i].reshape(2, 3) for i, lag in enumerate(lags)}
        n_lags, mx = v.compare_mats(str(d), "S", py)
        check("[MATS] rows align with sorted lag keys", n_lags == 3 and mx == 0.0,
              f"lags={n_lags} max={mx}")
        n_none, mx_none = v.compare_mats(str(d), "MISSING_SEQ", py)
        check("[MATS] absent dump returns 0 lags (SKIP path), never a false PASS",
              n_none == 0, f"lags={n_none}")
        n_empty, _ = v.compare_mats(str(d), "S", {})
        check("[MATS] empty python mats returns 0 lags", n_empty == 0)

    # The C++ stdout contract the report parser depends on.
    line = ("[SEQ] wg2022_ir_052_split_08        frames=200   "
            "md5=0cc175b9c0f1b6a831c399e269772661  fits/frm=6.19 warps/frm=22.00 "
            "compose/frm=0.19  state=14.10 MiB")
    m = re.search(r"^\[SEQ\]\s+(\S+)\s+frames=(\d+)\s+md5=([0-9a-f]{32})", line, re.M)
    check("[SEQ] stdout line parses (seq/frames/md5)", bool(m) and m.group(3) ==
          "0cc175b9c0f1b6a831c399e269772661")

    multi = ("[SEQ] a_seq frames=1 md5=" + "a" * 32 + "  fits/frm=1.0\n"
             "[SEQ] b_seq frames=2 md5=" + "b" * 32 + "  fits/frm=1.0\n")
    found = re.findall(r"^\[SEQ\]\s+(\S+)\s+frames=(\d+)\s+md5=([0-9a-f]{32})", multi, re.M)
    check("multi-sequence stdout parses in order", [f[0] for f in found] == ["a_seq", "b_seq"])

    print()
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL':<6}{name:<52}{detail}")
    failed = [r for r in results if not r[1]]
    print("\n" + "=" * 92)
    print(f"[VERDICT] {'PASS - mapping helpers are correct' if not failed else 'FAIL'}"
          f"  ({len(results) - len(failed)}/{len(results)})")
    print("=" * 92)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())