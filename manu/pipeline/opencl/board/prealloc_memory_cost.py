#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prealloc_memory_cost.py -- what does `--prealloc-pad 1024` actually cost, per sequence?

The delivery config applies `--prealloc-pad 1024` globally. That is the conservative
choice -- it guarantees zero runtime reallocation whatever the corpus throws at it --
but "conservative" and "free" are different words. 1024 is kMaxPadDevice, and only
TWO of the five sequences ever actually reach it.

Geometry is taken from the source, not assumed:
  * quantise_pad() (gmc_stream_ocl.cpp:833-839) rounds UP to powers of two from
    kPadQuantum=32, so the pad values that can ever occur are 32/64/128/.../1024.
  * gpu_slots_ = max_lag + 3 (gmc_stream_ocl.cpp:1266); the run banner prints
    "ring slots=45", so max_lag = 42 and there are 45 device images live at once.
  * the board reports CL_R, 1 byte per pixel (x86 falls back to CL_RGBA at 4 B/px,
    which is one of the reasons x86 device numbers are not comparable).

  bytes_per_slot(P) = (W + 2P) * (H + 2P) * 1
  device_bytes(P)   = bytes_per_slot(P) * 45

W,H = 640x512: every source frame in .bench_bmp is that size (verified).

devpad below is MEASURED, not assumed -- it is the value each sequence reached over
400 frames, and it is identical on x86 and on the board for all five sequences,
which is the one piece of cross-architecture agreement this project has.
"""
import os

W, H = 640, 512
SLOTS = 45                      # gpu_slots_ = max_lag + 3, max_lag = 42
BYTES_PER_PX = 1                # board CL_R; x86 PoCL would be 4
PREALLOC = 1024                 # the delivered --prealloc-pad value

# measured devpad: x86 arm A and the board agree on every one of these
MEASURED = [
    ("VIDEO00034", 512),
    ("VIDEO00028", 1024),
    ("VIDEO00032", 1024),
    ("VIDEO00002", 32),
    ("VIDEO00036", 128),
]
# VIDEO00036 under --fit-corners 200 measured 512 on x86 arm B; prealloc has to
# cover whichever configuration actually ships, so the larger value is listed.
MEASURED_ALT = {"VIDEO00036": 512}


def device_mib(pad):
    return (W + 2 * pad) * (H + 2 * pad) * BYTES_PER_PX * SLOTS / (1024.0 * 1024.0)


def main():
    print("=" * 96)
    print(f"DEVICE IMAGE MEMORY vs --prealloc-pad   "
          f"({W}x{H}, CL_R 1 B/px, {SLOTS} live ring slots)")
    print("=" * 96)
    print(f"{'sequence':<12}{'measured devpad':>17}{'as-is MiB':>12}"
          f"{'@prealloc1024':>15}{'delta MiB':>12}{'x':>8}")
    print("-" * 96)
    total_now = total_pre = 0.0
    rows = []
    for seq, pad in MEASURED:
        eff = max(pad, MEASURED_ALT.get(seq, 0))
        now = device_mib(eff)
        pre = device_mib(PREALLOC)
        rows.append((seq, pad, eff, now, pre, pre - now, (pre / now if now else 0)))
        total_now += now
        total_pre += pre
    for seq, pad, eff, now, pre, d, x in rows:
        note = "" if eff == pad else f"  (alt {eff} w/ corner budget)"
        print(f"{seq:<12}{pad:>13}{note:<18}{now:>10.1f}{pre:>13.1f}"
              f"{d:>+10.1f}{x:>7.1f}x")
    print("-" * 96)
    print(f"{'per-seq sum':<12}{'':>31}{total_now:>10.1f}{total_pre:>13.1f}"
          f"{total_pre - total_now:>+10.1f}{total_pre / total_now:>7.1f}x")
    print()
    print("Only ONE sequence is live at a time (bench_run.sh runs one process per")
    print("sequence), so the figure that matters for peak device memory is the")
    print("largest single row, not the sum:")
    worst = max(rows, key=lambda r: r[5])
    print(f"  worst single sequence : {worst[0]}  {worst[3]:.1f} -> {worst[4]:.1f} MiB "
          f"({worst[5]:+.1f} MiB, {worst[6]:.1f}x)")
    print()
    print("A per-sequence cap would remove the waste entirely. quantise_pad rounds")
    print("up to powers of two, so the cheapest cap that still covers each measured")
    print("devpad exactly IS that devpad:")
    print()
    print(f"{'sequence':<12}{'min safe cap':>14}{'as-is MiB':>12}{'delta vs prealloc1024':>23}")
    print("-" * 62)
    for seq, pad, eff, now, pre, d, x in rows:
        print(f"{seq:<12}{eff:>14}{now:>12.1f}{now - pre:>+23.1f}")
    print()
    print("CAVEAT, and it is not small: those caps are fitted to THIS corpus. A")
    print("sequence on unseen video can require a larger pad than 400 frames of")
    print("this corpus ever did, and the whole point of pre-allocation is that the")
    print("allocation is decided before the first frame. A cap fitted to observed")
    print("maxima gives that guarantee up in exchange for memory this board may")
    print("not need back. See the OOM history in bench_run.sh: a run was SIGKILLed")
    print("at RSS 5065 MiB with MemAvailable at 7.4 GiB, because Mali device memory")
    print("is not reflected in /proc/meminfo at all -- so headroom cannot be")
    print("measured from the host, only paid for.")
    print("=" * 96)


if __name__ == "__main__":
    main()