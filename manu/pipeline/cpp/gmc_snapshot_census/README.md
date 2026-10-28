# The GMC computation snapshot -- the read-set census

This directory holds the EVIDENCE for the one thing the snapshot has to get
right: **which physical frames does ONE computation of the GMC engine read?**

The snapshot copies that set out of the shared history ring before the first
fit and serves every later read of that computation from the private copy.  If
the set is wrong in either direction the change is wrong:

* too small -> reads fall back to the ring (the `n_snapshot_miss` counter
  goes above 0), and the structural guarantee is off;
* too big -> the snapshot wastes memory on frames nobody reads.

So the set was MEASURED, not derived by reading the code.

## What was measured

`frame_at_lag()` was temporarily instrumented to record the PHYSICAL index it
actually served -- not the index that was asked for, because the cold-start
clamp and the hole fallback serve a DIFFERENT frame than the one asked for.
The instrumentation was removed after the census; the traces below are its
output.

Five configurations were traced over the same 60-frame sequence:

| trace file | anchor step | tail | physical anchor |
|---|---|---|---|
| `census_anchorstep2.log`        | 2  | CPU | off |
| `census_anchorstep10.log`       | 10 | CPU | off |
| `census_anchorstep2_physanchor.log`  | 2  | CPU | on |
| `census_anchorstep10_physanchor.log` | 10 | CPU | on |
| `census_anchorstep10_gputail.log`     | 10 | CPU+GPU | off |

## The result

In the steady state (the frames whose whole window is inside the stream) EVERY
configuration reads **the same 22 physical frames**:

    {t} U {t-2, t-4, ..., t-42}

i.e. the frame being computed plus every lag of the median window.  The
per-frame read COUNTS differ (the anchor step changes how many fits run, and
the GPU tail reads the window a second time), but the set of physical frames
is identical.

Two numbers that the code alone did not give:

* the deepest frame ever read is **42** (= `max_lag_`).  A depth of 44 was
  never observed, in any configuration;
* the cold start collapses the deep lags onto the **clamp target** (`first_`),
  so early frames read `{t} U {first_}` -- which is why the snapshot also
  holds a copy of that frame.

## Reproducing it

The sequence is synthetic (a panned random texture -- the read set is a
property of the CODE, not of the pixels), and the traces were produced by the
native x86 build of `gmc_stream_ocl`:

    python3 make_synth.py 60 seq
    cmake --build build --target gmc_stream_ocl -j
    GMC_TRACE_READS=1 ./build/gmc_stream_ocl --raw-root . --sequence seq \
        --limit 60 --anchor-step 10 --fused cpu > census_anchorstep10.log
    python3 agg.py census_anchorstep10.log

`agg.py` prints the per-frame depth histogram and checks that the steady
state has exactly one depth signature.
