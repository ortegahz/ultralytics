/* ==========================================================================
 * mosaic_pipeline.h -- RK3588 f16 multi-scale mosaic inference pipeline.
 *
 * PUBLIC C ABI.  This is the interface the embedded team calls.  It is plain
 * C (no C++ types, no exceptions, no STL) so it can be bound from C, C++,
 * Python (ctypes) or any FFI.
 *
 * WHAT IT DOES
 * ------------
 *   raw infrared frame (8-bit grayscale, 640x512)
 *     -> GMC feature extraction on the Mali GPU (OpenCL fused tail)
 *     -> RGA multi-scale mosaic (native 640x512 + thumb160 + thumb80 into one
 *        640x640 canvas)
 *     -> RKNN fp16 NPU, ONE forward for all scales
 *     -> CPU float32 back-projection + 8-connected components
 *     -> detections in NATIVE frame pixels
 *
 * The pipeline is STREAMING: the GMC engine is causal (it needs the frame
 * history), so frames must be pushed in order.  push() does not block; the
 * internal threads overlap the GMC of frame N+1 with the NPU of frame N.
 *
 * THE SERVICE TOPOLOGY, AND WHERE A FRAME MAY BE LOST
 * ----------------------------------------------------
 *
 *   camera 25 fps
 *      |  push()   -- never blocks, never drops
 *      v
 *   [ camera queue ]  (bounded; on overflow it drops the OLDEST frame -- see
 *      |               below, this is the ONE drop that breaks the pipeline)
 *      v
 *   [ GMC SERVICE ]   2 engines, each with its own OpenCL context.  THIS
 *      |               STAGE MUST NOT LOSE A FRAME.  Its transform chain is
 *      |               built from the frame history, so a frame lost here
 *      |               mis-registers every frame after it.
 *      |  feature
 *      v
 *   [ jitter buffer ] (bounded; on overflow it drops the OLDEST FEATURE --
 *      |               the GMC service never waits for the inference)
 *      v
 *   [ NPU SERVICES ]  2 services, each with its own RKNN context and its own
 *      |               RGA canvas.  THIS STAGE MAY SKIP FRAMES: a forward
 *      |               pass is per-frame independent, so a skipped frame
 *      |               costs one missed detection and nothing else.
 *      |  heatmap
 *      v
 *   [ POST SERVICE ]  back-projection + 8-connected components.  May skip too.
 *      v
 *   detections -> the result callback
 *
 * WHY THE DROP SITS BELOW THE GMC: the inference stage is the slow one
 * (measured: 64.1 ms per frame of critical path with two NPU services, i.e.
 * 15.6 fps), while the GMC service keeps up with a 25 fps source (measured:
 * 31.9 ms per frame with two engines, i.e. 31.4 fps).  A live pipeline
 * therefore lets the INFERENCE skip frames and keeps the GMC complete:
 * every frame gets a correctly registered feature, and the detections arrive
 * as fast as the NPU can produce them.
 *
 * The counters tell the two cases apart:
 *   timing.dropped            -- the camera queue overflowed.  THE GMC
 *                                registration chain is broken.  Must be 0;
 *                                the library prints a [WARN] when it fires.
 *   timing.features_dropped   -- the inference skipped a frame.  Benign.
 *   timing.detections_dropped -- the back-projection skipped a frame.  Benign.
 *   timing.gmc_groups_done    -- the frames the GMC service really computed a
 *                                feature for.  THE AUTHORITATIVE denominator
 *                                for "how many results should I get": it is
 *                                NOT the number of pushed frames, because the
 *                                OPPORTUNISTIC scheduler computes the newest
 *                                frame only and skips the ones it could not
 *                                keep up with.
 *   timing.gmc_frames_skipped -- the frames that were seeded into the history
 *                                but never computed (the engine was busy with
 *                                a newer frame).  NOT an error: this is what
 *                                "opportunistic" means.
 *   timing.gmc_void_frames    -- the frames whose 21-frame history was NOT
 *                                whole (overwritten by a producer that ran
 *                                too far ahead, or a camera frame that was
 *                                lost).  They are VOIDED, never fitted on
 *                                torn data.  Must stay 0 in a healthy run.
 *
 * THREADING CONTRACT
 * ------------------
 *   * push() and collect() are thread-safe against each other, but frames
 *     must be pushed in order and each pushed frame must be collected
 *     exactly once.
 *   * collect() returns frames in COMPLETION order, not push order (with
 *     npu_workers > 1 the two NPU contexts finish out of order).  The frame
 *     id is returned so the caller can restore the original order.
 *   * destroy() drains nothing: call flush() first if you need every frame.
 *
 * ERROR HANDLING
 * --------------
 *   Every call returns a negative value on failure; the human-readable reason
 *   is available from mosaic_pipeline_last_error().  Nothing throws across
 *   the ABI boundary.
 * ========================================================================== */
#ifndef MOSAIC_PIPELINE_H
#define MOSAIC_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* v9: MosaicPipelineTiming gained gmc_void_frames and gmc_frames_skipped,
 * and MosaicPipelineConfig's gmc_compute_stride was RENAMED to gmc_min_stride
 * (same slot, same type -- the MEANING changed: it is no longer a fixed
 * cadence but the minimum physical-frame gap between two GMC groups), so
 * BOTH STRUCT LAYOUTS changed.  A caller built against v8 must be recompiled:
 * the version is checked in mosaic_pipeline_create() and a mismatch is
 * rejected instead of silently reading the wrong fields.
 *
 * The v8 note below still stands:
 * v8: MosaicPipelineTiming gained gmc_groups_done, so the STRUCT LAYOUT
 * changed.  A caller built against v7 must be recompiled: the version is
 * checked in mosaic_pipeline_create() and a mismatch is rejected instead of
 * silently reading the wrong fields.
 *
 * The v7 note below still stands:
 * v7: MosaicPipelineConfig gained gmc_compute_stride (renamed to gmc_min_stride
 * in v9), so the STRUCT LAYOUT
 * changed.  A caller built against v6 must be recompiled: the version is
 * checked in mosaic_pipeline_create() and a mismatch is rejected instead of
 * silently reading the wrong fields.
 *
 * v6: MosaicPipelineConfig gained gmc_phys_anchor and gmc_input_drop_policy,
 * and MosaicPipelineTiming gained gmc_history_gaps, so BOTH STRUCT LAYOUTS
 * changed.  A caller built against v5 must be recompiled: the version is
 * checked in mosaic_pipeline_create() and a mismatch is rejected instead of
 * silently reading the wrong fields.
 *
 * The v5 note below still stands:
 * v5: MosaicPipelineConfig gained gmc_anchor_threads, gmc_affinity_group and
 * npu_core_mask, and MosaicPipelineTiming gained the rknn_run distribution
 * (p50/p95/p99/max/samples), so BOTH STRUCT LAYOUTS changed.  A caller built
 * against v4 must be recompiled: the version is checked in
 * mosaic_pipeline_create() and a mismatch is rejected instead of silently
 * reading the wrong fields.
 *
 * The v4 note below still stands:
 * v4: MosaicPipelineConfig gained gmc_ring_slack and MosaicPipelineTiming
 * gained the ingest breakdown counters, so the STRUCT LAYOUT changed.  A
 * caller built against v3 must be recompiled: the version is checked in
 * mosaic_pipeline_create() and a mismatch is rejected instead of silently
 * reading the wrong fields. */
#define MOSAIC_API_VERSION 9

/* ---- canvas modes ------------------------------------------------------- */
enum {
  /* native 640x512 at (0,0) + thumb160 160x128 at (0,512) + thumb80 80x64 at
   * (160,512) on a 640x640 canvas whose background is 114 in all 3 channels.
   * ONE forward sees all three scales. */
  MOSAIC_CANVAS_MOSAIC = 0,
  /* the PC-side native letterbox replica: native 640x512 at (0,64), tag 0
   * only -- the single-scale reference layout. */
  MOSAIC_CANVAS_SINGLE = 1
};

/* ---- input upload modes ------------------------------------------------- */
enum {
  /* uint8 NHWC upload through rknn_inputs_set.  The runtime does the
   * uint8->fp16 widening itself, which measured 12.78 ms/frame on the board
   * and sat on the NPU's critical path. */
  MOSAIC_INPUT_SET = 0,
  /* Same numbers, a fraction of the host time: WE widen uint8->fp16 with
   * integer NEON (every 0..255 is exact in fp16, so there is no rounding to
   * disagree about) and hand the runtime a FLOAT16 buffer.  MEASURED: the
   * runtime's rknn_inputs_set copies at only ~60-75 MB/s, so handing it
   * FLOAT16 (2.45 MB) is SLOWER than handing it uint8 (0.98 MB) -- 41 ms
   * against 12.8 ms.  The widening is not the cost; the copy is.  Use the
   * io-mem mode below to take the copy off the host's critical path. */
  MOSAIC_INPUT_SET_F16 = 1,
  MOSAIC_INPUT_IO_MEM_NCHW = 2,
  /* The input tensor is bound to host memory and WE write it: the u8->fp16
   * widening costs ~0.8 ms of host time and the runtime's staging into NPU
   * memory happens inside rknn_run -- where the CPU is blocked anyway.  The
   * upload therefore leaves the host's critical path. */
  MOSAIC_INPUT_IO_MEM_NHWC = 3
};

/* ---- one detection, in NATIVE frame pixels (640x512 coordinates) --------- */
typedef struct {
  int32_t tag;   /* 0 = native tile, 1 = thumb160, 2 = thumb80 */
  float cx;      /* centroid x */
  float cy;      /* centroid y */
  int32_t w;     /* bbox width  */
  int32_t h;     /* bbox height */
  int32_t area;  /* component pixel count */
  float peak;    /* max heatmap value inside the component */
  float sum;     /* heatmap integral over the component */
} MosaicDet;

/* ---- configuration ------------------------------------------------------ */
typedef struct {
  int32_t version; /* must be MOSAIC_API_VERSION */

  /* model */
  const char* model_path; /* .rknn file, e.g. /mnt/manu/ptq/models/p0_off_A_none.rknn */

  /* GMC feature engine */
  const char* kernel_dir;     /* OpenCL kernel pair dir, e.g. /root/kernels */
  const char* sequence;       /* sequence name, used only in the md5-out lines */
  int32_t gmc_window;         /* median window, frozen = 21               */
  int32_t gmc_stride_step;    /* frozen = 2                              */
  int32_t gmc_anchor_step;    /* 2 = "official" per-lag semantics (THE DEFAULT):
                                 every lag is fitted DIRECTLY over the long
                                 baseline, with zero composition, so the board
                                 output is BIT-IDENTICAL to the server's frozen
                                 official features.
                                 10 = "tree" anchor cascade (6.19 fits/frame):
                                 ~3.4x cheaper, but NOT equivalent (40.7% of
                                 the frames differ from the server reference). */
  int32_t gmc_downscale;      /* frozen = 2                              */
  int32_t gmc_fit_corners;    /* frozen = 600                            */
  int32_t gmc_fit_win;        /* frozen = 15                             */
  int32_t gmc_fit_level;      /* frozen = 2                              */
  int32_t gmc_fit_iters;      /* frozen = 20                             */
  /* Fit gate: reject a GMC fit whose centre shift exceeds
   * gmc_gate_shift x frame width (or whose scale leaves
   * [gmc_gate_min, gmc_gate_max]).  The C++ reference default is 1.0
   * (kDefaultMaxShiftFrac) and the Python reference has NO gate at all, so
   * 1.0 is the value that keeps the gate from ever firing on healthy motion.
   * DO NOT lower it: a 0.25 gate rejects real FPV motion, the transform
   * degrades to identity and the GMC features stop being registered. */
  int32_t gmc_gate_shift;   /* 1000 == 1.0; 0 == disable the gate */
  int32_t gmc_gate_min;     /* 500 == 0.5 */
  int32_t gmc_gate_max;     /* 2000 == 2.0 */
  /* Wall-clock watchdog for ONE frame's GMC fit, in milliseconds.
   * 0 = OFF (the default, and what the accuracy validation uses: with the
   * watchdog on, a frame that blows its budget DEGRADES -- the transform
   * falls back to identity or to the last good one -- so the output stops
   * being bit-reproducible).
   *
   * This is the real-time safety valve: the source is a 25 fps camera, so
   * the feature stage has a 40 ms budget per frame.  Set it above the
   * expected fit time (the measured fit is ~60 ms per frame on ONE engine,
   * ~30 ms with two engines working on interleaved frames) so it only fires
   * in a pathological case, and a pathological frame degrades instead of
   * stalling the whole stream.
   */
  int32_t gmc_deadline_ms;
  /* RANSAC sampling cap for the GMC fit.  0 = OFF (the reference's own
   * default, and what the bit-exactness contract requires).  A positive
   * value bounds the sampling iterations so a pathological frame cannot spin
   * -- but it CHANGES THE NUMBERS, so it must be reported as a deviation. */
  int32_t gmc_ransac_max_iters;
  /* Reuse the per-frame half-res image, the Shi-Tomasi corners and the
   * optical-flow pyramid across fits.  The GMC header states this is
   * BIT-EXACT vs the uncached path -- it only removes recomputation.
   * VERIFIED, not assumed: on the 11,810-frame val3 track the cached and
   * uncached runs produce BYTE-IDENTICAL detection TSVs (md5
   * 0bf598a1d890040bcfa8d157dc69c032), and the per-frame transform chain
   * (mats) and all three feature channels match bit for bit over 60 frames.
   * Default 1 (on): it cuts the GMC fit from 71.4 to 60.1 ms/frame
   * (12.507 fps vs 11.855 fps end to end) at zero accuracy cost.
   * Set 0 to fall back to the reference implementation's default. */
  int32_t gmc_pyr_cache;
  /* Reserve the OpenCL device images once for a pad of N pixels, so the frame
   * loop can never reallocate.  0 = off. */
  int32_t gmc_prealloc_pad;
  /* Compute the per-frame feature fingerprint (the rolling MD5 over the
   * PRE-ENCODE channel planes).  1 = on (the default; the audit tools need
   * it), 0 = off.  Turning it off saves ~5-7 ms of CPU per frame in the GMC
   * service -- pure instrumentation overhead, no effect on any number. */
  int32_t gmc_feature_fingerprint;

  /* inference */
  int32_t canvas_mode;   /* MOSAIC_CANVAS_*                          */
  int32_t input_mode;    /* MOSAIC_INPUT_*                           */
  int32_t npu_workers;   /* 1 or 2.  The board tolerates exactly TWO
                          * concurrent RKNN contexts; a third trips the 6 s
                          * job timeout and soft-resets the NPU. */
  int32_t region_threshold; /* connected-component threshold, 0.06      */
  int32_t min_area;         /* component pixel floor, 4                 */
  int32_t max_detections;   /* components kept per frame, 100           */

  /* ---- the services -------------------------------------------------------
   * Every stage of the pipeline is a service: its own bounded queue, its own
   * worker pool, its own scratch.  The frame API stays push/collect; these
   * knobs only size the pools.  Defaults are the measured board values.
   */
  int32_t gmc_workers;   /* feature service: engines, each computes INTERLEAVED frames
                           * (the per-frame computation depends only on pixels, so
                           *  several engines may work on different frames).
                           * DEFAULT 1: the board A/B showed 1 engine and 2 engines
                           * produce byte-identical GMC features, and the NPU stage
                           * is the bottleneck, so the second engine buys nothing.
                           * The hard cap stays 2.
                           *
                           * THE OPPORTUNISTIC SCHEDULER FORCES THIS TO 1 (v9).
                           * WHY: several engines would all grab the SAME newest
                           * frame (they share one history and one head), so the
                           * extra engines would produce DUPLICATE results, not
                           * extra throughput; and two engines measured the GMC
                           * fit's p99 at 695 ms against 200 ms with one, because
                           * they fight over the same CPU.  A request for more
                           * than one is forced down to one and logged. */
  /* THE HISTORY RING'S SLACK, in frames.
   *
   * WHAT IT IS NOW (v9): one of the two terms of the ring's depth --
   * ring_depth = max(GMC_RING_BUFFER_SIZE = 64, max_lag + 1 + gmc_ring_slack).
   * It is no longer a "the ingest must not outrun the engines" budget: under
   * the OPPORTUNISTIC scheduler the ingest NEVER waits, so it runs ahead
   * freely, and the ring's depth is what keeps that freedom SAFE.
   *
   * WHY the depth matters now more than ever: seeding frame S overwrites the
   * slot that held S - ring_depth, and the engine computing frame J still
   * needs frames J-42..J.  The engine checks that the whole history is still
   * there (history_complete) and VOIDS the frame if it is not, so a producer
   * that runs too far ahead costs a frame instead of producing wrong pixels.
   *
   * THE EFFECTIVE SLACK of the default configuration: ring_depth 64 - 43
   * frames the algorithm reads = 21 frames of producer headroom = 840 ms at
   * 25 fps, against a worst measured GMC stall of 250 ms.  A gmc_ring_slack
   * above 21 only deepens the ring (more raw frames stay alive: ring_depth x
   * 640x512 B); it buys no accuracy.
   *
   * 0 = AUTO (the default): the library derives it from gmc_workers as
   * max(64, 32*gmc_workers).
   *
   * THE AUTO RULE IS RESOLVED BY THE LIBRARY, ONCE.  A request of 0 is
   * resolved during create()'s sanitize stage, and
   * mosaic_pipeline_effective_config() then reports the RESOLVED budget -- so
   * a caller reads the budget in force instead of re-deriving the rule (a
   * second copy of the formula would drift, and the report would then state
   * a budget the pipeline does not use). */
  int32_t gmc_ring_slack;
  int32_t post_workers;  /* back-projection + connected components service   */
  int32_t queue_depth;   /* depth of the queues BETWEEN the services         */
  /* THE JITTER BUFFER: the queue between the GMC service and the NPU
   * services, and its depth in frames (8 is the default).
   *
   * WHY: the GMC's per-frame cost is NOT flat.  The anchor frames pay for the
   * full Shi-Tomasi + pyramidal LK + RANSAC solve, the others only for the
   * cheap composition, and a texture-heavy scene makes even the cheap ones
   * expensive.  Without a buffer between the stages, one expensive frame
   * reaches the NPU as a stall: the NPU service drains its queue, idles,
   * and is then hit by the accumulated burst.  With the buffer the GMC may
   * run AHEAD by `jitter_depth` frames, so the burst is absorbed there and
   * the NPU sees a flat stream.  The proof is in the timing report:
   * npu_wait_ms / frames == 0 means the NPU never starved.
   *
   * IT IS ALSO THE DROP POINT.  When the NPU services cannot keep up with a
   * 25 fps source, this queue overflows and the OLDEST feature is dropped --
   * the GMC service never waits for the inference.  That is deliberate: the
   * inference stage is per-frame independent, so skipping a frame costs one
   * missed detection, whereas dropping the frame at the CAMERA queue costs
   * the GMC registration of every frame after it.
   *
   * IT MUST BE DEEPER THAN THE PRODUCER'S IN-FLIGHT WINDOW, or a healthy
   * self-paced run would overflow it and skip frames the producer had every
   * right to expect a result for.  8 against an in-flight window of 4.
   */
  int32_t jitter_depth;
  int32_t input_queue;   /* camera-facing queue depth.  On overflow the OLDEST
                           * frame is dropped -- the camera must never stall.
                           * This is the LAST RESORT: with the GMC service
                           * keeping up, it never overflows, and if it ever
                           * does, the GMC registration chain is broken. */
  /* Per-NPU-service CPU list, comma separated, one entry per NPU service:
   * "4,5" pins NPU service 0 to core 4 and service 1 to core 5.  NULL/empty
   * = inherit cpu_affinity.
   *
   * IMPLEMENTED in API v3 (it was declared but never consumed in v2 -- a dead
   * knob is worse than no knob: the caller believes it took effect).
   *
   * WHY it matters: the stages have very different profiles.  An NPU service
   * spends most of its wall time BLOCKED on the NPU hardware, while a GMC
   * fit is a 60 ms CPU burst that must not be preempted.  Measured on the
   * board: with every thread on the same four cores the two GMC engine
   * threads each received only ~0.85 of a core and the feature stage ran at
   * 23.3 fps -- below the 25 fps the camera delivers -- so the camera queue
   * overflowed and the registration chain broke.  Pinning the GMC engines to
   * their own cores is a SCHEDULING change only: it cannot alter a single
   * number (the fit result does not depend on which core runs it). */
  const char* npu_affinity;

  /* Per-GMC-engine CPU list, comma separated, one entry per GMC engine:
   * "4,5" pins GMC engine 0 to core 4 and engine 1 to core 5.  NULL/empty =
   * inherit cpu_affinity.  See npu_affinity for why this exists.
   *
   * NOTE the scope: ONE core per engine.  A GMC engine that is pinned to a
   * single core cannot use the anchor parallelism below -- the anchor workers
   * would all share that one core.  Use gmc_affinity_group for that. */
  const char* gmc_affinity;

  /* Per-GMC-engine CPU GROUP, ';'-separated, one group per GMC engine:
   * "4-5;6-7" gives engine 0 the cores {4,5} and engine 1 the cores {6,7}.
   * Within a group the engine's ANCHOR WORKERS are spread over the group's
   * cores: the engine thread takes the first core, the anchor helpers the
   * rest.  NULL/empty = no groups (the engines are placed by gmc_affinity,
   * or inherit cpu_affinity).
   *
   * WHY a GROUP and not one core: the GMC fit is 5 anchor fits, and their LK
   * stages can run concurrently (see gmc_anchor_threads).  On a single core
   * there is nothing to run them on, so the parallelism buys nothing -- the
   * knob exists so that the anchor workers have real cores to land on.
   *
   * This is a SCHEDULING change only: it cannot alter a single number. */
  const char* gmc_affinity_group;

  /* The number of WORKERS of the anchor LK stage, per GMC engine, the engine's
   * own thread INCLUDED: 1 == the reference's serial anchor loop, 5 == one
   * worker per anchor (the shipping anchor grid has 5 anchors).
   *
   * WHAT IT DOES: the 5 anchor fits are split into an LK stage and a RANSAC
   * stage.  The LK stages run CONCURRENTLY (each anchor still runs the very
   * same single-threaded OpenCV code on the very same bytes); the RANSAC
   * stages then run SEQUENTIALLY, in the anchor order, in the engine thread.
   * That order is what keeps the output BIT-IDENTICAL: RANSAC is the only
   * consumer of OpenCV's global RNG, so the draws must happen in the same
   * order as before.  The result is a faster fit with the SAME numbers.
   *
   * The effective worker count is capped by the engine's core group (see
   * gmc_affinity_group): more workers than cores would just oversubscribe
   * them.  1 (the default) is the reference behaviour. */
  int32_t gmc_anchor_threads;

  /* The NPU core mask (rknn_core_mask) for EVERY NPU context:
   *   0 = the shipped behaviour -- rknn_set_core_mask is NOT called and the
   *       runtime picks the cores itself (RKNN_NPU_CORE_AUTO).  THIS IS THE
   *       DEFAULT and the certified baseline was measured this way.
   *   1 = RKNN_NPU_CORE_0, 2 = RKNN_NPU_CORE_1, 4 = RKNN_NPU_CORE_2,
   *   3 = CORE_0_1, 6 = CORE_1_2, 5 = CORE_0_2, 7 = CORE_0_1_2.
   *
   * WHY the knob exists: the RK3588 has THREE NPU cores and the runtime was
   * never told which ones to use.  A single 640x640 fp16 forward pass costs
   * ~108-112 ms, and the NPU stage -- not the GMC -- is what caps the
   * pipeline at ~15.8 fps.  This is the only lever left on the NPU side.
   *
   * DEVIATION: a non-zero mask may change the throughput AND the numerics (a
   * different core is a different execution path).  A run with it is a
   * measured deviation and must be reported as one. */
  int32_t npu_core_mask;

  const char* cpu_affinity; /* e.g. "4-7", or NULL.  Affects speed only. */

  /* THE PHYSICAL-FRAME-NUMBER ANCHOR of the GMC history.  0 = OFF (the
   * default, and the certified behaviour): the GMC engine anchors its
   * 21-frame history on the CONSUMPTION counter, i.e. "the frame the engine
   * ingested N steps ago".  1 = ON: the engine anchors on the camera's
   * PHYSICAL frame number, i.e. "the frame the camera took N steps ago".
   *
   * WHY IT EXISTS: the two indexings only differ when a frame is LOST.  The
   * camera queue drops a frame when the pipeline cannot keep up, and the id
   * of that frame is gone for good: with the consumption counter the engine
   * cannot tell, so its "lag 2" silently becomes a physical lag 3 and the
   * 21-frame median window is stretched without anybody noticing (that is
   * the mechanism behind "the registration chain is broken").  With the
   * physical anchor the engine KNOWS: the ring is indexed by the physical
   * frame number and every slot carries the number it holds, so a missing
   * frame is a DETECTED hole (counted in timing.gmc_history_gaps, and
   * reported per frame on stderr) instead of a silently wrong frame.
   *
   * OFF BY DEFAULT: the switch exists so the new behaviour can be validated
   * on the board before it becomes the default.  With no frame ever lost,
   * the two indexings select the SAME pixels, so turning it on does not
   * change a single bit of the feature stream.
   *
   * NOTE: this only changes WHICH frames the GMC reads; it never changes the
   * frame ids the caller sees (mosaic_pipeline_push still returns the camera
   * frame id) and it never changes the artifact order. */
  int32_t gmc_phys_anchor;

  /* WHAT THE CAMERA QUEUE DOES WHEN IT OVERFLOWS.  0 = EVICT THE OLDEST
   * (the default, and the shipped behaviour): the freshest frame is kept and
   * the stalest queued frame is thrown away.  1 = REJECT THE NEWEST: the
   * queued frames are kept and the incoming frame is thrown away.
   *
   * The camera is NEVER blocked (a 25 fps camera must not be stalled by the
   * pipeline), so a frame MUST be lost when the queue is full -- the only
   * question is which one.
   *
   * WHY 0 IS THE DEFAULT: the queue is the camera's hand-off to a service
   * that may not keep up, and a live stream wants FRESH data, not a complete
   * but increasingly stale one.  Under a sustained overload, policy 1 keeps
   * the queue full of OLD frames and the GMC service falls further and
   * further behind (unbounded latency), while policy 0 keeps the GMC service
   * working on the newest frames (bounded latency, bounded staleness).
   *
   * WHAT BOTH POLICIES COST: the physical frame id is consumed at push time,
   * so a lost frame leaves a HOLE in the id sequence either way, and the
   * frames whose 21-frame history reaches across that hole have a stretched
   * temporal window.  gmc_phys_anchor=1 is what makes that hole visible
   * instead of silent; the policy only decides WHICH frames are lost and
   * how stale the output is. */
  int32_t gmc_input_drop_policy;

  /* THE MINIMUM GAP BETWEEN TWO GMC GROUPS, in frames.
   *
   * THIS IS NOT A CADENCE ANY MORE.  Under the OPPORTUNISTIC latest-frame
   * scheduler there is no fixed grid: the engine computes whatever the
   * NEWEST frame is when it becomes free, and this number only says how far
   * apart two consecutive groups have to be.
   *
   * 1 (the default) = NO LIMIT: the engine grabs the newest frame the moment
   * it is free.  This is the ADAPTIVE mode -- the service always works on the
   * freshest data and its throughput is whatever the hardware allows, and the
   * frames it could not keep up with are simply SKIPPED (they still seed the
   * history, so skipping never punches a hole in it).
   *
   * >1 = an artificial rate limit: the engine waits until the newest frame is
   * at least N frames past the one it computed last (2 = at least two frames
   * between two groups).  Use it only to pin a frame rate that the caller
   * needs downstream; it buys nothing in accuracy and it makes the output
   * older.
   *
   * The gap is counted in the SAME index space the engine computes in: the
   * camera's physical frame number when gmc_phys_anchor=1, the dense
   * consumption counter otherwise.
   *
   * RENAMED in v9 from gmc_compute_stride, whose meaning was the opposite
   * (a fixed "every Nth frame" cadence on the physical grid).  Same slot, same
   * type -- the layout is unchanged, only the semantics are. */
  int32_t gmc_min_stride;
} MosaicPipelineConfig;

/* ---- accumulated timing, milliseconds ------------------------------------ */
typedef struct {
  double gmc_fit_ms;
  /* The work the GMC service does AFTER the fit: the plane->interleaved
   * conversion, the optional per-frame feature fingerprint and the hand-off
   * to the NPU queue.  It is NOT part of gmc_total_ms (which measures the
   * algorithm inside process_frame), and it is NOT negligible: measured on
   * the board it was ~22 ms of CPU per frame, which is what pushed the
   * feature stage below the 25 fps of the camera. */
  double gmc_post_ms;
  /* The INGEST thread: it is the only thing between the camera queue and the
   * GMC engines, and it is where a starvation shows up first.  These counters
   * say WHICH gate it is stuck on:
   *   gmc_ingest_ms        the ingest's total wall time
   *   gmc_window_blocks    times it waited for the history-window guard
   *   gmc_space_blocks     times it waited for room in an engine's job queue
   * A "window_blocks" that grows with the run means the pipeline's RATE is
   * capped by the guard, not by the work.
   *
   * v9: under the OPPORTUNISTIC scheduler the ingest NEVER WAITS -- it seeds
   * the history and moves on.  The two block counters are therefore KEPT (the
   * layout is frozen) but they are never incremented any more, and their
   * reading of 0 IS the evidence that the back-pressure path is gone: the
   * camera queue is no longer stalled by the GMC service. */
  double gmc_ingest_ms;
  int64_t gmc_window_blocks;
  int64_t gmc_space_blocks;
  /* WHERE the ingest's wall time went.  gmc_ingest_ms is the ingest's TOTAL
   * wall time; these three split it into the waits it was blocked on:
   *   gmc_ingest_in_wait_ms  waiting for the camera queue to deliver a frame
   *                          (large = the pipeline is STARVED, i.e. it is
   *                          faster than the source -- the healthy case)
   *   gmc_ingest_window_ms   blocked by the history-window guard (the
   *                          lookahead budget was exhausted: the ingest was
   *                          faster than the engines)
   *   gmc_ingest_space_ms    blocked by an engine's job queue being full
   * The sum of the three plus the ingest's own work is gmc_ingest_ms, so the
   * split is a COMPLETE accounting of the ingest, not an estimate.
   *
   * v9: the ingest does not block any more, so the last two are always 0 --
   * they are KEPT (the layout is frozen) and their reading of 0 is the
   * evidence that the back-pressure is gone. */
  double gmc_ingest_in_wait_ms;
  double gmc_ingest_window_ms;
  double gmc_ingest_space_ms;
  /* The depth of the engine job queue AT THE MOMENT the engine popped a job
   * (the popped job included).  THIS IS THE PROOF THAT THE ENGINES ARE NOT
   * STARVING: a mean of ~1 means every engine picked its job up off an empty
   * queue (it was waiting for work), a mean well above 1 means the engine had
   * a backlog and never went idle.  gmc_queue_samples is the number of pops
   * the average is over.
   *
   * v9: there IS no job queue any more (the engine grabs the newest frame
   * straight from the history ring), so these two counters are never
   * incremented: they read 0 / 0.  They are KEPT so the layout stays frozen
   * and so an old reader sees a zero instead of a wild number. */
  double gmc_queue_depth_sum;
  int64_t gmc_queue_samples;
  double gmc_warp_ms;
  double gmc_median_ms;
  double gmc_total_ms;
  double rga_import_ms;
  double rga_improcess_ms;
  double upload_ms;
  double run_ms;
  double outputs_ms;
  double backproj_ms;
  int64_t frames;   /* frames that finished the whole path */
  double wall_ms;   /* wall clock since create()           */
  int64_t dropped;  /* frames dropped at the CAMERA queue because the pipeline
                     * could not keep up.  The camera is never stalled, so this
                     * is the back-pressure valve of a real-time stream.
                     *
                     * THIS IS THE ONE DROP THAT BREAKS THE PIPELINE: a frame
                     * lost here never reaches the GMC, so the transform chain
                     * loses a step and every later frame is mis-registered.
                     * It must stay 0 in a healthy run; a non-zero value means
                     * the GMC service could not keep up with the source. */
  /* Frames dropped DOWNSTREAM of the GMC service.  These are benign: the GMC
   * feature stream is complete and correctly registered, only the NPU (or the
   * back-projection) skipped a frame it could not process in time.  This is
   * the intended behaviour of a live pipeline: the inference stage runs as
   * fast as it can and skips what it cannot keep up with, because the
   * detection of a skipped frame costs nothing but a missed target, whereas
   * a frame lost at the GMC costs the registration of every frame after it. */
  int64_t features_dropped;   /* a GMC feature the NPU services never saw    */
  int64_t detections_dropped; /* a heatmap the back-projection never saw     */
  /* Starvation: how long each service waited for work.  In a healthy
   * pipeline these are ~0 -- that is what the jitter buffer is for. */
  double gmc_wait_ms;   /* the GMC engines waiting for a frame to compute   */
  double npu_wait_ms;   /* the NPU services waiting for a feature frame    */
  double post_wait_ms;  /* the post workers waiting for a heatmap          */
  double gmc_fit_max_ms;/* the worst SINGLE GMC fit (the jitter spike)      */
  /* The distribution of the per-frame GMC fit, over every frame the GMC
   * service fitted.  NEAREST-RANK percentiles (a percentile is always a
   * value some frame actually took -- an interpolated p99 is not an
   * observation, and a WCET claim has to be made of one).  The max alone is
   * not enough: a mean of 60 ms with a p99 of 75 ms and a max of 216 ms says
   * the stage has a long tail, and the tail is what a jitter buffer and a
   * watchdog have to absorb. */
  double gmc_fit_p50_ms;
  double gmc_fit_p95_ms;
  double gmc_fit_p99_ms;
  int64_t gmc_fit_samples;  /* how many frames the distribution is over */
  /* The distribution of the per-frame rknn_run, over every frame the NPU
   * services ran.  SAME shape and the same reason as the fit distribution:
   * the NPU is the pipeline's critical path, and a mean alone (112 ms) hides
   * whether the stage has a long tail.  It is also the measurement the NPU
   * core mask is judged on: a mask that only makes the MEAN faster is not a
   * throughput win if the tail grows. */
  double rknn_run_p50_ms;
  double rknn_run_p95_ms;
  double rknn_run_p99_ms;
  double rknn_run_max_ms;
  int64_t rknn_run_samples;  /* how many frames the distribution is over */
  /* WHY the fit costs what it costs: the Shi-Tomasi corner budget and how
   * often a frame saturates it.  The GMC fit cost is CONTENT-DRIVEN, not a
   * constant: on a sparse-texture sequence the same code fits in ~24 ms/frame,
   * on a dense one the corner budget saturates and the same code needs ~72
   * ms/frame.  Quoting a mean without these two numbers invites a "the GMC is
   * too slow" conclusion that the next sequence will contradict. */
  double gmc_corner_mean;     /* mean corners per detection, over the run */
  int64_t gmc_corner_capped;  /* frames that hit the corner budget        */
  int64_t gmc_deadline_blows; /* frames whose GMC fit blew the wall-clock
                               * watchdog and therefore DEGRADED.  In a healthy
                               * run this is 0: a degraded frame is a frame
                               * whose transform fell back to identity. */
  /* MISSING PHYSICAL HISTORY FRAMES, counted per LOOKUP (one frame can miss
   * several of its 21 lags).  It is 0 unless gmc_phys_anchor=1: with the
   * consumption-counter anchor the index sequence is dense by construction,
   * so a lost camera frame is INVISIBLE to the engine -- which is why the
   * loss used to be silent.  With the physical anchor the engine sees the
   * hole and counts it here, and the affected frames are printed on stderr
   * as they are computed.  In a healthy run this is 0. */
  int64_t gmc_history_gaps;
  /* The GMC GROUPS the GMC service ACTUALLY COMPLETED: the number of frames
   * for which a feature was really computed.
   *
   * WHAT IT COUNTS: one increment per GMC group that came out of the GMC
   * service, i.e. per frame whose feature exists.
   *
   * WHAT IT DOES NOT COUNT:
   *   * the frames the CAMERA queue dropped -- they never reached the GMC
   *     service, so they have no feature at all (that is timing.dropped);
   *   * the frames that were only SEEDED into the history ring without
   *     becoming a group -- under the OPPORTUNISTIC scheduler the engine
   *     computes the newest frame and SKIPS the ones it could not keep up
   *     with; those skipped frames are still seeded (so they are history and
   *     nothing else).  See gmc_frames_skipped and gmc_min_stride in
   *     MosaicPipelineConfig.
   *
   * WHY IT EXISTS: it is the AUTHORITATIVE denominator for "how many frames
   * SHOULD have produced a result".  The number of PUSHED frames is NOT that
   * number once frames are skipped, so a caller that checks the delivered
   * results against the pushed frames is wrong by construction.  With this
   * counter the caller computes
   *     expected results = gmc_groups_done - features_dropped
   *                       - detections_dropped
   * which holds for every scheduler. */
  int64_t gmc_groups_done;
  /* The frames the engine FOUND but refused to compute: their 21-frame
   * history was not whole.
   *
   * WHY IT CAN HAPPEN: the engine computes the NEWEST frame, and the producer
   * (the ingest) is never blocked, so the producer may run ahead of the
   * engine.  Once it is more than (ring_depth - 43) frames ahead, the frames
   * the engine still needs have been OVERWRITTEN in the history ring -- and
   * then the frame is VOIDED rather than fitted against torn pixels.  A
   * camera frame that was lost does the same thing to the frames whose
   * history reaches across the hole (with the physical anchor).
   *
   * It is 0 in a healthy run: the ring's 21 frames of producer headroom
   * (840 ms at 25 fps) cover the worst measured GMC stall (250 ms).  A
   * non-zero value means the producer outran the ring -- look at
   * gmc_frames_skipped and at the ingest's wall time. */
  int64_t gmc_void_frames;
  /* The frames that were SEEDED into the history but never became a group:
   * the engine was busy with a newer frame and simply moved on.
   *
   * THIS IS NOT AN ERROR.  It is the definition of the opportunistic
   * scheduler: the engine always works on the FRESHEST frame it can, and the
   * frames it skipped are still in the history (they are the lags the next
   * fit reads), so skipping never punches a hole in the history.
   *
   * It is computed as
   *     seeded frames - groups done - void frames
   * and is therefore the number of frames the camera delivered that the
   * engine deliberately did not compute. */
  int64_t gmc_frames_skipped;
} MosaicPipelineTiming;

/* ---- opaque handle ------------------------------------------------------ */
typedef struct MosaicPipeline MosaicPipeline;

/* Result callback.
 *
 * Invoked from a dedicated dispatcher thread as soon as a frame's detections
 * are ready -- in COMPLETION order, not frame order (with more than one NPU
 * service the frames finish out of order; `frame_id` says which frame this
 * is).  `dets` is valid ONLY for the duration of the call: copy what you
 * need.
 *
 * The callback runs on a pipeline thread, so keep it short.  A slow callback
 * does not stall the pipeline (the results queue in front of it is unbounded),
 * it only delays the delivery of the results that come after it.
 *
 * Registering a callback switches the result path: the results are then
 * delivered to the callback and are NOT buffered for mosaic_pipeline_collect(),
 * which would otherwise hand the same frame out twice.
 */
typedef void (*MosaicResultCallback)(void* user, int64_t frame_id, const MosaicDet* dets,
                                      int32_t n_dets);

/* Feature fingerprint.
 * The rolling MD5 over the GMC's PRE-ENCODE 3 x (H*W) channel planes, computed
 * exactly like the reference implementation's --md5-out (gmc_stream_ocl.cpp:
 * "Comparing decoded JPGs would only measure JPEG quantisation, a storage
 * artefact, not feature drift").  It is the fingerprint that proves the
 * on-board feature engine is bit-identical to the reference.  Returns a
 * 32-char hex string, or "" if the pipeline is NULL. */
const char* mosaic_pipeline_feature_md5(MosaicPipeline* p);

/* Optional per-frame feature fingerprint file, in the reference's --md5-out
 * format:  <seq>\t<index>\t<md5>  (one line per pushed frame).  The hash is
 * over the GMC's PRE-ENCODE 3 x (H*W) channel planes, so it measures FEATURE
 * drift and not the storage codec.  Pass NULL to disable. */
void mosaic_pipeline_set_feature_md5_out(MosaicPipeline* p, const char* path);

/* Optional per-frame feature DUMP, in the reference implementation's --out-dir
 * layout:  <dir>/<seq>_%06d.bmp
 * The file is an 8-bit 3-channel image whose channel order is
 * [Ch0 I_t, Ch1 diff, Ch2 residual] -- byte-for-byte the same array and the
 * same cv::imwrite call the reference uses, so the two sides are directly
 * comparable with a per-pixel diff (the MD5 stream cannot answer "how far
 * apart": a single flipped bit changes the whole hash).
 * Pass NULL to disable.  Costs one BMP encode per frame. */
void mosaic_pipeline_set_feature_dump_out(MosaicPipeline* p, const char* dir);

/* Optional dump of the GMC transform chain, in the reference implementation's
 * --mats-all format: a RAW float32 stream, 21 lags x 6 floats per frame
 * (a, b, tx, b, a, ty), lags in ascending order -- byte-for-byte the same
 * layout the reference writes, so the two sides are directly comparable.
 * This is the instrument that separates "the fit differs" from "the warp
 * differs": the transforms are the only thing the fit produces.
 * Pass NULL to disable. */
void mosaic_pipeline_set_mats_dump_out(MosaicPipeline* p, const char* path);

/* Fill cfg with the frozen defaults (frozen GMC knobs, mosaic canvas,
 * input_mode=set, npu_workers=2, region_threshold=0.06, min_area=4,
 * max_detections=100).  model_path and kernel_dir are NOT set. */
void mosaic_pipeline_config_defaults(MosaicPipelineConfig* cfg);

/* Create the pipeline.  Returns NULL on failure; the reason is written into
 * errbuf (if errbuf != NULL and errlen > 0). */
MosaicPipeline* mosaic_pipeline_create(const MosaicPipelineConfig* cfg, char* errbuf,
                                        size_t errlen);

/* Last error message of this pipeline ("" if none). */
const char* mosaic_pipeline_last_error(MosaicPipeline* p);

/* TEST HOOK -- make the pipeline fail on purpose.  Returns 0 on success.
 *
 * WHY IT EXISTS (measured: a hung board cost a full validation round)
 *   A FATAL error inside the library -- an NPU submit the driver refuses, for
 *   instance -- must ABORT the pipeline, not wedge it.  That path was SILENT:
 *   the error was recorded and the abort flag was set, but every other thread
 *   stayed asleep on its condition variable for ever, because a predicate is
 *   only re-evaluated when the thread is woken.  The pipeline hung with no
 *   message and the caller waited for a result that would never come.
 *
 *   Reproducing that needs a failing NPU, which is not something a test may
 *   arrange safely (the board tolerates exactly two contexts and a third
 *   takes the whole NPU down with it).  So the abort path is driven directly
 *   instead: this hook is the SAME set_error() the NPU service calls.
 *
 * IT IS A TEST HOOK.  It exists to prove the abort path and it must never be
 * called in production. */
int32_t mosaic_pipeline_inject_fault(MosaicPipeline* p, const char* why);

/* The configuration the pipeline ACTUALLY applied, i.e. AFTER the out-of-range
 * clamping mosaic_pipeline_create() performs.  A caller that asked for
 * something impossible needs to see what was really used, not what it
 * requested; the contract test asserts on this.  Returns 0 on success. */
int32_t mosaic_pipeline_effective_config(MosaicPipeline* p, MosaicPipelineConfig* out);

/* Enqueue one raw grayscale frame.  `gray` holds w*h bytes, 8-bit, and is
 * COPIED (the caller keeps ownership).  Returns the frame id (>= 0) or -1 on
 * error.  Frames must be pushed in order.
 *
 * NEVER BLOCKS.  The camera must not be stalled by a pipeline that cannot keep
 * up, so when the camera-facing queue is full the OLDEST frame is dropped
 * (the freshest data wins) and the drop is counted in
 * mosaic_pipeline_get_timing().  The queues BETWEEN the services do apply
 * back-pressure -- the pipeline paces itself, the camera never does.
 *
 * If the pipeline cannot take the frame at all (no pipeline, bad size), -1 is
 * returned and the reason is available from mosaic_pipeline_last_error().
 */
int64_t mosaic_pipeline_push(MosaicPipeline* p, const uint8_t* gray, int32_t w, int32_t h);

/* Register the result callback (see MosaicResultCallback).  Passing NULL
 * removes it and switches the result path back to mosaic_pipeline_collect().
 * The callback is invoked from a dedicated dispatcher thread; it must not call
 * back into the pipeline's own functions that take the result lock. */
int32_t mosaic_pipeline_set_result_callback(MosaicPipeline* p, MosaicResultCallback cb,
                                            void* user);

/* Block until the next frame has finished and copy its detections out.
 * `out` may be NULL if the caller only wants the count.  Returns the number
 * of detections (>= 0), or -1 on error.  *frame_id receives the id of the
 * frame (pass NULL to ignore it). */
int32_t mosaic_pipeline_collect(MosaicPipeline* p, int64_t* frame_id, MosaicDet* out,
                                int32_t max_out);

/* Stop accepting frames, wait for every queued frame to finish, and join the
 * internal threads.  Returns 0 on success.  collect() may still be called
 * afterwards to drain the results. */
int32_t mosaic_pipeline_flush(MosaicPipeline* p);

/* Accumulated timing.  Returns 0 on success. */
int32_t mosaic_pipeline_get_timing(MosaicPipeline* p, MosaicPipelineTiming* out);

/* Release the pipeline.  Safe on NULL.  Does NOT flush: call flush() first if
 * the queued frames matter. */
void mosaic_pipeline_destroy(MosaicPipeline* p);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* MOSAIC_PIPELINE_H */
