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

#define MOSAIC_API_VERSION 1

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
   * uint8->fp16 widening itself.  THIS IS THE ONLY MODE MEASURED AS CORRECT
   * on this board; the two io-mem modes are kept for experiments. */
  MOSAIC_INPUT_SET = 0,
  MOSAIC_INPUT_IO_MEM_NCHW = 1,
  MOSAIC_INPUT_IO_MEM_NHWC = 2
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
  int32_t gmc_anchor_step;    /* 2 = "official" per-lag semantics,
                                 10 = "tree" anchor cascade (6.19 fits/frame).
                                 10 is ~3.5x cheaper and costs a little
                                 accuracy -- see the report. */
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

  /* inference */
  int32_t canvas_mode;   /* MOSAIC_CANVAS_*                          */
  int32_t input_mode;    /* MOSAIC_INPUT_*                           */
  int32_t npu_workers;   /* 1 or 2.  The board tolerates exactly TWO
                          * concurrent RKNN contexts; a third trips the 6 s
                          * job timeout and soft-resets the NPU. */
  int32_t region_threshold; /* connected-component threshold, 0.06      */
  int32_t min_area;         /* component pixel floor, 4                 */
  int32_t max_detections;   /* components kept per frame, 100           */

  const char* cpu_affinity; /* e.g. "4-7", or NULL.  Affects speed only. */
} MosaicPipelineConfig;

/* ---- accumulated timing, milliseconds ------------------------------------ */
typedef struct {
  double gmc_fit_ms;
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
} MosaicPipelineTiming;

/* ---- opaque handle ------------------------------------------------------ */
typedef struct MosaicPipeline MosaicPipeline;

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

/* The configuration the pipeline ACTUALLY applied, i.e. AFTER the out-of-range
 * clamping mosaic_pipeline_create() performs.  A caller that asked for
 * something impossible needs to see what was really used, not what it
 * requested; the contract test asserts on this.  Returns 0 on success. */
int32_t mosaic_pipeline_effective_config(MosaicPipeline* p, MosaicPipelineConfig* out);

/* Enqueue one raw grayscale frame.  `gray` holds w*h bytes, 8-bit, and is
 * COPIED (the caller keeps ownership).  Returns the frame id (>= 0) or -1 on
 * error.  Frames must be pushed in order. */
int64_t mosaic_pipeline_push(MosaicPipeline* p, const uint8_t* gray, int32_t w, int32_t h);

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
