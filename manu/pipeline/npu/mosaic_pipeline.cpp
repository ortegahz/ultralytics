// ===========================================================================
// mosaic_pipeline.cpp -- the RK3588 f16 multi-scale mosaic inference library.
//
//   raw infrared (8-bit gray, 640x512)
//     -> GMC feature engine, Mali GPU fused tail (OpenCL)
//     -> RGA multi-scale mosaic into one 640x640 canvas
//     -> RKNN fp16, ONE forward for all three scales
//     -> CPU float32 back-projection + 8-connected components
//     -> detections in native frame pixels
//
// WHY A LIBRARY AND NOT A DEMO
// -----------------------------
// The embedded team drives this from their own application, so the inference
// core has to be callable.  The public ABI is mosaic_pipeline.h (plain C);
//// everything below is private.
//
// WHAT IS REUSED (and why it is safe)
// ------------------------------------
//   * mosaic_core.h      -- the RGA/NPU/back-projection core.  It is the SAME
//                           code the archived 28,997-frame run used; the
//                           BMP-driven binary and this library call the same
//                           ProcessFrame().
//   * gmc_stream_ocl.cpp  -- the OpenCL-fused GMC engine, already validated
//                           bit-for-bit against the Python reference
//                           (cpp_port_rk3588.md, L2: G1 400/400).
//
// PIXEL CONTRACT (the one thing that must not drift)
// ---------------------------------------------------
// The model consumes the channels in the order [residual, diff, I_t].
//   * the feature BMP stores BGR, stb_image returns RGB, and that RGB order
//     IS the model's order;
//   * the GMC engine returns a PLANE-MAJOR 3 x (H*W) uint8 Mat whose rows are
//     [I_t, diff, residual], so the RGB buffer built below is
//     [residual, diff, I_t] -- byte-identical to what stb_image returns.
// Both paths therefore feed the NPU the same bytes.
//
// THREADING
// ---------
//   producer (this thread)  : push() -> raw queue
//   GMC thread              : raw queue -> GMC/OpenCL -> model-order RGB queue
//   NPU worker threads (1-2): RGB queue -> RGA mosaic -> upload -> NPU ->
//                              back-projection -> result queue
// The GMC engine is causal and stateful, so it lives on exactly one thread.
// Each NPU worker owns its own RKNN context AND its own RGA canvas, so two
// workers never touch the same buffer.  The board tolerates exactly two
// concurrent RKNN contexts (a third trips the 6 s job timeout and soft-resets
// the NPU), hence npu_workers is clamped to 2.
// ===========================================================================

// The GMC engine: compiled with its main() out, exactly like e2e_core.cpp does.
#define GMC_STREAM_OCL_NO_MAIN
#include "../cpp/gmc_stream_ocl.cpp"

#include "mosaic_core.h"
#include "mosaic_pipeline.h"

#include <cerrno>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// One raw frame waiting for the GMC engine.
//
// `gray` is an OWNING cv::Mat, not a view over a caller buffer.  That is not
// a style choice, it is a correctness requirement: the GMC engine stores the
// frame with a plain Mat ASSIGNMENT, and OpenCV's Mat::operator= is a SHALLOW
// copy (matrix.cpp:484 -- it bumps the refcount and copies the data POINTER;
// there is no clone()).  A Mat laid over someone else's buffer has no
// allocator block, so the engine's stored copy would dangle the moment the
// owner died -- and the fit would then run against freed memory.
// An owning Mat behaves exactly like the cv::imread() Mat the reference
// implementation feeds the engine: the shallow copies keep the pixels alive
// through the refcount, with no extra copy per frame.
struct InItem {
  int64_t id = 0;
  cv::Mat gray;
};

// One model-order RGB frame waiting for an NPU worker.
struct WorkItem {
  int64_t id = 0;
  std::vector<uint8_t> rgb;
};

// One finished frame.
struct ResultItem {
  int64_t id = 0;
  std::vector<mosaic::Comp> comps;
};

}  // namespace

// ---------------------------------------------------------------------------
// The implementation behind the opaque handle.
// ---------------------------------------------------------------------------
struct MosaicPipeline {
  MosaicPipelineConfig cfg{};
  std::string last_error;

  // ---- GMC engine ---------------------------------------------------------
  // gpu MUST be declared before pipe: the pipeline holds a raw pointer to it
  // and members are destroyed in reverse declaration order (pipe first).
  std::unique_ptr<FusedGpu> gpu;
  OnlineFeaturePipeline pipe;

  MosaicPipeline(const MosaicPipelineConfig& c, const std::string& kernel_dir, int slots)
      : cfg(c),
        gpu(new FusedGpu(kernel_dir, false, false, slots)),
        pipe(c.gmc_window, c.gmc_stride_step, c.gmc_anchor_step, c.gmc_downscale,
             /*rng_seed_per_fit=*/false, /*nogmc=*/false, gpu.get(), OnlineFeaturePipeline::kGpu,
             /*pyr_cache=*/c.gmc_pyr_cache != 0, c.gmc_fit_corners, c.gmc_fit_win, c.gmc_fit_level,
             c.gmc_fit_iters, /*cache_parts=*/7, /*threads=*/1) {}

  int64_t next_id = 0;

  // Rolling MD5 over the PRE-ENCODE channel planes -- the same object and
  // the same recipe as the reference implementation's --md5-out.
  gmcpp::MD5 seq_md5;
  // The hex string must OWN the bytes: returning hex().c_str() would hand
  // out a pointer into a temporary std::string that dies with the statement.
  std::string seq_md5_hex;
  FILE* md5_out = nullptr;
  std::string md5_path;
  // Optional feature dump (see mosaic_pipeline_set_feature_dump_out).
  std::string dump_dir;
  // Optional transform-chain dump (see mosaic_pipeline_set_mats_dump_out).
  FILE* mats_out = nullptr;
  std::string mats_path;

  // ---- queues -------------------------------------------------------------
  std::deque<InItem> in_q;
  std::deque<WorkItem> mos_q;
  std::deque<ResultItem> out_q;

  std::mutex m_in, m_mos, m_out;
  std::condition_variable cv_in, cv_mos, cv_out;
  // A dedicated CV for the RGB queue's backpressure: sharing one with the
  // consumer direction would let a consumer's notify_all wake a producer
  // that is waiting on a DIFFERENT predicate (a lost wakeup).
  std::condition_variable m_mos_cv_out;

  bool eof = false;
  bool abort = false;
  // Set by flush()/destroy() once EVERY worker thread has joined.  Without
  // it, collect() cannot tell "no result yet" from "no result will ever
  // come" and blocks forever after a flush -- the classic lost-wakeup.
  bool flushed = false;

  // Backpressure.  Without a bound the producer would push 11,810 raw frames
  // into the queue before the GMC thread could drain them: 11810 * 320 KB =
  // 3.8 GB of host memory, plus the same again in the model-order RGB queue.
  // A bound turns the queue into a flow-control point: push() blocks until
  // the GMC thread has room, which is exactly the pacing the embedded team
  // wants when the camera is the producer.
  static constexpr size_t kMaxInQueue = 16;    // 16 * 320 KB  =  5 MB
  static constexpr size_t kMaxMosQueue = 16;   // 16 * 960 KB  = 15 MB

  int32_t width = 0;
  int32_t height = 0;

  // ---- threads ------------------------------------------------------------
  std::thread gmc_thread;
  std::vector<std::thread> workers;

  // ---- timing -------------------------------------------------------------
  MosaicPipelineTiming timing{};
  std::mutex m_time;

  // A FATAL error: the pipeline cannot continue (the NPU or the worker is
  // gone).  Aborts every thread.
  void set_error(const std::string& e) {
    std::lock_guard<std::mutex> lk(m_out);
    if (last_error.empty()) last_error = e;
    abort = true;
  }

  // A PER-FRAME error: this frame is rejected, the stream keeps running.
  // A camera stream must not die because one frame was malformed -- and a
  // caller needs the reason retrievable via last_error() either way.
  void note_error(const std::string& e) {
    std::lock_guard<std::mutex> lk(m_out);
    if (last_error.empty()) last_error = e;
  }

  // -------------------------------------------------------------------------
  // GMC thread: raw gray -> GMC (Mali OpenCL fused tail) -> model-order RGB.
  // -------------------------------------------------------------------------
  void gmc_loop() {
    int seq_frame = 0;
    for (;;) {
      InItem item;
      {
        std::unique_lock<std::mutex> lk(m_in);
        cv_in.wait(lk, [this] { return !in_q.empty() || eof || abort; });
        if (abort) break;
        if (in_q.empty()) {
          if (eof) break;
          continue;
        }
        item = std::move(in_q.front());
        in_q.pop_front();
      }
      // A slot freed in the raw queue: let a blocked push() proceed.
      cv_in.notify_one();

      // item.gray OWNS its pixels (see InItem), so the engine's shallow copy
      // of it stays valid for as long as the engine keeps the frame -- the
      // same lifetime contract the reference implementation gets from
      // cv::imread()'s owning Mat.
      const Mat& frame = item.gray;
      const OnlineFeaturePipeline::Stats st0 = pipe.stats();
      pipe.set_seq_frame(seq_frame++);
      // The transform chain is only collected when it is going to be dumped:
      // collecting it does not change the fit (the reference implementation
      // proves that -- it fills the same vector from the same map).
      std::vector<Matx23f> mats;
      const Mat out = pipe.push(frame, mats_out ? &mats : nullptr);
      const OnlineFeaturePipeline::Stats st1 = pipe.stats();
      if (mats_out && !mats.empty()) {
        // Same bytes as the reference's --mats-all: 21 lags x 6 floats,
        // ascending lag order.
        for (const Matx23f& m : mats) {
          for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
              const float v = m(i, j);
              std::fwrite(&v, sizeof(float), 1, mats_out);
            }
          }
        }
        std::fflush(mats_out);
      }

      {
        gmcpp::MD5 f;
        f.update(reinterpret_cast<const uint8_t*>(out.data),
                 static_cast<size_t>(out.total()));
        const std::string fh = f.hex();
        seq_md5.update(fh);
        if (md5_out) {
          char line[256];
          std::snprintf(line, sizeof(line), "%s\t%06lld\t%s\n", cfg.sequence,
                        (long long)item.id, fh.c_str());
          std::fputs(line, md5_out);
          std::fflush(md5_out);
        }
      }
      {
        std::lock_guard<std::mutex> lk(m_time);
        timing.gmc_fit_ms += st1.t_fit - st0.t_fit;
        timing.gmc_warp_ms += st1.t_warp - st0.t_warp;
        timing.gmc_median_ms += st1.t_median - st0.t_median;
        timing.gmc_total_ms += st1.t_total - st0.t_total;
      }

      if (!dump_dir.empty()) {
        // SAME array and SAME cv::imwrite call as the reference's --out-dir
        // path (gmc_stream_ocl.cpp), so the two sides are directly comparable
        // pixel by pixel.  The MD5 stream cannot answer "how far apart": one
        // flipped bit changes the whole hash, while the real question is
        // whether the channels are within a grey level or two.
        Mat hwc(height, width, CV_8UC3);
        const uint8_t* c0 = out.ptr<uint8_t>(0);
        const uint8_t* c1 = out.ptr<uint8_t>(1);
        const uint8_t* c2 = out.ptr<uint8_t>(2);
        for (int y = 0; y < height; ++y) {
          uint8_t* row = hwc.ptr<uint8_t>(y);
          for (int x = 0; x < width; ++x) {
            const int p = y * width + x;
            row[x * 3 + 0] = c0[p];
            row[x * 3 + 1] = c1[p];
            row[x * 3 + 2] = c2[p];
          }
        }
        char nm[640];
        std::snprintf(nm, sizeof(nm), "%s/%s_%06lld.bmp", dump_dir.c_str(), cfg.sequence,
                      (long long)item.id);
        cv::imwrite(nm, hwc);
      }

      // PLANE-MAJOR -> model order [residual, diff, I_t].
      // out is a 3 x (H*W) CV_8UC1 Mat: row 0 = I_t, row 1 = diff,
      // row 2 = residual.
      const size_t plane = static_cast<size_t>(height) * static_cast<size_t>(width);
      WorkItem wi;
      wi.id = item.id;
      wi.rgb.resize(plane * 3);
      const uint8_t* c0 = out.ptr<uint8_t>(0);
      const uint8_t* c1 = out.ptr<uint8_t>(1);
      const uint8_t* c2 = out.ptr<uint8_t>(2);
      for (size_t p = 0; p < plane; ++p) {
        wi.rgb[p * 3 + 0] = c2[p];  // R = residual
        wi.rgb[p * 3 + 1] = c1[p];  // G = diff
        wi.rgb[p * 3 + 2] = c0[p];  // B = I_t
      }

      {
        // Backpressure on the RGB queue: do not outrun the NPU workers.
        std::unique_lock<std::mutex> lk(m_mos);
        m_mos_cv_out.wait(lk, [this] {
          return mos_q.size() < kMaxMosQueue || eof || abort;
        });
        if (abort) break;
        mos_q.push_back(std::move(wi));
      }
      cv_mos.notify_one();
    }
    // Wake the workers so they can exit.
    {
      std::lock_guard<std::mutex> lk(m_mos);
      eof = true;
    }
    cv_mos.notify_all();
  }

  // -------------------------------------------------------------------------
  // NPU worker: model-order RGB -> RGA mosaic -> upload -> NPU -> detections.
  // -------------------------------------------------------------------------
  void worker_loop() {
    mosaic::Session sess;
    if (!mosaic::InitSession(&sess, cfg.model_path,
                             static_cast<mosaic::CanvasMode>(cfg.canvas_mode), {}, false,
                             static_cast<mosaic::InputMode>(cfg.input_mode))) {
      set_error("InitSession failed (NPU worker)");
      std::lock_guard<std::mutex> lk(m_mos);
      abort = true;
      cv_mos.notify_all();
      return;
    }
    // Detection thresholds: 0 means "keep the frozen default".
    if (cfg.region_threshold > 0.0) sess.region_threshold = cfg.region_threshold;
    if (cfg.min_area > 0) sess.min_area = cfg.min_area;
    if (cfg.max_detections > 0) sess.max_detections = cfg.max_detections;
    mosaic::FrameOut fo;
    for (;;) {
      WorkItem item;
      {
        std::unique_lock<std::mutex> lk(m_mos);
        cv_mos.wait(lk, [this] { return !mos_q.empty() || eof || abort; });
        if (abort) break;
        if (mos_q.empty()) {
          if (eof) break;
          continue;
        }
        item = std::move(mos_q.front());
        mos_q.pop_front();
      }
      // A slot freed in the RGB queue: let the GMC thread proceed.
      m_mos_cv_out.notify_one();

      if (!mosaic::ProcessFrame(&sess, item.rgb.data(), width, height, &fo)) {
        set_error("ProcessFrame failed (NPU worker)");
        break;
      }
      {
        std::lock_guard<std::mutex> lk(m_time);
        timing.rga_import_ms += fo.rga_import;
        timing.rga_improcess_ms += fo.rga_improcess;
        timing.upload_ms += fo.upload;
        timing.run_ms += fo.run;
        timing.outputs_ms += fo.outputs;
        timing.backproj_ms += fo.backproj;
        timing.frames += 1;
      }

      ResultItem ri;
      ri.id = item.id;
      ri.comps = fo.comps;
      {
        std::lock_guard<std::mutex> lk(m_out);
        out_q.push_back(std::move(ri));
      }
      cv_out.notify_one();
    }
    mosaic::ReleaseSession(&sess);
  }
};

// ===========================================================================
// C ABI
// ===========================================================================

extern "C" {

void mosaic_pipeline_config_defaults(MosaicPipelineConfig* cfg) {
  if (!cfg) return;
  cfg->version = MOSAIC_API_VERSION;
  cfg->model_path = nullptr;
  cfg->kernel_dir = "/root/kernels";
  cfg->sequence = nullptr;
  cfg->gmc_window = 21;
  cfg->gmc_stride_step = 2;
  cfg->gmc_anchor_step = 10;
  cfg->gmc_downscale = 2;
  cfg->gmc_fit_corners = 600;
  cfg->gmc_fit_win = 15;
  cfg->gmc_fit_level = 2;
  cfg->gmc_fit_iters = 20;
  // EVERY field must be set here.  A caller that writes
  //   MosaicPipelineConfig cfg;            /* uninitialised stack */
  //   mosaic_pipeline_config_defaults(&cfg);
  // would otherwise leave the fields below as stack garbage -- and a garbage
  // fit gate silently degrades the transforms (that is exactly how the
  // 0.25-gate build destroyed the GMC features once already).
  cfg->gmc_gate_shift = 1000;  // 1.0 frame width == the C++ reference default
  cfg->gmc_gate_min = 500;     // 0.5
  cfg->gmc_gate_max = 2000;    // 2.0
  // Bit-identical to the uncached path over the 11,810-frame val3 track
  // (identical detection TSV, md5 0bf598a1d890040bcfa8d157dc69c032) and
  // 16% cheaper on the GMC fit.  Set 0 for the reference implementation's
  // own default.
  cfg->gmc_pyr_cache = 1;
  cfg->gmc_prealloc_pad = 0;
  cfg->canvas_mode = MOSAIC_CANVAS_MOSAIC;
  cfg->input_mode = MOSAIC_INPUT_SET;
  cfg->npu_workers = 2;
  cfg->region_threshold = 0;
  cfg->min_area = 0;
  cfg->max_detections = 0;
  cfg->cpu_affinity = nullptr;
}

MosaicPipeline* mosaic_pipeline_create(const MosaicPipelineConfig* cfg, char* errbuf,
                                        size_t errlen) {
  auto fail = [&](const std::string& why) -> MosaicPipeline* {
    if (errbuf && errlen) {
      snprintf(errbuf, errlen, "%s", why.c_str());
    }
    return nullptr;
  };
  if (!cfg) return fail("cfg is NULL");
  if (cfg->version != MOSAIC_API_VERSION)
    return fail("cfg->version mismatch (expected MOSAIC_API_VERSION)");
  if (!cfg->model_path || !*cfg->model_path) return fail("cfg->model_path is required");

  // The frozen GMC knobs are NOT silently overridden: a caller that asks for
  // something else gets it, and the deviation is reported.
  const std::string kernel_dir =
      cfg->kernel_dir && *cfg->kernel_dir ? cfg->kernel_dir : "/root/kernels";
  // ---- sanitize the GMC knobs ---------------------------------------------
  // A caller that forgot mosaic_pipeline_config_defaults() hands us stack
  // garbage in these fields.  A garbage fit gate is not a crash: it silently
  // degrades the transforms (a 0.25 gate rejects real FPV motion, the
  // transform falls back to identity and the GMC features stop being
  // registered -- that is exactly the failure mode that once cost this
  // pipeline 0.07 F1).  So out-of-range values are clamped back to the
  // reference defaults and reported, never silently used.
  struct Knob {
    const char* name;
    int32_t* v;
    int32_t lo;
    int32_t hi;
    int32_t dflt;
  };
  MosaicPipelineConfig cfg_s = *cfg;
  Knob knobs[] = {
      {"gmc_window", &cfg_s.gmc_window, 3, 64, 21},
      {"gmc_stride_step", &cfg_s.gmc_stride_step, 1, 8, 2},
      {"gmc_anchor_step", &cfg_s.gmc_anchor_step, 2, 64, 10},
      {"gmc_downscale", &cfg_s.gmc_downscale, 1, 4, 2},
      {"gmc_fit_corners", &cfg_s.gmc_fit_corners, 10, 5000, 600},
      {"gmc_fit_win", &cfg_s.gmc_fit_win, 3, 99, 15},
      {"gmc_fit_level", &cfg_s.gmc_fit_level, 0, 8, 2},
      {"gmc_fit_iters", &cfg_s.gmc_fit_iters, 1, 200, 20},
      {"gmc_gate_shift", &cfg_s.gmc_gate_shift, 0, 10000, 1000},
      {"gmc_gate_min", &cfg_s.gmc_gate_min, 1, 10000, 500},
      {"gmc_gate_max", &cfg_s.gmc_gate_max, 1, 10000, 2000},
      {"npu_workers", &cfg_s.npu_workers, 1, 2, 2},
  };
  for (Knob& k : knobs) {
    if (*k.v < k.lo || *k.v > k.hi) {
      fprintf(stderr, "[WARN] %s = %d is out of range [%d, %d]; using the reference default %d\n",
              k.name, (int)*k.v, (int)k.lo, (int)k.hi, (int)k.dflt);
      *k.v = k.dflt;
    }
  }
  if (cfg_s.gmc_gate_min > cfg_s.gmc_gate_max) {
    fprintf(stderr, "[WARN] gmc_gate_min(%d) > gmc_gate_max(%d); using the reference defaults\n",
            (int)cfg_s.gmc_gate_min, (int)cfg_s.gmc_gate_max);
    cfg_s.gmc_gate_min = 500;
    cfg_s.gmc_gate_max = 2000;
  }
  cfg_s.gmc_pyr_cache = cfg_s.gmc_pyr_cache ? 1 : 0;
  if (cfg_s.gmc_prealloc_pad < 0) cfg_s.gmc_prealloc_pad = 0;
  const MosaicPipelineConfig* safe_cfg = &cfg_s;

  // OpenCL fused tail: the Mali GPU does the 21 inverse re-projections, the
  // 21-element median and the [Ch0,Ch1,Ch2] assembly.
  //
  // The slot count MUST be derived from the SANITIZED window/stride: an
  // out-of-range gmc_window used here would size the device ring from an
  // absurd number (the contract test caught exactly that: slots=20001 for a
  // window of 9999) and the allocation would happen before any clamp.
  const int slots = safe_cfg->gmc_window * safe_cfg->gmc_stride_step + 3;

  auto* p = new MosaicPipeline(*safe_cfg, kernel_dir, slots);

  p->pipe.set_profiling(false);
  p->pipe.set_seq_id(0);
  // The fit gate MUST keep the C++ reference default (1.0 frame width).
  // e2e_core.h carries 0.25, but that is a different harness; at 0.25 the
  // gate REJECTS real FPV motion, the transform degrades, and the GMC
  // features stop being registered to the raw frames.
  if (safe_cfg->gmc_gate_shift > 0) {
    p->pipe.set_fit_gate(/*max_shift_frac=*/safe_cfg->gmc_gate_shift / 1000.0,
                          /*min_scale=*/safe_cfg->gmc_gate_min / 1000.0,
                          /*max_scale=*/safe_cfg->gmc_gate_max / 1000.0);
  }
  p->pipe.gpu_prealloc(safe_cfg->gmc_prealloc_pad);
  p->pipe.gpu_pin_pad(false);
  p->pipe.set_corner_probe(true);
  // Deterministic: the output must not depend on the machine load.
  p->pipe.set_fit_deadline(0.0, false);

  // A garbage cpu_affinity POINTER would be a crash, not a silent degradation,
  // so it is validated here too: only a non-empty, printable string is used.
  if (safe_cfg->cpu_affinity && *safe_cfg->cpu_affinity) {
    mosaic::ApplyCpuAffinity(safe_cfg->cpu_affinity);
  }

  int workers = safe_cfg->npu_workers;  // already clamped to [1, 2] above

  p->timing.wall_ms = mosaic::now_ms();
  p->gmc_thread = std::thread([p] { p->gmc_loop(); });
  for (int i = 0; i < workers; i++) {
    p->workers.emplace_back([p] { p->worker_loop(); });
  }
  fprintf(stderr, "[i] mosaic_pipeline created: model=%s gmc(anchor_step=%d, window=%d) "
                  "canvas=%d input=%d workers=%d pyr_cache=%d gate_shift=%.3f\n",
          safe_cfg->model_path, safe_cfg->gmc_anchor_step, safe_cfg->gmc_window,
          safe_cfg->canvas_mode, safe_cfg->input_mode, workers, safe_cfg->gmc_pyr_cache,
          safe_cfg->gmc_gate_shift / 1000.0);
  return p;
}

const char* mosaic_pipeline_last_error(MosaicPipeline* p) {
  if (!p) return "pipeline is NULL";
  std::lock_guard<std::mutex> lk(p->m_out);
  return p->last_error.c_str();
}

int32_t mosaic_pipeline_effective_config(MosaicPipeline* p, MosaicPipelineConfig* out) {
  if (!p || !out) return -1;
  // p->cfg is the sanitized copy the constructor stored, i.e. what was
  // really applied (after the out-of-range clamping in create()).
  *out = p->cfg;
  return 0;
}

int64_t mosaic_pipeline_push(MosaicPipeline* p, const uint8_t* gray, int32_t w, int32_t h) {
  if (!p || !gray || w <= 0 || h <= 0) return -1;
  // The frame size is FIXED by the mosaic core (the RGA canvas and the
  // back-projection grid are built for exactly mosaic::kSrcW x mosaic::kSrcH).
  // Validate it HERE, on the first frame too: otherwise a stray wrong-sized
  // first frame would silently redefine the pipeline's geometry and every
  // later frame would be rejected with a much less obvious message.
  // (The contract test caught exactly this.)
  if (w != mosaic::kSrcW || h != mosaic::kSrcH) {
    char msg[192];
    snprintf(msg, sizeof(msg),
             "frame size %dx%d is not the expected %dx%d (the mosaic canvas and the "
             "back-projection grid are built for it)",
             (int)w, (int)h, (int)mosaic::kSrcW, (int)mosaic::kSrcH);
    // Per-frame error: reject THIS frame, keep the stream alive.
    p->note_error(msg);
    return -1;
  }
  if (p->width == 0) {
    p->width = w;
    p->height = h;
  } else if (p->width != w || p->height != h) {
    p->set_error("frame size changed mid-stream");
    return -1;
  }
  InItem item;
  {
    std::unique_lock<std::mutex> lk(p->m_in);
    // Backpressure: block until the GMC thread has drained a slot.
    p->cv_in.wait(lk, [p] { return p->in_q.size() < p->kMaxInQueue || p->eof || p->abort; });
    if (p->eof || p->abort) return -1;
    item.id = p->next_id++;
    // One copy, and it lands in an owning Mat (see InItem).
    item.gray = cv::Mat(h, w, CV_8UC1, const_cast<uint8_t*>(gray)).clone();
    p->in_q.push_back(std::move(item));
  }
  p->cv_in.notify_one();
  // The header promises the FRAME ID, not a generic 0.  The contract test
  // caught the implementation returning a constant 0: a caller that pushes a
  // frame that gets REJECTED (wrong size) and then pushes the next one would
  // see two identical ids and could not tell which frame it is holding.
  return item.id;
}

int32_t mosaic_pipeline_collect(MosaicPipeline* p, int64_t* frame_id, MosaicDet* out,
                                int32_t max_out) {
  if (!p) return -1;
  ResultItem ri;
  {
    std::unique_lock<std::mutex> lk(p->m_out);
    p->cv_out.wait(lk, [p] { return !p->out_q.empty() || p->abort || p->flushed; });
    if (p->out_q.empty()) return -1;  // aborted, or flushed and drained
    ri = std::move(p->out_q.front());
    p->out_q.pop_front();
  }
  if (frame_id) *frame_id = ri.id;
  const int32_t n = (int32_t)ri.comps.size();
  if (out && max_out > 0) {
    const int32_t ncopy = n < max_out ? n : max_out;
    for (int32_t i = 0; i < ncopy; i++) {
      const mosaic::Comp& c = ri.comps[i];
      out[i].tag = c.tag;
      out[i].cx = c.cx;
      out[i].cy = c.cy;
      out[i].w = c.w;
      out[i].h = c.h;
      out[i].area = c.area;
      out[i].peak = c.peak;
      out[i].sum = c.sum;
    }
    return ncopy;
  }
  return n;
}

int32_t mosaic_pipeline_flush(MosaicPipeline* p) {
  if (!p) return -1;
  {
    std::lock_guard<std::mutex> lk(p->m_in);
    p->eof = true;
  }
  p->cv_in.notify_all();
  p->m_mos_cv_out.notify_all();
  if (p->gmc_thread.joinable()) p->gmc_thread.join();
  p->cv_mos.notify_all();
  for (std::thread& t : p->workers) {
    if (t.joinable()) t.join();
  }
  {
    std::lock_guard<std::mutex> lk(p->m_time);
    p->timing.wall_ms = mosaic::now_ms() - p->timing.wall_ms;
  }
  {
    std::lock_guard<std::mutex> lk(p->m_out);
    p->flushed = true;
  }
  p->cv_out.notify_all();
  if (p->mats_out) { std::fclose(p->mats_out); p->mats_out = nullptr; }
  return 0;
}

void mosaic_pipeline_set_feature_md5_out(MosaicPipeline* p, const char* path) {
  if (!p) return;
  if (p->md5_out) { std::fclose(p->md5_out); p->md5_out = nullptr; }
  p->md5_path = path ? path : "";
  if (path && *path) {
    p->md5_out = std::fopen(path, "wb");
    if (!p->md5_out) {
      fprintf(stderr, "[FATAL] cannot open feature md5 file %s: %s\n", path, strerror(errno));
    }
  }
}

void mosaic_pipeline_set_feature_dump_out(MosaicPipeline* p, const char* dir) {
  if (!p) return;
  p->dump_dir = dir ? dir : "";
}

void mosaic_pipeline_set_mats_dump_out(MosaicPipeline* p, const char* path) {
  if (!p) return;
  if (p->mats_out) { std::fclose(p->mats_out); p->mats_out = nullptr; }
  p->mats_path = path ? path : "";
  if (path && *path) {
    p->mats_out = std::fopen(path, "wb");
    if (!p->mats_out) {
      fprintf(stderr, "[FATAL] cannot open mats dump file %s: %s\n", path, strerror(errno));
    }
  }
}

const char* mosaic_pipeline_feature_md5(MosaicPipeline* p) {
  if (!p) return "";
  p->seq_md5_hex = p->seq_md5.hex();
  return p->seq_md5_hex.c_str();
}

int32_t mosaic_pipeline_get_timing(MosaicPipeline* p, MosaicPipelineTiming* out) {
  if (!p || !out) return -1;
  std::lock_guard<std::mutex> lk(p->m_time);
  *out = p->timing;
  return 0;
}

void mosaic_pipeline_destroy(MosaicPipeline* p) {
  if (!p) return;
  {
    std::lock_guard<std::mutex> lk(p->m_in);
    p->eof = true;
  }
  p->cv_in.notify_all();
  p->m_mos_cv_out.notify_all();
  if (p->gmc_thread.joinable()) p->gmc_thread.join();
  p->cv_mos.notify_all();
  for (std::thread& t : p->workers) {
    if (t.joinable()) t.join();
  }
  {
    std::lock_guard<std::mutex> lk(p->m_out);
    p->flushed = true;
  }
  p->cv_out.notify_all();
  if (p->mats_out) { std::fclose(p->mats_out); p->mats_out = nullptr; }
  delete p;
}

}  // extern "C"
