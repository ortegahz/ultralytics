// ===========================================================================
// rknn_mosaic_e2e.cpp -- the end-to-end demo that DRIVES THE LIBRARY.
//
//   raw infrared sequence (640x512 gray JPGs)
//     -> mosaic_pipeline_push()   (the library does GMC/OpenCL + RGA mosaic
//                                   + RKNN fp16 + back-projection)
//     -> mosaic_pipeline_collect() (detections in native frame pixels)
//     -> <seq>_mosaic.tsv
//
// This program is a pure consumer of the public C ABI (mosaic_pipeline.h).
// It links against libmosaic_pipeline.a and calls nothing else from the
// project, so it exercises exactly the interface the embedded team gets.
//
// The TSV format is byte-compatible with the BMP-driven binary
// (rknn_mosaic_heatmap), so the SAME scorer
// (manu/pipeline/npu/ptq/35_mosaic_board_eval.py) reads both.
//
// Usage:
//   rknn_mosaic_e2e --raw-root DIR --sequence NAME --model M.rknn --out DIR
//                   [--kernel-dir /root/kernels] [--anchor-step 2]
//                   [--npu-workers 2] [--limit N] [--start N]
//                   [--cpu-affinity 4-7] [--threads 1] [--timing FILE]
//                   [--gmc-ring-slack N]   the GMC lookahead budget (0 = auto)
// ===========================================================================
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "mosaic_pipeline.h"

namespace {

// The streaming consumer state.
//
// push() NEVER blocks -- the camera must not be stalled by a pipeline that
// cannot keep up -- so the consumer has to pace itself.  The results come
// back through the CALLBACK (in completion order); the consumer keeps at
// most kInFlight frames outstanding (pushed but not yet delivered) and waits
// on the condition variable otherwise.  That wait IS the pipeline's own
// pace, so the wall clock the library measures contains no demo overhead
// beyond the pipeline's.
struct Consumer {
  std::mutex m;
  std::condition_variable cv;
  std::map<int64_t, std::vector<MosaicDet>> rows;
  // ATOMIC on purpose: the watchdog has to be able to read the progress
  // WITHOUT taking the lock, because it is called from inside the wait
  // (which holds the lock).
  std::atomic<int64_t> delivered{0};
  // THE END-TO-END LATENCY: the time from the camera's push of a frame to the
  // delivery of THAT frame's result.  This is the number a closed loop cares
  // about: it is how old the target is when the detection arrives.
  //
  // WHY IT IS MEASURED HERE AND NOT IN THE LIBRARY: the clock starts where
  // the camera hands the frame over (the only place the camera's own time is
  // known) and stops where the result is handed back.  The library's own
  // segments (ingest waits, gmc_fit, rknn_run) are already in the timing
  // file; this adds the ONE number that cannot be derived from them: the
  // latency the pipeline actually imposed on a live stream.
  std::map<int64_t, std::chrono::steady_clock::time_point> push_at;
  std::vector<double> lat_ms;
};

constexpr int kInFlight = 4;  // frames pushed but not yet delivered

void OnResult(void* user, int64_t frame_id, const MosaicDet* dets, int32_t n) {
  Consumer* c = (Consumer*)user;
  {
    std::lock_guard<std::mutex> lk(c->m);
    c->rows[frame_id].assign(dets, dets + n);
    c->delivered.fetch_add(1);
    // Stop the latency clock for THIS frame.  A frame with no entry was
    // delivered before the producer could record its push (impossible in
    // practice -- the pipeline needs ~100 ms to answer a push) and is
    // simply not counted.
    auto it = c->push_at.find(frame_id);
    if (it != c->push_at.end()) {
      c->lat_ms.push_back(std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - it->second)
                              .count());
      c->push_at.erase(it);
    }
  }
  c->cv.notify_all();
}

struct Options {
  std::string raw_root;
  std::string sequence;
  std::string model;
  std::string out;
  std::string kernel_dir = "/root/kernels";
  std::string timing;
  std::string cpu_affinity;
  std::string gmc_affinity;
  std::string npu_affinity;
  // The GMC engines' core GROUPS: ';'-separated, one group per engine
  // ("4-5;6-7").  A group is the set of cores an engine's thread AND its
  // anchor workers are spread over.
  std::string gmc_affinity_group;
  std::string feature_dump;
  std::string mats_dump;
  std::string md5_golden;
  int anchor_step = 2;
  int npu_workers = 2;
  // The service sizes.  -1 = "not specified": the library's own default.
  int gmc_workers = -1;
  int post_workers = -1;
  int queue_depth = -1;
  int input_queue = -1;
  int jitter_depth = -1;
  // The GMC service's LOOKAHEAD BUDGET, in frames.  0 = the library's own
  // default (AUTO = max(64, 32*gmc_workers)).  It is a THROUGHPUT knob: it
  // changes how far the ingest may run ahead of the engines, never which
  // pixels a fit reads, so the detection numbers do not change.
  int gmc_ring_slack = 0;
  int gmc_deadline_ms = -1;
  // The Shi-Tomasi corner budget.  -1 = "not specified": the library's own
  // default (600) applies.  LOWERING IT CHANGES THE NUMBERS (different
  // corners -> different transforms -> different features -> different
  // detections), so it is a measured DEVIATION, not a free speed knob.
  int gmc_fit_corners = -1;
  // The GMC FIT knobs.  -1 = "not specified": the library's own default
  // applies (win 15 / level 2 / iters 20 / downscale 2, set by
  // mosaic_pipeline_config_defaults()).  These CHANGE THE NUMBERS (a
  // different LK window / pyramid level / iteration count / fit resolution
  // gives different transforms, hence different features and detections), so
  // they are measured DEVIATIONS, not free speed knobs.  They exist here so a
  // parameter sweep can drive them through the REAL service pipeline instead
  // of the offline reference.
  int gmc_fit_win = -1;      // -> cfg.gmc_fit_win
  int gmc_fit_level = -1;    // -> cfg.gmc_fit_level
  int gmc_fit_iters = -1;    // -> cfg.gmc_fit_iters
  int gmc_downscale = -1;    // -> cfg.gmc_downscale
  // The RANSAC sampling cap for the affine solve.  0 = OFF: the library's own
  // default (the reference call's default arguments) applies.  A positive value
  // caps the sampling iterations: ENABLING IT CHANGES THE NUMBERS, so it is a
  // measured DEVIATION and must be reported with the results.
  int gmc_ransac_max_iters = 0;   // -> cfg.gmc_ransac_max_iters
  // THE PHYSICAL-FRAME-NUMBER ANCHOR of the GMC history.  -1 = "not
  // specified": the library's own default (0 = the consumption counter, i.e.
  // the certified behaviour) applies.  1 = the engine anchors on the camera's
  // physical frame number, so a frame lost at the camera queue becomes a
  // DETECTED hole (reported per frame and counted in the timing file) instead
  // of a silently stretched history window.
  int gmc_phys_anchor = -1;       // -> cfg.gmc_phys_anchor
  // THE MINIMUM GAP BETWEEN TWO GMC GROUPS, in frames.  1 = no limit: the
  // engine grabs the NEWEST frame as soon as it is free (the adaptive
  // default of the OPPORTUNISTIC scheduler).  >1 = an artificial rate limit:
  // the engine waits until the newest frame is at least N frames past the one
  // it computed last.
  //
  // RENAMED from gmc_compute_stride, whose meaning was the opposite (a fixed
  // "every Nth frame" cadence).  The OLD FLAG IS STILL ACCEPTED as an alias
  // and is reported as such, so an old command line keeps working.
  int gmc_min_stride = -1;          // -> cfg.gmc_min_stride
  int gmc_compute_stride = -1;      // DEPRECATED alias of gmc_min_stride
  // WHAT THE CAMERA QUEUE DOES ON OVERFLOW.  -1 = "not specified": the
  // library's own default (0 = evict the OLDEST, i.e. keep the freshest
  // data) applies.  1 = reject the NEWEST (keep the queued frames).
  int gmc_input_drop_policy = -1; // -> cfg.gmc_input_drop_policy
  int gmc_feature_fingerprint = -1;
  // The anchor LK stage's worker count per GMC engine, the engine's own
  // thread INCLUDED.  -1 = "not specified": the library's default (1 == the
  // reference's serial anchor loop) applies.  5 = one worker per anchor.
  int gmc_anchor_threads = -1;
  // -1 = "not specified": the library's own default (1) applies.
  //  0 = off (the reference implementation's default), 1 = on.
  int pyr_cache = -1;
  int prealloc_pad = 0;
  int limit = 0;
  int start = 0;
  // Simulate a camera with a FIXED frame rate: push one frame every
  // --pace-ms milliseconds, regardless of whether the pipeline keeps up.
  // 0 = off (the default: the producer is paced by the in-flight window, so
  // the camera queue never overflows and the measured wall clock IS the
  // pipeline's own throughput).  40 = a 25 fps camera.
  int pace_ms = 0;
  // The WATCHDOG: how long the demo waits for the pipeline to make PROGRESS
  // before it gives up, prints the counters and exits non-zero.  0 = off.
  // The default is 30 s: a healthy run delivers a result every ~70 ms, so
  // 30 s of silence is a wedged pipeline, not a slow one.
  int watchdog_ms = 30000;
  // The NPU core mask (rknn_core_mask) for every NPU context.
  //  -1 = "not specified": the library's default (0 = do NOT call
  //       rknn_set_core_mask) applies.
  //   1 = CORE_0, 2 = CORE_1, 4 = CORE_2, 3 = CORE_0_1, 6 = CORE_1_2,
  //   5 = CORE_0_2, 7 = CORE_0_1_2.
  // A non-zero mask is a MEASURED DEVIATION: it may change the throughput
  // and the numerics (a different core is a different execution path).
  int npu_core_mask = -1;
  // FAULT INJECTION (default OFF): after this many frames, make the pipeline
  // fail on purpose (the library's own set_error path).  It proves that a
  // FATAL error ABORTS the pipeline and that the demo REPORTS it and exits
  // non-zero, instead of wedging silently.  -1 = off.
  int fault_inject_at = -1;
  // The UPLOAD path.  -1 = "not specified": the library's default (set) applies.
  //   set        uint8 NHWC through rknn_inputs_set; the runtime does the
  //              uint8->fp16 widening (12.8 ms/frame on the board).  THE
  //              CERTIFIED PATH -- the baseline was measured this way.
  //   setf16     WE widen with integer NEON and hand the runtime a FLOAT16
  //              buffer.  Same values, but the runtime then copies 2.45 MB
  //              instead of 0.98 MB, which MEASURED SLOWER (41 vs 12.8 ms).
  //   iomem_nhwc the input tensor is bound to host memory and WE write it:
  //              the widening is ~0.8 ms and the runtime's staging happens
  //              inside rknn_run, off the CPU's critical path.
  //   iomem_nchw  as above but NCHW.  KNOWN BROKEN for this model (the NCHW
  //              padding path is wrong): do not use it.
  std::string input_mode;
  // ---- v10: THE THREE-STAGE DECOUPLED PIPELINE ----------------------------
  // 0 = the FUSED worker (Stage 1 + Stage 2 in one thread): the certified arm,
  // and the default, so the historical runs stay comparable.
  // 1 = the three-stage pipeline: Stage 1 (pre-process) -> Buffer A ->
  // Stage 2 (NPU) -> Buffer B -> Stage 3 (post-process).
  int decoupled_stages = 0;
  // Buffer A: the ring of NPU-ready slots between Stage 1 and Stage 2.
  int buffer_a_depth = 3;
  // Buffer B: the ring of raw heatmaps between Stage 2 and Stage 3.
  int buffer_b_depth = 3;
  // The Stage 1 thread count.  0 = AUTO (= npu_workers).  In the decoupled mode
  // it is FORCED to npu_workers, because the stages are paired 1:1.
  int pre_threads = 0;
  // Where Buffer A's slots live: 0 = DMA (rknn_create_mem), 1 = host memory.
  std::string buf_a_mode = "dma";
  // The core list for Stage 1 AND Stage 3.  Empty = the library's own default
  // ("6-7").
  std::string pre_affinity = "6-7";
  // The core list for Stage 2.  Empty = follow pre_affinity.
  std::string npu_stage_affinity;
};

void PrintUsage(const char* argv0) {
  fprintf(stderr,
          "usage: %s --raw-root DIR --sequence NAME --model M.rknn --out DIR\n"
          "          [--kernel-dir /root/kernels] [--anchor-step 2] [--npu-workers 2]\n"
          "          [--limit N] [--start N] [--cpu-affinity 4-7] [--timing FILE]\n"
          "          [--gmc-affinity 4,5]   pin the GMC engines to these cores (one per\n"
          "                                engine); keeps a 60 ms fit from being preempted\n"
          "          [--gmc-affinity-group 4-5;6-7]\n"
          "                                pin the GMC engines to these core GROUPS (one per\n"
          "                                engine).  A group is the set of cores an engine's\n"
          "                                thread AND its anchor workers are spread over --\n"
          "                                needed for --gmc-anchor-threads to buy anything,\n"
          "                                because a single-core engine shares that core with\n"
          "                                its own workers.\n"
          "          [--gmc-anchor-threads N]\n"
          "                                the anchor LK stage's workers per GMC engine, the\n"
          "                                engine's own thread INCLUDED (default 1 == the\n"
          "                                reference's serial anchor loop; 5 == one worker\n"
          "                                per anchor).  The RANSAC stage still runs\n"
          "                                sequentially in the anchor order, so the numbers\n"
          "                                stay BIT-IDENTICAL; only the scheduling changes.\n"
          "          [--npu-affinity 6,7]   pin the NPU/POST services to these cores\n"
          "          [--pyr-cache 0|1]      reuse the GMC pyramid/corners across fits\n"
          "                                (bit-identical; default = the library's 1)\n"
          "          [--prealloc-pad N]     reserve the OpenCL device images once\n"
          "          [--feature-dump DIR]   write the GMC feature planes as BMPs\n"
          "                                (<seq>_%%06d.bmp, [Ch0,Ch1,Ch2])\n"
          "          [--mats-dump FILE]     raw float32 stream of the transform\n"
          "                                chain (21 lags x 6 floats per frame)\n"
          "          [--feature-md5-golden F]  compare the produced per-frame feature\n"
          "                                fingerprints against a frozen golden file\n"
          "                                and exit 1 on any mismatch (self-check)\n"
          "          [--gmc-fit-corners N]   Shi-Tomasi corner budget (default 600).\n"
          "                                LOWERING IT CHANGES THE NUMBERS: it is a\n"
          "                                measured deviation, not a free speed knob.\n"
          "          [--gmc-fit-win N]       the LK fit window (default 15).\n"
          "          [--gmc-fit-level N]     the LK pyramid level (default 2).\n"
          "          [--gmc-fit-iters N]     the LK iteration count (default 20).\n"
          "          [--gmc-downscale N]     the GMC fit resolution divisor\n"
          "                                (default 2; the library accepts 1..4).\n"
          "                                These four CHANGE THE NUMBERS (different\n"
          "                                transforms -> different features -> different\n"
          "                                detections), so they are measured deviations,\n"
          "                                not free speed knobs.  Without them the\n"
          "                                library's own defaults apply.\n"
          "          [--gmc-ransac-max-iters N]  cap the RANSAC sampling iterations of\n"
          "                                the affine solve (default 0 = the library's own\n"
          "                                default).  ENABLING IT CHANGES THE NUMBERS, so\n"
          "                                it must be reported as a deviation with the\n"
          "                                results.\n"
          "          [--gmc-feature-fingerprint 0|1]  per-frame feature MD5;\n"
          "                                0 saves ~6 ms of CPU per frame (audit only)\n"
          "          [--gmc-phys-anchor 0|1]  anchor the GMC history on the camera's\n"
          "                                PHYSICAL frame number (1) instead of the\n"
          "                                consumption counter (0, the default =\n"
          "                                the certified behaviour).  With 1 a frame\n"
          "                                lost at the camera queue becomes a\n"
          "                                DETECTED hole (counted in\n"
          "                                gmc_history_gaps, printed per frame) and\n"
          "                                not a silently stretched history window.\n"
          "          [--gmc-input-drop-policy 0|1]  what the camera queue does when it\n"
          "                                overflows: 0 = evict the OLDEST (the\n"
          "                                default: keep the freshest data), 1 =\n"
          "                                reject the NEWEST (keep the queued\n"
          "                                frames, at the price of an unbounded\n"
          "                                latency under a sustained overload).\n"
          "          [--gmc-min-stride N]    the MINIMUM gap between two GMC groups,\n"
          "                                in frames.  1 (the default) = no limit:\n"
          "                                the engine grabs the NEWEST frame as\n"
          "                                soon as it is free (the OPPORTUNISTIC\n"
          "                                scheduler) and the frames it could not\n"
          "                                keep up with are skipped -- they are\n"
          "                                still seeded, so they are still history.\n"
          "                                >1 = an artificial rate limit.\n"
          "          [--gmc-compute-stride N] DEPRECATED alias of --gmc-min-stride\n"
          "                                (the old meaning was a fixed cadence);\n"
          "                                it is reported as such when used.\n"
          "          [--pace-ms N]          push one frame every N ms, i.e. simulate a\n"
          "                                camera of 1000/N fps that does NOT wait for\n"
          "                                the pipeline (40 = the 25 fps source)\n"
          "          [--watchdog-ms N]      give up if the pipeline makes NO progress for\n"
          "                                N ms (default 30000; 0 = off).  On a timeout\n"
          "                                the counters are printed and the run exits\n"
          "                                non-zero: a wedged pipeline must be REPORTED,\n"
          "                                not waited on for ever.\n"
          "          [--npu-core-mask M]     the NPU core mask for every RKNN context:\n"
          "                                1=CORE_0 2=CORE_1 4=CORE_2 3=CORE_0_1\n"
          "                                6=CORE_1_2 5=CORE_0_2 7=CORE_0_1_2.\n"
          "                                DEFAULT 0 = do NOT call rknn_set_core_mask\n"
          "                                (the shipped behaviour).  A non-zero mask is\n"
          "                                a MEASURED DEVIATION: it can change the\n"
          "                                throughput and the numerics.\n"
          "          [--fault-inject-at N]  fail the pipeline on purpose after N frames\n"
          "                                (TEST HOOK).  It proves that a FATAL error\n"
          "                                ABORTS the pipeline and that the demo\n"
          "                                reports it and exits non-zero, instead of\n"
          "                                wedging silently.  Default: off.\n"
          "          [--input-mode M]       the upload path: set (default, certified),\n"
          "                                setf16, iomem_nhwc, iomem_nchw (the last one\n"
          "                                is KNOWN BROKEN for this model).\n"
          "          [--gmc-ring-slack N]    the GMC service's LOOKAHEAD BUDGET, in\n"
          "                                frames (default 0 = AUTO = the library's\n"
          "                                max(64, 32*gmc_workers)).  THROUGHPUT\n"
          "                                ONLY: it changes how far the ingest may\n"
          "                                run ahead of the engines, never which\n"
          "                                pixels a fit reads, so the detections do\n"
          "                                not change.  A budget that is too small\n"
          "                                lets the engines starve and the camera\n"
          "                                queue overflow (which breaks the GMC\n"
          "                                registration chain).\n"
          "          [--decoupled-stages 0|1]  0 (the DEFAULT) = the FUSED worker of\n"
          "                                v9: the RGA compose, the upload and the\n"
          "                                rknn_run all happen in ONE thread.  That is\n"
          "                                the certified arm, so the default run\n"
          "                                stays comparable with the history.\n"
          "                                1 = the THREE-STAGE pipeline:\n"
          "                                  Stage 1 (pre-process, own cores):\n"
          "                                    RGA compose + the u8->fp16 widening,\n"
          "                                    written STRAIGHT into a Buffer A slot\n"
          "                                  Buffer A: a bounded ring of NPU-ready\n"
          "                                    slots; on overflow the OLDEST ready\n"
          "                                    slot is reclaimed (never a wait)\n"
          "                                  Stage 2 (NPU, own cores): bind the slot\n"
          "                                    + rknn_run + outputs, no CPU work\n"
          "                                  Buffer B: a bounded ring of heatmaps\n"
          "                                  Stage 3 (post): back-projection + NMS\n"
          "          [--buffer-a-depth N]    Buffer A's depth in slots (1..64,\n"
          "                                default 3).  Each slot holds one model\n"
          "                                input (2,457,600 B), so 3 is ~7 MB;\n"
          "                                D=16 is ~78.7 MB and D=64 ~315 MB\n"
          "                                on the board (both NPU services).\n"
          "          [--buffer-b-depth N]    Buffer B's depth in heatmaps (1..8,\n"
          "                                default 3).  On overflow the OLDEST\n"
          "                                heatmap is dropped.\n"
          "          [--pre-threads N]       the Stage 1 thread count.  0 (the\n"
          "                                default) = AUTO = npu_workers.  In the\n"
          "                                decoupled mode the count is FORCED to\n"
          "                                npu_workers: the stages are paired 1:1,\n"
          "                                because every NPU service owns its own\n"
          "                                Buffer A and its own RKNN context.\n"
          "          [--buf-a-mode dma|host] where Buffer A's slots live.  dma (the\n"
          "                                default) = rknn_create_mem, contiguous\n"
          "                                DMA memory the runtime reads the input\n"
          "                                tensor from (zero-copy hand-off).\n"
          "                                host = ordinary memory, handed over\n"
          "                                with pass_through=1 (the A/B arm).\n"
          "          [--pre-affinity LIST]   pin Stage 1 AND Stage 3 to these\n"
          "                                cores (default \"6-7\").  The upload\n"
          "                                bucket is not a fixed cost: it was\n"
          "                                inflated by the CPU and memory-bandwidth\n"
          "                                contention with the GMC, so the split\n"
          "                                gives it a core group of its own.\n"
          "          [--npu-stage-affinity LIST]\n"
          "                                pin Stage 2 (the NPU) to these cores.\n"
          "                                Empty (the default) = follow\n"
          "                                --pre-affinity.\n"
          "\n"
          "  raw-root   DIR   directory holding <sequence>/frame_%%06d.jpg\n"
          "  sequence   NAME  sequence to run\n"
          "  out        DIR   output root (<seq>_mosaic.tsv is written here)\n"
          "  model      PATH  .rknn heatmap detector\n",
          argv0);
}

// The frame files of a sequence, sorted by their numeric frame index.
//
// BMP first: the board's cross-built OpenCV is WITH_JPEG=OFF / WITH_PNG=OFF,
// so a JPEG cannot be decoded there.  A JPG is still accepted for hosts whose
// OpenCV has the codec.
bool ListFrames(const std::string& dir, std::vector<std::string>* out) {
  DIR* d = opendir(dir.c_str());
  if (!d) {
    fprintf(stderr, "[FATAL] cannot open %s: %s\n", dir.c_str(), strerror(errno));
    return false;
  }
  std::map<long, std::string> by_index;
  struct dirent* e = nullptr;
  while ((e = readdir(d)) != nullptr) {
    const std::string name = e->d_name;
    // frame_<digits>.bmp / frame_<digits>.jpg
    if (name.rfind("frame_", 0) != 0) continue;
    const size_t dot = name.rfind('.');
    if (dot == std::string::npos || dot + 4 != name.size()) continue;
    const std::string ext = name.substr(dot);
    if (ext != ".bmp" && ext != ".jpg" && ext != ".png") continue;
    const std::string digits = name.substr(6, dot - 6);
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) continue;
    // Prefer BMP when both exist for the same index.
    const long idx = atol(digits.c_str());
    auto it = by_index.find(idx);
    if (it == by_index.end() || it->second.size() > name.size() ||
        (it->second.size() == name.size() && ext == ".bmp")) {
      by_index[idx] = dir + "/" + name;
    }
  }
  closedir(d);
  for (const auto& kv : by_index) out->push_back(kv.second);
  return !out->empty();
}

// Create a directory (and its parents) if it does not exist.  Plain POSIX:
// the test program must not reach into the library's internals.
void EnsureDir(const std::string& dir) {
  if (dir.empty()) return;
  std::string cur;
  size_t i = 0;
  if (dir[0] == '/') { cur = "/"; i = 1; }
  while (i <= dir.size()) {
    const size_t slash = dir.find('/', i);
    const std::string part =
        dir.substr(i, (slash == std::string::npos ? dir.size() : slash) - i);
    if (!part.empty()) {
      if (!cur.empty() && cur.back() != '/') cur += "/";
      cur += part;
      mkdir(cur.c_str(), 0755);
    }
    if (slash == std::string::npos) break;
    i = slash + 1;
  }
}

std::string StemOf(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  const std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
  const size_t dot = base.find_last_of('.');
  return dot == std::string::npos ? base : base.substr(0, dot);
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") { PrintUsage(argv[0]); return 0; }
    else if (a == "--raw-root" && i + 1 < argc) opt.raw_root = argv[++i];
    else if (a == "--sequence" && i + 1 < argc) opt.sequence = argv[++i];
    else if (a == "--model" && i + 1 < argc) opt.model = argv[++i];
    else if (a == "--out" && i + 1 < argc) opt.out = argv[++i];
    else if (a == "--kernel-dir" && i + 1 < argc) opt.kernel_dir = argv[++i];
    else if (a == "--timing" && i + 1 < argc) opt.timing = argv[++i];
    else if (a == "--feature-dump" && i + 1 < argc) opt.feature_dump = argv[++i];
    else if (a == "--mats-dump" && i + 1 < argc) opt.mats_dump = argv[++i];
    else if (a == "--feature-md5-golden" && i + 1 < argc) opt.md5_golden = argv[++i];
    else if (a == "--cpu-affinity" && i + 1 < argc) opt.cpu_affinity = argv[++i];
    else if (a == "--gmc-affinity" && i + 1 < argc) opt.gmc_affinity = argv[++i];
    else if (a == "--gmc-affinity-group" && i + 1 < argc) opt.gmc_affinity_group = argv[++i];
    else if (a == "--npu-affinity" && i + 1 < argc) opt.npu_affinity = argv[++i];
    else if (a == "--anchor-step" && i + 1 < argc) opt.anchor_step = atoi(argv[++i]);
    else if (a == "--npu-workers" && i + 1 < argc) opt.npu_workers = atoi(argv[++i]);
    else if (a == "--gmc-workers" && i + 1 < argc) opt.gmc_workers = atoi(argv[++i]);
    else if (a == "--post-workers" && i + 1 < argc) opt.post_workers = atoi(argv[++i]);
    else if (a == "--queue-depth" && i + 1 < argc) opt.queue_depth = atoi(argv[++i]);
    else if (a == "--input-queue" && i + 1 < argc) opt.input_queue = atoi(argv[++i]);
    else if (a == "--jitter-depth" && i + 1 < argc) opt.jitter_depth = atoi(argv[++i]);
    else if (a == "--gmc-ring-slack" && i + 1 < argc) opt.gmc_ring_slack = atoi(argv[++i]);
    else if (a == "--gmc-deadline-ms" && i + 1 < argc) opt.gmc_deadline_ms = atoi(argv[++i]);
    else if (a == "--gmc-fit-corners" && i + 1 < argc) opt.gmc_fit_corners = atoi(argv[++i]);
    else if (a == "--gmc-fit-win" && i + 1 < argc) opt.gmc_fit_win = atoi(argv[++i]);
    else if (a == "--gmc-fit-level" && i + 1 < argc) opt.gmc_fit_level = atoi(argv[++i]);
    else if (a == "--gmc-fit-iters" && i + 1 < argc) opt.gmc_fit_iters = atoi(argv[++i]);
    else if (a == "--gmc-ransac-max-iters" && i + 1 < argc)
      opt.gmc_ransac_max_iters = atoi(argv[++i]);
    else if (a == "--gmc-phys-anchor" && i + 1 < argc) opt.gmc_phys_anchor = atoi(argv[++i]);
    else if (a == "--gmc-min-stride" && i + 1 < argc) opt.gmc_min_stride = atoi(argv[++i]);
    // DEPRECATED ALIAS of --gmc-min-stride.  The old flag meant "a fixed
    // cadence: every Nth frame gets a group"; the new one means "the minimum
    // gap between two groups".  The alias is kept so an old command line
    // keeps working, and it is reported as an alias below.
    else if (a == "--gmc-compute-stride" && i + 1 < argc) opt.gmc_compute_stride = atoi(argv[++i]);
    else if (a == "--gmc-input-drop-policy" && i + 1 < argc)
      opt.gmc_input_drop_policy = atoi(argv[++i]);
    else if (a == "--gmc-downscale" && i + 1 < argc) opt.gmc_downscale = atoi(argv[++i]);
    else if (a == "--gmc-anchor-threads" && i + 1 < argc)
      opt.gmc_anchor_threads = atoi(argv[++i]);
    else if (a == "--gmc-feature-fingerprint" && i + 1 < argc)
      opt.gmc_feature_fingerprint = atoi(argv[++i]);
    else if (a == "--pyr-cache" && i + 1 < argc) opt.pyr_cache = atoi(argv[++i]);
    else if (a == "--prealloc-pad" && i + 1 < argc) opt.prealloc_pad = atoi(argv[++i]);
    else if (a == "--limit" && i + 1 < argc) opt.limit = atoi(argv[++i]);
    else if (a == "--start" && i + 1 < argc) opt.start = atoi(argv[++i]);
    else if (a == "--pace-ms" && i + 1 < argc) opt.pace_ms = atoi(argv[++i]);
    else if (a == "--watchdog-ms" && i + 1 < argc) opt.watchdog_ms = atoi(argv[++i]);
    else if (a == "--npu-core-mask" && i + 1 < argc) opt.npu_core_mask = atoi(argv[++i]);
    else if (a == "--fault-inject-at" && i + 1 < argc) opt.fault_inject_at = atoi(argv[++i]);
    else if (a == "--input-mode" && i + 1 < argc) opt.input_mode = argv[++i];
    // ---- v10: the decoupled pipeline's flags ------------------------------
    else if (a == "--decoupled-stages" && i + 1 < argc) opt.decoupled_stages = atoi(argv[++i]);
    else if (a == "--buffer-a-depth" && i + 1 < argc) opt.buffer_a_depth = atoi(argv[++i]);
    else if (a == "--buffer-b-depth" && i + 1 < argc) opt.buffer_b_depth = atoi(argv[++i]);
    else if (a == "--pre-threads" && i + 1 < argc) opt.pre_threads = atoi(argv[++i]);
    else if (a == "--buf-a-mode" && i + 1 < argc) opt.buf_a_mode = argv[++i];
    else if (a == "--pre-affinity" && i + 1 < argc) opt.pre_affinity = argv[++i];
    else if (a == "--npu-stage-affinity" && i + 1 < argc) opt.npu_stage_affinity = argv[++i];
    else {
      fprintf(stderr, "[FATAL] unknown option: %s\n", argv[i]);
      PrintUsage(argv[0]);
      return 2;
    }
  }
  if (opt.raw_root.empty() || opt.sequence.empty() || opt.model.empty() || opt.out.empty()) {
    fprintf(stderr, "[FATAL] --raw-root, --sequence, --model and --out are all required\n");
    PrintUsage(argv[0]);
    return 2;
  }

  // The board's librga prints its debug dump to STDOUT.  Detach stdout so the
  // contract "logs on stderr, results in files" cannot be broken.
  if (FILE* devnull = fopen("/dev/null", "w")) {
    dup2(fileno(devnull), STDOUT_FILENO);
    fclose(devnull);
    fprintf(stderr, "[i] stdout detached (the board's librga prints RGA debug dumps there)\n");
  }

  const std::string seq_dir = opt.raw_root + "/" + opt.sequence;
  std::vector<std::string> frames;
  if (!ListFrames(seq_dir, &frames)) {
    fprintf(stderr, "[FATAL] no frames under %s\n", seq_dir.c_str());
    return 1;
  }
  if (opt.start > 0) {
    if ((size_t)opt.start >= frames.size()) {
      fprintf(stderr, "[FATAL] %zu frames, --start %d skips all\n", frames.size(), opt.start);
      return 1;
    }
    frames.erase(frames.begin(), frames.begin() + opt.start);
  }
  if (opt.limit > 0 && (int)frames.size() > opt.limit) frames.resize((size_t)opt.limit);
  fprintf(stderr, "[i] sequence %s: %zu frames\n", opt.sequence.c_str(), frames.size());

  // ---- the library --------------------------------------------------------
  // Zero-initialised as well: mosaic_pipeline_config_defaults() sets every
  // field, but a caller that forgets it must not end up with stack garbage
  // in the GMC knobs.
  MosaicPipelineConfig cfg{};
  mosaic_pipeline_config_defaults(&cfg);
  cfg.model_path = opt.model.c_str();
  cfg.kernel_dir = opt.kernel_dir.c_str();
  cfg.sequence = opt.sequence.c_str();
  cfg.gmc_anchor_step = opt.anchor_step;
  cfg.npu_workers = opt.npu_workers;
  if (opt.gmc_workers >= 0) cfg.gmc_workers = opt.gmc_workers;
  if (opt.post_workers >= 0) cfg.post_workers = opt.post_workers;
  if (opt.queue_depth >= 0) cfg.queue_depth = opt.queue_depth;
  if (opt.input_queue >= 0) cfg.input_queue = opt.input_queue;
  if (opt.jitter_depth >= 0) cfg.jitter_depth = opt.jitter_depth;
  // 0 = AUTO: the library derives the lookahead budget from gmc_workers.
  cfg.gmc_ring_slack = opt.gmc_ring_slack;
  if (opt.gmc_deadline_ms >= 0) cfg.gmc_deadline_ms = opt.gmc_deadline_ms;
  if (opt.gmc_fit_corners >= 0) cfg.gmc_fit_corners = opt.gmc_fit_corners;
  // The GMC fit knobs.  -1 = "not specified": leave the library's own default
  // (15 / 2 / 20 / 2) in place, so the DEFAULT run is byte-identical to the
  // certified build.
  if (opt.gmc_fit_win >= 0) cfg.gmc_fit_win = opt.gmc_fit_win;
  if (opt.gmc_fit_level >= 0) cfg.gmc_fit_level = opt.gmc_fit_level;
  if (opt.gmc_fit_iters >= 0) cfg.gmc_fit_iters = opt.gmc_fit_iters;
  if (opt.gmc_downscale >= 0) cfg.gmc_downscale = opt.gmc_downscale;
  // The RANSAC sampling cap.  0 = the library's own default (the reference
  // call's default arguments), so the DEFAULT run stays byte-identical to the
  // certified build.
  cfg.gmc_ransac_max_iters = opt.gmc_ransac_max_iters;
  // The physical-frame-number anchor and the camera-queue overflow policy.
  // -1 = "not specified": the library's own defaults apply (0 = the
  // consumption counter, i.e. the CERTIFIED behaviour, and evicting the
  // oldest frame on overflow).
  if (opt.gmc_phys_anchor >= 0) cfg.gmc_phys_anchor = opt.gmc_phys_anchor;
  // The minimum gap between two GMC groups.  The NEW flag wins if both are
  // given; the old one is accepted as an alias and reported as such.
  if (opt.gmc_min_stride >= 0) {
    cfg.gmc_min_stride = opt.gmc_min_stride;
  } else if (opt.gmc_compute_stride >= 0) {
    fprintf(stderr,
            "[i] --gmc-compute-stride %d is DEPRECATED: it is read as --gmc-min-stride %d "
            "(the minimum gap between two GMC groups, not a fixed cadence)\n",
            opt.gmc_compute_stride, opt.gmc_compute_stride);
    cfg.gmc_min_stride = opt.gmc_compute_stride;
  }
  if (opt.gmc_input_drop_policy >= 0)
    cfg.gmc_input_drop_policy = opt.gmc_input_drop_policy;
  if (opt.gmc_anchor_threads >= 0) cfg.gmc_anchor_threads = opt.gmc_anchor_threads;
  if (opt.npu_core_mask >= 0) cfg.npu_core_mask = opt.npu_core_mask;
  // The upload path.  The DEFAULT (no flag) stays `set`: that is the
  // certified path the baseline md5 was measured with.
  if (!opt.input_mode.empty()) {
    if (opt.input_mode == "set") cfg.input_mode = MOSAIC_INPUT_SET;
    else if (opt.input_mode == "setf16") cfg.input_mode = MOSAIC_INPUT_SET_F16;
    else if (opt.input_mode == "iomem_nchw") cfg.input_mode = MOSAIC_INPUT_IO_MEM_NCHW;
    else if (opt.input_mode == "iomem_nhwc") cfg.input_mode = MOSAIC_INPUT_IO_MEM_NHWC;
    else {
      fprintf(stderr, "[FATAL] unknown --input-mode '%s' (set|setf16|iomem_nchw|iomem_nhwc)\n",
              opt.input_mode.c_str());
      return 2;
    }
  }
  if (opt.gmc_feature_fingerprint >= 0)
    cfg.gmc_feature_fingerprint = opt.gmc_feature_fingerprint;
  if (opt.pyr_cache >= 0) cfg.gmc_pyr_cache = opt.pyr_cache;  // else: library default
  cfg.gmc_prealloc_pad = opt.prealloc_pad;
  cfg.cpu_affinity = opt.cpu_affinity.empty() ? nullptr : opt.cpu_affinity.c_str();
  cfg.gmc_affinity = opt.gmc_affinity.empty() ? nullptr : opt.gmc_affinity.c_str();
  cfg.gmc_affinity_group =
      opt.gmc_affinity_group.empty() ? nullptr : opt.gmc_affinity_group.c_str();
  cfg.npu_affinity = opt.npu_affinity.empty() ? nullptr : opt.npu_affinity.c_str();
  // ---- v10: the decoupled pipeline ----------------------------------------
  // The knobs are passed through as they are; the library clamps them (and
  // reports what it applied through mosaic_pipeline_effective_config()).
  cfg.decoupled_stages = opt.decoupled_stages;
  cfg.buffer_a_depth = opt.buffer_a_depth;
  cfg.buffer_b_depth = opt.buffer_b_depth;
  cfg.pre_threads = opt.pre_threads;
  if (opt.buf_a_mode == "dma") cfg.buf_a_mode = 0;
  else if (opt.buf_a_mode == "host") cfg.buf_a_mode = 1;
  else {
    fprintf(stderr, "[FATAL] unknown --buf-a-mode '%s' (dma|host)\n", opt.buf_a_mode.c_str());
    return 2;
  }
  cfg.pre_affinity = opt.pre_affinity.empty() ? nullptr : opt.pre_affinity.c_str();
  cfg.npu_stage_affinity =
      opt.npu_stage_affinity.empty() ? nullptr : opt.npu_stage_affinity.c_str();

  // The output root must exist before the TSV is written (the library never
  // touches the filesystem outside the two optional dumps).
  EnsureDir(opt.out);

  const std::string md5_out = opt.out + "/feature_md5.txt";
  char err[512];
  err[0] = '\0';
  MosaicPipeline* pipe = mosaic_pipeline_create(&cfg, err, sizeof(err));
  if (!pipe) {
    fprintf(stderr, "[FATAL] mosaic_pipeline_create failed: %s\n", err);
    return 1;
  }

  mosaic_pipeline_set_feature_md5_out(pipe, md5_out.c_str());
  if (!opt.feature_dump.empty()) {
    EnsureDir(opt.feature_dump);
    mosaic_pipeline_set_feature_dump_out(pipe, opt.feature_dump.c_str());
    fprintf(stderr, "[i] feature dump -> %s\n", opt.feature_dump.c_str());
  }
  if (!opt.mats_dump.empty()) {
    mosaic_pipeline_set_mats_dump_out(pipe, opt.mats_dump.c_str());
    fprintf(stderr, "[i] mats dump -> %s\n", opt.mats_dump.c_str());
  }

  // ---- the streaming consumer ---------------------------------------------
  Consumer consumer;
  mosaic_pipeline_set_result_callback(pipe, OnResult, &consumer);

  // The frames are pushed IN ORDER: the GMC engine is causal.
  //
  // TWO producer modes:
  //
  //  * default (--pace-ms 0): the producer waits for the pipeline.  At most
  //    kInFlight frames are outstanding, so the camera queue never overflows
  //    and the wall clock the library measures IS the pipeline's own pace.
  //
  //  * --pace-ms N: the producer behaves like a REAL camera of 1000/N fps: it
  //    pushes on a fixed schedule and does NOT wait for the pipeline.  This is
  //    the only way to observe what the camera queue does when the pipeline is
  //    slower than the source (the library drops the OLDEST frame and counts
  //    it in timing.dropped).
  const bool paced = opt.pace_ms > 0;
  const int64_t pace_ns = (int64_t)opt.pace_ms * 1000000LL;
  int64_t t_next = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
  int64_t pushed = 0;

  // ---------------------------------------------------------------------
  // THE WATCHDOG.
  //
  // The demo must NEVER hang silently.  A pipeline that wedges -- an NPU
  // submit that the driver refuses leaves every service thread asleep on a
  // condition variable, and the caller waits for a result that will never
  // come -- has to be REPORTED and ENDED, not waited on for ever.  (That
  // wedge cost a board: the run sat at frame 155 for ever and the only
  // evidence was a single "rknn_run failed" line in the log.)
  //
  // Two conditions end the run:
  //   * the pipeline reports an error  -> [FATAL] <err>, exit non-zero
  //   * no result arrives for --watchdog-ms -> the counters are printed and
  //     the run exits non-zero
  //
  // The progress signal is the number of DELIVERED results.  It is read
  // without the consumer's lock (it is atomic), so the watchdog can be
  // called from inside a wait that holds that lock.
  // ---------------------------------------------------------------------
  int64_t last_progress = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
  int64_t last_delivered = -1;
  auto now_ms = []() -> int64_t {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  // Returns false when the run must end.  `where` says what the demo was
  // doing, so the message says where it died.
  auto pipeline_faulted = [&](const char* where) -> bool {
    const char* err = mosaic_pipeline_last_error(pipe);
    if (err && err[0]) {
      fprintf(stderr, "[FATAL] the pipeline reported an error while %s: %s\n", where, err);
      return true;
    }
    const int64_t delivered = consumer.delivered.load();
    const int64_t now = now_ms();
    if (delivered != last_delivered) {
      last_delivered = delivered;
      last_progress = now;
      return false;
    }
    if (opt.watchdog_ms > 0 && now - last_progress > opt.watchdog_ms) {
      MosaicPipelineTiming t;
      mosaic_pipeline_get_timing(pipe, &t);
      fprintf(stderr,
              "[FATAL] no pipeline progress for %d ms while %s: pushed=%lld delivered=%lld "
              "frames=%lld dropped=%lld features_dropped=%lld detections_dropped=%lld "
              "gmc_fit=%.1f ms gmc_total=%.1f ms wall=%.1f ms\n",
              opt.watchdog_ms, where, (long long)pushed, (long long)delivered,
              (long long)t.frames, (long long)t.dropped, (long long)t.features_dropped,
              (long long)t.detections_dropped, t.gmc_fit_ms, t.gmc_total_ms, t.wall_ms);
      return true;
    }
    return false;
  };

  for (size_t i = 0; i < frames.size(); i++) {
    if (paced) {
      std::this_thread::sleep_until(
          std::chrono::steady_clock::time_point(std::chrono::nanoseconds(t_next)));
      t_next += pace_ns;
    } else {
      // Keep at most kInFlight frames outstanding.  This is the pacing the
      // non-blocking push() requires: the camera queue would otherwise drop
      // the OLDEST frame, and a dropped frame is a frame with no result.
      //
      // The wait is TIMED, so a wedged pipeline is caught here instead of
      // hanging for ever.
      std::unique_lock<std::mutex> lk(consumer.m);
      while ((pushed - consumer.delivered.load()) >= kInFlight) {
        if (consumer.cv.wait_for(lk, std::chrono::milliseconds(200), [&] {
              return (pushed - consumer.delivered.load()) < kInFlight;
            }))
          break;
        if (pipeline_faulted("pushing frames")) {
          mosaic_pipeline_destroy(pipe);
          return 1;
        }
      }
    }
    const cv::Mat img = cv::imread(frames[i], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      fprintf(stderr, "[FATAL] cannot read %s\n", frames[i].c_str());
      mosaic_pipeline_destroy(pipe);
      return 1;
    }
    const int64_t fid = mosaic_pipeline_push(pipe, img.data, img.cols, img.rows);
    if (fid < 0) {
      fprintf(stderr, "[FATAL] push failed at frame %zu: %s\n", i, mosaic_pipeline_last_error(pipe));
      mosaic_pipeline_destroy(pipe);
      return 1;
    }
    ++pushed;
    // The latency clock starts the moment the frame is IN the pipeline.  It
    // is taken right after the push so that the measurement includes
    // everything the pipeline does with the frame, and it excludes the
    // camera's own imread (which is the source's cost, not the pipeline's).
    {
      std::lock_guard<std::mutex> lk(consumer.m);
      consumer.push_at[fid] = std::chrono::steady_clock::now();
    }
    // The FAULT INJECTION: the library's own FATAL-error path.  Everything
    // after this point must unwind -- the pipeline must abort, the demo must
    // report the error and exit non-zero, and NOTHING may hang.
    if (opt.fault_inject_at >= 0 && pushed == opt.fault_inject_at) {
      fprintf(stderr, "[i] FAULT INJECTION: failing the pipeline on purpose after %lld frames\n",
              (long long)pushed);
      mosaic_pipeline_inject_fault(pipe, "injected fault (test hook)");
    }
  }
  mosaic_pipeline_flush(pipe);

  // flush() joins every worker AND the dispatcher, so no further result can
  // arrive after it returns.  The number of results is the number of GMC
  // GROUPS that completed, minus the two downstream drop counters:
  //   * gmc_groups_done    -- the frames the GMC service really computed a
  //                           feature for.  NOT the number of pushed frames:
  //                           under the OPPORTUNISTIC scheduler the engine
  //                           computes the NEWEST frame and SKIPS the ones it
  //                           could not keep up with (they are seeded as
  //                           history and produce nothing), and the frames the
  //                           camera queue dropped never reach the GMC service
  //                           at all.
  //   * features_dropped   -- the GMC computed it but the NPU services skipped it
  //   * detections_dropped -- the NPU ran but the back-projection skipped it
  // A frame dropped at the camera never reaches the GMC, and a frame dropped
  // after the GMC never produces a result, so waiting for `pushed` results
  // would hang forever in a paced run.
  //
  // WHY THE COUNTER AND NOT `pushed`: the formula assumes one result per
  // COMPUTED group, which is what the pipeline guarantees: every group that
  // completed produces exactly one feature and (unless the inference skipped
  // it) exactly one result.  The number of pushed frames is a different
  // number, and under the opportunistic scheduler it is NOT the denominator
  // any more -- the frames the engine skipped are in the manifest
  // (processed_frames.csv), not in the results.
  MosaicPipelineTiming tm_probe;
  mosaic_pipeline_get_timing(pipe, &tm_probe);
  // The expected number of results.
  //
  // v10: in the DECOUPLED mode there are TWO more places where a frame can be
  // lost between the GMC group and the result: Buffer A (Stage 1's ring) and
  // Buffer B (the heatmap ring).  Buffer B's losses are ALREADY counted in
  // detections_dropped, so only Buffer A's two counters are added here --
  // counting them twice would under-count the results.
  int64_t expect = tm_probe.gmc_groups_done - tm_probe.features_dropped -
                   tm_probe.detections_dropped;
  if (opt.decoupled_stages) {
    expect -= tm_probe.bufA_dropped + tm_probe.bufA_full_drops;
    if (expect < 0) expect = 0;
  }
  // The last results are delivered by the callback after the flush.
  // The wait is TIMED for the same reason as the producer's: a pipeline that
  // never delivers the last results must not hang the demo.
  {
    std::unique_lock<std::mutex> lk(consumer.m);
    while (consumer.delivered.load() < expect) {
      if (consumer.cv.wait_for(lk, std::chrono::milliseconds(200), [&] {
            return consumer.delivered.load() >= expect;
          }))
        break;
      if (pipeline_faulted("waiting for the last results")) {
        mosaic_pipeline_destroy(pipe);
        return 1;
      }
    }
  }
  // The workers finish out of order, so the rows are keyed by frame id and
  // written back in frame order.
  std::map<int64_t, std::vector<MosaicDet>> rows;
  {
    std::lock_guard<std::mutex> lk(consumer.m);
    rows = std::move(consumer.rows);
  }
  if ((int64_t)rows.size() != expect) {
    fprintf(stderr, "[FATAL] %lld frames pushed, %lld GMC groups computed, "
                    "%lld features and %lld detections dropped downstream, "
                    "but %zu results delivered (expected %lld)\n",
            (long long)pushed, (long long)tm_probe.gmc_groups_done,
            (long long)tm_probe.features_dropped, (long long)tm_probe.detections_dropped,
            rows.size(), (long long)expect);
    mosaic_pipeline_destroy(pipe);
    return 1;
  }
  if (paced && tm_probe.dropped > 0) {
    fprintf(stderr,
            "[i] the camera ran at %.1f fps and the CAMERA queue overflowed: %lld of %lld "
            "frames never reached the GMC service.  That breaks the GMC registration "
            "chain -- the GMC service is not keeping up with the source\n",
            1000.0 / opt.pace_ms, (long long)tm_probe.dropped, (long long)pushed);
  } else if (paced) {
    fprintf(stderr,
            "[i] the camera ran at %.1f fps: the GMC service produced %lld feature(s) "
            "(camera-queue drops %lld); the inference stage skipped %lld feature(s) "
            "and %lld detection(s) because it could not keep up -- by design\n",
            1000.0 / opt.pace_ms, (long long)tm_probe.gmc_groups_done,
            (long long)tm_probe.dropped,
            (long long)tm_probe.features_dropped, (long long)tm_probe.detections_dropped);
  }

  // ---- write the TSV ------------------------------------------------------
  char tsv_path[1024];
  snprintf(tsv_path, sizeof(tsv_path), "%s/%s_mosaic.tsv", opt.out.c_str(), opt.sequence.c_str());
  FILE* tsv = fopen(tsv_path, "w");
  if (!tsv) {
    fprintf(stderr, "[FATAL] cannot open %s: %s\n", tsv_path, strerror(errno));
    mosaic_pipeline_destroy(pipe);
    return 1;
  }
  // The stem must carry the frame number: the scorer's stem_to_frame_index()
  // takes the digits after the last "__".
  for (const auto& kv : rows) {
    char stem[256];
    snprintf(stem, sizeof(stem), "%s__%06lld", opt.sequence.c_str(), (long long)kv.first);
    fprintf(tsv, "%s\t%zu", stem, kv.second.size());
    for (const MosaicDet& d : kv.second) {
      fprintf(tsv, "\t%d %.4f %.4f %d %d %d %.6f %.4f", d.tag, (double)d.cx, (double)d.cy, d.w, d.h,
              d.area, (double)d.peak, (double)d.sum);
    }
    fprintf(tsv, "\n");
  }
  fclose(tsv);
  fprintf(stderr, "[i] TSV -> %s (%zu frames)\n", tsv_path, rows.size());
  fprintf(stderr, "[i] feature md5 (pre-encode planes) = %s\n", mosaic_pipeline_feature_md5(pipe));

  // ---- write the PROCESSED FRAME MANIFEST ---------------------------------
  // THE LIST OF THE FRAMES THIS RUN ACTUALLY PROCESSED.
  //
  // WHY IT EXISTS: the board-side inference program has to leave behind a
  // manifest of what it processed, so that the result set can be reconciled
  // with the run afterwards ("which frames did the pipeline see?").  Under
  // the OPPORTUNISTIC scheduler that question is NOT "how many frames were
  // pushed": the engine skips the frames it could not keep up with, so the
  // processed set is a SUBSET of the pushed set and only the manifest says
  // which subset.
  //
  // The rows arrive in COMPLETION order (the callback fires from the
  // dispatcher thread), so the manifest is written ONCE, at the end, from the
  // map -- which is keyed by the frame id and therefore already in frame
  // order.
  //
  // The per-frame feature fingerprint is NOT repeated here: it lives in the
  // library's own feature_md5.txt (same directory, one line per computed
  // frame, with the same frame id), so the two files can be joined on it.
  char csv_path[1024];
  snprintf(csv_path, sizeof(csv_path), "%s/processed_frames.csv", opt.out.c_str());
  {
    FILE* csv = fopen(csv_path, "w");
    if (!csv) {
      fprintf(stderr, "[FATAL] cannot open %s: %s\n", csv_path, strerror(errno));
      mosaic_pipeline_destroy(pipe);
      return 1;
    }
    fprintf(csv, "frame_id,n_dets\n");
    for (const auto& kv : rows) fprintf(csv, "%lld,%zu\n", (long long)kv.first, kv.second.size());
    fclose(csv);
    fprintf(stderr, "[i] processed frames -> %s (%zu frames)\n", csv_path, rows.size());
  }

  // ---- golden fingerprint self-check -------------------------------------
  // The GMC feature stream is the one thing that must be bit-identical to the
  // reference implementation.  Comparing it takes a second and needs no GT,
  // no scoring run and no NPU at all -- so the embedded team can verify a
  // deployment (kernel dir, OpenCL ICD, OpenCV build, input frames) before
  // wiring the library into their application.
  int golden_rc = 0;
  if (!opt.md5_golden.empty()) {
    FILE* gf = fopen(opt.md5_golden.c_str(), "r");
    FILE* pf = fopen(md5_out.c_str(), "r");
    if (!gf || !pf) {
      fprintf(stderr, "[FATAL] cannot open the golden or the produced md5 file\n");
      golden_rc = 2;
    } else {
      char gl[256], pl[256];
      long same = 0, diff = 0, n = 0;
      while (fgets(gl, sizeof(gl), gf) && fgets(pl, sizeof(pl), pf)) {
        char gseq[128], pseq[128], gidx[64], pidx[64], ghash[64], phash[64];
        if (sscanf(gl, "%127[^\t]\t%63[^\t]\t%63[^\n]", gseq, gidx, ghash) != 3) continue;
        if (sscanf(pl, "%127[^\t]\t%63[^\t]\t%63[^\n]", pseq, pidx, phash) != 3) continue;
        ++n;
        if (strcmp(ghash, phash) == 0) ++same; else ++diff;
      }
      if (n == 0) {
        fprintf(stderr, "[FATAL] the golden comparison compared 0 frames\n");
        golden_rc = 2;
      } else {
        fprintf(stderr, "[SELF-CHECK] feature md5 vs golden: %ld SAME / %ld DIFF "
                        "(of %ld frames)\n", same, diff, n);
        if (diff != 0) {
          fprintf(stderr, "[SELF-CHECK] **FAIL** -- the GMC feature stream is NOT the "
                          "reference one (check kernel_dir, OpenCL ICD, OpenCV build, "
                          "input frames, and that gmc_pyr_cache matches)\n");
          golden_rc = 1;
        } else {
          fprintf(stderr, "[SELF-CHECK] **PASS** -- the GMC feature stream is bit-identical "
                          "to the reference implementation\n");
        }
      }
    }
    if (gf) fclose(gf);
    if (pf) fclose(pf);
  }

  // ---- timing report ------------------------------------------------------
  MosaicPipelineTiming tm;
  mosaic_pipeline_get_timing(pipe, &tm);
  // TWO DENOMINATORS, AND THEY ARE NOT THE SAME NUMBER.
  //
  // The GMC-side counters are accumulated per frame the GMC service actually
  // FITTED (gmc_fit_samples), while the NPU/POST-side counters are accumulated
  // per frame that came out of the back-projection (frames).  In a paced run
  // the inference stage skips frames, so frames < gmc_fit_samples: dividing
  // the GMC counters by frames inflated every GMC average by the ratio
  // (measured on the board: 11011 fitted vs 6948 detected = 1.58x).  Each
  // side is now divided by its OWN denominator, and both denominators are
  // printed in the header so the table can be read without guessing.
  const double det_frames = tm.frames > 0 ? (double)tm.frames : 1.0;
  const double gmc_frames = tm.gmc_fit_samples > 0 ? (double)tm.gmc_fit_samples : 1.0;
  // The lookahead budget the library ACTUALLY applied.  The library resolves
  // AUTO in its sanitize stage and reports the RESOLVED value through
  // mosaic_pipeline_effective_config(), so there is deliberately NO second
  // copy of the rule here: one rule, one place, and the report cannot drift
  // away from the pipeline.
  MosaicPipelineConfig eff_cfg;
  mosaic_pipeline_effective_config(pipe, &eff_cfg);
  const int ring_slack_eff = eff_cfg.gmc_ring_slack;
  const char* timing_path = opt.timing.empty() ? nullptr : opt.timing.c_str();
  char tbuf[1024];
  if (!timing_path) {
    snprintf(tbuf, sizeof(tbuf), "%s/timing.txt", opt.out.c_str());
    timing_path = tbuf;
  }
  if (FILE* tf = fopen(timing_path, "w")) {
    fprintf(tf, "# rknn_mosaic_e2e timing (library-driven)\n");
    fprintf(tf, "seq=%s\n", opt.sequence.c_str());
    fprintf(tf, "anchor_step=%d\n", opt.anchor_step);
    fprintf(tf, "npu_workers=%d\n", opt.npu_workers);
    fprintf(tf, "gmc_workers=%d\n", cfg.gmc_workers);
    fprintf(tf, "post_workers=%d\n", cfg.post_workers);
    fprintf(tf, "jitter_depth=%d\n", cfg.jitter_depth);
    fprintf(tf, "input_queue=%d\n", cfg.input_queue);
    fprintf(tf, "input_mode=%d\n", cfg.input_mode);
    fprintf(tf, "gmc_fit_corners=%d\n", cfg.gmc_fit_corners);
    // The GMC FIT knobs that were IN FORCE, so a sweep arm can be reproduced
    // from the timing file alone.  These are read back with
    // mosaic_pipeline_effective_config(), i.e. they are the values the library
    // ACTUALLY applied after its out-of-range clamping -- not the requested
    // ones.
    fprintf(tf, "gmc_fit_win=%d\n", eff_cfg.gmc_fit_win);
    fprintf(tf, "gmc_fit_level=%d\n", eff_cfg.gmc_fit_level);
    fprintf(tf, "gmc_fit_iters=%d\n", eff_cfg.gmc_fit_iters);
    fprintf(tf, "gmc_downscale=%d\n", eff_cfg.gmc_downscale);
    // The RANSAC sampling cap that was IN FORCE, so a sweep arm is
    // reproducible from the timing file alone (the applied value, not the
    // requested one).
    fprintf(tf, "gmc_ransac_max_iters=%d\n", eff_cfg.gmc_ransac_max_iters);
    // THE THREE-LEVEL GMC CACHE (resize | corners | LK pyramid), which is ON
    // by default.  The cache only removes RECOMPUTATION: the cached and the
    // uncached path produce byte-identical features (md5
    // 0bf598a1d890040bcfa8d157dc69c032 over the 11,810-frame val3 track), so
    // turning it on costs nothing in accuracy.  It is printed here so the
    // board artefact carries the evidence for the "the cache is on" claim.
    // This is the value the library ACTUALLY applied (read back with
    // mosaic_pipeline_effective_config), not the requested one.
    fprintf(tf, "gmc_pyr_cache=%d\n", eff_cfg.gmc_pyr_cache);
    // The cache's own level mask: 7 = resize (1) | corners (2) | pyramid (4).
    // It is a constant of the library's engine construction
    // (mosaic_pipeline.cpp: /*cache_parts=*/7), not configurable through the
    // ABI, so it is printed as the literal it is -- verifiable by grep.
    fprintf(tf, "gmc_cache_parts=%d\n", 7);
    fprintf(tf, "pace_ms=%d\n", opt.pace_ms);
    fprintf(tf, "pushed=%lld\n", (long long)pushed);
    fprintf(tf, "frames=%lld\n", (long long)tm.frames);
    // The TWO denominators of the table below.
    //   gmc_frames  -- frames the GMC service fitted; the denominator of
    //                  every GMC-side segment.
    //   det_frames  -- frames that finished the whole path; the denominator of
    //                  every NPU/POST-side segment.
    fprintf(tf, "gmc_frames=%lld\n", (long long)tm.gmc_fit_samples);
    fprintf(tf, "det_frames=%lld\n", (long long)tm.frames);
    fprintf(tf, "%-18s %14s %14s\n", "segment", "total_ms", "avg_ms/frame");
    const struct { const char* name; double v; double den; } segs[] = {
        // GMC side: per frame the GMC service fitted.
        {"gmc_fit", tm.gmc_fit_ms, gmc_frames},
        {"gmc_warp", tm.gmc_warp_ms, gmc_frames},
        {"gmc_median", tm.gmc_median_ms, gmc_frames},
        {"gmc_total", tm.gmc_total_ms, gmc_frames},
        {"gmc_post", tm.gmc_post_ms, gmc_frames},
        {"gmc_ingest", tm.gmc_ingest_ms, gmc_frames},
        // The ingest's wait breakdown: where gmc_ingest_ms went.
        {"gmc_ingest_in", tm.gmc_ingest_in_wait_ms, gmc_frames},
        {"gmc_ingest_window", tm.gmc_ingest_window_ms, gmc_frames},
        {"gmc_ingest_space", tm.gmc_ingest_space_ms, gmc_frames},
        {"gmc_wait", tm.gmc_wait_ms, gmc_frames},
        // NPU/POST side: per frame that finished the whole path.
        {"rga_import", tm.rga_import_ms, det_frames},
        {"rga_improcess", tm.rga_improcess_ms, det_frames},
        {"upload", tm.upload_ms, det_frames},
        {"rknn_run", tm.run_ms, det_frames},
        {"outputs_get", tm.outputs_ms, det_frames},
        {"backproject_cc", tm.backproj_ms, det_frames},
        {"npu_wait", tm.npu_wait_ms, det_frames},
        {"post_wait", tm.post_wait_ms, det_frames},
    };
    for (const auto& s : segs)
      fprintf(tf, "%-18s %14.3f %14.4f\n", s.name, s.v, s.v / s.den);
    fprintf(tf, "%-18s %14.3f %14.4f\n", "total_wall", tm.wall_ms, tm.wall_ms / det_frames);
    // THE END-TO-END LATENCY: camera push -> result delivered.  This is the
    // number a closed loop cares about, and it is the one that would expose a
    // batching design (a buffer that has to fill up before the first frame
    // can be computed) as the ~1.7 s of dead time it would really cost.
    {
      std::vector<double> lat;
      {
        std::lock_guard<std::mutex> lk(consumer.m);
        lat = consumer.lat_ms;
      }
      std::sort(lat.begin(), lat.end());
      const size_t n = lat.size();
      auto pct = [&lat, n](double q) -> double {
        if (n == 0) return 0.0;
        size_t i = (size_t)std::ceil(q * (double)n);
        if (i == 0) i = 1;
        if (i > n) i = n;
        return lat[i - 1];
      };
      fprintf(tf, "e2e_latency_samples=%zu\n", n);
      fprintf(tf, "e2e_latency_mean=%.3f\n", n ? std::accumulate(lat.begin(), lat.end(), 0.0) / (double)n : 0.0);
      fprintf(tf, "e2e_latency_p50=%.3f\n", pct(0.50));
      fprintf(tf, "e2e_latency_p95=%.3f\n", pct(0.95));
      fprintf(tf, "e2e_latency_p99=%.3f\n", pct(0.99));
      fprintf(tf, "e2e_latency_max=%.3f\n", n ? lat[n - 1] : 0.0);
      fprintf(stderr,
              "[timing] END-TO-END latency (camera push -> result delivered, n=%zu): "
              "mean %.1f  p50 %.1f  p95 %.1f  p99 %.1f  max %.1f ms\n",
              n, n ? std::accumulate(lat.begin(), lat.end(), 0.0) / (double)n : 0.0,
              pct(0.50), pct(0.95), pct(0.99), n ? lat[n - 1] : 0.0);
    }
    fprintf(tf, "gmc_fit_max=%.3f\n", tm.gmc_fit_max_ms);
    // The fit DISTRIBUTION, nearest rank.  A mean alone hides the tail, and
    // the tail is what the jitter buffer absorbs and what the watchdog bounds.
    fprintf(tf, "gmc_fit_p50=%.3f\n", tm.gmc_fit_p50_ms);
    fprintf(tf, "gmc_fit_p95=%.3f\n", tm.gmc_fit_p95_ms);
    fprintf(tf, "gmc_fit_p99=%.3f\n", tm.gmc_fit_p99_ms);
    fprintf(tf, "gmc_fit_samples=%lld\n", (long long)tm.gmc_fit_samples);
    // The SAME distribution for the single rknn_run: the NPU is the pipeline's
    // critical path, and the core-mask experiment is judged on this one.
    fprintf(tf, "rknn_run_max=%.3f\n", tm.rknn_run_max_ms);
    fprintf(tf, "rknn_run_p50=%.3f\n", tm.rknn_run_p50_ms);
    fprintf(tf, "rknn_run_p95=%.3f\n", tm.rknn_run_p95_ms);
    fprintf(tf, "rknn_run_p99=%.3f\n", tm.rknn_run_p99_ms);
    fprintf(tf, "rknn_run_samples=%lld\n", (long long)tm.rknn_run_samples);
    // The knobs that were in force, so a report can be reproduced from the
    // timing file alone.
    fprintf(tf, "npu_core_mask=%d\n", cfg.npu_core_mask);
    fprintf(tf, "input_mode=%d\n", cfg.input_mode);
    // WHY the fit costs what it costs: the corner budget and how often it
    // saturated.  The GMC cost is content-driven, not a constant.
    fprintf(tf, "gmc_corner_mean=%.1f\n", tm.gmc_corner_mean);
    fprintf(tf, "gmc_corner_capped=%lld\n", (long long)tm.gmc_corner_capped);
    fprintf(tf, "dropped=%lld\n", (long long)tm.dropped);
    fprintf(tf, "features_dropped=%lld\n", (long long)tm.features_dropped);
    fprintf(tf, "detections_dropped=%lld\n", (long long)tm.detections_dropped);
    // The GMC GROUPS that completed: the frames a feature was really computed
    // for.  Under the OPPORTUNISTIC scheduler this is a fraction of `pushed`
    // (the engine computes the NEWEST frame and skips the ones it could not
    // keep up with), and it is the denominator the result count is checked
    // against.
    fprintf(tf, "gmc_groups_done=%lld\n", (long long)tm.gmc_groups_done);
    // The frames the engine VOIDED because their 21-frame history was not
    // whole (the producer outran the ring, or a camera frame was lost).  0 in
    // a healthy run.
    fprintf(tf, "gmc_void_frames=%lld\n", (long long)tm.gmc_void_frames);
    // The frames that were seeded into the history but never computed: the
    // engine was busy with a newer frame.  NOT a loss -- they are still
    // history, so skipping never punches a hole in it.
    fprintf(tf, "gmc_frames_skipped=%lld\n", (long long)tm.gmc_frames_skipped);
    // THE MINIMUM GAP between two GMC groups that was in force (read back
    // with mosaic_pipeline_effective_config), and the manifest of the frames
    // this run actually processed.
    fprintf(tf, "gmc_min_stride=%d\n", eff_cfg.gmc_min_stride);
    fprintf(tf, "processed_frames_csv=%s\n", csv_path);
    fprintf(tf, "deadline_blows=%lld\n", (long long)tm.gmc_deadline_blows);
    // The MISSING physical history frames.  0 unless gmc_phys_anchor=1: with
    // the consumption-counter anchor the engine cannot see a lost camera
    // frame, so the loss was silent.  This is the number that makes it
    // visible.
    fprintf(tf, "gmc_history_gaps=%lld\n", (long long)tm.gmc_history_gaps);
    // The two knobs that decide HOW the GMC history is anchored and WHAT the
    // camera queue does on overflow, as they were IN FORCE (read back with
    // mosaic_pipeline_effective_config).
    fprintf(tf, "gmc_phys_anchor=%d\n", eff_cfg.gmc_phys_anchor);
    fprintf(tf, "gmc_input_drop_policy=%d\n", eff_cfg.gmc_input_drop_policy);
    // The ingest's wait breakdown, in TOTAL ms (not per frame): the three
    // waits plus the ingest's own work are gmc_ingest_ms.
    fprintf(tf, "gmc_ingest_in_ms=%.3f\n", tm.gmc_ingest_in_wait_ms);
    fprintf(tf, "gmc_ingest_window_ms=%.3f\n", tm.gmc_ingest_window_ms);
    fprintf(tf, "gmc_ingest_space_ms=%.3f\n", tm.gmc_ingest_space_ms);
    // The depth of the engine job queue at the moment an engine popped a job:
    // ~1 means the engines were STARVING (they picked the job off an empty
    // queue), well above 1 means they always had a backlog.
    fprintf(tf, "gmc_queue_depth_mean=%.3f\n",
            tm.gmc_queue_samples > 0 ? tm.gmc_queue_depth_sum / (double)tm.gmc_queue_samples : 0.0);
    fprintf(tf, "gmc_queue_samples=%lld\n", (long long)tm.gmc_queue_samples);
    // The lookahead budget in force (0 = the library's AUTO rule).
    fprintf(tf, "gmc_ring_slack=%d\n", ring_slack_eff);
  fprintf(tf, "gmc_window_blocks=%lld\n", (long long)tm.gmc_window_blocks);
  fprintf(tf, "gmc_space_blocks=%lld\n", (long long)tm.gmc_space_blocks);
    fprintf(tf, "fps=%.3f\n", tm.wall_ms > 0 ? 1000.0 * (double)tm.frames / tm.wall_ms : 0.0);
    // -----------------------------------------------------------------------
    // v10: THE THREE-STAGE DECOUPLED PIPELINE.
    //
    // APPENDED, so every line above keeps its meaning and its position: the
    // history scripts read them and must not have to care whether this run was
    // fused or decoupled.  In the FUSED mode all the counters below are 0
    // (the fused worker reports into the old buckets).
    // -----------------------------------------------------------------------
    fprintf(tf, "decoupled_stages=%d\n", (int)eff_cfg.decoupled_stages);
    fprintf(tf, "buffer_a_depth=%d\n", (int)eff_cfg.buffer_a_depth);
    fprintf(tf, "buffer_b_depth=%d\n", (int)eff_cfg.buffer_b_depth);
    fprintf(tf, "pre_threads=%d\n", (int)eff_cfg.pre_threads);
    fprintf(tf, "buf_a_mode=%d\n", (int)eff_cfg.buf_a_mode);
    fprintf(tf, "pre_affinity=%s\n",
            (eff_cfg.pre_affinity && *eff_cfg.pre_affinity) ? eff_cfg.pre_affinity : "(inherit)");
    fprintf(tf, "npu_stage_affinity=%s\n",
            (eff_cfg.npu_stage_affinity && *eff_cfg.npu_stage_affinity)
                ? eff_cfg.npu_stage_affinity
                : "(follow pre_affinity)");
    // The stage timings, in TOTAL ms over the whole run.
    fprintf(tf, "stage1_ms=%.3f\n", tm.stage1_ms);
    fprintf(tf, "stage1_wait_ms=%.3f\n", tm.stage1_wait_ms);
    fprintf(tf, "stage2_ms=%.3f\n", tm.stage2_ms);
    fprintf(tf, "stage2_wait_ms=%.3f\n", tm.stage2_wait_ms);
    fprintf(tf, "bufA_upload_ms=%.3f\n", tm.bufA_upload_ms);
    // The bounded buffers' losses.  bufB_dropped is ALSO counted in
    // detections_dropped (a lost heatmap IS a lost detection).
    fprintf(tf, "bufA_dropped=%lld\n", (long long)tm.bufA_dropped);
    fprintf(tf, "bufA_full_drops=%lld\n", (long long)tm.bufA_full_drops);
    fprintf(tf, "bufB_dropped=%lld\n", (long long)tm.bufB_dropped);
    // The NPU's own busy time, and its utilisation against the wall clock.
    fprintf(tf, "npu_busy_ms=%.3f\n", tm.npu_busy_ms);
    {
      const double denom = tm.wall_ms * (double)(eff_cfg.npu_workers > 0 ? eff_cfg.npu_workers : 1);
      fprintf(tf, "npu_util_pct=%.3f\n", denom > 0 ? 100.0 * tm.npu_busy_ms / denom : 0.0);
    }
    fclose(tf);
  }
  fprintf(stderr,
          "[timing] %lld frames, wall %.1f ms (%.3f ms/frame), %.3f fps\n"
          "[timing] gmc_fit %.3f  gmc_total %.3f  rga %.3f  upload %.3f  run %.3f  "
          "outputs %.3f  backproj %.3f  (ms/frame)\n"
          "[timing] services[gmc=%d npu=%d post=%d] jitter=%d  "
          "starvation[gmc=%.3f npu=%.3f post=%.3f] ms/frame  "
          "gmc_fit_max=%.3f ms  dropped=%lld  features_dropped=%lld  "
          "detections_dropped=%lld  deadline_blows=%lld  gmc_history_gaps=%lld\n"
          "[timing] GMC fit distribution (n=%lld): p50 %.3f  p95 %.3f  p99 %.3f  max %.3f ms  |  "
          "corners mean %.1f, %lld frames saturated the corner budget\n"
          "[timing] ingest breakdown (ms/frame over %lld fitted frames): "
          "in %.3f  window %.3f  space %.3f  |  lookahead budget %d frames, "
          "engine job queue depth at pop %.2f (n=%lld)  window_blocks %lld  "
          "space_blocks %lld\n"
          "[timing] opportunistic schedule: %lld frames seeded into the history, "
          "%lld GMC groups computed, %lld frames skipped (the engine was busy with a "
          "newer frame -- they are still history), %lld frames voided (their 21-frame "
          "history was not whole)  |  processed frames -> %s\n",
          (long long)tm.frames, tm.wall_ms, tm.wall_ms / (tm.frames > 0 ? tm.frames : 1),
          tm.wall_ms > 0 ? 1000.0 * (double)tm.frames / tm.wall_ms : 0.0,
          tm.gmc_fit_ms / gmc_frames, tm.gmc_total_ms / gmc_frames,
          (tm.rga_import_ms + tm.rga_improcess_ms) / det_frames, tm.upload_ms / det_frames,
          tm.run_ms / det_frames,
          tm.outputs_ms / det_frames, tm.backproj_ms / det_frames,
          cfg.gmc_workers, cfg.npu_workers, cfg.post_workers, cfg.jitter_depth,
          tm.gmc_wait_ms / gmc_frames, tm.npu_wait_ms / det_frames, tm.post_wait_ms / det_frames,
          tm.gmc_fit_max_ms,
          (long long)tm.dropped, (long long)tm.features_dropped,
          (long long)tm.detections_dropped, (long long)tm.gmc_deadline_blows,
          (long long)tm.gmc_history_gaps,
          (long long)tm.gmc_fit_samples, tm.gmc_fit_p50_ms, tm.gmc_fit_p95_ms, tm.gmc_fit_p99_ms,
          tm.gmc_fit_max_ms, tm.gmc_corner_mean, (long long)tm.gmc_corner_capped,
          (long long)tm.gmc_fit_samples, tm.gmc_ingest_in_wait_ms / gmc_frames,
          tm.gmc_ingest_window_ms / gmc_frames, tm.gmc_ingest_space_ms / gmc_frames,
          ring_slack_eff,
          tm.gmc_queue_samples > 0 ? tm.gmc_queue_depth_sum / (double)tm.gmc_queue_samples : 0.0,
          (long long)tm.gmc_queue_samples, (long long)tm.gmc_window_blocks,
          (long long)tm.gmc_space_blocks,
          (long long)(tm.gmc_groups_done + tm.gmc_frames_skipped + tm.gmc_void_frames),
          (long long)tm.gmc_groups_done, (long long)tm.gmc_frames_skipped,
          (long long)tm.gmc_void_frames, csv_path);

  mosaic_pipeline_destroy(pipe);
  // A golden-fingerprint mismatch is a hard failure: the process must NOT
  // report success to whatever CI/脚本 is driving it.
  return golden_rc;
}
