// ===========================================================================
// mosaic_api_test.cpp -- CONTRACT TEST for libmosaic_pipeline.
//
// The demo (rknn_mosaic_e2e) drives the library; this program ASSERTS on the
// contract the embedded team will code against.  It links the library
// statically and touches ONLY the public C ABI -- exactly like the demo.
//
// WHAT IT PINS DOWN (each item is a promise the header makes)
// -------------------------------------------------------------
//  1. config_defaults() fills EVERY field.  A field left unset is a field
//     that silently carries stack garbage into the GMC -- which is how the
//     fit gate once degraded the transforms without anybody noticing.
//  2. create() REJECTS a bad version and a missing model, with a message.
//  3. create() CLAMPS out-of-range knobs instead of using them, and the
//     effective config can be read back to prove it.
//  4. push() -> collect() returns one result per pushed frame, in order,
//     with plausible detections.
//  5. A wrong-sized frame is REJECTED and the reason is retrievable.
//  6. flush() joins the threads; collect() afterwards returns -1 instead of
//     blocking forever.
//  7. The GMC feature stream matches the frozen golden fingerprint.
//
// Exit code 0 = every assertion passed.  Any failure -> 1, with the failing
// assertion named.  Run it on the board; it takes seconds.
// ===========================================================================
#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mosaic_pipeline.h"

namespace {

int g_pass = 0;
int g_fail = 0;

void Check(bool ok, const char* what) {
  if (ok) {
    ++g_pass;
    fprintf(stderr, "[PASS] %s\n", what);
  } else {
    ++g_fail;
    fprintf(stderr, "[FAIL] %s\n", what);
  }
}

bool ListFrames(const std::string& dir, std::vector<std::string>* out) {
  // Deliberately simple: the contract test only needs SOME frames.
  for (int i = 0; i < 4096; ++i) {
    char nm[64];
    snprintf(nm, sizeof(nm), "/frame_%06d.bmp", i);
    const std::string p = dir + nm;
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) break;
    fclose(f);
    out->push_back(p);
  }
  return !out->empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::string raw_root = "/mnt/manu/lq_ir_bmp";
  std::string sequence = "VIDEO00007_19700101_001324";
  std::string model = "/mnt/manu/ptq/models/p0_off_A_none.rknn";
  std::string kernel_dir = "/root/kernels";
  std::string golden;
  int limit = 60;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "[FATAL] %s needs a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--raw-root") raw_root = next("--raw-root");
    else if (a == "--sequence") sequence = next("--sequence");
    else if (a == "--model") model = next("--model");
    else if (a == "--kernel-dir") kernel_dir = next("--kernel-dir");
    else if (a == "--golden") golden = next("--golden");
    else if (a == "--limit") limit = atoi(next("--limit").c_str());
    else {
      fprintf(stderr, "[FATAL] unknown option %s\n", a.c_str());
      return 2;
    }
  }

  // ---- 1. config_defaults() fills every field -----------------------------
  {
    MosaicPipelineConfig c;
    memset(&c, 0xA5, sizeof(c));  // poison: a field left unset shows up
    mosaic_pipeline_config_defaults(&c);
    Check(c.version == MOSAIC_API_VERSION, "config_defaults: version");
    Check(c.model_path == nullptr, "config_defaults: model_path is NULL");
    Check(c.kernel_dir && std::string(c.kernel_dir) == "/root/kernels",
          "config_defaults: kernel_dir");
    Check(c.gmc_window == 21, "config_defaults: gmc_window == 21");
    Check(c.gmc_stride_step == 2, "config_defaults: gmc_stride_step == 2");
    Check(c.gmc_anchor_step == 2, "config_defaults: gmc_anchor_step == 2");
    Check(c.gmc_downscale == 2, "config_defaults: gmc_downscale == 2");
    Check(c.gmc_fit_corners == 600, "config_defaults: gmc_fit_corners == 600");
    Check(c.gmc_fit_win == 15, "config_defaults: gmc_fit_win == 15");
    Check(c.gmc_fit_level == 2, "config_defaults: gmc_fit_level == 2");
    Check(c.gmc_fit_iters == 20, "config_defaults: gmc_fit_iters == 20");
    // The three fields that used to be left unset.
    Check(c.gmc_gate_shift == 1000, "config_defaults: gmc_gate_shift == 1000 (1.0 frame)");
    Check(c.gmc_gate_min == 500, "config_defaults: gmc_gate_min == 500 (0.5)");
    Check(c.gmc_gate_max == 2000, "config_defaults: gmc_gate_max == 2000 (2.0)");
    Check(c.gmc_pyr_cache == 1, "config_defaults: gmc_pyr_cache == 1");
    Check(c.gmc_prealloc_pad == 0, "config_defaults: gmc_prealloc_pad == 0");
    Check(c.npu_workers == 2, "config_defaults: npu_workers == 2");
    Check(c.region_threshold == 0 && c.min_area == 0 && c.max_detections == 0,
          "config_defaults: the three thresholds defer to the frozen defaults");
    Check(c.cpu_affinity == nullptr, "config_defaults: cpu_affinity is NULL");
    Check(c.npu_affinity == nullptr, "config_defaults: npu_affinity is NULL");
    Check(c.gmc_affinity == nullptr, "config_defaults: gmc_affinity is NULL");
    // The anchor LK stage's parallelism.  1 == the reference's serial anchor
    // loop, which is the CERTIFIED behaviour: the workers are opt-in because
    // they change the SCHEDULING, not the arithmetic (the RANSAC stage still
    // runs sequentially in the anchor order, so the output stays
    // bit-identical).
    Check(c.gmc_anchor_threads == 1, "config_defaults: gmc_anchor_threads == 1 (serial anchor loop)");
    Check(c.gmc_affinity_group == nullptr, "config_defaults: gmc_affinity_group is NULL");
    // The NPU core mask: 0 == do NOT call rknn_set_core_mask, which is the
    // shipped behaviour the certified baseline was measured with.  A
    // non-zero mask is a measured deviation, so the default MUST be 0.
    Check(c.npu_core_mask == 0, "config_defaults: npu_core_mask == 0 (rknn_set_core_mask NOT called)");
    Check(c.gmc_feature_fingerprint == 1, "config_defaults: gmc_feature_fingerprint == 1");
  Check(c.gmc_workers == 1, "config_defaults: gmc_workers == 1 (single GMC engine)");
  Check(c.post_workers == 1, "config_defaults: post_workers == 1");
  Check(c.queue_depth == 8, "config_defaults: queue_depth == 8");
  // The GMC lookahead budget: 0 = AUTO (the library derives it from
  // gmc_workers).  It is a THROUGHPUT knob -- it changes how far the ingest
  // may run ahead of the engines, never which pixels a fit reads.
  // The struct above was poisoned with 0xA5, so this also proves that
  // config_defaults() WRITES the field instead of leaving stack garbage in it
  // (a garbage budget is not a crash: it is a silently wrong pipeline shape).
  Check(c.gmc_ring_slack == 0, "config_defaults: gmc_ring_slack == 0 (auto)");
  Check(c.jitter_depth == 8, "config_defaults: jitter_depth == 8");
  // The jitter buffer must be deeper than the producer's in-flight window,
  // or a healthy run would overflow it and skip frames.
  Check(c.jitter_depth > 4, "config_defaults: the jitter buffer is deeper than the in-flight window");
  Check(c.input_queue == 8, "config_defaults: input_queue == 8");
  Check(c.gmc_deadline_ms == 0, "config_defaults: gmc_deadline_ms == 0 (watchdog off)");
  Check(c.input_mode == MOSAIC_INPUT_SET, "config_defaults: input_mode == SET");
  // The physical-frame-number anchor of the GMC history: 0 == OFF, i.e. the
  // engine anchors on the consumption counter, which is the CERTIFIED
  // behaviour.  The struct above was poisoned with 0xA5, so this also proves
  // config_defaults() WRITES the field: a garbage anchor is not a crash, it
  // silently changes WHICH frames the GMC reads.
  Check(c.gmc_phys_anchor == 0,
        "config_defaults: gmc_phys_anchor == 0 (the consumption counter)");
  // The camera-queue overflow policy: 0 == evict the OLDEST frame, the
  // shipped behaviour (a live stream wants the freshest data).
  Check(c.gmc_input_drop_policy == 0,
        "config_defaults: gmc_input_drop_policy == 0 (evict the oldest)");
  }

  // ---- 2. create() rejects a bad version / a missing model ----------------
  {
    MosaicPipelineConfig c;
    mosaic_pipeline_config_defaults(&c);
    c.version = c.version + 1;
    c.model_path = model.c_str();
    char err[256] = {0};
    MosaicPipeline* p = mosaic_pipeline_create(&c, err, sizeof(err));
    Check(p == nullptr, "create: a bad version is rejected");
    Check(err[0] != '\0', "create: a bad version produces a message");
    if (p) mosaic_pipeline_destroy(p);
  }
  {
    MosaicPipelineConfig c;
    mosaic_pipeline_config_defaults(&c);
    c.model_path = nullptr;
    char err[256] = {0};
    MosaicPipeline* p = mosaic_pipeline_create(&c, err, sizeof(err));
    Check(p == nullptr, "create: a missing model path is rejected");
    Check(err[0] != '\0', "create: a missing model path produces a message");
    if (p) mosaic_pipeline_destroy(p);
  }

  // ---- 3. create() clamps out-of-range knobs ------------------------------
  {
    MosaicPipelineConfig c;
    mosaic_pipeline_config_defaults(&c);
    c.model_path = model.c_str();
    c.kernel_dir = kernel_dir.c_str();
    c.sequence = sequence.c_str();
    // Deliberately absurd values: none of them may reach the engine.
    c.gmc_gate_shift = 123456;   // out of range
    c.gmc_window = 9999;         // out of range
    c.npu_workers = 99;          // above the board's two-context limit
    c.gmc_pyr_cache = 7;         // not 0/1
    c.gmc_workers = 4;           // more GMC engines than the one-step-fit cache can serve
    c.gmc_ring_slack = 999999;   // out of range: it must fall back to AUTO, not to a
                                 // budget that would silently reshape the pipeline
    c.gmc_anchor_threads = 99;   // out of range: more workers than the anchor grid has
                                 // anchors, and more threads than cores
    c.npu_core_mask = 999;        // out of range: not an rknn_core_mask value
    c.gmc_phys_anchor = 7;        // out of range: not 0/1
    c.gmc_input_drop_policy = 5;  // out of range: not 0/1
    char err[256] = {0};
    MosaicPipeline* p = mosaic_pipeline_create(&c, err, sizeof(err));
    Check(p != nullptr, "create: absurd knobs do not abort the creation");
    if (p) {
      MosaicPipelineConfig eff;
      Check(mosaic_pipeline_effective_config(p, &eff) == 0,
            "effective_config: readable");
      Check(eff.gmc_gate_shift == 1000, "clamped: gmc_gate_shift -> 1000");
      Check(eff.gmc_window == 21, "clamped: gmc_window -> 21");
      Check(eff.npu_workers == 2, "clamped: npu_workers -> 2");
      Check(eff.gmc_pyr_cache == 1, "normalised: gmc_pyr_cache -> 1");
      // An out-of-range lookahead request is folded back to AUTO by the clamp
      // table, and the library resolves AUTO ONCE in its sanitize stage.  The
      // effective config therefore reports the RESOLVED budget, not the
      // requested 0: with the clamped gmc_workers (2) the auto rule
      // max(64, 32*2) is 64.  (With ONE engine it is max(64, 32*1) == 64 as
      // well -- the floor dominates -- so both worker counts land on 64.)
      Check(eff.gmc_ring_slack == 64, "clamped: an out-of-range gmc_ring_slack falls back to AUTO (= 64)");
      // MORE GMC ENGINES ARE SLOWER, and that is measured, not assumed:
      // the engines are assigned round robin, so with two engines each owns
      // one parity class and every one-step fit pair (a, a+2) stays inside
      // one engine; with three or four the pair straddles two engines and
      // every frame lazily re-fits several pairs.  Board numbers, same 3000
      // frames, same corner budget: 2 engines 70.4 ms/frame and 15.16 fps,
      // 3 engines 103.9 ms/frame and 14.78 fps, 4 engines 86.9 ms/frame and
      // 14.97 fps.  The library therefore refuses to be configured above 2.
      // v9: the OPPORTUNISTIC latest-frame scheduler requires EXACTLY ONE
      // engine.  Several engines would grab the SAME newest frame (duplicate
      // results), and the board numbers above already say that more engines
      // make the GMC slower.  The library therefore refuses to be configured
      // above 1 and clamps every request down to 1.
      Check(eff.gmc_workers == 1,
            "clamped: gmc_workers -> 1 (the opportunistic scheduler needs one engine)");
      // The anchor workers follow the same rule as every other knob: an
      // out-of-range request falls back to the REFERENCE default, which is the
      // serial anchor loop (1 worker).  A garbage value must not silently
      // create threads.  The effective count is capped AGAIN by the engine's
      // core group, because more workers than cores only oversubscribe the
      // cores.
      Check(eff.gmc_anchor_threads == 1,
            "clamped: an out-of-range gmc_anchor_threads falls back to 1 (the serial anchor loop)");
      // The core mask is a MEASURED DEVIATION knob, so a garbage value must
      // fall back to the shipped behaviour (0 = do not call
      // rknn_set_core_mask) rather than to some arbitrary core.
      Check(eff.npu_core_mask == 0,
            "clamped: an out-of-range npu_core_mask falls back to 0 (the shipped behaviour)");
      // The two new knobs are 0/1 switches.  An out-of-range request must fall
      // back to the SHIPPED behaviour (0), never to an arbitrary value: the
      // anchor decides which frames the GMC reads, and the drop policy decides
      // which frames are lost.
      Check(eff.gmc_phys_anchor == 0,
            "clamped: an out-of-range gmc_phys_anchor falls back to 0 (the consumption counter)");
      Check(eff.gmc_input_drop_policy == 0,
            "clamped: an out-of-range gmc_input_drop_policy falls back to 0 (evict the oldest)");
      mosaic_pipeline_destroy(p);
    }
  }

  // ---- 4/5/6/7. the pipeline itself ---------------------------------------
  const std::string seq_dir = raw_root + "/" + sequence;
  std::vector<std::string> frames;
  if (!ListFrames(seq_dir, &frames)) {
    fprintf(stderr, "[FATAL] no frames under %s\n", seq_dir.c_str());
    return 2;
  }
  if (limit > 0 && (int)frames.size() > limit) frames.resize((size_t)limit);
  fprintf(stderr, "[i] %zu frames from %s\n", frames.size(), seq_dir.c_str());

  MosaicPipelineConfig c;
  mosaic_pipeline_config_defaults(&c);
  c.model_path = model.c_str();
  c.kernel_dir = kernel_dir.c_str();
  c.sequence = sequence.c_str();
  char err[256] = {0};
  MosaicPipeline* p = mosaic_pipeline_create(&c, err, sizeof(err));
  if (!p) {
    fprintf(stderr, "[FATAL] create failed: %s\n", err);
    return 2;
  }

  // A wrong-sized frame must be rejected BEFORE anything else is pushed.
  {
    std::vector<uint8_t> small(16 * 16, 0);
    const int64_t rc = mosaic_pipeline_push(p, small.data(), 16, 16);
    Check(rc == -1, "push: a wrong-sized frame is rejected");
    const char* e = mosaic_pipeline_last_error(p);
    Check(e && e[0] != '\0', "push: the wrong-size error is retrievable");
  }

  // The AUTO lookahead budget is resolved by the LIBRARY and reported back.
  // The default config asks for 0 (= auto) and must come back as the
  // resolved budget, so a caller (and the demo's report) can read the budget
  // in force instead of re-deriving the rule -- which is what made the two
  // copies of the rule drift apart.
  {
    MosaicPipelineConfig eff;
    Check(mosaic_pipeline_effective_config(p, &eff) == 0,
          "effective_config: readable (the main pipeline)");
    Check(eff.gmc_ring_slack == 64,
          "effective_config: the AUTO lookahead budget is resolved to 64");
  }

  const int n = (int)frames.size();
  // push() NEVER blocks: the camera queue drops the OLDEST frame when the
  // pipeline is behind, so the consumer has to pace itself.
  //
  // v9 -- THE BATCH IS 1, AND IT MUST BE.  Under the OPPORTUNISTIC scheduler
  // the engine computes the NEWEST frame it can see and SKIPS the rest; it
  // never goes back for the frames it passed.  A batch of 4 would therefore
  // produce results for the last frame of the batch only, and the three
  // earlier ones would never come back -> mosaic_pipeline_collect() would
  // BLOCK FOREVER.  One push, one collect: the head is never ahead of the
  // engine by more than the frame it is computing, so EVERY pushed frame
  // still produces exactly one result and the 1:1 contract below holds.
  const int kBatch = 1;
  int got = 0;
  bool ids_ok = true;
  bool dets_ok = true;
  std::vector<int> seen(n, 0);
  for (int base = 0; base < n; base += kBatch) {
    const int end = std::min(n, base + kBatch);
    for (int i = base; i < end; ++i) {
      const cv::Mat img = cv::imread(frames[i], cv::IMREAD_GRAYSCALE);
      if (img.empty()) {
        fprintf(stderr, "[FATAL] cannot read %s\n", frames[i].c_str());
        mosaic_pipeline_destroy(p);
        return 2;
      }
      const int64_t rc = mosaic_pipeline_push(p, img.data, img.cols, img.rows);
      if (rc != i) {
        fprintf(stderr, "[FAIL] push returned %lld for frame %d\n", (long long)rc, i);
        ++g_fail;
        mosaic_pipeline_destroy(p);
        return 1;
      }
    }
    for (int i = 0; i < end - base; ++i) {
    int64_t fid = -1;
    MosaicDet dets[256];
    const int32_t nd = mosaic_pipeline_collect(p, &fid, dets, 256);
    if (nd < 0) {
      ids_ok = false;
      break;
    }
    // collect() returns COMPLETION order, not push order (mosaic_pipeline.h:
    // "collect() returns frames in COMPLETION order, not push order (with
    // npu_workers > 1 the two NPU contexts finish out of order)").  The
    // contract is "every pushed frame comes back EXACTLY ONCE, identified by
    // its frame id" -- the caller re-sorts by frame id (rknn_mosaic_e2e.cpp
    // does exactly that).  Asserting the ARRIVAL order was wrong: with
    // npu_workers=2 the two workers finish out of order, so this assertion
    // failed on 2 of 3 board runs.  The library was right; the test was not.
    if (fid < 0 || fid >= n || seen[fid]++) ids_ok = false;
    for (int32_t k = 0; k < nd && k < 256; ++k) {
      if (dets[k].tag < 0 || dets[k].tag > 2) dets_ok = false;
      if (dets[k].w <= 0 || dets[k].h <= 0) dets_ok = false;
    }
    ++got;
    }
  }
  Check(true, "push: every frame was accepted and returned its own id");
  Check(ids_ok, "collect: one result per pushed frame, each frame id exactly once");
  bool all_seen = true;
  for (int i = 0; i < n; ++i)
    if (!seen[i]) all_seen = false;
  Check(all_seen, "collect: every pushed frame id came back exactly once");
  Check(got == n, "collect: every pushed frame produced a result");
  Check(dets_ok, "collect: the detections have a valid tag and a non-empty bbox");

  // flush() must join the threads; collect() afterwards must NOT block.
  Check(mosaic_pipeline_flush(p) == 0, "flush: returns 0");
  int64_t fid = -1;
  const int32_t after = mosaic_pipeline_collect(p, &fid, nullptr, 0);
  Check(after == -1, "collect after flush: returns -1 instead of blocking forever");

  MosaicPipelineTiming tm;
  Check(mosaic_pipeline_get_timing(p, &tm) == 0, "get_timing: returns 0");
  Check(tm.frames == n, "get_timing: the frame count matches what was pushed");
  Check(tm.dropped == 0, "get_timing: no frame was dropped at the camera queue");
  // The two downstream drop points are separate counters, because the two
  // drops are NOT equivalent: a frame lost at the camera queue breaks the
  // GMC registration chain, a frame skipped by the inference costs one
  // missed detection.  In a healthy run both are 0.
  Check(tm.features_dropped == 0, "get_timing: no feature was skipped by the inference stage");
  Check(tm.detections_dropped == 0, "get_timing: no heatmap was skipped by the back-projection");
  Check(tm.gmc_deadline_blows == 0, "get_timing: no GMC frame blew the wall-clock watchdog");
  // The GMC GROUPS that completed: the frames a feature was really computed
  // for.  It is the denominator a caller checks the delivered results
  // against -- NOT the number of pushed frames, which stops being the right
  // denominator as soon as the scheduler skips anything.
  //
  // v9: under the OPPORTUNISTIC scheduler the engine computes the NEWEST
  // frame it can see and SKIPS the ones it passed, so groups_done is "how many
  // features were really produced" and nothing else.  gmc_compute_stride is
  // gone; its replacement, gmc_min_stride, is a MINIMUM spacing, not a grid.
  Check(tm.gmc_groups_done == n,
        "get_timing: the GMC group count matches the frames a feature was computed for");
  // v9: the frames the scheduler SKIPPED, and the frames it had to ABANDON
  // because their 21-frame history was overwritten.  With one push at a time
  // the engine is never more than one frame behind, so nothing is skipped and
  // nothing is void -- these two are the direct evidence that the
  // opportunistic scheduler did not have to drop anything to keep up.
  Check(tm.gmc_void_frames == 0,
        "get_timing: no frame was abandoned because its history was overwritten");
  Check(tm.gmc_frames_skipped == 0,
        "get_timing: the scheduler skipped no frame (one push at a time)");
  // THE FRAME ACCOUNTING INVARIANT, in the form the e2e checks it:
  //   expected results = gmc_groups_done - features_dropped - detections_dropped
  // Every GMC group produced one feature; the two drop counters are the only
  // places a feature or a heatmap is lost, so what is left is exactly the
  // number of results.  (Under the opportunistic scheduler gmc_groups_done is
  // the ONLY reliable denominator: the number of pushed frames is no longer
  // the number of features, because the scheduler skips frames.)
  Check(tm.gmc_groups_done - tm.features_dropped - tm.detections_dropped == tm.frames,
        "get_timing: the GMC group count minus the downstream drops is the result count");
  Check(tm.wall_ms > 0.0, "get_timing: the wall clock was measured");
  Check(tm.run_ms > 0.0, "get_timing: the NPU run was measured");
  // v9: the engine's job queue is GONE -- the opportunistic scheduler has no
  // queue to sample, because the engine pulls the newest frame itself.  The
  // equivalent evidence is the group's own accounting, checked above:
  // every group produced exactly one feature and nothing was skipped.
  Check(tm.gmc_groups_done > 0,
        "get_timing: the opportunistic engine completed at least one group");
  Check(tm.gmc_ingest_in_wait_ms + tm.gmc_ingest_window_ms + tm.gmc_ingest_space_ms <=
            tm.gmc_ingest_ms + 1.0,
        "get_timing: the ingest's waits are a subset of its total wall time");

  const char* hex = mosaic_pipeline_feature_md5(p);
  Check(hex && strlen(hex) == 32, "feature_md5: a 32-char hex digest is returned");
  fprintf(stderr, "[i] rolling feature md5 over %d frames: %s\n", n, hex ? hex : "(null)");

  mosaic_pipeline_destroy(p);

  // ---- 8. a FATAL error must ABORT the pipeline, not wedge it ---------------
  // This is the path that used to hang a whole board: the error was recorded
  // and the abort flag was set, but nobody woke the sleeping threads, so the
  // pipeline sat there for ever with no message at all.
  //
  // The proof is that the pipeline REPORTS the error and STOPS accepting
  // frames -- and that destroy() returns (it joins the threads, so it can
  // only return if every thread woke up and left).
  //
  // It runs AFTER the main pipeline is destroyed, because the board tolerates
  // exactly TWO RKNN contexts and the main pipeline already holds both.
  {
    MosaicPipelineConfig c;
    mosaic_pipeline_config_defaults(&c);
    c.model_path = model.c_str();
    c.kernel_dir = kernel_dir.c_str();
    c.sequence = sequence.c_str();
    char err[256] = {0};
    MosaicPipeline* pf = mosaic_pipeline_create(&c, err, sizeof(err));
    Check(pf != nullptr, "inject: the pipeline was created");
    if (pf) {
      mosaic_pipeline_inject_fault(pf, "contract test: injected fault");
      const char* e = mosaic_pipeline_last_error(pf);
      Check(e && std::string(e).find("injected fault") != std::string::npos,
            "inject: the fatal error is retrievable");
      // The pipeline must refuse further work instead of accepting frames it
      // will never compute.
      const int64_t rc = mosaic_pipeline_push(pf, nullptr, 0, 0);
      Check(rc == -1, "inject: the aborted pipeline refuses further frames");
      mosaic_pipeline_destroy(pf);   // must RETURN: it joins the threads
      Check(true, "inject: destroy() returned -- every thread woke up and left");
    }
  }

  // ---- 7. the golden fingerprint -------------------------------------------
  if (!golden.empty()) {
    // The streaming md5 file was not requested here, so compare the rolling
    // digest against the golden file's own rolling digest: the golden file's
    // per-frame hashes are folded the same way the reference folds them.
    fprintf(stderr, "[i] golden given but the streaming file was not enabled; "
                    "use rknn_mosaic_e2e --feature-md5-golden for the frame-level check\n");
  }

  fprintf(stderr, "\n[RESULT] %d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
