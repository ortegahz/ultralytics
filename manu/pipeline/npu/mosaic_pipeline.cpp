// ===========================================================================
// mosaic_pipeline.cpp -- the RK3588 mosaic pipeline behind the public C ABI.
//
// ARCHITECTURE: every stage is a SERVICE.
//
//   push(raw gray)  ->  [SERVICE GMC]   N engines, each computes INTERLEAVED
//                          frames.  The per-frame computation depends only on
//                          PIXELS (the anchor fits compare frame `abs` with
//                          frame `abs-lag`; the other lags are composed from
//                          those fits inside the same frame), so several
//                          engines can work on DIFFERENT frames without
//                          touching each other.
//                     ->  [SERVICE NPU]   2 independent services: each owns its
//                          own RKNN context, its own RGA canvas and its own
//                          core affinity.  RGA compose + upload + rknn_run.
//                     ->  [SERVICE POST]  back-projection + 8-connected
//                          components, off the NPU thread so it overlaps the
//                          inference.
//                     ->  callback / collect()
//
// THE SYNCHRONISATION RULES (the part that is easy to get wrong):
//
//  1. THE HISTORY WINDOW.  The GMC engine's history ring is max_lag+1 = 43
//     deep and a fit of frame J reads frames J-42..J.  Seeding frame S
//     overwrites the slot that held S-ring_depth.  So the producer of the
//     history (the ingest) may not run ahead of the engine that consumes it:
//     it may be at most `ring_slack` frames ahead of the OLDEST frame any
//     engine still owes.  Without the slack the fit would silently use
//     OVERWRITTEN pixels -- wrong numbers, no crash, no error.
//
//  2. THE FRAME NUMBER IS THE PUSH ORDER.  The camera may drop frames (the
//     camera-facing queue drops the OLDEST one when the pipeline is behind),
//     so the engine's absolute index counts the frames it RECEIVED, not the
//     source frame numbers.  The source frame id travels with the frame.
//
//  3. ONLY THE CAMERA QUEUE DROPS.  push() never blocks: a 25 fps camera must
//     not be stalled by a pipeline that cannot keep up.  The queues BETWEEN
//     the services apply back-pressure instead -- the pipeline paces itself,
//     the camera never does.
//
//  4. THE RESULT QUEUE IS UNBOUNDED.  The caller may push the whole sequence
//     and collect afterwards; a bounded result queue would deadlock that.
//
//  5. THE ORDER-SENSITIVE OUTPUTS (the rolling md5, the --mats-all dump) are
//     emitted by a dedicated writer thread in FRAME order, because the
//     engines finish out of order.
// ===========================================================================

// The GMC engine: compiled with its main() out, exactly like e2e_core.cpp does.
#define GMC_STREAM_OCL_NO_MAIN
#include "../cpp/gmc_stream_ocl.cpp"

#include "mosaic_core.h"
#include "mosaic_pipeline.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// The items that travel between the services.
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
// ---------------------------------------------------------------------------
struct InItem {
  int64_t id = 0;
  cv::Mat gray;
};

// One frame assigned to one GMC engine.
struct GmcJob {
  int64_t id = 0;
  long abs = 0;  // the engine's absolute frame index
  // THE ARTIFACT SEQUENCE: the DENSE index of the artifact this job will
  // produce.  It is NOT the consumption counter `abs`: with a cadence of N
  // only every Nth frame produces an artifact, so `abs` has GAPS -- and the
  // order-sensitive writer waits for the next CONTIGUOUS index, so it would
  // stall forever on the first frame that produced none.
  long art_seq = 0;
  cv::Mat gray;
};

// Feature frame -> NPU service.  Model order [residual, diff, I_t].
struct FeatureItem {
  int64_t id = 0;
  std::vector<uint8_t> rgb;
};

// Heatmap -> post-processing service.
struct HeatItem {
  int64_t id = 0;
  mosaic::Session* sess = nullptr;  // the NPU service that produced it
  std::vector<float> hm;
};

// Finished frame -> the caller.
struct ResultItem {
  int64_t id = 0;
  std::vector<mosaic::Comp> comps;
};

// Per-frame artifacts that must be emitted in FRAME ORDER: the rolling md5
// and the --mats-all dump.
struct OrderedArtifact {
  int64_t id = 0;
  std::string md5_hex;
  std::vector<Matx23f> mats;
};

}  // namespace

// ---------------------------------------------------------------------------
// The implementation behind the opaque handle.
// ---------------------------------------------------------------------------
struct MosaicPipeline {
  MosaicPipelineConfig cfg{};
  std::string last_error;

  // ---- SERVICE: GMC ------------------------------------------------------
  struct Engine {
    std::unique_ptr<FusedGpu> gpu;
    // The engine is not copyable (it owns the estimator), so it is held by
    // pointer and built in place.
    std::unique_ptr<OnlineFeaturePipeline> pipe;
    std::deque<GmcJob> jobs;
    std::mutex m;
    std::condition_variable cv_job;   // a job arrived
    std::condition_variable cv_space;  // the queue has room
    // The oldest absolute frame this engine still owes.  `has_pending` is
    // false while the engine is idle.
    long oldest_pending = 0;
    bool has_pending = false;
    long seq_frame = 0;
    // The engine's corner counters are CUMULATIVE, so the per-frame corner
    // count is a difference against the previous frame.  These two hold the
    // previous cumulative values.
    double prev_corner_dets = 0.0;
    double prev_corner_pts = 0.0;
    std::thread th;
  };
  std::vector<std::unique_ptr<Engine>> engines;
  std::thread ingest_thread;

  // The camera-facing queue.  Bounded, and it DROPS the oldest frame rather
  // than stalling the camera.
  std::deque<InItem> in_q;
  std::mutex m_in;
  std::condition_variable cv_in;
  int64_t next_id = 0;
  long abs_counter = 0;
  long ring_slack = 0;  // extra history slots = the lookahead the ingest may use

  // THE JITTER BUFFER: GMC service -> NPU services.  Its depth is the
  // jitter_depth knob; the GMC may run that many frames ahead of the NPU,
  // so an expensive GMC frame is absorbed HERE instead of reaching the NPU
  // as a stall.
  std::deque<FeatureItem> feat_q;
  std::mutex m_feat;
  std::condition_variable cv_feat;        // a feature arrived

  // ---- SERVICE: NPU ------------------------------------------------------
  struct NpuService {
    std::unique_ptr<mosaic::Session> sess;
    std::thread th;
    int index = 0;   // its position in the service list (for the core list)
  };
  std::vector<std::unique_ptr<NpuService>> npus;

  // ---- per-service CPU placement -----------------------------------------
  // Empty = inherit the process affinity.  Otherwise the Nth thread of that
  // service is pinned to cpus[N % cpus.size()].
  std::vector<int> gmc_cpus;
  std::vector<int> npu_cpus;
  // The GMC engines' core GROUPS (one group per engine, ';'-separated in the
  // config).  A group is a set of cores the engine's own thread AND its
  // anchor workers are spread over.  Empty = the engines are placed by
  // gmc_cpus (one core each) or inherit the process affinity.
  std::vector<std::vector<int>> gmc_groups;

  // ---- the ingest's own accounting ----------------------------------------
  int64_t ingest_window_blocks = 0;
  int64_t ingest_space_blocks = 0;

  // ---- SERVICE: POST -----------------------------------------------------
  std::deque<HeatItem> heat_q;
  std::mutex m_heat;
  std::condition_variable cv_heat;        // a heatmap arrived
  std::vector<std::thread> post_threads;
  std::vector<std::unique_ptr<mosaic::PostWorkspace>> post_ws;

  // ---- the results --------------------------------------------------------
  // out_q feeds mosaic_pipeline_collect(); cb_q feeds the callback.  A frame
  // goes to exactly one of them, decided when the result is produced.
  std::deque<ResultItem> out_q;
  std::deque<ResultItem> cb_q;
  std::mutex m_out;
  std::condition_variable cv_out;
  MosaicResultCallback cb = nullptr;
  void* cb_user = nullptr;
  std::thread dispatch_thread;

  // ---- the order-sensitive outputs ---------------------------------------
  std::map<long, OrderedArtifact> artifacts;
  std::mutex m_art;
  std::condition_variable cv_art;
  long next_artifact = 1;
  // The dense counter behind `next_artifact`: it is incremented once per
  // frame that ACTUALLY produces an artifact, so the writer's index stays
  // contiguous even when the computation cadence skips frames.
  long artifact_seq = 0;
  std::thread art_thread;

  // The history-window guard.  The ingest waits on this until the engines
  // have caught up far enough for the next frame to be seeded safely.
  std::mutex m_guard;
  std::condition_variable cv_guard;

  // ---- lifecycle ----------------------------------------------------------
  std::atomic<bool> eof{false};
  std::atomic<bool> abort{false};
  std::atomic<bool> flushed{false};
  std::atomic<bool> ingest_done{false};
  std::atomic<bool> gmc_done{false};
  std::atomic<bool> feat_done{false};

  // The rolling MD5 over the PRE-ENCODE channel planes -- the same object and
  // the same recipe as the reference implementation's --md5-out.  It is
  // folded in FRAME order by the artifact writer.
  gmcpp::MD5 seq_md5;
  // The hex string must OWN the bytes: returning hex().c_str() would hand
  // out a pointer into a temporary std::string that dies with the statement.
  std::string seq_md5_hex;
  std::mutex m_md5;
  FILE* md5_out = nullptr;
  std::string md5_path;
  // Optional feature dump (see mosaic_pipeline_set_feature_dump_out).
  std::string dump_dir;
  // Optional dump of the GMC transform chain (see mosaic_pipeline_set_mats_dump_out).
  FILE* mats_out = nullptr;
  std::string mats_path;

  int32_t width = 0;
  int32_t height = 0;

  // ---- timing -------------------------------------------------------------
  MosaicPipelineTiming timing{};
  std::mutex m_time;
  // One sample per frame the GMC service fitted: the cost of fitting ONE
  // frame.  The distribution over these is what a jitter buffer has to
  // absorb and what a watchdog has to bound.
  std::vector<double> gmc_fit_samples;
  // One sample per NPU forward pass: the cost of the SINGLE rknn_run.  The
  // NPU is the pipeline's critical path, and the core-mask experiment is
  // judged on this distribution -- not on the mean alone.
  std::vector<double> rknn_run_samples;
  double gmc_corner_pts = 0.0;   // sum of the per-detection corner counts
  double gmc_corner_dets = 0.0;  // sum of the per-detection counts

  size_t queue_depth = 8;
  size_t jitter_depth = 8;
  size_t input_depth = 8;
  // The ENGINE JOB QUEUE's depth.  It is NOT queue_depth: queue_depth sizes
  // the queues BETWEEN the services (the heat queue), while this one sizes
  // the queue of frames an engine has been ASSIGNED but has not computed yet.
  // It follows the lookahead budget, because a lookahead the job queue cannot
  // hold is a lookahead the pipeline cannot use: the ingest would be blocked
  // by the space guard exactly when the window guard would have released it.
  size_t gmc_job_depth = 8;

  // A FATAL error: the pipeline cannot continue.  Aborts every thread.
  //
  // WHY THE NOTIFY BELOW IS NOT OPTIONAL (measured: it cost a hung board)
  //   Every service thread waits on a condition variable whose predicate
  //   mentions `abort` -- but a predicate is only RE-EVALUATED when the thread
  //   is woken.  Setting the flag without waking anyone leaves every other
  //   thread asleep FOREVER: the NPU service reports "rknn_run failed" and
  //   exits, the ingest sits in the window guard, the engines sit in their job
  //   wait, and the caller waits for a result that will never come.  The
  //   pipeline then hangs with no error message at all -- the only way out is
  //   killing the process.  (That is exactly what two overlapping e2e runs
  //   did: they took four RKNN contexts, the NPU refused the submits, and the
  //   whole run wedged silently at frame 155.)
  //
  //   So: set the flag AND wake everybody.  The predicates all mention
  //   `abort`, so every woken thread re-evaluates its predicate and leaves.
  void set_error(const std::string& e) {
    {
      std::lock_guard<std::mutex> lk(m_out);
      if (last_error.empty()) last_error = e;
    }
    abort = true;
    wake_everyone();
  }

  // Wake every thread that can be asleep on a condition variable, so each of
  // them re-evaluates its predicate (they all mention `abort`) and exits.
  //
  // notify_all() does not take the mutex, so this is safe to call from
  // anywhere -- including from a thread that is about to release one of
  // them.  The predicates make the spurious wakeups harmless.
  void wake_everyone() {
    cv_in.notify_all();
    for (auto& e : engines) {
      e->cv_job.notify_all();
      e->cv_space.notify_all();
    }
    cv_feat.notify_all();
    cv_heat.notify_all();
    cv_guard.notify_all();
    cv_art.notify_all();
    cv_out.notify_all();
  }

  // A PER-FRAME error: this frame is rejected, the stream keeps running.
  // A camera stream must not die because one frame was malformed -- and a
  // malformed frame must not stall the camera either.
  void note_error(const std::string& e) {
    std::lock_guard<std::mutex> lk(m_out);
    if (last_error.empty()) last_error = e;
  }

  // -------------------------------------------------------------------------
  // THE HISTORY WINDOW GUARD.
  //
  // The ingest may seed frame `abs` only if every engine still owes at most
  // `ring_slack` frames of backlog: seeding `abs` overwrites the slot that
  // held `abs - ring_depth`, and the engine that computes frame J still
  // needs frames J-42..J.  See the header comment, rule 1.
  // -------------------------------------------------------------------------
  bool window_ok(long abs) const {
    for (const auto& e : engines) {
      std::lock_guard<std::mutex> lk(e->m);
      if (e->has_pending && abs - e->oldest_pending > ring_slack) return false;
    }
    return true;
  }

  // -------------------------------------------------------------------------
  // THE INGEST: raw frames -> the engines' history rings + the job queues.
  //
  // It is the only thread that writes the history, so the ring state has a
  // single writer.  It seeds EVERY engine with EVERY frame (each engine may
  // need any frame as history) and hands the frame to the engine that will
  // compute it.
  // -------------------------------------------------------------------------
  void ingest_loop() {
    for (;;) {
      const auto t_ingest0 = std::chrono::steady_clock::now();
      // The ingest's own accounting of THIS frame: how long it waited for the
      // camera, for the lookahead budget and for room in the job queue.  They
      // are accumulated TOGETHER WITH gmc_ingest_ms at the end of the frame,
      // so the three waits plus the ingest's own work always add up to the
      // ingest's total wall time.  A frame that was aborted half-way is not
      // accounted for at all -- it never became a GMC job.
      double in_wait_ms = 0.0;
      double win_ms = 0.0;
      double space_ms = 0.0;
      // The round-robin cursor over the GMC GROUPS (not over the frames) --
      // see the computation cadence below.
      long group_counter = 0;
      InItem item;
      {
        std::unique_lock<std::mutex> lk(m_in);
        cv_in.wait(lk, [this] { return !in_q.empty() || eof || abort; });
        if (abort) break;
        if (in_q.empty()) {
          if (eof) break;
          continue;
        }
        // The camera queue wait: how long the ingest stood there because the
        // camera had not delivered yet.  A LARGE value is the healthy case
        // (the pipeline is faster than the source and the camera paces it);
        // a small one means the ingest was the bottleneck.
        in_wait_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ingest0)
                .count();
        item = std::move(in_q.front());
        in_q.pop_front();
      }
      // A slot freed in the camera queue: let a blocked push() proceed.
      cv_in.notify_all();

      const long abs = ++abs_counter;
      // THE INDEX THE ENGINE ANCHORS ITS HISTORY ON.
      //
      // The consumption counter (abs) and the camera's PHYSICAL frame number
      // (item.id) are the same sequence as long as no frame is lost, and they
      // diverge exactly when the camera queue overflows.  The engine is told
      // which one to use: the consumption counter (the certified behaviour) or
      // the physical frame number (which makes a lost frame a DETECTED hole).
      //
      // NOTE the split of the two roles: `abs` stays the DENSE counter, because
      // it orders the artifacts (the rolling md5 and the mats dump are
      // emitted in frame order, and that order must not have holes), while
      // the engine's history is anchored on the physical number.
      const long eng_idx = cfg.gmc_phys_anchor ? (long)item.id : abs;

      // Wait until the history window allows this frame.
      {
        const auto t_win0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(m_guard);
        if (!abort && !window_ok(abs)) {
          {
            std::lock_guard<std::mutex> lt(m_time);
            timing.gmc_window_blocks += 1;
          }
          while (!abort && !window_ok(abs)) cv_guard.wait(lk);
        }
        if (abort) break;
        // The time the LOOKAHEAD BUDGET was exhausted: the ingest was faster
        // than the engines that owe the history.  This is the number that
        // says "the pipeline's rate is capped by the guard, not by the work".
        win_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_win0)
                .count();
      }

      // Seed every engine: each of them may need this frame as history.
      for (auto& e : engines) e->pipe->seed_frame(eng_idx, item.gray);

      // THE COMPUTATION CADENCE.
      //
      // With the default stride of 1 every frame gets its own GMC group --
      // the certified behaviour.  With a stride of N only every Nth camera
      // frame gets a group; the frames in between are seeded as history and
      // nothing else.  The groups are taken on the PHYSICAL grid, so a
      // group's 21 lags stay exactly on the f-2, f-4, ... f-42 positions the
      // certified pipeline reads for that same frame.
      //
      // WHY IT EXISTS: the GMC group costs ~65-116 ms, so one group per frame
      // covers only 15.3 camera frames per second -- LESS than a 25 fps
      // camera -- and the camera queue overflows, the physical frame ids
      // develop holes, and with the physical anchor those holes become
      // DETECTED history gaps.  One group per two frames covers 30.6 camera
      // frames per second, so the camera queue never overflows, the ids stay
      // dense, the history has no holes, and the frames that ARE computed
      // carry exactly the features the certified pipeline computes for the
      // same frame numbers.
      const int cstride = cfg.gmc_compute_stride > 0 ? cfg.gmc_compute_stride : 1;
      const bool is_group_frame = (cstride == 1) || ((item.id % cstride) == 0);
      if (is_group_frame) {
        // The engine that computes THIS group.  Round robin over the GROUPS,
        // not over the frames: with a stride of 2 the groups are frames 2, 4,
        // 6, ... and (abs-1) % n would hand every group to the same engine
        // and idle the others.
        Engine* eng = engines[(size_t)(group_counter % (long)engines.size())].get();
        ++group_counter;
        {
          const auto t_space0 = std::chrono::steady_clock::now();
          std::unique_lock<std::mutex> lk(eng->m);
          if (!abort && eng->jobs.size() >= gmc_job_depth) {
            std::lock_guard<std::mutex> lt(m_time);
            timing.gmc_space_blocks += 1;
          }
          eng->cv_space.wait(lk, [this, eng] {
            return abort || eng->jobs.size() < gmc_job_depth;
          });
          if (abort) break;
          if (!eng->has_pending) {
            eng->oldest_pending = abs;
            eng->has_pending = true;
          }
          GmcJob job;
          job.id = item.id;
          job.abs = abs;
          job.art_seq = ++artifact_seq;
          job.gray = item.gray;  // refcounted: no pixel copy
          eng->jobs.push_back(std::move(job));
          // The time the ingest waited for room in the engine's job queue.
          space_ms =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_space0)
                  .count();
        }
        eng->cv_job.notify_one();
      }
      {
        std::lock_guard<std::mutex> lt(m_time);
        timing.gmc_ingest_in_wait_ms += in_wait_ms;
        timing.gmc_ingest_window_ms += win_ms;
        timing.gmc_ingest_space_ms += space_ms;
        timing.gmc_ingest_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ingest0).count();
      }
    }
    ingest_done = true;
    // Wake everybody: the engines drain and stop, the upstream services see
    // that no more features are coming.
    cv_in.notify_all();
    for (auto& e : engines) {
      e->cv_job.notify_all();
      e->cv_space.notify_all();
    }
    cv_feat.notify_all();
    cv_guard.notify_all();
  }

  // -------------------------------------------------------------------------
  // SERVICE GMC: one engine computes one frame.
  // -------------------------------------------------------------------------
  void engine_loop(Engine* eng, size_t eng_index) {
    // Place this engine on its own core (if the caller asked for it).  A GMC
    // fit is a 60 ms CPU burst: if it shares a core with the NPU service
    // threads or the producer, the scheduler preempts it and the feature
    // stage falls behind the camera.
    //
    // A core GROUP wins over the legacy one-core-per-engine list: the group's
    // FIRST core is the engine's own, and the anchor workers take the rest.
    // (With the legacy list the engine gets exactly one core, and the anchor
    // workers -- if any -- would share it, which buys nothing.)
    if (!gmc_groups.empty()) {
      const std::vector<int>& g = gmc_groups[eng_index % gmc_groups.size()];
      if (!g.empty() && mosaic::PinCurrentThreadToCpu(g[0]))
        fprintf(stderr, "[i] GMC engine %zu pinned to core %d (group of %zu)\n", eng_index, g[0],
                g.size());
    } else if (!gmc_cpus.empty()) {
      const int cpu = gmc_cpus[eng_index % gmc_cpus.size()];
      if (mosaic::PinCurrentThreadToCpu(cpu))
        fprintf(stderr, "[i] GMC engine %zu pinned to core %d\n", eng_index, cpu);
    }
    for (;;) {
      GmcJob job;
      {
        const auto t_wait0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(eng->m);
        eng->cv_job.wait(lk, [this, eng] { return !eng->jobs.empty() || abort || ingest_done; });
        const double waited =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wait0)
                .count();
        {
          std::lock_guard<std::mutex> lt(m_time);
          timing.gmc_wait_ms += waited;
        }
        if (abort) break;
        if (eng->jobs.empty()) {
          if (ingest_done) break;
          continue;
        }
        // The depth of the job queue AT THE MOMENT the job is taken, the job
        // included.  THIS IS THE STARVATION PROOF: a mean of ~1 means the
        // engine picked the job up off an empty queue (it had been waiting
        // for work), a mean well above 1 means it always had a backlog.
        const size_t depth_before = eng->jobs.size();
        job = std::move(eng->jobs.front());
        eng->jobs.pop_front();
        {
          std::lock_guard<std::mutex> lt(m_time);
          timing.gmc_queue_depth_sum += (double)depth_before;
          timing.gmc_queue_samples += 1;
        }
      }
      eng->cv_space.notify_one();

      const OnlineFeaturePipeline::Stats st0 = eng->pipe->stats();
      const long blows_before = eng->pipe->n_fit_deadline();
      const long gaps_before = eng->pipe->n_history_gap();
      eng->pipe->set_seq_frame(eng->seq_frame++);
      // The transform chain is only collected when it is going to be dumped:
      // collecting it does not change the fit (the reference implementation
      // proves that -- it fills the same vector from the same map).
      std::vector<Matx23f> mats;
      const auto mark_entry = std::chrono::steady_clock::now();
      // THE INDEX THE ENGINE ANCHORS ITS HISTORY ON: the camera's PHYSICAL
      // frame number when the physical anchor is on, the consumption counter
      // otherwise.  `job.abs` stays the dense counter that orders the
      // artifacts; `job.id` is the physical number.
      const long eng_idx = cfg.gmc_phys_anchor ? (long)job.id : job.abs;
      const cv::Mat out =
          eng->pipe->process_frame(eng_idx, mats_out ? &mats : nullptr, mark_entry);
      const OnlineFeaturePipeline::Stats st1 = eng->pipe->stats();
      const long gaps_after = eng->pipe->n_history_gap();
      const auto mark_post0 = std::chrono::steady_clock::now();

      // The per-frame feature fingerprint: the rolling MD5 over the PRE-ENCODE
      // 3 x (H*W) channel planes, exactly like the reference's --md5-out.
      // It is INSTRUMENTATION: ~5-7 ms of CPU per frame that no production
      // deployment needs, so it can be switched off (gmc_feature_fingerprint).
      std::string fh;
      if (cfg.gmc_feature_fingerprint) {
        gmcpp::MD5 f;
        f.update(reinterpret_cast<const uint8_t*>(out.data),
                 static_cast<size_t>(out.total()));
        fh = f.hex();
      }
      // The order-sensitive artifacts go to the writer thread, which emits
      // them in FRAME order.
      publish_artifact(job.art_seq, job.id, fh, mats);

      // THE AUTHORITATIVE COUNT OF THE GMC GROUPS THAT COMPLETED.
      //
      // One increment per group the GMC service actually computed, i.e. per
      // frame a feature really exists for.  It is the ONLY denominator that
      // holds for every cadence: the number of PUSHED frames does not (with
      // gmc_compute_stride > 1 only every Nth frame becomes a group, the rest
      // are seeded as history and produce nothing), and the number of frames
      // that reached the GMC service does not either (the camera queue may
      // have dropped some of them).
      //
      // It is counted HERE, right after the fit, so it cannot drift from the
      // features that were produced.  A caller derives the number of results
      // it should see from this counter, NOT from the number of frames it
      // pushed -- see the header.
      {
        std::lock_guard<std::mutex> lt(m_time);
        timing.gmc_groups_done += 1;
      }

      // Everything above (the interleave, the fingerprint and the artifact
      // hand-off) is the GMC service's cost AFTER the fit.  It used to be
      // invisible: the only number the report had was gmc_total, which
      // measures the algorithm inside process_frame.  That is how a ~22 ms
      // per-frame overhead stayed hidden until a 25 fps camera exposed it.
      //
      // The accumulation itself happens at the END of this block (after the
      // feature is queued): the accounting used to stop at publish_artifact(),
      // so the NEON interleave and the enqueue -- the two biggest items -- were
      // NOT counted at all, and gmc_post_ms understated the real post-fit cost
      // by exactly those two.

      // The wall-clock watchdog fired: this frame's transform DEGRADED (it
      // fell back to identity or to the last good transform), so its features
      // are NOT registered to the raw frame.  That is a real event and it has
      // to be loud: a silently degraded frame is a missed detection.
      const long blows_after = eng->pipe->n_fit_deadline();
      if (blows_after > blows_before) {
        fprintf(stderr,
                "[WARN] GMC frame %lld (engine %zu) blew the %d ms fit watchdog: the "
                "transform DEGRADED (identity or last-good), so this frame's features are "
                "NOT registered to the raw frame\n",
                (long long)job.id, eng_index, cfg.gmc_deadline_ms);
      }
      // THE PHYSICAL HISTORY HAD A HOLE.  A camera frame was lost, so one or
      // more of this frame's 21 history frames do not exist.  The engine did
      // NOT silently read a different frame: it counted the hole and fitted
      // against the nearest available older frame, so this frame's temporal
      // window is stretched.  Loud on purpose: a stretched window is a frame
      // whose features are not comparable with the zero-drop run.
      if (gaps_after > gaps_before) {
        fprintf(stderr,
                "[WARN] GMC frame %lld (engine %zu): the PHYSICAL history has a HOLE: %ld "
                "history lookup(s) hit a MISSING physical frame (a camera frame was lost), "
                "so this frame's temporal window is stretched -- the fit used the nearest "
                "available older frame instead of the lost one\n",
                (long long)job.id, eng_index, gaps_after - gaps_before);
      }
      {
        std::lock_guard<std::mutex> lk(m_time);
        timing.gmc_deadline_blows += (blows_after - blows_before);
        // The frames whose PHYSICAL history had a hole.  This counter is 0
        // unless gmc_phys_anchor=1: with the consumption-counter anchor the
        // engine cannot see a lost camera frame at all, so the loss was
        // silent -- this is the counter that makes it visible.
        timing.gmc_history_gaps += (gaps_after - gaps_before);
        timing.gmc_fit_ms += st1.t_fit - st0.t_fit;
        timing.gmc_warp_ms += st1.t_warp - st0.t_warp;
        timing.gmc_median_ms += st1.t_median - st0.t_median;
        timing.gmc_total_ms += st1.t_total - st0.t_total;
        const double fit_ms = st1.t_fit - st0.t_fit;
        if (fit_ms > timing.gmc_fit_max_ms) timing.gmc_fit_max_ms = fit_ms;
        gmc_fit_samples.push_back(fit_ms);
        // The corner statistics: WHY the fit costs what it costs.  The engine's
        // corner counters are cumulative, so the per-frame corner count is the
        // difference against the previous frame.
        const double dets = (double)eng->pipe->corner_dets();
        const double mean_c = eng->pipe->corner_mean();
        const double cum_pts = mean_c * dets;
        const double d_dets = dets - eng->prev_corner_dets;
        const double d_pts = cum_pts - eng->prev_corner_pts;
        eng->prev_corner_dets = dets;
        eng->prev_corner_pts = cum_pts;
        if (d_dets > 0.0) {
          gmc_corner_dets += d_dets;
          gmc_corner_pts += d_pts;
        }
      }

      if (!dump_dir.empty()) {
        // SAME array and SAME cv::imwrite call as the reference's --out-dir
        // path (gmc_stream_ocl.cpp), so the two sides are directly comparable
        // pixel by pixel.  The MD5 stream cannot answer "how far apart": a
        // single flipped bit changes the whole hash, while the real question
        // is whether the channels are within a grey level or two.
        cv::Mat hwc(height, width, CV_8UC3);
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
                      (long long)job.id);
        cv::imwrite(nm, hwc);
      }

      // PLANE-MAJOR -> model order [residual, diff, I_t].
      // out is a 3 x (H*W) CV_8UC1 Mat: row 0 = I_t, row 1 = diff,
      // row 2 = residual.
      const size_t plane = static_cast<size_t>(height) * static_cast<size_t>(width);
      FeatureItem fi;
      fi.id = job.id;
      fi.rgb.resize(plane * 3);
      const uint8_t* c0 = out.ptr<uint8_t>(0);
      const uint8_t* c1 = out.ptr<uint8_t>(1);
      const uint8_t* c2 = out.ptr<uint8_t>(2);
      // NEON vst3q_u8: the same bytes, 16 pixels at a time.  The scalar
      // version of this loop cost ~15 ms of CPU per frame and was enough to
      // push the GMC service below the 25 fps of the camera.
      mosaic::Interleave3Planes(c0, c1, c2, plane, fi.rgb.data());

      // Into the NPU service queue.  THIS IS THE DROP POINT.
      //
      // The GMC service NEVER waits for the NPU services.  The inference stage
      // is per-frame independent, so a frame it cannot keep up with is
      // SKIPPED -- the freshest data is the one worth keeping.
      //
      // Blocking here instead would push the back-pressure all the way to the
      // camera queue, and a frame dropped THERE never reaches the GMC: the
      // transform chain loses a step and every later frame is mis-registered.
      // That is the one drop this pipeline must never make, so the drop moved
      // here, below the GMC.
      {
        std::lock_guard<std::mutex> lk(m_feat);
        if (feat_q.size() >= jitter_depth) {
          feat_q.pop_front();  // the OLDEST feature goes
          std::lock_guard<std::mutex> lt(m_time);
          timing.features_dropped += 1;
          // NO WARNING HERE, ON PURPOSE.  Skipping a frame in the inference
          // stage is the DESIGNED behaviour of a live pipeline: the inference
          // runs as fast as it can and skips what it cannot keep up with,
          // and the cost is one missed detection.  Only the loss of a GMC
          // FEATURE is worth an operator's attention, and that can only
          // happen upstream (the camera queue), where it is warned about.
          // The counters still carry the number for the final summary.
        }
        feat_q.push_back(std::move(fi));
      }
      cv_feat.notify_all();

      // The post-fit cost of the GMC service, in FULL: the per-frame
      // fingerprint, the artifact hand-off, the NEON interleave and the
      // enqueue into the NPU queue.  It is accumulated HERE, after the
      // enqueue, because the enqueue is the last thing the GMC service does
      // for this frame -- the accounting used to stop at publish_artifact()
      // and therefore left the interleave and the enqueue uncounted.
      {
        const double post_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mark_post0)
                .count();
        std::lock_guard<std::mutex> lk(m_time);
        timing.gmc_post_ms += post_ms;
      }

      // The engine has finished this frame: the oldest frame it still owes
      // moves forward, which may unblock the ingest.
      {
        std::lock_guard<std::mutex> lk(eng->m);
        if (eng->jobs.empty()) {
          eng->has_pending = false;
        } else {
          eng->oldest_pending = eng->jobs.front().abs;
        }
      }
      cv_guard.notify_all();
    }
  }

  // -------------------------------------------------------------------------
  // SERVICE NPU: RGA compose + upload + rknn_run + outputs.
  //
  // One independent service per RKNN context.  The board tolerates exactly
  // TWO concurrent contexts (a third trips the 6 s job timeout and soft-resets
  // the NPU), hence npu_workers is clamped to 2.
  // -------------------------------------------------------------------------
  void npu_loop(NpuService* svc) {
    // The NPU service spends most of its wall time blocked on the NPU
    // hardware, but the driver does copy and dequant work on the CPU: give it
    // its own core so it never steals the GMC engines' cores.
    if (!npu_cpus.empty()) {
      const int cpu = npu_cpus[(size_t)svc->index % npu_cpus.size()];
      if (mosaic::PinCurrentThreadToCpu(cpu))
        fprintf(stderr, "[i] NPU service %d pinned to core %d\n", svc->index, cpu);
    }
    mosaic::Session& sess = *svc->sess;
    mosaic::FrameOut fo;
    for (;;) {
      FeatureItem fi;
      {
        const auto t_wait0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(m_feat);
        cv_feat.wait(lk, [this] { return !feat_q.empty() || abort || gmc_done; });
        const double waited =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wait0)
                .count();
        {
          std::lock_guard<std::mutex> lt(m_time);
          timing.npu_wait_ms += waited;
        }
        if (abort) break;
        if (feat_q.empty()) {
          if (gmc_done) break;
          continue;
        }
        fi = std::move(feat_q.front());
        feat_q.pop_front();
      }

      // ---- stage 1: RGA compose + upload ---------------------------------
      if (!mosaic::PrepareFrame(&sess, fi.rgb.data(), width, height, &fo)) {
        set_error("PrepareFrame failed (NPU service)");
        break;
      }
      // ---- stage 2: rknn_run + outputs -----------------------------------
      std::vector<float> hm;
      if (!mosaic::RunFrame(&sess, &hm, &fo)) {
        set_error("RunFrame failed (NPU service)");
        break;
      }
      {
        std::lock_guard<std::mutex> lk(m_time);
        timing.rga_import_ms += fo.rga_import;
        timing.rga_improcess_ms += fo.rga_improcess;
        timing.upload_ms += fo.upload;
        timing.run_ms += fo.run;
        timing.outputs_ms += fo.outputs;
        // One sample per forward pass: the distribution of the SINGLE
        // rknn_run, which is what the NPU core mask is judged on.
        rknn_run_samples.push_back(fo.run);
        if (fo.run > timing.rknn_run_max_ms) timing.rknn_run_max_ms = fo.run;
      }

      HeatItem hi;
      hi.id = fi.id;
      hi.sess = &sess;
      hi.hm = std::move(hm);
      {
        // Same rule one stage further down: the inference does not wait for
        // the back-projection.  A heatmap that the post service cannot take
        // in time is dropped (its detection is lost, the GMC registration is
        // untouched).
        std::lock_guard<std::mutex> lk(m_heat);
        if (heat_q.size() >= queue_depth) {
          heat_q.pop_front();
          std::lock_guard<std::mutex> lt(m_time);
          timing.detections_dropped += 1;
          // Same rule as the feature skip: a downstream skip is normal
          // operation, not an operator-facing fault.
        }
        heat_q.push_back(std::move(hi));
      }
      cv_heat.notify_all();
    }
  }

  // -------------------------------------------------------------------------
  // SERVICE POST: back-projection + 8-connected components.
  //
  // This is the stage that used to run INSIDE the NPU thread, where it
  // blocked the next inference.  Here it overlaps the inference completely.
  // Each worker owns its own PostWorkspace: the scratch belongs to the
  // stage, not to the session that produced the heatmap.
  // -------------------------------------------------------------------------
  void post_loop(mosaic::PostWorkspace* ws, size_t post_index) {
    // The POST stage shares the NPU services' cores: the NPU service is
    // blocked on the hardware most of the time, so those cores are mostly
    // idle -- but the POST stage must never land on a GMC engine's core.
    if (!npu_cpus.empty()) {
      const int cpu = npu_cpus[post_index % npu_cpus.size()];
      if (mosaic::PinCurrentThreadToCpu(cpu))
        fprintf(stderr, "[i] POST service %zu pinned to core %d\n", post_index, cpu);
    }
    mosaic::FrameOut fo;
    for (;;) {
      HeatItem hi;
      {
        const auto t_wait0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(m_heat);
        cv_heat.wait(lk, [this] { return !heat_q.empty() || abort || feat_done; });
        const double waited =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wait0)
                .count();
        {
          std::lock_guard<std::mutex> lt(m_time);
          timing.post_wait_ms += waited;
        }
        if (abort) break;
        if (heat_q.empty()) {
          if (feat_done) break;
          continue;
        }
        hi = std::move(heat_q.front());
        heat_q.pop_front();
      }

      // ---- stage 3: back-projection + connected components ---------------
      std::vector<mosaic::Comp> comps;
      double backproj_ms = 0.0;
      mosaic::BackProjectFrame(hi.sess, hi.hm, ws, &comps, &backproj_ms);
      {
        std::lock_guard<std::mutex> lk(m_time);
        timing.backproj_ms += backproj_ms;
        timing.frames += 1;
      }

      ResultItem ri;
      ri.id = hi.id;
      ri.comps = std::move(comps);
      {
        std::lock_guard<std::mutex> lk(m_out);
        if (cb) cb_q.push_back(std::move(ri));
        else out_q.push_back(std::move(ri));
      }
      cv_out.notify_all();
    }
  }

  // -------------------------------------------------------------------------
  // The order-sensitive writer: the rolling md5 and the --mats-all dump, in
  // FRAME order, no matter in which order the engines finished.
  // -------------------------------------------------------------------------
  void publish_artifact(long abs, int64_t id, const std::string& md5_hex,
                        const std::vector<Matx23f>& mats) {
    {
      std::lock_guard<std::mutex> lk(m_art);
      OrderedArtifact a;
      a.id = id;
      a.md5_hex = md5_hex;
      a.mats = mats;
      artifacts[abs] = std::move(a);
    }
    cv_art.notify_all();
  }

  void artifact_loop() {
    for (;;) {
      OrderedArtifact a;
      {
        std::unique_lock<std::mutex> lk(m_art);
        cv_art.wait(lk, [this] {
          return abort || artifacts.count(next_artifact) || (gmc_done && artifacts.empty());
        });
        if (abort) break;
        auto it = artifacts.find(next_artifact);
        if (it == artifacts.end()) {
          if (gmc_done) break;
          continue;
        }
        a = std::move(it->second);
        artifacts.erase(it);
        ++next_artifact;
      }
      // The rolling md5 is folded in frame order -- that is the whole point
      // of this thread.
      {
        std::lock_guard<std::mutex> lk(m_md5);
        seq_md5.update(a.md5_hex);
        seq_md5_hex = seq_md5.hex();
      }
      if (md5_out) {
        char line[256];
        std::snprintf(line, sizeof(line), "%s\t%06lld\t%s\n", cfg.sequence, (long long)a.id,
                      a.md5_hex.c_str());
        std::fputs(line, md5_out);
        std::fflush(md5_out);
      }
      if (mats_out && !a.mats.empty()) {
        // Same bytes as the reference's --mats-all: 21 lags x 6 floats,
        // ascending lag order.
        for (const Matx23f& m : a.mats) {
          for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
              const float v = m(i, j);
              std::fwrite(&v, sizeof(float), 1, mats_out);
            }
          }
        }
        std::fflush(mats_out);
      }
    }
  }

  // -------------------------------------------------------------------------
  // The dispatcher: hands the results to the callback.
  // -------------------------------------------------------------------------
  void dispatch_loop() {
    for (;;) {
      ResultItem ri;
      {
        std::unique_lock<std::mutex> lk(m_out);
        cv_out.wait(lk, [this] { return !cb_q.empty() || abort || flushed; });
        if (abort) break;
        if (cb_q.empty()) {
          if (flushed) break;
          continue;
        }
        ri = std::move(cb_q.front());
        cb_q.pop_front();
      }
      MosaicResultCallback fn = nullptr;
      void* user = nullptr;
      {
        std::lock_guard<std::mutex> lk(m_out);
        fn = cb;
        user = cb_user;
        if (!fn) {
          // The callback was removed while this result was in flight: hand it
          // to the polling path instead of dropping it.
          out_q.push_back(std::move(ri));
          cv_out.notify_all();
          continue;
        }
      }
      // The callback speaks MosaicDet, the pipeline speaks Comp.
      std::vector<MosaicDet> dets(ri.comps.size());
      for (size_t i = 0; i < ri.comps.size(); ++i) {
        const mosaic::Comp& c = ri.comps[i];
        dets[i].tag = c.tag;
        dets[i].cx = c.cx;
        dets[i].cy = c.cy;
        dets[i].w = c.w;
        dets[i].h = c.h;
        dets[i].area = c.area;
        dets[i].peak = c.peak;
        dets[i].sum = c.sum;
      }
      fn(user, ri.id, dets.data(), (int32_t)dets.size());
    }
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
  // The GMC anchor step.  2 == the server's frozen OFFICIAL semantics: with
  // anchor_step=2 EVERY one of the 21 lags (2,4,...,42) is fitted DIRECTLY
  // over the long baseline fit(frame(t-L), frame(t)), with ZERO composition,
  // so the board output is BYTE-FOR-BYTE identical to the frozen server
  // features (measured: 300-frame MD5-SUM 9224745c39ca14070a5b94c20a44b730
  // on both sides).  10 == the TREE anchor cascade (anchors {2,12,22,32,42},
  // 6.19 fits/frame): ~3.4x cheaper, but NOT equivalent -- 122 of 300 frames
  // (40.7%) differ from the server reference.  2 is the default.
  cfg->gmc_anchor_step = 2;
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
  cfg->gmc_feature_fingerprint = 1;   // the audit tools need it; production can turn it off
  // OFF by default: with the watchdog on, a frame that blows its budget
  // DEGRADES, so the output stops being bit-reproducible.  The real-time
  // deployment turns it on (see the header).
  cfg->gmc_deadline_ms = 0;
  cfg->gmc_ransac_max_iters = 0;
  // THE PHYSICAL-FRAME-NUMBER ANCHOR of the GMC history.  0 = OFF (the
  // default, and the certified behaviour): the engine anchors on the
  // consumption counter, as it always has.  1 = ON: the engine anchors on the
  // camera's physical frame number, so a frame that never arrived is a
  // DETECTED hole instead of a silently stretched window.  See the header.
  cfg->gmc_phys_anchor = 0;
  // WHAT THE CAMERA QUEUE DOES ON OVERFLOW.  0 = evict the OLDEST (the
  // shipped behaviour, and the default: a live stream wants fresh data), 1 =
  // reject the NEWEST (keep the queued frames, throw away the incoming one).
  // See the header for why 0 is the default.
  cfg->gmc_input_drop_policy = 0;
  // The computation cadence: how many camera frames make one GMC group.
  // 1 == every frame gets its own group (the certified behaviour).  2 == one
  // group per two frames, which lets the GMC service cover MORE camera frames
  // per second than the camera produces, so the camera queue never overflows
  // and the physical history has no holes.  See the header.
  cfg->gmc_compute_stride = 1;
  // The anchor LK stage's parallelism.  1 == the reference's serial anchor
  // loop, which is the CERTIFIED behaviour: the anchor workers only exist to
  // make the same numbers faster, and they are opt-in because they change
  // the SCHEDULING (which core runs which anchor), not the arithmetic.
  cfg->gmc_anchor_threads = 1;
  // 0 == the shipped behaviour: rknn_set_core_mask is NOT called.  See the
  // header for what a non-zero mask is and why it is a deviation.
  cfg->npu_core_mask = 0;
  // NULL == no per-engine core groups: the engines are placed by
  // gmc_affinity (one core each) or inherit cpu_affinity.
  cfg->gmc_affinity_group = nullptr;
  cfg->canvas_mode = MOSAIC_CANVAS_MOSAIC;
  cfg->input_mode = MOSAIC_INPUT_SET;
  cfg->npu_workers = 2;
  cfg->region_threshold = 0;
  cfg->min_area = 0;
  cfg->max_detections = 0;
  // ---- the services -------------------------------------------------------
  // ONE engine.  The library still accepts 2 (the hard cap), but the shipped
  // default is 1 because the two-engine split buys NOTHING measurable here:
  // on the board, 1 engine and 2 engines produced BYTE-IDENTICAL GMC features
  // (300-frame A/B: feature_md5 ca5e3d30b95a6c55152483944216b35d for
  // {1 engine, serial anchors} / {1 engine, parallel anchors} / {2 engines,
  // serial} / {2 engines, parallel}), because the GMC is bounded by the NPU
  // stage (68.7 ms/frame of NPU work against 42.1 ms/frame of GMC work) and
  // the second engine only added the cross-engine interleaving.
  cfg->gmc_workers = 1;    // one engine; the per-frame computation is unchanged
  cfg->post_workers = 1;   // the post stage is cheap; one worker is enough
  cfg->queue_depth = 8;    // between the services
  // 0 = AUTO: the REQUEST stays 0 here; the library resolves it in its
  // sanitize stage (where the AUTO rule lives exactly once) and reports the
  // resolved budget through mosaic_pipeline_effective_config().
  // It is a THROUGHPUT knob only -- it changes how far the ingest may run
  // ahead of the engines, never which pixels a fit reads.
  cfg->gmc_ring_slack = 0;
  // The GMC -> NPU jitter buffer, in frames.  It has to be DEEPER than the
  // producer's in-flight window, or the buffer would overflow in a healthy
  // self-paced run and the inference would skip frames that the producer had
  // every right to expect a result for.  The demo keeps 4 frames in flight and
  // the contract test pushes batches of 4, so 8 leaves a 2x margin.
  cfg->jitter_depth = 8;
  cfg->input_queue = 8;    // camera-facing; on overflow the OLDEST is dropped
  cfg->npu_affinity = nullptr;
  cfg->gmc_affinity = nullptr;
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
      {"gmc_anchor_step", &cfg_s.gmc_anchor_step, 2, 64, 2},
      {"gmc_downscale", &cfg_s.gmc_downscale, 1, 4, 2},
      {"gmc_fit_corners", &cfg_s.gmc_fit_corners, 10, 5000, 600},
      {"gmc_fit_win", &cfg_s.gmc_fit_win, 3, 99, 15},
      {"gmc_fit_level", &cfg_s.gmc_fit_level, 0, 8, 2},
      {"gmc_fit_iters", &cfg_s.gmc_fit_iters, 1, 200, 20},
      {"gmc_gate_shift", &cfg_s.gmc_gate_shift, 0, 10000, 1000},
      {"gmc_gate_min", &cfg_s.gmc_gate_min, 1, 10000, 500},
      {"gmc_gate_max", &cfg_s.gmc_gate_max, 1, 10000, 2000},
      {"npu_workers", &cfg_s.npu_workers, 1, 2, 2},
      // gmc_workers is capped at 2, and MEASURED, not guessed:
      //
      //   engines=2  gmc_fit  70.365 ms/frame   total_wall 65.953 ms   15.162 fps
      //   engines=3  gmc_fit 103.852 ms/frame   total_wall 67.642 ms   14.784 fps
      //   (same 3000 frames, same corner budget 600, same idle board)
      //
      // MORE ENGINES ARE SLOWER, and the reason is structural, not thermal:
      // the GMC composes a non-anchor lag from a chain of ONE-STEP fits, and a
      // one-step fit is the pair of frames (a, a+stride).  The engines are
      // assigned round robin, so with TWO engines each engine owns one PARITY
      // class and every one-step pair falls inside a single engine -- the pair
      // is fitted once and cached.  With THREE (or four) engines the pair
      // (a, a+2) always straddles two engines, so every frame has to LAZILY
      // re-fit several pairs it cannot find in its own cache: the fit cost
      // rises ~48% and the tail with it (p99 173.6 -> 246.5 ms).
      //
      // Two engines are also enough for the 25 fps budget: the GMC service
      // then carries 70.4/2 = 35.2 ms per frame of the 40 ms budget, and the
      // pipeline's real critical path is the NPU service (64.1 ms/frame),
      // which no amount of GMC parallelism can shorten.
      {"gmc_workers", &cfg_s.gmc_workers, 1, 2, 2},
      {"post_workers", &cfg_s.post_workers, 1, 4, 1},
      {"queue_depth", &cfg_s.queue_depth, 1, 64, 8},
      // The lookahead budget.  0 = AUTO (the library's own default), and an
      // out-of-range value falls back to AUTO rather than to a number that
      // would silently change the pipeline's shape.  The ceiling (256) keeps
      // the history ring's memory bounded: ring_depth = 43 + slack frames of
      // 640x512 B, i.e. ~86 MB at the ceiling.
      {"gmc_ring_slack", &cfg_s.gmc_ring_slack, 0, 256, 0},
      {"jitter_depth", &cfg_s.jitter_depth, 1, 64, 8},
      // The camera-facing RAW buffer, in frames.  This is the queue that holds
      // the ORIGINAL pixels between the camera and the GMC service, so its
      // depth is the depth of the pipeline's history guarantee: a frame that
      // sits in here has NOT been lost, and the GMC service can still pick it
      // up and fit it against its 21-frame history.
      //
      // WHY THE CEILING IS 8192 AND NOT 64:  the GMC service consumes frames
      // at ITS OWN rate, which is below the camera's rate whenever the GMC
      // service costs more than the frame period (step2: ~161 ms/frame vs the
      // 40 ms budget of 25 fps).  Then the camera outruns the GMC service and
      // the raw buffer MUST absorb the difference -- otherwise the overflow
      // policy (drop the oldest) destroys the frame BEFORE the GMC service
      // ever sees it, the GMC history ring gets a hole, and the registration
      // chain degrades.  A 64-frame ceiling made that loss UNAVOIDABLE at
      // 25 fps: the buffer could not hold more than 2.5 s of overrun.
      //
      // The cost of a deep buffer is MEMORY and LATENCY, not accuracy: at
      // 640x512 B per frame, 8192 frames is 2.6 GB.  The board has ~6.6 GB
      // free, so the ceiling is set where the deepest useful buffer still
      // fits; deeper than that buys nothing a 25 fps camera could overrun.
      {"input_queue", &cfg_s.input_queue, 1, 8192, 8},
      {"gmc_deadline_ms", &cfg_s.gmc_deadline_ms, 0, 10000, 0},
    {"gmc_feature_fingerprint", &cfg_s.gmc_feature_fingerprint, 0, 1, 1},
      {"gmc_ransac_max_iters", &cfg_s.gmc_ransac_max_iters, 0, 100000, 0},
      // The physical-frame-number anchor of the GMC history.  0 = OFF (the
      // consumption counter, i.e. the certified behaviour), 1 = ON (the
      // camera's physical frame number, so a lost frame is a detected hole).
      {"gmc_phys_anchor", &cfg_s.gmc_phys_anchor, 0, 1, 0},
      // What the camera queue does on overflow.  0 = evict the oldest (the
      // shipped behaviour), 1 = reject the newest.
      {"gmc_input_drop_policy", &cfg_s.gmc_input_drop_policy, 0, 1, 0},
      // How many camera frames make one GMC group.  1 == every frame (the
      // certified behaviour).  2 == one group per two frames: the GMC service
      // then covers 30.6 camera frames per second against a 25 fps camera,
      // so the camera queue never overflows and the physical history stays
      // hole-free.  It changes WHICH frames get a feature, never what a
      // feature is computed from.
      {"gmc_compute_stride", &cfg_s.gmc_compute_stride, 1, 8, 1},
      // The anchor LK stage's worker count.  1 == the reference's serial
      // anchor loop (the default, and the certified behaviour).  The ceiling
      // is 8: the anchor grid has 5 anchors, so anything above that is a
      // thread that would never get any work.
      {"gmc_anchor_threads", &cfg_s.gmc_anchor_threads, 1, 8, 1},
      // The NPU core mask.  0 == "do not call rknn_set_core_mask" (the
      // shipped behaviour).  The valid masks are the rknn_core_mask bits:
      // 1 = core 0, 2 = core 1, 4 = core 2, 3 = 0+1, 6 = 1+2, 5 = 0+2,
      // 7 = all three.  Anything else is a typo and falls back to 0.
      {"npu_core_mask", &cfg_s.npu_core_mask, 0, 7, 0},
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
  if (cfg_s.input_mode < 0 || cfg_s.input_mode > MOSAIC_INPUT_IO_MEM_NHWC)
    cfg_s.input_mode = MOSAIC_INPUT_SET;
  // THE AUTO RULE FOR THE LOOKAHEAD BUDGET LIVES HERE AND NOWHERE ELSE.
  //
  // A request of 0 means "auto", and the auto value is resolved ONCE, HERE,
  // in the sanitize stage -- after the clamp table, so gmc_workers is already
  // the clamped value this rule reads.  Everything downstream then reads the
  // RESOLVED number: p->cfg (and therefore mosaic_pipeline_effective_config())
  // reports the budget in force, and the demo prints that number instead of
  // re-deriving it.  A second copy of this formula would silently drift and
  // the report would then state a budget the pipeline does not use.
  //
  // The clamp table's "out of range -> 0" stays as it is: an impossible
  // request is folded back to AUTO, and AUTO is resolved right here.
  if (cfg_s.gmc_ring_slack == 0)
    cfg_s.gmc_ring_slack = (int32_t)std::max<long>(64, 32 * cfg_s.gmc_workers);
  const MosaicPipelineConfig* safe_cfg = &cfg_s;

  auto* p = new MosaicPipeline();
  p->cfg = *safe_cfg;
  p->queue_depth = (size_t)safe_cfg->queue_depth;
  p->jitter_depth = (size_t)safe_cfg->jitter_depth;
  p->input_depth = (size_t)safe_cfg->input_queue;

  // OpenCL fused tail: the Mali GPU does the 21 inverse re-projections, the
  // 21-element median and the [Ch0,Ch1,Ch2] assembly.
  //
  // The slot count MUST be derived from the SANITIZED window/stride: an
  // out-of-range gmc_window used here would size the device ring from an
  // absurd number (the contract test caught exactly that: slots=20001 for a
  // window of 9999) and the allocation would happen before any clamp.
  const int slots = safe_cfg->gmc_window * safe_cfg->gmc_stride_step + 3;

  // The history window: the engines may lag the ingest by `ring_slack` frames.
  // The slack buys the LOOKAHEAD the multi-engine assignment needs; without
  // it the ingest could not run ahead of the oldest frame an engine still
  // owes, and the whole multi-engine scheme would collapse into one engine.
  //
  // WHY the budget is a knob and not a constant: the guard's release rate IS
  // the engine's completion rate, so a budget that is too small makes the
  // ingest wait for the engines, which starves the engines, which makes them
  // slower -- a self-sustaining rate limit.  Measured on the board with a 25
  // fps camera and 11,810 frames: a budget of 16 frames held the steady state
  // at 23.3 fps (below the camera), the camera queue overflowed (794 frames
  // lost, i.e. the GMC registration chain broken) and the ingest spent 42.9
  // ms per frame against a 40 ms frame period.  The budget is a THROUGHPUT
  // knob: it changes how far the ingest may run ahead, never which pixels a
  // fit reads, so the feature stream stays bit-identical.
  //
  // The value is TAKEN, not derived: the AUTO rule was already resolved in the
  // sanitize stage (one rule, one place), so this cannot drift from what
  // mosaic_pipeline_effective_config() reports.
  p->ring_slack = (long)safe_cfg->gmc_ring_slack;

  // The ENGINE JOB QUEUE must be able to hold the lookahead the window guard
  // allows, otherwise the two guards fight each other: the window guard
  // releases a frame, and the space guard immediately blocks it because the
  // queue is too shallow to hold the burst.  That is exactly what the old
  // coupling (ring_slack == 2 * queue_depth) produced -- raising queue_depth
  // alone did not help, because the budget still capped the job queue.
  //
  // The depth is therefore derived from the budget: half of it (the ingest
  // assigns every other frame to this engine) plus a small margin, and never
  // below queue_depth (a caller that asked for a deeper queue keeps it).
  p->gmc_job_depth = std::max<size_t>(safe_cfg->queue_depth,
                                      (size_t)(p->ring_slack / 2 + 4));

  // ---- the per-service core lists -----------------------------------------
  // A garbage cpu_affinity POINTER would be a crash, not a silent degradation,
  // so it is validated here too: only a non-empty, printable string is used.
  //
  // The PROCESS affinity is applied HERE, before the engines are built: the
  // anchor workers are threads created with the engine, and a thread inherits
  // the creator's mask.  Applying it later would leave the workers with the
  // unrestricted mask while every other service got the narrowed one.
  if (safe_cfg->cpu_affinity && *safe_cfg->cpu_affinity) {
    mosaic::ApplyCpuAffinity(safe_cfg->cpu_affinity);
  }

  // The GMC engines get their own cores so that a 60 ms fit is never
  // preempted.  This is a SCHEDULING change: it cannot alter a single number.
  //
  // This block runs BEFORE the engines are built: the anchor workers need to
  // know, at construction time, whether the engine will land on one core (and
  // therefore share it with its workers) or on a group.
  if (safe_cfg->gmc_affinity && *safe_cfg->gmc_affinity) {
    if (mosaic::ParseCpuList(safe_cfg->gmc_affinity, &p->gmc_cpus))
      fprintf(stderr, "[i] GMC engines -> cores %s\n", safe_cfg->gmc_affinity);
    else
      fprintf(stderr, "[WARN] gmc_affinity '%s' parsed to an empty list; ignored\n",
              safe_cfg->gmc_affinity);
  }
  if (safe_cfg->npu_affinity && *safe_cfg->npu_affinity) {
    if (mosaic::ParseCpuList(safe_cfg->npu_affinity, &p->npu_cpus))
      fprintf(stderr, "[i] NPU/POST services -> cores %s\n", safe_cfg->npu_affinity);
    else
      fprintf(stderr, "[WARN] npu_affinity '%s' parsed to an empty list; ignored\n",
              safe_cfg->npu_affinity);
  }

  // ---- the GMC engines ----------------------------------------------------
  // The per-engine core GROUPS are parsed HERE, before the engines are built:
  // the engine's anchor workers need their core list at construction time.
  //
  // The groups are ';'-separated ("4-5;6-7" = engine 0 gets {4,5}, engine 1
  // gets {6,7}) and each group is a normal cpu list ("4-5", "4,5").
  if (safe_cfg->gmc_affinity_group && *safe_cfg->gmc_affinity_group) {
    const std::string spec = safe_cfg->gmc_affinity_group;
    size_t pos = 0;
    while (pos <= spec.size()) {
      const size_t sep = spec.find(';', pos);
      const std::string one = spec.substr(pos, sep == std::string::npos ? std::string::npos : sep - pos);
      std::vector<int> cpus;
      if (!one.empty() && mosaic::ParseCpuList(one, &cpus) && !cpus.empty())
        p->gmc_groups.push_back(cpus);
      else
        fprintf(stderr, "[WARN] gmc_affinity_group '%s' has an unusable group '%s'; ignored\n",
                spec.c_str(), one.c_str());
      if (sep == std::string::npos) break;
      pos = sep + 1;
    }
    if (p->gmc_groups.empty())
      fprintf(stderr, "[WARN] gmc_affinity_group '%s' yielded no usable group; ignored\n",
              spec.c_str());
    else
      fprintf(stderr, "[i] GMC engines -> %zu core group(s)\n", p->gmc_groups.size());
  }
  for (int i = 0; i < safe_cfg->gmc_workers; ++i) {
    auto eng = std::unique_ptr<MosaicPipeline::Engine>(new MosaicPipeline::Engine());
    eng->gpu.reset(new FusedGpu(kernel_dir, false, false, slots));
    // The engine is constructed with the SAME knobs as the single-engine
    // build; only the history ring is deeper.
    eng->pipe.reset(new OnlineFeaturePipeline(
        safe_cfg->gmc_window, safe_cfg->gmc_stride_step, safe_cfg->gmc_anchor_step,
        safe_cfg->gmc_downscale,
        /*rng_seed_per_fit=*/false, /*nogmc=*/false, eng->gpu.get(), OnlineFeaturePipeline::kGpu,
        /*pyr_cache=*/safe_cfg->gmc_pyr_cache != 0, safe_cfg->gmc_fit_corners,
        safe_cfg->gmc_fit_win, safe_cfg->gmc_fit_level, safe_cfg->gmc_fit_iters,
        /*cache_parts=*/7, /*threads=*/1, /*ring_slack=*/(int)p->ring_slack));
    eng->pipe->set_profiling(false);
    eng->pipe->set_seq_id(0);
    // The fit gate MUST keep the C++ reference default (1.0 frame width).
    // e2e_core.h carries 0.25, but that is a different harness; at 0.25 the
    // gate REJECTS real FPV motion, the transform degrades, and the GMC
    // features stop being registered to the raw frames.
    if (safe_cfg->gmc_gate_shift > 0) {
      eng->pipe->set_fit_gate(/*max_shift_frac=*/safe_cfg->gmc_gate_shift / 1000.0,
                             /*min_scale=*/safe_cfg->gmc_gate_min / 1000.0,
                             /*max_scale=*/safe_cfg->gmc_gate_max / 1000.0);
    }
    eng->pipe->gpu_prealloc(safe_cfg->gmc_prealloc_pad);
    eng->pipe->gpu_pin_pad(false);
    eng->pipe->set_corner_probe(true);
    // The wall-clock watchdog.  OFF by default: with it on, a frame that
    // blows its budget DEGRADES, so the output stops being bit-reproducible.
    eng->pipe->set_fit_deadline((double)safe_cfg->gmc_deadline_ms, /*hold=*/true);
    eng->pipe->set_ransac_max_iters(safe_cfg->gmc_ransac_max_iters);
    // THE PHYSICAL-FRAME-NUMBER ANCHOR.  OFF by default: the engine anchors
    // on the consumption counter, exactly as before.  ON: it anchors on the
    // camera's physical frame number and DETECTS a missing history frame
    // instead of silently reading a different one.
    eng->pipe->set_phys_anchor(safe_cfg->gmc_phys_anchor != 0);
    // The anchor LK stage's workers.  The count is capped by the engine's
    // core GROUP: more workers than cores would only oversubscribe them, and
    // the point of the group is that every worker lands on its own core.
    // With no group the workers inherit the engine's affinity -- and a GMC
    // engine pinned to ONE core (the gmc_affinity knob) then gets NO speedup
    // from them, which is why the group knob exists.
    {
        std::vector<int> group;
        if (!p->gmc_groups.empty()) group = p->gmc_groups[(size_t)i % p->gmc_groups.size()];
        int workers = safe_cfg->gmc_anchor_threads;
        if (!group.empty() && (int)group.size() < workers) workers = (int)group.size();
        if (workers > 1) {
            std::string cores;
            for (size_t k = 0; k < group.size(); ++k) {
                if (k) cores += ",";
                cores += std::to_string(group[k]);
            }
            fprintf(stderr, "[i] GMC engine %d: anchor LK stage on %d workers, cores %s\n", i,
                    workers, cores.empty() ? "(inherited)" : cores.c_str());
            if (group.empty() && !p->gmc_cpus.empty())
                fprintf(stderr,
                        "[WARN] GMC engine %d asks for %d anchor workers but is pinned to the "
                        "single core %d: the workers will share it and the anchor LK stage gains "
                        "NOTHING.  Give the engine a core GROUP (gmc_affinity_group) instead.\n",
                        i, workers, p->gmc_cpus[(size_t)i % p->gmc_cpus.size()]);
        }
        eng->pipe->set_anchor_threads(workers, group);
    }
    p->engines.push_back(std::move(eng));
  }

  // ---- the NPU services ---------------------------------------------------
  // Each service owns its own RKNN context AND its own RGA canvas, so two
  // services never touch the same buffer.  The board tolerates exactly two
  // concurrent contexts (a third trips the 6 s job timeout and soft-resets
  // the NPU), hence the clamp above.
  for (int i = 0; i < safe_cfg->npu_workers; ++i) {
    auto svc = std::unique_ptr<MosaicPipeline::NpuService>(new MosaicPipeline::NpuService());
    svc->sess.reset(new mosaic::Session());
    if (!mosaic::InitSession(svc->sess.get(), safe_cfg->model_path,
                             static_cast<mosaic::CanvasMode>(safe_cfg->canvas_mode), {}, false,
                             static_cast<mosaic::InputMode>(safe_cfg->input_mode))) {
      mosaic::ReleaseSession(svc->sess.get());
      delete p;
      return fail("InitSession failed (NPU service)");
    }
    // The NPU core mask.  0 == do NOT call rknn_set_core_mask (the shipped
    // behaviour: the runtime picks the cores itself, RKNN_NPU_CORE_AUTO, and
    // the certified baseline was measured that way).  Any other value is set
    // on EVERY context and is a measured DEVIATION.
    //
    // WHY: the RK3588 has THREE NPU cores and nothing ever told the runtime
    // which ones to use.  A single 640x640 fp16 forward pass costs ~108-112
    // ms, and the NPU stage -- not the GMC -- is what caps the pipeline at
    // ~15.8 fps, so the core mask is the only lever left on that side.
    //
    // WHY IT LIVES HERE AND NOT IN mosaic_core.h: the BMP-driven binary
    // (rknn_mosaic_heatmap) shares that header, and a change there would
    // change ITS binary for no reason.  This is the same window the API
    // allows: right after the context is opened and before the first run.
    if (safe_cfg->npu_core_mask != 0) {
      const int rc = rknn_set_core_mask(svc->sess->npu.ctx,
                                        (rknn_core_mask)safe_cfg->npu_core_mask);
      if (rc != 0) {
        fprintf(stderr, "[FATAL] rknn_set_core_mask(0x%x) failed rc=%d\n",
                safe_cfg->npu_core_mask, rc);
        mosaic::ReleaseSession(svc->sess.get());
        delete p;
        return fail("rknn_set_core_mask failed (NPU service)");
      }
      fprintf(stderr,
              "[i] NPU service %d: core mask 0x%x (DEVIATION: the default 0 = do not call it)\n",
              i, safe_cfg->npu_core_mask);
    }
    // Detection thresholds: 0 means "keep the frozen default".
    if (safe_cfg->region_threshold > 0.0) svc->sess->region_threshold = safe_cfg->region_threshold;
    if (safe_cfg->min_area > 0) svc->sess->min_area = safe_cfg->min_area;
    if (safe_cfg->max_detections > 0) svc->sess->max_detections = safe_cfg->max_detections;
    svc->index = (int)p->npus.size();
    p->npus.push_back(std::move(svc));
  }

  // ---- the post-processing service ----------------------------------------
  for (int i = 0; i < safe_cfg->post_workers; ++i) {
    p->post_ws.push_back(std::unique_ptr<mosaic::PostWorkspace>(new mosaic::PostWorkspace()));
    mosaic::InitPostWorkspace(p->post_ws.back().get());
  }

  p->timing.wall_ms = mosaic::now_ms();

  // ---- start the services -------------------------------------------------
  p->ingest_thread = std::thread([p] { p->ingest_loop(); });
  for (auto& e : p->engines) {
    MosaicPipeline::Engine* ep = e.get();
    const size_t ei = (size_t)(&e - &p->engines[0]);
    ep->th = std::thread([p, ep, ei] { p->engine_loop(ep, ei); });
  }
  for (auto& svc : p->npus) {
    MosaicPipeline::NpuService* sp = svc.get();
    sp->th = std::thread([p, sp] { p->npu_loop(sp); });
  }
  for (size_t i = 0; i < p->post_ws.size(); ++i) {
    mosaic::PostWorkspace* ws = p->post_ws[i].get();
    const size_t pi = i;
    p->post_threads.emplace_back([p, ws, pi] { p->post_loop(ws, pi); });
  }
  p->dispatch_thread = std::thread([p] { p->dispatch_loop(); });
  p->art_thread = std::thread([p] { p->artifact_loop(); });

  fprintf(stderr,
          "[i] mosaic_pipeline created: model=%s gmc(anchor_step=%d, window=%d) canvas=%d input=%d "
          "services[gmc=%d npu=%d post=%d] queue=%d input_q=%d pyr_cache=%d gate_shift=%.3f "
          "deadline=%dms ring_slack=%ld job_depth=%zu anchor_threads=%d\n",
          safe_cfg->model_path, safe_cfg->gmc_anchor_step, safe_cfg->gmc_window,
          safe_cfg->canvas_mode, safe_cfg->input_mode, safe_cfg->gmc_workers,
          safe_cfg->npu_workers, safe_cfg->post_workers, safe_cfg->queue_depth,
          safe_cfg->input_queue, safe_cfg->gmc_pyr_cache, safe_cfg->gmc_gate_shift / 1000.0,
          safe_cfg->gmc_deadline_ms, p->ring_slack, p->gmc_job_depth,
          safe_cfg->gmc_anchor_threads);
  return p;
}

const char* mosaic_pipeline_last_error(MosaicPipeline* p) {
  if (!p) return "pipeline is NULL";
  std::lock_guard<std::mutex> lk(p->m_out);
  return p->last_error.c_str();
}

int32_t mosaic_pipeline_inject_fault(MosaicPipeline* p, const char* why) {
  if (!p) return -1;
  // The SAME set_error() the NPU service calls on a refused submit: it records
  // the error, sets the abort flag and WAKES every thread, so the pipeline
  // unwinds instead of wedging.  See the header for why this hook exists.
  p->set_error(why && *why ? why : "injected fault (test hook)");
  return 0;
}

int32_t mosaic_pipeline_effective_config(MosaicPipeline* p, MosaicPipelineConfig* out) {
  if (!p || !out) return -1;
  // p->cfg is the sanitized copy the constructor stored, i.e. what was
  // really applied (after the out-of-range clamping in create()).
  *out = p->cfg;
  return 0;
}

int32_t mosaic_pipeline_set_result_callback(MosaicPipeline* p, MosaicResultCallback cb,
                                            void* user) {
  if (!p) return -1;
  {
    std::lock_guard<std::mutex> lk(p->m_out);
    p->cb = cb;
    p->cb_user = user;
  }
  p->cv_out.notify_all();
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
  int64_t id;
  {
    std::lock_guard<std::mutex> lk(p->m_in);
    if (p->eof || p->abort) return -1;
    id = p->next_id++;
    InItem item;
    item.id = id;
    // One copy, and it lands in an owning Mat (see InItem).
    item.gray = cv::Mat(h, w, CV_8UC1, const_cast<uint8_t*>(gray)).clone();
    // NEVER BLOCK: a 25 fps camera must not be stalled by a pipeline that
    // cannot keep up.  When the camera queue is full a frame MUST go, so the
    // only question is WHICH one -- see gmc_input_drop_policy in the header.
    //
    // POLICY 0 (the default, and the shipped behaviour): the OLDEST queued
    // frame goes and the freshest one is kept.  A live stream wants FRESH
    // data: under a sustained overload policy 1 would keep the queue full of
    // OLD frames and the GMC service would fall further and further behind
    // (unbounded latency), while policy 0 keeps it working on the newest
    // frames (bounded latency).
    //
    // POLICY 1: the incoming (newest) frame is REJECTED and the queued frames
    // are kept.  The GMC service then consumes a CONTIGUOUS run of frames, so
    // the hole lands in the history a couple of frames later instead of
    // immediately -- the price is the staleness described above.
    //
    // EITHER WAY THE LOSS IS NOT BENIGN.  The physical frame id is consumed
    // at push time, so a lost frame leaves a HOLE in the id sequence and the
    // frames whose 21-frame history reaches across that hole have a stretched
    // temporal window.  gmc_phys_anchor=1 is what makes that hole DETECTED
    // (timing.gmc_history_gaps) instead of silent; the policy only decides
    // WHICH frames are lost and how stale the output is.
    //
    // When it does fire, say so loudly -- a silent registration break looks
    // like a model quality problem, and it is not.
    if (p->in_q.size() >= p->input_depth) {
      const bool reject_newest = p->cfg.gmc_input_drop_policy != 0;
      if (!reject_newest) p->in_q.pop_front();
      std::lock_guard<std::mutex> lt(p->m_time);
      p->timing.dropped += 1;
      if (p->timing.dropped == 1) {
        fprintf(stderr,
                "[WARN] a GMC FEATURE was LOST: the camera queue overflowed and %d "
                "frame(s) never reached the GMC service, so their features do not "
                "exist and the registration chain is broken from here on.  The GMC "
                "service is not keeping up with the source -- the inference skipping "
                "frames is fine, this is not\n",
                (int)p->timing.dropped);
      }
      if (reject_newest) {
        // The incoming frame is the one that goes: do not enqueue it.
        return id;
      }
    }
    p->in_q.push_back(std::move(item));
  }
  p->cv_in.notify_all();
  // The header promises the FRAME ID, not a generic 0.  The contract test
  // caught the implementation returning a constant 0: a caller that pushes a
  // frame that gets REJECTED (wrong size) and then pushes the next one would
  // see two identical ids and could not tell which frame it is holding.
  return id;
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
    for (int32_t i = 0; i < ncopy; ++i) {
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
  for (auto& e : p->engines) {
    e->cv_job.notify_all();
    e->cv_space.notify_all();
  }
  p->cv_feat.notify_all();
  p->cv_heat.notify_all();
  p->cv_guard.notify_all();
  p->cv_art.notify_all();

  if (p->ingest_thread.joinable()) p->ingest_thread.join();
  for (auto& e : p->engines) {
    if (e->th.joinable()) e->th.join();
  }
  p->gmc_done = true;
  p->cv_feat.notify_all();
  // The engines are done, so their final corner statistics are final.
  {
    std::lock_guard<std::mutex> lk(p->m_time);
    for (auto& e : p->engines) {
      if (e->pipe) p->timing.gmc_corner_capped += e->pipe->corner_cap_frames();
    }
  }
  // THE COMPUTATION SNAPSHOT's coverage, per engine.
  //
  // WHY it is printed and not folded into the timing struct: the timing
  // struct is the PUBLIC ABI (mosaic_pipeline.h) and its layout is frozen.
  // This number is a correctness statement, not a timing one, so it goes to
  // stderr as its own line.
  //
  // It MUST be 0 on a zero-drop run: a non-zero value means the snapshot did
  // not cover a history read and the run fell back to the ring -- correct
  // numbers, but the structural guarantee is gone, and that has to be
  // visible rather than inferred.
  for (size_t i = 0; i < p->engines.size(); ++i) {
    if (!p->engines[i]->pipe) continue;
    const long miss = p->engines[i]->pipe->n_snapshot_miss();
    const long frames = p->engines[i]->seq_frame;
    const double snap_ms =
        frames > 0 ? p->engines[i]->pipe->stats().t_snapshot / (double)frames : 0.0;
    std::fprintf(stderr,
                 "[SNAPSHOT] engine %zu: history reads not served from the private snapshot = %ld%s"
                 " | snapshot copy cost %.2f ms/frame\n",
                 i, miss, miss ? "  (the run fell back to the ring -- the structural guarantee is OFF)"
                               : "  (every history read came from the private snapshot)",
                 snap_ms);
  }
  for (auto& svc : p->npus) {
    if (svc->th.joinable()) svc->th.join();
  }
  p->feat_done = true;
  p->cv_heat.notify_all();
  for (auto& t : p->post_threads) {
    if (t.joinable()) t.join();
  }
  p->cv_art.notify_all();
  if (p->art_thread.joinable()) p->art_thread.join();
  {
    std::lock_guard<std::mutex> lk(p->m_out);
    p->flushed = true;
  }
  // The run is over: one summary line for the three fallback lines, so an
  // operator can see at a glance which one (if any) fired.
  {
    std::lock_guard<std::mutex> lk(p->m_time);
    const int64_t cam = p->timing.dropped;
    const int64_t feat = p->timing.features_dropped;
    const int64_t det = p->timing.detections_dropped;
    const int64_t blows = p->timing.gmc_deadline_blows;
    if (cam || feat || det || blows) {
      fprintf(stderr,
              "[i] fallback summary: camera-queue drops %lld (BREAKS the GMC registration), "
              "features skipped by the inference %lld, detections skipped %lld, "
              "GMC fits degraded by the watchdog %lld\n",
              (long long)cam, (long long)feat, (long long)det, (long long)blows);
    } else {
      fprintf(stderr,
              "[i] fallback summary: clean -- no camera-queue drop, no skipped feature, "
              "no skipped detection, no degraded GMC fit\n");
    }
  }
  p->cv_out.notify_all();
  if (p->dispatch_thread.joinable()) p->dispatch_thread.join();
  {
    std::lock_guard<std::mutex> lk(p->m_time);
    p->timing.wall_ms = mosaic::now_ms() - p->timing.wall_ms;
  }
  if (p->mats_out) { std::fclose(p->mats_out); p->mats_out = nullptr; }
  if (p->md5_out) { std::fclose(p->md5_out); p->md5_out = nullptr; }
  return 0;
}

int32_t mosaic_pipeline_get_timing(MosaicPipeline* p, MosaicPipelineTiming* out) {
  if (!p || !out) return -1;
  std::lock_guard<std::mutex> lk(p->m_time);
  // The whole struct is copied, so the ingest breakdown (gmc_ingest_in_wait_ms,
  // gmc_ingest_window_ms, gmc_ingest_space_ms) and the job-queue depth
  // counters (gmc_queue_depth_sum / gmc_queue_samples) come along with it.
  *out = p->timing;
  // The fit distribution is computed here rather than accumulated, so it is
  // always consistent with the samples collected SO FAR (a partially completed
  // run still reports a usable distribution).
  out->gmc_fit_samples = (int64_t)p->gmc_fit_samples.size();
  if (!p->gmc_fit_samples.empty()) {
    std::vector<double> v = p->gmc_fit_samples;
    std::sort(v.begin(), v.end());
    out->gmc_fit_p50_ms = pct_nearest_rank(v, 0.50);
    out->gmc_fit_p95_ms = pct_nearest_rank(v, 0.95);
    out->gmc_fit_p99_ms = pct_nearest_rank(v, 0.99);
  }
  // The same treatment for the single rknn_run.
  out->rknn_run_samples = (int64_t)p->rknn_run_samples.size();
  if (!p->rknn_run_samples.empty()) {
    std::vector<double> v = p->rknn_run_samples;
    std::sort(v.begin(), v.end());
    out->rknn_run_p50_ms = pct_nearest_rank(v, 0.50);
    out->rknn_run_p95_ms = pct_nearest_rank(v, 0.95);
    out->rknn_run_p99_ms = pct_nearest_rank(v, 0.99);
  }
  if (p->gmc_corner_dets > 0.0) {
    out->gmc_corner_mean = p->gmc_corner_pts / p->gmc_corner_dets;
  }
  return 0;
}

const char* mosaic_pipeline_feature_md5(MosaicPipeline* p) {
  if (!p) return "";
  std::lock_guard<std::mutex> lk(p->m_md5);
  return p->seq_md5_hex.c_str();
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

void mosaic_pipeline_destroy(MosaicPipeline* p) {
  if (!p) return;
  // Destroy does NOT flush (the header says so): stop the threads, join them,
  // release the contexts.
  p->abort = true;
  // The SAME wake-up set the error path uses: every thread that can be asleep
  // on a condition variable has to be woken, or the join below never returns.
  p->wake_everyone();
  if (p->ingest_thread.joinable()) p->ingest_thread.join();
  for (auto& e : p->engines) {
    if (e->th.joinable()) e->th.join();
  }
  for (auto& svc : p->npus) {
    if (svc->th.joinable()) svc->th.join();
    if (svc->sess) mosaic::ReleaseSession(svc->sess.get());
  }
  for (auto& t : p->post_threads) {
    if (t.joinable()) t.join();
  }
  if (p->art_thread.joinable()) p->art_thread.join();
  if (p->dispatch_thread.joinable()) p->dispatch_thread.join();
  if (p->mats_out) { std::fclose(p->mats_out); p->mats_out = nullptr; }
  if (p->md5_out) { std::fclose(p->md5_out); p->md5_out = nullptr; }
  delete p;
}

}  // extern "C"
