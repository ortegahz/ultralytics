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
//
// DEPRECATED (v9): the OPPORTUNISTIC scheduler has no job queue any more --
// the engine grabs the newest seeded frame out of the history ring itself.
// The struct is KEPT because Engine::jobs is still declared with this type
// (nothing is ever pushed into it), and deleting it would have touched the
// engine's layout for no benefit.
struct GmcJob {
  int64_t id = 0;
  long abs = 0;  // the engine's absolute frame index
  // THE ARTIFACT SEQUENCE: the DENSE index of the artifact this job will
  // produce.  It is NOT the consumption counter `abs`: only the frames that
  // become a group produce an artifact, so `abs` has GAPS -- and the
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
    // ------------------------------------------------------------------------
    // DEPRECATED (v9): THE ENGINE JOB QUEUE of the in-order scheduler.
    //
    // Under the OPPORTUNISTIC scheduler the engine does not take frames off a
    // queue: it grabs the NEWEST frame out of the history ring, and the
    // frames it skips are simply not computed.  Nothing is pushed here any
    // more, so `jobs` stays empty forever.
    //
    // The members are KEPT (not deleted) on purpose: they are part of this
    // struct's layout and of the code that reads them, and deleting them
    // would drag unrelated logic into the change.  They are simply never
    // used, and every reader below says so.
    // ------------------------------------------------------------------------
    std::deque<GmcJob> jobs;
    std::mutex m;
    std::condition_variable cv_job;   // DEPRECATED: no jobs are posted any more
    std::condition_variable cv_space;  // DEPRECATED: no jobs are posted any more
    // The oldest absolute frame this engine still owes.  `has_pending` is
    // false while the engine is idle.
    // DEPRECATED (v9): the in-order scheduler's backlog bookkeeping.  The
    // opportunistic engine never owes anything -- it takes the newest frame
    // and skips the rest -- so these two stay at their initial values.
    long oldest_pending = 0;
    bool has_pending = false;
    // THE HEAD WAKE-UP (v9): the ingest notifies it after it seeded a new
    // frame, so the engine can grab the newest one instead of waiting for a
    // job to be posted.
    std::condition_variable cv_head;
    // THE PHYSICAL ID of every frame this engine was seeded with, in the
    // order it was seeded, paired with the INDEX it was seeded under.
    //
    // WHY IT IS NEEDED: the engine computes the seeded INDEX (the consumption
    // counter, or the camera's physical number when the physical anchor is
    // on), but the artifacts and the RESULTS have to carry the camera's
    // PHYSICAL frame number -- that is the id the caller sees and the id the
    // reference's feature_md5.txt names.  The ingest knows both (it seeds
    // with one and receives the other) and hands the pair over here.
    //
    // The engine pops the entries it has passed (the frames it skipped), so
    // the deque only holds the frames between the last computed one and the
    // head.
    std::deque<std::pair<long, int64_t>> phys_ids;
    // THE NEWEST INDEX THIS ENGINE HAS DEALT WITH (computed or voided).  -1
    // until the first frame arrives.  Only the engine's own thread touches
    // it, so it needs no lock.
    long last_done = -1;
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
  // v10: BUFFER A -- the ring of NPU-ready slots between Stage 1 and Stage 2.
  //
  // A slot holds ONE model input, already widened to fp16, so the hand-off
  // between the two stages is an INTEGER (the slot id) and costs 0.00 ms of
  // copying: Stage 1 writes the NEON output straight into the slot's memory.
  //
  // THE SLOT STATE MACHINE.  Every transition happens under buf_a->m, so no
  // stage can ever see a half-written slot:
  //
  //   kSlotFree     the slot is free, Stage 1 may write into it
  //   kSlotReady    Stage 1 has filled it, Stage 2 may take it
  //   kSlotInflight the slot is OWNED: either Stage 1 is writing into it or
  //                 Stage 2 has it bound to its context
  //
  // WHY INFLIGHT COVERS BOTH ROLES: Stage 1 must be sure that the slot it is
  // about to write is not being read by Stage 2, and Stage 2 must be sure that
  // the slot it is about to bind is not being written.  One "owned" state
  // covers both, and the owner is unambiguous because the slot is marked
  // INFLIGHT under the ring's own lock, which is what makes the cross-service
  // delivery safe: whichever Stage 1 thread takes the slot owns it until it
  // publishes it as READY.
  enum { kSlotFree = 0, kSlotReady = 1, kSlotInflight = 2 };
  struct BufferA {
    struct Slot {
      rknn_tensor_mem* mem = nullptr;  // the DMA slot (the DMA arm)
      uint8_t* virt = nullptr;         // its host-visible address
      std::vector<uint16_t> host;      // the host arm's copy of the same slot
      uint32_t bytes = 0;              // the slot's byte size
      int state = kSlotFree;
      int64_t frame_id = 0;            // the camera's PHYSICAL frame number
    };
    std::vector<Slot> slots;
    std::deque<int> ready;      // the READY slot ids, oldest first
    std::mutex m;
    std::condition_variable cv;
    rknn_context owner_ctx = 0;  // the context that created the mems
    size_t elems = 0;            // the fp16 elements one slot holds
    // THE MODEL'S INPUT SIZE IN BYTES (without the stride padding).  It is the
    // size the host arm has to declare in rknn_input.size, because with
    // pass_through=1 the runtime reads exactly that many bytes.
    size_t tensor_bytes = 0;
  };

  struct NpuService {
    std::unique_ptr<mosaic::Session> sess;
    std::thread th;
    int index = 0;   // its position in the service list (for the core list)
    // v10: the DECOUPLED arm.  The Buffer A ring's MEMORY belongs to THIS
    // service and to nobody else: a slot's memory was created by THIS
    // service's context and rknn_set_io_mem binds it to the same context, so
    // there is never a cross-context binding.  The WRITER of a slot does not
    // have to be this service's Stage 1 thread any more -- the cross-service
    // delivery picks the service with room (see pick_target) -- but the
    // bytes in a slot are still read back by THIS service's Stage 2, with
    // THIS service's context.
    std::unique_ptr<BufferA> buf_a;
    // Stage 1's own canvas-only session: it composes into its OWN canvas and
    // has NO RKNN context (the board tolerates exactly two, and the NPU
    // services already use both).
    std::unique_ptr<mosaic::Session> pre_sess;
    std::thread pre_th;
    // v10 diagnostics: WHAT THIS SERVICE ACTUALLY DID.
    //
    // The two services run the SAME model at the SAME cost (121.9 ms/frame
    // each on the board), so their frame counts SHOULD match.  If they do
    // not, the Stage 1 -> Stage 2 pairing is feeding them unevenly -- which
    // is what the per-core hardware load points at (core0 mean 83.45% vs
    // core1 mean 41.28% on CANON21).
    //
    // THEY LIVE HERE, AND NOT IN THE PUBLIC TIMING STRUCT, because they are
    // per SERVICE: the public ABI is frozen at version 10, and a new field
    // in MosaicPipelineTiming would bump it.
    //
    // THE LOCK: the two drop counters are written by whichever Stage 1
    // thread wrote into THIS service's ring -- which, since the cross-service
    // delivery, is not necessarily THIS service's own Stage 1 thread -- and
    // read by its Stage 2 thread when that thread exits.  The two overlap
    // while an abort tears the pipeline down, so they are guarded by m_time,
    // like every other counter in this class.
    int64_t frames_run = 0;       // frames this service handed to rknn_run
    int64_t bufA_reclaimed = 0;   // frames this service's ring dropped (oldest READY)
    int64_t bufA_full_drops = 0;  // frames this service could not take (all slots INFLIGHT)
    //
    // v11 diagnostics: THE SERVICE'S OWN NPU CAPACITY.
    //
    // stage2_busy_ms is the wall clock this service's Stage 2 thread spent
    // INSIDE rknn_run, accumulated over every forward it ran.  It is the very
    // same quantity as the global timing.run_ms / timing.npu_busy_ms (they
    // are fo.run, which RunFrame measures with now_ms() around the rknn_run
    // call, mosaic_core.h), but kept PER SERVICE: the global buckets merge the
    // two services, and the whole point of this counter is to tell them
    // apart.
    //
    // WHY IT IS NEEDED: the two services do NOT have the same capacity.  On
    // CANON29 service 0's NPU core was busy 84.47% of the time and service
    // 1's only 55.82%, so a 1:1 feed leaves service 1 idle ~44% of the time
    // and pushes service 0 into its ring's overflow -- which is exactly what
    // was measured (all 850 Buffer A reclaims on service 0's ring, none on
    // service 1's).  The capacity estimate w_i = frames_run / stage2_busy_ms
    // (frames per millisecond of NPU time) is what turns that observation
    // into a delivery ratio; see pick_target.
    //
    // THE LOCK: frames_run and stage2_busy_ms are written by THIS service's
    // Stage 2 thread and READ by the Stage 1 threads (they compute w_i), so
    // they are guarded by m_time -- the same lock that already guards the two
    // drop counters.
    int64_t stage2_busy_ms = 0;
    // THE EXACT ACCUMULATOR: one rknn_run is a FRACTIONAL number of
    // milliseconds, so the sum is kept in a double and the int64 above is its
    // rounded mirror.  Truncating every frame's fraction would bias the
    // capacity estimate of the two services by a fraction of a percent.
    double stage2_busy_acc_ms = 0.0;
    // v11: THE FRAMES STAGE 1 DELIVERED INTO THIS SERVICE'S RING.
    //
    // It counts every frame Stage 1 WROTE into this service's Buffer A --
    // including the ones the ring later reclaimed -- so next to frames_run
    // (the frames the service ACTUALLY ran) it says how much work was
    // OFFERED to this service, and the difference between the two is the
    // service's own loss.
    //
    // THE LOCK: written by whichever Stage 1 thread wrote into THIS ring, so
    // it is guarded by m_time like the two drop counters.
    int64_t frames_delivered = 0;
  };
  std::vector<std::unique_ptr<NpuService>> npus;

  // ---- per-service CPU placement -----------------------------------------
  // Empty = inherit the process affinity.  Otherwise the Nth thread of that
  // service is pinned to cpus[N % cpus.size()].
  std::vector<int> gmc_cpus;
  std::vector<int> npu_cpus;
  // v10: the decoupled stages' core lists.  pre_cpus carries Stage 1 AND
  // Stage 3 (both are CPU work of the same family); npu_stage_cpus carries
  // Stage 2.  The NPU stage's list FOLLOWS the pre list unless it was given
  // its own.
  std::vector<int> pre_cpus;
  std::vector<int> npu_stage_cpus;
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
  //
  // v9: the ARTIFACT SEQUENCE is now produced by the ENGINE (the engine
  // decides which frames become groups), so it is an ATOMIC: with the
  // opportunistic scheduler there is exactly one engine, but the type makes
  // the hand-off safe if that ever changes, and it costs one uncontended
  // fetch-add per group.
  std::atomic<long> artifact_seq{0};
  // HOW MANY FRAMES THE INGEST SEEDED INTO THE HISTORY.  It is the
  // denominator of "how many frames did the camera deliver that the engine
  // did not compute": seeded - groups_done - void_frames = frames skipped.
  std::atomic<long> gmc_seeded_frames{0};
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
  // v10: HOW MANY STAGE 1 THREADS ARE STILL RUNNING.  It is the signal Stage 2
  // waits for before it may conclude that no slot will ever arrive again: a
  // Stage 1 thread may still be composing a frame that has not reached the
  // ring yet, so "the ring is empty" alone does NOT mean "the work is over".
  std::atomic<int> pre_active{0};

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
  // v10: THE DEPTH OF BUFFER B (the heat queue) IN THE DECOUPLED MODE.  The
  // fused mode keeps using queue_depth (8), so the certified arm does not
  // move; the decoupled mode uses buffer_b_depth (3 by default).
  size_t heat_depth = 8;
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
      e->cv_head.notify_all();
      e->cv_job.notify_all();
      e->cv_space.notify_all();
    }
    cv_feat.notify_all();
    cv_heat.notify_all();
    cv_guard.notify_all();
    cv_art.notify_all();
    cv_out.notify_all();
    // v10: the Buffer A rings.  Stage 2 waits on their condition variable, so
    // an abort has to wake it -- otherwise the join in the destroy path would
    // never return.
    for (auto& svc : npus) {
      if (svc->buf_a) svc->buf_a->cv.notify_all();
    }
  }

  // A PER-FRAME error: this frame is rejected, the stream keeps running.
  // A camera stream must not die because one frame was malformed -- and a
  // malformed frame must not stall the camera either.
  void note_error(const std::string& e) {
    std::lock_guard<std::mutex> lk(m_out);
    if (last_error.empty()) last_error = e;
  }

  // -------------------------------------------------------------------------
  // THE HISTORY WINDOW GUARD -- NOW A PURE PREDICATE, NEVER A WAIT.
  //
  // 它曾经问的是「producer 能不能跑到最老待算帧前面 ring_slack 帧以外」，
  // 而 INGEST 会【阻塞等待】这个答案为真。那条等待就是杀死相机队列的
  // 回压路径：ingest 一次 20.9 ms 的停顿就足以顶爆 64 深的相机 FIFO，
  // 在物理帧号上打出洞。
  //
  // 在机会主义调度下答案恒为真，原因是三条：
  //   * 二级快照是零拷贝 PIN（浅引用），producer 改写 ring 槽位只是重新绑定
  //     指针，正在跑的拟合读到的像素不变；
  //   * 环深 64 而一次计算只读 43 帧，producer 可以领先 21 帧（840 ms）
  //     才可能覆写最老的历史；
  //   * 万一真被覆写，`history_complete()` 会【检测到并作废该帧】，绝不在
  //     撕裂数据上计算。
  //
  // 为什么函数还留着：它记录的正是「producer 能不能前进」这个问题所在的
  // 位置，而且多引擎路径（gmc_workers > 1）仍然需要判断历史是否被覆写。
  // -------------------------------------------------------------------------
  bool window_ok(long abs) const {
    (void)abs;
    return true;
  }

  // -------------------------------------------------------------------------
  // THE INGEST: raw frames -> the engines' history rings.
  //
  // It is the only thread that writes the history, so the ring state has a
  // single writer.  It seeds EVERY engine with EVERY frame (each engine may
  // need any frame as history).
  //
  // v9 -- IT NEVER WAITS.  It used to block on the history-window guard and
  // on the engines' job queues, and that back-pressure was the thing that
  // killed the camera queue: one 20.9 ms stall in the ingest was enough to
  // overflow a 64-deep camera FIFO and punch a hole in the physical frame
  // numbers.  Now the ingest's whole job is: take the frame, seed it into
  // every engine's history, record its physical id, and tell the engine that
  // the head moved.  The engine decides on its own what to compute.
  // -------------------------------------------------------------------------
  void ingest_loop() {
    for (;;) {
      const auto t_ingest0 = std::chrono::steady_clock::now();
      // The ingest's own accounting of THIS frame: how long it waited for the
      // camera.  The two other waits (the history window and the job queue's
      // space) are GONE: the ingest does not block on them any more, so their
      // counters stay at 0 -- which is the evidence that the back-pressure
      // path is closed.
      double in_wait_ms = 0.0;
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

      // Seed every engine: each of them may need this frame as history.
      //
      // The PHYSICAL ID is recorded FIRST, and the head is published by
      // seed_frame() AFTER it: the engine must never see a head whose id it
      // cannot look up yet.
      for (auto& e : engines) {
        {
          std::lock_guard<std::mutex> lk(e->m);
          e->phys_ids.emplace_back(eng_idx, item.id);
        }
        e->pipe->seed_frame(eng_idx, item.gray);
        // The head moved: wake the engine so it can grab the newest frame.
        e->cv_head.notify_one();
      }
      gmc_seeded_frames.fetch_add(1, std::memory_order_relaxed);

      {
        std::lock_guard<std::mutex> lt(m_time);
        timing.gmc_ingest_in_wait_ms += in_wait_ms;
        // gmc_ingest_window_ms and gmc_ingest_space_ms are NOT incremented:
        // the ingest does not wait on the history window any more, and it
        // does not wait for room in a job queue (there is no job queue).
        // Their reading of 0 IS the measurement: the camera is never
        // stalled by the GMC service.
        timing.gmc_ingest_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ingest0).count();
      }
    }
    ingest_done = true;
    // Wake everybody: the engines take their last head and stop, the upstream
    // services see that no more features are coming.
    cv_in.notify_all();
    for (auto& e : engines) {
      e->cv_head.notify_all();
      e->cv_job.notify_all();
      e->cv_space.notify_all();
    }
    cv_feat.notify_all();
    cv_guard.notify_all();
  }

  // -------------------------------------------------------------------------
  // SERVICE GMC: one engine computes the NEWEST frame it can get.
  //
  // v9 -- THE OPPORTUNISTIC SCHEDULER.  The engine no longer takes frames off
  // a queue in order; it asks the history ring for the NEWEST seeded frame
  // and computes THAT one.  The frames it passes are not lost: they were
  // seeded into the history, so they are the lags the next fit reads -- and
  // that is the whole reason skipping is safe here (a frame skipped by the
  // engine NEVER punches a hole in the history, unlike a frame lost at the
  // camera queue).
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
    // THE MINIMUM GAP BETWEEN TWO GROUPS, in the index space the engine
    // computes in.  1 == no limit: grab the newest frame as soon as it is
    // there (the adaptive default).  >1 == an artificial rate limit.
    const long min_stride = cfg.gmc_min_stride > 0 ? cfg.gmc_min_stride : 1;
    for (;;) {
      // ---- 1) WAIT FOR A HEAD THAT IS NEW ENOUGH --------------------------
      long abs = 0;
      {
        const auto t_wait0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(eng->m);
        eng->cv_head.wait(lk, [this, eng, min_stride] {
          if (abort || ingest_done) return true;
          const long h = eng->pipe->latest_seeded();
          return h > eng->last_done && (h - eng->last_done) >= min_stride;
        });
        const double waited =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wait0)
                .count();
        {
          std::lock_guard<std::mutex> lt(m_time);
          timing.gmc_wait_ms += waited;
        }
        if (abort) break;
        const long h = eng->pipe->latest_seeded();
        if (h > eng->last_done && (h - eng->last_done) >= min_stride) {
          abs = h;
        } else {
          // The producer is done and what is left does not satisfy the
          // minimum gap (or it was already computed): there is nothing more
          // to come.  Note that the engine does NOT go back for the frames it
          // passed -- the whole point is to work on the freshest data.
          if (ingest_done) break;
          continue;  // a spurious wakeup
        }
      }

      // ---- 2) THE PHYSICAL ID OF THAT FRAME -------------------------------
      //
      // The engine computed the SEEDED INDEX (the consumption counter, or the
      // camera's physical number when the physical anchor is on), but the
      // artifacts and the results have to carry the camera's PHYSICAL frame
      // number -- that is the id the caller sees.
      int64_t phys_id = abs;
      {
        std::lock_guard<std::mutex> lk(eng->m);
        // Drop the ids of the frames this engine passed (the ones it skipped).
        while (!eng->phys_ids.empty() && eng->phys_ids.front().first < abs)
          eng->phys_ids.pop_front();
        if (!eng->phys_ids.empty() && eng->phys_ids.front().first == abs) {
          phys_id = eng->phys_ids.front().second;
        } else {
          // Should not happen: the ingest records the id BEFORE it publishes
          // the head, so an id is always there when the head is.  Loud anyway:
          // a wrong frame id in the results is a wrong answer, not a slowdown.
          fprintf(stderr,
                  "[WARN] GMC engine %zu: no physical id for the seeded index %lld -- the frame id "
                  "falls back to the seeded index\n",
                  eng_index, (long long)abs);
        }
      }

      // ---- 3) CONTINUITY: IS THE WHOLE HISTORY STILL THERE? --------------
      //
      // The producer is never blocked, so it is EXPECTED to run ahead of the
      // engine.  Once it is more than (ring_depth - 43) frames ahead, the
      // frames this fit needs have been OVERWRITTEN in the ring -- and a
      // camera frame that was lost does the same thing to the lags that reach
      // across the hole.  In both cases the frame is VOIDED: it is not
      // computed at all, rather than fitted against torn pixels.
      if (!eng->pipe->history_complete(abs)) {
        eng->last_done = abs;
        {
          std::lock_guard<std::mutex> lt(m_time);
          timing.gmc_void_frames += 1;
        }
        fprintf(stderr,
                "[WARN] GMC frame %lld (engine %zu) VOIDED: its 21-frame history is NOT whole "
                "(the producer ran too far ahead of the engine, or a camera frame was lost) -- the "
                "frame is NOT computed, and the engine does NOT go back for it\n",
                (long long)phys_id, eng_index);
        continue;
      }

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
      // otherwise.  `abs` IS that index -- it is the index the producer
      // seeded this frame under, which is exactly what process_frame() reads.
      const long eng_idx = abs;
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
      //
      // THE ARTIFACT SEQUENCE is now minted HERE, by the engine: under the
      // opportunistic scheduler the engine decides which frames become
      // groups, so the ingest cannot number them any more.  It stays DENSE
      // (1, 2, 3, ...) because the writer waits for the next CONTIGUOUS index.
      publish_artifact(++artifact_seq, phys_id, fh, mats);

      // THE AUTHORITATIVE COUNT OF THE GMC GROUPS THAT COMPLETED.
      //
      // One increment per group the GMC service actually computed, i.e. per
      // frame a feature really exists for.  It is the ONLY denominator that
      // holds for every scheduler: the number of PUSHED frames does not (the
      // opportunistic engine SKIPS the frames it could not keep up with, and
      // those frames are seeded as history and produce nothing), and the
      // number of frames that reached the GMC service does not either (the
      // camera queue may have dropped some of them).
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
                (long long)phys_id, eng_index, cfg.gmc_deadline_ms);
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
                (long long)phys_id, eng_index, gaps_after - gaps_before);
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
                      (long long)phys_id);
        cv::imwrite(nm, hwc);
      }

      // PLANE-MAJOR -> model order [residual, diff, I_t].
      // out is a 3 x (H*W) CV_8UC1 Mat: row 0 = I_t, row 1 = diff,
      // row 2 = residual.
      const size_t plane = static_cast<size_t>(height) * static_cast<size_t>(width);
      FeatureItem fi;
      fi.id = phys_id;
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

      // The engine has finished this frame.  Its "newest index dealt with"
      // moves forward, so the next head becomes eligible -- and the frames
      // this engine PASSED are simply dropped: they were seeded into the
      // history, so they are the lags the NEXT fit reads, and they are not
      // computed on their own.
      //
      // THERE IS NO NOTIFY FOR THE INGEST: it never waits, so there is
      // nothing to unblock.
      eng->last_done = abs;
    }
  }

  // -------------------------------------------------------------------------
  // v10: THE TARGET OF A STAGE 1 FRAME.
  //
  // WHY THE FRAME IS NOT PINNED TO "MY" SERVICE ANY MORE: Stage 1 used to
  // be PAIRED 1:1 with the NPU services -- thread i fed service i -- and on
  // the board that pairing turned "whose NPU is faster" into a POSITIVE
  // FEEDBACK LOOP.  CANON23 measured service 0 running 3616 frames against
  // service 1's 2223 (1.63 : 1), the per-core NPU load at 85.13% : 44.48%,
  // and ALL 1328 Buffer A drops on service 0's ring while service 1's ring
  // was never even full.  The mechanism is mechanical: both Stage 1 threads
  // pull from the SAME feature queue, each writes only into ITS OWN ring,
  // and a thread whose ring is full throws the frame away (or overwrites its
  // own oldest READY frame).  The service whose NPU is fast therefore keeps
  // taking frames from the shared queue, so it stays fast -- and the other
  // service starves.
  //
  // v11: THE ROTATION FIXED THE FRAME SPLIT, NOT THE CAPACITY SPLIT.  A
  // round-robin cursor over the services (the rule v10 ended with) delivered
  // CANON29 at a PERFECT 1.000:1 frame balance -- 3136 frames to service 0
  // and 3135 to service 1 -- and it still lost 850 frames to the rings and
  // still ran at 13.245 fps against a 15.04 fps supply.  The reason: the two
  // services do NOT have the same capacity.  In the same run service 0's NPU
  // core was busy 84.47% of the time and service 1's only 55.82%, and ALL
  // 850 Buffer A reclaims landed on service 0's ring (service 1: 0).  A 1:1
  // feed hands the slow service exactly as many frames as the fast one, so
  // the fast service idles ~44% of the time while the slow one overflows.
  // Balancing the FRAMES is therefore the wrong invariant; balancing the
  // CAPACITY is the right one -- hence the capacity-weighted rule below.
  //
  // So the frame now goes to the service that CAN take it:
  //
  //   * a service with a FREE slot is a candidate, and among the candidates
  //     the one with the highest CAPACITY-WEIGHTED score wins:
  //         score_i = w_i / (served_i + 1)
  //     where w_i is that service's MEASURED capacity (frames per millisecond
  //     of NPU time: frames_run / stage2_busy_ms) and served_i is how many
  //     frames this Stage 1 thread has already handed to it.  This is the
  //     standard weighted-fair-queueing rule, and it settles at
  //     served_i PROPORTIONAL TO w_i -- so every service is fed at its OWN
  //     rate: the slow one is offered fewer frames and stops overflowing,
  //     the fast one is offered more and stops idling.  When the two
  //     services have the SAME capacity the scores tie and the rule
  //     degenerates to 1:1 (the rotation it replaced), so a symmetric board
  //     behaves exactly as before.
  //   * if NO service has a FREE slot, the OLDEST READY frame is reclaimed
  //     ("drop the oldest, keep the newest"), and it is taken from the
  //     service holding the LONGEST READY queue -- that is where the stalest
  //     pending frame sits.
  //   * if every slot of every ring is IN FLIGHT there is nothing to
  //     reclaim and the frame itself is dropped.  The stage NEVER waits: a
  //     wait here would push the back-pressure back into the GMC.
  //
  // WHY A SLOT MAY BELONG TO ANOTHER SERVICE: the slot's memory was created
  // by rknn_create_mem() with the OWNING service's context, and
  // rknn_set_io_mem() requires the memory to be bound to the context that
  // owns it -- not that the thread that WROTE the bytes belongs to that
  // service.  Stage 2 still binds the slot with its OWN (owning) context, so
  // the ctx/memory pairing is exactly as strict as it was; only the WRITER
  // moved.
  // -------------------------------------------------------------------------
  //
  // THE STATE OF ONE RING, as the target selection sees it.
  struct RingProbe {
    size_t ready = 0;       // the READY queue's length
    size_t free_slots = 0;  // the slots that are FREE
    size_t inflight = 0;    // the slots that are IN FLIGHT
  };

  // Snapshot ONE ring.
  //
  // THE LOCK: only THIS ring's lock is taken, and it is released before the
  // next ring is looked at.  TWO RING LOCKS ARE NEVER HELD AT THE SAME TIME:
  // a thread holding ring A while waiting for ring B, against another
  // thread holding ring B while waiting for ring A, would hang the whole
  // pipeline.  One lock at a time also keeps the file's global lock order
  // ({m_feat, m_heat, buf_a->m} -> m_time) intact: this function takes
  // buf_a->m and NOTHING else.
  RingProbe probe_ring(BufferA* ba) {
    RingProbe rp;
    if (!ba) return rp;
    std::lock_guard<std::mutex> lk(ba->m);
    rp.ready = ba->ready.size();
    for (const BufferA::Slot& sl : ba->slots) {
      if (sl.state == kSlotFree) ++rp.free_slots;
      else if (sl.state == kSlotInflight) ++rp.inflight;
    }
    return rp;
  }

  // WHICH SERVICE TAKES THE FRAME.
  //
  // THE INPUTS ARE THE OBSERVATION, NOT THE FEEDER'S STATE: the caller hands
  // in the ring snapshot, the measured capacity of every service and the
  // number of frames it has already handed to each of them.  The function is
  // a PURE FUNCTION of (probe, capacity, served): no clock, no random, no
  // hidden state.  That is what makes a run reproducible -- the same input
  // sequence and the same configuration produce the same sequence of choices,
  // so the same frames are dropped in the same places.  (The capacity is a
  // clock reading, but it is passed IN: the reading is taken by the caller,
  // once per attempt, so the function itself stays pure.)
  //
  // The cases differ in what happens when the rings move under us, so the
  // case travels with the choice:
  //
  //   kCaseFree    the snapshot saw a FREE slot somewhere.  A loss may
  //                still be avoidable (the target can lose its FREE slot to
  //                the other Stage 1 thread in between), so the selection is
  //                retried with a fresh snapshot.
  //   kCaseReclaim the snapshot saw NO free slot anywhere.  The oldest READY
  //                frame goes, and that is the designed behaviour.
  //   kCaseDrop    every slot of every ring was IN FLIGHT: the frame itself
  //                is dropped.
  enum { kCaseFree = 1, kCaseReclaim = 2, kCaseDrop = 3 };
  struct TargetChoice {
    NpuService* svc = nullptr;
    int case_id = kCaseDrop;
    // THE INDEX OF THE CHOSEN SERVICE, so the caller can update the delivery
    // tally (served) of exactly that service.
    int svc_index = -1;
  };

  // THE MEASURED CAPACITY OF EVERY SERVICE, in frames per millisecond.
  //
  //   w_i = (frames_run + 1) / (stage2_busy_ms + 1)
  //
  // frames_run is the number of forwards the service really ran and
  // stage2_busy_ms is the wall clock it spent inside rknn_run, so w_i is the
  // service's OWN throughput: how many frames it retires per millisecond of
  // NPU time.  It is the measured version of "this service is half as fast as
  // that one".
  //
  // THE +1s: they keep the estimate finite and non-zero before the service
  // has run anything (both counters are 0 -> w_i = 1), so two fresh services
  // compare EQUAL and the rule degenerates to the 1:1 rotation until the
  // board has said otherwise.  That is the safe direction: the rule starts
  // from the old behaviour and only tilts once it has evidence.
  //
  // THE LOCK: the two counters are written by the service's own Stage 2
  // thread, so the snapshot is taken under m_time -- the same lock that
  // already guards the per-service drop counters.  It is taken ONCE for all
  // the services, so the whole snapshot is consistent, and NO ring lock is
  // held at the same time (the file's lock order {m_feat, m_heat, buf_a->m}
  // -> m_time stays intact: only m_time is taken here).
  void snapshot_capacities(std::vector<double>& w) {
    w.assign(npus.size(), 1.0);
    std::lock_guard<std::mutex> lt(m_time);
    for (size_t i = 0; i < npus.size(); ++i) {
      const NpuService* s = npus[i].get();
      w[i] = (double)(s->frames_run + 1) / (double)(s->stage2_busy_ms + 1);
    }
  }

  TargetChoice pick_target(const std::vector<RingProbe>& probe,
                           const std::vector<double>& w,
                           const std::vector<int64_t>& served) {
    TargetChoice tc;
    const size_t n = probe.size() < npus.size() ? probe.size() : npus.size();
    // PASS 1: a service that can take the frame as it is.  Among those, the
    // one with the highest CAPACITY-WEIGHTED score takes it.
    //
    // WHY THE OLD "SHORTEST READY QUEUE" RULE WAS BIASED: it picked the
    // service whose READY queue was shortest -- that is the service that had
    // just drained a frame and could swallow the next one soonest.  That is
    // a POSITIVE FEEDBACK LOOP, and it is self-reinforcing in the wrong
    // direction:
    //   * the FAST service keeps its READY queue empty, so it keeps winning
    //     the comparison;
    //   * the SLOW service's queue is not long either (it is consuming, just
    //     at a lower rate), so it LOSES the tie, and the tie was broken by
    //     the LOWER service index;
    //   * so the same service kept being fed, and the other one starved.
    // The measured result was a service 0 / service 1 frame split of 1.63:1,
    // and cross-service delivery only narrowed it to 1.52:1 -- the bias was
    // in the RULE, not in the delivery.  THE CONCLUSION STILL STANDS: the
    // rule has to follow the service that can CONSUME the most, not the one
    // that happens to be empty right now.
    //
    // THE ROTATION WAS NOT ENOUGH EITHER: it made the FRAME SPLIT perfect
    // (1.000:1 on CANON29) but left the CAPACITY split uneven (84.47% vs
    // 55.82% of the NPU core), so the slow service still overflowed its ring
    // and the fast one still idled.  Balancing the FRAMES is the wrong
    // invariant -- balancing the CAPACITY is the right one, and that is
    // what the rule below does.
    //
    // THE CAPACITY-WEIGHTED RULE: score_i = w_i / (served_i + 1), the
    // standard "virtual finish time" form of weighted fair queueing.  It is
    // a capacity divided by the load already put on that capacity, so:
    //   * a service that has been fed LESS has the higher score, and the
    //     feed evens out towards the capacity ratio;
    //   * at equal load the service with the HIGHER capacity has the higher
    //     score, so the frames tilt towards the faster service -- which is
    //     the whole point: the fast service is the one that can retire them
    //     without overflowing its ring;
    //   * the two effects balance at served_i proportional to w_i, so every
    //     service's ring is offered exactly what that service can consume.
    // A TIE (equal scores) keeps the LOWER service index, so the choice
    // stays deterministic.
    //
    // EVERY CANDIDATE IS STILL VISITED: a service that has room is NEVER
    // skipped, so the frame is never lost for want of a target -- the same
    // STRICT form of "whoever has room takes it" the rotation gave.
    size_t chosen = n;
    double best_score = 0.0;
    for (size_t i = 0; i < n; ++i) {
      if (!npus[i]->buf_a || probe[i].free_slots == 0) continue;
      const double score = w[i] / (double)(served[i] + 1);
      // STRICTLY greater: a tie keeps the LOWER index (the loop runs upwards).
      if (chosen == n || score > best_score) {
        chosen = i;
        best_score = score;
      }
    }
    if (chosen < n) {
      tc.svc = npus[chosen].get();
      tc.svc_index = (int)chosen;
      tc.case_id = kCaseFree;
      return tc;
    }
    // PASS 2: nobody can take the frame without a loss.  The OLDEST READY
    // frame is reclaimed, and it comes from the service holding the LONGEST
    // READY queue -- that is where the stalest pending frame sits.  A tie
    // keeps the LOWER service index.
    //
    // THE CAPACITY WEIGHT DOES NOT APPLY HERE: this path means the frame is
    // lost anyway, so there is nothing to balance -- and the "oldest first,
    // keep the newest" rule must stay DETERMINISTIC in exactly the way it
    // was.
    size_t best = n;
    for (size_t i = 0; i < n; ++i) {
      if (!npus[i]->buf_a || probe[i].ready == 0) continue;
      if (best == n || probe[i].ready > probe[best].ready) best = i;
    }
    if (best < n) {
      tc.svc = npus[best].get();
      tc.svc_index = (int)best;
      tc.case_id = kCaseReclaim;
      return tc;
    }
    // PASS 3: every slot of every ring is IN FLIGHT.  The frame itself is
    // dropped, and it is charged to the MOST LOADED service (the one with
    // the most slots in flight) so that the per-service counters keep
    // saying something true.  A tie keeps the LOWER service index.
    best = n;
    for (size_t i = 0; i < n; ++i) {
      if (!npus[i]->buf_a) continue;
      if (best == n || probe[i].inflight > probe[best].inflight) best = i;
    }
    if (best >= n) best = 0;  // no service at all: cannot happen (npu_workers >= 1)
    tc.svc = npus[best].get();
    tc.svc_index = (int)best;
    tc.case_id = kCaseDrop;
    return tc;
  }

  // -------------------------------------------------------------------------
  // v10: STAGE 1 -- THE PRE-PROCESS THREAD.
  //
  // It takes a feature, composes the mosaic canvas with the RGA, widens the
  // canvas to fp16 and writes the result STRAIGHT INTO a Buffer A slot.  Only
  // an INTEGER crosses the thread boundary (the slot id), so the hand-off
  // costs nothing.
  //
  // THE TWO RULES THIS THREAD LIVES BY:
  //
  //   1. IT NEVER WAITS FOR THE BUFFER.  If no slot is free, the OLDEST READY
  //      slot is reclaimed -- that frame is lost (bufA_dropped++) -- and if
  //      there is not even a READY slot to reclaim, the frame ITSELF is dropped
  //      (bufA_full_drops++) and the thread moves on to the next frame.  A
  //      wait here would push the back-pressure back into the GMC, and the
  //      GMC is the producer the whole pipeline depends on.
  //   2. IT NEVER WAITS FOR THE GMC EITHER.  It waits for a FEATURE, and that
  //      is a CONSUMER waiting for work: the GMC never waits for this thread.
  //
  // The only thing this thread ever waits for is a feature to arrive.
  // -------------------------------------------------------------------------
  void pre_loop(size_t pre_index, mosaic::Session* psess) {
    // The pre-process stage gets its OWN cores: the whole point of the split
    // is that the upload bucket is no longer inflated by the NPU stage and
    // the GMC fighting for the same CPU and memory bandwidth.
    if (!pre_cpus.empty()) {
      const int cpu = pre_cpus[pre_index % pre_cpus.size()];
      if (mosaic::PinCurrentThreadToCpu(cpu))
        fprintf(stderr, "[i] Stage 1 (pre) service %zu pinned to core %d\n", pre_index, cpu);
    } else {
      fprintf(stderr, "[i] Stage 1 (pre) service %zu is not pinned (no pre_affinity)\n", pre_index);
    }
    // THE SERVICE THIS THREAD USED TO FEED.  It does not feed ONE service
    // any more: the frame goes to whichever service has room (see
    // pick_target above), so the drop counters below belong to the service
    // whose RING the frame is ACTUALLY written into.
    //
    // psess -- this thread's OWN canvas-only session, the one without an
    // RKNN context -- stays exactly as it was: the CANVAS is Stage 1's
    // private work area (compose + widen read from it) and the SLOT is the
    // target service's.  The canvas is never shared with the target's own
    // Stage 1 session, so two threads never compose into the same bytes.
    //
    // THE RETRY BOUND: the rings move under us (Stage 2 frees slots, the
    // other Stage 1 thread takes them), so the selection is retried a
    // BOUNDED number of times.  A loss that only exists because the target
    // lost its FREE slot in between is then avoided, and the frame is
    // dropped only when NO service has room at all.
    const int kTargetAttempts = 4;
    // THE DELIVERY TALLY, AND WHOSE IT IS: it is a STACK LOCAL, so it belongs
    // to THIS Stage 1 thread alone -- one tally per Stage 1 thread, all of
    // them starting at 0, and no thread can see or touch another's.  It is
    // NOT a member and NOT per service: the tally answers "how much have I
    // already handed to this service", which is a property of the FEEDER,
    // not of the receiver.
    //
    // served[i] is the number of frames THIS THREAD has written into service
    // i's Buffer A, and it is the denominator of the capacity-weighted score
    // (see pick_target): score_i = w_i / (served_i + 1).
    //
    // IT IS NOT SHARED ON PURPOSE: the score only needs the LOCAL load of the
    // feeder that is making the choice, and a private tally needs no lock at
    // all.  The two threads' tallies add up to the per-service
    // frames_delivered counter, which is the global view of the same fact.
    std::vector<int64_t> served(npus.size(), 0);
    // THE MEASURED CAPACITY SNAPSHOT, refreshed before every attempt.
    std::vector<double> cap;
    std::vector<RingProbe> probe(npus.size());
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
          // The wait is accounted in its OWN bucket: npu_wait_ms keeps the
          // fused meaning, so the certified arm's numbers do not move.
          std::lock_guard<std::mutex> lt(m_time);
          timing.stage1_wait_ms += waited;
        }
        if (abort) break;
        if (feat_q.empty()) {
          // The GMC is done and its queue is empty: this stage is over.  The
          // engines published every feature before gmc_done was set, so
          // nothing is lost here.
          if (gmc_done) break;
          continue;
        }
        fi = std::move(feat_q.front());
        feat_q.pop_front();
      }

      // ---- take a slot (NEVER WAITS) -------------------------------------
      int slot_id = -1;
      bool drop_frame = false;
      uint8_t* slot_virt = nullptr;
      uint16_t* slot_host = nullptr;
      bool slot_is_dma = false;
      // THE RING THAT TAKES THIS FRAME.  It is NOT necessarily the ring of
      // the service this thread used to be paired with: the frame goes to
      // the service that has room (pick_target above).
      BufferA* tba = nullptr;
      bool settled = false;
      // THE SERVICE THIS FRAME IS GOING TO, AND ITS INDEX: they are set
      // when the take settles, and they are what the delivery bookkeeping
      // below (frames_delivered, served) is charged to.
      NpuService* target_svc = nullptr;
      int target_index = -1;
      for (int attempt = 0; attempt < kTargetAttempts; ++attempt) {
        // ---- PASS 1: the snapshot.  ONE ring lock at a time, released at
        // once: two ring locks are never held together, so the rings cannot
        // deadlock against each other.
        for (size_t i = 0; i < npus.size(); ++i) probe[i] = probe_ring(npus[i]->buf_a.get());
        // THE MEASURED CAPACITY OF EVERY SERVICE, refreshed for this
        // attempt: one consistent snapshot of (frames_run, stage2_busy_ms)
        // under m_time.  It is a clock reading, so it is taken HERE and
        // handed in -- pick_target itself stays a pure function.
        snapshot_capacities(cap);
        // ---- PASS 2: the choice -------------------------------------------
        const TargetChoice tc = pick_target(probe, cap, served);
        BufferA* cba = tc.svc ? tc.svc->buf_a.get() : nullptr;
        if (!cba) {
          // DEFENSIVE AND UNREACHABLE: every service of the decoupled mode
          // owns a Buffer A ring, and pre_loop only runs in that mode.  The
          // frame is dropped (and counted) rather than written nowhere.
          drop_frame = true;
          settled = true;
          std::lock_guard<std::mutex> lt(m_time);
          timing.bufA_full_drops += 1;
          npus[pre_index]->bufA_full_drops += 1;
          break;
        }
        // A loss is only avoidable when the snapshot saw a FREE slot
        // SOMEWHERE: then the target may have lost its slot to the other
        // Stage 1 thread in between, and a fresh snapshot can find the
        // frame another home.  In the "no free slot anywhere" case the loss
        // is the designed behaviour ("drop the oldest READY"), so it is not
        // retried.
        const bool may_retry = (tc.case_id == kCaseFree) && (attempt + 1 < kTargetAttempts);
        // ---- PASS 3: the take.  ONLY the target's ring lock is held here,
        // and the ring is MUTATED only when the outcome is final -- so a
        // retried attempt leaves the ring exactly as it found it.
        {
          std::lock_guard<std::mutex> lk(cba->m);
          int cslot = -1;
          for (size_t i = 0; i < cba->slots.size(); ++i) {
            if (cba->slots[i].state == kSlotFree) {
              cslot = (int)i;
              break;
            }
          }
          if (cslot >= 0) {
            // A FREE slot: take it.  No frame is lost.
            cba->slots[cslot].state = kSlotInflight;
            // The slot's addresses are captured UNDER THE LOCK, so the write
            // below cannot be aimed at a slot that changed hands in between.
            slot_id = cslot;
            slot_virt = cba->slots[cslot].virt;
            slot_host = cba->slots[cslot].host.data();
            slot_is_dma = cba->slots[cslot].mem != nullptr;
            tba = cba;
            target_svc = tc.svc;
            target_index = tc.svc_index;
            settled = true;
          } else if (!may_retry) {
            if (!cba->ready.empty()) {
              // No FREE slot.  Reclaim the OLDEST READY one: Stage 2 did not
              // take it in time, so that frame is lost -- but this stage
              // keeps running.
              slot_id = cba->ready.front();
              cba->ready.pop_front();
              cba->slots[slot_id].state = kSlotInflight;
              slot_virt = cba->slots[slot_id].virt;
              slot_host = cba->slots[slot_id].host.data();
              slot_is_dma = cba->slots[slot_id].mem != nullptr;
              tba = cba;
              // THE FRAME IS GOING TO THIS SERVICE'S RING, so this service is
              // the one the delivery bookkeeping below is charged to.
              target_svc = tc.svc;
              target_index = tc.svc_index;
              std::lock_guard<std::mutex> lt(m_time);
              timing.bufA_dropped += 1;
              // THE PER-SERVICE HALF OF THAT LOSS: it is charged to the
              // service whose ring dropped its oldest READY slot -- the
              // service this frame is ACTUALLY written into.  The two
              // services' numbers must add up to timing.bufA_dropped, and a
              // difference BETWEEN them is the whole point of counting them.
              tc.svc->bufA_reclaimed += 1;
              settled = true;
            } else {
              // Every slot of this ring is in flight: there is nothing to
              // reclaim, so THIS frame is dropped.  The stage does NOT wait
              // -- waiting here is exactly the back-pressure this design
              // forbids.
              drop_frame = true;
              tba = cba;
              std::lock_guard<std::mutex> lt(m_time);
              timing.bufA_full_drops += 1;
              // ... and THIS service could not take the frame at all.
              tc.svc->bufA_full_drops += 1;
              settled = true;
            }
          }
          // else: the target has no FREE slot, but the snapshot said that
          // SOME service had one -- so the target lost its slot to the
          // other Stage 1 thread in between.  The ring was NOT touched; the
          // selection is redone with a fresh snapshot.
        }
        if (settled) break;
      }
      if (drop_frame) continue;
      if (slot_id < 0) {
        // DEFENSIVE, AND UNREACHABLE: the retry is switched off on the last
        // attempt, so the take always settles there.  If it ever did not, the
        // frame is dropped AND counted, so that "the per-service counters
        // add up to the global ones" still holds.
        std::lock_guard<std::mutex> lt(m_time);
        timing.bufA_full_drops += 1;
        npus[pre_index]->bufA_full_drops += 1;
        continue;
      }

      // ---- compose + widen, straight into the slot ------------------------
      const double c0 = mosaic::now_ms();
      if (!mosaic::ComposeFrame(psess, fi.rgb.data(), width, height, &fo)) {
        set_error("ComposeFrame failed (Stage 1)");
        break;
      }
      const double c1 = mosaic::now_ms();
      // The widening's OUTPUT IS the slot's bytes: the NEON kernel writes
      // straight into the DMA memory, so there is no copy in between.  This is
      // the SAME u8_to_f16 call the io-mem path uses, so the bytes are
      // identical and the per-frame md5 is unchanged.
      uint16_t* dst = slot_is_dma ? (uint16_t*)slot_virt : slot_host;
      if (!mosaic::WidenCanvasToF16(psess, dst, psess->canvas.size)) {
        set_error("WidenCanvasToF16 failed (Stage 1)");
        break;
      }
      const double c2 = mosaic::now_ms();

      // ---- publish the slot ------------------------------------------------
      // Into the TARGET service's ring: its Stage 2 thread is the one that
      // waits on that ring's condition variable.
      {
        std::lock_guard<std::mutex> lk(tba->m);
        tba->slots[slot_id].state = kSlotReady;
        // The camera's PHYSICAL frame number travels with the slot, so the
        // result can be named even though the frame crossed two stages.
        tba->slots[slot_id].frame_id = fi.id;
        tba->ready.push_back(slot_id);
      }
      tba->cv.notify_all();
      const double c3 = mosaic::now_ms();
      {
        std::lock_guard<std::mutex> lt(m_time);
        timing.stage1_ms += c3 - c0;
        timing.bufA_upload_ms += c2 - c1;
        // The RGA buckets are the SAME ones the fused mode fills, so the two
        // arms can be compared directly.
        timing.rga_import_ms += fo.rga_import;
        timing.rga_improcess_ms += fo.rga_improcess;
        // THE DELIVERY IS NOW COMPLETE: the frame's bytes are in the target
        // service's ring, so that service has been OFFERED one more frame.
        // It is counted here -- and not when the slot was merely claimed --
        // because this is the moment the frame really landed in the ring.
        // A frame that the ring later reclaims is STILL counted: this counter
        // says how much was offered, frames_run says how much was retired,
        // and the difference between the two is the service's own loss.
        if (target_svc) target_svc->frames_delivered += 1;
      }
      // THE FEEDER'S OWN TALLY: the frame went to this service, so this
      // thread's served count for it goes up by one.  It is the denominator
      // of the capacity-weighted score (pick_target), and it is a stack
      // local, so it needs no lock.
      if (target_svc && target_index >= 0) served[(size_t)target_index] += 1;
    }
    // This stage is over: tell the Stage 2 threads that no new slot can
    // arrive, so they can leave once their ring is empty.
    pre_active.fetch_sub(1, std::memory_order_relaxed);
    for (auto& s : npus) {
      if (s->buf_a) s->buf_a->cv.notify_all();
    }
  }

  // -------------------------------------------------------------------------
  // v10: STAGE 2 -- THE NPU STAGE.
  //
  // It takes a READY slot out of Buffer A, binds it to its own context with
  // rknn_set_io_mem, runs the model and reads the outputs back.  It does NO
  // CPU work of its own: the canvas was composed and widened by Stage 1, and
  // the back-projection belongs to Stage 3.
  //
  // THE ORDER OF THE RKNN CALLS IS NOT NEGOTIABLE:
  //   * rknn_set_io_mem comes BEFORE rknn_run, because the input tensor has
  //     to point at THIS frame's slot when the model runs;
  //   * the slot goes back to FREE only AFTER rknn_outputs_get has copied
  //     the outputs out, because until then the runtime still owns it.
  // -------------------------------------------------------------------------
  void npu_stage_loop(NpuService* svc) {
    // Stage 2 gets its own cores, so the runtime's driver-side work (the
    // copy and the dequant) does not land on the GMC's cores nor on the
    // pre-process stage's.
    if (!npu_stage_cpus.empty()) {
      const int cpu = npu_stage_cpus[(size_t)svc->index % npu_stage_cpus.size()];
      if (mosaic::PinCurrentThreadToCpu(cpu))
        fprintf(stderr, "[i] Stage 2 (NPU) service %d pinned to core %d\n", svc->index, cpu);
    }
    mosaic::Session& sess = *svc->sess;
    BufferA* buf_a = svc->buf_a.get();
    mosaic::FrameOut fo;
    for (;;) {
      int slot_id = -1;
      int64_t frame_id = 0;
      {
        const auto t_wait0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(buf_a->m);
        buf_a->cv.wait(lk, [this, buf_a] {
          return !buf_a->ready.empty() || abort ||
                 (pre_active.load(std::memory_order_relaxed) == 0 && gmc_done);
        });
        const double waited =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wait0)
                .count();
        {
          std::lock_guard<std::mutex> lt(m_time);
          timing.stage2_wait_ms += waited;
        }
        if (abort) break;
        if (buf_a->ready.empty()) {
          // The ring is empty.  If every Stage 1 thread is gone, no slot will
          // ever arrive again and this stage is over; otherwise a slot may
          // still be on its way, so wait for it.
          if (pre_active.load(std::memory_order_relaxed) == 0) break;
          continue;
        }
        slot_id = buf_a->ready.front();
        buf_a->ready.pop_front();
        buf_a->slots[slot_id].state = kSlotInflight;
        frame_id = buf_a->slots[slot_id].frame_id;
      }

      // ---- bind the slot to THIS context, THEN run -------------------------
      const double s0 = mosaic::now_ms();
      bool bound = true;
      if (buf_a->slots[slot_id].mem) {
        // The DMA arm: the slot IS the input tensor's memory.
        if (rknn_set_io_mem(sess.npu.ctx, buf_a->slots[slot_id].mem, &sess.npu.in_attr) != 0) {
          fprintf(stderr, "[FATAL] rknn_set_io_mem failed (Stage 2)\n");
          bound = false;
        }
      } else {
        // The host arm: the same fp16 NHWC bytes, handed over with
        // pass_through=1, so the runtime does no conversion at all.  The
        // declared size is the MODEL's tensor size, not the slot's capacity.
        if (!mosaic::SetF16PassThrough(&sess.npu, buf_a->slots[slot_id].host.data(),
                                       buf_a->tensor_bytes / sizeof(uint16_t))) {
          bound = false;
        }
      }
      if (!bound) {
        {
          std::lock_guard<std::mutex> lk(buf_a->m);
          buf_a->slots[slot_id].state = kSlotFree;
        }
        buf_a->cv.notify_all();
        set_error("binding the Buffer A slot failed (Stage 2)");
        break;
      }

      // ---- stage 2: rknn_run + outputs -----------------------------------
      std::vector<float> hm;
      if (!mosaic::RunFrame(&sess, &hm, &fo)) {
        {
          std::lock_guard<std::mutex> lk(buf_a->m);
          buf_a->slots[slot_id].state = kSlotFree;
        }
        buf_a->cv.notify_all();
        set_error("RunFrame failed (Stage 2)");
        break;
      }
      // THE FRAME WAS REALLY RUN: rknn_run returned 0 and the outputs came
      // back (RunFrame does run + outputs_get + dequant in one go).  This is
      // the service's own tally, and it is what makes the two services
      // comparable -- they run the same model at the same cost, so their
      // tallies SHOULD match.
      const double s1 = mosaic::now_ms();
      {
        // THE PER-SERVICE COUNTERS, AND THEIR LOCK: they are READ by the
        // Stage 1 threads (the capacity estimate w_i), so they are written
        // under m_time -- the same lock that already guards the two drop
        // counters.
        std::lock_guard<std::mutex> lt(m_time);
        ++svc->frames_run;
        // THE SERVICE'S OWN NPU BUSY TIME.  fo.run is the wall clock of the
        // rknn_run call ITSELF: RunFrame takes it with now_ms() around
        // rknn_run (mosaic_core.h), which is exactly the measurement the
        // global timing.run_ms / timing.npu_busy_ms buckets use.  Kept per
        // service, it is the denominator of the capacity estimate w_i, and
        // it is what says "this service is the slow one".
        //
        // The sum is kept in a double (one rknn_run is a FRACTIONAL number of
        // milliseconds) and the int64 counter is its rounded mirror.
        svc->stage2_busy_acc_ms += fo.run;
        svc->stage2_busy_ms = (int64_t)std::llround(svc->stage2_busy_acc_ms);
      }
      // The slot goes back to FREE only NOW: the outputs have been read out,
      // so the runtime is done with it and Stage 1 may write over it.
      {
        std::lock_guard<std::mutex> lk(buf_a->m);
        buf_a->slots[slot_id].state = kSlotFree;
      }
      buf_a->cv.notify_all();
      {
        std::lock_guard<std::mutex> lk(m_time);
        timing.stage2_ms += s1 - s0;
        timing.run_ms += fo.run;
        timing.outputs_ms += fo.outputs;
        // The NPU's own busy time: the numerator of the utilisation.
        timing.npu_busy_ms += fo.run;
        // One sample per forward pass: the distribution of the SINGLE
        // rknn_run, which is what the NPU core mask is judged on.
        rknn_run_samples.push_back(fo.run);
        if (fo.run > timing.rknn_run_max_ms) timing.rknn_run_max_ms = fo.run;
      }

      HeatItem hi;
      // The camera's PHYSICAL frame number, carried through the slot.
      hi.id = frame_id;
      hi.sess = &sess;
      hi.hm = std::move(hm);
      {
        // BUFFER B: the same rule as the fused mode -- the inference does not
        // wait for the back-projection.  A heatmap the post stage cannot take
        // in time is dropped (its detection is lost, the GMC registration is
        // untouched).
        std::lock_guard<std::mutex> lk(m_heat);
        if (heat_q.size() >= heat_depth) {
          heat_q.pop_front();
          std::lock_guard<std::mutex> lt(m_time);
          timing.detections_dropped += 1;
          // The same loss, counted under the name of the buffer that made it.
          timing.bufB_dropped += 1;
        }
        heat_q.push_back(std::move(hi));
      }
      cv_heat.notify_all();
    }
    // THE SERVICE'S OWN TALLY, PRINTED ON EVERY EXIT PATH.  The loop above
    // leaves through a break in four places -- the abort, the normal drain
    // (the ring is empty and no Stage 1 thread is left) and the two error
    // paths -- and they ALL land here, so this is the ONE place that has to
    // be right: the line is printed whether the pipeline ran to the end or
    // was torn down.
    //
    // WHY IT MATTERS: the two services run the same model at the same cost,
    // so their frame counts should match.  If they do not, the Stage 1 ->
    // Stage 2 pairing is feeding them unevenly, and this line says WHICH
    // one is starving and WHICH one is throwing frames away -- the question
    // the board's per-core NPU load raised.
    //
    // v11: THE LINE ALSO CARRIES THE CAPACITY.  frames_delivered is what
    // Stage 1 offered this service and frames_run is what it retired, so
    // the two together say whether the service is the bottleneck (offered
    // more than it ran) or the one that is starving (ran everything it was
    // offered).  stage2_busy_ms is the wall clock the service spent inside
    // rknn_run, and w_i = frames_run / stage2_busy_ms is its measured
    // capacity in frames per millisecond -- the number the capacity-weighted
    // delivery rule (pick_target) divides by.
    //
    // THE FIELD NAMES AND THEIR MEANING ARE FROZEN: the board's analysis
    // scripts parse frames_run, bufA_reclaimed and bufA_full_drops, so
    // those three keep their names and their meaning.  The two new fields
    // are APPENDED, so an old parser still reads the same numbers in the
    // same places.
    {
      // The counters are read under m_time: the two drop counters and
      // frames_delivered are written by this service's Stage 1 thread, which
      // is still running when an abort tears this thread down.
      std::lock_guard<std::mutex> lt(m_time);
      fprintf(stderr,
              "[i] Buffer A service %d: frames_run=%lld frames_delivered=%lld "
              "stage2_busy_ms=%lld bufA_reclaimed=%lld bufA_full_drops=%lld\n",
              svc->index, (long long)svc->frames_run, (long long)svc->frames_delivered,
              (long long)svc->stage2_busy_ms, (long long)svc->bufA_reclaimed,
              (long long)svc->bufA_full_drops);
    }
  }

  // -------------------------------------------------------------------------
  // v10: free a service's Buffer A slots.
  //
  // The slots are destroyed with the context that CREATED them, so this has
  // to run BEFORE ReleaseSession() takes the context down.
  // -------------------------------------------------------------------------
  void free_buffer_a(NpuService* svc) {
    if (!svc || !svc->buf_a) return;
    for (auto& sl : svc->buf_a->slots) {
      if (sl.mem) rknn_destroy_mem(svc->buf_a->owner_ctx, sl.mem);
    }
    svc->buf_a.reset();
  }

  // The same, for EVERY service: the error path of create() tears the
  // pipeline down by hand, and the slots have to go with their contexts.
  void free_all_buffer_a() {
    for (auto& s : npus) free_buffer_a(s.get());
  }

  // -------------------------------------------------------------------------
  // SERVICE NPU: RGA compose + upload + rknn_run + outputs.
  //
  // One independent service per RKNN context.  The board tolerates exactly
  // TWO concurrent contexts (a third trips the 6 s job timeout and soft-resets
  // the NPU), hence npu_workers is clamped to 2.
  // -------------------------------------------------------------------------
  void npu_loop(NpuService* svc) {
    // v10: THE FUSED ARM AND THE DECOUPLED ARM ARE DIFFERENT FUNCTIONS ON
    // PURPOSE.  The fused arm below is the v9 code, UNCHANGED, character for
    // character: it is the certified arm and it has to stay comparable.  The
    // decoupled arm is npu_stage_loop(), which does no CPU work at all.
    if (cfg.decoupled_stages) {
      npu_stage_loop(svc);
      return;
    }
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
    //
    // v10: in the DECOUPLED mode the POST stage belongs to the PRE-PROCESS
    // core group instead.  The NPU stage has a group of its own, and the
    // back-projection is CPU work of the same family as the composition, so
    // the two belong together and away from the GMC.
    const std::vector<int>& post_cpus = cfg.decoupled_stages ? pre_cpus : npu_cpus;
    if (!post_cpus.empty()) {
      const int cpu = post_cpus[post_index % post_cpus.size()];
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
  void publish_artifact(long art_seq, int64_t id, const std::string& md5_hex,
                        const std::vector<Matx23f>& mats) {
    {
      std::lock_guard<std::mutex> lk(m_art);
      OrderedArtifact a;
      a.id = id;
      a.md5_hex = md5_hex;
      a.mats = mats;
      artifacts[art_seq] = std::move(a);
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
  // THE MINIMUM GAP BETWEEN TWO GMC GROUPS, in frames.
  // 1 == no limit: the engine grabs the NEWEST frame as soon as it is free
  // (the adaptive default -- the opportunistic scheduler).  >1 == an
  // artificial rate limit: the engine waits until the head is at least N
  // frames past the one it computed last.  See the header.
  cfg->gmc_min_stride = 1;
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
  // ---- v10: the three-stage decoupled pipeline -----------------------------
  // The DEFAULT is the FUSED worker of v9: it is the certified arm and it has
  // to stay comparable, so the split is strictly opt-in.
  cfg->decoupled_stages = 0;
  cfg->buffer_a_depth = 3;  // 3 slots x 2,457,600 B = ~7 MB of DMA memory
  cfg->buffer_b_depth = 3;  // 3 heatmaps
  cfg->pre_threads = 0;     // 0 = AUTO (= npu_workers); forced to npu_workers
                            // in the decoupled mode anyway (the stages are
                            // paired 1:1)
  cfg->buf_a_mode = 0;      // 0 = DMA (rknn_create_mem), 1 = host memory
  cfg->pre_affinity = nullptr;
  cfg->npu_stage_affinity = nullptr;  // NULL = follow pre_affinity
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
      // The MINIMUM GAP between two GMC groups, in frames.  1 == no limit: the
      // engine grabs the newest frame as soon as it is free (the adaptive
      // default).  >1 == an artificial rate limit.  The ceiling is 64 because
      // beyond that the "gap" is longer than the whole history window and no
      // frame would ever be eligible.
      {"gmc_min_stride", &cfg_s.gmc_min_stride, 1, 64, 1},
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
      // ---- v10: the decoupled pipeline's knobs ------------------------------
      // The fused worker is the DEFAULT: the split is strictly opt-in, because
      // the fused arm is the certified one and the two have to stay
      // comparable.
      {"decoupled_stages", &cfg_s.decoupled_stages, 0, 1, 0},
      // THE TWO BOUNDED BUFFERS.
      //
      // Buffer A's ceiling is 64, NOT 8.  The old argument ("deeper than 8
      // buys nothing, the NPU is 100+ ms per frame, so a deeper ring only
      // delays the loss") was WRONG, and the board measurements say so: with
      // 3 slots the decoupled arm CANON21 dropped 1567 of 7195 GMC groups,
      // and EVERY one of those losses was in Buffer A (bufA_dropped), while
      // the NPU stage was only 72.6% busy and spent 254 s waiting for work.
      // The NPU's own capacity (2 services / 121.9 ms per frame = 16.4 fps)
      // is ABOVE the GMC's supply (15.22 groups/s), so the stage was never
      // the bottleneck -- a 3-slot ring simply cannot hold the GMC's burst:
      // the fit tail (~200 ms, max ~700 ms) is followed by a catch-up burst
      // of 10+ frames, and the ring overflows inside that burst.
      //
      // The cost of the depth is MEMORY, not accuracy.  One slot is
      // 2,457,600 B and every NPU service owns its own ring, so D slots cost
      // D x 2,457,600 B x npu_workers: D=16 is 78.7 MB on the board, D=32 is
      // 157 MB, D=64 is 315 MB.  64 is the ceiling that still fits
      // comfortably in this board's 7.9 GB.
      //
      // Buffer B keeps its 1..8 ceiling: the heatmap ring measured
      // bufB_dropped=0, so it is NOT the bottleneck and widening it would buy
      // nothing but memory.
      {"buffer_a_depth", &cfg_s.buffer_a_depth, 1, 64, 3},
      {"buffer_b_depth", &cfg_s.buffer_b_depth, 1, 8, 3},
      // Where the Buffer A slots live.  0 = DMA (the recommended arm), 1 =
      // host memory (the A/B arm).
      {"buf_a_mode", &cfg_s.buf_a_mode, 0, 1, 0},
  };
  for (Knob& k : knobs) {
    if (*k.v < k.lo || *k.v > k.hi) {
      fprintf(stderr, "[WARN] %s = %d is out of range [%d, %d]; using the reference default %d\n",
              k.name, (int)*k.v, (int)k.lo, (int)k.hi, (int)k.dflt);
      *k.v = k.dflt;
    }
  }
  // -------------------------------------------------------------------------
  // v10: THE PRE-PROCESS THREAD COUNT.
  //
  // It is handled OUTSIDE the clamp table because 0 is a magic value there
  // (0 = AUTO) and because in the decoupled mode the count is FORCED to
  // npu_workers anyway.
  //
  // WHY IT IS FORCED: the stages still run ONE Stage 1 THREAD PER NPU
  // SERVICE.  The frames themselves are NOT pinned any more -- the
  // cross-service delivery hands each frame to the service whose Buffer A
  // ring has room (pick_target in pre_loop) -- but the THREAD COUNT is kept
  // at one per service on purpose: every service keeps a dedicated feeding
  // thread, and the Stage 1 / Stage 2 core placement stays one-to-one, so
  // the change is confined to "who receives the frame" and nothing else
  // moves.
  if (cfg_s.pre_threads < 0 || cfg_s.pre_threads > 4) {
    fprintf(stderr, "[WARN] pre_threads = %d is out of range [0, 4]; using AUTO\n",
            (int)cfg_s.pre_threads);
    cfg_s.pre_threads = 0;
  }
  if (cfg_s.decoupled_stages) {
    if (cfg_s.pre_threads == 0) {
      cfg_s.pre_threads = cfg_s.npu_workers;  // AUTO
    } else if (cfg_s.pre_threads != cfg_s.npu_workers) {
      fprintf(stderr,
              "[WARN] pre_threads=%d was requested, but the decoupled stages run ONE Stage 1 "
              "THREAD PER NPU SERVICE (one dedicated feeding thread per service, and a 1:1 "
              "Stage 1 / Stage 2 core placement): forcing pre_threads=%d\n",
              (int)cfg_s.pre_threads, (int)cfg_s.npu_workers);
      cfg_s.pre_threads = cfg_s.npu_workers;
    }
  } else {
    // The fused worker does the pre-processing itself, so the knob is moot.
    cfg_s.pre_threads = 0;
  }
  if (cfg_s.gmc_gate_min > cfg_s.gmc_gate_max) {
    fprintf(stderr, "[WARN] gmc_gate_min(%d) > gmc_gate_max(%d); using the reference defaults\n",
            (int)cfg_s.gmc_gate_min, (int)cfg_s.gmc_gate_max);
    cfg_s.gmc_gate_min = 500;
    cfg_s.gmc_gate_max = 2000;
  }
  cfg_s.gmc_pyr_cache = cfg_s.gmc_pyr_cache ? 1 : 0;
  if (cfg_s.gmc_prealloc_pad < 0) cfg_s.gmc_prealloc_pad = 0;
  if (cfg_s.input_mode < 0 || cfg_s.input_mode > MOSAIC_INPUT_SET_F16_PT)
    cfg_s.input_mode = MOSAIC_INPUT_SET;
  // v10: in the DECOUPLED mode the input tensor is bound PER FRAME (to a
  // Buffer A slot), so the session must not pre-bind an io-mem buffer, and
  // the upload path is decided by buf_a_mode and not by input_mode.  The
  // knob's value is kept in the config so the caller's request stays visible,
  // but the session is opened in the plain 'set' mode (which allocates no
  // io-mem buffer).
  if (cfg_s.decoupled_stages) {
    fprintf(stderr,
            "[i] decoupled stages: the input_mode knob (%d) is IGNORED -- the upload path is "
            "buf_a_mode=%d (%s)\n",
            (int)cfg_s.input_mode, (int)cfg_s.buf_a_mode,
            cfg_s.buf_a_mode == 0 ? "DMA slot bound with rknn_set_io_mem"
                                  : "host buffer handed over with pass_through=1");
  }
  // -------------------------------------------------------------------------
  // THE OPPORTUNISTIC SCHEDULER NEEDS EXACTLY ONE GMC ENGINE.
  //
  // WHY: the engine grabs the NEWEST seeded frame, and every engine shares
  // the same history ring -- so several engines would all compute THE SAME
  // frame and publish DUPLICATE results, not extra throughput.  And the
  // second engine is not free: two engines measured the GMC fit's p99 at
  // 695 ms against 200 ms with one, because they fight over the same cores.
  //
  // The clamp table above still allows 2 (the hard cap, and the default the
  // table falls back to); the request is forced down to one HERE, and only a
  // caller who actually asked for more is warned about it.
  // -------------------------------------------------------------------------
  if (cfg_s.gmc_workers != 1) {
    fprintf(stderr,
            "[WARN] gmc_workers=%d was requested, but the OPPORTUNISTIC latest-frame scheduler "
            "requires exactly ONE engine: several engines would grab the SAME newest frame "
            "(duplicate results), and two engines measured the GMC fit's p99 at 695 ms against "
            "200 ms with one.  Forcing gmc_workers=1.\n",
            (int)cfg_s.gmc_workers);
    cfg_s.gmc_workers = 1;
  }
  // THE AUTO RULE FOR THE HISTORY RING'S SLACK LIVES HERE AND NOWHERE ELSE.
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
  //
  // v9: the slack is no longer "how far the ingest may run ahead of the
  // engines" (the ingest never waits any more) -- it is one of the two terms
  // of the history ring's depth, the other being the GMC_RING_BUFFER_SIZE
  // floor of 64 slots (= 21 frames of producer headroom over the 43 frames
  // one fit reads).
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

  // The history window: the ring's SLACK, in frames.
  //
  // v9: the ingest NEVER WAITS any more, so the slack is no longer "how far
  // the ingest may run ahead of the engines" -- there is no such budget any
  // more, and the back-pressure that budget used to impose (and that killed
  // the camera queue) is gone.  The slack is now one of the two terms of the
  // history ring's depth:  ring_depth = max(64, max_lag + 1 + ring_slack),
  // i.e. how much room the producer has before it overwrites a frame the
  // running fit still needs.  The value is TAKEN, not derived: the AUTO rule
  // was already resolved in the sanitize stage (one rule, one place), so this
  // cannot drift from what mosaic_pipeline_effective_config() reports.
  p->ring_slack = (long)safe_cfg->gmc_ring_slack;

  // The ENGINE JOB QUEUE's depth.  DEPRECATED (v9): there is no job queue
  // any more -- the engine grabs the newest frame out of the history ring --
  // so this number is not used for anything.  It is KEPT (and still derived
  // from the slack) so the struct layout and the create-time report stay
  // intact.
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
  // ---- v10: the decoupled stages' core lists ------------------------------
  // The pre-process group carries Stage 1 AND Stage 3; the NPU stage gets a
  // group of its own, or follows the pre-process one when it was not given
  // one.  The GMC keeps its own group (--gmc-affinity-group), so the three
  // stages can never land on each other's cores.
  //
  // The FUSED mode does NOT read these lists: it keeps the NPU/POST cores it
  // has always used, so the certified arm stays comparable.  That is also why
  // the list is not even parsed there -- printing "Pre/Post services -> cores
  // 6-7" in the fused mode would be a lie: the POST stage stays on the NPU
  // services' cores.
  if (safe_cfg->decoupled_stages) {
    if (safe_cfg->pre_affinity && *safe_cfg->pre_affinity) {
      if (mosaic::ParseCpuList(safe_cfg->pre_affinity, &p->pre_cpus))
        fprintf(stderr, "[i] Pre/Post services -> cores %s\n", safe_cfg->pre_affinity);
      else
        fprintf(stderr, "[WARN] pre_affinity '%s' parsed to an empty list; ignored\n",
                safe_cfg->pre_affinity);
    }
    if (safe_cfg->npu_stage_affinity && *safe_cfg->npu_stage_affinity) {
      if (mosaic::ParseCpuList(safe_cfg->npu_stage_affinity, &p->npu_stage_cpus))
        fprintf(stderr, "[i] NPU stage services -> cores %s\n", safe_cfg->npu_stage_affinity);
      else
        fprintf(stderr, "[WARN] npu_stage_affinity '%s' parsed to an empty list; ignored\n",
                safe_cfg->npu_stage_affinity);
    } else {
      // NULL == follow the pre-process group.
      p->npu_stage_cpus = p->pre_cpus;
    }
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
    // v10: in the decoupled mode the session is opened WITHOUT an io-mem
    // buffer: the input tensor is bound per frame, to a Buffer A slot.
    const mosaic::InputMode sess_im =
        safe_cfg->decoupled_stages
            ? mosaic::InputMode::kSet
            : static_cast<mosaic::InputMode>(safe_cfg->input_mode);
    if (!mosaic::InitSession(svc->sess.get(), safe_cfg->model_path,
                             static_cast<mosaic::CanvasMode>(safe_cfg->canvas_mode), {}, false,
                             sess_im)) {
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

    // ---- v10: THE DECOUPLED ARM OF THIS SERVICE ---------------------------
    if (safe_cfg->decoupled_stages) {
      // Stage 1's own canvas-only session.  It has NO RKNN context: the board
      // tolerates exactly two and the NPU services already use both.  It
      // composes into its OWN canvas, so the two stages never share a buffer.
      svc->pre_sess.reset(new mosaic::Session());
      if (!mosaic::InitCanvasOnly(svc->pre_sess.get(), safe_cfg->model_path,
                                  static_cast<mosaic::CanvasMode>(safe_cfg->canvas_mode), {},
                                  false)) {
        p->free_all_buffer_a();
        mosaic::ReleaseSession(svc->pre_sess.get());
        mosaic::ReleaseSession(svc->sess.get());
        delete p;
        return fail("InitCanvasOnly failed (Stage 1)");
      }

      // The Buffer A ring.  Its slots are created by THIS service's context,
      // so a slot is only ever bound to the context that owns it.
      svc->buf_a.reset(new MosaicPipeline::BufferA());
      MosaicPipeline::BufferA* ba = svc->buf_a.get();
      ba->owner_ctx = svc->sess->npu.ctx;
      // The slot must hold a WHOLE model input.  The model's own attribute is
      // the authority; the fallback is the 640x640x3 fp16 tensor.
      //
      // TWO SIZES, and they are not the same thing:
      //   tensor_bytes  the input tensor's real size.  It is what the host arm
      //                 has to declare, because pass_through=1 makes the
      //                 runtime read exactly that many bytes.
      //   bytes         the size of the DMA slot.  It is size_with_stride when
      //                 the runtime asks for a strided buffer, because that is
      //                 what the io-mem path allocates too.
      ba->tensor_bytes = svc->sess->npu.in_attr.size;
      if (ba->tensor_bytes == 0)
        ba->tensor_bytes =
            (size_t)(mosaic::kCanvas * mosaic::kCanvas * mosaic::kSrcC * sizeof(uint16_t));
      uint32_t bytes = svc->sess->npu.in_attr.size_with_stride
                           ? svc->sess->npu.in_attr.size_with_stride
                           : svc->sess->npu.in_attr.size;
      if (bytes == 0) bytes = (uint32_t)ba->tensor_bytes;
      ba->elems = bytes / sizeof(uint16_t);
      // The canvas is u8 and the slot is fp16, so the slot needs exactly two
      // bytes per canvas element.  A slot that is too small would silently
      // truncate the input, so it is a FATAL, not a clamp.
      const size_t canvas_elems = svc->pre_sess->canvas.size;
      if (ba->elems < canvas_elems || ba->tensor_bytes < canvas_elems * sizeof(uint16_t)) {
        fprintf(stderr,
                "[FATAL] the Buffer A slot holds %zu fp16 elements (the model's tensor is %zu B) "
                "but the canvas needs %zu\n",
                ba->elems, ba->tensor_bytes, canvas_elems);
        p->free_all_buffer_a();
        p->free_buffer_a(svc.get());
        mosaic::ReleaseSession(svc->pre_sess.get());
        mosaic::ReleaseSession(svc->sess.get());
        delete p;
        return fail("the Buffer A slot is too small for the canvas");
      }
      for (int k = 0; k < safe_cfg->buffer_a_depth; ++k) {
        MosaicPipeline::BufferA::Slot sl;
        sl.bytes = bytes;
        if (safe_cfg->buf_a_mode == 0) {
          // The DMA arm: a pre-allocated, contiguous, DMA-capable buffer that
          // the runtime can read the input tensor from directly.
          sl.mem = rknn_create_mem(ba->owner_ctx, bytes);
          if (!sl.mem) {
            fprintf(stderr, "[FATAL] rknn_create_mem(%u) failed (Buffer A)\n", bytes);
            p->free_all_buffer_a();
            p->free_buffer_a(svc.get());
            mosaic::ReleaseSession(svc->pre_sess.get());
            mosaic::ReleaseSession(svc->sess.get());
            delete p;
            return fail("rknn_create_mem failed (Buffer A)");
          }
          sl.virt = (uint8_t*)sl.mem->virt_addr;
        } else {
          // The host arm: the same fp16 bytes in ordinary memory.  The buffer
          // is exactly the model's tensor, because pass_through=1 makes the
          // runtime read in.size bytes out of it.
          sl.host.assign(ba->tensor_bytes / sizeof(uint16_t), 0);
        }
        ba->slots.push_back(std::move(sl));
      }
      fprintf(stderr,
              "[i] NPU service %d: Buffer A with %d slot(s) of %u B (%zu fp16 elements), mode=%s\n",
              i, (int)safe_cfg->buffer_a_depth, bytes, ba->elems,
              safe_cfg->buf_a_mode == 0 ? "DMA" : "host");
    }
    p->npus.push_back(std::move(svc));
  }

  // ---- the post-processing service ----------------------------------------
  for (int i = 0; i < safe_cfg->post_workers; ++i) {
    p->post_ws.push_back(std::unique_ptr<mosaic::PostWorkspace>(new mosaic::PostWorkspace()));
    mosaic::InitPostWorkspace(p->post_ws.back().get());
  }

  p->timing.wall_ms = mosaic::now_ms();

  // v10: THE DEPTH OF BUFFER B.  The fused mode keeps the queue_depth (8) it
  // has always used, so the certified arm does not move; the decoupled mode
  // uses buffer_b_depth.
  p->heat_depth = safe_cfg->decoupled_stages ? (size_t)safe_cfg->buffer_b_depth
                                             : (size_t)safe_cfg->queue_depth;

  // ---- start the services -------------------------------------------------
  p->ingest_thread = std::thread([p] { p->ingest_loop(); });
  for (auto& e : p->engines) {
    MosaicPipeline::Engine* ep = e.get();
    const size_t ei = (size_t)(&e - &p->engines[0]);
    ep->th = std::thread([p, ep, ei] { p->engine_loop(ep, ei); });
  }
  // v10: the Stage 1 threads start BEFORE the Stage 2 threads, so a slot is
  // never published before its consumer exists.
  if (safe_cfg->decoupled_stages) {
    for (auto& svc : p->npus) {
      MosaicPipeline::NpuService* sp = svc.get();
      p->pre_active.fetch_add(1, std::memory_order_relaxed);
      sp->pre_th = std::thread([p, sp] { p->pre_loop((size_t)sp->index, sp->pre_sess.get()); });
    }
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
  // v10: the decoupled stages' effective configuration, on its OWN line so the
  // line above stays exactly as it was.
  fprintf(stderr,
          "[i] decoupled stages: decoupled=%d buffer_a=%d buffer_b=%d pre_threads=%d "
          "buf_a_mode=%d pre_affinity=%s npu_stage_affinity=%s\n",
          (int)safe_cfg->decoupled_stages, (int)safe_cfg->buffer_a_depth,
          (int)safe_cfg->buffer_b_depth, (int)safe_cfg->pre_threads,
          (int)safe_cfg->buf_a_mode,
          (safe_cfg->pre_affinity && *safe_cfg->pre_affinity) ? safe_cfg->pre_affinity : "(inherit)",
          (safe_cfg->npu_stage_affinity && *safe_cfg->npu_stage_affinity)
              ? safe_cfg->npu_stage_affinity
              : "(follow pre_affinity)");
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
    e->cv_head.notify_all();
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
  // v10: THE STAGE 1 THREADS STOP FIRST, THEN THE NPU STAGE.
  //
  // The order matters: a Stage 1 thread publishes into a Buffer A ring, and
  // the ring's slots are bound to the NPU context.  Stopping the NPU stage
  // first could leave a Stage 1 thread writing into a ring whose consumer is
  // gone -- and, worse, binding a slot to a context that is about to be
  // destroyed.
  if (p->cfg.decoupled_stages) {
    for (auto& svc : p->npus) {
      if (svc->pre_th.joinable()) svc->pre_th.join();
    }
    // Stage 2's exit condition is "every Stage 1 thread is gone AND my ring
    // is empty"; its condition variable has to be woken so it can see that.
    for (auto& svc : p->npus) {
      if (svc->buf_a) svc->buf_a->cv.notify_all();
    }
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
    // The frames the opportunistic engine VOIDED because their history was
    // not whole.  This is the one the operator has to look at: it means the
    // producer outran the history ring (or a camera frame was lost).
    const int64_t voids = p->timing.gmc_void_frames;
    const int64_t seeded = p->gmc_seeded_frames.load(std::memory_order_relaxed);
    int64_t skipped = seeded - p->timing.gmc_groups_done - voids;
    if (skipped < 0) skipped = 0;
    p->timing.gmc_frames_skipped = skipped;
    fprintf(stderr,
            "[i] opportunistic schedule: %lld frames seeded into the history, %lld groups "
            "computed, %lld frames skipped (the engine was busy with a newer frame), %lld frames "
            "VOIDED (their 21-frame history was not whole)\n",
            (long long)seeded, (long long)p->timing.gmc_groups_done, (long long)skipped,
            (long long)voids);
    if (cam || feat || det || blows || voids) {
      fprintf(stderr,
              "[i] fallback summary: camera-queue drops %lld (BREAKS the GMC registration), "
              "features skipped by the inference %lld, detections skipped %lld, "
              "GMC fits degraded by the watchdog %lld, GMC frames voided %lld\n",
              (long long)cam, (long long)feat, (long long)det, (long long)blows,
              (long long)voids);
    } else {
      fprintf(stderr,
              "[i] fallback summary: clean -- no camera-queue drop, no skipped feature, "
              "no skipped detection, no degraded GMC fit, no voided GMC frame\n");
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
  // THE FRAMES THE ENGINE DELIBERATELY DID NOT COMPUTE.
  //
  // It is a DIFFERENCE, not a counter: the frames the ingest seeded into the
  // history, minus the frames that became a group, minus the frames that were
  // voided.  What is left is the frames the engine passed because it was busy
  // with a newer one -- the whole point of the opportunistic scheduler, and
  // NOT a loss (they are still in the history).
  {
    const int64_t seeded = p->gmc_seeded_frames.load(std::memory_order_relaxed);
    int64_t skipped = seeded - out->gmc_groups_done - out->gmc_void_frames;
    if (skipped < 0) skipped = 0;   // the counters are sampled at different instants
    out->gmc_frames_skipped = skipped;
  }
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
  // v10: the Stage 1 threads stop BEFORE the NPU stage (see the same block in
  // flush()), so a Stage 1 thread can never bind a slot to a context that is
  // about to be destroyed.
  for (auto& svc : p->npus) {
    if (svc->pre_th.joinable()) svc->pre_th.join();
  }
  for (auto& svc : p->npus) {
    if (svc->th.joinable()) svc->th.join();
  }
  // The Buffer A slots are destroyed with THEIR OWN context, which is still
  // alive at this point -- rknn_destroy_mem needs the context that created
  // the memory.
  for (auto& svc : p->npus) {
    p->free_buffer_a(svc.get());
  }
  for (auto& svc : p->npus) {
    if (svc->sess) mosaic::ReleaseSession(svc->sess.get());
    if (svc->pre_sess) mosaic::ReleaseSession(svc->pre_sess.get());
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
