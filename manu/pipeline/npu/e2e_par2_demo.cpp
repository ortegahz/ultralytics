// =============================================================================
// e2e_par2_demo.cpp -- two-NPU-worker end-to-end demo (fp16-capable)
//
// WHY THIS FILE EXISTS
// --------------------
// e2e_par_demo.cpp runs ONE RKNN context, so its floor is a single rknn_run.
// For the fp16 model that floor is the whole problem: measured on this board,
//   int8 : inputs_set   0.18 ms + rknn_run  55.3 ms  -> NPU stage ~59 ms
//   fp16 : inputs_set  30.87 ms + rknn_run 104.4 ms  -> NPU stage ~138 ms
// i.e. one fp16 context caps the pipeline at ~7 fps no matter how well the
// host side is overlapped.
//
// The board tolerates TWO concurrent RKNN contexts (a THIRD one trips the
// RKNPU 6 s job timeout and soft-resets the whole NPU -- measured, see
// rules.md and memory). So the only way past the fp16 floor is to give the
// pipeline two contexts and let them run against each other.
//
// LAYOUT (4 threads, 3 stages)
//
//   GMC (main)  --e2ecore::feature_stream_next_rgb-->  queue1
//   letterbox   --e2ecore::letterbox_feature------>  queue2
//   NPU worker 0 and 1 --e2ecore::infer_preboxed--->   RKNN context 0 / 1
//
// The letterbox split is inherited from e2e_par_demo --split-lb, which showed
// letterbox costs 12.23 ms on its own thread vs 28.60 ms when it shares the NPU
// thread (the difference is preemption, not arithmetic).
//
// BYTE-ORDER AND EQUIVALENCE ARE STRUCTURAL, NOT ASSERTS
// -----------------------------------------------------
// Every stage calls the same production code e2e_par_demo.cpp calls:
//   feature_stream_next_rgb -> letterbox_feature -> infer_preboxed.
// The only thing that changes is WHICH THREAD runs them and how many RKNN
// contexts exist. No feature value and no tensor is recomputed differently.
//
// ACCURACY VERIFICATION
// ---------------------
// With two consumers pulling from one queue, the frame -> worker assignment is
// timing dependent, so a rolling chain hash over "frames in order" is NOT
// reproducible and must not be used to claim bit-exactness. Instead every frame
// carries a GLOBAL index assigned by the producer (frame_%06ld), each worker
// writes its own detections TSV, and the merge is done by sorting on that
// index. Two runs can then be compared frame-by-frame by NAME, which is
// order-independent. Per-worker chain hashes are still printed, but only as a
// secondary signal.
//
// Usage:
//   ./e2e_par2_demo --raw-root raw --sequence 34_1 --out-root OUT \
//                   --model M.rknn --depth 6 \
//                   [--cpu-feat 4-5] [--cpu-lb 6-7] [--cpu-npu0 0-1] [--cpu-npu1 2-3]
// =============================================================================

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "e2e_core.h"

namespace {

// Same warmup the profile ledger uses (gmc_stream_ocl --profile-warmup 100).
// The first frames are a cold-start plateau (Mali driver JIT, GPU ring fill,
// and -- for fp16 -- the first rknn_run with weights paging in), so reporting a
// max that is really just frame 0 would make the budget look violated when it
// is not.
const long WARMUP_FRAMES = 100;

struct Item {
    std::vector<uint8_t> rgb;    // GMC output, 512x512x3
    int w = 0, h = 0;
    std::vector<uint8_t> boxed;  // letterbox output, 640x640x3
    std::vector<uint16_t> boxed_f16;  // --f16-in: same values, pre-widened (REJECTED
                                      // by the runtime, kept as a negative control)
    e2ecore::F32Boxed boxed_f32;      // --f32-in: the upload the runtime accepts
    std::string name;            // GLOBAL producer-assigned index, see header
};

// Bounded queue. Single producer (GMC / letterbox), N consumers (the NPU
// workers). The original e2e_par_demo comment says SPSC, but the
// mutex+condvar implementation below is already MPMC-safe; only the wakeup
// policy matters and notify_one() is correct for both.
class FrameQueue {
public:
    explicit FrameQueue(size_t cap) : cap_(cap ? cap : 1) {}

    bool push(Item v) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return false;
        q_.push_back(std::move(v));
        not_empty_.notify_one();
        return true;
    }

    bool pop(Item* out) {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        *out = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable not_empty_, not_full_;
    std::deque<Item> q_;
    size_t cap_;
    bool closed_ = false;
};

struct Acc {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    long n() const { return (long)v.size(); }

    struct S {
        long n = 0;
        double avg = 0, p50 = 0, p95 = 0, max = 0;
    };
    S stats(long skip) const {
        S s;
        if ((long)v.size() <= skip) return s;
        std::vector<double> w(v.begin() + skip, v.end());
        std::sort(w.begin(), w.end());
        s.n = (long)w.size();
        double sum = 0;
        for (double x : w) sum += x;
        s.avg = sum / (double)w.size();
        s.p50 = w[w.size() / 2];
        s.p95 = w[(size_t)((double)(w.size() - 1) * 0.95)];
        s.max = w.back();
        return s;
    }
};

double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Pin the calling thread to a CPU list ("0-3", "4-7", or empty = leave alone).
// Same rationale as e2e_par_demo.cpp: without disjoint cores the feature
// pipeline's p95 blows out from 27 ms to 62 ms purely by preemption.
bool bind_thread(const char* cpulist) {
    if (!cpulist || !*cpulist) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    const char* p = cpulist;
    while (*p) {
        if (std::isdigit((unsigned char)*p)) {
            char* e;
            const long a = std::strtol(p, &e, 10);
            long b = a;
            if (*e == '-') {
                const char* q = e + 1;
                b = std::strtol(q, &e, 10);
            }
            for (long c = a; c <= b; c++) CPU_SET((int)c, &set);
            p = e;
            while (*p && !std::isdigit((unsigned char)*p)) ++p;
        } else {
            ++p;
        }
    }
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        std::fprintf(stderr, "[WARN] sched_setaffinity(%s) failed\n", cpulist);
        return false;
    }
    return true;
}

const char* need(int argc, char** argv, int& i, const char* name) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "[FATAL] %s needs a value\n", name);
        std::exit(2);
    }
    return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
    // Line-buffer stdout: this binary normally runs with output redirected to a
    // log on the board, where block buffering loses everything printed before an
    // abort() -- which is exactly when the log is needed.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string raw_root, sequence, out_root, model;
    std::string kernel_dir = "/root/kernels", fused = "gpu";
    std::string det_out_override;
    float th = 0.25f;
    int stride = 2, top_k = 100, anchor_step = 10, limit = 0;
    int npu_workers = 2;
    int no_det = 0;
    int want_hash = 1;
    int f16_in = 0;
    int f16_div255 = 0;
    int f32_in = 0;
    int f32_div255 = 0;
    int f32_passthrough = 1;
    int npu_core0 = 0;   // 0 = runtime default (RKNN_NPU_CORE_AUTO); 1/2/4 = NPU core 0/1/2
    int npu_core1 = 0;
    long warmup_frames = WARMUP_FRAMES;
    size_t depth = 6;
    std::string cpu_feat = "4-5", cpu_lb = "6-7";
    std::string cpu_npu0 = "0-1", cpu_npu1 = "2-3";
    int npu_nice = 0;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--raw-root"))      raw_root = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--sequence")) sequence = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--out-root")) out_root = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--model"))    model = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--kernel-dir")) kernel_dir = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--fused"))    fused = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--th"))       th = (float)std::atof(need(argc, argv, i, a));
        else if (!std::strcmp(a, "--stride"))   stride = std::atoi(need(argc, argv, i, a));
        else if (!std::strcmp(a, "--topk"))     top_k = std::atoi(need(argc, argv, i, a));
        else if (!std::strcmp(a, "--anchor-step")) anchor_step = std::atoi(need(argc, argv, i, a));
        else if (!std::strcmp(a, "--limit"))    limit = std::atoi(need(argc, argv, i, a));
        else if (!std::strcmp(a, "--depth"))    depth = (size_t)std::atol(need(argc, argv, i, a));
        else if (!std::strcmp(a, "--workers"))  npu_workers = std::atoi(need(argc, argv, i, a));
        else if (!std::strcmp(a, "--det-out"))  det_out_override = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--cpu-feat")) cpu_feat = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--cpu-lb"))   cpu_lb = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--cpu-npu0")) cpu_npu0 = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--cpu-npu1")) cpu_npu1 = need(argc, argv, i, a);
        else if (!std::strcmp(a, "--npu-nice")) npu_nice = std::atoi(need(argc, argv, i, a));
        else if (!std::strcmp(a, "--warmup"))   warmup_frames = std::atol(need(argc, argv, i, a));
        else if (!std::strcmp(a, "--no-det"))   no_det = 1;
        else if (!std::strcmp(a, "--no-hash"))  want_hash = 0;
        else if (!std::strcmp(a, "--f16-in"))   f16_in = 1;
        else if (!std::strcmp(a, "--f16-div255")) f16_div255 = 1;
        else if (!std::strcmp(a, "--f32-in"))   f32_in = 1;
        else if (!std::strcmp(a, "--f32-div255")) f32_div255 = 1;
        else if (!std::strcmp(a, "--f32-no-passthrough")) f32_passthrough = 0;
        else if (!std::strcmp(a, "--npu-core0")) npu_core0 = (int)std::strtol(need(argc, argv, i, a), nullptr, 0);
        else if (!std::strcmp(a, "--npu-core1")) npu_core1 = (int)std::strtol(need(argc, argv, i, a), nullptr, 0);
        else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
            std::printf("Usage: %s --raw-root DIR --sequence NAME --out-root DIR --model PATH\n"
                        "            [--workers N] [--depth N] [--th F] [--no-det] [--no-hash]\n"
                        "            [--cpu-feat LIST] [--cpu-lb LIST] [--cpu-npu0 LIST] [--cpu-npu1 LIST]\n"
                        "            [--f16-in] [--f16-div255]   (REJECTED by runtime: needs 4 B/px)\n"
            "            [--f32-in] [--f32-div255]   (float32 upload, pass_through=1)\n"
            "            [--f32-no-passthrough]       (same buffer, type=FLOAT32 instead)\n"
            "            [--npu-core0 MASK] [--npu-core1 MASK]   1/2/4 = NPU core 0/1/2, 0=auto\n",
                        argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[FATAL] unknown arg: %s\n", a);
            return 2;
        }
    }
    if (raw_root.empty() || sequence.empty() || out_root.empty() || model.empty()) {
        std::fprintf(stderr, "[FATAL] --raw-root/--sequence/--out-root/--model required\n");
        return 2;
    }
    if (npu_workers < 1 || npu_workers > 2) {
        // Measured: a THIRD concurrent RKNN context trips the RKNPU 6 s job
        // timeout and soft-resets the whole NPU, taking the in-flight contexts
        // with it. Refusing >2 here is cheaper than discovering it on the board.
        std::fprintf(stderr, "[FATAL] --workers must be 1 or 2 (3+ contexts soft-reset the NPU)\n");
        return 2;
    }

    e2ecore::FeatureParams fp;
    fp.raw_root = raw_root;
    fp.sequence = sequence;
    fp.out_dir = out_root + "/feat";
    fp.out_ext = "bmp";
    fp.kernel_dir = kernel_dir;
    fp.fused = fused;
    fp.anchor_step = anchor_step;
    fp.limit = limit;

    // Per-worker InferParams. Each RKNN context gets its OWN detections file:
    // two contexts cannot share one stream, and keeping them separate is what
    // lets the frames be merged back in producer order afterwards.
    std::vector<e2ecore::InferParams> ips(npu_workers);
    for (int k = 0; k < npu_workers; ++k) {
        ips[k].model_path = model;
        ips[k].th = th;
        ips[k].stride = stride;
        ips[k].top_k = top_k;
        ips[k].tag = sequence;
        if (!no_det) {
            char p[512];
            if (det_out_override.empty())
                std::snprintf(p, sizeof(p), "%s/npu_det.w%d.tsv", out_root.c_str(), k);
            else
                std::snprintf(p, sizeof(p), "%s.w%d", det_out_override.c_str(), k);
            ips[k].detections_out = p;
        }
    }

    std::printf("=====================================================================\n");
    std::printf(" RK3588 e2e demo 2xNPU  stages=gmc|lb|npu0|npu1  sequence=%s  "
                "workers=%d  cache_depth=%zu\n", sequence.c_str(), npu_workers, depth);
    std::printf(" th=%.4f stride=%d topk=%d anchor_step=%d fused=%s det=%s hash=%s\n",
                th, stride, top_k, anchor_step, fused.c_str(), no_det ? "OFF" : "ON",
                want_hash ? "on" : "off");
    std::printf("=====================================================================\n");

    Acc gmc_pipe, gmc_wall, lb_stage;
    std::vector<Acc> npu(npu_workers);
    std::vector<e2ecore::InferTiming> npu_stages(npu_workers);
    std::vector<long> npu_frames(npu_workers, 0), npu_dets(npu_workers, 0);
    std::vector<uint32_t> chain_hash(npu_workers, 0);
    std::vector<int> have_hash(npu_workers, 0);
    std::atomic<long> gmc_frames{0};
    std::atomic<bool> failed{false};

    FrameQueue queue(depth);   // GMC  -> letterbox
    FrameQueue queue2(depth);  // letterbox -> NPU workers

    // GMC runs on the main thread. The frame index is assigned HERE, by the
    // single producer, so names are globally unique and ordered no matter which
    // worker later picks the frame up.
    if (bind_thread(cpu_feat.c_str()))
        std::printf("[AFF] feature thread -> cpu %s\n", cpu_feat.c_str());

    const double t_start = now_s();

    e2ecore::FeatureStream* s = e2ecore::feature_stream_open(fp);
    if (!s) { std::fprintf(stderr, "[FATAL] feature stream open failed\n"); return 1; }

    // Both RKNN contexts are opened BEFORE any frame is produced so the two
    // one-time rknn_init costs land inside the measured window in both the
    // 1-worker and 2-worker arms -- otherwise the 2-worker arm pays two inits
    // and looks slower for a reason that has nothing to do with parallelism.
    std::vector<e2ecore::InferEngine*> engines(npu_workers, nullptr);
    for (int k = 0; k < npu_workers; ++k) {
        engines[k] = e2ecore::infer_open(ips[k]);
        if (!engines[k]) {
            std::fprintf(stderr, "[FATAL] infer engine %d open failed\n", k);
            return 1;
        }
        // One NPU core per worker, set before the first rknn_run. Default 0 leaves
        // the runtime on RKNN_NPU_CORE_AUTO, which is a different claim from "this
        // worker owns this core" -- so the split is opt-in and its result is
        // reported rather than assumed. A refused mask aborts the run instead of
        // quietly running the arm under a different topology than its label.
        const int cm = (k == 0) ? npu_core0 : npu_core1;
        if (cm) {
            std::printf("[2] worker %d -> NPU core mask 0x%x\n", k, cm);
            if (e2ecore::infer_set_npu_core(engines[k], cm) != 0) {
                std::fprintf(stderr, "[FATAL] worker %d core mask 0x%x rejected by runtime\n", k, cm);
                return 1;
            }
        }
        if (want_hash) e2ecore::infer_enable_hash(engines[k]);
    }

    // ---- letterbox stage ----------------------------------------------------
    std::thread lbthread([&] {
        if (bind_thread(cpu_lb.c_str()))
            std::printf("[AFF] letterbox thr  -> cpu %s\n", cpu_lb.c_str());
        Item it;
        while (queue.pop(&it)) {
            double ms = 0;
            if (f32_in) {
                // The widening + std_values normalisation that otherwise costs
                // 45.4 ms INSIDE rknn_inputs_set happens here instead, on a
                // thread that has budget to spare. NEON, 16 px per iteration.
                e2ecore::letterbox_feature_f32(it.rgb.data(), it.w, it.h, &it.boxed_f32,
                                              &ms, f32_div255);
            } else if (f16_in) {
                // Negative control. The runtime rejects a 2-byte-per-element
                // upload outright:
                //   param input size(2457600) < model input size(4915200)
                e2ecore::letterbox_feature_f16(it.rgb.data(), it.w, it.h, &it.boxed_f16,
                                              &ms, f16_div255);
            } else {
                std::vector<uint8_t> boxed;
                e2ecore::letterbox_feature(it.rgb.data(), it.w, it.h, &boxed, &ms);
                it.boxed = std::move(boxed);
            }
            lb_stage.add(ms);
            if (!queue2.push(std::move(it))) break;
        }
        queue2.close();
    });

    // ---- NPU workers --------------------------------------------------------
    std::vector<std::thread> workers;
    for (int k = 0; k < npu_workers; ++k) {
        workers.emplace_back([&, k] {
            const char* cpus = (k == 0) ? cpu_npu0.c_str() : cpu_npu1.c_str();
            if (bind_thread(cpus))
                std::printf("[AFF] npu worker %d  -> cpu %s\n", k, cpus);
            // Address THIS thread by tid, not by "who=0": setpriority(PRIO_PROCESS,
            // 0, n) renices the WHOLE process, which would renice the GMC and
            // letterbox threads too (that mistake tripled GMC p50 once already).
            if (npu_nice != 0) {
                const pid_t tid = (pid_t)syscall(SYS_gettid);
                if (setpriority(PRIO_PROCESS, tid, npu_nice) != 0)
                    std::fprintf(stderr, "[WARN] setpriority(tid=%d) failed\n", (int)tid);
            }
            e2ecore::InferEngine* e = engines[k];
            Acc& mine = npu[k];
            long frames = 0, dets = 0;
            Item it;
            while (queue2.pop(&it)) {
                e2ecore::InferTiming t;
                const bool ok = f32_in
                    ? e2ecore::infer_preboxed_f32(e, it.boxed_f32, it.name.c_str(), nullptr, &t,
                                                 f32_passthrough != 0)
                    : f16_in
                    ? e2ecore::infer_preboxed_f16(e, it.boxed_f16, it.name.c_str(), nullptr, &t)
                    : e2ecore::infer_preboxed(e, it.boxed, it.name.c_str(), nullptr, &t);
                if (!ok) { failed = true; break; }
                mine.add(t.total_ms);
                ++frames;
            }
            dets = e2ecore::infer_detections(e);
            npu_frames[k] = frames;
            npu_dets[k] = dets;
            e2ecore::infer_stage_totals(e, &npu_stages[k]);
            if (want_hash) { chain_hash[k] = e2ecore::infer_hash(e); have_hash[k] = 1; }
            e2ecore::infer_close(e);
        });
    }

    // ---- GMC producer (this thread) ----------------------------------------
    {
        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        e2ecore::FrameTiming ft;
        for (;;) {
            const double w0 = now_s();
            const bool got = e2ecore::feature_stream_next_rgb(s, &rgb, &w, &h, &ft);
            const double w1 = now_s();
            if (!got) break;
            gmc_pipe.add(ft.total_ms);
            gmc_wall.add((w1 - w0) * 1000.0);
            const long idx = gmc_frames.fetch_add(1);
            Item it;
            it.rgb = std::move(rgb);
            it.w = w;
            it.h = h;
            char nm[64];
            std::snprintf(nm, sizeof(nm), "frame_%06ld", idx);
            it.name = nm;
            if (!queue.push(std::move(it))) break;
        }
    }

    // Order matters: downstream threads may still be reading off the queues, so
    // they must be closed and joined BEFORE the feature stream (and the Mali /
    // OpenCL context behind it) is torn down.
    queue.close();
    lbthread.join();
    for (auto& t : workers) t.join();

    long fg = 0, ch = 0, fd = 0, cc = 0;
    e2ecore::feature_stream_handles(s, &fg, &ch, &fd, &cc);
    std::printf("[SEQ] %-24s frames=%-5ld md5=%s | fitgate=%ld chain=%ld fitdeadline=%ld "
                "cornercap=%ld\n", sequence.c_str(), gmc_frames.load(),
                e2ecore::feature_stream_md5(s).c_str(), fg, ch, fd, cc);
    e2ecore::feature_stream_close(s);

    long tot_frames = 0, tot_dets = 0;
    for (int k = 0; k < npu_workers; ++k) { tot_frames += npu_frames[k]; tot_dets += npu_dets[k]; }
    std::printf("[DONE] frames=%ld detections=%ld (workers: ", tot_frames, tot_dets);
    for (int k = 0; k < npu_workers; ++k)
        std::printf("w%d=%ld/%ld ", k, npu_frames[k], npu_dets[k]);
    std::printf(")\n");

    const double t_end = now_s();

    if (failed) {
        std::fprintf(stderr, "[FATAL] run aborted\n");
        return 1;
    }

    const double wall = t_end - t_start;
    const long n = tot_frames;
    // A run that produced nothing must never be reported as a speed result:
    // that is exactly the "RC=0 but frames=0" false-success shape (rules.md 9g).
    if (n == 0 || wall <= 0.0) {
        std::fprintf(stderr, "[FATAL] produced %ld frames in %.2f s -- nothing to report\n", n, wall);
        return 3;
    }
    if (gmc_frames.load() != n) {
        std::fprintf(stderr, "[FATAL] GMC produced %ld frames but the NPU workers completed %ld "
                             "-- detections and features are not the same set\n",
                     gmc_frames.load(), n);
        return 3;
    }

    const long warmup = std::min<long>(warmup_frames, gmc_frames.load() / 2);
    const Acc::S gp = gmc_pipe.stats(warmup);
    const Acc::S gw = gmc_wall.stats(warmup);
    const Acc::S ls = lb_stage.stats(warmup);

    // Worker mean of the per-worker averages: the NPU-side cost is per-context,
    // so averaging the two contexts then dividing by 2 gives the per-frame NPU
    // budget the pipeline actually has to hide.
    double np_avg = 0, np_p50 = 0, np_p95 = 0, np_max = 0;
    for (int k = 0; k < npu_workers; ++k) {
        const Acc::S ns = npu[k].stats(warmup);
        np_avg += ns.avg; np_p50 += ns.p50; np_p95 += ns.p95; np_max += ns.max;
    }
    np_avg /= npu_workers; np_p50 /= npu_workers; np_p95 /= npu_workers; np_max /= npu_workers;
    // infer_stage_totals() ALREADY divides by the engine's own frame count, so
    // these are per-frame averages per context. Dividing by npu_frames again
    // (which is what this did first) reported 1/20th of the truth -- e.g. a
    // 110 ms rknn_run printed as 5.53 ms. Do NOT normalise a second time.
    double is_avg = 0, run_avg = 0, og_avg = 0, dq_avg = 0, pk_avg = 0, wr_avg = 0, lb_in_npu = 0;
    for (int k = 0; k < npu_workers; ++k) {
        is_avg += npu_stages[k].inputs_set_ms;
        run_avg += npu_stages[k].run_ms;
        og_avg += npu_stages[k].outputs_get_ms;
        dq_avg += npu_stages[k].dequant_ms;
        pk_avg += npu_stages[k].peaks_ms;
        wr_avg += npu_stages[k].write_ms;
        lb_in_npu += npu_stages[k].letterbox_ms;
    }
    is_avg /= npu_workers; run_avg /= npu_workers; og_avg /= npu_workers;
    dq_avg /= npu_workers; pk_avg /= npu_workers; wr_avg /= npu_workers; lb_in_npu /= npu_workers;
    const double rknn_avg = is_avg + run_avg + og_avg;

    std::printf("\n=====================================================================\n");
    std::printf(" PERF  2xNPU  frames=%ld  wall=%.2f s  (warmup dropped: %ld)\n", n, wall, warmup);
    std::printf("   FPS          : %.2f fps\n", n / wall);
    std::printf("   --- steady state (post-warmup) ---\n");
    std::printf("   GMC pipeline : avg %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f ms   "
                "(fit+warp+median, the 40 ms item)\n", gp.avg, gp.p50, gp.p95, gp.max);
    std::printf("   GMC wall     : avg %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f ms\n",
                gw.avg, gw.p50, gw.p95, gw.max);
    std::printf("   LB stage     : avg %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f ms   (own thread)\n",
                ls.avg, ls.p50, ls.p95, ls.max);
    std::printf("   NPU wall/ctx : avg %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f ms   (mean over %d "
                "contexts)\n", np_avg, np_p50, np_p95, np_max, npu_workers);
    std::printf("   --- budget ---\n");
    std::printf("   GMC <=40 ms  : %s  (avg %.2f, p95 %.2f, max %.2f)\n",
                (gp.p95 <= 40.0 && gp.max <= 40.0) ? "PASS" : "VIOLATED",
                gp.avg, gp.p95, gp.max);
    std::printf("   per-frame wall: %.2f ms   (front-end must fit in %.2f ms to keep %d contexts fed)\n",
                wall * 1000.0 / (double)n, wall * 1000.0 / (double)n / (double)npu_workers, npu_workers);
    std::printf("   NPU ctx cost : avg %6.2f ms/frame/context   (%d contexts -> %6.2f ms/frame budget)\n",
                np_avg, npu_workers, np_avg / (double)npu_workers);
    std::printf("   NPU breakdown/frame: letterbox %6.2f  inputs_set %6.2f  rknn_run %6.2f  "
                "outputs_get %6.2f  dequant %6.2f  peaks %6.2f  write %6.2f  (ms)\n",
                lb_in_npu, is_avg, run_avg, og_avg, dq_avg, pk_avg, wr_avg);
    std::printf("   rknn split   : inputs_set %6.2f (%.0f%%)  rknn_run %6.2f (%.0f%%)  "
                "outputs_get %6.2f (%.0f%%)   [%% of rknn]\n",
                is_avg, rknn_avg > 0 ? 100.0 * is_avg / rknn_avg : 0.0,
                run_avg, rknn_avg > 0 ? 100.0 * run_avg / rknn_avg : 0.0,
                og_avg,  rknn_avg > 0 ? 100.0 * og_avg / rknn_avg : 0.0);
    for (int k = 0; k < npu_workers; ++k) {
        const Acc::S ns = npu[k].stats(warmup);
        std::printf("   worker %d     : frames=%-5ld dets=%-5ld  npu avg %6.2f  p50 %6.2f  "
                    "p95 %6.2f  max %6.2f ms", k, npu_frames[k], npu_dets[k],
                    ns.avg, ns.p50, ns.p95, ns.max);
        if (have_hash[k]) std::printf("  hash %08x", chain_hash[k]);
        std::printf("\n");
    }
    if (npu_workers == 2 && have_hash[0] && have_hash[1] && chain_hash[0] == chain_hash[1]) {
        std::printf("   NOTE: both workers hash identically. With dynamic queue assignment that is\n"
                    "   EVIDENCE OF NOTHING -- use the per-frame TSV merge to compare runs.\n");
    }
    std::printf("=====================================================================\n");
    return 0;
}
