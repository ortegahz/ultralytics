// =============================================================================
// e2e_par_demo.cpp -- speed-optimised end-to-end demo: feature extraction and
// NPU inference run OVERLAPPED.
//
//   serial    : stage A for the whole sequence, then stage B   (= run_hm_demo.sh)
//   parallel  : while the NPU chews frame N, the feature pipeline builds N+1
//
// Both modes live in THIS binary and are measured by the SAME code, so the FPS
// comparison cannot be an artefact of two different harnesses.
//
//   producer thread (main) : e2ecore::feature_stream_*  -> GMC fit/warp/median
//   consumer thread        : e2ecore::infer_one         -> RKNN int8
//   between them          : a bounded queue (--depth frames of cache)
//
// The GMC side keeps EVERY knob and every bad-case handle: the fit gate, the
// chain gate, the corner budget and the fit-deadline counter all stay armed, and
// --fit-deadline-ms stays 0 so the output cannot become load-dependent. Only the
// SCHEDULING changes; no feature value does.
//
// The cache is a path queue, not a pixel buffer, on purpose: cv::imwrite stores
// BGR and stb_image reads RGB, so the on-disk round trip is what fixes the byte
// order the NPU sees. Handing the in-memory plane to the NPU instead would swap
// the channels and change every heatmap.
//
// Usage:
//   ./e2e_par_demo --raw-root raw --sequence 34_1 --out-root OUT \
//                  --model M.rknn --mode parallel [--depth 6]
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

#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

#include "e2e_core.h"

namespace {

// Same warmup the profile ledger uses (gmc_stream_ocl --profile-warmup 100).
const long WARMUP_FRAMES = 100;

// ---------------------------------------------------------------------------
// Bounded SPSC queue. depth==0 would mean "unbounded", which is a trap: a fast
// producer would then hold an entire sequence of 512x512x3 BMPs in RAM.
// ---------------------------------------------------------------------------
// One unit of work crossing the cache: a path in file mode, an in-memory RGB
// feature in no-I/O mode.
struct Item {
    std::string path;                 // file mode
    std::vector<uint8_t> rgb;         // no-I/O, still 512x512
    int w = 0, h = 0;
    std::vector<uint8_t> boxed;       // split-lb: already 640x640x3
    std::string name;                 // split-lb: output base name
};

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
        if (q_.empty()) return false;          // closed and drained
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

// Latency samples. The first frames are a cold-start plateau (Mali driver JIT,
// GPU ring/slot fill), so the project convention -- gmc_stream_ocl --profile-warmup
// -- drops them from steady-state statistics. Reporting a max that is really
// just frame 0 makes the 40 ms budget look violated when it is not.
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
//
// Why this exists: with the two threads merely coexisting, the feature
// pipeline's p50 stays at 27 ms but its p95 blows out to 62 ms and its max to
// 76 -- i.e. it blows the 40 ms budget on preemption, not on the algorithm.
// Isolating the stages onto disjoint cores is what turns the speedup into one
// that still honours the budget.
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
    // Line-buffer stdout: this binary is normally run with output redirected to a
    // log on the board, where the default block buffering loses everything printed
    // before an abort() -- which is exactly when the log is needed.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string raw_root, sequence, out_root, model;
    std::string kernel_dir = "/root/kernels", fused = "gpu", mode = "parallel";
    std::string det_out_override, hm_dir_override;
    float th = 0.25f;
    int stride = 2, top_k = 100, anchor_step = 10, limit = 0, dump_heatmap = 1;
    long warmup_frames = WARMUP_FRAMES;
    uint32_t chain_hash = 0;
    e2ecore::InferTiming npu_stages{};
    int have_hash = 0;
    // 8 CPU cores, and they are NOT equal: cpu0-3 are A55 (capacity 405,
    // 1.8 GHz), cpu4-7 are A76 (capacity 1024, 2.352 GHz). The feature stage is
    // the compute-heavy one, so it must get the BIG cores; pinning it to 0-3
    // costs 3.4x (GMC p50 27 ms -> 93 ms). The NPU thread is mostly blocked in
    // rknn_run, so the little cores are plenty for it.
    std::string cpu_feat = "4-5", cpu_lb = "6-7", cpu_npu = "0-3";
    int npu_nice = 5;
    int no_io = 0, want_hash = 0, split_lb = 0, rknn_perf = 0;
    size_t depth = 6;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--raw-root"))      raw_root = need(argc, argv, i, "--raw-root");
        else if (!std::strcmp(a, "--sequence")) sequence = need(argc, argv, i, "--sequence");
        else if (!std::strcmp(a, "--out-root")) out_root = need(argc, argv, i, "--out-root");
        else if (!std::strcmp(a, "--model"))    model = need(argc, argv, i, "--model");
        else if (!std::strcmp(a, "--kernel-dir")) kernel_dir = need(argc, argv, i, "--kernel-dir");
        else if (!std::strcmp(a, "--fused"))    fused = need(argc, argv, i, "--fused");
        else if (!std::strcmp(a, "--mode"))     mode = need(argc, argv, i, "--mode");
        else if (!std::strcmp(a, "--depth"))    depth = (size_t)std::atoi(need(argc, argv, i, "--depth"));
        else if (!std::strcmp(a, "--det-out"))  det_out_override = need(argc, argv, i, "--det-out");
        else if (!std::strcmp(a, "--hm-dir"))   hm_dir_override = need(argc, argv, i, "--hm-dir");
        else if (!std::strcmp(a, "--th"))       th = (float)std::atof(need(argc, argv, i, "--th"));
        else if (!std::strcmp(a, "--stride"))   stride = std::atoi(need(argc, argv, i, "--stride"));
        else if (!std::strcmp(a, "--topk"))     top_k = std::atoi(need(argc, argv, i, "--topk"));
        else if (!std::strcmp(a, "--anchor-step")) anchor_step = std::atoi(need(argc, argv, i, "--anchor-step"));
        else if (!std::strcmp(a, "--limit"))    limit = std::atoi(need(argc, argv, i, "--limit"));
        else if (!std::strcmp(a, "--warmup"))   warmup_frames = std::atol(need(argc, argv, i, "--warmup"));
        else if (!std::strcmp(a, "--cpu-feat")) cpu_feat = need(argc, argv, i, "--cpu-feat");
        else if (!std::strcmp(a, "--cpu-npu"))  cpu_npu = need(argc, argv, i, "--cpu-npu");
        else if (!std::strcmp(a, "--cpu-lb"))   cpu_lb = need(argc, argv, i, "--cpu-lb");
        else if (!std::strcmp(a, "--npu-nice")) npu_nice = std::atoi(need(argc, argv, i, "--npu-nice"));
        else if (!std::strcmp(a, "--no-io"))  no_io = 1;
        else if (!std::strcmp(a, "--hash"))   want_hash = 1;
        else if (!std::strcmp(a, "--split-lb")) split_lb = 1;
        else if (!std::strcmp(a, "--rknn-perf")) rknn_perf = 1;
        else if (!std::strcmp(a, "--no-heatmap")) dump_heatmap = 0;
        else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
            std::printf("Usage: %s --raw-root DIR --sequence NAME --out-root DIR --model PATH\n"
                        "            [--mode serial|parallel] [--depth N] [--th F] [--limit N]\n",
                        argv[0]);
            return 0;
        } else { std::fprintf(stderr, "[FATAL] unknown arg: %s\n", a); return 2; }
    }
    if (raw_root.empty() || sequence.empty() || out_root.empty() || model.empty()) {
        std::fprintf(stderr, "[FATAL] --raw-root/--sequence/--out-root/--model required\n");
        return 2;
    }
    if (mode != "serial" && mode != "parallel") {
        std::fprintf(stderr, "[FATAL] --mode must be serial or parallel\n");
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

    if (no_io) {
        // Must be decided BEFORE the InferParams are assembled. Applying it
        // afterwards left ip.dump_dir pointing at the heatmap directory, so
        // "no I/O" still wrote 1.2 MB of tensors per frame -- visible as
        // NPU write ~25 ms/frame in a run that claimed to write nothing.
        dump_heatmap = 0;
    }

    e2ecore::InferParams ip;
    ip.model_path = model;
    ip.th = th;
    ip.stride = stride;
    ip.top_k = top_k;
    ip.detections_out = no_io ? std::string()
                              : (det_out_override.empty() ? out_root + "/npu_det.tsv"
                                                         : det_out_override);
    ip.tag = sequence;
    ip.rknn_perf = rknn_perf != 0;
    if (dump_heatmap) {
        ip.dump_dir = hm_dir_override.empty() ? out_root + "/hm" : hm_dir_override;
        ip.input_dump_dir = out_root + "/npu_input";
    }

    std::printf("=====================================================================\n");
    std::printf(" RK3588 e2e demo  mode=%s  io=%s  hash=%s  stages=%s  sequence=%s  cache_depth=%zu\n",
                mode.c_str(), no_io ? "NONE" : "bmp+heatmap", want_hash ? "on" : "off",
                split_lb ? "gmc|lb|npu" : "gmc|npu", sequence.c_str(), depth);
    std::printf(" th=%.4f stride=%d topk=%d anchor_step=%d fused=%s\n", th, stride, top_k,
                anchor_step, fused.c_str());
    std::printf("=====================================================================\n");

    Acc gmc_pipe, gmc_wall, npu, lb_stage;
    long lb_frames = 0;
    std::atomic<long> gmc_frames{0}, npu_frames{0}, npu_dets{0};
    bool failed = false;

    FrameQueue queue(depth);
    FrameQueue queue2(split_lb ? depth : 1);

    // ---- shared producer body ----------------------------------------------
    // `keep` non-null => serial mode (collect paths, run no inference here).
    // `q`    non-null => parallel mode (push into the bounded cache).
    auto produce = [&](e2ecore::FeatureStream* s, std::vector<Item>* keep,
                       FrameQueue* q) {
        std::string path;
        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        e2ecore::FrameTiming ft;
        for (;;) {
            const double w0 = now_s();
            const bool got = no_io ? e2ecore::feature_stream_next_rgb(s, &rgb, &w, &h, &ft)
                                   : e2ecore::feature_stream_next(s, &path, &ft);
            const double w1 = now_s();
            if (!got) break;
            gmc_pipe.add(ft.total_ms);
            gmc_wall.add((w1 - w0) * 1000.0);
            ++gmc_frames;
            if (keep) {
                Item ki;
                if (no_io) { ki.rgb = rgb; ki.w = w; ki.h = h; }
                else       { ki.path = path; }
                keep->push_back(std::move(ki));
                continue;
            }
            Item it;
            if (no_io) { it.rgb = std::move(rgb); it.w = w; it.h = h; }
            else       { it.path = path; }
            if (!q->push(std::move(it))) break;
        }
    };

    if (mode == "parallel") {
        // Producer is the main thread; pin it before any feature work starts.
        if (bind_thread(cpu_feat.c_str()))
            std::printf("[AFF] feature thread -> cpu %s\n", cpu_feat.c_str());
    }

    const double t_start = now_s();
    if (mode == "serial") {
        e2ecore::FeatureStream* s = e2ecore::feature_stream_open(fp);
        if (!s) { std::fprintf(stderr, "[FATAL] feature stream open failed\n"); return 1; }
        std::vector<Item> all;
        produce(s, &all, nullptr);
        long fg = 0, ch = 0, fd = 0, cc = 0;
        e2ecore::feature_stream_handles(s, &fg, &ch, &fd, &cc);
        std::printf("[SEQ] %-24s frames=%-5ld md5=%s | fitgate=%ld chain=%ld fitdeadline=%ld "
                    "cornercap=%ld\n", sequence.c_str(), (long)all.size(),
                    e2ecore::feature_stream_md5(s).c_str(), fg, ch, fd, cc);
        e2ecore::feature_stream_close(s);

        e2ecore::InferEngine* e = e2ecore::infer_open(ip);
        if (!e) { std::fprintf(stderr, "[FATAL] infer engine open failed\n"); return 1; }
        if (want_hash) e2ecore::infer_enable_hash(e);
        for (const Item& it2 : all) {
            e2ecore::InferTiming it;
            const bool ok = no_io ? e2ecore::infer_one_rgb(e, it2.rgb.data(), it2.w, it2.h,
                                                            nullptr, &it)
                                  : e2ecore::infer_one(e, it2.path, nullptr, &it);
            if (!ok) { failed = true; break; }
            npu.add(it.total_ms);
            ++npu_frames;
        }
        npu_dets = e2ecore::infer_detections(e);
        std::printf("[DONE] frames=%ld detections=%ld\n", e2ecore::infer_frames_done(e),
                    e2ecore::infer_detections(e));
        e2ecore::infer_stage_totals(e, &npu_stages);
        if (want_hash) { chain_hash = e2ecore::infer_hash(e); have_hash = 1; }
        e2ecore::infer_close(e);
    } else {
        e2ecore::FeatureStream* s = e2ecore::feature_stream_open(fp);
        if (!s) { std::fprintf(stderr, "[FATAL] feature stream open failed\n"); return 1; }
        // The NPU engine is opened BEFORE the pipeline starts so its one-time
        // rknn_init cost lands inside the measured window in both modes.
        e2ecore::InferEngine* e = e2ecore::infer_open(ip);
        if (!e) { std::fprintf(stderr, "[FATAL] infer engine open failed\n"); return 1; }

        if (want_hash) e2ecore::infer_enable_hash(e);

        // Middle stage: letterbox on its own thread, so the accelerator stops
        // sharing a core with a float bilinear resize.
        std::thread lbthread;
        if (split_lb) {
            lbthread = std::thread([&] {
                if (bind_thread(cpu_lb.c_str()))
                    std::printf("[AFF] letterbox thr  -> cpu %s\n", cpu_lb.c_str());
                Item it;
                while (queue.pop(&it)) {
                    double ms = 0;
                    std::vector<uint8_t> boxed;
                    e2ecore::letterbox_feature(it.rgb.data(), it.w, it.h, &boxed, &ms);
                    lb_stage.add(ms);
                    ++lb_frames;
                    char nm[64];
                    std::snprintf(nm, sizeof(nm), "frame_%06ld", lb_frames - 1);
                    it.boxed = std::move(boxed);
                    it.name = nm;
                    if (!queue2.push(std::move(it))) break;
                }
                queue2.close();
            });
        }

        std::thread consumer([&] {
            if (bind_thread(cpu_npu.c_str()))
                std::printf("[AFF] npu thread      -> cpu %s\n", cpu_npu.c_str());
            // Address THIS thread by tid, not by "who=0". On Linux a nice value is
            // per-thread, but setpriority(PRIO_PROCESS, 0, n) renices the WHOLE
            // process -- which quietly reniced the feature thread too and tripled its
            // latency (GMC p50 27 ms -> 91 ms) while looking like a correct fix.
            if (npu_nice != 0) {
                const pid_t tid = (pid_t)syscall(SYS_gettid);
                if (setpriority(PRIO_PROCESS, tid, npu_nice) != 0)
                    std::fprintf(stderr, "[WARN] setpriority(tid=%d) failed\n", (int)tid);
            }
            Item it2;
            const bool three_stage = split_lb;
            FrameQueue* in = three_stage ? &queue2 : &queue;
            while (in->pop(&it2)) {
                e2ecore::InferTiming it;
                const bool ok =
                    three_stage ? e2ecore::infer_preboxed(e, it2.boxed, it2.name.c_str(),
                                                         nullptr, &it)
                    : no_io      ? e2ecore::infer_one_rgb(e, it2.rgb.data(), it2.w, it2.h,
                                                          nullptr, &it)
                                : e2ecore::infer_one(e, it2.path, nullptr, &it);
                if (!ok) { failed = true; break; }
                npu.add(it.total_ms);
                ++npu_frames;
            }
            npu_dets = e2ecore::infer_detections(e);
            e2ecore::infer_stage_totals(e, &npu_stages);
            if (want_hash) { chain_hash = e2ecore::infer_hash(e); have_hash = 1; }
            e2ecore::infer_close(e);
        });

        produce(s, nullptr, &queue);
        // Order matters: downstream threads may still be reading off the queues,
        // so they must be closed and joined BEFORE the feature stream (and the
        // Mali/OpenCL context behind it) is torn down.
        queue.close();
        if (split_lb) lbthread.join();
        consumer.join();

        long fg = 0, ch = 0, fd = 0, cc = 0;
        e2ecore::feature_stream_handles(s, &fg, &ch, &fd, &cc);
        std::printf("[SEQ] %-24s frames=%-5ld md5=%s | fitgate=%ld chain=%ld fitdeadline=%ld "
                    "cornercap=%ld\n", sequence.c_str(), gmc_frames.load(),
                    e2ecore::feature_stream_md5(s).c_str(), fg, ch, fd, cc);
        e2ecore::feature_stream_close(s);
        std::printf("[DONE] frames=%ld detections=%ld\n", npu_frames.load(), npu_dets.load());
    }
    const double t_end = now_s();

    if (failed) {
        std::fprintf(stderr, "[FATAL] run aborted\n");
        return 1;
    }

    const double wall = t_end - t_start;
    const long n = npu_frames.load();
    const long warmup = std::min<long>(warmup_frames, gmc_frames.load() / 2);

    const Acc::S gp = gmc_pipe.stats(warmup);
    const Acc::S gw = gmc_wall.stats(warmup);
    const Acc::S np = npu.stats(warmup);

    std::printf("\n=====================================================================\n");
    std::printf(" PERF  mode=%s  frames=%ld  wall=%.2f s  (warmup dropped: %ld)\n",
                mode.c_str(), n, wall, warmup);
    std::printf("   FPS          : %.2f fps\n", n / wall);
    std::printf("   --- steady state (post-warmup) ---\n");
    std::printf("   GMC pipeline : avg %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f ms   "
                "(fit+warp+median, the 40 ms item)\n", gp.avg, gp.p50, gp.p95, gp.max);
    std::printf("   GMC wall     : avg %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f ms   "
                "(incl. BMP read+write)\n", gw.avg, gw.p50, gw.p95, gw.max);
    std::printf("   NPU wall     : avg %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f ms\n",
                np.avg, np.p50, np.p95, np.max);
    std::printf("   --- budget ---\n");
    std::printf("   GMC <=40 ms  : %s  (avg %.2f, p95 %.2f, max %.2f)\n",
                (gp.p95 <= 40.0 && gp.max <= 40.0) ? "PASS" : "VIOLATED",
                gp.avg, gp.p95, gp.max);
    std::printf("   serial equiv : avg %.2f ms/frame (gmc_wall + npu, no overlap)\n",
                gw.avg + np.avg);
    std::printf("   overlap gain : %.3fx\n", (gw.avg + np.avg) / (wall * 1000.0 / (double)n));
    if (split_lb) {
        const Acc::S ls = lb_stage.stats(warmup);
        std::printf("   LB stage     : avg %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f ms   "
                    "(own thread)\n", ls.avg, ls.p50, ls.p95, ls.max);
    }
    std::printf("   NPU breakdown: load %6.2f  letterbox %6.2f  rknn %6.2f  dequant %6.2f  "
                "peaks %6.2f  write %6.2f  (ms/frame)\n",
                npu_stages.load_ms, npu_stages.letterbox_ms, npu_stages.rknn_ms,
                npu_stages.dequant_ms, npu_stages.peaks_ms, npu_stages.write_ms);
    std::printf("   rknn split   : inputs_set %6.2f (%.0f%%)  rknn_run %6.2f (%.0f%%)  "
                "outputs_get %6.2f (%.0f%%)  [ms/frame, %% of rknn]\n",
                npu_stages.inputs_set_ms, 100.0 * npu_stages.inputs_set_ms / npu_stages.rknn_ms,
                npu_stages.run_ms, 100.0 * npu_stages.run_ms / npu_stages.rknn_ms,
                npu_stages.outputs_get_ms, 100.0 * npu_stages.outputs_get_ms / npu_stages.rknn_ms);
    if (rknn_perf) {
        std::printf("   rknn sdk perf: run_duration(hw, driver) %6.2f ms/frame   "
                    "[DIAGNOSTIC: COLLECT_PERF_MASK inflates the run, do not use as headline]\n",
                    npu_stages.run_hw_ms);
    }
    if (have_hash) std::printf("   heatmap chain hash: %08x\n", chain_hash);
    std::printf("=====================================================================\n");
    return 0;
}
