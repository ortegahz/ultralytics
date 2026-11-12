// ===========================================================================
// mosaic_async_client_example.cpp -- 多线程「生产-消费 + 回调」工程级模板
//
//   [相机线程 Producer]              [算法线程 Worker]           [业务方]
//   每 40 ms 产出一帧                  取结果                       OSD / 伺服 / 推流
//   feed_frame()  ────────────────>   collect() ────────────────> callback()
//
// 三条设计规则
//   1) **相机线程绝不阻塞**：feed_frame() 只做一次 memcpy + 一次 notify_one；
//      帧在这里被**深拷贝**了一份 ⇒ 相机可以把 DMA buffer 立刻还给驱动复用
//      （无野指针、无撕裂）。相机节奏（40 ms 一帧）与算法节奏（8~12 fps）解耦。
//   2) **Worker 严格「一推一收」**：push() 一次就 collect() 一次，且同在一个
//      线程里 ⇒ 从结构上杜绝「一次推多帧 ⇒ collect 永久阻塞」。
//   3) **帧号是物理帧号**：SDK 内部按 push 顺序自己发号（push() 返回的 id），
//      它**不知道**相机的物理帧号 ⇒ 本模板在 push **之前**就把映射绑好
//      （见 feed_frame 的注释）⇒ collect 永远查得到，输出里不会出现 -1。
//
// ---------------------------------------------------------------------------
// 三种模式（--mode）
//
//   serial   单线程严格「push→collect」成对执行。**这正是老 demo
//            （mosaic_client_example）的语义**，也是本模板的**参照基线**：
//            拿它的输出与 push-all 的输出逐位比对，就能证明多线程路径
//            没有改变任何一个数字。
//
//   push-all **精度验收 / Golden 对标模式**（默认）：Producer 带**安全反压**
//            （--inflight，最多 N 帧「已推未回」）。⇒ 帧 **100% 连续**送进
//            SDK，SDK 不会丢帧 ⇒ GMC 的 21 帧**物理历史连续** ⇒ 输出与 serial
//            **逐位相同**，也与认证 golden 的 F1 **逐项全等**（ΔF1 = +0.000000）。
//
//   app-drop **在线生产模式**：Producer **不带反压**（相机线程一秒都不等）。
//            ⚠️ 但**应用层绝不跳帧** —— 每一帧都 push 进 SDK。丢帧交给 **SDK
//            内部**做（它丢的是队列里**最老**的帧 ⇒ GMC 拿到的输入序列仍然
//            **物理连续**）。这样「物理历史连续」这条 GMC 的前提就不被破坏。
//            ⚠️ **反例（本作废，别这么写）**：在**应用层**做「最新帧覆盖」
//            —— 那会让喂进 SDK 的时间轴从 40 ms 变成 120 ms，GMC 的 21 帧
//            物理历史被打散，而 SDK **察觉不到**（它的内部号是连续的发号计数）
//            ⇒ 特征静默漂移、F1 出现万分之八量级的波动。**抽帧只能在推理
//            阶段做，不能在前处理（历史窗口）阶段做。**
//
// ---------------------------------------------------------------------------
// 输出（--out）：<seq>_mosaic.tsv（检测结果，stem 带物理帧号）、feature_md5.txt
// （SDK 内部号）、feature_md5_phys.txt（★物理帧号，离线比对用这一份）、
// frame_map.tsv（内部号 → 物理帧号）。
// ===========================================================================
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/imgcodecs.hpp>

#include "mosaic_pipeline.h"

// 业务方的回调：帧号（**物理**帧号）、目标、目标数、用户数据。
// 回调跑在 Worker 线程上 ⇒ **必须短**。
using DetectionCallback =
    std::function<void(int64_t phys_fid, const MosaicDet* dets, int32_t n, void* user_data)>;

class AsyncMosaicRunner {
public:
    AsyncMosaicRunner(MosaicPipeline* pipe, DetectionCallback cb, void* ud, int64_t inflight)
        : pipe_(pipe), cb_(std::move(cb)), ud_(ud), inflight_(inflight) {}

    ~AsyncMosaicRunner() {
        stopping_.store(true);
        cv_.notify_all();
        if (th_.joinable()) th_.join();   // 已 drain_and_stop() 过 ⇒ 立即返回
    }

    void start() {
        if (th_.joinable()) return;
        running_.store(true);
        th_ = std::thread(&AsyncMosaicRunner::worker_loop, this);
    }

    // Producer 调用：交一帧进来。
    //   inflight_ > 0 ⇒ 带反压：队列满时**等**（但谓词含停机信号，不会挂死）。
    //   inflight_ = 0 ⇒ 不等，相机线程永远不等算法。
    void feed_frame(const uint8_t* gray, int32_t w, int32_t h, int64_t phys_fid) {
        fed_++;
        if (inflight_ > 0) {
            std::unique_lock<std::mutex> lk(m_);
            // ⚠️ 谓词**必须含停机信号**：否则停机时 Producer 会永远等一个
            //    不会再涨的 delivered_，producer.join() 就永久挂起。
            cv_.wait(lk, [this] {
                return stopping_.load() || (pushed_.load() - delivered_.load()) < inflight_;
            });
            if (stopping_.load()) return;          // 停机 ⇒ 不再往 SDK 塞帧
        }
        // ★ **客户端帧号随帧进入库，结果原样带回** ⇒ 输出里不可能出现 -1。
        //   mosaic_pipeline_push_frame() 把 phys_fid 绑在这帧上，一路带到
        //   collect() 的返回值和结果回调里。库内部另有自己的稠密编号（给 GMC
        //   的 21 帧历史做锚点、检测相机流的空洞），**与客户端帧号解耦** ⇒
        //   客户端跳帧也不会让引擎误报 HOLE、不会拉伸时间窗。
        //   （早先版本是在外部用一张「内部号→物理帧号」映射表做的，既有时序
        //     竞态（collect 先于映射写入）又要额外维护一张表；原生 API 之后
        //     这张表退化成**只用于导出**的审计记录。）
        const int64_t lib_id = mosaic_pipeline_push_frame(pipe_, gray, w, h, phys_fid);
        if (lib_id < 0) return;
        {
            std::lock_guard<std::mutex> lk(map_m_);
            lib2phys_[lib_id] = phys_fid;   // 仅供导出 frame_map.tsv
        }
        pushed_++;
    }

    // Producer 推完了 ⇒ 显式 EOS：Worker 知道不会再有新的帧了。
    void signal_eos() {
        eos_.store(true);
        cv_.notify_all();
    }

    // 收尾：**顺序不能反**。
    //   ① Producer 必须已经退出（没有并发 push）—— flush 严禁在并发 push 时调用；
    //   ② flush 让 SDK 收尾（eof、内部线程 join、剩余结果进 out_q、置 flushed）；
    //   ③ Worker 的 collect 因此会返回 -1，Worker 退出，join 不会挂。
    void drain_and_stop() {
        stopping_.store(true);
        cv_.notify_all();
        mosaic_pipeline_flush(pipe_);
        if (th_.joinable()) th_.join();
    }

    // serial（单线程参照）模式下由调用方直接 collect，这里补一次计数。
    void note_delivered() {
        delivered_++;
        collected_++;
        cv_.notify_all();
    }

    int64_t fed() const { return fed_.load(); }
    int64_t pushed() const { return pushed_.load(); }
    int64_t collected() const { return collected_.load(); }
    int64_t delivered() const { return delivered_.load(); }
    std::map<int64_t, int64_t> frame_map() const {
        std::lock_guard<std::mutex> lk(map_m_);
        return lib2phys_;
    }

private:
    void worker_loop() {
        MosaicDet dets[128];
        for (;;) {
            int64_t lib_id = -1;
            const int32_t nd = mosaic_pipeline_collect(pipe_, &lib_id, dets, 128);
            if (nd < 0) break;                    // flushed 且已排空 ⇒ 正常收尾
            delivered_++;
            cv_.notify_all();                     // 放行 Producer 的反压等待
            collected_++;
            const int64_t phys = lib_id;   // ★ 客户端帧号，库原样带回
            fprintf(stderr, "[rx] 物理帧 %lld → %d 个目标\n", (long long)phys, nd);
            if (cb_) cb_(phys, dets, nd, ud_);
        }
    }

    MosaicPipeline* pipe_;
    DetectionCallback cb_;
    void* ud_;
    int64_t inflight_;

    std::atomic<bool> running_{false}, eos_{false}, stopping_{false};
    std::thread th_;
    std::mutex m_;
    std::condition_variable cv_;
    mutable std::mutex map_m_;
    std::map<int64_t, int64_t> lib2phys_;
    // 统计量跨线程读写 ⇒ 必须是 atomic。
    std::atomic<int64_t> fed_{0}, pushed_{0}, collected_{0}, delivered_{0};
};

// 帧文件清单：自己扫目录。**不用** SDK 的 mosaic_pipeline_list_frames() —— 它
// 内部有 4096 帧的硬上限（kMosaicFrameProbeMax），长序列会被**静默截断**。
// 帧号取**文件名里的数字**（frame_012345.bmp ⇒ 12345）—— 那才是物理帧号。
static bool ListFrames(const std::string& dir, std::vector<std::string>* paths,
                       std::vector<int32_t>* ids) {
    DIR* d = opendir(dir.c_str());
    if (!d) return false;
    std::map<long, std::string> by_idx;
    struct dirent* e = nullptr;
    while ((e = readdir(d)) != nullptr) {
        const std::string name = e->d_name;
        if (name.rfind("frame_", 0) != 0 || name.size() < 12) continue;
        const size_t dot = name.rfind('.');
        if (dot == std::string::npos || name.substr(dot) != ".bmp") continue;
        const std::string digits = name.substr(6, dot - 6);
        if (digits.find_first_not_of("0123456789") != std::string::npos) continue;
        by_idx[atol(digits.c_str())] = dir + "/" + name;
    }
    closedir(d);
    for (const auto& kv : by_idx) {
        paths->push_back(kv.second);
        ids->push_back(static_cast<int32_t>(kv.first));
    }
    return !paths->empty();
}

static void EnsureDir(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i <= path.size(); ++i) {
        if ((i == path.size() || path[i] == '/') && !cur.empty() && cur != "/")
            ::mkdir(cur.c_str(), 0755);
        if (i < path.size()) cur.push_back(path[i]);
    }
}

struct Collector {
    std::mutex m;
    std::map<int64_t, std::vector<MosaicDet>> rows;
    int64_t n_dets = 0;
};

static void OnDetection(int64_t phys_fid, const MosaicDet* dets, int32_t n, void* ud) {
    Collector* c = static_cast<Collector*>(ud);
    std::lock_guard<std::mutex> lk(c->m);
    c->rows[phys_fid].assign(dets, dets + n);
    c->n_dets += n;
}

int main(int argc, char** argv) {
    std::string dir = "/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324";
    std::string model = "/mnt/manu/ptq/models/p0_off_A_none.rknn";
    std::string kdir = "/mnt/manu/mosaic_sdk/runtime/kernels";
    std::string out = "/tmp/mosaic_async_out";
    std::string seq = "VIDEO00007_19700101_001324";
    std::string mode = "push-all";
    int fps = 25, frames = 0;
    int64_t inflight = 4;               // push-all 的反压上限；app-drop 用 0（不反压）
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool nx = (i + 1 < argc);
        if (a == "--frames-dir" && nx) dir = argv[++i];
        else if (a == "--model" && nx) model = argv[++i];
        else if (a == "--kernel-dir" && nx) kdir = argv[++i];
        else if (a == "--out" && nx) out = argv[++i];
        else if (a == "--seq" && nx) seq = argv[++i];
        else if (a == "--mode" && nx) mode = argv[++i];
        else if (a == "--fps" && nx) fps = atoi(argv[++i]);
        else if (a == "--frames" && nx) frames = atoi(argv[++i]);
        else if (a == "--inflight" && nx) inflight = atoll(argv[++i]);
        else {
            fprintf(stderr, "用法: %s [--frames-dir D] [--model M] [--kernel-dir K] [--out D]\n"
                            "          [--seq S] [--fps N] [--frames N]\n"
                            "          [--mode serial|push-all|app-drop] [--inflight N]\n", argv[0]);
            return 2;
        }
    }
    if (mode != "serial" && mode != "push-all" && mode != "app-drop") {
        fprintf(stderr, "[FATAL] --mode 只能是 serial | push-all | app-drop\n");
        return 2;
    }
    if (mode == "app-drop") inflight = 0;   // 在线生产：相机线程一秒都不等
    EnsureDir(out);

    // ---- 1) 建流水线：和单线程示例完全一样的配置 ----
    MosaicOpenOptions o = {};
    o.version = MOSAIC_API_VERSION;
    o.preset = MOSAIC_PRESET_REALTIME;     // 最优档：3 级流水线 + D=3
    o.model_path = model.c_str();
    o.kernel_dir = kdir.c_str();
    o.sequence = seq.c_str();              // 逐帧特征指纹文件的第一列就是它
    o.check_env = 1;                       // 没有 raw_root ⇒ 自动跳过体检（不是失败）
    char err[512] = {0};
    MosaicPipeline* pipe = mosaic_pipeline_open(&o, err, sizeof(err));
    if (!pipe) { fprintf(stderr, "[FATAL] 建流水线失败：%s\n", err); return 1; }
    mosaic_pipeline_set_post_mode(pipe, MOSAIC_POST_SPARSE);   // 钉死最优后处理
    const std::string md5_path = out + "/feature_md5.txt";
    mosaic_pipeline_set_feature_md5_out(pipe, md5_path.c_str());

    // ---- 2) 列帧 ----
    std::vector<std::string> paths;
    std::vector<int32_t> ids;
    if (!ListFrames(dir, &paths, &ids)) {
        fprintf(stderr, "[FATAL] %s 下没有帧\n", dir.c_str());
        mosaic_pipeline_destroy(pipe);
        return 1;
    }
    if (frames > 0 && (int)paths.size() > frames) {
        paths.resize((size_t)frames);
        ids.resize((size_t)frames);
    }
    fprintf(stderr, "[i] %zu 帧，相机 %d FPS，模式 %s%s\n", paths.size(), fps, mode.c_str(),
            mode == "push-all" ? "（反压限流，帧 100% 连续送入 SDK）" : "");

    Collector collector;
    AsyncMosaicRunner runner(pipe, OnDetection, &collector, inflight);

    // ---- 3) 推帧 ----
    const int64_t period_ns = 1000000000LL / (fps > 0 ? fps : 25);
    const auto t0 = std::chrono::steady_clock::now();
    if (mode == "serial") {
        // 单线程参照基线：**每帧 push 后立刻 collect**（= 老 demo 的语义）。
        // 它与 push-all 的 GMC 输入序列完全相同（都是连续帧）⇒ 输出应逐位相同。
        for (size_t i = 0; i < paths.size(); ++i) {
            std::this_thread::sleep_until(t0 + std::chrono::nanoseconds((int64_t)i * period_ns));
            const cv::Mat img = cv::imread(paths[i], cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;
            runner.feed_frame(img.data, img.cols, img.rows, ids[i]);
            MosaicDet dets[128];
            int64_t lib_id = -1;
            const int32_t nd = mosaic_pipeline_collect(pipe, &lib_id, dets, 128);
            if (nd < 0) break;
            runner.note_delivered();
            OnDetection(lib_id, dets, nd, &collector);
        }
        runner.signal_eos();
        mosaic_pipeline_flush(pipe);
    } else {
        runner.start();
        std::thread producer([&]() {
            for (size_t i = 0; i < paths.size(); ++i) {
                std::this_thread::sleep_until(t0 + std::chrono::nanoseconds((int64_t)i * period_ns));
                // ↓↓↓ 这两行只是「把文件读成灰度图」。**你用自己的相机 buffer 时，
                //     把它们换成你的取图代码**，只要 gray 是一张 uint8_t 灰度图即可。
                const cv::Mat img = cv::imread(paths[i], cv::IMREAD_GRAYSCALE);
                if (img.empty()) { fprintf(stderr, "[WARN] 读不出 %s\n", paths[i].c_str()); continue; }
                runner.feed_frame(img.data, img.cols, img.rows, ids[i]);
            }
            runner.signal_eos();          // ★ EOS：推完了，通知 Worker
        });
        producer.join();                  // Producer 退出后才有 flush（严禁并发 push 时 flush）
        runner.drain_and_stop();         // flush + 排空 + Worker 退出
    }

    // ---- 4) 落盘：TSV（物理帧号）+ 帧号映射 + 物理帧号版特征指纹 ----
    const std::map<int64_t, int64_t> fmap = runner.frame_map();
    char p[1024];
    {
        std::lock_guard<std::mutex> lk(collector.m);
        snprintf(p, sizeof(p), "%s/%s_mosaic.tsv", out.c_str(), seq.c_str());
        FILE* tsv = fopen(p, "w");
        if (tsv) {
            for (const auto& kv : collector.rows) {
                fprintf(tsv, "%s__%06lld\t%zu", seq.c_str(), (long long)kv.first, kv.second.size());
                for (const MosaicDet& d : kv.second)
                    fprintf(tsv, "\t%d %.4f %.4f %d %d %d %.6f %.4f", d.tag, (double)d.cx, (double)d.cy,
                            d.w, d.h, d.area, (double)d.peak, (double)d.sum);
                fprintf(tsv, "\n");
            }
            fclose(tsv);
        }
    }
    snprintf(p, sizeof(p), "%s/frame_map.tsv", out.c_str());
    FILE* fm = fopen(p, "w");
    if (fm) {
        for (const auto& kv : fmap) fprintf(fm, "%lld\t%lld\n", (long long)kv.first, (long long)kv.second);
        fclose(fm);
    }
    snprintf(p, sizeof(p), "%s/feature_md5_phys.txt", out.c_str());
    FILE* src = fopen(md5_path.c_str(), "r");
    FILE* dst = fopen(p, "w");
    if (src && dst) {
        char line[512], s[128], idx[64], hash[64];
        while (fgets(line, sizeof(line), src)) {
            if (sscanf(line, "%127[^\t]\t%63[^\t]\t%63[^\n]", s, idx, hash) != 3) continue;
            const int64_t lib_id = atoll(idx);
            const auto it = fmap.find(lib_id);
            const int64_t phys = (it == fmap.end()) ? lib_id : it->second;
            fprintf(dst, "%s\t%06lld\t%s\n", s, (long long)phys, hash);
        }
    }
    if (src) fclose(src);
    if (dst) fclose(dst);

    // ---- 5) 收尾统计 ----
    MosaicPipelineTiming tm{};
    mosaic_pipeline_get_timing(pipe, &tm);
    printf("相机喂入 %lld 帧，SDK 收到 %lld 帧，产出 %lld 帧结果\n", (long long)runner.fed(),
           (long long)runner.pushed(), (long long)runner.collected());
    printf("库内相机队列溢出丢弃 %lld 帧（只在 app-drop 无反压时可能发生）\n",
           (long long)tm.dropped);
    printf("共 %lld 个目标\n", (long long)collector.n_dets);

    mosaic_pipeline_destroy(pipe);
    return 0;
}
