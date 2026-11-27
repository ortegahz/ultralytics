// ===========================================================================
// mosaic_tracked_cb_example.cpp -- 嵌入式接「一目标一框」的示范
//
// ---------------------------------------------------------------------------
// 这份程序回答一个问题：**为什么嵌入式跑出来一个目标身上画了两三个框？**
//
//   因为嵌入式拿到的通常是**原始候选框**（mosaic_pipeline_set_result_callback
//   送的那一路）：同一个红外目标在**多个尺度**上各有一个连通域候选 ⇒ 一个目标
//   对应 2~3 个框。跟踪结果（时空融合 + 跨尺度合并 + 连续帧确认）才是
//   **一个目标一个框**。
//
//   v10.7 之前，拿「一目标一框」必须用**轮询**接口：
//       mosaic_pipeline_collect_tracked(p, &fid, out, n);
//   —— 但轮询接口和「回调式」集成天然别扭：嵌入式想在自己的线程里拿框，
//   而 collect_tracked 是**阻塞**的；而且如果嵌入式只想要跟踪结果、
//   永远不取原始候选，那条队列会**每帧涨一条、永不排空**。
//
//   v10.7 新增**跟踪结果回调**（纯追加，不改任何既有接口）：
//       typedef void (*MosaicTrackedCallback)(void* user, int64_t frame_id,
//                                             const MosaicTrackedDet* dets, int32_t n_dets);
//       mosaic_pipeline_set_tracked_callback(p, on_tracked, my_user);
//
//   ⇒ 嵌入式只要「**改 1 个函数名 + 1 个结构体类型**」：
//       - 函数名：mosaic_pipeline_set_result_callback  ->  mosaic_pipeline_set_tracked_callback
//       - 结构体：const MosaicDet*                    ->  const MosaicTrackedDet*
//     拿到的就是**一个目标一个框**。
//
// ---------------------------------------------------------------------------
// 本程序同时注册**两路**回调，做一次对照：
//   * raw 回调    —— 原始候选框（一目标多框，问题的来源）
//   * tracked 回调—— 跟踪结果（一目标一框，交付要的东西）
// 并在最后核对：**同一帧里 tracked 的框数 <= raw 的框数**，且
// **同一帧内 track_id 不重复**（重复 = 一个目标画了多个框 = 没做到位）。
//
// ---------------------------------------------------------------------------
// 线程规矩（很重要）：tracked 回调跑在**库自己的派发线程**上。
//   * 回调里**不要**再调用 mosaic_pipeline_* 的阻塞接口（collect /
//     collect_tracked）⇒ 会死锁。
//   * 回调里**不要** destroy 流水线。
//   * 需要慢活（画框、IO）就自己往队列里塞数据，别在回调里干。
// ===========================================================================
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "mosaic_pipeline.h"
#include "mosaic_paths.h"

namespace {

struct Stats
{
    std::mutex mu;
    std::map<int64_t, int32_t> raw_per_frame;   // 帧号 -> 原始候选框数
    std::map<int64_t, int32_t> tracked_per_frame;// 帧号 -> 跟踪结果框数
    int64_t raw_calls = 0;
    int64_t tracked_calls = 0;
    int64_t total_raw = 0;
    int64_t total_tracked = 0;
    int32_t max_raw = 0;
    int32_t max_tracked = 0;
    int64_t dup_track_id_frames = 0;   // 同一帧内 track_id 重复的帧数
    int64_t tracked_gt_raw_frames = 0; // tracked 框数 > raw 框数的帧数
    int64_t comparable_frames = 0;
};

// The RAW callback -- the OLD way (one target, several boxes).
void on_raw(void* user, int64_t frame_id, const MosaicDet* dets, int32_t n_dets)
{
    Stats* s = static_cast<Stats*>(user);
    std::lock_guard<std::mutex> lk(s->mu);
    ++s->raw_calls;
    s->total_raw += n_dets;
    s->raw_per_frame[frame_id] = n_dets;
    if (n_dets > s->max_raw) s->max_raw = n_dets;
    (void)dets;
}

// The TRACKED callback -- the NEW way (one target, ONE box).
void on_tracked(void* user, int64_t frame_id, const MosaicTrackedDet* dets, int32_t n_dets)
{
    Stats* s = static_cast<Stats*>(user);
    std::lock_guard<std::mutex> lk(s->mu);
    ++s->tracked_calls;
    s->total_tracked += n_dets;
    s->tracked_per_frame[frame_id] = n_dets;
    if (n_dets > s->max_tracked) s->max_tracked = n_dets;

    // ONE TARGET, ONE BOX: the track ids inside one frame must be unique.
    std::vector<int32_t> ids;
    ids.reserve((size_t)(n_dets > 0 ? n_dets : 0));
    for (int32_t i = 0; i < n_dets; ++i) ids.push_back(dets[i].track_id);
    std::sort(ids.begin(), ids.end());
    if (std::unique(ids.begin(), ids.end()) != ids.end()) ++s->dup_track_id_frames;

    // The same frame's RAW count, if the raw callback already delivered it.
    std::map<int64_t, int32_t>::const_iterator it = s->raw_per_frame.find(frame_id);
    if (it != s->raw_per_frame.end())
    {
        ++s->comparable_frames;
        if (n_dets > it->second) ++s->tracked_gt_raw_frames;
    }

    // The client draws ONE rectangle per entry -- this is the whole point.
    printf("[tracked] frame=%lld  targets=%d\n", (long long)frame_id, n_dets);
    for (int32_t i = 0; i < n_dets; ++i)
    {
        printf("            id=%-4d cx=%.1f cy=%.1f score=%.3f%s\n", dets[i].track_id,
               (double)dets[i].x, (double)dets[i].y, dets[i].score,
               dets[i].infilled ? " (interpolated)" : "");
    }
}

const char* kSequence = "VIDEO00007_19700101_001324";
const int kDefaultFrames = 60;

}  // namespace

int main(int argc, char** argv)
{
    std::string frame_dir, model, kernel;
    std::string seq = kSequence;
    // "both"       : 两路回调都注册（做同帧对照用）
    // "tracked-only": **只**注册跟踪回调（嵌入式推荐的最小用法）
    std::string mode = "both";
    int want = kDefaultFrames;

    int pos = 0;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const bool wants_value = (arg == "--frames-dir" || arg == "--model" ||
                                  arg == "--kernel-dir" || arg == "--seq" || arg == "--frames" ||
                                  arg == "--mode");
        if (wants_value && i + 1 >= argc)
        {
            fprintf(stderr, "[FATAL] %s 缺少参数值\n", arg.c_str());
            return 2;
        }
        if (arg == "--frames-dir") frame_dir = argv[++i];
        else if (arg == "--model") model = argv[++i];
        else if (arg == "--kernel-dir") kernel = argv[++i];
        else if (arg == "--seq") seq = argv[++i];
        else if (arg == "--frames") want = atoi(argv[++i]);
        else if (arg == "--mode") mode = argv[++i];
        else if (arg.size() > 1 && arg[0] == '-')
        {
            fprintf(stderr, "[FATAL] 未知参数 %s\n", arg.c_str());
            return 2;
        }
        else
        {
            switch (pos++)
            {
                case 0: frame_dir = arg; break;
                case 1: model = arg; break;
                case 2: kernel = arg; break;
                case 3: want = atoi(arg.c_str()); break;
                default: fprintf(stderr, "[FATAL] 多余的位置参数 %s\n", arg.c_str()); return 2;
            }
        }
    }

    if (frame_dir.empty()) frame_dir = mosaic_paths::FramesRoot() + "/" + seq;
    if (model.empty()) model = mosaic_paths::ModelPath();
    if (kernel.empty()) kernel = mosaic_paths::KernelDir();

    // ---- 1) 建流水线（和嵌入式完全一样的四行）------------------------------
    MosaicOpenOptions opt = {};
    opt.version     = MOSAIC_API_VERSION;
    opt.preset      = MOSAIC_PRESET_REALTIME;
    opt.model_path  = model.c_str();
    opt.kernel_dir  = kernel.c_str();
    opt.check_env   = 1;

    char err[512] = {0};
    MosaicPipeline* p = mosaic_pipeline_open(&opt, err, sizeof(err));
    if (!p)
    {
        fprintf(stderr, "[FATAL] 建流水线失败：%s\n", err);
        return 1;
    }

    // ---- 2) 两路回调都注册（嵌入式只要第 2 行；第 1 行是为了做对照）--------
    int failures = 0;
    Stats stats;
    const bool tracked_only = (mode == "tracked-only");
    if (!tracked_only)
    {
        // 【只做对照用】嵌入式不需要这一行。
        if (mosaic_pipeline_set_result_callback(p, on_raw, &stats) != 0)
        {
            fprintf(stderr, "[FATAL] 注册原始结果回调失败\n");
            mosaic_pipeline_destroy(p);
            return 1;
        }
    }
    // ★★★ 嵌入式要加的就是这一行 ★★★（**唯一必需**的一行）
    if (mosaic_pipeline_set_tracked_callback(p, on_tracked, &stats) != 0)
    {
        fprintf(stderr, "[FATAL] 注册跟踪结果回调失败\n");
        mosaic_pipeline_destroy(p);
        return 1;
    }
    // The RELEASE version of the SDK that is ACTUALLY loaded (the library),
    // next to the compile-time macro MOSAIC_SDK_VERSION (the header).  If
    // the two disagree, the header and the library are different builds.
    {
        char sdk_v[32];
        mosaic_pipeline_sdk_version(sdk_v, (int32_t)sizeof(sdk_v));
        printf("[demo] mosaic sdk v%s\n", sdk_v);
    }
    printf("[mode] %s\n", tracked_only
                                  ? "tracked-only（只注册跟踪回调 = 嵌入式最小用法）"
                                  : "both（两路回调都注册 = 同帧对照用）");

    // ---- 3) 列帧 ------------------------------------------------------------
    std::vector<std::string> frames;
    const auto on_frame = [](void* user, int32_t index, const char* frame_path)
    {
        (void)index;
        std::vector<std::string>* out = static_cast<std::vector<std::string>*>(user);
        if (out && frame_path) out->push_back(std::string(frame_path));
    };
    if (mosaic_pipeline_list_frames(frame_dir.c_str(), on_frame, &frames, 4096) <= 0 ||
        frames.empty())
    {
        fprintf(stderr, "[FATAL] %s 下没有帧\n", frame_dir.c_str());
        mosaic_pipeline_destroy(p);
        return 1;
    }
    if (want > 0 && (int)frames.size() > want) frames.resize((size_t)want);

    // ---- 4) 逐帧推进（不 collect：跟踪结果走回调）--------------------------
    for (size_t i = 0; i < frames.size(); ++i)
    {
        const cv::Mat img = cv::imread(frames[i], cv::IMREAD_GRAYSCALE);
        if (img.empty())
        {
            fprintf(stderr, "[FATAL] 读不出 %s\n", frames[i].c_str());
            break;
        }
        // 用**自己的**帧号推：回调回来的 frame_id 就是这个号。
        mosaic_pipeline_push_frame(p, img.data, img.cols, img.rows, (int64_t)(1000 + i));
        // 推进节奏：push 不阻塞 ⇒ 推太快会丢帧。等回调回来再推下一帧。
        // （嵌入式有自己的相机线程时不需要这一步。）
        for (int wait = 0; wait < 400; ++wait)
        {
            bool arrived = false;
            {
                std::lock_guard<std::mutex> lk(stats.mu);
                if (tracked_only)
                    arrived = (stats.tracked_calls > (int64_t)i);
                else
                    arrived = stats.raw_per_frame.count(1000 + (int64_t)i) != 0;
            }
            if (arrived) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    // 【tracked-only 的关键行为】只注册跟踪回调 ⇒ 原始候选流**不再缓冲**
    //（否则那条队列每帧涨一条、永不排空）⇒ collect() 必须立刻返回 -1，
    // 而不是阻塞到 flush。实测一下，把这个坑钉死。
    if (tracked_only)
    {
        MosaicDet rdet[16];
        int64_t rfid = -1;
        const int32_t rdc = mosaic_pipeline_collect(p, &rfid, rdet, 16);
        printf("[tracked-only] mosaic_pipeline_collect() 返回 %d（期望 -1：原始流未缓冲）\n", rdc);
        if (rdc != -1)
        {
            fprintf(stderr, "[FAIL] tracked-only 模式下 collect() 没有返回 -1\n");
            ++failures;
        }
    }

    // flush() 等所有已派发的回调执行完 ⇒ 统计是完整的。
    mosaic_pipeline_flush(p);
    mosaic_pipeline_destroy(p);

    // ---- 5) 结论 ------------------------------------------------------------
    std::lock_guard<std::mutex> lk(stats.mu);
    printf("\n================ 结果 ================\n");
    printf("推入帧数            : %zu\n", frames.size());
    printf("raw 回调次数        : %lld（原始候选框共 %lld 个，单帧最多 %d）\n",
           (long long)stats.raw_calls, (long long)stats.total_raw, stats.max_raw);
    printf("tracked 回调次数    : %lld（跟踪框共 %lld 个，单帧最多 %d）\n",
           (long long)stats.tracked_calls, (long long)stats.total_tracked, stats.max_tracked);
    printf("同帧内 track_id 重复: %lld 帧\n", (long long)stats.dup_track_id_frames);
    printf("tracked 框 > raw 框 : %lld 帧（可比帧 %lld）\n",
           (long long)stats.tracked_gt_raw_frames, (long long)stats.comparable_frames);

    if (stats.tracked_calls <= 0)
    {
        fprintf(stderr, "[FAIL] 跟踪回调一次都没触发\n");
        ++failures;
    }
    if (stats.total_tracked <= 0)
    {
        fprintf(stderr, "[FAIL] 跟踪回调没有交付任何目标\n");
        ++failures;
    }
    if (stats.dup_track_id_frames != 0)
    {
        fprintf(stderr, "[FAIL] 有 %lld 帧里同一个 track_id 出现了多次（一个目标画了多个框）\n",
                (long long)stats.dup_track_id_frames);
        ++failures;
    }
    if (stats.tracked_gt_raw_frames != 0)
    {
        fprintf(stderr, "[FAIL] 有 %lld 帧的跟踪框比原始候选框还多\n",
                (long long)stats.tracked_gt_raw_frames);
        ++failures;
    }
    if (failures == 0)
    {
        printf("[PASS] 一目标一框：跟踪回调交付 %lld 个目标，帧内 track_id 无重复\n",
               (long long)stats.total_tracked);
        printf("[PASS] 原始候选 %lld 个 -> 跟踪后 %lld 个（合并 %lld 个跨尺度重复框）\n",
               (long long)stats.total_raw, (long long)stats.total_tracked,
               (long long)(stats.total_raw - stats.total_tracked));
        return 0;
    }
    return 1;
}
