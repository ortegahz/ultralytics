// ===========================================================================
// mosaic_api_test.cpp -- 给客户的示例 demo (EXAMPLE DEMO)
//
// 客户端要写的全部代码，就是下面这四个调用：
//
//   1) mosaic_pipeline_open()     配置 + 环境体检 + 建流水线（一个函数搞定）
//   2) mosaic_pipeline_push()     送一帧灰度图进去
//   3) mosaic_pipeline_collect()  取回这一帧的检测框
//   4) mosaic_pipeline_destroy()  释放一切
//
// 库的边界行为（契约）由 mosaic_contract_test.cpp 负责，本文件不碰：
// 这里只讲「怎么用」，不讲「库保证了什么」。
// ===========================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "mosaic_pipeline.h"

// ---------------------------------------------------------------------------
// 把一帧的检测写成一行 TSV。
// 格式与 mosaic::FormatTsvLine()（mosaic_core.h）逐字相同，PC 评分器可以直接
// 吃这份文件：
//     <stem>\t<count>\t<tag> <cx> <cy> <w> <h> <area> <peak> <sum>\t...
// 0 检出的帧也要写一行（"<stem>\t0"）：评分器靠它判断「这一帧管线跑过了」。
// ---------------------------------------------------------------------------
static void WriteTsvLine(FILE* f, const std::string& stem, const MosaicDet* dets, int32_t n)
{
    fprintf(f, "%s\t%d", stem.c_str(), (int)n);
    for (int32_t k = 0; k < n; ++k)
    {
        fprintf(f, "\t%d %.4f %.4f %d %d %d %.6f %.4f", dets[k].tag, (double)dets[k].cx,
                (double)dets[k].cy, dets[k].w, dets[k].h, dets[k].area, (double)dets[k].peak,
                (double)dets[k].sum);
    }
    fprintf(f, "\n");
}

int main(int argc, char** argv)
{
    // ---- 命令行参数（默认值就是板卡上的那一套，不带参数也能直接跑）----------
    std::string raw_root = "/mnt/manu/lq_ir_bmp";
    std::string sequence = "VIDEO00007_19700101_001324";
    std::string model = "/mnt/manu/ptq/models/p0_off_A_none.rknn";
    std::string kernel_dir = "/root/kernels";
    std::string dump_tsv; // 空 = 不导出 TSV
    int limit = 60; // 最多跑多少帧
    bool env_gate = true; // 默认体检；--no-env-gate 才跳过
    // 后处理算法：sparse（默认，与 dense **逐字节等价**且快 71 倍）/ dense / verify / gpu
    int32_t post_mode = MOSAIC_POST_SPARSE;
    std::string post_mode_name = "sparse";

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const bool wants_value = (arg == "--raw-root" || arg == "--sequence" ||
            arg == "--model" || arg == "--kernel-dir" ||
            arg == "--limit" || arg == "--post-mode" || arg == "--dump-tsv");
        if (wants_value && i + 1 >= argc)
        {
            fprintf(stderr, "[FATAL] %s 缺少参数值\n", arg.c_str());
            return 2;
        }
        if (arg == "--raw-root") raw_root = argv[++i];
        else if (arg == "--sequence") sequence = argv[++i];
        else if (arg == "--model") model = argv[++i];
        else if (arg == "--kernel-dir") kernel_dir = argv[++i];
        else if (arg == "--limit") limit = atoi(argv[++i]);
        else if (arg == "--no-env-gate") env_gate = false;
        else if (arg == "--post-mode")
        {
            const std::string v = argv[++i];
            if (v == "dense") { post_mode = MOSAIC_POST_DENSE; post_mode_name = v; }
            else if (v == "sparse") { post_mode = MOSAIC_POST_SPARSE; post_mode_name = v; }
            else if (v == "verify") { post_mode = MOSAIC_POST_VERIFY; post_mode_name = v; }
            else if (v == "gpu") { post_mode = MOSAIC_POST_GPU; post_mode_name = v; }
            else
            {
                fprintf(stderr, "[FATAL] --post-mode 只支持 dense|sparse|verify|gpu\n");
                return 2;
            }
        }
        else if (arg == "--dump-tsv") dump_tsv = argv[++i];
        else
        {
            fprintf(stderr, "[FATAL] 未知参数 %s\n", arg.c_str());
            return 2;
        }
    }

    // ---- 1. open：配置 + 环境体检 + 建流水线 --------------------------------
    // 这一个函数就把「配置 + 体检 + 建流水线」全做完了：客户端不需要知道
    // preset 里有哪些旋钮，也不需要自己写环境检查。
    MosaicOpenOptions opt;
    memset(&opt, 0, sizeof(opt)); // 先清零：没填的字段就不会是栈上的垃圾
    opt.version = MOSAIC_API_VERSION;
    opt.preset = MOSAIC_PRESET_REALTIME; // D=3 实战档；吞吐档用 MOSAIC_PRESET_THROUGHPUT
    opt.model_path = model.c_str();
    opt.kernel_dir = kernel_dir.c_str();
    opt.sequence = sequence.c_str();
    opt.raw_root = raw_root.c_str(); // 只有环境体检用它
    opt.check_env = env_gate ? 1 : 0;

    char err[512] = {0};
    MosaicPipeline* pipeline = mosaic_pipeline_open(&opt, err, sizeof(err));
    if (!pipeline)
    {
        // ⚠️ err 里写的是「为什么失败」的可读原因：体检没过时是整段 [env] 报告
        //    （模型在不在、kernel 目录在不在、帧目录在不在…）。
        //    一定要把它打印出来 —— 只说「打开失败」等于什么信息都没给。
        fprintf(stderr, "[FATAL] 打开流水线失败：\n%s\n", err);
        return 1;
    }

    // ---- 后处理算法：dense（默认）/ sparse / verify / gpu --------------------
    // 在 open 之后、第一次 push 之前切：已经在处理中的那一帧仍按它开始时的模式
    // 跑完，下一帧才用新模式，所以运行时切也不需要停机。
    if (mosaic_pipeline_set_post_mode(pipeline, post_mode) != 0)
    {
        fprintf(stderr, "[FATAL] 设置后处理模式失败（%s）\n", post_mode_name.c_str());
        mosaic_pipeline_destroy(pipeline);
        return 1;
    }
    printf("[demo] 后处理模式 %s\n", post_mode_name.c_str());

    // ---- 可选：把逐帧检测导出成 TSV（给 PC 评分器打分）---------------------
    FILE* tsv = nullptr;
    if (!dump_tsv.empty())
    {
        tsv = fopen(dump_tsv.c_str(), "w");
        if (!tsv)
        {
            fprintf(stderr, "[FATAL] 打不开 TSV 输出文件 %s\n", dump_tsv.c_str());
            mosaic_pipeline_destroy(pipeline);
            return 1;
        }
        printf("[demo] 检测结果导出到 %s\n", dump_tsv.c_str());
    }

    // 出错时统一走到底部：destroy 只调一次，不会重复释放。
    int rc = 0;

    // ---- 帧列表：用库里的 mosaic_pipeline_list_frames -----------------------
    // 帧目录在 <raw_root>/<sequence> 下，格式是 frame_%06d.bmp。
    const std::string seq_dir = raw_root + "/" + sequence;
    std::vector<std::string> frames;
    // 收集回调：库把每一帧的路径交给我们，我们只负责收下。
    // ⚠️ frame_path 只在回调期间有效：回调一返回那个指针就失效了，
    //    所以必须「立刻拷贝」进 std::string，不能把指针存起来以后再用。
    const auto collect_frame = [](void* user, int32_t index, const char* frame_path)
    {
        (void)index;
        std::vector<std::string>* out = static_cast<std::vector<std::string>*>(user);
        if (out && frame_path) out->push_back(std::string(frame_path));
    };
    const int32_t found =
        mosaic_pipeline_list_frames(seq_dir.c_str(), collect_frame, &frames, 4096);
    if (found <= 0 || frames.empty())
    {
        fprintf(stderr, "[FATAL] %s 下没有帧\n", seq_dir.c_str());
        rc = 1;
    }
    if (limit > 0 && (int)frames.size() > limit) frames.resize((size_t)limit);

    // ---- 2) push 一帧 → 3) collect 一次 -------------------------------------
    // ⚠️ 必须「一推一收」（一次只推一帧）：流水线是「机会主义最新帧」调度 ——
    //    引擎只算它此刻能看到的最新的那一帧，其余的一律跳过。一次推多帧只会
    //    回最后一帧的结果，前面那些帧永远等不到 -> collect 会永久阻塞。
    int total_dets = 0;
    for (int i = 0; i < (int)frames.size() && rc == 0; ++i)
    {
        // 灰度图，尺寸必须对（由库校验，不对会 push 失败）。
        const cv::Mat img = cv::imread(frames[i], cv::IMREAD_GRAYSCALE);
        if (img.empty())
        {
            fprintf(stderr, "[FATAL] 读不出 %s\n", frames[i].c_str());
            rc = 1;
            break;
        }

        const int64_t id = mosaic_pipeline_push(pipeline, img.data, img.cols, img.rows);
        if (id < 0)
        {
            fprintf(stderr, "[FATAL] push 失败（帧 %d）\n", i);
            rc = 1;
            break;
        }

        int64_t fid = -1; // collect 回来的帧号
        MosaicDet dets[64];
        const int32_t nd = mosaic_pipeline_collect(pipeline, &fid, dets, 64);
        if (nd < 0)
        {
            // -1 = 流水线已经结束，不会再有结果
            fprintf(stderr, "[FATAL] collect 失败（帧 %d）\n", i);
            rc = 1;
            break;
        }

        total_dets += nd;
        printf("[demo] frame %-6d push_id=%-6lld collect_id=%-6lld 检出 %d 个目标\n",
               i, (long long)id, (long long)fid, nd);
        for (int32_t k = 0; k < nd; ++k)
        {
            // 坐标是原图（640x512）上的像素：cx/cy 是框中心，w/h 是宽高。
            // tag 是来源拼图块（0=原图 1=thumb160 2=thumb80），area 是连通域像素数。
            printf("        #%d tag=%d cx=%.1f cy=%.1f w=%d h=%d area=%d peak=%.3f\n",
                   k, dets[k].tag, dets[k].cx, dets[k].cy, dets[k].w, dets[k].h, dets[k].area,
                   dets[k].peak);
        }
        if (tsv)
        {
            // 帧号用 push 的序号（0 基、6 位补零），不是 collect 回来的 fid：
            // 评分器按 stem 里 "__" 之后的数字去对齐 PC 侧的帧索引。
            // 0 检出的帧也要写一行（"stem\t0"）。
            char stem[256];
            snprintf(stem, sizeof(stem), "%s__%06d", sequence.c_str(), i);
            WriteTsvLine(tsv, stem, dets, nd);
        }
    }

    // ---- 统计（跑完看看流水线自己的账）-------------------------------------
    // ⚠️ 必须先 flush 再取 timing：wall_ms 在 create 时写进去的是「起始时间戳」，
    //    只有 flush() 把线程收干净、把它换算成「已用时长」之后，它才是真正的
    //    已用毫秒数；flush 之前直接读，拿到的是「开机以来的毫秒数」（会离谱地大）。
    //    flush 同时把内部线程都 join 掉，退出时不留后台线程。
    //    想测「流水线吞吐上限」请用 rknn_mosaic_e2e（连续推帧）；本 demo 是
    //    一推一收、逐帧等结果的，端到端速率里含了读帧(NFS)和等待时间。
    mosaic_pipeline_flush(pipeline);
    MosaicPipelineTiming t;
    if (mosaic_pipeline_get_timing(pipeline, &t) == 0)
    {
        // 后处理耗时：backproj_ms 是累计值，除以帧数得到每帧平均。
        // 帧数为 0 时不除（避免除零），直接写 n/a。
        char post_ms[32];
        if (t.frames > 0)
            snprintf(post_ms, sizeof(post_ms), "%.3f", t.backproj_ms / (double)t.frames);
        else
            snprintf(post_ms, sizeof(post_ms), "n/a");

        printf("[demo] 完成 %lld 帧 / 用时 %.1f s（demo 端到端速率 %.2f 帧/秒，"
               "含读帧与等待，不是流水线吞吐上限），检出 %d 个目标，"
               "丢帧 camera=%lld feature=%lld detection=%lld，GMC 组 %lld，BufferA 丢弃 %lld，"
               "后处理 %s ms/帧\n",
               (long long)t.frames, t.wall_ms / 1000.0,
               t.frames * 1000.0 / (t.wall_ms > 0 ? t.wall_ms : 1),
               total_dets,
               (long long)t.dropped, (long long)t.features_dropped,
               (long long)t.detections_dropped, (long long)t.gmc_groups_done,
               (long long)t.bufA_dropped,
               post_ms);
    }

    // ---- 后处理耗时的分布（均值会掩盖抖动，抖动才是延时预算要吸收的）-----
    MosaicPostTiming pt;
    if (mosaic_pipeline_get_post_timing(pipeline, &pt) == 0 && pt.frames > 0)
    {
        printf("[demo] 后处理分布（%d 帧）：均值 %.3f ms｜标准差 %.3f ms｜"
               "p50 %.3f｜p90 %.3f｜p95 %.3f｜p99 %.3f｜min %.3f｜max %.3f\n",
               pt.frames, pt.mean_ms, pt.stddev_ms, pt.p50_ms, pt.p90_ms, pt.p95_ms,
               pt.p99_ms, pt.min_ms, pt.max_ms);
    }
    MosaicGpuPostTiming gt;
    if (mosaic_pipeline_get_gpu_post_timing(pipeline, &gt) == 0 && gt.frames > 0)
    {
        printf("[demo] GPU 分段（%d 帧，均值 ms）：上传 %.3f（σ %.3f）｜"
               "kernel %.3f（σ %.3f）｜回读 %.3f（σ %.3f）｜整段 %.3f（σ %.3f）｜"
               "p99 %.3f｜max %.3f｜丢弃候选 %lld\n",
               gt.frames, gt.upload.mean_ms, gt.upload.stddev_ms,
               gt.kernel.mean_ms, gt.kernel.stddev_ms,
               gt.readback.mean_ms, gt.readback.stddev_ms,
               gt.total.mean_ms, gt.total.stddev_ms,
               gt.total.p99_ms, gt.total.max_ms, (long long)gt.peaks_dropped);
    }
    if (tsv) fclose(tsv);

    // ---- 4. destroy ---------------------------------------------------------
    mosaic_pipeline_destroy(pipeline);
    return rc;
}
