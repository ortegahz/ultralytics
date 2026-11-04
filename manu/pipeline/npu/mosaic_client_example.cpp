// ===========================================================================
// mosaic_client_example.cpp -- 最简集成示范：这个 SDK 就这么用
//
// ---------------------------------------------------------------------------
// 嵌入式组真正要写的代码，就是下面这几句（配置只有「模型 + kernel + 档位」
// 三个字段，外加一行钉死后处理）：
//
//     MosaicOpenOptions opt = {};
//     opt.version     = MOSAIC_API_VERSION;
//     opt.preset      = MOSAIC_PRESET_REALTIME;    // 最优档：3 级流水线 + D=3
//     opt.model_path  = model.c_str();
//     opt.kernel_dir  = kernel.c_str();
//     // sequence / raw_root 留空：自己喂帧的客户端没有帧目录可指，留空 =
//     // 跳过环境体检（**不是**体检失败），流水线照常全套跑起来。
//     opt.check_env   = 1;
//
//     char err[512] = {0};
//     MosaicPipeline* p = mosaic_pipeline_open(&opt, err, sizeof(err));
//     if (!p) { fprintf(stderr, "%s\n", err); return 1; }
//
//     mosaic_pipeline_push(p, gray, w, h);                    // 一帧灰度图推进去
//     MosaicDet dets[128]; int64_t fid = -1;
//     int32_t nd = mosaic_pipeline_collect(p, &fid, dets, 128);   // 检测框取回来
//
//     mosaic_pipeline_destroy(p);
//
// ---------------------------------------------------------------------------
// 三条必须记住的规矩（照抄本文件就不会踩坑）：
//
//   1) **一推一收**：一次只推一帧。流水线是「机会主义最新帧」调度，一次推多帧
//      只会回最后一帧的结果，前面的帧永远等不到 ⇒ collect 会**永久阻塞**。
//   2) **collect 是阻塞的**：一直等到这一帧算完才返回。
//   3) **帧尺寸要一致**：本 SDK 吃 640x512 的 8-bit 灰度图（w*h 字节）。
//
// ---------------------------------------------------------------------------
// 本文件只用**公开 C ABI**（src/npu/mosaic_pipeline.h），一个内部符号都没碰。
// 下面读图那两行只是为了让它能独立跑起来；**你换成自己的相机 buffer 即可**
// —— 只要手里有一张 uint8_t 灰度图（w*h 字节）就能推。
//
// 后处理默认就是最优的 sparse 路径；下面那一行 set_post_mode 是为了**不受
// MOSAIC_POST_MODE 环境变量干扰**，让「我们跑的是最优路径」在代码里有据可查。
// ===========================================================================
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "mosaic_pipeline.h"

// ---------------------------------------------------------------------------
// 默认值：板卡上零参数直接跑。换自己的路径就覆盖对应的那一个（位置参数）。
// ---------------------------------------------------------------------------
static const char* kFrameDir  = "/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324";
static const char* kModel     = "/mnt/manu/ptq/models/p0_off_A_none.rknn";
static const char* kKernelDir = "/mnt/manu/mosaic_sdk/runtime/kernels";
// 21 帧 = GMC 的一个完整历史窗口。喂不满 21 帧也能出结果（早期帧的历史会被
// clamp），但**喂满 21 帧才是认证质量的拟合**。
static const int kDefaultFrames = 21;

int main(int argc, char** argv)
{
    // 用法：mosaic_client_example [帧目录] [模型路径] [kernel 目录] [帧数]
    if (argc > 5)
    {
        fprintf(stderr, "用法: %s [帧目录] [模型路径] [kernel 目录] [帧数]\n", argv[0]);
        return 2;
    }
    const char* frame_dir = (argc > 1) ? argv[1] : kFrameDir;
    const char* model     = (argc > 2) ? argv[2] : kModel;
    const char* kernel    = (argc > 3) ? argv[3] : kKernelDir;
    const int   want      = (argc > 4) ? atoi(argv[4]) : kDefaultFrames;

    // ---- 1) 建流水线：配置就上面那几行 -------------------------------------
    MosaicOpenOptions opt = {};
    opt.version     = MOSAIC_API_VERSION;
    opt.preset      = MOSAIC_PRESET_REALTIME;    // 最优档：3 级流水线 + D=3
    opt.model_path  = model;
    opt.kernel_dir  = kernel;
    opt.check_env   = 1;      // raw_root 为空 ⇒ 这一步自动跳过（不是失败）

    char err[512] = {0};
    MosaicPipeline* p = mosaic_pipeline_open(&opt, err, sizeof(err));
    if (!p)
    {
        fprintf(stderr, "[FATAL] 建流水线失败：%s\n", err);
        return 1;
    }

    // 唯一一行「最优配置」：把后处理钉死在 sparse（库默认也是它，写出来是为了
    // 不受 MOSAIC_POST_MODE 环境变量影响）。sparse 与 DENSE 逐比特一致，
    // 但后处理从 37.9 ms/帧降到 0.53 ms/帧。
    if (mosaic_pipeline_set_post_mode(p, MOSAIC_POST_SPARSE) != 0)
    {
        fprintf(stderr, "[FATAL] 设置后处理模式失败\n");
        mosaic_pipeline_destroy(p);
        return 1;
    }

    // ---- 2) 帧从哪来：列目录下的 frame_%06d.bmp ------------------------------
    // （帧也可以来自摄像头 buffer 或网络 —— 你手里有一张 uint8_t 灰度图就能推。）
    std::vector<std::string> frames;
    const auto on_frame = [](void* user, int32_t index, const char* frame_path)
    {
        (void)index;
        std::vector<std::string>* out = static_cast<std::vector<std::string>*>(user);
        // frame_path 只在回调期间有效 ⇒ 必须立刻拷贝，不能存指针。
        if (out && frame_path) out->push_back(std::string(frame_path));
    };
    if (mosaic_pipeline_list_frames(frame_dir, on_frame, &frames, 4096) <= 0 ||
        frames.empty())
    {
        fprintf(stderr, "[FATAL] %s 下没有帧\n", frame_dir);
        mosaic_pipeline_destroy(p);
        return 1;
    }
    if (want > 0 && (int)frames.size() > want) frames.resize((size_t)want);

    // ---- 3) 逐帧：一推一收 --------------------------------------------------
    int32_t total = 0;
    for (size_t i = 0; i < frames.size(); ++i)
    {
        // ↓↓↓ 这两行只是「把文件读成灰度图」而已。**你用自己的相机 buffer 时，
        //     把它们换成你的取图代码**，只要 gray 是一张 uint8_t 灰度图即可。
        const cv::Mat img = cv::imread(frames[i], cv::IMREAD_GRAYSCALE);
        if (img.empty())
        {
            fprintf(stderr, "[FATAL] 读不出 %s\n", frames[i].c_str());
            mosaic_pipeline_destroy(p);
            return 1;
        }

        mosaic_pipeline_push(p, img.data, img.cols, img.rows);

        MosaicDet dets[128];   // 库每帧最多保留 100 个连通域 ⇒ 缓冲区给 128
        int64_t fid = -1;
        const int32_t nd = mosaic_pipeline_collect(p, &fid, dets, 128);
        if (nd < 0)
        {
            fprintf(stderr, "[FATAL] collect 失败（帧 %zu）\n", i);
            mosaic_pipeline_destroy(p);
            return 1;
        }
        total += nd;

        printf("帧 %zu：%d 个目标\n", i, nd);
        for (int32_t k = 0; k < nd; ++k)
        {
            printf("    #%d tag=%d cx=%.1f cy=%.1f w=%d h=%d area=%d peak=%.3f\n",
                   k, dets[k].tag, (double)dets[k].cx, (double)dets[k].cy, dets[k].w,
                   dets[k].h, dets[k].area, (double)dets[k].peak);
        }
    }

    printf("共 %zu 帧，%d 个目标\n", frames.size(), total);
    mosaic_pipeline_destroy(p);
    return 0;
}
