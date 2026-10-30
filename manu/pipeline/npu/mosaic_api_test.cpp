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

int main(int argc, char** argv) {
  // ---- 命令行参数（默认值就是板卡上的那一套，不带参数也能直接跑）----------
  std::string raw_root = "/mnt/manu/lq_ir_bmp";
  std::string sequence = "VIDEO00007_19700101_001324";
  std::string model = "/mnt/manu/ptq/models/p0_off_A_none.rknn";
  std::string kernel_dir = "/root/kernels";
  int limit = 60;                // 最多跑多少帧
  bool env_gate = true;          // 默认体检；--no-env-gate 才跳过

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool wants_value = (arg == "--raw-root" || arg == "--sequence" ||
                              arg == "--model" || arg == "--kernel-dir" ||
                              arg == "--limit");
    if (wants_value && i + 1 >= argc) {
      fprintf(stderr, "[FATAL] %s 缺少参数值\n", arg.c_str());
      return 2;
    }
    if      (arg == "--raw-root")    raw_root = argv[++i];
    else if (arg == "--sequence")    sequence = argv[++i];
    else if (arg == "--model")       model = argv[++i];
    else if (arg == "--kernel-dir")  kernel_dir = argv[++i];
    else if (arg == "--limit")       limit = atoi(argv[++i]);
    else if (arg == "--no-env-gate") env_gate = false;
    else { fprintf(stderr, "[FATAL] 未知参数 %s\n", arg.c_str()); return 2; }
  }

  // ---- 1. open：配置 + 环境体检 + 建流水线 --------------------------------
  // 这一个函数就把「配置 + 体检 + 建流水线」全做完了：客户端不需要知道
  // preset 里有哪些旋钮，也不需要自己写环境检查。
  MosaicOpenOptions opt;
  memset(&opt, 0, sizeof(opt));   // 先清零：没填的字段就不会是栈上的垃圾
  opt.version    = MOSAIC_API_VERSION;
  opt.preset     = MOSAIC_PRESET_REALTIME;  // D=3 实战档；吞吐档用 MOSAIC_PRESET_THROUGHPUT
  opt.model_path = model.c_str();
  opt.kernel_dir = kernel_dir.c_str();
  opt.sequence   = sequence.c_str();
  opt.raw_root   = raw_root.c_str();        // 只有环境体检用它
  opt.check_env  = env_gate ? 1 : 0;

  char err[512] = {0};
  MosaicPipeline* pipeline = mosaic_pipeline_open(&opt, err, sizeof(err));
  if (!pipeline) {
    // ⚠️ err 里写的是「为什么失败」的可读原因：体检没过时是整段 [env] 报告
    //    （模型在不在、kernel 目录在不在、帧目录在不在…）。
    //    一定要把它打印出来 —— 只说「打开失败」等于什么信息都没给。
    fprintf(stderr, "[FATAL] 打开流水线失败：\n%s\n", err);
    return 1;
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
  const auto collect_frame = [](void* user, int32_t index, const char* frame_path) {
    (void)index;
    std::vector<std::string>* out = static_cast<std::vector<std::string>*>(user);
    if (out && frame_path) out->push_back(std::string(frame_path));
  };
  const int32_t found =
      mosaic_pipeline_list_frames(seq_dir.c_str(), collect_frame, &frames, 4096);
  if (found <= 0 || frames.empty()) {
    fprintf(stderr, "[FATAL] %s 下没有帧\n", seq_dir.c_str());
    rc = 1;
  }
  if (limit > 0 && (int)frames.size() > limit) frames.resize((size_t)limit);

  // ---- 2) push 一帧 → 3) collect 一次 -------------------------------------
  // ⚠️ 必须「一推一收」（一次只推一帧）：流水线是「机会主义最新帧」调度 ——
  //    引擎只算它此刻能看到的最新的那一帧，其余的一律跳过。一次推多帧只会
  //    回最后一帧的结果，前面那些帧永远等不到 -> collect 会永久阻塞。
  int total_dets = 0;
  for (int i = 0; i < (int)frames.size() && rc == 0; ++i) {
    // 灰度图，尺寸必须对（由库校验，不对会 push 失败）。
    const cv::Mat img = cv::imread(frames[i], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      fprintf(stderr, "[FATAL] 读不出 %s\n", frames[i].c_str());
      rc = 1;
      break;
    }

    const int64_t id = mosaic_pipeline_push(pipeline, img.data, img.cols, img.rows);
    if (id < 0) {
      fprintf(stderr, "[FATAL] push 失败（帧 %d）\n", i);
      rc = 1;
      break;
    }

    int64_t fid = -1;                 // collect 回来的帧号
    MosaicDet dets[64];
    const int32_t nd = mosaic_pipeline_collect(pipeline, &fid, dets, 64);
    if (nd < 0) {                     // -1 = 流水线已经结束，不会再有结果
      fprintf(stderr, "[FATAL] collect 失败（帧 %d）\n", i);
      rc = 1;
      break;
    }

    total_dets += nd;
    printf("[demo] frame %-6d push_id=%-6lld collect_id=%-6lld 检出 %d 个目标\n",
           i, (long long)id, (long long)fid, nd);
    for (int32_t k = 0; k < nd; ++k) {
      // 坐标是原图（640x512）上的像素：cx/cy 是框中心，w/h 是宽高。
      printf("        #%d cx=%.1f cy=%.1f w=%d h=%d peak=%.3f\n",
             k, dets[k].cx, dets[k].cy, dets[k].w, dets[k].h, dets[k].peak);
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
  if (mosaic_pipeline_get_timing(pipeline, &t) == 0) {
    printf("[demo] 完成 %lld 帧 / 用时 %.1f s（demo 端到端速率 %.2f 帧/秒，"
           "含读帧与等待，不是流水线吞吐上限），检出 %d 个目标，"
           "丢帧 camera=%lld feature=%lld detection=%lld，GMC 组 %lld，BufferA 丢弃 %lld\n",
           (long long)t.frames, t.wall_ms / 1000.0,
           t.frames * 1000.0 / (t.wall_ms > 0 ? t.wall_ms : 1),
           total_dets,
           (long long)t.dropped, (long long)t.features_dropped,
           (long long)t.detections_dropped, (long long)t.gmc_groups_done,
           (long long)t.bufA_dropped);
  }

  // ---- 4. destroy ---------------------------------------------------------
  mosaic_pipeline_destroy(pipeline);
  return rc;
}
