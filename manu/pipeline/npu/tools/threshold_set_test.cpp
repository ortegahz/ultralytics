// threshold_set_test.cpp -- 验证 mosaic_pipeline_set_detect_threshold() 的运行时行为
//
// 目的：给"客户在运行时改检测阈值"这条 API 做**可复现的板端验收**。
//   * 阈值 = 0（不调用 setter）⇒ 走库内冻结默认值 0.06，作为 golden；
//   * 之后依次设成若干阈值 ⇒ 每档跑同一批帧，统计"检出目标数"。
//   期望：阈值调高 ⇒ 检出数单调不增（连通域是在归一化热图上按 heatmap >= thr 长的，
//   thr 越高 ⇒ 候选像素越少 ⇒ 目标越少）。
//
// 用法（板端）：
//   ./threshold_set_test <帧目录> <模型> <kernel目录> [每档帧数]
// 例：
//   ./threshold_set_test /root/bts/frames/VIDEO00007_19700101_001324 \
//       /root/bts/p0_off_A_none.rknn /root/bts/kernels 40
//
// ⚠️ 本程序**不属于交付包**，只是 tools/ 下的验证工具（交付包由
//    packaging/make_build_dir.sh 单独整理，不收它）。
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "mosaic_pipeline.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "用法: %s <帧目录> <模型> <kernel目录> [每档帧数]\n", argv[0]);
    return 2;
  }
  const char* frame_dir = argv[1];
  const char* model = argv[2];
  const char* kernel_dir = argv[3];
  const int per_stage = (argc > 4) ? atoi(argv[4]) : 40;

  MosaicOpenOptions opt = {};
  opt.version = MOSAIC_API_VERSION;
  opt.preset = MOSAIC_PRESET_REALTIME;
  opt.model_path = model;
  opt.kernel_dir = kernel_dir;
  opt.check_env = 1;

  char err[512] = {0};
  MosaicPipeline* p = mosaic_pipeline_open(&opt, err, sizeof(err));
  if (!p) {
    fprintf(stderr, "[FATAL] 建流水线失败: %s\n", err);
    return 1;
  }
  if (mosaic_pipeline_set_post_mode(p, MOSAIC_POST_SPARSE) != 0) {
    fprintf(stderr, "[FATAL] 设置后处理模式失败\n");
    mosaic_pipeline_destroy(p);
    return 1;
  }

  std::vector<std::string> frames;
  const auto on_frame = [](void* user, int32_t index, const char* frame_path) {
    (void)index;
    std::vector<std::string>* out = static_cast<std::vector<std::string>*>(user);
    if (out && frame_path) out->push_back(std::string(frame_path));
  };
  if (mosaic_pipeline_list_frames(frame_dir, on_frame, &frames, 4096) <= 0 || frames.empty()) {
    fprintf(stderr, "[FATAL] %s 下没有帧\n", frame_dir);
    mosaic_pipeline_destroy(p);
    return 1;
  }
  if ((int)frames.size() > per_stage) frames.resize((size_t)per_stage);

  // 0.06 = 库的冻结默认值；0 = 不调用 setter（走默认）
  const double thresholds[] = {0.0, 0.02, 0.06, 0.12, 0.30};
  const int n_stage = (int)(sizeof(thresholds) / sizeof(thresholds[0]));

  printf("| 阈值 | 每帧检出数 | 目标总数 | 说明 |\n");
  printf("|---|---|---|---|\n");
  int rc_all = 0;
  for (int s = 0; s < n_stage; ++s) {
    const double thr = thresholds[s];
    const char* note = "库内默认值（未调用 setter）";
    if (thr > 0.0) {
      const int32_t rc = mosaic_pipeline_set_detect_threshold(p, &thr);
      if (rc != 0) {
        fprintf(stderr, "[FATAL] set_detect_threshold(%.4f) 返回 %d\n", thr, (int)rc);
        rc_all = 1;
        break;
      }
      note = "setter 生效";
    }
    long long total = 0;
    long long frames_with = 0;
    for (size_t i = 0; i < frames.size(); ++i) {
      const cv::Mat img = cv::imread(frames[i], cv::IMREAD_GRAYSCALE);
      if (img.empty()) continue;
      mosaic_pipeline_push(p, img.data, img.cols, img.rows);
      MosaicDet dets[128];
      int64_t fid = -1;
      const int32_t nd = mosaic_pipeline_collect(p, &fid, dets, 128);
      if (nd < 0) break;
      total += nd;
      if (nd > 0) ++frames_with;
    }
    printf("| %.2f | %lld/%zu 帧有目标 | %lld | %s |\n", thr, frames_with, frames.size(), total, note);
    fflush(stdout);
  }

  // 错误码路径（不依赖真实流水线状态的那几条）
  double dummy = 0.5;
  printf("[i] NULL pipeline -> %d（期望 -1）\n", mosaic_pipeline_set_detect_threshold(nullptr, &dummy));
  printf("[i] NULL threshold -> %d（期望 -1）\n", mosaic_pipeline_set_detect_threshold(p, nullptr));
  double bad = 2.0;
  printf("[i] 越界阈值 2.0 -> %d（期望 -2）\n", mosaic_pipeline_set_detect_threshold(p, &bad));

  mosaic_pipeline_destroy(p);
  return rc_all;
}
