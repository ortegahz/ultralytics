// ===========================================================================
// mosaic_track_dump.cpp -- 板卡侧「SDK 内置时空融合」验证工具（不进交付库）
//
//   目的：验证**交付 SDK 里那条真实管线**上的时空融合，与 Python 参考实现
//   （manu/evaluation/sdk_track_fusion_ref.py）**逐位一致**。
//
//   它同时导出三份 TSV：
//     1. cands.tsv     —— 这一轮**真正喂给融合引擎**的候选流（按引擎的更新序号）
//                         （从 mosaic_pipeline_collect() 拿到的检测，就是喂进引擎的那批）
//     2. tracked.tsv   —— 在线融合的逐帧输出（mosaic_pipeline_collect_tracked）
//     3. fused.tsv     —— 离线融合（缝合+插补+确认）的逐帧输出
//                         （mosaic_pipeline_collect_fused，flush 之后才有）
//
//   零 OpenCV 依赖（自带 8bit 灰度 BMP 读取器），只链 SDK 的 .so。
//
//   用法：
//     mosaic_track_dump --dir <bmp目录> --model <rknn> --kernel-dir <目录> \
//        --seq <名字> [--frames N] [--out-prefix P] [--tag-mask 7] \
//        [--reorder-window 0] [--th-base 0.45] ...
// ===========================================================================
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "mosaic_pipeline.h"

namespace {

using clk = std::chrono::steady_clock;

bool LoadBmpGray8(const std::string& path, int* w, int* h, std::vector<uint8_t>* out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  unsigned char hdr[54];
  if (fread(hdr, 1, 54, f) != 54) { fclose(f); return false; }
  if (hdr[0] != 'B' || hdr[1] != 'M') { fclose(f); return false; }
  const uint32_t data_off = (uint32_t)hdr[10] | ((uint32_t)hdr[11] << 8) | ((uint32_t)hdr[12] << 16) |
                            ((uint32_t)hdr[13] << 24);
  const int32_t width = (int32_t)((uint32_t)hdr[18] | ((uint32_t)hdr[19] << 8) |
                                   ((uint32_t)hdr[20] << 16) | ((uint32_t)hdr[21] << 24));
  const int32_t height = (int32_t)((uint32_t)hdr[22] | ((uint32_t)hdr[23] << 8) |
                                    ((uint32_t)hdr[24] << 16) | ((uint32_t)hdr[25] << 24));
  const uint16_t bpp = (uint16_t)((uint32_t)hdr[28] | ((uint32_t)hdr[29] << 8));
  const uint32_t compression = (uint32_t)hdr[30] | ((uint32_t)hdr[31] << 8) | ((uint32_t)hdr[32] << 16) |
                               ((uint32_t)hdr[33] << 24);
  if (bpp != 8 || compression != 0 || width <= 0 || height == 0) { fclose(f); return false; }
  const int aheight = height < 0 ? -height : height;
  const size_t row_bytes = ((size_t)width + 3u) & ~((size_t)3u);
  std::vector<uint8_t> raw(row_bytes * (size_t)aheight);
  if (fseek(f, (long)data_off, SEEK_SET) != 0) { fclose(f); return false; }
  if (fread(raw.data(), 1, raw.size(), f) != raw.size()) { fclose(f); return false; }
  fclose(f);
  out->assign((size_t)width * (size_t)aheight, 0);
  const bool bottom_up = height > 0;
  for (int y = 0; y < aheight; ++y) {
    const int src_row = bottom_up ? (aheight - 1 - y) : y;
    memcpy(out->data() + (size_t)y * (size_t)width, raw.data() + (size_t)src_row * row_bytes,
           (size_t)width);
  }
  *w = width;
  *h = aheight;
  return true;
}

bool ListFrames(const std::string& dir, std::vector<std::string>* paths) {
  DIR* d = opendir(dir.c_str());
  if (!d) return false;
  struct dirent* e = nullptr;
  while ((e = readdir(d)) != nullptr) {
    const std::string name = e->d_name;
    if (name.size() < 5) continue;
    if (name.compare(name.size() - 4, 4, ".bmp") != 0) continue;
    paths->push_back(dir + "/" + name);
  }
  closedir(d);
  std::sort(paths->begin(), paths->end());
  return !paths->empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = "/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324";
  std::string model = "/tmp/p0_off_A_none.rknn";
  std::string kdir = "/mnt/manu/mosaic_sdk/runtime/kernels";
  std::string seq = "VIDEO00007_19700101_001324";
  std::string prefix = "/tmp/track";
  std::string ids_path;  // 可选：每帧使用的 client_frame_id（物理帧号）
  int frames = 400;
  int npu_workers = 1;
  int tag_mask = 0x7;
  int reorder_window = 0;
  int fusion = 1;      // 0 = 不打开时空融合（A/B 对照）
  double thr = 0.02;   // 与 cands400.tsv 相同的检测阈值
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool nx = (i + 1 < argc);
    if (a == "--dir" && nx) dir = argv[++i];
    else if (a == "--model" && nx) model = argv[++i];
    else if (a == "--kernel-dir" && nx) kdir = argv[++i];
    else if (a == "--seq" && nx) seq = argv[++i];
    else if (a == "--out-prefix" && nx) prefix = argv[++i];
    else if (a == "--frames" && nx) frames = atoi(argv[++i]);
    else if (a == "--npu-workers" && nx) npu_workers = atoi(argv[++i]);
    else if (a == "--tag-mask" && nx) tag_mask = atoi(argv[++i]);
    else if (a == "--reorder-window" && nx) reorder_window = atoi(argv[++i]);
    else if (a == "--fusion" && nx) fusion = atoi(argv[++i]);
    else if (a == "--thr" && nx) thr = atof(argv[++i]);
    else if (a == "--ids" && nx) ids_path = argv[++i];
    else {
      fprintf(stderr,
              "用法: %s [--dir D] [--model M] [--kernel-dir K] [--seq S] [--out-prefix P]\n"
              "          [--frames N] [--npu-workers 1] [--tag-mask 7] [--reorder-window 0] [--thr T]\n"
              "          [--ids FILE]   每帧的 client_frame_id（物理帧号），模拟跳帧；\n"
              "                      不给时用稠密的 0,1,2,...（与 cands400.tsv 口径一致）\n",
              argv[0]);
      return 2;
    }
  }

  std::vector<std::string> paths;
  if (!ListFrames(dir, &paths)) { fprintf(stderr, "[FATAL] %s 下没有 bmp\n", dir.c_str()); return 1; }
  if (frames > 0 && (int)paths.size() > frames) paths.resize((size_t)frames);

  // ---- 可选的物理帧号表：ids[i] 是第 i 个被推帧的 client_frame_id ----
  // 用来在板卡上复现「跳帧（Δn>1）」的真实节奏：cands.tsv 的帧号列写物理帧号，
  // 于是板卡的融合引擎走的就是物理帧轴（Δn 语义），和 Python 参考 --physical-frame-index 对齐。
  std::vector<int64_t> ids;
  if (!ids_path.empty()) {
    FILE* fi = fopen(ids_path.c_str(), "r");
    if (!fi) { fprintf(stderr, "[FATAL] 打不开 ids 文件 %s\n", ids_path.c_str()); return 1; }
    long long v = 0;
    while (fscanf(fi, "%lld", &v) == 1) ids.push_back((int64_t)v);
    fclose(fi);
    if (ids.size() < paths.size()) {
      fprintf(stderr, "[FATAL] ids 文件只有 %zu 个编号，但要用 %zu 帧\n", ids.size(), paths.size());
      return 1;
    }
    fprintf(stderr, "[info] 物理帧号表：%zu 个编号，首/末 = %lld / %lld\n", ids.size(),
            (long long)ids.front(), (long long)ids[paths.size() - 1]);
  }

  MosaicPipelineConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  mosaic_pipeline_config_preset(&cfg, MOSAIC_PRESET_REALTIME);
  mosaic_pipeline_config_set_paths(&cfg, model.c_str(), kdir.c_str(), seq.c_str());
  cfg.npu_workers = npu_workers;
  cfg.post_workers = 1;
  char err[512] = {0};
  MosaicPipeline* pipe = mosaic_pipeline_create(&cfg, err, sizeof(err));
  if (!pipe) { fprintf(stderr, "[FATAL] 建流水线失败：%s\n", err); return 1; }
  mosaic_pipeline_set_post_mode(pipe, MOSAIC_POST_SPARSE);
  if (thr > 0.0) {
    const double t = thr;
    mosaic_pipeline_set_detect_threshold(pipe, &t);
  }

  // ---- 打开时空融合 ----
  if (fusion) {
    MosaicTrackFusionConfig tf;
    mosaic_pipeline_track_fusion_defaults(&tf);
    tf.tag_mask = tag_mask;
    tf.reorder_window = reorder_window;
    if (mosaic_pipeline_set_track_fusion(pipe, &tf) != 0) {
      fprintf(stderr, "[FATAL] 打开时空融合失败\n");
      return 1;
    }
  } else {
    const MosaicTrackFusionConfig* none = nullptr;
    mosaic_pipeline_set_track_fusion(pipe, none);
  }

  // ---- 推 + 收（1:1：推一帧等一帧，这样相机队列不会丢帧，候选流是稠密的，
  //      和 cands400.tsv 的口径一致）----
  std::vector<uint8_t> gray;
  int w = 0, h = 0;
  int64_t pushed = 0;
  FILE* fc = fopen((prefix + "_cands.tsv").c_str(), "w");
  FILE* ft = fopen((prefix + "_tracked.tsv").c_str(), "w");
  if (!fc || !ft) { fprintf(stderr, "[FATAL] 打不开输出\n"); return 1; }
  std::vector<MosaicDet> dets(512);
  std::vector<MosaicTrackedDet> tdets(512);
  int64_t fid = -1;
  int64_t update_idx = 0;
  size_t n_cands = 0;
  for (size_t i = 0; i < paths.size(); ++i) {
    if (!LoadBmpGray8(paths[i], &w, &h, &gray)) { fprintf(stderr, "[warn] 读不了 %s\n", paths[i].c_str()); continue; }
    const int64_t cid = ids.empty() ? (int64_t)i : ids[i];
    const int64_t id = mosaic_pipeline_push_frame(pipe, gray.data(), (int32_t)w, (int32_t)h, cid);
    if (id < 0) { fprintf(stderr, "[FATAL] push 失败 last_error=%s\n", mosaic_pipeline_last_error(pipe)); break; }
    ++pushed;

    // 检测结果（= 喂给引擎的候选流）
    const int32_t n = mosaic_pipeline_collect(pipe, &fid, dets.data(), (int32_t)dets.size());
    if (n < 0) { fprintf(stderr, "[FATAL] collect 失败\n"); break; }
    fprintf(fc, "%lld %d", (long long)(ids.empty() ? update_idx : cid), n);
    for (int32_t k = 0; k < n; ++k) {
      fprintf(fc, " %.9g %.9g %.9g", (double)dets[k].cx, (double)dets[k].cy, (double)dets[k].peak);
    }
    fprintf(fc, "\n");
    n_cands += (size_t)n;
    ++update_idx;

    // 在线跟踪结果
    int64_t tfid = -1;
    const int32_t tn = mosaic_pipeline_collect_tracked(pipe, &tfid, tdets.data(), (int32_t)tdets.size());
    if (tn >= 0) {
      fprintf(ft, "%lld %d", (long long)tfid, tn);
      for (int32_t k = 0; k < tn; ++k) {
        fprintf(ft, " %d %.9g %.9g %.9g %d", tdets[k].track_id, (double)tdets[k].x, (double)tdets[k].y,
                tdets[k].score, tdets[k].infilled);
      }
      fprintf(ft, "\n");
    }
  }
  fclose(fc);
  fclose(ft);

  // ---- 离线融合（flush 之后才有）----
  mosaic_pipeline_flush(pipe);
  FILE* ff = fopen((prefix + "_fused.tsv").c_str(), "w");
  if (!ff) { fprintf(stderr, "[FATAL] 打不开 fused 输出\n"); return 1; }
  size_t n_fused = 0;
  int64_t ffid = -1;
  for (;;) {
    const int32_t n = mosaic_pipeline_collect_fused(pipe, &ffid, tdets.data(), (int32_t)tdets.size());
    if (n < 0) break;
    fprintf(ff, "%lld %d", (long long)ffid, n);
    for (int32_t k = 0; k < n; ++k) {
      fprintf(ff, " %d %.9g %.9g %.9g %d", tdets[k].track_id, (double)tdets[k].x, (double)tdets[k].y,
              tdets[k].score, tdets[k].infilled);
    }
    fprintf(ff, "\n");
    n_fused += (size_t)n;
  }
  fclose(ff);

  MosaicPipelineTiming tm;
  memset(&tm, 0, sizeof(tm));
  mosaic_pipeline_get_timing(pipe, &tm);
  MosaicPostTiming pt;
  memset(&pt, 0, sizeof(pt));
  mosaic_pipeline_get_post_timing(pipe, &pt);
  printf("pushed=%lld updates=%lld cands=%zu fused_dets=%zu\n", (long long)pushed,
         (long long)update_idx, n_cands, n_fused);
  printf("post_ms_mean=%.3f stddev=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f frames=%d\n",
         pt.mean_ms, pt.stddev_ms, pt.p50_ms, pt.p95_ms, pt.p99_ms, pt.max_ms, pt.frames);
  printf("pipeline_frames=%lld wall_ms=%.1f fps=%.3f dropped=%lld features_dropped=%lld\n",
         (long long)tm.frames, tm.wall_ms,
         (tm.wall_ms > 0.0 && tm.frames > 0) ? 1000.0 * (double)tm.frames / tm.wall_ms : 0.0,
         (long long)tm.dropped, (long long)tm.features_dropped);
  printf("out_prefix=%s\n", prefix.c_str());
  fflush(stdout);

  mosaic_pipeline_destroy(pipe);
  return 0;
}
