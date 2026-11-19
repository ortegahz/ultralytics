// ===========================================================================
// mosaic_det_dump.cpp -- 板卡侧「SDK 检测流导出」工具（**不进交付库**）
//
//   目的：把 SDK 在一段 IR 序列上的逐帧检测结果导出成「候选流 TSV」，
//   供时空融合的精度对齐使用（C++ 与 Python 参考实现喂**同一份**候选流）。
//
//   为什么自己读 BMP 而不是用 OpenCV：
//     本板卡只有 OpenCV **4.8** 运行库（没有 dev 头），而交付包的 example 是
//     按 4.6 编的 ⇒ 直接跑会 "libopencv_imgcodecs.so.406 not found"。
//     本工具**零 OpenCV 依赖**，只链 SDK 的 .so ⇒ 与 OpenCV 版本无关。
//
//   输出 TSV（每行一帧，按分数降序前的原始顺序）：
//       <frame_idx> <n> [<x1> <y1> <score1> <x2> <y2> <score2> ...]
//   x,y 是原生 640x512 帧像素坐标；score 是热图峰值。
// ===========================================================================
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mosaic_pipeline.h"

namespace {

// ---------------------------------------------------------------------------
// 极简 8-bit 灰度 BMP 读取（BITMAPINFOHEADER，biCompression=0，行底向上）
// ---------------------------------------------------------------------------
bool LoadBmpGray8(const std::string& path, int* w, int* h, std::vector<uint8_t>* out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  unsigned char hdr[54];
  if (fread(hdr, 1, 54, f) != 54) { fclose(f); return false; }
  if (hdr[0] != 'B' || hdr[1] != 'M') { fclose(f); return false; }
  const uint32_t data_off = (uint32_t)hdr[10] | ((uint32_t)hdr[11] << 8) | ((uint32_t)hdr[12] << 16) |
                            ((uint32_t)hdr[13] << 24);
  const int32_t width = (int32_t)((uint32_t)hdr[18] | ((uint32_t)hdr[19] << 8) | ((uint32_t)hdr[20] << 16) |
                                   ((uint32_t)hdr[21] << 24));
  const int32_t height = (int32_t)((uint32_t)hdr[22] | ((uint32_t)hdr[23] << 8) | ((uint32_t)hdr[24] << 16) |
                                    ((uint32_t)hdr[25] << 24));
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
    memcpy(out->data() + (size_t)y * (size_t)width, raw.data() + (size_t)src_row * row_bytes, (size_t)width);
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
  std::string out = "/tmp/cands.tsv";
  std::string seq = "VIDEO00007_19700101_001324";
  int frames = 0;
  double thr = 0.0;  // >0 时覆盖 region_threshold
  int tag_filter = -1;  // 只导出这个 tag 的检测（-1 = 全部 tag）
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool nx = (i + 1 < argc);
    if (a == "--frames-dir" && nx) dir = argv[++i];
    else if (a == "--model" && nx) model = argv[++i];
    else if (a == "--kernel-dir" && nx) kdir = argv[++i];
    else if (a == "--out" && nx) out = argv[++i];
    else if (a == "--seq" && nx) seq = argv[++i];
    else if (a == "--frames" && nx) frames = atoi(argv[++i]);
    else if (a == "--tag" && nx) tag_filter = atoi(argv[++i]);
    else if (a == "--thr" && nx) thr = atof(argv[++i]);
    else {
      fprintf(stderr, "用法: %s [--frames-dir D] [--model M] [--kernel-dir K] [--out F] [--seq S]\n"
                      "          [--frames N] [--tag 0|1|2]\n", argv[0]);
      return 2;
    }
  }

  std::vector<std::string> paths;
  if (!ListFrames(dir, &paths)) { fprintf(stderr, "[FATAL] %s 下没有 bmp\n", dir.c_str()); return 1; }
  if (frames > 0 && (int)paths.size() > frames) paths.resize((size_t)frames);

  MosaicOpenOptions o = {};
  o.version = MOSAIC_API_VERSION;
  o.preset = MOSAIC_PRESET_REALTIME;
  o.model_path = model.c_str();
  o.kernel_dir = kdir.c_str();
  o.sequence = seq.c_str();
  o.check_env = 0;
  char err[512] = {0};
  MosaicPipeline* pipe = mosaic_pipeline_open(&o, err, sizeof(err));
  if (!pipe) { fprintf(stderr, "[FATAL] 建流水线失败：%s\n", err); return 1; }
  mosaic_pipeline_set_post_mode(pipe, MOSAIC_POST_SPARSE);
  if (thr > 0.0) {
    const double t = thr;
    mosaic_pipeline_set_detect_threshold(pipe, &t);
  }

  FILE* fo = fopen(out.c_str(), "w");
  if (!fo) { fprintf(stderr, "[FATAL] 打不开输出 %s\n", out.c_str()); return 1; }

  std::vector<MosaicDet> dets(256);
  std::vector<uint8_t> gray;
  int w = 0, h = 0;
  int64_t pushed = 0;
  for (size_t i = 0; i < paths.size(); ++i) {
    if (!LoadBmpGray8(paths[i], &w, &h, &gray)) { fprintf(stderr, "[warn] 读不了 %s\n", paths[i].c_str()); continue; }
    const int64_t id = mosaic_pipeline_push(pipe, gray.data(), (int32_t)w, (int32_t)h);
    if (id < 0) { fprintf(stderr, "[FATAL] push 失败 id=%lld\n", (long long)id); break; }
    ++pushed;
    int64_t got_id = -1;
    const int32_t n = mosaic_pipeline_collect(pipe, &got_id, dets.data(), (int32_t)dets.size());
    if (n < 0) { fprintf(stderr, "[FATAL] collect 失败\n"); break; }
    int32_t kept = 0;
    for (int32_t k = 0; k < n; ++k) {
      if (tag_filter >= 0 && dets[k].tag != tag_filter) continue;
      ++kept;
    }
    fprintf(fo, "%lld %d", (long long)got_id, kept);
    for (int32_t k = 0; k < n; ++k) {
      if (tag_filter >= 0 && dets[k].tag != tag_filter) continue;
      char bx[64], by[64], bs[64];
      snprintf(bx, sizeof(bx), "%.9g", (double)dets[k].cx);
      snprintf(by, sizeof(by), "%.9g", (double)dets[k].cy);
      snprintf(bs, sizeof(bs), "%.9g", (double)dets[k].peak);
      fprintf(fo, " %s %s %s", bx, by, bs);
    }
    fprintf(fo, "\n");
    if ((i % 50) == 0) fprintf(stderr, "[i] %zu/%zu 帧\n", i + 1, paths.size());
  }
  fclose(fo);
  fprintf(stderr, "[i] 共推入 %lld 帧 -> %s\n", (long long)pushed, out.c_str());
  mosaic_pipeline_flush(pipe);
  mosaic_pipeline_destroy(pipe);
  return 0;
}
