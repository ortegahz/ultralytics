// rga_pat_ab_test.cpp -- 受控 A/B：RGA pattern 缓冲「未 import」vs「正确 import」
//
// 背景（2026-11-12 实测）：板卡 dmesg 零星出现
//     rga_mm: This is not a buffer that has been imported, handle = -1
//     rga: buffer[0] mm release buffer failed! handle = -1
// 并两次升级为内核 oops（rga_mm_get_buffer -> rga_mm_sync_dma_sg_for_device
// -> dma_sync_single_for_device -> __clean_dcache_area_poc 解引用
// ffffff8000000000）。
//
// 生产库 mosaic_core.h 的 ComposeFrame() 把 `s->pat`（BufAlloc 后 handle 仍是 -1，
// 从未 BufImport）以 wrapbuffer_handle(-1, ...) 传给 improcess()。
// 本探针用于回答两个问题：
//   Q1  把 pat 正确 import 之后，合成输出字节会不会变？（= 修复是否破坏既有 golden）
//   Q2  「not a buffer that has been imported」是不是由 handle=-1 的 pat 引起？
//
// 用法（板卡上直接编译运行）：
//   g++ -O2 -o rga_pat_ab_test rga_pat_ab_test.cpp -lrga
//   ./rga_pat_ab_test <iters> <mode>
//     mode 0 = pat 不 import（现状，handle=-1）        <- 复现生产路径
//     mode 1 = pat 正确 import                          <- 候选修复
//     mode 2 = pat 不 import，但 pat_rect 非空
//     mode 3 = pat 不 import，pat_rect 空，但只提交一次 improcess（去掉 per-iteration
//              的 canvas release+import 抖动）
// 输出：canvas 的 FNV-1a hash + improcess 返回码分布。
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <rga/im2d.h>

static const int kSrcW = 640, kSrcH = 512;
static const int kDstW = 640, kDstH = 640;
static const int kPatW = 4, kPatH = 4;
static const int kFmt = RK_FORMAT_RGB_888;

static uint32_t fnv1a(const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 16777619u; }
  return h;
}

static void* alloc_aligned(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, 4096, bytes) != 0) return nullptr;
  memset(p, 0, bytes);
  return p;
}

static im_rect rect(int x, int y, int w, int h) {
  im_rect r; r.x = x; r.y = y; r.width = w; r.height = h; return r;
}

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? atoi(argv[1]) : 200;
  const int mode  = (argc > 2) ? atoi(argv[2]) : 0;

  const size_t src_bytes = (size_t)kSrcW * kSrcH * 3;
  const size_t dst_bytes = (size_t)kDstW * kDstH * 3;
  const size_t pat_bytes = (size_t)kPatW * kPatH * 3;

  uint8_t* src = (uint8_t*)alloc_aligned(src_bytes);
  uint8_t* dst = (uint8_t*)alloc_aligned(dst_bytes);
  uint8_t* pat = (uint8_t*)alloc_aligned(pat_bytes);
  if (!src || !dst || !pat) { fprintf(stderr, "alloc failed\n"); return 2; }

  // 确定性伪随机内容（每次都一样的序列，便于跨 run 比对）
  uint32_t st = 12345u;
  for (size_t i = 0; i < src_bytes; i++) { st = st * 1664525u + 1013904223u; src[i] = (uint8_t)(st >> 24); }

  rga_buffer_handle_t dst_handle = importbuffer_virtualaddr(dst, kDstW, kDstH, kFmt);
  if (dst_handle <= 0) { fprintf(stderr, "dst import failed (%d)\n", (int)dst_handle); return 3; }
  const rga_buffer_t dstb = wrapbuffer_handle(dst_handle, kDstW, kDstH, kFmt);

  // mode 1 才 import pat；其余模式保持 handle = -1（生产现状）
  rga_buffer_handle_t pat_handle = -1;
  if (mode == 1) {
    pat_handle = importbuffer_virtualaddr(pat, kPatW, kPatH, kFmt);
    if (pat_handle <= 0) { fprintf(stderr, "pat import failed (%d)\n", (int)pat_handle); return 4; }
  }
  const rga_buffer_t patb = wrapbuffer_handle(pat_handle, kPatW, kPatH, kFmt);

  int n_ok = 0, n_fail = 0;
  int last_status = 0;
  // churn: 生产路径每帧都会 release 旧 canvas handle 再 import 新的
  //        （BufImport 就是这么写的），mode>=4 才打开这个抖动。
  const bool churn = (mode >= 4);
  const int n_tiles = (mode >= 5) ? 2 : 1;
  rga_buffer_handle_t dst_handle_live = dst_handle;
  rga_buffer_t dstb_live = dstb;

  for (int it = 0; it < iters; it++) {
    if (churn) {
      // 复刻 BufImport(): 先 release 旧 handle，再 import 一个新的
      if (dst_handle_live > 0) releasebuffer_handle(dst_handle_live);
      dst_handle_live = importbuffer_virtualaddr(dst, kDstW, kDstH, kFmt);
      if (dst_handle_live <= 0) { fprintf(stderr, "dst re-import failed\n"); return 6; }
      dstb_live = wrapbuffer_handle(dst_handle_live, kDstW, kDstH, kFmt);
    }
    const rga_buffer_handle_t src_handle = importbuffer_virtualaddr(src, kSrcW, kSrcH, kFmt);
    if (src_handle <= 0) { fprintf(stderr, "src import failed\n"); return 5; }
    const rga_buffer_t srcb = wrapbuffer_handle(src_handle, kSrcW, kSrcH, kFmt);

    const im_rect pat_rect = (mode == 2) ? rect(0, 0, kPatW, kPatH) : rect(0, 0, 0, 0);
    IM_STATUS st = IM_STATUS_SUCCESS;
    for (int t = 0; t < n_tiles; t++) {
      st = improcess(srcb, dstb_live, patb,
                     rect(0, 0, kSrcW, kSrcH),
                     rect(0, t * 160, kDstW, 160),
                     pat_rect, 0);
      last_status = (int)st;
      if (st == IM_STATUS_SUCCESS || st == IM_STATUS_NOERROR) n_ok++; else n_fail++;
    }

    releasebuffer_handle(src_handle);
  }

  printf("mode=%d iters=%d ok=%d fail=%d last_status=%d canvas_fnv1a=%08x\n",
         mode, iters, n_ok, n_fail, last_status, fnv1a(dst, dst_bytes));

  if (pat_handle > 0) releasebuffer_handle(pat_handle);
  if (churn) { if (dst_handle_live > 0) releasebuffer_handle(dst_handle_live); }
  else releasebuffer_handle(dst_handle);
  free(src); free(dst); free(pat);
  return 0;
}
