// ===========================================================================
// mosaic_cadence_probe.cpp -- 板卡侧「检测器 -> 跟踪器更新间隔」测速探针
//
//   目的（为 RT pipeline 的时空后处理参数搜索做输入准备）：
//     量化 SDK 管线在真实相机输入下的输出节拍：
//       1) 输出帧率 fps 与波动（两次结果之间的间隔 p50/p90/p95/p99/max）；
//       2) 跟踪器实际看到的帧号步长 dn（= 两次结果之间被丢掉的源帧数）
//          —— 这就是「检测器能给跟踪器更新的间隔」的物理含义；
//       3) 各级耗时（GMC / NPU / 后处理）与各级丢帧计数。
//
//   与嵌入式一致的做法：
//     * 生产者线程按源帧率推帧（mosaic_pipeline_push_frame 带自己的帧号），
//       消费者线程阻塞收结果（mosaic_pipeline_collect）—— 与嵌入式 app 的
//       「相机回调推帧 + 主循环收结果」同构；
//     * 相机队列满时丢最旧的帧（管线内建行为），所以 dn > 1 是常态，
//       这正是真实部署里跟踪器看到的节奏。
//
//   零 OpenCV 依赖（自带 8bit 灰度 BMP 读取器），只链 SDK 的 .so
//   => 与板卡上的 OpenCV 版本无关。
//
//   用法：
//     mosaic_cadence_probe --dir <bmp目录> --model <rknn> --kernel-dir <目录>
//         --seq <名字> [--frames N] [--src-fps F] [--warmup W]
//         [--npu-workers 1] [--thr T] [--out <results.tsv>]
// ===========================================================================
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>  // setenv（POSIX）
#include <string>
#include <thread>
#include <vector>

#include "mosaic_pipeline.h"

namespace {

using clk = std::chrono::steady_clock;

// 极简 8-bit 灰度 BMP 读取（BITMAPINFOHEADER，biCompression=0，行底向上）
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

double Percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q * (double)(v.size() - 1);
  const size_t lo = (size_t)std::floor(pos);
  const size_t hi = std::min(v.size() - 1, lo + 1);
  const double frac = pos - (double)lo;
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

double Mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (double x : v) s += x;
  return s / (double)v.size();
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = "/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324";
  std::string model = "/tmp/p0_off_A_none.rknn";
  std::string kdir = "/mnt/manu/mosaic_sdk/runtime/kernels";
  std::string seq = "VIDEO00007_19700101_001324";
  std::string out = "/tmp/cadence.tsv";
  int frames = 600;
  double src_fps = 25.0;  // <=0 = 不限速（相机无限快）
  double duration_s = 0.0;  // >0 = 按时间推帧（不限速时必须用时间，否则会被 EOF 提前掐断）
  int warmup = 40;
  int npu_workers = 1;
  double thr = 0.0;
  std::string preset = "realtime";  // realtime | baseline | throughput
  int fusion = 0;                   // 1 = 打开 SDK 内置时空融合
  int reorder_window = 0;           // >0 = 跟踪器按相机帧号升序喂帧（延迟 N 帧）
  int tag_mask = 0x7;               // 喂给跟踪器的候选 tag
  // 跨尺度候选合并半径（px）：交付值 20（16~24 为可用带）；0 = 关。
  double cand_dedup_radius = 20.0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool nx = (i + 1 < argc);
    if (a == "--dir" && nx) dir = argv[++i];
    else if (a == "--model" && nx) model = argv[++i];
    else if (a == "--kernel-dir" && nx) kdir = argv[++i];
    else if (a == "--seq" && nx) seq = argv[++i];
    else if (a == "--out" && nx) out = argv[++i];
    else if (a == "--frames" && nx) frames = atoi(argv[++i]);
    else if (a == "--src-fps" && nx) src_fps = atof(argv[++i]);
    else if (a == "--duration" && nx) duration_s = atof(argv[++i]);
    else if (a == "--warmup" && nx) warmup = atoi(argv[++i]);
    else if (a == "--npu-workers" && nx) npu_workers = atoi(argv[++i]);
    else if (a == "--thr" && nx) thr = atof(argv[++i]);
    else if (a == "--preset" && nx) preset = argv[++i];
    else if (a == "--fusion" && nx) fusion = atoi(argv[++i]);
    else if (a == "--reorder-window" && nx) reorder_window = atoi(argv[++i]);
    else if (a == "--tag-mask" && nx) tag_mask = atoi(argv[++i]);
    else if (a == "--cand-dedup-radius" && nx) cand_dedup_radius = atof(argv[++i]);
    else {
      fprintf(stderr,
              "用法: %s [--dir D] [--model M] [--kernel-dir K] [--seq S] [--out F]\n"
              "          [--frames N] [--src-fps F] [--warmup W] [--npu-workers 1] [--thr T]\n"
              "          [--fusion 0|1] [--reorder-window N] [--tag-mask M]\n",
              argv[0]);
      return 2;
    }
  }

  std::vector<std::string> paths;
  if (!ListFrames(dir, &paths)) { fprintf(stderr, "[FATAL] %s 下没有 bmp\n", dir.c_str()); return 1; }
  if (frames > 0 && (int)paths.size() > frames) paths.resize((size_t)frames);
  fprintf(stderr, "[i] 源帧 %zu 帧, src_fps=%.1f, npu_workers=%d\n", paths.size(), src_fps, npu_workers);

  // ---- 建管线 ----
  MosaicPipelineConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  int32_t preset_id = MOSAIC_PRESET_REALTIME;
  if (preset == "baseline") preset_id = MOSAIC_PRESET_BASELINE;
  else if (preset == "throughput") preset_id = MOSAIC_PRESET_THROUGHPUT;
  mosaic_pipeline_config_preset(&cfg, preset_id);
  mosaic_pipeline_config_set_paths(&cfg, model.c_str(), kdir.c_str(), seq.c_str());
  // npu_workers=1：板卡上 2 个 NPU context 会触发 RKNN 6s 超时 -> NPU soft reset，
  // 只能物理断电恢复，所以测速一律用 1（其余参数保持交付档原值）。
  cfg.npu_workers = npu_workers;
  cfg.post_workers = 1;
  // 🔴 跨尺度候选合并半径：库内置 20 px（16~24 为可用带）；**不能**照搬服务器的
  //    768 px（那会让每帧只剩 1 个候选、丢掉次要目标）。v10.6 起它是库常量，
  //    唯一的覆盖口是 MOSAIC_TRACK_MERGE_RADIUS（校准 / A-B 用）⇒ 必须在
  //    create() **之前**设好（库在 create() 时读一次）。
  char rad[32];
  snprintf(rad, sizeof(rad), "%.10g", cand_dedup_radius);
  setenv("MOSAIC_TRACK_MERGE_RADIUS", rad, 1);

  char err[512] = {0};
  MosaicPipeline* pipe = mosaic_pipeline_create(&cfg, err, sizeof(err));
  if (!pipe) { fprintf(stderr, "[FATAL] 建流水线失败：%s\n", err); return 1; }
  mosaic_pipeline_set_post_mode(pipe, MOSAIC_POST_SPARSE);
  if (thr > 0.0) {
    const double t = thr;
    mosaic_pipeline_set_detect_threshold(pipe, &t);
  }
  // ---- 时空融合（A/B 对照）----
  if (fusion) {
    MosaicTrackFusionConfig tf;
    mosaic_pipeline_track_fusion_defaults(&tf);
    // 🔴 交付工作点（trial_0080 + v3 · cc2_md10）：**显式**覆盖这 9 个字段，
    //    不依赖 .so 里编译进去的默认值 —— 旧默认组（match_dist 16 / gap_tol 6 /
    //    r_slope 3.5 / max_age_phys 10）在全口径上 FAR 0.0352，破 0.03 红线。
    //    参数档 manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json
    //    （params_md5 a1157e013ebb496731be805c72c95dd8）。
    tf.th_base        = 0.48;   // 天空区高置信阈值
    tf.th_salvage     = 0.06;   // 天空区弱脉冲打捞阈值
    tf.th_ground      = 0.26;   // 地面区阈值
    tf.sky_ratio      = 1.0;    // 天地分界 = img_h * sky_ratio
    tf.confirm_consec = 2;      // 连续命中几次算确认
    tf.confirm_gap_tol= 8;      // 连续判定容忍的额外物理帧间隔
    tf.r_slope        = 5.0;    // 每多跳 1 个物理帧，关联门限放宽的像素数
    tf.match_dist     = 10.0;   // 关联基础门限
    tf.max_age_phys   = -1;     // 交付组「不设」⇒ <0 = legacy（写 10 会破 FAR 红线）
    tf.tag_mask = tag_mask;
    tf.reorder_window = reorder_window;
    if (mosaic_pipeline_set_track_fusion(pipe, &tf) != 0) {
      fprintf(stderr, "[FATAL] 打开时空融合失败\n");
      return 1;
    }
    // 🔴 跨尺度候选合并（交付值 20 px）：服务器喂融合前会合并三个尺度的峰，
    //    板端缺这一步 ⇒ 一个目标裂成多条航迹 ⇒ FAR 0.3832（红线 12.8 倍）。
    //    v10.6：合并半径是**库内置常量 20 px**，公开 setter 已删除 ⇒ 这里只把工具侧
    //    的 A/B 开关通过环境变量交给库（setenv 已在本文件 create() 之前做过）。
    fprintf(stderr, "[i] 时空融合 ON  tag_mask=0x%x reorder_window=%d cand_dedup_radius=%.1f\n",
            tag_mask, reorder_window, cand_dedup_radius);
  } else {
    mosaic_pipeline_set_track_fusion(pipe, nullptr);
    fprintf(stderr, "[i] 时空融合 OFF\n");
  }

  // ---- 生产者：按源帧率推帧（带自己的帧号，模拟相机帧号） ----
  const clk::time_point t_base = clk::now();
  std::atomic<bool> producer_done(false);
  std::atomic<int64_t> pushed(0);
  std::thread producer([&]() {
    const clk::time_point t0 = clk::now();
    const clk::time_point t_end =
        duration_s > 0.0 ? t0 + std::chrono::milliseconds((long long)(duration_s * 1000.0))
                         : clk::time_point::max();
    std::vector<uint8_t> gray;
    int w = 0, h = 0;
    const size_t n_max = duration_s > 0.0 ? (size_t)-1 : paths.size();
    for (size_t i = 0; i < n_max; ++i) {
      if (duration_s > 0.0 && clk::now() >= t_end) break;
      const std::string& path = paths[i % paths.size()];
      if (!LoadBmpGray8(path, &w, &h, &gray)) {
        fprintf(stderr, "[warn] 读不了 %s\n", path.c_str());
        continue;
      }
      if (src_fps > 0.0) {
        const long long us = (long long)((double)i * 1e6 / src_fps);
        std::this_thread::sleep_until(t0 + std::chrono::microseconds(us));
      }
      const int64_t id =
          mosaic_pipeline_push_frame(pipe, gray.data(), (int32_t)w, (int32_t)h, (int64_t)i);
      if (id < 0) {
        fprintf(stderr, "[FATAL] push 失败 id=%lld last_error=%s\n", (long long)id,
                mosaic_pipeline_last_error(pipe));
        break;
      }
      pushed.fetch_add(1);
    }
    // 收尾：flush 让消费者能排空队列并退出（collect 在 flushed 且队列空时返回 -1）
    mosaic_pipeline_flush(pipe);
    producer_done.store(true);
  });

  // ---- 消费者 B（跟踪器视角）：阻塞收「在线跟踪输出」，记录跟踪器实际
  //      被更新的时刻与对应的相机帧号 —— 这就是「检测器能给跟踪器更新的
  //      间隔」的实测值。融合关闭时该线程立刻收到 -1 并退出。 ----
  struct TRec {
    double t_ms = 0.0;
    int64_t fid = 0;
    int32_t n = 0;
  };
  std::vector<TRec> trecs;
  std::thread tracker_consumer([&]() {
    std::vector<MosaicTrackedDet> tdets(512);
    int64_t tfid = -1;
    for (;;) {
      const int32_t n =
          mosaic_pipeline_collect_tracked(pipe, &tfid, tdets.data(), (int32_t)tdets.size());
      if (n < 0) break;
      TRec r;
      r.t_ms = std::chrono::duration<double, std::milli>(clk::now() - t_base).count();
      r.fid = tfid;
      r.n = n;
      trecs.push_back(r);
    }
  });

  // ---- 消费者 A：阻塞收结果，记录「到达时刻 + 帧号 + 检测数」 ----
  struct Rec {
    double t_ms = 0.0;
    int64_t fid = 0;
    int32_t n = 0;
  };
  std::vector<Rec> recs;
  std::vector<MosaicDet> dets(512);
  const clk::time_point t_rec0 = t_base;
  int64_t fid = -1;
  for (;;) {
    const int32_t n = mosaic_pipeline_collect(pipe, &fid, dets.data(), (int32_t)dets.size());
    if (n < 0) break;
    Rec r;
    r.t_ms = std::chrono::duration<double, std::milli>(clk::now() - t_rec0).count();
    r.fid = fid;
    r.n = n;
    recs.push_back(r);
  }
  producer.join();
  tracker_consumer.join();

  // ---- 统计 ----
  std::vector<double> gaps;  // 结果之间的间隔 (ms)
  std::vector<int64_t> dn;   // 结果之间的帧号步长
  std::vector<double> gaps_w;
  std::vector<int64_t> dn_w;
  for (size_t i = 1; i < recs.size(); ++i) {
    gaps.push_back(recs[i].t_ms - recs[i - 1].t_ms);
    dn.push_back(recs[i].fid - recs[i - 1].fid);
    if ((int)i > warmup) {
      gaps_w.push_back(gaps.back());
      dn_w.push_back(dn.back());
    }
  }

  FILE* fo = fopen(out.c_str(), "w");
  if (fo) {
    fprintf(fo, "# t_collect_ms\tfid\tn_dets\n");
    for (const Rec& r : recs) fprintf(fo, "%.3f\t%lld\t%d\n", r.t_ms, (long long)r.fid, r.n);
    fclose(fo);
  }

  // ---- 跟踪器更新间隔（tracker update interval）统计 ----
  std::vector<double> tgaps;  // 两次「跟踪器被更新」之间的间隔 (ms)
  std::vector<int64_t> tdn;   // 两次更新之间的相机帧号步长
  for (size_t i = 1; i < trecs.size(); ++i) {
    tgaps.push_back(trecs[i].t_ms - trecs[i - 1].t_ms);
    tdn.push_back(trecs[i].fid - trecs[i - 1].fid);
  }
  std::vector<double> tgaps_w;
  std::vector<int64_t> tdn_w;
  for (size_t i = 0; i < tgaps.size(); ++i) {
    if ((int)(i + 1) > warmup) {
      tgaps_w.push_back(tgaps[i]);
      tdn_w.push_back(tdn[i]);
    }
  }
  {
    std::string tout = out;
    const size_t dot = tout.rfind(".tsv");
    if (dot != std::string::npos) tout = tout.substr(0, dot);
    tout += "_tracked.tsv";
    FILE* ft = fopen(tout.c_str(), "w");
    if (ft) {
      fprintf(ft, "# t_tracker_update_ms\tcam_fid\tn_tracked\n");
      for (const TRec& r : trecs) fprintf(ft, "%.3f\t%lld\t%d\n", r.t_ms, (long long)r.fid, r.n);
      fclose(ft);
    }
  }

  MosaicPipelineTiming tm;
  memset(&tm, 0, sizeof(tm));
  mosaic_pipeline_get_timing(pipe, &tm);
  MosaicPostTiming pt;
  memset(&pt, 0, sizeof(pt));
  mosaic_pipeline_get_post_timing(pipe, &pt);

  const double wall = (double)tm.wall_ms;
  const double fps =
      (wall > 0.0 && (double)tm.frames > 0.0) ? 1000.0 * (double)tm.frames / wall : 0.0;
  const double fps_out = (!gaps_w.empty()) ? 1000.0 / Mean(gaps_w) : 0.0;

  printf("src_fps=%.1f npu_workers=%d preset=%s duration_s=%.1f\n", src_fps, npu_workers,
         preset.c_str(), duration_s);
  printf("pushed=%lld collected=%zu warmup_skipped=%d\n", (long long)pushed.load(), recs.size(), warmup);
  printf("pipeline_frames=%lld pipeline_wall_ms=%.1f pipeline_fps=%.3f\n", (long long)tm.frames, wall,
         fps);
  printf(
      "out_fps_from_gaps=%.3f gap_ms_mean=%.3f p50=%.3f p90=%.3f p95=%.3f p99=%.3f max=%.3f min=%.3f\n",
      fps_out, Mean(gaps_w), Percentile(gaps_w, 0.50), Percentile(gaps_w, 0.90), Percentile(gaps_w, 0.95),
      Percentile(gaps_w, 0.99), Percentile(gaps_w, 1.0),
      gaps_w.empty() ? 0.0 : *std::min_element(gaps_w.begin(), gaps_w.end()));
  {
    std::vector<int64_t> sorted = dn_w;
    std::sort(sorted.begin(), sorted.end());
    std::vector<double> dnd;
    for (int64_t v : dn_w) dnd.push_back((double)v);
    printf("dn_mean=%.3f dn_hist:", Mean(dnd));
    size_t i = 0;
    while (i < sorted.size()) {
      size_t j = i;
      while (j < sorted.size() && sorted[j] == sorted[i]) ++j;
      printf(" %lld:%zu", (long long)sorted[i], j - i);
      i = j;
    }
    printf("\n");
  }
  printf("timing_dropped_camera=%lld features_dropped=%lld detections_dropped=%lld\n",
         (long long)tm.dropped, (long long)tm.features_dropped, (long long)tm.detections_dropped);
  printf("gmc_total_ms=%.1f run_ms=%.1f backproj_ms=%.1f outputs_ms=%.1f upload_ms=%.1f\n",
         tm.gmc_total_ms, tm.run_ms, tm.backproj_ms, tm.outputs_ms, tm.upload_ms);
  printf("rknn_run_p50=%.2f p95=%.2f p99=%.2f max=%.2f samples=%lld\n", tm.rknn_run_p50_ms,
         tm.rknn_run_p95_ms, tm.rknn_run_p99_ms, tm.rknn_run_max_ms, (long long)tm.rknn_run_samples);
  printf("gmc_fit_p50=%.2f p95=%.2f p99=%.2f max=%.2f\n", tm.gmc_fit_p50_ms, tm.gmc_fit_p95_ms,
         tm.gmc_fit_p99_ms, tm.gmc_fit_max_ms);
  printf(
      "post_ms_mean=%.3f stddev=%.3f p50=%.3f p90=%.3f p95=%.3f p99=%.3f max=%.3f frames=%d\n", pt.mean_ms,
      pt.stddev_ms, pt.p50_ms, pt.p90_ms, pt.p95_ms, pt.p99_ms, pt.max_ms, pt.frames);
  printf("results_tsv=%s\n", out.c_str());
  // ---- 跟踪器被更新的节拍（这就是 RT 参数搜索要用的 Δn / dt 口径）----
  printf("tracker_updates=%zu\n", trecs.size());
  if (!tgaps_w.empty()) {
    const double tfps = 1000.0 / Mean(tgaps_w);
    printf(
        "tracker_update_fps=%.3f gap_ms_mean=%.3f p50=%.3f p90=%.3f p95=%.3f p99=%.3f max=%.3f min=%.3f\n",
        tfps, Mean(tgaps_w), Percentile(tgaps_w, 0.50), Percentile(tgaps_w, 0.90),
        Percentile(tgaps_w, 0.95), Percentile(tgaps_w, 0.99), Percentile(tgaps_w, 1.0),
        *std::min_element(tgaps_w.begin(), tgaps_w.end()));
    std::vector<int64_t> s = tdn_w;
    std::sort(s.begin(), s.end());
    std::vector<double> d;
    for (int64_t v : tdn_w) d.push_back((double)v);
    printf("tracker_dn_mean=%.3f tracker_dn_hist:", Mean(d));
    size_t i = 0;
    while (i < s.size()) {
      size_t j = i;
      while (j < s.size() && s[j] == s[i]) ++j;
      printf(" %lld:%zu", (long long)s[i], j - i);
      i = j;
    }
    printf("\n");
  } else {
    printf("tracker_update_fps=0 (no online tracked output)\n");
  }
  printf("fusion=%d reorder_window=%d tag_mask=0x%x\n", fusion, reorder_window, tag_mask);
  fflush(stdout);

  mosaic_pipeline_destroy(pipe);
  return 0;
}
