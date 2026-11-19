// ===========================================================================
// track_fusion_bench.cpp -- 时空融合引擎的**逐帧耗时**测量工具（不进交付库）
//
//   目的：为 RT pipeline 的参数搜索提供「加完后处理的时间成本」基线：
//     * Pass 1  Step()        —— 在线（RT 路径，每帧一次）
//     * 在线输出 OnlineFused() —— 在线（RT 路径，每帧一次）
//     * Pass 2+3 Finalize()   —— 批处理（离线参考口径，RT 另议）
//
//   零第三方依赖 ⇒ host 与板卡都能原生编译。
//
//   用法：
//     track_fusion_bench --cands <cands.tsv> [--reps R] [--params <params.txt>]
// ===========================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mosaic_track_fusion.h"

namespace {

using clk = std::chrono::steady_clock;

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

void LoadParams(const char* path, mosaic::TrackFusionParams* p) {
  FILE* f = fopen(path, "r");
  if (!f) { fprintf(stderr, "[FATAL] 打不开参数文件 %s\n", path); exit(1); }
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t'))
      s.pop_back();
    if (s.empty() || s[0] == '#') continue;
    const size_t sep = s.find_first_of("= \t");
    if (sep == std::string::npos) continue;
    std::string key = s.substr(0, sep);
    std::string val = s.substr(sep + 1);
    if (!val.empty() && val[0] == '=') val = val.substr(1);
    const double v = atof(val.c_str());
    if (key == "th_base") p->th_base = v;
    else if (key == "th_salvage") p->th_salvage = v;
    else if (key == "th_ground") p->th_ground = v;
    else if (key == "sky_ratio") p->sky_ratio = v;
    else if (key == "img_h") p->img_h = (int32_t)v;
    else if (key == "max_age") p->max_age = (int32_t)v;
    else if (key == "min_hits") p->min_hits = (int32_t)v;
    else if (key == "match_dist") p->match_dist = v;
    else if (key == "max_match_dist") p->max_match_dist = v;
    else if (key == "min_track_score") p->min_track_score = v;
    else if (key == "instant_conf") p->instant_conf = v;
    else if (key == "min_displacement") p->min_displacement = v;
    else if (key == "th_deep") p->th_deep = v;
    else if (key == "min_hits_deep") p->min_hits_deep = (int32_t)v;
    else if (key == "stitch_gap") p->stitch_gap = (int32_t)v;
    else if (key == "stitch_max_dist") p->stitch_max_dist = v;
    else if (key == "infill_gap") p->infill_gap = (int32_t)v;
    else if (key == "min_hits_infill") p->min_hits_infill = (int32_t)v;
    else if (key == "min_rigid_disp") p->min_rigid_disp = v;
    else if (key == "max_rigid_var") p->max_rigid_var = v;
    else if (key == "min_hits_prune") p->min_hits_prune = (int32_t)v;
    else if (key == "min_rigid_disp_sky") p->min_rigid_disp_sky = v;
    else if (key == "confirm_consec") p->confirm_consec = (int32_t)v;
    else if (key == "confirm_gap_tol") p->confirm_gap_tol = (int32_t)v;
    else if (key == "confirm_trim_prefix") p->confirm_trim_prefix = (int32_t)v;
  }
  fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  const char* cands = nullptr;
  const char* params_path = nullptr;
  int reps = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool nx = (i + 1 < argc);
    if (a == "--cands" && nx) cands = argv[++i];
    else if (a == "--params" && nx) params_path = argv[++i];
    else if (a == "--reps" && nx) reps = atoi(argv[++i]);
    else {
      fprintf(stderr, "用法: %s --cands <cands.tsv> [--reps R] [--params <params.txt>]\n", argv[0]);
      return 2;
    }
  }
  if (!cands) {
    fprintf(stderr, "用法: %s --cands <cands.tsv> [--reps R] [--params <params.txt>]\n", argv[0]);
    return 2;
  }

  // ---- 读候选流 ----
  struct Frame {
    int64_t idx = 0;
    std::vector<mosaic::TrackFusionCand> cands;
  };
  std::vector<Frame> frames;
  {
    FILE* f = fopen(cands, "r");
    if (!f) { fprintf(stderr, "[FATAL] 打不开 %s\n", cands); return 1; }
    char* line = nullptr;
    size_t cap = 0;
    ssize_t len = 0;
    while ((len = getline(&line, &cap, f)) > 0) {
      char* p = line;
      char* end = nullptr;
      const long long idx = strtoll(p, &end, 10);
      if (end == p) continue;
      p = end;
      const long long n = strtoll(p, &end, 10);
      if (end == p) continue;
      p = end;
      Frame fr;
      fr.idx = (int64_t)idx;
      for (long long i = 0; i < n; ++i) {
        mosaic::TrackFusionCand c;
        c.x = (float)strtod(p, &end); p = end;
        c.y = (float)strtod(p, &end); p = end;
        c.score = (float)strtod(p, &end); p = end;
        fr.cands.push_back(c);
      }
      frames.push_back(std::move(fr));
    }
    free(line);
    fclose(f);
  }
  size_t total_cands = 0;
  size_t max_cands = 0;
  for (const auto& fr : frames) {
    total_cands += fr.cands.size();
    max_cands = std::max(max_cands, fr.cands.size());
  }
  printf("frames=%zu total_cands=%zu mean_cands_per_frame=%.2f max_cands_per_frame=%zu reps=%d\n",
         frames.size(), total_cands,
         frames.empty() ? 0.0 : (double)total_cands / (double)frames.size(), max_cands, reps);

  mosaic::TrackFusionParams params;
  if (params_path) LoadParams(params_path, &params);

  std::vector<double> step_ms;    // Pass 1 每帧耗时
  std::vector<double> online_ms;  // OnlineFused 每帧耗时
  std::vector<double> finalize_ms;// Pass 2+3 一次性耗时
  std::vector<double> rep_total_ms;
  size_t last_det_count = 0;
  size_t last_frame_count = 0;

  for (int r = 0; r < reps; ++r) {
    mosaic::TrackFusionEngine engine(params);
    std::vector<mosaic::TrackFusionDet> online;
    const clk::time_point rep_t0 = clk::now();
    for (const auto& fr : frames) {
      const clk::time_point t0 = clk::now();
      engine.Step(fr.idx, fr.cands.data(), fr.cands.size());
      const clk::time_point t1 = clk::now();
      online.clear();
      engine.OnlineFused(fr.idx, &online);
      const clk::time_point t2 = clk::now();
      step_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      online_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
      if (r == reps - 1) last_det_count += online.size();
    }
    std::vector<std::vector<mosaic::TrackFusionDet>> fused;
    const clk::time_point t3 = clk::now();
    engine.Finalize(&fused);
    const clk::time_point t4 = clk::now();
    finalize_ms.push_back(std::chrono::duration<double, std::milli>(t4 - t3).count());
    rep_total_ms.push_back(std::chrono::duration<double, std::milli>(t4 - rep_t0).count());
    last_frame_count = fused.size();
  }

  printf("step_ms_mean=%.4f p50=%.4f p95=%.4f p99=%.4f max=%.4f\n", Mean(step_ms),
         Percentile(step_ms, 0.50), Percentile(step_ms, 0.95), Percentile(step_ms, 0.99),
         Percentile(step_ms, 1.0));
  printf("step_us_mean=%.2f step_us_p99=%.2f step_us_max=%.2f\n", Mean(step_ms) * 1000.0,
         Percentile(step_ms, 0.99) * 1000.0, Percentile(step_ms, 1.0) * 1000.0);
  printf("online_ms_mean=%.4f p95=%.4f max=%.4f\n", Mean(online_ms), Percentile(online_ms, 0.95),
         Percentile(online_ms, 1.0));
  printf("online_us_mean=%.2f online_us_max=%.2f\n", Mean(online_ms) * 1000.0,
         Percentile(online_ms, 1.0) * 1000.0);
  printf("finalize_ms_mean=%.4f p95=%.4f max=%.4f\n", Mean(finalize_ms), Percentile(finalize_ms, 0.95),
         Percentile(finalize_ms, 1.0));
  printf("rep_total_ms_mean=%.3f  (= 整段序列的在线+离线总耗时)\n", Mean(rep_total_ms));
  printf("fused_frames=%zu online_dets_total=%zu\n", last_frame_count, last_det_count);
  return 0;
}
