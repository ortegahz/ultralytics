// ===========================================================================
// track_fusion_align_test.cpp -- 时空融合的**精度对齐工具**（不进交付库）
//
//   作用：读一份「候选流」TSV，跑 mosaic::TrackFusionEngine，输出「融合结果」
//   TSV；再与 Python 参考实现（manu/evaluation/sdk_track_fusion_ref.py）逐帧
//   逐字段比对 ⇒ 证明 C++ 移植与 osd_sys_sota_trial0080_th045_20261022 的
//   时空逻辑**一致**。
//
//   输入 TSV（每行一帧，纯空白分隔）：
//       <frame_idx> <n> [<x1> <y1> <score1> <x2> <y2> <score2> ...]
//   输出 TSV（每行一帧，按分数降序）：
//       <frame_idx> <n> [<track_id> <x> <y> <score> <infilled> ...]
//
//   本工具**零第三方依赖** ⇒ host 与板卡都能原生编译。
// ===========================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "mosaic_track_fusion.h"

namespace {

void Die(const char* msg) {
  fprintf(stderr, "[FATAL] %s\n", msg);
  exit(1);
}

std::string Trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
  return s.substr(b, e - b);
}

// 参数覆盖文件：每行 `key value` 或 `key=value`，# 开头是注释。
void LoadParams(const char* path, mosaic::TrackFusionParams* p) {
  FILE* f = fopen(path, "r");
  if (!f) Die("cannot open params file");
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    std::string s = Trim(line);
    if (s.empty() || s[0] == '#') continue;
    size_t sep = s.find_first_of("= \t");
    std::string key, val;
    if (sep == std::string::npos) {
      continue;
    }
    key = s.substr(0, sep);
    val = Trim(s.substr(sep + 1));
    if (!val.empty() && val[0] == '=') val = Trim(val.substr(1));
    if (key.empty() || val.empty()) continue;
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
    else if (key == "max_age_phys") p->max_age_phys = (int32_t)v;
    else if (key == "r_slope") p->r_slope = v;
    else if (key == "alpha_decay") p->alpha_decay = v;
    else {
      fprintf(stderr, "[warn] unknown param key '%s' (ignored)\n", key.c_str());
    }
  }
  fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  const char* cands = nullptr;
  const char* out = nullptr;
  const char* params_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool nx = (i + 1 < argc);
    if (a == "--cands" && nx) cands = argv[++i];
    else if (a == "--out" && nx) out = argv[++i];
    else if (a == "--params" && nx) params_path = argv[++i];
    else {
      fprintf(stderr,
              "用法: %s --cands <cands.tsv> --out <fused.tsv> [--params <params.txt>]\n", argv[0]);
      return 2;
    }
  }
  if (!cands || !out) {
    fprintf(stderr, "用法: %s --cands <cands.tsv> --out <fused.tsv> [--params <params.txt>]\n", argv[0]);
    return 2;
  }

  mosaic::TrackFusionParams params;
  if (params_path) LoadParams(params_path, &params);

  // ---- 读候选流 ----
  FILE* f = fopen(cands, "r");
  if (!f) Die("cannot open candidates file");
  struct Frame {
    int64_t idx = 0;
    std::vector<mosaic::TrackFusionCand> cands;
  };
  std::vector<Frame> frames;
  {
    char* line = nullptr;
    size_t cap = 0;
    ssize_t len = 0;
    while ((len = getline(&line, &cap, f)) > 0) {
      std::string s = Trim(line);
      if (s.empty() || s[0] == '#') continue;
      char* p = &line[0];
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
  }
  fclose(f);
  fprintf(stderr, "[i] 读到 %zu 帧候选\n", frames.size());

  // ---- 跑时空融合 ----
  mosaic::TrackFusionEngine engine(params);
  for (const auto& fr : frames) {
    engine.Step(fr.idx, fr.cands.data(), fr.cands.size());
  }
  std::vector<std::vector<mosaic::TrackFusionDet>> fused;
  engine.Finalize(&fused);

  // ---- 写结果 ----
  FILE* o = fopen(out, "w");
  if (!o) Die("cannot open output file");
  size_t total = 0;
  for (size_t i = 0; i < fused.size(); ++i) {
    fprintf(o, "%lld %zu", (long long)i, fused[i].size());
    for (const mosaic::TrackFusionDet& d : fused[i]) {
      char bx[64], by[64], bs[64];
      snprintf(bx, sizeof(bx), "%.9g", (double)d.x);
      snprintf(by, sizeof(by), "%.9g", (double)d.y);
      snprintf(bs, sizeof(bs), "%.9g", d.score);
      fprintf(o, " %d %s %s %s %d", d.track_id, bx, by, bs, d.infilled ? 1 : 0);
      ++total;
    }
    fprintf(o, "\n");
  }
  fclose(o);
  fprintf(stderr, "[i] 融合输出 %zu 帧 / %zu 条检测 -> %s\n", fused.size(), total, out);
  return 0;
}
