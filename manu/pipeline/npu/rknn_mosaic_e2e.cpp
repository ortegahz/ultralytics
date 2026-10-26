// ===========================================================================
// rknn_mosaic_e2e.cpp -- the end-to-end demo that DRIVES THE LIBRARY.
//
//   raw infrared sequence (640x512 gray JPGs)
//     -> mosaic_pipeline_push()   (the library does GMC/OpenCL + RGA mosaic
//                                   + RKNN fp16 + back-projection)
//     -> mosaic_pipeline_collect() (detections in native frame pixels)
//     -> <seq>_mosaic.tsv
//
// This program is a pure consumer of the public C ABI (mosaic_pipeline.h).
// It links against libmosaic_pipeline.a and calls nothing else from the
// project, so it exercises exactly the interface the embedded team gets.
//
// The TSV format is byte-compatible with the BMP-driven binary
// (rknn_mosaic_heatmap), so the SAME scorer
// (manu/pipeline/npu/ptq/35_mosaic_board_eval.py) reads both.
//
// Usage:
//   rknn_mosaic_e2e --raw-root DIR --sequence NAME --model M.rknn --out DIR
//                   [--kernel-dir /root/kernels] [--anchor-step 10]
//                   [--npu-workers 2] [--limit N] [--start N]
//                   [--cpu-affinity 4-7] [--threads 1] [--timing FILE]
// ===========================================================================
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "mosaic_pipeline.h"

namespace {

struct Options {
  std::string raw_root;
  std::string sequence;
  std::string model;
  std::string out;
  std::string kernel_dir = "/root/kernels";
  std::string timing;
  std::string cpu_affinity;
  std::string feature_dump;
  std::string mats_dump;
  std::string md5_golden;
  int anchor_step = 10;
  int npu_workers = 2;
  // -1 = "not specified": the library's own default (1) applies.
  //  0 = off (the reference implementation's default), 1 = on.
  int pyr_cache = -1;
  int prealloc_pad = 0;
  int limit = 0;
  int start = 0;
};

void PrintUsage(const char* argv0) {
  fprintf(stderr,
          "usage: %s --raw-root DIR --sequence NAME --model M.rknn --out DIR\n"
          "          [--kernel-dir /root/kernels] [--anchor-step 10] [--npu-workers 2]\n"
          "          [--limit N] [--start N] [--cpu-affinity 4-7] [--timing FILE]\n"
          "          [--pyr-cache 0|1]      reuse the GMC pyramid/corners across fits\n"
          "                                (bit-identical; default = the library's 1)\n"
          "          [--prealloc-pad N]     reserve the OpenCL device images once\n"
          "          [--feature-dump DIR]   write the GMC feature planes as BMPs\n"
          "                                (<seq>_%%06d.bmp, [Ch0,Ch1,Ch2])\n"
          "          [--mats-dump FILE]     raw float32 stream of the transform\n"
          "                                chain (21 lags x 6 floats per frame)\n"
          "          [--feature-md5-golden F]  compare the produced per-frame feature\n"
          "                                fingerprints against a frozen golden file\n"
          "                                and exit 1 on any mismatch (self-check)\n"
          "\n"
          "  raw-root   DIR   directory holding <sequence>/frame_%06d.jpg\n"
          "  sequence   NAME  sequence to run\n"
          "  out        DIR   output root (<seq>_mosaic.tsv is written here)\n"
          "  model      PATH  .rknn heatmap detector\n",
          argv0);
}

// The frame files of a sequence, sorted by their numeric frame index.
//
// BMP first: the board's cross-built OpenCV is WITH_JPEG=OFF / WITH_PNG=OFF,
// so a JPEG cannot be decoded there.  A JPG is still accepted for hosts whose
// OpenCV has the codec.
bool ListFrames(const std::string& dir, std::vector<std::string>* out) {
  DIR* d = opendir(dir.c_str());
  if (!d) {
    fprintf(stderr, "[FATAL] cannot open %s: %s\n", dir.c_str(), strerror(errno));
    return false;
  }
  std::map<long, std::string> by_index;
  struct dirent* e = nullptr;
  while ((e = readdir(d)) != nullptr) {
    const std::string name = e->d_name;
    // frame_<digits>.bmp / frame_<digits>.jpg
    if (name.rfind("frame_", 0) != 0) continue;
    const size_t dot = name.rfind('.');
    if (dot == std::string::npos || dot + 4 != name.size()) continue;
    const std::string ext = name.substr(dot);
    if (ext != ".bmp" && ext != ".jpg" && ext != ".png") continue;
    const std::string digits = name.substr(6, dot - 6);
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) continue;
    // Prefer BMP when both exist for the same index.
    const long idx = atol(digits.c_str());
    auto it = by_index.find(idx);
    if (it == by_index.end() || it->second.size() > name.size() ||
        (it->second.size() == name.size() && ext == ".bmp")) {
      by_index[idx] = dir + "/" + name;
    }
  }
  closedir(d);
  for (const auto& kv : by_index) out->push_back(kv.second);
  return !out->empty();
}

// Create a directory (and its parents) if it does not exist.  Plain POSIX:
// the test program must not reach into the library's internals.
void EnsureDir(const std::string& dir) {
  if (dir.empty()) return;
  std::string cur;
  size_t i = 0;
  if (dir[0] == '/') { cur = "/"; i = 1; }
  while (i <= dir.size()) {
    const size_t slash = dir.find('/', i);
    const std::string part =
        dir.substr(i, (slash == std::string::npos ? dir.size() : slash) - i);
    if (!part.empty()) {
      if (!cur.empty() && cur.back() != '/') cur += "/";
      cur += part;
      mkdir(cur.c_str(), 0755);
    }
    if (slash == std::string::npos) break;
    i = slash + 1;
  }
}

std::string StemOf(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  const std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
  const size_t dot = base.find_last_of('.');
  return dot == std::string::npos ? base : base.substr(0, dot);
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") { PrintUsage(argv[0]); return 0; }
    else if (a == "--raw-root" && i + 1 < argc) opt.raw_root = argv[++i];
    else if (a == "--sequence" && i + 1 < argc) opt.sequence = argv[++i];
    else if (a == "--model" && i + 1 < argc) opt.model = argv[++i];
    else if (a == "--out" && i + 1 < argc) opt.out = argv[++i];
    else if (a == "--kernel-dir" && i + 1 < argc) opt.kernel_dir = argv[++i];
    else if (a == "--timing" && i + 1 < argc) opt.timing = argv[++i];
    else if (a == "--feature-dump" && i + 1 < argc) opt.feature_dump = argv[++i];
    else if (a == "--mats-dump" && i + 1 < argc) opt.mats_dump = argv[++i];
    else if (a == "--feature-md5-golden" && i + 1 < argc) opt.md5_golden = argv[++i];
    else if (a == "--cpu-affinity" && i + 1 < argc) opt.cpu_affinity = argv[++i];
    else if (a == "--anchor-step" && i + 1 < argc) opt.anchor_step = atoi(argv[++i]);
    else if (a == "--npu-workers" && i + 1 < argc) opt.npu_workers = atoi(argv[++i]);
    else if (a == "--pyr-cache" && i + 1 < argc) opt.pyr_cache = atoi(argv[++i]);
    else if (a == "--prealloc-pad" && i + 1 < argc) opt.prealloc_pad = atoi(argv[++i]);
    else if (a == "--limit" && i + 1 < argc) opt.limit = atoi(argv[++i]);
    else if (a == "--start" && i + 1 < argc) opt.start = atoi(argv[++i]);
    else {
      fprintf(stderr, "[FATAL] unknown option: %s\n", argv[i]);
      PrintUsage(argv[0]);
      return 2;
    }
  }
  if (opt.raw_root.empty() || opt.sequence.empty() || opt.model.empty() || opt.out.empty()) {
    fprintf(stderr, "[FATAL] --raw-root, --sequence, --model and --out are all required\n");
    PrintUsage(argv[0]);
    return 2;
  }

  // The board's librga prints its debug dump to STDOUT.  Detach stdout so the
  // contract "logs on stderr, results in files" cannot be broken.
  if (FILE* devnull = fopen("/dev/null", "w")) {
    dup2(fileno(devnull), STDOUT_FILENO);
    fclose(devnull);
    fprintf(stderr, "[i] stdout detached (the board's librga prints RGA debug dumps there)\n");
  }

  const std::string seq_dir = opt.raw_root + "/" + opt.sequence;
  std::vector<std::string> frames;
  if (!ListFrames(seq_dir, &frames)) {
    fprintf(stderr, "[FATAL] no frames under %s\n", seq_dir.c_str());
    return 1;
  }
  if (opt.start > 0) {
    if ((size_t)opt.start >= frames.size()) {
      fprintf(stderr, "[FATAL] %zu frames, --start %d skips all\n", frames.size(), opt.start);
      return 1;
    }
    frames.erase(frames.begin(), frames.begin() + opt.start);
  }
  if (opt.limit > 0 && (int)frames.size() > opt.limit) frames.resize((size_t)opt.limit);
  fprintf(stderr, "[i] sequence %s: %zu frames\n", opt.sequence.c_str(), frames.size());

  // ---- the library --------------------------------------------------------
  // Zero-initialised as well: mosaic_pipeline_config_defaults() sets every
  // field, but a caller that forgets it must not end up with stack garbage
  // in the GMC knobs.
  MosaicPipelineConfig cfg{};
  mosaic_pipeline_config_defaults(&cfg);
  cfg.model_path = opt.model.c_str();
  cfg.kernel_dir = opt.kernel_dir.c_str();
  cfg.sequence = opt.sequence.c_str();
  cfg.gmc_anchor_step = opt.anchor_step;
  cfg.npu_workers = opt.npu_workers;
  if (opt.pyr_cache >= 0) cfg.gmc_pyr_cache = opt.pyr_cache;  // else: library default
  cfg.gmc_prealloc_pad = opt.prealloc_pad;
  cfg.cpu_affinity = opt.cpu_affinity.empty() ? nullptr : opt.cpu_affinity.c_str();

  // The output root must exist before the TSV is written (the library never
  // touches the filesystem outside the two optional dumps).
  EnsureDir(opt.out);

  const std::string md5_out = opt.out + "/feature_md5.txt";
  char err[512];
  err[0] = '\0';
  MosaicPipeline* pipe = mosaic_pipeline_create(&cfg, err, sizeof(err));
  if (!pipe) {
    fprintf(stderr, "[FATAL] mosaic_pipeline_create failed: %s\n", err);
    return 1;
  }

  mosaic_pipeline_set_feature_md5_out(pipe, md5_out.c_str());
  if (!opt.feature_dump.empty()) {
    EnsureDir(opt.feature_dump);
    mosaic_pipeline_set_feature_dump_out(pipe, opt.feature_dump.c_str());
    fprintf(stderr, "[i] feature dump -> %s\n", opt.feature_dump.c_str());
  }
  if (!opt.mats_dump.empty()) {
    mosaic_pipeline_set_mats_dump_out(pipe, opt.mats_dump.c_str());
    fprintf(stderr, "[i] mats dump -> %s\n", opt.mats_dump.c_str());
  }

  // ---- push every frame --------------------------------------------------
  // The frames are pushed IN ORDER: the GMC engine is causal.
  const double t_push0 = 0.0;  // timing comes from the library
  (void)t_push0;
  for (size_t i = 0; i < frames.size(); i++) {
    const cv::Mat img = cv::imread(frames[i], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      fprintf(stderr, "[FATAL] cannot read %s\n", frames[i].c_str());
      mosaic_pipeline_destroy(pipe);
      return 1;
    }
    if (mosaic_pipeline_push(pipe, img.data, img.cols, img.rows) < 0) {
      fprintf(stderr, "[FATAL] push failed at frame %zu: %s\n", i, mosaic_pipeline_last_error(pipe));
      mosaic_pipeline_destroy(pipe);
      return 1;
    }
  }
  mosaic_pipeline_flush(pipe);

  // ---- collect every result ----------------------------------------------
  // The workers finish out of order, so the rows are keyed by frame id and
  // written back in frame order.
  std::map<int64_t, std::vector<MosaicDet>> rows;
  std::vector<MosaicDet> buf(256);
  for (;;) {
    int64_t fid = -1;
    const int32_t n = mosaic_pipeline_collect(pipe, &fid, buf.data(), (int32_t)buf.size());
    if (n < 0) break;
    std::vector<MosaicDet> dets(buf.begin(), buf.begin() + n);
    rows[fid] = std::move(dets);
  }

  // ---- write the TSV ------------------------------------------------------
  char tsv_path[1024];
  snprintf(tsv_path, sizeof(tsv_path), "%s/%s_mosaic.tsv", opt.out.c_str(), opt.sequence.c_str());
  FILE* tsv = fopen(tsv_path, "w");
  if (!tsv) {
    fprintf(stderr, "[FATAL] cannot open %s: %s\n", tsv_path, strerror(errno));
    mosaic_pipeline_destroy(pipe);
    return 1;
  }
  // The stem must carry the frame number: the scorer's stem_to_frame_index()
  // takes the digits after the last "__".
  for (const auto& kv : rows) {
    char stem[256];
    snprintf(stem, sizeof(stem), "%s__%06lld", opt.sequence.c_str(), (long long)kv.first);
    fprintf(tsv, "%s\t%zu", stem, kv.second.size());
    for (const MosaicDet& d : kv.second) {
      fprintf(tsv, "\t%d %.4f %.4f %d %d %d %.6f %.4f", d.tag, (double)d.cx, (double)d.cy, d.w, d.h,
              d.area, (double)d.peak, (double)d.sum);
    }
    fprintf(tsv, "\n");
  }
  fclose(tsv);
  fprintf(stderr, "[i] TSV -> %s (%zu frames)\n", tsv_path, rows.size());
  fprintf(stderr, "[i] feature md5 (pre-encode planes) = %s\n", mosaic_pipeline_feature_md5(pipe));

  // ---- golden fingerprint self-check -------------------------------------
  // The GMC feature stream is the one thing that must be bit-identical to the
  // reference implementation.  Comparing it takes a second and needs no GT,
  // no scoring run and no NPU at all -- so the embedded team can verify a
  // deployment (kernel dir, OpenCL ICD, OpenCV build, input frames) before
  // wiring the library into their application.
  int golden_rc = 0;
  if (!opt.md5_golden.empty()) {
    FILE* gf = fopen(opt.md5_golden.c_str(), "r");
    FILE* pf = fopen(md5_out.c_str(), "r");
    if (!gf || !pf) {
      fprintf(stderr, "[FATAL] cannot open the golden or the produced md5 file\n");
      golden_rc = 2;
    } else {
      char gl[256], pl[256];
      long same = 0, diff = 0, n = 0;
      while (fgets(gl, sizeof(gl), gf) && fgets(pl, sizeof(pl), pf)) {
        char gseq[128], pseq[128], gidx[64], pidx[64], ghash[64], phash[64];
        if (sscanf(gl, "%127[^\t]\t%63[^\t]\t%63[^\n]", gseq, gidx, ghash) != 3) continue;
        if (sscanf(pl, "%127[^\t]\t%63[^\t]\t%63[^\n]", pseq, pidx, phash) != 3) continue;
        ++n;
        if (strcmp(ghash, phash) == 0) ++same; else ++diff;
      }
      if (n == 0) {
        fprintf(stderr, "[FATAL] the golden comparison compared 0 frames\n");
        golden_rc = 2;
      } else {
        fprintf(stderr, "[SELF-CHECK] feature md5 vs golden: %ld SAME / %ld DIFF "
                        "(of %ld frames)\n", same, diff, n);
        if (diff != 0) {
          fprintf(stderr, "[SELF-CHECK] **FAIL** -- the GMC feature stream is NOT the "
                          "reference one (check kernel_dir, OpenCL ICD, OpenCV build, "
                          "input frames, and that gmc_pyr_cache matches)\n");
          golden_rc = 1;
        } else {
          fprintf(stderr, "[SELF-CHECK] **PASS** -- the GMC feature stream is bit-identical "
                          "to the reference implementation\n");
        }
      }
    }
    if (gf) fclose(gf);
    if (pf) fclose(pf);
  }

  // ---- timing report ------------------------------------------------------
  MosaicPipelineTiming tm;
  mosaic_pipeline_get_timing(pipe, &tm);
  const double nf = tm.frames > 0 ? (double)tm.frames : 1.0;
  const char* timing_path = opt.timing.empty() ? nullptr : opt.timing.c_str();
  char tbuf[1024];
  if (!timing_path) {
    snprintf(tbuf, sizeof(tbuf), "%s/timing.txt", opt.out.c_str());
    timing_path = tbuf;
  }
  if (FILE* tf = fopen(timing_path, "w")) {
    fprintf(tf, "# rknn_mosaic_e2e timing (library-driven)\n");
    fprintf(tf, "seq=%s\n", opt.sequence.c_str());
    fprintf(tf, "anchor_step=%d\n", opt.anchor_step);
    fprintf(tf, "npu_workers=%d\n", opt.npu_workers);
    fprintf(tf, "frames=%lld\n", (long long)tm.frames);
    fprintf(tf, "%-18s %14s %14s\n", "segment", "total_ms", "avg_ms/frame");
    const struct { const char* name; double v; } segs[] = {
        {"gmc_fit", tm.gmc_fit_ms},        {"gmc_warp", tm.gmc_warp_ms},
        {"gmc_median", tm.gmc_median_ms}, {"gmc_total", tm.gmc_total_ms},
        {"rga_import", tm.rga_import_ms},  {"rga_improcess", tm.rga_improcess_ms},
        {"upload", tm.upload_ms},          {"rknn_run", tm.run_ms},
        {"outputs_get", tm.outputs_ms},    {"backproject_cc", tm.backproj_ms},
    };
    for (const auto& s : segs)
      fprintf(tf, "%-18s %14.3f %14.4f\n", s.name, s.v, s.v / nf);
    fprintf(tf, "%-18s %14.3f %14.4f\n", "total_wall", tm.wall_ms, tm.wall_ms / nf);
    fprintf(tf, "fps=%.3f\n", tm.wall_ms > 0 ? 1000.0 * (double)tm.frames / tm.wall_ms : 0.0);
    fclose(tf);
  }
  fprintf(stderr,
          "[timing] %lld frames, wall %.1f ms (%.3f ms/frame), %.3f fps\n"
          "[timing] gmc_fit %.3f  gmc_total %.3f  rga %.3f  upload %.3f  run %.3f  "
          "outputs %.3f  backproj %.3f  (ms/frame)\n",
          (long long)tm.frames, tm.wall_ms, tm.wall_ms / (tm.frames > 0 ? tm.frames : 1),
          tm.wall_ms > 0 ? 1000.0 * (double)tm.frames / tm.wall_ms : 0.0,
          tm.gmc_fit_ms / nf, tm.gmc_total_ms / nf,
          (tm.rga_import_ms + tm.rga_improcess_ms) / nf, tm.upload_ms / nf, tm.run_ms / nf,
          tm.outputs_ms / nf, tm.backproj_ms / nf);

  mosaic_pipeline_destroy(pipe);
  // A golden-fingerprint mismatch is a hard failure: the process must NOT
  // report success to whatever CI/脚本 is driving it.
  return golden_rc;
}
