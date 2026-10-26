// ===========================================================================
// rknn_mosaic_heatmap.cpp -- single-pass RK3588 NPU inference over a list of
// PRE-BUILT feature BMPs, on the multi-scale mosaic (Canvas Mosaic) canvas.
//
//   feature BMPs -> RGA mosaic (640x640) -> RKNN f16 -> back-projection ->
//                  connected components -> <seq>_mosaic.tsv
//
// This file is the BMP-driven arm.  The inference core now lives in
// mosaic_core.h so that the end-to-end demo (rknn_mosaic_e2e.cpp: raw
// infrared -> GMC/OpenCL feature -> RGA mosaic -> NPU -> detections) runs the
// SAME code instead of a re-implementation of it.
//
// The CLI, the output files and the TSV bytes are unchanged from the version
// that produced the archived 28,997-frame run; the refactor is verified by
// re-running a frame range and diffing the TSV byte for byte.
//
// Contract (all of it read out of the checkpoint or the ONNX graph):
//   input   uint8 NHWC 640x640x3, letterboxed/padded with 114
//   output0 heatmap  (1, 1, 320, 320) sigmoid
//   output1 offset   (1, 2, 320, 320) sub-pixel, unused by the CC detector
//
// Usage:
//   rknn_mosaic_heatmap --model <m.rknn> --list <manifest.txt> --out <dir>
//                        [--start N] [--limit N] [--tiles 0,1,2]
//                        [--canvas-mode mosaic|single]
//                        [--input-mode set|io-mem-nchw|io-mem-nhwc]
//                        [--cpu-affinity 4-7] [--timing <file>]
// ===========================================================================
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include "mosaic_core.h"

int main(int argc, char** argv) {
  mosaic::Options opt;
  bool have_tiles = false;

  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") { mosaic::PrintUsage(argv[0]); return 0; }
    else if (a == "--model" && i + 1 < argc) opt.model = argv[++i];
    else if (a == "--list" && i + 1 < argc) opt.list = argv[++i];
    else if (a == "--out" && i + 1 < argc) opt.out = argv[++i];
    else if (a == "--timing" && i + 1 < argc) opt.timing = argv[++i];
    else if (a == "--start" && i + 1 < argc) opt.start = atoi(argv[++i]);
    else if (a == "--limit" && i + 1 < argc) opt.limit = atoi(argv[++i]);
    else if (a == "--tiles" && i + 1 < argc) {
      std::string err;
      if (!mosaic::ParseTiles(argv[++i], &opt.tiles, &err)) {
        fprintf(stderr, "[FATAL] %s\n", err.c_str());
        return 2;
      }
      have_tiles = true;
    }
    else if (a == "--cpu-affinity" && i + 1 < argc) opt.cpu_affinity = argv[++i];
    else if (a == "--input-mode" && i + 1 < argc) {
      const std::string m = argv[++i];
      if (m == "set") opt.input_mode = mosaic::InputMode::kSet;
      else if (m == "io-mem-nchw") opt.input_mode = mosaic::InputMode::kIoMemNchw;
      else if (m == "io-mem-nhwc") opt.input_mode = mosaic::InputMode::kIoMemNhwc;
      else { fprintf(stderr, "[FATAL] unknown --input-mode: %s\n", m.c_str()); return 2; }
    }
    else if (a == "--canvas-mode" && i + 1 < argc) {
      const std::string m = argv[++i];
      if (m == "mosaic") opt.canvas_mode = mosaic::CanvasMode::kMosaic;
      else if (m == "single") opt.canvas_mode = mosaic::CanvasMode::kSingle;
      else { fprintf(stderr, "[FATAL] unknown --canvas-mode: %s\n", m.c_str()); return 2; }
    }
    else {
      fprintf(stderr, "[FATAL] unknown option: %s\n", argv[i]);
      mosaic::PrintUsage(argv[0]);
      return 2;
    }
  }

  if (opt.model.empty() || opt.list.empty() || opt.out.empty()) {
    fprintf(stderr, "[FATAL] --model, --list and --out are all required\n");
    mosaic::PrintUsage(argv[0]);
    return 2;
  }
  if (opt.start < 0) opt.start = 0;
  if (opt.limit < 0) opt.limit = 0;

  // The single-scale arm only has the native tile.
  if (opt.canvas_mode == mosaic::CanvasMode::kSingle && have_tiles) {
    for (int t : opt.tiles) {
      if (t != 0) {
        fprintf(stderr,
                "[FATAL] --canvas-mode single only has the native tile (tag 0); "
                "--tiles %d is not available in that mode.\n",
                t);
        return 2;
      }
    }
  }

  // The board's librga prints its debug dump to STDOUT.  Detach stdout so the
  // contract "logs on stderr, results in files" cannot be broken by a
  // library.
  if (FILE* devnull = fopen("/dev/null", "w")) {
    dup2(fileno(devnull), STDOUT_FILENO);
    fclose(devnull);
    fprintf(stderr, "[i] stdout detached (the board's librga prints RGA debug dumps there)\n");
  }

  if (!opt.cpu_affinity.empty()) mosaic::ApplyCpuAffinity(opt.cpu_affinity);

  // ---- frame list ---------------------------------------------------------
  std::vector<std::string> frames;
  {
    FILE* f = fopen(opt.list.c_str(), "r");
    if (!f) {
      fprintf(stderr, "[FATAL] cannot read list %s: %s\n", opt.list.c_str(), strerror(errno));
      return 1;
    }
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
      std::string s(line);
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
      if (!s.empty()) frames.push_back(s);
    }
    fclose(f);
  }
  if ((int)frames.size() <= opt.start) {
    fprintf(stderr, "[FATAL] list has %zu frames, --start %d skips all of them\n", frames.size(),
            opt.start);
    return 1;
  }
  frames.erase(frames.begin(), frames.begin() + opt.start);
  if (opt.limit > 0 && (int)frames.size() > opt.limit) frames.resize((size_t)opt.limit);
  if (frames.empty()) { fprintf(stderr, "[FATAL] no frames to process\n"); return 1; }

  const std::string seqname = mosaic::DeriveSeqName(opt.list, frames);
  mosaic::EnsureDir(opt.out);
  const std::string tsv_path = opt.out + "/" + seqname + "_mosaic.tsv";
  const std::string timing_path = opt.timing.empty() ? (opt.out + "/timing.txt") : opt.timing;

  // ---- session ------------------------------------------------------------
  mosaic::Session sess;
  if (!mosaic::InitSession(&sess, opt.model, opt.canvas_mode, opt.tiles, have_tiles,
                           opt.input_mode)) {
    mosaic::ReleaseSession(&sess);
    return 1;
  }

  // ---- output files -------------------------------------------------------
  FILE* tsv = fopen(tsv_path.c_str(), "w");
  if (!tsv) {
    fprintf(stderr, "[FATAL] cannot open %s: %s\n", tsv_path.c_str(), strerror(errno));
    mosaic::ReleaseSession(&sess);
    return 1;
  }
  fprintf(stderr, "[i] TSV    -> %s\n", tsv_path.c_str());
  fprintf(stderr, "[i] timing -> %s\n", timing_path.c_str());
  fprintf(stderr, "[i] canvas-mode=%s input-mode=%s frames=%zu seq=%s\n",
          opt.canvas_mode == mosaic::CanvasMode::kMosaic ? "mosaic" : "single",
          opt.input_mode == mosaic::InputMode::kSet ? "set"
          : (opt.input_mode == mosaic::InputMode::kIoMemNchw ? "io-mem-nchw" : "io-mem-nhwc"),
          frames.size(), seqname.c_str());

  mosaic::Timing tm;
  mosaic::FrameOut fo;
  const double wall0 = mosaic::now_ms();

  for (size_t fi = 0; fi < frames.size(); fi++) {
    const std::string& path = frames[fi];
    const std::string stem = mosaic::StemOf(path);

    // ---- decode ------------------------------------------------------------
    const double t0 = mosaic::now_ms();
    int w = 0, h = 0, comp = 0;
    uint8_t* pixels = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (!pixels) {
      fprintf(stderr, "[FATAL] stbi_load failed: %s\n", path.c_str());
      fclose(tsv);
      mosaic::ReleaseSession(&sess);
      return 1;
    }
    // PIXEL CONTRACT: the BMP stores BGR, stb returns RGB, and that RGB byte
    // order IS the PC side's model_order_bmp() == cv2.imread(bmp)[..., ::-1],
    // i.e. the model's channel order [residual, diff, I_t].  No flip here.
    if (w != mosaic::kSrcW || h != mosaic::kSrcH || comp != mosaic::kSrcC) {
      fprintf(stderr, "[FATAL] %s: expected %dx%dx%d, got %dx%dx%d\n", path.c_str(), mosaic::kSrcW,
              mosaic::kSrcH, mosaic::kSrcC, w, h, comp);
      free(pixels);
      fclose(tsv);
      mosaic::ReleaseSession(&sess);
      return 1;
    }
    const double t_decode = mosaic::now_ms() - t0;

    // ---- infer -------------------------------------------------------------
    if (!mosaic::ProcessFrame(&sess, pixels, w, h, &fo)) {
      free(pixels);
      fclose(tsv);
      mosaic::ReleaseSession(&sess);
      return 1;
    }
    free(pixels);

    // ---- write -------------------------------------------------------------
    const double w0 = mosaic::now_ms();
    const std::string line = mosaic::FormatTsvLine(stem, fo.comps);
    fwrite(line.data(), 1, line.size(), tsv);
    fflush(tsv);
    const double t_write = mosaic::now_ms() - w0;

    tm.decode += t_decode;
    tm.rga_import += fo.rga_import;
    tm.rga_improcess += fo.rga_improcess;
    tm.upload += fo.upload;
    tm.run += fo.run;
    tm.outputs += fo.outputs;
    tm.backproj += fo.backproj;
    tm.write += t_write;
    tm.frames++;
  }

  const double wall = mosaic::now_ms() - wall0;
  fclose(tsv);

  // ---- timing report ------------------------------------------------------
  const int nf = tm.frames > 0 ? tm.frames : 1;
  const double fps = (wall > 0.0) ? (1000.0 * (double)tm.frames / wall) : 0.0;
  {
    FILE* tf = fopen(timing_path.c_str(), "w");
    if (!tf) {
      fprintf(stderr, "[WARN] cannot open timing file %s: %s\n", timing_path.c_str(),
              strerror(errno));
    } else {
      fprintf(tf, "# rknn_mosaic_heatmap timing\n");
      fprintf(tf, "canvas_mode=%s\n",
              opt.canvas_mode == mosaic::CanvasMode::kMosaic ? "mosaic" : "single");
      fprintf(tf, "input_mode=%s\n",
              opt.input_mode == mosaic::InputMode::kSet ? "set"
              : (opt.input_mode == mosaic::InputMode::kIoMemNchw ? "io-mem-nchw" : "io-mem-nhwc"));
      fprintf(tf, "seq=%s\n", seqname.c_str());
      fprintf(tf, "frames=%d\n", tm.frames);
      fprintf(tf, "%-18s %14s %14s %8s\n", "segment", "total_ms", "avg_ms/frame", "frames");
      const struct { const char* name; double v; } segs[] = {
          {"decode", tm.decode},        {"rga_import", tm.rga_import},
          {"rga_improcess", tm.rga_improcess}, {"fp16_or_upload", tm.upload},
          {"rknn_run", tm.run},        {"outputs_get", tm.outputs},
          {"backproject_cc", tm.backproj}, {"write", tm.write},
      };
      for (const auto& s : segs)
        fprintf(tf, "%-18s %14.3f %14.4f %8d\n", s.name, s.v, s.v / (double)nf, tm.frames);
      const double sum_seg = tm.decode + tm.rga_import + tm.rga_improcess + tm.upload + tm.run +
                             tm.outputs + tm.backproj + tm.write;
      fprintf(tf, "%-18s %14.3f %14.4f %8d\n", "sum_of_segments", sum_seg, sum_seg / (double)nf,
              tm.frames);
      fprintf(tf, "%-18s %14.3f %14.4f %8d\n", "total_wall", wall, wall / (double)nf, tm.frames);
      fprintf(tf, "fps=%.3f\n", fps);
      fclose(tf);
    }
  }
  fprintf(stderr, "\n[timing] %d frames, total wall %.1f ms (%.3f ms/frame), %.3f fps\n", tm.frames,
          wall, wall / (double)nf, fps);
  fprintf(stderr, "[timing] decode %.3f  rga_import %.3f  rga_improcess %.3f  upload %.3f  "
                  "run %.3f  outputs %.3f  backproj %.3f  write %.3f  (ms/frame)\n",
          tm.decode / (double)nf, tm.rga_import / (double)nf, tm.rga_improcess / (double)nf,
          tm.upload / (double)nf, tm.run / (double)nf, tm.outputs / (double)nf,
          tm.backproj / (double)nf, tm.write / (double)nf);

  mosaic::ReleaseSession(&sess);
  return 0;
}
